//! Lineage event storage operations.

use tracing::debug;

use crate::autonomous::column_families::CF_AUTONOMOUS_LINEAGE;
use crate::autonomous::schema::{
    autonomous_lineage_key, autonomous_lineage_timestamp_prefix, parse_autonomous_lineage_key,
};

use super::super::error::{AutonomousStoreError, AutonomousStoreResult};
use super::super::types::LineageEvent;
use super::super::RocksDbAutonomousStore;

impl RocksDbAutonomousStore {
    /// Store a lineage event.
    ///
    /// # Arguments
    ///
    /// * `event` - The LineageEvent to store
    ///
    /// # Errors
    ///
    /// Returns error if serialization or RocksDB write fails.
    pub fn store_lineage_event(&self, event: &LineageEvent) -> AutonomousStoreResult<()> {
        let cf = self.get_cf(CF_AUTONOMOUS_LINEAGE)?;
        let key = autonomous_lineage_key(event.timestamp_ms(), &event.id);
        let data = Self::serialize_with_version(event)?;

        self.db.put_cf(cf, key, &data).map_err(|e| {
            AutonomousStoreError::rocksdb_op(
                "put",
                CF_AUTONOMOUS_LINEAGE,
                Some(&event.id.to_string()),
                e,
            )
        })?;

        debug!(
            "Stored LineageEvent {} ({}) - {} bytes",
            event.id,
            event.event_type,
            data.len()
        );
        Ok(())
    }

    /// Retrieve lineage history since a given timestamp.
    ///
    /// # Arguments
    ///
    /// * `since` - Optional timestamp in milliseconds. If None, returns all history.
    ///
    /// # Returns
    ///
    /// Vector of LineageEvents sorted by timestamp (oldest first).
    pub fn get_lineage_history(
        &self,
        since: Option<u64>,
    ) -> AutonomousStoreResult<Vec<LineageEvent>> {
        let cf = self.get_cf(CF_AUTONOMOUS_LINEAGE)?;
        let mut results = Vec::new();

        let start_key = match since {
            Some(ts) => autonomous_lineage_timestamp_prefix(ts as i64).to_vec(),
            None => Vec::new(),
        };

        let iter = if start_key.is_empty() {
            self.db.iterator_cf(cf, rocksdb::IteratorMode::Start)
        } else {
            self.db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
            )
        };

        for item in iter {
            let (key, value) = item.map_err(|e| {
                AutonomousStoreError::rocksdb_op("iterate", CF_AUTONOMOUS_LINEAGE, None, e)
            })?;

            let (timestamp_ms, event_id) = parse_autonomous_lineage_key(&key);

            // If we have a since filter, skip entries before it
            if let Some(since_ts) = since {
                if (timestamp_ms as u64) < since_ts {
                    continue;
                }
            }

            let event: LineageEvent = Self::deserialize_with_version(
                &value,
                CF_AUTONOMOUS_LINEAGE,
                &event_id.to_string(),
            )?;
            results.push(event);
        }

        debug!("Retrieved {} lineage events", results.len());
        Ok(results)
    }
}
