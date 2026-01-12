//! Consolidation record storage operations.

use tracing::debug;

use crate::autonomous::column_families::CF_CONSOLIDATION_HISTORY;
use crate::autonomous::schema::{
    consolidation_history_key, consolidation_history_timestamp_prefix,
    parse_consolidation_history_key,
};

use super::super::error::{AutonomousStoreError, AutonomousStoreResult};
use super::super::types::ConsolidationRecord;
use super::super::RocksDbAutonomousStore;

impl RocksDbAutonomousStore {
    /// Store a consolidation record.
    ///
    /// # Arguments
    ///
    /// * `record` - The ConsolidationRecord to store
    ///
    /// # Errors
    ///
    /// Returns error if serialization or RocksDB write fails.
    pub fn store_consolidation_record(
        &self,
        record: &ConsolidationRecord,
    ) -> AutonomousStoreResult<()> {
        let cf = self.get_cf(CF_CONSOLIDATION_HISTORY)?;
        let key = consolidation_history_key(record.timestamp_ms(), &record.id);
        let data = Self::serialize_with_version(record)?;

        self.db.put_cf(cf, key, &data).map_err(|e| {
            AutonomousStoreError::rocksdb_op(
                "put",
                CF_CONSOLIDATION_HISTORY,
                Some(&record.id.to_string()),
                e,
            )
        })?;

        debug!(
            "Stored ConsolidationRecord {} (success={}) - {} bytes",
            record.id,
            record.success,
            data.len()
        );
        Ok(())
    }

    /// Retrieve consolidation history since a given timestamp.
    ///
    /// # Arguments
    ///
    /// * `since` - Optional timestamp in milliseconds. If None, returns all history.
    ///
    /// # Returns
    ///
    /// Vector of ConsolidationRecords sorted by timestamp (oldest first).
    pub fn get_consolidation_history(
        &self,
        since: Option<u64>,
    ) -> AutonomousStoreResult<Vec<ConsolidationRecord>> {
        let cf = self.get_cf(CF_CONSOLIDATION_HISTORY)?;
        let mut results = Vec::new();

        let start_key = match since {
            Some(ts) => consolidation_history_timestamp_prefix(ts as i64).to_vec(),
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
                AutonomousStoreError::rocksdb_op("iterate", CF_CONSOLIDATION_HISTORY, None, e)
            })?;

            let (timestamp_ms, record_id) = parse_consolidation_history_key(&key);

            // If we have a since filter, skip entries before it
            if let Some(since_ts) = since {
                if (timestamp_ms as u64) < since_ts {
                    continue;
                }
            }

            let record: ConsolidationRecord = Self::deserialize_with_version(
                &value,
                CF_CONSOLIDATION_HISTORY,
                &record_id.to_string(),
            )?;
            results.push(record);
        }

        debug!("Retrieved {} consolidation records", results.len());
        Ok(results)
    }
}
