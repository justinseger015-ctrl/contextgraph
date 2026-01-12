//! Drift history storage operations.

use context_graph_core::autonomous::DriftDataPoint;
use tracing::debug;
use uuid::Uuid;

use crate::autonomous::column_families::CF_DRIFT_HISTORY;
use crate::autonomous::schema::{drift_history_key, drift_history_timestamp_prefix, parse_drift_history_key};

use super::super::error::{AutonomousStoreError, AutonomousStoreResult};
use super::super::RocksDbAutonomousStore;

impl RocksDbAutonomousStore {
    /// Store a drift data point.
    ///
    /// # Arguments
    ///
    /// * `point` - The DriftDataPoint to store
    ///
    /// # Errors
    ///
    /// Returns error if serialization or RocksDB write fails.
    pub fn store_drift_point(&self, point: &DriftDataPoint) -> AutonomousStoreResult<()> {
        let cf = self.get_cf(CF_DRIFT_HISTORY)?;
        let id = Uuid::new_v4();
        let key = drift_history_key(point.timestamp.timestamp_millis(), &id);
        let data = Self::serialize_with_version(point)?;

        self.db.put_cf(cf, key, &data).map_err(|e| {
            AutonomousStoreError::rocksdb_op("put", CF_DRIFT_HISTORY, Some(&id.to_string()), e)
        })?;

        debug!("Stored DriftDataPoint ({} bytes)", data.len());
        Ok(())
    }

    /// Retrieve drift history since a given timestamp.
    ///
    /// # Arguments
    ///
    /// * `since` - Optional timestamp in milliseconds. If None, returns all history.
    ///
    /// # Returns
    ///
    /// Vector of DriftDataPoints sorted by timestamp (oldest first).
    pub fn get_drift_history(
        &self,
        since: Option<i64>,
    ) -> AutonomousStoreResult<Vec<DriftDataPoint>> {
        let cf = self.get_cf(CF_DRIFT_HISTORY)?;
        let mut results = Vec::new();

        let start_key = match since {
            Some(ts) => drift_history_timestamp_prefix(ts).to_vec(),
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
                AutonomousStoreError::rocksdb_op("iterate", CF_DRIFT_HISTORY, None, e)
            })?;

            let (timestamp_ms, _id) = parse_drift_history_key(&key);

            // If we have a since filter, skip entries before it
            if let Some(since_ts) = since {
                if timestamp_ms < since_ts {
                    continue;
                }
            }

            let point: DriftDataPoint = Self::deserialize_with_version(
                &value,
                CF_DRIFT_HISTORY,
                &format!("ts:{}", timestamp_ms),
            )?;
            results.push(point);
        }

        debug!("Retrieved {} drift data points", results.len());
        Ok(results)
    }
}
