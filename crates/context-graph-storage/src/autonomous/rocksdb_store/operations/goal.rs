//! Goal activity metrics storage operations.

use context_graph_core::autonomous::GoalActivityMetrics;
use tracing::debug;
use uuid::Uuid;

use crate::autonomous::column_families::CF_GOAL_ACTIVITY_METRICS;
use crate::autonomous::schema::{goal_activity_metrics_key, parse_goal_activity_metrics_key};

use super::super::error::{AutonomousStoreError, AutonomousStoreResult};
use super::super::RocksDbAutonomousStore;

impl RocksDbAutonomousStore {
    /// Store goal activity metrics.
    ///
    /// # Arguments
    ///
    /// * `goal_id` - The goal ID
    /// * `metrics` - The GoalActivityMetrics to store
    ///
    /// # Errors
    ///
    /// Returns error if serialization or RocksDB write fails.
    pub fn store_goal_metrics(
        &self,
        goal_id: Uuid,
        metrics: &GoalActivityMetrics,
    ) -> AutonomousStoreResult<()> {
        let cf = self.get_cf(CF_GOAL_ACTIVITY_METRICS)?;
        let key = goal_activity_metrics_key(&goal_id);
        let data = Self::serialize_with_version(metrics)?;

        self.db.put_cf(cf, key, &data).map_err(|e| {
            AutonomousStoreError::rocksdb_op(
                "put",
                CF_GOAL_ACTIVITY_METRICS,
                Some(&goal_id.to_string()),
                e,
            )
        })?;

        debug!(
            "Stored GoalActivityMetrics for {} ({} bytes)",
            goal_id,
            data.len()
        );
        Ok(())
    }

    /// Retrieve goal activity metrics.
    ///
    /// # Arguments
    ///
    /// * `goal_id` - The goal ID
    ///
    /// # Returns
    ///
    /// * `Ok(Some(metrics))` - Metrics found
    /// * `Ok(None)` - No metrics for this goal
    /// * `Err(...)` - Read or deserialization error
    pub fn get_goal_metrics(
        &self,
        goal_id: Uuid,
    ) -> AutonomousStoreResult<Option<GoalActivityMetrics>> {
        let cf = self.get_cf(CF_GOAL_ACTIVITY_METRICS)?;
        let key = goal_activity_metrics_key(&goal_id);

        match self.db.get_cf(cf, key) {
            Ok(Some(data)) => {
                let metrics = Self::deserialize_with_version(
                    &data,
                    CF_GOAL_ACTIVITY_METRICS,
                    &goal_id.to_string(),
                )?;
                Ok(Some(metrics))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(AutonomousStoreError::rocksdb_op(
                "get",
                CF_GOAL_ACTIVITY_METRICS,
                Some(&goal_id.to_string()),
                e,
            )),
        }
    }

    /// List all goal activity metrics.
    ///
    /// # Returns
    ///
    /// Vector of (goal_id, metrics) tuples.
    pub fn list_all_goal_metrics(&self) -> AutonomousStoreResult<Vec<(Uuid, GoalActivityMetrics)>> {
        let cf = self.get_cf(CF_GOAL_ACTIVITY_METRICS)?;
        let mut results = Vec::new();

        let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);

        for item in iter {
            let (key, value) = item.map_err(|e| {
                AutonomousStoreError::rocksdb_op("iterate", CF_GOAL_ACTIVITY_METRICS, None, e)
            })?;

            let goal_id = parse_goal_activity_metrics_key(&key);
            let metrics: GoalActivityMetrics = Self::deserialize_with_version(
                &value,
                CF_GOAL_ACTIVITY_METRICS,
                &goal_id.to_string(),
            )?;
            results.push((goal_id, metrics));
        }

        debug!("Listed {} goal activity metrics", results.len());
        Ok(results)
    }
}
