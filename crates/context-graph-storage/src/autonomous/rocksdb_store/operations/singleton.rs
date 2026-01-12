//! Singleton storage operations for AutonomousConfig and AdaptiveThresholdState.

use context_graph_core::autonomous::{AdaptiveThresholdState, AutonomousConfig};
use tracing::debug;

use crate::autonomous::column_families::{CF_ADAPTIVE_THRESHOLD_STATE, CF_AUTONOMOUS_CONFIG};
use crate::autonomous::schema::{ADAPTIVE_THRESHOLD_STATE_KEY, AUTONOMOUS_CONFIG_KEY};

use super::super::error::{AutonomousStoreError, AutonomousStoreResult};
use super::super::RocksDbAutonomousStore;

impl RocksDbAutonomousStore {
    // ========================================================================
    // AutonomousConfig (Singleton)
    // ========================================================================

    /// Store the autonomous configuration.
    ///
    /// Overwrites any existing configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The AutonomousConfig to store
    ///
    /// # Errors
    ///
    /// Returns error if serialization or RocksDB write fails.
    pub fn store_autonomous_config(&self, config: &AutonomousConfig) -> AutonomousStoreResult<()> {
        let cf = self.get_cf(CF_AUTONOMOUS_CONFIG)?;
        let data = Self::serialize_with_version(config)?;

        self.db
            .put_cf(cf, AUTONOMOUS_CONFIG_KEY, &data)
            .map_err(|e| {
                AutonomousStoreError::rocksdb_op("put", CF_AUTONOMOUS_CONFIG, Some("config"), e)
            })?;

        debug!("Stored AutonomousConfig ({} bytes)", data.len());
        Ok(())
    }

    /// Retrieve the autonomous configuration.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(config))` - Configuration found
    /// * `Ok(None)` - No configuration stored
    /// * `Err(...)` - Read or deserialization error
    pub fn get_autonomous_config(&self) -> AutonomousStoreResult<Option<AutonomousConfig>> {
        let cf = self.get_cf(CF_AUTONOMOUS_CONFIG)?;

        match self.db.get_cf(cf, AUTONOMOUS_CONFIG_KEY) {
            Ok(Some(data)) => {
                let config = Self::deserialize_with_version(&data, CF_AUTONOMOUS_CONFIG, "config")?;
                Ok(Some(config))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(AutonomousStoreError::rocksdb_op(
                "get",
                CF_AUTONOMOUS_CONFIG,
                Some("config"),
                e,
            )),
        }
    }

    // ========================================================================
    // AdaptiveThresholdState (Singleton)
    // ========================================================================

    /// Store the adaptive threshold state.
    ///
    /// Overwrites any existing state.
    ///
    /// # Arguments
    ///
    /// * `state` - The AdaptiveThresholdState to store
    ///
    /// # Errors
    ///
    /// Returns error if serialization or RocksDB write fails.
    pub fn store_threshold_state(
        &self,
        state: &AdaptiveThresholdState,
    ) -> AutonomousStoreResult<()> {
        let cf = self.get_cf(CF_ADAPTIVE_THRESHOLD_STATE)?;
        let data = Self::serialize_with_version(state)?;

        self.db
            .put_cf(cf, ADAPTIVE_THRESHOLD_STATE_KEY, &data)
            .map_err(|e| {
                AutonomousStoreError::rocksdb_op(
                    "put",
                    CF_ADAPTIVE_THRESHOLD_STATE,
                    Some("state"),
                    e,
                )
            })?;

        debug!("Stored AdaptiveThresholdState ({} bytes)", data.len());
        Ok(())
    }

    /// Retrieve the adaptive threshold state.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(state))` - State found
    /// * `Ok(None)` - No state stored
    /// * `Err(...)` - Read or deserialization error
    pub fn get_threshold_state(&self) -> AutonomousStoreResult<Option<AdaptiveThresholdState>> {
        let cf = self.get_cf(CF_ADAPTIVE_THRESHOLD_STATE)?;

        match self.db.get_cf(cf, ADAPTIVE_THRESHOLD_STATE_KEY) {
            Ok(Some(data)) => {
                let state =
                    Self::deserialize_with_version(&data, CF_ADAPTIVE_THRESHOLD_STATE, "state")?;
                Ok(Some(state))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(AutonomousStoreError::rocksdb_op(
                "get",
                CF_ADAPTIVE_THRESHOLD_STATE,
                Some("state"),
                e,
            )),
        }
    }
}
