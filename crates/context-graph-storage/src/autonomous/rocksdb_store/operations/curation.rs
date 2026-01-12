//! Memory curation state storage operations.

use context_graph_core::autonomous::MemoryCurationState;
use tracing::debug;
use uuid::Uuid;

use crate::autonomous::column_families::CF_MEMORY_CURATION;
use crate::autonomous::schema::{memory_curation_key, parse_memory_curation_key};

use super::super::error::{AutonomousStoreError, AutonomousStoreResult};
use super::super::RocksDbAutonomousStore;

impl RocksDbAutonomousStore {
    /// Store memory curation state.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory ID
    /// * `state` - The MemoryCurationState to store
    ///
    /// # Errors
    ///
    /// Returns error if serialization or RocksDB write fails.
    pub fn store_curation_state(
        &self,
        memory_id: Uuid,
        state: &MemoryCurationState,
    ) -> AutonomousStoreResult<()> {
        let cf = self.get_cf(CF_MEMORY_CURATION)?;
        let key = memory_curation_key(&memory_id);
        let data = Self::serialize_with_version(state)?;

        self.db.put_cf(cf, key, &data).map_err(|e| {
            AutonomousStoreError::rocksdb_op(
                "put",
                CF_MEMORY_CURATION,
                Some(&memory_id.to_string()),
                e,
            )
        })?;

        debug!(
            "Stored MemoryCurationState for {} ({} bytes)",
            memory_id,
            data.len()
        );
        Ok(())
    }

    /// Retrieve memory curation state.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory ID
    ///
    /// # Returns
    ///
    /// * `Ok(Some(state))` - State found
    /// * `Ok(None)` - No state for this memory
    /// * `Err(...)` - Read or deserialization error
    pub fn get_curation_state(
        &self,
        memory_id: Uuid,
    ) -> AutonomousStoreResult<Option<MemoryCurationState>> {
        let cf = self.get_cf(CF_MEMORY_CURATION)?;
        let key = memory_curation_key(&memory_id);

        match self.db.get_cf(cf, key) {
            Ok(Some(data)) => {
                let state = Self::deserialize_with_version(
                    &data,
                    CF_MEMORY_CURATION,
                    &memory_id.to_string(),
                )?;
                Ok(Some(state))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(AutonomousStoreError::rocksdb_op(
                "get",
                CF_MEMORY_CURATION,
                Some(&memory_id.to_string()),
                e,
            )),
        }
    }

    /// List all memory curation states.
    ///
    /// # Returns
    ///
    /// Vector of (memory_id, state) tuples.
    pub fn list_all_curation_states(
        &self,
    ) -> AutonomousStoreResult<Vec<(Uuid, MemoryCurationState)>> {
        let cf = self.get_cf(CF_MEMORY_CURATION)?;
        let mut results = Vec::new();

        let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);

        for item in iter {
            let (key, value) = item.map_err(|e| {
                AutonomousStoreError::rocksdb_op("iterate", CF_MEMORY_CURATION, None, e)
            })?;

            let memory_id = parse_memory_curation_key(&key);
            let state: MemoryCurationState =
                Self::deserialize_with_version(&value, CF_MEMORY_CURATION, &memory_id.to_string())?;
            results.push((memory_id, state));
        }

        debug!("Listed {} memory curation states", results.len());
        Ok(results)
    }
}
