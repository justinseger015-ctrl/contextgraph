//! Maintenance operations for RocksDbAutonomousStore.

use tracing::{debug, info};

use crate::autonomous::column_families::AUTONOMOUS_CFS;

use super::super::error::{AutonomousStoreError, AutonomousStoreResult};
use super::super::RocksDbAutonomousStore;

impl RocksDbAutonomousStore {
    /// Flush all column families to disk.
    pub fn flush(&self) -> AutonomousStoreResult<()> {
        debug!("Flushing all autonomous column families");

        for cf_name in AUTONOMOUS_CFS {
            let cf = self.get_cf(cf_name)?;
            self.db
                .flush_cf(cf)
                .map_err(|e| AutonomousStoreError::RocksDbOperation {
                    operation: "flush",
                    cf: cf_name,
                    key: None,
                    source: e,
                })?;
        }

        info!("Flushed all autonomous column families");
        Ok(())
    }

    /// Compact all column families.
    pub fn compact(&self) -> AutonomousStoreResult<()> {
        debug!("Compacting all autonomous column families");

        for cf_name in AUTONOMOUS_CFS {
            let cf = self.get_cf(cf_name)?;
            self.db.compact_range_cf(cf, None::<&[u8]>, None::<&[u8]>);
        }

        info!("Compacted all autonomous column families");
        Ok(())
    }
}
