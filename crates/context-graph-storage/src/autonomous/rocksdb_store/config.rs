//! Configuration for RocksDbAutonomousStore.

/// Configuration for RocksDbAutonomousStore.
#[derive(Debug, Clone)]
pub struct AutonomousStoreConfig {
    /// Block cache size in bytes (default: 64MB).
    pub block_cache_size: usize,
    /// Maximum number of open files (default: 500).
    pub max_open_files: i32,
    /// Enable WAL (write-ahead log) for durability (default: true).
    pub enable_wal: bool,
    /// Create database if it doesn't exist (default: true).
    pub create_if_missing: bool,
}

impl Default for AutonomousStoreConfig {
    fn default() -> Self {
        Self {
            block_cache_size: 64 * 1024 * 1024, // 64MB
            max_open_files: 500,
            enable_wal: true,
            create_if_missing: true,
        }
    }
}
