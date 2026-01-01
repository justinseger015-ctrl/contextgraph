//! Configuration for RocksDB storage backend.
//!
//! Provides tunable options for RocksDB performance and behavior.

/// Default block cache size: 256MB (per constitution.yaml).
pub const DEFAULT_CACHE_SIZE: usize = 256 * 1024 * 1024;

/// Default maximum open files.
pub const DEFAULT_MAX_OPEN_FILES: i32 = 1000;

/// Configuration options for RocksDbMemex.
///
/// # Defaults
/// - `max_open_files`: 1000
/// - `block_cache_size`: 256MB (268,435,456 bytes)
/// - `enable_wal`: true (durability)
/// - `create_if_missing`: true (convenience)
#[derive(Debug, Clone)]
pub struct RocksDbConfig {
    /// Maximum open files (default: 1000).
    pub max_open_files: i32,
    /// Block cache size in bytes (default: 256MB).
    pub block_cache_size: usize,
    /// Enable Write-Ahead Logging (default: true).
    pub enable_wal: bool,
    /// Create database if missing (default: true).
    pub create_if_missing: bool,
}

impl Default for RocksDbConfig {
    fn default() -> Self {
        Self {
            max_open_files: DEFAULT_MAX_OPEN_FILES,
            block_cache_size: DEFAULT_CACHE_SIZE,
            enable_wal: true,
            create_if_missing: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_values() {
        let config = RocksDbConfig::default();
        assert_eq!(config.max_open_files, 1000);
        assert_eq!(config.block_cache_size, 256 * 1024 * 1024);
        assert!(config.enable_wal);
        assert!(config.create_if_missing);
    }

    #[test]
    fn test_config_custom_values() {
        let config = RocksDbConfig {
            max_open_files: 500,
            block_cache_size: 128 * 1024 * 1024,
            enable_wal: false,
            create_if_missing: false,
        };
        assert_eq!(config.max_open_files, 500);
        assert_eq!(config.block_cache_size, 128 * 1024 * 1024);
        assert!(!config.enable_wal);
        assert!(!config.create_if_missing);
    }

    #[test]
    fn test_config_clone() {
        let config = RocksDbConfig::default();
        let cloned = config.clone();
        assert_eq!(config.max_open_files, cloned.max_open_files);
    }

    #[test]
    fn test_config_debug() {
        let config = RocksDbConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("RocksDbConfig"));
        assert!(debug.contains("max_open_files"));
    }

    #[test]
    fn test_default_cache_size_constant() {
        assert_eq!(DEFAULT_CACHE_SIZE, 256 * 1024 * 1024);
        assert_eq!(DEFAULT_CACHE_SIZE, 268_435_456); // 256MB in bytes
    }

    #[test]
    fn test_default_max_open_files_constant() {
        assert_eq!(DEFAULT_MAX_OPEN_FILES, 1000);
    }
}
