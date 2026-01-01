//! Configuration for RocksDB storage backend.
//!
//! This module provides configuration options for tuning RocksDB performance
//! and behavior. All settings have sensible defaults per constitution.yaml.
//!
//! # Default Configuration
//!
//! | Setting | Default | Rationale |
//! |---------|---------|-----------|
//! | `block_cache_size` | 256MB | Constitution.yaml requirement |
//! | `max_open_files` | 1000 | Balance memory vs file handles |
//! | `enable_wal` | true | Durability before performance |
//! | `create_if_missing` | true | Developer convenience |
//!
//! # Example
//!
//! ```rust
//! use context_graph_storage::RocksDbConfig;
//!
//! // Use defaults
//! let default_config = RocksDbConfig::default();
//! assert_eq!(default_config.block_cache_size, 256 * 1024 * 1024);
//!
//! // Custom configuration
//! let custom_config = RocksDbConfig {
//!     block_cache_size: 512 * 1024 * 1024, // 512MB
//!     max_open_files: 2000,
//!     enable_wal: true,
//!     create_if_missing: true,
//! };
//! ```

/// Default block cache size: 256MB (per constitution.yaml).
///
/// The block cache stores uncompressed data blocks read from SST files.
/// Larger cache = fewer disk reads, but more memory usage.
///
/// # Constitution Reference
///
/// Per constitution.yaml performance requirements, the default cache
/// size is set to 256MB to balance memory usage with read performance.
///
/// # Example
///
/// ```rust
/// use context_graph_storage::DEFAULT_CACHE_SIZE;
///
/// assert_eq!(DEFAULT_CACHE_SIZE, 256 * 1024 * 1024);
/// assert_eq!(DEFAULT_CACHE_SIZE, 268_435_456); // 256MB in bytes
/// ```
pub const DEFAULT_CACHE_SIZE: usize = 256 * 1024 * 1024;

/// Default maximum open files: 1000.
///
/// Limits the number of file descriptors RocksDB can hold open.
/// Higher values reduce file open/close overhead but consume more FDs.
///
/// # Platform Considerations
///
/// - Linux default ulimit is often 1024, so 1000 leaves headroom
/// - Increase if you see "too many open files" errors
/// - Each column family can have multiple SST files
///
/// # Example
///
/// ```rust
/// use context_graph_storage::DEFAULT_MAX_OPEN_FILES;
///
/// assert_eq!(DEFAULT_MAX_OPEN_FILES, 1000);
/// ```
pub const DEFAULT_MAX_OPEN_FILES: i32 = 1000;

/// Configuration options for [`RocksDbMemex`](crate::RocksDbMemex).
///
/// Provides tunable options for RocksDB performance and behavior.
/// All fields have sensible defaults via [`Default`] implementation.
///
/// # Performance Tuning
///
/// - **Read-heavy workloads**: Increase `block_cache_size`
/// - **Write-heavy workloads**: Consider disabling WAL for speed (less durable)
/// - **Many small files**: Increase `max_open_files`
///
/// # Example: Default Configuration
///
/// ```rust
/// use context_graph_storage::RocksDbConfig;
///
/// let config = RocksDbConfig::default();
/// assert_eq!(config.block_cache_size, 256 * 1024 * 1024);
/// assert_eq!(config.max_open_files, 1000);
/// assert!(config.enable_wal);
/// assert!(config.create_if_missing);
/// ```
///
/// # Example: Production Configuration
///
/// ```rust
/// use context_graph_storage::RocksDbConfig;
///
/// let production_config = RocksDbConfig {
///     block_cache_size: 1024 * 1024 * 1024, // 1GB cache
///     max_open_files: 5000,
///     enable_wal: true, // Always enable in production
///     create_if_missing: true,
/// };
/// ```
///
/// # Example: Test Configuration
///
/// ```rust
/// use context_graph_storage::RocksDbConfig;
///
/// let test_config = RocksDbConfig {
///     block_cache_size: 64 * 1024 * 1024, // Smaller for tests
///     max_open_files: 100,
///     enable_wal: false, // Faster tests, ok to lose data
///     create_if_missing: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RocksDbConfig {
    /// Maximum number of open file handles.
    ///
    /// Controls how many SST files RocksDB keeps open simultaneously.
    /// Higher values reduce file open/close overhead but use more FDs.
    ///
    /// Default: 1000
    ///
    /// # Tuning Guidelines
    ///
    /// - Increase if you see "too many open files" errors
    /// - Each column family can have 10+ SST files
    /// - 12 column families * 10 files = ~120 minimum
    pub max_open_files: i32,

    /// Block cache size in bytes.
    ///
    /// The block cache stores uncompressed data blocks from SST files.
    /// This is the primary read cache and significantly impacts read latency.
    ///
    /// Default: 256MB (268,435,456 bytes)
    ///
    /// # Tuning Guidelines
    ///
    /// - Larger = fewer disk reads, more memory
    /// - Working set should fit in cache for best performance
    /// - Constitution requires minimum 256MB
    pub block_cache_size: usize,

    /// Enable Write-Ahead Logging (WAL).
    ///
    /// WAL provides durability by logging writes before applying them.
    /// If disabled, recent writes may be lost on crash.
    ///
    /// Default: true
    ///
    /// # When to Disable
    ///
    /// - Unit tests where durability doesn't matter
    /// - Temporary data that can be regenerated
    /// - Benchmarks measuring pure write speed
    ///
    /// # Warning
    ///
    /// Disabling WAL in production risks data loss on crash!
    pub enable_wal: bool,

    /// Create database directory if it doesn't exist.
    ///
    /// If true, RocksDB creates the directory and initializes the database.
    /// If false, fails with `OpenFailed` if database doesn't exist.
    ///
    /// Default: true
    ///
    /// # When to Set False
    ///
    /// - Production systems that shouldn't auto-create
    /// - Tests verifying database existence
    pub create_if_missing: bool,
}

impl Default for RocksDbConfig {
    /// Creates a configuration with constitution-compliant defaults.
    ///
    /// # Returns
    ///
    /// A `RocksDbConfig` with:
    /// - `max_open_files`: 1000
    /// - `block_cache_size`: 256MB
    /// - `enable_wal`: true
    /// - `create_if_missing`: true
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::RocksDbConfig;
    ///
    /// let config = RocksDbConfig::default();
    /// assert_eq!(config.block_cache_size, 256 * 1024 * 1024);
    /// ```
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
