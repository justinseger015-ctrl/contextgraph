//! Core RocksDbMemex struct and database operations.
//!
//! Provides the main database wrapper with open/close and health check functionality.

use rocksdb::{Cache, ColumnFamily, Options, DB};
use std::path::Path;

use crate::column_families::{cf_names, get_column_family_descriptors};

use super::config::RocksDbConfig;
use super::error::StorageError;

/// RocksDB-backed storage implementation.
///
/// Provides persistent storage for MemoryNodes and GraphEdges with
/// optimized column families for different access patterns.
///
/// # Thread Safety
/// RocksDB's `DB` type is internally thread-safe for concurrent reads and writes.
/// This struct can be shared across threads via `Arc<RocksDbMemex>`.
///
/// # Column Families
/// Opens all 12 column families defined in `column_families.rs`.
///
/// # Example
/// ```rust,ignore
/// use context_graph_storage::rocksdb_backend::{RocksDbMemex, RocksDbConfig};
/// use tempfile::TempDir;
///
/// let tmp = TempDir::new().unwrap();
/// let db = RocksDbMemex::open(tmp.path()).expect("open failed");
/// assert!(db.health_check().is_ok());
/// ```
pub struct RocksDbMemex {
    /// The RocksDB database instance.
    pub(crate) db: DB,
    /// Shared block cache (kept alive for DB lifetime).
    #[allow(dead_code)]
    cache: Cache,
    /// Database path for reference.
    path: String,
}

impl RocksDbMemex {
    /// Open a RocksDB database at the specified path with default configuration.
    ///
    /// Creates the database and all 12 column families if they don't exist.
    ///
    /// # Arguments
    /// * `path` - Path to the database directory
    ///
    /// # Returns
    /// * `Ok(RocksDbMemex)` - Successfully opened database
    /// * `Err(StorageError::OpenFailed)` - Database could not be opened
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        Self::open_with_config(path, RocksDbConfig::default())
    }

    /// Open a RocksDB database with custom configuration.
    ///
    /// # Arguments
    /// * `path` - Path to the database directory
    /// * `config` - Custom configuration options
    ///
    /// # Returns
    /// * `Ok(RocksDbMemex)` - Successfully opened database
    /// * `Err(StorageError::OpenFailed)` - Database could not be opened
    pub fn open_with_config<P: AsRef<Path>>(
        path: P,
        config: RocksDbConfig,
    ) -> Result<Self, StorageError> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Create shared block cache
        let cache = Cache::new_lru_cache(config.block_cache_size);

        // Create DB options
        let mut db_opts = Options::default();
        db_opts.create_if_missing(config.create_if_missing);
        db_opts.create_missing_column_families(true);
        db_opts.set_max_open_files(config.max_open_files);

        // WAL configuration
        if !config.enable_wal {
            db_opts.set_manual_wal_flush(true);
        }

        // Get column family descriptors with optimized options
        let cf_descriptors = get_column_family_descriptors(&cache);

        // Open database with all column families
        let db = DB::open_cf_descriptors(&db_opts, &path_str, cf_descriptors).map_err(|e| {
            StorageError::OpenFailed {
                path: path_str.clone(),
                message: e.to_string(),
            }
        })?;

        Ok(Self {
            db,
            cache,
            path: path_str,
        })
    }

    /// Get a reference to a column family by name.
    ///
    /// # Arguments
    /// * `name` - Column family name (use `cf_names::*` constants)
    ///
    /// # Returns
    /// * `Ok(&ColumnFamily)` - Reference to the column family
    /// * `Err(StorageError::ColumnFamilyNotFound)` - CF doesn't exist
    pub fn get_cf(&self, name: &str) -> Result<&ColumnFamily, StorageError> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound {
                name: name.to_string(),
            })
    }

    /// Get the database path.
    ///
    /// # Returns
    /// The path where the database is stored.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Check if the database is healthy.
    ///
    /// Verifies all 12 column families are accessible.
    ///
    /// # Returns
    /// * `Ok(())` - All CFs accessible
    /// * `Err(StorageError::ColumnFamilyNotFound)` - A CF is missing
    pub fn health_check(&self) -> Result<(), StorageError> {
        for cf_name in cf_names::ALL {
            self.get_cf(cf_name)?;
        }
        Ok(())
    }

    /// Flush all column families to disk.
    ///
    /// Forces all buffered writes to be persisted.
    ///
    /// # Returns
    /// * `Ok(())` - All CFs flushed successfully
    /// * `Err(StorageError::FlushFailed)` - Flush operation failed
    pub fn flush_all(&self) -> Result<(), StorageError> {
        for cf_name in cf_names::ALL {
            let cf = self.get_cf(cf_name)?;
            self.db
                .flush_cf(cf)
                .map_err(|e| StorageError::FlushFailed(e.to_string()))?;
        }
        Ok(())
    }

    /// Get a reference to the underlying RocksDB instance.
    ///
    /// Use this for advanced operations not covered by the high-level API.
    /// Be careful not to violate data invariants.
    pub fn db(&self) -> &DB {
        &self.db
    }
}

// DB is automatically closed when RocksDbMemex is dropped (RocksDB's Drop impl)

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_temp_db() -> (TempDir, RocksDbMemex) {
        let tmp = TempDir::new().expect("Failed to create temp dir");
        let db = RocksDbMemex::open(tmp.path()).expect("Failed to open database");
        (tmp, db)
    }

    #[test]
    fn test_open_creates_database() {
        println!("=== TEST: open() creates database ===");
        let tmp = TempDir::new().expect("create temp dir");
        let path = tmp.path();

        println!("BEFORE: Database path = {:?}", path);
        println!("BEFORE: Path exists = {}", path.exists());

        let db = RocksDbMemex::open(path).expect("open failed");

        println!("AFTER: Database opened successfully");
        println!("AFTER: db.path() = {}", db.path());

        assert!(path.exists(), "Database directory should exist");
        assert_eq!(db.path(), path.to_string_lossy());
    }

    #[test]
    fn test_open_with_custom_config() {
        println!("=== TEST: open_with_config() custom settings ===");
        let tmp = TempDir::new().expect("create temp dir");

        let config = RocksDbConfig {
            max_open_files: 100,
            block_cache_size: 64 * 1024 * 1024, // 64MB
            enable_wal: true,
            create_if_missing: true,
        };

        println!("BEFORE: Custom config = {:?}", config);

        let db = RocksDbMemex::open_with_config(tmp.path(), config).expect("open failed");

        println!("AFTER: Database opened with custom config");
        assert!(db.health_check().is_ok());
    }

    #[test]
    fn test_open_invalid_path_fails() {
        let config = RocksDbConfig {
            create_if_missing: false,
            ..Default::default()
        };

        let result = RocksDbMemex::open_with_config("/nonexistent/path/db", config);
        assert!(result.is_err());

        if let Err(StorageError::OpenFailed { path, message }) = result {
            assert!(path.contains("nonexistent"));
            assert!(!message.is_empty());
        }
    }

    #[test]
    fn test_get_cf_returns_valid_handle() {
        let (_tmp, db) = create_temp_db();

        for cf_name in cf_names::ALL {
            let cf = db.get_cf(cf_name);
            assert!(cf.is_ok(), "CF '{}' should exist", cf_name);
        }
    }

    #[test]
    fn test_get_cf_unknown_returns_error() {
        let (_tmp, db) = create_temp_db();

        let result = db.get_cf("nonexistent_cf");
        assert!(result.is_err());

        if let Err(StorageError::ColumnFamilyNotFound { name }) = result {
            assert_eq!(name, "nonexistent_cf");
        } else {
            panic!("Expected ColumnFamilyNotFound error");
        }
    }

    #[test]
    fn test_health_check_passes() {
        let (_tmp, db) = create_temp_db();
        let result = db.health_check();
        assert!(result.is_ok(), "Health check should pass: {:?}", result);
    }

    #[test]
    fn test_flush_all_succeeds() {
        let (_tmp, db) = create_temp_db();
        let result = db.flush_all();
        assert!(result.is_ok(), "Flush should succeed: {:?}", result);
    }

    #[test]
    fn test_db_accessor() {
        let (_tmp, db) = create_temp_db();
        let raw_db = db.db();
        let path = raw_db.path();
        assert!(!path.to_string_lossy().is_empty());
    }

    #[test]
    fn test_path_accessor() {
        let tmp = TempDir::new().expect("create temp dir");
        let expected_path = tmp.path().to_string_lossy().to_string();
        let db = RocksDbMemex::open(tmp.path()).expect("open failed");
        assert_eq!(db.path(), expected_path);
    }

    #[test]
    fn test_reopen_preserves_cfs() {
        println!("=== TEST: Reopen preserves column families ===");
        let tmp = TempDir::new().expect("create temp dir");
        let path = tmp.path().to_path_buf();

        {
            println!("BEFORE: Opening database first time");
            let db = RocksDbMemex::open(&path).expect("first open failed");
            assert!(db.health_check().is_ok());
            println!("AFTER: First open successful, dropping database");
        }

        {
            println!("BEFORE: Reopening database");
            let db = RocksDbMemex::open(&path).expect("reopen failed");
            println!("AFTER: Reopen successful");

            for cf_name in cf_names::ALL {
                let cf = db.get_cf(cf_name);
                assert!(cf.is_ok(), "CF '{}' should exist after reopen", cf_name);
            }
            println!("RESULT: All 12 CFs preserved after reopen");
        }
    }

    #[test]
    fn edge_case_multiple_opens_same_path_fails() {
        println!("=== EDGE CASE 1: Multiple opens on same path ===");
        let tmp = TempDir::new().expect("create temp dir");

        let db1 = RocksDbMemex::open(tmp.path()).expect("first open");
        println!("BEFORE: First database opened at {:?}", tmp.path());

        let result = RocksDbMemex::open(tmp.path());
        println!("AFTER: Second open attempt result = {:?}", result.is_err());

        assert!(result.is_err(), "Second open should fail due to lock");
        drop(db1);
        println!("RESULT: PASS - RocksDB prevents concurrent opens");
    }

    #[test]
    fn edge_case_minimum_cache_size() {
        println!("=== EDGE CASE 2: Minimum cache size (1MB) ===");
        let tmp = TempDir::new().expect("create temp dir");

        let config = RocksDbConfig {
            block_cache_size: 1024 * 1024, // 1MB
            ..Default::default()
        };

        println!("BEFORE: Opening with 1MB cache");
        let db = RocksDbMemex::open_with_config(tmp.path(), config).expect("open failed");
        println!("AFTER: Database opened with minimal cache");

        assert!(db.health_check().is_ok());
        println!("RESULT: PASS - Works with minimum cache");
    }

    #[test]
    fn edge_case_path_with_spaces() {
        println!("=== EDGE CASE 3: Path with spaces ===");
        let tmp = TempDir::new().expect("create temp dir");
        let path_with_spaces = tmp.path().join("path with spaces");
        std::fs::create_dir_all(&path_with_spaces).expect("create dir");

        println!(
            "BEFORE: Opening at path with spaces: {:?}",
            path_with_spaces
        );
        let db = RocksDbMemex::open(&path_with_spaces).expect("open failed");
        println!("AFTER: Database opened successfully");

        assert!(db.health_check().is_ok());
        assert!(db.path().contains("path with spaces"));
        println!("RESULT: PASS - Path with spaces handled correctly");
    }
}
