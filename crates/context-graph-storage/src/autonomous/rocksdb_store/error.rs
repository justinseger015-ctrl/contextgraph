//! Error types for autonomous store operations.
//!
//! # FAIL FAST Policy
//!
//! **NO FALLBACKS. NO MOCK DATA. ERRORS ARE FATAL.**
//!
//! Every error includes enough context for immediate debugging:
//! - Operation name
//! - Column family
//! - Key (if applicable)
//! - Underlying cause

use thiserror::Error;

/// Serialization version for autonomous storage types.
///
/// Bump this when struct layout changes. Version mismatches will cause errors.
pub const AUTONOMOUS_STORAGE_VERSION: u8 = 1;

/// Detailed error type for autonomous store operations.
///
/// Every error includes enough context for immediate debugging:
/// - Operation name
/// - Column family
/// - Key (if applicable)
/// - Underlying cause
#[derive(Debug, Error)]
pub enum AutonomousStoreError {
    /// RocksDB operation failed.
    #[error("RocksDB {operation} failed on CF '{cf}' with key '{key:?}': {source}")]
    RocksDbOperation {
        operation: &'static str,
        cf: &'static str,
        key: Option<String>,
        #[source]
        source: rocksdb::Error,
    },

    /// Database failed to open.
    #[error("Failed to open RocksDB at '{path}': {message}")]
    OpenFailed { path: String, message: String },

    /// Column family not found.
    #[error("Column family '{name}' not found in database")]
    ColumnFamilyNotFound { name: String },

    /// Serialization error.
    #[error("Serialization error for {type_name}: {message}")]
    Serialization {
        type_name: &'static str,
        message: String,
    },

    /// Deserialization error.
    #[error("Deserialization error for key '{key}' in CF '{cf}': {message}")]
    Deserialization {
        cf: &'static str,
        key: String,
        message: String,
    },

    /// Version mismatch error.
    #[error("Version mismatch in CF '{cf}': expected {expected}, got {actual}")]
    VersionMismatch {
        cf: &'static str,
        expected: u8,
        actual: u8,
    },
}

impl AutonomousStoreError {
    /// Create a RocksDB operation error.
    pub fn rocksdb_op(
        operation: &'static str,
        cf: &'static str,
        key: Option<&str>,
        source: rocksdb::Error,
    ) -> Self {
        Self::RocksDbOperation {
            operation,
            cf,
            key: key.map(String::from),
            source,
        }
    }
}

/// Result type for autonomous store operations.
pub type AutonomousStoreResult<T> = Result<T, AutonomousStoreError>;
