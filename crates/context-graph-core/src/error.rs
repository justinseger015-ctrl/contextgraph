//! Error types for context-graph-core.

use thiserror::Error;
use uuid::Uuid;

/// Top-level error type for context-graph-core.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Node not found: {id}")]
    NodeNotFound { id: Uuid },

    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Validation error: {field} - {message}")]
    ValidationError { field: String, message: String },

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Index error: {0}")]
    IndexError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("UTL computation error: {0}")]
    UtlError(String),

    #[error("Layer processing error in {layer}: {message}")]
    LayerError { layer: String, message: String },

    #[error("Feature disabled: {feature}")]
    FeatureDisabled { feature: String },

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<serde_json::Error> for CoreError {
    fn from(err: serde_json::Error) -> Self {
        CoreError::SerializationError(err.to_string())
    }
}

impl From<config::ConfigError> for CoreError {
    fn from(err: config::ConfigError) -> Self {
        CoreError::ConfigError(err.to_string())
    }
}

/// Result type alias for core operations.
pub type CoreResult<T> = Result<T, CoreError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CoreError::NodeNotFound { id: Uuid::nil() };
        assert!(err.to_string().contains("Node not found"));
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = CoreError::DimensionMismatch {
            expected: 1536,
            actual: 768,
        };
        assert!(err.to_string().contains("1536"));
        assert!(err.to_string().contains("768"));
    }
}
