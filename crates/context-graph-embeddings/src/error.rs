//! Error types for embedding operations.

use thiserror::Error;

/// Embedding-specific errors.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// Model loading failed.
    #[error("Failed to load embedding model: {0}")]
    ModelLoadError(String),

    /// Embedding generation failed.
    #[error("Failed to generate embedding: {0}")]
    GenerationError(String),

    /// Input validation failed.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for embedding operations.
pub type EmbeddingResult<T> = Result<T, EmbeddingError>;
