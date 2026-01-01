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

    /// Invalid embedding vector (contains NaN or Infinity).
    #[error("Invalid vector: {0}")]
    InvalidVector(String),

    /// Expert weights validation failed (e.g., sum != 1.0).
    #[error("Expert weights invalid: {0}")]
    ExpertWeightsInvalid(String),

    /// Invalid expert index (must be < NUM_EXPERTS).
    #[error("Invalid expert index: {index}, must be < {max}")]
    InvalidExpertIndex { index: u8, max: usize },

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Compression/decompression error.
    #[error("Compression error: {0}")]
    CompressionError(String),

    /// Invalid auxiliary data.
    #[error("Invalid auxiliary data: {0}")]
    InvalidAuxData(String),

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for embedding operations.
pub type EmbeddingResult<T> = Result<T, EmbeddingError>;
