//! Embedding pipeline for Context Graph.
//!
//! This crate provides text-to-embedding conversion using local models.
//! For Phase 0 (Ghost System), stub implementations return deterministic
//! random embeddings.
//!
//! # Architecture
//!
//! - **EmbeddingProvider**: Trait for embedding generation
//! - **StubEmbedder**: Deterministic stub for development
//! - **LocalEmbedder**: Future ONNX/Candle implementation
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_embeddings::{EmbeddingProvider, StubEmbedder};
//!
//! let embedder = StubEmbedder::new(1536);
//! let embedding = embedder.embed("Hello world").await?;
//! assert_eq!(embedding.len(), 1536);
//! ```

pub mod error;
pub mod provider;
pub mod stub;

pub use error::{EmbeddingError, EmbeddingResult};
pub use provider::EmbeddingProvider;
pub use stub::StubEmbedder;

/// Default embedding dimension (OpenAI ada-002 compatible).
pub const DEFAULT_DIMENSION: usize = 1536;
