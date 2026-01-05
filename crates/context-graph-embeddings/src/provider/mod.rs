//! Embedding provider trait definition and implementations.
//!
//! This module provides the core abstraction for embedding providers:
//! - [`EmbeddingProvider`]: Trait for all embedding providers
//! - [`FusedEmbeddingProvider`]: GPU-accelerated fused embedding provider (1536D output)
//!
//! # Architecture
//!
//! ```text
//! EmbeddingProvider (trait)
//! ├── embed(&str) -> Vec<f32>           // Single text embedding
//! ├── embed_batch(&[&str]) -> Vec<Vec<f32>>  // Batch embedding
//! ├── dimension() -> usize              // Output dimension
//! ├── model_name() -> &str              // Model identifier
//! └── max_tokens() -> usize             // Token limit
//!
//! FusedEmbeddingProvider (struct)
//! ├── semantic_model: SemanticModel     // E1: 1024D dense vectors
//! ├── projection: ProjectionLayer       // Linear 1024D -> 1536D
//! └── config: FusedProviderConfig       // GPU/batch settings
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use context_graph_embeddings::provider::{EmbeddingProvider, FusedEmbeddingProvider};
//!
//! async fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let provider = FusedEmbeddingProvider::new()?;
//!     provider.initialize().await?;
//!
//!     // Generate 1536D embedding (OpenAI ada-002 compatible)
//!     let embedding = provider.embed("Hello, world!").await?;
//!     assert_eq!(embedding.len(), 1536);
//!
//!     // Batch processing for efficiency
//!     let texts = ["Text 1", "Text 2", "Text 3"];
//!     let embeddings = provider.embed_batch(&texts).await?;
//!     assert_eq!(embeddings.len(), 3);
//!
//!     Ok(())
//! }
//! ```

mod fused;

use async_trait::async_trait;

use crate::error::EmbeddingResult;

pub use fused::{FusedEmbeddingProvider, FusedProviderConfig, ProjectionLayer};

/// Trait for embedding providers.
///
/// Implementations convert text to dense vector representations.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding for a single text.
    async fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>>;

    /// Generate embeddings for multiple texts.
    ///
    /// Default implementation calls `embed` for each text.
    /// Implementations may override for batch optimization.
    async fn embed_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Get the output dimension of embeddings.
    fn dimension(&self) -> usize;

    /// Get the model name/identifier.
    fn model_name(&self) -> &str;

    /// Get maximum input token count.
    fn max_tokens(&self) -> usize;
}
