//! Embedding provider trait definition.
//!
//! This module provides the core abstraction for embedding providers:
//! - [`EmbeddingProvider`]: Trait for all embedding providers
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
//! ```

use async_trait::async_trait;

use crate::error::EmbeddingResult;

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
