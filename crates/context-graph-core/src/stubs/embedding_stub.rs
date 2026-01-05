//! Stub embedding provider for testing.
//!
//! Provides deterministic embeddings without requiring GPU or model files.
//! Useful for unit tests and development environments.

use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::error::CoreResult;
use crate::traits::{EmbeddingOutput, EmbeddingProvider};

/// Stub embedding provider for testing.
///
/// Generates deterministic embeddings based on content hash.
/// Does not require GPU or model files.
///
/// # Example
///
/// ```ignore
/// use context_graph_core::stubs::StubEmbeddingProvider;
/// use context_graph_core::traits::EmbeddingProvider;
///
/// # async fn example() {
/// let provider = StubEmbeddingProvider::new();
/// let output = provider.embed("test content").await.unwrap();
/// assert_eq!(output.dimensions, 1536);
/// # }
/// ```
pub struct StubEmbeddingProvider {
    dimensions: usize,
    model_id: String,
}

impl Default for StubEmbeddingProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl StubEmbeddingProvider {
    /// Create a new stub embedding provider with default 1536 dimensions.
    pub fn new() -> Self {
        Self {
            dimensions: 1536,
            model_id: "stub-embedding-v1".to_string(),
        }
    }

    /// Create a stub provider with custom dimensions.
    pub fn with_dimensions(dimensions: usize) -> Self {
        Self {
            dimensions,
            model_id: "stub-embedding-v1".to_string(),
        }
    }

    /// Generate a deterministic embedding from content.
    ///
    /// Uses a simple hash-based approach to generate reproducible embeddings.
    fn generate_embedding(&self, content: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimensions];

        // Generate deterministic values based on content
        let bytes = content.as_bytes();
        for (i, chunk) in embedding.chunks_mut(1).enumerate() {
            // Use a simple hash combining index and content bytes
            let byte_idx = i % bytes.len().max(1);
            let byte_val = bytes.get(byte_idx).copied().unwrap_or(0) as f32;
            let idx_contribution = (i as f32 / self.dimensions as f32) * 0.5;
            let byte_contribution = (byte_val / 255.0) * 0.3;
            let base = 0.1 + idx_contribution + byte_contribution;

            // Normalize to [-1, 1] range
            chunk[0] = (base * 2.0) - 1.0;
        }

        // L2 normalize the embedding
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }
}

#[async_trait]
impl EmbeddingProvider for StubEmbeddingProvider {
    async fn embed(&self, content: &str) -> CoreResult<EmbeddingOutput> {
        let start = Instant::now();
        let vector = self.generate_embedding(content);
        let latency = start.elapsed();

        Ok(EmbeddingOutput::new(
            vector,
            self.model_id.clone(),
            latency,
        ))
    }

    async fn embed_batch(&self, contents: &[String]) -> CoreResult<Vec<EmbeddingOutput>> {
        let start = Instant::now();

        let outputs: Vec<EmbeddingOutput> = contents
            .iter()
            .map(|content| {
                let vector = self.generate_embedding(content);
                EmbeddingOutput::new(vector, self.model_id.clone(), Duration::from_micros(100))
            })
            .collect();

        let total_latency = start.elapsed();
        tracing::debug!(
            "Stub batch embed of {} items took {:?}",
            contents.len(),
            total_latency
        );

        Ok(outputs)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn is_ready(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stub_embed_dimensions() {
        let provider = StubEmbeddingProvider::new();
        let output = provider.embed("test content").await.unwrap();

        assert_eq!(output.dimensions, 1536);
        assert_eq!(output.vector.len(), 1536);
    }

    #[tokio::test]
    async fn test_stub_embed_deterministic() {
        let provider = StubEmbeddingProvider::new();
        let output1 = provider.embed("same content").await.unwrap();
        let output2 = provider.embed("same content").await.unwrap();

        assert_eq!(output1.vector, output2.vector);
    }

    #[tokio::test]
    async fn test_stub_embed_different_content() {
        let provider = StubEmbeddingProvider::new();
        let output1 = provider.embed("content A").await.unwrap();
        let output2 = provider.embed("content B").await.unwrap();

        // Different content should produce different embeddings
        assert_ne!(output1.vector, output2.vector);
    }

    #[tokio::test]
    async fn test_stub_embed_batch() {
        let provider = StubEmbeddingProvider::new();
        let contents = vec![
            "first".to_string(),
            "second".to_string(),
            "third".to_string(),
        ];

        let outputs = provider.embed_batch(&contents).await.unwrap();

        assert_eq!(outputs.len(), 3);
        for output in &outputs {
            assert_eq!(output.dimensions, 1536);
        }
    }

    #[tokio::test]
    async fn test_stub_is_ready() {
        let provider = StubEmbeddingProvider::new();
        assert!(provider.is_ready());
    }

    #[tokio::test]
    async fn test_stub_custom_dimensions() {
        let provider = StubEmbeddingProvider::with_dimensions(768);
        let output = provider.embed("test").await.unwrap();

        assert_eq!(output.dimensions, 768);
        assert_eq!(output.vector.len(), 768);
    }

    #[tokio::test]
    async fn test_embedding_is_normalized() {
        let provider = StubEmbeddingProvider::new();
        let output = provider.embed("test content").await.unwrap();

        // Check L2 norm is approximately 1.0
        let norm: f32 = output.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001, "Embedding should be L2 normalized");
    }
}
