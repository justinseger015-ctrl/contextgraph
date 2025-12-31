//! Stub embedding provider for development.

use async_trait::async_trait;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::error::EmbeddingResult;
use crate::provider::EmbeddingProvider;

/// Stub embedder for Ghost System phase.
///
/// Generates deterministic embeddings based on input hashing.
/// Same input always produces same embedding, enabling reproducible tests.
#[derive(Debug, Clone)]
pub struct StubEmbedder {
    dimension: usize,
    model_name: String,
    max_tokens: usize,
}

impl StubEmbedder {
    /// Create a new stub embedder with specified dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            model_name: "stub-embedder-v1".to_string(),
            max_tokens: 8192,
        }
    }

    /// Create with default 1536 dimensions.
    pub fn default_dimension() -> Self {
        Self::new(crate::DEFAULT_DIMENSION)
    }

    /// Generate deterministic embedding from text.
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = Vec::with_capacity(self.dimension);

        for i in 0..self.dimension {
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            (i as u64).hash(&mut hasher);
            let hash = hasher.finish();

            // Map to [-1.0, 1.0] range
            let value = ((hash as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
            embedding.push(value);
        }

        // Normalize to unit vector
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }

        embedding
    }
}

impl Default for StubEmbedder {
    fn default() -> Self {
        Self::default_dimension()
    }
}

#[async_trait]
impl EmbeddingProvider for StubEmbedder {
    async fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>> {
        Ok(self.generate_embedding(text))
    }

    async fn embed_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.generate_embedding(t)).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embed_dimension() {
        let embedder = StubEmbedder::new(384);
        let embedding = embedder.embed("test").await.unwrap();
        assert_eq!(embedding.len(), 384);
    }

    #[tokio::test]
    async fn test_deterministic() {
        let embedder = StubEmbedder::default();
        let e1 = embedder.embed("same text").await.unwrap();
        let e2 = embedder.embed("same text").await.unwrap();
        assert_eq!(e1, e2);
    }

    #[tokio::test]
    async fn test_different_inputs() {
        let embedder = StubEmbedder::default();
        let e1 = embedder.embed("text one").await.unwrap();
        let e2 = embedder.embed("text two").await.unwrap();
        assert_ne!(e1, e2);
    }

    #[tokio::test]
    async fn test_normalized() {
        let embedder = StubEmbedder::default();
        let embedding = embedder.embed("test normalization").await.unwrap();
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_batch() {
        let embedder = StubEmbedder::default();
        let texts = vec!["one", "two", "three"];
        let embeddings = embedder.embed_batch(&texts).await.unwrap();
        assert_eq!(embeddings.len(), 3);
    }

    // =========================================================================
    // TC-GHOST-003: Embedding Determinism Tests
    // =========================================================================

    #[tokio::test]
    async fn test_embedding_determinism_neural_network() {
        // TC-GHOST-003: Same input produces same embedding
        let embedder = StubEmbedder::default();
        let e1 = embedder.embed("Neural Network").await.unwrap();
        let e2 = embedder.embed("Neural Network").await.unwrap();

        assert_eq!(e1.len(), 1536, "Embedding must be 1536-dimensional");
        assert_eq!(e1, e2, "Same input must produce identical embedding");
    }

    #[tokio::test]
    async fn test_embedding_determinism_multiple_inputs() {
        // TC-GHOST-003: Verify determinism across multiple different inputs
        let embedder = StubEmbedder::default();

        let test_inputs = [
            "Machine Learning",
            "Deep Neural Networks",
            "Transformer Architecture",
            "Attention Mechanism",
            "Gradient Descent",
        ];

        for input in test_inputs {
            let e1 = embedder.embed(input).await.unwrap();
            let e2 = embedder.embed(input).await.unwrap();

            assert_eq!(
                e1.len(),
                1536,
                "Embedding for '{}' must be 1536-dimensional",
                input
            );
            assert_eq!(
                e1, e2,
                "Same input '{}' must produce identical embedding",
                input
            );
        }
    }

    #[tokio::test]
    async fn test_embedding_determinism_across_instances() {
        // TC-GHOST-003: Determinism must hold across embedder instances
        let embedder1 = StubEmbedder::default();
        let embedder2 = StubEmbedder::default();
        let embedder3 = StubEmbedder::new(1536);

        let e1 = embedder1.embed("Context Graph").await.unwrap();
        let e2 = embedder2.embed("Context Graph").await.unwrap();
        let e3 = embedder3.embed("Context Graph").await.unwrap();

        assert_eq!(e1, e2, "Same input across instances must produce identical embedding");
        assert_eq!(e2, e3, "Same input across instances must produce identical embedding");
    }

    #[tokio::test]
    async fn test_embedding_values_in_valid_range() {
        // TC-GHOST-003: All embedding values must be in [-1.0, 1.0] range
        let embedder = StubEmbedder::default();
        let embedding = embedder.embed("Test value range").await.unwrap();

        for (i, &value) in embedding.iter().enumerate() {
            assert!(
                value >= -1.0 && value <= 1.0,
                "Embedding value at index {} is {} but must be in [-1.0, 1.0]",
                i,
                value
            );
        }
    }

    #[tokio::test]
    async fn test_embedding_batch_determinism() {
        // TC-GHOST-003: Batch embeddings must match individual embeddings
        let embedder = StubEmbedder::default();
        let texts = vec!["batch one", "batch two", "batch three"];

        let batch_embeddings = embedder.embed_batch(&texts).await.unwrap();

        for (i, text) in texts.iter().enumerate() {
            let individual = embedder.embed(text).await.unwrap();
            assert_eq!(
                batch_embeddings[i], individual,
                "Batch embedding for '{}' must match individual embedding",
                text
            );
        }
    }

    #[tokio::test]
    async fn test_embedding_different_inputs_different_outputs() {
        // TC-GHOST-003: Different inputs must produce different embeddings
        let embedder = StubEmbedder::default();
        let e1 = embedder.embed("First distinct input").await.unwrap();
        let e2 = embedder.embed("Second distinct input").await.unwrap();

        assert_ne!(
            e1, e2,
            "Different inputs must produce different embeddings"
        );
    }

    #[tokio::test]
    async fn test_embedding_empty_input_deterministic() {
        // TC-GHOST-003: Even empty input must be deterministic
        let embedder = StubEmbedder::default();
        let e1 = embedder.embed("").await.unwrap();
        let e2 = embedder.embed("").await.unwrap();

        assert_eq!(e1.len(), 1536, "Empty input embedding must be 1536-dimensional");
        assert_eq!(e1, e2, "Empty input must produce identical embedding");
    }
}
