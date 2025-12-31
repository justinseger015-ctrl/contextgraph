//! In-memory graph index stub for vector similarity search.
//!
//! Provides a brute-force implementation for the Ghost System phase.
//! Production will replace this with FAISS GPU-accelerated search.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{CoreError, CoreResult};
use crate::traits::GraphIndex;
use crate::types::NodeId;

/// In-memory brute-force vector index.
///
/// Uses HashMap storage with linear search for simplicity.
/// Production will use FAISS GPU for high-performance ANN search.
///
/// # Thread Safety
///
/// All operations are thread-safe via `Arc<RwLock<_>>`.
///
/// # Performance
///
/// - add: O(1)
/// - search: O(n * d) where n = vectors, d = dimension
/// - remove: O(1)
///
/// Suitable for Ghost System phase with up to ~10,000 vectors.
/// Production FAISS will handle millions of vectors.
#[derive(Debug)]
pub struct InMemoryGraphIndex {
    /// Vector storage: NodeId -> embedding vector
    vectors: Arc<RwLock<HashMap<NodeId, Vec<f32>>>>,

    /// Vector dimension (e.g., 1536 for OpenAI embeddings)
    dimension: usize,
}

impl InMemoryGraphIndex {
    /// Create a new in-memory index with the specified dimension.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension of vectors to store (e.g., 1536)
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_core::stubs::InMemoryGraphIndex;
    ///
    /// let index = InMemoryGraphIndex::new(1536);
    /// ```
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: Arc::new(RwLock::new(HashMap::new())),
            dimension,
        }
    }

    /// Compute cosine similarity between two vectors.
    ///
    /// Returns a value in [-1.0, 1.0] where 1.0 is identical.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let mut dot_product = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for (ai, bi) in a.iter().zip(b.iter()) {
            dot_product += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }

        let magnitude = (norm_a.sqrt()) * (norm_b.sqrt());
        if magnitude < f32::EPSILON {
            return 0.0;
        }

        dot_product / magnitude
    }
}

impl Clone for InMemoryGraphIndex {
    fn clone(&self) -> Self {
        Self {
            vectors: Arc::clone(&self.vectors),
            dimension: self.dimension,
        }
    }
}

impl Default for InMemoryGraphIndex {
    /// Create a default index with 1536 dimensions (OpenAI-compatible).
    fn default() -> Self {
        Self::new(1536)
    }
}

#[async_trait]
impl GraphIndex for InMemoryGraphIndex {
    /// Add a vector to the index.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::DimensionMismatch` if vector dimension doesn't match.
    async fn add(&self, id: NodeId, vector: &[f32]) -> CoreResult<()> {
        if vector.len() != self.dimension {
            return Err(CoreError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }

        let mut vectors = self.vectors.write().await;
        vectors.insert(id, vector.to_vec());
        Ok(())
    }

    /// Search for nearest neighbors using brute-force cosine similarity.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector
    /// * `k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Vector of (node_id, similarity_score) pairs, sorted by similarity descending.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::DimensionMismatch` if query dimension doesn't match.
    async fn search(&self, query: &[f32], k: usize) -> CoreResult<Vec<(NodeId, f32)>> {
        if query.len() != self.dimension {
            return Err(CoreError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let vectors = self.vectors.read().await;

        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Compute similarity for all vectors
        let mut similarities: Vec<(NodeId, f32)> = vectors
            .iter()
            .map(|(id, vec)| (*id, Self::cosine_similarity(query, vec)))
            .collect();

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        similarities.truncate(k);
        Ok(similarities)
    }

    /// Remove a vector from the index.
    ///
    /// # Returns
    ///
    /// `true` if the vector was found and removed, `false` if not found.
    async fn remove(&self, id: NodeId) -> CoreResult<bool> {
        let mut vectors = self.vectors.write().await;
        Ok(vectors.remove(&id).is_some())
    }

    /// Get the dimension of vectors in this index.
    fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of vectors in the index.
    async fn size(&self) -> CoreResult<usize> {
        let vectors = self.vectors.read().await;
        Ok(vectors.len())
    }

    /// Rebuild the index for optimal search performance.
    ///
    /// For the in-memory stub, this is a no-op.
    /// Production FAISS will rebuild internal data structures.
    async fn rebuild(&self) -> CoreResult<()> {
        // No-op for brute-force index
        // Production FAISS would rebuild internal structures here
        Ok(())
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn create_test_vector(dimension: usize, base_value: f32) -> Vec<f32> {
        (0..dimension)
            .map(|i| base_value + (i as f32 * 0.001))
            .collect()
    }

    fn create_normalized_vector(dimension: usize, seed: u64) -> Vec<f32> {
        let mut vec: Vec<f32> = (0..dimension)
            .map(|i| ((seed as f32 + i as f32).sin() * 100.0).fract())
            .collect();

        // Normalize to unit length
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vec
    }

    #[tokio::test]
    async fn test_new_index() {
        let index = InMemoryGraphIndex::new(1536);
        assert_eq!(index.dimension(), 1536);
        assert_eq!(index.size().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_default_dimension() {
        let index = InMemoryGraphIndex::default();
        assert_eq!(index.dimension(), 1536);
    }

    #[tokio::test]
    async fn test_add_and_size() {
        let index = InMemoryGraphIndex::new(128);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let vec1 = create_test_vector(128, 0.1);
        let vec2 = create_test_vector(128, 0.2);

        index.add(id1, &vec1).await.unwrap();
        assert_eq!(index.size().await.unwrap(), 1);

        index.add(id2, &vec2).await.unwrap();
        assert_eq!(index.size().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_add_dimension_mismatch() {
        let index = InMemoryGraphIndex::new(128);
        let id = Uuid::new_v4();
        let wrong_dim_vec = create_test_vector(64, 0.1); // Wrong dimension

        let result = index.add(id, &wrong_dim_vec).await;
        assert!(result.is_err());

        if let Err(CoreError::DimensionMismatch { expected, actual }) = result {
            assert_eq!(expected, 128);
            assert_eq!(actual, 64);
        } else {
            panic!("Expected DimensionMismatch error");
        }
    }

    #[tokio::test]
    async fn test_search_empty_index() {
        let index = InMemoryGraphIndex::new(128);
        let query = create_test_vector(128, 0.5);

        let results = index.search(&query, 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_finds_exact_match() {
        let index = InMemoryGraphIndex::new(128);
        let id = Uuid::new_v4();
        let vec = create_normalized_vector(128, 42);

        index.add(id, &vec).await.unwrap();

        // Search with the same vector should return similarity ~1.0
        let results = index.search(&vec, 1).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
        assert!(
            (results[0].1 - 1.0).abs() < 0.001,
            "Expected ~1.0, got {}",
            results[0].1
        );
    }

    #[tokio::test]
    async fn test_search_returns_sorted_results() {
        let index = InMemoryGraphIndex::new(128);

        // Create query and vectors with varying similarity
        let query = create_normalized_vector(128, 100);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        // Vector similar to query
        let vec1 = create_normalized_vector(128, 100);
        // Less similar
        let vec2 = create_normalized_vector(128, 200);
        // Even less similar
        let vec3 = create_normalized_vector(128, 500);

        index.add(id3, &vec3).await.unwrap(); // Add in random order
        index.add(id1, &vec1).await.unwrap();
        index.add(id2, &vec2).await.unwrap();

        let results = index.search(&query, 3).await.unwrap();

        assert_eq!(results.len(), 3);

        // Results should be sorted by similarity descending
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);

        // First result should be most similar (exact match)
        assert_eq!(results[0].0, id1);
    }

    #[tokio::test]
    async fn test_search_respects_k_limit() {
        let index = InMemoryGraphIndex::new(128);

        // Add 10 vectors
        for i in 0..10 {
            let id = Uuid::new_v4();
            let vec = create_normalized_vector(128, i as u64);
            index.add(id, &vec).await.unwrap();
        }

        let query = create_normalized_vector(128, 5);

        // Request only 3 results
        let results = index.search(&query, 3).await.unwrap();
        assert_eq!(results.len(), 3);

        // Request more than available
        let results = index.search(&query, 100).await.unwrap();
        assert_eq!(results.len(), 10);
    }

    #[tokio::test]
    async fn test_search_dimension_mismatch() {
        let index = InMemoryGraphIndex::new(128);
        let id = Uuid::new_v4();
        let vec = create_test_vector(128, 0.1);
        index.add(id, &vec).await.unwrap();

        let wrong_dim_query = create_test_vector(64, 0.5);
        let result = index.search(&wrong_dim_query, 5).await;

        assert!(result.is_err());
        if let Err(CoreError::DimensionMismatch { expected, actual }) = result {
            assert_eq!(expected, 128);
            assert_eq!(actual, 64);
        } else {
            panic!("Expected DimensionMismatch error");
        }
    }

    #[tokio::test]
    async fn test_remove_existing() {
        let index = InMemoryGraphIndex::new(128);
        let id = Uuid::new_v4();
        let vec = create_test_vector(128, 0.1);

        index.add(id, &vec).await.unwrap();
        assert_eq!(index.size().await.unwrap(), 1);

        let removed = index.remove(id).await.unwrap();
        assert!(removed);
        assert_eq!(index.size().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_remove_nonexistent() {
        let index = InMemoryGraphIndex::new(128);
        let id = Uuid::new_v4();

        let removed = index.remove(id).await.unwrap();
        assert!(!removed);
    }

    #[tokio::test]
    async fn test_rebuild_succeeds() {
        let index = InMemoryGraphIndex::new(128);
        let id = Uuid::new_v4();
        let vec = create_test_vector(128, 0.1);

        index.add(id, &vec).await.unwrap();

        // Rebuild should succeed (no-op for stub)
        index.rebuild().await.unwrap();

        // Data should still be there
        assert_eq!(index.size().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_clone_shares_data() {
        let index = InMemoryGraphIndex::new(128);
        let id = Uuid::new_v4();
        let vec = create_test_vector(128, 0.1);

        index.add(id, &vec).await.unwrap();

        let cloned = index.clone();

        // Both should see the same data
        assert_eq!(index.size().await.unwrap(), 1);
        assert_eq!(cloned.size().await.unwrap(), 1);

        // Add via clone should be visible in original
        let id2 = Uuid::new_v4();
        let vec2 = create_test_vector(128, 0.2);
        cloned.add(id2, &vec2).await.unwrap();

        assert_eq!(index.size().await.unwrap(), 2);
        assert_eq!(cloned.size().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_deterministic_search() {
        let index = InMemoryGraphIndex::new(128);

        let id1 = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440001").unwrap();
        let id2 = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440002").unwrap();

        let vec1 = create_normalized_vector(128, 1);
        let vec2 = create_normalized_vector(128, 2);
        let query = create_normalized_vector(128, 1);

        index.add(id1, &vec1).await.unwrap();
        index.add(id2, &vec2).await.unwrap();

        // Search twice - should get same results
        let results1 = index.search(&query, 2).await.unwrap();
        let results2 = index.search(&query, 2).await.unwrap();

        assert_eq!(results1.len(), results2.len());
        assert_eq!(results1[0].0, results2[0].0);
        assert_eq!(results1[0].1, results2[0].1);
        assert_eq!(results1[1].0, results2[1].0);
        assert_eq!(results1[1].1, results2[1].1);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let vec = vec![0.5, 0.5, 0.5, 0.5];
        let similarity = InMemoryGraphIndex::cosine_similarity(&vec, &vec);
        assert!((similarity - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let vec1 = vec![1.0, 0.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0, 0.0];
        let similarity = InMemoryGraphIndex::cosine_similarity(&vec1, &vec2);
        assert!(similarity.abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let vec1 = vec![1.0, 0.0];
        let vec2 = vec![-1.0, 0.0];
        let similarity = InMemoryGraphIndex::cosine_similarity(&vec1, &vec2);
        assert!((similarity - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_mismatched_dimensions() {
        let vec1 = vec![1.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        let similarity = InMemoryGraphIndex::cosine_similarity(&vec1, &vec2);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let vec1 = vec![0.0, 0.0, 0.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        let similarity = InMemoryGraphIndex::cosine_similarity(&vec1, &vec2);
        assert_eq!(similarity, 0.0);
    }
}
