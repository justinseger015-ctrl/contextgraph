//! Graph index trait for vector similarity search.

use async_trait::async_trait;

use crate::error::CoreResult;
use crate::types::NodeId;

/// Vector index for similarity search.
///
/// Abstracts over FAISS, HNSW, or other ANN implementations.
#[async_trait]
pub trait GraphIndex: Send + Sync {
    /// Add a vector to the index.
    async fn add(&self, id: NodeId, vector: &[f32]) -> CoreResult<()>;

    /// Search for nearest neighbors.
    ///
    /// Returns (node_id, similarity_score) pairs.
    async fn search(&self, query: &[f32], k: usize) -> CoreResult<Vec<(NodeId, f32)>>;

    /// Remove a vector from the index.
    async fn remove(&self, id: NodeId) -> CoreResult<bool>;

    /// Get the dimension of vectors in this index.
    fn dimension(&self) -> usize;

    /// Get the number of vectors in the index.
    async fn size(&self) -> CoreResult<usize>;

    /// Rebuild the index for optimal search performance.
    async fn rebuild(&self) -> CoreResult<()>;
}
