//! Vector operations trait definition.

use async_trait::async_trait;

use crate::error::CudaResult;

/// GPU-accelerated vector operations.
///
/// Provides common operations for neural network and similarity search.
#[async_trait]
pub trait VectorOps: Send + Sync {
    /// Compute cosine similarity between two vectors.
    async fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> CudaResult<f32>;

    /// Compute dot product of two vectors.
    async fn dot_product(&self, a: &[f32], b: &[f32]) -> CudaResult<f32>;

    /// Normalize a vector to unit length.
    async fn normalize(&self, v: &[f32]) -> CudaResult<Vec<f32>>;

    /// Batch cosine similarity: compare query against multiple vectors.
    async fn batch_cosine_similarity(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> CudaResult<Vec<f32>>;

    /// Matrix multiplication for attention.
    async fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> CudaResult<Vec<f32>>;

    /// Softmax activation.
    async fn softmax(&self, v: &[f32]) -> CudaResult<Vec<f32>>;

    /// Check if GPU acceleration is available.
    fn is_gpu_available(&self) -> bool;

    /// Get device name.
    fn device_name(&self) -> &str;
}
