//! Q, K, V projection logic for self-attention.
//!
//! Handles the linear projections of hidden states into query, key, and value
//! tensors for multi-head self-attention.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

use super::EntityModel;

impl EntityModel {
    /// Compute query projection for self-attention.
    ///
    /// Performs: Q = hidden @ W_q.T + b_q
    pub(crate) fn compute_query_projection(
        hidden_flat: &Tensor,
        attention: &crate::gpu::AttentionWeights,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        layer_idx: usize,
    ) -> EmbeddingResult<Tensor> {
        let query = hidden_flat
            .matmul(
                &attention
                    .query_weight
                    .t()
                    .map_err(|e| EmbeddingError::GpuError {
                        message: format!(
                            "EntityModel layer {} Q transpose failed: {}",
                            layer_idx, e
                        ),
                    })?,
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel layer {} Q matmul failed: {}", layer_idx, e),
            })?
            .reshape((batch_size, seq_len, hidden_size))
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel layer {} Q reshape failed: {}", layer_idx, e),
            })?;

        query
            .broadcast_add(&attention.query_bias)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel layer {} Q bias failed: {}", layer_idx, e),
            })
    }

    /// Compute key projection for self-attention.
    ///
    /// Performs: K = hidden @ W_k.T + b_k
    pub(crate) fn compute_key_projection(
        hidden_flat: &Tensor,
        attention: &crate::gpu::AttentionWeights,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        layer_idx: usize,
    ) -> EmbeddingResult<Tensor> {
        let key = hidden_flat
            .matmul(
                &attention
                    .key_weight
                    .t()
                    .map_err(|e| EmbeddingError::GpuError {
                        message: format!(
                            "EntityModel layer {} K transpose failed: {}",
                            layer_idx, e
                        ),
                    })?,
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel layer {} K matmul failed: {}", layer_idx, e),
            })?
            .reshape((batch_size, seq_len, hidden_size))
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel layer {} K reshape failed: {}", layer_idx, e),
            })?;

        key.broadcast_add(&attention.key_bias)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel layer {} K bias failed: {}", layer_idx, e),
            })
    }

    /// Compute value projection for self-attention.
    ///
    /// Performs: V = hidden @ W_v.T + b_v
    pub(crate) fn compute_value_projection(
        hidden_flat: &Tensor,
        attention: &crate::gpu::AttentionWeights,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        layer_idx: usize,
    ) -> EmbeddingResult<Tensor> {
        let value = hidden_flat
            .matmul(
                &attention
                    .value_weight
                    .t()
                    .map_err(|e| EmbeddingError::GpuError {
                        message: format!(
                            "EntityModel layer {} V transpose failed: {}",
                            layer_idx, e
                        ),
                    })?,
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel layer {} V matmul failed: {}", layer_idx, e),
            })?
            .reshape((batch_size, seq_len, hidden_size))
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel layer {} V reshape failed: {}", layer_idx, e),
            })?;

        value
            .broadcast_add(&attention.value_bias)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel layer {} V bias failed: {}", layer_idx, e),
            })
    }

    /// Reshape and transpose a projection tensor for multi-head attention.
    ///
    /// Transforms from [batch, seq, hidden] to [batch, heads, seq, head_dim]
    /// and makes the tensor contiguous for efficient matmul operations.
    pub(crate) fn reshape_for_attention(
        tensor: Tensor,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        layer_idx: usize,
        name: &str,
    ) -> EmbeddingResult<Tensor> {
        tensor
            .reshape((batch_size, seq_len, num_heads, head_dim))
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} {} reshape failed: {}",
                    layer_idx, name, e
                ),
            })?
            .transpose(1, 2)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} {} transpose 1,2 failed: {}",
                    layer_idx, name, e
                ),
            })?
            .contiguous()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} {} contiguous failed: {}",
                    layer_idx, name, e
                ),
            })
    }
}
