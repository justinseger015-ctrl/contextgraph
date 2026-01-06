//! Attention score computation and output projection.
//!
//! Handles the attention score computation (Q @ K^T), softmax, context
//! aggregation, and final output projection.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

use super::EntityModel;

impl EntityModel {
    /// Compute scaled dot-product attention scores.
    ///
    /// Computes: scores = (Q @ K^T) / sqrt(head_dim) + attention_mask
    pub(crate) fn compute_attention_scores(
        query: &Tensor,
        key: &Tensor,
        attention_mask: &Tensor,
        head_dim: usize,
        layer_idx: usize,
    ) -> EmbeddingResult<Tensor> {
        // Transpose key for Q @ K^T
        let key_t = key
            .transpose(2, 3)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} K transpose 2,3 failed: {}",
                    layer_idx, e
                ),
            })?
            .contiguous()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} K^T contiguous failed: {}",
                    layer_idx, e
                ),
            })?;

        // Q @ K^T
        let scores = query.matmul(&key_t).map_err(|e| EmbeddingError::GpuError {
            message: format!("EntityModel layer {} QK matmul failed: {}", layer_idx, e),
        })?;

        // Scale by 1/sqrt(head_dim)
        let scale = (head_dim as f64).sqrt();
        let scores = scores
            .affine(1.0 / scale, 0.0)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} attention scale failed: {}",
                    layer_idx, e
                ),
            })?;

        // Apply attention mask (broadcast from [1, 1, 1, seq] to [batch, heads, seq, seq])
        scores
            .broadcast_add(attention_mask)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} attention mask add failed: {}",
                    layer_idx, e
                ),
            })
    }

    /// Apply softmax to attention scores.
    pub(crate) fn apply_attention_softmax(
        scores: &Tensor,
        layer_idx: usize,
    ) -> EmbeddingResult<Tensor> {
        candle_nn::ops::softmax(scores, candle_core::D::Minus1).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("EntityModel layer {} softmax failed: {}", layer_idx, e),
            }
        })
    }

    /// Compute attention context by aggregating values.
    ///
    /// Computes: context = softmax(scores) @ V
    /// Then reshapes back to [batch, seq, hidden]
    pub(crate) fn compute_attention_context(
        attention_probs: &Tensor,
        value: &Tensor,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        layer_idx: usize,
    ) -> EmbeddingResult<Tensor> {
        // Context: attention_probs @ V
        let context = attention_probs
            .matmul(value)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} context matmul failed: {}",
                    layer_idx, e
                ),
            })?;

        // Reshape back to [batch, seq_len, hidden_size]
        context
            .transpose(1, 2)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} context transpose failed: {}",
                    layer_idx, e
                ),
            })?
            .contiguous()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} context contiguous failed: {}",
                    layer_idx, e
                ),
            })?
            .reshape((batch_size, seq_len, hidden_size))
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} context reshape failed: {}",
                    layer_idx, e
                ),
            })
    }

    /// Apply output projection to attention context.
    ///
    /// Computes: output = context @ W_o.T + b_o
    pub(crate) fn apply_output_projection(
        context: &Tensor,
        attention: &crate::gpu::AttentionWeights,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        layer_idx: usize,
    ) -> EmbeddingResult<Tensor> {
        // Flatten for matmul
        let context_flat = context
            .reshape((batch_size * seq_len, hidden_size))
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} context flatten failed: {}",
                    layer_idx, e
                ),
            })?;

        // Output projection: context @ W_o.T
        let output = context_flat
            .matmul(
                &attention
                    .output_weight
                    .t()
                    .map_err(|e| EmbeddingError::GpuError {
                        message: format!(
                            "EntityModel layer {} output transpose failed: {}",
                            layer_idx, e
                        ),
                    })?,
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} output matmul failed: {}",
                    layer_idx, e
                ),
            })?
            .reshape((batch_size, seq_len, hidden_size))
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "EntityModel layer {} output reshape failed: {}",
                    layer_idx, e
                ),
            })?;

        // Add output bias
        output
            .broadcast_add(&attention.output_bias)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel layer {} output bias failed: {}", layer_idx, e),
            })
    }
}
