//! LayerNorm implementation for BERT models.
//!
//! Implements the standard LayerNorm operation:
//! `(x - mean) / sqrt(var + eps) * weight + bias`

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

use super::EntityModel;

impl EntityModel {
    /// Apply LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq_len, hidden_size]
    /// * `weight` - Scale parameter (gamma) of shape [hidden_size]
    /// * `bias` - Shift parameter (beta) of shape [hidden_size]
    /// * `eps` - Small constant for numerical stability
    ///
    /// # Returns
    /// Normalized tensor with same shape as input
    pub(crate) fn layer_norm(
        x: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f64,
    ) -> EmbeddingResult<Tensor> {
        // Compute mean along last dimension
        let mean =
            x.mean_keepdim(candle_core::D::Minus1)
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("EntityModel LayerNorm mean failed: {}", e),
                })?;

        // Center the input
        let x_centered = x
            .broadcast_sub(&mean)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel LayerNorm center failed: {}", e),
            })?;

        // Compute variance
        let var = x_centered
            .sqr()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel LayerNorm sqr failed: {}", e),
            })?
            .mean_keepdim(candle_core::D::Minus1)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel LayerNorm var mean failed: {}", e),
            })?;

        // Compute standard deviation with epsilon for stability
        let std = (var + eps)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel LayerNorm var add eps failed: {}", e),
            })?
            .sqrt()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel LayerNorm sqrt failed: {}", e),
            })?;

        // Normalize
        let normalized = x_centered
            .broadcast_div(&std)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel LayerNorm div failed: {}", e),
            })?;

        // Scale by weight (gamma)
        let scaled = normalized
            .broadcast_mul(weight)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel LayerNorm scale failed: {}", e),
            })?;

        // Add bias (beta)
        scaled
            .broadcast_add(bias)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel LayerNorm bias failed: {}", e),
            })
    }
}
