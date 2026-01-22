//! Tensor operations for Longformer model.
//!
//! This module contains LayerNorm, mean pooling, and L2 normalization.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

/// Apply LayerNorm.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> EmbeddingResult<Tensor> {
    let mean = x
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel LayerNorm mean failed: {}", e),
        })?;

    let x_centered = x
        .broadcast_sub(&mean)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel LayerNorm center failed: {}", e),
        })?;

    let var = x_centered
        .sqr()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel LayerNorm sqr failed: {}", e),
        })?
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel LayerNorm var mean failed: {}", e),
        })?;

    let eps_tensor = Tensor::ones_like(&var)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel LayerNorm create eps ones failed: {}", e),
        })?
        .affine(eps, 0.0)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel LayerNorm eps scale failed: {}", e),
        })?;

    let std = var
        .broadcast_add(&eps_tensor)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel LayerNorm var add eps failed: {}", e),
        })?
        .sqrt()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel LayerNorm sqrt failed: {}", e),
        })?;

    let normalized = x_centered
        .broadcast_div(&std)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel LayerNorm div failed: {}", e),
        })?;

    let scaled = normalized
        .broadcast_mul(weight)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel LayerNorm scale failed: {}", e),
        })?;

    scaled
        .broadcast_add(bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel LayerNorm bias failed: {}", e),
        })
}

/// Mean pooling over sequence dimension.
pub fn mean_pooling(hidden_states: &Tensor, attention_mask: &Tensor) -> EmbeddingResult<Tensor> {
    let mask_expanded = attention_mask
        .unsqueeze(2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel mask expand failed: {}", e),
        })?
        .broadcast_as(hidden_states.shape())
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel mask broadcast failed: {}", e),
        })?;

    let masked_hidden = (hidden_states * mask_expanded).map_err(|e| EmbeddingError::GpuError {
        message: format!("CausalModel mask apply failed: {}", e),
    })?;

    let sum_hidden = masked_hidden.sum(1).map_err(|e| EmbeddingError::GpuError {
        message: format!("CausalModel sum pooling failed: {}", e),
    })?;

    let mask_sum = attention_mask
        .sum_all()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel mask sum failed: {}", e),
        })?;

    sum_hidden
        .broadcast_div(&mask_sum)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel mean div failed: {}", e),
        })
}

/// Marker-weighted pooling over sequence dimension.
///
/// Similar to mean pooling, but gives higher weight to marker tokens.
/// This enables causal-aware pooling where cause or effect indicators
/// contribute more to the final embedding.
///
/// # Arguments
///
/// * `hidden_states` - Hidden states tensor [batch, seq_len, hidden_size]
/// * `attention_mask` - Attention mask [batch, seq_len]
/// * `marker_weights` - Per-token weights [batch, seq_len], e.g., 1.0 for normal, 2.0 for markers
///
/// # Returns
///
/// Pooled tensor [batch, hidden_size]
pub fn marker_weighted_pooling(
    hidden_states: &Tensor,
    attention_mask: &Tensor,
    marker_weights: &Tensor,
) -> EmbeddingResult<Tensor> {
    // Expand mask to hidden dimension: [batch, seq_len] -> [batch, seq_len, hidden_size]
    let mask_expanded = attention_mask
        .unsqueeze(2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel marker pool mask expand failed: {}", e),
        })?
        .broadcast_as(hidden_states.shape())
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel marker pool mask broadcast failed: {}", e),
        })?;

    // Expand marker weights: [batch, seq_len] -> [batch, seq_len, hidden_size]
    let weights_expanded = marker_weights
        .unsqueeze(2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel marker pool weights expand failed: {}", e),
        })?
        .broadcast_as(hidden_states.shape())
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel marker pool weights broadcast failed: {}", e),
        })?;

    // Combined weights: attention_mask * marker_weights
    let combined_weights = (mask_expanded * weights_expanded).map_err(|e| EmbeddingError::GpuError {
        message: format!("CausalModel marker pool combine weights failed: {}", e),
    })?;

    // Weighted hidden states
    let weighted_hidden =
        (hidden_states * combined_weights.clone()).map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel marker pool apply weights failed: {}", e),
        })?;

    // Sum over sequence dimension
    let sum_hidden = weighted_hidden.sum(1).map_err(|e| EmbeddingError::GpuError {
        message: format!("CausalModel marker pool sum failed: {}", e),
    })?;

    // Sum of weights for normalization
    let weight_sum = combined_weights
        .sum(1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel marker pool weight sum failed: {}", e),
        })?;

    // Avoid division by zero - take first element since all dims should be same after sum
    let weight_sum_scalar = weight_sum
        .sum_all()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel marker pool weight scalar failed: {}", e),
        })?;

    sum_hidden
        .broadcast_div(&weight_sum_scalar)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel marker pool div failed: {}", e),
        })
}

/// L2 normalize a tensor.
pub fn l2_normalize(tensor: &Tensor) -> EmbeddingResult<Tensor> {
    let norm = tensor
        .sqr()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel sqr failed: {}", e),
        })?
        .sum_keepdim(1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel norm sum failed: {}", e),
        })?
        .sqrt()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel sqrt failed: {}", e),
        })?;

    let eps_tensor = Tensor::ones_like(&norm)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel create eps ones failed: {}", e),
        })?
        .affine(1e-12, 0.0)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel eps scale failed: {}", e),
        })?;

    tensor
        .broadcast_div(
            &norm
                .broadcast_add(&eps_tensor)
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("CausalModel norm eps add failed: {}", e),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel normalize div failed: {}", e),
        })
}
