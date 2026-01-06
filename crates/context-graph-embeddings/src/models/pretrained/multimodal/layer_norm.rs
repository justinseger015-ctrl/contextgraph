//! Layer normalization implementations for CLIP.
//!
//! Provides layer normalization for both 2D and 3D tensors.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

/// Layer normalization for 3D tensors [batch, seq, hidden].
pub fn layer_norm(
    hidden_states: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> EmbeddingResult<Tensor> {
    // Compute mean and variance along last dimension
    let mean = hidden_states
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm mean failed: {}", e),
        })?;
    let centered = hidden_states
        .broadcast_sub(&mean)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm center failed: {}", e),
        })?;
    let var = centered
        .sqr()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm sqr failed: {}", e),
        })?
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm var mean failed: {}", e),
        })?;

    // Normalize
    let std = (var + eps)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm eps add failed: {}", e),
        })?
        .sqrt()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm sqrt failed: {}", e),
        })?;
    let normalized = centered
        .broadcast_div(&std)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm div failed: {}", e),
        })?;

    // Scale and shift
    let scaled = normalized
        .broadcast_mul(weight)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm scale failed: {}", e),
        })?;
    let output = scaled
        .broadcast_add(bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm shift failed: {}", e),
        })?;

    Ok(output)
}

/// Layer normalization for 2D tensors [batch, hidden].
pub fn layer_norm_1d(
    hidden_states: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> EmbeddingResult<Tensor> {
    // Compute mean and variance along last dimension
    let mean = hidden_states
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm1D mean failed: {}", e),
        })?;
    let centered = hidden_states
        .broadcast_sub(&mean)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm1D center failed: {}", e),
        })?;
    let var = centered
        .sqr()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm1D sqr failed: {}", e),
        })?
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm1D var mean failed: {}", e),
        })?;

    // Normalize
    let std = (var + eps)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm1D eps add failed: {}", e),
        })?
        .sqrt()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm1D sqrt failed: {}", e),
        })?;
    let normalized = centered
        .broadcast_div(&std)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm1D div failed: {}", e),
        })?;

    // Scale and shift
    let scaled = normalized
        .broadcast_mul(weight)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm1D scale failed: {}", e),
        })?;
    let output = scaled
        .broadcast_add(bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LayerNorm1D shift failed: {}", e),
        })?;

    Ok(output)
}
