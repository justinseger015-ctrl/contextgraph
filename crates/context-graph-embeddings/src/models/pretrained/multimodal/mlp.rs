//! MLP block implementation for CLIP transformer layers.
//!
//! Uses QuickGELU activation: x * sigmoid(1.702 * x)

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

/// MLP block: FC1 -> QuickGELU -> FC2.
///
/// Uses flatten/reshape pattern for Candle matmul compatibility (requires 2D tensors).
pub fn mlp(
    hidden_states: &Tensor,
    fc1_weight: &Tensor,
    fc1_bias: &Tensor,
    fc2_weight: &Tensor,
    fc2_bias: &Tensor,
) -> EmbeddingResult<Tensor> {
    let (batch, seq_len, hidden_size) =
        hidden_states
            .dims3()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("MLP dims3 failed: {}", e),
            })?;

    let intermediate_size = fc1_weight.dim(0).map_err(|e| EmbeddingError::GpuError {
        message: format!("FC1 get intermediate_size failed: {}", e),
    })?;

    // Flatten to [batch*seq, hidden] for Candle matmul compatibility
    let hidden_flat = hidden_states
        .reshape((batch * seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("MLP flatten hidden failed: {}", e),
        })?;

    // FC1 with flatten/reshape pattern
    let hidden = hidden_flat
        .matmul(&fc1_weight.t().map_err(|e| EmbeddingError::GpuError {
            message: format!("FC1 weight transpose failed: {}", e),
        })?)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("FC1 matmul failed: {}", e),
        })?
        .reshape((batch, seq_len, intermediate_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("FC1 reshape failed: {}", e),
        })?;
    let hidden = hidden
        .broadcast_add(fc1_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("FC1 bias failed: {}", e),
        })?;

    // QuickGELU: x * sigmoid(1.702 * x)
    let hidden = quick_gelu(&hidden)?;

    // Flatten for FC2
    let hidden_flat = hidden
        .reshape((batch * seq_len, intermediate_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("MLP flatten intermediate failed: {}", e),
        })?;

    // FC2 with flatten/reshape pattern
    let output = hidden_flat
        .matmul(&fc2_weight.t().map_err(|e| EmbeddingError::GpuError {
            message: format!("FC2 weight transpose failed: {}", e),
        })?)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("FC2 matmul failed: {}", e),
        })?
        .reshape((batch, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("FC2 reshape failed: {}", e),
        })?;
    let output = output
        .broadcast_add(fc2_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("FC2 bias failed: {}", e),
        })?;

    Ok(output)
}

/// QuickGELU activation: x * sigmoid(1.702 * x)
fn quick_gelu(hidden: &Tensor) -> EmbeddingResult<Tensor> {
    let sigmoid_input = (hidden.clone() * 1.702).map_err(|e| EmbeddingError::GpuError {
        message: format!("QuickGELU scale failed: {}", e),
    })?;
    let sigmoid =
        candle_nn::ops::sigmoid(&sigmoid_input).map_err(|e| EmbeddingError::GpuError {
            message: format!("Sigmoid failed: {}", e),
        })?;
    (hidden.clone() * sigmoid).map_err(|e| EmbeddingError::GpuError {
        message: format!("QuickGELU mul failed: {}", e),
    })
}
