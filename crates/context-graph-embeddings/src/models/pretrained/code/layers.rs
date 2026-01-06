//! Encoder layer and normalization for CodeT5+.
//!
//! Contains:
//! - RMSNorm (T5-style layer normalization)
//! - Encoder layer forward pass
//! - Feed-forward network (DenseReluDense)

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

use super::attention::self_attention_forward;
use super::config::CodeT5pConfig;
use super::weights::{T5EncoderLayerWeights, T5FfnWeights};

/// Apply RMSNorm (T5 style LayerNorm without bias).
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> EmbeddingResult<Tensor> {
    let variance = x
        .sqr()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel RMSNorm sqr failed: {}", e),
        })?
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel RMSNorm mean failed: {}", e),
        })?;

    let eps_tensor = Tensor::ones_like(&variance)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel RMSNorm create eps ones failed: {}", e),
        })?
        .affine(eps, 0.0)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel RMSNorm eps scale failed: {}", e),
        })?;

    let normalized = x
        .broadcast_div(
            &variance
                .broadcast_add(&eps_tensor)
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("CodeModel RMSNorm eps add failed: {}", e),
                })?
                .sqrt()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("CodeModel RMSNorm sqrt failed: {}", e),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel RMSNorm div failed: {}", e),
        })?;

    normalized
        .broadcast_mul(weight)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel RMSNorm mul failed: {}", e),
        })
}

/// Run single encoder layer forward pass.
pub fn encoder_layer_forward(
    hidden_states: &Tensor,
    layer: &T5EncoderLayerWeights,
    attention_mask: &Tensor,
    position_bias: &Tensor,
    config: &CodeT5pConfig,
    layer_idx: usize,
) -> EmbeddingResult<Tensor> {
    // Pre-norm: apply layer norm before attention
    let normed = rms_norm(
        hidden_states,
        &layer.attention_layer_norm_weight,
        config.layer_norm_epsilon,
    )?;

    // Self-attention
    let attention_output = self_attention_forward(
        &normed,
        &layer.attention,
        attention_mask,
        position_bias,
        config,
        layer_idx,
    )?;

    // Residual connection
    let hidden_states =
        hidden_states
            .add(&attention_output)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "CodeModel layer {} attention residual failed: {}",
                    layer_idx, e
                ),
            })?;

    // Pre-norm: apply layer norm before FFN
    let normed = rms_norm(
        &hidden_states,
        &layer.ffn.layer_norm_weight,
        config.layer_norm_epsilon,
    )?;

    // FFN
    let ffn_output = ffn_forward(&normed, &layer.ffn, layer_idx)?;

    // Residual connection
    hidden_states
        .add(&ffn_output)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel layer {} FFN residual failed: {}", layer_idx, e),
        })
}

/// Run FFN forward pass (DenseReluDense).
pub fn ffn_forward(
    hidden_states: &Tensor,
    ffn: &T5FfnWeights,
    layer_idx: usize,
) -> EmbeddingResult<Tensor> {
    let (batch_size, seq_len, hidden_size) =
        hidden_states
            .dims3()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("CodeModel layer {} FFN get dims failed: {}", layer_idx, e),
            })?;

    // Flatten to [batch*seq, hidden] for Candle matmul compatibility
    let hidden_flat = hidden_states
        .reshape((batch_size * seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} FFN flatten hidden failed: {}",
                layer_idx, e
            ),
        })?;

    // wi: [d_model] -> [d_ff]
    let intermediate = hidden_flat
        .matmul(&ffn.wi_weight.t().map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} FFN wi transpose failed: {}",
                layer_idx, e
            ),
        })?)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel layer {} FFN wi matmul failed: {}", layer_idx, e),
        })?;

    // ReLU activation
    let intermediate = intermediate.relu().map_err(|e| EmbeddingError::GpuError {
        message: format!("CodeModel layer {} ReLU failed: {}", layer_idx, e),
    })?;

    // wo: [d_ff] -> [d_model]
    intermediate
        .matmul(&ffn.wo_weight.t().map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} FFN wo transpose failed: {}",
                layer_idx, e
            ),
        })?)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel layer {} FFN wo matmul failed: {}", layer_idx, e),
        })?
        .reshape((batch_size, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} FFN reshape output failed: {}",
                layer_idx, e
            ),
        })
}
