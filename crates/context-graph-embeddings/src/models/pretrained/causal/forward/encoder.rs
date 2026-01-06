//! Encoder layer and FFN forward pass for Longformer model.
//!
//! This module implements the encoder layer with attention + FFN
//! and the feed-forward network (FFN) component.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

use super::super::config::LongformerConfig;
use super::super::weights::{
    LongformerEncoderLayerWeights, LongformerFfnWeights, LongformerWeights,
};
use super::attention::self_attention_forward;
use super::ops::layer_norm;

/// Run encoder layers.
pub fn run_encoder(
    embeddings: Tensor,
    attention_mask_tensor: &Tensor,
    weights: &LongformerWeights,
    config: &LongformerConfig,
    _seq_len: usize,
) -> EmbeddingResult<Tensor> {
    let mut hidden_states = embeddings;

    // Create attention mask for broadcasting
    let extended_attention_mask = attention_mask_tensor
        .unsqueeze(1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel attention mask unsqueeze 1 failed: {}", e),
        })?
        .unsqueeze(2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel attention mask unsqueeze 2 failed: {}", e),
        })?;

    // Invert mask: (1.0 - mask) * -10000.0
    let ones =
        Tensor::ones_like(&extended_attention_mask).map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel create ones tensor failed: {}", e),
        })?;

    let inverted_mask =
        ones.broadcast_sub(&extended_attention_mask)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("CausalModel attention mask invert failed: {}", e),
            })?;

    let extended_attention_mask =
        (inverted_mask * (-10000.0f64)).map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel attention mask scale failed: {}", e),
        })?;

    for (layer_idx, layer) in weights.encoder_layers.iter().enumerate() {
        hidden_states = encoder_layer_forward(
            &hidden_states,
            layer,
            &extended_attention_mask,
            config,
            layer_idx,
        )?;
    }

    Ok(hidden_states)
}

/// Run single encoder layer forward pass.
fn encoder_layer_forward(
    hidden_states: &Tensor,
    layer: &LongformerEncoderLayerWeights,
    attention_mask: &Tensor,
    config: &LongformerConfig,
    layer_idx: usize,
) -> EmbeddingResult<Tensor> {
    // Self-attention
    let attention_output = self_attention_forward(
        hidden_states,
        &layer.attention,
        attention_mask,
        config,
        layer_idx,
    )?;

    // Add & Norm (attention)
    let attention_output =
        hidden_states
            .add(&attention_output)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "CausalModel layer {} attention residual failed: {}",
                    layer_idx, e
                ),
            })?;

    let attention_output = layer_norm(
        &attention_output,
        &layer.attention.layer_norm_weight,
        &layer.attention.layer_norm_bias,
        config.layer_norm_eps,
    )?;

    // FFN
    let ffn_output = ffn_forward(&attention_output, &layer.ffn, layer_idx)?;

    // Add & Norm (FFN)
    let output = attention_output
        .add(&ffn_output)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel layer {} FFN residual failed: {}", layer_idx, e),
        })?;

    layer_norm(
        &output,
        &layer.ffn.layer_norm_weight,
        &layer.ffn.layer_norm_bias,
        config.layer_norm_eps,
    )
}

/// Run FFN forward pass.
fn ffn_forward(
    hidden_states: &Tensor,
    ffn: &LongformerFfnWeights,
    layer_idx: usize,
) -> EmbeddingResult<Tensor> {
    let (batch_size, seq_len, hidden_size) =
        hidden_states
            .dims3()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("CausalModel layer {} FFN get dims failed: {}", layer_idx, e),
            })?;

    let intermediate_size =
        ffn.intermediate_weight
            .dim(0)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "CausalModel layer {} FFN get intermediate_size failed: {}",
                    layer_idx, e
                ),
            })?;

    // Flatten to [batch*seq, hidden]
    let hidden_flat = hidden_states
        .reshape((batch_size * seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} FFN flatten hidden failed: {}",
                layer_idx, e
            ),
        })?;

    // First linear
    let intermediate = hidden_flat
        .matmul(
            &ffn.intermediate_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!(
                        "CausalModel layer {} FFN intermediate transpose failed: {}",
                        layer_idx, e
                    ),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} FFN intermediate matmul failed: {}",
                layer_idx, e
            ),
        })?
        .reshape((batch_size, seq_len, intermediate_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} FFN intermediate reshape failed: {}",
                layer_idx, e
            ),
        })?;

    let intermediate = intermediate
        .broadcast_add(&ffn.intermediate_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} FFN intermediate bias failed: {}",
                layer_idx, e
            ),
        })?;

    // GELU activation
    let intermediate = intermediate.gelu().map_err(|e| EmbeddingError::GpuError {
        message: format!("CausalModel layer {} GELU failed: {}", layer_idx, e),
    })?;

    // Flatten for second linear
    let intermediate_flat = intermediate
        .reshape((batch_size * seq_len, intermediate_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} FFN flatten intermediate failed: {}",
                layer_idx, e
            ),
        })?;

    // Second linear
    let output = intermediate_flat
        .matmul(
            &ffn.output_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!(
                        "CausalModel layer {} FFN output transpose failed: {}",
                        layer_idx, e
                    ),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} FFN output matmul failed: {}",
                layer_idx, e
            ),
        })?
        .reshape((batch_size, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} FFN output reshape failed: {}",
                layer_idx, e
            ),
        })?;

    output
        .broadcast_add(&ffn.output_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} FFN output bias failed: {}",
                layer_idx, e
            ),
        })
}
