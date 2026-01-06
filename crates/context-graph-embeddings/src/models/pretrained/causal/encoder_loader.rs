//! Encoder layer weight loader for Longformer model.
//!
//! This module handles loading encoder layer weights including
//! attention and feed-forward network components from safetensors.

use candle_nn::VarBuilder;

use crate::error::{EmbeddingError, EmbeddingResult};

use super::config::LongformerConfig;
use super::weights::{
    LongformerAttentionWeights, LongformerEncoderLayerWeights, LongformerFfnWeights,
};

/// Load a single encoder layer.
pub fn load_encoder_layer(
    vb: &VarBuilder,
    config: &LongformerConfig,
    layer_idx: usize,
) -> EmbeddingResult<LongformerEncoderLayerWeights> {
    let attention = load_attention_weights(vb, config, layer_idx)?;
    let ffn = load_ffn_weights(vb, config, layer_idx)?;
    Ok(LongformerEncoderLayerWeights { attention, ffn })
}

/// Load attention weights for a layer.
fn load_attention_weights(
    vb: &VarBuilder,
    config: &LongformerConfig,
    layer_idx: usize,
) -> EmbeddingResult<LongformerAttentionWeights> {
    let prefix = format!("longformer.encoder.layer.{}.attention", layer_idx);

    let query_weight = vb
        .get(
            (config.hidden_size, config.hidden_size),
            &format!("{}.self.query.weight", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} query weight load failed: {}",
                layer_idx, e
            ),
        })?;
    let query_bias = vb
        .get(
            (config.hidden_size,),
            &format!("{}.self.query.bias", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} query bias load failed: {}",
                layer_idx, e
            ),
        })?;

    let key_weight = vb
        .get(
            (config.hidden_size, config.hidden_size),
            &format!("{}.self.key.weight", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} key weight load failed: {}",
                layer_idx, e
            ),
        })?;
    let key_bias = vb
        .get((config.hidden_size,), &format!("{}.self.key.bias", prefix))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} key bias load failed: {}",
                layer_idx, e
            ),
        })?;

    let value_weight = vb
        .get(
            (config.hidden_size, config.hidden_size),
            &format!("{}.self.value.weight", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} value weight load failed: {}",
                layer_idx, e
            ),
        })?;
    let value_bias = vb
        .get(
            (config.hidden_size,),
            &format!("{}.self.value.bias", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} value bias load failed: {}",
                layer_idx, e
            ),
        })?;

    let output_weight = vb
        .get(
            (config.hidden_size, config.hidden_size),
            &format!("{}.output.dense.weight", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} output weight load failed: {}",
                layer_idx, e
            ),
        })?;
    let output_bias = vb
        .get(
            (config.hidden_size,),
            &format!("{}.output.dense.bias", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} output bias load failed: {}",
                layer_idx, e
            ),
        })?;

    let layer_norm_weight = vb
        .get(
            (config.hidden_size,),
            &format!("{}.output.LayerNorm.weight", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} attention LayerNorm weight load failed: {}",
                layer_idx, e
            ),
        })?;
    let layer_norm_bias = vb
        .get(
            (config.hidden_size,),
            &format!("{}.output.LayerNorm.bias", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} attention LayerNorm bias load failed: {}",
                layer_idx, e
            ),
        })?;

    Ok(LongformerAttentionWeights {
        query_weight,
        query_bias,
        key_weight,
        key_bias,
        value_weight,
        value_bias,
        output_weight,
        output_bias,
        layer_norm_weight,
        layer_norm_bias,
    })
}

/// Load FFN weights for a layer.
fn load_ffn_weights(
    vb: &VarBuilder,
    config: &LongformerConfig,
    layer_idx: usize,
) -> EmbeddingResult<LongformerFfnWeights> {
    let prefix = format!("longformer.encoder.layer.{}", layer_idx);

    let intermediate_weight = vb
        .get(
            (config.intermediate_size, config.hidden_size),
            &format!("{}.intermediate.dense.weight", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} intermediate weight load failed: {}",
                layer_idx, e
            ),
        })?;
    let intermediate_bias = vb
        .get(
            (config.intermediate_size,),
            &format!("{}.intermediate.dense.bias", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} intermediate bias load failed: {}",
                layer_idx, e
            ),
        })?;

    let output_weight = vb
        .get(
            (config.hidden_size, config.intermediate_size),
            &format!("{}.output.dense.weight", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} FFN output weight load failed: {}",
                layer_idx, e
            ),
        })?;
    let output_bias = vb
        .get(
            (config.hidden_size,),
            &format!("{}.output.dense.bias", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} FFN output bias load failed: {}",
                layer_idx, e
            ),
        })?;

    let layer_norm_weight = vb
        .get(
            (config.hidden_size,),
            &format!("{}.output.LayerNorm.weight", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} FFN LayerNorm weight load failed: {}",
                layer_idx, e
            ),
        })?;
    let layer_norm_bias = vb
        .get(
            (config.hidden_size,),
            &format!("{}.output.LayerNorm.bias", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CausalModel layer {} FFN LayerNorm bias load failed: {}",
                layer_idx, e
            ),
        })?;

    Ok(LongformerFfnWeights {
        intermediate_weight,
        intermediate_bias,
        output_weight,
        output_bias,
        layer_norm_weight,
        layer_norm_bias,
    })
}
