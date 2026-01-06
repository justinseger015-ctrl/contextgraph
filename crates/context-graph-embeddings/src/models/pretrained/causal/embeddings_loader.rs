//! Embedding weight loader for Longformer model.
//!
//! This module handles loading the embedding layer weights from safetensors.

use candle_nn::VarBuilder;

use crate::error::{EmbeddingError, EmbeddingResult};

use super::config::LongformerConfig;
use super::weights::LongformerEmbeddingWeights;

/// Load embedding layer weights.
pub fn load_embeddings(
    vb: &VarBuilder,
    config: &LongformerConfig,
) -> EmbeddingResult<LongformerEmbeddingWeights> {
    let prefix = "longformer.embeddings";

    let word_embeddings = vb
        .get(
            (config.vocab_size, config.hidden_size),
            &format!("{}.word_embeddings.weight", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel word_embeddings load failed: {}", e),
        })?;

    let position_embeddings = vb
        .get(
            (config.max_position_embeddings, config.hidden_size),
            &format!("{}.position_embeddings.weight", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel position_embeddings load failed: {}", e),
        })?;

    let token_type_embeddings = vb
        .get(
            (config.type_vocab_size, config.hidden_size),
            &format!("{}.token_type_embeddings.weight", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel token_type_embeddings load failed: {}", e),
        })?;

    let layer_norm_weight = vb
        .get(
            (config.hidden_size,),
            &format!("{}.LayerNorm.weight", prefix),
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel embedding LayerNorm weight load failed: {}", e),
        })?;

    let layer_norm_bias = vb
        .get((config.hidden_size,), &format!("{}.LayerNorm.bias", prefix))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel embedding LayerNorm bias load failed: {}", e),
        })?;

    Ok(LongformerEmbeddingWeights {
        word_embeddings,
        position_embeddings,
        token_type_embeddings,
        layer_norm_weight,
        layer_norm_bias,
    })
}
