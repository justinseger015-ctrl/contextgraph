//! Forward pass functions for Longformer model.
//!
//! This module implements the neural network forward pass including
//! embeddings computation, encoder layers, and output pooling.
//!
//! # Submodules
//!
//! - `ops`: LayerNorm, mean pooling, L2 normalization
//! - `attention`: Multi-head self-attention
//! - `encoder`: Encoder layers and FFN

mod attention;
mod encoder;
mod ops;

use candle_core::Tensor;
use tokenizers::Tokenizer;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::ModelId;

use super::config::{LongformerConfig, CAUSAL_MAX_TOKENS};
use super::weights::LongformerWeights;

use encoder::run_encoder;
pub use ops::layer_norm;
use ops::{l2_normalize, mean_pooling};

/// GPU-accelerated forward pass for Longformer.
///
/// Note: This implementation uses standard full attention (not sliding window)
/// for simplicity. For very long sequences (>512 tokens), sliding window
/// attention would be more efficient.
pub fn gpu_forward(
    text: &str,
    weights: &LongformerWeights,
    tokenizer: &Tokenizer,
) -> EmbeddingResult<Vec<f32>> {
    let device = weights.device;
    let config = &weights.config;

    // Tokenize input text
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| EmbeddingError::TokenizationError {
            model_id: ModelId::Causal,
            message: format!("CausalModel tokenization failed: {}", e),
        })?;

    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<f32> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as f32)
        .collect();

    // Truncate to max_position_embeddings if needed
    let max_len = config.max_position_embeddings.min(CAUSAL_MAX_TOKENS);
    let seq_len = token_ids.len().min(max_len);
    let token_ids = &token_ids[..seq_len];
    let attention_mask = &attention_mask[..seq_len];

    // Create GPU tensors
    let input_ids = Tensor::from_slice(token_ids, (1, seq_len), device).map_err(|e| {
        EmbeddingError::GpuError {
            message: format!("CausalModel input_ids tensor failed: {}", e),
        }
    })?;

    let attention_mask_tensor =
        Tensor::from_slice(attention_mask, (1, seq_len), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("CausalModel attention_mask tensor failed: {}", e),
            }
        })?;

    // Token type IDs (all zeros for Longformer)
    let token_type_ids: Vec<u32> = vec![0u32; seq_len];
    let token_type_tensor =
        Tensor::from_slice(&token_type_ids, (1, seq_len), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("CausalModel token_type tensor failed: {}", e),
            }
        })?;

    // Position IDs
    let position_ids: Vec<u32> = (0..seq_len as u32).collect();
    let position_tensor = Tensor::from_slice(&position_ids, (1, seq_len), device).map_err(|e| {
        EmbeddingError::GpuError {
            message: format!("CausalModel position_ids tensor failed: {}", e),
        }
    })?;

    // === EMBEDDING LAYER ===
    let embeddings = compute_embeddings(
        &input_ids,
        &position_tensor,
        &token_type_tensor,
        weights,
        config,
        seq_len,
    )?;

    // === ENCODER LAYERS ===
    let hidden_states = run_encoder(embeddings, &attention_mask_tensor, weights, config, seq_len)?;

    // === POOLING ===
    let pooled = mean_pooling(&hidden_states, &attention_mask_tensor)?;

    // L2 normalize
    let normalized = l2_normalize(&pooled)?;

    // Convert to Vec<f32>
    let vector: Vec<f32> = normalized
        .flatten_all()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel flatten output failed: {}", e),
        })?
        .to_vec1()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel to_vec1 failed: {}", e),
        })?;

    Ok(vector)
}

/// Compute embeddings from input tokens.
fn compute_embeddings(
    input_ids: &Tensor,
    position_tensor: &Tensor,
    token_type_tensor: &Tensor,
    weights: &LongformerWeights,
    config: &LongformerConfig,
    seq_len: usize,
) -> EmbeddingResult<Tensor> {
    let word_embeds = weights
        .embeddings
        .word_embeddings
        .index_select(
            &input_ids
                .flatten_all()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("CausalModel flatten input_ids failed: {}", e),
                })?,
            0,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel word embedding lookup failed: {}", e),
        })?
        .reshape((1, seq_len, config.hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel word embedding reshape failed: {}", e),
        })?;

    let position_embeds = weights
        .embeddings
        .position_embeddings
        .index_select(
            &position_tensor
                .flatten_all()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("CausalModel flatten position_ids failed: {}", e),
                })?,
            0,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel position embedding lookup failed: {}", e),
        })?
        .reshape((1, seq_len, config.hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel position embedding reshape failed: {}", e),
        })?;

    let token_type_embeds = weights
        .embeddings
        .token_type_embeddings
        .index_select(
            &token_type_tensor
                .flatten_all()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("CausalModel flatten token_type_ids failed: {}", e),
                })?,
            0,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel token_type embedding lookup failed: {}", e),
        })?
        .reshape((1, seq_len, config.hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel token_type embedding reshape failed: {}", e),
        })?;

    // Sum embeddings
    let embeddings = word_embeds
        .add(&position_embeds)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel embedding add 1 failed: {}", e),
        })?
        .add(&token_type_embeds)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel embedding add 2 failed: {}", e),
        })?;

    // Apply LayerNorm to embeddings
    layer_norm(
        &embeddings,
        &weights.embeddings.layer_norm_weight,
        &weights.embeddings.layer_norm_bias,
        config.layer_norm_eps,
    )
}
