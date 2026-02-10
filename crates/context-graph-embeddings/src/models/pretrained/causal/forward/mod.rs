//! Forward pass for NomicBERT (nomic-embed-text-v1.5).
//!
//! # Submodules
//!
//! - `ops`: LayerNorm, mean pooling, L2 normalization
//! - `attention`: Fused QKV self-attention with rotary position embeddings
//! - `encoder`: Encoder layers with SwiGLU FFN
//!
//! # Dual Forward Pass for Asymmetric Causal Embeddings
//!
//! `gpu_forward_dual()` produces differentiated cause/effect embeddings
//! via two separate forward passes with different instruction prefixes:
//! - Cause: "search_query: Identify the cause in: {text}"
//! - Effect: "search_query: Identify the effect of: {text}"
//!
//! This leverages nomic-embed's contrastive training to naturally create
//! different vector representations for different instruction contexts.

mod attention;
mod encoder;
mod ops;

use candle_core::Tensor;
use tokenizers::Tokenizer;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::ModelId;

use super::config::CAUSAL_MAX_TOKENS;
use super::weights::NomicWeights;

use encoder::run_encoder;
pub use ops::layer_norm;
use ops::{l2_normalize, mean_pooling};

/// GPU-accelerated forward pass for NomicBERT.
///
/// Tokenizes input, computes embeddings (word + token_type + LayerNorm),
/// runs encoder layers with RoPE attention and SwiGLU FFN,
/// then mean-pools and L2-normalizes.
pub fn gpu_forward(
    text: &str,
    weights: &NomicWeights,
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

    // Truncate to max tokens
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

    // Token type IDs (all zeros for NomicBERT)
    let token_type_ids: Vec<u32> = vec![0u32; seq_len];
    let token_type_tensor =
        Tensor::from_slice(&token_type_ids, (1, seq_len), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("CausalModel token_type tensor failed: {}", e),
            }
        })?;

    // === EMBEDDING LAYER ===
    // NomicBERT: word + token_type + LayerNorm (no position embeddings — RoPE in attention)
    let embeddings = compute_embeddings(&input_ids, &token_type_tensor, weights, seq_len)?;

    // === ENCODER LAYERS ===
    let hidden_states = run_encoder(embeddings, &attention_mask_tensor, weights)?;

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

/// Compute embeddings: word + token_type + LayerNorm.
///
/// NomicBERT has no position embeddings — position information comes from
/// rotary position embeddings (RoPE) applied in the attention layer.
fn compute_embeddings(
    input_ids: &Tensor,
    token_type_tensor: &Tensor,
    weights: &NomicWeights,
    seq_len: usize,
) -> EmbeddingResult<Tensor> {
    let config = &weights.config;

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

    // Sum embeddings (no position embeddings for NomicBERT)
    let embeddings = word_embeds
        .add(&token_type_embeds)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CausalModel embedding add failed: {}", e),
        })?;

    // Apply LayerNorm to embeddings
    layer_norm(
        &embeddings,
        &weights.embeddings.layer_norm_weight,
        &weights.embeddings.layer_norm_bias,
        config.layer_norm_eps,
    )
}

// =============================================================================
// Dual Forward Pass for Asymmetric Causal Embeddings
// =============================================================================

/// GPU-accelerated dual forward pass for cause/effect embeddings.
///
/// Produces two distinct 768D vectors via two encoder passes with
/// different instruction prefixes. This leverages nomic-embed's contrastive
/// training where different instruction prefixes produce genuinely different
/// embedding spaces.
///
/// ```text
/// Input Text
///     |
///     +--------------------------------------+
///     |                                      |
/// "search_query: Identify cause..."    "search_query: Identify effect..."
///     |                                      |
/// [Tokenize + Embed + Encode]         [Tokenize + Embed + Encode]
///     |                                      |
/// [Mean Pool + L2 Normalize]          [Mean Pool + L2 Normalize]
///     |                                      |
/// cause_vec (768D)                    effect_vec (768D)
/// ```
pub fn gpu_forward_dual(
    text: &str,
    weights: &NomicWeights,
    tokenizer: &Tokenizer,
) -> EmbeddingResult<(Vec<f32>, Vec<f32>)> {
    let cause_text = format!("{}{}", super::config::CAUSE_INSTRUCTION, text);
    let effect_text = format!("{}{}", super::config::EFFECT_INSTRUCTION, text);

    let cause_vec = gpu_forward(&cause_text, weights, tokenizer)?;
    let effect_vec = gpu_forward(&effect_text, weights, tokenizer)?;

    Ok((cause_vec, effect_vec))
}
