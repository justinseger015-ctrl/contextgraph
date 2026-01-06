//! GPU forward pass implementation for CodeT5+.
//!
//! Contains the main T5 encoder forward pass.

use candle_core::Tensor;
use tokenizers::Tokenizer;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::ModelId;

use super::constants::CODE_MAX_TOKENS;
use super::layers::{encoder_layer_forward, rms_norm};
use super::position::compute_position_bias;
use super::weights::CodeT5pWeights;

/// GPU-accelerated forward pass for CodeT5p.
///
/// T5 encoder uses:
/// - Relative position bias instead of absolute position embeddings
/// - RMSNorm instead of LayerNorm (no bias, mean-centered)
/// - Pre-norm architecture (norm before attention/FFN)
pub fn gpu_forward(
    text: &str,
    weights: &CodeT5pWeights,
    tokenizer: &Tokenizer,
) -> EmbeddingResult<Vec<f32>> {
    let device = weights.device;
    let config = &weights.config;

    // Tokenize input text
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| EmbeddingError::TokenizationError {
            model_id: ModelId::Code,
            message: format!("CodeModel tokenization failed: {}", e),
        })?;

    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<f32> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as f32)
        .collect();

    // Truncate to max tokens if needed
    let seq_len = token_ids.len().min(CODE_MAX_TOKENS);
    let token_ids = &token_ids[..seq_len];
    let attention_mask = &attention_mask[..seq_len];

    // Create GPU tensors
    let input_ids = Tensor::from_slice(token_ids, (1, seq_len), device).map_err(|e| {
        EmbeddingError::GpuError {
            message: format!("CodeModel input_ids tensor failed: {}", e),
        }
    })?;

    let attention_mask_tensor =
        Tensor::from_slice(attention_mask, (1, seq_len), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("CodeModel attention_mask tensor failed: {}", e),
            }
        })?;

    // === EMBEDDING LAYER ===
    let hidden_states = embed_tokens(&input_ids, weights, seq_len)?;

    // === COMPUTE RELATIVE POSITION BIAS ===
    let position_bias = compute_position_bias(
        weights.encoder_layers[0]
            .attention
            .relative_attention_bias
            .as_ref()
            .unwrap(),
        seq_len,
        config,
        device,
    )?;

    // Create extended attention mask for broadcasting
    let extended_attention_mask = create_extended_attention_mask(&attention_mask_tensor)?;

    // === ENCODER LAYERS ===
    let mut hidden_states = hidden_states;
    for (layer_idx, layer) in weights.encoder_layers.iter().enumerate() {
        hidden_states = encoder_layer_forward(
            &hidden_states,
            layer,
            &extended_attention_mask,
            &position_bias,
            config,
            layer_idx,
        )?;
    }

    // === FINAL LAYER NORM ===
    hidden_states = rms_norm(
        &hidden_states,
        &weights.final_layer_norm_weight,
        config.layer_norm_epsilon,
    )?;

    // === POOLING & NORMALIZE ===
    let pooled = mean_pool(&hidden_states, &attention_mask_tensor)?;
    l2_normalize(&pooled)
}

/// Embed token IDs using shared embeddings.
fn embed_tokens(
    input_ids: &Tensor,
    weights: &CodeT5pWeights,
    seq_len: usize,
) -> EmbeddingResult<Tensor> {
    weights
        .shared_embeddings
        .index_select(
            &input_ids
                .flatten_all()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("CodeModel flatten input_ids failed: {}", e),
                })?,
            0,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel embedding lookup failed: {}", e),
        })?
        .reshape((1, seq_len, weights.config.d_model))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel embedding reshape failed: {}", e),
        })
}

/// Create extended attention mask for broadcasting.
fn create_extended_attention_mask(attention_mask: &Tensor) -> EmbeddingResult<Tensor> {
    let extended = attention_mask
        .unsqueeze(1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel attention mask unsqueeze 1 failed: {}", e),
        })?
        .unsqueeze(2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel attention mask unsqueeze 2 failed: {}", e),
        })?;

    let ones = Tensor::ones_like(&extended).map_err(|e| EmbeddingError::GpuError {
        message: format!("CodeModel create ones tensor failed: {}", e),
    })?;

    let inverted = ones
        .broadcast_sub(&extended)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel attention mask invert failed: {}", e),
        })?;

    (inverted * (-10000.0f64)).map_err(|e| EmbeddingError::GpuError {
        message: format!("CodeModel attention mask scale failed: {}", e),
    })
}

/// Mean pooling over sequence dimension.
fn mean_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> EmbeddingResult<Tensor> {
    let mask_expanded = attention_mask
        .unsqueeze(2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel mask expand failed: {}", e),
        })?
        .broadcast_as(hidden_states.shape())
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel mask broadcast failed: {}", e),
        })?;

    let masked = (hidden_states.clone() * mask_expanded).map_err(|e| EmbeddingError::GpuError {
        message: format!("CodeModel mask apply failed: {}", e),
    })?;

    let sum = masked.sum(1).map_err(|e| EmbeddingError::GpuError {
        message: format!("CodeModel sum pooling failed: {}", e),
    })?;

    let count = attention_mask
        .sum_all()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel mask sum failed: {}", e),
        })?;

    sum.broadcast_div(&count)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel mean div failed: {}", e),
        })
}

/// L2 normalize and convert to Vec<f32>.
fn l2_normalize(tensor: &Tensor) -> EmbeddingResult<Vec<f32>> {
    let norm = tensor
        .sqr()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel sqr failed: {}", e),
        })?
        .sum_keepdim(1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel norm sum failed: {}", e),
        })?
        .sqrt()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel sqrt failed: {}", e),
        })?;

    let eps = Tensor::ones_like(&norm)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel create eps ones failed: {}", e),
        })?
        .affine(1e-12, 0.0)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel eps scale failed: {}", e),
        })?;

    let normalized = tensor
        .broadcast_div(
            &norm
                .broadcast_add(&eps)
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("CodeModel norm eps add failed: {}", e),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel normalize div failed: {}", e),
        })?;

    normalized
        .flatten_all()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel flatten output failed: {}", e),
        })?
        .to_vec1()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel to_vec1 failed: {}", e),
        })
}
