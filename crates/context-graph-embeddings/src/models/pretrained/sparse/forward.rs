//! GPU forward pass implementation for the sparse SPLADE model.
//!
//! This module implements the main forward pass through the BERT encoder
//! and MLM head to produce sparse term importance scores.

use candle_core::Tensor;
use tokenizers::Tokenizer;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::BertWeights;
use crate::types::{InputType, ModelId, ModelInput};

use super::embeddings::{compute_embeddings, run_encoder};
use super::mlm_head::{apply_splade_activation, convert_to_sparse, run_mlm_head};
use super::types::{MlmHeadWeights, SparseVector, SPARSE_MAX_TOKENS};

/// Extract text content from model input.
pub(crate) fn extract_text(input: &ModelInput) -> EmbeddingResult<String> {
    match input {
        ModelInput::Text {
            content,
            instruction,
        } => {
            let mut full = content.clone();
            if let Some(inst) = instruction {
                full = format!("{} {}", inst, full);
            }
            Ok(full)
        }
        _ => Err(EmbeddingError::UnsupportedModality {
            model_id: ModelId::Sparse,
            input_type: InputType::from(input),
        }),
    }
}

/// Run GPU-accelerated SPLADE forward pass returning sparse vector.
pub(crate) fn gpu_forward_sparse(
    text: &str,
    weights: &BertWeights,
    tokenizer: &Tokenizer,
    mlm_head: &MlmHeadWeights,
) -> EmbeddingResult<SparseVector> {
    let device = weights.device();
    let config = &weights.config;

    // Tokenize input text
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| EmbeddingError::TokenizationError {
            model_id: ModelId::Sparse,
            message: format!("SparseModel tokenization failed: {}", e),
        })?;

    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<f32> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as f32)
        .collect();

    // Truncate to max_position_embeddings if needed
    let max_len = config.max_position_embeddings.min(SPARSE_MAX_TOKENS);
    let seq_len = token_ids.len().min(max_len);
    let token_ids = &token_ids[..seq_len];
    let attention_mask = &attention_mask[..seq_len];

    // Create GPU tensors
    let input_ids = Tensor::from_slice(token_ids, (1, seq_len), device).map_err(|e| {
        EmbeddingError::GpuError {
            message: format!("SparseModel input_ids tensor failed: {}", e),
        }
    })?;

    let attention_mask_tensor =
        Tensor::from_slice(attention_mask, (1, seq_len), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("SparseModel attention_mask tensor failed: {}", e),
            }
        })?;

    // Token type IDs (all zeros)
    let token_type_ids: Vec<u32> = vec![0u32; seq_len];
    let token_type_tensor =
        Tensor::from_slice(&token_type_ids, (1, seq_len), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("SparseModel token_type tensor failed: {}", e),
            }
        })?;

    // Position IDs
    let position_ids: Vec<u32> = (0..seq_len as u32).collect();
    let position_tensor = Tensor::from_slice(&position_ids, (1, seq_len), device).map_err(|e| {
        EmbeddingError::GpuError {
            message: format!("SparseModel position_ids tensor failed: {}", e),
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
    let hidden_states = run_encoder(embeddings, &attention_mask_tensor, weights, config)?;

    // === MLM HEAD ===
    let logits = run_mlm_head(&hidden_states, mlm_head, config)?;

    // === SPLADE ACTIVATION ===
    let sparse_vector = apply_splade_activation(logits, &attention_mask_tensor)?;

    // Convert to sparse format
    convert_to_sparse(sparse_vector)
}
