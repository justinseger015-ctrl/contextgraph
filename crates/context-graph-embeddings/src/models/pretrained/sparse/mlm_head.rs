//! MLM head and SPLADE activation for the sparse model.
//!
//! This module implements the Masked Language Model head that projects
//! BERT hidden states to vocabulary logits, followed by the SPLADE
//! activation function for sparse term importance scoring.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

use super::encoder::layer_norm;
use super::types::{MlmHeadWeights, SparseVector};

/// Run through MLM head.
pub(crate) fn run_mlm_head(
    hidden_states: &Tensor,
    mlm_head: &MlmHeadWeights,
    config: &crate::gpu::BertConfig,
) -> EmbeddingResult<Tensor> {
    let (batch_size, seq_len, hidden_size) =
        hidden_states
            .dims3()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("SparseModel MLM get dims failed: {}", e),
            })?;

    // Flatten to [batch*seq, hidden] for Candle matmul compatibility
    let hidden_flat = hidden_states
        .reshape((batch_size * seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM flatten failed: {}", e),
        })?;

    // Dense transform with flatten/reshape pattern
    let mlm_hidden = hidden_flat
        .matmul(
            &mlm_head
                .dense_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("SparseModel MLM dense transpose failed: {}", e),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM dense matmul failed: {}", e),
        })?
        .reshape((batch_size, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM dense reshape failed: {}", e),
        })?;

    let mlm_hidden = mlm_hidden
        .broadcast_add(&mlm_head.dense_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM dense bias failed: {}", e),
        })?;

    // GELU activation
    let mlm_hidden = mlm_hidden.gelu().map_err(|e| EmbeddingError::GpuError {
        message: format!("SparseModel MLM GELU failed: {}", e),
    })?;

    // LayerNorm
    let mlm_hidden = layer_norm(
        &mlm_hidden,
        &mlm_head.layer_norm_weight,
        &mlm_head.layer_norm_bias,
        config.layer_norm_eps,
    )?;

    // Get vocab size from decoder weight
    let vocab_size = mlm_head
        .decoder_weight
        .dim(0)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM get vocab_size failed: {}", e),
        })?;

    // Flatten for decoder projection
    let mlm_flat = mlm_hidden
        .reshape((batch_size * seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM flatten for decoder failed: {}", e),
        })?;

    // Project to vocabulary with flatten/reshape pattern
    let logits = mlm_flat
        .matmul(
            &mlm_head
                .decoder_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("SparseModel MLM decoder transpose failed: {}", e),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM decoder matmul failed: {}", e),
        })?
        .reshape((batch_size, seq_len, vocab_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM decoder reshape failed: {}", e),
        })?;

    logits
        .broadcast_add(&mlm_head.decoder_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM decoder bias failed: {}", e),
        })
}

/// Apply SPLADE activation: log(1 + ReLU(x)) with max pooling.
pub(crate) fn apply_splade_activation(
    logits: Tensor,
    attention_mask_tensor: &Tensor,
) -> EmbeddingResult<Tensor> {
    // log(1 + ReLU(x))
    let splade_scores = logits.relu().map_err(|e| EmbeddingError::GpuError {
        message: format!("SparseModel ReLU failed: {}", e),
    })?;

    let splade_scores = (splade_scores + 1.0)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel add 1.0 failed: {}", e),
        })?
        .log()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel log failed: {}", e),
        })?;

    // Apply attention mask
    let mask_expanded = attention_mask_tensor
        .unsqueeze(2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel mask expand failed: {}", e),
        })?
        .broadcast_as(splade_scores.shape())
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel mask broadcast failed: {}", e),
        })?;

    let splade_scores = (splade_scores * mask_expanded).map_err(|e| EmbeddingError::GpuError {
        message: format!("SparseModel mask apply failed: {}", e),
    })?;

    // Max pooling over sequence dimension
    splade_scores.max(1).map_err(|e| EmbeddingError::GpuError {
        message: format!("SparseModel max pooling failed: {}", e),
    })
}

/// Convert dense tensor to sparse vector format.
pub(crate) fn convert_to_sparse(sparse_vector: Tensor) -> EmbeddingResult<SparseVector> {
    let sparse_dense: Vec<f32> = sparse_vector
        .flatten_all()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel flatten output failed: {}", e),
        })?
        .to_vec1()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel to_vec1 failed: {}", e),
        })?;

    // Extract non-zero indices and weights
    let threshold = 0.01f32;
    let (indices, weights): (Vec<usize>, Vec<f32>) = sparse_dense
        .iter()
        .enumerate()
        .filter(|(_, &w)| w > threshold)
        .map(|(i, &w)| (i, w))
        .unzip();

    Ok(SparseVector::new(indices, weights))
}
