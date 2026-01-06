//! Projection and output conversion for ColBERT.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

use super::types::{ColBertProjection, TokenEmbeddings};

/// Project from 768D to 128D and L2 normalize each token.
pub(crate) fn project_and_normalize(
    hidden_states: Tensor,
    projection: &ColBertProjection,
) -> EmbeddingResult<Tensor> {
    let (batch_size, seq_len_proj, hidden_size_proj) =
        hidden_states
            .dims3()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("LateInteractionModel projection get dims failed: {}", e),
            })?;

    let proj_dim = projection
        .weight
        .dim(0)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel get proj_dim failed: {}", e),
        })?;

    // Flatten to [batch*seq, hidden] for Candle matmul compatibility
    let hidden_flat = hidden_states
        .reshape((batch_size * seq_len_proj, hidden_size_proj))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel projection flatten failed: {}", e),
        })?;

    let projected = hidden_flat
        .matmul(
            &projection
                .weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("LateInteractionModel projection transpose failed: {}", e),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel projection matmul failed: {}", e),
        })?
        .reshape((batch_size, seq_len_proj, proj_dim))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel projection reshape failed: {}", e),
        })?;

    // L2 normalize each token
    let norm = projected
        .sqr()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel sqr failed: {}", e),
        })?
        .sum_keepdim(candle_core::D::Minus1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel sum norm failed: {}", e),
        })?
        .sqrt()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel sqrt norm failed: {}", e),
        })?;

    projected
        .broadcast_div(&(norm + 1e-9f64).map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel norm add eps failed: {}", e),
        })?)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel normalize failed: {}", e),
        })
}

/// Convert normalized tensor to TokenEmbeddings.
pub(crate) fn convert_to_token_embeddings(
    normalized: Tensor,
    token_strings: Vec<String>,
    attention_mask: &[f32],
    seq_len: usize,
) -> EmbeddingResult<TokenEmbeddings> {
    let normalized_2d = normalized
        .squeeze(0)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel squeeze failed: {}", e),
        })?;

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let token_vec: Vec<f32> = normalized_2d
            .get(i)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("LateInteractionModel get token {} failed: {}", i, e),
            })?
            .to_vec1()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("LateInteractionModel to_vec1 token {} failed: {}", i, e),
            })?;
        vectors.push(token_vec);
    }

    let mask: Vec<bool> = attention_mask.iter().map(|&m| m > 0.5).collect();
    TokenEmbeddings::new(vectors, token_strings, mask)
}
