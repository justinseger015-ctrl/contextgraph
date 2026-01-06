//! Self-attention implementation for CodeT5+.
//!
//! Contains the multi-head self-attention forward pass.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

use super::config::CodeT5pConfig;
use super::weights::T5AttentionWeights;

/// Run self-attention forward pass.
pub fn self_attention_forward(
    hidden_states: &Tensor,
    attention: &T5AttentionWeights,
    attention_mask: &Tensor,
    position_bias: &Tensor,
    config: &CodeT5pConfig,
    layer_idx: usize,
) -> EmbeddingResult<Tensor> {
    let (batch_size, seq_len, hidden_size) =
        hidden_states
            .dims3()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("CodeModel layer {} get dims failed: {}", layer_idx, e),
            })?;

    let head_dim = config.d_kv;
    let num_heads = config.num_heads;

    // Flatten to [batch*seq, hidden] for Candle matmul compatibility
    let hidden_flat = hidden_states
        .reshape((batch_size * seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel layer {} flatten hidden failed: {}", layer_idx, e),
        })?;

    // Q, K, V projections (no bias in T5)
    let query = project_qkv(
        &hidden_flat,
        &attention.q_weight,
        batch_size,
        seq_len,
        hidden_size,
        layer_idx,
        "Q",
    )?;
    let key = project_qkv(
        &hidden_flat,
        &attention.k_weight,
        batch_size,
        seq_len,
        hidden_size,
        layer_idx,
        "K",
    )?;
    let value = project_qkv(
        &hidden_flat,
        &attention.v_weight,
        batch_size,
        seq_len,
        hidden_size,
        layer_idx,
        "V",
    )?;

    // Reshape to [batch, heads, seq_len, head_dim]
    let query = reshape_for_attention(
        &query, batch_size, seq_len, num_heads, head_dim, layer_idx, "Q",
    )?;
    let key = reshape_for_attention(
        &key, batch_size, seq_len, num_heads, head_dim, layer_idx, "K",
    )?;
    let value = reshape_for_attention(
        &value, batch_size, seq_len, num_heads, head_dim, layer_idx, "V",
    )?;

    // K^T with contiguous() for matmul
    let key_t = key
        .transpose(2, 3)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} K transpose 2,3 failed: {}",
                layer_idx, e
            ),
        })?
        .contiguous()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel layer {} K^T contiguous failed: {}", layer_idx, e),
        })?;

    // Attention scores: [batch, heads, seq_len, seq_len]
    let scores = query.matmul(&key_t).map_err(|e| EmbeddingError::GpuError {
        message: format!("CodeModel layer {} QK matmul failed: {}", layer_idx, e),
    })?;

    // Add position bias and attention mask
    let scores = scores
        .broadcast_add(position_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} position bias add failed: {}",
                layer_idx, e
            ),
        })?
        .broadcast_add(attention_mask)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} attention mask add failed: {}",
                layer_idx, e
            ),
        })?;

    // Softmax
    let attention_probs =
        candle_nn::ops::softmax(&scores, candle_core::D::Minus1).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("CodeModel layer {} softmax failed: {}", layer_idx, e),
            }
        })?;

    // Apply attention to values
    let context = attention_probs
        .matmul(&value)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel layer {} context matmul failed: {}", layer_idx, e),
        })?;

    // Reshape back: [batch, seq_len, d_model]
    let context = context
        .transpose(1, 2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} context transpose failed: {}",
                layer_idx, e
            ),
        })?
        .contiguous()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} context contiguous failed: {}",
                layer_idx, e
            ),
        })?
        .reshape((batch_size, seq_len, num_heads * head_dim))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} context reshape failed: {}",
                layer_idx, e
            ),
        })?;

    // Output projection
    let context_flat = context
        .reshape((batch_size * seq_len, num_heads * head_dim))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel layer {} O flatten failed: {}", layer_idx, e),
        })?;

    context_flat
        .matmul(
            &attention
                .o_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("CodeModel layer {} O transpose failed: {}", layer_idx, e),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel layer {} O matmul failed: {}", layer_idx, e),
        })?
        .reshape((batch_size, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel layer {} O reshape failed: {}", layer_idx, e),
        })
}

/// Project Q, K, or V.
fn project_qkv(
    hidden_flat: &Tensor,
    weight: &Tensor,
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    layer_idx: usize,
    name: &str,
) -> EmbeddingResult<Tensor> {
    hidden_flat
        .matmul(&weight.t().map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} {} transpose failed: {}",
                layer_idx, name, e
            ),
        })?)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} {} matmul failed: {}",
                layer_idx, name, e
            ),
        })?
        .reshape((batch_size, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} {} reshape failed: {}",
                layer_idx, name, e
            ),
        })
}

/// Reshape for multi-head attention.
fn reshape_for_attention(
    tensor: &Tensor,
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    layer_idx: usize,
    name: &str,
) -> EmbeddingResult<Tensor> {
    tensor
        .reshape((batch_size, seq_len, num_heads, head_dim))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} {} head reshape failed: {}",
                layer_idx, name, e
            ),
        })?
        .transpose(1, 2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} {} transpose 1,2 failed: {}",
                layer_idx, name, e
            ),
        })?
        .contiguous()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "CodeModel layer {} {} contiguous failed: {}",
                layer_idx, name, e
            ),
        })
}
