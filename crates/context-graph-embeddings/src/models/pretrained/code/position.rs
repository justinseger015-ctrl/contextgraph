//! Relative position bias computation for T5 attention.

use candle_core::{Device, Tensor};

use crate::error::{EmbeddingError, EmbeddingResult};

use super::config::CodeT5pConfig;

/// Compute relative position bias for T5 attention.
pub fn compute_position_bias(
    relative_attention_bias: &Tensor,
    seq_len: usize,
    config: &CodeT5pConfig,
    device: &Device,
) -> EmbeddingResult<Tensor> {
    let mut relative_positions = vec![0i64; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let rel_pos = (j as i64) - (i as i64);
            let bucket = relative_position_bucket(
                rel_pos,
                config.relative_attention_num_buckets as i64,
                config.relative_attention_max_distance as i64,
                false,
            );
            relative_positions[i * seq_len + j] = bucket;
        }
    }

    let position_indices = Tensor::from_slice(&relative_positions, (seq_len, seq_len), device)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel position indices tensor failed: {}", e),
        })?;

    relative_attention_bias
        .index_select(
            &position_indices
                .flatten_all()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("CodeModel flatten positions failed: {}", e),
                })?,
            0,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel bias lookup failed: {}", e),
        })?
        .reshape((seq_len, seq_len, config.num_heads))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel bias reshape failed: {}", e),
        })?
        .permute((2, 0, 1))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel bias permute failed: {}", e),
        })?
        .unsqueeze(0)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel bias unsqueeze failed: {}", e),
        })
}

/// Compute bucket for relative position (T5 logarithmic buckets).
fn relative_position_bucket(
    relative_position: i64,
    num_buckets: i64,
    max_distance: i64,
    bidirectional: bool,
) -> i64 {
    let mut ret = 0i64;
    let mut n = -relative_position;

    let num_buckets = if bidirectional {
        num_buckets / 2
    } else {
        num_buckets
    };

    if bidirectional {
        ret += if n < 0 { num_buckets } else { 0 };
        n = n.abs();
    } else {
        n = n.max(0);
    }

    let max_exact = num_buckets / 2;
    let is_small = n < max_exact;

    if is_small {
        ret + n
    } else {
        let val_if_large = max_exact as f64
            + (((n as f64 / max_exact as f64).ln()
                / (max_distance as f64 / max_exact as f64).ln())
                * (num_buckets - max_exact) as f64);
        ret + val_if_large.min((num_buckets - 1) as f64) as i64
    }
}
