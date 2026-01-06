//! Mean pooling and L2 normalization for BERT outputs.
//!
//! Implements attention-masked mean pooling followed by L2 normalization
//! for sentence embeddings.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::{normalize_gpu, BertConfig};

use super::EntityModel;

impl EntityModel {
    /// Mean pool hidden states using attention mask.
    ///
    /// # Arguments
    /// * `hidden_states` - Encoder output [batch, seq_len, hidden_size]
    /// * `attention_mask` - Token mask [batch, seq_len]
    /// * `config` - Model config for dimensions
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Pooled tensor [batch, hidden_size]
    pub(crate) fn mean_pool(
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        config: &BertConfig,
        seq_len: usize,
    ) -> EmbeddingResult<Tensor> {
        // Expand mask to hidden dimension: [batch, seq_len, 1] -> [batch, seq_len, hidden_size]
        let mask_expanded = attention_mask
            .unsqueeze(2)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel mask expand failed: {}", e),
            })?
            .broadcast_as((1, seq_len, config.hidden_size))
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel mask broadcast failed: {}", e),
            })?;

        // Mask hidden states
        let masked_hidden =
            (hidden_states * mask_expanded).map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel masked multiply failed: {}", e),
            })?;

        // Sum over sequence dimension
        let sum_hidden = masked_hidden.sum(1).map_err(|e| EmbeddingError::GpuError {
            message: format!("EntityModel sum hidden failed: {}", e),
        })?;

        // Compute sum of mask for averaging
        let mask_sum = attention_mask
            .sum(1)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel mask sum failed: {}", e),
            })?
            .unsqueeze(1)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel mask sum unsqueeze failed: {}", e),
            })?
            .broadcast_as(sum_hidden.shape())
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel mask sum broadcast failed: {}", e),
            })?;

        // Divide by mask sum (add eps for stability)
        let pooled = (sum_hidden
            / (mask_sum + 1e-9f64).map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel mask sum add eps failed: {}", e),
            })?)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("EntityModel mean pooling div failed: {}", e),
        })?;

        Ok(pooled)
    }

    /// L2 normalize pooled embeddings.
    ///
    /// # Arguments
    /// * `pooled` - Pooled tensor [batch, hidden_size]
    ///
    /// # Returns
    /// L2-normalized tensor
    pub(crate) fn l2_normalize(pooled: &Tensor) -> EmbeddingResult<Tensor> {
        normalize_gpu(pooled).map_err(|e| EmbeddingError::GpuError {
            message: format!("EntityModel L2 normalize failed: {}", e),
        })
    }

    /// Convert normalized tensor to Vec<f32>.
    ///
    /// # Arguments
    /// * `normalized` - Normalized tensor
    ///
    /// # Returns
    /// Flattened embedding as Vec<f32>
    pub(crate) fn tensor_to_vec(normalized: &Tensor) -> EmbeddingResult<Vec<f32>> {
        normalized
            .flatten_all()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel flatten output failed: {}", e),
            })?
            .to_vec1()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("EntityModel to_vec1 failed: {}", e),
            })
    }
}
