//! Weight structures for CodeT5+ model.
//!
//! Contains tensor weight definitions for T5-style attention,
//! feed-forward networks, and encoder layers.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::ModelId;

use super::config::CodeT5pConfig;

/// T5-style self-attention weights.
#[derive(Debug)]
pub struct T5AttentionWeights {
    /// Query projection: [d_model, d_model]
    pub q_weight: Tensor,
    /// Key projection: [d_model, d_model]
    pub k_weight: Tensor,
    /// Value projection: [d_model, d_model]
    pub v_weight: Tensor,
    /// Output projection: [d_model, d_model]
    pub o_weight: Tensor,
    /// Relative attention bias (only in first layer): [num_buckets, num_heads]
    pub relative_attention_bias: Option<Tensor>,
}

/// T5-style FFN weights (DenseReluDense).
#[derive(Debug)]
pub struct T5FfnWeights {
    /// Input projection: [d_ff, d_model]
    pub wi_weight: Tensor,
    /// Output projection: [d_model, d_ff]
    pub wo_weight: Tensor,
    /// Layer norm weight: [d_model]
    pub layer_norm_weight: Tensor,
}

/// T5-style encoder layer weights.
#[derive(Debug)]
pub struct T5EncoderLayerWeights {
    /// Self-attention weights.
    pub attention: T5AttentionWeights,
    /// Attention layer norm weight: [d_model]
    pub attention_layer_norm_weight: Tensor,
    /// FFN weights.
    pub ffn: T5FfnWeights,
}

/// Complete CodeT5p encoder weights.
#[derive(Debug)]
pub struct CodeT5pWeights {
    /// Model configuration.
    pub config: CodeT5pConfig,
    /// Shared embeddings: [vocab_size, d_model]
    pub shared_embeddings: Tensor,
    /// Encoder layers.
    pub encoder_layers: Vec<T5EncoderLayerWeights>,
    /// Final layer norm weight: [d_model]
    pub final_layer_norm_weight: Tensor,
    /// GPU device reference.
    pub device: &'static Device,
}

impl CodeT5pWeights {
    /// Load CodeT5p weights from safetensors file.
    pub fn from_path(model_path: &Path, device: &'static Device) -> EmbeddingResult<Self> {
        let safetensors_path = model_path.join("model.safetensors");
        if !safetensors_path.exists() {
            return Err(EmbeddingError::ModelLoadError {
                model_id: ModelId::Code,
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "model.safetensors not found at {}",
                        safetensors_path.display()
                    ),
                )),
            });
        }

        // Parse config.json for model dimensions
        let config = CodeT5pConfig::from_path(model_path)?;

        // Load safetensors
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&safetensors_path], DType::F32, device).map_err(
                |e| EmbeddingError::GpuError {
                    message: format!("CodeModel safetensors load failed: {}", e),
                },
            )?
        };

        // Load shared embeddings
        let shared_embeddings = vb
            .get((config.vocab_size, config.d_model), "shared.weight")
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("CodeModel shared embeddings load failed: {}", e),
            })?;

        // Load encoder layers
        let mut encoder_layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layer = Self::load_encoder_layer(&vb, &config, layer_idx)?;
            encoder_layers.push(layer);
        }

        // Load final layer norm
        let final_layer_norm_weight = vb
            .get((config.d_model,), "encoder.final_layer_norm.weight")
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("CodeModel final layer norm weight load failed: {}", e),
            })?;

        Ok(CodeT5pWeights {
            config,
            shared_embeddings,
            encoder_layers,
            final_layer_norm_weight,
            device,
        })
    }

    /// Load a single encoder layer.
    fn load_encoder_layer(
        vb: &VarBuilder,
        config: &CodeT5pConfig,
        layer_idx: usize,
    ) -> EmbeddingResult<T5EncoderLayerWeights> {
        let attention = Self::load_attention_weights(vb, config, layer_idx)?;
        let ffn = Self::load_ffn_weights(vb, config, layer_idx)?;

        let attention_layer_norm_weight = vb
            .get(
                (config.d_model,),
                &format!("encoder.block.{}.layer.0.layer_norm.weight", layer_idx),
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "CodeModel layer {} attention layer norm weight load failed: {}",
                    layer_idx, e
                ),
            })?;

        Ok(T5EncoderLayerWeights {
            attention,
            attention_layer_norm_weight,
            ffn,
        })
    }

    /// Load attention weights for a layer.
    fn load_attention_weights(
        vb: &VarBuilder,
        config: &CodeT5pConfig,
        layer_idx: usize,
    ) -> EmbeddingResult<T5AttentionWeights> {
        let prefix = format!("encoder.block.{}.layer.0.SelfAttention", layer_idx);

        let q_weight = vb
            .get(
                (config.d_model, config.d_model),
                &format!("{}.q.weight", prefix),
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("CodeModel layer {} q weight load failed: {}", layer_idx, e),
            })?;

        let k_weight = vb
            .get(
                (config.d_model, config.d_model),
                &format!("{}.k.weight", prefix),
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("CodeModel layer {} k weight load failed: {}", layer_idx, e),
            })?;

        let v_weight = vb
            .get(
                (config.d_model, config.d_model),
                &format!("{}.v.weight", prefix),
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("CodeModel layer {} v weight load failed: {}", layer_idx, e),
            })?;

        let o_weight = vb
            .get(
                (config.d_model, config.d_model),
                &format!("{}.o.weight", prefix),
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("CodeModel layer {} o weight load failed: {}", layer_idx, e),
            })?;

        // Relative attention bias is only in the first layer
        let relative_attention_bias = if layer_idx == 0 {
            Some(
                vb.get(
                    (config.relative_attention_num_buckets, config.num_heads),
                    &format!("{}.relative_attention_bias.weight", prefix),
                )
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("CodeModel relative attention bias load failed: {}", e),
                })?,
            )
        } else {
            None
        };

        Ok(T5AttentionWeights {
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            relative_attention_bias,
        })
    }

    /// Load FFN weights for a layer.
    fn load_ffn_weights(
        vb: &VarBuilder,
        config: &CodeT5pConfig,
        layer_idx: usize,
    ) -> EmbeddingResult<T5FfnWeights> {
        let prefix = format!("encoder.block.{}.layer.1", layer_idx);

        let wi_weight = vb
            .get(
                (config.d_ff, config.d_model),
                &format!("{}.DenseReluDense.wi.weight", prefix),
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("CodeModel layer {} wi weight load failed: {}", layer_idx, e),
            })?;

        let wo_weight = vb
            .get(
                (config.d_model, config.d_ff),
                &format!("{}.DenseReluDense.wo.weight", prefix),
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("CodeModel layer {} wo weight load failed: {}", layer_idx, e),
            })?;

        let layer_norm_weight = vb
            .get((config.d_model,), &format!("{}.layer_norm.weight", prefix))
            .map_err(|e| EmbeddingError::GpuError {
                message: format!(
                    "CodeModel layer {} FFN layer norm weight load failed: {}",
                    layer_idx, e
                ),
            })?;

        Ok(T5FfnWeights {
            wi_weight,
            wo_weight,
            layer_norm_weight,
        })
    }
}
