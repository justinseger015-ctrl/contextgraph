//! Weight loading functions for Longformer model.
//!
//! This module handles loading model weights from safetensors files
//! and parsing configuration from config.json.

use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::ModelId;

use super::config::LongformerConfig;
use super::embeddings_loader::load_embeddings;
use super::encoder_loader::load_encoder_layer;
use super::weights::LongformerWeights;

/// Load Longformer weights from safetensors file.
pub fn load_longformer_weights(
    model_path: &Path,
    device: &'static Device,
) -> EmbeddingResult<LongformerWeights> {
    let safetensors_path = model_path.join("model.safetensors");
    if !safetensors_path.exists() {
        return Err(EmbeddingError::ModelLoadError {
            model_id: ModelId::Causal,
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
    let config = load_config(model_path)?;

    // Load safetensors
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&safetensors_path], DType::F32, device).map_err(
            |e| EmbeddingError::GpuError {
                message: format!("CausalModel safetensors load failed: {}", e),
            },
        )?
    };

    // Load embeddings
    let embeddings = load_embeddings(&vb, &config)?;

    // Load encoder layers
    let mut encoder_layers = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        let layer = load_encoder_layer(&vb, &config, layer_idx)?;
        encoder_layers.push(layer);
    }

    Ok(LongformerWeights {
        config,
        embeddings,
        encoder_layers,
        device,
    })
}

/// Load config.json
pub fn load_config(model_path: &Path) -> EmbeddingResult<LongformerConfig> {
    let config_path = model_path.join("config.json");
    let config_content =
        std::fs::read_to_string(&config_path).map_err(|e| EmbeddingError::ModelLoadError {
            model_id: ModelId::Causal,
            source: Box::new(e),
        })?;

    #[derive(serde::Deserialize)]
    struct RawConfig {
        vocab_size: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        max_position_embeddings: usize,
        #[serde(default = "default_type_vocab")]
        type_vocab_size: usize,
        #[serde(default = "default_layer_norm_eps")]
        layer_norm_eps: f64,
    }

    fn default_type_vocab() -> usize {
        1
    }
    fn default_layer_norm_eps() -> f64 {
        1e-5
    }

    let raw: RawConfig =
        serde_json::from_str(&config_content).map_err(|e| EmbeddingError::ConfigError {
            message: format!("CausalModel config parse failed: {}", e),
        })?;

    Ok(LongformerConfig {
        vocab_size: raw.vocab_size,
        hidden_size: raw.hidden_size,
        num_hidden_layers: raw.num_hidden_layers,
        num_attention_heads: raw.num_attention_heads,
        intermediate_size: raw.intermediate_size,
        max_position_embeddings: raw.max_position_embeddings,
        type_vocab_size: raw.type_vocab_size,
        layer_norm_eps: raw.layer_norm_eps,
    })
}
