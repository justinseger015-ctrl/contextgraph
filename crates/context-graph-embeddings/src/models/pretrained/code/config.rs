//! Configuration types for CodeT5+ model.

use std::path::Path;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::ModelId;

/// CodeT5p configuration parsed from config.json.
#[derive(Debug, Clone)]
pub struct CodeT5pConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden layer size (d_model).
    pub d_model: usize,
    /// Embedding dimension (output).
    pub embed_dim: usize,
    /// Key-value dimension.
    pub d_kv: usize,
    /// FFN dimension.
    pub d_ff: usize,
    /// Number of encoder layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of relative attention buckets.
    pub relative_attention_num_buckets: usize,
    /// Maximum distance for relative attention.
    pub relative_attention_max_distance: usize,
    /// Layer norm epsilon.
    pub layer_norm_epsilon: f64,
}

impl Default for CodeT5pConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32103,
            d_model: 768,
            embed_dim: 256,
            d_kv: 64,
            d_ff: 3072,
            num_layers: 12,
            num_heads: 12,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_epsilon: 1e-6,
        }
    }
}

impl CodeT5pConfig {
    /// Load config from JSON file.
    pub fn from_path(model_path: &Path) -> EmbeddingResult<Self> {
        let config_path = model_path.join("config.json");
        let config_content =
            std::fs::read_to_string(&config_path).map_err(|e| EmbeddingError::ModelLoadError {
                model_id: ModelId::Code,
                source: Box::new(e),
            })?;

        #[derive(serde::Deserialize)]
        struct RawConfig {
            vocab_size: usize,
            d_model: usize,
            embed_dim: usize,
            d_kv: usize,
            d_ff: usize,
            num_layers: usize,
            num_heads: usize,
            #[serde(default = "default_rel_buckets")]
            relative_attention_num_buckets: usize,
            #[serde(default = "default_rel_max_dist")]
            relative_attention_max_distance: usize,
            #[serde(default = "default_layer_norm_eps")]
            layer_norm_epsilon: f64,
        }

        fn default_rel_buckets() -> usize {
            32
        }
        fn default_rel_max_dist() -> usize {
            128
        }
        fn default_layer_norm_eps() -> f64 {
            1e-6
        }

        let raw: RawConfig =
            serde_json::from_str(&config_content).map_err(|e| EmbeddingError::ConfigError {
                message: format!("CodeModel config parse failed: {}", e),
            })?;

        Ok(CodeT5pConfig {
            vocab_size: raw.vocab_size,
            d_model: raw.d_model,
            embed_dim: raw.embed_dim,
            d_kv: raw.d_kv,
            d_ff: raw.d_ff,
            num_layers: raw.num_layers,
            num_heads: raw.num_heads,
            relative_attention_num_buckets: raw.relative_attention_num_buckets,
            relative_attention_max_distance: raw.relative_attention_max_distance,
            layer_norm_epsilon: raw.layer_norm_epsilon,
        })
    }
}
