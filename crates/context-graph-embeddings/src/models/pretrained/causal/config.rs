//! Longformer configuration and constants.
//!
//! This module contains the model configuration parsed from config.json
//! and the fixed constants for the Longformer/causal embedding model.

/// Native dimension for Longformer-base model.
pub const CAUSAL_DIMENSION: usize = 768;

/// Maximum tokens for Longformer (extended context).
pub const CAUSAL_MAX_TOKENS: usize = 4096;

/// Latency budget in milliseconds (P95 target).
pub const CAUSAL_LATENCY_BUDGET_MS: u32 = 8;

/// Default attention window size for sliding window attention.
pub const DEFAULT_ATTENTION_WINDOW: usize = 512;

/// Longformer configuration parsed from config.json.
#[derive(Debug, Clone)]
pub struct LongformerConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden layer size.
    pub hidden_size: usize,
    /// Number of hidden layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Intermediate FFN size.
    pub intermediate_size: usize,
    /// Maximum position embeddings.
    pub max_position_embeddings: usize,
    /// Token type vocabulary size.
    pub type_vocab_size: usize,
    /// Layer normalization epsilon.
    pub layer_norm_eps: f64,
}

impl Default for LongformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 4098,
            type_vocab_size: 1,
            layer_norm_eps: 1e-5,
        }
    }
}
