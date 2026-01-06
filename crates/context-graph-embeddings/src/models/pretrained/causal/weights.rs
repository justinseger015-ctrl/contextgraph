//! Longformer weight structures.
//!
//! This module contains the tensor weight structures for embedding layers,
//! attention mechanisms, feed-forward networks, and complete model weights.

use candle_core::{Device, Tensor};

use super::config::LongformerConfig;

/// Longformer embedding weights.
#[derive(Debug)]
pub struct LongformerEmbeddingWeights {
    /// Word embeddings: [vocab_size, hidden_size]
    pub word_embeddings: Tensor,
    /// Position embeddings: [max_position, hidden_size]
    pub position_embeddings: Tensor,
    /// Token type embeddings: [type_vocab_size, hidden_size]
    pub token_type_embeddings: Tensor,
    /// LayerNorm weight: [hidden_size]
    pub layer_norm_weight: Tensor,
    /// LayerNorm bias: [hidden_size]
    pub layer_norm_bias: Tensor,
}

/// Longformer self-attention weights (includes global attention).
#[derive(Debug)]
pub struct LongformerAttentionWeights {
    /// Query projection: [hidden_size, hidden_size]
    pub query_weight: Tensor,
    /// Query bias: [hidden_size]
    pub query_bias: Tensor,
    /// Key projection: [hidden_size, hidden_size]
    pub key_weight: Tensor,
    /// Key bias: [hidden_size]
    pub key_bias: Tensor,
    /// Value projection: [hidden_size, hidden_size]
    pub value_weight: Tensor,
    /// Value bias: [hidden_size]
    pub value_bias: Tensor,
    /// Output projection: [hidden_size, hidden_size]
    pub output_weight: Tensor,
    /// Output bias: [hidden_size]
    pub output_bias: Tensor,
    /// Attention output LayerNorm weight: [hidden_size]
    pub layer_norm_weight: Tensor,
    /// Attention output LayerNorm bias: [hidden_size]
    pub layer_norm_bias: Tensor,
    // Note: Global attention weights omitted for simplicity in initial implementation.
    // For full sliding window + global attention, we would include:
    // query_global_weight, key_global_weight, value_global_weight, etc.
}

/// Longformer FFN weights.
#[derive(Debug)]
pub struct LongformerFfnWeights {
    /// Intermediate projection: [intermediate_size, hidden_size]
    pub intermediate_weight: Tensor,
    /// Intermediate bias: [intermediate_size]
    pub intermediate_bias: Tensor,
    /// Output projection: [hidden_size, intermediate_size]
    pub output_weight: Tensor,
    /// Output bias: [hidden_size]
    pub output_bias: Tensor,
    /// Output LayerNorm weight: [hidden_size]
    pub layer_norm_weight: Tensor,
    /// Output LayerNorm bias: [hidden_size]
    pub layer_norm_bias: Tensor,
}

/// Longformer encoder layer weights.
#[derive(Debug)]
pub struct LongformerEncoderLayerWeights {
    /// Self-attention weights.
    pub attention: LongformerAttentionWeights,
    /// FFN weights.
    pub ffn: LongformerFfnWeights,
}

/// Complete Longformer model weights.
#[derive(Debug)]
pub struct LongformerWeights {
    /// Model configuration.
    pub config: LongformerConfig,
    /// Embedding layer weights.
    pub embeddings: LongformerEmbeddingWeights,
    /// Encoder layer weights.
    pub encoder_layers: Vec<LongformerEncoderLayerWeights>,
    /// GPU device reference.
    pub(crate) device: &'static Device,
}
