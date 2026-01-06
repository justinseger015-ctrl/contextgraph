//! CLIP model configuration types.
//!
//! Contains configuration structures for the text and vision encoders
//! used in the CLIP multimodal model.

/// CLIP text encoder configuration.
#[derive(Debug, Clone)]
pub struct ClipTextConfig {
    /// Vocabulary size (49408 for CLIP).
    pub vocab_size: usize,
    /// Hidden layer size (768 for clip-vit-large-patch14 text encoder).
    pub hidden_size: usize,
    /// Number of hidden layers (12 for text encoder).
    pub num_hidden_layers: usize,
    /// Number of attention heads (12 for text encoder).
    pub num_attention_heads: usize,
    /// Intermediate FFN size (3072 for text encoder).
    pub intermediate_size: usize,
    /// Maximum sequence length (77 tokens).
    pub max_position_embeddings: usize,
    /// Layer normalization epsilon.
    pub layer_norm_eps: f64,
    /// Projection dimension (768).
    pub projection_dim: usize,
}

impl Default for ClipTextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            projection_dim: 768,
        }
    }
}

/// CLIP vision encoder configuration.
#[derive(Debug, Clone)]
pub struct ClipVisionConfig {
    /// Hidden layer size (1024 for clip-vit-large-patch14 vision encoder).
    pub hidden_size: usize,
    /// Number of hidden layers (24 for vision encoder).
    pub num_hidden_layers: usize,
    /// Number of attention heads (16 for vision encoder).
    pub num_attention_heads: usize,
    /// Intermediate FFN size (4096 for vision encoder).
    pub intermediate_size: usize,
    /// Image size (224x224).
    pub image_size: usize,
    /// Patch size (14x14).
    pub patch_size: usize,
    /// Layer normalization epsilon.
    pub layer_norm_eps: f64,
    /// Projection dimension (768).
    pub projection_dim: usize,
}

impl Default for ClipVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            image_size: 224,
            patch_size: 14,
            layer_norm_eps: 1e-5,
            projection_dim: 768,
        }
    }
}
