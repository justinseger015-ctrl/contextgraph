//! CLIP model weight structures.
//!
//! Contains all weight tensor structures for the text and vision encoders.

use candle_core::{Device, Tensor};

use super::config::{ClipTextConfig, ClipVisionConfig};

// ============================================================================
// Text Encoder Weight Structures
// ============================================================================

/// Text encoder attention weights for a single layer.
#[derive(Debug)]
pub struct ClipTextAttentionWeights {
    /// Q projection: [hidden_size, hidden_size]
    pub q_proj_weight: Tensor,
    pub q_proj_bias: Tensor,
    /// K projection: [hidden_size, hidden_size]
    pub k_proj_weight: Tensor,
    pub k_proj_bias: Tensor,
    /// V projection: [hidden_size, hidden_size]
    pub v_proj_weight: Tensor,
    pub v_proj_bias: Tensor,
    /// Output projection: [hidden_size, hidden_size]
    pub out_proj_weight: Tensor,
    pub out_proj_bias: Tensor,
}

/// Text encoder MLP weights for a single layer.
#[derive(Debug)]
pub struct ClipTextMlpWeights {
    /// FC1: [hidden_size, intermediate_size]
    pub fc1_weight: Tensor,
    pub fc1_bias: Tensor,
    /// FC2: [intermediate_size, hidden_size]
    pub fc2_weight: Tensor,
    pub fc2_bias: Tensor,
}

/// Text encoder layer weights.
#[derive(Debug)]
pub struct ClipTextLayerWeights {
    /// Self-attention weights.
    pub attention: ClipTextAttentionWeights,
    /// MLP weights.
    pub mlp: ClipTextMlpWeights,
    /// Layer norm before attention.
    pub layer_norm1_weight: Tensor,
    pub layer_norm1_bias: Tensor,
    /// Layer norm before MLP.
    pub layer_norm2_weight: Tensor,
    pub layer_norm2_bias: Tensor,
}

/// Complete text encoder weights.
#[derive(Debug)]
pub struct ClipTextWeights {
    /// Configuration.
    pub config: ClipTextConfig,
    /// Word embeddings: [vocab_size, hidden_size]
    pub token_embedding: Tensor,
    /// Position embeddings: [max_position, hidden_size]
    pub position_embedding: Tensor,
    /// Encoder layers.
    pub layers: Vec<ClipTextLayerWeights>,
    /// Final layer norm.
    pub final_layer_norm_weight: Tensor,
    pub final_layer_norm_bias: Tensor,
    /// Text projection: [hidden_size, projection_dim]
    pub text_projection: Tensor,
}

// ============================================================================
// Vision Encoder Weight Structures
// ============================================================================

/// Vision encoder attention weights for a single layer.
#[derive(Debug)]
pub struct ClipVisionAttentionWeights {
    /// Q projection: [hidden_size, hidden_size]
    pub q_proj_weight: Tensor,
    pub q_proj_bias: Tensor,
    /// K projection: [hidden_size, hidden_size]
    pub k_proj_weight: Tensor,
    pub k_proj_bias: Tensor,
    /// V projection: [hidden_size, hidden_size]
    pub v_proj_weight: Tensor,
    pub v_proj_bias: Tensor,
    /// Output projection: [hidden_size, hidden_size]
    pub out_proj_weight: Tensor,
    pub out_proj_bias: Tensor,
}

/// Vision encoder MLP weights for a single layer.
#[derive(Debug)]
pub struct ClipVisionMlpWeights {
    /// FC1: [hidden_size, intermediate_size]
    pub fc1_weight: Tensor,
    pub fc1_bias: Tensor,
    /// FC2: [intermediate_size, hidden_size]
    pub fc2_weight: Tensor,
    pub fc2_bias: Tensor,
}

/// Vision encoder layer weights.
#[derive(Debug)]
pub struct ClipVisionLayerWeights {
    /// Self-attention weights.
    pub attention: ClipVisionAttentionWeights,
    /// MLP weights.
    pub mlp: ClipVisionMlpWeights,
    /// Layer norm before attention.
    pub layer_norm1_weight: Tensor,
    pub layer_norm1_bias: Tensor,
    /// Layer norm before MLP.
    pub layer_norm2_weight: Tensor,
    pub layer_norm2_bias: Tensor,
}

/// Complete vision encoder weights.
#[derive(Debug)]
pub struct ClipVisionWeights {
    /// Configuration.
    pub config: ClipVisionConfig,
    /// Patch embedding convolution: [hidden_size, 3, patch_size, patch_size]
    pub patch_embedding_weight: Tensor,
    /// Class token: [1, 1, hidden_size]
    pub class_embedding: Tensor,
    /// Position embeddings: [1, num_patches+1, hidden_size]
    pub position_embedding: Tensor,
    /// Pre-LayerNorm (before transformer).
    pub pre_layernorm_weight: Tensor,
    pub pre_layernorm_bias: Tensor,
    /// Encoder layers.
    pub layers: Vec<ClipVisionLayerWeights>,
    /// Post-LayerNorm (after transformer).
    pub post_layernorm_weight: Tensor,
    pub post_layernorm_bias: Tensor,
    /// Visual projection: [hidden_size, projection_dim]
    pub visual_projection: Tensor,
}

// ============================================================================
// Complete Model Weights
// ============================================================================

/// Complete CLIP model weights.
#[derive(Debug)]
pub struct ClipWeights {
    /// Text encoder weights.
    pub text: ClipTextWeights,
    /// Vision encoder weights.
    pub vision: ClipVisionWeights,
    /// Device reference.
    device: &'static Device,
}

impl ClipWeights {
    /// Create new ClipWeights.
    pub fn new(text: ClipTextWeights, vision: ClipVisionWeights, device: &'static Device) -> Self {
        Self {
            text,
            vision,
            device,
        }
    }

    /// Get the device these weights are loaded on.
    pub fn device(&self) -> &'static Device {
        self.device
    }

    /// Get total parameter count.
    pub fn param_count(&self) -> usize {
        let text_params = self.text.token_embedding.elem_count()
            + self.text.position_embedding.elem_count()
            + self.text.final_layer_norm_weight.elem_count()
            + self.text.final_layer_norm_bias.elem_count()
            + self.text.text_projection.elem_count()
            + self
                .text
                .layers
                .iter()
                .map(|l| {
                    l.attention.q_proj_weight.elem_count()
                        + l.attention.q_proj_bias.elem_count()
                        + l.attention.k_proj_weight.elem_count()
                        + l.attention.k_proj_bias.elem_count()
                        + l.attention.v_proj_weight.elem_count()
                        + l.attention.v_proj_bias.elem_count()
                        + l.attention.out_proj_weight.elem_count()
                        + l.attention.out_proj_bias.elem_count()
                        + l.mlp.fc1_weight.elem_count()
                        + l.mlp.fc1_bias.elem_count()
                        + l.mlp.fc2_weight.elem_count()
                        + l.mlp.fc2_bias.elem_count()
                        + l.layer_norm1_weight.elem_count()
                        + l.layer_norm1_bias.elem_count()
                        + l.layer_norm2_weight.elem_count()
                        + l.layer_norm2_bias.elem_count()
                })
                .sum::<usize>();

        let vision_params = self.vision.patch_embedding_weight.elem_count()
            + self.vision.class_embedding.elem_count()
            + self.vision.position_embedding.elem_count()
            + self.vision.pre_layernorm_weight.elem_count()
            + self.vision.pre_layernorm_bias.elem_count()
            + self.vision.post_layernorm_weight.elem_count()
            + self.vision.post_layernorm_bias.elem_count()
            + self.vision.visual_projection.elem_count()
            + self
                .vision
                .layers
                .iter()
                .map(|l| {
                    l.attention.q_proj_weight.elem_count()
                        + l.attention.q_proj_bias.elem_count()
                        + l.attention.k_proj_weight.elem_count()
                        + l.attention.k_proj_bias.elem_count()
                        + l.attention.v_proj_weight.elem_count()
                        + l.attention.v_proj_bias.elem_count()
                        + l.attention.out_proj_weight.elem_count()
                        + l.attention.out_proj_bias.elem_count()
                        + l.mlp.fc1_weight.elem_count()
                        + l.mlp.fc1_bias.elem_count()
                        + l.mlp.fc2_weight.elem_count()
                        + l.mlp.fc2_bias.elem_count()
                        + l.layer_norm1_weight.elem_count()
                        + l.layer_norm1_bias.elem_count()
                        + l.layer_norm2_weight.elem_count()
                        + l.layer_norm2_bias.elem_count()
                })
                .sum::<usize>();

        text_params + vision_params
    }

    /// Get estimated VRAM usage in bytes (F32).
    pub fn vram_bytes(&self) -> usize {
        self.param_count() * std::mem::size_of::<f32>()
    }
}
