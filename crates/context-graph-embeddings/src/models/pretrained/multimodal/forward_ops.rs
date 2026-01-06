//! Common forward pass operations for CLIP.
//!
//! Contains shared operations used by both text and vision encoders:
//! - Self-attention mechanism (via attention module)
//! - MLP block with QuickGELU (via mlp module)
//! - Layer normalization (via layer_norm module)
//! - Causal mask creation
//! - Patch embedding convolution

use candle_core::{Device, Tensor};

use crate::error::{EmbeddingError, EmbeddingResult};

// Re-export from submodules for backwards compatibility
pub use super::attention::self_attention;
pub use super::layer_norm::{layer_norm, layer_norm_1d};
pub use super::mlp::mlp;

/// Create causal attention mask for text encoder.
pub fn create_causal_mask(seq_len: usize, device: &Device) -> EmbeddingResult<Tensor> {
    // Create lower triangular mask: 0 for attend, -inf for mask
    let mut mask_data = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask_data[i * seq_len + j] = 0.0;
        }
    }

    Tensor::from_slice(&mask_data, (1, 1, seq_len, seq_len), device).map_err(|e| {
        EmbeddingError::GpuError {
            message: format!("Causal mask creation failed: {}", e),
        }
    })
}

/// 2D convolution for patch embedding (stride = kernel_size).
pub fn conv2d_patch_embed(
    image: &Tensor,
    weight: &Tensor,
    patch_size: usize,
) -> EmbeddingResult<Tensor> {
    // Use Candle's conv2d with stride = patch_size
    // weight shape: [out_channels, in_channels, kH, kW]
    image
        .conv2d(weight, 0, patch_size, 1, 1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("Conv2D patch embedding failed: {}", e),
        })
}
