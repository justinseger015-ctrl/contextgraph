//! Weight loading functions for CLIP model.
//!
//! Contains functions to load CLIP model weights from safetensors files
//! into the weight structures for both text and vision encoders.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::error::{EmbeddingError, EmbeddingResult};

use super::config::{ClipTextConfig, ClipVisionConfig};
use super::model::MultimodalModel;
use super::weights::ClipWeights;

impl MultimodalModel {
    /// Load CLIP weights from VarBuilder.
    pub(crate) fn load_clip_weights(
        vb: &VarBuilder,
        device: &'static Device,
    ) -> EmbeddingResult<ClipWeights> {
        let text_config = ClipTextConfig::default();
        let vision_config = ClipVisionConfig::default();

        // Load text encoder weights
        let text = Self::load_text_weights(vb, &text_config)?;

        // Load vision encoder weights
        let vision = Self::load_vision_weights(vb, &vision_config)?;

        Ok(ClipWeights::new(text, vision, device))
    }

    /// Get a tensor from VarBuilder with shape validation.
    pub(crate) fn get_tensor(
        vb: &VarBuilder,
        name: &str,
        expected_shape: &[usize],
    ) -> EmbeddingResult<Tensor> {
        vb.get(expected_shape, name)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Failed to load weight '{}': {}", name, e),
            })
    }
}
