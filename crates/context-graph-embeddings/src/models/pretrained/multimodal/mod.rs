//! Multimodal embedding model using openai/clip-vit-large-patch14.
//!
//! This model (E10) produces 768D vectors for text-image pairs using CLIP
//! (Contrastive Language-Image Pretraining), enabling unified embeddings
//! across text and image modalities.
//!
//! # Dimension
//!
//! - Native output: 768D (CLIP shared embedding space, clip-vit-large-patch14)
//!
//! # GPU Acceleration
//!
//! When the `candle` feature is enabled, this model uses GPU-accelerated inference
//! via Candle with the following pipeline:
//!
//! ## Text Encoder
//! 1. Tokenization with CLIP BPE tokenizer (max 77 tokens)
//! 2. Word + position embeddings
//! 3. 12-layer transformer with causal attention mask
//! 4. Extract [EOS] token embedding
//! 5. Project to 768D and L2 normalize
//!
//! ## Vision Encoder
//! 1. Patch embedding: 14x14 patches -> 256 patch tokens
//! 2. Prepend [CLS] token, add position embeddings
//! 3. 24-layer Vision Transformer
//! 4. Extract [CLS] token embedding
//! 5. Project to 768D and L2 normalize
//!
//! # Thread Safety
//! - `AtomicBool` for `loaded` state (lock-free reads)
//! - `RwLock` for model state (thread-safe state transitions)

// Submodules - organized by functionality
mod attention;
mod config;
mod constants;
mod forward_ops;
mod forward_text;
mod forward_vision;
mod image_processor;
mod layer_norm;
mod mlp;
mod model;
mod trait_impl;
mod weight_loading;
mod weight_loading_text;
mod weight_loading_vision;
mod weights;

#[cfg(test)]
mod tests;

// Re-export everything for backwards compatibility

// Constants
pub use constants::{
    CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_NUM_PATCHES, CLIP_NUM_PATCHES_PER_DIM, CLIP_PATCH_SIZE,
    CLIP_STD, MULTIMODAL_DIMENSION, MULTIMODAL_LATENCY_BUDGET_MS, MULTIMODAL_MAX_TOKENS,
    MULTIMODAL_MODEL_NAME,
};

// Configuration types
pub use config::{ClipTextConfig, ClipVisionConfig};

// Weight structures
pub use weights::{
    ClipTextAttentionWeights, ClipTextLayerWeights, ClipTextMlpWeights, ClipTextWeights,
    ClipVisionAttentionWeights, ClipVisionLayerWeights, ClipVisionMlpWeights, ClipVisionWeights,
    ClipWeights,
};

// Image processor
pub use image_processor::ImageProcessor;

// Model
pub use model::MultimodalModel;
