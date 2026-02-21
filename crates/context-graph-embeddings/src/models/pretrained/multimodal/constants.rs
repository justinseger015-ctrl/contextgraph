//! Constants for the multimodal CLIP embedding model.
//!
//! This module contains all configuration constants for the CLIP model,
//! including dimensions, token limits, image parameters, and normalization values.

/// Native dimension for CLIP embeddings (shared text-image space).
/// Note: Matches ModelId::Contextual.dimension() for clip-vit-large-patch14.
pub const MULTIMODAL_DIMENSION: usize = 768;

/// Maximum tokens for CLIP text encoder.
pub const MULTIMODAL_MAX_TOKENS: usize = 77;

/// Latency budget in milliseconds (P95 target).
/// Note: Matches ModelId::Contextual.latency_budget_ms() from constitution.yaml.
pub const MULTIMODAL_LATENCY_BUDGET_MS: u64 = 15;

/// HuggingFace model repository name.
pub const MULTIMODAL_MODEL_NAME: &str = "openai/clip-vit-large-patch14";

/// CLIP image input size (224x224 pixels).
pub const CLIP_IMAGE_SIZE: u32 = 224;

/// CLIP RGB normalization mean values.
pub const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP RGB normalization standard deviation values.
#[allow(clippy::excessive_precision)]
pub const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// CLIP patch size for Vision Transformer.
pub const CLIP_PATCH_SIZE: usize = 14;

/// Number of patches per dimension (224 / 14 = 16).
pub const CLIP_NUM_PATCHES_PER_DIM: usize = CLIP_IMAGE_SIZE as usize / CLIP_PATCH_SIZE;

/// Total number of patches (16 * 16 = 256).
pub const CLIP_NUM_PATCHES: usize = CLIP_NUM_PATCHES_PER_DIM * CLIP_NUM_PATCHES_PER_DIM;
