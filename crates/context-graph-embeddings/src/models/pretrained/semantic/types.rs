//! Type definitions for the semantic embedding model.
//!
//! Contains the ModelState enum and SemanticModel struct definition.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use crate::gpu::BertWeights;
use crate::traits::SingleModelConfig;
use tokenizers::Tokenizer;

/// Internal state that varies based on feature flags.
#[allow(dead_code)]
pub(crate) enum ModelState {
    /// Unloaded - no weights in memory.
    Unloaded,

    /// Loaded with candle model and tokenizer (GPU-accelerated).
    Loaded {
        /// BERT model weights on GPU (boxed to reduce enum size).
        weights: Box<BertWeights>,
        /// HuggingFace tokenizer for text encoding (boxed to reduce enum size).
        tokenizer: Box<Tokenizer>,
    },
}

/// Semantic embedding model using intfloat/e5-large-v2.
///
/// This is the primary semantic understanding model producing 1024D dense vectors.
/// Uses instruction prefixes to distinguish between queries and passages.
///
/// # Thread Safety
/// - `AtomicBool` for `loaded` state (lock-free reads)
/// - Inner model/tokenizer require explicit synchronization if mutable
///
/// # Memory Layout
/// - Total estimated: 1.3GB for FP32 weights
/// - With FP16 quantization: ~650MB
pub struct SemanticModel {
    /// Model weights and inference engine.
    /// NOTE: Type depends on candle feature flag.
    /// For now, placeholder that compiles without candle.
    #[allow(dead_code)]
    pub(crate) model_state: std::sync::RwLock<ModelState>,

    /// Path to model weights directory.
    #[allow(dead_code)]
    pub(crate) model_path: PathBuf,

    /// Configuration for this model instance.
    #[allow(dead_code)]
    pub(crate) config: SingleModelConfig,

    /// Whether model weights are loaded and ready.
    pub(crate) loaded: AtomicBool,

    /// Memory used by model weights (bytes).
    #[allow(dead_code)]
    pub(crate) memory_size: usize,
}

// Implement Send and Sync manually since RwLock is involved
unsafe impl Send for SemanticModel {}
unsafe impl Sync for SemanticModel {}
