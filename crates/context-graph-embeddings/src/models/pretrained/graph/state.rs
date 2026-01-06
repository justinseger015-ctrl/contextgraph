//! Internal state management for GraphModel.
//!
//! Manages the loaded/unloaded state of model weights and tokenizer.

use crate::gpu::BertWeights;
use tokenizers::Tokenizer;

/// Internal state that varies based on whether the model is loaded.
#[allow(dead_code)]
pub enum ModelState {
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
