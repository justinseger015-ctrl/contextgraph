//! Code embedding model using Salesforce/codet5p-110m-embedding.
//!
//! This model (E7) produces 256D native vectors optimized for code understanding.
//! Uses SentencePiece BPE tokenization for processing source code.
//!
//! # Dimension Projection
//!
//! - Native output: 256D (embed_dim)
//! - Internal representation: 768D (d_model)
//! - Projected output: 768D (for multi-array storage compatibility)
//!
//! The projection from 256D to 768D is learned during training.
//!
//! # Thread Safety
//! - `AtomicBool` for `loaded` state (lock-free reads)
//! - Inner model/tokenizer require explicit synchronization if mutable
//!
//! # Memory Layout
//! - Total estimated: ~440MB for FP32 weights (110M parameters)
//! - With FP16 quantization: ~220MB

mod attention;
mod config;
mod constants;
mod forward;
mod layers;
mod model;
mod position;
mod weights;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_batch;
#[cfg(test)]
mod tests_edge_cases;

// Re-export used public types
pub use constants::{
    CODE_LATENCY_BUDGET_MS, CODE_MAX_TOKENS, CODE_MODEL_NAME, CODE_NATIVE_DIMENSION,
    CODE_PROJECTED_DIMENSION,
};
pub use model::CodeModel;
