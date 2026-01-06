//! Entity embedding model using sentence-transformers/all-MiniLM-L6-v2.
//!
//! This model (E11) produces 384D vectors optimized for named entity embeddings
//! and TransE-style knowledge graph operations where `head + relation = tail`.
//!
//! # GPU Acceleration
//!
//! When the `candle` feature is enabled, this model uses GPU-accelerated BERT inference
//! via Candle with the following pipeline:
//! 1. Tokenization with HuggingFace tokenizers
//! 2. GPU embedding lookup and position encoding
//! 3. GPU-accelerated transformer forward pass (6 layers)
//! 4. Mean pooling over sequence dimension
//! 5. L2 normalization on GPU
//!
//! # Dimension
//!
//! - Native output: 384D (final dimension, no projection needed)
//!
//! Similar to GraphModel, EntityModel uses 384D directly as this is the native
//! MiniLM embedding dimension.
//!
//! # Thread Safety
//! - `AtomicBool` for `loaded` state (lock-free reads)
//! - `RwLock` for model state (thread-safe state transitions)
//!
//! # Memory Layout
//! - Total estimated: ~80MB for FP32 weights (22M parameters)
//! - With FP16 quantization: ~40MB
//!
//! # TransE Operations
//!
//! The model provides static methods for TransE-style knowledge graph operations:
//! - `transe_score(h, r, t)` - Compute TransE score: -||h + r - t||_2
//! - `predict_tail(h, r)` - Predict tail embedding: t_hat = h + r
//! - `predict_relation(h, t)` - Predict relation embedding: r_hat = t - h

mod attention;
mod attention_projection;
mod attention_scores;
mod encoding;
mod ffn;
mod forward;
mod layernorm;
mod model;
mod pooling;
mod trait_impl;
mod transe;
mod types;

#[cfg(test)]
mod tests;

// Re-export public API for backwards compatibility
pub use types::{
    EntityModel, ENTITY_DIMENSION, ENTITY_LATENCY_BUDGET_MS, ENTITY_MAX_TOKENS, ENTITY_MODEL_NAME,
};
