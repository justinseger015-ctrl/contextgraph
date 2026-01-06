//! Constants for the GraphModel (sentence-transformers/paraphrase-MiniLM-L6-v2).
//!
//! These constants define the model's architecture and operational parameters.

/// Native dimension for MiniLM graph embeddings (384D, no projection).
pub const GRAPH_DIMENSION: usize = 384;

/// Maximum tokens for MiniLM (standard BERT-family limit).
pub const GRAPH_MAX_TOKENS: usize = 512;

/// Latency budget in milliseconds (P95 target).
pub const GRAPH_LATENCY_BUDGET_MS: u64 = 5;

/// HuggingFace model repository name.
pub const GRAPH_MODEL_NAME: &str = "sentence-transformers/paraphrase-MiniLM-L6-v2";

/// Maximum number of neighbor context entries for encode_context.
pub const MAX_CONTEXT_NEIGHBORS: usize = 5;
