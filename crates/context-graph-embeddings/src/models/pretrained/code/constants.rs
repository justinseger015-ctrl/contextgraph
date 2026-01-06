//! Constants for the CodeT5+ embedding model.
//!
//! These constants define the architectural parameters for the
//! Salesforce/codet5p-110m-embedding model.

/// Native dimension for CodeT5p embedding output.
pub const CODE_NATIVE_DIMENSION: usize = 256;

/// Projected dimension (internal d_model) for multi-array storage compatibility.
pub const CODE_PROJECTED_DIMENSION: usize = 768;

/// Maximum tokens for CodeT5p (standard BERT-family limit).
pub const CODE_MAX_TOKENS: usize = 512;

/// Latency budget in milliseconds (P95 target).
pub const CODE_LATENCY_BUDGET_MS: u32 = 10;

/// HuggingFace model repository name.
pub const CODE_MODEL_NAME: &str = "Salesforce/codet5p-110m-embedding";
