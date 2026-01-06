//! Types and constants for the late-interaction (ColBERT) embedding model.

use candle_core::Tensor;
use tokenizers::Tokenizer;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::BertWeights;

/// Native dimension for ColBERT per-token embeddings.
pub const LATE_INTERACTION_DIMENSION: usize = 128;

/// Maximum tokens for ColBERT (standard BERT-family limit).
pub const LATE_INTERACTION_MAX_TOKENS: usize = 512;

/// Latency budget in milliseconds (P95 target).
pub const LATE_INTERACTION_LATENCY_BUDGET_MS: u64 = 8;

/// HuggingFace model repository name.
pub const LATE_INTERACTION_MODEL_NAME: &str = "colbert-ir/colbertv2.0";

/// Per-token embeddings from ColBERT.
///
/// Each token in the input produces a 128D embedding vector.
/// The mask indicates which tokens are valid (non-padding).
#[derive(Debug, Clone)]
pub struct TokenEmbeddings {
    /// Token vectors [num_tokens, 128] - each inner Vec is 128D
    pub vectors: Vec<Vec<f32>>,
    /// Token strings for debugging/analysis
    pub tokens: Vec<String>,
    /// Mask for valid tokens (excludes padding)
    pub mask: Vec<bool>,
}

impl TokenEmbeddings {
    /// Create new token embeddings with validation.
    ///
    /// # Errors
    /// - `EmbeddingError::InvalidDimension` if any vector is not 128D
    /// - `EmbeddingError::EmptyInput` if vectors is empty
    pub fn new(
        vectors: Vec<Vec<f32>>,
        tokens: Vec<String>,
        mask: Vec<bool>,
    ) -> EmbeddingResult<Self> {
        if vectors.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Validate all vectors are 128D
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != LATE_INTERACTION_DIMENSION {
                tracing::error!(
                    "Token {} has invalid dimension: expected {}, got {}",
                    i,
                    LATE_INTERACTION_DIMENSION,
                    vec.len()
                );
                return Err(EmbeddingError::InvalidDimension {
                    expected: LATE_INTERACTION_DIMENSION,
                    actual: vec.len(),
                });
            }
        }

        // Validate lengths match
        if vectors.len() != tokens.len() || vectors.len() != mask.len() {
            tracing::error!(
                "Length mismatch: vectors={}, tokens={}, mask={}",
                vectors.len(),
                tokens.len(),
                mask.len()
            );
            return Err(EmbeddingError::InvalidDimension {
                expected: vectors.len(),
                actual: tokens.len().min(mask.len()),
            });
        }

        Ok(Self {
            vectors,
            tokens,
            mask,
        })
    }

    /// Count of valid (non-padding) tokens.
    pub fn valid_token_count(&self) -> usize {
        self.mask.iter().filter(|&&v| v).count()
    }
}

/// ColBERT projection layer weights for 768D -> 128D.
#[derive(Debug)]
pub struct ColBertProjection {
    /// Linear projection weight: [128, 768]
    pub weight: Tensor,
}

/// Internal state for model weights.
#[allow(dead_code)]
pub(crate) enum ModelState {
    /// Unloaded - no weights in memory.
    Unloaded,

    /// Loaded with candle model and tokenizer (GPU-accelerated).
    Loaded {
        /// BERT model weights on GPU (boxed to reduce enum size).
        weights: Box<BertWeights>,
        /// ColBERT projection layer (768D -> 128D).
        projection: ColBertProjection,
        /// HuggingFace tokenizer for text encoding (boxed to reduce enum size).
        tokenizer: Box<Tokenizer>,
    },
}
