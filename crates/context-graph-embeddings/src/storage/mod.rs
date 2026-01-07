//! Storage types for quantized embeddings.
//!
//! This module provides types for storing and indexing quantized embeddings
//! as specified in the Constitution's storage architecture.
//!
//! # Key Types
//!
//! - `StoredQuantizedFingerprint`: Complete fingerprint with quantized embeddings (~17KB)
//! - `IndexEntry`: Entry in per-embedder HNSW index (dequantized for search)
//! - `EmbedderQueryResult`: Result from single embedder search
//! - `MultiSpaceQueryResult`: RRF-fused result from multi-space retrieval
//! - `MultiSpaceSearchEngine`: Stage 3 multi-space search with RRF fusion
//!
//! # Relationship to Other Types
//!
//! - `TeleologicalFingerprint` (context-graph-core): ~63KB unquantized, used for computation
//! - `StoredQuantizedFingerprint` (this module): ~17KB quantized, used for storage
//!
//! The conversion between these types happens in the Logic Layer (TASK-EMB-022).

mod types;
pub mod multi_space;

#[cfg(test)]
mod full_state_verification;

pub use types::{
    // Constants
    EXPECTED_QUANTIZED_SIZE_BYTES,
    MAX_QUANTIZED_SIZE_BYTES,
    MIN_QUANTIZED_SIZE_BYTES,
    NUM_EMBEDDERS,
    STORAGE_VERSION,
    RRF_K,
    // Types
    StoredQuantizedFingerprint,
    IndexEntry,
    EmbedderQueryResult,
    MultiSpaceQueryResult,
};

pub use multi_space::{
    MultiSpaceSearchEngine,
    QuantizedFingerprintRetriever,
    MultiSpaceIndexProvider,
};
