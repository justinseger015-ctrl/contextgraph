//! Storage module for embeddings.
//!
//! NOTE: The FusedEmbedding binary storage system has been removed.
//! This module is currently a placeholder for future embedding storage needs.
//!
//! Previously contained:
//! - `binary`: GDS-compatible binary codec (REMOVED - depended on FusedEmbedding)
//! - `batch`: Batch encoder for multi-embedding files (REMOVED - depended on FusedEmbedding)
//! - `gds`: GDS file reader (REMOVED - depended on FusedEmbedding)
//!
//! Future storage implementations should use SemanticFingerprint or JohariFingerprint
//! from the context-graph-teleology crate.
