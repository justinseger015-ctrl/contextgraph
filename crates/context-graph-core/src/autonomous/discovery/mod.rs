//! Goal Discovery Pipeline for autonomous goal emergence.
//!
//! This module implements TASK-LOGIC-009: Goal Discovery Pipeline using K-means
//! clustering on TeleologicalArrays (13-embedder vectors) to discover emergent goals.
//!
//! # Architecture
//!
//! From constitution.yaml:
//! - ARCH-01: TeleologicalArray is atomic - never split, compare only as unit
//! - ARCH-02: Apples-to-apples - E1 compares with E1, never cross-embedder
//! - ARCH-03: Autonomous operation - goals emerge from data patterns
//!
//! # Design Philosophy
//!
//! FAIL FAST. NO FALLBACKS. NO RECOVERY ATTEMPTS.
//!
//! All errors are FATAL:
//! - InsufficientData -> panic with clear message
//! - ClusteringFailed -> panic with algorithm details
//! - NoClustersFound -> panic with threshold info
//! - InvalidCentroid -> panic with embedder state
//!
//! # Module Structure
//!
//! - `types` - Configuration, cluster, and result types
//! - `centroid` - Centroid computation for teleological arrays
//! - `clustering` - K-means clustering implementation
//! - `pipeline` - Main discovery pipeline orchestration
//! - `tests` - Comprehensive test suite

mod centroid;
mod clustering;
mod pipeline;
#[cfg(test)]
mod tests;
mod types;

// Re-export all public types for backwards compatibility
pub use centroid::{
    average_dense_vectors, average_sparse_vectors, average_token_vectors, compute_centroid,
    create_zeroed_fingerprint, l2_norm,
};
pub use clustering::{cluster_arrays, compute_cluster_coherence};
pub use pipeline::GoalDiscoveryPipeline;
pub use types::{
    Cluster, ClusteringAlgorithm, DiscoveredGoal, DiscoveryConfig, DiscoveryResult, GoalCandidate,
    GoalRelationship, NumClusters,
};
