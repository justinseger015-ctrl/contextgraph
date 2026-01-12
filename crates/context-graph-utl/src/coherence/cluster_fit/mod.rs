//! ClusterFit types for silhouette-based coherence component.
//!
//! # Constitution Reference
//!
//! Per constitution.yaml line 166:
//! ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
//!
//! ClusterFit measures how well a vertex fits within its semantic cluster
//! using the silhouette coefficient: s = (b - a) / max(a, b)
//!
//! # Output Range
//!
//! All outputs are clamped per AP-10 (no NaN/Infinity):
//! - `ClusterFitResult.score`: [0, 1]
//! - `ClusterFitResult.silhouette`: [-1, 1]
//!
//! # Module Structure
//!
//! - `types`: Configuration, context, and result types
//! - `distance`: Distance computation functions
//! - `compute`: Main `compute_cluster_fit` function
//! - `tests`: Comprehensive test suite

pub mod compute;
pub mod distance;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use self::compute::compute_cluster_fit;
pub use self::types::{ClusterContext, ClusterFitConfig, ClusterFitResult, DistanceMetric};
