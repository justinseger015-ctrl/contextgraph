//! Main computation function for ClusterFit.
//!
//! Contains the `compute_cluster_fit` function which computes the silhouette
//! coefficient for a query embedding.

use super::distance::{magnitude, mean_distance_to_cluster};
use super::types::{ClusterContext, ClusterFitConfig, ClusterFitResult, DistanceMetric};

/// Compute silhouette coefficient for a query embedding.
///
/// The silhouette coefficient measures how well a point fits within its assigned
/// cluster compared to the nearest other cluster:
///
/// ```text
/// silhouette = (b - a) / max(a, b)
/// ```
///
/// Where:
/// - `a` = mean intra-cluster distance (distance to same-cluster members)
/// - `b` = mean nearest-cluster distance (distance to nearest other cluster)
///
/// # Constitution Reference
///
/// Per constitution.yaml line 166:
/// - ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
/// - Output range [0, 1] (normalized from silhouette [-1, 1])
/// - No NaN/Infinity (AP-10)
///
/// # Arguments
///
/// * `query` - The embedding vector to evaluate
/// * `context` - Cluster context containing same-cluster and nearest-cluster members
/// * `config` - Configuration for the calculation
///
/// # Returns
///
/// `ClusterFitResult` containing:
/// - `score`: Normalized score [0, 1] for use in UTL formula
/// - `silhouette`: Raw silhouette coefficient [-1, 1]
/// - `intra_distance`: Mean distance to same-cluster members (a)
/// - `inter_distance`: Mean distance to nearest-cluster members (b)
///
/// # Edge Cases
///
/// Returns fallback result when:
/// - Query is empty or zero-magnitude
/// - Same-cluster has fewer than `min_cluster_size - 1` valid members
/// - Nearest-cluster is empty
/// - All distances are zero or invalid
///
/// # Example
///
/// ```ignore
/// use context_graph_utl::coherence::{
///     compute_cluster_fit, ClusterContext, ClusterFitConfig, DistanceMetric,
/// };
///
/// let query = vec![0.1, 0.2, 0.3, 0.4];
/// let same_cluster = vec![
///     vec![0.12, 0.22, 0.28, 0.38],
///     vec![0.11, 0.21, 0.29, 0.39],
/// ];
/// let nearest_cluster = vec![
///     vec![0.8, 0.1, 0.05, 0.05],
///     vec![0.7, 0.2, 0.05, 0.05],
/// ];
///
/// let context = ClusterContext::new(same_cluster, nearest_cluster);
/// let config = ClusterFitConfig::default();
///
/// let result = compute_cluster_fit(&query, &context, &config);
/// assert!(result.score >= 0.0 && result.score <= 1.0);
/// assert!(result.silhouette >= -1.0 && result.silhouette <= 1.0);
/// ```
pub fn compute_cluster_fit(
    query: &[f32],
    context: &ClusterContext,
    config: &ClusterFitConfig,
) -> ClusterFitResult {
    // Edge case: empty query
    if query.is_empty() {
        return ClusterFitResult::fallback(config.fallback_value);
    }

    // Edge case: zero-magnitude query (for cosine distance)
    if config.distance_metric == DistanceMetric::Cosine && magnitude(query) < 1e-10 {
        return ClusterFitResult::fallback(config.fallback_value);
    }

    // Check minimum cluster size requirement
    // We need at least min_cluster_size - 1 other members (query is one member)
    let min_required = config.min_cluster_size.saturating_sub(1);
    if context.same_cluster.len() < min_required {
        return ClusterFitResult::fallback(config.fallback_value);
    }

    // Edge case: empty nearest cluster
    if context.nearest_cluster.is_empty() {
        return ClusterFitResult::fallback(config.fallback_value);
    }

    // Compute intra-cluster distance (a)
    let intra_distance = match mean_distance_to_cluster(
        query,
        &context.same_cluster,
        config.distance_metric,
        config.max_sample_size,
    ) {
        Some(d) => d,
        None => return ClusterFitResult::fallback(config.fallback_value),
    };

    // Compute inter-cluster distance (b) - distance to nearest cluster
    let inter_distance = match mean_distance_to_cluster(
        query,
        &context.nearest_cluster,
        config.distance_metric,
        config.max_sample_size,
    ) {
        Some(d) => d,
        None => return ClusterFitResult::fallback(config.fallback_value),
    };

    // Compute silhouette coefficient: s = (b - a) / max(a, b)
    let max_dist = intra_distance.max(inter_distance);

    let silhouette = if max_dist < 1e-10 {
        // Both distances are effectively zero - neutral result
        0.0
    } else {
        let s = (inter_distance - intra_distance) / max_dist;
        // Ensure no NaN/Infinity per AP-10
        if s.is_nan() || s.is_infinite() {
            0.0
        } else {
            s.clamp(-1.0, 1.0)
        }
    };

    ClusterFitResult::new(silhouette, intra_distance, inter_distance)
}
