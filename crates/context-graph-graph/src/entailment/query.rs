//! Entailment query operations for O(1) IS-A hierarchy checks.
//!
//! This module provides query functions that use EntailmentCone containment
//! to efficiently determine hierarchical relationships between concepts.
//!
//! # Algorithm Overview
//!
//! 1. Use BFS from query node to generate candidate nodes
//! 2. For each candidate, check cone containment with O(1) angle check
//! 3. Filter and rank by membership score
//!
//! # Performance Targets
//!
//! - Single containment check: <1ms (O(1) angle computation)
//! - Batch check (1000 pairs): <100ms
//! - BFS + filter (depth 3): <10ms
//!
//! # Constitution Reference
//!
//! - perf.latency.entailment_check: <1ms
//! - M04-T20: Implement entailment query operations

use std::collections::{HashSet, VecDeque};

use crate::config::HyperbolicConfig;
use crate::entailment::cones::EntailmentCone;
use crate::error::{GraphError, GraphResult};
use crate::hyperbolic::poincare::PoincarePoint;
use crate::hyperbolic::PoincareBall;
use crate::storage::GraphStorage;
use crate::storage::storage_impl::{
    EntailmentCone as StorageEntailmentCone, PoincarePoint as StoragePoincarePoint,
};

// ========== Type Conversion Helpers ==========

/// Convert storage PoincarePoint to hyperbolic PoincarePoint.
fn storage_to_hyperbolic_point(storage_point: &StoragePoincarePoint) -> PoincarePoint {
    PoincarePoint::from_coords(storage_point.coords)
}

/// Convert storage EntailmentCone to entailment EntailmentCone.
fn storage_to_entailment_cone(storage_cone: &StorageEntailmentCone) -> GraphResult<EntailmentCone> {
    let apex = storage_to_hyperbolic_point(&storage_cone.apex);
    let mut cone = EntailmentCone::with_aperture(apex, storage_cone.aperture, storage_cone.depth)?;
    cone.aperture_factor = storage_cone.aperture_factor;
    Ok(cone)
}

/// Direction of entailment query traversal.
///
/// - `Ancestors`: Find concepts that entail (are more general than) the query node
/// - `Descendants`: Find concepts that are entailed by (are more specific than) the query node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntailmentDirection {
    /// Find ancestors (more general concepts that contain this node in their cones)
    Ancestors,
    /// Find descendants (more specific concepts contained in this node's cone)
    Descendants,
}

/// Result of an entailment query for a single node.
///
/// Contains the node's hyperbolic embedding, entailment cone, and
/// membership score indicating strength of entailment relationship.
#[derive(Debug, Clone)]
pub struct EntailmentResult {
    /// Node ID (i64 for RocksDB compatibility)
    pub node_id: i64,
    /// Hyperbolic position in Poincare ball
    pub point: PoincarePoint,
    /// Entailment cone for this node
    pub cone: EntailmentCone,
    /// Membership score in [0, 1] indicating entailment strength
    pub membership_score: f32,
    /// Depth in hierarchy (from root)
    pub depth: u32,
    /// Whether this is a direct relationship (depth 1 in BFS)
    pub is_direct: bool,
}

/// Parameters for entailment queries.
///
/// Controls traversal depth, result limits, and filtering thresholds.
#[derive(Debug, Clone)]
pub struct EntailmentQueryParams {
    /// Maximum BFS depth for candidate generation (default: 3)
    pub max_depth: u32,
    /// Maximum number of results to return (default: 100)
    pub max_results: usize,
    /// Minimum membership score threshold (default: 0.7)
    pub min_membership_score: f32,
    /// Hyperbolic configuration for distance calculations
    pub hyperbolic_config: HyperbolicConfig,
}

impl Default for EntailmentQueryParams {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_results: 100,
            min_membership_score: 0.7,
            hyperbolic_config: HyperbolicConfig::default(),
        }
    }
}

impl EntailmentQueryParams {
    /// Create params with custom max depth.
    pub fn with_max_depth(mut self, depth: u32) -> Self {
        self.max_depth = depth;
        self
    }

    /// Create params with custom max results.
    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    /// Create params with custom minimum membership score.
    pub fn with_min_score(mut self, score: f32) -> Self {
        self.min_membership_score = score;
        self
    }

    /// Create params with custom hyperbolic config.
    pub fn with_hyperbolic_config(mut self, config: HyperbolicConfig) -> Self {
        self.hyperbolic_config = config;
        self
    }
}

/// Query for entailment relationships starting from a node.
///
/// Uses BFS to generate candidates, then filters by cone containment.
///
/// # Algorithm
///
/// 1. Initialize BFS queue with query node
/// 2. For each node at current depth:
///    - Get neighbors from adjacency list
///    - For Ancestors: check if neighbor's cone contains query point
///    - For Descendants: check if query's cone contains neighbor point
/// 3. Filter results by membership score >= min_membership_score
/// 4. Sort by membership score descending, limit to max_results
///
/// # Arguments
///
/// * `storage` - GraphStorage for retrieving hyperbolic embeddings and cones
/// * `query_node` - Starting node ID for the query
/// * `direction` - Whether to find ancestors or descendants
/// * `params` - Query parameters (depth, limits, thresholds)
///
/// # Returns
///
/// * `Ok(Vec<EntailmentResult>)` - Sorted results by membership score
/// * `Err(GraphError::NodeNotFound)` - If query node doesn't exist
/// * `Err(GraphError::MissingHyperbolicData)` - If required data is missing
///
/// # Performance
///
/// O(n * d) where n = nodes visited by BFS, d = constant (angle computation)
/// Target: <10ms for depth 3 on typical graph
///
/// # Example
///
/// ```ignore
/// use context_graph_graph::entailment::query::{entailment_query, EntailmentDirection, EntailmentQueryParams};
///
/// let results = entailment_query(
///     &storage,
///     node_id,
///     EntailmentDirection::Ancestors,
///     &EntailmentQueryParams::default(),
/// )?;
///
/// for result in results {
///     println!("Node {} entails query with score {}", result.node_id, result.membership_score);
/// }
/// ```
pub fn entailment_query(
    storage: &GraphStorage,
    query_node: i64,
    direction: EntailmentDirection,
    params: &EntailmentQueryParams,
) -> GraphResult<Vec<EntailmentResult>> {
    // FAIL FAST: Get query node's hyperbolic data and convert to hyperbolic types
    let storage_query_point = storage.get_hyperbolic(query_node)?.ok_or_else(|| {
        tracing::error!(
            node_id = query_node,
            "Missing hyperbolic data for query node"
        );
        GraphError::MissingHyperbolicData(query_node)
    })?;
    let query_point = storage_to_hyperbolic_point(&storage_query_point);

    let storage_query_cone = storage.get_cone(query_node)?.ok_or_else(|| {
        tracing::error!(node_id = query_node, "Missing cone data for query node");
        GraphError::NodeNotFound(query_node.to_string())
    })?;
    let query_cone = storage_to_entailment_cone(&storage_query_cone)?;

    let ball = PoincareBall::new(params.hyperbolic_config.clone());

    // BFS for candidate generation
    let mut visited: HashSet<i64> = HashSet::new();
    let mut queue: VecDeque<(i64, u32)> = VecDeque::new(); // (node_id, depth)
    let mut results: Vec<EntailmentResult> = Vec::new();

    visited.insert(query_node);
    queue.push_back((query_node, 0));

    while let Some((current_id, current_depth)) = queue.pop_front() {
        // Stop if we've exceeded max depth
        if current_depth >= params.max_depth {
            continue;
        }

        // Get neighbors
        let neighbors = storage.get_adjacency(current_id)?;

        for edge in neighbors {
            let neighbor_id = edge.target;

            // Skip already visited
            if visited.contains(&neighbor_id) {
                continue;
            }
            visited.insert(neighbor_id);

            // Get neighbor's hyperbolic data and convert
            let storage_neighbor_point = match storage.get_hyperbolic(neighbor_id)? {
                Some(p) => p,
                None => {
                    tracing::debug!(
                        node_id = neighbor_id,
                        "Skipping node without hyperbolic data"
                    );
                    continue;
                }
            };
            let neighbor_point = storage_to_hyperbolic_point(&storage_neighbor_point);

            let storage_neighbor_cone = match storage.get_cone(neighbor_id)? {
                Some(c) => c,
                None => {
                    tracing::debug!(node_id = neighbor_id, "Skipping node without cone data");
                    continue;
                }
            };
            let neighbor_cone = storage_to_entailment_cone(&storage_neighbor_cone)?;

            // Check containment based on direction
            let (is_entailed, score) = match direction {
                EntailmentDirection::Ancestors => {
                    // Ancestor's cone should contain query point
                    let contains = neighbor_cone.contains(&query_point, &ball);
                    let score = neighbor_cone.membership_score(&query_point, &ball);
                    (contains, score)
                }
                EntailmentDirection::Descendants => {
                    // Query's cone should contain neighbor point
                    let contains = query_cone.contains(&neighbor_point, &ball);
                    let score = query_cone.membership_score(&neighbor_point, &ball);
                    (contains, score)
                }
            };

            // Only include if membership score meets threshold
            if score >= params.min_membership_score {
                results.push(EntailmentResult {
                    node_id: neighbor_id,
                    point: neighbor_point,
                    cone: neighbor_cone.clone(),
                    membership_score: score,
                    depth: neighbor_cone.depth,
                    is_direct: current_depth == 0,
                });
            }

            // Continue BFS even if not entailed (might have descendants that are)
            if is_entailed || score > 0.0 {
                queue.push_back((neighbor_id, current_depth + 1));
            }
        }
    }

    // Sort by membership score (descending)
    results.sort_by(|a, b| {
        b.membership_score
            .partial_cmp(&a.membership_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Limit to max_results
    results.truncate(params.max_results);

    Ok(results)
}

/// Check if node A is entailed by node B (A is an ancestor of B).
///
/// This is an O(1) operation using cone containment check.
///
/// # Definition
///
/// A is entailed by B iff B's hyperbolic point is contained in A's cone.
/// In other words, A is a more general concept that subsumes B.
///
/// # Arguments
///
/// * `storage` - GraphStorage for retrieving data
/// * `ancestor_id` - Potential ancestor node (more general)
/// * `descendant_id` - Potential descendant node (more specific)
/// * `config` - Hyperbolic configuration
///
/// # Returns
///
/// * `Ok(true)` - descendant is contained in ancestor's cone
/// * `Ok(false)` - descendant is NOT contained in ancestor's cone
/// * `Err(GraphError::MissingHyperbolicData)` - If required data is missing
///
/// # Performance
///
/// O(1) - single angle computation, target <1ms
///
/// # Example
///
/// ```ignore
/// // Check if "Animal" entails "Dog" (Dog is a kind of Animal)
/// let animal_id = 1;
/// let dog_id = 42;
/// let is_kind_of = is_entailed_by(&storage, animal_id, dog_id, &config)?;
/// assert!(is_kind_of); // Dog IS-A Animal
/// ```
pub fn is_entailed_by(
    storage: &GraphStorage,
    ancestor_id: i64,
    descendant_id: i64,
    config: &HyperbolicConfig,
) -> GraphResult<bool> {
    // FAIL FAST: Get ancestor's cone and convert
    let storage_ancestor_cone = storage.get_cone(ancestor_id)?.ok_or_else(|| {
        tracing::error!(node_id = ancestor_id, "Missing cone data for ancestor node");
        GraphError::MissingHyperbolicData(ancestor_id)
    })?;
    let ancestor_cone = storage_to_entailment_cone(&storage_ancestor_cone)?;

    // FAIL FAST: Get descendant's hyperbolic point and convert
    let storage_descendant_point = storage.get_hyperbolic(descendant_id)?.ok_or_else(|| {
        tracing::error!(
            node_id = descendant_id,
            "Missing hyperbolic data for descendant node"
        );
        GraphError::MissingHyperbolicData(descendant_id)
    })?;
    let descendant_point = storage_to_hyperbolic_point(&storage_descendant_point);

    let ball = PoincareBall::new(config.clone());
    let is_contained = ancestor_cone.contains(&descendant_point, &ball);

    tracing::trace!(
        ancestor_id = ancestor_id,
        descendant_id = descendant_id,
        is_entailed = is_contained,
        "Entailment check completed"
    );

    Ok(is_contained)
}

/// Get the membership score for a descendant relative to an ancestor.
///
/// This quantifies "how much" the descendant belongs to the ancestor's concept.
///
/// # Returns
///
/// Score in [0, 1]:
/// - 1.0 = fully contained in cone (strong IS-A relationship)
/// - <1.0 = partially outside cone (weak relationship)
/// - approaching 0 = far outside cone (no relationship)
///
/// # Formula (CANONICAL - DO NOT MODIFY)
///
/// - If angle <= aperture: score = 1.0
/// - If angle > aperture: score = exp(-2.0 * (angle - aperture))
///
/// # Arguments
///
/// * `storage` - GraphStorage for retrieving data
/// * `ancestor_id` - Ancestor node whose cone defines the concept
/// * `descendant_id` - Descendant node to score
/// * `config` - Hyperbolic configuration
///
/// # Returns
///
/// * `Ok(f32)` - Membership score in [0, 1]
/// * `Err(GraphError::MissingHyperbolicData)` - If required data is missing
///
/// # Performance
///
/// O(1) - single angle computation, target <1ms
pub fn entailment_score(
    storage: &GraphStorage,
    ancestor_id: i64,
    descendant_id: i64,
    config: &HyperbolicConfig,
) -> GraphResult<f32> {
    // FAIL FAST: Get ancestor's cone and convert
    let storage_ancestor_cone = storage.get_cone(ancestor_id)?.ok_or_else(|| {
        tracing::error!(node_id = ancestor_id, "Missing cone data for ancestor node");
        GraphError::MissingHyperbolicData(ancestor_id)
    })?;
    let ancestor_cone = storage_to_entailment_cone(&storage_ancestor_cone)?;

    // FAIL FAST: Get descendant's hyperbolic point and convert
    let storage_descendant_point = storage.get_hyperbolic(descendant_id)?.ok_or_else(|| {
        tracing::error!(
            node_id = descendant_id,
            "Missing hyperbolic data for descendant node"
        );
        GraphError::MissingHyperbolicData(descendant_id)
    })?;
    let descendant_point = storage_to_hyperbolic_point(&storage_descendant_point);

    let ball = PoincareBall::new(config.clone());
    let score = ancestor_cone.membership_score(&descendant_point, &ball);

    tracing::trace!(
        ancestor_id = ancestor_id,
        descendant_id = descendant_id,
        score = score,
        "Entailment score computed"
    );

    Ok(score)
}

/// Result of a batch entailment check.
#[derive(Debug, Clone)]
pub struct BatchEntailmentResult {
    /// Ancestor node ID
    pub ancestor_id: i64,
    /// Descendant node ID
    pub descendant_id: i64,
    /// Whether descendant is entailed by ancestor
    pub is_entailed: bool,
    /// Membership score in [0, 1]
    pub score: f32,
}

/// Check entailment for multiple (ancestor, descendant) pairs.
///
/// More efficient than individual calls due to reduced storage round-trips
/// when nodes appear in multiple pairs.
///
/// # Arguments
///
/// * `storage` - GraphStorage for retrieving data
/// * `pairs` - Vec of (ancestor_id, descendant_id) pairs to check
/// * `config` - Hyperbolic configuration
///
/// # Returns
///
/// * `Ok(Vec<BatchEntailmentResult>)` - Results for all pairs (same order as input)
/// * `Err(GraphError::MissingHyperbolicData)` - If any required data is missing
///
/// # Performance
///
/// O(n) where n = number of pairs, target <100ms for 1000 pairs
///
/// # Example
///
/// ```ignore
/// let pairs = vec![
///     (1, 42),  // Animal -> Dog
///     (1, 43),  // Animal -> Cat
///     (42, 100), // Dog -> Poodle
/// ];
/// let results = entailment_check_batch(&storage, &pairs, &config)?;
/// ```
pub fn entailment_check_batch(
    storage: &GraphStorage,
    pairs: &[(i64, i64)],
    config: &HyperbolicConfig,
) -> GraphResult<Vec<BatchEntailmentResult>> {
    let ball = PoincareBall::new(config.clone());
    let mut results = Vec::with_capacity(pairs.len());

    // Cache for repeated lookups (store converted types)
    let mut cone_cache: std::collections::HashMap<i64, EntailmentCone> =
        std::collections::HashMap::new();
    let mut point_cache: std::collections::HashMap<i64, PoincarePoint> =
        std::collections::HashMap::new();

    for &(ancestor_id, descendant_id) in pairs {
        // Get or cache ancestor cone (with conversion)
        let ancestor_cone = if let Some(cone) = cone_cache.get(&ancestor_id) {
            cone.clone()
        } else {
            let storage_cone = storage.get_cone(ancestor_id)?.ok_or_else(|| {
                tracing::error!(node_id = ancestor_id, "Missing cone data in batch check");
                GraphError::MissingHyperbolicData(ancestor_id)
            })?;
            let cone = storage_to_entailment_cone(&storage_cone)?;
            cone_cache.insert(ancestor_id, cone.clone());
            cone
        };

        // Get or cache descendant point (with conversion)
        let descendant_point = if let Some(point) = point_cache.get(&descendant_id) {
            point.clone()
        } else {
            let storage_point = storage.get_hyperbolic(descendant_id)?.ok_or_else(|| {
                tracing::error!(
                    node_id = descendant_id,
                    "Missing hyperbolic data in batch check"
                );
                GraphError::MissingHyperbolicData(descendant_id)
            })?;
            let point = storage_to_hyperbolic_point(&storage_point);
            point_cache.insert(descendant_id, point.clone());
            point
        };

        // Compute containment and score
        let is_entailed = ancestor_cone.contains(&descendant_point, &ball);
        let score = ancestor_cone.membership_score(&descendant_point, &ball);

        results.push(BatchEntailmentResult {
            ancestor_id,
            descendant_id,
            is_entailed,
            score,
        });
    }

    tracing::debug!(
        pair_count = pairs.len(),
        cache_hits = cone_cache.len() + point_cache.len(),
        "Batch entailment check completed"
    );

    Ok(results)
}

/// Result of a lowest common ancestor query.
#[derive(Debug, Clone)]
pub struct LcaResult {
    /// Lowest common ancestor node ID (None if no common ancestor found)
    pub lca_id: Option<i64>,
    /// Hyperbolic point of LCA (if found)
    pub lca_point: Option<PoincarePoint>,
    /// Entailment cone of LCA (if found)
    pub lca_cone: Option<EntailmentCone>,
    /// Distance from node_a to LCA (in BFS depth)
    pub depth_from_a: u32,
    /// Distance from node_b to LCA (in BFS depth)
    pub depth_from_b: u32,
}

/// Find the lowest common ancestor of two nodes in the entailment hierarchy.
///
/// The LCA is the most specific (deepest) concept that entails both input nodes.
///
/// # Algorithm
///
/// 1. Collect all ancestors of node_a using BFS up the hierarchy
/// 2. Collect all ancestors of node_b using BFS up the hierarchy
/// 3. Find intersection of ancestor sets
/// 4. Return the ancestor with highest depth (most specific)
///
/// # Arguments
///
/// * `storage` - GraphStorage for retrieving data
/// * `node_a` - First node
/// * `node_b` - Second node
/// * `params` - Query parameters (max_depth limits search)
///
/// # Returns
///
/// * `Ok(LcaResult)` - LCA result (lca_id may be None if no common ancestor)
/// * `Err(GraphError::MissingHyperbolicData)` - If required data is missing
///
/// # Performance
///
/// O(n + m) where n, m = nodes visited in each BFS, target <10ms
///
/// # Example
///
/// ```ignore
/// // Find LCA of "Dog" and "Cat" (should be "Animal")
/// let result = lowest_common_ancestor(&storage, dog_id, cat_id, &params)?;
/// if let Some(lca) = result.lca_id {
///     println!("LCA: {} at depth {}", lca, result.depth_from_a + result.depth_from_b);
/// }
/// ```
pub fn lowest_common_ancestor(
    storage: &GraphStorage,
    node_a: i64,
    node_b: i64,
    params: &EntailmentQueryParams,
) -> GraphResult<LcaResult> {
    let ball = PoincareBall::new(params.hyperbolic_config.clone());

    // Collect ancestors of node_a with their depths
    let ancestors_a = collect_ancestors(storage, node_a, params.max_depth, &ball)?;

    // Collect ancestors of node_b with their depths
    let ancestors_b = collect_ancestors(storage, node_b, params.max_depth, &ball)?;

    // Find intersection and pick the one with maximum hierarchy depth (most specific)
    let mut best_lca: Option<(i64, u32, u32, u32)> = None; // (id, depth_from_a, depth_from_b, hierarchy_depth)

    for (&ancestor_id, &(depth_a, hierarchy_depth_a)) in &ancestors_a {
        if let Some(&(depth_b, _)) = ancestors_b.get(&ancestor_id) {
            match &best_lca {
                None => {
                    best_lca = Some((ancestor_id, depth_a, depth_b, hierarchy_depth_a));
                }
                Some((_, _, _, best_depth)) => {
                    // Prefer higher hierarchy depth (more specific ancestor)
                    if hierarchy_depth_a > *best_depth {
                        best_lca = Some((ancestor_id, depth_a, depth_b, hierarchy_depth_a));
                    }
                }
            }
        }
    }

    match best_lca {
        Some((lca_id, depth_from_a, depth_from_b, _)) => {
            // Convert storage types to hyperbolic/entailment types
            let lca_point = storage
                .get_hyperbolic(lca_id)?
                .map(|p| storage_to_hyperbolic_point(&p));
            let lca_cone = match storage.get_cone(lca_id)? {
                Some(c) => Some(storage_to_entailment_cone(&c)?),
                None => None,
            };

            Ok(LcaResult {
                lca_id: Some(lca_id),
                lca_point,
                lca_cone,
                depth_from_a,
                depth_from_b,
            })
        }
        None => Ok(LcaResult {
            lca_id: None,
            lca_point: None,
            lca_cone: None,
            depth_from_a: 0,
            depth_from_b: 0,
        }),
    }
}

/// Helper: Collect ancestors of a node using BFS with cone containment.
///
/// Returns HashMap of ancestor_id -> (bfs_depth, hierarchy_depth)
fn collect_ancestors(
    storage: &GraphStorage,
    start_node: i64,
    max_depth: u32,
    ball: &PoincareBall,
) -> GraphResult<std::collections::HashMap<i64, (u32, u32)>> {
    let mut ancestors: std::collections::HashMap<i64, (u32, u32)> =
        std::collections::HashMap::new();

    // Get start node's point and convert
    let start_point = match storage.get_hyperbolic(start_node)? {
        Some(p) => storage_to_hyperbolic_point(&p),
        None => return Ok(ancestors), // No ancestors if no hyperbolic data
    };

    let mut visited: HashSet<i64> = HashSet::new();
    let mut queue: VecDeque<(i64, u32)> = VecDeque::new();

    visited.insert(start_node);
    queue.push_back((start_node, 0));

    while let Some((current_id, current_depth)) = queue.pop_front() {
        if current_depth >= max_depth {
            continue;
        }

        // Get neighbors (potential ancestors)
        let neighbors = storage.get_adjacency(current_id)?;

        for edge in neighbors {
            let neighbor_id = edge.target;

            if visited.contains(&neighbor_id) {
                continue;
            }
            visited.insert(neighbor_id);

            // Get neighbor's cone and convert to check if it's an ancestor
            let storage_neighbor_cone = match storage.get_cone(neighbor_id)? {
                Some(c) => c,
                None => continue,
            };
            let neighbor_cone = match storage_to_entailment_cone(&storage_neighbor_cone) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Ancestor's cone should contain the start point
            if neighbor_cone.contains(&start_point, ball) {
                ancestors.insert(neighbor_id, (current_depth + 1, neighbor_cone.depth));
            }

            // Continue BFS regardless
            queue.push_back((neighbor_id, current_depth + 1));
        }
    }

    Ok(ancestors)
}

// ============================================================================
// TESTS - MUST USE REAL DATA, NO MOCKS (per constitution REQ-KG-TEST)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HyperbolicConfig;
    use crate::storage::{GraphStorage, StorageConfig};
    use tempfile::TempDir;

    fn create_test_storage() -> (GraphStorage, TempDir) {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = StorageConfig::default();
        let storage =
            GraphStorage::open(temp_dir.path(), config).expect("Failed to open storage");
        (storage, temp_dir)
    }

    fn create_test_point(x: f32) -> PoincarePoint {
        let mut coords = [0.0f32; 64];
        coords[0] = x;
        PoincarePoint::from_coords(coords)
    }

    fn create_test_cone(apex_x: f32, aperture: f32, depth: u32) -> EntailmentCone {
        let apex = create_test_point(apex_x);
        EntailmentCone::with_aperture(apex, aperture, depth).expect("valid cone")
    }

    // ========== Test-only Type Conversion Helpers ==========

    /// Convert hyperbolic PoincarePoint to storage PoincarePoint (test-only).
    fn hyperbolic_to_storage_point(hyp_point: &PoincarePoint) -> StoragePoincarePoint {
        StoragePoincarePoint {
            coords: hyp_point.coords,
        }
    }

    /// Convert entailment EntailmentCone to storage EntailmentCone (test-only).
    fn entailment_to_storage_cone(ent_cone: &EntailmentCone) -> StorageEntailmentCone {
        StorageEntailmentCone {
            apex: hyperbolic_to_storage_point(&ent_cone.apex),
            aperture: ent_cone.aperture,
            aperture_factor: ent_cone.aperture_factor,
            depth: ent_cone.depth,
        }
    }

    // Helper to store entailment cone (converts to storage type)
    fn store_cone(storage: &GraphStorage, node_id: i64, cone: &EntailmentCone) {
        let storage_cone = entailment_to_storage_cone(cone);
        storage.put_cone(node_id, &storage_cone).expect("put cone");
    }

    // Helper to store hyperbolic point (converts to storage type)
    fn store_point(storage: &GraphStorage, node_id: i64, point: &PoincarePoint) {
        let storage_point = hyperbolic_to_storage_point(point);
        storage.put_hyperbolic(node_id, &storage_point).expect("put hyperbolic");
    }

    // ========== EntailmentDirection Tests ==========

    #[test]
    fn test_entailment_direction_equality() {
        assert_eq!(EntailmentDirection::Ancestors, EntailmentDirection::Ancestors);
        assert_eq!(
            EntailmentDirection::Descendants,
            EntailmentDirection::Descendants
        );
        assert_ne!(EntailmentDirection::Ancestors, EntailmentDirection::Descendants);
    }

    #[test]
    fn test_entailment_direction_copy() {
        let dir = EntailmentDirection::Ancestors;
        let copy = dir; // Copy trait
        assert_eq!(dir, copy);
    }

    // ========== EntailmentQueryParams Tests ==========

    #[test]
    fn test_params_default() {
        let params = EntailmentQueryParams::default();
        assert_eq!(params.max_depth, 3);
        assert_eq!(params.max_results, 100);
        assert!((params.min_membership_score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_params_builder() {
        let params = EntailmentQueryParams::default()
            .with_max_depth(5)
            .with_max_results(50)
            .with_min_score(0.5);

        assert_eq!(params.max_depth, 5);
        assert_eq!(params.max_results, 50);
        assert!((params.min_membership_score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_params_with_hyperbolic_config() {
        let config = HyperbolicConfig::with_curvature(-0.5);
        let params = EntailmentQueryParams::default().with_hyperbolic_config(config);

        assert_eq!(params.hyperbolic_config.curvature, -0.5);
    }

    // ========== is_entailed_by Tests ==========

    #[test]
    fn test_is_entailed_by_missing_ancestor_cone() {
        let (storage, _temp_dir) = create_test_storage();
        let config = HyperbolicConfig::default();

        // No data stored - should fail fast
        let result = is_entailed_by(&storage, 1, 2, &config);
        assert!(matches!(result, Err(GraphError::MissingHyperbolicData(1))));
    }

    #[test]
    fn test_is_entailed_by_missing_descendant_point() {
        let (storage, _temp_dir) = create_test_storage();
        let config = HyperbolicConfig::default();

        // Store ancestor cone but no descendant point
        let ancestor_cone = create_test_cone(0.0, 1.0, 0);
        store_cone(&storage, 1, &ancestor_cone);

        let result = is_entailed_by(&storage, 1, 2, &config);
        assert!(matches!(result, Err(GraphError::MissingHyperbolicData(2))));
    }

    #[test]
    fn test_is_entailed_by_contained() {
        let (storage, _temp_dir) = create_test_storage();
        let config = HyperbolicConfig::default();

        // Ancestor at origin with wide cone
        let ancestor_cone = create_test_cone(0.0, 1.5, 0);
        store_cone(&storage, 1, &ancestor_cone);

        // Descendant point inside cone
        let descendant_point = create_test_point(0.3);
        store_point(&storage, 2, &descendant_point);

        let result = is_entailed_by(&storage, 1, 2, &config).expect("check should succeed");
        assert!(result, "Descendant should be contained in ancestor's cone");
    }

    #[test]
    fn test_is_entailed_by_not_contained() {
        let (storage, _temp_dir) = create_test_storage();
        let config = HyperbolicConfig::default();

        // Ancestor with narrow cone away from origin
        let mut apex_coords = [0.0f32; 64];
        apex_coords[0] = 0.5;
        let apex = PoincarePoint::from_coords(apex_coords);
        let ancestor_cone =
            EntailmentCone::with_aperture(apex, 0.2, 1).expect("valid cone");
        store_cone(&storage, 1, &ancestor_cone);

        // Descendant point perpendicular to cone axis
        let mut desc_coords = [0.0f32; 64];
        desc_coords[1] = 0.5; // Perpendicular direction
        let descendant_point = PoincarePoint::from_coords(desc_coords);
        store_point(&storage, 2, &descendant_point);

        let result = is_entailed_by(&storage, 1, 2, &config).expect("check should succeed");
        assert!(!result, "Perpendicular point should not be contained");
    }

    // ========== entailment_score Tests ==========

    #[test]
    fn test_entailment_score_fully_contained() {
        let (storage, _temp_dir) = create_test_storage();
        let config = HyperbolicConfig::default();

        // Ancestor at origin with wide cone
        let ancestor_cone = create_test_cone(0.0, 1.5, 0);
        store_cone(&storage, 1, &ancestor_cone);

        // Descendant point inside cone (score should be 1.0)
        let descendant_point = create_test_point(0.3);
        store_point(&storage, 2, &descendant_point);

        let score = entailment_score(&storage, 1, 2, &config).expect("score should succeed");
        assert_eq!(score, 1.0, "Fully contained point should have score 1.0");
    }

    #[test]
    fn test_entailment_score_partial() {
        let (storage, _temp_dir) = create_test_storage();
        let config = HyperbolicConfig::default();

        // Ancestor with narrow cone
        let mut apex_coords = [0.0f32; 64];
        apex_coords[0] = 0.5;
        let apex = PoincarePoint::from_coords(apex_coords);
        let ancestor_cone =
            EntailmentCone::with_aperture(apex, 0.3, 1).expect("valid cone");
        store_cone(&storage, 1, &ancestor_cone);

        // Descendant point outside but near cone
        let mut desc_coords = [0.0f32; 64];
        desc_coords[1] = 0.4;
        let descendant_point = PoincarePoint::from_coords(desc_coords);
        store_point(&storage, 2, &descendant_point);

        let score = entailment_score(&storage, 1, 2, &config).expect("score should succeed");
        assert!(score > 0.0, "Nearby point should have positive score");
        assert!(score < 1.0, "Outside point should have score < 1.0");
    }

    // ========== entailment_check_batch Tests ==========

    #[test]
    fn test_batch_check_empty() {
        let (storage, _temp_dir) = create_test_storage();
        let config = HyperbolicConfig::default();

        let results = entailment_check_batch(&storage, &[], &config).expect("batch should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_check_single_pair() {
        let (storage, _temp_dir) = create_test_storage();
        let config = HyperbolicConfig::default();

        // Setup data
        let ancestor_cone = create_test_cone(0.0, 1.5, 0);
        store_cone(&storage, 1, &ancestor_cone);

        let descendant_point = create_test_point(0.3);
        store_point(&storage, 2, &descendant_point);

        let pairs = vec![(1, 2)];
        let results = entailment_check_batch(&storage, &pairs, &config).expect("batch should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].ancestor_id, 1);
        assert_eq!(results[0].descendant_id, 2);
        assert!(results[0].is_entailed);
        assert_eq!(results[0].score, 1.0);
    }

    #[test]
    fn test_batch_check_multiple_pairs() {
        let (storage, _temp_dir) = create_test_storage();
        let config = HyperbolicConfig::default();

        // Ancestor cone at origin
        let ancestor_cone = create_test_cone(0.0, 1.5, 0);
        store_cone(&storage, 1, &ancestor_cone);

        // Multiple descendants
        for i in 2..=5 {
            let point = create_test_point(0.1 * i as f32);
            store_point(&storage, i, &point);
        }

        let pairs: Vec<(i64, i64)> = (2..=5).map(|i| (1, i)).collect();
        let results = entailment_check_batch(&storage, &pairs, &config).expect("batch should succeed");

        assert_eq!(results.len(), 4);
        for (idx, result) in results.iter().enumerate() {
            assert_eq!(result.ancestor_id, 1);
            assert_eq!(result.descendant_id, (idx + 2) as i64);
        }
    }

    #[test]
    fn test_batch_check_caching() {
        let (storage, _temp_dir) = create_test_storage();
        let config = HyperbolicConfig::default();

        // Same ancestor, multiple descendants
        let ancestor_cone = create_test_cone(0.0, 1.5, 0);
        store_cone(&storage, 1, &ancestor_cone);

        for i in 2..=10 {
            let point = create_test_point(0.05 * i as f32);
            store_point(&storage, i, &point);
        }

        // All pairs share ancestor - should benefit from caching
        let pairs: Vec<(i64, i64)> = (2..=10).map(|i| (1, i)).collect();
        let results = entailment_check_batch(&storage, &pairs, &config).expect("batch should succeed");

        assert_eq!(results.len(), 9);
    }

    // ========== LcaResult Tests ==========

    #[test]
    fn test_lca_result_no_common_ancestor() {
        let result = LcaResult {
            lca_id: None,
            lca_point: None,
            lca_cone: None,
            depth_from_a: 0,
            depth_from_b: 0,
        };
        assert!(result.lca_id.is_none());
    }

    #[test]
    fn test_lca_result_with_ancestor() {
        let point = create_test_point(0.3);
        let cone = create_test_cone(0.3, 1.0, 2);

        let result = LcaResult {
            lca_id: Some(42),
            lca_point: Some(point),
            lca_cone: Some(cone),
            depth_from_a: 2,
            depth_from_b: 1,
        };

        assert_eq!(result.lca_id, Some(42));
        assert_eq!(result.depth_from_a, 2);
        assert_eq!(result.depth_from_b, 1);
    }

    // ========== EntailmentResult Tests ==========

    #[test]
    fn test_entailment_result_construction() {
        let point = create_test_point(0.5);
        let cone = create_test_cone(0.5, 0.8, 3);

        let result = EntailmentResult {
            node_id: 42,
            point: point.clone(),
            cone: cone.clone(),
            membership_score: 0.95,
            depth: 3,
            is_direct: true,
        };

        assert_eq!(result.node_id, 42);
        assert!((result.membership_score - 0.95).abs() < 1e-6);
        assert_eq!(result.depth, 3);
        assert!(result.is_direct);
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_self_entailment() {
        let (storage, _temp_dir) = create_test_storage();
        let config = HyperbolicConfig::default();

        // Node with both cone and point
        let point = create_test_point(0.3);
        let apex = point.clone();
        let cone = EntailmentCone::with_aperture(apex, 1.0, 1).expect("valid cone");

        store_cone(&storage, 1, &cone);
        store_point(&storage, 1, &point);

        // Node should entail itself (point at apex is always contained)
        let result = is_entailed_by(&storage, 1, 1, &config).expect("check should succeed");
        assert!(result, "Node should entail itself");

        let score = entailment_score(&storage, 1, 1, &config).expect("score should succeed");
        assert_eq!(score, 1.0, "Self-entailment score should be 1.0");
    }

    #[test]
    fn test_batch_check_missing_data() {
        let (storage, _temp_dir) = create_test_storage();
        let config = HyperbolicConfig::default();

        // Only partial data
        let cone = create_test_cone(0.0, 1.5, 0);
        store_cone(&storage, 1, &cone);
        // No descendant point

        let pairs = vec![(1, 2)];
        let result = entailment_check_batch(&storage, &pairs, &config);

        assert!(matches!(result, Err(GraphError::MissingHyperbolicData(2))));
    }
}
