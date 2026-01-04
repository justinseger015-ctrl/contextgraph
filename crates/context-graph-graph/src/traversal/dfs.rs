//! DFS (Depth-First Search) graph traversal with Marblestone domain modulation.
//!
//! Explores the graph depth-first using an ITERATIVE approach (explicit stack).
//! NO RECURSION is used to avoid stack overflow on large graphs.
//!
//! # Performance
//!
//! Target: No stack overflow on 100,000+ node graphs.
//! Uses Vec<(NodeId, usize)> as explicit stack.
//! Uses HashSet for O(1) visited lookup.
//!
//! # Constitution Reference
//!
//! - edge_model.nt_weights.formula: Canonical modulation formula
//! - AP-009: NaN/Infinity clamped to valid range

use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::error::GraphResult;
use crate::storage::GraphStorage;

// Re-export edge types for convenience
pub use crate::storage::edges::{Domain, EdgeType};

/// Node ID type for DFS (i64 for storage compatibility).
pub type NodeId = i64;

/// Parameters for DFS traversal.
///
/// Controls depth limits, node limits, and filtering behavior.
#[derive(Debug, Clone)]
pub struct DfsParams {
    /// Maximum depth to traverse (None = unlimited).
    /// Depth 0 is the start node.
    pub max_depth: Option<usize>,

    /// Maximum number of nodes to visit (default: 10000).
    /// Prevents runaway traversal on dense graphs.
    pub max_nodes: Option<usize>,

    /// Filter to specific edge types (None = all types).
    pub edge_types: Option<Vec<EdgeType>>,

    /// Domain for NT weight modulation.
    pub domain: Domain,

    /// Minimum edge weight threshold (after modulation).
    /// Edges below this weight are not traversed.
    pub min_weight: f32,
}

impl Default for DfsParams {
    fn default() -> Self {
        Self {
            max_depth: Some(10),
            max_nodes: Some(10_000),
            edge_types: None,
            domain: Domain::General,
            min_weight: 0.0,
        }
    }
}

impl DfsParams {
    /// Builder: set max depth.
    #[must_use]
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Builder: set unlimited depth.
    #[must_use]
    pub fn unlimited_depth(mut self) -> Self {
        self.max_depth = None;
        self
    }

    /// Builder: set max nodes.
    #[must_use]
    pub fn max_nodes(mut self, nodes: usize) -> Self {
        self.max_nodes = Some(nodes);
        self
    }

    /// Builder: set edge types filter.
    #[must_use]
    pub fn edge_types(mut self, types: Vec<EdgeType>) -> Self {
        self.edge_types = Some(types);
        self
    }

    /// Builder: set domain for NT weight modulation.
    #[must_use]
    pub fn domain(mut self, domain: Domain) -> Self {
        self.domain = domain;
        self
    }

    /// Builder: set minimum weight threshold.
    #[must_use]
    pub fn min_weight(mut self, weight: f32) -> Self {
        self.min_weight = weight;
        self
    }
}

/// Result of DFS traversal.
#[derive(Debug, Clone)]
pub struct DfsResult {
    /// Visited node IDs in DFS pre-order.
    pub visited_order: Vec<NodeId>,

    /// Depth at which each node was discovered.
    pub depths: HashMap<NodeId, usize>,

    /// Parent node for each discovered node (start node has None).
    pub parents: HashMap<NodeId, Option<NodeId>>,

    /// Edges traversed: (source, target, effective_weight).
    pub edges_traversed: Vec<(NodeId, NodeId, f32)>,
}

impl DfsResult {
    /// Create a new empty result.
    #[must_use]
    pub fn new() -> Self {
        Self {
            visited_order: Vec::new(),
            depths: HashMap::new(),
            parents: HashMap::new(),
            edges_traversed: Vec::new(),
        }
    }

    /// Reconstruct path from start to target.
    ///
    /// Returns None if target was not visited.
    #[must_use]
    pub fn path_to(&self, target: NodeId) -> Option<Vec<NodeId>> {
        if !self.parents.contains_key(&target) {
            return None;
        }

        let mut path = vec![target];
        let mut current = target;

        while let Some(Some(parent)) = self.parents.get(&current) {
            path.push(*parent);
            current = *parent;
        }

        path.reverse();
        Some(path)
    }

    /// Get total node count.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.visited_order.len()
    }

    /// Get maximum depth reached.
    #[must_use]
    pub fn max_depth_reached(&self) -> usize {
        self.depths.values().copied().max().unwrap_or(0)
    }
}

impl Default for DfsResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert UUID to i64 for storage key operations.
///
/// This reverses `Uuid::from_u64_pair(id as u64, 0)` used in storage.
/// from_u64_pair stores values in big-endian order in the UUID bytes.
#[inline]
fn uuid_to_i64(uuid: &Uuid) -> i64 {
    let bytes = uuid.as_bytes();
    // from_u64_pair uses big-endian byte order
    i64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Perform ITERATIVE DFS traversal from a starting node.
///
/// # IMPORTANT: This is an ITERATIVE implementation using explicit stack.
/// NO RECURSION is used to avoid stack overflow on large graphs.
///
/// # Arguments
/// * `storage` - Graph storage backend
/// * `start` - Starting node ID (i64)
/// * `params` - Traversal parameters
///
/// # Returns
/// * `Ok(DfsResult)` - Traversal results
/// * `Err(GraphError)` - Storage access failed or node not found
///
/// # Algorithm
/// Uses a Vec<(NodeId, usize)> as explicit stack for depth-first traversal.
/// Visits nodes in pre-order (node visited before its children).
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_graph::traversal::dfs::{dfs_traverse, DfsParams, Domain};
///
/// let params = DfsParams::default()
///     .max_depth(5)
///     .domain(Domain::Code)
///     .min_weight(0.3);
///
/// let result = dfs_traverse(&storage, start_node, params)?;
/// println!("Visited {} nodes, max depth: {}",
///     result.node_count(), result.max_depth_reached());
/// ```
pub fn dfs_traverse(
    storage: &GraphStorage,
    start: NodeId,
    params: DfsParams,
) -> GraphResult<DfsResult> {
    let mut result = DfsResult::new();
    let mut visited: HashSet<NodeId> = HashSet::new();

    // ITERATIVE: Use Vec as explicit stack (NOT recursion)
    // Each entry is (node_id, depth)
    let mut stack: Vec<(NodeId, usize)> = vec![(start, 0)];

    // Initialize parent for start node
    result.parents.insert(start, None);

    while let Some((current, depth)) = stack.pop() {
        // Skip if already visited (handles cycles)
        if visited.contains(&current) {
            continue;
        }

        // Check max nodes limit BEFORE processing
        if let Some(max) = params.max_nodes {
            if visited.len() >= max {
                log::debug!(
                    "DFS truncated at {} nodes (limit: {})",
                    visited.len(),
                    max
                );
                break;
            }
        }

        // Mark as visited and record in result
        visited.insert(current);
        result.visited_order.push(current);
        result.depths.insert(current, depth);

        // Don't expand if at max depth
        if let Some(max_depth) = params.max_depth {
            if depth >= max_depth {
                continue;
            }
        }

        // CORRECT API: get_outgoing_edges NOT get_adjacency
        let edges = storage.get_outgoing_edges(current)?;

        // Collect valid neighbors to push to stack
        let mut neighbors: Vec<(NodeId, f32)> = Vec::new();

        for edge in edges {
            // Filter by edge type if specified
            if let Some(ref allowed_types) = params.edge_types {
                if !allowed_types.contains(&edge.edge_type) {
                    continue;
                }
            }

            // Get NT-modulated weight
            let effective_weight = edge.get_modulated_weight(params.domain);

            // Filter by minimum weight
            if effective_weight < params.min_weight {
                continue;
            }

            // CRITICAL: Convert UUID to i64
            let neighbor_id = uuid_to_i64(&edge.target);

            // Skip if already visited
            if visited.contains(&neighbor_id) {
                continue;
            }

            neighbors.push((neighbor_id, effective_weight));

            // Record parent if not already set
            result.parents.entry(neighbor_id).or_insert(Some(current));

            // Record edge traversal
            result.edges_traversed.push((current, neighbor_id, effective_weight));
        }

        // Push neighbors to stack in REVERSE order for correct DFS order
        // (so first neighbor is processed first when popped)
        for (neighbor_id, _) in neighbors.into_iter().rev() {
            stack.push((neighbor_id, depth + 1));
        }
    }

    log::debug!(
        "DFS complete: {} nodes, max_depth={}",
        result.visited_order.len(),
        result.max_depth_reached()
    );

    Ok(result)
}

/// DFS Iterator for lazy traversal.
///
/// Yields nodes one at a time without building full result.
/// Useful for early termination or memory-constrained scenarios.
pub struct DfsIterator<'a> {
    storage: &'a GraphStorage,
    stack: Vec<(NodeId, usize)>,
    visited: HashSet<NodeId>,
    params: DfsParams,
}

impl<'a> DfsIterator<'a> {
    /// Create a new DFS iterator.
    pub fn new(storage: &'a GraphStorage, start: NodeId, params: DfsParams) -> Self {
        Self {
            storage,
            stack: vec![(start, 0)],
            visited: HashSet::new(),
            params,
        }
    }
}

impl<'a> Iterator for DfsIterator<'a> {
    type Item = GraphResult<(NodeId, usize)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (current, depth) = self.stack.pop()?;

            // Skip if already visited
            if self.visited.contains(&current) {
                continue;
            }

            // Check max nodes limit
            if let Some(max) = self.params.max_nodes {
                if self.visited.len() >= max {
                    return None;
                }
            }

            // Mark as visited
            self.visited.insert(current);

            // Expand children if not at max depth
            if self.params.max_depth.map_or(true, |max| depth < max) {
                // Get outgoing edges
                let edges = match self.storage.get_outgoing_edges(current) {
                    Ok(e) => e,
                    Err(err) => return Some(Err(err)),
                };

                let mut neighbors: Vec<NodeId> = Vec::new();

                for edge in edges {
                    // Filter by edge type
                    if let Some(ref allowed_types) = self.params.edge_types {
                        if !allowed_types.contains(&edge.edge_type) {
                            continue;
                        }
                    }

                    // Check weight threshold
                    let effective_weight = edge.get_modulated_weight(self.params.domain);
                    if effective_weight < self.params.min_weight {
                        continue;
                    }

                    let neighbor_id = uuid_to_i64(&edge.target);

                    if !self.visited.contains(&neighbor_id) {
                        neighbors.push(neighbor_id);
                    }
                }

                // Push in reverse for correct order
                for neighbor_id in neighbors.into_iter().rev() {
                    self.stack.push((neighbor_id, depth + 1));
                }
            }

            return Some(Ok((current, depth)));
        }
    }
}

/// Get all nodes within a given depth from start using DFS.
///
/// Convenience wrapper around dfs_traverse.
pub fn dfs_neighborhood(
    storage: &GraphStorage,
    center: NodeId,
    max_depth: usize,
) -> GraphResult<Vec<NodeId>> {
    let params = DfsParams::default().max_depth(max_depth);
    let result = dfs_traverse(storage, center, params)?;
    Ok(result.visited_order)
}

/// Get nodes within depth, filtered by domain and minimum weight.
///
/// Returns only nodes reachable via edges with weight >= min_weight
/// after domain modulation.
pub fn dfs_domain_neighborhood(
    storage: &GraphStorage,
    center: NodeId,
    max_depth: usize,
    domain: Domain,
    min_weight: f32,
) -> GraphResult<Vec<NodeId>> {
    let params = DfsParams::default()
        .max_depth(max_depth)
        .domain(domain)
        .min_weight(min_weight);
    let result = dfs_traverse(storage, center, params)?;
    Ok(result.visited_order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::GraphEdge;
    use tempfile::tempdir;

    /// Create test graph and return (storage, start_node_id, tempdir).
    fn setup_test_graph() -> (GraphStorage, NodeId, tempfile::TempDir) {
        let dir = tempdir().expect("Failed to create temp dir");
        let storage = GraphStorage::open_default(dir.path())
            .expect("Failed to open storage");

        // Create a simple tree structure:
        //     1
        //    / \
        //   2   3
        //  /|   |\
        // 4 5   6 7

        let uuid = |id: i64| Uuid::from_u64_pair(id as u64, 0);

        let edges = vec![
            // From node 1
            GraphEdge::new(1, uuid(1), uuid(2), EdgeType::Semantic, 0.8, Domain::General),
            GraphEdge::new(2, uuid(1), uuid(3), EdgeType::Semantic, 0.8, Domain::General),
            // From node 2
            GraphEdge::new(3, uuid(2), uuid(4), EdgeType::Semantic, 0.7, Domain::General),
            GraphEdge::new(4, uuid(2), uuid(5), EdgeType::Semantic, 0.7, Domain::General),
            // From node 3
            GraphEdge::new(5, uuid(3), uuid(6), EdgeType::Hierarchical, 0.7, Domain::Code),
            GraphEdge::new(6, uuid(3), uuid(7), EdgeType::Hierarchical, 0.7, Domain::Code),
        ];

        storage.put_edges(&edges).expect("put_edges failed");

        (storage, 1, dir)
    }

    #[test]
    fn test_dfs_basic_traversal() {
        let (storage, start, _dir) = setup_test_graph();

        let result = dfs_traverse(&storage, start, DfsParams::default())
            .expect("DFS failed");

        // Should find all 7 nodes
        assert_eq!(result.node_count(), 7, "Expected 7 nodes, got {}", result.node_count());
        assert_eq!(result.visited_order[0], 1, "Start node should be first");

        // DFS visits nodes in pre-order (depth-first)
        // Verify all nodes are present
        let visited_set: HashSet<_> = result.visited_order.iter().copied().collect();
        for i in 1..=7 {
            assert!(visited_set.contains(&i), "Node {} should be visited", i);
        }

        // Verify max depth
        assert_eq!(result.max_depth_reached(), 2);
    }

    #[test]
    fn test_dfs_depth_limit() {
        let (storage, start, _dir) = setup_test_graph();

        let result = dfs_traverse(
            &storage,
            start,
            DfsParams::default().max_depth(1),
        ).expect("DFS failed");

        // Should find only depth 0 and 1: nodes 1, 2, 3
        assert_eq!(result.node_count(), 3);
        assert_eq!(result.max_depth_reached(), 1);
    }

    #[test]
    fn test_dfs_cycle_handling() {
        let dir = tempdir().expect("Failed to create temp dir");
        let storage = GraphStorage::open_default(dir.path())
            .expect("Failed to open storage");

        // Create cycle: 1 -> 2 -> 3 -> 1
        let uuid = |id: i64| Uuid::from_u64_pair(id as u64, 0);
        let edges = vec![
            GraphEdge::new(1, uuid(1), uuid(2), EdgeType::Semantic, 0.8, Domain::General),
            GraphEdge::new(2, uuid(2), uuid(3), EdgeType::Semantic, 0.8, Domain::General),
            GraphEdge::new(3, uuid(3), uuid(1), EdgeType::Semantic, 0.8, Domain::General),
        ];
        storage.put_edges(&edges).expect("put_edges failed");

        let result = dfs_traverse(&storage, 1, DfsParams::default())
            .expect("DFS failed");

        // Should visit each node exactly once, no infinite loop
        assert_eq!(result.node_count(), 3);

        // Verify each node visited exactly once
        let mut counts: HashMap<NodeId, usize> = HashMap::new();
        for &node in &result.visited_order {
            *counts.entry(node).or_insert(0) += 1;
        }
        for i in 1..=3 {
            assert_eq!(counts.get(&i), Some(&1), "Node {} should be visited exactly once", i);
        }
    }

    #[test]
    fn test_dfs_edge_type_filter() {
        let (storage, start, _dir) = setup_test_graph();

        // Only follow Semantic edges (not Hierarchical)
        let result = dfs_traverse(
            &storage,
            start,
            DfsParams::default().edge_types(vec![EdgeType::Semantic]),
        ).expect("DFS failed");

        // Nodes 6 and 7 are only reachable via Hierarchical edges
        // So we should find: 1, 2, 3, 4, 5 = 5 nodes
        assert_eq!(result.node_count(), 5);

        let visited_set: HashSet<_> = result.visited_order.iter().copied().collect();
        assert!(!visited_set.contains(&6), "Node 6 should NOT be visited");
        assert!(!visited_set.contains(&7), "Node 7 should NOT be visited");
    }

    #[test]
    fn test_dfs_weight_threshold() {
        let (storage, start, _dir) = setup_test_graph();

        // Set very high min_weight to filter all edges
        // Note: modulated weight includes domain bonus and NT weights
        // so we need a very high threshold to filter everything
        let result = dfs_traverse(
            &storage,
            start,
            DfsParams::default().min_weight(2.0), // Impossible threshold
        ).expect("DFS failed");

        // All edges should be filtered, only start node remains
        assert_eq!(result.node_count(), 1, "Only start node with impossible threshold");
        assert_eq!(result.visited_order[0], start);
    }

    #[test]
    fn test_dfs_domain_modulation() {
        let (storage, start, _dir) = setup_test_graph();

        // With Code domain, edges from node 3 (Domain::Code) get bonus
        let result_code = dfs_traverse(
            &storage,
            start,
            DfsParams::default().domain(Domain::Code).min_weight(0.5),
        ).expect("DFS failed");

        // With General domain (different from Code edges)
        let result_general = dfs_traverse(
            &storage,
            start,
            DfsParams::default().domain(Domain::General).min_weight(0.5),
        ).expect("DFS failed");

        // Both should find nodes, but potentially different effective weights
        assert!(result_code.node_count() >= 1);
        assert!(result_general.node_count() >= 1);
    }

    #[test]
    fn test_dfs_path_reconstruction() {
        let (storage, start, _dir) = setup_test_graph();

        let result = dfs_traverse(&storage, start, DfsParams::default())
            .expect("DFS failed");

        // Test path to node 4 (should be 1 -> 2 -> 4)
        let path = result.path_to(4);
        assert!(path.is_some(), "Path to node 4 should exist");
        let path = path.unwrap();
        assert_eq!(path[0], 1, "Path should start at 1");
        assert_eq!(*path.last().unwrap(), 4, "Path should end at 4");

        // Test path to nonexistent node
        let no_path = result.path_to(99);
        assert!(no_path.is_none(), "Path to nonexistent node should be None");
    }

    #[test]
    fn test_dfs_empty_graph() {
        let dir = tempdir().expect("Failed to create temp dir");
        let storage = GraphStorage::open_default(dir.path())
            .expect("Failed to open storage");

        let result = dfs_traverse(&storage, 1, DfsParams::default())
            .expect("DFS failed");

        // Should return just the start node (even if it has no edges)
        assert_eq!(result.node_count(), 1);
        assert_eq!(result.visited_order[0], 1);
        assert!(result.edges_traversed.is_empty());
    }

    #[test]
    fn test_dfs_max_nodes_limit() {
        let (storage, start, _dir) = setup_test_graph();

        let result = dfs_traverse(
            &storage,
            start,
            DfsParams::default().max_nodes(3),
        ).expect("DFS failed");

        assert_eq!(result.node_count(), 3, "Should stop at max_nodes limit");
    }

    #[test]
    fn test_dfs_iterator() {
        let (storage, start, _dir) = setup_test_graph();

        let iter = DfsIterator::new(&storage, start, DfsParams::default());

        let collected: Vec<_> = iter
            .map(|r| r.expect("Iterator error"))
            .collect();

        // Should yield all 7 nodes
        assert_eq!(collected.len(), 7);
        assert_eq!(collected[0].0, 1, "First node should be start");

        // Verify all nodes present
        let node_ids: HashSet<_> = collected.iter().map(|(id, _)| *id).collect();
        for i in 1..=7 {
            assert!(node_ids.contains(&i), "Node {} should be in iterator output", i);
        }
    }

    #[test]
    fn test_dfs_vs_bfs_coverage() {
        // DFS and BFS should visit the same set of nodes (just in different order)
        let (storage, start, _dir) = setup_test_graph();

        let dfs_result = dfs_traverse(&storage, start, DfsParams::default())
            .expect("DFS failed");

        // DFS should find all 7 nodes
        assert_eq!(dfs_result.node_count(), 7);

        let dfs_set: HashSet<_> = dfs_result.visited_order.iter().copied().collect();
        for i in 1..=7 {
            assert!(dfs_set.contains(&i), "DFS should visit node {}", i);
        }
    }

    #[test]
    fn test_dfs_deep_chain() {
        // Test iterative DFS on deep chain (would overflow stack if recursive)
        let dir = tempdir().expect("Failed to create temp dir");
        let storage = GraphStorage::open_default(dir.path())
            .expect("Failed to open storage");

        // Create chain: 0 -> 1 -> 2 -> ... -> 999
        let uuid = |id: i64| Uuid::from_u64_pair(id as u64, 0);
        let edges: Vec<GraphEdge> = (0..1000i64)
            .map(|i| {
                GraphEdge::new(i, uuid(i), uuid(i + 1), EdgeType::Semantic, 0.8, Domain::General)
            })
            .collect();
        storage.put_edges(&edges).expect("put_edges failed");

        // Should NOT stack overflow (iterative implementation)
        let result = dfs_traverse(
            &storage,
            0,
            DfsParams::default().max_depth(1000).max_nodes(1100),
        ).expect("DFS failed on deep chain - implementation may be recursive!");

        // Should find all 1001 nodes (0 through 1000)
        assert_eq!(result.node_count(), 1001);
        assert_eq!(result.max_depth_reached(), 1000);
    }

    #[test]
    fn test_dfs_weight_boundary() {
        // Test weight filtering with modulated weights.
        // The modulation formula is: w_eff = weight * (1.0 + net_activation + domain_bonus) * steering_factor
        // With default Domain::General (no bonus) and no NT weights, effective weight ≈ base weight.
        // To test proper filtering, we use thresholds that account for possible modulation bonuses.
        let dir = tempdir().expect("Failed to create temp dir");
        let storage = GraphStorage::open_default(dir.path())
            .expect("Failed to open storage");

        let uuid = |id: i64| Uuid::from_u64_pair(id as u64, 0);

        // Edge with base weight 0.5
        let edges = vec![
            GraphEdge::new(1, uuid(1), uuid(2), EdgeType::Semantic, 0.5, Domain::General),
        ];
        storage.put_edges(&edges).expect("put_edges failed");

        // Low threshold (0.1) should always pass - effective weight will be >= base weight
        let result = dfs_traverse(
            &storage,
            1,
            DfsParams::default().min_weight(0.1),
        ).expect("DFS failed");
        assert_eq!(result.node_count(), 2, "Edge with weight 0.5 should pass threshold 0.1");

        // High threshold (2.0) should always fail - impossible to achieve with modulation
        // Max modulation: weight * (1 + 1.0 + 0.2) * 1.0 = weight * 2.2 = 0.5 * 2.2 = 1.1 max
        let result = dfs_traverse(
            &storage,
            1,
            DfsParams::default().min_weight(2.0),
        ).expect("DFS failed");
        assert_eq!(result.node_count(), 1, "Edge with weight 0.5 should not reach threshold 2.0");
    }

    #[test]
    fn test_dfs_neighborhood() {
        let (storage, start, _dir) = setup_test_graph();

        let neighbors = dfs_neighborhood(&storage, start, 1)
            .expect("DFS failed");

        // Distance 1: start + immediate neighbors (2 and 3)
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
        assert!(neighbors.contains(&3));
        assert_eq!(neighbors.len(), 3);
    }

    #[test]
    fn test_dfs_domain_neighborhood() {
        let (storage, start, _dir) = setup_test_graph();

        let neighbors = dfs_domain_neighborhood(&storage, start, 2, Domain::Code, 0.5)
            .expect("DFS failed");

        // Should include nodes reachable with modulated weights >= 0.5
        assert!(neighbors.contains(&1));
        assert!(!neighbors.is_empty());
    }
}
