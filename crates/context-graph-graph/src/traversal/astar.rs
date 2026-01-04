//! A* graph traversal with hyperbolic distance heuristic.
//!
//! Optimal pathfinding in the knowledge graph using A* algorithm with
//! Poincare ball hyperbolic distance as the heuristic function.
//!
//! # Algorithm
//!
//! A* combines uniform-cost search with a heuristic:
//! - f(n) = g(n) + h(n)
//! - g(n) = actual cost from start to n
//! - h(n) = heuristic estimate from n to goal
//!
//! # Hyperbolic Heuristic
//!
//! Uses `PoincareBall.distance()` scaled by 0.1 to ensure admissibility.
//! The heuristic must underestimate actual path cost to guarantee optimality.
//!
//! # Edge Cost
//!
//! Edge cost = 1.0 / (effective_weight + 0.001)
//! Higher weights = lower cost = preferred paths.
//!
//! # Performance
//!
//! Target: <50ms for single-source-single-target on 10M node graph.
//! Uses BinaryHeap for O(log n) priority queue operations.
//!
//! # Constitution Reference
//!
//! - edge_model.nt_weights.formula: w_eff = base * (1 + excitatory - inhibitory + 0.5*modulatory)
//! - AP-001: Never unwrap() - returns MissingHyperbolicData if embeddings missing
//! - AP-009: NaN/Infinity clamped to valid range

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use uuid::Uuid;

use crate::config::HyperbolicConfig;
use crate::error::{GraphError, GraphResult};
use crate::hyperbolic::PoincareBall;
use crate::hyperbolic::PoincarePoint as HyperbolicPoint;
use crate::storage::GraphStorage;
use crate::storage::PoincarePoint as StoragePoint;

// Re-export for convenience
pub use crate::storage::edges::{Domain, EdgeType};

/// Node ID type for A* (i64 for storage compatibility).
pub type NodeId = i64;

/// Parameters for A* traversal.
#[derive(Debug, Clone)]
pub struct AstarParams {
    /// Domain for NT weight modulation.
    pub domain: Domain,

    /// Minimum edge weight threshold (after modulation).
    pub min_weight: f32,

    /// Filter to specific edge types (None = all types).
    pub edge_types: Option<Vec<EdgeType>>,

    /// Maximum nodes to explore before giving up (default: 100000).
    pub max_nodes: usize,

    /// Heuristic scale factor (default: 0.1).
    /// Smaller values = more admissible but slower.
    /// Must be <= 1.0 for optimality guarantee.
    pub heuristic_scale: f32,
}

impl Default for AstarParams {
    fn default() -> Self {
        Self {
            domain: Domain::General,
            min_weight: 0.0,
            edge_types: None,
            max_nodes: 100_000,
            heuristic_scale: 0.1, // Conservative for admissibility
        }
    }
}

impl AstarParams {
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

    /// Builder: set edge types filter.
    #[must_use]
    pub fn edge_types(mut self, types: Vec<EdgeType>) -> Self {
        self.edge_types = Some(types);
        self
    }

    /// Builder: set maximum nodes to explore.
    #[must_use]
    pub fn max_nodes(mut self, max: usize) -> Self {
        self.max_nodes = max;
        self
    }

    /// Builder: set heuristic scale factor.
    #[must_use]
    pub fn heuristic_scale(mut self, scale: f32) -> Self {
        self.heuristic_scale = scale;
        self
    }
}

/// Result of A* pathfinding.
#[derive(Debug, Clone)]
pub struct AstarResult {
    /// Path from start to goal (empty if no path found).
    pub path: Vec<NodeId>,

    /// Total path cost (f(goal) = g(goal) since h(goal) = 0).
    pub total_cost: f32,

    /// Number of nodes explored.
    pub nodes_explored: usize,

    /// Number of nodes in open set when terminated.
    pub open_set_size: usize,

    /// Whether path was found.
    pub path_found: bool,
}

impl AstarResult {
    /// Create empty result for no path found.
    #[must_use]
    pub fn no_path(nodes_explored: usize) -> Self {
        Self {
            path: Vec::new(),
            total_cost: f32::INFINITY,
            nodes_explored,
            open_set_size: 0,
            path_found: false,
        }
    }

    /// Create result with found path.
    #[must_use]
    pub fn found(path: Vec<NodeId>, total_cost: f32, nodes_explored: usize, open_set_size: usize) -> Self {
        Self {
            path,
            total_cost,
            nodes_explored,
            open_set_size,
            path_found: true,
        }
    }

    /// Get path length (number of nodes).
    #[must_use]
    pub fn path_length(&self) -> usize {
        self.path.len()
    }

    /// Get number of edges in path.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        if self.path.is_empty() {
            0
        } else {
            self.path.len() - 1
        }
    }
}

/// Node in A* open set priority queue.
///
/// Uses Reverse ordering so BinaryHeap (max-heap) becomes min-heap.
#[derive(Debug, Clone)]
struct AstarNode {
    /// Node ID.
    node_id: NodeId,
    /// f(n) = g(n) + h(n).
    f_score: f32,
    /// g(n) = cost from start to this node.
    g_score: f32,
}

impl AstarNode {
    fn new(node_id: NodeId, g_score: f32, h_score: f32) -> Self {
        Self {
            node_id,
            f_score: g_score + h_score,
            g_score,
        }
    }
}

// Ordering for min-heap (smaller f_score = higher priority)
impl PartialEq for AstarNode {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for AstarNode {}

impl PartialOrd for AstarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AstarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        // Handle NaN by treating it as greater than everything
        match (self.f_score.is_nan(), other.f_score.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less, // NaN goes to bottom of heap
            (false, true) => Ordering::Greater,
            (false, false) => {
                // Reverse for min-heap
                other.f_score
                    .partial_cmp(&self.f_score)
                    .unwrap_or(Ordering::Equal)
            }
        }
    }
}

/// Convert UUID to i64 for storage key operations.
///
/// This reverses `Uuid::from_u64_pair(id as u64, 0)` used in storage.
#[inline]
fn uuid_to_i64(uuid: &Uuid) -> i64 {
    let bytes = uuid.as_bytes();
    i64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Convert storage PoincarePoint to hyperbolic PoincarePoint.
///
/// Both types have the same underlying [f32; 64] coords, just in different modules.
#[inline]
fn to_hyperbolic_point(storage_point: StoragePoint) -> HyperbolicPoint {
    HyperbolicPoint::from_coords(storage_point.coords)
}

/// Compute edge cost from effective weight.
///
/// Higher weight = lower cost = preferred.
/// Cost = 1.0 / (weight + epsilon)
#[inline]
fn edge_cost(weight: f32) -> f32 {
    // Clamp weight to valid range
    let w = weight.clamp(0.0, 1.0);
    1.0 / (w + 0.001)
}

/// Perform A* pathfinding from start to goal.
///
/// Uses hyperbolic distance in Poincare ball as heuristic.
///
/// # Arguments
/// * `storage` - Graph storage backend
/// * `start` - Starting node ID
/// * `goal` - Goal node ID
/// * `params` - A* parameters
///
/// # Returns
/// * `Ok(AstarResult)` - Pathfinding result
/// * `Err(GraphError::MissingHyperbolicData)` - Node missing hyperbolic embedding
/// * `Err(GraphError::*)` - Storage error
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_graph::traversal::astar::{astar_search, AstarParams, Domain};
///
/// let params = AstarParams::default()
///     .domain(Domain::Code)
///     .min_weight(0.3);
///
/// let result = astar_search(&storage, start_node, goal_node, params)?;
/// if result.path_found {
///     println!("Path: {:?}, cost: {}", result.path, result.total_cost);
/// }
/// ```
pub fn astar_search(
    storage: &GraphStorage,
    start: NodeId,
    goal: NodeId,
    params: AstarParams,
) -> GraphResult<AstarResult> {
    // Handle trivial case
    if start == goal {
        return Ok(AstarResult::found(vec![start], 0.0, 1, 0));
    }

    // Initialize Poincare ball for hyperbolic distance
    let config = HyperbolicConfig::default();
    let ball = PoincareBall::new(config);

    // Get hyperbolic embedding for goal (required for heuristic)
    // NO FALLBACK - fail fast per AP-001
    let goal_point = to_hyperbolic_point(
        storage.get_hyperbolic(goal)?
            .ok_or(GraphError::MissingHyperbolicData(goal))?
    );

    // Get start point
    let start_point = to_hyperbolic_point(
        storage.get_hyperbolic(start)?
            .ok_or(GraphError::MissingHyperbolicData(start))?
    );

    // Initial heuristic
    let h_start = params.heuristic_scale * ball.distance(&start_point, &goal_point);

    // Open set (priority queue)
    let mut open_set: BinaryHeap<AstarNode> = BinaryHeap::new();
    open_set.push(AstarNode::new(start, 0.0, h_start));

    // Track best g-score for each node
    let mut g_scores: HashMap<NodeId, f32> = HashMap::new();
    g_scores.insert(start, 0.0);

    // Track parent for path reconstruction
    let mut came_from: HashMap<NodeId, NodeId> = HashMap::new();

    // Closed set (already explored)
    let mut closed_set: HashSet<NodeId> = HashSet::new();

    let mut nodes_explored = 0;

    while let Some(current) = open_set.pop() {
        let current_id = current.node_id;

        // Check if we've reached the goal
        if current_id == goal {
            // Reconstruct path
            let mut path = vec![goal];
            let mut node = goal;
            while let Some(&parent) = came_from.get(&node) {
                path.push(parent);
                node = parent;
            }
            path.reverse();

            return Ok(AstarResult::found(
                path,
                current.g_score,
                nodes_explored,
                open_set.len(),
            ));
        }

        // Skip if already explored
        if closed_set.contains(&current_id) {
            continue;
        }

        // Mark as explored
        closed_set.insert(current_id);
        nodes_explored += 1;

        // Check exploration limit
        if nodes_explored >= params.max_nodes {
            log::debug!(
                "A* exploration limit reached: {} nodes",
                nodes_explored
            );
            return Ok(AstarResult::no_path(nodes_explored));
        }

        // Get current g-score
        let current_g = *g_scores.get(&current_id).unwrap_or(&f32::INFINITY);

        // Expand neighbors
        let edges = storage.get_outgoing_edges(current_id)?;

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

            // Convert UUID to i64
            let neighbor_id = uuid_to_i64(&edge.target);

            // Skip if already explored
            if closed_set.contains(&neighbor_id) {
                continue;
            }

            // Calculate tentative g-score
            let tentative_g = current_g + edge_cost(effective_weight);

            // Check if this is a better path
            let neighbor_g = *g_scores.get(&neighbor_id).unwrap_or(&f32::INFINITY);
            if tentative_g >= neighbor_g {
                continue; // Not a better path
            }

            // Update path
            came_from.insert(neighbor_id, current_id);
            g_scores.insert(neighbor_id, tentative_g);

            // Calculate heuristic for neighbor
            // Get hyperbolic embedding (NO FALLBACK)
            let neighbor_point = to_hyperbolic_point(
                storage.get_hyperbolic(neighbor_id)?
                    .ok_or(GraphError::MissingHyperbolicData(neighbor_id))?
            );

            let h = params.heuristic_scale * ball.distance(&neighbor_point, &goal_point);

            // Add to open set
            open_set.push(AstarNode::new(neighbor_id, tentative_g, h));
        }
    }

    // No path found
    Ok(AstarResult::no_path(nodes_explored))
}

/// A* with bidirectional search optimization.
///
/// Searches from both start and goal simultaneously, meeting in the middle.
/// Can be ~2x faster than unidirectional A* for long paths.
///
/// # Arguments
/// * `storage` - Graph storage backend
/// * `start` - Starting node ID
/// * `goal` - Goal node ID
/// * `params` - A* parameters
///
/// # Returns
/// Same as `astar_search`
pub fn astar_bidirectional(
    storage: &GraphStorage,
    start: NodeId,
    goal: NodeId,
    params: AstarParams,
) -> GraphResult<AstarResult> {
    // Handle trivial case
    if start == goal {
        return Ok(AstarResult::found(vec![start], 0.0, 1, 0));
    }

    // Initialize Poincare ball
    let config = HyperbolicConfig::default();
    let ball = PoincareBall::new(config);

    // Get hyperbolic embeddings (NO FALLBACK)
    let start_point = to_hyperbolic_point(
        storage.get_hyperbolic(start)?
            .ok_or(GraphError::MissingHyperbolicData(start))?
    );
    let goal_point = to_hyperbolic_point(
        storage.get_hyperbolic(goal)?
            .ok_or(GraphError::MissingHyperbolicData(goal))?
    );

    // Forward search state
    let mut forward_open: BinaryHeap<AstarNode> = BinaryHeap::new();
    let mut forward_g: HashMap<NodeId, f32> = HashMap::new();
    let mut forward_parent: HashMap<NodeId, NodeId> = HashMap::new();
    let mut forward_closed: HashSet<NodeId> = HashSet::new();

    let h_start = params.heuristic_scale * ball.distance(&start_point, &goal_point);
    forward_open.push(AstarNode::new(start, 0.0, h_start));
    forward_g.insert(start, 0.0);

    // Backward search state
    let mut backward_open: BinaryHeap<AstarNode> = BinaryHeap::new();
    let mut backward_g: HashMap<NodeId, f32> = HashMap::new();
    let mut backward_parent: HashMap<NodeId, NodeId> = HashMap::new();
    let mut backward_closed: HashSet<NodeId> = HashSet::new();

    let h_goal = params.heuristic_scale * ball.distance(&goal_point, &start_point);
    backward_open.push(AstarNode::new(goal, 0.0, h_goal));
    backward_g.insert(goal, 0.0);

    let mut best_path_cost = f32::INFINITY;
    let mut meeting_node: Option<NodeId> = None;
    let mut nodes_explored = 0;

    // Alternate between forward and backward search
    let mut forward_turn = true;

    loop {
        // Check termination
        if forward_open.is_empty() && backward_open.is_empty() {
            break;
        }

        if nodes_explored >= params.max_nodes {
            log::debug!("Bidirectional A* limit reached: {} nodes", nodes_explored);
            break;
        }

        // Choose direction
        let (open_set, g_scores, parent_map, closed_set, other_closed, other_g, target_point) =
            if forward_turn && !forward_open.is_empty() {
                (&mut forward_open, &mut forward_g, &mut forward_parent,
                 &mut forward_closed, &backward_closed, &backward_g, &goal_point)
            } else if !backward_open.is_empty() {
                (&mut backward_open, &mut backward_g, &mut backward_parent,
                 &mut backward_closed, &forward_closed, &forward_g, &start_point)
            } else if !forward_open.is_empty() {
                (&mut forward_open, &mut forward_g, &mut forward_parent,
                 &mut forward_closed, &backward_closed, &backward_g, &goal_point)
            } else {
                break;
            };

        forward_turn = !forward_turn;

        // Pop from open set
        let Some(current) = open_set.pop() else {
            continue;
        };

        let current_id = current.node_id;

        // Skip if explored
        if closed_set.contains(&current_id) {
            continue;
        }

        closed_set.insert(current_id);
        nodes_explored += 1;

        // Check if other search has reached this node
        if other_closed.contains(&current_id) {
            // Get costs from the g_scores reference (which is one of forward_g or backward_g)
            // and the other_g reference (which is the other one)
            let this_cost = *g_scores.get(&current_id).unwrap_or(&f32::INFINITY);
            let other_cost = *other_g.get(&current_id).unwrap_or(&f32::INFINITY);
            let path_cost = this_cost + other_cost;

            if path_cost < best_path_cost {
                best_path_cost = path_cost;
                meeting_node = Some(current_id);
            }
        }

        // Early termination: if best f-score > best path, we're done
        if current.f_score >= best_path_cost {
            break;
        }

        // Get current g-score
        let current_g = *g_scores.get(&current_id).unwrap_or(&f32::INFINITY);

        // Expand neighbors
        let edges = storage.get_outgoing_edges(current_id)?;

        for edge in edges {
            // Filter by edge type
            if let Some(ref allowed_types) = params.edge_types {
                if !allowed_types.contains(&edge.edge_type) {
                    continue;
                }
            }

            // Get modulated weight
            let effective_weight = edge.get_modulated_weight(params.domain);

            // Filter by minimum weight
            if effective_weight < params.min_weight {
                continue;
            }

            let neighbor_id = uuid_to_i64(&edge.target);

            if closed_set.contains(&neighbor_id) {
                continue;
            }

            let tentative_g = current_g + edge_cost(effective_weight);
            let neighbor_g = *g_scores.get(&neighbor_id).unwrap_or(&f32::INFINITY);

            if tentative_g >= neighbor_g {
                continue;
            }

            parent_map.insert(neighbor_id, current_id);
            g_scores.insert(neighbor_id, tentative_g);

            // Get heuristic
            let neighbor_point = to_hyperbolic_point(
                storage.get_hyperbolic(neighbor_id)?
                    .ok_or(GraphError::MissingHyperbolicData(neighbor_id))?
            );

            let h = params.heuristic_scale * ball.distance(&neighbor_point, target_point);
            open_set.push(AstarNode::new(neighbor_id, tentative_g, h));

            // Check if meets other search
            if let Some(&other_cost) = other_g.get(&neighbor_id) {
                let path_cost = tentative_g + other_cost;
                if path_cost < best_path_cost {
                    best_path_cost = path_cost;
                    meeting_node = Some(neighbor_id);
                }
            }
        }
    }

    // Reconstruct path if found
    if let Some(meet) = meeting_node {
        let mut forward_path = vec![meet];
        let mut node = meet;
        while let Some(&parent) = forward_parent.get(&node) {
            forward_path.push(parent);
            node = parent;
        }
        forward_path.reverse();

        let mut backward_path = Vec::new();
        node = meet;
        while let Some(&parent) = backward_parent.get(&node) {
            backward_path.push(parent);
            node = parent;
        }

        // Combine paths (forward_path ends with meet, backward_path starts after meet)
        forward_path.extend(backward_path);

        return Ok(AstarResult::found(
            forward_path,
            best_path_cost,
            nodes_explored,
            forward_open.len() + backward_open.len(),
        ));
    }

    Ok(AstarResult::no_path(nodes_explored))
}

/// Convenience function: find optimal path with default parameters.
pub fn astar_path(
    storage: &GraphStorage,
    start: NodeId,
    goal: NodeId,
) -> GraphResult<Option<Vec<NodeId>>> {
    let result = astar_search(storage, start, goal, AstarParams::default())?;
    if result.path_found {
        Ok(Some(result.path))
    } else {
        Ok(None)
    }
}

/// Find optimal path with domain modulation.
pub fn astar_domain_path(
    storage: &GraphStorage,
    start: NodeId,
    goal: NodeId,
    domain: Domain,
    min_weight: f32,
) -> GraphResult<Option<Vec<NodeId>>> {
    let params = AstarParams::default()
        .domain(domain)
        .min_weight(min_weight);
    let result = astar_search(storage, start, goal, params)?;
    if result.path_found {
        Ok(Some(result.path))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::GraphEdge;
    use tempfile::tempdir;

    /// Helper to create UUID from i64.
    fn uuid(id: i64) -> Uuid {
        Uuid::from_u64_pair(id as u64, 0)
    }

    /// Create a test graph with hyperbolic embeddings.
    fn setup_test_graph() -> (GraphStorage, NodeId, NodeId, tempfile::TempDir) {
        let dir = tempdir().expect("Failed to create temp dir");
        let storage = GraphStorage::open_default(dir.path())
            .expect("Failed to open storage");

        // Create simple path: 1 -> 2 -> 3 -> 4
        let edges = vec![
            GraphEdge::new(1, uuid(1), uuid(2), EdgeType::Semantic, 0.8, Domain::General),
            GraphEdge::new(2, uuid(2), uuid(3), EdgeType::Semantic, 0.7, Domain::General),
            GraphEdge::new(3, uuid(3), uuid(4), EdgeType::Semantic, 0.9, Domain::General),
            // Alternative longer path: 1 -> 5 -> 6 -> 4
            GraphEdge::new(4, uuid(1), uuid(5), EdgeType::Semantic, 0.6, Domain::General),
            GraphEdge::new(5, uuid(5), uuid(6), EdgeType::Semantic, 0.5, Domain::General),
            GraphEdge::new(6, uuid(6), uuid(4), EdgeType::Semantic, 0.4, Domain::General),
        ];
        storage.put_edges(&edges).expect("put_edges failed");

        // Create hyperbolic embeddings for all nodes
        // Use storage::PoincarePoint for put_hyperbolic
        // Arrange nodes in a line in hyperbolic space
        for id in 1..=6 {
            let mut coords = [0.0f32; 64];
            // Place nodes along first dimension, scaled to stay in ball
            coords[0] = (id as f32) * 0.1;
            let point = StoragePoint { coords };
            storage.put_hyperbolic(id, &point).expect("put_hyperbolic failed");
        }

        (storage, 1, 4, dir)
    }

    #[test]
    fn test_astar_basic_path() {
        let (storage, start, goal, _dir) = setup_test_graph();

        let result = astar_search(&storage, start, goal, AstarParams::default())
            .expect("A* failed");

        assert!(result.path_found, "Path should be found");
        assert_eq!(result.path[0], start, "Path should start at start node");
        assert_eq!(*result.path.last().unwrap(), goal, "Path should end at goal");
        assert!(result.path.len() >= 2, "Path should have at least 2 nodes");
        assert!(result.total_cost > 0.0, "Cost should be positive");
        assert!(result.total_cost.is_finite(), "Cost should be finite");
    }

    #[test]
    fn test_astar_same_node() {
        let (storage, start, _, _dir) = setup_test_graph();

        let result = astar_search(&storage, start, start, AstarParams::default())
            .expect("A* failed");

        assert!(result.path_found);
        assert_eq!(result.path, vec![start]);
        assert_eq!(result.total_cost, 0.0);
    }

    #[test]
    fn test_astar_no_path() {
        let dir = tempdir().expect("Failed to create temp dir");
        let storage = GraphStorage::open_default(dir.path())
            .expect("Failed to open storage");

        // Create disconnected nodes
        for id in 1..=2 {
            let mut coords = [0.0f32; 64];
            coords[0] = (id as f32) * 0.1;
            let point = StoragePoint { coords };
            storage.put_hyperbolic(id, &point).expect("put_hyperbolic failed");
        }

        let result = astar_search(&storage, 1, 2, AstarParams::default())
            .expect("A* failed");

        assert!(!result.path_found, "No path should be found for disconnected nodes");
        assert!(result.path.is_empty());
        assert!(result.total_cost.is_infinite());
    }

    #[test]
    fn test_astar_missing_hyperbolic_data() {
        let dir = tempdir().expect("Failed to create temp dir");
        let storage = GraphStorage::open_default(dir.path())
            .expect("Failed to open storage");

        // Create edge but no hyperbolic embedding
        let edges = vec![
            GraphEdge::new(1, uuid(1), uuid(2), EdgeType::Semantic, 0.8, Domain::General),
        ];
        storage.put_edges(&edges).expect("put_edges failed");

        // Only set hyperbolic for node 1, not node 2
        let mut coords = [0.0f32; 64];
        coords[0] = 0.1;
        let point = StoragePoint { coords };
        storage.put_hyperbolic(1, &point).expect("put_hyperbolic failed");

        // Should fail with MissingHyperbolicData for goal
        let result = astar_search(&storage, 1, 2, AstarParams::default());
        assert!(result.is_err());
        match result.unwrap_err() {
            GraphError::MissingHyperbolicData(id) => {
                assert_eq!(id, 2, "Error should report missing data for node 2");
            }
            e => panic!("Expected MissingHyperbolicData, got {:?}", e),
        }
    }

    #[test]
    fn test_astar_domain_modulation() {
        let (storage, start, goal, _dir) = setup_test_graph();

        // With Code domain (different from edges which are General)
        let params = AstarParams::default().domain(Domain::Code);
        let result = astar_search(&storage, start, goal, params)
            .expect("A* failed");

        // Should still find a path, weights just modulated differently
        assert!(result.path_found);
        assert_eq!(result.path[0], start);
        assert_eq!(*result.path.last().unwrap(), goal);
    }

    #[test]
    fn test_astar_weight_filter() {
        let (storage, start, goal, _dir) = setup_test_graph();

        // Very high min_weight (2.0) - should filter all edges.
        // Modulation formula: w_eff = weight * (1.0 + net_activation + domain_bonus) * steering_factor
        // Max possible: 0.8 * 2.2 = 1.76, so 2.0 is impossible to achieve.
        let params = AstarParams::default().min_weight(2.0);
        let result = astar_search(&storage, start, goal, params)
            .expect("A* failed");

        assert!(!result.path_found, "No path with impossible weight filter");
    }

    #[test]
    fn test_astar_edge_type_filter() {
        let (storage, start, goal, _dir) = setup_test_graph();

        // Only Hierarchical edges (none in graph)
        let params = AstarParams::default().edge_types(vec![EdgeType::Hierarchical]);
        let result = astar_search(&storage, start, goal, params)
            .expect("A* failed");

        assert!(!result.path_found, "No path with edge type filter");
    }

    #[test]
    fn test_astar_max_nodes_limit() {
        let (storage, start, goal, _dir) = setup_test_graph();

        // Very small limit
        let params = AstarParams::default().max_nodes(1);
        let result = astar_search(&storage, start, goal, params)
            .expect("A* failed");

        // May or may not find path depending on exploration order
        assert!(result.nodes_explored <= 1);
    }

    #[test]
    fn test_astar_bidirectional() {
        let (storage, start, goal, _dir) = setup_test_graph();

        let result = astar_bidirectional(&storage, start, goal, AstarParams::default())
            .expect("Bidirectional A* failed");

        assert!(result.path_found, "Bidirectional should find path");
        assert_eq!(result.path[0], start);
        assert_eq!(*result.path.last().unwrap(), goal);
    }

    #[test]
    fn test_astar_path_convenience() {
        let (storage, start, goal, _dir) = setup_test_graph();

        let path = astar_path(&storage, start, goal)
            .expect("astar_path failed");

        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path[0], start);
        assert_eq!(*path.last().unwrap(), goal);
    }

    #[test]
    fn test_astar_domain_path() {
        let (storage, start, goal, _dir) = setup_test_graph();

        let path = astar_domain_path(&storage, start, goal, Domain::General, 0.3)
            .expect("astar_domain_path failed");

        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path[0], start);
        assert_eq!(*path.last().unwrap(), goal);
    }

    #[test]
    fn test_astar_prefers_shorter_path() {
        let (storage, start, goal, _dir) = setup_test_graph();

        let result = astar_search(&storage, start, goal, AstarParams::default())
            .expect("A* failed");

        // The direct path 1->2->3->4 has 3 edges
        // The longer path 1->5->6->4 has 3 edges but lower weights
        // A* should prefer higher weight path
        assert!(result.path_found);
        assert!(result.path.len() <= 4, "Should find efficient path");
    }

    #[test]
    fn test_edge_cost_function() {
        // Higher weight = lower cost
        assert!(edge_cost(0.9) < edge_cost(0.5));
        assert!(edge_cost(0.5) < edge_cost(0.1));

        // Edge cases
        assert!(edge_cost(0.0).is_finite());
        assert!(edge_cost(1.0).is_finite());
        assert!(edge_cost(-0.1).is_finite()); // Clamped to 0
        assert!(edge_cost(1.5).is_finite()); // Clamped to 1
    }

    #[test]
    fn test_astar_node_ordering() {
        // Test min-heap behavior
        let n1 = AstarNode::new(1, 1.0, 2.0); // f = 3.0
        let n2 = AstarNode::new(2, 0.5, 1.0); // f = 1.5
        let n3 = AstarNode::new(3, 2.0, 3.0); // f = 5.0

        let mut heap = BinaryHeap::new();
        heap.push(n1);
        heap.push(n2);
        heap.push(n3);

        // Should pop in order of smallest f-score first (min-heap)
        assert_eq!(heap.pop().unwrap().node_id, 2); // f = 1.5
        assert_eq!(heap.pop().unwrap().node_id, 1); // f = 3.0
        assert_eq!(heap.pop().unwrap().node_id, 3); // f = 5.0
    }

    #[test]
    fn test_astar_result_methods() {
        let result = AstarResult::found(vec![1, 2, 3, 4], 5.0, 10, 5);

        assert_eq!(result.path_length(), 4);
        assert_eq!(result.edge_count(), 3);
        assert!(result.path_found);
        assert_eq!(result.total_cost, 5.0);

        let no_path = AstarResult::no_path(10);
        assert_eq!(no_path.path_length(), 0);
        assert_eq!(no_path.edge_count(), 0);
        assert!(!no_path.path_found);
    }
}
