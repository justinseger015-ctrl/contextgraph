# M04-T17a: Implement A* Hyperbolic Traversal

## Task Metadata
| Field | Value |
|-------|-------|
| **Task ID** | M04-T17a |
| **Module** | 04 - Graph Layer |
| **Sequence** | 24 |
| **Status** | COMPLETE ✓ |
| **Priority** | High |
| **Estimated Effort** | Medium-High |
| **Dependencies** | M04-T17 (DFS Traversal - COMPLETE) |
| **Traceability** | REQ-KG-061 (Graph traversal algorithms) |
| **Completed** | 2026-01-04 |
| **Implementation** | `crates/context-graph-graph/src/traversal/astar.rs` (968 lines) |
| **Verified By** | Sherlock-Holmes Forensic Agent |

## MANDATORY READING BEFORE STARTING

**You MUST read these files before implementation:**
1. `/home/cabdru/contextgraph/docs2/constitution.yaml` - System constitution with NT weight formula
2. `/home/cabdru/contextgraph/docs2/improved_agent_prompt.md` - Agent development guidelines
3. `/home/cabdru/contextgraph/crates/context-graph-graph/src/traversal/bfs.rs` - Reference for patterns
4. `/home/cabdru/contextgraph/crates/context-graph-graph/src/hyperbolic/mobius.rs` - PoincareBall.distance() implementation

## Objective

Implement A* pathfinding algorithm using **hyperbolic distance** as an admissible heuristic for optimal path finding in the knowledge graph. The Poincare ball distance provides a lower bound on actual path length, making it an ideal heuristic.

## Critical Implementation Requirements

### ABSOLUTE RULES

1. **USE CORRECT API**: Use `storage.get_outgoing_edges(node_id)` NOT `get_adjacency()`. The old task document is WRONG.

2. **NO BACKWARDS COMPATIBILITY**: The implementation must work correctly or fail fast. No fallbacks to Dijkstra if hyperbolic data is missing - return an error instead.

3. **FAIL FAST**: Use `GraphError` for all error conditions. If hyperbolic coordinates are missing for required nodes, fail with clear error.

4. **NO MOCK DATA IN TESTS**: All tests must use real graph data with real hyperbolic coordinates.

5. **ADMISSIBLE HEURISTIC**: The hyperbolic distance heuristic must NEVER overestimate the true cost.

## Technical Specification

### File Location
```
crates/context-graph-graph/src/traversal/astar.rs
```

### Module Integration
Update `crates/context-graph-graph/src/traversal/mod.rs`:
```rust
// Add after dfs module declaration
pub mod astar;

// Add to re-exports
pub use astar::{
    astar_traverse, astar_shortest_path,
    AstarParams, AstarResult,
};
```

### Required Imports
```rust
use crate::error::{GraphError, GraphResult};
use crate::storage::GraphStorage;
use crate::hyperbolic::{PoincareBall, PoincarePoint};
use context_graph_core::marblestone::edge_type::EdgeType;
use context_graph_core::marblestone::graph_edge::GraphEdge;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use uuid::Uuid;
```

### CRITICAL Type Information

**NodeId**: `pub type NodeId = i64;` - Storage key type
**GraphEdge.source/target**: `Uuid` - Must convert with uuid_to_i64()

```rust
fn uuid_to_i64(uuid: &Uuid) -> i64 {
    let bytes = uuid.as_bytes();
    i64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}
```

### Hyperbolic Distance Function (FROM mobius.rs)

The `PoincareBall` struct has a `distance()` method:
```rust
// In crates/context-graph-graph/src/hyperbolic/mobius.rs
impl PoincareBall {
    /// Calculate hyperbolic distance: d(x,y) = (2/√c) * arctanh(√c * ||(-x) ⊕ y||)
    pub fn distance(&self, x: &PoincarePoint, y: &PoincarePoint) -> f32
}
```

Use this for the heuristic. Default curvature c=1.0.

### AstarParams Structure
```rust
#[derive(Debug, Clone)]
pub struct AstarParams {
    /// Maximum nodes to expand before giving up
    pub max_expansions: usize,
    /// Maximum path length allowed
    pub max_path_length: usize,
    /// Edge types to traverse (None = all)
    pub edge_types: Option<Vec<EdgeType>>,
    /// Minimum edge weight threshold
    pub min_weight: f32,
    /// Domain for NT weight modulation
    pub domain: Domain,
    /// Curvature for Poincare ball (default 1.0)
    pub curvature: f32,
}

impl Default for AstarParams {
    fn default() -> Self {
        Self {
            max_expansions: 100_000,
            max_path_length: 100,
            edge_types: None,
            min_weight: 0.0,
            domain: Domain::General,
            curvature: 1.0,
        }
    }
}

// Builder methods: max_expansions(), max_path_length(), edge_types(), min_weight(), domain(), curvature()
```

### AstarResult Structure
```rust
#[derive(Debug, Clone)]
pub struct AstarResult {
    /// Path from start to goal (inclusive), None if no path
    pub path: Option<Vec<NodeId>>,
    /// Total cost of the path (sum of edge costs)
    pub total_cost: f32,
    /// Number of nodes expanded during search
    pub nodes_expanded: usize,
    /// Whether search was truncated by limits
    pub truncated: bool,
}

impl AstarResult {
    pub fn found(&self) -> bool { self.path.is_some() }
    pub fn path_length(&self) -> usize { self.path.as_ref().map(|p| p.len().saturating_sub(1)).unwrap_or(0) }
}
```

### Priority Queue Node
```rust
#[derive(Debug, Clone)]
struct AstarNode {
    node_id: NodeId,
    g_cost: f32,  // Cost from start to this node
    f_cost: f32,  // g_cost + heuristic
}

// Implement Ord for min-heap (lowest f_cost first)
impl Ord for AstarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap
        other.f_cost.partial_cmp(&self.f_cost).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for AstarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for AstarNode {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for AstarNode {}
```

### Core A* Algorithm

```rust
pub fn astar_traverse(
    storage: &GraphStorage,
    start: NodeId,
    goal: NodeId,
    params: AstarParams,
) -> GraphResult<AstarResult> {
    // Early return if start == goal
    if start == goal {
        return Ok(AstarResult {
            path: Some(vec![start]),
            total_cost: 0.0,
            nodes_expanded: 0,
            truncated: false,
        });
    }

    // Get hyperbolic coordinates - FAIL if missing
    let start_point = storage.get_hyperbolic_point(start)?
        .ok_or_else(|| GraphError::MissingHyperbolicData(start))?;
    let goal_point = storage.get_hyperbolic_point(goal)?
        .ok_or_else(|| GraphError::MissingHyperbolicData(goal))?;

    let ball = PoincareBall::new(params.curvature);

    let mut open_set: BinaryHeap<AstarNode> = BinaryHeap::new();
    let mut g_costs: HashMap<NodeId, f32> = HashMap::new();
    let mut came_from: HashMap<NodeId, NodeId> = HashMap::new();
    let mut closed_set: HashSet<NodeId> = HashSet::new();
    let mut nodes_expanded: usize = 0;

    // Initialize with start node
    let start_h = ball.distance(&start_point, &goal_point);
    open_set.push(AstarNode {
        node_id: start,
        g_cost: 0.0,
        f_cost: start_h,
    });
    g_costs.insert(start, 0.0);

    while let Some(current) = open_set.pop() {
        // Check expansion limit
        if nodes_expanded >= params.max_expansions {
            return Ok(AstarResult {
                path: None,
                total_cost: f32::INFINITY,
                nodes_expanded,
                truncated: true,
            });
        }

        // Goal reached
        if current.node_id == goal {
            let path = reconstruct_path(&came_from, start, goal);

            if path.len() > params.max_path_length {
                return Ok(AstarResult {
                    path: None,
                    total_cost: f32::INFINITY,
                    nodes_expanded,
                    truncated: true,
                });
            }

            return Ok(AstarResult {
                path: Some(path),
                total_cost: current.g_cost,
                nodes_expanded,
                truncated: false,
            });
        }

        // Skip if already processed
        if closed_set.contains(&current.node_id) {
            continue;
        }

        closed_set.insert(current.node_id);
        nodes_expanded += 1;

        // CORRECT API: get_outgoing_edges NOT get_adjacency
        let edges = storage.get_outgoing_edges(current.node_id)?;

        for edge in edges {
            // Filter by edge type
            if let Some(ref allowed) = params.edge_types {
                if !allowed.contains(&edge.edge_type) {
                    continue;
                }
            }

            // Get NT-modulated weight
            let edge_weight = edge.get_modulated_weight(params.domain);
            if edge_weight < params.min_weight {
                continue;
            }

            // Convert UUID to NodeId
            let neighbor_id = uuid_to_i64(&edge.target);

            if closed_set.contains(&neighbor_id) {
                continue;
            }

            // Edge cost = inverse of weight (higher weight = lower cost)
            let edge_cost = 1.0 / (edge_weight + 0.001); // Avoid division by zero
            let tentative_g = current.g_cost + edge_cost;

            let existing_g = g_costs.get(&neighbor_id).copied().unwrap_or(f32::INFINITY);

            if tentative_g < existing_g {
                came_from.insert(neighbor_id, current.node_id);
                g_costs.insert(neighbor_id, tentative_g);

                // Get hyperbolic heuristic
                let neighbor_point = storage.get_hyperbolic_point(neighbor_id)?
                    .ok_or_else(|| GraphError::MissingHyperbolicData(neighbor_id))?;
                let h = ball.distance(&neighbor_point, &goal_point);

                // Scale heuristic to be admissible (never overestimate)
                let h_scaled = h * 0.1; // Conservative scaling

                open_set.push(AstarNode {
                    node_id: neighbor_id,
                    g_cost: tentative_g,
                    f_cost: tentative_g + h_scaled,
                });
            }
        }
    }

    // No path found
    Ok(AstarResult {
        path: None,
        total_cost: f32::INFINITY,
        nodes_expanded,
        truncated: false,
    })
}

fn reconstruct_path(
    came_from: &HashMap<NodeId, NodeId>,
    start: NodeId,
    goal: NodeId,
) -> Vec<NodeId> {
    let mut path = vec![goal];
    let mut current = goal;

    while current != start {
        if let Some(&prev) = came_from.get(&current) {
            path.push(prev);
            current = prev;
        } else {
            break;
        }
    }

    path.reverse();
    path
}
```

### Convenience Function
```rust
/// Find shortest path between two nodes using A*
pub fn astar_shortest_path(
    storage: &GraphStorage,
    start: NodeId,
    goal: NodeId,
    max_depth: usize,
) -> GraphResult<Option<Vec<NodeId>>> {
    let params = AstarParams::default().max_path_length(max_depth);
    let result = astar_traverse(storage, start, goal, params)?;
    Ok(result.path)
}
```

## Testing Requirements

### Test File Location
```
crates/context-graph-graph/src/traversal/astar.rs (inline tests)
```

### Required Test Cases (NO MOCKS - REAL DATA ONLY)

1. `test_astar_basic_path` - Find path in simple graph with hyperbolic coords
2. `test_astar_same_node` - Start == goal returns single-node path
3. `test_astar_no_path` - Returns None when no path exists
4. `test_astar_optimal_path` - A* finds same optimal path as exhaustive search
5. `test_astar_heuristic_efficiency` - A* expands fewer nodes than BFS/DFS
6. `test_astar_missing_hyperbolic` - Returns GraphError::MissingHyperbolicData
7. `test_astar_edge_type_filter` - Respects edge type filtering
8. `test_astar_weight_threshold` - Respects min_weight with NT modulation
9. `test_astar_max_expansions` - Truncates when hitting expansion limit
10. `test_astar_max_path_length` - Truncates when path exceeds length limit
11. `test_astar_vs_bfs_path` - A* finds same path as BFS shortest path

## Acceptance Criteria

- [x] A* finds optimal path using hyperbolic heuristic - Uses `PoincareBall.distance()`
- [x] Uses BinaryHeap for O(log n) priority queue operations - `std::collections::BinaryHeap`
- [x] Heuristic is admissible (never overestimates) - Scaled by 0.1 for admissibility
- [x] Returns GraphError for missing hyperbolic data (NO fallback) - `MissingHyperbolicData(i64)`
- [x] Respects edge type filtering - Filters by `params.edge_types`
- [x] NT weight modulation via get_modulated_weight() - Used for edge costs
- [x] Uses get_outgoing_edges() (NOT get_adjacency()) - Lines 390, 578
- [x] Path reconstruction works correctly - `reconstruct_path()` function
- [x] All errors return GraphError (no panics) - No unwrap() in production code

## Full State Verification Protocol

### Source of Truth
1. **File Existence**: `crates/context-graph-graph/src/traversal/astar.rs` must exist
2. **Module Export**: `mod.rs` must contain `pub mod astar;` and re-exports
3. **Compilation**: `cargo build --package context-graph-graph` succeeds
4. **Tests**: `cargo test --package context-graph-graph astar` all pass

### Execute & Inspect Commands
```bash
# 1. Verify file exists and has content
ls -la crates/context-graph-graph/src/traversal/astar.rs
wc -l crates/context-graph-graph/src/traversal/astar.rs
# EXPECTED: File exists, 350+ lines

# 2. Verify module export in mod.rs
grep -n "pub mod astar" crates/context-graph-graph/src/traversal/mod.rs
grep -n "pub use astar::" crates/context-graph-graph/src/traversal/mod.rs
# EXPECTED: Both lines exist

# 3. Verify compilation
cargo build --package context-graph-graph 2>&1 | tail -20
# EXPECTED: "Finished" with no errors

# 4. Run tests with output
cargo test --package context-graph-graph astar -- --nocapture 2>&1
# EXPECTED: All tests pass, "test result: ok"

# 5. Verify correct API usage
grep "get_outgoing_edges" crates/context-graph-graph/src/traversal/astar.rs
grep "get_adjacency" crates/context-graph-graph/src/traversal/astar.rs
# EXPECTED: get_outgoing_edges found, get_adjacency NOT found

# 6. Verify hyperbolic distance is used
grep "ball.distance" crates/context-graph-graph/src/traversal/astar.rs
grep "PoincareBall" crates/context-graph-graph/src/traversal/astar.rs
# EXPECTED: Both found

# 7. Verify BinaryHeap is used
grep "BinaryHeap" crates/context-graph-graph/src/traversal/astar.rs
# EXPECTED: Found
```

### Boundary & Edge Case Audit
Test these explicitly and print before/after state:
1. **Start == Goal**: Single-node path, zero cost, zero expansions
2. **Disconnected nodes**: Returns None with nodes_expanded count
3. **Missing hyperbolic data**: Returns GraphError::MissingHyperbolicData
4. **Maximum expansions reached**: Returns truncated=true

### Evidence of Success
```bash
# Create verification log
echo "=== M04-T17a A* Verification $(date) ===" >> /tmp/m04-t17a-verification.log
cargo test --package context-graph-graph astar -- --nocapture 2>&1 >> /tmp/m04-t17a-verification.log
echo "=== Compilation Check ===" >> /tmp/m04-t17a-verification.log
cargo build --package context-graph-graph 2>&1 >> /tmp/m04-t17a-verification.log
cat /tmp/m04-t17a-verification.log
```

## Sherlock Verification (MANDATORY FINAL STEP)

**You MUST spawn a sherlock-holmes subagent before marking complete:**

```
Task: "Forensic verification of M04-T17a A* implementation"
Subagent: sherlock-holmes

Investigation checklist:
1. VERIFY astar.rs exists at correct path with substantial content
2. VERIFY mod.rs exports astar module and public API
3. VERIFY get_outgoing_edges is used (NOT get_adjacency)
4. VERIFY uuid_to_i64 conversion is present and used
5. VERIFY PoincareBall.distance() is used for heuristic
6. VERIFY BinaryHeap is used for priority queue
7. VERIFY GraphError::MissingHyperbolicData is returned (not fallback)
8. RUN cargo build --package context-graph-graph - must succeed
9. RUN cargo test --package context-graph-graph astar - all must pass
10. VERIFY heuristic is admissible (scaled appropriately)
11. CHECK for any clippy warnings
12. CREATE evidence report with all findings
```

**If sherlock finds ANY issues, fix them before marking complete.**

## Git Commit Format
```
feat(graph): implement M04-T17a A* hyperbolic traversal

- Add astar.rs with A* pathfinding using hyperbolic heuristic
- Use PoincareBall.distance() for admissible heuristic
- Implement AstarParams/AstarResult for configuration
- Support edge type filtering and NT weight modulation
- Use get_outgoing_edges (correct API)
- Fail fast on missing hyperbolic data (no fallbacks)
- Add comprehensive test suite with real graph data

Implements: REQ-KG-061
Closes: M04-T17a
```

## Related Tasks
- **M04-T16** (COMPLETE): BFS traversal - Reference for patterns
- **M04-T17** (COMPLETE): DFS traversal - Iterative implementation
- **M04-T22**: Traversal utilities
- **M04-T23**: CUDA-accelerated traversal (future)

---

## Sherlock-Holmes Verification Report

**Case ID**: HOLMES-2026-01-04-TRAVERSAL
**Verdict**: INNOCENT - COMPLETE

### Evidence Summary

| Check | Expected | Actual | Verdict |
|-------|----------|--------|---------|
| File exists | astar.rs at correct path | 968 lines | ✓ INNOCENT |
| Module export | `pub mod astar;` in mod.rs | Present at line 78 | ✓ INNOCENT |
| Hyperbolic heuristic | PoincareBall.distance() | Lines 328, 436 | ✓ INNOCENT |
| BinaryHeap | Priority queue | Imported and used | ✓ INNOCENT |
| Correct API | get_outgoing_edges | Lines 390, 578 | ✓ INNOCENT |
| Fail fast | MissingHyperbolicData | Lines 316-319, 322-325, 431-434 | ✓ INNOCENT |
| NT modulation | get_modulated_weight | Used correctly | ✓ INNOCENT |
| No unwrap() | No unwrap in prod code | Verified | ✓ INNOCENT |
| Build | cargo build succeeds | 0.14s | ✓ INNOCENT |
| Tests | All tests pass | 17 A* tests pass | ✓ INNOCENT |

### Test Results
```
running 17 tests
test traversal::astar::tests::test_astar_basic ... ok
test traversal::astar::tests::test_astar_bidirectional ... ok
test traversal::astar::tests::test_astar_domain_path ... ok
test traversal::astar::tests::test_astar_edge_type_filter ... ok
test traversal::astar::tests::test_astar_heuristic_efficiency ... ok
test traversal::astar::tests::test_astar_max_expansions ... ok
test traversal::astar::tests::test_astar_max_path_length ... ok
test traversal::astar::tests::test_astar_missing_hyperbolic_data ... ok
test traversal::astar::tests::test_astar_no_path ... ok
test traversal::astar::tests::test_astar_optimal_path ... ok
test traversal::astar::tests::test_astar_params_builder ... ok
test traversal::astar::tests::test_astar_path ... ok
test traversal::astar::tests::test_astar_result_methods ... ok
test traversal::astar::tests::test_astar_same_node ... ok
test traversal::astar::tests::test_astar_vs_bfs_path ... ok
test traversal::astar::tests::test_astar_weight_threshold ... ok
test traversal::astar::tests::test_hyperbolic_heuristic ... ok
test result: ok. 17 passed; 0 failed; 0 ignored
```

### Minor Style Suggestions (Non-Blocking)
- Line 266: `max().min()` pattern could use `clamp()`
