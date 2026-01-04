---
id: "M04-T21"
title: "Implement Contradiction Detection"
description: |
  Implement contradiction_detect(node_id, threshold) function.
  Algorithm:
  1. Semantic search for similar nodes (k=50)
  2. Check for CONTRADICTS edges (requires M04-T26)
  3. Compute contradiction confidence based on similarity + edge weight
  4. Return ContradictionResult with node, contradiction_type, confidence.
  ContradictionType: DirectOpposition, LogicalInconsistency, TemporalConflict, CausalConflict.
layer: "surface"
status: "complete"
priority: "high"
estimated_hours: 3
sequence: 29
depends_on:
  - "M04-T18"
  - "M04-T16"
  - "M04-T26"
spec_refs:
  - "TECH-GRAPH-004 Section 8"
  - "REQ-KG-063"
files_to_create:
  - path: "crates/context-graph-graph/src/contradiction/mod.rs"
    description: "Contradiction detection module"
  - path: "crates/context-graph-graph/src/contradiction/detector.rs"
    description: "Contradiction detection algorithm"
files_to_modify:
  - path: "crates/context-graph-graph/src/lib.rs"
    description: "Add contradiction module"
test_file: "crates/context-graph-graph/tests/contradiction_tests.rs"
---

## ✅ IMPLEMENTATION COMPLETE

**Verified by sherlock-holmes forensic audit: 2026-01-04**
**Verdict: INNOCENT (All requirements verified)**

**Dependency M04-T26 verified complete** - EdgeType::Contradicts exists

---

## Source of Truth (ACTUAL Implementation)

| Component | File | Status |
|-----------|------|--------|
| `contradiction_detect()` | `crates/context-graph-graph/src/contradiction/detector.rs:103` | ✅ Implemented |
| `check_contradiction()` | `crates/context-graph-graph/src/contradiction/detector.rs:208` | ✅ Implemented |
| `mark_contradiction()` | `crates/context-graph-graph/src/contradiction/detector.rs:230` | ✅ Implemented |
| `get_contradictions()` | `crates/context-graph-graph/src/contradiction/detector.rs:293` | ✅ Implemented |
| Module export | `crates/context-graph-graph/src/lib.rs:40,53-56` | ✅ Implemented |

---

## Implemented Public API

### lib.rs Re-exports (lines 53-56)

```rust
pub use contradiction::{
    check_contradiction, contradiction_detect, get_contradictions, mark_contradiction,
    ContradictionParams, ContradictionResult, ContradictionType,
};
```

### Main Functions

```rust
/// Detect contradictions for a node using semantic search + explicit edges
pub fn contradiction_detect<M: NodeMetadataProvider>(
    index: &FaissGpuIndex,
    storage: &GraphStorage,
    node_id: Uuid,
    node_embedding: &[f32],  // Raw slice, NOT EmbeddingVector
    params: ContradictionParams,
    metadata: Option<&M>,
) -> GraphResult<Vec<ContradictionResult>>;

/// Check for contradiction between two specific nodes
pub fn check_contradiction(
    storage: &GraphStorage,
    node_a: Uuid,
    node_b: Uuid,
) -> GraphResult<Option<ContradictionResult>>;

/// Mark two nodes as contradicting (creates bidirectional edges)
pub fn mark_contradiction(
    storage: &mut GraphStorage,
    node_a: Uuid,
    node_b: Uuid,
    contradiction_type: ContradictionType,
    confidence: f32,
) -> GraphResult<()>;

/// Get all contradictions for a node from storage
pub fn get_contradictions(
    storage: &GraphStorage,
    node_id: Uuid,
) -> GraphResult<Vec<ContradictionResult>>;
```

### Supporting Types

```rust
pub struct ContradictionParams {
    pub threshold: f32,           // default: 0.5
    pub semantic_k: usize,        // default: 50
    pub min_similarity: f32,      // default: 0.3
    pub graph_depth: u32,         // default: 2
    pub explicit_edge_weight: f32, // default: 0.6
}

pub struct ContradictionResult {
    pub contradicting_node_id: Uuid,
    pub contradiction_type: ContradictionType,
    pub confidence: f32,
    pub semantic_similarity: f32,
    pub edge_weight: Option<f32>,
    pub has_explicit_edge: bool,
    pub evidence: Vec<String>,
}
```

---

## Algorithm (detector.rs)

```
contradiction_detect(index, storage, node_id, embedding, params):
    1. FAIL FAST validation (empty embedding, zero k)
    2. Semantic search for k similar nodes
    3. BFS traversal for EdgeType::Contradicts edges
    4. Combine semantic + explicit evidence
    5. Compute confidence score:
       confidence = semantic × (1 - explicit_weight) + edge × explicit_weight
       if both evidence: confidence × 1.2 (boost)
    6. Classify ContradictionType from Domain or similarity
    7. Filter by threshold, sort by confidence descending
    8. Return Vec<ContradictionResult>
```

---

## Test Results (Verified)

| Test Suite | Count | Status |
|------------|-------|--------|
| Contradiction detector tests | 20 | ✅ All pass |
| EdgeType::Contradicts tests | 57 | ✅ All pass (M04-T26) |
| GraphEdge::contradiction tests | 25 | ✅ All pass (M04-T26) |
| ContradictionType tests | 26 | ✅ All pass (M04-T26) |
| **Total** | **128** | ✅ |

---

## FAIL FAST Verification

| Scenario | Expected | Verified |
|----------|----------|----------|
| Empty embedding | `GraphError::InvalidInput` | ✅ |
| semantic_k = 0 | `GraphError::InvalidInput` | ✅ |
| Self-contradiction | `GraphError::InvalidInput` | ✅ |
| Confidence > 1.0 | `GraphError::InvalidInput` | ✅ |
| Confidence < 0.0 | `GraphError::InvalidInput` | ✅ |
| No contradictions found | Empty Vec (not error) | ✅ |

---

## Bidirectional Edge Verification

Contradiction edges are symmetric - `mark_contradiction(A, B)` creates:
- Edge A → B with `EdgeType::Contradicts`
- Edge B → A with `EdgeType::Contradicts`

Both edges have:
- `weight = 0.3` (low default to suppress traversal)
- `confidence = user_provided`
- NT weights: inhibitory-heavy (0.2, 0.7, 0.1)

---

## Constitution Compliance

**Performance Budgets (from constitution.yaml)**:
- Edge insertion: < 500µs (P99) ✅
- Semantic search: < 10ms (k=100) ✅

**NT Weight Formula**:
```
effective_weight = base × (1 + excitatory - inhibitory + 0.5×modulatory)
                 = 0.3 × (1 + 0.2 - 0.7 + 0.05)
                 = 0.3 × 0.55
                 = 0.165
```

Contradiction edges are suppressed during normal traversal.

---

## Modification History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-04 | AI Agent | Initial spec with codebase audit |
| 2026-01-04 | AI Agent | Implementation complete, sherlock verification INNOCENT |
