---
id: "M04-T26"
title: "Add EdgeType::Contradicts Variant and ContradictionType Enum"
description: |
  Add CONTRADICTS variant to EdgeType enum and define ContradictionType enum.
  This is REQUIRED for contradiction detection in M04-T21.
  EdgeType should now have: Semantic, Temporal, Causal, Hierarchical, Contradicts.
  Update all match statements, as_u8/from_u8, and default_weight to handle the new variant.
layer: "logic"
status: "complete"
priority: "high"
estimated_hours: 2
sequence: 25
depends_on:
  - "M04-T14a"
  - "M04-T15"
spec_refs:
  - "TECH-GRAPH-004 Section 4.1"
  - "REQ-KG-063"
files_to_create:
  - path: "crates/context-graph-core/src/marblestone/contradiction_type.rs"
    description: "ContradictionType enum definition"
files_to_modify:
  - path: "crates/context-graph-core/src/marblestone/edge_type.rs"
    description: "Add Contradicts variant to EdgeType enum"
  - path: "crates/context-graph-core/src/marblestone/mod.rs"
    description: "Re-export ContradictionType"
  - path: "crates/context-graph-graph/src/storage/edges.rs"
    description: "Add GraphEdge::contradiction() factory method"
test_file: "crates/context-graph-core/src/marblestone/tests_edge_type.rs"
---

## ✅ IMPLEMENTATION COMPLETE

**Verified by sherlock-holmes forensic audit: 2026-01-04**
**Verdict: INNOCENT (All requirements verified)**

---

## Source of Truth (ACTUAL Implementation)

| Component | File | Status |
|-----------|------|--------|
| `EdgeType::Contradicts` | `crates/context-graph-core/src/marblestone/edge_type.rs:24` | ✅ Implemented |
| `ContradictionType` enum | `crates/context-graph-core/src/marblestone/contradiction_type.rs` | ✅ Implemented |
| `ContradictionType` re-export | `crates/context-graph-core/src/marblestone/mod.rs:15` | ✅ Implemented |
| `GraphEdge::contradiction()` | `crates/context-graph-graph/src/storage/edges.rs:234` | ✅ Implemented |

---

## Implemented Public API

### EdgeType (edge_type.rs)

```rust
pub enum EdgeType {
    Semantic,      // as_u8 = 0, default_weight = 0.5
    Temporal,      // as_u8 = 1, default_weight = 0.7
    Causal,        // as_u8 = 2, default_weight = 0.8
    Hierarchical,  // as_u8 = 3, default_weight = 0.9
    Contradicts,   // as_u8 = 4, default_weight = 0.3
}

impl EdgeType {
    pub fn all() -> [EdgeType; 5];
    pub fn as_u8(&self) -> u8;
    pub fn from_u8(value: u8) -> Option<Self>;
    pub fn default_weight(&self) -> f32;
    pub fn is_contradiction(&self) -> bool;
    pub fn is_symmetric(&self) -> bool;
    pub fn is_directional(&self) -> bool;
    pub fn description(&self) -> &'static str;
    pub fn reverse_description(&self) -> &'static str;
}
```

### ContradictionType (contradiction_type.rs)

```rust
pub enum ContradictionType {
    DirectOpposition,      // severity = 1.0
    LogicalInconsistency,  // severity = 0.8
    TemporalConflict,      // severity = 0.7
    CausalConflict,        // severity = 0.6
}

impl ContradictionType {
    pub fn all() -> [ContradictionType; 4];
    pub fn severity(&self) -> f32;
    pub fn description(&self) -> &'static str;
    pub fn as_u8(&self) -> u8;
    pub fn from_u8(value: u8) -> Option<Self>;
}
```

### GraphEdge::contradiction() (storage/edges.rs)

```rust
impl GraphEdge {
    /// Creates contradiction edge with inhibitory-heavy NT profile
    /// NT weights: excitatory=0.2, inhibitory=0.7, modulatory=0.1
    pub fn contradiction(
        id: EdgeId,
        source: NodeId,
        target: NodeId,
        confidence: f32,
    ) -> Self;

    pub fn is_contradiction(&self) -> bool;
}
```

---

## Test Results (Verified)

| Test Suite | Count | Status |
|------------|-------|--------|
| EdgeType tests | 57 | ✅ All pass |
| ContradictionType tests | 26 | ✅ All pass |
| GraphEdge::contradiction tests | 25 | ✅ All pass |
| **Total** | **108** | ✅ |

---

## Implementation Details

### EdgeType::Contradicts Characteristics

- **Value**: `as_u8() = 4`
- **Default Weight**: `0.3` (low to suppress accidental traversal)
- **Symmetric**: `true` (A contradicts B implies B contradicts A)
- **Serde**: Serializes as `"contradicts"` (snake_case)

### NT Weight Formula (Constitution Reference)

Contradiction edges use inhibitory-heavy profile:
```
effective_weight = base_weight × (1 + excitatory - inhibitory + 0.5×modulatory)
                 = 0.3 × (1 + 0.2 - 0.7 + 0.5×0.1)
                 = 0.3 × 0.55
                 = 0.165
```

This suppresses contradiction edge traversal during normal graph exploration.

---

## FAIL FAST Verification

| Scenario | Expected | Verified |
|----------|----------|----------|
| `EdgeType::from_u8(4)` | `Some(Contradicts)` | ✅ |
| `EdgeType::from_u8(5)` | `None` | ✅ |
| `EdgeType::from_u8(255)` | `None` | ✅ |
| `ContradictionType::from_u8(4)` | `None` | ✅ |
| Invalid serde input | Error (no silent failures) | ✅ |

---

## Modification History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-04 | AI Agent | Initial spec with correct file paths |
| 2026-01-04 | AI Agent | Implementation complete, sherlock verification INNOCENT |
