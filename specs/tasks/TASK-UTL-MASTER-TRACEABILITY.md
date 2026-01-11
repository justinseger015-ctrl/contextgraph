# UTL Learning Core - Master Task Traceability Matrix

## Overview

This document provides complete traceability from constitution.yaml through specifications to implementation tasks for the UTL (Unified Temporal Learning) system.

**Last Updated:** 2026-01-11
**Completion Status:** 75% -> Target 100%

---

## Constitution-to-Spec-to-Task Traceability

### 1. Delta-SC Computation (compute_delta_sc MCP Tool)

| Constitution Reference | Specification | Task ID | Status |
|------------------------|---------------|---------|--------|
| `gwt_tools: [..., compute_delta_sc]` | SPEC-UTL-001 | TASK-UTL-P1-001 | Ready |
| `delta_sc.ΔS_methods` (13 embedders) | SPEC-UTL-003 | TASK-UTL-P1-003 to 006 | Pending |
| `delta_sc.ΔC: "α×Connectivity + β×ClusterFit + γ×Consistency"` | SPEC-UTL-002 | TASK-UTL-P1-002, 007, 008 | Pending |
| MCP Handler Registration | SPEC-UTL-004 | TASK-UTL-P1-009 | Blocked |

### 2. Entropy (Delta-S) Per Embedder

| Embedder | Constitution Method | Spec | Task | Status |
|----------|---------------------|------|------|--------|
| E1 (Semantic) | GMM+Mahalanobis | SPEC-UTL-003 | Existing | Complete |
| E2-4, E8 | KNN | SPEC-UTL-003 | DefaultKnnEntropy | Complete |
| E5 (Causal) | Asymmetric KNN | SPEC-UTL-003 | Existing | Complete |
| E6, E13 | IDF/Jaccard | SPEC-UTL-003 | Existing | Complete |
| **E7 (Code)** | **GMM+KNN Hybrid** | SPEC-UTL-003 | **TASK-UTL-P1-003** | **Pending** |
| E9 (Binary) | Hamming | SPEC-UTL-003 | Existing | Complete |
| **E10 (Multimodal)** | **Cross-modal KNN** | SPEC-UTL-003 | **TASK-UTL-P1-004** | **Pending** |
| **E11 (Entity)** | **TransE** | SPEC-UTL-003 | **TASK-UTL-P1-005** | **Pending** |
| **E12 (LateInteraction)** | **MaxSim Token** | SPEC-UTL-003 | **TASK-UTL-P1-006** | **Pending** |

### 3. Coherence (Delta-C) Components

| Component | Weight | Specification | Task | Status |
|-----------|--------|---------------|------|--------|
| Connectivity (EdgeAlign) | 0.4 | SPEC-UTL-002 | Existing | Complete |
| **ClusterFit (Silhouette)** | **0.4** | **SPEC-UTL-002** | **TASK-UTL-P1-002, 007** | **Pending** |
| Consistency (Variance) | 0.2 | SPEC-UTL-002 | Existing | Complete |
| Integration | N/A | SPEC-UTL-002 | TASK-UTL-P1-008 | Blocked |

---

## Task Dependency Graph

```
Layer 0 (Types & Interfaces)
├── TASK-UTL-P1-001: ComputeDeltaScRequest/Response types [Ready]
└── TASK-UTL-P1-002: ClusterFit types & config [Ready]

Layer 1 (Algorithms)
├── TASK-UTL-P1-003: HybridGmmKnnEntropy (E7) [Pending]
├── TASK-UTL-P1-004: CrossModalEntropy (E10) [Pending]
├── TASK-UTL-P1-005: TransEEntropy (E11) [Pending]
├── TASK-UTL-P1-006: MaxSimTokenEntropy (E12) [Pending]
└── TASK-UTL-P1-007: ClusterFit silhouette calculation [Blocked by 002]

Layer 2 (Integration)
├── TASK-UTL-P1-008: CoherenceTracker + ClusterFit [Blocked by 007]
└── TASK-UTL-P1-009: MCP Handler registration [Blocked by 001, 008]
```

---

## Execution Order

| Order | Task ID | Title | Depends On | Est. Hours |
|-------|---------|-------|------------|------------|
| 1 | TASK-UTL-P1-001 | Request/Response Types | - | 2-3 |
| 2 | TASK-UTL-P1-002 | ClusterFit Types | - | 2-3 |
| 3a | TASK-UTL-P1-003 | HybridGmmKnnEntropy | - | 4-5 |
| 3b | TASK-UTL-P1-004 | CrossModalEntropy | - | 3-4 |
| 3c | TASK-UTL-P1-005 | TransEEntropy | - | 3-4 |
| 3d | TASK-UTL-P1-006 | MaxSimTokenEntropy | - | 3-4 |
| 4 | TASK-UTL-P1-007 | Silhouette Calculation | 002 | 4-6 |
| 5 | TASK-UTL-P1-008 | CoherenceTracker Integration | 007 | 3-4 |
| 6 | TASK-UTL-P1-009 | MCP Handler | 001, 008 | 3-4 |

**Total Estimated Hours:** 28-37 hours
**Parallelizable:** Tasks 3a-3d can run concurrently

---

## Specification Coverage Matrix

### SPEC-UTL-001: compute_delta_sc Types

| Requirement | Task | Test Coverage |
|-------------|------|---------------|
| REQ-UTL-001-01: Request type with vertex_id | TASK-UTL-P1-001 | Unit test |
| REQ-UTL-001-02: Response with Johari quadrant | TASK-UTL-P1-001 | Unit test |
| REQ-UTL-001-03: 13-embedder array validation | TASK-UTL-P1-001 | Unit test |
| REQ-UTL-001-04: Diagnostic mode | TASK-UTL-P1-001 | Unit test |

### SPEC-UTL-002: ClusterFit Coherence

| Requirement | Task | Test Coverage |
|-------------|------|---------------|
| REQ-UTL-002-01: Compute silhouette coefficient | TASK-UTL-P1-007 | sklearn validation |
| REQ-UTL-002-02: Intra-cluster distance (a) | TASK-UTL-P1-007 | Unit test |
| REQ-UTL-002-03: Inter-cluster distance (b) | TASK-UTL-P1-007 | Unit test |
| REQ-UTL-002-04: Normalize to [0,1] | TASK-UTL-P1-007 | Unit test |
| REQ-UTL-002-05: Three-component formula | TASK-UTL-P1-008 | Integration test |
| REQ-UTL-002-06: Default weights (0.4, 0.4, 0.2) | TASK-UTL-P1-008 | Unit test |
| REQ-UTL-002-07: Configurable weights | TASK-UTL-P1-002, 008 | Unit test |

### SPEC-UTL-003: Specialized Entropy Methods

| Requirement | Task | Test Coverage |
|-------------|------|---------------|
| REQ-UTL-003-01: E7 GMM+KNN hybrid | TASK-UTL-P1-003 | Unit test, GMM validation |
| REQ-UTL-003-02: E7 weight 0.5 + 0.5 | TASK-UTL-P1-003 | Unit test |
| REQ-UTL-003-03: E10 cross-modal KNN | TASK-UTL-P1-004 | Unit test |
| REQ-UTL-003-04: E10 modality detection | TASK-UTL-P1-004 | Unit test |
| REQ-UTL-003-05: E11 TransE distance | TASK-UTL-P1-005 | Unit test |
| REQ-UTL-003-06: E11 L1/L2 norm support | TASK-UTL-P1-005 | Unit test |
| REQ-UTL-003-07: E12 MaxSim aggregation | TASK-UTL-P1-006 | Unit test |
| REQ-UTL-003-08: E12 variable-length tokens | TASK-UTL-P1-006 | Unit test |

### SPEC-UTL-004: MCP Handler Registration

| Requirement | Task | Test Coverage |
|-------------|------|---------------|
| REQ-UTL-004-01: Register in gwt_tools | TASK-UTL-P1-009 | Integration test |
| REQ-UTL-004-02: Deserialize request | TASK-UTL-P1-009 | Unit test |
| REQ-UTL-004-03: Serialize response | TASK-UTL-P1-009 | Unit test |
| REQ-UTL-004-04: Error code mapping | TASK-UTL-P1-009 | Unit test |
| REQ-UTL-004-05: Tracing spans | TASK-UTL-P1-009 | Log inspection |
| REQ-UTL-004-06: JSON Schema exposure | TASK-UTL-P1-009 | Unit test |
| REQ-UTL-004-07: ARCH-05 validation | TASK-UTL-P1-009 | Unit test |

---

## Gap Analysis Alignment

| Gap ID | Description | Resolution | Tasks |
|--------|-------------|------------|-------|
| GAP-1 | compute_delta_sc not registered | Add MCP handler | TASK-UTL-P1-009 |
| GAP-2 | ClusterFit missing from ΔC | Add silhouette calculation | TASK-UTL-P1-002, 007, 008 |
| GAP-3 | E7, E10, E11, E12 use DefaultKnn | Specialized implementations | TASK-UTL-P1-003 to 006 |

---

## File Impact Summary

| File | Tasks | Operation |
|------|-------|-----------|
| `crates/context-graph-mcp/src/types/delta_sc.rs` | 001 | Create |
| `crates/context-graph-utl/src/coherence/cluster_fit.rs` | 002, 007 | Create |
| `crates/context-graph-utl/src/surprise/embedder_entropy/hybrid_gmm_knn.rs` | 003 | Create |
| `crates/context-graph-utl/src/surprise/embedder_entropy/cross_modal.rs` | 004 | Create |
| `crates/context-graph-utl/src/surprise/embedder_entropy/transe.rs` | 005 | Create |
| `crates/context-graph-utl/src/surprise/embedder_entropy/maxsim_token.rs` | 006 | Create |
| `crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs` | 003-006 | Modify |
| `crates/context-graph-utl/src/coherence/tracker.rs` | 008 | Modify |
| `crates/context-graph-mcp/src/handlers/gwt/compute_delta_sc.rs` | 009 | Create |
| `crates/context-graph-mcp/src/handlers/gwt/mod.rs` | 009 | Modify |

---

## Validation Checklist

### Completeness
- [x] All constitution delta_sc methods have tasks
- [x] All 13 embedders have entropy implementation (or task)
- [x] ClusterFit component has full task chain
- [x] MCP handler registration has task
- [x] Task dependencies form valid DAG

### Test Coverage
- [x] Unit tests specified for all new code
- [x] Integration tests for MCP roundtrip
- [x] sklearn validation for silhouette
- [x] Edge cases documented (empty, NaN, etc.)

### Constitution Alignment
- [x] E7: GMM+KNN (TASK-UTL-P1-003)
- [x] E10: Cross-modal KNN (TASK-UTL-P1-004)
- [x] E11: TransE (TASK-UTL-P1-005)
- [x] E12: MaxSim Token (TASK-UTL-P1-006)
- [x] ΔC: 0.4 + 0.4 + 0.2 weights (TASK-UTL-P1-008)

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-11 | Initial creation with full traceability |
| 2026-01-11 | Updated E7 from Jaccard to GMM+KNN per constitution |
| 2026-01-11 | Updated E11 from ExponentialDecay to TransE per constitution |
| 2026-01-11 | Added TASK-UTL-P1-009 for MCP handler registration |
| 2026-01-11 | Created SPEC-UTL-004 for MCP handler specification |
