# Embedding Pipeline Task Dependency Graph Index

<task_index id="INDEX-EMB-001" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Document ID** | INDEX-EMB-001 |
| **Title** | Task Dependency Graph Index |
| **Version** | 1.0 |
| **Created** | 2026-01-06 |
| **Status** | Complete |
| **Total Tasks** | 25 |
| **Layers** | 3 (Foundation, Logic, Surface) |

---

## Task Dependency Graph

```mermaid
graph TD
    subgraph Foundation["Foundation Layer (TASK-EMB-001 to TASK-EMB-010)"]
        T001[TASK-EMB-001<br/>Fix Dimension Constants]
        T002[TASK-EMB-002<br/>ProjectionMatrix Struct]
        T003[TASK-EMB-003<br/>ProjectionError Enum]
        T004[TASK-EMB-004<br/>Quantization Structs]
        T005[TASK-EMB-005<br/>Storage Types]
        T006[TASK-EMB-006<br/>WarmLoadResult Struct]
        T007[TASK-EMB-007<br/>Consolidated Errors]
        T008[TASK-EMB-008<br/>Update SparseVector]
        T009[TASK-EMB-009<br/>Weight File Spec]
        T010[TASK-EMB-010<br/>Golden Fixtures]
    end

    subgraph Logic["Logic Layer (TASK-EMB-011 to TASK-EMB-020)"]
        T011[TASK-EMB-011<br/>ProjectionMatrix::load]
        T012[TASK-EMB-012<br/>ProjectionMatrix::project]
        T013[TASK-EMB-013<br/>Real Weight Loading]
        T014[TASK-EMB-014<br/>Real VRAM Allocation]
        T015[TASK-EMB-015<br/>Real Inference Validation]
        T016[TASK-EMB-016<br/>PQ-8 Quantization]
        T017[TASK-EMB-017<br/>Float8 Quantization]
        T018[TASK-EMB-018<br/>Binary Quantization]
        T019[TASK-EMB-019<br/>Remove Stub Mode]
        T020[TASK-EMB-020<br/>QuantizationRouter]
    end

    subgraph Surface["Surface Layer (TASK-EMB-021 to TASK-EMB-025)"]
        T021[TASK-EMB-021<br/>Integrate ProjectionMatrix]
        T022[TASK-EMB-022<br/>Storage Backend]
        T023[TASK-EMB-023<br/>Multi-Space Search]
        T024[TASK-EMB-024<br/>Update MCP Handlers]
        T025[TASK-EMB-025<br/>Integration Tests]
    end

    %% Foundation Dependencies
    T001 --> T002
    T001 --> T004
    T001 --> T006
    T001 --> T008
    T002 --> T003
    T002 --> T009
    T003 --> T007
    T004 --> T005
    T005 --> T007
    T006 --> T007

    %% Logic Dependencies
    T002 --> T011
    T003 --> T011
    T011 --> T012
    T006 --> T013
    T013 --> T014
    T014 --> T015
    T010 --> T015
    T014 --> T019
    T004 --> T016
    T004 --> T017
    T004 --> T018
    T016 --> T020
    T017 --> T020
    T018 --> T020

    %% Surface Dependencies
    T012 --> T021
    T020 --> T022
    T022 --> T023
    T021 --> T024
    T023 --> T024
    T024 --> T025

    %% Style
    classDef foundation fill:#e1f5fe,stroke:#01579b
    classDef logic fill:#fff3e0,stroke:#e65100
    classDef surface fill:#e8f5e9,stroke:#1b5e20
    classDef critical fill:#ffebee,stroke:#b71c1c

    class T001,T002,T003,T004,T005,T006,T007,T008,T009,T010 foundation
    class T011,T012,T013,T014,T015,T016,T017,T018,T019,T020 logic
    class T021,T022,T023,T024,T025 surface
```

---

## Execution Order Table

### Phase 1: Foundation Layer

| Order | Task ID | Title | Dependencies | Complexity | Parallel Group |
|-------|---------|-------|--------------|------------|----------------|
| 1 | TASK-EMB-001 | Fix Dimension Constants | None | Low | A |
| 1 | TASK-EMB-010 | Create Golden Fixtures | None | Low | A |
| 2 | TASK-EMB-002 | Create ProjectionMatrix Struct | 001 | Medium | B |
| 2 | TASK-EMB-004 | Create Quantization Structs | 001 | Medium | B |
| 2 | TASK-EMB-006 | Create WarmLoadResult Struct | 001 | Medium | B |
| 2 | TASK-EMB-008 | Update SparseVector Struct | 001 | Low | B |
| 3 | TASK-EMB-003 | Create ProjectionError Enum | 002 | Low | C |
| 3 | TASK-EMB-005 | Create Storage Types | 004 | Medium | C |
| 3 | TASK-EMB-009 | Create Weight File Spec | 002 | Low | C |
| 4 | TASK-EMB-007 | Create Consolidated Errors | 003, 005, 006 | Medium | D |

### Phase 2: Logic Layer

| Order | Task ID | Title | Dependencies | Complexity | Parallel Group |
|-------|---------|-------|--------------|------------|----------------|
| 5 | TASK-EMB-011 | Implement ProjectionMatrix::load() | 002, 003 | High | E |
| 5 | TASK-EMB-013 | Implement Real Weight Loading | 006 | High | E |
| 5 | TASK-EMB-016 | Implement PQ-8 Quantization | 004 | High | E |
| 5 | TASK-EMB-017 | Implement Float8 Quantization | 004 | Medium | E |
| 5 | TASK-EMB-018 | Implement Binary Quantization | 004 | Low | E |
| 6 | TASK-EMB-012 | Implement ProjectionMatrix::project() | 011 | High | F |
| 6 | TASK-EMB-014 | Implement Real VRAM Allocation | 013 | High | F |
| 6 | TASK-EMB-020 | Implement QuantizationRouter | 016, 017, 018 | Medium | F |
| 7 | TASK-EMB-015 | Implement Real Inference Validation | 014, 010 | High | G |
| 7 | TASK-EMB-019 | Remove Stub Mode from Preflight | 014 | Low | G |

### Phase 3: Surface Layer

| Order | Task ID | Title | Dependencies | Complexity | Parallel Group |
|-------|---------|-------|--------------|------------|----------------|
| 8 | TASK-EMB-021 | Integrate ProjectionMatrix into SparseModel | 012 | Medium | H |
| 8 | TASK-EMB-022 | Implement Storage Backend | 020 | High | H |
| 9 | TASK-EMB-023 | Implement Multi-Space Search | 022 | High | I |
| 10 | TASK-EMB-024 | Update MCP Handlers | 021, 023 | Medium | J |
| 11 | TASK-EMB-025 | Integration Tests | All | High | K |

---

## Critical Path

The critical path (longest sequence of dependent tasks):

```
TASK-EMB-001 (Foundation)
    |
    v
TASK-EMB-002 (Foundation)
    |
    v
TASK-EMB-003 (Foundation)
    |
    v
TASK-EMB-011 (Logic)
    |
    v
TASK-EMB-012 (Logic)
    |
    v
TASK-EMB-021 (Surface)
    |
    v
TASK-EMB-024 (Surface)
    |
    v
TASK-EMB-025 (Surface)
```

**Critical Path Length:** 8 tasks

---

## Parallel Execution Groups

### Group A (No Dependencies - Start Immediately)
- TASK-EMB-001: Fix Dimension Constants
- TASK-EMB-010: Create Golden Reference Fixtures

### Group B (Depends on TASK-EMB-001)
- TASK-EMB-002: Create ProjectionMatrix Struct
- TASK-EMB-004: Create Quantization Structs
- TASK-EMB-006: Create WarmLoadResult Struct
- TASK-EMB-008: Update SparseVector Struct

### Group C (Mixed Dependencies)
- TASK-EMB-003: Create ProjectionError Enum (depends on 002)
- TASK-EMB-005: Create Storage Types (depends on 004)
- TASK-EMB-009: Create Weight File Spec (depends on 002)

### Group D (Consolidation)
- TASK-EMB-007: Create Consolidated Errors (depends on 003, 005, 006)

### Group E (Logic - Independent Tracks)
- TASK-EMB-011: ProjectionMatrix::load() (depends on 002, 003)
- TASK-EMB-013: Real Weight Loading (depends on 006)
- TASK-EMB-016: PQ-8 Quantization (depends on 004)
- TASK-EMB-017: Float8 Quantization (depends on 004)
- TASK-EMB-018: Binary Quantization (depends on 004)

### Group F (Logic - Second Wave)
- TASK-EMB-012: ProjectionMatrix::project() (depends on 011)
- TASK-EMB-014: Real VRAM Allocation (depends on 013)
- TASK-EMB-020: QuantizationRouter (depends on 016, 017, 018)

### Group G (Logic - Third Wave)
- TASK-EMB-015: Real Inference Validation (depends on 014, 010)
- TASK-EMB-019: Remove Stub Mode (depends on 014)

### Group H (Surface - Initial)
- TASK-EMB-021: Integrate ProjectionMatrix (depends on 012)
- TASK-EMB-022: Storage Backend (depends on 020)

### Group I (Surface - Search)
- TASK-EMB-023: Multi-Space Search (depends on 022)

### Group J (Surface - Handlers)
- TASK-EMB-024: Update MCP Handlers (depends on 021, 023)

### Group K (Surface - Final)
- TASK-EMB-025: Integration Tests (depends on all)

---

## Status Tracking Template

### Foundation Layer Status

| Task ID | Title | Status | Assignee | Start | Complete |
|---------|-------|--------|----------|-------|----------|
| TASK-EMB-001 | Fix Dimension Constants | [ ] Pending | - | - | - |
| TASK-EMB-002 | Create ProjectionMatrix Struct | [ ] Pending | - | - | - |
| TASK-EMB-003 | Create ProjectionError Enum | [ ] Pending | - | - | - |
| TASK-EMB-004 | Create Quantization Structs | [ ] Pending | - | - | - |
| TASK-EMB-005 | Create Storage Types | [ ] Pending | - | - | - |
| TASK-EMB-006 | Create WarmLoadResult Struct | [ ] Pending | - | - | - |
| TASK-EMB-007 | Create Consolidated Errors | [ ] Pending | - | - | - |
| TASK-EMB-008 | Update SparseVector Struct | [ ] Pending | - | - | - |
| TASK-EMB-009 | Create Weight File Spec | [ ] Pending | - | - | - |
| TASK-EMB-010 | Create Golden Reference Fixtures | [ ] Pending | - | - | - |

### Logic Layer Status

| Task ID | Title | Status | Assignee | Start | Complete |
|---------|-------|--------|----------|-------|----------|
| TASK-EMB-011 | Implement ProjectionMatrix::load() | [ ] Pending | - | - | - |
| TASK-EMB-012 | Implement ProjectionMatrix::project() | [ ] Pending | - | - | - |
| TASK-EMB-013 | Implement Real Weight Loading | [ ] Pending | - | - | - |
| TASK-EMB-014 | Implement Real VRAM Allocation | [ ] Pending | - | - | - |
| TASK-EMB-015 | Implement Real Inference Validation | [ ] Pending | - | - | - |
| TASK-EMB-016 | Implement PQ-8 Quantization | [ ] Pending | - | - | - |
| TASK-EMB-017 | Implement Float8 Quantization | [ ] Pending | - | - | - |
| TASK-EMB-018 | Implement Binary Quantization | [ ] Pending | - | - | - |
| TASK-EMB-019 | Remove Stub Mode from Preflight | [ ] Pending | - | - | - |
| TASK-EMB-020 | Implement QuantizationRouter | [ ] Pending | - | - | - |

### Surface Layer Status

| Task ID | Title | Status | Assignee | Start | Complete |
|---------|-------|--------|----------|-------|----------|
| TASK-EMB-021 | Integrate ProjectionMatrix into SparseModel | [ ] Pending | - | - | - |
| TASK-EMB-022 | Implement Storage Backend | [ ] Pending | - | - | - |
| TASK-EMB-023 | Implement Multi-Space Search | [ ] Pending | - | - | - |
| TASK-EMB-024 | Update MCP Handlers | [ ] Pending | - | - | - |
| TASK-EMB-025 | Integration Tests | [ ] Pending | - | - | - |

---

## Complexity Summary

| Complexity | Count | Tasks |
|------------|-------|-------|
| **Low** | 6 | 001, 003, 008, 009, 010, 018, 019 |
| **Medium** | 9 | 002, 004, 005, 006, 007, 017, 020, 021, 024 |
| **High** | 10 | 011, 012, 013, 014, 015, 016, 022, 023, 025 |

---

## Layer Summary

| Layer | Task Range | Count | Primary Focus |
|-------|------------|-------|---------------|
| **Foundation** | 001-010 | 10 | Data structures, types, constants |
| **Logic** | 011-020 | 10 | Business logic, algorithms, CUDA |
| **Surface** | 021-025 | 5 | API integration, handlers, tests |

---

## Document References

| Document | Description | Path |
|----------|-------------|------|
| SPEC-EMB-001 | Master Functional Specification | `SPEC-EMB-001-master-functional.md` |
| TECH-EMB-001 | Sparse Projection Tech Spec | `TECH-EMB-001-sparse-projection.md` |
| TECH-EMB-002 | Warm Loading Tech Spec | `TECH-EMB-002-warm-loading.md` |
| TECH-EMB-003 | Quantization Tech Spec | `TECH-EMB-003-quantization.md` |
| TECH-EMB-004 | Storage Module Tech Spec | `TECH-EMB-004-storage-module.md` |
| TASK-EMB-FOUNDATION | Foundation Layer Tasks | `TASK-EMB-FOUNDATION.md` |
| TASK-EMB-LOGIC | Logic Layer Tasks | `TASK-EMB-LOGIC.md` |
| TASK-EMB-SURFACE | Surface Layer Tasks | `TASK-EMB-SURFACE.md` |
| TRACEABILITY-MATRIX | Full Traceability | `TRACEABILITY-MATRIX.md` |
| TASK-INDEX | This Document | `TASK-INDEX.md` |

---

## Quick Reference

### Start Here (No Dependencies)
```
TASK-EMB-001: Fix Dimension Constants
TASK-EMB-010: Create Golden Reference Fixtures
```

### Blocked On Foundation
```
Logic Layer: All tasks blocked until Foundation complete
Surface Layer: All tasks blocked until Logic complete
```

### Final Deliverable
```
TASK-EMB-025: Integration Tests (depends on ALL other tasks)
```

---

## Memory Key

Store index reference:
```
contextgraph/embedding-issues/task-index-complete
```

</task_index>
