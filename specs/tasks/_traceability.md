# Task Traceability Matrix

---

## Coverage: SPEC-ATC-001 -> TASK-ATC-P2-*

| Spec Item | Type | Task ID | Status |
|-----------|------|---------|--------|
| REQ-ATC-001: GWT thresholds use ATC | requirement | TASK-ATC-P2-003 | Ready |
| REQ-ATC-002: UTL thresholds use ATC | requirement | TASK-ATC-P2-006 | Ready |
| REQ-ATC-003: Dream layer thresholds use ATC | requirement | TASK-ATC-P2-005 | Ready |
| REQ-ATC-004: Neuromodulation bounds use ATC | requirement | DEFERRED | Evaluate |
| REQ-ATC-005: Layer thresholds use ATC | requirement | TASK-ATC-P2-004 | Ready |
| REQ-ATC-006: Domain context support | requirement | TASK-ATC-P2-002 | Ready |
| REQ-ATC-007: General domain fallback | requirement | TASK-ATC-P2-002 | Ready |
| REQ-ATC-008: EWMA tracking observable | requirement | TASK-ATC-P2-008 | Ready |
| NFR-ATC-001: Performance <1us | non_functional | TASK-ATC-P2-008 | Ready |
| NFR-ATC-002: No allocations per access | non_functional | TASK-ATC-P2-008 | Ready |
| NFR-ATC-003: Backward compatibility | non_functional | TASK-ATC-P2-008 | Ready |
| US-ATC-001: Developer threshold visibility | user_story | TASK-ATC-P2-001 | Ready |
| US-ATC-002: Domain-adaptive behavior | user_story | TASK-ATC-P2-002 | Ready |
| US-ATC-003: Threshold drift tracking | user_story | TASK-ATC-P2-008 | Ready |

### ATC Task Dependency Graph

```
TASK-ATC-P2-001 (Discovery) ─┬─> TASK-ATC-P2-002 (Extend Struct) ─┬─> TASK-ATC-P2-003 (GWT)
                              │                                     ├─> TASK-ATC-P2-004 (Layers)
                              │                                     ├─> TASK-ATC-P2-005 (Dream)
                              │                                     ├─> TASK-ATC-P2-006 (Johari)
                              │                                     └─> TASK-ATC-P2-007 (Autonomous)
                              │
                              └──────────────────────────────────────────────────────────────┐
                                                                                              ↓
                                              TASK-ATC-P2-003 ─┐
                                              TASK-ATC-P2-004 ─┤
                                              TASK-ATC-P2-005 ─┼─> TASK-ATC-P2-008 (Validation)
                                              TASK-ATC-P2-006 ─┤
                                              TASK-ATC-P2-007 ─┘
```

### ATC Test Coverage Matrix

| Test Case ID | Requirement | Task ID | Status |
|--------------|-------------|---------|--------|
| TC-ATC-001 | REQ-ATC-006 | TASK-ATC-P2-002 | Pending |
| TC-ATC-002 | REQ-ATC-006 | TASK-ATC-P2-002 | Pending |
| TC-ATC-003 | REQ-ATC-006 | TASK-ATC-P2-002 | Pending |
| TC-ATC-004 | REQ-ATC-007 | TASK-ATC-P2-002 | Pending |
| TC-ATC-010 | REQ-ATC-001 | TASK-ATC-P2-003 | Pending |
| TC-ATC-011 | REQ-ATC-003 | TASK-ATC-P2-005 | Pending |
| TC-ATC-012 | REQ-ATC-005 | TASK-ATC-P2-004 | Pending |
| TC-ATC-013 | REQ-ATC-008 | TASK-ATC-P2-008 | Pending |
| TC-ATC-020 | NFR-ATC-003 | TASK-ATC-P2-008 | Pending |
| TC-ATC-021 | NFR-ATC-003 | TASK-ATC-P2-008 | Pending |

---

## Coverage: SPEC-NEURO-001 -> TASK-NEURO-P2-*

| Spec Item | Type | Task ID | Status |
|-----------|------|---------|--------|
| FR-NEURO-001-01: Goal Progress Event Handler (Manager) | requirement | TASK-NEURO-P2-001 | Ready |
| FR-NEURO-001-02: Dopamine Modulator Goal Progress Method | requirement | TASK-NEURO-P2-001 | Ready |
| FR-NEURO-001-03: Steering-to-Neuromodulation Integration | requirement | TASK-NEURO-P2-002 | Ready |
| FR-NEURO-001-04: Sensitivity Configuration | requirement | TASK-NEURO-P2-001 | Ready |
| NFR-NEURO-001-01: Performance (<1ms latency) | non_functional | TASK-NEURO-P2-001 | Ready |
| NFR-NEURO-001-02: Compatibility (existing behavior) | non_functional | TASK-NEURO-P2-001 | Ready |
| NFR-NEURO-001-03: Observability (logging) | non_functional | TASK-NEURO-P2-001 | Ready |
| Section 8.1: DA Cascade to Subsystems | cascade | TASK-NEURO-P2-001 | Ready |
| Section 8.2: Cross-NT Interactions | cascade | TASK-NEURO-P2-003 | Ready |
| Section 8.3: UTL Integration | integration | TASK-NEURO-P2-003 | Ready |
| Section 8.4: GWT Integration | integration | TASK-NEURO-P2-003 | Ready |
| Section 9: MCP Integration | integration | TASK-NEURO-P2-002 | Ready |
| DA_GOAL_SENSITIVITY constant | constant | TASK-NEURO-P2-001 | Ready |
| DopamineModulator::on_goal_progress() | method | TASK-NEURO-P2-001 | Ready |
| NeuromodulationManager::on_goal_progress() | method | TASK-NEURO-P2-001 | Ready |
| NeuromodulationManager::on_goal_progress_with_cascades() | method | TASK-NEURO-P2-003 | Ready |
| cascade module constants | module | TASK-NEURO-P2-003 | Ready |
| CascadeReport struct | struct | TASK-NEURO-P2-003 | Ready |
| Handlers.neuromod_manager field | field | TASK-NEURO-P2-002 | Ready |
| try_modulate_da() method | method | TASK-NEURO-P2-002 | Ready |

---

## Uncovered Items

| Item | Reason | Resolution |
|------|--------|------------|
| None | - | All items covered |

---

## Test Coverage Matrix

| Test Case ID | Requirement | Task ID | Status |
|--------------|-------------|---------|--------|
| TC-NEURO-001-01 | FR-NEURO-001-02 | TASK-NEURO-P2-001 | Pending |
| TC-NEURO-001-02 | FR-NEURO-001-02 | TASK-NEURO-P2-001 | Pending |
| TC-NEURO-001-03 | FR-NEURO-001-02 | TASK-NEURO-P2-001 | Pending |
| TC-NEURO-001-04 | FR-NEURO-001-02 | TASK-NEURO-P2-001 | Pending |
| TC-NEURO-001-05 | FR-NEURO-001-01 | TASK-NEURO-P2-001 | Pending |
| TC-NEURO-001-06 | FR-NEURO-001-03 | TASK-NEURO-P2-002 | Pending |
| TC-CASCADE-01 | Section 8.2 Row 1 | TASK-NEURO-P2-003 | Pending |
| TC-CASCADE-02 | Section 8.2 Row 2 | TASK-NEURO-P2-003 | Pending |
| TC-CASCADE-03 | Section 8.2 Row 3 | TASK-NEURO-P2-003 | Pending |
| TC-CASCADE-04 | CascadeReport accuracy | TASK-NEURO-P2-003 | Pending |
| TC-MCP-01 | MCP lock handling | TASK-NEURO-P2-002 | Pending |
| TC-MCP-02 | MCP NaN handling | TASK-NEURO-P2-002 | Pending |

---

## Validation Checklist

### Completeness
- [x] All functional requirements have tasks
- [x] All non-functional requirements have tasks
- [x] All methods specified have implementation tasks
- [x] All error states covered in test plan
- [x] Task dependencies form valid DAG (no cycles)
- [x] Layer ordering correct (logic -> surface)

### Test Coverage
- [x] Unit tests specified for all methods
- [x] Integration tests identified (future task)
- [x] Edge cases documented (6 cases)
- [x] Error handling tested (NaN, bounds)

### Traceability
- [x] All tasks trace to requirements
- [x] All requirements trace to specification
- [x] All test cases trace to requirements
- [x] Gap analysis reference included

---

## Dependency Validation

### Valid Order Check

```
TASK-NEURO-P2-001 (logic) --> TASK-NEURO-P2-002 (surface)
                          \-> TASK-NEURO-P2-003 (logic)
```

The dependency order is valid because:
1. TASK-NEURO-P2-001 creates the `on_goal_progress()` API in the core crate
2. TASK-NEURO-P2-002 consumes that API from the MCP crate
3. TASK-NEURO-P2-003 extends the API with cascade effects
4. Logic layer must be complete before surface layer integration
5. P2-002 and P2-003 can execute in parallel after P2-001

### No Circular Dependencies

Dependency graph is acyclic:
- TASK-NEURO-P2-001: No dependencies
- TASK-NEURO-P2-002: Depends only on TASK-NEURO-P2-001
- TASK-NEURO-P2-003: Depends only on TASK-NEURO-P2-001

---

## Gap Analysis Alignment

| Gap Analysis Item | Priority | Task Coverage | Status |
|-------------------|----------|---------------|--------|
| REFINEMENT 3: Neuromodulation Feedback Loop | P2 | TASK-NEURO-P2-001, TASK-NEURO-P2-002, TASK-NEURO-P2-003 | Ready |
| Direct DA modulation from steering | P2 | TASK-NEURO-P2-001 | Ready |
| MCP handler integration | P2 | TASK-NEURO-P2-002 | Ready |
| Cascade effects to other NTs | P3 | TASK-NEURO-P2-003 | Ready |

---

## Approval Status

| Check | Status | Verified By | Date |
|-------|--------|-------------|------|
| All requirements covered | Pass | Specification Agent | 2026-01-11 |
| Dependencies valid | Pass | Specification Agent | 2026-01-11 |
| Test plan complete | Pass | Specification Agent | 2026-01-11 |
| Traceability complete | Pass | Specification Agent | 2026-01-11 |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-11 | Initial creation | Specification Agent |
| 2026-01-11 | Added TASK-NEURO-P2-002, TASK-NEURO-P2-003; Updated traceability matrix | Agent 7 - Neuromodulation Specialist |
| 2026-01-11 | Added cascade effects coverage; Updated test coverage matrix | Agent 7 - Neuromodulation Specialist |
