# Functional Specification: UTL Domain Remediation

## Metadata

| Field | Value |
|-------|-------|
| **ID** | SPEC-UTL-001 |
| **Title** | Johari Quadrant Action Naming Correction |
| **Status** | draft |
| **Owner** | UTL Domain Team |
| **Last Updated** | 2026-01-12 |
| **Related Issues** | ISS-011 |
| **Related Specs** | constitution.yaml v5.0.0, PRD v4.0.0 |
| **Priority** | MEDIUM |
| **Estimated Effort** | 1-2 hours |

---

## 1. Overview

### 1.1 Problem Statement

The Johari quadrant action mapping in the UTL crate incorrectly swaps the recommended actions for the **Blind** and **Unknown** quadrants. This mismatch between the Constitution (authoritative source) and the implementation causes incorrect system behavior when the system determines which action to recommend based on a memory's entropy (deltaS) and coherence (deltaC) values.

### 1.2 Why This Matters

The Johari Window model is central to the UTL (Unified Theory of Learning) framework. When a memory is classified into a quadrant, the system recommends specific actions:

- **Correct routing ensures optimal learning**: A memory in the Blind quadrant (high entropy, low coherence) represents a discovery opportunity that the system should consolidate via dreaming.
- **Incorrect routing causes confusion**: The current implementation suggests EpistemicAction for Blind quadrant items when dream consolidation is more appropriate.
- **Downstream systems depend on correct mapping**: Any code consuming `get_suggested_action()` will receive incorrect recommendations.
- **Documentation/code mismatch erodes trust**: When the code doesn't match the Constitution, it creates debugging confusion and undermines system reliability.

### 1.3 Current State vs Constitution

| Quadrant | deltaS | deltaC | Constitution Action | Current Implementation | Status |
|----------|--------|--------|---------------------|----------------------|--------|
| Open | <0.5 | >0.5 | DirectRecall | DirectRecall | CORRECT |
| Hidden | <0.5 | <0.5 | GetNeighborhood | GetNeighborhood | CORRECT |
| **Blind** | >0.5 | <0.5 | **TriggerDream** | EpistemicAction | **WRONG** |
| **Unknown** | >0.5 | >0.5 | **EpistemicAction** | TriggerDream | **WRONG** |

### 1.4 Constitution Reference

From `constitution.yaml` v5.0.0, Section `utl.johari`:

```yaml
johari:
  Open: "deltaS<0.5, deltaC>0.5 -> DirectRecall"
  Blind: "deltaS>0.5, deltaC<0.5 -> TriggerDream"
  Hidden: "deltaS<0.5, deltaC<0.5 -> GetNeighborhood"
  Unknown: "deltaS>0.5, deltaC>0.5 -> EpistemicAction"
```

---

## 2. User Stories

### US-UTL-001: Correct Action for Blind Quadrant Discovery

**Priority**: must-have

**Narrative**:
As the GWT consciousness system,
I want memories classified as Blind (high entropy, low coherence) to trigger dream consolidation,
So that novel patterns without sufficient coherence get properly integrated during sleep cycles.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-01 | A memory has deltaS=0.7 (>0.5) and deltaC=0.3 (<0.5) | The system calls `get_suggested_action(JohariQuadrant::Blind)` | The result is `SuggestedAction::TriggerDream` |
| AC-02 | A memory is classified as Blind | The cognitive pulse suggests an action | The suggestion is "TriggerDream" or "dream consolidation" |
| AC-03 | The Blind quadrant action is queried | System checks against Constitution | The action matches "TriggerDream" as specified in `utl.johari.Blind` |

### US-UTL-002: Correct Action for Unknown Quadrant Frontier

**Priority**: must-have

**Narrative**:
As the GWT consciousness system,
I want memories classified as Unknown (high entropy, high coherence) to trigger epistemic actions,
So that frontier knowledge (both surprising AND coherent) generates clarifying questions.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-04 | A memory has deltaS=0.7 (>0.5) and deltaC=0.7 (>0.5) | The system calls `get_suggested_action(JohariQuadrant::Unknown)` | The result is `SuggestedAction::EpistemicAction` |
| AC-05 | A memory is classified as Unknown | The cognitive pulse suggests an action | The suggestion is "EpistemicAction" or "epistemic action" |
| AC-06 | The Unknown quadrant action is queried | System checks against Constitution | The action matches "EpistemicAction" as specified in `utl.johari.Unknown` |

### US-UTL-003: Open and Hidden Quadrants Remain Correct

**Priority**: must-have

**Narrative**:
As the UTL system,
I want the already-correct Open and Hidden quadrant mappings to remain unchanged,
So that the fix doesn't introduce regressions.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-07 | A memory has deltaS=0.3, deltaC=0.7 (Open) | System queries action | Result is `SuggestedAction::DirectRecall` |
| AC-08 | A memory has deltaS=0.3, deltaC=0.3 (Hidden) | System queries action | Result is `SuggestedAction::GetNeighborhood` |

---

## 3. Requirements

### 3.1 Functional Requirements

#### REQ-UTL-001: Blind Quadrant MUST Map to TriggerDream

**Description**: When `get_suggested_action(JohariQuadrant::Blind)` is called, the function MUST return `SuggestedAction::TriggerDream`.

**Rationale**: The Constitution specifies that Blind quadrant (deltaS>0.5, deltaC<0.5) represents discovery opportunities where the system knows something novel but lacks coherence. Dream consolidation (Hebbian replay + hyperbolic walk) is the appropriate mechanism to integrate such discoveries.

**Constitution Reference**: `utl.johari.Blind: "deltaS>0.5, deltaC<0.5 -> TriggerDream"`

**Priority**: MUST

**Verification**: Unit test asserting `get_suggested_action(JohariQuadrant::Blind) == SuggestedAction::TriggerDream`

#### REQ-UTL-002: Unknown Quadrant MUST Map to EpistemicAction

**Description**: When `get_suggested_action(JohariQuadrant::Unknown)` is called, the function MUST return `SuggestedAction::EpistemicAction`.

**Rationale**: The Constitution specifies that Unknown quadrant (deltaS>0.5, deltaC>0.5) represents frontier knowledge that is both surprising AND coherent. This indicates the system should ask clarifying questions (epistemic action) to better understand and integrate this frontier.

**Constitution Reference**: `utl.johari.Unknown: "deltaS>0.5, deltaC>0.5 -> EpistemicAction"`

**Priority**: MUST

**Verification**: Unit test asserting `get_suggested_action(JohariQuadrant::Unknown) == SuggestedAction::EpistemicAction`

#### REQ-UTL-003: Action Mapping MUST Match Constitution Exactly

**Description**: The complete action mapping table MUST exactly match the Constitution specification with no deviations.

**Rationale**: The Constitution is the authoritative source for system behavior. Any intentional deviation MUST be documented with explicit rationale. Currently, there is no documented rationale for the swap, so it is treated as a bug.

**Constitution Reference**: `utl.johari` section, all four quadrant definitions

**Priority**: MUST

**Verification**: Full state verification test comparing all four quadrant->action mappings against Constitution

#### REQ-UTL-004: No Intentional Deviation Without Documentation

**Description**: If there is ever an intentional deviation from Constitution in the Johari action mapping, it MUST be documented with:
1. A code comment explaining the rationale
2. An entry in the decision log
3. A note in the relevant test explaining why the test expects different behavior

**Rationale**: Prevents future confusion about whether differences are bugs or intentional design decisions.

**Priority**: SHOULD

### 3.2 Non-Functional Requirements

#### NFR-UTL-001: Fix Must Not Break API Compatibility

**Category**: Compatibility

**Description**: The fix changes internal implementation only. The public API signatures (`get_suggested_action`, `SuggestedAction` enum) MUST remain unchanged.

**Metric**: Zero breaking changes to public function signatures

#### NFR-UTL-002: Fix Must Not Impact Performance

**Category**: Performance

**Description**: The fix is a simple match arm swap with no performance implications. The function must remain O(1) with no allocations.

**Metric**: Function remains <100ns per call

---

## 4. Edge Cases

### 4.1 Boundary Value Cases

| ID | Scenario | deltaS | deltaC | Expected Quadrant | Expected Action |
|----|----------|--------|--------|-------------------|-----------------|
| EC-01 | Exactly at threshold (Open) | 0.5 | 0.5 | Open (per < vs <=) | DirectRecall |
| EC-02 | Just above entropy threshold | 0.51 | 0.49 | Blind | TriggerDream |
| EC-03 | Just above both thresholds | 0.51 | 0.51 | Unknown | EpistemicAction |
| EC-04 | High entropy, very low coherence | 0.9 | 0.1 | Blind | TriggerDream |
| EC-05 | Maximum entropy and coherence | 1.0 | 1.0 | Unknown | EpistemicAction |
| EC-06 | Zero entropy, maximum coherence | 0.0 | 1.0 | Open | DirectRecall |
| EC-07 | Zero both values | 0.0 | 0.0 | Hidden | GetNeighborhood |

### 4.2 Quadrant Transition Cases

| ID | Scenario | From | To | Expected Old Action | Expected New Action |
|----|----------|------|-----|---------------------|---------------------|
| EC-08 | Memory becomes coherent | Blind | Open | TriggerDream | DirectRecall |
| EC-09 | Memory gains entropy | Open | Unknown | DirectRecall | EpistemicAction |
| EC-10 | Memory loses coherence | Unknown | Blind | EpistemicAction | TriggerDream |

### 4.3 Error Handling Cases

| ID | Scenario | Expected Behavior |
|----|----------|-------------------|
| EC-11 | Invalid quadrant value (if possible) | Rust's exhaustive match prevents this at compile time |
| EC-12 | get_suggested_action called with any valid variant | Returns appropriate action without panic |

---

## 5. Error States

| ID | Condition | HTTP Code (if MCP) | Message | Recovery |
|----|-----------|-------------------|---------|----------|
| ERR-UTL-01 | N/A - function is infallible | - | - | Function cannot fail; returns valid action for all inputs |

Note: The `get_suggested_action` function is infallible by design. It takes an enum (which Rust enforces at compile time) and returns an enum variant. There are no error states.

---

## 6. Implementation Details

### 6.1 Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-utl/src/johari/retrieval/functions.rs` | Swap Blind and Unknown match arms |

### 6.2 Current Implementation (WRONG)

```rust
// File: crates/context-graph-utl/src/johari/retrieval/functions.rs:34-40
#[inline]
pub fn get_suggested_action(quadrant: JohariQuadrant) -> SuggestedAction {
    match quadrant {
        JohariQuadrant::Open => SuggestedAction::DirectRecall,
        JohariQuadrant::Blind => SuggestedAction::EpistemicAction,  // WRONG
        JohariQuadrant::Hidden => SuggestedAction::GetNeighborhood,
        JohariQuadrant::Unknown => SuggestedAction::TriggerDream,   // WRONG
    }
}
```

### 6.3 Correct Implementation

```rust
// File: crates/context-graph-utl/src/johari/retrieval/functions.rs:34-40
#[inline]
pub fn get_suggested_action(quadrant: JohariQuadrant) -> SuggestedAction {
    match quadrant {
        JohariQuadrant::Open => SuggestedAction::DirectRecall,
        JohariQuadrant::Blind => SuggestedAction::TriggerDream,     // FIXED: per constitution
        JohariQuadrant::Hidden => SuggestedAction::GetNeighborhood,
        JohariQuadrant::Unknown => SuggestedAction::EpistemicAction, // FIXED: per constitution
    }
}
```

### 6.4 Docstring Updates

Update the docstring example in `functions.rs` to reflect correct behavior:

```rust
/// # Example
///
/// ```
/// use context_graph_utl::johari::{get_suggested_action, SuggestedAction, JohariQuadrant};
///
/// assert_eq!(get_suggested_action(JohariQuadrant::Open), SuggestedAction::DirectRecall);
/// assert_eq!(get_suggested_action(JohariQuadrant::Blind), SuggestedAction::TriggerDream);
/// assert_eq!(get_suggested_action(JohariQuadrant::Hidden), SuggestedAction::GetNeighborhood);
/// assert_eq!(get_suggested_action(JohariQuadrant::Unknown), SuggestedAction::EpistemicAction);
/// ```
```

### 6.5 Related Comments in action.rs

The comments in `action.rs` are partially confusing. Update to clarify:

```rust
/// Epistemic action or dream-based discovery - for Unknown quadrant.
///                                            ^^^^^^^^^^^^^^^^^^^^^
/// (Fix: Changed from "Blind quadrant" to "Unknown quadrant")
```

---

## 7. Test Plan

### 7.1 Unit Tests

#### TC-UTL-001: Verify Blind Quadrant Maps to TriggerDream

**Type**: unit
**Requirement Reference**: REQ-UTL-001

**Description**: Test that Blind quadrant returns TriggerDream action

**Inputs**: `JohariQuadrant::Blind`

**Expected Output**: `SuggestedAction::TriggerDream`

```rust
#[test]
fn test_blind_quadrant_triggers_dream() {
    let action = get_suggested_action(JohariQuadrant::Blind);
    assert_eq!(action, SuggestedAction::TriggerDream);
}
```

#### TC-UTL-002: Verify Unknown Quadrant Maps to EpistemicAction

**Type**: unit
**Requirement Reference**: REQ-UTL-002

**Description**: Test that Unknown quadrant returns EpistemicAction

**Inputs**: `JohariQuadrant::Unknown`

**Expected Output**: `SuggestedAction::EpistemicAction`

```rust
#[test]
fn test_unknown_quadrant_epistemic_action() {
    let action = get_suggested_action(JohariQuadrant::Unknown);
    assert_eq!(action, SuggestedAction::EpistemicAction);
}
```

#### TC-UTL-003: Verify Open Quadrant Unchanged (Regression)

**Type**: unit
**Requirement Reference**: REQ-UTL-003

**Description**: Ensure Open quadrant still returns DirectRecall

**Inputs**: `JohariQuadrant::Open`

**Expected Output**: `SuggestedAction::DirectRecall`

```rust
#[test]
fn test_open_quadrant_direct_recall() {
    let action = get_suggested_action(JohariQuadrant::Open);
    assert_eq!(action, SuggestedAction::DirectRecall);
}
```

#### TC-UTL-004: Verify Hidden Quadrant Unchanged (Regression)

**Type**: unit
**Requirement Reference**: REQ-UTL-003

**Description**: Ensure Hidden quadrant still returns GetNeighborhood

**Inputs**: `JohariQuadrant::Hidden`

**Expected Output**: `SuggestedAction::GetNeighborhood`

```rust
#[test]
fn test_hidden_quadrant_get_neighborhood() {
    let action = get_suggested_action(JohariQuadrant::Hidden);
    assert_eq!(action, SuggestedAction::GetNeighborhood);
}
```

### 7.2 Full State Verification Test

#### TC-UTL-005: Complete Quadrant-Action Mapping Verification

**Type**: integration
**Requirement Reference**: REQ-UTL-003

**Description**: Verify all four quadrant->action mappings match Constitution

```rust
#[test]
fn test_all_quadrant_actions_match_constitution() {
    // Constitution says:
    // Open: DirectRecall
    // Blind: TriggerDream
    // Hidden: GetNeighborhood
    // Unknown: EpistemicAction

    let expected: [(JohariQuadrant, SuggestedAction); 4] = [
        (JohariQuadrant::Open, SuggestedAction::DirectRecall),
        (JohariQuadrant::Blind, SuggestedAction::TriggerDream),
        (JohariQuadrant::Hidden, SuggestedAction::GetNeighborhood),
        (JohariQuadrant::Unknown, SuggestedAction::EpistemicAction),
    ];

    for (quadrant, expected_action) in expected {
        let actual_action = get_suggested_action(quadrant);
        assert_eq!(
            actual_action, expected_action,
            "Quadrant {:?} should map to {:?} per Constitution, but got {:?}",
            quadrant, expected_action, actual_action
        );
    }
}
```

### 7.3 Property-Based Test

#### TC-UTL-006: Exhaustive Quadrant Coverage

**Type**: property
**Requirement Reference**: REQ-UTL-003

**Description**: Verify every quadrant variant produces a valid action

```rust
#[test]
fn test_all_quadrants_produce_valid_actions() {
    for quadrant in JohariQuadrant::all() {
        let action = get_suggested_action(quadrant);
        // Action should be one of the four valid variants
        assert!(
            matches!(
                action,
                SuggestedAction::DirectRecall
                    | SuggestedAction::TriggerDream
                    | SuggestedAction::GetNeighborhood
                    | SuggestedAction::EpistemicAction
            ),
            "Quadrant {:?} produced invalid action",
            quadrant
        );
    }
}
```

### 7.4 QuadrantRetrieval Integration Test

#### TC-UTL-007: QuadrantRetrieval Uses Correct Mapping

**Type**: integration
**Requirement Reference**: REQ-UTL-001, REQ-UTL-002

**Description**: Verify QuadrantRetrieval struct delegates correctly

```rust
#[test]
fn test_quadrant_retrieval_correct_actions() {
    let retrieval = QuadrantRetrieval::with_default_weights();

    assert_eq!(retrieval.get_action(JohariQuadrant::Blind), SuggestedAction::TriggerDream);
    assert_eq!(retrieval.get_action(JohariQuadrant::Unknown), SuggestedAction::EpistemicAction);
}
```

---

## 8. Verification Commands

After implementation, run:

```bash
# Run UTL crate tests
cargo test -p context-graph-utl johari

# Run specific test for Johari actions
cargo test -p context-graph-utl test_all_quadrant_actions_match_constitution

# Run doc tests to verify examples
cargo test -p context-graph-utl --doc

# Full test suite
cargo test --all

# Verify no regressions in dependent crates
cargo test -p context-graph-mcp johari
cargo test -p context-graph-core johari
```

---

## 9. Definition of Done

- [ ] `get_suggested_action` returns correct action for all four quadrants
- [ ] Blind quadrant returns TriggerDream
- [ ] Unknown quadrant returns EpistemicAction
- [ ] Open and Hidden quadrants unchanged (regression check)
- [ ] Docstring examples updated and pass
- [ ] All existing Johari tests pass
- [ ] No new compiler warnings
- [ ] Code matches Constitution v5.0.0 exactly

---

## 10. Appendix: Semantic Rationale

### Why Blind -> TriggerDream?

**Blind Quadrant** (High Entropy, Low Coherence):
- The system has encountered something novel (high entropy)
- But it doesn't fit well with existing knowledge (low coherence)
- This is a "discovery opportunity" - the system sees patterns it hasn't integrated
- **Dream consolidation** (NREM Hebbian replay + REM hyperbolic walk) is the mechanism to integrate novel discoveries into coherent knowledge

### Why Unknown -> EpistemicAction?

**Unknown Quadrant** (High Entropy, High Coherence):
- The system has encountered something novel (high entropy)
- AND it fits well with existing knowledge (high coherence)
- This is "frontier knowledge" - new but connected
- **Epistemic action** (asking clarifying questions) helps the system understand this frontier better before full integration
- The coherence suggests the system has enough context to formulate good questions

---

## 11. References

- Constitution v5.0.0: `docs2/constitution.yaml` (Section `utl.johari`)
- PRD v4.0.0: `docs2/contextprd.md` (Section 2.2 Johari Quadrants)
- Master Issues Plan: `docs/MASTER-ISSUES-REMEDIATION-PLAN.md` (ISS-011)
- PRD Analysis: `docs/specs/PRD-ANALYSIS-REMEDIATION.md` (REQ-UTL-001 through REQ-UTL-003)
