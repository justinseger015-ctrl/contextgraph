# SPEC-NEURO-001: Direct Dopamine Feedback Loop

## Metadata

| Field | Value |
|-------|-------|
| **Spec ID** | SPEC-NEURO-001 |
| **Title** | Direct Dopamine Feedback Loop from Steering System |
| **Status** | Approved |
| **Priority** | P2 (Minor Refinement) |
| **Owner** | ContextGraph Core Team |
| **Created** | 2026-01-11 |
| **Last Updated** | 2026-01-11 |
| **Related Specs** | SPEC-STEERING-001, SPEC-GWT-001 |
| **Constitution Ref** | neuromod.Dopamine (lines 162-170), steering.components |

---

## 1. Overview

### 1.1 Purpose

This specification defines the implementation of a **Direct Dopamine Feedback Loop** that connects the Steering Subsystem directly to the Neuromodulation Manager. Currently, steering feedback updates edge weights which indirectly affect dopamine through workspace events. This creates a delayed, indirect modulation path.

The enhancement establishes a direct path:

```
Steering → Direct DA Modulation → Cascade Effects (Hopfield beta, retrieval sharpness)
```

### 1.2 Problem Statement

**Current Flow (Indirect)**:
```
SteeringFeedback.reward → Edge Weight Updates → Memory Workspace Entry → on_workspace_entry() → DA++
```

This indirect path has several limitations:
1. **Latency**: Dopamine changes are delayed by edge weight propagation and workspace entry events
2. **Weak Signal**: The steering signal is diluted through multiple transformation layers
3. **Missing Negative Feedback**: Negative steering rewards have no direct path to decrease DA
4. **No Goal Progress Tracking**: Goal achievement/regression has no direct neurochemical impact

**Proposed Flow (Direct)**:
```
SteeringFeedback.reward → on_goal_progress(delta) → DA += delta * SENSITIVITY → Immediate cascade
```

### 1.3 Success Criteria

| Criterion | Metric | Threshold |
|-----------|--------|-----------|
| DA responds to positive steering | DA increases when reward > 0 | Delta > 0 within 1 tick |
| DA responds to negative steering | DA decreases when reward < 0 | Delta < 0 within 1 tick |
| Cascade propagates to Hopfield | hopfield.beta reflects DA change | Within 10ms |
| Homeostatic regulation preserved | DA returns to baseline over time | < 60s to baseline |
| No disruption to existing triggers | workspace_entry still functions | 100% compatibility |

---

## 2. Requirements

### 2.1 Functional Requirements

#### FR-NEURO-001-01: Goal Progress Event Handler

**Description**: The `NeuromodulationManager` SHALL provide an `on_goal_progress(delta: f32)` method that directly modulates dopamine based on goal achievement progress.

**Acceptance Criteria**:
- Given a positive delta (goal progress), dopamine increases proportionally
- Given a negative delta (goal regression), dopamine decreases proportionally
- Given a zero delta, dopamine remains unchanged
- Modulation respects DA_MIN (1.0) and DA_MAX (5.0) bounds
- Method logs the adjustment for observability

**Rationale**: Enables direct neurochemical response to goal-related feedback without intermediate edge weight transformations.

---

#### FR-NEURO-001-02: Dopamine Modulator Goal Progress Method

**Description**: The `DopamineModulator` SHALL provide an `on_goal_progress(delta: f32)` method that adjusts dopamine using a configurable sensitivity factor.

**Acceptance Criteria**:
- Delta is scaled by `DA_GOAL_SENSITIVITY` (default: 0.1)
- Adjustment formula: `DA_new = DA_old + delta * DA_GOAL_SENSITIVITY`
- Result is clamped to [DA_MIN, DA_MAX]
- Method updates `last_trigger` timestamp when adjustment is non-zero
- Debug logging captures adjustment magnitude and new value

**Rationale**: Encapsulates the dopamine adjustment logic with proper sensitivity tuning and bounds checking.

---

#### FR-NEURO-001-03: Steering-to-Neuromodulation Integration

**Description**: The Steering MCP handler SHALL invoke `on_goal_progress()` after computing steering feedback, using the reward value as the delta.

**Acceptance Criteria**:
- After `compute_feedback()` returns, `on_goal_progress(reward.value)` is called
- Integration point is in `call_get_steering_feedback()` handler
- Neuromodulation manager is accessed via shared state
- Errors in neuromodulation do not fail the steering response
- Adjustment is logged for correlation with steering feedback

**Rationale**: Closes the loop between steering assessment and neurochemical modulation.

---

#### FR-NEURO-001-04: Sensitivity Configuration

**Description**: The dopamine goal sensitivity factor SHALL be configurable through the neuromodulation configuration.

**Acceptance Criteria**:
- Default `DA_GOAL_SENSITIVITY = 0.1` defined as constant
- Configuration can override the default value
- Sensitivity clamped to reasonable range [0.01, 0.5]
- Configuration changes apply without restart

**Rationale**: Allows tuning the strength of goal-related dopamine modulation without code changes.

---

### 2.2 Non-Functional Requirements

#### NFR-NEURO-001-01: Performance

| Metric | Target |
|--------|--------|
| `on_goal_progress()` latency | < 1ms |
| Memory overhead | < 100 bytes per call |
| No allocation in hot path | Zero heap allocations |

#### NFR-NEURO-001-02: Compatibility

- MUST NOT break existing `on_workspace_entry()` behavior
- MUST NOT break existing `on_negative_event()` behavior
- MUST maintain homeostatic decay behavior
- MUST preserve all existing test cases

#### NFR-NEURO-001-03: Observability

- All adjustments logged at DEBUG level
- Adjustment events include: delta, sensitivity, old_value, new_value
- Traceable correlation with steering feedback events

---

## 3. Technical Design

### 3.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MCP Layer                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              call_get_steering_feedback()                     │   │
│  │                           │                                   │   │
│  │                           ▼                                   │   │
│  │              steering.compute_feedback()                      │   │
│  │                           │                                   │   │
│  │                           ▼                                   │   │
│  │              SteeringFeedback { reward: f32 }                 │   │
│  │                           │                                   │   │
│  │              ┌────────────┴────────────┐                     │   │
│  │              │                         │                      │   │
│  │              ▼                         ▼                      │   │
│  │      Return JSON Response    neuromod_manager                 │   │
│  │                              .on_goal_progress(reward.value)  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Neuromodulation Layer                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              NeuromodulationManager                           │   │
│  │                           │                                   │   │
│  │              on_goal_progress(delta: f32)                     │   │
│  │                           │                                   │   │
│  │                           ▼                                   │   │
│  │              self.dopamine.on_goal_progress(delta)            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                    │                                 │
│                                    ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              DopamineModulator                                │   │
│  │                           │                                   │   │
│  │  on_goal_progress(delta):                                     │   │
│  │    adjustment = delta * DA_GOAL_SENSITIVITY (0.1)             │   │
│  │    self.value = clamp(self.value + adjustment, 1.0, 5.0)      │   │
│  │    if adjustment != 0: self.last_trigger = now()              │   │
│  │                           │                                   │   │
│  │                           ▼                                   │   │
│  │  Effects:                                                     │   │
│  │    - hopfield.beta = self.value                               │   │
│  │    - Retrieval sharpness adjusted                             │   │
│  │    - High DA: Sharp, focused retrieval                        │   │
│  │    - Low DA: Diffuse, exploratory retrieval                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

```
Input: SteeringReward.value ∈ [-1.0, 1.0]
       │
       ▼
DA_GOAL_SENSITIVITY = 0.1
       │
       ▼
adjustment = value * 0.1 ∈ [-0.1, 0.1]
       │
       ▼
DA_new = clamp(DA_old + adjustment, 1.0, 5.0)
       │
       ▼
Output: hopfield.beta = DA_new
```

### 3.3 Component Contracts

#### DopamineModulator (Updated)

```rust
/// Dopamine increase per goal progress unit (steering reward in [-1, 1])
pub const DA_GOAL_SENSITIVITY: f32 = 0.1;

impl DopamineModulator {
    /// Handle goal progress event from steering subsystem.
    ///
    /// Adjusts dopamine based on goal achievement delta:
    /// - Positive delta (goal progress): DA increases
    /// - Negative delta (goal regression): DA decreases
    ///
    /// # Arguments
    /// * `delta` - Goal progress delta, typically from SteeringReward.value [-1, 1]
    ///
    /// # Effects
    /// * DA adjusted by delta * DA_GOAL_SENSITIVITY
    /// * Clamped to [DA_MIN, DA_MAX]
    /// * Updates last_trigger if adjustment is non-zero
    pub fn on_goal_progress(&mut self, delta: f32) {
        let adjustment = delta * DA_GOAL_SENSITIVITY;
        if adjustment.abs() > f32::EPSILON {
            self.level.value = (self.level.value + adjustment).clamp(DA_MIN, DA_MAX);
            self.level.last_trigger = Some(Utc::now());
            tracing::debug!(
                delta = delta,
                adjustment = adjustment,
                new_value = self.level.value,
                "Dopamine adjusted on goal progress"
            );
        }
    }
}
```

#### NeuromodulationManager (Updated)

```rust
impl NeuromodulationManager {
    /// Handle goal progress from steering subsystem.
    ///
    /// Propagates goal achievement/regression to dopamine modulator.
    /// This provides direct neurochemical response to steering feedback.
    ///
    /// # Arguments
    /// * `delta` - Goal progress delta, typically SteeringReward.value [-1, 1]
    pub fn on_goal_progress(&mut self, delta: f32) {
        self.dopamine.on_goal_progress(delta);
    }
}
```

---

## 4. Edge Cases

### 4.1 Edge Case Matrix

| ID | Scenario | Input | Expected Behavior |
|----|----------|-------|-------------------|
| EC-01 | Zero delta | delta = 0.0 | No adjustment, no logging |
| EC-02 | Maximum positive | delta = 1.0 | DA += 0.1, clamped to 5.0 |
| EC-03 | Maximum negative | delta = -1.0 | DA -= 0.1, clamped to 1.0 |
| EC-04 | DA at ceiling | DA = 5.0, delta = 1.0 | DA remains 5.0 (no overflow) |
| EC-05 | DA at floor | DA = 1.0, delta = -1.0 | DA remains 1.0 (no underflow) |
| EC-06 | Very small delta | delta = 0.001 | Adjustment = 0.0001 (applied) |
| EC-07 | Epsilon delta | delta = f32::EPSILON | No adjustment (below threshold) |
| EC-08 | Concurrent calls | Rapid successive calls | Each applies independently |
| EC-09 | NaN delta | delta = f32::NAN | No change (NaN check) |
| EC-10 | Infinity delta | delta = f32::INFINITY | Clamped to max adjustment |

### 4.2 Error States

| ID | Error Condition | Detection | Recovery |
|----|-----------------|-----------|----------|
| ERR-01 | Invalid delta (NaN) | `delta.is_nan()` | Log warning, skip adjustment |
| ERR-02 | Neuromod manager not initialized | Null check in handler | Return response without DA update |

---

## 5. Test Plan

### 5.1 Unit Tests

#### TC-NEURO-001-01: Basic Goal Progress Positive

```rust
#[test]
fn test_dopamine_on_goal_progress_positive() {
    let mut modulator = DopamineModulator::new();
    let initial = modulator.value();

    modulator.on_goal_progress(0.5);

    let expected = initial + 0.5 * DA_GOAL_SENSITIVITY;
    assert!((modulator.value() - expected).abs() < f32::EPSILON);
}
```

#### TC-NEURO-001-02: Basic Goal Progress Negative

```rust
#[test]
fn test_dopamine_on_goal_progress_negative() {
    let mut modulator = DopamineModulator::new();
    let initial = modulator.value();

    modulator.on_goal_progress(-0.5);

    let expected = initial - 0.5 * DA_GOAL_SENSITIVITY;
    assert!((modulator.value() - expected).abs() < f32::EPSILON);
}
```

#### TC-NEURO-001-03: Goal Progress Ceiling Clamp

```rust
#[test]
fn test_dopamine_on_goal_progress_ceiling_clamp() {
    let mut modulator = DopamineModulator::new();
    modulator.set_value(DA_MAX);

    modulator.on_goal_progress(1.0);

    assert!((modulator.value() - DA_MAX).abs() < f32::EPSILON);
}
```

#### TC-NEURO-001-04: Goal Progress Floor Clamp

```rust
#[test]
fn test_dopamine_on_goal_progress_floor_clamp() {
    let mut modulator = DopamineModulator::new();
    modulator.set_value(DA_MIN);

    modulator.on_goal_progress(-1.0);

    assert!((modulator.value() - DA_MIN).abs() < f32::EPSILON);
}
```

#### TC-NEURO-001-05: Manager Propagates Goal Progress

```rust
#[test]
fn test_manager_on_goal_progress() {
    let mut manager = NeuromodulationManager::new();
    let initial = manager.get_hopfield_beta();

    manager.on_goal_progress(0.8);

    assert!(manager.get_hopfield_beta() > initial);
}
```

### 5.2 Integration Tests

#### TC-NEURO-001-06: Steering to DA Integration

```rust
#[tokio::test]
async fn test_steering_feedback_modulates_dopamine() {
    // Setup: Create handlers with neuromod manager
    let handlers = create_test_handlers().await;

    // Get initial DA
    let initial_da = handlers.neuromod_manager.lock().await.get_hopfield_beta();

    // Trigger steering feedback (positive scenario)
    let response = handlers.call_get_steering_feedback(None).await;

    // Verify DA changed based on steering reward
    let final_da = handlers.neuromod_manager.lock().await.get_hopfield_beta();

    // DA should have changed (direction depends on reward)
    assert!((final_da - initial_da).abs() > f32::EPSILON);
}
```

---

## 6. Implementation Notes

### 6.1 Key Considerations

1. **Thread Safety**: `NeuromodulationManager` is not `Send` due to `Instant`. If accessed from async context, wrap in `Arc<Mutex<>>` or use thread-local state.

2. **Decay Interaction**: Goal progress adjustments are subject to the same homeostatic decay as other dopamine triggers. This is intentional - the system naturally returns to baseline.

3. **Additive vs Multiplicative**: The adjustment is additive (`DA += delta * sensitivity`) rather than multiplicative. This provides linear, predictable behavior.

4. **Sensitivity Tuning**: The default sensitivity of 0.1 means:
   - Maximum reward (+1.0) increases DA by 0.1
   - Maximum penalty (-1.0) decreases DA by 0.1
   - From baseline (3.0), this allows ~20 positive rewards to hit ceiling

### 6.2 Dependencies

| Component | Dependency Type | Notes |
|-----------|-----------------|-------|
| `DopamineModulator` | Direct modification | Add `on_goal_progress()` |
| `NeuromodulationManager` | Direct modification | Add forwarding method |
| `steering.rs` (MCP) | Integration point | Call after compute_feedback |
| `Handlers` struct | Access pattern | Need neuromod_manager ref |

### 6.3 Migration Path

1. Add `DA_GOAL_SENSITIVITY` constant to `dopamine.rs`
2. Add `on_goal_progress()` to `DopamineModulator`
3. Add `on_goal_progress()` to `NeuromodulationManager`
4. Add `NeuromodulationManager` to `Handlers` struct (if not present)
5. Call `on_goal_progress()` in `call_get_steering_feedback()`
6. Add unit tests
7. Add integration tests

---

## 7. Traceability

### 7.1 Requirement to Task Mapping

| Requirement | Task ID | Status | Layer |
|-------------|---------|--------|-------|
| FR-NEURO-001-01 | TASK-NEURO-P2-001 | Ready | logic |
| FR-NEURO-001-02 | TASK-NEURO-P2-001 | Ready | logic |
| FR-NEURO-001-03 | TASK-NEURO-P2-002 | Ready | surface |
| FR-NEURO-001-04 | TASK-NEURO-P2-001 | Ready | logic |
| NFR-NEURO-001-01 | TASK-NEURO-P2-001 | Ready | logic |
| NFR-NEURO-001-02 | TASK-NEURO-P2-001 | Ready | logic |
| NFR-NEURO-001-03 | TASK-NEURO-P2-001 | Ready | logic |
| Cascade Effects | TASK-NEURO-P2-003 | Ready | logic |
| MCP Integration | TASK-NEURO-P2-002 | Ready | surface |

### 7.2 Gap Analysis Reference

This specification addresses **REFINEMENT 3** from the Master Consciousness Gap Analysis:

> **Location**: `src/neuromodulation/steering.ts` [Note: Rust equivalent]
> **Impact**: Steering feedback updates edge weights, not DA directly
> **Current**: `Steering -> Edge Weights -> Indirect DA effect`
> **Preferred**: `Steering -> Direct DA modulation -> Cascade effects`

---

## 8. Cascade Effects

### 8.1 Dopamine Cascade to Subsystems

When dopamine changes via `on_goal_progress()`, the following cascade effects occur:

```
on_goal_progress(delta)
    |
    v
DA_new = clamp(DA + delta * 0.1, 1.0, 5.0)
    |
    +---> hopfield.beta = DA_new
    |         |
    |         +---> Retrieval sharpness changes
    |         |         High DA (5): Sharp, focused retrieval
    |         |         Low DA (1): Diffuse, exploratory retrieval
    |         |
    |         +---> Memory workspace entry probability
    |                   High DA: More selective, high-phi only
    |                   Low DA: Broader inclusion
    |
    +---> Steering edge weight modulation
              |
              +---> get_modulated_weight() uses DA
                        w_eff = base * (1 + steering_reward * 0.2)
                        (DA indirectly affects edge selection)
```

### 8.2 Cross-Neuromodulator Interactions

Dopamine changes can trigger secondary effects on other neuromodulators:

| Trigger Condition | Effect | Rationale |
|-------------------|--------|-----------|
| DA > 4.0 (high reward) | 5HT += 0.05 | Success breeds positive mood |
| DA < 2.0 (low reward) | 5HT -= 0.05 | Failure lowers mood |
| DA change > 0.3 | NE += 0.1 | Significant change increases alertness |
| Sustained DA > 4.5 for 30s | ACh triggers (via GWT) | High performance triggers learning boost |

**Note**: Cross-neuromodulator effects are OPTIONAL refinements for future implementation. The core DA feedback loop is the primary deliverable.

### 8.3 Integration with UTL (Unified Theory of Learning)

The dopamine level affects UTL operations:

```rust
/// UTL learning rate modifier based on dopamine
fn get_utl_modifier(da: f32) -> f32 {
    // High DA = focused learning on high-reward patterns
    // Low DA = exploratory learning across patterns
    match da {
        d if d >= 4.0 => 1.2,  // 20% boost for high-value patterns
        d if d >= 3.0 => 1.0,  // Normal learning rate
        d if d >= 2.0 => 0.9,  // Slightly reduced for low-confidence
        _ => 0.8,              // Exploratory mode
    }
}
```

### 8.4 Integration with GWT (Global Workspace Theory)

Dopamine affects GWT workspace dynamics:

| DA Level | Workspace Behavior | Competition Threshold |
|----------|-------------------|----------------------|
| 5.0 | Sharp focus | phi > 0.8 required |
| 4.0 | Focused | phi > 0.7 required |
| 3.0 (baseline) | Balanced | phi > 0.5 required |
| 2.0 | Exploratory | phi > 0.3 required |
| 1.0 | Broad scan | phi > 0.2 required |

The `hopfield.beta` parameter directly controls the softmax temperature in workspace competition, affecting which memories reach consciousness.

---

## 9. MCP Integration Specification

### 9.1 Steering Handler Integration Point

The MCP steering handler SHALL invoke `on_goal_progress()` after computing steering feedback.

**File**: `crates/context-graph-mcp/src/handlers/steering.rs`

**Integration Pattern**:

```rust
impl Handlers {
    pub(super) async fn call_get_steering_feedback(
        &self,
        id: Option<JsonRpcId>,
    ) -> JsonRpcResponse {
        // ... existing metric collection ...

        // Compute steering feedback
        let feedback = steering.compute_feedback(/* ... */);

        // NEW: Invoke direct DA modulation
        // Access neuromod manager from shared state
        if let Ok(mut neuromod) = self.neuromod_manager.try_write() {
            neuromod.on_goal_progress(feedback.reward.value);
            debug!(
                delta = feedback.reward.value,
                "Steering feedback -> DA modulation"
            );
        } else {
            warn!("Could not acquire neuromod lock for goal progress");
            // Non-fatal: continue with response
        }

        self.tool_result_with_pulse(id, json!({
            // ... existing response ...
            "neuromod_updated": true,
            "da_delta": feedback.reward.value * dopamine::DA_GOAL_SENSITIVITY
        }))
    }
}
```

### 9.2 Handlers Struct Requirements

The `Handlers` struct must include a reference to `NeuromodulationManager`:

```rust
pub struct Handlers {
    // ... existing fields ...

    /// Neuromodulation manager for DA/5HT/NE control
    /// Wrapped in Arc<RwLock> for thread-safe access
    pub neuromod_manager: Arc<RwLock<NeuromodulationManager>>,
}
```

### 9.3 Error Handling

| Error Condition | Behavior | Logging |
|----------------|----------|---------|
| Neuromod lock unavailable | Continue without DA update | WARN level |
| DA update fails | Continue with response | ERROR level |
| NaN in reward value | Skip DA update | WARN level |

The MCP response should NEVER fail due to neuromodulation errors. The steering feedback is the primary response; DA modulation is a side effect.

---

## 10. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | Specification Agent | 2026-01-11 | Auto-generated |
| Reviewer | - | - | Pending |
| Approver | - | - | Pending |

---

## Appendix A: Constitution Reference

```yaml
neuromod:
  Dopamine:
    range: "[1, 5]"
    parameter: hopfield.beta
    trigger: memory_enters_workspace  # Existing trigger
    # NEW: goal_progress trigger via steering feedback

steering:
  components:
    - Gardener: Graph maintenance and pruning
    - Curator: Quality assessment and curation
    - Assessor: Performance evaluation
  reward:
    range: "[-1, 1]"
    # NEW: feeds into neuromod.Dopamine.on_goal_progress()
```
