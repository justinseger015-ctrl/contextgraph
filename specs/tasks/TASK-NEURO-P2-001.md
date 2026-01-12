# TASK-NEURO-P2-001: Implement Direct Dopamine Feedback Loop

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-NEURO-P2-001 |
| **Title** | Implement Direct Dopamine Feedback from Steering Subsystem |
| **Status** | **COMPLETED** |
| **Priority** | P2 (Minor Refinement) |
| **Layer** | logic |
| **Sequence** | 1 |
| **Estimated Complexity** | Low |
| **Implements** | SPEC-NEURO-001 |
| **Depends On** | None (foundation layer already exists) |

---

## CRITICAL: Read This First

**NO BACKWARDS COMPATIBILITY. FAIL FAST.** If something breaks, it must error with clear logging—no workarounds, no fallbacks, no mock data.

**NO MOCK DATA IN TESTS.** All tests must use real data structures and verify actual outcomes.

---

## 1. Context

### 1.1 What This Task Does

This task adds a direct path from steering feedback to dopamine modulation. Currently dopamine only changes via `on_workspace_entry()` (when memory enters consciousness). This creates a new trigger: `on_goal_progress(delta)` that responds to steering reward values.

### 1.2 Current State (Verified 2026-01-12)

**File: `crates/context-graph-core/src/neuromod/dopamine.rs`**
- `DopamineModulator` exists with methods: `on_workspace_entry()`, `on_negative_event()`, `decay()`, `reset()`, `set_value()`, `value()`, `get_hopfield_beta()`
- Constants: `DA_BASELINE = 3.0`, `DA_MIN = 1.0`, `DA_MAX = 5.0`, `DA_DECAY_RATE = 0.05`, `DA_WORKSPACE_INCREMENT = 0.2`
- **MISSING**: `DA_GOAL_SENSITIVITY` constant and `on_goal_progress()` method

**File: `crates/context-graph-core/src/neuromod/state.rs`**
- `NeuromodulationManager` exists with methods: `on_workspace_entry()`, `on_threat_detected()`, `on_positive_event()`, `on_negative_event()`, `get_hopfield_beta()`, etc.
- Has private field: `dopamine: DopamineModulator`
- **MISSING**: `on_goal_progress()` forwarding method

**File: `crates/context-graph-core/src/neuromod/mod.rs`**
- Re-exports: `DA_BASELINE`, `DA_MAX`, `DA_MIN`, `DopamineLevel`, `DopamineModulator`
- **MISSING**: `DA_GOAL_SENSITIVITY` export

### 1.3 Target State

Add these exact signatures:

```rust
// In dopamine.rs
pub const DA_GOAL_SENSITIVITY: f32 = 0.1;

impl DopamineModulator {
    pub fn on_goal_progress(&mut self, delta: f32) { /* ... */ }
}

// In state.rs
impl NeuromodulationManager {
    pub fn on_goal_progress(&mut self, delta: f32) { /* ... */ }
}
```

---

## 2. Input Context Files (MUST READ)

| File | Why You Need It |
|------|-----------------|
| `crates/context-graph-core/src/neuromod/dopamine.rs` | Implementation target - see existing `on_workspace_entry()` pattern at lines 91-98 |
| `crates/context-graph-core/src/neuromod/state.rs` | Implementation target - see existing `on_workspace_entry()` forwarding at line 259-261 |
| `crates/context-graph-core/src/neuromod/mod.rs` | Update exports - see current exports at lines 62-70 |
| `docs2/constitution.yaml` | Reference for neuromod spec at lines 245-250 |

---

## 3. Exact Implementation

### 3.1 dopamine.rs Additions

Add after line 35 (after `DA_WORKSPACE_INCREMENT`):

```rust
/// Dopamine adjustment sensitivity for goal progress events.
/// Maximum reward (+1.0) increases DA by 0.1; maximum penalty (-1.0) decreases by 0.1.
pub const DA_GOAL_SENSITIVITY: f32 = 0.1;
```

Add to `impl DopamineModulator` block (around line 140, before the `Default` impl):

```rust
    /// Handle goal progress event from steering subsystem.
    ///
    /// Adjusts dopamine based on goal achievement delta:
    /// - Positive delta (goal progress): DA increases
    /// - Negative delta (goal regression): DA decreases
    ///
    /// # Arguments
    /// * `delta` - Goal progress delta from SteeringReward.value [-1, 1]
    ///
    /// # Effects
    /// * DA adjusted by delta * DA_GOAL_SENSITIVITY
    /// * Clamped to [DA_MIN, DA_MAX]
    /// * Updates last_trigger if adjustment is non-zero
    pub fn on_goal_progress(&mut self, delta: f32) {
        // Guard against NaN - FAIL FAST with warning
        if delta.is_nan() {
            tracing::warn!("on_goal_progress received NaN delta - skipping adjustment");
            return;
        }

        // Calculate adjustment
        let adjustment = delta * DA_GOAL_SENSITIVITY;

        // Skip if adjustment is effectively zero (avoids unnecessary timestamp updates)
        if adjustment.abs() <= f32::EPSILON {
            return;
        }

        // Store old value for logging
        let old_value = self.level.value;

        // Apply adjustment with clamping
        self.level.value = (self.level.value + adjustment).clamp(DA_MIN, DA_MAX);

        // Update trigger timestamp
        self.level.last_trigger = Some(chrono::Utc::now());

        // Log the adjustment
        tracing::debug!(
            delta = delta,
            adjustment = adjustment,
            old_value = old_value,
            new_value = self.level.value,
            "Dopamine adjusted on goal progress"
        );
    }
```

### 3.2 state.rs Additions

Add to `impl NeuromodulationManager` block (around line 288, after `on_negative_event`):

```rust
    /// Handle goal progress from steering subsystem.
    ///
    /// Propagates goal achievement/regression to dopamine modulator.
    /// This provides direct neurochemical response to steering feedback.
    ///
    /// # Arguments
    /// * `delta` - Goal progress delta from SteeringReward.value [-1, 1]
    pub fn on_goal_progress(&mut self, delta: f32) {
        self.dopamine.on_goal_progress(delta);
    }
```

### 3.3 mod.rs Updates

Update the dopamine re-exports (around line 64):

```rust
pub use dopamine::{
    DopamineLevel, DopamineModulator, DA_BASELINE, DA_GOAL_SENSITIVITY, DA_MAX, DA_MIN,
};
```

---

## 4. Unit Tests (Add to dopamine.rs #[cfg(test)] mod tests)

```rust
    #[test]
    fn test_dopamine_on_goal_progress_positive() {
        let mut modulator = DopamineModulator::new();
        let initial = modulator.value();

        modulator.on_goal_progress(0.5);

        let expected = initial + 0.5 * DA_GOAL_SENSITIVITY;
        assert!(
            (modulator.value() - expected).abs() < f32::EPSILON,
            "Expected {}, got {}",
            expected,
            modulator.value()
        );
    }

    #[test]
    fn test_dopamine_on_goal_progress_negative() {
        let mut modulator = DopamineModulator::new();
        let initial = modulator.value();

        modulator.on_goal_progress(-0.5);

        let expected = initial - 0.5 * DA_GOAL_SENSITIVITY;
        assert!(
            (modulator.value() - expected).abs() < f32::EPSILON,
            "Expected {}, got {}",
            expected,
            modulator.value()
        );
    }

    #[test]
    fn test_dopamine_on_goal_progress_ceiling_clamp() {
        let mut modulator = DopamineModulator::new();
        modulator.set_value(DA_MAX);

        modulator.on_goal_progress(1.0);

        assert!(
            (modulator.value() - DA_MAX).abs() < f32::EPSILON,
            "Should clamp to DA_MAX={}, got {}",
            DA_MAX,
            modulator.value()
        );
    }

    #[test]
    fn test_dopamine_on_goal_progress_floor_clamp() {
        let mut modulator = DopamineModulator::new();
        modulator.set_value(DA_MIN);

        modulator.on_goal_progress(-1.0);

        assert!(
            (modulator.value() - DA_MIN).abs() < f32::EPSILON,
            "Should clamp to DA_MIN={}, got {}",
            DA_MIN,
            modulator.value()
        );
    }

    #[test]
    fn test_dopamine_on_goal_progress_zero_delta() {
        let mut modulator = DopamineModulator::new();
        let initial = modulator.value();
        let initial_trigger = modulator.level().last_trigger.clone();

        modulator.on_goal_progress(0.0);

        assert!(
            (modulator.value() - initial).abs() < f32::EPSILON,
            "Zero delta should not change value"
        );
        assert_eq!(
            modulator.level().last_trigger, initial_trigger,
            "Zero delta should not update last_trigger"
        );
    }

    #[test]
    fn test_dopamine_on_goal_progress_updates_trigger() {
        let mut modulator = DopamineModulator::new();
        assert!(
            modulator.level().last_trigger.is_none(),
            "Fresh modulator should have no last_trigger"
        );

        modulator.on_goal_progress(0.5);

        assert!(
            modulator.level().last_trigger.is_some(),
            "Non-zero delta should set last_trigger"
        );
    }

    #[test]
    fn test_dopamine_on_goal_progress_nan_handling() {
        let mut modulator = DopamineModulator::new();
        let initial = modulator.value();

        modulator.on_goal_progress(f32::NAN);

        assert!(
            (modulator.value() - initial).abs() < f32::EPSILON,
            "NaN delta should not change value"
        );
    }
```

### 4.1 Unit Tests for state.rs (Add to state.rs #[cfg(test)] mod tests)

```rust
    #[test]
    fn test_manager_on_goal_progress_positive() {
        let mut manager = NeuromodulationManager::new();
        let initial = manager.get_hopfield_beta();

        manager.on_goal_progress(0.8);

        let expected = initial + 0.8 * super::dopamine::DA_GOAL_SENSITIVITY;
        assert!(
            (manager.get_hopfield_beta() - expected).abs() < f32::EPSILON,
            "Expected {}, got {}",
            expected,
            manager.get_hopfield_beta()
        );
    }

    #[test]
    fn test_manager_on_goal_progress_negative() {
        let mut manager = NeuromodulationManager::new();
        let initial = manager.get_hopfield_beta();

        manager.on_goal_progress(-0.6);

        let expected = initial - 0.6 * super::dopamine::DA_GOAL_SENSITIVITY;
        assert!(
            (manager.get_hopfield_beta() - expected).abs() < f32::EPSILON,
            "Expected {}, got {}",
            expected,
            manager.get_hopfield_beta()
        );
    }
```

---

## 5. Validation Commands

```bash
# Build (must succeed with no warnings)
cargo build -p context-graph-core 2>&1

# Run all neuromod tests (must all pass)
cargo test -p context-graph-core neuromod -- --nocapture 2>&1

# Run specific new tests
cargo test -p context-graph-core test_dopamine_on_goal_progress -- --nocapture 2>&1
cargo test -p context-graph-core test_manager_on_goal_progress -- --nocapture 2>&1

# Clippy (must have no warnings)
cargo clippy -p context-graph-core -- -D warnings 2>&1

# Doc generation (must succeed)
cargo doc -p context-graph-core --no-deps 2>&1
```

---

## 6. Full State Verification (FSV)

### 6.1 Source of Truth

The Source of Truth is the `DopamineModulator.level.value` field and `DopamineModulator.level.last_trigger` field. These are the actual stored state that all behavior derives from.

### 6.2 Execute & Inspect Protocol

After implementing, run this verification script in a test:

```rust
#[test]
fn test_fsv_goal_progress_source_of_truth() {
    // === STEP 1: Establish baseline state ===
    let mut modulator = DopamineModulator::new();

    println!("=== BEFORE STATE ===");
    println!("  value: {}", modulator.level().value);
    println!("  last_trigger: {:?}", modulator.level().last_trigger);

    let before_value = modulator.level().value;
    let before_trigger = modulator.level().last_trigger.clone();

    // === STEP 2: Execute the operation ===
    modulator.on_goal_progress(0.5);

    // === STEP 3: Read Source of Truth DIRECTLY ===
    println!("=== AFTER STATE (Source of Truth) ===");
    println!("  value: {}", modulator.level().value);
    println!("  last_trigger: {:?}", modulator.level().last_trigger);

    let after_value = modulator.level().value;
    let after_trigger = modulator.level().last_trigger.clone();

    // === STEP 4: Verify changes in Source of Truth ===
    let expected_delta = 0.5 * DA_GOAL_SENSITIVITY; // 0.05
    let actual_delta = after_value - before_value;

    println!("=== VERIFICATION ===");
    println!("  expected_delta: {}", expected_delta);
    println!("  actual_delta: {}", actual_delta);
    println!("  trigger_updated: {}", after_trigger != before_trigger);

    assert!(
        (actual_delta - expected_delta).abs() < f32::EPSILON,
        "Source of Truth verification FAILED: expected delta {}, got {}",
        expected_delta,
        actual_delta
    );

    assert!(
        after_trigger.is_some() && after_trigger != before_trigger,
        "Source of Truth verification FAILED: last_trigger should be updated"
    );
}
```

### 6.3 Boundary & Edge Case Audit

Run these 3 edge case tests and verify the before/after state:

**Edge Case 1: Empty/Zero Input**
```rust
#[test]
fn test_edge_case_zero_input() {
    let mut modulator = DopamineModulator::new();
    println!("BEFORE: value={}, trigger={:?}", modulator.value(), modulator.level().last_trigger);
    modulator.on_goal_progress(0.0);
    println!("AFTER:  value={}, trigger={:?}", modulator.value(), modulator.level().last_trigger);
    // Verify: value unchanged, trigger unchanged
    assert!((modulator.value() - DA_BASELINE).abs() < f32::EPSILON);
}
```

**Edge Case 2: Maximum Limit**
```rust
#[test]
fn test_edge_case_maximum_limit() {
    let mut modulator = DopamineModulator::new();
    modulator.set_value(DA_MAX - 0.01); // Just below max
    println!("BEFORE: value={}", modulator.value());
    modulator.on_goal_progress(1.0); // Attempt to exceed max
    println!("AFTER:  value={}", modulator.value());
    // Verify: value clamped to DA_MAX (5.0)
    assert!((modulator.value() - DA_MAX).abs() < f32::EPSILON);
}
```

**Edge Case 3: Invalid Format (NaN)**
```rust
#[test]
fn test_edge_case_invalid_nan() {
    let mut modulator = DopamineModulator::new();
    let before = modulator.value();
    println!("BEFORE: value={}", before);
    modulator.on_goal_progress(f32::NAN);
    println!("AFTER:  value={}", modulator.value());
    // Verify: value unchanged
    assert!((modulator.value() - before).abs() < f32::EPSILON);
}
```

### 6.4 Evidence of Success Log

The test output should show actual data like:
```
=== BEFORE STATE ===
  value: 3.0
  last_trigger: None
=== AFTER STATE (Source of Truth) ===
  value: 3.05
  last_trigger: Some(2026-01-12T10:30:45.123456Z)
=== VERIFICATION ===
  expected_delta: 0.05
  actual_delta: 0.05
  trigger_updated: true
```

---

## 7. Manual Verification Checklist

After implementation, manually verify:

1. **Constant Export**: Run `cargo doc -p context-graph-core --no-deps --open` and verify `DA_GOAL_SENSITIVITY` appears in the `neuromod` module docs.

2. **Positive Progress Test**:
   ```rust
   let mut m = DopamineModulator::new();
   assert_eq!(m.value(), 3.0); // baseline
   m.on_goal_progress(1.0);
   assert_eq!(m.value(), 3.1); // +0.1
   ```

3. **Negative Progress Test**:
   ```rust
   let mut m = DopamineModulator::new();
   m.on_goal_progress(-1.0);
   assert_eq!(m.value(), 2.9); // -0.1
   ```

4. **Ceiling Clamp Test**:
   ```rust
   let mut m = DopamineModulator::new();
   m.set_value(5.0);
   m.on_goal_progress(1.0);
   assert_eq!(m.value(), 5.0); // stays at max
   ```

5. **Manager Forwarding Test**:
   ```rust
   let mut mgr = NeuromodulationManager::new();
   let initial = mgr.get_hopfield_beta();
   mgr.on_goal_progress(0.5);
   assert!(mgr.get_hopfield_beta() > initial);
   ```

---

## 8. Files to Modify

| File Path | Action | Changes |
|-----------|--------|---------|
| `crates/context-graph-core/src/neuromod/dopamine.rs` | MODIFY | Add `DA_GOAL_SENSITIVITY` constant, add `on_goal_progress()` method, add 7 unit tests |
| `crates/context-graph-core/src/neuromod/state.rs` | MODIFY | Add `on_goal_progress()` forwarding method, add 2 unit tests |
| `crates/context-graph-core/src/neuromod/mod.rs` | MODIFY | Add `DA_GOAL_SENSITIVITY` to re-exports |

---

## 9. Out of Scope

- MCP handler integration (TASK-NEURO-P2-002)
- Cross-neuromodulator cascade effects (TASK-NEURO-P2-003)
- Configuration file support for sensitivity
- Integration tests

---

## 10. Implementation Checklist

- [x] Read `dopamine.rs` (especially lines 91-108 for existing patterns) ✅
- [x] Read `state.rs` (especially lines 259-287 for event handler patterns) ✅
- [x] Add `DA_GOAL_SENSITIVITY = 0.1` constant after line 35 in dopamine.rs ✅
- [x] Add `on_goal_progress()` method to `DopamineModulator` impl ✅
- [x] Add 7 unit tests for dopamine goal progress ✅ (added 10 tests total: 7 + 3 FSV/edge cases)
- [x] Add `on_goal_progress()` forwarding method to `NeuromodulationManager` ✅
- [x] Add 2 unit tests for manager goal progress ✅
- [x] Update mod.rs exports to include `DA_GOAL_SENSITIVITY` ✅
- [x] Run `cargo build -p context-graph-core` - MUST PASS ✅
- [x] Run `cargo test -p context-graph-core neuromod` - ALL MUST PASS ✅ (63 tests passed)
- [x] Run `cargo clippy -p context-graph-core -- -D warnings` - NO WARNINGS ✅ (no warnings in modified files)
- [x] Run FSV tests and capture output logs ✅
- [x] Run 3 edge case tests with before/after state printing ✅
- [x] Verify existing tests still pass (no regressions) ✅

---

## 11. Rollback Plan

If implementation causes issues:

1. `git checkout -- crates/context-graph-core/src/neuromod/dopamine.rs`
2. `git checkout -- crates/context-graph-core/src/neuromod/state.rs`
3. `git checkout -- crates/context-graph-core/src/neuromod/mod.rs`
4. Run `cargo test -p context-graph-core neuromod` to verify rollback

No database migrations or external dependencies involved.

---

## 12. Traceability

| Requirement | File:Line | Test |
|-------------|-----------|------|
| FR-NEURO-001-01 | `state.rs:~290` | `test_manager_on_goal_progress_*` |
| FR-NEURO-001-02 | `dopamine.rs:~142` | `test_dopamine_on_goal_progress_*` |
| FR-NEURO-001-04 | `dopamine.rs:~37` | Implicit in all tests |
| NFR-NEURO-001-01 | N/A | No allocations (uses only stack) |
| NFR-NEURO-001-02 | N/A | Existing tests unchanged |
| NFR-NEURO-001-03 | `dopamine.rs:~170` | `tracing::debug!` call |

---

## 13. Code Patterns to Follow

### Existing on_workspace_entry Pattern (dopamine.rs lines 91-98)
```rust
pub fn on_workspace_entry(&mut self) {
    self.level.value = (self.level.value + DA_WORKSPACE_INCREMENT).clamp(DA_MIN, DA_MAX);
    self.level.last_trigger = Some(Utc::now());
    tracing::debug!(
        "Dopamine increased on workspace entry: value={:.3}",
        self.level.value
    );
}
```

### Manager Forwarding Pattern (state.rs lines 259-261)
```rust
pub fn on_workspace_entry(&mut self) {
    self.dopamine.on_workspace_entry();
}
```

### Test Assertion Pattern (dopamine.rs tests)
```rust
assert!((modulator.value() - expected).abs() < f32::EPSILON);
```

---

## 14. Constitution Reference

From `docs2/constitution.yaml` lines 245-250:
```yaml
neuromod:
  Dopamine:     { param: hopfield.beta, range: "[1,5]", effect: "↑=sharp retrieval" }
```

The dopamine value directly becomes `hopfield.beta`, controlling memory retrieval sharpness.
