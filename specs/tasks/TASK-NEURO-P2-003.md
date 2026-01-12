# TASK-NEURO-P2-003: Implement Cross-Neuromodulator Cascade Effects

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-NEURO-P2-003 |
| **Title** | Implement Cascade Effects Between Neuromodulators |
| **Status** | **COMPLETED** (2026-01-12) |
| **Priority** | P3 (Enhancement) |
| **Layer** | logic |
| **Sequence** | 3 (after TASK-NEURO-P2-001) |
| **Estimated Complexity** | Medium |
| **Implements** | SPEC-NEURO-001 Section 8.2 |
| **Depends On** | TASK-NEURO-P2-001 (**COMPLETED** - `on_goal_progress()` exists) |

---

## CRITICAL: Read This First

**NO BACKWARDS COMPATIBILITY. FAIL FAST.** If something breaks, it must error with clear logging—no workarounds, no fallbacks, no mock data.

**NO MOCK DATA IN TESTS.** All tests must use real data structures and verify actual outcomes against the Source of Truth.

**VERIFY OUTPUTS PHYSICALLY.** After any operation, you MUST manually inspect the actual stored values to confirm the expected outcome occurred.

---

## 1. Context

### 1.1 What This Task Does

This task implements cross-neuromodulator cascade effects where changes in Dopamine (DA) can trigger secondary effects in Serotonin (5HT) and Norepinephrine (NE). This creates a more biologically plausible interconnected neuromodulation system.

### 1.2 Current State (Verified 2026-01-12)

**File: `crates/context-graph-core/src/neuromod/state.rs`**

The `NeuromodulationManager` struct (line 133-138):
```rust
pub struct NeuromodulationManager {
    dopamine: DopamineModulator,        // PRIVATE field
    serotonin: SerotoninModulator,      // PRIVATE field
    noradrenaline: NoradrenalineModulator, // PRIVATE field
    last_update: Instant,
}
```

**Existing methods on `NeuromodulationManager`:**
- `on_goal_progress(&mut self, delta: f32)` - line 296 - forwards to dopamine only
- `adjust(&mut self, modulator: ModulatorType, delta: f32)` - line 190 - adjusts any modulator
- `dopamine(&self) -> &DopamineModulator` - line 348 - IMMUTABLE reference
- `serotonin(&self) -> &SerotoninModulator` - line 353 - IMMUTABLE reference
- `noradrenaline(&self) -> &NoradrenalineModulator` - line 358 - IMMUTABLE reference

**Current `on_goal_progress` implementation (line 296-298):**
```rust
pub fn on_goal_progress(&mut self, delta: f32) {
    self.dopamine.on_goal_progress(delta);
}
```

**Key insight:** The manager has `&mut self` access to all private fields. Cascade methods can directly mutate `self.serotonin` and `self.noradrenaline` because they're within the same `impl` block.

**File: `crates/context-graph-core/src/neuromod/dopamine.rs`**

Constants (lines 23-39):
```rust
pub const DA_BASELINE: f32 = 3.0;
pub const DA_MIN: f32 = 1.0;
pub const DA_MAX: f32 = 5.0;
pub const DA_GOAL_SENSITIVITY: f32 = 0.1;
```

**File: `crates/context-graph-core/src/neuromod/serotonin.rs`**

- `SerotoninModulator.adjust(&mut self, delta: f32)` - line 140 - adjusts value by delta
- `SerotoninModulator.value(&self) -> f32` - line 90 - returns current value
- Constants: `SEROTONIN_MIN = 0.0`, `SEROTONIN_MAX = 1.0`, `SEROTONIN_BASELINE = 0.5`

**File: `crates/context-graph-core/src/neuromod/noradrenaline.rs`**

- `NoradrenalineModulator.set_value(&mut self, value: f32)` - line 169 - sets clamped value
- `NoradrenalineModulator.value(&self) -> f32` - line 93 - returns current value
- Constants: `NE_MIN = 0.5`, `NE_MAX = 2.0`, `NE_BASELINE = 1.0`

### 1.3 Target State

Add cascade effects so that `on_goal_progress()` triggers:
1. **DA → 5HT cascade (mood correlation):** High DA (>4.0) boosts 5HT; Low DA (<2.0) lowers 5HT
2. **DA → NE cascade (alertness):** Significant DA change (|delta| > 0.3) increases NE

---

## 2. Input Context Files (MUST READ)

| File | Lines | Why You Need It |
|------|-------|-----------------|
| `crates/context-graph-core/src/neuromod/state.rs` | 133-360 | Implementation target - `NeuromodulationManager` struct and methods |
| `crates/context-graph-core/src/neuromod/dopamine.rs` | 23-39, 158-190 | DA constants and `on_goal_progress()` implementation |
| `crates/context-graph-core/src/neuromod/serotonin.rs` | 26-35, 140-147 | 5HT constants and `adjust()` method |
| `crates/context-graph-core/src/neuromod/noradrenaline.rs` | 27-34, 169-171 | NE constants and `set_value()` method |
| `crates/context-graph-core/src/neuromod/mod.rs` | 56-72 | Current exports - will need to add cascade exports |
| `docs2/constitution.yaml` | 245-250 | Neuromod spec reference |

---

## 3. Cascade Rules

### 3.1 Dopamine → Serotonin (Mood Correlation)

| DA Condition | 5HT Effect | Biological Rationale |
|--------------|------------|----------------------|
| DA > 4.0 after goal_progress | 5HT += 0.05 | Success improves mood |
| DA < 2.0 after goal_progress | 5HT -= 0.05 | Failure lowers mood |

### 3.2 Dopamine → Norepinephrine (Alertness)

| DA Change Condition | NE Effect | Biological Rationale |
|---------------------|-----------|----------------------|
| \|DA_delta\| > 0.3 | NE += 0.1 | Significant events increase alertness |

**Note:** The DA_delta is the ACTUAL change in DA value after `on_goal_progress()`, not the raw input delta. With `DA_GOAL_SENSITIVITY = 0.1`, an input delta of 1.0 produces DA_delta of 0.1, which is BELOW the 0.3 threshold.

---

## 4. Scope

### 4.1 In Scope

1. Add `cascade` constants module in `state.rs`
2. Add `CascadeReport` struct to report all cascade effects
3. Add `on_goal_progress_with_cascades(&mut self, delta: f32) -> CascadeReport` method
4. Add private helper methods: `apply_mood_cascade()`, `apply_alertness_cascade()`
5. Add cascade logging with tracing::debug
6. Add unit tests for all cascade paths
7. Export `cascade` module and `CascadeReport` from `mod.rs`

### 4.2 Out of Scope

- Sustained state monitoring (requires background task)
- ACh cascades (managed by GWT MetaCognitiveLoop)
- 5HT/NE initiated cascades
- Modifying existing `on_goal_progress()` behavior (new method added instead)
- MCP integration (separate task)

---

## 5. Exact Implementation

### 5.1 Add to `state.rs` (after line 28, before NeuromodulationState struct)

```rust
/// Cascade configuration constants
pub mod cascade {
    /// DA threshold for positive 5HT cascade (upper quartile of DA range)
    pub const DA_HIGH_THRESHOLD: f32 = 4.0;
    /// DA threshold for negative 5HT cascade (lower quartile of DA range)
    pub const DA_LOW_THRESHOLD: f32 = 2.0;
    /// 5HT adjustment magnitude for DA cascades (~5% of 5HT range)
    pub const SEROTONIN_CASCADE_DELTA: f32 = 0.05;
    /// DA change threshold for NE alertness cascade (~10% of DA range)
    pub const DA_CHANGE_THRESHOLD: f32 = 0.3;
    /// NE adjustment for significant DA change (~7% of NE range)
    pub const NE_ALERTNESS_DELTA: f32 = 0.1;
}

/// Report of cascade effects applied during goal progress
#[derive(Debug, Clone, PartialEq)]
pub struct CascadeReport {
    /// DA delta actually applied (after clamping)
    pub da_delta: f32,
    /// New DA value after adjustment
    pub da_new: f32,
    /// 5HT delta applied from mood cascade (0.0 if not triggered)
    pub serotonin_delta: f32,
    /// New 5HT value after cascade
    pub serotonin_new: f32,
    /// NE delta applied from alertness cascade (0.0 if not triggered)
    pub ne_delta: f32,
    /// New NE value after cascade
    pub ne_new: f32,
    /// Whether mood cascade was triggered
    pub mood_cascade_triggered: bool,
    /// Whether alertness cascade was triggered
    pub alertness_cascade_triggered: bool,
}

impl Default for CascadeReport {
    fn default() -> Self {
        Self {
            da_delta: 0.0,
            da_new: 0.0,
            serotonin_delta: 0.0,
            serotonin_new: 0.0,
            ne_delta: 0.0,
            ne_new: 0.0,
            mood_cascade_triggered: false,
            alertness_cascade_triggered: false,
        }
    }
}
```

### 5.2 Add methods to `impl NeuromodulationManager` (after `on_goal_progress` at ~line 298)

```rust
    /// Handle goal progress with cascade effects to other neuromodulators.
    ///
    /// Applies direct DA modulation, then triggers cascades to 5HT and NE.
    ///
    /// # Cascade Effects
    /// - DA > 4.0 after adjustment: 5HT += 0.05 (mood boost)
    /// - DA < 2.0 after adjustment: 5HT -= 0.05 (mood drop)
    /// - |DA_actual_change| > 0.3: NE += 0.1 (alertness spike)
    ///
    /// # Arguments
    /// * `delta` - Goal progress delta from steering [-1, 1]
    ///
    /// # Returns
    /// `CascadeReport` with all changes applied and new values
    pub fn on_goal_progress_with_cascades(&mut self, delta: f32) -> CascadeReport {
        // Guard against NaN - FAIL FAST
        if delta.is_nan() {
            tracing::warn!("on_goal_progress_with_cascades received NaN delta - returning empty report");
            return CascadeReport {
                da_new: self.dopamine.value(),
                serotonin_new: self.serotonin.value(),
                ne_new: self.noradrenaline.value(),
                ..Default::default()
            };
        }

        // Step 1: Capture DA before adjustment
        let da_old = self.dopamine.value();

        // Step 2: Apply direct DA modulation
        self.dopamine.on_goal_progress(delta);
        let da_new = self.dopamine.value();
        let da_actual_delta = da_new - da_old;

        // Step 3: Apply mood cascade (DA -> 5HT)
        let (serotonin_delta, mood_cascade_triggered) = self.apply_mood_cascade(da_new);
        let serotonin_new = self.serotonin.value();

        // Step 4: Apply alertness cascade (DA change -> NE)
        let (ne_delta, alertness_cascade_triggered) = self.apply_alertness_cascade(da_actual_delta);
        let ne_new = self.noradrenaline.value();

        // Step 5: Log cascade effects
        if mood_cascade_triggered || alertness_cascade_triggered {
            tracing::debug!(
                da_old = da_old,
                da_new = da_new,
                da_actual_delta = da_actual_delta,
                serotonin_delta = serotonin_delta,
                serotonin_new = serotonin_new,
                ne_delta = ne_delta,
                ne_new = ne_new,
                mood_cascade = mood_cascade_triggered,
                alertness_cascade = alertness_cascade_triggered,
                "Neuromodulation cascades applied"
            );
        }

        CascadeReport {
            da_delta: da_actual_delta,
            da_new,
            serotonin_delta,
            serotonin_new,
            ne_delta,
            ne_new,
            mood_cascade_triggered,
            alertness_cascade_triggered,
        }
    }

    /// Apply mood cascade: DA level affects 5HT
    /// Returns (serotonin_delta, triggered)
    fn apply_mood_cascade(&mut self, da_new: f32) -> (f32, bool) {
        if da_new > cascade::DA_HIGH_THRESHOLD {
            self.serotonin.adjust(cascade::SEROTONIN_CASCADE_DELTA);
            (cascade::SEROTONIN_CASCADE_DELTA, true)
        } else if da_new < cascade::DA_LOW_THRESHOLD {
            self.serotonin.adjust(-cascade::SEROTONIN_CASCADE_DELTA);
            (-cascade::SEROTONIN_CASCADE_DELTA, true)
        } else {
            (0.0, false)
        }
    }

    /// Apply alertness cascade: Significant DA change affects NE
    /// Returns (ne_delta, triggered)
    fn apply_alertness_cascade(&mut self, da_actual_delta: f32) -> (f32, bool) {
        if da_actual_delta.abs() > cascade::DA_CHANGE_THRESHOLD {
            let new_ne = self.noradrenaline.value() + cascade::NE_ALERTNESS_DELTA;
            self.noradrenaline.set_value(new_ne);
            (cascade::NE_ALERTNESS_DELTA, true)
        } else {
            (0.0, false)
        }
    }
```

### 5.3 Update `mod.rs` exports (around line 72)

Add after existing exports:
```rust
pub use state::{cascade, CascadeReport};
```

---

## 6. Unit Tests (Add to state.rs `#[cfg(test)] mod tests`)

```rust
    // =========================================================================
    // Cascade Effect Tests (TASK-NEURO-P2-003)
    // =========================================================================

    #[test]
    fn test_cascade_high_da_boosts_serotonin() {
        use super::cascade;

        let mut manager = NeuromodulationManager::new();
        // Set DA just below high threshold
        manager.dopamine.set_value(3.95);
        let initial_5ht = manager.serotonin.value();

        // Large positive delta to push DA above 4.0
        // delta=1.0 * 0.1 sensitivity = 0.1 increase -> DA=4.05
        let report = manager.on_goal_progress_with_cascades(1.0);

        // Verify Source of Truth
        println!("=== HIGH DA -> 5HT CASCADE ===");
        println!("  DA before: 3.95, DA after: {}", manager.dopamine.value());
        println!("  5HT before: {}, 5HT after: {}", initial_5ht, manager.serotonin.value());
        println!("  Report: {:?}", report);

        assert!(report.da_new > cascade::DA_HIGH_THRESHOLD, "DA should exceed 4.0");
        assert!(report.mood_cascade_triggered, "Mood cascade should trigger");
        assert!(
            (report.serotonin_delta - cascade::SEROTONIN_CASCADE_DELTA).abs() < f32::EPSILON,
            "5HT should increase by {}", cascade::SEROTONIN_CASCADE_DELTA
        );
        assert!(
            (manager.serotonin.value() - (initial_5ht + cascade::SEROTONIN_CASCADE_DELTA)).abs() < f32::EPSILON,
            "Source of Truth: 5HT actual value should be increased"
        );
    }

    #[test]
    fn test_cascade_low_da_lowers_serotonin() {
        use super::cascade;

        let mut manager = NeuromodulationManager::new();
        // Set DA just above low threshold
        manager.dopamine.set_value(2.05);
        let initial_5ht = manager.serotonin.value();

        // Large negative delta to push DA below 2.0
        // delta=-1.0 * 0.1 sensitivity = -0.1 decrease -> DA=1.95
        let report = manager.on_goal_progress_with_cascades(-1.0);

        println!("=== LOW DA -> 5HT CASCADE ===");
        println!("  DA before: 2.05, DA after: {}", manager.dopamine.value());
        println!("  5HT before: {}, 5HT after: {}", initial_5ht, manager.serotonin.value());

        assert!(report.da_new < cascade::DA_LOW_THRESHOLD, "DA should be below 2.0");
        assert!(report.mood_cascade_triggered, "Mood cascade should trigger");
        assert!(
            (report.serotonin_delta + cascade::SEROTONIN_CASCADE_DELTA).abs() < f32::EPSILON,
            "5HT should decrease by {}", cascade::SEROTONIN_CASCADE_DELTA
        );
        assert!(
            (manager.serotonin.value() - (initial_5ht - cascade::SEROTONIN_CASCADE_DELTA)).abs() < f32::EPSILON,
            "Source of Truth: 5HT actual value should be decreased"
        );
    }

    #[test]
    fn test_cascade_significant_da_change_increases_ne() {
        use super::cascade;
        use super::super::dopamine::DA_GOAL_SENSITIVITY;

        let mut manager = NeuromodulationManager::new();
        let initial_ne = manager.noradrenaline.value();

        // To get |DA_delta| > 0.3, we need input delta > 3.0 (since 3.0 * 0.1 = 0.3)
        // But delta is typically [-1, 1], so we need to set up DA to cause large actual change
        // Alternative: Manually test the alertness cascade logic directly

        // Test the helper directly by setting DA values that will cause large change
        manager.dopamine.set_value(DA_MIN); // Start at minimum (1.0)

        // Positive delta that would try to increase DA significantly
        // But with sensitivity 0.1, even delta=1.0 only gives 0.1 change
        // So alertness cascade WON'T trigger with normal goal progress

        // Let's test the internal method directly
        let (ne_delta, triggered) = manager.apply_alertness_cascade(0.5); // Simulate large DA change

        println!("=== ALERTNESS CASCADE (direct) ===");
        println!("  Simulated DA change: 0.5");
        println!("  NE before: {}, NE after: {}", initial_ne, manager.noradrenaline.value());
        println!("  NE delta: {}, triggered: {}", ne_delta, triggered);

        assert!(triggered, "Alertness cascade should trigger for |delta|=0.5 > 0.3");
        assert!(
            (ne_delta - cascade::NE_ALERTNESS_DELTA).abs() < f32::EPSILON,
            "NE should increase by {}", cascade::NE_ALERTNESS_DELTA
        );
        assert!(
            manager.noradrenaline.value() > initial_ne,
            "Source of Truth: NE actual value should be increased"
        );
    }

    #[test]
    fn test_cascade_no_trigger_in_normal_range() {
        let mut manager = NeuromodulationManager::new();
        // DA at baseline (3.0), small delta (0.1)
        let initial_5ht = manager.serotonin.value();
        let initial_ne = manager.noradrenaline.value();

        let report = manager.on_goal_progress_with_cascades(0.1);

        println!("=== NO CASCADE (normal range) ===");
        println!("  DA: {} -> {}", 3.0, report.da_new);
        println!("  5HT unchanged: {}", manager.serotonin.value());
        println!("  NE unchanged: {}", manager.noradrenaline.value());

        // DA change = 0.01 (below 0.3 threshold)
        // DA new = 3.01 (between 2.0 and 4.0)
        assert!(!report.mood_cascade_triggered, "Mood cascade should NOT trigger");
        assert!(!report.alertness_cascade_triggered, "Alertness cascade should NOT trigger");
        assert!(
            (manager.serotonin.value() - initial_5ht).abs() < f32::EPSILON,
            "5HT should be unchanged"
        );
        assert!(
            (manager.noradrenaline.value() - initial_ne).abs() < f32::EPSILON,
            "NE should be unchanged"
        );
    }

    #[test]
    fn test_cascade_report_accuracy() {
        use super::cascade;

        let mut manager = NeuromodulationManager::new();
        manager.dopamine.set_value(4.5); // High DA
        let initial_da = manager.dopamine.value();
        let initial_5ht = manager.serotonin.value();

        let report = manager.on_goal_progress_with_cascades(0.5);

        println!("=== REPORT ACCURACY ===");
        println!("  Report DA: old={}, new={}, delta={}", initial_da, report.da_new, report.da_delta);
        println!("  Report 5HT: delta={}, new={}", report.serotonin_delta, report.serotonin_new);
        println!("  Actual DA: {}", manager.dopamine.value());
        println!("  Actual 5HT: {}", manager.serotonin.value());

        // Verify report matches actual state
        assert!(
            (report.da_new - manager.dopamine.value()).abs() < f32::EPSILON,
            "Report DA should match actual"
        );
        assert!(
            (report.serotonin_new - manager.serotonin.value()).abs() < f32::EPSILON,
            "Report 5HT should match actual"
        );
        assert!(
            (report.ne_new - manager.noradrenaline.value()).abs() < f32::EPSILON,
            "Report NE should match actual"
        );

        // DA > 4.0, so mood cascade should trigger
        assert!(report.mood_cascade_triggered);
        assert!(
            (report.serotonin_delta - cascade::SEROTONIN_CASCADE_DELTA).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_cascade_nan_handling() {
        let mut manager = NeuromodulationManager::new();
        let initial_da = manager.dopamine.value();
        let initial_5ht = manager.serotonin.value();
        let initial_ne = manager.noradrenaline.value();

        let report = manager.on_goal_progress_with_cascades(f32::NAN);

        println!("=== NaN HANDLING ===");
        println!("  DA unchanged: {}", manager.dopamine.value());
        println!("  5HT unchanged: {}", manager.serotonin.value());
        println!("  NE unchanged: {}", manager.noradrenaline.value());

        // Nothing should change
        assert!((manager.dopamine.value() - initial_da).abs() < f32::EPSILON);
        assert!((manager.serotonin.value() - initial_5ht).abs() < f32::EPSILON);
        assert!((manager.noradrenaline.value() - initial_ne).abs() < f32::EPSILON);
        assert!(!report.mood_cascade_triggered);
        assert!(!report.alertness_cascade_triggered);
    }

    #[test]
    fn test_cascade_serotonin_clamping() {
        use super::cascade;
        use super::super::serotonin::{SEROTONIN_MAX, SEROTONIN_MIN};

        let mut manager = NeuromodulationManager::new();

        // Test ceiling clamp: 5HT at max, high DA should not exceed max
        manager.serotonin.set_value(SEROTONIN_MAX);
        manager.dopamine.set_value(4.5);
        let report = manager.on_goal_progress_with_cascades(0.1);

        println!("=== 5HT CEILING CLAMP ===");
        println!("  5HT after cascade: {} (max={})", manager.serotonin.value(), SEROTONIN_MAX);

        assert!(
            manager.serotonin.value() <= SEROTONIN_MAX,
            "5HT must not exceed max"
        );

        // Test floor clamp: 5HT at min, low DA should not go below min
        manager.serotonin.set_value(SEROTONIN_MIN);
        manager.dopamine.set_value(1.5);
        let _report = manager.on_goal_progress_with_cascades(-0.1);

        println!("=== 5HT FLOOR CLAMP ===");
        println!("  5HT after cascade: {} (min={})", manager.serotonin.value(), SEROTONIN_MIN);

        assert!(
            manager.serotonin.value() >= SEROTONIN_MIN,
            "5HT must not go below min"
        );
    }
```

---

## 7. Full State Verification (FSV)

### 7.1 Source of Truth Definition

The Source of Truth for neuromodulation state is:
- **Dopamine:** `NeuromodulationManager.dopamine.level.value` (accessed via `manager.dopamine.value()`)
- **Serotonin:** `NeuromodulationManager.serotonin.level.value` (accessed via `manager.serotonin.value()`)
- **Norepinephrine:** `NeuromodulationManager.noradrenaline.level.value` (accessed via `manager.noradrenaline.value()`)

### 7.2 Execute & Inspect Protocol

After implementing, run this FSV test:

```rust
#[test]
fn test_fsv_cascade_source_of_truth() {
    use super::cascade;

    let mut manager = NeuromodulationManager::new();

    // === STEP 1: Establish baseline state ===
    manager.dopamine.set_value(3.95); // Just below high threshold

    println!("=== BEFORE STATE (Source of Truth) ===");
    println!("  DA value: {}", manager.dopamine.value());
    println!("  5HT value: {}", manager.serotonin.value());
    println!("  NE value: {}", manager.noradrenaline.value());

    let before_da = manager.dopamine.value();
    let before_5ht = manager.serotonin.value();
    let before_ne = manager.noradrenaline.value();

    // === STEP 2: Execute the cascade operation ===
    let report = manager.on_goal_progress_with_cascades(1.0);

    // === STEP 3: Read Source of Truth DIRECTLY ===
    println!("=== AFTER STATE (Source of Truth) ===");
    println!("  DA value: {}", manager.dopamine.value());
    println!("  5HT value: {}", manager.serotonin.value());
    println!("  NE value: {}", manager.noradrenaline.value());

    let after_da = manager.dopamine.value();
    let after_5ht = manager.serotonin.value();
    let after_ne = manager.noradrenaline.value();

    // === STEP 4: Verify changes match expectations ===
    println!("=== VERIFICATION ===");
    println!("  DA change: {} -> {} (delta={})", before_da, after_da, after_da - before_da);
    println!("  5HT change: {} -> {} (delta={})", before_5ht, after_5ht, after_5ht - before_5ht);
    println!("  NE change: {} -> {} (delta={})", before_ne, after_ne, after_ne - before_ne);
    println!("  Report matches actual: DA={}, 5HT={}, NE={}",
        (report.da_new - after_da).abs() < f32::EPSILON,
        (report.serotonin_new - after_5ht).abs() < f32::EPSILON,
        (report.ne_new - after_ne).abs() < f32::EPSILON
    );

    // DA should have increased
    assert!(after_da > before_da, "DA should increase");
    // DA > 4.0, so 5HT should increase
    assert!(after_da > cascade::DA_HIGH_THRESHOLD, "DA should exceed threshold");
    assert!(
        (after_5ht - before_5ht - cascade::SEROTONIN_CASCADE_DELTA).abs() < f32::EPSILON,
        "5HT should increase by cascade delta"
    );
    // Report must match actual state
    assert!((report.da_new - after_da).abs() < f32::EPSILON);
    assert!((report.serotonin_new - after_5ht).abs() < f32::EPSILON);
}
```

### 7.3 Boundary & Edge Case Audit

**Edge Case 1: Zero Input Delta**
```rust
#[test]
fn test_edge_case_zero_delta() {
    let mut manager = NeuromodulationManager::new();
    println!("BEFORE: DA={}, 5HT={}, NE={}",
        manager.dopamine.value(), manager.serotonin.value(), manager.noradrenaline.value());

    let report = manager.on_goal_progress_with_cascades(0.0);

    println!("AFTER: DA={}, 5HT={}, NE={}",
        manager.dopamine.value(), manager.serotonin.value(), manager.noradrenaline.value());

    // Verify: nothing changes with zero delta
    assert!(!report.mood_cascade_triggered);
    assert!(!report.alertness_cascade_triggered);
    assert!((report.da_delta).abs() < f32::EPSILON);
}
```

**Edge Case 2: Maximum DA Already at Ceiling**
```rust
#[test]
fn test_edge_case_da_at_ceiling() {
    use super::super::dopamine::DA_MAX;

    let mut manager = NeuromodulationManager::new();
    manager.dopamine.set_value(DA_MAX);
    let initial_5ht = manager.serotonin.value();

    println!("BEFORE: DA={} (max), 5HT={}", manager.dopamine.value(), initial_5ht);

    let report = manager.on_goal_progress_with_cascades(1.0);

    println!("AFTER: DA={}, 5HT={}", manager.dopamine.value(), manager.serotonin.value());

    // DA stays at max, 5HT still increases (DA > 4.0)
    assert!((manager.dopamine.value() - DA_MAX).abs() < f32::EPSILON);
    assert!(report.mood_cascade_triggered); // DA=5.0 > 4.0
    assert!(manager.serotonin.value() > initial_5ht);
}
```

**Edge Case 3: Multiple Cascades in Sequence**
```rust
#[test]
fn test_edge_case_sequential_cascades() {
    let mut manager = NeuromodulationManager::new();
    manager.dopamine.set_value(3.95);

    println!("=== SEQUENTIAL CASCADE TEST ===");

    // First cascade
    let report1 = manager.on_goal_progress_with_cascades(1.0);
    println!("After 1st: DA={}, 5HT={}", manager.dopamine.value(), manager.serotonin.value());

    // Second cascade
    let report2 = manager.on_goal_progress_with_cascades(1.0);
    println!("After 2nd: DA={}, 5HT={}", manager.dopamine.value(), manager.serotonin.value());

    // Third cascade
    let report3 = manager.on_goal_progress_with_cascades(1.0);
    println!("After 3rd: DA={}, 5HT={}", manager.dopamine.value(), manager.serotonin.value());

    // Each should trigger mood cascade if DA > 4.0
    // 5HT should accumulate but clamp at max
}
```

### 7.4 Evidence of Success Log Format

Expected test output should show:
```
=== BEFORE STATE (Source of Truth) ===
  DA value: 3.95
  5HT value: 0.5
  NE value: 1.0
=== AFTER STATE (Source of Truth) ===
  DA value: 4.05
  5HT value: 0.55
  NE value: 1.0
=== VERIFICATION ===
  DA change: 3.95 -> 4.05 (delta=0.1)
  5HT change: 0.5 -> 0.55 (delta=0.05)
  NE change: 1.0 -> 1.0 (delta=0.0)
  Report matches actual: DA=true, 5HT=true, NE=true
```

---

## 8. Files to Modify

| File Path | Action | Description |
|-----------|--------|-------------|
| `crates/context-graph-core/src/neuromod/state.rs` | MODIFY | Add `cascade` module, `CascadeReport` struct, `on_goal_progress_with_cascades()`, helper methods, and tests |
| `crates/context-graph-core/src/neuromod/mod.rs` | MODIFY | Add `pub use state::{cascade, CascadeReport};` export |

---

## 9. Validation Commands

```bash
# Build (must succeed with no errors)
cargo build -p context-graph-core 2>&1

# Run all neuromod tests (must all pass)
cargo test -p context-graph-core neuromod -- --nocapture 2>&1

# Run specific cascade tests
cargo test -p context-graph-core test_cascade -- --nocapture 2>&1

# Run FSV test
cargo test -p context-graph-core test_fsv_cascade -- --nocapture 2>&1

# Clippy (must have no warnings in modified files)
cargo clippy -p context-graph-core -- -D warnings 2>&1

# Doc generation (must succeed)
cargo doc -p context-graph-core --no-deps 2>&1
```

---

## 10. Manual Verification Checklist

After implementation, manually verify by running tests with `--nocapture` and checking:

1. **High DA Cascade Test:**
   - [ ] Set DA to 3.95, call `on_goal_progress_with_cascades(1.0)`
   - [ ] Verify DA increased to ~4.05
   - [ ] Verify 5HT increased by 0.05
   - [ ] Verify `mood_cascade_triggered = true` in report

2. **Low DA Cascade Test:**
   - [ ] Set DA to 2.05, call `on_goal_progress_with_cascades(-1.0)`
   - [ ] Verify DA decreased to ~1.95
   - [ ] Verify 5HT decreased by 0.05
   - [ ] Verify `mood_cascade_triggered = true` in report

3. **Alertness Cascade Test (direct):**
   - [ ] Call `apply_alertness_cascade(0.5)` directly
   - [ ] Verify NE increased by 0.1
   - [ ] Verify returned `triggered = true`

4. **No Cascade Test:**
   - [ ] With DA at baseline (3.0), call `on_goal_progress_with_cascades(0.1)`
   - [ ] Verify 5HT unchanged
   - [ ] Verify NE unchanged
   - [ ] Verify both cascade flags are `false`

5. **Logging Test:**
   - [ ] Run with `RUST_LOG=debug`
   - [ ] Verify cascade log messages appear when cascades trigger
   - [ ] Verify log shows old/new values and cascade flags

---

## 11. Implementation Checklist

- [ ] Read `state.rs` lines 133-360 to understand manager structure
- [ ] Read `dopamine.rs` lines 158-190 for `on_goal_progress` pattern
- [ ] Read `serotonin.rs` line 140 for `adjust()` method signature
- [ ] Read `noradrenaline.rs` line 169 for `set_value()` method signature
- [ ] Add `cascade` module with 5 constants after line 28 in state.rs
- [ ] Add `CascadeReport` struct with 8 fields
- [ ] Add `Default` impl for `CascadeReport`
- [ ] Add `on_goal_progress_with_cascades()` method (~50 lines)
- [ ] Add `apply_mood_cascade()` private helper method
- [ ] Add `apply_alertness_cascade()` private helper method
- [ ] Add 8 unit tests to state.rs test module
- [ ] Update mod.rs exports to include cascade and CascadeReport
- [ ] Run `cargo build -p context-graph-core` - MUST PASS
- [ ] Run `cargo test -p context-graph-core neuromod -- --nocapture` - ALL MUST PASS
- [ ] Run `cargo clippy -p context-graph-core` - NO WARNINGS in modified files
- [ ] Verify all FSV tests show correct before/after state
- [ ] Update task status to COMPLETED

---

## 12. Rollback Plan

If implementation causes issues:

```bash
git checkout -- crates/context-graph-core/src/neuromod/state.rs
git checkout -- crates/context-graph-core/src/neuromod/mod.rs
cargo test -p context-graph-core neuromod  # Verify rollback
```

No database migrations or external dependencies involved.

---

## 13. Traceability

| Requirement | Implemented By | Test Coverage |
|-------------|----------------|---------------|
| SPEC-NEURO-001 Section 8.2 Row 1 (DA→5HT positive) | `apply_mood_cascade()` | `test_cascade_high_da_boosts_serotonin` |
| SPEC-NEURO-001 Section 8.2 Row 2 (DA→5HT negative) | `apply_mood_cascade()` | `test_cascade_low_da_lowers_serotonin` |
| SPEC-NEURO-001 Section 8.2 Row 3 (DA→NE alertness) | `apply_alertness_cascade()` | `test_cascade_significant_da_change_increases_ne` |
| Observability | `tracing::debug!` logging | Manual verification |
| Value clamping | Serotonin/NE modulators | `test_cascade_serotonin_clamping` |
| NaN handling | Guard clause | `test_cascade_nan_handling` |

---

## 14. Important Notes for Implementing Agent

### 14.1 Why New Method Instead of Modifying Existing

The existing `on_goal_progress()` is kept as-is for backwards compatibility in existing call sites. The new `on_goal_progress_with_cascades()` method provides cascade effects when needed. Call sites can be migrated gradually.

### 14.2 Alertness Cascade Reality Check

With `DA_GOAL_SENSITIVITY = 0.1`, normal goal progress deltas [-1, 1] produce DA changes of at most ±0.1, which is BELOW the alertness threshold of 0.3. The alertness cascade is designed for larger DA swings that might occur through:
- Multiple rapid goal progress events
- Direct DA manipulation via other triggers
- Future cascade sources

The test uses `apply_alertness_cascade()` directly to verify the logic works correctly.

### 14.3 Private Field Access

The cascade methods are implemented within `impl NeuromodulationManager` and have direct mutable access to `self.dopamine`, `self.serotonin`, and `self.noradrenaline` because they're private fields within the same struct. No new accessor methods are needed.

---

## Appendix A: Biological Rationale

### Dopamine-Serotonin Interaction

In biological systems, dopamine and serotonin have complex interactions:
- Success/reward (high DA) often correlates with improved mood (5HT)
- Chronic failure/low DA can lead to mood disorders (low 5HT)
- This cascade mimics the reward-mood pathway

### Dopamine-Norepinephrine Interaction

Significant dopaminergic events trigger alertness:
- Large positive rewards increase vigilance for similar opportunities
- Large negative outcomes increase vigilance to avoid repetition
- This cascade mimics the salience-alertness pathway

### Why These Constants?

| Constant | Value | Rationale |
|----------|-------|-----------|
| DA_HIGH_THRESHOLD | 4.0 | Upper quartile of DA range [1,5] |
| DA_LOW_THRESHOLD | 2.0 | Lower quartile of DA range [1,5] |
| SEROTONIN_CASCADE_DELTA | 0.05 | 5% of 5HT range [0,1] per cascade |
| DA_CHANGE_THRESHOLD | 0.3 | ~10% of DA range as "significant" |
| NE_ALERTNESS_DELTA | 0.1 | ~7% of NE range [0.5,2] per cascade |
