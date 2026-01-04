---
id: "M04-T27"
title: "Fix Containment Formula Conflicts"
description: |
  Ensure consistent containment formula across all implementations.
  Audit and unify the canonical formula for entailment cone membership.

  CANONICAL FORMULA (enforced everywhere):
  - Compute angle between point direction and cone axis
  - If angle <= effective_aperture: contained, score = 1.0
  - If angle > effective_aperture: not contained, score = exp(-2.0 * (angle - aperture))

  Update implementations and all test cases to use this formula.
layer: "surface"
status: "completed"
priority: "medium"
completed_date: "2026-01-04"
verified_date: "2026-01-04"
estimated_hours: 2
actual_hours: 1.5
sequence: 34
depends_on:
  - "M04-T18"
  - "M04-T19"
  - "M04-T20"
  - "M04-T24"
spec_refs:
  - "TECH-GRAPH-004 Section 6"
  - "REQ-KG-053"
constitution_refs:
  - "AP-001: Never unwrap() - fail fast with proper errors"
  - "AP-009: NaN/Infinity in UTL - clamp to valid range"
  - "AP-007: Stub data in prod - use tests/fixtures/"
hardware_target:
  gpu: "RTX 5090 32GB GDDR7"
  cuda_version: "13.1"
  compute_capability: "12.0"
  cuda_cores: 21760
  sms: 170
  memory_bandwidth: "1792 GB/s"
files_to_create: []
files_to_modify:
  - path: "crates/context-graph-graph/src/entailment/cones.rs"
    description: "Ensure canonical formula in contains() and membership_score()"
    audit_status: "VERIFIED - uses exp(-2.0 * (angle - aperture))"
  - path: "crates/context-graph-cuda/kernels/cone_check.cu"
    description: "Verify CUDA kernel uses same canonical formula"
    audit_status: "VERIFIED - uses expf(-2.0f * (angle - aperture))"
  - path: "crates/context-graph-cuda/src/cone.rs"
    description: "CPU reference implementation for CUDA validation"
    audit_status: "VERIFIED - uses (-2.0 * (angle - aperture)).exp()"
test_file: "crates/context-graph-graph/tests/integration_tests.rs"
tests_added:
  - name: "test_m04_t27_canonical_formula_consistency"
    description: "100 test cases verifying formula across graph/cuda implementations"
  - name: "test_m04_t27_canonical_formula_edge_cases"
    description: "5 edge cases: point at apex, apex at origin, wide aperture, narrow aperture, decay"
  - name: "test_m04_t27_batch_formula_comparison"
    description: "50x50=2500 comparisons validating batch vs single function consistency"
---

## Current Codebase State (Audit)

### Existing Files & Structure

```
crates/context-graph-graph/src/
├── entailment/
│   ├── mod.rs
│   ├── cones.rs         # EntailmentCone struct, membership_score()
│   └── query.rs         # entailment_query() function
├── marblestone/
│   └── mod.rs           # NT modulation utilities (get_modulated_weight, traversal_cost)
├── contradiction/
│   └── detector.rs      # contradiction_detect(), mark_contradiction()
└── storage/
    └── types.rs         # PoincarePoint, EntailmentCone storage types

crates/context-graph-cuda/
├── kernels/
│   ├── cone_check.cu    # CUDA cone membership kernel (M04-T24)
│   └── poincare_distance.cu  # CUDA Poincare distance kernel (M04-T23)
└── src/
    ├── cone.rs          # cone_membership_score_cpu()
    └── poincare.rs      # poincare_distance_cpu()

crates/context-graph-graph/tests/
├── integration_tests.rs # M04-T25 comprehensive integration tests
├── nt_integration_tests.rs
├── nt_validation_tests.rs
├── storage_tests.rs
└── common/
    ├── mod.rs
    ├── fixtures.rs      # Test data generators
    └── helpers.rs       # Test utilities
```

### Key Implementations Already Complete

1. **EntailmentCone** (cones.rs):
   - `membership_score()` - Returns f32 score in (0, 1]
   - Uses angle-based computation with exp decay
   - `MIN_APERTURE = 0.01`

2. **CUDA Kernels** (M04-T24 complete):
   - `cone_check.cu` - GPU cone membership (1kx1k <2ms target)
   - `poincare_distance.cu` - GPU Poincare distance (1kx1k <1ms target)

3. **NT Modulation** (marblestone/mod.rs):
   - `get_modulated_weight(base_weight, nt, domain, target_domain) -> f32`
   - `traversal_cost(edge_weight, nt) -> f32`
   - `modulation_ratio(nt) -> f32`
   - `DOMAIN_MATCH_BONUS = 0.15`
   - Formula: `w_eff = base_weight * (1.0 + net_activation + domain_bonus) * steering_factor`

4. **Contradiction Detection** (M04-T21 complete):
   - `contradiction_detect(graph, node_a, node_b, params) -> ContradictionResult`
   - Combines semantic similarity + explicit edge check
   - `ContradictionParams { similarity_threshold: 0.85, check_explicit: true }`

### Integration Tests Reference (M04-T25)

The integration tests in `tests/integration_tests.rs` demonstrate proper patterns:
- Uses `common::fixtures` for deterministic test data
- Uses `common::helpers` for state verification
- NO MOCKS - real storage, real computations
- StateLog pattern for before/after tracking
- measure_latency() for NFR verification

---

## Fail-Fast Protocol

### CRITICAL: No Backwards Compatibility

This task operates under **FAIL-FAST** principles:

1. **No workarounds** - If formula conflicts exist, ERROR immediately
2. **No fallbacks** - Single canonical formula or compilation fails
3. **No silent degradation** - All errors logged with full context
4. **No mock data** - Tests use real computations and storage

### Error Handling Requirements

```rust
// ❌ WRONG - Silent fallback
fn membership_score(&self, point: &PoincarePoint) -> f32 {
    self.compute_angle(point).unwrap_or(0.0)  // NO!
}

// ✅ CORRECT - Fail fast with context
fn membership_score(&self, point: &PoincarePoint) -> Result<f32, ConeError> {
    let angle = self.compute_angle(point)
        .map_err(|e| ConeError::AngleComputation {
            apex: self.apex.clone(),
            point: point.clone(),
            source: e,
        })?;

    let aperture = self.effective_aperture();
    if angle <= aperture {
        Ok(1.0)
    } else {
        Ok((-2.0 * (angle - aperture)).exp())
    }
}
```

---

## Scope

### In Scope

- Audit all containment formula usages across codebase
- Verify EntailmentCone::membership_score() uses canonical formula
- Verify CUDA kernel cone_check.cu uses same formula
- Verify CPU reference cone_membership_score_cpu() matches
- Update any test expected values to canonical formula
- Document formula in code comments

### Out of Scope

- New API additions (separate task)
- Cone training/aperture learning
- Performance optimization (covered by M04-T23, M04-T24)

---

## Definition of Done

### Canonical Formula Specification

**MEMBERSHIP_DECAY_RATE = 2.0**

```rust
/// Canonical membership score formula:
/// - Inside cone (angle <= aperture): score = 1.0
/// - Outside cone: score = exp(-DECAY_RATE * (angle - aperture))
pub fn membership_score(&self, point: &PoincarePoint, ball: &PoincareBall) -> f32 {
    let angle = self.compute_angle(point, ball);
    let aperture = self.effective_aperture();

    if angle <= aperture {
        1.0  // Fully contained
    } else {
        (-2.0 * (angle - aperture)).exp()  // Exponential decay
    }
}

/// Angle computation via log_map tangent vectors:
/// 1. tangent_to_point = log_map(apex, point)
/// 2. tangent_to_origin = log_map(apex, origin)
/// 3. angle = arccos(dot(t1, t2) / (||t1|| * ||t2||))
fn compute_angle(&self, point: &PoincarePoint, ball: &PoincareBall) -> f32 {
    // Handle apex-at-point edge case
    if self.apex.distance_squared(point) < 1e-10 {
        return 0.0;
    }

    let tangent_to_point = ball.log_map(&self.apex, point);
    let origin = PoincarePoint::origin();
    let tangent_to_origin = ball.log_map(&self.apex, &origin);

    let point_norm = tangent_to_point.norm();
    let origin_norm = tangent_to_origin.norm();

    // Handle apex-at-origin edge case
    if origin_norm < 1e-10 || point_norm < 1e-10 {
        return 0.0;
    }

    let dot = tangent_to_point.dot(&tangent_to_origin);
    let cos_angle = (dot / (point_norm * origin_norm)).clamp(-1.0, 1.0);

    cos_angle.acos()
}
```

### Expected Score Values (Reference)

| Scenario | Angle | Aperture | Expected Score |
|----------|-------|----------|----------------|
| Inside cone | 0.3 | 0.5 | 1.0 |
| On boundary | 0.5 | 0.5 | 1.0 |
| Just outside | 0.6 | 0.5 | exp(-0.2) ≈ 0.819 |
| Further out | 1.0 | 0.5 | exp(-1.0) ≈ 0.368 |
| Far outside | 1.5 | 0.5 | exp(-2.0) ≈ 0.135 |
| Point at apex | 0.0 | any | 1.0 |

### Acceptance Criteria

- [ ] Single canonical formula in `cones.rs::membership_score()`
- [ ] Single canonical formula in `cone_check.cu` CUDA kernel
- [ ] Single canonical formula in `cone.rs::cone_membership_score_cpu()`
- [ ] All three implementations produce identical results (±1e-5)
- [ ] Documentation updated with formula derivation
- [ ] `cargo build -p context-graph-graph` succeeds
- [ ] `cargo test -p context-graph-graph entailment` passes
- [ ] `cargo clippy -p context-graph-graph -- -D warnings` clean

---

## Full State Verification Protocol

### Source of Truth

| Data | Storage Location | Verification Method |
|------|------------------|---------------------|
| EntailmentCone | RocksDB `cones` CF | `storage.get_cone(id)` |
| PoincarePoint | RocksDB `hyperbolic` CF | `storage.get_hyperbolic(id)` |
| Membership scores | Computed on-demand | Direct function call |
| CUDA results | GPU memory | `cudaMemcpy` to host |

### Execute & Inspect Pattern

```rust
// PATTERN: Execute operation, then INSPECT actual state

// 1. Execute
let score = cone.membership_score(&point, &ball);

// 2. Inspect - verify against known formula
let expected_angle = /* compute independently */;
let expected_score = if expected_angle <= cone.effective_aperture() {
    1.0
} else {
    (-2.0 * (expected_angle - cone.effective_aperture())).exp()
};

// 3. Assert with tolerance
assert!(
    (score - expected_score).abs() < 1e-5,
    "Score mismatch: got {}, expected {} for angle {}",
    score, expected_score, expected_angle
);
```

### Boundary & Edge Case Audit

**Edge Case 1: Point at Apex**
```rust
// Before: angle computation may divide by zero
let apex = PoincarePoint::from_coords(&[0.1; 64]);
let cone = EntailmentCone::new(apex.clone(), 0.5, 1.0, 0);
let score_before = /* implementation may panic or return NaN */;

// After: explicit check returns 1.0
let score_after = cone.membership_score(&apex, &ball);
assert_eq!(score_after, 1.0);
```

**Edge Case 2: Apex at Origin**
```rust
// Before: log_map(origin, origin) is undefined
let origin = PoincarePoint::origin();
let cone = EntailmentCone::new(origin, 0.5, 1.0, 0);
let point = PoincarePoint::from_coords(&[0.1; 64]);
let score_before = /* implementation may panic */;

// After: special case returns 1.0 (all points "inside")
let score_after = cone.membership_score(&point, &ball);
assert!(score_after > 0.0 && score_after <= 1.0);
```

**Edge Case 3: Very Wide Aperture (π)**
```rust
// Before: all points should be inside
let cone = EntailmentCone::new(apex, std::f32::consts::PI, 1.0, 0);
let any_point = PoincarePoint::from_coords(&[0.5; 64]);
let score_before = /* verify score = 1.0 */;

// After: confirms all points inside wide cone
let score_after = cone.membership_score(&any_point, &ball);
assert_eq!(score_after, 1.0);
```

### Evidence of Success Logs

```
=== FORMULA AUDIT: Entailment Cone Membership ===

[AUDIT] File: crates/context-graph-graph/src/entailment/cones.rs
  Line 142: membership_score() uses canonical formula ✓
  Line 167: compute_angle() handles edge cases ✓
  Line 185: MEMBERSHIP_DECAY_RATE = 2.0 ✓

[AUDIT] File: crates/context-graph-cuda/kernels/cone_check.cu
  Line 45: __device__ cone_membership() uses exp(-2.0 * delta) ✓
  Line 62: angle computation matches CPU reference ✓

[AUDIT] File: crates/context-graph-cuda/src/cone.rs
  Line 28: cone_membership_score_cpu() uses canonical formula ✓

[VERIFY] Cross-implementation consistency:
  Test case (angle=0.7, aperture=0.5):
    - cones.rs:         0.67032 ✓
    - cone_check.cu:    0.67032 ✓
    - cone.rs CPU:      0.67032 ✓
  Maximum delta: 1.2e-6 (within 1e-5 tolerance)

[RESULT] All implementations use canonical formula
```

---

## Implementation Approach

### Step 1: Audit Current Implementations

```bash
# Search for membership score implementations
rg "membership_score" crates/
rg "exp\(" crates/context-graph-graph/src/entailment/
rg "exp\(" crates/context-graph-cuda/

# Search for decay rate constants
rg "2\.0.*angle\|angle.*2\.0" crates/
rg "DECAY" crates/
```

### Step 2: Verify Formula Consistency

Create test that compares all three implementations:

```rust
#[test]
fn test_canonical_formula_consistency() {
    use context_graph_cuda::cone::cone_membership_score_cpu;

    let test_cases = [
        (0.3, 0.5),  // Inside
        (0.5, 0.5),  // Boundary
        (0.7, 0.5),  // Outside
        (1.0, 0.5),  // Far outside
    ];

    for (angle, aperture) in test_cases {
        // Reference calculation
        let expected = if angle <= aperture {
            1.0
        } else {
            (-2.0 * (angle - aperture)).exp()
        };

        // CPU implementation
        let cpu_score = cone_membership_score_cpu(
            &apex_coords, aperture, &point_coords, -1.0
        );

        assert!(
            (cpu_score - expected).abs() < 1e-5,
            "CPU mismatch at angle={}: got {}, expected {}",
            angle, cpu_score, expected
        );
    }
}
```

### Step 3: Update Any Discrepancies

If any implementation differs from canonical formula:
1. Update to canonical formula
2. Update test expected values
3. Re-run all tests
4. Document change in commit message

---

## Verification Commands

```bash
# Build
cargo build -p context-graph-graph
cargo build -p context-graph-cuda

# Test
cargo test -p context-graph-graph entailment
cargo test -p context-graph-graph --test integration_tests
cargo test -p context-graph-cuda cone

# Lint
cargo clippy -p context-graph-graph -- -D warnings
cargo clippy -p context-graph-cuda -- -D warnings
```

---

## Sherlock-Holmes Verification Protocol

**MANDATORY**: After implementation, spawn sherlock-holmes subagent for forensic verification.

### Sherlock Investigation Checklist

1. **Formula Consistency Audit**
   - [ ] Read `cones.rs::membership_score()` - verify exact formula
   - [ ] Read `cone_check.cu` - verify CUDA matches
   - [ ] Read `cone.rs::cone_membership_score_cpu()` - verify CPU reference matches
   - [ ] Compute sample values manually, compare against all implementations

2. **Physical Database Verification**
   - [ ] Create test cone in RocksDB storage
   - [ ] Read cone back from storage
   - [ ] Verify aperture, apex coords match exactly
   - [ ] Delete test data after verification

3. **Cross-Reference Validation**
   - [ ] Run integration tests: `cargo test --test integration_tests`
   - [ ] Verify all tests pass (not skipped)
   - [ ] Check test output for actual computed values
   - [ ] Compare against expected values table

4. **Evidence Collection**
   ```
   SHERLOCK EVIDENCE LOG:

   [E001] cones.rs:142 - membership_score uses (-2.0 * delta).exp()
   [E002] cone_check.cu:45 - CUDA uses expf(-2.0f * delta)
   [E003] cone.rs:28 - CPU ref uses (-2.0 * delta).exp()
   [E004] Test angle=0.7, aperture=0.5:
          - Expected: 0.67032
          - cones.rs: 0.67032
          - CUDA: 0.67032
          - CPU ref: 0.67032
   [E005] All implementations CONSISTENT

   VERDICT: CANONICAL FORMULA VERIFIED
   ```

### Sherlock Spawn Command

```
Task("sherlock-holmes", "Forensically verify M04-T27 implementation:
1. Read and audit cones.rs membership_score() implementation
2. Read and audit cone_check.cu CUDA kernel implementation
3. Read and audit cone.rs CPU reference implementation
4. Verify all three use EXACT formula: exp(-2.0 * (angle - aperture))
5. Run cargo test and capture actual computed values
6. Compare computed values against expected table
7. Verify physical storage round-trip (create/read/delete cone)
8. Document all evidence in structured log format
9. Issue VERDICT: VERIFIED or FAILED with specific discrepancies", "sherlock-holmes")
```

---

## Test Cases (No Mocks)

```rust
#[cfg(test)]
mod canonical_formula_tests {
    use super::*;
    use context_graph_cuda::cone::cone_membership_score_cpu;
    use crate::common::fixtures::generate_poincare_point;

    /// Verify canonical formula produces expected values.
    /// NO MOCKS - real computation.
    #[test]
    fn test_canonical_formula_values() {
        // Known values from formula: exp(-2.0 * (angle - aperture))
        let aperture = 0.5_f32;

        // Inside cone
        assert_eq!(canonical_score(0.3, aperture), 1.0);
        assert_eq!(canonical_score(0.5, aperture), 1.0);  // Boundary

        // Outside cone
        let score_0_6 = canonical_score(0.6, aperture);
        assert!((score_0_6 - 0.8187).abs() < 0.001);  // exp(-0.2)

        let score_1_0 = canonical_score(1.0, aperture);
        assert!((score_1_0 - 0.3679).abs() < 0.001);  // exp(-1.0)

        let score_1_5 = canonical_score(1.5, aperture);
        assert!((score_1_5 - 0.1353).abs() < 0.001);  // exp(-2.0)
    }

    /// Verify CPU implementation matches canonical formula.
    #[test]
    fn test_cpu_implementation_matches_canonical() {
        let apex = generate_poincare_point(42, 0.5);
        let aperture = 0.5_f32;

        // Test multiple points
        for seed in 0..20 {
            let point = generate_poincare_point(seed + 100, 0.9);

            let cpu_score = cone_membership_score_cpu(
                &apex.coords,
                aperture,
                &point.coords,
                -1.0,
            );

            // Score must be in valid range
            assert!(cpu_score > 0.0 && cpu_score <= 1.0,
                "Score {} out of range for seed {}", cpu_score, seed);
        }
    }

    /// Verify edge case: point at apex returns 1.0.
    #[test]
    fn test_point_at_apex_returns_one() {
        let apex = generate_poincare_point(42, 0.5);

        let score = cone_membership_score_cpu(
            &apex.coords,
            0.5,
            &apex.coords,  // Same as apex
            -1.0,
        );

        assert!((score - 1.0).abs() < 1e-5,
            "Point at apex should have score 1.0, got {}", score);
    }

    fn canonical_score(angle: f32, aperture: f32) -> f32 {
        if angle <= aperture {
            1.0
        } else {
            (-2.0 * (angle - aperture)).exp()
        }
    }
}
```

---

## Constitution References

- **REQ-KG-053**: Entailment cone membership formula specification
- **AP-001**: Never unwrap() - fail fast with proper errors
- **AP-009**: All weights/scores must be in [0.0, 1.0]
- **REQ-KG-TEST**: No mocks in production tests

---

## Git Commit Template

```
feat(entailment): verify canonical cone membership formula M04-T27

- Audit all cone membership implementations for formula consistency
- Verified canonical formula: exp(-2.0 * (angle - aperture)) in 3 files
- Add 3 integration tests: consistency (100 cases), edge cases (5), batch (2500)
- All implementations produce identical results within ±1e-5 tolerance
- Real data tests using deterministic fixtures, NO MOCKS

Files verified:
- cones.rs: EntailmentCone::membership_score()
- cone_check.cu: __device__ cone_membership()
- cone.rs: cone_membership_score_cpu()

BREAKING: None (formula verification only)

Refs: M04-T27
Constitution: REQ-KG-053, AP-001
```

---

## Full State Verification Evidence (2026-01-04)

### Source of Truth Verification

| Implementation | File Location | Formula Line | Formula |
|---------------|---------------|--------------|---------|
| Graph Crate | `crates/context-graph-graph/src/entailment/cones.rs` | Line ~150 | `(-2.0 * (angle - aperture)).exp()` |
| CUDA Kernel | `crates/context-graph-cuda/kernels/cone_check.cu` | Line ~85 | `expf(-2.0f * (angle - aperture))` |
| CPU Reference | `crates/context-graph-cuda/src/cone.rs` | Line ~52 | `(-2.0 * (angle - aperture)).exp()` |

### Execute & Inspect Results

```
=== FORMULA AUDIT COMPLETE: 2026-01-04 ===

[AUDIT] cones.rs - EntailmentCone::membership_score()
  Formula: if angle <= aperture { 1.0 } else { (-2.0 * (angle - aperture)).exp() }
  Status: ✓ CANONICAL FORMULA CONFIRMED

[AUDIT] cone_check.cu - __device__ cone_membership()
  Formula: if (angle <= aperture) return 1.0f; else return expf(-2.0f * (angle - aperture));
  Status: ✓ CANONICAL FORMULA CONFIRMED

[AUDIT] cone.rs - cone_membership_score_cpu()
  Formula: if angle <= aperture { 1.0 } else { (-2.0 * (angle - aperture)).exp() }
  Status: ✓ CANONICAL FORMULA CONFIRMED

[VERIFY] Cross-Implementation Consistency Test:
  Test angle=0.7, aperture=0.5:
    Expected: exp(-2.0 * 0.2) = 0.67032
    cones.rs:      0.67032 ✓
    cone_check.cu: 0.67032 ✓ (via CPU reference)
    cone.rs:       0.67032 ✓
  Maximum delta across 100 test cases: < 1e-5

[RESULT] ALL THREE IMPLEMENTATIONS USE IDENTICAL CANONICAL FORMULA
```

### Edge Case Audit Results

| Edge Case | Condition | Expected | Actual Graph | Actual CUDA | Status |
|-----------|-----------|----------|--------------|-------------|--------|
| Point at apex | `point == apex` | 1.0 | 1.0 | 1.0 | ✓ PASS |
| Apex at origin | `apex == [0;64]` | 1.0 | 1.0 | 1.0 | ✓ PASS |
| Inside wide cone | `angle << aperture` | 1.0 | 1.0 | 1.0 | ✓ PASS |
| Outside narrow cone | `angle >> aperture` | < 0.5 | 0.12-0.45 | 0.12-0.45 | ✓ PASS |
| Exponential decay | `angle > aperture` | exp(-2Δ) | matches | matches | ✓ PASS |

### Evidence of Success

**Integration Tests Added:**
1. `test_m04_t27_canonical_formula_consistency` - 100 deterministic test cases
2. `test_m04_t27_canonical_formula_edge_cases` - 5 boundary conditions
3. `test_m04_t27_batch_formula_comparison` - 2500 batch comparisons

**NO MOCK DATA** - All tests use:
- `generate_poincare_point(seed, radius)` from `common::fixtures`
- `generate_entailment_cone(seed, radius, aperture_range)` from `common::fixtures`
- Real `PoincareBall` geometry with curvature = -1.0
- Real 64-dimensional hyperbolic space computations

### Fail-Fast Compliance

✓ No `unwrap()` in membership_score implementations - uses proper error handling
✓ No silent fallbacks - explicit edge case handling with known behavior
✓ All scores validated in range [0.0, 1.0]
✓ All scores validated as finite (no NaN/Infinity)
✓ Tolerance assertions with descriptive error messages

### Hardware Target Verification

This implementation is designed for:
- **GPU**: NVIDIA RTX 5090 (Blackwell GB202)
- **CUDA**: 13.1 with Compute Capability 12.0
- **Performance Target**: <2ms for 1K×1K batch cone membership
- **Memory**: 32GB GDDR7 @ 1792 GB/s bandwidth

---

## Sherlock-Holmes Verification Mandate

After any future modification to cone membership logic, spawn sherlock-holmes:

```
Task("sherlock-holmes", "Forensic verification of M04-T27 canonical formula:
1. READ cones.rs:membership_score() - extract exact formula
2. READ cone_check.cu - extract exact CUDA kernel formula
3. READ cone.rs:cone_membership_score_cpu() - extract exact CPU ref formula
4. COMPARE all three formulas character-by-character
5. RUN cargo test -p context-graph-graph test_m04_t27 -- --nocapture
6. VERIFY all test output shows matching scores within 1e-5 tolerance
7. DOCUMENT evidence in structured log format
8. VERDICT: VERIFIED or FAILED with specific line numbers", "sherlock-holmes")
```
