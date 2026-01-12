# TASK-UTL-P1-005: Implement TransEEntropy for E11 (Entity/KnowledgeGraph)

**Priority:** P1
**Status:** completed
**Completed:** 2026-01-12
**Spec Reference:** SPEC-UTL-003
**Estimated Effort:** 3-4 hours
**Implements:** REQ-UTL-003-05, REQ-UTL-003-06

---

## CRITICAL: Read First

**Constitution Reference (line 165 in docs2/constitution.yaml):**
```yaml
ΔS: { E1: "GMM+Mahalanobis", E5: "Asymmetric KNN", E7: "GMM+KNN hybrid", E9: "Hamming", E10: "Cross-modal KNN", E11: "TransE ||h+r-t||", E12: "Token KNN", E13: "Jaccard", default: "KNN" }
```

**Current State (WRONG):** Factory routes `Embedder::Entity` to `DefaultKnnEntropy` (line 87-89 in `factory.rs`).
**Required State:** Factory MUST route `Embedder::Entity` to `TransEEntropy`.

**NO BACKWARDS COMPATIBILITY.** If something doesn't work, it MUST error out with clear error messages. Do not create workarounds, fallbacks, or mock data.

---

## Summary

Create a specialized entropy calculator for E11 (Entity) embeddings using TransE distance metrics.

**TransE Model:** Knowledge graph relationships as translations in embedding space.
- For valid triple (h, r, t): h + r ≈ t
- Distance: d(h,r,t) = ||h + r - t||
- E11 dimension: **384D** (MiniLM)

---

## Current Codebase State (Verified 2026-01-12)

### Files That Exist

| File | Purpose | Status |
|------|---------|--------|
| `crates/context-graph-utl/src/surprise/embedder_entropy/mod.rs` | Trait definition, module exports | EXISTS - needs update |
| `crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs` | Factory routing | EXISTS - needs update |
| `crates/context-graph-utl/src/surprise/embedder_entropy/cross_modal.rs` | E10 implementation (REFERENCE) | EXISTS - use as template |
| `crates/context-graph-utl/src/surprise/embedder_entropy/hybrid_gmm_knn.rs` | E7 implementation (REFERENCE) | EXISTS - use as template |
| `crates/context-graph-utl/src/config/surprise.rs` | SurpriseConfig struct | EXISTS - needs update |
| `crates/context-graph-utl/src/error.rs` | UtlError, UtlResult types | EXISTS |
| `crates/context-graph-embeddings/src/models/pretrained/entity/transe.rs` | TransE scoring (REFERENCE) | EXISTS - shows TransE math |

### Files To Create

| File | Purpose |
|------|---------|
| `crates/context-graph-utl/src/surprise/embedder_entropy/transe.rs` | TransEEntropy implementation |

### Key Constants (from codebase)

```rust
// From crates/context-graph-core/src/teleological/embedder.rs
pub const E11_DIM: usize = 384;  // Entity dimension

// From constitution.yaml line 364
// E11_EntityKG: { dim: 384, type: dense, quant: Float8, use: "Multi-modal links" }
```

---

## Implementation Requirements

### 1. Create `transe.rs` File

**Path:** `crates/context-graph-utl/src/surprise/embedder_entropy/transe.rs`

**Algorithm:**
1. Parse current embedding as (head_context, relation_context)
   - First half (0..192) = head entity context
   - Second half (192..384) = relation context
2. For each history embedding (treated as potential tail):
   - Compute TransE distance: ||head + relation - tail||
3. Average top-k smallest distances
4. Normalize via sigmoid
5. Clamp to [0.0, 1.0]

**Struct Definition:**
```rust
/// E11 (Entity) entropy using TransE distance.
///
/// TransE models knowledge graph relationships as translations:
/// For triple (head, relation, tail): head + relation ≈ tail
///
/// # Constitution Reference
/// E11: "TransE: ΔS=||h+r-t||" (constitution.yaml line 165)
#[derive(Debug, Clone)]
pub struct TransEEntropy {
    /// Dimension split point for head vs relation.
    /// Default: dim / 2 (E11 is 384D, so split at 192)
    split_point: usize,
    /// L-norm for distance (1 = L1/Manhattan, 2 = L2/Euclidean).
    /// Default: 2 (L2 norm per original TransE paper)
    norm: u8,
    /// Running mean for distance normalization.
    running_mean: f32,
    /// Running variance for distance normalization.
    running_variance: f32,
    /// Number of samples seen.
    sample_count: usize,
    /// k neighbors for averaging.
    k_neighbors: usize,
}
```

**Required Methods:**
```rust
impl TransEEntropy {
    pub fn new() -> Self;
    pub fn with_norm(norm: u8) -> Self;
    pub fn from_config(config: &SurpriseConfig) -> Self;
    pub fn with_split_point(self, split_point: usize) -> Self;
    pub fn with_k_neighbors(self, k: usize) -> Self;

    fn extract_head(&self, embedding: &[f32]) -> &[f32];
    fn extract_relation(&self, embedding: &[f32]) -> &[f32];
    fn compute_transe_distance(&self, head: &[f32], relation: &[f32], tail: &[f32]) -> f32;
    fn sigmoid(x: f32) -> f32;
}

impl Default for TransEEntropy { ... }

impl EmbedderEntropy for TransEEntropy {
    fn compute_delta_s(&self, current: &[f32], history: &[Vec<f32>], k: usize) -> UtlResult<f32>;
    fn embedder_type(&self) -> Embedder;  // MUST return Embedder::Entity
    fn reset(&mut self);
}
```

### 2. Update Module Exports

**File:** `crates/context-graph-utl/src/surprise/embedder_entropy/mod.rs`

**Add after line 42:**
```rust
mod transe;
pub use transe::TransEEntropy;
```

### 3. Update Factory Routing

**File:** `crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs`

**Change lines 87-89 FROM:**
```rust
| Embedder::Entity
| Embedder::LateInteraction => {
    Box::new(DefaultKnnEntropy::from_config(embedder, config))
}
```

**TO:**
```rust
// E11 (Entity): TransE distance per constitution.yaml delta_methods.ΔS E11
Embedder::Entity => Box::new(TransEEntropy::from_config(config)),

// E12 (LateInteraction): DefaultKnn (Token KNN pending TASK-UTL-P1-006)
Embedder::LateInteraction => {
    Box::new(DefaultKnnEntropy::from_config(embedder, config))
}
```

### 4. Update SurpriseConfig

**File:** `crates/context-graph-utl/src/config/surprise.rs`

**Add after line 134 (after multimodal_k_neighbors):**
```rust
// --- TransE (E11 Entity) ---

/// L-norm for TransE distance (1 = L1, 2 = L2).
/// Default: 2 (L2 per original TransE paper)
/// Range: `[1, 2]`
pub entity_transe_norm: u8,

/// Split ratio for head/relation in embedding.
/// Default: 0.5 (split at midpoint: 192 for 384D)
/// Range: `[0.1, 0.9]`
pub entity_split_ratio: f32,

/// k neighbors for TransE entropy averaging.
/// Range: `[1, 20]`
pub entity_k_neighbors: usize,
```

**Add to Default impl after line 169:**
```rust
// TransE (E11 Entity) - per constitution.yaml delta_methods.ΔS E11
entity_transe_norm: 2,
entity_split_ratio: 0.5,
entity_k_neighbors: 5,
```

---

## Required Tests (in transe.rs tests module)

| Test Name | Description | Expected Outcome |
|-----------|-------------|------------------|
| `test_transe_empty_history_returns_one` | Empty history | delta_s = 1.0 |
| `test_transe_empty_input_error` | Empty current | Err(UtlError::EmptyInput) |
| `test_transe_identical_returns_low` | Same embedding | delta_s < 0.5 |
| `test_transe_perfect_translation` | h + r = t exactly | delta_s near 0 |
| `test_transe_orthogonal_returns_high` | Unrelated entities | delta_s > 0.5 |
| `test_transe_embedder_type` | Verify type | Embedder::Entity |
| `test_transe_valid_range` | Various inputs | All outputs in [0.0, 1.0] |
| `test_transe_no_nan_infinity` | Edge cases | No NaN/Infinity (AP-10) |
| `test_transe_l1_vs_l2_norm` | Different norms | Different distances |
| `test_transe_from_config` | Config values | Config applied correctly |
| `test_transe_head_extraction` | Extract first half | Correct slice returned |
| `test_transe_relation_extraction` | Extract second half | Correct slice returned |
| `test_transe_distance_formula` | Manual calculation | ||h + r - t|| correct |
| `test_transe_reset` | Reset state | State cleared properly |
| `test_transe_nan_input_error` | NaN in input | Err(UtlError::EntropyError) |
| `test_transe_infinity_input_error` | Infinity in input | Err(UtlError::EntropyError) |

---

## Validation Commands

```bash
# 1. Build (MUST pass)
cargo build -p context-graph-utl

# 2. Run TransE tests (MUST pass all)
cargo test -p context-graph-utl transe -- --nocapture

# 3. Run factory tests (MUST pass - verify routing)
cargo test -p context-graph-utl factory -- --nocapture

# 4. Run full module tests (MUST pass)
cargo test -p context-graph-utl embedder_entropy -- --nocapture

# 5. Clippy (MUST have no warnings)
cargo clippy -p context-graph-utl -- -D warnings

# 6. Check formatter
cargo fmt --check -p context-graph-utl
```

---

## Full State Verification (MANDATORY)

After implementing, you MUST verify:

### Source of Truth Verification

1. **Factory Creates Correct Type:**
```rust
// In test or manual verification:
let config = SurpriseConfig::default();
let calculator = EmbedderEntropyFactory::create(Embedder::Entity, &config);
assert_eq!(calculator.embedder_type(), Embedder::Entity);
// Print to verify: println!("E11 calculator type: {:?}", calculator.embedder_type());
```

2. **TransE Distance Calculation:**
```rust
// Manual verification with known values:
// h = [1.0; 192], r = [0.5; 192], t = [1.5; 192]
// Expected: ||h + r - t|| = 0.0 (perfect translation)
let h = vec![1.0f32; 192];
let r = vec![0.5f32; 192];
let t = vec![1.5f32; 192];
// Compute and verify distance is ~0
```

### Boundary & Edge Case Audit

**Edge Case 1: Empty History**
```rust
let current = vec![0.5f32; 384];
let history: Vec<Vec<f32>> = vec![];
println!("BEFORE: history.len() = {}", history.len());
let result = calculator.compute_delta_s(&current, &history, 5);
println!("AFTER: delta_s = {:?}", result);
// Expected: Ok(1.0)
```

**Edge Case 2: Perfect Translation (Low Entropy)**
```rust
// current = [head(192) | relation(192)] where head + relation = history[0]
let mut current = vec![0.0f32; 384];
for i in 0..192 { current[i] = 0.5; }      // head = 0.5
for i in 192..384 { current[i] = 0.3; }    // relation = 0.3
let tail = vec![0.8f32; 384];              // 0.5 + 0.3 = 0.8 (perfect)
let history = vec![tail; 10];
println!("BEFORE: perfect translation setup");
let result = calculator.compute_delta_s(&current, &history, 5);
println!("AFTER: delta_s = {:?} (expected low, near 0)", result);
```

**Edge Case 3: Maximum Dimension Embedding**
```rust
let current = vec![f32::MAX / 1000.0; 384];
let history = vec![vec![f32::MAX / 1000.0; 384]; 5];
println!("BEFORE: large values");
let result = calculator.compute_delta_s(&current, &history, 5);
println!("AFTER: delta_s = {:?}, must not be NaN/Inf", result);
assert!(!result.unwrap().is_nan());
assert!(!result.unwrap().is_infinite());
```

### Evidence of Success

After running tests, output MUST show:
```
[PASS] test_transe_empty_history_returns_one - delta_s = 1.0
[PASS] test_transe_empty_input_error - Err(EmptyInput)
[PASS] test_transe_identical_returns_low - delta_s = X.XX (X.XX < 0.5)
[PASS] test_transe_perfect_translation - delta_s = X.XX (near 0)
[PASS] test_transe_embedder_type - Embedder::Entity
[PASS] test_transe_valid_range - All outputs in [0.0, 1.0]
[PASS] test_transe_no_nan_infinity - AP-10 compliant
...
```

---

## Reference Implementation Pattern

Use `cross_modal.rs` as the template. Key patterns to follow:

1. **Constants at top:**
```rust
const E11_DIM: usize = 384;
const DEFAULT_SPLIT_POINT: usize = 192;
const DEFAULT_NORM: u8 = 2;
const DEFAULT_K_NEIGHBORS: usize = 5;
const MIN_STD_DEV: f32 = 0.1;
```

2. **Input validation in compute_delta_s:**
```rust
if current.is_empty() {
    return Err(UtlError::EmptyInput);
}
for &v in current {
    if v.is_nan() || v.is_infinite() {
        return Err(UtlError::EntropyError(
            "Invalid value (NaN/Infinity) in current embedding".to_string(),
        ));
    }
}
if history.is_empty() {
    return Ok(1.0);
}
```

3. **Final validation per AP-10:**
```rust
let clamped = delta_s.clamp(0.0, 1.0);
if clamped.is_nan() || clamped.is_infinite() {
    return Err(UtlError::EntropyError(
        "Computed delta_s is NaN or Infinity - violates AP-10".to_string(),
    ));
}
Ok(clamped)
```

---

## TransE Math Reference

From `crates/context-graph-embeddings/src/models/pretrained/entity/transe.rs`:

```rust
// L2 distance: ||h + r - t||_2
let sum_sq: f32 = head.iter()
    .zip(relation.iter())
    .zip(tail.iter())
    .map(|((h, r), t)| {
        let diff = h + r - t;
        diff * diff
    })
    .sum();
let distance = sum_sq.sqrt();

// L1 distance: ||h + r - t||_1
let distance: f32 = head.iter()
    .zip(relation.iter())
    .zip(tail.iter())
    .map(|((h, r), t)| (h + r - t).abs())
    .sum();
```

---

## Anti-Patterns to Avoid

1. **NO** mock data in tests - use real synthetic vectors with known expected outputs
2. **NO** fallback to DefaultKnnEntropy if TransE fails - error out clearly
3. **NO** backwards compatibility shims - old code using E11 must use TransE or fail
4. **NO** silent failures - every error must have clear error message
5. **NO** NaN/Infinity outputs (violates AP-10 in constitution.yaml)

---

## Dependencies

**Must Complete Before:**
- None (can start immediately)

**Blocks:**
- TASK-UTL-P1-006 (MaxSimTokenEntropy for E12) - needs factory pattern reference
- TASK-DELTA-P1-002 (DeltaScComputer) - needs all embedder calculators

---

## Related Files for Context

| File | Why Read |
|------|----------|
| `specs/functional/SPEC-UTL-003.md` | Full specification |
| `docs2/constitution.yaml` lines 165, 364 | Constitution requirements |
| `crates/context-graph-embeddings/src/models/pretrained/entity/transe.rs` | TransE math reference |
| `crates/context-graph-utl/src/surprise/embedder_entropy/cross_modal.rs` | Implementation pattern |

---

## Rollback Plan

If implementation causes issues:
1. Revert factory.rs line 87-89 to route Entity to DefaultKnnEntropy
2. Remove `mod transe;` and `pub use transe::TransEEntropy;` from mod.rs
3. Delete transe.rs file
4. Revert SurpriseConfig changes

---

## Completion Checklist

- [x] Create `transe.rs` with TransEEntropy struct
- [x] Implement EmbedderEntropy trait
- [x] Add to mod.rs exports
- [x] Update factory.rs routing (Entity -> TransEEntropy)
- [x] Add config fields to SurpriseConfig
- [x] All 16+ tests written and passing (25 tests total)
- [x] `cargo build -p context-graph-utl` passes
- [x] `cargo test -p context-graph-utl transe` all pass (25/25)
- [x] `cargo test -p context-graph-utl factory` all pass (13/13)
- [x] `cargo clippy -p context-graph-utl -- -D warnings` no warnings
- [x] Manual edge case verification completed with printed output
- [x] Source of truth verification: factory returns TransEEntropy for Entity

## Completion Notes (2026-01-12)

### Test Results
- **TransE tests:** 25/25 passed
- **Factory tests:** 13/13 passed (including `test_factory_routes_entity_to_transe`)
- **Full embedder_entropy module:** 127/127 passed

### Key Implementation Details
- Created `transe.rs` with full TransEEntropy implementation (~660 lines)
- TransE distance formula: `||h + r - t||` (L2 norm by default)
- E11 dimension: 384D with split at 192 (first half = head, second half = relation)
- Config fields: `entity_transe_norm`, `entity_split_ratio`, `entity_k_neighbors`

### Edge Cases Verified
1. Empty history → returns 1.0 (maximum surprise)
2. Perfect TransE translation (h + r = t) → low surprise
3. Imperfect translation → high surprise
4. Empty input → Err(EmptyInput)
5. NaN/Infinity input → Err(EntropyError)
6. Output range always [0.0, 1.0] (AP-10 compliant)
