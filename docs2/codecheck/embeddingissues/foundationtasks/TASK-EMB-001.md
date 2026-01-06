# TASK-EMB-001: Fix SPARSE_PROJECTED_DIMENSION Constant

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-001 |
| **Title** | Fix SPARSE_PROJECTED_DIMENSION Constant Mismatch |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 1 |
| **Depends On** | None (first task) |
| **Estimated Complexity** | low |

---

## Problem Statement

**There is a dimension constant mismatch in the sparse embedding module.**

The `ModelId::Sparse.projected_dimension()` returns `1536` (correct per Constitution E6_Sparse spec).
However, `SPARSE_PROJECTED_DIMENSION` in `models/pretrained/sparse/types.rs` is set to `768` (incorrect).

This causes:
1. `SparseVector::to_dense_projected()` to output 768D vectors instead of 1536D
2. Tests that validate 768D output (they pass because they match the broken constant)
3. Potential dimension mismatches when sparse embeddings are used with the rest of the pipeline

---

## Current State (Verified 2026-01-06)

### Files with CORRECT values (DO NOT MODIFY):

| File | Constant | Current Value | Status |
|------|----------|---------------|--------|
| `crates/context-graph-embeddings/src/types/dimensions/constants.rs` | `SPARSE` | 1536 | ✅ Correct |
| `crates/context-graph-embeddings/src/types/dimensions/constants.rs` | `SPLADE` | 1536 | ✅ Correct |
| `crates/context-graph-embeddings/src/types/model_id/core.rs` | `ModelId::Sparse.projected_dimension()` | 1536 | ✅ Correct |
| `crates/context-graph-embeddings/src/types/dimensions/aggregates.rs` | `TOTAL_DIMENSION` | 9856 | ✅ Correct |

### Files with INCORRECT values (MUST FIX):

| File | Constant | Current Value | Required Value |
|------|----------|---------------|----------------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` | `SPARSE_PROJECTED_DIMENSION` | 768 | 1536 |

### Tests that MUST be updated (match broken constant):

| File | Test | Current Check | Required Check |
|------|------|---------------|----------------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/tests.rs` | `test_embed_returns_768d_vector` | 768 | 1536 |
| `crates/context-graph-embeddings/src/models/pretrained/sparse/tests.rs` | `test_sparse_vector_to_dense` | 768 | 1536 |

---

## Exact Changes Required

### 1. Fix `types.rs` (Line 33)

**File:** `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs`

**Current (WRONG):**
```rust
/// Projected dimension for multi-array storage compatibility (768D).
pub const SPARSE_PROJECTED_DIMENSION: usize = 768;
```

**Required (CORRECT):**
```rust
/// Projected dimension for multi-array storage compatibility.
/// Per Constitution E6_Sparse: "~30K 5%active" projects to 1536D.
/// Must match dimensions::constants::SPARSE and ModelId::Sparse.projected_dimension().
pub const SPARSE_PROJECTED_DIMENSION: usize = 1536;
```

### 2. Add Compile-Time Assertion

**File:** `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs`

Add after the constant definition:
```rust
/// Compile-time assertion: SPARSE_PROJECTED_DIMENSION must match the canonical value.
const _: () = assert!(
    SPARSE_PROJECTED_DIMENSION == 1536,
    "SPARSE_PROJECTED_DIMENSION must be 1536 per Constitution E6_Sparse"
);
```

### 3. Update `to_dense_projected` Comment

**File:** `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs`

**Current (WRONG):**
```rust
    /// Convert sparse vector to dense format (for multi-array storage).
    ///
    /// Returns a 768D projected representation using the top-k terms.
    pub fn to_dense_projected(&self, projected_dim: usize) -> Vec<f32> {
```

**Required (CORRECT):**
```rust
    /// Convert sparse vector to dense format (for multi-array storage).
    ///
    /// Returns a 1536D projected representation using hash-based projection.
    /// The projected_dim parameter should be SPARSE_PROJECTED_DIMENSION (1536).
    pub fn to_dense_projected(&self, projected_dim: usize) -> Vec<f32> {
```

### 4. Update Test: `test_embed_returns_768d_vector`

**File:** `crates/context-graph-embeddings/src/models/pretrained/sparse/tests.rs`

**Current (WRONG):**
```rust
    #[tokio::test]
    async fn test_embed_returns_768d_vector() {
        let model = create_and_load_model().await;
        let input = ModelInput::text("Test sparse embedding").expect("Input");

        let embedding = model.embed(&input).await.expect("Embed should succeed");

        assert_eq!(embedding.vector.len(), SPARSE_HIDDEN_SIZE);
        assert_eq!(embedding.vector.len(), 768);
    }
```

**Required (CORRECT):**
```rust
    #[tokio::test]
    async fn test_embed_returns_1536d_vector() {
        let model = create_and_load_model().await;
        let input = ModelInput::text("Test sparse embedding").expect("Input");

        let embedding = model.embed(&input).await.expect("Embed should succeed");

        // SPLADE outputs projected 1536D vectors per Constitution E6_Sparse
        assert_eq!(embedding.vector.len(), SPARSE_PROJECTED_DIMENSION);
        assert_eq!(embedding.vector.len(), 1536);
    }
```

### 5. Update Test: `test_sparse_vector_to_dense`

**File:** `crates/context-graph-embeddings/src/models/pretrained/sparse/tests.rs`

**Current (WRONG):**
```rust
    #[test]
    fn test_sparse_vector_to_dense() {
        let sparse = SparseVector::new(vec![0, 100, 500], vec![1.0, 0.5, 0.8]);
        let dense = sparse.to_dense_projected(768);

        assert_eq!(dense.len(), 768);
        // ... normalization check
    }
```

**Required (CORRECT):**
```rust
    #[test]
    fn test_sparse_vector_to_dense() {
        use super::SPARSE_PROJECTED_DIMENSION;

        let sparse = SparseVector::new(vec![0, 100, 500], vec![1.0, 0.5, 0.8]);
        let dense = sparse.to_dense_projected(SPARSE_PROJECTED_DIMENSION);

        assert_eq!(dense.len(), 1536);
        assert_eq!(dense.len(), SPARSE_PROJECTED_DIMENSION);

        // Should be L2 normalized
        let norm: f32 = dense.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.001,
            "Dense projection should be L2 normalized, got norm={}",
            norm
        );
    }
```

### 6. Add Test Import

**File:** `crates/context-graph-embeddings/src/models/pretrained/sparse/tests.rs`

Add `SPARSE_PROJECTED_DIMENSION` to the imports:
```rust
    use crate::models::pretrained::sparse::{
        SparseModel, SparseVector, SPARSE_HIDDEN_SIZE, SPARSE_LATENCY_BUDGET_MS,
        SPARSE_MAX_TOKENS, SPARSE_MODEL_NAME, SPARSE_PROJECTED_DIMENSION, SPARSE_VOCAB_SIZE,
    };
```

---

## Verification Commands

Execute these in order. ALL must pass:

```bash
# 1. Compile check (compile-time assertion catches mismatches)
cargo check -p context-graph-embeddings

# 2. Run dimension tests
cargo test -p context-graph-embeddings test_projected_dimensions

# 3. Run sparse module tests
cargo test -p context-graph-embeddings sparse::tests

# 4. Verify no 768 references in sparse module (should return empty)
grep -rn "768" crates/context-graph-embeddings/src/models/pretrained/sparse/ | grep -v HIDDEN_SIZE

# 5. Verify constant value
grep -n "SPARSE_PROJECTED_DIMENSION" crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs
```

---

## Full State Verification Protocol

After completing the changes, you MUST verify the system state:

### Source of Truth Locations

| Data | Location | Expected Value |
|------|----------|----------------|
| Projected dimension constant | `sparse/types.rs:SPARSE_PROJECTED_DIMENSION` | 1536 |
| Dimension helper | `dimensions::projected_dimension_by_index(5)` | 1536 |
| ModelId method | `ModelId::Sparse.projected_dimension()` | 1536 |
| Total dimension | `dimensions::TOTAL_DIMENSION` | 9856 |

### Verification Steps

1. **Read the constant directly:**
```bash
grep "SPARSE_PROJECTED_DIMENSION.*=" crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs
# Expected output: pub const SPARSE_PROJECTED_DIMENSION: usize = 1536;
```

2. **Run compilation (compile-time assertion fires if wrong):**
```bash
cargo build -p context-graph-embeddings 2>&1 | grep -i "error\|SPARSE"
# Expected: No errors
```

3. **Execute verification test:**
```bash
cargo test -p context-graph-embeddings test_constants_are_correct -- --nocapture
cargo test -p context-graph-embeddings test_sparse_vector_to_dense -- --nocapture
```

### Edge Cases to Manually Verify

Execute these checks and document the before/after state:

**Edge Case 1: Empty sparse vector**
```rust
// Test: SparseVector with no active indices
let sparse = SparseVector::new(vec![], vec![]);
let dense = sparse.to_dense_projected(1536);
assert_eq!(dense.len(), 1536);
// All zeros, norm = 0 (no normalization applied)
```

**Edge Case 2: Single active index**
```rust
// Test: Single term in sparse vector
let sparse = SparseVector::new(vec![100], vec![1.0]);
let dense = sparse.to_dense_projected(1536);
assert_eq!(dense.len(), 1536);
// Only index 100 % 1536 = 100 has non-zero value
```

**Edge Case 3: Hash collision handling**
```rust
// Test: Multiple indices that hash to same bucket
let sparse = SparseVector::new(vec![0, 1536], vec![0.5, 0.5]);
let dense = sparse.to_dense_projected(1536);
// Index 0 and 1536 both map to dense[0], weights should sum
```

### Evidence of Success

After implementation, provide a log showing:
```
[VERIFICATION LOG]
1. SPARSE_PROJECTED_DIMENSION = 1536 (confirmed in types.rs)
2. Compile-time assertion present and passing
3. cargo check: SUCCESS (no errors)
4. cargo test sparse::tests: ALL PASSED
5. grep for "768" in sparse module: Only SPARSE_HIDDEN_SIZE (expected)
6. Total dimension still 9856 (unchanged)
```

---

## What NOT to Change

| File | Constant | Current Value | Reason |
|------|----------|---------------|--------|
| `types/dimensions/constants.rs` | `SPARSE` | 1536 | Already correct |
| `types/dimensions/constants.rs` | `SPLADE` | 1536 | Already correct |
| `types/model_id/core.rs` | `projected_dimension()` | 1536 | Already correct |
| `sparse/types.rs` | `SPARSE_HIDDEN_SIZE` | 768 | This is BERT hidden size, correct |
| `sparse/types.rs` | `SPARSE_VOCAB_SIZE` | 30522 | This is BERT vocab, correct |

**CRITICAL:** `SPARSE_HIDDEN_SIZE = 768` is CORRECT. It represents the BERT backbone hidden dimension. Do NOT confuse this with `SPARSE_PROJECTED_DIMENSION` which is the output projection.

---

## Why This Matters

The Constitution specifies:
```yaml
E6_Sparse: { dim: "~30K 5%active", math: TopK, hw: SparseTensor, lat: "<3ms", purpose: "V_selectivity" }
```

And the aggregates show:
```rust
// 1024 + 512 + 512 + 512 + 768 + 1536 + 768 + 384 + 1024 + 768 + 384 + 128 + 1536 = 9856
pub const TOTAL_DIMENSION: usize = 9856;
```

The 1536 for E6 (SPARSE) is already factored into `TOTAL_DIMENSION`. The `SPARSE_PROJECTED_DIMENSION` being 768 means actual output doesn't match the specification.

---

## Traceability

| Requirement | Constitution Ref | Code Location |
|-------------|------------------|---------------|
| E6_Sparse dim | `embeddings.models.E6_Sparse` | `ModelId::Sparse.projected_dimension()` |
| Total dimension 9856 | `embeddings.quantization` | `dimensions::TOTAL_DIMENSION` |

---

## Post-Implementation Checklist

- [ ] `SPARSE_PROJECTED_DIMENSION` changed from 768 to 1536
- [ ] Compile-time assertion added
- [ ] Test `test_embed_returns_1536d_vector` updated and passing
- [ ] Test `test_sparse_vector_to_dense` updated and passing
- [ ] `cargo check -p context-graph-embeddings` succeeds
- [ ] `cargo test -p context-graph-embeddings sparse::tests` all pass
- [ ] No grep hits for "768" in sparse module except `SPARSE_HIDDEN_SIZE`
- [ ] Full state verification log provided
- [ ] Edge cases manually verified with before/after state
