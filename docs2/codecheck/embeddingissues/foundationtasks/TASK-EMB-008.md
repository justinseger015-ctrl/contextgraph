# TASK-EMB-008: Update SparseVector Struct

<task_spec id="TASK-EMB-008" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-008 |
| **Title** | Update SparseVector Struct |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 8 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-001 |
| **Estimated Complexity** | low |
| **Constitution Ref** | `E6_Sparse: { dim: "~30K 5%active" }` |

---

## Context

TECH-EMB-001 specifies removing the broken `to_dense_projected()` method from SparseVector and adding `to_csr()` for cuBLAS integration. The broken hash-based projection is replaced by ProjectionMatrix.

**Why This Matters:**
- Current `to_dense_projected()` uses hash projection (`idx % projected_dim`)
- This destroys semantic information (unrelated tokens map to same dimension)
- Must be replaced with learned ProjectionMatrix (TASK-EMB-012)

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Current SparseVector | `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |

---

## Prerequisites

- [ ] TASK-EMB-001 completed (dimension constants fixed)

---

## Scope

### In Scope
- Remove `to_dense_projected()` method (or mark deprecated)
- Add `to_csr()` method for cuBLAS integration
- Update documentation to reference ProjectionMatrix

### Out of Scope
- `to_csr()` complex implementation (Logic Layer)
- ProjectionMatrix integration (Logic Layer - TASK-EMB-012)

---

## Definition of Done

### Exact Changes

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs

/// Sparse vector output with term indices and weights.
///
/// # Constitution Alignment
/// - Dimension: SPARSE_VOCAB_SIZE (30522)
/// - Expected sparsity: ~95% zeros (~5% active)
/// - Output after projection: 1536D dense (via ProjectionMatrix)
///
/// # BREAKING CHANGE
/// `to_dense_projected()` has been REMOVED. Use `ProjectionMatrix::project()` instead.
#[derive(Debug, Clone)]
pub struct SparseVector {
    /// Token indices with non-zero weights (sorted ascending).
    pub indices: Vec<usize>,
    /// Corresponding weights for each index.
    pub weights: Vec<f32>,
    /// Total number of dimensions (vocabulary size = 30522).
    pub dimension: usize,
}

impl SparseVector {
    /// Create a new sparse vector.
    ///
    /// # Invariants
    /// - indices.len() == weights.len()
    /// - All indices < SPARSE_VOCAB_SIZE (30522)
    /// - Indices are sorted ascending
    pub fn new(indices: Vec<usize>, weights: Vec<f32>) -> Self {
        debug_assert_eq!(indices.len(), weights.len());
        Self {
            indices,
            weights,
            dimension: SPARSE_VOCAB_SIZE,
        }
    }

    /// Convert to CSR (Compressed Sparse Row) format for cuBLAS.
    ///
    /// # Returns
    /// (row_ptr, col_indices, values) tuple for CSR representation.
    ///
    /// # Implementation Note
    /// For a single vector (1 row), CSR is:
    /// - row_ptr = [0, nnz]
    /// - col_indices = indices as i32
    /// - values = weights
    pub fn to_csr(&self) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let nnz = self.indices.len() as i32;
        let row_ptr = vec![0i32, nnz];
        let col_indices: Vec<i32> = self.indices.iter().map(|&i| i as i32).collect();
        let values = self.weights.clone();
        (row_ptr, col_indices, values)
    }

    /// Get number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Get sparsity as percentage of zeros.
    pub fn sparsity(&self) -> f32 {
        1.0 - (self.indices.len() as f32 / self.dimension as f32)
    }

    // REMOVED: to_dense_projected()
    // Use ProjectionMatrix::project() instead.
    // The old hash-based projection (idx % projected_dim) destroyed semantic information.
}
```

### Constraints
- `to_dense_projected()` must be deleted or commented out
- Documentation must reference ProjectionMatrix
- `to_csr()` can have simple implementation

### Verification
- `to_dense_projected` method no longer exists
- `to_csr` returns correct format
- Documentation updated

---

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` | Remove old method, add new |

---

## Validation Criteria

- [ ] `to_dense_projected()` removed or marked deprecated
- [ ] `to_csr()` method present
- [ ] Documentation references ProjectionMatrix
- [ ] No compile errors

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
grep -n "to_dense_projected" crates/context-graph-embeddings/
```

---

## Traceability

| Requirement | Tech Spec | Issue |
|-------------|-----------|-------|
| REQ-EMB-001 | TECH-EMB-001 | ISSUE-001 |
| REQ-EMB-002 | TECH-EMB-001 | ISSUE-002 |

---

## Code to Remove

```rust
// This MUST be removed (anti-pattern from ISSUE-001):
pub fn to_dense_projected(&self, projected_dim: usize) -> Vec<f32> {
    let mut dense = vec![0.0; projected_dim];
    for (&idx, &weight) in self.indices.iter().zip(self.weights.iter()) {
        dense[idx % projected_dim] += weight;  // HASH COLLISION BUG
    }
    dense
}
```

---

## Notes

- The CSR format is required for efficient cuBLAS sparse matrix operations
- After this task, callers must use ProjectionMatrix::project() (TASK-EMB-012)
- This is a breaking API change that will require updates to callers

</task_spec>
