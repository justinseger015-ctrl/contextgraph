# TASK-EMB-021: Integrate ProjectionMatrix into SparseModel

<task_spec id="TASK-EMB-021" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-021 |
| **Title** | Integrate Learned Projection into SparseModel |
| **Status** | ready |
| **Layer** | surface |
| **Sequence** | 21 |
| **Implements** | REQ-EMB-001 (Learned Sparse Projection Matrix) |
| **Depends On** | TASK-EMB-012 (ProjectionMatrix::project()), TASK-EMB-008 (SparseVector updated) |
| **Estimated Complexity** | medium |
| **Created** | 2026-01-06 |
| **Constitution Reference** | v4.0.0 |

---

## Context

TECH-EMB-001 specifies that the SparseModel must use the learned ProjectionMatrix instead of the broken hash-based projection (`idx % projected_dim`). This task integrates the Foundation and Logic Layer work into the actual SparseModel API, replacing the hash-based sparse-to-dense conversion with proper neural projection.

**Why this matters:**
- The hash-based projection destroys semantic meaning
- Real projection uses learned weights that preserve similarity
- Output dimension changes from 768 to 1536 per Constitution

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Current SparseModel | `crates/context-graph-embeddings/src/models/pretrained/sparse/model.rs` |
| ProjectionMatrix | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` |
| SparseVector types | `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |

---

## Prerequisites

- [ ] TASK-EMB-012 completed (ProjectionMatrix::project() works)
- [ ] TASK-EMB-008 completed (SparseVector updated with proper dimension support)
- [ ] `projection.rs` module exists with working `ProjectionMatrix::load()` and `ProjectionMatrix::project()`
- [ ] Dimension constant `SPARSE_PROJECTED_DIMENSION = 1536` defined

---

## Scope

### In Scope

- Add `projection: Option<ProjectionMatrix>` field to SparseModel struct
- Load projection matrix during `SparseModel::load()`
- Replace `sparse.to_dense_projected()` with `projection.project(&sparse)` in `embed()`
- Update output dimension from 768 to 1536
- Add `get_projection()` helper method
- Log projection matrix load with checksum

### Out of Scope

- Storage integration (TASK-EMB-022)
- MCP handler updates (TASK-EMB-024)
- Quantization of output vectors (separate task)
- Multi-space search (TASK-EMB-023)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/model.rs

use super::projection::ProjectionMatrix;
use super::types::SPARSE_PROJECTED_DIMENSION;

pub struct SparseModel {
    // ... existing fields ...

    /// Learned projection matrix for sparse-to-dense conversion.
    /// CRITICAL: This replaces the broken hash-based projection.
    pub(crate) projection: Option<ProjectionMatrix>,
}

impl SparseModel {
    /// Load model weights into memory.
    ///
    /// # CRITICAL CHANGE
    /// Now also loads the projection matrix. System will PANIC if
    /// projection weights are missing (no fallback to hash).
    pub async fn load(&self) -> EmbeddingResult<()>;

    /// Embed input to dense vector (for multi-array storage compatibility).
    ///
    /// # CRITICAL CHANGE
    /// - Output dimension is now 1536 (was 768)
    /// - Uses learned neural projection (was hash modulo)
    pub async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding>;

    /// Get projection matrix, error if not loaded.
    fn get_projection(&self) -> EmbeddingResult<&ProjectionMatrix>;
}
```

### Implementation Pattern

```rust
impl SparseModel {
    pub async fn load(&self) -> EmbeddingResult<()> {
        // ... existing BERT and MLM loading ...

        // Load projection matrix (REQUIRED - no fallback)
        let projection = ProjectionMatrix::load(&self.model_path)?;
        tracing::info!(
            "ProjectionMatrix loaded: shape [{}, {}], checksum {:?}",
            super::types::SPARSE_VOCAB_SIZE,
            SPARSE_PROJECTED_DIMENSION,
            hex::encode(&projection.checksum()[..8])
        );

        // Store projection in model state
        // (Update internal state struct to hold projection)

        Ok(())
    }

    pub async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
        // ... validation ...

        let start = std::time::Instant::now();

        // Get sparse vector (existing)
        let sparse = self.embed_sparse(input).await?;

        // Project using learned weights (NOT hash modulo!)
        let projection = self.get_projection()?;
        let vector = projection.project(&sparse)?;

        // Verify output dimension
        debug_assert_eq!(
            vector.len(),
            SPARSE_PROJECTED_DIMENSION,
            "Output dimension mismatch: expected {}, got {}",
            SPARSE_PROJECTED_DIMENSION,
            vector.len()
        );

        let latency_us = start.elapsed().as_micros() as u64;
        Ok(ModelEmbedding::new(self.model_id, vector, latency_us))
    }

    fn get_projection(&self) -> EmbeddingResult<&ProjectionMatrix> {
        self.projection.as_ref().ok_or_else(|| {
            EmbeddingError::NotInitialized {
                model_id: self.model_id,
            }
        })
    }
}
```

### Constraints

- Output dimension MUST be 1536 (not 768)
- Projection matrix MUST be loaded during load()
- NO fallback to hash-based projection
- PANIC if projection weights file missing
- Must log checksum on load for verification
- Latency budget: projection step < 3ms

### Verification

- `cargo check -p context-graph-embeddings --features cuda` passes
- `cargo test -p context-graph-embeddings sparse::model` passes
- Output dimension is 1536
- No reference to `to_dense_projected()` in embed path
- System panics if projection file missing

---

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/model.rs` | Add projection field, update `load()` and `embed()` |
| `crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs` | Re-export projection module |

---

## Validation Criteria

- [ ] `SparseModel` struct has `projection: Option<ProjectionMatrix>` field
- [ ] `load()` loads projection matrix and logs checksum
- [ ] `embed()` returns 1536-dimensional vector
- [ ] No reference to `to_dense_projected()` or `idx % dim` in embed path
- [ ] System panics if projection file missing (no silent fallback)
- [ ] Semantic similarity test: related terms have similarity > 0.7
- [ ] Latency: single embed < 10ms p95

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings --features cuda
cargo test -p context-graph-embeddings sparse::model -- --nocapture

# Verify no hash-based projection
grep -rn "to_dense_projected\|idx % " crates/context-graph-embeddings/src/models/pretrained/sparse/model.rs

# Verify dimension constant
grep -rn "SPARSE_PROJECTED_DIMENSION" crates/context-graph-embeddings/src/
```

---

## Anti-Patterns to Avoid

| Pattern | Why Forbidden | Constitution Ref |
|---------|---------------|------------------|
| `idx % projected_dim` | Hash destroys semantics | AP-007 |
| Fallback to hash on missing file | Silent degradation | AP-007 |
| Output dim != 1536 | Misaligned with Constitution | `E6_Sparse: { dim: "~30K 5%active" }` |
| Skip checksum logging | Can't verify correct weights | SEC-03 |

---

## Memory Key

Store completion status:
```
contextgraph/embedding-issues/task-emb-021-complete
```

</task_spec>
