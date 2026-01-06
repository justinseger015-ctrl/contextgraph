# TASK-EMB-002: Create ProjectionMatrix Struct

<task_spec id="TASK-EMB-002" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-002 |
| **Title** | Create ProjectionMatrix Struct |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 2 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-001 |
| **Estimated Complexity** | medium |
| **Constitution Ref** | `E6_Sparse: { dim: "~30K 5%active" }`, `AP-007` |

---

## Context

TECH-EMB-001 specifies a learned projection matrix to replace the broken hash-based sparse-to-dense projection. This task creates the **data structure only**. The actual loading and projection logic are Logic Layer tasks.

**Why This Matters:**
- The current hash-based projection (`idx % projected_dim`) destroys semantic information
- Constitution AP-007 forbids stub data in production
- A learned projection matrix preserves semantic similarity

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Dimension constants | `crates/context-graph-embeddings/src/types/dimensions/constants.rs` |
| SparseVector type | `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |

---

## Prerequisites

- [ ] TASK-EMB-001 completed (dimension constants fixed)
- [ ] Candle crate available for Tensor type
- [ ] Device type available from Candle

---

## Scope

### In Scope
- Create `ProjectionMatrix` struct with weights, device, checksum
- Add struct documentation with Constitution references
- Define associated constants for expected shape
- Basic accessor methods (read-only)

### Out of Scope
- `load()` implementation (Logic Layer - TASK-EMB-011)
- `project()` implementation (Logic Layer - TASK-EMB-012)
- CUDA integration (Logic Layer)

---

## Definition of Done

### Exact Signatures

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs

use candle_core::{Device, Tensor};
use super::types::{SPARSE_PROJECTED_DIMENSION, SPARSE_VOCAB_SIZE};

/// Expected weight file path relative to model directory.
pub const PROJECTION_WEIGHT_FILE: &str = "sparse_projection.safetensors";

/// Expected tensor name in SafeTensors file.
pub const PROJECTION_TENSOR_NAME: &str = "projection.weight";

/// Learned projection matrix for sparse-to-dense conversion.
///
/// # Constitution Alignment
/// - E6_Sparse: `dim: "~30K 5%active"` input, 1536D output
/// - E13_Splade: Same architecture, same projection
///
/// # Weight Source
/// - Pre-trained via contrastive learning on MS MARCO
/// - Fine-tuned to preserve semantic similarity
///
/// # CRITICAL: No Fallback
/// If weight file is missing, system MUST panic. Hash fallback is FORBIDDEN.
#[derive(Debug)]
pub struct ProjectionMatrix {
    /// Weight tensor on GPU: [SPARSE_VOCAB_SIZE x SPARSE_PROJECTED_DIMENSION]
    /// Shape: [30522, 1536]
    weights: Tensor,

    /// Device where weights are loaded (must be CUDA)
    device: Device,

    /// SHA256 checksum of the weight file for validation
    weight_checksum: [u8; 32],
}

impl ProjectionMatrix {
    /// Expected weight matrix shape: [vocab_size, projected_dim]
    pub const EXPECTED_SHAPE: (usize, usize) = (SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION);

    /// Expected file size in bytes: vocab_size * proj_dim * sizeof(f32)
    pub const EXPECTED_FILE_SIZE: usize = SPARSE_VOCAB_SIZE * SPARSE_PROJECTED_DIMENSION * 4;

    /// Get the weight tensor reference.
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the weight checksum.
    pub fn checksum(&self) -> &[u8; 32] {
        &self.weight_checksum
    }

    /// Check if weights are on CUDA device.
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }
}
```

### Constraints
- Struct only, no method implementations beyond accessors
- Must use Candle types
- Must reference dimension constants from TASK-EMB-001

### Verification
- `cargo check` succeeds
- Struct size is reasonable (Tensor is opaque)

---

## Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | ProjectionMatrix struct and constants |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs` | Add `pub mod projection;` |

---

## Validation Criteria

- [ ] `ProjectionMatrix` struct compiles
- [ ] `EXPECTED_SHAPE` constant equals (30522, 1536)
- [ ] Documentation references Constitution
- [ ] No load/project implementations yet

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo doc -p context-graph-embeddings --no-deps
```

---

## Traceability

| Requirement | Tech Spec | Issue |
|-------------|-----------|-------|
| REQ-EMB-001 | TECH-EMB-001 | ISSUE-001 |

---

## Notes

- This struct will be populated by TASK-EMB-011 (load) and used by TASK-EMB-012 (project)
- The checksum field ensures weight file integrity
- CUDA device requirement is documented but not enforced until Logic Layer

</task_spec>
