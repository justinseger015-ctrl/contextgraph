# TASK-EMB-003: Create ProjectionError Enum

<task_spec id="TASK-EMB-003" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-003 |
| **Title** | Create ProjectionError Enum |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 3 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-002 |
| **Estimated Complexity** | low |
| **Constitution Ref** | `AP-007`, Error Taxonomy from SPEC-EMB-001 |

---

## Context

The projection system needs specific error types for clear error messages. These errors implement the Constitution's "fail fast" requirement with specific remediation steps in error messages.

**Why This Matters:**
- Clear error messages help users understand what went wrong
- Remediation steps guide users to fix issues
- Error codes enable programmatic error handling

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Projection module | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` |
| Tech spec errors | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |
| Functional spec errors | `docs2/codecheck/embeddingissues/SPEC-EMB-001-master-functional.md` |

---

## Prerequisites

- [ ] TASK-EMB-002 completed (projection module exists)
- [ ] thiserror crate available

---

## Scope

### In Scope
- Create `ProjectionError` enum with 5 variants
- Each variant has descriptive error message with error code
- Include remediation steps in error messages

### Out of Scope
- Error handling logic (Logic Layer)
- Integration with other error types (TASK-EMB-007)

---

## Definition of Done

### Exact Signatures

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
// (Add to existing file from TASK-EMB-002)

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during sparse projection.
#[derive(Debug, Error)]
pub enum ProjectionError {
    /// Projection weights file not found.
    ///
    /// # Remediation
    /// Download from: https://huggingface.co/contextgraph/sparse-projection
    #[error("[EMB-E006] PROJECTION_MATRIX_MISSING: Weight file not found at {path}
  Expected: models/sparse_projection.safetensors
  Remediation: Download projection weights or train custom matrix")]
    MatrixMissing { path: PathBuf },

    /// Weight file checksum does not match expected value.
    #[error("[EMB-E004] WEIGHT_CHECKSUM_MISMATCH: Corrupted weight file
  Expected: {expected}
  Actual: {actual}
  File: {path}
  Remediation: Re-download weight file")]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },

    /// Weight matrix has wrong shape.
    #[error("[EMB-E005] DIMENSION_MISMATCH: Projection matrix has wrong shape
  Expected: [30522, 1536]
  Actual: [{actual_rows}, {actual_cols}]
  File: {path}")]
    DimensionMismatch {
        path: PathBuf,
        actual_rows: usize,
        actual_cols: usize,
    },

    /// CUDA operation failed.
    #[error("[EMB-E001] CUDA_ERROR: GPU operation failed
  Operation: {operation}
  Details: {details}
  Remediation: Check GPU availability and driver version")]
    GpuError { operation: String, details: String },

    /// Projection not initialized (weights not loaded).
    #[error("[EMB-E008] NOT_INITIALIZED: Projection weights not loaded
  Remediation: Call ProjectionMatrix::load() before projection")]
    NotInitialized,
}
```

### Constraints
- Use thiserror for derive
- Error codes must match SPEC-EMB-001 error taxonomy
- Each error includes remediation steps

### Verification
- All error variants compile
- Error messages are formatted correctly

---

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | Add ProjectionError enum |

---

## Validation Criteria

- [ ] 5 error variants defined
- [ ] All error codes match SPEC-EMB-001 (EMB-E001, EMB-E004, EMB-E005, EMB-E006, EMB-E008)
- [ ] Error messages include remediation steps
- [ ] thiserror::Error derived

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings projection -- --nocapture 2>&1 | head -20
```

---

## Traceability

| Requirement | Tech Spec | Issue |
|-------------|-----------|-------|
| REQ-EMB-001 | TECH-EMB-001 | ISSUE-001 |

---

## Error Code Reference

| Code | Name | Description |
|------|------|-------------|
| EMB-E001 | CUDA_ERROR | GPU operation failed |
| EMB-E004 | WEIGHT_CHECKSUM_MISMATCH | Corrupted weight file |
| EMB-E005 | DIMENSION_MISMATCH | Wrong matrix shape |
| EMB-E006 | PROJECTION_MATRIX_MISSING | Weight file not found |
| EMB-E008 | NOT_INITIALIZED | Weights not loaded |

</task_spec>
