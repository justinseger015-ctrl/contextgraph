# TASK-EMB-007: Create Consolidated Error Types

<task_spec id="TASK-EMB-007" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-007 |
| **Title** | Create Consolidated Error Enum |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 7 |
| **Implements** | All REQs |
| **Depends On** | TASK-EMB-003, TASK-EMB-005, TASK-EMB-006 |
| **Estimated Complexity** | medium |
| **Constitution Ref** | Error Taxonomy from SPEC-EMB-001 |

---

## Context

The embedding system needs a consolidated error type that encompasses all error scenarios from projection, warm loading, quantization, and storage. This provides a single error type for the public API.

**Why This Matters:**
- Unified error handling for all embedding operations
- Clear error codes for programmatic handling
- Remediation steps guide users to fix issues

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Functional spec errors | `docs2/codecheck/embeddingissues/SPEC-EMB-001-master-functional.md` |
| ProjectionError | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` |

---

## Prerequisites

- [ ] TASK-EMB-003 completed (ProjectionError exists)
- [ ] TASK-EMB-005 completed (Storage types exist)
- [ ] TASK-EMB-006 completed (Warm loading types exist)

---

## Scope

### In Scope
- Create `EmbeddingError` enum covering all error codes from SPEC-EMB-001
- Implement From conversions from specific error types
- Include all 12 error codes from error taxonomy

### Out of Scope
- Error handling logic (Logic Layer)
- Recovery strategies (Logic Layer)

---

## Definition of Done

### Exact Signatures

```rust
// File: crates/context-graph-embeddings/src/error.rs

use std::path::PathBuf;
use thiserror::Error;
use crate::types::model_id::ModelId;

/// Consolidated errors for the embedding system.
///
/// Error codes from SPEC-EMB-001 Error Taxonomy:
/// - EMB-E001: CUDA_UNAVAILABLE
/// - EMB-E002: INSUFFICIENT_VRAM
/// - EMB-E003: WEIGHT_FILE_MISSING
/// - EMB-E004: WEIGHT_CHECKSUM_MISMATCH
/// - EMB-E005: DIMENSION_MISMATCH
/// - EMB-E006: PROJECTION_MATRIX_MISSING
/// - EMB-E007: OOM_DURING_BATCH
/// - EMB-E008: INFERENCE_VALIDATION_FAILED
/// - EMB-E009: INPUT_TOO_LARGE
/// - EMB-E010: STORAGE_CORRUPTION
/// - EMB-E011: CODEBOOK_MISSING
/// - EMB-E012: RECALL_LOSS_EXCEEDED
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// EMB-E001: CUDA is required but unavailable.
    #[error("[EMB-E001] CUDA_UNAVAILABLE: {message}
  Required: RTX 5090 (Blackwell, CC 12.0)
  Remediation: Install CUDA 13.1+ and verify GPU")]
    CudaUnavailable { message: String },

    /// EMB-E002: Insufficient GPU VRAM.
    #[error("[EMB-E002] INSUFFICIENT_VRAM: Not enough GPU memory
  Required: {required_bytes} bytes ({required_gb:.1} GB)
  Available: {available_bytes} bytes ({available_gb:.1} GB)
  Remediation: Free GPU memory or upgrade hardware")]
    InsufficientVram {
        required_bytes: usize,
        available_bytes: usize,
        required_gb: f64,
        available_gb: f64,
    },

    /// EMB-E003: Weight file not found.
    #[error("[EMB-E003] WEIGHT_FILE_MISSING: Model weight file not found
  Model: {model_id:?}
  Path: {path}
  Remediation: Download weights from model repository")]
    WeightFileMissing { model_id: ModelId, path: PathBuf },

    /// EMB-E004: Weight file checksum mismatch.
    #[error("[EMB-E004] WEIGHT_CHECKSUM_MISMATCH: Corrupted weight file
  Model: {model_id:?}
  Expected: {expected}
  Actual: {actual}
  Remediation: Re-download weight file")]
    WeightChecksumMismatch {
        model_id: ModelId,
        expected: String,
        actual: String,
    },

    /// EMB-E005: Dimension mismatch.
    #[error("[EMB-E005] DIMENSION_MISMATCH: Embedding dimension mismatch
  Model: {model_id:?}
  Expected: {expected}
  Actual: {actual}
  Remediation: Verify model configuration")]
    DimensionMismatch {
        model_id: ModelId,
        expected: usize,
        actual: usize,
    },

    /// EMB-E006: Projection matrix missing.
    #[error("[EMB-E006] PROJECTION_MATRIX_MISSING: Sparse projection weights not found
  Path: {path}
  Remediation: Download from https://huggingface.co/contextgraph/sparse-projection")]
    ProjectionMatrixMissing { path: PathBuf },

    /// EMB-E007: Out of memory during batch processing.
    #[error("[EMB-E007] OOM_DURING_BATCH: GPU out of memory during batch
  Batch size: {batch_size}
  Remediation: Reduce batch size or free GPU memory")]
    OomDuringBatch { batch_size: usize },

    /// EMB-E008: Inference validation failed.
    #[error("[EMB-E008] INFERENCE_VALIDATION_FAILED: Model inference output invalid
  Model: {model_id:?}
  Reason: {reason}
  Remediation: Check model weights and configuration")]
    InferenceValidationFailed { model_id: ModelId, reason: String },

    /// EMB-E009: Input too large.
    #[error("[EMB-E009] INPUT_TOO_LARGE: Input exceeds maximum size
  Max tokens: {max_tokens}
  Actual tokens: {actual_tokens}
  Remediation: Truncate input or split into chunks")]
    InputTooLarge { max_tokens: usize, actual_tokens: usize },

    /// EMB-E010: Storage corruption detected.
    #[error("[EMB-E010] STORAGE_CORRUPTION: Stored data is corrupted
  ID: {id}
  Reason: {reason}
  Remediation: Re-index from source")]
    StorageCorruption { id: String, reason: String },

    /// EMB-E011: Quantization codebook missing.
    #[error("[EMB-E011] CODEBOOK_MISSING: PQ-8 codebook not found
  Model: {model_id:?}
  Remediation: Train codebook or download from model repository")]
    CodebookMissing { model_id: ModelId },

    /// EMB-E012: Recall loss exceeded.
    #[error("[EMB-E012] RECALL_LOSS_EXCEEDED: Quantization quality too low
  Model: {model_id:?}
  Measured: {measured:.4}
  Max allowed: {max_allowed:.4}")]
    RecallLossExceeded {
        model_id: ModelId,
        measured: f32,
        max_allowed: f32,
    },
}

impl EmbeddingError {
    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(self, Self::InputTooLarge { .. })
    }

    /// Get error code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::CudaUnavailable { .. } => "EMB-E001",
            Self::InsufficientVram { .. } => "EMB-E002",
            Self::WeightFileMissing { .. } => "EMB-E003",
            Self::WeightChecksumMismatch { .. } => "EMB-E004",
            Self::DimensionMismatch { .. } => "EMB-E005",
            Self::ProjectionMatrixMissing { .. } => "EMB-E006",
            Self::OomDuringBatch { .. } => "EMB-E007",
            Self::InferenceValidationFailed { .. } => "EMB-E008",
            Self::InputTooLarge { .. } => "EMB-E009",
            Self::StorageCorruption { .. } => "EMB-E010",
            Self::CodebookMissing { .. } => "EMB-E011",
            Self::RecallLossExceeded { .. } => "EMB-E012",
        }
    }
}
```

### Constraints
- All error codes from SPEC-EMB-001 must be present
- Each error includes remediation steps
- `is_recoverable()` returns true only for EMB-E009

### Verification
- All 12 error variants defined
- Error codes match specification

---

## Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/error.rs` | Consolidated error enum |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/lib.rs` | Add `pub mod error;` and re-export |

---

## Validation Criteria

- [ ] All 12 error codes from SPEC-EMB-001 present
- [ ] `is_recoverable()` only returns true for InputTooLarge
- [ ] `code()` returns correct error code string
- [ ] All errors include remediation steps

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings error -- --nocapture
```

---

## Traceability

| Requirement | Tech Spec | Issue |
|-------------|-----------|-------|
| All | All | All |

---

## Error Code Reference

| Code | Name | Recoverable | Severity |
|------|------|-------------|----------|
| EMB-E001 | CUDA_UNAVAILABLE | No | Critical |
| EMB-E002 | INSUFFICIENT_VRAM | No | Critical |
| EMB-E003 | WEIGHT_FILE_MISSING | No | Critical |
| EMB-E004 | WEIGHT_CHECKSUM_MISMATCH | No | Critical |
| EMB-E005 | DIMENSION_MISMATCH | No | Critical |
| EMB-E006 | PROJECTION_MATRIX_MISSING | No | Critical |
| EMB-E007 | OOM_DURING_BATCH | No | High |
| EMB-E008 | INFERENCE_VALIDATION_FAILED | No | Critical |
| EMB-E009 | INPUT_TOO_LARGE | Yes | Medium |
| EMB-E010 | STORAGE_CORRUPTION | No | High |
| EMB-E011 | CODEBOOK_MISSING | No | High |
| EMB-E012 | RECALL_LOSS_EXCEEDED | No | Medium |

</task_spec>
