# TASK-EMB-011: Implement ProjectionMatrix::load()

<task_spec id="TASK-EMB-011" version="2.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Implement ProjectionMatrix Load Method |
| **Status** | **COMPLETE** |
| **Layer** | logic |
| **Sequence** | 11 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-002 (COMPLETE), TASK-EMB-003 (COMPLETE) |
| **Estimated Complexity** | high |
| **Updated** | 2026-01-06 |
| **Codebase Audit** | VERIFIED |

---

## CRITICAL: Codebase Audit Summary

**This section documents the ACTUAL current state of the codebase as of 2026-01-06.**

### What EXISTS (Foundation Layer COMPLETE)

| Component | File Path | Status |
|-----------|-----------|--------|
| ProjectionMatrix struct | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | EXISTS |
| ProjectionError enum (5 variants) | Same file | EXISTS |
| PROJECTION_WEIGHT_FILE constant | Same file | EXISTS (`"sparse_projection.safetensors"`) |
| PROJECTION_TENSOR_NAME constant | Same file | EXISTS (`"projection.weight"`) |
| EXPECTED_SHAPE constant | Same file | EXISTS (`[30522, 1536]`) |
| SparseVector::to_csr() | `sparse/types.rs` | EXISTS |
| SPARSE_VOCAB_SIZE | `sparse/types.rs` | EXISTS (30522) |
| SPARSE_PROJECTED_DIMENSION | `sparse/types.rs` | EXISTS (1536) |
| WarmError (20+ variants) | `warm/error.rs` | EXISTS |

### What Does NOT Exist (This Task Implements)

| Component | Required Action |
|-----------|-----------------|
| `ProjectionMatrix::load()` method | IMPLEMENT |
| `ProjectionMatrix::project()` method | TASK-EMB-012 (not this task) |
| `safetensors` crate dependency | ADD to Cargo.toml |
| `sha2` crate dependency | ADD to Cargo.toml |

### Current ProjectionMatrix Struct (VERIFIED)

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
// Lines 45-56

#[derive(Debug)]
pub struct ProjectionMatrix {
    /// Weight tensor on GPU: [SPARSE_VOCAB_SIZE x SPARSE_PROJECTED_DIMENSION]
    /// Shape: [30522, 1536]
    weights: Tensor,

    /// Device where weights are loaded (must be CUDA for production)
    device: Device,

    /// SHA256 checksum of the weight file for integrity validation
    weight_checksum: [u8; 32],
}
```

### Current ProjectionError Enum (VERIFIED)

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
// Lines 144-190

pub enum ProjectionError {
    MatrixMissing { path: PathBuf },                                    // EMB-E006
    ChecksumMismatch { path: PathBuf, expected: String, actual: String }, // EMB-E004
    DimensionMismatch { path: PathBuf, actual_rows: usize, actual_cols: usize }, // EMB-E005
    GpuError { operation: String, details: String },                    // EMB-E001
    NotInitialized,                                                     // EMB-E008
}
```

### Current Cargo.toml Dependencies (MISSING)

```toml
# File: crates/context-graph-embeddings/Cargo.toml
# MISSING - MUST ADD:
# safetensors = "0.4"
# sha2 = "0.10"
```

---

## Context

TECH-EMB-001 specifies loading a learned projection matrix from SafeTensors format. This replaces the broken hash-based projection (deleted in TASK-EMB-008) with real neural weights. The Foundation Layer created the struct and error types; **this task implements the actual loading logic**.

### Constitution Reference

```yaml
# From docs2/constitution.yaml
embeddings:
  models:
    E6_Sparse:
      dim: "~30K 5%active"
      model: "naver/splade-cocondenser-ensembledistil"
      # Requires learned projection matrix to convert to 1536D
```

### AP-007 Compliance

**FORBIDDEN behaviors (will cause task REJECTION):**
- Hash-based projection fallback (`idx % projected_dim`)
- Fake checksums (`0xDEAD_BEEF_CAFE_BABE`)
- CPU fallback when CUDA unavailable
- Simulated weight loading

---

## Input Context Files

| Purpose | File Path | Must Read |
|---------|-----------|-----------|
| Struct definition | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | YES |
| Type constants | `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` | YES |
| Error types | `crates/context-graph-embeddings/src/warm/error.rs` | YES (for reference) |
| Cargo dependencies | `crates/context-graph-embeddings/Cargo.toml` | YES (must modify) |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` | RECOMMENDED |
| Constitution | `docs2/constitution.yaml` | RECOMMENDED |

---

## Prerequisites

- [x] TASK-EMB-002 completed (ProjectionMatrix struct exists) - **VERIFIED**
- [x] TASK-EMB-003 completed (ProjectionError enum exists) - **VERIFIED**
- [ ] `safetensors` crate added to Cargo.toml - **MUST ADD**
- [ ] `sha2` crate added to Cargo.toml - **MUST ADD**
- [x] `candle-core` with CUDA feature enabled - **EXISTS (optional feature)**

---

## Scope

### In Scope

- Add `safetensors = "0.4"` and `sha2 = "0.10"` to Cargo.toml
- Implement `ProjectionMatrix::load(model_dir: &Path) -> Result<Self, ProjectionError>`
- Read SafeTensors file from disk
- Compute SHA256 checksum of file bytes (REAL, not fake)
- Parse tensor header and validate shape is `[30522, 1536]`
- Load tensor to Candle Device (GPU)
- Verify CUDA device (NOT CPU)
- Return populated ProjectionMatrix struct

### Out of Scope

- Projection computation (`project()` method) - **TASK-EMB-012**
- Weight training - separate offline process
- Fallback to hash - **FORBIDDEN by AP-007**
- CPU inference path - **FORBIDDEN**

---

## Definition of Done

### Exact Implementation

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
// ADD these imports at the top:

use candle_core::{Device, Tensor, DType};
use safetensors::SafeTensors;
use sha2::{Sha256, Digest};
use std::fs;
use std::path::Path;

// ADD this impl block (after existing impl block, around line 120):

impl ProjectionMatrix {
    /// Load projection matrix from SafeTensors file.
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing `sparse_projection.safetensors`
    ///
    /// # Returns
    /// * `Ok(Self)` - Loaded projection matrix on GPU
    /// * `Err(ProjectionError)` - If loading fails
    ///
    /// # Errors
    /// - `MatrixMissing` - File not found at `{model_dir}/sparse_projection.safetensors`
    /// - `DimensionMismatch` - Tensor shape is not [30522, 1536]
    /// - `GpuError` - CUDA device unavailable or tensor upload failed
    ///
    /// # CRITICAL: No Fallback Policy (Constitution AP-007)
    /// If the weight file is missing, this function returns an error.
    /// Hash-based projection fallback (`idx % 1536`) is FORBIDDEN.
    /// If CUDA is unavailable, this function returns an error.
    /// CPU fallback is FORBIDDEN.
    ///
    /// # Example
    /// ```rust,ignore
    /// let model_dir = Path::new("/models/sparse");
    /// let projection = ProjectionMatrix::load(model_dir)?;
    /// assert!(projection.is_cuda());
    /// ```
    pub fn load(model_dir: &Path) -> Result<Self, ProjectionError> {
        let weight_path = model_dir.join(PROJECTION_WEIGHT_FILE);

        // Step 1: Read file bytes (REAL file read, not simulation)
        let file_bytes = fs::read(&weight_path).map_err(|e| {
            tracing::error!(
                "[EMB-E006] Weight file not found: {:?}, error: {}",
                weight_path, e
            );
            ProjectionError::MatrixMissing { path: weight_path.clone() }
        })?;

        tracing::info!(
            "Read {} bytes from {:?}",
            file_bytes.len(),
            weight_path
        );

        // Step 2: Compute REAL SHA256 checksum (NOT fake 0xDEAD_BEEF)
        let mut hasher = Sha256::new();
        hasher.update(&file_bytes);
        let checksum: [u8; 32] = hasher.finalize().into();

        tracing::debug!(
            "Computed SHA256 checksum: {:02x}{:02x}{:02x}{:02x}...",
            checksum[0], checksum[1], checksum[2], checksum[3]
        );

        // Step 3: Parse SafeTensors format
        let tensors = SafeTensors::deserialize(&file_bytes).map_err(|e| {
            tracing::error!(
                "[EMB-E001] SafeTensors parse failed: {}",
                e
            );
            ProjectionError::GpuError {
                operation: "SafeTensors::deserialize".to_string(),
                details: e.to_string(),
            }
        })?;

        // Step 4: Get the projection.weight tensor
        let tensor_view = tensors.tensor(PROJECTION_TENSOR_NAME).map_err(|e| {
            tracing::error!(
                "[EMB-E006] Tensor '{}' not found in SafeTensors file: {}",
                PROJECTION_TENSOR_NAME, e
            );
            ProjectionError::MatrixMissing { path: weight_path.clone() }
        })?;

        // Step 5: Validate shape is [30522, 1536]
        let shape = tensor_view.shape();
        if shape.len() != 2 || shape[0] != SPARSE_VOCAB_SIZE || shape[1] != SPARSE_PROJECTED_DIMENSION {
            tracing::error!(
                "[EMB-E005] Shape mismatch: expected [{}, {}], got {:?}",
                SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION, shape
            );
            return Err(ProjectionError::DimensionMismatch {
                path: weight_path,
                actual_rows: shape.get(0).copied().unwrap_or(0),
                actual_cols: shape.get(1).copied().unwrap_or(0),
            });
        }

        tracing::info!(
            "Tensor shape validated: [{}, {}]",
            SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION
        );

        // Step 6: Create CUDA device (NO CPU fallback)
        let device = Device::cuda_if_available(0).map_err(|e| {
            tracing::error!(
                "[EMB-E001] CUDA device creation failed: {}",
                e
            );
            ProjectionError::GpuError {
                operation: "Device::cuda_if_available".to_string(),
                details: e.to_string(),
            }
        })?;

        // Step 7: VERIFY we got CUDA, not CPU (AP-007 compliance)
        if !matches!(&device, Device::Cuda(_)) {
            tracing::error!(
                "[EMB-E001] CUDA device required but got CPU. No CPU fallback allowed."
            );
            return Err(ProjectionError::GpuError {
                operation: "CUDA verification".to_string(),
                details: "No CUDA device available. CPU fallback is FORBIDDEN per Constitution AP-007.".to_string(),
            });
        }

        tracing::info!("CUDA device acquired successfully");

        // Step 8: Load tensor data to GPU
        let weights = Tensor::from_raw_buffer(
            tensor_view.data(),
            DType::F32,
            &[SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION],
            &device,
        ).map_err(|e| {
            tracing::error!(
                "[EMB-E001] Tensor GPU upload failed: {}",
                e
            );
            ProjectionError::GpuError {
                operation: "Tensor::from_raw_buffer".to_string(),
                details: e.to_string(),
            }
        })?;

        tracing::info!(
            "Loaded projection matrix to GPU: {:?}, checksum prefix: {:02x}{:02x}{:02x}{:02x}",
            weights.shape(),
            checksum[0], checksum[1], checksum[2], checksum[3]
        );

        Ok(Self {
            weights,
            device,
            weight_checksum: checksum,
        })
    }
}
```

### Cargo.toml Changes

```toml
# File: crates/context-graph-embeddings/Cargo.toml
# ADD these lines in [dependencies] section:

# SafeTensors for loading pre-trained weights (AP-007 compliant)
safetensors = "0.4"

# SHA256 for weight file integrity verification
sha2 = "0.10"
```

---

## Constraints

| Constraint | Rationale |
|------------|-----------|
| MUST use SafeTensors crate | No manual binary parsing |
| MUST compute REAL SHA256 | No fake checksums (0xDEAD_BEEF forbidden) |
| MUST verify CUDA device | No CPU fallback (AP-007) |
| MUST validate shape [30522, 1536] | Constitution E6_Sparse compliance |
| MUST log all errors with EMB-E0XX codes | SPEC-EMB-001 error taxonomy |
| MUST NOT fall back to hash projection | AP-007 strictly forbids |

---

## Full State Verification (MANDATORY)

### 1. Source of Truth Inspection

After implementing `load()`, you MUST verify:

```bash
# Verify ProjectionMatrix struct has load() method
grep -n "pub fn load" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs

# Verify dependencies added
grep -E "safetensors|sha2" crates/context-graph-embeddings/Cargo.toml

# Verify no fake checksums
grep -rn "DEAD_BEEF\|CAFE_BABE" crates/context-graph-embeddings/src/

# Verify compilation
cargo check -p context-graph-embeddings --features cuda
```

### 2. Edge Case Tests (3 Required)

**Edge Case 1: Missing Weight File**
```rust
#[test]
fn test_load_missing_file() {
    let result = ProjectionMatrix::load(Path::new("/nonexistent/path"));
    assert!(matches!(result, Err(ProjectionError::MatrixMissing { .. })));
    let err = result.unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("EMB-E006"), "Error must contain code EMB-E006");
}
```

**Edge Case 2: Wrong Tensor Shape**
```rust
#[test]
fn test_load_wrong_shape() {
    // Create a temp SafeTensors file with wrong shape [100, 100]
    // Attempt to load
    // Verify DimensionMismatch error with actual dimensions reported
}
```

**Edge Case 3: No CUDA Device**
```rust
#[cfg(not(feature = "cuda"))]
#[test]
fn test_load_no_cuda() {
    // This should fail at compile time due to compile_error!
    // Or return GpuError at runtime if feature is enabled but no GPU
}
```

### 3. Evidence of Success (Physical Proof Required)

After running tests, you MUST capture:

```bash
# Run the tests and capture output
cargo test -p context-graph-embeddings projection::tests -- --nocapture 2>&1 | tee /tmp/projection_test_output.txt

# Verify output contains success indicators
grep -E "test.*ok|PASSED|Loaded projection matrix" /tmp/projection_test_output.txt
```

**Expected Physical Evidence:**
- Test output showing `test_load_missing_file ... ok`
- Test output showing `test_expected_shape_constants ... ok`
- Log output showing `Loaded projection matrix to GPU`
- No output containing `DEAD_BEEF` or `simulate`

---

## Manual Verification Checklist

Before marking this task COMPLETE, manually verify:

- [ ] `projection.rs` contains `pub fn load(model_dir: &Path) -> Result<Self, ProjectionError>`
- [ ] `Cargo.toml` contains `safetensors = "0.4"` and `sha2 = "0.10"`
- [ ] `cargo check -p context-graph-embeddings --features cuda` passes
- [ ] No `simulate_weight_loading` function anywhere in codebase
- [ ] No `0xDEAD_BEEF` or `0xCAFE_BABE` magic numbers
- [ ] All error messages contain EMB-E0XX codes
- [ ] Tests pass: `cargo test -p context-graph-embeddings projection`

---

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-embeddings/Cargo.toml` | ADD safetensors, sha2 dependencies |
| `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | ADD load() implementation |

---

## Validation Criteria

- [ ] `load()` returns `Result<Self, ProjectionError>` - **SIGNATURE CHECK**
- [ ] Missing file returns `ProjectionError::MatrixMissing` - **ERROR HANDLING**
- [ ] Wrong shape returns `ProjectionError::DimensionMismatch` - **VALIDATION**
- [ ] No CPU fallback (error if CUDA unavailable) - **AP-007 COMPLIANCE**
- [ ] Checksum is 32 bytes SHA256 (not fake) - **INTEGRITY**
- [ ] All tracing logs include EMB-E0XX codes - **OBSERVABILITY**

---

## Test Commands

```bash
cd /home/cabdru/contextgraph

# Step 1: Add dependencies and verify compilation
cargo check -p context-graph-embeddings --features cuda

# Step 2: Run projection tests
cargo test -p context-graph-embeddings projection --features cuda -- --nocapture

# Step 3: Verify no simulation code remains
grep -rn "simulate_weight_loading\|DEAD_BEEF\|CAFE_BABE" crates/context-graph-embeddings/src/
# Expected output: (empty - no matches)

# Step 4: Verify load() method exists
grep -n "pub fn load" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
# Expected output: line number with "pub fn load(model_dir: &Path)"
```

---

## Related Tasks

| Task ID | Title | Relationship |
|---------|-------|--------------|
| TASK-EMB-002 | Create ProjectionMatrix Struct | **PREREQUISITE** (COMPLETE) |
| TASK-EMB-003 | Create ProjectionError Enum | **PREREQUISITE** (COMPLETE) |
| TASK-EMB-012 | Implement ProjectionMatrix::project() | **NEXT TASK** |
| TASK-EMB-013 | Replace simulate_weight_loading() | PARALLEL (warm module) |

---

## SIMULATION CODE TO DELETE (Reference)

The following code exists in `warm/loader/operations.rs` and MUST be deleted in TASK-EMB-013:

```rust
// DO NOT COPY THIS - IT IS FORBIDDEN
// This is documented here so you know what to look for and DELETE

pub fn simulate_weight_loading(model_id: &str, _size_bytes: usize) -> WarmResult<u64> {
    let mut checksum = 0u64;
    for (i, byte) in model_id.bytes().enumerate() {
        checksum ^= (byte as u64) << ((i % 8) * 8);
    }
    checksum ^= 0xDEAD_BEEF_CAFE_BABEu64;  // FAKE CHECKSUM - FORBIDDEN
    Ok(checksum)
}

// Also DELETE this fake pointer generation:
let base_ptr = 0x7f80_0000_0000u64;  // FAKE POINTER - FORBIDDEN
let offset = memory_pools.list_model_allocations().len() as u64 * 0x1_0000_0000;
let vram_ptr = base_ptr + offset;

// Also DELETE this sin wave output:
let output: Vec<f32> = (0..expected_dimension)
    .map(|i| (i as f32 * 0.001).sin())  // FAKE OUTPUT - FORBIDDEN
    .collect();
```

**Note:** This task (TASK-EMB-011) does NOT delete the above code. It is deleted in TASK-EMB-013. This reference is provided so you understand what patterns are FORBIDDEN.

---

## Pseudo Code Summary

```
load(model_dir):
    weight_path = model_dir / "sparse_projection.safetensors"

    file_bytes = fs::read(weight_path)
        -> ERROR: MatrixMissing if file not found

    checksum = SHA256::digest(file_bytes)  // REAL checksum, not fake

    tensors = SafeTensors::deserialize(file_bytes)
        -> ERROR: GpuError if parse fails

    tensor_view = tensors.tensor("projection.weight")
        -> ERROR: MatrixMissing if tensor not found

    ASSERT shape == [30522, 1536]
        -> ERROR: DimensionMismatch if wrong shape

    device = Device::cuda_if_available(0)
        -> ERROR: GpuError if CUDA unavailable

    ASSERT device is CUDA (not CPU)
        -> ERROR: GpuError if CPU fallback attempted

    weights = Tensor::from_raw_buffer(tensor_view.data(), ...)
        -> ERROR: GpuError if upload fails

    RETURN ProjectionMatrix { weights, device, checksum }
```

</task_spec>
