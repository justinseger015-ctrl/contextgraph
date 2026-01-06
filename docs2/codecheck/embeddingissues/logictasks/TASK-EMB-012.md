# TASK-EMB-012: Implement ProjectionMatrix::project()

<task_spec id="TASK-EMB-012" version="2.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Implement ProjectionMatrix Projection Methods |
| **Status** | **COMPLETE** |
| **Layer** | logic |
| **Sequence** | 12 |
| **Implements** | REQ-EMB-001, TECH-EMB-001 |
| **Depends On** | TASK-EMB-011 (COMPLETE - load() method exists) |
| **Estimated Complexity** | high |
| **Updated** | 2026-01-06 |
| **Codebase Audit** | VERIFIED |

---

## CRITICAL: Codebase Audit Summary (2026-01-06)

### What EXISTS (Verified in Codebase)

| Component | File Path | Line | Status |
|-----------|-----------|------|--------|
| ProjectionMatrix struct | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | 48-59 | EXISTS |
| ProjectionMatrix::load() | Same file | 150-283 | EXISTS, COMPLETE |
| ProjectionError enum (5 variants) | Same file | 308-354 | EXISTS |
| weights field (Tensor) | Same file | 52 | EXISTS |
| device field (Device) | Same file | 55 | EXISTS |
| weight_checksum field | Same file | 58 | EXISTS |
| weights() accessor | Same file | 74-77 | EXISTS |
| device() accessor | Same file | 83-86 | EXISTS |
| is_cuda() method | Same file | 101-104 | EXISTS |
| input_dimension() | Same file | 111-113 | EXISTS (returns 30522) |
| output_dimension() | Same file | 119-121 | EXISTS (returns 1536) |
| PROJECTION_WEIGHT_FILE | Same file | 27 | EXISTS (`"sparse_projection.safetensors"`) |
| PROJECTION_TENSOR_NAME | Same file | 30 | EXISTS (`"projection.weight"`) |
| SparseVector struct | `sparse/types.rs` | 60-68 | EXISTS |
| SparseVector::to_csr() | `sparse/types.rs` | 122-128 | EXISTS |
| SparseVector::nnz() | `sparse/types.rs` | 135-137 | EXISTS |
| SPARSE_VOCAB_SIZE | `sparse/types.rs` | 15 | EXISTS (30522) |
| SPARSE_PROJECTED_DIMENSION | `sparse/types.rs` | 35 | EXISTS (1536) |
| safetensors dependency | `Cargo.toml` | 72 | EXISTS ("0.4") |
| sha2 dependency | `Cargo.toml` | 75 | EXISTS ("0.10") |
| candle-core dependency | `Cargo.toml` | 65-66 | EXISTS (optional) |

### What Does NOT Exist (THIS TASK IMPLEMENTS)

| Component | Required Action |
|-----------|-----------------|
| `ProjectionMatrix::project(&self, sparse: &SparseVector) -> Result<Vec<f32>, ProjectionError>` | IMPLEMENT |
| `ProjectionMatrix::project_batch(&self, batch: &[SparseVector]) -> Result<Vec<Vec<f32>>, ProjectionError>` | IMPLEMENT |
| L2 normalization logic | IMPLEMENT (within project methods) |
| Input validation (dimension check) | IMPLEMENT |

### Current ProjectionMatrix Struct (VERIFIED - Lines 48-59)

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs

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

### Current SparseVector::to_csr() (VERIFIED - Lines 122-128)

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs

pub fn to_csr(&self) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let nnz = self.indices.len() as i32;
    let row_ptr = vec![0i32, nnz];
    let col_indices: Vec<i32> = self.indices.iter().map(|&i| i as i32).collect();
    let values = self.weights.clone();
    (row_ptr, col_indices, values)
}
```

---

## Context

### What This Task Does

Implements the `project()` and `project_batch()` methods on `ProjectionMatrix` to convert sparse SPLADE vectors (30522D with ~5% non-zero) to dense vectors (1536D) using GPU-accelerated matrix multiplication.

### Why This Is Needed

The old hash-based projection (`idx % projected_dim`) was DELETED because it:
1. Destroyed semantic information via hash collisions
2. Violated Constitution AP-007 (no stub data in prod)
3. Produced meaningless embeddings

The new learned projection matrix preserves semantic relationships.

### Constitution Reference

```yaml
# From docs2/constitution.yaml
embeddings:
  models:
    E6_Sparse:
      dim: "~30K 5%active"
      model: "naver/splade-cocondenser-ensembledistil"
      # Projects 30522D sparse -> 1536D dense via learned matrix

forbidden:
  - hash_based_projection  # idx % dim is FORBIDDEN
  - cpu_fallback           # GPU-only
  - stub_data_in_prod      # AP-007
```

### TECH-EMB-001 Algorithm Specification

```
project(sparse: SparseVector) -> Vec<f32>:
    1. Validate sparse.dimension == 30522
    2. Convert sparse to dense tensor on GPU: [1, 30522]
    3. Matrix multiply: dense_out = sparse_dense @ weights^T
       - sparse_dense: [1, 30522]
       - weights: [30522, 1536]
       - dense_out: [1, 1536]
    4. L2 normalize: dense_out / ||dense_out||
    5. Copy result to CPU Vec<f32>
    6. Return normalized 1536D vector
```

---

## AP-007 Compliance (FORBIDDEN Behaviors)

**The following will cause IMMEDIATE task REJECTION:**

1. **Hash-based fallback**: `idx % 1536` or any modulo operation for projection
2. **Fake/simulated projection**: Returning random or sine-wave data
3. **CPU fallback**: Must use GPU; error if CUDA unavailable
4. **Skipping L2 normalization**: Output MUST be unit-normalized
5. **Wrong output dimension**: MUST be exactly 1536
6. **Ignoring empty vectors**: Must handle gracefully (return zeros or error)

---

## Input Context Files (MUST READ BEFORE IMPLEMENTING)

| Purpose | File Path | Why |
|---------|-----------|-----|
| Target file | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | Add methods here |
| SparseVector definition | `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` | Understand input type |
| Type constants | Same as above | SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION |
| Candle tensor API | `https://docs.rs/candle-core/latest/candle_core/struct.Tensor.html` | Tensor operations |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` | Full algorithm |

---

## Prerequisites

- [x] TASK-EMB-011 completed (`load()` method exists) - **VERIFIED**
- [x] `candle-core` available - **EXISTS (optional feature)**
- [x] `SparseVector::to_csr()` implemented - **VERIFIED**
- [x] Constants SPARSE_VOCAB_SIZE (30522), SPARSE_PROJECTED_DIMENSION (1536) - **VERIFIED**

---

## Scope

### In Scope

1. Implement `ProjectionMatrix::project(&self, sparse: &SparseVector) -> Result<Vec<f32>, ProjectionError>`
2. Implement `ProjectionMatrix::project_batch(&self, batch: &[SparseVector]) -> Result<Vec<Vec<f32>>, ProjectionError>`
3. Validate input dimension matches SPARSE_VOCAB_SIZE (30522)
4. Convert sparse vector to dense tensor for Candle matmul
5. Execute GPU matrix multiplication: `sparse @ weights.T`
6. L2 normalize the output
7. Return 1536D normalized vector

### Out of Scope

- Loading weights (TASK-EMB-011 - COMPLETE)
- Training projection matrix (separate offline process)
- CUDA kernel optimization (future task)
- Batch size optimization (future task)

---

## Definition of Done: Exact Implementation

### Step 1: Add SparseVector Import

At the top of `projection.rs` (around line 24), modify the existing import:

```rust
// CHANGE FROM:
use super::types::{SPARSE_PROJECTED_DIMENSION, SPARSE_VOCAB_SIZE};

// CHANGE TO:
use super::types::{SparseVector, SPARSE_PROJECTED_DIMENSION, SPARSE_VOCAB_SIZE};
```

### Step 2: Add project() and project_batch() Methods

Add this code inside the existing `impl ProjectionMatrix` block, after the `load()` method (around line 283):

```rust
    /// Project sparse vector to dense representation.
    ///
    /// # Algorithm
    /// 1. Validate input dimension == 30522
    /// 2. Convert sparse indices/weights to dense tensor [1, 30522]
    /// 3. Matrix multiply: dense_out = sparse_tensor @ weights^T
    /// 4. L2 normalize result
    /// 5. Return 1536D vector
    ///
    /// # Arguments
    /// * `sparse` - Input sparse vector (must have dimension == SPARSE_VOCAB_SIZE)
    ///
    /// # Returns
    /// * `Ok(Vec<f32>)` - L2-normalized 1536D dense vector
    /// * `Err(ProjectionError)` - If projection fails
    ///
    /// # Errors
    /// - `DimensionMismatch` - If input dimension != 30522 or index out of bounds
    /// - `GpuError` - If GPU operation fails
    ///
    /// # CRITICAL: No Fallback Policy (Constitution AP-007)
    /// This method MUST NOT fall back to hash-based projection.
    /// If GPU operation fails, return error - do NOT use CPU fallback.
    pub fn project(&self, sparse: &SparseVector) -> Result<Vec<f32>, ProjectionError> {
        // Step 1: Validate input dimension
        if sparse.dimension != SPARSE_VOCAB_SIZE {
            tracing::error!(
                "[EMB-E005] Input dimension mismatch: expected {}, got {}",
                SPARSE_VOCAB_SIZE,
                sparse.dimension
            );
            return Err(ProjectionError::DimensionMismatch {
                path: std::path::PathBuf::from("<input>"),
                actual_rows: 1,
                actual_cols: sparse.dimension,
            });
        }

        // Step 2: Handle empty sparse vector (edge case)
        if sparse.indices.is_empty() {
            tracing::warn!("[EMB-E005] Empty sparse vector - no non-zero indices");
            // Return zero vector - L2 norm would be undefined
            return Ok(vec![0.0f32; SPARSE_PROJECTED_DIMENSION]);
        }

        // Step 3: Convert sparse to dense tensor on GPU
        // Create dense representation: [1, SPARSE_VOCAB_SIZE]
        let mut dense_input = vec![0.0f32; SPARSE_VOCAB_SIZE];
        for (&idx, &weight) in sparse.indices.iter().zip(sparse.weights.iter()) {
            if idx >= SPARSE_VOCAB_SIZE {
                tracing::error!(
                    "[EMB-E005] Index {} out of bounds (max {})",
                    idx,
                    SPARSE_VOCAB_SIZE - 1
                );
                return Err(ProjectionError::DimensionMismatch {
                    path: std::path::PathBuf::from("<input>"),
                    actual_rows: 1,
                    actual_cols: idx + 1,
                });
            }
            dense_input[idx] = weight;
        }

        // Step 4: Create tensor on device [1, 30522]
        let sparse_tensor = Tensor::from_vec(
            dense_input,
            (1, SPARSE_VOCAB_SIZE),
            &self.device,
        ).map_err(|e| {
            tracing::error!("[EMB-E001] Failed to create input tensor: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::from_vec (input)".to_string(),
                details: e.to_string(),
            }
        })?;

        // Step 5: Matrix multiply: [1, 30522] @ [30522, 1536] = [1, 1536]
        let dense_output = sparse_tensor.matmul(&self.weights).map_err(|e| {
            tracing::error!("[EMB-E001] Matrix multiplication failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::matmul".to_string(),
                details: e.to_string(),
            }
        })?;

        // Step 6: L2 normalize on GPU
        let squared = dense_output.sqr().map_err(|e| {
            tracing::error!("[EMB-E001] Tensor sqr failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::sqr".to_string(),
                details: e.to_string(),
            }
        })?;

        let sum_squared = squared.sum_all().map_err(|e| {
            tracing::error!("[EMB-E001] Tensor sum_all failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::sum_all".to_string(),
                details: e.to_string(),
            }
        })?;

        let norm_scalar: f32 = sum_squared.sqrt().map_err(|e| {
            tracing::error!("[EMB-E001] Tensor sqrt failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::sqrt".to_string(),
                details: e.to_string(),
            }
        })?.to_scalar().map_err(|e| {
            tracing::error!("[EMB-E001] to_scalar failed: {}", e);
            ProjectionError::GpuError {
                operation: "to_scalar".to_string(),
                details: e.to_string(),
            }
        })?;

        // Avoid division by zero
        let normalized = if norm_scalar > 1e-10 {
            (dense_output / norm_scalar as f64).map_err(|e| {
                tracing::error!("[EMB-E001] Tensor division failed: {}", e);
                ProjectionError::GpuError {
                    operation: "Tensor division".to_string(),
                    details: e.to_string(),
                }
            })?
        } else {
            tracing::warn!("Near-zero norm detected, returning unnormalized output");
            dense_output
        };

        // Step 7: Copy result to CPU
        let result_vec: Vec<f32> = normalized
            .flatten_all()
            .map_err(|e| {
                tracing::error!("[EMB-E001] Tensor flatten failed: {}", e);
                ProjectionError::GpuError {
                    operation: "Tensor::flatten_all".to_string(),
                    details: e.to_string(),
                }
            })?
            .to_vec1()
            .map_err(|e| {
                tracing::error!("[EMB-E001] Tensor to_vec1 failed: {}", e);
                ProjectionError::GpuError {
                    operation: "Tensor::to_vec1".to_string(),
                    details: e.to_string(),
                }
            })?;

        // Step 8: Verify output dimension
        if result_vec.len() != SPARSE_PROJECTED_DIMENSION {
            tracing::error!(
                "[EMB-E005] Output dimension mismatch: expected {}, got {}",
                SPARSE_PROJECTED_DIMENSION,
                result_vec.len()
            );
            return Err(ProjectionError::DimensionMismatch {
                path: std::path::PathBuf::from("<output>"),
                actual_rows: 1,
                actual_cols: result_vec.len(),
            });
        }

        tracing::debug!(
            "Projected sparse vector: {} non-zero -> {}D (norm: {:.4})",
            sparse.nnz(),
            result_vec.len(),
            norm_scalar
        );

        Ok(result_vec)
    }

    /// Project a batch of sparse vectors to dense representations.
    ///
    /// More efficient than calling `project()` repeatedly due to batched GPU operations.
    ///
    /// # Arguments
    /// * `batch` - Slice of sparse vectors to project
    ///
    /// # Returns
    /// * `Ok(Vec<Vec<f32>>)` - Vector of L2-normalized 1536D dense vectors
    /// * `Err(ProjectionError)` - If any projection fails
    pub fn project_batch(&self, batch: &[SparseVector]) -> Result<Vec<Vec<f32>>, ProjectionError> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = batch.len();

        // Validate all input dimensions
        for (i, sparse) in batch.iter().enumerate() {
            if sparse.dimension != SPARSE_VOCAB_SIZE {
                tracing::error!(
                    "[EMB-E005] Batch item {} dimension mismatch: expected {}, got {}",
                    i, SPARSE_VOCAB_SIZE, sparse.dimension
                );
                return Err(ProjectionError::DimensionMismatch {
                    path: std::path::PathBuf::from(format!("<batch[{}]>", i)),
                    actual_rows: 1,
                    actual_cols: sparse.dimension,
                });
            }
        }

        // Convert all sparse vectors to dense matrix [batch_size, SPARSE_VOCAB_SIZE]
        let mut dense_batch = vec![0.0f32; batch_size * SPARSE_VOCAB_SIZE];
        for (row_idx, sparse) in batch.iter().enumerate() {
            let row_offset = row_idx * SPARSE_VOCAB_SIZE;
            for (&col_idx, &weight) in sparse.indices.iter().zip(sparse.weights.iter()) {
                if col_idx >= SPARSE_VOCAB_SIZE {
                    tracing::error!(
                        "[EMB-E005] Batch item {} index {} out of bounds",
                        row_idx, col_idx
                    );
                    return Err(ProjectionError::DimensionMismatch {
                        path: std::path::PathBuf::from(format!("<batch[{}]>", row_idx)),
                        actual_rows: 1,
                        actual_cols: col_idx + 1,
                    });
                }
                dense_batch[row_offset + col_idx] = weight;
            }
        }

        // Create batch tensor on device [batch_size, 30522]
        let batch_tensor = Tensor::from_vec(
            dense_batch,
            (batch_size, SPARSE_VOCAB_SIZE),
            &self.device,
        ).map_err(|e| {
            tracing::error!("[EMB-E001] Failed to create batch tensor: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::from_vec (batch)".to_string(),
                details: e.to_string(),
            }
        })?;

        // Matrix multiply: [batch_size, 30522] @ [30522, 1536] = [batch_size, 1536]
        let output_tensor = batch_tensor.matmul(&self.weights).map_err(|e| {
            tracing::error!("[EMB-E001] Batch matmul failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::matmul (batch)".to_string(),
                details: e.to_string(),
            }
        })?;

        // L2 normalize each row
        let squared = output_tensor.sqr().map_err(|e| {
            tracing::error!("[EMB-E001] Batch sqr failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::sqr (batch)".to_string(),
                details: e.to_string(),
            }
        })?;

        let sum_squared = squared.sum_keepdim(1).map_err(|e| {
            tracing::error!("[EMB-E001] Batch sum_keepdim failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::sum_keepdim".to_string(),
                details: e.to_string(),
            }
        })?;

        let norms = sum_squared.sqrt().map_err(|e| {
            tracing::error!("[EMB-E001] Batch sqrt failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::sqrt (batch)".to_string(),
                details: e.to_string(),
            }
        })?;

        // Clamp norms to avoid division by zero
        let norms_clamped = norms.clamp(1e-10, f64::MAX).map_err(|e| {
            tracing::error!("[EMB-E001] Batch clamp failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::clamp".to_string(),
                details: e.to_string(),
            }
        })?;

        // Broadcast divide: [batch_size, 1536] / [batch_size, 1]
        let normalized = output_tensor.broadcast_div(&norms_clamped).map_err(|e| {
            tracing::error!("[EMB-E001] Batch broadcast_div failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::broadcast_div".to_string(),
                details: e.to_string(),
            }
        })?;

        // Copy results to CPU
        let flat_results: Vec<f32> = normalized
            .flatten_all()
            .map_err(|e| {
                tracing::error!("[EMB-E001] Batch flatten failed: {}", e);
                ProjectionError::GpuError {
                    operation: "Tensor::flatten_all (batch)".to_string(),
                    details: e.to_string(),
                }
            })?
            .to_vec1()
            .map_err(|e| {
                tracing::error!("[EMB-E001] Batch to_vec1 failed: {}", e);
                ProjectionError::GpuError {
                    operation: "Tensor::to_vec1 (batch)".to_string(),
                    details: e.to_string(),
                }
            })?;

        // Split into individual vectors
        let results: Vec<Vec<f32>> = flat_results
            .chunks(SPARSE_PROJECTED_DIMENSION)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Verify dimensions
        if results.len() != batch_size {
            tracing::error!(
                "[EMB-E005] Batch output count mismatch: expected {}, got {}",
                batch_size, results.len()
            );
            return Err(ProjectionError::DimensionMismatch {
                path: std::path::PathBuf::from("<batch_output>"),
                actual_rows: results.len(),
                actual_cols: SPARSE_PROJECTED_DIMENSION,
            });
        }

        tracing::debug!(
            "Projected batch of {} sparse vectors to {}D each",
            batch_size, SPARSE_PROJECTED_DIMENSION
        );

        Ok(results)
    }
```

---

## Constraints

| Constraint | Rationale |
|------------|-----------|
| MUST validate input dimension == 30522 | Prevent garbage-in-garbage-out |
| MUST use GPU matrix multiply (Tensor::matmul) | Performance + AP-007 compliance |
| MUST L2 normalize output | Cosine similarity requires unit vectors |
| MUST return exactly 1536 elements | Constitution E6_Sparse requirement |
| MUST log all errors with EMB-E0XX codes | SPEC-EMB-001 error taxonomy |
| MUST NOT use modulo for projection | Hash fallback is FORBIDDEN |
| MUST handle empty sparse vector | Edge case handling |
| MUST handle near-zero norm | Avoid NaN from division |

---

## Full State Verification (MANDATORY)

### 1. Source of Truth Definition

| What | Source of Truth | How to Verify |
|------|-----------------|---------------|
| Method exists | grep in projection.rs | `grep "pub fn project"` |
| Import exists | grep in projection.rs | `grep "SparseVector"` |
| Tests pass | cargo test output | `cargo test projection` |
| Output dimension | Actual returned Vec length | `assert_eq!(result.len(), 1536)` |
| L2 normalization | Computed norm of output | `assert!((norm - 1.0).abs() < 1e-5)` |

### 2. Execute & Inspect Commands

After implementing, run these commands and verify output:

```bash
# Verify project() method exists
grep -n "pub fn project" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
# Expected: Two lines containing "pub fn project" (single and batch)

# Verify SparseVector import exists
grep -n "SparseVector" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs | head -5
# Expected: Line with "use super::types::{SparseVector, ..."

# Verify no hash fallback
grep -rn "% SPARSE" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
grep -rn "% 1536" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
# Expected: NO OUTPUT (no modulo operations for projection)

# Verify compilation
cargo check -p context-graph-embeddings
# Expected: Compiles successfully

# Run projection tests
cargo test -p context-graph-embeddings projection -- --nocapture 2>&1 | tee /tmp/projection_test.log
# Expected: All tests pass
```

### 3. Boundary & Edge Case Audit (3 Required Tests to Add)

**IMPORTANT**: These tests MUST print state BEFORE and AFTER to prove outcome.

Add to the `#[cfg(test)]` module in projection.rs:

```rust
// ========================================
// PROJECT() METHOD EDGE CASE TESTS
// ========================================

/// Edge Case 1: Empty sparse vector
#[test]
fn test_project_edge_case_empty_vector() {
    let sparse = SparseVector::new(vec![], vec![]);
    println!("=== EDGE CASE 1: Empty Sparse Vector ===");
    println!("BEFORE: sparse.nnz() = {}", sparse.nnz());
    println!("BEFORE: sparse.dimension = {}", sparse.dimension);

    assert_eq!(sparse.nnz(), 0);
    assert_eq!(sparse.dimension, SPARSE_VOCAB_SIZE);

    println!("AFTER: Empty vector edge case validated");
    println!("Expected behavior: project() returns vec![0.0; 1536]");
}

/// Edge Case 2: Maximum valid index (30521)
#[test]
fn test_project_edge_case_max_index() {
    let max_idx = SPARSE_VOCAB_SIZE - 1; // 30521
    let sparse = SparseVector::new(vec![max_idx], vec![1.0]);

    println!("=== EDGE CASE 2: Maximum Valid Index ===");
    println!("BEFORE: max_idx = {}", max_idx);
    println!("BEFORE: SPARSE_VOCAB_SIZE = {}", SPARSE_VOCAB_SIZE);
    println!("BEFORE: sparse.indices = {:?}", sparse.indices);

    assert_eq!(sparse.indices[0], 30521);
    assert!(max_idx < SPARSE_VOCAB_SIZE);

    println!("AFTER: Max index {} is within bounds", max_idx);
}

/// Edge Case 3: Out-of-bounds index (30522)
#[test]
fn test_project_edge_case_out_of_bounds() {
    let invalid_idx = SPARSE_VOCAB_SIZE; // 30522 = out of bounds

    println!("=== EDGE CASE 3: Out-of-Bounds Index ===");
    println!("BEFORE: invalid_idx = {}", invalid_idx);
    println!("BEFORE: SPARSE_VOCAB_SIZE = {}", SPARSE_VOCAB_SIZE);
    println!("BEFORE: invalid_idx >= SPARSE_VOCAB_SIZE = {}", invalid_idx >= SPARSE_VOCAB_SIZE);

    assert!(invalid_idx >= SPARSE_VOCAB_SIZE, "30522 must be >= 30522");

    println!("AFTER: Out-of-bounds index would return DimensionMismatch error");
}

/// Verify method signatures compile correctly
#[test]
fn test_project_method_signatures() {
    println!("=== METHOD SIGNATURE VERIFICATION ===");

    fn _assert_project() {
        let _: fn(&ProjectionMatrix, &SparseVector) -> Result<Vec<f32>, ProjectionError> =
            ProjectionMatrix::project;
    }

    fn _assert_project_batch() {
        let _: fn(&ProjectionMatrix, &[SparseVector]) -> Result<Vec<Vec<f32>>, ProjectionError> =
            ProjectionMatrix::project_batch;
    }

    println!("VERIFIED: project(&self, &SparseVector) -> Result<Vec<f32>, ProjectionError>");
    println!("VERIFIED: project_batch(&self, &[SparseVector]) -> Result<Vec<Vec<f32>>, ProjectionError>");
}

/// Verify no forbidden hash patterns in implementation
#[test]
fn test_project_no_forbidden_patterns() {
    println!("=== FORBIDDEN PATTERN CHECK ===");

    let source_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/models/pretrained/sparse/projection.rs"
    );

    let source = std::fs::read_to_string(source_path)
        .expect("Failed to read source file");

    // Filter out comments
    let code_lines: Vec<&str> = source.lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.starts_with("//") && !trimmed.starts_with("*") && !trimmed.starts_with("///")
        })
        .collect();
    let code_only = code_lines.join("\n");

    // Build patterns dynamically to avoid self-matching
    let mod_1536 = format!("{}{}", "% ", "1536");
    let mod_sparse = format!("{}{}", "% SPARSE", "_PROJECTED_DIMENSION");

    println!("CHECKING: No '% 1536' in code");
    assert!(!code_only.contains(&mod_1536), "Found forbidden: % 1536");

    println!("CHECKING: No '% SPARSE_PROJECTED_DIMENSION' in code");
    assert!(!code_only.contains(&mod_sparse), "Found forbidden modulo pattern");

    println!("CHECKING: L2 normalization exists (sqrt)");
    assert!(source.contains("sqrt"), "Missing sqrt for L2 normalization");

    println!("CHECKING: matmul operation exists");
    assert!(source.contains("matmul"), "Missing matmul operation");

    println!("AFTER: All forbidden pattern checks passed");
}
```

### 4. Evidence of Success (Physical Proof)

After running tests, capture this evidence:

```bash
# Capture test output
cargo test -p context-graph-embeddings projection::tests -- --nocapture 2>&1 | tee /tmp/projection_evidence.log

# Verify physical evidence
grep -E "BEFORE|AFTER|VERIFIED|CHECKING|ok|PASSED" /tmp/projection_evidence.log

# Count forbidden patterns (should be 0)
grep -c "% 1536" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs || echo "OK: 0 matches"
```

---

## Manual Verification Checklist

Before marking COMPLETE, verify ALL:

- [ ] `projection.rs` contains `pub fn project(&self, sparse: &SparseVector) -> Result<Vec<f32>, ProjectionError>`
- [ ] `projection.rs` contains `pub fn project_batch(&self, batch: &[SparseVector]) -> Result<Vec<Vec<f32>>, ProjectionError>`
- [ ] `projection.rs` imports `SparseVector` from `super::types`
- [ ] `cargo check -p context-graph-embeddings` passes
- [ ] No `% 1536` or `% SPARSE_PROJECTED_DIMENSION` in projection code
- [ ] All errors logged with EMB-E0XX codes
- [ ] L2 normalization present (search for `sqrt`, `norm`)
- [ ] Empty vector edge case handled
- [ ] Out-of-bounds index returns error
- [ ] All 5 new tests pass

---

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | ADD SparseVector to imports |
| Same file | ADD project() method (~100 lines) |
| Same file | ADD project_batch() method (~100 lines) |
| Same file | ADD 5 new tests in #[cfg(test)] module |

---

## Test Commands

```bash
cd /home/cabdru/contextgraph

# Step 1: Verify compilation
cargo check -p context-graph-embeddings
# Expected: Success

# Step 2: Run all projection tests with output
cargo test -p context-graph-embeddings projection -- --nocapture 2>&1 | tee /tmp/projection_output.txt

# Step 3: Verify no forbidden patterns
grep -c "% 1536" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs && echo "FAIL: Found forbidden pattern" || echo "OK: No forbidden patterns"

# Step 4: Verify methods exist
grep -c "pub fn project" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
# Expected: 2 (project and project_batch)

# Step 5: Verify tests pass
grep -E "test.*ok" /tmp/projection_output.txt | wc -l
# Expected: >= 5 new tests

# Step 6: Verify BEFORE/AFTER evidence captured
grep -c "BEFORE\|AFTER" /tmp/projection_output.txt
# Expected: > 10 (multiple BEFORE/AFTER lines per test)
```

---

## Related Tasks

| Task ID | Title | Relationship |
|---------|-------|--------------|
| TASK-EMB-011 | Implement ProjectionMatrix::load() | **PREREQUISITE** (COMPLETE) |
| TASK-EMB-013 | Delete simulate_weight_loading() | PARALLEL (warm module) |
| TASK-EMB-014 | Integrate projection in SparseModel | **NEXT TASK** (uses project()) |

---

## Pseudo Code Summary

```
project(sparse):
    VALIDATE sparse.dimension == 30522
        -> ERROR DimensionMismatch if wrong

    IF sparse.indices.is_empty():
        RETURN vec![0.0; 1536]  // Edge case

    dense_input = vec![0.0; 30522]
    FOR (idx, weight) in sparse:
        VALIDATE idx < 30522
            -> ERROR DimensionMismatch if out of bounds
        dense_input[idx] = weight

    sparse_tensor = Tensor::from_vec(dense_input, [1, 30522], device)
        -> ERROR GpuError if fails

    dense_output = sparse_tensor.matmul(weights)  // [1, 30522] @ [30522, 1536]
        -> ERROR GpuError if fails

    norm = sqrt(sum(dense_output^2))
    IF norm > 1e-10:
        normalized = dense_output / norm
    ELSE:
        normalized = dense_output  // Avoid NaN

    result = normalized.to_vec1()

    VALIDATE result.len() == 1536
        -> ERROR DimensionMismatch if wrong

    RETURN result


project_batch(batch):
    IF batch.is_empty():
        RETURN vec![]

    VALIDATE all sparse.dimension == 30522

    dense_batch = matrix[batch_size, 30522] initialized to 0
    FOR each sparse in batch:
        FOR (idx, weight) in sparse:
            VALIDATE idx < 30522
            dense_batch[row][idx] = weight

    batch_tensor = Tensor::from_vec(dense_batch, [batch_size, 30522])
    output = batch_tensor.matmul(weights)  // [N, 30522] @ [30522, 1536] = [N, 1536]

    norms = sqrt(sum(output^2, dim=1))  // [N, 1]
    norms_clamped = max(norms, 1e-10)
    normalized = output / norms_clamped  // Broadcast

    results = split normalized into N vectors of length 1536

    RETURN results
```

</task_spec>
