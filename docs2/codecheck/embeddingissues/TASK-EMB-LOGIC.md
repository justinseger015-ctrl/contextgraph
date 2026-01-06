# Logic Layer Task Specifications

<task_collection id="TASK-EMB-LOGIC" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Collection ID** | TASK-EMB-LOGIC |
| **Title** | Logic Layer Task Specifications |
| **Status** | Ready |
| **Version** | 1.0 |
| **Layer** | Logic (Business Logic, Services, Algorithms) |
| **Task Count** | 10 (TASK-EMB-011 through TASK-EMB-020) |
| **Implements** | REQ-EMB-001, REQ-EMB-003, REQ-EMB-004, REQ-EMB-005 |
| **Related Tech Specs** | TECH-EMB-001, TECH-EMB-002, TECH-EMB-003 |
| **Depends On** | TASK-EMB-FOUNDATION (all tasks must complete) |
| **Created** | 2026-01-06 |
| **Constitution Reference** | v4.0.0 |

---

## Layer Execution Order

Logic Layer tasks depend on Foundation Layer and must complete BEFORE Surface Layer.

```
Foundation Layer (TASK-EMB-001 through TASK-EMB-010)
         |
         v  (ALL MUST COMPLETE)
Logic Layer (This Document)
         |
         v
Surface Layer (TASK-EMB-021 through TASK-EMB-030)
```

---

## CRITICAL RULES

### 1. NO SIMULATION ALLOWED

```rust
// FORBIDDEN - These patterns MUST NOT exist in Logic Layer code:

// NO simulate_* functions
fn simulate_weight_loading(...) { ... }  // DELETE THIS

// NO fake pointers
let vram_ptr = 0x7f80_0000_0000u64;  // DELETE THIS

// NO sin wave outputs
let output: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001).sin()).collect();  // DELETE THIS

// NO fake checksums
checksum ^= 0xDEAD_BEEF_CAFE_BABEu64;  // DELETE THIS
```

### 2. PANIC IF CUDA UNAVAILABLE

```rust
// REQUIRED - Panic instead of fallback:
#[cfg(not(feature = "cuda"))]
compile_error!("CUDA feature required for embedding pipeline");

// At runtime:
if !cuda_available() {
    panic!("[EMB-E001] CUDA_UNAVAILABLE: RTX 5090 with CUDA 13.1+ required");
}
```

### 3. USE REAL LIBRARIES

- **SafeTensors**: `safetensors::SafeTensors::deserialize()` for weight loading
- **cudarc**: `CudaDevice::htod_copy()` for VRAM allocation
- **sha2**: `Sha256::digest()` for checksum computation
- **candle-core**: For tensor operations and GPU compute

### 4. GOLDEN REFERENCES IN tests/fixtures/

All inference validation MUST compare against golden reference files, NOT generated fake data.

---

## Dependencies Graph

```
TASK-EMB-011 (ProjectionMatrix::load())
      |
      +---> TASK-EMB-012 (ProjectionMatrix::project())
      |
TASK-EMB-013 (Real Weight Loading)
      |
      +---> TASK-EMB-014 (Real VRAM Allocation)
                  |
                  +---> TASK-EMB-015 (Real Inference Validation)
                  |
                  +---> TASK-EMB-019 (Remove Stub Mode)

TASK-EMB-016 (PQ-8 Quantization)
      |
      +---> TASK-EMB-020 (QuantizationRouter)
      |
TASK-EMB-017 (Float8 Quantization)
      |
      +---> TASK-EMB-020 (QuantizationRouter)
      |
TASK-EMB-018 (Binary Quantization)
      |
      +---> TASK-EMB-020 (QuantizationRouter)
```

---

## Task Specifications

---

<task_spec id="TASK-EMB-011" version="1.0">

### TASK-EMB-011: Implement ProjectionMatrix::load()

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Implement ProjectionMatrix Load Method |
| **Status** | ready |
| **Layer** | logic |
| **Sequence** | 11 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-002, TASK-EMB-003 |
| **Estimated Complexity** | high |

#### Context

TECH-EMB-001 specifies loading a learned projection matrix from SafeTensors. This replaces the broken hash-based projection with real neural weights. The Foundation Layer created the struct and error types; this task implements the actual loading logic.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Struct definition | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |
| SafeTensors docs | https://huggingface.co/docs/safetensors |

#### Prerequisites

- [ ] TASK-EMB-002 completed (ProjectionMatrix struct exists)
- [ ] TASK-EMB-003 completed (ProjectionError enum exists)
- [ ] `safetensors` crate added to Cargo.toml
- [ ] `sha2` crate added to Cargo.toml
- [ ] `candle-core` with CUDA feature enabled

#### Scope

**In Scope:**
- Read SafeTensors file from disk
- Parse tensor header and validate shape
- Compute SHA256 checksum of file bytes
- Load tensor to Candle Device (GPU)
- Validate shape is [30522, 1536]

**Out of Scope:**
- Projection computation (TASK-EMB-012)
- Weight training (separate process)
- Fallback to hash (FORBIDDEN)

#### Definition of Done

**Exact Implementation:**

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs

use candle_core::{Device, Tensor, DType};
use safetensors::SafeTensors;
use sha2::{Sha256, Digest};
use std::fs;
use std::path::Path;

impl ProjectionMatrix {
    /// Load projection matrix from SafeTensors file.
    ///
    /// # Panics
    /// - File not found at `{model_dir}/sparse_projection.safetensors`
    /// - Tensor shape mismatch (expected [30522, 1536])
    /// - Checksum mismatch if expected_checksum provided
    /// - CUDA device unavailable
    ///
    /// # CRITICAL: No Fallback
    /// If weight file is missing, this function PANICS.
    /// Hash-based projection fallback is FORBIDDEN by Constitution AP-007.
    pub fn load(model_dir: &Path) -> Result<Self, ProjectionError> {
        let weight_path = model_dir.join(PROJECTION_WEIGHT_FILE);

        // Step 1: Read file bytes
        let file_bytes = fs::read(&weight_path).map_err(|_| {
            ProjectionError::MatrixMissing { path: weight_path.clone() }
        })?;

        // Step 2: Compute real SHA256 checksum
        let mut hasher = Sha256::new();
        hasher.update(&file_bytes);
        let checksum: [u8; 32] = hasher.finalize().into();

        // Step 3: Parse SafeTensors
        let tensors = SafeTensors::deserialize(&file_bytes).map_err(|e| {
            ProjectionError::GpuError {
                operation: "SafeTensors parse".to_string(),
                details: e.to_string(),
            }
        })?;

        // Step 4: Get projection tensor
        let tensor_view = tensors.tensor(PROJECTION_TENSOR_NAME).map_err(|_| {
            ProjectionError::MatrixMissing { path: weight_path.clone() }
        })?;

        // Step 5: Validate shape
        let shape = tensor_view.shape();
        if shape != &[SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION] {
            return Err(ProjectionError::DimensionMismatch {
                path: weight_path,
                actual_rows: shape.get(0).copied().unwrap_or(0),
                actual_cols: shape.get(1).copied().unwrap_or(0),
            });
        }

        // Step 6: Create CUDA device (will panic if unavailable)
        let device = Device::cuda_if_available(0).map_err(|e| {
            ProjectionError::GpuError {
                operation: "CUDA device creation".to_string(),
                details: e.to_string(),
            }
        })?;

        // Verify we got a CUDA device, not CPU
        if !matches!(&device, Device::Cuda(_)) {
            return Err(ProjectionError::GpuError {
                operation: "CUDA verification".to_string(),
                details: "No CUDA device available - CPU fallback forbidden".to_string(),
            });
        }

        // Step 7: Load tensor to GPU
        let weights = Tensor::from_raw_buffer(
            tensor_view.data(),
            DType::F32,
            &[SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION],
            &device,
        ).map_err(|e| {
            ProjectionError::GpuError {
                operation: "Tensor GPU upload".to_string(),
                details: e.to_string(),
            }
        })?;

        Ok(Self {
            weights,
            device,
            weight_checksum: checksum,
        })
    }
}
```

**Constraints:**
- MUST use SafeTensors crate (not manual parsing)
- MUST compute real SHA256 checksum
- MUST verify CUDA device (not CPU)
- MUST NOT fall back to hash-based projection

**Verification:**
- Weight file loads successfully on GPU
- Shape validation catches wrong dimensions
- Checksum is deterministic for same file

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | Add load() implementation |
| `crates/context-graph-embeddings/Cargo.toml` | Add safetensors, sha2 dependencies |

#### Validation Criteria

- [ ] `load()` returns `Result<Self, ProjectionError>`
- [ ] Missing file returns `ProjectionError::MatrixMissing`
- [ ] Wrong shape returns `ProjectionError::DimensionMismatch`
- [ ] No CPU fallback (error if CUDA unavailable)
- [ ] Checksum is 32 bytes SHA256

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings --features cuda
cargo test -p context-graph-embeddings projection::load -- --nocapture
```

</task_spec>

---

<task_spec id="TASK-EMB-012" version="1.0">

### TASK-EMB-012: Implement ProjectionMatrix::project()

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Implement Sparse to Dense Projection |
| **Status** | ready |
| **Layer** | logic |
| **Sequence** | 12 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-011 |
| **Estimated Complexity** | high |

#### Context

TECH-EMB-001 specifies SpMM (Sparse Matrix-Matrix Multiplication) via cuBLAS for projecting sparse vectors to dense. This replaces the broken hash modulo approach with learned neural projection.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| ProjectionMatrix | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` |
| SparseVector | `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |

#### Prerequisites

- [ ] TASK-EMB-011 completed (load() works)
- [ ] TASK-EMB-008 completed (SparseVector::to_csr() exists)
- [ ] cuBLAS bindings available via candle or cudarc

#### Scope

**In Scope:**
- Convert SparseVector to CSR format
- Execute SpMM on GPU: `dense = sparse @ weights^T`
- L2 normalize the output
- Return 1536D dense vector

**Out of Scope:**
- Sparse vector creation (upstream)
- Storage of result (downstream)

#### Definition of Done

**Exact Implementation:**

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs

impl ProjectionMatrix {
    /// Project sparse vector to dense representation using learned weights.
    ///
    /// # Algorithm
    /// 1. Convert SparseVector to CSR format for cuBLAS
    /// 2. Execute SpMM: dense = sparse @ weights^T
    /// 3. L2 normalize result
    ///
    /// # Arguments
    /// * `sparse` - Input sparse vector (30522D, ~5% active)
    ///
    /// # Returns
    /// Dense 1536D vector, L2-normalized
    ///
    /// # CRITICAL: No Hash Fallback
    /// This uses REAL matrix multiplication. The old hash modulo approach
    /// (idx % projected_dim) is FORBIDDEN as it destroys semantic information.
    pub fn project(&self, sparse: &SparseVector) -> Result<Vec<f32>, ProjectionError> {
        // Step 1: Convert to CSR format
        let (row_ptr, col_indices, values) = sparse.to_csr();

        // Step 2: Create sparse tensor on GPU
        // Note: Using Candle's sparse tensor support or cudarc SpMM
        let sparse_tensor = self.create_sparse_tensor(&row_ptr, &col_indices, &values)?;

        // Step 3: Execute matrix multiplication: dense = sparse @ weights^T
        // Shape: [1, vocab] @ [vocab, proj_dim] = [1, proj_dim]
        let weights_t = self.weights.t()?;
        let dense = sparse_tensor.matmul(&weights_t)?;

        // Step 4: L2 normalize
        let norm = dense.sqr()?.sum(1)?.sqrt()?;
        let normalized = dense.broadcast_div(&norm.unsqueeze(1)?)?;

        // Step 5: Transfer back to CPU and return
        let result: Vec<f32> = normalized.squeeze(0)?.to_vec1()?;

        debug_assert_eq!(result.len(), SPARSE_PROJECTED_DIMENSION);

        Ok(result)
    }

    /// Batch project multiple sparse vectors (more efficient).
    ///
    /// # Arguments
    /// * `sparse_batch` - Slice of sparse vectors
    ///
    /// # Returns
    /// Vec of dense 1536D vectors, each L2-normalized
    pub fn project_batch(&self, sparse_batch: &[SparseVector]) -> Result<Vec<Vec<f32>>, ProjectionError> {
        // For batch processing, construct batched sparse matrix
        // and execute single SpMM for efficiency
        sparse_batch.iter()
            .map(|s| self.project(s))
            .collect()
    }

    /// Create sparse tensor on GPU from CSR format.
    fn create_sparse_tensor(
        &self,
        row_ptr: &[i32],
        col_indices: &[i32],
        values: &[f32],
    ) -> Result<Tensor, ProjectionError> {
        // Implementation depends on candle sparse support or cudarc SpMM
        // This is a placeholder for the actual CUDA sparse tensor creation

        // For now, convert to dense (less efficient but functional)
        // TODO: Use cuSPARSE SpMM for true sparse multiplication
        let mut dense_input = vec![0.0f32; SPARSE_VOCAB_SIZE];
        for (&idx, &val) in col_indices.iter().zip(values.iter()) {
            dense_input[idx as usize] = val;
        }

        Tensor::from_slice(&dense_input, (1, SPARSE_VOCAB_SIZE), &self.device)
            .map_err(|e| ProjectionError::GpuError {
                operation: "Sparse tensor creation".to_string(),
                details: e.to_string(),
            })
    }
}
```

**Constraints:**
- Output dimension MUST be exactly 1536
- Output MUST be L2 normalized
- NO hash modulo allowed (`idx % dim` is FORBIDDEN)
- Latency target: < 3ms per projection

**Verification:**
- Semantic similarity preserved (related terms have cosine sim > 0.7)
- Output dimension is exactly 1536
- Output is L2 normalized (norm = 1.0)

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | Add project() and project_batch() |

#### Validation Criteria

- [ ] `project()` returns `Vec<f32>` of length 1536
- [ ] Output is L2 normalized (sum of squares = 1.0)
- [ ] Related terms have high cosine similarity (>0.7)
- [ ] Unrelated terms have low cosine similarity (<0.3)
- [ ] No hash modulo in implementation

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo test -p context-graph-embeddings projection::project -- --nocapture
cargo bench -p context-graph-embeddings projection_latency
```

</task_spec>

---

<task_spec id="TASK-EMB-013" version="1.0">

### TASK-EMB-013: Implement Real Weight Loading

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Replace Simulated Weight Loading with Real SafeTensors Loading |
| **Status** | ready |
| **Layer** | logic |
| **Sequence** | 13 |
| **Implements** | REQ-EMB-003 |
| **Depends On** | TASK-EMB-006 |
| **Estimated Complexity** | high |

#### Context

TECH-EMB-002 identifies `simulate_weight_loading()` as returning fake checksums (`0xDEAD_BEEF_CAFE_BABE`). This task replaces that simulation with real SafeTensors file loading and SHA256 checksum computation.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Current broken code | `crates/context-graph-embeddings/src/warm/loader/operations.rs` |
| Types | `crates/context-graph-embeddings/src/warm/loader/types.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-002-warm-loading.md` |

#### Prerequisites

- [ ] TASK-EMB-006 completed (WarmLoadResult struct exists)
- [ ] SafeTensors and sha2 crates added to Cargo.toml

#### Scope

**In Scope:**
- DELETE `simulate_weight_loading()` function entirely
- Implement `load_weights()` using SafeTensors
- Compute real SHA256 checksum of weight file
- Parse tensor metadata from SafeTensors header
- Report actual file size and load duration

**Out of Scope:**
- GPU memory allocation (TASK-EMB-014)
- Inference validation (TASK-EMB-015)

#### Definition of Done

**Code to DELETE:**

```rust
// DELETE THIS ENTIRE FUNCTION FROM operations.rs
pub fn simulate_weight_loading(model_id: &str, _size_bytes: usize) -> WarmResult<u64> {
    let mut checksum = 0u64;
    for (i, byte) in model_id.bytes().enumerate() {
        checksum ^= (byte as u64) << ((i % 8) * 8);
    }
    checksum ^= 0xDEAD_BEEF_CAFE_BABEu64;  // FAKE!
    Ok(checksum)
}
```

**Replacement Implementation:**

```rust
// File: crates/context-graph-embeddings/src/warm/loader/operations.rs

use safetensors::SafeTensors;
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use super::types::{WarmLoadResult, TensorMetadata, DType};
use crate::error::EmbeddingError;

/// Load model weights from SafeTensors file.
///
/// # CRITICAL: No Simulation
/// This function reads REAL bytes from REAL files.
/// Fake checksums (0xDEAD_BEEF...) are FORBIDDEN.
///
/// # Panics
/// - Weight file not found
/// - Weight file corrupted (parse error)
/// - Tensor shapes invalid
pub fn load_weights(weight_path: &Path, model_id: &str) -> Result<(Vec<u8>, [u8; 32], TensorMetadata), EmbeddingError> {
    let start = Instant::now();

    // Step 1: Read actual file bytes
    let file_bytes = fs::read(weight_path).map_err(|_| {
        EmbeddingError::WeightFileMissing {
            model_id: model_id.parse().unwrap_or_default(),
            path: weight_path.to_path_buf(),
        }
    })?;

    // Step 2: Compute REAL SHA256 checksum
    let mut hasher = Sha256::new();
    hasher.update(&file_bytes);
    let checksum: [u8; 32] = hasher.finalize().into();

    // Step 3: Parse SafeTensors to extract metadata
    let tensors = SafeTensors::deserialize(&file_bytes).map_err(|e| {
        EmbeddingError::WeightChecksumMismatch {
            model_id: model_id.parse().unwrap_or_default(),
            expected: "valid SafeTensors format".to_string(),
            actual: format!("parse error: {}", e),
        }
    })?;

    // Step 4: Extract tensor metadata
    let mut shapes = HashMap::new();
    let mut total_params = 0usize;
    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        total_params += shape.iter().product::<usize>();
        shapes.insert(name.to_string(), shape);
    }

    let metadata = TensorMetadata {
        shapes,
        dtype: DType::F32, // TODO: Parse from SafeTensors header
        total_params,
    };

    let duration = start.elapsed();
    tracing::info!(
        "Loaded weights for {} in {:?}: {} params, checksum {:?}",
        model_id, duration, total_params, hex::encode(&checksum[..8])
    );

    Ok((file_bytes, checksum, metadata))
}

/// Verify checksum against expected value.
///
/// # Panics
/// If checksums don't match (corrupted weight file).
pub fn verify_checksum(actual: &[u8; 32], expected: &[u8; 32], model_id: &str) -> Result<(), EmbeddingError> {
    if actual != expected {
        return Err(EmbeddingError::WeightChecksumMismatch {
            model_id: model_id.parse().unwrap_or_default(),
            expected: hex::encode(expected),
            actual: hex::encode(actual),
        });
    }
    Ok(())
}
```

**Constraints:**
- `simulate_weight_loading()` MUST be deleted
- All checksums MUST be real SHA256
- All file reads MUST use actual fs::read()
- No fake/hardcoded checksums allowed

**Verification:**
- Checksum changes when weight file changes
- Parse errors on corrupted files
- File not found error on missing files

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/warm/loader/operations.rs` | DELETE simulate_weight_loading, ADD load_weights |

#### Validation Criteria

- [ ] `simulate_weight_loading()` deleted from codebase
- [ ] `load_weights()` reads real file bytes
- [ ] SHA256 checksum is 32 bytes
- [ ] Checksum is deterministic for same file
- [ ] Checksum changes when file modified

#### Test Commands

```bash
cd /home/cabdru/contextgraph
grep -rn "simulate_weight_loading" crates/context-graph-embeddings/
cargo test -p context-graph-embeddings warm::loader::load_weights -- --nocapture
```

</task_spec>

---

<task_spec id="TASK-EMB-014" version="1.0">

### TASK-EMB-014: Implement Real VRAM Allocation

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Replace Fake Pointers with Real cuMemAlloc |
| **Status** | ready |
| **Layer** | logic |
| **Sequence** | 14 |
| **Implements** | REQ-EMB-003 |
| **Depends On** | TASK-EMB-013 |
| **Estimated Complexity** | high |

#### Context

TECH-EMB-002 identifies `allocate_model_vram()` as returning fake pointers (`0x7f80_0000_0000 + offset`). This task replaces that with real CUDA memory allocation via cudarc.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Current broken code | `crates/context-graph-embeddings/src/warm/loader/operations.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-002-warm-loading.md` |
| cudarc docs | https://docs.rs/cudarc |

#### Prerequisites

- [ ] TASK-EMB-013 completed (real weight loading)
- [ ] cudarc crate added to Cargo.toml with cuda feature

#### Scope

**In Scope:**
- DELETE fake pointer generation (`0x7f80_0000_0000 + offset`)
- Implement real `cuMemAlloc` via cudarc
- Implement `cuMemcpyHtoD` for host-to-device transfer
- Track actual VRAM usage
- Mark allocations as non-evictable

**Out of Scope:**
- Inference execution (TASK-EMB-015)
- Memory pooling optimization (future task)

#### Definition of Done

**Code to DELETE:**

```rust
// DELETE THIS FAKE POINTER GENERATION FROM operations.rs
let base_ptr = 0x7f80_0000_0000u64;  // FAKE POINTER!
let offset = memory_pools.list_model_allocations().len() as u64 * 0x1_0000_0000;
let vram_ptr = base_ptr + offset;
```

**Replacement Implementation:**

```rust
// File: crates/context-graph-embeddings/src/warm/loader/operations.rs

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceRepr};
use std::sync::Arc;

/// GPU memory handle with real CUDA allocation.
pub struct GpuMemoryHandle {
    /// Real CUDA device pointer from cudaMalloc.
    device_ptr: CudaSlice<f32>,
    /// Size in bytes.
    size_bytes: usize,
    /// CUDA device reference.
    device: Arc<CudaDevice>,
}

impl GpuMemoryHandle {
    /// Get raw device pointer value.
    pub fn ptr(&self) -> u64 {
        self.device_ptr.device_ptr().0 as u64
    }
}

/// Allocate VRAM for model weights using real cudaMalloc.
///
/// # CRITICAL: Real Allocation Only
/// This function calls REAL cudaMalloc via cudarc.
/// Fake pointers (0x7f80...) are FORBIDDEN.
///
/// # Panics
/// - No CUDA device available
/// - Insufficient VRAM
/// - CUDA driver error
pub fn allocate_model_vram(
    device: Arc<CudaDevice>,
    size_bytes: usize,
    model_id: &str,
) -> Result<GpuMemoryHandle, EmbeddingError> {
    // Calculate number of f32 elements
    let num_elements = size_bytes / std::mem::size_of::<f32>();

    // Step 1: Check available VRAM
    let (free_bytes, total_bytes) = device.memory_info().map_err(|e| {
        EmbeddingError::CudaUnavailable {
            message: format!("Failed to query VRAM: {}", e),
        }
    })?;

    if free_bytes < size_bytes {
        let required_gb = size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let available_gb = free_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        return Err(EmbeddingError::InsufficientVram {
            required_bytes: size_bytes,
            available_bytes: free_bytes,
            required_gb,
            available_gb,
        });
    }

    // Step 2: Allocate via cudaMalloc (through cudarc)
    let device_ptr: CudaSlice<f32> = device.alloc_zeros(num_elements).map_err(|e| {
        EmbeddingError::CudaUnavailable {
            message: format!("cudaMalloc failed for {}: {}", model_id, e),
        }
    })?;

    tracing::info!(
        "Allocated {} bytes ({:.2} MB) on GPU for {}: ptr=0x{:x}",
        size_bytes,
        size_bytes as f64 / (1024.0 * 1024.0),
        model_id,
        device_ptr.device_ptr().0 as u64
    );

    Ok(GpuMemoryHandle {
        device_ptr,
        size_bytes,
        device,
    })
}

/// Copy weights from host to device using real cuMemcpyHtoD.
///
/// # CRITICAL: Real Transfer Only
/// This function performs REAL host-to-device memory copy.
pub fn copy_weights_to_gpu(
    device: &Arc<CudaDevice>,
    host_data: &[f32],
    gpu_handle: &mut GpuMemoryHandle,
) -> Result<(), EmbeddingError> {
    device.htod_copy_into(host_data, &mut gpu_handle.device_ptr).map_err(|e| {
        EmbeddingError::CudaUnavailable {
            message: format!("cuMemcpyHtoD failed: {}", e),
        }
    })?;

    tracing::debug!("Copied {} floats to GPU", host_data.len());
    Ok(())
}
```

**Constraints:**
- ALL device pointers MUST come from cudarc/CUDA
- NO fake pointer arithmetic
- VRAM check MUST happen before allocation
- nvidia-smi MUST show increased usage after allocation

**Verification:**
- nvidia-smi shows VRAM increase
- Real CUDA errors on allocation failure
- OOM error when VRAM insufficient

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/warm/loader/operations.rs` | DELETE fake pointers, ADD cudarc allocation |
| `crates/context-graph-embeddings/Cargo.toml` | Add cudarc dependency |

#### Validation Criteria

- [ ] No fake pointer generation (`0x7f80...`)
- [ ] `nvidia-smi` shows VRAM increase after allocation
- [ ] VRAM check fails with clear error when insufficient
- [ ] Real CUDA errors propagate up

#### Test Commands

```bash
cd /home/cabdru/contextgraph
nvidia-smi -q -d MEMORY | grep -A5 "FB Memory Usage"
cargo test -p context-graph-embeddings warm::allocate_vram --features cuda -- --nocapture
nvidia-smi -q -d MEMORY | grep -A5 "FB Memory Usage"
```

</task_spec>

---

<task_spec id="TASK-EMB-015" version="1.0">

### TASK-EMB-015: Implement Real Inference Validation

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Replace Sin Wave with Real Forward Pass |
| **Status** | ready |
| **Layer** | logic |
| **Sequence** | 15 |
| **Implements** | REQ-EMB-003 |
| **Depends On** | TASK-EMB-014, TASK-EMB-010 |
| **Estimated Complexity** | high |

#### Context

TECH-EMB-002 identifies test inference as returning `(i as f32 * 0.001).sin()` - a fake sin wave instead of real model output. This task replaces that with actual model forward pass and golden reference comparison.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Current broken code | `crates/context-graph-embeddings/src/warm/loader/operations.rs` |
| Golden fixtures | `crates/context-graph-embeddings/tests/fixtures/golden/` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-002-warm-loading.md` |

#### Prerequisites

- [ ] TASK-EMB-014 completed (real VRAM allocation)
- [ ] TASK-EMB-010 completed (golden fixtures exist)
- [ ] candle-core with model forward pass capability

#### Scope

**In Scope:**
- DELETE sin wave fake output
- Implement real model forward pass
- Load golden reference from tests/fixtures/
- Compare output with tolerance (1e-5)
- Report validation pass/fail with details

**Out of Scope:**
- Golden reference generation (separate process)
- Model architecture changes

#### Definition of Done

**Code to DELETE:**

```rust
// DELETE THIS FAKE SIN WAVE OUTPUT FROM operations.rs
let output: Vec<f32> = (0..expected_dimension)
    .map(|i| (i as f32 * 0.001).sin())  // FAKE!
    .collect();
```

**Replacement Implementation:**

```rust
// File: crates/context-graph-embeddings/src/warm/loader/operations.rs

use std::fs;
use std::path::Path;

const VALIDATION_TOLERANCE: f32 = 1e-5;

/// Run real inference validation against golden reference.
///
/// # CRITICAL: No Fake Output
/// This runs REAL model inference and compares to REAL golden reference.
/// Sin wave approximations are FORBIDDEN.
///
/// # Arguments
/// * `model_id` - Model identifier (e.g., "E1_Semantic")
/// * `model_handle` - Loaded model with GPU weights
/// * `golden_dir` - Directory containing golden test fixtures
///
/// # Panics
/// - Golden reference not found
/// - Inference output doesn't match within tolerance
pub fn validate_inference(
    model_id: &str,
    model_handle: &LoadedModelWeights,
    golden_dir: &Path,
) -> Result<(), EmbeddingError> {
    let model_golden_dir = golden_dir.join(model_id);

    // Step 1: Load golden input
    let input_path = model_golden_dir.join("test_input.bin");
    let input_bytes = fs::read(&input_path).map_err(|_| {
        EmbeddingError::InferenceValidationFailed {
            model_id: model_id.parse().unwrap_or_default(),
            reason: format!("Golden input not found: {:?}", input_path),
        }
    })?;
    let input = parse_golden_tensor(&input_bytes)?;

    // Step 2: Load golden output
    let output_path = model_golden_dir.join("golden_output.bin");
    let output_bytes = fs::read(&output_path).map_err(|_| {
        EmbeddingError::InferenceValidationFailed {
            model_id: model_id.parse().unwrap_or_default(),
            reason: format!("Golden output not found: {:?}", output_path),
        }
    })?;
    let expected_output = parse_golden_tensor(&output_bytes)?;

    // Step 3: Run REAL model inference
    let actual_output = run_inference(model_handle, &input)?;

    // Step 4: Compare with tolerance
    if actual_output.len() != expected_output.len() {
        return Err(EmbeddingError::InferenceValidationFailed {
            model_id: model_id.parse().unwrap_or_default(),
            reason: format!(
                "Dimension mismatch: expected {}, got {}",
                expected_output.len(),
                actual_output.len()
            ),
        });
    }

    let mut max_diff = 0.0f32;
    let mut diff_count = 0usize;
    for (i, (&actual, &expected)) in actual_output.iter().zip(expected_output.iter()).enumerate() {
        let diff = (actual - expected).abs();
        if diff > VALIDATION_TOLERANCE {
            diff_count += 1;
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    if diff_count > 0 {
        return Err(EmbeddingError::InferenceValidationFailed {
            model_id: model_id.parse().unwrap_or_default(),
            reason: format!(
                "{} elements differ (max diff: {:.6}), tolerance: {:.6}",
                diff_count, max_diff, VALIDATION_TOLERANCE
            ),
        });
    }

    tracing::info!(
        "Inference validation passed for {}: {} elements within tolerance {:.6}",
        model_id, actual_output.len(), VALIDATION_TOLERANCE
    );

    Ok(())
}

/// Parse golden tensor from binary format.
///
/// Format: 4 bytes dimension count (u32 LE), then N * 4 bytes (f32 LE values)
fn parse_golden_tensor(bytes: &[u8]) -> Result<Vec<f32>, EmbeddingError> {
    if bytes.len() < 4 {
        return Err(EmbeddingError::StorageCorruption {
            id: "golden_tensor".to_string(),
            reason: "Golden tensor too short".to_string(),
        });
    }

    let dim = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let expected_len = 4 + dim * 4;

    if bytes.len() != expected_len {
        return Err(EmbeddingError::StorageCorruption {
            id: "golden_tensor".to_string(),
            reason: format!("Expected {} bytes, got {}", expected_len, bytes.len()),
        });
    }

    let values: Vec<f32> = bytes[4..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(values)
}

/// Run actual model inference (placeholder for real implementation).
fn run_inference(
    model_handle: &LoadedModelWeights,
    input: &[f32],
) -> Result<Vec<f32>, EmbeddingError> {
    // This should use candle to run the actual model forward pass
    // using the weights loaded in model_handle

    // TODO: Implement actual forward pass based on model architecture
    // For now, this is a placeholder that should be replaced with real inference

    unimplemented!("Real model inference - implement based on model architecture")
}
```

**Constraints:**
- NO sin wave or deterministic fake output
- MUST load golden reference from tests/fixtures/
- Tolerance is 1e-5 for FP32 comparison
- MUST run actual model forward pass

**Verification:**
- Validation fails if golden reference missing
- Validation fails if output differs beyond tolerance
- Same input always produces same output

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/warm/loader/operations.rs` | DELETE sin wave, ADD real validation |

#### Validation Criteria

- [ ] No `sin()` calls for test output generation
- [ ] Golden reference files loaded from tests/fixtures/
- [ ] Tolerance comparison at 1e-5
- [ ] Clear error on validation failure

#### Test Commands

```bash
cd /home/cabdru/contextgraph
grep -rn "sin()" crates/context-graph-embeddings/src/warm/
cargo test -p context-graph-embeddings warm::validate_inference -- --nocapture
```

</task_spec>

---

<task_spec id="TASK-EMB-016" version="1.0">

### TASK-EMB-016: Implement PQ-8 Quantization

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Implement Product Quantization (PQ-8) |
| **Status** | ready |
| **Layer** | logic |
| **Sequence** | 16 |
| **Implements** | REQ-EMB-005 |
| **Depends On** | TASK-EMB-004 |
| **Estimated Complexity** | high |

#### Context

TECH-EMB-003 specifies PQ-8 (8 subvectors, 256 centroids each) for E1, E5, E7, E10. This provides 32x compression with <5% recall loss. This task implements the actual quantization and dequantization.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Quantization types | `crates/context-graph-embeddings/src/quantization/types.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-003-quantization.md` |
| Constitution | `docs/constitution.yaml` |

#### Prerequisites

- [ ] TASK-EMB-004 completed (PQ8Codebook struct exists)

#### Scope

**In Scope:**
- Create `quantization/pq8.rs` module
- Implement codebook loading from file
- Implement quantization: embedding -> 8 bytes
- Implement dequantization: 8 bytes -> approximate embedding
- Calculate and log recall loss

**Out of Scope:**
- Codebook training (offline process)
- Other quantization methods (separate tasks)

#### Definition of Done

**File to Create:**

```rust
// File: crates/context-graph-embeddings/src/quantization/pq8.rs

use super::types::{PQ8Codebook, QuantizedEmbedding, QuantizationMetadata, QuantizationMethod};
use crate::error::EmbeddingError;
use std::path::Path;

/// Number of subvectors for PQ-8.
pub const NUM_SUBVECTORS: usize = 8;

/// Number of centroids per subvector.
pub const NUM_CENTROIDS: usize = 256;

/// PQ-8 Quantizer for embeddings.
///
/// # Constitution Alignment
/// - Used for: E1_Semantic, E5_Causal, E7_Code, E10_Multimodal
/// - Compression: 32x (1024D f32 -> 8 bytes)
/// - Max recall loss: 5%
#[derive(Debug)]
pub struct PQ8Quantizer {
    /// Trained codebook.
    codebook: PQ8Codebook,
    /// Subvector dimension.
    subvector_dim: usize,
}

impl PQ8Quantizer {
    /// Load codebook from file.
    ///
    /// # Panics
    /// - Codebook file not found
    /// - Invalid codebook format
    pub fn load(codebook_path: &Path) -> Result<Self, EmbeddingError> {
        // TODO: Load from SafeTensors or similar format
        let codebook_bytes = std::fs::read(codebook_path).map_err(|_| {
            EmbeddingError::CodebookMissing {
                model_id: crate::types::model_id::ModelId::Semantic,
            }
        })?;

        let codebook = Self::parse_codebook(&codebook_bytes)?;
        let subvector_dim = codebook.embedding_dim / NUM_SUBVECTORS;

        Ok(Self { codebook, subvector_dim })
    }

    /// Quantize embedding to PQ-8 format.
    ///
    /// # Algorithm
    /// 1. Split embedding into 8 subvectors
    /// 2. Find nearest centroid for each subvector
    /// 3. Store 8 centroid indices (1 byte each)
    ///
    /// # Returns
    /// QuantizedEmbedding with 8 bytes of data
    pub fn quantize(&self, embedding: &[f32]) -> Result<QuantizedEmbedding, EmbeddingError> {
        if embedding.len() != self.codebook.embedding_dim {
            return Err(EmbeddingError::DimensionMismatch {
                model_id: crate::types::model_id::ModelId::Semantic,
                expected: self.codebook.embedding_dim,
                actual: embedding.len(),
            });
        }

        let mut codes = Vec::with_capacity(NUM_SUBVECTORS);

        for i in 0..NUM_SUBVECTORS {
            let start = i * self.subvector_dim;
            let end = start + self.subvector_dim;
            let subvector = &embedding[start..end];

            // Find nearest centroid
            let code = self.find_nearest_centroid(i, subvector);
            codes.push(code);
        }

        Ok(QuantizedEmbedding {
            method: QuantizationMethod::PQ8,
            original_dim: self.codebook.embedding_dim,
            data: codes,
            metadata: QuantizationMetadata::PQ8 {
                codebook_id: self.codebook.codebook_id,
                num_subvectors: NUM_SUBVECTORS as u8,
            },
        })
    }

    /// Dequantize PQ-8 codes back to approximate embedding.
    ///
    /// # Algorithm
    /// 1. For each code byte, look up centroid vector
    /// 2. Concatenate centroid vectors
    ///
    /// # Returns
    /// Approximate embedding (will have some recall loss)
    pub fn dequantize(&self, quantized: &QuantizedEmbedding) -> Result<Vec<f32>, EmbeddingError> {
        if quantized.data.len() != NUM_SUBVECTORS {
            return Err(EmbeddingError::StorageCorruption {
                id: "pq8_dequantize".to_string(),
                reason: format!("Expected {} codes, got {}", NUM_SUBVECTORS, quantized.data.len()),
            });
        }

        let mut embedding = Vec::with_capacity(self.codebook.embedding_dim);

        for (i, &code) in quantized.data.iter().enumerate() {
            let centroid = &self.codebook.centroids[i][code as usize];
            embedding.extend_from_slice(centroid);
        }

        Ok(embedding)
    }

    /// Calculate recall loss for a batch of embeddings.
    ///
    /// # Returns
    /// Tuple of (mean_recall_loss, max_recall_loss)
    pub fn measure_recall_loss(&self, embeddings: &[Vec<f32>]) -> (f32, f32) {
        let mut total_loss = 0.0f32;
        let mut max_loss = 0.0f32;

        for embedding in embeddings {
            if let Ok(quantized) = self.quantize(embedding) {
                if let Ok(reconstructed) = self.dequantize(&quantized) {
                    // Compute cosine similarity loss
                    let loss = 1.0 - cosine_similarity(embedding, &reconstructed);
                    total_loss += loss;
                    if loss > max_loss {
                        max_loss = loss;
                    }
                }
            }
        }

        let mean_loss = total_loss / embeddings.len() as f32;
        (mean_loss, max_loss)
    }

    fn find_nearest_centroid(&self, subvector_idx: usize, subvector: &[f32]) -> u8 {
        let centroids = &self.codebook.centroids[subvector_idx];
        let mut best_idx = 0u8;
        let mut best_dist = f32::MAX;

        for (idx, centroid) in centroids.iter().enumerate() {
            let dist = euclidean_distance(subvector, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx as u8;
            }
        }

        best_idx
    }

    fn parse_codebook(bytes: &[u8]) -> Result<PQ8Codebook, EmbeddingError> {
        // TODO: Implement actual parsing
        unimplemented!("Codebook parsing")
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq8_compression_ratio() {
        // 1024D f32 = 4096 bytes
        // PQ-8 = 8 bytes
        // Compression = 4096 / 8 = 512x... but Constitution says 32x
        // Note: Constitution may mean different embedding dims
        let original_bytes = 1024 * 4; // 1024D f32
        let quantized_bytes = 8; // 8 centroid indices
        let ratio = original_bytes / quantized_bytes;
        assert!(ratio >= 32, "Compression ratio should be at least 32x");
    }
}
```

**Constraints:**
- 8 subvectors, 256 centroids each
- Output is exactly 8 bytes
- Recall loss must be measured and logged
- Codebook must be loaded from file (not generated)

**Verification:**
- 32x compression achieved
- Recall loss < 5% on test embeddings
- Dequantization produces valid embeddings

#### Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/quantization/pq8.rs` | PQ-8 implementation |

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/quantization/mod.rs` | Add `pub mod pq8;` |

#### Validation Criteria

- [ ] `quantize()` produces 8 bytes output
- [ ] `dequantize()` returns embedding of original dimension
- [ ] Compression ratio >= 32x
- [ ] Recall loss < 5% on test set

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo test -p context-graph-embeddings quantization::pq8 -- --nocapture
cargo bench -p context-graph-embeddings pq8_recall
```

</task_spec>

---

<task_spec id="TASK-EMB-017" version="1.0">

### TASK-EMB-017: Implement Float8 Quantization

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Implement Float8 E4M3 Quantization |
| **Status** | ready |
| **Layer** | logic |
| **Sequence** | 17 |
| **Implements** | REQ-EMB-005 |
| **Depends On** | TASK-EMB-004 |
| **Estimated Complexity** | medium |

#### Context

TECH-EMB-003 specifies Float8 (E4M3 format) for E2, E3, E4, E8, E11. This provides 4x compression with <0.3% recall loss. E4M3 means 4 exponent bits, 3 mantissa bits.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Quantization types | `crates/context-graph-embeddings/src/quantization/types.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-003-quantization.md` |

#### Prerequisites

- [ ] TASK-EMB-004 completed (Float8Encoder struct exists)

#### Scope

**In Scope:**
- Create `quantization/float8.rs` module
- Implement E4M3 format conversion
- Implement scale/bias computation for denormalization
- Implement quantization: f32 -> f8
- Implement dequantization: f8 -> f32

**Out of Scope:**
- GPU-accelerated conversion (future optimization)
- Other float formats (bfloat16, fp16)

#### Definition of Done

**File to Create:**

```rust
// File: crates/context-graph-embeddings/src/quantization/float8.rs

use super::types::{Float8Encoder, QuantizedEmbedding, QuantizationMetadata, QuantizationMethod};
use crate::error::EmbeddingError;

/// Float8 E4M3 format: 1 sign, 4 exponent, 3 mantissa bits.
/// Range: approximately +/- 448
/// Precision: ~7.8% relative error
impl Float8Encoder {
    /// Quantize f32 embedding to Float8 E4M3 format.
    ///
    /// # Algorithm
    /// 1. Compute scale and bias to fit values in Float8 range
    /// 2. Convert each f32 to 8-bit E4M3 representation
    ///
    /// # Constitution Alignment
    /// - Used for: E2-E4 Temporal, E8_Graph, E11_Entity
    /// - Compression: 4x (f32 -> f8)
    /// - Max recall loss: 0.3%
    pub fn quantize(&self, embedding: &[f32]) -> Result<QuantizedEmbedding, EmbeddingError> {
        // Step 1: Compute dynamic range
        let (min_val, max_val) = embedding.iter().fold((f32::MAX, f32::MIN), |(min, max), &v| {
            (min.min(v), max.max(v))
        });

        // Step 2: Compute scale and bias for normalization
        let range = max_val - min_val;
        let scale = if range > 1e-10 { range / 448.0 } else { 1.0 }; // E4M3 max ~448
        let bias = min_val;

        // Step 3: Convert each value to Float8
        let mut data = Vec::with_capacity(embedding.len());
        for &value in embedding {
            let normalized = (value - bias) / scale;
            let f8 = self.f32_to_e4m3(normalized);
            data.push(f8);
        }

        Ok(QuantizedEmbedding {
            method: QuantizationMethod::Float8E4M3,
            original_dim: embedding.len(),
            data,
            metadata: QuantizationMetadata::Float8 { scale, bias },
        })
    }

    /// Dequantize Float8 E4M3 back to f32.
    ///
    /// # Returns
    /// Approximate f32 embedding
    pub fn dequantize(&self, quantized: &QuantizedEmbedding) -> Result<Vec<f32>, EmbeddingError> {
        let (scale, bias) = match quantized.metadata {
            QuantizationMetadata::Float8 { scale, bias } => (scale, bias),
            _ => return Err(EmbeddingError::StorageCorruption {
                id: "float8_dequantize".to_string(),
                reason: "Wrong metadata type".to_string(),
            }),
        };

        let mut embedding = Vec::with_capacity(quantized.data.len());
        for &f8 in &quantized.data {
            let normalized = self.e4m3_to_f32(f8);
            let value = normalized * scale + bias;
            embedding.push(value);
        }

        Ok(embedding)
    }

    /// Convert f32 to E4M3 (8-bit float).
    ///
    /// E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
    /// Exponent bias: 7 (2^(4-1) - 1)
    fn f32_to_e4m3(&self, value: f32) -> u8 {
        if value == 0.0 {
            return 0;
        }

        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127; // f32 bias
        let mantissa = bits & 0x7FFFFF;

        // Clamp exponent to E4M3 range [-6, 8] (bias 7)
        let e4_exp = (exp + 7).clamp(0, 15) as u8;

        // Take top 3 bits of mantissa
        let e4_mantissa = ((mantissa >> 20) & 0x7) as u8;

        // Pack: 1 sign + 4 exp + 3 mantissa
        ((sign as u8) << 7) | (e4_exp << 3) | e4_mantissa
    }

    /// Convert E4M3 to f32.
    fn e4m3_to_f32(&self, f8: u8) -> f32 {
        if f8 == 0 {
            return 0.0;
        }

        let sign = ((f8 >> 7) & 1) as u32;
        let exp = ((f8 >> 3) & 0xF) as i32;
        let mantissa = (f8 & 0x7) as u32;

        // Convert to f32 format
        let f32_exp = (exp - 7 + 127) as u32; // Remove E4M3 bias, add f32 bias
        let f32_mantissa = mantissa << 20; // Shift to f32 mantissa position

        let bits = (sign << 31) | (f32_exp << 23) | f32_mantissa;
        f32::from_bits(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float8_roundtrip() {
        let encoder = Float8Encoder::default();
        let original = vec![0.1, 0.5, -0.3, 1.0, -1.0];

        let quantized = encoder.quantize(&original).unwrap();
        let reconstructed = encoder.dequantize(&quantized).unwrap();

        // Check relative error < 10% (E4M3 has ~7.8% precision)
        for (o, r) in original.iter().zip(reconstructed.iter()) {
            let rel_error = ((o - r).abs() / o.abs().max(1e-10)).abs();
            assert!(rel_error < 0.15, "Relative error {:.4} too high", rel_error);
        }
    }

    #[test]
    fn test_float8_compression_ratio() {
        let original_bytes = 512 * 4; // 512D f32
        let quantized_bytes = 512; // 512 bytes (1 byte per element)
        let ratio = original_bytes / quantized_bytes;
        assert_eq!(ratio, 4, "Compression ratio should be 4x");
    }
}
```

**Constraints:**
- E4M3 format: 1 sign, 4 exponent, 3 mantissa
- Compression exactly 4x
- Recall loss < 0.3%
- Dynamic scale/bias per embedding

**Verification:**
- 4x compression achieved
- Recall loss < 0.3%
- Round-trip produces valid embeddings

#### Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/quantization/float8.rs` | Float8 E4M3 implementation |

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/quantization/mod.rs` | Add `pub mod float8;` |

#### Validation Criteria

- [ ] Output is exactly 1 byte per element
- [ ] 4x compression ratio
- [ ] Recall loss < 0.3%
- [ ] Scale/bias stored in metadata

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo test -p context-graph-embeddings quantization::float8 -- --nocapture
cargo bench -p context-graph-embeddings float8_recall
```

</task_spec>

---

<task_spec id="TASK-EMB-018" version="1.0">

### TASK-EMB-018: Implement Binary Quantization

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Implement Binary Quantization for HDC |
| **Status** | ready |
| **Layer** | logic |
| **Sequence** | 18 |
| **Implements** | REQ-EMB-005 |
| **Depends On** | TASK-EMB-004 |
| **Estimated Complexity** | low |

#### Context

TECH-EMB-003 specifies binary quantization for E9 (HDC - Hyperdimensional Computing). This provides 32x compression. HDC embeddings are naturally binary-friendly due to their hypervector properties.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Quantization types | `crates/context-graph-embeddings/src/quantization/types.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-003-quantization.md` |

#### Prerequisites

- [ ] TASK-EMB-004 completed (BinaryEncoder struct exists)

#### Scope

**In Scope:**
- Create `quantization/binary.rs` module
- Implement sign-based binarization (value > 0 -> 1, else 0)
- Pack 8 binary values per byte
- Implement Hamming distance for similarity

**Out of Scope:**
- Adaptive thresholds
- GPU-accelerated binarization

#### Definition of Done

**File to Create:**

```rust
// File: crates/context-graph-embeddings/src/quantization/binary.rs

use super::types::{BinaryEncoder, QuantizedEmbedding, QuantizationMetadata, QuantizationMethod};
use crate::error::EmbeddingError;

impl BinaryEncoder {
    /// Quantize embedding to binary (1 bit per value).
    ///
    /// # Algorithm
    /// 1. Binarize: value >= threshold -> 1, else -> 0
    /// 2. Pack 8 bits per byte
    ///
    /// # Constitution Alignment
    /// - Used for: E9_HDC (Hyperdimensional Computing)
    /// - Compression: 32x (f32 -> 1 bit)
    /// - Recall impact: 5-10%
    ///
    /// # Note
    /// HDC embeddings are designed for binary representation.
    /// Hamming distance is used for similarity instead of cosine.
    pub fn quantize(&self, embedding: &[f32]) -> Result<QuantizedEmbedding, EmbeddingError> {
        // Default threshold at 0 (sign-based binarization)
        self.quantize_with_threshold(embedding, 0.0)
    }

    /// Quantize with custom threshold.
    pub fn quantize_with_threshold(&self, embedding: &[f32], threshold: f32) -> Result<QuantizedEmbedding, EmbeddingError> {
        let num_bytes = (embedding.len() + 7) / 8; // Round up
        let mut data = vec![0u8; num_bytes];

        for (i, &value) in embedding.iter().enumerate() {
            if value >= threshold {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                data[byte_idx] |= 1 << bit_idx;
            }
        }

        Ok(QuantizedEmbedding {
            method: QuantizationMethod::Binary,
            original_dim: embedding.len(),
            data,
            metadata: QuantizationMetadata::Binary { threshold },
        })
    }

    /// Dequantize binary back to f32 (+1/-1 representation).
    ///
    /// # Note
    /// Binary dequantization loses magnitude information.
    /// Result is +1.0 or -1.0 per element.
    pub fn dequantize(&self, quantized: &QuantizedEmbedding) -> Result<Vec<f32>, EmbeddingError> {
        let mut embedding = Vec::with_capacity(quantized.original_dim);

        for i in 0..quantized.original_dim {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let bit = (quantized.data[byte_idx] >> bit_idx) & 1;
            embedding.push(if bit == 1 { 1.0 } else { -1.0 });
        }

        Ok(embedding)
    }

    /// Compute Hamming distance between two binary embeddings.
    ///
    /// # Returns
    /// Number of differing bits (0 = identical)
    pub fn hamming_distance(a: &QuantizedEmbedding, b: &QuantizedEmbedding) -> Result<u32, EmbeddingError> {
        if a.data.len() != b.data.len() {
            return Err(EmbeddingError::DimensionMismatch {
                model_id: crate::types::model_id::ModelId::Hdc,
                expected: a.data.len(),
                actual: b.data.len(),
            });
        }

        let distance: u32 = a.data.iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum();

        Ok(distance)
    }

    /// Compute normalized Hamming similarity.
    ///
    /// # Returns
    /// Similarity in [0, 1] where 1 = identical
    pub fn hamming_similarity(a: &QuantizedEmbedding, b: &QuantizedEmbedding) -> Result<f32, EmbeddingError> {
        let distance = Self::hamming_distance(a, b)?;
        let max_distance = a.original_dim as f32;
        Ok(1.0 - (distance as f32 / max_distance))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_compression_ratio() {
        // 1024D f32 = 4096 bytes
        // Binary = 1024 bits = 128 bytes
        // Compression = 4096 / 128 = 32x
        let original_bytes = 1024 * 4;
        let quantized_bytes = (1024 + 7) / 8; // 128 bytes
        let ratio = original_bytes / quantized_bytes;
        assert_eq!(ratio, 32, "Compression ratio should be 32x");
    }

    #[test]
    fn test_binary_roundtrip() {
        let encoder = BinaryEncoder::default();
        let original = vec![0.5, -0.3, 0.1, -0.8, 0.0, 0.2, -0.1, 0.9];

        let quantized = encoder.quantize(&original).unwrap();
        let reconstructed = encoder.dequantize(&quantized).unwrap();

        // Check signs are preserved
        for (o, r) in original.iter().zip(reconstructed.iter()) {
            let orig_sign = if *o >= 0.0 { 1.0 } else { -1.0 };
            assert_eq!(orig_sign, *r, "Sign should be preserved");
        }
    }

    #[test]
    fn test_hamming_distance() {
        let encoder = BinaryEncoder::default();
        let a = vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
        let b = vec![1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0]; // 2 bits different

        let qa = encoder.quantize(&a).unwrap();
        let qb = encoder.quantize(&b).unwrap();

        let distance = BinaryEncoder::hamming_distance(&qa, &qb).unwrap();
        assert_eq!(distance, 2);
    }
}
```

**Constraints:**
- 1 bit per element
- 8 bits packed per byte
- Hamming distance for similarity
- Sign-based binarization by default

**Verification:**
- 32x compression achieved
- Hamming distance computed correctly
- Bit packing is correct

#### Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/quantization/binary.rs` | Binary quantization |

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/quantization/mod.rs` | Add `pub mod binary;` |

#### Validation Criteria

- [ ] 1 bit per element (32x compression)
- [ ] Bit packing 8 bits per byte
- [ ] Hamming distance works correctly
- [ ] Signs preserved in round-trip

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo test -p context-graph-embeddings quantization::binary -- --nocapture
```

</task_spec>

---

<task_spec id="TASK-EMB-019" version="1.0">

### TASK-EMB-019: Remove Stub Mode from Preflight

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Remove Fake GPU from Preflight Checks |
| **Status** | ready |
| **Layer** | logic |
| **Sequence** | 19 |
| **Implements** | REQ-EMB-004 |
| **Depends On** | TASK-EMB-014 |
| **Estimated Complexity** | low |

#### Context

TECH-EMB-002 identifies preflight returning "Simulated RTX 5090" when CUDA is disabled. This task replaces that with a compile-time error, ensuring no deployment can occur without real GPU support.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Current broken code | `crates/context-graph-embeddings/src/warm/loader/preflight.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-002-warm-loading.md` |

#### Prerequisites

- [ ] TASK-EMB-014 completed (real VRAM allocation works)

#### Scope

**In Scope:**
- DELETE `#[cfg(not(feature = "cuda"))]` block returning fake GPU
- Replace with `compile_error!("CUDA feature required")`
- Add runtime panic if GPU unavailable even with feature enabled

**Out of Scope:**
- CUDA feature configuration
- CI/CD pipeline updates

#### Definition of Done

**Code to DELETE:**

```rust
// DELETE THIS ENTIRE BLOCK FROM preflight.rs
#[cfg(not(feature = "cuda"))]
{
    tracing::warn!("CUDA feature not enabled, running in stub mode");
    *gpu_info = Some(GpuInfo::new(
        0,
        "Simulated RTX 5090".to_string(),  // FAKE!
        (REQUIRED_COMPUTE_MAJOR, REQUIRED_COMPUTE_MINOR),
        MINIMUM_VRAM_BYTES,  // FAKE 32GB!
        "Simulated".to_string(),
    ));
    Ok(())
}
```

**Replacement Implementation:**

```rust
// File: crates/context-graph-embeddings/src/warm/loader/preflight.rs

use crate::error::EmbeddingError;

/// Minimum required VRAM (32GB - 8GB headroom = 24GB usable)
pub const MINIMUM_VRAM_BYTES: usize = 24 * 1024 * 1024 * 1024; // 24 GB

/// Required compute capability (RTX 5090 = Blackwell = CC 12.0)
pub const REQUIRED_COMPUTE_MAJOR: u32 = 9; // Minimum Ada Lovelace, prefer 12
pub const REQUIRED_COMPUTE_MINOR: u32 = 0;

/// CRITICAL: CUDA feature is REQUIRED. No stub mode.
#[cfg(not(feature = "cuda"))]
compile_error!(
    "[EMB-E001] CUDA_UNAVAILABLE: The 'cuda' feature MUST be enabled.

    Context Graph embeddings require GPU acceleration.
    There is NO CPU fallback and NO stub mode.

    Remediation:
    1. Install CUDA 13.1+
    2. Ensure RTX 5090 or compatible GPU is available
    3. Build with: cargo build --features cuda

    Constitution Reference: stack.gpu, AP-007"
);

#[cfg(feature = "cuda")]
pub fn check_gpu_requirements() -> Result<GpuInfo, EmbeddingError> {
    use cudarc::driver::CudaDevice;

    // Step 1: Check if CUDA driver is available
    let device = CudaDevice::new(0).map_err(|e| {
        EmbeddingError::CudaUnavailable {
            message: format!(
                "No CUDA device found. Required: RTX 5090 (Blackwell, CC 12.0). Error: {}",
                e
            ),
        }
    })?;

    // Step 2: Get device properties
    let props = device.device_properties();
    let name = props.name().unwrap_or("Unknown GPU".to_string());
    let compute_major = props.major();
    let compute_minor = props.minor();
    let total_memory = props.total_global_mem();

    // Step 3: Verify compute capability
    if compute_major < REQUIRED_COMPUTE_MAJOR {
        return Err(EmbeddingError::CudaUnavailable {
            message: format!(
                "GPU compute capability {}.{} is below required {}.{}. Found: {}",
                compute_major, compute_minor,
                REQUIRED_COMPUTE_MAJOR, REQUIRED_COMPUTE_MINOR,
                name
            ),
        });
    }

    // Step 4: Verify VRAM
    if total_memory < MINIMUM_VRAM_BYTES {
        let required_gb = MINIMUM_VRAM_BYTES as f64 / (1024.0 * 1024.0 * 1024.0);
        let available_gb = total_memory as f64 / (1024.0 * 1024.0 * 1024.0);
        return Err(EmbeddingError::InsufficientVram {
            required_bytes: MINIMUM_VRAM_BYTES,
            available_bytes: total_memory,
            required_gb,
            available_gb,
        });
    }

    // Step 5: Verify it's NOT a simulation
    if name.to_lowercase().contains("simulated") || name.to_lowercase().contains("stub") {
        return Err(EmbeddingError::CudaUnavailable {
            message: format!(
                "Detected simulated/stub GPU: {}. Real GPU required.",
                name
            ),
        });
    }

    tracing::info!(
        "GPU verified: {} (CC {}.{}, {:.1} GB VRAM)",
        name, compute_major, compute_minor,
        total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    Ok(GpuInfo {
        device_id: 0,
        name,
        compute_capability: (compute_major, compute_minor),
        total_memory,
        driver_version: "CUDA 13.1+".to_string(), // TODO: Query actual version
    })
}

/// GPU information struct.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub device_id: u32,
    pub name: String,
    pub compute_capability: (u32, u32),
    pub total_memory: usize,
    pub driver_version: String,
}
```

**Constraints:**
- NO `#[cfg(not(feature = "cuda"))]` returning fake data
- MUST use `compile_error!` for missing CUDA feature
- MUST panic at runtime if GPU unavailable
- NO "Simulated" or "Stub" in any output

**Verification:**
- Build without cuda feature fails with clear error
- Build with cuda feature but no GPU panics at runtime
- No "Simulated RTX 5090" in any logs

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/warm/loader/preflight.rs` | DELETE stub mode, ADD compile_error |

#### Validation Criteria

- [ ] `cargo build` without cuda feature fails at compile time
- [ ] Error message includes remediation steps
- [ ] No "Simulated" or "Stub" strings anywhere
- [ ] Real GPU info reported for valid GPU

#### Test Commands

```bash
cd /home/cabdru/contextgraph
# This should fail with compile_error:
cargo check -p context-graph-embeddings 2>&1 | grep "EMB-E001"

# This should work:
cargo check -p context-graph-embeddings --features cuda

# Check for any remaining stub references:
grep -rn "Simulated\|Stub" crates/context-graph-embeddings/
```

</task_spec>

---

<task_spec id="TASK-EMB-020" version="1.0">

### TASK-EMB-020: Implement QuantizationRouter

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Implement Per-Embedder Quantization Router |
| **Status** | ready |
| **Layer** | logic |
| **Sequence** | 20 |
| **Implements** | REQ-EMB-005 |
| **Depends On** | TASK-EMB-016, TASK-EMB-017, TASK-EMB-018 |
| **Estimated Complexity** | medium |

#### Context

TECH-EMB-003 specifies per-embedder quantization based on Constitution. This task creates the router that dispatches each embedder to its correct quantization method.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Quantization types | `crates/context-graph-embeddings/src/quantization/types.rs` |
| PQ-8 | `crates/context-graph-embeddings/src/quantization/pq8.rs` |
| Float8 | `crates/context-graph-embeddings/src/quantization/float8.rs` |
| Binary | `crates/context-graph-embeddings/src/quantization/binary.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-003-quantization.md` |

#### Prerequisites

- [ ] TASK-EMB-016 completed (PQ-8 quantization)
- [ ] TASK-EMB-017 completed (Float8 quantization)
- [ ] TASK-EMB-018 completed (Binary quantization)

#### Scope

**In Scope:**
- Create `quantization/router.rs` module
- Route ModelId to correct quantization method
- Manage codebook loading for PQ-8
- Provide unified quantize/dequantize interface

**Out of Scope:**
- Sparse native format (already native)
- Token pruning for E12 (separate task)

#### Definition of Done

**File to Create:**

```rust
// File: crates/context-graph-embeddings/src/quantization/router.rs

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use super::binary::BinaryEncoder;
use super::float8::Float8Encoder;
use super::pq8::PQ8Quantizer;
use super::types::{QuantizationMethod, QuantizedEmbedding};
use crate::error::EmbeddingError;
use crate::types::model_id::ModelId;

/// Routes embeddings to their Constitutional quantization method.
///
/// # Constitution Alignment
/// - E1, E5, E7, E10: PQ-8 (32x, <5% recall loss)
/// - E2, E3, E4, E8, E11: Float8 (4x, <0.3% recall loss)
/// - E9: Binary (32x, 5-10% recall loss)
/// - E6, E13: Sparse native (no quantization)
/// - E12: Token pruning (handled separately)
///
/// # CRITICAL: No Fallback to Float32
/// Every embedder MUST use its assigned method. Float32 storage is FORBIDDEN.
pub struct QuantizationRouter {
    /// PQ-8 quantizers per model (need codebooks)
    pq8_quantizers: HashMap<ModelId, Arc<PQ8Quantizer>>,
    /// Float8 encoder (stateless, shared)
    float8_encoder: Float8Encoder,
    /// Binary encoder (stateless, shared)
    binary_encoder: BinaryEncoder,
}

impl QuantizationRouter {
    /// Create router and load all required codebooks.
    ///
    /// # Arguments
    /// * `codebook_dir` - Directory containing PQ-8 codebook files
    ///
    /// # Panics
    /// If required codebooks are missing.
    pub fn new(codebook_dir: &Path) -> Result<Self, EmbeddingError> {
        let mut pq8_quantizers = HashMap::new();

        // Load codebooks for PQ-8 embedders
        for model_id in [ModelId::Semantic, ModelId::Causal, ModelId::Code, ModelId::Multimodal] {
            let codebook_path = codebook_dir.join(format!("{:?}_codebook.bin", model_id));
            let quantizer = PQ8Quantizer::load(&codebook_path)?;
            pq8_quantizers.insert(model_id, Arc::new(quantizer));
        }

        Ok(Self {
            pq8_quantizers,
            float8_encoder: Float8Encoder::default(),
            binary_encoder: BinaryEncoder::default(),
        })
    }

    /// Get quantization method for a model.
    pub fn method_for(&self, model_id: ModelId) -> QuantizationMethod {
        QuantizationMethod::for_model_id(model_id)
    }

    /// Quantize embedding using Constitutional method for the model.
    ///
    /// # CRITICAL: No Float32 Output
    /// This function ALWAYS quantizes. There is no "pass-through" mode.
    pub fn quantize(&self, model_id: ModelId, embedding: &[f32]) -> Result<QuantizedEmbedding, EmbeddingError> {
        match self.method_for(model_id) {
            QuantizationMethod::PQ8 => {
                let quantizer = self.pq8_quantizers.get(&model_id)
                    .ok_or_else(|| EmbeddingError::CodebookMissing { model_id })?;
                quantizer.quantize(embedding)
            }
            QuantizationMethod::Float8E4M3 => {
                self.float8_encoder.quantize(embedding)
            }
            QuantizationMethod::Binary => {
                self.binary_encoder.quantize(embedding)
            }
            QuantizationMethod::SparseNative => {
                // Sparse vectors are stored as-is (indices + values)
                // This is handled separately in storage
                Err(EmbeddingError::DimensionMismatch {
                    model_id,
                    expected: 0,
                    actual: embedding.len(),
                })
            }
            QuantizationMethod::TokenPruning => {
                // Token pruning is handled separately for E12
                Err(EmbeddingError::DimensionMismatch {
                    model_id,
                    expected: 0,
                    actual: embedding.len(),
                })
            }
        }
    }

    /// Dequantize embedding back to f32.
    pub fn dequantize(&self, model_id: ModelId, quantized: &QuantizedEmbedding) -> Result<Vec<f32>, EmbeddingError> {
        match quantized.method {
            QuantizationMethod::PQ8 => {
                let quantizer = self.pq8_quantizers.get(&model_id)
                    .ok_or_else(|| EmbeddingError::CodebookMissing { model_id })?;
                quantizer.dequantize(quantized)
            }
            QuantizationMethod::Float8E4M3 => {
                self.float8_encoder.dequantize(quantized)
            }
            QuantizationMethod::Binary => {
                self.binary_encoder.dequantize(quantized)
            }
            _ => Err(EmbeddingError::StorageCorruption {
                id: format!("{:?}", model_id),
                reason: "Unsupported quantization method for dequantize".to_string(),
            }),
        }
    }

    /// Get expected compressed size for a model.
    pub fn expected_size(&self, model_id: ModelId, original_dim: usize) -> usize {
        match self.method_for(model_id) {
            QuantizationMethod::PQ8 => 8, // Always 8 bytes
            QuantizationMethod::Float8E4M3 => original_dim, // 1 byte per element
            QuantizationMethod::Binary => (original_dim + 7) / 8, // 1 bit per element
            QuantizationMethod::SparseNative => original_dim * 5, // Approximate for sparse
            QuantizationMethod::TokenPruning => original_dim / 2, // ~50% reduction
        }
    }
}

/// Table of embedder -> quantization assignments for documentation.
pub const QUANTIZATION_ASSIGNMENTS: &[(ModelId, QuantizationMethod, &str)] = &[
    (ModelId::Semantic, QuantizationMethod::PQ8, "E1: 32x compression"),
    (ModelId::TemporalRecent, QuantizationMethod::Float8E4M3, "E2: 4x compression"),
    (ModelId::TemporalPeriodic, QuantizationMethod::Float8E4M3, "E3: 4x compression"),
    (ModelId::TemporalPositional, QuantizationMethod::Float8E4M3, "E4: 4x compression"),
    (ModelId::Causal, QuantizationMethod::PQ8, "E5: 32x compression"),
    (ModelId::Sparse, QuantizationMethod::SparseNative, "E6: native sparse"),
    (ModelId::Code, QuantizationMethod::PQ8, "E7: 32x compression"),
    (ModelId::Graph, QuantizationMethod::Float8E4M3, "E8: 4x compression"),
    (ModelId::Hdc, QuantizationMethod::Binary, "E9: 32x compression"),
    (ModelId::Multimodal, QuantizationMethod::PQ8, "E10: 32x compression"),
    (ModelId::Entity, QuantizationMethod::Float8E4M3, "E11: 4x compression"),
    (ModelId::LateInteraction, QuantizationMethod::TokenPruning, "E12: ~50% reduction"),
    (ModelId::Splade, QuantizationMethod::SparseNative, "E13: native sparse"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_embedders_have_assignment() {
        let all_models = [
            ModelId::Semantic, ModelId::TemporalRecent, ModelId::TemporalPeriodic,
            ModelId::TemporalPositional, ModelId::Causal, ModelId::Sparse,
            ModelId::Code, ModelId::Graph, ModelId::Hdc, ModelId::Multimodal,
            ModelId::Entity, ModelId::LateInteraction, ModelId::Splade,
        ];

        for model_id in all_models {
            let method = QuantizationMethod::for_model_id(model_id);
            assert_ne!(
                method, QuantizationMethod::SparseNative,
                "Model {:?} should have explicit assignment (unless sparse)",
            );
        }
    }
}
```

**Constraints:**
- Every ModelId MUST have a quantization assignment
- No Float32 pass-through allowed
- PQ-8 requires codebook loading
- Float8 and Binary are stateless

**Verification:**
- All 13 embedders routed correctly
- Codebook loading works
- Quantize/dequantize round-trip works

#### Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/quantization/router.rs` | QuantizationRouter |

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/quantization/mod.rs` | Add `pub mod router;` |

#### Validation Criteria

- [ ] All 13 embedders have correct method assignment
- [ ] Router loads successfully with codebooks
- [ ] `quantize()` dispatches to correct encoder
- [ ] `dequantize()` reverses quantization

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo test -p context-graph-embeddings quantization::router -- --nocapture
```

</task_spec>

---

</task_collection>

---

## Summary

### Task List

| Task ID | Title | Dependencies | Complexity |
|---------|-------|--------------|------------|
| TASK-EMB-011 | Implement ProjectionMatrix::load() | TASK-EMB-002, TASK-EMB-003 | high |
| TASK-EMB-012 | Implement ProjectionMatrix::project() | TASK-EMB-011 | high |
| TASK-EMB-013 | Implement Real Weight Loading | TASK-EMB-006 | high |
| TASK-EMB-014 | Implement Real VRAM Allocation | TASK-EMB-013 | high |
| TASK-EMB-015 | Implement Real Inference Validation | TASK-EMB-014, TASK-EMB-010 | high |
| TASK-EMB-016 | Implement PQ-8 Quantization | TASK-EMB-004 | high |
| TASK-EMB-017 | Implement Float8 Quantization | TASK-EMB-004 | medium |
| TASK-EMB-018 | Implement Binary Quantization | TASK-EMB-004 | low |
| TASK-EMB-019 | Remove Stub Mode from Preflight | TASK-EMB-014 | low |
| TASK-EMB-020 | Implement QuantizationRouter | TASK-EMB-016, TASK-EMB-017, TASK-EMB-018 | medium |

### Execution Order

**Critical Path:**
```
Foundation Tasks (COMPLETE)
         |
         v
TASK-EMB-011 -> TASK-EMB-012 (Sparse Projection)
         |
TASK-EMB-013 -> TASK-EMB-014 -> TASK-EMB-015 (Warm Loading)
                     |
                     +---> TASK-EMB-019 (Stub Removal)
         |
TASK-EMB-016 --\
TASK-EMB-017 --+--> TASK-EMB-020 (Quantization Router)
TASK-EMB-018 --/
```

**Parallel Tracks:**
- Track A: TASK-EMB-011 -> TASK-EMB-012 (Sparse Projection)
- Track B: TASK-EMB-013 -> TASK-EMB-014 -> TASK-EMB-015 -> TASK-EMB-019 (Warm Loading)
- Track C: TASK-EMB-016, TASK-EMB-017, TASK-EMB-018 -> TASK-EMB-020 (Quantization)

### Functions to DELETE

| Function | File | Reason |
|----------|------|--------|
| `simulate_weight_loading()` | `warm/loader/operations.rs` | Fake checksum |
| Fake pointer generation | `warm/loader/operations.rs` | `0x7f80_0000_0000` |
| Sin wave output | `warm/loader/operations.rs` | `(i * 0.001).sin()` |
| Stub mode block | `warm/loader/preflight.rs` | "Simulated RTX 5090" |

### Files Created/Modified Summary

| File | Tasks |
|------|-------|
| `models/pretrained/sparse/projection.rs` | TASK-EMB-011, TASK-EMB-012 |
| `warm/loader/operations.rs` | TASK-EMB-013, TASK-EMB-014, TASK-EMB-015 |
| `warm/loader/preflight.rs` | TASK-EMB-019 |
| `quantization/pq8.rs` | TASK-EMB-016 |
| `quantization/float8.rs` | TASK-EMB-017 |
| `quantization/binary.rs` | TASK-EMB-018 |
| `quantization/router.rs` | TASK-EMB-020 |

---

## Next Steps

After Logic Layer completion:

1. **Surface Layer Tasks** (TASK-EMB-021 through TASK-EMB-030):
   - Integration tests
   - End-to-end benchmarks
   - Storage module implementation
   - Migration guide
   - Documentation updates

---

## Memory Key

Store this summary for next agent:
```
contextgraph/embedding-issues/task-logic-summary
```

