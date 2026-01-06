# TASK-EMB-006: Create WarmLoadResult Struct

<task_spec id="TASK-EMB-006" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-006 |
| **Title** | Create Warm Loading Data Types |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 6 |
| **Implements** | REQ-EMB-003 |
| **Depends On** | TASK-EMB-001 |
| **Estimated Complexity** | medium |
| **Constitution Ref** | `stack.gpu`, `AP-007` |

---

## Context

TECH-EMB-002 specifies real weight loading to replace simulated operations. This task creates the data structures for loaded weights, GPU tensors, and loading results. Actual loading implementation is Logic Layer.

**Why This Matters:**
- Current warm loading is SIMULATED (fake pointers, fake checksums)
- Constitution AP-007 forbids stub data in production
- Real VRAM allocation is required for inference

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-002-warm-loading.md` |
| Current warm loader | `crates/context-graph-embeddings/src/warm/loader/` |

---

## Prerequisites

- [ ] TASK-EMB-001 completed
- [ ] cudarc types available (for DevicePtr)
- [ ] Instant type from std::time

---

## Scope

### In Scope
- Create `WarmLoadResult` struct
- Create `GpuTensor` struct
- Create `LoadedModelWeights` struct
- Create `TensorMetadata` struct
- Create `DType` enum

### Out of Scope
- Actual loading implementations (Logic Layer - TASK-EMB-013)
- CUDA calls (Logic Layer - TASK-EMB-014)
- SafeTensors parsing (Logic Layer)

---

## Definition of Done

### Exact Signatures

```rust
// File: crates/context-graph-embeddings/src/warm/loader/types.rs

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Data type for model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

/// Result of loading a model's weights into GPU memory.
///
/// # Constitution Alignment
/// - REQ-WARM-003: Non-evictable VRAM allocation
/// - REQ-WARM-005: Weight integrity verification
///
/// # CRITICAL: No Simulation
/// All fields contain REAL data from actual loading operations.
#[derive(Debug)]
pub struct WarmLoadResult {
    /// Real GPU device pointer from cudaMalloc.
    pub gpu_ptr: u64, // Raw pointer as u64 for portability
    /// Real SHA256 checksum of the weight file.
    pub checksum: [u8; 32],
    /// Actual size of weights in GPU memory.
    pub size_bytes: usize,
    /// Loading duration for performance monitoring.
    pub load_duration: Duration,
    /// Tensor metadata from SafeTensors header.
    pub tensor_metadata: TensorMetadata,
}

/// Metadata extracted from SafeTensors file header.
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Tensor name -> shape mapping.
    pub shapes: HashMap<String, Vec<usize>>,
    /// Data type of tensors.
    pub dtype: DType,
    /// Total number of parameters.
    pub total_params: usize,
}

/// GPU-resident tensor with real CUDA allocation.
///
/// # Safety
/// The `device_ptr` is a REAL CUDA device pointer obtained from cudaMalloc.
/// It is ONLY valid while the CUDA context is active.
#[derive(Debug)]
pub struct GpuTensor {
    /// Real CUDA device pointer from cudaMalloc.
    device_ptr: u64,
    /// Tensor shape (e.g., [vocab_size, hidden_dim]).
    shape: Vec<usize>,
    /// Total number of elements.
    numel: usize,
    /// CUDA device ID where tensor is allocated.
    device_id: u32,
}

impl GpuTensor {
    /// Create a new GPU tensor record.
    ///
    /// Note: This does NOT allocate memory. Allocation is done in Logic Layer.
    pub fn new(device_ptr: u64, shape: Vec<usize>, device_id: u32) -> Self {
        let numel = shape.iter().product();
        Self { device_ptr, shape, numel, device_id }
    }

    /// Get the raw device pointer.
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    /// Get tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get number of elements.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Get device ID.
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

/// Complete set of weights for a model loaded into GPU memory.
#[derive(Debug)]
pub struct LoadedModelWeights {
    /// Model identifier (e.g., "E1_Semantic").
    pub model_id: String,
    /// Named tensors loaded to GPU.
    pub tensors: HashMap<String, GpuTensor>,
    /// SHA256 checksum of source weight file.
    pub file_checksum: [u8; 32],
    /// Total GPU memory used (bytes).
    pub total_gpu_bytes: usize,
    /// CUDA device where weights are loaded.
    pub device_id: u32,
    /// Timestamp when weights were loaded.
    pub loaded_at: Instant,
}

impl LoadedModelWeights {
    /// Get a specific tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&GpuTensor> {
        self.tensors.get(name)
    }

    /// Check if all expected tensors are present.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}
```

### Constraints
- Use u64 for device pointers (portable across CUDA wrappers)
- No actual CUDA calls in this task
- Structs document real-data requirement

### Verification
- All types compile
- No dependencies on CUDA at compile time

---

## Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/warm/loader/types.rs` | Warm loading types |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/warm/loader/mod.rs` | Add `pub mod types;` |

---

## Validation Criteria

- [ ] `WarmLoadResult` has checksum as `[u8; 32]`
- [ ] `GpuTensor` uses u64 for device pointer
- [ ] `LoadedModelWeights` has HashMap for tensors
- [ ] All structs document CRITICAL: No Simulation

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
```

---

## Traceability

| Requirement | Tech Spec | Issue |
|-------------|-----------|-------|
| REQ-EMB-003 | TECH-EMB-002 | ISSUE-003 |

---

## Anti-Patterns to Avoid

| Current Code | Problem | This Task |
|--------------|---------|-----------|
| `0x7f80_0000_0000` | Fake GPU pointer | Use real u64 from cudaMalloc |
| `0xDEAD_BEEF_CAFE_BABE` | Fake checksum | Use real SHA256 |
| `simulate_weight_loading()` | No actual loading | Types for real loading |

</task_spec>
