# TASK-CORE-011: GPU Memory Management Infrastructure

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-CORE-011 |
| **Title** | GPU Memory Management Infrastructure |
| **Status** | :white_circle: todo |
| **Layer** | Foundation |
| **Sequence** | 11 |
| **Estimated Days** | 3 |
| **Complexity** | High |

## Implements

- **ARCH-08**: CUDA GPU is Required for Production
- **Performance Budget**: GPU memory usage < 8GB

## Dependencies

| Task | Reason |
|------|--------|
| None | Foundation task with no upstream dependencies |

## Objective

Create GPU memory management infrastructure for loading and managing 13 embedding models within the 8GB VRAM budget on RTX 5090 / Blackwell architecture.

## Context

The constitution (ARCH-08) mandates:
- CUDA GPU is required for production
- No CPU fallbacks in production builds
- System fails fast if GPU unavailable
- Test environments may use stubs via `test-utils` feature flag
- GPU memory budget is < 8GB for all 13 models loaded

## Scope

### In Scope

- `GpuMemoryPool` struct with allocation/deallocation
- Model slot management for 13 embedders
- Memory pressure detection and eviction
- CUDA device selection and initialization
- Test-mode CPU fallback (feature-gated)
- Memory usage metrics and monitoring
- Fail-fast behavior when GPU unavailable

### Out of Scope

- Model loading (see TASK-CORE-012)
- Quantization (see TASK-CORE-013)
- Specific embedding model implementations

## Definition of Done

### Signatures

```rust
// crates/context-graph-cuda/src/memory.rs

use std::sync::atomic::AtomicU64;

/// GPU memory pool with allocation tracking
pub struct GpuMemoryPool {
    device: CudaDevice,
    allocated: AtomicU64,
    limit: u64,
}

impl GpuMemoryPool {
    /// Create new pool on specified device with memory limit
    pub fn new(device_id: usize, limit_bytes: u64) -> CudaResult<Self>;

    /// Allocate GPU buffer
    pub fn allocate(&self, size: usize) -> CudaResult<GpuBuffer>;

    /// Deallocate GPU buffer
    pub fn deallocate(&self, buffer: GpuBuffer);

    /// Get available memory
    pub fn available(&self) -> u64;

    /// Get current memory pressure level
    pub fn pressure_level(&self) -> MemoryPressure;

    /// Check if GPU is available (fail-fast check)
    pub fn check_availability() -> CudaResult<()>;
}

/// Memory pressure levels for eviction decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    Low,      // <50% used
    Medium,   // 50-80% used
    High,     // 80-95% used
    Critical, // >95% used
}

/// RAII GPU buffer wrapper
pub struct GpuBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
    pool: Arc<GpuMemoryPool>,
}

impl Drop for GpuBuffer {
    fn drop(&mut self);
}

// crates/context-graph-cuda/src/device.rs

/// CUDA device handle
pub struct CudaDevice {
    id: usize,
    properties: CudaDeviceProperties,
}

impl CudaDevice {
    /// Initialize device with fail-fast
    pub fn init(device_id: usize) -> CudaResult<Self>;

    /// Get total VRAM
    pub fn total_memory(&self) -> u64;

    /// Get free VRAM
    pub fn free_memory(&self) -> u64;

    /// Get device name
    pub fn name(&self) -> &str;

    /// Check if device supports required compute capability
    pub fn supports_required_features(&self) -> bool;
}

#[derive(Debug)]
pub struct CudaDeviceProperties {
    pub compute_capability: (i32, i32),
    pub total_memory: u64,
    pub multiprocessor_count: i32,
    pub name: String,
}
```

### Test Stubs (Feature-Gated)

```rust
// crates/context-graph-cuda/src/mock.rs
#[cfg(feature = "test-utils")]

/// Mock GPU pool for testing without hardware
pub struct MockGpuMemoryPool {
    allocated: AtomicU64,
    limit: u64,
}

impl MockGpuMemoryPool {
    pub fn new(limit_bytes: u64) -> Self;
    pub fn allocate(&self, size: usize) -> CudaResult<MockGpuBuffer>;
    // ... same interface as GpuMemoryPool
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Total GPU memory limit | < 8GB for 13 models |
| Allocation latency | < 1ms |
| Deallocation latency | < 0.5ms |
| Fail-fast startup check | < 100ms |

## Verification

- [ ] `GpuMemoryPool::new()` fails fast with clear error if no GPU
- [ ] Memory allocation tracks usage correctly
- [ ] Memory pressure detection triggers at correct thresholds
- [ ] `test-utils` feature allows tests to run without GPU
- [ ] Device properties correctly detected
- [ ] Buffer RAII cleanup on drop

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-cuda/src/lib.rs` | Crate root |
| `crates/context-graph-cuda/src/memory.rs` | Memory pool implementation |
| `crates/context-graph-cuda/src/device.rs` | Device management |
| `crates/context-graph-cuda/src/error.rs` | CUDA error types |
| `crates/context-graph-cuda/src/mock.rs` | Test stubs (feature-gated) |
| `crates/context-graph-cuda/Cargo.toml` | Crate manifest |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CUDA API changes | Low | Medium | Pin to specific CUDA version |
| OOM during operation | Medium | High | Memory pressure monitoring |
| Device detection fails | Low | High | Clear error messages |

## Traceability

- Source: ARCH-08 (Constitution lines 301-313)
- Performance Budget: GPU memory < 8GB (Constitution line 587)
