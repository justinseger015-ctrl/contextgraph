# Functional Specification: Architecture Domain Remediation

## Metadata

| Field | Value |
|-------|-------|
| **ID** | SPEC-ARCH-001 |
| **Title** | CUDA FFI Consolidation - Architecture Domain Remediation |
| **Status** | Draft |
| **Version** | 1.0.0 |
| **Owner** | Architecture Team |
| **Created** | 2026-01-12 |
| **Last Updated** | 2026-01-12 |
| **Related Specs** | SPEC-GWT-REMEDIATION, SPEC-PERFORMANCE-REMEDIATION |
| **Source Issue** | ISS-005: CUDA FFI Scattered Across 3 Crates |
| **Constitution Rule** | `rules: "CUDA FFI only in context-graph-cuda"` |
| **Severity** | CRITICAL |

---

## Overview

### Problem Statement

CUDA Foreign Function Interface (FFI) declarations are currently scattered across three crates instead of being consolidated in the designated `context-graph-cuda` crate. This architectural violation creates multiple security, maintainability, and reliability risks.

### Why FFI Consolidation Matters

1. **Safety Auditing**: All `unsafe` FFI code in one place enables systematic security review. Scattered FFI makes auditing exponentially harder.

2. **Symbol Conflict Prevention**: Duplicate `extern "C"` declarations risk linker conflicts and undefined behavior at runtime.

3. **Error Handling Consistency**: Centralized FFI ensures uniform CUDA error handling patterns, preventing silent failures.

4. **Maintainability**: When CUDA APIs change (e.g., v13.1 WSL2 segfault workarounds), one crate to update vs. three.

5. **Build System Simplicity**: Single FFI crate simplifies linking and reduces build configuration complexity.

6. **Testability**: Safe wrappers can be unit-tested in isolation; scattered FFI cannot.

### Current State (Violations)

| Crate | File | FFI Type | Functions |
|-------|------|----------|-----------|
| context-graph-embeddings | `src/gpu/device/utils.rs:29` | CUDA Driver API | `cuInit`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceTotalMem_v2`, `cuDeviceGetAttribute`, `cuDriverGetVersion` |
| context-graph-embeddings | `src/warm/cuda_alloc/allocator_cuda.rs:70` | CUDA Driver API | `cuInit`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceTotalMem_v2`, `cuDeviceGetAttribute`, `cuDriverGetVersion` (DUPLICATE) |
| context-graph-graph | `src/index/faiss_ffi/bindings.rs:26` | FAISS GPU C API | `faiss_index_factory`, `faiss_StandardGpuResources_new`, `faiss_StandardGpuResources_free`, `faiss_index_cpu_to_gpu`, `faiss_Index_*`, `faiss_get_num_gpus` |

### Target State

All CUDA/GPU FFI declarations consolidated in `context-graph-cuda` with:
- Raw FFI in `context-graph-cuda/src/ffi/`
- Safe Rust wrappers in `context-graph-cuda/src/safe/`
- Re-exports through `context-graph-cuda/src/lib.rs`
- CI gate preventing new FFI in non-cuda crates

---

## User Stories

### US-ARCH-001: Developer Audits Unsafe Code

**Priority**: Must-Have

**Narrative**:
```
As a security auditor
I want all CUDA FFI declarations in a single crate
So that I can review all unsafe GPU code in one location
```

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-001-01 | A security auditor opens context-graph-cuda | They search for `extern "C"` | All CUDA FFI declarations are found in `src/ffi/` subdirectory |
| AC-001-02 | An auditor searches context-graph-embeddings for `extern "C"` | The search completes | Zero matches are returned |
| AC-001-03 | An auditor searches context-graph-graph for `extern "C"` | The search completes | Zero matches are returned (FAISS FFI moved) |

---

### US-ARCH-002: Developer Uses Safe GPU APIs

**Priority**: Must-Have

**Narrative**:
```
As a Rust developer in the embeddings crate
I want safe wrapper functions for CUDA operations
So that I don't need to write unsafe code for GPU queries
```

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-002-01 | A developer needs GPU device info | They import from context-graph-cuda | Safe `GpuDevice::query_info()` is available |
| AC-002-02 | A developer uses the safe wrapper | CUDA returns an error | A typed `CudaError` is returned (no raw i32) |
| AC-002-03 | A developer calls `GpuDevice::query_info()` | The call succeeds | No `unsafe` block required at call site |

---

### US-ARCH-003: CI Prevents FFI Drift

**Priority**: Must-Have

**Narrative**:
```
As a maintainer
I want CI to fail if extern "C" appears in non-cuda crates
So that the architecture constraint is automatically enforced
```

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-003-01 | A developer adds `extern "C"` to context-graph-embeddings | They push to the repo | Pre-merge CI fails with explicit error message |
| AC-003-02 | CI runs the FFI check | The check finds violations | Exit code is non-zero and violation locations are logged |
| AC-003-03 | All FFI is in context-graph-cuda only | CI runs the FFI check | Check passes with exit code 0 |

---

### US-ARCH-004: Build System Links Correctly

**Priority**: Must-Have

**Narrative**:
```
As a build engineer
I want single-source FFI linking
So that symbol conflicts are impossible
```

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-004-01 | The workspace is built with `cargo build --workspace` | Build completes | No duplicate symbol warnings or errors |
| AC-004-02 | FAISS GPU is linked | Runtime loads libfaiss_c | Single link, no conflicts |
| AC-004-03 | CUDA Driver API is used | Runtime initializes | Single `cuInit` call path |

---

### US-ARCH-005: FAISS GPU Operations Work

**Priority**: Must-Have

**Narrative**:
```
As the graph indexing system
I want FAISS GPU FFI available through context-graph-cuda
So that vector search uses GPU acceleration
```

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-005-01 | context-graph-graph needs FAISS GPU | It imports from context-graph-cuda | `faiss::GpuResources` and `faiss::GpuIndex` are available |
| AC-005-02 | A GPU index is created | Resources are allocated | Safe RAII wrapper manages GPU memory lifetime |
| AC-005-03 | A GPU index is dropped | Destructor runs | GPU resources are freed automatically |

---

## Requirements

### REQ-ARCH-001: Remove extern "C" from context-graph-embeddings

**Story Ref**: US-ARCH-001, US-ARCH-002
**Priority**: CRITICAL
**Constitution Rule**: `rules: "CUDA FFI only in context-graph-cuda"`

**Description**:
All `extern "C"` blocks in the `context-graph-embeddings` crate MUST be removed. The affected files are:
- `crates/context-graph-embeddings/src/gpu/device/utils.rs` (lines 29-50)
- `crates/context-graph-embeddings/src/warm/cuda_alloc/allocator_cuda.rs` (lines 70-77)

**Rationale**:
These files contain duplicate CUDA Driver API declarations (`cuInit`, `cuDeviceGet`, etc.) that violate the single-crate FFI rule. Duplication increases maintenance burden and risks linker conflicts.

**Verification**:
```bash
grep -r 'extern "C"' crates/context-graph-embeddings/
# Expected output: (empty)
```

---

### REQ-ARCH-002: Remove extern "C" from context-graph-graph

**Story Ref**: US-ARCH-001, US-ARCH-005
**Priority**: CRITICAL
**Constitution Rule**: `rules: "CUDA FFI only in context-graph-cuda"`

**Description**:
All `extern "C"` blocks in the `context-graph-graph` crate MUST be removed. The affected file is:
- `crates/context-graph-graph/src/index/faiss_ffi/bindings.rs` (lines 26-188)

**Rationale**:
FAISS GPU bindings belong in the CUDA crate. The current location violates architectural boundaries and makes it harder to audit GPU-related unsafe code.

**Verification**:
```bash
grep -r 'extern "C"' crates/context-graph-graph/
# Expected output: (empty)
```

---

### REQ-ARCH-003: Consolidate CUDA Driver API FFI

**Story Ref**: US-ARCH-002
**Priority**: CRITICAL

**Description**:
Create a unified CUDA Driver API FFI module in `context-graph-cuda`:
- Raw FFI: `context-graph-cuda/src/ffi/cuda_driver.rs`
- Safe wrappers: `context-graph-cuda/src/safe/device.rs`

**Required Functions**:
```rust
// In context-graph-cuda/src/ffi/cuda_driver.rs
extern "C" {
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(value: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult;
    pub fn cuDriverGetVersion(version: *mut c_int) -> CUresult;
}
```

**Safe Wrapper Interface**:
```rust
// In context-graph-cuda/src/safe/device.rs
pub struct GpuDevice { /* ... */ }

impl GpuDevice {
    pub fn init() -> Result<(), CudaError>;
    pub fn count() -> Result<usize, CudaError>;
    pub fn get(ordinal: usize) -> Result<Self, CudaError>;
    pub fn name(&self) -> Result<String, CudaError>;
    pub fn total_memory(&self) -> Result<usize, CudaError>;
    pub fn compute_capability(&self) -> Result<(u32, u32), CudaError>;
    pub fn driver_version() -> Result<(u32, u32), CudaError>;
}
```

---

### REQ-ARCH-004: Consolidate FAISS GPU FFI

**Story Ref**: US-ARCH-005
**Priority**: CRITICAL

**Description**:
Move all FAISS C API bindings to `context-graph-cuda`:
- Raw FFI: `context-graph-cuda/src/ffi/faiss.rs`
- Safe wrappers: `context-graph-cuda/src/safe/faiss.rs`

**Required Functions** (moved from context-graph-graph):
```rust
// In context-graph-cuda/src/ffi/faiss.rs
#[link(name = "faiss_c")]
extern "C" {
    pub fn faiss_index_factory(...) -> c_int;
    pub fn faiss_StandardGpuResources_new(...) -> c_int;
    pub fn faiss_StandardGpuResources_free(...);
    pub fn faiss_index_cpu_to_gpu(...) -> c_int;
    pub fn faiss_Index_train(...) -> c_int;
    pub fn faiss_Index_is_trained(...) -> c_int;
    pub fn faiss_Index_add_with_ids(...) -> c_int;
    pub fn faiss_Index_search(...) -> c_int;
    pub fn faiss_IndexIVF_set_nprobe(...);
    pub fn faiss_Index_ntotal(...) -> c_long;
    pub fn faiss_write_index(...) -> c_int;
    pub fn faiss_read_index(...) -> c_int;
    pub fn faiss_Index_free(...);
    pub fn faiss_get_num_gpus(...) -> c_int;
}
```

**Safe Wrapper Interface**:
```rust
// In context-graph-cuda/src/safe/faiss.rs
pub struct GpuResources { /* RAII wrapper */ }
pub struct GpuIndex { /* RAII wrapper */ }

impl GpuResources {
    pub fn new() -> Result<Self, FaissError>;
}

impl Drop for GpuResources {
    fn drop(&mut self) { /* calls faiss_StandardGpuResources_free */ }
}

impl GpuIndex {
    pub fn from_cpu(resources: &GpuResources, cpu_index: CpuIndex) -> Result<Self, FaissError>;
    pub fn search(&self, queries: &[f32], k: usize) -> Result<SearchResults, FaissError>;
}
```

---

### REQ-ARCH-005: CI Gate for FFI Violations

**Story Ref**: US-ARCH-003
**Priority**: HIGH
**Constitution Rule**: `testing.gates.pre-merge`

**Description**:
Add a CI check that fails the build if `extern "C"` is found in any crate other than `context-graph-cuda`.

**Implementation**:
```bash
#!/bin/bash
# scripts/check-ffi-consolidation.sh

set -e

echo "Checking for extern \"C\" in non-cuda crates..."

VIOLATIONS=""

# Check context-graph-embeddings
if grep -rn 'extern "C"' crates/context-graph-embeddings/src/ 2>/dev/null; then
    VIOLATIONS="${VIOLATIONS}\n- context-graph-embeddings"
fi

# Check context-graph-graph
if grep -rn 'extern "C"' crates/context-graph-graph/src/ 2>/dev/null; then
    VIOLATIONS="${VIOLATIONS}\n- context-graph-graph"
fi

# Check context-graph-core
if grep -rn 'extern "C"' crates/context-graph-core/src/ 2>/dev/null; then
    VIOLATIONS="${VIOLATIONS}\n- context-graph-core"
fi

# Check context-graph-mcp
if grep -rn 'extern "C"' crates/context-graph-mcp/src/ 2>/dev/null; then
    VIOLATIONS="${VIOLATIONS}\n- context-graph-mcp"
fi

# Check context-graph-storage
if grep -rn 'extern "C"' crates/context-graph-storage/src/ 2>/dev/null; then
    VIOLATIONS="${VIOLATIONS}\n- context-graph-storage"
fi

if [ -n "$VIOLATIONS" ]; then
    echo ""
    echo "=========================================="
    echo "CONSTITUTION VIOLATION: CUDA FFI only in context-graph-cuda"
    echo "=========================================="
    echo ""
    echo "extern \"C\" blocks found in prohibited crates:"
    echo -e "$VIOLATIONS"
    echo ""
    echo "Move all FFI declarations to:"
    echo "  - context-graph-cuda/src/ffi/ (raw bindings)"
    echo "  - context-graph-cuda/src/safe/ (safe wrappers)"
    echo ""
    exit 1
fi

echo "FFI consolidation check PASSED"
exit 0
```

**CI Integration** (in `.github/workflows/ci.yml` or similar):
```yaml
- name: Check FFI Consolidation
  run: ./scripts/check-ffi-consolidation.sh
```

---

## Edge Cases

| ID | Related Req | Scenario | Expected Behavior |
|----|-------------|----------|-------------------|
| EC-ARCH-001 | REQ-ARCH-003 | CUDA driver not installed | `GpuDevice::init()` returns `Err(CudaError::DriverNotFound)` |
| EC-ARCH-002 | REQ-ARCH-003 | No CUDA-capable GPU | `GpuDevice::count()` returns `Ok(0)` |
| EC-ARCH-003 | REQ-ARCH-004 | FAISS compiled without GPU support | `GpuResources::new()` returns `Err(FaissError::GpuNotAvailable)` |
| EC-ARCH-004 | REQ-ARCH-004 | GPU out of memory during index transfer | `GpuIndex::from_cpu()` returns `Err(FaissError::OutOfMemory)` |
| EC-ARCH-005 | REQ-ARCH-001 | Link-time symbol conflict | Build fails with duplicate symbol error (prevented by consolidation) |
| EC-ARCH-006 | REQ-ARCH-005 | Developer accidentally adds `extern "C"` | CI fails pre-merge with explicit error message |
| EC-ARCH-007 | REQ-ARCH-003 | WSL2 CUDA 13.1 Runtime API segfault | Driver API path (`cuInit`) is used instead of Runtime API |

---

## Error States

| ID | HTTP/Exit | Condition | Message | Recovery |
|----|-----------|-----------|---------|----------|
| ERR-ARCH-001 | Exit 108 | CUDA driver initialization failed | `"CUDA driver init failed: {cuda_error}"` | Install CUDA drivers, verify GPU connection |
| ERR-ARCH-002 | Exit 108 | No CUDA-capable GPU found | `"No CUDA GPU detected"` | Install CUDA-capable GPU |
| ERR-ARCH-003 | Exit 108 | FAISS GPU resources allocation failed | `"FAISS GPU resources failed: {error}"` | Reduce GPU memory usage, restart |
| ERR-ARCH-004 | Exit 108 | FAISS index GPU transfer failed | `"CPU to GPU transfer failed: {error}"` | Free GPU memory, retry |
| ERR-ARCH-005 | CI Fail | FFI found in non-cuda crate | `"CONSTITUTION VIOLATION: CUDA FFI only in context-graph-cuda"` | Move FFI to context-graph-cuda |
| ERR-ARCH-006 | Exit 109 | Fake CUDA allocation detected | `"[AP-007] Fake GPU allocation detected"` | Fix CUDA installation |

---

## Test Plan

### Unit Tests

| ID | Type | Req Ref | Description | Inputs | Expected |
|----|------|---------|-------------|--------|----------|
| TC-ARCH-001 | Unit | REQ-ARCH-003 | Test `GpuDevice::init()` success path | Valid CUDA environment | `Ok(())` |
| TC-ARCH-002 | Unit | REQ-ARCH-003 | Test `GpuDevice::count()` returns GPU count | System with 1+ GPUs | `Ok(n)` where n >= 1 |
| TC-ARCH-003 | Unit | REQ-ARCH-003 | Test `GpuDevice::name()` returns real name | Valid GPU | Non-empty string |
| TC-ARCH-004 | Unit | REQ-ARCH-004 | Test `GpuResources::new()` allocates | GPU available | `Ok(GpuResources)` |
| TC-ARCH-005 | Unit | REQ-ARCH-004 | Test `GpuResources::drop()` frees memory | Allocated resources | No memory leak |

### Integration Tests

| ID | Type | Req Ref | Description | Inputs | Expected |
|----|------|---------|-------------|--------|----------|
| TC-ARCH-010 | Integration | REQ-ARCH-001 | Embeddings crate uses cuda safe wrappers | Import from context-graph-cuda | Compiles and works |
| TC-ARCH-011 | Integration | REQ-ARCH-002 | Graph crate uses FAISS safe wrappers | Import from context-graph-cuda | Compiles and works |
| TC-ARCH-012 | Integration | REQ-ARCH-004 | End-to-end GPU vector search | 1000 vectors, k=10 | Correct results |

### CI Gate Tests

| ID | Type | Req Ref | Description | Inputs | Expected |
|----|------|---------|-------------|--------|----------|
| TC-ARCH-020 | CI | REQ-ARCH-005 | FFI check passes on clean codebase | No FFI in non-cuda crates | Exit 0 |
| TC-ARCH-021 | CI | REQ-ARCH-005 | FFI check fails on violation | FFI in embeddings crate | Exit 1 + error message |
| TC-ARCH-022 | CI | REQ-ARCH-005 | FFI check ignores cuda crate | FFI in context-graph-cuda | Exit 0 |

### Verification Commands

```bash
# Verify no extern "C" in embeddings crate
! grep -r 'extern "C"' crates/context-graph-embeddings/src/

# Verify no extern "C" in graph crate
! grep -r 'extern "C"' crates/context-graph-graph/src/

# Verify FFI exists in cuda crate
grep -r 'extern "C"' crates/context-graph-cuda/src/ffi/

# Verify safe wrappers exist
test -f crates/context-graph-cuda/src/safe/device.rs
test -f crates/context-graph-cuda/src/safe/faiss.rs

# Verify CI script exists and is executable
test -x scripts/check-ffi-consolidation.sh

# Run CI check
./scripts/check-ffi-consolidation.sh

# Full workspace build (no duplicate symbols)
cargo build --workspace

# Run all tests
cargo test --workspace
```

---

## Implementation Notes

### Directory Structure (After Remediation)

```
crates/context-graph-cuda/
├── src/
│   ├── lib.rs                    # Re-exports public API
│   ├── error.rs                  # CudaError, FaissError types
│   ├── ffi/
│   │   ├── mod.rs
│   │   ├── cuda_driver.rs        # Raw CUDA Driver API bindings
│   │   ├── cuda_runtime.rs       # Raw CUDA Runtime API bindings (future)
│   │   └── faiss.rs              # Raw FAISS C API bindings
│   ├── safe/
│   │   ├── mod.rs
│   │   ├── device.rs             # Safe GpuDevice wrapper
│   │   └── faiss.rs              # Safe FAISS wrappers (GpuResources, GpuIndex)
│   ├── poincare/                 # Existing hyperbolic geometry
│   └── cone/                     # Existing cone check
└── Cargo.toml
```

### Migration Sequence

1. **Phase 1**: Create new FFI modules in context-graph-cuda
   - Add `src/ffi/cuda_driver.rs` with consolidated declarations
   - Add `src/ffi/faiss.rs` with moved FAISS bindings

2. **Phase 2**: Create safe wrappers
   - Add `src/safe/device.rs` with `GpuDevice` API
   - Add `src/safe/faiss.rs` with `GpuResources`, `GpuIndex` RAII wrappers

3. **Phase 3**: Update dependent crates
   - Modify context-graph-embeddings to use `context_graph_cuda::safe::GpuDevice`
   - Modify context-graph-graph to use `context_graph_cuda::safe::faiss`

4. **Phase 4**: Remove old FFI
   - Delete `extern "C"` blocks from context-graph-embeddings
   - Delete faiss_ffi module from context-graph-graph (or reduce to re-exports)

5. **Phase 5**: Add CI gate
   - Add `scripts/check-ffi-consolidation.sh`
   - Integrate into CI workflow

### Estimated Effort

| Task | Effort |
|------|--------|
| Create FFI modules | 2-3 hours |
| Create safe wrappers | 4-6 hours |
| Update embeddings crate | 2-3 hours |
| Update graph crate | 2-3 hours |
| Remove old FFI | 1-2 hours |
| Add CI gate | 1-2 hours |
| Testing and validation | 2-4 hours |
| **Total** | **14-23 hours** |

---

## Appendix: Detailed FFI Violations

### Violation 1: context-graph-embeddings/src/gpu/device/utils.rs

**Lines**: 29-50
**Functions**: 6 CUDA Driver API functions

```rust
extern "C" {
    fn cuInit(flags: std::os::raw::c_uint) -> i32;
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
    fn cuDeviceGetName(name: *mut std::os::raw::c_char, len: i32, dev: i32) -> i32;
    fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: i32) -> i32;
    fn cuDeviceGetAttribute(value: *mut i32, attrib: i32, dev: i32) -> i32;
    fn cuDriverGetVersion(version: *mut i32) -> i32;
}
```

**Usage**: GPU info queries for embeddings service initialization.

---

### Violation 2: context-graph-embeddings/src/warm/cuda_alloc/allocator_cuda.rs

**Lines**: 70-77 (inside function scope)
**Functions**: Same 6 CUDA Driver API functions (DUPLICATE)

```rust
extern "C" {
    fn cuInit(flags: std::os::raw::c_uint) -> i32;
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
    fn cuDeviceGetName(name: *mut std::os::raw::c_char, len: i32, dev: i32) -> i32;
    fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: i32) -> i32;
    fn cuDeviceGetAttribute(value: *mut i32, attrib: i32, dev: i32) -> i32;
    fn cuDriverGetVersion(version: *mut i32) -> i32;
}
```

**Usage**: VRAM allocator GPU info queries.
**Note**: This is a complete duplicate of Violation 1.

---

### Violation 3: context-graph-graph/src/index/faiss_ffi/bindings.rs

**Lines**: 26-188
**Functions**: 14 FAISS C API functions

```rust
#[link(name = "faiss_c")]
extern "C" {
    pub fn faiss_index_factory(...) -> c_int;
    pub fn faiss_StandardGpuResources_new(...) -> c_int;
    pub fn faiss_StandardGpuResources_free(...);
    pub fn faiss_index_cpu_to_gpu(...) -> c_int;
    pub fn faiss_Index_train(...) -> c_int;
    pub fn faiss_Index_is_trained(...) -> c_int;
    pub fn faiss_Index_add_with_ids(...) -> c_int;
    pub fn faiss_Index_search(...) -> c_int;
    pub fn faiss_IndexIVF_set_nprobe(...);
    pub fn faiss_Index_ntotal(...) -> c_long;
    pub fn faiss_write_index(...) -> c_int;
    pub fn faiss_read_index(...) -> c_int;
    pub fn faiss_Index_free(...);
    pub fn faiss_get_num_gpus(...) -> c_int;
}
```

**Usage**: FAISS GPU-accelerated vector search indexing.

---

### Existing Compliant FFI: context-graph-cuda

The following `extern "C"` blocks are COMPLIANT (already in the correct crate):

| File | Functions |
|------|-----------|
| `src/cone/ffi.rs:12` | `launch_cone_check`, etc. |
| `src/cone/gpu.rs:92` | `cuInit`, `cuDeviceGetCount` |
| `src/poincare/ffi.rs:12` | `launch_poincare_distance`, etc. |
| `src/poincare/kernel.rs:91` | `cuInit`, `cuDeviceGetCount` |

---

## References

- **Constitution v5.0.0**: Rule `"CUDA FFI only in context-graph-cuda"`
- **ISS-005**: CUDA FFI Scattered Across 3 Crates
- **PRD-ANALYSIS-REMEDIATION.md**: Architecture Domain Requirements
- **MASTER-ISSUES-REMEDIATION-PLAN.md**: Issue analysis and remediation phases
