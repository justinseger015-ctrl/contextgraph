# TASK-01: VERIFICATION - context-graph-cuda Crate Skeleton

```yaml
id: TASK-01
original_id: TASK-ARCH-001
version: 2.0.0
updated: 2026-01-12
status: COMPLETE
layer: foundation
sequence: 1
depends_on: []
blocks: [TASK-02, TASK-03]
estimated_hours: 0.5
```

## STATUS: ALREADY IMPLEMENTED

**This task has been completed with a DIFFERENT architecture than originally specified.**

The `context-graph-cuda` crate exists and compiles. However:
- Original spec called for `src/ffi/mod.rs` and `src/safe/mod.rs` structure
- Actual implementation uses `src/cone/` and `src/poincare/` with FFI inside each

**This task now requires VERIFICATION ONLY - no code changes needed.**

---

## CURRENT STATE AUDIT (2026-01-12)

### Crate Structure (ACTUAL)
```
crates/context-graph-cuda/
├── Cargo.toml                    # EXISTS - correct workspace config
├── build.rs                      # EXISTS - CUDA kernel compilation
├── kernels/
│   ├── cone_check.cu            # EXISTS - CUDA kernel source
│   └── poincare_distance.cu     # EXISTS - CUDA kernel source
├── src/
│   ├── lib.rs                   # EXISTS - module exports
│   ├── error.rs                 # EXISTS - CudaError, CudaResult types
│   ├── ops.rs                   # EXISTS - VectorOps trait
│   ├── stub.rs                  # EXISTS - test-only stubs (gated)
│   ├── cone/                    # EXISTS - cone operations
│   │   ├── mod.rs
│   │   ├── ffi.rs              # FFI bindings for cone kernel
│   │   ├── cpu.rs              # CPU fallback impl
│   │   ├── gpu.rs              # GPU impl
│   │   ├── config.rs           # ConeCudaConfig
│   │   ├── types.rs            # ConeData
│   │   ├── constants.rs
│   │   └── tests.rs
│   └── poincare/                # EXISTS - poincare distance
│       ├── mod.rs
│       ├── ffi.rs              # FFI bindings for poincare kernel
│       ├── cpu.rs              # CPU fallback impl
│       ├── gpu.rs              # GPU impl
│       ├── config.rs           # PoincareCudaConfig
│       ├── kernel.rs
│       ├── constants.rs
│       └── tests/
└── tests/                        # EXISTS - integration tests
    ├── cuda_cone_test.rs
    └── cuda_poincare_test.rs
```

### Cargo.toml (ACTUAL)
```toml
[package]
name = "context-graph-cuda"
version = "0.1.0"
edition = "2021"

[dependencies]
context-graph-core = { path = "../context-graph-core" }
tokio = { workspace = true }
serde = { workspace = true }
thiserror = { workspace = true }
async-trait = { workspace = true }
tracing = { workspace = true }

[features]
default = ["cuda"]
cuda = []
```

### Key Differences from Original Spec
| Original Spec | Actual Implementation | Impact |
|---------------|----------------------|--------|
| `src/ffi/mod.rs` | FFI in `cone/ffi.rs`, `poincare/ffi.rs` | Structure differs, same function |
| `src/safe/mod.rs` | Safe wrappers in `cone/gpu.rs`, `poincare/gpu.rs` | Structure differs, same function |
| `GpuError` in `error.rs` | `CudaError` in `error.rs` | Name differs, same purpose |
| Dependencies: `libc`, `nvml-wrapper` | Dependencies: `context-graph-core`, `tokio`, etc. | Uses Candle/cudarc instead |

---

## TASK REQUIREMENTS (VERIFICATION ONLY)

### Accept Criteria
All criteria are already met. Verify each:

1. **Crate exists in workspace**
   ```bash
   grep -q "context-graph-cuda" Cargo.toml && echo "PASS" || echo "FAIL"
   ```
   Expected: PASS

2. **Crate compiles**
   ```bash
   cargo check -p context-graph-cuda
   ```
   Expected: Compiles with no errors

3. **Tests pass**
   ```bash
   cargo test -p context-graph-cuda
   ```
   Expected: All tests pass (18 unit + 14 doc tests)

4. **Metadata correct**
   ```bash
   cargo metadata --format-version=1 | jq '.packages[] | select(.name=="context-graph-cuda") | {name, version}'
   ```
   Expected: `{"name": "context-graph-cuda", "version": "0.1.0"}`

---

## FULL STATE VERIFICATION PROTOCOL

### Source of Truth
The source of truth is the compiled Rust crate metadata and test results.

### Verification Steps

#### Step 1: Execute Compilation Check
```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-cuda 2>&1 | tee /tmp/cuda-check.log
echo "Exit code: $?"
```

**Expected Output**:
- Build succeeds (exit code 0)
- CUDA kernels compile: `Successfully compiled CUDA kernel: kernels/*.cu`
- No compiler errors

**Evidence Required**: Capture full output showing `Finished` status.

#### Step 2: Execute Tests
```bash
cargo test -p context-graph-cuda 2>&1 | tee /tmp/cuda-tests.log
echo "Exit code: $?"
```

**Expected Output**:
- `test result: ok. 18 passed; 0 failed`
- `test result: ok. 14 passed; 0 failed; 2 ignored` (doc tests)

**Evidence Required**: Full test output with pass counts.

#### Step 3: Verify Workspace Membership
```bash
cargo metadata --format-version=1 2>/dev/null | jq '.packages[] | select(.name=="context-graph-cuda") | {name, version, dependencies: [.dependencies[].name]}' | tee /tmp/cuda-metadata.json
```

**Expected Output**:
```json
{
  "name": "context-graph-cuda",
  "version": "0.1.0",
  "dependencies": ["async-trait", "context-graph-core", "serde", "thiserror", "tokio", "tracing", "context-graph-core", "tokio-test"]
}
```

**Evidence Required**: JSON output showing correct package info.

---

## BOUNDARY & EDGE CASE AUDIT

### Edge Case 1: GPU Not Available
**Input**: Run tests on system without CUDA GPU
**Before State**: GPU check should fail gracefully
**Action**:
```bash
cargo test -p context-graph-cuda -- --test-threads=1 2>&1
```
**Expected After State**: Tests should pass (they use CPU fallbacks for testing)
**Verification**: Check test output for `test_integration_gpu_availability_check ... ok`

### Edge Case 2: Invalid CUDA Kernel
**Input**: Malformed CUDA kernel source
**Before State**: build.rs attempts kernel compilation
**Expected After State**: Build fails fast with clear NVCC error
**How to Test**: Temporarily modify `kernels/cone_check.cu` with syntax error, run `cargo build -p context-graph-cuda`
**Verification**: Build error must include NVCC error message, not silent failure

### Edge Case 3: Missing cuda-toolkit
**Input**: System without CUDA toolkit installed
**Before State**: build.rs runs
**Expected After State**: Clear error: `Could not find CUDA toolkit` or similar
**Note**: This is environment-dependent; document expected behavior

---

## BLOCKING ISSUES FOR DOWNSTREAM TASKS

### CRITICAL: FFI NOT CONSOLIDATED

**TASK-02 and TASK-03 depend on TASK-01 being a skeleton to consolidate into.**

Current reality:
- CUDA FFI exists in MULTIPLE places:
  - `crates/context-graph-cuda/src/cone/ffi.rs` (Cone kernel FFI)
  - `crates/context-graph-cuda/src/poincare/ffi.rs` (Poincare kernel FFI)
  - `crates/context-graph-embeddings/src/gpu/device/utils.rs` (CUDA Driver API FFI) **NOT CONSOLIDATED**
  - `crates/context-graph-embeddings/src/warm/cuda_alloc/allocator_cuda.rs` (cuMemAlloc FFI) **NOT CONSOLIDATED**
- FAISS FFI exists outside context-graph-cuda:
  - `crates/context-graph-graph/src/index/faiss_ffi/bindings.rs` **NOT CONSOLIDATED**

**Action for TASK-02/TASK-03**:
- Must ADD new `src/ffi/` directory to context-graph-cuda
- Move scattered FFI from other crates INTO context-graph-cuda
- Update imports in source crates to use context-graph-cuda

---

## MANUAL TESTING CHECKLIST

For an AI agent verifying this task is complete:

```
[ ] 1. Ran `cargo check -p context-graph-cuda` - PASSED
[ ] 2. Ran `cargo test -p context-graph-cuda` - 18 passed, 0 failed
[ ] 3. Ran `cargo metadata` - crate is in workspace
[ ] 4. Verified lib.rs exports: cone, poincare, error, ops modules
[ ] 5. Verified build.rs compiles CUDA kernels
[ ] 6. Documented that FFI consolidation is NOT done (blocking TASK-02/03)
```

---

## EVIDENCE LOG (Captured 2026-01-12)

### Compilation Check Output
```
$ cargo check -p context-graph-cuda
warning: context-graph-cuda@0.1.0: Running: "/usr/local/cuda-13.1/bin/nvcc" "-c" "-O3" "-arch" "sm_120" ...
warning: context-graph-cuda@0.1.0: Successfully compiled CUDA kernel: kernels/poincare_distance.cu
warning: context-graph-cuda@0.1.0: Successfully compiled CUDA kernel: kernels/cone_check.cu
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.10s

EXIT CODE: 0 (SUCCESS)
```

### Test Results
```
$ cargo test -p context-graph-cuda

running 54 tests (unit tests)
test result: ok. 54 passed; 0 failed; 0 ignored

running 34 tests (cuda_cone_test.rs)
test result: ok. 34 passed; 0 failed; 0 ignored

running 18 tests (cuda_poincare_test.rs)
test result: ok. 18 passed; 0 failed; 0 ignored

running 16 tests (doc tests)
test result: ok. 14 passed; 0 failed; 2 ignored

TOTAL: 120 tests, 120 passed, 0 failed, 2 ignored
EXIT CODE: 0 (SUCCESS)
```

### Workspace Membership
```
$ grep -q "context-graph-cuda" Cargo.toml && echo "PASS"
PASS: Crate in workspace
```

---

## CONSTITUTION COMPLIANCE

| Rule | Status | Notes |
|------|--------|-------|
| AP-07: No CPU fallbacks in production | PARTIAL | stub.rs is test-only (#[cfg(test)]) |
| AP-08: CUDA FFI only in context-graph-cuda | VIOLATED | FFI still exists in embeddings + graph crates |
| ARCH-08: CUDA GPU required for production | COMPLIANT | Feature-gated, no silent fallbacks |

---

## NEXT STEPS

1. **Mark TASK-01 as COMPLETE** (crate exists, compiles, tests pass)
2. **Proceed to TASK-02**: Consolidate CUDA driver FFI from:
   - `crates/context-graph-embeddings/src/gpu/device/utils.rs` (lines 29-50)
   - `crates/context-graph-embeddings/src/warm/cuda_alloc/allocator_cuda.rs` (line 70+)
3. **Proceed to TASK-03**: Consolidate FAISS FFI from:
   - `crates/context-graph-graph/src/index/faiss_ffi/bindings.rs` (entire file)

---

*Task Version 2.0.0 - Updated 2026-01-12 - Reflects actual implementation state*
