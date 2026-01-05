# Module 7: CUDA Optimization - Atomic Tasks

```yaml
metadata:
  module_id: "module-07"
  module_name: "CUDA Optimization"
  version: "1.0.0"
  phase: 2
  total_tasks: 20
  approach: "inside-out-bottom-up"
  created: "2025-12-31"
  dependencies:
    - module-03-embedding-pipeline
    - module-06-bio-nervous-system
  estimated_duration: "3 weeks"
  target_hardware: "NVIDIA RTX 5090 (Blackwell)"
  cuda_version: "13.1"
  spec_refs:
    - FUNC-CUDA-007 (Functional)
    - TECH-CUDA-007 (Technical)
```

---

## Task Overview

This module implements GPU acceleration targeting RTX 5090 (32GB GDDR7) with CUDA 13.1. Tasks are organized in inside-out, bottom-up order:

1. **Foundation Layer** (Tasks 1-7): FFI bindings, error handling, configuration types
2. **Logic Layer** (Tasks 8-14): Memory pools, 8-stream management, Green Contexts
3. **Surface Layer** (Tasks 15-20): Custom kernels, kernel wrappers, integration tests

---

## Foundation Layer: FFI and Configuration

```yaml
tasks:
  # ============================================================
  # FOUNDATION: Error Handling and Configuration
  # ============================================================

  - id: "M07-T01"
    title: "Define CudaError Enum with Error Categories"
    description: |
      Implement CudaError struct and CudaErrorCategory enum for comprehensive error handling.
      CudaErrorCategory variants: Memory, Launch, Synchronization, Device, InvalidArgument, CuDnn, Unknown.
      CudaError fields: code (cudaError_t), category, message (String), context (Option<String>).
      Methods: from_code(code), categorize(code), is_recoverable().
      Recoverable categories: Memory, Synchronization.
      Use thiserror for derivation, include cuda_check! macro.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-cuda/src/error.rs"
    dependencies: []
    acceptance_criteria:
      - "CudaError struct with 4 fields compiles"
      - "CudaErrorCategory enum with 7 variants"
      - "from_code() retrieves error string via cudaGetErrorString FFI"
      - "categorize() maps error codes to categories correctly"
      - "is_recoverable() returns true for Memory and Synchronization"
      - "cuda_check! macro returns Err on non-zero code"
      - "CudaResult<T> type alias defined"
    test_file: "crates/context-graph-cuda/tests/error_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 8"

  - id: "M07-T02"
    title: "Define CUDA Runtime FFI Bindings"
    description: |
      Implement cuda_runtime FFI module with C bindings to CUDA runtime API.
      Types: cudaError_t (i32), cudaStream_t (*mut c_void), cudaEvent_t (*mut c_void),
             cudaMemPool_t (*mut c_void), cudaMemPoolProps struct.
      Memory Pool FFI: cudaMemPoolCreate, cudaMemPoolDestroy, cudaMallocAsync,
                       cudaFreeAsync, cudaMallocFromPoolAsync.
      Stream FFI: cudaStreamCreate, cudaStreamCreateWithPriority, cudaStreamDestroy,
                  cudaStreamSynchronize, cudaStreamWaitEvent.
      Event FFI: cudaEventCreate, cudaEventRecord, cudaEventSynchronize, cudaEventElapsedTime.
      Error FFI: cudaGetLastError, cudaPeekAtLastError, cudaGetErrorString.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-cuda/src/ffi/cuda_runtime.rs"
    dependencies:
      - "M07-T01"
    acceptance_criteria:
      - "All extern 'C' declarations compile"
      - "cudaMemPoolProps #[repr(C)] struct matches CUDA header"
      - "CUDA_SUCCESS constant = 0"
      - "#[link(name = 'cudart')] directive present"
      - "Types are Send + Sync where appropriate"
    test_file: "crates/context-graph-cuda/tests/ffi_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 3.1"

  - id: "M07-T03"
    title: "Define cuDNN FFI Bindings"
    description: |
      Implement cudnn FFI module with C bindings to cuDNN library.
      Types: cudnnHandle_t (*mut c_void), cudnnTensorDescriptor_t (*mut c_void),
             cudnnStatus_t (i32), cudnnDataType_t enum (FLOAT=0, HALF=2, BFLOAT16=9, FP8_E4M3=12).
      FFI functions: cudnnCreate, cudnnDestroy, cudnnSetStream, cudnnSoftmaxForward.
      CUDNN_STATUS_SUCCESS constant = 0.
    layer: "foundation"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-cuda/src/ffi/cudnn.rs"
    dependencies:
      - "M07-T02"
    acceptance_criteria:
      - "cudnnHandle_t and cudnnTensorDescriptor_t types defined"
      - "cudnnDataType_t enum with 4 variants"
      - "cudnnSoftmaxForward signature matches cuDNN API"
      - "#[link(name = 'cudnn')] directive present"
    test_file: "crates/context-graph-cuda/tests/cudnn_ffi_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 3.2"

  - id: "M07-T04"
    title: "Define cuBLAS FFI Bindings"
    description: |
      Implement cublas FFI module with C bindings to cuBLAS library.
      Types: cublasHandle_t (*mut c_void), cublasStatus_t (i32).
      FFI functions: cublasCreate, cublasDestroy, cublasSetStream,
                     cublasSgemm (single precision GEMM).
      CUBLAS_STATUS_SUCCESS constant = 0.
    layer: "foundation"
    priority: "medium"
    estimated_hours: 1.5
    file_path: "crates/context-graph-cuda/src/ffi/cublas.rs"
    dependencies:
      - "M07-T02"
    acceptance_criteria:
      - "cublasHandle_t type defined"
      - "cublasSgemm signature matches cuBLAS API"
      - "#[link(name = 'cublas')] directive present"
    test_file: "crates/context-graph-cuda/tests/cublas_ffi_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 3"

  - id: "M07-T05"
    title: "Define MemoryPoolConfig for cuMemAllocAsync"
    description: |
      Implement MemoryPoolConfig struct for CUDA memory pool configuration.
      Fields: initial_size (1GB default), max_size (24GB - leaves 8GB for system on 32GB card),
              release_threshold (0.5 default).
      Include validation ensuring max_size <= device memory - 8GB reserve.
      Derive Clone, Debug, Serialize, Deserialize.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1.5
    file_path: "crates/context-graph-cuda/src/memory.rs"
    dependencies: []
    acceptance_criteria:
      - "MemoryPoolConfig struct with 3 fields"
      - "Default returns initial_size=1GB, max_size=24GB, release_threshold=0.5"
      - "Validation method checks max_size constraint"
      - "Serde traits implemented"
    test_file: "crates/context-graph-cuda/tests/config_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 4"

  - id: "M07-T06"
    title: "Define StreamRole Enum and StreamManager Config"
    description: |
      Implement StreamRole enum for 8-stream role-based management.
      Roles: Embedding (0, high priority -1), Hopfield (1, high priority -1),
             Similarity (2, high priority -1), TransferH2D (3, normal priority 0),
             TransferD2H (4, normal priority 0), Graph (5, normal priority 0),
             CuDnn (6, normal priority 0), General (7, low priority 1).
      NUM_STREAMS constant = 8.
      Include priority() method returning stream priority for each role.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1
    file_path: "crates/context-graph-cuda/src/stream.rs"
    dependencies: []
    acceptance_criteria:
      - "StreamRole enum with 8 variants"
      - "NUM_STREAMS = 8 constant defined"
      - "priority() returns [-1, -1, -1, 0, 0, 0, 0, 1] for roles 0-7"
      - "Clone, Copy, PartialEq traits derived"
    test_file: "crates/context-graph-cuda/tests/stream_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 5"

  - id: "M07-T07"
    title: "Define GreenContextConfig for SM Partitioning"
    description: |
      Implement GreenContextConfig struct for CUDA 13.1 Green Context power management.
      Fields: idle_threshold_ms (50ms default), efficiency_target (0.85),
              min_active_ms (10ms minimum active time before transition).
      PowerState enum: Active, Throttled, Idle, Sleep.
      Include validate() method ensuring targets are in valid ranges.
    layer: "foundation"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-cuda/src/context.rs"
    dependencies: []
    acceptance_criteria:
      - "GreenContextConfig struct with 3 fields"
      - "PowerState enum with 4 variants"
      - "Default returns idle_threshold_ms=50, efficiency_target=0.85, min_active_ms=10"
      - "efficiency_target constrained to [0.0, 1.0]"
    test_file: "crates/context-graph-cuda/tests/context_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 7"

  # ============================================================
  # LOGIC LAYER: Memory and Stream Management
  # ============================================================

  - id: "M07-T08"
    title: "Implement MemoryPool with cuMemAllocAsync"
    description: |
      Implement MemoryPool struct wrapping CUDA memory pool for async allocation.
      Fields: handle (cudaMemPool_t), config (MemoryPoolConfig),
              allocations (Mutex<HashMap<*mut c_void, usize>>).
      Methods: new(config, device_id), alloc_async(size, stream), free_async(ptr, stream).
      Uses cudaMallocFromPoolAsync for allocation, cudaFreeAsync for deallocation.
      Track allocations for debugging and leak detection.
      Performance target: <5us per allocation.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-cuda/src/memory.rs"
    dependencies:
      - "M07-T02"
      - "M07-T05"
    acceptance_criteria:
      - "new() creates cudaMemPool with config-derived props"
      - "alloc_async() returns valid device pointer"
      - "free_async() releases memory back to pool"
      - "allocations HashMap tracks active allocations"
      - "Drop impl calls cudaMemPoolDestroy"
      - "Send + Sync implemented (unsafe)"
      - "Allocation latency <5us verified"
    test_file: "crates/context-graph-cuda/tests/memory_pool_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 4"
      - "REQ-CUDA-019, REQ-CUDA-020"

  - id: "M07-T09"
    title: "Implement StreamManager for 8-Stream Parallelism"
    description: |
      Implement StreamManager struct managing 8 concurrent CUDA streams.
      Fields: streams ([cudaStream_t; 8]), events ([cudaEvent_t; 8]).
      Methods: new(), get_stream(role), create_dependency(source, target),
               sync_stream(role), sync_all().
      Uses cudaStreamCreateWithPriority for priority-aware stream creation.
      Uses cudaEventRecord + cudaStreamWaitEvent for dependencies.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-cuda/src/stream.rs"
    dependencies:
      - "M07-T02"
      - "M07-T06"
    acceptance_criteria:
      - "new() creates 8 streams with correct priorities"
      - "get_stream() returns stream for role"
      - "create_dependency() inserts event wait between streams"
      - "sync_stream() calls cudaStreamSynchronize"
      - "sync_all() synchronizes all 8 streams"
      - "Drop impl destroys all streams and events"
    test_file: "crates/context-graph-cuda/tests/stream_manager_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 5"
      - "REQ-CUDA-003, REQ-CUDA-026, REQ-CUDA-027"

  - id: "M07-T10"
    title: "Implement GreenContext for Power Efficiency"
    description: |
      Implement GreenContext struct for CUDA 13.1 power state management.
      Fields: config (GreenContextConfig), power_state (AtomicU64),
              last_activity (AtomicU64), active_kernels (AtomicU64).
      Methods: kernel_start(), kernel_end(), power_state(), transition_to(state),
               consider_power_transition(), efficiency_metrics().
      Uses atomic operations for thread-safe state tracking.
      Auto-transitions to Idle when no kernels active for idle_threshold_ms.
    layer: "logic"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-cuda/src/context.rs"
    dependencies:
      - "M07-T07"
    acceptance_criteria:
      - "kernel_start() increments active_kernels, sets Active state"
      - "kernel_end() decrements active_kernels, considers transition"
      - "consider_power_transition() transitions to Idle after threshold"
      - "efficiency_metrics() returns PowerEfficiencyMetrics struct"
      - "All operations are lock-free (atomics only)"
      - "Power reduction >20% vs baseline verified"
    test_file: "crates/context-graph-cuda/tests/green_context_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 7"
      - "REQ-CUDA-033"

  - id: "M07-T11"
    title: "Implement check_last_error Helper"
    description: |
      Implement check_last_error() function for kernel launch validation.
      Calls cudaGetLastError() and returns CudaError if non-zero.
      Include synchronous and asynchronous variants.
      check_last_error_sync() also calls cudaDeviceSynchronize first.
    layer: "logic"
    priority: "high"
    estimated_hours: 1
    file_path: "crates/context-graph-cuda/src/error.rs"
    dependencies:
      - "M07-T01"
      - "M07-T02"
    acceptance_criteria:
      - "check_last_error() returns Ok(()) on CUDA_SUCCESS"
      - "check_last_error() returns Err(CudaError) on error"
      - "check_last_error_sync() synchronizes before checking"
      - "Both are pub functions"
    test_file: "crates/context-graph-cuda/tests/error_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 8"

  - id: "M07-T12"
    title: "Implement Device Detection and Initialization"
    description: |
      Implement device module for RTX 5090 detection and initialization.
      Functions: detect_device(device_id), verify_compute_capability(major, minor),
                 get_device_properties(), init_device(device_id).
      Verify compute capability 10.0 (sm_100) for RTX 5090 Blackwell.
      Return device properties: name, total_memory (32GB), multiprocessor_count.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-cuda/src/device.rs"
    dependencies:
      - "M07-T01"
      - "M07-T02"
    acceptance_criteria:
      - "detect_device() returns Ok if CUDA device exists"
      - "verify_compute_capability() checks major >= 10 for Blackwell"
      - "get_device_properties() returns device name and memory"
      - "init_device() sets current device context"
      - "Returns error if no compatible GPU found"
    test_file: "crates/context-graph-cuda/tests/device_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 1"
      - "REQ-CUDA-001, REQ-CUDA-004"

  - id: "M07-T13"
    title: "Write common.cuh CUDA Header"
    description: |
      Create common.cuh header with shared CUDA kernel utilities.
      Defines: BLOCK_SIZE (256), WARP_SIZE (32), FUSED_OUTPUT_DIM (1536), HOPFIELD_DIM (1536).
      Utility macros: CUDA_CHECK, DIV_CEIL(a, b).
      Inline device functions: warp_reduce_sum, warp_reduce_max, block_reduce_sum.
      Include guards and extern "C" compatibility.
    layer: "logic"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-cuda/kernels/common.cuh"
    dependencies: []
    acceptance_criteria:
      - "Header compiles with nvcc"
      - "BLOCK_SIZE = 256 defined"
      - "warp_reduce_sum uses __shfl_down_sync"
      - "block_reduce_sum uses shared memory"
      - "Include guards prevent multiple inclusion"
    test_file: "N/A (header file)"
    spec_refs:
      - "TECH-CUDA-007 Section 6"

  - id: "M07-T14"
    title: "Configure Cargo.toml and build.rs for CUDA Compilation"
    description: |
      Create Cargo.toml and build.rs for CUDA kernel compilation.
      Cargo.toml: features (cuda-13, cudnn, cublas), build-dependencies (cc with parallel).
      build.rs: Compile .cu files with nvcc, flags: -gencode arch=compute_100,code=sm_100,
                -O3, --use_fast_math. Link cudart, cudnn, cublas.
      Environment variable CUDA_PATH for CUDA toolkit location.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-cuda/Cargo.toml"
    dependencies:
      - "M07-T13"
    acceptance_criteria:
      - "Cargo.toml defines cuda-13, cudnn, cublas features"
      - "build.rs compiles all .cu files in kernels/ directory"
      - "nvcc flags target sm_100 (RTX 5090)"
      - "cargo:rustc-link-lib directives for cudart, cudnn, cublas"
      - "CUDA_PATH environment variable supported"
    test_file: "N/A (build configuration)"
    spec_refs:
      - "TECH-CUDA-007 Section 11"

  # ============================================================
  # SURFACE LAYER: Custom CUDA Kernels
  # ============================================================

  - id: "M07-T15"
    title: "Implement embedding_fuse CUDA Kernel"
    description: |
      Implement embedding_fuse.cu kernel for gated mixture-of-experts embedding fusion.
      Input: embeddings[batch, 12, max_dim], dimensions[12], gating_weights[12, 1536],
             expert_projs[12, model_dim, 1536].
      Output: output[batch, 1536].
      Algorithm: Softmax gating per output dimension, weighted sum of expert projections.
      Use shared memory for gating weights, __syncthreads for coordination.
      Grid: (batch_size, ceil(1536/BLOCK_SIZE)), Block: 256.
      Performance target: <10us launch latency, >100k samples/sec throughput.
    layer: "surface"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-cuda/kernels/embedding_fuse.cu"
    dependencies:
      - "M07-T13"
    acceptance_criteria:
      - "Kernel compiles with nvcc"
      - "Shared memory used for s_gates[12]"
      - "Softmax computed in shared memory"
      - "launch_embedding_fuse extern 'C' wrapper defined"
      - "Output matches CPU reference within 1e-5"
      - "Launch latency <10us on RTX 5090"
    test_file: "crates/context-graph-cuda/tests/kernel_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 6.1"
      - "REQ-CUDA-006, REQ-CUDA-008, REQ-CUDA-009"

  - id: "M07-T16"
    title: "Implement hopfield_update CUDA Kernel"
    description: |
      Implement hopfield_update.cu kernel for Modern Hopfield Network retrieval.
      Input: patterns[num_patterns, dim], query[batch, dim], beta (inverse temperature).
      Output: output[batch, dim], attention[batch, num_patterns].
      Algorithm:
      1. Load query to shared memory
      2. Compute attention scores: score = beta * dot(query, pattern)
      3. Stable softmax with max reduction
      4. Weighted pattern retrieval
      Grid: batch_size, Block: 256.
      Performance target: <10us latency, >500k patterns/sec.
    layer: "surface"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-cuda/kernels/hopfield_update.cu"
    dependencies:
      - "M07-T13"
    acceptance_criteria:
      - "Kernel compiles with nvcc"
      - "Shared memory for s_query[HOPFIELD_DIM] and s_attn[1024]"
      - "Stable softmax (subtract max before exp)"
      - "launch_hopfield_update extern 'C' wrapper defined"
      - "Output matches CPU reference within 1e-5"
      - "Launch latency <10us on RTX 5090"
    test_file: "crates/context-graph-cuda/tests/kernel_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 6.2"
      - "REQ-CUDA-010, REQ-CUDA-011, REQ-CUDA-012"

  - id: "M07-T17"
    title: "Implement similarity_batch CUDA Kernel"
    description: |
      Implement similarity_batch.cu kernel for batch cosine similarity computation.
      Input: queries[num_queries, dim], corpus[corpus_size, dim],
             query_norms[num_queries], corpus_norms[corpus_size].
      Output: similarities[num_queries, corpus_size].
      Algorithm: Tiled matrix multiplication with TILE_SIZE=32.
      Use shared memory for query and corpus tiles.
      Similarity = dot / (query_norm * corpus_norm + 1e-8).
      Grid: (ceil(corpus_size/32), ceil(num_queries/32)), Block: (32, 32).
      Performance target: <10us latency, >10M pairs/sec.
    layer: "surface"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-cuda/kernels/similarity_batch.cu"
    dependencies:
      - "M07-T13"
    acceptance_criteria:
      - "Kernel compiles with nvcc"
      - "TILE_SIZE = 32 for optimal occupancy"
      - "Shared memory for s_queries[32][33] and s_corpus[32][33] (padding for bank conflicts)"
      - "launch_similarity_batch extern 'C' wrapper defined"
      - "Output matches CPU reference within 1e-5"
      - "Throughput >10M pairs/sec on RTX 5090"
    test_file: "crates/context-graph-cuda/tests/kernel_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 6.3"

  - id: "M07-T18"
    title: "Implement EmbeddingFuseKernel Rust Wrapper"
    description: |
      Implement EmbeddingFuseKernel struct wrapping embedding_fuse CUDA kernel.
      Extern declaration: launch_embedding_fuse(embeddings, dimensions, gating_weights,
                          expert_projs, output, batch_size, stream).
      Methods: launch(embeddings, dimensions, gating_weights, expert_projs, output, batch_size, stream).
      Calls launch_embedding_fuse, then check_last_error().
      Uses GreenContext for power tracking.
    layer: "surface"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-cuda/src/kernels/embedding_fuse.rs"
    dependencies:
      - "M07-T10"
      - "M07-T11"
      - "M07-T15"
    acceptance_criteria:
      - "extern 'C' declaration for launch_embedding_fuse"
      - "launch() method is pub and returns CudaResult<()>"
      - "Calls green_context.kernel_start() before launch"
      - "Calls green_context.kernel_end() after launch"
      - "check_last_error() validates kernel success"
    test_file: "crates/context-graph-cuda/tests/wrapper_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 9"

  - id: "M07-T19"
    title: "Implement HopfieldUpdateKernel and SimilarityBatchKernel Rust Wrappers"
    description: |
      Implement HopfieldUpdateKernel and SimilarityBatchKernel structs.
      HopfieldUpdateKernel: launch(patterns, query, output, attention, num_patterns, dim, batch_size, beta, stream).
      SimilarityBatchKernel: launch(queries, corpus, similarities, query_norms, corpus_norms,
                                    num_queries, corpus_size, dim, stream).
      Both integrate with GreenContext and check_last_error.
    layer: "surface"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-cuda/src/kernels/mod.rs"
    dependencies:
      - "M07-T10"
      - "M07-T11"
      - "M07-T16"
      - "M07-T17"
    acceptance_criteria:
      - "HopfieldUpdateKernel struct with launch() method"
      - "SimilarityBatchKernel struct with launch() method"
      - "Both use GreenContext for power tracking"
      - "Both return CudaResult<()>"
      - "kernel mod.rs exports all three kernel wrappers"
    test_file: "crates/context-graph-cuda/tests/wrapper_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 9"

  - id: "M07-T20"
    title: "Create Module Integration Tests and Benchmarks"
    description: |
      Implement comprehensive integration tests for Module 7:
      - Device detection and initialization (RTX 5090 compute capability 10.0)
      - Memory pool allocation/deallocation cycle (<5us per alloc)
      - 8-stream concurrent execution with dependency tracking
      - Green Context power state transitions
      - embedding_fuse kernel correctness and performance (<10us, >100k/sec)
      - hopfield_update kernel correctness and performance (<10us, >500k/sec)
      - similarity_batch kernel correctness and performance (>10M pairs/sec)
      - CPU vs GPU result comparison (tolerance 1e-5)
      Performance benchmarks against all NFR targets.
      Tests requiring GPU marked with #[requires_gpu] for CI.
    layer: "surface"
    priority: "critical"
    estimated_hours: 6
    file_path: "crates/context-graph-cuda/tests/integration_tests.rs"
    dependencies:
      - "M07-T08"
      - "M07-T09"
      - "M07-T10"
      - "M07-T12"
      - "M07-T18"
      - "M07-T19"
    acceptance_criteria:
      - "Device init test verifies compute capability >= 10.0"
      - "Memory pool test measures allocation latency <5us"
      - "Stream test verifies 8 concurrent streams"
      - "Green Context test verifies power state transitions"
      - "Kernel tests verify correctness within 1e-5 tolerance"
      - "Performance benchmarks verify all latency/throughput targets"
      - "H2D/D2H bandwidth test verifies >25 GB/s (PCIe 5.0)"
      - "#[requires_gpu] attribute for CI skip on non-GPU"
    test_file: "crates/context-graph-cuda/tests/integration_tests.rs"
    spec_refs:
      - "TECH-CUDA-007 Section 10"
      - "All REQ-CUDA requirements"
```

---

## Dependency Graph

```
M07-T01 (CudaError) ─────────────────────────────────────────────────────────────┐
        │                                                                         │
        └──► M07-T02 (CUDA FFI) ──┬──► M07-T03 (cuDNN FFI)                        │
                                  │                                               │
                                  ├──► M07-T04 (cuBLAS FFI)                       │
                                  │                                               │
                                  ├──► M07-T08 (MemoryPool) ◄── M07-T05 (PoolConfig)
                                  │                                               │
                                  ├──► M07-T09 (StreamManager) ◄── M07-T06 (StreamRole)
                                  │                                               │
                                  ├──► M07-T11 (check_last_error)                │
                                  │                                               │
                                  └──► M07-T12 (Device Detection)                │
                                                                                  │
M07-T07 (GreenContextConfig) ──► M07-T10 (GreenContext) ─────────────────────────┤
                                                                                  │
M07-T13 (common.cuh) ──┬──► M07-T15 (embedding_fuse.cu) ──► M07-T18 (Wrapper) ───┤
                       │                                                          │
                       ├──► M07-T16 (hopfield_update.cu) ──► M07-T19 (Wrappers) ──┤
                       │                                                          │
                       └──► M07-T17 (similarity_batch.cu) ──────────────────────► │
                                                                                  │
M07-T14 (build.rs) ◄── M07-T13                                                   │
                                                                                  │
M07-T08 + M07-T09 + M07-T10 + M07-T12 + M07-T18 + M07-T19 ──► M07-T20 (Integration)
```

---

## Implementation Order (Recommended)

### Week 1: Foundation and FFI
1. M07-T01: CudaError enum and error handling
2. M07-T02: CUDA Runtime FFI bindings
3. M07-T03: cuDNN FFI bindings
4. M07-T04: cuBLAS FFI bindings
5. M07-T05: MemoryPoolConfig
6. M07-T06: StreamRole enum
7. M07-T07: GreenContextConfig

### Week 2: Memory, Streams, and Build
8. M07-T08: MemoryPool implementation
9. M07-T09: StreamManager implementation
10. M07-T10: GreenContext implementation
11. M07-T11: check_last_error helper
12. M07-T12: Device detection
13. M07-T13: common.cuh header
14. M07-T14: Cargo.toml and build.rs

### Week 3: Kernels and Integration
15. M07-T15: embedding_fuse CUDA kernel
16. M07-T16: hopfield_update CUDA kernel
17. M07-T17: similarity_batch CUDA kernel
18. M07-T18: EmbeddingFuseKernel wrapper
19. M07-T19: HopfieldUpdateKernel and SimilarityBatchKernel wrappers
20. M07-T20: Integration tests and benchmarks

---

## Quality Gates

| Gate | Criteria | Required For |
|------|----------|--------------|
| Foundation Complete | M07-T01 through M07-T07 pass all tests | Week 2 start |
| Runtime Operational | M07-T08 through M07-T14 pass all tests | Week 3 start |
| Kernels Verified | M07-T15 through M07-T19 pass all tests | Integration |
| Module Complete | All 20 tasks complete, GPU benchmarks pass | Module 8 start |

---

## Performance Targets Summary

| Operation | Target | Conditions |
|-----------|--------|------------|
| Memory pool allocation | <5us | Cached allocation |
| embedding_fuse launch | <10us | Single batch |
| embedding_fuse throughput | >100k samples/sec | batch_size=64 |
| hopfield_update launch | <10us | Single batch |
| hopfield_update throughput | >500k patterns/sec | 10K patterns |
| similarity_batch throughput | >10M pairs/sec | 1K x 1K |
| H2D bandwidth | >25 GB/s | PCIe 5.0 |
| D2H bandwidth | >25 GB/s | PCIe 5.0 |
| Stream overlap efficiency | >90% | 8 streams |
| GPU utilization | >85% | Mixed workload |
| Green Context efficiency | >85% | Active utilization |
| Power reduction | >20% | vs baseline |

---

## Memory Budget (RTX 5090 32GB GDDR7)

| Component | Budget |
|-----------|--------|
| Memory pool (max) | 24GB |
| System reserve | 8GB |
| **Total VRAM** | **32GB** |

---

## Hardware Specifications Reference

| Specification | RTX 5090 (Blackwell) |
|---------------|---------------------|
| Architecture | Blackwell |
| Compute Capability | 10.0 (sm_100) |
| CUDA Cores | 21,760 |
| Tensor Cores | 680 (5th gen) |
| Memory | 32GB GDDR7 |
| Memory Bandwidth | 1,792 GB/s |
| FP32 TFLOPS | 109.6 |
| FP16 TFLOPS | 219.2 |
| Tensor FP16 TFLOPS | 875.2 |
| TDP | 575W |

---

## CUDA 13.1 Features Used

| Feature | Usage |
|---------|-------|
| Green Contexts | Power efficiency via SM partitioning |
| cuMemAllocAsync | Memory pools for <5us allocation |
| cudaGraph | Graph execution optimization (future) |
| WMMA API | Tensor Core operations |
| FP8 Support | FP8 inference precision (future) |

---

## Critical Constraints

**RTX 5090 REQUIRED**: All GPU tests require RTX 5090 or compatible Blackwell GPU.
- Tests marked `#[requires_gpu]` for CI skip on non-GPU machines
- Compute capability 10.0 minimum verified at initialization

**8 STREAMS MANDATORY**: Per spec, exactly 8 CUDA streams must be used.
- Stream roles are fixed and priority-assigned
- Dependencies tracked via events

**GREEN CONTEXTS**: CUDA 13.1 Green Contexts for power efficiency.
- Kernel start/end hooks for power state tracking
- Auto-transition to Idle after idle threshold

---

*Generated: 2025-12-31*
*Module: 07 - CUDA Optimization*
*Version: 1.0.0*
*Total Tasks: 20*
