# Module 8: GPU Direct Storage - Atomic Tasks

```yaml
metadata:
  module: Module 8 - GPU Direct Storage
  phase: 7
  approach: inside-out-bottom-up
  spec_refs:
    - SPEC-GDS (Functional)
    - TECH-GDS-001 (Technical)
  total_tasks: 20
  created: 2025-12-31
  performance_targets:
    read_throughput: 4 GB/s
    write_throughput: 2 GB/s
    latency: < 100 microseconds (4KB aligned)
  dependencies:
    - Module 7 (CUDA Optimization)
  layers:
    foundation: 7 tasks
    logic: 8 tasks
    surface: 5 tasks
```

---

## Foundation Layer Tasks (Build First)

These tasks establish FFI bindings, core types, and configuration structures for GPU Direct Storage.

```yaml
- id: TASK-GDS-001
  title: Create GDS module structure with feature flags
  type: implementation
  layer: foundation
  requirement_refs: [REQ-GDS-001, REQ-GDS-002]
  dependencies: [Module 7 CUDA infrastructure]
  acceptance_criteria:
    - src/gds/mod.rs module entry with feature-gated exports
    - Cargo.toml includes cufile dependency under [dependencies.cufile]
    - Feature flag "gds" enables GPU Direct Storage compilation
    - Module compiles with and without gds feature
    - Re-exports all public types from submodules
    - cargo check -p context-graph-cuda --features gds succeeds
  estimated_complexity: low
  files_affected:
    - crates/context-graph-cuda/Cargo.toml
    - crates/context-graph-cuda/src/gds/mod.rs
    - crates/context-graph-cuda/src/lib.rs

- id: TASK-GDS-002
  title: Implement GDSConfig and GDSBufferPoolConfig structures
  type: implementation
  layer: foundation
  requirement_refs: [REQ-GDS-001, REQ-GDS-003, REQ-GDS-004]
  dependencies: [TASK-GDS-001]
  acceptance_criteria:
    - GDSConfig struct with all fields (enabled, nvme_devices, buffer_size, max_concurrent_ops, etc.)
    - Default impl sets buffer_size = 1GB, max_concurrent_ops = 64
    - GDSBufferPoolConfig struct with total_size, num_buffers, alignment, pin_buffers
    - Default impl sets total_size = 2GB, num_buffers = 16, alignment = 4096
    - validate() method checks buffer_size is power of 2 and >= 64KB
    - validate() ensures alignment is power of 2
    - PrefetchConfig struct with auto_prefetch, trigger_threshold, max_queue_depth, priority
    - PrefetchPriority enum with Background, Normal, High, Critical variants
    - Unit tests for validation logic and default values
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-cuda/src/gds/config.rs

- id: TASK-GDS-003
  title: Implement cuFile FFI type definitions
  type: implementation
  layer: foundation
  requirement_refs: [REQ-GDS-012, REQ-GDS-013]
  dependencies: [TASK-GDS-001]
  acceptance_criteria:
    - CUfileHandle, CUfileDescr, CUdeviceptr, CUstream type aliases
    - CUfileStatus enum with all error codes (Success, DriverNotInitialized, InvalidValue, etc.)
    - CUfileOpError struct for batch operation errors
    - CUfileOpenFlags struct with RDONLY, WRONLY, RDWR, CREATE, DIRECT constants
    - CUfileBatchIOParams struct for batch operations
    - CUfileDriverProps struct for driver properties
    - All types are repr(C) for FFI compatibility
    - Unit tests verify struct layout matches C ABI
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-cuda/src/gds/ffi/types.rs

- id: TASK-GDS-004
  title: Implement cuFile FFI function declarations
  type: implementation
  layer: foundation
  requirement_refs: [REQ-GDS-012, REQ-GDS-013, REQ-GDS-014]
  dependencies: [TASK-GDS-003]
  acceptance_criteria:
    - extern "C" block with #[link(name = "cufile")]
    - cuFileDriverOpen() and cuFileDriverClose() declarations
    - cuFileDriverGetProperties() and cuFileDriverSetPollMode() declarations
    - cuFileBufRegister() and cuFileBufDeregister() declarations
    - cuFileHandleRegister() and cuFileHandleDeregister() declarations
    - cuFileRead() and cuFileWrite() synchronous I/O declarations
    - cuFileBatchIOSetUp(), cuFileBatchIOSubmit(), cuFileBatchIOGetStatus(), cuFileBatchIODestroy()
    - cuFileReadAsync() and cuFileWriteAsync() stream-ordered declarations
    - All return types and parameters match NVIDIA cuFile API exactly
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-cuda/src/gds/ffi/functions.rs
    - crates/context-graph-cuda/src/gds/ffi/mod.rs

- id: TASK-GDS-005
  title: Implement safe Rust wrappers for cuFile driver and handles
  type: implementation
  layer: foundation
  requirement_refs: [REQ-GDS-012, REQ-GDS-015, REQ-GDS-016]
  dependencies: [TASK-GDS-004]
  acceptance_criteria:
    - init_driver() safely initializes cuFile driver with AtomicBool guard
    - shutdown_driver() safely closes driver
    - GDSFileHandle struct wrapping CUfileHandle with Drop impl
    - GDSFileHandle::open() opens file with O_DIRECT flag
    - GDSFileHandle::raw() returns handle for FFI calls
    - RegisteredBuffer struct wrapping GPU buffer registration
    - RegisteredBuffer::new() registers buffer with cuFileBufRegister
    - RegisteredBuffer implements Drop for automatic deregistration
    - All wrappers return GDSError on failure
    - Unit tests verify handle lifecycle
  estimated_complexity: high
  files_affected:
    - crates/context-graph-cuda/src/gds/driver.rs
    - crates/context-graph-cuda/src/gds/handle.rs
    - crates/context-graph-cuda/src/gds/buffer.rs

- id: TASK-GDS-006
  title: Implement GDSError type hierarchy
  type: implementation
  layer: foundation
  requirement_refs: [REQ-GDS-038, REQ-GDS-039, REQ-GDS-040, REQ-GDS-041, REQ-GDS-042]
  dependencies: [TASK-GDS-003]
  acceptance_criteria:
    - GDSError enum with thiserror derive
    - Variants: InvalidConfig, DriverInit, DriverShutdown, FileOpen, HandleRegister
    - Variants: BufferRegister, ReadFailed, WriteFailed, BatchSubmit
    - Variants: AlignmentError, GpuMemoryAlloc, ChannelClosed, ThreadPanic
    - Variants: EvictionDirty, Timeout, NotSupported, InsufficientMemory
    - From<CUfileStatus> impl for automatic conversion
    - From<std::io::Error> impl for file operations
    - All variants have descriptive #[error()] messages
    - GDSResult<T> type alias defined
    - Error type is Send + Sync
  estimated_complexity: low
  files_affected:
    - crates/context-graph-cuda/src/gds/error.rs

- id: TASK-GDS-007
  title: Implement buffer alignment utilities
  type: implementation
  layer: foundation
  requirement_refs: [REQ-GDS-003]
  dependencies: [TASK-GDS-002]
  acceptance_criteria:
    - align_up(value, alignment) rounds up to alignment boundary
    - align_down(value, alignment) rounds down to alignment boundary
    - is_aligned(value, alignment) checks alignment
    - All functions use bitwise operations for efficiency
    - debug_assert! verifies alignment is power of 2
    - AlignedBuffer struct with ptr, size, alignment, gpu_ptr fields
    - Unit tests for edge cases (zero, max values, various alignments)
  estimated_complexity: low
  files_affected:
    - crates/context-graph-cuda/src/gds/alignment.rs
```

---

## Logic Layer Tasks (Build Second)

These tasks implement double buffering, prefetch scheduling, and core I/O operations.

```yaml
- id: TASK-GDS-008
  title: Implement GDSBuffer and BufferState types
  type: implementation
  layer: logic
  requirement_refs: [REQ-GDS-017, REQ-GDS-019]
  dependencies: [TASK-GDS-005, TASK-GDS-007]
  acceptance_criteria:
    - BufferState enum with Idle, Filling, Ready, Processing, Flushing variants
    - GDSBuffer struct with gpu_ptr, size, state, file_offset, valid_bytes, registration
    - GDSBuffer wraps RegisteredBuffer for lifecycle management
    - State transitions are thread-safe using AtomicU8 or Mutex
    - Unit tests verify state transition correctness
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-cuda/src/gds/buffer_pool.rs

- id: TASK-GDS-009
  title: Implement DoubleBufferManager for ping-pong I/O
  type: implementation
  layer: logic
  requirement_refs: [REQ-GDS-017, REQ-GDS-018, REQ-GDS-020]
  dependencies: [TASK-GDS-008]
  acceptance_criteria:
    - DoubleBufferManager struct with two Arc<Mutex<GDSBuffer>> buffers
    - new() allocates and registers two GPU buffers
    - front_buffer() returns current consumption buffer
    - back_buffer() returns current fill buffer
    - swap() atomically switches active buffer index
    - overlapped_read() implements double-buffered streaming pattern
    - Fill and process operations overlap without stalls
    - Unit tests verify buffer swap correctness
    - Integration test measures throughput meets 4 GB/s target
  estimated_complexity: high
  files_affected:
    - crates/context-graph-cuda/src/gds/double_buffer.rs

- id: TASK-GDS-010
  title: Implement AsyncIOPipeline for concurrent I/O
  type: implementation
  layer: logic
  requirement_refs: [REQ-GDS-004, REQ-GDS-014]
  dependencies: [TASK-GDS-005, TASK-GDS-008]
  acceptance_criteria:
    - IORequest struct with operation, buffer_index, file_offset, size, callback
    - IOOperation enum with Read and Write variants
    - AsyncIOPipeline struct with request/completion channels
    - new() spawns I/O worker thread
    - submit() queues request, blocks if at max_outstanding
    - wait_one() waits for single completion
    - drain() waits for all pending operations
    - Worker thread executes cuFileRead/cuFileWrite calls
    - Unit tests verify concurrent operation handling
  estimated_complexity: high
  files_affected:
    - crates/context-graph-cuda/src/gds/async_io/pipeline.rs
    - crates/context-graph-cuda/src/gds/async_io/mod.rs

- id: TASK-GDS-011
  title: Implement PrefetchScheduler with pattern detection
  type: implementation
  layer: logic
  requirement_refs: [REQ-GDS-021, REQ-GDS-022, REQ-GDS-023, REQ-GDS-024, REQ-GDS-025]
  dependencies: [TASK-GDS-010]
  acceptance_criteria:
    - PrefetchRequest struct with file_offset, size, priority, deadline
    - PrefetchRequest implements Ord for priority queue ordering
    - AccessPattern enum with Sequential, Strided, Random, Unknown variants
    - PrefetchScheduler struct with pending queue, access history, pattern detection
    - record_access() updates access history and detects pattern
    - detect_pattern() analyzes history for sequential/strided/random access
    - schedule_prefetch() creates requests based on detected pattern
    - execute_prefetches() submits requests to AsyncIOPipeline
    - hit_rate() returns prefetch hit rate metric
    - Unit tests verify pattern detection accuracy
    - Integration test verifies >75% prefetch hit rate for sequential access
  estimated_complexity: high
  files_affected:
    - crates/context-graph-cuda/src/gds/prefetch.rs

- id: TASK-GDS-012
  title: Implement ErrorRecovery with retry and fallback logic
  type: implementation
  layer: logic
  requirement_refs: [REQ-GDS-031, REQ-GDS-032, REQ-GDS-036, REQ-GDS-037, REQ-GDS-038]
  dependencies: [TASK-GDS-006]
  acceptance_criteria:
    - ErrorRecovery struct with max_retries, backoff_base_ms, fallback_enabled
    - with_retry() executes operation with exponential backoff
    - is_retryable() determines if error should be retried
    - with_fallback() tries GDS then falls back to standard I/O
    - should_fallback() determines if fallback is appropriate
    - Exponential backoff: delay = base_ms * 2^attempt
    - Unit tests verify retry behavior
    - Unit tests verify fallback triggers correctly
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-cuda/src/gds/fallback/recovery.rs

- id: TASK-GDS-013
  title: Implement GDSDetector for hardware capability detection
  type: implementation
  layer: logic
  requirement_refs: [REQ-GDS-031, REQ-GDS-048]
  dependencies: [TASK-GDS-005]
  acceptance_criteria:
    - GDSDetector struct with cached detection results
    - DetectionResult struct with available, reason, capabilities, devices, timestamp
    - AvailabilityReason enum with all failure cases
    - NVMeDevice struct with path, model, gds_compatible, max_bandwidth
    - detect() checks driver, GPU, filesystem, permissions
    - is_available() returns cached availability
    - refresh() forces re-detection
    - diagnostics() returns detailed GDSDiagnostics
    - Detection completes in < 100ms (REQ-GDS-048)
    - Unit tests verify detection logic
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-cuda/src/gds/fallback/detector.rs
    - crates/context-graph-cuda/src/gds/fallback/mod.rs

- id: TASK-GDS-014
  title: Implement StandardIOFallback for non-GDS systems
  type: implementation
  layer: logic
  requirement_refs: [REQ-GDS-032, REQ-GDS-033, REQ-GDS-034, REQ-GDS-035]
  dependencies: [TASK-GDS-006]
  acceptance_criteria:
    - FallbackConfig struct with enabled, buffer_size, thread_count, log_events, log_level
    - StandardIOFallback struct with thread pool and metrics
    - load_model_weights() reads file via standard I/O, copies to GPU
    - load_faiss_index() loads index via standard I/O
    - stream_shards() streams shards via standard I/O
    - metrics() returns FallbackMetrics with invocation_count, bytes_transferred, avg_bandwidth
    - All methods produce identical results to GDS (REQ-GDS-035)
    - Integration tests verify fallback correctness
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-cuda/src/gds/fallback/standard_io.rs

- id: TASK-GDS-015
  title: Implement GracefulDegradation handler
  type: implementation
  layer: logic
  requirement_refs: [REQ-GDS-036, REQ-GDS-037]
  dependencies: [TASK-GDS-012, TASK-GDS-013, TASK-GDS-014]
  acceptance_criteria:
    - DegradationMode enum with FullGDS, PartialGDS, FullFallback
    - DegradationPolicy struct with failure_threshold, failure_window, recovery_interval, auto_recovery
    - DegradationEvent struct for event logging
    - GracefulDegradation struct combining GDS manager and fallback
    - load() automatically selects best available method
    - current_mode() returns current degradation mode
    - set_mode() forces mode change
    - try_recover() attempts to upgrade to higher mode
    - event_history() returns degradation events
    - Unit tests verify mode transitions
    - Integration test verifies seamless failover
  estimated_complexity: high
  files_affected:
    - crates/context-graph-cuda/src/gds/fallback/degradation.rs
```

---

## Surface Layer Tasks (Build Last)

These tasks implement high-level loaders, metrics, and the integrated GDS context.

```yaml
- id: TASK-GDS-016
  title: Implement ModelLoader for direct model weight loading
  type: implementation
  layer: surface
  requirement_refs: [REQ-GDS-006, REQ-GDS-009, REQ-GDS-026]
  dependencies: [TASK-GDS-009, TASK-GDS-011]
  acceptance_criteria:
    - ModelLoaderConfig struct with chunk_size, verify_checksums, alignment, dtype_conversion
    - DTypeConversion enum with F32ToF16, F32ToBF16, F32ToFP8, None
    - ModelLoader struct with GDS manager and double buffer
    - load() loads weights to newly allocated GPU buffer
    - load_into() loads weights to pre-allocated buffer
    - load_sharded() loads multiple weight shards
    - progress() returns LoadProgress with bytes_loaded, total_bytes, bandwidth, eta
    - LoadStats struct with bytes_loaded, duration, bandwidth, speedup
    - ModelWeights struct with buffer, tensors, dtype, size, checksum
    - TensorMetadata struct with name, offset, shape, dtype
    - Benchmark demonstrates 4x speedup over standard I/O (REQ-GDS-026)
  estimated_complexity: high
  files_affected:
    - crates/context-graph-cuda/src/gds/loader/model_loader.rs
    - crates/context-graph-cuda/src/gds/loader/mod.rs

- id: TASK-GDS-017
  title: Implement FAISSLoader for direct index loading
  type: implementation
  layer: surface
  requirement_refs: [REQ-GDS-007, REQ-GDS-010, REQ-GDS-027]
  dependencies: [TASK-GDS-009]
  acceptance_criteria:
    - FAISSLoaderConfig struct with gpu_direct, gpu_device, precompute_search
    - FAISSLoader struct with GDS manager
    - load() loads FAISS index directly to GPU memory
    - load_ivf_pq() loads IVF-PQ structured index with centroids and codebooks
    - warmup() precomputes search structures
    - FAISSGPUIndex struct with index_type, buffers, metadata, search_config
    - FAISSIndexType enum with Flat, IVF, IVFPQ, IVFSQ, HNSW
    - FAISSGPUBuffers struct with vectors, centroids, codebooks, invlists
    - FAISSMetadata struct with num_vectors, dimension, nlist, pq_params
    - Integration test verifies 70% load time reduction (REQ-GDS-027)
  estimated_complexity: high
  files_affected:
    - crates/context-graph-cuda/src/gds/loader/faiss_loader.rs

- id: TASK-GDS-018
  title: Implement ShardStreamer for memory shard streaming
  type: implementation
  layer: surface
  requirement_refs: [REQ-GDS-008, REQ-GDS-028, REQ-GDS-029]
  dependencies: [TASK-GDS-009, TASK-GDS-011]
  acceptance_criteria:
    - ShardConfig struct with shard_size, prefetch_count, hot_threshold, compression
    - ShardId newtype with u64 inner value
    - MemoryShard struct with id, buffer, metadata, stats
    - ShardMetadata struct with node_count, time_range, salience_range, compression_ratio
    - ShardStats struct with access_count, last_access, is_hot
    - ShardStreamer struct with GDS manager, readers, prefetch scheduler
    - stream() returns ShardStream iterator over shards
    - get_shard() loads specific shard by ID
    - prefetch_shards() schedules prefetch for upcoming shards
    - evict_cold_shards() removes infrequently accessed shards
    - stats() returns ShardStreamStats with hit rate and timing
    - Integration test verifies continuous streaming without CPU copies
  estimated_complexity: high
  files_affected:
    - crates/context-graph-cuda/src/gds/loader/shard_streamer.rs

- id: TASK-GDS-019
  title: Implement GDSMetrics and performance monitoring
  type: implementation
  layer: surface
  requirement_refs: [REQ-GDS-026, REQ-GDS-027, REQ-GDS-028, REQ-GDS-029, REQ-GDS-030]
  dependencies: [TASK-GDS-009, TASK-GDS-011]
  acceptance_criteria:
    - GDSMetrics struct with atomic counters for bytes, ops, latency, buffer stats
    - record_read() updates read counters and latency
    - record_write() updates write counters and latency
    - read_throughput() calculates bytes/second
    - write_throughput() calculates bytes/second
    - avg_read_latency() calculates average latency in microseconds
    - prefetch_hit_rate() returns hit rate
    - check_targets() returns PerformanceStatus comparing to targets
    - PerformanceStatus struct with read_target_met, write_target_met, latency_target_met
    - All operations use Ordering::Relaxed for minimal overhead
    - Unit tests verify metric calculations
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-cuda/src/gds/metrics/mod.rs
    - crates/context-graph-cuda/src/gds/metrics/bandwidth.rs

- id: TASK-GDS-020
  title: Implement GDSContext high-level API
  type: implementation
  layer: surface
  requirement_refs: [REQ-GDS-006, REQ-GDS-007, REQ-GDS-008]
  dependencies: [TASK-GDS-015, TASK-GDS-016, TASK-GDS-017, TASK-GDS-018, TASK-GDS-019]
  acceptance_criteria:
    - GDSContext struct combining all GDS components
    - new() initializes driver, allocates buffers, creates components
    - read_to_gpu() reads file directly to GPU memory with retry
    - write_from_gpu() writes GPU memory to file with retry
    - stream_read() performs double-buffered streaming with callback
    - metrics() returns Arc<GDSMetrics>
    - performance_status() returns current PerformanceStatus
    - Drop impl shuts down driver and frees resources
    - Integration test verifies end-to-end operation
    - Benchmark validates 4 GB/s read, 2 GB/s write targets
  estimated_complexity: high
  files_affected:
    - crates/context-graph-cuda/src/gds/context.rs
```

---

## Test Tasks

```yaml
- id: TASK-GDS-TEST-001
  title: Unit tests for GDS configuration and types
  type: test
  layer: foundation
  requirement_refs: [REQ-GDS-049]
  dependencies: [TASK-GDS-002, TASK-GDS-003, TASK-GDS-006, TASK-GDS-007]
  acceptance_criteria:
    - Tests for GDSConfig validation (buffer size, alignment)
    - Tests for GDSConfig default values
    - Tests for CUfileStatus enum conversion
    - Tests for alignment utilities (edge cases)
    - Tests for GDSError display formatting
    - All tests pass with cargo test -p context-graph-cuda --features gds
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-cuda/src/gds/config.rs (test module)
    - crates/context-graph-cuda/src/gds/error.rs (test module)
    - crates/context-graph-cuda/src/gds/alignment.rs (test module)

- id: TASK-GDS-TEST-002
  title: Unit tests for double buffering and prefetch
  type: test
  layer: logic
  requirement_refs: [REQ-GDS-049]
  dependencies: [TASK-GDS-009, TASK-GDS-011]
  acceptance_criteria:
    - TC-GDS-006: Double buffer continuous streaming test
    - TC-GDS-007: Buffer state transitions test
    - TC-GDS-008: Prefetch scheduling test
    - TC-GDS-009: Prefetch cancellation test
    - Tests verify no stalls during buffer swaps
    - Tests verify pattern detection accuracy
    - All tests pass with cargo test
  estimated_complexity: high
  files_affected:
    - crates/context-graph-cuda/src/gds/double_buffer.rs (test module)
    - crates/context-graph-cuda/src/gds/prefetch.rs (test module)

- id: TASK-GDS-TEST-003
  title: Integration tests for GDS performance benchmarks
  type: test
  layer: surface
  requirement_refs: [REQ-GDS-050]
  dependencies: [TASK-GDS-020]
  acceptance_criteria:
    - TC-GDS-001: Model load speedup test (4x target)
    - TC-GDS-002: FAISS index load time test (70% reduction)
    - TC-GDS-003: Zero CPU copies during streaming
    - TC-GDS-004: I/O bandwidth measurement (10 GB/s)
    - TC-GDS-005: Concurrent operations stress test (64 ops)
    - TC-GDS-015: End-to-end model loading test
    - TC-GDS-016: FAISS index search after GDS load
    - TC-GDS-017: Memory shard round-trip test
    - Tests require GDS-capable hardware or skip gracefully
    - All tests in tests/integration/gds_performance_tests.rs pass
  estimated_complexity: high
  files_affected:
    - tests/integration/gds_performance_tests.rs

- id: TASK-GDS-TEST-004
  title: Integration tests for fallback mechanisms
  type: test
  layer: logic
  requirement_refs: [REQ-GDS-050, REQ-GDS-053]
  dependencies: [TASK-GDS-015]
  acceptance_criteria:
    - TC-GDS-010: GDS detection test
    - TC-GDS-011: Automatic fallback test
    - TC-GDS-012: Fallback correctness test (identical results)
    - TC-GDS-013: Fallback logging test
    - TC-GDS-014: Graceful degradation recovery test
    - Tests verify 100% functionality without GDS hardware
    - All tests in tests/integration/fallback_tests.rs pass
  estimated_complexity: medium
  files_affected:
    - tests/integration/fallback_tests.rs
```

---

## Dependency Graph

```
TASK-GDS-001 (module structure)
    |
    +-- TASK-GDS-002 (config) --+
    |                            |
    +-- TASK-GDS-003 (FFI types)-+-- TASK-GDS-006 (errors)
    |       |                    |
    |       +-- TASK-GDS-004 (FFI functions)
    |               |
    |               +-- TASK-GDS-005 (safe wrappers)
    |                       |
    +-- TASK-GDS-007 (alignment)
            |
            +-- TASK-GDS-008 (buffer/state types)
                    |
    +---------------+---------------+
    |               |               |
TASK-GDS-009    TASK-GDS-010    TASK-GDS-012 (recovery)
(double buffer) (async pipeline)     |
    |               |               |
    +-------+-------+           TASK-GDS-013 (detector)
            |                       |
    TASK-GDS-011 (prefetch)    TASK-GDS-014 (std I/O fallback)
            |                       |
            +-----------------------+
                    |
            TASK-GDS-015 (graceful degradation)
                    |
    +---------------+---------------+---------------+
    |               |               |               |
TASK-GDS-016    TASK-GDS-017    TASK-GDS-018    TASK-GDS-019
(model loader)  (FAISS loader)  (shard streamer) (metrics)
    |               |               |               |
    +---------------+---------------+---------------+
                    |
            TASK-GDS-020 (GDSContext)
```

---

## Traceability Matrix

| Task ID | Requirements Covered |
|---------|---------------------|
| TASK-GDS-001 | REQ-GDS-001, REQ-GDS-002 |
| TASK-GDS-002 | REQ-GDS-001, REQ-GDS-003, REQ-GDS-004 |
| TASK-GDS-003 | REQ-GDS-012, REQ-GDS-013 |
| TASK-GDS-004 | REQ-GDS-012, REQ-GDS-013, REQ-GDS-014 |
| TASK-GDS-005 | REQ-GDS-012, REQ-GDS-015, REQ-GDS-016 |
| TASK-GDS-006 | REQ-GDS-038, REQ-GDS-039, REQ-GDS-040, REQ-GDS-041, REQ-GDS-042 |
| TASK-GDS-007 | REQ-GDS-003 |
| TASK-GDS-008 | REQ-GDS-017, REQ-GDS-019 |
| TASK-GDS-009 | REQ-GDS-017, REQ-GDS-018, REQ-GDS-020 |
| TASK-GDS-010 | REQ-GDS-004, REQ-GDS-014 |
| TASK-GDS-011 | REQ-GDS-021, REQ-GDS-022, REQ-GDS-023, REQ-GDS-024, REQ-GDS-025 |
| TASK-GDS-012 | REQ-GDS-031, REQ-GDS-032, REQ-GDS-036, REQ-GDS-037, REQ-GDS-038 |
| TASK-GDS-013 | REQ-GDS-031, REQ-GDS-048 |
| TASK-GDS-014 | REQ-GDS-032, REQ-GDS-033, REQ-GDS-034, REQ-GDS-035 |
| TASK-GDS-015 | REQ-GDS-036, REQ-GDS-037 |
| TASK-GDS-016 | REQ-GDS-006, REQ-GDS-009, REQ-GDS-026 |
| TASK-GDS-017 | REQ-GDS-007, REQ-GDS-010, REQ-GDS-027 |
| TASK-GDS-018 | REQ-GDS-008, REQ-GDS-028, REQ-GDS-029 |
| TASK-GDS-019 | REQ-GDS-026, REQ-GDS-027, REQ-GDS-028, REQ-GDS-029, REQ-GDS-030 |
| TASK-GDS-020 | REQ-GDS-006, REQ-GDS-007, REQ-GDS-008 |
| TASK-GDS-TEST-001 | REQ-GDS-049 |
| TASK-GDS-TEST-002 | REQ-GDS-049 |
| TASK-GDS-TEST-003 | REQ-GDS-050 |
| TASK-GDS-TEST-004 | REQ-GDS-050, REQ-GDS-053 |

---

## Performance Verification Criteria

| Metric | Target | Verification Task |
|--------|--------|------------------|
| Read throughput | 4 GB/s | TASK-GDS-TEST-003 (TC-GDS-001, TC-GDS-004) |
| Write throughput | 2 GB/s | TASK-GDS-TEST-003 (TC-GDS-004) |
| Latency (4KB aligned) | < 100 us | TASK-GDS-019, TASK-GDS-TEST-003 |
| Model load speedup | 4x | TASK-GDS-TEST-003 (TC-GDS-001) |
| FAISS load reduction | 70% | TASK-GDS-TEST-003 (TC-GDS-002) |
| CPU utilization during I/O | < 10% | TASK-GDS-TEST-003 (TC-GDS-003) |
| Prefetch hit rate | > 75% | TASK-GDS-011, TASK-GDS-TEST-002 |
| GDS detection time | < 100 ms | TASK-GDS-013 |
| Fallback success rate | 100% | TASK-GDS-TEST-004 |
| Unit test coverage | > 90% | All TASK-GDS-TEST-* |
| Integration test coverage | > 80% | TASK-GDS-TEST-003, TASK-GDS-TEST-004 |

---

*Document generated: 2025-12-31*
*Task Specification Version: 1.0*
*Module: GPU Direct Storage (Phase 7)*
*Total Tasks: 24 (20 implementation + 4 test)*
