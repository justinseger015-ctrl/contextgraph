# Module 7: CUDA Optimization - Technical Specification

```yaml
metadata:
  id: TECH-CUDA-007
  version: 1.0.0
  module: CUDA Optimization
  phase: 2
  status: draft
  created: 2025-12-31
  target_hardware: NVIDIA RTX 5090
  cuda_version: 13.1
  dependencies:
    - TECH-EMBED-003 (Module 3: Embedding Pipeline)
```

---

## 1. RTX 5090 Hardware Specifications

```yaml
nvidia_rtx_5090:
  architecture: Blackwell
  cuda_cores: 21760
  tensor_cores: 680  # 5th generation
  memory:
    type: GDDR7
    capacity_gb: 32
    bandwidth_gbps: 1792
    bus_width_bits: 512
  compute:
    fp32_tflops: 109.6
    fp16_tflops: 219.2
    tensor_fp16_tflops: 875.2
  power:
    tdp_watts: 575
    green_context_efficiency: 0.85
```

### 1.1 CUDA 13.1 Features

```yaml
cuda_features:
  green_contexts: true          # Power efficiency
  cuMemAllocAsync: true         # Memory pools
  cudaGraph: true               # Graph execution
  wmma_api: true                # Tensor operations
  fp8_support: true             # FP8 for inference
```

---

## 2. Module Structure

```
context-graph-cuda/
├── src/
│   ├── lib.rs                    # Public API
│   ├── device.rs                 # Device management
│   ├── context.rs                # Green Contexts
│   ├── stream.rs                 # 8-stream management
│   ├── memory.rs                 # Memory pools
│   ├── error.rs                  # Error handling
│   ├── ffi/
│   │   ├── cuda_runtime.rs       # Runtime API bindings
│   │   ├── cudnn.rs              # cuDNN bindings
│   │   └── cublas.rs             # cuBLAS bindings
│   └── kernels/
│       ├── embedding_fuse.rs     # Embedding fusion wrapper
│       ├── hopfield_update.rs    # Hopfield network wrapper
│       └── similarity_batch.rs   # Batch similarity wrapper
├── kernels/
│   ├── embedding_fuse.cu         # CUDA kernel
│   ├── hopfield_update.cu        # CUDA kernel
│   ├── similarity_batch.cu       # CUDA kernel
│   └── common.cuh                # Shared utilities
└── Cargo.toml
```

---

## 3. FFI Bindings

### 3.1 Core CUDA Runtime Bindings

```rust
// src/ffi/cuda_runtime.rs

pub type cudaError_t = i32;
pub type cudaStream_t = *mut std::ffi::c_void;
pub type cudaEvent_t = *mut std::ffi::c_void;
pub type cudaMemPool_t = *mut std::ffi::c_void;

pub const CUDA_SUCCESS: cudaError_t = 0;

extern "C" {
    // Memory Pool (cuMemAllocAsync)
    pub fn cudaMemPoolCreate(pool: *mut cudaMemPool_t, props: *const cudaMemPoolProps) -> cudaError_t;
    pub fn cudaMemPoolDestroy(pool: cudaMemPool_t) -> cudaError_t;
    pub fn cudaMallocAsync(ptr: *mut *mut c_void, size: usize, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaFreeAsync(ptr: *mut c_void, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaMallocFromPoolAsync(ptr: *mut *mut c_void, size: usize, pool: cudaMemPool_t, stream: cudaStream_t) -> cudaError_t;

    // Stream Management
    pub fn cudaStreamCreate(stream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamCreateWithPriority(stream: *mut cudaStream_t, flags: u32, priority: i32) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: u32) -> cudaError_t;

    // Events
    pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;

    // Error Handling
    pub fn cudaGetLastError() -> cudaError_t;
    pub fn cudaPeekAtLastError() -> cudaError_t;
    pub fn cudaGetErrorString(error: cudaError_t) -> *const c_char;
}

#[repr(C)]
pub struct cudaMemPoolProps {
    pub allocType: i32,
    pub handleTypes: i32,
    pub location: cudaMemLocation,
    pub maxSize: usize,
    pub reserved: [i8; 56],
}
```

### 3.2 cuDNN Bindings

```rust
// src/ffi/cudnn.rs

pub type cudnnHandle_t = *mut c_void;
pub type cudnnTensorDescriptor_t = *mut c_void;
pub type cudnnStatus_t = i32;
pub const CUDNN_STATUS_SUCCESS: cudnnStatus_t = 0;

#[repr(C)]
pub enum cudnnDataType_t {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_HALF = 2,
    CUDNN_DATA_BFLOAT16 = 9,
    CUDNN_DATA_FP8_E4M3 = 12,
}

extern "C" {
    pub fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
    pub fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;
    pub fn cudnnSetStream(handle: cudnnHandle_t, stream: cudaStream_t) -> cudnnStatus_t;
    pub fn cudnnSoftmaxForward(
        handle: cudnnHandle_t, algo: i32, mode: i32,
        alpha: *const c_void, xDesc: cudnnTensorDescriptor_t, x: *const c_void,
        beta: *const c_void, yDesc: cudnnTensorDescriptor_t, y: *mut c_void,
    ) -> cudnnStatus_t;
}
```

---

## 4. Memory Pool Management

```rust
// src/memory.rs

pub struct MemoryPool {
    handle: cudaMemPool_t,
    config: MemoryPoolConfig,
    allocations: Mutex<HashMap<*mut c_void, usize>>,
}

#[derive(Clone)]
pub struct MemoryPoolConfig {
    pub initial_size: usize,   // 1GB default
    pub max_size: usize,       // 24GB (leaving 8GB for system on 32GB card)
    pub release_threshold: f32, // 0.5 default
}

impl MemoryPool {
    pub fn new(config: MemoryPoolConfig, device_id: i32) -> CudaResult<Self> {
        let props = cudaMemPoolProps {
            allocType: 1, // cudaMemAllocationTypePinned
            location: cudaMemLocation { location_type: 0, id: device_id },
            maxSize: config.max_size,
            ..Default::default()
        };
        let mut handle = std::ptr::null_mut();
        unsafe { cuda_check!(cudaMemPoolCreate(&mut handle, &props)); }
        Ok(Self { handle, config, allocations: Mutex::new(HashMap::new()) })
    }

    /// Async allocation from pool
    pub fn alloc_async(&self, size: usize, stream: cudaStream_t) -> CudaResult<*mut c_void> {
        let mut ptr = std::ptr::null_mut();
        unsafe { cuda_check!(cudaMallocFromPoolAsync(&mut ptr, size, self.handle, stream)); }
        self.allocations.lock().unwrap().insert(ptr, size);
        Ok(ptr)
    }

    /// Async free back to pool
    pub fn free_async(&self, ptr: *mut c_void, stream: cudaStream_t) -> CudaResult<()> {
        self.allocations.lock().unwrap().remove(&ptr);
        unsafe { cuda_check!(cudaFreeAsync(ptr, stream)); }
        Ok(())
    }
}
```

---

## 5. 8-Stream Parallelism

```rust
// src/stream.rs

pub const NUM_STREAMS: usize = 8;

#[derive(Clone, Copy, PartialEq)]
pub enum StreamRole {
    Embedding,    // Stream 0: High priority
    Hopfield,     // Stream 1: High priority
    Similarity,   // Stream 2: High priority
    TransferH2D,  // Stream 3: Normal - Host to Device
    TransferD2H,  // Stream 4: Normal - Device to Host
    Graph,        // Stream 5: Normal - Graph execution
    CuDnn,        // Stream 6: Normal - cuDNN ops
    General,      // Stream 7: Low priority
}

pub struct StreamManager {
    streams: [cudaStream_t; NUM_STREAMS],
    events: [cudaEvent_t; NUM_STREAMS],
}

impl StreamManager {
    pub fn new() -> CudaResult<Self> {
        let priorities = [-1, -1, -1, 0, 0, 0, 0, 1]; // High, Normal, Low
        let mut streams = [std::ptr::null_mut(); NUM_STREAMS];
        let mut events = [std::ptr::null_mut(); NUM_STREAMS];

        for i in 0..NUM_STREAMS {
            unsafe {
                cuda_check!(cudaStreamCreateWithPriority(&mut streams[i], 1, priorities[i]));
                cuda_check!(cudaEventCreate(&mut events[i]));
            }
        }
        Ok(Self { streams, events })
    }

    pub fn get_stream(&self, role: StreamRole) -> cudaStream_t {
        self.streams[role as usize]
    }

    /// Create dependency: target waits for source
    pub fn create_dependency(&self, source: StreamRole, target: StreamRole) -> CudaResult<()> {
        unsafe {
            cuda_check!(cudaEventRecord(self.events[source as usize], self.streams[source as usize]));
            cuda_check!(cudaStreamWaitEvent(self.streams[target as usize], self.events[source as usize], 0));
        }
        Ok(())
    }

    pub fn sync_stream(&self, role: StreamRole) -> CudaResult<()> {
        unsafe { cuda_check!(cudaStreamSynchronize(self.streams[role as usize])); }
        Ok(())
    }
}
```

---

## 6. Custom CUDA Kernels

### 6.1 Embedding Fusion Kernel

```cuda
// kernels/embedding_fuse.cu
#define BLOCK_SIZE 256
#define FUSED_OUTPUT_DIM 1536

__global__ void embedding_fuse_kernel(
    const float* __restrict__ embeddings,    // [batch, 12, max_dim]
    const int* __restrict__ dimensions,      // [12] - actual dims per model
    const float* __restrict__ gating_weights,// [12, 1536]
    const float* __restrict__ expert_projs,  // [12, model_dim, 1536]
    float* __restrict__ output,              // [batch, 1536]
    int batch_size
) {
    __shared__ float s_gates[12];
    const int batch_idx = blockIdx.x;
    const int out_dim = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    if (batch_idx >= batch_size || out_dim >= FUSED_OUTPUT_DIM) return;

    // Softmax gating
    if (threadIdx.x < 12) {
        float sum = 0.0f;
        for (int i = 0; i < 12; i++) sum += expf(gating_weights[i * FUSED_OUTPUT_DIM + out_dim]);
        s_gates[threadIdx.x] = expf(gating_weights[threadIdx.x * FUSED_OUTPUT_DIM + out_dim]) / sum;
    }
    __syncthreads();

    // Fuse with gated mixture
    float result = 0.0f;
    for (int model = 0; model < 12; model++) {
        float contrib = 0.0f;
        for (int d = 0; d < dimensions[model]; d++) {
            contrib += embeddings[batch_idx * 12 * 4096 + model * 4096 + d] *
                       expert_projs[/* offset */];
        }
        result += s_gates[model] * contrib;
    }
    output[batch_idx * FUSED_OUTPUT_DIM + out_dim] = result;
}

extern "C" void launch_embedding_fuse(
    const float* embeddings, const int* dimensions, const float* gating_weights,
    const float* expert_projs, float* output, int batch_size, cudaStream_t stream
) {
    dim3 grid(batch_size, (FUSED_OUTPUT_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE);
    embedding_fuse_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        embeddings, dimensions, gating_weights, expert_projs, output, batch_size);
}
```

### 6.2 Hopfield Update Kernel

```cuda
// kernels/hopfield_update.cu
#define HOPFIELD_DIM 1536
#define BLOCK_SIZE 256

__global__ void hopfield_update_kernel(
    const float* __restrict__ patterns,  // [num_patterns, dim]
    const float* __restrict__ query,     // [batch, dim]
    float* __restrict__ output,          // [batch, dim]
    float* __restrict__ attention,       // [batch, num_patterns]
    int num_patterns, int dim, int batch_size, float beta
) {
    __shared__ float s_query[HOPFIELD_DIM];
    __shared__ float s_attn[1024];
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // Load query to shared memory
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE)
        s_query[i] = query[batch_idx * dim + i];
    __syncthreads();

    // Compute attention scores with softmax
    float max_score = -INFINITY;
    for (int p = threadIdx.x; p < num_patterns; p += BLOCK_SIZE) {
        float score = 0.0f;
        for (int d = 0; d < dim; d++) score += s_query[d] * patterns[p * dim + d];
        s_attn[p] = score * beta;
        max_score = fmaxf(max_score, s_attn[p]);
    }
    // ... softmax normalization and weighted sum
}

extern "C" void launch_hopfield_update(
    const float* patterns, const float* query, float* output, float* attention,
    int num_patterns, int dim, int batch_size, float beta, cudaStream_t stream
) {
    hopfield_update_kernel<<<batch_size, BLOCK_SIZE, 0, stream>>>(
        patterns, query, output, attention, num_patterns, dim, batch_size, beta);
}
```

### 6.3 Batch Similarity Kernel

```cuda
// kernels/similarity_batch.cu
#define TILE_SIZE 32

__global__ void similarity_batch_kernel(
    const float* __restrict__ queries,      // [num_queries, dim]
    const float* __restrict__ corpus,       // [corpus_size, dim]
    float* __restrict__ similarities,       // [num_queries, corpus_size]
    const float* __restrict__ query_norms,  // [num_queries]
    const float* __restrict__ corpus_norms, // [corpus_size]
    int num_queries, int corpus_size, int dim
) {
    __shared__ float s_queries[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_corpus[TILE_SIZE][TILE_SIZE + 1];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float dot = 0.0f;

    // Tiled matrix multiplication
    for (int t = 0; t < (dim + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int q_col = t * TILE_SIZE + threadIdx.x;
        int c_row = t * TILE_SIZE + threadIdx.y;
        s_queries[threadIdx.y][threadIdx.x] = (row < num_queries && q_col < dim) ?
            queries[row * dim + q_col] : 0.0f;
        s_corpus[threadIdx.y][threadIdx.x] = (col < corpus_size && c_row < dim) ?
            corpus[col * dim + c_row] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) dot += s_queries[threadIdx.y][k] * s_corpus[k][threadIdx.x];
        __syncthreads();
    }

    if (row < num_queries && col < corpus_size)
        similarities[row * corpus_size + col] = dot / (query_norms[row] * corpus_norms[col] + 1e-8f);
}

extern "C" void launch_similarity_batch(
    const float* queries, const float* corpus, float* similarities,
    const float* query_norms, const float* corpus_norms,
    int num_queries, int corpus_size, int dim, cudaStream_t stream
) {
    dim3 grid((corpus_size + TILE_SIZE - 1) / TILE_SIZE, (num_queries + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);
    similarity_batch_kernel<<<grid, block, 0, stream>>>(
        queries, corpus, similarities, query_norms, corpus_norms, num_queries, corpus_size, dim);
}
```

---

## 7. Green Contexts for Power Efficiency

```rust
// src/context.rs

#[derive(Clone, Copy, PartialEq)]
pub enum PowerState { Active, Throttled, Idle, Sleep }

pub struct GreenContext {
    config: GreenContextConfig,
    power_state: AtomicU64,
    last_activity: AtomicU64,
    active_kernels: AtomicU64,
}

#[derive(Clone)]
pub struct GreenContextConfig {
    pub idle_threshold_ms: u64,    // 50ms default
    pub efficiency_target: f32,    // 0.85 target
    pub min_active_ms: u64,        // 10ms minimum
}

impl GreenContext {
    pub fn kernel_start(&self) {
        self.active_kernels.fetch_add(1, Ordering::SeqCst);
        self.update_activity();
        if self.power_state.load(Ordering::SeqCst) != PowerState::Active as u64 {
            self.transition_to(PowerState::Active);
        }
    }

    pub fn kernel_end(&self) {
        let prev = self.active_kernels.fetch_sub(1, Ordering::SeqCst);
        self.update_activity();
        if prev == 1 { self.consider_power_transition(); }
    }

    fn consider_power_transition(&self) {
        let idle_us = Self::timestamp_us() - self.last_activity.load(Ordering::SeqCst);
        if idle_us > self.config.idle_threshold_ms * 1000 &&
           self.active_kernels.load(Ordering::SeqCst) == 0 {
            self.transition_to(PowerState::Idle);
        }
    }

    pub fn efficiency_metrics(&self) -> PowerEfficiencyMetrics {
        PowerEfficiencyMetrics {
            current_state: self.power_state(),
            efficiency_ratio: /* calculate */,
            state_transitions: /* count */,
        }
    }
}
```

---

## 8. Error Handling

```rust
// src/error.rs

#[derive(Debug)]
pub struct CudaError {
    pub code: cudaError_t,
    pub category: CudaErrorCategory,
    pub message: String,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum CudaErrorCategory {
    Memory, Launch, Synchronization, Device, InvalidArgument, CuDnn, Unknown,
}

impl CudaError {
    pub fn from_code(code: cudaError_t) -> Self {
        let message = unsafe {
            CStr::from_ptr(cudaGetErrorString(code)).to_string_lossy().into_owned()
        };
        Self { code, category: Self::categorize(code), message, context: None }
    }

    fn categorize(code: cudaError_t) -> CudaErrorCategory {
        match code {
            2 | 35 | 46 => CudaErrorCategory::Memory,
            9 | 98 | 719 => CudaErrorCategory::Launch,
            4 | 5 | 6 => CudaErrorCategory::Synchronization,
            _ => CudaErrorCategory::Unknown,
        }
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(self.category, CudaErrorCategory::Memory | CudaErrorCategory::Synchronization)
    }
}

pub type CudaResult<T> = Result<T, CudaError>;

#[macro_export]
macro_rules! cuda_check {
    ($expr:expr) => {{
        let code = $expr;
        if code != CUDA_SUCCESS {
            return Err(CudaError::from_code(code));
        }
    }};
}

pub fn check_last_error() -> CudaResult<()> {
    unsafe {
        let code = cudaGetLastError();
        if code != CUDA_SUCCESS { return Err(CudaError::from_code(code)); }
    }
    Ok(())
}
```

---

## 9. Kernel Wrapper Interfaces

```rust
// src/kernels/embedding_fuse.rs

extern "C" {
    fn launch_embedding_fuse(
        embeddings: *const f32, dimensions: *const i32, gating_weights: *const f32,
        expert_projs: *const f32, output: *mut f32, batch_size: i32, stream: cudaStream_t);
}

pub struct EmbeddingFuseKernel;

impl EmbeddingFuseKernel {
    pub fn launch(
        embeddings: &[f32], dimensions: &[i32; 12], gating_weights: &[f32],
        expert_projs: &[f32], output: &mut [f32], batch_size: usize, stream: cudaStream_t
    ) -> CudaResult<()> {
        unsafe {
            launch_embedding_fuse(
                embeddings.as_ptr(), dimensions.as_ptr(), gating_weights.as_ptr(),
                expert_projs.as_ptr(), output.as_mut_ptr(), batch_size as i32, stream);
        }
        check_last_error()
    }
}

// Similar wrappers for HopfieldUpdateKernel and SimilarityBatchKernel
```

---

## 10. Performance Targets

```yaml
performance_targets:
  kernel_launch:
    embedding_fuse:    "<10us latency, >100k samples/sec"
    hopfield_update:   "<10us latency, >500k patterns/sec"
    similarity_batch:  "<10us latency, >10M pairs/sec"

  memory_transfer:
    h2d_bandwidth:     ">25 GB/s (PCIe 5.0)"
    d2h_bandwidth:     ">25 GB/s"
    overlap_efficiency: ">95%"
    pool_alloc:        "<5us"

  stream_parallelism:
    concurrent_kernels: 8
    overlap_efficiency: ">90%"

  power_efficiency:
    active_utilization: ">85%"
    idle_reduction:     ">60%"
    state_transition:   "<1ms"
```

---

## 11. Build Configuration

```toml
# Cargo.toml
[package]
name = "context-graph-cuda"
version = "0.1.0"
edition = "2021"

[features]
default = ["cuda-13"]
cuda-13 = []
cudnn = []
cublas = []

[build-dependencies]
cc = { version = "1.0", features = ["parallel"] }
```

```rust
// build.rs
fn main() {
    let cuda_path = std::env::var("CUDA_PATH").unwrap_or("/usr/local/cuda".into());
    cc::Build::new()
        .cuda(true)
        .flag("-gencode").flag("arch=compute_100,code=sm_100") // RTX 5090
        .flag("-O3").flag("--use_fast_math")
        .include(format!("{}/include", cuda_path))
        .file("kernels/embedding_fuse.cu")
        .file("kernels/hopfield_update.cu")
        .file("kernels/similarity_batch.cu")
        .compile("context_graph_kernels");

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
}
```

---

## 12. Summary

| Component | Implementation | Performance Target |
|-----------|---------------|-------------------|
| FFI Bindings | cuda_runtime, cuDNN | Full CUDA 13.1 API |
| Memory Pools | cuMemAllocAsync | <5us allocation |
| Stream Manager | 8 streams, role-based | >90% overlap |
| embedding_fuse | Gated MoE fusion | <10us launch |
| hopfield_update | Continuous Hopfield | <10us launch |
| similarity_batch | Tiled GEMM + TopK | <10us launch |
| Green Contexts | Adaptive power states | >85% efficiency |
| Error Handling | cudaGetLastError | Categorized errors |

The CUDA optimization layer provides high-throughput GPU acceleration for ContextGraph embedding operations while maintaining power efficiency through Green Context management on RTX 5090 hardware.
