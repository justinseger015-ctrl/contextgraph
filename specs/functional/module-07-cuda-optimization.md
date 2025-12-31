# Module 07: CUDA Optimization - Functional Specification

**Version**: 1.0.0
**Status**: Draft
**Author**: Agent #7/28
**Module**: 7 of 14
**Phase**: 6
**Duration**: 3 weeks
**Dependencies**: Module 6 (Bio-Nervous System)
**Last Updated**: 2025-12-31

---

## 1. Executive Summary

The CUDA Optimization module accelerates critical computational paths using GPU computation targeting the RTX 5090 (Blackwell architecture) with CUDA 13.1. This module implements custom CUDA kernels for embedding operations, Modern Hopfield attention, neuromodulation updates, and hyperbolic cone containment checks. Advanced features include Green Contexts for SM partitioning, FP8/FP4 precision for inference, CUDA Tile for memory-efficient attention, and GPU Direct Storage integration.

### 1.1 Key Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| GPU Utilization | >80% | NVIDIA profiler |
| Speedup vs CPU | 5x minimum | Benchmark suite |
| VRAM Usage | <24GB under load | nvidia-smi |
| Multi-Stream Execution | 8 concurrent streams | Stream profiler |
| FP8 Embedding Throughput | >1000/sec | Batch benchmark |
| Kernel Launch Overhead | <10us | CUDA events |

### 1.2 Target Hardware Specifications

| Specification | RTX 5090 (Blackwell) |
|---------------|---------------------|
| Compute Capability | 12.0 |
| VRAM | 32GB GDDR7 |
| Streaming Multiprocessors | 170 SMs |
| CUDA Cores | 21,760 |
| Tensor Cores | 680 (5th gen) |
| Memory Bandwidth | 1.79 TB/s |
| FP8 TFLOPS | 1.79 PFLOPS |
| FP4 TFLOPS | 3.58 PFLOPS |

---

## 2. Architecture Overview

### 2.1 CUDA Module Structure

```
context-graph-cuda/
+-- src/
|   +-- lib.rs                    # Module entry, feature detection
|   +-- config.rs                 # CUDA configuration management
|   +-- device.rs                 # Device initialization, Green Contexts
|   +-- memory/
|   |   +-- mod.rs                # Memory management module
|   |   +-- pool.rs               # Memory pool implementation
|   |   +-- uvm.rs                # Unified Virtual Memory
|   |   +-- pinned.rs             # Pinned memory for transfers
|   +-- kernels/
|   |   +-- mod.rs                # Kernel module exports
|   |   +-- fused_embedding.cu    # Multi-model embedding kernel
|   |   +-- hopfield_attention.cu # Modern Hopfield energy computation
|   |   +-- neuromodulation.cu    # Parallel neuromodulator updates
|   |   +-- cone_containment.cu   # Batch hyperbolic cone checks
|   +-- streams/
|   |   +-- mod.rs                # Stream orchestration
|   |   +-- scheduler.rs          # Kernel dependency scheduling
|   +-- precision/
|   |   +-- mod.rs                # Precision management
|   |   +-- fp8.rs                # FP8 conversion and operations
|   |   +-- fp4.rs                # FP4 conversion and operations
|   +-- tile/
|   |   +-- mod.rs                # CUDA Tile attention
|   |   +-- attention.rs          # Memory-efficient attention
|   +-- gds/
|   |   +-- mod.rs                # GPU Direct Storage integration
|   |   +-- loader.rs             # Model/index loading
|   +-- fallback/
|   |   +-- mod.rs                # CPU fallback implementations
|   +-- profiling/
|       +-- mod.rs                # Performance profiling utilities
+-- kernels/
    +-- fused_embedding.cu        # CUDA kernel source
    +-- hopfield_attention.cu     # CUDA kernel source
    +-- neuromodulation.cu        # CUDA kernel source
    +-- cone_containment.cu       # CUDA kernel source
```

### 2.2 Execution Flow

```
                          +------------------------+
                          |   Host Application     |
                          +------------------------+
                                     |
                                     v
+------------------------------------------------------------------------+
|                        CUDA Configuration Layer                         |
|  - Device detection (compute capability 12.0)                          |
|  - Green Context initialization (4 x 170 SMs partitioning)             |
|  - Memory pool setup                                                   |
|  - Stream allocation (8 streams)                                       |
+------------------------------------------------------------------------+
                                     |
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
+------------------------+  +------------------------+  +------------------------+
|   Stream 0-1           |  |   Stream 2-3           |  |   Stream 4-7           |
|   Embedding Pipeline   |  |   Hopfield Operations  |  |   Neuromod + Utility   |
+------------------------+  +------------------------+  +------------------------+
              |                      |                      |
              v                      v                      v
+------------------------------------------------------------------------+
|                         Memory Pool Layer                              |
|  - Unified Virtual Memory for large graphs (>VRAM)                     |
|  - Pinned memory for CPU-GPU transfers                                 |
|  - Device memory pools to reduce allocation overhead                   |
+------------------------------------------------------------------------+
              |                      |                      |
              v                      v                      v
+------------------------+  +------------------------+  +------------------------+
| fused_embedding_kernel |  | hopfield_attention_    |  | neuromodulation_kernel |
| - FP8/FP4 precision    |  | kernel                 |  | - Parallel updates     |
| - Multi-model fusion   |  | - Energy computation   |  | cone_containment_kernel|
| - Tensor Core ops      |  | - Softmax attention    |  | - Batch hyperbolic     |
+------------------------+  +------------------------+  +------------------------+
                                     |
                                     v
                          +------------------------+
                          |   Results to Host      |
                          |   (Async Copy)         |
                          +------------------------+
```

### 2.3 Green Contexts SM Partitioning

CUDA 13.1 Green Contexts enable SM partitioning for workload isolation:

```
RTX 5090: 170 SMs Total
+------------------------------------------------------------------------+
|  Green Context 0: Embeddings (42 SMs)                                  |
|  - fused_embedding_kernel execution                                     |
|  - FP8 Tensor Core operations                                          |
|  - Dedicated L2 cache partition                                        |
+------------------------------------------------------------------------+
|  Green Context 1: Memory Operations (42 SMs)                           |
|  - hopfield_attention_kernel execution                                 |
|  - FAISS search operations                                             |
|  - Memory-intensive workloads                                          |
+------------------------------------------------------------------------+
|  Green Context 2: Learning (42 SMs)                                    |
|  - neuromodulation_kernel execution                                    |
|  - UTL gradient computations                                           |
|  - Weight updates                                                      |
+------------------------------------------------------------------------+
|  Green Context 3: Utility (44 SMs)                                     |
|  - cone_containment_kernel execution                                   |
|  - Graph traversal operations                                          |
|  - Miscellaneous compute                                               |
+------------------------------------------------------------------------+
```

---

## 3. Configuration

### 3.1 CUDA Configuration Structure

```rust
/// CUDA configuration for RTX 5090 / CUDA 13.1
pub struct CudaConfig {
    /// Device ID for multi-GPU systems
    pub device_id: u32,
    /// Compute capability requirement (12, 0) for RTX 5090
    pub compute_capability: (u32, u32),
    /// Total memory pool size in bytes (default: 20GB)
    pub memory_pool_size: usize,
    /// Number of concurrent CUDA streams (default: 8)
    pub stream_count: u32,
    /// Enable CUDA 13.1 Green Contexts for SM partitioning
    pub green_contexts_enabled: bool,
    /// Green Context partition configuration
    pub green_context_partitions: GreenContextConfig,
    /// Default precision for inference
    pub inference_precision: InferencePrecision,
    /// Enable Unified Virtual Memory for large graphs
    pub uvm_enabled: bool,
    /// UVM migration threshold in bytes
    pub uvm_migration_threshold: usize,
    /// Enable GPU Direct Storage
    pub gds_enabled: bool,
    /// GDS buffer size
    pub gds_buffer_size: usize,
    /// Enable graceful CPU fallback
    pub cpu_fallback_enabled: bool,
}

/// Green Context SM partition configuration
pub struct GreenContextConfig {
    /// SMs allocated for embedding operations
    pub embedding_sms: u32,     // Default: 42
    /// SMs allocated for memory operations
    pub memory_sms: u32,        // Default: 42
    /// SMs allocated for learning operations
    pub learning_sms: u32,      // Default: 42
    /// SMs allocated for utility operations
    pub utility_sms: u32,       // Default: 44
}

/// Inference precision modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferencePrecision {
    /// Full precision (FP32)
    FP32,
    /// Half precision (FP16)
    FP16,
    /// Brain float (BF16)
    BF16,
    /// 8-bit floating point (E4M3 or E5M2)
    FP8,
    /// 4-bit floating point
    FP4,
    /// Mixed precision (FP8 compute, FP16 accumulate)
    Mixed,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            compute_capability: (12, 0),
            memory_pool_size: 20 * 1024 * 1024 * 1024, // 20GB
            stream_count: 8,
            green_contexts_enabled: true,
            green_context_partitions: GreenContextConfig::default(),
            inference_precision: InferencePrecision::FP8,
            uvm_enabled: true,
            uvm_migration_threshold: 1024 * 1024 * 1024, // 1GB
            gds_enabled: true,
            gds_buffer_size: 1024 * 1024 * 1024, // 1GB
            cpu_fallback_enabled: true,
        }
    }
}

impl Default for GreenContextConfig {
    fn default() -> Self {
        Self {
            embedding_sms: 42,
            memory_sms: 42,
            learning_sms: 42,
            utility_sms: 44, // Total: 170 SMs
        }
    }
}
```

### 3.2 Configuration File

```toml
[cuda]
device_id = 0
compute_capability = [12, 0]
memory_pool_size = 21474836480  # 20GB
stream_count = 8
green_contexts_enabled = true
inference_precision = "fp8"
uvm_enabled = true
uvm_migration_threshold = 1073741824  # 1GB
gds_enabled = true
gds_buffer_size = 1073741824  # 1GB
cpu_fallback_enabled = true

[cuda.green_context_partitions]
embedding_sms = 42
memory_sms = 42
learning_sms = 42
utility_sms = 44

[cuda.memory_pool]
# Sub-pool configurations
embedding_pool_size = 8589934592    # 8GB for embeddings
hopfield_pool_size = 6442450944     # 6GB for Hopfield
scratch_pool_size = 4294967296      # 4GB scratch space
transfer_buffer_size = 1073741824   # 1GB for transfers

[cuda.streams]
# Stream priorities (0 = highest)
embedding_stream_priority = 0
hopfield_stream_priority = 0
neuromod_stream_priority = 1
utility_stream_priority = 2

[cuda.kernels]
# Kernel launch configurations
fused_embedding_block_size = 256
hopfield_attention_block_size = 128
neuromodulation_block_size = 256
cone_containment_block_size = 256
```

---

## 4. Custom CUDA Kernels

### 4.1 fused_embedding_kernel

Multi-model embedding computation in a single GPU pass with FP8/FP4 support.

```rust
/// Fused embedding kernel interface
pub struct FusedEmbeddingKernel {
    /// Kernel module handle
    module: CudaModule,
    /// Kernel function handle
    function: CudaFunction,
    /// Model weights (12 models)
    model_weights: Vec<DeviceBuffer<f16>>,
    /// FP8 quantization scales
    fp8_scales: DeviceBuffer<f32>,
    /// Output dimension
    output_dim: usize,
}

/// Embedding input batch
pub struct EmbeddingBatch {
    /// Tokenized input sequences
    pub tokens: DeviceBuffer<u32>,
    /// Sequence lengths
    pub lengths: DeviceBuffer<u32>,
    /// Batch size
    pub batch_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

/// Embedding output
pub struct EmbeddingOutput {
    /// Fused embedding vectors (batch_size x 1536)
    pub embeddings: DeviceBuffer<f32>,
    /// Per-model embeddings for debugging (optional)
    pub per_model_embeddings: Option<Vec<DeviceBuffer<f32>>>,
    /// FP8 quantized embeddings (for storage)
    pub embeddings_fp8: Option<DeviceBuffer<i8>>,
}

impl FusedEmbeddingKernel {
    /// Execute fused embedding computation
    ///
    /// # Arguments
    /// * `batch` - Input token batch
    /// * `stream` - CUDA stream for execution
    /// * `precision` - Computation precision
    ///
    /// # Returns
    /// * `EmbeddingOutput` with fused 1536D embeddings
    ///
    /// `Constraint: Latency < 10ms for batch_size=1`
    /// `Constraint: Throughput > 1000/sec for batch_size=64`
    pub async fn execute(
        &self,
        batch: &EmbeddingBatch,
        stream: &CudaStream,
        precision: InferencePrecision,
    ) -> Result<EmbeddingOutput, CudaError>;

    /// Warm up kernel with dummy data
    pub async fn warmup(&self, batch_size: usize) -> Result<(), CudaError>;

    /// Get kernel profiling statistics
    pub fn profiling_stats(&self) -> KernelStats;
}
```

**Kernel Implementation (CUDA C++):**

```cuda
// fused_embedding.cu
// Fused multi-model embedding kernel for RTX 5090 / CUDA 13.1

#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <mma.h>

// FP8 E4M3 format for weights
typedef __nv_fp8_e4m3 fp8_e4m3;

// Model configuration
#define NUM_MODELS 12
#define OUTPUT_DIM 1536
#define TILE_SIZE 128
#define WARP_SIZE 32

// FuseMoE configuration
#define TOP_K_EXPERTS 4
#define NUM_EXPERTS 8

/// Fused embedding kernel with FP8 computation
/// Uses Tensor Core WMMA operations for maximum throughput
__global__ void fused_embedding_kernel(
    const uint32_t* __restrict__ tokens,
    const uint32_t* __restrict__ lengths,
    const fp8_e4m3* __restrict__ model_weights[NUM_MODELS],
    const float* __restrict__ fp8_scales,
    const float* __restrict__ expert_weights,
    float* __restrict__ output_embeddings,
    const int batch_size,
    const int max_seq_len,
    const int hidden_dim
) {
    // Shared memory for intermediate results
    __shared__ float smem_embeddings[TILE_SIZE][OUTPUT_DIM / NUM_MODELS];
    __shared__ float smem_expert_scores[TOP_K_EXPERTS];

    const int batch_idx = blockIdx.x;
    const int model_idx = blockIdx.y;
    const int thread_idx = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const int seq_len = lengths[batch_idx];
    const float scale = fp8_scales[model_idx];

    // Phase 1: Compute per-model embedding using Tensor Cores
    // Uses FP8 x FP8 -> FP16 accumulate -> FP32 output

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, fp8_e4m3, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, fp8_e4m3, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> acc_frag;

    nvcuda::wmma::fill_fragment(acc_frag, __float2half(0.0f));

    // Process sequence tokens
    for (int t = 0; t < seq_len; t += TILE_SIZE) {
        // Load token embeddings from model weights
        // ... WMMA matrix multiply accumulate ...
    }

    // Store intermediate result
    // ...

    __syncthreads();

    // Phase 2: FuseMoE gating and expert selection
    if (model_idx == 0) {
        // Compute expert scores and select top-k
        // ...
    }

    __syncthreads();

    // Phase 3: CAME-AB cross-attention blending
    // ...

    // Phase 4: Write final fused embedding
    if (model_idx == 0 && thread_idx < OUTPUT_DIM) {
        float fused_value = 0.0f;
        for (int m = 0; m < NUM_MODELS; m++) {
            fused_value += smem_embeddings[0][thread_idx % (OUTPUT_DIM / NUM_MODELS)] * expert_weights[m];
        }
        output_embeddings[batch_idx * OUTPUT_DIM + thread_idx] = fused_value;
    }
}

/// Optimized kernel launch configuration
extern "C" void launch_fused_embedding(
    const uint32_t* tokens,
    const uint32_t* lengths,
    const void** model_weights,
    const float* fp8_scales,
    const float* expert_weights,
    float* output_embeddings,
    int batch_size,
    int max_seq_len,
    int hidden_dim,
    cudaStream_t stream
) {
    dim3 grid(batch_size, NUM_MODELS);
    dim3 block(256);

    fused_embedding_kernel<<<grid, block, 0, stream>>>(
        tokens, lengths,
        (const fp8_e4m3**)model_weights,
        fp8_scales, expert_weights, output_embeddings,
        batch_size, max_seq_len, hidden_dim
    );
}
```

### 4.2 hopfield_attention_kernel

Modern Hopfield Network energy computation and attention mechanism.

```rust
/// Hopfield attention kernel interface
pub struct HopfieldAttentionKernel {
    /// Kernel module handle
    module: CudaModule,
    /// Kernel function handle
    function: CudaFunction,
    /// Default beta parameter
    default_beta: f32,
}

/// Hopfield query input
pub struct HopfieldQuery {
    /// Query vectors (batch_size x dim)
    pub queries: DeviceBuffer<f32>,
    /// Stored patterns (num_patterns x dim)
    pub patterns: DeviceBuffer<f32>,
    /// Pattern metadata indices
    pub pattern_indices: DeviceBuffer<u32>,
    /// Beta parameter (inverse temperature)
    pub beta: f32,
    /// Number of patterns to retrieve
    pub top_k: usize,
}

/// Hopfield retrieval output
pub struct HopfieldOutput {
    /// Retrieved pattern indices (batch_size x top_k)
    pub indices: DeviceBuffer<u32>,
    /// Attention weights (batch_size x top_k)
    pub attention_weights: DeviceBuffer<f32>,
    /// Energy values (batch_size)
    pub energies: DeviceBuffer<f32>,
    /// Weighted pattern sum (batch_size x dim)
    pub retrieved_patterns: DeviceBuffer<f32>,
}

impl HopfieldAttentionKernel {
    /// Execute Hopfield attention retrieval
    ///
    /// # Energy Function
    /// E(x) = -beta * log(sum_i exp(beta * x^T * xi))
    ///
    /// # Attention
    /// p_i = softmax(beta * query^T * patterns)
    ///
    /// `Constraint: Latency < 1ms for 100K patterns, k=100`
    pub async fn execute(
        &self,
        query: &HopfieldQuery,
        stream: &CudaStream,
    ) -> Result<HopfieldOutput, CudaError>;

    /// Update beta parameter (dopamine modulation)
    pub fn set_beta(&mut self, beta: f32);
}
```

**Kernel Implementation (CUDA C++):**

```cuda
// hopfield_attention.cu
// Modern Hopfield Network attention kernel with CUDA Tile optimization

#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define TILE_K 32
#define TILE_N 128
#define DIM 1536

/// CUDA Tile based attention for memory efficiency
/// Processes patterns in tiles to minimize memory bandwidth
__global__ void hopfield_attention_kernel(
    const float* __restrict__ queries,      // [batch, dim]
    const float* __restrict__ patterns,     // [num_patterns, dim]
    const uint32_t* __restrict__ pattern_indices,
    float* __restrict__ attention_weights,  // [batch, top_k]
    uint32_t* __restrict__ top_k_indices,   // [batch, top_k]
    float* __restrict__ energies,           // [batch]
    float* __restrict__ retrieved,          // [batch, dim]
    const float beta,
    const int batch_size,
    const int num_patterns,
    const int top_k
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    __shared__ float smem_query[DIM];
    __shared__ float smem_scores[TILE_N];
    __shared__ float smem_max_score;
    __shared__ float smem_sum_exp;

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    // Load query into shared memory
    for (int d = tid; d < DIM; d += blockDim.x) {
        smem_query[d] = queries[batch_idx * DIM + d];
    }
    block.sync();

    // Phase 1: Compute all dot products with patterns (tiled)
    float local_max = -INFINITY;
    float local_scores[TILE_N / 256]; // Per-thread score buffer

    for (int tile = 0; tile < num_patterns; tile += TILE_N) {
        // Compute tile of dot products
        for (int p = tid; p < TILE_N && (tile + p) < num_patterns; p += blockDim.x) {
            float dot = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < DIM; d++) {
                dot += smem_query[d] * patterns[(tile + p) * DIM + d];
            }
            dot *= beta;
            smem_scores[p] = dot;
            local_max = fmaxf(local_max, dot);
        }
        block.sync();
    }

    // Phase 2: Stable softmax with max reduction
    // Reduce max across block
    smem_max_score = cg::reduce(block, local_max, cg::greater<float>());
    block.sync();

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int tile = 0; tile < num_patterns; tile += TILE_N) {
        for (int p = tid; p < TILE_N && (tile + p) < num_patterns; p += blockDim.x) {
            float exp_score = expf(smem_scores[p] - smem_max_score);
            smem_scores[p] = exp_score;
            local_sum += exp_score;
        }
    }
    smem_sum_exp = cg::reduce(block, local_sum, cg::plus<float>());
    block.sync();

    // Phase 3: Normalize and find top-k
    // Use partial sorting for top-k selection
    // ...

    // Phase 4: Compute energy
    if (tid == 0) {
        energies[batch_idx] = -logf(smem_sum_exp) / beta + smem_max_score;
    }

    // Phase 5: Compute weighted pattern sum
    // ...
}

extern "C" void launch_hopfield_attention(
    const float* queries,
    const float* patterns,
    const uint32_t* pattern_indices,
    float* attention_weights,
    uint32_t* top_k_indices,
    float* energies,
    float* retrieved,
    float beta,
    int batch_size,
    int num_patterns,
    int top_k,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(256);

    hopfield_attention_kernel<<<grid, block, 0, stream>>>(
        queries, patterns, pattern_indices,
        attention_weights, top_k_indices, energies, retrieved,
        beta, batch_size, num_patterns, top_k
    );
}
```

### 4.3 neuromodulation_kernel

Parallel neuromodulator update computation.

```rust
/// Neuromodulation kernel interface
pub struct NeuromodulationKernel {
    /// Kernel module handle
    module: CudaModule,
    /// Kernel function handle
    function: CudaFunction,
}

/// Neuromodulator state batch
pub struct NeuromodulatorBatch {
    /// UTL state vectors (batch_size x 4): [delta_s, delta_c, w_e, phi]
    pub utl_states: DeviceBuffer<f32>,
    /// Current modulator levels (batch_size x 4): [dopamine, serotonin, noradrenaline, acetylcholine]
    pub modulator_levels: DeviceBuffer<f32>,
    /// Update rates
    pub update_rates: DeviceBuffer<f32>,
}

/// Neuromodulator output
pub struct NeuromodulatorOutput {
    /// Updated modulator levels (batch_size x 4)
    pub updated_levels: DeviceBuffer<f32>,
    /// Mapped parameter values (batch_size x 4): [hopfield_beta, fuse_moe_top_k, attention_temp, learning_rate]
    pub mapped_parameters: DeviceBuffer<f32>,
}

impl NeuromodulationKernel {
    /// Execute parallel neuromodulator updates
    ///
    /// `Constraint: Latency < 200us per batch`
    pub async fn execute(
        &self,
        batch: &NeuromodulatorBatch,
        stream: &CudaStream,
    ) -> Result<NeuromodulatorOutput, CudaError>;
}
```

**Kernel Implementation (CUDA C++):**

```cuda
// neuromodulation.cu
// Parallel neuromodulator update kernel

#define DOPAMINE_IDX 0
#define SEROTONIN_IDX 1
#define NORADRENALINE_IDX 2
#define ACETYLCHOLINE_IDX 3

#define DELTA_S_IDX 0
#define DELTA_C_IDX 1
#define W_E_IDX 2
#define PHI_IDX 3

// Parameter mapping ranges
__constant__ float c_dopamine_range[2] = {1.0f, 5.0f};      // hopfield.beta
__constant__ float c_serotonin_range[2] = {2.0f, 8.0f};     // fuse_moe.top_k
__constant__ float c_noradrenaline_range[2] = {0.5f, 2.0f}; // attention.temperature
__constant__ float c_acetylcholine_range[2] = {0.001f, 0.002f}; // utl.learning_rate

/// Parallel neuromodulator update kernel
__global__ void neuromodulation_kernel(
    const float* __restrict__ utl_states,      // [batch, 4]
    float* __restrict__ modulator_levels,      // [batch, 4]
    const float* __restrict__ update_rates,    // [4]
    float* __restrict__ mapped_parameters,     // [batch, 4]
    const int batch_size
) {
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    // Load UTL state
    const float delta_s = utl_states[batch_idx * 4 + DELTA_S_IDX];
    const float delta_c = utl_states[batch_idx * 4 + DELTA_C_IDX];
    const float w_e = utl_states[batch_idx * 4 + W_E_IDX];
    const float phi = utl_states[batch_idx * 4 + PHI_IDX];

    // Load current modulator levels
    float dopamine = modulator_levels[batch_idx * 4 + DOPAMINE_IDX];
    float serotonin = modulator_levels[batch_idx * 4 + SEROTONIN_IDX];
    float noradrenaline = modulator_levels[batch_idx * 4 + NORADRENALINE_IDX];
    float acetylcholine = modulator_levels[batch_idx * 4 + ACETYLCHOLINE_IDX];

    // Update rates
    const float rate_d = update_rates[DOPAMINE_IDX];
    const float rate_s = update_rates[SEROTONIN_IDX];
    const float rate_n = update_rates[NORADRENALINE_IDX];
    const float rate_a = update_rates[ACETYLCHOLINE_IDX];

    // Dopamine: Responds to surprise (delta_s) and positive outcomes
    // High surprise -> dopamine surge
    float target_dopamine = delta_s * w_e;
    dopamine = dopamine + rate_d * (target_dopamine - dopamine);

    // Serotonin: Responds to coherence and exploration needs
    // Low coherence -> increase exploration (higher top_k)
    float target_serotonin = 1.0f - delta_c;
    serotonin = serotonin + rate_s * (target_serotonin - serotonin);

    // Noradrenaline: Responds to arousal/surprise
    // High surprise -> flatter attention (exploration)
    float target_noradrenaline = delta_s;
    noradrenaline = noradrenaline + rate_n * (target_noradrenaline - noradrenaline);

    // Acetylcholine: Responds to learning potential
    // High learning score -> higher learning rate
    float learning_score = (delta_s * delta_c) * w_e * cosf(phi);
    float target_acetylcholine = fminf(1.0f, learning_score * 2.0f);
    acetylcholine = acetylcholine + rate_a * (target_acetylcholine - acetylcholine);

    // Clamp to [0, 1]
    dopamine = fmaxf(0.0f, fminf(1.0f, dopamine));
    serotonin = fmaxf(0.0f, fminf(1.0f, serotonin));
    noradrenaline = fmaxf(0.0f, fminf(1.0f, noradrenaline));
    acetylcholine = fmaxf(0.0f, fminf(1.0f, acetylcholine));

    // Store updated levels
    modulator_levels[batch_idx * 4 + DOPAMINE_IDX] = dopamine;
    modulator_levels[batch_idx * 4 + SEROTONIN_IDX] = serotonin;
    modulator_levels[batch_idx * 4 + NORADRENALINE_IDX] = noradrenaline;
    modulator_levels[batch_idx * 4 + ACETYLCHOLINE_IDX] = acetylcholine;

    // Map to parameter ranges
    float hopfield_beta = c_dopamine_range[0] + dopamine * (c_dopamine_range[1] - c_dopamine_range[0]);
    float fuse_moe_top_k = c_serotonin_range[0] + serotonin * (c_serotonin_range[1] - c_serotonin_range[0]);
    float attention_temp = c_noradrenaline_range[0] + noradrenaline * (c_noradrenaline_range[1] - c_noradrenaline_range[0]);
    float learning_rate = c_acetylcholine_range[0] + acetylcholine * (c_acetylcholine_range[1] - c_acetylcholine_range[0]);

    // Store mapped parameters
    mapped_parameters[batch_idx * 4 + 0] = hopfield_beta;
    mapped_parameters[batch_idx * 4 + 1] = fuse_moe_top_k;
    mapped_parameters[batch_idx * 4 + 2] = attention_temp;
    mapped_parameters[batch_idx * 4 + 3] = learning_rate;
}

extern "C" void launch_neuromodulation(
    const float* utl_states,
    float* modulator_levels,
    const float* update_rates,
    float* mapped_parameters,
    int batch_size,
    cudaStream_t stream
) {
    dim3 grid((batch_size + 255) / 256);
    dim3 block(256);

    neuromodulation_kernel<<<grid, block, 0, stream>>>(
        utl_states, modulator_levels, update_rates, mapped_parameters, batch_size
    );
}
```

### 4.4 cone_containment_kernel

Batch hyperbolic cone containment checks for entailment queries.

```rust
/// Cone containment kernel interface
pub struct ConeContainmentKernel {
    /// Kernel module handle
    module: CudaModule,
    /// Kernel function handle
    function: CudaFunction,
}

/// Hyperbolic cone batch
pub struct ConeBatch {
    /// Cone apex points in Poincare ball (num_cones x 64)
    pub apexes: DeviceBuffer<f32>,
    /// Cone aperture angles (num_cones)
    pub apertures: DeviceBuffer<f32>,
    /// Query points to check (num_queries x 64)
    pub query_points: DeviceBuffer<f32>,
}

/// Containment check output
pub struct ContainmentOutput {
    /// Containment matrix (num_queries x num_cones)
    pub contained: DeviceBuffer<u8>,
    /// Distance to cone boundary (num_queries x num_cones)
    pub distances: DeviceBuffer<f32>,
}

impl ConeContainmentKernel {
    /// Execute batch containment checks
    ///
    /// # Hyperbolic Distance in Poincare Ball
    /// d(x, y) = arcosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2)(1 - ||y||^2)))
    ///
    /// # Cone Containment
    /// Point p is in cone with apex a and aperture theta if:
    /// angle(a, p) <= theta
    ///
    /// `Constraint: O(1) per check, Latency < 1ms for 10K checks`
    pub async fn execute(
        &self,
        batch: &ConeBatch,
        stream: &CudaStream,
    ) -> Result<ContainmentOutput, CudaError>;
}
```

**Kernel Implementation (CUDA C++):**

```cuda
// cone_containment.cu
// Batch hyperbolic cone containment kernel for Poincare ball model

#define HYPERBOLIC_DIM 64
#define EPS 1e-6f

/// Compute squared norm
__device__ __forceinline__ float squared_norm(const float* x, int dim) {
    float norm = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < dim; i++) {
        norm += x[i] * x[i];
    }
    return norm;
}

/// Compute hyperbolic distance in Poincare ball
__device__ float poincare_distance(const float* x, const float* y, int dim) {
    float sq_norm_x = squared_norm(x, dim);
    float sq_norm_y = squared_norm(y, dim);

    float sq_diff = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - y[i];
        sq_diff += d * d;
    }

    float denom = (1.0f - sq_norm_x) * (1.0f - sq_norm_y);
    denom = fmaxf(denom, EPS);

    float arg = 1.0f + 2.0f * sq_diff / denom;
    return acoshf(fmaxf(arg, 1.0f + EPS));
}

/// Compute angle between vectors in hyperbolic space
__device__ float hyperbolic_angle(const float* apex, const float* point, int dim) {
    // Project to tangent space at apex
    float sq_norm_apex = squared_norm(apex, dim);
    float lambda_apex = 2.0f / fmaxf(1.0f - sq_norm_apex, EPS);

    // Logarithmic map
    float diff[HYPERBOLIC_DIM];
    float apex_plus_point[HYPERBOLIC_DIM];

    for (int i = 0; i < dim; i++) {
        apex_plus_point[i] = apex[i] + point[i];
    }

    // Mobius addition: -apex + point
    float sq_norm_sum = squared_norm(apex_plus_point, dim);
    float dot = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += apex[i] * point[i];
    }

    float coef1 = (1.0f + 2.0f * dot + sq_norm_apex);
    float coef2 = (1.0f - sq_norm_apex);
    float denom = coef1 + sq_norm_sum * sq_norm_apex;
    denom = fmaxf(denom, EPS);

    float v_norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        float v_i = (coef1 * point[i] - coef2 * apex[i]) / denom;
        v_norm += v_i * v_i;
    }

    // Angle from tangent vector
    v_norm = sqrtf(fmaxf(v_norm, EPS));
    return 2.0f * atanhf(fminf(v_norm, 1.0f - EPS)) / lambda_apex;
}

/// Batch cone containment check kernel
__global__ void cone_containment_kernel(
    const float* __restrict__ apexes,     // [num_cones, dim]
    const float* __restrict__ apertures,  // [num_cones]
    const float* __restrict__ points,     // [num_queries, dim]
    uint8_t* __restrict__ contained,      // [num_queries, num_cones]
    float* __restrict__ distances,        // [num_queries, num_cones]
    const int num_cones,
    const int num_queries
) {
    const int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cone_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (query_idx >= num_queries || cone_idx >= num_cones) return;

    // Load apex and aperture
    float apex[HYPERBOLIC_DIM];
    for (int i = 0; i < HYPERBOLIC_DIM; i++) {
        apex[i] = apexes[cone_idx * HYPERBOLIC_DIM + i];
    }
    float aperture = apertures[cone_idx];

    // Load query point
    float point[HYPERBOLIC_DIM];
    for (int i = 0; i < HYPERBOLIC_DIM; i++) {
        point[i] = points[query_idx * HYPERBOLIC_DIM + i];
    }

    // Compute angle from apex to point
    float angle = hyperbolic_angle(apex, point, HYPERBOLIC_DIM);

    // Check containment
    int out_idx = query_idx * num_cones + cone_idx;
    contained[out_idx] = (angle <= aperture) ? 1 : 0;
    distances[out_idx] = aperture - angle; // Positive if inside, negative if outside
}

extern "C" void launch_cone_containment(
    const float* apexes,
    const float* apertures,
    const float* points,
    uint8_t* contained,
    float* distances,
    int num_cones,
    int num_queries,
    cudaStream_t stream
) {
    dim3 grid((num_queries + 15) / 16, (num_cones + 15) / 16);
    dim3 block(16, 16);

    cone_containment_kernel<<<grid, block, 0, stream>>>(
        apexes, apertures, points, contained, distances, num_cones, num_queries
    );
}
```

---

## 5. Memory Management

### 5.1 Memory Pool Architecture

```rust
/// GPU memory pool manager
pub struct MemoryPoolManager {
    /// Device memory pool for general allocations
    device_pool: CudaMemPool,
    /// Pinned host memory pool for transfers
    pinned_pool: PinnedMemPool,
    /// UVM allocator for large graphs
    uvm_allocator: UVMAllocator,
    /// Pool statistics
    stats: MemoryPoolStats,
}

/// Memory pool configuration
pub struct MemoryPoolConfig {
    /// Maximum device memory pool size
    pub device_pool_size: usize,
    /// Maximum pinned memory pool size
    pub pinned_pool_size: usize,
    /// UVM migration threshold
    pub uvm_threshold: usize,
    /// Allocation alignment (default: 256)
    pub alignment: usize,
    /// Enable memory trimming
    pub trim_enabled: bool,
    /// Trim threshold (percentage)
    pub trim_threshold: f32,
}

impl MemoryPoolManager {
    /// Allocate device memory from pool
    ///
    /// `Constraint: Allocation latency < 1us for cached sizes`
    pub fn allocate_device(&self, size: usize) -> Result<DeviceBuffer, CudaError>;

    /// Allocate pinned host memory
    ///
    /// `Constraint: Allocation latency < 10us`
    pub fn allocate_pinned(&self, size: usize) -> Result<PinnedBuffer, CudaError>;

    /// Allocate UVM memory for large graphs
    ///
    /// Automatically migrates between host and device based on access patterns
    pub fn allocate_uvm(&self, size: usize) -> Result<UVMBuffer, CudaError>;

    /// Free memory back to pool
    pub fn free(&self, buffer: impl MemoryBuffer);

    /// Get pool statistics
    pub fn stats(&self) -> MemoryPoolStats;

    /// Trim unused memory
    pub fn trim(&self) -> Result<usize, CudaError>;
}

/// Memory pool statistics
pub struct MemoryPoolStats {
    /// Total allocated device memory
    pub device_allocated: usize,
    /// Total allocated pinned memory
    pub pinned_allocated: usize,
    /// Total allocated UVM memory
    pub uvm_allocated: usize,
    /// Pool hit rate (allocations satisfied from pool)
    pub pool_hit_rate: f32,
    /// Average allocation latency
    pub avg_allocation_latency: Duration,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Current fragmentation ratio
    pub fragmentation: f32,
}
```

### 5.2 Unified Virtual Memory (UVM)

```rust
/// UVM allocator for large graph storage
pub struct UVMAllocator {
    /// Base UVM allocation
    base_ptr: *mut u8,
    /// Total capacity
    capacity: usize,
    /// Page size
    page_size: usize,
    /// Migration hints
    migration_hints: MigrationHints,
}

/// Migration hints for UVM pages
pub struct MigrationHints {
    /// Preferred location (device or host)
    pub preferred_location: MemoryLocation,
    /// Access pattern hint
    pub access_pattern: AccessPattern,
    /// Read-mostly optimization
    pub read_mostly: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryLocation {
    Device(u32),
    Host,
    Any,
}

#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    /// Mostly sequential access
    Sequential,
    /// Random access patterns
    Random,
    /// Streaming access
    Streaming,
}

impl UVMAllocator {
    /// Prefetch memory to device
    pub async fn prefetch_to_device(&self, offset: usize, size: usize, stream: &CudaStream) -> Result<(), CudaError>;

    /// Prefetch memory to host
    pub async fn prefetch_to_host(&self, offset: usize, size: usize) -> Result<(), CudaError>;

    /// Set access hints for memory range
    pub fn set_access_hints(&self, offset: usize, size: usize, hints: MigrationHints) -> Result<(), CudaError>;

    /// Get page fault statistics
    pub fn page_fault_stats(&self) -> PageFaultStats;
}
```

### 5.3 Pinned Memory for CPU-GPU Transfers

```rust
/// Pinned memory buffer for efficient transfers
pub struct PinnedBuffer {
    /// Host pointer (pinned)
    host_ptr: *mut u8,
    /// Size in bytes
    size: usize,
    /// Transfer direction hint
    direction: TransferDirection,
}

#[derive(Debug, Clone, Copy)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
    Bidirectional,
}

impl PinnedBuffer {
    /// Async copy to device
    pub async fn copy_to_device(&self, device_buffer: &DeviceBuffer, stream: &CudaStream) -> Result<(), CudaError>;

    /// Async copy from device
    pub async fn copy_from_device(&self, device_buffer: &DeviceBuffer, stream: &CudaStream) -> Result<(), CudaError>;

    /// Get transfer bandwidth statistics
    pub fn transfer_stats(&self) -> TransferStats;
}

pub struct TransferStats {
    /// Average transfer bandwidth (GB/s)
    pub avg_bandwidth: f64,
    /// Total bytes transferred
    pub total_bytes: usize,
    /// Transfer count
    pub transfer_count: u64,
}
```

---

## 6. Stream Orchestration

### 6.1 Multi-Stream Execution

```rust
/// CUDA stream manager for parallel kernel execution
pub struct StreamManager {
    /// Primary streams (8 by default)
    streams: Vec<CudaStream>,
    /// Stream priorities
    priorities: Vec<StreamPriority>,
    /// Kernel dependency graph
    dependency_graph: DependencyGraph,
    /// Stream utilization metrics
    utilization: StreamUtilization,
}

/// Stream priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamPriority {
    High = 0,
    Normal = 1,
    Low = 2,
}

/// Stream assignment for workload types
pub struct StreamAssignment {
    /// Streams for embedding operations (0-1)
    pub embedding_streams: [usize; 2],
    /// Streams for Hopfield operations (2-3)
    pub hopfield_streams: [usize; 2],
    /// Streams for neuromodulation (4-5)
    pub neuromod_streams: [usize; 2],
    /// Streams for utility operations (6-7)
    pub utility_streams: [usize; 2],
}

impl StreamManager {
    /// Submit kernel to appropriate stream
    pub fn submit<K: CudaKernel>(&self, kernel: K, assignment: StreamAssignment) -> StreamFuture<K::Output>;

    /// Wait for all streams to complete
    pub async fn synchronize_all(&self) -> Result<(), CudaError>;

    /// Wait for specific stream
    pub async fn synchronize_stream(&self, stream_idx: usize) -> Result<(), CudaError>;

    /// Add dependency between kernels
    pub fn add_dependency(&mut self, source: KernelId, target: KernelId);

    /// Get stream utilization metrics
    pub fn utilization(&self) -> StreamUtilization;
}

pub struct StreamUtilization {
    /// Per-stream utilization percentage
    pub stream_utilization: [f32; 8],
    /// Overall GPU utilization
    pub gpu_utilization: f32,
    /// Kernel overlap percentage
    pub kernel_overlap: f32,
    /// Average queue depth
    pub avg_queue_depth: f32,
}
```

### 6.2 Kernel Dependency Scheduling

```rust
/// Kernel dependency graph for scheduling
pub struct DependencyGraph {
    /// Kernel nodes
    nodes: HashMap<KernelId, KernelNode>,
    /// Dependency edges
    edges: Vec<(KernelId, KernelId)>,
}

/// Kernel node in dependency graph
pub struct KernelNode {
    /// Kernel identifier
    pub id: KernelId,
    /// Kernel type
    pub kernel_type: KernelType,
    /// Assigned stream
    pub stream: Option<usize>,
    /// Dependencies (must complete before this kernel)
    pub dependencies: Vec<KernelId>,
    /// Dependents (kernels waiting on this)
    pub dependents: Vec<KernelId>,
    /// Estimated execution time
    pub estimated_time: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
    FusedEmbedding,
    HopfieldAttention,
    Neuromodulation,
    ConeContainment,
    MemoryCopy,
    Custom(u32),
}

impl DependencyGraph {
    /// Get optimal kernel execution order
    pub fn topological_sort(&self) -> Vec<KernelId>;

    /// Assign kernels to streams based on dependencies
    pub fn assign_streams(&mut self, num_streams: usize) -> StreamAssignment;

    /// Insert CUDA events for synchronization
    pub fn create_synchronization_events(&self) -> Vec<(KernelId, CudaEvent)>;
}
```

---

## 7. CPU Fallback

### 7.1 Fallback Implementation

```rust
/// CPU fallback for when GPU is unavailable
pub struct CpuFallback {
    /// Thread pool for parallel CPU execution
    thread_pool: rayon::ThreadPool,
    /// SIMD support flags
    simd_support: SimdSupport,
}

pub struct SimdSupport {
    pub avx512: bool,
    pub avx2: bool,
    pub sse4: bool,
}

impl CpuFallback {
    /// CPU implementation of fused embedding
    pub fn fused_embedding(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput, FallbackError>;

    /// CPU implementation of Hopfield attention
    pub fn hopfield_attention(&self, query: &HopfieldQuery) -> Result<HopfieldOutput, FallbackError>;

    /// CPU implementation of neuromodulation
    pub fn neuromodulation(&self, batch: &NeuromodulatorBatch) -> Result<NeuromodulatorOutput, FallbackError>;

    /// CPU implementation of cone containment
    pub fn cone_containment(&self, batch: &ConeBatch) -> Result<ContainmentOutput, FallbackError>;
}
```

### 7.2 Automatic Fallback Selection

```rust
/// Automatic GPU/CPU selection
pub struct ComputeBackend {
    /// GPU backend (if available)
    gpu: Option<GpuBackend>,
    /// CPU fallback
    cpu: CpuFallback,
    /// Selection policy
    policy: SelectionPolicy,
}

pub enum SelectionPolicy {
    /// Always use GPU if available
    PreferGpu,
    /// Use GPU for large batches only
    AdaptiveBatchSize { threshold: usize },
    /// Always use CPU (debugging)
    ForceCpu,
    /// Load balance between GPU and CPU
    LoadBalance { gpu_weight: f32 },
}

impl ComputeBackend {
    /// Execute kernel with automatic backend selection
    pub async fn execute<K: ComputeKernel>(&self, kernel: K) -> Result<K::Output, ComputeError>;

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool;

    /// Get current backend selection
    pub fn current_backend(&self) -> BackendType;

    /// Force backend switch
    pub fn set_backend(&mut self, backend: BackendType);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Gpu,
    Cpu,
}
```

---

## 8. GPU Direct Storage Integration

### 8.1 GDS Interface

```rust
/// GPU Direct Storage for direct NVMe-GPU transfers
pub struct GDSManager {
    /// GDS configuration
    config: GDSConfig,
    /// cuFile driver handle
    driver: CuFileDriver,
    /// Buffer pool for GDS operations
    buffer_pool: GDSBufferPool,
}

pub struct GDSConfig {
    /// Enable GDS
    pub enabled: bool,
    /// NVMe device paths
    pub nvme_devices: Vec<PathBuf>,
    /// Buffer size per operation
    pub buffer_size: usize,
    /// Maximum concurrent operations
    pub max_concurrent_ops: u32,
}

impl GDSManager {
    /// Load model weights directly to GPU via GDS
    ///
    /// `Constraint: 4x faster than standard filesystem load`
    pub async fn load_model_weights(&self, path: &Path, device_buffer: &mut DeviceBuffer) -> Result<(), GDSError>;

    /// Load FAISS index directly to GPU
    pub async fn load_faiss_index(&self, path: &Path) -> Result<FAISSIndex, GDSError>;

    /// Stream memory shards from NVMe to GPU
    pub async fn stream_shards(&self, shard_paths: &[PathBuf], stream: &CudaStream) -> Result<Vec<DeviceBuffer>, GDSError>;

    /// Check GDS availability
    pub fn is_available(&self) -> bool;

    /// Get GDS bandwidth statistics
    pub fn bandwidth_stats(&self) -> GDSBandwidthStats;
}

pub struct GDSBandwidthStats {
    /// Read bandwidth (GB/s)
    pub read_bandwidth: f64,
    /// Write bandwidth (GB/s)
    pub write_bandwidth: f64,
    /// Operations completed
    pub ops_completed: u64,
    /// Total bytes transferred
    pub bytes_transferred: usize,
}
```

---

## 9. Requirements Specification

### 9.1 Functional Requirements

#### Device Configuration

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-CUDA-001 | System SHALL detect CUDA device with compute capability 12.0 | Critical | Unit test |
| REQ-CUDA-002 | System SHALL initialize Green Contexts with configurable SM partitioning | Critical | Integration test |
| REQ-CUDA-003 | System SHALL support 8 concurrent CUDA streams | Critical | Benchmark |
| REQ-CUDA-004 | System SHALL verify 32GB VRAM availability on RTX 5090 | High | Unit test |
| REQ-CUDA-005 | System SHALL fall back to CPU when GPU unavailable | Critical | Integration test |

#### Custom Kernels

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-CUDA-006 | fused_embedding_kernel SHALL compute 12-model embeddings in single pass | Critical | Unit test |
| REQ-CUDA-007 | fused_embedding_kernel SHALL support FP8/FP4 precision modes | High | Benchmark |
| REQ-CUDA-008 | fused_embedding_kernel SHALL achieve <10ms latency for batch_size=1 | Critical | Latency test |
| REQ-CUDA-009 | fused_embedding_kernel SHALL achieve >1000/sec throughput for batch_size=64 | Critical | Throughput test |
| REQ-CUDA-010 | hopfield_attention_kernel SHALL compute Modern Hopfield energy function | Critical | Unit test |
| REQ-CUDA-011 | hopfield_attention_kernel SHALL retrieve top-k patterns in <1ms for 100K patterns | Critical | Latency test |
| REQ-CUDA-012 | hopfield_attention_kernel SHALL support dynamic beta modulation | High | Unit test |
| REQ-CUDA-013 | neuromodulation_kernel SHALL update 4 neuromodulator channels in parallel | Critical | Unit test |
| REQ-CUDA-014 | neuromodulation_kernel SHALL complete updates in <200us | Critical | Latency test |
| REQ-CUDA-015 | neuromodulation_kernel SHALL map modulators to parameter ranges | High | Unit test |
| REQ-CUDA-016 | cone_containment_kernel SHALL perform O(1) hyperbolic cone checks | Critical | Unit test |
| REQ-CUDA-017 | cone_containment_kernel SHALL process 10K checks in <1ms | Critical | Latency test |
| REQ-CUDA-018 | cone_containment_kernel SHALL compute Poincare ball distances correctly | High | Unit test |

#### Memory Management

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-CUDA-019 | System SHALL implement memory pool to reduce allocation overhead | Critical | Benchmark |
| REQ-CUDA-020 | Memory pool allocation latency SHALL be <1us for cached sizes | High | Latency test |
| REQ-CUDA-021 | System SHALL support Unified Virtual Memory for graphs >VRAM | High | Integration test |
| REQ-CUDA-022 | UVM SHALL automatically migrate pages based on access patterns | High | Benchmark |
| REQ-CUDA-023 | System SHALL use pinned memory for CPU-GPU transfers | Critical | Benchmark |
| REQ-CUDA-024 | System SHALL maintain <24GB VRAM usage under load | Critical | Load test |
| REQ-CUDA-025 | Memory pool fragmentation SHALL remain <20% | Medium | Benchmark |

#### Stream Orchestration

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-CUDA-026 | System SHALL execute kernels on 8 concurrent streams | Critical | Benchmark |
| REQ-CUDA-027 | Stream scheduler SHALL resolve kernel dependencies | Critical | Unit test |
| REQ-CUDA-028 | System SHALL support stream priorities | High | Unit test |
| REQ-CUDA-029 | Multi-stream execution SHALL achieve 3x wall-time reduction | High | Benchmark |

#### Performance

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-CUDA-030 | System SHALL achieve >80% GPU utilization | Critical | Profiling |
| REQ-CUDA-031 | System SHALL achieve 5x speedup over CPU baseline | Critical | Benchmark |
| REQ-CUDA-032 | Kernel launch overhead SHALL be <10us | High | Profiling |
| REQ-CUDA-033 | Green Contexts power reduction SHALL be >20% vs baseline | Medium | Power test |

#### GPU Direct Storage

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-CUDA-034 | GDS SHALL load models 4x faster than standard filesystem | High | Benchmark |
| REQ-CUDA-035 | GDS SHALL load FAISS index directly to GPU memory | High | Integration test |
| REQ-CUDA-036 | System SHALL fall back to standard I/O when GDS unavailable | Critical | Integration test |

#### Precision

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-CUDA-037 | FP8 precision SHALL maintain embedding quality within 2% of FP32 | High | Quality test |
| REQ-CUDA-038 | FP4 precision SHALL be available for inference-only workloads | Medium | Unit test |
| REQ-CUDA-039 | System SHALL support mixed precision (FP8 compute, FP16 accumulate) | High | Unit test |

### 9.2 Non-Functional Requirements

| ID | Requirement | Category | Target |
|----|-------------|----------|--------|
| REQ-CUDA-040 | GPU memory usage | Resource | <24GB under load |
| REQ-CUDA-041 | Peak GPU utilization | Performance | >80% |
| REQ-CUDA-042 | Speedup vs CPU | Performance | >5x |
| REQ-CUDA-043 | Kernel launch overhead | Performance | <10us |
| REQ-CUDA-044 | Memory allocation latency | Performance | <1us (cached) |
| REQ-CUDA-045 | Stream overlap efficiency | Performance | >70% |
| REQ-CUDA-046 | Unit test coverage | Quality | >90% |
| REQ-CUDA-047 | Integration test coverage | Quality | >80% |
| REQ-CUDA-048 | API documentation coverage | Quality | >80% |
| REQ-CUDA-049 | Zero CUDA runtime errors | Reliability | 0 under stress |
| REQ-CUDA-050 | Graceful degradation | Reliability | 100% fallback success |

---

## 10. Test Cases

### 10.1 Kernel Performance Tests

```rust
#[cfg(test)]
mod cuda_performance_tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// TC-CUDA-001: Fused embedding kernel latency
    #[tokio::test]
    async fn test_fused_embedding_latency() {
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let kernel = FusedEmbeddingKernel::new(&cuda_ctx).unwrap();
        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();

        // Warmup
        let warmup_batch = create_test_batch(1, 512);
        kernel.warmup(1).await.unwrap();

        // Measure single inference latency
        let batch = create_test_batch(1, 512);
        let mut latencies = Vec::with_capacity(100);

        for _ in 0..100 {
            let start = Instant::now();
            let _ = kernel.execute(&batch, &stream, InferencePrecision::FP8).await.unwrap();
            stream.synchronize().await.unwrap();
            latencies.push(start.elapsed());
        }

        let p95 = percentile(&latencies, 0.95);
        assert!(
            p95 < Duration::from_millis(10),
            "Fused embedding P95 latency {} exceeds 10ms",
            p95.as_millis()
        );
    }

    /// TC-CUDA-002: Fused embedding kernel throughput
    #[tokio::test]
    async fn test_fused_embedding_throughput() {
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let kernel = FusedEmbeddingKernel::new(&cuda_ctx).unwrap();
        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();

        let batch_size = 64;
        let batch = create_test_batch(batch_size, 512);

        // Measure throughput
        let start = Instant::now();
        let iterations = 100;

        for _ in 0..iterations {
            let _ = kernel.execute(&batch, &stream, InferencePrecision::FP8).await.unwrap();
        }
        stream.synchronize().await.unwrap();

        let elapsed = start.elapsed();
        let throughput = (batch_size * iterations) as f64 / elapsed.as_secs_f64();

        assert!(
            throughput > 1000.0,
            "Fused embedding throughput {} below 1000/sec",
            throughput
        );
    }

    /// TC-CUDA-003: Hopfield attention latency
    #[tokio::test]
    async fn test_hopfield_attention_latency() {
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let kernel = HopfieldAttentionKernel::new(&cuda_ctx).unwrap();
        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();

        // Create 100K patterns
        let query = create_hopfield_query(1, 100_000, 100);

        let mut latencies = Vec::with_capacity(100);
        for _ in 0..100 {
            let start = Instant::now();
            let _ = kernel.execute(&query, &stream).await.unwrap();
            stream.synchronize().await.unwrap();
            latencies.push(start.elapsed());
        }

        let p95 = percentile(&latencies, 0.95);
        assert!(
            p95 < Duration::from_millis(1),
            "Hopfield attention P95 latency {} exceeds 1ms",
            p95.as_micros()
        );
    }

    /// TC-CUDA-004: Neuromodulation latency
    #[tokio::test]
    async fn test_neuromodulation_latency() {
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let kernel = NeuromodulationKernel::new(&cuda_ctx).unwrap();
        let stream = cuda_ctx.create_stream(StreamPriority::Normal).unwrap();

        let batch = create_neuromod_batch(1000);

        let mut latencies = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let start = Instant::now();
            let _ = kernel.execute(&batch, &stream).await.unwrap();
            stream.synchronize().await.unwrap();
            latencies.push(start.elapsed());
        }

        let p95 = percentile(&latencies, 0.95);
        assert!(
            p95 < Duration::from_micros(200),
            "Neuromodulation P95 latency {} exceeds 200us",
            p95.as_nanos()
        );
    }

    /// TC-CUDA-005: Cone containment latency
    #[tokio::test]
    async fn test_cone_containment_latency() {
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let kernel = ConeContainmentKernel::new(&cuda_ctx).unwrap();
        let stream = cuda_ctx.create_stream(StreamPriority::Normal).unwrap();

        // 10K checks
        let batch = create_cone_batch(100, 100); // 100 cones x 100 queries = 10K checks

        let mut latencies = Vec::with_capacity(100);
        for _ in 0..100 {
            let start = Instant::now();
            let _ = kernel.execute(&batch, &stream).await.unwrap();
            stream.synchronize().await.unwrap();
            latencies.push(start.elapsed());
        }

        let p95 = percentile(&latencies, 0.95);
        assert!(
            p95 < Duration::from_millis(1),
            "Cone containment P95 latency {} exceeds 1ms for 10K checks",
            p95.as_micros()
        );
    }
}
```

### 10.2 Memory Management Tests

```rust
#[cfg(test)]
mod memory_tests {
    use super::*;

    /// TC-CUDA-006: Memory pool allocation performance
    #[test]
    fn test_memory_pool_allocation() {
        let pool = MemoryPoolManager::new(MemoryPoolConfig::default()).unwrap();

        // Test cached allocation
        let sizes = [1024, 4096, 16384, 65536, 262144];

        for size in sizes {
            // First allocation (may not be cached)
            let buf1 = pool.allocate_device(size).unwrap();
            pool.free(buf1);

            // Second allocation should be fast (cached)
            let start = Instant::now();
            let buf2 = pool.allocate_device(size).unwrap();
            let elapsed = start.elapsed();

            assert!(
                elapsed < Duration::from_micros(1),
                "Cached allocation took {} for size {}",
                elapsed.as_nanos(),
                size
            );

            pool.free(buf2);
        }
    }

    /// TC-CUDA-007: VRAM usage under load
    #[tokio::test]
    async fn test_vram_usage_under_load() {
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();

        // Simulate full workload
        let embedding_kernel = FusedEmbeddingKernel::new(&cuda_ctx).unwrap();
        let hopfield_kernel = HopfieldAttentionKernel::new(&cuda_ctx).unwrap();
        let neuromod_kernel = NeuromodulationKernel::new(&cuda_ctx).unwrap();
        let cone_kernel = ConeContainmentKernel::new(&cuda_ctx).unwrap();

        // Allocate typical working set
        let embedding_batch = create_test_batch(64, 512);
        let hopfield_query = create_hopfield_query(64, 1_000_000, 100);
        let neuromod_batch = create_neuromod_batch(10000);
        let cone_batch = create_cone_batch(1000, 1000);

        // Check VRAM usage
        let vram_used = cuda_ctx.get_memory_usage().used;
        let vram_limit = 24 * 1024 * 1024 * 1024; // 24GB

        assert!(
            vram_used < vram_limit,
            "VRAM usage {} exceeds 24GB limit",
            vram_used / (1024 * 1024 * 1024)
        );
    }

    /// TC-CUDA-008: UVM migration test
    #[tokio::test]
    async fn test_uvm_migration() {
        let cuda_ctx = CudaContext::new(CudaConfig {
            uvm_enabled: true,
            ..Default::default()
        }).unwrap();

        let uvm = cuda_ctx.uvm_allocator();

        // Allocate UVM buffer larger than typical prefetch
        let size = 2 * 1024 * 1024 * 1024; // 2GB
        let buffer = uvm.allocate(size).unwrap();

        // Set hints for device access
        uvm.set_access_hints(
            0,
            size,
            MigrationHints {
                preferred_location: MemoryLocation::Device(0),
                access_pattern: AccessPattern::Sequential,
                read_mostly: true,
            },
        ).unwrap();

        // Prefetch to device
        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();
        uvm.prefetch_to_device(0, size, &stream).await.unwrap();

        // Access should not cause page faults
        let initial_faults = uvm.page_fault_stats().device_page_faults;
        // ... perform device access ...
        let final_faults = uvm.page_fault_stats().device_page_faults;

        assert_eq!(initial_faults, final_faults, "Page faults occurred after prefetch");
    }
}
```

### 10.3 Multi-Stream Tests

```rust
#[cfg(test)]
mod stream_tests {
    use super::*;

    /// TC-CUDA-009: Multi-stream parallel execution
    #[tokio::test]
    async fn test_multistream_execution() {
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let stream_mgr = StreamManager::new(&cuda_ctx, 8).unwrap();

        // Create independent kernels
        let embedding_kernel = FusedEmbeddingKernel::new(&cuda_ctx).unwrap();
        let hopfield_kernel = HopfieldAttentionKernel::new(&cuda_ctx).unwrap();

        let embedding_batch = create_test_batch(64, 512);
        let hopfield_query = create_hopfield_query(64, 100_000, 100);

        // Execute in parallel
        let start = Instant::now();

        let (embedding_result, hopfield_result) = tokio::join!(
            embedding_kernel.execute(&embedding_batch, &stream_mgr.get_stream(0), InferencePrecision::FP8),
            hopfield_kernel.execute(&hopfield_query, &stream_mgr.get_stream(2))
        );

        stream_mgr.synchronize_all().await.unwrap();
        let parallel_time = start.elapsed();

        // Execute sequentially for comparison
        let start = Instant::now();
        let _ = embedding_kernel.execute(&embedding_batch, &stream_mgr.get_stream(0), InferencePrecision::FP8).await.unwrap();
        stream_mgr.synchronize_stream(0).await.unwrap();
        let _ = hopfield_kernel.execute(&hopfield_query, &stream_mgr.get_stream(0)).await.unwrap();
        stream_mgr.synchronize_stream(0).await.unwrap();
        let sequential_time = start.elapsed();

        // Parallel should be significantly faster
        let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
        assert!(
            speedup > 1.5,
            "Multi-stream speedup {} below 1.5x",
            speedup
        );
    }

    /// TC-CUDA-010: Stream utilization metrics
    #[tokio::test]
    async fn test_stream_utilization() {
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let stream_mgr = StreamManager::new(&cuda_ctx, 8).unwrap();

        // Run mixed workload
        for _ in 0..100 {
            // Submit to different streams
            // ...
        }

        stream_mgr.synchronize_all().await.unwrap();
        let utilization = stream_mgr.utilization();

        assert!(
            utilization.gpu_utilization > 0.8,
            "GPU utilization {} below 80%",
            utilization.gpu_utilization
        );

        assert!(
            utilization.kernel_overlap > 0.7,
            "Kernel overlap {} below 70%",
            utilization.kernel_overlap
        );
    }
}
```

### 10.4 CPU Fallback Tests

```rust
#[cfg(test)]
mod fallback_tests {
    use super::*;

    /// TC-CUDA-011: CPU fallback activation
    #[tokio::test]
    async fn test_cpu_fallback_activation() {
        // Create backend with GPU disabled
        let backend = ComputeBackend::new(ComputeBackendConfig {
            policy: SelectionPolicy::ForceCpu,
            ..Default::default()
        }).unwrap();

        assert_eq!(backend.current_backend(), BackendType::Cpu);

        // Execute kernel
        let batch = create_test_batch(1, 512);
        let result = backend.execute(FusedEmbeddingTask { batch }).await;

        assert!(result.is_ok(), "CPU fallback failed");
    }

    /// TC-CUDA-012: Automatic fallback on GPU error
    #[tokio::test]
    async fn test_automatic_fallback() {
        let backend = ComputeBackend::new(ComputeBackendConfig {
            policy: SelectionPolicy::PreferGpu,
            cpu_fallback_enabled: true,
            ..Default::default()
        }).unwrap();

        // Simulate GPU unavailable
        backend.simulate_gpu_failure();

        // Should automatically fall back to CPU
        let batch = create_test_batch(1, 512);
        let result = backend.execute(FusedEmbeddingTask { batch }).await;

        assert!(result.is_ok(), "Automatic fallback failed");
        assert_eq!(backend.current_backend(), BackendType::Cpu);
    }

    /// TC-CUDA-013: CPU fallback correctness
    #[tokio::test]
    async fn test_fallback_correctness() {
        let gpu_backend = ComputeBackend::new(ComputeBackendConfig {
            policy: SelectionPolicy::PreferGpu,
            ..Default::default()
        }).unwrap();

        let cpu_backend = ComputeBackend::new(ComputeBackendConfig {
            policy: SelectionPolicy::ForceCpu,
            ..Default::default()
        }).unwrap();

        let batch = create_test_batch(1, 512);

        let gpu_result = gpu_backend.execute(FusedEmbeddingTask { batch: batch.clone() }).await.unwrap();
        let cpu_result = cpu_backend.execute(FusedEmbeddingTask { batch }).await.unwrap();

        // Results should be close (within FP tolerance)
        let max_diff = compute_max_diff(&gpu_result.embeddings, &cpu_result.embeddings);
        assert!(
            max_diff < 0.01,
            "GPU/CPU result difference {} exceeds tolerance",
            max_diff
        );
    }
}
```

### 10.5 Green Contexts Tests

```rust
#[cfg(test)]
mod green_context_tests {
    use super::*;

    /// TC-CUDA-014: Green Context initialization
    #[test]
    fn test_green_context_init() {
        let config = CudaConfig {
            green_contexts_enabled: true,
            green_context_partitions: GreenContextConfig {
                embedding_sms: 42,
                memory_sms: 42,
                learning_sms: 42,
                utility_sms: 44,
            },
            ..Default::default()
        };

        let cuda_ctx = CudaContext::new(config).unwrap();

        // Verify SM partitioning
        let partitions = cuda_ctx.get_green_context_partitions();
        assert_eq!(partitions.embedding_sms, 42);
        assert_eq!(partitions.memory_sms, 42);
        assert_eq!(partitions.learning_sms, 42);
        assert_eq!(partitions.utility_sms, 44);

        // Total should equal device SMs
        let total = partitions.embedding_sms + partitions.memory_sms +
                   partitions.learning_sms + partitions.utility_sms;
        assert_eq!(total, 170); // RTX 5090 SMs
    }

    /// TC-CUDA-015: Green Context power efficiency
    #[tokio::test]
    async fn test_green_context_power() {
        let config_with_gc = CudaConfig {
            green_contexts_enabled: true,
            ..Default::default()
        };

        let config_without_gc = CudaConfig {
            green_contexts_enabled: false,
            ..Default::default()
        };

        // Measure power with Green Contexts
        let ctx_gc = CudaContext::new(config_with_gc).unwrap();
        let power_gc = run_workload_and_measure_power(&ctx_gc).await;

        // Measure power without Green Contexts
        let ctx_no_gc = CudaContext::new(config_without_gc).unwrap();
        let power_no_gc = run_workload_and_measure_power(&ctx_no_gc).await;

        // Green Contexts should reduce power by >20%
        let reduction = 1.0 - (power_gc / power_no_gc);
        assert!(
            reduction > 0.2,
            "Green Context power reduction {} below 20%",
            reduction
        );
    }
}
```

---

## 11. Metrics and Monitoring

### 11.1 Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `cuda.gpu_utilization` | Gauge | Current GPU utilization percentage |
| `cuda.memory_used` | Gauge | Current GPU memory usage in bytes |
| `cuda.memory_pool_hit_rate` | Gauge | Memory pool allocation hit rate |
| `cuda.kernel.fused_embedding.latency_p95` | Histogram | Fused embedding P95 latency |
| `cuda.kernel.hopfield_attention.latency_p95` | Histogram | Hopfield attention P95 latency |
| `cuda.kernel.neuromodulation.latency_p95` | Histogram | Neuromodulation P95 latency |
| `cuda.kernel.cone_containment.latency_p95` | Histogram | Cone containment P95 latency |
| `cuda.stream.utilization` | Gauge | Per-stream utilization |
| `cuda.stream.overlap` | Gauge | Kernel overlap percentage |
| `cuda.gds.bandwidth` | Gauge | GDS transfer bandwidth |
| `cuda.fallback.cpu_executions` | Counter | CPU fallback execution count |
| `cuda.errors.kernel` | Counter | CUDA kernel errors |
| `cuda.errors.memory` | Counter | CUDA memory errors |

### 11.2 Alert Thresholds

| Alert | Condition | Severity |
|-------|-----------|----------|
| GPUUtilizationLow | gpu_utilization < 60% for 5m | Warning |
| GPUMemoryHigh | memory_used > 22GB for 5m | Warning |
| GPUMemoryCritical | memory_used > 23GB for 1m | Critical |
| KernelLatencyHigh | any kernel P95 > 2x target for 5m | Warning |
| MemoryPoolFragmented | fragmentation > 30% for 10m | Warning |
| StreamUtilizationLow | stream utilization < 50% for 5m | Warning |
| CUDAErrors | errors > 0 in 1m | Critical |
| CPUFallbackActive | fallback executions > 10/min | Warning |

---

## 12. Configuration File Reference

```toml
# config/cuda.toml - Complete CUDA configuration

[cuda]
# Device selection
device_id = 0
compute_capability = [12, 0]

# Memory configuration
memory_pool_size = 21474836480  # 20GB
uvm_enabled = true
uvm_migration_threshold = 1073741824  # 1GB

# Stream configuration
stream_count = 8

# Green Contexts (CUDA 13.1)
green_contexts_enabled = true

# Precision
inference_precision = "fp8"  # fp32, fp16, bf16, fp8, fp4, mixed

# GDS configuration
gds_enabled = true
gds_buffer_size = 1073741824  # 1GB

# Fallback
cpu_fallback_enabled = true

[cuda.green_context_partitions]
embedding_sms = 42
memory_sms = 42
learning_sms = 42
utility_sms = 44

[cuda.memory_pool]
embedding_pool_size = 8589934592    # 8GB
hopfield_pool_size = 6442450944     # 6GB
scratch_pool_size = 4294967296      # 4GB
transfer_buffer_size = 1073741824   # 1GB
allocation_alignment = 256
trim_enabled = true
trim_threshold = 0.8

[cuda.streams]
embedding_stream_priority = 0  # Highest
hopfield_stream_priority = 0
neuromod_stream_priority = 1
utility_stream_priority = 2

[cuda.kernels]
# Block sizes
fused_embedding_block_size = 256
hopfield_attention_block_size = 128
neuromodulation_block_size = 256
cone_containment_block_size = 256

# Kernel parameters
hopfield_default_beta = 1.0
hopfield_max_patterns = 10000000

[cuda.gds]
nvme_devices = ["/dev/nvme0n1", "/dev/nvme1n1"]
max_concurrent_ops = 64

[cuda.profiling]
enabled = true
trace_kernels = false  # Enable for detailed profiling
metrics_interval_ms = 1000
```

---

## 13. Acceptance Criteria

### 13.1 Module Completion Checklist

- [ ] CUDA device detection with compute capability 12.0 verification
- [ ] Green Contexts initialization with configurable SM partitioning (4 x ~42 SMs)
- [ ] 8 concurrent CUDA streams operational
- [ ] fused_embedding_kernel: <10ms latency, >1000/sec throughput
- [ ] hopfield_attention_kernel: <1ms latency for 100K patterns
- [ ] neuromodulation_kernel: <200us latency
- [ ] cone_containment_kernel: <1ms for 10K checks
- [ ] Memory pool with <1us cached allocation
- [ ] UVM for graphs >VRAM with automatic migration
- [ ] Pinned memory for CPU-GPU transfers
- [ ] <24GB VRAM under load
- [ ] >80% GPU utilization
- [ ] 5x speedup over CPU baseline
- [ ] FP8/FP4 precision support
- [ ] GPU Direct Storage integration
- [ ] Graceful CPU fallback
- [ ] All REQ-CUDA-001 through REQ-CUDA-050 verified
- [ ] >90% unit test coverage
- [ ] >80% integration test coverage

### 13.2 Quality Gates

| Gate | Criteria |
|------|----------|
| Code Review | All code reviewed by CUDA Engineer |
| Unit Tests | >90% coverage, all passing |
| Integration Tests | >80% coverage, all passing |
| Performance Tests | All latency/throughput targets met |
| Memory Tests | <24GB VRAM under stress |
| Profiling | >80% GPU utilization verified |
| Security Review | No buffer overflows or memory leaks |
| Documentation | All APIs documented |

---

## 14. Glossary

| Term | Definition |
|------|------------|
| Green Contexts | CUDA 13.1 feature for SM partitioning and power management |
| FP8 | 8-bit floating point (E4M3 or E5M2 format) |
| FP4 | 4-bit floating point for inference |
| UVM | Unified Virtual Memory - automatic CPU/GPU memory migration |
| GDS | GPU Direct Storage - direct NVMe to GPU data path |
| WMMA | Warp Matrix Multiply-Accumulate - Tensor Core operations |
| Poincare Ball | Model of hyperbolic space for entailment cones |
| SM | Streaming Multiprocessor - GPU compute unit |
| Tensor Core | Specialized matrix multiply hardware |
| cuFile | CUDA library for GPU Direct Storage |

---

## 15. References

- [CUDA 13.1 Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [RTX 5090 Architecture Whitepaper](https://www.nvidia.com/blackwell-architecture)
- [CUDA Green Contexts](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)
- [FP8 Training and Inference](https://arxiv.org/abs/2209.05433)
- [GPU Direct Storage Guide](https://docs.nvidia.com/gpudirect-storage/)
- [Modern Hopfield Networks](https://arxiv.org/abs/2008.02217)
- [Hyperbolic Embeddings](https://arxiv.org/abs/1705.08039)
- [constitution.yaml](../../docs2/constitution.yaml) - Tech Stack section
- [implementationplan.md](../../docs2/implementationplan.md) - Module 7

---

*Document Version: 1.0.0*
*Generated: 2025-12-31*
*Specification Agent: #7/28*
