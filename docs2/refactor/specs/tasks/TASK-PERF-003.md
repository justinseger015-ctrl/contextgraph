# TASK-PERF-003: GPU Optimization

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-PERF-003 |
| **Title** | GPU Optimization |
| **Status** | :white_circle: todo |
| **Layer** | Performance |
| **Sequence** | 53 |
| **Estimated Days** | 3 |
| **Complexity** | High |

## Implements

- ARCH-08: GPU required for embedding inference
- Constitution performance budget for embeddings
- Optimal GPU utilization

## Dependencies

| Task | Reason |
|------|--------|
| TASK-CORE-011 | GPU memory management |
| TASK-CORE-012 | Model loading infrastructure |
| TASK-PERF-001 | Benchmarks for verification |

## Objective

Optimize GPU utilization for embedding generation:
1. Batch inference optimization
2. Mixed precision (FP16/BF16)
3. Tensor core utilization
4. Memory-compute overlap
5. Multi-GPU scaling

## Context

Embedding generation is the primary GPU workload. Optimization targets:
- Throughput: > 100 items/sec per GPU
- Latency: < 50ms for single embedding
- Utilization: > 80% GPU compute

## Scope

### In Scope

- Batch size optimization
- Mixed precision inference
- CUDA stream management
- Memory pinning for transfers
- Multi-GPU load balancing

### Out of Scope

- Model training
- Custom CUDA kernels
- Distributed multi-node

## Definition of Done

### Batch Optimization

```rust
// crates/context-graph-core/src/gpu/batch.rs

use std::collections::VecDeque;
use tokio::sync::{mpsc, oneshot};

/// Batching configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum wait time for batch accumulation
    pub max_wait: std::time::Duration,
    /// Minimum batch size (won't wait if reached)
    pub min_batch_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_wait: std::time::Duration::from_millis(10),
            min_batch_size: 8,
        }
    }
}

/// Dynamic batch accumulator
pub struct BatchAccumulator<T> {
    config: BatchConfig,
    pending: VecDeque<(T, oneshot::Sender<BatchResult>)>,
    last_flush: std::time::Instant,
}

impl<T> BatchAccumulator<T> {
    pub fn new(config: BatchConfig) -> Self;

    /// Add item to batch, returns true if batch is ready
    pub fn add(&mut self, item: T, response: oneshot::Sender<BatchResult>) -> bool {
        self.pending.push_back((item, response));

        // Flush if max size reached
        if self.pending.len() >= self.config.max_batch_size {
            return true;
        }

        // Flush if min size reached and waited long enough
        if self.pending.len() >= self.config.min_batch_size
            && self.last_flush.elapsed() >= self.config.max_wait
        {
            return true;
        }

        false
    }

    /// Get pending batch for processing
    pub fn flush(&mut self) -> Vec<(T, oneshot::Sender<BatchResult>)> {
        self.last_flush = std::time::Instant::now();
        self.pending.drain(..).collect()
    }

    /// Check if should flush based on timeout
    pub fn should_flush(&self) -> bool {
        !self.pending.is_empty()
            && self.last_flush.elapsed() >= self.config.max_wait
    }
}

/// Optimal batch size finder
pub struct BatchSizeOptimizer {
    model_memory: usize,
    gpu_memory: usize,
    sequence_length: usize,
}

impl BatchSizeOptimizer {
    /// Calculate optimal batch size for given GPU memory
    pub fn optimal_batch_size(&self) -> usize {
        // Estimate memory per sample
        let per_sample = self.estimate_per_sample_memory();

        // Reserve 20% for workspace
        let available = (self.gpu_memory as f64 * 0.8) as usize;

        // Subtract model memory
        let for_batches = available.saturating_sub(self.model_memory);

        // Calculate batch size
        let batch = for_batches / per_sample;

        // Clamp to reasonable range
        batch.clamp(1, 256)
    }

    fn estimate_per_sample_memory(&self) -> usize {
        // Activations + gradients (inference only = activations)
        // Rough estimate: 4 bytes * sequence_length * hidden_dim
        self.sequence_length * 768 * 4
    }
}
```

### Mixed Precision Inference

```rust
// crates/context-graph-core/src/gpu/precision.rs

use half::{bf16, f16};

/// Precision modes for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferencePrecision {
    /// Full 32-bit float
    FP32,
    /// 16-bit float (better on consumer GPUs)
    FP16,
    /// Brain float 16 (better on A100/H100)
    BF16,
    /// 8-bit quantized
    INT8,
}

impl InferencePrecision {
    /// Detect optimal precision for current GPU
    pub fn detect_optimal() -> Self {
        #[cfg(feature = "cuda")]
        {
            let device = cudarc::driver::CudaDevice::new(0).unwrap();
            let props = device.get_device_properties();

            // A100/H100 have good BF16 support
            if props.major >= 8 {
                return Self::BF16;
            }

            // Volta/Turing/Ampere have good FP16 support
            if props.major >= 7 {
                return Self::FP16;
            }

            Self::FP32
        }

        #[cfg(not(feature = "cuda"))]
        Self::FP32
    }

    /// Memory reduction factor vs FP32
    pub fn memory_factor(&self) -> f32 {
        match self {
            Self::FP32 => 1.0,
            Self::FP16 | Self::BF16 => 0.5,
            Self::INT8 => 0.25,
        }
    }

    /// Speedup factor (approximate)
    pub fn speedup_factor(&self) -> f32 {
        match self {
            Self::FP32 => 1.0,
            Self::FP16 | Self::BF16 => 2.0, // Tensor cores
            Self::INT8 => 4.0,
        }
    }
}

/// Convert tensor to target precision
pub trait PrecisionConvert {
    fn to_precision(&self, precision: InferencePrecision) -> Self;
}

impl PrecisionConvert for Vec<f32> {
    fn to_precision(&self, precision: InferencePrecision) -> Self {
        match precision {
            InferencePrecision::FP32 => self.clone(),
            InferencePrecision::FP16 => {
                // Note: Would use GPU conversion in real impl
                self.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
            }
            InferencePrecision::BF16 => {
                self.iter().map(|&x| bf16::from_f32(x).to_f32()).collect()
            }
            InferencePrecision::INT8 => {
                // Quantization would happen on GPU
                self.clone()
            }
        }
    }
}
```

### CUDA Stream Management

```rust
// crates/context-graph-core/src/gpu/streams.rs

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaStream};

/// Manages multiple CUDA streams for overlapping operations
pub struct StreamPool {
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    #[cfg(feature = "cuda")]
    compute_streams: Vec<Arc<CudaStream>>,
    #[cfg(feature = "cuda")]
    transfer_stream: Arc<CudaStream>,
    current_idx: std::sync::atomic::AtomicUsize,
}

impl StreamPool {
    /// Create pool with specified number of compute streams
    pub fn new(num_streams: usize) -> Self {
        #[cfg(feature = "cuda")]
        {
            let device = CudaDevice::new(0).unwrap();
            let compute_streams: Vec<_> = (0..num_streams)
                .map(|_| Arc::new(device.fork_default_stream().unwrap()))
                .collect();
            let transfer_stream = Arc::new(device.fork_default_stream().unwrap());

            Self {
                device: Arc::new(device),
                compute_streams,
                transfer_stream,
                current_idx: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        #[cfg(not(feature = "cuda"))]
        Self {
            current_idx: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Get next compute stream (round-robin)
    #[cfg(feature = "cuda")]
    pub fn next_compute_stream(&self) -> Arc<CudaStream> {
        let idx = self.current_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.compute_streams[idx % self.compute_streams.len()].clone()
    }

    /// Get transfer stream (for H2D/D2H copies)
    #[cfg(feature = "cuda")]
    pub fn transfer_stream(&self) -> Arc<CudaStream> {
        self.transfer_stream.clone()
    }

    /// Synchronize all streams
    #[cfg(feature = "cuda")]
    pub fn sync_all(&self) {
        for stream in &self.compute_streams {
            stream.synchronize().unwrap();
        }
        self.transfer_stream.synchronize().unwrap();
    }
}

/// Overlapped compute-transfer executor
pub struct OverlappedExecutor {
    pool: Arc<StreamPool>,
}

impl OverlappedExecutor {
    /// Execute with memory-compute overlap
    ///
    /// While batch N is computing, batch N+1 is being transferred
    #[cfg(feature = "cuda")]
    pub async fn execute_overlapped<T, F>(
        &self,
        batches: Vec<T>,
        compute_fn: F,
    ) -> Vec<Vec<f32>>
    where
        F: Fn(&T, &CudaStream) -> Vec<f32>,
    {
        let mut results = Vec::with_capacity(batches.len());

        // Pipeline: transfer[i+1] overlaps compute[i]
        for (i, batch) in batches.iter().enumerate() {
            let compute_stream = self.pool.next_compute_stream();

            // Start transfer of next batch (if any)
            if i + 1 < batches.len() {
                let _transfer_stream = self.pool.transfer_stream();
                // Transfer next batch async
            }

            // Compute current batch
            let result = compute_fn(batch, &compute_stream);
            results.push(result);
        }

        results
    }
}
```

### Multi-GPU Scaling

```rust
// crates/context-graph-core/src/gpu/multi_gpu.rs

/// Multi-GPU load balancer
pub struct MultiGpuBalancer {
    #[cfg(feature = "cuda")]
    devices: Vec<Arc<CudaDevice>>,
    loads: Vec<std::sync::atomic::AtomicU64>,
}

impl MultiGpuBalancer {
    /// Initialize with all available GPUs
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        {
            let count = cudarc::driver::CudaDevice::count().unwrap_or(1) as usize;
            let devices: Vec<_> = (0..count)
                .filter_map(|i| CudaDevice::new(i).ok())
                .map(Arc::new)
                .collect();
            let loads = (0..devices.len())
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect();

            Self { devices, loads }
        }

        #[cfg(not(feature = "cuda"))]
        Self {
            loads: vec![std::sync::atomic::AtomicU64::new(0)],
        }
    }

    /// Get GPU with lowest current load
    #[cfg(feature = "cuda")]
    pub fn least_loaded_gpu(&self) -> (usize, Arc<CudaDevice>) {
        let idx = self.loads
            .iter()
            .enumerate()
            .min_by_key(|(_, load)| load.load(std::sync::atomic::Ordering::Relaxed))
            .map(|(i, _)| i)
            .unwrap_or(0);

        (idx, self.devices[idx].clone())
    }

    /// Record work submission to GPU
    pub fn record_submit(&self, gpu_idx: usize, batch_size: usize) {
        self.loads[gpu_idx].fetch_add(batch_size as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record work completion from GPU
    pub fn record_complete(&self, gpu_idx: usize, batch_size: usize) {
        self.loads[gpu_idx].fetch_sub(batch_size as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get number of GPUs
    pub fn gpu_count(&self) -> usize {
        #[cfg(feature = "cuda")]
        { self.devices.len() }
        #[cfg(not(feature = "cuda"))]
        { 0 }
    }
}

/// Multi-GPU embedding service
pub struct MultiGpuEmbedder {
    balancer: Arc<MultiGpuBalancer>,
    models: Vec<Arc<dyn EmbeddingModel>>,
}

impl MultiGpuEmbedder {
    /// Generate embeddings with multi-GPU distribution
    pub async fn embed_batch(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
        let gpu_count = self.balancer.gpu_count();
        if gpu_count <= 1 {
            // Single GPU fallback
            return self.models[0].embed_batch(&texts).await;
        }

        // Split across GPUs
        let chunk_size = (texts.len() + gpu_count - 1) / gpu_count;
        let chunks: Vec<_> = texts.chunks(chunk_size).collect();

        let handles: Vec<_> = chunks
            .into_iter()
            .enumerate()
            .map(|(gpu_idx, chunk)| {
                let model = self.models[gpu_idx].clone();
                let chunk = chunk.to_vec();
                let balancer = self.balancer.clone();

                tokio::spawn(async move {
                    balancer.record_submit(gpu_idx, chunk.len());
                    let result = model.embed_batch(&chunk).await;
                    balancer.record_complete(gpu_idx, chunk.len());
                    result
                })
            })
            .collect();

        // Collect results
        let mut results = Vec::new();
        for handle in handles {
            results.extend(handle.await.unwrap());
        }
        results
    }
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Throughput | > 100 items/sec/GPU |
| GPU utilization | > 80% |
| Memory efficiency | < 50% overhead |
| Multi-GPU scaling | > 80% linear |

## Verification

- [ ] Batch size optimization improves throughput
- [ ] Mixed precision reduces memory 50%
- [ ] Stream overlap improves latency
- [ ] Multi-GPU scales near-linearly
- [ ] Benchmarks show > 100 items/sec

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gpu/batch.rs` | Batch optimization |
| `crates/context-graph-core/src/gpu/precision.rs` | Mixed precision |
| `crates/context-graph-core/src/gpu/streams.rs` | Stream management |
| `crates/context-graph-core/src/gpu/multi_gpu.rs` | Multi-GPU scaling |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Hardware variance | High | Medium | Multiple code paths |
| Memory fragmentation | Medium | Medium | Pool management |
| Sync overhead | Low | Low | Careful stream use |

## Traceability

- Source: Constitution ARCH-08, performance_budgets
- Related: TASK-CORE-011 (GPU Memory)
