# Module 08: GPU Direct Storage (GDS) - Functional Specification

**Version**: 1.0.0
**Status**: Draft
**Author**: Agent #8/28
**Module**: 8 of 14
**Phase**: 7
**Duration**: 3 weeks
**Dependencies**: Module 7 (CUDA Optimization)
**Last Updated**: 2025-12-31

---

## 1. Executive Summary

The GPU Direct Storage (GDS) module enables direct data paths between NVMe storage and GPU memory, bypassing CPU bottlenecks for dramatically faster loading of embedding models, FAISS indices, and memory shards. This module integrates NVIDIA's cuFile API with the CUDA infrastructure established in Module 7 to achieve 4x faster model loading and direct GPU memory population.

### 1.1 Key Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Model Load Speedup | 4x vs standard I/O | Benchmark suite |
| FAISS Index Load | Direct to GPU memory | Integration test |
| I/O Bandwidth Reduction | >60% load time reduction | Benchmark suite |
| Memory Shard Streaming | Zero CPU copies | Profiling |
| Max Concurrent Operations | 64 parallel ops | Load test |
| GDS Buffer Size | 1GB default | Configuration |

### 1.2 Target Hardware Requirements

| Specification | Requirement |
|---------------|-------------|
| GPU | RTX 5090 with GDS support |
| Storage | NVMe SSD with GPUDirect RDMA |
| CUDA Version | 13.1+ |
| Driver Version | 535+ with GDS support |
| cuFile Version | 1.8+ |
| Filesystem | ext4/xfs with GDS compatibility |

---

## 2. Architecture Overview

### 2.1 GDS Module Structure

```
context-graph-cuda/
+-- src/
|   +-- gds/
|   |   +-- mod.rs                    # GDS module entry and exports
|   |   +-- config.rs                 # GDS configuration management
|   |   +-- driver.rs                 # cuFile driver initialization
|   |   +-- buffer_pool.rs            # GDS buffer pool management
|   |   +-- loader/
|   |   |   +-- mod.rs                # Loader module exports
|   |   |   +-- model_loader.rs       # Model weights direct loading
|   |   |   +-- faiss_loader.rs       # FAISS index direct loading
|   |   |   +-- shard_streamer.rs     # Memory shard streaming
|   |   +-- async_io/
|   |   |   +-- mod.rs                # Async I/O pipeline
|   |   |   +-- cufile_wrapper.rs     # cuFile API Rust bindings
|   |   |   +-- double_buffer.rs      # Double buffering implementation
|   |   |   +-- prefetch.rs           # Prefetch scheduling
|   |   +-- fallback/
|   |   |   +-- mod.rs                # Fallback mechanisms
|   |   |   +-- standard_io.rs        # Standard filesystem fallback
|   |   |   +-- detector.rs           # GDS support auto-detection
|   |   +-- metrics/
|   |       +-- mod.rs                # GDS metrics and monitoring
|   |       +-- bandwidth.rs          # Bandwidth tracking
+-- tests/
    +-- gds_integration_tests.rs      # GDS integration tests
    +-- gds_performance_tests.rs      # GDS performance benchmarks
    +-- fallback_tests.rs             # Fallback mechanism tests
```

### 2.2 Data Flow Architecture

```
+-----------------------------------------------------------------------------------+
|                              NVMe Storage Devices                                  |
|                     (/dev/nvme0n1, /dev/nvme1n1, ...)                             |
+-----------------------------------------------------------------------------------+
                                        |
                                        | GPUDirect RDMA
                                        v
+-----------------------------------------------------------------------------------+
|                              cuFile Driver Layer                                   |
|  - File handle management                                                         |
|  - Async I/O queue management                                                     |
|  - Buffer registration                                                            |
|  - Error handling and retry logic                                                 |
+-----------------------------------------------------------------------------------+
                                        |
            +---------------------------+---------------------------+
            |                           |                           |
            v                           v                           v
+---------------------+     +---------------------+     +---------------------+
|   Model Loader      |     |   FAISS Loader      |     |   Shard Streamer    |
| - Weight files      |     | - IVF index         |     | - Memory shards     |
| - Checkpoint data   |     | - PQ codebooks      |     | - Incremental load  |
| - Optimizer states  |     | - Metadata          |     | - Hot/cold tiers    |
+---------------------+     +---------------------+     +---------------------+
            |                           |                           |
            v                           v                           v
+-----------------------------------------------------------------------------------+
|                              GDS Buffer Pool                                       |
|  - Registered GPU buffers (1GB default)                                           |
|  - Double buffering for continuous streaming                                      |
|  - Buffer lifecycle management                                                    |
+-----------------------------------------------------------------------------------+
            |                           |                           |
            v                           v                           v
+-----------------------------------------------------------------------------------+
|                              GPU Memory (VRAM)                                     |
|  - Model weights resident in GPU memory                                           |
|  - FAISS index directly accessible                                                |
|  - Memory shards for inference                                                    |
+-----------------------------------------------------------------------------------+
```

### 2.3 GDS vs Traditional I/O Path

```
Traditional I/O Path (Bounce Buffer):
+----------+     +-----------+     +----------+     +-----------+
|  NVMe    | --> | Page      | --> | CPU      | --> | GPU       |
|  Storage |     | Cache     |     | Memory   |     | Memory    |
+----------+     +-----------+     +----------+     +-----------+
              Latency: ~10ms per 1GB, CPU utilization: High

GPU Direct Storage Path:
+----------+                                        +-----------+
|  NVMe    | -------------------------------------> | GPU       |
|  Storage |         Direct DMA Transfer            | Memory    |
+----------+                                        +-----------+
              Latency: ~2.5ms per 1GB, CPU utilization: Minimal
```

---

## 3. Configuration

### 3.1 GDS Configuration Structure

```rust
/// GPU Direct Storage configuration
pub struct GDSConfig {
    /// Enable GPU Direct Storage
    pub enabled: bool,
    /// List of NVMe device paths for GDS operations
    pub nvme_devices: Vec<PathBuf>,
    /// Buffer size for GDS operations (default: 1GB)
    pub buffer_size: usize,
    /// Maximum concurrent GDS operations (default: 64)
    pub max_concurrent_ops: u32,
    /// Enable double buffering for continuous streaming
    pub double_buffering_enabled: bool,
    /// Prefetch lookahead depth (number of chunks)
    pub prefetch_depth: u32,
    /// Minimum file size to use GDS (smaller files use standard I/O)
    pub min_file_size: usize,
    /// Enable compression during transfer (if supported)
    pub compression_enabled: bool,
    /// Retry count for failed operations
    pub retry_count: u32,
    /// Timeout for individual operations (milliseconds)
    pub operation_timeout_ms: u64,
}

impl Default for GDSConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            nvme_devices: vec![
                PathBuf::from("/dev/nvme0n1"),
            ],
            buffer_size: 1024 * 1024 * 1024, // 1GB
            max_concurrent_ops: 64,
            double_buffering_enabled: true,
            prefetch_depth: 4,
            min_file_size: 16 * 1024 * 1024, // 16MB minimum
            compression_enabled: false,
            retry_count: 3,
            operation_timeout_ms: 30000, // 30 seconds
        }
    }
}

/// GDS buffer pool configuration
pub struct GDSBufferPoolConfig {
    /// Total buffer pool size
    pub total_size: usize,
    /// Number of buffers in pool
    pub num_buffers: usize,
    /// Buffer alignment (typically 4KB for NVMe)
    pub alignment: usize,
    /// Enable buffer pinning for RDMA
    pub pin_buffers: bool,
}

impl Default for GDSBufferPoolConfig {
    fn default() -> Self {
        Self {
            total_size: 2 * 1024 * 1024 * 1024, // 2GB total
            num_buffers: 16,
            alignment: 4096, // 4KB alignment
            pin_buffers: true,
        }
    }
}

/// Prefetch scheduling configuration
pub struct PrefetchConfig {
    /// Enable automatic prefetch scheduling
    pub auto_prefetch: bool,
    /// Prefetch trigger threshold (percentage of buffer consumed)
    pub trigger_threshold: f32,
    /// Maximum prefetch queue depth
    pub max_queue_depth: usize,
    /// Priority for prefetch operations
    pub priority: PrefetchPriority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchPriority {
    /// Background prefetch (lowest priority)
    Background,
    /// Normal prefetch priority
    Normal,
    /// High priority prefetch
    High,
    /// Critical prefetch (highest priority)
    Critical,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            auto_prefetch: true,
            trigger_threshold: 0.75, // Prefetch when 75% consumed
            max_queue_depth: 8,
            priority: PrefetchPriority::Normal,
        }
    }
}
```

### 3.2 Configuration File

```toml
# config/gds.toml - GPU Direct Storage configuration

[gds]
# Enable GPU Direct Storage
enabled = true

# NVMe devices for GDS operations
nvme_devices = ["/dev/nvme0n1", "/dev/nvme1n1"]

# Buffer size per operation (bytes) - 1GB default
buffer_size = 1073741824

# Maximum concurrent GDS operations
max_concurrent_ops = 64

# Enable double buffering for streaming
double_buffering_enabled = true

# Prefetch lookahead depth
prefetch_depth = 4

# Minimum file size to use GDS (bytes) - 16MB
min_file_size = 16777216

# Enable compression during transfer
compression_enabled = false

# Retry count for failed operations
retry_count = 3

# Operation timeout in milliseconds
operation_timeout_ms = 30000

[gds.buffer_pool]
# Total buffer pool size (bytes) - 2GB
total_size = 2147483648

# Number of buffers in pool
num_buffers = 16

# Buffer alignment (bytes) - 4KB
alignment = 4096

# Pin buffers for RDMA
pin_buffers = true

[gds.prefetch]
# Enable automatic prefetch scheduling
auto_prefetch = true

# Prefetch trigger threshold (0.0-1.0)
trigger_threshold = 0.75

# Maximum prefetch queue depth
max_queue_depth = 8

# Prefetch priority: background, normal, high, critical
priority = "normal"

[gds.fallback]
# Enable automatic fallback to standard I/O
enabled = true

# Log level for fallback events: debug, info, warn, error
log_level = "warn"

# Fallback trigger conditions
triggers = ["gds_unavailable", "device_error", "timeout"]
```

---

## 4. Core Components

### 4.1 GDS Manager

```rust
/// GPU Direct Storage manager for direct NVMe-GPU transfers
pub struct GDSManager {
    /// GDS configuration
    config: GDSConfig,
    /// cuFile driver handle
    driver: CuFileDriver,
    /// Buffer pool for GDS operations
    buffer_pool: GDSBufferPool,
    /// Active operation tracker
    operations: Arc<RwLock<HashMap<OperationId, GDSOperation>>>,
    /// Metrics collector
    metrics: GDSMetrics,
    /// Fallback handler
    fallback: FallbackHandler,
}

/// cuFile driver wrapper
pub struct CuFileDriver {
    /// Driver handle
    handle: CuFileDriverHandle,
    /// Registered devices
    devices: Vec<CuFileDevice>,
    /// Driver status
    status: DriverStatus,
}

/// GDS operation tracking
pub struct GDSOperation {
    /// Unique operation ID
    pub id: OperationId,
    /// Operation type
    pub op_type: GDSOperationType,
    /// Source file path
    pub source: PathBuf,
    /// Target GPU buffer
    pub target: DeviceBuffer,
    /// Bytes transferred
    pub bytes_transferred: AtomicUsize,
    /// Total bytes to transfer
    pub total_bytes: usize,
    /// Operation status
    pub status: AtomicOperationStatus,
    /// Start time
    pub start_time: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GDSOperationType {
    ModelLoad,
    FAISSIndexLoad,
    ShardStream,
    Checkpoint,
    Custom,
}

impl GDSManager {
    /// Initialize GDS manager with configuration
    ///
    /// # Arguments
    /// * `config` - GDS configuration
    /// * `cuda_ctx` - CUDA context from Module 7
    ///
    /// # Returns
    /// * `Result<Self, GDSError>` - Initialized manager or error
    ///
    /// `Constraint: Initialization < 1s`
    pub fn new(config: GDSConfig, cuda_ctx: &CudaContext) -> Result<Self, GDSError>;

    /// Check if GDS is available and properly configured
    ///
    /// `Constraint: Detection < 100ms`
    pub fn is_available(&self) -> bool;

    /// Get GDS capabilities and limitations
    pub fn capabilities(&self) -> GDSCapabilities;

    /// Load model weights directly to GPU via GDS
    ///
    /// # Arguments
    /// * `path` - Path to model weights file
    /// * `device_buffer` - Pre-allocated GPU buffer
    /// * `stream` - CUDA stream for async operation
    ///
    /// # Returns
    /// * `Result<LoadStats, GDSError>` - Load statistics or error
    ///
    /// `Constraint: 4x faster than standard filesystem load`
    /// `Constraint: Zero CPU memory copies`
    pub async fn load_model_weights(
        &self,
        path: &Path,
        device_buffer: &mut DeviceBuffer,
        stream: &CudaStream,
    ) -> Result<LoadStats, GDSError>;

    /// Load FAISS index directly to GPU memory
    ///
    /// # Arguments
    /// * `path` - Path to FAISS index file
    /// * `stream` - CUDA stream for async operation
    ///
    /// # Returns
    /// * `Result<FAISSGPUIndex, GDSError>` - GPU-resident index or error
    ///
    /// `Constraint: Direct to GPU memory, no CPU staging`
    pub async fn load_faiss_index(
        &self,
        path: &Path,
        stream: &CudaStream,
    ) -> Result<FAISSGPUIndex, GDSError>;

    /// Stream memory shards from NVMe to GPU
    ///
    /// # Arguments
    /// * `shard_paths` - List of shard file paths
    /// * `stream` - CUDA stream for async operations
    ///
    /// # Returns
    /// * `Result<Vec<DeviceBuffer>, GDSError>` - GPU buffers or error
    ///
    /// `Constraint: Continuous streaming without stalls`
    pub async fn stream_shards(
        &self,
        shard_paths: &[PathBuf],
        stream: &CudaStream,
    ) -> Result<Vec<DeviceBuffer>, GDSError>;

    /// Get current GDS bandwidth statistics
    pub fn bandwidth_stats(&self) -> GDSBandwidthStats;

    /// Get active operation status
    pub fn operation_status(&self, op_id: OperationId) -> Option<OperationStatus>;

    /// Cancel an active operation
    pub async fn cancel_operation(&self, op_id: OperationId) -> Result<(), GDSError>;
}

/// GDS bandwidth statistics
pub struct GDSBandwidthStats {
    /// Read bandwidth (GB/s)
    pub read_bandwidth: f64,
    /// Write bandwidth (GB/s)
    pub write_bandwidth: f64,
    /// Peak read bandwidth achieved
    pub peak_read_bandwidth: f64,
    /// Operations completed successfully
    pub ops_completed: u64,
    /// Operations failed
    pub ops_failed: u64,
    /// Total bytes transferred
    pub bytes_transferred: usize,
    /// Average operation latency
    pub avg_latency: Duration,
    /// Current queue depth
    pub queue_depth: usize,
}

/// GDS capabilities report
pub struct GDSCapabilities {
    /// GDS driver version
    pub driver_version: String,
    /// cuFile version
    pub cufile_version: String,
    /// Supported features
    pub features: GDSFeatures,
    /// Maximum transfer size
    pub max_transfer_size: usize,
    /// Optimal alignment
    pub optimal_alignment: usize,
    /// Supported filesystems
    pub supported_filesystems: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GDSFeatures {
    /// Direct I/O support
    pub direct_io: bool,
    /// Async I/O support
    pub async_io: bool,
    /// Batch operations support
    pub batch_ops: bool,
    /// Compression support
    pub compression: bool,
    /// Multi-device support
    pub multi_device: bool,
}
```

### 4.2 cuFile API Integration

```rust
/// Low-level cuFile API wrapper
pub struct CuFileWrapper {
    /// cuFile handle
    handle: cuFileHandle_t,
    /// File descriptor
    fd: RawFd,
    /// Buffer registration status
    registered: bool,
}

/// cuFile operation parameters
pub struct CuFileParams {
    /// Offset in file
    pub file_offset: u64,
    /// Size to transfer
    pub size: usize,
    /// GPU buffer offset
    pub buf_offset: usize,
    /// CUDA stream for async operations
    pub stream: Option<cudaStream_t>,
}

impl CuFileWrapper {
    /// Open file for GDS operations
    ///
    /// # Arguments
    /// * `path` - File path
    /// * `flags` - Open flags (O_RDONLY, O_WRONLY, O_RDWR)
    ///
    /// # Returns
    /// * `Result<Self, CuFileError>` - Wrapper or error
    pub fn open(path: &Path, flags: i32) -> Result<Self, CuFileError>;

    /// Register GPU buffer for cuFile operations
    ///
    /// # Arguments
    /// * `buffer` - GPU device buffer
    ///
    /// # Returns
    /// * `Result<BufferRegistration, CuFileError>`
    pub fn register_buffer(&mut self, buffer: &DeviceBuffer) -> Result<BufferRegistration, CuFileError>;

    /// Deregister GPU buffer
    pub fn deregister_buffer(&mut self, registration: BufferRegistration) -> Result<(), CuFileError>;

    /// Synchronous read from file to GPU buffer
    ///
    /// # Arguments
    /// * `buffer` - Registered GPU buffer
    /// * `params` - Operation parameters
    ///
    /// # Returns
    /// * `Result<usize, CuFileError>` - Bytes read
    pub fn read_sync(
        &self,
        buffer: &DeviceBuffer,
        params: &CuFileParams,
    ) -> Result<usize, CuFileError>;

    /// Asynchronous read from file to GPU buffer
    ///
    /// # Arguments
    /// * `buffer` - Registered GPU buffer
    /// * `params` - Operation parameters with stream
    ///
    /// # Returns
    /// * `Result<CuFileAsyncOp, CuFileError>` - Async operation handle
    pub fn read_async(
        &self,
        buffer: &DeviceBuffer,
        params: &CuFileParams,
    ) -> Result<CuFileAsyncOp, CuFileError>;

    /// Wait for async operation completion
    pub async fn wait(&self, op: CuFileAsyncOp) -> Result<usize, CuFileError>;

    /// Close file handle
    pub fn close(self) -> Result<(), CuFileError>;
}

/// Async operation handle
pub struct CuFileAsyncOp {
    /// Operation ID
    id: u64,
    /// Associated stream
    stream: cudaStream_t,
    /// Completion event
    event: CudaEvent,
}

/// Buffer registration handle
pub struct BufferRegistration {
    /// Registration token
    token: cuFileBufRegistration,
    /// Buffer pointer
    ptr: *mut c_void,
    /// Buffer size
    size: usize,
}
```

### 4.3 Double Buffering

```rust
/// Double buffer manager for continuous streaming
pub struct DoubleBufferManager {
    /// Primary buffer
    buffer_a: RegisteredBuffer,
    /// Secondary buffer
    buffer_b: RegisteredBuffer,
    /// Current active buffer
    active: AtomicBool, // false = A, true = B
    /// Buffer size
    buffer_size: usize,
    /// Swap synchronization
    swap_event: CudaEvent,
}

/// Registered GPU buffer for GDS
pub struct RegisteredBuffer {
    /// GPU device buffer
    device_buffer: DeviceBuffer,
    /// cuFile registration
    registration: BufferRegistration,
    /// Buffer state
    state: AtomicBufferState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    /// Buffer is available for filling
    Empty,
    /// Buffer is being filled from storage
    Filling,
    /// Buffer is ready for consumption
    Ready,
    /// Buffer is being consumed by GPU
    InUse,
}

impl DoubleBufferManager {
    /// Create double buffer manager
    ///
    /// # Arguments
    /// * `buffer_size` - Size of each buffer
    /// * `gds_wrapper` - cuFile wrapper for registration
    ///
    /// # Returns
    /// * `Result<Self, GDSError>`
    pub fn new(buffer_size: usize, gds_wrapper: &CuFileWrapper) -> Result<Self, GDSError>;

    /// Get buffer for filling (producer side)
    ///
    /// Blocks if no buffer is available
    pub async fn acquire_for_fill(&self) -> &RegisteredBuffer;

    /// Mark buffer as ready for consumption
    pub fn mark_ready(&self, buffer: &RegisteredBuffer);

    /// Get buffer for consumption (consumer side)
    ///
    /// Blocks if no ready buffer is available
    pub async fn acquire_for_consume(&self) -> &RegisteredBuffer;

    /// Mark buffer as empty (available for filling)
    pub fn mark_empty(&self, buffer: &RegisteredBuffer);

    /// Swap buffers atomically
    pub fn swap(&self);

    /// Get current buffer states
    pub fn states(&self) -> (BufferState, BufferState);
}

/// Continuous stream reader using double buffering
pub struct ContinuousStreamReader {
    /// Double buffer manager
    buffers: DoubleBufferManager,
    /// cuFile wrapper
    cufile: Arc<CuFileWrapper>,
    /// Current file offset
    offset: AtomicU64,
    /// Total file size
    file_size: u64,
    /// Prefetch task handle
    prefetch_task: Option<JoinHandle<()>>,
}

impl ContinuousStreamReader {
    /// Create continuous stream reader
    ///
    /// # Arguments
    /// * `path` - File path to stream
    /// * `buffer_size` - Size of each double buffer
    ///
    /// # Returns
    /// * `Result<Self, GDSError>`
    pub fn new(path: &Path, buffer_size: usize) -> Result<Self, GDSError>;

    /// Start streaming with prefetch
    ///
    /// # Arguments
    /// * `stream` - CUDA stream for GPU operations
    ///
    /// `Constraint: No stalls during streaming`
    pub async fn start(&mut self, stream: &CudaStream) -> Result<(), GDSError>;

    /// Get next chunk of data
    ///
    /// # Returns
    /// * `Option<&DeviceBuffer>` - Next chunk or None if EOF
    pub async fn next_chunk(&mut self) -> Option<&DeviceBuffer>;

    /// Get streaming progress
    pub fn progress(&self) -> StreamProgress;

    /// Stop streaming and cleanup
    pub async fn stop(&mut self) -> Result<(), GDSError>;
}

pub struct StreamProgress {
    /// Bytes streamed so far
    pub bytes_streamed: u64,
    /// Total bytes to stream
    pub total_bytes: u64,
    /// Current bandwidth
    pub bandwidth: f64,
    /// Estimated time remaining
    pub eta: Duration,
}
```

### 4.4 Prefetch Scheduler

```rust
/// Prefetch scheduler for anticipatory data loading
pub struct PrefetchScheduler {
    /// Configuration
    config: PrefetchConfig,
    /// Prefetch queue
    queue: VecDeque<PrefetchRequest>,
    /// Active prefetch operations
    active: HashMap<OperationId, PrefetchOperation>,
    /// GDS manager reference
    gds: Arc<GDSManager>,
    /// Worker thread handle
    worker: Option<JoinHandle<()>>,
}

/// Prefetch request
pub struct PrefetchRequest {
    /// Request ID
    pub id: RequestId,
    /// File path to prefetch
    pub path: PathBuf,
    /// Offset in file
    pub offset: u64,
    /// Size to prefetch
    pub size: usize,
    /// Priority
    pub priority: PrefetchPriority,
    /// Deadline (optional)
    pub deadline: Option<Instant>,
}

/// Active prefetch operation
pub struct PrefetchOperation {
    /// Request that spawned this operation
    pub request: PrefetchRequest,
    /// GDS operation ID
    pub gds_op_id: OperationId,
    /// Target buffer
    pub buffer: DeviceBuffer,
    /// Start time
    pub start_time: Instant,
}

impl PrefetchScheduler {
    /// Create prefetch scheduler
    ///
    /// # Arguments
    /// * `config` - Prefetch configuration
    /// * `gds` - GDS manager reference
    ///
    /// # Returns
    /// * `Result<Self, GDSError>`
    pub fn new(config: PrefetchConfig, gds: Arc<GDSManager>) -> Result<Self, GDSError>;

    /// Start the prefetch scheduler
    pub fn start(&mut self) -> Result<(), GDSError>;

    /// Schedule a prefetch request
    ///
    /// # Arguments
    /// * `request` - Prefetch request
    ///
    /// # Returns
    /// * `Result<RequestId, GDSError>`
    pub fn schedule(&mut self, request: PrefetchRequest) -> Result<RequestId, GDSError>;

    /// Schedule multiple prefetch requests (batch)
    ///
    /// # Arguments
    /// * `requests` - List of prefetch requests
    ///
    /// # Returns
    /// * `Result<Vec<RequestId>, GDSError>`
    pub fn schedule_batch(&mut self, requests: Vec<PrefetchRequest>) -> Result<Vec<RequestId>, GDSError>;

    /// Get prefetch result (blocks if not ready)
    ///
    /// # Arguments
    /// * `request_id` - Request ID
    ///
    /// # Returns
    /// * `Result<DeviceBuffer, GDSError>` - Prefetched buffer
    pub async fn get_result(&mut self, request_id: RequestId) -> Result<DeviceBuffer, GDSError>;

    /// Try to get prefetch result (non-blocking)
    ///
    /// # Arguments
    /// * `request_id` - Request ID
    ///
    /// # Returns
    /// * `Option<Result<DeviceBuffer, GDSError>>` - Result if ready
    pub fn try_get_result(&mut self, request_id: RequestId) -> Option<Result<DeviceBuffer, GDSError>>;

    /// Cancel a pending prefetch request
    pub fn cancel(&mut self, request_id: RequestId) -> Result<(), GDSError>;

    /// Get scheduler statistics
    pub fn stats(&self) -> PrefetchStats;

    /// Stop the prefetch scheduler
    pub fn stop(&mut self) -> Result<(), GDSError>;
}

pub struct PrefetchStats {
    /// Total requests scheduled
    pub total_scheduled: u64,
    /// Requests completed
    pub completed: u64,
    /// Requests cancelled
    pub cancelled: u64,
    /// Cache hit rate (prefetch ready when requested)
    pub hit_rate: f32,
    /// Average prefetch latency
    pub avg_latency: Duration,
    /// Current queue depth
    pub queue_depth: usize,
}
```

---

## 5. Direct Load Operations

### 5.1 Model Weights Loading

```rust
/// Model weights loader via GDS
pub struct ModelLoader {
    /// GDS manager
    gds: Arc<GDSManager>,
    /// Double buffer manager
    buffers: DoubleBufferManager,
    /// Loading configuration
    config: ModelLoaderConfig,
}

pub struct ModelLoaderConfig {
    /// Chunk size for streaming large models
    pub chunk_size: usize,
    /// Enable verification of loaded data
    pub verify_checksums: bool,
    /// Memory alignment for weights
    pub alignment: usize,
    /// Data type conversion on load
    pub dtype_conversion: Option<DTypeConversion>,
}

#[derive(Debug, Clone, Copy)]
pub enum DTypeConversion {
    /// FP32 to FP16
    F32ToF16,
    /// FP32 to BF16
    F32ToBF16,
    /// FP32 to FP8
    F32ToFP8,
    /// No conversion
    None,
}

impl ModelLoader {
    /// Create model loader
    ///
    /// # Arguments
    /// * `gds` - GDS manager
    /// * `config` - Loader configuration
    ///
    /// # Returns
    /// * `Result<Self, GDSError>`
    pub fn new(gds: Arc<GDSManager>, config: ModelLoaderConfig) -> Result<Self, GDSError>;

    /// Load model weights from file
    ///
    /// # Arguments
    /// * `path` - Path to weights file
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    /// * `Result<ModelWeights, GDSError>`
    ///
    /// `Constraint: 4x faster than standard I/O`
    pub async fn load(&self, path: &Path, stream: &CudaStream) -> Result<ModelWeights, GDSError>;

    /// Load model weights to pre-allocated buffer
    ///
    /// # Arguments
    /// * `path` - Path to weights file
    /// * `buffer` - Pre-allocated GPU buffer
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    /// * `Result<LoadStats, GDSError>`
    pub async fn load_into(
        &self,
        path: &Path,
        buffer: &mut DeviceBuffer,
        stream: &CudaStream,
    ) -> Result<LoadStats, GDSError>;

    /// Load sharded model weights
    ///
    /// # Arguments
    /// * `shard_paths` - Paths to weight shards
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    /// * `Result<Vec<ModelWeights>, GDSError>`
    pub async fn load_sharded(
        &self,
        shard_paths: &[PathBuf],
        stream: &CudaStream,
    ) -> Result<Vec<ModelWeights>, GDSError>;

    /// Get loading progress
    pub fn progress(&self) -> LoadProgress;
}

/// Model weights container
pub struct ModelWeights {
    /// GPU buffer containing weights
    pub buffer: DeviceBuffer,
    /// Weight tensor metadata
    pub tensors: Vec<TensorMetadata>,
    /// Data type
    pub dtype: DataType,
    /// Total size in bytes
    pub size: usize,
    /// Checksum (if verified)
    pub checksum: Option<u64>,
}

pub struct TensorMetadata {
    /// Tensor name
    pub name: String,
    /// Offset in buffer
    pub offset: usize,
    /// Shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DataType,
}

pub struct LoadStats {
    /// Total bytes loaded
    pub bytes_loaded: usize,
    /// Load duration
    pub duration: Duration,
    /// Effective bandwidth (GB/s)
    pub bandwidth: f64,
    /// Speedup vs baseline
    pub speedup: f64,
}

pub struct LoadProgress {
    /// Bytes loaded so far
    pub bytes_loaded: usize,
    /// Total bytes to load
    pub total_bytes: usize,
    /// Current bandwidth
    pub bandwidth: f64,
    /// Percentage complete
    pub percentage: f32,
    /// Estimated time remaining
    pub eta: Duration,
}
```

### 5.2 FAISS Index Loading

```rust
/// FAISS index loader via GDS
pub struct FAISSLoader {
    /// GDS manager
    gds: Arc<GDSManager>,
    /// FAISS configuration
    config: FAISSLoaderConfig,
}

pub struct FAISSLoaderConfig {
    /// Load directly to GPU memory
    pub gpu_direct: bool,
    /// GPU device ID for multi-GPU
    pub gpu_device: u32,
    /// Precompute search structures
    pub precompute_search: bool,
}

impl FAISSLoader {
    /// Create FAISS loader
    ///
    /// # Arguments
    /// * `gds` - GDS manager
    /// * `config` - Loader configuration
    ///
    /// # Returns
    /// * `Result<Self, GDSError>`
    pub fn new(gds: Arc<GDSManager>, config: FAISSLoaderConfig) -> Result<Self, GDSError>;

    /// Load FAISS index directly to GPU
    ///
    /// # Arguments
    /// * `path` - Path to FAISS index file
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    /// * `Result<FAISSGPUIndex, GDSError>`
    ///
    /// `Constraint: Direct to GPU memory, no CPU staging`
    pub async fn load(&self, path: &Path, stream: &CudaStream) -> Result<FAISSGPUIndex, GDSError>;

    /// Load FAISS index with IVF structure
    ///
    /// # Arguments
    /// * `index_path` - Path to main index
    /// * `ivf_path` - Path to IVF lists
    /// * `pq_path` - Path to PQ codebooks
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    /// * `Result<FAISSGPUIndex, GDSError>`
    pub async fn load_ivf_pq(
        &self,
        index_path: &Path,
        ivf_path: &Path,
        pq_path: &Path,
        stream: &CudaStream,
    ) -> Result<FAISSGPUIndex, GDSError>;

    /// Warm up index after loading (precompute)
    pub async fn warmup(&self, index: &FAISSGPUIndex, stream: &CudaStream) -> Result<(), GDSError>;
}

/// GPU-resident FAISS index
pub struct FAISSGPUIndex {
    /// Index type
    pub index_type: FAISSIndexType,
    /// GPU buffers for index data
    pub buffers: FAISSGPUBuffers,
    /// Index metadata
    pub metadata: FAISSMetadata,
    /// Search configuration
    pub search_config: FAISSSearchConfig,
}

#[derive(Debug, Clone, Copy)]
pub enum FAISSIndexType {
    /// Flat index (brute force)
    Flat,
    /// IVF index
    IVF,
    /// IVF with PQ
    IVFPQ,
    /// IVF with scalar quantizer
    IVFSQ,
    /// Hierarchical Navigable Small World
    HNSW,
}

pub struct FAISSGPUBuffers {
    /// Vector data buffer
    pub vectors: DeviceBuffer,
    /// IVF cluster centroids (if applicable)
    pub centroids: Option<DeviceBuffer>,
    /// PQ codebooks (if applicable)
    pub codebooks: Option<DeviceBuffer>,
    /// Inverted lists (if applicable)
    pub invlists: Option<DeviceBuffer>,
}

pub struct FAISSMetadata {
    /// Number of vectors
    pub num_vectors: usize,
    /// Vector dimension
    pub dimension: usize,
    /// Number of clusters (for IVF)
    pub nlist: Option<u32>,
    /// PQ parameters
    pub pq_params: Option<PQParams>,
    /// Index trained flag
    pub is_trained: bool,
}

pub struct FAISSSearchConfig {
    /// Number of clusters to probe
    pub nprobe: u32,
    /// Top-k results
    pub k: u32,
    /// Use precomputed tables
    pub use_precomputed: bool,
}
```

### 5.3 Memory Shard Streaming

```rust
/// Memory shard streamer for incremental loading
pub struct ShardStreamer {
    /// GDS manager
    gds: Arc<GDSManager>,
    /// Continuous stream readers
    readers: HashMap<ShardId, ContinuousStreamReader>,
    /// Prefetch scheduler
    prefetch: PrefetchScheduler,
    /// Shard configuration
    config: ShardConfig,
}

pub struct ShardConfig {
    /// Shard size in bytes
    pub shard_size: usize,
    /// Number of shards to prefetch
    pub prefetch_count: usize,
    /// Hot/cold tier threshold
    pub hot_threshold: f32,
    /// Enable compression
    pub compression: bool,
}

/// Shard identifier
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ShardId(pub u64);

/// Memory shard data
pub struct MemoryShard {
    /// Shard ID
    pub id: ShardId,
    /// GPU buffer containing shard data
    pub buffer: DeviceBuffer,
    /// Shard metadata
    pub metadata: ShardMetadata,
    /// Access statistics
    pub stats: ShardStats,
}

pub struct ShardMetadata {
    /// Number of memory nodes in shard
    pub node_count: usize,
    /// Time range covered
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
    /// Salience range
    pub salience_range: (f32, f32),
    /// Compression ratio (if compressed)
    pub compression_ratio: Option<f32>,
}

pub struct ShardStats {
    /// Access count
    pub access_count: u64,
    /// Last access time
    pub last_access: Instant,
    /// Is hot (frequently accessed)
    pub is_hot: bool,
}

impl ShardStreamer {
    /// Create shard streamer
    ///
    /// # Arguments
    /// * `gds` - GDS manager
    /// * `config` - Shard configuration
    ///
    /// # Returns
    /// * `Result<Self, GDSError>`
    pub fn new(gds: Arc<GDSManager>, config: ShardConfig) -> Result<Self, GDSError>;

    /// Stream shards from storage
    ///
    /// # Arguments
    /// * `shard_paths` - Ordered list of shard file paths
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    /// * `Result<ShardStream, GDSError>`
    ///
    /// `Constraint: Continuous streaming without CPU copies`
    pub async fn stream(
        &mut self,
        shard_paths: &[PathBuf],
        stream: &CudaStream,
    ) -> Result<ShardStream, GDSError>;

    /// Get specific shard by ID
    ///
    /// # Arguments
    /// * `shard_id` - Shard identifier
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    /// * `Result<MemoryShard, GDSError>`
    pub async fn get_shard(
        &mut self,
        shard_id: ShardId,
        stream: &CudaStream,
    ) -> Result<MemoryShard, GDSError>;

    /// Prefetch upcoming shards
    ///
    /// # Arguments
    /// * `shard_ids` - Shards to prefetch
    ///
    /// # Returns
    /// * `Result<(), GDSError>`
    pub fn prefetch_shards(&mut self, shard_ids: &[ShardId]) -> Result<(), GDSError>;

    /// Evict cold shards from GPU memory
    pub fn evict_cold_shards(&mut self) -> Result<usize, GDSError>;

    /// Get streaming statistics
    pub fn stats(&self) -> ShardStreamStats;
}

/// Shard stream iterator
pub struct ShardStream {
    /// Stream handle
    handle: ShardStreamHandle,
    /// Current shard index
    current_index: usize,
    /// Total shard count
    total_shards: usize,
}

impl ShardStream {
    /// Get next shard in stream
    pub async fn next(&mut self) -> Option<Result<MemoryShard, GDSError>>;

    /// Skip to specific shard
    pub async fn seek(&mut self, shard_id: ShardId) -> Result<(), GDSError>;

    /// Get remaining shard count
    pub fn remaining(&self) -> usize;
}

pub struct ShardStreamStats {
    /// Total shards streamed
    pub shards_streamed: u64,
    /// Bytes streamed
    pub bytes_streamed: usize,
    /// Prefetch hit rate
    pub prefetch_hit_rate: f32,
    /// Average shard load time
    pub avg_load_time: Duration,
    /// Hot shard count
    pub hot_shard_count: usize,
}
```

---

## 6. Fallback Mechanisms

### 6.1 GDS Support Detection

```rust
/// GDS support detector
pub struct GDSDetector {
    /// Detection results cache
    cache: Option<DetectionResult>,
    /// Detection timeout
    timeout: Duration,
}

/// GDS detection result
pub struct DetectionResult {
    /// Is GDS available
    pub available: bool,
    /// Availability reason
    pub reason: AvailabilityReason,
    /// Detected capabilities
    pub capabilities: Option<GDSCapabilities>,
    /// Detected devices
    pub devices: Vec<NVMeDevice>,
    /// Detection timestamp
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub enum AvailabilityReason {
    /// GDS is fully available
    Available,
    /// cuFile driver not loaded
    DriverNotLoaded,
    /// Incompatible GPU
    IncompatibleGPU,
    /// No compatible NVMe devices
    NoNVMeDevices,
    /// Filesystem not supported
    UnsupportedFilesystem,
    /// Insufficient permissions
    InsufficientPermissions,
    /// Driver version too old
    DriverVersionMismatch,
    /// Other reason
    Other(String),
}

pub struct NVMeDevice {
    /// Device path
    pub path: PathBuf,
    /// Device model
    pub model: String,
    /// GDS compatible
    pub gds_compatible: bool,
    /// Maximum bandwidth
    pub max_bandwidth: f64,
}

impl GDSDetector {
    /// Create detector
    pub fn new() -> Self;

    /// Detect GDS support
    ///
    /// `Constraint: Detection < 100ms`
    pub fn detect(&mut self) -> DetectionResult;

    /// Check if GDS is available (cached)
    pub fn is_available(&self) -> bool;

    /// Force re-detection
    pub fn refresh(&mut self) -> DetectionResult;

    /// Get detailed diagnostics
    pub fn diagnostics(&self) -> GDSDiagnostics;
}

pub struct GDSDiagnostics {
    /// cuFile driver status
    pub driver_status: String,
    /// GPU compatibility check
    pub gpu_check: String,
    /// Filesystem compatibility check
    pub fs_check: String,
    /// Permission check
    pub permission_check: String,
    /// Recommended actions
    pub recommendations: Vec<String>,
}
```

### 6.2 Standard I/O Fallback

```rust
/// Standard filesystem fallback for when GDS is unavailable
pub struct StandardIOFallback {
    /// Configuration
    config: FallbackConfig,
    /// Thread pool for async I/O
    thread_pool: ThreadPool,
    /// Metrics
    metrics: FallbackMetrics,
}

pub struct FallbackConfig {
    /// Enable fallback
    pub enabled: bool,
    /// Buffer size for reads
    pub buffer_size: usize,
    /// Number of worker threads
    pub thread_count: usize,
    /// Log fallback events
    pub log_events: bool,
    /// Log level
    pub log_level: LogLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl StandardIOFallback {
    /// Create standard I/O fallback
    ///
    /// # Arguments
    /// * `config` - Fallback configuration
    ///
    /// # Returns
    /// * `Result<Self, FallbackError>`
    pub fn new(config: FallbackConfig) -> Result<Self, FallbackError>;

    /// Load model weights via standard I/O
    ///
    /// # Arguments
    /// * `path` - Path to weights file
    /// * `device_buffer` - Target GPU buffer
    /// * `stream` - CUDA stream for async transfer
    ///
    /// # Returns
    /// * `Result<LoadStats, FallbackError>`
    ///
    /// Note: This is slower than GDS but provides fallback functionality
    pub async fn load_model_weights(
        &self,
        path: &Path,
        device_buffer: &mut DeviceBuffer,
        stream: &CudaStream,
    ) -> Result<LoadStats, FallbackError>;

    /// Load FAISS index via standard I/O
    ///
    /// # Arguments
    /// * `path` - Path to index file
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    /// * `Result<FAISSGPUIndex, FallbackError>`
    pub async fn load_faiss_index(
        &self,
        path: &Path,
        stream: &CudaStream,
    ) -> Result<FAISSGPUIndex, FallbackError>;

    /// Stream shards via standard I/O
    ///
    /// # Arguments
    /// * `shard_paths` - List of shard file paths
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    /// * `Result<Vec<DeviceBuffer>, FallbackError>`
    pub async fn stream_shards(
        &self,
        shard_paths: &[PathBuf],
        stream: &CudaStream,
    ) -> Result<Vec<DeviceBuffer>, FallbackError>;

    /// Get fallback metrics
    pub fn metrics(&self) -> FallbackMetrics;
}

pub struct FallbackMetrics {
    /// Number of fallback invocations
    pub invocation_count: u64,
    /// Total bytes transferred via fallback
    pub bytes_transferred: usize,
    /// Average transfer bandwidth
    pub avg_bandwidth: f64,
    /// Performance penalty vs GDS
    pub performance_penalty: f64,
}
```

### 6.3 Graceful Degradation

```rust
/// Graceful degradation handler
pub struct GracefulDegradation {
    /// GDS manager (primary)
    gds: Option<Arc<GDSManager>>,
    /// Standard I/O fallback (secondary)
    fallback: StandardIOFallback,
    /// Current mode
    mode: AtomicDegradationMode,
    /// Degradation policy
    policy: DegradationPolicy,
    /// Event log
    events: RwLock<Vec<DegradationEvent>>,
}

#[derive(Debug, Clone, Copy)]
pub enum DegradationMode {
    /// Full GDS operation
    FullGDS,
    /// GDS with occasional fallback
    PartialGDS,
    /// Full fallback mode
    FullFallback,
}

pub struct DegradationPolicy {
    /// Consecutive failures before degrading
    pub failure_threshold: u32,
    /// Time window for failure counting
    pub failure_window: Duration,
    /// Recovery check interval
    pub recovery_interval: Duration,
    /// Auto-recovery enabled
    pub auto_recovery: bool,
}

pub struct DegradationEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Event type
    pub event_type: DegradationEventType,
    /// Old mode
    pub old_mode: DegradationMode,
    /// New mode
    pub new_mode: DegradationMode,
    /// Reason
    pub reason: String,
}

#[derive(Debug, Clone)]
pub enum DegradationEventType {
    /// Degraded to lower mode
    Degraded,
    /// Recovered to higher mode
    Recovered,
    /// Failure detected
    FailureDetected,
    /// Manual mode change
    ManualChange,
}

impl GracefulDegradation {
    /// Create graceful degradation handler
    ///
    /// # Arguments
    /// * `gds` - GDS manager (if available)
    /// * `fallback` - Standard I/O fallback
    /// * `policy` - Degradation policy
    ///
    /// # Returns
    /// * `Self`
    pub fn new(
        gds: Option<Arc<GDSManager>>,
        fallback: StandardIOFallback,
        policy: DegradationPolicy,
    ) -> Self;

    /// Load with automatic backend selection
    ///
    /// # Arguments
    /// * `path` - Path to load
    /// * `load_type` - Type of load operation
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    /// * `Result<LoadResult, DegradationError>`
    ///
    /// `Constraint: Always returns a result via best available method`
    pub async fn load(
        &self,
        path: &Path,
        load_type: LoadType,
        stream: &CudaStream,
    ) -> Result<LoadResult, DegradationError>;

    /// Get current degradation mode
    pub fn current_mode(&self) -> DegradationMode;

    /// Force mode change
    pub fn set_mode(&self, mode: DegradationMode);

    /// Try to recover to higher mode
    pub async fn try_recover(&self) -> Result<DegradationMode, DegradationError>;

    /// Get degradation event history
    pub fn event_history(&self) -> Vec<DegradationEvent>;
}

#[derive(Debug, Clone, Copy)]
pub enum LoadType {
    ModelWeights,
    FAISSIndex,
    MemoryShard,
    Checkpoint,
}

pub enum LoadResult {
    /// Model weights
    ModelWeights(ModelWeights),
    /// FAISS index
    FAISSIndex(FAISSGPUIndex),
    /// Memory shard
    MemoryShard(MemoryShard),
    /// Checkpoint data
    Checkpoint(DeviceBuffer),
}
```

---

## 7. Requirements Specification

### 7.1 Functional Requirements

#### GDS Configuration

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-GDS-001 | System SHALL support GDSConfig with enabled flag | Critical | Unit test |
| REQ-GDS-002 | System SHALL support configurable NVMe device paths | Critical | Integration test |
| REQ-GDS-003 | System SHALL support buffer_size configuration (default: 1GB) | Critical | Unit test |
| REQ-GDS-004 | System SHALL support max_concurrent_ops configuration (default: 64) | High | Load test |
| REQ-GDS-005 | System SHALL persist configuration via TOML files | Medium | Integration test |

#### Direct Load Operations

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-GDS-006 | System SHALL load model weights 4x faster via GDS than standard I/O | Critical | Benchmark |
| REQ-GDS-007 | System SHALL load FAISS index directly to GPU memory | Critical | Integration test |
| REQ-GDS-008 | System SHALL stream memory shards without CPU memory copies | Critical | Profiling |
| REQ-GDS-009 | System SHALL support loading sharded model weights | High | Integration test |
| REQ-GDS-010 | System SHALL support loading IVF-PQ structured FAISS indices | High | Integration test |
| REQ-GDS-011 | System SHALL report loading progress and statistics | Medium | Unit test |

#### cuFile API Integration

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-GDS-012 | System SHALL initialize cuFile driver on startup | Critical | Integration test |
| REQ-GDS-013 | System SHALL register GPU buffers with cuFile | Critical | Unit test |
| REQ-GDS-014 | System SHALL support async read operations | Critical | Integration test |
| REQ-GDS-015 | System SHALL handle cuFile errors gracefully | Critical | Error injection test |
| REQ-GDS-016 | System SHALL properly cleanup cuFile resources on shutdown | High | Integration test |

#### Double Buffering

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-GDS-017 | System SHALL implement double buffering for continuous streaming | Critical | Integration test |
| REQ-GDS-018 | System SHALL swap buffers without stalling data flow | Critical | Benchmark |
| REQ-GDS-019 | System SHALL track buffer states (Empty/Filling/Ready/InUse) | High | Unit test |
| REQ-GDS-020 | System SHALL synchronize buffer swaps with CUDA streams | High | Integration test |

#### Prefetch Scheduling

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-GDS-021 | System SHALL implement prefetch scheduling for anticipatory loading | High | Integration test |
| REQ-GDS-022 | System SHALL support configurable prefetch depth | High | Unit test |
| REQ-GDS-023 | System SHALL prioritize prefetch requests | Medium | Unit test |
| REQ-GDS-024 | System SHALL cancel pending prefetch requests on demand | Medium | Integration test |
| REQ-GDS-025 | System SHALL track prefetch hit rate | Medium | Metrics test |

#### Performance Targets

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-GDS-026 | Model load time SHALL be reduced by >60% vs standard I/O | Critical | Benchmark |
| REQ-GDS-027 | FAISS index load time SHALL be reduced by >70% vs standard I/O | Critical | Benchmark |
| REQ-GDS-028 | Streaming bandwidth SHALL exceed 10 GB/s on supported hardware | High | Benchmark |
| REQ-GDS-029 | CPU utilization SHALL be <10% during GDS operations | High | Profiling |
| REQ-GDS-030 | GDS operations SHALL not cause GPU memory fragmentation | High | Memory test |

#### Fallback Mechanisms

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-GDS-031 | System SHALL auto-detect GDS support on initialization | Critical | Integration test |
| REQ-GDS-032 | System SHALL fall back to standard I/O when GDS unavailable | Critical | Fallback test |
| REQ-GDS-033 | System SHALL log fallback events with appropriate severity | High | Log test |
| REQ-GDS-034 | System SHALL maintain functionality without GDS hardware | Critical | Integration test |
| REQ-GDS-035 | Fallback SHALL provide identical results to GDS operations | Critical | Correctness test |
| REQ-GDS-036 | System SHALL implement graceful degradation policy | High | Integration test |
| REQ-GDS-037 | System SHALL attempt recovery from fallback mode | Medium | Recovery test |

#### Error Handling

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| REQ-GDS-038 | System SHALL retry failed GDS operations (configurable count) | High | Error injection test |
| REQ-GDS-039 | System SHALL handle file not found errors gracefully | Critical | Unit test |
| REQ-GDS-040 | System SHALL handle permission denied errors gracefully | Critical | Unit test |
| REQ-GDS-041 | System SHALL handle out of memory errors gracefully | Critical | Resource test |
| REQ-GDS-042 | System SHALL timeout operations exceeding configured limit | High | Timeout test |

### 7.2 Non-Functional Requirements

| ID | Requirement | Category | Target |
|----|-------------|----------|--------|
| REQ-GDS-043 | Model load speedup | Performance | 4x vs standard I/O |
| REQ-GDS-044 | Index load time reduction | Performance | >70% |
| REQ-GDS-045 | I/O bandwidth reduction | Performance | >60% load time reduction |
| REQ-GDS-046 | CPU utilization during I/O | Resource | <10% |
| REQ-GDS-047 | Buffer pool memory usage | Resource | Configurable (default 2GB) |
| REQ-GDS-048 | GDS detection latency | Performance | <100ms |
| REQ-GDS-049 | Unit test coverage | Quality | >90% |
| REQ-GDS-050 | Integration test coverage | Quality | >80% |
| REQ-GDS-051 | API documentation coverage | Quality | >80% |
| REQ-GDS-052 | Zero data corruption | Reliability | 0 errors under stress |
| REQ-GDS-053 | Fallback success rate | Reliability | 100% when GDS unavailable |

---

## 8. Test Cases

### 8.1 GDS Performance Tests

```rust
#[cfg(test)]
mod gds_performance_tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// TC-GDS-001: Model load speedup
    #[tokio::test]
    async fn test_model_load_speedup() {
        let gds_config = GDSConfig::default();
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let gds = GDSManager::new(gds_config, &cuda_ctx).unwrap();

        let model_path = Path::new("/data/models/embedding_model.bin");
        let model_size = std::fs::metadata(model_path).unwrap().len() as usize;
        let mut buffer = cuda_ctx.allocate_device(model_size).unwrap();
        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();

        // Measure GDS load time
        let start = Instant::now();
        let gds_stats = gds.load_model_weights(model_path, &mut buffer, &stream).await.unwrap();
        stream.synchronize().await.unwrap();
        let gds_time = start.elapsed();

        // Measure standard I/O load time
        let fallback = StandardIOFallback::new(FallbackConfig::default()).unwrap();
        let mut buffer2 = cuda_ctx.allocate_device(model_size).unwrap();

        let start = Instant::now();
        let _ = fallback.load_model_weights(model_path, &mut buffer2, &stream).await.unwrap();
        stream.synchronize().await.unwrap();
        let std_time = start.elapsed();

        let speedup = std_time.as_secs_f64() / gds_time.as_secs_f64();

        assert!(
            speedup >= 4.0,
            "GDS speedup {} below 4x target",
            speedup
        );
    }

    /// TC-GDS-002: FAISS index load time
    #[tokio::test]
    async fn test_faiss_load_time() {
        let gds_config = GDSConfig::default();
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let gds = GDSManager::new(gds_config, &cuda_ctx).unwrap();
        let faiss_loader = FAISSLoader::new(Arc::new(gds.clone()), FAISSLoaderConfig::default()).unwrap();

        let index_path = Path::new("/data/indices/faiss_ivfpq.index");
        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();

        // Measure GDS load time
        let start = Instant::now();
        let _index = faiss_loader.load(index_path, &stream).await.unwrap();
        stream.synchronize().await.unwrap();
        let gds_time = start.elapsed();

        // Measure standard load time (via FAISS native)
        let start = Instant::now();
        // ... standard FAISS load ...
        let std_time = start.elapsed();

        let reduction = 1.0 - (gds_time.as_secs_f64() / std_time.as_secs_f64());

        assert!(
            reduction >= 0.70,
            "FAISS load time reduction {} below 70% target",
            reduction
        );
    }

    /// TC-GDS-003: Zero CPU copies during streaming
    #[tokio::test]
    async fn test_zero_cpu_copies() {
        let gds_config = GDSConfig::default();
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let gds = Arc::new(GDSManager::new(gds_config, &cuda_ctx).unwrap());
        let mut streamer = ShardStreamer::new(gds, ShardConfig::default()).unwrap();

        let shard_paths: Vec<PathBuf> = (0..10)
            .map(|i| PathBuf::from(format!("/data/shards/shard_{}.bin", i)))
            .collect();
        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();

        // Record initial CPU memory usage
        let initial_cpu_mem = get_process_memory_usage();

        // Stream all shards
        let mut shard_stream = streamer.stream(&shard_paths, &stream).await.unwrap();
        while let Some(result) = shard_stream.next().await {
            let _ = result.unwrap();
        }

        // Check CPU memory usage didn't increase significantly
        let final_cpu_mem = get_process_memory_usage();
        let mem_increase = final_cpu_mem.saturating_sub(initial_cpu_mem);

        // Allow for some overhead but should be minimal
        let shard_size = 128 * 1024 * 1024; // 128MB shards
        assert!(
            mem_increase < shard_size / 10, // Less than 10% of one shard
            "CPU memory increased by {} during streaming (should be near zero)",
            mem_increase
        );
    }

    /// TC-GDS-004: I/O bandwidth measurement
    #[tokio::test]
    async fn test_io_bandwidth() {
        let gds_config = GDSConfig::default();
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let gds = GDSManager::new(gds_config, &cuda_ctx).unwrap();

        // Load large file to measure bandwidth
        let large_file = Path::new("/data/benchmark/large_file.bin"); // 10GB file
        let file_size = std::fs::metadata(large_file).unwrap().len() as usize;
        let mut buffer = cuda_ctx.allocate_device(file_size).unwrap();
        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();

        let start = Instant::now();
        let _ = gds.load_model_weights(large_file, &mut buffer, &stream).await.unwrap();
        stream.synchronize().await.unwrap();
        let elapsed = start.elapsed();

        let bandwidth_gbps = (file_size as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);

        assert!(
            bandwidth_gbps >= 10.0,
            "GDS bandwidth {} GB/s below 10 GB/s target",
            bandwidth_gbps
        );
    }

    /// TC-GDS-005: Concurrent operations stress test
    #[tokio::test]
    async fn test_concurrent_operations() {
        let gds_config = GDSConfig {
            max_concurrent_ops: 64,
            ..Default::default()
        };
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let gds = Arc::new(GDSManager::new(gds_config, &cuda_ctx).unwrap());

        // Launch 64 concurrent load operations
        let mut handles = Vec::new();
        for i in 0..64 {
            let gds_clone = gds.clone();
            let cuda_ctx_clone = cuda_ctx.clone();

            let handle = tokio::spawn(async move {
                let path = Path::new(&format!("/data/chunks/chunk_{}.bin", i));
                let mut buffer = cuda_ctx_clone.allocate_device(1024 * 1024).unwrap(); // 1MB
                let stream = cuda_ctx_clone.create_stream(StreamPriority::Normal).unwrap();
                gds_clone.load_model_weights(path, &mut buffer, &stream).await
            });
            handles.push(handle);
        }

        // Wait for all operations
        let mut success_count = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 64,
            "Only {} of 64 concurrent operations succeeded",
            success_count
        );
    }
}
```

### 8.2 Double Buffering Tests

```rust
#[cfg(test)]
mod double_buffer_tests {
    use super::*;

    /// TC-GDS-006: Double buffer continuous streaming
    #[tokio::test]
    async fn test_double_buffer_streaming() {
        let gds_config = GDSConfig::default();
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let gds = GDSManager::new(gds_config, &cuda_ctx).unwrap();

        let buffer_size = 128 * 1024 * 1024; // 128MB
        let cufile = gds.open_file(Path::new("/data/stream_test.bin")).unwrap();
        let db_manager = DoubleBufferManager::new(buffer_size, &cufile).unwrap();

        // Track stall events
        let mut stall_count = 0;
        let mut last_swap = Instant::now();

        for _ in 0..100 {
            // Simulate producer
            let fill_buf = db_manager.acquire_for_fill().await;
            // ... fill buffer ...
            db_manager.mark_ready(fill_buf);

            // Check for stalls
            let swap_time = last_swap.elapsed();
            if swap_time > Duration::from_millis(10) {
                stall_count += 1;
            }
            last_swap = Instant::now();

            // Simulate consumer
            let consume_buf = db_manager.acquire_for_consume().await;
            // ... consume buffer ...
            db_manager.mark_empty(consume_buf);
        }

        assert!(
            stall_count < 5,
            "Double buffering stalled {} times (should be < 5)",
            stall_count
        );
    }

    /// TC-GDS-007: Buffer state transitions
    #[test]
    fn test_buffer_state_transitions() {
        let gds_config = GDSConfig::default();
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let gds = GDSManager::new(gds_config, &cuda_ctx).unwrap();
        let cufile = gds.open_file(Path::new("/dev/null")).unwrap();
        let db_manager = DoubleBufferManager::new(1024, &cufile).unwrap();

        // Initial state: both empty
        let (state_a, state_b) = db_manager.states();
        assert_eq!(state_a, BufferState::Empty);
        assert_eq!(state_b, BufferState::Empty);

        // After marking ready
        let buf = tokio::runtime::Runtime::new().unwrap()
            .block_on(db_manager.acquire_for_fill());
        db_manager.mark_ready(buf);
        let (state_a, _) = db_manager.states();
        assert_eq!(state_a, BufferState::Ready);

        // After consumption
        let buf = tokio::runtime::Runtime::new().unwrap()
            .block_on(db_manager.acquire_for_consume());
        db_manager.mark_empty(buf);
        let (state_a, _) = db_manager.states();
        assert_eq!(state_a, BufferState::Empty);
    }
}
```

### 8.3 Prefetch Tests

```rust
#[cfg(test)]
mod prefetch_tests {
    use super::*;

    /// TC-GDS-008: Prefetch scheduling
    #[tokio::test]
    async fn test_prefetch_scheduling() {
        let gds_config = GDSConfig::default();
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let gds = Arc::new(GDSManager::new(gds_config, &cuda_ctx).unwrap());
        let mut scheduler = PrefetchScheduler::new(PrefetchConfig::default(), gds).unwrap();

        scheduler.start().unwrap();

        // Schedule prefetch requests
        let requests: Vec<_> = (0..10)
            .map(|i| PrefetchRequest {
                id: RequestId(i),
                path: PathBuf::from(format!("/data/chunks/chunk_{}.bin", i)),
                offset: 0,
                size: 1024 * 1024, // 1MB
                priority: PrefetchPriority::Normal,
                deadline: None,
            })
            .collect();

        let request_ids = scheduler.schedule_batch(requests).unwrap();

        // Wait for all prefetches
        for id in request_ids {
            let result = scheduler.get_result(id).await;
            assert!(result.is_ok(), "Prefetch {} failed", id.0);
        }

        let stats = scheduler.stats();
        assert!(
            stats.hit_rate > 0.9,
            "Prefetch hit rate {} below 90%",
            stats.hit_rate
        );

        scheduler.stop().unwrap();
    }

    /// TC-GDS-009: Prefetch cancellation
    #[tokio::test]
    async fn test_prefetch_cancellation() {
        let gds_config = GDSConfig::default();
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let gds = Arc::new(GDSManager::new(gds_config, &cuda_ctx).unwrap());
        let mut scheduler = PrefetchScheduler::new(PrefetchConfig::default(), gds).unwrap();

        scheduler.start().unwrap();

        // Schedule and immediately cancel
        let request = PrefetchRequest {
            id: RequestId(0),
            path: PathBuf::from("/data/large_file.bin"),
            offset: 0,
            size: 10 * 1024 * 1024 * 1024, // 10GB (takes time)
            priority: PrefetchPriority::Low,
            deadline: None,
        };

        let request_id = scheduler.schedule(request).unwrap();

        // Cancel before completion
        scheduler.cancel(request_id).unwrap();

        // Verify cancellation
        let stats = scheduler.stats();
        assert_eq!(stats.cancelled, 1);

        scheduler.stop().unwrap();
    }
}
```

### 8.4 Fallback Tests

```rust
#[cfg(test)]
mod fallback_tests {
    use super::*;

    /// TC-GDS-010: GDS detection
    #[test]
    fn test_gds_detection() {
        let mut detector = GDSDetector::new();
        let result = detector.detect();

        // Should complete within timeout
        assert!(result.timestamp.elapsed() < Duration::from_millis(100));

        // Should have a valid reason
        match result.reason {
            AvailabilityReason::Available => {
                assert!(result.available);
                assert!(result.capabilities.is_some());
            }
            _ => {
                assert!(!result.available);
            }
        }
    }

    /// TC-GDS-011: Automatic fallback
    #[tokio::test]
    async fn test_automatic_fallback() {
        // Create degradation handler with GDS disabled
        let fallback = StandardIOFallback::new(FallbackConfig::default()).unwrap();
        let degradation = GracefulDegradation::new(
            None, // No GDS
            fallback,
            DegradationPolicy::default(),
        );

        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();

        // Should automatically use fallback
        let result = degradation.load(
            Path::new("/data/models/test_model.bin"),
            LoadType::ModelWeights,
            &stream,
        ).await;

        assert!(result.is_ok());
        assert_eq!(degradation.current_mode(), DegradationMode::FullFallback);
    }

    /// TC-GDS-012: Fallback correctness
    #[tokio::test]
    async fn test_fallback_correctness() {
        let gds_config = GDSConfig::default();
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let gds = GDSManager::new(gds_config, &cuda_ctx).unwrap();
        let fallback = StandardIOFallback::new(FallbackConfig::default()).unwrap();

        let model_path = Path::new("/data/models/test_model.bin");
        let model_size = std::fs::metadata(model_path).unwrap().len() as usize;
        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();

        // Load via GDS
        let mut gds_buffer = cuda_ctx.allocate_device(model_size).unwrap();
        gds.load_model_weights(model_path, &mut gds_buffer, &stream).await.unwrap();
        stream.synchronize().await.unwrap();

        // Load via fallback
        let mut fallback_buffer = cuda_ctx.allocate_device(model_size).unwrap();
        fallback.load_model_weights(model_path, &mut fallback_buffer, &stream).await.unwrap();
        stream.synchronize().await.unwrap();

        // Compare results
        let gds_data = gds_buffer.download_to_host().await.unwrap();
        let fallback_data = fallback_buffer.download_to_host().await.unwrap();

        assert_eq!(
            gds_data, fallback_data,
            "GDS and fallback results differ"
        );
    }

    /// TC-GDS-013: Fallback logging
    #[tokio::test]
    async fn test_fallback_logging() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Custom logger that counts warnings
        let warning_count = Arc::new(AtomicUsize::new(0));
        let warning_count_clone = warning_count.clone();

        // Setup test logger
        // ...

        let fallback = StandardIOFallback::new(FallbackConfig {
            log_events: true,
            log_level: LogLevel::Warn,
            ..Default::default()
        }).unwrap();

        let degradation = GracefulDegradation::new(
            None, // No GDS
            fallback,
            DegradationPolicy::default(),
        );

        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();

        // Trigger fallback
        let _ = degradation.load(
            Path::new("/data/models/test_model.bin"),
            LoadType::ModelWeights,
            &stream,
        ).await;

        // Verify warning was logged
        assert!(
            warning_count.load(Ordering::SeqCst) > 0,
            "Fallback event was not logged"
        );
    }

    /// TC-GDS-014: Graceful degradation recovery
    #[tokio::test]
    async fn test_degradation_recovery() {
        let gds_config = GDSConfig::default();
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();

        // Start with GDS available
        let gds = Some(Arc::new(GDSManager::new(gds_config, &cuda_ctx).unwrap()));
        let fallback = StandardIOFallback::new(FallbackConfig::default()).unwrap();

        let degradation = GracefulDegradation::new(
            gds,
            fallback,
            DegradationPolicy {
                auto_recovery: true,
                recovery_interval: Duration::from_secs(1),
                ..Default::default()
            },
        );

        // Force degradation
        degradation.set_mode(DegradationMode::FullFallback);
        assert_eq!(degradation.current_mode(), DegradationMode::FullFallback);

        // Wait for recovery attempt
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Try recovery
        let new_mode = degradation.try_recover().await.unwrap();
        assert_eq!(new_mode, DegradationMode::FullGDS);
    }
}
```

### 8.5 Integration Tests

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;

    /// TC-GDS-015: End-to-end model loading
    #[tokio::test]
    async fn test_e2e_model_loading() {
        let gds_config = GDSConfig::default();
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let gds = Arc::new(GDSManager::new(gds_config, &cuda_ctx).unwrap());
        let loader = ModelLoader::new(
            gds,
            ModelLoaderConfig {
                verify_checksums: true,
                ..Default::default()
            },
        ).unwrap();

        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();

        // Load complete model
        let weights = loader.load(
            Path::new("/data/models/full_model.bin"),
            &stream,
        ).await.unwrap();

        // Verify loaded correctly
        assert!(weights.size > 0);
        assert!(!weights.tensors.is_empty());
        assert!(weights.checksum.is_some());

        stream.synchronize().await.unwrap();
    }

    /// TC-GDS-016: FAISS index search after GDS load
    #[tokio::test]
    async fn test_faiss_search_after_load() {
        let gds_config = GDSConfig::default();
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let gds = Arc::new(GDSManager::new(gds_config, &cuda_ctx).unwrap());
        let faiss_loader = FAISSLoader::new(gds, FAISSLoaderConfig::default()).unwrap();

        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();

        // Load index via GDS
        let index = faiss_loader.load(
            Path::new("/data/indices/test_index.faiss"),
            &stream,
        ).await.unwrap();

        // Perform search
        let query = vec![0.1f32; 1536]; // 1536D query vector
        let (distances, indices) = index.search(&query, 10, &stream).await.unwrap();

        // Verify search results
        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
        assert!(distances.iter().all(|&d| d >= 0.0));
    }

    /// TC-GDS-017: Memory shard round-trip
    #[tokio::test]
    async fn test_shard_round_trip() {
        let gds_config = GDSConfig::default();
        let cuda_ctx = CudaContext::new(CudaConfig::default()).unwrap();
        let gds = Arc::new(GDSManager::new(gds_config, &cuda_ctx).unwrap());
        let mut streamer = ShardStreamer::new(gds, ShardConfig::default()).unwrap();

        let stream = cuda_ctx.create_stream(StreamPriority::High).unwrap();
        let shard_paths: Vec<PathBuf> = (0..5)
            .map(|i| PathBuf::from(format!("/data/shards/shard_{}.bin", i)))
            .collect();

        // Stream and verify each shard
        let mut shard_stream = streamer.stream(&shard_paths, &stream).await.unwrap();
        let mut shard_count = 0;

        while let Some(result) = shard_stream.next().await {
            let shard = result.unwrap();
            assert!(shard.buffer.size() > 0);
            assert!(shard.metadata.node_count > 0);
            shard_count += 1;
        }

        assert_eq!(shard_count, 5);
    }
}
```

---

## 9. Metrics and Monitoring

### 9.1 Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `gds.available` | Gauge | GDS availability (1=available, 0=unavailable) |
| `gds.read_bandwidth_gbps` | Gauge | Current read bandwidth in GB/s |
| `gds.write_bandwidth_gbps` | Gauge | Current write bandwidth in GB/s |
| `gds.ops_completed` | Counter | Total GDS operations completed |
| `gds.ops_failed` | Counter | Total GDS operations failed |
| `gds.bytes_transferred` | Counter | Total bytes transferred via GDS |
| `gds.avg_latency_ms` | Histogram | Average operation latency |
| `gds.queue_depth` | Gauge | Current operation queue depth |
| `gds.buffer_pool_used` | Gauge | Buffer pool usage in bytes |
| `gds.buffer_pool_available` | Gauge | Buffer pool available in bytes |
| `gds.prefetch_hit_rate` | Gauge | Prefetch cache hit rate |
| `gds.prefetch_queue_depth` | Gauge | Prefetch queue depth |
| `gds.fallback_invocations` | Counter | Number of fallback invocations |
| `gds.degradation_mode` | Gauge | Current degradation mode |
| `gds.cpu_utilization` | Gauge | CPU utilization during I/O |
| `gds.model_load_speedup` | Gauge | Speedup vs standard I/O |

### 9.2 Alert Thresholds

| Alert | Condition | Severity |
|-------|-----------|----------|
| GDSUnavailable | gds.available == 0 for 1m | Warning |
| GDSBandwidthLow | read_bandwidth_gbps < 5 for 5m | Warning |
| GDSOperationsFailing | ops_failed > 10 in 1m | Warning |
| GDSOperationsFailingCritical | ops_failed > 50 in 1m | Critical |
| GDSBufferPoolExhausted | buffer_pool_available < 100MB for 1m | Warning |
| GDSPrefetchHitRateLow | prefetch_hit_rate < 0.5 for 5m | Warning |
| GDSFallbackActive | fallback_invocations > 0 in 1m | Warning |
| GDSDegraded | degradation_mode != FullGDS for 5m | Warning |
| GDSCPUUtilizationHigh | cpu_utilization > 0.2 for 5m | Warning |
| GDSSpeedupBelowTarget | model_load_speedup < 3 for 5m | Warning |

---

## 10. Configuration File Reference

```toml
# config/gds.toml - Complete GPU Direct Storage configuration

[gds]
# Enable GPU Direct Storage
enabled = true

# NVMe devices for GDS operations
nvme_devices = ["/dev/nvme0n1", "/dev/nvme1n1"]

# Buffer size per operation (bytes) - 1GB
buffer_size = 1073741824

# Maximum concurrent GDS operations
max_concurrent_ops = 64

# Enable double buffering for streaming
double_buffering_enabled = true

# Prefetch lookahead depth
prefetch_depth = 4

# Minimum file size to use GDS (bytes) - 16MB
min_file_size = 16777216

# Enable compression during transfer
compression_enabled = false

# Retry count for failed operations
retry_count = 3

# Operation timeout in milliseconds
operation_timeout_ms = 30000

[gds.buffer_pool]
# Total buffer pool size (bytes) - 2GB
total_size = 2147483648

# Number of buffers in pool
num_buffers = 16

# Buffer alignment (bytes) - 4KB
alignment = 4096

# Pin buffers for RDMA
pin_buffers = true

[gds.prefetch]
# Enable automatic prefetch scheduling
auto_prefetch = true

# Prefetch trigger threshold (0.0-1.0)
trigger_threshold = 0.75

# Maximum prefetch queue depth
max_queue_depth = 8

# Prefetch priority: background, normal, high, critical
priority = "normal"

[gds.fallback]
# Enable automatic fallback to standard I/O
enabled = true

# Log level for fallback events: debug, info, warn, error
log_level = "warn"

# Buffer size for standard I/O operations
buffer_size = 67108864  # 64MB

# Number of worker threads for async I/O
thread_count = 4

[gds.fallback.policy]
# Consecutive failures before degrading
failure_threshold = 3

# Time window for failure counting (seconds)
failure_window_secs = 60

# Recovery check interval (seconds)
recovery_interval_secs = 300

# Enable automatic recovery
auto_recovery = true

[gds.model_loader]
# Chunk size for streaming large models (bytes) - 256MB
chunk_size = 268435456

# Verify checksums after loading
verify_checksums = true

# Memory alignment for weights
alignment = 256

# Data type conversion: none, f32_to_f16, f32_to_bf16, f32_to_fp8
dtype_conversion = "none"

[gds.faiss_loader]
# Load directly to GPU memory
gpu_direct = true

# GPU device ID for multi-GPU
gpu_device = 0

# Precompute search structures after loading
precompute_search = true

[gds.shard_streamer]
# Shard size in bytes - 128MB
shard_size = 134217728

# Number of shards to prefetch
prefetch_count = 4

# Hot/cold tier threshold (access frequency)
hot_threshold = 0.8

# Enable compression for shards
compression = false
```

---

## 11. Acceptance Criteria

### 11.1 Module Completion Checklist

- [ ] GDS configuration with all required parameters (enabled, nvme_devices, buffer_size, max_concurrent_ops)
- [ ] cuFile driver initialization and cleanup
- [ ] Buffer registration with GPU memory
- [ ] Model weight loading 4x faster than standard I/O
- [ ] FAISS index loading directly to GPU memory
- [ ] Memory shard streaming without CPU copies
- [ ] Double buffering for continuous streaming
- [ ] Prefetch scheduling with configurable depth
- [ ] GDS support auto-detection
- [ ] Standard filesystem fallback
- [ ] Graceful degradation with recovery
- [ ] All REQ-GDS-001 through REQ-GDS-053 verified
- [ ] >90% unit test coverage
- [ ] >80% integration test coverage
- [ ] Documentation complete

### 11.2 Quality Gates

| Gate | Criteria |
|------|----------|
| Code Review | All code reviewed by Storage Engineer |
| Unit Tests | >90% coverage, all passing |
| Integration Tests | >80% coverage, all passing |
| Performance Tests | 4x speedup achieved |
| Fallback Tests | 100% functionality without GDS |
| Memory Tests | Zero data corruption |
| Security Review | No buffer overflows or leaks |
| Documentation | All APIs documented |

---

## 12. Glossary

| Term | Definition |
|------|------------|
| GDS | GPU Direct Storage - NVIDIA technology for direct NVMe-GPU data paths |
| cuFile | CUDA File I/O library for GDS operations |
| NVMe | Non-Volatile Memory Express - high-speed storage interface |
| RDMA | Remote Direct Memory Access - zero-copy data transfer |
| Double Buffering | Using two buffers alternately to overlap I/O and compute |
| Prefetch | Loading data before it is needed |
| Fallback | Alternative method when primary is unavailable |
| Graceful Degradation | Maintaining functionality with reduced performance |
| Shard | Partition of data for incremental loading |
| IVF | Inverted File Index - FAISS clustering structure |
| PQ | Product Quantization - vector compression technique |

---

## 13. References

- [NVIDIA GPU Direct Storage User Guide](https://docs.nvidia.com/gpudirect-storage/)
- [cuFile API Reference](https://docs.nvidia.com/cuda/cufile-api/)
- [NVIDIA GDS Best Practices](https://developer.nvidia.com/blog/gpudirect-storage/)
- [FAISS GPU Documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)
- [Module 7: CUDA Optimization](./module-07-cuda-optimization.md)
- [constitution.yaml](../../docs2/constitution.yaml) - Tech Stack section
- [implementationplan.md](../../docs2/implementationplan.md) - Module 8

---

*Document Version: 1.0.0*
*Generated: 2025-12-31*
*Specification Agent: #8/28*
