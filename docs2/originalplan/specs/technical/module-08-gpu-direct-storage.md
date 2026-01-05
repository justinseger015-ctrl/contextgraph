# Module 8: GPU Direct Storage Technical Specification

## Overview

GPU Direct Storage (GDS) enables direct memory access between NVMe storage and GPU memory, bypassing CPU and system memory bottlenecks. This module provides Rust FFI bindings to NVIDIA's cuFile API with double buffering, prefetch scheduling, and comprehensive error handling.

**Performance Targets:**
- Read throughput: 4 GB/s sustained
- Write throughput: 2 GB/s sustained
- Latency: < 100 microseconds for 4KB aligned reads

---

## 1. Core Configuration

### 1.1 GDSConfig Structure

```rust
use std::path::PathBuf;

/// GPU Direct Storage configuration
#[derive(Debug, Clone)]
pub struct GDSConfig {
    /// Primary buffer size in bytes (must be power of 2, min 64KB)
    pub buffer_size: usize,

    /// Memory alignment requirement (typically 4KB for NVMe)
    pub alignment: usize,

    /// Number of prefetch slots for read-ahead
    pub prefetch_slots: usize,

    /// Maximum outstanding I/O operations
    pub max_outstanding_io: usize,

    /// Enable double buffering for overlapped I/O
    pub double_buffering: bool,

    /// Prefetch distance in bytes (how far ahead to read)
    pub prefetch_distance: usize,

    /// GPU device ordinal for memory allocation
    pub gpu_device: i32,

    /// Enable batch submission mode
    pub batch_mode: bool,

    /// Maximum batch size for coalesced operations
    pub max_batch_size: usize,

    /// Polling interval for completion checks (microseconds)
    pub poll_interval_us: u64,

    /// Enable memory-mapped access fallback
    pub mmap_fallback: bool,
}

impl Default for GDSConfig {
    fn default() -> Self {
        Self {
            buffer_size: 16 * 1024 * 1024,      // 16 MB default
            alignment: 4096,                     // 4KB NVMe alignment
            prefetch_slots: 8,                   // 8 concurrent prefetch ops
            max_outstanding_io: 32,              // 32 outstanding requests
            double_buffering: true,              // Enable by default
            prefetch_distance: 64 * 1024 * 1024, // 64 MB lookahead
            gpu_device: 0,                       // Primary GPU
            batch_mode: true,                    // Batch for throughput
            max_batch_size: 16,                  // 16 ops per batch
            poll_interval_us: 10,                // 10 microsecond poll
            mmap_fallback: true,                 // Fallback enabled
        }
    }
}

impl GDSConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), GDSError> {
        // Buffer size must be power of 2 and >= 64KB
        if self.buffer_size < 65536 || !self.buffer_size.is_power_of_two() {
            return Err(GDSError::InvalidConfig(
                "buffer_size must be power of 2 and >= 64KB".into()
            ));
        }

        // Alignment must be power of 2
        if !self.alignment.is_power_of_two() {
            return Err(GDSError::InvalidConfig(
                "alignment must be power of 2".into()
            ));
        }

        // Prefetch distance must be multiple of buffer size
        if self.prefetch_distance % self.buffer_size != 0 {
            return Err(GDSError::InvalidConfig(
                "prefetch_distance must be multiple of buffer_size".into()
            ));
        }

        Ok(())
    }

    /// Calculate optimal buffer count for double buffering
    pub fn buffer_count(&self) -> usize {
        if self.double_buffering { 2 } else { 1 }
    }

    /// Calculate total GPU memory required
    pub fn total_gpu_memory(&self) -> usize {
        self.buffer_size * self.buffer_count() * self.prefetch_slots
    }
}
```

### 1.2 Buffer Alignment Utilities

```rust
/// Align value up to specified alignment
#[inline]
pub fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

/// Align value down to specified alignment
#[inline]
pub fn align_down(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    value & !(alignment - 1)
}

/// Check if value is aligned
#[inline]
pub fn is_aligned(value: usize, alignment: usize) -> bool {
    value & (alignment - 1) == 0
}

/// Aligned buffer descriptor for GDS operations
#[derive(Debug)]
pub struct AlignedBuffer {
    pub ptr: *mut u8,
    pub size: usize,
    pub alignment: usize,
    pub gpu_ptr: u64,  // CUdeviceptr
}
```

---

## 2. cuFile API FFI Bindings

### 2.1 Type Definitions

```rust
use std::os::raw::{c_int, c_void, c_char, c_uint};

// Opaque handle types
pub type CUfileHandle = *mut c_void;
pub type CUfileDescr = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUstream = *mut c_void;

/// cuFile driver status
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CUfileStatus {
    Success = 0,
    DriverNotInitialized = 1,
    InvalidValue = 2,
    NotSupported = 3,
    IOError = 4,
    InternalError = 5,
    InvalidFileHandle = 6,
    InvalidDevicePointer = 7,
    OutOfMemory = 8,
    PlatformNotSupported = 9,
    VersionMismatch = 10,
    BatchSubmitFailed = 11,
}

/// cuFile operation status for batch operations
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUfileOpError {
    pub status: CUfileStatus,
    pub errno: c_int,
}

/// File open flags
#[repr(C)]
pub struct CUfileOpenFlags(pub c_uint);

impl CUfileOpenFlags {
    pub const RDONLY: Self = Self(0x0001);
    pub const WRONLY: Self = Self(0x0002);
    pub const RDWR: Self = Self(0x0003);
    pub const CREATE: Self = Self(0x0100);
    pub const DIRECT: Self = Self(0x0200);
}

/// Batch operation descriptor
#[repr(C)]
pub struct CUfileBatchIOParams {
    pub mode: c_uint,           // Read (0) or Write (1)
    pub fh: CUfileHandle,       // File handle
    pub devPtr: CUdeviceptr,    // GPU memory pointer
    pub offset: i64,            // File offset
    pub size: usize,            // Transfer size
    pub bufferOffset: usize,    // Offset within GPU buffer
    pub reserved: [u64; 4],     // Reserved for future use
}

/// Driver properties
#[repr(C)]
pub struct CUfileDriverProps {
    pub major_version: c_uint,
    pub minor_version: c_uint,
    pub poll_mode: c_uint,
    pub poll_threshold: usize,
    pub max_direct_io_size: usize,
    pub max_batch_entries: usize,
    pub max_streams: usize,
}
```

### 2.2 FFI Function Declarations

```rust
#[link(name = "cufile")]
extern "C" {
    // Driver lifecycle
    pub fn cuFileDriverOpen() -> CUfileStatus;
    pub fn cuFileDriverClose() -> CUfileStatus;
    pub fn cuFileDriverGetProperties(props: *mut CUfileDriverProps) -> CUfileStatus;
    pub fn cuFileDriverSetPollMode(poll_mode: c_uint, poll_threshold: usize) -> CUfileStatus;

    // Buffer registration
    pub fn cuFileBufRegister(
        devPtr: CUdeviceptr,
        size: usize,
        flags: c_uint,
    ) -> CUfileStatus;

    pub fn cuFileBufDeregister(devPtr: CUdeviceptr) -> CUfileStatus;

    // File handle operations
    pub fn cuFileHandleRegister(
        fh: *mut CUfileHandle,
        descr: *const CUfileDescr,
    ) -> CUfileStatus;

    pub fn cuFileHandleDeregister(fh: CUfileHandle) -> CUfileStatus;

    // Synchronous I/O
    pub fn cuFileRead(
        fh: CUfileHandle,
        devPtr: CUdeviceptr,
        size: usize,
        fileOffset: i64,
        bufferOffset: usize,
    ) -> isize;

    pub fn cuFileWrite(
        fh: CUfileHandle,
        devPtr: CUdeviceptr,
        size: usize,
        fileOffset: i64,
        bufferOffset: usize,
    ) -> isize;

    // Batch operations
    pub fn cuFileBatchIOSetUp(
        batch_id: *mut c_uint,
        max_entries: c_uint,
    ) -> CUfileStatus;

    pub fn cuFileBatchIOSubmit(
        batch_id: c_uint,
        params: *const CUfileBatchIOParams,
        num_entries: c_uint,
        flags: c_uint,
    ) -> CUfileStatus;

    pub fn cuFileBatchIOGetStatus(
        batch_id: c_uint,
        completed: *mut c_uint,
        errors: *mut CUfileOpError,
        num_entries: c_uint,
    ) -> CUfileStatus;

    pub fn cuFileBatchIODestroy(batch_id: c_uint) -> CUfileStatus;

    // Stream-ordered operations (async)
    pub fn cuFileReadAsync(
        fh: CUfileHandle,
        devPtr: CUdeviceptr,
        size: *mut usize,
        fileOffset: i64,
        bufferOffset: usize,
        bytesRead: *mut isize,
        stream: CUstream,
    ) -> CUfileStatus;

    pub fn cuFileWriteAsync(
        fh: CUfileHandle,
        devPtr: CUdeviceptr,
        size: *mut usize,
        fileOffset: i64,
        bufferOffset: usize,
        bytesWritten: *mut isize,
        stream: CUstream,
    ) -> CUfileStatus;
}
```

### 2.3 Safe Rust Wrappers

```rust
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};

static DRIVER_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Initialize cuFile driver (call once per process)
pub fn init_driver() -> Result<(), GDSError> {
    if DRIVER_INITIALIZED.swap(true, Ordering::SeqCst) {
        return Ok(()); // Already initialized
    }

    let status = unsafe { cuFileDriverOpen() };
    if status != CUfileStatus::Success {
        DRIVER_INITIALIZED.store(false, Ordering::SeqCst);
        return Err(GDSError::DriverInit(status));
    }

    Ok(())
}

/// Shutdown cuFile driver
pub fn shutdown_driver() -> Result<(), GDSError> {
    if !DRIVER_INITIALIZED.swap(false, Ordering::SeqCst) {
        return Ok(()); // Not initialized
    }

    let status = unsafe { cuFileDriverClose() };
    if status != CUfileStatus::Success {
        return Err(GDSError::DriverShutdown(status));
    }

    Ok(())
}

/// Safe wrapper for cuFile handle
pub struct GDSFileHandle {
    handle: CUfileHandle,
    file: File,
    registered: bool,
}

impl GDSFileHandle {
    /// Open file for GDS operations
    pub fn open(path: &std::path::Path, flags: CUfileOpenFlags) -> Result<Self, GDSError> {
        use std::os::unix::fs::OpenOptionsExt;

        let file = std::fs::OpenOptions::new()
            .read((flags.0 & CUfileOpenFlags::RDONLY.0) != 0)
            .write((flags.0 & CUfileOpenFlags::WRONLY.0) != 0)
            .create((flags.0 & CUfileOpenFlags::CREATE.0) != 0)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .map_err(GDSError::FileOpen)?;

        let mut handle: CUfileHandle = std::ptr::null_mut();

        // Create file descriptor structure
        #[repr(C)]
        struct FileDescr {
            descr_type: c_uint,
            fd: c_int,
        }

        let descr = FileDescr {
            descr_type: 0, // CU_FILE_HANDLE_TYPE_OPAQUE_FD
            fd: file.as_raw_fd(),
        };

        let status = unsafe {
            cuFileHandleRegister(
                &mut handle,
                &descr as *const _ as *const CUfileDescr,
            )
        };

        if status != CUfileStatus::Success {
            return Err(GDSError::HandleRegister(status));
        }

        Ok(Self {
            handle,
            file,
            registered: true,
        })
    }

    /// Get raw handle for FFI calls
    pub fn raw(&self) -> CUfileHandle {
        self.handle
    }
}

impl Drop for GDSFileHandle {
    fn drop(&mut self) {
        if self.registered {
            unsafe { cuFileHandleDeregister(self.handle) };
        }
    }
}

/// Safe wrapper for registered GPU buffer
pub struct RegisteredBuffer {
    gpu_ptr: CUdeviceptr,
    size: usize,
    registered: bool,
}

impl RegisteredBuffer {
    /// Register GPU memory for cuFile operations
    pub fn new(gpu_ptr: CUdeviceptr, size: usize) -> Result<Self, GDSError> {
        let status = unsafe { cuFileBufRegister(gpu_ptr, size, 0) };

        if status != CUfileStatus::Success {
            return Err(GDSError::BufferRegister(status));
        }

        Ok(Self {
            gpu_ptr,
            size,
            registered: true,
        })
    }

    pub fn ptr(&self) -> CUdeviceptr {
        self.gpu_ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for RegisteredBuffer {
    fn drop(&mut self) {
        if self.registered {
            unsafe { cuFileBufDeregister(self.gpu_ptr) };
        }
    }
}
```

---

## 3. Double Buffering Implementation

### 3.1 Double Buffer Manager

```rust
use std::sync::Arc;
use parking_lot::Mutex;

/// Buffer state for ping-pong pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    /// Buffer is idle and can be used
    Idle,
    /// Buffer is being filled with data (I/O in progress)
    Filling,
    /// Buffer contains valid data ready for consumption
    Ready,
    /// Buffer is being consumed (GPU processing)
    Processing,
    /// Buffer is being written to storage
    Flushing,
}

/// Single buffer in double-buffer pair
pub struct GDSBuffer {
    pub gpu_ptr: CUdeviceptr,
    pub size: usize,
    pub state: BufferState,
    pub file_offset: i64,
    pub valid_bytes: usize,
    pub registration: RegisteredBuffer,
}

/// Double buffer manager for overlapped I/O
pub struct DoubleBufferManager {
    buffers: [Arc<Mutex<GDSBuffer>>; 2],
    active_index: usize,
    config: GDSConfig,
}

impl DoubleBufferManager {
    /// Create double buffer manager with GPU memory allocation
    pub fn new(config: &GDSConfig) -> Result<Self, GDSError> {
        let mut buffers = Vec::with_capacity(2);

        for i in 0..2 {
            // Allocate aligned GPU memory (CUDA API call)
            let gpu_ptr = unsafe { allocate_gpu_memory(config.buffer_size, config.alignment)? };

            let registration = RegisteredBuffer::new(gpu_ptr, config.buffer_size)?;

            let buffer = GDSBuffer {
                gpu_ptr,
                size: config.buffer_size,
                state: BufferState::Idle,
                file_offset: 0,
                valid_bytes: 0,
                registration,
            };

            buffers.push(Arc::new(Mutex::new(buffer)));
        }

        Ok(Self {
            buffers: [buffers.remove(0), buffers.remove(0)],
            active_index: 0,
            config: config.clone(),
        })
    }

    /// Get the front buffer (for consumption)
    pub fn front_buffer(&self) -> Arc<Mutex<GDSBuffer>> {
        self.buffers[self.active_index].clone()
    }

    /// Get the back buffer (for filling)
    pub fn back_buffer(&self) -> Arc<Mutex<GDSBuffer>> {
        self.buffers[1 - self.active_index].clone()
    }

    /// Swap buffers (ping-pong)
    pub fn swap(&mut self) {
        self.active_index = 1 - self.active_index;
    }

    /// Execute overlapped read pattern
    pub fn overlapped_read<F>(
        &mut self,
        file: &GDSFileHandle,
        total_size: usize,
        mut process_fn: F,
    ) -> Result<(), GDSError>
    where
        F: FnMut(CUdeviceptr, usize) -> Result<(), GDSError>,
    {
        let mut file_offset: i64 = 0;
        let mut remaining = total_size;

        // Initial fill of back buffer
        {
            let mut back = self.back_buffer().lock();
            let read_size = remaining.min(back.size);

            let bytes_read = unsafe {
                cuFileRead(
                    file.raw(),
                    back.gpu_ptr,
                    read_size,
                    file_offset,
                    0,
                )
            };

            if bytes_read < 0 {
                return Err(GDSError::ReadFailed(bytes_read as i32));
            }

            back.valid_bytes = bytes_read as usize;
            back.file_offset = file_offset;
            back.state = BufferState::Ready;

            file_offset += bytes_read as i64;
            remaining -= bytes_read as usize;
        }

        // Main processing loop with overlapped I/O
        while remaining > 0 || self.has_pending_data() {
            self.swap();

            // Start async fill of back buffer
            let fill_handle = if remaining > 0 {
                let back = self.back_buffer();
                let read_size = remaining.min(self.config.buffer_size);

                Some(std::thread::spawn(move || {
                    let mut back = back.lock();
                    back.state = BufferState::Filling;

                    let bytes_read = unsafe {
                        cuFileRead(
                            file.raw(),
                            back.gpu_ptr,
                            read_size,
                            file_offset,
                            0,
                        )
                    };

                    if bytes_read >= 0 {
                        back.valid_bytes = bytes_read as usize;
                        back.file_offset = file_offset;
                        back.state = BufferState::Ready;
                        Ok(bytes_read as usize)
                    } else {
                        back.state = BufferState::Idle;
                        Err(GDSError::ReadFailed(bytes_read as i32))
                    }
                }))
            } else {
                None
            };

            // Process front buffer while back buffer fills
            {
                let mut front = self.front_buffer().lock();
                if front.state == BufferState::Ready && front.valid_bytes > 0 {
                    front.state = BufferState::Processing;
                    process_fn(front.gpu_ptr, front.valid_bytes)?;
                    front.state = BufferState::Idle;
                    front.valid_bytes = 0;
                }
            }

            // Wait for fill to complete
            if let Some(handle) = fill_handle {
                let bytes_read = handle.join().map_err(|_| GDSError::ThreadPanic)??;
                file_offset += bytes_read as i64;
                remaining -= bytes_read;
            }
        }

        Ok(())
    }

    fn has_pending_data(&self) -> bool {
        let front = self.front_buffer().lock();
        front.state == BufferState::Ready && front.valid_bytes > 0
    }
}

// Placeholder for CUDA memory allocation
unsafe fn allocate_gpu_memory(size: usize, alignment: usize) -> Result<CUdeviceptr, GDSError> {
    // Actual implementation would call cuMemAlloc or cuMemAllocHost
    todo!("Implement CUDA memory allocation")
}
```

### 3.2 Async I/O Pipeline

```rust
use std::collections::VecDeque;
use crossbeam::channel::{bounded, Sender, Receiver};

/// I/O request for async pipeline
#[derive(Debug)]
pub struct IORequest {
    pub operation: IOOperation,
    pub buffer_index: usize,
    pub file_offset: i64,
    pub size: usize,
    pub callback: Option<Box<dyn FnOnce(Result<usize, GDSError>) + Send>>,
}

#[derive(Debug, Clone, Copy)]
pub enum IOOperation {
    Read,
    Write,
}

/// Async I/O pipeline with request coalescing
pub struct AsyncIOPipeline {
    request_tx: Sender<IORequest>,
    completion_rx: Receiver<(usize, Result<usize, GDSError>)>,
    pending_count: usize,
    max_outstanding: usize,
}

impl AsyncIOPipeline {
    pub fn new(
        config: &GDSConfig,
        file: Arc<GDSFileHandle>,
        buffers: Vec<Arc<Mutex<GDSBuffer>>>,
    ) -> Self {
        let (request_tx, request_rx) = bounded::<IORequest>(config.max_outstanding_io);
        let (completion_tx, completion_rx) = bounded(config.max_outstanding_io);

        // Spawn I/O worker thread
        std::thread::spawn(move || {
            Self::io_worker(request_rx, completion_tx, file, buffers);
        });

        Self {
            request_tx,
            completion_rx,
            pending_count: 0,
            max_outstanding: config.max_outstanding_io,
        }
    }

    fn io_worker(
        request_rx: Receiver<IORequest>,
        completion_tx: Sender<(usize, Result<usize, GDSError>)>,
        file: Arc<GDSFileHandle>,
        buffers: Vec<Arc<Mutex<GDSBuffer>>>,
    ) {
        while let Ok(request) = request_rx.recv() {
            let buffer = buffers[request.buffer_index].lock();

            let result = match request.operation {
                IOOperation::Read => {
                    let bytes = unsafe {
                        cuFileRead(
                            file.raw(),
                            buffer.gpu_ptr,
                            request.size,
                            request.file_offset,
                            0,
                        )
                    };
                    if bytes >= 0 {
                        Ok(bytes as usize)
                    } else {
                        Err(GDSError::ReadFailed(bytes as i32))
                    }
                }
                IOOperation::Write => {
                    let bytes = unsafe {
                        cuFileWrite(
                            file.raw(),
                            buffer.gpu_ptr,
                            request.size,
                            request.file_offset,
                            0,
                        )
                    };
                    if bytes >= 0 {
                        Ok(bytes as usize)
                    } else {
                        Err(GDSError::WriteFailed(bytes as i32))
                    }
                }
            };

            let _ = completion_tx.send((request.buffer_index, result));
        }
    }

    /// Submit I/O request (non-blocking)
    pub fn submit(&mut self, request: IORequest) -> Result<(), GDSError> {
        if self.pending_count >= self.max_outstanding {
            // Wait for at least one completion
            self.wait_one()?;
        }

        self.request_tx.send(request).map_err(|_| GDSError::ChannelClosed)?;
        self.pending_count += 1;

        Ok(())
    }

    /// Wait for single completion
    pub fn wait_one(&mut self) -> Result<(usize, usize), GDSError> {
        let (buffer_idx, result) = self.completion_rx
            .recv()
            .map_err(|_| GDSError::ChannelClosed)?;

        self.pending_count -= 1;
        Ok((buffer_idx, result?))
    }

    /// Wait for all pending operations
    pub fn drain(&mut self) -> Result<Vec<(usize, usize)>, GDSError> {
        let mut results = Vec::with_capacity(self.pending_count);
        while self.pending_count > 0 {
            results.push(self.wait_one()?);
        }
        Ok(results)
    }
}
```

---

## 4. Prefetch Scheduling Algorithm

### 4.1 Prefetch Scheduler

```rust
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Prefetch request with priority
#[derive(Debug, Eq, PartialEq)]
pub struct PrefetchRequest {
    pub file_offset: i64,
    pub size: usize,
    pub priority: u32,
    pub deadline: std::time::Instant,
}

impl Ord for PrefetchRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier deadline
        other.priority.cmp(&self.priority)
            .then_with(|| self.deadline.cmp(&other.deadline))
    }
}

impl PartialOrd for PrefetchRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Access pattern for prefetch prediction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    Sequential,
    Strided { stride: i64 },
    Random,
    Unknown,
}

/// Prefetch scheduler with adaptive prediction
pub struct PrefetchScheduler {
    config: GDSConfig,
    pending_prefetches: BinaryHeap<PrefetchRequest>,
    access_history: VecDeque<i64>,
    detected_pattern: AccessPattern,
    hit_count: u64,
    miss_count: u64,
    prefetch_buffers: Vec<Arc<Mutex<GDSBuffer>>>,
    active_prefetches: usize,
}

impl PrefetchScheduler {
    pub fn new(config: &GDSConfig, buffers: Vec<Arc<Mutex<GDSBuffer>>>) -> Self {
        Self {
            config: config.clone(),
            pending_prefetches: BinaryHeap::new(),
            access_history: VecDeque::with_capacity(64),
            detected_pattern: AccessPattern::Unknown,
            hit_count: 0,
            miss_count: 0,
            prefetch_buffers: buffers,
            active_prefetches: 0,
        }
    }

    /// Record access and update pattern detection
    pub fn record_access(&mut self, offset: i64) {
        self.access_history.push_back(offset);

        if self.access_history.len() > 64 {
            self.access_history.pop_front();
        }

        if self.access_history.len() >= 4 {
            self.detected_pattern = self.detect_pattern();
        }
    }

    /// Detect access pattern from history
    fn detect_pattern(&self) -> AccessPattern {
        if self.access_history.len() < 4 {
            return AccessPattern::Unknown;
        }

        let history: Vec<i64> = self.access_history.iter().copied().collect();
        let mut strides: Vec<i64> = Vec::with_capacity(history.len() - 1);

        for i in 1..history.len() {
            strides.push(history[i] - history[i - 1]);
        }

        // Check for sequential pattern
        let buffer_size = self.config.buffer_size as i64;
        let sequential_count = strides.iter()
            .filter(|&&s| s > 0 && s <= buffer_size)
            .count();

        if sequential_count as f64 / strides.len() as f64 > 0.8 {
            return AccessPattern::Sequential;
        }

        // Check for strided pattern
        if strides.len() >= 3 {
            let first_stride = strides[0];
            let stride_matches = strides.iter()
                .filter(|&&s| (s - first_stride).abs() < buffer_size / 10)
                .count();

            if stride_matches as f64 / strides.len() as f64 > 0.7 {
                return AccessPattern::Strided { stride: first_stride };
            }
        }

        // Check for random pattern
        let unique_strides: std::collections::HashSet<_> = strides.iter().collect();
        if unique_strides.len() as f64 / strides.len() as f64 > 0.8 {
            return AccessPattern::Random;
        }

        AccessPattern::Unknown
    }

    /// Schedule prefetch based on detected pattern
    pub fn schedule_prefetch(&mut self, current_offset: i64, file_size: i64) {
        if self.active_prefetches >= self.config.prefetch_slots {
            return; // All slots busy
        }

        let predictions = self.predict_next_accesses(current_offset, file_size);

        for (offset, priority) in predictions {
            if self.is_already_prefetched(offset) {
                continue;
            }

            let request = PrefetchRequest {
                file_offset: offset,
                size: self.config.buffer_size,
                priority,
                deadline: std::time::Instant::now()
                    + std::time::Duration::from_millis(100),
            };

            self.pending_prefetches.push(request);
        }
    }

    /// Predict next access locations
    fn predict_next_accesses(&self, current: i64, file_size: i64) -> Vec<(i64, u32)> {
        let buffer_size = self.config.buffer_size as i64;
        let max_predictions = self.config.prefetch_slots - self.active_prefetches;
        let mut predictions = Vec::with_capacity(max_predictions);

        match self.detected_pattern {
            AccessPattern::Sequential => {
                // Prefetch next N sequential buffers
                for i in 1..=max_predictions {
                    let offset = align_down(
                        (current + i as i64 * buffer_size) as usize,
                        self.config.alignment,
                    ) as i64;

                    if offset < file_size {
                        // Higher priority for closer offsets
                        let priority = 100 - i as u32 * 10;
                        predictions.push((offset, priority));
                    }
                }
            }

            AccessPattern::Strided { stride } => {
                // Prefetch along stride pattern
                for i in 1..=max_predictions {
                    let offset = align_down(
                        (current + i as i64 * stride) as usize,
                        self.config.alignment,
                    ) as i64;

                    if offset >= 0 && offset < file_size {
                        let priority = 80 - i as u32 * 10;
                        predictions.push((offset, priority));
                    }
                }
            }

            AccessPattern::Random | AccessPattern::Unknown => {
                // Conservative prefetch - just next buffer
                let offset = align_down(
                    (current + buffer_size) as usize,
                    self.config.alignment,
                ) as i64;

                if offset < file_size {
                    predictions.push((offset, 50));
                }
            }
        }

        predictions
    }

    /// Execute pending prefetch operations
    pub fn execute_prefetches(
        &mut self,
        file: &GDSFileHandle,
        pipeline: &mut AsyncIOPipeline,
    ) -> Result<usize, GDSError> {
        let mut executed = 0;

        while let Some(request) = self.pending_prefetches.pop() {
            if self.active_prefetches >= self.config.prefetch_slots {
                self.pending_prefetches.push(request);
                break;
            }

            // Find idle buffer
            if let Some(buffer_idx) = self.find_idle_buffer() {
                let io_request = IORequest {
                    operation: IOOperation::Read,
                    buffer_index: buffer_idx,
                    file_offset: request.file_offset,
                    size: request.size,
                    callback: None,
                };

                pipeline.submit(io_request)?;
                self.active_prefetches += 1;
                executed += 1;
            } else {
                self.pending_prefetches.push(request);
                break;
            }
        }

        Ok(executed)
    }

    fn find_idle_buffer(&self) -> Option<usize> {
        for (idx, buffer) in self.prefetch_buffers.iter().enumerate() {
            if buffer.lock().state == BufferState::Idle {
                return Some(idx);
            }
        }
        None
    }

    fn is_already_prefetched(&self, offset: i64) -> bool {
        for buffer in &self.prefetch_buffers {
            let buf = buffer.lock();
            if buf.state == BufferState::Ready && buf.file_offset == offset {
                return true;
            }
        }

        self.pending_prefetches.iter().any(|r| r.file_offset == offset)
    }

    /// Get prefetch hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }
}
```

---

## 5. Memory-Mapped GPU Access Patterns

### 5.1 Memory Map Manager

```rust
use std::collections::HashMap;

/// Memory region for GPU mapping
#[derive(Debug)]
pub struct GpuMappedRegion {
    pub file_offset: i64,
    pub size: usize,
    pub gpu_ptr: CUdeviceptr,
    pub host_ptr: *mut u8,
    pub access_count: u64,
    pub last_access: std::time::Instant,
    pub dirty: bool,
}

/// Memory-mapped GPU access manager
pub struct MemoryMapManager {
    config: GDSConfig,
    regions: HashMap<i64, GpuMappedRegion>,
    total_mapped: usize,
    max_mapped: usize,
    lru_list: VecDeque<i64>,
}

impl MemoryMapManager {
    pub fn new(config: &GDSConfig) -> Self {
        Self {
            config: config.clone(),
            regions: HashMap::new(),
            total_mapped: 0,
            max_mapped: config.total_gpu_memory(),
            lru_list: VecDeque::new(),
        }
    }

    /// Map file region to GPU memory
    pub fn map_region(
        &mut self,
        file: &GDSFileHandle,
        file_offset: i64,
        size: usize,
    ) -> Result<&GpuMappedRegion, GDSError> {
        // Align offset and size
        let aligned_offset = align_down(file_offset as usize, self.config.alignment) as i64;
        let aligned_size = align_up(size, self.config.alignment);

        // Check if already mapped
        if self.regions.contains_key(&aligned_offset) {
            self.touch(aligned_offset);
            return Ok(self.regions.get(&aligned_offset).unwrap());
        }

        // Evict if necessary
        while self.total_mapped + aligned_size > self.max_mapped {
            self.evict_lru()?;
        }

        // Allocate GPU memory
        let gpu_ptr = unsafe { allocate_gpu_memory(aligned_size, self.config.alignment)? };

        // Register with cuFile
        let _registration = RegisteredBuffer::new(gpu_ptr, aligned_size)?;

        // Read data from file
        let bytes_read = unsafe {
            cuFileRead(
                file.raw(),
                gpu_ptr,
                aligned_size,
                aligned_offset,
                0,
            )
        };

        if bytes_read < 0 {
            return Err(GDSError::ReadFailed(bytes_read as i32));
        }

        let region = GpuMappedRegion {
            file_offset: aligned_offset,
            size: aligned_size,
            gpu_ptr,
            host_ptr: std::ptr::null_mut(),
            access_count: 1,
            last_access: std::time::Instant::now(),
            dirty: false,
        };

        self.regions.insert(aligned_offset, region);
        self.lru_list.push_back(aligned_offset);
        self.total_mapped += aligned_size;

        Ok(self.regions.get(&aligned_offset).unwrap())
    }

    /// Mark region as modified
    pub fn mark_dirty(&mut self, file_offset: i64) {
        let aligned_offset = align_down(file_offset as usize, self.config.alignment) as i64;
        if let Some(region) = self.regions.get_mut(&aligned_offset) {
            region.dirty = true;
        }
    }

    /// Flush dirty regions to storage
    pub fn flush(&mut self, file: &GDSFileHandle) -> Result<usize, GDSError> {
        let mut flushed = 0;

        for region in self.regions.values_mut() {
            if region.dirty {
                let bytes_written = unsafe {
                    cuFileWrite(
                        file.raw(),
                        region.gpu_ptr,
                        region.size,
                        region.file_offset,
                        0,
                    )
                };

                if bytes_written < 0 {
                    return Err(GDSError::WriteFailed(bytes_written as i32));
                }

                region.dirty = false;
                flushed += bytes_written as usize;
            }
        }

        Ok(flushed)
    }

    /// Update LRU position
    fn touch(&mut self, offset: i64) {
        self.lru_list.retain(|&o| o != offset);
        self.lru_list.push_back(offset);

        if let Some(region) = self.regions.get_mut(&offset) {
            region.access_count += 1;
            region.last_access = std::time::Instant::now();
        }
    }

    /// Evict least recently used region
    fn evict_lru(&mut self) -> Result<(), GDSError> {
        if let Some(offset) = self.lru_list.pop_front() {
            if let Some(region) = self.regions.remove(&offset) {
                // Flush if dirty
                if region.dirty {
                    // Note: Need file handle here - simplified for spec
                    return Err(GDSError::EvictionDirty);
                }

                self.total_mapped -= region.size;
                // GPU memory freed when registration drops
            }
        }
        Ok(())
    }

    /// Unmap all regions
    pub fn unmap_all(&mut self, file: &GDSFileHandle) -> Result<(), GDSError> {
        self.flush(file)?;
        self.regions.clear();
        self.lru_list.clear();
        self.total_mapped = 0;
        Ok(())
    }
}
```

---

## 6. Error Handling

### 6.1 Error Types

```rust
use thiserror::Error;

/// GDS error types
#[derive(Error, Debug)]
pub enum GDSError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Driver initialization failed: {0:?}")]
    DriverInit(CUfileStatus),

    #[error("Driver shutdown failed: {0:?}")]
    DriverShutdown(CUfileStatus),

    #[error("File open failed: {0}")]
    FileOpen(#[from] std::io::Error),

    #[error("Handle registration failed: {0:?}")]
    HandleRegister(CUfileStatus),

    #[error("Buffer registration failed: {0:?}")]
    BufferRegister(CUfileStatus),

    #[error("Read operation failed with error code: {0}")]
    ReadFailed(i32),

    #[error("Write operation failed with error code: {0}")]
    WriteFailed(i32),

    #[error("Batch submission failed: {0:?}")]
    BatchSubmit(CUfileStatus),

    #[error("Buffer alignment error: expected {expected}, got {actual}")]
    AlignmentError { expected: usize, actual: usize },

    #[error("GPU memory allocation failed")]
    GpuMemoryAlloc,

    #[error("I/O channel closed unexpectedly")]
    ChannelClosed,

    #[error("I/O worker thread panicked")]
    ThreadPanic,

    #[error("Cannot evict dirty region without flushing")]
    EvictionDirty,

    #[error("Operation timed out after {0} ms")]
    Timeout(u64),

    #[error("GDS not supported on this platform")]
    NotSupported,

    #[error("Insufficient GPU memory: needed {needed}, available {available}")]
    InsufficientMemory { needed: usize, available: usize },
}

/// Convert cuFile error codes to descriptive messages
impl From<CUfileStatus> for GDSError {
    fn from(status: CUfileStatus) -> Self {
        match status {
            CUfileStatus::DriverNotInitialized => GDSError::DriverInit(status),
            CUfileStatus::InvalidValue => GDSError::InvalidConfig("Invalid parameter".into()),
            CUfileStatus::NotSupported => GDSError::NotSupported,
            CUfileStatus::IOError => GDSError::ReadFailed(-1),
            CUfileStatus::OutOfMemory => GDSError::GpuMemoryAlloc,
            _ => GDSError::DriverInit(status),
        }
    }
}
```

### 6.2 Error Recovery

```rust
/// Error recovery strategies
pub struct ErrorRecovery {
    max_retries: usize,
    backoff_base_ms: u64,
    fallback_enabled: bool,
}

impl ErrorRecovery {
    pub fn new(max_retries: usize, fallback_enabled: bool) -> Self {
        Self {
            max_retries,
            backoff_base_ms: 10,
            fallback_enabled,
        }
    }

    /// Execute operation with retry logic
    pub fn with_retry<T, F>(&self, mut operation: F) -> Result<T, GDSError>
    where
        F: FnMut() -> Result<T, GDSError>,
    {
        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            match operation() {
                Ok(result) => return Ok(result),
                Err(e) if self.is_retryable(&e) => {
                    last_error = Some(e);

                    if attempt < self.max_retries {
                        let backoff = self.backoff_base_ms * (1 << attempt);
                        std::thread::sleep(std::time::Duration::from_millis(backoff));
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Err(last_error.unwrap())
    }

    /// Check if error is retryable
    fn is_retryable(&self, error: &GDSError) -> bool {
        matches!(
            error,
            GDSError::ReadFailed(_)
                | GDSError::WriteFailed(_)
                | GDSError::Timeout(_)
                | GDSError::ChannelClosed
        )
    }

    /// Execute with fallback to standard I/O
    pub fn with_fallback<T, F, G>(
        &self,
        gds_operation: F,
        fallback_operation: G,
    ) -> Result<T, GDSError>
    where
        F: FnOnce() -> Result<T, GDSError>,
        G: FnOnce() -> Result<T, GDSError>,
    {
        match gds_operation() {
            Ok(result) => Ok(result),
            Err(e) if self.fallback_enabled && self.should_fallback(&e) => {
                log::warn!("GDS operation failed, falling back to standard I/O: {}", e);
                fallback_operation()
            }
            Err(e) => Err(e),
        }
    }

    fn should_fallback(&self, error: &GDSError) -> bool {
        matches!(
            error,
            GDSError::NotSupported
                | GDSError::DriverInit(_)
                | GDSError::GpuMemoryAlloc
        )
    }
}
```

---

## 7. Performance Metrics and Monitoring

### 7.1 Performance Tracker

```rust
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Performance metrics for GDS operations
#[derive(Debug, Default)]
pub struct GDSMetrics {
    // Throughput counters
    pub bytes_read: AtomicU64,
    pub bytes_written: AtomicU64,

    // Operation counts
    pub read_ops: AtomicU64,
    pub write_ops: AtomicU64,
    pub prefetch_hits: AtomicU64,
    pub prefetch_misses: AtomicU64,

    // Latency tracking (microseconds)
    pub read_latency_sum: AtomicU64,
    pub write_latency_sum: AtomicU64,
    pub max_read_latency: AtomicU64,
    pub max_write_latency: AtomicU64,

    // Buffer utilization
    pub buffer_waits: AtomicU64,
    pub active_buffers: AtomicUsize,
}

impl GDSMetrics {
    /// Record read operation
    pub fn record_read(&self, bytes: usize, latency: Duration) {
        self.bytes_read.fetch_add(bytes as u64, Ordering::Relaxed);
        self.read_ops.fetch_add(1, Ordering::Relaxed);

        let latency_us = latency.as_micros() as u64;
        self.read_latency_sum.fetch_add(latency_us, Ordering::Relaxed);
        self.max_read_latency.fetch_max(latency_us, Ordering::Relaxed);
    }

    /// Record write operation
    pub fn record_write(&self, bytes: usize, latency: Duration) {
        self.bytes_written.fetch_add(bytes as u64, Ordering::Relaxed);
        self.write_ops.fetch_add(1, Ordering::Relaxed);

        let latency_us = latency.as_micros() as u64;
        self.write_latency_sum.fetch_add(latency_us, Ordering::Relaxed);
        self.max_write_latency.fetch_max(latency_us, Ordering::Relaxed);
    }

    /// Calculate read throughput (bytes/sec)
    pub fn read_throughput(&self, elapsed: Duration) -> f64 {
        let bytes = self.bytes_read.load(Ordering::Relaxed);
        bytes as f64 / elapsed.as_secs_f64()
    }

    /// Calculate write throughput (bytes/sec)
    pub fn write_throughput(&self, elapsed: Duration) -> f64 {
        let bytes = self.bytes_written.load(Ordering::Relaxed);
        bytes as f64 / elapsed.as_secs_f64()
    }

    /// Calculate average read latency (microseconds)
    pub fn avg_read_latency(&self) -> f64 {
        let sum = self.read_latency_sum.load(Ordering::Relaxed);
        let ops = self.read_ops.load(Ordering::Relaxed);
        if ops == 0 { 0.0 } else { sum as f64 / ops as f64 }
    }

    /// Calculate prefetch hit rate
    pub fn prefetch_hit_rate(&self) -> f64 {
        let hits = self.prefetch_hits.load(Ordering::Relaxed);
        let misses = self.prefetch_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 { 0.0 } else { hits as f64 / total as f64 }
    }

    /// Check if performance targets are met
    pub fn check_targets(&self, elapsed: Duration) -> PerformanceStatus {
        const TARGET_READ_BPS: f64 = 4_000_000_000.0;   // 4 GB/s
        const TARGET_WRITE_BPS: f64 = 2_000_000_000.0;  // 2 GB/s
        const TARGET_LATENCY_US: f64 = 100.0;          // 100 microseconds

        let read_bps = self.read_throughput(elapsed);
        let write_bps = self.write_throughput(elapsed);
        let avg_latency = self.avg_read_latency();

        PerformanceStatus {
            read_target_met: read_bps >= TARGET_READ_BPS,
            write_target_met: write_bps >= TARGET_WRITE_BPS,
            latency_target_met: avg_latency <= TARGET_LATENCY_US,
            actual_read_bps: read_bps,
            actual_write_bps: write_bps,
            actual_latency_us: avg_latency,
        }
    }
}

#[derive(Debug)]
pub struct PerformanceStatus {
    pub read_target_met: bool,
    pub write_target_met: bool,
    pub latency_target_met: bool,
    pub actual_read_bps: f64,
    pub actual_write_bps: f64,
    pub actual_latency_us: f64,
}
```

---

## 8. Integration API

### 8.1 High-Level GDS Context

```rust
/// GPU Direct Storage context for high-level operations
pub struct GDSContext {
    config: GDSConfig,
    double_buffer: DoubleBufferManager,
    prefetch_scheduler: PrefetchScheduler,
    mmap_manager: MemoryMapManager,
    metrics: Arc<GDSMetrics>,
    recovery: ErrorRecovery,
    start_time: Instant,
}

impl GDSContext {
    /// Create new GDS context
    pub fn new(config: GDSConfig) -> Result<Self, GDSError> {
        config.validate()?;
        init_driver()?;

        let prefetch_buffers = Self::allocate_prefetch_buffers(&config)?;

        Ok(Self {
            double_buffer: DoubleBufferManager::new(&config)?,
            prefetch_scheduler: PrefetchScheduler::new(&config, prefetch_buffers),
            mmap_manager: MemoryMapManager::new(&config),
            metrics: Arc::new(GDSMetrics::default()),
            recovery: ErrorRecovery::new(3, config.mmap_fallback),
            config,
            start_time: Instant::now(),
        })
    }

    fn allocate_prefetch_buffers(config: &GDSConfig) -> Result<Vec<Arc<Mutex<GDSBuffer>>>, GDSError> {
        let mut buffers = Vec::with_capacity(config.prefetch_slots);
        for _ in 0..config.prefetch_slots {
            let gpu_ptr = unsafe { allocate_gpu_memory(config.buffer_size, config.alignment)? };
            let registration = RegisteredBuffer::new(gpu_ptr, config.buffer_size)?;

            buffers.push(Arc::new(Mutex::new(GDSBuffer {
                gpu_ptr,
                size: config.buffer_size,
                state: BufferState::Idle,
                file_offset: 0,
                valid_bytes: 0,
                registration,
            })));
        }
        Ok(buffers)
    }

    /// Read file to GPU memory with prefetching
    pub fn read_to_gpu(
        &mut self,
        path: &std::path::Path,
        gpu_dest: CUdeviceptr,
        size: usize,
    ) -> Result<usize, GDSError> {
        let file = GDSFileHandle::open(path, CUfileOpenFlags::RDONLY)?;
        let start = Instant::now();

        let result = self.recovery.with_retry(|| {
            let bytes_read = unsafe {
                cuFileRead(file.raw(), gpu_dest, size, 0, 0)
            };

            if bytes_read >= 0 {
                Ok(bytes_read as usize)
            } else {
                Err(GDSError::ReadFailed(bytes_read as i32))
            }
        })?;

        self.metrics.record_read(result, start.elapsed());
        Ok(result)
    }

    /// Write GPU memory to file
    pub fn write_from_gpu(
        &mut self,
        path: &std::path::Path,
        gpu_src: CUdeviceptr,
        size: usize,
    ) -> Result<usize, GDSError> {
        let file = GDSFileHandle::open(
            path,
            CUfileOpenFlags(CUfileOpenFlags::WRONLY.0 | CUfileOpenFlags::CREATE.0),
        )?;
        let start = Instant::now();

        let result = self.recovery.with_retry(|| {
            let bytes_written = unsafe {
                cuFileWrite(file.raw(), gpu_src, size, 0, 0)
            };

            if bytes_written >= 0 {
                Ok(bytes_written as usize)
            } else {
                Err(GDSError::WriteFailed(bytes_written as i32))
            }
        })?;

        self.metrics.record_write(result, start.elapsed());
        Ok(result)
    }

    /// Stream read with double buffering and processing callback
    pub fn stream_read<F>(
        &mut self,
        path: &std::path::Path,
        total_size: usize,
        process_fn: F,
    ) -> Result<(), GDSError>
    where
        F: FnMut(CUdeviceptr, usize) -> Result<(), GDSError>,
    {
        let file = GDSFileHandle::open(path, CUfileOpenFlags::RDONLY)?;
        self.double_buffer.overlapped_read(&file, total_size, process_fn)
    }

    /// Get current performance metrics
    pub fn metrics(&self) -> Arc<GDSMetrics> {
        self.metrics.clone()
    }

    /// Check if performance targets are being met
    pub fn performance_status(&self) -> PerformanceStatus {
        self.metrics.check_targets(self.start_time.elapsed())
    }
}

impl Drop for GDSContext {
    fn drop(&mut self) {
        // Cleanup handled by individual component Drop implementations
        let _ = shutdown_driver();
    }
}
```

---

## 9. Data Flow Diagram

```
+------------------+     +------------------+     +------------------+
|    Application   |     |   GDS Context    |     |   cuFile API     |
+------------------+     +------------------+     +------------------+
         |                       |                       |
         | read_to_gpu()         |                       |
         |---------------------->|                       |
         |                       | cuFileRead()          |
         |                       |---------------------->|
         |                       |                       |
         |                       |<-- DMA Transfer ----->| NVMe SSD
         |                       |                       |
         |                       |<----------------------|
         |<----------------------|                       |
         |                       |                       |
         | stream_read()         |                       |
         |---------------------->|                       |
         |                       | Double Buffer Swap    |
         |                       |----+                  |
         |                       |    |                  |
         |                       |<---+                  |
         |                       |                       |
         |                       | Prefetch Schedule     |
         |                       |----+                  |
         |                       |    | cuFileRead()     |
         |                       |    |----------------->|
         |                       |<---+                  |
         |                       |                       |
         | process(buffer)       |                       |
         |<----------------------|                       |
         |                       |                       |
```

---

## 10. Performance Tuning Guidelines

| Parameter | Low Latency | High Throughput | Balanced |
|-----------|-------------|-----------------|----------|
| buffer_size | 64 KB | 64 MB | 16 MB |
| prefetch_slots | 2 | 16 | 8 |
| max_outstanding_io | 4 | 64 | 32 |
| double_buffering | false | true | true |
| prefetch_distance | 256 KB | 256 MB | 64 MB |
| batch_mode | false | true | true |
| max_batch_size | 1 | 32 | 16 |

---

## References

- NVIDIA GPUDirect Storage Documentation
- cuFile API Reference
- NVMe Specification 1.4
- CUDA Programming Guide - Memory Management
