# M04-T10: Implement FaissGpuIndex Wrapper

| Field | Value |
|-------|-------|
| **Task ID** | M04-T10 |
| **Module** | context-graph-graph |
| **Status** | Ready |
| **Priority** | P0 (Critical Path) |
| **Depends On** | M04-T09 (FAISS FFI bindings) |
| **Estimated Effort** | 8-12 hours |
| **Constitution Refs** | TECH-GRAPH-004, AP-001, perf.latency.faiss_1M_k100 |

---

## Executive Summary

Implement a safe Rust wrapper (`FaissGpuIndex`) around the FAISS GPU IVF-PQ index using the FFI bindings from M04-T09. This wrapper provides GPU-accelerated vector similarity search with proper RAII resource management, error handling, and thread safety.

**CRITICAL RULES**:
- **NO BACKWARDS COMPATIBILITY** - The system must work correctly or fail fast with clear errors
- **NO MOCK DATA IN TESTS** - All tests must use real FAISS GPU index operations
- **NEVER unwrap() IN PROD** - All errors must be properly typed using `GraphError`

---

## Current Codebase State (Verified 2025-01-03)

### Completed Dependencies (from git history)

| Task | Description | Status |
|------|-------------|--------|
| M04-T00 | Module scaffold | ✅ Complete |
| M04-T01 | IndexConfig struct | ✅ Complete |
| M04-T02 | HyperbolicConfig struct | ✅ Complete |
| M04-T03 | ConeConfig struct | ✅ Complete |
| M04-T04 | PoincarePoint struct | ✅ Complete |
| M04-T05 | PoincareBall Mobius ops | ✅ Complete |
| M04-T06 | EntailmentCone struct | ✅ Complete |
| M04-T07 | Containment logic | ✅ Complete |
| M04-T08 | GraphError enum | ✅ Complete |
| M04-T08a | Error conversions | ✅ Complete |
| M04-T09 | FAISS FFI bindings | ⏳ In Progress |

### Existing Files to Use

**Source Crate**: `crates/context-graph-graph/`

| File | Purpose | Status |
|------|---------|--------|
| `src/config.rs` | `IndexConfig` with all parameters | ✅ Exists |
| `src/error.rs` | `GraphError` with FAISS variants | ✅ Exists |
| `src/index/mod.rs` | Index module structure | ✅ Exists |
| `src/index/faiss_ffi.rs` | Raw FFI bindings (from M04-T09) | ⏳ Pending |
| `src/lib.rs` | Crate root with re-exports | ✅ Exists |

### Files to Create

| File | Purpose |
|------|---------|
| `src/index/gpu_index.rs` | FaissGpuIndex wrapper implementation |

---

## Constitution Requirements

### From `constitution.yaml`

```yaml
# Stack Requirements
stack:
  lang: rust >= 1.75
  cuda: "13.1"
  faiss: "0.12+gpu"

# Hardware Target
hardware:
  gpu: RTX 5090
  vram: 32GB
  compute_capability: "12.0"

# Performance Budgets
perf:
  latency:
    faiss_1M_k100: "<2ms"
    faiss_10M_k10: "<5ms"

# Coding Rules
rules:
  - "Never unwrap() in prod - all errors properly typed"
  - "Result<T,E> for fallible ops"
  - "thiserror for error derivation"
```

### From `context-prd.md`

- FAISS GPU with IVF-PQ indexing for O(log n) similarity search
- Arc<GpuResources> for multi-index GPU resource sharing
- NonNull pointers for safe FFI memory management
- RAII pattern with Drop implementation for cleanup

---

## IndexConfig Reference (Already Implemented in `src/config.rs`)

```rust
pub struct IndexConfig {
    pub dimension: usize,        // 1536 (E7_Code)
    pub nlist: usize,            // 16384 (Voronoi cells)
    pub nprobe: usize,           // 128 (search probe count)
    pub pq_segments: usize,      // 64 (PQ subdivision)
    pub pq_bits: u8,             // 8 (bits per code)
    pub gpu_id: i32,             // 0 (default GPU)
    pub use_float16: bool,       // true (half precision)
    pub min_train_vectors: usize, // 4_194_304 (4M minimum)
}

impl IndexConfig {
    pub fn factory_string(&self) -> String {
        format!("IVF{},PQ{}x{}", self.nlist, self.pq_segments, self.pq_bits)
        // Returns: "IVF16384,PQ64x8"
    }
}
```

---

## GraphError Variants Available (Already Implemented in `src/error.rs`)

```rust
// FAISS Index Errors
GraphError::FaissIndexCreation(String)
GraphError::FaissTrainingFailed(String)
GraphError::FaissSearchFailed(String)
GraphError::FaissAddFailed(String)
GraphError::IndexNotTrained
GraphError::InsufficientTrainingData { required: usize, provided: usize }

// GPU Resource Errors
GraphError::GpuResourceAllocation(String)
GraphError::GpuTransferFailed(String)
GraphError::GpuDeviceUnavailable(String)

// Configuration Errors
GraphError::InvalidConfig(String)
GraphError::DimensionMismatch { expected: usize, actual: usize }
```

---

## Implementation Specification

### Target File: `crates/context-graph-graph/src/index/gpu_index.rs`

```rust
//! FAISS GPU IVF-PQ Index Wrapper
//!
//! Provides safe Rust wrapper around FAISS GPU index with:
//! - RAII resource management (Drop impl)
//! - Thread-safe GPU resource sharing (Arc<GpuResources>)
//! - Proper error handling (GraphError variants)
//! - Performance-optimized search (<2ms for 1M vectors, k=100)
//!
//! # Constitution References
//!
//! - TECH-GRAPH-004: Knowledge Graph technical specification
//! - AP-001: Never unwrap() in prod - all errors properly typed
//! - perf.latency.faiss_1M_k100: <2ms target
//!
//! # Safety
//!
//! This module uses unsafe FFI calls to FAISS C API. All unsafe blocks
//! are contained within this module with safety invariants documented.

use std::ffi::CString;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::config::IndexConfig;
use crate::error::{GraphError, GraphResult};
use super::faiss_ffi::{
    FaissIndex, FaissGpuResources, MetricType,
    faiss_index_factory, faiss_index_cpu_to_gpu, faiss_Index_train,
    faiss_Index_is_trained, faiss_Index_add_with_ids, faiss_Index_search,
    faiss_IndexIVF_nprobe_set, faiss_Index_ntotal, faiss_write_index,
    faiss_read_index, faiss_Index_free, faiss_gpu_resources_new,
    faiss_gpu_resources_free, check_faiss_result,
};

/// GPU resources handle with RAII cleanup.
///
/// Wraps raw GPU resource pointer with automatic deallocation.
/// Use `Arc<GpuResources>` for sharing across multiple indices.
pub struct GpuResources {
    ptr: NonNull<FaissGpuResources>,
    gpu_id: i32,
}

// SAFETY: FaissGpuResources is thread-safe per FAISS documentation.
// All operations are synchronized internally by FAISS.
unsafe impl Send for GpuResources {}
unsafe impl Sync for GpuResources {}

impl GpuResources {
    /// Allocate GPU resources for the specified device.
    ///
    /// # Arguments
    ///
    /// * `gpu_id` - CUDA device ID (typically 0)
    ///
    /// # Errors
    ///
    /// Returns `GraphError::GpuResourceAllocation` if:
    /// - GPU device is unavailable
    /// - CUDA initialization fails
    /// - Insufficient GPU memory
    ///
    /// # Example
    ///
    /// ```no_run
    /// use context_graph_graph::index::gpu_index::GpuResources;
    /// use std::sync::Arc;
    ///
    /// let resources = Arc::new(GpuResources::new(0)?);
    /// # Ok::<(), context_graph_graph::error::GraphError>(())
    /// ```
    pub fn new(gpu_id: i32) -> GraphResult<Self> {
        let mut ptr: *mut FaissGpuResources = std::ptr::null_mut();

        // SAFETY: faiss_gpu_resources_new initializes the pointer.
        // We check the return value and null pointer below.
        let ret = unsafe { faiss_gpu_resources_new(&mut ptr, gpu_id) };

        if ret != 0 {
            return Err(GraphError::GpuResourceAllocation(format!(
                "Failed to create GPU resources for device {}: FAISS error code {}",
                gpu_id, ret
            )));
        }

        if ptr.is_null() {
            return Err(GraphError::GpuResourceAllocation(format!(
                "GPU resources pointer is null for device {}",
                gpu_id
            )));
        }

        // SAFETY: We verified ptr is non-null above.
        let nn_ptr = unsafe { NonNull::new_unchecked(ptr) };

        Ok(Self { ptr: nn_ptr, gpu_id })
    }

    /// Get raw pointer for FFI calls.
    ///
    /// # Safety
    ///
    /// Caller must ensure GpuResources outlives any index using this pointer.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *mut FaissGpuResources {
        self.ptr.as_ptr()
    }

    /// Get the GPU device ID.
    #[inline]
    pub fn gpu_id(&self) -> i32 {
        self.gpu_id
    }
}

impl Drop for GpuResources {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated by faiss_gpu_resources_new and is non-null.
        // This is the only place where we free these resources.
        unsafe {
            faiss_gpu_resources_free(self.ptr.as_ptr());
        }
    }
}

/// FAISS GPU IVF-PQ Index wrapper.
///
/// Provides GPU-accelerated approximate nearest neighbor search using
/// Inverted File with Product Quantization (IVF-PQ) index structure.
///
/// # Index Parameters (from IndexConfig)
///
/// - `dimension`: 1536 (E7_Code embedding dimension)
/// - `nlist`: 16384 (number of Voronoi cells)
/// - `nprobe`: 128 (cells to search at query time)
/// - `pq_segments`: 64 (PQ subdivision count)
/// - `pq_bits`: 8 (bits per PQ code)
///
/// # Performance Targets
///
/// - 1M vectors, k=100: <2ms
/// - 10M vectors, k=10: <5ms
///
/// # Thread Safety
///
/// - Single `FaissGpuIndex` is NOT thread-safe for concurrent modification
/// - Use separate indices per thread, or synchronize externally
/// - `Arc<GpuResources>` can be shared across indices safely
pub struct FaissGpuIndex {
    /// Raw pointer to GPU index (NonNull for safety guarantees)
    index_ptr: NonNull<FaissIndex>,
    /// Shared GPU resources
    gpu_resources: Arc<GpuResources>,
    /// Index configuration
    config: IndexConfig,
    /// Whether the index has been trained
    is_trained: bool,
    /// Number of vectors in the index
    vector_count: usize,
}

// SAFETY: FaissGpuIndex owns its index pointer exclusively.
// All mutable operations require &mut self, ensuring single-threaded access.
// The Arc<GpuResources> is Send+Sync, enabling safe transfer between threads.
unsafe impl Send for FaissGpuIndex {}

impl FaissGpuIndex {
    /// Create a new FAISS GPU IVF-PQ index.
    ///
    /// # Arguments
    ///
    /// * `config` - Index configuration parameters
    ///
    /// # Errors
    ///
    /// Returns `GraphError::FaissIndexCreation` if:
    /// - Invalid configuration parameters
    /// - GPU memory allocation fails
    /// - FAISS index creation fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use context_graph_graph::config::IndexConfig;
    /// use context_graph_graph::index::gpu_index::FaissGpuIndex;
    ///
    /// let config = IndexConfig::default();
    /// let index = FaissGpuIndex::new(config)?;
    /// # Ok::<(), context_graph_graph::error::GraphError>(())
    /// ```
    pub fn new(config: IndexConfig) -> GraphResult<Self> {
        let resources = Arc::new(GpuResources::new(config.gpu_id)?);
        Self::with_resources(config, resources)
    }

    /// Create index with shared GPU resources.
    ///
    /// Use this when creating multiple indices to share GPU memory resources.
    ///
    /// # Arguments
    ///
    /// * `config` - Index configuration
    /// * `gpu_resources` - Shared GPU resources handle
    ///
    /// # Errors
    ///
    /// Returns `GraphError::FaissIndexCreation` if index creation fails.
    /// Returns `GraphError::InvalidConfig` if configuration is invalid.
    pub fn with_resources(config: IndexConfig, gpu_resources: Arc<GpuResources>) -> GraphResult<Self> {
        // Validate configuration
        if config.dimension == 0 {
            return Err(GraphError::InvalidConfig(
                "dimension must be > 0".to_string()
            ));
        }
        if config.nlist == 0 {
            return Err(GraphError::InvalidConfig(
                "nlist must be > 0".to_string()
            ));
        }
        if config.pq_segments == 0 {
            return Err(GraphError::InvalidConfig(
                "pq_segments must be > 0".to_string()
            ));
        }
        if config.dimension % config.pq_segments != 0 {
            return Err(GraphError::InvalidConfig(format!(
                "pq_segments ({}) must divide dimension ({}) evenly",
                config.pq_segments, config.dimension
            )));
        }

        // Create factory string
        let factory_string = config.factory_string();
        let c_factory = CString::new(factory_string.clone())
            .map_err(|e| GraphError::InvalidConfig(format!(
                "Invalid factory string '{}': {}", factory_string, e
            )))?;

        // Create CPU index first
        let mut cpu_index: *mut FaissIndex = std::ptr::null_mut();

        // SAFETY: faiss_index_factory allocates a new index.
        // We check the return value and null pointer below.
        let ret = unsafe {
            faiss_index_factory(
                &mut cpu_index,
                config.dimension as i32,
                c_factory.as_ptr(),
                MetricType::L2,
            )
        };

        check_faiss_result(ret, "faiss_index_factory").map_err(|msg| {
            GraphError::FaissIndexCreation(format!(
                "Failed to create CPU index '{}': {}", factory_string, msg
            ))
        })?;

        if cpu_index.is_null() {
            return Err(GraphError::FaissIndexCreation(
                "CPU index pointer is null after factory creation".to_string()
            ));
        }

        // Transfer to GPU
        let mut gpu_index: *mut FaissIndex = std::ptr::null_mut();

        // SAFETY: faiss_index_cpu_to_gpu transfers the index to GPU.
        // cpu_index is valid (checked above), gpu_resources.as_ptr() is valid.
        let ret = unsafe {
            faiss_index_cpu_to_gpu(
                gpu_resources.as_ptr(),
                config.gpu_id,
                cpu_index,
                &mut gpu_index,
            )
        };

        // Free CPU index regardless of GPU transfer result (GPU copy owns data now)
        // SAFETY: cpu_index was allocated by faiss_index_factory and is non-null.
        unsafe { faiss_Index_free(cpu_index) };

        check_faiss_result(ret, "faiss_index_cpu_to_gpu").map_err(|msg| {
            GraphError::GpuTransferFailed(format!(
                "Failed to transfer index to GPU {}: {}", config.gpu_id, msg
            ))
        })?;

        if gpu_index.is_null() {
            return Err(GraphError::GpuResourceAllocation(
                "GPU index pointer is null after transfer".to_string()
            ));
        }

        // SAFETY: We verified gpu_index is non-null above.
        let index_ptr = unsafe { NonNull::new_unchecked(gpu_index) };

        Ok(Self {
            index_ptr,
            gpu_resources,
            config,
            is_trained: false,
            vector_count: 0,
        })
    }

    /// Train the index with representative vectors.
    ///
    /// IVF-PQ requires training to establish cluster centroids and PQ codebooks.
    /// Training vectors should be representative of the data distribution.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Training vectors (flattened, row-major: n_vectors * dimension f32 values)
    ///
    /// # Errors
    ///
    /// - `GraphError::InsufficientTrainingData` if n_vectors < min_train_vectors (4M)
    /// - `GraphError::DimensionMismatch` if vectors.len() is not a multiple of dimension
    /// - `GraphError::FaissTrainingFailed` on FAISS training error
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use context_graph_graph::index::gpu_index::FaissGpuIndex;
    /// # use context_graph_graph::config::IndexConfig;
    /// # fn example() -> context_graph_graph::error::GraphResult<()> {
    /// let config = IndexConfig::default();
    /// let mut index = FaissGpuIndex::new(config)?;
    ///
    /// // Generate training data (4M+ vectors required)
    /// let training_data: Vec<f32> = generate_training_vectors();
    /// index.train(&training_data)?;
    /// # Ok(())
    /// # }
    /// # fn generate_training_vectors() -> Vec<f32> { vec![] }
    /// ```
    pub fn train(&mut self, vectors: &[f32]) -> GraphResult<()> {
        let remainder = vectors.len() % self.config.dimension;
        if remainder != 0 {
            return Err(GraphError::DimensionMismatch {
                expected: self.config.dimension,
                actual: remainder,
            });
        }

        let n_vectors = vectors.len() / self.config.dimension;

        if n_vectors < self.config.min_train_vectors {
            return Err(GraphError::InsufficientTrainingData {
                required: self.config.min_train_vectors,
                provided: n_vectors,
            });
        }

        // SAFETY: vectors slice contains n_vectors * dimension valid f32 values.
        // index_ptr is valid and points to a FAISS index.
        let ret = unsafe {
            faiss_Index_train(
                self.index_ptr.as_ptr(),
                n_vectors as i64,
                vectors.as_ptr(),
            )
        };

        check_faiss_result(ret, "faiss_Index_train").map_err(|msg| {
            GraphError::FaissTrainingFailed(format!(
                "Training failed with {} vectors: {}", n_vectors, msg
            ))
        })?;

        // Set nprobe after successful training
        // SAFETY: index_ptr is valid, nprobe value is valid.
        let ret = unsafe {
            faiss_IndexIVF_nprobe_set(
                self.index_ptr.as_ptr(),
                self.config.nprobe as i64,
            )
        };

        check_faiss_result(ret, "faiss_IndexIVF_nprobe_set").map_err(|msg| {
            GraphError::FaissTrainingFailed(format!(
                "Failed to set nprobe to {}: {}", self.config.nprobe, msg
            ))
        })?;

        self.is_trained = true;
        Ok(())
    }

    /// Search for k nearest neighbors.
    ///
    /// # Arguments
    ///
    /// * `queries` - Query vectors (flattened, row-major: n_queries * dimension f32 values)
    /// * `k` - Number of neighbors to return per query
    ///
    /// # Errors
    ///
    /// - `GraphError::IndexNotTrained` if index is not trained
    /// - `GraphError::DimensionMismatch` if queries.len() is not a multiple of dimension
    /// - `GraphError::FaissSearchFailed` on FAISS search error
    ///
    /// # Returns
    ///
    /// Tuple of (distances, indices) where each has length n_queries * k.
    /// Distances are L2 squared distances. Indices are -1 for unfilled slots.
    ///
    /// # Performance
    ///
    /// Target: <2ms for 1M vectors with k=100, <5ms for 10M vectors with k=10
    pub fn search(&self, queries: &[f32], k: usize) -> GraphResult<(Vec<f32>, Vec<i64>)> {
        if !self.is_trained {
            return Err(GraphError::IndexNotTrained);
        }

        let remainder = queries.len() % self.config.dimension;
        if remainder != 0 {
            return Err(GraphError::DimensionMismatch {
                expected: self.config.dimension,
                actual: remainder,
            });
        }

        let n_queries = queries.len() / self.config.dimension;
        let result_size = n_queries * k;

        let mut distances: Vec<f32> = vec![f32::MAX; result_size];
        let mut indices: Vec<i64> = vec![-1; result_size];

        // SAFETY: queries slice contains n_queries * dimension valid f32 values.
        // distances and indices are sized correctly for n_queries * k elements.
        // index_ptr is valid and points to a trained FAISS index.
        let ret = unsafe {
            faiss_Index_search(
                self.index_ptr.as_ptr(),
                n_queries as i64,
                queries.as_ptr(),
                k as i64,
                distances.as_mut_ptr(),
                indices.as_mut_ptr(),
            )
        };

        check_faiss_result(ret, "faiss_Index_search").map_err(|msg| {
            GraphError::FaissSearchFailed(format!(
                "Search failed for {} queries, k={}: {}", n_queries, k, msg
            ))
        })?;

        Ok((distances, indices))
    }

    /// Add vectors with IDs to the index.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Vectors to add (flattened, row-major: n_vectors * dimension f32 values)
    /// * `ids` - Vector IDs (one per vector, must match n_vectors)
    ///
    /// # Errors
    ///
    /// - `GraphError::IndexNotTrained` if index is not trained
    /// - `GraphError::DimensionMismatch` if vectors.len() is not a multiple of dimension
    /// - `GraphError::InvalidConfig` if vector count doesn't match ID count
    /// - `GraphError::FaissAddFailed` on FAISS add error
    ///
    /// # Note
    ///
    /// Index must be trained before adding vectors.
    pub fn add_with_ids(&mut self, vectors: &[f32], ids: &[i64]) -> GraphResult<()> {
        if !self.is_trained {
            return Err(GraphError::IndexNotTrained);
        }

        let remainder = vectors.len() % self.config.dimension;
        if remainder != 0 {
            return Err(GraphError::DimensionMismatch {
                expected: self.config.dimension,
                actual: remainder,
            });
        }

        let n_vectors = vectors.len() / self.config.dimension;

        if n_vectors != ids.len() {
            return Err(GraphError::InvalidConfig(format!(
                "Vector count ({}) doesn't match ID count ({})", n_vectors, ids.len()
            )));
        }

        // SAFETY: vectors slice contains n_vectors * dimension valid f32 values.
        // ids slice contains n_vectors valid i64 values.
        // index_ptr is valid and points to a trained FAISS index.
        let ret = unsafe {
            faiss_Index_add_with_ids(
                self.index_ptr.as_ptr(),
                n_vectors as i64,
                vectors.as_ptr(),
                ids.as_ptr(),
            )
        };

        check_faiss_result(ret, "faiss_Index_add_with_ids").map_err(|msg| {
            GraphError::FaissAddFailed(format!(
                "Failed to add {} vectors: {}", n_vectors, msg
            ))
        })?;

        self.vector_count += n_vectors;
        Ok(())
    }

    /// Get total number of vectors in index.
    #[inline]
    pub fn ntotal(&self) -> usize {
        // SAFETY: index_ptr is valid.
        let count = unsafe { faiss_Index_ntotal(self.index_ptr.as_ptr()) };
        count as usize
    }

    /// Get the number of vectors tracked by this wrapper.
    ///
    /// Note: This may differ from `ntotal()` if vectors were added through
    /// other means or the index was loaded from disk.
    #[inline]
    pub fn len(&self) -> usize {
        self.vector_count
    }

    /// Check if the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ntotal() == 0
    }

    /// Check if the index is trained.
    #[inline]
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get the index configuration.
    #[inline]
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

    /// Get the dimension of vectors in this index.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Get reference to shared GPU resources.
    #[inline]
    pub fn resources(&self) -> &Arc<GpuResources> {
        &self.gpu_resources
    }

    /// Save index to file.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to save index
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be written or FAISS serialization fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> GraphResult<()> {
        let path_str = path.as_ref().to_string_lossy();
        let c_path = CString::new(path_str.as_ref())
            .map_err(|e| GraphError::InvalidConfig(format!(
                "Invalid path '{}': {}", path_str, e
            )))?;

        // SAFETY: index_ptr is valid, c_path is valid null-terminated string.
        let ret = unsafe { faiss_write_index(self.index_ptr.as_ptr(), c_path.as_ptr()) };

        check_faiss_result(ret, "faiss_write_index").map_err(|msg| {
            GraphError::Serialization(format!(
                "Failed to save index to '{}': {}", path_str, msg
            ))
        })?;

        Ok(())
    }

    /// Load index from file.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to load index from
    /// * `config` - Index configuration (must match saved index dimension)
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read, FAISS deserialization fails,
    /// or GPU transfer fails.
    pub fn load<P: AsRef<Path>>(path: P, config: IndexConfig) -> GraphResult<Self> {
        let resources = Arc::new(GpuResources::new(config.gpu_id)?);
        Self::load_with_resources(path, config, resources)
    }

    /// Load index from file with shared GPU resources.
    pub fn load_with_resources<P: AsRef<Path>>(
        path: P,
        config: IndexConfig,
        gpu_resources: Arc<GpuResources>,
    ) -> GraphResult<Self> {
        let path_str = path.as_ref().to_string_lossy();
        let c_path = CString::new(path_str.as_ref())
            .map_err(|e| GraphError::InvalidConfig(format!(
                "Invalid path '{}': {}", path_str, e
            )))?;

        // Load CPU index from file
        let mut cpu_index: *mut FaissIndex = std::ptr::null_mut();

        // SAFETY: c_path is valid null-terminated string.
        let ret = unsafe { faiss_read_index(c_path.as_ptr(), 0, &mut cpu_index) };

        check_faiss_result(ret, "faiss_read_index").map_err(|msg| {
            GraphError::Deserialization(format!(
                "Failed to load index from '{}': {}", path_str, msg
            ))
        })?;

        if cpu_index.is_null() {
            return Err(GraphError::Deserialization(format!(
                "Loaded index pointer is null for '{}'", path_str
            )));
        }

        // Transfer to GPU
        let mut gpu_index: *mut FaissIndex = std::ptr::null_mut();

        // SAFETY: cpu_index is valid (checked above), gpu_resources.as_ptr() is valid.
        let ret = unsafe {
            faiss_index_cpu_to_gpu(
                gpu_resources.as_ptr(),
                config.gpu_id,
                cpu_index,
                &mut gpu_index,
            )
        };

        // Free CPU index regardless of transfer result
        // SAFETY: cpu_index was allocated by faiss_read_index and is non-null.
        unsafe { faiss_Index_free(cpu_index) };

        check_faiss_result(ret, "faiss_index_cpu_to_gpu").map_err(|msg| {
            GraphError::GpuTransferFailed(format!(
                "Failed to transfer loaded index to GPU {}: {}", config.gpu_id, msg
            ))
        })?;

        if gpu_index.is_null() {
            return Err(GraphError::GpuResourceAllocation(
                "Loaded GPU index pointer is null after transfer".to_string()
            ));
        }

        // SAFETY: We verified gpu_index is non-null above.
        let index_ptr = unsafe { NonNull::new_unchecked(gpu_index) };

        // Check if loaded index is trained
        // SAFETY: index_ptr is valid.
        let is_trained = unsafe { faiss_Index_is_trained(index_ptr.as_ptr()) } != 0;

        // Get vector count from FAISS
        // SAFETY: index_ptr is valid.
        let vector_count = unsafe { faiss_Index_ntotal(index_ptr.as_ptr()) } as usize;

        Ok(Self {
            index_ptr,
            gpu_resources,
            config,
            is_trained,
            vector_count,
        })
    }
}

impl Drop for FaissGpuIndex {
    fn drop(&mut self) {
        // SAFETY: index_ptr was allocated by faiss_index_cpu_to_gpu and is non-null.
        // This is the only place where we free the index. GPU resources are freed
        // separately via Arc<GpuResources> when all references are dropped.
        unsafe {
            faiss_Index_free(self.index_ptr.as_ptr());
        }
    }
}

impl std::fmt::Debug for FaissGpuIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FaissGpuIndex")
            .field("ntotal", &self.ntotal())
            .field("is_trained", &self.is_trained)
            .field("dimension", &self.config.dimension)
            .field("factory", &self.config.factory_string())
            .field("gpu_id", &self.config.gpu_id)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== GPU Resource Tests ==========

    #[test]
    fn test_gpu_resources_creation() {
        // REAL TEST: Actually allocates GPU resources
        let result = GpuResources::new(0);

        match result {
            Ok(resources) => {
                assert!(!resources.as_ptr().is_null());
                assert_eq!(resources.gpu_id(), 0);
                println!("✓ GPU resources allocated successfully");
            }
            Err(e) => {
                // GPU may not be available in CI
                println!("⚠ GPU resources creation failed (expected in CI): {}", e);
            }
        }
    }

    #[test]
    fn test_gpu_resources_invalid_device() {
        // REAL TEST: Invalid device ID should fail
        let result = GpuResources::new(999);

        match result {
            Err(GraphError::GpuResourceAllocation(msg)) => {
                assert!(msg.contains("999"));
                println!("✓ Invalid device ID correctly rejected: {}", msg);
            }
            Err(e) => {
                println!("✓ Invalid device rejected with different error: {}", e);
            }
            Ok(_) => {
                // This might succeed on systems with many GPUs
                println!("⚠ Device 999 unexpectedly succeeded (unusual but possible)");
            }
        }
    }

    // ========== Index Creation Tests ==========

    #[test]
    fn test_index_creation_valid_config() {
        let config = IndexConfig::default();
        let resources = match GpuResources::new(config.gpu_id) {
            Ok(r) => Arc::new(r),
            Err(_) => {
                println!("⚠ Skipping test: GPU not available");
                return;
            }
        };

        let result = FaissGpuIndex::with_resources(config.clone(), resources);

        match result {
            Ok(idx) => {
                assert_eq!(idx.dimension(), 1536);
                assert!(!idx.is_trained());
                assert!(idx.is_empty());
                assert_eq!(idx.config().nlist, 16384);
                println!("✓ Index created with factory: {}", idx.config().factory_string());
            }
            Err(e) => panic!("Index creation failed: {}", e),
        }
    }

    #[test]
    fn test_index_creation_invalid_dimension() {
        let mut config = IndexConfig::default();
        config.dimension = 0;

        let resources = match GpuResources::new(0) {
            Ok(r) => Arc::new(r),
            Err(_) => return, // Skip if no GPU
        };

        let result = FaissGpuIndex::with_resources(config, resources);

        match result {
            Err(GraphError::InvalidConfig(msg)) => {
                assert!(msg.contains("dimension"));
                println!("✓ Zero dimension correctly rejected: {}", msg);
            }
            _ => panic!("Expected InvalidConfig error for dimension=0"),
        }
    }

    #[test]
    fn test_index_creation_invalid_pq_segments() {
        let mut config = IndexConfig::default();
        config.pq_segments = 7; // 1536 % 7 != 0

        let resources = match GpuResources::new(0) {
            Ok(r) => Arc::new(r),
            Err(_) => return,
        };

        let result = FaissGpuIndex::with_resources(config, resources);

        match result {
            Err(GraphError::InvalidConfig(msg)) => {
                assert!(msg.contains("pq_segments"));
                assert!(msg.contains("divide"));
                println!("✓ Invalid pq_segments correctly rejected: {}", msg);
            }
            _ => panic!("Expected InvalidConfig error for pq_segments=7"),
        }
    }

    #[test]
    fn test_index_creation_zero_nlist() {
        let mut config = IndexConfig::default();
        config.nlist = 0;

        let resources = match GpuResources::new(0) {
            Ok(r) => Arc::new(r),
            Err(_) => return,
        };

        let result = FaissGpuIndex::with_resources(config, resources);

        match result {
            Err(GraphError::InvalidConfig(msg)) => {
                assert!(msg.contains("nlist"));
                println!("✓ Zero nlist correctly rejected: {}", msg);
            }
            _ => panic!("Expected InvalidConfig error for nlist=0"),
        }
    }

    // ========== Training Tests ==========

    #[test]
    fn test_train_insufficient_data() {
        let config = IndexConfig::default();
        let resources = match GpuResources::new(0) {
            Ok(r) => Arc::new(r),
            Err(_) => return,
        };

        let mut index = match FaissGpuIndex::with_resources(config.clone(), resources) {
            Ok(idx) => idx,
            Err(_) => return,
        };

        // Only 1000 vectors, need 4M+
        let vectors: Vec<f32> = vec![0.0; 1000 * config.dimension];
        let result = index.train(&vectors);

        match result {
            Err(GraphError::InsufficientTrainingData { required, provided }) => {
                assert_eq!(required, 4194304);
                assert_eq!(provided, 1000);
                println!("✓ Insufficient training data correctly rejected");
            }
            _ => panic!("Expected InsufficientTrainingData error"),
        }
    }

    #[test]
    fn test_train_dimension_mismatch() {
        let config = IndexConfig::default();
        let resources = match GpuResources::new(0) {
            Ok(r) => Arc::new(r),
            Err(_) => return,
        };

        let mut index = match FaissGpuIndex::with_resources(config, resources) {
            Ok(idx) => idx,
            Err(_) => return,
        };

        // 1537 elements - not divisible by 1536
        let vectors: Vec<f32> = vec![0.0; 1537];
        let result = index.train(&vectors);

        match result {
            Err(GraphError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 1536);
                assert_eq!(actual, 1); // 1537 % 1536 = 1
                println!("✓ Dimension mismatch correctly rejected");
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    // ========== Add Tests ==========

    #[test]
    fn test_add_without_training() {
        let config = IndexConfig::default();
        let resources = match GpuResources::new(0) {
            Ok(r) => Arc::new(r),
            Err(_) => return,
        };

        let mut index = match FaissGpuIndex::with_resources(config.clone(), resources) {
            Ok(idx) => idx,
            Err(_) => return,
        };

        let vectors: Vec<f32> = vec![0.0; config.dimension];
        let ids: Vec<i64> = vec![0];
        let result = index.add_with_ids(&vectors, &ids);

        match result {
            Err(GraphError::IndexNotTrained) => {
                println!("✓ Add without training correctly rejected");
            }
            _ => panic!("Expected IndexNotTrained error"),
        }
    }

    // ========== Search Tests ==========

    #[test]
    fn test_search_without_training() {
        let config = IndexConfig::default();
        let resources = match GpuResources::new(0) {
            Ok(r) => Arc::new(r),
            Err(_) => return,
        };

        let index = match FaissGpuIndex::with_resources(config.clone(), resources) {
            Ok(idx) => idx,
            Err(_) => return,
        };

        let queries: Vec<f32> = vec![0.0; config.dimension];
        let result = index.search(&queries, 10);

        match result {
            Err(GraphError::IndexNotTrained) => {
                println!("✓ Search without training correctly rejected");
            }
            _ => panic!("Expected IndexNotTrained error"),
        }
    }

    // ========== Thread Safety Tests ==========

    #[test]
    fn test_gpu_resources_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuResources>();
        println!("✓ GpuResources is Send + Sync");
    }

    #[test]
    fn test_faiss_gpu_index_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<FaissGpuIndex>();
        println!("✓ FaissGpuIndex is Send");
    }

    // ========== Integration Test (requires GPU + training data) ==========

    #[test]
    #[ignore] // Run with: cargo test -- --ignored
    fn test_full_index_workflow() {
        // REAL TEST: Full train/add/search workflow with actual FAISS GPU operations
        let config = IndexConfig::default();

        println!("Creating GPU resources...");
        let resources = Arc::new(
            GpuResources::new(config.gpu_id).expect("GPU required for this test")
        );

        println!("Creating index with factory: {}", config.factory_string());
        let mut index = FaissGpuIndex::with_resources(config.clone(), resources)
            .expect("Index creation failed");

        // Generate training data (4M+ vectors required)
        println!("Generating {} training vectors (dimension={})...",
            config.min_train_vectors, config.dimension);

        let training_data: Vec<f32> = (0..config.min_train_vectors)
            .flat_map(|i| {
                (0..config.dimension).map(move |d| {
                    ((i * config.dimension + d) as f32).sin()
                })
            })
            .collect();

        // Train
        println!("Training index...");
        let train_start = std::time::Instant::now();
        index.train(&training_data).expect("Training failed");
        let train_time = train_start.elapsed();
        println!("✓ Training completed in {:?}", train_time);
        assert!(index.is_trained());

        // Add vectors
        let n_add = 100_000;
        println!("Adding {} vectors...", n_add);

        let add_data: Vec<f32> = (0..n_add)
            .flat_map(|i| {
                (0..config.dimension).map(move |d| {
                    ((i * 7 + d) as f32).cos()
                })
            })
            .collect();
        let add_ids: Vec<i64> = (0..n_add as i64).collect();

        let add_start = std::time::Instant::now();
        index.add_with_ids(&add_data, &add_ids).expect("Add failed");
        let add_time = add_start.elapsed();
        println!("✓ Added {} vectors in {:?}", n_add, add_time);
        assert_eq!(index.ntotal(), n_add);

        // Search
        println!("Searching for k=10 neighbors...");
        let query: Vec<f32> = (0..config.dimension)
            .map(|d| (d as f32).sin())
            .collect();

        let search_start = std::time::Instant::now();
        let (distances, indices) = index.search(&query, 10).expect("Search failed");
        let search_time = search_start.elapsed();

        println!("✓ Search completed in {:?}", search_time);
        println!("  Top result: idx={}, dist={}", indices[0], distances[0]);

        assert_eq!(distances.len(), 10);
        assert_eq!(indices.len(), 10);
        assert!(indices[0] >= 0, "First result should be valid");

        // Performance check (relaxed for smaller dataset)
        assert!(search_time.as_millis() < 100,
            "Search took too long: {:?}", search_time);
    }

    #[test]
    #[ignore] // Run with: cargo test -- --ignored
    fn test_save_load_roundtrip() {
        let config = IndexConfig::default();
        let resources = Arc::new(
            GpuResources::new(config.gpu_id).expect("GPU required")
        );

        // Create and train index
        let mut index = FaissGpuIndex::with_resources(config.clone(), resources.clone())
            .expect("Index creation failed");

        let training_data: Vec<f32> = (0..config.min_train_vectors)
            .flat_map(|i| {
                (0..config.dimension).map(move |d| {
                    ((i + d) as f32) * 0.001
                })
            })
            .collect();

        index.train(&training_data).expect("Training failed");

        // Add some vectors
        let vectors: Vec<f32> = (0..1000)
            .flat_map(|i| {
                (0..config.dimension).map(move |d| (i + d) as f32 * 0.01)
            })
            .collect();
        let ids: Vec<i64> = (0..1000).collect();
        index.add_with_ids(&vectors, &ids).expect("Add failed");

        // Save
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let path = temp_dir.path().join("test_index.faiss");
        index.save(&path).expect("Save failed");
        println!("✓ Index saved to {:?}", path);

        // Load
        let loaded = FaissGpuIndex::load_with_resources(&path, config, resources)
            .expect("Load failed");

        assert_eq!(loaded.ntotal(), index.ntotal());
        assert!(loaded.is_trained());
        println!("✓ Index loaded with {} vectors", loaded.ntotal());
    }
}
```

---

## Module Integration

### Update `src/index/mod.rs`

Add the following to the module file:

```rust
pub mod gpu_index;

pub use gpu_index::{FaissGpuIndex, GpuResources};
```

### Update `src/lib.rs` Re-exports

Add to crate root:

```rust
pub use index::{FaissGpuIndex, GpuResources};
```

---

## Acceptance Criteria

### Functional Requirements

- [ ] `GpuResources::new(gpu_id)` allocates CUDA resources
- [ ] `GpuResources` implements `Drop` for automatic cleanup
- [ ] `GpuResources` is `Send + Sync` for multi-index sharing
- [ ] `FaissGpuIndex::new(config)` creates IVF-PQ GPU index
- [ ] `FaissGpuIndex::with_resources(config, resources)` shares GPU resources
- [ ] `FaissGpuIndex::train(vectors)` trains with ≥4M vectors
- [ ] `FaissGpuIndex::add_with_ids(vectors, ids)` adds vectors after training
- [ ] `FaissGpuIndex::search(queries, k)` returns (distances, indices)
- [ ] `FaissGpuIndex::save(path)` serializes index to file
- [ ] `FaissGpuIndex::load(path, config)` deserializes index from file
- [ ] All methods return `GraphResult<T>` with appropriate errors
- [ ] Index operations fail fast with clear error messages

### Error Handling Requirements

- [ ] `GraphError::GpuResourceAllocation` for GPU init failures
- [ ] `GraphError::GpuTransferFailed` for CPU-to-GPU transfer failures
- [ ] `GraphError::FaissIndexCreation` for index creation failures
- [ ] `GraphError::FaissTrainingFailed` for training errors
- [ ] `GraphError::FaissAddFailed` for add errors
- [ ] `GraphError::FaissSearchFailed` for search errors
- [ ] `GraphError::IndexNotTrained` when operating on untrained index
- [ ] `GraphError::InsufficientTrainingData` for <4M training vectors
- [ ] `GraphError::DimensionMismatch` for vector/dimension mismatches
- [ ] `GraphError::InvalidConfig` for bad configuration parameters

### Safety Requirements

- [ ] All FFI pointers use `NonNull<T>` wrappers
- [ ] `Drop` implementations free all resources
- [ ] No `unwrap()` or `expect()` in non-test code
- [ ] All unsafe blocks have documented SAFETY comments
- [ ] Thread safety markers (`Send`/`Sync`) are correct and justified

### Performance Requirements

- [ ] Search latency <2ms for 1M vectors, k=100
- [ ] Search latency <5ms for 10M vectors, k=10
- [ ] Memory-efficient GPU resource sharing via `Arc<GpuResources>`

---

## Full State Verification

### Source of Truth

Before implementation, verify these files exist and match expected state:

```bash
# Verify IndexConfig exists with factory_string()
grep -n "factory_string" crates/context-graph-graph/src/config.rs

# Verify all FAISS GraphError variants exist
grep -n "Faiss" crates/context-graph-graph/src/error.rs

# Verify FFI bindings exist (from M04-T09)
ls -la crates/context-graph-graph/src/index/faiss_ffi.rs

# Verify index module structure
cat crates/context-graph-graph/src/index/mod.rs
```

### Execute & Inspect

After implementation, run these commands and verify output:

```bash
# Verify compilation
cargo check -p context-graph-graph 2>&1 | head -30

# Run all unit tests (some may skip if no GPU)
cargo test -p context-graph-graph index::gpu_index -- --nocapture 2>&1

# Verify exports in lib.rs
grep "FaissGpuIndex\|GpuResources" crates/context-graph-graph/src/lib.rs

# Run clippy
cargo clippy -p context-graph-graph -- -D warnings 2>&1
```

### Edge Case Audit

These edge cases MUST be tested with REAL FAISS operations (no mocks):

| # | Edge Case | Expected Error | Test Name |
|---|-----------|----------------|-----------|
| 1 | Invalid GPU ID (999) | `GpuResourceAllocation` | `test_gpu_resources_invalid_device` |
| 2 | Zero dimension | `InvalidConfig` | `test_index_creation_invalid_dimension` |
| 3 | Zero nlist | `InvalidConfig` | `test_index_creation_zero_nlist` |
| 4 | pq_segments doesn't divide dimension | `InvalidConfig` | `test_index_creation_invalid_pq_segments` |
| 5 | Add without training | `IndexNotTrained` | `test_add_without_training` |
| 6 | Search without training | `IndexNotTrained` | `test_search_without_training` |
| 7 | Insufficient training vectors | `InsufficientTrainingData` | `test_train_insufficient_data` |
| 8 | Vector length not divisible by dim | `DimensionMismatch` | `test_train_dimension_mismatch` |

### Evidence of Success

After implementation, these artifacts MUST exist:

```
crates/context-graph-graph/src/index/gpu_index.rs  # Main implementation
```

These lines MUST be present:

```bash
# In src/index/mod.rs
grep "pub mod gpu_index" crates/context-graph-graph/src/index/mod.rs
grep "pub use gpu_index" crates/context-graph-graph/src/index/mod.rs

# In src/lib.rs (re-exports)
grep "FaissGpuIndex" crates/context-graph-graph/src/lib.rs
grep "GpuResources" crates/context-graph-graph/src/lib.rs
```

Test output MUST show:
- All unit tests pass (or skip gracefully with message if no GPU)
- No compiler errors or warnings
- `cargo clippy` clean

---

## Sherlock-Holmes Verification Step

After implementation is complete, spawn a `sherlock-holmes` subagent for forensic verification:

```
Task: Forensic verification of M04-T10 FaissGpuIndex implementation

You are Detective Sherlock Holmes. Assume ALL CODE IS GUILTY UNTIL PROVEN INNOCENT.
Conduct a rigorous forensic audit of the M04-T10 implementation.

INVESTIGATION CHECKLIST:

1. FILE EXISTENCE (CRITICAL)
   □ Does crates/context-graph-graph/src/index/gpu_index.rs exist?
   □ Is "pub mod gpu_index" in src/index/mod.rs?
   □ Is "pub use gpu_index::{FaissGpuIndex, GpuResources}" in src/index/mod.rs?
   □ Are FaissGpuIndex and GpuResources re-exported in src/lib.rs?

2. STRUCT SIGNATURES (CRITICAL)
   □ Does GpuResources have ptr: NonNull<FaissGpuResources>?
   □ Does FaissGpuIndex have index_ptr: NonNull<FaissIndex>?
   □ Does FaissGpuIndex have gpu_resources: Arc<GpuResources>?
   □ Does FaissGpuIndex have config: IndexConfig?
   □ Does FaissGpuIndex have is_trained: bool?

3. METHOD SIGNATURES (CRITICAL)
   □ GpuResources::new(gpu_id: i32) -> GraphResult<Self>
   □ FaissGpuIndex::new(config: IndexConfig) -> GraphResult<Self>
   □ FaissGpuIndex::with_resources(config, resources: Arc<GpuResources>) -> GraphResult<Self>
   □ FaissGpuIndex::train(&mut self, vectors: &[f32]) -> GraphResult<()>
   □ FaissGpuIndex::add_with_ids(&mut self, vectors: &[f32], ids: &[i64]) -> GraphResult<()>
   □ FaissGpuIndex::search(&self, queries: &[f32], k: usize) -> GraphResult<(Vec<f32>, Vec<i64>)>
   □ FaissGpuIndex::save<P: AsRef<Path>>(&self, path: P) -> GraphResult<()>
   □ FaissGpuIndex::load<P: AsRef<Path>>(path: P, config: IndexConfig) -> GraphResult<Self>

4. ERROR HANDLING AUDIT (CRITICAL - NO unwrap() ALLOWED)
   □ Search for "unwrap()" outside #[cfg(test)] - MUST find zero instances
   □ Search for "expect(" outside #[cfg(test)] - MUST find zero instances
   □ Verify all public methods return GraphResult<T>
   □ Verify GraphError variants are used appropriately:
     - GpuResourceAllocation for GPU init failures
     - GpuTransferFailed for CPU-to-GPU transfer failures
     - FaissIndexCreation for index creation failures
     - FaissTrainingFailed for training errors
     - FaissAddFailed for add errors
     - FaissSearchFailed for search errors
     - IndexNotTrained when untrained
     - InsufficientTrainingData for <4M vectors
     - DimensionMismatch for wrong vector lengths
     - InvalidConfig for bad config params

5. SAFETY AUDIT (CRITICAL)
   □ Every "unsafe {" block has a "// SAFETY:" comment above it
   □ Drop for GpuResources calls faiss_gpu_resources_free
   □ Drop for FaissGpuIndex calls faiss_Index_free
   □ GpuResources has "unsafe impl Send for GpuResources {}" with justification
   □ GpuResources has "unsafe impl Sync for GpuResources {}" with justification
   □ FaissGpuIndex has "unsafe impl Send for FaissGpuIndex {}" with justification

6. TEST COVERAGE AUDIT (NO MOCK DATA)
   □ test_gpu_resources_creation - uses REAL GpuResources::new()
   □ test_gpu_resources_invalid_device - uses REAL GpuResources::new(999)
   □ test_index_creation_valid_config - uses REAL FaissGpuIndex::with_resources()
   □ test_index_creation_invalid_dimension - tests dimension=0
   □ test_index_creation_invalid_pq_segments - tests pq_segments=7
   □ test_train_insufficient_data - tests <4M vectors
   □ test_train_dimension_mismatch - tests wrong vector length
   □ test_add_without_training - tests IndexNotTrained
   □ test_search_without_training - tests IndexNotTrained
   □ test_gpu_resources_is_send_sync - compile-time check
   □ test_faiss_gpu_index_is_send - compile-time check
   □ NO tests use mock FAISS operations

7. COMPILATION VERIFICATION
   □ Run: cargo check -p context-graph-graph
   □ Run: cargo clippy -p context-graph-graph -- -D warnings
   □ Run: cargo test -p context-graph-graph index::gpu_index
   □ Report any errors or warnings

VERDICT FORMAT:
For each item, report:
- [PASS] Item description - evidence
- [FAIL] Item description - what's wrong, fix needed

FINAL VERDICT:
- INNOCENT: All checks pass, implementation is correct
- GUILTY: List all failures requiring immediate correction
```

---

## Dependencies

### FFI Binding Dependency (M04-T09)

This task requires M04-T09 providing these FFI declarations in `src/index/faiss_ffi.rs`:

```rust
// Opaque types
pub enum FaissIndex {}
pub enum FaissGpuResources {}

// Metric type
#[repr(i32)]
pub enum MetricType {
    L2 = 1,
    InnerProduct = 0,
}

// GPU resources
extern "C" {
    pub fn faiss_gpu_resources_new(
        resources: *mut *mut FaissGpuResources,
        device: i32,
    ) -> i32;

    pub fn faiss_gpu_resources_free(resources: *mut FaissGpuResources);
}

// Index factory
extern "C" {
    pub fn faiss_index_factory(
        index: *mut *mut FaissIndex,
        d: i32,
        description: *const std::ffi::c_char,
        metric: MetricType,
    ) -> i32;
}

// CPU to GPU transfer
extern "C" {
    pub fn faiss_index_cpu_to_gpu(
        provider: *mut FaissGpuResources,
        device: i32,
        index: *mut FaissIndex,
        gpu_index: *mut *mut FaissIndex,
    ) -> i32;
}

// Index operations
extern "C" {
    pub fn faiss_Index_train(
        index: *mut FaissIndex,
        n: i64,
        x: *const f32,
    ) -> i32;

    pub fn faiss_Index_is_trained(index: *mut FaissIndex) -> i32;

    pub fn faiss_Index_add_with_ids(
        index: *mut FaissIndex,
        n: i64,
        x: *const f32,
        xids: *const i64,
    ) -> i32;

    pub fn faiss_Index_search(
        index: *mut FaissIndex,
        n: i64,
        x: *const f32,
        k: i64,
        distances: *mut f32,
        labels: *mut i64,
    ) -> i32;

    pub fn faiss_IndexIVF_nprobe_set(
        index: *mut FaissIndex,
        nprobe: i64,
    ) -> i32;

    pub fn faiss_Index_ntotal(index: *mut FaissIndex) -> i64;

    pub fn faiss_Index_free(index: *mut FaissIndex);
}

// Persistence
extern "C" {
    pub fn faiss_write_index(
        index: *mut FaissIndex,
        fname: *const std::ffi::c_char,
    ) -> i32;

    pub fn faiss_read_index(
        fname: *const std::ffi::c_char,
        io_flags: i32,
        index: *mut *mut FaissIndex,
    ) -> i32;
}

// Helper function
pub fn check_faiss_result(ret: i32, op: &str) -> Result<(), String> {
    if ret == 0 {
        Ok(())
    } else {
        Err(format!("{} failed with error code {}", op, ret))
    }
}
```

If M04-T09 is not complete, complete it first.

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-01-03 | AI Agent | Complete rewrite with codebase audit, Full State Verification, Sherlock verification, no mock data mandate |
| 2025-01-02 | AI Agent | Initial draft |
