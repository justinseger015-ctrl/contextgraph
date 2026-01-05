# Module 4: Knowledge Graph - Technical Specification

```yaml
metadata:
  id: TECH-GRAPH-004
  version: 2.1.0
  module: Knowledge Graph
  phase: 2
  status: draft
  created: 2025-12-31
  dependencies:
    - TECH-GHOST-001
    - TECH-CORE-002
    - TECH-EMBED-003
  functional_spec_ref: SPEC-GRAPH-004
```

---

## 1. Architecture Overview

```
context-graph-graph/
├── src/
│   ├── lib.rs                    # Public API
│   ├── config.rs                 # IndexConfig, ConeConfig
│   ├── index/
│   │   ├── mod.rs
│   │   ├── faiss_ffi.rs          # FAISS C API bindings
│   │   └── gpu_index.rs          # RTX 5090 GPU backend
│   ├── storage/
│   │   ├── mod.rs
│   │   ├── rocksdb.rs            # Graph storage backend
│   │   └── edges.rs              # GraphEdge with Marblestone
│   ├── hyperbolic/
│   │   ├── mod.rs
│   │   ├── poincare.rs           # 64D Poincare ball
│   │   └── mobius.rs             # Mobius operations
│   ├── entailment/
│   │   ├── mod.rs
│   │   └── cones.rs              # EntailmentCone struct
│   ├── traversal/
│   │   ├── mod.rs
│   │   ├── bfs.rs                # Breadth-first search
│   │   └── dfs.rs                # Depth-first search
│   ├── marblestone/
│   │   ├── mod.rs
│   │   └── domain_search.rs      # Domain-aware ANN search
│   └── error.rs                  # Error types
├── kernels/
│   ├── poincare_distance.cu      # CUDA distance kernel
│   └── cone_check.cu             # CUDA cone membership
└── Cargo.toml
```

---

## 2. Configuration Types

```rust
// src/config.rs

/// IVF-PQ index configuration for RTX 5090 GPU
/// REQ-KG-001 through REQ-KG-008
#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub dimension: usize,         // 1536 (FuseMoE output)
    pub nlist: usize,             // 16384 clusters
    pub pq_segments: usize,       // 64 segments
    pub pq_bits: usize,           // 8 bits (256 centroids/segment)
    pub nprobe: usize,            // 128 probes
    pub gpu_id: i32,              // 0 (RTX 5090 primary)
    pub use_float16: bool,        // true for memory efficiency
    pub min_train_vectors: usize, // 4_194_304 (256 * nlist)
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimension: 1536, nlist: 16384, pq_segments: 64, pq_bits: 8,
            nprobe: 128, gpu_id: 0, use_float16: true, min_train_vectors: 4_194_304,
        }
    }
}

impl IndexConfig {
    pub fn factory_string(&self) -> String {
        format!("IVF{},PQ{}x{}", self.nlist, self.pq_segments, self.pq_bits)
    }
}

/// Hyperbolic geometry configuration
#[derive(Debug, Clone)]
pub struct HyperbolicConfig {
    pub dim: usize,           // 64
    pub curvature: f32,       // -1.0
    pub eps: f32,             // 1e-7
    pub max_norm: f32,        // 1.0 - 1e-5
}

impl Default for HyperbolicConfig {
    fn default() -> Self {
        Self { dim: 64, curvature: -1.0, eps: 1e-7, max_norm: 1.0 - 1e-5 }
    }
}

/// Entailment cone configuration
#[derive(Debug, Clone)]
pub struct ConeConfig {
    pub min_aperture: f32,        // 0.1 radians
    pub max_aperture: f32,        // 1.5 radians
    pub base_aperture: f32,       // 1.0 radians
    pub aperture_decay: f32,      // 0.85 per level
    pub membership_threshold: f32, // 0.7
}

impl Default for ConeConfig {
    fn default() -> Self {
        Self { min_aperture: 0.1, max_aperture: 1.5, base_aperture: 1.0,
               aperture_decay: 0.85, membership_threshold: 0.7 }
    }
}
```

---

## 3. FAISS GPU Index

### 3.1 FFI Bindings

```rust
// src/index/faiss_ffi.rs

use std::ffi::{c_char, c_float, c_int, c_long, c_void, CString};
use std::ptr::NonNull;

#[link(name = "faiss_c")]
extern "C" {
    fn faiss_index_factory(d: c_int, description: *const c_char, metric: c_int) -> *mut c_void;
    fn faiss_StandardGpuResources_new() -> *mut c_void;
    fn faiss_StandardGpuResources_free(res: *mut c_void);
    fn faiss_index_cpu_to_gpu(res: *mut c_void, dev: c_int, idx: *mut c_void) -> *mut c_void;
    fn faiss_index_gpu_to_cpu(index: *mut c_void) -> *mut c_void;
    fn faiss_Index_train(idx: *mut c_void, n: c_long, x: *const c_float) -> c_int;
    fn faiss_Index_is_trained(idx: *mut c_void) -> c_int;
    fn faiss_Index_add_with_ids(idx: *mut c_void, n: c_long, x: *const c_float, ids: *const c_long) -> c_int;
    fn faiss_Index_search(idx: *mut c_void, n: c_long, x: *const c_float, k: c_long, d: *mut c_float, l: *mut c_long) -> c_int;
    fn faiss_IndexIVF_nprobe_set(idx: *mut c_void, nprobe: c_long);
    fn faiss_Index_ntotal(idx: *mut c_void) -> c_long;
    fn faiss_write_index(idx: *mut c_void, path: *const c_char) -> c_int;
    fn faiss_read_index(path: *const c_char, io_flags: c_int) -> *mut c_void;
    fn faiss_Index_free(idx: *mut c_void);
}

#[repr(i32)]
pub enum MetricType { InnerProduct = 0, L2 = 1 }

pub struct GpuResources { ptr: NonNull<c_void> }

impl GpuResources {
    pub fn new() -> Result<Self, GraphError> {
        let ptr = unsafe { faiss_StandardGpuResources_new() };
        NonNull::new(ptr).map(|p| Self { ptr: p }).ok_or(GraphError::GpuResourceAllocation)
    }
    pub fn as_ptr(&self) -> *mut c_void { self.ptr.as_ptr() }
}

impl Drop for GpuResources {
    fn drop(&mut self) { unsafe { faiss_StandardGpuResources_free(self.ptr.as_ptr()) } }
}

unsafe impl Send for GpuResources {}
unsafe impl Sync for GpuResources {}
```

### 3.2 GPU Index Manager

```rust
// src/index/gpu_index.rs

use crate::config::IndexConfig;
use crate::error::GraphError;
use crate::index::faiss_ffi::*;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub ids: Vec<i64>,
    pub distances: Vec<f32>,
    pub k: usize,
    pub num_queries: usize,
}

impl SearchResult {
    pub fn query_results(&self, idx: usize) -> impl Iterator<Item = (i64, f32)> + '_ {
        let s = idx * self.k;
        self.ids[s..s+self.k].iter().copied()
            .zip(self.distances[s..s+self.k].iter().copied())
            .filter(|(id, _)| *id >= 0)
    }
}

pub struct FaissGpuIndex {
    gpu_ptr: NonNull<c_void>,
    resources: Arc<GpuResources>,
    config: IndexConfig,
    trained: bool,
}

impl FaissGpuIndex {
    pub fn new(config: IndexConfig) -> Result<Self, GraphError> {
        let resources = Arc::new(GpuResources::new()?);
        let factory = CString::new(config.factory_string()).map_err(|_| GraphError::InvalidConfig)?;

        let cpu_ptr = unsafe { faiss_index_factory(config.dimension as c_int, factory.as_ptr(), MetricType::L2 as c_int) };
        if cpu_ptr.is_null() { return Err(GraphError::FaissIndexCreation); }

        let gpu_ptr = unsafe { faiss_index_cpu_to_gpu(resources.as_ptr(), config.gpu_id, cpu_ptr) };
        unsafe { faiss_Index_free(cpu_ptr) };

        NonNull::new(gpu_ptr).map(|ptr| Self { gpu_ptr: ptr, resources, config, trained: false })
            .ok_or(GraphError::GpuTransferFailed)
    }

    pub fn train(&mut self, vectors: &[f32]) -> Result<(), GraphError> {
        let n = vectors.len() / self.config.dimension;
        if n < self.config.min_train_vectors {
            return Err(GraphError::InsufficientTrainingData { provided: n, required: self.config.min_train_vectors });
        }
        if unsafe { faiss_Index_train(self.gpu_ptr.as_ptr(), n as c_long, vectors.as_ptr()) } != 0 {
            return Err(GraphError::FaissTrainingFailed);
        }
        unsafe { faiss_IndexIVF_nprobe_set(self.gpu_ptr.as_ptr(), self.config.nprobe as c_long) };
        self.trained = true;
        Ok(())
    }

    /// k-NN search: <5ms for k=10, nprobe=128, 10M vectors
    pub fn search(&self, queries: &[f32], k: usize) -> Result<SearchResult, GraphError> {
        if !self.trained { return Err(GraphError::IndexNotTrained); }
        let n = queries.len() / self.config.dimension;
        let mut distances = vec![f32::MAX; n * k];
        let mut labels = vec![-1i64; n * k];
        if unsafe { faiss_Index_search(self.gpu_ptr.as_ptr(), n as c_long, queries.as_ptr(), k as c_long, distances.as_mut_ptr(), labels.as_mut_ptr()) } != 0 {
            return Err(GraphError::FaissSearchFailed);
        }
        Ok(SearchResult { ids: labels, distances, k, num_queries: n })
    }

    pub fn add_with_ids(&self, vectors: &[f32], ids: &[i64]) -> Result<(), GraphError> {
        if !self.trained { return Err(GraphError::IndexNotTrained); }
        let n = vectors.len() / self.config.dimension;
        if n != ids.len() { return Err(GraphError::VectorIdMismatch); }
        if unsafe { faiss_Index_add_with_ids(self.gpu_ptr.as_ptr(), n as c_long, vectors.as_ptr(), ids.as_ptr()) } != 0 {
            return Err(GraphError::FaissAddFailed);
        }
        Ok(())
    }
}

impl Drop for FaissGpuIndex {
    fn drop(&mut self) { unsafe { faiss_Index_free(self.gpu_ptr.as_ptr()) } }
}

unsafe impl Send for FaissGpuIndex {}
unsafe impl Sync for FaissGpuIndex {}
```

---

## 4. Graph Storage (RocksDB)

### 4.1 Edge Types with Marblestone Fields

```rust
// src/storage/edges.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EdgeType { Semantic = 0, Temporal = 1, Causal = 2, Hierarchical = 3 }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Domain { Code, Legal, Medical, Creative, Research, #[default] General }

/// Neurotransmitter-inspired edge weights (Marblestone REQ-KG-065)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct NeurotransmitterWeights {
    pub excitatory: f32,  // Glutamate-like: strengthens activation
    pub inhibitory: f32,  // GABA-like: suppresses activation
    pub modulatory: f32,  // Dopamine/Serotonin-like: adjusts sensitivity
}

impl Default for NeurotransmitterWeights {
    fn default() -> Self { Self { excitatory: 0.5, inhibitory: 0.5, modulatory: 0.0 } }
}

impl NeurotransmitterWeights {
    pub fn for_domain(domain: Domain) -> Self {
        match domain {
            Domain::Code     => Self { excitatory: 0.7, inhibitory: 0.3, modulatory: 0.2 },
            Domain::Legal    => Self { excitatory: 0.5, inhibitory: 0.4, modulatory: 0.1 },
            Domain::Medical  => Self { excitatory: 0.6, inhibitory: 0.5, modulatory: 0.3 },
            Domain::Creative => Self { excitatory: 0.8, inhibitory: 0.2, modulatory: 0.5 },
            Domain::Research => Self { excitatory: 0.6, inhibitory: 0.3, modulatory: 0.4 },
            Domain::General  => Self::default(),
        }
    }

    /// Net activation: excitatory - inhibitory + (modulatory * 0.5)
    #[inline]
    pub fn net_activation(&self) -> f32 {
        self.excitatory - self.inhibitory + (self.modulatory * 0.5)
    }
}

/// Graph edge with Marblestone modulation (REQ-KG-040-044, REQ-KG-065)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: Uuid,
    pub source: Uuid,
    pub target: Uuid,
    pub edge_type: EdgeType,
    pub weight: f32,                              // [0, 1]
    pub confidence: f32,                          // [0, 1]
    pub domain: Domain,
    pub neurotransmitter_weights: NeurotransmitterWeights,
    pub is_amortized_shortcut: bool,
    pub steering_reward: f32,                     // [-1.0, 1.0]
    pub traversal_count: u32,
    pub created_at: DateTime<Utc>,
    pub last_traversed_at: Option<DateTime<Utc>>,
}

impl GraphEdge {
    pub fn new(source: Uuid, target: Uuid, edge_type: EdgeType, weight: f32, domain: Domain) -> Self {
        Self {
            id: Uuid::new_v4(), source, target, edge_type, weight: weight.clamp(0.0, 1.0),
            confidence: 0.5, domain, neurotransmitter_weights: NeurotransmitterWeights::for_domain(domain),
            is_amortized_shortcut: false, steering_reward: 0.0, traversal_count: 0,
            created_at: Utc::now(), last_traversed_at: None,
        }
    }

    /// Get modulated edge weight (Marblestone REQ-KG-065)
    /// Formula: effective = base * (1 + net_activation + domain_bonus) * (1 + steering * 0.2)
    #[inline]
    pub fn get_modulated_weight(&self, query_domain: &Domain) -> f32 {
        let net = self.neurotransmitter_weights.net_activation();
        let domain_bonus = if &self.domain == query_domain { 0.1 } else { 0.0 };
        let steering_factor = 1.0 + (self.steering_reward * 0.2);
        (self.weight * (1.0 + net + domain_bonus) * steering_factor).clamp(0.0, 1.0)
    }

    pub fn record_traversal(&mut self, reward: f32) {
        self.traversal_count = self.traversal_count.saturating_add(1);
        self.last_traversed_at = Some(Utc::now());
        self.steering_reward = (self.steering_reward * 0.9 + reward * 0.1).clamp(-1.0, 1.0);
    }
}
```

### 4.2 RocksDB Backend

```rust
// src/storage/rocksdb.rs

use crate::error::GraphError;
use crate::hyperbolic::PoincarePoint;
use crate::entailment::EntailmentCone;
use rocksdb::{Options, DB, ColumnFamilyDescriptor, WriteBatch};

pub const CF_ADJACENCY: &str = "adjacency";
pub const CF_HYPERBOLIC: &str = "hyperbolic";
pub const CF_CONES: &str = "entailment_cones";

pub struct GraphStorage { db: std::sync::Arc<DB> }

impl GraphStorage {
    pub fn open(path: &str, config: &StorageConfig) -> Result<Self, GraphError> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_write_buffer_size(config.write_buffer_size);

        let cfs = vec![
            ColumnFamilyDescriptor::new(CF_ADJACENCY, Options::default()),
            ColumnFamilyDescriptor::new(CF_HYPERBOLIC, Options::default()),
            ColumnFamilyDescriptor::new(CF_CONES, Options::default()),
        ];

        let db = DB::open_cf_descriptors(&opts, path, cfs)
            .map_err(|e| GraphError::StorageOpen(e.to_string()))?;
        Ok(Self { db: std::sync::Arc::new(db) })
    }

    pub fn get_hyperbolic(&self, node_id: u64) -> Result<Option<PoincarePoint>, GraphError> {
        let cf = self.db.cf_handle(CF_HYPERBOLIC).ok_or(GraphError::ColumnFamilyNotFound(CF_HYPERBOLIC.into()))?;
        match self.db.get_cf(&cf, node_id.to_le_bytes())? {
            Some(data) if data.len() == 256 => {
                let mut coords = [0.0f32; 64];
                for (i, chunk) in data.chunks(4).enumerate() { coords[i] = f32::from_le_bytes(chunk.try_into().unwrap()); }
                Ok(Some(PoincarePoint { coords }))
            }
            Some(_) => Err(GraphError::CorruptedData("hyperbolic point")),
            None => Ok(None),
        }
    }

    pub fn put_hyperbolic(&self, node_id: u64, point: &PoincarePoint) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_HYPERBOLIC).ok_or(GraphError::ColumnFamilyNotFound(CF_HYPERBOLIC.into()))?;
        let mut value = Vec::with_capacity(256);
        for coord in &point.coords { value.extend_from_slice(&coord.to_le_bytes()); }
        self.db.put_cf(&cf, node_id.to_le_bytes(), value)?;
        Ok(())
    }

    pub fn get_cone(&self, node_id: u64) -> Result<Option<EntailmentCone>, GraphError> {
        let cf = self.db.cf_handle(CF_CONES).ok_or(GraphError::ColumnFamilyNotFound(CF_CONES.into()))?;
        match self.db.get_cf(&cf, node_id.to_le_bytes())? {
            Some(data) if data.len() >= 268 => {
                let mut coords = [0.0f32; 64];
                for (i, chunk) in data[..256].chunks(4).enumerate() { coords[i] = f32::from_le_bytes(chunk.try_into().unwrap()); }
                Ok(Some(EntailmentCone {
                    apex: PoincarePoint { coords },
                    aperture: f32::from_le_bytes(data[256..260].try_into().unwrap()),
                    aperture_factor: f32::from_le_bytes(data[260..264].try_into().unwrap()),
                    depth: u32::from_le_bytes(data[264..268].try_into().unwrap()),
                }))
            }
            Some(_) => Err(GraphError::CorruptedData("entailment cone")),
            None => Ok(None),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub write_buffer_size: usize,  // 256MB default
    pub block_cache_size: usize,   // 8GB default
}

impl Default for StorageConfig {
    fn default() -> Self { Self { write_buffer_size: 256 * 1024 * 1024, block_cache_size: 8 * 1024 * 1024 * 1024 } }
}
```

---

## 5. Hyperbolic Geometry (64D Poincare Ball)

### 5.1 Poincare Point

```rust
// src/hyperbolic/poincare.rs

use crate::config::HyperbolicConfig;

/// 64-dimensional point in Poincare ball model (REQ-KG-050, REQ-KG-054)
/// Constraint: ||coords|| < 1.0
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct PoincarePoint { pub coords: [f32; 64] }

impl PoincarePoint {
    pub fn origin() -> Self { Self { coords: [0.0; 64] } }

    #[inline]
    pub fn norm_squared(&self) -> f32 { self.coords.iter().map(|x| x * x).sum() }

    #[inline]
    pub fn norm(&self) -> f32 { self.norm_squared().sqrt() }

    pub fn project(&mut self, config: &HyperbolicConfig) {
        let norm = self.norm();
        if norm >= config.max_norm {
            let scale = config.max_norm / norm;
            for x in &mut self.coords { *x *= scale; }
        }
    }
}
```

### 5.2 Mobius Operations

```rust
// src/hyperbolic/mobius.rs

use crate::config::HyperbolicConfig;
use crate::hyperbolic::PoincarePoint;

pub struct PoincareBall { pub config: HyperbolicConfig }

impl Default for PoincareBall {
    fn default() -> Self { Self { config: HyperbolicConfig::default() } }
}

impl PoincareBall {
    /// Mobius addition: x (+)_c y
    /// ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) / (1 + 2c<x,y> + c^2||x||^2||y||^2)
    pub fn mobius_add(&self, x: &PoincarePoint, y: &PoincarePoint) -> PoincarePoint {
        let c = -self.config.curvature;
        let x_sq = x.norm_squared();
        let y_sq = y.norm_squared();
        let xy: f32 = x.coords.iter().zip(y.coords.iter()).map(|(a, b)| a * b).sum();

        let num_x = 1.0 + 2.0 * c * xy + c * y_sq;
        let num_y = 1.0 - c * x_sq;
        let denom = 1.0 + 2.0 * c * xy + c * c * x_sq * y_sq + self.config.eps;

        let mut result = PoincarePoint::origin();
        for i in 0..64 { result.coords[i] = (num_x * x.coords[i] + num_y * y.coords[i]) / denom; }
        result.project(&self.config);
        result
    }

    /// Poincare distance: d(x, y) = (2/sqrt(c)) * arctanh(sqrt(c * ||x-y||^2 / ((1-c||x||^2)(1-c||y||^2))))
    /// Performance: <10us per pair
    #[inline]
    pub fn distance(&self, x: &PoincarePoint, y: &PoincarePoint) -> f32 {
        let c = -self.config.curvature;
        let x_sq = x.norm_squared();
        let y_sq = y.norm_squared();
        let diff_sq: f32 = x.coords.iter().zip(y.coords.iter()).map(|(a, b)| (a - b).powi(2)).sum();

        let denom = (1.0 - c * x_sq) * (1.0 - c * y_sq) + self.config.eps;
        let mob_norm = (c * diff_sq / denom).sqrt().min(1.0 - self.config.eps);
        (2.0 / c.sqrt()) * mob_norm.atanh()
    }

    /// Exponential map: tangent vector v at x -> point on manifold
    pub fn exp_map(&self, x: &PoincarePoint, v: &[f32; 64]) -> PoincarePoint {
        let c = -self.config.curvature;
        let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if v_norm < self.config.eps { return x.clone(); }

        let lambda_x = 2.0 / (1.0 - c * x.norm_squared());
        let scale = (c.sqrt() * lambda_x * v_norm / 2.0).tanh() / (c.sqrt() * v_norm);

        let mut tangent = PoincarePoint::origin();
        for i in 0..64 { tangent.coords[i] = scale * v[i]; }
        self.mobius_add(x, &tangent)
    }

    /// Logarithmic map: point y -> tangent vector at x
    pub fn log_map(&self, x: &PoincarePoint, y: &PoincarePoint) -> [f32; 64] {
        let c = -self.config.curvature;
        let mut neg_x = x.clone();
        for coord in &mut neg_x.coords { *coord = -*coord; }

        let diff = self.mobius_add(&neg_x, y);
        let diff_norm = diff.norm();
        if diff_norm < self.config.eps { return [0.0; 64]; }

        let lambda_x = 2.0 / (1.0 - c * x.norm_squared());
        let scale = (2.0 / (c.sqrt() * lambda_x)) * (c.sqrt() * diff_norm).atanh() / diff_norm;

        let mut result = [0.0; 64];
        for i in 0..64 { result[i] = scale * diff.coords[i]; }
        result
    }
}
```

---

## 6. Entailment Cones

```rust
// src/entailment/cones.rs

use crate::config::ConeConfig;
use crate::hyperbolic::{PoincareBall, PoincarePoint};

/// Entailment cone for O(1) hierarchy queries (REQ-KG-052, REQ-KG-053)
#[derive(Debug, Clone)]
pub struct EntailmentCone {
    pub apex: PoincarePoint,
    pub aperture: f32,         // Half-aperture in radians
    pub aperture_factor: f32,  // Learned adjustment
    pub depth: u32,            // 0 = root
}

impl EntailmentCone {
    pub fn new(apex: PoincarePoint, depth: u32, config: &ConeConfig) -> Self {
        let aperture = (config.base_aperture * config.aperture_decay.powi(depth as i32))
            .clamp(config.min_aperture, config.max_aperture);
        Self { apex, aperture, aperture_factor: 1.0, depth }
    }

    #[inline]
    pub fn effective_aperture(&self) -> f32 { self.aperture * self.aperture_factor }

    /// Hard membership check. Performance: <50us
    pub fn contains(&self, ball: &PoincareBall, point: &PoincarePoint) -> bool {
        self.membership_score(ball, point) >= 0.99
    }

    /// Soft membership score [0, 1]
    /// Algorithm: compute angle from apex->point vs apex->origin, compare to aperture
    pub fn membership_score(&self, ball: &PoincareBall, point: &PoincarePoint) -> f32 {
        let apex_norm = self.apex.norm();
        if apex_norm < ball.config.eps { return 1.0; }

        let tangent = ball.log_map(&self.apex, point);
        let to_origin = ball.log_map(&self.apex, &PoincarePoint::origin());

        let t_norm: f32 = tangent.iter().map(|x| x * x).sum::<f32>().sqrt();
        let o_norm: f32 = to_origin.iter().map(|x| x * x).sum::<f32>().sqrt();
        if t_norm < ball.config.eps || o_norm < ball.config.eps { return 1.0; }

        let dot: f32 = tangent.iter().zip(to_origin.iter()).map(|(a, b)| a * b).sum();
        let angle = (dot / (t_norm * o_norm)).clamp(-1.0, 1.0).acos();
        let aperture = self.effective_aperture();

        if angle <= aperture { 1.0 } else { (-2.0 * (angle - aperture)).exp() }
    }

    /// Update aperture from training signal
    pub fn update_aperture(&mut self, target: bool, actual: f32, lr: f32) {
        let error = if target { 1.0 } else { 0.0 } - actual;
        self.aperture_factor = (self.aperture_factor + lr * error).clamp(0.5, 2.0);
    }
}
```

---

## 7. Graph Traversal

### 7.1 BFS Traversal

```rust
// src/traversal/bfs.rs

use crate::storage::{GraphStorage, EdgeType, Domain};
use std::collections::{HashSet, VecDeque};

#[derive(Debug)]
pub struct BfsResult {
    pub nodes: Vec<u64>,
    pub edges: Vec<(u64, u64, EdgeType)>,
    pub depth_counts: Vec<usize>,
}

pub struct BfsParams {
    pub max_depth: u32,
    pub max_nodes: usize,
    pub edge_types: Option<Vec<EdgeType>>,
    pub domain_filter: Option<Domain>,
}

impl Default for BfsParams {
    fn default() -> Self { Self { max_depth: 6, max_nodes: 10000, edge_types: None, domain_filter: None } }
}

pub fn bfs_traverse(storage: &GraphStorage, start: u64, params: &BfsParams) -> Result<BfsResult, crate::error::GraphError> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut result = BfsResult { nodes: Vec::new(), edges: Vec::new(), depth_counts: Vec::new() };

    queue.push_back((start, 0u32));
    visited.insert(start);

    while let Some((node_id, depth)) = queue.pop_front() {
        if result.nodes.len() >= params.max_nodes { break; }
        result.nodes.push(node_id);

        while result.depth_counts.len() <= depth as usize { result.depth_counts.push(0); }
        result.depth_counts[depth as usize] += 1;

        if depth >= params.max_depth { continue; }

        if let Some(adj) = storage.get_adjacency(node_id)? {
            for edge in adj.all_outgoing() {
                if let Some(ref types) = params.edge_types {
                    if !types.contains(&edge.get_edge_type()) { continue; }
                }
                if !visited.contains(&edge.other_id) {
                    visited.insert(edge.other_id);
                    queue.push_back((edge.other_id, depth + 1));
                    result.edges.push((node_id, edge.other_id, edge.get_edge_type()));
                }
            }
        }
    }
    Ok(result)
}
```

### 7.2 DFS Traversal

```rust
// src/traversal/dfs.rs

use crate::storage::{GraphStorage, EdgeType};
use std::collections::HashSet;

pub fn dfs_traverse(storage: &GraphStorage, start: u64, max_depth: u32, max_nodes: usize) -> Result<Vec<u64>, crate::error::GraphError> {
    let mut visited = HashSet::new();
    let mut stack = vec![(start, 0u32)];
    let mut result = Vec::new();

    while let Some((node_id, depth)) = stack.pop() {
        if visited.contains(&node_id) || result.len() >= max_nodes { continue; }
        visited.insert(node_id);
        result.push(node_id);

        if depth >= max_depth { continue; }

        if let Some(adj) = storage.get_adjacency(node_id)? {
            for edge in adj.all_outgoing() {
                if !visited.contains(&edge.other_id) {
                    stack.push((edge.other_id, depth + 1));
                }
            }
        }
    }
    Ok(result)
}
```

---

## 8. Marblestone Integration (REQ-KG-065)

```rust
// src/marblestone/domain_search.rs

use crate::config::IndexConfig;
use crate::index::gpu_index::{GpuIndexManager, SearchResult};
use crate::storage::edges::{Domain, NeurotransmitterWeights};
use crate::error::GraphError;

#[derive(Debug)]
pub struct DomainSearchResult {
    pub id: i64,
    pub base_distance: f32,
    pub modulated_score: f32,
    pub nt_weights: NeurotransmitterWeights,
}

/// Domain-aware ANN search with neurotransmitter profiles (REQ-KG-065)
pub struct DomainAwareSearch {
    index: GpuIndexManager,
}

impl DomainAwareSearch {
    pub fn new(index: GpuIndexManager) -> Self { Self { index } }

    /// Domain-aware search. Performance: <10ms for k=10, 10M vectors
    /// Algorithm:
    /// 1. FAISS k-NN search (fetch 3x candidates)
    /// 2. Apply neurotransmitter modulation
    /// 3. Re-rank by modulated score
    pub async fn domain_aware_search(&self, query: &[f32], domain: Domain, k: usize) -> Result<Vec<DomainSearchResult>, GraphError> {
        let nt = NeurotransmitterWeights::for_domain(domain);
        let fetch_k = (k * 3).min(1000);
        let raw = self.index.search(query, fetch_k).await?;

        let mut results: Vec<DomainSearchResult> = raw.query_results(0).map(|(id, dist)| {
            let sim = 1.0 / (1.0 + dist);
            let modulated = sim * (1.0 + nt.net_activation());
            DomainSearchResult { id, base_distance: dist, modulated_score: modulated, nt_weights: nt }
        }).collect();

        results.sort_by(|a, b| b.modulated_score.partial_cmp(&a.modulated_score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }
}
```

---

## 9. Error Handling

```rust
// src/error.rs

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("FAISS index creation failed")]
    FaissIndexCreation,
    #[error("FAISS training failed")]
    FaissTrainingFailed,
    #[error("FAISS search failed")]
    FaissSearchFailed,
    #[error("FAISS add failed")]
    FaissAddFailed,
    #[error("Index not trained")]
    IndexNotTrained,
    #[error("Insufficient training data: {provided} < {required}")]
    InsufficientTrainingData { provided: usize, required: usize },
    #[error("GPU resource allocation failed")]
    GpuResourceAllocation,
    #[error("GPU transfer failed")]
    GpuTransferFailed,
    #[error("Storage open failed: {0}")]
    StorageOpen(String),
    #[error("Storage error: {0}")]
    Storage(String),
    #[error("Column family not found: {0}")]
    ColumnFamilyNotFound(String),
    #[error("Corrupted data: {0}")]
    CorruptedData(&'static str),
    #[error("Vector ID mismatch")]
    VectorIdMismatch,
    #[error("Invalid configuration")]
    InvalidConfig,
}

impl From<rocksdb::Error> for GraphError {
    fn from(e: rocksdb::Error) -> Self { GraphError::Storage(e.to_string()) }
}
```

---

## 10. CUDA Kernels

### 10.1 Poincare Distance Kernel

```cuda
// kernels/poincare_distance.cu

#define DIM 64

__device__ float poincare_distance_device(const float* x, const float* y, float c) {
    float x_sq = 0, y_sq = 0, xy = 0;
    #pragma unroll
    for (int i = 0; i < DIM; i++) { x_sq += x[i]*x[i]; y_sq += y[i]*y[i]; xy += x[i]*y[i]; }

    float diff_sq = x_sq + y_sq - 2.0f * xy;
    float denom = (1.0f - c*x_sq) * (1.0f - c*y_sq) + 1e-7f;
    float mob = sqrtf(fminf(c * diff_sq / denom, 1.0f - 1e-5f));
    return (2.0f / sqrtf(c)) * atanhf(mob);
}

/// Batch distance: <1ms for 1K x 1K
__global__ void poincare_distance_batch(const float* queries, const float* database,
    float* distances, int n_q, int n_db, float c) {
    int q = blockIdx.y, db = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_q || db >= n_db) return;

    __shared__ float sq[DIM];
    if (threadIdx.x < DIM) sq[threadIdx.x] = queries[q*DIM + threadIdx.x];
    __syncthreads();

    float pt[DIM];
    for (int i = 0; i < DIM; i++) pt[i] = database[db*DIM + i];
    distances[q*n_db + db] = poincare_distance_device(sq, pt, -c);
}
```

### 10.2 Cone Membership Kernel

```cuda
// kernels/cone_check.cu

#define DIM 64

/// Batch cone membership: <2ms for 1K x 1K
__global__ void cone_check_batch(const float* cones, const float* points,
    float* scores, int n_cones, int n_pts, float c) {
    int cone = blockIdx.y, pt = blockIdx.x * blockDim.x + threadIdx.x;
    if (cone >= n_cones || pt >= n_pts) return;

    __shared__ float sc[65]; // apex[64] + aperture
    if (threadIdx.x < 65) sc[threadIdx.x] = cones[cone*65 + threadIdx.x];
    __syncthreads();

    float p[DIM];
    for (int i = 0; i < DIM; i++) p[i] = points[pt*DIM + i];

    // Simplified cone membership (full impl in CPU)
    float apex_norm = 0, pt_norm = 0, dot = 0;
    for (int i = 0; i < DIM; i++) { apex_norm += sc[i]*sc[i]; pt_norm += p[i]*p[i]; dot += sc[i]*p[i]; }

    float angle = acosf(fminf(dot / (sqrtf(apex_norm)*sqrtf(pt_norm) + 1e-7f), 1.0f));
    scores[cone*n_pts + pt] = angle <= sc[64] ? 1.0f : expf(-2.0f * (angle - sc[64]));
}
```

---

## 11. Performance Targets

| Operation | Target | Conditions |
|-----------|--------|------------|
| FAISS k=10 search | <5ms | nprobe=128, 10M vectors |
| Hyperbolic distance | <10us | Single pair, CPU |
| Batch distance 1Kx1K | <1ms | GPU |
| Cone membership | <50us | Single check, CPU |
| Batch cone 1Kx1K | <2ms | GPU |
| BFS depth=6 | <100ms | 10M nodes |
| Domain-aware search | <10ms | k=10, 10M vectors |

### Memory Budget

| Component | Budget |
|-----------|--------|
| FAISS GPU index | 8GB |
| Hyperbolic coords (10M) | 2.5GB |
| Entailment cones (10M) | 2.7GB |
| RocksDB cache | 8GB |
| **Total VRAM** | **24GB/32GB RTX 5090** |

---

## 12. Cargo.toml

```toml
[package]
name = "context-graph-graph"
version = "0.1.0"
edition = "2021"

[dependencies]
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"
half = "2.3"
rocksdb = { version = "0.21", features = ["multi-threaded-cf"] }
tokio = { version = "1.35", features = ["full"] }
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1.6", features = ["v4", "serde"] }
context-graph-core = { path = "../context-graph-core" }

[build-dependencies]
cc = "1.0"
cuda-builder = "0.3"
```

---

## 13. Summary

Module 4 provides:

1. **FAISS GPU Index**: IVF16384,PQ64x8, nprobe=128, <5ms k=10 search on RTX 5090
2. **Graph Storage**: RocksDB adjacency lists, 4 edge types (semantic/temporal/causal/hierarchical)
3. **Hyperbolic Geometry**: 64D Poincare ball, Mobius addition, exp/log maps, <10us distance
4. **Entailment Cones**: Learned apertures, hierarchy discovery, <50us membership
5. **Marblestone (REQ-KG-065)**: `get_modulated_weight()`, `domain_aware_search()`, `NeurotransmitterWeights`
6. **Traversal**: BFS, DFS with domain filtering
7. **CUDA Kernels**: GPU-accelerated distance and cone membership
