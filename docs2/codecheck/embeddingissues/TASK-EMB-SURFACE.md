# Surface Layer Task Specifications

<task_collection id="TASK-EMB-SURFACE" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Collection ID** | TASK-EMB-SURFACE |
| **Title** | Surface Layer Task Specifications |
| **Status** | Ready |
| **Version** | 1.0 |
| **Layer** | Surface (API Integration, Handlers, MCP Endpoints) |
| **Task Count** | 5 (TASK-EMB-021 through TASK-EMB-025) |
| **Implements** | REQ-EMB-001, REQ-EMB-006, REQ-EMB-007 |
| **Related Tech Specs** | TECH-EMB-001, TECH-EMB-004 |
| **Depends On** | TASK-EMB-FOUNDATION (complete), TASK-EMB-LOGIC (complete) |
| **Created** | 2026-01-06 |
| **Constitution Reference** | v4.0.0 |

---

## Layer Execution Order

Surface Layer tasks depend on both Foundation and Logic Layers.

```
Foundation Layer (TASK-EMB-001 through TASK-EMB-010)
         |
         v  (ALL MUST COMPLETE)
Logic Layer (TASK-EMB-011 through TASK-EMB-020)
         |
         v  (ALL MUST COMPLETE)
Surface Layer (This Document)
         |
         v
Production Ready
```

---

## Surface Layer Purpose

The Surface Layer integrates all Foundation and Logic components into the public API:

1. **API Integration** - Wire projection/quantization/storage into SparseModel
2. **Handler Updates** - Update MCP handlers to use real embeddings
3. **End-to-End Tests** - Validate complete pipeline with real data

---

## Dependencies Graph

```
TASK-EMB-012 (ProjectionMatrix::project())
      |
      +---> TASK-EMB-021 (Integrate into SparseModel)
                  |
                  +---> TASK-EMB-024 (Update MCP Handlers)
                              |
                              +---> TASK-EMB-025 (Integration Tests)

TASK-EMB-020 (QuantizationRouter)
      |
      +---> TASK-EMB-022 (Storage Backend)
                  |
                  +---> TASK-EMB-023 (Multi-Space Search)
                              |
                              +---> TASK-EMB-024 (Update MCP Handlers)
```

---

## Task Specifications

---

<task_spec id="TASK-EMB-021" version="1.0">

### TASK-EMB-021: Integrate ProjectionMatrix into SparseModel

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Integrate Learned Projection into SparseModel |
| **Status** | ready |
| **Layer** | surface |
| **Sequence** | 21 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-012 |
| **Estimated Complexity** | medium |

#### Context

TECH-EMB-001 specifies that the SparseModel must use the learned ProjectionMatrix instead of the broken hash-based projection. This task integrates the Foundation and Logic Layer work into the actual SparseModel API.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Current SparseModel | `crates/context-graph-embeddings/src/models/pretrained/sparse/model.rs` |
| ProjectionMatrix | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |

#### Prerequisites

- [ ] TASK-EMB-012 completed (ProjectionMatrix::project() works)
- [ ] TASK-EMB-008 completed (SparseVector updated)

#### Scope

**In Scope:**
- Add `projection: Option<ProjectionMatrix>` to SparseModel
- Load projection matrix during `SparseModel::load()`
- Replace `sparse.to_dense_projected()` with `projection.project(&sparse)`
- Update output dimension from 768 to 1536

**Out of Scope:**
- Storage integration (TASK-EMB-022)
- MCP handler updates (TASK-EMB-024)

#### Definition of Done

**Modifications to SparseModel:**

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/model.rs

use super::projection::ProjectionMatrix;
use super::types::SPARSE_PROJECTED_DIMENSION;

pub struct SparseModel {
    // ... existing fields ...

    /// Learned projection matrix for sparse-to-dense conversion.
    /// CRITICAL: This replaces the broken hash-based projection.
    pub(crate) projection: Option<ProjectionMatrix>,
}

impl SparseModel {
    /// Load model weights into memory.
    ///
    /// # CRITICAL CHANGE
    /// Now also loads the projection matrix. System will PANIC if
    /// projection weights are missing (no fallback to hash).
    pub async fn load(&self) -> EmbeddingResult<()> {
        // ... existing BERT and MLM loading ...

        // Load projection matrix (REQUIRED - no fallback)
        let projection = ProjectionMatrix::load(&self.model_path)?;
        tracing::info!(
            "ProjectionMatrix loaded: shape [{}, {}], checksum {:?}",
            super::types::SPARSE_VOCAB_SIZE,
            SPARSE_PROJECTED_DIMENSION,
            hex::encode(&projection.checksum()[..8])
        );

        // Store projection in model state
        // (Update model state struct to hold projection)

        // ... rest of load ...
    }

    /// Embed input to dense vector (for multi-array storage compatibility).
    ///
    /// # CRITICAL CHANGE
    /// - Output dimension is now 1536 (was 768)
    /// - Uses learned neural projection (was hash modulo)
    pub async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
        // ... validation ...

        let start = std::time::Instant::now();

        // Get sparse vector
        let sparse = self.embed_sparse(input).await?;

        // Project using learned weights (NOT hash modulo!)
        let projection = self.get_projection()?;
        let vector = projection.project(&sparse)?;

        // Verify output dimension
        debug_assert_eq!(
            vector.len(),
            SPARSE_PROJECTED_DIMENSION,
            "Output dimension mismatch: expected {}, got {}",
            SPARSE_PROJECTED_DIMENSION,
            vector.len()
        );

        let latency_us = start.elapsed().as_micros() as u64;
        Ok(ModelEmbedding::new(self.model_id, vector, latency_us))
    }

    /// Get projection matrix, panic if not loaded.
    fn get_projection(&self) -> EmbeddingResult<&ProjectionMatrix> {
        self.projection.as_ref().ok_or_else(|| {
            EmbeddingError::NotInitialized {
                model_id: self.model_id,
            }
        })
    }
}
```

**Constraints:**
- Output dimension MUST be 1536 (not 768)
- Projection matrix MUST be loaded during load()
- NO fallback to hash-based projection
- PANIC if projection weights missing

**Verification:**
- `cargo test` passes
- Output dimension is 1536
- Semantic similarity preserved (>0.7 for related terms)

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/model.rs` | Add projection field, update load() and embed() |
| `crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs` | Re-export projection module |

#### Validation Criteria

- [ ] `SparseModel` has `projection: Option<ProjectionMatrix>` field
- [ ] `load()` loads projection matrix
- [ ] `embed()` returns 1536D vector
- [ ] No reference to `to_dense_projected()` in embed path
- [ ] System panics if projection file missing

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings --features cuda
cargo test -p context-graph-embeddings sparse::model -- --nocapture
```

</task_spec>

---

<task_spec id="TASK-EMB-022" version="1.0">

### TASK-EMB-022: Implement Storage Backend

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Implement FingerprintStorage Trait and RocksDB Backend |
| **Status** | ready |
| **Layer** | surface |
| **Sequence** | 22 |
| **Implements** | REQ-EMB-006 |
| **Depends On** | TASK-EMB-020 |
| **Estimated Complexity** | high |

#### Context

TECH-EMB-004 specifies the storage module for persisting quantized TeleologicalFingerprints. The current module is just a placeholder. This task implements the real storage backend.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Current placeholder | `crates/context-graph-embeddings/src/storage/mod.rs` |
| Storage types | `crates/context-graph-embeddings/src/storage/types.rs` |
| QuantizationRouter | `crates/context-graph-embeddings/src/quantization/router.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-004-storage-module.md` |

#### Prerequisites

- [ ] TASK-EMB-005 completed (StoredFingerprint struct exists)
- [ ] TASK-EMB-020 completed (QuantizationRouter works)

#### Scope

**In Scope:**
- Create `FingerprintStorage` trait
- Implement RocksDB backend for development
- Store each embedder's quantized vector separately
- Store purpose vector and Johari data
- Implement basic retrieval by UUID

**Out of Scope:**
- ScyllaDB backend for production (future task)
- HNSW index integration (TASK-EMB-023)
- Multi-embedder queries (TASK-EMB-023)

#### Definition of Done

**New Files:**

```rust
// File: crates/context-graph-embeddings/src/storage/backend.rs

use std::path::Path;
use uuid::Uuid;

use super::types::StoredFingerprint;
use crate::error::EmbeddingError;

/// Storage backend for TeleologicalFingerprints.
///
/// # Constitution Alignment
/// - DEV: RocksDB (local, embedded)
/// - PROD: ScyllaDB (distributed)
///
/// # Storage Layout
/// Each embedder's quantized embedding is stored in a separate column
/// for per-space indexing and lazy loading.
pub trait FingerprintStorage: Send + Sync {
    /// Store a complete fingerprint.
    ///
    /// All 13 embeddings are quantized per Constitution and stored.
    fn store(&self, fingerprint: &StoredFingerprint) -> Result<(), EmbeddingError>;

    /// Retrieve a fingerprint by UUID.
    fn retrieve(&self, id: Uuid) -> Result<Option<StoredFingerprint>, EmbeddingError>;

    /// Delete a fingerprint.
    fn delete(&self, id: Uuid) -> Result<bool, EmbeddingError>;

    /// Check if a fingerprint exists.
    fn exists(&self, id: Uuid) -> Result<bool, EmbeddingError>;

    /// Count total stored fingerprints.
    fn count(&self) -> Result<usize, EmbeddingError>;

    /// Retrieve only specific embedders (lazy loading).
    fn retrieve_embeddings(
        &self,
        id: Uuid,
        embedder_indices: &[u8],
    ) -> Result<Option<Vec<(u8, Vec<f32>)>>, EmbeddingError>;
}

// File: crates/context-graph-embeddings/src/storage/rocksdb.rs

use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use super::backend::FingerprintStorage;
use super::types::StoredFingerprint;
use crate::error::EmbeddingError;
use crate::quantization::router::QuantizationRouter;

/// RocksDB-based storage for development.
///
/// # Column Families
/// - `meta`: Fingerprint metadata (purpose, johari, timestamps)
/// - `emb_0` through `emb_12`: Per-embedder quantized vectors
/// - `purpose`: 13D purpose vectors for fast filtering
///
/// # Key Format
/// UUID bytes (16 bytes) as key, serialized data as value.
pub struct RocksDbStorage {
    /// RocksDB instance.
    db: Arc<DB>,
    /// Quantization router for encoding/decoding.
    quantizer: Arc<QuantizationRouter>,
}

impl RocksDbStorage {
    /// Open or create RocksDB storage at path.
    ///
    /// # Arguments
    /// * `path` - Directory for RocksDB files
    /// * `quantizer` - Quantization router for encoding embeddings
    pub fn open(path: &Path, quantizer: Arc<QuantizationRouter>) -> Result<Self, EmbeddingError> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Create column families for each embedder + meta
        let cf_names: Vec<&str> = (0..13)
            .map(|i| format!("emb_{}", i))
            .chain(["meta".to_string(), "purpose".to_string()])
            .collect::<Vec<_>>()
            .iter()
            .map(|s| s.as_str())
            .collect();

        let cfs: Vec<ColumnFamilyDescriptor> = cf_names
            .iter()
            .map(|name| ColumnFamilyDescriptor::new(*name, Options::default()))
            .collect();

        let db = DB::open_cf_descriptors(&opts, path, cfs).map_err(|e| {
            EmbeddingError::StorageCorruption {
                id: path.display().to_string(),
                reason: format!("RocksDB open failed: {}", e),
            }
        })?;

        Ok(Self {
            db: Arc::new(db),
            quantizer,
        })
    }
}

impl FingerprintStorage for RocksDbStorage {
    fn store(&self, fingerprint: &StoredFingerprint) -> Result<(), EmbeddingError> {
        let key = fingerprint.id.as_bytes();

        // Store metadata in meta column family
        let cf_meta = self.db.cf_handle("meta").ok_or_else(|| {
            EmbeddingError::StorageCorruption {
                id: fingerprint.id.to_string(),
                reason: "meta column family missing".to_string(),
            }
        })?;

        let meta_bytes = bincode::serialize(&(
            fingerprint.version,
            fingerprint.purpose_vector,
            fingerprint.theta_to_north_star,
            fingerprint.johari_quadrants,
            fingerprint.dominant_quadrant,
            fingerprint.johari_confidence,
            fingerprint.content_hash,
            fingerprint.created_at_ms,
            fingerprint.last_updated_ms,
            fingerprint.access_count,
            fingerprint.deleted,
        )).map_err(|e| EmbeddingError::StorageCorruption {
            id: fingerprint.id.to_string(),
            reason: format!("Serialization failed: {}", e),
        })?;

        self.db.put_cf(cf_meta, key, &meta_bytes).map_err(|e| {
            EmbeddingError::StorageCorruption {
                id: fingerprint.id.to_string(),
                reason: format!("Write failed: {}", e),
            }
        })?;

        // Store each embedder's quantized embedding in its column family
        for (embedder_idx, quantized) in &fingerprint.embeddings {
            let cf_name = format!("emb_{}", embedder_idx);
            let cf = self.db.cf_handle(&cf_name).ok_or_else(|| {
                EmbeddingError::StorageCorruption {
                    id: fingerprint.id.to_string(),
                    reason: format!("{} column family missing", cf_name),
                }
            })?;

            let emb_bytes = bincode::serialize(&quantized).map_err(|e| {
                EmbeddingError::StorageCorruption {
                    id: fingerprint.id.to_string(),
                    reason: format!("Embedding serialization failed: {}", e),
                }
            })?;

            self.db.put_cf(cf, key, &emb_bytes).map_err(|e| {
                EmbeddingError::StorageCorruption {
                    id: fingerprint.id.to_string(),
                    reason: format!("Embedding write failed: {}", e),
                }
            })?;
        }

        // Store purpose vector for fast filtering
        let cf_purpose = self.db.cf_handle("purpose").ok_or_else(|| {
            EmbeddingError::StorageCorruption {
                id: fingerprint.id.to_string(),
                reason: "purpose column family missing".to_string(),
            }
        })?;

        let purpose_bytes: Vec<u8> = fingerprint.purpose_vector
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        self.db.put_cf(cf_purpose, key, &purpose_bytes).map_err(|e| {
            EmbeddingError::StorageCorruption {
                id: fingerprint.id.to_string(),
                reason: format!("Purpose write failed: {}", e),
            }
        })?;

        Ok(())
    }

    fn retrieve(&self, id: Uuid) -> Result<Option<StoredFingerprint>, EmbeddingError> {
        let key = id.as_bytes();

        // Get metadata
        let cf_meta = self.db.cf_handle("meta").ok_or_else(|| {
            EmbeddingError::StorageCorruption {
                id: id.to_string(),
                reason: "meta column family missing".to_string(),
            }
        })?;

        let meta_bytes = match self.db.get_cf(cf_meta, key) {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return Ok(None),
            Err(e) => return Err(EmbeddingError::StorageCorruption {
                id: id.to_string(),
                reason: format!("Read failed: {}", e),
            }),
        };

        // Deserialize and reconstruct
        // ... implementation continues ...

        todo!("Complete retrieval implementation")
    }

    fn delete(&self, id: Uuid) -> Result<bool, EmbeddingError> {
        // Delete from all column families
        todo!("Implement delete")
    }

    fn exists(&self, id: Uuid) -> Result<bool, EmbeddingError> {
        let key = id.as_bytes();
        let cf_meta = self.db.cf_handle("meta").ok_or_else(|| {
            EmbeddingError::StorageCorruption {
                id: id.to_string(),
                reason: "meta column family missing".to_string(),
            }
        })?;

        Ok(self.db.get_cf(cf_meta, key)
            .map_err(|e| EmbeddingError::StorageCorruption {
                id: id.to_string(),
                reason: format!("Exists check failed: {}", e),
            })?
            .is_some())
    }

    fn count(&self) -> Result<usize, EmbeddingError> {
        // Count entries in meta column family
        todo!("Implement count")
    }

    fn retrieve_embeddings(
        &self,
        id: Uuid,
        embedder_indices: &[u8],
    ) -> Result<Option<Vec<(u8, Vec<f32>)>>, EmbeddingError> {
        // Lazy load only requested embedders
        todo!("Implement lazy loading")
    }
}
```

**Constraints:**
- Each embedder's data in separate column family
- Quantization applied during store
- Dequantization on retrieve
- bincode for serialization
- UUID as key

**Verification:**
- Store/retrieve roundtrip works
- Each column family created
- Size is ~17KB per fingerprint

#### Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/storage/backend.rs` | FingerprintStorage trait |
| `crates/context-graph-embeddings/src/storage/rocksdb.rs` | RocksDB implementation |

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/storage/mod.rs` | Export backend and rocksdb modules |
| `crates/context-graph-embeddings/Cargo.toml` | Add rocksdb, bincode dependencies |

#### Validation Criteria

- [ ] `FingerprintStorage` trait defined
- [ ] RocksDB backend compiles
- [ ] 15 column families created (13 embedders + meta + purpose)
- [ ] Store/retrieve roundtrip preserves data

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings storage::rocksdb -- --nocapture
```

</task_spec>

---

<task_spec id="TASK-EMB-023" version="1.0">

### TASK-EMB-023: Implement Multi-Space Search

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Implement Per-Embedder HNSW Search with RRF Fusion |
| **Status** | ready |
| **Layer** | surface |
| **Sequence** | 23 |
| **Implements** | REQ-EMB-006 |
| **Depends On** | TASK-EMB-022 |
| **Estimated Complexity** | high |

#### Context

TECH-EMB-004 specifies 13 separate HNSW indexes (one per embedder) with RRF (Reciprocal Rank Fusion) for combining results. This enables purpose-weighted multi-space retrieval.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Storage backend | `crates/context-graph-embeddings/src/storage/backend.rs` |
| Storage types | `crates/context-graph-embeddings/src/storage/types.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-004-storage-module.md` |

#### Prerequisites

- [ ] TASK-EMB-022 completed (storage backend works)
- [ ] `instant-distance` or `hnsw` crate added for HNSW implementation

#### Scope

**In Scope:**
- Create 13 HNSW indexes (one per embedder)
- Implement per-embedder search
- Implement RRF score fusion
- Implement purpose-weighted retrieval
- Support search on subset of embedders

**Out of Scope:**
- ScyllaDB distributed index (future)
- Graph-based reranking (separate feature)

#### Definition of Done

**New File:**

```rust
// File: crates/context-graph-embeddings/src/storage/multi_space.rs

use instant_distance::{Builder, Hnsw, Search};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

use super::backend::FingerprintStorage;
use super::types::{EmbedderQueryResult, MultiSpaceQueryResult, StoredFingerprint};
use crate::error::EmbeddingError;
use crate::types::model_id::ModelId;

/// RRF (Reciprocal Rank Fusion) constant.
/// Higher values give more weight to top results.
const RRF_K: f32 = 60.0;

/// Multi-space search engine with per-embedder HNSW indexes.
///
/// # Architecture
/// - 13 separate HNSW indexes (one per embedder)
/// - Cosine similarity metric
/// - RRF fusion for combining results
/// - Purpose-weighted filtering
///
/// # Constitution Alignment
/// Each embedder has its own searchable space per REQ-EMB-006.
pub struct MultiSpaceSearch {
    /// Per-embedder HNSW indexes.
    /// Key: embedder index (0-12)
    /// Value: HNSW index with UUID points
    indexes: HashMap<u8, RwLock<EmbedderIndex>>,

    /// Storage backend for fingerprint retrieval.
    storage: Arc<dyn FingerprintStorage>,

    /// UUID to internal ID mapping.
    uuid_to_id: RwLock<HashMap<Uuid, usize>>,

    /// Internal ID to UUID mapping.
    id_to_uuid: RwLock<Vec<Uuid>>,
}

/// Per-embedder HNSW index.
struct EmbedderIndex {
    /// HNSW graph structure.
    hnsw: Hnsw<Point>,
    /// Vector dimension.
    dimension: usize,
    /// Embedder index (0-12).
    embedder_idx: u8,
}

/// Point in HNSW space.
#[derive(Clone)]
struct Point {
    /// Vector data.
    vector: Vec<f32>,
    /// Internal ID.
    id: usize,
}

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance = 1 - cosine_similarity
        1.0 - cosine_similarity(&self.vector, &other.vector)
    }
}

impl MultiSpaceSearch {
    /// Create new multi-space search engine.
    pub fn new(storage: Arc<dyn FingerprintStorage>) -> Self {
        let indexes = (0u8..13)
            .map(|i| (i, RwLock::new(EmbedderIndex {
                hnsw: Builder::default().build([], []),
                dimension: ModelId::from_index(i as usize).unwrap().dimension(),
                embedder_idx: i,
            })))
            .collect();

        Self {
            indexes,
            storage,
            uuid_to_id: RwLock::new(HashMap::new()),
            id_to_uuid: RwLock::new(Vec::new()),
        }
    }

    /// Index a fingerprint into all 13 HNSW indexes.
    pub fn index(&self, fingerprint: &StoredFingerprint) -> Result<(), EmbeddingError> {
        // Assign internal ID
        let internal_id = {
            let mut uuid_map = self.uuid_to_id.write().unwrap();
            let mut id_vec = self.id_to_uuid.write().unwrap();

            if let Some(&existing) = uuid_map.get(&fingerprint.id) {
                existing
            } else {
                let new_id = id_vec.len();
                uuid_map.insert(fingerprint.id, new_id);
                id_vec.push(fingerprint.id);
                new_id
            }
        };

        // Add to each embedder's index
        for (&embedder_idx, quantized) in &fingerprint.embeddings {
            if let Some(index_lock) = self.indexes.get(&embedder_idx) {
                // Dequantize for indexing
                // Note: HNSW stores approximate vectors; exact retrieval from storage
                let vector = self.storage.retrieve_embeddings(
                    fingerprint.id,
                    &[embedder_idx],
                )?.map(|v| v.into_iter().next().map(|(_, vec)| vec))
                    .flatten()
                    .unwrap_or_default();

                if !vector.is_empty() {
                    // Rebuild index with new point (simplified - real impl uses incremental add)
                    // TODO: Use incremental HNSW insertion
                }
            }
        }

        Ok(())
    }

    /// Search a single embedder's space.
    pub fn search_embedder(
        &self,
        embedder_idx: u8,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<EmbedderQueryResult>, EmbeddingError> {
        let index_lock = self.indexes.get(&embedder_idx).ok_or_else(|| {
            EmbeddingError::DimensionMismatch {
                model_id: ModelId::from_index(embedder_idx as usize).unwrap_or(ModelId::Semantic),
                expected: 13,
                actual: embedder_idx as usize,
            }
        })?;

        let index = index_lock.read().unwrap();

        // Validate query dimension
        if query.len() != index.dimension {
            return Err(EmbeddingError::DimensionMismatch {
                model_id: ModelId::from_index(embedder_idx as usize).unwrap_or(ModelId::Semantic),
                expected: index.dimension,
                actual: query.len(),
            });
        }

        let query_point = Point { vector: query.to_vec(), id: usize::MAX };
        let mut search = Search::default();
        let results = index.hnsw.search(&query_point, &mut search);

        let id_map = self.id_to_uuid.read().unwrap();

        results
            .take(k)
            .enumerate()
            .map(|(rank, item)| {
                let uuid = id_map.get(item.point.id).copied().unwrap_or(Uuid::nil());
                Ok(EmbedderQueryResult {
                    id: uuid,
                    embedder_idx,
                    similarity: 1.0 - item.distance, // Convert distance back to similarity
                    distance: item.distance,
                    rank,
                })
            })
            .collect()
    }

    /// Multi-space search with RRF fusion.
    ///
    /// # Arguments
    /// * `queries` - Map of embedder_idx -> query vector
    /// * `weights` - Optional per-embedder weights (defaults to purpose vector)
    /// * `k` - Number of results per embedder
    /// * `final_k` - Number of final fused results
    pub fn search_multi_space(
        &self,
        queries: &HashMap<u8, Vec<f32>>,
        weights: Option<&[f32; 13]>,
        k: usize,
        final_k: usize,
    ) -> Result<Vec<MultiSpaceQueryResult>, EmbeddingError> {
        // Collect results from each embedder
        let mut all_results: HashMap<Uuid, Vec<EmbedderQueryResult>> = HashMap::new();

        for (&embedder_idx, query) in queries {
            let results = self.search_embedder(embedder_idx, query, k)?;
            for result in results {
                all_results
                    .entry(result.id)
                    .or_default()
                    .push(result);
            }
        }

        // Apply RRF fusion
        let mut fused_scores: Vec<(Uuid, f32, Vec<EmbedderQueryResult>)> = all_results
            .into_iter()
            .map(|(id, results)| {
                let rrf_score = compute_rrf_score(&results, weights);
                (id, rrf_score, results)
            })
            .collect();

        // Sort by RRF score descending
        fused_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Build final results
        fused_scores
            .into_iter()
            .take(final_k)
            .map(|(id, rrf_score, results)| {
                let mut embedder_similarities = [f32::NAN; 13];
                let mut weighted_sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for r in &results {
                    embedder_similarities[r.embedder_idx as usize] = r.similarity;
                    let w = weights.map(|w| w[r.embedder_idx as usize]).unwrap_or(1.0);
                    weighted_sum += r.similarity * w;
                    weight_sum += w;
                }

                let weighted_similarity = if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    0.0
                };

                Ok(MultiSpaceQueryResult {
                    id,
                    embedder_similarities,
                    rrf_score,
                    weighted_similarity,
                    purpose_alignment: 0.0, // Computed separately
                    embedder_count: results.len(),
                })
            })
            .collect()
    }
}

/// Compute RRF score for a set of per-embedder results.
fn compute_rrf_score(results: &[EmbedderQueryResult], weights: Option<&[f32; 13]>) -> f32 {
    results.iter().map(|r| {
        let w = weights.map(|w| w[r.embedder_idx as usize]).unwrap_or(1.0);
        w / (RRF_K + r.rank as f32 + 1.0)
    }).sum()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}
```

**Constraints:**
- 13 separate HNSW indexes
- Cosine similarity metric
- RRF fusion with k=60
- Optional purpose-weighted scoring

**Verification:**
- Per-embedder search works
- RRF fusion produces ranked results
- Purpose weights affect ranking

#### Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/storage/multi_space.rs` | Multi-space search |

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/storage/mod.rs` | Add `pub mod multi_space;` |
| `crates/context-graph-embeddings/Cargo.toml` | Add instant-distance dependency |

#### Validation Criteria

- [ ] 13 HNSW indexes created
- [ ] Per-embedder search returns ranked results
- [ ] RRF fusion combines results correctly
- [ ] Weights affect final ranking

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings storage::multi_space -- --nocapture
```

</task_spec>

---

<task_spec id="TASK-EMB-024" version="1.0">

### TASK-EMB-024: Update MCP Handlers

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Update MCP Handlers to Use Real Embeddings |
| **Status** | ready |
| **Layer** | surface |
| **Sequence** | 24 |
| **Implements** | All REQs |
| **Depends On** | TASK-EMB-021, TASK-EMB-023 |
| **Estimated Complexity** | medium |

#### Context

The MCP handlers need to be updated to use the real embedding pipeline instead of any stub references. This includes multi-embedding search, purpose handlers, and Johari handlers.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Multi-embedding search | `crates/context-graph-mcp/src/handlers/multi_embedding_search.rs` |
| Purpose handlers | `crates/context-graph-mcp/src/handlers/purpose.rs` |
| Johari handlers | `crates/context-graph-mcp/src/handlers/johari.rs` |
| Meta-UTL handlers | `crates/context-graph-mcp/src/handlers/meta_utl.rs` |

#### Prerequisites

- [ ] TASK-EMB-021 completed (SparseModel uses projection)
- [ ] TASK-EMB-023 completed (multi-space search works)

#### Scope

**In Scope:**
- Update handlers to use real embedding models
- Remove any stub/mock data references
- Wire in multi-space search
- Ensure output dimensions match Constitution

**Out of Scope:**
- Adding new handlers (separate task)
- Performance optimization (REQ-EMB-007 separate)

#### Definition of Done

**Handler Update Pattern:**

```rust
// BEFORE (with stub):
let embeddings = create_stub_embeddings();

// AFTER (real):
let embeddings = embedding_service.embed_all(&input).await?;
```

**Key Updates:**

1. **multi_embedding_search.rs:**
   - Use `MultiSpaceSearch::search_multi_space()` for queries
   - Return real similarity scores from HNSW
   - Support embedder selection in query

2. **purpose.rs:**
   - Compute purpose vectors from real 13-embedding fingerprint
   - Store/retrieve from `FingerprintStorage`

3. **johari.rs:**
   - Compute Johari quadrants from real embeddings
   - Store quadrant data in fingerprint

4. **meta_utl.rs:**
   - Use real UTL alignment calculations
   - No stub theta values

**Constraints:**
- NO stub or mock data in production paths
- All dimensions match Constitution
- Real GPU inference for embeddings

**Verification:**
- MCP handlers return real embedding data
- Dimensions correct (1536 for E6/E13)
- Multi-space search works end-to-end

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/multi_embedding_search.rs` | Use real multi-space search |
| `crates/context-graph-mcp/src/handlers/purpose.rs` | Use real embeddings |
| `crates/context-graph-mcp/src/handlers/johari.rs` | Use real embeddings |
| `crates/context-graph-mcp/src/handlers/meta_utl.rs` | Use real UTL |

#### Validation Criteria

- [ ] No `stub` or `mock` in handler code
- [ ] Multi-embedding search uses HNSW
- [ ] Purpose handlers compute from real vectors
- [ ] Johari handlers use real alignment

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-mcp
cargo test -p context-graph-mcp handlers -- --nocapture
grep -rn "stub\|mock" crates/context-graph-mcp/src/handlers/
```

</task_spec>

---

<task_spec id="TASK-EMB-025" version="1.0">

### TASK-EMB-025: Integration Tests

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | End-to-End Integration Tests with Real Data |
| **Status** | ready |
| **Layer** | surface |
| **Sequence** | 25 |
| **Implements** | REQ-EMB-007 |
| **Depends On** | All previous tasks |
| **Estimated Complexity** | high |

#### Context

The final task creates comprehensive integration tests that verify the entire embedding pipeline works correctly with real data, real GPU, and real storage.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Test fixtures | `crates/context-graph-embeddings/tests/fixtures/` |
| Golden references | `crates/context-graph-embeddings/tests/fixtures/golden/` |
| Existing tests | `crates/context-graph-embeddings/tests/` |

#### Prerequisites

- [ ] All TASK-EMB-001 through TASK-EMB-024 completed
- [ ] Test fixtures with real data
- [ ] GPU available for tests

#### Scope

**In Scope:**
- End-to-end embedding pipeline tests
- Storage roundtrip tests
- Multi-space search tests
- Performance benchmarks
- Dimension validation across all 13 embedders

**Out of Scope:**
- Chaos/fault injection tests (separate suite)
- Load testing (separate infrastructure)

#### Definition of Done

**New Test Files:**

```rust
// File: crates/context-graph-embeddings/tests/integration/pipeline_test.rs

//! End-to-end integration tests for the embedding pipeline.
//!
//! # Requirements
//! - CUDA-capable GPU
//! - Model weights in tests/fixtures/models/
//! - Golden references in tests/fixtures/golden/

use context_graph_embeddings::models::SparseModel;
use context_graph_embeddings::storage::{FingerprintStorage, RocksDbStorage, MultiSpaceSearch};
use context_graph_embeddings::quantization::QuantizationRouter;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

/// Test: E6 Sparse produces 1536D output (not 768D).
#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_sparse_dimension_is_1536() {
    let model_path = Path::new("tests/fixtures/models/sparse");
    let model = SparseModel::new(model_path, Default::default()).unwrap();
    model.load().await.unwrap();

    let input = ModelInput::Text { content: "machine learning".to_string() };
    let embedding = model.embed(&input).await.unwrap();

    assert_eq!(embedding.vector().len(), 1536, "E6 dimension should be 1536, not 768");
}

/// Test: Sparse projection preserves semantic similarity.
#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_semantic_similarity_preserved() {
    let model_path = Path::new("tests/fixtures/models/sparse");
    let model = SparseModel::new(model_path, Default::default()).unwrap();
    model.load().await.unwrap();

    let ml_input = ModelInput::Text { content: "machine learning algorithms".to_string() };
    let dl_input = ModelInput::Text { content: "deep learning neural networks".to_string() };
    let car_input = ModelInput::Text { content: "automobile vehicle transportation".to_string() };

    let ml_emb = model.embed(&ml_input).await.unwrap();
    let dl_emb = model.embed(&dl_input).await.unwrap();
    let car_emb = model.embed(&car_input).await.unwrap();

    let ml_dl_sim = cosine_similarity(ml_emb.vector(), dl_emb.vector());
    let ml_car_sim = cosine_similarity(ml_emb.vector(), car_emb.vector());

    assert!(ml_dl_sim > 0.7, "Related terms should have similarity > 0.7, got {}", ml_dl_sim);
    assert!(ml_car_sim < 0.5, "Unrelated terms should have similarity < 0.5, got {}", ml_car_sim);
}

/// Test: Storage roundtrip preserves all embeddings.
#[tokio::test]
async fn test_storage_roundtrip() {
    let temp_dir = tempfile::tempdir().unwrap();
    let codebook_dir = Path::new("tests/fixtures/codebooks");
    let quantizer = Arc::new(QuantizationRouter::new(codebook_dir).unwrap());
    let storage = Arc::new(RocksDbStorage::open(temp_dir.path(), quantizer).unwrap());

    let fingerprint = create_test_fingerprint();
    let id = fingerprint.id;

    storage.store(&fingerprint).unwrap();
    let retrieved = storage.retrieve(id).unwrap().expect("Fingerprint should exist");

    assert_eq!(fingerprint.id, retrieved.id);
    assert_eq!(fingerprint.purpose_vector, retrieved.purpose_vector);
    assert_eq!(fingerprint.johari_quadrants, retrieved.johari_quadrants);
    assert_eq!(fingerprint.embeddings.len(), retrieved.embeddings.len());
}

/// Test: Multi-space search returns ranked results.
#[tokio::test]
async fn test_multi_space_search() {
    let temp_dir = tempfile::tempdir().unwrap();
    let codebook_dir = Path::new("tests/fixtures/codebooks");
    let quantizer = Arc::new(QuantizationRouter::new(codebook_dir).unwrap());
    let storage = Arc::new(RocksDbStorage::open(temp_dir.path(), quantizer).unwrap());
    let search = MultiSpaceSearch::new(storage.clone());

    // Index some test fingerprints
    for i in 0..10 {
        let fp = create_test_fingerprint_with_id(Uuid::new_v4());
        storage.store(&fp).unwrap();
        search.index(&fp).unwrap();
    }

    // Search single embedder
    let query = vec![0.1; 1024]; // E1 dimension
    let results = search.search_embedder(0, &query, 5).unwrap();
    assert_eq!(results.len(), 5);

    // Verify results are ranked by similarity
    for i in 1..results.len() {
        assert!(results[i-1].similarity >= results[i].similarity);
    }
}

/// Test: All 13 embedders produce correct dimensions.
#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_all_embedder_dimensions() {
    let expected_dimensions = [
        (ModelId::Semantic, 1024),
        (ModelId::TemporalRecent, 512),
        (ModelId::TemporalPeriodic, 512),
        (ModelId::TemporalPositional, 512),
        (ModelId::Causal, 768),
        (ModelId::Sparse, 1536),
        (ModelId::Code, 1536),
        (ModelId::Graph, 384),
        (ModelId::Hdc, 1024),
        (ModelId::Multimodal, 768),
        (ModelId::Entity, 384),
        (ModelId::LateInteraction, 128), // per token
        (ModelId::Splade, 1536),
    ];

    for (model_id, expected_dim) in expected_dimensions {
        let actual_dim = model_id.dimension();
        assert_eq!(
            actual_dim, expected_dim,
            "{:?} dimension mismatch: expected {}, got {}",
            model_id, expected_dim, actual_dim
        );
    }
}

/// Test: No stub or simulated data in output.
#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_no_fake_data() {
    let model_path = Path::new("tests/fixtures/models/sparse");
    let model = SparseModel::new(model_path, Default::default()).unwrap();
    model.load().await.unwrap();

    let input = ModelInput::Text { content: "test input".to_string() };
    let embedding = model.embed(&input).await.unwrap();

    // Check for sin wave pattern (old fake output)
    let is_sin_wave = embedding.vector().iter().enumerate().all(|(i, &v)| {
        (v - (i as f32 * 0.001).sin()).abs() < 0.01
    });
    assert!(!is_sin_wave, "Output should NOT be sin wave (indicates fake data)");

    // Check for all zeros (another fake pattern)
    let is_all_zeros = embedding.vector().iter().all(|&v| v.abs() < 1e-10);
    assert!(!is_all_zeros, "Output should NOT be all zeros");
}

/// Benchmark: Single embedding latency.
#[tokio::test]
#[cfg(feature = "cuda")]
async fn bench_single_embed_latency() {
    let model_path = Path::new("tests/fixtures/models/sparse");
    let model = SparseModel::new(model_path, Default::default()).unwrap();
    model.load().await.unwrap();

    let input = ModelInput::Text { content: "benchmark input text".to_string() };

    // Warmup
    for _ in 0..10 {
        let _ = model.embed(&input).await.unwrap();
    }

    // Measure
    let start = std::time::Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        let _ = model.embed(&input).await.unwrap();
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() / iterations;

    println!("Average single_embed latency: {} us", avg_us);
    assert!(avg_us < 10_000, "single_embed should be < 10ms, got {} us", avg_us);
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

fn create_test_fingerprint() -> StoredFingerprint {
    create_test_fingerprint_with_id(Uuid::new_v4())
}

fn create_test_fingerprint_with_id(id: Uuid) -> StoredFingerprint {
    // Create a test fingerprint with random embeddings
    todo!("Implement test fingerprint creation")
}
```

**Constraints:**
- Tests MUST use real GPU (gated by `#[cfg(feature = "cuda")]`)
- Tests MUST use real weight files from fixtures
- NO mock data allowed
- Performance tests document latency

**Verification:**
- All tests pass with real GPU
- Latencies within Constitutional bounds
- Dimensions match across all embedders

#### Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/tests/integration/mod.rs` | Module declaration |
| `crates/context-graph-embeddings/tests/integration/pipeline_test.rs` | Pipeline tests |
| `crates/context-graph-embeddings/tests/integration/storage_test.rs` | Storage tests |
| `crates/context-graph-embeddings/tests/integration/search_test.rs` | Search tests |

#### Validation Criteria

- [ ] All integration tests pass
- [ ] Tests use real GPU when available
- [ ] No mock/stub data in tests
- [ ] Latencies within budget (< 10ms single embed)
- [ ] Dimensions correct for all 13 embedders

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo test -p context-graph-embeddings --test integration --features cuda -- --nocapture
cargo bench -p context-graph-embeddings
```

</task_spec>

---

</task_collection>

---

## Summary

### Task List

| Task ID | Title | Dependencies | Complexity |
|---------|-------|--------------|------------|
| TASK-EMB-021 | Integrate ProjectionMatrix into SparseModel | TASK-EMB-012 | medium |
| TASK-EMB-022 | Implement Storage Backend | TASK-EMB-020 | high |
| TASK-EMB-023 | Implement Multi-Space Search | TASK-EMB-022 | high |
| TASK-EMB-024 | Update MCP Handlers | TASK-EMB-021, TASK-EMB-023 | medium |
| TASK-EMB-025 | Integration Tests | All previous | high |

### Execution Order

**Critical Path:**
```
Logic Layer Tasks (COMPLETE)
         |
         v
TASK-EMB-021 (SparseModel Integration)
         |
         +---> TASK-EMB-024 (MCP Handlers)
         |
TASK-EMB-022 (Storage Backend)
         |
         +---> TASK-EMB-023 (Multi-Space Search)
                    |
                    +---> TASK-EMB-024 (MCP Handlers)
                              |
                              +---> TASK-EMB-025 (Integration Tests)
```

**Parallel Tracks:**
- Track A: TASK-EMB-021 (SparseModel)
- Track B: TASK-EMB-022 -> TASK-EMB-023 (Storage + Search)

After both converge:
- TASK-EMB-024 (MCP Handlers)
- TASK-EMB-025 (Integration Tests)

### Files Summary

| File | Tasks |
|------|-------|
| `models/pretrained/sparse/model.rs` | TASK-EMB-021 |
| `storage/backend.rs` | TASK-EMB-022 |
| `storage/rocksdb.rs` | TASK-EMB-022 |
| `storage/multi_space.rs` | TASK-EMB-023 |
| MCP handlers | TASK-EMB-024 |
| `tests/integration/` | TASK-EMB-025 |

---

## Next Steps

After Surface Layer completion:

1. **Production Deployment** - Deploy with real GPU
2. **Performance Tuning** - Optimize based on benchmarks
3. **Monitoring** - Add observability for embedding operations
4. **Documentation** - Update API docs with new behavior

---

## Memory Key

Store this summary for reference:
```
contextgraph/embedding-issues/task-surface-summary
```
