# TASK-EMB-022: Implement Storage Backend

<task_spec id="TASK-EMB-022" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-022 |
| **Title** | Implement FingerprintStorage Trait and RocksDB Backend |
| **Status** | ready |
| **Layer** | surface |
| **Sequence** | 22 |
| **Implements** | REQ-EMB-006 (Storage Module Implementation) |
| **Depends On** | TASK-EMB-005 (Storage Types), TASK-EMB-020 (QuantizationRouter) |
| **Estimated Complexity** | high |
| **Created** | 2026-01-06 |
| **Constitution Reference** | v4.0.0 |

---

## Context

TECH-EMB-004 specifies the storage module for persisting quantized TeleologicalFingerprints. The current module is just a placeholder. This task implements the real storage backend with:

- Per-embedder column families for lazy loading
- Quantization applied during store
- Dequantization on retrieve
- RocksDB for development, ScyllaDB pattern for production

**Constitution Alignment:**
- `storage.layer1_primary: { dev: rocksdb, prod: scylladb }`
- Each of 13 embedders stored in separate column family
- Quantization per embedder per `embeddings.quantization`

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Current placeholder | `crates/context-graph-embeddings/src/storage/mod.rs` |
| Storage types | `crates/context-graph-embeddings/src/storage/types.rs` |
| QuantizationRouter | `crates/context-graph-embeddings/src/quantization/router.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-004-storage-module.md` |
| Constitution | `docs2/constitution.yaml` (storage section) |

---

## Prerequisites

- [ ] TASK-EMB-005 completed (StoredFingerprint struct exists)
- [ ] TASK-EMB-020 completed (QuantizationRouter works)
- [ ] `rocksdb` crate added to Cargo.toml
- [ ] `bincode` crate added for serialization

---

## Scope

### In Scope

- Create `FingerprintStorage` trait with store/retrieve/delete/exists/count
- Implement RocksDB backend with 15 column families (13 embedders + meta + purpose)
- Store each embedder's quantized vector separately
- Store purpose vector (13D) and Johari data in metadata
- Implement basic retrieval by UUID
- Implement lazy loading of specific embedders
- Quantization during store, dequantization during retrieve

### Out of Scope

- ScyllaDB backend for production (future task)
- HNSW index integration (TASK-EMB-023)
- Multi-embedder queries (TASK-EMB-023)
- Full text search

---

## Definition of Done

### New File: `backend.rs`

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
```

### New File: `rocksdb.rs`

```rust
// File: crates/context-graph-embeddings/src/storage/rocksdb.rs

use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use super::backend::FingerprintStorage;
use super::types::StoredFingerprint;
use crate::error::EmbeddingError;
use crate::quantization::router::QuantizationRouter;

/// Column family names for storage layout.
const CF_META: &str = "meta";
const CF_PURPOSE: &str = "purpose";

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
    ///
    /// # Column Families Created
    /// - `meta`: Metadata storage
    /// - `purpose`: 13D purpose vector storage
    /// - `emb_0` through `emb_12`: Per-embedder quantized vectors
    pub fn open(path: &Path, quantizer: Arc<QuantizationRouter>) -> Result<Self, EmbeddingError>;

    /// Get column family names (15 total).
    fn column_family_names() -> Vec<String>;
}

impl FingerprintStorage for RocksDbStorage {
    fn store(&self, fingerprint: &StoredFingerprint) -> Result<(), EmbeddingError>;
    fn retrieve(&self, id: Uuid) -> Result<Option<StoredFingerprint>, EmbeddingError>;
    fn delete(&self, id: Uuid) -> Result<bool, EmbeddingError>;
    fn exists(&self, id: Uuid) -> Result<bool, EmbeddingError>;
    fn count(&self) -> Result<usize, EmbeddingError>;
    fn retrieve_embeddings(
        &self,
        id: Uuid,
        embedder_indices: &[u8],
    ) -> Result<Option<Vec<(u8, Vec<f32>)>>, EmbeddingError>;
}
```

### Storage Schema

```
Column Families (15 total):
├── meta          # Metadata: version, johari, timestamps, content_hash, etc.
├── purpose       # 13D purpose vector (52 bytes, float32)
├── emb_0         # E1_Semantic quantized (PQ-8)
├── emb_1         # E2_Temporal_Recent quantized (Float8)
├── emb_2         # E3_Temporal_Periodic quantized (Float8)
├── emb_3         # E4_Temporal_Positional quantized (Float8)
├── emb_4         # E5_Causal quantized (PQ-8)
├── emb_5         # E6_Sparse quantized (native sparse)
├── emb_6         # E7_Code quantized (PQ-8)
├── emb_7         # E8_Graph quantized (Float8)
├── emb_8         # E9_HDC quantized (Binary)
├── emb_9         # E10_Multimodal quantized (PQ-8)
├── emb_10        # E11_Entity quantized (Float8)
├── emb_11        # E12_LateInteraction quantized (TokenPruning)
└── emb_12        # E13_SPLADE quantized (native sparse)

Key Format: UUID bytes (16 bytes)
Value Format: bincode serialized
```

### Constraints

- Each embedder's data in separate column family
- Quantization applied during store per Constitution
- Dequantization on retrieve
- bincode for serialization
- UUID as key (16 bytes)
- Total size target: ~17KB per fingerprint (63% reduction from 46KB)

### Verification

- Store/retrieve roundtrip works
- All 15 column families created
- Size is ~17KB per fingerprint (quantized)
- Lazy loading retrieves only requested embedders

---

## Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/storage/backend.rs` | FingerprintStorage trait |
| `crates/context-graph-embeddings/src/storage/rocksdb.rs` | RocksDB implementation |

---

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/storage/mod.rs` | Export backend and rocksdb modules |
| `crates/context-graph-embeddings/Cargo.toml` | Add rocksdb, bincode dependencies |

---

## Validation Criteria

- [ ] `FingerprintStorage` trait defined with all methods
- [ ] RocksDB backend compiles without errors
- [ ] 15 column families created (verify with RocksDB tools)
- [ ] Store/retrieve roundtrip preserves all data
- [ ] Quantization applied (verify compressed size < 20KB)
- [ ] Lazy loading works (retrieve_embeddings with subset)
- [ ] Error handling for missing fingerprints
- [ ] Error handling for corrupted data

---

## Test Commands

```bash
cd /home/cabdru/contextgraph

# Add dependencies
cargo add rocksdb bincode -p context-graph-embeddings

# Check compilation
cargo check -p context-graph-embeddings

# Run storage tests
cargo test -p context-graph-embeddings storage::rocksdb -- --nocapture

# Verify column families
# (requires manual inspection of RocksDB directory)
```

---

## Serialization Format

### Meta Column Family Value

```rust
#[derive(Serialize, Deserialize)]
struct StoredMeta {
    version: u8,
    purpose_vector: [f32; 13],
    theta_to_north_star: f32,
    johari_quadrants: [JohariQuadrant; 13],
    dominant_quadrant: JohariQuadrant,
    johari_confidence: [f32; 13],
    content_hash: [u8; 32],
    created_at_ms: i64,
    last_updated_ms: i64,
    access_count: u64,
    deleted: bool,
}
```

### Embedder Column Family Value

```rust
#[derive(Serialize, Deserialize)]
struct StoredEmbedding {
    quantization_type: QuantizationType,  // PQ8, Float8, Binary, Sparse
    data: Vec<u8>,                        // Quantized bytes
    original_dim: usize,                  // For validation
}
```

---

## Error Handling

| Error Case | Handling |
|------------|----------|
| DB open failure | `EmbeddingError::StorageCorruption` |
| Missing column family | `EmbeddingError::StorageCorruption` |
| Serialization failure | `EmbeddingError::StorageCorruption` |
| Write failure | `EmbeddingError::StorageCorruption` |
| Dequantization failure | `EmbeddingError::QuantizationFailed` |

---

## Memory Key

Store completion status:
```
contextgraph/embedding-issues/task-emb-022-complete
```

</task_spec>
