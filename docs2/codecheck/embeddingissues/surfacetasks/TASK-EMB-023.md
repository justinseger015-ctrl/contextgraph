# TASK-EMB-023: Implement Multi-Space Search

<task_spec id="TASK-EMB-023" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-023 |
| **Title** | Implement Per-Embedder HNSW Search with RRF Fusion |
| **Status** | ready |
| **Layer** | surface |
| **Sequence** | 23 |
| **Implements** | REQ-EMB-006 (Storage Module Implementation) |
| **Depends On** | TASK-EMB-022 (Storage Backend) |
| **Estimated Complexity** | high |
| **Created** | 2026-01-06 |
| **Constitution Reference** | v4.0.0 |

---

## Context

TECH-EMB-004 specifies 13 separate HNSW indexes (one per embedder) with RRF (Reciprocal Rank Fusion) for combining results. This enables purpose-weighted multi-space retrieval as described in the Constitution's 5-stage retrieval pipeline.

**Constitution Alignment:**
- `storage.layer2c_per_embedder`: "13× HNSW (E1-E13, one per embedder)"
- `embeddings.similarity.method`: "Reciprocal Rank Fusion (RRF) across per-space results"
- `embeddings.similarity.rrf_constant`: 60
- Stage 3 of 5-stage pipeline: "RRF fusion across 13 spaces"

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Storage backend | `crates/context-graph-embeddings/src/storage/backend.rs` |
| Storage types | `crates/context-graph-embeddings/src/storage/types.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-004-storage-module.md` |
| Constitution | `docs2/constitution.yaml` (embeddings.similarity, storage sections) |

---

## Prerequisites

- [ ] TASK-EMB-022 completed (storage backend works)
- [ ] `instant-distance` or similar HNSW crate added
- [ ] Storage backend can retrieve embeddings by ID
- [ ] Per-embedder dimensions defined in constants

---

## Scope

### In Scope

- Create 13 HNSW indexes (one per embedder)
- Implement per-embedder search (single space query)
- Implement RRF score fusion across multiple spaces
- Implement purpose-weighted retrieval
- Support search on subset of embedders
- UUID mapping for HNSW internal IDs
- Cosine similarity metric

### Out of Scope

- ScyllaDB distributed index (future)
- Graph-based reranking (separate feature)
- Full 5-stage pipeline (this is Stage 3 only)
- Stage 1 sparse prefilter (separate task)
- Stage 5 late interaction (separate task)

---

## Definition of Done

### New File: `multi_space.rs`

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
/// Constitution: embeddings.similarity.rrf_constant = 60
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
/// Each embedder has its own searchable space per storage.layer2c_per_embedder.
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

impl MultiSpaceSearch {
    /// Create new multi-space search engine.
    pub fn new(storage: Arc<dyn FingerprintStorage>) -> Self;

    /// Index a fingerprint into all 13 HNSW indexes.
    pub fn index(&self, fingerprint: &StoredFingerprint) -> Result<(), EmbeddingError>;

    /// Remove a fingerprint from all indexes.
    pub fn remove(&self, id: Uuid) -> Result<(), EmbeddingError>;

    /// Search a single embedder's space.
    pub fn search_embedder(
        &self,
        embedder_idx: u8,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<EmbedderQueryResult>, EmbeddingError>;

    /// Multi-space search with RRF fusion.
    ///
    /// # Arguments
    /// * `queries` - Map of embedder_idx -> query vector
    /// * `weights` - Optional per-embedder weights (defaults to 1.0)
    /// * `k` - Number of results per embedder
    /// * `final_k` - Number of final fused results
    ///
    /// # Formula (RRF)
    /// RRF(d) = Σᵢ wᵢ/(k + rankᵢ(d)) where k=60
    pub fn search_multi_space(
        &self,
        queries: &HashMap<u8, Vec<f32>>,
        weights: Option<&[f32; 13]>,
        k: usize,
        final_k: usize,
    ) -> Result<Vec<MultiSpaceQueryResult>, EmbeddingError>;

    /// Search with purpose vector weighting.
    ///
    /// Uses the query's purpose vector as weights for RRF fusion.
    pub fn search_purpose_weighted(
        &self,
        queries: &HashMap<u8, Vec<f32>>,
        purpose_vector: &[f32; 13],
        k: usize,
        final_k: usize,
    ) -> Result<Vec<MultiSpaceQueryResult>, EmbeddingError>;

    /// Rebuild indexes from storage (for recovery/startup).
    pub fn rebuild_from_storage(&self) -> Result<usize, EmbeddingError>;
}
```

### Result Types

```rust
/// Result from single embedder search.
#[derive(Debug, Clone)]
pub struct EmbedderQueryResult {
    /// Fingerprint UUID.
    pub id: Uuid,
    /// Which embedder (0-12).
    pub embedder_idx: u8,
    /// Cosine similarity [0, 1].
    pub similarity: f32,
    /// Cosine distance (1 - similarity).
    pub distance: f32,
    /// Rank in this embedder's results (0-based).
    pub rank: usize,
}

/// Result from multi-space RRF fusion.
#[derive(Debug, Clone)]
pub struct MultiSpaceQueryResult {
    /// Fingerprint UUID.
    pub id: Uuid,
    /// Per-embedder similarities (NaN if not queried).
    pub embedder_similarities: [f32; 13],
    /// RRF fusion score.
    pub rrf_score: f32,
    /// Weighted average similarity.
    pub weighted_similarity: f32,
    /// Alignment to query's purpose.
    pub purpose_alignment: f32,
    /// How many embedders contributed.
    pub embedder_count: usize,
}
```

### RRF Formula Implementation

```rust
/// Compute RRF score for a set of per-embedder results.
///
/// Formula: RRF(d) = Σᵢ wᵢ/(k + rankᵢ(d) + 1)
/// where k=60 (RRF_K constant)
fn compute_rrf_score(results: &[EmbedderQueryResult], weights: Option<&[f32; 13]>) -> f32 {
    results.iter().map(|r| {
        let w = weights.map(|w| w[r.embedder_idx as usize]).unwrap_or(1.0);
        w / (RRF_K + r.rank as f32 + 1.0)
    }).sum()
}

/// Cosine similarity between two vectors.
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

### HNSW Configuration

```rust
/// HNSW parameters per Constitution storage.layer2c_per_embedder.hnsw_params
const HNSW_M: usize = 16;              // Max connections per node
const HNSW_EF_CONSTRUCTION: usize = 200; // Build-time search width
const HNSW_EF_SEARCH: usize = 100;     // Query-time search width
```

### Constraints

- 13 separate HNSW indexes (one per embedder)
- Cosine similarity metric (distance = 1 - similarity)
- RRF fusion with k=60
- Optional purpose-weighted scoring
- Thread-safe with RwLock
- UUID ↔ internal ID mapping maintained

### Verification

- Per-embedder search returns ranked results
- RRF fusion produces correctly ordered results
- Purpose weights affect final ranking
- Index rebuild from storage works
- Concurrent access safe

---

## Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/storage/multi_space.rs` | Multi-space search engine |

---

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/storage/mod.rs` | Add `pub mod multi_space;` |
| `crates/context-graph-embeddings/src/storage/types.rs` | Add result types if not present |
| `crates/context-graph-embeddings/Cargo.toml` | Add instant-distance dependency |

---

## Validation Criteria

- [ ] 13 HNSW indexes created (one per embedder)
- [ ] Per-embedder search returns correctly ranked results
- [ ] RRF fusion combines results with correct formula
- [ ] Weights affect final ranking (verify with test cases)
- [ ] Cosine similarity calculated correctly
- [ ] UUID mapping works bidirectionally
- [ ] Index survives rebuild from storage
- [ ] Thread-safe concurrent access

---

## Test Commands

```bash
cd /home/cabdru/contextgraph

# Add HNSW dependency
cargo add instant-distance -p context-graph-embeddings

# Check compilation
cargo check -p context-graph-embeddings

# Run multi-space tests
cargo test -p context-graph-embeddings storage::multi_space -- --nocapture

# Benchmark search performance
cargo bench -p context-graph-embeddings multi_space
```

---

## Test Cases

### Unit Tests

```rust
#[test]
fn test_single_embedder_search() {
    // Create search engine
    // Index test fingerprints
    // Query single embedder
    // Verify results ranked by similarity
}

#[test]
fn test_rrf_fusion() {
    // Index fingerprints
    // Query multiple embedders
    // Verify RRF scores computed correctly
    // Verify final ranking matches expected
}

#[test]
fn test_purpose_weighted_search() {
    // Index fingerprints
    // Create purpose vector with varied weights
    // Query with purpose weighting
    // Verify weights affect final ranking
}

#[test]
fn test_index_rebuild() {
    // Index fingerprints
    // Clear indexes
    // Rebuild from storage
    // Verify same results as before
}
```

### Performance Targets

| Metric | Target | Constitution Ref |
|--------|--------|------------------|
| Per-embedder search (1K candidates) | < 10ms | Stage 3 latency |
| RRF fusion (13 spaces × 1K) | < 20ms | Stage 3 total |
| Index memory | < 100MB per 100K fingerprints | - |

---

## Memory Key

Store completion status:
```
contextgraph/embedding-issues/task-emb-023-complete
```

</task_spec>
