# TASK-EMB-005: Create Storage Types

<task_spec id="TASK-EMB-005" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-005 |
| **Title** | Create Storage Data Types |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 5 |
| **Implements** | REQ-EMB-006 |
| **Depends On** | TASK-EMB-004 |
| **Estimated Complexity** | medium |
| **Constitution Ref** | `storage.layer1_primary`, `storage.layer2c_per_embedder` |

---

## Context

TECH-EMB-004 specifies storage schema for quantized embeddings. This task creates the data structures for stored fingerprints, index entries, and query results. Actual storage implementation is Logic Layer.

**Why This Matters:**
- Proper storage schema enables per-embedder HNSW indexing
- Lazy loading (only fetch needed embedders) reduces latency
- Target size: ~17KB per fingerprint (Constitution requirement)

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-004-storage-module.md` |
| Quantization types | `crates/context-graph-embeddings/src/quantization/types.rs` |
| Existing storage | `crates/context-graph-storage/src/teleological/` |

---

## Prerequisites

- [ ] TASK-EMB-004 completed (quantization types exist)
- [ ] UUID crate available

---

## Scope

### In Scope
- Create `StoredFingerprint` struct
- Create `IndexEntry` struct
- Create `EmbedderQueryResult` struct
- Create `MultiSpaceQueryResult` struct
- Create `QuantizedStorage` trait signature

### Out of Scope
- Trait implementations (Logic Layer - TASK-EMB-022)
- RocksDB/ScyllaDB integration (Logic Layer)
- HNSW index management (Logic Layer)

---

## Definition of Done

### Exact Signatures

```rust
// File: crates/context-graph-embeddings/src/storage/types.rs

use crate::quantization::types::QuantizedEmbedding;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Complete stored fingerprint with quantized embeddings.
///
/// # Storage Layout
/// Each embedder's quantized embedding is stored separately for:
/// 1. Per-embedder HNSW indexing
/// 2. Lazy loading (only fetch needed embedders)
/// 3. Independent quantization per embedder
///
/// # Size Target
/// ~17KB per fingerprint (Constitution requirement)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredFingerprint {
    /// UUID of the fingerprint (primary key).
    pub id: Uuid,
    /// Storage version for migration support.
    pub version: u8,
    /// Per-embedder quantized embeddings (13 entries).
    pub embeddings: HashMap<u8, QuantizedEmbedding>,
    /// 13D purpose vector (NOT quantized - 52 bytes).
    pub purpose_vector: [f32; 13],
    /// Aggregate alignment to North Star.
    pub theta_to_north_star: f32,
    /// Johari quadrant weights [Open, Hidden, Blind, Unknown].
    pub johari_quadrants: [f32; 4],
    /// Dominant Johari quadrant index (0-3).
    pub dominant_quadrant: u8,
    /// Johari confidence score [0.0, 1.0].
    pub johari_confidence: f32,
    /// SHA-256 content hash.
    pub content_hash: [u8; 32],
    /// Creation timestamp (Unix millis).
    pub created_at_ms: i64,
    /// Last update timestamp (Unix millis).
    pub last_updated_ms: i64,
    /// Access count for LRU/importance scoring.
    pub access_count: u64,
    /// Soft-delete flag.
    pub deleted: bool,
}

impl StoredFingerprint {
    /// Storage version constant.
    pub const VERSION: u8 = 1;
    /// Expected size in bytes after quantization.
    pub const EXPECTED_SIZE_BYTES: usize = 17_000;
    /// Maximum allowed size.
    pub const MAX_SIZE_BYTES: usize = 25_000;
}

/// Entry in a per-embedder HNSW index.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// UUID of the fingerprint.
    pub id: Uuid,
    /// Dequantized embedding vector.
    pub vector: Vec<f32>,
    /// Precomputed L2 norm.
    pub norm: f32,
}

impl IndexEntry {
    /// Create index entry with precomputed norm.
    pub fn new(id: Uuid, vector: Vec<f32>) -> Self {
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        Self { id, vector, norm }
    }

    /// Get normalized vector for cosine similarity.
    pub fn normalized(&self) -> Vec<f32> {
        if self.norm > 1e-10 {
            self.vector.iter().map(|x| x / self.norm).collect()
        } else {
            vec![0.0; self.vector.len()]
        }
    }
}

/// Result from per-embedder index search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderQueryResult {
    /// Fingerprint UUID.
    pub id: Uuid,
    /// Embedder index (0-12).
    pub embedder_idx: u8,
    /// Similarity score [0.0, 1.0].
    pub similarity: f32,
    /// Distance (metric-specific).
    pub distance: f32,
    /// Rank in this embedder's result list.
    pub rank: usize,
}

/// Aggregated result from multi-space retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSpaceQueryResult {
    /// Fingerprint UUID.
    pub id: Uuid,
    /// Per-embedder similarities (13 values, NaN if not searched).
    pub embedder_similarities: [f32; 13],
    /// RRF fused score.
    pub rrf_score: f32,
    /// Weighted average similarity.
    pub weighted_similarity: f32,
    /// Purpose alignment score.
    pub purpose_alignment: f32,
    /// Number of embedders that contributed.
    pub embedder_count: usize,
}
```

### Constraints
- Structs only, trait implementations are Logic Layer
- Must use serde for serialization
- Size constants must match Constitution

### Verification
- All types compile
- Serde serialization works

---

## Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/storage/types.rs` | Storage data types |
| `crates/context-graph-embeddings/src/storage/mod.rs` | Module exports (replace empty placeholder) |

---

## Validation Criteria

- [ ] `StoredFingerprint::EXPECTED_SIZE_BYTES` equals 17000
- [ ] `embeddings` HashMap has capacity for 13 entries
- [ ] `purpose_vector` is exactly 13 elements
- [ ] `johari_quadrants` is exactly 4 elements

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings storage -- --nocapture
```

---

## Traceability

| Requirement | Tech Spec | Issue |
|-------------|-----------|-------|
| REQ-EMB-006 | TECH-EMB-004 | ISSUE-005 |

---

## Notes

- The 13D purpose vector is NOT quantized (only 52 bytes)
- Per-embedder storage enables efficient lazy loading
- Content hash enables deduplication

</task_spec>
