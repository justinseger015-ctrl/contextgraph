# Foundation Layer Task Specifications

<task_collection id="TASK-EMB-FOUNDATION" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Collection ID** | TASK-EMB-FOUNDATION |
| **Title** | Foundation Layer Task Specifications |
| **Status** | Ready |
| **Version** | 1.0 |
| **Layer** | Foundation (Data Models, Types, Constants) |
| **Task Count** | 10 (TASK-EMB-001 through TASK-EMB-010) |
| **Implements** | REQ-EMB-001, REQ-EMB-002, REQ-EMB-003, REQ-EMB-005, REQ-EMB-006 |
| **Related Tech Specs** | TECH-EMB-001, TECH-EMB-002, TECH-EMB-003, TECH-EMB-004 |
| **Created** | 2026-01-06 |
| **Constitution Reference** | v4.0.0 |

---

## Layer Execution Order

Foundation Layer tasks MUST complete before Logic Layer tasks can begin.

```
Foundation Layer (This Document)
         |
         v
Logic Layer (TASK-EMB-011 through TASK-EMB-020)
         |
         v
Surface Layer (TASK-EMB-021 through TASK-EMB-030)
```

---

## Dependencies Graph

```
TASK-EMB-001 (Dimension Constants)
      |
      +---> TASK-EMB-002 (ProjectionMatrix Struct)
      |          |
      |          +---> TASK-EMB-003 (ProjectionError Enum)
      |                     |
      |                     +---> TASK-EMB-007 (Error Types)
      |
      +---> TASK-EMB-004 (Quantization Structs)
      |          |
      |          +---> TASK-EMB-005 (Storage Types)
      |                     |
      |                     +---> TASK-EMB-007 (Error Types)
      |
      +---> TASK-EMB-006 (WarmLoadResult Struct)
      |          |
      |          +---> TASK-EMB-007 (Error Types)
      |
      +---> TASK-EMB-008 (Update SparseVector)

TASK-EMB-002 ---> TASK-EMB-009 (Weight File Spec)

TASK-EMB-010 (Golden Reference Fixtures) [Parallel - No Dependencies]
```

---

## Task Specifications

---

<task_spec id="TASK-EMB-001" version="1.0">

### TASK-EMB-001: Fix Dimension Constants

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Fix SPARSE_PROJECTED_DIMENSION Constant |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 1 |
| **Implements** | REQ-EMB-002 |
| **Depends On** | None (first task) |
| **Estimated Complexity** | low |

#### Context

The Constitution specifies E6_Sparse as `dim: "~30K 5%active"` with 1536D projected output. The current implementation incorrectly uses 768D. This is the root cause of dimension mismatch errors throughout the embedding pipeline.

This is the FIRST task because all other tasks depend on correct dimension constants.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Current constant | `crates/context-graph-embeddings/src/types/dimensions/constants.rs` |
| ModelId reference | `crates/context-graph-embeddings/src/types/model_id/core.rs` |
| Constitution | `docs/constitution.yaml` |

#### Prerequisites

- [ ] Read current `SPARSE_PROJECTED_DIMENSION` value (should be 768)
- [ ] Confirm Constitution requires 1536D for E6_Sparse

#### Scope

**In Scope:**
- Change `SPARSE_PROJECTED_DIMENSION` from 768 to 1536
- Change `SPARSE` constant in dimensions module to 1536
- Change `SPLADE` constant to 1536 (same as E6)
- Add compile-time assertion for dimension alignment

**Out of Scope:**
- Changing ModelId enum (handled separately)
- Changing embedding logic (Logic Layer)
- Changing storage formats (handled in TASK-EMB-005)

#### Definition of Done

**Exact Signatures:**

```rust
// File: crates/context-graph-embeddings/src/types/dimensions/constants.rs

/// Sparse vector vocabulary size (BERT WordPiece vocabulary)
pub const SPARSE_VOCAB_SIZE: usize = 30522;

/// E6: Sparse projected dimension (30K sparse -> 1536D via learned projection)
/// FIXED: Was 768, Constitution requires 1536
pub const SPARSE: usize = 1536;

/// E13: SPLADE v3 projected dimension (same as E6)
pub const SPLADE: usize = 1536;

// File: crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs

/// Projected dimension for multi-array storage compatibility.
/// CRITICAL FIX: Changed from 768 to 1536 per Constitution
pub const SPARSE_PROJECTED_DIMENSION: usize = 1536;

// Compile-time assertion
const _: () = assert!(
    SPARSE_PROJECTED_DIMENSION == crate::types::dimensions::constants::SPARSE,
    "SPARSE_PROJECTED_DIMENSION must match dimensions::constants::SPARSE"
);
```

**Constraints:**
- No runtime logic changes
- Constants only
- Must be compile-time verifiable

**Verification:**
- `cargo check` succeeds
- Compile-time assertion passes
- `grep -r "768" crates/context-graph-embeddings/` returns no SPARSE-related hits

#### Files to Modify

| File | Change Type |
|------|-------------|
| `crates/context-graph-embeddings/src/types/dimensions/constants.rs` | MODIFY |
| `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` | MODIFY |

#### Validation Criteria

- [ ] `SPARSE_PROJECTED_DIMENSION` equals 1536
- [ ] `dimensions::constants::SPARSE` equals 1536
- [ ] `dimensions::constants::SPLADE` equals 1536
- [ ] Compile-time assertion present and passing
- [ ] No references to 768 in sparse-related code

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
grep -rn "SPARSE_PROJECTED_DIMENSION" crates/context-graph-embeddings/
grep -rn "768" crates/context-graph-embeddings/src/models/pretrained/sparse/
```

</task_spec>

---

<task_spec id="TASK-EMB-002" version="1.0">

### TASK-EMB-002: Create ProjectionMatrix Struct

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Create ProjectionMatrix Struct |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 2 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-001 |
| **Estimated Complexity** | medium |

#### Context

TECH-EMB-001 specifies a learned projection matrix to replace the broken hash-based sparse-to-dense projection. This task creates the data structure. The actual loading and projection logic are Logic Layer tasks.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Dimension constants | `crates/context-graph-embeddings/src/types/dimensions/constants.rs` |
| SparseVector type | `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |

#### Prerequisites

- [ ] TASK-EMB-001 completed (dimension constants fixed)
- [ ] Candle crate available for Tensor type
- [ ] Device type available from Candle

#### Scope

**In Scope:**
- Create `ProjectionMatrix` struct with weights, device, checksum
- Add struct documentation with Constitution references
- Define associated constants for expected shape

**Out of Scope:**
- `load()` implementation (Logic Layer)
- `project()` implementation (Logic Layer)
- CUDA integration (Logic Layer)

#### Definition of Done

**Exact Signatures:**

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs

use candle_core::{Device, Tensor};
use super::types::{SPARSE_PROJECTED_DIMENSION, SPARSE_VOCAB_SIZE};

/// Expected weight file path relative to model directory.
pub const PROJECTION_WEIGHT_FILE: &str = "sparse_projection.safetensors";

/// Expected tensor name in SafeTensors file.
pub const PROJECTION_TENSOR_NAME: &str = "projection.weight";

/// Learned projection matrix for sparse-to-dense conversion.
///
/// # Constitution Alignment
/// - E6_Sparse: `dim: "~30K 5%active"` input, 1536D output
/// - E13_Splade: Same architecture, same projection
///
/// # Weight Source
/// - Pre-trained via contrastive learning on MS MARCO
/// - Fine-tuned to preserve semantic similarity
///
/// # CRITICAL: No Fallback
/// If weight file is missing, system MUST panic. Hash fallback is FORBIDDEN.
#[derive(Debug)]
pub struct ProjectionMatrix {
    /// Weight tensor on GPU: [SPARSE_VOCAB_SIZE x SPARSE_PROJECTED_DIMENSION]
    /// Shape: [30522, 1536]
    weights: Tensor,

    /// Device where weights are loaded (must be CUDA)
    device: Device,

    /// SHA256 checksum of the weight file for validation
    weight_checksum: [u8; 32],
}

impl ProjectionMatrix {
    /// Expected weight matrix shape: [vocab_size, projected_dim]
    pub const EXPECTED_SHAPE: (usize, usize) = (SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION);

    /// Expected file size in bytes: vocab_size * proj_dim * sizeof(f32)
    pub const EXPECTED_FILE_SIZE: usize = SPARSE_VOCAB_SIZE * SPARSE_PROJECTED_DIMENSION * 4;

    /// Get the weight tensor reference.
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the weight checksum.
    pub fn checksum(&self) -> &[u8; 32] {
        &self.weight_checksum
    }

    /// Check if weights are on CUDA device.
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }
}
```

**Constraints:**
- Struct only, no method implementations beyond accessors
- Must use Candle types
- Must reference dimension constants from TASK-EMB-001

**Verification:**
- `cargo check` succeeds
- Struct size is reasonable (Tensor is opaque)

#### Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | ProjectionMatrix struct and constants |

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs` | Add `pub mod projection;` |

#### Validation Criteria

- [ ] `ProjectionMatrix` struct compiles
- [ ] `EXPECTED_SHAPE` constant equals (30522, 1536)
- [ ] Documentation references Constitution
- [ ] No load/project implementations yet

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo doc -p context-graph-embeddings --no-deps
```

</task_spec>

---

<task_spec id="TASK-EMB-003" version="1.0">

### TASK-EMB-003: Create ProjectionError Enum

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Create ProjectionError Enum |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 3 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-002 |
| **Estimated Complexity** | low |

#### Context

The projection system needs specific error types for clear error messages. These errors implement the Constitution's "fail fast" requirement with specific remediation steps in error messages.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Projection module | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` |
| Tech spec errors | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |
| Functional spec errors | `docs2/codecheck/embeddingissues/SPEC-EMB-001-master-functional.md` |

#### Prerequisites

- [ ] TASK-EMB-002 completed (projection module exists)
- [ ] thiserror crate available

#### Scope

**In Scope:**
- Create `ProjectionError` enum with 5 variants
- Each variant has descriptive error message with error code
- Include remediation steps in error messages

**Out of Scope:**
- Error handling logic (Logic Layer)
- Integration with other error types (TASK-EMB-007)

#### Definition of Done

**Exact Signatures:**

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
// (Add to existing file from TASK-EMB-002)

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during sparse projection.
#[derive(Debug, Error)]
pub enum ProjectionError {
    /// Projection weights file not found.
    ///
    /// # Remediation
    /// Download from: https://huggingface.co/contextgraph/sparse-projection
    #[error("[EMB-E006] PROJECTION_MATRIX_MISSING: Weight file not found at {path}
  Expected: models/sparse_projection.safetensors
  Remediation: Download projection weights or train custom matrix")]
    MatrixMissing { path: PathBuf },

    /// Weight file checksum does not match expected value.
    #[error("[EMB-E004] WEIGHT_CHECKSUM_MISMATCH: Corrupted weight file
  Expected: {expected}
  Actual: {actual}
  File: {path}
  Remediation: Re-download weight file")]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },

    /// Weight matrix has wrong shape.
    #[error("[EMB-E005] DIMENSION_MISMATCH: Projection matrix has wrong shape
  Expected: [30522, 1536]
  Actual: [{actual_rows}, {actual_cols}]
  File: {path}")]
    DimensionMismatch {
        path: PathBuf,
        actual_rows: usize,
        actual_cols: usize,
    },

    /// CUDA operation failed.
    #[error("[EMB-E001] CUDA_ERROR: GPU operation failed
  Operation: {operation}
  Details: {details}
  Remediation: Check GPU availability and driver version")]
    GpuError { operation: String, details: String },

    /// Projection not initialized (weights not loaded).
    #[error("[EMB-E008] NOT_INITIALIZED: Projection weights not loaded
  Remediation: Call ProjectionMatrix::load() before projection")]
    NotInitialized,
}
```

**Constraints:**
- Use thiserror for derive
- Error codes must match SPEC-EMB-001 error taxonomy
- Each error includes remediation steps

**Verification:**
- All error variants compile
- Error messages are formatted correctly

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | Add ProjectionError enum |

#### Validation Criteria

- [ ] 5 error variants defined
- [ ] All error codes match SPEC-EMB-001 (EMB-E001, EMB-E004, EMB-E005, EMB-E006, EMB-E008)
- [ ] Error messages include remediation steps
- [ ] thiserror::Error derived

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings projection -- --nocapture 2>&1 | head -20
```

</task_spec>

---

<task_spec id="TASK-EMB-004" version="1.0">

### TASK-EMB-004: Create Quantization Structs

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Create Quantization Data Structures |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 4 |
| **Implements** | REQ-EMB-005 |
| **Depends On** | TASK-EMB-001 |
| **Estimated Complexity** | medium |

#### Context

TECH-EMB-003 specifies quantization methods per Constitutional requirements. This task creates the data structures for quantized embeddings, codebooks, and encoders. Actual quantization logic is Logic Layer.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-003-quantization.md` |
| Constitution | `docs/constitution.yaml` |
| ModelId | `crates/context-graph-embeddings/src/types/model_id/core.rs` |

#### Prerequisites

- [ ] TASK-EMB-001 completed (dimension constants)
- [ ] Understand per-embedder quantization assignments from Constitution

#### Scope

**In Scope:**
- Create `QuantizationMethod` enum with 5 variants
- Create `QuantizedEmbedding` struct
- Create `QuantizationMetadata` enum (per-method metadata)
- Create `PQ8Codebook` struct
- Create `Float8Encoder` struct (stateless)
- Create `BinaryEncoder` struct (stateless)

**Out of Scope:**
- Quantization implementations (Logic Layer)
- Dequantization implementations (Logic Layer)
- Codebook loading/training (Logic Layer)

#### Definition of Done

**Exact Signatures:**

```rust
// File: crates/context-graph-embeddings/src/quantization/types.rs

use crate::types::model_id::ModelId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantization methods aligned with Constitution.
///
/// # Constitution Alignment
/// - PQ_8: E1, E5, E7, E10 (32x compression, <5% recall impact)
/// - Float8: E2, E3, E4, E8, E11 (4x compression, <0.3% recall impact)
/// - Binary: E9 (32x compression, 5-10% recall impact)
/// - Sparse: E6, E13 (native format, 0% recall impact)
/// - TokenPruning: E12 (~50% compression, <2% recall impact)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationMethod {
    /// Product Quantization with 8 subvectors, 256 centroids each.
    /// Used for: E1_Semantic, E5_Causal, E7_Code, E10_Multimodal
    PQ8,

    /// 8-bit floating point in E4M3 format.
    /// Used for: E2-E4 Temporal, E8_Graph, E11_Entity
    Float8E4M3,

    /// Binary quantization (sign bit only).
    /// Used for: E9_HDC
    Binary,

    /// Sparse format: indices + values.
    /// Used for: E6_Sparse, E13_SPLADE
    SparseNative,

    /// Token pruning: keep top 50% tokens.
    /// Used for: E12_LateInteraction
    TokenPruning,
}

impl QuantizationMethod {
    /// Get quantization method for a given ModelId.
    pub fn for_model_id(model_id: ModelId) -> Self {
        match model_id {
            ModelId::Semantic | ModelId::Causal | ModelId::Code | ModelId::Multimodal => Self::PQ8,
            ModelId::TemporalRecent | ModelId::TemporalPeriodic | ModelId::TemporalPositional
            | ModelId::Graph | ModelId::Entity => Self::Float8E4M3,
            ModelId::Hdc => Self::Binary,
            ModelId::Sparse | ModelId::Splade => Self::SparseNative,
            ModelId::LateInteraction => Self::TokenPruning,
        }
    }

    /// Theoretical compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        match self {
            Self::PQ8 => 32.0,
            Self::Float8E4M3 => 4.0,
            Self::Binary => 32.0,
            Self::SparseNative => 1.0, // Variable
            Self::TokenPruning => 2.0,
        }
    }

    /// Maximum acceptable recall loss.
    pub fn max_recall_loss(&self) -> f32 {
        match self {
            Self::PQ8 => 0.05,
            Self::Float8E4M3 => 0.003,
            Self::Binary => 0.10,
            Self::SparseNative => 0.0,
            Self::TokenPruning => 0.02,
        }
    }
}

/// Quantized embedding ready for storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedEmbedding {
    /// Quantization method used.
    pub method: QuantizationMethod,
    /// Original embedding dimension.
    pub original_dim: usize,
    /// Compressed embedding bytes.
    pub data: Vec<u8>,
    /// Method-specific metadata.
    pub metadata: QuantizationMetadata,
}

impl QuantizedEmbedding {
    /// Compute compressed size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Compute compression ratio vs float32.
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.original_dim * 4;
        original_bytes as f32 / self.data.len().max(1) as f32
    }
}

/// Method-specific metadata for dequantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationMetadata {
    /// PQ-8: Codebook identifier.
    PQ8 { codebook_id: u32, num_subvectors: u8 },
    /// Float8: Scale and bias for denormalization.
    Float8 { scale: f32, bias: f32 },
    /// Binary: Threshold used for binarization.
    Binary { threshold: f32 },
    /// Sparse: Number of non-zero elements.
    Sparse { vocab_size: usize, nnz: usize },
    /// Token pruning: Pruning details.
    TokenPruning { original_tokens: usize, kept_tokens: usize, threshold: f32 },
}

/// PQ-8 codebook with 8 subvectors, 256 centroids each.
#[derive(Debug)]
pub struct PQ8Codebook {
    /// Embedding dimension this codebook was trained for.
    pub embedding_dim: usize,
    /// Number of subvectors (typically 8).
    pub num_subvectors: usize,
    /// Centroids per subvector (typically 256).
    pub num_centroids: usize,
    /// Centroid vectors: [num_subvectors][num_centroids][subvector_dim]
    pub centroids: Vec<Vec<Vec<f32>>>,
    /// Codebook identifier.
    pub codebook_id: u32,
}

/// Float8 E4M3 encoder (stateless).
#[derive(Debug, Clone, Copy, Default)]
pub struct Float8Encoder;

/// Binary encoder (stateless).
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryEncoder;
```

**Constraints:**
- Structs only, minimal method implementations
- Must use serde for serialization
- Compression ratios must match Constitution

**Verification:**
- All types compile
- Serde serialization works

#### Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/quantization/types.rs` | All quantization types |
| `crates/context-graph-embeddings/src/quantization/mod.rs` | Module exports |

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/lib.rs` | Add `pub mod quantization;` |

#### Validation Criteria

- [ ] `QuantizationMethod` has 5 variants
- [ ] `for_model_id()` returns correct method for all 13 embedders
- [ ] Compression ratios match Constitution
- [ ] Max recall loss values match Constitution

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings quantization -- --nocapture
```

</task_spec>

---

<task_spec id="TASK-EMB-005" version="1.0">

### TASK-EMB-005: Create Storage Types

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Create Storage Data Types |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 5 |
| **Implements** | REQ-EMB-006 |
| **Depends On** | TASK-EMB-004 |
| **Estimated Complexity** | medium |

#### Context

TECH-EMB-004 specifies storage schema for quantized embeddings. This task creates the data structures for stored fingerprints, index entries, and query results. Actual storage implementation is Logic Layer.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-004-storage-module.md` |
| Quantization types | `crates/context-graph-embeddings/src/quantization/types.rs` |
| Existing storage | `crates/context-graph-storage/src/teleological/` |

#### Prerequisites

- [ ] TASK-EMB-004 completed (quantization types exist)
- [ ] UUID crate available

#### Scope

**In Scope:**
- Create `StoredFingerprint` struct
- Create `IndexEntry` struct
- Create `EmbedderQueryResult` struct
- Create `MultiSpaceQueryResult` struct
- Create `QuantizedStorage` trait signature

**Out of Scope:**
- Trait implementations (Logic Layer)
- RocksDB/ScyllaDB integration (Logic Layer)
- HNSW index management (Logic Layer)

#### Definition of Done

**Exact Signatures:**

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

**Constraints:**
- Structs only, trait implementations are Logic Layer
- Must use serde for serialization
- Size constants must match Constitution

**Verification:**
- All types compile
- Serde serialization works

#### Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/storage/types.rs` | Storage data types |
| `crates/context-graph-embeddings/src/storage/mod.rs` | Module exports (replace empty placeholder) |

#### Validation Criteria

- [ ] `StoredFingerprint::EXPECTED_SIZE_BYTES` equals 17000
- [ ] `embeddings` HashMap has capacity for 13 entries
- [ ] `purpose_vector` is exactly 13 elements
- [ ] `johari_quadrants` is exactly 4 elements

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings storage -- --nocapture
```

</task_spec>

---

<task_spec id="TASK-EMB-006" version="1.0">

### TASK-EMB-006: Create WarmLoadResult Struct

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Create Warm Loading Data Types |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 6 |
| **Implements** | REQ-EMB-003 |
| **Depends On** | TASK-EMB-001 |
| **Estimated Complexity** | medium |

#### Context

TECH-EMB-002 specifies real weight loading to replace simulated operations. This task creates the data structures for loaded weights, GPU tensors, and loading results. Actual loading implementation is Logic Layer.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-002-warm-loading.md` |
| Current warm loader | `crates/context-graph-embeddings/src/warm/loader/` |

#### Prerequisites

- [ ] TASK-EMB-001 completed
- [ ] cudarc types available (for DevicePtr)
- [ ] Instant type from std::time

#### Scope

**In Scope:**
- Create `WarmLoadResult` struct
- Create `GpuTensor` struct
- Create `LoadedModelWeights` struct
- Create `TensorMetadata` struct

**Out of Scope:**
- Actual loading implementations (Logic Layer)
- CUDA calls (Logic Layer)
- SafeTensors parsing (Logic Layer)

#### Definition of Done

**Exact Signatures:**

```rust
// File: crates/context-graph-embeddings/src/warm/loader/types.rs

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Data type for model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

/// Result of loading a model's weights into GPU memory.
///
/// # Constitution Alignment
/// - REQ-WARM-003: Non-evictable VRAM allocation
/// - REQ-WARM-005: Weight integrity verification
///
/// # CRITICAL: No Simulation
/// All fields contain REAL data from actual loading operations.
#[derive(Debug)]
pub struct WarmLoadResult {
    /// Real GPU device pointer from cudaMalloc.
    pub gpu_ptr: u64, // Raw pointer as u64 for portability
    /// Real SHA256 checksum of the weight file.
    pub checksum: [u8; 32],
    /// Actual size of weights in GPU memory.
    pub size_bytes: usize,
    /// Loading duration for performance monitoring.
    pub load_duration: Duration,
    /// Tensor metadata from SafeTensors header.
    pub tensor_metadata: TensorMetadata,
}

/// Metadata extracted from SafeTensors file header.
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Tensor name -> shape mapping.
    pub shapes: HashMap<String, Vec<usize>>,
    /// Data type of tensors.
    pub dtype: DType,
    /// Total number of parameters.
    pub total_params: usize,
}

/// GPU-resident tensor with real CUDA allocation.
///
/// # Safety
/// The `device_ptr` is a REAL CUDA device pointer obtained from cudaMalloc.
/// It is ONLY valid while the CUDA context is active.
#[derive(Debug)]
pub struct GpuTensor {
    /// Real CUDA device pointer from cudaMalloc.
    device_ptr: u64,
    /// Tensor shape (e.g., [vocab_size, hidden_dim]).
    shape: Vec<usize>,
    /// Total number of elements.
    numel: usize,
    /// CUDA device ID where tensor is allocated.
    device_id: u32,
}

impl GpuTensor {
    /// Create a new GPU tensor record.
    ///
    /// Note: This does NOT allocate memory. Allocation is done in Logic Layer.
    pub fn new(device_ptr: u64, shape: Vec<usize>, device_id: u32) -> Self {
        let numel = shape.iter().product();
        Self { device_ptr, shape, numel, device_id }
    }

    /// Get the raw device pointer.
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    /// Get tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get number of elements.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Get device ID.
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

/// Complete set of weights for a model loaded into GPU memory.
#[derive(Debug)]
pub struct LoadedModelWeights {
    /// Model identifier (e.g., "E1_Semantic").
    pub model_id: String,
    /// Named tensors loaded to GPU.
    pub tensors: HashMap<String, GpuTensor>,
    /// SHA256 checksum of source weight file.
    pub file_checksum: [u8; 32],
    /// Total GPU memory used (bytes).
    pub total_gpu_bytes: usize,
    /// CUDA device where weights are loaded.
    pub device_id: u32,
    /// Timestamp when weights were loaded.
    pub loaded_at: Instant,
}

impl LoadedModelWeights {
    /// Get a specific tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&GpuTensor> {
        self.tensors.get(name)
    }

    /// Check if all expected tensors are present.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}
```

**Constraints:**
- Use u64 for device pointers (portable across CUDA wrappers)
- No actual CUDA calls in this task
- Structs document real-data requirement

**Verification:**
- All types compile
- No dependencies on CUDA at compile time

#### Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/warm/loader/types.rs` | Warm loading types |

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/warm/loader/mod.rs` | Add `pub mod types;` |

#### Validation Criteria

- [ ] `WarmLoadResult` has checksum as `[u8; 32]`
- [ ] `GpuTensor` uses u64 for device pointer
- [ ] `LoadedModelWeights` has HashMap for tensors
- [ ] All structs document CRITICAL: No Simulation

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
```

</task_spec>

---

<task_spec id="TASK-EMB-007" version="1.0">

### TASK-EMB-007: Create Consolidated Error Types

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Create Consolidated Error Enum |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 7 |
| **Implements** | All REQs |
| **Depends On** | TASK-EMB-003, TASK-EMB-005, TASK-EMB-006 |
| **Estimated Complexity** | medium |

#### Context

The embedding system needs a consolidated error type that encompasses all error scenarios from projection, warm loading, quantization, and storage. This provides a single error type for the public API.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Functional spec errors | `docs2/codecheck/embeddingissues/SPEC-EMB-001-master-functional.md` |
| ProjectionError | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` |

#### Prerequisites

- [ ] TASK-EMB-003 completed (ProjectionError exists)
- [ ] TASK-EMB-005 completed (Storage types exist)
- [ ] TASK-EMB-006 completed (Warm loading types exist)

#### Scope

**In Scope:**
- Create `EmbeddingError` enum covering all error codes from SPEC-EMB-001
- Implement From conversions from specific error types
- Include all 10 error codes from error taxonomy

**Out of Scope:**
- Error handling logic (Logic Layer)
- Recovery strategies (Logic Layer)

#### Definition of Done

**Exact Signatures:**

```rust
// File: crates/context-graph-embeddings/src/error.rs

use std::path::PathBuf;
use thiserror::Error;
use crate::types::model_id::ModelId;

/// Consolidated errors for the embedding system.
///
/// Error codes from SPEC-EMB-001 Error Taxonomy:
/// - EMB-E001: CUDA_UNAVAILABLE
/// - EMB-E002: INSUFFICIENT_VRAM
/// - EMB-E003: WEIGHT_FILE_MISSING
/// - EMB-E004: WEIGHT_CHECKSUM_MISMATCH
/// - EMB-E005: DIMENSION_MISMATCH
/// - EMB-E006: PROJECTION_MATRIX_MISSING
/// - EMB-E007: OOM_DURING_BATCH
/// - EMB-E008: INFERENCE_VALIDATION_FAILED
/// - EMB-E009: INPUT_TOO_LARGE
/// - EMB-E010: STORAGE_CORRUPTION
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// EMB-E001: CUDA is required but unavailable.
    #[error("[EMB-E001] CUDA_UNAVAILABLE: {message}
  Required: RTX 5090 (Blackwell, CC 12.0)
  Remediation: Install CUDA 13.1+ and verify GPU")]
    CudaUnavailable { message: String },

    /// EMB-E002: Insufficient GPU VRAM.
    #[error("[EMB-E002] INSUFFICIENT_VRAM: Not enough GPU memory
  Required: {required_bytes} bytes ({required_gb:.1} GB)
  Available: {available_bytes} bytes ({available_gb:.1} GB)
  Remediation: Free GPU memory or upgrade hardware")]
    InsufficientVram {
        required_bytes: usize,
        available_bytes: usize,
        required_gb: f64,
        available_gb: f64,
    },

    /// EMB-E003: Weight file not found.
    #[error("[EMB-E003] WEIGHT_FILE_MISSING: Model weight file not found
  Model: {model_id:?}
  Path: {path}
  Remediation: Download weights from model repository")]
    WeightFileMissing { model_id: ModelId, path: PathBuf },

    /// EMB-E004: Weight file checksum mismatch.
    #[error("[EMB-E004] WEIGHT_CHECKSUM_MISMATCH: Corrupted weight file
  Model: {model_id:?}
  Expected: {expected}
  Actual: {actual}
  Remediation: Re-download weight file")]
    WeightChecksumMismatch {
        model_id: ModelId,
        expected: String,
        actual: String,
    },

    /// EMB-E005: Dimension mismatch.
    #[error("[EMB-E005] DIMENSION_MISMATCH: Embedding dimension mismatch
  Model: {model_id:?}
  Expected: {expected}
  Actual: {actual}
  Remediation: Verify model configuration")]
    DimensionMismatch {
        model_id: ModelId,
        expected: usize,
        actual: usize,
    },

    /// EMB-E006: Projection matrix missing.
    #[error("[EMB-E006] PROJECTION_MATRIX_MISSING: Sparse projection weights not found
  Path: {path}
  Remediation: Download from https://huggingface.co/contextgraph/sparse-projection")]
    ProjectionMatrixMissing { path: PathBuf },

    /// EMB-E007: Out of memory during batch processing.
    #[error("[EMB-E007] OOM_DURING_BATCH: GPU out of memory during batch
  Batch size: {batch_size}
  Remediation: Reduce batch size or free GPU memory")]
    OomDuringBatch { batch_size: usize },

    /// EMB-E008: Inference validation failed.
    #[error("[EMB-E008] INFERENCE_VALIDATION_FAILED: Model inference output invalid
  Model: {model_id:?}
  Reason: {reason}
  Remediation: Check model weights and configuration")]
    InferenceValidationFailed { model_id: ModelId, reason: String },

    /// EMB-E009: Input too large.
    #[error("[EMB-E009] INPUT_TOO_LARGE: Input exceeds maximum size
  Max tokens: {max_tokens}
  Actual tokens: {actual_tokens}
  Remediation: Truncate input or split into chunks")]
    InputTooLarge { max_tokens: usize, actual_tokens: usize },

    /// EMB-E010: Storage corruption detected.
    #[error("[EMB-E010] STORAGE_CORRUPTION: Stored data is corrupted
  ID: {id}
  Reason: {reason}
  Remediation: Re-index from source")]
    StorageCorruption { id: String, reason: String },

    /// EMB-E011: Quantization codebook missing.
    #[error("[EMB-E011] CODEBOOK_MISSING: PQ-8 codebook not found
  Model: {model_id:?}
  Remediation: Train codebook or download from model repository")]
    CodebookMissing { model_id: ModelId },

    /// EMB-E012: Recall loss exceeded.
    #[error("[EMB-E012] RECALL_LOSS_EXCEEDED: Quantization quality too low
  Model: {model_id:?}
  Measured: {measured:.4}
  Max allowed: {max_allowed:.4}")]
    RecallLossExceeded {
        model_id: ModelId,
        measured: f32,
        max_allowed: f32,
    },
}

impl EmbeddingError {
    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(self, Self::InputTooLarge { .. })
    }

    /// Get error code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::CudaUnavailable { .. } => "EMB-E001",
            Self::InsufficientVram { .. } => "EMB-E002",
            Self::WeightFileMissing { .. } => "EMB-E003",
            Self::WeightChecksumMismatch { .. } => "EMB-E004",
            Self::DimensionMismatch { .. } => "EMB-E005",
            Self::ProjectionMatrixMissing { .. } => "EMB-E006",
            Self::OomDuringBatch { .. } => "EMB-E007",
            Self::InferenceValidationFailed { .. } => "EMB-E008",
            Self::InputTooLarge { .. } => "EMB-E009",
            Self::StorageCorruption { .. } => "EMB-E010",
            Self::CodebookMissing { .. } => "EMB-E011",
            Self::RecallLossExceeded { .. } => "EMB-E012",
        }
    }
}
```

**Constraints:**
- All error codes from SPEC-EMB-001 must be present
- Each error includes remediation steps
- `is_recoverable()` returns true only for EMB-E009

**Verification:**
- All 12 error variants defined
- Error codes match specification

#### Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/error.rs` | Consolidated error enum |

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/lib.rs` | Add `pub mod error;` and re-export |

#### Validation Criteria

- [ ] All 10 error codes from SPEC-EMB-001 present
- [ ] `is_recoverable()` only returns true for InputTooLarge
- [ ] `code()` returns correct error code string
- [ ] All errors include remediation steps

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings error -- --nocapture
```

</task_spec>

---

<task_spec id="TASK-EMB-008" version="1.0">

### TASK-EMB-008: Update SparseVector Struct

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Update SparseVector Struct |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 8 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-001 |
| **Estimated Complexity** | low |

#### Context

TECH-EMB-001 specifies removing the broken `to_dense_projected()` method from SparseVector and adding `to_csr()` for cuBLAS integration. The broken hash-based projection is replaced by ProjectionMatrix.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Current SparseVector | `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |

#### Prerequisites

- [ ] TASK-EMB-001 completed (dimension constants fixed)

#### Scope

**In Scope:**
- Remove `to_dense_projected()` method
- Add `to_csr()` method signature
- Update documentation to reference ProjectionMatrix

**Out of Scope:**
- `to_csr()` implementation (Logic Layer)
- ProjectionMatrix integration (Logic Layer)

#### Definition of Done

**Exact Changes:**

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs

/// Sparse vector output with term indices and weights.
///
/// # Constitution Alignment
/// - Dimension: SPARSE_VOCAB_SIZE (30522)
/// - Expected sparsity: ~95% zeros (~5% active)
/// - Output after projection: 1536D dense (via ProjectionMatrix)
///
/// # BREAKING CHANGE
/// `to_dense_projected()` has been REMOVED. Use `ProjectionMatrix::project()` instead.
#[derive(Debug, Clone)]
pub struct SparseVector {
    /// Token indices with non-zero weights (sorted ascending).
    pub indices: Vec<usize>,
    /// Corresponding weights for each index.
    pub weights: Vec<f32>,
    /// Total number of dimensions (vocabulary size = 30522).
    pub dimension: usize,
}

impl SparseVector {
    /// Create a new sparse vector.
    ///
    /// # Invariants
    /// - indices.len() == weights.len()
    /// - All indices < SPARSE_VOCAB_SIZE (30522)
    /// - Indices are sorted ascending
    pub fn new(indices: Vec<usize>, weights: Vec<f32>) -> Self {
        debug_assert_eq!(indices.len(), weights.len());
        Self {
            indices,
            weights,
            dimension: SPARSE_VOCAB_SIZE,
        }
    }

    /// Convert to CSR (Compressed Sparse Row) format for cuBLAS.
    ///
    /// # Returns
    /// (row_ptr, col_indices, values) tuple for CSR representation.
    ///
    /// # Implementation Note
    /// Actual implementation in Logic Layer.
    pub fn to_csr(&self) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        // For a single vector (1 row), CSR is:
        // row_ptr = [0, nnz]
        // col_indices = indices as i32
        // values = weights
        let nnz = self.indices.len() as i32;
        let row_ptr = vec![0i32, nnz];
        let col_indices: Vec<i32> = self.indices.iter().map(|&i| i as i32).collect();
        let values = self.weights.clone();
        (row_ptr, col_indices, values)
    }

    /// Get number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Get sparsity as percentage of zeros.
    pub fn sparsity(&self) -> f32 {
        1.0 - (self.indices.len() as f32 / self.dimension as f32)
    }

    // REMOVED: to_dense_projected()
    // Use ProjectionMatrix::project() instead.
    // The old hash-based projection (idx % projected_dim) destroyed semantic information.
}
```

**Constraints:**
- `to_dense_projected()` must be deleted or commented out
- Documentation must reference ProjectionMatrix
- `to_csr()` can have simple implementation

**Verification:**
- `to_dense_projected` method no longer exists
- `to_csr` returns correct format
- Documentation updated

#### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` | Remove old method, add new |

#### Validation Criteria

- [ ] `to_dense_projected()` removed or marked deprecated
- [ ] `to_csr()` method present
- [ ] Documentation references ProjectionMatrix
- [ ] No compile errors

#### Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
grep -n "to_dense_projected" crates/context-graph-embeddings/
```

</task_spec>

---

<task_spec id="TASK-EMB-009" version="1.0">

### TASK-EMB-009: Create Weight File Specification

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Document Weight File Specification |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 9 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-002 |
| **Estimated Complexity** | low |

#### Context

The sparse projection weight file needs formal specification for validation during loading. This includes expected shape, checksum verification, and download instructions.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |
| ProjectionMatrix | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` |

#### Prerequisites

- [ ] TASK-EMB-002 completed (ProjectionMatrix struct exists)

#### Scope

**In Scope:**
- Create weight file specification in models/README.md
- Document expected tensor shape [30522, 1536]
- Document checksum verification process
- Document download URL

**Out of Scope:**
- Actual weight training (separate process)
- Weight file creation (separate process)

#### Definition of Done

**Exact File Content:**

```markdown
# File: crates/context-graph-embeddings/models/README.md

# Model Weight Files

This directory contains pre-trained model weights required for the embedding pipeline.

## Sparse Projection Matrix

### File Details

| Property | Value |
|----------|-------|
| **File Name** | `sparse_projection.safetensors` |
| **Format** | SafeTensors v0.4+ |
| **Tensor Name** | `projection.weight` |
| **Shape** | [30522, 1536] |
| **Data Type** | float32 |
| **Size** | ~187 MB (30522 * 1536 * 4 bytes) |
| **Download URL** | https://huggingface.co/contextgraph/sparse-projection |

### Checksum Verification

The embedding system verifies weight file integrity via SHA-256 checksum.

Expected checksum (placeholder until training complete):
```
SHA256: <TBD_AFTER_TRAINING>
```

### Training Details

- **Trained On**: MS MARCO passages
- **Training Objective**: Contrastive learning
- **Semantic Preservation Score**: >0.85
- **Constitution Version**: 4.0.0

### Usage

The projection matrix is loaded automatically during model initialization:

```rust
// Automatic loading in SparseModel::load()
let projection = ProjectionMatrix::load(&model_path)?;

// Project sparse to dense
let dense = projection.project(&sparse_vector)?;
```

### CRITICAL: No Hash Fallback

If this file is missing, the system WILL PANIC. There is NO fallback to hash-based projection.

The hash-based approach (`idx % projected_dim`) destroys semantic information and is
explicitly forbidden by Constitution AP-007 (no stub data in production).
```

**Constraints:**
- Must be placed in models/ directory
- Checksum placeholder is acceptable (TBD)
- Must emphasize no fallback policy

**Verification:**
- README exists
- Correct tensor shape documented
- Download URL present

#### Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/models/README.md` | Weight file specification |

#### Validation Criteria

- [ ] Shape documented as [30522, 1536]
- [ ] File size documented as ~187 MB
- [ ] Download URL present
- [ ] No fallback policy emphasized

#### Test Commands

```bash
ls -la /home/cabdru/contextgraph/crates/context-graph-embeddings/models/
cat /home/cabdru/contextgraph/crates/context-graph-embeddings/models/README.md
```

</task_spec>

---

<task_spec id="TASK-EMB-010" version="1.0">

### TASK-EMB-010: Create Golden Reference Fixtures

#### Metadata

| Field | Value |
|-------|-------|
| **Title** | Create Golden Reference Test Fixtures |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 10 (parallel with others) |
| **Implements** | REQ-EMB-003 |
| **Depends On** | None (can run in parallel) |
| **Estimated Complexity** | low |

#### Context

TECH-EMB-002 specifies golden reference outputs for inference validation. This task creates the directory structure and placeholder files. Actual golden data generation requires model inference.

#### Input Context Files

| Purpose | File Path |
|---------|-----------|
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-002-warm-loading.md` |

#### Prerequisites

- [ ] None (independent task)

#### Scope

**In Scope:**
- Create `tests/fixtures/golden/` directory structure
- Create placeholder files for each model
- Document expected file format

**Out of Scope:**
- Actual golden data generation (requires model inference)
- Test implementation (Logic Layer)

#### Definition of Done

**Directory Structure:**

```
crates/context-graph-embeddings/tests/fixtures/golden/
    README.md                    # Format documentation
    E1_Semantic/
        test_input.bin           # Placeholder
        golden_output.bin        # Placeholder
        checksum.txt             # Placeholder
    E2_TemporalRecent/
        test_input.bin
        golden_output.bin
        checksum.txt
    E3_TemporalPeriodic/
        ...
    E4_TemporalPositional/
        ...
    E5_Causal/
        ...
    E6_Sparse/
        ...
    E7_Code/
        ...
    E8_Graph/
        ...
    E9_Hdc/
        ...
    E10_Multimodal/
        ...
    E11_Entity/
        ...
    E12_LateInteraction/
        ...
    E13_Splade/
        ...
```

**README.md Content:**

```markdown
# Golden Reference Test Fixtures

## Purpose

These fixtures contain golden reference outputs for inference validation.
They ensure that model loading produces correct inference results.

## File Format

### test_input.bin
Binary format:
- 4 bytes: dimension count (u32, little-endian)
- N * 4 bytes: f32 values (little-endian)

### golden_output.bin
Same binary format as test_input.bin.

### checksum.txt
SHA-256 checksum of the model weight file that produced this golden output.

## Generating Golden Data

Golden data is generated by running real inference:

```bash
cargo run --bin generate-golden -- --model E1_Semantic --output tests/fixtures/golden/E1_Semantic/
```

## Validation Tolerance

Inference output is validated against golden reference with tolerance:
- FP32: 1e-5 absolute difference
- Mixed precision: 1e-3 absolute difference
```

**Constraints:**
- Placeholder files can be empty or minimal
- Structure must match TECH-EMB-002 specification
- README must document format clearly

**Verification:**
- Directory structure exists
- README explains format
- 13 model directories present

#### Files to Create

| Directory/File | Content |
|----------------|---------|
| `tests/fixtures/golden/README.md` | Format documentation |
| `tests/fixtures/golden/E1_Semantic/` | Placeholder directory |
| `tests/fixtures/golden/E2_TemporalRecent/` | Placeholder directory |
| ... (13 total model directories) | Placeholder directories |

#### Validation Criteria

- [ ] 13 model directories created
- [ ] README.md present and documents format
- [ ] Each directory has placeholder structure

#### Test Commands

```bash
ls -la /home/cabdru/contextgraph/crates/context-graph-embeddings/tests/fixtures/golden/
find /home/cabdru/contextgraph/crates/context-graph-embeddings/tests/fixtures/golden/ -type d | wc -l
```

</task_spec>

---

</task_collection>

---

## Summary

### Task List

| Task ID | Title | Dependencies | Complexity |
|---------|-------|--------------|------------|
| TASK-EMB-001 | Fix Dimension Constants | None | low |
| TASK-EMB-002 | Create ProjectionMatrix Struct | TASK-EMB-001 | medium |
| TASK-EMB-003 | Create ProjectionError Enum | TASK-EMB-002 | low |
| TASK-EMB-004 | Create Quantization Structs | TASK-EMB-001 | medium |
| TASK-EMB-005 | Create Storage Types | TASK-EMB-004 | medium |
| TASK-EMB-006 | Create WarmLoadResult Struct | TASK-EMB-001 | medium |
| TASK-EMB-007 | Create Consolidated Error Types | TASK-EMB-003, TASK-EMB-005, TASK-EMB-006 | medium |
| TASK-EMB-008 | Update SparseVector Struct | TASK-EMB-001 | low |
| TASK-EMB-009 | Create Weight File Spec | TASK-EMB-002 | low |
| TASK-EMB-010 | Create Golden Reference Fixtures | None | low |

### Execution Order

**Critical Path:**
```
TASK-EMB-001 -> TASK-EMB-002 -> TASK-EMB-003 -> TASK-EMB-007
```

**Parallel Tracks:**
- Track A: TASK-EMB-001 -> TASK-EMB-004 -> TASK-EMB-005 -> TASK-EMB-007
- Track B: TASK-EMB-001 -> TASK-EMB-006 -> TASK-EMB-007
- Track C: TASK-EMB-001 -> TASK-EMB-008
- Track D: TASK-EMB-002 -> TASK-EMB-009
- Track E: TASK-EMB-010 (independent)

### Files Modified Summary

| File | Tasks |
|------|-------|
| `types/dimensions/constants.rs` | TASK-EMB-001 |
| `models/pretrained/sparse/types.rs` | TASK-EMB-001, TASK-EMB-008 |
| `models/pretrained/sparse/projection.rs` | TASK-EMB-002, TASK-EMB-003 |
| `quantization/types.rs` | TASK-EMB-004 |
| `storage/types.rs` | TASK-EMB-005 |
| `warm/loader/types.rs` | TASK-EMB-006 |
| `error.rs` | TASK-EMB-007 |
| `models/README.md` | TASK-EMB-009 |
| `tests/fixtures/golden/` | TASK-EMB-010 |

---

## Next Steps

After Foundation Layer completion:

1. **Logic Layer Tasks** (TASK-EMB-011 through TASK-EMB-020):
   - Implement projection loading and inference
   - Implement quantization/dequantization
   - Implement storage backend
   - Implement warm loading

2. **Surface Layer Tasks** (TASK-EMB-021 through TASK-EMB-030):
   - Integration tests
   - Performance benchmarks
   - Documentation updates
   - Migration guide

---

## Memory Key

Store this summary for next agent:
```
contextgraph/embedding-issues/task-foundation-summary
```
