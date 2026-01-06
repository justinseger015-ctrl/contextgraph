# TASK-EMB-004: Create Quantization Structs

<task_spec id="TASK-EMB-004" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-004 |
| **Title** | Create Quantization Data Structures |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 4 |
| **Implements** | REQ-EMB-005 |
| **Depends On** | TASK-EMB-001 |
| **Estimated Complexity** | medium |
| **Constitution Ref** | `embeddings.quantization`, `embeddings.quantization_by_embedder` |

---

## Context

TECH-EMB-003 specifies quantization methods per Constitutional requirements. This task creates the data structures for quantized embeddings, codebooks, and encoders. Actual quantization logic is Logic Layer.

**Why This Matters:**
- Storage per fingerprint target: ~17KB (vs 46KB uncompressed) = 63% reduction
- Different embedders need different quantization methods
- Compression ratios must match Constitution requirements

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-003-quantization.md` |
| Constitution | `docs/constitution.yaml` |
| ModelId | `crates/context-graph-embeddings/src/types/model_id/core.rs` |

---

## Prerequisites

- [ ] TASK-EMB-001 completed (dimension constants)
- [ ] Understand per-embedder quantization assignments from Constitution

---

## Scope

### In Scope
- Create `QuantizationMethod` enum with 5 variants
- Create `QuantizedEmbedding` struct
- Create `QuantizationMetadata` enum (per-method metadata)
- Create `PQ8Codebook` struct
- Create `Float8Encoder` struct (stateless)
- Create `BinaryEncoder` struct (stateless)

### Out of Scope
- Quantization implementations (Logic Layer - TASK-EMB-016, 017, 018)
- Dequantization implementations (Logic Layer)
- Codebook loading/training (Logic Layer)

---

## Definition of Done

### Exact Signatures

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

### Constraints
- Structs only, minimal method implementations
- Must use serde for serialization
- Compression ratios must match Constitution

### Verification
- All types compile
- Serde serialization works

---

## Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/quantization/types.rs` | All quantization types |
| `crates/context-graph-embeddings/src/quantization/mod.rs` | Module exports |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/lib.rs` | Add `pub mod quantization;` |

---

## Validation Criteria

- [ ] `QuantizationMethod` has 5 variants
- [ ] `for_model_id()` returns correct method for all 13 embedders
- [ ] Compression ratios match Constitution
- [ ] Max recall loss values match Constitution

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings quantization -- --nocapture
```

---

## Traceability

| Requirement | Tech Spec | Issue |
|-------------|-----------|-------|
| REQ-EMB-005 | TECH-EMB-003 | ISSUE-006 |

---

## Constitution Reference

| Method | Embedders | Compression | Max Recall Loss |
|--------|-----------|-------------|-----------------|
| PQ_8 | E1, E5, E7, E10 | 32x | <5% |
| Float8 | E2, E3, E4, E8, E11 | 4x | <0.3% |
| Binary | E9 | 32x | 5-10% |
| Sparse | E6, E13 | native | 0% |
| TokenPruning | E12 | ~50% | <2% |

</task_spec>
