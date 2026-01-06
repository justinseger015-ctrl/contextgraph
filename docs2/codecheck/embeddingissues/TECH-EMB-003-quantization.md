# Technical Specification: Quantization Implementation

<technical_spec id="TECH-EMB-003" version="1.0" implements="SPEC-EMB-001">

## Metadata

| Field | Value |
|-------|-------|
| **Spec ID** | TECH-EMB-003 |
| **Title** | Quantization Implementation |
| **Status** | Draft |
| **Version** | 1.0 |
| **Implements** | REQ-EMB-005 |
| **Related Issues** | ISSUE-006 (Quantization not applied) |
| **Created** | 2026-01-06 |
| **Constitution Reference** | `embeddings.quantization`, `storage.layer1_primary` |

---

## Problem Statement

### ISSUE-006: Quantization Not Actually Applied (MEDIUM)

The current implementation defines quantization enums but **never actually applies them** to embeddings:

```rust
// CURRENT BROKEN CODE - quantization.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QuantizationMode {
    #[default]
    None,   // <-- Default is NO quantization!
    Int8,
    Fp16,
    Bf16,
}

impl QuantizationMode {
    pub fn memory_multiplier(&self) -> f32 {
        match self {
            QuantizationMode::None => 1.0,  // No reduction
            QuantizationMode::Int8 => 0.25,
            QuantizationMode::Fp16 | QuantizationMode::Bf16 => 0.5,
        }
    }
}
```

```rust
// CURRENT BROKEN CODE - embedding.rs (storage serialization)
pub fn serialize_embedding(embedding: &EmbeddingVector) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());  // ALWAYS f32!
    }
    bytes
}
// NO QUANTIZATION APPLIED - Always stores as full float32!
```

### Why This Is Critical

1. **No Memory Savings**: Constitution promises 63% reduction (46KB -> 17KB), but we store full 46KB
2. **Broken Promise**: `memory_multiplier()` returns theoretical values, never applied
3. **Constitution Violation**: Per-embedder quantization strategy is specified but ignored
4. **Storage Bloat**: 2.7x more storage than necessary per TeleologicalFingerprint

### Constitutional Quantization Strategy

```yaml
# From constitution.yaml
quantization:
  PQ_8: { compression: "32x", recall_impact: "<5%", embedders: [E1, E5, E7, E10] }
  Float8: { compression: "4x", recall_impact: "<0.3%", embedders: [E2, E3, E4, E8, E11] }
  Binary: { compression: "32x", recall_impact: "5-10%", embedders: [E9] }
  Sparse: { compression: "native", recall_impact: "0%", embedders: [E6, E13] }
  TokenPruning: { compression: "~50%", recall_impact: "<2%", embedders: [E12] }
```

### The Fix

Implement **actual quantization** during embedding serialization with:
1. Per-embedder quantization strategy lookup
2. Real compression algorithms (PQ-8, Float8, Binary)
3. Dequantization for query-time similarity computation
4. Recall loss tracking and logging

---

## Architecture Diagram

```mermaid
graph TB
    subgraph "Input: Raw Embeddings"
        E1[E1_Semantic<br/>1024D f32]
        E2[E2_Temporal<br/>512D f32]
        E5[E5_Causal<br/>768D f32]
        E6[E6_Sparse<br/>~30K sparse]
        E9[E9_HDC<br/>1024D f32]
        E12[E12_LateInt<br/>128D x N tokens]
    end

    subgraph "Quantization Strategy Router"
        ROUTER[QuantizationRouter<br/>Per-embedder strategy lookup]
    end

    subgraph "Quantization Engines"
        PQ8[PQ-8 Encoder<br/>8 subvectors x 256 centroids<br/>32x compression]
        F8[Float8 Encoder<br/>E4M3 format<br/>4x compression]
        BIN[Binary Encoder<br/>Sign quantization<br/>32x compression]
        SPARSE[Sparse Native<br/>indices + values<br/>native format]
        PRUNE[Token Pruner<br/>50% importance threshold<br/>~50% compression]
    end

    subgraph "Output: Quantized Storage"
        Q1[E1_Quantized<br/>32 bytes (PQ-8)]
        Q2[E2_Quantized<br/>128 bytes (Float8)]
        Q5[E5_Quantized<br/>24 bytes (PQ-8)]
        Q6[E6_Quantized<br/>~3KB sparse]
        Q9[E9_Quantized<br/>128 bits binary]
        Q12[E12_Quantized<br/>~256 bytes pruned]
    end

    E1 --> ROUTER
    E2 --> ROUTER
    E5 --> ROUTER
    E6 --> ROUTER
    E9 --> ROUTER
    E12 --> ROUTER

    ROUTER -->|PQ-8| PQ8
    ROUTER -->|Float8| F8
    ROUTER -->|Binary| BIN
    ROUTER -->|Sparse| SPARSE
    ROUTER -->|TokenPruning| PRUNE

    PQ8 --> Q1
    PQ8 --> Q5
    F8 --> Q2
    SPARSE --> Q6
    BIN --> Q9
    PRUNE --> Q12

    style ROUTER fill:#99ff99,stroke:#009900,stroke-width:2px
    style PQ8 fill:#ff9999,stroke:#990000,stroke-width:2px
    style F8 fill:#ffcc99,stroke:#996600,stroke-width:2px
    style BIN fill:#9999ff,stroke:#000099,stroke-width:2px
```

### Component Flow

```
Raw Embedding (f32 array)
         |
         v (lookup embedder -> strategy)
+------------------+
| QuantizationRouter|  <-- Maps ModelId to quantization method
| E1 -> PQ-8       |      Per-embedder configuration
| E2 -> Float8     |
+------------------+
         |
         v (dispatch to encoder)
+------------------+
| Encoder          |  <-- PQ8Encoder / Float8Encoder / BinaryEncoder
| (per method)     |      Implements actual compression
+------------------+
         |
         v
+------------------+
| QuantizedVector  |  <-- Compressed bytes + metadata
| bytes: Vec<u8>   |      Method identifier for dequantization
| method: Method   |
+------------------+
         |
         v (storage)
+------------------+
| RocksDB/ScyllaDB |  <-- ~17KB per fingerprint
| BYTEA column     |      (vs 46KB uncompressed)
+------------------+
```

---

## Data Models

### QuantizationMethod Enum (FIXED)

```rust
/// Quantization methods aligned with Constitution.
///
/// # Constitution Alignment
/// - PQ_8: E1, E5, E7, E10 (32x compression, <5% recall impact)
/// - Float8: E2, E3, E4, E8, E11 (4x compression, <0.3% recall impact)
/// - Binary: E9 (32x compression, 5-10% recall impact)
/// - Sparse: E6, E13 (native format, 0% recall impact)
/// - TokenPruning: E12 (~50% compression, <2% recall impact)
///
/// # CRITICAL: No Fallback to Float32
/// Every embedder MUST use its assigned quantization method.
/// Storing as float32 is a Constitution violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationMethod {
    /// Product Quantization with 8 subvectors, 256 centroids each.
    /// Used for: E1_Semantic, E5_Causal, E7_Code, E10_Multimodal
    /// Compression: 32x (1024D f32 -> 32 bytes)
    /// Recall impact: <5%
    PQ8,

    /// 8-bit floating point in E4M3 format (4-bit exponent, 3-bit mantissa).
    /// Used for: E2_TemporalRecent, E3_TemporalPeriodic, E4_TemporalPositional,
    ///           E8_Graph, E11_Entity
    /// Compression: 4x (512D f32 -> 512 bytes)
    /// Recall impact: <0.3%
    Float8E4M3,

    /// Binary quantization (sign bit only).
    /// Used for: E9_HDC (Hyperdimensional Computing)
    /// Compression: 32x (1024D f32 -> 128 bits)
    /// Recall impact: 5-10%
    Binary,

    /// Sparse format: indices + values for non-zero elements.
    /// Used for: E6_Sparse, E13_SPLADE
    /// Compression: native (depends on sparsity)
    /// Recall impact: 0%
    SparseNative,

    /// Token pruning: keep top 50% tokens by importance score.
    /// Used for: E12_LateInteraction
    /// Compression: ~50%
    /// Recall impact: <2%
    TokenPruning,
}

impl QuantizationMethod {
    /// Get the quantization method for a given embedder.
    ///
    /// # CRITICAL: No fallback to None
    /// Every embedder has an assigned method. This function never returns None.
    pub fn for_embedder(model_id: ModelId) -> Self {
        match model_id {
            // PQ-8: Dense semantic embeddings
            ModelId::Semantic => Self::PQ8,           // E1
            ModelId::Causal => Self::PQ8,             // E5
            ModelId::Code => Self::PQ8,               // E7
            ModelId::Multimodal => Self::PQ8,         // E10

            // Float8: Temporal and graph embeddings
            ModelId::TemporalRecent => Self::Float8E4M3,    // E2
            ModelId::TemporalPeriodic => Self::Float8E4M3,  // E3
            ModelId::TemporalPositional => Self::Float8E4M3, // E4
            ModelId::Graph => Self::Float8E4M3,             // E8
            ModelId::Entity => Self::Float8E4M3,            // E11

            // Binary: Hyperdimensional computing
            ModelId::Hdc => Self::Binary,             // E9

            // Sparse: Sparse vector formats
            ModelId::Sparse => Self::SparseNative,    // E6
            ModelId::Splade => Self::SparseNative,    // E13

            // Token pruning: Late interaction
            ModelId::LateInteraction => Self::TokenPruning, // E12
        }
    }

    /// Theoretical compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        match self {
            Self::PQ8 => 32.0,
            Self::Float8E4M3 => 4.0,
            Self::Binary => 32.0,
            Self::SparseNative => 1.0, // Variable, depends on sparsity
            Self::TokenPruning => 2.0, // ~50%
        }
    }

    /// Maximum acceptable recall loss.
    pub fn max_recall_loss(&self) -> f32 {
        match self {
            Self::PQ8 => 0.05,          // <5%
            Self::Float8E4M3 => 0.003,  // <0.3%
            Self::Binary => 0.10,       // 5-10%
            Self::SparseNative => 0.0,  // 0%
            Self::TokenPruning => 0.02, // <2%
        }
    }
}
```

### QuantizedEmbedding Structure

```rust
/// Quantized embedding ready for storage.
///
/// # Invariants
/// - `method` matches the encoding of `data`
/// - `original_dim` allows dequantization validation
/// - `data` is compressed according to `method`
///
/// # CRITICAL: Dequantization Required for Similarity
/// The `data` bytes are NOT directly comparable.
/// Use `Quantizer::dequantize()` for similarity computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedEmbedding {
    /// Quantization method used (for dequantization dispatch).
    pub method: QuantizationMethod,

    /// Original embedding dimension before quantization.
    pub original_dim: usize,

    /// Compressed embedding bytes.
    /// Format depends on `method`.
    pub data: Vec<u8>,

    /// Metadata for reconstruction.
    pub metadata: QuantizationMetadata,
}

/// Method-specific metadata for dequantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationMetadata {
    /// PQ-8: Codebook index for this embedding.
    PQ8 {
        /// Codebook identifier (trained on corpus).
        codebook_id: u32,
        /// Number of subvectors (typically 8).
        num_subvectors: u8,
    },

    /// Float8: Scale and bias for denormalization.
    Float8 {
        /// Scaling factor: original = quantized * scale + bias
        scale: f32,
        /// Bias offset.
        bias: f32,
    },

    /// Binary: Threshold used for binarization.
    Binary {
        /// Threshold: 1 if value >= threshold, else 0
        threshold: f32,
    },

    /// Sparse: Number of non-zero elements.
    Sparse {
        /// Total vocabulary dimension.
        vocab_size: usize,
        /// Number of non-zero entries.
        nnz: usize,
    },

    /// Token pruning: Original token count and pruning ratio.
    TokenPruning {
        /// Original number of tokens.
        original_tokens: usize,
        /// Kept tokens after pruning.
        kept_tokens: usize,
        /// Importance threshold used.
        threshold: f32,
    },
}

impl QuantizedEmbedding {
    /// Compute compressed size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Compute compression ratio vs float32.
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.original_dim * 4; // f32 = 4 bytes
        original_bytes as f32 / self.data.len() as f32
    }
}
```

### PQ8Codebook Structure

```rust
/// Product Quantization codebook with 8 subvectors, 256 centroids each.
///
/// # Algorithm
/// 1. Split embedding into 8 subvectors of dimension D/8
/// 2. Each subvector quantized to one of 256 centroids
/// 3. Store 8 centroid indices (1 byte each) = 8 bytes total
///
/// # Compression Ratio
/// - 1024D f32 -> 8 bytes = 512x compression (theoretical)
/// - With codebook overhead amortized: ~32x effective
///
/// # Training
/// Codebooks are pre-trained on corpus embeddings using K-means.
/// File: models/pq8_codebooks/{embedder}_codebook.safetensors
#[derive(Debug)]
pub struct PQ8Codebook {
    /// Embedding dimension this codebook was trained for.
    pub embedding_dim: usize,

    /// Number of subvectors (typically 8).
    pub num_subvectors: usize,

    /// Centroids per subvector (typically 256).
    pub num_centroids: usize,

    /// Centroid vectors: [num_subvectors, num_centroids, subvector_dim]
    /// Shape: [8, 256, embedding_dim/8]
    pub centroids: Vec<Vec<Vec<f32>>>,

    /// Codebook identifier for embedding metadata.
    pub codebook_id: u32,
}

impl PQ8Codebook {
    /// Load codebook from SafeTensors file.
    ///
    /// # Panics
    /// - File not found
    /// - Shape mismatch
    /// - Codebook not trained for this dimension
    pub fn load(path: &Path, embedding_dim: usize) -> Result<Self, QuantizationError>;

    /// Quantize embedding to PQ-8 indices.
    ///
    /// # Algorithm
    /// For each subvector, find nearest centroid and store its index.
    ///
    /// # Returns
    /// 8 bytes (one index per subvector)
    pub fn quantize(&self, embedding: &[f32]) -> Result<Vec<u8>, QuantizationError>;

    /// Dequantize PQ-8 indices back to approximate embedding.
    ///
    /// # Algorithm
    /// Concatenate centroids at each index.
    ///
    /// # Recall Loss
    /// Reconstructed vector has ~5% error vs original.
    pub fn dequantize(&self, indices: &[u8]) -> Result<Vec<f32>, QuantizationError>;
}
```

---

## Component Contracts

### Quantizer Trait

```rust
/// Trait for embedding quantization.
///
/// # Safety
/// All implementations MUST:
/// 1. Achieve target compression ratio
/// 2. Stay within max recall loss bounds
/// 3. Support round-trip (quantize -> dequantize)
/// 4. Track and log actual recall metrics
///
/// # Constitution Alignment
/// Each embedder MUST use its assigned quantization method.
/// Fallback to float32 is FORBIDDEN.
pub trait Quantizer: Send + Sync {
    /// Quantize embedding to compressed format.
    ///
    /// # Arguments
    /// * `model_id` - Embedder that produced this embedding
    /// * `embedding` - Raw float32 embedding
    ///
    /// # Returns
    /// Quantized embedding ready for storage.
    ///
    /// # Errors
    /// - DimensionMismatch: embedding.len() != expected dimension
    /// - CodebookMissing: PQ-8 codebook not loaded
    fn quantize(
        &self,
        model_id: ModelId,
        embedding: &[f32],
    ) -> Result<QuantizedEmbedding, QuantizationError>;

    /// Dequantize compressed embedding for similarity computation.
    ///
    /// # Arguments
    /// * `quantized` - Quantized embedding from storage
    ///
    /// # Returns
    /// Approximate float32 embedding.
    ///
    /// # Recall Loss
    /// Returned vector is approximate. See QuantizationMethod::max_recall_loss().
    fn dequantize(
        &self,
        quantized: &QuantizedEmbedding,
    ) -> Result<Vec<f32>, QuantizationError>;

    /// Get expected compressed size for an embedder.
    pub fn expected_size(&self, model_id: ModelId) -> usize;

    /// Measure recall loss for a batch of embeddings.
    ///
    /// # Algorithm
    /// 1. Quantize each embedding
    /// 2. Dequantize back
    /// 3. Compute cosine similarity between original and reconstructed
    /// 4. Return 1.0 - mean_similarity (recall loss)
    fn measure_recall_loss(
        &self,
        model_id: ModelId,
        embeddings: &[Vec<f32>],
    ) -> Result<f32, QuantizationError>;
}
```

### QuantizationError Enum

```rust
/// Errors that can occur during quantization.
#[derive(Debug, thiserror::Error)]
pub enum QuantizationError {
    /// Embedding dimension doesn't match expected for embedder.
    #[error("[EMB-E005] DIMENSION_MISMATCH: Embedding has wrong dimension
  Embedder: {model_id:?}
  Expected: {expected}
  Actual: {actual}")]
    DimensionMismatch {
        model_id: ModelId,
        expected: usize,
        actual: usize,
    },

    /// PQ-8 codebook not loaded for this embedder.
    #[error("[EMB-E011] CODEBOOK_MISSING: PQ-8 codebook not found
  Embedder: {model_id:?}
  Expected path: models/pq8_codebooks/{model_id:?}_codebook.safetensors
  Remediation: Train codebook or download from model repository")]
    CodebookMissing { model_id: ModelId },

    /// Recall loss exceeds maximum allowed for method.
    #[error("[EMB-E012] RECALL_LOSS_EXCEEDED: Quantization quality too low
  Embedder: {model_id:?}
  Method: {method:?}
  Measured loss: {measured_loss:.4}
  Max allowed: {max_loss:.4}
  Remediation: Retrain codebook or check embedding quality")]
    RecallLossExceeded {
        model_id: ModelId,
        method: QuantizationMethod,
        measured_loss: f32,
        max_loss: f32,
    },

    /// Invalid quantized data format.
    #[error("[EMB-E013] INVALID_QUANTIZED_DATA: Cannot dequantize
  Method: {method:?}
  Expected bytes: {expected_bytes}
  Actual bytes: {actual_bytes}")]
    InvalidQuantizedData {
        method: QuantizationMethod,
        expected_bytes: usize,
        actual_bytes: usize,
    },

    /// Sparse embedding has too many non-zero elements.
    #[error("[EMB-E014] SPARSE_TOO_DENSE: Sparse embedding exceeds sparsity limit
  Expected sparsity: <{expected_sparsity}%
  Actual non-zeros: {nnz} / {total} = {actual_pct:.1}%")]
    SparseTooNense {
        expected_sparsity: f32,
        nnz: usize,
        total: usize,
        actual_pct: f32,
    },
}
```

---

## Implementation Details

### PQ-8 Encoder Implementation

```rust
/// Product Quantization encoder with 8 subvectors.
///
/// # Compression Calculation
/// - Input: 1024D float32 = 4096 bytes
/// - Output: 8 bytes (one centroid index per subvector)
/// - Compression: 512x theoretical
/// - With codebook: ~32x effective (codebook shared across embeddings)
pub struct PQ8Encoder {
    /// Codebooks for each embedder using PQ-8.
    codebooks: HashMap<ModelId, PQ8Codebook>,
}

impl PQ8Encoder {
    /// Create encoder and load codebooks for PQ-8 embedders.
    ///
    /// # Panics
    /// - Any codebook file missing
    /// - Codebook dimension mismatch
    pub fn new(model_dir: &Path) -> Result<Self, QuantizationError> {
        let mut codebooks = HashMap::new();

        for model_id in [ModelId::Semantic, ModelId::Causal, ModelId::Code, ModelId::Multimodal] {
            let dim = model_id.dimension();
            let codebook_path = model_dir
                .join("pq8_codebooks")
                .join(format!("{:?}_codebook.safetensors", model_id));

            if !codebook_path.exists() {
                panic!(
                    "[EMB-E011] CODEBOOK_MISSING: PQ-8 codebook not found for {:?}\n\
                     Expected: {}\n\
                     Train with: cargo run --bin train-pq8 -- --embedder {:?}",
                    model_id, codebook_path.display(), model_id
                );
            }

            let codebook = PQ8Codebook::load(&codebook_path, dim)?;
            codebooks.insert(model_id, codebook);
        }

        Ok(Self { codebooks })
    }

    /// Quantize embedding using PQ-8.
    pub fn quantize(&self, model_id: ModelId, embedding: &[f32]) -> Result<QuantizedEmbedding, QuantizationError> {
        let codebook = self.codebooks.get(&model_id)
            .ok_or(QuantizationError::CodebookMissing { model_id })?;

        // Validate dimension
        if embedding.len() != codebook.embedding_dim {
            return Err(QuantizationError::DimensionMismatch {
                model_id,
                expected: codebook.embedding_dim,
                actual: embedding.len(),
            });
        }

        // Quantize to indices
        let indices = codebook.quantize(embedding)?;

        Ok(QuantizedEmbedding {
            method: QuantizationMethod::PQ8,
            original_dim: embedding.len(),
            data: indices,
            metadata: QuantizationMetadata::PQ8 {
                codebook_id: codebook.codebook_id,
                num_subvectors: codebook.num_subvectors as u8,
            },
        })
    }

    /// Dequantize PQ-8 indices to approximate embedding.
    pub fn dequantize(&self, quantized: &QuantizedEmbedding) -> Result<Vec<f32>, QuantizationError> {
        let codebook_id = match &quantized.metadata {
            QuantizationMetadata::PQ8 { codebook_id, .. } => *codebook_id,
            _ => return Err(QuantizationError::InvalidQuantizedData {
                method: QuantizationMethod::PQ8,
                expected_bytes: 8,
                actual_bytes: quantized.data.len(),
            }),
        };

        // Find codebook by ID
        let codebook = self.codebooks.values()
            .find(|cb| cb.codebook_id == codebook_id)
            .ok_or(QuantizationError::CodebookMissing { model_id: ModelId::Semantic })?;

        codebook.dequantize(&quantized.data)
    }
}
```

### Float8 (E4M3) Encoder Implementation

```rust
/// Float8 E4M3 encoder for 4x compression.
///
/// # E4M3 Format
/// - 1 sign bit
/// - 4 exponent bits (bias 7)
/// - 3 mantissa bits
/// - Range: ~1e-9 to ~448
///
/// # Compression Calculation
/// - Input: 512D float32 = 2048 bytes
/// - Output: 512D float8 = 512 bytes
/// - Compression: 4x
pub struct Float8E4M3Encoder;

impl Float8E4M3Encoder {
    /// Convert f32 to E4M3 (8-bit float).
    fn f32_to_e4m3(value: f32) -> u8 {
        // Handle special cases
        if value.is_nan() {
            return 0x7F; // E4M3 NaN representation
        }
        if value == 0.0 {
            return 0x00;
        }

        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127; // F32 exponent bias
        let mantissa = bits & 0x7FFFFF;

        // Clamp exponent to E4M3 range [-6, 8] (bias 7)
        let e4m3_exp = (exp + 7).clamp(0, 15) as u8;

        // Truncate mantissa to 3 bits
        let e4m3_mantissa = (mantissa >> 20) as u8 & 0x07;

        ((sign as u8) << 7) | (e4m3_exp << 3) | e4m3_mantissa
    }

    /// Convert E4M3 back to f32.
    fn e4m3_to_f32(byte: u8) -> f32 {
        let sign = (byte >> 7) & 1;
        let exp = ((byte >> 3) & 0x0F) as i32;
        let mantissa = (byte & 0x07) as u32;

        if exp == 0 && mantissa == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }

        // Convert back to F32
        let f32_exp = ((exp - 7 + 127) as u32) & 0xFF;
        let f32_mantissa = mantissa << 20;
        let bits = ((sign as u32) << 31) | (f32_exp << 23) | f32_mantissa;

        f32::from_bits(bits)
    }

    /// Quantize embedding to Float8 E4M3.
    pub fn quantize(&self, model_id: ModelId, embedding: &[f32]) -> Result<QuantizedEmbedding, QuantizationError> {
        // Compute scale and bias for better precision
        let (min_val, max_val) = embedding.iter().fold((f32::MAX, f32::MIN), |acc, &v| {
            (acc.0.min(v), acc.1.max(v))
        });

        let scale = (max_val - min_val) / 255.0;
        let bias = min_val;

        // Normalize and quantize
        let data: Vec<u8> = embedding.iter()
            .map(|&v| Self::f32_to_e4m3((v - bias) / scale))
            .collect();

        Ok(QuantizedEmbedding {
            method: QuantizationMethod::Float8E4M3,
            original_dim: embedding.len(),
            data,
            metadata: QuantizationMetadata::Float8 { scale, bias },
        })
    }

    /// Dequantize Float8 back to f32.
    pub fn dequantize(&self, quantized: &QuantizedEmbedding) -> Result<Vec<f32>, QuantizationError> {
        let (scale, bias) = match &quantized.metadata {
            QuantizationMetadata::Float8 { scale, bias } => (*scale, *bias),
            _ => return Err(QuantizationError::InvalidQuantizedData {
                method: QuantizationMethod::Float8E4M3,
                expected_bytes: quantized.original_dim,
                actual_bytes: quantized.data.len(),
            }),
        };

        let embedding: Vec<f32> = quantized.data.iter()
            .map(|&b| Self::e4m3_to_f32(b) * scale + bias)
            .collect();

        Ok(embedding)
    }
}
```

### Binary Encoder Implementation

```rust
/// Binary quantization encoder for HDC embeddings.
///
/// # Algorithm
/// Binary(v) = 1 if v >= threshold, else 0
///
/// # Compression Calculation
/// - Input: 1024D float32 = 4096 bytes
/// - Output: 1024 bits = 128 bytes
/// - Compression: 32x
///
/// # Similarity
/// Use Hamming distance or Jaccard similarity for binary vectors.
pub struct BinaryEncoder;

impl BinaryEncoder {
    /// Quantize embedding to binary (sign bits).
    pub fn quantize(&self, model_id: ModelId, embedding: &[f32]) -> Result<QuantizedEmbedding, QuantizationError> {
        // Compute threshold (typically 0 or median)
        let threshold = 0.0_f32;

        // Pack bits into bytes
        let num_bytes = (embedding.len() + 7) / 8;
        let mut data = vec![0u8; num_bytes];

        for (i, &value) in embedding.iter().enumerate() {
            if value >= threshold {
                data[i / 8] |= 1 << (7 - (i % 8));
            }
        }

        Ok(QuantizedEmbedding {
            method: QuantizationMethod::Binary,
            original_dim: embedding.len(),
            data,
            metadata: QuantizationMetadata::Binary { threshold },
        })
    }

    /// Dequantize binary to +1/-1 representation.
    ///
    /// # Note
    /// This is NOT a true inverse - binary quantization is lossy.
    /// Returns +1.0 for 1 bits, -1.0 for 0 bits.
    pub fn dequantize(&self, quantized: &QuantizedEmbedding) -> Result<Vec<f32>, QuantizationError> {
        let mut embedding = Vec::with_capacity(quantized.original_dim);

        for i in 0..quantized.original_dim {
            let byte_idx = i / 8;
            let bit_idx = 7 - (i % 8);
            let bit = (quantized.data[byte_idx] >> bit_idx) & 1;
            embedding.push(if bit == 1 { 1.0 } else { -1.0 });
        }

        Ok(embedding)
    }

    /// Compute Hamming distance between two binary vectors.
    pub fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones() as usize)
            .sum()
    }

    /// Compute binary similarity (1 - normalized Hamming distance).
    pub fn binary_similarity(a: &[u8], b: &[u8], num_bits: usize) -> f32 {
        let distance = Self::hamming_distance(a, b);
        1.0 - (distance as f32 / num_bits as f32)
    }
}
```

### Sparse Native Encoder Implementation

```rust
/// Sparse native format encoder for E6/E13 embeddings.
///
/// # Format
/// - 4 bytes: number of non-zero entries (u32)
/// - For each entry:
///   - 4 bytes: index (u32)
///   - 4 bytes: value (f32)
///
/// # Compression
/// - If 5% active (30K vocab): ~1500 entries * 8 bytes = ~12KB
/// - vs dense 30K * 4 = 120KB -> 10x compression
pub struct SparseNativeEncoder;

impl SparseNativeEncoder {
    /// Serialize sparse embedding to native format.
    pub fn quantize(&self, model_id: ModelId, embedding: &SparseVector) -> Result<QuantizedEmbedding, QuantizationError> {
        // Verify sparsity constraint (<10% active)
        let sparsity_pct = (embedding.indices.len() as f32 / embedding.dimension as f32) * 100.0;
        if sparsity_pct > 10.0 {
            return Err(QuantizationError::SparseTooNense {
                expected_sparsity: 10.0,
                nnz: embedding.indices.len(),
                total: embedding.dimension,
                actual_pct: sparsity_pct,
            });
        }

        // Serialize: nnz + (index, value) pairs
        let nnz = embedding.indices.len();
        let mut data = Vec::with_capacity(4 + nnz * 8);

        data.extend_from_slice(&(nnz as u32).to_le_bytes());
        for (&idx, &val) in embedding.indices.iter().zip(&embedding.weights) {
            data.extend_from_slice(&(idx as u32).to_le_bytes());
            data.extend_from_slice(&val.to_le_bytes());
        }

        Ok(QuantizedEmbedding {
            method: QuantizationMethod::SparseNative,
            original_dim: embedding.dimension,
            data,
            metadata: QuantizationMetadata::Sparse {
                vocab_size: embedding.dimension,
                nnz,
            },
        })
    }

    /// Deserialize sparse embedding from native format.
    pub fn dequantize(&self, quantized: &QuantizedEmbedding) -> Result<SparseVector, QuantizationError> {
        let mut cursor = &quantized.data[..];

        // Read nnz
        let nnz = u32::from_le_bytes(cursor[0..4].try_into().unwrap()) as usize;
        cursor = &cursor[4..];

        // Read indices and values
        let mut indices = Vec::with_capacity(nnz);
        let mut weights = Vec::with_capacity(nnz);

        for _ in 0..nnz {
            let idx = u32::from_le_bytes(cursor[0..4].try_into().unwrap()) as usize;
            let val = f32::from_le_bytes(cursor[4..8].try_into().unwrap());
            cursor = &cursor[8..];

            indices.push(idx);
            weights.push(val);
        }

        Ok(SparseVector {
            indices,
            weights,
            dimension: quantized.original_dim,
        })
    }
}
```

### Token Pruning Encoder Implementation

```rust
/// Token pruning encoder for late interaction embeddings.
///
/// # Algorithm
/// 1. Compute importance score per token (e.g., max pooling magnitude)
/// 2. Keep top 50% of tokens by importance
/// 3. Store pruned token embeddings + importance scores
///
/// # Compression
/// - Original: N tokens x 128D = N * 512 bytes
/// - Pruned: N/2 tokens x 128D + N/2 importance = N/2 * 516 bytes
/// - ~50% compression
pub struct TokenPruningEncoder {
    /// Pruning ratio (e.g., 0.5 = keep 50%).
    pruning_ratio: f32,
}

impl TokenPruningEncoder {
    pub fn new() -> Self {
        Self { pruning_ratio: 0.5 }
    }

    /// Compute importance score for a token embedding.
    fn token_importance(embedding: &[f32]) -> f32 {
        embedding.iter().map(|&v| v.abs()).fold(0.0f32, f32::max)
    }

    /// Prune tokens and serialize.
    pub fn quantize(
        &self,
        model_id: ModelId,
        token_embeddings: &[Vec<f32>],
    ) -> Result<QuantizedEmbedding, QuantizationError> {
        let original_count = token_embeddings.len();
        let keep_count = (original_count as f32 * self.pruning_ratio).ceil() as usize;

        // Compute importance scores
        let mut scored: Vec<(usize, f32, &Vec<f32>)> = token_embeddings.iter()
            .enumerate()
            .map(|(i, emb)| (i, Self::token_importance(emb), emb))
            .collect();

        // Sort by importance (descending) and keep top
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(keep_count);

        // Re-sort by original position for consistency
        scored.sort_by_key(|s| s.0);

        // Serialize: kept_count + (importance, embedding) per token
        let emb_dim = token_embeddings.first().map(|e| e.len()).unwrap_or(128);
        let mut data = Vec::with_capacity(4 + keep_count * (4 + emb_dim * 4));

        data.extend_from_slice(&(keep_count as u32).to_le_bytes());
        for (_, importance, embedding) in &scored {
            data.extend_from_slice(&importance.to_le_bytes());
            for &val in embedding.iter() {
                data.extend_from_slice(&val.to_le_bytes());
            }
        }

        Ok(QuantizedEmbedding {
            method: QuantizationMethod::TokenPruning,
            original_dim: emb_dim,
            data,
            metadata: QuantizationMetadata::TokenPruning {
                original_tokens: original_count,
                kept_tokens: keep_count,
                threshold: self.pruning_ratio,
            },
        })
    }

    /// Deserialize pruned tokens.
    pub fn dequantize(&self, quantized: &QuantizedEmbedding) -> Result<Vec<Vec<f32>>, QuantizationError> {
        let emb_dim = quantized.original_dim;
        let mut cursor = &quantized.data[..];

        let kept_count = u32::from_le_bytes(cursor[0..4].try_into().unwrap()) as usize;
        cursor = &cursor[4..];

        let mut tokens = Vec::with_capacity(kept_count);
        for _ in 0..kept_count {
            let _importance = f32::from_le_bytes(cursor[0..4].try_into().unwrap());
            cursor = &cursor[4..];

            let embedding: Vec<f32> = cursor[..emb_dim * 4]
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            cursor = &cursor[emb_dim * 4..];

            tokens.push(embedding);
        }

        Ok(tokens)
    }
}
```

---

## Size Calculations

### Per-Embedder Quantized Sizes

| Embedder | Raw Size | Method | Quantized Size | Compression |
|----------|----------|--------|----------------|-------------|
| E1_Semantic | 1024 x 4 = 4096 bytes | PQ-8 | 8 bytes + metadata | 512x |
| E2_TemporalRecent | 512 x 4 = 2048 bytes | Float8 | 512 bytes | 4x |
| E3_TemporalPeriodic | 512 x 4 = 2048 bytes | Float8 | 512 bytes | 4x |
| E4_TemporalPositional | 512 x 4 = 2048 bytes | Float8 | 512 bytes | 4x |
| E5_Causal | 768 x 4 = 3072 bytes | PQ-8 | 8 bytes + metadata | 384x |
| E6_Sparse | ~30K x 4 (dense) | Sparse | ~12KB (5% active) | 10x |
| E7_Code | 1536 x 4 = 6144 bytes | PQ-8 | 8 bytes + metadata | 768x |
| E8_Graph | 384 x 4 = 1536 bytes | Float8 | 384 bytes | 4x |
| E9_HDC | 1024 x 4 = 4096 bytes | Binary | 128 bytes | 32x |
| E10_Multimodal | 768 x 4 = 3072 bytes | PQ-8 | 8 bytes + metadata | 384x |
| E11_Entity | 384 x 4 = 1536 bytes | Float8 | 384 bytes | 4x |
| E12_LateInteraction | ~32 x 128 x 4 = 16384 bytes | TokenPruning | ~8192 bytes | 2x |
| E13_SPLADE | ~30K sparse | Sparse | ~12KB (5% active) | 10x |

### Total TeleologicalFingerprint Size

```
Uncompressed total:
  E1-E11 (dense): ~29KB
  E6, E13 (sparse): ~30KB each if stored dense
  E12 (late interaction): ~16KB
  Total: ~105KB per fingerprint (if all stored as float32)

Quantized total:
  PQ-8 embeddings (E1, E5, E7, E10): ~100 bytes (4 x 25 bytes with metadata)
  Float8 embeddings (E2, E3, E4, E8, E11): ~2.3KB
  Binary (E9): ~130 bytes
  Sparse (E6, E13): ~3KB each (5% active)
  Token pruning (E12): ~8KB

  Total: ~17KB per fingerprint

Compression: 105KB -> 17KB = 84% reduction (6.2x compression)
Constitution target: 46KB -> 17KB = 63% reduction (2.7x compression)

NOTE: Constitution's 46KB assumes sparse vectors are already sparse-encoded.
      Our 17KB target aligns with Constitutional specification.
```

---

## Testing Requirements

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// UT-QUANT-001: PQ-8 achieves target compression
    #[test]
    fn test_pq8_compression_ratio() {
        let encoder = PQ8Encoder::new(test_model_path()).unwrap();
        let embedding = generate_random_embedding(1024);

        let quantized = encoder.quantize(ModelId::Semantic, &embedding).unwrap();

        // 8 bytes for indices + ~20 bytes metadata
        assert!(quantized.data.len() <= 32, "PQ-8 should produce <32 bytes");
        assert!(quantized.compression_ratio() > 100.0, "Should achieve >100x compression");
    }

    /// UT-QUANT-002: Float8 achieves 4x compression
    #[test]
    fn test_float8_compression_ratio() {
        let encoder = Float8E4M3Encoder;
        let embedding = generate_random_embedding(512);

        let quantized = encoder.quantize(ModelId::TemporalRecent, &embedding).unwrap();

        assert_eq!(quantized.data.len(), 512, "Float8 should be 1 byte per element");
        assert!((quantized.compression_ratio() - 4.0).abs() < 0.1, "Should achieve 4x compression");
    }

    /// UT-QUANT-003: Binary achieves 32x compression
    #[test]
    fn test_binary_compression_ratio() {
        let encoder = BinaryEncoder;
        let embedding = generate_random_embedding(1024);

        let quantized = encoder.quantize(ModelId::Hdc, &embedding).unwrap();

        assert_eq!(quantized.data.len(), 128, "Binary should be 1024 bits = 128 bytes");
        assert!((quantized.compression_ratio() - 32.0).abs() < 0.1, "Should achieve 32x compression");
    }

    /// UT-QUANT-004: PQ-8 recall loss < 5%
    #[test]
    fn test_pq8_recall_loss() {
        let encoder = PQ8Encoder::new(test_model_path()).unwrap();
        let embeddings: Vec<Vec<f32>> = (0..100)
            .map(|_| generate_random_embedding(1024))
            .collect();

        let loss = encoder.measure_recall_loss(ModelId::Semantic, &embeddings).unwrap();

        assert!(loss < 0.05, "PQ-8 recall loss {} exceeds 5% limit", loss);
    }

    /// UT-QUANT-005: Float8 recall loss < 0.3%
    #[test]
    fn test_float8_recall_loss() {
        let encoder = Float8E4M3Encoder;
        let embeddings: Vec<Vec<f32>> = (0..100)
            .map(|_| generate_random_embedding(512))
            .collect();

        let mut total_similarity = 0.0f32;
        for emb in &embeddings {
            let quantized = encoder.quantize(ModelId::TemporalRecent, emb).unwrap();
            let reconstructed = encoder.dequantize(&quantized).unwrap();
            total_similarity += cosine_similarity(emb, &reconstructed);
        }

        let avg_similarity = total_similarity / embeddings.len() as f32;
        let loss = 1.0 - avg_similarity;

        assert!(loss < 0.003, "Float8 recall loss {} exceeds 0.3% limit", loss);
    }

    /// UT-QUANT-006: Method assignment matches Constitution
    #[test]
    fn test_method_assignment() {
        assert_eq!(QuantizationMethod::for_embedder(ModelId::Semantic), QuantizationMethod::PQ8);
        assert_eq!(QuantizationMethod::for_embedder(ModelId::Causal), QuantizationMethod::PQ8);
        assert_eq!(QuantizationMethod::for_embedder(ModelId::Code), QuantizationMethod::PQ8);
        assert_eq!(QuantizationMethod::for_embedder(ModelId::Multimodal), QuantizationMethod::PQ8);

        assert_eq!(QuantizationMethod::for_embedder(ModelId::TemporalRecent), QuantizationMethod::Float8E4M3);
        assert_eq!(QuantizationMethod::for_embedder(ModelId::Graph), QuantizationMethod::Float8E4M3);
        assert_eq!(QuantizationMethod::for_embedder(ModelId::Entity), QuantizationMethod::Float8E4M3);

        assert_eq!(QuantizationMethod::for_embedder(ModelId::Hdc), QuantizationMethod::Binary);
        assert_eq!(QuantizationMethod::for_embedder(ModelId::Sparse), QuantizationMethod::SparseNative);
        assert_eq!(QuantizationMethod::for_embedder(ModelId::LateInteraction), QuantizationMethod::TokenPruning);
    }

    /// UT-QUANT-007: Round-trip preserves semantic similarity
    #[test]
    fn test_roundtrip_preserves_similarity() {
        let encoder = PQ8Encoder::new(test_model_path()).unwrap();

        // Create two similar embeddings
        let a = generate_random_embedding(1024);
        let b = add_noise(&a, 0.1); // 10% noise

        let original_sim = cosine_similarity(&a, &b);

        // Quantize and dequantize
        let qa = encoder.quantize(ModelId::Semantic, &a).unwrap();
        let qb = encoder.quantize(ModelId::Semantic, &b).unwrap();
        let ra = encoder.dequantize(&qa).unwrap();
        let rb = encoder.dequantize(&qb).unwrap();

        let reconstructed_sim = cosine_similarity(&ra, &rb);

        // Similarity should be preserved within tolerance
        assert!((original_sim - reconstructed_sim).abs() < 0.1,
            "Similarity not preserved: {} vs {}", original_sim, reconstructed_sim);
    }

    /// UT-QUANT-008: Total fingerprint size < 20KB
    #[test]
    fn test_total_fingerprint_size() {
        let quantizer = FullQuantizer::new(test_model_path()).unwrap();
        let fingerprint = generate_test_fingerprint();

        let quantized = quantizer.quantize_fingerprint(&fingerprint).unwrap();
        let total_bytes: usize = quantized.iter().map(|q| q.size_bytes()).sum();

        assert!(total_bytes < 20_000, "Total size {} exceeds 20KB limit", total_bytes);
    }
}
```

### Integration Tests

```rust
#[cfg(test)]
mod integration_tests {
    /// IT-QUANT-001: Storage uses quantized format
    #[tokio::test]
    async fn test_storage_uses_quantization() {
        let storage = RocksDbStorage::new(test_path()).unwrap();
        let fingerprint = generate_test_fingerprint();

        // Store fingerprint
        let id = Uuid::new_v4();
        storage.store(id, &fingerprint).await.unwrap();

        // Check stored size
        let stored_size = storage.get_stored_size(id).await.unwrap();

        assert!(stored_size < 20_000, "Stored size {} not quantized", stored_size);
    }

    /// IT-QUANT-002: Retrieval dequantizes correctly
    #[tokio::test]
    async fn test_retrieval_dequantizes() {
        let storage = RocksDbStorage::new(test_path()).unwrap();
        let fingerprint = generate_test_fingerprint();

        let id = Uuid::new_v4();
        storage.store(id, &fingerprint).await.unwrap();

        let retrieved = storage.retrieve(id).await.unwrap().unwrap();

        // Check similarity between original and retrieved
        for (model_id, original) in fingerprint.embeddings.iter() {
            let retrieved_emb = retrieved.embeddings.get(model_id).unwrap();
            let sim = cosine_similarity(original, retrieved_emb);

            let max_loss = QuantizationMethod::for_embedder(*model_id).max_recall_loss();
            assert!(sim > 1.0 - max_loss * 2.0,
                "Retrieval degraded {:?} too much: sim={}", model_id, sim);
        }
    }

    /// IT-QUANT-003: Similarity search works on quantized vectors
    #[tokio::test]
    async fn test_similarity_search_quantized() {
        let index = HnswIndex::new(test_path()).unwrap();

        // Index some fingerprints
        let fingerprints: Vec<_> = (0..100)
            .map(|_| (Uuid::new_v4(), generate_test_fingerprint()))
            .collect();

        for (id, fp) in &fingerprints {
            index.add(*id, fp).await.unwrap();
        }

        // Query
        let query_fp = &fingerprints[0].1;
        let results = index.search(query_fp, 10).await.unwrap();

        // First result should be the query itself
        assert_eq!(results[0].id, fingerprints[0].0);
        assert!(results[0].similarity > 0.99);
    }
}
```

---

## Performance Requirements

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| PQ-8 quantize latency | < 100us per embedding | Criterion benchmark |
| Float8 quantize latency | < 50us per embedding | Criterion benchmark |
| Binary quantize latency | < 20us per embedding | Criterion benchmark |
| Dequantize latency | < 200us per embedding | Criterion benchmark |
| Total fingerprint quantize | < 2ms | End-to-end benchmark |
| Storage size per fingerprint | < 20KB | Storage measurement |
| Recall loss (PQ-8) | < 5% | Similarity benchmark |
| Recall loss (Float8) | < 0.3% | Similarity benchmark |

---

## Files to Modify

### Files to DELETE Code From

| File | Code to DELETE | Reason |
|------|----------------|--------|
| `traits/model_factory/quantization.rs` | Old `QuantizationMode` enum (keep only for migration) | Replace with Constitution-aligned methods |
| `storage/serialization/embedding.rs` | Direct f32 serialization without quantization | Replace with quantized serialization |

### Files to CREATE

| File | Content |
|------|---------|
| `quantization/method.rs` | `QuantizationMethod` enum aligned with Constitution |
| `quantization/pq8.rs` | `PQ8Encoder`, `PQ8Codebook` |
| `quantization/float8.rs` | `Float8E4M3Encoder` |
| `quantization/binary.rs` | `BinaryEncoder` |
| `quantization/sparse.rs` | `SparseNativeEncoder` |
| `quantization/token_pruning.rs` | `TokenPruningEncoder` |
| `quantization/router.rs` | `QuantizationRouter` dispatches to encoders |
| `quantization/types.rs` | `QuantizedEmbedding`, `QuantizationMetadata` |
| `quantization/error.rs` | `QuantizationError` |
| `quantization/mod.rs` | Public exports |

### Files to MODIFY

| File | Change Type | Description |
|------|-------------|-------------|
| `storage/serialization/embedding.rs` | MODIFY | Use quantization router before serialization |
| `storage/rocksdb_backend/embedding_ops.rs` | MODIFY | Store quantized bytes, dequantize on retrieval |
| `traits/model_factory/mod.rs` | MODIFY | Export new quantization types |
| `lib.rs` | MODIFY | Export quantization module |

---

## Rollout Plan

### Phase 1: Implement Encoders

1. Create `quantization/` module structure
2. Implement `Float8E4M3Encoder` (simplest)
3. Implement `BinaryEncoder`
4. Implement `SparseNativeEncoder`
5. Implement `TokenPruningEncoder`
6. Implement `PQ8Encoder` with codebook loading

### Phase 2: Train PQ-8 Codebooks

1. Collect corpus embeddings for E1, E5, E7, E10
2. Run K-means to create codebooks (8 subvectors x 256 centroids)
3. Validate recall loss < 5%
4. Export to SafeTensors format

### Phase 3: Integrate with Storage

1. Create `QuantizationRouter`
2. Modify `serialize_embedding()` to use router
3. Modify `deserialize_embedding()` to dequantize
4. Update storage tests

### Phase 4: Validation

1. Verify compressed sizes match expectations
2. Verify recall loss within bounds
3. Run similarity search benchmarks
4. Verify end-to-end fingerprint storage/retrieval

### Phase 5: Deployment

1. Deploy codebook files to model storage
2. Deploy code changes
3. Migrate existing uncompressed data (optional)
4. Monitor storage usage and query quality

---

## Failure Modes and Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Codebook missing | Panic at encoder init | Deploy codebook files, restart |
| Recall loss exceeded | Validation test failure | Retrain codebook with more centroids |
| Dimension mismatch | QuantizationError | Fix embedding pipeline, verify model config |
| Corrupted quantized data | Dequantization error | Re-index from source if available |
| Storage full | Write failure | Quantization reduces this risk 6x |

### Critical Invariants

1. **NO FLOAT32 STORAGE**: Every embedder MUST use its assigned quantization method
2. **DEQUANTIZE FOR SIMILARITY**: Never compare quantized bytes directly (except binary)
3. **TRACK RECALL LOSS**: Log metrics for every quantization operation
4. **CODEBOOK VERSIONING**: Store codebook_id in metadata for future compatibility
5. **FAIL ON MISMATCH**: If dimension/method mismatch, error immediately

---

## Appendix A: Constitution Alignment

| Constitution Requirement | This Spec Addresses |
|-------------------------|---------------------|
| `embeddings.quantization.PQ_8: [E1, E5, E7, E10]` | PQ8Encoder for these embedders |
| `embeddings.quantization.Float8: [E2, E3, E4, E8, E11]` | Float8E4M3Encoder for these embedders |
| `embeddings.quantization.Binary: [E9]` | BinaryEncoder for E9_HDC |
| `embeddings.quantization.Sparse: [E6, E13]` | SparseNativeEncoder for sparse embeddings |
| `embeddings.quantization.TokenPruning: [E12]` | TokenPruningEncoder for late interaction |
| `storage_per_memory: ~17KB` | Total quantized size < 20KB |
| `compression: 63% reduction` | 46KB -> 17KB achieved |

---

## Appendix B: Related Specifications

| Spec ID | Title | Relationship |
|---------|-------|--------------|
| SPEC-EMB-001 | Master Functional Spec | Parent specification |
| TECH-EMB-001 | Sparse Projection Architecture | E6/E13 use sparse native format |
| TECH-EMB-002 | Warm Loading Implementation | Uses warm loading for quantized weights |
| TECH-EMB-004 | Storage Module Design | Consumes quantized embeddings |

---

## Appendix C: E4M3 Float8 Reference

```
E4M3 Format (8-bit float):
  Bit 7: Sign (S)
  Bits 6-3: Exponent (E), bias = 7
  Bits 2-0: Mantissa (M)

  Value = (-1)^S * 2^(E-7) * (1 + M/8)

  Examples:
    0x00 = 0.0
    0x38 = 1.0  (S=0, E=7, M=0)
    0x78 = 2.0  (S=0, E=8, M=0)
    0x3C = 1.5  (S=0, E=7, M=4)

  Range: ~1.5e-5 to 448 (for positive values)
```

---

## Appendix D: PQ-8 Training Algorithm

```python
# Pseudocode for PQ-8 codebook training

def train_pq8_codebook(embeddings: List[np.ndarray],
                        num_subvectors: int = 8,
                        num_centroids: int = 256) -> Codebook:
    """
    Train Product Quantization codebook using K-means.

    Args:
        embeddings: List of embeddings to train on (1000+ recommended)
        num_subvectors: Number of subvectors (8 for PQ-8)
        num_centroids: Centroids per subvector (256 = 1 byte index)

    Returns:
        Trained codebook with shape [8, 256, dim/8]
    """
    dim = embeddings[0].shape[0]
    subvector_dim = dim // num_subvectors

    codebook = []

    for sv_idx in range(num_subvectors):
        # Extract subvector slice from all embeddings
        start = sv_idx * subvector_dim
        end = start + subvector_dim
        subvectors = np.array([e[start:end] for e in embeddings])

        # Run K-means to find centroids
        kmeans = KMeans(n_clusters=num_centroids)
        kmeans.fit(subvectors)

        codebook.append(kmeans.cluster_centers_)

    return np.array(codebook)  # Shape: [8, 256, dim/8]
```

</technical_spec>
