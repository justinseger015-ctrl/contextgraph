# TASK-CORE-013: Embedding Quantization Infrastructure

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-CORE-013 |
| **Title** | Embedding Quantization Infrastructure |
| **Status** | :white_circle: todo |
| **Layer** | Foundation |
| **Sequence** | 13 |
| **Estimated Days** | 4 |
| **Complexity** | High |

## Implements

- **REQ-STORAGE-QUANTIZE-01**: Storage footprint reduction via quantization
- **Performance Budget**: Memory per array < 17KB (quantized)

## Dependencies

| Task | Reason |
|------|--------|
| TASK-CORE-003 | Uses TeleologicalArray and EmbedderOutput types |
| TASK-CORE-002 | Uses Embedder enum for per-embedder configuration |

## Objective

Implement quantization schemes (PQ-8, Float8, Binary) to achieve 63% storage reduction for teleological arrays (46KB -> 17KB per array).

## Context

The constitution specifies:
- Unquantized array size: ~46KB
- Target quantized size: < 17KB (63% reduction)
- Quantization types: PQ-8 (Product Quantization), Float8, Binary

Different embedders benefit from different quantization:
- Dense embeddings (E1, E2, E3, etc.): PQ-8 or Float8
- Sparse embeddings (E6, E13): Sparse-specific encoding
- Binary embeddings: Native binary storage

## Scope

### In Scope

- `QuantizationScheme` enum (PQ8, Float8, Binary, None)
- `ProductQuantizer` for PQ-8 encoding
- `Float8Quantizer` for 8-bit float encoding
- `BinaryQuantizer` for HDC embeddings
- `QuantizedEmbedderOutput` type
- Encode/decode functions per scheme
- Per-embedder default quantization selection
- Codebook management for PQ

### Out of Scope

- Quantization-aware training
- Dynamic quantization selection based on content
- GPU-accelerated quantization (future optimization)

## Definition of Done

### Signatures

```rust
// crates/context-graph-embeddings/src/quantization/mod.rs

/// Available quantization schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationScheme {
    /// No quantization - full f32 precision
    None,
    /// Product Quantization with 8-bit codes
    PQ8,
    /// 8-bit floating point (E4M3 or E5M2)
    Float8,
    /// Binary quantization (sign bits)
    Binary,
}

/// Trait for quantization implementations
pub trait Quantizer: Send + Sync {
    /// Encode full-precision embedding to quantized form
    fn encode(&self, embedding: &[f32]) -> QuantizedVec;

    /// Decode quantized back to approximate f32
    fn decode(&self, quantized: &QuantizedVec) -> Vec<f32>;

    /// Get quantization scheme
    fn scheme(&self) -> QuantizationScheme;

    /// Compression ratio (original_size / quantized_size)
    fn compression_ratio(&self) -> f32;

    /// Quantized size for given input dimensions
    fn quantized_size(&self, dims: usize) -> usize;
}

/// Quantized vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizedVec {
    /// Full precision (no quantization)
    Full(Vec<f32>),
    /// Product quantization codes
    PQ8 {
        codes: Vec<u8>,
        codebook_id: u32,
    },
    /// 8-bit float values
    Float8(Vec<u8>),
    /// Binary (sign bits packed)
    Binary(BitVec),
}

impl QuantizedVec {
    /// Get size in bytes
    pub fn size_bytes(&self) -> usize;

    /// Get original dimensions (if known)
    pub fn original_dims(&self) -> Option<usize>;
}

// crates/context-graph-embeddings/src/quantization/product.rs

/// Product Quantizer for PQ-8 encoding
pub struct ProductQuantizer {
    /// Number of subvectors
    num_subvectors: usize,
    /// Bits per subvector (8 for PQ-8)
    bits_per_subvector: usize,
    /// Codebooks (one per subvector)
    codebooks: Vec<Codebook>,
}

impl ProductQuantizer {
    /// Create new PQ with specified parameters
    pub fn new(dims: usize, num_subvectors: usize) -> Self;

    /// Train codebooks from sample embeddings
    pub fn train(&mut self, samples: &[Vec<f32>]) -> QuantizationResult<()>;

    /// Load pre-trained codebooks
    pub fn load(path: &Path) -> QuantizationResult<Self>;

    /// Save trained codebooks
    pub fn save(&self, path: &Path) -> QuantizationResult<()>;
}

impl Quantizer for ProductQuantizer {
    fn encode(&self, embedding: &[f32]) -> QuantizedVec;
    fn decode(&self, quantized: &QuantizedVec) -> Vec<f32>;
    fn scheme(&self) -> QuantizationScheme { QuantizationScheme::PQ8 }
    fn compression_ratio(&self) -> f32;
    fn quantized_size(&self, dims: usize) -> usize;
}

/// Single codebook for one subvector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    /// Centroids (256 for 8-bit, dims = subvector_dims)
    centroids: Vec<Vec<f32>>,
}

// crates/context-graph-embeddings/src/quantization/float8.rs

/// Float8 quantizer (E4M3 format)
pub struct Float8Quantizer;

impl Float8Quantizer {
    pub fn new() -> Self;
}

impl Quantizer for Float8Quantizer {
    fn encode(&self, embedding: &[f32]) -> QuantizedVec;
    fn decode(&self, quantized: &QuantizedVec) -> Vec<f32>;
    fn scheme(&self) -> QuantizationScheme { QuantizationScheme::Float8 }
    fn compression_ratio(&self) -> f32 { 4.0 } // f32 -> f8
    fn quantized_size(&self, dims: usize) -> usize { dims }
}

// crates/context-graph-embeddings/src/quantization/binary.rs

/// Binary quantizer (sign bits)
pub struct BinaryQuantizer;

impl BinaryQuantizer {
    pub fn new() -> Self;
}

impl Quantizer for BinaryQuantizer {
    fn encode(&self, embedding: &[f32]) -> QuantizedVec;
    fn decode(&self, quantized: &QuantizedVec) -> Vec<f32>;
    fn scheme(&self) -> QuantizationScheme { QuantizationScheme::Binary }
    fn compression_ratio(&self) -> f32 { 32.0 } // f32 -> 1 bit
    fn quantized_size(&self, dims: usize) -> usize { (dims + 7) / 8 }
}

// crates/context-graph-embeddings/src/quantization/config.rs

/// Get default quantization scheme for embedder
pub fn default_scheme_for(embedder: Embedder) -> QuantizationScheme {
    match embedder {
        Embedder::Semantic |
        Embedder::TemporalRecent |
        Embedder::TemporalPeriodic |
        Embedder::EntityRelationship |
        Embedder::Causal |
        Embedder::Contextual |
        Embedder::Emotional |
        Embedder::Syntactic |
        Embedder::Pragmatic |
        Embedder::CrossModal => QuantizationScheme::PQ8,

        Embedder::Splade |
        Embedder::KeywordSplade => QuantizationScheme::None, // Sparse, different handling

        Embedder::LateInteraction => QuantizationScheme::Float8,
    }
}

/// Quantized teleological array wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTeleologicalArray {
    pub id: Uuid,
    pub embeddings: [QuantizedVec; 13],
    pub metadata: ArrayMetadata,
}

impl QuantizedTeleologicalArray {
    /// Create from full-precision array
    pub fn from_array(
        array: &TeleologicalArray,
        quantizers: &[Box<dyn Quantizer>; 13],
    ) -> Self;

    /// Decode to full-precision array
    pub fn to_array(
        &self,
        quantizers: &[Box<dyn Quantizer>; 13],
    ) -> TeleologicalArray;

    /// Get total size in bytes
    pub fn size_bytes(&self) -> usize;
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Quantized array size | < 17KB |
| Recall degradation (PQ-8) | < 5% |
| Encode latency | < 1ms per embedding |
| Decode latency | < 0.5ms per embedding |

## Verification

- [ ] Quantized array size < 17KB
- [ ] Round-trip (encode -> decode) maintains acceptable accuracy
- [ ] PQ-8 recall degradation < 5% on benchmark
- [ ] Per-embedder defaults applied correctly
- [ ] Codebook serialization/deserialization works
- [ ] Float8 precision meets requirements

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-embeddings/src/quantization/mod.rs` | Module root, traits |
| `crates/context-graph-embeddings/src/quantization/product.rs` | PQ-8 implementation |
| `crates/context-graph-embeddings/src/quantization/float8.rs` | Float8 implementation |
| `crates/context-graph-embeddings/src/quantization/binary.rs` | Binary implementation |
| `crates/context-graph-embeddings/src/quantization/config.rs` | Per-embedder defaults |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Recall degradation too high | Medium | High | Tune PQ parameters, benchmark |
| Codebook training slow | Low | Medium | Pre-train codebooks offline |
| Float8 precision issues | Low | Medium | Careful range handling |

## Traceability

- Source: Constitution directory_structure (line 123-124)
- Performance Budget: Memory per array < 17KB (line 574)
