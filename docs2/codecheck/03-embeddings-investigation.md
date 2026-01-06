# Sherlock Holmes Forensic Investigation Report

## Case ID: SHERLOCK-03-EMBEDDINGS
## Date: 2026-01-06
## Subject: Context Graph Embeddings Crate - 13-Model Pipeline Investigation
## Investigator: Sherlock Holmes

---

## EXECUTIVE SUMMARY

**VERDICT: GUILTY - MULTIPLE CRITICAL ISSUES FOUND**

The embeddings crate (`context-graph-embeddings`) contains a mixture of REAL and FAKE implementations. While the individual pretrained model implementations (SemanticModel, CodeModel, CausalModel, etc.) appear to use legitimate GPU-accelerated inference via Candle, there are CRITICAL issues in the infrastructure:

1. **FAKE Sparse Projection**: Claims "learned projection" but uses hash-based modular arithmetic
2. **DIMENSION MISMATCH**: Specification says 1536D for Sparse, implementation uses 768D
3. **SIMULATED Warm Loading**: Entire warm loading pipeline is a simulation
4. **STUB Mode in Preflight**: Returns fake GPU info when CUDA unavailable
5. **Placeholder Storage Module**: Storage is an empty placeholder

---

## EMBEDDER STATUS CHECKLIST (E1-E13)

| Model | Status | Evidence |
|-------|--------|----------|
| E1: Semantic (1024D) | REAL | GPU BERT forward pass via Candle |
| E2: TemporalRecent (512D) | REAL | Mathematical exponential decay computation |
| E3: TemporalPeriodic (512D) | REAL | Fourier basis mathematical encoding |
| E4: TemporalPositional (512D) | REAL | Sinusoidal positional encoding |
| E5: Causal (768D) | REAL | GPU Longformer forward pass via Candle |
| E6: Sparse (30K->1536D) | PARTIAL | Real SPLADE inference, FAKE projection |
| E7: Code (256->768D) | REAL | GPU CodeT5p forward pass via Candle |
| E8: Graph (384D) | REAL | GPU MiniLM forward pass |
| E9: HDC (10K->1024D) | REAL | Hyperdimensional computing with XOR binding |
| E10: Multimodal (768D) | REAL | GPU CLIP forward pass |
| E11: Entity (384D) | REAL | GPU MiniLM forward pass |
| E12: LateInteraction (128D/tok) | REAL | GPU ColBERT forward pass |
| E13: Splade (30K->1536D) | PARTIAL | Reuses E6 implementation with FAKE projection |

---

## CRITICAL FINDINGS

### FINDING 1: FAKE Sparse-to-Dense Projection (CRITICAL)

**File**: `/home/cabdru/contextgraph/crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs`

**Lines**: 62-82

**Exact Code**:
```rust
/// Convert sparse vector to dense format (for multi-array storage).
///
/// Returns a 768D projected representation using the top-k terms.
pub fn to_dense_projected(&self, projected_dim: usize) -> Vec<f32> {
    // Project sparse vocab-sized vector to dense hidden dimension
    // This uses hash-based projection for efficiency
    let mut dense = vec![0.0f32; projected_dim];

    for (&idx, &weight) in self.indices.iter().zip(&self.weights) {
        // Use modular hashing to map vocab indices to dense dimensions
        let dense_idx = idx % projected_dim;
        dense[dense_idx] += weight;
    }

    // L2 normalize
    let norm: f32 = dense.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for v in &mut dense {
            *v /= norm;
        }
    }

    dense
}
```

**Why Problematic**:
- The specification says "30K sparse -> 1536D via **learned projection**"
- The implementation uses `idx % projected_dim` - a simple modular hash, NOT a learned neural network projection
- This destroys semantic information by colliding sparse indices into hash buckets
- Multiple vocabulary terms with completely different meanings will map to the same dense dimension

**Severity**: CRITICAL

---

### FINDING 2: Dimension Mismatch - Specification vs Implementation (CRITICAL)

**Evidence**:

1. **ModelId specification** (`/home/cabdru/contextgraph/crates/context-graph-embeddings/src/types/model_id/core.rs`, line 104):
   ```rust
   Self::Sparse => 1536,  // 30K -> 1536 via learned projection
   ```

2. **Dimension constant** (`/home/cabdru/contextgraph/crates/context-graph-embeddings/src/types/dimensions/constants.rs`, line 70):
   ```rust
   pub const SPARSE: usize = 1536;
   ```

3. **Implementation constant** (`/home/cabdru/contextgraph/crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs`, line 33):
   ```rust
   pub const SPARSE_PROJECTED_DIMENSION: usize = 768;
   ```

4. **Actual output** (`/home/cabdru/contextgraph/crates/context-graph-embeddings/src/models/pretrained/sparse/model.rs`, line 290):
   ```rust
   let vector = sparse.to_dense_projected(SPARSE_HIDDEN_SIZE); // 768
   ```

**Why Problematic**:
- The type system claims Sparse produces 1536D
- The actual implementation produces 768D
- Any code checking `ModelId::Sparse.projected_dimension()` gets 1536 but receives 768D vectors
- This will cause runtime panics or data corruption in multi-array storage

**Severity**: CRITICAL

---

### FINDING 3: Simulated Warm Loading Pipeline (CRITICAL)

**File**: `/home/cabdru/contextgraph/crates/context-graph-embeddings/src/warm/loader/operations.rs`

**Evidence**:

1. **Simulated Weight Loading** (lines 143-165):
```rust
/// Simulate loading model weights.
///
/// In a real implementation, this would:
/// 1. Open SafeTensors file
/// 2. Read tensors
/// 3. Transfer to GPU
/// 4. Compute SHA256 checksum
pub fn simulate_weight_loading(model_id: &str, _size_bytes: usize) -> WarmResult<u64> {
    // Generate a deterministic checksum based on model ID
    let mut checksum = 0u64;
    for (i, byte) in model_id.bytes().enumerate() {
        checksum ^= (byte as u64) << ((i % 8) * 8);
    }
    checksum ^= 0xDEAD_BEEF_CAFE_BABEu64;
    // ...
    Ok(checksum)
}
```

2. **Simulated VRAM Allocation** (lines 124-128):
```rust
// Generate a simulated VRAM pointer
// In a real implementation, this would come from cudaMalloc
let base_ptr = 0x7f80_0000_0000u64;
let offset = memory_pools.list_model_allocations().len() as u64 * 0x1_0000_0000;
let vram_ptr = base_ptr + offset;
```

3. **Fake Inference Validation** (lines 184-187):
```rust
// Simulate test inference output
let output: Vec<f32> = (0..expected_dimension)
    .map(|i| (i as f32 * 0.001).sin())
    .collect();
```

**Why Problematic**:
- The entire "warm loading" system DOES NOTHING REAL
- Weight loading generates a fake checksum from the model name string
- VRAM allocation generates fake pointers without calling cudaMalloc
- Validation runs fake inference using sin() instead of the actual model
- Code that depends on "warm loaded" models will receive NULL models

**Severity**: CRITICAL

---

### FINDING 4: Stub Mode in Preflight Checks (HIGH)

**File**: `/home/cabdru/contextgraph/crates/context-graph-embeddings/src/warm/loader/preflight.rs`

**Lines**: 31-43

**Exact Code**:
```rust
#[cfg(not(feature = "cuda"))]
{
    tracing::warn!("CUDA feature not enabled, running in stub mode");
    // In stub mode, we simulate successful checks for testing
    *gpu_info = Some(GpuInfo::new(
        0,
        "Simulated RTX 5090".to_string(),
        (REQUIRED_COMPUTE_MAJOR, REQUIRED_COMPUTE_MINOR),
        MINIMUM_VRAM_BYTES,
        "Simulated".to_string(),
    ));
    Ok(())
}
```

**Why Problematic**:
- Without CUDA feature, preflight returns a FAKE "Simulated RTX 5090"
- This makes the system APPEAR to pass all GPU checks when no GPU exists
- Downstream code will attempt GPU operations that will fail

**Severity**: HIGH

---

### FINDING 5: Placeholder Storage Module (MEDIUM)

**File**: `/home/cabdru/contextgraph/crates/context-graph-embeddings/src/storage/mod.rs`

**Entire Content**:
```rust
//! Storage module for embeddings.
//!
//! This module is a placeholder for Multi-Array Storage implementations.
//!
//! Storage implementations should use SemanticFingerprint or JohariFingerprint
//! from the context-graph-teleology crate. Each embedding is stored SEPARATELY
//! at its native dimension for per-space indexing.
```

**Why Problematic**:
- The storage module is an EMPTY PLACEHOLDER
- No actual multi-array storage implementation exists
- The 13-embedding array (teleological vector) cannot be stored

**Severity**: MEDIUM (referenced as handled by teleology crate)

---

### FINDING 6: Quantization Not Actually Applied (MEDIUM)

**Evidence**:

The `QuantizationMode` enum exists with FP16, BF16, Int8 options, but there is NO evidence that quantization is actually applied during weight loading or inference. The enum provides memory estimates but the actual Candle forward passes use whatever dtype the model was loaded with.

**Files**:
- `/home/cabdru/contextgraph/crates/context-graph-embeddings/src/traits/model_factory/quantization.rs`
- `/home/cabdru/contextgraph/crates/context-graph-embeddings/src/warm/config.rs`

**Why Problematic**:
- Configuration claims to support quantization
- No code actually converts weights to lower precision
- Memory estimates will be wrong if FP16 is "configured" but FP32 is used

**Severity**: MEDIUM

---

## CHAIN OF CUSTODY

| Timestamp | File Examined | Finding |
|-----------|---------------|---------|
| 2026-01-06 | lib.rs | 12 models claimed (should be 13) |
| 2026-01-06 | models/mod.rs | Factory exports all 13 models |
| 2026-01-06 | pretrained/semantic/ | REAL GPU implementation |
| 2026-01-06 | custom/hdc/ | REAL HDC implementation |
| 2026-01-06 | custom/temporal_recent/ | REAL decay implementation |
| 2026-01-06 | pretrained/sparse/types.rs | FAKE hash-based projection |
| 2026-01-06 | warm/loader/operations.rs | SIMULATED weight loading |
| 2026-01-06 | warm/loader/preflight.rs | STUB mode with fake GPU |
| 2026-01-06 | storage/mod.rs | PLACEHOLDER module |

---

## RECOMMENDATIONS

### IMMEDIATE ACTIONS (CRITICAL)

1. **Implement Real Sparse Projection**:
   - Add a learned projection layer (768x1536 weight matrix) for E6 Sparse
   - Train or load pretrained projection weights
   - Remove hash-based modular arithmetic

2. **Fix Dimension Mismatch**:
   - Either change `SPARSE_PROJECTED_DIMENSION` to 1536 and implement proper projection
   - OR change `ModelId::Sparse.projected_dimension()` to return 768

3. **Remove Simulated Warm Loading**:
   - Delete `simulate_weight_loading` function
   - Implement actual weight loading from SafeTensors files
   - Use real cudaMalloc for VRAM allocation
   - Run actual model inference for validation

### SHORT-TERM ACTIONS (HIGH)

4. **Remove Stub Mode**:
   - Make CUDA feature truly mandatory (already compile_error! exists)
   - Remove the `#[cfg(not(feature = "cuda"))]` fake GPU path
   - Or make it properly error out instead of returning fake success

5. **Implement Quantization**:
   - Add actual FP16/BF16/Int8 weight conversion during loading
   - Use Candle's dtype conversion functions

### LONG-TERM ACTIONS (MEDIUM)

6. **Complete Storage Module**:
   - Verify teleology crate provides actual storage
   - Or implement multi-array storage here

---

## EVIDENCE PRESERVATION

All evidence has been documented with exact file paths, line numbers, and code snippets. The codebase has been examined using:
- Direct file reads
- Pattern searches (grep)
- Cross-reference analysis

---

## VERDICT

**GUILTY AS CHARGED**

The embeddings crate, while containing real GPU-accelerated model implementations, is GUILTY of:

1. **False Claims**: Claiming "learned projection" when using hash modulo
2. **Specification Violations**: Dimension mismatches between spec and implementation
3. **Simulated Infrastructure**: Entire warm loading pipeline does nothing real
4. **Hidden Stubs**: Silent fallback to fake GPU when CUDA unavailable
5. **Incomplete Implementation**: Placeholder storage module

The individual model implementations (E1-E12 core models) appear legitimate, but the infrastructure around them (warm loading, projection, storage) is fundamentally broken or fake.

---

## CASE STATUS: OPEN

This case remains OPEN until all CRITICAL issues are fixed and verified.

---

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*

**Sherlock Holmes**
**Consulting Detective, Code Forensics Division**
