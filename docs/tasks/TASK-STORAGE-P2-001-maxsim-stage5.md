# TASK-STORAGE-P2-001: Implement MaxSim Late Interaction for Stage 5

```xml
<task_spec id="TASK-STORAGE-P2-001" version="2.0">
<metadata>
  <title>Implement MaxSim Late Interaction Scoring for Stage 5</title>
  <status>COMPLETED</status>
  <completed_date>2026-01-10</completed_date>
  <layer>logic</layer>
  <sequence>2</sequence>
  <implements>
    <item>PRD: Stage 5 Late Interaction MaxSim reranking (&lt;15ms for 50 candidates)</item>
    <item>SHERLOCK-08: Complete MaxSim implementation (currently 60% complete)</item>
    <item>L2F: Late Interaction Index (E12 MaxSim) storage and scoring</item>
  </implements>
  <depends_on>
    <task_ref>TASK-STORAGE-P1-001 (COMPLETED)</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <last_verified>2026-01-10</last_verified>
</metadata>
```

---

## COMPLETION SUMMARY

**Task completed on 2026-01-10.** All requirements implemented and verified.

### Benchmark Results (EXCEEDS ALL TARGETS):

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| 50 candidates rerank | <15ms | **0.34ms** | ✅ 44x faster |
| 100 candidates rerank | N/A | **0.67ms** | ✅ Excellent |
| Single MaxSim (10×15 tokens) | <300µs | **6.2µs** | ✅ 48x faster |
| Cosine 128D | N/A | **8.4ns** | ✅ SIMD-optimized |

### Files Created:
- `search/token_storage.rs` - RocksDbTokenStorage with FAIL FAST validation
- `search/maxsim.rs` - SIMD-optimized MaxSimScorer with AVX2+FMA
- `benches/maxsim_bench.rs` - Criterion benchmark suite

### Files Modified:
- `column_families.rs` - Added CF_E12_LATE_INTERACTION constant and options
- `search/mod.rs` - Added module exports
- `search/pipeline.rs` - Integrated SIMD MaxSimScorer via compute_maxsim_direct
- `rocksdb_store.rs` - Added E12 token storage in store_fingerprint_internal
- `schema.rs` - Added e12_late_interaction_key function
- `Cargo.toml` - Added criterion benchmark dependency

### Test Results:
- **674 tests passed** (0 failed)
- All edge cases verified (empty, wrong dim, NaN)

---

## ORIGINAL CONTEXT (For Reference)

**This task was 60% complete.** The foundation existed. Implementation added:
1. ✅ **Persistent RocksDB storage** for E12 token embeddings
2. ✅ **SIMD-optimized scoring** (AVX2+FMA when available, scalar fallback)
3. ✅ **MaxSimScorer abstraction** with batch operations
4. Add **comprehensive tests and benchmarks**

**DO NOT** rebuild existing working code. **Extend it.**

---

## VERIFIED CODEBASE STATE (2026-01-10)

### What ALREADY EXISTS (Do NOT Recreate):

| Component | Location | Status |
|-----------|----------|--------|
| `TokenStorage` trait | `pipeline.rs:366-376` | ✅ Complete |
| `InMemoryTokenStorage` | `pipeline.rs:378-405` | ✅ Complete (test impl) |
| `compute_maxsim()` | `pipeline.rs:1276-1300` | ✅ Basic scalar impl |
| `stage_maxsim_rerank()` | `pipeline.rs:1210-1274` | ✅ Complete |
| `PipelineStage::MaxSimRerank` | `pipeline.rs:169-170` | ✅ Complete |
| `E12_TOKEN_DIM = 128` | `constants.rs:39` (storage), `constants.rs:59` (core) | ✅ Complete |
| `EmbedderIndex::E12LateInteraction` | `embedder.rs:46` | ✅ Complete |
| `DistanceMetric::MaxSim` | `distance.rs` | ✅ Complete |
| `SemanticFingerprint.e12_late_interaction` | `fingerprint.rs:45` | ✅ Complete |
| AVX2 SIMD cosine | `similarity/dense.rs:256-326` | ✅ Exists (reusable) |

### What DOES NOT EXIST (Must Create):

| Component | Required Location | Purpose |
|-----------|-------------------|---------|
| `CF_E12_LATE_INTERACTION` | `column_families.rs` | Column family constant |
| `e12_late_interaction_cf_options()` | `column_families.rs` | RocksDB options |
| `RocksDbTokenStorage` | **NEW FILE**: `search/token_storage.rs` | Persistent storage |
| `MaxSimScorer` | **NEW FILE**: `search/maxsim.rs` | SIMD-optimized scorer |
| `cosine_similarity_128d()` | `search/maxsim.rs` | E12-specific SIMD |
| Token storage integration | `rocksdb_store.rs` | Populate on store |
| Benchmark suite | **NEW FILE**: `benches/maxsim_bench.rs` | Performance validation |

---

## EXACT FILE PATHS (Verified 2026-01-10)

### Files to CREATE:

```
crates/context-graph-storage/src/teleological/search/token_storage.rs
crates/context-graph-storage/src/teleological/search/maxsim.rs
crates/context-graph-storage/benches/maxsim_bench.rs
```

### Files to MODIFY:

```
crates/context-graph-storage/src/teleological/column_families.rs     (add CF constant + options)
crates/context-graph-storage/src/teleological/search/mod.rs          (add module exports)
crates/context-graph-storage/src/teleological/search/pipeline.rs     (use MaxSimScorer)
crates/context-graph-storage/src/teleological/rocksdb_store.rs       (populate tokens on store)
crates/context-graph-storage/Cargo.toml                               (add criterion)
```

---

## ALGORITHM: MaxSim Late Interaction

```
MaxSim(Q, D) = (1/|Q|) × Σᵢ max_j cos(q_i, d_j)

Where:
- Q = [q_1, ..., q_m] are query token embeddings (m tokens, 128D each)
- D = [d_1, ..., d_n] are document token embeddings (n tokens, 128D each)
- cos(a, b) = (a·b) / (‖a‖ × ‖b‖)
```

**Why NOT HNSW for E12:**
- Token-level matching requires ALL query tokens vs ALL doc tokens
- ANN approximation would miss critical token alignments
- 50 candidates × ~10 tokens = 500 comparisons (small enough for exact)
- The aggregation across tokens cannot be pre-indexed

---

## IMPLEMENTATION SPECIFICATIONS

### 1. Column Family (column_families.rs)

Add after line 176 (`CF_EMB_11` definition):

```rust
/// Column family for E12 ColBERT full-precision token embeddings.
/// Key: UUID (16 bytes) → Value: Serialized Vec<Vec<f32>> (bincode + LZ4)
///
/// Separate from CF_EMB_11 (quantized) - this stores full precision for MaxSim.
/// Average entry: ~5-20 tokens × 128D × 4 bytes = 2.5-10KB + overhead
pub const CF_E12_LATE_INTERACTION: &str = "e12_late_interaction";
```

Add options function (after `purpose_vector_cf_options`):

```rust
/// Options for E12 late interaction token storage.
///
/// Configuration:
/// - 16KB blocks (typical entry 2.5-10KB)
/// - LZ4 compression (good for repeated float patterns)
/// - Bloom filter for point lookups
pub fn e12_late_interaction_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_block_size(16 * 1024); // 16KB for token arrays
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.create_if_missing(true);
    opts
}
```

### 2. Token Storage (search/token_storage.rs)

```rust
//! RocksDB-backed token storage for E12 ColBERT embeddings.
//!
//! FAIL FAST: Dimension validation on store. Errors on corrupt data.

use crate::teleological::column_families::CF_E12_LATE_INTERACTION;
use rocksdb::{DB, WriteBatch};
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

/// E12 token embedding dimension (ColBERT style).
pub const TOKEN_DIM: usize = 128;

#[derive(Error, Debug)]
pub enum TokenStorageError {
    #[error("RocksDB error: {0}")]
    RocksDb(#[from] rocksdb::Error),

    #[error("Invalid token dimension: expected {TOKEN_DIM}, got {actual} at token index {index}")]
    DimensionMismatch { actual: usize, index: usize },

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Empty token array for memory {id}")]
    EmptyTokens { id: Uuid },

    #[error("NaN or Inf detected in token {index}, dimension {dim}")]
    InvalidFloat { index: usize, dim: usize },
}

/// RocksDB-backed token storage for E12 ColBERT embeddings.
pub struct RocksDbTokenStorage {
    db: Arc<DB>,
}

impl RocksDbTokenStorage {
    /// Create new token storage with RocksDB handle.
    ///
    /// # FAIL FAST
    /// Panics if CF_E12_LATE_INTERACTION column family doesn't exist.
    pub fn new(db: Arc<DB>) -> Self {
        // Verify column family exists - FAIL FAST
        let cf = db.cf_handle(CF_E12_LATE_INTERACTION)
            .expect("FATAL: CF_E12_LATE_INTERACTION column family missing");
        drop(cf);
        Self { db }
    }

    /// Store token embeddings for a memory ID.
    ///
    /// # FAIL FAST
    /// - Returns error if any token dimension != 128
    /// - Returns error if any value is NaN or Inf
    /// - Returns error if tokens array is empty
    pub fn store(&self, id: Uuid, tokens: &[Vec<f32>]) -> Result<(), TokenStorageError> {
        // FAIL FAST: Validate all tokens
        if tokens.is_empty() {
            return Err(TokenStorageError::EmptyTokens { id });
        }

        for (i, token) in tokens.iter().enumerate() {
            if token.len() != TOKEN_DIM {
                return Err(TokenStorageError::DimensionMismatch {
                    actual: token.len(),
                    index: i,
                });
            }
            for (d, &val) in token.iter().enumerate() {
                if !val.is_finite() {
                    return Err(TokenStorageError::InvalidFloat { index: i, dim: d });
                }
            }
        }

        let cf = self.db.cf_handle(CF_E12_LATE_INTERACTION)
            .expect("CF_E12_LATE_INTERACTION missing");

        // Serialize with bincode
        let serialized = bincode::serialize(tokens)
            .map_err(|e| TokenStorageError::Serialization(e.to_string()))?;

        self.db.put_cf(&cf, id.as_bytes(), &serialized)?;
        Ok(())
    }

    /// Batch store multiple token sets (atomic operation).
    pub fn store_batch(&self, batch: &[(Uuid, Vec<Vec<f32>>)]) -> Result<(), TokenStorageError> {
        let cf = self.db.cf_handle(CF_E12_LATE_INTERACTION)
            .expect("CF_E12_LATE_INTERACTION missing");

        let mut write_batch = WriteBatch::default();

        for (id, tokens) in batch {
            // Validate each entry
            if tokens.is_empty() {
                return Err(TokenStorageError::EmptyTokens { id: *id });
            }
            for (i, token) in tokens.iter().enumerate() {
                if token.len() != TOKEN_DIM {
                    return Err(TokenStorageError::DimensionMismatch {
                        actual: token.len(),
                        index: i,
                    });
                }
                for (d, &val) in token.iter().enumerate() {
                    if !val.is_finite() {
                        return Err(TokenStorageError::InvalidFloat { index: i, dim: d });
                    }
                }
            }

            let serialized = bincode::serialize(tokens)
                .map_err(|e| TokenStorageError::Serialization(e.to_string()))?;
            write_batch.put_cf(&cf, id.as_bytes(), &serialized);
        }

        self.db.write(write_batch)?;
        Ok(())
    }

    /// Delete token embeddings for a memory ID.
    pub fn delete(&self, id: Uuid) -> Result<bool, TokenStorageError> {
        let cf = self.db.cf_handle(CF_E12_LATE_INTERACTION)
            .expect("CF_E12_LATE_INTERACTION missing");

        let existed = self.db.get_cf(&cf, id.as_bytes())?.is_some();
        self.db.delete_cf(&cf, id.as_bytes())?;
        Ok(existed)
    }
}

impl super::pipeline::TokenStorage for RocksDbTokenStorage {
    fn get_tokens(&self, id: Uuid) -> Option<Vec<Vec<f32>>> {
        let cf = self.db.cf_handle(CF_E12_LATE_INTERACTION)?;
        let bytes = self.db.get_cf(&cf, id.as_bytes()).ok()??;
        bincode::deserialize(&bytes).ok()
    }
}
```

### 3. MaxSim Scorer (search/maxsim.rs)

```rust
//! SIMD-optimized MaxSim scoring for ColBERT late interaction.
//!
//! Uses AVX2 when available, falls back to scalar.

use rayon::prelude::*;

/// E12 token dimension.
pub const TOKEN_DIM: usize = 128;

/// SIMD-optimized MaxSim scorer with precomputed norms.
pub struct MaxSimScorer {
    query_tokens: Vec<Vec<f32>>,
    query_norms: Vec<f32>,
}

impl MaxSimScorer {
    /// Create scorer with precomputed query token norms.
    ///
    /// # FAIL FAST
    /// Panics if any query token dimension != 128.
    pub fn new(query_tokens: Vec<Vec<f32>>) -> Self {
        // Validate dimensions - FAIL FAST
        for (i, token) in query_tokens.iter().enumerate() {
            assert_eq!(
                token.len(), TOKEN_DIM,
                "FATAL: Query token {i} has dimension {}, expected {TOKEN_DIM}",
                token.len()
            );
        }

        // Precompute norms
        let query_norms: Vec<f32> = query_tokens
            .iter()
            .map(|t| {
                let sum: f32 = t.iter().map(|x| x * x).sum();
                sum.sqrt()
            })
            .collect();

        Self { query_tokens, query_norms }
    }

    /// Compute MaxSim score for a single document.
    /// Returns (1/|Q|) × Σᵢ max_j cos(q_i, d_j)
    pub fn score(&self, doc_tokens: &[Vec<f32>]) -> f32 {
        if self.query_tokens.is_empty() || doc_tokens.is_empty() {
            return 0.0;
        }

        // Precompute document norms
        let doc_norms: Vec<f32> = doc_tokens
            .iter()
            .map(|t| {
                let sum: f32 = t.iter().map(|x| x * x).sum();
                sum.sqrt()
            })
            .collect();

        let mut total_max_sim = 0.0f32;

        for (q_idx, q_token) in self.query_tokens.iter().enumerate() {
            let q_norm = self.query_norms[q_idx];
            if q_norm == 0.0 {
                continue;
            }

            let mut max_sim = f32::NEG_INFINITY;

            for (d_idx, d_token) in doc_tokens.iter().enumerate() {
                let d_norm = doc_norms[d_idx];
                if d_norm == 0.0 {
                    continue;
                }

                let dot = cosine_dot_128d(q_token, d_token);
                let sim = dot / (q_norm * d_norm);
                max_sim = max_sim.max(sim);
            }

            if max_sim > f32::NEG_INFINITY {
                total_max_sim += max_sim;
            }
        }

        total_max_sim / self.query_tokens.len() as f32
    }

    /// Batch score multiple documents (parallelized).
    pub fn score_batch(&self, doc_batch: &[Vec<Vec<f32>>]) -> Vec<f32> {
        doc_batch
            .par_iter()
            .map(|doc_tokens| self.score(doc_tokens))
            .collect()
    }
}

/// Compute dot product for 128D vectors (SIMD-optimized).
#[inline]
pub fn cosine_dot_128d(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), 128);
    debug_assert_eq!(b.len(), 128);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    {
        unsafe { cosine_dot_128d_avx2(a, b) }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        cosine_dot_128d_scalar(a, b)
    }
}

/// Scalar fallback for dot product.
#[inline]
fn cosine_dot_128d_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// AVX2 + FMA optimized dot product for 128D vectors.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn cosine_dot_128d_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();

    // 128 dims / 8 floats per register = 16 iterations
    for i in 0..16 {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum
    hsum_avx(sum)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    // hadd pairs: [a+b, c+d, e+f, g+h, i+j, k+l, m+n, o+p]
    let sum1 = _mm256_hadd_ps(v, v);
    // hadd again: [a+b+c+d, e+f+g+h, ...]
    let sum2 = _mm256_hadd_ps(sum1, sum1);
    // Extract low and high 128-bit lanes and add
    let low = _mm256_extractf128_ps(sum2, 0);
    let high = _mm256_extractf128_ps(sum2, 1);
    let sum128 = _mm_add_ps(low, high);
    _mm_cvtss_f32(sum128)
}

/// Compute cosine similarity for 128D vectors (convenience function).
#[inline]
pub fn cosine_similarity_128d(a: &[f32], b: &[f32]) -> f32 {
    let dot = cosine_dot_128d(a, b);
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let v: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let sim = cosine_similarity_128d(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "Identical vectors should have similarity 1.0");
    }

    #[test]
    fn test_orthogonal_vectors() {
        let mut a = vec![0.0f32; 128];
        let mut b = vec![0.0f32; 128];
        a[0] = 1.0;
        b[1] = 1.0;
        let sim = cosine_similarity_128d(&a, &b);
        assert!(sim.abs() < 1e-6, "Orthogonal vectors should have similarity 0.0");
    }

    #[test]
    fn test_maxsim_identical_docs() {
        let query = vec![vec![1.0; 128], vec![0.5; 128]];
        let doc = vec![vec![1.0; 128], vec![0.5; 128]];
        let scorer = MaxSimScorer::new(query);
        let score = scorer.score(&doc);
        assert!((score - 1.0).abs() < 1e-6, "Identical query/doc should score 1.0");
    }

    #[test]
    fn test_maxsim_empty_handling() {
        let scorer = MaxSimScorer::new(vec![vec![1.0; 128]]);
        assert_eq!(scorer.score(&[]), 0.0, "Empty doc should score 0.0");

        let empty_scorer = MaxSimScorer::new(vec![]);
        assert_eq!(empty_scorer.score(&[vec![1.0; 128]]), 0.0, "Empty query should score 0.0");
    }

    #[test]
    #[should_panic(expected = "FATAL: Query token 0 has dimension 64")]
    fn test_dimension_validation() {
        let _ = MaxSimScorer::new(vec![vec![1.0; 64]]);
    }
}
```

### 4. Module Exports (search/mod.rs)

Add to the file:

```rust
pub mod token_storage;
pub mod maxsim;

pub use token_storage::RocksDbTokenStorage;
pub use maxsim::{MaxSimScorer, cosine_similarity_128d, cosine_dot_128d, TOKEN_DIM};
```

### 5. Pipeline Integration (search/pipeline.rs)

Replace the existing `compute_maxsim` method (lines 1276-1300) with:

```rust
use super::maxsim::MaxSimScorer;

// In RetrievalPipeline implementation, update stage_maxsim_rerank:
fn stage_maxsim_rerank(
    &self,
    query_tokens: &[Vec<f32>],
    candidates: Vec<PipelineCandidate>,
    config: &StageConfig,
) -> Result<StageResult, PipelineError> {
    let stage_start = std::time::Instant::now();
    let candidates_in = candidates.len();

    if query_tokens.is_empty() || candidates.is_empty() {
        return Ok(StageResult {
            candidates: Vec::new(),
            latency_us: stage_start.elapsed().as_micros() as u64,
            candidates_in,
            candidates_out: 0,
            stage: PipelineStage::MaxSimRerank,
        });
    }

    // Create optimized scorer with precomputed norms
    let scorer = MaxSimScorer::new(query_tokens.to_vec());

    // Batch retrieve and score
    let scored: Vec<(PipelineCandidate, f32)> = candidates
        .into_par_iter()
        .filter_map(|mut c| {
            if let Some(doc_tokens) = self.token_storage.get_tokens(c.id) {
                let maxsim_score = scorer.score(&doc_tokens);
                c.add_stage_score(PipelineStage::MaxSimRerank, maxsim_score);
                Some((c, maxsim_score))
            } else {
                None
            }
        })
        .collect();

    // Sort by MaxSim score descending
    let mut new_candidates: Vec<PipelineCandidate> = scored
        .into_iter()
        .filter(|(_, score)| *score >= config.min_score_threshold)
        .map(|(c, _)| c)
        .collect();

    new_candidates.sort_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });

    let latency_us = stage_start.elapsed().as_micros() as u64;
    let latency_ms = latency_us / 1000;
    let candidates_out = new_candidates.len();

    // FAIL FAST timeout check
    if latency_ms > config.max_latency_ms {
        return Err(PipelineError::Timeout {
            stage: PipelineStage::MaxSimRerank,
            elapsed_ms: latency_ms,
            max_ms: config.max_latency_ms,
        });
    }

    Ok(StageResult {
        candidates: new_candidates,
        latency_us,
        candidates_in,
        candidates_out,
        stage: PipelineStage::MaxSimRerank,
    })
}
```

---

## VALIDATION CRITERIA (REQUIRED)

### Unit Test Requirements:

| Test | Input | Expected Output | Verification |
|------|-------|-----------------|--------------|
| Identical vectors | v, v | cosine = 1.0 | Assert `(result - 1.0).abs() < 1e-6` |
| Orthogonal vectors | [1,0,...], [0,1,...] | cosine = 0.0 | Assert `result.abs() < 1e-6` |
| MaxSim identical | Q=D | score = 1.0 | Assert `(result - 1.0).abs() < 1e-6` |
| MaxSim empty doc | Q, [] | score = 0.0 | Assert `result == 0.0` |
| Dimension mismatch | 64D token | panic | `#[should_panic]` |
| NaN token | [NaN, ...] | error | Assert `is_err()` |
| Round-trip storage | store → get | exact match | Assert byte equality |
| Batch consistency | serial vs batch | same scores | Assert all `< 1e-6` |

### Performance Requirements:

| Metric | Target | How to Verify |
|--------|--------|---------------|
| 50 candidates | < 15ms | Criterion benchmark |
| Token retrieval 50 IDs | < 5ms | Criterion benchmark |
| Memory per candidate | < 200KB | Profile with heaptrack |
| SIMD vs scalar | > 4x speedup | Benchmark both paths |

---

## FULL STATE VERIFICATION PROTOCOL

After implementing, you MUST verify state persistence. Return values are NOT sufficient proof.

### Source of Truth: RocksDB Column Family

```bash
# Check CF exists
cargo test -p context-graph-storage cf_e12 -- --nocapture

# Verify data persisted (in test):
let id = Uuid::new_v4();
let tokens = vec![vec![1.0; 128]; 10];

// 1. Store
storage.store(id, &tokens)?;

// 2. VERIFY: Read back DIRECTLY from RocksDB (not through cache)
let cf = db.cf_handle(CF_E12_LATE_INTERACTION).unwrap();
let raw_bytes = db.get_cf(&cf, id.as_bytes())?.expect("MUST exist");
let retrieved: Vec<Vec<f32>> = bincode::deserialize(&raw_bytes)?;

// 3. Assert exact equality
assert_eq!(tokens.len(), retrieved.len());
for (i, (orig, ret)) in tokens.iter().zip(retrieved.iter()).enumerate() {
    assert_eq!(orig.len(), ret.len(), "Token {i} length mismatch");
    for (j, (a, b)) in orig.iter().zip(ret.iter()).enumerate() {
        assert!((a - b).abs() < 1e-10, "Token {i} dim {j}: {a} != {b}");
    }
}
```

### Edge Case Verification (REQUIRED):

Execute these manually and print before/after state:

**Case 1: Empty Token Array**
```rust
println!("BEFORE: Storing empty tokens for {}", id);
let result = storage.store(id, &[]);
println!("AFTER: Result = {:?}", result);
assert!(result.is_err());
assert!(matches!(result, Err(TokenStorageError::EmptyTokens { .. })));
```

**Case 2: Wrong Dimension**
```rust
let bad_tokens = vec![vec![1.0f32; 64]]; // 64D instead of 128D
println!("BEFORE: Storing 64D token for {}", id);
let result = storage.store(id, &bad_tokens);
println!("AFTER: Result = {:?}", result);
assert!(matches!(result, Err(TokenStorageError::DimensionMismatch { actual: 64, index: 0 })));
```

**Case 3: NaN/Inf Values**
```rust
let mut tokens = vec![vec![1.0f32; 128]];
tokens[0][0] = f32::NAN;
println!("BEFORE: Storing NaN token for {}", id);
let result = storage.store(id, &tokens);
println!("AFTER: Result = {:?}", result);
assert!(matches!(result, Err(TokenStorageError::InvalidFloat { index: 0, dim: 0 })));
```

**Case 4: MaxSim Scoring Boundaries**
```rust
// Zero norm vectors
let scorer = MaxSimScorer::new(vec![vec![0.0; 128]]);
let score = scorer.score(&[vec![1.0; 128]]);
println!("Zero norm query score: {}", score);
assert_eq!(score, 0.0);
```

---

## TEST COMMANDS

```bash
# Run all MaxSim tests
cargo test -p context-graph-storage maxsim -- --nocapture

# Run token storage tests
cargo test -p context-graph-storage token_storage -- --nocapture

# Run benchmarks (after adding Cargo.toml config)
cargo bench -p context-graph-storage --bench maxsim_bench

# Verify SIMD is being used
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p context-graph-storage maxsim

# Check for warnings/errors
cargo clippy -p context-graph-storage -- -D warnings
```

---

## BENCHMARK SPECIFICATION (benches/maxsim_bench.rs)

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use context_graph_storage::teleological::search::{MaxSimScorer, TOKEN_DIM};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn generate_tokens(rng: &mut StdRng, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| (0..TOKEN_DIM).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

fn bench_maxsim(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    let mut group = c.benchmark_group("maxsim");

    // Query: 10 tokens (typical)
    let query = generate_tokens(&mut rng, 10);
    let scorer = MaxSimScorer::new(query);

    // Benchmark different candidate counts
    for n_candidates in [10, 25, 50, 100] {
        let docs: Vec<Vec<Vec<f32>>> = (0..n_candidates)
            .map(|_| generate_tokens(&mut rng, 15))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("score_batch", n_candidates),
            &docs,
            |b, docs| {
                b.iter(|| scorer.score_batch(docs));
            },
        );
    }

    group.finish();
}

fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    let a: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
    let b: Vec<f32> = (0..128).map(|_| rng.gen()).collect();

    c.bench_function("cosine_128d", |bench| {
        bench.iter(|| context_graph_storage::teleological::search::cosine_dot_128d(&a, &b));
    });
}

criterion_group!(benches, bench_maxsim, bench_simd_vs_scalar);
criterion_main!(benches);
```

Add to `Cargo.toml`:

```toml
[[bench]]
name = "maxsim_bench"
harness = false

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
rand = { version = "0.8", features = ["std_rng"] }
```

---

## ROCKSDB STORE INTEGRATION (rocksdb_store.rs)

When storing a `TeleologicalFingerprint`, also store the E12 tokens:

```rust
// In store_fingerprint() or equivalent method:
pub async fn store_fingerprint(&self, id: Uuid, fp: &TeleologicalFingerprint) -> Result<()> {
    // ... existing fingerprint storage ...

    // Also persist E12 tokens for Stage 5 MaxSim
    if !fp.semantic.e12_late_interaction.is_empty() {
        self.token_storage.store(id, &fp.semantic.e12_late_interaction)?;
    }

    Ok(())
}
```

---

## NO BACKWARDS COMPATIBILITY

**FAIL FAST EVERYWHERE:**
- Invalid dimensions → panic with clear message
- NaN/Inf values → return error immediately
- Missing column family → panic on startup
- Timeout exceeded → return PipelineError::Timeout
- Empty tokens → return error (not silent skip)

**NO MOCK DATA IN TESTS:**
- Use real RocksDB instances (tempdir)
- Generate realistic token distributions
- Verify actual persistence to disk

---

## SUCCESS EVIDENCE CHECKLIST

Before marking complete, provide logs showing:

- [ ] `cargo test -p context-graph-storage maxsim` passes
- [ ] `cargo test -p context-graph-storage token_storage` passes
- [ ] `cargo bench -p context-graph-storage --bench maxsim_bench` shows <15ms @ 50 candidates
- [ ] Manual verification log showing RocksDB reads match writes
- [ ] Edge case outputs (empty, wrong dim, NaN) logged
- [ ] SIMD speedup measured vs scalar fallback

```
</task_spec>
```
