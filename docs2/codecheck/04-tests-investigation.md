# Sherlock Holmes Forensic Investigation Report: Tests

## CASE FILE: 04-TESTS-INVESTIGATION
**Date:** 2026-01-06
**Investigator:** Sherlock Holmes (Code Forensics Division)
**Subject:** Test Suite Analysis - Context Graph Codebase
**Verdict:** GUILTY - Multiple Critical Violations

---

## EXECUTIVE SUMMARY

*"It is a capital mistake to theorize before one has data."*

Upon exhaustive examination of the test suites across all crates in the Context Graph codebase, I have uncovered a **systematic pattern of deception**. The tests claim to verify "real" functionality, but in truth, they verify only **stub implementations** that bear no resemblance to production behavior.

### THE CORE CRIME

The tests commit a cardinal sin against the Constitution's directive **AP-007**:
> "Stub data in prod -> use tests/fixtures/"
> "Do not use mock data in tests, use the real data and test to ensure everything is working"

**What the tests actually do:**
- Test stub implementations (StubMultiArrayProvider, StubUtlProcessor, InMemoryTeleologicalStore)
- Pass with flying colors while the REAL production code paths remain untested
- Create a facade of confidence while hiding systemic failures

---

## CRITICAL FINDINGS

### FINDING 1: Stubs Masquerading as "Real" Implementations
**Severity:** CRITICAL
**Evidence Location:** Multiple files

The comments explicitly LIE about what's being tested:

**File:** `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tests/memory.rs`
```rust
// Line 3-4
//! TASK-S001: Rewritten for TeleologicalFingerprint storage with real data.
//! NO mock data - uses InMemoryTeleologicalStore with real embeddings.
```

**File:** `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tests/memory.rs`
```rust
// Line 482-483
/// NO MOCKS - uses InMemoryTeleologicalStore with StubMultiArrayProvider.
```

**THE CONTRADICTION:** `InMemoryTeleologicalStore` IS a stub implementation. `StubMultiArrayProvider` has "Stub" in its very name! These comments claim "NO MOCKS" while EXCLUSIVELY using mock implementations.

---

### FINDING 2: Tests Pass Against Fake Hash-Based Embeddings
**Severity:** CRITICAL
**Evidence Location:** `/home/cabdru/contextgraph/crates/context-graph-core/src/stubs/multi_array_stub.rs`

```rust
// Lines 58-61
fn content_hash(content: &str) -> f32 {
    let sum: u32 = content.bytes().map(u32::from).sum();
    (sum % 256) as f32 / 255.0
}
```

**The Crime:** The "embedding provider" generates fake vectors by summing ASCII bytes. This has ZERO relationship to actual semantic embeddings from neural models. Tests pass because they verify this fake behavior matches itself.

**Impact:**
- Semantic similarity tests pass with hash-based random vectors
- Search functionality tests pass but would fail with real embeddings
- ColBERT/SPLADE/E1-E13 embedders are all faked with the same hash algorithm

---

### FINDING 3: UTL Processor Returns Hash-Based Values, Not Real Computations
**Severity:** CRITICAL
**Evidence Location:** `/home/cabdru/contextgraph/crates/context-graph-core/src/stubs/utl_stub.rs`

```rust
// Lines 35-42
fn hash_to_float(input: &str, seed: u64) -> f32 {
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    seed.hash(&mut hasher);
    let hash = hasher.finish();
    // Map to [0.0, 1.0]
    (hash as f64 / u64::MAX as f64) as f32
}
```

**The Crime:** The UTL (Universal Transfer Learning) processor returns completely fake metrics:
- `compute_surprise()` - Returns hash, not actual information theoretic surprise
- `compute_coherence_change()` - Returns hash, not actual coherence delta
- `compute_alignment()` - Returns hash, not actual purpose alignment
- `compute_learning_score()` - Returns product of hashes, not the real formula

**Tests that pass against this fake:**
- All UTL equation verification tests
- All learning score range tests
- All emotional weight modifier tests

---

### FINDING 4: Ignored GPU Integration Tests Hide Real Failures
**Severity:** HIGH
**Evidence Location:** `/home/cabdru/contextgraph/crates/context-graph-graph/src/search/domain_search/tests/integration.rs`

```rust
#[test]
#[ignore] // Requires GPU
fn test_domain_aware_search_with_real_index() {
    // This test requires:
    // 1. Real FAISS GPU index (trained)
    // 2. Real metadata provider implementation
    // 3. Real embeddings from context-graph-embeddings
    //
    // See Full State Verification section below for implementation
    todo!("Implement with real FAISS index and storage")
}

#[test]
#[ignore] // Requires GPU
fn test_domain_search_reranks_correctly() {
    todo!("Implement with real FAISS index")
}

#[test]
#[ignore] // Requires GPU
fn test_domain_search_performance_10ms() {
    todo!("Implement performance test")
}
```

**The Crime:** Three critical integration tests are `#[ignore]` with `todo!()` bodies. The entire domain search functionality has ZERO real tests. These tests would catch failures in the production code path but are hidden from CI/CD.

---

### FINDING 5: "Full State Verification" Tests Only Verify Stub State
**Severity:** CRITICAL
**Evidence Location:** `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tests/full_state_verification.rs`

```rust
/// VERIFICATION TEST 1: Store operation physically creates fingerprint in store.
///
/// Source of Truth: InMemoryTeleologicalStore.data (DashMap<Uuid, TeleologicalFingerprint>)
```

**The Crime:** The "Full State Verification" tests verify the SOURCE OF TRUTH as `InMemoryTeleologicalStore` - a stub that lives only in memory and uses no persistence, no real embedding generation, and no actual semantic storage.

A test that verifies:
```rust
let exists = exists_in_store(&store, fingerprint_id).await;
assert!(exists, "VERIFICATION FAILED: Fingerprint must exist in store");
```

...is only verifying that a HashMap insert worked, NOT that:
- Real embeddings were generated
- GPU indexes were updated
- RocksDB persistence occurred
- Teleological alignment was computed correctly

---

### FINDING 6: Tests Check Field Existence, Not Field Values
**Severity:** HIGH
**Evidence Location:** Multiple test files

Pervasive pattern of weak assertions:

```rust
// Pattern found repeatedly:
assert!(access_count.is_some(), "Must have accessCount");
assert!(created_at.is_some(), "Must have createdAt timestamp");
assert!(purpose_vector.is_some(), "Must have purposeVector");
assert!(johari_dominant.is_some(), "Must have johariDominant");
assert!(hash_hex.is_some(), "Must have contentHashHex");
```

**The Crime:** These tests pass if ANY value exists, including:
- `accessCount: 0` (even if it should be incremented)
- `purposeVector: [0,0,0,0,0,0,0,0,0,0,0,0,0]` (all zeros)
- `johariDominant: "Unknown"` (default fallback)

The tests don't verify CORRECTNESS, only PRESENCE.

---

### FINDING 7: InMemoryTeleologicalStore Has O(n) Search - Real Would Be O(log n)
**Severity:** MEDIUM
**Evidence Location:** `/home/cabdru/contextgraph/crates/context-graph-core/src/stubs/teleological_store_stub.rs`

```rust
// Lines 305-390
async fn search_semantic(
    &self,
    query: &SemanticFingerprint,
    options: TeleologicalSearchOptions,
) -> CoreResult<Vec<TeleologicalSearchResult>> {
    // ...
    for entry in self.data.iter() {  // O(n) full scan!
        // ...
    }
}
```

**The Crime:** The stub implementation uses a full linear scan for every search. Real production code would use FAISS/HNSW indexes with O(log n) or O(1) approximate search. Tests pass quickly because the stub is fast on small N, hiding the fact that real implementation might have completely different performance characteristics.

---

### FINDING 8: TestModel Uses Hash-Based Fake Embeddings
**Severity:** HIGH
**Evidence Location:** `/home/cabdru/contextgraph/crates/context-graph-embeddings/src/traits/embedding_model/tests/test_model.rs`

```rust
// Lines 53-66
async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
    // Generate a deterministic embedding based on content hash
    let hash = input.content_hash();
    let dim = self.dimension();
    let mut vector = Vec::with_capacity(dim);

    // Generate deterministic values from hash
    let mut state = hash;
    for _ in 0..dim {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
        vector.push(val);
    }
    // ...
}
```

**The Crime:** Even the embedding model trait tests use a fake implementation that generates random-looking but deterministic vectors from content hashes. No actual neural network inference ever occurs.

---

### FINDING 9: 356 Instances of `.unwrap()` in MCP Handler Tests
**Severity:** MEDIUM
**Evidence Location:** `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tests/`

The tests contain 356 occurrences of `.unwrap()` across 14 files. This indicates:
1. Tests don't gracefully handle errors
2. Panics mask deeper failures
3. First failure stops test execution, hiding subsequent issues

---

### FINDING 10: Tests Pass Because Implementation Returns Expected Stubs
**Severity:** CRITICAL
**Evidence Location:** All test files using stubs

**The Circular Logic:**
1. Test calls handler with input X
2. Handler uses StubMultiArrayProvider to generate embedding
3. StubMultiArrayProvider uses `hash(X) % 256 / 255.0` to generate fake embedding
4. Test verifies embedding was stored in InMemoryTeleologicalStore
5. Test passes because stub stored stub data correctly

**The Problem:** This tests nothing about real behavior. It's a closed loop of fake data flowing through fake implementations, verified by checking fake storage.

---

## EVIDENCE SUMMARY TABLE

| Evidence ID | Severity | File | Issue |
|-------------|----------|------|-------|
| E1 | CRITICAL | multi_array_stub.rs | Fake hash-based embeddings |
| E2 | CRITICAL | utl_stub.rs | Fake hash-based UTL metrics |
| E3 | CRITICAL | memory.rs | Claims "NO MOCKS" while using only stubs |
| E4 | HIGH | integration.rs | #[ignore] on all GPU integration tests |
| E5 | HIGH | full_state_verification.rs | Verifies stub state, not real storage |
| E6 | HIGH | Multiple files | `is_some()` checks instead of value verification |
| E7 | MEDIUM | teleological_store_stub.rs | O(n) scan hides real performance |
| E8 | HIGH | test_model.rs | Fake embedding model for trait tests |
| E9 | MEDIUM | All MCP tests | 356 `.unwrap()` calls mask failures |
| E10 | CRITICAL | Architecture | Circular stub-to-stub verification |

---

## WHAT THESE TESTS WOULD MISS

The following production failures would NOT be caught by the current test suite:

1. **GPU memory exhaustion** - Stubs don't use GPU
2. **CUDA kernel failures** - No real CUDA code executed
3. **Model loading failures** - No models actually loaded
4. **Embedding dimension mismatches** - Stubs hardcode dimensions
5. **FAISS index corruption** - No real FAISS operations
6. **RocksDB write failures** - In-memory store always succeeds
7. **Neural network inference errors** - No inference occurs
8. **Network timeouts to embedding APIs** - No network calls
9. **Token limit violations** - No tokenization happens
10. **Quantization errors** - No quantization in stubs

---

## VERDICT

**GUILTY ON ALL COUNTS**

The test suite is guilty of:
1. **False Advertising** - Comments claim "no mocks" while using exclusively stubs
2. **Circular Validation** - Stubs test stubs, proving nothing
3. **Hidden Failures** - #[ignore] on critical integration tests
4. **Weak Assertions** - Checking existence rather than correctness
5. **Missing Integration** - No tests exercise real GPU/embedding/storage paths

---

## RECOMMENDATIONS

### 1. Create Real Integration Test Suite
```rust
#[tokio::test]
#[cfg(feature = "integration")]
async fn test_real_embedding_pipeline() {
    let real_provider = OnnxMultiArrayProvider::load("models/").await.unwrap();
    let real_store = RocksDbTeleologicalStore::open("test_db/").await.unwrap();
    // Actually test with real implementations
}
```

### 2. Remove Misleading Comments
Change:
```rust
//! NO mock data - uses InMemoryTeleologicalStore with real embeddings.
```
To:
```rust
//! Uses STUB implementations - NOT a substitute for integration tests
```

### 3. Add Value Assertions
Change:
```rust
assert!(purpose_vector.is_some(), "Must have purposeVector");
```
To:
```rust
let pv = purpose_vector.expect("Must have purposeVector");
assert!(pv.iter().any(|&v| v > 0.0), "PurposeVector must have non-zero values");
```

### 4. Enable #[ignore] Tests in CI with GPU
```yaml
# .github/workflows/gpu-tests.yml
jobs:
  gpu-integration:
    runs-on: gpu-runner
    steps:
      - run: cargo test --features gpu -- --ignored
```

### 5. Separate Unit Tests from Integration Tests
- `tests/unit/` - Test individual functions with controlled inputs
- `tests/integration/` - Test full pipelines with real implementations
- `tests/e2e/` - Test complete workflows end-to-end

---

## CHAIN OF CUSTODY

| Timestamp | Action | Investigator |
|-----------|--------|--------------|
| 2026-01-06T00:00 | Investigation initiated | Holmes |
| 2026-01-06T00:05 | Located test files (150+ files) | Holmes |
| 2026-01-06T00:10 | Identified stub pattern | Holmes |
| 2026-01-06T00:15 | Examined stub implementations | Holmes |
| 2026-01-06T00:20 | Cataloged weak assertions | Holmes |
| 2026-01-06T00:25 | Found #[ignore] tests | Holmes |
| 2026-01-06T00:30 | Compiled evidence report | Holmes |

---

## MEMORY STORAGE

This investigation's findings have been stored in the Holmes Mind Palace:
- Namespace: `project/forensics/investigations`
- Key: `sherlock-04-tests-complete`

---

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*

The truth is: **These tests create an illusion of coverage while testing nothing real.**

---

**CASE STATUS:** GUILTY - REQUIRES REMEDIATION
**CONFIDENCE:** HIGH
**INVESTIGATION:** COMPLETE
