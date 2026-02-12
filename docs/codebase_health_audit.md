# Context Graph Codebase Health Audit

**Date:** 2026-02-12
**Branch:** casetrack
**Build Status:** Release build succeeds with ~60+ warnings
**Tests:** 655 MCP + 2780 core pass, zero failures

---

## Executive Summary

A comprehensive audit of the Context Graph codebase (~1,485 .rs files, ~3.8M lines across 10+ crates) reveals **4 critical issues**, **12 high-severity issues**, and numerous medium/low findings. The most concerning pattern is code that **appears to work but silently produces wrong results** — particularly around error swallowing with `.ok()` / `.unwrap_or(0.0)` and stub implementations that return hardcoded values without indication of failure.

### Severity Counts

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 4 | Silent data corruption, breaking serialization, production panics |
| HIGH | 12 | Dead code masking failures, duplicate constants creating skew risk |
| MEDIUM | 20+ | Redundant implementations, unused fields/imports, incomplete features |
| LOW | 15+ | Style issues, debug artifacts, unnecessary clones |

---

## CRITICAL Issues

### C1. Bincode Serialization Still Used Despite Known Breakage

**Files:** `crates/context-graph-storage/src/code/store.rs` (20+ call sites: lines 184, 188, 234, 259, 370, 427, 466, 486, 513, 587, 697, 743, 754, 766, 774, 788, 795, 808, 817, 832)

**Problem:** The project memory explicitly states: *"bincode + skip_serializing_if = BROKEN; use JSON for all provenance/SourceMetadata"*. SourceMetadata correctly uses JSON (`rocksdb_store/source_metadata.rs`), but `CodeEntity`, `SemanticFingerprint`, and `CodeFileIndexEntry` are still serialized with bincode.

**Impact:** Silent data corruption risk on schema evolution. Bincode is order-dependent — adding, removing, or reordering fields will silently produce corrupt data without errors. This code **appears to work** today but will break on the next schema change.

**Fix:** Migrate all 20+ bincode call sites in `code/store.rs` to serde_json, matching the pattern used in `source_metadata.rs`.

---

### C2. RocksDB Error Handling Silently Returns Wrong Values

**File:** `crates/context-graph-storage/src/rocksdb_backend/memex_impl.rs`

**Lines 90-93** — Key count estimation:
```rust
let count = self.db
    .property_int_value_cf(cf, "rocksdb.estimate-num-keys")
    .ok()           // silently swallows RocksDB error
    .flatten()
    .unwrap_or(0);  // returns 0 on error
```

**Lines 106-113** — Storage size calculation:
```rust
if let Some(size) = self.db
    .property_int_value_cf(cf, "rocksdb.total-sst-files-size")
    .ok()           // silently swallows RocksDB error
    .flatten()
{
    total_bytes += size;  // simply skipped on error, corrupting total
}
```

**Impact:** If RocksDB property reads fail (disk error, CF not found, etc.), these functions report 0 keys and undercount storage. Callers (including MCP tools like `get_memetic_status`) present these as real metrics. This **appears to work** but silently reports wrong values.

---

### C3. Feature-Gate Logic Allows Deprecation Bypass

**Files:**
- `crates/context-graph-causal-agent/src/service/mod.rs:244-251` — `CausalDiscoveryService::new()`
- `crates/context-graph-graph-agent/src/service/mod.rs:143-154` — `GraphDiscoveryService::with_config()`

**Problem:** Both deprecated constructors use:
```rust
#[cfg_attr(not(feature = "test-mode"), deprecated(...))]
```

This means with `test-mode` feature enabled, the deprecation is completely invisible. The deprecated constructors create services **without real ML models** (`E5EmbedderActivator` without `CausalModel`, `E8Activator` without `GraphModel`), meaning the services appear functional but produce no meaningful embeddings.

**Active callers of deprecated APIs in production code:**
- `crates/context-graph-benchmark/src/bin/causal_e2e_bench.rs:786`
- `crates/context-graph-benchmark/src/bin/graph_linking_gpu_bench.rs:1108`
- `crates/context-graph-graph-agent/examples/benchmark_graph.rs:260`
- `crates/context-graph-graph-agent/examples/benchmark_graph_code.rs:469`
- `crates/context-graph-graph-agent/examples/benchmark_graph_multi.rs:259`

---

### C4. GPU Device Accessor Panics Without Initialization

**File:** `crates/context-graph-embeddings/src/gpu/device/accessors.rs:42`

```rust
GPU_DEVICE.get().expect(...)  // PANICS if GPU_DEVICE not initialized
```

**Impact:** Any code path that touches GPU device without prior initialization causes an unrecoverable panic. No graceful fallback exists.

---

## HIGH-Severity Issues

### H1. Unreachable Code in Benchmark Runner

**File:** `crates/context-graph-benchmark/src/realdata/runner.rs:149`

```rust
#[cfg(not(feature = "real-embeddings"))]
let embedded = {
    let _ = embedder;
    return Err(RealDataError::EmbedError(
        "Real embeddings require 'real-embeddings' feature".to_string(),
    ));
};
self.embedded = Some(embedded);  // UNREACHABLE
Ok(self.embedded.as_ref().unwrap())
```

The `return` inside the `#[cfg]` block makes lines 149+ unreachable. The compiler warns about this. Additionally, `dataset` (line 125) and `embedded` (line 142) are unused variables.

---

### H2. E5 Causal Score Silent Parse Failures

**File:** `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs` (lines 681, 800)

```rust
.unwrap_or(0.0) as f32;  // parse failure → 0.0
```

E5 causal scores default to 0.0 when string-to-float parsing fails. Since the causal gate threshold is 0.30, a score of 0.0 is treated as "non-causal". This means **parse errors silently suppress causal detection** — the system appears to work but misclassifies memories.

---

### H3. Token Batch Retrieval Silently Skips Missing Memories

**File:** `crates/context-graph-storage/src/teleological/search/token_storage.rs:229-236`

```rust
/// Returns a vector of (id, tokens) pairs for all found memories.
/// Missing memories are silently skipped.
pub fn get_batch(&self, ids: &[Uuid]) -> TokenStorageResult<Vec<(Uuid, Vec<Vec<f32>>)>>
```

Callers cannot distinguish between "no memories found" and "some memories missing due to corruption". This is documented behavior but creates a data-loss blind spot.

---

### H4. Duplicate Constitutional Constants — Version Skew Risk

**`MAX_WEIGHTED_AGREEMENT = 8.5`** defined in 4 separate locations:
1. `crates/context-graph-core/src/injection/candidate.rs:157`
2. `crates/context-graph-core/src/clustering/manager.rs`
3. `crates/context-graph-embeddings/src/models/pretrained/weight_projection/constants.rs:41`
4. `crates/context-graph-mcp/src/handlers/tools/topic_dtos.rs`

**`TOPIC_THRESHOLD = 2.5`** defined in 4 separate locations:
1. `crates/context-graph-core/src/clustering/manager.rs`
2. `crates/context-graph-core/src/injection/priority.rs`
3. `crates/context-graph-embeddings/src/models/pretrained/weight_projection/constants.rs:44`
4. `crates/context-graph-mcp/src/handlers/tools/topic_dtos.rs`

**Impact:** If one location is updated and others aren't, the system silently uses inconsistent thresholds. These are constitutional constants (ARCH-09) requiring a single source of truth.

---

### H5. 10+ Duplicate Cosine Similarity Implementations

Independent implementations of `cosine_similarity(a, b) -> f32`:

| # | File | Function |
|---|------|----------|
| 1 | `storage/src/code/store.rs` | `cosine_similarity_with_norm()` |
| 2 | `storage/src/teleological/search/maxsim.rs` | `cosine_similarity_128d()` |
| 3 | `storage/src/teleological/search/maxsim.rs` | `cosine_similarity_scalar()` |
| 4 | `storage/src/teleological/search/maxsim.rs` | `cosine_similarity_avx2()` |
| 5 | `storage/src/teleological/search/temporal_boost.rs` | `cosine_similarity()` |
| 6 | `storage/src/teleological/indexes/metrics.rs:137` | `cosine_similarity()` |
| 7 | `storage/src/teleological/rocksdb_store/helpers.rs` | `compute_cosine_similarity()` |
| 8 | `embeddings/src/warm/inference/validation.rs` | `cosine_similarity()` |
| 9 | `storage/tests/e9_storage_roundtrip_test.rs` | test helper |
| 10 | `embeddings/tests/qodo_embed_integration.rs` | test helper |
| 11 | `embeddings/tests/e9_typo_tolerance_test.rs` | test helper |
| 12 | `embeddings/tests/e9_vector_differentiation_test.rs` | test helper |

The AVX2/scalar variants in maxsim.rs are intentional SIMD optimizations, but the remaining 8+ are identical standard implementations. Risk of subtle divergence (e.g., one handles zero-norm differently).

---

### H6. 15+ Duplicate L2 Norm Calculations

Pattern `v.iter().map(|x| x*x).sum::<f32>().sqrt()` appears in:

- `storage/src/code/store.rs`
- `storage/src/graph_edges/builder.rs`
- `storage/src/teleological/search/maxsim.rs`
- `storage/src/teleological/rocksdb_store/causal_hnsw_index.rs`
- `embeddings/src/quantization/float8.rs`
- `embeddings/src/quantization/pq8/training.rs`
- `embeddings/src/models/custom/temporal_recent/compute.rs`
- `embeddings/src/models/custom/hdc/encoding.rs`
- `embeddings/src/models/custom/temporal_positional/session_signature.rs`

Should be extracted to a shared `compute_l2_norm()` utility.

---

### H7. GNN Enhancement Stage — 100-Line Dead Function

**File:** `crates/context-graph-storage/src/teleological/search/pipeline/stages.rs:557-657`

```rust
#[allow(dead_code)]
pub fn stage_gnn_enhance(...) -> Result<StageResult, PipelineError> {
    // 100 lines of R-GCN implementation
    // Comment at line 627: "For now, we use simplified GNN scoring..."
}
```

Full R-GCN stage implementation exists but is marked `#[allow(dead_code)]` and never called. The comments admit it's a stub. This is 100 lines of unmaintained dead code.

---

### H8. Orphaned Binary File Not in Cargo.toml

**File:** `crates/context-graph-benchmark/src/bin/e10_modifier_tuning.rs`

This binary source file exists but is **not declared** in `Cargo.toml`'s `[[bin]]` section. It compiles as part of the crate but is invisible to `cargo run --bin`. It imports `HashMap` which is unused.

---

### H9. Weight Profile Arrays — 3 Divergent Definitions

Three separate 13-element weight arrays serve similar purposes:

1. `crates/context-graph-core/src/retrieval/config.rs` — `SPACE_WEIGHTS`
2. `crates/context-graph-storage/src/teleological/rocksdb_store/search.rs` — `DEFAULT_SEMANTIC_WEIGHTS`
3. `crates/context-graph-embeddings/src/models/pretrained/weight_projection/constants.rs` — `DEFAULT_CATEGORY_WEIGHTS`

Any change to embedder weights must be synchronized across three locations.

---

### H10. Stub Implementations Shipped in Production Crate

| Stub File | What It Fakes | Risk |
|-----------|---------------|------|
| `context-graph-cuda/src/stub.rs` (219 lines) | GPU vector ops (cosine, dot, normalize) as CPU fallback | Marked "TEST ONLY" but compiles in release |
| `context-graph-core/src/stubs/graph_index.rs` (502 lines) | In-memory brute-force O(n*d) vector index | `rebuild()` is a no-op that returns `Ok(())` |
| `context-graph-core/src/embeddings/stubs.rs` (677 lines) | All 13 embedders using content hash | Generates fake deterministic embeddings |
| `context-graph-core/src/stubs/multi_array_stub.rs` (735 lines) | Full MultiArrayEmbeddingProvider | All 13 embedders produce fake embeddings |
| `context-graph-graph-agent/src/stubs.rs` (95 lines) | GraphDiscoveryService with unloaded LLM | `run_discovery_cycle()` returns `Err(LlmNotInitialized)` |

Total: **2,228 lines** of stub code in production crates. While gated behind test-mode features, the code compiles in release builds.

---

### H11. Model Pooler Loading Silently Skipped

**File:** `crates/context-graph-embeddings/src/gpu/model_loader/loader.rs:203`

```rust
let pooler = load_pooler(&vb, &config, model_dir, prefix).ok();
```

If the pooler fails to load, `.ok()` converts the error to `None` silently. Downstream code may produce degraded embeddings without any indication of why.

---

### H12. CUDA Memory Pool Reports 0 on Lock Failure

**File:** `crates/context-graph-embeddings/src/gpu/memory/pool.rs:59`

```rust
self.inner.read().ok().map(|t| t.available()).unwrap_or(0)
```

If the RwLock is poisoned (another thread panicked while holding it), this reports 0 bytes available instead of propagating the error. Callers cannot distinguish "no GPU memory" from "lock poisoned".

---

## MEDIUM-Severity Issues

### M1. Unused Struct Fields Across Benchmarks

| File | Struct | Unused Fields |
|------|--------|---------------|
| `benchmark_causal.rs:28-34` | `Chunk` | `id`, `doc_id`, `word_count` |
| `benchmark_causal_large.rs:27-34` | `Chunk` | `title`, `source_dataset` |
| `benchmark_causal_enhanced.rs:31` | `Chunk` | `id` |
| `causal_provenance_e2e_bench.rs:809-811` | (unnamed) | `cause`, `effect`, `explanation` |
| `weight_profile_bench.rs:210-215` | `ProfileTestResult` | `target_content_type`, `precision_at_3`, `precision_at_10` |
| `e7_tuning.rs:133-135` | `TuningQueryResult` | `query_id`, `query_type` |

---

### M2. Unused Imports Across 15+ Files

| File | Unused Import(s) |
|------|-----------------|
| `benchmark_causal.rs:12,15` | `Duration`, `CausalLinkDirection` |
| `benchmark_causal_enhanced.rs:18` | `CausalLinkDirection` |
| `causal_provenance_e2e_bench.rs:61,66` | `Arc`, `Uuid` |
| `weight_profile_bench.rs:22` | `HashMap` |
| `report.rs:12` | `AblationResults`, `CrossEmbedderAnalysis`, `EmbedderResults`, `FusionResults` |
| `temporal_injector.rs:17` | `Timelike` |
| `runner.rs:742` | `TempDir` |
| `device.rs:19` | `CUresult` |
| `gpu_bench.rs:16-17` | `Write`, `Instant` |
| `sparse_bench.rs:19,22,28` | `HashSet`, `Instant`, `Uuid` |
| `e10_modifier_tuning.rs:27` | `HashMap` |
| `unified_realdata.rs:32,38` | `Serialize`, `Deserialize`, `QueryGroundTruth` |

---

### M3. Unused `#[allow(dead_code)]` Fields in Production Code

| File | Field | Note |
|------|-------|------|
| `pipeline/execution.rs:33-34` | `multi_search: MultiEmbedderSearch` | "Reserved for enhanced RRF" |
| `graph_edges/builder.rs:164-165` | `multi_array_provider: Arc<dyn ...>` | Injected but never used |
| `teleological/indexes/hnsw_impl/types.rs:167-205` | `with_config()` constructor | Alternative constructor never called |
| `rocksdb_backend/core.rs:130-131` | `cache: Cache` | RocksDB manages its own cache |
| `teleological/rocksdb_store/store.rs:76-77` | `cache: Cache` | Same — allocated, never accessed |
| `batch/processor/core.rs:55-56` | `registry: Arc<ModelRegistry>` | Stored but never referenced |

---

### M4. Dead Functions Never Called

| File | Function | Lines |
|------|----------|-------|
| `weight_profile_bench.rs:259` | `print_results_table()` | Never invoked |
| `runner.rs:744` | `create_test_dataset()` | Never invoked |
| `session_signature.rs:98-104` | `signature_similarity()` | Public but no production callers |

---

### M5. Duplicate Query Validation Logic

Two near-identical `validate_query()` implementations:
1. `storage/src/teleological/search/single/search.rs` (~lines 50-70)
2. `storage/src/teleological/search/multi/executor.rs` (~lines 80-110)

Both check for empty query and dimension mismatch. Should be extracted to shared utility.

---

### M6. 6 Error Types That Could Be Consolidated

| Error Type | File |
|-----------|------|
| `CodeStorageError` | `storage/src/code/error.rs` |
| `StorageError` | `storage/src/rocksdb_backend/error.rs` |
| `GraphEdgeStorageError` | `storage/src/graph_edges/types.rs` |
| `SearchError` | `storage/src/teleological/search/error.rs` |
| `TeleologicalStoreError` | `storage/src/teleological/rocksdb_store/types.rs` |
| `TokenStorageError` | `storage/src/teleological/search/token_storage.rs` |

All wrap RocksDB errors with context-specific variants. Could benefit from a shared error hierarchy.

---

### M7. 10+ Duplicate Test Helpers for Normalized Embeddings

Pattern `let val = 1.0 / (dim as f32).sqrt(); vec![val; dim]` is reimplemented in:

- `storage/tests/storage_integration/common.rs` — `create_valid_embedding()`
- `storage/src/rocksdb_backend/tests_node.rs` — `create_normalized_embedding()`
- `storage/src/rocksdb_backend/tests_embedding.rs` — `create_normalized_embedding()`
- `storage/src/serialization/tests/node_tests.rs` — `create_normalized_embedding()`
- `storage/src/serialization/tests/embedding_tests.rs` — `create_normalized_embedding()`
- `storage/examples/basic_storage.rs`
- `storage/examples/marblestone_edges.rs`
- `core/src/types/memory_node/tests_node.rs`
- `core/src/teleological/comparator/tests.rs`

---

### M8. Unnecessary Cloning Pattern

**File:** `crates/context-graph-core/src/stubs/teleological_store_stub/trait_impl.rs`

```rust
Ok(self.data.get(&id).map(|r| r.clone()))  // 10+ occurrences
```

Clones entire fingerprints (768D vectors x 13 embedders) on every retrieval. Could use `Arc` or return references.

---

### M9. Unnecessary Parentheses

**File:** `crates/context-graph-benchmark/src/tuning/e7_tuning.rs:581`

```rust
(h as f32 / u64::MAX as f32)  // Unnecessary outer parentheses
```

---

### M10. Pre-release Dependency Lock-in

**Root Cargo.toml:**
```toml
candle-core = { version = "0.9.2-alpha", features = ["cuda"] }
candle-nn = { version = "0.9.2-alpha", features = ["cuda"] }
```

Tightly coupled to unstable alpha versions with no fallback strategy.

---

## LOW-Severity Issues

### L1. Benchmark println! Spam

`crates/context-graph-storage/benches/e11_quality_bench.rs` contains 30+ `println!()` calls that pollute criterion benchmark output.

### L2. Test eprintln! Spam

`crates/context-graph-embeddings/src/storage/full_state_verification.rs:91-117` — massive stderr output for a single test case.

### L3. Unused Variables in Benchmark Runners

12+ unused variables across `e1_semantic.rs`, `runner.rs`, `sparse.rs`, `unified_realdata.rs`, `arch_compliance.rs`. See M2 section for full list.

### L4. Platform-Specific Gap

`crates/context-graph-storage/Cargo.toml` has `[target.'cfg(unix)'.dependencies] libc = "0.2"` for file locking with no Windows equivalent. Fails silently on non-Unix.

### L5. Image Crate Feature Scope

`crates/context-graph-embeddings/Cargo.toml` enables PNG/JPEG support globally even though only E7 (code embeddings) uses it.

---

## Recommended Fix Priority

### Immediate (Data Integrity)
1. **C1** — Migrate `code/store.rs` bincode → JSON (20+ sites)
2. **C2** — Add proper error propagation in `memex_impl.rs` RocksDB calls
3. **H2** — Replace `unwrap_or(0.0)` in `memory_tools.rs` with proper error handling

### Short-Term (Correctness)
4. **H4** — Consolidate `MAX_WEIGHTED_AGREEMENT` and `TOPIC_THRESHOLD` to single definitions in `context-graph-core/src/constants.rs`
5. **C3** — Replace all deprecated constructor calls in benchmarks/examples with `with_models()`
6. **H1** — Fix unreachable code in `runner.rs` cfg-dependent return
7. **H8** — Register `e10_modifier_tuning.rs` in Cargo.toml or remove it

### Medium-Term (Code Quality)
8. **H5/H6** — Extract cosine_similarity and L2 norm to shared math utilities
9. **H7** — Remove or complete the 100-line dead GNN stage
10. **H10** — Audit stub code to ensure it cannot be reached in release builds
11. **M2** — Run `cargo fix --workspace` to clean unused imports
12. **M6** — Consider error type consolidation

### Long-Term (Maintenance)
13. **H9** — Unify weight profile arrays into single versioned definition
14. **M7** — Create shared test fixture crate for normalized embeddings
15. **M8** — Replace clone patterns with Arc in stub store
