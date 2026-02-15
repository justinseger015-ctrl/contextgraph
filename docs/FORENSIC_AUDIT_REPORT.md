# Context Graph Forensic Audit Report

**Date**: 2026-02-15
**Branch**: casetrack
**Scope**: Full codebase audit for phantom functionality, silent failures, and hidden breakage
**Method**: 5 parallel forensic investigation agents covering MCP handlers, storage, embeddings, tests, and server infrastructure

---

## Executive Summary

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 1 | Unbounded `read_line` in daemon proxy (OOM vector) |
| HIGH | 1 | HNSW memory leak on vector deletion |
| MEDIUM | 9 | Correctness gaps, missing safeguards, metadata inconsistencies |
| LOW | 12 | Stale docs, cosmetic issues, minor gaps |
| INFO | 2 | Non-functional observations |

**Overall verdict**: The core functionality (55 MCP tools, 13 embedders, 4 search strategies, soft-delete, causal gate) is working correctly. No phantom functionality was found in the primary code paths. The issues found are in infrastructure/lifecycle management, not in the data pipeline itself.

---

## CRITICAL Findings

### C1: Unbounded `read_line` in Daemon Proxy

**Files**: `crates/context-graph-mcp/src/main.rs:575,606`
**Impact**: Remote OOM denial-of-service via daemon TCP socket

The daemon stdio-to-TCP proxy uses raw `reader.read_line()` without the bounded wrapper that was specifically created to prevent OOM (AGT-04 fix). The main server loop and TCP handler both correctly use `transport::read_line_bounded()` with `MAX_LINE_BYTES = 10MB`, but the daemon proxy was written afterward and does not use it.

```rust
// main.rs:575 - reading from daemon TCP socket (UNBOUNDED)
match reader.read_line(&mut line).await { ... }

// main.rs:606 - reading from stdin (UNBOUNDED)
match stdin.read_line(&mut line).await { ... }
```

A malicious or malfunctioning daemon could send unbounded data, causing OOM.

**Fix**: Replace both `read_line` calls with `transport::read_line_bounded()`. Two line changes.

---

## HIGH Findings

### H1: HNSW Memory Leak on Vector Deletion

**File**: `crates/context-graph-storage/src/teleological/indexes/hnsw_impl/ops.rs:74-83`
**Impact**: Memory grows unboundedly with updates/deletes across all 15 HNSW indexes

The usearch HNSW library does not support vector deletion. When a vector is "removed," only the UUID mapping is cleared — the vector data and graph connections remain permanently in memory.

```rust
fn remove(&self, id: Uuid) -> IndexResult<bool> {
    // Note: Vector remains in usearch index (doesn't support deletion)
    // but won't be mapped back to UUID
    key_to_id.remove(&key);
}
```

Each of the 15 HNSW indexes accumulates orphaned vectors. For E1 (1024D), each orphan wastes ~4KB. For E7 (1536D), ~6KB. Search quality also degrades as usearch traverses phantom graph nodes.

The `rebuild_indexes_from_store()` path (called on startup when persistence restore fails) would clean this up, but there is no periodic rebuild mechanism.

**Fix**: Add periodic HNSW compaction when `removed_count / total_count > 0.25`, calling `rebuild_indexes_from_store()` to rebuild clean indexes.

---

## MEDIUM Findings

### M1: Fire-and-Forget `tokio::spawn` for GC and HNSW Persistence Tasks

**File**: `crates/context-graph-mcp/src/server/mod.rs:187-216`
**Impact**: If either background task panics, it silently dies — GC stops or HNSW persistence stops

Two critical infinite-loop background tasks have their `JoinHandle` dropped immediately:

```rust
tokio::spawn(async move { loop { /* GC */ } });        // Line 187: JoinHandle DROPPED
tokio::spawn(async move { loop { /* HNSW persist */ } }); // Line 207: JoinHandle DROPPED
```

The codebase convention requires storing JoinHandles (watchers.rs does this correctly). The server's own comment states: "Constitution: JoinHandle must be awaited or aborted — never silently dropped."

**Fix**: Store both JoinHandles in McpServer, add AtomicBool shutdown flags, await them in `shutdown()`.

### M2: HNSW Persistence Not Called on Shutdown

**File**: `crates/context-graph-storage/src/teleological/rocksdb_store/store.rs:741`
**Impact**: Every server restart does a full O(n) HNSW rebuild from CF_FINGERPRINTS

`persist_hnsw_indexes()` is a public method called every 10 minutes by the background task, but there is no evidence of it being called on graceful shutdown. Combined with M1 (the background task could silently die), this means HNSW indexes may never be persisted.

For 100K fingerprints across 15 indexes, rebuild could take 30+ seconds on startup.

**Fix**: Call `persist_hnsw_indexes()` in the server shutdown path.

### M3: Stdio Transport Has No Request Timeout

**File**: `crates/context-graph-mcp/src/server/mod.rs:735-736`
**Impact**: A hung handler blocks the entire MCP session indefinitely

The stdio transport dispatches requests without timeout, unlike the TCP transport which correctly uses `tokio::time::timeout()`. Since stdio is single-threaded (one request at a time), a deadlocked handler stalls the entire Claude Desktop session.

**Fix**: Wrap `self.handle_request()` with `tokio::time::timeout()` matching the TCP transport behavior.

### M4: Unnecessary `unsafe Send/Sync` on LazyMultiArrayProvider

**File**: `crates/context-graph-mcp/src/adapters/lazy_provider.rs:176-177`
**Impact**: Adding a non-Send field in the future would silently create undefined behavior

```rust
unsafe impl Send for LazyMultiArrayProvider {}
unsafe impl Sync for LazyMultiArrayProvider {}
```

All current fields (`Arc<RwLock<...>>`, `Arc<AtomicBool>`) are already `Send + Sync`. The manual unsafe impls are unnecessary and dangerous — they defeat the compiler's auto-trait checking.

**Fix**: Delete both lines. If compilation fails, that correctly identifies a real problem.

### M5: Static Embedding Version Descriptors (False Provenance)

**File**: `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs:429-445`
**Impact**: `get_provenance_chain` returns fake version tracking data

Embedding version provenance stores hardcoded static strings ("pretrained-semantic-1024d", etc.) that never change regardless of actual model versions loaded at runtime. The code acknowledges this:

```rust
// NOTE: These are static descriptors, not dynamic model versions.
// The embedding provider does not currently expose runtime model metadata.
```

An operator querying provenance would see identical versions for all memories, creating false confidence that stale-embedding detection is operational.

**Fix**: Either remove the static descriptors or wire them to actual model metadata from the embedding provider.

### M6: ModelId::Entity Dimension Mismatch (384 vs 768)

**File**: `crates/context-graph-embeddings/src/types/model_id/core.rs:90`
**Impact**: Latent mismatch if deprecated ModelId variant is used

```rust
Self::Entity => 384,   // Legacy MiniLM-L6-v2 (production E11 uses Kepler 768D)
```

`ModelId::Entity.dimension()` returns 384, but the E11 fingerprint slot validates against `E11_DIM = 768`. Production uses `ModelId::Kepler` (768D), so this doesn't currently cause issues, but calling the deprecated `Entity` variant anywhere would produce wrong dimensions.

**Fix**: Update `ModelId::Entity.dimension()` to return 768, or add a `#[deprecated]` attribute with compile warning.

### M7: EmbedderConfig `is_asymmetric` Mismatch for E8/E10

**Files**: `crates/context-graph-core/src/embeddings/config.rs:178` (E8), `:205` (E10)
**Impact**: Config metadata doesn't reflect actual dual-vector capabilities

Both E8 and E10 have `is_asymmetric: false` in their `EmbedderConfig`, despite the `SemanticFingerprint` storing dual vectors (`e8_graph_as_source/target`, `e10_multimodal_as_doc/query`) and having accessor methods like `has_asymmetric_e8()`.

The search code independently handles asymmetry (it doesn't check the config flag), so this is metadata inconsistency rather than a functional bug.

**Fix**: Set `is_asymmetric: true` for E8 and E10 configs.

### M8: 365 Benchmark Tests Gated Behind Never-Default Feature

**File**: `crates/context-graph-benchmark/Cargo.toml:12`
**Impact**: Benchmark metric formulas (NDCG, TPR/TNR, directional accuracy) could silently rot

All 365 benchmark tests were moved behind `#[cfg(all(test, feature = "benchmark-tests"))]`. The feature is not in default features. Normal `cargo test --workspace` runs only 6 benchmark tests.

If a metric formula is broken (e.g., NDCG@k calculation), no test catches it in standard CI.

**Fix**: Either add `benchmark-tests` to CI pipeline, or move critical metric formula tests back to `#[cfg(test)]` and only gate the slow/data-dependent tests.

### M9: Usearch HNSW Soft-Leak Persisted to Disk

**File**: `crates/context-graph-storage/src/teleological/indexes/hnsw_impl/ops.rs:82`
**Impact**: CF_HNSW_GRAPHS stores bloated indexes with orphaned vectors

Extension of H1: When `persist_hnsw_indexes()` serializes the HNSW graph, it includes all orphaned vectors. The persisted graph grows monotonically with deletes/updates, never shrinks. On restart, the bloated graph is loaded back, preserving the leak across server restarts.

**Fix**: Same as H1 — periodic compaction via `rebuild_indexes_from_store()`.

---

## LOW Findings

### L1: `is_ready()` Returns True When Provider Has Failed

**File**: `crates/context-graph-mcp/src/adapters/lazy_provider.rs:155-163`

When model loading fails, `is_ready()` still returns `true` (it only checks `loading`, not `failed`). The actual embed calls correctly check the `failed` flag, so operations fail with clear errors — but `get_memetic_status` health reporting is dishonest.

### L2: `env::set_var` in Multi-Threaded Context

**File**: `crates/context-graph-mcp/src/main.rs:678`

`env::set_var("CONTEXT_GRAPH_MCP_QUIET", "1")` runs inside `#[tokio::main]` which is already multi-threaded. Since Rust 1.66, this is technically UB per POSIX. Rust 2024 edition will make this a hard error.

### L3: 3 Deprecated CFs Opened but Never Used

**File**: `crates/context-graph-storage/src/teleological/column_families.rs:144,239,258`

CF_ENTITY_PROVENANCE, CF_TOOL_CALL_INDEX, CF_CONSOLIDATION_RECOMMENDATIONS are opened (required by RocksDB) but have no read/write paths. Comments say "DEPRECATED: Trait methods not yet wired."

### L4: CF_TOPIC_PROFILES Not Populated by Main Store Path

**File**: `crates/context-graph-storage/src/teleological/rocksdb_store/store.rs:917`

The topic profile is embedded within the serialized TeleologicalFingerprint in CF_FINGERPRINTS. The separate CF_TOPIC_PROFILES may not be populated via the main store path, risking stale/empty reads from profile-only queries.

### L5: Pipeline Stage 2 Double-Deserializes Fingerprints

**File**: `crates/context-graph-storage/src/teleological/rocksdb_store/search.rs:1036-1043`

Fingerprints (~63KB each) are deserialized in Stage 2 for scoring, discarded, then re-read from RocksDB in final result building. For top_k=50, this is 50 extra ~63KB reads.

### L6: 3 Phantom `test_verification_log()` Tests (Zero Assertions)

**Files**:
- `crates/context-graph-core/src/retrieval/pipeline/stage5.rs:472-495`
- `crates/context-graph-core/src/retrieval/sparse_index.rs:948-983`
- `crates/context-graph-storage/tests/e12_e13_source_of_truth_verification.rs:498-526`

These tests contain only `println!` statements with zero assertions. They can never fail and inflate the test count.

### L7: Marblestone Domain/EdgeType Tests Deleted Without Full Replacement

**File**: `crates/context-graph-core/src/marblestone/mod.rs`

Three test files deleted (103 tests, 819 lines). The remaining `edge_case_verification.rs` only covers NeurotransmitterWeights boundary cases — Domain enum (default, description, variant count) and EdgeType tests are no longer covered.

### L8: Stale Documentation Comments (3 instances)

- `validation.rs:34`: Says "E8: 384 dimensions" — actual is 1024D
- `fingerprint.rs:675`: Says "E8, E11: 384D each" — actual E8=1024D, E11=768D
- `weights/mod.rs:296`: Comment says max=6.5 — actual category max=8.5

### L9: Graph Builder Started Without Idempotency Guard

**Files**: `mod.rs:669`, `transport.rs:151`, `transport.rs:418`

`start_graph_builder()` is called from 3 transport paths without checking if already running. In practice, only one transport runs at a time.

### L10: Coarse `#[allow(dead_code)]` on McpServer Struct

**File**: `crates/context-graph-mcp/src/server/mod.rs:95`

Suppresses warnings for the entire struct instead of individual fields, potentially hiding genuinely unused fields.

### L11: Windows Stale Lock Detection Removes Lock of Live Process

**File**: `crates/context-graph-storage/src/teleological/rocksdb_store/store.rs:371-374`

On Windows, `try_remove_lock_file` is called unconditionally without flock probe. On Unix, it correctly tests with `LOCK_EX | LOCK_NB`.

### L12: Stale "17 Column Families" Log Message

**File**: `crates/context-graph-mcp/src/server/mod.rs:168-169`

Log message says "17 column families" but actual count is 51.

---

## INFO Findings

### I1: Code Watcher Never Started

**File**: `crates/context-graph-mcp/src/server/watchers.rs`

The code watcher is gated behind `CODE_PIPELINE_ENABLED=true` and is never called from `main.rs`. Only `start_file_watcher()` is invoked. This appears to be by design (code pipeline not yet ready).

### I2: `trigger_consolidation` Name Slightly Misleading

**File**: `crates/context-graph-mcp/src/handlers/tools/consolidation.rs:418`

The tool finds consolidation candidates but does NOT auto-merge them. Returns `action_required: true` for caller review. Correct behavior per SEC-06 (30-day reversal window), but name implies execution.

---

## Verified Correct (Exonerated)

The following were thoroughly investigated and confirmed working as intended:

| Area | Verification |
|------|-------------|
| **55/55 MCP tools dispatched** | tool_dispatch! macro routes all 55 tools, unknown names return TOOL_NOT_FOUND |
| **LLM fallback** | 3 LLM tools return explicit errors when service unavailable, 52 tools work independently |
| **graph_discovery_service None handling** | All 4 call sites return proper errors |
| **All 51 CFs opened on startup** | `get_all_column_family_descriptors()` chains all groups, `create_missing_column_families(true)` |
| **HNSW excludes E6/E12/E13** | `uses_hnsw()` returns false, `all_hnsw()` returns 15 indexes, panic guards on invalid access |
| **Soft-delete persists and survives restart** | CF_SYSTEM markers loaded via DashMap before index rebuild, filtered from all search paths |
| **Multi-space searches 6 active embedders** | E1, E5, E7, E8, E10, E11 with weighted RRF fusion |
| **Asymmetric E5 direction** | Cross-paired correctly: cause query -> effect index, effect query -> cause index |
| **Causal gate thresholds** | CAUSAL_THRESHOLD=0.04, NON_CAUSAL_THRESHOLD=0.008, BOOST=1.10x, DEMOTION=0.85x verified |
| **All 14 weight profiles** | Sum to ~1.0, temporal (E2-E4) correctly zero-weighted in semantic profiles |
| **with_custom_weights() MCP-03 fix** | Forces MultiSpace strategy when custom weights are used |
| **Pipeline cascade** | E13 SPLADE -> E1+E5+E7+E8+E11 HNSW -> E12 ColBERT reranking verified |
| **All 13 dimension constants** | Internally consistent across 3 definition sites |
| **E5 LoRA training/inference prefix** | `search_document:` prefix matches between train and inference |
| **Content storage** | store_memory stores content alongside fingerprints, search retrieves via get_content_batch() |
| **Audit trail** | All mutations emit AuditRecord entries (non-fatal on audit write failure) |
| **WAL enabled** | RocksDB WAL on by default, WriteBatch for atomic multi-CF writes |
| **Secondary index lock** | Prevents lost-update races on posting lists |
| **Build health** | Zero errors, zero warnings, ~8,080 tests across workspace |

---

## Recommended Fix Priority

| Priority | ID | Fix | Complexity |
|----------|-----|-----|-----------|
| 1 | C1 | Bounded read_line in daemon proxy | 2 line changes |
| 2 | M4 | Remove unsafe Send/Sync | 2 line deletions |
| 3 | M3 | Add stdio request timeout | ~5 lines |
| 4 | M1 | Store GC/persist JoinHandles | ~20 lines |
| 5 | M2 | Call persist_hnsw on shutdown | ~5 lines |
| 6 | H1/M9 | HNSW periodic compaction | ~30 lines |
| 7 | M8 | Restore critical benchmark tests to default | Config change |
| 8 | L6 | Delete phantom test_verification_log | 3 deletions |
| 9 | L12/L8 | Fix stale log message and doc comments | Trivial |
| 10 | M5-M7 | Metadata consistency fixes | ~10 lines each |

---

*Generated by 5 parallel forensic investigation agents examining MCP handlers, storage layer, embeddings/search, tests/build, and server infrastructure.*
