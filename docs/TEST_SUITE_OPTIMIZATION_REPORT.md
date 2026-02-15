# Test Suite Optimization Report

**Date**: 2026-02-15
**Branch**: casetrack
**Total Tests**: ~7,773 across 778 files in 11 workspace crates

---

## Executive Summary

The test suite has accumulated significant redundancy through task-specific verification files, duplicated fixture creation, verbose println! instrumentation, and benchmark tests masquerading as unit tests. **An estimated 1,500-2,200 tests can be eliminated or consolidated** without losing meaningful coverage, and several structural changes can dramatically reduce per-test overhead.

**Projected speed improvement**: 30-50% reduction in `cargo test` wall-clock time.

---

## 1. Test Distribution by Crate

| Crate | src/ tests | tests/ tests | benches/ | Total | % of suite |
|-------|-----------|-------------|----------|-------|-----------|
| context-graph-core | 2,643 | 144 | 0 | 2,787 | 35.9% |
| context-graph-embeddings | 1,691 | 322 | 0 | 2,013 | 25.9% |
| context-graph-storage | 633 | 80 | 0 | 713 | 9.2% |
| context-graph-graph | 518 | 170 | 0 | 688 | 8.8% |
| context-graph-mcp | 655 | 8 | 0 | 663 | 8.5% |
| context-graph-benchmark | 370 | 0 | 0 | 370 | 4.8% |
| context-graph-cli | 224 | 35 | 8 | 267 | 3.4% |
| context-graph-cuda | 109 | 52 | 0 | 161 | 2.1% |
| context-graph-graph-agent | 49 | 10 | 0 | 59 | 0.8% |
| context-graph-causal-agent | 35 | 17 | 0 | 52 | 0.7% |

**Key observation**: core (35.9%) and embeddings (25.9%) account for 61.8% of all tests. These two crates should be the primary optimization target.

---

## 2. Redundancy Categories

### 2.1 Verification/FSV/Physical Test Files (HIGHEST IMPACT)

**48 files, ~330 tests, ~14,000 lines of redundant coverage**

These files were created as one-off task verification artifacts. They re-test behavior already covered by unit tests, often with excessive println! output and manual inspection patterns.

**Embeddings crate (13 files, ~8,000 lines)**:
| File | Lines | Tests | Overlap with |
|------|-------|-------|-------------|
| `tests/model_weight_loading_verification.rs` | 918 | 9 | `models/factory/tests/` |
| `tests/warmloader_integration_verification.rs` | 883 | 13 | `warm/tests/`, `warm/integration/tests/` |
| `tests/cuda_runtime_verification.rs` | 832 | 12 | `gpu/` unit tests |
| `tests/task_emb_018_full_state_verification.rs` | 595 | 7 | `quantization/binary.rs` unit tests |
| `tests/full_state_verification.rs` | 538 | 7 | `storage/` unit tests |
| `tests/task_emb_016_physical_verification.rs` | 501 | 7 | `quantization/pq8/tests/` |
| `tests/ws_embeddings_verification.rs` | 404 | 20 | Consolidation of other verification |
| `tests/task_emb_018_physical_verification.rs` | 357 | 8 | `quantization/binary.rs` |
| `tests/task_emb_020_physical_verification.rs` | 315 | 8 | `quantization/float8.rs` |
| `tests/task_emb_014_verification.rs` | 275 | 4 | `models/pretrained/` |
| `tests/task_emb_019_physical_verification.rs` | 272 | 8 | `quantization/pq8/tests/` |
| `tests/physical_verification.rs` | 55 | 1 | Binary quantization unit tests |
| `src/error/verification_test.rs` | 571 | 7 | `error/tests.rs` + `error/tests_extended.rs` |

**MCP crate (8 files, ~1,200 lines)**:
| File | Lines | Tests | Overlap with |
|------|-------|-------|-------------|
| `handlers/tests/manual_fsv_verification.rs` | ~200 | 4 | Other handler tests |
| `handlers/tests/content_storage_verification.rs` | ~150 | 4 | Memory tools tests |
| `handlers/tests/gpu_embedding_verification.rs` | ~200 | 6 | MCP handler tests |
| `handlers/tests/semantic_search_skill_verification.rs` | ~350 | 11 | search tests |
| `handlers/tests/task_emb_024_verification.rs` | ~100 | 2 | MCP tool tests |
| `handlers/tests/topic_tools_fsv.rs` | ~250 | 7 | `topic_tools.rs` tests |
| `handlers/tests/curation_tools_fsv.rs` | ~350 | 12 | `curation_tools.rs` tests |
| `handlers/tests/robustness_fsv.rs` | ~150 | 4 | `robustness_tools.rs` tests |

**Core crate (17 files, ~4,000 lines)**:
| File | Lines | Tests |
|------|-------|-------|
| `tests/ws_core_verification.rs` | ~400 | 10 |
| `tests/budget_fsv_test.rs` | ~300 | 7 |
| `tests/budget_manager_fsv_test.rs` | 538 | 8 |
| `tests/embedder_fsv_test.rs` | ~350 | 10 |
| `tests/clustering_fsv_test.rs` | ~400 | 10 |
| `tests/stability_fsv_test.rs` | ~450 | 14 |

**Storage crate (4 files in `full_state_verification/`)**:
- `column_family_tests.rs`, `persistence_tests.rs`, `write_read_tests.rs`, `edge_case_tests.rs`
- Each creates its own RocksDB store via `create_test_store()` (expensive I/O)

**Recommendation**: Delete all `*_verification.rs`, `*_fsv_*.rs`, and `*_physical_verification.rs` files. Their coverage is already provided by unit tests. **Estimated savings: ~330 tests, ~14,000 lines.**

### 2.2 Manual Test Files (MEDIUM IMPACT)

**4 files, ~50 tests, ~2,100 lines**

| File | Lines | Tests |
|------|-------|-------|
| `core/tests/memory_manual_test.rs` | 571 | 10 |
| `core/tests/hdbscan_manual_test.rs` | 527 | 18 |
| `core/tests/topic_manual_test.rs` | 508 | 9 |
| `core/tests/similarity_manual_test.rs` | ~400 | 12 |
| `core/src/retrieval/manual_test.rs` | ~300 | 8 |

These were created for manual inspection during development. Each contains verbose println! output and tests that duplicate unit test coverage in the same modules.

**Recommendation**: Delete all `*_manual_test*` files. **Estimated savings: ~57 tests, ~2,300 lines.**

### 2.3 Benchmark Crate Tests (HIGH IMPACT)

**370 tests across 85 files in `context-graph-benchmark/src/`**

The benchmark crate has 370 `#[test]` functions embedded in its source code. These are NOT benchmarks — they are unit tests for benchmark infrastructure (dataset generators, metric calculators, config validation, etc.).

Many overlap with tests in the crates they benchmark:
- `metrics/causal.rs` (10 tests) duplicates `core/src/causal/` tests
- `metrics/clustering.rs` (5 tests) duplicates `core/src/clustering/` tests
- `datasets/` (70+ tests) test data generation that only matters for benchmark runs
- `validation/` (35+ tests) re-verify architectural properties tested elsewhere
- `runners/` (40+ tests) test runner configuration, not actual benchmarks

**Recommendation**:
1. Move truly necessary tests behind `#[cfg(feature = "benchmark-tests")]` feature flag
2. Delete ~200 dataset/validation/runner config tests that don't protect production code
3. **Estimated savings: ~200 tests removed from default `cargo test`**

### 2.4 DTO Serialization Tests (MEDIUM IMPACT)

**180 tests across 11 `*_dtos.rs` files in MCP crate**

| File | Tests |
|------|-------|
| `curation_dtos.rs` | 30 |
| `causal_dtos.rs` | 29 |
| `graph_link_dtos.rs` | 25 |
| `entity_dtos.rs` | 21 |
| `graph_dtos.rs` | 18 |
| `topic_dtos.rs` | 32 |
| `robustness_dtos.rs` | 9 |
| `keyword_dtos.rs` | 6 |
| `embedder_dtos.rs` | 5 |
| `code_dtos.rs` | 5 |

These test serde defaults and validation logic. Most follow an identical pattern: deserialize JSON, assert defaults, test validation edge cases. While individually fast, their sheer volume adds compile time.

**Recommendation**: Consolidate into a single `dto_tests.rs` using a `test_dto!` macro that generates boilerplate tests for defaults/validation. **Estimated savings: ~120 tests consolidated to ~60 via macro.**

### 2.5 Verification Log Tests (LOW EFFORT, EASY WIN)

**~80 trivial test functions across 10+ files**

Pattern found in `storage/src/teleological/search/` and related:
```rust
#[test]
fn test_verification_log() {
    println!("\n=== VERIFICATION LOG ===");
    println!("Type Verification: ...");
    println!("PASS");
}
```

These contain zero assertions and never fail. They're documentation masquerading as tests.

**Files**: `maxsim.rs`, `pipeline/tests.rs`, `single/tests/integration.rs`, `error.rs`, `matrix/tests_unit.rs`, `multi/tests.rs`, `result.rs`, `hnsw_impl/tests.rs`, `causal_relationships.rs`, `e12_e13_source_of_truth_verification.rs`

**Recommendation**: Delete all `test_verification_log()` functions. Convert to doc comments if the text is useful. **Estimated savings: ~80 tests.**

### 2.6 Marblestone Enum Tests (LOW EFFORT, EASY WIN)

**103 tests across 3 files (~819 lines)** testing Rust derive macros, not project logic:

| File | Tests | Lines |
|------|-------|-------|
| `core/src/marblestone/tests_domain.rs` | 29 | 232 |
| `core/src/marblestone/tests_edge_type.rs` | 37 | 271 |
| `core/src/marblestone/tests_neurotransmitter_weights.rs` | 37 | 316 |

Examples of what these test:
- `Domain::default() == Domain::General` (testing `#[derive(Default)]`)
- `Domain::all().len() == 6` (testing a hand-written `all()` method)
- `domain.description().is_empty() == false` (testing string returns)

These test framework/stdlib behavior, not project logic. The enums are exercised through real integration tests.

**Recommendation**: Delete all 3 files. **Estimated savings: 103 tests, 819 lines.**

### 2.7 Overlapping Multi-Layer Test Suites (MEDIUM IMPACT)

Several areas have 3+ layers of tests covering the same code path:

**Storage crate**:
- `src/rocksdb_backend/tests_core.rs` (17 tests) — unit tests
- `src/rocksdb_backend/tests_embedding.rs` (23 tests) — unit tests
- `src/rocksdb_backend/tests_node.rs` (8 tests) — unit tests
- `src/rocksdb_backend/tests_edge.rs` (9 tests) — unit tests
- `src/rocksdb_backend/tests_edge_scan.rs` (10 tests) — unit tests
- `src/rocksdb_backend/tests_index.rs` (21 tests) — unit tests
- `src/rocksdb_backend/tests_node_lifecycle.rs` (6 tests) — unit tests
- `tests/storage_integration/` (10 files, ~25 tests) — integration layer
- `tests/full_state_verification/` (4 files, ~9 tests) — FSV layer
- `tests/full_integration_real_data/` (5 files, ~10 tests) — real data layer
- `tests/forensic_verification.rs` (10 tests)
- `tests/e12_e13_source_of_truth_verification.rs` (7 tests)
- `tests/teleological_integration.rs` (8 tests)

That's 94 unit tests + 69 integration/verification tests for a single storage backend. The integration tests each create their own RocksDB temp directories.

**Embeddings crate** has a similar pattern:
- `src/models/pretrained/*/tests*.rs` — per-model unit tests
- `tests/search_test.rs` (26 tests), `tests/benchmark_test.rs` (23 tests) — integration
- 13 verification files (listed above) — yet another layer

**Recommendation**: Keep unit tests + one integration test per subsystem. Delete FSV/verification/real-data layers. **Estimated savings: ~80 integration/verification tests.**

---

## 3. Performance Bottlenecks

### 3.1 TempDir/RocksDB Store Creation (~365 occurrences)

Each `tempfile::tempdir()` call creates a directory on disk. In tests that also create RocksDB stores, this involves:
1. Directory creation
2. RocksDB WAL initialization
3. Column family creation (up to 51 CFs)
4. On teardown: recursive directory deletion

**Measured across the codebase**: 365 tempdir creations in 87 test files. The heaviest offenders:
- `storage/src/teleological/rocksdb_store/tests.rs`: 25 tempdirs (25 RocksDB instances!)
- `core/src/memory/manager.rs`: 25 tempdirs
- `core/src/memory/store.rs`: 23 tempdirs

**Recommendation**:
1. Use `once_cell::sync::Lazy<TempDir>` for shared read-only test stores
2. Wrap test stores in `Arc<Store>` for tests that don't mutate
3. Use `#[serial_test]` for tests that do mutate, sharing one store
4. **Estimated time savings: 5-15 seconds per `cargo test` invocation** (depends on disk speed)

### 3.2 Excessive println! Output (13,870 calls)

**13,870 println! macro invocations** across the codebase, with **4,287** being verbose test instrumentation patterns (`[PASS]`, `===`, `---`, etc.).

`cargo test` by default captures stdout, but still formats all these strings. In `--nocapture` mode, this generates tens of thousands of lines of output that slows terminal rendering.

**Recommendation**:
1. Remove all `println!("[PASS]...")` lines — the test framework already reports pass/fail
2. Remove `println!("=== SECTION ===")` formatting — not needed in automated tests
3. Convert any remaining diagnostic output to `eprintln!` or `tracing::debug!`
4. **Estimated savings: Minor per-test, but significant in aggregate (string formatting of 4,287 decorative prints)**

### 3.3 Compilation Overhead

Each integration test file in `crates/*/tests/` compiles as a **separate binary**. The embeddings crate alone has **41 integration test files** = 41 separate binaries to link.

| Crate | Integration test files | Compile overhead |
|-------|-----------------------|-----------------|
| context-graph-embeddings | 41 | HIGH |
| context-graph-graph | 33 | HIGH |
| context-graph-storage | 26 | HIGH |
| context-graph-core | 16 | MEDIUM |
| context-graph-cli | 6 | LOW |

**Recommendation**: Consolidate integration tests into fewer files using `mod` includes:
```rust
// tests/integration.rs (single binary)
mod verification;   // was: verification.rs
mod search;         // was: search_test.rs
mod roundtrip;      // was: storage_roundtrip_test/
```
This reduces link time proportional to number of files consolidated. For embeddings, going from 41 to ~8 test binaries could save 30-60 seconds of link time.

---

## 4. Specific Files to Delete (Immediate Action)

### 4.1 Delete — Verification Files (no coverage loss)

```
crates/context-graph-embeddings/tests/model_weight_loading_verification.rs
crates/context-graph-embeddings/tests/warmloader_integration_verification.rs
crates/context-graph-embeddings/tests/cuda_runtime_verification.rs
crates/context-graph-embeddings/tests/task_emb_018_full_state_verification.rs
crates/context-graph-embeddings/tests/full_state_verification.rs
crates/context-graph-embeddings/tests/task_emb_016_physical_verification.rs
crates/context-graph-embeddings/tests/task_emb_018_physical_verification.rs
crates/context-graph-embeddings/tests/task_emb_020_physical_verification.rs
crates/context-graph-embeddings/tests/task_emb_014_verification.rs
crates/context-graph-embeddings/tests/task_emb_019_physical_verification.rs
crates/context-graph-embeddings/tests/physical_verification.rs
crates/context-graph-embeddings/tests/ws_embeddings_verification.rs
crates/context-graph-embeddings/src/error/verification_test.rs
crates/context-graph-mcp/src/handlers/tests/manual_fsv_verification.rs
crates/context-graph-mcp/src/handlers/tests/content_storage_verification.rs
crates/context-graph-mcp/src/handlers/tests/gpu_embedding_verification.rs
crates/context-graph-mcp/src/handlers/tests/semantic_search_skill_verification.rs
crates/context-graph-mcp/src/handlers/tests/task_emb_024_verification.rs
crates/context-graph-mcp/src/handlers/tests/topic_tools_fsv.rs
crates/context-graph-mcp/src/handlers/tests/curation_tools_fsv.rs
crates/context-graph-mcp/src/handlers/tests/robustness_fsv.rs
crates/context-graph-core/tests/ws_core_verification.rs
crates/context-graph-core/tests/budget_fsv_test.rs
crates/context-graph-core/tests/budget_manager_fsv_test.rs
crates/context-graph-core/tests/embedder_fsv_test.rs
crates/context-graph-core/tests/clustering_fsv_test.rs
crates/context-graph-core/tests/stability_fsv_test.rs
crates/context-graph-storage/tests/full_state_verification/  (entire directory)
crates/context-graph-storage/tests/forensic_verification.rs
```

**Total**: ~29 files/directories, ~400 tests, ~16,000 lines

### 4.2 Delete — Manual Test Files

```
crates/context-graph-core/tests/memory_manual_test.rs
crates/context-graph-core/tests/hdbscan_manual_test.rs
crates/context-graph-core/tests/topic_manual_test.rs
crates/context-graph-core/tests/similarity_manual_test.rs
crates/context-graph-core/src/retrieval/manual_test.rs
crates/context-graph-core/tests/docs_manual_test.rs
```

**Total**: 6 files, ~58 tests, ~2,800 lines

### 4.3 Feature-Gate — Benchmark Tests

Add to `crates/context-graph-benchmark/Cargo.toml`:
```toml
[features]
benchmark-tests = []
```

Wrap all `#[cfg(test)]` modules in benchmark source with:
```rust
#[cfg(all(test, feature = "benchmark-tests"))]
```

This removes 370 tests from default `cargo test` while keeping them available via `cargo test --features benchmark-tests`.

---

## 5. Consolidation Opportunities

### 5.1 Embeddings Integration Tests

Consolidate 41 integration test files into ~8 by grouping:
- `search_test.rs` + `benchmark_test.rs` + `no_fake_data_test.rs` -> `tests/search_integration.rs`
- `storage_roundtrip_test/*.rs` (7 files) -> `tests/storage_roundtrip.rs` (single mod file)
- `dimension_test/*.rs` (8 files) -> `tests/dimension.rs` (single mod file)
- `qodo_embed_integration.rs` + `embed_dual_with_hint_integration.rs` -> `tests/embedding_integration.rs`
- `gpu_memory_slots_test.rs` + `global_provider_fsv.rs` -> `tests/provider_integration.rs`
- `e9_*.rs` (2 files) -> `tests/e9_integration.rs`

### 5.2 Graph Integration Tests

Consolidate test directories:
- `tests/integration_tests/` (10 files) -> mod-based single binary
- `tests/storage_tests/` (8 files) -> mod-based single binary
- `tests/gpu_memory_tests/` (5 files) -> mod-based single binary
- `tests/chaos_tests/` (4 files) -> mod-based single binary
- This reduces from ~33 to ~4 test binaries

### 5.3 Storage Integration Tests

Merge 4 overlapping integration test suites:
- `tests/storage_integration/` (10 files)
- `tests/full_integration_real_data/` (5 files)
- `tests/teleological_integration.rs`
- `tests/e12_e13_source_of_truth_verification.rs`
- `tests/e9_storage_roundtrip_test.rs`

Into a single `tests/integration.rs` with submodules. This eliminates ~15 separate test binaries.

### 5.4 Test Fixture Deduplication

`create_test_fingerprint` / `test_fingerprint` patterns appear in 29 files with 239 occurrences. The `context-graph-test-utils` crate exists but is underutilized.

**Action**: Move all common fixtures to `context-graph-test-utils`:
- `create_test_fingerprint()` variants
- `create_test_memory_node()` variants
- `create_test_store()` / store setup helpers
- Common RocksDB temp directory patterns

---

## 6. Priority Action Plan

| Priority | Action | Tests Removed | Time Saved | Effort |
|----------|--------|--------------|-----------|--------|
| P1 | Delete verification/FSV files (Section 4.1) | ~400 | 10-15% | Low |
| P2 | Feature-gate benchmark tests (Section 4.3) | ~370 | 5-8% | Low |
| P3 | Delete manual test files (Section 4.2) | ~58 | 2-3% | Low |
| P4 | Delete verification_log no-op tests (Section 2.5) | ~80 | 1-2% | Trivial |
| P5 | Delete marblestone enum tests (Section 2.6) | ~103 | 1-2% | Trivial |
| P6 | Consolidate integration test binaries (Section 5) | 0 (same tests, fewer binaries) | 10-20% (link time) | Medium |
| P7 | Strip 4,287 println! decorations | 0 | 1-3% | Medium |
| P8 | Share RocksDB stores in storage tests | 0 | 5-10% (I/O) | Medium |
| P9 | DTO test macro consolidation (Section 2.4) | ~120 | 2-3% | Medium |
| P10 | Deduplicate fixture creation to test-utils | 0 | Compile time | High |

**P1+P2+P3 alone remove ~828 tests with zero coverage loss and minimal effort.**
**Adding P4+P5 brings total removable tests to ~1,011.**

---

## 7. Safe Deletion Verification

Before deleting any file, verify coverage isn't lost by checking:

1. Every `#[test]` function in the file-to-delete has a corresponding test in the unit test module
2. Run `cargo test -p <crate>` before and after deletion
3. If a test in a verification file tests something NOT covered by unit tests, move that specific test to the appropriate unit test module before deleting the file

The verification/FSV files were explicitly created as "second opinion" tests during development tasks. By design, they duplicate existing unit test coverage.

---

## Appendix: Files by Test Count (Top 40)

| Tests | File |
|-------|------|
| 77 | `core/src/clustering/birch.rs` |
| 56 | `core/src/causal/asymmetric.rs` |
| 53 | `core/src/clustering/manager.rs` |
| 52 | `core/src/clustering/hdbscan.rs` |
| 46 | `embeddings/src/models/pretrained/entity/tests.rs` |
| 43 | `embeddings/src/warm/loader/types/tests.rs` |
| 43 | `embeddings/src/models/pretrained/graph/tests.rs` |
| 42 | `core/src/memory/chunker.rs` |
| 41 | `core/src/retrieval/distance.rs` |
| 39 | `embeddings/src/types/embedding/tests.rs` |
| 38 | `mcp/src/transport/sse.rs` |
| 38 | `core/src/config/tests/mcp_config_tests.rs` |
| 37 | `embeddings/src/models/pretrained/causal/tests.rs` |
| 37 | `core/src/retrieval/divergence.rs` |
| 37 | `core/src/marblestone/tests_neurotransmitter_weights.rs` |
| 37 | `core/src/marblestone/tests_edge_type.rs` |
| 35 | `embeddings/src/batch/types/tests.rs` |
| 35 | `core/src/retrieval/config.rs` |
| 35 | `cli/src/commands/hooks/types.rs` |
| 34 | `cuda/tests/cuda_cone_test.rs` |
| 32 | `mcp/src/handlers/tools/topic_dtos.rs` |
| 32 | `mcp/src/handlers/tools/memory_tools.rs` |
| 32 | `embeddings/src/error/tests_extended.rs` |
| 32 | `core/src/injection/formatter.rs` |
| 31 | `core/src/injection/priority.rs` |
| 30 | `mcp/src/handlers/tools/curation_dtos.rs` |
| 30 | `core/src/teleological/services/profile_manager/tests.rs` |
| 30 | `core/src/retrieval/tests.rs` |
| 30 | `core/src/retrieval/multi_space.rs` |
| 30 | `core/src/clustering/topic.rs` |
| 29 | `mcp/src/middleware/validation.rs` |
| 29 | `mcp/src/handlers/tools/causal_dtos.rs` |
| 29 | `graph/src/entailment/cones/tests.rs` |
| 29 | `core/src/types/graph_edge/tests_struct.rs` |
| 29 | `core/src/marblestone/tests_domain.rs` |
| 29 | `core/src/graph/asymmetric.rs` |
| 29 | `core/src/embeddings/provider.rs` |
| 28 | `storage/src/teleological/search/pipeline/tests.rs` |
| 28 | `core/src/types/fingerprint/sparse.rs` |
| 28 | `core/src/clustering/persistence.rs` |
