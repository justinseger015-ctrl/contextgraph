# Context Graph - Forensic Investigation Report

**Date**: 2026-02-07
**Branch**: casetrack
**Scope**: Full codebase audit across all crates
**Methodology**: 6 parallel forensic agents investigating dead code, test gaps, storage integrity, MCP correctness, async/concurrency bugs, and type safety

---

## Executive Summary

This investigation uncovered **62 distinct findings** across 6 categories. Of these, **7 are CRITICAL**, **21 are HIGH**, **24 are MEDIUM**, and **10 are LOW**. The most dangerous issues fall into three themes:

1. **Phantom features**: Core systems (memory decay, access counting, consolidation, background discovery) appear fully implemented but are never wired into production paths
2. **Silent data corruption**: Bincode serialization of evolving structs causes silent data loss on schema changes
3. **Concurrency hazards**: ABBA deadlock in the graph builder, fire-and-forget spawns, unkillable threads

The codebase compiles, tests pass, and MCP tools respond -- but significant subsystems produce degenerate results (always-zero access counts, biased retrievals, ignored parameters) that are invisible to callers.

---

## Table of Contents

1. [Critical Findings (7)](#1-critical-findings)
2. [High Severity Findings (21)](#2-high-severity-findings)
3. [Medium Severity Findings (24)](#3-medium-severity-findings)
4. [Low Severity Findings (10)](#4-low-severity-findings)
5. [What Was Found Innocent](#5-what-was-found-innocent)
6. [Remediation Priority Matrix](#6-remediation-priority-matrix)

---

## 1. Critical Findings

### CRIT-01: `record_access()` Never Called in Production
- **Category**: Dead Code / Phantom Feature
- **Files**: `context-graph-core/src/types/memory_node/node.rs:111`, `context-graph-core/src/types/fingerprint/teleological/core.rs:140`
- **Impact**: The entire memory decay and retrieval frequency system is non-functional
- **Details**: `record_access()` increments `access_count` and updates `accessed_at`. It is never called from any MCP handler, storage layer, or search path. Only called in unit tests. This means:
  - `access_count` is always 0 for all memories in production
  - `compute_decay()` always returns `1 + ln(0 + 1) = 1.0` (constant)
  - BM25 importance formula always returns minimum value
  - `should_consolidate()` sees `access_freq = 0.0` for all memories

### CRIT-02: CausalRelationship Bincode + Struct Evolution = Silent Data Loss
- **Category**: Storage / Data Integrity
- **File**: `context-graph-core/src/types/causal_relationship.rs:293-371`, `context-graph-storage/src/teleological/rocksdb_store/causal_relationships.rs:77`
- **Impact**: Old causal relationships are permanently unreadable after schema changes
- **Details**: Bincode is positional/fixed-layout and does NOT encode field names. CausalRelationship has 7 `#[serde(default)]` fields added over time (e5_source_cause, e5_source_effect, e8_graph_source, e8_graph_target, e11_entity, source_spans, llm_provenance). Each addition broke binary compatibility with previously stored data. Evidence:
  - `rebuild_causal_e11_index()` catches deserialization errors with `warn!` + `continue` (silently skips)
  - `repair_corrupted_causal_relationships()` exists specifically to handle this -- it DELETES corrupted entries
  - No migration path from old binary format to new

### CRIT-03: ABBA Deadlock in BackgroundGraphBuilder
- **Category**: Async / Concurrency
- **File**: `context-graph-storage/src/graph_edges/builder.rs:232-352`
- **Impact**: Graph builder can permanently deadlock under concurrent load
- **Details**: Inconsistent lock ordering between two methods:
  - `process_batch()`: acquires `stats` lock, then `pending_queue` lock (line 344-352)
  - `enqueue()`: acquires `pending_queue` lock, then `stats` lock (line 232-248)
  - Classic ABBA deadlock: if both execute concurrently, each holds one lock and waits for the other

### CRIT-04: parking_lot RwLock Guard Held Across .await
- **Category**: Async / Concurrency
- **File**: `context-graph-causal-agent/src/service/mod.rs:504-509`
- **Impact**: !Send violation; undefined behavior if called from spawned task context
- **Details**: In `CausalDiscoveryService::stop()`, a `parking_lot::RwLockReadGuard` (which is `!Send`) is held while calling `tx.send(()).await`. Same class of bug previously fixed in `graph_tools.rs`.

### CRIT-05: Fire-and-Forget tokio::spawn Without Panic Recovery
- **Category**: Async / Concurrency
- **Files**: `context-graph-causal-agent/src/service/mod.rs:466-485`, `context-graph-graph-agent/src/service/mod.rs:430-447`
- **Impact**: Background service silently dies; status reports "Running" forever; `stop()` hangs
- **Details**: Both agent services spawn background loops with `tokio::spawn` but discard the `JoinHandle`. If the task panics:
  - The `running` flag is never set to `false`
  - `ServiceStatus` is never set to `Stopped`
  - `stop()` loops forever waiting for `is_running()` to become false

### CRIT-06: File Watcher Thread Never Joins / No Shutdown Mechanism
- **Category**: Async / Concurrency
- **File**: `context-graph-mcp/src/server.rs:788-838`
- **Impact**: Zombie thread survives server shutdown, holds RocksDB lock
- **Details**: The markdown file watcher spawns an OS thread with an infinite loop. There is no shutdown flag (unlike the code watcher which has `code_watcher_running`), no `stop_file_watcher()` method, and the `JoinHandle` is dropped. The thread creates its own tokio runtime that outlives the main server.

### CRIT-07: CUDA u64-to-u32 Truncation in Kernel Launch
- **Category**: Type Safety / Numeric
- **File**: `context-graph-cuda/src/ffi/knn.rs:510-511`
- **Impact**: Silent wrong results for large GPU workloads (>4B pairs)
- **Details**: `total_pairs` is cast UP to `u64` to avoid overflow, then IMMEDIATELY truncated back to `u32` for block count calculation. If `total_pairs > u32::MAX`, the truncation wraps silently, launching only a fraction of needed GPU threads.

---

## 2. High Severity Findings

### HIGH-01: `boost_importance` Documentation Lies About access_count
- **Category**: Dead Code / Phantom Feature
- **File**: `context-graph-mcp/src/handlers/tools/curation_tools.rs:153-160`
- **Details**: Doc comment says "increases access_count to boost computed importance score." Implementation directly modifies the `importance` float field instead. Line 221 comment admits: "Read actual importance field (not computed from access_count)."

### HIGH-02: Three Ghost Column Families (No Read/Write Methods)
- **Category**: Dead Code / Phantom Feature
- **File**: `context-graph-storage/src/teleological/column_families.rs:160,252,270`
- **Details**: CF_ENTITY_PROVENANCE, CF_TOOL_CALL_INDEX, CF_CONSOLIDATION_RECOMMENDATIONS are fully defined with bloom filters and options, opened in RocksDB, but have ZERO trait methods. Perpetually empty.

### HIGH-03: TransE _predicted_embedding Computed and Discarded
- **Category**: Dead Code / Phantom Feature
- **File**: `context-graph-mcp/src/handlers/tools/entity_tools.rs:988`
- **Details**: `find_related_entities` computes a TransE predicted embedding via `KeplerModel::predict_tail()`. The result is stored in `_predicted_embedding` (underscore = unused). Actual search uses naive text concatenation (`"{entity} {relation} ?"`) via E1 semantic search instead.

### HIGH-04: trigger_consolidation Finds Candidates But Never Merges
- **Category**: Dead Code / Phantom Feature
- **File**: `context-graph-mcp/src/handlers/tools/consolidation.rs:362-435`
- **Details**: Reports "candidates_found, action_required" but no mechanism exists to act on recommendations. ConsolidationRecommendation is never persisted. `merge_concepts` exists separately but has no integration.

### HIGH-05: causal_by_source Read-Modify-Write Race Condition
- **Category**: Storage / Data Integrity
- **File**: `context-graph-storage/src/teleological/rocksdb_store/causal_relationships.rs:158-194`
- **Details**: No atomicity between read (line 167) and write (line 192). Two concurrent stores for the same source_id will silently drop entries from the secondary index. No WriteBatch is used.

### HIGH-06: SourceMetadata Bincode Migration Path Lossy
- **Category**: Storage / Data Integrity
- **Files**: `context-graph-storage/src/teleological/rocksdb_store/source_metadata.rs:29-50`
- **Details**: Tries JSON first, falls back to bincode. But old bincode data with `skip_serializing_if` fields may deserialize into wrong field positions, producing silently corrupt SourceMetadata rather than failing.

### HIGH-07: Restore Operation Not Implemented
- **Category**: Storage / Data Integrity
- **File**: `context-graph-storage/src/teleological/rocksdb_store/persistence.rs:301-318`
- **Details**: `restore_async()` ALWAYS returns error: "In-place restore not supported." Checkpoints can be created but never restored programmatically.

### HIGH-08: `search_graph` customWeights Never Validated
- **Category**: MCP Tool Correctness
- **File**: `context-graph-mcp/src/handlers/tools/memory_tools.rs:747-768`
- **Details**: Parses weights into `[f32; 13]` and passes directly to search engine WITHOUT calling `validate_weights()`. Negative values, sums != 1.0 are silently accepted. Three other tools properly validate.

### HIGH-09: `search_by_intent` blendWithSemantic Parameter Ignored
- **Category**: MCP Tool Correctness
- **File**: `context-graph-mcp/src/handlers/tools/intent_tools.rs:97`
- **Details**: DTO parses and validates `blend_with_semantic`. Handler creates `IntentBoostConfig::default()` and NEVER reads the parsed value. Response hardcodes `blend_weight: 0.0`.

### HIGH-10: `compare_embedder_views` includeContent Never Implemented
- **Category**: MCP Tool Correctness
- **File**: `context-graph-mcp/src/handlers/tools/embedder_tools.rs:439`
- **Details**: Comment says "Content would be fetched if include_content=true" but fetch code was never written. All entries have `content: None`.

### HIGH-11: `get_typed_edges` Direction Parameter Ignored in Query
- **Category**: MCP Tool Correctness
- **File**: `context-graph-mcp/src/handlers/tools/graph_link_tools.rs:337-381`
- **Details**: Query ALWAYS uses `get_typed_edges_from()` (outgoing only). When `direction` is "incoming", outgoing edges are returned but labeled as "incoming" in the response.

### HIGH-12: Infinite Spin-Loop in graph-agent stop()
- **Category**: Async / Concurrency
- **File**: `context-graph-graph-agent/src/service/mod.rs:464-467`
- **Details**: Polls `is_running()` every 100ms with no timeout. If spawned task panics, the flag stays true and `stop()` loops forever.

### HIGH-13: Background spawn Without Panic Handling (Server)
- **Category**: Async / Concurrency
- **Files**: `context-graph-mcp/src/server.rs:324,644,993`
- **Details**: Background model loading task has no panic handler. If `initialize_global_warm_provider()` panics, `models_loading` flag stays true forever. Server enters zombie state where all embedding operations return "still loading."

### HIGH-14: Unsafe Send/Sync on FFI Types Without Verification
- **Category**: Async / Concurrency
- **File**: `context-graph-causal-agent/src/llm/mod.rs:135-136`
- **Details**: `unsafe impl Send for LlmState {}` and `unsafe impl Sync for LlmState {}` on a type containing `LlamaModel` (C FFI). If llama-cpp-2's C library is not thread-safe, this causes UB.

### HIGH-15: TCP Request Timeout Parameter Unused
- **Category**: Async / Concurrency
- **File**: `context-graph-mcp/src/server.rs:1419-1501`
- **Details**: `_request_timeout` parameter (note underscore) is accepted but never applied. Each request can take unbounded time. 32 slow requests exhaust all semaphore permits.

### HIGH-16: FAISS i64-to-usize Negative-to-MAX Cast
- **Category**: Type Safety / Numeric
- **Files**: `context-graph-graph/src/index/gpu_index/index.rs:191-192`, `context-graph-graph/src/index/gpu_index/persistence.rs:129`
- **Details**: `faiss_Index_ntotal()` returns `i64`. Cast to `usize` without checking for -1 (error signal). If FAISS returns -1, this becomes `usize::MAX` (18 quintillion).

### HIGH-17: 44 Poisonable RwLock .unwrap() Calls in Storage Layer
- **Category**: Type Safety / Correctness
- **Files**: `context-graph-storage/src/teleological/rocksdb_store/causal_hnsw_index.rs` (14), `context-graph-storage/src/teleological/indexes/hnsw_impl/ops.rs` (12), `context-graph-storage/src/teleological/search/pipeline/traits.rs` (14), others
- **Details**: Every `read()` and `write()` call uses `.unwrap()`. One panic while holding any lock permanently poisons it, making all subsequent HNSW operations panic.

### HIGH-18: GPU Tests Compiled Out by Feature Gate
- **Category**: Test Coverage
- **File**: `context-graph-mcp/src/handlers/tests/gpu_embedding_verification.rs:20`
- **Details**: 13 GPU embedding tests behind `#[cfg(feature = "cuda")]` - file header explicitly states "MUST NOT be ignored" yet they silently vanish from non-CUDA test runs. These are the ONLY tests using real embeddings.

### HIGH-19: min_similarity Test With Zero Assertions
- **Category**: Test Coverage
- **File**: `context-graph-core/src/retrieval/tests.rs:607-624`
- **Details**: Sets min_similarity to 0.99, executes search, makes ZERO assertions. Prints "[VERIFIED]" without verifying anything. Would pass even if min_similarity were completely broken.

### HIGH-20: All Non-CUDA MCP Tests Use Stub Embeddings
- **Category**: Test Coverage
- **File**: `context-graph-mcp/src/handlers/tests/mod.rs:279-296`
- **Details**: All MCP handler tests use `StubMultiArrayProvider` producing hash-based fake embeddings. Tests verify protocol plumbing but can NEVER catch embedding bugs, wrong dimensions, broken similarity, or CUDA errors.

### HIGH-21: static mut Data Race in Test Code
- **Category**: Type Safety
- **File**: `context-graph-storage/src/teleological/search/single/tests/search.rs:22-26`
- **Details**: `static mut SEED: u32 = 42` accessed in `rand_float()` without synchronization. Tests run in parallel by default = undefined behavior.

---

## 3. Medium Severity Findings

### MED-01: compute_decay() and should_consolidate() Never Invoked
- **File**: `context-graph-core/src/types/memory_node/node.rs:147-168`
- **Details**: Sophisticated memory decay formula exists but is never called from MCP or storage layers.

### MED-02: CausalDiscoveryService Background Loop Never Started
- **File**: `context-graph-causal-agent/src/service/mod.rs:149`
- **Details**: Full background service exists but MCP server never instantiates it. Only on-demand discovery via MCP tool works.

### MED-03: Hardcoded Embedding Version Strings
- **File**: `context-graph-mcp/src/handlers/tools/memory_tools.rs:496-517`
- **Details**: All version strings are hardcoded fiction (e.g., "pretrained-semantic-1024d"), never derived from actual model metadata. Defeats embedding version registry purpose.

### MED-04: MemoryContent.text Always Empty String
- **File**: `context-graph-mcp/src/handlers/tools/consolidation.rs:280`
- **Details**: Constructed with `String::new()` every time. Text-based consolidation analysis operates on nothing.

### MED-05: EntityProvenance Struct Without Storage Wiring
- **File**: `context-graph-core/src/entity/mod.rs:183`
- **Details**: Well-documented struct with doc comment "Stored in CF_ENTITY_PROVENANCE for full audit trail." Never constructed or stored. Entity extraction has zero provenance.

### MED-06: 23+ Audit warn!() Sites With No Failure Monitoring
- **Files**: Multiple MCP handler files
- **Details**: Every audit write follows: `if let Err(e) = append_audit { warn!(...) }`. If audit system goes down, ALL writes silently fail with no health metric, counter, or alert.

### MED-07: Column Family Count Documentation Stale
- **Files**: Multiple (lib.rs, store.rs, mod.rs, column_families.rs)
- **Details**: Comments say 39, 48, or 24 in various places while actual count is 52. Test assertion is correct but its comment says 48.

### MED-08: Deprecated CFs Waste Resources + Orphan Data
- **File**: `context-graph-storage/src/teleological/column_families.rs`
- **Details**: 5 deprecated CFs (CF_ENTITY_PROVENANCE, CF_TOOL_CALL_INDEX, CF_CONSOLIDATION_RECOMMENDATIONS, CF_SESSION_IDENTITY, CF_EGO_NODE) opened on every DB open. Waste memory/FDs. Data in them is orphaned and inaccessible.

### MED-09: FileIndexEntry Bincode Time Bomb
- **Files**: `context-graph-core/src/types/file_index.rs:14-22`, `context-graph-storage/src/teleological/rocksdb_store/file_index.rs`
- **Details**: Uses bincode with no version byte and no `#[serde(default)]`. If struct ever evolves, all existing entries become unreadable. Same problem that already happened with CausalRelationship.

### MED-10: E7/E11 Dimension Comment Inconsistencies
- **File**: `context-graph-storage/src/teleological/column_families.rs`
- **Details**: CF_EMB_6 comment says "256D" but E7_DIM=1536. CF_EMB_10 says "384D" but E11_DIM=768. Runtime uses constants (correct), but comments mislead.

### MED-11: std::sync::RwLock Poisoning Cascade
- **File**: `context-graph-storage/src/teleological/rocksdb_store/store.rs:77,80,710-713`
- **Details**: `soft_deleted` HashMap and `fingerprint_count` use std::sync::RwLock with `.expect()`. One panic permanently breaks the store instance.

### MED-12: search_by_intent weightProfile Schema/Handler Mismatch
- **File**: `context-graph-mcp/src/handlers/tools/intent_dtos.rs:523-527`
- **Details**: JSON schema lists 15 valid profiles. Handler validation whitelist allows only 11. Five valid profiles ("graph_reasoning", "typo_tolerant", "pipeline_*") are rejected.

### MED-13: trigger_consolidation Biased Memory Retrieval
- **File**: `context-graph-mcp/src/handlers/tools/consolidation.rs:233`
- **Details**: Uses `embed_all("context memory patterns")` as fixed query to retrieve candidates. Memories about unrelated topics will never appear as consolidation candidates.

### MED-14: get_session_timeline Uses Biased Semantic Search
- **File**: `context-graph-mcp/src/handlers/tools/sequence_tools.rs:326-350`
- **Details**: Embeds "session timeline" as query instead of scanning by sequence number. Memories semantically dissimilar to "session timeline" may be omitted even with valid sequence numbers.

### MED-15: compare_session_states Same Biased Retrieval
- **File**: `context-graph-mcp/src/handlers/tools/sequence_tools.rs:692-715`
- **Details**: Embeds "session state comparison" as fixed query. Same bias pattern as MED-14.

### MED-16: Global Provider try_read() Fails During Init Window
- **File**: `context-graph-embeddings/src/global_provider.rs:241-246`
- **Details**: During 20-30 second model initialization, all `try_read()` calls return confusing "busy (lock contention)" error instead of "still loading."

### MED-17: Race Between Running Flag and Actual Task Start
- **File**: `context-graph-graph-agent/src/service/mod.rs:419-421`
- **Details**: `running` flag set to `true` BEFORE background task starts executing. If task fails immediately, status remains "Running."

### MED-18: Broadcast Channel Silently Drops Messages
- **File**: `context-graph-mcp/src/transport/sse.rs:580-585`
- **Details**: SSE broadcast channel lag drops N messages silently. For MCP transport, dropped messages mean lost tool results.

### MED-19: Division by Zero in Benchmark Metrics
- **Files**: `context-graph-benchmark/src/metrics/e1_semantic.rs:549`, `context-graph-benchmark/src/metrics/e4_hybrid_session.rs:422`
- **Details**: `members.len() - 1` without empty check. If len==1, divides by 0.0 producing NaN/Inf.

### MED-20: .len() - 1 Underflow on Empty Collections
- **Files**: `context-graph-core/src/clustering/fingerprint_matrix.rs:290`, `context-graph-benchmark/src/bin/temporal_realdata_bench.rs:649`
- **Details**: `for i in 0..(values.len() - 1)` -- if empty, underflows to usize::MAX creating astronomical loop.

### MED-21: .ok() Losing Error Information (5+ sites)
- **Files**: `context-graph-mcp/src/weights.rs:20`, `server.rs:925-926`, `graph_tools.rs:680`, `causal_discovery_tools.rs:201`, `code_tools.rs:281`
- **Details**: Converts Result to Option, losing specific error info. Caller cannot distinguish "doesn't exist" from "is corrupted."

### MED-22: .expect("validated") in MCP Request Handlers
- **Files**: `context-graph-mcp/src/handlers/tools/embedder_tools.rs:1062-1063`, multiple `serde_json::to_value().expect()` calls
- **Details**: Panics MCP server on malformed request if validation has a bug. Defense in depth would use `map_err`.

### MED-23: 17 Chaos Tests Permanently Ignored
- **Files**: `context-graph-graph/tests/chaos_tests/*.rs`
- **Details**: 17 tests for concurrent mutation, resource exhaustion, memory pressure, GPU OOM recovery -- all `#[ignore]`. Only verification of concurrent safety, never run in CI.

### MED-24: Deprecated `with_config` in Production Code
- **File**: `context-graph-graph-agent/src/service/mod.rs:134`
- **Details**: Creates E8Activator WITHOUT GraphModel. In production without test-mode, embedding operations will silently fail.

---

## 4. Low Severity Findings

### LOW-01: max_daily_merges Config Parsed But Never Enforced
- **File**: `context-graph-mcp/src/handlers/tools/consolidation.rs:79-80`

### LOW-02: Unreachable Default Match Arm in Consolidation
- **File**: `context-graph-mcp/src/handlers/tools/consolidation.rs:349`

### LOW-03: GPU Batch Similarity Functions Exported But Never Called
- **File**: `context-graph-cuda/src/similarity.rs:206,284,312`

### LOW-04: search_by_intent parse_strategy/validate Inconsistency
- **File**: `context-graph-mcp/src/handlers/tools/intent_dtos.rs:448 vs 512`

### LOW-05: get_entity_graph Comment Claims TransE But Uses Co-occurrence
- **File**: `context-graph-mcp/src/handlers/tools/entity_tools.rs:1603`

### LOW-06: f64-to-f32 Precision Loss in Weight Parsing
- **File**: `context-graph-mcp/src/weights.rs:58-62`

### LOW-07: Blocking std::fs in Async new()
- **File**: `context-graph-mcp/src/server.rs:1266`

### LOW-08: TOCTOU Race in LazyMultiArrayProvider
- **File**: `context-graph-mcp/src/adapters/lazy_provider.rs:83-104`

### LOW-09: 100+ Compiler Warnings (mostly benchmarks)
- **Files**: Multiple benchmark binaries

### LOW-10: n_points as i32 Truncation in CUDA (>2.1B points)
- **File**: `context-graph-cuda/src/ffi/knn.rs:384-385`

---

## 5. What Was Found Innocent

The investigation confirmed these areas are correctly implemented:

- **Key format consistency**: All schema key functions correctly return raw 16-byte UUIDs. Session temporal keys use proper big-endian encoding.
- **HNSW dimension configuration**: All 15 HNSW indexes have correct dimensionality matching actual vectors.
- **Core search tools**: store_memory, search_recent, search_periodic, search_code, search_by_keywords, search_connections, search_robust all work correctly.
- **Causal E5 asymmetric search**: search_causes, search_effects, get_causal_chain all correctly implement asymmetric embedding.
- **Entity tools**: extract_entities, search_by_entities, infer_relationship, find_related_entities, validate_knowledge work correctly.
- **adaptive_search E1Only bug**: Previously known bug is confirmed FIXED.
- **RocksDB teleological store tests**: Exemplary quality with Full State Verification.
- **Distance computation guards**: Core production paths in distance.rs properly guard against empty inputs.

---

## 6. Remediation Priority Matrix

### Tier 1: Fix Immediately (Data Loss / Deadlock Risk)

| ID | Finding | Effort | Impact |
|----|---------|--------|--------|
| CRIT-02 | CausalRelationship bincode -> JSON migration | Medium | Prevents silent data loss on schema changes |
| CRIT-03 | ABBA deadlock in BackgroundGraphBuilder | Low | Fix lock ordering or restructure to avoid nesting |
| CRIT-05 | Fire-and-forget tokio::spawn | Low | Store JoinHandle, add panic catch |
| CRIT-06 | Unkillable file watcher thread | Low | Add AtomicBool shutdown flag, join on shutdown |
| HIGH-05 | causal_by_source race condition | Low | Use WriteBatch or Mutex for index update |

### Tier 2: Fix Soon (Wrong Results / API Lies)

| ID | Finding | Effort | Impact |
|----|---------|--------|--------|
| CRIT-01 | record_access() never called | Medium | Wire into search/retrieve paths to enable memory decay |
| HIGH-08 | search_graph customWeights not validated | Low | Add validate_weights() call |
| HIGH-09 | blendWithSemantic ignored | Low | Remove param from schema or wire implementation |
| HIGH-11 | get_typed_edges direction ignored | Medium | Add get_typed_edges_to() call for incoming |
| HIGH-03 | TransE _predicted_embedding discarded | Medium | Use computed embedding for HNSW search |
| MED-12 | weightProfile schema/handler mismatch | Low | Sync validation whitelist with schema |

### Tier 3: Fix When Possible (Correctness / Safety)

| ID | Finding | Effort | Impact |
|----|---------|--------|--------|
| CRIT-04 | parking_lot guard across .await | Low | Extract tx clone before await |
| CRIT-07 | CUDA u64->u32 truncation | Low | Add bounds check before cast |
| HIGH-16 | FAISS i64->usize cast | Low | Check for -1 before cast |
| HIGH-17 | 44 poisonable RwLock .unwrap() | Medium | Replace with .unwrap_or_else or error propagation |
| HIGH-15 | TCP timeout param unused | Low | Apply timeout or remove param |
| MED-09 | FileIndexEntry bincode time bomb | Medium | Add version byte or migrate to JSON |

### Tier 4: Cleanup (Documentation / Tests / Dead Code)

| ID | Finding | Effort | Impact |
|----|---------|--------|--------|
| HIGH-01 | boost_importance docs lie | Low | Fix doc comment |
| HIGH-02 | Ghost column families | Low | Document or remove |
| MED-07 | CF count comments stale | Low | Update all comments to 52 |
| MED-10 | Dimension comment inconsistencies | Low | Fix comments |
| HIGH-18 | GPU tests compiled out | Medium | Add CPU-based alternatives |
| HIGH-19 | min_similarity test no assertions | Low | Add assertions |
| MED-23 | 17 chaos tests never run | Low | Add nightly CI job with --ignored |

---

## Systemic Patterns Identified

### Pattern 1: "Biased Retrieval via Fixed Query"
Three tools (trigger_consolidation, get_session_timeline, compare_session_states) need "all memories" but use semantic search with hardcoded query strings. This creates invisible selection bias. **Root cause**: TeleologicalMemoryStore trait lacks `list_all()` / `list_by_session()` / `sample()` methods.

### Pattern 2: "Parameter Accepted But Ignored"
Three tools accept parameters in their JSON schema that have no effect on behavior (blendWithSemantic, includeContent, direction). **Root cause**: API surface evolves faster than implementation; parameter usage is not integration-tested.

### Pattern 3: "Bincode + Struct Evolution = Data Loss"
At least 3 types (CausalRelationship, FileIndexEntry, E12 token embeddings) use bincode for evolving structs. **Root cause**: No project-wide serialization policy enforcing JSON/MessagePack for mutable types.

### Pattern 4: "warn! as Error Swallowing"
23+ audit write sites and 5+ `.ok()` conversions silently discard errors. **Root cause**: Correct individual decision (non-blocking audit) but no aggregate failure monitoring.

---

*Report generated by 6 parallel forensic investigation agents examining all crates under /home/cabdru/contextgraph/crates/*
