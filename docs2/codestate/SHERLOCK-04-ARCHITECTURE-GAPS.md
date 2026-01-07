# SHERLOCK HOLMES FORENSIC REPORT: ARCHITECTURE COMPLIANCE GAPS

**Case ID:** SHERLOCK-04-ARCH-GAPS-20260106
**Investigator:** Sherlock Holmes Agent #4
**Date:** 2026-01-06
**Subject:** Architecture Compliance Analysis vs Constitution v4.0.0 and PRD

---

## EXECUTIVE SUMMARY

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*

After exhaustive examination of the codebase against the specified Constitution v4.0.0 architecture, I have identified **significant structural deviations** across multiple compliance dimensions. The codebase shows evidence of organic evolution rather than strict adherence to constitutional specifications.

**OVERALL COMPLIANCE VERDICT:** PARTIAL (Approximately 55-60% compliant)

| Category | Compliance Level | Risk |
|----------|------------------|------|
| Directory Structure | PARTIAL | MEDIUM |
| 5-Layer Bio-Nervous System | PARTIAL (Stubs only) | HIGH |
| Crate Dependencies | PARTIAL | MEDIUM |
| Type Definitions | COMPLIANT | LOW |
| MCP Protocol | COMPLIANT | LOW |
| Performance Budgets | UNKNOWN (No benchmarks) | HIGH |
| Naming Conventions | COMPLIANT | LOW |
| Anti-Pattern Violations | VIOLATED | HIGH |
| Module Size Limits | VIOLATED (22 files exceed) | MEDIUM |
| Integration Points | PARTIAL | MEDIUM |

---

## 1. DIRECTORY STRUCTURE COMPLIANCE

### Constitution Requirement (Specified)
```
crates/:
  context-graph-mcp/: "MCP server (tools/, resources/, handlers/, adapters/)"
  context-graph-core/: "Domain logic (graph/, search/, utl/, session/, curation/, teleological/)"
  context-graph-cuda/: "GPU (kernels/, hnsw/, hopfield/, neuromod/)"
  context-graph-embeddings/: "13-model pipeline (models/, fingerprint/, purpose_vector.rs, semantic_fingerprint.rs)"
  context-graph-storage/: "Teleological storage (rocksdb/, scylla/, indexes/, temporal/)"
```

### Actual Implementation (Observed)
```
crates/:
  context-graph-mcp/src/: handlers/, adapters/, middleware/
    MISSING: tools/, resources/ directories

  context-graph-core/src/: alignment/, config/, index/, johari/, marblestone/,
                           monitoring/, purpose/, retrieval/, similarity/, stubs/, traits/, types/
    MISSING: graph/, search/, utl/, session/, curation/
    NOTE: "teleological/" is in fingerprint/teleological/ subdirectory

  context-graph-cuda/src/: cone/, poincare/, error/, ops/, stub/
    MISSING: hnsw/, hopfield/, neuromod/
    EXISTS: kernels/ (poincare_distance.cu, cone_check.cu)

  context-graph-embeddings/src/: batch/, cache/, config/, error/, gpu/, models/,
                                  provider/, storage/, traits/, types/, warm/, quantization/
    MISSING: fingerprint/, purpose_vector.rs, semantic_fingerprint.rs at top level

  context-graph-storage/src/: rocksdb_backend/, teleological/, serialization/,
                               column_families.rs, indexes.rs, memex.rs
    MISSING: scylla/ directory (ScyllaDB not implemented)
    MISSING: temporal/ as separate directory
```

### Unexpected Crates (Not in Constitution)
- `context-graph-graph/` - Separate crate for graph operations (should be in core)
- `context-graph-utl/` - Separate crate for UTL (should be in core)

### Compliance Level: **PARTIAL**
| Item | Status | Notes |
|------|--------|-------|
| context-graph-mcp structure | PARTIAL | Missing tools/, resources/ |
| context-graph-core structure | PARTIAL | Different organization, missing modules |
| context-graph-cuda structure | PARTIAL | Missing hnsw/, hopfield/, neuromod/ |
| context-graph-embeddings structure | PARTIAL | Different organization |
| context-graph-storage structure | PARTIAL | Missing scylla/, temporal/ |

### Remediation Required
1. Consolidate `context-graph-graph/` into `context-graph-core/src/graph/`
2. Consolidate `context-graph-utl/` into `context-graph-core/src/utl/`
3. Create missing CUDA subdirectories or document deviation
4. Implement ScyllaDB adapter or update constitution

---

## 2. 5-LAYER BIO-NERVOUS SYSTEM

### Constitution/PRD Requirement
```
L1_Sensing: <5ms, 13-model embed, PII scrub
L2_Reflex: <100us, Hopfield cache, >80% hit rate
L3_Memory: <1ms, MHN + FAISS GPU, 2^768 capacity
L4_Learning: 100Hz, UTL optimizer, neuromod controller
L5_Coherence: 10ms sync, Thalamic gate, GW broadcast
```

### Actual Implementation

**EVIDENCE FOUND:**
- `/home/cabdru/contextgraph/crates/context-graph-core/src/stubs/layers/` contains:
  - `sensing.rs` (StubSensingLayer)
  - `reflex.rs` (StubReflexLayer)
  - `memory.rs` (StubMemoryLayer)
  - `learning.rs` (StubLearningLayer)
  - `coherence.rs` (StubCoherenceLayer)

**CRITICAL OBSERVATION:**
From `mod.rs` line 6-10:
```rust
//! These implementations provide deterministic, instant responses for the
//! Ghost System phase (Phase 0). Production implementations will replace
//! these with real processing logic.
```

**Layer Component Analysis:**

| Layer | Component Required | Implementation Status |
|-------|-------------------|----------------------|
| L1_Sensing | 13-model embed | EXISTS (in embeddings crate) |
| L1_Sensing | PII scrub | NOT FOUND |
| L2_Reflex | Hopfield cache | STUB ONLY (referenced but not implemented) |
| L2_Reflex | >80% hit rate monitoring | NOT FOUND |
| L3_Memory | MHN (Modern Hopfield Network) | NOT FOUND |
| L3_Memory | FAISS GPU | EXISTS (in context-graph-graph crate) |
| L3_Memory | 2^768 capacity | NOT MEASURABLE |
| L4_Learning | UTL optimizer | EXISTS (context-graph-utl crate) |
| L4_Learning | Neuromod controller | NOT FOUND as separate component |
| L5_Coherence | Thalamic gate | NOT FOUND |
| L5_Coherence | GW broadcast | NOT FOUND |

### Compliance Level: **PARTIAL (Stubs Only)**

**VERDICT:** The 5-layer architecture exists as a **stub framework** only. Production implementations are not present. The `NervousLayer` trait exists at `/home/cabdru/contextgraph/crates/context-graph-core/src/traits/nervous_layer.rs` but all implementations are stubs.

### Remediation Required
1. Implement real L1_Sensing with PII scrubbing
2. Implement Hopfield network for L2_Reflex cache
3. Implement Modern Hopfield Network for L3_Memory
4. Implement Thalamic gate and GW broadcast for L5_Coherence
5. Add latency monitoring to verify budget compliance

---

## 3. CRATE DEPENDENCIES

### Constitution Requirement
```
Dependencies:
- tokio@1.35+, serde@1.0+, uuid@1.6+, chrono@0.4+
- rmcp@0.1+, cudarc@0.10+, faiss@0.12+gpu
- rocksdb@0.21+, scylladb@1.0+
```

### Actual Implementation (from Cargo.toml files)

| Dependency | Required | Actual | Status |
|------------|----------|--------|--------|
| tokio | @1.35+ | @1.35 | COMPLIANT |
| serde | @1.0+ | @1.0 | COMPLIANT |
| uuid | @1.6+ | @1.6 | COMPLIANT |
| chrono | @0.4+ | @0.4 | COMPLIANT |
| rmcp | @0.1+ | NOT FOUND | MISSING |
| cudarc | @0.10+ | NOT FOUND | MISSING |
| faiss | @0.12+gpu | NOT FOUND (FFI bindings exist) | PARTIAL |
| rocksdb | @0.21+ | @0.22 | COMPLIANT |
| scylladb | @1.0+ | NOT FOUND | MISSING |

**OBSERVED ALTERNATIVES:**
- Instead of `rmcp`, using custom JSON-RPC implementation
- Instead of `cudarc`, using `candle-core` with CUDA features
- FAISS integration via custom FFI bindings in `context-graph-graph/src/index/faiss_ffi/`

### Compliance Level: **PARTIAL**

**Notable Observations:**
- GPU acceleration uses Candle framework (HuggingFace) instead of cudarc
- MCP implementation is custom, not using rmcp crate
- ScyllaDB is completely absent (only RocksDB implemented)

### Remediation Required
1. Either add rmcp dependency or update constitution to reflect custom implementation
2. Document decision to use Candle instead of cudarc
3. Implement ScyllaDB adapter or remove from constitution

---

## 4. TYPE DEFINITIONS

### Constitution/PRD Requirement
- KnowledgeNode with TeleologicalFingerprint
- GraphEdge with NT weights
- SessionState with neuromodulation levels

### Actual Implementation

**TeleologicalFingerprint** (FOUND at `/home/cabdru/contextgraph/crates/context-graph-core/src/types/fingerprint/teleological/types.rs`):
```rust
pub struct TeleologicalFingerprint {
    pub id: Uuid,
    pub semantic: SemanticFingerprint,  // 13-embedding array
    pub purpose_vector: PurposeVector,   // 13D alignment
    pub johari: JohariFingerprint,       // Per-embedder awareness
    pub purpose_evolution: Vec<PurposeSnapshot>,
    pub theta_to_north_star: f32,
    pub content_hash: [u8; 32],
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub access_count: u64,
}
```
**STATUS:** COMPLIANT

**GraphEdge with NT weights** (FOUND at `/home/cabdru/contextgraph/crates/context-graph-core/src/types/graph_edge/edge.rs`):
```rust
pub struct GraphEdge {
    pub id: EdgeId,
    pub source_id: NodeId,
    pub target_id: NodeId,
    pub edge_type: EdgeType,
    pub weight: f32,
    pub confidence: f32,
    pub domain: Domain,
    pub neurotransmitter_weights: NeurotransmitterWeights,  // NT weights present!
    pub is_amortized_shortcut: bool,
    pub steering_reward: f32,
    pub traversal_count: u64,
    pub created_at: DateTime<Utc>,
    pub last_traversed_at: Option<DateTime<Utc>>,
}
```
**STATUS:** COMPLIANT

**SessionState with neuromodulation levels** (FOUND references in config):
- Neuromodulation mentioned in config subsystems
- No dedicated SessionState struct with neuromodulation levels found
**STATUS:** PARTIAL

### Compliance Level: **COMPLIANT (Core Types)**

---

## 5. MCP PROTOCOL COMPLIANCE

### Constitution Requirement
```
- Version: "2024-11-05"
- Transport: stdio, sse
- Caps: tools, resources, prompts, logging
- Error codes: -32700 through -32004
```

### Actual Implementation

**Protocol Version** (FOUND in `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/lifecycle.rs` line 35):
```rust
"protocolVersion": "2024-11-05",
```
**STATUS:** COMPLIANT

**Transport:**
- stdio: IMPLEMENTED (in `/home/cabdru/contextgraph/crates/context-graph-mcp/src/server.rs`)
- SSE: NOT IMPLEMENTED (only stdio transport)
**STATUS:** PARTIAL

**Capabilities** (from lifecycle.rs):
```rust
"capabilities": {
    "tools": { "listChanged": true }
}
```
- tools: PRESENT
- resources: MISSING
- prompts: MISSING
- logging: MISSING
**STATUS:** PARTIAL

**Error Codes** (FOUND in `/home/cabdru/contextgraph/crates/context-graph-mcp/src/protocol.rs`):
```rust
pub const PARSE_ERROR: i32 = -32700;
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;
pub const INTERNAL_ERROR: i32 = -32603;
// Context Graph specific codes through -32050+
```
**STATUS:** COMPLIANT (exceeds requirements with custom codes)

### Compliance Level: **PARTIAL**

### Remediation Required
1. Implement SSE transport
2. Add resources capability
3. Add prompts capability
4. Add logging capability

---

## 6. PERFORMANCE BUDGET VIOLATIONS

### Constitution Requirement
```
- inject_context: p95 <25ms, p99 <50ms
- hopfield: <1ms
- reflex_cache: <100us
- single_embed: <10ms
- batch_embed_64: <50ms
- faiss_1M_k100: <2ms
```

### Actual Implementation

**CRITICAL FINDING:** No performance benchmarks or monitoring found that measure these specific budgets.

**Evidence:**
1. No `#[bench]` attributes found measuring these operations
2. No Criterion benchmarks targeting these specific metrics
3. Stub implementations return "instant" but real implementations not measured

**FAISS benchmarks found but not aligned to spec:**
- `/home/cabdru/contextgraph/crates/context-graph-graph/benches/benchmark_suite/main.rs`
- Does not measure "faiss_1M_k100" scenario specifically

### Compliance Level: **UNKNOWN (Unverifiable)**

### Remediation Required
1. Add Criterion benchmarks for each performance budget item
2. Add runtime latency monitoring with p95/p99 metrics
3. Add CI gates that fail if budgets exceeded

---

## 7. NAMING CONVENTION VIOLATIONS

### Constitution Requirement
```
- files: snake_case.rs
- types: PascalCase
- funcs: snake_case_verb_first
- const: SCREAMING_SNAKE
```

### Actual Implementation

**File names:** All `.rs` files observed use snake_case - COMPLIANT

**Type names:** All structs/enums use PascalCase - COMPLIANT
- `TeleologicalFingerprint`, `GraphEdge`, `SemanticFingerprint`, etc.

**Function names:** snake_case with verb-first observed - COMPLIANT
- `embed_all()`, `handle_initialize()`, `serialize_node()`, etc.

**Constants:** SCREAMING_SNAKE observed - COMPLIANT
- `PARSE_ERROR`, `E1_DIM`, `NUM_EMBEDDERS`, etc.

### Compliance Level: **COMPLIANT**

---

## 8. ANTI-PATTERN VIOLATIONS

### Constitution Requirement
```
Forbidden:
- AP-001: unwrap() in prod
- AP-002: Hardcoded secrets
- AP-003: Magic numbers
- AP-004: Blocking I/O in async
- AP-005: FAISS mutation without lock
- AP-007: Stub data in prod
- AP-015: GPU alloc without pool
```

### Actual Implementation

**AP-001: unwrap() in prod**
**EVIDENCE:** Grep found **2,663 occurrences** of `.unwrap()` across 252 files.

Most concerning files:
- `hnsw_purpose.rs`: 98 occurrences
- `tests_index.rs`: 67 occurrences
- `teleological_memory_store_tests.rs`: 48 occurrences

**VERDICT:** VIOLATED (many in non-test code)

**AP-002: Hardcoded secrets**
- Grep for "password", "secret", "api_key", "credentials" found NO matches
**VERDICT:** COMPLIANT

**AP-003: Magic numbers**
- Grep found 569 occurrences of 2+ digit numbers in core source
- Many are dimension constants (1024, 384, etc.) which are documented
**VERDICT:** PARTIAL (needs audit)

**AP-004: Blocking I/O in async**
- Found `std::fs::` usage in 21 files
- Most in non-async contexts or tests
- Potential issues in:
  - `/home/cabdru/contextgraph/crates/context-graph-mcp/src/server.rs` (line 9: `std::io::`)
**VERDICT:** NEEDS AUDIT

**AP-005: FAISS mutation without lock**
- FAISS operations in `context-graph-graph/src/index/gpu_index/`
- Lock usage needs verification
**VERDICT:** NEEDS AUDIT

**AP-007: Stub data in prod**
- Grep found **605 occurrences** of stub/mock/fake/todo!/unimplemented!
- Stubs ARE used but with FAIL FAST pattern (seen in server.rs)
- `LazyFailMultiArrayProvider` returns errors, not fake data
**VERDICT:** PARTIAL (fail-fast pattern mitigates)

**AP-015: GPU alloc without pool**
- GPU memory pool found in `/home/cabdru/contextgraph/crates/context-graph-embeddings/src/warm/memory_pool/`
**VERDICT:** COMPLIANT

### Compliance Level: **VIOLATED**

### Remediation Required
1. **CRITICAL:** Audit and replace ~2,663 `.unwrap()` calls with proper error handling
2. Audit magic numbers and define as named constants
3. Audit blocking I/O in async contexts
4. Verify FAISS mutation locking

---

## 9. MODULE SIZE VIOLATIONS

### Constitution Requirement
```
"One primary type per module, max 500 lines (excl tests)"
```

### Actual Implementation

**FILES EXCEEDING 500 LINES (Source only, excluding test files):**

| File | Lines | Over By |
|------|-------|---------|
| warm/loader/types.rs | 1,904 | 1,404 |
| index/hnsw_impl.rs | 1,898 | 1,398 |
| alignment/calculator.rs | 1,553 | 1,053 |
| index/purpose/hnsw_purpose.rs | 1,422 | 922 |
| johari/default_manager.rs | 1,390 | 890 |
| index/purpose/clustering.rs | 1,294 | 794 |
| handlers/purpose.rs | 1,281 | 781 |
| teleological/rocksdb_store.rs | 1,267 | 767 |
| pretrained/sparse/projection.rs | 1,266 | 766 |
| stubs/utl_stub.rs | 1,202 | 702 |
| index/purpose/query.rs | 1,189 | 689 |
| handlers/utl.rs | 1,100 | 600 |
| handlers/johari.rs | 1,075 | 575 |
| stubs/teleological_store_stub.rs | 927 | 427 |
| storage/types.rs | 903 | 403 |
| teleological/quantized.rs | 873 | 373 |
| retrieval/pipeline.rs | 856 | 356 |
| cache/mod.rs | 840 | 340 |
| purpose/default_computer.rs | 839 | 339 |
| handlers/search.rs | 808 | 308 |
| traits/multi_array_embedding.rs | 807 | 307 |
| quantization/router.rs | 790 | 290 |

**Total files violating limit:** 22 source files

### Compliance Level: **VIOLATED**

### Remediation Required
1. Refactor large modules into smaller focused modules
2. Priority: hnsw_impl.rs (1,898 lines) and warm/loader/types.rs (1,904 lines)

---

## 10. MISSING INTEGRATION POINTS

### Constitution/PRD Requirement
- Layers should communicate via defined interfaces
- Proper service layer between MCP and core
- Adapters for storage backends

### Actual Implementation

**Layer Communication:**
- `NervousLayer` trait exists but all implementations are stubs
- No inter-layer communication protocol observed
- No message bus or event system
**STATUS:** PARTIAL

**Service Layer between MCP and Core:**
- `Handlers` struct in MCP creates integration
- Direct dependency on core traits: `TeleologicalMemoryStore`, `MultiArrayEmbeddingProvider`, `UtlProcessor`
- Adapter pattern used for UTL: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/adapters/utl_adapter.rs`
**STATUS:** COMPLIANT

**Storage Backend Adapters:**
- RocksDB: IMPLEMENTED (`RocksDbTeleologicalStore`)
- ScyllaDB: NOT IMPLEMENTED
- FAISS GPU: PARTIAL (FFI bindings exist)
**STATUS:** PARTIAL

### Compliance Level: **PARTIAL**

### Remediation Required
1. Define inter-layer communication protocol
2. Implement remaining storage adapters (ScyllaDB) or update constitution

---

## EVIDENCE CHAIN OF CUSTODY

| Timestamp | Evidence Item | Location | Verification Method |
|-----------|--------------|----------|---------------------|
| 2026-01-06T19:00 | Cargo.toml files | /home/cabdru/contextgraph/crates/*/Cargo.toml | Read tool |
| 2026-01-06T19:02 | Directory structures | /home/cabdru/contextgraph/crates/*/src/ | Bash find |
| 2026-01-06T19:05 | Type definitions | /home/cabdru/contextgraph/crates/context-graph-core/src/types/ | Read tool |
| 2026-01-06T19:08 | MCP protocol | /home/cabdru/contextgraph/crates/context-graph-mcp/src/protocol.rs | Read tool |
| 2026-01-06T19:10 | Stub implementations | /home/cabdru/contextgraph/crates/context-graph-core/src/stubs/layers/ | Read/Grep |
| 2026-01-06T19:12 | unwrap() occurrences | All *.rs files | Grep count |
| 2026-01-06T19:15 | Module line counts | All src/*.rs files | Bash wc -l |

---

## SUMMARY OF FINDINGS

### Critical Issues Requiring Immediate Attention

1. **AP-001 Violation:** 2,663 `.unwrap()` calls must be audited and replaced
2. **5-Layer System:** Production implementations do not exist (stubs only)
3. **Module Size:** 22 files exceed 500-line limit
4. **ScyllaDB:** Required by constitution but not implemented

### Moderate Issues

1. **Directory Structure:** Deviation from specified layout
2. **MCP Capabilities:** Missing resources, prompts, logging
3. **Performance Budgets:** No measurement system in place
4. **SSE Transport:** Not implemented

### Architectural Decisions Requiring Documentation

1. **Candle vs cudarc:** GPU framework choice differs from spec
2. **Custom MCP:** Not using rmcp crate
3. **Separate crates:** context-graph-graph and context-graph-utl vs consolidated core
4. **Stub pattern:** FAIL FAST approach acceptable for Phase 0

---

## RECOMMENDATIONS

### Priority 1 (Immediate - Security/Reliability)
- [ ] Audit and fix unwrap() usage in production code paths
- [ ] Document or implement missing ScyllaDB adapter

### Priority 2 (Short-term - Compliance)
- [ ] Refactor oversized modules
- [ ] Add MCP capabilities for resources, prompts, logging
- [ ] Add SSE transport

### Priority 3 (Medium-term - Architecture)
- [ ] Implement real 5-layer processing (remove stubs)
- [ ] Add performance benchmarks matching constitution budgets
- [ ] Consolidate scattered graph/utl functionality

### Priority 4 (Long-term - Documentation)
- [ ] Document all architectural deviations with rationale
- [ ] Update constitution to reflect current state OR migrate code to match

---

*"The world is full of obvious things which nobody by any chance ever observes."*

**Case Status:** OPEN - Multiple remediation items required

**Signed:** Sherlock Holmes, Forensic Code Investigator
**Date:** 2026-01-06
