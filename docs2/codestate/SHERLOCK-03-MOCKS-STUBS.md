# SHERLOCK HOLMES FORENSIC REPORT - CASE #003

## MOCKS, STUBS, AND FALLBACKS INVESTIGATION

**Case ID**: SHERLOCK-03-MOCKS-STUBS
**Date**: 2026-01-06
**Subject**: Identification of all MOCKS, STUBS, FALLBACKS, and WORKAROUNDS that mask the system not working as the PRD expects

---

## CASE SUMMARY

```
HOLMES: *adjusts magnifying glass*

The game is afoot! I have conducted an exhaustive forensic investigation
of the Ultimate Context Graph codebase to identify all mechanisms that
mask the system not working as the PRD expects.

VERDICT: GUILTY - Multiple stub implementations exist that mask
         the absence of real functionality.
```

---

## EXECUTIVE FINDINGS

| Category | Count | Risk Level |
|----------|-------|------------|
| **Stub Implementations** | 15+ major stubs | CRITICAL |
| **In-Memory Fallbacks** | 5 instances | HIGH |
| **Default Fallback Values** | 50+ locations | MEDIUM |
| **GPU Fallback Paths** | 3 conditional paths | HIGH |
| **Hardcoded Thresholds** | 10+ locations | MEDIUM |
| **TODO/FIXME Comments** | 20+ outstanding | MEDIUM |

---

## SECTION 1: STUB IMPLEMENTATIONS

### 1.1 Core Stubs Module (`crates/context-graph-core/src/stubs/mod.rs`)

**LOCATION**: `/home/cabdru/contextgraph/crates/context-graph-core/src/stubs/`

**EVIDENCE COLLECTED**:

```rust
// Module exports the following stubs:
pub use embedding_stub::StubEmbeddingProvider;
pub use graph_index::InMemoryGraphIndex;
pub use layers::{
    StubCoherenceLayer, StubLearningLayer, StubMemoryLayer, StubReflexLayer, StubSensingLayer,
};
pub use multi_array_stub::StubMultiArrayProvider;
pub use teleological_store_stub::InMemoryTeleologicalStore;
pub use utl_stub::StubUtlProcessor;
```

**STUBS IDENTIFIED**:

| Stub Name | Purpose | PRD Requirement | Risk |
|-----------|---------|-----------------|------|
| `StubEmbeddingProvider` | Returns deterministic fake embeddings | Real 13 embedding models | CRITICAL |
| `StubMultiArrayProvider` | Returns 13 fake embeddings | Real CUDA 13.1 embeddings | CRITICAL |
| `InMemoryTeleologicalStore` | HashMap storage | RocksDB/ScyllaDB | HIGH |
| `InMemoryGraphIndex` | In-memory HNSW | Persistent HNSW | HIGH |
| `StubUtlProcessor` | Fake UTL processing | Real UTL computation | HIGH |
| `StubSensingLayer` | Fake sensing layer | Real sensing | MEDIUM |
| `StubReflexLayer` | Fake reflex layer | Real reflex | MEDIUM |
| `StubMemoryLayer` | Fake memory layer | Real memory | MEDIUM |
| `StubLearningLayer` | Fake learning layer | Real learning | MEDIUM |
| `StubCoherenceLayer` | Fake coherence layer | Real coherence | MEDIUM |

**VERDICT**: These stubs are EXPLICITLY documented as "Ghost System phase (Phase 0)" but they are USED IN PRODUCTION CODE PATH.

---

### 1.2 System Monitor Stubs (`crates/context-graph-core/src/monitoring.rs`)

**LOCATION**: Lines 427-613

**EVIDENCE**:

```rust
/// Stub implementation that FAILS with explicit "not implemented" errors.
#[derive(Debug, Clone, Default)]
pub struct StubSystemMonitor;

/// Stub implementation for layer status that reports honest stub status.
#[derive(Debug, Clone, Default)]
pub struct StubLayerStatusProvider;
```

**USAGE IN PRODUCTION** (`crates/context-graph-mcp/src/handlers/core.rs:264-266`):

```rust
// TASK-EMB-024: Default to stub monitors (will fail with explicit errors)
let system_monitor: Arc<dyn SystemMonitor> = Arc::new(StubSystemMonitor::new());
let layer_status_provider: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider::new());
```

**IMPACT**: The Handlers struct defaults to stub monitors. Any call to `coherence_recovery_time_ms()`, `attack_detection_rate()`, or `false_positive_rate()` will FAIL.

**RISK LEVEL**: CRITICAL - Production code defaults to stubs.

---

### 1.3 Configuration Defaults Pointing to Stubs

**LOCATION**: `/home/cabdru/contextgraph/crates/context-graph-core/src/config/sub_configs.rs`

**EVIDENCE**:

```rust
// EmbeddingConfig default (lines 86-93):
impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "stub".to_string(),  // <-- DEFAULT IS STUB
            dimension: 1536,
            max_input_length: 8191,
        }
    }
}

// UtlConfig default (lines 122-129):
impl Default for UtlConfig {
    fn default() -> Self {
        Self {
            mode: "stub".to_string(),  // <-- DEFAULT IS STUB
            default_emotional_weight: 1.0,
            consolidation_threshold: 0.7,
        }
    }
}

// StorageConfig default (lines 68-75):
impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: "memory".to_string(),  // <-- DEFAULT IS IN-MEMORY
            path: "./data/storage".to_string(),
            compression: true,
        }
    }
}

// IndexConfig default (lines 104-111):
impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            backend: "memory".to_string(),  // <-- DEFAULT IS IN-MEMORY
            hnsw_m: 16,
            hnsw_ef_construction: 200,
        }
    }
}
```

**VERDICT**: Default configuration points to stubs/in-memory, not real implementations.

---

## SECTION 2: IN-MEMORY FALLBACK STORAGE

### 2.1 InMemoryTeleologicalStore

**LOCATION**: `/home/cabdru/contextgraph/crates/context-graph-core/src/stubs/teleological_store_stub.rs`

**EVIDENCE**: This is a HashMap-based implementation that replaces RocksDB.

**USAGE LOCATIONS** (27 files reference InMemory storage):

- `crates/context-graph-mcp/src/handlers/tests/search.rs`
- `crates/context-graph-mcp/src/handlers/tests/memory.rs`
- `crates/context-graph-mcp/src/handlers/tests/mod.rs`
- `crates/context-graph-core/src/retrieval/tests.rs`
- `crates/context-graph-core/src/johari/default_manager.rs`
- ...and 22 more files

**PRD REQUIREMENT**: Real RocksDB/ScyllaDB storage with 13 column families per embedder.

**RISK LEVEL**: HIGH - Tests pass but production would fail with real persistence.

---

### 2.2 InMemoryGraphIndex

**LOCATION**: `/home/cabdru/contextgraph/crates/context-graph-core/src/stubs/graph_index.rs`

**EVIDENCE**:

```rust
pub struct InMemoryGraphIndex {
    /// HNSW index using instant-distance
    inner: RwLock<Option<Hnsw<Point, EuclideanDistance>>>,
    /// Mapping from UUID to data ID
    id_to_data_id: RwLock<HashMap<Uuid, usize>>,
    // ...
}
```

**PRD REQUIREMENT**: Persistent HNSW indexes with GPU-accelerated search.

---

## SECTION 3: GPU/CUDA FALLBACK PATHS

### 3.1 Conditional CUDA Feature Flags

**LOCATION**: Multiple files in `crates/context-graph-embeddings/src/models/pretrained/`

**EVIDENCE**:

```rust
// causal/tests.rs:241
let budget_ms = if cfg!(feature = "cuda") { 10 } else { 200 };

// entity/tests.rs:296
let budget_ms = if cfg!(feature = "cuda") { 10 } else { 200 };

// late_interaction/tests_extended.rs:85
let budget_ms = if cfg!(feature = "cuda") { 10 } else { 200 };

// code/tests_batch.rs:128
let budget_ms = if cfg!(feature = "cuda") { 10 } else { 200 };

// code/tests_edge_cases.rs:274
let budget_ms = if cfg!(feature = "cuda") { 10 } else { 500 };
```

**IMPLICATION**: Code has CPU fallback paths that run 20x slower.

### 3.2 CPU Fallback Implementations

**LOCATION**: `crates/context-graph-cuda/src/poincare/cpu.rs`

```rust
/// CPU implementation of Poincare distance for testing
/// the GPU kernel. Used for testing and as a fallback when GPU is unavailable.
```

**LOCATION**: `crates/context-graph-cuda/src/cone/cpu.rs`

```rust
/// CPU fallback and reference implementations for testing GPU kernel correctness.
```

**PRD REQUIREMENT**: Constitution AP-007 requires "RTX 5090 GPU must be available. No fallback stubs."

**HOWEVER**: The build.rs DOES panic if CUDA is unavailable:

```rust
// crates/context-graph-cuda/build.rs:47
panic!("CUDA feature is required. RTX 5090 GPU must be available. No fallback stubs.");
```

**VERDICT**: Build enforces CUDA, but CPU reference implementations exist for testing. These are ACCEPTABLE for verification purposes only.

---

## SECTION 4: DEFAULT VALUE FALLBACKS (unwrap_or patterns)

**LOCATION**: Throughout `crates/context-graph-mcp/src/handlers/`

**CRITICAL EXAMPLES**:

```rust
// utl_adapter.rs:117 - Falls back to zero vector
.unwrap_or_else(|| vec![0.0; 128])

// tools.rs:382 - Falls back to "Infancy"
.unwrap_or("Infancy");

// tools.rs:388-410 - Falls back to 0.0 for metrics
.unwrap_or(0.0);

// purpose.rs:727 - Falls back to 0.55 alignment threshold
.unwrap_or(0.55);

// core.rs:166 - Falls back to uniform weights
accuracies[i] = self.get_embedder_accuracy(i).unwrap_or(1.0 / NUM_EMBEDDERS as f32);
```

**COUNT**: 50+ instances of `unwrap_or` or `unwrap_or_else` with fallback values.

**RISK LEVEL**: MEDIUM - Masks failures with default values instead of fail-fast.

---

## SECTION 5: HARDCODED THRESHOLD VALUES

**LOCATIONS AND VALUES**:

| File | Line | Threshold | Purpose |
|------|------|-----------|---------|
| `teleological_query.rs` | 85 | `0.55` | min_alignment_threshold |
| `teleological_result.rs` | 197 | `0.55` | critical threshold |
| `teleological_result.rs` | 284 | `0.75` | OPTIMAL alignment |
| `teleological_result.rs` | 290 | `0.55` | CRITICAL alignment |
| `purpose.rs` | 721 | `0.55` | Warning threshold |
| `storage/types.rs` | 534 | `0.55` | Constitution default |
| `layers/coherence.rs` | 42 | `0.75` | Coherence range base |
| `layers/learning.rs` | 42 | `0.55` | Learning range base |
| `layers/memory.rs` | 51 | `0.55` | Memory retrieval threshold |

**CONSTITUTION VIOLATION**: "NO hardcoded thresholds" - These should be configurable.

**RISK LEVEL**: MEDIUM - Thresholds are hardcoded rather than learned/configured.

---

## SECTION 6: TODO/FIXME COMMENTS (Incomplete Work)

**LOCATIONS**:

```rust
// storage/rocksdb_store.rs:658
// TODO: Integrate with HNSW indexes for proper ANN search

// storage/rocksdb_store.rs:684
// TODO: Compute scores for all 13 embedders

// storage/rocksdb_store.rs:761
let embedder_scores = [0.0f32; 13]; // TODO: Compute actual scores

// storage/rocksdb_store.rs:819
// TODO: Implement BM25 or other scoring

// storage/indexes.rs:13
// TODO: Implement in TASK-M02-023

// johari/default_manager.rs:81
// TODO: In a production system, we might want to limit the size

// graph/query/mod.rs:8-12
// - **Semantic Search**: Vector similarity + graph context (TODO: M04-T18)
// - **Entailment Query**: IS-A hierarchy using cones (TODO: M04-T20)
// - **Contradiction Detection**: Identify conflicting knowledge (TODO: M04-T21)
// - **Query Builder**: Fluent API for complex queries (TODO: M04-T27)
// - **Graph API**: High-level CRUD operations (TODO: M04-T28)

// storage/quantized.rs:503
// TEMPORARY: Return a static router. This is safe because QuantizationRouter
```

**COUNT**: 20+ TODO/FIXME comments indicating incomplete functionality.

---

## SECTION 7: DIMENSION VERIFICATION

**STATUS**: PASS - Dimensions are correctly defined.

**EVIDENCE** (`/home/cabdru/contextgraph/crates/context-graph-embeddings/src/types/dimensions/mod.rs`):

```rust
assert_eq!(projected_dimension_by_index(0), 1024);  // Semantic (E1)
assert_eq!(projected_dimension_by_index(5), 1536);  // Sparse (E6)
assert_eq!(projected_dimension_by_index(6), 768);   // Code (E7)
assert_eq!(projected_dimension_by_index(8), 1024);  // HDC (E9)
assert_eq!(TOTAL_DIMENSION, 9856);  // Sum of all 13 embeddings
```

**VERDICT**: Dimension constants are correctly defined per PRD.

---

## SECTION 8: TESTS USING STUBS

**CRITICAL FINDING**: 27+ test files use `InMemoryTeleologicalStore` and stub implementations:

```
crates/context-graph-mcp/src/handlers/tests/search.rs
crates/context-graph-mcp/src/handlers/tests/memory.rs
crates/context-graph-mcp/src/handlers/tests/mod.rs
crates/context-graph-mcp/src/handlers/tests/full_state_verification*.rs
crates/context-graph-core/src/retrieval/tests.rs
...
```

**IMPACT**: Tests pass with stubs but DO NOT verify real RocksDB behavior.

---

## SECTION 9: POSITIVE FINDINGS (What IS Working)

1. **NO `cfg(feature = "mock")` or `cfg(feature = "stub")` feature flags** - Good, no hidden mock modes.

2. **Stubs are HONESTLY labeled** - All stubs are in a `stubs/` module with clear naming.

3. **StubSystemMonitor FAILS explicitly** - Returns `Err(NotImplemented)` rather than fake data.

4. **Build enforces CUDA** - `context-graph-cuda/build.rs` panics if CUDA unavailable.

5. **Dimensions are validated** - Compile-time assertions verify correct dimensions.

6. **No silent try/catch** - Error handling generally propagates errors.

---

## VERDICT AND RECOMMENDATIONS

### GUILTY COMPONENTS

| Component | Status | Required Action |
|-----------|--------|-----------------|
| `StubMultiArrayProvider` | GUILTY | Replace with real 13-model embeddings |
| `InMemoryTeleologicalStore` | GUILTY | Replace with RocksDB implementation |
| `StubUtlProcessor` | GUILTY | Replace with real UTL computation |
| `StubSystemMonitor` | GUILTY | Implement real metrics collection |
| Default config (`model: "stub"`) | GUILTY | Change defaults to real implementations |
| Hardcoded thresholds | GUILTY | Make configurable |

### INNOCENT (BUT WATCH CAREFULLY)

| Component | Status | Notes |
|-----------|--------|-------|
| CPU fallback implementations | INNOCENT | Only for testing, build enforces CUDA |
| Dimension constants | INNOCENT | Correctly defined |
| Error propagation | INNOCENT | Generally fail-fast |

---

## CHAIN OF CUSTODY

| Timestamp | Action | Evidence |
|-----------|--------|----------|
| 2026-01-06 | Grep scan for "mock" | 95+ matches |
| 2026-01-06 | Grep scan for "stub" | 500+ matches |
| 2026-01-06 | Grep scan for "fallback" | 150+ matches |
| 2026-01-06 | Grep scan for "TODO/FIXME" | 20+ matches |
| 2026-01-06 | Read stubs/mod.rs | Identified 10 stub types |
| 2026-01-06 | Read monitoring.rs | Identified stub monitors |
| 2026-01-06 | Read sub_configs.rs | Identified default stubs |
| 2026-01-06 | Read dimensions/mod.rs | Verified dimension correctness |

---

## CASE CLOSED

```
HOLMES: *slams fist on table*

The evidence is OVERWHELMING. This codebase is RIDDLED with stub
implementations that mask the absence of real functionality.

The PRD requires:
- Real CUDA 13.1 embeddings from 13 models
- Real RocksDB/ScyllaDB storage
- Real HNSW indexes
- Real UTL computation

But the DEFAULT configuration uses:
- Stub embedding providers
- In-memory HashMap storage
- Stub UTL processor
- Stub system monitors

The system APPEARS to work because stubs return deterministic
fake data. Tests PASS but they test NOTHING real.

REMEDIATION REQUIRED:
1. Replace all default configurations to point to real implementations
2. Ensure InMemoryTeleologicalStore is ONLY used in tests/fixtures
3. Implement real SystemMonitor with actual metrics
4. Complete all TODO items in rocksdb_store.rs
5. Make hardcoded thresholds configurable

This case remains OPEN until the Ghost System phase ends.
```

---

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."* - Sherlock Holmes
