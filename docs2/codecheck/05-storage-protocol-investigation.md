# Sherlock Holmes Forensic Investigation Report: Storage, Protocol, and CUDA Layers

**Case ID:** SHERLOCK-05-STORAGE-PROTOCOL-CUDA
**Date:** 2026-01-06
**Investigator:** Sherlock Holmes (Code Forensics Division)
**Subject:** Storage Crate, Protocol Layer, and CUDA Acceleration
**Verdict:** MIXED - Real Implementations Exist Alongside Critical Gaps

---

## EXECUTIVE SUMMARY

*"The world is full of obvious things which nobody by any chance ever observes."*

Upon forensic examination of the storage, protocol, and CUDA layers, I have uncovered a **SURPRISING FINDING**: Unlike the MCP handlers and core crate stubs, the storage layer contains **REAL, PRODUCTION-GRADE IMPLEMENTATIONS**. The RocksDB backend is genuine, the CUDA kernels are professionally written, and the HNSW configuration is well-architected.

However, critical infrastructure gaps remain that prevent the real implementations from being used effectively.

---

## VERDICT MATRIX

| Layer | Status | Severity |
|-------|--------|----------|
| RocksDB Backend | REAL IMPLEMENTATION | INNOCENT |
| HNSW Configuration | REAL, CONFIG-ONLY | INNOCENT |
| CUDA Kernels (.cu) | REAL, PROFESSIONAL | INNOCENT |
| CUDA Rust Bindings | STUB (documented) | MEDIUM |
| Protocol Layer | REAL JSON-RPC | INNOCENT |
| FAISS GPU Index | NOT IMPLEMENTED | CRITICAL |
| Redis Cache | NOT IMPLEMENTED | HIGH |
| TimescaleDB | NOT IMPLEMENTED | HIGH |
| ScyllaDB | NOT IMPLEMENTED | HIGH |

---

## DETAILED FINDINGS

### FINDING 1: RocksDB Backend is REAL (INNOCENT)

**Location:** `/home/cabdru/contextgraph/crates/context-graph-storage/src/rocksdb_backend/`

**Evidence:**
The RocksDB backend is a genuine, production-quality implementation:

```rust
// From rocksdb_backend/core.rs
use rocksdb::{Cache, Options, DB};

impl RocksDbMemex {
    pub fn open(path: impl AsRef<Path>) -> StorageResult<Self> {
        // Real RocksDB initialization with column families
    }
}
```

**Integration Tests Verified:**
- `/home/cabdru/contextgraph/crates/context-graph-storage/tests/teleological_integration.rs`
- Tests open REAL RocksDB with 16 column families
- Tests persist and retrieve REAL TeleologicalFingerprints
- Tests verify REAL SPLADE inverted index operations

**Verdict:** INNOCENT - This is genuine production code.

---

### FINDING 2: CUDA Kernels are REAL and Professional (INNOCENT)

**Location:** `/home/cabdru/contextgraph/crates/context-graph-cuda/kernels/`

**Evidence - poincare_distance.cu:**
```cuda
// CUDA kernel for batch Poincare ball distance computation
// Target: RTX 5090 (Compute Capability 12.0, CUDA 13.1)
// Performance: <1ms for 1K x 1K distance matrix

__global__ void poincare_distance_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ distances,
    int n_queries,
    int n_database,
    float curvature
) {
    // Professional implementation with:
    // - Shared memory optimization
    // - Coalesced memory access
    // - Numerical stability (EPS, ARCTANH_CLAMP)
    // - Warp-aligned block dimensions (32x8 = 256 threads)
}
```

**Evidence - cone_check.cu (15,310 bytes):**
- Real CUDA implementation for cone membership checking
- Properly tuned for Blackwell architecture

**Verdict:** INNOCENT - These are production-grade CUDA kernels.

---

### FINDING 3: CUDA Rust Bindings are Stubs (DOCUMENTED - MEDIUM)

**Location:** `/home/cabdru/contextgraph/crates/context-graph-cuda/src/stub.rs`

**Evidence:**
```rust
//! Stub implementations for CPU fallback.
//!
//! These stubs allow the crate to compile and run without CUDA,
//! using CPU implementations for testing and development.

pub struct StubVectorOps {
    // CPU fallback implementation
}

impl VectorOps for StubVectorOps {
    fn is_gpu_available(&self) -> bool {
        false  // Honest about being CPU-only
    }

    fn device_name(&self) -> &str {
        "CPU (Stub)"  // Clearly labeled
    }
}
```

**Analysis:**
- The stub is HONESTLY LABELED as a stub
- The `.cu` kernel files exist and are real
- The Rust bindings need FFI connection to kernels
- This is a documented Phase 0 (Ghost System) gap

**Verdict:** MEDIUM - Documented limitation, real kernels exist but aren't connected.

---

### FINDING 4: HNSW Configuration is Well-Architected (INNOCENT)

**Location:** `/home/cabdru/contextgraph/crates/context-graph-storage/src/teleological/indexes/`

**Evidence:**
```rust
// From hnsw_config/config.rs
pub struct HnswConfig {
    pub m: usize,              // Bi-directional links per node
    pub ef_construction: usize, // Construction quality
    pub ef_search: usize,       // Search candidates
    pub metric: DistanceMetric,
    pub dimension: usize,
}

impl HnswConfig {
    /// FAIL FAST: No fallbacks - panics on invalid config
    pub fn new(...) -> Self {
        if m < 2 {
            panic!("HNSW CONFIG ERROR: M must be >= 2, got {}", m);
        }
        // ... validation continues
    }
}
```

**Key Points:**
- 12 HNSW configurations for E1-E11 + PurposeVector + Matryoshka128D
- 2 inverted index configurations for E6 Sparse and E13 SPLADE
- Proper fail-fast validation with panics
- Dimension constants match constitution.yaml

**Verdict:** INNOCENT - Proper configuration architecture.

---

### FINDING 5: Protocol Layer is Real JSON-RPC (INNOCENT)

**Location:** `/home/cabdru/contextgraph/crates/context-graph-mcp/src/protocol.rs`

**Evidence:**
```rust
/// JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<JsonRpcId>,
    pub method: String,
    pub params: Option<serde_json::Value>,
}
```

**Key Points:**
- Proper JSON-RPC 2.0 implementation
- Comprehensive error codes for teleological operations
- CognitivePulse extension header support
- No fake success paths - errors are properly propagated

**Verdict:** INNOCENT - Standard protocol implementation.

---

### FINDING 6: Missing Database Backends (CRITICAL)

**Evidence Search:**
```bash
grep -r "ScyllaDB|TimescaleDB|Redis|PostgreSQL" crates/
# Result: ZERO implementation files
# Only mentions in comments/docs
```

**Constitution Requirements (NOT IMPLEMENTED):**

| Database | Purpose | Status |
|----------|---------|--------|
| ScyllaDB | Distributed storage for memories | NOT FOUND |
| TimescaleDB | Purpose evolution timeline | NOT FOUND |
| Redis | Hot cache layer | NOT FOUND |
| PostgreSQL | Graph metadata | NOT FOUND (only mentioned) |
| FAISS GPU | Vector similarity index | NOT FOUND |

**Verdict:** CRITICAL - Required databases do not exist.

---

### FINDING 7: In-Memory Store is the ONLY TeleologicalMemoryStore (HIGH)

**Location:** `/home/cabdru/contextgraph/crates/context-graph-core/src/stubs/teleological_store_stub.rs`

**Evidence:**
```rust
/// In-memory implementation of TeleologicalMemoryStore.
///
/// Thread-safe via `DashMap`. No persistence - data lost on drop.
/// Uses real algorithms for search (not mocks).
///
/// # Performance
///
/// - O(n) search operations (no indexing)
/// - O(1) CRUD operations via HashMap
/// - ~46KB per fingerprint in memory
pub struct InMemoryTeleologicalStore {
    data: DashMap<Uuid, TeleologicalFingerprint>,
    deleted: DashMap<Uuid, ()>,
    size_bytes: AtomicUsize,
}
```

**The Crime:**
- The TeleologicalMemoryStore trait exists
- RocksDB backend exists for Memex (old API)
- But NO RocksDB implementation of TeleologicalMemoryStore exists!
- The MCP handlers use InMemoryTeleologicalStore exclusively

**Impact:**
- All teleological fingerprint storage is IN-MEMORY ONLY
- Data is lost on restart
- No real persistence despite RocksDB existing

**Verdict:** HIGH - Architectural gap between old Memex and new TeleologicalMemoryStore.

---

### FINDING 8: HNSW Index is CONFIG-ONLY (CRITICAL)

**Evidence Search:**
```bash
grep -r "impl.*HnswIndex|struct.*HnswIndex" crates/context-graph-storage/
# Result: ZERO implementation structs
# Only configuration and functions that return configs
```

**Analysis:**
- HnswConfig struct EXISTS and is well-designed
- HnswIndex implementation DOES NOT EXIST
- Configuration without implementation = unusable

**What exists:**
- `get_hnsw_config()` - returns configuration
- `all_hnsw_configs()` - returns HashMap of 12 configs
- `EmbedderIndex` enum with 15 variants

**What is missing:**
- Actual HNSW graph data structure
- Insert/delete operations
- ANN search implementation
- Memory management

**Verdict:** CRITICAL - Configuration exists but no index implementation.

---

## EVIDENCE CHAIN OF CUSTODY

| Timestamp | Action | Finding |
|-----------|--------|---------|
| 2026-01-06 | Glob storage crate files | 50+ Rust files found |
| 2026-01-06 | Read rocksdb_backend/core.rs | Real RocksDB implementation |
| 2026-01-06 | Read CUDA kernels | Professional .cu files (25KB total) |
| 2026-01-06 | Read stub.rs | Documented CPU fallback |
| 2026-01-06 | Read HNSW config | Config-only, no index impl |
| 2026-01-06 | Read protocol.rs | Real JSON-RPC 2.0 |
| 2026-01-06 | Search for ScyllaDB/Redis | Not found |
| 2026-01-06 | Verify teleological integration tests | Real RocksDB tests exist |

---

## ARCHITECTURAL GAP ANALYSIS

### The "Two Storage APIs" Problem

```
OLD API (Memex trait):           NEW API (TeleologicalMemoryStore trait):
+-------------------+            +--------------------------------+
| RocksDbMemex      | <-- REAL   | InMemoryTeleologicalStore      | <-- STUB
| - MemoryNode      |            | - TeleologicalFingerprint      |
| - GraphEdge       |            | - 13 embeddings                |
| - Embeddings      |            | - PurposeVector                |
+-------------------+            | - JohariFingerprint            |
        |                        +--------------------------------+
        v                                    |
    RocksDB                              DashMap (in-memory)
    (REAL PERSISTENCE)                   (NO PERSISTENCE)
```

**The Gap:** The system evolved from Memex to TeleologicalMemoryStore but the RocksDB implementation was never migrated.

---

### What Would Actually Work Today

If you ran the system today:

1. **Works:** JSON-RPC protocol parsing
2. **Works:** RocksDB for old MemoryNode/GraphEdge storage
3. **Works:** HNSW configuration generation
4. **Works:** CUDA kernel compilation (with CUDA SDK)

5. **Broken:** TeleologicalFingerprint persistence (in-memory only)
6. **Broken:** CUDA kernel execution (Rust bindings are stubs)
7. **Broken:** 13-index ANN search (no HNSW index implementation)
8. **Broken:** Purpose evolution timeline (no TimescaleDB)
9. **Broken:** Hot caching (no Redis)

---

## RECOMMENDATIONS

### CRITICAL Priority (Week 1)

1. **Create RocksDbTeleologicalStore**
   - Implement TeleologicalMemoryStore trait for RocksDB
   - Use existing column family architecture
   - Migrate 16 CFs from test to production code

2. **Connect CUDA FFI**
   - Link Rust bindings to .cu kernels
   - Add cudarc integration
   - Test on RTX 5090 (or fallback detection)

### HIGH Priority (Week 2-3)

3. **Implement HNSW Index**
   - Use instant-distance or hnsw crate
   - Create HnswIndex struct with insert/search
   - Wire to HnswConfig

4. **Remove Architectural Confusion**
   - Deprecate old Memex API or migrate it
   - Single storage abstraction for TeleologicalFingerprint

### MEDIUM Priority (Month 1)

5. **Add External Databases**
   - Redis for hot cache
   - Consider if ScyllaDB/TimescaleDB are actually needed
   - Or simplify to RocksDB + Redis only

---

## VERDICT

**MIXED JUDGMENT**

The storage layer contains **more real code than expected**:
- RocksDB backend: REAL
- CUDA kernels: REAL
- Protocol layer: REAL
- HNSW configuration: REAL

But critical gaps prevent production use:
- TeleologicalMemoryStore has no persistent implementation
- CUDA bindings are stubs
- HNSW is config-only
- External databases don't exist

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*

The truth is: **The foundation is real, but the bridges aren't built.**

---

## CASE STATUS

**PARTIALLY GUILTY** - Real code exists but isn't wired together.

**CONFIDENCE:** HIGH

**INVESTIGATION:** COMPLETE

---

*"Data! Data! Data! I can't make bricks without clay."*

**Sherlock Holmes**
**Consulting Detective, Code Forensics Division**
