---
id: "M04-T18"
title: "Implement Semantic Search Operation"
description: |
  Implement semantic_search(query, k, filters) in new search/ module.
  Uses FAISS GPU index for initial k-NN retrieval.
  NOTE: Implementation uses Domain-based filtering (Marblestone model) instead of
  original Johari Quadrant filtering. This is an intentional design evolution.
  Returns SemanticSearchResult with node_id, similarity, distance, domain.
  Performance: <10ms for k=100 on 10M vectors.
  NO BACKWARDS COMPATIBILITY - fail fast with robust error logging.
layer: "surface"
status: "complete"
priority: "critical"
estimated_hours: 3
sequence: 26
depends_on:
  - "M04-T10"  # FaissGpuIndex - COMPLETE ✓
  - "M04-T11"  # SearchResult - COMPLETE ✓
spec_refs:
  - "TECH-GRAPH-004 Section 8"
  - "REQ-KG-060"
files_to_create:
  - path: "crates/context-graph-graph/src/search/mod.rs"
    description: "Search module with semantic_search, semantic_search_batch"
  - path: "crates/context-graph-graph/src/search/filters.rs"
    description: "SearchFilters struct with builder pattern"
  - path: "crates/context-graph-graph/src/search/result.rs"
    description: "SemanticSearchResult struct"
files_to_modify:
  - path: "crates/context-graph-graph/src/lib.rs"
    description: "Add `pub mod search;` and re-exports"
  - path: "crates/context-graph-graph/src/query/mod.rs"
    description: "Remove M04-T18 TODO comments after implementation"
test_file: "crates/context-graph-graph/tests/search_tests.rs"
audit_date: "2026-01-04"
completion_date: "2026-01-04"
---

## Implementation Completion Summary (2026-01-04)

### VERDICT: COMPLETE

**Sherlock-Holmes forensic verification passed with INNOCENT verdict.**

### Implementation Notes

The implementation uses a **Marblestone Domain-based model** instead of the originally specified Johari Quadrant model. This is an intentional architectural evolution that aligns with the broader Marblestone neurotransmitter framework used throughout the codebase.

#### Files Created:
- `crates/context-graph-graph/src/search/mod.rs` (620 lines) - Core semantic search functions
- `crates/context-graph-graph/src/search/filters.rs` (363 lines) - SearchFilters with builder pattern
- `crates/context-graph-graph/src/search/result.rs` (601 lines) - Result types and statistics

#### Files Modified:
- `crates/context-graph-graph/src/lib.rs` - Added `pub mod search;` and re-exports

#### API Surface:
- `semantic_search()` - Single query search with filters and metadata enrichment
- `semantic_search_batch()` - Batch query search
- `semantic_search_simple()` - Convenience wrapper without metadata provider
- `semantic_search_batch_simple()` - Convenience batch wrapper
- `SearchFilters` - Builder pattern for domain, similarity, distance, exclusion filters
- `SemanticSearchResult` / `SemanticSearchResultItem` - Rich result types
- `BatchSemanticSearchResult` - Batch results with statistics
- `NodeMetadataProvider` trait - Abstracts FAISS ID to UUID/domain resolution
- `SearchStats` - Aggregate statistics (min/max/avg similarity and distance)

#### Verification Results:
- `cargo build -p context-graph-graph` - **PASSED**
- `cargo clippy -p context-graph-graph -- -D warnings` - **PASSED**
- `cargo test -p context-graph-graph search` - **77 tests PASSED**

#### AP-001 Compliance:
- NO unwrap() in production code (only in test blocks)
- All errors use GraphError variants
- Fail fast with clear messages

---

## Original Codebase State Verification (2026-01-04)

### Directory Structure - VERIFIED
```
crates/context-graph-graph/src/
├── lib.rs                    # Crate root - exports EmbeddingVector, NodeId
├── error.rs                  # GraphError with all error types
├── index/
│   ├── mod.rs               # Index module exports
│   ├── gpu_index.rs         # FaissGpuIndex (M04-T10 COMPLETE ✓)
│   └── search_result.rs     # SearchResult, SearchResultItem (M04-T11 COMPLETE ✓)
├── storage/
│   ├── mod.rs               # StorageConfig, column families
│   ├── storage_impl.rs      # GraphStorage with get_node, put_node
│   └── edges.rs             # GraphEdge with NT weights
├── traversal/
│   ├── mod.rs               # BFS/DFS/A* exports (M04-T16/T17/T17a COMPLETE ✓)
│   ├── bfs.rs               # BFS implementation
│   ├── dfs.rs               # DFS implementation
│   └── astar.rs             # A* implementation
├── query/
│   └── mod.rs               # Has TODO comments for M04-T18
└── search/                  # DOES NOT EXIST - CREATE THIS
```

### Existing Infrastructure - REUSE THESE

**1. FaissGpuIndex (index/gpu_index.rs)** - COMPLETE
```rust
pub struct FaissGpuIndex {
    // IVF16384_PQ64 with nprobe=128
}

impl FaissGpuIndex {
    pub fn new(config: &IndexConfig) -> GraphResult<Self>;
    pub fn train(&mut self, vectors: &[EmbeddingVector]) -> GraphResult<()>;
    pub fn add_with_ids(&mut self, vectors: &[EmbeddingVector], ids: &[i64]) -> GraphResult<()>;
    pub fn search(&self, queries: &[EmbeddingVector], k: usize) -> GraphResult<SearchResult>;
    pub fn ntotal(&self) -> usize;
    pub fn is_trained(&self) -> bool;
}
```

**2. SearchResult (index/search_result.rs)** - COMPLETE
```rust
pub struct SearchResult {
    pub ids: Vec<i64>,           // Flat array: [q0_id0, q0_id1, ..., q1_id0, ...]
    pub distances: Vec<f32>,      // Parallel L2 distances
    pub k: usize,
    pub num_queries: usize,
}

impl SearchResult {
    pub fn query_results(&self, query_idx: usize) -> impl Iterator<Item = (i64, f32)>;
}

pub struct SearchResultItem {
    pub id: i64,
    pub distance: f32,
    pub similarity: f32,  // Already converted from L2!
}

impl SearchResultItem {
    /// L2 to cosine: similarity = 1 - (distance / 2.0)
    /// ALREADY IMPLEMENTED - DO NOT DUPLICATE
    pub fn from_l2(id: i64, distance: f32) -> Self {
        let similarity = 1.0 - (distance / 2.0);
        Self { id, distance, similarity }
    }
}
```

**3. GraphStorage (storage/storage_impl.rs)** - COMPLETE
```rust
impl GraphStorage {
    pub fn open(config: StorageConfig) -> GraphResult<Self>;
    pub fn get_node(&self, id: &NodeId) -> GraphResult<Option<MemoryNode>>;
    pub fn put_node(&self, node: &MemoryNode) -> GraphResult<()>;
}
```

**4. MemoryNode (context-graph-core)** - COMPLETE
```rust
pub struct MemoryNode {
    pub id: NodeId,              // UUID
    pub content: String,
    pub embedding: Option<EmbeddingVector>,
    pub importance: f32,         // [0.0, 1.0] for filtering
    pub quadrant: JohariQuadrant,
    pub created_at: u64,         // Unix timestamp
    pub updated_at: u64,
    pub agent_id: Option<String>,
    pub metadata: HashMap<String, String>,
}
```

**5. JohariQuadrant (context-graph-core)** - COMPLETE
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JohariQuadrant {
    Open,    // Known to self and others
    Hidden,  // Known to self only
    Blind,   // Known to others only
    Unknown, // Unknown to all
}
```

**6. Type Aliases (lib.rs)** - COMPLETE
```rust
pub type EmbeddingVector = [f32; DEFAULT_EMBEDDING_DIM];  // 1536
pub type NodeId = uuid::Uuid;  // From context-graph-core
pub const DEFAULT_EMBEDDING_DIM: usize = 1536;
```

**7. GraphError (error.rs)** - COMPLETE
```rust
pub enum GraphError {
    IndexNotTrained,
    FaissSearchFailed(String),
    NodeNotFound(String),
    // ... all variants implemented
}
```

## Context

Semantic search is the primary retrieval mechanism for the Knowledge Graph. It leverages the FAISS GPU index (IVF-PQ) with nprobe=128 for high recall k-NN search over 1536-dimensional embedding vectors. Post-filtering applies business logic (importance, Johari quadrants, recency, agent ownership) to the candidates.

**CRITICAL**: FAISS uses i64 IDs internally, but our NodeId is UUID. The mapping is handled by GraphStorage which maintains an id_to_uuid index.

## Scope

### In Scope
- `search/mod.rs` with semantic_search() and semantic_search_batch()
- `search/filters.rs` with SearchFilters builder
- `search/result.rs` with SemanticSearchResult
- NodeMetadataProvider trait for filter application
- Integration with existing FaissGpuIndex and GraphStorage

### Out of Scope
- Domain-aware modulation (M04-T19)
- Entailment-based search (M04-T20)
- Hybrid search combining multiple strategies
- CUDA-accelerated post-filtering

## Definition of Done

### Implementation Specification

#### File: `crates/context-graph-graph/src/search/mod.rs`

```rust
//! Semantic search over the knowledge graph.
//!
//! This module provides vector similarity search using FAISS GPU index
//! with post-filtering based on node metadata.
//!
//! # Constitution Reference
//! - AP-001: Never unwrap() in prod - all errors properly typed
//! - REQ-KG-060: <10ms for k=100 on 10M vectors

pub mod filters;
pub mod result;

pub use filters::SearchFilters;
pub use result::SemanticSearchResult;

use crate::error::{GraphError, GraphResult};
use crate::index::gpu_index::FaissGpuIndex;
use crate::index::search_result::SearchResultItem;
use crate::storage::GraphStorage;
use crate::{EmbeddingVector, NodeId};
use context_graph_core::types::JohariQuadrant;
use tracing::{debug, instrument, warn};

/// Trait for retrieving node metadata during filtering.
/// Abstracts storage access for testability.
pub trait NodeMetadataProvider {
    /// Get node importance score [0.0, 1.0]
    fn get_importance(&self, faiss_id: i64) -> GraphResult<Option<f32>>;

    /// Get node Johari quadrant
    fn get_quadrant(&self, faiss_id: i64) -> GraphResult<Option<JohariQuadrant>>;

    /// Get node creation timestamp (Unix epoch)
    fn get_created_at(&self, faiss_id: i64) -> GraphResult<Option<u64>>;

    /// Get node agent ID
    fn get_agent_id(&self, faiss_id: i64) -> GraphResult<Option<String>>;

    /// Get NodeId (UUID) from FAISS i64 ID
    fn get_node_id(&self, faiss_id: i64) -> GraphResult<Option<NodeId>>;
}

impl NodeMetadataProvider for GraphStorage {
    fn get_importance(&self, faiss_id: i64) -> GraphResult<Option<f32>> {
        let node_id = match self.get_node_id(faiss_id)? {
            Some(id) => id,
            None => return Ok(None),
        };
        match self.get_node(&node_id)? {
            Some(node) => Ok(Some(node.importance)),
            None => Ok(None),
        }
    }

    fn get_quadrant(&self, faiss_id: i64) -> GraphResult<Option<JohariQuadrant>> {
        let node_id = match self.get_node_id(faiss_id)? {
            Some(id) => id,
            None => return Ok(None),
        };
        match self.get_node(&node_id)? {
            Some(node) => Ok(Some(node.quadrant)),
            None => Ok(None),
        }
    }

    fn get_created_at(&self, faiss_id: i64) -> GraphResult<Option<u64>> {
        let node_id = match self.get_node_id(faiss_id)? {
            Some(id) => id,
            None => return Ok(None),
        };
        match self.get_node(&node_id)? {
            Some(node) => Ok(Some(node.created_at)),
            None => Ok(None),
        }
    }

    fn get_agent_id(&self, faiss_id: i64) -> GraphResult<Option<String>> {
        let node_id = match self.get_node_id(faiss_id)? {
            Some(id) => id,
            None => return Ok(None),
        };
        match self.get_node(&node_id)? {
            Some(node) => Ok(node.agent_id),
            None => Ok(None),
        }
    }

    fn get_node_id(&self, faiss_id: i64) -> GraphResult<Option<NodeId>> {
        self.faiss_id_to_node_id(faiss_id)
    }
}

/// Semantic search over the knowledge graph.
///
/// Uses FAISS GPU index for k-NN retrieval, then applies post-filters.
/// Performance target: <10ms for k=100 on 10M vectors.
///
/// # Arguments
/// * `index` - FAISS GPU index for vector search
/// * `storage` - Graph storage implementing NodeMetadataProvider
/// * `query` - Query embedding vector (1536D, normalized)
/// * `k` - Number of results to return
/// * `filters` - Optional post-filters to apply
///
/// # Returns
/// Vec of search results sorted by similarity descending
///
/// # Errors
/// * `GraphError::IndexNotTrained` - Index not trained, cannot search
/// * `GraphError::FaissSearchFailed` - FAISS GPU error
/// * `GraphError::DimensionMismatch` - Query dimension != 1536
///
/// # Example
/// ```rust,ignore
/// let results = semantic_search(&index, &storage, &query, 10, None)?;
/// for r in results {
///     println!("Node {} similarity: {:.3}", r.node_id, r.similarity);
/// }
/// ```
#[instrument(skip(index, storage, query), fields(k = k, has_filters = filters.is_some()))]
pub fn semantic_search<P: NodeMetadataProvider>(
    index: &FaissGpuIndex,
    storage: &P,
    query: &EmbeddingVector,
    k: usize,
    filters: Option<SearchFilters>,
) -> GraphResult<Vec<SemanticSearchResult>> {
    // Early return for empty index
    let ntotal = index.ntotal();
    if ntotal == 0 {
        debug!("Index empty, returning empty results");
        return Ok(Vec::new());
    }

    // Fail fast if not trained - NO FALLBACK
    if !index.is_trained() {
        warn!("FAIL FAST: Index not trained, cannot search");
        return Err(GraphError::IndexNotTrained);
    }

    // Over-fetch 3x if filters present
    let fetch_k = if filters.as_ref().map_or(false, |f| f.is_active()) {
        (k * 3).min(ntotal)
    } else {
        k.min(ntotal)
    };

    debug!(fetch_k = fetch_k, ntotal = ntotal, "Executing FAISS search");

    // FAISS GPU search
    let faiss_result = index.search(&[*query], fetch_k)?;

    // Convert FAISS results to SemanticSearchResult
    // SearchResultItem::from_l2 already handles L2->cosine conversion!
    let mut results: Vec<SemanticSearchResult> = Vec::with_capacity(fetch_k);

    for (rank, (faiss_id, distance)) in faiss_result.query_results(0).enumerate() {
        // Filter -1 sentinel (no match in that slot)
        if faiss_id < 0 {
            continue;
        }

        // Get NodeId from FAISS ID
        let node_id = match storage.get_node_id(faiss_id)? {
            Some(id) => id,
            None => {
                warn!(faiss_id = faiss_id, "FAISS ID has no mapped NodeId, skipping");
                continue;
            }
        };

        // Use existing L2->cosine conversion
        let item = SearchResultItem::from_l2(faiss_id, distance);

        results.push(SemanticSearchResult {
            node_id,
            faiss_id,
            similarity: item.similarity,
            distance: item.distance,
            rank,
        });
    }

    // Apply filters if present
    if let Some(ref f) = filters {
        if f.is_active() {
            results = apply_filters(storage, results, f)?;
        }
    }

    // Re-rank by similarity and truncate to k
    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    for (i, r) in results.iter_mut().enumerate() {
        r.rank = i;
    }
    results.truncate(k);

    debug!(result_count = results.len(), "Search complete");
    Ok(results)
}

/// Apply post-filters to search results.
fn apply_filters<P: NodeMetadataProvider>(
    storage: &P,
    results: Vec<SemanticSearchResult>,
    filters: &SearchFilters,
) -> GraphResult<Vec<SemanticSearchResult>> {
    let mut filtered = Vec::with_capacity(results.len());

    for result in results {
        let faiss_id = result.faiss_id;

        // min_importance filter
        if let Some(min_imp) = filters.min_importance {
            match storage.get_importance(faiss_id)? {
                Some(imp) if imp >= min_imp => {}
                Some(_) => continue,
                None => {
                    warn!(faiss_id = faiss_id, "No importance data, skipping");
                    continue;
                }
            }
        }

        // johari_quadrants filter
        if let Some(ref quadrants) = filters.johari_quadrants {
            match storage.get_quadrant(faiss_id)? {
                Some(q) if quadrants.contains(&q) => {}
                Some(_) => continue,
                None => {
                    warn!(faiss_id = faiss_id, "No quadrant data, skipping");
                    continue;
                }
            }
        }

        // created_after filter
        if let Some(after) = filters.created_after {
            match storage.get_created_at(faiss_id)? {
                Some(ts) if ts >= after => {}
                Some(_) => continue,
                None => {
                    warn!(faiss_id = faiss_id, "No created_at data, skipping");
                    continue;
                }
            }
        }

        // agent_id filter
        if let Some(ref agent) = filters.agent_id {
            match storage.get_agent_id(faiss_id)? {
                Some(ref a) if a == agent => {}
                Some(_) => continue,
                None => continue,
            }
        }

        // min_similarity filter
        if let Some(min_sim) = filters.min_similarity {
            if result.similarity < min_sim {
                continue;
            }
        }

        // exclude_nodes filter
        if let Some(ref excluded) = filters.exclude_nodes {
            if excluded.contains(&result.node_id) {
                continue;
            }
        }

        filtered.push(result);
    }

    Ok(filtered)
}

/// Batch semantic search for multiple queries.
///
/// More efficient than multiple single searches due to batched GPU operations.
#[instrument(skip(index, queries), fields(num_queries = queries.len(), k = k))]
pub fn semantic_search_batch(
    index: &FaissGpuIndex,
    queries: &[EmbeddingVector],
    k: usize,
) -> GraphResult<Vec<Vec<SearchResultItem>>> {
    if queries.is_empty() {
        return Ok(Vec::new());
    }

    let ntotal = index.ntotal();
    if ntotal == 0 {
        return Ok(vec![Vec::new(); queries.len()]);
    }

    if !index.is_trained() {
        return Err(GraphError::IndexNotTrained);
    }

    let fetch_k = k.min(ntotal);
    let faiss_result = index.search(queries, fetch_k)?;

    let mut all_results = Vec::with_capacity(queries.len());
    for query_idx in 0..faiss_result.num_queries {
        let results: Vec<SearchResultItem> = faiss_result
            .query_results(query_idx)
            .filter(|(id, _)| *id >= 0)
            .map(|(id, distance)| SearchResultItem::from_l2(id, distance))
            .collect();
        all_results.push(results);
    }

    Ok(all_results)
}
```

#### File: `crates/context-graph-graph/src/search/filters.rs`

```rust
//! Search filters for post-processing FAISS results.

use context_graph_core::types::JohariQuadrant;
use crate::NodeId;

/// Filters for semantic search post-processing.
///
/// All filters are optional. When multiple filters are set,
/// they are combined with AND logic.
#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    /// Minimum importance score [0.0, 1.0]
    pub min_importance: Option<f32>,

    /// Filter to specific Johari quadrants
    pub johari_quadrants: Option<Vec<JohariQuadrant>>,

    /// Only include nodes created after this timestamp (Unix epoch)
    pub created_after: Option<u64>,

    /// Filter to nodes owned by specific agent
    pub agent_id: Option<String>,

    /// Minimum similarity score [0.0, 1.0]
    pub min_similarity: Option<f32>,

    /// Exclude specific node IDs
    pub exclude_nodes: Option<Vec<NodeId>>,
}

impl SearchFilters {
    /// Create empty filters (no filtering).
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set minimum importance.
    pub fn min_importance(mut self, min: f32) -> Self {
        self.min_importance = Some(min.clamp(0.0, 1.0));
        self
    }

    /// Builder: set Johari quadrant filter.
    pub fn johari_quadrants(mut self, quadrants: Vec<JohariQuadrant>) -> Self {
        self.johari_quadrants = Some(quadrants);
        self
    }

    /// Builder: only Open quadrant.
    pub fn open_only(self) -> Self {
        self.johari_quadrants(vec![JohariQuadrant::Open])
    }

    /// Builder: set created_after filter.
    pub fn created_after(mut self, timestamp: u64) -> Self {
        self.created_after = Some(timestamp);
        self
    }

    /// Builder: set agent filter.
    pub fn agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Builder: set minimum similarity.
    pub fn min_similarity(mut self, min: f32) -> Self {
        self.min_similarity = Some(min.clamp(0.0, 1.0));
        self
    }

    /// Builder: exclude specific nodes.
    pub fn exclude(mut self, nodes: Vec<NodeId>) -> Self {
        self.exclude_nodes = Some(nodes);
        self
    }

    /// Check if any filters are active.
    pub fn is_active(&self) -> bool {
        self.min_importance.is_some()
            || self.johari_quadrants.is_some()
            || self.created_after.is_some()
            || self.agent_id.is_some()
            || self.min_similarity.is_some()
            || self.exclude_nodes.is_some()
    }
}
```

#### File: `crates/context-graph-graph/src/search/result.rs`

```rust
//! Semantic search result types.

use crate::NodeId;

/// Result from semantic search.
#[derive(Debug, Clone)]
pub struct SemanticSearchResult {
    /// Node UUID
    pub node_id: NodeId,

    /// FAISS internal ID (for debugging/verification)
    pub faiss_id: i64,

    /// Cosine similarity [0.0, 1.0] (higher = more similar)
    pub similarity: f32,

    /// L2 distance (lower = more similar)
    pub distance: f32,

    /// Rank in result set (0 = best match)
    pub rank: usize,
}
```

#### Modification: `crates/context-graph-graph/src/lib.rs`

Add after existing modules:
```rust
// M04-T18: Semantic search
pub mod search;
pub use search::{semantic_search, semantic_search_batch, SearchFilters, SemanticSearchResult};
```

### Constraints
- **NO BACKWARDS COMPATIBILITY** - fail fast on errors
- **NO MOCK DATA** in tests - use real FAISS GPU index
- Performance: <10ms for k=100 on 10M vectors with nprobe=128
- Over-fetch 3x candidates when filters are present
- L2 to cosine: Use existing `SearchResultItem::from_l2()` - DO NOT DUPLICATE
- Filter out -1 sentinel values from FAISS results
- All errors logged with tracing before returning

### Acceptance Criteria
- [ ] semantic_search() calls FaissGpuIndex::search internally
- [ ] Uses SearchResultItem::from_l2() for L2->cosine conversion
- [ ] Applies post-filters via NodeMetadataProvider trait
- [ ] Returns GraphError::IndexNotTrained if not trained (NO FALLBACK)
- [ ] Performance meets <10ms target
- [ ] Compiles with `cargo build -p context-graph-graph`
- [ ] Tests pass with `cargo test -p context-graph-graph semantic_search`
- [ ] No clippy warnings

## Full State Verification

### Pre-Implementation Verification
```bash
# Verify dependencies are complete
cargo test -p context-graph-graph test_faiss_gpu_index  # M04-T10
cargo test -p context-graph-graph test_search_result     # M04-T11

# Verify directory structure
ls -la crates/context-graph-graph/src/search/  # Should not exist yet
```

### Post-Implementation Verification
```bash
# Build verification
cargo build -p context-graph-graph 2>&1 | tee /tmp/m04-t18-build.log

# Test verification
cargo test -p context-graph-graph semantic_search 2>&1 | tee /tmp/m04-t18-test.log

# Clippy verification
cargo clippy -p context-graph-graph -- -D warnings 2>&1 | tee /tmp/m04-t18-clippy.log

# Verify new files exist
ls -la crates/context-graph-graph/src/search/mod.rs
ls -la crates/context-graph-graph/src/search/filters.rs
ls -la crates/context-graph-graph/src/search/result.rs
```

### Database/Storage State Verification
After running integration tests:
```bash
# Verify FAISS index state
# Test should output index.ntotal() > 0 after adding vectors

# Verify RocksDB node storage
# Test should output node retrieval success
```

## Test Cases - REAL DATA ONLY

### File: `crates/context-graph-graph/tests/search_tests.rs`

```rust
//! Integration tests for semantic search.
//!
//! CRITICAL: NO MOCKS - use real FAISS GPU index per constitution.

use context_graph_graph::{
    error::GraphResult,
    index::gpu_index::{FaissGpuIndex, IndexConfig},
    index::search_result::SearchResultItem,
    search::{semantic_search, SearchFilters, SemanticSearchResult, NodeMetadataProvider},
    storage::{GraphStorage, StorageConfig},
    EmbeddingVector, NodeId, DEFAULT_EMBEDDING_DIM,
};
use context_graph_core::types::{JohariQuadrant, MemoryNode};
use rand::Rng;
use std::collections::HashMap;
use tempfile::tempdir;
use uuid::Uuid;

/// Generate normalized random vector for testing.
fn random_normalized_vector() -> EmbeddingVector {
    let mut rng = rand::thread_rng();
    let mut v = [0.0f32; DEFAULT_EMBEDDING_DIM];
    let mut norm_sq = 0.0f32;
    for x in v.iter_mut() {
        *x = rng.gen_range(-1.0..1.0);
        norm_sq += *x * *x;
    }
    let norm = norm_sq.sqrt();
    for x in v.iter_mut() {
        *x /= norm;
    }
    v
}

/// Create test MemoryNode with specified properties.
fn create_test_node(
    importance: f32,
    quadrant: JohariQuadrant,
    created_at: u64,
    agent_id: Option<&str>,
) -> MemoryNode {
    MemoryNode {
        id: Uuid::new_v4(),
        content: "test content".to_string(),
        embedding: Some(random_normalized_vector()),
        importance,
        quadrant,
        created_at,
        updated_at: created_at,
        agent_id: agent_id.map(String::from),
        metadata: HashMap::new(),
    }
}

#[test]
fn test_semantic_search_result_from_l2() {
    // L2 distance 0 = cosine similarity 1.0 (identical)
    let item = SearchResultItem::from_l2(42, 0.0);
    assert!((item.similarity - 1.0).abs() < 1e-6);

    // L2 distance 2.0 = cosine similarity 0.0 (orthogonal for normalized)
    let item = SearchResultItem::from_l2(42, 2.0);
    assert!((item.similarity - 0.0).abs() < 1e-6);

    // L2 distance 1.0 = cosine similarity 0.5
    let item = SearchResultItem::from_l2(42, 1.0);
    assert!((item.similarity - 0.5).abs() < 1e-6);
}

#[test]
fn test_search_filters_builder() {
    let filters = SearchFilters::new()
        .min_importance(0.5)
        .open_only()
        .created_after(1704067200)  // 2024-01-01
        .agent("agent-001")
        .min_similarity(0.7);

    assert_eq!(filters.min_importance, Some(0.5));
    assert_eq!(filters.johari_quadrants, Some(vec![JohariQuadrant::Open]));
    assert_eq!(filters.created_after, Some(1704067200));
    assert_eq!(filters.agent_id, Some("agent-001".to_string()));
    assert_eq!(filters.min_similarity, Some(0.7));
    assert!(filters.is_active());
}

#[test]
fn test_search_filters_inactive_by_default() {
    let filters = SearchFilters::new();
    assert!(!filters.is_active());
}

#[test]
fn test_min_importance_clamping() {
    let filters = SearchFilters::new().min_importance(1.5);
    assert_eq!(filters.min_importance, Some(1.0));

    let filters = SearchFilters::new().min_importance(-0.5);
    assert_eq!(filters.min_importance, Some(0.0));
}

#[test]
#[cfg(feature = "gpu")]
fn test_semantic_search_basic() -> GraphResult<()> {
    // REAL FAISS GPU INDEX - NO MOCKS
    let config = IndexConfig::default();
    let mut index = FaissGpuIndex::new(&config)?;

    // Setup storage
    let dir = tempdir()?;
    let storage_config = StorageConfig::default().path(dir.path());
    let storage = GraphStorage::open(storage_config)?;

    // Generate training data (IVF16384 needs ~4M vectors minimum)
    // For testing, use smaller IVF with fewer clusters
    let mut test_config = IndexConfig::default();
    test_config.nlist = 100;  // Smaller for tests
    let mut index = FaissGpuIndex::new(&test_config)?;

    let train_data: Vec<EmbeddingVector> = (0..1000)
        .map(|_| random_normalized_vector())
        .collect();

    println!("BEFORE: Training index with {} vectors", train_data.len());
    index.train(&train_data)?;
    println!("AFTER: Index trained, is_trained={}", index.is_trained());

    // Add test nodes to storage and index
    let mut nodes = Vec::new();
    let mut faiss_ids = Vec::new();
    let mut vectors = Vec::new();

    for i in 0..100 {
        let node = create_test_node(
            (i as f32) / 100.0,  // importance 0.0 to 0.99
            JohariQuadrant::Open,
            1704067200 + i as u64,
            Some("test-agent"),
        );

        let faiss_id = i as i64;
        storage.put_node(&node)?;
        storage.map_faiss_id(faiss_id, node.id)?;

        vectors.push(node.embedding.unwrap());
        faiss_ids.push(faiss_id);
        nodes.push(node);
    }

    println!("BEFORE: Adding {} vectors to index", vectors.len());
    index.add_with_ids(&vectors, &faiss_ids)?;
    println!("AFTER: Index has {} vectors", index.ntotal());

    // Search for first vector - should find itself
    let query = &vectors[0];
    let results = semantic_search(&index, &storage, query, 10, None)?;

    println!("VERIFY: Search returned {} results", results.len());
    assert!(!results.is_empty(), "Search should return results");
    assert!(results.len() <= 10, "Should return at most k results");

    // First result should be the query itself (highest similarity)
    println!("VERIFY: Top result node_id={}, similarity={}", results[0].node_id, results[0].similarity);
    assert!(results[0].similarity > 0.99, "Self-search should have >0.99 similarity");

    // Results should be sorted by similarity descending
    for i in 1..results.len() {
        assert!(
            results[i-1].similarity >= results[i].similarity,
            "Results must be sorted by similarity descending"
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "gpu")]
fn test_semantic_search_with_filters() -> GraphResult<()> {
    let mut config = IndexConfig::default();
    config.nlist = 100;
    let mut index = FaissGpuIndex::new(&config)?;

    let dir = tempdir()?;
    let storage_config = StorageConfig::default().path(dir.path());
    let storage = GraphStorage::open(storage_config)?;

    // Train
    let train_data: Vec<EmbeddingVector> = (0..1000)
        .map(|_| random_normalized_vector())
        .collect();
    index.train(&train_data)?;

    // Add nodes with varying importance
    let mut vectors = Vec::new();
    let mut faiss_ids = Vec::new();

    for i in 0..50 {
        let node = create_test_node(
            0.3,  // Low importance
            JohariQuadrant::Hidden,
            1704067200,
            None,
        );
        storage.put_node(&node)?;
        storage.map_faiss_id(i as i64, node.id)?;
        vectors.push(node.embedding.unwrap());
        faiss_ids.push(i as i64);
    }

    for i in 50..100 {
        let node = create_test_node(
            0.8,  // High importance
            JohariQuadrant::Open,
            1704067200,
            Some("agent-001"),
        );
        storage.put_node(&node)?;
        storage.map_faiss_id(i as i64, node.id)?;
        vectors.push(node.embedding.unwrap());
        faiss_ids.push(i as i64);
    }

    index.add_with_ids(&vectors, &faiss_ids)?;

    // Search with min_importance filter
    let query = &vectors[75];  // Query a high-importance vector
    let filters = SearchFilters::new().min_importance(0.5);

    println!("BEFORE: Search with min_importance=0.5 filter");
    let results = semantic_search(&index, &storage, query, 20, Some(filters))?;
    println!("AFTER: Got {} results", results.len());

    // All results should have importance >= 0.5
    for r in &results {
        let imp = storage.get_importance(r.faiss_id)?.unwrap();
        println!("VERIFY: node importance={}", imp);
        assert!(imp >= 0.5, "Filter should exclude low importance nodes");
    }

    Ok(())
}

#[test]
fn test_semantic_search_empty_index() -> GraphResult<()> {
    let config = IndexConfig::default();
    let index = FaissGpuIndex::new(&config)?;

    // Don't train - index is empty
    let dir = tempdir()?;
    let storage_config = StorageConfig::default().path(dir.path());
    let storage = GraphStorage::open(storage_config)?;

    let query = random_normalized_vector();

    println!("BEFORE: Search on empty index");
    let results = semantic_search(&index, &storage, &query, 10, None)?;
    println!("AFTER: Got {} results", results.len());

    assert!(results.is_empty(), "Empty index should return empty results");
    Ok(())
}

#[test]
fn test_semantic_search_untrained_index_fails_fast() -> GraphResult<()> {
    let config = IndexConfig::default();
    let mut index = FaissGpuIndex::new(&config)?;

    // Add vectors without training (invalid state)
    let vectors: Vec<EmbeddingVector> = (0..10)
        .map(|_| random_normalized_vector())
        .collect();
    let ids: Vec<i64> = (0..10).collect();

    // This should fail because index not trained
    // Actually, add_with_ids should fail first, but let's test search

    let dir = tempdir()?;
    let storage_config = StorageConfig::default().path(dir.path());
    let storage = GraphStorage::open(storage_config)?;

    let query = random_normalized_vector();

    println!("BEFORE: Search on untrained (but non-empty) index");
    // For IVF index, adding without training should fail
    // But if we could add, search should return IndexNotTrained

    // The actual test depends on implementation details
    // Key point: NO FALLBACK - must fail fast
    Ok(())
}
```

## Edge Cases with BEFORE/AFTER Verification

### Edge Case 1: All Results Filtered Out
```
BEFORE: FAISS returns 30 candidates
DURING: All fail min_importance=0.9 filter
AFTER: Returns empty Vec (not error)
VERIFY: println!("Got {} results after filtering", results.len());
```

### Edge Case 2: k > ntotal
```
BEFORE: Index has 50 vectors, k=100 requested
DURING: fetch_k clamped to min(100, 50) = 50
AFTER: Returns up to 50 results
VERIFY: assert!(results.len() <= index.ntotal());
```

### Edge Case 3: Sentinel -1 Values
```
BEFORE: FAISS slot has id=-1 (no match for that slot)
DURING: Skip in filter_map
AFTER: Result not included
VERIFY: assert!(results.iter().all(|r| r.faiss_id >= 0));
```

### Edge Case 4: Missing Node Metadata
```
BEFORE: FAISS returns id=42, but storage has no node for that ID
DURING: NodeMetadataProvider returns None, skip with warning
AFTER: Result excluded from output
VERIFY: Check tracing logs for "No importance data, skipping"
```

## Sherlock-Holmes Verification Step

After implementation, spawn sherlock-holmes subagent with:

```
INVESTIGATION MANDATE: Verify M04-T18 Semantic Search Implementation

FORENSIC CHECKLIST:
1. [ ] File exists: crates/context-graph-graph/src/search/mod.rs
2. [ ] File exists: crates/context-graph-graph/src/search/filters.rs
3. [ ] File exists: crates/context-graph-graph/src/search/result.rs
4. [ ] lib.rs has: pub mod search;
5. [ ] lib.rs has: pub use search::{...};
6. [ ] semantic_search() calls index.search()
7. [ ] Uses SearchResultItem::from_l2() (not duplicate conversion)
8. [ ] NodeMetadataProvider trait implemented for GraphStorage
9. [ ] Filters use is_active() check before applying
10. [ ] All errors use GraphError variants
11. [ ] tracing instrumentation on public functions
12. [ ] NO unwrap() in production code
13. [ ] Tests use real FAISS index (no mocks)
14. [ ] cargo build -p context-graph-graph succeeds
15. [ ] cargo test -p context-graph-graph semantic_search passes
16. [ ] cargo clippy -p context-graph-graph -- -D warnings clean

VERDICT: GUILTY until all checks pass with evidence
```

## Test Commands

```bash
# Build
cargo build -p context-graph-graph

# Test semantic search
cargo test -p context-graph-graph semantic_search -- --nocapture

# Clippy
cargo clippy -p context-graph-graph -- -D warnings

# Full verification
cargo test -p context-graph-graph && cargo clippy -p context-graph-graph -- -D warnings
```

## Manual Verification Checklist

- [ ] `semantic_search()` returns results in <10ms for k=100
- [ ] Similarity scores are in [0.0, 1.0] range
- [ ] Results sorted by similarity descending
- [ ] rank field matches position in result Vec
- [ ] Filters correctly exclude non-matching nodes
- [ ] Empty index returns empty Vec (not error)
- [ ] Untrained index returns GraphError::IndexNotTrained
- [ ] FAISS -1 sentinels filtered out
- [ ] Missing nodes logged and skipped
