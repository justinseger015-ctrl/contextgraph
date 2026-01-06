# TASK-S001: Update MCP Memory Handlers for TeleologicalFingerprint

```yaml
metadata:
  id: "TASK-S001"
  title: "Update MCP Memory Handlers for TeleologicalFingerprint"
  layer: "surface"
  priority: "P0"
  estimated_hours: 8
  created: "2026-01-04"
  updated: "2026-01-05"
  status: "pending"
  dependencies:
    - "TASK-F001"  # SemanticFingerprint (COMPLETE - exists)
    - "TASK-F002"  # TeleologicalFingerprint (COMPLETE - exists)
    - "TASK-F003"  # JohariFingerprint (COMPLETE - exists)
    - "TASK-F008"  # TeleologicalMemoryStore trait (COMPLETE - exists)
  traces_to:
    - "FR-101"  # 13-Embedding Array Storage
    - "FR-301"  # Primary Storage
    - constitution.yaml section: embeddings, storage, mcp
```

## CRITICAL CORRECTIONS FROM CODEBASE AUDIT (2026-01-05)

The original task document contained multiple errors. This section documents corrections:

| Original Claim | Actual State | Impact |
|----------------|--------------|--------|
| 12 embedders | **13 embedders** (E1-E13 including SPLADE) | All schemas wrong |
| Files to delete: `fused_memory.rs`, `vector_store.rs` | **DO NOT EXIST** | No deletions needed |
| `router.rs` exists | **NO router.rs** - dispatch in `handlers/core.rs` | Wrong file references |
| `error.rs` exists | **NO error.rs** - uses `protocol::error_codes` module | Wrong file references |
| Purpose vector 12D | **13D** per constitution.yaml | Schema wrong |
| E7 dimension 1536 | **256** per `fingerprint/mod.rs:E7_DIM` | Schema wrong |

## Problem Statement

Replace legacy MCP memory handlers (using single-vector `MemoryNode`) with new handlers using 13-embedding `TeleologicalFingerprint` architecture.

## Current State Analysis

### Files That Currently Exist

**MCP Handlers** (`crates/context-graph-mcp/src/handlers/`):
- `mod.rs` - Module exports, re-exports `Handlers` from `core`
- `core.rs` - **LEGACY**: Uses `Arc<dyn MemoryStore>` (line 13), dispatches requests
- `memory.rs` - **LEGACY**: Uses `MemoryNode::new(content, embedding_output.vector)` single-vector
- `tools.rs` - **LEGACY**: `inject_context`, `store_memory` use `MemoryNode`
- `lifecycle.rs` - MCP lifecycle (initialize, shutdown) - minimal changes needed
- `system.rs` - System status handlers
- `utl.rs` - UTL computation handlers
- `tests/` - Test modules

**Core Types** (`crates/context-graph-core/src/`):
- `traits/teleological_memory_store.rs` - `TeleologicalMemoryStore` trait ✓
- `traits/mod.rs` - Re-exports trait ✓
- `types/fingerprint/mod.rs` - Re-exports all fingerprint types ✓
- `types/fingerprint/semantic.rs` - `SemanticFingerprint` 13 embeddings ✓
- `types/fingerprint/teleological.rs` - `TeleologicalFingerprint` complete ✓
- `types/fingerprint/purpose.rs` - `PurposeVector` 13D ✓
- `types/fingerprint/johari.rs` - `JohariFingerprint` per-embedder ✓
- `stubs/teleological_store_stub.rs` - `InMemoryTeleologicalStore` for tests ✓

### Embedding Dimensions (AUTHORITATIVE from constitution.yaml + codebase)

| Embedder | Name | Dimension | Type | Source |
|----------|------|-----------|------|--------|
| E1 | Semantic | 1024 | Dense | `E1_DIM` |
| E2 | Temporal_Recent | 512 | Dense | `E2_DIM` |
| E3 | Temporal_Periodic | 512 | Dense | `E3_DIM` |
| E4 | Temporal_Positional | 512 | Dense | `E4_DIM` |
| E5 | Causal | 768 | Dense | `E5_DIM` |
| E6 | Sparse | ~30K vocab, 5% active | Sparse | `E6_SPARSE_VOCAB` |
| E7 | Code | 256 | Dense | `E7_DIM` |
| E8 | Graph | 384 | Dense | `E8_DIM` |
| E9 | HDC | 10000 | Dense (binary in storage) | `E9_DIM` |
| E10 | Multimodal | 768 | Dense | `E10_DIM` |
| E11 | Entity | 384 | Dense | `E11_DIM` |
| E12 | LateInteraction | 128 × N tokens | Token-level | `E12_TOKEN_DIM` |
| E13 | SPLADE | ~30K vocab | Sparse | `E13_SPLADE_VOCAB` |

## Implementation Specification

### Step 1: Modify handlers/core.rs

**Current** (lines 7-16):
```rust
use context_graph_core::traits::{EmbeddingProvider, MemoryStore, UtlProcessor};

pub struct Handlers {
    pub(super) memory_store: Arc<dyn MemoryStore>,
    pub(super) utl_processor: Arc<dyn UtlProcessor>,
    pub(super) embedding_provider: Arc<dyn EmbeddingProvider>,
}
```

**Target**:
```rust
use context_graph_core::traits::{
    EmbeddingProvider, TeleologicalMemoryStore, UtlProcessor,
    MultiArrayEmbeddingProvider, TeleologicalSearchOptions,
};
use context_graph_core::types::fingerprint::{
    SemanticFingerprint, TeleologicalFingerprint, PurposeVector,
    JohariFingerprint, SparseVector,
};

pub struct Handlers {
    pub(super) teleological_store: Arc<dyn TeleologicalMemoryStore>,
    pub(super) utl_processor: Arc<dyn UtlProcessor>,
    pub(super) embedding_provider: Arc<dyn EmbeddingProvider>,
    pub(super) multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
}

impl Handlers {
    pub fn new(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
    ) -> Self {
        Self {
            teleological_store,
            utl_processor,
            embedding_provider,
            multi_array_provider,
        }
    }
}
```

### Step 2: Rewrite handlers/memory.rs

**Replace entire file**. Key handlers:

```rust
//! Teleological memory operation handlers.
//!
//! Operates on TeleologicalFingerprint with 13-embedding arrays.
//! NO BACKWARDS COMPATIBILITY with legacy single-vector MemoryNode.

use serde_json::json;
use tracing::{debug, error, warn};
use uuid::Uuid;

use context_graph_core::traits::{
    TeleologicalMemoryStore, TeleologicalSearchOptions, TeleologicalSearchResult,
};
use context_graph_core::types::fingerprint::{
    SemanticFingerprint, TeleologicalFingerprint, PurposeVector, JohariFingerprint,
    SparseVector, NUM_EMBEDDERS, E1_DIM, E2_DIM, E3_DIM, E4_DIM, E5_DIM,
    E7_DIM, E8_DIM, E9_DIM, E10_DIM, E11_DIM, E12_TOKEN_DIM,
    E6_SPARSE_VOCAB, E13_SPLADE_VOCAB,
};

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};
use super::Handlers;

/// Expected dimensions for validation [E1..E13]
const EXPECTED_DIMS: [usize; 13] = [
    E1_DIM,   // 1024
    E2_DIM,   // 512
    E3_DIM,   // 512
    E4_DIM,   // 512
    E5_DIM,   // 768
    0,        // E6 sparse - variable
    E7_DIM,   // 256
    E8_DIM,   // 384
    E9_DIM,   // 10000
    E10_DIM,  // 768
    E11_DIM,  // 384
    0,        // E12 token-level - variable
    0,        // E13 sparse - variable
];

impl Handlers {
    /// Handle memory/store request with TeleologicalFingerprint.
    pub(super) async fn handle_memory_store(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let content = match params.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'content' parameter",
                );
            }
        };

        // Generate 13-embedding SemanticFingerprint
        let semantic_fp = match self.multi_array_provider.embed_all(&content).await {
            Ok(fp) => fp,
            Err(e) => {
                error!(error = %e, "Multi-array embedding generation FAILED");
                return JsonRpcResponse::error(
                    id,
                    error_codes::MULTI_ARRAY_EMBED_FAILED,
                    format!("13-embedding generation failed: {}", e),
                );
            }
        };

        // Compute purpose vector (13D alignment to North Star)
        let purpose_vector = self.compute_purpose_vector(&semantic_fp);

        // Compute Johari fingerprint (per-embedder awareness)
        let johari_fp = self.compute_johari_fingerprint(&semantic_fp);

        // Create content hash
        let content_hash = self.compute_content_hash(&content);

        // Build TeleologicalFingerprint
        let fingerprint = TeleologicalFingerprint::new(
            semantic_fp,
            purpose_vector.clone(),
            johari_fp.clone(),
            content_hash,
        );

        let fp_id = fingerprint.id;
        let theta = fingerprint.theta_to_north_star;

        // Store via TeleologicalMemoryStore
        match self.teleological_store.store(fingerprint).await {
            Ok(stored_id) => {
                debug!(id = %stored_id, "TeleologicalFingerprint stored");

                // Build response
                let response_data = json!({
                    "id": stored_id.to_string(),
                    "storage_size_bytes": self.teleological_store.storage_size_bytes(),
                    "theta_to_north_star": theta,
                    "dominant_embedder": johari_fp.dominant_quadrant_index(),
                    "johari_dominant": format!("{:?}", johari_fp.aggregate_quadrant()),
                    "created_at": chrono::Utc::now().to_rfc3339(),
                });

                JsonRpcResponse::success(id, response_data)
            }
            Err(e) => {
                error!(error = %e, "Storage FAILED for TeleologicalFingerprint");
                JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Storage failed: {}", e),
                )
            }
        }
    }

    /// Handle memory/retrieve request - returns full TeleologicalFingerprint.
    pub(super) async fn handle_memory_retrieve(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let node_id_str = match params.get("nodeId").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'nodeId' parameter",
                );
            }
        };

        let node_id = match Uuid::parse_str(node_id_str) {
            Ok(u) => u,
            Err(_) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid UUID format: {}", node_id_str),
                );
            }
        };

        match self.teleological_store.retrieve(node_id).await {
            Ok(Some(fp)) => {
                // Serialize full TeleologicalFingerprint
                let response_data = json!({
                    "id": fp.id.to_string(),
                    "fingerprint": {
                        "semantic_fingerprint": self.serialize_semantic_fingerprint(&fp.semantic_fingerprint),
                        "purpose_vector": fp.purpose_vector.alignments(),
                        "johari_fingerprint": {
                            "quadrants": fp.johari_fingerprint.quadrants_as_strings(),
                            "confidences": fp.johari_fingerprint.confidences(),
                        },
                        "theta_to_north_star": fp.theta_to_north_star,
                        "content_hash": hex::encode(fp.content_hash),
                    },
                    "created_at": fp.created_at.to_rfc3339(),
                    "last_accessed": fp.last_accessed.to_rfc3339(),
                });

                JsonRpcResponse::success(id, response_data)
            }
            Ok(None) => {
                JsonRpcResponse::error(
                    id,
                    error_codes::NODE_NOT_FOUND,
                    format!("Memory not found: {}", node_id),
                )
            }
            Err(e) => {
                error!(error = %e, id = %node_id, "Retrieve FAILED");
                JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Retrieve failed: {}", e),
                )
            }
        }
    }

    /// Handle memory/search - uses TeleologicalSearchOptions.
    pub(super) async fn handle_memory_search(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let query = match params.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'query' parameter",
                );
            }
        };

        let top_k = params.get("topK").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let min_alignment = params.get("minAlignment").and_then(|v| v.as_f64()).map(|v| v as f32);

        let mut options = TeleologicalSearchOptions::quick(top_k);
        if let Some(threshold) = min_alignment {
            options = options.with_min_alignment(threshold);
        }

        match self.teleological_store.search_text(query, options).await {
            Ok(results) => {
                let results_json: Vec<_> = results
                    .iter()
                    .map(|r| {
                        json!({
                            "id": r.fingerprint.id.to_string(),
                            "similarity": r.similarity,
                            "purpose_alignment": r.purpose_alignment,
                            "embedder_scores": r.embedder_scores,
                            "dominant_embedder": r.dominant_embedder(),
                            "theta_to_north_star": r.fingerprint.theta_to_north_star,
                        })
                    })
                    .collect();

                JsonRpcResponse::success(id, json!({
                    "results": results_json,
                    "count": results_json.len(),
                }))
            }
            Err(e) => {
                error!(error = %e, "Search FAILED");
                JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Search failed: {}", e),
                )
            }
        }
    }

    /// Handle memory/delete request.
    pub(super) async fn handle_memory_delete(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let node_id_str = match params.get("nodeId").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'nodeId' parameter",
                );
            }
        };

        let node_id = match Uuid::parse_str(node_id_str) {
            Ok(u) => u,
            Err(_) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Invalid UUID format",
                );
            }
        };

        let soft = params.get("soft").and_then(|v| v.as_bool()).unwrap_or(true);

        match self.teleological_store.delete(node_id, soft).await {
            Ok(deleted) => {
                JsonRpcResponse::success(id, json!({
                    "deleted": deleted,
                    "soft": soft,
                    "id": node_id.to_string(),
                }))
            }
            Err(e) => {
                error!(error = %e, id = %node_id, "Delete FAILED");
                JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Delete failed: {}", e),
                )
            }
        }
    }

    // Helper methods
    fn compute_purpose_vector(&self, _fp: &SemanticFingerprint) -> PurposeVector {
        // TODO: Implement actual alignment computation
        PurposeVector::default()
    }

    fn compute_johari_fingerprint(&self, _fp: &SemanticFingerprint) -> JohariFingerprint {
        // TODO: Implement actual Johari computation
        JohariFingerprint::zeroed()
    }

    fn compute_content_hash(&self, content: &str) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hasher.finalize().into()
    }

    fn serialize_semantic_fingerprint(&self, fp: &SemanticFingerprint) -> serde_json::Value {
        // Serialize 13 embeddings with their dimensions
        json!({
            "e1_semantic": fp.get_embedding_vec(0),
            "e2_temporal_recent": fp.get_embedding_vec(1),
            "e3_temporal_periodic": fp.get_embedding_vec(2),
            "e4_temporal_positional": fp.get_embedding_vec(3),
            "e5_causal": fp.get_embedding_vec(4),
            "e6_sparse": fp.get_sparse_vector(5),
            "e7_code": fp.get_embedding_vec(6),
            "e8_graph": fp.get_embedding_vec(7),
            "e9_hdc": fp.get_embedding_vec(8),
            "e10_multimodal": fp.get_embedding_vec(9),
            "e11_entity": fp.get_embedding_vec(10),
            "e12_late_interaction": fp.get_token_embeddings(11),
            "e13_splade": fp.get_sparse_vector(12),
        })
    }
}
```

### Step 3: Add Error Codes

**File**: `crates/context-graph-mcp/src/protocol/error_codes.rs`

Add these constants:

```rust
// Teleological Memory Errors (-32100 to -32199)
pub const DIMENSION_MISMATCH: i32 = -32100;
pub const MISSING_EMBEDDER: i32 = -32101;
pub const INVALID_SPARSE_INDEX: i32 = -32102;
pub const FINGERPRINT_VALIDATION: i32 = -32103;
pub const PURPOSE_VECTOR_INVALID: i32 = -32104;
pub const JOHARI_COMPUTE_FAILED: i32 = -32105;
pub const MULTI_ARRAY_EMBED_FAILED: i32 = -32106;
```

### Step 4: Update tools.rs

Update `call_inject_context` and `call_store_memory` to use TeleologicalFingerprint instead of MemoryNode.

## Testing Requirements

### NO MOCK DATA

All tests MUST use:
- `InMemoryTeleologicalStore` from `crates/context-graph-core/src/stubs/teleological_store_stub.rs`
- Real `TeleologicalFingerprint` structures with correct dimensions
- `SemanticFingerprint::zeroed()` for deterministic tests (NOT random data)

### Test File Location

`crates/context-graph-mcp/src/handlers/tests/memory.rs` (already exists, must be updated)

### Required Test Cases

```rust
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use context_graph_core::stubs::InMemoryTeleologicalStore;
    use context_graph_core::types::fingerprint::{
        SemanticFingerprint, TeleologicalFingerprint, PurposeVector, JohariFingerprint,
    };

    fn create_test_fingerprint() -> TeleologicalFingerprint {
        let semantic = SemanticFingerprint::zeroed();
        let purpose = PurposeVector::default();
        let johari = JohariFingerprint::zeroed();
        let content_hash = [0u8; 32];
        TeleologicalFingerprint::new(semantic, purpose, johari, content_hash)
    }

    #[tokio::test]
    async fn test_store_creates_fingerprint_in_store() {
        let store = Arc::new(InMemoryTeleologicalStore::new());
        let handlers = create_handlers(store.clone());

        let before_count = store.count().await.unwrap();
        assert_eq!(before_count, 0, "BEFORE: Store should be empty");

        let request = create_store_request("Test content");
        let response = handlers.handle_memory_store(Some(1.into()), Some(request)).await;

        assert!(response.is_success(), "Store should succeed");

        let after_count = store.count().await.unwrap();
        assert_eq!(after_count, 1, "AFTER: Store should have 1 entry");

        // Extract ID and verify in store
        let id = extract_id_from_response(&response);
        let retrieved = store.retrieve(id).await.unwrap();
        assert!(retrieved.is_some(), "SOURCE OF TRUTH: Fingerprint must exist in store");
    }

    #[tokio::test]
    async fn test_retrieve_returns_all_13_embeddings() {
        let store = Arc::new(InMemoryTeleologicalStore::new());
        let fp = create_test_fingerprint();
        let id = store.store(fp).await.unwrap();

        let handlers = create_handlers(store.clone());
        let request = json!({ "nodeId": id.to_string() });
        let response = handlers.handle_memory_retrieve(Some(1.into()), Some(request)).await;

        assert!(response.is_success());

        // Verify all 13 embeddings in response
        let result = response.result.unwrap();
        let fingerprint = result.get("fingerprint").unwrap();
        let semantic = fingerprint.get("semantic_fingerprint").unwrap();

        assert!(semantic.get("e1_semantic").is_some());
        assert!(semantic.get("e13_splade").is_some());
    }

    #[tokio::test]
    async fn test_search_returns_teleological_results() {
        let store = Arc::new(InMemoryTeleologicalStore::new());
        let fp = create_test_fingerprint();
        store.store(fp).await.unwrap();

        let handlers = create_handlers(store.clone());
        let request = json!({ "query": "test", "topK": 5 });
        let response = handlers.handle_memory_search(Some(1.into()), Some(request)).await;

        assert!(response.is_success());

        let result = response.result.unwrap();
        let results = result.get("results").unwrap().as_array().unwrap();

        // Verify teleological fields present
        for r in results {
            assert!(r.get("similarity").is_some());
            assert!(r.get("purpose_alignment").is_some());
            assert!(r.get("embedder_scores").is_some());
            assert!(r.get("dominant_embedder").is_some());
        }
    }

    #[tokio::test]
    async fn test_delete_removes_from_store() {
        let store = Arc::new(InMemoryTeleologicalStore::new());
        let fp = create_test_fingerprint();
        let id = store.store(fp).await.unwrap();

        assert!(store.retrieve(id).await.unwrap().is_some(), "BEFORE: Exists");

        let handlers = create_handlers(store.clone());
        let request = json!({ "nodeId": id.to_string(), "soft": false });
        let response = handlers.handle_memory_delete(Some(1.into()), Some(request)).await;

        assert!(response.is_success());
        assert!(store.retrieve(id).await.unwrap().is_none(), "AFTER: Deleted");
    }
}
```

## Full State Verification Protocol

### Source of Truth

- **Development**: `InMemoryTeleologicalStore` internal HashMap
- **Production**: RocksDB column families per `constitution.yaml:storage`

### Execute & Inspect After Every Operation

```rust
// PATTERN: Every store operation must be verified
let store_response = handlers.handle_memory_store(id, params).await;
let stored_id = extract_id(&store_response);

// IMMEDIATELY verify in source of truth
let exists = store.retrieve(stored_id).await.unwrap().is_some();
assert!(exists, "VERIFICATION FAILED: Memory not in store after successful response");
```

### Edge Case Audit (MANDATORY)

Execute and log before/after for each:

**Case 1: Empty Content**
```
BEFORE: store.count() = 0
ACTION: handle_memory_store with content=""
RESPONSE: Error (INVALID_PARAMS)
AFTER: store.count() = 0
RESULT: PASS - Empty content rejected
```

**Case 2: Maximum Valid Sparse Index (30521)**
```
BEFORE: store.count() = 0
ACTION: handle_memory_store with E13 sparse index = 30521
RESPONSE: Success
AFTER: store.count() = 1
AFTER: retrieved.semantic_fingerprint.e13.max_index() = 30521
RESULT: PASS - Boundary value accepted
```

**Case 3: Invalid Sparse Index (40000)**
```
BEFORE: store.count() = 0
ACTION: handle_memory_store with E13 sparse index = 40000
RESPONSE: Error (INVALID_SPARSE_INDEX)
AFTER: store.count() = 0
RESULT: PASS - Invalid index rejected
```

### Evidence of Success Log Format

Every test MUST produce:

```
=== TASK-S001 Verification: [test_name] ===
TRIGGER: [operation] called with [params summary]
PROCESS: [what happened internally]
SOURCE_OF_TRUTH: [store state verification]
OUTCOME: [success/failure with data]
EDGE_CASE_VERIFIED: [yes/no]
RESULT: [PASS/FAIL]
===
```

## Definition of Done

### Implementation Checklist

- [ ] `handlers/core.rs` uses `Arc<dyn TeleologicalMemoryStore>`
- [ ] `handlers/core.rs` has `multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>`
- [ ] `handlers/memory.rs` rewritten with all handlers using TeleologicalFingerprint
- [ ] `handlers/tools.rs` updated for `inject_context`, `store_memory`
- [ ] `protocol/error_codes.rs` has teleological error codes
- [ ] All 13 embedders validated (dimensions per EXPECTED_DIMS constant)
- [ ] Sparse indices validated (0-30521 for E6, E13)
- [ ] Search returns `TeleologicalSearchResult` with all scores
- [ ] All legacy `MemoryNode` references removed from handlers/
- [ ] Tests use `InMemoryTeleologicalStore`, NOT mocks

### Verification Commands

```bash
# Run handler tests with logging
RUST_LOG=debug cargo test -p context-graph-mcp handlers::memory -- --nocapture

# Verify no MemoryNode in handlers
rg "MemoryNode" crates/context-graph-mcp/src/handlers/ && echo "FAIL: Legacy MemoryNode found" || echo "PASS: No MemoryNode"

# Verify TeleologicalMemoryStore is used
rg "TeleologicalMemoryStore" crates/context-graph-mcp/src/handlers/ | head -5

# Verify 13 embedders constant
cargo test -p context-graph-core fingerprint::tests::test_num_embedders
```

## Files Summary

### Must Modify

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/core.rs` | `MemoryStore` → `TeleologicalMemoryStore`, add `multi_array_provider` |
| `crates/context-graph-mcp/src/handlers/memory.rs` | Complete rewrite for TeleologicalFingerprint |
| `crates/context-graph-mcp/src/handlers/tools.rs` | Update `inject_context`, `store_memory` |
| `crates/context-graph-mcp/src/protocol/error_codes.rs` | Add teleological error codes |
| `crates/context-graph-mcp/src/handlers/tests/memory.rs` | Update tests for TeleologicalFingerprint |

### DO NOT Create (Already Exist)

- `crates/context-graph-core/src/traits/teleological_memory_store.rs`
- `crates/context-graph-core/src/types/fingerprint/*.rs`
- `crates/context-graph-core/src/stubs/teleological_store_stub.rs`

### DO NOT Delete (Don't Exist)

- `fused_memory.rs` - DOES NOT EXIST
- `vector_store.rs` - DOES NOT EXIST
- `router.rs` - DOES NOT EXIST

## Traceability

| Requirement | Source | Implementation |
|-------------|--------|----------------|
| 13-embedding storage | constitution.yaml:embeddings:models | SemanticFingerprint |
| TeleologicalMemoryStore | TASK-F008 | handlers/core.rs dependency |
| Fail fast | constitution.yaml:forbidden:AP-007 | Error codes, no fallbacks |
| Cognitive pulse | constitution.yaml:mcp:pulse | Response field |
| NO backwards compat | This task | Complete API replacement |

---

*Original: 2026-01-04*
*Audited: 2026-01-05*
*Layer: Surface*
*Priority: P0*
