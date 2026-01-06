# TASK-EMB-024: Update MCP Handlers

<task_spec id="TASK-EMB-024" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-024 |
| **Title** | Update MCP Handlers to Use Real Embeddings |
| **Status** | ready |
| **Layer** | surface |
| **Sequence** | 24 |
| **Implements** | All REQs (REQ-EMB-001 through REQ-EMB-007) |
| **Depends On** | TASK-EMB-021 (SparseModel integration), TASK-EMB-023 (Multi-space search) |
| **Estimated Complexity** | medium |
| **Created** | 2026-01-06 |
| **Constitution Reference** | v4.0.0 |

---

## Context

The MCP handlers need to be updated to use the real embedding pipeline instead of any stub references. This includes:

- Multi-embedding search handlers
- Purpose handlers
- Johari handlers
- Meta-UTL handlers

**Constitution Alignment:**
- `forbidden.AP-007`: "Stub data in prod → use tests/fixtures/"
- All handlers must use real GPU inference
- All dimensions must match Constitution specifications

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Multi-embedding search | `crates/context-graph-mcp/src/handlers/multi_embedding_search.rs` |
| Purpose handlers | `crates/context-graph-mcp/src/handlers/purpose.rs` |
| Johari handlers | `crates/context-graph-mcp/src/handlers/johari.rs` |
| Meta-UTL handlers | `crates/context-graph-mcp/src/handlers/meta_utl.rs` |
| Handler registry | `crates/context-graph-mcp/src/handlers/mod.rs` |
| MultiSpaceSearch | `crates/context-graph-embeddings/src/storage/multi_space.rs` |
| SparseModel | `crates/context-graph-embeddings/src/models/pretrained/sparse/model.rs` |

---

## Prerequisites

- [ ] TASK-EMB-021 completed (SparseModel uses projection)
- [ ] TASK-EMB-023 completed (multi-space search works)
- [ ] All embedding models loaded and functional
- [ ] Storage backend operational

---

## Scope

### In Scope

- Update multi_embedding_search handlers to use real HNSW search
- Update purpose handlers to compute from real 13-embedding fingerprint
- Update johari handlers to compute from real ΔS/ΔC values
- Update meta_utl handlers to use real UTL alignment
- Remove ALL stub/mock data references from production paths
- Ensure output dimensions match Constitution

### Out of Scope

- Adding new handlers (separate task)
- Performance optimization beyond basic correctness
- MCP protocol changes
- New tool definitions

---

## Definition of Done

### Handler Update Pattern

**BEFORE (with stub):**
```rust
// ❌ FORBIDDEN
let embeddings = create_stub_embeddings();
let fingerprint = stub_fingerprint();
let johari = stub_johari_quadrants();
```

**AFTER (real):**
```rust
// ✓ CORRECT
let embeddings = embedding_service.embed_all(&input).await?;
let fingerprint = storage.retrieve(id)?;
let johari = compute_johari_quadrants(&fingerprint)?;
```

### Multi-Embedding Search Handler Updates

```rust
// File: crates/context-graph-mcp/src/handlers/multi_embedding_search.rs

impl MultiEmbeddingSearchHandler {
    /// Search across multiple embedding spaces with RRF fusion.
    ///
    /// # Changes from Stub
    /// - Uses real HNSW indexes via MultiSpaceSearch
    /// - Returns real similarity scores
    /// - Supports embedder selection in query
    pub async fn handle_search(
        &self,
        request: MultiEmbeddingSearchRequest,
    ) -> Result<MultiEmbeddingSearchResponse, McpError> {
        // Generate query embeddings using real models
        let query_embeddings = self.embed_query(&request.query).await?;

        // Select which embedders to search
        let embedder_indices = request.embedders.unwrap_or_else(|| (0..13).collect());

        // Build query map for selected embedders
        let queries: HashMap<u8, Vec<f32>> = embedder_indices
            .iter()
            .filter_map(|&idx| {
                query_embeddings.get(idx as usize).map(|v| (idx, v.clone()))
            })
            .collect();

        // Search using real HNSW indexes
        let results = self.multi_space_search.search_multi_space(
            &queries,
            request.weights.as_ref(),
            request.k_per_space.unwrap_or(100),
            request.final_k.unwrap_or(10),
        )?;

        // Convert to response format
        Ok(MultiEmbeddingSearchResponse {
            results: results.into_iter().map(Into::into).collect(),
            query_embedders: embedder_indices,
            total_candidates_searched: queries.len() * request.k_per_space.unwrap_or(100),
        })
    }
}
```

### Purpose Handler Updates

```rust
// File: crates/context-graph-mcp/src/handlers/purpose.rs

impl PurposeHandler {
    /// Compute purpose vector from real 13-embedding fingerprint.
    ///
    /// # Changes from Stub
    /// - Uses real embeddings from storage
    /// - Computes real alignment to North Star
    /// - No hardcoded values
    pub async fn handle_get_purpose(
        &self,
        request: GetPurposeRequest,
    ) -> Result<GetPurposeResponse, McpError> {
        // Retrieve real fingerprint from storage
        let fingerprint = self.storage.retrieve(request.id)?
            .ok_or(McpError::not_found("Fingerprint not found"))?;

        // Purpose vector is pre-computed and stored
        let purpose_vector = fingerprint.purpose_vector;

        // Compute alignment to North Star (if not cached)
        let north_star_alignment = fingerprint.theta_to_north_star;

        Ok(GetPurposeResponse {
            id: request.id,
            purpose_vector,
            north_star_alignment,
            dominant_embedder: fingerprint.dominant_embedder,
        })
    }

    /// Compute alignment between two fingerprints.
    pub async fn handle_compute_alignment(
        &self,
        request: ComputeAlignmentRequest,
    ) -> Result<ComputeAlignmentResponse, McpError> {
        let fp_a = self.storage.retrieve(request.id_a)?
            .ok_or(McpError::not_found("Fingerprint A not found"))?;
        let fp_b = self.storage.retrieve(request.id_b)?
            .ok_or(McpError::not_found("Fingerprint B not found"))?;

        // Compute cosine similarity between purpose vectors
        let alignment = cosine_similarity(&fp_a.purpose_vector, &fp_b.purpose_vector);

        Ok(ComputeAlignmentResponse {
            alignment,
            per_embedder_alignment: compute_per_embedder_alignment(&fp_a, &fp_b),
        })
    }
}
```

### Johari Handler Updates

```rust
// File: crates/context-graph-mcp/src/handlers/johari.rs

impl JohariHandler {
    /// Get Johari classification for a fingerprint.
    ///
    /// # Changes from Stub
    /// - Uses real ΔS/ΔC computed from embeddings
    /// - Per-embedder classification (13 quadrants)
    /// - Real confidence values
    pub async fn handle_get_johari(
        &self,
        request: GetJohariRequest,
    ) -> Result<GetJohariResponse, McpError> {
        let fingerprint = self.storage.retrieve(request.id)?
            .ok_or(McpError::not_found("Fingerprint not found"))?;

        Ok(GetJohariResponse {
            id: request.id,
            quadrants: fingerprint.johari_quadrants,
            confidence: fingerprint.johari_confidence,
            dominant_quadrant: fingerprint.dominant_quadrant,
            insights: self.generate_insights(&fingerprint.johari_quadrants),
        })
    }

    /// Compute fresh Johari classification (recompute ΔS/ΔC).
    pub async fn handle_compute_johari(
        &self,
        request: ComputeJohariRequest,
    ) -> Result<ComputeJohariResponse, McpError> {
        let fingerprint = self.storage.retrieve(request.id)?
            .ok_or(McpError::not_found("Fingerprint not found"))?;

        // Compute ΔS and ΔC per embedder using real methods
        let delta_s = self.compute_delta_s(&fingerprint).await?;
        let delta_c = self.compute_delta_c(&fingerprint).await?;

        // Classify each embedder
        let quadrants: [JohariQuadrant; 13] = std::array::from_fn(|i| {
            classify_johari(delta_s[i], delta_c[i])
        });

        let confidence: [f32; 13] = std::array::from_fn(|i| {
            (delta_s[i] - 0.5).abs() + (delta_c[i] - 0.5).abs()
        });

        Ok(ComputeJohariResponse {
            delta_s,
            delta_c,
            quadrants,
            confidence,
        })
    }
}
```

### Meta-UTL Handler Updates

```rust
// File: crates/context-graph-mcp/src/handlers/meta_utl.rs

impl MetaUtlHandler {
    /// Get Meta-UTL status including real predictions.
    ///
    /// # Changes from Stub
    /// - Uses real UTL alignment calculations
    /// - No fake theta values
    /// - Real prediction accuracy from history
    pub async fn handle_get_status(
        &self,
        request: GetMetaUtlStatusRequest,
    ) -> Result<GetMetaUtlStatusResponse, McpError> {
        // Get real consciousness state
        let consciousness = self.consciousness_service.get_state(&request.session_id)?;

        // Get real prediction accuracy from history
        let prediction_accuracy = self.meta_utl.get_prediction_accuracy()?;

        // Compute real meta score
        let meta_score = compute_meta_score(
            consciousness.r,  // Kuramoto order parameter
            prediction_accuracy,
            consciousness.differentiation,
        );

        Ok(GetMetaUtlStatusResponse {
            consciousness_level: consciousness.c,
            integration: consciousness.r,
            self_reflection: prediction_accuracy,
            differentiation: consciousness.differentiation,
            meta_score,
            state: consciousness.state,
        })
    }
}
```

### Constraints

- NO stub or mock data in production paths
- All dimensions match Constitution (E6=1536, etc.)
- Real GPU inference for embeddings
- All computations use actual stored data
- Error handling for missing data

### Verification

- MCP handlers return real embedding data
- Dimensions correct (1536 for E6/E13)
- Multi-space search uses real HNSW
- Purpose/Johari computed from real values
- No stub references in handler code

---

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/multi_embedding_search.rs` | Use real multi-space search |
| `crates/context-graph-mcp/src/handlers/purpose.rs` | Use real embeddings, storage |
| `crates/context-graph-mcp/src/handlers/johari.rs` | Use real ΔS/ΔC computation |
| `crates/context-graph-mcp/src/handlers/meta_utl.rs` | Use real UTL calculations |

---

## Validation Criteria

- [ ] No `stub`, `mock`, `fake` in handler code (grep verification)
- [ ] Multi-embedding search uses HNSW indexes
- [ ] Purpose handlers compute from real vectors
- [ ] Johari handlers use real ΔS/ΔC
- [ ] Meta-UTL uses real consciousness state
- [ ] All handler responses have correct dimensions
- [ ] Error handling works for missing data

---

## Test Commands

```bash
cd /home/cabdru/contextgraph

# Check for stub/mock references
grep -rn "stub\|mock\|fake" crates/context-graph-mcp/src/handlers/
# Should return NO results in production code

# Check compilation
cargo check -p context-graph-mcp

# Run handler tests
cargo test -p context-graph-mcp handlers -- --nocapture

# Integration test (requires GPU)
cargo test -p context-graph-mcp --features cuda integration -- --nocapture
```

---

## Anti-Patterns to Remove

| Pattern | Location | Replacement |
|---------|----------|-------------|
| `create_stub_embeddings()` | All handlers | `embedding_service.embed_all()` |
| `stub_fingerprint()` | All handlers | `storage.retrieve(id)` |
| `stub_johari_quadrants()` | johari.rs | `compute_johari_quadrants()` |
| `fake_theta` | meta_utl.rs | Real alignment calculation |
| `Vec::from([0.0; 13])` | Various | Real computed values |
| `"Simulated"` strings | Various | Remove entirely |

---

## Memory Key

Store completion status:
```
contextgraph/embedding-issues/task-emb-024-complete
```

</task_spec>
