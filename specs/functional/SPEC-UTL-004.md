# SPEC-UTL-004: Compute Delta-SC MCP Tool Handler Registration

## Metadata

| Field | Value |
|-------|-------|
| **ID** | SPEC-UTL-004 |
| **Version** | 1.0 |
| **Status** | approved |
| **Owner** | ContextGraph Team |
| **Created** | 2026-01-11 |
| **Last Updated** | 2026-01-11 |
| **Related Specs** | SPEC-UTL-001, SPEC-UTL-002, SPEC-UTL-003 |
| **Priority** | P1 (Critical Path) |

---

## 1. Overview

### 1.1 Purpose

This specification defines the MCP handler registration and integration for the `compute_delta_sc` tool. While SPEC-UTL-001 defines the request/response types and algorithms, this specification covers the actual MCP handler implementation, routing, and integration with the GWT (Global Workspace Theory) toolset.

### 1.2 Problem Statement

Per constitution.yaml, `compute_delta_sc` is listed in `gwt_tools` but:
1. No MCP handler is registered for the tool
2. No routing exists to connect external requests to the computation engine
3. No integration exists between the MCP layer and the UTL computation layer

### 1.3 Success Criteria

1. MCP tool discoverable via `tools/list` as `gwt/compute_delta_sc`
2. Handler correctly routes to `DeltaScComputer`
3. Response serializes correctly to MCP protocol
4. Error states map to appropriate MCP error codes
5. Latency budget of <25ms p95 maintained

---

## 2. User Stories

### US-UTL-004-01: External Tool Invocation
**Priority**: must-have

**Narrative**:
> As an external system (Claude Code, hooks, dashboard),
> I want to call `gwt/compute_delta_sc` via MCP protocol,
> So that I can compute learning scores for vertices.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-01 | MCP server running | `tools/list` called | Returns `gwt/compute_delta_sc` in tool list |
| AC-02 | Valid request params | `tools/call` with compute_delta_sc | Returns ComputeDeltaScResponse |
| AC-03 | Invalid vertex_id | `tools/call` with bad UUID | Returns MCP error -32602 |
| AC-04 | Incomplete fingerprint | `tools/call` with partial fingerprint | Returns MCP error -32602 |

### US-UTL-004-02: Tool Schema Discovery
**Priority**: must-have

**Narrative**:
> As an MCP client,
> I want to discover the tool's input/output schema,
> So that I can construct valid requests.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-05 | MCP connection active | Schema requested for compute_delta_sc | Returns JSON schema for input |
| AC-06 | MCP connection active | Schema requested for compute_delta_sc | Returns JSON schema for output |

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Story Ref | Priority | Rationale |
|----|-------------|-----------|----------|-----------|
| REQ-UTL-004-01 | System SHALL register `gwt/compute_delta_sc` in MCP tool registry | US-UTL-004-01 | must | Tool discoverability |
| REQ-UTL-004-02 | Handler SHALL deserialize ComputeDeltaScRequest from MCP params | US-UTL-004-01 | must | Input contract |
| REQ-UTL-004-03 | Handler SHALL serialize ComputeDeltaScResponse to MCP result | US-UTL-004-01 | must | Output contract |
| REQ-UTL-004-04 | Handler SHALL map UtlError to MCP error codes per constitution | US-UTL-004-01 | must | Error handling |
| REQ-UTL-004-05 | Handler SHALL emit tracing span with tool invocation metadata | US-UTL-004-01 | should | Observability |
| REQ-UTL-004-06 | System SHALL expose JSON Schema for request/response types | US-UTL-004-02 | should | Schema discovery |
| REQ-UTL-004-07 | Handler SHALL validate ARCH-05 (all 13 embedders present) | US-UTL-004-01 | must | Constitution compliance |

### 3.2 Non-Functional Requirements

| ID | Category | Requirement | Metric | Rationale |
|----|----------|-------------|--------|-----------|
| NFR-UTL-004-01 | Performance | Handler overhead SHALL be < 1ms | Latency | Minimal MCP layer cost |
| NFR-UTL-004-02 | Reliability | Handler SHALL not panic on malformed input | Error handling | Robustness |
| NFR-UTL-004-03 | Testability | Handler SHALL have integration tests | Coverage > 90% | Quality |

---

## 4. Technical Design

### 4.1 MCP Handler Implementation

```rust
/// MCP handler for gwt/compute_delta_sc tool
///
/// Computes entropy (Delta-S) and coherence (Delta-C) changes when
/// updating a vertex in the knowledge graph.
///
/// # Constitution Reference
/// gwt_tools: [..., compute_delta_sc]
pub struct ComputeDeltaScHandler {
    /// Delta-SC computation engine
    computer: Arc<DeltaScComputer>,
    /// Graph context provider for coherence
    graph_context: Arc<dyn GraphContextProvider>,
    /// Cluster context provider for ClusterFit
    cluster_context: Arc<dyn ClusterContextProvider>,
}

impl ComputeDeltaScHandler {
    /// Create new handler with dependencies
    pub fn new(
        computer: Arc<DeltaScComputer>,
        graph_context: Arc<dyn GraphContextProvider>,
        cluster_context: Arc<dyn ClusterContextProvider>,
    ) -> Self {
        Self { computer, graph_context, cluster_context }
    }
}

#[async_trait]
impl McpToolHandler for ComputeDeltaScHandler {
    fn name(&self) -> &'static str {
        "gwt/compute_delta_sc"
    }

    fn description(&self) -> &'static str {
        "Compute entropy (Delta-S) and coherence (Delta-C) changes for a vertex update"
    }

    fn input_schema(&self) -> serde_json::Value {
        schemars::schema_for!(ComputeDeltaScRequest)
    }

    async fn call(&self, params: serde_json::Value) -> McpResult<serde_json::Value> {
        // 1. Deserialize request
        let request: ComputeDeltaScRequest = serde_json::from_value(params)
            .map_err(|e| McpError::invalid_params(format!("Invalid request: {}", e)))?;

        // 2. Validate ARCH-05 (all 13 embedders)
        if !request.new_fingerprint.has_all_embedders() {
            return Err(McpError::invalid_params(
                "Fingerprint must contain all 13 embedders (ARCH-05)"
            ));
        }

        // 3. Get contexts
        let graph_ctx = self.graph_context
            .get_context(request.vertex_id)
            .await
            .map_err(|e| McpError::internal(format!("Graph context unavailable: {}", e)))?;

        let cluster_ctx = self.cluster_context
            .get_context(request.vertex_id)
            .await
            .map_err(|e| McpError::internal(format!("Cluster context unavailable: {}", e)))?;

        // 4. Compute
        let response = self.computer
            .compute(&request, &graph_ctx, &cluster_ctx)
            .map_err(|e| match e {
                UtlError::EmptyInput => McpError::invalid_params("Empty input embedding"),
                UtlError::EntropyError(msg) => McpError::internal(format!("Entropy: {}", msg)),
                UtlError::CoherenceError(msg) => McpError::internal(format!("Coherence: {}", msg)),
                _ => McpError::internal(format!("UTL error: {}", e)),
            })?;

        // 5. Serialize response
        serde_json::to_value(response)
            .map_err(|e| McpError::internal(format!("Serialization failed: {}", e)))
    }
}
```

### 4.2 Error Code Mapping

| UtlError | MCP Error Code | Message Pattern |
|----------|----------------|-----------------|
| EmptyInput | -32602 (InvalidParams) | "Empty input embedding" |
| DimensionMismatch | -32602 (InvalidParams) | "Dimension mismatch: expected {}, got {}" |
| EntropyError | -32603 (Internal) | "Entropy computation failed: {}" |
| CoherenceError | -32603 (Internal) | "Coherence computation failed: {}" |
| ClusterFitError | -32603 (Internal) | "ClusterFit failed: {}" |

### 4.3 Tool Registration

```rust
/// Register compute_delta_sc in MCP tool registry
pub fn register_gwt_tools(registry: &mut ToolRegistry, deps: &GwtDependencies) {
    // ... other GWT tools ...

    registry.register(Box::new(ComputeDeltaScHandler::new(
        deps.delta_sc_computer.clone(),
        deps.graph_context.clone(),
        deps.cluster_context.clone(),
    )));
}
```

---

## 5. Context Provider Interfaces

### 5.1 GraphContextProvider

```rust
/// Provides graph connectivity context for coherence computation
#[async_trait]
pub trait GraphContextProvider: Send + Sync {
    /// Get graph context for a vertex
    ///
    /// Returns connectivity information needed for Delta-C's
    /// Connectivity component (0.4 weight).
    async fn get_context(&self, vertex_id: Uuid) -> Result<GraphContext, ContextError>;
}

/// Graph context for coherence computation
pub struct GraphContext {
    /// Adjacent edge weights
    pub edge_weights: Vec<f32>,
    /// Neighbor fingerprints for consistency check
    pub neighbor_fingerprints: Vec<TeleologicalFingerprint>,
    /// Historical coherence values for variance
    pub coherence_history: VecDeque<f32>,
}
```

### 5.2 ClusterContextProvider

```rust
/// Provides cluster context for ClusterFit computation
#[async_trait]
pub trait ClusterContextProvider: Send + Sync {
    /// Get cluster context for a vertex
    ///
    /// Returns cluster membership information needed for Delta-C's
    /// ClusterFit component (0.4 weight).
    async fn get_context(&self, vertex_id: Uuid) -> Result<ClusterContext, ContextError>;
}
```

---

## 6. Test Plan

### 6.1 Unit Tests

| ID | Type | Description | Req Ref |
|----|------|-------------|---------|
| TC-01 | unit | Handler name returns "gwt/compute_delta_sc" | REQ-UTL-004-01 |
| TC-02 | unit | Valid request deserializes correctly | REQ-UTL-004-02 |
| TC-03 | unit | Response serializes to JSON | REQ-UTL-004-03 |
| TC-04 | unit | Invalid UUID returns -32602 | REQ-UTL-004-04 |
| TC-05 | unit | Incomplete fingerprint returns -32602 | REQ-UTL-004-07 |
| TC-06 | unit | EntropyError maps to -32603 | REQ-UTL-004-04 |
| TC-07 | unit | Input schema is valid JSON Schema | REQ-UTL-004-06 |

### 6.2 Integration Tests

| ID | Type | Description | Req Ref |
|----|------|-------------|---------|
| TC-08 | integration | Tool appears in tools/list | REQ-UTL-004-01 |
| TC-09 | integration | Full round-trip with real fingerprints | US-UTL-004-01 |
| TC-10 | integration | Context providers wired correctly | REQ-UTL-004-02 |
| TC-11 | integration | Handler latency < 25ms p95 | NFR-UTL-004-01 |

### 6.3 Contract Tests

| ID | Type | Description |
|----|------|-------------|
| TC-12 | contract | Request matches SPEC-UTL-001 schema |
| TC-13 | contract | Response matches SPEC-UTL-001 schema |
| TC-14 | contract | Error codes match constitution.yaml |

---

## 7. Dependencies

### 7.1 Upstream Dependencies

| Dependency | Source | Usage |
|------------|--------|-------|
| ComputeDeltaScRequest/Response | SPEC-UTL-001 | Request/response types |
| DeltaScComputer | TASK-UTL-P1-001+ | Computation engine |
| ClusterFitCalculator | SPEC-UTL-002 | ClusterFit component |
| EmbedderEntropyFactory | SPEC-UTL-003 | Per-embedder entropy |

### 7.2 Downstream Consumers

| Consumer | Usage |
|----------|-------|
| Claude Code hooks | Trigger UTL learning on edits |
| Dashboard | Display Johari Window state |
| Memory consolidation | Decide consolidation priority |
| SELF_EGO_NODE | Update identity trajectory |

---

## 8. Implementation Notes

### 8.1 Thread Safety

The handler must be `Send + Sync` for async MCP runtime:
- `Arc<DeltaScComputer>` for shared computation engine
- `Arc<dyn GraphContextProvider>` for context providers
- All interior mutability via `RwLock` or atomic types

### 8.2 Tracing

```rust
#[tracing::instrument(
    name = "compute_delta_sc",
    skip(self, params),
    fields(
        vertex_id = %request.vertex_id,
        include_diagnostics = request.include_diagnostics
    )
)]
async fn call(&self, params: serde_json::Value) -> McpResult<serde_json::Value> {
    // ... implementation ...
}
```

---

## 9. Rollout Plan

1. **Phase 1**: Implement handler with mock providers (this spec)
2. **Phase 2**: Wire real GraphContextProvider
3. **Phase 3**: Wire real ClusterContextProvider
4. **Phase 4**: Integration tests with full system
5. **Phase 5**: Performance validation

---

## 10. References

- constitution.yaml: `gwt_tools` list
- constitution.yaml: `mcp.errors` section
- SPEC-UTL-001: Request/Response types
- SPEC-UTL-002: ClusterFit component
- SPEC-UTL-003: Specialized entropy methods
