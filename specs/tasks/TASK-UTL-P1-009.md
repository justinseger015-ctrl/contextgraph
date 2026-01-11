# TASK-UTL-P1-009: Register compute_delta_sc MCP Handler

## Metadata

| Field | Value |
|-------|-------|
| **ID** | TASK-UTL-P1-009 |
| **Title** | Register compute_delta_sc MCP Handler |
| **Status** | blocked |
| **Layer** | surface (Layer 3) |
| **Sequence** | 9 |
| **Priority** | P1 |
| **Estimated Complexity** | medium |
| **Implements** | REQ-UTL-004-01 through REQ-UTL-004-07, REQ-UTL-001 |
| **Depends On** | TASK-UTL-P1-001, TASK-UTL-P1-008 |
| **Spec Ref** | SPEC-UTL-001, SPEC-UTL-004 |

---

## Context

This task registers the `compute_delta_sc` MCP tool handler, completing the critical path for external UTL access. This is the final integration task that exposes the complete UTL computation pipeline via MCP protocol.

**Gap Being Addressed:**
> constitution.yaml: `gwt_tools: [..., compute_delta_sc]`
> The tool is listed in gwt_tools but no MCP handler exists.

---

## Input Context Files

| File | Purpose | Read Before |
|------|---------|-------------|
| `crates/context-graph-mcp/src/types/delta_sc.rs` | Request/Response types from TASK-UTL-P1-001 | Required |
| `crates/context-graph-mcp/src/handlers/gwt/mod.rs` | GWT tool handlers module | Required |
| `crates/context-graph-utl/src/delta_sc/computer.rs` | DeltaScComputer implementation | Required |
| `crates/context-graph-utl/src/coherence/tracker.rs` | CoherenceTracker with ClusterFit | Required |
| `specs/functional/SPEC-UTL-004.md` | Full handler specification | Required |
| `docs2/constitution.yaml` | MCP error codes and gwt_tools | Reference |

---

## Prerequisites

- [ ] **TASK-UTL-P1-001 completed** (types exist)
- [ ] **TASK-UTL-P1-008 completed** (CoherenceTracker has ClusterFit)
- [ ] DeltaScComputer implemented and tested
- [ ] MCP server framework exists

---

## Scope

### In Scope

- Create `ComputeDeltaScHandler` implementing `McpToolHandler`
- Implement request deserialization with ARCH-05 validation
- Implement response serialization
- Map UtlError variants to MCP error codes per constitution
- Register handler in GWT tools module
- Add tracing instrumentation
- Create GraphContextProvider trait and implementation
- Create ClusterContextProvider trait and implementation
- Unit and integration tests

### Out of Scope

- DeltaScComputer implementation (separate task)
- ClusterFitCalculator implementation (TASK-UTL-P1-007)
- Entropy calculator implementations (TASK-UTL-P1-003 through 006)

---

## Definition of Done

### Exact Signatures Required

```rust
// File: crates/context-graph-mcp/src/handlers/gwt/compute_delta_sc.rs

use async_trait::async_trait;
use std::sync::Arc;
use uuid::Uuid;
use crate::types::delta_sc::{ComputeDeltaScRequest, ComputeDeltaScResponse};
use crate::error::{McpError, McpResult};
use crate::handler::McpToolHandler;
use context_graph_utl::delta_sc::DeltaScComputer;

/// MCP handler for gwt/compute_delta_sc tool.
///
/// Computes entropy (Delta-S) and coherence (Delta-C) changes when
/// updating a vertex in the knowledge graph.
///
/// # Constitution Reference
/// - `gwt_tools: [..., compute_delta_sc]`
/// - `mcp.errors: { -32602: InvalidParams, -32603: Internal }`
#[derive(Clone)]
pub struct ComputeDeltaScHandler {
    /// Delta-SC computation engine
    computer: Arc<DeltaScComputer>,
    /// Graph context provider for coherence
    graph_context: Arc<dyn GraphContextProvider>,
    /// Cluster context provider for ClusterFit
    cluster_context: Arc<dyn ClusterContextProvider>,
}

impl ComputeDeltaScHandler {
    /// Create new handler with dependencies.
    ///
    /// # Arguments
    /// * `computer` - DeltaScComputer instance
    /// * `graph_context` - Provider for graph connectivity context
    /// * `cluster_context` - Provider for cluster membership context
    pub fn new(
        computer: Arc<DeltaScComputer>,
        graph_context: Arc<dyn GraphContextProvider>,
        cluster_context: Arc<dyn ClusterContextProvider>,
    ) -> Self;
}

#[async_trait]
impl McpToolHandler for ComputeDeltaScHandler {
    /// Returns "gwt/compute_delta_sc"
    fn name(&self) -> &'static str;

    /// Returns tool description
    fn description(&self) -> &'static str;

    /// Returns JSON Schema for input validation
    fn input_schema(&self) -> serde_json::Value;

    /// Execute the tool
    ///
    /// # Errors
    /// - MCP -32602 for invalid parameters
    /// - MCP -32603 for internal computation errors
    async fn call(&self, params: serde_json::Value) -> McpResult<serde_json::Value>;
}

/// Provides graph connectivity context for coherence computation.
#[async_trait]
pub trait GraphContextProvider: Send + Sync {
    /// Get graph context for a vertex.
    async fn get_context(&self, vertex_id: Uuid) -> Result<GraphContext, ContextError>;
}

/// Provides cluster context for ClusterFit computation.
#[async_trait]
pub trait ClusterContextProvider: Send + Sync {
    /// Get cluster context for a vertex.
    async fn get_context(&self, vertex_id: Uuid) -> Result<ClusterContext, ContextError>;
}

/// Graph context for coherence computation.
#[derive(Debug, Clone)]
pub struct GraphContext {
    /// Adjacent edge weights
    pub edge_weights: Vec<f32>,
    /// Neighbor fingerprints for consistency check
    pub neighbor_fingerprints: Vec<TeleologicalFingerprint>,
    /// Historical coherence values for variance
    pub coherence_history: VecDeque<f32>,
}
```

### Error Code Mapping

Implement in handler according to constitution.yaml:

| UtlError Variant | MCP Code | Message |
|------------------|----------|---------|
| EmptyInput | -32602 | "Empty input embedding" |
| DimensionMismatch { exp, got } | -32602 | "Dimension mismatch: expected {exp}, got {got}" |
| EntropyError(msg) | -32603 | "Entropy computation failed: {msg}" |
| CoherenceError(msg) | -32603 | "Coherence computation failed: {msg}" |
| ClusterFitError(msg) | -32603 | "ClusterFit computation failed: {msg}" |

### Registration

```rust
// File: crates/context-graph-mcp/src/handlers/gwt/mod.rs

mod compute_delta_sc;
pub use compute_delta_sc::ComputeDeltaScHandler;

/// Register all GWT tools
pub fn register_gwt_tools(registry: &mut ToolRegistry, deps: &GwtDependencies) {
    // ... existing tools ...

    registry.register(Box::new(ComputeDeltaScHandler::new(
        deps.delta_sc_computer.clone(),
        deps.graph_context.clone(),
        deps.cluster_context.clone(),
    )));
}
```

### Constraints

- Handler MUST be `Send + Sync` for async runtime
- Handler MUST validate ARCH-05 (all 13 embedders)
- Handler overhead MUST be < 1ms (computation budget is 24ms)
- Error messages MUST be descriptive (include context)
- NO panics on malformed input

### Verification Commands

```bash
# Type check
cargo check -p context-graph-mcp

# Run handler tests
cargo test -p context-graph-mcp --lib -- handlers::gwt::compute_delta_sc --nocapture

# Lint check
cargo clippy -p context-graph-mcp -- -D warnings

# Integration test (requires MCP server running)
cargo test -p context-graph-mcp --test mcp_integration -- compute_delta_sc
```

---

## Pseudo-code

```rust
impl ComputeDeltaScHandler {
    pub fn new(computer, graph_context, cluster_context) -> Self {
        Self { computer, graph_context, cluster_context }
    }
}

#[async_trait]
impl McpToolHandler for ComputeDeltaScHandler {
    fn name(&self) -> &'static str {
        "gwt/compute_delta_sc"
    }

    fn description(&self) -> &'static str {
        "Compute entropy (ΔS) and coherence (ΔC) changes for vertex update"
    }

    fn input_schema(&self) -> serde_json::Value {
        // Return JSON Schema for ComputeDeltaScRequest
        serde_json::json!({
            "type": "object",
            "required": ["vertex_id", "old_fingerprint", "new_fingerprint"],
            "properties": {
                "vertex_id": { "type": "string", "format": "uuid" },
                "old_fingerprint": { "$ref": "#/definitions/TeleologicalFingerprint" },
                "new_fingerprint": { "$ref": "#/definitions/TeleologicalFingerprint" },
                "include_diagnostics": { "type": "boolean", "default": false },
                "johari_threshold": { "type": "number", "minimum": 0.35, "maximum": 0.65 }
            }
        })
    }

    #[tracing::instrument(name = "gwt/compute_delta_sc", skip(self, params))]
    async fn call(&self, params: serde_json::Value) -> McpResult<serde_json::Value> {
        // 1. Deserialize request
        let request: ComputeDeltaScRequest = serde_json::from_value(params)
            .map_err(|e| McpError::invalid_params(format!("Invalid request: {}", e)))?;

        // 2. Validate ARCH-05 (all 13 embedders present)
        if !request.new_fingerprint.has_all_embedders() {
            return Err(McpError::invalid_params(
                "Fingerprint must contain all 13 embedders (ARCH-05)"
            ));
        }
        if !request.old_fingerprint.has_all_embedders() {
            return Err(McpError::invalid_params(
                "Old fingerprint must contain all 13 embedders (ARCH-05)"
            ));
        }

        // 3. Get graph context for connectivity component
        let graph_ctx = self.graph_context
            .get_context(request.vertex_id)
            .await
            .map_err(|e| McpError::internal(format!("Graph context: {}", e)))?;

        // 4. Get cluster context for ClusterFit component
        let cluster_ctx = self.cluster_context
            .get_context(request.vertex_id)
            .await
            .map_err(|e| McpError::internal(format!("Cluster context: {}", e)))?;

        // 5. Compute Delta-SC
        let response = self.computer
            .compute(&request, &graph_ctx, &cluster_ctx)
            .map_err(|e| Self::map_utl_error(e))?;

        // 6. Serialize response
        serde_json::to_value(response)
            .map_err(|e| McpError::internal(format!("Serialization: {}", e)))
    }
}

impl ComputeDeltaScHandler {
    fn map_utl_error(e: UtlError) -> McpError {
        match e {
            UtlError::EmptyInput =>
                McpError::invalid_params("Empty input embedding"),
            UtlError::DimensionMismatch { expected, got } =>
                McpError::invalid_params(format!("Dimension mismatch: expected {}, got {}", expected, got)),
            UtlError::EntropyError(msg) =>
                McpError::internal(format!("Entropy computation failed: {}", msg)),
            UtlError::CoherenceError(msg) =>
                McpError::internal(format!("Coherence computation failed: {}", msg)),
            UtlError::ClusterFitError(msg) =>
                McpError::internal(format!("ClusterFit computation failed: {}", msg)),
            other =>
                McpError::internal(format!("UTL error: {}", other)),
        }
    }
}
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-mcp/src/handlers/gwt/compute_delta_sc.rs` | Handler implementation |
| `crates/context-graph-mcp/src/context/graph.rs` | GraphContextProvider trait and impl |
| `crates/context-graph-mcp/src/context/cluster.rs` | ClusterContextProvider trait and impl |
| `crates/context-graph-mcp/src/context/mod.rs` | Context module exports |

---

## Files to Modify

| Path | Modification |
|------|--------------|
| `crates/context-graph-mcp/src/handlers/gwt/mod.rs` | Add compute_delta_sc module and registration |
| `crates/context-graph-mcp/src/handlers/mod.rs` | Export GwtDependencies with new fields |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| Tool name is "gwt/compute_delta_sc" | Unit test assertion |
| Tool appears in tools/list | Integration test |
| Valid request returns response | Integration test with fixture |
| Invalid UUID returns -32602 | Unit test with bad input |
| Missing fingerprint returns -32602 | Unit test with partial fingerprint |
| Computation error returns -32603 | Unit test with error injection |
| Handler latency < 25ms p95 | Benchmark test |
| No clippy warnings | `cargo clippy` clean |
| Tracing spans emitted | Log inspection |

---

## Required Tests

| Test Name | Description |
|-----------|-------------|
| `test_handler_name` | Returns "gwt/compute_delta_sc" |
| `test_handler_description` | Returns non-empty description |
| `test_input_schema_valid` | Input schema is valid JSON Schema |
| `test_valid_request_roundtrip` | Full request/response cycle |
| `test_invalid_uuid_error` | Bad UUID returns -32602 |
| `test_incomplete_fingerprint_arch05` | Partial fingerprint returns -32602 |
| `test_entropy_error_maps_to_32603` | EntropyError -> -32603 |
| `test_coherence_error_maps_to_32603` | CoherenceError -> -32603 |
| `test_handler_is_send_sync` | Handler satisfies Send + Sync |
| `test_handler_clone` | Handler can be cloned |
| `test_integration_tools_list` | Tool in tools/list |
| `test_integration_full_cycle` | End-to-end with real data |

---

## Notes

- This is the final "surface" integration task for the UTL compute pipeline
- The handler is intentionally thin - all logic is in DeltaScComputer
- Context providers are injected to allow testing with mocks
- Tracing is essential for debugging production issues
- The 25ms p95 budget includes all three layers: MCP handler (~1ms), entropy (~10ms), coherence (~5ms), with ~9ms headroom

---

## Related Tasks

- **TASK-UTL-P1-001**: Request/Response types (prerequisite)
- **TASK-UTL-P1-008**: CoherenceTracker integration (prerequisite)
- **TASK-UTL-P1-003-006**: Entropy calculators (consumed by DeltaScComputer)
