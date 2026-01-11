# SPEC-MCP-001: MCP Protocol Compliance and Tool Registry

## Metadata

| Field | Value |
|-------|-------|
| **Spec ID** | SPEC-MCP-001 |
| **Title** | MCP Protocol Compliance, Tool Registry, and Naming Aliases |
| **Status** | Approved |
| **Priority** | P1 (Protocol Compliance) / P2 (Aliases) |
| **Owner** | ContextGraph Team |
| **Created** | 2025-01-11 |
| **Last Updated** | 2026-01-11 |
| **Related Specs** | SPEC-UTL-001, SPEC-GWT-001, PRD-CONSCIOUSNESS-001 |
| **Gap Reference** | MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md - GAP 1, Refinement 2 |

---

## 1. Overview

### 1.1 Purpose

This specification defines the complete MCP (Model Context Protocol) tool registry for the ContextGraph computational consciousness system. It addresses:

1. **P1: Protocol Compliance** - Ensuring all constitution-mandated tools are implemented
2. **P2: Naming Aliases** - Backwards compatibility for PRD-specified tool names
3. **Tool Organization** - Logical grouping and categorization of 35+ MCP tools

### 1.2 Problem Statement

The MCP tool implementation has two distinct issues:

**Issue 1 (P1 - Critical): Missing Tool**
The `compute_delta_sc` tool is **NOT IMPLEMENTED** but is mandated by constitution.yaml:

```yaml
gwt_tools: [get_consciousness_state, get_workspace_status, get_kuramoto_sync,
            get_ego_state, trigger_workspace_broadcast, adjust_coupling,
            get_johari_classification, compute_delta_sc]  # <-- MISSING
```

**Issue 2 (P2 - Minor): Naming Inconsistencies**

| PRD Name | Actual Name | Category |
|----------|-------------|----------|
| `discover_goals` | `discover_sub_goals` | Autonomous Tools |
| `consolidate_memories` | `trigger_consolidation` | Autonomous Tools |

### 1.3 Scope

This specification covers:
1. Complete MCP tool registry with 35+ tools organized by category
2. Tool alias resolution for backwards compatibility
3. Protocol compliance verification against constitution.yaml
4. Error code standardization across all tools
5. Response format consistency (CognitivePulse injection)

### 1.4 Success Criteria

**P1 (Protocol Compliance)**:
- `compute_delta_sc` tool registered and functional (SPEC-UTL-001)
- All constitution.yaml `gwt_tools` implemented and discoverable
- MCP 2024-11-05 protocol compliance verified

**P2 (Naming Aliases)**:
- Both `discover_goals` and `discover_sub_goals` invoke the same handler
- Both `consolidate_memories` and `trigger_consolidation` invoke the same handler
- `tools/list` returns canonical names with aliases metadata
- Zero breaking changes to existing integrations
- Test coverage for alias resolution

---

## 2. Complete Tool Registry

This section documents the complete MCP tool registry organized by subsystem.

### 2.1 Tool Registry Overview

| Category | Tool Count | Status | Priority |
|----------|------------|--------|----------|
| Core Memory | 3 | Implemented | - |
| Core Status | 3 | Implemented | - |
| GWT/Consciousness | 6 | Implemented | - |
| ATC (Adaptive Thresholds) | 3 | Implemented | - |
| Dream System | 4 | Implemented | - |
| Neuromodulation | 2 | Implemented | - |
| Steering | 1 | Implemented | - |
| Causal Inference | 1 | Implemented | - |
| Teleological | 5 | Implemented | - |
| Autonomous | 7 | Implemented | - |
| UTL (compute_delta_sc) | 1 | **NOT IMPLEMENTED** | P1 |
| **Total** | **36** | 35/36 (97%) | - |

### 2.2 Constitution-Mandated GWT Tools

Per constitution.yaml `gwt_tools` list:

| Tool Name | Status | Handler Location |
|-----------|--------|------------------|
| `get_consciousness_state` | Implemented | `handlers/tools.rs` |
| `get_workspace_status` | Implemented | `handlers/tools.rs` |
| `get_kuramoto_sync` | Implemented | `handlers/tools.rs` |
| `get_ego_state` | Implemented | `handlers/tools.rs` |
| `trigger_workspace_broadcast` | Implemented | `handlers/tools.rs` |
| `adjust_coupling` | Implemented | `handlers/tools.rs` |
| `get_johari_classification` | Implemented (via `get_memetic_status`) | `handlers/tools.rs` |
| `compute_delta_sc` | **NOT IMPLEMENTED** | See SPEC-UTL-001 |

### 2.3 Tool Categories

#### 2.3.1 Core Memory Tools

| Tool | Description | Required Params |
|------|-------------|-----------------|
| `inject_context` | Inject context with UTL processing | `content`, `rationale` |
| `store_memory` | Store memory without UTL processing | `content` |
| `search_graph` | Semantic search | `query` |

#### 2.3.2 Core Status Tools

| Tool | Description | Required Params |
|------|-------------|-----------------|
| `get_memetic_status` | System status with UTL metrics | (none) |
| `get_graph_manifest` | 5-layer architecture description | (none) |
| `utl_status` | UTL system state | (none) |

#### 2.3.3 GWT/Consciousness Tools

| Tool | Description | Required Params |
|------|-------------|-----------------|
| `get_consciousness_state` | Consciousness level C(t) | (none) |
| `get_kuramoto_sync` | Oscillator synchronization | (none) |
| `get_workspace_status` | Global Workspace state | (none) |
| `get_ego_state` | Self-Ego Node state | (none) |
| `trigger_workspace_broadcast` | Force WTA selection | `memory_id` |
| `adjust_coupling` | Adjust Kuramoto K | `new_K` |

#### 2.3.4 Adaptive Threshold Calibration (ATC) Tools

| Tool | Description | Required Params |
|------|-------------|-----------------|
| `get_threshold_status` | Current ATC thresholds | (none) |
| `get_calibration_metrics` | ECE, MCE, Brier scores | (none) |
| `trigger_recalibration` | Manual recalibration | `level` |

#### 2.3.5 Dream System Tools

| Tool | Description | Required Params |
|------|-------------|-----------------|
| `trigger_dream` | Start dream cycle | (none) |
| `get_dream_status` | Dream system state | (none) |
| `abort_dream` | Abort dream cycle | (none) |
| `get_amortized_shortcuts` | Shortcut candidates | (none) |

#### 2.3.6 Neuromodulation Tools

| Tool | Description | Required Params |
|------|-------------|-----------------|
| `get_neuromodulation_state` | All 4 modulators | (none) |
| `adjust_neuromodulator` | Adjust DA/5-HT/NE | `modulator`, `delta` |

#### 2.3.7 Steering Tools

| Tool | Description | Required Params |
|------|-------------|-----------------|
| `get_steering_feedback` | Gardener/Curator/Assessor feedback | (none) |

#### 2.3.8 Causal Inference Tools

| Tool | Description | Required Params |
|------|-------------|-----------------|
| `omni_infer` | Omni-directional inference | `source` |

#### 2.3.9 Teleological Tools

| Tool | Description | Required Params |
|------|-------------|-----------------|
| `search_teleological` | Cross-correlation search | (none) |
| `compute_teleological_vector` | Compute 13-embedder vector | `content` |
| `fuse_embeddings` | Fuse with synergy matrix | `memory_id` |
| `update_synergy_matrix` | Update synergy from feedback | `query_vector_id`, `result_vector_id`, `feedback` |
| `manage_teleological_profile` | CRUD for profiles | `action` |

#### 2.3.10 Autonomous Tools

| Tool | Canonical Name | Alias | Description |
|------|----------------|-------|-------------|
| Bootstrap | `auto_bootstrap_north_star` | - | Initialize autonomous system |
| Drift | `get_alignment_drift` | - | Get drift state |
| Correction | `trigger_drift_correction` | - | Correct drift |
| Pruning | `get_pruning_candidates` | - | Get prune candidates |
| Consolidation | `trigger_consolidation` | `consolidate_memories` | Merge similar memories |
| Goals | `discover_sub_goals` | `discover_goals` | Discover sub-goals |
| Status | `get_autonomous_status` | - | Autonomous system status |

#### 2.3.11 UTL Tools (Pending Implementation)

| Tool | Status | Specification |
|------|--------|---------------|
| `compute_delta_sc` | **NOT IMPLEMENTED** | SPEC-UTL-001 |

**Implementation Tasks**:
- TASK-DELTA-P1-001: Request/Response types
- TASK-DELTA-P1-002: DeltaScComputer logic
- TASK-DELTA-P1-003: MCP handler registration
- TASK-DELTA-P1-004: Integration tests

---

## 3. Requirements

### 3.1 Protocol Compliance Requirements

#### FR-MCP-P1: Constitution Tool Implementation
| ID | Description | Priority |
|----|-------------|----------|
| FR-P1-001 | System SHALL implement all tools listed in constitution.yaml `gwt_tools` | Must-Have |
| FR-P1-002 | `compute_delta_sc` tool SHALL compute Delta-S and Delta-C per SPEC-UTL-001 | Must-Have |
| FR-P1-003 | All tools SHALL return MCP 2024-11-05 compliant responses | Must-Have |
| FR-P1-004 | All tool calls SHALL include `_cognitive_pulse` in response | Must-Have |

### 3.2 Alias Resolution Requirements

#### FR-MCP-P2: Alias Resolution in Tool Dispatch
| ID | Description | Priority |
|----|-------------|----------|
| FR-001 | The `tools/call` handler MUST resolve tool aliases before dispatching | Must-Have |
| FR-002 | Alias resolution MUST map PRD names to canonical implementation names | Must-Have |
| FR-003 | Alias resolution MUST be case-sensitive | Must-Have |
| FR-004 | Unknown tool names MUST return error code `-32004` (TOOL_NOT_FOUND) | Must-Have |

#### FR-MCP-002: Tool Listing with Aliases
| ID | Description | Priority |
|----|-------------|----------|
| FR-010 | `tools/list` MUST include alias information in tool metadata | Should-Have |
| FR-011 | Each tool definition MAY include an `aliases` field (array of strings) | Should-Have |
| FR-012 | Aliases SHOULD be documented in tool descriptions | Should-Have |

#### FR-MCP-P2-003: Logging and Observability
| ID | Description | Priority |
|----|-------------|----------|
| FR-020 | Alias resolution MUST log when an alias is used (debug level) | Should-Have |
| FR-021 | Metrics SHOULD track alias usage vs canonical name usage | Nice-to-Have |

### 3.3 Non-Functional Requirements

| ID | Category | Requirement | Metric |
|----|----------|-------------|--------|
| NFR-001 | Performance | Alias resolution overhead < 1us | p99 latency |
| NFR-002 | Performance | compute_delta_sc latency < 25ms | p95 latency |
| NFR-003 | Maintainability | Alias mappings centralized in single location | Code review |
| NFR-004 | Testability | All aliases covered by unit tests | Test coverage |
| NFR-005 | Reliability | All tools return valid CognitivePulse | Response validation |

---

## 4. Error Codes

### 4.1 Standard MCP Error Codes

| Code | Name | Description |
|------|------|-------------|
| -32700 | PARSE_ERROR | Invalid JSON received |
| -32600 | INVALID_REQUEST | Invalid JSON-RPC request |
| -32601 | METHOD_NOT_FOUND | Unknown MCP method |
| -32602 | INVALID_PARAMS | Invalid method parameters |
| -32603 | INTERNAL_ERROR | Internal server error |

### 4.2 ContextGraph Custom Error Codes

| Code | Name | Constant | Description |
|------|------|----------|-------------|
| -32001 | STORAGE_ERROR | `error_codes::STORAGE_ERROR` | TeleologicalStore operation failed |
| -32002 | EMBEDDING_ERROR | `error_codes::EMBEDDING_ERROR` | Multi-array embedding generation failed |
| -32003 | UTL_ERROR | `error_codes::UTL_ERROR` | UTL metrics computation failed |
| -32004 | TOOL_NOT_FOUND | `error_codes::TOOL_NOT_FOUND` | Unknown tool name (after alias resolution) |
| -32005 | GWT_NOT_INITIALIZED | `error_codes::GWT_NOT_INITIALIZED` | GWT components not initialized |
| -32006 | WORKSPACE_ERROR | `error_codes::WORKSPACE_ERROR` | Workspace operation failed |
| -32007 | CONSCIOUSNESS_COMPUTATION_FAILED | `error_codes::CONSCIOUSNESS_COMPUTATION_FAILED` | C(t) computation failed |
| -32008 | DREAM_ERROR | `error_codes::DREAM_ERROR` | Dream system error |
| -32009 | NEUROMOD_ERROR | `error_codes::NEUROMOD_ERROR` | Neuromodulation error |
| -32010 | THRESHOLD_ERROR | `error_codes::THRESHOLD_ERROR` | ATC threshold error |

### 4.3 UTL-Specific Error Codes (compute_delta_sc)

| Code | Name | Description |
|------|------|-------------|
| -32801 | INVALID_FINGERPRINT | Fingerprint missing required embedders (ARCH-05 violation) |
| -32802 | COMPUTATION_ERROR | Entropy or coherence computation failed |
| -32803 | DIMENSION_MISMATCH | Embedding dimension mismatch for embedder |

---

## 5. Technical Design

### 5.1 Alias Registry Structure

```rust
/// Tool alias registry for backwards compatibility
/// Maps PRD names to canonical implementation names
pub mod tool_aliases {
    use std::collections::HashMap;
    use once_cell::sync::Lazy;

    /// Static alias map: PRD name -> canonical name
    pub static ALIAS_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
        let mut m = HashMap::new();
        // PRD Refinement 2 aliases
        m.insert("discover_goals", "discover_sub_goals");
        m.insert("consolidate_memories", "trigger_consolidation");
        m
    });

    /// Resolve a tool name, returning canonical name if alias exists
    #[inline]
    pub fn resolve(name: &str) -> &str {
        ALIAS_MAP.get(name).copied().unwrap_or(name)
    }
}
```

### 5.2 Integration Point

The alias resolution integrates into `handle_tools_call` in `handlers/tools.rs`:

```rust
pub(super) async fn handle_tools_call(
    &self,
    id: Option<JsonRpcId>,
    params: Option<serde_json::Value>,
) -> JsonRpcResponse {
    // ... parameter extraction ...

    let tool_name = match params.get("name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => {
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                "Missing 'name' parameter in tools/call",
            );
        }
    };

    // Resolve alias to canonical name
    let canonical_name = tool_aliases::resolve(tool_name);

    if canonical_name != tool_name {
        debug!(
            alias = tool_name,
            canonical = canonical_name,
            "Resolved tool alias to canonical name"
        );
    }

    // Dispatch using canonical_name instead of tool_name
    match canonical_name {
        // ... existing dispatch ...
    }
}
```

### 5.3 Tool Definition Enhancement

```rust
/// Enhanced tool definition with optional aliases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,

    /// Optional aliases for backwards compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aliases: Option<Vec<String>>,
}
```

### 5.4 Affected Tool Definitions

#### 5.4.1 discover_sub_goals (Canonical)

```rust
ToolDefinition::new(
    "discover_sub_goals",
    "Discover potential sub-goals from memory clusters. Analyzes stored memories to find \
     emergent themes and patterns that could become strategic or tactical goals. \
     Helps evolve the goal hierarchy based on actual content. \
     \n\nAlias: discover_goals (PRD compatibility)",
    // ... schema unchanged ...
).with_aliases(vec!["discover_goals"])
```

#### 5.4.2 trigger_consolidation (Canonical)

```rust
ToolDefinition::new(
    "trigger_consolidation",
    "Trigger memory consolidation to merge similar memories and reduce redundancy. \
     Uses similarity-based, temporal, or semantic strategies to identify merge candidates. \
     Helps optimize memory storage and improve retrieval efficiency. \
     \n\nAlias: consolidate_memories (PRD compatibility)",
    // ... schema unchanged ...
).with_aliases(vec!["consolidate_memories"])
```

---

## 6. Acceptance Criteria

### 6.1 Gherkin Scenarios

```gherkin
Feature: MCP Tool Naming Aliases

  Background:
    Given the MCP server is running
    And the tool registry is initialized

  Scenario: Invoke tool using canonical name
    When I call tools/call with name "discover_sub_goals"
    Then the discover_sub_goals handler is invoked
    And no alias resolution log is emitted

  Scenario: Invoke tool using PRD alias
    When I call tools/call with name "discover_goals"
    Then the discover_sub_goals handler is invoked
    And a debug log is emitted: "Resolved tool alias to canonical name"

  Scenario: Invoke consolidation using PRD alias
    When I call tools/call with name "consolidate_memories"
    Then the trigger_consolidation handler is invoked
    And a debug log is emitted: "Resolved tool alias to canonical name"

  Scenario: Tool list includes alias information
    When I call tools/list
    Then the response includes "discover_sub_goals" with aliases ["discover_goals"]
    And the response includes "trigger_consolidation" with aliases ["consolidate_memories"]

  Scenario: Unknown tool returns error
    When I call tools/call with name "nonexistent_tool"
    Then the response contains error code -32004
    And the error message contains "Unknown tool: nonexistent_tool"
```

### 6.2 Test Coverage Requirements

| Test Type | Scope | Coverage Target |
|-----------|-------|-----------------|
| Unit | `tool_aliases::resolve()` | 100% |
| Unit | `ToolDefinition::with_aliases()` | 100% |
| Integration | `handle_tools_call` alias dispatch | All aliases |
| Integration | `tools/list` alias serialization | All aliases |

---

## 7. Edge Cases

### 7.1 Alias Collision Prevention

**Scenario**: An alias matches an existing canonical tool name.

**Prevention**: The alias registry MUST be validated at compile time or server startup to ensure:
1. No alias matches any canonical tool name
2. No two tools share the same alias
3. Aliases do not form circular references

```rust
#[cfg(test)]
mod alias_validation {
    use super::*;

    #[test]
    fn test_no_alias_collisions_with_canonical() {
        let canonical_names: HashSet<_> = get_tool_definitions()
            .iter()
            .map(|t| t.name.as_str())
            .collect();

        for (alias, _) in tool_aliases::ALIAS_MAP.iter() {
            assert!(
                !canonical_names.contains(alias),
                "Alias '{}' collides with canonical tool name",
                alias
            );
        }
    }
}
```

### 7.2 Case Sensitivity

Tool names and aliases are **case-sensitive**. The following are distinct:
- `discover_goals` (valid alias)
- `Discover_Goals` (unknown tool)
- `DISCOVER_GOALS` (unknown tool)

---

## 8. Migration Notes

### 8.1 Backwards Compatibility

This change is **fully backwards compatible**:
- Existing clients using canonical names continue to work unchanged
- New clients can use PRD aliases immediately
- No API version bump required

### 8.2 Deprecation Strategy

Aliases should be considered **permanent backwards compatibility shims**. If deprecation is desired in the future:

1. Add deprecation notice to tool descriptions
2. Emit deprecation warning log on alias use
3. Track alias usage metrics for impact analysis
4. Provide 6-month minimum deprecation window

---

## 9. Dependencies

### 9.1 Implementation Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| `once_cell` | Lazy static initialization for alias map | 1.x |
| `tracing` | Debug logging for alias resolution | existing |

### 9.2 Specification Dependencies

| Spec ID | Relationship |
|---------|--------------|
| SPEC-MCP-TOOLS-001 | Base tool definitions |
| PRD-CONSCIOUSNESS-001 | Source of PRD tool names |

---

## 10. Validation Checklist

### P1: Protocol Compliance

- [ ] `compute_delta_sc` tool implemented (TASK-DELTA-P1-003)
- [ ] All 8 constitution.yaml `gwt_tools` discoverable via `tools/list`
- [ ] All tools return MCP 2024-11-05 compliant responses
- [ ] All tool responses include `_cognitive_pulse`
- [ ] Error codes match documented values in Section 4
- [ ] compute_delta_sc latency < 25ms p95

### P2: Naming Aliases

- [ ] All aliases map to existing canonical tool names
- [ ] No alias collides with a canonical tool name
- [ ] `tools/list` output includes alias metadata
- [ ] Debug logging enabled for alias resolution
- [ ] Unit tests cover all alias mappings
- [ ] Integration tests verify end-to-end alias invocation
- [ ] Performance benchmark confirms <1us overhead

---

## 11. References

### 11.1 Internal References

| Reference | Path | Purpose |
|-----------|------|---------|
| Gap Analysis | `/docs/MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md` | Gap identification (GAP 1, Refinement 2) |
| Constitution | `/constitution.yaml` | gwt_tools mandate |
| Tool Definitions | `/crates/context-graph-mcp/src/tools.rs` | Tool registry |
| Tool Handlers | `/crates/context-graph-mcp/src/handlers/tools.rs` | Dispatch logic |
| Autonomous Handlers | `/crates/context-graph-mcp/src/handlers/autonomous.rs` | Aliased handlers |
| Protocol | `/crates/context-graph-mcp/src/protocol.rs` | Error codes |
| UTL Spec | `SPEC-UTL-001` | compute_delta_sc specification |

### 11.2 Related Tasks

| Task ID | Title | Priority | Status |
|---------|-------|----------|--------|
| TASK-DELTA-P1-001 | DeltaScRequest/Response types | P1 | ready |
| TASK-DELTA-P1-002 | DeltaScComputer implementation | P1 | ready |
| TASK-DELTA-P1-003 | MCP handler registration | P1 | ready |
| TASK-DELTA-P1-004 | Integration tests | P1 | ready |
| TASK-MCP-P2-001 | MCP tool naming aliases | P2 | ready |

### 11.3 External References

- [MCP Protocol Specification 2024-11-05](https://modelcontextprotocol.io/specification)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)

---

## Appendix A: Complete Tool List (Alphabetical)

| Tool Name | Category | Alias |
|-----------|----------|-------|
| `abort_dream` | Dream | - |
| `adjust_coupling` | GWT | - |
| `adjust_neuromodulator` | Neuromod | - |
| `auto_bootstrap_north_star` | Autonomous | - |
| `compute_delta_sc` | UTL | - |
| `compute_teleological_vector` | Teleological | - |
| `discover_sub_goals` | Autonomous | `discover_goals` |
| `fuse_embeddings` | Teleological | - |
| `get_alignment_drift` | Autonomous | - |
| `get_amortized_shortcuts` | Dream | - |
| `get_autonomous_status` | Autonomous | - |
| `get_calibration_metrics` | ATC | - |
| `get_consciousness_state` | GWT | - |
| `get_dream_status` | Dream | - |
| `get_ego_state` | GWT | - |
| `get_graph_manifest` | Core | - |
| `get_kuramoto_sync` | GWT | - |
| `get_memetic_status` | Core | - |
| `get_neuromodulation_state` | Neuromod | - |
| `get_pruning_candidates` | Autonomous | - |
| `get_steering_feedback` | Steering | - |
| `get_threshold_status` | ATC | - |
| `get_workspace_status` | GWT | - |
| `inject_context` | Core | - |
| `manage_teleological_profile` | Teleological | - |
| `omni_infer` | Causal | - |
| `search_graph` | Core | - |
| `search_teleological` | Teleological | - |
| `store_memory` | Core | - |
| `trigger_consolidation` | Autonomous | `consolidate_memories` |
| `trigger_dream` | Dream | - |
| `trigger_drift_correction` | Autonomous | - |
| `trigger_recalibration` | ATC | - |
| `update_synergy_matrix` | Teleological | - |
| `utl_status` | Core | - |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-11 | ContextGraph Team | Initial draft (aliases only) |
| 2.0 | 2026-01-11 | Agent 8 | Complete tool registry, error codes, P1 compliance |
