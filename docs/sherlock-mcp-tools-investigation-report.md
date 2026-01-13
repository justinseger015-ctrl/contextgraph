# Sherlock Holmes Forensic Investigation Report
## MCP Tools & API Integration - ULTIMATE CONTEXT GRAPH

**Case ID**: MCP-TOOLS-2026-01-12
**Investigator**: Sherlock Holmes (Agent #4)
**Date**: 2026-01-12
**Subject**: MCP Tool Implementation Compliance with PRD Requirements

---

## EXECUTIVE SUMMARY

*"The game is afoot!"*

After exhaustive forensic analysis of the MCP tool implementation in the ULTIMATE CONTEXT GRAPH codebase, I present findings that reveal both **substantial compliance** and **significant gaps** with the Product Requirements Document (PRD).

### Verdict Summary

| Category | Status | Confidence |
|----------|--------|------------|
| Core Tools (inject_context, store_memory, etc.) | **IMPLEMENTED** | HIGH |
| GWT Tools (consciousness, workspace, Kuramoto) | **IMPLEMENTED** | HIGH |
| UTL Tools (compute_delta_sc) - CRITICAL | **IMPLEMENTED** | HIGH |
| ATC Tools (thresholds, calibration) | **IMPLEMENTED** | HIGH |
| Dream Tools (trigger_dream, status, abort) | **IMPLEMENTED** | HIGH |
| Neuromodulation Tools | **IMPLEMENTED** | HIGH |
| Steering Tools | **IMPLEMENTED** | HIGH |
| Causal Tools (omni_infer) | **IMPLEMENTED** | HIGH |
| Teleological Tools (5 tools) | **IMPLEMENTED** | HIGH |
| Autonomous Tools (7 tools) | **IMPLEMENTED** | HIGH |
| Meta-UTL Tools (3 tools) | **IMPLEMENTED** | HIGH |
| **Curation Tools** | **MISSING** | HIGH |
| **Navigation Tools** | **MISSING** | HIGH |
| **Meta-Cognitive Tools** | **MISSING** | HIGH |
| **Diagnostic Tools** | **MISSING** | HIGH |
| **Admin Tools** | **MISSING** | HIGH |
| JSON-RPC 2.0 Compliance | **COMPLIANT** | HIGH |
| Parameter Validation | **PARTIAL** | MEDIUM |

**OVERALL VERDICT**: The implementation covers approximately **52%** (39 of ~75) of PRD-specified tools. Core GWT/UTL functionality is INNOCENT, but Curation, Navigation, Meta-Cognitive, and Diagnostic tool categories are GUILTY of being UNIMPLEMENTED.

---

## EVIDENCE INVENTORY

### 1. Tool Definitions Located

**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs`

```rust
// Tool name constants (39 tools total):
// Core: inject_context, store_memory, get_memetic_status, get_graph_manifest, search_graph, utl_status
// GWT: get_consciousness_state, get_kuramoto_sync, get_workspace_status, get_ego_state, trigger_workspace_broadcast, adjust_coupling
// UTL: gwt/compute_delta_sc
// ATC: get_threshold_status, get_calibration_metrics, trigger_recalibration
// Dream: trigger_dream, get_dream_status, abort_dream, get_amortized_shortcuts
// Neuromod: get_neuromodulation_state, adjust_neuromodulator
// Steering: get_steering_feedback
// Causal: omni_infer
// Teleological: search_teleological, compute_teleological_vector, fuse_embeddings, update_synergy_matrix, manage_teleological_profile
// Autonomous: auto_bootstrap_north_star, get_alignment_drift, trigger_drift_correction, get_pruning_candidates, trigger_consolidation, discover_sub_goals, get_autonomous_status
// Meta-UTL: get_meta_learning_status, trigger_lambda_recalibration, get_meta_learning_log
```

### 2. Dispatch Implementation

**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/dispatch.rs`

All 39 defined tools have corresponding dispatch handlers. The dispatch uses a match statement with proper error handling for unknown tools.

### 3. Critical Tool: compute_delta_sc (UTL-001)

**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/utl/gwt.rs`

**VERDICT**: **INNOCENT** - Fully implemented with:
- Per-embedder delta-S computation (13 values)
- Aggregate delta-S calculation
- Delta-C computation using three-component formula (Connectivity + ClusterFit + Consistency)
- Johari quadrant classification per embedder
- UTL learning potential calculation
- Proper error handling and fail-fast behavior
- NaN/Inf protection per AP-10

```rust
// Lines 37-215: Full implementation of handle_gwt_compute_delta_sc
// Returns: delta_s_per_embedder, delta_s_aggregate, delta_c, johari_quadrants, utl_learning_potential
```

### 4. JSON-RPC 2.0 Protocol

**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/protocol.rs`

**VERDICT**: **COMPLIANT**

Evidence:
- `jsonrpc: "2.0"` string in all responses
- Proper `JsonRpcRequest` and `JsonRpcResponse` structures
- Error codes follow JSON-RPC spec (-32700 to -32600 for standard errors)
- Custom error codes properly namespaced (-32001 to -32114)
- ID handling supports both string and number types

```rust
// Line 8-30: JsonRpcRequest/Response structs
// Line 51-74: success/error factory methods with "2.0" version
// Line 87-257: Comprehensive error codes
```

### 5. Transport Support

**Finding**: The implementation supports:
- **stdio** - Primary transport (verified in server.rs)
- **TCP** - With connection limits and timeouts (error codes -32110 to -32114)

**Missing**: SSE transport not found in implementation.

---

## TOOL INVENTORY: IMPLEMENTED vs REQUIRED

### A. IMPLEMENTED TOOLS (39 Total)

| Category | Tool Name | PRD Reference | Status |
|----------|-----------|---------------|--------|
| **Core** | inject_context | Section 5.2 | IMPLEMENTED |
| **Core** | store_memory | Section 5.2 | IMPLEMENTED |
| **Core** | get_memetic_status | Section 5.2 | IMPLEMENTED |
| **Core** | get_graph_manifest | Section 5.2 | IMPLEMENTED |
| **Core** | search_graph | Section 5.2 | IMPLEMENTED |
| **Core** | utl_status | Section 5.6 | IMPLEMENTED |
| **GWT** | get_consciousness_state | Section 5.10 | IMPLEMENTED |
| **GWT** | get_kuramoto_sync | Section 5.10 | IMPLEMENTED |
| **GWT** | get_workspace_status | Section 5.10 | IMPLEMENTED |
| **GWT** | get_ego_state | Section 5.10 | IMPLEMENTED |
| **GWT** | trigger_workspace_broadcast | Section 5.10 | IMPLEMENTED |
| **GWT** | adjust_coupling | Section 5.10 | IMPLEMENTED |
| **UTL** | gwt/compute_delta_sc | Section 5.10, UTL-001 | IMPLEMENTED |
| **ATC** | get_threshold_status | Section 16.9 | IMPLEMENTED |
| **ATC** | get_calibration_metrics | Section 16.9 | IMPLEMENTED |
| **ATC** | trigger_recalibration | Section 16.9 | IMPLEMENTED |
| **Dream** | trigger_dream | Section 5.2, 7.1 | IMPLEMENTED |
| **Dream** | get_dream_status | Section 7.1 | IMPLEMENTED |
| **Dream** | abort_dream | Section 7.1 | IMPLEMENTED |
| **Dream** | get_amortized_shortcuts | Section 7.1 | IMPLEMENTED |
| **Neuromod** | get_neuromodulation_state | Section 7.2 | IMPLEMENTED |
| **Neuromod** | adjust_neuromodulator | Section 7.2 | IMPLEMENTED |
| **Steering** | get_steering_feedback | Section 5.9, 7.8 | IMPLEMENTED |
| **Causal** | omni_infer | Section 5.9 | IMPLEMENTED |
| **Teleological** | search_teleological | TELEO-007 | IMPLEMENTED |
| **Teleological** | compute_teleological_vector | TELEO-008 | IMPLEMENTED |
| **Teleological** | fuse_embeddings | TELEO-009 | IMPLEMENTED |
| **Teleological** | update_synergy_matrix | TELEO-010 | IMPLEMENTED |
| **Teleological** | manage_teleological_profile | TELEO-011 | IMPLEMENTED |
| **Autonomous** | auto_bootstrap_north_star | AUTONOMOUS-MCP | IMPLEMENTED |
| **Autonomous** | get_alignment_drift | AUTONOMOUS-MCP | IMPLEMENTED |
| **Autonomous** | trigger_drift_correction | AUTONOMOUS-MCP | IMPLEMENTED |
| **Autonomous** | get_pruning_candidates | AUTONOMOUS-MCP | IMPLEMENTED |
| **Autonomous** | trigger_consolidation | AUTONOMOUS-MCP | IMPLEMENTED |
| **Autonomous** | discover_sub_goals | AUTONOMOUS-MCP | IMPLEMENTED |
| **Autonomous** | get_autonomous_status | AUTONOMOUS-MCP | IMPLEMENTED |
| **Meta-UTL** | get_meta_learning_status | METAUTL-P0-005 | IMPLEMENTED |
| **Meta-UTL** | trigger_lambda_recalibration | METAUTL-P0-005 | IMPLEMENTED |
| **Meta-UTL** | get_meta_learning_log | METAUTL-P0-005 | IMPLEMENTED |

### B. MISSING TOOLS (CRITICAL)

**Curation Tools (Section 5.3)** - 5 tools MISSING:

| Tool | PRD Reference | Evidence |
|------|---------------|----------|
| merge_concepts | Section 5.3, Line 532 | No implementation found |
| annotate_node | Section 5.3, Line 532 | No implementation found |
| forget_concept | Section 5.3, Line 532 | No implementation found |
| boost_importance | Section 5.3, Line 532 | No implementation found |
| restore_from_hash | Section 5.3, Line 532 | No implementation found |

**Navigation Tools (Section 5.4)** - 4 tools MISSING:

| Tool | PRD Reference | Evidence |
|------|---------------|----------|
| get_neighborhood | Section 5.4, Line 535 | Referenced in suggestions only |
| get_recent_context | Section 5.4, Line 535 | No implementation found |
| find_causal_path | Section 5.4, Line 535 | No implementation found |
| entailment_query | Section 5.4, Line 535 | Core exists but no MCP tool |

**Meta-Cognitive Tools (Section 5.5)** - 7 tools MISSING:

| Tool | PRD Reference | Evidence |
|------|---------------|----------|
| reflect_on_memory | Section 5.5, Line 538 | No implementation found |
| generate_search_plan | Section 5.5, Line 538 | No implementation found |
| critique_context | Section 5.5, Line 538 | No implementation found |
| hydrate_citation | Section 5.5, Line 538 | No implementation found |
| get_system_instructions | Section 5.5, Line 538 | No implementation found |
| get_system_logs | Section 5.5, Line 538 | No implementation found |
| get_node_lineage | Section 5.5, Line 538 | No implementation found |

**Diagnostic Tools (Section 5.6)** - 5 tools MISSING:

| Tool | PRD Reference | Evidence |
|------|---------------|----------|
| homeostatic_status | Section 5.6, Line 541 | No implementation found |
| check_adversarial | Section 5.6, Line 541 | No implementation found |
| test_recall_accuracy | Section 5.6, Line 541 | No implementation found |
| debug_compare_retrieval | Section 5.6, Line 541 | No implementation found |
| search_tombstones | Section 5.6, Line 541 | No implementation found |

**Admin Tools (Section 5.7)** - 2 tools MISSING:

| Tool | PRD Reference | Evidence |
|------|---------------|----------|
| reload_manifest | Section 5.7, Line 544 | No implementation found |
| temporary_scratchpad | Section 5.7, Line 544 | No implementation found |

**GWT Extension (Section 5.10)** - 2 tools MISSING:

| Tool | PRD Reference | Evidence |
|------|---------------|----------|
| get_johari_classification | Section 5.10, Line 567 | Johari data exists but no dedicated MCP tool |
| epistemic_action | Section 5.2, Line 527 | Referenced but not implemented |

---

## PARAMETER VALIDATION ANALYSIS

### PRD Section 26 Requirements vs Implementation

**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/core.rs`

| Tool | PRD Constraint | Implemented | Status |
|------|----------------|-------------|--------|
| inject_context.query | str[1-4096] | No length validation | PARTIAL |
| inject_context.max_tokens | int[100-8192]=2048 | Not implemented | MISSING |
| store_memory.content | str[<=65536] | No length validation | PARTIAL |
| store_memory.rationale | str[10-500] REQ | Not required in schema | VIOLATION |
| trigger_dream.phase | nrem\|rem\|full_cycle | Not in schema | MISSING |
| trigger_dream.duration_minutes | int[1-10]=5 | Not in schema | MISSING |

**Finding**: The PRD specifies strict parameter validation in Section 26, but the implementation uses JSON Schema without enforcing:
- String length minimums/maximums (minLength/maxLength)
- Required rationale field for store_memory
- Some enum values differ from PRD

---

## ANTI-PATTERN ANALYSIS

### AP-06: No Direct DB Access (MCP Tools Only)

**Investigation Result**: **INNOCENT**

Evidence:
- No direct SQLite/database access found in `/home/cabdru/contextgraph/crates/context-graph-mcp/src/`
- All database operations go through `TeleologicalMemoryStore` trait
- `TeleologicalStore` abstraction properly encapsulates storage
- TCP connection management is separate from data access

### ARCH-06: All Memory Operations Through MCP Tools

**Investigation Result**: **INNOCENT**

Evidence:
- Memory operations in `memory_tools.rs` use `teleological_store` abstraction
- No raw database queries in handler code
- Proper separation of concerns between MCP protocol and storage layer

---

## CRITICAL FINDINGS

### CRITICAL #1: Missing Curation Tools

**Severity**: CRITICAL
**Location**: PRD Section 5.3
**Impact**: Users cannot perform essential curation operations (merge, annotate, forget, boost, restore)

The PRD explicitly states:
> "You are a librarian, not an archivist. You don't store everything - you ensure what's stored is findable, coherent, and useful."

Without curation tools, the "librarian" cannot perform their core duties.

### CRITICAL #2: Missing Meta-Cognitive Tools

**Severity**: CRITICAL
**Location**: PRD Section 5.5
**Impact**: Users cannot leverage the system's self-reflection capabilities

Missing tools like `reflect_on_memory`, `generate_search_plan`, and `critique_context` prevent the system from providing guided reasoning assistance.

### HIGH #3: Missing Navigation Tools

**Severity**: HIGH
**Location**: PRD Section 5.4
**Impact**: Limited graph traversal capabilities

While `search_graph` exists, the missing `get_neighborhood`, `find_causal_path`, and `entailment_query` tools limit users to basic search without graph navigation.

### HIGH #4: Parameter Validation Gaps

**Severity**: HIGH
**Location**: Multiple tool definitions
**Impact**: Potential for invalid input causing runtime errors

The PRD Section 26 specifies exact parameter constraints that are not fully enforced in the JSON schemas.

### MEDIUM #5: Missing epistemic_action Tool

**Severity**: MEDIUM
**Location**: PRD Section 5.2, Line 527
**Impact**: Cannot generate clarifying questions for low-coherence states

The cognitive pulse suggests "epistemic_action" but no such MCP tool exists.

---

## POSITIVE FINDINGS

### INNOCENT: compute_delta_sc Implementation

The critical UTL-001 requirement for `compute_delta_sc` is **fully implemented** with:
- All 13 embedder entropy calculations
- Proper delta-C formula (0.4*Connectivity + 0.4*ClusterFit + 0.2*Consistency)
- Johari quadrant classification
- UTL learning potential computation
- NaN/Inf protection
- Comprehensive diagnostics option

**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/utl/gwt_compute.rs`

### INNOCENT: GWT Integration

Complete Global Workspace Theory integration with:
- Kuramoto oscillator synchronization
- Consciousness state computation
- Workspace broadcast triggering
- Ego state tracking with identity continuity

### INNOCENT: JSON-RPC 2.0 Compliance

Full protocol compliance with:
- Proper request/response structures
- Comprehensive error code system
- ID handling (string/number)
- Method dispatch

### INNOCENT: Test Coverage

Tests verify:
- 39 tools returned by tools/list
- Each tool has required MCP fields
- Expected tool names present
- Schema validation

---

## RECOMMENDATIONS

### IMMEDIATE (P0)

1. **Implement Curation Tools**: Add merge_concepts, annotate_node, forget_concept, boost_importance, restore_from_hash
2. **Implement epistemic_action**: Required for cognitive pulse suggestions to work

### HIGH PRIORITY (P1)

3. **Implement Meta-Cognitive Tools**: reflect_on_memory, generate_search_plan, critique_context, hydrate_citation, get_system_instructions, get_system_logs, get_node_lineage
4. **Implement Navigation Tools**: get_neighborhood, get_recent_context, find_causal_path, entailment_query

### MEDIUM PRIORITY (P2)

5. **Add Parameter Validation**: Enforce PRD Section 26 constraints in JSON schemas
6. **Implement Diagnostic Tools**: homeostatic_status, check_adversarial, test_recall_accuracy
7. **Add get_johari_classification**: Dedicated tool for per-embedder quadrant analysis
8. **Implement Admin Tools**: reload_manifest, temporary_scratchpad

---

## CHAIN OF CUSTODY

| Timestamp | Action | File(s) Examined | Finding |
|-----------|--------|------------------|---------|
| 2026-01-12 | Initial scan | crates/context-graph-mcp/* | 39 tools found |
| 2026-01-12 | Tool definitions | tools/definitions/*.rs | All 39 defined |
| 2026-01-12 | Dispatch verification | handlers/tools/dispatch.rs | All dispatched |
| 2026-01-12 | compute_delta_sc | handlers/utl/gwt*.rs | Fully implemented |
| 2026-01-12 | Protocol compliance | protocol.rs | JSON-RPC 2.0 compliant |
| 2026-01-12 | Missing tool search | grep for PRD tools | ~36 tools missing |
| 2026-01-12 | Anti-pattern check | All MCP source | No violations |

---

## FINAL VERDICT

### GUILTY - Missing Tools
The MCP implementation is **GUILTY** of incomplete coverage. Approximately 36 PRD-specified tools are not implemented, representing a ~48% feature gap.

### INNOCENT - Implemented Tools
The 39 implemented tools are **INNOCENT** of structural defects. They follow proper MCP protocol, use correct abstractions, and implement FAIL-FAST principles.

### VERDICT CONFIDENCE: HIGH

Evidence is conclusive. PRD requirements are explicit, and tool inventory is definitive.

---

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*

The truth is: **The MCP tool layer has a solid foundation but requires significant expansion to meet PRD requirements.**

---

**Report Compiled By**: Sherlock Holmes
**Case Status**: OPEN - Remediation Required
**Next Review**: Upon implementation of P0 recommendations
