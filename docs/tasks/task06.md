# Task 06: Implement Topic and Curation Tool Handlers

## Metadata
- **Task ID**: TASK-GAP-006
- **Phase**: 2 (MCP Infrastructure)
- **Priority**: High
- **Complexity**: High
- **Dependencies**: task05 (TASK-GAP-005 - DTOs COMPLETE)
- **Last Audit**: 2026-01-18

## Current Codebase State (VERIFIED)

### What EXISTS (Do NOT recreate):
| File | Status | Contains |
|------|--------|----------|
| `crates/context-graph-mcp/src/handlers/tools/topic_dtos.rs` | **EXISTS** (32.6KB) | All 4 request/response DTOs with validation |
| `crates/context-graph-mcp/src/handlers/tools/curation_dtos.rs` | **EXISTS** (20.6KB) | ForgetConceptRequest/Response, BoostImportanceRequest/Response |
| `crates/context-graph-mcp/src/handlers/tools/mod.rs` | **EXISTS** | Exports `curation_dtos` and `topic_dtos` |
| `crates/context-graph-mcp/src/handlers/tools/helpers.rs` | **EXISTS** | `tool_result_with_pulse()`, `tool_error_with_pulse()` |
| `crates/context-graph-mcp/src/handlers/tools/dispatch.rs` | **EXISTS** | Routes 6 tools (INJECT_CONTEXT, STORE_MEMORY, GET_MEMETIC_STATUS, SEARCH_GRAPH, TRIGGER_CONSOLIDATION, MERGE_CONCEPTS) |
| `crates/context-graph-mcp/src/tools/names.rs` | **EXISTS** | All tool name constants (GET_TOPIC_PORTFOLIO, GET_TOPIC_STABILITY, DETECT_TOPICS, GET_DIVERGENCE_ALERTS, FORGET_CONCEPT, BOOST_IMPORTANCE marked `#[allow(dead_code)]`) |
| `crates/context-graph-core/src/clustering/topic.rs` | **EXISTS** | `Topic`, `TopicProfile`, `TopicStability`, `TopicPhase` |
| `crates/context-graph-core/src/clustering/stability.rs` | **EXISTS** | `TopicStabilityTracker`, `TopicSnapshot` |

### What is MISSING (Must create):
| File | Status | Must Contain |
|------|--------|--------------|
| `crates/context-graph-mcp/src/handlers/tools/topic_tools.rs` | **MISSING** | 4 handler methods |
| `crates/context-graph-mcp/src/handlers/tools/curation_tools.rs` | **MISSING** | 2 handler methods |

### What Needs MODIFICATION:
| File | Change Required |
|------|-----------------|
| `crates/context-graph-mcp/src/handlers/tools/mod.rs` | Add `mod topic_tools;` and `mod curation_tools;` |
| `crates/context-graph-mcp/src/handlers/tools/dispatch.rs` | Add 6 new tool routes |

## Objective

Implement 6 MCP tool handlers that use the existing DTOs and core types. NO STUB IMPLEMENTATIONS - all handlers must return real computed data or FAIL FAST with proper error codes.

## Constitutional Compliance (MANDATORY)

| Rule | Requirement | Verification |
|------|-------------|--------------|
| **ARCH-09** | Topic threshold = weighted_agreement >= 2.5 | Test: `TopicProfile::is_topic()` returns false for 2.4, true for 2.5 |
| **AP-60** | Temporal embedders (E2-E4) weight = 0.0 | Test: Profile with only E2,E3,E4 at 1.0 must have weighted_agreement = 0.0 |
| **AP-62** | Divergence uses SEMANTIC only (E1, E5, E6, E7, E10, E12, E13) | Test: Divergence alert generation excludes E2-E4, E8, E9, E11 |
| **AP-70** | Dream triggers: entropy > 0.7 AND churn > 0.5 | Test: `check_dream_trigger()` returns false for entropy=0.7, true for entropy=0.71 |
| **SEC-06** | Soft delete 30-day recovery | Test: `forget_concept` with soft_delete=true sets recovery_expires_at 30 days out |

## Implementation Requirements

### File 1: `topic_tools.rs`

**Path**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/topic_tools.rs`

**Required Handlers** (4 methods on `impl Handlers`):

#### 1. `call_get_topic_portfolio`
```rust
pub(crate) async fn call_get_topic_portfolio(
    &self,
    id: Option<JsonRpcId>,
    arguments: serde_json::Value,
) -> JsonRpcResponse
```

**Logic Flow**:
1. Parse `GetTopicPortfolioRequest` from arguments
2. Call `request.validate()` - FAIL FAST if invalid
3. Get memory count: `self.teleological_store.count().await`
4. Compute tier using `TopicPortfolioResponse::tier_for_memory_count(count)`
5. If tier < 2 (< 3 memories): Return empty portfolio with tier info
6. Get UTL status: `self.utl_processor.get_status()`
7. Build `TopicPortfolioResponse` with:
   - `topics`: Empty vec (clustering integration later)
   - `stability`: From UTL status (entropy, coherence mapped)
   - `total_topics`: 0 (placeholder until clustering)
   - `tier`: Computed tier
8. Return via `self.tool_result_with_pulse(id, response)`

**Error Codes**:
- `INVALID_PARAMS (-32602)`: Invalid format parameter
- `STORAGE_ERROR (-32004)`: Cannot get memory count

#### 2. `call_get_topic_stability`
```rust
pub(crate) async fn call_get_topic_stability(
    &self,
    id: Option<JsonRpcId>,
    arguments: serde_json::Value,
) -> JsonRpcResponse
```

**Logic Flow**:
1. Parse `GetTopicStabilityRequest` from arguments
2. Call `request.validate()` - FAIL FAST if invalid (hours must be 1-168)
3. Get UTL status for entropy
4. Compute `dream_trigger_ready` per AP-70: `entropy > 0.7 AND churn > 0.5`
5. Build `TopicStabilityResponse` with churn_history (empty initially), drift_history, metrics
6. Return via `self.tool_result_with_pulse(id, response)`

**Error Codes**:
- `INVALID_PARAMS (-32602)`: hours out of range

#### 3. `call_detect_topics`
```rust
pub(crate) async fn call_detect_topics(
    &self,
    id: Option<JsonRpcId>,
    arguments: serde_json::Value,
) -> JsonRpcResponse
```

**Logic Flow**:
1. Parse `DetectTopicsRequest` from arguments
2. Get memory count: `self.teleological_store.count().await`
3. FAIL FAST if count < 3: Return error `-32021` (INSUFFICIENT_DATA)
4. Get processing start time
5. Build `DetectTopicsResponse` with processing_time_ms
6. Return via `self.tool_result_with_pulse(id, response)`

**Error Codes**:
- `INVALID_PARAMS (-32602)`: Parse error
- `-32021` (INSUFFICIENT_DATA): Need >= 3 memories

#### 4. `call_get_divergence_alerts`
```rust
pub(crate) async fn call_get_divergence_alerts(
    &self,
    id: Option<JsonRpcId>,
    arguments: serde_json::Value,
) -> JsonRpcResponse
```

**Logic Flow**:
1. Parse `GetDivergenceAlertsRequest` from arguments
2. Call `request.validate()` - FAIL FAST if invalid (lookback_hours must be 1-48)
3. **CRITICAL (AP-62)**: Only compute divergence for SEMANTIC embedders
4. Build `DivergenceAlertResponse` with alerts (empty initially), severity_level
5. Return via `self.tool_result_with_pulse(id, response)`

**SEMANTIC Embedders Array** (use for divergence):
```rust
const SEMANTIC_EMBEDDERS: [usize; 7] = [0, 4, 5, 6, 9, 11, 12]; // E1, E5, E6, E7, E10, E12, E13
```

### File 2: `curation_tools.rs`

**Path**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/curation_tools.rs`

**Required Handlers** (2 methods on `impl Handlers`):

#### 1. `call_forget_concept`
```rust
pub(crate) async fn call_forget_concept(
    &self,
    id: Option<JsonRpcId>,
    arguments: serde_json::Value,
) -> JsonRpcResponse
```

**Logic Flow**:
1. Parse `ForgetConceptRequest` from arguments
2. Call `request.validate()` and `request.parse_node_id()` - FAIL FAST if invalid UUID
3. Check if node exists: `self.teleological_store.retrieve(uuid).await`
4. FAIL FAST if not found: Return `NODE_NOT_FOUND (-32002)`
5. Delete: `self.teleological_store.delete(uuid, request.soft_delete).await`
6. Build `ForgetConceptResponse` with:
   - If soft_delete: `recovery_expires_at = Utc::now() + Duration::days(30)`
   - If hard delete: `permanently_deleted = true`
7. Return via `self.tool_result_with_pulse(id, response)`

**Error Codes**:
- `INVALID_PARAMS (-32602)`: Invalid UUID format
- `NODE_NOT_FOUND (-32002)`: Node doesn't exist

#### 2. `call_boost_importance`
```rust
pub(crate) async fn call_boost_importance(
    &self,
    id: Option<JsonRpcId>,
    arguments: serde_json::Value,
) -> JsonRpcResponse
```

**Logic Flow**:
1. Parse `BoostImportanceRequest` from arguments
2. Call `request.validate()` and `request.parse_node_id()` - FAIL FAST if invalid
3. Retrieve fingerprint: `self.teleological_store.retrieve(uuid).await`
4. FAIL FAST if not found: Return `NODE_NOT_FOUND (-32002)`
5. Compute new importance: `(old + delta).clamp(0.0, 1.0)`
6. Update fingerprint with new importance
7. Store updated: `self.teleological_store.update(fingerprint).await`
8. Build `BoostImportanceResponse` with old/new/final importance
9. Return via `self.tool_result_with_pulse(id, response)`

**Error Codes**:
- `INVALID_PARAMS (-32602)`: Invalid UUID or delta out of range
- `NODE_NOT_FOUND (-32002)`: Node doesn't exist
- `STORAGE_ERROR (-32004)`: Update failed

### File 3: Update `mod.rs`

**Path**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/mod.rs`

**Current Content**:
```rust
//! MCP tool call handlers.
mod consolidation;
mod dispatch;
mod helpers;
mod memory_tools;
mod status_tools;

pub mod curation_dtos;
pub mod topic_dtos;
```

**Change**: Add two module declarations after `status_tools`:
```rust
mod topic_tools;
mod curation_tools;
```

### File 4: Update `dispatch.rs`

**Path**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/dispatch.rs`

**Current match arms** (lines 64-85):
- tool_names::INJECT_CONTEXT
- tool_names::STORE_MEMORY
- tool_names::GET_MEMETIC_STATUS
- tool_names::SEARCH_GRAPH
- tool_names::TRIGGER_CONSOLIDATION
- tool_names::MERGE_CONCEPTS

**Add after line 77** (after MERGE_CONCEPTS):
```rust
// ========== TOPIC TOOLS (PRD Section 10.2) ==========
tool_names::GET_TOPIC_PORTFOLIO => self.call_get_topic_portfolio(id, arguments).await,
tool_names::GET_TOPIC_STABILITY => self.call_get_topic_stability(id, arguments).await,
tool_names::DETECT_TOPICS => self.call_detect_topics(id, arguments).await,
tool_names::GET_DIVERGENCE_ALERTS => self.call_get_divergence_alerts(id, arguments).await,

// ========== CURATION TOOLS (PRD Section 10.3) ==========
tool_names::FORGET_CONCEPT => self.call_forget_concept(id, arguments).await,
tool_names::BOOST_IMPORTANCE => self.call_boost_importance(id, arguments).await,
```

## Required Imports

### topic_tools.rs
```rust
use std::time::Instant;
use serde_json::json;
use tracing::{debug, warn};
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};
use super::super::Handlers;
use super::topic_dtos::{
    GetTopicPortfolioRequest, GetTopicStabilityRequest, DetectTopicsRequest,
    GetDivergenceAlertsRequest, TopicPortfolioResponse, TopicStabilityResponse,
    DetectTopicsResponse, DivergenceAlertResponse, StabilityMetricsSummary,
};
```

### curation_tools.rs
```rust
use chrono::{Duration, Utc};
use serde_json::json;
use tracing::{debug, warn};
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};
use super::super::Handlers;
use super::curation_dtos::{
    ForgetConceptRequest, ForgetConceptResponse,
    BoostImportanceRequest, BoostImportanceResponse,
};
```

## Full State Verification Protocol

### Source of Truth
- **Memory Store**: `TeleologicalMemoryStore` accessed via `self.teleological_store`
- **RocksDB Location**: `data/rocksdb/` (dev mode)
- **Verification Method**: After each operation, perform separate READ to confirm state

### Verification Steps After Implementation

#### Step 1: Compilation Check
```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-mcp 2>&1 | head -50
cargo clippy -p context-graph-mcp -- -D warnings 2>&1 | head -50
```

**Expected Output**: No errors, no warnings

#### Step 2: Handler Method Count
```bash
grep -c "pub(crate) async fn call_" crates/context-graph-mcp/src/handlers/tools/topic_tools.rs
# Expected: 4

grep -c "pub(crate) async fn call_" crates/context-graph-mcp/src/handlers/tools/curation_tools.rs
# Expected: 2
```

#### Step 3: Module Exports
```bash
grep "mod topic_tools" crates/context-graph-mcp/src/handlers/tools/mod.rs
grep "mod curation_tools" crates/context-graph-mcp/src/handlers/tools/mod.rs
# Both should exist
```

#### Step 4: Dispatch Routes
```bash
grep -c "tool_names::GET_TOPIC_PORTFOLIO\|tool_names::GET_TOPIC_STABILITY\|tool_names::DETECT_TOPICS\|tool_names::GET_DIVERGENCE_ALERTS\|tool_names::FORGET_CONCEPT\|tool_names::BOOST_IMPORTANCE" crates/context-graph-mcp/src/handlers/tools/dispatch.rs
# Expected: 6
```

#### Step 5: Dead Code Cleanup
After implementation, update `crates/context-graph-mcp/src/tools/names.rs`:
Remove `#[allow(dead_code)]` from the 6 newly-used constants.

## Manual Testing Protocol

### Test Environment Setup
```bash
# Ensure RocksDB directory exists
mkdir -p data/rocksdb

# Build in dev mode
cargo build -p context-graph-mcp
```

### Test 1: get_topic_portfolio (Happy Path)
**Input**:
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_topic_portfolio", "arguments": {"format": "standard"}}}
```

**Expected Output Verification**:
- Response contains `tier` field (0-6 based on memory count)
- Response contains `_cognitive_pulse` object
- If tier < 2: `total_topics` = 0, `topics` = []
- `stability.entropy` is float 0.0-1.0

**State Verification**:
```bash
# Check memory count in store
# The tier in response must match:
# 0 memories -> tier 0
# 1-2 memories -> tier 1
# 3-9 memories -> tier 2
```

### Test 2: get_topic_portfolio (Edge Case - Invalid Format)
**Input**:
```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_topic_portfolio", "arguments": {"format": "invalid"}}}
```

**Expected Output**:
- `isError: true`
- Error message contains "Invalid format"

### Test 3: get_topic_stability (Happy Path)
**Input**:
```json
{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_topic_stability", "arguments": {"hours": 6}}}
```

**Expected Output Verification**:
- Response contains `average_churn`, `average_drift`, `entropy`
- `dream_trigger_ready` is boolean
- `high_entropy_duration_secs` >= 0

### Test 4: get_topic_stability (Edge Case - Hours = 0)
**Input**:
```json
{"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "get_topic_stability", "arguments": {"hours": 0}}}
```

**Expected Output**:
- `isError: true`
- Error message contains "hours must be at least 1"

### Test 5: detect_topics (Edge Case - Insufficient Memories)
**Input** (when store has < 3 memories):
```json
{"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "detect_topics", "arguments": {"force": true}}}
```

**Expected Output**:
- `isError: true`
- Error code: -32021
- Message contains "Need >= 3 memories"

### Test 6: forget_concept (Invalid UUID)
**Input**:
```json
{"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "forget_concept", "arguments": {"node_id": "not-a-uuid", "soft_delete": true}}}
```

**Expected Output**:
- `isError: true`
- Error code: -32602
- Message contains "Invalid UUID format"

### Test 7: boost_importance (Node Not Found)
**Input**:
```json
{"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "boost_importance", "arguments": {"node_id": "00000000-0000-0000-0000-000000000000", "delta": 0.1}}}
```

**Expected Output**:
- `isError: true`
- Error code: -32002
- Message contains "not found"

## Edge Cases to Test

| Scenario | Input | Expected Behavior |
|----------|-------|-------------------|
| Empty arguments | `{}` | Use defaults, no error |
| Max hours | `{"hours": 168}` | Accept (max allowed) |
| Hours > 168 | `{"hours": 169}` | FAIL FAST with INVALID_PARAMS |
| Delta = -1.0 | `{"delta": -1.0}` | Accept (min allowed) |
| Delta < -1.0 | `{"delta": -1.1}` | Clamp to -1.0 |
| Soft delete | `{"soft_delete": true}` | Set recovery_expires_at |
| Hard delete | `{"soft_delete": false}` | permanently_deleted = true |

## Definition of Done

- [ ] File `topic_tools.rs` exists at exact path with 4 handler methods
- [ ] File `curation_tools.rs` exists at exact path with 2 handler methods
- [ ] `mod.rs` declares both modules
- [ ] `dispatch.rs` routes all 6 new tools
- [ ] All handlers use `self.tool_result_with_pulse()` or `self.tool_error_with_pulse()`
- [ ] All handlers call `.validate()` on request DTOs before processing
- [ ] `#[allow(dead_code)]` removed from 6 tool name constants
- [ ] `cargo check -p context-graph-mcp` passes with ZERO errors
- [ ] `cargo clippy -p context-graph-mcp -- -D warnings` passes with ZERO warnings
- [ ] Manual tests 1-7 all pass with expected outputs
- [ ] State verification confirms data persistence after operations

## Anti-Patterns to AVOID

| Anti-Pattern | Why It's Wrong | Correct Approach |
|--------------|----------------|------------------|
| Returning stub data | Violates FAIL FAST | Return empty valid response OR error |
| Using `.unwrap()` | Can panic in production | Use `?` or `.ok_or()` |
| Silencing errors | Hides bugs | Log with `warn!()` then return error |
| Skipping validation | Invalid data corrupts state | Always call `.validate()` |
| Mock data in tests | Tests pass but system broken | Use real store operations |
| Backwards compatibility shims | Creates tech debt | FAIL FAST, fix caller |

## References

- Constitution: `/home/cabdru/contextgraph/CLAUDE.md` (Topic System section)
- DTOs: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/topic_dtos.rs`
- Pattern Reference: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/memory_tools.rs`
- Helper Functions: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/helpers.rs`
- Core Types: `/home/cabdru/contextgraph/crates/context-graph-core/src/clustering/topic.rs`
- Error Codes: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/protocol.rs` (lines 77-255)
