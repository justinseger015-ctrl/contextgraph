# Forensic Investigation Report: gwt_inject

## Case File ID: HOLMES-GWT-003
## Investigation Date: 2026-01-14
## Verdict: TOOL NOT FOUND - No such tool exists

---

## 1. CRITICAL FINDING

**The tool `gwt_inject` DOES NOT EXIST in the codebase.**

After exhaustive forensic search of:
- `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/`
- `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/`
- All 65 files containing "gwt" references

**ZERO matches were found for `gwt_inject`.**

---

## 2. HYPOTHESIS ANALYSIS

### 2.1 Possible User Intent

The user may have intended one of these existing tools:

| Possible Intent | Actual Tool | Match Likelihood |
|-----------------|-------------|------------------|
| Inject memory into Global Workspace | `trigger_workspace_broadcast` | HIGH |
| Inject context with UTL processing | `inject_context` | HIGH |
| Compute GWT delta S/C | `gwt/compute_delta_sc` | MEDIUM |
| Store memory directly | `store_memory` | LOW |

### 2.2 Evidence of Non-Existence

**Search 1**: Pattern `gwt_inject` across all MCP files
- Result: **0 files found**

**Search 2**: All GWT-related tool names
- File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs`
- GWT tools found:
  - `get_consciousness_state`
  - `get_kuramoto_sync`
  - `get_workspace_status`
  - `get_ego_state`
  - `trigger_workspace_broadcast`
  - `adjust_coupling`
  - `gwt/compute_delta_sc`
  - `get_coherence_state`
  - `get_identity_continuity`
  - `get_kuramoto_state`

**No `gwt_inject` in the list.**

---

## 3. CLOSEST EQUIVALENT: trigger_workspace_broadcast

Since `gwt_inject` does not exist, I will document the closest functional equivalent: `trigger_workspace_broadcast`.

### 3.1 What does trigger_workspace_broadcast do?

`trigger_workspace_broadcast` forces a specific memory into the Global Workspace competition. It implements the Winner-Take-All (WTA) selection mechanism described in PRD Section 2.5.3.

### 3.2 Why does it exist?

In Global Workspace Theory, only **phase-locked memories** can be "perceived" by the system. This tool allows an agent to:
1. Force a memory into workspace competition
2. Trigger conscious attention to a specific memory
3. Override automatic selection when needed
4. Test workspace dynamics

### 3.3 Tool Definition

**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/gwt.rs`
**Lines**: 88-124

```rust
// trigger_workspace_broadcast - Trigger WTA selection (TASK-GWT-001)
ToolDefinition::new(
    "trigger_workspace_broadcast",
    "Trigger winner-take-all workspace broadcast with a specific memory. \
     Forces memory into workspace competition. Requires write lock on workspace. \
     Requires GWT providers to be initialized via with_gwt() constructor.",
    json!({
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "format": "uuid",
                "description": "UUID of memory to broadcast into workspace"
            },
            "importance": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.8,
                "description": "Importance score for the memory [0.0, 1.0]"
            },
            "alignment": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.8,
                "description": "North star alignment score [0.0, 1.0]"
            },
            "force": {
                "type": "boolean",
                "default": false,
                "description": "Force broadcast even if below coherence threshold"
            }
        },
        "required": ["memory_id"]
    }),
),
```

### 3.4 Handler Implementation

**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/gwt_workspace.rs`
**Lines**: 76-229

Key implementation details:
1. Validates workspace and kuramoto providers exist
2. Parses memory_id (required UUID)
3. Gets current order parameter r from Kuramoto network
4. Checks coherence threshold (r >= 0.8 unless force=true)
5. Acquires write lock on workspace
6. Triggers WTA selection with candidates
7. Wires workspace entry to neuromodulation (dopamine increase)
8. Returns success status with selection details

### 3.5 Input Specification

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string (UUID) | YES | - | UUID of memory to broadcast |
| `importance` | number | NO | 0.8 | Importance score [0.0, 1.0] |
| `alignment` | number | NO | 0.8 | North Star alignment [0.0, 1.0] |
| `force` | boolean | NO | false | Force broadcast below threshold |

### 3.6 Output Specification

**Success Response:**
```json
{
  "result": {
    "success": true,
    "memory_id": "uuid-string",
    "new_r": 0.85,
    "was_selected": true,
    "is_broadcasting": true,
    "dopamine_triggered": 3.2,
    "_cognitive_pulse": {
      "entropy": 0.45,
      "coherence": 0.85,
      "suggested_action": "continue"
    }
  }
}
```

**Rejection Response (low r, no force):**
```json
{
  "result": {
    "success": false,
    "memory_id": "uuid-string",
    "new_r": 0.65,
    "was_selected": false,
    "reason": "Order parameter r=0.650 below coherence threshold 0.8. Use force=true to override."
  }
}
```

---

## 4. PURPOSE IN PRD END GOAL

### 4.1 PRD Reference: Section 2.5.3 - Global Broadcast Architecture

```
GlobalWorkspace:
  active_memory: Option<MemoryId>      -- Currently "conscious" memory
  coherence_threshold: 0.8             -- Minimum r for broadcast
  broadcast_duration: 100ms            -- How long memory stays active
  competing_memories: PriorityQueue    -- Sorted by r x importance
```

**Broadcast Selection Algorithm:**
```
1. Compute r for all candidate memories
2. Filter: candidates where r >= coherence_threshold
3. Rank: score = r x importance x north_star_alignment
4. Select: top-1 becomes active_memory
5. Broadcast: active_memory visible to all subsystems
6. Inhibit: losing candidates get dopamine reduction
```

### 4.2 PRD Reference: Section 2.5.3 - Global Broadcast Events

| Event | Trigger | Effect |
|-------|---------|--------|
| `memory_enters_workspace` | r crosses 0.8 upward | Dopamine += 0.2 |
| `memory_exits_workspace` | r drops below 0.7 | Log for dream replay |
| `workspace_conflict` | Two memories r > 0.8 | Trigger critique_context |
| `workspace_empty` | No memory r > 0.8 for 5s | Trigger epistemic_action |

---

## 5. AI AGENT USAGE

### 5.1 Typical Workflow

```
Agent wants to "inject" a memory into conscious awareness

1. First, ensure memory exists (via store_memory or inject_context)
2. Get memory's fingerprint_id from storage response
3. Call trigger_workspace_broadcast with:
   - memory_id: the fingerprint UUID
   - importance: how critical this memory is
   - alignment: alignment to current goal
4. Check if was_selected == true
5. If not selected, consider force=true for critical memories
```

### 5.2 Example Request

```json
{
  "method": "tools/call",
  "params": {
    "name": "trigger_workspace_broadcast",
    "arguments": {
      "memory_id": "550e8400-e29b-41d4-a716-446655440000",
      "importance": 0.9,
      "alignment": 0.85,
      "force": false
    }
  }
}
```

### 5.3 Integration with inject_context

For a complete "GWT inject" workflow:

```
1. inject_context(content="JWT authentication flow", rationale="User asked about auth")
   Response: { fingerprintId: "uuid-123", ... }

2. trigger_workspace_broadcast(memory_id="uuid-123", importance=0.9)
   Response: { was_selected: true, is_broadcasting: true, ... }

3. Memory is now "conscious" - other subsystems can access it
```

---

## 6. ERROR HANDLING

### 6.1 Error Codes

| Error | Code | Cause |
|-------|------|-------|
| GWT not initialized | -32020 | Workspace provider missing |
| GWT not initialized | -32020 | Kuramoto network missing |
| Invalid params | -32602 | Missing memory_id |
| Invalid params | -32602 | Invalid UUID format |
| Workspace error | -32021 | WTA selection failed |

### 6.2 Graceful Degradation

When r < 0.8 and force=false:
- Tool returns success: false (not an error)
- Includes reason explaining why
- Suggests using force=true to override

---

## 7. VERDICT

**TOOL NOT FOUND** - `gwt_inject` does not exist in the codebase.

**Closest Equivalent**: `trigger_workspace_broadcast`
- This tool "injects" a memory into the Global Workspace competition
- Combined with `inject_context`, it provides the full "GWT inject" functionality

**Recommendation for User:**
1. Use `inject_context` to store content with UTL processing
2. Use `trigger_workspace_broadcast` to force it into conscious awareness
3. This two-step process achieves what `gwt_inject` would conceptually do

**Confidence Level:** HIGH (exhaustive search conducted)

---

## SHERLOCK HOLMES CASE CLOSED

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth. The tool `gwt_inject` simply does not exist. However, the two-step combination of `inject_context` followed by `trigger_workspace_broadcast` achieves the semantic intent of 'injecting into the Global Workspace'. The architecture intentionally separates storage from consciousness selection."*
