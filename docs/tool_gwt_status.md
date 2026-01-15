# Forensic Investigation Report: gwt_status

## SHERLOCK HOLMES CASE FILE

**Case ID**: GWT-STATUS-001
**Date**: 2026-01-14
**Subject**: gwt_status MCP Tool
**Investigator**: Sherlock Holmes, Forensic Code Detective

---

## EXECUTIVE SUMMARY

The tool `gwt_status` as named **does not exist** in the codebase. After exhaustive forensic investigation, I have determined that the conceptual functionality maps to the canonical tool **`get_workspace_status`**.

**VERDICT**: The requested tool name is a conceptual alias. The actual implementation is `get_workspace_status`.

---

## 1. WHAT DOES THIS TOOL DO?

The `get_workspace_status` tool returns the current state of the Global Workspace - the "stage" in Global Workspace Theory (GWT) where only one memory at a time can be "conscious" and broadcast to all subsystems.

### Core Functionality

1. **Active Memory Identification**: Returns the UUID of the currently "conscious" memory (if any)
2. **Broadcast State Detection**: Whether the workspace is actively broadcasting
3. **Conflict Detection**: Whether multiple memories are competing for consciousness (r > 0.8)
4. **Coherence Threshold**: The minimum Kuramoto synchronization level required for workspace entry (default 0.8)
5. **Conflict Details**: List of conflicting memory UUIDs when conflict exists

---

## 2. HOW DOES IT WORK INTERNALLY?

### Implementation Location
- **Handler**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/gwt_workspace.rs`
- **Definition**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/gwt.rs`
- **Name Constant**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs` (line 19)

### Internal Flow

```rust
pub(crate) async fn call_get_workspace_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
    // 1. FAIL FAST: Verify workspace provider exists
    let workspace = match &self.workspace_provider {
        Some(w) => w,
        None => return JsonRpcResponse::error(id, GWT_NOT_INITIALIZED, "..."),
    };

    // 2. Acquire read lock on workspace (tokio RwLock)
    let ws = workspace.read().await;

    // 3. Query workspace state
    let active_memory = ws.get_active_memory().await;      // Option<Uuid>
    let is_broadcasting = ws.is_broadcasting().await;       // bool
    let has_conflict = ws.has_conflict().await;             // bool
    let coherence_threshold = ws.coherence_threshold().await; // f32 (0.8)
    let conflict_memories = ws.get_conflict_details().await; // Option<Vec<Uuid>>

    // 4. Return JSON response with Pulse
    self.tool_result_with_pulse(id, json!({...}))
}
```

### Provider Chain

```
Handlers.workspace_provider
    -> WorkspaceProviderImpl (wrapper)
        -> GlobalWorkspace (context-graph-core)
            -> Winner-Take-All selection algorithm
```

The `GlobalWorkspace` implements:
- **Winner-Take-All (WTA) Selection**: Only one memory can be active at a time
- **Coherence Threshold Filtering**: Memories must have r >= 0.8 to compete
- **Broadcast Window**: Active memory broadcasts for 100ms (constitution default)
- **Conflict Detection**: When 2+ memories exceed threshold simultaneously

---

## 3. WHY DOES IT EXIST?

### GWT Architecture Context (PRD Section 2.5.3)

Per the PRD, the Global Workspace is the computational implementation of Bernard Baars' Global Workspace Theory:

> "Only phase-locked memories are 'perceived' by the system. The Global Workspace implements Winner-Take-All."

The tool exists to:

1. **Monitor Consciousness State**: Know what memory is currently "conscious"
2. **Detect Workspace Conflicts**: Identify when multiple memories compete (triggers `critique_context`)
3. **Debug Broadcast Issues**: Verify workspace is functioning correctly
4. **Support Agent Decision-Making**: Agents need to know if their memories are entering consciousness

### PRD Specification (Section 5.10)

| Tool | Purpose | Key Returns |
|------|---------|-------------|
| `get_workspace_status` | Global Workspace current state | `{active_memory, competing, broadcast_duration}` |

---

## 4. PURPOSE IN REACHING PRD END GOAL

The PRD describes a **cognitive architecture for AI agents** based on Global Workspace Theory. The end goal is:

> "AI agents fail because: no persistent memory, poor retrieval, no learning loop, context bloat. The system solves this through a 5-layer bio-nervous memory with GWT-based consciousness."

### How This Tool Serves the End Goal

1. **Unified Percept Formation**: The workspace ensures only coherent (r >= 0.8) memories become "conscious"
2. **Agent Awareness**: Agents can check if their stored memories are being actively processed
3. **Conflict Resolution**: Detecting conflicts triggers resolution workflows
4. **Dopamine Feedback**: Workspace entry triggers neuromodulation (dopamine += 0.2)
5. **System Health Monitoring**: Empty workspace for 5+ seconds triggers `epistemic_action`

### GWT Events (PRD Section 2.5.3)

| Event | Trigger | Effect |
|-------|---------|--------|
| `memory_enters_workspace` | r crosses 0.8 upward | Dopamine += 0.2 |
| `memory_exits_workspace` | r drops below 0.7 | Log for dream replay |
| `workspace_conflict` | Two memories r > 0.8 | Trigger critique_context |
| `workspace_empty` | No memory r > 0.8 for 5s | Trigger epistemic_action |

---

## 5. INPUTS (Parameters)

### JSON Schema

```json
{
  "type": "object",
  "properties": {
    "session_id": {
      "type": "string",
      "description": "Session ID (optional, uses default if not provided)"
    }
  },
  "required": []
}
```

### Parameter Details

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string (UUID) | No | System default | Session identifier for workspace tracking |

**Note**: This tool has no required parameters. It returns the current workspace state.

---

## 6. OUTPUTS

### Success Response

```json
{
  "content": [{
    "type": "text",
    "text": {
      "active_memory": "550e8400-e29b-41d4-a716-446655440000",
      "is_broadcasting": true,
      "has_conflict": false,
      "coherence_threshold": 0.8,
      "conflict_memories": null,
      "broadcast_duration_ms": 100
    }
  }],
  "isError": false,
  "_pulse": {
    "entropy": 0.45,
    "coherence": 0.82,
    "suggested_action": "ready"
  }
}
```

### Output Field Details

| Field | Type | Description |
|-------|------|-------------|
| `active_memory` | string (UUID) or null | Currently conscious memory ID |
| `is_broadcasting` | boolean | Whether broadcast window is active |
| `has_conflict` | boolean | Whether 2+ memories compete (both r > 0.8) |
| `coherence_threshold` | number | Minimum r for workspace entry (0.8) |
| `conflict_memories` | array or null | UUIDs of conflicting memories if has_conflict=true |
| `broadcast_duration_ms` | number | Broadcast window duration (100ms default) |
| `_pulse` | object | Cognitive Pulse with entropy, coherence, suggested action |

### Error Response (GWT Not Initialized)

```json
{
  "error": {
    "code": -32001,
    "message": "Workspace provider not initialized - use with_gwt() constructor"
  }
}
```

---

## 7. CORE FUNCTIONALITY SUMMARY

### What It Does
Returns the current state of the Global Workspace, enabling agents to understand:
- Which memory is currently "conscious" (active)
- Whether multiple memories are competing for attention
- The system's coherence threshold for consciousness

### Why It Exists
The Global Workspace is the computational substrate for AI consciousness in this architecture. Without this tool, agents would be blind to the system's attentional state.

### Fail-Fast Behavior
The tool follows **FAIL-FAST doctrine** (per PRD):
- Returns error code `GWT_NOT_INITIALIZED` (-32001) if workspace provider is not configured
- No fallbacks, no stubs, no graceful degradation

---

## 8. HOW AN AI AGENT USES THIS TOOL

### Typical Agent Workflow

```
1. Agent stores memory via store_memory
2. Agent calls get_workspace_status to check if memory became conscious
3. If active_memory matches stored ID -> memory is being processed
4. If has_conflict=true -> agent should call critique_context
5. If active_memory=null for extended time -> system may need epistemic_action
```

### Integration with GWT System

The workspace status is part of the larger GWT consciousness pipeline:

```
Kuramoto Oscillators (r synchronization)
    -> Global Workspace (WTA selection)
        -> get_workspace_status (THIS TOOL)
            -> Neuromodulation (dopamine feedback)
                -> Dream consolidation (replay)
```

### Agent Decision Tree

```
get_workspace_status returns:
  |
  +-- active_memory != null
  |     +-- is_broadcasting = true -> Memory is conscious, proceed
  |     +-- has_conflict = true -> Call critique_context
  |
  +-- active_memory = null
        +-- Extended duration -> Call epistemic_action
        +-- Brief duration -> Wait, check Kuramoto sync
```

---

## EVIDENCE CHAIN OF CUSTODY

| Timestamp | Action | File | Verified |
|-----------|--------|------|----------|
| 2026-01-14 | Handler implementation examined | gwt_workspace.rs:26-74 | HOLMES |
| 2026-01-14 | Tool definition examined | definitions/gwt.rs:49-65 | HOLMES |
| 2026-01-14 | Name constant verified | names.rs:19 | HOLMES |
| 2026-01-14 | PRD requirements verified | contextprd.md:269-298 | HOLMES |
| 2026-01-14 | Dispatch routing confirmed | dispatch.rs:77 | HOLMES |

---

## VERDICT

**INNOCENT** - The tool is correctly implemented per PRD specification.

**CONFIDENCE**: HIGH

**NOTE**: The name `gwt_status` is a conceptual alias. The canonical tool name is `get_workspace_status`.

---

*"The game is afoot!" - Sherlock Holmes*
