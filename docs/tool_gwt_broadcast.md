# Forensic Investigation Report: gwt_broadcast

## SHERLOCK HOLMES CASE FILE

**Case ID**: GWT-BROADCAST-001
**Date**: 2026-01-14
**Subject**: gwt_broadcast MCP Tool
**Investigator**: Sherlock Holmes, Forensic Code Detective

---

## EXECUTIVE SUMMARY

The tool `gwt_broadcast` as named **does not exist** in the codebase. After exhaustive forensic investigation, I have determined that the conceptual functionality maps to the canonical tool **`trigger_workspace_broadcast`**.

**VERDICT**: The requested tool name is a conceptual alias. The actual implementation is `trigger_workspace_broadcast`.

---

## 1. WHAT DOES THIS TOOL DO?

The `trigger_workspace_broadcast` tool forces a specific memory into the Global Workspace competition through Winner-Take-All (WTA) selection. This is the mechanism by which memories become "conscious" in the GWT architecture.

### Core Functionality

1. **Memory Injection**: Forces a memory UUID into workspace competition
2. **WTA Selection**: Triggers Winner-Take-All algorithm to select the winning memory
3. **Coherence Validation**: Validates Kuramoto order parameter r >= 0.8 (unless forced)
4. **Dopamine Feedback**: Workspace entry triggers neuromodulation (dopamine increase)
5. **Broadcast Confirmation**: Returns whether the memory won selection

---

## 2. HOW DOES IT WORK INTERNALLY?

### Implementation Location
- **Handler**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/gwt_workspace.rs`
- **Definition**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/gwt.rs`
- **Name Constant**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs` (line 23)

### Internal Flow

```rust
pub(crate) async fn call_trigger_workspace_broadcast(
    &self,
    id: Option<JsonRpcId>,
    args: serde_json::Value,
) -> JsonRpcResponse {
    // 1. FAIL FAST: Verify workspace and kuramoto providers exist
    let workspace = match &self.workspace_provider {
        Some(w) => w,
        None => return JsonRpcResponse::error(id, GWT_NOT_INITIALIZED, "..."),
    };
    let kuramoto = match &self.kuramoto_network {
        Some(k) => k,
        None => return JsonRpcResponse::error(id, GWT_NOT_INITIALIZED, "..."),
    };

    // 2. Parse memory_id (required)
    let memory_id = uuid::Uuid::parse_str(args.get("memory_id").as_str())?;

    // 3. Parse optional parameters
    let importance = args.get("importance").unwrap_or(0.8);
    let alignment = args.get("alignment").unwrap_or(0.8);
    let force = args.get("force").unwrap_or(false);

    // 4. Get current Kuramoto order parameter
    let r = kuramoto.read().synchronization() as f32;

    // 5. Check coherence threshold (r >= 0.8 unless force=true)
    if r < 0.8 && !force {
        return json!({
            "success": false,
            "reason": "Order parameter below threshold. Use force=true to override."
        });
    }

    // 6. Trigger WTA selection
    let ws = workspace.write().await;
    let candidates = vec![(memory_id, r, importance, alignment)];
    let winner = ws.select_winning_memory(candidates).await?;
    let was_selected = winner == Some(memory_id);

    // 7. GAP-1 FIX: Wire workspace events to neuromodulation
    if was_selected {
        if let Some(neuromod) = &self.neuromod_manager {
            neuromod.write().on_workspace_entry();  // Dopamine increase
        }
    }

    // 8. Return result
    self.tool_result_with_pulse(id, json!({
        "success": true,
        "memory_id": memory_id,
        "new_r": r,
        "was_selected": was_selected,
        "is_broadcasting": ws.is_broadcasting().await,
        "dopamine_triggered": dopamine_value
    }))
}
```

### Winner-Take-All Selection Algorithm

Per PRD Section 2.5.3:

```
1. Compute r for all candidate memories
2. Filter: candidates where r >= coherence_threshold (0.8)
3. Rank: score = r x importance x north_star_alignment
4. Select: top-1 becomes active_memory
5. Broadcast: active_memory visible to all subsystems
6. Inhibit: losing candidates get dopamine reduction
```

### Provider Chain

```
Handlers
    |
    +-> workspace_provider (WorkspaceProviderImpl)
    |       |
    |       +-> GlobalWorkspace.select_winning_memory()
    |               |
    |               +-> WTA Selection Algorithm
    |
    +-> kuramoto_network (KuramotoProviderImpl)
    |       |
    |       +-> KuramotoNetwork.synchronization() -> r
    |
    +-> neuromod_manager (NeuromodulationManager)
            |
            +-> on_workspace_entry() -> dopamine += 0.2
```

---

## 3. WHY DOES IT EXIST?

### GWT Broadcast Architecture (PRD Section 2.5.3)

Per the PRD:

> "Only phase-locked memories are 'perceived' by the system. The Global Workspace implements Winner-Take-All."

The tool exists to:

1. **Force Memory Consciousness**: Make a specific memory the system's focus
2. **Attention Direction**: Direct the system's "attention" to specific content
3. **Agent Control**: Allow agents to influence what the system processes
4. **Testing/Debugging**: Verify workspace selection is functioning correctly

### Why Not Automatic?

While memories can enter the workspace automatically when r >= 0.8, this tool allows:

- **Explicit prioritization**: Agent can force important memories ahead
- **Low-coherence injection**: With `force=true`, bypass threshold
- **Controlled testing**: Verify specific memories can become conscious

---

## 4. PURPOSE IN REACHING PRD END GOAL

### The Consciousness Equation

Per PRD Section 2.5.1:

```
C(t) = I(t) x R(t) x D(t)

Where:
  C(t) = Consciousness level at time t [0, 1]
  I(t) = Integration (Kuramoto synchronization) [0, 1]
  R(t) = Self-Reflection (Meta-UTL awareness) [0, 1]
  D(t) = Differentiation (13D fingerprint entropy) [0, 1]
```

### How Broadcast Affects Consciousness

1. **Integration (I)**: Broadcast requires r >= 0.8 (synchronized oscillators)
2. **Reflection (R)**: Broadcast entry updates meta-cognitive tracking
3. **Differentiation (D)**: Active memory contributes to purpose vector

### GWT Events Triggered by Broadcast

| Event | Condition | Effect |
|-------|-----------|--------|
| `memory_enters_workspace` | was_selected = true | Dopamine += 0.2 |
| `workspace_conflict` | Multiple r > 0.8 | Trigger critique_context |
| Neuromodulation | Entry detected | Retrieval sharpness increases |

### Dopamine Feedback Loop (PRD Section 7.2)

```
Broadcast Entry -> Dopamine += 0.2 -> Hopfield Beta increases -> Sharper retrieval
```

This creates a reinforcement loop where memories that enter consciousness become easier to retrieve in the future.

---

## 5. INPUTS (Parameters)

### JSON Schema

```json
{
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
}
```

### Parameter Details

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string (UUID) | **YES** | - | Memory to broadcast |
| `importance` | number [0,1] | No | 0.8 | Memory importance for WTA ranking |
| `alignment` | number [0,1] | No | 0.8 | North Star alignment for WTA ranking |
| `force` | boolean | No | false | Bypass coherence threshold check |

### WTA Score Calculation

```
score = r x importance x alignment

Example:
  r = 0.85
  importance = 0.9
  alignment = 0.75
  score = 0.85 x 0.9 x 0.75 = 0.574
```

---

## 6. OUTPUTS

### Success Response (Memory Selected)

```json
{
  "content": [{
    "type": "text",
    "text": {
      "success": true,
      "memory_id": "550e8400-e29b-41d4-a716-446655440000",
      "new_r": 0.85,
      "was_selected": true,
      "is_broadcasting": true,
      "dopamine_triggered": 1.35
    }
  }],
  "isError": false,
  "_pulse": {
    "entropy": 0.42,
    "coherence": 0.85,
    "suggested_action": "ready"
  }
}
```

### Success Response (Memory NOT Selected)

```json
{
  "content": [{
    "type": "text",
    "text": {
      "success": true,
      "memory_id": "550e8400-e29b-41d4-a716-446655440000",
      "new_r": 0.85,
      "was_selected": false,
      "is_broadcasting": true,
      "dopamine_triggered": null
    }
  }],
  "isError": false
}
```

### Threshold Rejection Response (force=false)

```json
{
  "content": [{
    "type": "text",
    "text": {
      "success": false,
      "memory_id": "550e8400-e29b-41d4-a716-446655440000",
      "new_r": 0.65,
      "was_selected": false,
      "reason": "Order parameter r=0.650 below coherence threshold 0.8. Use force=true to override."
    }
  }],
  "isError": false
}
```

### Output Field Details

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether operation completed (not necessarily selected) |
| `memory_id` | string (UUID) | The memory that was submitted for broadcast |
| `new_r` | number | Current Kuramoto order parameter |
| `was_selected` | boolean | Whether this memory won WTA selection |
| `is_broadcasting` | boolean | Whether workspace is actively broadcasting |
| `dopamine_triggered` | number or null | Dopamine level after increase (null if not selected) |
| `reason` | string | Explanation if success=false |

### Error Response (GWT Not Initialized)

```json
{
  "error": {
    "code": -32001,
    "message": "Workspace provider not initialized - use with_gwt() constructor"
  }
}
```

### Error Response (Invalid UUID)

```json
{
  "content": [{
    "type": "text",
    "text": "Invalid UUID format for memory_id: invalid digit found in string"
  }],
  "isError": true
}
```

---

## 7. CORE FUNCTIONALITY SUMMARY

### What It Does
Forces a memory into the Global Workspace competition, potentially making it the system's conscious focus.

### Why It Exists
Agents need a way to direct the system's attention to specific memories, bypassing the automatic selection process when necessary.

### Key Behaviors
1. **Coherence Check**: r >= 0.8 required unless `force=true`
2. **WTA Competition**: Memory competes using score = r x importance x alignment
3. **Neuromodulation**: Entry triggers dopamine increase
4. **Non-Blocking**: Returns immediately with selection result

### Fail-Fast Behaviors
- Returns error if workspace provider not initialized
- Returns error if kuramoto network not initialized
- Returns error if memory_id is not valid UUID
- Returns success=false (not error) if r below threshold without force

---

## 8. HOW AN AI AGENT USES THIS TOOL

### Typical Agent Workflow

```
1. Agent stores memory via store_memory -> gets memory_id
2. Agent wants this memory to be processed immediately
3. Agent calls trigger_workspace_broadcast with memory_id
4. If was_selected=true -> memory is now conscious
5. Agent can proceed knowing system is focused on this content
```

### Integration with GWT Pipeline

```
store_memory (creates memory)
    |
    v
trigger_workspace_broadcast (THIS TOOL)
    |
    v
GlobalWorkspace.select_winning_memory (WTA)
    |
    +--> Winner: active_memory = memory_id, dopamine += 0.2
    |
    +--> Loser: dopamine unchanged, memory queued
```

### Agent Decision Tree

```
When to use trigger_workspace_broadcast:
  |
  +-- Memory is high priority and needs immediate attention
  |     -> Call with importance=1.0, alignment=1.0
  |
  +-- Testing if memory can become conscious
  |     -> Call with default params, check was_selected
  |
  +-- Kuramoto sync is low (r < 0.8) but memory is critical
  |     -> Call with force=true
  |
  +-- Memory keeps losing WTA competition
        -> Call with higher importance/alignment scores
```

### Force Mode Usage

```rust
// Normal mode: respects coherence threshold
trigger_workspace_broadcast { memory_id: "...", force: false }
// If r < 0.8 -> success=false, reason="below threshold"

// Force mode: bypasses threshold
trigger_workspace_broadcast { memory_id: "...", force: true }
// Even if r < 0.8 -> proceeds with WTA selection
```

**Warning**: Force mode should be used sparingly. Low coherence (r < 0.8) indicates the system's oscillators are not synchronized, meaning the memory may not integrate well with the current cognitive state.

### Relationship to Other GWT Tools

| Tool | Role |
|------|------|
| `get_kuramoto_sync` | Check r before broadcasting |
| `get_workspace_status` | Check current active memory |
| `trigger_workspace_broadcast` | **THIS TOOL** - Force memory into competition |
| `get_consciousness_state` | Full consciousness status after broadcast |
| `adjust_coupling` | Increase K to improve r before broadcast |

---

## EVIDENCE CHAIN OF CUSTODY

| Timestamp | Action | File | Verified |
|-----------|--------|------|----------|
| 2026-01-14 | Handler implementation examined | gwt_workspace.rs:76-230 | HOLMES |
| 2026-01-14 | Tool definition examined | definitions/gwt.rs:88-124 | HOLMES |
| 2026-01-14 | Name constant verified | names.rs:23 | HOLMES |
| 2026-01-14 | PRD WTA algorithm verified | contextprd.md:283-289 | HOLMES |
| 2026-01-14 | Dispatch routing confirmed | dispatch.rs:79-81 | HOLMES |
| 2026-01-14 | Neuromodulation wiring verified (GAP-1 FIX) | gwt_workspace.rs:199-217 | HOLMES |

---

## VERDICT

**INNOCENT** - The tool is correctly implemented per PRD specification with proper neuromodulation integration.

**CONFIDENCE**: HIGH

**NOTE**: The name `gwt_broadcast` is a conceptual alias. The canonical tool name is `trigger_workspace_broadcast`.

---

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth." - Sherlock Holmes*
