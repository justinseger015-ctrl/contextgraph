# FORENSIC INVESTIGATION: dream_listen

## CASE SUMMARY

**Case ID**: TOOL-DREAM-LISTEN-001
**Date**: 2026-01-14
**Subject**: Investigation of `dream_listen` MCP tool
**Investigator**: Sherlock Holmes (Forensic Code Agent)

---

## VERDICT: TOOL DOES NOT EXIST

**GUILTY OF NON-EXISTENCE**

After exhaustive forensic analysis of the entire codebase at `/home/cabdru/contextgraph`, I must report that the tool named `dream_listen` **DOES NOT EXIST**.

---

## EVIDENCE LOG

### Search Results

| Search Pattern | Location Searched | Matches Found |
|----------------|-------------------|---------------|
| `dream_listen` | Full codebase | **0** |
| `dreamListen` | Full codebase | **0** |
| `listen_dream` | Full codebase | **0** |

### Chain of Custody

```
TIMESTAMP: 2026-01-14T00:00:00Z
ACTION: grep -r "dream_listen" /home/cabdru/contextgraph
RESULT: No matches found
VERIFIED BY: Holmes
```

### Source Files Examined

1. `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/dream.rs`
   - Contains 8 Dream tools (NONE named `dream_listen`)
   - Tools: `trigger_dream`, `get_dream_status`, `abort_dream`, `get_amortized_shortcuts`, `get_gpu_status`, `trigger_mental_check`, `get_trigger_config`, `get_trigger_history`

2. `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tests/tools_list.rs`
   - Declares exactly 59 tools in the system
   - "dream_listen" appears nowhere

---

## WHAT THE USER MAY BE LOOKING FOR

The term "dream_listen" suggests functionality to:
1. Monitor dream cycle events
2. Subscribe to dream state changes
3. Get notifications when dream phases transition

### Related Codebase Concepts

The codebase DOES have a `DreamEventListener` but it is an **internal struct**, NOT an MCP tool:

**Location**: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/listeners/dream.rs`

```rust
/// Listener that queues exiting memories for dream replay and triggers
/// dream consolidation on identity crisis.
pub struct DreamEventListener {
    /// Queue for memories exiting workspace (for dream replay)
    dream_queue: Arc<RwLock<Vec<Uuid>>>,

    /// TriggerManager for IC-based dream triggering (REQUIRED per AP-26)
    trigger_manager: Arc<Mutex<TriggerManager>>,

    /// Callback for dream consolidation signaling (REQUIRED per AP-26)
    consolidation_callback: DreamConsolidationCallback,
}
```

**Purpose**: This listener is registered internally with the workspace to:
- Queue memories exiting workspace for dream replay (MemoryExits events)
- Trigger dream consolidation when Identity Continuity (IC) drops below 0.5

This is NOT exposed as an MCP tool because it is internal system infrastructure.

---

## CLOSEST EQUIVALENT TOOLS

### 1. `get_dream_status`

**What it does**: Get current dream system status

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/dream.rs:49`

**Description**: Get current dream system status including state (Awake/NREM/REM/Waking), GPU usage, activity level, and time since last dream cycle.

**What it returns**:
```json
{
  "state": "Awake",
  "is_dreaming": false,
  "gpu_usage": 0.15,
  "activity_level": 0.25,
  "completed_cycles": 5,
  "time_since_last_dream_secs": 3600,
  "scheduler": {
    "average_activity": 0.12,
    "activity_threshold": 0.15,
    "cooldown_remaining_secs": null
  },
  "constitution_compliance": {
    "gpu_under_30_percent": true,
    "current_gpu_usage": 0.15,
    "max_wake_latency_ms": 100
  }
}
```

**Inputs**: None required

**How to use for "listening"**:
```javascript
// Poll at intervals to "listen" for dream state changes
const status = await call_tool("get_dream_status", {});
if (status.is_dreaming) {
  console.log("Dream cycle in progress:", status.state);
}
```

### 2. `get_trigger_history`

**What it does**: Get recent trigger history

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/dream.rs:166`

**Description**: Returns a list of trigger events with timestamp, entropy value, reason, and workflow status.

**What it returns**:
```json
[
  {
    "timestamp": "2026-01-14T10:30:00Z",
    "entropy": 0.78,
    "reason": "HighEntropy",
    "workflow_initiated": true,
    "workflow_id": "mental_check_20260114_103000_abcd"
  }
]
```

**Inputs**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| limit | integer | No | 20 | Max entries (1-100) |

---

## WHY NO "dream_listen" TOOL EXISTS

The MCP (Model Context Protocol) is a **request-response** protocol. It does not support:
- Server-pushed events
- Subscriptions
- WebSocket-style streaming

Dream "listening" would require:
1. A streaming transport (SSE, WebSocket)
2. Server-side event emission infrastructure
3. Client-side event handling

The current architecture uses **polling** via `get_dream_status` instead.

---

## INTERNAL EVENT SYSTEM (Non-MCP)

The dream system DOES have internal event infrastructure:

### Workspace Events

**Location**: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/workspace/events.rs`

```rust
pub enum WorkspaceEvent {
    MemoryEnters { id: Uuid, order_parameter: f32, timestamp: DateTime<Utc>, fingerprint: Option<TeleologicalFingerprint> },
    MemoryExits { id: Uuid, order_parameter: f32, timestamp: DateTime<Utc> },
    IdentityCritical { identity_coherence: f32, previous_status: String, current_status: String, reason: String, timestamp: DateTime<Utc> },
    WorkspaceConflict { memory_ids: Vec<Uuid>, conflict_type: String, timestamp: DateTime<Utc> },
    WorkspaceEmpty { duration_ms: u64, timestamp: DateTime<Utc> },
}
```

These events are used internally to trigger:
- Dream consolidation (via DreamEventListener)
- Neuromodulation adjustments (via NeuromodulationEventListener)
- Meta-cognitive actions (via MetaCognitiveEventListener)
- Identity monitoring (via IdentityContinuityListener)

---

## RECOMMENDATION

If you need to monitor dream activity:

```bash
# Option 1: Poll dream status at intervals
loop {
  status = call_tool("get_dream_status", {})
  if status.state != last_state {
    log("Dream state changed:", status.state)
  }
  last_state = status.state
  sleep(1000)  # Poll every second
}

# Option 2: Check trigger history for recent activity
history = call_tool("get_trigger_history", { "limit": 10 })
```

---

## FORENSIC CONCLUSION

**The tool `dream_listen` does not exist.**

The user may be:
1. Expecting an event subscription mechanism (not supported by MCP protocol)
2. Confusing internal `DreamEventListener` struct with an MCP tool
3. Requesting a feature that was planned but not implemented

The equivalent functionality is achieved by **polling** `get_dream_status` and checking `get_trigger_history`.

---

**CASE STATUS**: CLOSED - Tool confirmed non-existent
**EVIDENCE PRESERVED**: Yes, in this document
**REMEDIATION**: Use `get_dream_status` polling or `get_trigger_history` instead
