# FORENSIC INVESTIGATION: dream_abort

## CASE SUMMARY

**Case ID**: TOOL-DREAM-ABORT-001
**Date**: 2026-01-14
**Subject**: Investigation of `dream_abort` MCP tool
**Investigator**: Sherlock Holmes (Forensic Code Agent)

---

## VERDICT: TOOL NAME INCORRECT - ACTUAL NAME IS `abort_dream`

**PARTIALLY GUILTY OF MISNAMING**

After exhaustive forensic analysis, I found that `dream_abort` **DOES NOT EXIST**, but the equivalent tool `abort_dream` **DOES EXIST** and provides the requested functionality.

---

## EVIDENCE LOG

### Search Results

| Search Pattern | Location Searched | Matches Found |
|----------------|-------------------|---------------|
| `dream_abort` | Full codebase | **0** |
| `abort_dream` | Full codebase | **Many** |

### Chain of Custody

```
TIMESTAMP: 2026-01-14T00:00:00Z
ACTION: grep -r "abort_dream" /home/cabdru/contextgraph/crates/context-graph-mcp
RESULT: Found in tools/definitions/dream.rs:63, handlers/dream/handlers.rs:326
VERIFIED BY: Holmes
```

---

## THE ACTUAL TOOL: `abort_dream`

### What Does This Tool Do?

**Aborts the current dream consolidation cycle** with a constitutional mandate of completing the wake operation within **100 milliseconds**.

From PRD Section 7.1:
> Dream Layer (SRC): Schedule: activity<0.15 for 10min -> trigger, **wake<100ms on query**

### Why Does It Exist?

The dream system runs background consolidation cycles (NREM + REM phases) that:
1. Replay exiting memories (Hebbian learning)
2. Discover blind spots (hyperbolic random walks)
3. Create amortized shortcuts (3+ hop paths traversed 5+ times)

However, when a user query arrives during a dream cycle, the system must **wake immediately** (within 100ms per constitution mandate) to serve the query. This tool provides:
1. Manual abort capability for testing/debugging
2. Verification of the 100ms wake latency mandate
3. Clean shutdown of dream cycles

### Purpose in Reaching PRD End Goal

The PRD establishes a bio-inspired memory system with dream consolidation. The `abort_dream` tool ensures:
1. **Responsiveness**: System can always respond to queries within latency SLAs
2. **Safety**: Dream cycles can be interrupted without corruption
3. **Observability**: Wake latency can be measured and verified against constitution

---

## HOW IT WORKS INTERNALLY

### Handler Implementation

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/dream/handlers.rs:326-426`

```rust
pub(crate) async fn call_abort_dream(
    &self,
    id: Option<JsonRpcId>,
    args: serde_json::Value,
) -> JsonRpcResponse {
    // FAIL FAST: Check dream controller
    let dream_controller = match &self.dream_controller {
        Some(dc) => dc,
        None => {
            return JsonRpcResponse::error(
                id,
                error_codes::DREAM_NOT_INITIALIZED,
                "DreamController not initialized - use with_dream() constructor",
            );
        }
    };

    // Get previous state
    let previous_state = {
        let controller = dream_controller.read();
        controller.get_status().state.phase_name().to_string()
    };

    // Check if actually dreaming
    let is_dreaming = {
        let controller = dream_controller.read();
        controller.get_status().is_dreaming
    };

    if !is_dreaming {
        return self.tool_result_with_pulse(id, json!({
            "aborted": false,
            "abort_latency_ms": 0,
            "previous_state": previous_state,
            "mandate_met": true,
            "reason": "Not currently dreaming - nothing to abort"
        }));
    }

    // Execute abort
    let abort_result = {
        let mut controller = dream_controller.write();
        controller.abort()
    };

    match abort_result {
        Ok(wake_latency) => {
            let mandate_met = wake_latency.as_millis() < 100;
            // ... return success response
        }
        Err(e) => {
            // ... return error response
        }
    }
}
```

### Key Internal Flow

1. **Check DreamController initialization** (FAIL FAST per AP-26)
2. **Get previous state** for return value
3. **Check if actually dreaming** (no-op if not)
4. **Execute abort** via `DreamController.abort()`
5. **Measure wake latency** and compare to 100ms mandate
6. **Log warning** if mandate violated
7. **Record completion** in DreamScheduler

---

## INPUTS

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| reason | string | No | "Manual abort requested" | Reason for abort (for logging) |

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "reason": {
      "type": "string",
      "description": "Reason for abort (optional)"
    }
  },
  "required": []
}
```

---

## OUTPUTS

### Success Response (Dream was aborted)

```json
{
  "aborted": true,
  "abort_latency_ms": 45,
  "previous_state": "Nrem",
  "mandate_met": true,
  "reason": "Manual abort requested"
}
```

### Success Response (No dream running)

```json
{
  "aborted": false,
  "abort_latency_ms": 0,
  "previous_state": "Awake",
  "mandate_met": true,
  "reason": "Not currently dreaming - nothing to abort"
}
```

### Error Response (DreamController not initialized)

```json
{
  "error": {
    "code": -32001,
    "message": "DreamController not initialized - use with_dream() constructor"
  }
}
```

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| aborted | boolean | Whether abort was executed |
| abort_latency_ms | u64 | Time taken to abort (milliseconds) |
| previous_state | string | Dream state before abort (Awake/EnteringDream/Nrem/Rem/Waking) |
| mandate_met | boolean | Whether <100ms constitution mandate was met |
| reason | string | Reason for abort or why nothing was aborted |

---

## CONSTITUTION COMPLIANCE

From `/home/cabdru/contextgraph/docs2/contextprd.md` Section 7.1:

```
Dream Layer (SRC)
Schedule: activity<0.15 for 10min -> trigger, wake<100ms on query
```

The tool enforces:
- **100ms wake latency mandate**: `mandate_met = wake_latency.as_millis() < 100`
- **Warning on violation**: Logs warning if mandate is not met
- **Explicit reporting**: Returns `mandate_met` boolean for observability

---

## HOW AN AI AGENT USES THIS TOOL

### Scenario 1: User Query During Dream

```javascript
// Agent receives urgent user query during dream cycle
const dreamStatus = await call_tool("get_dream_status", {});

if (dreamStatus.is_dreaming) {
  // Abort dream to serve query immediately
  const abortResult = await call_tool("abort_dream", {
    reason: "User query requires immediate response"
  });

  if (!abortResult.mandate_met) {
    log("WARNING: Wake latency exceeded 100ms mandate");
  }
}

// Now safe to process user query
```

### Scenario 2: System Health Check

```javascript
// Verify wake latency is within SLA
const dreamStatus = await call_tool("get_dream_status", {});

if (dreamStatus.is_dreaming) {
  const abort = await call_tool("abort_dream", {
    reason: "Health check - verifying wake latency"
  });

  metrics.record("dream_wake_latency_ms", abort.abort_latency_ms);

  if (abort.abort_latency_ms > 80) {
    alert("Wake latency approaching mandate threshold");
  }
}
```

### Scenario 3: Graceful Shutdown

```javascript
// Before system shutdown, ensure dream cycle completes cleanly
const status = await call_tool("get_dream_status", {});

if (status.is_dreaming) {
  const result = await call_tool("abort_dream", {
    reason: "Graceful system shutdown"
  });
  log("Dream aborted cleanly, previous state:", result.previous_state);
}
```

---

## RELATED TOOLS

| Tool | Purpose |
|------|---------|
| `trigger_dream` | Manually start a dream cycle |
| `get_dream_status` | Check current dream state |
| `get_gpu_status` | Check GPU availability for dreams |
| `get_trigger_config` | Get trigger thresholds |
| `get_trigger_history` | See past trigger events |

---

## FORENSIC CONCLUSION

**The tool `dream_abort` does not exist.**
**The correct tool name is `abort_dream`.**

The tool:
- **EXISTS** at `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/dream.rs:63`
- **IS IMPLEMENTED** at `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/dream/handlers.rs:326`
- **SERVES** to abort dream cycles with <100ms wake latency mandate
- **IS TESTED** at `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tests/full_state_verification_gwt/dream_tests.rs:196`

---

**CASE STATUS**: CLOSED - Tool exists under different name
**EVIDENCE PRESERVED**: Yes, in this document
**CORRECT TOOL NAME**: `abort_dream` (not `dream_abort`)
