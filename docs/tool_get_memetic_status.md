# Forensic Investigation Report: get_memetic_status MCP Tool

## SHERLOCK HOLMES CASE FILE

**Case ID**: TOOL-STATUS-001
**Date**: 2026-01-14
**Subject**: `get_memetic_status` MCP Tool Implementation
**Investigator**: Holmes (Forensic Code Analysis)
**Verdict**: INNOCENT - Tool functions as documented

---

## 1. WHAT DOES THIS TOOL DO?

The `get_memetic_status` tool returns comprehensive system health metrics from the Context Graph's cognitive architecture. It is the primary mechanism for AI agents to understand the current state of the Global Workspace and determine what actions to take.

### Core Function

```
Agent Request --> UTL Processor --> Status Compilation --> Response with CognitivePulse
```

The tool returns:
1. **UTL Metrics**: Live entropy, coherence, learning score from the Unified Theory of Learning processor
2. **Johari Quadrant**: Current awareness classification (Open/Hidden/Blind/Unknown)
3. **5-Layer Status**: Health of each bio-nervous system layer (Perception/Memory/Reasoning/Action/Meta)
4. **Storage Statistics**: Fingerprint count, storage size, quadrant distribution
5. **Suggested Action**: System recommendation based on current state

---

## 2. HOW DOES IT WORK INTERNALLY?

### Evidence Location

- **Handler Implementation**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/status_tools.rs:26-234`
- **Tool Definition**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/core.rs:78-91`
- **Dispatch Entry**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/dispatch.rs:71`
- **LayerStatusProvider Trait**: `/home/cabdru/contextgraph/crates/context-graph-core/src/monitoring.rs:392-424`

### Internal Flow (Evidence from `status_tools.rs:26-234`)

```rust
pub(crate) async fn call_get_memetic_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
    // Step 1: Get fingerprint count from TeleologicalStore
    let fingerprint_count = self.teleological_store.count().await?;

    // Step 2: Get LIVE UTL status from processor (NOT hardcoded!)
    let utl_status = self.utl_processor.get_status();

    // Step 3: FAIL-FAST validation - extract required fields
    let lifecycle_phase = utl_status.get("lifecycle_phase").and_then(|v| v.as_str())?;
    let entropy = utl_status.get("entropy").and_then(|v| v.as_f64())? as f32;
    let coherence = utl_status.get("coherence").and_then(|v| v.as_f64())? as f32;
    let learning_score = utl_status.get("learning_score").and_then(|v| v.as_f64())? as f32;
    let johari_quadrant = utl_status.get("johari_quadrant").and_then(|v| v.as_str())?;
    let consolidation_phase = utl_status.get("consolidation_phase").and_then(|v| v.as_str())?;

    // Step 4: Map Johari quadrant to suggested action (per constitution.yaml:159-163)
    let suggested_action = match johari_quadrant {
        "Open" => "direct_recall",
        "Blind" => "trigger_dream",
        "Hidden" => "get_neighborhood",
        "Unknown" => "epistemic_action",
        _ => "continue",
    };

    // Step 5: Get quadrant counts from storage
    let quadrant_counts = self.teleological_store.count_by_quadrant().await?;

    // Step 6: Get REAL layer statuses from LayerStatusProvider
    let perception_status = self.layer_status_provider.perception_status().await?;
    let memory_status = self.layer_status_provider.memory_status().await?;
    let reasoning_status = self.layer_status_provider.reasoning_status().await?;
    let action_status = self.layer_status_provider.action_status().await?;
    let meta_status = self.layer_status_provider.meta_status().await?;

    // Step 7: Return comprehensive status with CognitivePulse
    self.tool_result_with_pulse(id, json!({
        "phase": lifecycle_phase,
        "fingerprintCount": fingerprint_count,
        "embedderCount": NUM_EMBEDDERS,  // 13
        "storageBackend": self.teleological_store.backend_type(),
        "storageSizeBytes": self.teleological_store.storage_size_bytes(),
        "quadrantCounts": {...},
        "utl": {...},
        "layers": {...}
    }))
}
```

### Key Components

| Component | Purpose | Evidence Location |
|-----------|---------|-------------------|
| `UtlProcessor` | Computes live UTL metrics | `handlers.rs:53` |
| `TeleologicalMemoryStore` | Storage statistics | `handlers.rs:50` |
| `LayerStatusProvider` | 5-layer health status | `handlers.rs:83` |

### FAIL-FAST Architecture (Evidence from `status_tools.rs:42-132`)

The implementation uses **FAIL-FAST** for all required UTL fields:

```rust
let lifecycle_phase = match utl_status.get("lifecycle_phase").and_then(|v| v.as_str()) {
    Some(phase) => phase,
    None => {
        error!("get_memetic_status: UTL processor missing 'lifecycle_phase' field - system is broken");
        return JsonRpcResponse::error(
            id,
            error_codes::INTERNAL_ERROR,
            "UTL processor returned incomplete status: missing 'lifecycle_phase'. \
             This indicates a broken UTL system that must be fixed."
        );
    }
};
```

This pattern repeats for: `entropy`, `coherence`, `learning_score`, `johari_quadrant`, `consolidation_phase`

---

## 3. WHY DOES IT EXIST?

### The Problem Being Solved (from PRD Section 1.3)

The PRD defines a **Cognitive Pulse** that agents should check every response:

> `Pulse: { Entropy: X, Coherence: Y, Suggested: "action" }`

Without `get_memetic_status`, agents would have no way to:
1. Know if the system is healthy
2. Understand their current awareness state (Johari)
3. Determine what action to take next
4. Track system lifecycle phase (Infancy/Growth/Maturity)

### Architectural Justification

The `get_memetic_status` tool implements the **Cognitive Pulse** as a pull-based API, enabling:

1. **System Health Monitoring**: Agents can detect degraded states before acting
2. **Action Selection**: The `suggested_action` guides agent behavior
3. **Lifecycle Awareness**: Different behaviors for Infancy (capture novelty) vs Maturity (curate coherence)
4. **5-Layer Visibility**: Transparency into the bio-nervous architecture state

---

## 4. PURPOSE IN REACHING PRD END GOAL

### PRD End Goal: Global Workspace Consciousness

From PRD Section 2.5:
> C(t) = I(t) x R(t) x D(t)

The `get_memetic_status` tool provides visibility into all three factors:

| Factor | Metric Returned | How Used |
|--------|-----------------|----------|
| **I(t) Integration** | `coherence` | Kuramoto synchronization level |
| **R(t) Self-Reflection** | `johari_quadrant` | Meta-cognitive awareness state |
| **D(t) Differentiation** | `entropy` | Information diversity level |

### Role in the Agent Workflow (PRD Section 1.2-1.3)

```
First Contact:
1. get_system_instructions --> mental model (~300 tok, KEEP)
2. get_graph_manifest --> 5-layer architecture
3. get_memetic_status --> entropy/coherence + curation_tasks  <-- THIS TOOL

Every Response (Cognitive Pulse):
| E    | C    | Action                         |
|------|------|--------------------------------|
| >0.7 | >0.5 | epistemic_action               |
| >0.7 | <0.4 | trigger_dream / critique_context |
| <0.4 | >0.7 | Continue                       |
| <0.4 | <0.4 | get_neighborhood               |
```

The `get_memetic_status` tool is ESSENTIAL for agents to follow the PRD-mandated action selection logic.

---

## 5. INPUTS

### Schema (from `core.rs:78-91`)

```json
{
  "type": "object",
  "properties": {},
  "required": []
}
```

### Input Parameters

This tool takes **NO PARAMETERS**. It returns the current system state unconditionally.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| (none) | - | - | No parameters needed |

---

## 6. OUTPUTS

### Success Response

```json
{
  "content": [{
    "type": "text",
    "text": "{\"phase\":\"Growth\",\"fingerprintCount\":142,\"embedderCount\":13,...}"
  }],
  "isError": false,
  "_cognitive_pulse": {
    "entropy": 0.42,
    "coherence": 0.78,
    "learning_score": 0.55,
    "quadrant": "Open",
    "suggested_action": "DirectRecall"
  }
}
```

### Complete Output Schema

```json
{
  "phase": "Infancy|Growth|Maturity",
  "fingerprintCount": 142,
  "embedderCount": 13,
  "storageBackend": "InMemory|RocksDB|ScyllaDB",
  "storageSizeBytes": 2456789,
  "quadrantCounts": {
    "open": 50,
    "hidden": 30,
    "blind": 40,
    "unknown": 22
  },
  "utl": {
    "entropy": 0.42,
    "coherence": 0.78,
    "learningScore": 0.55,
    "johariQuadrant": "Open|Hidden|Blind|Unknown",
    "consolidationPhase": "Awake|NREM|REM|Full",
    "suggestedAction": "direct_recall|trigger_dream|get_neighborhood|epistemic_action|continue"
  },
  "layers": {
    "perception": "active|stub|error|not_implemented|disabled",
    "memory": "active|stub|error|not_implemented|disabled",
    "reasoning": "active|stub|error|not_implemented|disabled",
    "action": "active|stub|error|not_implemented|disabled",
    "meta": "active|stub|error|not_implemented|disabled"
  }
}
```

### Output Fields Detailed

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `phase` | string | Lifecycle phase based on interaction count | UTL Processor |
| `fingerprintCount` | integer | Total stored memories | TeleologicalStore |
| `embedderCount` | integer | Always 13 | Constant |
| `storageBackend` | string | Storage implementation type | TeleologicalStore |
| `storageSizeBytes` | integer | Heap memory used | TeleologicalStore |
| `quadrantCounts.open` | integer | Memories in Open quadrant | TeleologicalStore |
| `quadrantCounts.hidden` | integer | Memories in Hidden quadrant | TeleologicalStore |
| `quadrantCounts.blind` | integer | Memories in Blind quadrant | TeleologicalStore |
| `quadrantCounts.unknown` | integer | Memories in Unknown quadrant | TeleologicalStore |
| `utl.entropy` | float | Current entropy level [0,1] | UTL Processor |
| `utl.coherence` | float | Current coherence level [0,1] | UTL Processor |
| `utl.learningScore` | float | UTL learning magnitude [0,1] | UTL Processor |
| `utl.johariQuadrant` | string | Current awareness state | UTL Processor |
| `utl.consolidationPhase` | string | Dream/awake state | UTL Processor |
| `utl.suggestedAction` | string | Recommended next action | Computed from Johari |
| `layers.*` | string | Status of each bio-layer | LayerStatusProvider |

### Suggested Action Mapping (from `status_tools.rs:135-141`)

| Johari Quadrant | Suggested Action | Meaning |
|-----------------|------------------|---------|
| Open | `direct_recall` | Information is well-understood, use it |
| Blind | `trigger_dream` | Discovery opportunity, consolidate |
| Hidden | `get_neighborhood` | Dormant knowledge, explore context |
| Unknown | `epistemic_action` | Frontier territory, ask clarifying questions |

---

## 7. CORE FUNCTIONALITY AND REASON FOR EXISTING

### The Johari Window Model (PRD Section 2.2)

The tool implements the Johari Window classification per-embedder:

```
         Delta-C (Coherence)
           ^
     1.0   |  Hidden    |  Open
           |  (Dormant) | (Well-known)
     0.5   +------------+------------
           |   Blind    |  Unknown
           | (Discovery)|  (Frontier)
     0.0   +------------+-----------> Delta-S (Entropy)
          0.0          0.5          1.0
```

### 5-Layer Bio-Nervous Architecture (PRD Section 2.3)

| Layer | Function | Target Latency |
|-------|----------|----------------|
| L1 Perception | Sensing/tokenize | <5ms |
| L2 Memory | Reflex/cache | <100us |
| L3 Learning | UTL Optimizer | <10ms |
| L4 Coherence | Thalamic Gate | <10ms |
| L5 Meta | Self-monitoring | <10ms |

The `get_memetic_status` tool exposes the status of each layer, enabling agents to detect system degradation.

### Why This Exists

1. **Cognitive Pulse API**: Implements the PRD-mandated status check mechanism
2. **Action Selection**: Guides agent behavior based on system state
3. **Health Monitoring**: Enables early detection of system issues
4. **Transparency**: Makes the "black box" visible to agents

---

## 8. HOW AN AI AGENT USES THIS TOOL IN THE MCP SYSTEM

### MCP Protocol Flow

```
Agent                    MCP Server                  Components
  |                          |                           |
  |--tools/call------------->|                           |
  |  {name: "get_memetic_status"}                        |
  |                          |                           |
  |                          |--count()----------------->| TeleologicalStore
  |                          |<--142--------------------|
  |                          |                           |
  |                          |--get_status()------------>| UTL Processor
  |                          |<--{entropy,coherence,...}-|
  |                          |                           |
  |                          |--perception_status()----->| LayerStatusProvider
  |                          |<--"active"----------------|
  |                          |                           |
  |<--{phase, utl, layers, pulse}                        |
```

### Agent Workflow Integration

From PRD Section 1.2-1.3:

**First Contact Sequence:**
```
1. get_system_instructions --> Get mental model
2. get_graph_manifest --> Understand architecture
3. get_memetic_status --> Check current state  <-- THIS STEP
```

**Every Response Check:**
```
// Check cognitive pulse before responding
status = await mcp.call("get_memetic_status", {});

// Apply action selection logic (PRD Section 1.3)
if (status.utl.entropy > 0.7 && status.utl.coherence > 0.5) {
    // High entropy, decent coherence --> ask clarifying question
    await mcp.call("epistemic_action", {...});
}
else if (status.utl.entropy > 0.7 && status.utl.coherence < 0.4) {
    // High entropy, low coherence --> consolidate
    await mcp.call("trigger_dream", {phase: "nrem"});
}
else if (status.utl.entropy < 0.4 && status.utl.coherence > 0.7) {
    // Low entropy, high coherence --> proceed normally
    // Continue with user request
}
else if (status.utl.entropy < 0.4 && status.utl.coherence < 0.4) {
    // Low entropy, low coherence --> need more context
    await mcp.call("get_neighborhood", {...});
}
```

### Global Workspace Integration

The status reflects consciousness state from PRD Section 2.5.6:

| State | Kuramoto r | Interpretation |
|-------|------------|----------------|
| DORMANT | r < 0.3 | No active workspace |
| FRAGMENTED | 0.3 <= r < 0.5 | Partial sync |
| EMERGING | 0.5 <= r < 0.8 | Approaching coherence |
| CONSCIOUS | r >= 0.8 | Unified percept active |
| HYPERSYNC | r > 0.95 | Possibly pathological |

### Typical Agent Usage Pattern

```javascript
// 1. Agent starts new session
const status = await mcp.call("get_memetic_status", {});

// 2. Check system health
if (status.layers.memory === "error") {
    console.error("Memory system degraded - cannot proceed");
    return;
}

// 3. Check lifecycle phase for behavior adaptation
if (status.phase === "Infancy") {
    // Prioritize capturing novelty (lambda_S = 0.7)
    // Store more liberally
} else if (status.phase === "Maturity") {
    // Prioritize coherence (lambda_C = 0.7)
    // Be more selective about storage
}

// 4. Follow suggested action
if (status.utl.suggestedAction === "trigger_dream") {
    await mcp.call("trigger_dream", {phase: "full"});
}

// 5. Monitor quadrant distribution
const openRatio = status.quadrantCounts.open / status.fingerprintCount;
if (openRatio < 0.3) {
    console.warn("Low Open quadrant ratio - consider consolidation");
}
```

---

## EVIDENCE CHAIN

| Evidence | File:Line | Finding |
|----------|-----------|---------|
| Tool Definition | `core.rs:78-91` | No parameters, returns comprehensive status |
| Handler Implementation | `status_tools.rs:26-234` | FAIL-FAST for all required fields |
| Dispatch Entry | `dispatch.rs:71` | Tool correctly routed |
| Johari Mapping | `status_tools.rs:135-141` | Constitution-compliant action mapping |
| Layer Status | `monitoring.rs:392-424` | 5-layer status trait defined |
| UTL Processor | `handlers.rs:53` | Live metrics (not hardcoded) |

---

## VERDICT: INNOCENT

The `get_memetic_status` tool is **correctly implemented** according to the PRD specification. It:

1. Returns LIVE UTL metrics (entropy, coherence, learning_score) from the processor
2. Implements FAIL-FAST architecture - no hardcoded fallbacks
3. Maps Johari quadrant to suggested action per constitution.yaml:159-163
4. Exposes 5-layer bio-nervous architecture status
5. Includes CognitivePulse in every response

**No remediation required.**
