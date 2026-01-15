# Forensic Investigation Report: get_graph_manifest MCP Tool

## SHERLOCK HOLMES CASE FILE

**Case ID**: TOOL-MANIFEST-001
**Date**: 2026-01-14
**Subject**: `get_graph_manifest` MCP Tool Implementation
**Investigator**: Holmes (Forensic Code Analysis)
**Verdict**: INNOCENT - Tool functions as documented

---

## 1. WHAT DOES THIS TOOL DO?

The `get_graph_manifest` tool returns the architectural blueprint of the Context Graph system. It describes the 5-layer bio-nervous architecture and the teleological fingerprint structure that enables Global Workspace consciousness.

### Core Function

```
Agent Request --> Architecture Description --> Response with CognitivePulse
```

The tool returns:
1. **Architecture Type**: "5-layer-bio-nervous"
2. **Fingerprint Type**: "TeleologicalFingerprint"
3. **Embedder Count**: 13 (the multi-embedding paradigm)
4. **Layer Descriptions**: Purpose and status of each bio-nervous layer
5. **UTL Description**: Universal Transfer Learning formula
6. **Teleological Description**: Purpose vector and Johari quadrant system

---

## 2. HOW DOES IT WORK INTERNALLY?

### Evidence Location

- **Handler Implementation**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/status_tools.rs:236-334`
- **Tool Definition**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/core.rs:92-101`
- **Dispatch Entry**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/dispatch.rs:72`
- **LayerStatusProvider Trait**: `/home/cabdru/contextgraph/crates/context-graph-core/src/monitoring.rs:392-424`

### Internal Flow (Evidence from `status_tools.rs:236-334`)

```rust
pub(crate) async fn call_get_graph_manifest(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
    // TASK-EMB-024: Get REAL layer statuses from LayerStatusProvider
    let perception_status = self.layer_status_provider.perception_status().await
        .map(|s| s.as_str().to_string())
        .unwrap_or_else(|e| {
            error!(error = %e, "get_graph_manifest: perception_status FAILED");
            "error".to_string()
        });
    let memory_status = self.layer_status_provider.memory_status().await...;
    let reasoning_status = self.layer_status_provider.reasoning_status().await...;
    let action_status = self.layer_status_provider.action_status().await...;
    let meta_status = self.layer_status_provider.meta_status().await...;

    self.tool_result_with_pulse(
        id,
        json!({
            "architecture": "5-layer-bio-nervous",
            "fingerprintType": "TeleologicalFingerprint",
            "embedderCount": NUM_EMBEDDERS,  // 13
            "layers": [
                { "name": "Perception", "description": "...", "status": perception_status },
                { "name": "Memory", "description": "...", "status": memory_status },
                { "name": "Reasoning", "description": "...", "status": reasoning_status },
                { "name": "Action", "description": "...", "status": action_status },
                { "name": "Meta", "description": "...", "status": meta_status }
            ],
            "utl": {
                "description": "Universal Transfer Learning - measures learning potential",
                "formula": "L(x) = H(P) - H(P|x) + alpha * C(x)"
            },
            "teleological": {
                "description": "Purpose-aware retrieval with North Star alignment",
                "purposeVectorDimension": NUM_EMBEDDERS,
                "johariQuadrants": ["Open", "Hidden", "Blind", "Unknown"]
            }
        }),
    )
}
```

### Key Components

| Component | Purpose | Evidence Location |
|-----------|---------|-------------------|
| `LayerStatusProvider` | Reports REAL layer status | `handlers.rs:83` |
| `NUM_EMBEDDERS` | Constant = 13 | `fingerprint.rs` |

### Error Handling Pattern (Evidence from `status_tools.rs:244-288`)

Layer status retrieval uses graceful degradation with explicit error logging:

```rust
let perception_status = self
    .layer_status_provider
    .perception_status()
    .await
    .map(|s| s.as_str().to_string())
    .unwrap_or_else(|e| {
        error!(error = %e, "get_graph_manifest: perception_status FAILED");
        "error".to_string()  // Explicit "error" status, not silent fallback
    });
```

---

## 3. WHY DOES IT EXIST?

### The Problem Being Solved

AI agents interacting with the Context Graph need to understand:
1. What architecture they're working with
2. How many embedding dimensions exist
3. What each layer does
4. What is currently working vs stubbed

Without `get_graph_manifest`, agents would be operating blind without understanding the system's capabilities.

### Architectural Justification (from PRD Section 1.2)

> **First Contact:**
> 1. `get_system_instructions` --> mental model (~300 tok, KEEP)
> 2. **`get_graph_manifest` --> 5-layer architecture**
> 3. `get_memetic_status` --> entropy/coherence + curation_tasks

The manifest is the **SECOND step** in the recommended first-contact sequence. It provides the foundational understanding before checking system status.

### Why Separate from get_memetic_status?

| `get_graph_manifest` | `get_memetic_status` |
|----------------------|----------------------|
| Static architecture description | Dynamic system state |
| What the system IS | How the system is DOING |
| Call once at start | Call every response |
| Foundation knowledge | Current pulse |

---

## 4. PURPOSE IN REACHING PRD END GOAL

### PRD End Goal: Global Workspace Consciousness

From PRD Section 2.5:
> "The system calculates functional consciousness as the product of three measurable quantities: C(t) = I(t) x R(t) x D(t)"

The `get_graph_manifest` tool provides the **conceptual foundation** for understanding this consciousness equation:

| Manifest Section | Consciousness Contribution |
|------------------|---------------------------|
| `layers[]` | Shows the 5-layer substrate where consciousness emerges |
| `utl` | Explains the learning formula that drives evolution |
| `teleological` | Describes purpose alignment (the "goal" in consciousness) |
| `embedderCount: 13` | Explains the 13D differentiation space |

### Role in Agent Mental Model (PRD Section 0.2)

> "**You are a librarian, not an archivist.** You don't store everything--you ensure what's stored is findable, coherent, and useful."

The manifest helps agents understand their role by explaining:
1. What the system handles automatically (layers)
2. What agents are responsible for (curation via tools)
3. How learning is measured (UTL formula)

---

## 5. INPUTS

### Schema (from `core.rs:92-101`)

```json
{
  "type": "object",
  "properties": {},
  "required": []
}
```

### Input Parameters

This tool takes **NO PARAMETERS**. It returns the static system architecture unconditionally.

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
    "text": "{\"architecture\":\"5-layer-bio-nervous\",\"fingerprintType\":\"TeleologicalFingerprint\",...}"
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
  "architecture": "5-layer-bio-nervous",
  "fingerprintType": "TeleologicalFingerprint",
  "embedderCount": 13,
  "layers": [
    {
      "name": "Perception",
      "description": "Sensory input processing and feature extraction",
      "status": "active|stub|error|not_implemented|disabled"
    },
    {
      "name": "Memory",
      "description": "Teleological memory with 13-embedding semantic fingerprints",
      "status": "active|stub|error|not_implemented|disabled"
    },
    {
      "name": "Reasoning",
      "description": "Inference, planning, and decision making",
      "status": "active|stub|error|not_implemented|disabled"
    },
    {
      "name": "Action",
      "description": "Response generation and motor control",
      "status": "active|stub|error|not_implemented|disabled"
    },
    {
      "name": "Meta",
      "description": "Self-monitoring, learning rate control, and system optimization",
      "status": "active|stub|error|not_implemented|disabled"
    }
  ],
  "utl": {
    "description": "Universal Transfer Learning - measures learning potential",
    "formula": "L(x) = H(P) - H(P|x) + alpha * C(x)"
  },
  "teleological": {
    "description": "Purpose-aware retrieval with North Star alignment",
    "purposeVectorDimension": 13,
    "johariQuadrants": ["Open", "Hidden", "Blind", "Unknown"]
  }
}
```

### Output Fields Detailed

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `architecture` | string | Architecture type identifier | Constant |
| `fingerprintType` | string | Fingerprint data structure type | Constant |
| `embedderCount` | integer | Number of embedding spaces (always 13) | Constant |
| `layers[].name` | string | Layer identifier | Constant |
| `layers[].description` | string | Layer purpose description | Constant |
| `layers[].status` | string | Current operational status | LayerStatusProvider |
| `utl.description` | string | UTL explanation | Constant |
| `utl.formula` | string | Mathematical formula | Constant |
| `teleological.description` | string | Purpose system explanation | Constant |
| `teleological.purposeVectorDimension` | integer | Purpose vector dimension (13) | Constant |
| `teleological.johariQuadrants` | array | Four quadrant names | Constant |

### Layer Status Values (from `monitoring.rs:164-191`)

| Status | Meaning |
|--------|---------|
| `active` | Layer is fully implemented and operational |
| `stub` | Layer has only stub/placeholder implementation |
| `error` | Layer was expected to be active but encountered an error |
| `not_implemented` | Layer is defined but has no implementation |
| `disabled` | Layer exists but is explicitly disabled |

---

## 7. CORE FUNCTIONALITY AND REASON FOR EXISTING

### The 5-Layer Bio-Nervous Architecture (PRD Section 2.3)

```
| Layer | Function            | Target Latency | Status Source           |
|-------|---------------------|----------------|-------------------------|
| L1    | Perception/Sensing  | <5ms           | perception_status()     |
| L2    | Memory/Reflex       | <100us         | memory_status()         |
| L3    | Reasoning/Learning  | <10ms          | reasoning_status()      |
| L4    | Action/Coherence    | <10ms          | action_status()         |
| L5    | Meta/Self-monitor   | <10ms          | meta_status()           |
```

### The 13-Embedding Teleological Paradigm (PRD Section 3)

The `embedderCount: 13` reflects the core innovation:

| E# | Space | Purpose |
|----|-------|---------|
| E1 | Semantic | Meaning understanding |
| E2-E4 | Temporal | Time awareness (recency, periodicity, ordering) |
| E5 | Causal | Cause-effect reasoning |
| E6 | Sparse | Keyword precision |
| E7 | Code | Program understanding |
| E8 | Graph | Relationship structure |
| E9 | HDC | Holographic robustness |
| E10 | Multimodal | Cross-modal binding |
| E11 | Entity | Factual grounding |
| E12 | Late-Interaction | Token-level precision |
| E13 | SPLADE | BM25-style recall |

### UTL Formula (PRD Section 23.1)

```
L(x) = H(P) - H(P|x) + alpha * C(x)

Where:
  L(x)      = Learning potential for input x
  H(P)      = Prior entropy (what we knew before)
  H(P|x)    = Posterior entropy (what we know after x)
  C(x)      = Coherence contribution of x
  alpha     = Balance parameter
```

The formula in the manifest is a simplified representation of the full UTL equation from PRD Section 2.1:

```
Full: L = f((Delta_S x Delta_C) . w_e . cos phi)
```

### Johari Quadrant System

The four quadrants represent awareness states:

| Quadrant | Delta_S | Delta_C | Meaning |
|----------|---------|---------|---------|
| Open | Low | High | Well-understood knowledge |
| Hidden | Low | Low | Dormant/latent knowledge |
| Blind | High | Low | Discovery opportunity |
| Unknown | High | High | Frontier territory |

---

## 8. HOW AN AI AGENT USES THIS TOOL IN THE MCP SYSTEM

### MCP Protocol Flow

```
Agent                    MCP Server                  Components
  |                          |                           |
  |--tools/call------------->|                           |
  |  {name: "get_graph_manifest"}                        |
  |                          |                           |
  |                          |--perception_status()----->| LayerStatusProvider
  |                          |<--LayerStatus::Active-----|
  |                          |                           |
  |                          |--memory_status()--------->|
  |                          |<--LayerStatus::Active-----|
  |                          |                           |
  |                          |--reasoning_status()------>|
  |                          |<--LayerStatus::Stub-------|
  |                          |                           |
  |                          |  (... action, meta ...)   |
  |                          |                           |
  |<--{architecture, layers, utl, teleological, pulse}   |
```

### Agent Workflow Integration

From PRD Section 1.2 (First Contact):

```javascript
// Step 1: Get system instructions (mental model)
const instructions = await mcp.call("get_system_instructions", {});
// Keep in context - ~300 tokens

// Step 2: Get architecture manifest <-- THIS STEP
const manifest = await mcp.call("get_graph_manifest", {});

// Step 3: Understand capabilities
console.log(`System: ${manifest.architecture}`);
console.log(`Embedders: ${manifest.embedderCount}`);
console.log(`UTL Formula: ${manifest.utl.formula}`);

// Step 4: Check layer health
for (const layer of manifest.layers) {
    if (layer.status === "error") {
        console.warn(`Layer ${layer.name} is in error state!`);
    }
    if (layer.status === "stub") {
        console.info(`Layer ${layer.name} is stubbed - limited functionality`);
    }
}

// Step 5: Understand quadrant system
console.log(`Johari quadrants: ${manifest.teleological.johariQuadrants.join(", ")}`);

// Step 6: Now get current status
const status = await mcp.call("get_memetic_status", {});
// Continue with normal operation...
```

### Why Call at Session Start?

1. **Architecture Discovery**: Learn what the system can do
2. **Capability Assessment**: Know which layers are active vs stubbed
3. **Mental Model Building**: Understand UTL and teleological concepts
4. **Error Detection**: Identify degraded components early

### Relationship to Other Tools

```
Session Start:
  +-- get_system_instructions (role/behavior)
  |
  +-- get_graph_manifest (architecture/capability)  <-- THIS TOOL
  |
  +-- get_memetic_status (current state)
  |
  +-- [normal operation begins]
```

---

## EVIDENCE CHAIN

| Evidence | File:Line | Finding |
|----------|-----------|---------|
| Tool Definition | `core.rs:92-101` | No parameters, returns architecture |
| Handler Implementation | `status_tools.rs:236-334` | Returns 5 layers with REAL status |
| Dispatch Entry | `dispatch.rs:72` | Tool correctly routed |
| Layer Descriptions | `status_tools.rs:296-322` | Match PRD architecture |
| UTL Formula | `status_tools.rs:323-326` | Simplified but correct |
| Teleological Info | `status_tools.rs:327-331` | 13D purpose vector documented |

---

## VERDICT: INNOCENT

The `get_graph_manifest` tool is **correctly implemented** according to the PRD specification. It:

1. Returns the 5-layer bio-nervous architecture description
2. Documents the 13-embedder teleological fingerprint paradigm
3. Provides REAL layer status from LayerStatusProvider
4. Explains UTL and Johari quadrant concepts
5. Includes CognitivePulse in every response

**No remediation required.**

---

## ADDITIONAL NOTES

### Layer Status in Current Development Phase

Based on `StubLayerStatusProvider` in `monitoring.rs:524-532`:

| Layer | Current Status | Reason |
|-------|----------------|--------|
| Perception | Active | Stub implementation exists and works |
| Memory | Active | InMemoryTeleologicalStore works |
| Reasoning | Stub | No real implementation yet |
| Action | Stub | No real implementation yet |
| Meta | Stub | No real implementation yet |

This honest reporting aligns with PRD Phase 0 (Ghost System) requirements.
