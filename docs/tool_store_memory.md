# Forensic Investigation Report: store_memory MCP Tool

## SHERLOCK HOLMES CASE FILE

**Case ID**: TOOL-STORE-001
**Date**: 2026-01-14
**Subject**: `store_memory` MCP Tool Implementation
**Investigator**: Holmes (Forensic Code Analysis)
**Verdict**: INNOCENT - Tool functions as documented

---

## 1. WHAT DOES THIS TOOL DO?

The `store_memory` tool stores content as a **TeleologicalFingerprint** in the Context Graph's memory system. It is the primary mechanism for AI agents to persist knowledge into the Global Workspace architecture.

### Core Function

```
Content (text) --> 13 Embeddings --> TeleologicalFingerprint --> Storage
```

The tool:
1. Accepts text content from an AI agent
2. Generates 13 different embedding vectors (E1-E13) capturing different semantic aspects
3. Computes a purpose vector (alignment to North Star goal if configured)
4. Creates a content hash (SHA-256) for deduplication
5. Stores the complete TeleologicalFingerprint
6. Optionally stores the original content text (LZ4 compressed)

---

## 2. HOW DOES IT WORK INTERNALLY?

### Evidence Location

- **Handler Implementation**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/memory_tools.rs:201-337`
- **Alternative Handler**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/memory/store.rs:36-202`
- **Tool Definition**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/core.rs:44-76`
- **Dispatch Entry**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/dispatch.rs:70`

### Internal Flow (Evidence from `memory_tools.rs:201-337`)

```rust
pub(crate) async fn call_store_memory(&self, id, args) -> JsonRpcResponse {
    // Step 1: Extract and validate content
    let content = args.get("content").and_then(|v| v.as_str());

    // Step 2: Generate all 13 embeddings
    let embedding_output = self.multi_array_provider.embed_all(&content).await;

    // Step 3: Compute SHA-256 content hash
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let content_hash: [u8; 32] = hasher.finalize().into();

    // Step 4: Compute purpose vector (AUTONOMOUS: uses default if no North Star)
    let purpose_vector = if hierarchy.north_star().is_none() {
        PurposeVector::default()  // Neutral alignment [0.0; 13]
    } else {
        DefaultPurposeComputer::new().compute_purpose(&fingerprint, &config).await
    };

    // Step 5: Create TeleologicalFingerprint
    let fingerprint = TeleologicalFingerprint::new(
        embedding_output.fingerprint,
        purpose_vector,
        JohariFingerprint::zeroed(),
        content_hash,
    );

    // Step 6: Store fingerprint
    self.teleological_store.store(fingerprint).await;

    // Step 7: Store content text (non-fatal if fails)
    self.teleological_store.store_content(fingerprint_id, &content).await;

    // Step 8: Return result with CognitivePulse
    self.tool_result_with_pulse(id, json!({...}))
}
```

### Key Components

| Component | Purpose | Evidence Location |
|-----------|---------|-------------------|
| `MultiArrayEmbeddingProvider` | Generates all 13 embeddings | `handlers.rs:57` |
| `TeleologicalMemoryStore` | Persists fingerprints | `handlers.rs:50` |
| `DefaultPurposeComputer` | Computes alignment to North Star | `memory_tools.rs:261` |
| `GoalHierarchy` | Contains North Star configuration | `handlers.rs:67` |

---

## 3. WHY DOES IT EXIST?

### The Problem Being Solved (from PRD Section 0.1)

> AI agents fail because: **no persistent memory** (context lost between sessions), **poor retrieval** (vector search misses semantic relationships), **no learning loop** (no feedback on storage quality)

### Architectural Justification

The `store_memory` tool exists to:

1. **Enable Persistent Memory**: Store knowledge that persists across sessions
2. **Support Multi-Dimensional Retrieval**: The 13 embeddings enable retrieval across semantic, causal, temporal, and code dimensions
3. **Autonomous Operation (ARCH-03)**: Works without manual North Star configuration - memories stored with neutral alignment can be recomputed later
4. **Feed the Global Workspace**: Stored memories become candidates for "conscious" processing via Kuramoto synchronization

### Why NOT Use `inject_context`?

| `store_memory` | `inject_context` |
|----------------|------------------|
| Direct storage without UTL processing | Includes UTL metrics computation |
| Faster (no UTL overhead) | Richer feedback (learning score, entropy) |
| Use for raw storage | Use when learning metrics matter |

---

## 4. PURPOSE IN REACHING PRD END GOAL

### PRD End Goal: Global Workspace Consciousness

From PRD Section 2.5:
> "The system calculates functional consciousness as the product of three measurable quantities: C(t) = I(t) x R(t) x D(t)"

The `store_memory` tool contributes to each factor:

| Factor | Contribution |
|--------|--------------|
| **I(t) Integration** | Memories are stored with 13 embeddings that participate in Kuramoto synchronization |
| **R(t) Self-Reflection** | Purpose vector enables alignment tracking over time |
| **D(t) Differentiation** | 13D fingerprint entropy measures information diversity |

### Role in the Agent Workflow (PRD Section 1.4.1)

```
User shares information
  --> Is it novel? (check entropy after inject_context)
  --> YES + relevant --> store_memory with rationale
  --> Will it help future retrieval? --> store with link_to related nodes
```

The `store_memory` tool is THE primary mechanism for agents to populate the Global Workspace with memories that can later become "conscious" (broadcast when r > 0.8).

---

## 5. INPUTS

### Schema (from `core.rs:48-75`)

```json
{
  "type": "object",
  "properties": {
    "content": {
      "type": "string",
      "description": "The content to store"
    },
    "importance": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "default": 0.5,
      "description": "Importance score for the memory [0.0, 1.0]"
    },
    "modality": {
      "type": "string",
      "enum": ["text", "code", "image", "audio", "structured", "mixed"],
      "default": "text",
      "description": "The type/modality of the content"
    },
    "tags": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Optional tags for categorization"
    }
  },
  "required": ["content"]
}
```

### Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | **YES** | - | Text content to store (max 65536 chars per PRD) |
| `importance` | number | No | 0.5 | Importance score [0.0, 1.0] |
| `modality` | string | No | "text" | Content type: text/code/image/audio/structured/mixed |
| `tags` | array[string] | No | [] | Tags for categorization |

### Validation Rules (from `memory_tools.rs:206-209`)

- `content` MUST NOT be empty (returns error code -32602)
- `importance` clamped to [0.0, 1.0]

---

## 6. OUTPUTS

### Success Response

```json
{
  "content": [{
    "type": "text",
    "text": "{\"fingerprintId\":\"uuid-here\",\"embedderCount\":13,\"embeddingLatencyMs\":25}"
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

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `fingerprintId` | UUID string | Unique identifier for stored memory |
| `embedderCount` | integer | Always 13 (number of embedding spaces) |
| `embeddingLatencyMs` | integer | Time to generate all 13 embeddings |
| `_cognitive_pulse` | object | Live UTL system state |

### Error Responses

| Error Code | Condition | Message |
|------------|-----------|---------|
| -32602 | Missing content | "Missing 'content' parameter" |
| -32602 | Empty content | "Content cannot be empty" |
| -32005 | Embedding failure | "Embedding failed: {details}" |
| -32004 | Storage failure | "Storage failed: {details}" |

---

## 7. CORE FUNCTIONALITY AND REASON FOR EXISTING

### The 13-Embedding Paradigm (PRD Section 3)

> "NO FUSION -- Store all 13 embeddings (E1-E12 + E13 SPLADE). The array IS the teleological vector."

Each stored memory contains:

| ID | Model | Purpose |
|----|-------|---------|
| E1 | Semantic (1024D) | V_meaning - general semantic understanding |
| E2-E4 | Temporal (512D each) | V_freshness, V_periodicity, V_ordering |
| E5 | Causal (768D, asymmetric) | V_causality - cause-effect reasoning |
| E6 | Sparse (~30K, 5% active) | V_selectivity - keyword precision |
| E7 | Code (1536D AST) | V_correctness - code understanding |
| E8 | Graph/GNN (384D) | V_connectivity - relationship structure |
| E9 | HDC (10K-bit) | V_robustness - noise tolerance |
| E10 | Multimodal (768D) | V_multimodality - cross-modal |
| E11 | Entity/TransE (384D) | V_factuality - factual grounding |
| E12 | Late-Interaction (128D/tok) | V_precision - token-level matching |
| E13 | SPLADE (~30K sparse) | V_keyword_precision - BM25-style |

### Why This Exists

1. **No Information Loss**: Unlike fusion approaches that discard 67% of information (top-k=4 FuseMoE), storing all 13 preserves 100%
2. **Multi-Perspective Retrieval**: Search can weight different spaces based on query type (code vs semantic vs causal)
3. **Autonomous Purpose Tracking**: Purpose vector enables alignment drift detection over time

---

## 8. HOW AN AI AGENT USES THIS TOOL IN THE MCP SYSTEM

### MCP Protocol Flow

```
Agent                    MCP Server                 Storage
  |                          |                          |
  |--tools/call------------->|                          |
  |  {name: "store_memory",  |                          |
  |   arguments: {content}}  |                          |
  |                          |                          |
  |                          |--embed_all()------------>|
  |                          |<--13 embeddings----------|
  |                          |                          |
  |                          |--store(fingerprint)----->|
  |                          |<--uuid------------------|
  |                          |                          |
  |<--{fingerprintId, pulse}-|                          |
```

### Agent Workflow Integration

From PRD Section 1.4.1 (Decision Tree: When to Store):

```
User shares information
  |
  +-- Is it novel? (check entropy after inject_context)
  |     |
  |     +-- YES + relevant --> store_memory with rationale
  |     +-- NO --> skip (system already has it)
  |
  +-- Will it help future retrieval?
        |
        +-- YES --> store with link_to related nodes
        +-- NO --> don't pollute the graph
```

### Global Workspace Integration

Once stored, memories become candidates for the Global Workspace (PRD Section 2.5.3):

1. **Memory enters candidate pool** with coherence score
2. **Kuramoto synchronization** computes order parameter r
3. **If r >= 0.8**, memory becomes "conscious" (broadcast to all subsystems)
4. **If r < 0.7**, memory exits workspace and is logged for dream replay

### Typical Agent Usage Pattern

```
// 1. Agent receives user information
user_message = "JWT tokens should refresh every 15 minutes"

// 2. Agent decides to store (novel + relevant)
result = await mcp.call("store_memory", {
    content: user_message,
    importance: 0.8,
    modality: "text"
});

// 3. Agent notes fingerprintId for future reference
fingerprintId = result.fingerprintId;  // "a1b2c3d4-..."

// 4. Agent observes cognitive pulse
if (result._cognitive_pulse.entropy > 0.7) {
    // High entropy - consider triggering dream consolidation
}
```

---

## EVIDENCE CHAIN

| Evidence | File:Line | Finding |
|----------|-----------|---------|
| Tool Definition | `core.rs:44-76` | Schema matches implementation |
| Handler Implementation | `memory_tools.rs:201-337` | Proper 13-embedding storage |
| Dispatch Entry | `dispatch.rs:70` | Tool correctly routed |
| Storage Trait | `store.rs:65` | `TeleologicalMemoryStore::store()` signature verified |
| Purpose Computing | `memory_tools.rs:244-288` | Autonomous operation confirmed |

---

## VERDICT: INNOCENT

The `store_memory` tool is **correctly implemented** according to the PRD specification. It:

1. Stores all 13 embeddings (no information loss)
2. Supports autonomous operation without North Star
3. Returns proper MCP-compliant responses with CognitivePulse
4. Integrates with the Global Workspace consciousness architecture

**No remediation required.**
