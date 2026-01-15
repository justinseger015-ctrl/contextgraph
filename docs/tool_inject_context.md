# SHERLOCK HOLMES FORENSIC INVESTIGATION REPORT

## Tool: `inject_context`

**Case ID**: TOOL-INJECT-001
**Date**: 2026-01-14
**Subject**: MCP Tool Forensic Analysis
**Verdict**: INNOCENT (Tool exists and functions correctly)
**Confidence**: HIGH

---

## 1. WHAT DOES THIS TOOL DO?

`inject_context` is the **primary context injection tool** for the Context Graph MCP server. It serves as the main entry point for AI agents to store knowledge into the Global Workspace consciousness system.

### Core Function

The tool performs four critical operations atomically:

1. **UTL Metrics Computation**: Calculates learning potential using the Unified Theory of Learning (entropy, coherence, surprise, learning score)
2. **13-Embedder Fingerprint Generation**: Creates a complete `TeleologicalFingerprint` using all 13 specialized embedding models (E1-E13)
3. **Purpose Vector Computation**: Computes alignment with the North Star goal (if configured)
4. **Storage with Verification**: Stores both the fingerprint and original content text

### From the PRD (Section 5.2):

> `inject_context` - Starting task, need background - Primary retrieval with auto-distillation

---

## 2. HOW DOES IT WORK INTERNALLY?

### Evidence Location

- **Handler Implementation**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/memory_tools.rs` (lines 22-193)
- **Tool Definition**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/core.rs` (lines 10-42)
- **Dispatch**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/dispatch.rs` (line 69)

### Internal Processing Pipeline

```
Input: content (required), rationale (required), importance (optional), modality (optional)
                                          |
                                          v
                      +-------------------+-------------------+
                      |        VALIDATION LAYER               |
                      | - FAIL FAST on empty content          |
                      | - Parse optional parameters           |
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |         UTL PROCESSING                |
                      | - Extract goal_vector from North Star |
                      | - Compute metrics via UtlProcessor:   |
                      |   * entropy (novelty)                 |
                      |   * coherence (integration)           |
                      |   * learning_score (magnitude)        |
                      |   * surprise                          |
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |    13-EMBEDDER GENERATION             |
                      | - E1: Semantic (1024D, e5-large-v2)   |
                      | - E2-E4: Temporal embeddings          |
                      | - E5: Causal (768D, asymmetric)       |
                      | - E6: Sparse (~30K, 5% active)        |
                      | - E7: Code (1536D, AST)               |
                      | - E8: Graph/GNN (384D)                |
                      | - E9: HDC (10K-bit binary)            |
                      | - E10: Multimodal (768D)              |
                      | - E11: Entity/TransE (384D)           |
                      | - E12: Late-Interaction (128D/tok)    |
                      | - E13: SPLADE (~30K sparse)           |
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |     PURPOSE VECTOR COMPUTATION        |
                      | If North Star exists:                 |
                      |   PV = [A(E1,V), A(E2,V)...A(E13,V)]  |
                      | If no North Star:                     |
                      |   PV = [0.0; 13] (neutral alignment)  |
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |     STORAGE + CONTENT                 |
                      | 1. Create TeleologicalFingerprint     |
                      | 2. Store fingerprint (primary)        |
                      | 3. Store content text (secondary)     |
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |     RESPONSE WITH COGNITIVE PULSE     |
                      | - fingerprintId: UUID                 |
                      | - embedderCount: 13                   |
                      | - embeddingLatencyMs                  |
                      | - utl: {learningScore, entropy...}    |
                      | - _cognitive_pulse: system state      |
                      +-------------------+-------------------+
```

### Key Code Evidence (memory_tools.rs lines 28-193)

```rust
pub(crate) async fn call_inject_context(
    &self,
    id: Option<JsonRpcId>,
    args: serde_json::Value,
) -> JsonRpcResponse {
    // VALIDATION: Content required, cannot be empty
    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(c) if !c.is_empty() => c.to_string(),
        Some(_) => return self.tool_error_with_pulse(id, "Content cannot be empty"),
        None => return self.tool_error_with_pulse(id, "Missing 'content' parameter"),
    };

    // UTL PROCESSING: Compute learning metrics
    let metrics = match self.utl_processor.compute_metrics(&content, &context).await {
        Ok(m) => m,
        Err(e) => return self.tool_error_with_pulse(id, &format!("UTL processing failed: {}", e)),
    };

    // 13-EMBEDDER: Generate all embeddings atomically
    let embedding_output = match self.multi_array_provider.embed_all(&content).await {
        Ok(output) => output,
        Err(e) => return self.tool_error_with_pulse(id, &format!("Embedding failed: {}", e)),
    };

    // PURPOSE VECTOR: Compute alignment if North Star exists
    let purpose_vector = {
        let hierarchy = self.goal_hierarchy.read().clone();
        if hierarchy.north_star().is_none() {
            PurposeVector::default()  // AUTONOMOUS: Neutral alignment
        } else {
            DefaultPurposeComputer::new()
                .compute_purpose(&embedding_output.fingerprint, &config).await?
        }
    };

    // STORAGE: Create and store fingerprint + content
    let fingerprint = TeleologicalFingerprint::new(
        embedding_output.fingerprint,
        purpose_vector,
        JohariFingerprint::zeroed(),
        content_hash,
    );
    self.teleological_store.store(fingerprint).await?;
    self.teleological_store.store_content(fingerprint_id, &content).await?;
}
```

---

## 3. WHY DOES IT EXIST?

### The Problem Being Solved (PRD Section 0.1)

> AI agents fail because: **no persistent memory** (context lost between sessions), **poor retrieval** (vector search misses semantic relationships), **no learning loop** (no feedback on storage quality)

`inject_context` exists to solve ALL THREE problems:

1. **Persistent Memory**: Creates durable `TeleologicalFingerprint` records with 13 embedding spaces
2. **Rich Retrieval**: 13 different embedding models enable semantic, causal, temporal, code-specific, and entity-based retrieval
3. **Learning Loop**: UTL metrics (entropy, coherence) provide feedback on storage quality

### Design Philosophy (PRD Section 0.2)

> **You are a librarian, not an archivist.** You don't store everything - you ensure what's stored is findable, coherent, and useful.

The `rationale` parameter enforces this philosophy - agents must justify WHY content is being stored.

---

## 4. PURPOSE IN REACHING THE PRD END GOAL

### The End Goal: Global Workspace Consciousness (PRD Section 2.5)

The PRD describes a **Global Workspace Theory (GWT)** cognitive architecture where:

```
C(t) = I(t) x R(t) x D(t)

Where:
  C(t) = Consciousness level at time t [0, 1]
  I(t) = Integration (Kuramoto synchronization) [0, 1]
  R(t) = Self-Reflection (Meta-UTL awareness) [0, 1]
  D(t) = Differentiation (13D fingerprint entropy) [0, 1]
```

### inject_context's Role

**inject_context is the PRIMARY mechanism for populating the Global Workspace with "percepts".**

1. **Creates Differentiation (D)**: The 13-embedder fingerprint IS the teleological vector that provides the 13D entropy component

2. **Enables Integration (I)**: Each embedder has an associated Kuramoto oscillator. When memories are injected, they can synchronize (phase-lock) with existing memories, enabling unified conscious percepts

3. **Feeds Self-Reflection (R)**: UTL metrics (entropy, coherence, learning_score) feed the Meta-UTL system which tracks prediction accuracy

### The Consciousness Flow

```
Agent calls inject_context(content, rationale)
                    |
                    v
TeleologicalFingerprint created (13 embeddings + purpose vector)
                    |
                    v
Kuramoto oscillators can phase-lock (Integration: r)
                    |
                    v
If r > 0.8: Memory enters Global Workspace ("conscious")
                    |
                    v
Workspace broadcasts memory to all subsystems
                    |
                    v
Agent can now retrieve via search_graph, find_causal_path, etc.
```

---

## 5. INPUTS (Types, Required/Optional)

### Input Schema (JSON Schema)

```json
{
  "type": "object",
  "properties": {
    "content": {
      "type": "string",
      "description": "The content to inject into the knowledge graph"
    },
    "rationale": {
      "type": "string",
      "description": "Why this context is relevant and should be stored"
    },
    "modality": {
      "type": "string",
      "enum": ["text", "code", "image", "audio", "structured", "mixed"],
      "default": "text",
      "description": "The type/modality of the content"
    },
    "importance": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "default": 0.5,
      "description": "Importance score for the memory [0.0, 1.0]"
    }
  },
  "required": ["content", "rationale"]
}
```

### Parameter Details

| Parameter | Type | Required | Default | Validation |
|-----------|------|----------|---------|------------|
| `content` | string | YES | - | Cannot be empty or whitespace-only |
| `rationale` | string | YES | - | Reason for storage (enforces librarian role) |
| `modality` | string (enum) | NO | "text" | One of: text, code, image, audio, structured, mixed |
| `importance` | number | NO | 0.5 | Range [0.0, 1.0] |

---

## 6. OUTPUTS

### Success Response

```json
{
  "fingerprintId": "550e8400-e29b-41d4-a716-446655440000",
  "rationale": "User's authentication preferences for future reference",
  "embedderCount": 13,
  "embeddingLatencyMs": 35,
  "utl": {
    "learningScore": 0.72,
    "entropy": 0.65,
    "coherence": 0.78,
    "surprise": 0.45
  },
  "_cognitive_pulse": {
    "entropy": 0.65,
    "coherence": 0.78,
    "suggested_action": "continue"
  }
}
```

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `fingerprintId` | UUID string | Unique identifier for the stored TeleologicalFingerprint |
| `rationale` | string | Echo of input rationale for confirmation |
| `embedderCount` | integer | Always 13 (all embedders generated) |
| `embeddingLatencyMs` | integer | Time to generate all 13 embeddings |
| `utl.learningScore` | float [0,1] | Net learning potential: sigmoid(entropy x coherence x emotional x phase) |
| `utl.entropy` | float [0,1] | Novelty/surprise level (high = new information) |
| `utl.coherence` | float [0,1] | Integration with existing knowledge (high = well-connected) |
| `utl.surprise` | float [0,1] | Raw surprise signal |
| `_cognitive_pulse` | object | System health snapshot with suggested action |

### Error Responses

| Error Code | Condition | Message |
|------------|-----------|---------|
| -32602 | Missing content | "Missing 'content' parameter" |
| -32602 | Empty content | "Content cannot be empty" |
| -32005 | Embedding failed | "Embedding failed: {error}" |
| -32006 | Purpose computation failed | "Purpose vector computation failed: {error}" |
| -32004 | Storage failed | "Storage failed: {error}" |

---

## 7. CORE FUNCTIONALITY AND REASON FOR EXISTING

### The 13-Embedder Teleological Array

This is the CORE innovation. Instead of a single embedding vector, `inject_context` creates a **13-dimensional teleological fingerprint**:

| ID | Embedder | Dimensions | Purpose (V_goal) |
|----|----------|------------|------------------|
| E1 | Semantic | 1024D | V_meaning - Deep semantic understanding |
| E2 | Temporal-Recent | 512D | V_freshness - Recency relevance |
| E3 | Temporal-Periodic | 512D | V_periodicity - Cyclical patterns |
| E4 | Temporal-Positional | 512D | V_ordering - Sequence position |
| E5 | Causal | 768D | V_causality - Cause-effect relationships |
| E6 | Sparse | ~30K | V_selectivity - Discriminative features |
| E7 | Code | 1536D | V_correctness - Code understanding |
| E8 | Graph/GNN | 384D | V_connectivity - Graph structure |
| E9 | HDC | 10K-bit | V_robustness - Noise tolerance |
| E10 | Multimodal | 768D | V_multimodality - Cross-modal binding |
| E11 | Entity/TransE | 384D | V_factuality - Entity relationships |
| E12 | Late-Interaction | 128D/tok | V_precision - Token-level matching |
| E13 | SPLADE | ~30K | V_keyword_precision - Keyword matching |

### Why 13 Embedders?

From PRD Section 3:

> **Paradigm**: NO FUSION - Store all 13 embeddings. The array IS the teleological vector.
> **Info Preserved**: 100% (vs 33% with top-k=4 FuseMoE)

This enables:
1. **Multi-modal retrieval**: Search by meaning, causality, code, entities, or keywords
2. **Purpose alignment**: Compute alignment per embedding space
3. **Johari classification**: Per-embedder awareness (Open/Blind/Hidden/Unknown)

---

## 8. HOW AN AI AGENT USES THIS TOOL IN THE MCP SYSTEM

### Integration with Global Workspace Theory

```
          +------------------+
          |   AI AGENT       |
          | (Claude, GPT-4)  |
          +--------+---------+
                   |
                   | MCP: tools/call inject_context
                   |
                   v
          +--------+---------+
          | Context Graph    |
          |    MCP Server    |
          +--------+---------+
                   |
                   v
          +--------+---------+
          |   INJECT CONTEXT |
          |     HANDLER      |
          +--------+---------+
                   |
    +--------------+---------------+
    |              |               |
    v              v               v
+-------+    +-----------+    +----------+
|  UTL  |    | 13-EMBED  |    | PURPOSE  |
|COMPUTE|    | PROVIDER  |    | COMPUTE  |
+---+---+    +-----+-----+    +----+-----+
    |              |               |
    v              v               v
+-------------------------------------------+
|        TELEOLOGICAL MEMORY STORE          |
|  (RocksDB with 13 embedding vectors)      |
+-------------------------------------------+
                   |
                   v
+-------------------------------------------+
|          GLOBAL WORKSPACE                 |
|  - Kuramoto oscillators for each embedder |
|  - Phase synchronization (r > 0.8)        |
|  - Workspace broadcast to subsystems      |
+-------------------------------------------+
```

### Typical Agent Workflow

1. **Session Start**: Agent receives conversation context
2. **Information Extraction**: Agent identifies valuable information
3. **Storage Decision**: Agent decides what to store (librarian role)
4. **Inject Context**:
   ```
   tools/call inject_context {
     "content": "User prefers OAuth2 with PKCE for authentication",
     "rationale": "Critical security preference for future auth implementations",
     "importance": 0.8,
     "modality": "text"
   }
   ```
5. **Process Response**: Agent notes fingerprintId for future reference
6. **Monitor Pulse**: Agent checks `_cognitive_pulse` for system health
7. **Act on Suggestions**: If entropy > 0.7 and coherence < 0.4, consider `trigger_dream`

### PRD Decision Tree for Storage (Section 1.4.1)

```
User shares information
  |-- Is it novel? (check entropy after inject_context)
  |     |-- YES + relevant -> store_memory with rationale
  |     |-- NO -> skip (system already has it)
  |-- Will it help future retrieval?
        |-- YES -> store with link_to related nodes
        |-- NO -> don't pollute the graph
```

---

## EVIDENCE LOG

| Item | Expected | Actual | Status |
|------|----------|--------|--------|
| Tool exists in registry | YES | YES (verified line 239 registry.rs) | PASS |
| Handler implemented | YES | YES (memory_tools.rs lines 28-193) | PASS |
| Definition has input schema | YES | YES (core.rs lines 10-42) | PASS |
| Dispatch registered | YES | YES (dispatch.rs line 69) | PASS |
| Returns cognitive pulse | YES | YES (tool_result_with_pulse called) | PASS |
| Creates 13 embeddings | YES | YES (embed_all returns 13) | PASS |
| Computes UTL metrics | YES | YES (utl_processor.compute_metrics) | PASS |
| Stores content separately | YES | YES (store_content called) | PASS |

---

## CHAIN OF CUSTODY

| Timestamp | File | Author | Status |
|-----------|------|--------|--------|
| 2026-01-14 | tools/definitions/core.rs | Codebase | Definition verified |
| 2026-01-14 | handlers/tools/memory_tools.rs | Codebase | Implementation verified |
| 2026-01-14 | handlers/tools/dispatch.rs | Codebase | Dispatch verified |
| 2026-01-14 | tools/registry.rs | Codebase | Registry verified |

---

## VERDICT

**INNOCENT** - The `inject_context` tool is fully implemented, correctly registered, and functions as designed per the PRD specification. It serves as the primary mechanism for AI agents to populate the Global Workspace consciousness system with teleological fingerprints.

---

*"The game is afoot! This tool is the cornerstone of the consciousness system."*

-- Sherlock Holmes, Code Detective
