# Context Graph MCP Tools Reference

This document provides a comprehensive reference for all 41 MCP (Model Context Protocol) tools exposed by the Context Graph server. The server follows the MCP 2024-11-05 protocol specification.

## Table of Contents

1. [Core Memory Tools](#core-memory-tools)
2. [System Status Tools](#system-status-tools)
3. [GWT/Consciousness Tools](#gwtconsciousness-tools)
4. [Adaptive Threshold Calibration (ATC) Tools](#adaptive-threshold-calibration-atc-tools)
5. [Dream Tools](#dream-tools)
6. [Neuromodulation Tools](#neuromodulation-tools)
7. [Steering Tools](#steering-tools)
8. [Causal Inference Tools](#causal-inference-tools)
9. [North Star Goal Tools](#north-star-goal-tools)
10. [Teleological Tools](#teleological-tools)
11. [Autonomous Tools](#autonomous-tools)
12. [MCP Protocol Compliance](#mcp-protocol-compliance)
13. [Cognitive Pulse (Every Response)](#cognitive-pulse-every-response)
14. [The 13 Embedders](#the-13-embedders)
15. [Error Codes](#error-codes)
16. [Claude Code Hook Integration](#claude-code-hook-integration)

---

## Core Memory Tools

### `inject_context`

Inject context into the knowledge graph with UTL (Unified Theory of Learning) processing. Analyzes content for learning potential and stores with computed metrics.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | - | The content to inject into the knowledge graph |
| `rationale` | string | Yes | - | Why this context is relevant and should be stored |
| `modality` | enum | No | "text" | Content type: `text`, `code`, `image`, `audio`, `structured`, `mixed` |
| `importance` | number | No | 0.5 | Importance score [0.0, 1.0] |

**Response includes:**
- `fingerprintId` - Unique identifier for the stored memory
- `utl` object with `learningScore`, `entropy`, `coherence`, `surprise`

**Example:**
```json
{
  "name": "inject_context",
  "arguments": {
    "content": "The Kuramoto model describes synchronization of coupled oscillators",
    "rationale": "Core concept for consciousness modeling",
    "modality": "text",
    "importance": 0.8
  }
}
```

---

### `store_memory`

Store a memory node directly in the knowledge graph without UTL processing. Use this for raw storage when UTL analysis is not needed.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | - | The content to store |
| `importance` | number | No | 0.5 | Importance score [0.0, 1.0] |
| `modality` | enum | No | "text" | Content type: `text`, `code`, `image`, `audio`, `structured`, `mixed` |
| `tags` | array[string] | No | - | Optional tags for categorization |

**Example:**
```json
{
  "name": "store_memory",
  "arguments": {
    "content": "function fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }",
    "modality": "code",
    "tags": ["algorithm", "recursion", "math"]
  }
}
```

---

### `search_graph`

Search the knowledge graph using semantic similarity. Returns nodes matching the query with relevance scores.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | The search query text |
| `topK` | integer | No | 10 | Maximum results to return (1-100) |
| `minSimilarity` | number | No | 0.0 | Minimum similarity threshold [0.0, 1.0] |
| `modality` | enum | No | - | Filter results by modality |

**Example:**
```json
{
  "name": "search_graph",
  "arguments": {
    "query": "neural oscillator synchronization",
    "topK": 5,
    "minSimilarity": 0.7
  }
}
```

---

## System Status Tools

### `get_memetic_status`

Get current system status with LIVE UTL metrics including entropy (novelty), coherence (understanding), learning score (magnitude), Johari quadrant classification, consolidation phase, and suggested action. Also returns node count and 5-layer bio-nervous system status.

**Parameters:** None required

**Response includes:**
- `phase` - Current lifecycle phase
- `fingerprintCount` - Total stored fingerprints
- `utl` - Object with `entropy`, `coherence`, `learningScore`, `johariQuadrant`, `consolidationPhase`, `suggestedAction`
- `layers` - Status of 5-layer bio-nervous system

---

### `get_graph_manifest`

Get the 5-layer bio-nervous system architecture description and current layer statuses.

**Parameters:** None required

**Response includes:**
- Architecture description
- Layer statuses: Perception, Memory, Reasoning, Meta, Action

---

### `utl_status`

Query current UTL (Unified Theory of Learning) system state including lifecycle phase, entropy, coherence, learning score, Johari quadrant, and consolidation phase.

**Parameters:** None required

---

## GWT/Consciousness Tools

These tools implement Global Workspace Theory (GWT) and Kuramoto oscillator network for computational consciousness modeling.

### `get_consciousness_state`

Get current consciousness state including Kuramoto sync (r), consciousness level (C), meta-cognitive score, differentiation, workspace status, and identity coherence.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | No | default | Session ID for consciousness tracking |

**Response includes:**
- `r` - Kuramoto order parameter (synchronization measure)
- `C` - Consciousness level
- `meta_score` - Meta-cognitive evaluation
- `differentiation` - Information differentiation measure
- `workspace` - Global workspace status
- `identity` - Identity coherence metrics

---

### `get_kuramoto_sync`

Get Kuramoto oscillator network synchronization state including order parameter (r), mean phase (psi), all 13 oscillator phases, natural frequencies, and coupling strength.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | No | default | Session ID |

**Response includes:**
- `r` - Order parameter [0,1] (1 = fully synchronized)
- `psi` - Mean phase
- `phases` - Array of 13 oscillator phases
- `natural_freqs` - Natural frequencies per oscillator
- `coupling` - Current coupling strength K

---

### `get_workspace_status`

Get Global Workspace status including active memory, competing candidates, broadcast state, and coherence threshold. Returns WTA (Winner-Take-All) selection details.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | No | default | Session ID |

**Response includes:**
- `active_memory` - Currently active memory in workspace
- `is_broadcasting` - Whether workspace is broadcasting
- `coherence_threshold` - Current threshold for competition
- `conflict_memories` - Competing memory candidates

---

### `get_ego_state`

Get Self-Ego Node state including purpose vector (13D), identity continuity, coherence with actions, and trajectory length. Used for identity monitoring.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | No | default | Session ID |

**Response includes:**
- `purpose_vector` - 13-dimensional purpose embedding
- `identity_status` - Current identity state
- `coherence_with_actions` - How aligned actions are with identity
- `trajectory_length` - Length of identity trajectory

---

### `trigger_workspace_broadcast`

Trigger winner-take-all workspace broadcast with a specific memory. Forces memory into workspace competition.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string (UUID) | Yes | - | UUID of memory to broadcast |
| `importance` | number | No | 0.8 | Importance score [0.0, 1.0] |
| `alignment` | number | No | 0.8 | North star alignment score [0.0, 1.0] |
| `force` | boolean | No | false | Force broadcast even if below threshold |

---

### `adjust_coupling`

Adjust Kuramoto oscillator network coupling strength K. Higher K leads to faster synchronization. K is clamped to [0, 10].

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `new_K` | number | Yes | - | New coupling strength (0-10) |

**Response includes:**
- `old_K` - Previous coupling strength
- `new_K` - Applied coupling strength
- `predicted_r` - Predicted order parameter after adjustment

---

## Adaptive Threshold Calibration (ATC) Tools

### `get_threshold_status`

Get current ATC threshold status including all thresholds, calibration state, and adaptation metrics. Returns per-embedder temperatures, drift scores, and bandit exploration stats.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `domain` | enum | No | "General" | Domain: `Code`, `Medical`, `Legal`, `Creative`, `Research`, `General` |
| `embedder_id` | integer | No | - | Specific embedder (1-13) for detailed info |

---

### `get_calibration_metrics`

Get calibration quality metrics: ECE (Expected Calibration Error), MCE (Maximum Calibration Error), Brier Score, drift scores per threshold, and calibration status.

**Targets:** ECE < 0.05 (excellent), < 0.10 (good)

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `timeframe` | enum | No | "24h" | Timeframe: `1h`, `24h`, `7d`, `30d` |

---

### `trigger_recalibration`

Manually trigger recalibration at a specific ATC level.

**Levels:**
- Level 1: EWMA drift adjustment
- Level 2: Temperature scaling
- Level 3: Thompson Sampling exploration
- Level 4: Bayesian meta-optimization

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `level` | integer | Yes | - | ATC level (1-4) |
| `domain` | enum | No | "General" | Domain context |

---

## Dream Tools

Implements sleep-inspired memory consolidation cycles (NREM + REM phases).

### `trigger_dream`

Manually trigger a dream consolidation cycle. System must be idle (activity < 0.15). Executes NREM (3 min) + REM (2 min) phases. Returns DreamReport with metrics. Aborts automatically on external query (wake latency < 100ms).

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `force` | boolean | No | false | Force dream even if activity is above threshold (not recommended) |

---

### `get_dream_status`

Get current dream system status including state (Awake/NREM/REM/Waking), GPU usage, activity level, and time since last dream cycle.

**Parameters:** None required

---

### `abort_dream`

Abort the current dream cycle. Must complete wake within 100ms (constitution mandate). Returns wake latency and partial dream report.

**Parameters:** None required

---

### `get_amortized_shortcuts`

Get shortcut candidates from amortized learning. Returns paths traversed 5+ times with 3+ hops that qualify for direct edge creation.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_confidence` | number | No | 0.7 | Minimum confidence threshold |
| `limit` | integer | No | 20 | Maximum shortcuts to return (1-100) |

---

## Neuromodulation Tools

### `get_neuromodulation_state`

Get current neuromodulation state including all 4 modulators:
- **Dopamine** → hopfield.beta [1,5]
- **Serotonin** → space_weights [0,1]
- **Noradrenaline** → attention.temp [0.5,2]
- **Acetylcholine** → utl.lr [0.001,0.002] (read-only, managed by GWT)

**Parameters:** None required

---

### `adjust_neuromodulator`

Adjust a specific neuromodulator level. ACh is read-only (managed by GWT). Changes are clamped to constitution-mandated ranges.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `modulator` | enum | Yes | - | Which modulator: `dopamine`, `serotonin`, `noradrenaline` |
| `delta` | number | Yes | - | Amount to add (positive) or subtract (negative) |

---

## Steering Tools

### `get_steering_feedback`

Get steering feedback from the Gardener (graph health), Curator (memory quality), and Assessor (performance) components. Returns a SteeringReward in [-1, 1] with detailed component scores and recommendations.

**Parameters:** None required

---

## Causal Inference Tools

### `omni_infer`

Perform omni-directional causal inference. Supports 5 directions:
- **forward** - A→B effect
- **backward** - B→A cause
- **bidirectional** - A↔B mutual
- **bridge** - Cross-domain inference
- **abduction** - Best hypothesis

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | string (UUID) | Yes | - | Source node UUID |
| `target` | string (UUID) | No | - | Target node UUID (required for forward/backward/bidirectional) |
| `direction` | enum | No | "forward" | Inference direction |

---

## North Star Goal Tools (DEPRECATED - REMOVED)

> **CRITICAL: Manual North Star tools have been REMOVED**
>
> The following tools no longer exist: `set_north_star`, `get_north_star`, `update_north_star`,
> `delete_north_star`, `init_north_star_from_documents`, `get_goal_hierarchy`
>
> **Why removed:** Manual North Star creation produced single 1024D embeddings that **CANNOT** be
> meaningfully compared to 13-embedder teleological arrays. This is comparing apples to oranges.
>
> **The problem:**
> - Manual North Star = ONE vector (1024D from text-embedding-3-large)
> - Teleological fingerprints = 13 DIFFERENT embeddings from 13 DIFFERENT models
> - Each embedder has different dimensions: E1 (1024D), E5 (768D), E7 (1536D), E9 (binary), E13 (sparse ~30K)
> - Cosine similarity between a manual 1024D vector and a multi-dimensional teleological array is meaningless
>
> **The solution:** Use the autonomous system (see Autonomous Tools below) which works directly
> with teleological embeddings for apples-to-apples comparisons.

### Correct Approach: Autonomous Purpose Discovery

Instead of manually setting a North Star, the system **discovers purpose autonomously** from stored
teleological fingerprints:

1. **Store memories** with `store_memory` - each gets a full 13-embedder teleological fingerprint
2. **Bootstrap** with `auto_bootstrap_north_star` - discovers emergent purpose patterns from stored fingerprints
3. **Monitor alignment** with `get_alignment_drift` - tracks drift in 13D purpose vector space
4. **Correct drift** with `trigger_drift_correction` - operates entirely within teleological space

**Valid comparisons (apples-to-apples):**
- `PurposeVector ↔ PurposeVector` (13D alignment signatures)
- `TeleologicalFingerprint ↔ TeleologicalFingerprint` (full 13-array)
- `E_i ↔ E_i` (same embedder only)

---

## Teleological Tools

These tools implement the 13-embedder fusion system for multi-perspective semantic understanding.

### `search_teleological`

Perform teleological matrix search across all 13 embedder dimensions. Computes cross-correlation similarity at multiple levels.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query_content` | string | No | - | Content to search for (will be embedded) |
| `query_vector_id` | string | No | - | Alternative: ID of existing teleological vector |
| `strategy` | enum | No | "adaptive" | Search strategy: `cosine`, `euclidean`, `synergy_weighted`, `group_hierarchical`, `cross_correlation_dominant`, `tucker_compressed`, `adaptive` |
| `scope` | enum | No | "full" | Comparison scope: `full`, `purpose_vector_only`, `cross_correlations_only`, `group_alignments_only` |
| `specific_groups` | array[enum] | No | - | Compare only specific groups: `factual`, `temporal`, `causal`, `relational`, `qualitative`, `implementation` |
| `specific_embedder` | integer | No | - | Compare single embedder (0-12) |
| `weight_purpose` | number | No | 0.4 | Weight for purpose vector similarity [0,1] |
| `weight_correlations` | number | No | 0.35 | Weight for cross-correlation similarity [0,1] |
| `weight_groups` | number | No | 0.15 | Weight for group alignments similarity [0,1] |
| `min_similarity` | number | No | 0.3 | Minimum similarity threshold |
| `max_results` | integer | No | 20 | Maximum results (1-1000) |
| `include_breakdown` | boolean | No | true | Include per-component similarity breakdown |

---

### `compute_teleological_vector`

Compute a complete teleological vector from content using all 13 embedders. Returns purpose vector (13D), cross-correlations (78D), group alignments (6D), and optional Tucker core decomposition.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | - | Content to compute teleological vector for |
| `profile_id` | string | No | - | Profile ID for task-specific weighting |
| `compute_tucker` | boolean | No | false | Compute Tucker decomposition |
| `tucker_ranks` | array[integer] | No | [4,4,128] | Tucker decomposition ranks [r1, r2, r3] |
| `include_per_embedder` | boolean | No | false | Include raw per-embedder outputs (large) |

---

### `fuse_embeddings`

Fuse embedding outputs using the synergy matrix and optional profile weights. Supports multiple fusion methods.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string | Yes | - | ID of memory to fuse embeddings for |
| `fusion_method` | enum | No | "hierarchical" | Method: `linear`, `attention`, `gated`, `hierarchical`, `tucker` |
| `profile_id` | string | No | - | Profile ID for task-specific fusion weights |
| `custom_weights` | array[number] | No | - | Custom per-embedder weights [E1..E13] (13 values) |
| `apply_synergy` | boolean | No | true | Apply synergy matrix weighting |
| `store_result` | boolean | No | true | Store fused vector in database |

---

### `update_synergy_matrix`

Update the synergy matrix based on feedback from retrieval success/failure. Implements online learning to adapt cross-embedding relationships.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query_vector_id` | string | Yes | - | ID of query teleological vector |
| `result_vector_id` | string | Yes | - | ID of retrieved result vector |
| `feedback` | enum | Yes | - | User feedback: `relevant`, `not_relevant`, `partially_relevant` |
| `relevance_score` | number | No | - | Fine-grained relevance score [0.0, 1.0] |
| `learning_rate` | number | No | 0.01 | Learning rate for synergy update (0.001-0.5) |
| `update_scope` | enum | No | "contributing_pairs" | Scope: `all_pairs`, `high_synergy_only`, `contributing_pairs` |

---

### `manage_teleological_profile`

Manage teleological profiles for task-specific embedding fusion. Profiles define per-embedder weights, fusion strategy, and group priorities.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | enum | Yes | - | CRUD action: `create`, `read`, `update`, `delete`, `list` |
| `profile_id` | string | No | - | Profile ID (required for read/update/delete) |
| `name` | string | No | - | Human-readable profile name |
| `task_type` | enum | No | - | Predefined type: `code_implementation`, `research`, `creative`, `analysis`, `debugging`, `documentation`, `custom` |
| `embedder_weights` | array[number] | No | - | Per-embedder weights [E1..E13] |
| `group_priorities` | object | No | - | Priority weights for groups: `factual`, `temporal`, `causal`, `relational`, `qualitative`, `implementation` |
| `fusion_strategy` | enum | No | "hierarchical" | Default fusion strategy |

---

## Autonomous Tools

These tools enable autonomous goal maintenance, drift correction, and memory hygiene.

### `auto_bootstrap_north_star`

Bootstrap the autonomous North Star system from an existing North Star goal. Initializes drift detection, pruning, consolidation, and sub-goal discovery services.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `confidence_threshold` | number | No | 0.7 | Minimum confidence for bootstrapping [0,1] |
| `max_candidates` | integer | No | 10 | Maximum candidates to evaluate (1-100) |

---

### `get_alignment_drift`

Get the current alignment drift state including severity, trend, and recommendations. Drift measures how far the system has deviated from the North Star goal alignment.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `timeframe` | enum | No | "24h" | Timeframe: `1h`, `24h`, `7d`, `30d` |
| `include_history` | boolean | No | false | Include full drift history |

**Response includes:**
- `severity` - Current drift severity
- `trend` - Drift trend (improving/worsening)
- `recommendation` - Suggested corrective action

---

### `trigger_drift_correction`

Manually trigger a drift correction cycle. Applies correction strategies based on current drift severity.

**Strategies by severity:**
- Threshold adjustment
- Weight rebalancing
- Goal reinforcement
- Emergency intervention (severe drift)

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `force` | boolean | No | false | Force correction even if drift severity is low |
| `target_alignment` | number | No | - | Target alignment to achieve (adaptive if not set) |

---

### `get_pruning_candidates`

Identify memories that are candidates for pruning based on staleness, low alignment, redundancy, or orphaned status.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | integer | No | 20 | Maximum candidates to return (1-1000) |
| `min_staleness_days` | integer | No | 30 | Minimum age in days for staleness consideration |
| `min_alignment` | number | No | 0.4 | Memories below this alignment are candidates |

---

### `trigger_consolidation`

Trigger memory consolidation to merge similar memories and reduce redundancy.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `max_memories` | integer | No | 100 | Maximum memories to process (1-10000) |
| `strategy` | enum | No | "similarity" | Strategy: `similarity`, `temporal`, `semantic` |
| `min_similarity` | number | No | 0.85 | Minimum similarity threshold for consolidation |

---

### `discover_sub_goals`

Discover potential sub-goals from memory clusters. Analyzes stored memories to find emergent themes and patterns that could become strategic or tactical goals.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_confidence` | number | No | 0.6 | Minimum confidence for a discovered sub-goal [0,1] |
| `max_goals` | integer | No | 5 | Maximum sub-goals to discover (1-20) |
| `parent_goal_id` | string | No | - | Parent goal ID (defaults to North Star) |

---

### `get_autonomous_status`

Get comprehensive status of the autonomous North Star system including all services.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `include_metrics` | boolean | No | false | Include detailed per-service metrics |
| `include_history` | boolean | No | false | Include recent operation history |
| `history_count` | integer | No | 10 | Number of history entries (1-100) |

**Response includes:**
- `overall_health` - System health score and status
- `services` - Status of all autonomous services:
  - `bootstrap_service`
  - `drift_detector`
  - `drift_corrector`
  - `pruning_service`
  - `consolidation_service`
  - `subgoal_discovery`
- `recommendations` - Prioritized action recommendations

---

## MCP Protocol Compliance

All tools follow the MCP 2024-11-05 specification:

1. **Request Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": { ... }
  }
}
```

2. **Response Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{...}"
      }
    ],
    "isError": false
  }
}
```

3. **Tool Discovery:**
Use `tools/list` to get all available tools and their schemas.

---

## Cognitive Pulse (Every Response)

**Every MCP tool response includes a `_cognitive_pulse` field** with live UTL metrics. This provides real-time cognitive state visibility in every response, enabling agents to make informed decisions about next actions.

**Response Structure with Pulse:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{"type": "text", "text": "{...}"}],
    "isError": false,
    "_cognitive_pulse": {
      "entropy": 0.42,
      "coherence": 0.78,
      "learning_score": 0.55,
      "quadrant": "Open",
      "suggested_action": "direct_recall"
    }
  }
}
```

**Pulse Fields:**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `entropy` | float | [0, 1] | Novelty/surprise level (ΔS) |
| `coherence` | float | [0, 1] | Understanding/integration level (ΔC) |
| `learning_score` | float | [0, 1] | UTL learning magnitude |
| `quadrant` | string | - | Johari quadrant classification |
| `suggested_action` | string | - | Recommended next action |

**Johari Quadrant → Action Mapping:**

| Entropy | Coherence | Quadrant | Suggested Action |
|---------|-----------|----------|------------------|
| < 0.5 | > 0.5 | Open | `direct_recall` |
| > 0.5 | < 0.5 | Blind | `trigger_dream` |
| < 0.5 | < 0.5 | Hidden | `get_neighborhood` |
| > 0.5 | > 0.5 | Unknown | `epistemic_action` |

**Agent Response Pattern:**
```
1. Call any MCP tool
2. Check _cognitive_pulse in response
3. If suggested_action != current plan:
   - Consider triggering the suggested action
   - High entropy (>0.7) for 5+ min → trigger_dream
   - Low coherence (<0.4) → get_neighborhood or epistemic_action
4. Continue with task
```

**Performance:** Pulse computation targets < 1ms latency. If computation fails, the tool call fails fast (no fallbacks).

---

## The 13 Embedders

The teleological system uses 13 specialized embedders (E1-E13) as defined in the PRD. Each embedder captures a different semantic dimension, and together they form the "teleological fingerprint" for each memory.

| ID | Name | Model | Dimension | Purpose (V_goal) | Quantization |
|----|------|-------|-----------|------------------|--------------|
| E1 | Semantic | e5-large-v2 | 1024D (Matryoshka) | V_meaning | PQ-8 |
| E2 | Temporal-Recent | Exponential decay | 512D | V_freshness | Float8 |
| E3 | Temporal-Periodic | Fourier | 512D | V_periodicity | Float8 |
| E4 | Temporal-Positional | Sinusoidal PE | 512D | V_ordering | Float8 |
| E5 | Causal | Longformer (SCM) | 768D | V_causality | PQ-8 |
| E6 | Sparse | SPLADE | ~30K (5% active) | V_selectivity | Sparse |
| E7 | Code | Qodo-Embed (AST) | 1536D | V_correctness | PQ-8 |
| E8 | Graph | MiniLM (GNN) | 384D | V_connectivity | Float8 |
| E9 | HDC | Hyperdimensional | 1024D | V_robustness | Binary |
| E10 | Multimodal | CLIP | 768D | V_multimodality | PQ-8 |
| E11 | Entity | MiniLM (TransE) | 384D | V_factuality | Float8 |
| E12 | Late-Interaction | ColBERT | 128D/token | V_precision | Token pruning |
| E13 | SPLADE v3 | SPLADE | ~30K sparse | V_keyword_precision | Sparse |

**Kuramoto Natural Frequencies (Hz):**
| E1 | E2-E4 | E5 | E6 | E7 | E8 | E9 | E10 | E11 | E12 | E13 |
|----|-------|----|----|----|----|----|----|-----|-----|-----|
| 40 | 8 | 25 | 4 | 25 | 12 | 80 | 40 | 15 | 60 | 4 |

**Functional Groups:**
- **Semantic Core** (E1): Primary meaning representation
- **Temporal** (E2, E3, E4): Time-aware embeddings (recent, periodic, positional)
- **Causal** (E5): Cause-effect relationships
- **Sparse/Keyword** (E6, E13): Sparse activations and keyword precision
- **Code** (E7): Source code and AST understanding
- **Relational** (E8, E11): Graph structure and entity relationships
- **Holographic** (E9): Robust distributed representations
- **Cross-Modal** (E10): Visual-text alignment
- **Fine-Grained** (E12): Token-level precision via late interaction

---

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| -32700 | Parse Error | Invalid JSON |
| -32600 | Invalid Request | Invalid request structure |
| -32601 | Method Not Found | Unknown method |
| -32602 | Invalid Params | Invalid parameters |
| -32603 | Internal Error | Server error |
| -32110 | Bootstrap Error | North Star bootstrap failed |
| -32111 | Drift Detector Error | Drift detection failed |
| -32112 | Drift Corrector Error | Drift correction failed |
| -32113 | Pruning Error | Pruning operation failed |
| -32114 | Consolidation Error | Consolidation failed |
| -32115 | Subgoal Discovery Error | Sub-goal discovery failed |
| -32116 | Status Aggregation Error | Status aggregation failed |

---

## Claude Code Hook Integration

The context-graph MCP server is designed for seamless integration with Claude Code hooks for **autonomous context injection**. This enables the system to feed relevant context from the 13-embedding teleological arrays into Claude Code as it works.

### MCP Server Configuration

Add the context-graph MCP server to Claude Code:

```bash
# Add to Claude Code's MCP configuration
claude mcp add context-graph -- cargo run --manifest-path /path/to/contextgraph/crates/context-graph-mcp/Cargo.toml

# Or using a pre-built binary
claude mcp add context-graph -- /path/to/context-graph-mcp
```

### Hook Integration Patterns

#### Pre-Task Hook (Inject Relevant Context)

```yaml
# .claude/hooks/pre-task.yaml
hooks:
  pre-task:
    - name: context-inject
      command: |
        # Get current cognitive state
        STATE=$(mcp call context-graph get_memetic_status)

        # Search for relevant context based on task
        CONTEXT=$(mcp call context-graph search_graph '{"query": "$TASK_DESCRIPTION", "topK": 5}')

        # Inject into working memory
        echo "$CONTEXT"
```

#### Post-Edit Hook (Store Code Changes)

```yaml
# .claude/hooks/post-edit.yaml
hooks:
  post-edit:
    - name: context-store
      command: |
        mcp call context-graph inject_context '{
          "content": "'"$FILE_CHANGES"'",
          "rationale": "Code changes from edit to '"$FILE_PATH"'",
          "modality": "code",
          "importance": 0.7
        }'
```

#### Session End Hook (Dream Consolidation)

```yaml
# .claude/hooks/session-end.yaml
hooks:
  session-end:
    - name: context-consolidate
      command: |
        # Check if entropy is high enough for dreaming
        STATUS=$(mcp call context-graph get_memetic_status)
        ENTROPY=$(echo "$STATUS" | jq -r '.utl.entropy')

        if (( $(echo "$ENTROPY > 0.7" | bc -l) )); then
          mcp call context-graph trigger_dream '{"force": false}'
        fi
```

### Autonomous Operation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS CONTEXT FLOW                           │
├─────────────────────────────────────────────────────────────────────┤
│  1. Task Start                                                       │
│     └─ pre-task hook → get_memetic_status → check suggested_action  │
│     └─ search_graph → inject relevant 13-embedding fingerprints     │
│                                                                     │
│  2. During Work                                                     │
│     └─ Every tool response includes _cognitive_pulse                │
│     └─ Monitor entropy/coherence for drift                          │
│     └─ post-edit hook → inject_context with code changes            │
│                                                                     │
│  3. Periodic Check                                                  │
│     └─ get_consciousness_state → check C(t) level                   │
│     └─ get_alignment_drift → detect goal misalignment               │
│     └─ If drift detected → trigger_drift_correction                 │
│                                                                     │
│  4. Session End                                                     │
│     └─ session-end hook → trigger_dream if entropy > 0.7            │
│     └─ trigger_consolidation for memory hygiene                     │
│     └─ discover_sub_goals from session learnings                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Quick Start for Autonomous Mode

> **Note:** Manual North Star creation has been removed. The autonomous system discovers purpose
> from stored teleological fingerprints. See "North Star Goal Tools (DEPRECATED - REMOVED)" above.

1. **Store Initial Memories** (builds teleological fingerprint corpus):
```json
{
  "name": "store_memory",
  "arguments": {
    "content": "Project overview: A Rust MCP server implementing bio-nervous context graph with GWT consciousness...",
    "rationale": "Initial project understanding for autonomous operation",
    "importance": 0.8
  }
}
```

2. **Bootstrap Autonomous System** (discovers purpose from stored fingerprints):
```json
{
  "name": "auto_bootstrap_north_star",
  "arguments": {
    "confidence_threshold": 0.7,
    "max_candidates": 10
  }
}
```
The bootstrap analyzes stored teleological fingerprints to discover emergent purpose patterns
and initialize drift detection, pruning, consolidation, and sub-goal discovery services.

3. **Continue Injecting Context**:
```json
{
  "name": "inject_context",
  "arguments": {
    "content": "Additional context about the project...",
    "rationale": "Expanding autonomous understanding"
  }
}
```

The system will then autonomously:
- Compute 13-embedding teleological fingerprints for all content
- Track Johari quadrants per embedder (Open/Hidden/Blind/Unknown)
- Synchronize Kuramoto oscillators toward coherence (r → 0.8+)
- Suggest curation actions via Cognitive Pulse in every response
- Detect and correct alignment drift from North Star
- Consolidate memories during dream cycles when idle

### Performance Targets

| Operation | Target Latency |
|-----------|----------------|
| Single Embed (all 13) | < 35ms |
| inject_context P95 | < 40ms |
| search_graph P95 | < 30ms |
| Any tool P99 | < 60ms |
| Cognitive Pulse | < 1ms |
| Dream wake | < 100ms |
