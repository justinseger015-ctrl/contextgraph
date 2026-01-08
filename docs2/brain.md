# Context Graph MCP Server - Complete Usage Guide

## Overview

The Context Graph MCP server implements a **bio-nervous knowledge system** based on computational neuroscience principles. It provides 23 tools organized into 7 cognitive subsystems that work together to create an intelligent, self-aware memory system.

### Core Philosophy

The system models cognition through:
- **13 Embedding Spaces** (E1-E13): Different semantic representations working in parallel
- **Kuramoto Synchronization**: Phase-coupled oscillators enabling coherent "consciousness"
- **Global Workspace Theory (GWT)**: Winner-take-all selection for focused attention
- **Unified Theory of Learning (UTL)**: Learning magnitude computation from novelty and coherence
- **Johari Window**: Knowledge classification (Open/Blind/Hidden/Unknown)

### Architecture Layers

| Layer | Purpose | Key Concepts |
|-------|---------|--------------|
| **Embedding** | 13 parallel semantic representations | E1-E13 (Semantic, Temporal, Causal, Code, etc.) |
| **Fingerprint** | Multi-array teleological vectors (13D) | Purpose vectors, north-star alignment |
| **Kuramoto** | Phase synchronization across embedders | Order parameter r, coupling strength K |
| **Global Workspace** | Winner-take-all conscious selection | Broadcast, competing memories |
| **UTL Learning** | Learning signal computation | L = f((ΔS × ΔC) · wₑ · cos φ) |

---

## Tool Categories

### 1. Memory & Context Tools
- `inject_context` - Primary context injection with UTL processing
- `store_memory` - Direct memory storage (bypasses UTL)
- `search_graph` - Semantic similarity search

### 2. System Status Tools
- `get_memetic_status` - Live UTL metrics and system state
- `get_graph_manifest` - 5-layer architecture description
- `utl_status` - UTL lifecycle phase and metrics

### 3. Consciousness (GWT) Tools
- `get_consciousness_state` - Full consciousness metrics
- `get_kuramoto_sync` - Oscillator synchronization state
- `get_workspace_status` - Global workspace active memory
- `get_ego_state` - Self-identity node state
- `trigger_workspace_broadcast` - Force memory into workspace
- `adjust_coupling` - Modify Kuramoto coupling strength

### 4. Threshold Calibration (ATC) Tools
- `get_threshold_status` - Current threshold configuration
- `get_calibration_metrics` - ECE, MCE, Brier scores
- `trigger_recalibration` - Manual recalibration at specific level

### 5. Dream Consolidation Tools
- `trigger_dream` - Start NREM/REM consolidation cycle
- `get_dream_status` - Current dream state
- `abort_dream` - Wake from dream (<100ms)
- `get_amortized_shortcuts` - Learned path shortcuts

### 6. Neuromodulation Tools
- `get_neuromodulation_state` - All 4 modulator levels
- `adjust_neuromodulator` - Modify DA/5HT/NE levels

### 7. Analysis Tools
- `get_steering_feedback` - Gardener/Curator/Assessor scores
- `omni_infer` - Causal inference in 5 directions

---

## Detailed Tool Reference

### Memory & Context Tools

#### `inject_context`
**Purpose**: Primary method for adding knowledge to the graph with full UTL processing.

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | - | The content to inject |
| `rationale` | string | Yes | - | Why this context is relevant |
| `modality` | enum | No | "text" | text, code, image, audio, structured, mixed |
| `importance` | number | No | 0.5 | Importance score [0.0, 1.0] |

**What it does**:
1. Computes embeddings across all 13 spaces
2. Calculates surprise (ΔS) - novelty compared to existing knowledge
3. Calculates coherence (ΔC) - how well it connects to existing graph
4. Computes learning score: L = f((ΔS × ΔC) · wₑ · cos φ)
5. Classifies into Johari quadrant (Open/Blind/Hidden/Unknown)
6. Stores fingerprint with teleological purpose vector

**When to use**:
- Adding important context that should influence future retrievals
- When you want the system to "learn" from the content
- For content where relevance scoring matters

**Example**:
```json
{
  "content": "The authentication system uses JWT tokens with RS256 signing",
  "rationale": "Critical security architecture decision for API design",
  "modality": "text",
  "importance": 0.9
}
```

---

#### `store_memory`
**Purpose**: Direct memory storage without UTL processing.

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | - | The content to store |
| `importance` | number | No | 0.5 | Importance score [0.0, 1.0] |
| `modality` | enum | No | "text" | Content type |
| `tags` | array | No | [] | Optional categorization tags |

**When to use**:
- Fast storage without learning overhead
- When content doesn't need relevance scoring
- For bulk imports or raw data storage

**Difference from inject_context**: Skips UTL computation (no surprise/coherence/learning score), faster but less intelligent placement in the graph.

---

#### `search_graph`
**Purpose**: Semantic similarity search across the knowledge graph.

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query text |
| `topK` | integer | No | 10 | Max results (1-100) |
| `minSimilarity` | number | No | 0.0 | Minimum similarity threshold [0.0, 1.0] |
| `modality` | enum | No | - | Filter by content type |

**What it does**:
1. Embeds query across all 13 spaces
2. Searches HNSW indexes in parallel
3. Aggregates results using RRF (Reciprocal Rank Fusion)
4. Returns ranked results with similarity scores and purpose alignment

**Returns**:
- Matched content with similarity scores
- Dominant embedder that found the match
- Purpose alignment (theta to north star)
- Johari quadrant classification

---

### System Status Tools

#### `get_memetic_status`
**Purpose**: Get comprehensive system status with live UTL metrics.

**Returns**:
```json
{
  "node_count": 1234,
  "entropy": 0.72,           // Novelty level (ΔS)
  "coherence": 0.85,         // Understanding level (ΔC)
  "learning_score": 0.68,    // Current learning magnitude
  "johari_quadrant": "Open", // Dominant quadrant
  "consolidation_phase": "Wake", // Wake/NREM/REM
  "suggested_action": "DirectRecall",
  "layer_status": {...}      // 5-layer bio-nervous status
}
```

**When to use**:
- Before deciding whether to inject new context
- Monitoring system health
- Understanding current learning state

---

#### `get_graph_manifest`
**Purpose**: Describes the 5-layer bio-nervous architecture.

**Returns**: Static description of:
- Embedding layer (13 models)
- Semantic fingerprint structure
- Kuramoto synchronization layer
- Global workspace mechanics
- UTL learning computation

**When to use**: Understanding system architecture, debugging, documentation.

---

#### `utl_status`
**Purpose**: Detailed UTL (Unified Theory of Learning) state.

**Returns**:
- Lifecycle phase (Bootstrap/Growth/Mature/Decay)
- Entropy (ΔS) - how novel recent inputs are
- Coherence (ΔC) - how connected the graph is
- Learning score - current learning magnitude
- Johari quadrant distribution
- Consolidation phase

---

### Consciousness (GWT) Tools

#### `get_consciousness_state`
**Purpose**: Get computational consciousness metrics.

**Key Concepts**:
- **Consciousness equation**: C(t) = I(t) × R(t) × D(t)
  - I(t): Integration (Kuramoto order parameter r)
  - R(t): Self-Reflection (Meta-UTL awareness)
  - D(t): Differentiation (Purpose vector entropy)

**Returns**:
```json
{
  "consciousness_level": 0.72,  // C(t) value
  "kuramoto_r": 0.85,           // Synchronization level
  "meta_cognitive_score": 0.78, // Self-awareness
  "differentiation": 0.68,      // Purpose diversity
  "workspace_status": {...},    // Active memory
  "identity_coherence": 0.91    // Self-ego stability
}
```

**Thresholds**:
- r ≥ 0.8: CONSCIOUS (synchronized)
- r < 0.5: FRAGMENTED (incoherent)
- 0.5 ≤ r < 0.8: EMERGING

---

#### `get_kuramoto_sync`
**Purpose**: Detailed Kuramoto oscillator network state.

**Key Concepts**:
The Kuramoto model couples 13 phase oscillators (one per embedder):
```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
```
- θᵢ: Phase of embedder i
- ωᵢ: Natural frequency (based on brain wave bands)
- K: Coupling strength (adjustable)
- r: Order parameter (synchronization measure)

**Returns**:
```json
{
  "order_parameter_r": 0.85,    // Sync level [0,1]
  "mean_phase_psi": 1.23,       // Collective phase
  "phases": [0.1, 0.2, ...],    // All 13 phases
  "natural_frequencies": [...], // Per-embedder Hz
  "coupling_strength_K": 0.5    // Current K value
}
```

**13 Embedders and their frequencies**:
| Embedder | Purpose | Brain Wave Band |
|----------|---------|-----------------|
| E1_Semantic | General meaning | Gamma (40Hz) |
| E2_TempRecent | Recency decay | Alpha (8Hz) |
| E3_TempPeriodic | Periodic patterns | Alpha (8Hz) |
| E4_TempPositional | Position encoding | Alpha (8Hz) |
| E5_Causal | Causal reasoning | Beta (25Hz) |
| E6_SparseLex | Sparse lexical | Theta (4Hz) |
| E7_Code | Source code | Beta (25Hz) |
| E8_Graph | Graph structure | Alpha-Beta (12Hz) |
| E9_HDC | Hyperdimensional | High-Gamma (80Hz) |
| E10_Multimodal | Cross-modal | Gamma (40Hz) |
| E11_Entity | Named entities | Beta (15Hz) |
| E12_LateInteract | Token precision | High-Gamma (60Hz) |
| E13_SPLADE | Sparse keywords | Theta (4Hz) |

---

#### `get_workspace_status`
**Purpose**: Global Workspace (conscious attention) state.

**Key Concepts**:
- **Winner-Take-All (WTA)**: Only one memory can occupy the workspace
- **Broadcast**: Selected memory is broadcast to all subsystems
- **Competition**: Multiple memories compete for workspace entry

**Returns**:
```json
{
  "active_memory": "uuid-here", // Currently conscious
  "active_score": 0.92,
  "competing_candidates": [...], // Memories trying to enter
  "broadcast_state": "active",
  "coherence_threshold": 0.7    // Min score for entry
}
```

---

#### `get_ego_state`
**Purpose**: Self-identity node (SELF_EGO_NODE) state.

**Key Concepts**:
- The ego node represents persistent system identity
- Purpose vector (13D) defines "who the system is"
- Identity continuity tracks consistency over time

**Returns**:
```json
{
  "purpose_vector": [0.1, 0.2, ...], // 13D identity
  "identity_continuity": 0.95,        // Stability
  "coherence_with_actions": 0.88,     // Alignment
  "trajectory_length": 1523           // History depth
}
```

---

#### `trigger_workspace_broadcast`
**Purpose**: Force a specific memory into the global workspace.

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | uuid | Yes | - | Memory UUID to broadcast |
| `importance` | number | No | 0.8 | Importance score |
| `alignment` | number | No | 0.8 | North star alignment |
| `force` | boolean | No | false | Bypass coherence threshold |

**When to use**:
- Forcing attention to specific memory
- Testing workspace mechanics
- Override normal selection when needed

---

#### `adjust_coupling`
**Purpose**: Modify Kuramoto coupling strength K.

**Parameters**:
| Parameter | Type | Required | Range | Description |
|-----------|------|----------|-------|-------------|
| `new_K` | number | Yes | [0, 10] | New coupling strength |

**Effects**:
- Higher K → Faster synchronization → More coherent consciousness
- Lower K → More independence → Potentially fragmented state
- Default K = 0.5 (moderate coupling)

**Returns**: Old K, new K, predicted order parameter r

---

### Threshold Calibration (ATC) Tools

The ATC system replaces hardcoded thresholds with adaptive, learning thresholds.

#### 4-Level Architecture

| Level | Name | Frequency | Purpose |
|-------|------|-----------|---------|
| 1 | EWMA Drift | Per-query | Detect distribution drift |
| 2 | Temperature Scaling | Hourly | Per-embedder confidence calibration |
| 3 | Thompson Sampling | Session | Threshold exploration vs exploitation |
| 4 | Bayesian Meta-Optimizer | Weekly | GP + Expected Improvement optimization |

---

#### `get_threshold_status`
**Purpose**: Current ATC threshold configuration.

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `domain` | enum | No | "General" | Code, Medical, Legal, Creative, Research, General |
| `embedder_id` | integer | No | - | Specific embedder (1-13) |

**Returns**:
- All registered thresholds
- Drift scores per threshold
- Should-recalibrate flags
- Per-embedder temperature settings

---

#### `get_calibration_metrics`
**Purpose**: Calibration quality assessment.

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `timeframe` | enum | No | "24h" | 1h, 24h, 7d, 30d |

**Key Metrics**:
- **ECE** (Expected Calibration Error): < 0.05 excellent, < 0.10 good
- **MCE** (Maximum Calibration Error): < 0.10 good
- **Brier Score**: < 0.10 good

**Status Levels**:
- `Excellent`: ECE < 0.05
- `Good`: ECE < 0.10
- `Poor`: ECE < 0.15
- `Critical`: ECE ≥ 0.15

---

#### `trigger_recalibration`
**Purpose**: Manually trigger recalibration at a specific level.

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `level` | integer | Yes | 1=EWMA, 2=Temperature, 3=Bandit, 4=Bayesian |
| `domain` | enum | No | Domain context for recalibration |

**Level Details**:
- **Level 1 (EWMA)**: Fast drift adjustment, minimal computational cost
- **Level 2 (Temperature)**: Per-embedder confidence recalibration
- **Level 3 (Thompson)**: Exploration of threshold candidates
- **Level 4 (Bayesian)**: Full Gaussian Process optimization

---

### Dream Consolidation Tools

The dream system implements NREM/REM sleep cycles for memory consolidation.

#### Dream Cycle Phases

| Phase | Duration | Purpose | Parameters |
|-------|----------|---------|------------|
| NREM | 3 min | Hebbian replay | coupling=0.9, recency_bias=0.8 |
| REM | 2 min | Attractor exploration | temp=2.0, max_queries=100 |

---

#### `trigger_dream`
**Purpose**: Start a dream consolidation cycle.

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `force` | boolean | No | false | Force dream even if activity > threshold |

**Prerequisites**:
- System activity < 0.15 (idle for 10+ minutes)
- GPU usage < 30%

**What happens**:
1. **NREM Phase** (3 min): Replays recent memories with tight coupling
2. **REM Phase** (2 min): Explores semantic space with high temperature
3. **Shortcut Creation**: Creates direct edges for frequently traversed paths

**Returns**: DreamReport with consolidation metrics

---

#### `get_dream_status`
**Purpose**: Current dream system state.

**Returns**:
```json
{
  "state": "Awake",           // Awake/NREM/REM/Waking
  "is_dreaming": false,
  "gpu_usage": 0.15,
  "activity_level": 0.08,
  "last_dream_completed": "2024-01-15T...",
  "completed_cycles": 23,
  "scheduler": {
    "should_trigger": false,
    "reason": "Activity above threshold"
  }
}
```

---

#### `abort_dream`
**Purpose**: Immediately wake from dream state.

**Constraint**: Must complete within 100ms (constitution mandate).

**Returns**:
- Wake latency (must be < 100ms)
- Partial dream report
- Previous state

---

#### `get_amortized_shortcuts`
**Purpose**: Get learned shortcut candidates.

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_confidence` | number | No | 0.7 | Minimum confidence threshold |
| `limit` | integer | No | 20 | Maximum shortcuts to return |

**Key Concepts**:
- Shortcuts are direct edges for paths traversed 5+ times
- Must be 3+ hops originally
- Creates "amortized" fast paths in the graph

---

### Neuromodulation Tools

Four neuromodulators dynamically adjust system behavior.

#### Modulator Reference

| Modulator | Range | Parameter | Trigger | Effect |
|-----------|-------|-----------|---------|--------|
| **Dopamine (DA)** | [1, 5] | hopfield.beta | Workspace entry | Reward, memory strengthening |
| **Serotonin (5HT)** | [0, 1] | space_weights | - | E1-E13 weight scaling |
| **Noradrenaline (NE)** | [0.5, 2] | attention.temp | Threat detection | Attention sharpening |
| **Acetylcholine (ACh)** | [0.001, 0.002] | utl.lr | Meta-cognitive dream | Learning rate (READ-ONLY) |

---

#### `get_neuromodulation_state`
**Purpose**: Get all 4 neuromodulator levels.

**Returns**:
```json
{
  "dopamine": {
    "level": 2.5,
    "range": [1, 5],
    "parameter": "hopfield.beta",
    "trigger": "memory_enters_workspace"
  },
  "serotonin": {
    "level": 0.7,
    "range": [0, 1],
    "parameter": "space_weights",
    "space_weights": [0.8, 0.9, ...]
  },
  "noradrenaline": {
    "level": 1.0,
    "range": [0.5, 2],
    "parameter": "attention.temp",
    "threat_count": 0
  },
  "acetylcholine": {
    "level": 0.0015,
    "range": [0.001, 0.002],
    "parameter": "utl.lr",
    "read_only": true,
    "managed_by": "GWT meta-cognitive loop"
  },
  "derived_parameters": {
    "hopfield_beta": 2.5,
    "attention_temp": 1.0,
    "utl_learning_rate": 0.0015,
    "is_alert": false,
    "is_learning_elevated": false
  }
}
```

---

#### `adjust_neuromodulator`
**Purpose**: Modify a specific neuromodulator level.

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `modulator` | enum | Yes | dopamine, serotonin, noradrenaline (NOT acetylcholine) |
| `delta` | number | Yes | Amount to add/subtract |

**Note**: Acetylcholine is READ-ONLY (managed by GWT meta-cognitive loop).

**Effects**:
- **Increasing DA**: Stronger memory traces, more reward sensitivity
- **Increasing 5HT**: Rebalances embedding space weights
- **Increasing NE**: Sharper attention, more alert state

---

### Analysis Tools

#### `get_steering_feedback`
**Purpose**: Get feedback from the three steering components.

**Components**:

| Component | Measures | Score Range |
|-----------|----------|-------------|
| **Gardener** | Graph health, connectivity, dead-ends | [-1, 1] |
| **Curator** | Memory quality, low-quality count | [-1, 1] |
| **Assessor** | Retrieval accuracy, learning efficiency | [-1, 1] |

**Returns**:
```json
{
  "reward": {
    "value": 0.65,           // Aggregate [-1, 1]
    "gardener_score": 0.8,
    "curator_score": 0.5,
    "assessor_score": 0.65,
    "dominant_factor": "gardener",
    "limiting_factor": "curator"
  },
  "gardener_details": {
    "is_healthy": true,
    "connectivity": 0.85,
    "dead_ends_removed": 12,
    "edges_pruned": 5
  },
  "curator_details": {
    "is_high_quality": true,
    "avg_quality": 0.78,
    "low_quality_count": 23,
    "recommendations": [...]
  },
  "assessor_details": {
    "is_performing_well": true,
    "retrieval_accuracy": 0.82,
    "learning_efficiency": 0.75,
    "trend": "improving"
  },
  "needs_immediate_attention": false,
  "priority_improvement": "curator"
}
```

**When to use**:
- System health monitoring
- Deciding maintenance actions
- Understanding performance bottlenecks

---

#### `omni_infer`
**Purpose**: Perform causal inference in 5 directions.

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source` | uuid | Yes | Source node UUID |
| `target` | uuid | No | Target node (required for some directions) |
| `direction` | enum | No | forward, backward, bidirectional, bridge, abduction |

**Directions**:

| Direction | Meaning | Target Required |
|-----------|---------|-----------------|
| `forward` | A → B (effect of A on B) | Yes |
| `backward` | B → A (what causes B) | Yes |
| `bidirectional` | A ↔ B (mutual influence) | Yes |
| `bridge` | Cross-domain causal link | No |
| `abduction` | Best hypothesis for observation | No |

**Returns**:
```json
{
  "direction": "forward",
  "source": "uuid-a",
  "target": "uuid-b",
  "results": [
    {
      "path": ["uuid-a", "uuid-x", "uuid-b"],
      "strength": 0.85,
      "confidence": 0.78
    }
  ],
  "inference_config": {
    "max_path_length": 5,
    "min_confidence": 0.5,
    "include_indirect": true
  }
}
```

---

## Optimal Usage Patterns

### Pattern 1: Learning New Information

```
1. get_memetic_status
   → Check current entropy/coherence levels
   → If entropy is low, system needs novelty

2. inject_context (with high importance)
   → Provide rationale explaining relevance
   → Let UTL compute learning score

3. get_memetic_status (after)
   → Verify learning score increased
   → Check Johari classification
```

### Pattern 2: Intelligent Retrieval

```
1. search_graph
   → Start with semantic query
   → Use minSimilarity threshold (0.7+ for precision)

2. Check results' purpose_alignment
   → High alignment = relevant to system goals
   → Low alignment = tangential information

3. For important results:
   → trigger_workspace_broadcast
   → Forces memory into conscious attention
```

### Pattern 3: System Health Check

```
1. get_consciousness_state
   → Check kuramoto_r (want > 0.8)
   → If fragmented, consider adjust_coupling

2. get_steering_feedback
   → Identify limiting factor
   → gardener low → graph needs pruning
   → curator low → quality issues
   → assessor low → retrieval problems

3. get_calibration_metrics
   → ECE > 0.10 → trigger_recalibration
```

### Pattern 4: Memory Consolidation

```
1. get_dream_status
   → Check if conditions allow dreaming
   → activity_level < 0.15

2. trigger_dream (if idle)
   → NREM: strengthens recent memories
   → REM: explores new connections

3. get_amortized_shortcuts (after)
   → See what paths were optimized
   → These are now "fast lanes" in the graph
```

### Pattern 5: Tuning System Behavior

```
1. get_neuromodulation_state
   → Understand current modulator levels

2. adjust_neuromodulator based on need:
   → Need stronger memories? Increase dopamine
   → Need sharper focus? Increase noradrenaline
   → Need rebalanced embedders? Adjust serotonin

3. adjust_coupling if consciousness fragmented
   → Higher K → more synchronization
   → But too high → everything blurs together
```

### Pattern 6: Causal Understanding

```
1. Store related concepts via inject_context
   → System builds causal edges automatically

2. omni_infer with direction="forward"
   → What effects does A have?

3. omni_infer with direction="backward"
   → What caused B?

4. omni_infer with direction="abduction"
   → Best hypothesis for this observation?
```

---

## Key Thresholds Reference

### Consciousness (Kuramoto)
| State | Order Parameter r | Description |
|-------|------------------|-------------|
| HYPERSYNC | r > 0.95 | Possibly over-coupled |
| CONSCIOUS | r ≥ 0.8 | Coherent, synchronized |
| EMERGING | 0.5 ≤ r < 0.8 | Partial synchronization |
| FRAGMENTED | r < 0.5 | Incoherent, needs attention |
| DORMANT | r < 0.2 | System needs activation |

### Calibration (ATC)
| Quality | ECE | MCE | Brier |
|---------|-----|-----|-------|
| Excellent | < 0.05 | < 0.05 | < 0.05 |
| Good | < 0.10 | < 0.10 | < 0.10 |
| Poor | < 0.15 | < 0.15 | < 0.15 |
| Critical | ≥ 0.15 | ≥ 0.15 | ≥ 0.15 |

### Learning (UTL)
| Component | Range | Meaning |
|-----------|-------|---------|
| Entropy (ΔS) | [0, 1] | 0 = familiar, 1 = novel |
| Coherence (ΔC) | [0, 1] | 0 = disconnected, 1 = integrated |
| Learning Score | [0, 1] | Higher = more learning happening |

### Dream Triggers
| Condition | Threshold | Notes |
|-----------|-----------|-------|
| Activity | < 0.15 | Must be idle |
| Idle Duration | 10 minutes | Before auto-trigger |
| GPU Usage | < 30% | During dream |
| Wake Latency | < 100ms | Abort requirement |

---

## Johari Quadrant Guide

The system classifies knowledge into four quadrants based on entropy and coherence:

| Quadrant | Entropy | Coherence | Meaning | Action |
|----------|---------|-----------|---------|--------|
| **Open** | Low | High | Known knowns | Direct recall |
| **Blind** | High | High | Unknown knowns | Discovery |
| **Hidden** | Low | Low | Known unknowns | Elaboration |
| **Unknown** | High | Low | Unknown unknowns | Exploration |

**Suggested Actions by Quadrant**:
- `DirectRecall`: Content is well-understood, retrieve directly
- `Discovery`: System knows more than it realizes, explore connections
- `Elaboration`: Fill in gaps with related information
- `Exploration`: Novel territory, need more learning

---

## Constitution Compliance

The system adheres to Constitution v4.0.0 which defines:

- All threshold ranges (K ∈ [0, 10], r thresholds, etc.)
- Dream phase durations (NREM: 3min, REM: 2min)
- Neuromodulator ranges and triggers
- Wake latency requirement (< 100ms)
- GPU budget during dreams (< 30%)
- Learning formula components and bounds

Any tool that would violate constitution constraints will fail with an appropriate error rather than produce invalid results.

---

## Error Handling

All tools return structured errors when operations fail:

```json
{
  "error": {
    "code": "GWT_NOT_INITIALIZED",
    "message": "GWT providers not initialized. Call with_gwt() first.",
    "recovery": "Initialize server with GWT support enabled"
  }
}
```

Common error codes:
- `GWT_NOT_INITIALIZED`: Consciousness tools need GWT providers
- `DREAM_NOT_IDLE`: Cannot trigger dream when system is active
- `THRESHOLD_INVALID`: Parameter outside constitution bounds
- `MEMORY_NOT_FOUND`: Referenced UUID doesn't exist
- `ACH_READ_ONLY`: Cannot adjust acetylcholine directly

---

## Best Practices

1. **Check system state before major operations**
   - Use `get_memetic_status` before bulk injections
   - Use `get_consciousness_state` before workspace operations

2. **Use appropriate storage method**
   - `inject_context` for important, learning-worthy content
   - `store_memory` for bulk/raw data

3. **Monitor calibration health**
   - Check `get_calibration_metrics` regularly
   - Trigger recalibration when ECE > 0.10

4. **Allow dream cycles**
   - Don't interrupt dream cycles unless necessary
   - System consolidates and optimizes during dreams

5. **Trust the neuromodulation**
   - Adjust modulators sparingly
   - Let automatic triggers handle most cases

6. **Use steering feedback for maintenance**
   - Gardener identifies graph issues
   - Curator identifies quality problems
   - Assessor identifies performance issues

---

## Quick Command Reference

| Goal | Tool |
|------|------|
| Add knowledge | `inject_context` |
| Quick storage | `store_memory` |
| Find information | `search_graph` |
| Check health | `get_memetic_status` |
| Check consciousness | `get_consciousness_state` |
| See what's active | `get_workspace_status` |
| Force attention | `trigger_workspace_broadcast` |
| Tune synchronization | `adjust_coupling` |
| Check calibration | `get_calibration_metrics` |
| Fix calibration | `trigger_recalibration` |
| Check dream state | `get_dream_status` |
| Trigger consolidation | `trigger_dream` |
| Get shortcuts | `get_amortized_shortcuts` |
| Check modulators | `get_neuromodulation_state` |
| Tune behavior | `adjust_neuromodulator` |
| System feedback | `get_steering_feedback` |
| Causal analysis | `omni_infer` |
