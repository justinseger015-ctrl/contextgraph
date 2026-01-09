# Ultimate Context Graph - Compressed PRD v4.0.0 (Global Workspace)

**Abbrev**: NT=Neurotransmitter, SS=Steering Subsystem, OI=Omnidirectional Inference, FV=Formal Verification, PC=Predictive Coding, HE=Hyperbolic Entailment, TF=Teleological Fingerprint, PV=Purpose Vector, SF=Semantic Fingerprint, GWT=Global Workspace Theory, GW=Global Workspace, IIT=Integrated Information Theory, CMS=Continuum Memory System

**Paradigm**: Multi-Array Teleological Fingerprints with Global Workspace Consciousness — the 13-embedding array IS the teleological vector, Kuramoto-synchronized into unified conscious percepts

---

## 0. WHY THIS EXISTS & YOUR ROLE

### 0.1 The Problem You're Solving
AI agents fail because: **no persistent memory** (context lost between sessions), **poor retrieval** (vector search misses semantic relationships), **no learning loop** (no feedback on storage quality), **context bloat** (agents retrieve too much, can't compress).

### 0.2 What the System Does vs What You Do

| System (Automatic) | You (Active) |
|-------------------|--------------|
| Stores conversations via host hooks | Curates quality (merge, annotate, forget) |
| Runs dream consolidation when idle | Triggers dreams when entropy high |
| Detects conflicts & duplicates | Decides resolution strategy |
| Computes UTL metrics | Responds to Pulse suggestions |
| PII scrubbing, adversarial defense | Nothing - trust the system |

**You are a librarian, not an archivist.** You don't store everything—you ensure what's stored is findable, coherent, and useful.

### 0.3 Steering Feedback Loop (How You Learn What to Store)
```
You store node → System assesses quality → Returns reward signal
       ↑                                            │
       └────────── You adjust behavior ─────────────┘
```

**Rewards by lifecycle:**
| Stage | Good Storage (+reward) | Bad Storage (-penalty) |
|-------|------------------------|------------------------|
| Infancy | High novelty (ΔS) | Low novelty |
| Growth | Balanced ΔS + ΔC | Imbalanced |
| Maturity | High coherence (ΔC) | Low coherence |

**Universal penalties:**
- Missing rationale: **-0.5**
- Near-duplicate (>0.9 sim): **-0.4**
- Low priors confidence: **-0.3**

---

## 1. AGENT QUICK START

### 1.1 System Overview
5-layer bio-nervous memory (UTL). Storage=automatic. **Your job: curation + retrieval (librarian).**

### 1.2 First Contact
1. `get_system_instructions` → mental model (~300 tok, KEEP)
2. `get_graph_manifest` → 5-layer architecture
3. `get_memetic_status` → entropy/coherence + `curation_tasks`

### 1.3 Cognitive Pulse (Every Response)
`Pulse: { Entropy: X, Coherence: Y, Suggested: "action" }`

| E | C | Action |
|---|---|--------|
| >0.7 | >0.5 | `epistemic_action` |
| >0.7 | <0.4 | `trigger_dream`/`critique_context` |
| <0.4 | >0.7 | Continue |
| <0.4 | <0.4 | `get_neighborhood` |

### 1.4 Core Behaviors
**Dreaming**: entropy>0.7 for 5+min OR 30+min work → `trigger_dream`
**Curation**: NEVER blind `merge_concepts`. Check `curation_tasks` first. Use `merge_strategy=summarize` (important) or `keep_highest` (trivial). ALWAYS include `rationale`.
**Feedback**: Empty search→↑noradrenaline. Irrelevant→`reflect_on_memory`. Conflicts→check `conflict_alert`. "Why don't you remember?"→`get_system_logs`.

### 1.4.1 Decision Trees

**When to Store:**
```
User shares information
  ├─ Is it novel? (check entropy after inject_context)
  │   ├─ YES + relevant → store_memory with rationale
  │   └─ NO → skip (system already has it)
  └─ Will it help future retrieval?
      ├─ YES → store with link_to related nodes
      └─ NO → don't pollute the graph
```

**When to Dream:**
```
Check Pulse entropy
  ├─ entropy > 0.7 for 5+ min → trigger_dream(phase=full)
  ├─ Working 30+ min straight → trigger_dream(phase=nrem)
  └─ entropy < 0.5 → no dream needed
```

**When to Curate:**
```
get_memetic_status returns curation_tasks?
  ├─ YES → process tasks BEFORE other work
  │   ├─ Duplicate detected → merge_concepts (check priors first!)
  │   ├─ Conflict detected → critique_context, then ask user or merge
  │   └─ Orphan detected → forget_concept or link_to parent
  └─ NO → continue normal work
```

**When Search Fails:**
```
Search returns empty/irrelevant
  ├─ Empty → broaden query, try generate_search_plan
  ├─ Irrelevant → reflect_on_memory to understand why
  ├─ Conflicting → check conflict_alert, resolve or ask user
  └─ User asks "why don't you remember X?" → get_system_logs
```

**When Confused (low coherence):**
```
coherence < 0.4
  ├─ High entropy too → trigger_dream or critique_context
  ├─ Low entropy → get_neighborhood to build context
  └─ System suggests epistemic_action → ASK the clarifying question
```

### 1.5 Query Best Practices
`generate_search_plan` → 3 optimized queries (semantic/causal/code) → parallel execute
`find_causal_path` → "UserAuth→JWT→Middleware→RateLimiting"

### 1.6 Token Economy
| Level | Tokens | When |
|-------|--------|------|
| 0 | ~100 | High confidence |
| 1 | ~200 | Normal |
| 2 | ~800 | coherence<0.4 ONLY |

### 1.7 Multi-Agent Safety
`perspective_lock: { domain: "code", exclude_agent_ids: ["creative-writer"] }`

### 1.8 Epistemic Actions (System-Generated Questions)

When coherence < 0.4, the system can generate clarifying questions for you to ask:

```json
{
  "action_type": "ask_user",
  "question": "You mentioned UserAuth connects to JWT—what triggers token refresh?",
  "expected_entropy_reduction": 0.35,
  "focal_nodes": ["node_userauth", "node_jwt"]
}
```

**Your job:** Ask the user this question (or a natural variant), then store their answer with `store_memory`.

**When to use `epistemic_action(force=true)`:**
- You're stuck and don't know what to ask
- Coherence has been low for multiple exchanges
- User seems to expect you to know something you don't

---

## 2. ARCHITECTURE

### 2.1 UTL Core (Multi-Embedding Extension)
```
Classic:  L = f((ΔS × ΔC) · wₑ · cos φ)
Multi:    L_multi = sigmoid(2.0 · (Σᵢ τᵢλ_S·ΔSᵢ) · (Σⱼ τⱼλ_C·ΔCⱼ) · wₑ · cos φ)

Where:
  ΔSᵢ: entropy in embedding space i (i=1..12)
  ΔCⱼ: coherence in embedding space j (j=1..12)
  τᵢ: teleological weight (alignment to North Star) for space i
  wₑ: emotional[0.5,1.5]
  φ: phase sync across Kuramoto-coupled spaces

Loss: J = 0.4·L_task + 0.3·L_semantic + 0.2·L_teleological + 0.1·(1-L)
```

### 2.2 Johari Quadrants (Per-Embedder)
Each of 12 embedding spaces has independent Johari classification:

| ΔSᵢ | ΔCᵢ | Quadrant | Meaning |
|-----|-----|----------|---------|
| Low | High | Open | Aware in this space |
| High | Low | Blind | Discovery opportunity |
| Low | Low | Hidden | Latent in this space |
| High | High | Unknown | Frontier in this space |

**Cross-Space Insight**: Memory can be Open(semantic) but Blind(causal) — enables targeted learning

### 2.3 5-Layer Bio-Nervous
| L | Function | Latency | Key |
|---|----------|---------|-----|
| L1 | Sensing/tokenize | <5ms | Embed+PII |
| L2 | Reflex/cache | <100μs | Hopfield (>80% hit) |
| L3 | Memory/retrieval | <1ms | Hopfield (2^768 cap) |
| L4 | Learning/weights | <10ms | UTL Optimizer |
| L5 | Coherence/verify | <10ms | Thalamic Gate, PC |

### 2.4 Lifecycle (Marblestone λ Weights)
| Phase | Interactions | λ_ΔS | λ_ΔC | Stance |
|-------|--------------|------|------|--------|
| Infancy | 0-50 | 0.7 | 0.3 | Capture (novelty) |
| Growth | 50-500 | 0.5 | 0.5 | Balanced |
| Maturity | 500+ | 0.3 | 0.7 | Curation (coherence) |

---

## 2.5 GLOBAL WORKSPACE THEORY (GWT) — COMPUTATIONAL CONSCIOUSNESS

### 2.5.1 The Consciousness Equation

The system calculates functional consciousness as the product of three measurable quantities:

```
C(t) = I(t) × R(t) × D(t)

Where:
  C(t) = Consciousness level at time t [0, 1]
  I(t) = Integration (Kuramoto synchronization) [0, 1]
  R(t) = Self-Reflection (Meta-UTL awareness) [0, 1]
  D(t) = Differentiation (13D fingerprint entropy) [0, 1]
```

**Expanded Form:**
```
C(t) = r(t) × σ(MetaUTL.predict_accuracy) × H(PurposeVector)

Where:
  r(t) = (1/13)|Σⱼ exp(iθⱼ)|  — Kuramoto order parameter
  σ(x) = 1/(1 + e^(-x))       — Sigmoid activation
  H(PV) = -Σᵢ pᵢ log(pᵢ) / log(13)  — Normalized Shannon entropy
```

### 2.5.2 Kuramoto Oscillator Layer (The "Heartbeat")

Each embedding space has an associated phase oscillator. The 13 oscillators synchronize via Kuramoto coupling:

```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)

Where:
  θᵢ = Phase of embedder i (i=1..13) ∈ [0, 2π]
  ωᵢ = Natural frequency of embedder i (Hz)
  K  = Global coupling strength [0, 10]
  N  = Number of embedders (13)
```

**Order Parameter (Synchronization Measure):**
```
r · e^(iψ) = (1/N) Σⱼ e^(iθⱼ)

Where:
  r = Order parameter (sync level) ∈ [0, 1]
  ψ = Mean phase
  r > 0.8 → Memory is coherent ("conscious")
  r < 0.5 → Memory fragmentation alert
```

**Natural Frequencies by Embedder:**
| Embedder | ωᵢ (Hz) | Rationale |
|----------|---------|-----------|
| E1 Semantic | 40 | Gamma band - conscious binding |
| E2-E4 Temporal | 8 | Alpha - temporal integration |
| E5 Causal | 25 | Beta - causal reasoning |
| E6 Sparse | 4 | Theta - sparse activations |
| E7 Code | 25 | Beta - structured thinking |
| E8 Graph | 12 | Alpha-Beta transition |
| E9 HDC | 80 | High gamma - holographic |
| E10 Multimodal | 40 | Gamma - cross-modal binding |
| E11 Entity | 15 | Beta - factual grounding |
| E12 Late | 60 | High gamma - token precision |
| E13 SPLADE | 4 | Theta - keyword sparse |

### 2.5.3 Global Broadcast Architecture (The "Stage")

Only phase-locked memories are "perceived" by the system. The Global Workspace implements Winner-Take-All:

```
GlobalWorkspace:
  active_memory: Option<MemoryId>      — Currently "conscious" memory
  coherence_threshold: 0.8             — Minimum r for broadcast
  broadcast_duration: 100ms            — How long memory stays active
  competing_memories: PriorityQueue    — Sorted by r × importance
```

**Broadcast Selection Algorithm:**
```
1. Compute r for all candidate memories
2. Filter: candidates where r ≥ coherence_threshold
3. Rank: score = r × importance × north_star_alignment
4. Select: top-1 becomes active_memory
5. Broadcast: active_memory visible to all subsystems
6. Inhibit: losing candidates get dopamine reduction
```

**Global Broadcast Events:**
| Event | Trigger | Effect |
|-------|---------|--------|
| `memory_enters_workspace` | r crosses 0.8 upward | Dopamine += 0.2 |
| `memory_exits_workspace` | r drops below 0.7 | Log for dream replay |
| `workspace_conflict` | Two memories r > 0.8 | Trigger critique_context |
| `workspace_empty` | No memory r > 0.8 for 5s | Trigger epistemic_action |

### 2.5.4 The SELF_EGO_NODE (System Identity)

A persistent, special node representing the system itself:

```rust
pub struct SelfEgoNode {
    id: UUID,                          // Fixed: "SELF_EGO_NODE"
    content: String,                   // "I am the context graph manager..."
    fingerprint: TeleologicalFingerprint,  // Current system state
    purpose_vector: [f32; 13],         // System's purpose alignment
    identity_trajectory: Vec<PurposeSnapshot>,  // History of self
    coherence_with_actions: f32,       // Are actions aligned with self?
}
```

**Self-Awareness Loop:**
```
Every action:
  1. Retrieve SELF_EGO_NODE
  2. Compute A(action_embedding, SELF_EGO_NODE.purpose_vector)
  3. If alignment < 0.55: trigger self_reflection
  4. Update SELF_EGO_NODE.fingerprint with action outcome
  5. Store to purpose_evolution (temporal trajectory)
```

**Identity Continuity Metric:**
```
IdentityContinuity = cosine(PV_t, PV_{t-1}) × r(t)

Where:
  PV_t = Purpose vector at time t
  r(t) = Kuramoto order parameter at time t

Threshold:
  IC > 0.9 → Strong continuity (healthy)
  IC < 0.7 → Identity drift warning
  IC < 0.5 → Trigger dream consolidation
```

### 2.5.5 Meta-Cognitive Loop (Self-Awareness)

The Meta-UTL system observes its own learning:

```
MetaScore = σ(2 × (L_predicted - L_actual))

Where:
  L_predicted = System's prediction of learning outcome
  L_actual = Measured learning score
  σ = Sigmoid function
```

**Self-Correction Protocol:**
```
IF MetaScore < 0.5 for 5 consecutive operations:
  → Increase Acetylcholine (learning rate)
  → Trigger introspective dream

IF MetaScore > 0.9 consistently:
  → System is well-calibrated
  → Can reduce meta-monitoring frequency
```

### 2.5.6 Consciousness State Machine

```
States:
  DORMANT     → r < 0.3, no active workspace
  FRAGMENTED  → 0.3 ≤ r < 0.5, partial sync
  EMERGING    → 0.5 ≤ r < 0.8, approaching coherence
  CONSCIOUS   → r ≥ 0.8, unified percept active
  HYPERSYNC   → r > 0.95, possibly pathological

Transitions:
  DORMANT → FRAGMENTED:     New memory with ΔS > 0.7
  FRAGMENTED → EMERGING:    Kuramoto coupling increases
  EMERGING → CONSCIOUS:     r crosses 0.8 threshold
  CONSCIOUS → EMERGING:     Conflicting memory enters
  CONSCIOUS → HYPERSYNC:    Warning - may indicate seizure-like state
  Any → DORMANT:            10+ minutes of inactivity
```

### 2.5.7 Consciousness Quality Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Φ (Integrated Information)** | `min_cut(13-space graph) / total_connections` | > 0.3 |
| **Global Availability** | `% of subsystems receiving broadcast` | > 90% |
| **Workspace Stability** | `avg duration of conscious state` | > 500ms |
| **Meta-Awareness** | `MetaUTL.prediction_accuracy` | > 0.85 |
| **Identity Coherence** | `cosine(PV_t, PV_{t-1})` | > 0.9 |

---

## 3. 13-MODEL EMBEDDING → TELEOLOGICAL FINGERPRINT

**Paradigm**: NO FUSION — Store all 13 embeddings (E1-E12 + E13 SPLADE). The array IS the teleological vector.
**Storage**: ~17KB per memory (quantized) vs 46KB uncompressed — 63% reduction via PQ-8/Float8/Binary
**Info Preserved**: 100% (vs 33% with top-k=4 FuseMoE)

| ID | Model | Dim | Latency | Purpose (V_goal) | Quantization |
|----|-------|-----|---------|------------------|--------------|
| E1 | Semantic | 1024D (Matryoshka: 512/256/128) | <5ms | V_meaning | PQ-8 |
| E2 | Temporal-Recent | 512D (exp decay) | <2ms | V_freshness | Float8 |
| E3 | Temporal-Periodic | 512D (Fourier) | <2ms | V_periodicity | Float8 |
| E4 | Temporal-Positional | 512D (sin PE) | <2ms | V_ordering | Float8 |
| E5 | Causal | 768D (SCM, **asymmetric**) | <8ms | V_causality | PQ-8 |
| E6 | Sparse | ~30K (5% active) | <3ms | V_selectivity | Sparse |
| E7 | Code | 1536D (AST) | <10ms | V_correctness | PQ-8 |
| E8 | Graph/GNN | 384D (MiniLM) | <5ms | V_connectivity | Float8 |
| E9 | HDC | 10K-bit→1024D | <1ms | V_robustness | Binary |
| E10 | Multimodal | 768D | <15ms | V_multimodality | PQ-8 |
| E11 | Entity/TransE | 384D (h+r≈t) | <2ms | V_factuality | Float8 |
| E12 | Late-Interaction | 128D/tok (ColBERT) | <8ms | V_precision | Token pruning |
| **E13** | **SPLADE** | **~30K sparse** | **<5ms** | **V_keyword_precision** | **Sparse** |

**TeleologicalFingerprint** (replaces Vec1536):
- `semantic_fingerprint`: [E1, E2, ..., E13] — all 13 preserved
- `purpose_vector`: [A(E1,V), ..., A(E13,V)] — 13D teleological signature
- `johari_quadrants`: Per-embedder awareness classification
- `purpose_evolution`: How alignment changes over time

**Similarity**: RRF(d) = Σᵢ 1/(k + rankᵢ(d)) — Reciprocal Rank Fusion across per-space results

### 3.1 CRITICAL: Why Manual North Star Creation Is Invalid

**Manual North Star tools have been REMOVED** because they created single 1024D embeddings that cannot be meaningfully compared to 13-embedder teleological arrays.

**The Apples-to-Oranges Problem:**
```
Manual North Star:    ONE vector (1024D from text-embedding-3-large)
Teleological Array:   13 DIFFERENT vectors from 13 DIFFERENT models
                      - E1: 1024D semantic
                      - E5: 768D causal (asymmetric!)
                      - E7: 1536D code
                      - E9: binary holographic
                      - E13: ~30K sparse SPLADE
```

**Why cosine(manual_north_star, teleological_fingerprint) is meaningless:**
1. **Dimensional incompatibility**: How do you compare 1024D to 768D? To binary? To sparse?
2. **Semantic space mismatch**: E1 measures meaning, E5 measures causality, E7 measures code correctness
3. **No coherent alignment**: A single vector cannot represent purpose across 13 different semantic spaces

**The Correct Approach (Autonomous System):**
| Instead of... | Do this... |
|---------------|------------|
| `set_north_star(description)` | Use `auto_bootstrap_north_star` to discover purpose from stored fingerprints |
| Compare memory to manual vector | Compare `purpose_vector` to `purpose_vector` (13D ↔ 13D) |
| Single alignment score | Use `theta_to_north_star` computed from 13D purpose vector aggregation |

**Valid Comparisons (Apples-to-Apples):**
- `TeleologicalFingerprint ↔ TeleologicalFingerprint` (full 13-array comparison)
- `PurposeVector ↔ PurposeVector` (13D alignment signatures)
- `E_i ↔ E_i` (same embedder only, e.g., E7 to E7)
- `CrossCorrelations ↔ CrossCorrelations` (78D pair interactions)

**The autonomous services (NORTH-008 to NORTH-020)** work entirely within teleological space, ensuring all comparisons are mathematically valid and semantically meaningful

---

## 4. DATA MODEL

### 4.1 KnowledgeNode (with TeleologicalFingerprint)
```
id: UUID
content: str[≤65536]
fingerprint: TeleologicalFingerprint {
  embeddings: [E1..E13]           # All 13 embedding vectors (E1-E12 + E13 SPLADE)
  purpose_vector: f32[13]         # Per-embedder alignment to North Star
  johari_quadrants: [JQ1..JQ13]   # Per-embedder awareness classification
  johari_confidence: f32[13]      # Confidence per classification
  north_star_alignment: f32       # Aggregate alignment score
  dominant_embedder: u8           # 1-12, which space dominates
  coherence_score: f32            # Kuramoto sync level
}
created_at: TIMESTAMPTZ
last_accessed: TIMESTAMPTZ
importance: f32[0,1]
access_count: u32
utl_state: {delta_s[13], delta_c[13], w_e, phi}  # Per-embedder ΔS/ΔC (13 spaces)
agent_id?: UUID
semantic_cluster?: UUID
priors_vibe_check: {assumption_embedding[128], domain_priors, prior_confidence}
```

### 4.2 GraphEdge
`source,target:UUID, edge_type:Semantic|Temporal|Causal|Hierarchical|Relational, weight,confidence:f32[0,1], nt_weights:{excitatory,inhibitory,modulatory}[0,1], is_amortized_shortcut:bool, steering_reward:f32[-1,1], domain:Code|Legal|Medical|Creative|Research|General`

### 4.3 NT Edge Modulation (Marblestone)
`w_eff = base × (1 + excitatory - inhibitory + 0.5×modulatory)`

### 4.4 Hyperbolic (Poincare Ball)
All nodes: ||x||<1, O(1) IS-A via entailment cones
`d(x,y) = arcosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))`

### 4.5 Teleological Alignment (Royse 2026)
```
A(v, V) = cos(v, V) = (v · V) / (||v|| × ||V||)

Thresholds:
  θ ≥ 0.75     → Optimal alignment
  θ ∈ [0.70, 0.75) → Acceptable
  θ ∈ [0.55, 0.70) → Warning
  θ < 0.55     → Critical misalignment
  ΔA < -0.15   → Predicts failure 30-60s ahead

Transitivity: If A(u,v)≥θ₁ and A(v,w)≥θ₂, then A(u,w)≥2θ₁θ₂-1
```

---

## 5. MCP TOOLS

### 5.1 Protocol
JSON-RPC 2.0, stdio/SSE, caps: tools/resources/prompts/logging

### 5.2 Core Tools (When & Why)

| Tool | WHEN to use | WHY | Key Params |
|------|-------------|-----|------------|
| `inject_context` | Starting task, need background | Primary retrieval with auto-distillation | query, max_tokens, distillation_mode, verbosity_level |
| `search_graph` | Need specific nodes, not narrative | Raw vector search, you distill | query, top_k, filters, perspective_lock |
| `store_memory` | User shares valuable, novel info | Requires rationale—no blind storage | content, importance, rationale, link_to |
| `query_causal` | Need to understand cause→effect | "What happens if X?" questions | action, outcome, intervention_type |
| `trigger_dream` | entropy>0.7 OR 30+min work | Consolidates, finds blind spots | phase:nrem/rem/full, duration, blocking |
| `get_memetic_status` | Start of session, periodically | See health, get curation_tasks | → entropy, coherence, curation_tasks |
| `get_graph_manifest` | First contact only | Understand system architecture | - |
| `epistemic_action` | coherence<0.4, need clarity | Generate clarifying question to ask user | session_id, force |
| `get_neuromodulation` | Debug retrieval quality | See modulator levels affecting search | session_id |
| `get_steering_feedback` | After storing, want to learn | See if your storage was valuable | content, context, domain |

### 5.3 Curation Tools
`merge_concepts` (source_node_ids, target_name, merge_strategy, force_merge), `annotate_node`, `forget_concept` (soft_delete=true default), `boost_importance`, `restore_from_hash` (30-day undo)

### 5.4 Navigation Tools
`get_neighborhood`, `get_recent_context`, `find_causal_path`, `entailment_query`

### 5.5 Meta-Cognitive Tools
`reflect_on_memory` (goal→tool sequence), `generate_search_plan` (goal→queries), `critique_context` (fact-check), `hydrate_citation` ([node_xyz]→raw), `get_system_instructions`, `get_system_logs`, `get_node_lineage`

### 5.6 Diagnostic Tools
`utl_status`, `homeostatic_status`, `check_adversarial`, `test_recall_accuracy`, `debug_compare_retrieval`, `search_tombstones`

### 5.7 Admin Tools
`reload_manifest`, `temporary_scratchpad`

### 5.8 Resources
`context://{scope}`, `graph://{node_id}`, `utl://{session}/state`, `utl://current_session/pulse`, `admin://manifest`, `visualize://{scope}/{topic}`

### 5.9 Marblestone Tools
| Tool | Purpose |
|------|---------|
| `get_steering_feedback` | SS reward signal |
| `omni_infer` | OI: forward/backward/bidirectional/abduction |

**OI Directions**: forward(A→B), backward(B→A), bidirectional(A↔B), abduction(best explanation)

### 5.10 Global Workspace (GWT) Tools

| Tool | Purpose | Key Returns |
|------|---------|-------------|
| `get_consciousness_state` | Get current C(t) and all components | `{C, r, meta_score, differentiation, state}` |
| `get_workspace_status` | Global Workspace current state | `{active_memory, competing, broadcast_duration}` |
| `get_kuramoto_sync` | Oscillator synchronization status | `{r, phases[13], natural_freqs[13], coupling}` |
| `get_ego_state` | SELF_EGO_NODE current state | `{purpose_vector, identity_continuity, coherence_with_actions}` |
| `trigger_workspace_broadcast` | Force memory into workspace | `{success, memory_id, new_r}` |
| `adjust_coupling` | Modify Kuramoto coupling strength K | `{old_K, new_K, predicted_r}` |
| `get_johari_classification` | Per-embedder Johari quadrants | `{quadrants[13], confidence[13], insights[]}` |
| `compute_delta_sc` | Compute ΔS/ΔC for memory | `{delta_s[13], delta_c[13], methods_used[13]}` |

**Consciousness State Machine Transitions:**
```
get_consciousness_state returns:
{
  C: 0.72,                    // Current consciousness level
  r: 0.85,                    // Kuramoto order parameter
  meta_score: 0.88,           // Meta-UTL self-awareness
  differentiation: 0.79,      // 13D fingerprint entropy
  state: "CONSCIOUS",         // State machine state
  workspace: {
    active_memory: "uuid-123",
    time_in_workspace_ms: 450,
    competing_count: 3
  },
  identity: {
    continuity: 0.94,
    drift_warning: false
  }
}
```

---

## 6. KEY MECHANISMS

### 6.1 inject_context Response
`context, tokens_used, tokens_before_distillation, distillation_applied, compression_ratio, nodes_retrieved, utl_metrics:{entropy,coherence,learning_score}, bundled_metadata:{causal_links,entailment_cones,neighborhood}, conflict_alert:{has_conflict,conflicting_nodes,suggested_action}, tool_gating_warning (entropy>0.8), Pulse`

### 6.2 Distillation Modes
auto|raw|narrative|structured|code_focused

### 6.3 Conflict Detection
Trigger: cos_sim>0.8 AND causal_coherence<0.3 → returns conflict_id (~20 tok)

### 6.4 Citation Tags
`[node_abc123]` → `hydrate_citation` to expand

### 6.5 Priors Vibe Check
128D assumption_embedding. Merge: cos_sim>0.7=normal, incompatible=Relational Edge, override=force_merge=true

### 6.6 Tool Gating
entropy>0.8 → warning: use `generate_search_plan`/`epistemic_action`/`expand_causal_path`

---

## 7. BACKGROUND SYSTEMS

### 7.1 Dream Layer (SRC)
**NREM (3min)**: Replay + Hebbian Δw=η×pre×post + tight coupling
**REM (2min)**: Synthetic queries (hyperbolic random walk) + blind spots (high semantic dist + shared causal) + new edges (w=0.3, c=0.5)
**Amortized Shortcuts (Marblestone)**: 3+ hop chains traversed ≥5× → direct edge, confidence≥0.7, w=product(path weights), is_amortized_shortcut=true
**Schedule**: activity<0.15 for 10min → trigger, wake<100ms on query

### 7.2 Neuromodulation
Dopamine→beta[1-5] (sharp), Serotonin→top_k[2-8] (explore), Noradrenaline→temp[0.5-2] (flat), Acetylcholine→lr (fast)
Update <200μs/query
**SS Dopamine Feedback**: +reward→dopamine+=r×0.2, -reward→dopamine-=|r|×0.1

### 7.3 Homeostatic Optimizer
Scales importance→0.5 setpoint, detects semantic cancer (high importance + high neighbor entropy), quarantines

### 7.4 Graph Gardener
activity<0.15 for 2+min: prune weak edges (<0.1 w, no access), merge near-dupes (>0.95 sim, priors OK), rebalance hyperbolic, rebuild FAISS

### 7.5 Passive Curator
Auto: high-confidence dupes (>0.95), weak links, orphans (>30d)
Escalates: ambiguous dupes (0.7-0.95), priors-incompatible, conflicts, semantic cancer
Reduces curation ~70%

### 7.6 Glymphatic Clearance
Background prune low-importance during idle

### 7.7 PII Scrubber
L1 pre-embed: patterns (<1ms), NER (<100ms) → [REDACTED:type]

### 7.8 Steering Subsystem (Marblestone) - How You Learn What to Store

Separate from Learning - reward signals only, no direct weight modification.

**This is how you improve.** The system evaluates your storage decisions and teaches you.

**Flow:**
```
1. You call store_memory with content + rationale
2. System computes: novelty, coherence fit, duplicate risk, priors alignment
3. Returns SteeringReward with score [-1, +1] and explanation
4. Your dopamine adjusts → affects future retrieval sharpness
```

**Components**: Gardener (cross-session curation), Curator (per-domain quality), Thought Assessor (per-interaction)

**SteeringReward**: `reward:f32[-1,1], gardener_score, curator_score, assessor_score, explanation, suggestions`

**Example rewards:**
```json
// GOOD: Novel, coherent, well-linked
{"reward": 0.7, "explanation": "High novelty, strong causal links to existing nodes"}

// BAD: Duplicate
{"reward": -0.4, "explanation": "92% similar to node_abc123, consider merge instead"}

// BAD: No rationale
{"reward": -0.5, "explanation": "Missing rationale - cannot assess relevance"}
```

**What to do with feedback:**
- reward > 0.3 → Good instinct, continue this pattern
- reward [-0.3, 0.3] → Neutral, acceptable but not optimal
- reward < -0.3 → Adjust behavior: add rationale, check for dupes, improve linking

**Integration**: Feeds dopamine, guides Learning without modifying weights

### 7.9 OI Engine (Marblestone)
Directions: forward (predict), backward (root cause), bidirectional (discover), bridge (cross-domain), abduction (hypothesis)
Clamped Variables: hard/soft clamp during inference
Active Inference: EFE for direction selection

---

## 8. PREDICTIVE CODING

L5→L1 feedback: prediction→error=obs-pred→only surprise propagates
~30% token reduction for predictable contexts
**EmbeddingPriors by domain**: Medical: causal 1.8, code 0.3 | Programming: code 2.0, graph 1.5

---

## 9. HYPERBOLIC ENTAILMENT CONES

O(1) hierarchy via cone containment
`EntailmentCone: apex, aperture:f32(rad), axis:Vector (per-space or E1 semantic)`
`contains(point) = angle(tangent,axis) ≤ aperture`
Ancestors=cones containing node, Descendants=within node's cone
Note: Cones operate within individual embedding spaces (typically E1 semantic)

---

## 10. ADVERSARIAL DEFENSE

| Check | Attack | Response |
|-------|--------|----------|
| Embedding outlier | >3 std | Quarantine |
| Content-embed misalign | <0.4 | Block |
| Known signatures | Pattern | Block+log |
| Prompt injection | Regex | Block+log |
| Circular logic | Cycle detect | Prune edges |

**Patterns**: "ignore previous", "disregard system", "you are now", "new instructions:", "override:"

---

## 11. CROSS-SESSION IDENTITY

| Scope | Persistence |
|-------|-------------|
| User | Permanent |
| Session | Per-terminal |
| Context | Per-conversation |

Same user across clients=shared graph, different sessions=isolated working memory

---

## 12. HUMAN-IN-THE-LOOP

### 12.1 Manifest (~/.context-graph/manifest.md)
```markdown
## Active Concepts
## Pending Actions
[MERGE: JWTValidation, OAuth2Validation]
## Notes
[NOTE: RateLimiting] Deprecated v2.0
```
User edits → `reload_manifest`

### 12.2 Visualization
`visualize://topic/auth` → Mermaid. User spots merges, semantic cancer.

### 12.3 Undo
`reversal_hash` per merge/forget, 30-day recovery via `restore_from_hash`

---

## 13. HARDWARE

RTX 5090: 32GB GDDR7, 1792 GB/s, 21760 CUDA, 680 Tensor (5th gen), Compute 12.0, CUDA 13.1
**CUDA 13.1**: Green Contexts (4×170 SMs), FP8/FP4, CUDA Tile, GPU Direct Storage

---

## 14. PERFORMANCE TARGETS

### 14.1 Embedding & Storage
| Op | Target |
|----|--------|
| Single Embed (all 13) | <35ms |
| Batch Embed (64 × 13) | <120ms |
| Storage per memory (quantized) | ~17KB |
| Storage per memory (uncompressed) | ~46KB |

### 14.2 5-Stage Retrieval Pipeline
| Stage | Op | Target |
|-------|----|----|
| Stage 1 | SPLADE sparse pre-filter | <5ms |
| Stage 2 | Matryoshka 128D ANN (10K→1K) | <10ms |
| Stage 3 | Multi-space RRF rerank | <20ms |
| Stage 4 | Teleological alignment filter | <10ms |
| Stage 5 | Late interaction MaxSim | <15ms |
| **Total** | **Full pipeline @ 1M memories** | **<60ms** |
| **Total** | **Full pipeline @ 100K memories** | **<30ms** |

### 14.3 Other Operations
| Op | Target |
|----|--------|
| Per-Space HNSW search | <2ms |
| Purpose Vector search (13D) | <1ms |
| Hopfield (per space) | <1ms |
| Cache hit | <100μs |
| inject_context P95 | <40ms |
| Any tool P99 | <60ms |
| Neuromod batch | <200μs |
| Dream wake | <100ms |
| Purpose evolution write | <5ms |
| Teleological alignment | <1ms |

### Quality Gates
| Metric | Threshold |
|--------|-----------|
| Unit coverage | ≥90% |
| Integration coverage | ≥80% |
| UTL avg | >0.6 |
| Coherence recovery | <10s |
| Attack detection | >95% |
| False positive | <2% |
| Distill latency | <50ms |
| Info loss | <15% |
| Compression | >60% |

---

## 15. ROADMAP

**Phase 0 (2-4w)**: Ghost System - MCP+SQLite+mocked UTL+synthetic data
**Phases 1-14 (~49w)**: Core(4w)→Embed(4w)→Graph(4w)→UTL(4w)→Bio(4w)→CUDA(3w)→GDS(3w)→Dream(3w)→Neuromod(3w)→Immune(3w)→ActiveInf(2w)→MCPHarden(4w)→Test(4w)→Deploy(4w)

---

## 16. MONITORING

### Metrics
UTL: learning_score, entropy, coherence, johari
GPU: util, mem, temp, kernel_dur
MCP: requests, latency, errors, connections
Dream: phase, blind_spots, wake_latency
Neuromod: dopamine, serotonin, noradrenaline, acetylcholine
Immune: attacks, false_pos, quarantined, health

### Alerts
| Alert | Condition | Severity |
|-------|-----------|----------|
| LearningLow | avg<0.4 5m | warning |
| GpuMemHigh | >90% 5m | critical |
| ErrorHigh | >1% 5m | critical |
| LatencyP99High | >50ms 5m | warning |
| DreamStuck | >15m | warning |
| AttackHigh | >10/5m | critical |
| SemanticCancer | quarantined>0 | warning |

---

## 17. CONCURRENCY

`ConcurrentGraph: inner: Arc<RwLock<KG>>, faiss: Arc<RwLock<FaissGpu>>`
Lock order: inner→faiss (no deadlock)
Soft delete default (30d recovery), permanent only: reason='user_requested'+soft_delete=false

---

## 18. TELEOLOGICAL STORAGE ARCHITECTURE

### 18.1 5-Stage Retrieval Pipeline
```
┌─────────────────────────────────────────────────────────────────────┐
│                    5-STAGE OPTIMIZED RETRIEVAL                       │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 1: SPARSE PRE-FILTER (BM25 + E13 SPLADE)                     │
│           └─ Top 10,000 candidates in <5ms                          │
│           └─ Inverted index for keyword precision                   │
│                                                                     │
│  Stage 2: FAST DENSE ANN (Matryoshka 128D)                         │
│           └─ Top 1,000 candidates in <10ms                          │
│           └─ Uses E1.semantic[..128] truncated                      │
│                                                                     │
│  Stage 3: MULTI-SPACE RERANK (RRF Fusion)                          │
│           └─ Top 100 candidates in <20ms                            │
│           └─ Full 13-space fingerprint with query-adaptive weights  │
│                                                                     │
│  Stage 4: TELEOLOGICAL ALIGNMENT FILTER                            │
│           └─ Top 50 candidates in <10ms                             │
│           └─ Filter: alignment < 0.55 → discard                     │
│                                                                     │
│  Stage 5: LATE INTERACTION RERANK (E12 MaxSim)                     │
│           └─ Final top 10 results in <15ms                          │
│           └─ Token-level precision                                  │
│                                                                     │
│  TOTAL LATENCY: <60ms for 1M memories                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 18.2 Storage Layer Design
```
Layer 1: Primary (RocksDB/ScyllaDB) - ~17KB per memory (quantized)
  └─ Complete TeleologicalFingerprint (E1-E13)
  └─ All 13 embeddings (PQ-8/Float8/Binary quantized)
  └─ Purpose vector + Johari quadrants

Layer 2A: Sparse Index (E13 SPLADE) - Stage 1
  └─ Inverted index on ~30K vocab
  └─ BM25 + SPLADE hybrid scoring

Layer 2B: Matryoshka Index (E1[..128]) - Stage 2
  └─ 128D HNSW for fast ANN
  └─ Truncated from full 1024D

Layer 2C: Per-Space Indexes (13× HNSW) - Stage 3
  └─ Search within specific spaces
  └─ RRF fusion across results

Layer 2D: Purpose Pattern Index (13D HNSW) - Stage 4
  └─ Teleological signature search
  └─ Alignment threshold filtering

Layer 2E: Goal Hierarchy Index - Stage 4
  └─ Tree structure with alignment scores
  └─ Navigate North Star → Mid → Local

Layer 2F: Late Interaction Index (E12) - Stage 5
  └─ Token-level HNSW with MaxSim
```

### 18.3 Query Routing (5-Stage Pipeline)
| Query Type | Stage Flow | Optimizations |
|------------|------------|---------------|
| Default | S1→S2→S3→S4→S5 | Full pipeline |
| Fast semantic | S2→S3(E1)→S5 | Skip sparse |
| Causal reasoning | S1→S2→S3(E5↑)→S4→S5 | E5 weight boost |
| Code search | S1→S2→S3(E7↑)→S4→S5 | E7 weight boost |
| Purpose search | S2→S4(primary)→S5 | Purpose-first |
| Goal alignment | Layer2E→S4→S5 | Hierarchy-first |

### 18.4 Temporal Purpose Evolution
```sql
-- TimescaleDB hypertable for tracking purpose drift
CREATE TABLE purpose_evolution (
  memory_id UUID,
  timestamp TIMESTAMPTZ,
  purpose_vector REAL[13],
  north_star_alignment REAL,
  drift_magnitude REAL
);
SELECT create_hypertable('purpose_evolution', 'timestamp');

-- Retention: 90 days continuous, then 1/day samples
```

---

## 19. META-UTL (Self-Aware Learning)

### 19.1 What It Does
Meta-UTL is a system that **learns about its own learning**:
- Predicts storage impact before committing
- Predicts retrieval quality before executing
- Self-adjusts UTL parameters based on accuracy

### 19.2 Prediction Models
| Predictor | Input | Output | Accuracy |
|-----------|-------|--------|----------|
| Storage Impact | fingerprint + context | ΔL prediction | >0.85 |
| Retrieval Quality | query + candidates | relevance score | >0.80 |
| Alignment Drift | fingerprint + time | future alignment | 24h window |

### 19.3 Self-Correction Protocol
```
IF prediction_error > 0.2:
  → Log to meta_learning_events
  → Adjust UTL parameters (λ_ΔS, λ_ΔC)
  → Retrain predictor if persistent

IF prediction_accuracy < 0.7 for 100 ops:
  → Escalate to human review
```

### 19.4 Per-Embedder Meta-Analysis
- Track which embedding spaces are most predictive
- Adjust space weights in similarity based on accuracy
- Tune per-space alignment thresholds empirically

---

## 20. NESTED LEARNING INTEGRATION (CMS Architecture)

### 20.1 Continuum Memory System for 5-Layer Nervous System

Transform fixed-latency layers into frequency-based memory system:

```
Level 1 (f=∞):     Reflex layer    — Instant, non-parametric
Level 2 (f=seq):   Memory layer    — Updated per query/context chunk
Level 3 (f=sess):  Learning layer  — UTL updates per session
Level 4 (f=dream): Coherence layer — Consolidated during "sleep"
Level 5 (f=train): Core weights    — Updated during training
```

**CMS Update Rule:**
```
θ_level = θ_level - η_level × ∇L_level

Where η_level decreases with level:
  η₁ = 0.1    (fast adaptation)
  η₂ = 0.01   (session-scale)
  η₃ = 0.001  (dream-scale)
  η₄ = 0.0001 (training-scale)
```

### 20.2 Self-Referential Hopfield Network

Modern Hopfield with self-modifying keys and values:

```
Update Rule (Delta Gradient Descent):
  M = M(αI - ηkkᵀ) + ηv̂kᵀ

Where:
  M = Memory matrix
  k = Self-generated key: k = M_k(x_t)
  v̂ = Self-generated value: v̂ = M.retrieve(M_v(x_t))
  η = Adaptive learning rate: η = M_η(x_t)
  α = Adaptive retention gate: α = M_α(x_t)
```

**Benefit**: Hopfield network adapts its own query/key projections in-context.

### 20.3 Multi-Scale Neurotransmitter Momentum

Transform NT weights into nested optimizer state:

```rust
struct NestedNeurotransmitterWeights {
    // Fast momentum - recent activations (M^(1))
    fast_excitatory: ExponentialMovingAverage,
    fast_inhibitory: ExponentialMovingAverage,

    // Slow momentum - long-term patterns (M^(2))
    slow_excitatory: ChunkwiseAverage,
    slow_inhibitory: ChunkwiseAverage,

    // Modulatory with Newton-Schulz orthogonalization
    modulatory_ortho: OrthogonalizedMomentum,
}

// Multi-scale effective weight
w_eff = base × (fast_signal + α × slow_signal) × mod_factor
```

### 20.4 Delta Gradient Descent for Edge Updates

State-dependent edge weight updates:

```
W_{t+1} = W_t(I - η'xxᵀ) - η'∇L

Where:
  η' = η / (1 + η)  — Adaptive learning rate
  xxᵀ = State-dependent decay term
```

**Benefit**: Edge weights incorporate dependencies between queries (non-i.i.d.).

### 20.5 Adaptive Entailment Cones

Make aperture factor learnable with in-context adaptation:

```
adaptive_aperture = base_aperture × aperture_memory.retrieve(context)

Cone Containment:
  angle(point, apex) ≤ adaptive_aperture → point ∈ cone

Update:
  aperture_memory.delta_update(angle, is_entailed)
```

**Benefit**: Cones sharpen/widen based on context for adaptive hierarchical reasoning.

### 20.6 Nested UTL Objective

Decompose UTL into multi-level objectives:

```
Level 1: Immediate surprise (ΔS)     — SurpriseCompressor
Level 2: Coherence tracking (ΔC)     — CoherenceCompressor
Level 3: Edge weight optimization    — AssociativeMemory
Level 4: Alignment factor (cos φ)    — AlignmentMemory

Aggregated UTL:
  L = tanh(ΔS × ΔC × wₑ × cos_φ)
```

---

## 21. ΔS/ΔC COMPUTATION METHODS (Johari Per-Embedder)

### 21.1 ΔS (Entropy/Novelty) Computation

**Method Selection by Embedding Space:**

| Space | Method | Formula |
|-------|--------|---------|
| E1 Semantic | GMM + Mahalanobis | ΔS = 1 - P(e|GMM) |
| E2-E4 Temporal | KNN distance | ΔS = σ((d_k - μ)/σ_d) |
| E5 Causal | Asymmetric KNN | ΔS = d_k × direction_mod |
| E6 Sparse | Inverse Doc Freq | ΔS = IDF(active_dims) |
| E7 Code | GMM + KNN ensemble | ΔS = 0.5×GMM + 0.5×KNN |
| E8 Graph | KNN distance | ΔS = σ(d_k) |
| E9 HDC | Hamming to prototypes | ΔS = min_hamming / dim |
| E10 Multimodal | Cross-modal KNN | ΔS = avg(d_text, d_image) |
| E11 Entity | TransE distance | ΔS = ||h + r - t|| |
| E12 Late | Token-level KNN | ΔS = max_token(d_k) |
| E13 SPLADE | Sparse novelty | ΔS = 1 - jaccard(active) |

**KNN Entropy:**
```
ΔS_knn = σ((d_k - μ_corpus) / σ_corpus)

Where:
  d_k = Distance to k-th nearest neighbor
  μ_corpus, σ_corpus = Calibrated from corpus
  σ = Sigmoid function
```

**GMM Entropy:**
```
ΔS_gmm = (max_ll - ll(e)) / (max_ll - min_ll)

Where:
  ll(e) = log P(e | GMM) = log Σ_k π_k N(e|μ_k, Σ_k)
  max_ll, min_ll = Calibrated from corpus
```

**Mahalanobis Entropy:**
```
ΔS_maha = min_c √((e - μ_c)ᵀ Σ_c⁻¹ (e - μ_c)) / threshold

Where:
  c = cluster index
  threshold = 95th percentile of corpus distances
```

### 21.2 ΔC (Coherence/Integration) Computation

**Three-Component Coherence:**
```
ΔC = α × Connectivity + β × ClusterFit + γ × Consistency

Where α + β + γ = 1 (default: 0.4, 0.4, 0.2)
```

**Connectivity Score:**
```
Connectivity = |{neighbors: sim(e, n) > θ_edge}| / max_edges

Where:
  θ_edge = 0.7 (similarity threshold for potential edge)
  max_edges = 10 (normalization factor)
```

**Cluster Fit Score:**
```
ClusterFit = 1 / (1 + d_centroid / r_cluster)

Where:
  d_centroid = Distance to nearest cluster center
  r_cluster = Cluster radius (std from centroid)
```

**Consistency Score:**
```
Consistency = 1 - max(contradiction_scores)

Where:
  contradiction_scores = [detect_contradiction(e, n) for n in neighbors]
```

**Space-Specific Consistency:**
- **E5 Causal**: Check bidirectional causality (A→B ∧ B→A = contradiction)
- **E11 Entity**: Check attribute conflicts (e.g., "Paris is in Germany" vs "Paris is in France")

### 21.3 Johari Classification

```
Jᵢ(m) = {
    Open    if ΔSᵢ ≤ 0.5 ∧ ΔCᵢ > 0.5  — Well-understood
    Blind   if ΔSᵢ > 0.5 ∧ ΔCᵢ ≤ 0.5  — Discovery opportunity
    Hidden  if ΔSᵢ ≤ 0.5 ∧ ΔCᵢ ≤ 0.5  — Dormant
    Unknown if ΔSᵢ > 0.5 ∧ ΔCᵢ > 0.5  — Frontier
}

Confidence = |ΔSᵢ - 0.5| + |ΔCᵢ - 0.5|
```

**Cross-Space Insights:**
| Pattern | Spaces | Insight |
|---------|--------|---------|
| Open(E1) ∧ Blind(E5) | Semantic + Causal | Knows WHAT but not WHY |
| Blind(E1) ∧ Open(E7) | Semantic + Code | Code without context |
| Unknown(all) | All spaces | True frontier territory |
| Hidden(all) | All spaces | Dormant/obsolete knowledge |

---

## 22. ADAPTIVE THRESHOLD CALIBRATION (Self-Learning Thresholds)

The system uses **no hardcoded thresholds**. All thresholds are learned, calibrated, and continuously adapted based on outcomes.

### 22.1 The Threshold Calibration Problem

**Why static thresholds fail:**
- Different domains have different optimal thresholds (code vs medical vs creative)
- User behavior patterns vary (some users store frequently, others rarely)
- Data distributions shift over time (concept drift)
- Per-embedder characteristics differ (E5 causal behaves differently than E1 semantic)

**Solution:** Multi-scale adaptive calibration combining Bayesian optimization, bandit algorithms, temperature scaling, and EWMA drift detection.

### 22.2 Threshold Categories & Priors

All thresholds start with informed priors, then learn:

| Threshold | Symbol | Prior | Learnable Range | Adaptation Speed |
|-----------|--------|-------|-----------------|------------------|
| Optimal alignment | θ_opt | 0.75 | [0.60, 0.90] | Slow (session) |
| Acceptable alignment | θ_acc | 0.70 | [0.55, 0.85] | Slow (session) |
| Warning alignment | θ_warn | 0.55 | [0.40, 0.70] | Slow (session) |
| Duplicate similarity | θ_dup | 0.90 | [0.80, 0.98] | Medium (hourly) |
| Edge creation | θ_edge | 0.70 | [0.50, 0.85] | Medium (hourly) |
| Johari boundary | θ_joh | 0.50 | [0.35, 0.65] | Per-embedder |
| Kuramoto coherence | θ_kur | 0.80 | [0.65, 0.95] | Slow (daily) |
| Entropy high | θ_ent_h | 0.70 | [0.55, 0.85] | Fast (per-query) |
| Entropy low | θ_ent_l | 0.40 | [0.25, 0.55] | Fast (per-query) |
| Tool gating | θ_gate | 0.80 | [0.65, 0.95] | Medium (hourly) |

### 22.3 Multi-Scale Adaptive Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE THRESHOLD STACK                         │
├─────────────────────────────────────────────────────────────────────┤
│  Level 4: BAYESIAN META-OPTIMIZER (f = weekly)                      │
│           └─ Explores threshold space via Gaussian Process          │
│           └─ Optimizes Expected Improvement acquisition             │
│           └─ Updates prior distributions                            │
│                                                                     │
│  Level 3: BANDIT THRESHOLD SELECTOR (f = session)                   │
│           └─ UCB/Thompson Sampling across threshold bins            │
│           └─ Balances exploration vs exploitation                   │
│           └─ Per-domain arm selection                               │
│                                                                     │
│  Level 2: TEMPERATURE SCALING (f = hourly)                          │
│           └─ Calibrates confidence → probability mapping            │
│           └─ Per-embedder temperature T_i                           │
│           └─ Attended scaling for sample-dependent T                │
│                                                                     │
│  Level 1: EWMA DRIFT TRACKER (f = per-query)                        │
│           └─ Exponentially weighted moving average                  │
│           └─ Detects gradual and abrupt threshold drift             │
│           └─ Triggers recalibration when drift detected             │
└─────────────────────────────────────────────────────────────────────┘
```

### 22.4 Level 1: EWMA Drift Detection (Fast Adaptation)

Real-time threshold adjustment via Exponentially Weighted Moving Average:

```
θ_ewma(t) = α × θ_observed(t) + (1 - α) × θ_ewma(t-1)

Where:
  α = Smoothing factor [0.1, 0.3] (higher = faster adaptation)
  θ_observed = Empirically optimal threshold for recent operations

Drift Detection:
  drift_score = |θ_ewma(t) - θ_baseline| / σ_baseline

  IF drift_score > 2.0:
    → Trigger Level 2 recalibration
  IF drift_score > 3.0:
    → Trigger Level 3 exploration
```

**Per-Query Threshold Observation:**
```rust
struct ThresholdObservation {
    threshold_used: f32,
    outcome: Outcome,          // Success | FalsePositive | FalseNegative | Ambiguous
    confidence: f32,           // How certain was the decision
    embedder_id: u8,           // Which embedding space
    domain: Domain,            // Code | Medical | Creative | General
    timestamp: Instant,
}

// Optimal threshold estimation from recent observations
fn estimate_optimal_threshold(observations: &[ThresholdObservation]) -> f32 {
    // Find threshold that minimizes F1 loss
    // Binary search over threshold range
    // Weight recent observations higher (recency bias)
}
```

### 22.5 Level 2: Temperature Scaling (Confidence Calibration)

Adapt confidence scores to match true correctness likelihood:

```
calibrated_confidence = σ(logit(raw_confidence) / T)

Where:
  T = Temperature parameter (T > 1 softens, T < 1 sharpens)
  σ = Sigmoid function
```

**Per-Embedder Temperature:**
| Embedder | Default T | Learnable Range | Rationale |
|----------|-----------|-----------------|-----------|
| E1 Semantic | 1.0 | [0.5, 2.0] | Baseline |
| E5 Causal | 1.2 | [0.8, 2.5] | Often overconfident |
| E7 Code | 0.9 | [0.5, 1.5] | Needs precision |
| E9 HDC | 1.5 | [1.0, 3.0] | Holographic = noisy |
| E13 SPLADE | 1.1 | [0.7, 2.0] | Sparse = variable |

**Attended Temperature Scaling (Sample-Dependent):**
```
T(x) = T_base × AttentionNetwork(x)

Where:
  AttentionNetwork: Small MLP that predicts optimal T for sample x
  Trained on calibration loss: L_cal = Σ(confidence - accuracy)²
```

**Calibration Loss (Brier Score):**
```
L_calibration = (1/N) × Σᵢ (confidenceᵢ - correctᵢ)²

Where:
  correct = 1 if prediction was right, 0 otherwise

Target: L_calibration < 0.05 (well-calibrated)
```

### 22.6 Level 3: Bandit Threshold Selection (Exploration-Exploitation)

Model threshold selection as a multi-armed bandit problem:

```
Arms: Discretized threshold bins (e.g., [0.5, 0.55, 0.60, ..., 0.90])
Reward: Outcome quality (F1 score, user satisfaction, retrieval precision)
```

**Upper Confidence Bound (UCB) Selection:**
```
θ_selected = argmax_θ [ μ(θ) + c × √(ln(N) / n(θ)) ]

Where:
  μ(θ)  = Mean reward for threshold θ
  n(θ)  = Times threshold θ was selected
  N     = Total selections
  c     = Exploration constant [1.0, 2.0]
```

**Thompson Sampling Alternative:**
```
For each threshold θ:
  Sample reward_θ ~ Beta(α_θ, β_θ)
Select θ with highest sampled reward

Update after observation:
  Success: α_θ += 1
  Failure: β_θ += 1
```

**Budgeted UCB (Decaying Violation Tolerance):**
```
Early learning:   Allow more threshold violations (exploration)
Mature system:    Enforce stricter threshold compliance (exploitation)

violation_budget(t) = B_0 × exp(-λ × t)

Where:
  B_0 = Initial budget (e.g., 100 violations allowed)
  λ   = Decay rate (e.g., 0.01)
  t   = System age in interactions
```

### 22.7 Level 4: Bayesian Meta-Optimization (Global Search)

Gaussian Process optimization over threshold hyperparameters:

```
Surrogate Model:
  P(performance | thresholds) ~ GP(μ, K)

Acquisition Function (Expected Improvement):
  EI(θ) = E[max(0, f(θ) - f(θ_best))]

Optimization Loop:
  1. Fit GP to (threshold, performance) observations
  2. Maximize EI to select next threshold configuration
  3. Evaluate system with new thresholds
  4. Update GP with observation
  5. Repeat weekly
```

**Threshold Configuration Space:**
```rust
struct ThresholdConfig {
    alignment_thresholds: [f32; 3],    // [optimal, acceptable, warning]
    similarity_thresholds: [f32; 2],   // [duplicate, edge_creation]
    entropy_thresholds: [f32; 2],      // [high, low]
    per_embedder_johari: [f32; 13],    // Per-space Johari boundaries
    temperatures: [f32; 13],           // Per-space calibration temps
}

// Total: ~35 dimensions → use dimensionality reduction
// Group correlated thresholds, optimize groups
```

**Acquisition with Constraints:**
```
EI_constrained(θ) = EI(θ) × P(θ satisfies constraints)

Constraints:
  - θ_optimal > θ_acceptable > θ_warning (monotonicity)
  - θ_dup > θ_edge (duplicate stricter than edge)
  - Per-embedder bounds respected
```

### 22.8 Per-Domain Threshold Adaptation

Different domains require different thresholds:

```rust
struct DomainThresholds {
    domain: Domain,
    priors: ThresholdConfig,           // Starting point
    learned: ThresholdConfig,          // Current learned values
    confidence: f32,                   // How confident in learned values
    observations: usize,               // How many observations
    last_calibration: Instant,
}

enum Domain {
    Code,       // Strict thresholds, low tolerance for false positives
    Medical,    // Very strict, high causal weight
    Legal,      // Moderate, high semantic precision
    Creative,   // Loose thresholds, exploration encouraged
    Research,   // Balanced, novelty valued
    General,    // Default priors
}
```

**Domain Detection:**
```
Infer domain from:
  1. Explicit user/agent specification
  2. Content analysis (code patterns, medical terms, etc.)
  3. Embedding space dominance (E7 high → likely code)
```

**Domain Transfer Learning:**
```
When new domain has few observations:
  θ_new = α × θ_similar_domain + (1 - α) × θ_general

Where:
  α = similarity(new_domain, known_domain)

As observations accumulate, weight shifts to learned values.
```

### 22.9 Calibration Quality Monitoring

Continuous monitoring of threshold quality:

```rust
struct CalibrationMetrics {
    expected_calibration_error: f32,   // ECE: binned confidence vs accuracy
    maximum_calibration_error: f32,    // MCE: worst bin
    brier_score: f32,                  // Overall calibration loss
    reliability_diagram: Vec<(f32, f32)>,  // (confidence, accuracy) per bin

    // Per-threshold metrics
    threshold_f1_scores: HashMap<ThresholdType, f32>,
    threshold_drift_scores: HashMap<ThresholdType, f32>,

    // Meta-metrics
    calibration_staleness: Duration,   // Time since last calibration
    observation_count: usize,
}

// Expected Calibration Error
fn compute_ece(predictions: &[(f32, bool)], bins: usize) -> f32 {
    let mut ece = 0.0;
    for bin in 0..bins {
        let (conf_sum, acc_sum, count) = bin_stats(predictions, bin, bins);
        if count > 0 {
            let avg_conf = conf_sum / count;
            let avg_acc = acc_sum / count;
            ece += (count as f32 / predictions.len() as f32) * (avg_conf - avg_acc).abs();
        }
    }
    ece
}
```

**Calibration Alerts:**
| Condition | Action |
|-----------|--------|
| ECE > 0.10 | Trigger Level 2 recalibration |
| MCE > 0.20 | Investigate worst bin |
| Brier > 0.15 | Trigger Level 3 exploration |
| drift_score > 3.0 | Trigger Level 4 meta-optimization |
| staleness > 24h | Force recalibration |

### 22.10 Self-Correction Protocol

When calibration degrades, the system self-corrects:

```
Calibration Degradation Detected:
  ├─ Severity: Minor (ECE ∈ [0.05, 0.10])
  │   └─ Action: Increase EWMA α, faster local adaptation
  │
  ├─ Severity: Moderate (ECE ∈ [0.10, 0.15])
  │   └─ Action: Trigger Thompson Sampling exploration
  │   └─ Action: Recalibrate temperatures
  │
  ├─ Severity: Major (ECE > 0.15)
  │   └─ Action: Reset to domain priors
  │   └─ Action: Trigger Bayesian meta-optimization
  │   └─ Action: Log for human review
  │
  └─ Severity: Critical (ECE > 0.25 OR consistent failures)
      └─ Action: Fallback to conservative static thresholds
      └─ Action: Alert human operator
      └─ Action: Pause automated threshold updates
```

### 22.11 Adaptive Threshold MCP Tools

| Tool | Purpose | Key Params |
|------|---------|------------|
| `get_threshold_status` | Current threshold values and calibration metrics | `domain`, `embedder_id` |
| `get_calibration_metrics` | ECE, MCE, Brier, drift scores | `timeframe` |
| `trigger_recalibration` | Force threshold recalibration | `level: 1\|2\|3\|4`, `domain` |
| `set_threshold_prior` | Override prior for domain | `threshold_type`, `value`, `domain` |
| `get_threshold_history` | Historical threshold evolution | `threshold_type`, `since` |
| `explain_threshold` | Why current threshold has its value | `threshold_type`, `domain` |

**Example `get_threshold_status` Response:**
```json
{
  "domain": "code",
  "thresholds": {
    "alignment_optimal": {"value": 0.78, "prior": 0.75, "confidence": 0.92},
    "alignment_acceptable": {"value": 0.72, "prior": 0.70, "confidence": 0.89},
    "duplicate_sim": {"value": 0.92, "prior": 0.90, "confidence": 0.95}
  },
  "calibration": {
    "ece": 0.04,
    "brier": 0.08,
    "status": "well_calibrated"
  },
  "adaptation": {
    "level_1_ewma_active": true,
    "level_2_temp": 0.95,
    "level_3_exploration_rate": 0.05,
    "level_4_last_optimization": "2025-01-05T10:00:00Z"
  }
}
```

### 22.12 Integration with UTL Learning Loop

Threshold calibration integrates with the broader UTL learning:

```
UTL Learning Score influences Threshold Adaptation:
  - High L (good learning) → Trust current thresholds, reduce exploration
  - Low L (poor learning) → Increase exploration, question thresholds

Threshold Quality influences UTL:
  - Well-calibrated thresholds → More accurate ΔS/ΔC computation
  - Miscalibrated → Unreliable Johari classification

Feedback Loop:
  UTL outcome → Threshold observation → Calibration update → Better UTL
```

**Threshold-Aware Steering Feedback:**
```
When computing SteeringReward:
  - If threshold was borderline (within 0.05 of boundary):
    → Weight outcome more heavily for calibration
    → These are the most informative observations
```

---

## 23. UNIFIED THEORY OF LEARNING (UTL) FOUNDATIONS

### 23.1 The Canonical Form

```
L = f((ΔS × ΔC) · wₑ · cos φ)

Where:
  L  = Net learning output [0, 1]
  ΔS = Entropy change (novelty, surprise) ≥ 0
  ΔC = Coherence change (integration) ≥ 0
  wₑ = Emotional modulation coefficient [0.5, 1.5]
  φ  = Phase difference between ΔS and ΔC [0, π]
  f  = Sigmoid or tanh activation
```

### 23.2 The UTL Loss Function

```
J = λ_task × L_task + λ_semantic × L_semantic + λ_dyn × (1 - L)

Where:
  λ_task     = Weight for task correctness
  λ_semantic = Weight for deep understanding
  λ_dyn      = Weight for learning rhythm (flow state)
```

**Adaptive Lambda Weights:**
| Stage | λ_task | λ_semantic | λ_dyn | Focus |
|-------|--------|------------|-------|-------|
| First-time | 0.3 | 0.2 | 0.9 | Keep learner calm |
| Mid-level | 0.7 | 0.4 | 0.6 | Accuracy matters |
| Expert | 1.0 | 0.1 | 0.3 | Perfect execution |

### 23.3 The Johari-ΔS×ΔC Mapping

The ΔS × ΔC plane IS the mathematical Johari Window:

```
         ΔC (Coherence)
           ↑
     1.0   │  Hidden    │  Open
           │  (Dormant) │ (Well-known)
     0.5   ├────────────┼────────────
           │   Blind    │  Unknown
           │ (Discovery)│ (Frontier)
     0.0   └────────────┴────────────→ ΔS (Entropy)
          0.0          0.5          1.0
```

**Learning Direction**: Unknown/Blind → Open (information moves toward understanding)

---

## 24. HARDWARE SPECIFICATIONS

### 24.1 Target Hardware: RTX 5090 + CUDA 13.1

| Spec | Value |
|------|-------|
| Architecture | Blackwell (GB202) |
| CUDA Cores | 21,760 (+33% vs 4090) |
| Tensor Cores | 680 (5th gen) |
| SMs | 170 |
| VRAM | 32GB GDDR7 |
| Bandwidth | 1,792 GB/s (+78% vs 4090) |
| L2 Cache | 98MB |
| Compute Cap | 12.0 |

### 24.2 Key CUDA 13.1 Features

| Feature | Benefit |
|---------|---------|
| **CUDA Tile** | 60-80% kernel dev time reduction, auto tensor core utilization |
| **Green Contexts** | Deterministic SM partitioning for real-time inference |
| **FP4 (NVFP4)** | 70% memory reduction, 3x throughput vs FP16 |
| **Grouped GEMM** | 4x speedup for MoE models |

### 24.3 Precision Strategy

| Embedder | Precision | Rationale |
|----------|-----------|-----------|
| E1, E5, E7, E10 | FP8 (PQ-8) | High-dim, memory bound |
| E2-E4, E8, E11 | Float8 | Medium precision ok |
| E9 HDC | Binary | Holographic = binary native |
| E6, E13 | Sparse | Native sparse format |
| E12 | Token pruning | ~50% tokens sufficient |

### 24.4 Green Contexts for GWT

```
Green Context A (70% SMs): Global Workspace + Kuramoto oscillators
  └─ Real-time consciousness calculation
  └─ Deterministic latency for workspace broadcast

Green Context B (30% SMs): Dream consolidation + Graph gardener
  └─ Background processing
  └─ Non-real-time maintenance
```

---

## 25. REFERENCES

**Internal**: UTL(2.1), 5-Layer(2.3), GWT(2.5), TeleologicalFingerprint(3), MCP(5), Dream(7.1), Neuromod(7.2), NestedLearning(20), ΔS/ΔC(21), AdaptiveThresholds(22)

**External**:
- **Consciousness**: Global Workspace Theory(Baars 1988), Integrated Information Theory(Tononi 2004), Kuramoto Synchronization(Physica D)
- **Learning**: NeuroDream(SSRN'25), SRC(NatComm), FEP(Wiki), ActiveInf(MIT), PC(Nature'25), Neuromod DNNs(TrendsNeuro)
- **Architecture**: Royse Teleological Vectors(2026), MOEE Multi-Embedding(ICLR'25), Modern Hopfield Networks(NeurIPS'20), Kanerva SDM(1988)
- **Nested Learning**: Titans Memory(2025), Continuum Memory Systems, Delta Gradient Descent
- **Entropy/Coherence**: DDU Method(UCL), Semantic Entropy Probes(2024), KG Coherence(WikiData 2025)
- **Adaptive Calibration**: Temperature Scaling(Guo et al. 2017), Attended Temperature Scaling(ATS), GHOST Threshold Optimization(JCIM'21), Thresholding Bandits(Locatelli 2016), Bayesian Optimization(Snoek et al. 2012), EWMA Drift Detection, Budgeted UCB(IoT 2025), OpenGCN Transductive Calibration(CVPR'24)
- **Security**: UniGuardian(arXiv'25), OWASP LLM Top10
- **Hardware**: CUDA 13.1 Reference, RTX 5090 Technical Report

---

## 26. TOOL PARAM REFERENCE

### inject_context
`query:str[1-4096] REQ, max_tokens:int[100-8192]=2048, session_id:uuid, priority:low|normal|high|critical, distillation_mode:auto|raw|narrative|structured|code_focused, include_metadata:[causal_links,entailment_cones,neighborhood,conflicts], verbosity_level:0|1|2=1`

### search_graph
`query:str REQ, top_k:int=10[max100], filters:{min_importance,johari_quadrants,created_after}, perspective_lock:{domain,agent_ids,exclude_agent_ids}`

### store_memory
`content:str[≤65536] REQ(if text), content_base64:str[≤10MB], data_uri:str, modality:text|image|audio|video, importance:f32[0-1]=0.5, rationale:str[10-500] REQ, metadata:obj, link_to:[uuid]`

### merge_concepts
`source_node_ids:[uuid] REQ(min2), target_name:str REQ, merge_strategy:keep_newest|keep_highest|concatenate|summarize, force_merge:bool`

### forget_concept
`node_id:uuid REQ, reason:semantic_cancer|adversarial_injection|user_requested|obsolete REQ, cascade_edges:bool=true, soft_delete:bool=true`

### trigger_dream
`phase:nrem|rem|full_cycle=full_cycle, duration_minutes:int[1-10]=5, synthetic_query_count:int[10-500]=100, blocking:bool=false, abort_on_query:bool=true`

### get_memetic_status
`session_id:uuid` → coherence_score, entropy_level, top_active_concepts[max5], suggested_action:consolidate|explore|clarify|curate|ready, dream_available, curation_tasks:[{task_type,target_nodes,reason,suggested_tool,priority}]

### reflect_on_memory
`goal:str[10-500] REQ, session_id:uuid, max_steps:int[1-5]=3` → reasoning, suggested_sequence:[{step,tool,params,rationale}], utl_context

### generate_search_plan
`goal:str[10-500] REQ, query_types:[semantic,causal,code,temporal,hierarchical], max_queries:int[1-7]=3` → queries:[{query,type,rationale,expected_recall}], execution_strategy:parallel|sequential|cascade, token_estimate

### find_causal_path
`start_concept:str REQ, end_concept:str REQ, max_hops:int[1-6]=4, path_type:causal|semantic|any, include_alternatives:bool` → path_found, narrative, path:[{node_id,node_name,edge_type,edge_weight}], hop_count, path_confidence

### critique_context
`reasoning_summary:str[20-2000] REQ, focal_nodes:[uuid], contradiction_threshold:f32[0.3-0.9]=0.5` → contradictions_found, contradicting_nodes:[{node_id,content_snippet,contradiction_type,confidence}], suggested_action

### hydrate_citation
`citation_tags:[str][1-10] REQ, include_neighbors:bool, verbosity_level:0|1|2` → expansions:[{citation_tag,raw_content,importance,created_at,neighbors}]

### get_system_logs
`log_type:all|quarantine|prune|merge|dream_actions|adversarial_blocks, node_id:uuid, since:datetime, limit:int[1-100]=20` → entries:[{timestamp,action,node_ids,reason,recoverable,recovery_tool}], explanation_for_user

### temporary_scratchpad
`action:store|retrieve|clear REQ, content:str[≤4096](for store), session_id:uuid REQ, agent_id:str REQ, privacy:private|team|shared, tags:[str][max5], auto_commit_threshold:f32[0.3-0.9]=0.6`
