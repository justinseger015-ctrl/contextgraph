# Causal Embedder & LLM System: Comprehensive Benchmarking Plan

## Goal

Benchmark every component of the causal reasoning pipeline — from E5 embedding generation through LLM causal discovery to multi-embedder retrieval — with the specific objective of **optimizing end-to-end accuracy**. Each benchmark produces quantitative metrics that feed directly into tunable parameters.

---

## System Under Test

### Architecture Summary

```
                         DISCOVERY PIPELINE
                         ==================
Memory Store ──► MemoryScanner (candidate pairs)
                      │
                      ▼
              CausalDiscoveryLLM (Hermes-2-Pro-Mistral-7B)
              ├─ GBNF grammar-constrained JSON output
              ├─ {has_causal_link, direction, confidence, mechanism}
              └─ Few-shot (5 examples) or zero-shot mode
                      │
                      ▼
              E5EmbedderActivator
              ├─ CausalModel.embed_dual_guided()
              ├─ LLM guidance injection (CausalHintGuidance)
              └─ Returns (cause_vec 768D, effect_vec 768D)
                      │
                      ▼
              RocksDB (CF_CAUSAL_RELATIONSHIPS)
              ├─ E5 cause vector (768D)
              ├─ E5 effect vector (768D)
              ├─ E1 semantic fallback (1024D)
              └─ Source metadata + provenance

                         SEARCH PIPELINE
                         ================
Query ──► detect_causal_query_intent()
              ├─ 100+ cause indicators
              └─ 90+ effect indicators
                      │
                      ▼
          embed_all() → SemanticFingerprint (13 embedders)
                      │
                      ▼
          ┌─────────────────────────────────────┐
          │        SEARCH MODE                   │
          ├─ search_causes:  query as effect,    │
          │    search cause vectors, 0.8x mod    │
          ├─ search_effects: query as cause,     │
          │    search effect vectors, 1.2x mod   │
          └─ semantic: E1 fallback               │
          └─────────────────────────────────────┘
                      │
                      ▼
          Multi-Embedder RRF Fusion
          ├─ E1 (0.30): Semantic foundation
          ├─ E5 (0.35): Causal asymmetric
          ├─ E8 (0.15): Graph structure
          └─ E11 (0.20): Entity knowledge
                      │
                      ▼
          Ranked results with provenance
```

### Key Components

| Component | Location | Model/Config |
|-----------|----------|--------------|
| E5 Causal Embedder | `context-graph-embeddings/.../causal/` | Longformer-base-4096, 768D |
| LLM Discovery | `context-graph-causal-agent/src/llm/` | Hermes-2-Pro-Mistral-7B (GGUF Q4_K_M) |
| Asymmetric Similarity | `context-graph-core/src/causal/asymmetric.rs` | 1.2x/0.8x direction mods |
| Causal Search Tools | `context-graph-mcp/.../causal_tools.rs` | search_causes, search_effects, get_causal_chain |
| Training Module | `context-graph-embeddings/src/training/` | DirectionalContrastiveLoss (4-component) |
| Weight Profiles | `context-graph-core/src/retrieval/` | causal_reasoning: E5@0.45 |

### Known Issues From Prior Research

1. **E5 encodes causal STRUCTURE (markers), not domain-specific causation** — E5 scores cluster 0.93-0.98 for ALL causal text regardless of topic. E1 (semantic) needed for topical discrimination.
2. **Asymmetric projection REDUCES E5 scores** — cause=0.635, effect=0.668, none=0.856 for same content.
3. **Direction modifiers confirmed** — search_effects/search_causes ratio = 1.46x (theoretical 1.5x).
4. **E5 weight recommended at 0.20-0.30** in fusion (not current 0.45).
5. **Intervention overlap disabled** — showed -15.9% correlation, formula simplified to `base_cos * direction_mod`.

---

## Phase 1: E5 Embedding Quality Benchmarks

**Objective**: Quantify how well the E5 model produces distinct, directionally meaningful cause/effect vectors.

### 1.1 Vector Distinctness

**What**: Measure cosine distance between cause and effect vectors for the same input text.

**Dataset**: 500 texts spanning 5 categories:
- Explicit causal (100): "Smoking causes lung cancer"
- Implicit causal (100): "After the new policy, unemployment dropped significantly"
- Non-causal factual (100): "Paris is the capital of France"
- Technical/code (100): "The function returns a sorted array"
- Mixed/ambiguous (100): "Temperature rose as CO2 levels increased"

**Metrics**:
| Metric | Target | Current Baseline |
|--------|--------|-----------------|
| Mean cause-effect cosine distance | > 0.30 | ~0.30 (from prior research) |
| Distance for explicit causal | > 0.40 | Unknown |
| Distance for non-causal | < 0.15 | Unknown |
| % vectors with distance < 0.10 (degenerate) | < 5% | Unknown |

**Optimization lever**: Projection matrix perturbation scale (currently `N(0, 0.02)` with seed `0xCA05A1`).

### 1.2 Marker Detection Accuracy

**What**: Evaluate the accuracy of the causal marker detection system in `marker_detection.rs`.

**Dataset**: 1000 sentences with human-annotated cause/effect spans:
- 200 with explicit cause markers ("because", "due to")
- 200 with explicit effect markers ("therefore", "results in")
- 200 with implicit causation (no markers)
- 200 with distractor markers ("because of the rain" in non-causal context)
- 200 non-causal baseline

**Metrics**:
| Metric | Target |
|--------|--------|
| Marker detection precision | > 0.90 |
| Marker detection recall | > 0.85 |
| False positive rate on distractors | < 0.10 |
| Implicit causation detection rate | > 0.40 |

**Optimization lever**: Marker weight boost factor (currently 2.5x), marker vocabulary expansion.

### 1.3 Projection Matrix Effectiveness

**What**: Compare trained vs perturbed-identity projections on retrieval accuracy.

**Experiment**:
1. Baseline: No projection (mean-pooled output only)
2. Current: Perturbed identity (`I + N(0, 0.02)`)
3. Trained: After contrastive training (from `TrainableProjection`)
4. Ablation: Vary perturbation scale {0.01, 0.02, 0.05, 0.10}

**Dataset**: COPA (Choice of Plausible Alternatives) — 500 premise→{correct,incorrect} pairs.

**Metrics**:
| Metric | Target |
|--------|--------|
| COPA accuracy (no projection) | Baseline |
| COPA accuracy (perturbed identity) | > baseline + 5% |
| COPA accuracy (trained) | > baseline + 15% |
| MRR improvement over symmetric E1 | > 10% |

**Optimization lever**: Perturbation scale, training epochs, learning rate.

### 1.4 Instruction Prefix Impact

**What**: Measure whether `CAUSE_INSTRUCTION` / `EFFECT_INSTRUCTION` prefixes improve asymmetry.

**Experiment**:
1. No prefix (raw text)
2. Cause prefix only
3. Effect prefix only
4. Both prefixes (current behavior)

**Metrics**:
| Metric | Target |
|--------|--------|
| Asymmetry ratio (no prefix) | Baseline |
| Asymmetry ratio (both prefixes) | 1.3-1.7 |
| Retrieval accuracy delta | Quantify improvement |

### 1.5 LLM Guidance Enhancement

**What**: Measure accuracy improvement from `CausalHintGuidance` (key phrases from LLM analysis injected into embedding).

**Experiment**:
1. `embed_dual()` — no guidance
2. `embed_dual_guided()` — with LLM-extracted key phrases and confidence

**Dataset**: 200 causal text pairs where the LLM has already analyzed the relationship.

**Metrics**:
| Metric | Target |
|--------|--------|
| MRR without guidance | Baseline |
| MRR with guidance | > baseline + 5% |
| Cause-effect vector distance increase | > 10% |

---

## Phase 2: LLM Causal Discovery Accuracy Benchmarks

**Objective**: Measure the accuracy of Hermes-2-Pro-Mistral-7B in detecting causal relationships, their direction, and mechanism types.

### 2.1 Causal Link Detection (Binary Classification)

**What**: Does the LLM correctly identify whether a text pair contains a causal relationship?

**Dataset**: 1000 text pairs with ground truth labels:
- **SemEval-2010 Task 8** (causal subset) — 400 pairs, labeled cause-effect
- **BECauSE 2.0** — 300 pairs, annotated causal/non-causal
- **Synthetic control** — 300 pairs (topically related but non-causal)

**Metrics**:
| Metric | Target | Current Baseline |
|--------|--------|-----------------|
| Precision | > 0.85 | ~0.75 (estimated from SciFact) |
| Recall | > 0.80 | Unknown |
| F1 Score | > 0.82 | Unknown |
| False positive rate | < 0.15 | Unknown |
| GBNF JSON parse success rate | > 0.98 | ~0.95 |

**Breakdown by pair type** (from `benchmark_causal_large.rs` categories):
| Pair Type | Expected Causal Rate |
|-----------|---------------------|
| Adjacent (same doc, consecutive) | 40-60% |
| SameTopic (different docs) | 15-30% |
| Random (baseline) | 0-5% |

### 2.2 Direction Classification

**What**: When a causal link exists, does the LLM correctly identify A→B vs B→A vs Bidirectional?

**Dataset**: 500 pairs with ground truth direction labels.

**Metrics**:
| Metric | Target |
|--------|--------|
| Direction accuracy (3-class) | > 0.80 |
| A→B precision | > 0.85 |
| B→A precision | > 0.85 |
| Bidirectional detection rate | > 0.60 |
| Confusion matrix: A→B predicted as B→A | < 10% |

**Optimization lever**: Prompt template ordering (cause text first vs effect text first), few-shot examples.

### 2.3 Confidence Calibration

**What**: Are the LLM's confidence scores well-calibrated (i.e., does 80% confidence mean 80% of predictions are correct)?

**Dataset**: All 1000 pairs from 2.1 with LLM confidence scores.

**Metrics**:
| Metric | Target |
|--------|--------|
| Expected Calibration Error (ECE) | < 0.10 |
| Maximum Calibration Error (MCE) | < 0.20 |
| Brier score | < 0.20 |
| Reliability diagram R-squared | > 0.85 |

**Calibration bins** (10 bins, 0.0-1.0):
- For each bin: `avg_confidence` vs `actual_accuracy`
- Plot reliability diagram

**Optimization lever**: Temperature parameter, confidence threshold (`min_confidence` currently 0.7).

### 2.4 Mechanism Type Classification

**What**: Accuracy of mechanism type detection (direct, mediated, feedback, temporal).

**Dataset**: 400 pairs annotated with mechanism type:
- Direct (100): "Heat causes expansion"
- Mediated (100): "Stress → cortisol → immune suppression"
- Feedback (100): "Anxiety causes insomnia which worsens anxiety"
- Temporal (100): "Drought preceded famine which preceded migration"

**Metrics**:
| Metric | Target |
|--------|--------|
| Mechanism accuracy (4-class) | > 0.70 |
| Direct detection F1 | > 0.80 |
| Feedback loop detection F1 | > 0.65 |
| Mediated chain detection F1 | > 0.65 |

### 2.5 Few-Shot vs Zero-Shot Comparison

**What**: Quantify the accuracy/latency tradeoff between prompting modes.

**Experiment**: Run same 500 pairs through both modes on same hardware.

**Metrics**:
| Metric | Few-Shot (5 examples) | Zero-Shot |
|--------|----------------------|-----------|
| Link detection F1 | Target > 0.82 | Target > 0.75 |
| Direction accuracy | Target > 0.80 | Target > 0.72 |
| Avg inference time | Measured | Measured |
| Speedup factor | 1.0x (baseline) | Target > 1.5x |
| Tokens per inference | Measured | Measured |

**Optimization lever**: Number of few-shot examples (1, 3, 5, 7), example selection strategy.

### 2.6 GBNF Grammar Compliance

**What**: How often does the grammar-constrained output produce valid, parseable JSON?

**Dataset**: 1000 LLM invocations.

**Metrics**:
| Metric | Target |
|--------|--------|
| Valid JSON rate | > 0.98 |
| Fallback parse needed | < 0.05 |
| Complete schema compliance | > 0.95 |
| Source span populated rate | > 0.80 |
| Source span offset validity | > 0.90 |
| Text excerpt exact match rate | > 0.70 (pre-fix), > 0.99 (post-fix) |

---

## Phase 3: Retrieval Quality Benchmarks

**Objective**: Measure end-to-end search accuracy for each causal search tool.

### 3.1 search_causes Accuracy (Abductive Reasoning)

**What**: Given an effect, does `search_causes` retrieve the correct causes?

**Dataset**: 200 effect queries with labeled correct causes (ranked by relevance).

**Protocol**:
1. Seed 500 memories spanning 10 domains (medicine, economics, climate, etc.)
2. Store 200 known causal relationships via CausalDiscoveryService
3. Query each effect, measure retrieval of ground-truth causes

**Metrics**:
| Metric | Target |
|--------|--------|
| MRR@10 | > 0.60 |
| Precision@1 | > 0.50 |
| Precision@5 | > 0.40 |
| Recall@10 | > 0.70 |
| NDCG@10 | > 0.55 |

**Scope breakdown**:
- `scope: "memories"` (fingerprint HNSW)
- `scope: "relationships"` (CF_CAUSAL_RELATIONSHIPS)
- `scope: "all"` (merged)

### 3.2 search_effects Accuracy (Predictive Reasoning)

**What**: Given a cause, does `search_effects` retrieve correct effects?

Same protocol as 3.1, reversed direction.

**Metrics**: Same as 3.1 with separate measurements.

**Key comparison**: `search_effects` MRR should be ~1.2x higher than `search_causes` MRR due to direction modifiers (1.2x vs 0.8x).

### 3.3 Asymmetric Retrieval Effectiveness

**What**: Does the asymmetric E5 search outperform symmetric E1-only search?

**Experiment**: Run same queries through:
1. E1-only (symmetric cosine)
2. E5-only (asymmetric, direction-modified)
3. E5 + E1 fusion (current default)
4. Full multi-embedder (E1+E5+E8+E11)

**Metrics**:
| Config | MRR@10 Target | NDCG@10 Target |
|--------|---------------|----------------|
| E1-only | Baseline | Baseline |
| E5-only | > baseline (causal queries) | > baseline |
| E5+E1 fusion | > E1-only + 10% | > E1-only + 10% |
| Full multi-embedder | > E5+E1 + 5% | > E5+E1 + 5% |

**Asymmetry ratio**: `MRR(cause→effect) / MRR(effect→cause)` — target 1.3-1.7 (ideal ~1.5).

### 3.4 Query Intent Detection Accuracy

**What**: Does `detect_causal_query_intent()` correctly classify query direction?

**Dataset**: 500 queries with labeled intent:
- Cause queries (200): "Why does X happen?", "What causes Y?"
- Effect queries (200): "What happens when X?", "What are effects of Y?"
- Neutral queries (100): "Tell me about X", "Describe Y"

**Metrics**:
| Metric | Target |
|--------|--------|
| 3-class accuracy | > 0.90 |
| Cause precision/recall | > 0.92 / > 0.88 |
| Effect precision/recall | > 0.90 / > 0.88 |
| Neutral precision/recall | > 0.85 / > 0.80 |

**Optimization lever**: Indicator vocabulary (currently 100+ cause, 90+ effect patterns).

### 3.5 Causal Chain Traversal

**What**: Does `get_causal_chain` correctly follow multi-hop causal paths?

**Dataset**: 50 known causal chains of length 2-5 hops seeded into the graph.

**Metrics**:
| Metric | Target |
|--------|--------|
| 2-hop chain completion rate | > 0.80 |
| 3-hop chain completion rate | > 0.60 |
| 4-hop chain completion rate | > 0.40 |
| Correct ordering rate | > 0.85 |
| Cumulative strength correlation with ground truth | > 0.70 |
| Hop attenuation calibration (0.9^i) | Measure actual vs theoretical |

### 3.6 Multi-Embedder Fusion Weight Optimization

**What**: Find the optimal fusion weights for causal retrieval.

**Experiment**: Grid search over weight combinations:

```
E1:  [0.15, 0.20, 0.25, 0.30, 0.35]
E5:  [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
E8:  [0.05, 0.10, 0.15, 0.20]
E11: [0.05, 0.10, 0.15, 0.20]
Constraint: E1 + E5 + E8 + E11 = 1.0
```

**Metric**: MRR@10 on the search_causes + search_effects benchmark sets.

**Current weights** (from `causal_relationship_tools.rs`):
```
E1: 0.30, E5: 0.35, E8: 0.15, E11: 0.20
```

**Profile weights** (from `causal_reasoning` profile):
```
E1: 0.25, E5: 0.45, E7: 0.15, E11: 0.15
```

**Prior research recommendation**: E5 at 0.20-0.30 (not 0.45).

**Deliverable**: Optimal weight vector per query type (cause, effect, semantic).

---

## Phase 4: Direction Modifier Calibration

**Objective**: Determine if the 1.2x/0.8x direction modifiers are optimal.

### 4.1 Modifier Sweep

**Experiment**: Sweep cause→effect modifier from 1.0 to 2.0 in 0.1 increments (effect→cause is automatically `2.0 - cause_mod` to maintain product = 0.96).

| cause→effect mod | effect→cause mod | MRR@10 causes | MRR@10 effects | Combined |
|-----------------|-----------------|---------------|----------------|----------|
| 1.0 | 1.0 | Measure | Measure | Measure |
| 1.1 | 0.9 | Measure | Measure | Measure |
| 1.2 | 0.8 | Measure (current) | Measure (current) | Measure |
| 1.3 | 0.7 | Measure | Measure | Measure |
| 1.4 | 0.6 | Measure | Measure | Measure |
| 1.5 | 0.5 | Measure | Measure | Measure |

**Metric**: Maximize combined MRR@10 across both search directions.

### 4.2 Per-Mechanism Modifiers

**What**: Should direction modifiers vary by mechanism type?

**Hypothesis**: Feedback loops need lower asymmetry (bidirectional), direct causation needs higher.

**Experiment**: Train separate modifiers per mechanism type.

| Mechanism | cause→effect | effect→cause | Rationale |
|-----------|-------------|--------------|-----------|
| Direct | 1.3-1.5 | 0.5-0.7 | Strong unidirectional |
| Mediated | 1.2-1.3 | 0.7-0.8 | Moderate unidirectional |
| Feedback | 1.0-1.1 | 0.9-1.0 | Nearly bidirectional |
| Temporal | 1.1-1.2 | 0.8-0.9 | Weak temporal direction |

---

## Phase 5: Training Pipeline Benchmarks

**Objective**: Evaluate the contrastive training system for E5 projection optimization.

### 5.1 Loss Component Ablation

**What**: Measure contribution of each loss component in `DirectionalContrastiveLoss`.

**Experiment**: Train with each component disabled one at a time:

| Configuration | Components | COPA Accuracy | MRR@10 |
|--------------|------------|---------------|--------|
| Full loss | InfoNCE + Directional + Separation + Distillation | Baseline | Baseline |
| No InfoNCE | Directional + Separation + Distillation | Measure | Measure |
| No Directional | InfoNCE + Separation + Distillation | Measure | Measure |
| No Separation | InfoNCE + Directional + Distillation | Measure | Measure |
| No Distillation | InfoNCE + Directional + Separation | Measure | Measure |

**Loss hyperparameters**:
```
InfoNCE:      λ_c = 1.0, τ = 0.05
Directional:  λ_d = 0.3, margin = 0.2
Separation:   λ_s = 0.1
Distillation: λ_soft = 0.2, T = 2.0
```

### 5.2 Training Data Quality

**What**: How does training data quality/quantity affect downstream accuracy?

**Experiment**: Train projections on varying dataset sizes and quality tiers.

| Dataset | Size | Quality | Expected MRR@10 |
|---------|------|---------|-----------------|
| SciFact only | ~5K pairs | High | Moderate |
| SemEval + BECauSE | ~10K pairs | High | Good |
| + Synthetic augmentation | ~50K pairs | Mixed | Better |
| + LLM-generated pairs | ~100K pairs | Variable | Best (if filtered) |

**Quality filtering**: Only include pairs where LLM confidence > 0.85 AND GBNF parse succeeded.

### 5.3 Training Convergence

**What**: How many epochs are needed? Does the model overfit?

**Metrics per epoch**:
- Training loss (total and per-component)
- Validation COPA accuracy
- Validation MRR@10
- Cause-effect vector distance on held-out set
- Direction modifier effectiveness

**Expected behavior**: Convergence at 10-30 epochs; watch for separation loss divergence.

---

## Phase 6: End-to-End Pipeline Benchmarks

**Objective**: Measure accuracy of the complete Discovery → Embedding → Storage → Retrieval pipeline.

### 6.1 Full Pipeline Accuracy

**Dataset**: 100 documents across 10 domains with human-annotated causal relationships.

**Pipeline stages measured**:

| Stage | Metric | Target |
|-------|--------|--------|
| Scanner (candidate selection) | Recall of true causal pairs | > 0.80 |
| LLM (link detection) | F1 | > 0.82 |
| LLM (direction) | Accuracy | > 0.80 |
| LLM (confidence) | ECE | < 0.10 |
| E5 (dual embedding) | Vector distinctness > 0.30 | > 95% |
| Storage (roundtrip) | Lossless | 100% |
| Retrieval (search_causes) | MRR@10 | > 0.60 |
| Retrieval (search_effects) | MRR@10 | > 0.65 |
| Provenance (span accuracy) | Text match rate (post-fix) | > 0.99 |

### 6.2 Domain Transfer

**What**: How well does accuracy hold across different knowledge domains?

**Domains**:
1. Biomedical ("Aspirin inhibits COX-2, reducing inflammation")
2. Economics ("Interest rate hikes reduce inflation")
3. Climate science ("CO2 emissions cause global warming")
4. Software engineering ("Memory leak causes OOM crash")
5. Psychology ("Sleep deprivation impairs cognitive function")
6. Physics ("Heat causes metal expansion")
7. Sociology ("Poverty correlates with crime rates")
8. History ("Treaty of Versailles led to WWII")
9. Nutrition ("Vitamin D deficiency causes rickets")
10. Cybersecurity ("SQL injection enables data exfiltration")

**Per-domain metrics**: Link detection F1, direction accuracy, retrieval MRR@10.

**Target**: Less than 15% variance across domains (no single domain below 0.65 F1).

### 6.3 Adversarial Robustness

**What**: How does the system handle edge cases and adversarial inputs?

**Test cases**:
| Category | Example | Expected Behavior |
|----------|---------|-------------------|
| Correlation vs causation | "Ice cream sales correlate with drowning" | No causal link (or low confidence) |
| Reversed direction | "The fire was caused by the alarm" (nonsensical) | Reject or low confidence |
| Spurious causation | "Wearing a seatbelt causes accidents" | Reject |
| Nested causation | "A causes B which causes C which prevents D" | Correct chain extraction |
| Negated causation | "Smoking does NOT cause cancer" (false claim) | No causal link |
| Hypothetical | "If X were true, Y would follow" | Low confidence |
| Temporal conflation | "After the rooster crowed, the sun rose" | No causal link |

**Metrics**:
| Metric | Target |
|--------|--------|
| Adversarial rejection rate | > 0.80 |
| False positive rate on spurious pairs | < 0.10 |
| Negation detection rate | > 0.70 |

---

## Phase 7: Performance & Latency Benchmarks

**Objective**: Ensure accuracy optimizations don't degrade throughput.

### 7.1 Latency Budget

| Operation | Target | Hard Limit |
|-----------|--------|------------|
| E5 embed_dual() single text | < 20ms | < 50ms |
| E5 embed_dual_guided() | < 25ms | < 75ms |
| LLM inference (single pair) | < 500ms | < 2000ms |
| search_causes (100 relationships) | < 10ms | < 50ms |
| search_causes (5000 relationships) | < 100ms | < 500ms |
| get_causal_chain (3 hops) | < 150ms | < 500ms |
| Full discovery cycle (10 pairs) | < 5s | < 15s |

### 7.2 Throughput Under Load

| Metric | Target |
|--------|--------|
| Concurrent search queries/sec | > 100 |
| Discovery pairs/minute | > 30 |
| E5 embeddings/sec (batch) | > 50 |

### 7.3 Accuracy vs Latency Tradeoff

Plot Pareto frontier for:
- Few-shot examples (0, 1, 3, 5, 7) vs inference time vs F1
- Multi-embedder count (1, 2, 3, 4) vs search time vs MRR
- HNSW ef_search parameter vs recall@10 vs latency

---

## Benchmark Implementation Plan

### Dataset Preparation

```
datasets/
├── semeval2010_task8/         # Causal relation extraction
│   ├── train.jsonl            # 8000 pairs, 1325 causal
│   └── test.jsonl             # 2717 pairs
├── because_2.0/               # Causal connectives corpus
│   ├── causal_pairs.jsonl     # ~3000 annotated pairs
│   └── metadata.json
├── copa/                      # Choice of Plausible Alternatives
│   ├── copa_train.jsonl       # 500 pairs
│   └── copa_test.jsonl        # 500 pairs
├── scifact/                   # Scientific claims (already available)
│   ├── queries.jsonl
│   ├── chunks.jsonl
│   └── qrels.json
├── synthetic/
│   ├── direction_test.jsonl   # 500 pairs with direction labels
│   ├── mechanism_test.jsonl   # 400 pairs with mechanism labels
│   ├── adversarial.jsonl      # 200 edge cases
│   ├── domain_transfer/       # 100 per domain x 10 domains
│   └── marker_detection.jsonl # 1000 annotated sentences
└── causal_chains/
    └── chains.jsonl           # 50 multi-hop chains (2-5 hops)
```

### Benchmark Execution Order

```
Phase 1 (E5 Quality)         ─── No dependencies, run first
  1.1 Vector Distinctness         ~30 min
  1.2 Marker Detection            ~20 min
  1.3 Projection Comparison       ~2 hours (requires training)
  1.4 Instruction Prefix          ~30 min
  1.5 LLM Guidance                ~1 hour

Phase 2 (LLM Accuracy)       ─── Requires Phase 1 baselines
  2.1 Link Detection              ~4 hours (1000 LLM calls)
  2.2 Direction Classification    ~2 hours
  2.3 Confidence Calibration      ~1 hour (post-processing of 2.1)
  2.4 Mechanism Classification    ~2 hours
  2.5 Few-Shot vs Zero-Shot       ~4 hours (duplicate runs)
  2.6 GBNF Compliance             ~1 hour (subset of 2.1)

Phase 3 (Retrieval Quality)  ─── Requires Phase 2 results stored
  3.1 search_causes               ~2 hours
  3.2 search_effects              ~2 hours
  3.3 Asymmetric vs Symmetric     ~3 hours
  3.4 Query Intent Detection      ~30 min
  3.5 Causal Chain Traversal      ~1 hour
  3.6 Fusion Weight Optimization  ~8 hours (grid search)

Phase 4 (Direction Modifiers) ── Requires Phase 3 benchmarks
  4.1 Modifier Sweep              ~4 hours
  4.2 Per-Mechanism Modifiers     ~2 hours

Phase 5 (Training Pipeline)  ─── Can run parallel to Phase 3-4
  5.1 Loss Ablation               ~12 hours (5 training runs)
  5.2 Data Quality                ~8 hours (4 training runs)
  5.3 Convergence                 ~4 hours (1 long run)

Phase 6 (End-to-End)        ─── Requires all prior phases
  6.1 Full Pipeline               ~6 hours
  6.2 Domain Transfer             ~4 hours
  6.3 Adversarial                 ~2 hours

Phase 7 (Performance)       ─── Run throughout, final validation
  7.1 Latency                     ~1 hour
  7.2 Throughput                  ~1 hour
  7.3 Pareto Frontier             ~4 hours
```

### Output Artifacts

All results stored in `benchmark_results/causal_accuracy/`:

```
benchmark_results/causal_accuracy/
├── phase1_e5_quality/
│   ├── vector_distinctness.json
│   ├── marker_detection.json
│   ├── projection_comparison.json
│   ├── instruction_prefix.json
│   └── llm_guidance.json
├── phase2_llm_accuracy/
│   ├── link_detection.json
│   ├── direction_classification.json
│   ├── confidence_calibration.json
│   ├── mechanism_classification.json
│   ├── fewshot_vs_zeroshot.json
│   └── gbnf_compliance.json
├── phase3_retrieval/
│   ├── search_causes.json
│   ├── search_effects.json
│   ├── asymmetric_vs_symmetric.json
│   ├── query_intent.json
│   ├── causal_chains.json
│   └── fusion_weights_grid.json
├── phase4_direction_mods/
│   ├── modifier_sweep.json
│   └── per_mechanism.json
├── phase5_training/
│   ├── loss_ablation.json
│   ├── data_quality.json
│   └── convergence_curves.json
├── phase6_e2e/
│   ├── full_pipeline.json
│   ├── domain_transfer.json
│   └── adversarial.json
├── phase7_performance/
│   ├── latency.json
│   ├── throughput.json
│   └── pareto_frontier.json
└── summary_report.json         # Aggregated results with recommendations
```

---

## Optimization Targets (Deliverables)

Based on benchmark results, the following parameters will be tuned:

### Immediate Tuning (No Retraining Required)

| Parameter | Current | Search Range | File |
|-----------|---------|-------------|------|
| E5 fusion weight (causal_reasoning profile) | 0.45 | 0.20-0.35 | `retrieval/weights.rs` |
| E5 fusion weight (multi-embedder search) | 0.35 | 0.20-0.35 | `causal_relationship_tools.rs` |
| E1 fusion weight | 0.30 | 0.25-0.40 | `causal_relationship_tools.rs` |
| cause→effect direction modifier | 1.2 | 1.0-1.5 | `causal/asymmetric.rs` |
| effect→cause direction modifier | 0.8 | 0.5-1.0 | `causal/asymmetric.rs` |
| LLM min_confidence threshold | 0.7 | 0.6-0.85 | `service/mod.rs` |
| Marker boost factor | 2.5 | 1.5-4.0 | `marker_detection.rs` |
| Source/explanation weights | 0.6/0.4 | sweep | `causal_relationship_tools.rs` |
| Hop attenuation factor | 0.9 | 0.7-0.95 | `causal/chain.rs` |

### Training-Required Tuning

| Parameter | Current | Optimization | File |
|-----------|---------|-------------|------|
| Projection matrices (W_cause, W_effect) | Perturbed identity | Contrastive training | `weights.rs` |
| Perturbation scale | 0.02 | Sweep 0.01-0.10 | `weights.rs` |
| InfoNCE temperature (τ) | 0.05 | Sweep 0.01-0.20 | `training/loss.rs` |
| Directional margin | 0.2 | Sweep 0.1-0.5 | `training/loss.rs` |
| Loss component weights (λ) | 1.0/0.3/0.1/0.2 | Grid search | `training/loss.rs` |

### LLM Optimization

| Parameter | Current | Optimization |
|-----------|---------|-------------|
| Prompt template | Fixed | A/B test 3-5 variants |
| Few-shot example count | 5 | Sweep 0, 1, 3, 5, 7 |
| Few-shot selection strategy | Fixed | Random, diverse, hard-case |
| Temperature | Default | Sweep 0.0-0.8 |
| GBNF grammar strictness | Current | Relax/tighten schema |
| Quantization level | Q4_K_M | Compare Q4/Q5/Q6/Q8 |

---

## Success Criteria

The benchmarking effort is considered successful when:

1. **All Phase 1-6 benchmarks have quantitative results** stored in `benchmark_results/causal_accuracy/`
2. **Optimal fusion weights identified** — backed by grid search data showing statistical significance
3. **Direction modifiers calibrated** — improvement over current 1.2/0.8 demonstrated or current values confirmed
4. **LLM accuracy baselined** — F1, calibration, and per-domain scores documented
5. **Training pipeline validated** — at least one successful training run showing improvement over perturbed identity
6. **End-to-end MRR@10 > 0.60** for both search_causes and search_effects
7. **Domain transfer variance < 15%** across all 10 domains
8. **No performance regression** — all Phase 7 latency targets met

---

## Existing Benchmark Inventory

The following benchmarks already exist and should be integrated (not duplicated):

| Existing Benchmark | File | Integrates Into |
|-------------------|------|-----------------|
| `benchmark_causal.rs` | `context-graph-causal-agent/examples/` | Phase 2.1, 2.2 |
| `benchmark_causal_large.rs` | `context-graph-causal-agent/examples/` | Phase 2.1 (pair types) |
| `benchmark_causal_enhanced.rs` | `context-graph-causal-agent/examples/` | Phase 2.5 |
| `causal_relationships_bench.rs` | `context-graph-storage/benches/` | Phase 7.1 |
| `causal_realdata_bench.rs` | `context-graph-benchmark/src/bin/` | Phase 1.3, 3.3 |
| `causal_provenance_llm_bench.rs` | `context-graph-benchmark/src/bin/` | Phase 2.6 |
| `causal_span_fix_bench.rs` | `context-graph-benchmark/src/bin/` | Phase 2.6 |
| `causal_provenance_e2e_bench.rs` | `context-graph-benchmark/src/bin/` | Phase 6.1 |
| `causal_provenance_bench.rs` | `context-graph-benchmark/src/bin/` | Phase 6.1 |
| `causal_discovery_mcp_bench.rs` | `context-graph-benchmark/src/bin/` | Phase 7.1 |
| `causal.rs` (metrics) | `context-graph-benchmark/src/metrics/` | All phases (metrics framework) |

### Reuse Strategy

- Extend `causal.rs` metrics module with new metric types (ECE, Brier, NDCG)
- Augment existing benchmark binaries with new datasets rather than creating parallel ones
- Use `ResultsAggregator` from `context-graph-benchmark/src/validation/` for unified reporting
- Write new benchmarks only for gaps: Phases 1.1, 1.2, 3.4, 3.5, 3.6, 4.1, 4.2, 5.x, 6.2, 6.3
