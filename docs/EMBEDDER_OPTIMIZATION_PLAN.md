# Embedder Optimization Plan

> **Status:** Research Complete | **Date:** 2026-01-20 | **Version:** 1.0

## Executive Summary

Benchmark results show that most embedders have 0 ablation delta - meaning E1 semantic is doing most of the heavy lifting. Only E12 Late Interaction showed unique contribution (+0.17). This plan outlines research-backed optimizations to extract meaningful information from each embedder.

## Benchmark Results (Baseline)

| Embedder | MRR | Success | Ablation Δ | Issue |
|----------|-----|---------|------------|-------|
| E6 Sparse | 1.00 | 100% | 0.00 | No unique contribution vs E1 |
| E10 Multimodal | 1.00 | 100% | 0.00 | No unique contribution vs E1 |
| E11 Entity | 1.00 | 100% | 0.00 | No unique contribution vs E1 |
| E13 SPLADE | 1.00 | 100% | 0.00 | No unique contribution vs E1 |
| **E12 Late** | 0.83 | 67% | **0.17** | Only embedder with unique value |
| E9 HDC | 0.63 | 50% | 0.00 | Moderate performance |
| E7 Code | 0.58 | 25% | 0.00 | Prefers NL over code syntax |
| E5 Causal | 0.50 | 0% | 0.00 | Not discriminating causal direction |

---

## Per-Embedder Optimization Plans

### E5 Causal (768D) - CRITICAL FIX NEEDED

**Current Problem:** All similarity scores within 0.01 of each other. Not distinguishing causal direction.

**Root Cause:** Using symmetric similarity (cosine) when causal relationships are inherently ASYMMETRIC.

**Research Finding:** [CAWAI: Causal Retrieval with Semantic Consideration](https://arxiv.org/html/2504.04700v1) shows that causal retrieval requires:
- **Dual Encoders**: Separate Cause Encoder and Effect Encoder
- **Asymmetric Training**: Map cause-effect pairs closely, not symmetric similarity
- **Direction Matters**: "A causes B" ≠ "B causes A"

**Proposed Changes:**

```rust
// Current (WRONG): Symmetric cosine similarity
fn e5_similarity(query: &[f32], doc: &[f32]) -> f32 {
    cosine_similarity(query, doc)  // Treats cause and effect the same
}

// Proposed (CORRECT): Asymmetric similarity
struct CausalEmbedding {
    cause_vector: Vec<f32>,   // Encode as potential cause
    effect_vector: Vec<f32>,  // Encode as potential effect
}

fn e5_similarity(query: &CausalEmbedding, doc: &CausalEmbedding) -> f32 {
    // Query as cause → Document as effect
    let cause_to_effect = cosine_similarity(&query.cause_vector, &doc.effect_vector);
    // Or query as effect → Document as cause (depending on query intent)
    let effect_to_cause = cosine_similarity(&query.effect_vector, &doc.cause_vector);

    // Return based on query type ("why" → cause_to_effect, "what happens" → effect_to_cause)
    cause_to_effect
}
```

**Constitution Update Required:**
```yaml
# New rule
ARCH-15: "E5 Causal MUST use asymmetric similarity with separate cause/effect encodings"
```

**Implementation Phases:**
1. Update `SemanticFingerprint` to store `e5_cause` and `e5_effect` vectors
2. Update embedding pipeline to generate both vectors
3. Implement query-type detection ("why" vs "what happens")
4. Update search to use asymmetric similarity

---

### E7 Code (1536D) - FIX NEEDED

**Current Problem:** Prefers natural language descriptions over actual code syntax.

**Root Cause:** Using generic code embeddings that don't distinguish code from NL descriptions.

**Research Finding:** [LoRACode](https://arxiv.org/html/2503.05315v1) and [CodeCSE](https://arxiv.org/html/2407.06360v1) show:
- **Separate Adapters**: Text2Code adapters vs Code2Code adapters
- **Contrastive Learning**: Train to distinguish code from NL
- **AST Awareness**: Use code structure, not just tokens

**Proposed Changes:**

```rust
// Current (WRONG): Single embedding for both code and NL queries
fn e7_embed(text: &str) -> Vec<f32> {
    code_encoder.encode(text)  // Same treatment for code and NL
}

// Proposed (CORRECT): Query-aware embedding
enum QueryType {
    Code,     // "fn process_batch<T>" - actual code
    NLToCode, // "batch processing function" - NL query about code
}

fn e7_embed(text: &str, query_type: QueryType) -> Vec<f32> {
    match query_type {
        QueryType::Code => code_code_encoder.encode(text),
        QueryType::NLToCode => text_code_encoder.encode(text),
    }
}
```

**Query Type Detection Heuristics:**
```rust
fn detect_code_query_type(query: &str) -> QueryType {
    let code_indicators = [
        "fn ", "struct ", "impl ", "async fn", "pub fn",
        "::", "->", "Vec<", "Result<", "Option<"
    ];

    if code_indicators.iter().any(|i| query.contains(i)) {
        QueryType::Code
    } else {
        QueryType::NLToCode
    }
}
```

**Constitution Update Required:**
```yaml
# New rule
ARCH-16: "E7 Code MUST detect query type and use appropriate encoder (Code2Code vs Text2Code)"
```

---

### E12 Late Interaction (128D/token) - OPTIMIZE USAGE

**Current Status:** Only embedder showing unique contribution (+0.17 ablation delta).

**Research Finding:** [ColBERT Best Practices](https://sease.io/2025/11/colbert-in-practice-bridging-research-and-industry.html):
- **Use for Re-ranking ONLY**: Not initial retrieval (per AP-74)
- **Token Pruning**: Aggressive pruning for efficiency
- **MaxSim Scoring**: Sum of max similarities across query tokens

**Current Constitution:** AP-74 already states "E12 ColBERT MUST only be used for re-ranking"

**Verify Implementation:**
```rust
// E12 should ONLY be used in Stage 3 (re-ranking)
// NOT in Stage 1 (recall) or Stage 2 (scoring)

// CORRECT usage:
async fn pipeline_search(...) {
    // Stage 1: Recall (E13 SPLADE + E1)
    let candidates = stage1_recall(query, 100);

    // Stage 2: Multi-space scoring (E1, E5, E7, E10)
    let scored = stage2_score(query, candidates, 50);

    // Stage 3: E12 ColBERT re-ranking (ONLY HERE)
    let final_results = stage3_rerank_with_e12(query, scored, 10);
}
```

**No Constitution Change Needed** - AP-74 is correct.

---

### E13 SPLADE (~30K sparse) - OPTIMIZE USAGE

**Current Status:** 100% success on stress tests, but 0 ablation delta.

**Research Finding:** [SPLADE Best Practices](https://www.pinecone.io/learn/splade/):
- **Use for Recall ONLY**: Stage 1 candidate generation
- **Term Expansion**: Learns implicit term relationships
- **NOT for Final Ranking**: Use denser embeddings for precision

**Current Constitution:** AP-75 states "E13 SPLADE MUST be used for Stage 1 recall, NOT final ranking"

**Proposed Enhancement:**
```rust
// Current weight profiles may include E13 in scoring
// Fix: Ensure E13 weight is 0 in scoring profiles

pub const WEIGHT_PROFILES: &[(&str, [f32; 13])] = &[
    ("semantic_search", [
        0.35,  // E1 Semantic
        0.0, 0.0, 0.0,  // E2-E4 Temporal (excluded)
        0.15,  // E5 Causal
        0.05,  // E6 Sparse (keyword backup)
        0.20,  // E7 Code
        0.05,  // E8 Graph
        0.0,   // E9 HDC
        0.15,  // E10 Multimodal
        0.05,  // E11 Entity
        0.0,   // E12 Late Interaction (rerank only)
        0.0,   // E13 SPLADE (recall only) <-- MUST BE 0 IN SCORING
    ]),
];
```

**Constitution Update Required:**
```yaml
# Update existing rule
ARCH-13: "Weight profiles MUST have E2-E4=0.0 (temporal), E12=0.0 (rerank-only), E13=0.0 (recall-only) for semantic search"
```

---

### E11 Entity (384D) - ENHANCE

**Current Status:** 100% success, but 0 ablation delta.

**Root Cause:** Using embedding similarity when entity matching requires LINKING.

**Research Finding:** [Entity Linking with Graph Embeddings](https://arxiv.org/html/2506.03895v1):
- **Entity Disambiguation**: Same name may refer to different entities
- **Knowledge Graph Linking**: Connect to external KG (Wikipedia, Wikidata)
- **Graph Embeddings**: TransE style ||h+r-t|| scoring

**Proposed Changes:**

```rust
// Current (LIMITED): Just embedding similarity
fn e11_similarity(query: &[f32], doc: &[f32]) -> f32 {
    cosine_similarity(query, doc)
}

// Proposed (ENHANCED): Entity linking + disambiguation
struct EntityMatch {
    entity_id: String,           // Wikidata/internal ID
    canonical_name: String,      // "PostgreSQL" not "postgres"
    entity_type: EntityType,     // Person, Organization, Software, etc.
    confidence: f32,
}

fn e11_match(query_text: &str, doc_entities: &[EntityMatch]) -> f32 {
    // 1. Extract entities from query
    let query_entities = extract_entities(query_text);

    // 2. Link to knowledge graph
    let linked_query = link_entities(&query_entities);

    // 3. Compare entity IDs, not just text
    let overlap = compute_entity_overlap(&linked_query, doc_entities);

    overlap
}
```

**Constitution Update Required:**
```yaml
# New rule
ARCH-17: "E11 Entity SHOULD use entity linking for disambiguation, not just embedding similarity"
```

---

### E9 HDC (1024D binary) - VERIFY ENCODING

**Current Status:** 50% success, moderate performance.

**Research Finding:** [Hyperdimensional Computing](https://pmc.ncbi.nlm.nih.gov/articles/PMC11214273/):
- **High Dimensionality**: 10K+ dimensions for robustness
- **Binary/Bipolar Encoding**: {0,1} or {-1,+1}
- **Holographic**: Information distributed across entire vector
- **Noise Robustness**: Main advantage over dense embeddings

**Verify Implementation:**
```rust
// HDC should use Hamming distance, not cosine
fn e9_similarity(query: &[u8], doc: &[u8]) -> f32 {
    // Count matching bits
    let hamming_distance: usize = query.iter()
        .zip(doc.iter())
        .map(|(a, b)| (a ^ b).count_ones() as usize)
        .sum();

    // Convert to similarity [0, 1]
    1.0 - (hamming_distance as f32 / (query.len() * 8) as f32)
}
```

**Current Constitution:** UTL already specifies `E9: "Hamming"` for Delta_S method.

**No Constitution Change Needed** - Verify implementation uses Hamming distance.

---

### E10 Multimodal (768D) - ENHANCE

**Current Status:** 100% success, but 0 ablation delta.

**Research Finding:** [CLIP and Multimodal Embeddings](https://medium.com/data-science/clip-model-and-the-importance-of-multimodal-embeddings-1c8f6b13bf72):
- **Modality Gap**: Text and image embeddings cluster separately
- **Cross-Modal Alignment**: Need explicit alignment training
- **Use for Intent**: Not just visual descriptions

**Current Role:** Used for "intent" detection per constitution.

**Proposed Enhancement:**
```rust
// E10 should focus on INTENT, not just visual descriptions
// Examples of intent queries:
// - "I want to add a feature" → intent: ADD
// - "Fix this bug" → intent: FIX
// - "Explain how this works" → intent: UNDERSTAND

fn e10_extract_intent(text: &str) -> IntentVector {
    // Use multimodal encoder trained on intent classification
    multimodal_encoder.encode_with_intent(text)
}
```

**Constitution Update Consideration:**
The current role as "V_multimodality" for "Intent" is appropriate. Consider adding:
```yaml
# Clarification
E10_usage: "E10 Multimodal captures user INTENT (add, fix, explain, refactor) not visual content in text-only mode"
```

---

### E6 Sparse (~30K sparse) - OPTIMIZE

**Current Status:** 100% success on keyword queries.

**Role:** Exact keyword matching for rare technical terms.

**No Changes Needed:** E6 performs well on its intended purpose (exact keyword matching).

**Best Practice:** Use in Stage 1 alongside E13 SPLADE for recall.

---

### Temporal Embedders (E2, E3, E4) - CORRECT USAGE

**Current Constitution:** Already correct - temporal excluded from topics (weight 0.0).

**Research Finding:** [Temporal Embeddings](https://arxiv.org/html/2509.19376):
- Temporal proximity ≠ semantic similarity
- Use exponential decay functions: `score = semantic_sim * exp(-λ * time_diff)`
- Apply as POST-retrieval boost, not similarity measure

**Current Constitution:** ARCH-14 states "Recency boost is applied POST-retrieval"

**No Constitution Change Needed** - Already correct.

---

## Multi-Embedder Fusion Strategy

### Current Issue
All embedders use simple weighted sum, which dilutes unique signals.

### Research-Backed Solution

**Rank Fusion Instead of Score Fusion:**
Based on [Elastic's Weighted RRF](https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf):

```rust
// Current (LIMITED): Weighted score sum
let score = weights.iter()
    .zip(similarities.iter())
    .map(|(w, s)| w * s)
    .sum();

// Proposed (BETTER): Weighted Reciprocal Rank Fusion
fn weighted_rrf(rankings: &[(EmbedderIndex, Vec<(DocId, usize)>)], k: usize) -> Vec<DocId> {
    let mut doc_scores: HashMap<DocId, f32> = HashMap::new();

    for (embedder, ranked_docs) in rankings {
        let weight = get_embedder_weight(embedder);
        for (doc_id, rank) in ranked_docs {
            let rrf_score = weight / (rank as f32 + 60.0); // k=60 constant
            *doc_scores.entry(*doc_id).or_insert(0.0) += rrf_score;
        }
    }

    // Sort by combined RRF score
    let mut results: Vec<_> = doc_scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.into_iter().map(|(id, _)| id).collect()
}
```

### Constitution Update Required:
```yaml
# New rule
ARCH-18: "Multi-space fusion SHOULD use Weighted RRF, not weighted score sum"
```

---

## Proposed Pipeline Architecture

Based on research, optimal pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: RECALL                               │
│  E13 SPLADE (term expansion) + E6 Sparse (keywords) + E1 (semantic) │
│  Output: 100 candidates                                              │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: MULTI-SPACE SCORING                      │
│  Weighted RRF across:                                                │
│  - E1 Semantic (0.35)                                                │
│  - E5 Causal (0.15) - asymmetric                                     │
│  - E7 Code (0.20) - query-aware                                      │
│  - E10 Multimodal (0.15) - intent                                    │
│  - E11 Entity (0.10) - linked                                        │
│  - E8 Graph (0.05) - structural                                      │
│  Output: 20 candidates                                               │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     STAGE 3: PRECISION RE-RANK                       │
│  E12 Late Interaction (ColBERT MaxSim)                               │
│  Output: 10 final results                                            │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: POST-RETRIEVAL BOOST                     │
│  E2 Temporal (recency decay)                                         │
│  E3 Periodic (cyclic patterns)                                       │
│  E4 Positional (sequence)                                            │
│  E9 HDC (noise-robust backup verification)                           │
│  Output: Final ranked results with temporal metadata                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Constitution Updates Summary

### New Rules Required

| Rule | Description |
|------|-------------|
| ARCH-15 | E5 Causal MUST use asymmetric similarity with separate cause/effect encodings |
| ARCH-16 | E7 Code MUST detect query type and use appropriate encoder (Code2Code vs Text2Code) |
| ARCH-17 | E11 Entity SHOULD use entity linking for disambiguation |
| ARCH-18 | Multi-space fusion SHOULD use Weighted RRF, not weighted score sum |

### Updated Rules

| Rule | Current | Proposed |
|------|---------|----------|
| ARCH-13 | E2-E4=0.0 | E2-E4=0.0, E12=0.0, E13=0.0 for scoring profiles |

### Rules Already Correct

| Rule | Description |
|------|-------------|
| ARCH-14 | Recency boost POST-retrieval |
| AP-73 | Temporal not in similarity fusion |
| AP-74 | E12 for reranking only |
| AP-75 | E13 for recall only |

---

## Implementation Priority

### Phase 1: Critical Fixes (E5, E7)
1. **E5 Causal Asymmetric Encoding** - Highest impact, currently 0% success
2. **E7 Code Query-Type Detection** - Currently preferring NL over code

### Phase 2: Pipeline Optimization
3. **Weighted RRF Fusion** - Replace weighted score sum
4. **Verify E12/E13 Stage Usage** - Ensure correct pipeline positions

### Phase 3: Enhancements
5. **E11 Entity Linking** - Disambiguation
6. **E9 HDC Hamming Verification** - Ensure binary distance

### Phase 4: Validation
7. **Re-run Stress Tests** - Verify improvements
8. **Measure Ablation Deltas** - Each embedder should contribute uniquely

---

## Sources

- [CAWAI: Causal Retrieval with Semantic Consideration](https://arxiv.org/html/2504.04700v1)
- [ColBERT in Practice](https://sease.io/2025/11/colbert-in-practice-bridging-research-and-industry.html)
- [SPLADE Explained - Pinecone](https://www.pinecone.io/learn/splade/)
- [Weighted RRF - Elasticsearch](https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf)
- [LoRACode: LoRA Adapters for Code](https://arxiv.org/html/2503.05315v1)
- [Entity Linking with Graph Embeddings](https://arxiv.org/html/2506.03895v1)
- [HDC for Biomedical Sciences](https://pmc.ncbi.nlm.nih.gov/articles/PMC12192801/)
- [CLIP Multimodal Embeddings](https://medium.com/data-science/clip-model-and-the-importance-of-multimodal-embeddings-1c8f6b13bf72)
- [Temporal Embeddings for RAG](https://arxiv.org/html/2509.19376)
