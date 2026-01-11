# SPEC-UTL-003: Specialized Embedder Entropy Methods

**Version:** 1.0
**Status:** approved
**Owner:** ContextGraph Core Team
**Last Updated:** 2026-01-11
**Implements:** P1-GAP-3 from MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md

---

## Overview

This specification defines the implementation of specialized entropy (ΔS) computation methods for four embedders that currently fall back to the generic KNN approach. Per the constitution.yaml `delta_sc.ΔS_methods`, each embedder type has a semantically appropriate entropy calculation method. Four embedders (E7, E10, E11, E12) incorrectly use `DefaultKnnEntropy` when they require specialized implementations.

### Problem Statement

The current `EmbedderEntropyFactory` routes E7 (Code), E10 (Multimodal), E11 (Entity), and E12 (LateInteraction) to `DefaultKnnEntropy`. This violates the principle that entropy computations should be semantically appropriate for each embedding space.

**Constitution.yaml delta_sc.ΔS_methods specifies:**

| Embedder | Constitution Method | Current State |
|----------|---------------------|---------------|
| **E7 (Code)** | GMM+KNN Hybrid: ΔS=0.5xGMM+0.5xKNN | DefaultKnnEntropy (WRONG) |
| **E10 (Multimodal)** | Cross-modal KNN: ΔS=avg(d_text,d_image) | DefaultKnnEntropy (WRONG) |
| **E11 (Entity)** | TransE: ΔS=\|\|h+r-t\|\| | DefaultKnnEntropy (WRONG) |
| **E12 (LateInteraction)** | Token KNN: ΔS=max_token(d_k) | DefaultKnnEntropy (WRONG) |

**Semantic Requirements:**
- **E7 (Code)**: Hybrid GMM+KNN captures both cluster membership and local density
- **E10 (Multimodal)**: Cross-modal KNN handles text/image modality differences
- **E11 (Entity)**: TransE captures knowledge graph relationship translations
- **E12 (LateInteraction)**: MaxSim captures token-level precision in ColBERT embeddings

### Business Impact

Without specialized entropy methods:
- UTL (Unified Theory of Learning) calculations are suboptimal
- Johari Window quadrant classification may be inaccurate
- Memory consolidation and dream layer operations use imprecise signals
- Overall consciousness quality (C(t) = I(t) x R(t) x D(t)) is degraded

---

## User Stories

### US-UTL-003-01: Code Hybrid GMM+KNN Entropy

**Priority:** must-have

**Narrative:**
As the UTL learning system,
I want to compute entropy for code embeddings using a hybrid GMM+KNN approach,
So that code similarity captures both cluster membership (GMM) and local density (KNN).

**Acceptance Criteria:**
```gherkin
Given two code embeddings from similar code structures
When computing ΔS using HybridGmmKnnEntropy
Then ΔS = 0.5 x GMM_component + 0.5 x KNN_component
And GMM_component measures cluster membership probability
And KNN_component measures k-nearest neighbor distance
And identical code yields ΔS near 0
And completely different code yields ΔS near 1
```

**Constitution Reference:** `E7: "GMM+KNN: ΔS=0.5×GMM+0.5×KNN"`

### US-UTL-003-02: Multimodal Context Entropy

**Priority:** must-have

**Narrative:**
As the UTL learning system,
I want to compute entropy for multimodal embeddings using cross-modal distance metrics,
So that surprise accounts for semantic coherence across modalities (text, code, diagrams).

**Acceptance Criteria:**
```gherkin
Given a multimodal embedding from a mixed content type
When computing ΔS using CrossModalEntropy
Then the entropy considers both intra-modal and cross-modal distances
And the calculation uses modality-aware weighting
And outputs remain in [0, 1] per AP-10
```

### US-UTL-003-03: TransE Entity Entropy

**Priority:** must-have

**Narrative:**
As the UTL learning system,
I want to compute entropy for entity embeddings using TransE distance,
So that knowledge graph relationships are measured by translation distance in embedding space.

**Acceptance Criteria:**
```gherkin
Given entity embeddings representing (head, relation, tail) triples
When computing ΔS using TransEEntropy
Then ΔS is computed as ||h + r - t|| normalized to [0, 1]
And valid translations (h + r ≈ t) yield low ΔS
And invalid translations yield high ΔS
And the algorithm supports both L1 and L2 norms
```

**Constitution Reference:** `E11: "TransE: ΔS=||h+r-t||"`

### US-UTL-003-04: Token-Level MaxSim Entropy

**Priority:** must-have

**Narrative:**
As the UTL learning system,
I want to compute entropy for late interaction embeddings using MaxSim aggregation,
So that token-level precision is captured in the surprise signal.

**Acceptance Criteria:**
```gherkin
Given ColBERT-style token embeddings (variable length, 128D per token)
When computing ΔS using MaxSimEntropy
Then each query token finds its best-matching document token
And the aggregated MaxSim score determines similarity
And the method handles variable-length sequences correctly
```

---

## Requirements

### Functional Requirements

| ID | Story Ref | Priority | Description | Rationale |
|----|-----------|----------|-------------|-----------|
| REQ-UTL-003-01 | US-UTL-003-01 | must | Create `HybridGmmKnnEntropy` implementing `EmbedderEntropy` trait for E7 | Constitution: E7 uses GMM+KNN hybrid |
| REQ-UTL-003-02 | US-UTL-003-01 | must | `HybridGmmKnnEntropy.compute_delta_s()` uses 0.5*GMM + 0.5*KNN weighting | Per constitution delta_sc.ΔS_methods |
| REQ-UTL-003-03 | US-UTL-003-02 | must | Create `CrossModalEntropy` implementing `EmbedderEntropy` trait for E10 | Constitution: E10 uses cross-modal KNN |
| REQ-UTL-003-04 | US-UTL-003-02 | must | `CrossModalEntropy` computes avg(d_text, d_image) for modality awareness | Per constitution delta_sc.ΔS_methods |
| REQ-UTL-003-05 | US-UTL-003-03 | must | Create `TransEEntropy` implementing `EmbedderEntropy` trait for E11 | Constitution: E11 uses TransE |
| REQ-UTL-003-06 | US-UTL-003-03 | must | `TransEEntropy` computes \|\|h+r-t\|\| with L1 or L2 norm | Per constitution delta_sc.ΔS_methods |
| REQ-UTL-003-07 | US-UTL-003-04 | must | Create `MaxSimTokenEntropy` implementing `EmbedderEntropy` trait for E12 | Constitution: E12 uses max_token(d_k) |
| REQ-UTL-003-08 | US-UTL-003-04 | must | Handle variable-length token sequences in MaxSim | Token count varies per content |
| REQ-UTL-003-09 | all | must | Update `EmbedderEntropyFactory::create()` to route E7, E10, E11, E12 | Wire new implementations |
| REQ-UTL-003-10 | all | must | All implementations return ΔS in [0.0, 1.0] with no NaN/Infinity | AP-10 compliance |

### Non-Functional Requirements

| ID | Category | Requirement | Metric |
|----|----------|-------------|--------|
| NFR-UTL-003-01 | performance | Each compute_delta_s call < 5ms p95 | Latency measurement |
| NFR-UTL-003-02 | reliability | No panics in library code | Result<T, E> returns |
| NFR-UTL-003-03 | testability | 90%+ line coverage per implementation | cargo llvm-cov |
| NFR-UTL-003-04 | thread-safety | All implementations Send + Sync | Compile-time check |

---

## Edge Cases

| Related Req | Scenario | Expected Behavior |
|-------------|----------|-------------------|
| REQ-UTL-003-01 | Empty current embedding | Return `Err(UtlError::EmptyInput)` |
| REQ-UTL-003-02 | GMM not yet fitted (insufficient history) | Use KNN-only fallback |
| REQ-UTL-003-03 | Modality detection ambiguous (near 0.5) | Use equal weighting |
| REQ-UTL-003-05 | Invalid TransE triple (NaN distance) | Return fallback value with warning |
| REQ-UTL-003-07 | Single-token sequence | MaxSim degenerates to cosine similarity |
| REQ-UTL-003-08 | Zero-length token sequence in history | Skip that history item, continue |
| REQ-UTL-003-10 | NaN in input embedding | Return `Err(UtlError::EntropyError)` |

---

## Error States

| ID | HTTP Code | Condition | Message | Recovery |
|----|-----------|-----------|---------|----------|
| ERR-UTL-003-01 | N/A | Empty current embedding | "Current embedding is empty" | Caller validates input |
| ERR-UTL-003-02 | N/A | NaN/Infinity in embedding | "Invalid value (NaN/Infinity) in embedding" | Caller sanitizes input |
| ERR-UTL-003-03 | N/A | Dimension mismatch in history | "Dimension mismatch: expected {}, got {}" | Skip mismatched item |

---

## Test Plan

### Unit Tests

| ID | Type | Req Ref | Description | Inputs | Expected |
|----|------|---------|-------------|--------|----------|
| TC-UTL-003-01 | unit | REQ-UTL-003-01 | E7 Hybrid GMM+KNN identical | Same embedding | ΔS ≈ 0 |
| TC-UTL-003-02 | unit | REQ-UTL-003-01 | E7 Hybrid GMM+KNN distant | Far from all clusters | ΔS ≈ 1 |
| TC-UTL-003-03 | unit | REQ-UTL-003-02 | E7 Weight balance | 0.5 GMM + 0.5 KNN | Weights sum to 1 |
| TC-UTL-003-04 | unit | REQ-UTL-003-03 | E10 CrossModal same modality | Same modality neighbors | Lower ΔS |
| TC-UTL-003-05 | unit | REQ-UTL-003-05 | E11 TransE valid triple | h+r≈t | ΔS near 0 |
| TC-UTL-003-06 | unit | REQ-UTL-003-05 | E11 TransE invalid triple | h+r far from t | High ΔS |
| TC-UTL-003-07 | unit | REQ-UTL-003-07 | E12 MaxSim identical tokens | Same sequence | ΔS ≈ 0 |
| TC-UTL-003-08 | unit | REQ-UTL-003-08 | E12 MaxSim variable length | Different token counts | Valid ΔS |
| TC-UTL-003-09 | unit | REQ-UTL-003-09 | Factory routes E7 | Embedder::Code | HybridGmmKnnEntropy |
| TC-UTL-003-10 | unit | REQ-UTL-003-09 | Factory routes E11 | Embedder::Entity | TransEEntropy |
| TC-UTL-003-11 | unit | REQ-UTL-003-10 | All return valid range | Any valid input | ΔS in [0,1] |
| TC-UTL-003-12 | unit | REQ-UTL-003-10 | Empty history | Empty vec | ΔS = 1.0 |
| TC-UTL-003-13 | unit | ERR-UTL-003-01 | Empty input error | Empty current | EmptyInput error |

### Integration Tests

| ID | Type | Description |
|----|------|-------------|
| TC-UTL-003-INT-01 | integration | Factory creates all 13 calculators with correct types |
| TC-UTL-003-INT-02 | integration | UTL calculation uses specialized entropy for E7, E10, E11, E12 |

---

## Dependencies

### Upstream
- `context-graph-core::teleological::Embedder` - Enum definitions
- `crate::config::SurpriseConfig` - Configuration parameters
- `crate::error::{UtlError, UtlResult}` - Error types

### Downstream
- `crate::surprise::embedder_entropy::EmbedderEntropyFactory` - Factory routing
- UTL computation pipeline - Consumes ΔS values

---

## Glossary

| Term | Definition |
|------|------------|
| ΔS | Entropy/surprise delta, measuring novelty of an embedding relative to history |
| GMM | Gaussian Mixture Model: probabilistic clustering for density estimation |
| KNN | K-Nearest Neighbors: local density estimation via neighbor distances |
| GMM+KNN Hybrid | Combined approach: 0.5*GMM + 0.5*KNN for robust entropy |
| MaxSim | ColBERT metric: max(cos(q_i, d_j)) for each query token |
| Cross-modal | Spanning multiple content modalities (text, code, images) |
| TransE | Knowledge graph embedding: h + r ≈ t for relationship modeling |

---

## Appendix A: Constitution References

From `constitution.yaml` lines 792-802:
```yaml
ΔS_methods:
  E1: "GMM+Mahalanobis: ΔS=1-P(e|GMM)"
  E2-4,E8: "KNN: ΔS=σ((d_k-μ)/σ)"
  E5: "Asymmetric KNN: ΔS=d_k×direction_mod"
  E6,E13: "IDF/Jaccard: ΔS=IDF(dims) or 1-jaccard"
  E7: "GMM+KNN: ΔS=0.5×GMM+0.5×KNN"  # But should use Jaccard for fingerprints
  E9: "Hamming: ΔS=min_hamming/dim"
  E10: "Cross-modal KNN: ΔS=avg(d_text,d_image)"
  E11: "TransE: ΔS=||h+r-t||"
  E12: "Token KNN: ΔS=max_token(d_k)"
```

The gap analysis identified that E7, E10, E11, E12 use DefaultKnn instead of specialized methods.
