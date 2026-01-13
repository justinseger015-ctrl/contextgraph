# Sherlock Holmes Forensic Investigation Report

## UTL (Unified Theory of Learning) and Delta-S/Delta-C Computation

**Case ID**: UTL-INVESTIGATION-2026-01-12
**Date**: 2026-01-12
**Investigator**: Sherlock Holmes Agent #1
**Subject**: UTL Learning System Implementation Compliance

---

## Executive Summary

*HOLMES: \*steeples fingers\* After exhaustive forensic examination of the codebase, I present my findings on the UTL implementation.*

The UTL (Unified Theory of Learning) system implementing the formula `L = f((Delta-S x Delta-C) . w_e . cos phi)` has been **substantially implemented** with the following compliance status:

| Requirement | Status | Verdict |
|-------------|--------|---------|
| UTL-001: compute_delta_sc MCP tool | **EXISTS** | INNOCENT |
| UTL-002: Delta-C formula (0.4xConn + 0.4xClusterFit + 0.2xConsistency) | **CORRECT** | INNOCENT |
| UTL-003: Per-embedder Delta-S methods | **CORRECT** | INNOCENT |
| UTL-004: Multi-embedding aggregation | **EXISTS** | INNOCENT |
| UTL-005: Johari->action mapping | **DISCREPANCY FOUND** | SUSPICIOUS |
| AP-33: ClusterFit inclusion | **INCLUDED** | INNOCENT |

**Overall Verdict**: The system is largely INNOCENT with one MEDIUM severity issue in the Johari-to-action mapping.

---

## Evidence Log

### 1. UTL-001: compute_delta_sc MCP Tool Verification

**Source of Truth**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs:30`
```rust
/// TASK-UTL-P1-001: Compute per-embedder delta-S and aggregate delta-C
pub const COMPUTE_DELTA_SC: &str = "gwt/compute_delta_sc";
```

**Dispatch Verification**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/dispatch.rs:84-86`
```rust
// TASK-UTL-P1-001: UTL delta S/C computation
tool_names::COMPUTE_DELTA_SC => {
    self.handle_gwt_compute_delta_sc(id, Some(arguments)).await
}
```

**Verdict**: INNOCENT - The MCP tool exists and is properly dispatched.

---

### 2. UTL-002: Delta-C Formula Verification

**Constitution Requirement**: `Delta-C = 0.4 x Connectivity + 0.4 x ClusterFit + 0.2 x Consistency`

**Source of Truth**: `/home/cabdru/contextgraph/crates/context-graph-utl/src/coherence/tracker.rs:161-167`
```rust
/// Weight for connectivity component (alpha). Default: 0.4
connectivity_weight: f32,

/// Weight for cluster fit component (beta). Default: 0.4
cluster_fit_weight: f32,

/// Weight for consistency component (gamma). Default: 0.2
consistency_weight: f32,
```

**Formula Implementation**: `/home/cabdru/contextgraph/crates/context-graph-utl/src/coherence/tracker.rs:316-318`
```rust
// 4. Apply three-component formula per constitution.yaml line 166:
// Delta-C = alpha x Connectivity + beta x ClusterFit + gamma x Consistency
let coherence = self.connectivity_weight * connectivity
    + self.cluster_fit_weight * cluster_fit
    + self.consistency_weight * consistency;
```

**Constants Verification**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/utl/constants.rs`
- ALPHA = 0.4 (Connectivity)
- BETA = 0.4 (ClusterFit)
- GAMMA = 0.2 (Consistency)

**Verdict**: INNOCENT - The formula is correctly implemented per constitution.

---

### 3. AP-33: ClusterFit Inclusion (CRITICAL)

**Anti-Pattern Requirement**: "Delta-C MUST include ClusterFit (FORBIDDEN to omit)"

**Evidence**: ClusterFit is computed via silhouette coefficient in:
- `/home/cabdru/contextgraph/crates/context-graph-utl/src/coherence/cluster_fit/mod.rs`
- `/home/cabdru/contextgraph/crates/context-graph-utl/src/coherence/cluster_fit/compute.rs`

**Usage in Delta-C**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/utl/gwt_compute.rs:163-164`
```rust
let cluster_fit_result = compute_cluster_fit(new_semantic, &cluster_context, &cluster_fit_config);
let cluster_fit = cluster_fit_result.score;
```

**Verdict**: INNOCENT - ClusterFit is properly included in Delta-C calculation.

---

### 4. UTL-003: Per-Embedder Delta-S Methods

**Constitution Requirements**:
| Embedder | Required Method | Implementation |
|----------|-----------------|----------------|
| E1 Semantic | GMM + Mahalanobis | GmmMahalanobisEntropy |
| E5 Causal | Asymmetric KNN | AsymmetricKnnEntropy |
| E7 Code | GMM+KNN hybrid | HybridGmmKnnEntropy |
| E9 HDC | Hamming to prototypes | HammingPrototypeEntropy |
| E13 SPLADE | 1 - jaccard(active) | JaccardActiveEntropy |
| Default | KNN distance | DefaultKnnEntropy |

**Source of Truth**: `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs:51-98`

```rust
pub fn create(embedder: Embedder, config: &SurpriseConfig) -> Box<dyn EmbedderEntropy> {
    match embedder {
        // E1: GMM + Mahalanobis distance
        Embedder::Semantic => Box::new(GmmMahalanobisEntropy::from_config(config)),

        // E5: Asymmetric KNN with direction modifiers
        Embedder::Causal => Box::new(
            AsymmetricKnnEntropy::new(config.k_neighbors)
                .with_direction_modifiers(
                    config.causal_cause_to_effect_mod,
                    config.causal_effect_to_cause_mod,
                ),
        ),

        // E9: Hamming distance to prototypes
        Embedder::Hdc => Box::new(
            HammingPrototypeEntropy::new(config.hdc_max_prototypes)
                .with_threshold(config.hdc_binarization_threshold),
        ),

        // E13: Jaccard similarity of active dimensions
        Embedder::KeywordSplade => Box::new(
            JaccardActiveEntropy::new()
                .with_threshold(config.splade_activation_threshold)
                .with_smoothing(config.splade_smoothing),
        ),

        // E7 (Code): Hybrid GMM+KNN
        Embedder::Code => Box::new(HybridGmmKnnEntropy::from_config(config)),

        // E10 (Multimodal): Cross-modal KNN
        Embedder::Multimodal => Box::new(CrossModalEntropy::from_config(config)),

        // E11 (Entity): TransE distance
        Embedder::Entity => Box::new(TransEEntropy::from_config(config)),

        // E12 (LateInteraction): MaxSim token-level entropy
        Embedder::LateInteraction => Box::new(MaxSimTokenEntropy::from_config(config)),

        // E2-E4, E6, E8: Default KNN-based entropy
        _ => Box::new(DefaultKnnEntropy::from_config(embedder, config)),
    }
}
```

**Additional Implementations Beyond Constitution**:
- E10 Multimodal: CrossModalEntropy (not specified in constitution, but reasonable)
- E11 Entity: TransEEntropy (not specified in constitution, but appropriate for entity embeddings)
- E12 LateInteraction: MaxSimTokenEntropy (not specified, appropriate for ColBERT-style)

**Verdict**: INNOCENT - All constitution-specified methods are implemented, plus additional specialized methods.

---

### 5. UTL-004: Multi-Embedding Aggregation

**Source of Truth**: `/home/cabdru/contextgraph/crates/context-graph-core/src/similarity/multi_utl.rs`

**Formula Implementation**:
```rust
/// L_multi = sigmoid(2.0 * (SUM_i tau_i * lambda_S * Delta_S_i) *
///                          (SUM_j tau_j * lambda_C * Delta_C_j) *
///                          w_e * cos(phi))
```

**Computation**: Lines 170-192 implement the formula correctly:
```rust
pub fn compute(&self) -> f32 {
    let semantic_sum: f32 = self.semantic_deltas.iter()
        .zip(self.tau_weights.iter())
        .map(|(delta, tau)| tau * self.lambda_s * delta)
        .sum();

    let coherence_sum: f32 = self.coherence_deltas.iter()
        .zip(self.tau_weights.iter())
        .map(|(delta, tau)| tau * self.lambda_c * delta)
        .sum();

    let raw = 2.0 * semantic_sum * coherence_sum * self.w_e * self.phi.cos();
    sigmoid(raw)
}
```

**GIGO Prevention (AP-007)**: Lines 289-304 implement garbage input rejection:
```rust
if semantic_sum < MIN_SIGNAL_THRESHOLD && coherence_sum < MIN_SIGNAL_THRESHOLD {
    return Some(format!(
        "GIGO rejected: Both semantic_deltas (sum={:.6}) and coherence_deltas (sum={:.6}) are effectively zero."
    ));
}
```

**Verdict**: INNOCENT - Multi-embedding aggregation with tau weights, lambdas, w_e, and phi is implemented.

---

### 6. UTL-005: Johari->Action Mapping (DISCREPANCY FOUND)

**Constitution Requirement**:
```
Open:    Delta-S < 0.5, Delta-C > 0.5 -> DirectRecall
Blind:   Delta-S > 0.5, Delta-C < 0.5 -> TriggerDream
Hidden:  Delta-S < 0.5, Delta-C < 0.5 -> GetNeighborhood
Unknown: Delta-S > 0.5, Delta-C > 0.5 -> EpistemicAction
```

**Actual Implementation**: `/home/cabdru/contextgraph/crates/context-graph-utl/src/johari/retrieval/action.rs:21-46`
```rust
pub enum SuggestedAction {
    /// Direct memory recall - for Open quadrant.
    DirectRecall,

    /// Epistemic action or dream-based discovery - for Blind quadrant.
    EpistemicAction,    // <-- Constitution says TriggerDream for Blind

    /// Neighborhood exploration - for Hidden quadrant.
    GetNeighborhood,

    /// Frontier exploration via dream consolidation - for Unknown quadrant.
    TriggerDream,       // <-- Constitution says EpistemicAction for Unknown
}
```

**Issue Analysis**:
The mapping appears to have **swapped** the actions for Blind and Unknown quadrants:

| Quadrant | Constitution | Implementation | Status |
|----------|--------------|----------------|--------|
| Open | DirectRecall | DirectRecall | CORRECT |
| Blind | TriggerDream | EpistemicAction | **SWAPPED** |
| Hidden | GetNeighborhood | GetNeighborhood | CORRECT |
| Unknown | EpistemicAction | TriggerDream | **SWAPPED** |

**HOWEVER**, reviewing the docstrings more closely:
- `EpistemicAction` doc says "for Blind quadrant"
- `TriggerDream` doc says "for Unknown quadrant"

This appears to be a **semantic naming disagreement** rather than a functional bug. The actions are:
- Blind (high surprise, low coherence): Named "EpistemicAction" but described as "Epistemic action or dream-based discovery"
- Unknown (high surprise, high coherence): Named "TriggerDream" but described as "Frontier exploration via dream consolidation"

**Verdict**: SUSPICIOUS (MEDIUM) - The action naming appears to swap Blind<->Unknown terminology from constitution, but the functional descriptions are reasonable. This needs clarification.

---

### 7. Lifecycle Lambda Weights Verification

**Constitution Requirement**:
```
Infancy:  lambda_s = 0.7, lambda_c = 0.3 (favor novelty)
Growth:   lambda_s = 0.5, lambda_c = 0.5 (balanced)
Maturity: lambda_s = 0.3, lambda_c = 0.7 (favor coherence)
```

**Source of Truth**: `/home/cabdru/contextgraph/crates/context-graph-utl/src/lifecycle/lambda.rs:154-159`
```rust
pub fn for_stage(stage: LifecycleStage) -> Self {
    match stage {
        LifecycleStage::Infancy => Self::new_unchecked(0.7, 0.3),
        LifecycleStage::Growth => Self::new_unchecked(0.5, 0.5),
        LifecycleStage::Maturity => Self::new_unchecked(0.3, 0.7),
    }
}
```

**Stage Thresholds**: `/home/cabdru/contextgraph/crates/context-graph-utl/src/lifecycle/stage.rs:70-73`
```rust
pub const INFANCY_THRESHOLD: u64 = 50;
pub const GROWTH_THRESHOLD: u64 = 500;
```

**Verdict**: INNOCENT - Lambda weights match constitution exactly.

---

### 8. Johari Quadrant Classification

**Source of Truth**: `/home/cabdru/contextgraph/crates/context-graph-utl/src/johari/classifier.rs:70-85`
```rust
fn classify_with_thresholds(
    delta_s: f32,
    delta_c: f32,
    surprise_threshold: f32,
    coherence_threshold: f32,
) -> JohariQuadrant {
    let low_surprise = delta_s < surprise_threshold;
    let high_coherence = delta_c > coherence_threshold;

    match (low_surprise, high_coherence) {
        (true, true) => JohariQuadrant::Open,   // Low S, High C
        (false, false) => JohariQuadrant::Blind, // High S, Low C
        (true, false) => JohariQuadrant::Hidden, // Low S, Low C
        (false, true) => JohariQuadrant::Unknown, // High S, High C
    }
}
```

**Verification against Constitution**:
| Condition | Constitution | Implementation | Match |
|-----------|--------------|----------------|-------|
| Delta-S < 0.5, Delta-C > 0.5 | Open | Open | YES |
| Delta-S > 0.5, Delta-C < 0.5 | Blind | Blind | YES |
| Delta-S < 0.5, Delta-C < 0.5 | Hidden | Hidden | YES |
| Delta-S > 0.5, Delta-C > 0.5 | Unknown | Unknown | YES |

**Verdict**: INNOCENT - Quadrant classification is correct.

---

### 9. Embedder Enum Verification

**Source of Truth**: `/home/cabdru/contextgraph/crates/context-graph-core/src/teleological/embedder.rs:32-61`

| Index | Enum Variant | Name | Constitution |
|-------|--------------|------|--------------|
| 0 | Semantic | E1_Semantic | Matches |
| 1 | TemporalRecent | E2_Temporal_Recent | Matches |
| 2 | TemporalPeriodic | E3_Temporal_Periodic | Matches |
| 3 | TemporalPositional | E4_Temporal_Positional | Matches |
| 4 | Causal | E5_Causal | Matches |
| 5 | Sparse | E6_Sparse_Lexical | Matches |
| 6 | Code | E7_Code | Matches |
| 7 | Graph | E8_Graph | Matches |
| 8 | Hdc | E9_HDC | Matches |
| 9 | Multimodal | E10_Multimodal | Matches |
| 10 | Entity | E11_Entity | Matches |
| 11 | LateInteraction | E12_Late_Interaction | Matches |
| 12 | KeywordSplade | E13_SPLADE | Matches |

**Verdict**: INNOCENT - All 13 embedders are correctly defined.

---

## Issues Found

### ISSUE-001: Johari Action Mapping Terminology Swap (MEDIUM)

**Severity**: MEDIUM
**Location**: `/home/cabdru/contextgraph/crates/context-graph-utl/src/johari/retrieval/action.rs`
**Line Numbers**: 21-46

**Description**: The `SuggestedAction` enum appears to swap the Blind and Unknown action names compared to constitution:
- Constitution: Blind -> TriggerDream, Unknown -> EpistemicAction
- Implementation: Blind -> EpistemicAction, Unknown -> TriggerDream

**Impact**: Could cause confusion when debugging or when external systems rely on action names.

**Recommendation**: Either:
1. Rename the enum variants to match constitution exactly, OR
2. Document the intentional deviation with clear reasoning

---

## Missing Implementations

None critical. All core UTL functionality is present.

---

## Anti-Pattern Compliance

| Anti-Pattern | Requirement | Status |
|--------------|-------------|--------|
| AP-007 | GIGO Prevention | COMPLIANT - compute_validated() rejects garbage input |
| AP-010 | No NaN/Infinity | COMPLIANT - Explicit checks throughout |
| AP-33 | ClusterFit MUST be included | COMPLIANT - Included with 0.4 weight |

---

## Test Coverage Verification

**Executed Tests**:
- `test_coherence_three_component_formula` - PASS
- `test_coherence_default_weights` - PASS
- `test_factory_creates_correct_types` - PASS
- `test_constitution_compliance` - PASS
- `test_classify_quadrant_*` - PASS (all quadrant tests)
- `test_lambda_weights_*` - PASS (all lifecycle tests)

---

## Chain of Custody

| Timestamp | Action | File | Evidence |
|-----------|--------|------|----------|
| 2026-01-12 | MCP Tool Verified | `tools/names.rs:30` | COMPUTE_DELTA_SC exists |
| 2026-01-12 | Delta-C Formula Verified | `coherence/tracker.rs:316` | 0.4/0.4/0.2 weights |
| 2026-01-12 | ClusterFit Verified | `gwt_compute.rs:163` | compute_cluster_fit called |
| 2026-01-12 | Factory Routing Verified | `factory.rs:51` | All 13 embedders routed |
| 2026-01-12 | Lambda Weights Verified | `lambda.rs:154` | 0.7/0.3, 0.5/0.5, 0.3/0.7 |
| 2026-01-12 | Johari Classification Verified | `classifier.rs:70` | Correct quadrant logic |

---

## Final Verdict

```
===============================================================
                    CASE CLOSED
===============================================================

THE CRIME: None critical - UTL system is largely compliant

THE SUSPECT: Action.rs terminology swap for Blind/Unknown

THE EVIDENCE:
  1. compute_delta_sc MCP tool EXISTS and DISPATCHES correctly
  2. Delta-C formula CORRECTLY uses 0.4*Conn + 0.4*ClusterFit + 0.2*Cons
  3. ClusterFit IS INCLUDED per AP-33
  4. All 13 per-embedder Delta-S methods ARE CORRECT
  5. Multi-embedding aggregation WITH tau weights EXISTS
  6. Lifecycle lambda weights MATCH constitution
  7. Johari quadrant classification IS CORRECT

THE VERDICT: SUBSTANTIALLY INNOCENT
  - 6/7 requirements fully compliant
  - 1 MEDIUM issue: Action naming swap (functional but confusing)

THE SENTENCE: Document or fix the Johari action naming swap

===============================================================
        UTL-INVESTIGATION-2026-01-12 - VERDICT: INNOCENT
===============================================================
```

---

## Recommendations

1. **MEDIUM Priority**: Clarify or fix the Johari action naming for Blind/Unknown quadrants to match constitution terminology exactly.

2. **LOW Priority**: Add explicit documentation in action.rs explaining why TriggerDream is for Unknown and EpistemicAction is for Blind (if intentional).

3. **VERIFICATION**: Run full integration test suite to verify end-to-end UTL computation produces expected learning signals.

---

*"The game is afoot!"*

*- Sherlock Holmes, Code Detective*
