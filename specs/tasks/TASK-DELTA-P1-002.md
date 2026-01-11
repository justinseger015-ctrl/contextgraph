# TASK-DELTA-P1-002: DeltaScComputer Implementation

## Metadata

| Field | Value |
|-------|-------|
| **ID** | TASK-DELTA-P1-002 |
| **Version** | 1.0 |
| **Status** | ready |
| **Layer** | logic |
| **Sequence** | 2 of 4 |
| **Priority** | P1 |
| **Estimated Complexity** | high |
| **Estimated Duration** | 4-6 hours |
| **Implements** | REQ-UTL-009 through REQ-UTL-019 |
| **Depends On** | TASK-DELTA-P1-001 |
| **Spec Ref** | SPEC-UTL-001 |
| **Gap Ref** | MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md GAP 1 |

---

## Context

This task implements the core business logic for computing Delta-S (entropy) and Delta-C (coherence) across all 13 embedders. It integrates existing per-embedder entropy calculators with coherence computation and Johari classification.

**Why This Second**: Following Inside-Out, Bottom-Up:
1. Types (TASK-DELTA-P1-001) must exist first
2. Business logic depends on types but not on MCP registration
3. Logic can be unit tested in isolation before surface wiring

**Gap Being Addressed**:
> GAP 1: UTL compute_delta_sc MCP Tool Missing
> Core computation logic that powers the MCP tool

---

## Input Context Files

| Purpose | File |
|---------|------|
| Request/Response types | `crates/context-graph-mcp/src/types/delta_sc.rs` (from TASK-DELTA-P1-001) |
| Embedder entropy trait | `crates/context-graph-utl/src/surprise/embedder_entropy/mod.rs` |
| Entropy factory | `crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs` |
| All entropy calculators | `crates/context-graph-utl/src/surprise/embedder_entropy/*.rs` |
| Coherence tracker | `crates/context-graph-utl/src/coherence/` (if exists) |
| Johari manager | `crates/context-graph-core/src/johari/default_manager.rs` |
| Constitution | `docs2/constitution.yaml#delta_sc` |

---

## Prerequisites

| Check | Verification |
|-------|--------------|
| TASK-DELTA-P1-001 complete | Types exist in `crates/context-graph-mcp/src/types/delta_sc.rs` |
| EmbedderEntropy trait exists | `grep "pub trait EmbedderEntropy" crates/context-graph-utl/` |
| EmbedderEntropyFactory exists | `grep "EmbedderEntropyFactory" crates/context-graph-utl/` |
| JohariQuadrant::classify exists | `grep "JohariQuadrant" crates/context-graph-core/` |
| NUM_EMBEDDERS = 13 | `grep "NUM_EMBEDDERS.*13" crates/context-graph-core/` |

---

## Scope

### In Scope

- Create `DeltaScComputer` struct that orchestrates computation
- Implement `compute()` method returning `ComputeDeltaScResponse`
- Wire existing `EmbedderEntropyFactory::create_all()` for per-embedder Delta-S
- Implement coherence computation with three components:
  - Connectivity (use existing or stub returning 0.5)
  - ClusterFit (from SPEC-UTL-002 or stub if not ready)
  - Consistency (use existing or stub returning 0.5)
- Implement Johari classification using thresholds from constitution
- Add comprehensive unit tests for all 13 embedder methods
- Add property-based tests for output bounds

### Out of Scope

- MCP handler registration (TASK-DELTA-P1-003)
- Integration tests with real graph data (TASK-DELTA-P1-004)
- Full ClusterFit implementation (covered by SPEC-UTL-002)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-mcp/src/services/delta_sc_computer.rs

use std::time::Instant;
use uuid::Uuid;

use context_graph_core::johari::NUM_EMBEDDERS;
use context_graph_core::teleological::Embedder;
use context_graph_core::types::fingerprint::TeleologicalFingerprint;
use context_graph_core::types::JohariQuadrant;
use context_graph_utl::surprise::embedder_entropy::{EmbedderEntropy, EmbedderEntropyFactory};
use context_graph_utl::config::SurpriseConfig;

use crate::error::McpResult;
use crate::types::delta_sc::{ComputeDeltaScRequest, ComputeDeltaScResponse, DeltaScDiagnostics};

/// Constitution weights for coherence computation.
const COHERENCE_CONNECTIVITY_WEIGHT: f32 = 0.4;
const COHERENCE_CLUSTER_FIT_WEIGHT: f32 = 0.4;
const COHERENCE_CONSISTENCY_WEIGHT: f32 = 0.2;

/// Default Johari threshold from constitution.
const DEFAULT_JOHARI_THRESHOLD: f32 = 0.5;

/// Method names for diagnostics, per constitution.yaml delta_sc.Delta_S_methods.
const EMBEDDER_METHODS: [&str; 13] = [
    "GMM+Mahalanobis",   // E1 Semantic
    "KNN",               // E2 TemporalRecent
    "KNN",               // E3 TemporalPeriodic
    "KNN",               // E4 TemporalPositional
    "Asymmetric KNN",    // E5 Causal
    "IDF/Jaccard",       // E6 Sparse
    "GMM+KNN Hybrid",    // E7 Code
    "KNN",               // E8 Graph
    "Hamming",           // E9 Hdc
    "Cross-modal KNN",   // E10 Multimodal
    "TransE",            // E11 Entity
    "Token MaxSim",      // E12 LateInteraction
    "Jaccard",           // E13 KeywordSplade
];

/// Computes Delta-S (entropy) and Delta-C (coherence) for fingerprint updates.
///
/// Orchestrates per-embedder entropy calculators and coherence components
/// per constitution.yaml delta_sc section.
///
/// # Thread Safety
/// This struct is Send + Sync for use in async MCP handlers.
pub struct DeltaScComputer {
    /// Per-embedder entropy calculators (created once, reused).
    entropy_calculators: Vec<Box<dyn EmbedderEntropy>>,

    /// Surprise configuration.
    config: SurpriseConfig,
}

impl DeltaScComputer {
    /// Create a new DeltaScComputer with default configuration.
    pub fn new() -> Self;

    /// Create with custom configuration.
    pub fn with_config(config: SurpriseConfig) -> Self;

    /// Compute Delta-S and Delta-C for a fingerprint update.
    ///
    /// # Arguments
    /// * `request` - The computation request with old/new fingerprints
    ///
    /// # Returns
    /// Complete response with per-embedder and aggregate values.
    ///
    /// # Errors
    /// - `McpError::InvalidFingerprint` if fingerprints incomplete
    /// - `McpError::ComputationError` if entropy calculation fails
    ///
    /// # Performance
    /// Target: < 25ms p95 per constitution.yaml perf.latency.inject_context
    pub async fn compute(&self, request: &ComputeDeltaScRequest) -> McpResult<ComputeDeltaScResponse>;

    /// Validate that both fingerprints contain all 13 embedders.
    fn validate_fingerprints(
        &self,
        old: &TeleologicalFingerprint,
        new: &TeleologicalFingerprint,
    ) -> McpResult<()>;

    /// Compute Delta-S for all 13 embedders.
    async fn compute_delta_s_all(
        &self,
        old: &TeleologicalFingerprint,
        new: &TeleologicalFingerprint,
    ) -> McpResult<[f32; 13]>;

    /// Compute weighted aggregate Delta-S.
    fn compute_aggregate_delta_s(&self, per_embedder: &[f32; 13], weights: &[f32; 13]) -> f32;

    /// Compute Delta-C with three components.
    /// Returns (delta_c, connectivity, cluster_fit, consistency).
    async fn compute_delta_c(
        &self,
        vertex_id: Uuid,
        fingerprint: &TeleologicalFingerprint,
    ) -> McpResult<(f32, f32, f32, f32)>;

    /// Classify Johari quadrant for a single (Delta-S, Delta-C) pair.
    fn classify_johari(delta_s: f32, delta_c: f32, threshold: f32) -> JohariQuadrant;

    /// Classify Johari for all 13 embedders.
    fn classify_johari_per_embedder(
        &self,
        delta_s_per_embedder: &[f32; 13],
        delta_c: f32,
        threshold: f32,
    ) -> [JohariQuadrant; 13];
}

impl Default for DeltaScComputer {
    fn default() -> Self {
        Self::new()
    }
}
```

### Constraints

- All f32 outputs MUST be clamped to [0.0, 1.0] per AP-10
- NO NaN or Infinity values (validate and clamp)
- Use existing `EmbedderEntropyFactory` - do NOT reimplement calculators
- Johari classification MUST match constitution thresholds exactly
- Coherence weights MUST be 0.4, 0.4, 0.2 per constitution
- Method names in diagnostics MUST match constitution.yaml delta_sc.Delta_S_methods
- If ClusterFit not available from SPEC-UTL-002, use stub returning 0.5

### Verification

```bash
# Unit tests pass
cargo test -p context-graph-mcp delta_sc_computer::tests -- --nocapture

# All outputs in valid range
cargo test -p context-graph-mcp test_all_outputs_bounded

# Johari classification correct
cargo test -p context-graph-mcp test_johari_classification

# No clippy warnings
cargo clippy -p context-graph-mcp -- -D warnings
```

---

## Pseudo Code

```rust
// compute() implementation:
fn compute(&self, request: &ComputeDeltaScRequest) -> McpResult<ComputeDeltaScResponse> {
    let start = Instant::now();

    // 1. Validate fingerprints (ARCH-05: all 13 embedders required)
    self.validate_fingerprints(&request.old_fingerprint, &request.new_fingerprint)?;

    // 2. Compute per-embedder Delta-S
    let delta_s_per_embedder = self.compute_delta_s_all(
        &request.old_fingerprint,
        &request.new_fingerprint,
    ).await?;

    // 3. Compute aggregate Delta-S (uniform weights for now)
    let embedder_weights = [1.0 / 13.0; 13];
    let delta_s_aggregate = self.compute_aggregate_delta_s(&delta_s_per_embedder, &embedder_weights);

    // 4. Compute Delta-C (coherence)
    let (delta_c, connectivity, cluster_fit, consistency) = self.compute_delta_c(
        request.vertex_id,
        &request.new_fingerprint,
    ).await?;

    // 5. Classify Johari quadrants
    let johari_threshold = request.johari_threshold.unwrap_or(DEFAULT_JOHARI_THRESHOLD);
    let johari_quadrants = self.classify_johari_per_embedder(&delta_s_per_embedder, delta_c, johari_threshold);
    let johari_aggregate = Self::classify_johari(delta_s_aggregate, delta_c, johari_threshold);

    // 6. Compute UTL learning potential
    let utl_learning_potential = delta_s_aggregate * delta_c;

    // 7. Build diagnostics if requested
    let diagnostics = if request.include_diagnostics {
        Some(DeltaScDiagnostics {
            methods_used: EMBEDDER_METHODS.map(String::from),
            connectivity,
            cluster_fit,
            consistency,
            computation_time_us: start.elapsed().as_micros() as u64,
            embedder_weights,
        })
    } else {
        None
    };

    Ok(ComputeDeltaScResponse {
        delta_s_per_embedder,
        delta_s_aggregate,
        delta_c,
        johari_quadrants,
        johari_aggregate,
        utl_learning_potential,
        diagnostics,
    })
}

// compute_delta_s_all():
async fn compute_delta_s_all(&self, old: &TeleologicalFingerprint, new: &TeleologicalFingerprint) -> McpResult<[f32; 13]> {
    let mut delta_s = [0.0f32; 13];

    for idx in 0..13 {
        let old_embedding = old.semantic.get_embedding(Embedder::from_index(idx)?);
        let new_embedding = new.semantic.get_embedding(Embedder::from_index(idx)?);

        // Build history from old embedding
        let history = vec![old_embedding.to_vec()];

        // Use appropriate calculator
        let calculator = &self.entropy_calculators[idx];
        let raw_delta_s = calculator.compute_delta_s(new_embedding, &history, 5)?;

        // Clamp to [0, 1] per AP-10
        delta_s[idx] = raw_delta_s.clamp(0.0, 1.0);
    }

    Ok(delta_s)
}

// classify_johari():
fn classify_johari(delta_s: f32, delta_c: f32, threshold: f32) -> JohariQuadrant {
    // Per constitution.yaml delta_sc.johari
    match (delta_s <= threshold, delta_c > threshold) {
        (true, true) => JohariQuadrant::Open,    // Low surprise, high coherence
        (false, false) => JohariQuadrant::Blind, // High surprise, low coherence
        (true, false) => JohariQuadrant::Hidden, // Low surprise, low coherence
        (false, true) => JohariQuadrant::Unknown, // High surprise, high coherence
    }
}
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-mcp/src/services/delta_sc_computer.rs` | Core computation logic |

---

## Files to Modify

| Path | Change |
|------|--------|
| `crates/context-graph-mcp/src/services/mod.rs` | Add `pub mod delta_sc_computer;` |
| `crates/context-graph-mcp/Cargo.toml` | Ensure `context-graph-utl` dependency |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| All 13 embedder methods wired | Test each embedder produces output |
| Delta-S in [0, 1] for all embedders | Property test with random fingerprints |
| Delta-C in [0, 1] | Property test |
| Johari thresholds match constitution | Unit tests for boundary conditions |
| Coherence weights are 0.4/0.4/0.2 | Code review + unit test |
| No NaN/Infinity (AP-10) | Test with edge cases (zero vectors, etc.) |
| Performance < 25ms | Benchmark test |

---

## Test Commands

```bash
# Unit tests
cargo test -p context-graph-mcp delta_sc_computer -- --nocapture

# Property-based tests
cargo test -p context-graph-mcp test_delta_bounds

# Johari classification tests
cargo test -p context-graph-mcp test_johari

# Benchmark (if configured)
cargo bench -p context-graph-mcp delta_sc

# Full test suite
cargo test -p context-graph-mcp --lib
```

---

## Notes

- This task implements the bulk of SPEC-UTL-001 requirements
- Existing EmbedderEntropyFactory already handles per-embedder method selection
- If SPEC-UTL-002 (ClusterFit) is not complete, use stub returning 0.5
- The coherence components may need stubs initially; wire real implementations when available
- async is used even though computation is CPU-bound to match handler pattern

---

## Appendix A: Specialized Delta-S Calculator Specifications

This appendix provides complete technical specifications for the specialized Delta-S calculation methods per constitution.yaml delta_sc.Delta_S_methods. Currently, some embedders fall back to DefaultKnnEntropy. This section specifies the proper implementations.

### A.1 Factory Routing Summary

| Embedder | Method | Current Implementation | Target Implementation |
|----------|--------|------------------------|----------------------|
| E1 (Semantic) | GMM+Mahalanobis | GmmMahalanobisEntropy | GmmMahalanobisEntropy (DONE) |
| E2 (TemporalRecent) | KNN | DefaultKnnEntropy | DefaultKnnEntropy (DONE) |
| E3 (TemporalPeriodic) | KNN | DefaultKnnEntropy | DefaultKnnEntropy (DONE) |
| E4 (TemporalPositional) | KNN | DefaultKnnEntropy | DefaultKnnEntropy (DONE) |
| E5 (Causal) | Asymmetric KNN | AsymmetricKnnEntropy | AsymmetricKnnEntropy (DONE) |
| E6 (Sparse) | IDF/Jaccard | DefaultKnnEntropy (FALLBACK) | **SparseIdfJaccardEntropy (NEW)** |
| E7 (Code) | GMM+KNN Hybrid | DefaultKnnEntropy (FALLBACK) | **HybridGmmKnnEntropy (NEW)** |
| E8 (Graph) | KNN | DefaultKnnEntropy | DefaultKnnEntropy (DONE) |
| E9 (Hdc) | Hamming | HammingPrototypeEntropy | HammingPrototypeEntropy (DONE) |
| E10 (Multimodal) | Cross-modal KNN | DefaultKnnEntropy (FALLBACK) | **CrossModalKnnEntropy (NEW)** |
| E11 (Entity) | TransE | DefaultKnnEntropy (FALLBACK) | **TransEEntropy (NEW)** |
| E12 (LateInteraction) | Token MaxSim | DefaultKnnEntropy (FALLBACK) | **MaxSimTokenEntropy (NEW)** |
| E13 (KeywordSplade) | Jaccard | JaccardActiveEntropy | JaccardActiveEntropy (DONE) |

---

### A.2 E6 (Sparse) - SparseIdfJaccardEntropy

**Purpose**: Compute entropy for sparse lexical embeddings using IDF-weighted Jaccard distance.

**Mathematical Foundation**:
```
Delta_S = 1 - J_idf(A, B)

where:
J_idf(A, B) = sum(IDF(d) for d in A intersection B) / sum(IDF(d) for d in A union B)

IDF(d) = log(N / (1 + df(d)))
  - N = total documents seen
  - df(d) = document frequency of dimension d

For sparse vectors:
- active_A = {i : A[i] > threshold}
- active_B = {i : B[i] > threshold}
- intersection = active_A intersect active_B
- union = active_A union active_B
```

**Implementation Signature**:
```rust
/// Sparse embedding entropy using IDF-weighted Jaccard distance.
///
/// Specialized for E6 (Sparse) embeddings per constitution.yaml delta_sc.
/// Uses inverted document frequency to weight dimension importance.
pub struct SparseIdfJaccardEntropy {
    /// Activation threshold for sparse dimensions
    threshold: f32,

    /// IDF weights per dimension (learned from history)
    idf_weights: Vec<f32>,

    /// Document frequency counts
    doc_freq: Vec<u32>,

    /// Total documents seen
    total_docs: u32,
}

impl SparseIdfJaccardEntropy {
    /// Create with activation threshold (default: 0.01)
    pub fn new(threshold: f32) -> Self;

    /// Update IDF weights from observed history
    fn update_idf_weights(&mut self, embeddings: &[Vec<f32>]);

    /// Compute IDF-weighted Jaccard distance
    fn idf_jaccard(&self, a: &[f32], b: &[f32]) -> f32;
}

impl EmbedderEntropy for SparseIdfJaccardEntropy {
    fn compute_delta_s(&self, current: &[f32], history: &[Vec<f32>], k: usize) -> UtlResult<f32> {
        // 1. Find active dimensions in current (above threshold)
        // 2. For each history embedding, compute IDF-weighted Jaccard
        // 3. Return 1 - max(Jaccard) as Delta-S (higher Jaccard = lower surprise)
        // 4. Clamp to [0.0, 1.0]
    }

    fn embedder_type(&self) -> Embedder { Embedder::Sparse }

    fn reset(&mut self) {
        self.idf_weights.clear();
        self.doc_freq.clear();
        self.total_docs = 0;
    }
}
```

**Test Criteria**:
- TC-E6-01: Identical sparse vectors return Delta-S = 0.0
- TC-E6-02: Disjoint active dimensions return Delta-S = 1.0
- TC-E6-03: Partial overlap returns Delta-S in (0, 1)
- TC-E6-04: IDF weights boost rare dimension matches
- TC-E6-05: Empty sparse vector (all below threshold) returns Delta-S = 1.0
- TC-E6-06: Output always in [0.0, 1.0], no NaN/Infinity

---

### A.3 E7 (Code) - HybridGmmKnnEntropy

**Purpose**: Compute entropy for code embeddings using a hybrid GMM + KNN approach.

**Mathematical Foundation**:
```
Delta_S = 0.5 * Delta_S_gmm + 0.5 * Delta_S_knn

where:
Delta_S_gmm = 1 - P(e | GMM)  [from GmmMahalanobisEntropy]
Delta_S_knn = sigmoid((d_k - mu) / sigma)  [from DefaultKnnEntropy]

Rationale:
- GMM captures code structure clusters (function signatures, patterns)
- KNN captures local neighborhood surprise
- Hybrid balances global structure with local novelty
```

**Implementation Signature**:
```rust
/// Hybrid GMM+KNN entropy for code embeddings.
///
/// Specialized for E7 (Code) per constitution.yaml delta_sc.
/// Combines structural pattern detection (GMM) with local novelty (KNN).
pub struct HybridGmmKnnEntropy {
    /// GMM component for structural patterns
    gmm: GmmMahalanobisEntropy,

    /// KNN component for local novelty
    knn: DefaultKnnEntropy,

    /// Weight for GMM component (default: 0.5)
    gmm_weight: f32,

    /// Weight for KNN component (default: 0.5)
    knn_weight: f32,
}

impl HybridGmmKnnEntropy {
    /// Create with default weights (0.5, 0.5)
    pub fn new(config: &SurpriseConfig) -> Self;

    /// Create with custom weights (must sum to 1.0)
    pub fn with_weights(config: &SurpriseConfig, gmm_weight: f32, knn_weight: f32) -> Self;
}

impl EmbedderEntropy for HybridGmmKnnEntropy {
    fn compute_delta_s(&self, current: &[f32], history: &[Vec<f32>], k: usize) -> UtlResult<f32> {
        // 1. Compute GMM-based surprise
        let delta_s_gmm = self.gmm.compute_delta_s(current, history, k)?;

        // 2. Compute KNN-based surprise
        let delta_s_knn = self.knn.compute_delta_s(current, history, k)?;

        // 3. Combine with weights
        let delta_s = self.gmm_weight * delta_s_gmm + self.knn_weight * delta_s_knn;

        // 4. Clamp to [0.0, 1.0]
        Ok(delta_s.clamp(0.0, 1.0))
    }

    fn embedder_type(&self) -> Embedder { Embedder::Code }

    fn reset(&mut self) {
        self.gmm.reset();
        self.knn.reset();
    }
}
```

**Test Criteria**:
- TC-E7-01: Weights sum to 1.0 (invariant)
- TC-E7-02: Result is interpolation of GMM and KNN
- TC-E7-03: When GMM returns high surprise, result reflects this
- TC-E7-04: When KNN returns high surprise, result reflects this
- TC-E7-05: Empty history returns Delta-S = 1.0
- TC-E7-06: Output always in [0.0, 1.0], no NaN/Infinity

---

### A.4 E10 (Multimodal) - CrossModalKnnEntropy

**Purpose**: Compute entropy for multimodal embeddings by averaging across modality components.

**Mathematical Foundation**:
```
Delta_S = (Delta_S_text + Delta_S_image + Delta_S_audio) / num_modalities

where each modality component:
Delta_S_modality = sigmoid((d_k - mu_modality) / sigma_modality)

Multimodal embedding structure (768D total):
- dims[0..256]: text component
- dims[256..512]: image component
- dims[512..768]: audio/other component

Cross-modal alignment check:
- If modality components are well-aligned (low variance across modalities),
  the content is semantically consistent
- High variance suggests semantic drift or multimodal mismatch
```

**Implementation Signature**:
```rust
/// Cross-modal entropy for multimodal embeddings.
///
/// Specialized for E10 (Multimodal) per constitution.yaml delta_sc.
/// Computes per-modality surprise and aggregates with alignment penalty.
pub struct CrossModalKnnEntropy {
    /// Per-modality KNN calculators
    modality_calculators: Vec<DefaultKnnEntropy>,

    /// Modality dimension ranges (start, end)
    modality_ranges: Vec<(usize, usize)>,

    /// Alignment penalty weight (penalizes cross-modal mismatch)
    alignment_weight: f32,

    /// Number of neighbors
    k: usize,
}

impl CrossModalKnnEntropy {
    /// Create for standard 768D multimodal embedding (3 modalities)
    pub fn new(config: &SurpriseConfig) -> Self {
        Self {
            modality_calculators: vec![
                DefaultKnnEntropy::from_config(Embedder::Multimodal, config),
                DefaultKnnEntropy::from_config(Embedder::Multimodal, config),
                DefaultKnnEntropy::from_config(Embedder::Multimodal, config),
            ],
            modality_ranges: vec![(0, 256), (256, 512), (512, 768)],
            alignment_weight: 0.1,
            k: config.k_neighbors,
        }
    }

    /// Extract modality slice from embedding
    fn extract_modality(&self, embedding: &[f32], modality_idx: usize) -> Vec<f32>;

    /// Compute cross-modal alignment variance
    fn cross_modal_variance(&self, modality_surprises: &[f32]) -> f32;
}

impl EmbedderEntropy for CrossModalKnnEntropy {
    fn compute_delta_s(&self, current: &[f32], history: &[Vec<f32>], k: usize) -> UtlResult<f32> {
        // 1. For each modality, extract component and compute KNN surprise
        let mut modality_surprises = Vec::new();
        for (idx, (start, end)) in self.modality_ranges.iter().enumerate() {
            let current_mod = &current[*start..*end];
            let history_mod: Vec<Vec<f32>> = history.iter()
                .map(|h| h[*start..*end].to_vec())
                .collect();

            let surprise = self.modality_calculators[idx]
                .compute_delta_s(current_mod, &history_mod, k)?;
            modality_surprises.push(surprise);
        }

        // 2. Average modality surprises
        let avg_surprise = modality_surprises.iter().sum::<f32>()
            / modality_surprises.len() as f32;

        // 3. Add alignment penalty for cross-modal variance
        let variance = self.cross_modal_variance(&modality_surprises);
        let aligned_surprise = avg_surprise + self.alignment_weight * variance;

        // 4. Clamp to [0.0, 1.0]
        Ok(aligned_surprise.clamp(0.0, 1.0))
    }

    fn embedder_type(&self) -> Embedder { Embedder::Multimodal }

    fn reset(&mut self) {
        for calc in &mut self.modality_calculators {
            calc.reset();
        }
    }
}
```

**Test Criteria**:
- TC-E10-01: Aligned modalities (same surprise) have low penalty
- TC-E10-02: Misaligned modalities (varied surprise) have higher penalty
- TC-E10-03: Each modality contributes equally to average
- TC-E10-04: Empty history returns Delta-S = 1.0
- TC-E10-05: Output always in [0.0, 1.0], no NaN/Infinity
- TC-E10-06: Works with partial modalities (graceful degradation)

---

### A.5 E11 (Entity) - TransEEntropy

**Purpose**: Compute entropy for entity embeddings using TransE-style distance.

**Mathematical Foundation**:
```
TransE Model: h + r approx t
where:
- h = head entity embedding
- r = relation embedding
- t = tail entity embedding

Delta_S = min(||h + r - t||) for all (h, r, t) triples in history

For entity surprise:
1. Current embedding represents an entity
2. History contains entity embeddings from related contexts
3. High distance from predicted positions = high surprise

Distance normalization:
Delta_S = sigmoid(d - mu) / sigma
where d = min L2 distance to any history embedding
```

**Implementation Signature**:
```rust
/// TransE-based entropy for entity embeddings.
///
/// Specialized for E11 (Entity) per constitution.yaml delta_sc.
/// Uses translation-based distance for knowledge graph entities.
pub struct TransEEntropy {
    /// Learned relation vectors (context-dependent)
    relation_embeddings: HashMap<String, Vec<f32>>,

    /// Default relation for general entity comparison
    default_relation: Vec<f32>,

    /// Running statistics for normalization
    mean_distance: f32,
    std_distance: f32,
    n_samples: u32,
}

impl TransEEntropy {
    /// Create with embedding dimension
    pub fn new(dim: usize) -> Self {
        Self {
            relation_embeddings: HashMap::new(),
            default_relation: vec![0.0; dim], // Identity relation
            mean_distance: 0.5,
            std_distance: 0.25,
            n_samples: 0,
        }
    }

    /// Compute TransE distance: ||current - (history + relation)||
    fn transe_distance(&self, current: &[f32], history_entity: &[f32], relation: &[f32]) -> f32 {
        current.iter()
            .zip(history_entity.iter().zip(relation.iter()))
            .map(|(c, (h, r))| (c - (h + r)).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Update running statistics
    fn update_statistics(&mut self, distance: f32);
}

impl EmbedderEntropy for TransEEntropy {
    fn compute_delta_s(&self, current: &[f32], history: &[Vec<f32>], _k: usize) -> UtlResult<f32> {
        if history.is_empty() {
            return Ok(1.0);
        }

        // 1. Compute TransE distance to each history entity
        let distances: Vec<f32> = history.iter()
            .map(|h| self.transe_distance(current, h, &self.default_relation))
            .collect();

        // 2. Take minimum distance (closest match in entity space)
        let min_dist = distances.iter().cloned().fold(f32::MAX, f32::min);

        // 3. Normalize using running statistics
        let normalized = (min_dist - self.mean_distance) / self.std_distance.max(0.01);

        // 4. Sigmoid to [0, 1]
        let delta_s = 1.0 / (1.0 + (-normalized).exp());

        Ok(delta_s.clamp(0.0, 1.0))
    }

    fn embedder_type(&self) -> Embedder { Embedder::Entity }

    fn reset(&mut self) {
        self.relation_embeddings.clear();
        self.default_relation.fill(0.0);
        self.mean_distance = 0.5;
        self.std_distance = 0.25;
        self.n_samples = 0;
    }
}
```

**Test Criteria**:
- TC-E11-01: Identical entities return low Delta-S (approx 0.0)
- TC-E11-02: Unrelated entities return high Delta-S (approx 1.0)
- TC-E11-03: Related entities (via translation) return moderate Delta-S
- TC-E11-04: Empty history returns Delta-S = 1.0
- TC-E11-05: Output always in [0.0, 1.0], no NaN/Infinity
- TC-E11-06: Running statistics converge over time

---

### A.6 E12 (LateInteraction) - MaxSimTokenEntropy

**Purpose**: Compute entropy for token-level late interaction embeddings using MaxSim.

**Mathematical Foundation**:
```
ColBERT MaxSim: S(Q, D) = sum(max(q_i . d_j for all j) for all i)

For entropy:
1. Current embedding is a matrix [num_tokens x 128]
2. History embeddings are also token matrices
3. Compute MaxSim score between current and each history
4. Convert to surprise: Delta_S = 1 - max(normalized_maxsim)

Token-level scoring:
- For each token in current, find max similarity to any token in history doc
- Sum over all current tokens
- Normalize by number of tokens
```

**Implementation Signature**:
```rust
/// MaxSim token-level entropy for late interaction embeddings.
///
/// Specialized for E12 (LateInteraction) per constitution.yaml delta_sc.
/// Uses ColBERT-style MaxSim scoring at token level.
pub struct MaxSimTokenEntropy {
    /// Token embedding dimension (typically 128)
    token_dim: usize,

    /// Running statistics for normalization
    mean_maxsim: f32,
    std_maxsim: f32,
    n_samples: u32,
}

impl MaxSimTokenEntropy {
    /// Create with token dimension
    pub fn new(token_dim: usize) -> Self {
        Self {
            token_dim,
            mean_maxsim: 0.5,
            std_maxsim: 0.25,
            n_samples: 0,
        }
    }

    /// Parse flat vector into token matrix
    /// Flat format: [tok1_dim1, tok1_dim2, ..., tok1_dimN, tok2_dim1, ...]
    fn to_token_matrix(&self, flat: &[f32]) -> Vec<Vec<f32>> {
        flat.chunks(self.token_dim)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Compute MaxSim between two token matrices
    fn maxsim_score(&self, query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
        if query_tokens.is_empty() || doc_tokens.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0;
        for q_tok in query_tokens {
            let max_sim = doc_tokens.iter()
                .map(|d_tok| cosine_similarity(q_tok, d_tok))
                .fold(f32::NEG_INFINITY, f32::max);
            total_score += max_sim.max(0.0);
        }

        total_score / query_tokens.len() as f32
    }
}

impl EmbedderEntropy for MaxSimTokenEntropy {
    fn compute_delta_s(&self, current: &[f32], history: &[Vec<f32>], _k: usize) -> UtlResult<f32> {
        if history.is_empty() {
            return Ok(1.0);
        }

        // 1. Parse current into token matrix
        let current_tokens = self.to_token_matrix(current);

        // 2. Compute MaxSim to each history document
        let scores: Vec<f32> = history.iter()
            .map(|h| {
                let h_tokens = self.to_token_matrix(h);
                self.maxsim_score(&current_tokens, &h_tokens)
            })
            .collect();

        // 3. Take maximum score (best match)
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // 4. Convert to surprise: high similarity = low surprise
        // Normalize and invert
        let normalized = (max_score - self.mean_maxsim) / self.std_maxsim.max(0.01);
        let similarity = 1.0 / (1.0 + (-normalized).exp());
        let delta_s = 1.0 - similarity;

        Ok(delta_s.clamp(0.0, 1.0))
    }

    fn embedder_type(&self) -> Embedder { Embedder::LateInteraction }

    fn reset(&mut self) {
        self.mean_maxsim = 0.5;
        self.std_maxsim = 0.25;
        self.n_samples = 0;
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}
```

**Test Criteria**:
- TC-E12-01: Identical token sequences return Delta-S approx 0.0
- TC-E12-02: Completely different tokens return Delta-S approx 1.0
- TC-E12-03: Partial token overlap returns moderate Delta-S
- TC-E12-04: Variable token counts handled correctly
- TC-E12-05: Empty history returns Delta-S = 1.0
- TC-E12-06: Output always in [0.0, 1.0], no NaN/Infinity

---

### A.7 Updated Factory Routing

The `EmbedderEntropyFactory::create()` method should be updated to route to specialized implementations:

```rust
impl EmbedderEntropyFactory {
    pub fn create(embedder: Embedder, config: &SurpriseConfig) -> Box<dyn EmbedderEntropy> {
        match embedder {
            // Specialized implementations
            Embedder::Semantic => Box::new(GmmMahalanobisEntropy::from_config(config)),
            Embedder::Causal => Box::new(AsymmetricKnnEntropy::new(config.k_neighbors)
                .with_direction_modifiers(
                    config.causal_cause_to_effect_mod,
                    config.causal_effect_to_cause_mod,
                )),
            Embedder::Hdc => Box::new(HammingPrototypeEntropy::new(config.hdc_max_prototypes)
                .with_threshold(config.hdc_binarization_threshold)),
            Embedder::KeywordSplade => Box::new(JaccardActiveEntropy::new()
                .with_threshold(config.splade_activation_threshold)
                .with_smoothing(config.splade_smoothing)),

            // NEW: Specialized implementations per constitution.yaml
            Embedder::Sparse => Box::new(SparseIdfJaccardEntropy::new(
                config.splade_activation_threshold)),
            Embedder::Code => Box::new(HybridGmmKnnEntropy::new(config)),
            Embedder::Multimodal => Box::new(CrossModalKnnEntropy::new(config)),
            Embedder::Entity => Box::new(TransEEntropy::new(384)), // E11 dim
            Embedder::LateInteraction => Box::new(MaxSimTokenEntropy::new(128)), // Token dim

            // KNN fallback for temporal and graph embedders
            Embedder::TemporalRecent
            | Embedder::TemporalPeriodic
            | Embedder::TemporalPositional
            | Embedder::Graph => Box::new(DefaultKnnEntropy::from_config(embedder, config)),
        }
    }
}
```

---

### A.8 Acceptance Criteria Summary

| Embedder | Specialized Method | Acceptance Criteria |
|----------|-------------------|---------------------|
| E1 | GMM+Mahalanobis | DONE - existing implementation |
| E2-4, E8 | KNN | DONE - DefaultKnnEntropy |
| E5 | Asymmetric KNN | DONE - existing implementation |
| E6 | IDF/Jaccard | NEW - SparseIdfJaccardEntropy |
| E7 | GMM+KNN Hybrid | NEW - HybridGmmKnnEntropy |
| E9 | Hamming | DONE - existing implementation |
| E10 | Cross-modal KNN | NEW - CrossModalKnnEntropy |
| E11 | TransE | NEW - TransEEntropy |
| E12 | Token MaxSim | NEW - MaxSimTokenEntropy |
| E13 | Jaccard | DONE - JaccardActiveEntropy |

**Validation Checklist**:
- [ ] All 13 embedders have specialized or appropriate fallback implementation
- [ ] All implementations conform to `EmbedderEntropy` trait
- [ ] All outputs clamped to [0.0, 1.0] per AP-10
- [ ] No NaN/Infinity values possible
- [ ] Thread-safe (Send + Sync)
- [ ] Unit tests cover all boundary conditions
- [ ] Performance meets < 5ms per embedder target
