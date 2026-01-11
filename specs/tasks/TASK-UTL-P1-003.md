# TASK-UTL-P1-003: Implement HybridGmmKnnEntropy for E7 (Code)

**Priority:** P1
**Status:** pending
**Spec Reference:** SPEC-UTL-003
**Estimated Effort:** 4-5 hours
**Implements:** REQ-UTL-003-01, REQ-UTL-003-02

---

## Summary

Create a specialized entropy calculator for E7 (Code) embeddings using a hybrid GMM+KNN approach. Code embeddings from Qodo-Embed (1536D) represent code structure and semantics. The hybrid method combines Gaussian Mixture Model cluster membership probability with k-nearest neighbor distance for robust entropy estimation.

**Constitution Reference (line 798):**
```yaml
E7: "GMM+KNN: ΔS=0.5×GMM+0.5×KNN"
```

---

## Input Context Files

Read these files before implementation:

| File | Purpose |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/mod.rs` | `EmbedderEntropy` trait definition |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs` | Factory routing (to be updated) |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/gmm_mahalanobis.rs` | Reference: GMM implementation for E1 |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/default_knn.rs` | Reference: KNN implementation |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/teleological/embedder.rs` | `Embedder::Code` definition |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/config.rs` | `SurpriseConfig` for parameters |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/error.rs` | `UtlError`, `UtlResult` types |

---

## Background: Hybrid GMM+KNN

The hybrid approach combines two complementary entropy measures:

1. **GMM Component (Global)**: Measures how well the embedding fits within learned clusters
   - Uses Gaussian Mixture Model fitted on history
   - ΔS_GMM = 1 - P(e|GMM) where P is cluster membership probability
   - Captures global code structure patterns

2. **KNN Component (Local)**: Measures local neighborhood density
   - Uses k-nearest neighbor distances
   - ΔS_KNN = σ((d_k - μ) / σ) where d_k is k-th neighbor distance
   - Captures fine-grained code similarity

3. **Hybrid Formula**: ΔS = 0.5 × ΔS_GMM + 0.5 × ΔS_KNN

---

## Definition of Done

### 1. Create New File

**Path:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/hybrid_gmm_knn.rs`

**Exact Signatures Required:**

```rust
/// E7 (Code) entropy using hybrid GMM+KNN approach.
///
/// Combines Gaussian Mixture Model cluster membership with k-nearest
/// neighbor distance for robust entropy estimation.
///
/// # Algorithm
///
/// 1. Fit GMM on history embeddings (or use cached fit)
/// 2. Compute GMM component: ΔS_GMM = 1 - P(e|GMM)
/// 3. Compute KNN component: ΔS_KNN = σ((d_k - μ) / σ)
/// 4. Combine: ΔS = 0.5 × ΔS_GMM + 0.5 × ΔS_KNN
///
/// # Constitution Reference
/// E7: "GMM+KNN: ΔS=0.5×GMM+0.5×KNN"
#[derive(Debug, Clone)]
pub struct HybridGmmKnnEntropy {
    /// GMM component weight. Constitution default: 0.5
    gmm_weight: f32,
    /// KNN component weight. Constitution default: 0.5
    knn_weight: f32,
    /// Number of GMM components. Default: 5
    n_components: usize,
    /// k for KNN component. Default: 5
    k_neighbors: usize,
    /// Cached GMM model (fitted on history)
    gmm_model: Option<GmmModel>,
    /// Running statistics for KNN normalization
    knn_mean: f32,
    knn_variance: f32,
    /// Sample count for statistics
    sample_count: usize,
    /// EMA alpha for updating statistics
    ema_alpha: f32,
}

impl HybridGmmKnnEntropy {
    /// Create a new hybrid GMM+KNN entropy calculator.
    pub fn new() -> Self;

    /// Create with custom weights.
    ///
    /// # Arguments
    /// * `gmm_weight` - Weight for GMM component (default 0.5)
    /// * `knn_weight` - Weight for KNN component (default 0.5)
    ///
    /// # Panics
    /// Panics if weights don't sum to approximately 1.0
    pub fn with_weights(gmm_weight: f32, knn_weight: f32) -> Self;

    /// Create from SurpriseConfig.
    pub fn from_config(config: &SurpriseConfig) -> Self;

    /// Set the number of GMM components.
    pub fn with_n_components(self, n: usize) -> Self;

    /// Set the k for KNN.
    pub fn with_k_neighbors(self, k: usize) -> Self;

    /// Compute GMM component of entropy.
    ///
    /// # Algorithm
    /// 1. If no GMM model, fit on history
    /// 2. Compute log-likelihood of current embedding
    /// 3. Convert to probability: P = exp(log_likelihood)
    /// 4. ΔS_GMM = 1 - P
    ///
    /// # Returns
    /// GMM entropy component in [0, 1]
    fn compute_gmm_component(
        &mut self,
        current: &[f32],
        history: &[Vec<f32>],
    ) -> f32;

    /// Compute KNN component of entropy.
    ///
    /// # Algorithm
    /// 1. Compute distance to k-th nearest neighbor
    /// 2. Normalize: z = (d_k - μ) / σ
    /// 3. Apply sigmoid: ΔS_KNN = σ(z)
    ///
    /// # Returns
    /// KNN entropy component in [0, 1]
    fn compute_knn_component(
        &self,
        current: &[f32],
        history: &[Vec<f32>],
    ) -> f32;

    /// Fit GMM model on history embeddings.
    fn fit_gmm(&mut self, history: &[Vec<f32>]);

    /// Compute log-likelihood under GMM.
    fn gmm_log_likelihood(&self, embedding: &[f32]) -> f32;

    /// Sigmoid function for normalization.
    fn sigmoid(x: f32) -> f32;
}

impl Default for HybridGmmKnnEntropy {
    fn default() -> Self;
}

impl EmbedderEntropy for HybridGmmKnnEntropy {
    fn compute_delta_s(
        &self,
        current: &[f32],
        history: &[Vec<f32>],
        k: usize,
    ) -> UtlResult<f32>;

    fn embedder_type(&self) -> Embedder;

    fn reset(&mut self);
}
```

### 2. GMM Model Structure

```rust
/// Gaussian Mixture Model for entropy computation
#[derive(Debug, Clone)]
pub struct GmmModel {
    /// Number of components
    n_components: usize,
    /// Mixing weights (sum to 1)
    weights: Vec<f32>,
    /// Component means [n_components x dim]
    means: Vec<Vec<f32>>,
    /// Component covariances (diagonal for efficiency)
    covariances: Vec<Vec<f32>>,
    /// Is model fitted?
    is_fitted: bool,
}

impl GmmModel {
    /// Create new unfitted GMM
    pub fn new(n_components: usize) -> Self;

    /// Fit GMM using EM algorithm
    ///
    /// # Arguments
    /// * `data` - Training embeddings
    /// * `max_iter` - Maximum EM iterations (default 100)
    /// * `tol` - Convergence tolerance (default 1e-4)
    pub fn fit(&mut self, data: &[Vec<f32>], max_iter: usize, tol: f32);

    /// Compute log-likelihood of a sample
    pub fn log_likelihood(&self, sample: &[f32]) -> f32;

    /// Predict cluster membership probabilities
    pub fn predict_proba(&self, sample: &[f32]) -> Vec<f32>;
}
```

### 3. Update Module Exports

**File:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/mod.rs`

Add:
```rust
mod hybrid_gmm_knn;
pub use hybrid_gmm_knn::{HybridGmmKnnEntropy, GmmModel};
```

### 4. Update Factory Routing

**File:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs`

Change the `Embedder::Code` arm from:
```rust
Embedder::Code => Box::new(DefaultKnnEntropy::from_config(embedder, config))
```

To:
```rust
Embedder::Code => Box::new(HybridGmmKnnEntropy::from_config(config))
```

### 5. Required Tests

Implement in `hybrid_gmm_knn.rs` tests module:

| Test Name | Description |
|-----------|-------------|
| `test_hybrid_empty_history_returns_one` | Empty history = max surprise |
| `test_hybrid_identical_returns_low` | Same embedding = low ΔS |
| `test_hybrid_distant_returns_high` | Far from all clusters = high ΔS |
| `test_hybrid_weight_balance` | Verify 0.5 + 0.5 = 1.0 |
| `test_hybrid_gmm_component_range` | GMM component in [0, 1] |
| `test_hybrid_knn_component_range` | KNN component in [0, 1] |
| `test_hybrid_empty_input_error` | Empty current = EmptyInput error |
| `test_hybrid_embedder_type` | Returns `Embedder::Code` |
| `test_hybrid_valid_range` | All outputs in [0, 1] |
| `test_hybrid_no_nan_infinity` | No NaN/Infinity outputs |
| `test_hybrid_from_config` | Config values applied |
| `test_hybrid_gmm_fit` | GMM fits on history |
| `test_hybrid_gmm_log_likelihood` | Log-likelihood computed correctly |
| `test_hybrid_reset` | State clears properly |
| `test_gmm_model_fit_em` | EM algorithm converges |
| `test_gmm_model_predict_proba` | Probabilities sum to 1 |

---

## Validation Criteria

| Check | Command | Expected |
|-------|---------|----------|
| Compiles | `cargo build -p context-graph-utl` | Success |
| Tests pass | `cargo test -p context-graph-utl hybrid_gmm_knn` | All tests pass |
| No warnings | `cargo clippy -p context-graph-utl -- -D warnings` | No warnings |
| Factory routes correctly | Run factory test | E7 -> HybridGmmKnnEntropy |

---

## Implementation Notes

1. **GMM Fitting**: The GMM should be fitted lazily on first call or when history changes significantly. Consider:
   - Fit when history size reaches threshold (e.g., 50 samples)
   - Refit periodically or when log-likelihood drops
   - Use diagonal covariance for efficiency (1536D is high)

2. **Dimensionality Reduction**: For 1536D embeddings, consider PCA to reduce before GMM fitting:
   - Reduce to 64-128 dimensions
   - Preserves main variance
   - Faster GMM operations

3. **EM Algorithm**: Simplified diagonal-covariance GMM:
   ```
   E-step: γ_ik = w_k × N(x_i | μ_k, Σ_k) / Σ_j w_j × N(x_i | μ_j, Σ_j)
   M-step: μ_k = Σ_i γ_ik × x_i / Σ_i γ_ik
           σ²_k = Σ_i γ_ik × (x_i - μ_k)² / Σ_i γ_ik
           w_k = Σ_i γ_ik / N
   ```

4. **Config Fields**: Add to `SurpriseConfig`:
   - `code_gmm_weight: f32` (default 0.5)
   - `code_knn_weight: f32` (default 0.5)
   - `code_n_components: usize` (default 5)
   - `code_k_neighbors: usize` (default 5)

5. **Caching**: GMM model should be cached and reused across calls. Invalidate when:
   - History changes significantly (>10% new samples)
   - Explicit reset() called

---

## Mathematical Background

**GMM Component:**
```
P(x) = Σ_k w_k × N(x | μ_k, Σ_k)
ΔS_GMM = 1 - min(1, P(x))  // Clamp probability
```

**KNN Component:**
```
d_k = distance to k-th nearest neighbor
z = (d_k - μ) / σ  // Normalize with running stats
ΔS_KNN = σ(z) = 1 / (1 + exp(-z))
```

**Hybrid Formula:**
```
ΔS = α × ΔS_GMM + β × ΔS_KNN
where α = β = 0.5 per constitution
```

---

## Rollback Plan

If implementation causes issues:
1. Revert factory routing to `DefaultKnnEntropy` for `Embedder::Code`
2. Remove `hybrid_gmm_knn.rs` from module exports
3. Delete `hybrid_gmm_knn.rs` file

---

## Related Tasks

- **TASK-UTL-P1-004**: CrossModalEntropy for E10
- **TASK-UTL-P1-005**: TransEEntropy for E11
- **TASK-UTL-P1-006**: MaxSimTokenEntropy for E12
