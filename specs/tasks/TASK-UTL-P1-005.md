# TASK-UTL-P1-005: Implement TransEEntropy for E11 (Entity/KnowledgeGraph)

**Priority:** P1
**Status:** pending
**Spec Reference:** SPEC-UTL-003
**Estimated Effort:** 3-4 hours
**Implements:** REQ-UTL-003-05, REQ-UTL-003-06

---

## Summary

Create a specialized entropy calculator for E11 (Entity) embeddings using TransE distance metrics. Entity embeddings represent knowledge graph relations where entities are connected by relation vectors. The TransE model computes distance as `||h + r - t||` where h=head, r=relation, t=tail.

**Constitution Reference (line 801):**
```yaml
E11: "TransE: ΔS=||h+r-t||"
```

---

## Input Context Files

Read these files before implementation:

| File | Purpose |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/mod.rs` | `EmbedderEntropy` trait definition |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs` | Factory routing (to be updated) |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/default_knn.rs` | Reference: KNN baseline |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/teleological/embedder.rs` | `Embedder::Entity` definition (index 10) |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/config.rs` | `SurpriseConfig` for parameters |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/error.rs` | `UtlError`, `UtlResult` types |

---

## Background: TransE Model

TransE (Translating Embeddings) models knowledge graph relationships as translations in embedding space:
- Each entity has an embedding vector
- Each relation has a translation vector
- For a valid triple (h, r, t): h + r ≈ t
- Distance: d(h,r,t) = ||h + r - t||

For entropy computation:
- Current embedding represents the "head + relation" context
- History embeddings represent potential "tail" entities
- Low TransE distance = familiar entity pattern = low entropy
- High TransE distance = novel entity relationship = high entropy

---

## Definition of Done

### 1. Create New File

**Path:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/transe.rs`

**Exact Signatures Required:**

```rust
/// E11 (Entity) entropy using TransE distance.
///
/// TransE models knowledge graph relationships as translations:
/// For triple (head, relation, tail): head + relation ≈ tail
///
/// # Algorithm
///
/// 1. Parse current embedding as (head_context, relation_context)
///    - First half = entity head context
///    - Second half = relation context
/// 2. For each history embedding:
///    a. Parse as tail_context
///    b. Compute TransE distance: ||head + relation - tail||
/// 3. Average the top-k smallest distances (most similar patterns)
/// 4. Normalize via sigmoid, clamp to [0, 1]
///
/// # Constitution Reference
/// E11: "TransE: ΔS=||h+r-t||"
#[derive(Debug, Clone)]
pub struct TransEEntropy {
    /// Dimension split point for head vs relation.
    /// Default: dim / 2 (E11 is 384D, so split at 192)
    split_point: usize,
    /// L-norm for distance (1 = Manhattan, 2 = Euclidean).
    /// TransE typically uses L1 or L2. Default: 2 (L2 norm)
    norm: u8,
    /// Running mean for distance normalization.
    running_mean: f32,
    /// Running variance for distance normalization.
    running_variance: f32,
    /// Number of samples seen.
    sample_count: usize,
    /// EMA alpha for updating statistics.
    ema_alpha: f32,
    /// Margin for negative sampling (if used). Default: 1.0
    margin: f32,
}

impl TransEEntropy {
    /// Create a new TransE entropy calculator.
    pub fn new() -> Self;

    /// Create with a specific norm (1 for L1, 2 for L2).
    pub fn with_norm(norm: u8) -> Self;

    /// Create from SurpriseConfig.
    pub fn from_config(config: &SurpriseConfig) -> Self;

    /// Set the dimension split point.
    /// For 384D embeddings, default is 192 (half).
    pub fn with_split_point(self, split_point: usize) -> Self;

    /// Set the margin for contrastive scoring.
    pub fn with_margin(self, margin: f32) -> Self;

    /// Extract head context from embedding.
    /// Returns first `split_point` dimensions.
    fn extract_head(&self, embedding: &[f32]) -> &[f32];

    /// Extract relation context from embedding.
    /// Returns dimensions from `split_point` to end.
    fn extract_relation(&self, embedding: &[f32]) -> &[f32];

    /// Compute TransE distance: ||h + r - t||
    ///
    /// # Arguments
    /// * `head` - Head entity embedding
    /// * `relation` - Relation embedding
    /// * `tail` - Tail entity embedding
    ///
    /// # Returns
    /// L-norm distance (L1 or L2 based on configuration)
    fn compute_transe_distance(
        &self,
        head: &[f32],
        relation: &[f32],
        tail: &[f32],
    ) -> f32;

    /// Sigmoid normalization function.
    fn sigmoid(x: f32) -> f32;
}

impl Default for TransEEntropy {
    fn default() -> Self;
}

impl EmbedderEntropy for TransEEntropy {
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

### 2. Update Module Exports

**File:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/mod.rs`

Add:
```rust
mod transe;
pub use transe::TransEEntropy;
```

### 3. Update Factory Routing

**File:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs`

Change the `Embedder::Entity` arm from:
```rust
Embedder::Entity => Box::new(DefaultKnnEntropy::from_config(embedder, config))
```

To:
```rust
Embedder::Entity => Box::new(TransEEntropy::from_config(config))
```

### 4. Required Tests

Implement in `transe.rs` tests module:

| Test Name | Description |
|-----------|-------------|
| `test_transe_empty_history_returns_one` | Empty history = max surprise |
| `test_transe_identical_returns_low` | Same embedding = low ΔS |
| `test_transe_perfect_translation` | h + r = t exactly -> ΔS ≈ 0 |
| `test_transe_orthogonal_returns_high` | Unrelated entities = high ΔS |
| `test_transe_empty_input_error` | Empty current = EmptyInput error |
| `test_transe_embedder_type` | Returns `Embedder::Entity` |
| `test_transe_valid_range` | All outputs in [0, 1] |
| `test_transe_no_nan_infinity` | No NaN/Infinity outputs |
| `test_transe_l1_vs_l2_norm` | Different norms produce different distances |
| `test_transe_from_config` | Config values applied |
| `test_transe_head_extraction` | First half extracted correctly |
| `test_transe_relation_extraction` | Second half extracted correctly |
| `test_transe_distance_formula` | ||h + r - t|| computed correctly |
| `test_transe_reset` | State clears properly |

---

## Validation Criteria

| Check | Command | Expected |
|-------|---------|----------|
| Compiles | `cargo build -p context-graph-utl` | Success |
| Tests pass | `cargo test -p context-graph-utl transe` | All tests pass |
| No warnings | `cargo clippy -p context-graph-utl -- -D warnings` | No warnings |
| Factory routes correctly | Run factory test | E11 -> TransEEntropy |

---

## Implementation Notes

1. **Embedding Structure**: E11 entity embeddings are 384D from MiniLM. The TransE interpretation splits this:
   - Dimensions 0-191: Head entity context
   - Dimensions 192-383: Relation context
   - History embeddings serve as potential tails

2. **TransE Formula**:
   ```
   d(h, r, t) = ||h + r - t||_p
   ```
   Where p is the L-norm (1 for Manhattan, 2 for Euclidean).

3. **Entropy Interpretation**:
   - Low TransE distance: The current (head, relation) pair has seen similar tails before
   - High TransE distance: Novel relationship pattern = high surprise

4. **L-Norm Selection**:
   - L1 (Manhattan): More robust to outliers
   - L2 (Euclidean): Smoother gradient, typical for TransE
   - Default to L2 for consistency with original TransE paper

5. **Config Fields**: May need to add to `SurpriseConfig`:
   - `entity_transe_norm: u8` (default 2)
   - `entity_split_ratio: f32` (default 0.5)

---

## Mathematical Background

TransE (Bordes et al., 2013) models relationships as translations:

Given a knowledge graph with entities E and relations R:
- Each entity e has embedding **e** in R^d
- Each relation r has embedding **r** in R^d
- For valid triple (h, r, t): **h** + **r** ≈ **t**

Scoring function:
```
f(h, r, t) = -||h + r - t||_{L1 or L2}
```

Training minimizes:
```
L = Σ_{(h,r,t)∈S} Σ_{(h',r,t')∈S'} [γ + d(h+r, t) - d(h'+r, t')]_+
```

Where γ is margin and S' are corrupted (negative) triples.

For entropy:
```
ΔS = σ(d_normalized)
```
Where d_normalized = (d - μ) / σ with running statistics.

---

## Rollback Plan

If implementation causes issues:
1. Revert factory routing to `DefaultKnnEntropy` for `Embedder::Entity`
2. Remove `transe.rs` from module exports
3. Delete `transe.rs` file

---

## Related Tasks

- **TASK-UTL-P1-003**: JaccardCodeEntropy for E7
- **TASK-UTL-P1-004**: CrossModalEntropy for E10
- **TASK-UTL-P1-006**: MaxSimTokenEntropy for E12
