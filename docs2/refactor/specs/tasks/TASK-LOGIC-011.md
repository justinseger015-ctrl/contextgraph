# TASK-LOGIC-011: RRF Fusion Implementation

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-LOGIC-011 |
| **Title** | Reciprocal Rank Fusion Implementation |
| **Status** | :white_circle: todo |
| **Layer** | Logic |
| **Sequence** | 21 |
| **Estimated Days** | 1.5 |
| **Complexity** | Medium |

## Implements

- **REQ-SEARCH-07**: Multi-space ranking fusion
- **ARCH-04**: Entry-point discovery with reranking

## Dependencies

| Task | Reason |
|------|--------|
| TASK-LOGIC-001 | Dense similarity scores |
| TASK-LOGIC-002 | Sparse similarity scores |
| TASK-LOGIC-003 | Token-level similarity scores |

## Objective

Implement Reciprocal Rank Fusion (RRF) to combine rankings from multiple embedding spaces into a single unified ranking for multi-space search aggregation.

## Context

RRF Formula: `RRF(d) = Σ 1/(k + rank_i(d))`

Where:
- `k` is a constant (typically 60) that reduces the impact of high rankings
- `rank_i(d)` is the rank of document `d` in ranking `i`

RRF is crucial for:
- Stage 3 of the 5-stage pipeline (Multi-space rerank)
- Combining results from different embedders
- Weighted fusion when some spaces are more relevant

## Scope

### In Scope

- `RRFFusion` struct with configurable `k` parameter
- Standard RRF fusion across multiple rankings
- Weighted RRF fusion with per-source weights
- Normalization of fused scores
- Handling of missing documents in some rankings

### Out of Scope

- Alternative fusion methods (CombSUM, CombMNZ)
- Learning-to-rank reranking
- Dynamic k optimization

## Definition of Done

### Signatures

```rust
// crates/context-graph-core/src/teleology/similarity/fusion.rs

use uuid::Uuid;

/// Reciprocal Rank Fusion for combining multi-space rankings
pub struct RRFFusion {
    /// K parameter (typically 60)
    k: f32,
}

impl RRFFusion {
    /// Create new RRF with specified k parameter
    pub fn new(k: f32) -> Self;

    /// Create with default k=60
    pub fn default() -> Self {
        Self::new(60.0)
    }

    /// Fuse multiple rankings into single unified ranking
    ///
    /// # Arguments
    /// * `rankings` - Each inner vec is (doc_id, score) sorted by score descending
    ///
    /// # Returns
    /// Unified ranking sorted by RRF score descending
    pub fn fuse(&self, rankings: &[Vec<(Uuid, f32)>]) -> Vec<(Uuid, f32)>;

    /// Fuse with per-source weights
    ///
    /// # Arguments
    /// * `rankings` - Each tuple is (ranking, weight) where weight > 0
    ///
    /// # Returns
    /// Weighted unified ranking
    pub fn fuse_weighted(
        &self,
        rankings: &[(Vec<(Uuid, f32)>, f32)],
    ) -> Vec<(Uuid, f32)>;

    /// Fuse rankings by embedder type
    ///
    /// # Arguments
    /// * `rankings` - Map from embedder to ranking
    /// * `weights` - Optional per-embedder weights (defaults to 1.0)
    pub fn fuse_by_embedder(
        &self,
        rankings: &HashMap<Embedder, Vec<(Uuid, f32)>>,
        weights: Option<&HashMap<Embedder, f32>>,
    ) -> Vec<(Uuid, f32)>;
}

/// Configuration for RRF fusion
#[derive(Debug, Clone)]
pub struct RRFConfig {
    /// K parameter
    pub k: f32,
    /// Minimum number of rankings a doc must appear in
    pub min_rankings: usize,
    /// Whether to normalize final scores to [0, 1]
    pub normalize: bool,
    /// Maximum results to return
    pub max_results: usize,
}

impl Default for RRFConfig {
    fn default() -> Self {
        Self {
            k: 60.0,
            min_rankings: 1,
            normalize: true,
            max_results: 100,
        }
    }
}

/// Extended RRF with configuration
pub struct ConfigurableRRF {
    config: RRFConfig,
}

impl ConfigurableRRF {
    pub fn new(config: RRFConfig) -> Self;

    /// Fuse with full configuration options
    pub fn fuse(&self, rankings: &[Vec<(Uuid, f32)>]) -> Vec<(Uuid, f32)>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_basic() {
        let rrf = RRFFusion::new(60.0);
        let ranking1 = vec![
            (uuid!("a"), 0.9),
            (uuid!("b"), 0.8),
            (uuid!("c"), 0.7),
        ];
        let ranking2 = vec![
            (uuid!("b"), 0.95),
            (uuid!("c"), 0.85),
            (uuid!("a"), 0.75),
        ];

        let fused = rrf.fuse(&[ranking1, ranking2]);

        // 'b' should rank highest: 1/(60+2) + 1/(60+1) = 0.016 + 0.016 = 0.032
        // 'a' should be second: 1/(60+1) + 1/(60+3) = 0.016 + 0.016 = 0.032
        assert_eq!(fused[0].0, uuid!("b"));
    }

    #[test]
    fn test_rrf_weighted() {
        let rrf = RRFFusion::new(60.0);
        // ... weighted test
    }

    #[test]
    fn test_rrf_missing_docs() {
        let rrf = RRFFusion::new(60.0);
        // Doc appears in only one ranking - should still be included
        // ...
    }
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Fusion latency (100 docs, 13 rankings) | < 1ms |
| Memory (100 candidates) | < 50KB |
| Score precision | f32 sufficient |

## Verification

- [ ] Basic RRF produces correct scores for known inputs
- [ ] Weighted RRF applies weights correctly
- [ ] Documents missing from some rankings handled correctly
- [ ] Results sorted by RRF score descending
- [ ] Normalization produces [0, 1] scores when enabled
- [ ] Performance: < 1ms for 100 docs, 13 rankings

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/teleology/similarity/fusion.rs` | RRF implementation |
| Update `crates/context-graph-core/src/teleology/similarity/mod.rs` | Export fusion module |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| K parameter suboptimal | Low | Medium | Make configurable, benchmark |
| Score overflow | Very Low | Medium | Use f64 internally if needed |

## Traceability

- Source: Constitution ARCH-04 (Entry-point discovery pattern)
- Reference: TASK-LOGIC-008 Stage 3 (Multi-space rerank)
