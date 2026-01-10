# TASK-LOGIC-006: Multi-Embedder Parallel Search

```xml
<task_spec id="TASK-LOGIC-006" version="3.0">
<metadata>
  <title>Multi-Embedder Parallel Search with Score Aggregation</title>
  <status>DONE</status>
  <layer>logic</layer>
  <sequence>16</sequence>
  <implements>
    <requirement_ref>REQ-SEARCH-WEIGHTED-01</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="DONE">TASK-LOGIC-004</task_ref>
    <task_ref status="DONE">TASK-LOGIC-005</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <completed_on>2025-01-09</completed_on>
</metadata>

<context>
## What Was Actually Built

This task implemented `MultiEmbedderSearch` in `multi.rs` (NOT `WeightedFullSearch` in `weighted.rs` as originally specified). The implementation provides parallel search across multiple HNSW embedders with score normalization and aggregation.

## File Locations (Verified Against Codebase)

| File | Purpose | Line Count |
|------|---------|------------|
| `crates/context-graph-storage/src/teleological/search/multi.rs` | Multi-embedder search implementation | ~1600 |
| `crates/context-graph-storage/src/teleological/search/mod.rs` | Module exports | 110 |
| `crates/context-graph-storage/src/teleological/search/single.rs` | SingleEmbedderSearch (dependency) | ~900 |
| `crates/context-graph-storage/src/teleological/search/error.rs` | SearchError types | ~100 |
| `crates/context-graph-storage/src/teleological/search/result.rs` | Result types | ~150 |

## Implemented Types (in multi.rs)

### Enums
- `NormalizationStrategy` - 4 variants: `None`, `MinMax`, `ZScore`, `RankNorm`
- `AggregationStrategy` - 4 variants: `Max`, `Sum`, `Mean`, `WeightedSum`

### Structs
- `MultiEmbedderSearchConfig` - 6 fields: k, threshold, normalization, aggregation, embedder_weights, max_per_embedder
- `AggregatedHit` - id, aggregated_score, contributing_embedders (HashMap)
- `PerEmbedderResults` - embedder, hits, count, latency_us
- `MultiEmbedderSearchResults` - hits, per_embedder, total_embedders_searched, aggregation_strategy, latency_us, source_of_truth_verified
- `MultiEmbedderSearch` - single_search, config
- `MultiSearchBuilder` - queries, k, threshold, normalization, aggregation

## API Design (Actual Implementation)

```rust
// Builder pattern with HashMap<EmbedderIndex, Vec<f32>> for queries
let queries: HashMap<EmbedderIndex, Vec<f32>> = [
    (EmbedderIndex::E1Semantic, vec![0.5f32; 1024]),
    (EmbedderIndex::E8Graph, vec![0.5f32; 384]),
].into_iter().collect();

let results = MultiSearchBuilder::new(queries)
    .k(10)
    .threshold(Some(0.5))
    .normalization(NormalizationStrategy::MinMax)
    .aggregation(AggregationStrategy::Max)
    .execute(&multi_search)?;
```

## 12 HNSW-Capable Embedders (with Correct Dimensions)

| Index | Embedder | Dimension | Notes |
|-------|----------|-----------|-------|
| 0 | E1Semantic | 1024 | Primary semantic |
| 1 | E2TemporalRecent | 512 | Recency |
| 2 | E3TemporalPeriodic | 512 | Periodic patterns |
| 3 | E4TemporalPositional | 512 | Position-based |
| 4 | E5Causal | 768 | Causal (NOT 1024) |
| 5 | E7Code | 1536 | Code-specific (NOT 1024) |
| 6 | E8Graph | 384 | Graph structure |
| 7 | E9HDC | 1024 | Hyperdimensional (NOT 10000) |
| 8 | E10Multimodal | 768 | Cross-modal (NOT 1024) |
| 9 | E11Entity | 384 | Named entities |
| 10 | E1Matryoshka128 | 128 | Fast filtering |
| 11 | PurposeVector | 13 | Teleological |

## 3 Non-HNSW Embedders (NEVER Query via HNSW)

- **E6Sparse** - Requires inverted index with BM25
- **E12LateInteraction** - Requires ColBERT MaxSim token-level
- **E13Splade** - Requires inverted index with learned expansion

Querying these returns `SearchError::UnsupportedEmbedder`.
</context>

<test_verification>
## Tests That MUST Pass

```bash
# Run all multi.rs tests
cargo test -p context-graph-storage search::multi -- --nocapture

# Expected: 23 tests, all PASS
```

## Test Names (Verified Working)

1. `test_aggregated_hit_methods` - AggregatedHit helper methods
2. `test_aggregation_max` - Max takes highest score
3. `test_aggregation_mean` - Mean averages scores
4. `test_aggregation_sum` - Sum adds all scores
5. `test_aggregation_weighted_sum` - WeightedSum applies weights
6. `test_builder_add_query` - Builder API works
7. `test_dimension_mismatch_fails_fast` - Wrong dimension errors
8. `test_empty_indexes_return_empty` - Empty indexes handled
9. `test_empty_query_vector_fails_fast` - Empty query errors
10. `test_empty_queries_fails_fast` - Empty map errors
11. `test_full_state_verification` - **CRITICAL**: Verifies real data flow
12. `test_latency_is_recorded` - Latency tracking works
13. `test_multi_search_builder_fluent` - Fluent builder API
14. `test_nan_in_query_fails_fast` - NaN detection
15. `test_normalization_minmax` - MinMax scales [0,1]
16. `test_normalization_none` - None preserves scores
17. `test_normalization_ranknorm` - RankNorm uses 1/rank
18. `test_search_multiple_embedders_different_ids` - Cross-embedder search
19. `test_results_helper_methods` - Result struct methods
20. `test_search_single_embedder` - Single embedder mode
21. `test_search_multiple_embedders_same_id` - ID deduplication
22. `test_unsupported_embedder_fails_fast` - E6/E12/E13 rejected
23. `test_verification_comprehensive` - Type and method verification
</test_verification>

<validation_criteria>
## FAIL FAST Compliance (Verified)

All of these MUST error immediately, NOT return empty results:

| Input | Expected Error |
|-------|----------------|
| Empty queries map | `SearchError::Store("FAIL FAST: queries map is empty")` |
| Empty query vector | `SearchError::InvalidVector("empty query")` |
| NaN in query | `SearchError::InvalidVector("NaN at index N")` |
| Inf in query | `SearchError::InvalidVector("Inf at index N")` |
| Wrong dimension | `SearchError::DimensionMismatch { ... }` |
| E6Sparse | `SearchError::UnsupportedEmbedder(E6Sparse)` |
| E12LateInteraction | `SearchError::UnsupportedEmbedder(E12LateInteraction)` |
| E13Splade | `SearchError::UnsupportedEmbedder(E13Splade)` |
| k = 0 | `SearchError::InvalidParameter("k must be > 0")` |

## NO Fallbacks. NO Recovery. NO Silent Failures.
</validation_criteria>

<source_of_truth>
## Full State Verification Protocol

After ANY implementation, verify against the actual source of truth:

### 1. Registry Verification
```rust
// BEFORE: Check registry state
println!("BEFORE: registry.len() = {}", registry.get(embedder).map(|i| i.len()).unwrap_or(0));

// AFTER: Verify inserts persisted
println!("AFTER: registry.len() = {}", registry.get(embedder).map(|i| i.len()).unwrap_or(0));
assert!(after > before, "Insert must increase count");
```

### 2. Search Verification
```rust
// Execute search
let results = multi_search.search(&queries, &config)?;

// VERIFY: Results contain expected IDs
for hit in &results.hits {
    assert!(expected_ids.contains(&hit.id), "Unknown ID in results");
}

// VERIFY: Scores are valid
for hit in &results.hits {
    assert!(!hit.aggregated_score.is_nan(), "Score is NaN");
    assert!(hit.aggregated_score >= 0.0, "Score is negative");
}

// VERIFY: Results are sorted descending
for window in results.hits.windows(2) {
    assert!(window[0].aggregated_score >= window[1].aggregated_score);
}
```

### 3. Edge Case Verification

| Edge Case | Before State | Action | Expected After State |
|-----------|--------------|--------|---------------------|
| Empty index | len=0 | search | Empty results, no error |
| Single match | len=1 | search | 1 result |
| Identical vectors | 2 with same embedding | search for exact | Both appear, score ~1.0 |
| Orthogonal vectors | v1 perpendicular v2 | search for v1 | v2 has score ~0.0 |
</source_of_truth>

<architecture_constraints>
From constitution.yaml:

- **ARCH-01**: TeleologicalArray is atomic - all 13 embeddings stored/retrieved together
- **ARCH-02**: Apples-to-apples comparison - E1 compares with E1, NEVER cross-embedder
- **FAIL FAST**: All errors are fatal. No recovery attempts. No fallbacks.
- **Performance**: inject_context less than 25ms, full retrieval less than 30ms
</architecture_constraints>

<implementation_notes>
## What Changed From Original Spec

| Original Spec | Actual Implementation | Reason |
|---------------|----------------------|--------|
| `weighted.rs` | `multi.rs` | Better naming |
| `WeightedFullSearch` | `MultiEmbedderSearch` | Clearer purpose |
| `[&[f32]; 13]` queries | `HashMap<EmbedderIndex, Vec<f32>>` | More flexible, search any subset |
| `ParallelStrategy` enum | Always parallel via rayon | Simpler, sufficient |
| async/await | Synchronous rayon | CPU-bound work, simpler |
| `FusionMethod` enum | `AggregationStrategy` enum | Same concepts, better naming |

## Dependencies Added

`crates/context-graph-storage/Cargo.toml`:
```toml
rayon = "1.10"
```

## Module Exports (mod.rs)

```rust
// Re-export multi-embedder search types
pub use multi::{
    MultiEmbedderSearch,
    MultiSearchBuilder,
    MultiEmbedderSearchConfig,
    NormalizationStrategy,
    AggregationStrategy,
    AggregatedHit,
    PerEmbedderResults,
    MultiEmbedderSearchResults,
};
```
</implementation_notes>

<test_commands>
```bash
# Compile check
cargo check -p context-graph-storage

# Run multi.rs tests (should show 23 tests, all PASS)
cargo test -p context-graph-storage search::multi -- --nocapture

# Run with single thread for deterministic output
cargo test -p context-graph-storage search::multi -- --test-threads=1 --nocapture

# Clippy validation
cargo clippy -p context-graph-storage -- -D warnings
```
</test_commands>

<common_mistakes>
## DO NOT Do These

1. **DO NOT** create `weighted.rs` - `multi.rs` already exists and is complete
2. **DO NOT** use `[&[f32]; 13]` - use `HashMap<EmbedderIndex, Vec<f32>>`
3. **DO NOT** query E6/E12/E13 via HNSW - they use different algorithms
4. **DO NOT** use wrong dimensions (see table above)
5. **DO NOT** swallow errors - propagate all `SearchError` variants
6. **DO NOT** use async - the implementation is synchronous with rayon
7. **DO NOT** modify SingleEmbedderSearch - it's a dependency, not part of this task
</common_mistakes>
</task_spec>
```
