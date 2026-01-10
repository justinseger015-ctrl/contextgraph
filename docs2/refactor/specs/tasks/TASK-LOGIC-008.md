# TASK-LOGIC-008: 5-Stage Retrieval Pipeline

```xml
<task_spec id="TASK-LOGIC-008" version="4.0">
<metadata>
  <title>5-Stage Retrieval Pipeline with Progressive Filtering</title>
  <status>in_progress</status>
  <layer>logic</layer>
  <sequence>18</sequence>
  <implements>
    <requirement_ref>REQ-SEARCH-PIPELINE-01</requirement_ref>
    <requirement_ref>REQ-LATENCY-01</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="DONE">TASK-LOGIC-005</task_ref>
    <task_ref status="DONE">TASK-LOGIC-006</task_ref>
    <task_ref status="DONE">TASK-LOGIC-007</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
</metadata>

<context>
## ⚠️ CRITICAL: CURRENT STATE (Verified 2026-01-09)

### pipeline.rs EXISTS AND COMPILES

The file `crates/context-graph-storage/src/teleological/search/pipeline.rs` **already exists** with:
- **1934 lines** of code
- **21 unit tests** (all passing)
- **Full implementation** of the 5-stage pipeline structure
- **Exports in mod.rs** already configured

### WHAT IS IMPLEMENTED (DO NOT RE-IMPLEMENT)

| Component | Status | Details |
|-----------|--------|---------|
| `PipelineError` enum | ✅ DONE | Stage, Timeout, MissingQuery, EmptyCandidates errors |
| `PipelineStage` enum | ✅ DONE | SpladeFilter, MatryoshkaAnn, RrfRerank, AlignmentFilter, MaxSimRerank |
| `StageConfig` struct | ✅ DONE | enabled, candidate_multiplier, min_score_threshold, max_latency_ms |
| `PipelineConfig` struct | ✅ DONE | stages[5], k, purpose_vector, rrf_k |
| `PipelineCandidate` struct | ✅ DONE | id, score, stage_scores |
| `StageResult` struct | ✅ DONE | candidates, latency_us, candidates_in, candidates_out, stage |
| `PipelineResult` struct | ✅ DONE | results, stage_results, total_latency_us, stages_executed, alignment_verified |
| `RetrievalPipeline` struct | ✅ DONE | single_search, multi_search, config, splade_index, token_storage |
| `PipelineBuilder` struct | ✅ DONE | Builder pattern for queries |
| `TokenStorage` trait | ✅ DONE | Trait for ColBERT token storage |
| `SpladeIndex` trait | ✅ DONE | Trait for SPLADE inverted index |
| `InMemoryTokenStorage` | ✅ DONE | Test implementation |
| `InMemorySpladeIndex` | ✅ DONE | Test implementation |
| Stage 1: SPLADE filter | ✅ DONE | Uses SpladeIndex trait (NOT HNSW) |
| Stage 2: Matryoshka ANN | ✅ DONE | Uses E1Matryoshka128 HNSW |
| Stage 3: RRF rerank | ✅ DONE | Multi-space with RRF scoring |
| Stage 4: Alignment filter | ✅ DONE | PurposeVector filtering |
| Stage 5: MaxSim rerank | ✅ DONE | Uses TokenStorage trait (NOT HNSW) |
| Unit tests | ✅ DONE | 21 tests passing |

### Current Test Results (VERIFIED)

```
cargo test -p context-graph-storage teleological::search::pipeline
running 21 tests
test teleological::search::pipeline::tests::test_builder_basic ... ok
test teleological::search::pipeline::tests::test_builder_chain ... ok
test teleological::search::pipeline::tests::test_default_config ... ok
test teleological::search::pipeline::tests::test_empty_index_returns_empty ... ok
test teleological::search::pipeline::tests::test_execute_stages_selective ... ok
test teleological::search::pipeline::tests::test_fail_fast_dimension_mismatch ... ok
test teleological::search::pipeline::tests::test_fail_fast_empty_query ... ok
test teleological::search::pipeline::tests::test_fail_fast_inf_in_query ... ok
test teleological::search::pipeline::tests::test_fail_fast_nan_in_query ... ok
test teleological::search::pipeline::tests::test_funnel_shape ... ok
test teleological::search::pipeline::tests::test_full_pipeline_execution ... ok
test teleological::search::pipeline::tests::test_latency_tracking ... ok
test teleological::search::pipeline::tests::test_pipeline_creation ... ok
test teleological::search::pipeline::tests::test_pipeline_stage_enum ... ok
test teleological::search::pipeline::tests::test_rrf_score_calculation ... ok
test teleological::search::pipeline::tests::test_single_candidate ... ok
test teleological::search::pipeline::tests::test_stage_1_not_hnsw ... ok
test teleological::search::pipeline::tests::test_stage_5_not_hnsw ... ok
test teleological::search::pipeline::tests::test_stage_config_custom ... ok
test teleological::search::pipeline::tests::test_stage_scores_tracked ... ok
test teleological::search::pipeline::tests::test_verification_log ... ok

test result: ok. 21 passed; 0 failed; 0 ignored
```

### File Locations (VERIFIED)

| File | Purpose | Status | Lines |
|------|---------|--------|-------|
| `crates/context-graph-storage/src/teleological/search/pipeline.rs` | **PIPELINE** | **EXISTS** | 1934 |
| `crates/context-graph-storage/src/teleological/search/mod.rs` | Module exports | EXISTS | ~150 |
| `crates/context-graph-storage/src/teleological/search/single.rs` | SingleEmbedderSearch | EXISTS | ~900 |
| `crates/context-graph-storage/src/teleological/search/multi.rs` | MultiEmbedderSearch | EXISTS | ~1600 |
| `crates/context-graph-storage/src/teleological/search/matrix.rs` | MatrixStrategySearch | EXISTS | ~1478 |
| `crates/context-graph-storage/src/teleological/search/error.rs` | SearchError types | EXISTS | ~100 |
| `crates/context-graph-storage/src/teleological/search/result.rs` | Result types | EXISTS | ~50 |
</context>

<objective>
## REMAINING WORK

The core pipeline structure is implemented. The remaining work is **integration**:

### 1. Real SPLADE Index Integration (Stage 1)

The pipeline defines a `SpladeIndex` trait. Create a real implementation that uses inverted index infrastructure:

```rust
/// Trait for SPLADE inverted index operations. NOT HNSW.
pub trait SpladeIndex: Send + Sync {
    fn search(&self, sparse_query: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>, SearchError>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}
```

### 2. Real Token Storage Integration (Stage 5)

The pipeline defines a `TokenStorage` trait. Create a real implementation that retrieves ColBERT token embeddings:

```rust
/// Trait for storing/retrieving token-level embeddings for ColBERT MaxSim.
pub trait TokenStorage: Send + Sync {
    fn get_tokens(&self, id: &Uuid) -> Result<Option<Vec<Vec<f32>>>, SearchError>;
    fn store_tokens(&self, id: &Uuid, tokens: Vec<Vec<f32>>) -> Result<(), SearchError>;
}
```

### 3. Performance Validation at Scale

Run at scale to verify:
- Total pipeline latency <60ms at 1M memories
- Per-stage latency targets met
- Funnel shape maintained

### 4. End-to-End Integration Tests with Real Data

Write tests that:
- Create real EmbedderIndexRegistry with actual HNSW indexes
- Insert real TeleologicalArrays with all 13 embeddings
- Execute pipeline with real queries
- Verify results against ground truth
</objective>

<rationale>
The 5-stage pipeline enables:
1. Early elimination of non-candidates (Stage 1-2): 1M -> 1K in <15ms
2. Progressive precision increase (Stages 3-5): 1K -> k with increasing accuracy
3. Sub-60ms total latency even at scale
4. Stage skipping for specialized queries
5. Configurable precision-speed tradeoff per stage
</rationale>

<architecture_constraints>
## From constitution.yaml (MUST NOT VIOLATE)

- **ARCH-01**: TeleologicalArray is atomic - all 13 embeddings stored/retrieved together
- **ARCH-02**: Apples-to-apples comparison - E1 compares with E1, NEVER cross-embedder
- **ARCH-04**: Entry-point discovery for retrieval (exactly what this pipeline implements)
- **FAIL FAST**: All errors are fatal. No recovery attempts. No fallbacks.
- **Performance**: inject_context <25ms, full retrieval <30ms

## Performance Budgets

| Stage | Target Latency | Input | Output |
|-------|---------------|-------|--------|
| S1: SPLADE | <5ms | 1M+ | 10K |
| S2: Matryoshka | <10ms | 10K | 1K |
| S3: RRF | <20ms | 1K | 100 |
| S4: Alignment | <10ms | 100 | 50 |
| S5: MaxSim | <15ms | 50 | k |
| **Total** | **<60ms** | **1M+** | **k** |

## CRITICAL: Non-HNSW Embedders

These 3 embedders CANNOT be queried via HNSW:
- **E6Sparse** - Requires inverted index with BM25
- **E12LateInteraction** - Requires ColBERT MaxSim token-level
- **E13Splade** - Requires inverted index with learned expansion

The pipeline correctly uses trait abstractions for Stage 1 (`SpladeIndex`) and Stage 5 (`TokenStorage`).
</architecture_constraints>

<source_of_truth>
## Full State Verification Protocol

### 1. Before ANY Test

```rust
// Print registry state
println!("=== SOURCE OF TRUTH: Registry State ===");
for embedder in [E1Matryoshka128, E1Semantic, PurposeVector] {
    let count = registry.get(embedder).map(|i| i.len()).unwrap_or(0);
    println!("{:?}: {} vectors indexed", embedder, count);
}
```

### 2. After Pipeline Execution

```rust
let result = pipeline.execute(&query_splade, &query_matryoshka, &query_semantic, &query_tokens)?;

// VERIFY: Results non-empty
assert!(!result.results.is_empty(), "Pipeline returned empty results");

// VERIFY: All stages executed
assert_eq!(result.stages_executed.len(), 5);

// VERIFY: Funnel shape
for window in result.stage_results.windows(2) {
    assert!(window[1].candidates_in <= window[0].candidates_out);
}

// VERIFY: Latency
assert!(result.total_latency_us / 1000 < 60, "Exceeded 60ms target");

// VERIFY: Scores valid (no NaN)
for candidate in &result.results {
    assert!(!candidate.score.is_nan());
    assert!(candidate.score >= 0.0);
}

// VERIFY: Results sorted descending
for window in result.results.windows(2) {
    assert!(window[0].score >= window[1].score);
}
```

### 3. Edge Case Verification

| Edge Case | Before State | Action | Expected After State |
|-----------|--------------|--------|---------------------|
| Empty index | len=0 | execute | Empty results, no error |
| Single match | 1 vector | execute | 1 result with score |
| Skip stage | Stage 2 disabled | execute_stages([1,3,4,5]) | 4 stages executed |
| Timeout | Stage 1 > 5ms | execute | `PipelineError::Timeout` |
| Bad query | NaN in splade | execute | `SearchError::InvalidVector` |
</source_of_truth>

<test_commands>
```bash
# Compile check (MUST PASS)
cargo check -p context-graph-storage

# Run pipeline tests (MUST SHOW 21 PASS)
cargo test -p context-graph-storage teleological::search::pipeline -- --nocapture

# Clippy validation (MUST HAVE NO ERRORS)
cargo clippy -p context-graph-storage -- -D warnings

# Run full test suite
cargo test -p context-graph-storage -- --nocapture

# Check exports are correct
grep -A20 "pub use pipeline" crates/context-graph-storage/src/teleological/search/mod.rs
```
</test_commands>

<fail_fast_compliance>
## FAIL FAST Compliance

All errors MUST propagate immediately:

| Input | Expected Error |
|-------|----------------|
| Empty query vector | `SearchError::InvalidVector("empty query")` |
| NaN in query | `SearchError::InvalidVector("NaN at index N")` |
| Inf in query | `SearchError::InvalidVector("Inf at index N")` |
| Wrong dimension | `SearchError::DimensionMismatch { expected, got }` |
| Stage timeout | `PipelineError::Timeout { stage, elapsed_ms, max_ms }` |
| Missing required query | `PipelineError::MissingQuery { stage }` |

## NO Fallbacks. NO Recovery. NO Silent Failures.

```rust
// WRONG - Silent failure
if query.is_empty() {
    return Ok(vec![]); // BAD
}

// CORRECT - Fail fast
if query.is_empty() {
    return Err(SearchError::InvalidVector("FAIL FAST: empty query".into()));
}
```
</fail_fast_compliance>

<common_mistakes>
## DO NOT Do These

1. **DO NOT** re-implement pipeline.rs - it already exists with 1934 lines
2. **DO NOT** modify pipeline.rs to use HNSW for E6Sparse, E12LateInteraction, or E13Splade
3. **DO NOT** use async/await - the pipeline is synchronous with rayon for parallelism
4. **DO NOT** swallow errors - propagate all `SearchError` and `PipelineError` variants
5. **DO NOT** return empty results on error - FAIL FAST with proper error
6. **DO NOT** use mock data in integration tests - use real HNSW indexes
7. **DO NOT** skip validation - check all query vectors for NaN/Inf/empty
8. **DO NOT** ignore latency constraints - each stage has max_latency_ms
9. **DO NOT** break funnel shape - each stage MUST reduce candidates
</common_mistakes>

<next_steps>
## Recommended Next Steps

1. **Verify existing implementation** compiles and tests pass:
   ```bash
   cargo check -p context-graph-storage
   cargo test -p context-graph-storage teleological::search::pipeline
   ```

2. **Identify real SPLADE index** - Check if one exists or needs creation:
   ```bash
   grep -r "SpladeIndex" crates/
   grep -r "inverted" crates/
   ```

3. **Identify real token storage** - Check if ColBERT token storage exists:
   ```bash
   grep -r "TokenStorage" crates/
   grep -r "ColBERT" crates/
   ```

4. **Write integration tests** with real data at scale

5. **Run benchmarks** to validate <60ms target at 1M memories
</next_steps>
</task_spec>
```
