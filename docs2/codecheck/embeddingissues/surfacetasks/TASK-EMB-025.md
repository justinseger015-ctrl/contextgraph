# TASK-EMB-025: Integration Tests

<task_spec id="TASK-EMB-025" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-025 |
| **Title** | End-to-End Integration Tests with Real Data |
| **Status** | ready |
| **Layer** | surface |
| **Sequence** | 25 |
| **Implements** | REQ-EMB-007 (Performance Compliance) |
| **Depends On** | All previous tasks (TASK-EMB-001 through TASK-EMB-024) |
| **Estimated Complexity** | high |
| **Created** | 2026-01-06 |
| **Constitution Reference** | v4.0.0 |

---

## Context

The final task creates comprehensive integration tests that verify the entire embedding pipeline works correctly with real data, real GPU, and real storage. These tests validate:

- Correct dimensions across all 13 embedders
- Semantic similarity preservation
- Storage roundtrip integrity
- Multi-space search functionality
- Performance within Constitutional bounds

**Constitution Alignment:**
- `perf.latency.single_embed: "<10ms"`
- `perf.latency.batch_embed_64: "<50ms"`
- `embeddings.models`: All dimension specifications
- `forbidden.AP-007`: No stub data in prod

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Test fixtures dir | `crates/context-graph-embeddings/tests/fixtures/` |
| Golden references | `crates/context-graph-embeddings/tests/fixtures/golden/` |
| Existing tests | `crates/context-graph-embeddings/tests/` |
| Constitution | `docs2/constitution.yaml` |
| All embedding modules | `crates/context-graph-embeddings/src/` |
| All storage modules | `crates/context-graph-embeddings/src/storage/` |

---

## Prerequisites

- [ ] All TASK-EMB-001 through TASK-EMB-024 completed
- [ ] Test fixtures with real data created (TASK-EMB-010)
- [ ] GPU available for tests
- [ ] Golden reference files for validation
- [ ] All embedding models loadable

---

## Scope

### In Scope

- End-to-end embedding pipeline tests
- Storage roundtrip tests (store → retrieve → verify)
- Multi-space search tests (per-embedder + RRF)
- Performance benchmarks (latency verification)
- Dimension validation across all 13 embedders
- Semantic similarity tests
- No-fake-data verification tests

### Out of Scope

- Chaos/fault injection tests (separate suite)
- Load testing (separate infrastructure)
- ScyllaDB production tests (requires cluster)
- UI/CLI integration tests

---

## Definition of Done

### Test File Structure

```
crates/context-graph-embeddings/tests/
├── integration/
│   ├── mod.rs                    # Module declaration
│   ├── pipeline_test.rs          # End-to-end pipeline tests
│   ├── storage_test.rs           # Storage roundtrip tests
│   ├── search_test.rs            # Multi-space search tests
│   ├── dimension_test.rs         # Dimension validation tests
│   └── benchmark_test.rs         # Performance benchmarks
└── fixtures/
    ├── models/                   # Test model weights
    ├── golden/                   # Golden reference outputs
    └── inputs/                   # Test input data
```

### Pipeline Tests (`pipeline_test.rs`)

```rust
//! End-to-end integration tests for the embedding pipeline.
//!
//! # Requirements
//! - CUDA-capable GPU
//! - Model weights in tests/fixtures/models/
//! - Golden references in tests/fixtures/golden/

use context_graph_embeddings::models::SparseModel;
use context_graph_embeddings::storage::{FingerprintStorage, RocksDbStorage, MultiSpaceSearch};
use context_graph_embeddings::quantization::QuantizationRouter;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

/// Test: E6 Sparse produces 1536D output (not 768D).
///
/// Validates TASK-EMB-001 and TASK-EMB-021 are correctly integrated.
#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_sparse_dimension_is_1536() {
    let model_path = Path::new("tests/fixtures/models/sparse");
    let model = SparseModel::new(model_path, Default::default()).unwrap();
    model.load().await.unwrap();

    let input = ModelInput::Text { content: "machine learning".to_string() };
    let embedding = model.embed(&input).await.unwrap();

    assert_eq!(
        embedding.vector().len(),
        1536,
        "E6 dimension should be 1536 (Constitution), not 768 (broken)"
    );
}

/// Test: Sparse projection preserves semantic similarity.
///
/// Related terms should have high similarity, unrelated terms low.
#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_semantic_similarity_preserved() {
    let model = load_sparse_model().await;

    let ml_emb = model.embed(&text("machine learning algorithms")).await.unwrap();
    let dl_emb = model.embed(&text("deep learning neural networks")).await.unwrap();
    let car_emb = model.embed(&text("automobile vehicle transportation")).await.unwrap();

    let ml_dl_sim = cosine_similarity(ml_emb.vector(), dl_emb.vector());
    let ml_car_sim = cosine_similarity(ml_emb.vector(), car_emb.vector());

    assert!(
        ml_dl_sim > 0.7,
        "Related terms should have similarity > 0.7, got {}", ml_dl_sim
    );
    assert!(
        ml_car_sim < 0.5,
        "Unrelated terms should have similarity < 0.5, got {}", ml_car_sim
    );
}

/// Test: No stub or simulated data in output.
///
/// Verifies AP-007 compliance.
#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_no_fake_data() {
    let model = load_sparse_model().await;
    let embedding = model.embed(&text("test input")).await.unwrap();

    // Check for sin wave pattern (old fake output: (i * 0.001).sin())
    let is_sin_wave = embedding.vector().iter().enumerate().all(|(i, &v)| {
        (v - (i as f32 * 0.001).sin()).abs() < 0.01
    });
    assert!(!is_sin_wave, "Output should NOT be sin wave (indicates fake data)");

    // Check for all zeros (another fake pattern)
    let is_all_zeros = embedding.vector().iter().all(|&v| v.abs() < 1e-10);
    assert!(!is_all_zeros, "Output should NOT be all zeros");

    // Check for constant values
    let first = embedding.vector()[0];
    let is_constant = embedding.vector().iter().all(|&v| (v - first).abs() < 1e-6);
    assert!(!is_constant, "Output should NOT be constant");
}
```

### Storage Tests (`storage_test.rs`)

```rust
/// Test: Storage roundtrip preserves all embeddings.
#[tokio::test]
async fn test_storage_roundtrip() {
    let temp_dir = tempfile::tempdir().unwrap();
    let quantizer = create_test_quantizer();
    let storage = Arc::new(RocksDbStorage::open(temp_dir.path(), quantizer).unwrap());

    let fingerprint = create_test_fingerprint();
    let id = fingerprint.id;

    storage.store(&fingerprint).unwrap();
    let retrieved = storage.retrieve(id).unwrap().expect("Fingerprint should exist");

    assert_eq!(fingerprint.id, retrieved.id);
    assert_eq!(fingerprint.purpose_vector, retrieved.purpose_vector);
    assert_eq!(fingerprint.johari_quadrants, retrieved.johari_quadrants);
    assert_eq!(fingerprint.embeddings.len(), retrieved.embeddings.len());

    // Verify embedding data within quantization tolerance
    for (idx, original) in &fingerprint.embeddings {
        let retrieved_emb = retrieved.embeddings.get(idx).unwrap();
        assert_vectors_similar(original, retrieved_emb, 0.01); // 1% tolerance for quantization
    }
}

/// Test: Lazy loading retrieves only requested embedders.
#[tokio::test]
async fn test_lazy_loading() {
    let storage = create_test_storage();
    let fingerprint = create_test_fingerprint();
    storage.store(&fingerprint).unwrap();

    // Request only embedders 0, 5, 12
    let requested = vec![0, 5, 12];
    let partial = storage.retrieve_embeddings(fingerprint.id, &requested).unwrap().unwrap();

    assert_eq!(partial.len(), 3);
    assert!(partial.iter().all(|(idx, _)| requested.contains(idx)));
}
```

### Search Tests (`search_test.rs`)

```rust
/// Test: Multi-space search returns ranked results.
#[tokio::test]
async fn test_multi_space_search() {
    let (storage, search) = create_test_search_engine();

    // Index test fingerprints
    let fingerprints: Vec<_> = (0..100).map(|_| create_random_fingerprint()).collect();
    for fp in &fingerprints {
        storage.store(fp).unwrap();
        search.index(fp).unwrap();
    }

    // Search single embedder
    let query = create_random_vector(1024); // E1 dimension
    let results = search.search_embedder(0, &query, 10).unwrap();

    assert_eq!(results.len(), 10);

    // Verify results are ranked by similarity (descending)
    for i in 1..results.len() {
        assert!(
            results[i-1].similarity >= results[i].similarity,
            "Results should be ranked by similarity"
        );
    }
}

/// Test: RRF fusion produces correct scores.
#[tokio::test]
async fn test_rrf_fusion() {
    let (storage, search) = create_test_search_engine();

    // Index fingerprints
    for fp in create_test_fingerprints(50) {
        storage.store(&fp).unwrap();
        search.index(&fp).unwrap();
    }

    // Multi-space query
    let queries = create_multi_space_query();
    let results = search.search_multi_space(&queries, None, 100, 10).unwrap();

    // Verify RRF scores are positive and ordered
    assert!(!results.is_empty());
    for i in 1..results.len() {
        assert!(results[i-1].rrf_score >= results[i].rrf_score);
        assert!(results[i].rrf_score > 0.0);
    }
}

/// Test: Purpose weights affect ranking.
#[tokio::test]
async fn test_purpose_weighted_search() {
    let (storage, search) = create_test_search_engine();

    // Create fingerprints with known characteristics
    // Some strong in E1 (semantic), others in E5 (causal)
    let semantic_strong = create_fingerprint_strong_in(0); // E1
    let causal_strong = create_fingerprint_strong_in(4);   // E5

    storage.store(&semantic_strong).unwrap();
    storage.store(&causal_strong).unwrap();
    search.index(&semantic_strong).unwrap();
    search.index(&causal_strong).unwrap();

    let queries = create_multi_space_query();

    // Search with semantic weight emphasis
    let semantic_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let results_semantic = search.search_multi_space(&queries, Some(&semantic_weights), 100, 10).unwrap();

    // Search with causal weight emphasis
    let causal_weights = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let results_causal = search.search_multi_space(&queries, Some(&causal_weights), 100, 10).unwrap();

    // Verify rankings differ based on weights
    assert_ne!(results_semantic[0].id, results_causal[0].id,
        "Different weights should produce different rankings");
}
```

### Dimension Tests (`dimension_test.rs`)

```rust
/// Test: All 13 embedders produce correct dimensions.
#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_all_embedder_dimensions() {
    let expected_dimensions = [
        (ModelId::Semantic, 1024),
        (ModelId::TemporalRecent, 512),
        (ModelId::TemporalPeriodic, 512),
        (ModelId::TemporalPositional, 512),
        (ModelId::Causal, 768),
        (ModelId::Sparse, 1536),       // CRITICAL: Must be 1536, not 768
        (ModelId::Code, 1536),
        (ModelId::Graph, 384),
        (ModelId::Hdc, 1024),
        (ModelId::Multimodal, 768),
        (ModelId::Entity, 384),
        (ModelId::LateInteraction, 128), // per token
        (ModelId::Splade, 1536),
    ];

    for (model_id, expected_dim) in expected_dimensions {
        let actual_dim = model_id.dimension();
        assert_eq!(
            actual_dim, expected_dim,
            "{:?} dimension mismatch: expected {}, got {}",
            model_id, expected_dim, actual_dim
        );
    }
}

/// Test: ModelId.projected_dimension() matches actual output.
#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_projected_dimension_matches_output() {
    let sparse_model = load_sparse_model().await;
    let embedding = sparse_model.embed(&text("test")).await.unwrap();

    assert_eq!(
        embedding.vector().len(),
        ModelId::Sparse.dimension(),
        "Actual output dimension must match ModelId.dimension()"
    );
}
```

### Benchmark Tests (`benchmark_test.rs`)

```rust
/// Benchmark: Single embedding latency.
/// Constitution: perf.latency.single_embed < 10ms
#[tokio::test]
#[cfg(feature = "cuda")]
async fn bench_single_embed_latency() {
    let model = load_sparse_model().await;
    let input = text("benchmark input text");

    // Warmup
    for _ in 0..10 {
        let _ = model.embed(&input).await.unwrap();
    }

    // Measure
    let start = std::time::Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        let _ = model.embed(&input).await.unwrap();
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() / iterations;

    println!("Average single_embed latency: {} us ({:.2} ms)", avg_us, avg_us as f64 / 1000.0);
    assert!(
        avg_us < 10_000,
        "single_embed should be < 10ms (Constitution), got {} us", avg_us
    );
}

/// Benchmark: Batch embedding latency.
/// Constitution: perf.latency.batch_embed_64 < 50ms
#[tokio::test]
#[cfg(feature = "cuda")]
async fn bench_batch_embed_64_latency() {
    let model = load_sparse_model().await;
    let inputs: Vec<_> = (0..64).map(|i| text(&format!("batch input {}", i))).collect();

    // Warmup
    for _ in 0..5 {
        for input in &inputs {
            let _ = model.embed(input).await.unwrap();
        }
    }

    // Measure batch
    let start = std::time::Instant::now();
    for input in &inputs {
        let _ = model.embed(input).await.unwrap();
    }
    let elapsed = start.elapsed();

    println!("Batch 64 latency: {} ms", elapsed.as_millis());
    assert!(
        elapsed.as_millis() < 50,
        "batch_embed_64 should be < 50ms (Constitution), got {} ms", elapsed.as_millis()
    );
}

/// Benchmark: Sparse projection step latency.
/// Target: < 3ms per Constitution
#[tokio::test]
#[cfg(feature = "cuda")]
async fn bench_sparse_projection_latency() {
    // This would test the projection step in isolation
    // Implementation depends on how projection is exposed
}
```

### Helper Functions

```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

fn text(s: &str) -> ModelInput {
    ModelInput::Text { content: s.to_string() }
}

async fn load_sparse_model() -> SparseModel {
    let path = Path::new("tests/fixtures/models/sparse");
    let model = SparseModel::new(path, Default::default()).unwrap();
    model.load().await.unwrap();
    model
}

fn create_test_fingerprint() -> StoredFingerprint {
    StoredFingerprint {
        id: Uuid::new_v4(),
        version: 1,
        purpose_vector: [0.5; 13],
        theta_to_north_star: 0.75,
        johari_quadrants: [JohariQuadrant::Open; 13],
        johari_confidence: [0.8; 13],
        content_hash: [0u8; 32],
        embeddings: (0..13).map(|i| (i as u8, create_random_vector(get_dim(i)))).collect(),
        created_at_ms: chrono::Utc::now().timestamp_millis(),
        last_updated_ms: chrono::Utc::now().timestamp_millis(),
        access_count: 0,
        deleted: false,
        dominant_quadrant: JohariQuadrant::Open,
    }
}
```

### Constraints

- Tests MUST use real GPU (gated by `#[cfg(feature = "cuda")]`)
- Tests MUST use real weight files from fixtures
- NO mock data allowed
- Performance tests document latency with assertions
- All dimension assertions match Constitution

### Verification

- All tests pass with real GPU
- Latencies within Constitutional bounds
- Dimensions match across all embedders
- No stub/fake data detected

---

## Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/tests/integration/mod.rs` | Module declaration |
| `crates/context-graph-embeddings/tests/integration/pipeline_test.rs` | Pipeline tests |
| `crates/context-graph-embeddings/tests/integration/storage_test.rs` | Storage tests |
| `crates/context-graph-embeddings/tests/integration/search_test.rs` | Search tests |
| `crates/context-graph-embeddings/tests/integration/dimension_test.rs` | Dimension validation |
| `crates/context-graph-embeddings/tests/integration/benchmark_test.rs` | Performance benchmarks |

---

## Validation Criteria

- [ ] All integration tests pass with `--features cuda`
- [ ] No tests use mock/stub data
- [ ] E6 Sparse output is 1536D (not 768D)
- [ ] Semantic similarity preserved (related > 0.7, unrelated < 0.5)
- [ ] Storage roundtrip preserves data within quantization tolerance
- [ ] Multi-space search returns correctly ordered results
- [ ] RRF fusion scores are correct
- [ ] Purpose weights affect ranking
- [ ] single_embed latency < 10ms
- [ ] batch_embed_64 latency < 50ms

---

## Test Commands

```bash
cd /home/cabdru/contextgraph

# Run all integration tests (requires GPU)
cargo test -p context-graph-embeddings --test '*' --features cuda -- --nocapture

# Run specific test file
cargo test -p context-graph-embeddings --test integration --features cuda -- pipeline_test --nocapture

# Run benchmarks
cargo bench -p context-graph-embeddings --features cuda

# Run with verbose output
RUST_LOG=debug cargo test -p context-graph-embeddings --features cuda -- --nocapture
```

---

## Performance Targets (Constitution)

| Metric | Target | Test |
|--------|--------|------|
| single_embed latency | < 10ms p95 | bench_single_embed_latency |
| batch_embed_64 latency | < 50ms p95 | bench_batch_embed_64_latency |
| sparse projection | < 3ms | bench_sparse_projection_latency |
| storage per fingerprint | < 20KB | (verify with file size) |

---

## Memory Key

Store completion status:
```
contextgraph/embedding-issues/task-emb-025-complete
```

---

## Final Checklist

When TASK-EMB-025 is complete, verify:

- [ ] All 25 embedding pipeline tasks complete
- [ ] No stub data in any production code
- [ ] All dimensions match Constitution
- [ ] Performance within bounds
- [ ] Integration tests comprehensive
- [ ] Ready for production deployment

</task_spec>
