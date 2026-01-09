# TASK-PERF-001: Performance Benchmark Suite

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-PERF-001 |
| **Title** | Performance Benchmark Suite |
| **Status** | :white_circle: todo |
| **Layer** | Performance |
| **Sequence** | 51 |
| **Estimated Days** | 2 |
| **Complexity** | Medium |

## Implements

- Constitution performance_budgets (lines 518-599)
- REQ-LATENCY-01: End-to-end retrieval < 30ms
- REQ-LATENCY-02: Entry-point search < 5ms
- REQ-THROUGHPUT-01: Embedding generation > 100 items/sec
- REQ-THROUGHPUT-02: Search throughput > 1000 qps

## Dependencies

| Task | Reason |
|------|--------|
| TASK-LOGIC-008 | Search pipeline to benchmark |
| TASK-CORE-003 | Store operations to benchmark |
| All embedding tasks | Embedding generation to benchmark |

## Objective

Create comprehensive performance benchmarks using `criterion` to:
1. Verify constitution performance requirements
2. Detect performance regressions
3. Provide baseline metrics for optimization
4. Enable CI performance gates

## Context

**Constitution Performance Budgets:**

| Operation | Latency Target | Throughput Target |
|-----------|----------------|-------------------|
| Entry-point search | < 5ms | > 1000 qps |
| Full retrieval | < 30ms | - |
| Embedding generation | - | > 100 items/sec |
| MCP tool response | < 100ms | - |
| Store operation | < 10ms | > 500 ops/sec |

## Scope

### In Scope

- Criterion benchmark suite
- Latency benchmarks for all critical paths
- Throughput benchmarks
- Memory usage benchmarks
- CI integration for regression detection
- HTML report generation

### Out of Scope

- Distributed benchmarking
- Load testing
- Profiling (see PERF-002)

## Definition of Done

### Benchmark Structure

```
benches/
├── Cargo.toml
├── criterion.rs           # Shared configuration
├── embedding_bench.rs     # Embedding benchmarks
├── search_bench.rs        # Search benchmarks
├── store_bench.rs         # Storage benchmarks
├── pipeline_bench.rs      # Full pipeline benchmarks
└── mcp_bench.rs           # MCP handler benchmarks
```

### Benchmark Configuration

```rust
// benches/criterion.rs

use criterion::{Criterion, BenchmarkId, Throughput};
use std::time::Duration;

/// Standard criterion configuration
pub fn config() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(100)
        .warm_up_time(Duration::from_secs(3))
        .with_plots()
        .configure_from_args()
}

/// Quick benchmark configuration (CI)
pub fn quick_config() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(5))
        .sample_size(50)
        .warm_up_time(Duration::from_secs(1))
}
```

### Embedding Benchmarks

```rust
// benches/embedding_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use context_graph_core::teleology::embedder::*;

fn bench_embedding_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_generation");

    // Test different content lengths
    for size in [100, 500, 1000, 5000].iter() {
        let content = "a".repeat(*size);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("semantic", size),
            &content,
            |b, content| {
                b.iter(|| {
                    // Generate semantic embedding (E1)
                    let embedder = SemanticEmbedder::new();
                    embedder.embed(content)
                })
            },
        );
    }

    group.finish();
}

fn bench_batch_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_embedding");

    for batch_size in [10, 50, 100, 500].iter() {
        let contents: Vec<_> = (0..*batch_size)
            .map(|i| format!("Content item {}", i))
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &contents,
            |b, contents| {
                b.iter(|| {
                    let embedder = SemanticEmbedder::new();
                    embedder.embed_batch(contents)
                })
            },
        );
    }

    group.finish();
}

fn bench_all_embedders(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_embedders");
    let content = "This is a test sentence for embedding benchmarks.";

    let embedders = vec![
        ("E1_semantic", Embedder::Semantic),
        ("E2_temporal_recent", Embedder::TemporalRecent),
        ("E3_temporal_periodic", Embedder::TemporalPeriodic),
        ("E4_entity", Embedder::EntityRelationship),
        ("E5_causal", Embedder::Causal),
        ("E6_splade", Embedder::Splade),
        ("E7_contextual", Embedder::Contextual),
        ("E8_emotional", Embedder::Emotional),
        ("E9_syntactic", Embedder::Syntactic),
        ("E10_pragmatic", Embedder::Pragmatic),
        ("E11_crossmodal", Embedder::CrossModal),
        ("E12_late_interaction", Embedder::LateInteraction),
        ("E13_keyword", Embedder::KeywordSplade),
    ];

    for (name, embedder_type) in embedders {
        group.bench_function(name, |b| {
            let embedder = create_embedder(embedder_type);
            b.iter(|| embedder.embed(content))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_embedding_generation,
    bench_batch_embedding,
    bench_all_embedders,
);
criterion_main!(benches);
```

### Search Benchmarks

```rust
// benches/search_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use context_graph_storage::teleological::*;

fn bench_entry_point_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Setup: Create store with test data
    let store = rt.block_on(async {
        let store = TeleologicalArrayStore::in_memory().await.unwrap();
        // Populate with 10K entries
        for i in 0..10_000 {
            let mut array = TeleologicalArray::new();
            array.content = format!("Test content {}", i);
            store.store(array).await.unwrap();
        }
        store
    });

    let mut group = c.benchmark_group("entry_point_search");
    group.throughput(Throughput::Elements(1));

    // Benchmark different embedder entry points
    let entry_points = vec![
        ("E1_semantic", Embedder::Semantic),
        ("E6_splade", Embedder::Splade),
        ("E4_entity", Embedder::EntityRelationship),
    ];

    for (name, entry_point) in entry_points {
        group.bench_function(name, |b| {
            let query = TeleologicalArray::new();
            b.iter(|| {
                rt.block_on(async {
                    store.search_entry_point(&query, entry_point, 100).await
                })
            })
        });
    }

    group.finish();

    // TARGET: < 5ms (REQ-LATENCY-02)
}

fn bench_full_retrieval(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let store = rt.block_on(async {
        let store = TeleologicalArrayStore::in_memory().await.unwrap();
        for i in 0..10_000 {
            let mut array = TeleologicalArray::new();
            array.content = format!("Test content for full retrieval {}", i);
            store.store(array).await.unwrap();
        }
        store
    });

    let mut group = c.benchmark_group("full_retrieval");

    // Different result limits
    for limit in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("limit", limit),
            limit,
            |b, &limit| {
                let query = TeleologicalArray::new();
                b.iter(|| {
                    rt.block_on(async {
                        store.search(&query, limit).await
                    })
                })
            },
        );
    }

    group.finish();

    // TARGET: < 30ms (REQ-LATENCY-01)
}

fn bench_search_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let store = rt.block_on(async {
        let store = TeleologicalArrayStore::in_memory().await.unwrap();
        for i in 0..10_000 {
            let mut array = TeleologicalArray::new();
            array.content = format!("Content {}", i);
            store.store(array).await.unwrap();
        }
        store
    });

    let mut group = c.benchmark_group("search_throughput");
    group.throughput(Throughput::Elements(100)); // 100 queries per iteration

    group.bench_function("batch_100", |b| {
        let queries: Vec<_> = (0..100)
            .map(|_| TeleologicalArray::new())
            .collect();

        b.iter(|| {
            rt.block_on(async {
                for query in &queries {
                    store.search(query, 10).await.unwrap();
                }
            })
        })
    });

    group.finish();

    // TARGET: > 1000 qps (REQ-THROUGHPUT-02)
}

criterion_group!(
    benches,
    bench_entry_point_search,
    bench_full_retrieval,
    bench_search_throughput,
);
criterion_main!(benches);
```

### Store Benchmarks

```rust
// benches/store_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

fn bench_store_operation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("store_operation");
    group.throughput(Throughput::Elements(1));

    group.bench_function("single_store", |b| {
        b.iter(|| {
            rt.block_on(async {
                let store = TeleologicalArrayStore::in_memory().await.unwrap();
                let array = TeleologicalArray::new();
                store.store(array).await
            })
        })
    });

    group.finish();

    // TARGET: < 10ms per store
}

fn bench_batch_store(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("batch_store");

    for batch_size in [10, 50, 100, 500].iter() {
        let arrays: Vec<_> = (0..*batch_size)
            .map(|i| {
                let mut array = TeleologicalArray::new();
                array.content = format!("Batch item {}", i);
                array
            })
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &arrays,
            |b, arrays| {
                b.iter(|| {
                    rt.block_on(async {
                        let store = TeleologicalArrayStore::in_memory().await.unwrap();
                        store.store_batch(arrays).await
                    })
                })
            },
        );
    }

    group.finish();

    // TARGET: > 500 ops/sec
}

fn bench_get_operation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let (store, id) = rt.block_on(async {
        let store = TeleologicalArrayStore::in_memory().await.unwrap();
        let array = TeleologicalArray::new();
        let id = array.id;
        store.store(array).await.unwrap();
        (store, id)
    });

    c.bench_function("get_by_id", |b| {
        b.iter(|| {
            rt.block_on(async {
                store.get(id).await
            })
        })
    });
}

criterion_group!(
    benches,
    bench_store_operation,
    bench_batch_store,
    bench_get_operation,
);
criterion_main!(benches);
```

### Full Pipeline Benchmarks

```rust
// benches/pipeline_bench.rs

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_5_stage_pipeline(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let pipeline = rt.block_on(async {
        // Setup full pipeline
        SearchPipeline::new(SearchPipelineConfig::default()).await.unwrap()
    });

    let mut group = c.benchmark_group("5_stage_pipeline");

    group.bench_function("full_pipeline", |b| {
        let query = TeleologicalArray::new();
        b.iter(|| {
            rt.block_on(async {
                pipeline.search(&query, 100).await
            })
        })
    });

    // Benchmark individual stages
    group.bench_function("stage1_prefilter", |b| {
        b.iter(|| { /* Stage 1 only */ })
    });

    group.bench_function("stage2_ann", |b| {
        b.iter(|| { /* Stage 2 only */ })
    });

    group.bench_function("stage3_rrf", |b| {
        b.iter(|| { /* Stage 3 only */ })
    });

    group.bench_function("stage4_alignment", |b| {
        b.iter(|| { /* Stage 4 only */ })
    });

    group.bench_function("stage5_late_interaction", |b| {
        b.iter(|| { /* Stage 5 only */ })
    });

    group.finish();
}

criterion_group!(benches, bench_5_stage_pipeline);
criterion_main!(benches);
```

### MCP Handler Benchmarks

```rust
// benches/mcp_bench.rs

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_mcp_handlers(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let server = rt.block_on(async {
        McpServer::new_test().await.unwrap()
    });

    let mut group = c.benchmark_group("mcp_handlers");

    group.bench_function("inject_context", |b| {
        let params = InjectContextParams {
            content: "Test content".into(),
            metadata: Default::default(),
            importance: Some(0.5),
        };
        b.iter(|| {
            rt.block_on(async {
                server.handle_inject_context("session", params.clone()).await
            })
        })
    });

    group.bench_function("store_memory", |b| {
        let params = StoreMemoryParams {
            content: "Test memory".into(),
            purpose: "test".into(),
            metadata: Default::default(),
        };
        b.iter(|| {
            rt.block_on(async {
                server.handle_store_memory("session", params.clone()).await
            })
        })
    });

    group.bench_function("search_graph", |b| {
        let params = SearchGraphParams {
            query: "test query".into(),
            limit: Some(10),
            filters: Default::default(),
        };
        b.iter(|| {
            rt.block_on(async {
                server.handle_search_graph("session", params.clone()).await
            })
        })
    });

    group.finish();

    // TARGET: < 100ms per handler
}

criterion_group!(benches, bench_mcp_handlers);
criterion_main!(benches);
```

### CI Integration Script

```bash
#!/bin/bash
# scripts/bench.sh

set -e

# Run benchmarks and save baseline
cargo bench --bench embedding_bench -- --save-baseline main
cargo bench --bench search_bench -- --save-baseline main
cargo bench --bench store_bench -- --save-baseline main
cargo bench --bench pipeline_bench -- --save-baseline main
cargo bench --bench mcp_bench -- --save-baseline main

# Compare with baseline (for CI)
cargo bench -- --baseline main --load-baseline pr

# Check for regressions > 10%
# (criterion exits with error if regression detected)
```

### Constraints

| Constraint | Target |
|------------|--------|
| Benchmark runtime | < 5 minutes total |
| Sample size | 100 (50 for CI) |
| Warm-up | 3s (1s for CI) |
| Regression threshold | 10% |

## Verification

- [ ] All benchmarks compile and run
- [ ] Entry-point search < 5ms (10K entries)
- [ ] Full retrieval < 30ms (10K entries)
- [ ] Embedding throughput > 100/sec
- [ ] Search throughput > 1000 qps
- [ ] Store operation < 10ms
- [ ] MCP handlers < 100ms
- [ ] CI integration working

## Files to Create

| File | Purpose |
|------|---------|
| `benches/criterion.rs` | Shared config |
| `benches/embedding_bench.rs` | Embedding benchmarks |
| `benches/search_bench.rs` | Search benchmarks |
| `benches/store_bench.rs` | Store benchmarks |
| `benches/pipeline_bench.rs` | Pipeline benchmarks |
| `benches/mcp_bench.rs` | MCP benchmarks |
| `scripts/bench.sh` | CI script |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Slow benchmarks | Medium | Medium | Quick mode for CI |
| Flaky results | Low | Low | Sufficient samples |
| Hardware variance | Medium | Low | Relative comparisons |

## Traceability

- Source: Constitution performance_budgets (lines 518-599)
