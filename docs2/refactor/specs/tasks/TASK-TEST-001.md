# TASK-TEST-001: Property-Based Testing Infrastructure

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-TEST-001 |
| **Title** | Property-Based Testing Infrastructure |
| **Status** | :white_circle: todo |
| **Layer** | Testing |
| **Sequence** | 41 |
| **Estimated Days** | 2 |
| **Complexity** | Medium |

## Implements

- Constitution data_integrity_guarantees (atomicity, consistency)
- Testing requirements for TeleologicalArray system
- ARCH-01: Atomic storage verification

## Dependencies

| Task | Reason |
|------|--------|
| TASK-CORE-002 | Embedder types to test |
| TASK-CORE-003 | Store operations to verify |
| TASK-LOGIC-008 | Search pipeline to test |

## Objective

Implement property-based testing using `proptest` to verify invariants across the teleological array system:
1. Embedding dimension invariants
2. Storage atomicity properties
3. Search consistency guarantees
4. Serialization round-trip correctness

## Context

Property-based testing generates random inputs to verify that invariants hold across all possible states. This is critical for:
- Catching edge cases unit tests miss
- Verifying mathematical properties (dimensions, normalization)
- Ensuring atomicity of storage operations
- Validating search result consistency

## Scope

### In Scope

- `proptest` integration with test infrastructure
- Arbitrary implementations for all core types
- Property tests for embedder dimensions
- Property tests for store atomicity
- Property tests for search consistency
- Shrinking support for minimal failing cases

### Out of Scope

- Stateful property testing (model checking)
- Distributed system property testing
- Performance property testing

## Definition of Done

### Signatures

```rust
// crates/context-graph-core/src/teleology/testing/arbitrary.rs

use proptest::prelude::*;
use uuid::Uuid;

/// Arbitrary implementation for TeleologicalArray
impl Arbitrary for TeleologicalArray {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        (
            any::<String>().prop_filter("non-empty", |s| !s.is_empty()),
            prop::collection::hash_map(any::<String>(), any::<String>(), 0..10),
            arb_embeddings(),
        )
            .prop_map(|(content, metadata, embeddings)| {
                let mut array = TeleologicalArray::new();
                array.content = content;
                array.metadata = metadata;
                array.embeddings = embeddings;
                array
            })
            .boxed()
    }
}

/// Generate arbitrary embeddings for all 13 embedders
fn arb_embeddings() -> impl Strategy<Value = HashMap<Embedder, EmbedderOutput>> {
    prop::collection::hash_map(
        arb_embedder(),
        arb_embedder_output(),
        1..=13,
    )
}

/// Generate arbitrary embedder type
fn arb_embedder() -> impl Strategy<Value = Embedder> {
    prop_oneof![
        Just(Embedder::Semantic),
        Just(Embedder::TemporalRecent),
        Just(Embedder::TemporalPeriodic),
        Just(Embedder::EntityRelationship),
        Just(Embedder::Causal),
        Just(Embedder::Splade),
        Just(Embedder::Contextual),
        Just(Embedder::Emotional),
        Just(Embedder::Syntactic),
        Just(Embedder::Pragmatic),
        Just(Embedder::CrossModal),
        Just(Embedder::LateInteraction),
        Just(Embedder::KeywordSplade),
    ]
}

/// Generate arbitrary embedder output with correct dimensions
fn arb_embedder_output() -> impl Strategy<Value = EmbedderOutput> {
    prop_oneof![
        // Dense embeddings
        prop::collection::vec(any::<f32>(), 256..=1024)
            .prop_map(EmbedderOutput::Dense),
        // Sparse embeddings
        (
            prop::collection::vec(any::<u32>(), 10..100),
            prop::collection::vec(any::<f32>(), 10..100),
        )
            .prop_map(|(indices, values)| EmbedderOutput::Sparse(SparseVec {
                indices: indices.into_iter().zip(values).collect(),
                dim: 30_000,
            })),
    ]
}

/// Generate embedder output with exact dimensions for specific embedder
fn arb_embedder_output_for(embedder: Embedder) -> impl Strategy<Value = EmbedderOutput> {
    let dim = embedder.dimension();
    match embedder.output_type() {
        EmbedderOutputType::Dense => {
            prop::collection::vec(-1.0f32..1.0f32, dim)
                .prop_map(EmbedderOutput::Dense)
                .boxed()
        }
        EmbedderOutputType::Sparse => {
            (
                prop::collection::vec(0u32..dim as u32, 10..100),
                prop::collection::vec(0.0f32..1.0f32, 10..100),
            )
                .prop_map(move |(indices, values)| EmbedderOutput::Sparse(SparseVec {
                    indices: indices.into_iter().zip(values).collect(),
                    dim,
                }))
                .boxed()
        }
    }
}

/// Generate valid UUID
fn arb_uuid() -> impl Strategy<Value = Uuid> {
    prop::array::uniform16(any::<u8>())
        .prop_map(|bytes| Uuid::from_bytes(bytes))
}
```

### Property Tests

```rust
// crates/context-graph-core/src/teleology/testing/properties.rs

use proptest::prelude::*;

proptest! {
    /// Property: All embeddings must have correct dimensions for their type
    #[test]
    fn prop_embedding_dimensions_correct(array in any::<TeleologicalArray>()) {
        for (embedder, output) in &array.embeddings {
            let expected_dim = embedder.dimension();
            let actual_dim = output.dimension();
            prop_assert_eq!(
                actual_dim, expected_dim,
                "Embedder {:?} should have dim {}, got {}",
                embedder, expected_dim, actual_dim
            );
        }
    }

    /// Property: Serialization round-trip preserves all data
    #[test]
    fn prop_serialization_roundtrip(array in any::<TeleologicalArray>()) {
        let serialized = bincode::serialize(&array).unwrap();
        let deserialized: TeleologicalArray = bincode::deserialize(&serialized).unwrap();

        prop_assert_eq!(array.id, deserialized.id);
        prop_assert_eq!(array.content, deserialized.content);
        prop_assert_eq!(array.metadata, deserialized.metadata);
        prop_assert_eq!(array.embeddings.len(), deserialized.embeddings.len());
    }

    /// Property: Store operations are atomic (all-or-nothing)
    #[test]
    fn prop_store_atomic(
        arrays in prop::collection::vec(any::<TeleologicalArray>(), 1..10)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = TeleologicalArrayStore::in_memory().await.unwrap();

            // Store all arrays
            for array in &arrays {
                store.store(array.clone()).await.unwrap();
            }

            // All should be retrievable
            for array in &arrays {
                let retrieved = store.get(array.id).await.unwrap();
                prop_assert!(retrieved.is_some());
            }

            Ok(())
        })?
    }

    /// Property: Search returns results in score-descending order
    #[test]
    fn prop_search_ordering(
        query in any::<TeleologicalArray>(),
        arrays in prop::collection::vec(any::<TeleologicalArray>(), 5..20)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = TeleologicalArrayStore::in_memory().await.unwrap();
            for array in &arrays {
                store.store(array.clone()).await.unwrap();
            }

            let results = store.search(&query, 10).await.unwrap();

            // Verify descending score order
            for window in results.windows(2) {
                prop_assert!(
                    window[0].score >= window[1].score,
                    "Results not in descending order"
                );
            }

            Ok(())
        })?
    }

    /// Property: Duplicate store operations are idempotent
    #[test]
    fn prop_store_idempotent(array in any::<TeleologicalArray>()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = TeleologicalArrayStore::in_memory().await.unwrap();

            // Store twice
            store.store(array.clone()).await.unwrap();
            store.store(array.clone()).await.unwrap();

            // Only one copy should exist
            let count = store.count().await.unwrap();
            // Note: Depends on implementation - update or reject duplicates

            Ok(())
        })?
    }

    /// Property: Delete removes and get returns None
    #[test]
    fn prop_delete_removes(array in any::<TeleologicalArray>()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = TeleologicalArrayStore::in_memory().await.unwrap();

            store.store(array.clone()).await.unwrap();
            store.delete(array.id).await.unwrap();

            let retrieved = store.get(array.id).await.unwrap();
            prop_assert!(retrieved.is_none());

            Ok(())
        })?
    }

    /// Property: Normalized embeddings have unit length
    #[test]
    fn prop_normalized_unit_length(
        vec in prop::collection::vec(-10.0f32..10.0f32, 100..1000)
    ) {
        let normalized = normalize_l2(&vec);
        let length: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Allow small floating point error
        prop_assert!((length - 1.0).abs() < 1e-5, "Length should be 1.0, got {}", length);
    }
}

/// Test module with proptest config
mod proptest_config {
    use proptest::test_runner::Config;

    pub fn config() -> Config {
        Config {
            cases: 1000,
            max_shrink_iters: 10000,
            ..Config::default()
        }
    }
}
```

### Store Property Tests

```rust
// crates/context-graph-storage/src/teleological/testing/store_properties.rs

proptest! {
    #![proptest_config(proptest_config::config())]

    /// Property: Batch store is atomic (all succeed or all fail)
    #[test]
    fn prop_batch_atomic(
        arrays in prop::collection::vec(any::<TeleologicalArray>(), 5..20)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = TeleologicalArrayStore::in_memory().await.unwrap();

            // Batch store
            let result = store.store_batch(&arrays).await;

            match result {
                Ok(_) => {
                    // All should be present
                    for array in &arrays {
                        let retrieved = store.get(array.id).await.unwrap();
                        prop_assert!(retrieved.is_some());
                    }
                }
                Err(_) => {
                    // None should be present (rolled back)
                    for array in &arrays {
                        let retrieved = store.get(array.id).await.unwrap();
                        prop_assert!(retrieved.is_none());
                    }
                }
            }

            Ok(())
        })?
    }

    /// Property: Search limit is respected
    #[test]
    fn prop_search_limit_respected(
        query in any::<TeleologicalArray>(),
        arrays in prop::collection::vec(any::<TeleologicalArray>(), 20..50),
        limit in 1usize..20
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = TeleologicalArrayStore::in_memory().await.unwrap();
            for array in &arrays {
                store.store(array.clone()).await.unwrap();
            }

            let results = store.search(&query, limit).await.unwrap();
            prop_assert!(results.len() <= limit);

            Ok(())
        })?
    }

    /// Property: Count matches actual stored items
    #[test]
    fn prop_count_accurate(
        arrays in prop::collection::vec(any::<TeleologicalArray>(), 1..50)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = TeleologicalArrayStore::in_memory().await.unwrap();

            // Use unique IDs (dedup)
            let unique: Vec<_> = arrays.into_iter()
                .map(|mut a| { a.id = Uuid::new_v4(); a })
                .collect();

            for array in &unique {
                store.store(array.clone()).await.unwrap();
            }

            let count = store.count().await.unwrap();
            prop_assert_eq!(count, unique.len());

            Ok(())
        })?
    }
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Test cases per property | 1000 default |
| Max shrink iterations | 10000 |
| Property test timeout | 60s per property |

## Verification

- [ ] `proptest` dependency added to test crates
- [ ] Arbitrary implementations for all core types
- [ ] Dimension invariant property passes 1000 cases
- [ ] Serialization roundtrip property passes
- [ ] Store atomicity property verified
- [ ] Search ordering property verified
- [ ] Shrinking produces minimal failing cases

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/teleology/testing/mod.rs` | Test module root |
| `crates/context-graph-core/src/teleology/testing/arbitrary.rs` | Arbitrary impls |
| `crates/context-graph-core/src/teleology/testing/properties.rs` | Property tests |
| `crates/context-graph-storage/src/teleological/testing/store_properties.rs` | Store property tests |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Slow property tests | Medium | Low | Configure case count |
| Flaky tests | Low | Medium | Deterministic seeds |
| Missing invariants | Medium | Medium | Review constitution |

## Traceability

- Source: Constitution data_integrity_guarantees
- Related: ARCH-01 atomicity requirement
