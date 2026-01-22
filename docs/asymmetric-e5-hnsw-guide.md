# Asymmetric E5 HNSW Retrieval - Implementation Guide

**Version:** 1.0.0
**Date:** 2026-01-21
**Status:** PROPOSED
**Priority:** CRITICAL (fixes fundamental retrieval gap)

---

## Problem Statement

### Current State (Broken)

The current implementation applies asymmetric E5 similarity in the **scoring phase**, but HNSW candidate retrieval remains symmetric:

```
Current Flow:
┌─────────────────────────────────────────────────────────────┐
│ 1. HNSW Search (SYMMETRIC)                                  │
│    query.e5_active_vector() vs index[e5_active_vector]      │
│    → Returns candidates based on symmetric similarity       │
│    → MISSES documents with complementary cause/effect       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Scoring (ASYMMETRIC) - Too Late!                         │
│    Applies direction modifiers to candidates                │
│    → Can only rerank what HNSW found                        │
│    → Cannot recover missed documents                        │
└─────────────────────────────────────────────────────────────┘
```

### Why This Fails

| Query Type | What We Search | What We Should Search | Result |
|------------|----------------|----------------------|--------|
| "why does X fail" (Cause) | `query.e5_active` vs `doc.e5_active` | `query.e5_as_cause` vs `doc.e5_as_effect` | MISS effects |
| "what happens when X" (Effect) | `query.e5_active` vs `doc.e5_active` | `query.e5_as_effect` vs `doc.e5_as_cause` | MISS causes |

**Core Insight:** Causal queries seek COMPLEMENTARY vectors, not SIMILAR vectors.

- Cause-seeking query has strong CAUSE vector → should find docs with strong EFFECT vectors
- Effect-seeking query has strong EFFECT vector → should find docs with strong CAUSE vectors

---

## Solution: Dual E5 HNSW Indexes

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INDEXING (at store time)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Document with E5 dual vectors:                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ e5_as_cause     │  │ e5_as_effect    │                  │
│  │ (768D)          │  │ (768D)          │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                            │
│           ▼                    ▼                            │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ E5_CAUSE_INDEX  │  │ E5_EFFECT_INDEX │                  │
│  │ (HNSW)          │  │ (HNSW)          │                  │
│  └─────────────────┘  └─────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL (at query time)                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query: "why does authentication fail"                      │
│  Detected Direction: CAUSE (seeking causes)                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 1: Select complementary index                   │   │
│  │         Cause query → Search EFFECT index            │   │
│  │         (Find docs that describe effects/outcomes)   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 2: Search with query's cause vector             │   │
│  │         query.e5_as_cause vs E5_EFFECT_INDEX         │   │
│  │         → Finds docs whose effects match our cause   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 3: Apply direction modifier (1.2x boost)        │   │
│  │         cause→effect transition gets boosted         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Retrieval Logic Matrix

| Query Direction | Query Vector Used | Index Searched | Direction Modifier |
|-----------------|-------------------|----------------|-------------------|
| Cause (why?) | `query.e5_as_cause` | `E5_EFFECT_INDEX` | 1.2x (cause→effect) |
| Effect (what happens?) | `query.e5_as_effect` | `E5_CAUSE_INDEX` | 0.8x (effect→cause) |
| Unknown | `query.e5_active` | `E5_ACTIVE_INDEX` (legacy) | 1.0x |

---

## Implementation Plan

### Phase 1: Add New Index Types

**File:** `crates/context-graph-storage/src/teleological/indexes/hnsw_config/embedder.rs`

```rust
/// Embedder index variants for HNSW indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbedderIndex {
    // ... existing indexes ...

    /// E5 Causal - Active vector (legacy, for backward compatibility)
    E5Causal,

    /// E5 Causal - Cause vector only (NEW)
    /// Used when query seeks effects (what happens when X?)
    E5CausalCause,

    /// E5 Causal - Effect vector only (NEW)
    /// Used when query seeks causes (why does X happen?)
    E5CausalEffect,
}

impl EmbedderIndex {
    /// All HNSW-capable embedder indexes including new E5 dual indexes.
    pub fn all_hnsw() -> &'static [EmbedderIndex] {
        &[
            Self::E1Semantic,
            Self::E1Matryoshka128,
            Self::E2TemporalRecent,
            Self::E3TemporalPeriodic,
            Self::E4TemporalPositional,
            Self::E5Causal,        // Legacy (active vector)
            Self::E5CausalCause,   // NEW
            Self::E5CausalEffect,  // NEW
            Self::E7Code,
            Self::E8Graph,
            Self::E9HDC,
            Self::E10Multimodal,
            Self::E11Entity,
        ]
    }

    /// E5 dual indexes only (for asymmetric retrieval).
    pub fn e5_dual_indexes() -> &'static [EmbedderIndex] {
        &[Self::E5CausalCause, Self::E5CausalEffect]
    }
}
```

### Phase 2: Configure HNSW for New Indexes

**File:** `crates/context-graph-storage/src/teleological/indexes/hnsw_config/functions.rs`

```rust
/// Get HNSW configuration for an embedder index.
pub fn get_hnsw_config(embedder: EmbedderIndex) -> HnswConfig {
    match embedder {
        // ... existing configs ...

        // E5 Causal dual indexes - same config as E5Causal
        EmbedderIndex::E5CausalCause | EmbedderIndex::E5CausalEffect => HnswConfig {
            dimension: 768,
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            distance_metric: DistanceMetric::Cosine,
        },

        // ... rest ...
    }
}
```

### Phase 3: Update Vector Extraction

**File:** `crates/context-graph-storage/src/teleological/rocksdb_store/index_ops.rs`

```rust
/// Extract vector for specific embedder from SemanticFingerprint.
pub(crate) fn get_embedder_vector(
    semantic: &SemanticFingerprint,
    embedder: EmbedderIndex,
) -> &[f32] {
    match embedder {
        // ... existing cases ...

        // Legacy E5 (active vector based on magnitude)
        EmbedderIndex::E5Causal => semantic.e5_active_vector(),

        // NEW: E5 Cause vector
        EmbedderIndex::E5CausalCause => &semantic.e5_causal_as_cause,

        // NEW: E5 Effect vector
        EmbedderIndex::E5CausalEffect => &semantic.e5_causal_as_effect,

        // ... rest ...
    }
}
```

### Phase 4: Update Index Registration

**File:** `crates/context-graph-storage/src/teleological/rocksdb_store/store.rs`

```rust
impl RocksDbTeleologicalStore {
    /// Initialize all HNSW indexes including new E5 dual indexes.
    fn init_indexes(&mut self) -> Result<(), TeleologicalStoreError> {
        for embedder in EmbedderIndex::all_hnsw() {
            let config = get_hnsw_config(embedder);
            let index = HnswEmbedderIndex::new(config)?;
            self.index_registry.register(embedder, index);
        }
        Ok(())
    }
}
```

### Phase 5: Update Search to Use Dual Indexes

**File:** `crates/context-graph-storage/src/teleological/rocksdb_store/search.rs`

```rust
impl RocksDbTeleologicalStore {
    /// Search E5 with direction-aware index selection.
    ///
    /// Per ARCH-15: Asymmetric E5 retrieval uses complementary indexes.
    /// - Cause query → search EFFECT index (find docs describing effects)
    /// - Effect query → search CAUSE index (find docs describing causes)
    async fn search_e5_asymmetric(
        &self,
        query: &SemanticFingerprint,
        direction: CausalDirection,
        k: usize,
    ) -> CoreResult<Vec<(Uuid, f32)>> {
        let (query_vector, index_type) = match direction {
            // Cause-seeking: use query's cause vector, search effect index
            CausalDirection::Cause => {
                (&query.e5_causal_as_cause, EmbedderIndex::E5CausalEffect)
            }
            // Effect-seeking: use query's effect vector, search cause index
            CausalDirection::Effect => {
                (&query.e5_causal_as_effect, EmbedderIndex::E5CausalCause)
            }
            // Unknown: fall back to legacy symmetric search
            CausalDirection::Unknown => {
                (query.e5_active_vector(), EmbedderIndex::E5Causal)
            }
        };

        let index = self.index_registry.get(index_type).ok_or_else(|| {
            CoreError::IndexError(format!("Index {:?} not found", index_type))
        })?;

        let candidates = index.search(query_vector, k, None).map_err(|e| {
            CoreError::IndexError(format!("E5 asymmetric search failed: {}", e))
        })?;

        // Convert distance to similarity
        Ok(candidates
            .into_iter()
            .map(|(id, dist)| (id, 1.0 - dist.min(1.0)))
            .collect())
    }

    /// Multi-space search with asymmetric E5 retrieval.
    async fn search_multi_space(
        &self,
        query: &SemanticFingerprint,
        options: &TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        let weights = self.resolve_weights(options);
        let k = (options.top_k * 3).max(50);

        let mut embedder_rankings: Vec<EmbedderRanking> = Vec::new();

        // E1 Semantic (unchanged)
        // ... existing E1 code ...

        // E5 Causal with ASYMMETRIC retrieval (CHANGED)
        if weights[4] > 0.0 {
            let e5_candidates = self.search_e5_asymmetric(
                query,
                options.causal_direction,  // Use direction from options
                k,
            ).await?;

            let e5_ranked: Vec<(Uuid, f32)> = e5_candidates
                .into_iter()
                .filter(|(id, _)| options.include_deleted || !self.is_soft_deleted(id))
                .collect();

            if !e5_ranked.is_empty() {
                embedder_rankings.push(EmbedderRanking::new("E5", weights[4], e5_ranked));
            }
        }

        // ... rest of multi-space search ...
    }
}
```

### Phase 6: Update Pipeline Search

**File:** `crates/context-graph-storage/src/teleological/rocksdb_store/search.rs`

```rust
async fn search_pipeline(
    &self,
    query: &SemanticFingerprint,
    options: &TeleologicalSearchOptions,
) -> CoreResult<Vec<TeleologicalSearchResult>> {
    let recall_k = options.top_k * STAGE1_RECALL_MULTIPLIER;

    // STAGE 1: FAST RECALL with asymmetric E5
    let mut candidate_ids: HashSet<Uuid> = HashSet::new();

    // E13 SPLADE (unchanged)
    // ... existing code ...

    // E1 Semantic HNSW (unchanged)
    // ... existing code ...

    // E5 Causal with ASYMMETRIC retrieval (CHANGED)
    let e5_candidates = self.search_e5_asymmetric(
        query,
        options.causal_direction,
        recall_k / 2,
    ).await?;

    let e5_count = e5_candidates.len();
    candidate_ids.extend(e5_candidates.into_iter().map(|(id, _)| id));
    debug!("Stage 1: E5 asymmetric returned {} candidates", e5_count);

    // ... rest of pipeline ...
}
```

---

## Migration Strategy

### Step 1: Add Indexes Without Breaking Changes

1. Add `E5CausalCause` and `E5CausalEffect` as NEW indexes
2. Keep `E5Causal` (legacy) for backward compatibility
3. Both old and new code continues to work

### Step 2: Backfill Existing Data

```rust
/// Backfill E5 dual indexes from existing fingerprints.
async fn backfill_e5_dual_indexes(&self) -> CoreResult<u64> {
    let mut count = 0u64;

    for (id, fp) in self.iter_all_fingerprints()? {
        // Add to cause index
        if let Some(cause_index) = self.index_registry.get(EmbedderIndex::E5CausalCause) {
            cause_index.insert(id, &fp.semantic.e5_causal_as_cause)?;
        }

        // Add to effect index
        if let Some(effect_index) = self.index_registry.get(EmbedderIndex::E5CausalEffect) {
            effect_index.insert(id, &fp.semantic.e5_causal_as_effect)?;
        }

        count += 1;
    }

    info!("Backfilled {} fingerprints to E5 dual indexes", count);
    Ok(count)
}
```

### Step 3: Update Store Operations

Update `add_to_indexes` to insert into all three E5 indexes:

```rust
pub(crate) fn add_to_indexes(&self, fp: &TeleologicalFingerprint) -> Result<(), IndexError> {
    let id = fp.id;

    for embedder in EmbedderIndex::all_hnsw() {
        if let Some(index) = self.index_registry.get(embedder) {
            let vector = Self::get_embedder_vector(&fp.semantic, embedder);
            index.insert(id, vector)?;
        }
    }

    Ok(())
}
```

### Step 4: Deprecate Legacy E5Causal

After migration is complete and verified:

1. Mark `E5Causal` as deprecated
2. Log warnings when legacy index is used
3. Eventually remove in future version

---

## Storage Impact

| Index | Dimension | Per-Document Size | 1M Documents |
|-------|-----------|-------------------|--------------|
| E5Causal (legacy) | 768 | 3 KB | 3 GB |
| E5CausalCause (new) | 768 | 3 KB | 3 GB |
| E5CausalEffect (new) | 768 | 3 KB | 3 GB |
| **Total E5 Storage** | - | **9 KB** | **9 GB** |

**Overhead:** 2x additional E5 storage (6 GB for 1M docs)

---

## Performance Considerations

### Query Time

| Operation | Before | After | Change |
|-----------|--------|-------|--------|
| E5 HNSW search | 1 search | 1 search | Same |
| Index selection | None | O(1) | Negligible |
| **Total** | ~5ms | ~5ms | **No change** |

The direction-based index selection is O(1) and doesn't add latency.

### Index Build Time

| Operation | Before | After | Change |
|-----------|--------|-------|--------|
| E5 index inserts | 1 insert | 3 inserts | 3x |
| Index build | ~10ms/doc | ~30ms/doc | 3x |

Insert operations are 3x slower due to maintaining 3 indexes.

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_asymmetric_retrieval_finds_complementary_docs() {
    // Create cause-dominant query
    let query = create_fingerprint_with_direction(CausalDirection::Cause);

    // Create effect-dominant document (should be found by cause query)
    let effect_doc = create_fingerprint_with_direction(CausalDirection::Effect);

    // Insert into dual indexes
    store.add_to_indexes(&effect_doc)?;

    // Search with cause direction
    let results = store.search_e5_asymmetric(
        &query,
        CausalDirection::Cause,
        10,
    ).await?;

    // Should find the effect document
    assert!(results.iter().any(|(id, _)| *id == effect_doc.id));
}

#[test]
fn test_symmetric_misses_complementary_docs() {
    // Same setup as above
    let query = create_fingerprint_with_direction(CausalDirection::Cause);
    let effect_doc = create_fingerprint_with_direction(CausalDirection::Effect);
    store.add_to_indexes(&effect_doc)?;

    // Search with symmetric (legacy) method
    let results = store.search_e5_symmetric(&query, 10).await?;

    // May NOT find the effect document (this is the bug we're fixing)
    // This test documents the limitation of symmetric search
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_causal_query_recall_improvement() {
    // Insert known causal relationships
    let cause_doc = insert_doc("The server crashes because of memory leaks");
    let effect_doc = insert_doc("Memory leaks lead to server crashes");

    // Query seeking causes
    let query = "why does the server crash";

    // Compare recall with and without asymmetric retrieval
    let symmetric_results = search_symmetric(query, 10).await;
    let asymmetric_results = search_asymmetric(query, CausalDirection::Cause, 10).await;

    // Asymmetric should have better recall for causal relationships
    let symmetric_recall = compute_recall(&symmetric_results, &[cause_doc, effect_doc]);
    let asymmetric_recall = compute_recall(&asymmetric_results, &[cause_doc, effect_doc]);

    assert!(asymmetric_recall >= symmetric_recall);
}
```

---

## Rollout Plan

### Week 1: Infrastructure
- [ ] Add `EmbedderIndex::E5CausalCause` and `E5CausalEffect`
- [ ] Configure HNSW for new indexes
- [ ] Update `get_embedder_vector()` for new indexes
- [ ] Update `add_to_indexes()` to populate all three

### Week 2: Migration
- [ ] Implement `backfill_e5_dual_indexes()`
- [ ] Run backfill on dev environment
- [ ] Verify index integrity
- [ ] Measure storage overhead

### Week 3: Search Integration
- [ ] Implement `search_e5_asymmetric()`
- [ ] Update `search_multi_space()` to use asymmetric E5
- [ ] Update `search_pipeline()` to use asymmetric E5
- [ ] Add unit tests

### Week 4: Validation
- [ ] Run recall benchmarks
- [ ] Compare symmetric vs asymmetric on causal queries
- [ ] Performance testing
- [ ] Documentation

---

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Cause query recall | +25% improvement | Benchmark dataset |
| Effect query recall | +25% improvement | Benchmark dataset |
| Query latency | <5% increase | p95 latency |
| Storage overhead | <3x E5 storage | Disk usage |
| Test coverage | 100% new code | Code coverage |

---

## Appendix: Why This Works

### The Math Behind Asymmetric Retrieval

Given:
- Query Q with cause vector Q_c and effect vector Q_e
- Document D with cause vector D_c and effect vector D_e

**Symmetric search (broken):**
```
sim(Q, D) = cosine(Q_active, D_active)
```
This compares whatever vector has higher magnitude, ignoring causal direction.

**Asymmetric search (correct):**
```
For cause-seeking query:
  sim(Q, D) = cosine(Q_c, D_e) × 1.2

For effect-seeking query:
  sim(Q, D) = cosine(Q_e, D_c) × 0.8
```
This compares complementary vectors, finding documents that complete the causal relationship.

### Example

Query: "why does the server crash" (Cause-seeking)
- Q_c = embedding of "why does the server crash" as a CAUSE inquiry
- Q_e = embedding of "why does the server crash" as an EFFECT inquiry

Document: "Memory leaks lead to crashes"
- D_c = embedding emphasizing "memory leaks" as CAUSE
- D_e = embedding emphasizing "crashes" as EFFECT

**Symmetric:** cosine(Q_active, D_active) → may miss if vectors aren't aligned
**Asymmetric:** cosine(Q_c, D_e) → high similarity (query's cause matches doc's effect)

The document describes an effect (crashes) that matches what our cause query is asking about.

---

## References

- [Asymmetric Semantic Search](https://www.sbert.net/examples/applications/semantic-search/README.html)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
- [Causal Embedding Models](https://arxiv.org/abs/2104.08914)
- Constitution: ARCH-15, AP-77
