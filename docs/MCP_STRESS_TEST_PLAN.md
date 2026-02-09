# Context Graph MCP Server - Comprehensive Stress Test Plan

## Overview

This plan covers stress testing all 55 MCP tools across 11 categories with focus on:
- **Correctness under load** - Tools return accurate results when hammered
- **Concurrency safety** - No data races, deadlocks, or corruption
- **Resource limits** - Behavior at VRAM, disk, memory boundaries
- **Pipeline resilience** - Multi-tool workflows under stress
- **Recovery** - Graceful handling of failures and corrupt data

### Test Infrastructure

| Component | Details |
|-----------|---------|
| DB isolation | `TempDir` per test, auto-cleanup |
| Embeddings | `ProductionMultiArrayProvider` warm singleton (RTX 5090, 32GB VRAM) |
| Helpers | `create_test_handlers()`, `create_test_handlers_with_rocksdb_store_access()` |
| Framework | `#[tokio::test]`, `criterion` benchmarks |
| Env vars | `CONTEXT_GRAPH_STORAGE_PATH`, `CONTEXT_GRAPH_MODELS_PATH` |

### Tool Count by Category

| Category | Tools | IDs |
|----------|-------|-----|
| Core Memory | 4 | store_memory, get_memetic_status, search_graph, trigger_consolidation |
| Topic Detection | 4 | get_topic_portfolio, get_topic_stability, detect_topics, get_divergence_alerts |
| Curation | 3 | merge_concepts, forget_concept, boost_importance |
| File Watcher | 4 | list_watched_files, get_file_watcher_stats, delete_file_content, reconcile_files |
| Sequence/Temporal Nav | 4 | get_conversation_context, get_session_timeline, traverse_memory_chain, compare_session_states |
| Causal | 4 | search_causal_relationships, search_causes, search_effects, get_causal_chain |
| Causal Discovery | 2 | trigger_causal_discovery, get_causal_discovery_status |
| Graph | 4 | search_connections, get_graph_path, discover_graph_relationships, validate_graph_link |
| Keyword/Code/Robust | 3 | search_by_keywords, search_code, search_robust |
| Entity KG | 6 | extract_entities, search_by_entities, infer_relationship, find_related_entities, validate_knowledge, get_entity_graph |
| Embedder-First | 7 | search_by_embedder, get_embedder_clusters, compare_embedder_views, list_embedder_indexes, get_memory_fingerprint, create_weight_profile, search_cross_embedder_anomalies |
| Temporal Search | 2 | search_recent, search_periodic |
| Graph Linking | 4 | get_memory_neighbors, get_typed_edges, traverse_graph, get_unified_neighbors |
| Maintenance | 1 | repair_causal_relationships |
| Provenance | 3 | get_audit_trail, get_merge_history, get_provenance_chain |
| **Total** | **55** | |

---

## Phase 0: Data Seeding Infrastructure

Before stress tests run, a shared seeding phase populates the database with realistic data at various scales.

### S0.1 - Seed Tiers

| Tier | Memories | Causal Rels | Entities | Topics | Purpose |
|------|----------|-------------|----------|--------|---------|
| **Small** | 50 | 20 | 30 | 3 | Smoke tests, correctness validation |
| **Medium** | 500 | 200 | 150 | 10 | Functional stress, concurrency |
| **Large** | 5,000 | 1,000 | 500 | 30 | Performance regression, throughput |
| **XL** | 50,000 | 10,000 | 2,000 | 100 | Saturation, resource limits |

### S0.2 - Seed Content Categories

Each tier includes diverse content to exercise all 13 embedders:

| Content Type | Exercises | Count (% of tier) |
|-------------|-----------|-------------------|
| Natural language prose | E1, E10, E11, E12, E13 | 30% |
| Source code (Rust, Python, JS) | E7 code embeddings | 20% |
| Causal descriptions ("X causes Y") | E5 asymmetric | 15% |
| Entity-dense text (tech terms) | E11 KEPLER, E6 keywords | 15% |
| Temporal/session-bound content | E2, E3, E4 | 10% |
| Graph-structured relationships | E8 asymmetric | 10% |

### S0.3 - Seeding Implementation

```rust
async fn seed_tier(handlers: &Handlers, tier: SeedTier) -> SeedResult {
    let mut ids = Vec::new();
    for batch in tier.batches(BATCH_SIZE) {
        for item in batch {
            let params = json!({
                "name": "store_memory",
                "arguments": {
                    "content": item.content,
                    "rationale": item.rationale,
                    "source_type": item.source_type,
                }
            });
            let request = make_request("tools/call", Some(id), Some(params));
            let response = handlers.dispatch(request).await;
            ids.push(extract_id(&response));
        }
    }
    SeedResult { ids, tier }
}
```

---

## Phase 1: Individual Tool Stress Tests

### 1.1 Core Memory Tools

#### ST-1.1.1: `store_memory` - Burst Ingestion

| Test | Description | Parameters | Pass Criteria |
|------|-------------|------------|---------------|
| **Burst-100** | 100 stores in tight loop | content: 100-500 chars each | All succeed, unique IDs |
| **Burst-1K** | 1,000 stores sequential | Mixed content types | < 10s total, no failures |
| **Burst-10K** | 10,000 stores in batches of 100 | Mixed content, varied rationale | < 120s, no OOM |
| **Large-content** | Single store with 100KB content | content: 100,000 chars | Succeeds, searchable |
| **Unicode-stress** | CJK, emoji, RTL, combining chars | content: mixed scripts | Correct round-trip |
| **Empty-edge** | Empty string content | content: "" | Graceful error |
| **Duplicate-content** | Same content stored 50x | Identical strings | 50 unique IDs, dedup in consolidation |
| **Rapid-fire** | 50 concurrent tokio::spawn stores | Parallel execution | No data races, all succeed |
| **Source-types** | All source_type variants | Manual, HookDescription, ClaudeResponse, MDFileChunk | Each type stored correctly |

#### ST-1.1.2: `get_memetic_status` - State Reporting

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Empty-DB** | Call on fresh database | Returns 0 counts, no crash |
| **After-1** | After 1 store | Reflects 1 memory |
| **After-10K** | After 10K stores | Correct counts, < 100ms |
| **Concurrent-read** | 50 parallel status calls | All return consistent state |
| **Post-merge** | After merge_concepts | Counts reflect merge |
| **Post-delete** | After forget_concept | Counts reflect soft delete |

#### ST-1.1.3: `search_graph` - Multi-Space Retrieval

| Test | Description | Parameters | Pass Criteria |
|------|-------------|------------|---------------|
| **Basic-10** | Search 10-memory DB | query: "test", topK: 5 | Returns ≤5 results |
| **Basic-5K** | Search 5K-memory DB | query: "authentication patterns" | Results in < 500ms |
| **All-profiles** | Each of 14 weight profiles | semantic_search through pipeline_full | All return results, different rankings |
| **Custom-weights** | Manual E1-E13 weights | weights: {E1: 0.5, E7: 0.5} | Respects custom distribution |
| **TopK-boundary** | topK: 1, 10, 50, 100 | Boundary values | Correct result counts |
| **Min-similarity-0** | minSimilarity: 0.0 | Returns all memories | Max results |
| **Min-similarity-1** | minSimilarity: 1.0 | Near-impossible threshold | 0 or very few results |
| **Causal-direction** | direction: cause, effect, none | E5 asymmetric activation | Different rankings per direction |
| **Code-query** | query: "async fn main()" | Code content in DB | E7 boosts code results |
| **Typo-query** | query: "authentcation" (misspelled) | E9 HDC handles typos | Still finds "authentication" |
| **Empty-query** | query: "" | Edge case | Graceful error or empty results |
| **Long-query** | query: 10,000 chars | Stress embedding pipeline | Succeeds or truncates gracefully |
| **Concurrent-50** | 50 parallel searches | Different queries | All return, no deadlock |
| **Session-scope** | sessionScope: current, all, recent | E4 filtering | Correct scoping |
| **Rerank-on** | enableRerank: true | E12 ColBERT re-ranking | Different order than without |
| **Conversation-ctx** | conversationContext: {turnsBack: 5} | E4 sequence-based | Sequence-ordered results |

#### ST-1.1.4: `trigger_consolidation` - Merge Under Load

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Small-batch** | 10 similar memories, strategy: similarity | Merges detected, count reduced |
| **Large-batch** | 1,000 memories, max_memories: 1000 | Completes < 30s |
| **All-strategies** | similarity, temporal, semantic | Each produces different merges |
| **High-threshold** | min_similarity: 0.99 | Minimal or no merges |
| **Low-threshold** | min_similarity: 0.5 | Aggressive merging |
| **Concurrent-trigger** | 5 parallel consolidation triggers | No crash, consistent final state |
| **Post-consolidation-search** | Search after consolidation | Merged memories searchable |

---

### 1.2 Topic Detection Tools

#### ST-1.2.1: `detect_topics` - HDBSCAN Clustering

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Min-data** | 3 memories (HDBSCAN minimum) | At least 1 topic detected |
| **Below-min** | 2 memories | Graceful: 0 topics or informative message |
| **100-memories** | 100 diverse memories | Multiple topics, no mega-cluster |
| **5K-memories** | 5,000 memories | Topics detected < 5s |
| **Force-recompute** | force: true after prior detection | New results computed |
| **Homogeneous** | 50 near-identical memories | 1 topic, no noise |
| **Heterogeneous** | 50 memories, 10 distinct themes | ~10 topics |
| **Concurrent-detect** | 3 parallel detect_topics | No crash, consistent clustering |

#### ST-1.2.2: `get_topic_portfolio`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **All-formats** | brief, standard, verbose | Increasing detail levels |
| **Empty-DB** | No topics detected yet | Empty portfolio, no crash |
| **After-detect** | After detect_topics on 100 memories | Topics with stability metrics |
| **Concurrent-read** | 20 parallel portfolio reads | Consistent results |

#### ST-1.2.3: `get_topic_stability`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Default-hours** | hours: 6 (default) | Churn rate, entropy returned |
| **Boundary-hours** | hours: 1, hours: 168 | Valid responses at boundaries |
| **Evolution** | Store 50 memories, detect, store 50 more, detect | Stability changes reflected |

#### ST-1.2.4: `get_divergence_alerts`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **No-divergence** | Consistent recent activity | No alerts |
| **Topic-shift** | Store code, then store prose | Alert about content shift |
| **Lookback-boundary** | lookback_hours: 1, 48 | Alerts scale with window |

---

### 1.3 Curation Tools

#### ST-1.3.1: `merge_concepts` - Merge Stress

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **2-way-merge** | Merge 2 concepts | Unified node, reversal_hash |
| **10-way-merge** | Merge 10 concepts (max) | All sources consumed |
| **11-way-merge** | Merge 11 concepts (over max) | Graceful error |
| **All-strategies** | union, intersection, weighted_average | Different merge results |
| **Force-merge** | Conflicting priors, force: true | Succeeds despite conflicts |
| **Invalid-IDs** | Non-existent UUIDs in source_ids | Error: "not found" |
| **Merge-chain** | Merge A+B→C, then C+D→E | Chain works, provenance tracks |
| **Concurrent-merge** | 10 parallel merges of disjoint pairs | All succeed independently |
| **Overlapping-merge** | 2 merges sharing a source_id | One succeeds, other errors |
| **Post-merge-search** | Search for merged content | Finds merged result |

#### ST-1.3.2: `forget_concept` - Deletion Stress

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Soft-delete** | soft_delete: true | deleted_at timestamp, 30-day window |
| **Hard-delete** | soft_delete: false | Permanently removed |
| **Double-delete** | Delete same ID twice | Second call: graceful error |
| **Invalid-ID** | Non-existent UUID | Error message |
| **Bulk-delete** | Delete 100 memories sequentially | All deleted, search excludes them |
| **Concurrent-delete** | 20 parallel deletes of distinct IDs | All succeed |
| **Delete-then-search** | Delete, then search for deleted content | Not found in results |
| **Delete-then-status** | Delete, then get_memetic_status | Count decremented |

#### ST-1.3.3: `boost_importance` - Score Adjustment

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Boost-up** | delta: 0.5 | new = old + 0.5 |
| **Boost-down** | delta: -0.3 | new = old - 0.3 |
| **Clamp-high** | delta: 1.0 on memory at 0.8 | Clamped to 1.0 |
| **Clamp-low** | delta: -1.0 on memory at 0.2 | Clamped to 0.0 |
| **Zero-delta** | delta: 0.0 | No change |
| **Boundary-delta** | delta: 1.0 and delta: -1.0 | Clamp works at extremes |
| **Rapid-boost** | 100 sequential boosts on same ID | Final value correct |
| **Concurrent-boost** | 20 parallel boosts on same ID, delta: 0.01 | Final value ≈ old + 0.20 (within tolerance) |
| **Invalid-ID** | Non-existent UUID | Error message |
| **Post-boost-search** | Boost a memory, then search | Boosted memory ranks higher |

---

### 1.4 File Watcher Tools

#### ST-1.4.1: `list_watched_files`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Empty** | No file content stored | Empty list |
| **With-files** | Store file-sourced memories | Files listed with counts |
| **Path-filter** | path_filter: "**/*.rs" | Only Rust files |
| **Include-counts** | include_counts: true/false | Chunk counts present/absent |

#### ST-1.4.2: `get_file_watcher_stats`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Empty** | No file content | Zero stats |
| **With-data** | After storing file chunks | Correct totals, averages |

#### ST-1.4.3: `delete_file_content`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Delete-file** | Delete known file's chunks | All chunks removed |
| **Non-existent** | Delete unknown file path | Graceful: 0 deleted |
| **Soft-delete** | soft_delete: true | 30-day recovery |

#### ST-1.4.4: `reconcile_files`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Dry-run** | dry_run: true | Preview only, no modifications |
| **With-orphans** | Delete source file, keep embeddings | Orphans detected |
| **Clean-state** | All files exist | No orphans found |

---

### 1.5 Sequence/Temporal Navigation Tools

#### ST-1.5.1: `get_conversation_context`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Default** | windowSize: 10, direction: before | Returns up to 10 recent |
| **All-directions** | before, after, both | Correct ordering |
| **With-query** | Semantic filter + sequence ordering | Filtered by both |
| **No-session** | Without CLAUDE_SESSION_ID | Graceful error or empty |
| **Large-window** | windowSize: 50 | Returns up to 50 |

#### ST-1.5.2: `get_session_timeline`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Full-timeline** | limit: 200 | All memories ordered |
| **Pagination** | offset: 0 limit: 10, then offset: 10 limit: 10 | Pages don't overlap |
| **Source-filter** | sourceTypes: ["Manual"] | Only manual memories |
| **No-session** | No CLAUDE_SESSION_ID | Error or empty |

#### ST-1.5.3: `traverse_memory_chain`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Forward-5** | direction: forward, hops: 5 | Chain of ≤5 memories |
| **Backward-5** | direction: backward, hops: 5 | Reverse chain |
| **Bidirectional** | direction: bidirectional | Both directions |
| **Max-hops** | hops: 20 | Doesn't hang or OOM |
| **Min-similarity** | minSimilarity: 0.9 | Short chain (strict) |
| **Semantic-filter** | semanticFilter: "authentication" | Only related hops |
| **Invalid-anchor** | Non-existent anchorId | Error message |
| **Dead-end** | Anchor with no neighbors | Returns anchor only |

#### ST-1.5.4: `compare_session_states`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Start-to-current** | beforeSequence: "start", afterSequence: "current" | Full session diff |
| **Narrow-range** | beforeSequence: 5, afterSequence: 10 | Subset comparison |
| **Topic-filter** | topicFilter: "authentication" | Focused comparison |
| **No-session** | No CLAUDE_SESSION_ID | Graceful handling |

---

### 1.6 Causal Tools

#### ST-1.6.1: `search_causal_relationships`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Cause-direction** | direction: cause | E5 asymmetric cause mode |
| **Effect-direction** | direction: effect | E5 asymmetric effect mode |
| **All-direction** | direction: all | Both directions |
| **TopK-boundary** | topK: 1, 50, 100 | Correct counts |
| **Empty-DB** | No causal relationships | Empty results, no crash |
| **Concurrent-10** | 10 parallel causal searches | All return correctly |

#### ST-1.6.2: `search_causes` (Abductive)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Basic** | query: "memory problems" | Returns plausible causes |
| **High-min** | minScore: 0.9 | Fewer, higher-confidence results |
| **With-content** | includeContent: true | Content included in response |
| **Scope-all** | searchScope: all (memories + relationships) | Broader results |
| **0.8x-dampening** | Verify effect→cause dampening | Scores reflect AP-77 dampening |

#### ST-1.6.3: `search_effects` (Predictive)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Basic** | query: "increase cache size" | Returns effects |
| **1.2x-boost** | Verify cause→effect boost | Scores reflect AP-77 boost |
| **Scope-relationships** | searchScope: relationships | Only relationship results |

#### ST-1.6.4: `get_causal_chain`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Forward-chain** | direction: forward, maxHops: 5 | Transitive chain |
| **Backward-chain** | direction: backward | Reverse causality |
| **Hop-attenuation** | Verify 0.9^hop scoring | Scores decrease per hop |
| **Max-hops-10** | maxHops: 10 | Doesn't hang |
| **No-chain** | Anchor with no causal links | Returns anchor only |
| **With-content** | includeContent: true | Content at each hop |

---

### 1.7 Causal Discovery Tools

#### ST-1.7.1: `trigger_causal_discovery`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Basic-run** | Default params, 50-memory DB | Discovers relationships |
| **Dry-run** | dryRun: true | Analyzes without persisting |
| **Skip-analyzed** | skipAnalyzed: true | Skips prior pairs |
| **Max-pairs** | maxPairs: 200 | Processes up to 200 |
| **High-confidence** | minConfidence: 0.9 | Fewer, high-quality results |
| **Low-threshold** | similarityThreshold: 0.3 | More candidate pairs |
| **Concurrent-trigger** | 2 parallel triggers | One runs, other queues or errors |

#### ST-1.7.2: `get_causal_discovery_status`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Before-run** | Before any discovery | Status: idle |
| **During-run** | While discovery runs | Status: running |
| **After-run** | After discovery completes | Results + stats |
| **Graph-stats** | includeGraphStats: true | Node/edge counts |

---

### 1.8 Graph Tools

#### ST-1.8.1: `search_connections`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Source-direction** | direction: source | Only outgoing |
| **Target-direction** | direction: target | Only incoming |
| **Both-directions** | direction: both | All connections |
| **E8-asymmetric** | Verify asymmetric scoring | Different rankings source vs target |

#### ST-1.8.2: `get_graph_path`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Forward-path** | direction: forward, maxHops: 5 | Multi-hop path |
| **Backward-path** | direction: backward | Reverse path |
| **Hop-attenuation** | 0.9^hop scoring | Decreasing scores |

#### ST-1.8.3: `discover_graph_relationships`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **2-memories** | Minimum pair | Relationship found or not |
| **50-memories** | Maximum allowed | Batch discovery |
| **Type-filter** | relationship_types: ["imports", "extends"] | Only those types |
| **Category-filter** | relationship_categories: ["dependency"] | Category filtering |
| **Domain-hint** | content_domain: "code" | Code-aware discovery |
| **High-confidence** | min_confidence: 0.9 | Fewer results |

#### ST-1.8.4: `validate_graph_link`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Valid-link** | Related memories | High confidence |
| **Invalid-link** | Unrelated memories | Low confidence |
| **With-expected-type** | expected_relationship_type set | Validates specific type |
| **Non-existent-IDs** | Invalid UUIDs | Error handling |

---

### 1.9 Specialized Search Tools

#### ST-1.9.1: `search_by_keywords` (E6 Sparse)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Pure-keyword** | blendWithSemantic: 1.0 | E6-only results |
| **Pure-semantic** | blendWithSemantic: 0.0 | E1-only results |
| **Blended** | blendWithSemantic: 0.3 | Mixed ranking |
| **SPLADE-on** | useSpladeExpansion: true | E13 term expansion |
| **SPLADE-off** | useSpladeExpansion: false | No expansion |
| **Exact-keyword** | Known exact term in DB | Found in top results |
| **Concurrent-20** | 20 parallel keyword searches | All return correctly |

#### ST-1.9.2: `search_code` (E7)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Rust-code** | query: "impl Display for" | Finds Rust implementations |
| **Python-code** | query: "def __init__(self" | Finds Python constructors |
| **Blended** | blendWithSemantic: 0.4 | E7 + E1 mix |
| **Code-only** | blendWithSemantic: 1.0 | Pure E7 ranking |
| **Natural-language** | query: "error handling in async" | Semantic + code blend |

#### ST-1.9.3: `search_robust` (E9 HDC)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Typo-query** | query: "authentcation" | Finds "authentication" |
| **Case-variation** | query: "TOKIO RUNTIME" | Finds "tokio runtime" |
| **Morphological** | query: "authenticating" | Finds "authentication" |
| **Min-length** | query: "ab" (2 chars, under min 3) | Graceful error |
| **E9-discovery** | e9DiscoveryThreshold: 0.7 | Blind spot markers |
| **E1-weakness** | e1WeaknessThreshold: 0.5 | E1 miss detection |

---

### 1.10 Entity Knowledge Graph Tools

#### ST-1.10.1: `extract_entities`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Tech-text** | "Using PostgreSQL with Redis caching" | Extracts postgres, redis |
| **Canonicalization** | "postgres", "pg", "PostgreSQL" | All → "postgresql" |
| **Group-by-type** | groupByType: true | Grouped output |
| **Unknown-entities** | includeUnknown: true | Heuristic detection |
| **Empty-text** | text: "" | No entities |
| **Dense-text** | 50 entity mentions in 1 paragraph | All extracted |
| **Concurrent-20** | 20 parallel extractions | All return correctly |

#### ST-1.10.2: `search_by_entities`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Single-entity** | entities: ["rust"] | Memories mentioning Rust |
| **Multi-any** | entities: ["rust", "python"], matchMode: any | Union |
| **Multi-all** | entities: ["rust", "tokio"], matchMode: all | Intersection |
| **Exact-boost** | boostExactMatch: 2.0 | Exact matches ranked higher |
| **Type-filter** | entityTypes: ["Framework"] | Only framework entities |

#### ST-1.10.3: `infer_relationship` (TransE)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Known-relation** | head: "Tokio", tail: "Rust" | Plausible relations |
| **Type-hints** | headType, tailType provided | More focused inference |
| **TopK** | topK: 1, 5, 20 | Correct counts |

#### ST-1.10.4: `find_related_entities` (TransE)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Outgoing** | entity: "Rust", relation: "has_framework", direction: outgoing | Finds frameworks |
| **Incoming** | direction: incoming | Reverse lookup |
| **With-memories** | searchMemories: true | Only entities in stored memories |
| **Type-filter** | entityType: "Framework" | Filtered results |

#### ST-1.10.5: `validate_knowledge` (TransE)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Valid-triple** | subject: "Tokio", predicate: "is_framework_for", object: "Rust" | Result: valid/uncertain |
| **Invalid-triple** | subject: "Rust", predicate: "compiles_to", object: "JavaScript" | Result: unlikely |
| **Unknown-entities** | Novel entities not in graph | Result: uncertain |

#### ST-1.10.6: `get_entity_graph`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Centered** | centerEntity: "Rust" | Graph around Rust |
| **No-center** | No centerEntity | Full entity graph |
| **Depth-1** | maxDepth: 1 | Immediate neighbors |
| **Depth-5** | maxDepth: 5 (max) | Deep graph |
| **Max-nodes** | maxNodes: 500 (max) | Doesn't exceed |
| **Type-filter** | entityTypes: ["Framework", "Database"] | Filtered graph |
| **Empty-graph** | No entity data | Empty graph, no crash |

---

### 1.11 Embedder-First Search Tools

#### ST-1.11.1: `search_by_embedder`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Each-embedder** | E1 through E13, same query | 13 different rankings |
| **All-scores** | includeAllScores: true | 13 scores per result |
| **High-topK** | topK: 100 | Large result set |
| **Min-similarity** | minSimilarity: 0.8 | Strict filtering |
| **Concurrent-13** | All 13 embedders in parallel | All return correctly |

#### ST-1.11.2: `get_embedder_clusters`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Each-embedder** | E1, E7, E11 clustering | Clusters formed |
| **Min-cluster-2** | minClusterSize: 2 | Smaller clusters |
| **Min-cluster-50** | minClusterSize: 50 | Fewer, larger clusters |
| **Samples** | includeSamples: true, samplesPerCluster: 5 | Sample memories included |
| **Known-limitation** | Verify known limitation behavior | Documented in results |

#### ST-1.11.3: `compare_embedder_views`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **2-embedders** | embedders: [E1, E7] | Agreement/disagreement shown |
| **5-embedders** | embedders: [E1, E5, E7, E10, E11] (max) | All views compared |
| **Code-query** | Code query, E1 vs E7 | E7 ranks code higher |
| **Entity-query** | Entity-dense query, E1 vs E11 | E11 ranks entities higher |

#### ST-1.11.4: `list_embedder_indexes`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **With-details** | includeDetails: true | Memory usage, latency |
| **Without-details** | includeDetails: false | Basic stats only |
| **After-stores** | After 1K stores | Vector counts match |
| **Concurrent-read** | 10 parallel reads | Consistent results |

#### ST-1.11.5: `get_memory_fingerprint`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **All-embedders** | No filter | All 13 embedder vectors |
| **Specific-embedders** | embedders: [E1, E7] | Only those 2 |
| **Vector-norms** | includeVectorNorms: true | L2 norms present |
| **Invalid-ID** | Non-existent UUID | Error message |
| **After-boost** | After boost_importance | Fingerprint unchanged (importance is metadata) |

#### ST-1.11.6: `create_weight_profile`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Valid-profile** | name: "my_profile", weights: balanced | Created successfully |
| **Duplicate-name** | Same name twice | Error: already exists |
| **Built-in-name** | name: "semantic_search" | Rejected |
| **Zero-weights** | All weights 0 | Error or normalization |
| **Use-in-search** | Create, then use in search_graph | Custom profile applied |
| **Long-name** | name: 65 chars (over 64 max) | Error |

#### ST-1.11.7: `search_cross_embedder_anomalies`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **E1-vs-E7** | Code in semantic blind spot | Finds code-only memories |
| **E7-vs-E1** | Prose in code blind spot | Finds prose-only memories |
| **Same-embedder** | highEmbedder == lowEmbedder | Error or empty |
| **Strict-thresholds** | highThreshold: 0.9, lowThreshold: 0.1 | Very few anomalies |

---

### 1.12 Temporal Search Tools

#### ST-1.12.1: `search_recent` (E2)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Default-decay** | decayFunction: exponential | Recent memories ranked higher |
| **Linear-decay** | decayFunction: linear | Linear temporal boost |
| **Step-decay** | decayFunction: step | Step-function boost |
| **All-scales** | micro, meso, macro, long | Different temporal granularity |
| **High-temporal** | temporalWeight: 1.0 | Pure recency ranking |
| **Low-temporal** | temporalWeight: 0.1 | Mostly semantic |

#### ST-1.12.2: `search_periodic` (E3)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Target-hour** | targetHour: 14 | Afternoon memories |
| **Target-day** | targetDayOfWeek: 1 (Monday) | Monday memories |
| **Auto-detect** | autoDetect: true | Uses current time |
| **Combined** | targetHour + targetDayOfWeek | Double filter |

---

### 1.13 Graph Linking Tools

#### ST-1.13.1: `get_memory_neighbors` (HNSW K-NN)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **E1-neighbors** | embedder_id: 0 | E1 semantic neighbors |
| **E5-neighbors** | embedder_id: 4 | E5 causal neighbors |
| **E7-neighbors** | embedder_id: 6 | E7 code neighbors |
| **All-embedders** | embedder_id: 0-12 | Each embedder's neighbors |
| **TopK-50** | top_k: 50 | Large neighbor set |
| **Min-similarity** | min_similarity: 0.8 | Strict filtering |
| **Invalid-ID** | Non-existent UUID | Error or empty |
| **Empty-edges** | EdgeRepository not populated | Returns 0 (known) |

#### ST-1.13.2: `get_typed_edges`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **All-types** | Each edge_type enum value | Different edges per type |
| **Outgoing** | direction: outgoing | Only outgoing |
| **Incoming** | direction: incoming | Only incoming |
| **Both** | direction: both | Both directions |
| **Min-weight** | min_weight: 0.5 | Filtered by weight |

#### ST-1.13.3: `traverse_graph`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **1-hop** | max_hops: 1 | Immediate neighbors |
| **5-hops** | max_hops: 5 (max) | Deep traversal |
| **Type-filter** | edge_type: "code_related" | Only code edges |
| **Max-results** | max_results: 100 | Bounded output |
| **Cycle-detection** | Graph with cycles | No infinite loop |

#### ST-1.13.4: `get_unified_neighbors` (Weighted RRF)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **All-profiles** | Each of 14 weight profiles | Different neighbor sets |
| **Custom-weights** | custom_weights: {E1: 0.5, E7: 0.5} | Custom fusion |
| **Exclude-embedders** | exclude_embedders: [E2, E3, E4] | Temporal excluded |
| **Embedder-breakdown** | include_embedder_breakdown: true | Per-embedder scores |
| **Min-score** | min_score: 0.5 | Filtered |

---

### 1.14 Maintenance & Provenance Tools

#### ST-1.14.1: `repair_causal_relationships`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Clean-DB** | No corruption | 0 deleted, total scanned |
| **After-stores** | After many causal discoveries | Counts correct |
| **Concurrent** | Repair while storing | No crash |

#### ST-1.14.2: `get_audit_trail`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **By-target** | target_id: known UUID | Audit records for that memory |
| **By-time-range** | start_time + end_time | Time-bounded results |
| **High-limit** | limit: 500 | Large audit fetch |
| **After-operations** | After store, merge, delete, boost | All ops in trail |

#### ST-1.14.3: `get_merge_history`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Merged-memory** | ID of merged result | Shows source IDs, strategy |
| **Unmerged-memory** | Never-merged memory | Empty history |
| **With-metadata** | include_source_metadata: true | Source details included |
| **Chain-merge** | Result of multi-step merge | Full lineage |

#### ST-1.14.4: `get_provenance_chain`

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Basic** | Known memory ID | Source type, timestamps |
| **With-audit** | include_audit: true | Audit trail included |
| **With-embedding** | include_embedding_version: true | Embedding version info |
| **Full-chain** | Both flags true | Complete provenance |
| **File-sourced** | MDFileChunk source | File path, chunk info |

---

## Phase 2: Concurrency Stress Tests

### 2.1 Read-Read Concurrency

| Test ID | Description | Parallel Ops | Pass Criteria |
|---------|-------------|--------------|---------------|
| **CC-RR-01** | 100 parallel `search_graph` with different queries | 100 | All return, no deadlock |
| **CC-RR-02** | 50 parallel `get_memetic_status` | 50 | Consistent state |
| **CC-RR-03** | 30 parallel `search_by_embedder` across all embedders | 30 | No VRAM contention |
| **CC-RR-04** | 20 parallel `get_topic_portfolio` | 20 | Same results |
| **CC-RR-05** | Mixed reads: search + status + portfolio + fingerprint | 50 | All succeed |

### 2.2 Write-Write Concurrency

| Test ID | Description | Parallel Ops | Pass Criteria |
|---------|-------------|--------------|---------------|
| **CC-WW-01** | 50 parallel `store_memory` | 50 | All unique IDs, no corruption |
| **CC-WW-02** | 20 parallel `boost_importance` on different memories | 20 | Each correct |
| **CC-WW-03** | 10 parallel `merge_concepts` on disjoint pairs | 10 | All succeed |
| **CC-WW-04** | 20 parallel `forget_concept` on different memories | 20 | All deleted |
| **CC-WW-05** | 5 parallel `trigger_consolidation` | 5 | No crash, consistent result |

### 2.3 Read-Write Concurrency

| Test ID | Description | Parallel Ops | Pass Criteria |
|---------|-------------|--------------|---------------|
| **CC-RW-01** | Store + Search interleaved | 50 stores + 50 searches | No stale reads after completion |
| **CC-RW-02** | Delete + Search interleaved | 20 deletes + 20 searches | Deleted memories eventually excluded |
| **CC-RW-03** | Boost + Search interleaved | 20 boosts + 20 searches | No crashes |
| **CC-RW-04** | Merge + Search interleaved | 10 merges + 20 searches | Merged results findable |
| **CC-RW-05** | Store + detect_topics interleaved | 50 stores + 5 detects | Topics eventually updated |

### 2.4 Contention Tests

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| **CC-CT-01** | 20 parallel boosts on SAME memory | Final value mathematically correct |
| **CC-CT-02** | 2 parallel merges sharing 1 source | Exactly 1 succeeds, 1 errors |
| **CC-CT-03** | Store + immediate delete of same ID | Consistent final state |
| **CC-CT-04** | Store + immediate boost of same ID | Both operations applied |
| **CC-CT-05** | 2 parallel causal discoveries | One runs, one queues/errors |

---

## Phase 3: Pipeline Stress Tests

### 3.1 Ingest-Search-Curate Pipeline

```
store_memory ×100 → search_graph → merge_concepts → search_graph (verify)
                   → detect_topics → get_topic_portfolio
                   → trigger_consolidation → get_memetic_status
```

| Test ID | Description | Iterations | Pass Criteria |
|---------|-------------|------------|---------------|
| **PL-ISC-01** | Full pipeline, 100 memories | 1 | Completes, merged memories searchable |
| **PL-ISC-02** | Full pipeline, 1K memories | 1 | Completes < 60s |
| **PL-ISC-03** | Pipeline repeated 10x (10K total) | 10 | Consistent growth, no degradation |

### 3.2 Causal Discovery Pipeline

```
store_memory ×50 (causal content) → trigger_causal_discovery
  → get_causal_discovery_status → search_causal_relationships
  → search_causes → search_effects → get_causal_chain
```

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| **PL-CD-01** | Full causal pipeline | Relationships discovered, chains built |
| **PL-CD-02** | Repeated discovery after new stores | Incremental discovery works |

### 3.3 Entity Knowledge Pipeline

```
store_memory ×50 (entity-rich) → extract_entities
  → search_by_entities → infer_relationship
  → find_related_entities → validate_knowledge → get_entity_graph
```

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| **PL-EK-01** | Full entity pipeline | Entities extracted, graph built |
| **PL-EK-02** | Cross-referencing entities across memories | Relationships inferred |

### 3.4 Multi-Embedder Analysis Pipeline

```
store_memory ×50 → search_by_embedder (E1) → search_by_embedder (E7)
  → compare_embedder_views → search_cross_embedder_anomalies
  → create_weight_profile → search_graph (custom profile)
```

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| **PL-ME-01** | Full embedder analysis | Anomalies found, custom profile works |
| **PL-ME-02** | All 14 weight profiles through search | Each produces results |

### 3.5 Provenance Chain Pipeline

```
store_memory → boost_importance → merge_concepts → forget_concept
  → get_audit_trail → get_merge_history → get_provenance_chain
```

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| **PL-PC-01** | Full provenance pipeline | Complete audit trail of all operations |
| **PL-PC-02** | Complex merge lineage | Multi-generation merge history tracked |

### 3.6 Session Navigation Pipeline

```
store_memory ×20 (with session IDs) → get_session_timeline
  → get_conversation_context → traverse_memory_chain
  → compare_session_states
```

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| **PL-SN-01** | Full session pipeline | Timeline correct, chains navigable |

---

## Phase 4: Resource & Saturation Tests

### 4.1 Memory Pressure

| Test ID | Description | Metric | Pass Criteria |
|---------|-------------|--------|---------------|
| **RS-MP-01** | Store 50K memories, monitor RSS | Process memory | < 8GB RSS |
| **RS-MP-02** | Search after 50K stores | Latency | p99 < 2s |
| **RS-MP-03** | detect_topics on 50K memories | Time + memory | Completes < 60s |
| **RS-MP-04** | get_memetic_status after 50K stores | Latency | < 500ms |

### 4.2 Disk Pressure

| Test ID | Description | Metric | Pass Criteria |
|---------|-------------|--------|---------------|
| **RS-DP-01** | Store 50K memories, check disk | RocksDB size | Reasonable growth |
| **RS-DP-02** | Store + delete cycle (50K store, 25K delete) | Disk after compaction | Reclaimed space |
| **RS-DP-03** | Many small CFs with few entries | Disk overhead | No excessive overhead |

### 4.3 VRAM Pressure

| Test ID | Description | Metric | Pass Criteria |
|---------|-------------|--------|---------------|
| **RS-VP-01** | 100 concurrent embedding requests | VRAM usage | No OOM |
| **RS-VP-02** | Causal discovery + parallel searches | VRAM peak | Within 32GB |
| **RS-VP-03** | All 13 embedders queried in parallel | VRAM | Stable, no leak |

### 4.4 Throughput Saturation

| Test ID | Description | Target | Pass Criteria |
|---------|-------------|--------|---------------|
| **RS-TS-01** | Max store_memory throughput | ops/sec | Measure baseline |
| **RS-TS-02** | Max search_graph throughput | ops/sec | Measure baseline |
| **RS-TS-03** | Max extract_entities throughput | ops/sec | Measure baseline |
| **RS-TS-04** | Mixed workload (70% read, 30% write) | ops/sec | Stable throughput |

---

## Phase 5: Error Handling & Recovery Tests

### 5.1 Invalid Input Tests

| Test ID | Tool | Invalid Input | Pass Criteria |
|---------|------|---------------|---------------|
| **EH-II-01** | store_memory | Missing required `content` | Error message, no crash |
| **EH-II-02** | search_graph | topK: -1 | Error or clamped |
| **EH-II-03** | search_graph | topK: 999999 | Error or clamped |
| **EH-II-04** | merge_concepts | source_ids: [] (empty) | Error |
| **EH-II-05** | merge_concepts | source_ids: 1 ID only | Error (min 2) |
| **EH-II-06** | boost_importance | delta: 5.0 (over max) | Error or clamped |
| **EH-II-07** | search_by_embedder | embedder: "E99" | Error |
| **EH-II-08** | get_memory_fingerprint | memoryId: "not-a-uuid" | Error |
| **EH-II-09** | search_graph | minSimilarity: 2.0 | Error or clamped |
| **EH-II-10** | create_weight_profile | name: "" (empty) | Error |
| **EH-II-11** | traverse_memory_chain | hops: 0 | Error (min 1) |
| **EH-II-12** | discover_graph_relationships | memory_ids: 1 ID | Error (min 2) |
| **EH-II-13** | discover_graph_relationships | memory_ids: 51 IDs | Error (max 50) |
| **EH-II-14** | search_robust | query: "ab" (2 chars, min 3) | Error |
| **EH-II-15** | compare_embedder_views | embedders: 1 only | Error (min 2) |
| **EH-II-16** | compare_embedder_views | embedders: 6 | Error (max 5) |

### 5.2 State Recovery Tests

| Test ID | Scenario | Pass Criteria |
|---------|----------|---------------|
| **EH-SR-01** | Store 100 memories, "crash" (drop DB), reopen | All 100 recoverable |
| **EH-SR-02** | Soft-delete, then verify 30-day recovery metadata | Recovery data present |
| **EH-SR-03** | Merge, then check reversal_hash | Reversal possible |
| **EH-SR-04** | repair_causal_relationships after simulated corruption | Corrupted entries cleaned |

### 5.3 Graceful Degradation

| Test ID | Scenario | Pass Criteria |
|---------|----------|---------------|
| **EH-GD-01** | Search on empty database | Empty results, no crash |
| **EH-GD-02** | detect_topics with 1 memory | Graceful: insufficient data message |
| **EH-GD-03** | get_causal_chain on isolated memory | Returns anchor only |
| **EH-GD-04** | get_unified_neighbors with empty EdgeRepository | Returns 0 results |
| **EH-GD-05** | Provenance query on never-modified memory | Minimal chain |

---

## Phase 6: Performance Regression Benchmarks

### 6.1 Latency Benchmarks (Criterion)

| Benchmark | DB Size | Target p99 |
|-----------|---------|------------|
| `store_memory` | 0-50K | < 50ms |
| `search_graph` (semantic_search profile) | 100 | < 200ms |
| `search_graph` (semantic_search profile) | 1K | < 500ms |
| `search_graph` (semantic_search profile) | 10K | < 1s |
| `extract_entities` | N/A | < 10ms |
| `search_by_keywords` | 1K | < 300ms |
| `search_code` | 1K | < 300ms |
| `search_robust` | 1K | < 300ms |
| `detect_topics` | 100 | < 2s |
| `detect_topics` | 1K | < 10s |
| `merge_concepts` (2-way) | 1K | < 100ms |
| `get_memetic_status` | 50K | < 100ms |
| `get_memory_fingerprint` | N/A | < 50ms |
| `trigger_consolidation` | 100 | < 5s |

### 6.2 Throughput Benchmarks

| Benchmark | Concurrency | Target |
|-----------|-------------|--------|
| `store_memory` sustained | 1 | > 100 ops/s |
| `search_graph` sustained | 10 | > 50 ops/s |
| `extract_entities` sustained | 10 | > 200 ops/s |
| Mixed read/write (70/30) | 20 | > 100 ops/s |

---

## Implementation Strategy

### File Organization

```
crates/context-graph-mcp/tests/
  stress/
    mod.rs                          # Module root, shared utilities
    seed.rs                         # Phase 0: Data seeding infrastructure
    core_memory_stress.rs           # ST-1.1: store, status, search, consolidation
    topic_stress.rs                 # ST-1.2: topic detection tools
    curation_stress.rs              # ST-1.3: merge, forget, boost
    file_watcher_stress.rs          # ST-1.4: file watcher tools
    sequence_nav_stress.rs          # ST-1.5: session/timeline tools
    causal_stress.rs                # ST-1.6 + 1.7: causal search + discovery
    graph_stress.rs                 # ST-1.8: graph tools
    specialized_search_stress.rs    # ST-1.9: keyword, code, robust
    entity_stress.rs                # ST-1.10: entity KG tools
    embedder_stress.rs              # ST-1.11: embedder-first tools
    temporal_stress.rs              # ST-1.12: temporal search tools
    graph_linking_stress.rs         # ST-1.13: K-NN, typed edges, traversal
    provenance_stress.rs            # ST-1.14: maintenance + provenance
    concurrency_stress.rs           # Phase 2: all concurrency tests
    pipeline_stress.rs              # Phase 3: multi-tool pipelines
    saturation_stress.rs            # Phase 4: resource limits
    error_handling_stress.rs        # Phase 5: invalid inputs, recovery
  benches/
    mcp_latency_bench.rs            # Phase 6: criterion latency benchmarks
    mcp_throughput_bench.rs         # Phase 6: criterion throughput benchmarks
```

### Test Execution

```bash
# Run all stress tests (sequential to avoid VRAM contention)
cargo test -p context-graph-mcp --test stress -- --test-threads=1

# Run specific phase
cargo test -p context-graph-mcp --test stress core_memory_stress -- --test-threads=1
cargo test -p context-graph-mcp --test stress concurrency_stress -- --test-threads=1

# Run benchmarks
cargo bench -p context-graph-mcp --bench mcp_latency_bench
cargo bench -p context-graph-mcp --bench mcp_throughput_bench

# Run with verbose output for latency analysis
RUST_LOG=info cargo test -p context-graph-mcp --test stress -- --test-threads=1 --nocapture
```

### Test Pattern Template

```rust
#[tokio::test]
async fn st_1_1_1_burst_100_stores() {
    let (handlers, _tempdir) = create_test_handlers().await;
    let start = Instant::now();

    let mut ids = Vec::with_capacity(100);
    for i in 0..100 {
        let params = json!({
            "name": "store_memory",
            "arguments": {
                "content": format!("Stress test memory #{}: {}", i, generate_content(i)),
                "rationale": "Burst ingestion stress test",
            }
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(i as i64)), Some(params));
        let response = handlers.dispatch(request).await;
        let data = extract_mcp_tool_data(&response.result.unwrap());
        let id = data["id"].as_str().unwrap();
        assert!(!ids.contains(&id.to_string()), "Duplicate ID detected");
        ids.push(id.to_string());
    }

    let elapsed = start.elapsed();
    assert_eq!(ids.len(), 100);
    println!("Burst-100 stores completed in {:?}", elapsed);
}

#[tokio::test]
async fn cc_rr_01_100_parallel_searches() {
    let (handlers, _tempdir) = create_test_handlers().await;
    // Seed 500 memories first
    seed_tier(&handlers, SeedTier::Medium).await;

    let handlers = Arc::new(handlers);
    let mut handles = Vec::new();

    for i in 0..100 {
        let h = Arc::clone(&handlers);
        handles.push(tokio::spawn(async move {
            let params = json!({
                "name": "search_graph",
                "arguments": {
                    "query": format!("stress test query {}", i),
                    "topK": 5,
                }
            });
            let request = make_request("tools/call", Some(JsonRpcId::Number(i)), Some(params));
            let response = h.dispatch(request).await;
            assert!(response.result.is_some());
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }
}
```

### Tagging Convention

Tests are tagged with feature flags for selective execution:

```rust
#[cfg(feature = "stress-small")]   // Runs in CI (< 5 min)
#[cfg(feature = "stress-medium")]  // Runs nightly (< 30 min)
#[cfg(feature = "stress-large")]   // Manual only (< 2 hours)
#[cfg(feature = "stress-xl")]      // Manual only, requires RTX 5090 (< 8 hours)
```

---

## Test Matrix Summary

| Phase | Tests | Estimated Duration | Requires GPU |
|-------|-------|--------------------|--------------|
| Phase 0: Seeding | 4 tiers | 1-60 min per tier | Yes |
| Phase 1: Individual tools | ~180 tests | 30-90 min | Yes |
| Phase 2: Concurrency | ~25 tests | 15-30 min | Yes |
| Phase 3: Pipelines | ~12 tests | 20-40 min | Yes |
| Phase 4: Saturation | ~12 tests | 30-120 min | Yes |
| Phase 5: Error handling | ~25 tests | 10-20 min | Yes |
| Phase 6: Benchmarks | ~20 benchmarks | 20-40 min | Yes |
| **Total** | **~278 tests** | **2-6 hours** | **Yes** |

---

## Success Criteria

1. **Zero crashes** - No panics, segfaults, or deadlocks across all tests
2. **Zero data corruption** - All stored data retrievable and correct after stress
3. **Bounded latency** - p99 latencies within specified targets
4. **Consistent concurrency** - No race conditions in parallel operations
5. **Graceful errors** - All invalid inputs produce informative error messages
6. **Resource stability** - No memory leaks, VRAM leaks, or unbounded disk growth
7. **Provenance integrity** - Audit trail accurately reflects all operations performed
