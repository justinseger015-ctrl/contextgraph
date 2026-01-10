# TASK-LOGIC-013: Search Result Caching

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-LOGIC-013 |
| **Title** | Search Result Caching |
| **Status** | :white_circle: todo |
| **Layer** | Logic |
| **Sequence** | 27 |
| **Complexity** | Medium |

## Implements

- **REQ-LATENCY-01**: End-to-end retrieval < 30ms (caching enables sub-5ms for repeated queries)
- **Performance Budget**: Search throughput > 1000 qps (constitution.yaml perf.throughput)

## Dependencies

| Task | Reason | Status |
|------|--------|--------|
| TASK-LOGIC-008 | 5-Stage Pipeline that produces PipelineResult | :white_check_mark: COMPLETE (56ae42e) |

## Objective

Implement `SearchCache` with LRU eviction, TTL expiry, and invalidation hooks. Cache `PipelineResult` keyed by query hash. Integrate with existing 5-stage pipeline.

---

## Current State Audit (2026-01-09)

### Existing Infrastructure

1. **Pipeline Implementation**: `crates/context-graph-storage/src/teleological/search/pipeline.rs`
   - `RetrievalPipeline` struct exists with `execute()` method
   - `PipelineResult` contains `Vec<PipelineCandidate>`, latency info, stage results
   - `PipelineCandidate` has `id: Uuid`, `score: f32`, `stage_scores: Vec<(PipelineStage, f32)>`

2. **Existing Embedding Cache**: `crates/context-graph-embeddings/src/config/cache.rs`
   - Has `CacheConfig` with `EvictionPolicy` enum (Lru, Lfu, TtlLru, Arc)
   - **NOT for search results** - this caches computed embeddings, not search results
   - Can reference for configuration patterns

3. **Search Module**: `crates/context-graph-storage/src/teleological/search/mod.rs`
   - Exports: `RetrievalPipeline`, `PipelineBuilder`, `PipelineConfig`, `PipelineResult`, etc.
   - **No cache module exists yet** - must create `cache.rs`

4. **Dependencies in Cargo.toml** (`crates/context-graph-storage/Cargo.toml`):
   - `uuid` (with serde)
   - `chrono` (with serde)
   - `tokio` (sync, rt-multi-thread)
   - `rayon` (parallel iteration)
   - **Missing**: `lru` or `moka` crate for LRU implementation

### Files to Modify

| File | Action |
|------|--------|
| `crates/context-graph-storage/Cargo.toml` | Add `lru = "0.12"` dependency |
| `crates/context-graph-storage/src/teleological/search/cache.rs` | **CREATE** - Main cache implementation |
| `crates/context-graph-storage/src/teleological/search/mod.rs` | Add `mod cache;` and re-exports |

---

## Context

### Why Caching?

The 5-stage pipeline targets <60ms latency. For repeated queries:
- **Without cache**: Full pipeline execution (~30-60ms)
- **With cache**: Direct lookup (<0.1ms) = 300-600x faster

### Cache Challenges

1. **Query Hashing**: Pipeline queries have multiple components (SPLADE, Matryoshka, semantic, tokens, purpose). Need deterministic hash.
2. **Invalidation**: When memories are added/updated/deleted, cached results become stale.
3. **TTL**: Even without mutations, old results may not reflect current relevance.
4. **Memory**: Each cached result ~1-2KB. With 10K entries = ~10-20MB.

### Design Decision: Use `lru` Crate

Based on research:
- [`moka`](https://github.com/moka-rs/moka): Feature-rich but heavyweight for this use case
- [`lru`](https://crates.io/crates/lru): Simple, proven, 2.6M downloads, perfect for our needs
- Wrap in `RwLock<LruCache>` for thread safety (already have tokio sync)

---

## Scope

### In Scope

- `SearchCache` struct with LRU eviction
- TTL-based expiry for freshness
- Query hash computation from `PipelineBuilder` inputs
- Invalidation by memory ID
- Batch invalidation
- Cache warming API
- Hit/miss/eviction statistics

### Out of Scope

- Distributed cache (single-process only per constitution)
- Semantic similarity-based cache lookup (too complex for v1)
- Predictive cache pre-filling (future enhancement)

---

## Definition of Done

### Implementation Signatures

```rust
// crates/context-graph-storage/src/teleological/search/cache.rs

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::RwLock;
use std::time::{Duration, Instant};
use lru::LruCache;
use uuid::Uuid;

use super::{PipelineCandidate, PipelineResult, PipelineStage, StageResult};

// ============================================================================
// CACHE CONFIGURATION
// ============================================================================

/// Cache configuration.
#[derive(Debug, Clone)]
pub struct SearchCacheConfig {
    /// Maximum number of cached results.
    /// Default: 10,000 (constitution.yaml target)
    pub max_entries: NonZeroUsize,

    /// Time-to-live for cached entries.
    /// Default: 5 minutes (300 seconds)
    pub ttl: Duration,

    /// Enable cache (for testing, can disable)
    pub enabled: bool,
}

impl Default for SearchCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: NonZeroUsize::new(10_000).unwrap(),
            ttl: Duration::from_secs(300),
            enabled: true,
        }
    }
}

// ============================================================================
// CACHED RESULT
// ============================================================================

/// A cached pipeline result.
#[derive(Debug, Clone)]
pub struct CachedResult {
    /// The cached pipeline result (cloned, not referenced).
    pub result: CachedPipelineResult,

    /// When this entry was created.
    pub created_at: Instant,

    /// Memory IDs referenced in results (for invalidation).
    pub referenced_ids: Vec<Uuid>,
}

/// Cached version of PipelineResult (implements Clone).
#[derive(Debug, Clone)]
pub struct CachedPipelineResult {
    /// Final ranked results.
    pub results: Vec<PipelineCandidate>,

    /// Total pipeline latency in microseconds (original).
    pub total_latency_us: u64,

    /// Stages that were executed.
    pub stages_executed: Vec<PipelineStage>,

    /// Whether purpose alignment was verified.
    pub alignment_verified: bool,
}

impl CachedResult {
    /// Check if this result has expired.
    #[inline]
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }
}

// ============================================================================
// CACHE STATISTICS
// ============================================================================

/// Cache performance statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache hits.
    pub hits: u64,

    /// Total cache misses.
    pub misses: u64,

    /// Total evictions (LRU).
    pub evictions: u64,

    /// Total invalidations (explicit).
    pub invalidations: u64,

    /// Total expired entries removed.
    pub expirations: u64,
}

impl CacheStats {
    /// Calculate hit rate [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Total accesses.
    pub fn total_accesses(&self) -> u64 {
        self.hits + self.misses
    }
}

// ============================================================================
// SEARCH CACHE
// ============================================================================

/// Thread-safe LRU cache for pipeline search results.
///
/// # Thread Safety
///
/// Uses `RwLock` for concurrent read access with exclusive writes.
/// Multiple readers can check cache simultaneously.
///
/// # Invalidation
///
/// Supports both:
/// - TTL-based expiration (checked on access)
/// - Explicit invalidation by memory ID
pub struct SearchCache {
    /// LRU cache: query_hash -> CachedResult
    cache: RwLock<LruCache<u64, CachedResult>>,

    /// Reverse index: memory_id -> [query_hashes that reference it]
    /// For efficient invalidation when a memory is mutated.
    reverse_index: RwLock<HashMap<Uuid, Vec<u64>>>,

    /// Configuration.
    config: SearchCacheConfig,

    /// Statistics.
    stats: RwLock<CacheStats>,
}

impl SearchCache {
    /// Create new cache with configuration.
    pub fn new(config: SearchCacheConfig) -> Self;

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SearchCacheConfig::default())
    }

    /// Get cached result if exists and not expired.
    ///
    /// # Arguments
    /// * `query_hash` - Hash of the query parameters
    ///
    /// # Returns
    /// - `Some(CachedPipelineResult)` if cache hit and not expired
    /// - `None` if miss or expired
    ///
    /// # Thread Safety
    /// Takes read lock, upgrades to write if expired entry needs removal.
    pub fn get(&self, query_hash: u64) -> Option<CachedPipelineResult>;

    /// Store result in cache.
    ///
    /// # Arguments
    /// * `query_hash` - Hash of the query
    /// * `result` - Pipeline result to cache
    ///
    /// # Thread Safety
    /// Takes write lock.
    pub fn put(&self, query_hash: u64, result: &PipelineResult);

    /// Store with explicit referenced IDs (for fine-grained invalidation).
    pub fn put_with_refs(
        &self,
        query_hash: u64,
        result: &PipelineResult,
        referenced_ids: Vec<Uuid>,
    );

    /// Invalidate all cached results.
    ///
    /// Use when: Major data changes, index rebuild, etc.
    pub fn invalidate_all(&self);

    /// Invalidate results containing a specific memory.
    ///
    /// Use when: A memory is updated or deleted.
    pub fn invalidate_containing(&self, memory_id: Uuid);

    /// Batch invalidation for multiple memories.
    ///
    /// More efficient than calling invalidate_containing N times.
    pub fn invalidate_batch(&self, memory_ids: &[Uuid]);

    /// Get current statistics.
    pub fn stats(&self) -> CacheStats;

    /// Get current cache size.
    pub fn len(&self) -> usize;

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool;

    /// Clear all entries.
    pub fn clear(&self);

    /// Prune expired entries (call periodically or on idle).
    ///
    /// Returns number of entries pruned.
    pub fn prune_expired(&self) -> usize;
}

// ============================================================================
// QUERY HASHING
// ============================================================================

/// Compute deterministic hash from pipeline query components.
///
/// # Hash Components
/// - SPLADE query vector (if present)
/// - Matryoshka 128D vector (if present)
/// - Semantic 1024D vector (if present)
/// - Token embeddings (if present)
/// - Purpose vector (if present)
/// - k limit
///
/// # Determinism
/// Same inputs MUST produce same hash across runs.
/// Uses `std::hash::DefaultHasher` (not random).
pub fn hash_pipeline_query(
    splade: Option<&[(usize, f32)]>,
    matryoshka: Option<&[f32]>,
    semantic: Option<&[f32]>,
    tokens: Option<&[Vec<f32>]>,
    purpose: Option<&[f32; 13]>,
    k: usize,
) -> u64;

/// Hash a dense vector (converts f32 to bits for determinism).
fn hash_dense(hasher: &mut impl std::hash::Hasher, v: &[f32]);

/// Hash a sparse vector.
fn hash_sparse(hasher: &mut impl std::hash::Hasher, v: &[(usize, f32)]);
```

### Constraints

| Constraint | Target | Rationale |
|------------|--------|-----------|
| Cache lookup latency | < 0.1ms | Must be << pipeline latency (30-60ms) |
| Memory per entry | ~1-2KB | 10K entries = ~10-20MB RAM |
| Default TTL | 5 minutes | Balance freshness vs. hit rate |
| Default max entries | 10,000 | Constitution target |
| Lock contention | < 1% overhead | RwLock allows concurrent reads |

---

## Verification Checklist

### Unit Tests Required

- [ ] Cache hit returns cloned result without re-searching
- [ ] Cache miss returns None and increments miss counter
- [ ] TTL expiry removes stale entries on access
- [ ] TTL expiry detected during `prune_expired()`
- [ ] LRU eviction keeps most recently used entries
- [ ] `invalidate_containing()` removes correct entries via reverse index
- [ ] `invalidate_batch()` handles multiple IDs efficiently
- [ ] `invalidate_all()` clears everything
- [ ] Cache stats accurately track hit/miss/eviction/invalidation
- [ ] Query hash is deterministic (same inputs = same hash)
- [ ] Query hash differs for different inputs
- [ ] Thread safety: concurrent reads don't block
- [ ] Thread safety: write blocks readers appropriately

### Integration Tests Required

- [ ] Cache integrates with `RetrievalPipeline`
- [ ] First query executes pipeline, second query returns cached
- [ ] Cache invalidation after memory mutation

---

## Full State Verification Protocol

### Source of Truth

The cache state is stored in:
1. **Primary**: `SearchCache.cache` (LruCache in memory)
2. **Reverse Index**: `SearchCache.reverse_index` (HashMap)
3. **Statistics**: `SearchCache.stats` (counters)

### Execute & Inspect Pattern

After every cache operation, verify by reading the source of truth:

```rust
// Example verification pattern
let cache = SearchCache::with_defaults();

// BEFORE state
println!("BEFORE: cache.len()={}, stats={:?}", cache.len(), cache.stats());

// Execute operation
cache.put(query_hash, &pipeline_result);

// AFTER state - VERIFY
println!("AFTER: cache.len()={}, stats={:?}", cache.len(), cache.stats());
assert_eq!(cache.len(), 1, "Cache should have exactly 1 entry");
assert!(cache.get(query_hash).is_some(), "Entry should be retrievable");
```

### Boundary & Edge Case Audit

You MUST manually simulate these 3 edge cases with explicit before/after state logging:

#### Edge Case 1: Empty Cache Access
```rust
// Input: Query hash for non-existent entry
let cache = SearchCache::with_defaults();
println!("BEFORE: len={}, hits={}, misses={}", cache.len(), cache.stats().hits, cache.stats().misses);

let result = cache.get(12345);

println!("AFTER: len={}, hits={}, misses={}", cache.len(), cache.stats().hits, cache.stats().misses);
assert!(result.is_none());
assert_eq!(cache.stats().misses, 1);
```

#### Edge Case 2: Cache at Capacity (LRU Eviction)
```rust
// Input: Insert max_entries + 1
let config = SearchCacheConfig {
    max_entries: NonZeroUsize::new(2).unwrap(),
    ..Default::default()
};
let cache = SearchCache::new(config);

cache.put(1, &result1);
cache.put(2, &result2);
println!("BEFORE EVICTION: len={}, evictions={}", cache.len(), cache.stats().evictions);

cache.put(3, &result3); // Should evict hash=1 (LRU)

println!("AFTER EVICTION: len={}, evictions={}", cache.len(), cache.stats().evictions);
assert_eq!(cache.len(), 2);
assert_eq!(cache.stats().evictions, 1);
assert!(cache.get(1).is_none(), "Entry 1 should be evicted");
assert!(cache.get(2).is_some(), "Entry 2 should exist");
assert!(cache.get(3).is_some(), "Entry 3 should exist");
```

#### Edge Case 3: TTL Expiration
```rust
// Input: Entry older than TTL
let config = SearchCacheConfig {
    ttl: Duration::from_millis(50),
    ..Default::default()
};
let cache = SearchCache::new(config);

cache.put(999, &result);
println!("BEFORE EXPIRY: len={}, get(999)={:?}", cache.len(), cache.get(999).is_some());

std::thread::sleep(Duration::from_millis(100));

println!("AFTER EXPIRY: get(999)={:?}, expirations={}",
    cache.get(999).is_some(), cache.stats().expirations);
assert!(cache.get(999).is_none(), "Entry should be expired");
```

### Evidence of Success

After implementation, provide a log showing:
1. Actual cache contents after operations
2. Statistics counter values
3. Timing measurements for lookup latency

---

## Manual Testing Procedures

### Synthetic Data for Testing

Use these known inputs with expected outputs:

#### Test Query 1: Semantic-Only Query
```rust
let query_hash = hash_pipeline_query(
    None,                           // no splade
    Some(&[0.5f32; 128]),          // matryoshka
    Some(&[0.1f32; 1024]),         // semantic
    None,                           // no tokens
    None,                           // no purpose
    10,                             // k
);
// Expected: Deterministic u64 hash
println!("Query 1 hash: {}", query_hash);
```

#### Test Query 2: Full Query
```rust
let query_hash = hash_pipeline_query(
    Some(&[(100, 0.5), (200, 0.3)]), // splade
    Some(&[0.5f32; 128]),            // matryoshka
    Some(&[0.1f32; 1024]),           // semantic
    Some(&vec![vec![0.5f32; 128]; 5]), // 5 tokens
    Some(&[0.5f32; 13]),             // purpose
    10,
);
println!("Query 2 hash: {}", query_hash);
```

### Happy Path Testing

1. **Create cache** → verify len=0, stats all zero
2. **Put 5 entries** → verify len=5, stats.puts=5
3. **Get existing entry** → verify returns Some, stats.hits=1
4. **Get same entry again** → verify stats.hits=2
5. **Get non-existent** → verify returns None, stats.misses=1
6. **Invalidate one ID** → verify affected entries removed
7. **Check hit rate** → verify calculation correct

### Latency Verification

```rust
// Measure cache lookup vs. pipeline execution
let pipeline = RetrievalPipeline::new(...);
let cache = SearchCache::with_defaults();

// First call: pipeline execution
let start = Instant::now();
let result = pipeline_builder.execute(&pipeline)?;
let pipeline_latency = start.elapsed();

// Cache the result
let query_hash = compute_query_hash(&pipeline_builder);
cache.put(query_hash, &result);

// Second call: cache lookup
let start = Instant::now();
let cached = cache.get(query_hash);
let cache_latency = start.elapsed();

println!("Pipeline latency: {:?}", pipeline_latency);
println!("Cache latency: {:?}", cache_latency);
assert!(cache_latency < Duration::from_micros(100), "Cache lookup must be < 100us");
```

---

## Integration Points

| Component | Integration |
|-----------|-------------|
| `RetrievalPipeline` | Check cache before `execute()`, cache result after |
| `TASK-INTEG-001` (store_memory) | Call `cache.invalidate_containing(new_id)` |
| `TASK-INTEG-004` (PostToolUse hook) | Invalidate on Edit/Write tools |
| `TASK-INTEG-013` (consolidate_memories) | Batch invalidation on consolidation |

### Integration Pattern

```rust
// In RetrievalPipeline or a wrapper
impl CachedPipeline {
    pub fn execute_cached(
        &self,
        builder: &PipelineBuilder,
        cache: &SearchCache,
    ) -> Result<PipelineResult, PipelineError> {
        let query_hash = hash_pipeline_query(/* from builder */);

        // Check cache first
        if let Some(cached) = cache.get(query_hash) {
            return Ok(cached.into_pipeline_result());
        }

        // Execute pipeline
        let result = builder.execute(&self.pipeline)?;

        // Cache result with referenced IDs
        let ids: Vec<Uuid> = result.results.iter().map(|c| c.id).collect();
        cache.put_with_refs(query_hash, &result, ids);

        Ok(result)
    }
}
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Stale results | Medium | Medium | TTL default 5min; explicit invalidation hooks |
| Memory bloat | Low | Medium | LRU eviction; 10K entry limit |
| Hash collisions | Very Low | High | Use full 64-bit hash; probability ~1/2^64 |
| Lock contention | Low | Medium | RwLock allows concurrent reads |
| Invalidation overhead | Low | Low | Reverse index for O(1) lookup |

---

## Implementation Steps

1. **Add lru dependency** to `crates/context-graph-storage/Cargo.toml`
2. **Create cache.rs** with all types and implementations
3. **Add mod and exports** to `search/mod.rs`
4. **Implement core functionality**:
   - `SearchCache::new()`, `get()`, `put()`
   - `hash_pipeline_query()` function
   - Statistics tracking
5. **Implement invalidation**:
   - Reverse index maintenance
   - `invalidate_containing()`, `invalidate_batch()`, `invalidate_all()`
6. **Add tests** (unit + integration)
7. **Run full state verification** with synthetic data
8. **Measure latency** to confirm < 0.1ms lookup

---

## Traceability

- Source: Constitution performance_budgets (lines 236-240)
- Source: Constitution perf.latency.inject_context < 25ms
- Reference: TASK-LOGIC-008 (pipeline implementation)
- Reference: [moka-rs](https://github.com/moka-rs/moka) for cache patterns
- Reference: [lru crate](https://crates.io/crates/lru) for implementation

---

## NO BACKWARDS COMPATIBILITY

This implementation:
- **MUST** work correctly or fail fast with clear errors
- **MUST NOT** use mock data in tests - use real cache operations
- **MUST NOT** create workarounds or fallbacks
- **MUST** have robust error logging for debugging

If something doesn't work:
1. Error message must include: what failed, where, with what inputs
2. No silent failures
3. No graceful degradation - if cache is broken, surface it immediately
