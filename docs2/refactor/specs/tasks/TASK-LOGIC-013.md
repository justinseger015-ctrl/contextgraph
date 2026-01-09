# TASK-LOGIC-013: Search Result Caching

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-LOGIC-013 |
| **Title** | Search Result Caching |
| **Status** | :white_circle: todo |
| **Layer** | Logic |
| **Sequence** | 23 |
| **Estimated Days** | 1.5 |
| **Complexity** | Medium |

## Implements

- **REQ-LATENCY-01**: End-to-end retrieval < 30ms (caching enables sub-5ms for repeated queries)
- **Performance Budget**: Search throughput > 1000 qps

## Dependencies

| Task | Reason |
|------|--------|
| TASK-LOGIC-008 | 5-Stage Pipeline that produces search results |

## Objective

Implement `SearchCache` with LRU eviction, TTL expiry, and invalidation hooks. Support both exact query matching and semantic query similarity caching.

## Context

Performance budget requires < 30ms for full pipeline. Caching enables:
- Sub-5ms responses for repeated queries
- Reduced GPU/index pressure
- Higher concurrent throughput

Cache challenges:
- Query embeddings must be compared for similarity
- Results must be invalidated when memories change
- TTL prevents stale results

## Scope

### In Scope

- `SearchCache` struct with LRU eviction
- TTL-based expiry for freshness
- Exact query hash matching
- Invalidation on memory mutations
- Cache warming with frequent patterns
- Metrics for hit/miss rates

### Out of Scope

- Distributed cache (single-process only)
- Semantic similarity-based cache lookup
- Predictive cache pre-filling

## Definition of Done

### Signatures

```rust
// crates/context-graph-storage/src/teleological/search/cache.rs

use std::sync::RwLock;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use lru::LruCache;
use uuid::Uuid;

/// Search result cache with LRU eviction and TTL
pub struct SearchCache {
    cache: RwLock<LruCache<u64, CachedResult>>,
    ttl: Duration,
    max_size: usize,
    stats: RwLock<CacheStats>,
}

/// Cached search result
#[derive(Debug, Clone)]
pub struct CachedResult {
    /// The search results
    pub results: Vec<SearchResult>,
    /// When this entry was created
    pub created_at: Instant,
    /// Hash of the original query
    pub query_hash: u64,
    /// Which arrays are referenced (for invalidation)
    pub referenced_ids: Vec<Uuid>,
}

impl CachedResult {
    /// Check if this result is still valid (not expired)
    pub fn is_valid(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() < ttl
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub invalidations: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

impl SearchCache {
    /// Create new cache with specified size and TTL
    pub fn new(max_size: usize, ttl: Duration) -> Self;

    /// Get cached result if exists and not expired
    ///
    /// # Arguments
    /// * `query_hash` - Hash of the query TeleologicalArray
    ///
    /// # Returns
    /// Cached results if hit and valid, None otherwise
    pub fn get(&self, query_hash: u64) -> Option<Vec<SearchResult>>;

    /// Store search result in cache
    ///
    /// # Arguments
    /// * `query_hash` - Hash of the query
    /// * `results` - Search results to cache
    pub fn put(&self, query_hash: u64, results: Vec<SearchResult>);

    /// Store with explicit referenced IDs for invalidation
    pub fn put_with_refs(
        &self,
        query_hash: u64,
        results: Vec<SearchResult>,
        referenced_ids: Vec<Uuid>,
    );

    /// Invalidate all cached results
    pub fn invalidate_all(&self);

    /// Invalidate results containing specific array
    ///
    /// Called when an array is updated or deleted
    pub fn invalidate_containing(&self, array_id: Uuid);

    /// Invalidate multiple arrays (batch mutation)
    pub fn invalidate_batch(&self, array_ids: &[Uuid]);

    /// Warm cache with frequent query patterns
    ///
    /// # Arguments
    /// * `patterns` - Query patterns to pre-compute results for
    /// * `searcher` - Search function to execute queries
    pub async fn warm<F, Fut>(
        &self,
        patterns: &[TeleologicalArray],
        searcher: F,
    ) where
        F: Fn(&TeleologicalArray) -> Fut,
        Fut: std::future::Future<Output = Vec<SearchResult>>;

    /// Get current cache statistics
    pub fn stats(&self) -> CacheStats;

    /// Get current cache size
    pub fn len(&self) -> usize;

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool;

    /// Clear all entries
    pub fn clear(&self);
}

/// Hash a TeleologicalArray for cache key
pub fn hash_query(query: &TeleologicalArray) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();

    // Hash each embedding's bytes
    for embedding in &query.embeddings {
        match embedding {
            EmbedderOutput::Dense(v) => {
                for &f in v {
                    f.to_bits().hash(&mut hasher);
                }
            }
            EmbedderOutput::Sparse(s) => {
                for (&idx, &val) in &s.indices {
                    idx.hash(&mut hasher);
                    val.to_bits().hash(&mut hasher);
                }
            }
            // ... other variants
            _ => {}
        }
    }

    hasher.finish()
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Time-to-live for entries
    pub ttl: Duration,
    /// Enable cache warming on startup
    pub warm_on_start: bool,
    /// Patterns for warming (query templates)
    pub warm_patterns: Vec<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            ttl: Duration::from_secs(300), // 5 minutes
            warm_on_start: false,
            warm_patterns: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit() {
        let cache = SearchCache::new(100, Duration::from_secs(60));
        let results = vec![/* mock results */];

        cache.put(12345, results.clone());

        let cached = cache.get(12345);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), results.len());
    }

    #[test]
    fn test_cache_expiry() {
        let cache = SearchCache::new(100, Duration::from_millis(50));
        cache.put(12345, vec![]);

        std::thread::sleep(Duration::from_millis(100));

        assert!(cache.get(12345).is_none());
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = SearchCache::new(100, Duration::from_secs(60));
        let array_id = Uuid::new_v4();

        cache.put_with_refs(12345, vec![], vec![array_id]);

        cache.invalidate_containing(array_id);

        assert!(cache.get(12345).is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let cache = SearchCache::new(2, Duration::from_secs(60));

        cache.put(1, vec![]);
        cache.put(2, vec![]);
        cache.put(3, vec![]); // Should evict 1

        assert!(cache.get(1).is_none());
        assert!(cache.get(2).is_some());
        assert!(cache.get(3).is_some());
    }
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Cache lookup latency | < 0.1ms |
| Memory per entry | ~1KB average |
| Default TTL | 5 minutes |
| Default max entries | 10,000 |

## Verification

- [ ] Cache hit returns results without re-searching
- [ ] TTL expiry removes stale entries
- [ ] LRU eviction keeps frequently used entries
- [ ] Invalidation removes affected entries
- [ ] Cache stats accurately track hit/miss
- [ ] Warm-up pre-populates cache

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-storage/src/teleological/search/cache.rs` | Cache implementation |
| Update `crates/context-graph-storage/src/teleological/search/mod.rs` | Export cache |

## Integration Points

| Component | Integration |
|-----------|-------------|
| `TASK-LOGIC-008` | Check cache before pipeline execution |
| `TASK-INTEG-001` | Invalidate on store_memory |
| `TASK-INTEG-004` | Hook invalidation on PostToolUse |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Stale results | Medium | Medium | TTL prevents long staleness |
| Memory bloat | Low | Medium | LRU eviction, size limit |
| Invalidation overhead | Low | Low | Batch invalidation |

## Traceability

- Source: Constitution performance_budgets (lines 518-599)
- Reference: TASK-LOGIC-007 mentions query_cache stub
