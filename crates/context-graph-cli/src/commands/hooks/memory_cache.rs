//! Memory cache for pre_tool_use context injection.
//!
//! # Purpose
//! pre_tool_use has a 500ms total timeout (per constitution.yaml) with ~100ms CLI logic target.
//! This budget is too tight for MCP network calls. This cache stores memories retrieved during
//! user_prompt_submit (2s budget) so they can be accessed instantly by pre_tool_use.
//!
//! # Architecture
//! - user_prompt_submit retrieves memories from MCP and caches them
//! - pre_tool_use reads from cache (no network calls)
//! - Cache is keyed by session_id for isolation
//!
//! # Constitution References
//! - AP-50: NO internal hooks - shell scripts call CLI
//! - hooks.timeout_ms.pre_tool_use: 500ms total (FAST PATH, CLI logic ~100ms)

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

/// Maximum number of cached sessions before eviction.
const MAX_CACHED_SESSIONS: usize = 10;

/// Cache TTL - memories expire after 5 minutes.
const CACHE_TTL: Duration = Duration::from_secs(300);

/// Maximum memories to cache per session.
const MAX_MEMORIES_PER_SESSION: usize = 10;

/// Global singleton for memory cache.
static MEMORY_CACHE: OnceLock<RwLock<MemoryCache>> = OnceLock::new();

/// A cached memory with content and similarity score.
#[derive(Debug, Clone)]
pub struct CachedMemory {
    /// Memory content text.
    pub content: String,
    /// Similarity score to the query.
    pub similarity: f32,
}

/// Entry in the cache with expiration.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Cached memories.
    memories: Vec<CachedMemory>,
    /// When the entry was created.
    created_at: Instant,
}

impl CacheEntry {
    fn new(memories: Vec<CachedMemory>) -> Self {
        Self {
            memories,
            created_at: Instant::now(),
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > CACHE_TTL
    }
}

/// In-memory cache for retrieved memories.
#[derive(Debug, Default)]
struct MemoryCache {
    /// Cached entries by session ID.
    entries: HashMap<String, CacheEntry>,
}

impl MemoryCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Store memories for a session.
    fn store(&mut self, session_id: String, memories: Vec<CachedMemory>) {
        // Evict oldest entries if at capacity
        if self.entries.len() >= MAX_CACHED_SESSIONS {
            self.evict_oldest();
        }

        // Limit memories per session
        let memories = if memories.len() > MAX_MEMORIES_PER_SESSION {
            memories.into_iter().take(MAX_MEMORIES_PER_SESSION).collect()
        } else {
            memories
        };

        self.entries.insert(session_id, CacheEntry::new(memories));
    }

    /// Get memories for a session if not expired.
    fn get(&self, session_id: &str) -> Option<&CacheEntry> {
        self.entries.get(session_id).filter(|e| !e.is_expired())
    }

    /// Evict the oldest entry.
    fn evict_oldest(&mut self) {
        if let Some((oldest_id, _)) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.created_at)
        {
            let oldest_id = oldest_id.clone();
            self.entries.remove(&oldest_id);
        }
    }

    /// Clear expired entries.
    #[allow(dead_code)]
    fn clear_expired(&mut self) {
        self.entries.retain(|_, entry| !entry.is_expired());
    }
}

/// Get the global memory cache.
fn global_cache() -> &'static RwLock<MemoryCache> {
    MEMORY_CACHE.get_or_init(|| RwLock::new(MemoryCache::new()))
}

// =============================================================================
// Public API
// =============================================================================

/// Store memories in the cache for a session.
///
/// Called by user_prompt_submit after retrieving memories from MCP.
///
/// # Arguments
/// * `session_id` - Session identifier
/// * `memories` - Retrieved memories to cache
pub fn cache_memories(session_id: &str, memories: Vec<CachedMemory>) {
    let memory_count = memories.len();
    if let Ok(mut cache) = global_cache().write() {
        cache.store(session_id.to_string(), memories);
        tracing::debug!(
            session_id,
            memory_count,
            "MEMORY_CACHE: Stored memories"
        );
    }
}

/// Get cached memories for a session.
///
/// Called by pre_tool_use to get memories without MCP calls.
///
/// # Arguments
/// * `session_id` - Session identifier
///
/// # Returns
/// * Cached memories if available and not expired, empty vec otherwise
pub fn get_cached_memories(session_id: &str) -> Vec<CachedMemory> {
    if let Ok(cache) = global_cache().read() {
        if let Some(entry) = cache.get(session_id) {
            tracing::debug!(
                session_id,
                memory_count = entry.memories.len(),
                "MEMORY_CACHE: Retrieved cached memories"
            );
            return entry.memories.clone();
        }
    }
    Vec::new()
}


// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_and_retrieve_memories() {
        let session_id = "test-session-001";
        let memories = vec![
            CachedMemory {
                content: "Test memory content".to_string(),
                similarity: 0.85,
            },
            CachedMemory {
                content: "Another memory".to_string(),
                similarity: 0.72,
            },
        ];

        cache_memories(session_id, memories.clone());

        let retrieved = get_cached_memories(session_id);
        assert_eq!(retrieved.len(), 2);
        assert_eq!(retrieved[0].content, "Test memory content");
        assert_eq!(retrieved[0].similarity, 0.85);
    }

    #[test]
    fn test_empty_cache_returns_empty() {
        let retrieved = get_cached_memories("nonexistent-session");
        assert!(retrieved.is_empty());
    }
}
