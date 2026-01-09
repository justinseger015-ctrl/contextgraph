# TASK-PERF-004: Memory Optimization

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-PERF-004 |
| **Title** | Memory Optimization |
| **Status** | :white_circle: todo |
| **Layer** | Performance |
| **Sequence** | 54 |
| **Estimated Days** | 2 |
| **Complexity** | Medium |

## Implements

- Constitution memory constraints (46KB per array)
- REQ-MEMORY-01: Efficient embedding storage
- Large-scale deployment (10M+ memories)

## Dependencies

| Task | Reason |
|------|--------|
| TASK-CORE-013 | Quantization for compression |
| TASK-CORE-003 | Store to optimize |
| TASK-PERF-002 | Memory profiling |

## Objective

Optimize memory usage for large-scale deployments:
1. Embedding compression (quantization)
2. Memory-mapped storage
3. Lazy loading strategies
4. Cache optimization
5. Memory pooling

## Context

At scale (10M memories × 46KB each = 460GB), memory optimization is critical. Target:
- 60-70% memory reduction via quantization
- Minimal latency impact (< 5%)
- Support for memory-constrained environments

## Scope

### In Scope

- Embedding quantization (PQ, scalar, binary)
- Memory-mapped index access
- LRU cache for hot embeddings
- Memory pool for allocations
- Streaming for large operations

### Out of Scope

- Distributed storage
- Disk compression
- External cache (Redis, etc.)

## Definition of Done

### Memory-Mapped Storage

```rust
// crates/context-graph-storage/src/mmap/mod.rs

use memmap2::{MmapMut, MmapOptions};
use std::fs::File;
use std::path::Path;

/// Memory-mapped embedding storage
pub struct MmapEmbeddingStore {
    file: File,
    mmap: MmapMut,
    header: MmapHeader,
    dimension: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MmapHeader {
    magic: [u8; 4],        // "CGEM"
    version: u32,
    count: u64,
    dimension: u32,
    element_size: u32,
    flags: u32,
}

impl MmapEmbeddingStore {
    /// Create new memory-mapped store
    pub fn create(path: &Path, dimension: usize, capacity: usize) -> StorageResult<Self> {
        let element_size = dimension * std::mem::size_of::<f32>();
        let file_size = std::mem::size_of::<MmapHeader>() + (capacity * element_size);

        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        file.set_len(file_size as u64)?;

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        let header = MmapHeader {
            magic: *b"CGEM",
            version: 1,
            count: 0,
            dimension: dimension as u32,
            element_size: element_size as u32,
            flags: 0,
        };

        let mut store = Self {
            file,
            mmap,
            header,
            dimension,
        };

        store.write_header()?;
        Ok(store)
    }

    /// Open existing memory-mapped store
    pub fn open(path: &Path) -> StorageResult<Self> {
        let file = File::options()
            .read(true)
            .write(true)
            .open(path)?;

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        let header: MmapHeader = unsafe {
            std::ptr::read(mmap.as_ptr() as *const MmapHeader)
        };

        if &header.magic != b"CGEM" {
            return Err(StorageError::InvalidFormat);
        }

        Ok(Self {
            file,
            mmap,
            header,
            dimension: header.dimension as usize,
        })
    }

    /// Get embedding by index (zero-copy)
    pub fn get(&self, index: usize) -> Option<&[f32]> {
        if index >= self.header.count as usize {
            return None;
        }

        let offset = std::mem::size_of::<MmapHeader>()
            + index * self.header.element_size as usize;

        let ptr = unsafe {
            self.mmap.as_ptr().add(offset) as *const f32
        };

        Some(unsafe {
            std::slice::from_raw_parts(ptr, self.dimension)
        })
    }

    /// Append embedding (returns index)
    pub fn append(&mut self, embedding: &[f32]) -> StorageResult<usize> {
        if embedding.len() != self.dimension {
            return Err(StorageError::DimensionMismatch);
        }

        let index = self.header.count as usize;
        let offset = std::mem::size_of::<MmapHeader>()
            + index * self.header.element_size as usize;

        // Check capacity
        if offset + self.header.element_size as usize > self.mmap.len() {
            self.grow()?;
        }

        // Copy embedding
        unsafe {
            let dst = self.mmap.as_mut_ptr().add(offset) as *mut f32;
            std::ptr::copy_nonoverlapping(
                embedding.as_ptr(),
                dst,
                self.dimension,
            );
        }

        self.header.count += 1;
        self.write_header()?;

        Ok(index)
    }

    /// Grow file by 2x
    fn grow(&mut self) -> StorageResult<()> {
        let new_size = self.mmap.len() * 2;
        self.file.set_len(new_size as u64)?;
        self.mmap = unsafe { MmapOptions::new().map_mut(&self.file)? };
        Ok(())
    }

    fn write_header(&mut self) -> StorageResult<()> {
        unsafe {
            std::ptr::write(
                self.mmap.as_mut_ptr() as *mut MmapHeader,
                self.header,
            );
        }
        self.mmap.flush()?;
        Ok(())
    }

    /// Sync to disk
    pub fn sync(&self) -> StorageResult<()> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Count of stored embeddings
    pub fn len(&self) -> usize {
        self.header.count as usize
    }
}
```

### LRU Cache for Hot Embeddings

```rust
// crates/context-graph-storage/src/cache/embedding_cache.rs

use lru::LruCache;
use std::sync::RwLock;
use uuid::Uuid;

/// LRU cache for frequently accessed embeddings
pub struct EmbeddingCache {
    cache: RwLock<LruCache<Uuid, CachedEmbedding>>,
    max_memory: usize,
    current_memory: std::sync::atomic::AtomicUsize,
}

#[derive(Clone)]
struct CachedEmbedding {
    data: Vec<f32>,
    access_count: u32,
    memory_size: usize,
}

impl EmbeddingCache {
    /// Create cache with memory limit
    pub fn new(max_memory: usize) -> Self {
        // Estimate max entries based on average embedding size
        let avg_embedding_size = 1024 * 4; // 1024-dim float
        let max_entries = max_memory / avg_embedding_size;

        Self {
            cache: RwLock::new(LruCache::new(
                std::num::NonZeroUsize::new(max_entries).unwrap()
            )),
            max_memory,
            current_memory: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Get embedding from cache
    pub fn get(&self, id: Uuid) -> Option<Vec<f32>> {
        let mut cache = self.cache.write().ok()?;
        cache.get(&id).map(|e| {
            // Update access count
            e.access_count;
            e.data.clone()
        })
    }

    /// Put embedding in cache
    pub fn put(&self, id: Uuid, embedding: Vec<f32>) {
        let memory_size = embedding.len() * std::mem::size_of::<f32>();

        // Evict if necessary
        while self.current_memory.load(std::sync::atomic::Ordering::Relaxed)
            + memory_size > self.max_memory
        {
            self.evict_one();
        }

        let cached = CachedEmbedding {
            data: embedding,
            access_count: 1,
            memory_size,
        };

        if let Ok(mut cache) = self.cache.write() {
            if let Some(old) = cache.put(id, cached) {
                self.current_memory.fetch_sub(
                    old.memory_size,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            self.current_memory.fetch_add(
                memory_size,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }

    /// Evict least recently used entry
    fn evict_one(&self) {
        if let Ok(mut cache) = self.cache.write() {
            if let Some((_, evicted)) = cache.pop_lru() {
                self.current_memory.fetch_sub(
                    evicted.memory_size,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        }
    }

    /// Cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read().unwrap();
        CacheStats {
            entries: cache.len(),
            memory_used: self.current_memory.load(std::sync::atomic::Ordering::Relaxed),
            max_memory: self.max_memory,
            hit_rate: 0.0, // Would need tracking
        }
    }

    /// Clear cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        self.current_memory.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub memory_used: usize,
    pub max_memory: usize,
    pub hit_rate: f64,
}
```

### Memory Pool

```rust
// crates/context-graph-core/src/memory/pool.rs

use std::alloc::{alloc, dealloc, Layout};
use std::sync::Mutex;

/// Fixed-size memory pool for embedding allocations
pub struct EmbeddingPool {
    chunk_size: usize,
    chunks: Mutex<Vec<*mut u8>>,
    free_list: Mutex<Vec<*mut u8>>,
    allocated: std::sync::atomic::AtomicUsize,
}

unsafe impl Send for EmbeddingPool {}
unsafe impl Sync for EmbeddingPool {}

impl EmbeddingPool {
    /// Create pool with given chunk size (embedding byte size)
    pub fn new(chunk_size: usize, initial_chunks: usize) -> Self {
        let pool = Self {
            chunk_size,
            chunks: Mutex::new(Vec::new()),
            free_list: Mutex::new(Vec::new()),
            allocated: std::sync::atomic::AtomicUsize::new(0),
        };

        // Pre-allocate initial chunks
        for _ in 0..initial_chunks {
            pool.grow();
        }

        pool
    }

    /// Allocate chunk from pool
    pub fn alloc(&self) -> PooledChunk {
        let ptr = {
            let mut free_list = self.free_list.lock().unwrap();
            if let Some(ptr) = free_list.pop() {
                ptr
            } else {
                self.grow();
                self.free_list.lock().unwrap().pop().unwrap()
            }
        };

        self.allocated.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        PooledChunk {
            ptr,
            size: self.chunk_size,
            pool: self,
        }
    }

    /// Return chunk to pool
    fn free(&self, ptr: *mut u8) {
        self.free_list.lock().unwrap().push(ptr);
        self.allocated.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Grow pool by allocating more chunks
    fn grow(&self) {
        let layout = Layout::from_size_align(self.chunk_size, 64).unwrap();

        // Allocate batch of 64 chunks
        for _ in 0..64 {
            let ptr = unsafe { alloc(layout) };
            if ptr.is_null() {
                panic!("Failed to allocate memory pool chunk");
            }
            self.chunks.lock().unwrap().push(ptr);
            self.free_list.lock().unwrap().push(ptr);
        }
    }

    /// Pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            chunk_size: self.chunk_size,
            total_chunks: self.chunks.lock().unwrap().len(),
            allocated: self.allocated.load(std::sync::atomic::Ordering::Relaxed),
            free: self.free_list.lock().unwrap().len(),
        }
    }
}

impl Drop for EmbeddingPool {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.chunk_size, 64).unwrap();
        for ptr in self.chunks.lock().unwrap().drain(..) {
            unsafe { dealloc(ptr, layout) };
        }
    }
}

/// RAII guard for pooled chunk
pub struct PooledChunk<'a> {
    ptr: *mut u8,
    size: usize,
    pool: &'a EmbeddingPool,
}

impl<'a> PooledChunk<'a> {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr as *const f32,
                self.size / std::mem::size_of::<f32>(),
            )
        }
    }
}

impl<'a> Drop for PooledChunk<'a> {
    fn drop(&mut self) {
        self.pool.free(self.ptr);
    }
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub chunk_size: usize,
    pub total_chunks: usize,
    pub allocated: usize,
    pub free: usize,
}
```

### Lazy Loading

```rust
// crates/context-graph-storage/src/lazy/mod.rs

use std::sync::Arc;
use tokio::sync::OnceCell;

/// Lazy-loaded embedding with on-demand decompression
pub struct LazyEmbedding {
    id: Uuid,
    compressed: Option<Vec<u8>>,
    decompressed: OnceCell<Vec<f32>>,
    store: Arc<dyn EmbeddingStore>,
}

impl LazyEmbedding {
    /// Create lazy embedding from compressed data
    pub fn from_compressed(id: Uuid, compressed: Vec<u8>, store: Arc<dyn EmbeddingStore>) -> Self {
        Self {
            id,
            compressed: Some(compressed),
            decompressed: OnceCell::new(),
            store,
        }
    }

    /// Create lazy embedding that loads on demand
    pub fn from_id(id: Uuid, store: Arc<dyn EmbeddingStore>) -> Self {
        Self {
            id,
            compressed: None,
            decompressed: OnceCell::new(),
            store,
        }
    }

    /// Get embedding (loads/decompresses on first access)
    pub async fn get(&self) -> &[f32] {
        self.decompressed.get_or_init(|| async {
            if let Some(ref compressed) = self.compressed {
                // Decompress
                decompress_embedding(compressed)
            } else {
                // Load from store
                self.store.get_embedding(self.id).await.unwrap_or_default()
            }
        }).await
    }

    /// Check if loaded
    pub fn is_loaded(&self) -> bool {
        self.decompressed.initialized()
    }

    /// Unload (drop decompressed data)
    pub fn unload(&mut self) {
        // Note: OnceCell doesn't support reset, would need RwLock<Option<>>
    }
}

/// Batch lazy loading for efficiency
pub struct LazyBatch {
    embeddings: Vec<LazyEmbedding>,
    prefetch_threshold: usize,
}

impl LazyBatch {
    /// Prefetch embeddings if batch exceeds threshold
    pub async fn prefetch(&self) {
        if self.embeddings.len() >= self.prefetch_threshold {
            // Batch load all embeddings
            let futures: Vec<_> = self.embeddings.iter()
                .map(|e| e.get())
                .collect();

            futures::future::join_all(futures).await;
        }
    }
}

fn decompress_embedding(compressed: &[u8]) -> Vec<f32> {
    // Would use actual decompression (LZ4, etc.)
    vec![]
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Memory reduction | 60-70% via quantization |
| Cache hit rate | > 90% for hot data |
| Allocation overhead | < 1% via pooling |
| Lazy load latency | < 1ms |

## Verification

- [ ] Memory-mapped store works with 10M entries
- [ ] LRU cache maintains < 1GB memory
- [ ] Memory pool reduces allocation time
- [ ] Lazy loading doesn't impact latency
- [ ] Overall memory reduction > 60%

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-storage/src/mmap/mod.rs` | Memory-mapped storage |
| `crates/context-graph-storage/src/cache/embedding_cache.rs` | LRU cache |
| `crates/context-graph-core/src/memory/pool.rs` | Memory pool |
| `crates/context-graph-storage/src/lazy/mod.rs` | Lazy loading |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Memory fragmentation | Medium | Medium | Pool allocation |
| mmap file corruption | Low | High | Checksums, journaling |
| Cache thrashing | Low | Medium | Adaptive sizing |

## Traceability

- Source: Constitution memory constraints, large-scale requirements
- Related: TASK-CORE-013 (Quantization)
