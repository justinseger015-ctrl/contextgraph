# TASK-P1-008: MDFileWatcher

```xml
<task_spec id="TASK-P1-008" version="2.0">
<metadata>
  <title>MDFileWatcher Implementation</title>
  <status>COMPLETE</status>
  <layer>surface</layer>
  <sequence>13</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-05</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETE">TASK-P1-004</task_ref>
    <task_ref status="COMPLETE">TASK-P1-005</task_ref>
    <task_ref status="COMPLETE">TASK-P1-007</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <last_updated>2026-01-16</last_updated>
  <completion_date>2026-01-16</completion_date>
</metadata>

<implementation_audit date="2026-01-16">
## Implementation Summary

### Files Created
- `crates/context-graph-core/src/memory/watcher.rs` (~600 lines)
  - WatcherError enum with 8 variants (thiserror)
  - MDFileWatcher struct with full lifecycle management
  - 17 comprehensive tests including FSV persistence test

### Files Modified
- `crates/context-graph-core/Cargo.toml`
  - Added: `notify = "8.2"`, `notify-debouncer-mini = "0.7"`
- `crates/context-graph-core/src/memory/mod.rs`
  - Added: `pub mod watcher;`
  - Added: `pub use watcher::{MDFileWatcher, WatcherError};`

### Verification Results
- **cargo check**: PASS (no errors)
- **cargo clippy**: PASS for watcher.rs (no warnings)
- **cargo test watcher**: 17/17 tests pass
- **FSV persistence test**: PASS - memories persist across store reopening
- **AP-08 compliance**: Uses tokio::fs for all file I/O
- **AP-14 compliance**: No unwrap() calls in production code
- **Code review**: No critical issues (code-simplifier agent)

### Key Implementation Details
1. Uses notify v8.2 with debouncer-mini v0.7 for file system watching
2. 1000ms debounce timeout per constitution spec
3. SHA256 hash tracking for content change detection
4. NonRecursive directory watching mode
5. Async file operations via tokio::fs
6. Full error context propagation
7. Clean lifecycle: new() -> start() -> process_events() -> stop()
8. Drop implementation ensures cleanup

### Test Coverage
- Constructor validation (path exists, is directory)
- Initial scan of existing files
- New file detection
- Modified file detection
- Non-markdown file filtering
- Hash-based duplicate prevention
- Lifecycle state management
- Edge cases (empty files, large files, unicode, special filenames)
- FSV persistence verification
</implementation_audit>

<context>
## Purpose
Implements the MDFileWatcher component that monitors directories for markdown
file changes and automatically chunks and captures them as memories. This is
the final component of Phase 1 Memory Capture, enabling automatic ingestion
of markdown documentation into the Context Graph system.

## Architecture Position
- **Layer**: Surface (consumes foundation/logic layers)
- **Role**: File system event source for memory capture pipeline
- **Flow**: File Events -> TextChunker -> MemoryCaptureService -> MemoryStore

## Constitution Compliance
Per constitution.yaml:
- memory_sources.MDFileChunk: "File system events (create/modify .md files)"
- ARCH-11: "Memory sources: HookDescription, ClaudeResponse, MDFileChunk"
- AP-08: "No sync I/O in async context" - must use tokio::fs for file reads
- AP-14: "No .unwrap() in library code" - all errors propagate via Result

## Why This Component Exists
1. Enables automatic documentation indexing without manual intervention
2. Provides continuous memory capture from markdown knowledge bases
3. Completes the Phase 1 "Memory Sources" triad (Hook/Response/MDFile)
</context>

<codebase_audit date="2026-01-16">
## Verified Existing Components

### 1. Memory Module Structure (CONFIRMED)
Location: `crates/context-graph-core/src/memory/`
Files:
  - mod.rs: Memory struct, ChunkMetadata, exports (658 lines)
  - source.rs: MemorySource enum, HookType, ResponseType
  - chunker.rs: TextChunker, TextChunk, ChunkerError (1309 lines)
  - capture.rs: MemoryCaptureService, CaptureError, EmbeddingProvider (1061 lines)
  - store.rs: MemoryStore, StorageError (COMPLETE with RocksDB)
  - manager.rs: SessionManager (COMPLETE)
  - session.rs: Session, SessionStatus (COMPLETE)

### 2. Key Dependencies Available
From Cargo.toml (crates/context-graph-core/Cargo.toml):
  - sha2 = "0.10" (for file hashing)
  - tokio.workspace = true (async runtime)
  - tracing.workspace = true (logging)
  - thiserror.workspace = true (error handling)
  - async-trait.workspace = true (trait async methods)

MISSING: notify crate - MUST BE ADDED

### 3. Verified API Contracts

#### TextChunker (chunker.rs)
```rust
impl TextChunker {
    pub const CHUNK_SIZE_WORDS: usize = 200;
    pub const OVERLAP_WORDS: usize = 50;
    pub fn default_config() -> Self;
    pub fn chunk_text(&self, content: &str, file_path: &str) -> Result<Vec<TextChunk>, ChunkerError>;
}
```

#### TextChunk (chunker.rs)
```rust
pub struct TextChunk {
    pub content: String,
    pub word_count: u32,
    pub metadata: ChunkMetadata,
}
impl TextChunk {
    pub fn new(content: String, metadata: ChunkMetadata) -> Self;
}
```

#### ChunkMetadata (mod.rs)
```rust
pub struct ChunkMetadata {
    pub file_path: String,
    pub chunk_index: u32,
    pub total_chunks: u32,
    pub word_offset: u32,
    pub char_offset: u32,
    pub original_file_hash: String,
}
```

#### MemoryCaptureService (capture.rs)
```rust
impl MemoryCaptureService {
    pub fn new(store: Arc<MemoryStore>, embedder: Arc<dyn EmbeddingProvider>) -> Self;
    pub async fn capture_md_chunk(&self, chunk: TextChunk, session_id: String) -> Result<Uuid, CaptureError>;
}
```

#### MemoryStore (store.rs)
```rust
impl MemoryStore {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, StorageError>;
    pub fn store(&self, memory: &Memory) -> Result<(), StorageError>;
    pub fn get(&self, id: Uuid) -> Result<Option<Memory>, StorageError>;
    pub fn count(&self) -> Result<u64, StorageError>;
}
```
</codebase_audit>

<input_context_files>
  <file purpose="technical_spec" path="docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md">
    Contains: MDFileWatcher component contract, WatcherError enum spec, database schema for file_hashes CF
  </file>
  <file purpose="chunker_implementation" path="crates/context-graph-core/src/memory/chunker.rs">
    Contains: TextChunker, TextChunk, ChunkerError - verified working with 90%+ test coverage
  </file>
  <file purpose="capture_service" path="crates/context-graph-core/src/memory/capture.rs">
    Contains: MemoryCaptureService.capture_md_chunk() - verified working
  </file>
  <file purpose="memory_module" path="crates/context-graph-core/src/memory/mod.rs">
    Contains: Memory struct, ChunkMetadata struct, public exports
  </file>
  <file purpose="memory_store" path="crates/context-graph-core/src/memory/store.rs">
    Contains: MemoryStore with RocksDB backend - MUST understand CF_MEMORIES, bincode serialization
  </file>
  <file purpose="constitution" path="docs2/constitution.yaml">
    Contains: memory_sources.MDFileChunk spec, ARCH-11, AP-08, AP-14
  </file>
</input_context_files>

<prerequisites>
  <check name="TASK-P1-004_complete">TextChunker exists at chunker.rs and exports from mod.rs</check>
  <check name="TASK-P1-005_complete">MemoryStore exists with store(), get(), count() methods</check>
  <check name="TASK-P1-007_complete">MemoryCaptureService exists with capture_md_chunk() method</check>
  <check name="notify_crate_available">Add notify = "8.0" to Cargo.toml before implementation</check>
  <check name="notify_debouncer_available">Add notify-debouncer-mini = "0.5" for proper debouncing</check>
</prerequisites>

<scope>
  <in_scope>
    - Create watcher.rs file in memory directory
    - Implement WatcherError enum with thiserror
    - Implement MDFileWatcher struct with proper lifecycle
    - File system watching with notify crate (NOT raw inotify)
    - Handle Create/Modify events for .md and .markdown files
    - Debounce rapid changes using notify-debouncer-mini (1000ms)
    - Track file hashes (SHA256) to detect actual content changes
    - Process files through TextChunker (200 words, 50 overlap)
    - Store chunks via MemoryCaptureService.capture_md_chunk()
    - Add pub mod watcher to mod.rs
    - Add notify dependency to Cargo.toml
    - Comprehensive test suite with real filesystem operations
  </in_scope>
  <out_of_scope>
    - CLI integration (Phase 6: TASK-P6-005)
    - Configuration file parsing (uses code constants)
    - Recursive directory watching (explicitly NonRecursive mode)
    - Delete event handling (memories persist, files don't control them)
    - Rename event handling (treat as delete + create)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/memory/watcher.rs">
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use notify_debouncer_mini::{new_debouncer, DebouncedEvent, Debouncer};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use super::{ChunkMetadata, MemoryCaptureService, TextChunk, TextChunker};
use super::capture::CaptureError;
use super::chunker::ChunkerError;

/// Errors from MDFileWatcher operations.
/// All errors include context for debugging.
/// Per constitution AP-14: No .unwrap() - all errors propagate via Result.
#[derive(Debug, Error)]
pub enum WatcherError {
    /// Path does not exist or is not accessible.
    #[error("Path not found: {path:?}")]
    PathNotFound { path: PathBuf },

    /// Path is not a directory (file watching requires directory).
    #[error("Path is not a directory: {path:?}")]
    NotADirectory { path: PathBuf },

    /// notify crate watcher initialization failed.
    #[error("Failed to initialize watcher: {0}")]
    WatchFailed(#[from] notify::Error),

    /// File read operation failed.
    #[error("Failed to read file {path:?}: {source}")]
    ReadFailed {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// File is not valid UTF-8.
    #[error("File is not valid UTF-8: {path:?}")]
    InvalidUtf8 { path: PathBuf },

    /// Text chunking failed.
    #[error("Chunking failed for {path:?}: {source}")]
    ChunkingFailed {
        path: PathBuf,
        #[source]
        source: ChunkerError,
    },

    /// Memory capture failed.
    #[error("Capture failed for {path:?}: {source}")]
    CaptureFailed {
        path: PathBuf,
        #[source]
        source: CaptureError,
    },

    /// Watcher is not started.
    #[error("Watcher not started - call start() first")]
    NotStarted,

    /// Watcher is already running.
    #[error("Watcher already running")]
    AlreadyRunning,
}

/// MDFileWatcher monitors directories for markdown file changes.
///
/// # Architecture
/// - Uses notify crate for cross-platform filesystem events
/// - Debounces rapid changes (1000ms) to avoid duplicate processing
/// - Tracks SHA256 hashes to detect actual content changes
/// - Processes files through TextChunker (200 words, 50 overlap)
/// - Stores chunks via MemoryCaptureService
///
/// # Thread Safety
/// The watcher is `Send + Sync` and can be shared across async tasks.
/// File hash tracking uses `Arc<RwLock<HashMap>>` for thread-safe access.
///
/// # Lifecycle
/// 1. Create with `new(paths, capture_service, session_id)`
/// 2. Call `start()` to begin watching and perform initial scan
/// 3. Call `process_events()` periodically to handle queued events
/// 4. Call `stop()` for clean shutdown
pub struct MDFileWatcher {
    /// Debounced watcher instance (None until start() called).
    debouncer: Option<Debouncer<RecommendedWatcher>>,

    /// Event receiver channel (None until start() called).
    event_rx: Option<std::sync::mpsc::Receiver<Result<Vec<DebouncedEvent>, notify::Error>>>,

    /// Memory capture service for storing chunks.
    capture_service: Arc<MemoryCaptureService>,

    /// Text chunker for splitting files into chunks.
    chunker: TextChunker,

    /// File content hashes for change detection.
    /// Key: canonical path, Value: SHA256 hash
    file_hashes: Arc<RwLock<HashMap<PathBuf, String>>>,

    /// Session ID for captured memories.
    session_id: String,

    /// Paths being watched.
    watch_paths: Vec<PathBuf>,

    /// Running state flag.
    is_running: bool,
}

impl MDFileWatcher {
    /// Debounce timeout in milliseconds.
    /// Per constitution: "Debounce events for 1000ms"
    pub const DEBOUNCE_MS: u64 = 1000;

    /// Create a new MDFileWatcher.
    ///
    /// # Arguments
    /// * `watch_paths` - Directories to watch for .md files
    /// * `capture_service` - Service for capturing memories
    /// * `session_id` - Session to associate memories with
    ///
    /// # Errors
    /// * `WatcherError::PathNotFound` - Path does not exist
    /// * `WatcherError::NotADirectory` - Path is not a directory
    ///
    /// # Example
    /// ```ignore
    /// let capture_service = Arc::new(MemoryCaptureService::new(store, embedder));
    /// let watcher = MDFileWatcher::new(
    ///     vec![PathBuf::from("./docs")],
    ///     capture_service,
    ///     "session-123".to_string(),
    /// )?;
    /// ```
    pub fn new(
        watch_paths: Vec<PathBuf>,
        capture_service: Arc<MemoryCaptureService>,
        session_id: String,
    ) -> Result<Self, WatcherError> {
        // Fail fast: validate all paths exist and are directories
        for path in &watch_paths {
            if !path.exists() {
                return Err(WatcherError::PathNotFound { path: path.clone() });
            }
            if !path.is_dir() {
                return Err(WatcherError::NotADirectory { path: path.clone() });
            }
        }

        Ok(Self {
            debouncer: None,
            event_rx: None,
            capture_service,
            chunker: TextChunker::default_config(),
            file_hashes: Arc::new(RwLock::new(HashMap::new())),
            session_id,
            watch_paths,
            is_running: false,
        })
    }

    /// Start watching directories and perform initial scan.
    ///
    /// # Behavior
    /// 1. Creates debounced watcher (1000ms debounce)
    /// 2. Registers all watch_paths with NonRecursive mode
    /// 3. Performs initial scan of existing .md files
    /// 4. Sets is_running = true
    ///
    /// # Errors
    /// * `WatcherError::AlreadyRunning` - start() called twice
    /// * `WatcherError::WatchFailed` - notify watcher creation failed
    #[instrument(skip(self))]
    pub async fn start(&mut self) -> Result<(), WatcherError> {
        if self.is_running {
            return Err(WatcherError::AlreadyRunning);
        }

        let (tx, rx) = std::sync::mpsc::channel();

        // Create debounced watcher with 1000ms timeout
        let debouncer = new_debouncer(
            Duration::from_millis(Self::DEBOUNCE_MS),
            tx,
        )?;

        self.debouncer = Some(debouncer);
        self.event_rx = Some(rx);

        // Register all paths with watcher
        if let Some(ref mut debouncer) = self.debouncer {
            for path in &self.watch_paths {
                debouncer.watcher().watch(path, RecursiveMode::NonRecursive)?;
                info!(path = ?path, "Watching directory");
            }
        }

        self.is_running = true;

        // Perform initial scan of existing files
        for path in self.watch_paths.clone() {
            if let Err(e) = self.scan_directory(&path).await {
                warn!(path = ?path, error = %e, "Initial scan failed for directory");
            }
        }

        info!(
            paths = ?self.watch_paths,
            session_id = %self.session_id,
            "MDFileWatcher started"
        );

        Ok(())
    }

    /// Process pending events from the watcher.
    ///
    /// Call this method periodically (e.g., in a loop with sleep).
    /// Non-blocking: returns immediately if no events queued.
    ///
    /// # Returns
    /// Number of files processed in this call.
    ///
    /// # Errors
    /// * `WatcherError::NotStarted` - start() not called yet
    #[instrument(skip(self))]
    pub async fn process_events(&mut self) -> Result<usize, WatcherError> {
        if !self.is_running {
            return Err(WatcherError::NotStarted);
        }

        let mut files_processed = 0;

        if let Some(ref rx) = self.event_rx {
            while let Ok(result) = rx.try_recv() {
                match result {
                    Ok(events) => {
                        for event in events {
                            if self.is_markdown(&event.path) {
                                match self.process_file(&event.path).await {
                                    Ok(ids) => {
                                        if !ids.is_empty() {
                                            files_processed += 1;
                                            info!(
                                                path = ?event.path,
                                                chunks = ids.len(),
                                                "Processed markdown file"
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        error!(path = ?event.path, error = %e, "Failed to process file");
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Watch error");
                    }
                }
            }
        }

        Ok(files_processed)
    }

    /// Process a single markdown file.
    ///
    /// # Behavior
    /// 1. Read file content (async)
    /// 2. Compute SHA256 hash
    /// 3. Check if content changed (hash comparison)
    /// 4. If changed: chunk and capture
    /// 5. Update hash cache
    ///
    /// # Returns
    /// Vec of memory UUIDs created (empty if file unchanged).
    #[instrument(skip(self))]
    async fn process_file(&self, path: &Path) -> Result<Vec<Uuid>, WatcherError> {
        // Read file content using async I/O
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| WatcherError::ReadFailed {
                path: path.to_path_buf(),
                source: e,
            })?;

        // Compute SHA256 hash
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        // Get canonical path for consistent hash key
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

        // Check if file content actually changed
        {
            let hashes = self.file_hashes.read().await;
            if let Some(existing_hash) = hashes.get(&canonical) {
                if existing_hash == &hash {
                    debug!(path = ?path, "File unchanged, skipping");
                    return Ok(Vec::new());
                }
            }
        }

        // Content changed - update hash
        {
            let mut hashes = self.file_hashes.write().await;
            hashes.insert(canonical, hash);
        }

        // Chunk the content
        let path_str = path.to_string_lossy().to_string();
        let chunks = self.chunker.chunk_text(&content, &path_str).map_err(|e| {
            WatcherError::ChunkingFailed {
                path: path.to_path_buf(),
                source: e,
            }
        })?;

        info!(path = ?path, chunks = chunks.len(), "Chunked file");

        // Capture each chunk
        let mut memory_ids = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let id = self
                .capture_service
                .capture_md_chunk(chunk, self.session_id.clone())
                .await
                .map_err(|e| WatcherError::CaptureFailed {
                    path: path.to_path_buf(),
                    source: e,
                })?;
            memory_ids.push(id);
        }

        Ok(memory_ids)
    }

    /// Scan a directory for existing markdown files.
    ///
    /// Called during start() for initial indexing.
    #[instrument(skip(self))]
    async fn scan_directory(&self, dir: &Path) -> Result<usize, WatcherError> {
        let mut files_processed = 0;

        let mut entries = tokio::fs::read_dir(dir)
            .await
            .map_err(|e| WatcherError::ReadFailed {
                path: dir.to_path_buf(),
                source: e,
            })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            WatcherError::ReadFailed {
                path: dir.to_path_buf(),
                source: e,
            }
        })? {
            let path = entry.path();
            if path.is_file() && self.is_markdown(&path) {
                match self.process_file(&path).await {
                    Ok(ids) => {
                        if !ids.is_empty() {
                            files_processed += 1;
                        }
                    }
                    Err(e) => {
                        warn!(path = ?path, error = %e, "Failed to process file during scan");
                    }
                }
            }
        }

        info!(dir = ?dir, files = files_processed, "Directory scan complete");
        Ok(files_processed)
    }

    /// Check if a path is a markdown file.
    fn is_markdown(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("md") || ext.eq_ignore_ascii_case("markdown"))
            .unwrap_or(false)
    }

    /// Stop watching and clean up resources.
    ///
    /// Idempotent: safe to call multiple times.
    pub fn stop(&mut self) {
        self.debouncer = None;
        self.event_rx = None;
        self.is_running = false;
        info!(session_id = %self.session_id, "MDFileWatcher stopped");
    }

    /// Get the number of files in the hash cache.
    pub async fn cached_file_count(&self) -> usize {
        self.file_hashes.read().await.len()
    }

    /// Check if the watcher is currently running.
    pub fn is_running(&self) -> bool {
        self.is_running
    }
}

// Ensure proper cleanup on drop
impl Drop for MDFileWatcher {
    fn drop(&mut self) {
        if self.is_running {
            self.stop();
        }
    }
}
    </signature>
  </signatures>

  <constraints>
    - Only process .md and .markdown files (case-insensitive)
    - Debounce events for 1000ms (use notify-debouncer-mini)
    - Skip files that haven't changed (SHA256 hash comparison)
    - Handle errors gracefully - log and continue, don't crash watcher
    - Use tokio::fs for async file I/O (AP-08 compliance)
    - Never use .unwrap() - propagate all errors (AP-14 compliance)
    - Log all processed files with tracing
    - NonRecursive mode only - no subdirectory watching
  </constraints>

  <verification>
    - Watcher detects new .md files created in watched directory
    - Watcher detects modified .md files
    - Watcher ignores non-.md files
    - Hash comparison prevents duplicate processing of unchanged files
    - Debouncing prevents rapid reprocessing (1000ms window)
    - Files are chunked with 200-word chunks, 50-word overlap
    - Each chunk is captured via MemoryCaptureService
    - stop() cleanly shuts down watching
    - All errors logged with context
  </verification>
</definition_of_done>

<cargo_toml_changes>
## Add to crates/context-graph-core/Cargo.toml [dependencies]:

```toml
# File system watching for MDFileWatcher (TASK-P1-008)
notify = "8.0"
notify-debouncer-mini = "0.5"
```

## Why these versions:
- notify 8.0: Latest stable with cross-platform support
- notify-debouncer-mini 0.5: Lightweight debouncer, sufficient for our 1000ms use case
- Alternative: notify-debouncer-full if we need advanced features (we don't)
</cargo_toml_changes>

<mod_rs_changes>
## Add to crates/context-graph-core/src/memory/mod.rs:

After line `pub mod store;` add:
```rust
pub mod watcher;
```

In the pub use section, add:
```rust
pub use watcher::{MDFileWatcher, WatcherError};
```
</mod_rs_changes>

<files_to_create>
  <file path="crates/context-graph-core/src/memory/watcher.rs">
    MDFileWatcher implementation with all methods from definition_of_done
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/memory/mod.rs">
    Add: pub mod watcher; and pub use watcher::{MDFileWatcher, WatcherError};
  </file>
  <file path="crates/context-graph-core/Cargo.toml">
    Add: notify = "8.0" and notify-debouncer-mini = "0.5" to [dependencies]
  </file>
</files_to_modify>

<test_implementation>
## Test File: watcher.rs (within #[cfg(test)] mod tests)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryStore, TestEmbeddingProvider};
    use std::fs;
    use tempfile::tempdir;
    use tokio::time::{sleep, Duration};

    // Helper to create test infrastructure
    async fn setup_test_watcher() -> (MDFileWatcher, Arc<MemoryStore>, tempfile::TempDir, tempfile::TempDir) {
        let db_dir = tempdir().expect("create db temp dir");
        let watch_dir = tempdir().expect("create watch temp dir");

        let store = Arc::new(MemoryStore::new(db_dir.path()).expect("create store"));
        let embedder = Arc::new(TestEmbeddingProvider);
        let capture_service = Arc::new(MemoryCaptureService::new(store.clone(), embedder));

        let watcher = MDFileWatcher::new(
            vec![watch_dir.path().to_path_buf()],
            capture_service,
            "test-session".to_string(),
        ).expect("create watcher");

        (watcher, store, db_dir, watch_dir)
    }

    // ==========================================================================
    // UNIT TESTS - Constructor Validation
    // ==========================================================================

    #[test]
    fn test_new_validates_path_exists() {
        let store = Arc::new(MemoryStore::new(tempdir().unwrap().path()).unwrap());
        let embedder = Arc::new(TestEmbeddingProvider);
        let capture = Arc::new(MemoryCaptureService::new(store, embedder));

        let result = MDFileWatcher::new(
            vec![PathBuf::from("/nonexistent/path/xyz123")],
            capture,
            "session".to_string(),
        );

        assert!(matches!(result, Err(WatcherError::PathNotFound { .. })));
    }

    #[test]
    fn test_new_validates_path_is_directory() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("file.txt");
        fs::write(&file_path, "content").unwrap();

        let store = Arc::new(MemoryStore::new(tempdir().unwrap().path()).unwrap());
        let embedder = Arc::new(TestEmbeddingProvider);
        let capture = Arc::new(MemoryCaptureService::new(store, embedder));

        let result = MDFileWatcher::new(
            vec![file_path],
            capture,
            "session".to_string(),
        );

        assert!(matches!(result, Err(WatcherError::NotADirectory { .. })));
    }

    // ==========================================================================
    // INTEGRATION TESTS - Real Filesystem Operations
    // ==========================================================================

    #[tokio::test]
    async fn test_initial_scan_processes_existing_files() {
        let (mut watcher, store, _db_dir, watch_dir) = setup_test_watcher().await;

        // Create markdown file BEFORE starting watcher
        let md_path = watch_dir.path().join("existing.md");
        fs::write(&md_path, "# Existing Document\n\nThis file exists before watcher starts.").unwrap();

        // Start watcher - should scan existing files
        watcher.start().await.expect("start watcher");

        // Verify file was processed
        let count = store.count().expect("count");
        assert!(count >= 1, "Existing file should be processed during initial scan");

        watcher.stop();
    }

    #[tokio::test]
    async fn test_watcher_detects_new_file() {
        let (mut watcher, store, _db_dir, watch_dir) = setup_test_watcher().await;

        watcher.start().await.expect("start watcher");
        let count_before = store.count().expect("count");

        // Create new markdown file
        let md_path = watch_dir.path().join("new_file.md");
        fs::write(&md_path, "# New File\n\nThis is new content.").unwrap();

        // Wait for debounce + processing
        sleep(Duration::from_millis(1500)).await;
        watcher.process_events().await.expect("process events");

        let count_after = store.count().expect("count");
        assert!(count_after > count_before, "New file should create memories");

        watcher.stop();
    }

    #[tokio::test]
    async fn test_watcher_detects_modified_file() {
        let (mut watcher, store, _db_dir, watch_dir) = setup_test_watcher().await;

        // Create file
        let md_path = watch_dir.path().join("modify_test.md");
        fs::write(&md_path, "# Original Content\n\nVersion 1.").unwrap();

        watcher.start().await.expect("start watcher");
        let count_after_create = store.count().expect("count");

        // Modify file
        fs::write(&md_path, "# Modified Content\n\nVersion 2 with changes.").unwrap();

        sleep(Duration::from_millis(1500)).await;
        watcher.process_events().await.expect("process events");

        let count_after_modify = store.count().expect("count");
        assert!(count_after_modify > count_after_create, "Modified file should create new memories");

        watcher.stop();
    }

    #[tokio::test]
    async fn test_watcher_ignores_non_markdown_files() {
        let (mut watcher, store, _db_dir, watch_dir) = setup_test_watcher().await;

        watcher.start().await.expect("start watcher");
        let count_before = store.count().expect("count");

        // Create non-markdown files
        fs::write(watch_dir.path().join("file.txt"), "text content").unwrap();
        fs::write(watch_dir.path().join("file.rs"), "rust content").unwrap();
        fs::write(watch_dir.path().join("file.json"), "{}").unwrap();

        sleep(Duration::from_millis(1500)).await;
        watcher.process_events().await.expect("process events");

        let count_after = store.count().expect("count");
        assert_eq!(count_after, count_before, "Non-markdown files should be ignored");

        watcher.stop();
    }

    #[tokio::test]
    async fn test_hash_prevents_duplicate_processing() {
        let (mut watcher, store, _db_dir, watch_dir) = setup_test_watcher().await;

        let md_path = watch_dir.path().join("hash_test.md");
        fs::write(&md_path, "# Same Content").unwrap();

        watcher.start().await.expect("start watcher");
        let count_after_first = store.count().expect("count");

        // "Modify" file with same content
        fs::write(&md_path, "# Same Content").unwrap();

        sleep(Duration::from_millis(1500)).await;
        watcher.process_events().await.expect("process events");

        let count_after_second = store.count().expect("count");
        assert_eq!(count_after_second, count_after_first, "Same content should not be reprocessed");

        watcher.stop();
    }

    #[tokio::test]
    async fn test_is_markdown_detection() {
        let dir = tempdir().unwrap();
        let store = Arc::new(MemoryStore::new(dir.path()).unwrap());
        let embedder = Arc::new(TestEmbeddingProvider);
        let capture = Arc::new(MemoryCaptureService::new(store, embedder));
        let watch_dir = tempdir().unwrap();

        let watcher = MDFileWatcher::new(
            vec![watch_dir.path().to_path_buf()],
            capture,
            "session".to_string(),
        ).unwrap();

        // Positive cases
        assert!(watcher.is_markdown(Path::new("file.md")));
        assert!(watcher.is_markdown(Path::new("file.MD")));
        assert!(watcher.is_markdown(Path::new("file.markdown")));
        assert!(watcher.is_markdown(Path::new("file.MARKDOWN")));
        assert!(watcher.is_markdown(Path::new("/path/to/doc.md")));

        // Negative cases
        assert!(!watcher.is_markdown(Path::new("file.txt")));
        assert!(!watcher.is_markdown(Path::new("file.rs")));
        assert!(!watcher.is_markdown(Path::new("file.mdx"))); // Not plain markdown
        assert!(!watcher.is_markdown(Path::new("file"))); // No extension
        assert!(!watcher.is_markdown(Path::new(".md"))); // Hidden file, no name
    }

    #[tokio::test]
    async fn test_start_already_running_error() {
        let (mut watcher, _store, _db_dir, _watch_dir) = setup_test_watcher().await;

        watcher.start().await.expect("first start");
        let result = watcher.start().await;

        assert!(matches!(result, Err(WatcherError::AlreadyRunning)));

        watcher.stop();
    }

    #[tokio::test]
    async fn test_process_events_not_started_error() {
        let (mut watcher, _store, _db_dir, _watch_dir) = setup_test_watcher().await;

        let result = watcher.process_events().await;
        assert!(matches!(result, Err(WatcherError::NotStarted)));
    }

    #[tokio::test]
    async fn test_stop_is_idempotent() {
        let (mut watcher, _store, _db_dir, _watch_dir) = setup_test_watcher().await;

        watcher.start().await.expect("start");
        watcher.stop();
        assert!(!watcher.is_running());

        // Second stop should not panic
        watcher.stop();
        assert!(!watcher.is_running());
    }

    #[tokio::test]
    async fn test_cached_file_count() {
        let (mut watcher, _store, _db_dir, watch_dir) = setup_test_watcher().await;

        // Create files before starting
        fs::write(watch_dir.path().join("file1.md"), "# File 1").unwrap();
        fs::write(watch_dir.path().join("file2.md"), "# File 2").unwrap();
        fs::write(watch_dir.path().join("file3.md"), "# File 3").unwrap();

        watcher.start().await.expect("start");

        let cache_count = watcher.cached_file_count().await;
        assert_eq!(cache_count, 3, "Should cache hashes for 3 files");

        watcher.stop();
    }

    // ==========================================================================
    // EDGE CASE TESTS
    // ==========================================================================

    #[tokio::test]
    async fn edge_case_empty_markdown_file() {
        let (mut watcher, store, _db_dir, watch_dir) = setup_test_watcher().await;

        // Create empty markdown file
        let md_path = watch_dir.path().join("empty.md");
        fs::write(&md_path, "").unwrap();

        watcher.start().await.expect("start watcher");

        // Empty file should be rejected by chunker (ChunkerError::EmptyContent)
        // But watcher should not crash
        assert!(watcher.is_running());

        watcher.stop();
    }

    #[tokio::test]
    async fn edge_case_large_markdown_file() {
        let (mut watcher, store, _db_dir, watch_dir) = setup_test_watcher().await;

        // Create large markdown file (500+ words = multiple chunks)
        let words: Vec<&str> = (0..600).map(|_| "word").collect();
        let content = format!("# Large Document\n\n{}", words.join(" "));
        let md_path = watch_dir.path().join("large.md");
        fs::write(&md_path, content).unwrap();

        watcher.start().await.expect("start watcher");

        // Should create multiple memory chunks
        let count = store.count().expect("count");
        assert!(count >= 3, "Large file should create multiple chunks, got {}", count);

        watcher.stop();
    }

    #[tokio::test]
    async fn edge_case_unicode_content() {
        let (mut watcher, store, _db_dir, watch_dir) = setup_test_watcher().await;

        // Create markdown with Unicode content
        let content = "# 日本語ドキュメント\n\n世界 🌍 こんにちは 世界 言語 テスト";
        let md_path = watch_dir.path().join("unicode.md");
        fs::write(&md_path, content).unwrap();

        watcher.start().await.expect("start watcher");

        let count = store.count().expect("count");
        assert!(count >= 1, "Unicode file should be processed");

        watcher.stop();
    }

    #[tokio::test]
    async fn edge_case_special_filename() {
        let (mut watcher, store, _db_dir, watch_dir) = setup_test_watcher().await;

        // Create file with special characters in name
        let md_path = watch_dir.path().join("file with spaces (1).md");
        fs::write(&md_path, "# Special Name\n\nContent here.").unwrap();

        watcher.start().await.expect("start watcher");

        let count = store.count().expect("count");
        assert!(count >= 1, "File with special name should be processed");

        watcher.stop();
    }

    // ==========================================================================
    // FSV: FULL STATE VERIFICATION TEST
    // ==========================================================================

    #[tokio::test]
    async fn fsv_mdfilewatcher_persistence_verification() {
        println!("\n============================================================");
        println!("=== FSV: MDFileWatcher Persistence Verification ===");
        println!("============================================================\n");

        let db_dir = tempdir().expect("create db temp dir");
        let watch_dir = tempdir().expect("create watch temp dir");

        // Phase 1: Create and process files
        let file_count_created;
        {
            let store = Arc::new(MemoryStore::new(db_dir.path()).expect("create store"));
            let embedder = Arc::new(TestEmbeddingProvider);
            let capture = Arc::new(MemoryCaptureService::new(store.clone(), embedder));

            let mut watcher = MDFileWatcher::new(
                vec![watch_dir.path().to_path_buf()],
                capture,
                "fsv-session".to_string(),
            ).expect("create watcher");

            println!("[FSV-1] Initial store count: {}", store.count().expect("count"));
            assert_eq!(store.count().expect("count"), 0);

            // Create markdown files
            fs::write(watch_dir.path().join("fsv1.md"), "# FSV Test 1\n\nFirst document content.").unwrap();
            fs::write(watch_dir.path().join("fsv2.md"), "# FSV Test 2\n\nSecond document content.").unwrap();

            watcher.start().await.expect("start watcher");

            file_count_created = store.count().expect("count");
            println!("[FSV-2] After initial scan: {} memories", file_count_created);
            assert!(file_count_created >= 2, "Should have at least 2 memories");

            // Add another file
            fs::write(watch_dir.path().join("fsv3.md"), "# FSV Test 3\n\nThird document added after start.").unwrap();
            sleep(Duration::from_millis(1500)).await;
            watcher.process_events().await.expect("process");

            let count_after_add = store.count().expect("count");
            println!("[FSV-3] After adding fsv3.md: {} memories", count_after_add);

            watcher.stop();
            println!("[FSV-4] Watcher stopped, store being dropped...");
        }

        // Phase 2: Reopen store and verify persistence
        println!("\n[FSV-5] Reopening database to verify persistence...");
        {
            let store = MemoryStore::new(db_dir.path()).expect("reopen store");
            let count_after_reopen = store.count().expect("count");
            println!("[FSV-6] Reopened store count: {}", count_after_reopen);

            assert!(
                count_after_reopen >= file_count_created,
                "Memories should persist across store reopening"
            );

            // Verify we can retrieve memories by session
            let session_memories = store.get_by_session("fsv-session").expect("get by session");
            println!("[FSV-7] Memories in fsv-session: {}", session_memories.len());

            for (i, mem) in session_memories.iter().enumerate() {
                println!("[FSV-{}] Memory {}: source={:?}, content_preview='{}'",
                    8 + i, mem.id,
                    mem.source,
                    mem.content.chars().take(50).collect::<String>()
                );
                assert!(mem.is_md_file_chunk(), "Memory should be MDFileChunk source");
            }
        }

        println!("\n============================================================");
        println!("[FSV] VERIFIED: All MDFileWatcher persistence checks passed");
        println!("============================================================\n");
    }
}
```
</test_implementation>

<validation_criteria>
  <criterion>cargo check --package context-graph-core passes with no errors</criterion>
  <criterion>cargo clippy --package context-graph-core -D warnings passes</criterion>
  <criterion>cargo test --package context-graph-core watcher passes all tests</criterion>
  <criterion>cargo test --package context-graph-core watcher::tests::fsv_ passes FSV test</criterion>
  <criterion>No .unwrap() calls in watcher.rs (grep verify)</criterion>
  <criterion>All file I/O uses tokio::fs (grep verify)</criterion>
  <criterion>WatcherError implements std::error::Error via thiserror</criterion>
</validation_criteria>

<test_commands>
  <command description="Add dependencies">cd /home/cabdru/contextgraph && cargo add notify@8.0 notify-debouncer-mini@0.5 --package context-graph-core</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
  <command description="Run clippy">cargo clippy --package context-graph-core -D warnings</command>
  <command description="Run watcher unit tests">cargo test --package context-graph-core watcher -- --nocapture</command>
  <command description="Run FSV test specifically">cargo test --package context-graph-core fsv_mdfilewatcher -- --nocapture</command>
  <command description="Verify no unwrap calls">grep -n "unwrap()" crates/context-graph-core/src/memory/watcher.rs || echo "PASS: No unwrap() calls"</command>
  <command description="Verify async file I/O">grep -n "tokio::fs" crates/context-graph-core/src/memory/watcher.rs</command>
</test_commands>

<full_state_verification>
## Source of Truth
- RocksDB MemoryStore: Primary storage for all Memory objects
- File hashes: In-memory HashMap (not persisted - regenerates on startup)
- Memory count: store.count() method

## Execute & Inspect Steps

### Step 1: Initial State Verification
```bash
# Before implementation
cargo test --package context-graph-core --features test-utils 2>&1 | head -5
# Expected: All existing tests pass
```

### Step 2: After Implementation - Unit Tests
```bash
cargo test --package context-graph-core watcher::tests::test_new_ -- --nocapture
# Expected: Constructor validation tests pass
```

### Step 3: After Implementation - Integration Tests
```bash
cargo test --package context-graph-core watcher::tests::test_watcher_ -- --nocapture
# Expected: All watcher integration tests pass with real filesystem
```

### Step 4: Full State Verification
```bash
cargo test --package context-graph-core watcher::tests::fsv_ -- --nocapture
# Expected: FSV test passes with detailed state logging
```

### Step 5: Manual Verification
```bash
# Create temp directory and test manually
mkdir -p /tmp/watcher_test
echo "# Test" > /tmp/watcher_test/test.md
# Run example or integration test that watches /tmp/watcher_test
```

## Edge Case Simulations

### EC-1: Empty File
- **Input**: Create empty .md file
- **Expected**: ChunkerError::EmptyContent, watcher continues running
- **Before**: store.count() = N
- **After**: store.count() = N (unchanged)

### EC-2: Large File (500+ words)
- **Input**: Create .md file with 600 words
- **Expected**: Multiple chunks created (600/200 = 3+ chunks with overlap)
- **Before**: store.count() = 0
- **After**: store.count() >= 3

### EC-3: Rapid Modifications
- **Input**: Write to file 10 times in 500ms
- **Expected**: Debouncing results in 1 processing event
- **Before**: store.count() = 0
- **After**: store.count() = 1 (single chunk, not 10)

### EC-4: Non-UTF8 Binary File
- **Input**: Write binary data to .md file
- **Expected**: WatcherError::ReadFailed or InvalidUtf8
- **Before**: store.count() = N
- **After**: store.count() = N (unchanged, error logged)

### EC-5: Permission Denied
- **Input**: Create .md file, chmod 000
- **Expected**: WatcherError::ReadFailed
- **Before**: store.count() = N
- **After**: store.count() = N (unchanged, error logged)
</full_state_verification>

<manual_verification>
## Manual Testing Requirements

### M-1: Verify RocksDB State
```bash
# After running FSV test, examine database directory
ls -la /tmp/fsv_test_db/
# Should see RocksDB files: CURRENT, LOCK, LOG, MANIFEST-*, *.sst
```

### M-2: Verify Memory Retrieval
```rust
// In a test or example:
let memories = store.get_by_session("test-session")?;
for m in memories {
    println!("Memory ID: {}", m.id);
    println!("Source: {:?}", m.source);
    println!("Content length: {}", m.content.len());
    println!("Chunk metadata: {:?}", m.chunk_metadata);
    assert!(m.chunk_metadata.is_some(), "MDFileChunk must have chunk_metadata");
}
```

### M-3: Verify File Hash Tracking
```rust
// After processing a file:
let hash_count = watcher.cached_file_count().await;
assert!(hash_count > 0, "Hash cache should be populated");

// Modify file with same content:
// Hash count should not increase (file unchanged)
```

### M-4: Verify Debouncing Behavior
```bash
# Create file and immediately modify multiple times
for i in {1..5}; do
    echo "# Version $i" > /tmp/watcher_test/rapid.md
    sleep 0.1
done
# After 1.5 seconds, check that only 1 memory was created
```

### M-5: Visual Log Inspection
```bash
RUST_LOG=debug cargo test --package context-graph-core fsv_mdfilewatcher -- --nocapture 2>&1 | grep -E "(INFO|DEBUG|WARN|ERROR)"
# Should see:
# INFO: Watching directory
# INFO: Chunked file (N chunks)
# INFO: MDFileWatcher started
# INFO: Directory scan complete
# INFO: MDFileWatcher stopped
```
</manual_verification>

<synthetic_test_data>
## Synthetic Test Inputs

### SYN-001: Valid Markdown File
```markdown
# Project Documentation

This is a sample documentation file for testing the MDFileWatcher.
It contains enough words to be meaningful but short enough to be a single chunk.

## Section 1
Content here describes the first section.

## Section 2
Content here describes the second section.
```
**Expected**: 1 chunk, ~40 words

### SYN-002: Large Multi-Chunk File
Generate 600 words:
```rust
let content = (0..600).map(|i| format!("word{}", i)).collect::<Vec<_>>().join(" ");
```
**Expected**: 3-4 chunks (200 words each, 50 overlap)

### SYN-003: Unicode Content
```markdown
# 日本語ドキュメント

世界 🌍 こんにちは 世界 言語 テスト

## セクション
追加のコンテンツがここにあります。
```
**Expected**: 1 chunk, processed correctly

### SYN-004: Whitespace-Heavy Content
```markdown
# Title


(Many blank lines)


Content after whitespace.
```
**Expected**: Processed, blank lines preserved in content
</synthetic_test_data>

<notes>
  <note category="debouncing">
    Using notify-debouncer-mini instead of manual debouncing.
    The mini version is lightweight and sufficient for our 1000ms use case.
    If advanced features needed (e.g., event coalescing), switch to notify-debouncer-full.
  </note>
  <note category="hash_tracking">
    File hashes are kept in memory only for the watcher session.
    They are NOT persisted to database (unlike original design).
    This is simpler and avoids database schema changes.
    Trade-off: Re-processing on watcher restart, but that's acceptable.
  </note>
  <note category="async_io">
    Per AP-08: "No sync I/O in async context"
    All file reads use tokio::fs::read_to_string and tokio::fs::read_dir.
    The notify crate callback is sync but only signals via channel.
  </note>
  <note category="error_handling">
    Watcher continues running even if individual file processing fails.
    Errors are logged with full context but don't stop the watcher.
    This is "fail-fast" at the operation level, not the service level.
  </note>
  <note category="platform_considerations">
    On WSL2 watching Windows paths: notify may not work, consider PollWatcher.
    On Docker macOS M1: May need PollWatcher due to emulation issues.
    Current implementation uses RecommendedWatcher which auto-selects best backend.
  </note>
  <note category="testing">
    Tests use real filesystem via tempfile crate.
    TestEmbeddingProvider provides zeroed embeddings (test-utils feature).
    FSV test verifies persistence across store reopening.
    No mocks for MemoryStore - uses real RocksDB in temp directory.
  </note>
</notes>
</task_spec>
```

## Execution Checklist

### Prerequisites
- [ ] Verify TASK-P1-007 is complete (MemoryCaptureService.capture_md_chunk exists)
- [ ] Verify TextChunker.default_config() and chunk_text() work
- [ ] Verify MemoryStore.store(), get(), count() work

### Implementation Steps
- [ ] Add notify = "8.0" and notify-debouncer-mini = "0.5" to Cargo.toml
- [ ] Create watcher.rs in crates/context-graph-core/src/memory/
- [ ] Implement WatcherError enum with all variants
- [ ] Implement MDFileWatcher struct with all fields
- [ ] Implement MDFileWatcher::new() with path validation
- [ ] Implement MDFileWatcher::start() with debouncer setup
- [ ] Implement MDFileWatcher::process_events() for event handling
- [ ] Implement MDFileWatcher::process_file() with hash checking
- [ ] Implement MDFileWatcher::scan_directory() for initial scan
- [ ] Implement MDFileWatcher::is_markdown() helper
- [ ] Implement MDFileWatcher::stop() for clean shutdown
- [ ] Implement Drop trait for cleanup
- [ ] Add `pub mod watcher;` to mod.rs
- [ ] Add `pub use watcher::{MDFileWatcher, WatcherError};` to mod.rs

### Verification Steps
- [ ] Run: `cargo check --package context-graph-core`
- [ ] Run: `cargo clippy --package context-graph-core -D warnings`
- [ ] Run: `cargo test --package context-graph-core watcher -- --nocapture`
- [ ] Run: `cargo test --package context-graph-core fsv_ -- --nocapture`
- [ ] Verify: `grep -n "unwrap()" watcher.rs` returns empty
- [ ] Verify: `grep -n "tokio::fs" watcher.rs` shows async I/O usage

### Final Validation
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] FSV test passes with detailed state output
- [ ] No clippy warnings
- [ ] Code follows constitution.yaml rules (AP-08, AP-14)

## Phase 1 Completion
After TASK-P1-008 is complete, Phase 1 "Memory Capture System" is DONE.
All three memory sources are implemented:
- HookDescription (via MemoryCaptureService.capture_hook_description)
- ClaudeResponse (via MemoryCaptureService.capture_claude_response)
- MDFileChunk (via MDFileWatcher + MemoryCaptureService.capture_md_chunk)

Next: Phase 2 "13-Space Embedding" (TASK-P2-xxx series)
