# TASK-P1-002: TextChunk Type and Chunker Module

```xml
<task_spec id="TASK-P1-002" version="2.0">
<metadata>
  <title>TextChunk Type and Chunker Module Setup</title>
  <status>COMPLETE</status>
  <layer>foundation</layer>
  <sequence>7</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-03</requirement_ref>
    <requirement_ref>REQ-P1-04</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETE">TASK-P1-001</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
  <last_audit>2026-01-16</last_audit>
</metadata>
```

## CRITICAL: Current Codebase State (VERIFIED 2026-01-16)

### What ALREADY EXISTS (DO NOT RECREATE)

**1. ChunkMetadata struct** - FULLY IMPLEMENTED
- **Location**: `crates/context-graph-core/src/memory/mod.rs` lines 44-62
- **Re-exported from**: `crates/context-graph-core/src/lib.rs` line 91
- **Status**: Complete with all 6 fields per spec

```rust
// EXISTING - DO NOT RECREATE
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkMetadata {
    pub file_path: String,           // Path to source file
    pub chunk_index: u32,            // Zero-based chunk index
    pub total_chunks: u32,           // Total chunks from file
    pub word_offset: u32,            // Word offset from file start
    pub char_offset: u32,            // Character offset from file start
    pub original_file_hash: String,  // SHA256 of original file
}
```

**2. Memory module structure**
- **Directory**: `crates/context-graph-core/src/memory/` EXISTS
- **Files present**:
  - `mod.rs` - Memory struct, ChunkMetadata, re-exports
  - `source.rs` - MemorySource enum, HookType, ResponseType
- **Memory struct** uses `Option<ChunkMetadata>` for chunk tracking

**3. Dependencies in Cargo.toml** - ALL PRESENT
```toml
serde.workspace = true       # Serialize, Deserialize
sha2.workspace = true        # SHA256 hashing (verify with cargo tree)
```

### What DOES NOT EXIST (MUST CREATE)

**1. TextChunk struct** - NOT IMPLEMENTED
- **Target Location**: `crates/context-graph-core/src/memory/chunker.rs`
- **Purpose**: Transient container combining content + word count + metadata
- **NOT stored directly** - converted to Memory before storage

**2. chunker.rs module file** - NOT CREATED
- File `crates/context-graph-core/src/memory/chunker.rs` does NOT exist

## Scope Definition

### IN SCOPE (This Task)
1. Create `chunker.rs` with `TextChunk` struct only
2. Add `pub mod chunker;` to memory/mod.rs
3. Re-export `TextChunk` from memory module
4. Unit tests for TextChunk creation and field access

### OUT OF SCOPE (Future Tasks)
- `TextChunker` component with chunking algorithm (TASK-P1-004)
- Moving `ChunkMetadata` from mod.rs to chunker.rs (NOT REQUIRED)
- Chunking logic, sentence boundary detection (TASK-P1-004)
- File reading, hash computation (TASK-P1-004)

## Implementation

### File to Create: `crates/context-graph-core/src/memory/chunker.rs`

```rust
//! Text chunking types for the Context Graph system.
//!
//! This module provides the [`TextChunk`] type used by the TextChunker
//! to represent chunks of text with metadata.
//!
//! # Note
//! [`ChunkMetadata`] is defined in the parent module (`memory/mod.rs`)
//! and imported here for use with TextChunk.
//!
//! # Constitution Compliance
//! - memory_sources.MDFileChunk.chunking: 200 words, 50 overlap
//! - TextChunk is transient - NOT stored directly, converted to Memory

use super::ChunkMetadata;

/// A chunk of text with associated metadata.
///
/// TextChunk is a **transient** container used during the chunking process.
/// It combines:
/// - The chunk content (text)
/// - Word count (pre-computed for efficiency)
/// - Full metadata about chunk origin and position
///
/// # Lifecycle
/// 1. Created by TextChunker during file processing
/// 2. Passed to MemoryCaptureService
/// 3. Converted to Memory struct (with TeleologicalArray embedding)
/// 4. Memory is persisted to storage
///
/// TextChunk itself is NEVER stored directly.
///
/// # Example
/// ```rust
/// use context_graph_core::memory::{TextChunk, ChunkMetadata};
///
/// let metadata = ChunkMetadata {
///     file_path: "docs/readme.md".to_string(),
///     chunk_index: 0,
///     total_chunks: 3,
///     word_offset: 0,
///     char_offset: 0,
///     original_file_hash: "abc123...".to_string(),
/// };
///
/// let chunk = TextChunk::new(
///     "This is the first chunk of text content.".to_string(),
///     metadata,
/// );
///
/// assert_eq!(chunk.word_count, 8);
/// assert_eq!(chunk.metadata.chunk_index, 0);
/// ```
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// The chunk content (text extracted from source).
    pub content: String,

    /// Number of words in content.
    /// Pre-computed on creation for efficiency.
    pub word_count: u32,

    /// Full metadata about chunk origin and position.
    pub metadata: ChunkMetadata,
}

impl TextChunk {
    /// Create a new TextChunk with auto-computed word count.
    ///
    /// # Arguments
    /// * `content` - The text content of this chunk
    /// * `metadata` - Metadata about chunk origin and position
    ///
    /// # Word Count
    /// Computed as whitespace-separated tokens: `content.split_whitespace().count()`
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::memory::{TextChunk, ChunkMetadata};
    ///
    /// let meta = ChunkMetadata {
    ///     file_path: "test.md".to_string(),
    ///     chunk_index: 0,
    ///     total_chunks: 1,
    ///     word_offset: 0,
    ///     char_offset: 0,
    ///     original_file_hash: "hash".to_string(),
    /// };
    ///
    /// let chunk = TextChunk::new("hello world".to_string(), meta);
    /// assert_eq!(chunk.word_count, 2);
    /// ```
    pub fn new(content: String, metadata: ChunkMetadata) -> Self {
        let word_count = content.split_whitespace().count() as u32;
        Self {
            content,
            word_count,
            metadata,
        }
    }

    /// Create TextChunk with explicit word count (for testing/reconstruction).
    ///
    /// # Warning
    /// Caller is responsible for ensuring word_count matches content.
    /// Use `new()` in production code for automatic counting.
    pub fn with_word_count(content: String, word_count: u32, metadata: ChunkMetadata) -> Self {
        Self {
            content,
            word_count,
            metadata,
        }
    }

    /// Check if this chunk is empty (zero words).
    pub fn is_empty(&self) -> bool {
        self.word_count == 0
    }

    /// Get the content length in bytes.
    pub fn byte_len(&self) -> usize {
        self.content.len()
    }

    /// Get the content length in characters.
    pub fn char_len(&self) -> usize {
        self.content.chars().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metadata() -> ChunkMetadata {
        ChunkMetadata {
            file_path: "/test/file.md".to_string(),
            chunk_index: 0,
            total_chunks: 1,
            word_offset: 0,
            char_offset: 0,
            original_file_hash: "testhash123".to_string(),
        }
    }

    #[test]
    fn test_text_chunk_new_computes_word_count() {
        let chunk = TextChunk::new(
            "the quick brown fox jumps over the lazy dog".to_string(),
            test_metadata(),
        );
        assert_eq!(chunk.word_count, 9);
    }

    #[test]
    fn test_text_chunk_empty_content() {
        let chunk = TextChunk::new(String::new(), test_metadata());
        assert_eq!(chunk.word_count, 0);
        assert!(chunk.is_empty());
    }

    #[test]
    fn test_text_chunk_whitespace_only() {
        let chunk = TextChunk::new("   \t\n   ".to_string(), test_metadata());
        assert_eq!(chunk.word_count, 0);
        assert!(chunk.is_empty());
    }

    #[test]
    fn test_text_chunk_single_word() {
        let chunk = TextChunk::new("hello".to_string(), test_metadata());
        assert_eq!(chunk.word_count, 1);
        assert!(!chunk.is_empty());
    }

    #[test]
    fn test_text_chunk_extra_whitespace() {
        let chunk = TextChunk::new("  hello   world  ".to_string(), test_metadata());
        assert_eq!(chunk.word_count, 2);
    }

    #[test]
    fn test_text_chunk_metadata_access() {
        let mut meta = test_metadata();
        meta.chunk_index = 5;
        meta.total_chunks = 10;

        let chunk = TextChunk::new("content".to_string(), meta);

        assert_eq!(chunk.metadata.chunk_index, 5);
        assert_eq!(chunk.metadata.total_chunks, 10);
        assert_eq!(chunk.metadata.file_path, "/test/file.md");
    }

    #[test]
    fn test_text_chunk_with_word_count() {
        let chunk = TextChunk::with_word_count(
            "hello world".to_string(),
            2,
            test_metadata(),
        );
        assert_eq!(chunk.word_count, 2);
    }

    #[test]
    fn test_text_chunk_byte_len() {
        let chunk = TextChunk::new("hello".to_string(), test_metadata());
        assert_eq!(chunk.byte_len(), 5);
    }

    #[test]
    fn test_text_chunk_char_len_unicode() {
        // "hello" in Japanese: こんにちは (5 chars, 15 bytes)
        let chunk = TextChunk::new("こんにちは".to_string(), test_metadata());
        assert_eq!(chunk.char_len(), 5);
        assert_eq!(chunk.byte_len(), 15); // 3 bytes per char
    }

    #[test]
    fn test_text_chunk_clone() {
        let original = TextChunk::new("test content".to_string(), test_metadata());
        let cloned = original.clone();

        assert_eq!(original.content, cloned.content);
        assert_eq!(original.word_count, cloned.word_count);
        assert_eq!(original.metadata.file_path, cloned.metadata.file_path);
    }

    #[test]
    fn test_text_chunk_debug_format() {
        let chunk = TextChunk::new("debug test".to_string(), test_metadata());
        let debug_str = format!("{:?}", chunk);

        assert!(debug_str.contains("TextChunk"));
        assert!(debug_str.contains("debug test"));
        assert!(debug_str.contains("word_count"));
    }
}
```

### File to Modify: `crates/context-graph-core/src/memory/mod.rs`

Add after line 34 (`pub mod source;`):
```rust
pub mod chunker;
```

Update the pub use statement (around line 36) to add TextChunk:
```rust
pub use chunker::TextChunk;
```

### Verify Re-export in lib.rs

The existing line 91 should become:
```rust
pub use memory::{ChunkMetadata, HookType, Memory, MemorySource, ResponseType, TextChunk, MAX_CONTENT_LENGTH};
```

## Definition of Done

### Compilation Checks (MUST ALL PASS)
```bash
# Check compilation - zero errors
cargo check --package context-graph-core

# Run tests - all must pass
cargo test --package context-graph-core memory::chunker::tests::

# Clippy - zero warnings
cargo clippy --package context-graph-core -- -D warnings

# Verify exports work
cargo doc --package context-graph-core --no-deps 2>&1 | grep -i "textchunk"
```

### Signatures Verification
```rust
// MUST compile after implementation:
use context_graph_core::memory::{TextChunk, ChunkMetadata};

let meta = ChunkMetadata {
    file_path: "test.md".to_string(),
    chunk_index: 0,
    total_chunks: 1,
    word_offset: 0,
    char_offset: 0,
    original_file_hash: "abc".to_string(),
};

let chunk = TextChunk::new("hello world".to_string(), meta);
assert_eq!(chunk.word_count, 2);
```

## Full State Verification Protocol

### Source of Truth
- **Primary**: File existence at `crates/context-graph-core/src/memory/chunker.rs`
- **Secondary**: Module declaration in `crates/context-graph-core/src/memory/mod.rs`
- **Tertiary**: Re-export in `crates/context-graph-core/src/lib.rs`

### Execute & Inspect Sequence

1. **Verify file creation**:
```bash
ls -la crates/context-graph-core/src/memory/chunker.rs
# EXPECTED: File exists with size > 0
```

2. **Verify module declaration**:
```bash
grep "pub mod chunker" crates/context-graph-core/src/memory/mod.rs
# EXPECTED: "pub mod chunker;" found
```

3. **Verify re-export**:
```bash
grep "TextChunk" crates/context-graph-core/src/memory/mod.rs
grep "TextChunk" crates/context-graph-core/src/lib.rs
# EXPECTED: Both files contain TextChunk in pub use statement
```

4. **Verify compilation**:
```bash
cargo check --package context-graph-core 2>&1
# EXPECTED: "Finished" with no errors
```

5. **Run tests and capture output**:
```bash
cargo test --package context-graph-core memory::chunker::tests:: -- --nocapture 2>&1 | tee /tmp/task-p1-002-evidence.log
# EXPECTED: All tests pass (9 tests)
```

### Boundary & Edge Case Audit

**Edge Case 1: Empty Content String**
```
INPUT:  TextChunk::new("".to_string(), metadata)
EXPECTED OUTPUT: word_count = 0, is_empty() = true
VERIFICATION: Run test_text_chunk_empty_content, verify assertion passes
STATE BEFORE: chunk does not exist
STATE AFTER: chunk.word_count == 0
```

**Edge Case 2: Whitespace-Only Content**
```
INPUT:  TextChunk::new("   \t\n   ".to_string(), metadata)
EXPECTED OUTPUT: word_count = 0 (no words in whitespace)
VERIFICATION: Run test_text_chunk_whitespace_only
STATE BEFORE: N/A
STATE AFTER: chunk.word_count == 0, chunk.content == "   \t\n   "
```

**Edge Case 3: Unicode Content (Multi-byte Characters)**
```
INPUT:  TextChunk::new("こんにちは".to_string(), metadata)
EXPECTED OUTPUT:
  - char_len() = 5 (5 Japanese characters)
  - byte_len() = 15 (3 bytes per character)
  - word_count = 1 (single word, no whitespace)
VERIFICATION: Run test_text_chunk_char_len_unicode
STATE BEFORE: N/A
STATE AFTER: chunk.byte_len() == 15, chunk.char_len() == 5
```

### Evidence of Success

After implementation, verify with:
```bash
# Run all chunker tests and save output
cargo test --package context-graph-core memory::chunker::tests:: 2>&1 | tee /tmp/task-p1-002-results.log

# Verify expected test count
grep -c "test.*ok" /tmp/task-p1-002-results.log
# EXPECTED: 9 (or more if additional tests added)

# Verify no failures
grep "FAILED" /tmp/task-p1-002-results.log
# EXPECTED: No output (no failures)

# Verify TextChunk is exported
cargo doc --package context-graph-core --no-deps 2>&1 | grep "pub struct TextChunk"
# EXPECTED: Documentation generated for TextChunk
```

### Physical Proof Verification

The Source of Truth for this task is the Rust type system and cargo toolchain:

1. **File System Verification**:
```bash
# Verify chunker.rs exists and has content
wc -l crates/context-graph-core/src/memory/chunker.rs
# EXPECTED: ~130 lines (approximately)

stat crates/context-graph-core/src/memory/chunker.rs
# EXPECTED: File exists with recent modification time
```

2. **Compilation Verification** (proves types are valid):
```bash
cargo build --package context-graph-core 2>&1 | tail -5
# EXPECTED: "Compiling context-graph-core" and "Finished"
```

3. **Test Output Verification** (proves logic works):
```bash
cargo test --package context-graph-core memory::chunker 2>&1 | grep -E "^test|passed"
# EXPECTED:
# test memory::chunker::tests::test_text_chunk_new_computes_word_count ... ok
# test memory::chunker::tests::test_text_chunk_empty_content ... ok
# ... (all 9 tests)
# test result: ok. 9 passed; 0 failed
```

## Anti-Patterns to Avoid

1. **DO NOT move ChunkMetadata** - It lives in mod.rs and works fine there
2. **DO NOT add Serialize/Deserialize to TextChunk** - It's transient, never stored
3. **DO NOT create Default impl for TextChunk** - Requires valid ChunkMetadata
4. **DO NOT use .unwrap()** - This is library code
5. **NO backwards compatibility shims** - Fail fast on errors

## Manual Testing Protocol

### Synthetic Test 1: Basic Creation
```rust
// Create TextChunk with known input
let meta = ChunkMetadata {
    file_path: "test.md".to_string(),
    chunk_index: 2,
    total_chunks: 5,
    word_offset: 400,
    char_offset: 2500,
    original_file_hash: "abc123def456".to_string(),
};
let chunk = TextChunk::new("This is synthetic test content".to_string(), meta);

// Verify outputs
assert_eq!(chunk.word_count, 5);           // 5 words
assert_eq!(chunk.metadata.chunk_index, 2);  // Preserved
assert_eq!(chunk.byte_len(), 31);           // 31 ASCII chars
assert!(!chunk.is_empty());                 // Not empty
```

### Synthetic Test 2: Large Content
```rust
// Create 200-word chunk (typical chunk size per constitution)
let words: Vec<&str> = (0..200).map(|_| "word").collect();
let content = words.join(" ");
let chunk = TextChunk::new(content.clone(), test_metadata());

assert_eq!(chunk.word_count, 200);
assert_eq!(chunk.byte_len(), 200 * 5 - 1); // 200 words * 4 chars + 199 spaces
```

## Execution Checklist

- [ ] Create `crates/context-graph-core/src/memory/chunker.rs` with TextChunk struct
- [ ] Add `pub mod chunker;` to `crates/context-graph-core/src/memory/mod.rs`
- [ ] Add `pub use chunker::TextChunk;` to memory/mod.rs exports
- [ ] Add `TextChunk` to lib.rs re-exports
- [ ] Run `cargo check --package context-graph-core` - MUST PASS
- [ ] Run `cargo test --package context-graph-core memory::chunker::tests::` - MUST PASS
- [ ] Run `cargo clippy --package context-graph-core -- -D warnings` - MUST PASS
- [ ] Save test output as evidence
- [ ] Verify all 3 edge cases pass
- [ ] Verify physical file existence
- [ ] Mark task COMPLETE
- [ ] Proceed to TASK-P1-003 (Session types) or TASK-P1-004 (TextChunker algorithm)

## Dependencies for Future Tasks

This task creates the foundation for:
- **TASK-P1-004** (TextChunker): Will use TextChunk as output type
- **TASK-P1-007** (MemoryCaptureService): Will convert TextChunk to Memory
- **TASK-P1-008** (MDFileWatcher): Will process files into TextChunks
