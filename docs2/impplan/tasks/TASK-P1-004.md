# TASK-P1-004: TextChunker Implementation

```xml
<task_spec id="TASK-P1-004" version="2.0">
<metadata>
  <title>TextChunker Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>9</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-03</requirement_ref>
    <requirement_ref>REQ-P1-04</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETE">TASK-P1-001</task_ref>
    <task_ref status="COMPLETE">TASK-P1-002</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <last_audit>2025-01-16</last_audit>
</metadata>

<context>
Implements the TextChunker component that splits long text into overlapping
chunks suitable for embedding. Uses 200-word chunks with 50-word overlap
and attempts to preserve sentence boundaries.

This is the core text processing component used for MD file ingestion.

## CODEBASE STATE (Audited 2025-01-16)

### Dependencies Status:
- TASK-P1-001 (Memory struct): **COMPLETE** - Memory struct in memory/mod.rs
- TASK-P1-002 (TextChunk type): **COMPLETE** - TextChunk struct in memory/chunker.rs
- TASK-P1-003 (Session types): **COMPLETE** - Session, SessionStatus in memory/session.rs

### Existing Code:
- `crates/context-graph-core/src/memory/mod.rs`:
  - Memory struct (lines 73-89)
  - MemorySource enum (HookDescription, ClaudeResponse, MDFileChunk)
  - ChunkMetadata struct (lines 52-66) - **ALREADY EXISTS**
  - Re-exports TextChunk from chunker.rs

- `crates/context-graph-core/src/memory/chunker.rs`:
  - TextChunk struct (lines 53-64) - **ALREADY EXISTS**
  - TextChunk::new() with auto word count
  - TextChunk::with_word_count() for testing
  - Comprehensive tests for TextChunk
  - **TextChunker component DOES NOT EXIST** - this is what we implement

### Missing Dependencies:
- **sha2 crate NOT in Cargo.toml** - MUST BE ADDED

### Key Imports Required:
```rust
// ChunkMetadata is in parent module, NOT chunker.rs
use super::ChunkMetadata;  // Already imported at top of chunker.rs
use sha2::{Sha256, Digest}; // REQUIRES adding sha2 to Cargo.toml
```
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md#component_contracts</file>
  <file purpose="types_textchunk">crates/context-graph-core/src/memory/chunker.rs</file>
  <file purpose="types_chunkmetadata">crates/context-graph-core/src/memory/mod.rs</file>
  <file purpose="constitution">docs2/constitution.yaml</file>
</input_context_files>

<prerequisites>
  <check status="VERIFIED">TASK-P1-001 complete (Memory struct exists in mod.rs)</check>
  <check status="VERIFIED">TASK-P1-002 complete (TextChunk exists in chunker.rs)</check>
  <check status="VERIFIED">ChunkMetadata exists in mod.rs with all 6 fields</check>
  <check status="ACTION_REQUIRED">sha2 crate MUST be added to Cargo.toml</check>
  <check status="VERIFIED">thiserror crate already in Cargo.toml</check>
</prerequisites>

<scope>
  <in_scope>
    - Add sha2 = "0.10" to Cargo.toml
    - Implement ChunkerError enum in chunker.rs
    - Implement TextChunker struct with configuration
    - Implement chunk_text method with sentence boundary detection
    - Implement find_sentence_boundary helper
    - Compute SHA256 hash of input content
    - Use existing ChunkMetadata from super module
    - Add unit tests with REAL data (no mocks)
  </in_scope>
  <out_of_scope>
    - File I/O (MDFileWatcher handles this)
    - Embedding (Phase 2)
    - Storage (TASK-P1-005)
    - Modifying existing TextChunk struct
    - Modifying existing ChunkMetadata struct
  </out_of_scope>
</scope>

<architectural_constraints>
  <!-- From constitution.yaml -->
  <constraint id="ARCH-01">TeleologicalArray is atomic - store all 13 embeddings or nothing</constraint>
  <constraint id="ARCH-11">Memory sources: HookDescription, ClaudeResponse, MDFileChunk</constraint>
  <constraint id="constitution.memory_sources.MDFileChunk.chunking">chunk_size: 200 words, overlap: 50 words</constraint>
  <constraint id="rust_standards.error_handling">thiserror for library, Never panic in lib, Propagate with ?</constraint>

  <!-- CRITICAL: NO BACKWARDS COMPATIBILITY -->
  <constraint id="FAIL_FAST">System MUST work correctly or fail immediately with descriptive errors</constraint>
  <constraint id="NO_SILENT_FAILURES">All errors MUST be logged with context before propagation</constraint>
</architectural_constraints>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/memory/chunker.rs">
      // Error type for chunking operations
      #[derive(Debug, Error)]
      pub enum ChunkerError {
          #[error("Content is empty or contains only whitespace")]
          EmptyContent,
          #[error("Configuration error: chunk_size ({chunk_size}) must be > overlap ({overlap})")]
          InvalidOverlap { chunk_size: usize, overlap: usize },
          #[error("Configuration error: chunk_size ({chunk_size}) must be >= MIN_CHUNK_WORDS ({min})")]
          ChunkSizeTooSmall { chunk_size: usize, min: usize },
          #[error("Hash computation failed: {reason}")]
          HashError { reason: String },
      }

      pub struct TextChunker {
          chunk_size_words: usize,
          overlap_words: usize,
      }

      impl TextChunker {
          pub const CHUNK_SIZE_WORDS: usize = 200;
          pub const OVERLAP_WORDS: usize = 50;
          pub const MIN_CHUNK_WORDS: usize = 50;

          pub fn new(chunk_size: usize, overlap: usize) -> Result&lt;Self, ChunkerError&gt;;
          pub fn default_config() -> Self;
          pub fn chunk_text(&amp;self, content: &amp;str, file_path: &amp;str) -> Result&lt;Vec&lt;TextChunk&gt;, ChunkerError&gt;;
          fn find_sentence_boundary(&amp;self, words: &amp;[&amp;str], start: usize, end: usize) -> usize;
          fn compute_hash(content: &amp;str) -> String;
          fn estimate_chunk_count(&amp;self, total_words: usize) -> u32;
      }
    </signature>
  </signatures>

  <constraints>
    - Default chunk_size = 200 words, overlap = 50 words (per constitution)
    - chunk_size must be > overlap (fail fast with ChunkerError::InvalidOverlap)
    - chunk_size must be >= MIN_CHUNK_WORDS (50) (fail fast with ChunkerError::ChunkSizeTooSmall)
    - Empty/whitespace-only content returns ChunkerError::EmptyContent
    - Sentence boundaries: '.', '!', '?'
    - Adjust to sentence boundary only within 20% of target end
    - SHA256 hash computed with sha2 crate
    - All errors must include context (actual values in error messages)
  </constraints>

  <verification>
    - Unit tests pass with REAL text data (not mock data)
    - Sentence boundary detection works correctly
    - Overlap is correct between consecutive chunks
    - Hash is deterministic (same content = same hash)
    - Error messages include actual parameter values
  </verification>
</definition_of_done>

<source_of_truth>
  <!-- Defines what constitutes correct behavior -->
  <truth id="SOT-1" description="Chunk count">
    For content with N words where N > chunk_size:
    chunks = ceil((N - overlap) / (chunk_size - overlap))
  </truth>
  <truth id="SOT-2" description="Overlap verification">
    For consecutive chunks i and i+1:
    last_overlap_words(chunk[i]) == first_overlap_words(chunk[i+1])
  </truth>
  <truth id="SOT-3" description="Hash determinism">
    compute_hash(content) always returns same value for identical content
  </truth>
  <truth id="SOT-4" description="Word offset accuracy">
    chunk[i].metadata.word_offset == sum of effective words in chunks 0..i
  </truth>
</source_of_truth>

<full_state_verification>
  <protocol>
    After each chunk_text() call:
    1. EXECUTE: Call chunk_text with known input
    2. INSPECT: Verify returned Vec&lt;TextChunk&gt; against Source of Truth
    3. VALIDATE: Each chunk's metadata matches expected values
    4. EVIDENCE: Log actual chunk boundaries and metadata
  </protocol>

  <boundary_edge_cases>
    <case id="EDGE-1" description="Exactly chunk_size words">
      Input: Exactly 200 words
      Expected: Single chunk with word_offset=0, chunk_index=0, total_chunks=1
      Verify: chunks.len() == 1, chunks[0].word_count == 200
    </case>
    <case id="EDGE-2" description="chunk_size + 1 words">
      Input: 201 words
      Expected: 2 chunks with overlap
      Verify: chunks.len() == 2, overlap words match between chunks
    </case>
    <case id="EDGE-3" description="Sentence boundary at exact 20% mark">
      Input: 200 words with '.' at word 160 (exactly 20% from end)
      Expected: Chunk ends at word 160 (includes sentence terminator)
      Verify: chunk boundary aligned to sentence
    </case>
    <case id="EDGE-4" description="No sentence boundary in search range">
      Input: 200 words with no '.', '!', '?' in last 40 words
      Expected: Chunk ends at word 200 (no adjustment)
      Verify: chunk uses original end position
    </case>
    <case id="EDGE-5" description="Unicode content">
      Input: Mix of ASCII and Unicode (e.g., "hello 世界 rust 言語")
      Expected: Word count based on whitespace split, not bytes
      Verify: word_count correct, char_offset tracks actual characters
    </case>
  </boundary_edge_cases>
</full_state_verification>

<evidence_of_success>
  After implementation, run these verification steps:

  ```rust
  // Verification test that MUST pass
  #[test]
  fn verify_chunk_text_evidence() {
      let chunker = TextChunker::default_config();

      // Create 450-word content (should produce ~3 chunks)
      let words: Vec&lt;String&gt; = (0..450).map(|i| {
          if i % 50 == 49 { format!("word{}.", i) }  // Add sentence boundaries
          else { format!("word{}", i) }
      }).collect();
      let content = words.join(" ");

      let chunks = chunker.chunk_text(&amp;content, "evidence_test.md").unwrap();

      // EVIDENCE: Print actual state
      println!("=== EVIDENCE OF SUCCESS ===");
      println!("Total chunks: {}", chunks.len());
      for (i, chunk) in chunks.iter().enumerate() {
          println!("Chunk {}: word_count={}, word_offset={}, first_word={:?}, last_word={:?}",
              i,
              chunk.word_count,
              chunk.metadata.word_offset,
              chunk.content.split_whitespace().next(),
              chunk.content.split_whitespace().last()
          );
      }

      // Verify overlap between consecutive chunks
      for i in 0..chunks.len()-1 {
          let current_last_words: Vec&lt;&amp;str&gt; = chunks[i].content
              .split_whitespace()
              .rev()
              .take(50)
              .collect();
          let next_first_words: Vec&lt;&amp;str&gt; = chunks[i+1].content
              .split_whitespace()
              .take(50)
              .collect();

          println!("Overlap {}->{}: last_50={:?}, first_50={:?}",
              i, i+1,
              current_last_words.iter().rev().take(5).collect::&lt;Vec&lt;_&gt;&gt;(),
              next_first_words.iter().take(5).collect::&lt;Vec&lt;_&gt;&gt;()
          );
      }
      println!("=== END EVIDENCE ===");

      // Assertions
      assert!(chunks.len() >= 2, "Should produce multiple chunks");
      assert_eq!(chunks[0].metadata.chunk_index, 0);
      assert_eq!(chunks.last().unwrap().metadata.chunk_index as usize, chunks.len() - 1);
  }
  ```
</evidence_of_success>

<pseudo_code>
// File: crates/context-graph-core/src/memory/chunker.rs
// ADD to existing file (which already has TextChunk)

use sha2::{Sha256, Digest};
use thiserror::Error;

// Import ChunkMetadata from parent module (already imported via `use super::ChunkMetadata;`)

/// Errors that can occur during text chunking operations.
///
/// All errors include context values for debugging.
/// Per constitution: "Never panic in lib, Propagate with ?"
#[derive(Debug, Error)]
pub enum ChunkerError {
    #[error("Content is empty or contains only whitespace")]
    EmptyContent,

    #[error("Configuration error: chunk_size ({chunk_size}) must be > overlap ({overlap})")]
    InvalidOverlap { chunk_size: usize, overlap: usize },

    #[error("Configuration error: chunk_size ({chunk_size}) must be >= MIN_CHUNK_WORDS ({min})")]
    ChunkSizeTooSmall { chunk_size: usize, min: usize },
}

/// Text chunker that splits content into overlapping chunks.
///
/// Per constitution.yaml memory_sources.MDFileChunk.chunking:
/// - chunk_size: 200 words
/// - overlap: 50 words (25%)
/// - boundary: "Preserve sentence boundaries when possible"
///
/// # Example
/// ```rust
/// use context_graph_core::memory::{TextChunker, TextChunk};
///
/// let chunker = TextChunker::default_config();
/// let chunks = chunker.chunk_text("Long document content...", "docs/readme.md")?;
/// ```
pub struct TextChunker {
    chunk_size_words: usize,
    overlap_words: usize,
}

impl TextChunker {
    /// Default chunk size per constitution: 200 words
    pub const CHUNK_SIZE_WORDS: usize = 200;
    /// Default overlap per constitution: 50 words (25%)
    pub const OVERLAP_WORDS: usize = 50;
    /// Minimum chunk size to ensure meaningful content
    pub const MIN_CHUNK_WORDS: usize = 50;
    /// Sentence terminator characters for boundary detection
    const SENTENCE_TERMINATORS: [char; 3] = ['.', '!', '?'];

    /// Create a new TextChunker with custom configuration.
    ///
    /// # Errors
    /// - `ChunkerError::InvalidOverlap` if chunk_size <= overlap
    /// - `ChunkerError::ChunkSizeTooSmall` if chunk_size < MIN_CHUNK_WORDS
    ///
    /// # Fail-Fast Behavior
    /// Invalid configuration fails immediately with descriptive error.
    pub fn new(chunk_size: usize, overlap: usize) -> Result&lt;Self, ChunkerError&gt; {
        if chunk_size &lt;= overlap {
            return Err(ChunkerError::InvalidOverlap {
                chunk_size,
                overlap
            });
        }
        if chunk_size &lt; Self::MIN_CHUNK_WORDS {
            return Err(ChunkerError::ChunkSizeTooSmall {
                chunk_size,
                min: Self::MIN_CHUNK_WORDS
            });
        }
        Ok(Self { chunk_size_words: chunk_size, overlap_words: overlap })
    }

    /// Create TextChunker with default constitution values.
    ///
    /// Uses CHUNK_SIZE_WORDS=200, OVERLAP_WORDS=50 per constitution.
    pub fn default_config() -> Self {
        Self {
            chunk_size_words: Self::CHUNK_SIZE_WORDS,
            overlap_words: Self::OVERLAP_WORDS,
        }
    }

    /// Chunk text content into overlapping TextChunk instances.
    ///
    /// # Arguments
    /// * `content` - The text to chunk
    /// * `file_path` - Source file path for metadata
    ///
    /// # Returns
    /// Vec of TextChunk with proper metadata including:
    /// - chunk_index (0-based)
    /// - total_chunks
    /// - word_offset (cumulative)
    /// - char_offset (cumulative)
    /// - original_file_hash (SHA256)
    ///
    /// # Errors
    /// - `ChunkerError::EmptyContent` if content is empty or whitespace-only
    pub fn chunk_text(&amp;self, content: &amp;str, file_path: &amp;str) -> Result&lt;Vec&lt;TextChunk&gt;, ChunkerError&gt; {
        // Fail fast on empty content
        if content.is_empty() || content.trim().is_empty() {
            return Err(ChunkerError::EmptyContent);
        }

        let words: Vec&lt;&amp;str&gt; = content.split_whitespace().collect();
        if words.is_empty() {
            return Err(ChunkerError::EmptyContent);
        }

        // Compute deterministic SHA256 hash
        let hash = Self::compute_hash(content);

        // Single chunk case
        if words.len() &lt;= self.chunk_size_words {
            let metadata = ChunkMetadata {
                file_path: file_path.to_string(),
                chunk_index: 0,
                total_chunks: 1,
                word_offset: 0,
                char_offset: 0,
                original_file_hash: hash,
            };
            return Ok(vec![TextChunk::new(content.to_string(), metadata)]);
        }

        let mut chunks = Vec::new();
        let mut word_offset: u32 = 0;
        let mut char_offset: u32 = 0;
        let mut chunk_index: u32 = 0;

        // Estimate total chunks for metadata
        let total_chunks = self.estimate_chunk_count(words.len());

        let mut current_word_idx = 0;

        while current_word_idx &lt; words.len() {
            let end_word_idx = std::cmp::min(
                current_word_idx + self.chunk_size_words,
                words.len()
            );

            // Find sentence boundary (only adjusts within last 20%)
            let adjusted_end = self.find_sentence_boundary(&amp;words, current_word_idx, end_word_idx);

            // Build chunk content
            let chunk_words = &amp;words[current_word_idx..adjusted_end];
            let chunk_content = chunk_words.join(" ");

            let metadata = ChunkMetadata {
                file_path: file_path.to_string(),
                chunk_index,
                total_chunks,
                word_offset,
                char_offset,
                original_file_hash: hash.clone(),
            };

            chunks.push(TextChunk::new(chunk_content.clone(), metadata));

            // Update offsets for next chunk
            let words_in_chunk = (adjusted_end - current_word_idx) as u32;
            char_offset += chunk_content.len() as u32 + 1; // +1 for implicit space
            chunk_index += 1;

            // Move to next chunk start (with overlap)
            if adjusted_end >= words.len() {
                break;
            }

            // Next chunk starts overlap_words before this chunk ended
            let effective_advance = adjusted_end.saturating_sub(current_word_idx)
                .saturating_sub(self.overlap_words);
            current_word_idx += std::cmp::max(effective_advance, 1);
            word_offset += effective_advance as u32;
        }

        Ok(chunks)
    }

    /// Find sentence boundary within search window.
    ///
    /// Searches backwards from `end` within last 20% of chunk.
    /// Returns adjusted end position if boundary found, otherwise original end.
    fn find_sentence_boundary(&amp;self, words: &amp;[&amp;str], start: usize, end: usize) -> usize {
        // Search window: last 20% of chunk
        let search_window = self.chunk_size_words / 5;
        let search_start = end.saturating_sub(search_window).max(start);

        // Search backwards for sentence terminator
        for i in (search_start..end).rev() {
            if let Some(word) = words.get(i) {
                if Self::SENTENCE_TERMINATORS.iter().any(|&amp;t| word.ends_with(t)) {
                    return i + 1; // Include word with terminator
                }
            }
        }

        // No boundary found, use original end
        end
    }

    /// Compute SHA256 hash of content.
    fn compute_hash(content: &amp;str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Estimate total chunk count for metadata.
    fn estimate_chunk_count(&amp;self, total_words: usize) -> u32 {
        if total_words &lt;= self.chunk_size_words {
            return 1;
        }
        let effective_step = self.chunk_size_words - self.overlap_words;
        let remaining = total_words.saturating_sub(self.chunk_size_words);
        (1 + (remaining + effective_step - 1) / effective_step) as u32
    }
}
</pseudo_code>

<files_to_modify>
  <file path="crates/context-graph-core/Cargo.toml" action="ADD_DEPENDENCY">
    Add: sha2 = "0.10"
  </file>
  <file path="crates/context-graph-core/src/memory/chunker.rs" action="APPEND">
    Add ChunkerError enum and TextChunker implementation after existing TextChunk code
  </file>
  <file path="crates/context-graph-core/src/memory/mod.rs" action="UPDATE_EXPORTS">
    Add: ChunkerError, TextChunker to public exports
  </file>
</files_to_modify>

<validation_criteria>
  <criterion id="VC-1">TextChunker::new rejects chunk_size &lt;= overlap with ChunkerError::InvalidOverlap</criterion>
  <criterion id="VC-2">TextChunker::new rejects chunk_size &lt; 50 with ChunkerError::ChunkSizeTooSmall</criterion>
  <criterion id="VC-3">chunk_text returns ChunkerError::EmptyContent for empty/whitespace content</criterion>
  <criterion id="VC-4">Single chunk returned for content &lt;= 200 words</criterion>
  <criterion id="VC-5">Multiple chunks have correct 50-word overlap</criterion>
  <criterion id="VC-6">Sentence boundary detection adjusts within 20% of chunk end</criterion>
  <criterion id="VC-7">SHA256 hash is deterministic (same content = same hash)</criterion>
  <criterion id="VC-8">All metadata fields populated correctly (chunk_index, total_chunks, word_offset, char_offset)</criterion>
  <criterion id="VC-9">Unicode content handled correctly (word count by whitespace, not bytes)</criterion>
  <criterion id="VC-10">All tests use REAL data, no mocks</criterion>
</validation_criteria>

<test_commands>
  <command description="Run chunker tests">cargo test --package context-graph-core chunker -- --nocapture</command>
  <command description="Run with evidence output">cargo test --package context-graph-core verify_chunk_text_evidence -- --nocapture</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
  <command description="Run all memory tests">cargo test --package context-graph-core memory -- --nocapture</command>
</test_commands>

<manual_verification_protocol>
  After implementation, manually verify:

  1. **Cargo.toml**: Confirm sha2 = "0.10" is present
  2. **chunker.rs**: Confirm TextChunker struct exists with all methods
  3. **mod.rs exports**: Confirm ChunkerError and TextChunker are exported
  4. **Test output**: Run tests with --nocapture and verify EVIDENCE output shows:
     - Correct chunk counts
     - Overlapping words between chunks
     - Sentence boundary alignment where applicable
  5. **Error messages**: Trigger each error type and verify context values appear
</manual_verification_protocol>
</task_spec>
```

## Execution Checklist

### Pre-Implementation Verification
- [x] TASK-P1-001 complete (Memory struct exists)
- [x] TASK-P1-002 complete (TextChunk exists)
- [x] TASK-P1-003 complete (Session types exist)
- [x] ChunkMetadata in mod.rs verified
- [ ] sha2 crate added to Cargo.toml

### Implementation Steps
- [ ] Add sha2 = "0.10" to Cargo.toml `[dependencies]`
- [ ] Implement ChunkerError enum in chunker.rs
- [ ] Implement TextChunker struct
- [ ] Implement TextChunker::new() with fail-fast validation
- [ ] Implement TextChunker::default_config()
- [ ] Implement TextChunker::chunk_text() with algorithm
- [ ] Implement TextChunker::find_sentence_boundary()
- [ ] Implement TextChunker::compute_hash()
- [ ] Implement TextChunker::estimate_chunk_count()
- [ ] Update mod.rs to export ChunkerError, TextChunker

### Testing (REAL DATA ONLY - NO MOCKS)
- [ ] Write test_chunker_new_validates_config
- [ ] Write test_chunker_empty_content_error
- [ ] Write test_chunker_single_chunk
- [ ] Write test_chunker_multiple_chunks_with_overlap
- [ ] Write test_chunker_sentence_boundary_detection
- [ ] Write test_chunker_hash_determinism
- [ ] Write test_chunker_unicode_content
- [ ] Write verify_chunk_text_evidence (full evidence test)

### Edge Case Tests
- [ ] EDGE-1: Exactly 200 words
- [ ] EDGE-2: 201 words (boundary)
- [ ] EDGE-3: Sentence at exact 20% mark
- [ ] EDGE-4: No sentence boundary in range
- [ ] EDGE-5: Unicode mixed content

### Post-Implementation Verification
- [ ] Run `cargo test --package context-graph-core chunker -- --nocapture`
- [ ] Verify EVIDENCE output in test logs
- [ ] Run `cargo clippy --package context-graph-core`
- [ ] Verify error messages include context values
- [ ] Manual review of chunk boundaries

### Completion
- [ ] All tests pass
- [ ] Evidence of success logged
- [ ] Ready for TASK-P1-005

## Synthetic Test Data

Use these specific test cases (REAL DATA):

```rust
// Test Case 1: Short content (single chunk)
const SHORT_CONTENT: &str = "The quick brown fox jumps over the lazy dog. This is a test.";

// Test Case 2: Exactly 200 words
fn generate_200_words() -> String {
    (0..200).map(|i| format!("word{}", i)).collect::<Vec<_>>().join(" ")
}

// Test Case 3: 450 words with sentence boundaries
fn generate_450_words_with_sentences() -> String {
    (0..450).map(|i| {
        if i % 50 == 49 { format!("sentence{}.", i) }
        else { format!("word{}", i) }
    }).collect::<Vec<_>>().join(" ")
}

// Test Case 4: Unicode content
const UNICODE_CONTENT: &str = "Hello world. 你好世界 this is a test. Rust 言語 programming.";

// Test Case 5: No sentence boundaries
fn generate_no_sentences(count: usize) -> String {
    (0..count).map(|i| format!("word{}", i)).collect::<Vec<_>>().join(" ")
}
```
