# TASK-P1-001: Memory Struct and MemorySource Enum

```xml
<task_spec id="TASK-P1-001" version="2.0">
<metadata>
  <title>Memory Struct and MemorySource Enum</title>
  <status>COMPLETE</status>
  <layer>foundation</layer>
  <sequence>6</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-01</requirement_ref>
    <requirement_ref>REQ-P1-02</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETE">TASK-P0-005</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
  <last_audit>2026-01-16</last_audit>
</metadata>
```

## Critical Codebase Context (READ FIRST)

### What Already Exists (DO NOT RECREATE)

1. **`TeleologicalArray` type alias** - Already defined at:
   - Path: `crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs:26`
   - Definition: `pub type TeleologicalArray = SemanticFingerprint;`
   - Re-exported from: `crates/context-graph-core/src/types/fingerprint/mod.rs`

2. **`SemanticFingerprint` struct** - The actual 13-embedding storage:
   - Path: `crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs`
   - Contains all 13 embeddings: E1-E13
   - Has `#[derive(Debug, Clone, Serialize, Deserialize)]`
   - Has `validate_strict()` method for validation

3. **`MemoryNode` struct** - Different purpose (legacy graph node):
   - Path: `crates/context-graph-core/src/types/memory_node/node.rs`
   - Uses single 1536D embedding (NOT 13-space TeleologicalArray)
   - DO NOT confuse with the Memory struct we need to create

4. **`NodeMetadata` struct** - Reusable metadata container:
   - Path: `crates/context-graph-core/src/types/memory_node/metadata.rs`
   - Has `source: Option<String>` field - BUT this is generic, not typed
   - We need typed `MemorySource` enum to discriminate source types

5. **No `memory/` module exists** in context-graph-core/src/
   - Glob `crates/context-graph-core/src/memory/**/*.rs` returns NO FILES
   - This module must be CREATED

### Dependencies Already in Cargo.toml

```toml
# crates/context-graph-core/Cargo.toml already has:
uuid.workspace = true        # Uuid type available
chrono.workspace = true      # DateTime<Utc> available
serde.workspace = true       # Serialize, Deserialize derives
bincode.workspace = true     # Binary serialization
thiserror.workspace = true   # Error derives
```

NO additional dependencies needed.

## Task Objective

Create a new `memory` module in `context-graph-core` containing:
1. `Memory` struct - Primary data unit for captured memories
2. `MemorySource` enum - Discriminated union for source types (HookDescription, ClaudeResponse, MDFileChunk)
3. `HookType` enum - Specific hook event types
4. `ResponseType` enum - Specific response types

## Detailed Implementation

### File Structure to Create

```
crates/context-graph-core/src/memory/
├── mod.rs           # Memory struct + module re-exports
└── source.rs        # MemorySource, HookType, ResponseType enums
```

### File 1: `crates/context-graph-core/src/memory/source.rs`

```rust
//! Memory source types for discriminating origin of captured memories.
//!
//! # Constitution Compliance
//! - ARCH-11: Memory sources: HookDescription, ClaudeResponse, MDFileChunk
//! - Hook types per .claude/settings.json native hook architecture

use serde::{Deserialize, Serialize};

/// Discriminated source type for Memory origin.
///
/// Per constitution.yaml ARCH-11 and memory_sources section:
/// - HookDescription: From Claude Code hook events
/// - ClaudeResponse: From session end/stop captured responses
/// - MDFileChunk: From markdown file watcher chunks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemorySource {
    /// Memory captured from a Claude Code hook event.
    HookDescription {
        /// The type of hook that triggered capture.
        hook_type: HookType,
        /// Tool name if applicable (PreToolUse, PostToolUse).
        tool_name: Option<String>,
    },
    /// Memory captured from Claude's response.
    ClaudeResponse {
        /// The type of response captured.
        response_type: ResponseType,
    },
    /// Memory captured from markdown file chunking.
    MDFileChunk {
        /// Path to the source markdown file.
        file_path: String,
        /// Zero-based index of this chunk.
        chunk_index: u32,
        /// Total number of chunks from the file.
        total_chunks: u32,
    },
}

/// Hook event types matching .claude/settings.json native hooks.
///
/// Per constitution.yaml claude_code.hooks section:
/// - SessionStart: Session initialization
/// - UserPromptSubmit: User sends a prompt
/// - PreToolUse: Before tool execution (Edit, Write, Bash)
/// - PostToolUse: After any tool execution
/// - Stop: Claude stops responding
/// - SessionEnd: Session cleanup
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HookType {
    /// Session initialization hook (timeout: 5000ms).
    SessionStart,
    /// User prompt submission hook (timeout: 2000ms).
    UserPromptSubmit,
    /// Pre-tool-use hook for Edit|Write|Bash (timeout: 500ms).
    PreToolUse,
    /// Post-tool-use hook for all tools (timeout: 3000ms, async).
    PostToolUse,
    /// Stop hook when Claude stops (timeout: 3000ms).
    Stop,
    /// Session end hook (timeout: 30000ms).
    SessionEnd,
}

/// Response types for ClaudeResponse memory source.
///
/// Distinguishes the context in which Claude's response was captured.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ResponseType {
    /// Summary captured at session end.
    SessionSummary,
    /// Response captured at Stop hook.
    StopResponse,
    /// Significant response worth persisting.
    SignificantResponse,
}

impl std::fmt::Display for HookType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HookType::SessionStart => write!(f, "SessionStart"),
            HookType::UserPromptSubmit => write!(f, "UserPromptSubmit"),
            HookType::PreToolUse => write!(f, "PreToolUse"),
            HookType::PostToolUse => write!(f, "PostToolUse"),
            HookType::Stop => write!(f, "Stop"),
            HookType::SessionEnd => write!(f, "SessionEnd"),
        }
    }
}

impl std::fmt::Display for ResponseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResponseType::SessionSummary => write!(f, "SessionSummary"),
            ResponseType::StopResponse => write!(f, "StopResponse"),
            ResponseType::SignificantResponse => write!(f, "SignificantResponse"),
        }
    }
}

impl std::fmt::Display for MemorySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemorySource::HookDescription { hook_type, tool_name } => {
                if let Some(tool) = tool_name {
                    write!(f, "HookDescription({}, tool={})", hook_type, tool)
                } else {
                    write!(f, "HookDescription({})", hook_type)
                }
            }
            MemorySource::ClaudeResponse { response_type } => {
                write!(f, "ClaudeResponse({})", response_type)
            }
            MemorySource::MDFileChunk { file_path, chunk_index, total_chunks } => {
                write!(f, "MDFileChunk({}, {}/{})", file_path, chunk_index + 1, total_chunks)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_type_variants() {
        // Verify all 6 variants exist per constitution
        let types = [
            HookType::SessionStart,
            HookType::UserPromptSubmit,
            HookType::PreToolUse,
            HookType::PostToolUse,
            HookType::Stop,
            HookType::SessionEnd,
        ];
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn test_response_type_variants() {
        // Verify all 3 variants exist
        let types = [
            ResponseType::SessionSummary,
            ResponseType::StopResponse,
            ResponseType::SignificantResponse,
        ];
        assert_eq!(types.len(), 3);
    }

    #[test]
    fn test_memory_source_variants() {
        // Verify all 3 variants exist per ARCH-11
        let sources = [
            MemorySource::HookDescription {
                hook_type: HookType::SessionStart,
                tool_name: None,
            },
            MemorySource::ClaudeResponse {
                response_type: ResponseType::SessionSummary,
            },
            MemorySource::MDFileChunk {
                file_path: "test.md".to_string(),
                chunk_index: 0,
                total_chunks: 1,
            },
        ];
        assert_eq!(sources.len(), 3);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let source = MemorySource::HookDescription {
            hook_type: HookType::PostToolUse,
            tool_name: Some("Edit".to_string()),
        };

        let serialized = serde_json::to_string(&source).expect("serialize failed");
        let deserialized: MemorySource = serde_json::from_str(&serialized).expect("deserialize failed");

        assert_eq!(source, deserialized);
    }

    #[test]
    fn test_bincode_serialization() {
        let source = MemorySource::MDFileChunk {
            file_path: "/path/to/file.md".to_string(),
            chunk_index: 5,
            total_chunks: 10,
        };

        let bytes = bincode::serialize(&source).expect("bincode serialize failed");
        let restored: MemorySource = bincode::deserialize(&bytes).expect("bincode deserialize failed");

        assert_eq!(source, restored);
    }

    #[test]
    fn test_display_implementations() {
        assert_eq!(HookType::SessionStart.to_string(), "SessionStart");
        assert_eq!(ResponseType::StopResponse.to_string(), "StopResponse");

        let source = MemorySource::HookDescription {
            hook_type: HookType::PreToolUse,
            tool_name: Some("Bash".to_string()),
        };
        assert!(source.to_string().contains("PreToolUse"));
        assert!(source.to_string().contains("Bash"));
    }
}
```

### File 2: `crates/context-graph-core/src/memory/mod.rs`

```rust
//! Memory capture types for the Context Graph system.
//!
//! This module provides the core data types for memory capture:
//! - [`Memory`] - Primary memory unit with 13-embedding TeleologicalArray
//! - [`MemorySource`] - Discriminated source type (Hook, Response, MDChunk)
//! - [`HookType`] - Hook event types per .claude/settings.json
//! - [`ResponseType`] - Claude response capture types
//!
//! # Constitution Compliance
//! - ARCH-01: TeleologicalArray is atomic (all 13 embeddings or nothing)
//! - ARCH-05: All 13 embedders required
//! - ARCH-11: Memory sources: HookDescription, ClaudeResponse, MDFileChunk
//!
//! # Example
//! ```ignore
//! use context_graph_core::memory::{Memory, MemorySource, HookType};
//! use context_graph_core::types::fingerprint::TeleologicalArray;
//!
//! // TeleologicalArray must come from real embedding pipeline
//! let teleological_array: TeleologicalArray = embed_pipeline.embed_all(&content).await?;
//!
//! let memory = Memory::new(
//!     "Claude edited the config file".to_string(),
//!     MemorySource::HookDescription {
//!         hook_type: HookType::PostToolUse,
//!         tool_name: Some("Edit".to_string()),
//!     },
//!     "session-123".to_string(),
//!     teleological_array,
//!     None,
//! );
//! ```

pub mod source;

pub use source::{HookType, MemorySource, ResponseType};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Import TeleologicalArray from existing location
use crate::types::fingerprint::TeleologicalArray;

/// Placeholder for ChunkMetadata until TASK-P1-002 implements it.
///
/// This will be replaced by the full ChunkMetadata struct in TASK-P1-002.
/// For now, stores minimal chunk information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkMetadata {
    /// Path to the source file.
    pub file_path: String,
    /// Zero-based chunk index.
    pub chunk_index: u32,
    /// Total chunks from source file.
    pub total_chunks: u32,
    /// Word offset from start of file.
    pub word_offset: u32,
    /// Character offset from start of file.
    pub char_offset: u32,
    /// SHA256 hash of original file content.
    pub original_file_hash: String,
}

/// Memory: The primary data unit for captured memories.
///
/// Each Memory contains:
/// - Unique identifier (UUID)
/// - Content text
/// - Discriminated source type (MemorySource)
/// - Full 13-embedding TeleologicalArray
/// - Session association
/// - Optional chunk metadata for MDFileChunk sources
///
/// # Constitution Compliance
/// - ARCH-01: TeleologicalArray is atomic storage unit
/// - ARCH-05: All 13 embedders required - TeleologicalArray enforces this
/// - ARCH-11: Three source types: HookDescription, ClaudeResponse, MDFileChunk
///
/// # Storage
/// Typical size: ~46KB (TeleologicalArray) + content + metadata
/// Per constitution.yaml embeddings.paradigm: "NO FUSION - Store all 13 embeddings"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// Unique identifier (UUID v4).
    pub id: Uuid,

    /// The actual content/knowledge being stored.
    /// Max 10,000 characters per TECH-PHASE1 spec.
    pub content: String,

    /// Discriminated source type indicating memory origin.
    pub source: MemorySource,

    /// Timestamp when this memory was created.
    pub created_at: DateTime<Utc>,

    /// Session identifier this memory belongs to.
    pub session_id: String,

    /// Full 13-embedding array (ARCH-01: atomic storage unit).
    /// MUST contain valid embeddings for all 13 spaces.
    pub teleological_array: TeleologicalArray,

    /// Optional chunk metadata for MDFileChunk sources.
    pub chunk_metadata: Option<ChunkMetadata>,

    /// Word count of content (for chunking/stats).
    pub word_count: u32,
}

impl Memory {
    /// Create a new Memory with generated UUID and current timestamp.
    ///
    /// # Arguments
    /// * `content` - The text content to store
    /// * `source` - Discriminated source type
    /// * `session_id` - Session this memory belongs to
    /// * `teleological_array` - Full 13-embedding array (MUST be valid)
    /// * `chunk_metadata` - Optional chunk info for MDFileChunk sources
    ///
    /// # Panics
    /// Does NOT panic - validation should be done by caller via `validate()`.
    pub fn new(
        content: String,
        source: MemorySource,
        session_id: String,
        teleological_array: TeleologicalArray,
        chunk_metadata: Option<ChunkMetadata>,
    ) -> Self {
        let word_count = content.split_whitespace().count() as u32;

        Self {
            id: Uuid::new_v4(),
            content,
            source,
            created_at: Utc::now(),
            session_id,
            teleological_array,
            chunk_metadata,
            word_count,
        }
    }

    /// Create Memory with a specific UUID (for testing/reconstruction).
    pub fn with_id(
        id: Uuid,
        content: String,
        source: MemorySource,
        session_id: String,
        teleological_array: TeleologicalArray,
        chunk_metadata: Option<ChunkMetadata>,
    ) -> Self {
        let word_count = content.split_whitespace().count() as u32;

        Self {
            id,
            content,
            source,
            created_at: Utc::now(),
            session_id,
            teleological_array,
            chunk_metadata,
            word_count,
        }
    }

    /// Validate the Memory struct.
    ///
    /// # Checks
    /// 1. Content is not empty
    /// 2. Content <= 10,000 characters
    /// 3. Session ID is not empty
    /// 4. TeleologicalArray passes strict validation (all 13 embeddings valid)
    /// 5. ChunkMetadata consistency with source type
    ///
    /// # Returns
    /// `Ok(())` if valid, `Err(String)` with description on failure.
    pub fn validate(&self) -> Result<(), String> {
        // 1. Content not empty
        if self.content.is_empty() {
            return Err("Memory content cannot be empty".to_string());
        }

        // 2. Content length check
        if self.content.len() > 10_000 {
            return Err(format!(
                "Memory content exceeds 10,000 chars: {} chars",
                self.content.len()
            ));
        }

        // 3. Session ID not empty
        if self.session_id.is_empty() {
            return Err("Session ID cannot be empty".to_string());
        }

        // 4. TeleologicalArray validation
        if let Err(e) = self.teleological_array.validate_strict() {
            return Err(format!("TeleologicalArray validation failed: {}", e));
        }

        // 5. ChunkMetadata consistency
        match &self.source {
            MemorySource::MDFileChunk { .. } => {
                if self.chunk_metadata.is_none() {
                    return Err("MDFileChunk source requires chunk_metadata".to_string());
                }
            }
            _ => {
                // chunk_metadata optional for other sources
            }
        }

        Ok(())
    }

    /// Check if this memory is from a hook event.
    pub fn is_hook_description(&self) -> bool {
        matches!(self.source, MemorySource::HookDescription { .. })
    }

    /// Check if this memory is from a Claude response.
    pub fn is_claude_response(&self) -> bool {
        matches!(self.source, MemorySource::ClaudeResponse { .. })
    }

    /// Check if this memory is from an MD file chunk.
    pub fn is_md_file_chunk(&self) -> bool {
        matches!(self.source, MemorySource::MDFileChunk { .. })
    }

    /// Get the hook type if this is a HookDescription source.
    pub fn hook_type(&self) -> Option<HookType> {
        match &self.source {
            MemorySource::HookDescription { hook_type, .. } => Some(*hook_type),
            _ => None,
        }
    }

    /// Estimate memory size in bytes.
    pub fn estimated_size(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let content_size = self.content.len();
        let teleological_size = self.teleological_array.storage_size();
        let chunk_meta_size = self.chunk_metadata.as_ref().map_or(0, |m| {
            m.file_path.len() + m.original_file_hash.len() + 16
        });

        base + content_size + teleological_size + chunk_meta_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::fingerprint::SemanticFingerprint;

    // Helper to create valid test fingerprint (TEST ONLY)
    #[cfg(feature = "test-utils")]
    fn test_fingerprint() -> TeleologicalArray {
        SemanticFingerprint::zeroed()
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn test_memory_new_generates_uuid() {
        let fp = test_fingerprint();
        let memory = Memory::new(
            "test content".to_string(),
            MemorySource::HookDescription {
                hook_type: HookType::SessionStart,
                tool_name: None,
            },
            "session-1".to_string(),
            fp,
            None,
        );

        assert!(!memory.id.is_nil());
        assert_eq!(memory.content, "test content");
        assert_eq!(memory.word_count, 2);
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn test_memory_source_detection() {
        let fp = test_fingerprint();

        let hook_mem = Memory::new(
            "hook content".to_string(),
            MemorySource::HookDescription {
                hook_type: HookType::PostToolUse,
                tool_name: Some("Edit".to_string()),
            },
            "session".to_string(),
            fp.clone(),
            None,
        );
        assert!(hook_mem.is_hook_description());
        assert!(!hook_mem.is_claude_response());
        assert!(!hook_mem.is_md_file_chunk());
        assert_eq!(hook_mem.hook_type(), Some(HookType::PostToolUse));

        let response_mem = Memory::new(
            "response content".to_string(),
            MemorySource::ClaudeResponse {
                response_type: ResponseType::SessionSummary,
            },
            "session".to_string(),
            fp.clone(),
            None,
        );
        assert!(!response_mem.is_hook_description());
        assert!(response_mem.is_claude_response());
        assert!(response_mem.hook_type().is_none());

        let chunk_mem = Memory::new(
            "chunk content".to_string(),
            MemorySource::MDFileChunk {
                file_path: "test.md".to_string(),
                chunk_index: 0,
                total_chunks: 1,
            },
            "session".to_string(),
            fp,
            Some(ChunkMetadata {
                file_path: "test.md".to_string(),
                chunk_index: 0,
                total_chunks: 1,
                word_offset: 0,
                char_offset: 0,
                original_file_hash: "abc123".to_string(),
            }),
        );
        assert!(chunk_mem.is_md_file_chunk());
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn test_memory_validation_empty_content() {
        let fp = test_fingerprint();
        let memory = Memory::new(
            "".to_string(),
            MemorySource::HookDescription {
                hook_type: HookType::SessionStart,
                tool_name: None,
            },
            "session".to_string(),
            fp,
            None,
        );

        let result = memory.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn test_memory_validation_empty_session() {
        let fp = test_fingerprint();
        let memory = Memory::new(
            "content".to_string(),
            MemorySource::HookDescription {
                hook_type: HookType::SessionStart,
                tool_name: None,
            },
            "".to_string(),
            fp,
            None,
        );

        let result = memory.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Session"));
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn test_memory_serialization_roundtrip() {
        let fp = test_fingerprint();
        let memory = Memory::new(
            "serialization test".to_string(),
            MemorySource::HookDescription {
                hook_type: HookType::UserPromptSubmit,
                tool_name: None,
            },
            "session-serialize".to_string(),
            fp,
            None,
        );

        // Test bincode serialization (used for storage)
        let bytes = bincode::serialize(&memory).expect("serialize failed");
        let restored: Memory = bincode::deserialize(&bytes).expect("deserialize failed");

        assert_eq!(memory.id, restored.id);
        assert_eq!(memory.content, restored.content);
        assert_eq!(memory.session_id, restored.session_id);
    }

    #[test]
    fn test_chunk_metadata_fields() {
        let meta = ChunkMetadata {
            file_path: "/path/to/file.md".to_string(),
            chunk_index: 3,
            total_chunks: 10,
            word_offset: 600,
            char_offset: 4500,
            original_file_hash: "sha256hash".to_string(),
        };

        assert_eq!(meta.file_path, "/path/to/file.md");
        assert_eq!(meta.chunk_index, 3);
        assert_eq!(meta.total_chunks, 10);
    }
}
```

### File 3: Update `crates/context-graph-core/src/lib.rs`

Add this line after the existing module declarations (around line 55):

```rust
pub mod memory;
```

And add re-exports after the existing re-exports:

```rust
// Memory capture types (Phase 1)
pub use memory::{ChunkMetadata, HookType, Memory, MemorySource, ResponseType};
```

## Definition of Done

### Compilation Checks
```bash
# MUST pass - no warnings
cargo check --package context-graph-core

# MUST pass - all tests including the new ones
cargo test --package context-graph-core --features test-utils memory::

# MUST pass - verify exports
cargo doc --package context-graph-core --no-deps
```

### Verification Commands (MUST RUN)
```bash
# Verify module exists
ls -la crates/context-graph-core/src/memory/

# Verify exports compile
echo 'use context_graph_core::memory::{Memory, MemorySource, HookType, ResponseType, ChunkMetadata};' | \
  rustfmt && echo "Exports OK"

# Verify type sizes are reasonable
cargo test --package context-graph-core --features test-utils -- memory::tests --nocapture 2>&1 | grep -E "(size|bytes)"
```

## Full State Verification Protocol

### Source of Truth
- **Location**: `crates/context-graph-core/src/memory/` directory
- **Verification**: Files must exist and compile with zero errors/warnings

### Execute & Inspect Sequence

1. **Create files** - Write mod.rs and source.rs
2. **Update lib.rs** - Add module declaration and re-exports
3. **Run cargo check** - Must pass with no errors
4. **Run cargo test** - All tests must pass
5. **Inspect output** - Verify test output shows expected behavior

### Boundary & Edge Case Audit

**Edge Case 1: Empty Content**
```rust
// Input: Memory with empty content string
// Expected: validate() returns Err containing "empty"
// Verify: Run test_memory_validation_empty_content
```

**Edge Case 2: Maximum Content Length**
```rust
// Input: Memory with content > 10,000 chars
// Expected: validate() returns Err with char count
// Verify: Create test with 10,001 char string
```

**Edge Case 3: MDFileChunk without ChunkMetadata**
```rust
// Input: Memory with MDFileChunk source but chunk_metadata = None
// Expected: validate() returns Err about missing chunk_metadata
// Verify: Create test for this specific case
```

### Evidence of Success
After implementation, run:
```bash
# This MUST show passing tests
cargo test --package context-graph-core --features test-utils memory:: 2>&1 | tee /tmp/task-p1-001-evidence.log

# Verify the log contains:
# - "test memory::source::tests::test_hook_type_variants ... ok"
# - "test memory::source::tests::test_response_type_variants ... ok"
# - "test memory::source::tests::test_memory_source_variants ... ok"
# - "test memory::tests::test_memory_new_generates_uuid ... ok"
# - etc.

cat /tmp/task-p1-001-evidence.log | grep -E "^test.*ok$"
```

## Manual Testing Protocol with Synthetic Data

### Test 1: Create Memory and Verify Structure

```rust
// Synthetic input
let content = "User asked about implementing a REST API endpoint";
let hook_type = HookType::UserPromptSubmit;
let session_id = "test-session-001";

// Expected output after Memory::new()
// - id: non-nil UUID v4
// - content: exactly as input
// - source: HookDescription with UserPromptSubmit
// - created_at: within 1 second of now
// - word_count: 8 (count of whitespace-separated words)
```

### Test 2: Serialization Size Check

```rust
// Create Memory with known content
// Serialize with bincode
// Expected: Size < 50KB (fingerprint ~46KB + overhead)
// Verify: Print actual size and compare
```

### Test 3: Source Type Discrimination

```rust
// Create 3 memories, one of each source type
// Verify: is_hook_description(), is_claude_response(), is_md_file_chunk()
// return correct booleans for each
```

## Anti-Patterns to Avoid

1. **DO NOT** create a new TeleologicalArray type - use existing from `types::fingerprint`
2. **DO NOT** use `unwrap()` in library code - use `expect()` with context or `?`
3. **DO NOT** implement Default for Memory - zeroed fingerprints cause silent failures
4. **DO NOT** skip validation - always call `validate()` before storage
5. **DO NOT** create backwards compatibility shims - fail fast if something is wrong

## Execution Checklist

- [ ] Create `crates/context-graph-core/src/memory/` directory
- [ ] Create `source.rs` with MemorySource, HookType, ResponseType
- [ ] Create `mod.rs` with Memory struct and ChunkMetadata
- [ ] Update `lib.rs` with `pub mod memory;` and re-exports
- [ ] Run `cargo check --package context-graph-core` - MUST PASS
- [ ] Run `cargo test --package context-graph-core --features test-utils memory::` - MUST PASS
- [ ] Run `cargo clippy --package context-graph-core -- -D warnings` - MUST PASS
- [ ] Save test output to evidence log
- [ ] Verify all edge case tests pass
- [ ] Proceed to TASK-P1-002 (ChunkMetadata expansion)
