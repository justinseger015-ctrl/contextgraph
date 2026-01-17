# TASK-P1-007: MemoryCaptureService

```xml
<task_spec id="TASK-P1-007" version="2.0">
<metadata>
  <title>MemoryCaptureService Implementation</title>
  <status>COMPLETE</status>
  <layer>logic</layer>
  <sequence>12</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-01</requirement_ref>
    <requirement_ref>REQ-P1-02</requirement_ref>
    <requirement_ref>REQ-P1-03</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETE">TASK-P1-004</task_ref>
    <task_ref status="COMPLETE">TASK-P1-005</task_ref>
    <task_ref status="COMPLETE">TASK-P1-006</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <last_audit>2025-01-16</last_audit>
</metadata>

<context>
Implements the central MemoryCaptureService that coordinates memory capture
from various sources. It orchestrates embedding via EmbeddingProvider trait
and storage via MemoryStore.

This is the primary interface used by CLI commands and hooks to capture memories.
The service follows fail-fast semantics - any error immediately propagates up.
</context>

<!-- =========================================================================
     CODEBASE STATE AUDIT (Verified 2025-01-16)
     ========================================================================= -->
<codebase_audit>
  <verified_files>
    <!-- DEPENDENCY: TextChunker (TASK-P1-004) - EXISTS -->
    <file path="crates/context-graph-core/src/memory/chunker.rs" status="EXISTS">
      <exports>TextChunker, TextChunk, ChunkerError</exports>
      <note>TextChunk has fields: content: String, metadata: ChunkMetadata, content_hash: String</note>
    </file>

    <!-- DEPENDENCY: MemoryStore (TASK-P1-005) - COMPLETE -->
    <file path="crates/context-graph-core/src/memory/store.rs" status="COMPLETE">
      <exports>MemoryStore, StorageError, CF_MEMORIES, CF_SESSION_INDEX</exports>
      <methods>
        new(path: impl AsRef&lt;Path&gt;) -> Result&lt;Self, StorageError&gt;
        store(&amp;self, memory: Memory) -> Result&lt;(), StorageError&gt;
        get(&amp;self, id: Uuid) -> Result&lt;Option&lt;Memory&gt;, StorageError&gt;
        get_by_session(&amp;self, session_id: &amp;str) -> Result&lt;Vec&lt;Memory&gt;, StorageError&gt;
        count(&amp;self) -> Result&lt;usize, StorageError&gt;
        delete(&amp;self, id: Uuid) -> Result&lt;bool, StorageError&gt;
      </methods>
      <backend>RocksDB with bincode serialization</backend>
    </file>

    <!-- DEPENDENCY: SessionManager (TASK-P1-006) - COMPLETE -->
    <file path="crates/context-graph-core/src/memory/manager.rs" status="COMPLETE">
      <exports>SessionManager, SessionError, CF_SESSIONS</exports>
      <methods>
        new(path: impl AsRef&lt;Path&gt;) -> Result&lt;Self, SessionError&gt;
        start_session(&amp;self) -> Result&lt;Session, SessionError&gt;
        end_session(&amp;self, session_id: &amp;str) -> Result&lt;Session, SessionError&gt;
        abandon_session(&amp;self, session_id: &amp;str) -> Result&lt;Session, SessionError&gt;
        get_current_session(&amp;self) -> Result&lt;Option&lt;Session&gt;, SessionError&gt;
        get_session(&amp;self, session_id: &amp;str) -> Result&lt;Option&lt;Session&gt;, SessionError&gt;
        list_active_sessions(&amp;self) -> Result&lt;Vec&lt;Session&gt;, SessionError&gt;
        increment_memory_count(&amp;self, session_id: &amp;str) -> Result&lt;u32, SessionError&gt;
      </methods>
      <backend>RocksDB with bincode serialization</backend>
    </file>

    <!-- Memory struct definition -->
    <file path="crates/context-graph-core/src/memory/mod.rs" status="EXISTS">
      <exports>Memory, ChunkMetadata, MemorySource, HookType, ResponseType, TextChunker, TextChunk, ChunkerError, MemoryStore, StorageError, SessionManager, SessionError, Session, SessionStatus</exports>
      <memory_struct>
        id: Uuid
        content: String
        source: MemorySource
        created_at: DateTime&lt;Utc&gt;
        session_id: String
        teleological_array: TeleologicalArray
        chunk_metadata: Option&lt;ChunkMetadata&gt;
        word_count: u32
      </memory_struct>
      <memory_methods>
        Memory::new(content, source, session_id, teleological_array, chunk_metadata) -> Self
        Memory::with_id(id, content, source, session_id, teleological_array, chunk_metadata) -> Self
        Memory::validate(&amp;self) -> Result&lt;(), String&gt;
      </memory_methods>
      <constants>MAX_CONTENT_LENGTH = 10_000</constants>
    </file>

    <!-- Source types -->
    <file path="crates/context-graph-core/src/memory/source.rs" status="EXISTS">
      <memorysource_variants>
        HookDescription { hook_type: HookType, tool_name: Option&lt;String&gt; }
        ClaudeResponse { response_type: ResponseType }
        MDFileChunk { file_path: String, chunk_index: u32, total_chunks: u32 }
      </memorysource_variants>
      <hooktype_variants>SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, Stop, SessionEnd</hooktype_variants>
      <responsetype_variants>SessionSummary, StopResponse, SignificantResponse</responsetype_variants>
    </file>

    <!-- TeleologicalArray type definition -->
    <file path="crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs" status="EXISTS">
      <type_alias>pub type TeleologicalArray = SemanticFingerprint;</type_alias>
      <import_path>crate::types::fingerprint::TeleologicalArray</import_path>
      <note>SemanticFingerprint::zeroed() only available with test-utils feature</note>
    </file>

    <!-- ChunkMetadata struct -->
    <file path="crates/context-graph-core/src/memory/mod.rs#ChunkMetadata" status="EXISTS">
      <fields>
        file_path: String
        chunk_index: u32
        total_chunks: u32
        word_offset: u32
        char_offset: u32
        original_file_hash: String
      </fields>
    </file>
  </verified_files>

  <compilation_status verified="2025-01-16">
    <result>PASS with warnings</result>
    <warnings>3 unused imports in coherence module (unrelated to memory)</warnings>
    <command>cargo check --package context-graph-core</command>
  </compilation_status>

  <dependencies_cargo_toml>
    <dependency name="async-trait" status="PRESENT">async-trait.workspace = true</dependency>
    <dependency name="thiserror" status="PRESENT">thiserror.workspace = true</dependency>
    <dependency name="uuid" status="PRESENT">uuid.workspace = true</dependency>
    <dependency name="chrono" status="PRESENT">chrono.workspace = true</dependency>
    <dependency name="serde" status="PRESENT">serde.workspace = true</dependency>
    <dependency name="bincode" status="PRESENT">bincode.workspace = true</dependency>
  </dependencies_cargo_toml>
</codebase_audit>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md#component_contracts</file>
  <file purpose="memory_store" verified="true">crates/context-graph-core/src/memory/store.rs</file>
  <file purpose="session_manager" verified="true">crates/context-graph-core/src/memory/manager.rs</file>
  <file purpose="memory_types" verified="true">crates/context-graph-core/src/memory/mod.rs</file>
  <file purpose="source_types" verified="true">crates/context-graph-core/src/memory/source.rs</file>
  <file purpose="teleological_array" verified="true">crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs</file>
  <file purpose="chunker" verified="true">crates/context-graph-core/src/memory/chunker.rs</file>
</input_context_files>

<prerequisites>
  <check status="COMPLETE">TASK-P1-004 complete - TextChunker exists in chunker.rs</check>
  <check status="COMPLETE">TASK-P1-005 complete - MemoryStore exists in store.rs with RocksDB backend</check>
  <check status="COMPLETE">TASK-P1-006 complete - SessionManager exists in manager.rs with RocksDB backend</check>
  <check status="VERIFIED">TeleologicalArray type exists at crate::types::fingerprint::TeleologicalArray</check>
  <check status="VERIFIED">async-trait dependency present in Cargo.toml</check>
</prerequisites>

<scope>
  <in_scope>
    - Create MemoryCaptureService struct in capture.rs
    - Implement EmbeddingProvider trait (async, returns TeleologicalArray)
    - Implement EmbedderError enum with thiserror
    - Implement CaptureError enum with thiserror
    - Implement capture_hook_description() method
    - Implement capture_claude_response() method
    - Implement capture_md_chunk() method
    - Implement internal capture_memory() method
    - All methods fail-fast on any error (no recovery attempts)
    - Validate content before embedding (empty content = error)
  </in_scope>
  <out_of_scope>
    - MDFileWatcher (TASK-P1-008)
    - Actual embedding implementation (Phase 2 - GPU pipeline)
    - CLI integration (Phase 6)
    - Session lifecycle management (already in SessionManager)
  </out_of_scope>
</scope>

<!-- =========================================================================
     FULL STATE VERIFICATION PROTOCOL
     ========================================================================= -->
<full_state_verification>
  <source_of_truth>
    <primary name="RocksDB CF_MEMORIES">
      <location>Database column family "memories"</location>
      <access>Via MemoryStore::get(id) and MemoryStore::get_by_session(session_id)</access>
      <serialization>bincode</serialization>
      <verification>After store(), call get() with returned UUID to verify persistence</verification>
    </primary>
    <secondary name="RocksDB CF_SESSION_INDEX">
      <location>Database column family "session_index"</location>
      <access>Via MemoryStore::get_by_session(session_id)</access>
      <verification>Verify memory appears in session's memory list</verification>
    </secondary>
  </source_of_truth>

  <execute_and_inspect>
    <step order="1">
      <action>Create MemoryCaptureService with mock embedder and real MemoryStore</action>
      <inspect>MemoryStore::count() returns 0</inspect>
    </step>
    <step order="2">
      <action>Call capture_hook_description() with valid content</action>
      <inspect>Returns Ok(Uuid)</inspect>
      <inspect>MemoryStore::count() returns 1</inspect>
      <inspect>MemoryStore::get(uuid) returns Some(memory) with correct fields</inspect>
    </step>
    <step order="3">
      <action>Call capture_claude_response() with valid content</action>
      <inspect>Returns Ok(Uuid)</inspect>
      <inspect>MemoryStore::count() returns 2</inspect>
    </step>
    <step order="4">
      <action>Call capture_md_chunk() with valid TextChunk</action>
      <inspect>Returns Ok(Uuid)</inspect>
      <inspect>MemoryStore::get(uuid).chunk_metadata is Some</inspect>
    </step>
    <step order="5">
      <action>Call capture_hook_description() with empty content ""</action>
      <inspect>Returns Err(CaptureError::EmptyContent)</inspect>
      <inspect>MemoryStore::count() unchanged (still 3)</inspect>
    </step>
  </execute_and_inspect>

  <boundary_edge_cases minimum="3">
    <edge_case id="EC-001" category="input_validation">
      <name>Empty Content</name>
      <input>content = "" or "   " (whitespace only)</input>
      <expected>Err(CaptureError::EmptyContent)</expected>
      <rationale>Empty memories have no semantic value; fail fast</rationale>
    </edge_case>

    <edge_case id="EC-002" category="input_validation">
      <name>Content at MAX_CONTENT_LENGTH boundary</name>
      <input>content = "x".repeat(10_000) exactly</input>
      <expected>Ok(Uuid) - should succeed at exactly the limit</expected>
      <rationale>Boundary value testing for content length validation</rationale>
    </edge_case>

    <edge_case id="EC-003" category="input_validation">
      <name>Content exceeds MAX_CONTENT_LENGTH</name>
      <input>content = "x".repeat(10_001)</input>
      <expected>Err(CaptureError::ContentTooLong) or validation fails at Memory::validate()</expected>
      <rationale>Memory struct validates content &lt;= 10,000 chars</rationale>
    </edge_case>

    <edge_case id="EC-004" category="error_propagation">
      <name>Embedding provider failure</name>
      <input>MockEmbeddingProvider configured to return Err(EmbedderError::Unavailable)</input>
      <expected>Err(CaptureError::EmbeddingFailed(EmbedderError::Unavailable))</expected>
      <rationale>Embedding errors must propagate immediately (fail-fast)</rationale>
    </edge_case>

    <edge_case id="EC-005" category="error_propagation">
      <name>Storage failure</name>
      <input>MemoryStore with closed/corrupted database</input>
      <expected>Err(CaptureError::StorageFailed(...))</expected>
      <rationale>Storage errors must propagate immediately (fail-fast)</rationale>
    </edge_case>

    <edge_case id="EC-006" category="data_integrity">
      <name>MDFileChunk requires ChunkMetadata</name>
      <input>capture_md_chunk with TextChunk containing empty metadata fields</input>
      <expected>Memory stored with valid chunk_metadata populated from TextChunk.metadata</expected>
      <rationale>Memory::validate() requires chunk_metadata for MDFileChunk source</rationale>
    </edge_case>

    <edge_case id="EC-007" category="concurrency">
      <name>Concurrent captures to same session</name>
      <input>Multiple concurrent capture calls with same session_id</input>
      <expected>All succeed; get_by_session returns all memories</expected>
      <rationale>RocksDB handles concurrent writes; verify no data loss</rationale>
    </edge_case>
  </boundary_edge_cases>

  <evidence_of_success>
    <log level="INFO" when="capture_start">[MemoryCaptureService] Capturing {source_type} for session {session_id}, content_len={len}</log>
    <log level="DEBUG" when="embedding_complete">[MemoryCaptureService] Embedding complete, TeleologicalArray size={size} bytes</log>
    <log level="INFO" when="capture_complete">[MemoryCaptureService] Stored memory {uuid} for session {session_id}</log>
    <log level="ERROR" when="capture_failed">[MemoryCaptureService] Capture failed: {error}</log>
    <metric name="capture_latency_ms">Time from capture call to store completion</metric>
    <metric name="embedding_latency_ms">Time spent in embed_all()</metric>
  </evidence_of_success>
</full_state_verification>

<!-- =========================================================================
     DEFINITION OF DONE
     ========================================================================= -->
<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/memory/capture.rs">
      // Error types
      #[derive(Debug, Error)]
      pub enum EmbedderError {
          #[error("Embedding service unavailable")]
          Unavailable,
          #[error("Embedding computation failed: {message}")]
          ComputationFailed { message: String },
          #[error("Invalid input for embedding: {reason}")]
          InvalidInput { reason: String },
      }

      #[derive(Debug, Error)]
      pub enum CaptureError {
          #[error("Content is empty or whitespace-only")]
          EmptyContent,
          #[error("Content exceeds maximum length of {max} characters: got {actual}")]
          ContentTooLong { max: usize, actual: usize },
          #[error("Embedding failed: {0}")]
          EmbeddingFailed(#[from] EmbedderError),
          #[error("Storage failed: {0}")]
          StorageFailed(#[from] StorageError),
          #[error("Memory validation failed: {reason}")]
          ValidationFailed { reason: String },
      }

      // Trait for embedding providers (GPU pipeline in Phase 2)
      #[async_trait]
      pub trait EmbeddingProvider: Send + Sync {
          /// Embed content into full 13-embedding TeleologicalArray.
          /// MUST return valid array with all 13 embeddings or error.
          async fn embed_all(&amp;self, content: &amp;str) -> Result&lt;TeleologicalArray, EmbedderError&gt;;
      }

      // Main service
      pub struct MemoryCaptureService {
          store: Arc&lt;MemoryStore&gt;,
          embedder: Arc&lt;dyn EmbeddingProvider&gt;,
      }

      impl MemoryCaptureService {
          pub fn new(store: Arc&lt;MemoryStore&gt;, embedder: Arc&lt;dyn EmbeddingProvider&gt;) -> Self;

          pub async fn capture_hook_description(
              &amp;self,
              content: String,
              hook_type: HookType,
              session_id: String,
              tool_name: Option&lt;String&gt;,
          ) -> Result&lt;Uuid, CaptureError&gt;;

          pub async fn capture_claude_response(
              &amp;self,
              content: String,
              response_type: ResponseType,
              session_id: String,
          ) -> Result&lt;Uuid, CaptureError&gt;;

          pub async fn capture_md_chunk(
              &amp;self,
              chunk: TextChunk,
              session_id: String,
          ) -> Result&lt;Uuid, CaptureError&gt;;
      }
    </signature>
  </signatures>

  <constraints>
    <constraint id="C-001" type="fail-fast">All capture methods fail immediately on any error - no retries, no partial states</constraint>
    <constraint id="C-002" type="validation">Empty or whitespace-only content returns CaptureError::EmptyContent before embedding</constraint>
    <constraint id="C-003" type="validation">Content &gt; MAX_CONTENT_LENGTH (10,000) returns CaptureError::ContentTooLong</constraint>
    <constraint id="C-004" type="propagation">Embedding errors propagate as CaptureError::EmbeddingFailed</constraint>
    <constraint id="C-005" type="propagation">Storage errors propagate as CaptureError::StorageFailed</constraint>
    <constraint id="C-006" type="atomic">No partial storage - embed and store succeed together or both fail</constraint>
    <constraint id="C-007" type="validation">Memory::validate() must pass before storage</constraint>
    <constraint id="C-008" type="logging">All capture attempts logged with content length and session_id</constraint>
  </constraints>

  <verification>
    <test name="capture_hook_description_success">Creates memory with MemorySource::HookDescription</test>
    <test name="capture_claude_response_success">Creates memory with MemorySource::ClaudeResponse</test>
    <test name="capture_md_chunk_success">Creates memory with MemorySource::MDFileChunk and chunk_metadata</test>
    <test name="capture_empty_content_fails">Returns CaptureError::EmptyContent for ""</test>
    <test name="capture_whitespace_content_fails">Returns CaptureError::EmptyContent for "   \n\t  "</test>
    <test name="capture_content_at_max_length_succeeds">10,000 char content succeeds</test>
    <test name="capture_content_over_max_length_fails">10,001 char content fails</test>
    <test name="embedding_error_propagates">EmbedderError::Unavailable becomes CaptureError::EmbeddingFailed</test>
    <test name="storage_error_propagates">StorageError becomes CaptureError::StorageFailed</test>
    <test name="memory_persists_to_database">MemoryStore::get(uuid) returns stored memory</test>
    <test name="memory_indexed_by_session">MemoryStore::get_by_session returns the memory</test>
  </verification>
</definition_of_done>

<!-- =========================================================================
     SYNTHETIC TEST DATA
     ========================================================================= -->
<synthetic_test_data>
  <test_case id="SYN-001" name="hook_description_valid">
    <input>
      content: "Claude used the Edit tool to modify src/main.rs, adding a new function called process_data that handles incoming API requests."
      hook_type: HookType::PostToolUse
      session_id: "sess_abc123"
      tool_name: Some("Edit")
    </input>
    <expected_memory>
      id: [generated UUID]
      content: [same as input]
      source: MemorySource::HookDescription { hook_type: PostToolUse, tool_name: Some("Edit") }
      session_id: "sess_abc123"
      word_count: 22
      chunk_metadata: None
      teleological_array: [from mock embedder]
    </expected_memory>
    <expected_result>Ok(uuid)</expected_result>
  </test_case>

  <test_case id="SYN-002" name="claude_response_valid">
    <input>
      content: "I've completed the implementation of the authentication module. The key changes include: 1) Added JWT token validation, 2) Implemented refresh token rotation, 3) Added rate limiting for login attempts."
      response_type: ResponseType::SessionSummary
      session_id: "sess_xyz789"
    </input>
    <expected_memory>
      id: [generated UUID]
      content: [same as input]
      source: MemorySource::ClaudeResponse { response_type: SessionSummary }
      session_id: "sess_xyz789"
      word_count: 35
      chunk_metadata: None
    </expected_memory>
    <expected_result>Ok(uuid)</expected_result>
  </test_case>

  <test_case id="SYN-003" name="md_chunk_valid">
    <input>
      chunk: TextChunk {
        content: "## Authentication Flow\n\nThe system uses JWT tokens for stateless authentication. When a user logs in, they receive an access token (15 min expiry) and a refresh token (7 day expiry).",
        metadata: ChunkMetadata {
          file_path: "docs/auth.md",
          chunk_index: 0,
          total_chunks: 5,
          word_offset: 0,
          char_offset: 0,
          original_file_hash: "sha256:abc123def456..."
        },
        content_hash: "sha256:chunk_hash_here"
      }
      session_id: "sess_chunk001"
    </input>
    <expected_memory>
      source: MemorySource::MDFileChunk { file_path: "docs/auth.md", chunk_index: 0, total_chunks: 5 }
      chunk_metadata: Some(ChunkMetadata { file_path: "docs/auth.md", ... })
      word_count: 35
    </expected_memory>
    <expected_result>Ok(uuid)</expected_result>
  </test_case>

  <test_case id="SYN-004" name="empty_content_rejected">
    <input>
      content: ""
      hook_type: HookType::SessionStart
      session_id: "sess_empty"
      tool_name: None
    </input>
    <expected_result>Err(CaptureError::EmptyContent)</expected_result>
    <expected_side_effects>
      - No embedding call made
      - No storage call made
      - MemoryStore::count() unchanged
    </expected_side_effects>
  </test_case>

  <test_case id="SYN-005" name="whitespace_only_rejected">
    <input>
      content: "   \n\t  \r\n   "
      hook_type: HookType::UserPromptSubmit
      session_id: "sess_ws"
      tool_name: None
    </input>
    <expected_result>Err(CaptureError::EmptyContent)</expected_result>
  </test_case>

  <test_case id="SYN-006" name="max_length_boundary">
    <input>
      content: "x".repeat(10_000)  // exactly at limit
      hook_type: HookType::Stop
      session_id: "sess_boundary"
      tool_name: None
    </input>
    <expected_result>Ok(uuid)</expected_result>
    <expected_memory>
      word_count: 1  // one giant "word"
    </expected_memory>
  </test_case>

  <test_case id="SYN-007" name="over_max_length_rejected">
    <input>
      content: "x".repeat(10_001)  // one over limit
      hook_type: HookType::Stop
      session_id: "sess_toolong"
      tool_name: None
    </input>
    <expected_result>Err(CaptureError::ContentTooLong { max: 10000, actual: 10001 })</expected_result>
  </test_case>
</synthetic_test_data>

<!-- =========================================================================
     MANUAL VERIFICATION PROTOCOL
     ========================================================================= -->
<manual_verification>
  <step order="1" name="database_verification">
    <command>cargo test --package context-graph-core capture -- --nocapture</command>
    <verify>All tests pass</verify>
    <verify>Test output shows INFO logs for capture operations</verify>
  </step>

  <step order="2" name="physical_storage_check">
    <description>Verify memory actually persists to RocksDB</description>
    <code_snippet>
      // In integration test:
      let dir = tempfile::tempdir().unwrap();
      let store = Arc::new(MemoryStore::new(dir.path()).unwrap());
      let service = MemoryCaptureService::new(store.clone(), embedder);

      let uuid = service.capture_hook_description(...).await.unwrap();

      // Verify persistence
      let retrieved = store.get(uuid).unwrap();
      assert!(retrieved.is_some());
      let memory = retrieved.unwrap();
      assert_eq!(memory.id, uuid);
      assert_eq!(memory.content, original_content);

      // Verify session index
      let session_memories = store.get_by_session("test-session").unwrap();
      assert!(session_memories.iter().any(|m| m.id == uuid));
    </code_snippet>
  </step>

  <step order="3" name="error_path_verification">
    <description>Verify errors propagate correctly and don't leave partial state</description>
    <code_snippet>
      // Create failing embedder
      struct FailingEmbedder;
      #[async_trait]
      impl EmbeddingProvider for FailingEmbedder {
          async fn embed_all(&amp;self, _: &amp;str) -> Result&lt;TeleologicalArray, EmbedderError&gt; {
              Err(EmbedderError::Unavailable)
          }
      }

      let count_before = store.count().unwrap();
      let result = service.capture_hook_description(...).await;
      assert!(matches!(result, Err(CaptureError::EmbeddingFailed(_))));
      let count_after = store.count().unwrap();
      assert_eq!(count_before, count_after); // No partial storage
    </code_snippet>
  </step>

  <step order="4" name="compilation_check">
    <command>cargo check --package context-graph-core</command>
    <verify>No errors</verify>
    <verify>No new warnings in memory module</verify>
  </step>

  <step order="5" name="clippy_check">
    <command>cargo clippy --package context-graph-core -- -D warnings</command>
    <verify>No warnings</verify>
  </step>
</manual_verification>

<!-- =========================================================================
     IMPLEMENTATION CODE
     ========================================================================= -->
<pseudo_code>
File: crates/context-graph-core/src/memory/capture.rs

//! Memory capture service for coordinating embedding and storage.
//!
//! # Architecture
//!
//! The MemoryCaptureService coordinates:
//! 1. Content validation (empty, length)
//! 2. Embedding via EmbeddingProvider trait
//! 3. Memory construction with proper source type
//! 4. Persistence via MemoryStore
//!
//! # Fail-Fast Semantics
//!
//! All operations fail immediately on any error. No retries, no partial states.
//! This follows constitution.yaml AP-14: "No .unwrap() in library code".
//!
//! # Constitution Compliance
//! - ARCH-01: TeleologicalArray is atomic (all 13 embeddings)
//! - ARCH-06: All memory ops through MCP tools (this is the service layer)
//! - AP-14: No .unwrap() - all errors propagate via Result

use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;
use tracing::{debug, error, info, instrument};
use uuid::Uuid;

use super::{
    ChunkMetadata, HookType, Memory, MemorySource, ResponseType, TextChunk,
    store::{MemoryStore, StorageError},
    MAX_CONTENT_LENGTH,
};
use crate::types::fingerprint::TeleologicalArray;

/// Errors from embedding operations.
///
/// These errors indicate failures in the embedding pipeline.
/// In Phase 1, only the mock embedder is used; in Phase 2,
/// the GPU pipeline will produce these errors.
#[derive(Debug, Error)]
pub enum EmbedderError {
    /// Embedding service is not available (GPU offline, model not loaded).
    #[error("Embedding service unavailable")]
    Unavailable,

    /// Embedding computation failed (GPU error, memory exhaustion).
    #[error("Embedding computation failed: {message}")]
    ComputationFailed { message: String },

    /// Input is invalid for embedding (e.g., unsupported characters).
    #[error("Invalid input for embedding: {reason}")]
    InvalidInput { reason: String },
}

/// Errors from memory capture operations.
///
/// Captures all failure modes in the capture pipeline:
/// validation, embedding, and storage.
#[derive(Debug, Error)]
pub enum CaptureError {
    /// Content is empty or contains only whitespace.
    #[error("Content is empty or whitespace-only")]
    EmptyContent,

    /// Content exceeds maximum allowed length.
    #[error("Content exceeds maximum length of {max} characters: got {actual}")]
    ContentTooLong { max: usize, actual: usize },

    /// Embedding operation failed.
    #[error("Embedding failed: {0}")]
    EmbeddingFailed(#[from] EmbedderError),

    /// Storage operation failed.
    #[error("Storage failed: {0}")]
    StorageFailed(#[from] StorageError),

    /// Memory validation failed after construction.
    #[error("Memory validation failed: {reason}")]
    ValidationFailed { reason: String },
}

/// Trait for embedding providers.
///
/// Implementations must produce a complete TeleologicalArray with all
/// 13 embeddings. Partial arrays are not allowed (ARCH-01).
///
/// # Phase 1
///
/// Uses TestEmbeddingProvider with zeroed arrays for testing.
///
/// # Phase 2+
///
/// GPU pipeline implementation will provide real embeddings.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed content into a full 13-embedding TeleologicalArray.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content to embed (assumed non-empty)
    ///
    /// # Returns
    ///
    /// * `Ok(TeleologicalArray)` - Complete array with all 13 embeddings
    /// * `Err(EmbedderError)` - If embedding fails for any reason
    ///
    /// # Contract
    ///
    /// The returned TeleologicalArray MUST pass `validate_strict()`.
    /// Partial or invalid arrays are considered bugs in the implementation.
    async fn embed_all(&amp;self, content: &amp;str) -> Result&lt;TeleologicalArray, EmbedderError&gt;;
}

/// Memory capture service coordinating embedding and storage.
///
/// This is the primary interface for capturing memories from:
/// - Hook events (SessionStart, PostToolUse, etc.)
/// - Claude responses (SessionSummary, StopResponse)
/// - Markdown file chunks (from MDFileWatcher)
///
/// # Thread Safety
///
/// The service is `Send + Sync` and can be shared across async tasks.
/// Both MemoryStore and EmbeddingProvider are accessed through Arc.
pub struct MemoryCaptureService {
    store: Arc&lt;MemoryStore&gt;,
    embedder: Arc&lt;dyn EmbeddingProvider&gt;,
}

impl MemoryCaptureService {
    /// Create a new MemoryCaptureService.
    ///
    /// # Arguments
    ///
    /// * `store` - The MemoryStore for persistence
    /// * `embedder` - The embedding provider (mock in Phase 1, GPU in Phase 2+)
    pub fn new(store: Arc&lt;MemoryStore&gt;, embedder: Arc&lt;dyn EmbeddingProvider&gt;) -> Self {
        Self { store, embedder }
    }

    /// Capture a hook description as memory.
    ///
    /// # Arguments
    ///
    /// * `content` - Description of what Claude did during the hook
    /// * `hook_type` - Which hook triggered this capture
    /// * `session_id` - Current session identifier
    /// * `tool_name` - Tool name for PreToolUse/PostToolUse hooks
    ///
    /// # Returns
    ///
    /// * `Ok(Uuid)` - The ID of the stored memory
    /// * `Err(CaptureError)` - If validation, embedding, or storage fails
    #[instrument(skip(self, content), fields(content_len = content.len()))]
    pub async fn capture_hook_description(
        &amp;self,
        content: String,
        hook_type: HookType,
        session_id: String,
        tool_name: Option&lt;String&gt;,
    ) -> Result&lt;Uuid, CaptureError&gt; {
        info!(
            hook_type = %hook_type,
            session_id = %session_id,
            tool_name = ?tool_name,
            "Capturing hook description"
        );

        let source = MemorySource::HookDescription { hook_type, tool_name };
        self.capture_memory(content, source, session_id, None).await
    }

    /// Capture a Claude response as memory.
    ///
    /// # Arguments
    ///
    /// * `content` - The response content to capture
    /// * `response_type` - Type of response (SessionSummary, StopResponse, etc.)
    /// * `session_id` - Current session identifier
    ///
    /// # Returns
    ///
    /// * `Ok(Uuid)` - The ID of the stored memory
    /// * `Err(CaptureError)` - If validation, embedding, or storage fails
    #[instrument(skip(self, content), fields(content_len = content.len()))]
    pub async fn capture_claude_response(
        &amp;self,
        content: String,
        response_type: ResponseType,
        session_id: String,
    ) -> Result&lt;Uuid, CaptureError&gt; {
        info!(
            response_type = %response_type,
            session_id = %session_id,
            "Capturing Claude response"
        );

        let source = MemorySource::ClaudeResponse { response_type };
        self.capture_memory(content, source, session_id, None).await
    }

    /// Capture a markdown file chunk as memory.
    ///
    /// # Arguments
    ///
    /// * `chunk` - The TextChunk containing content and metadata
    /// * `session_id` - Current session identifier
    ///
    /// # Returns
    ///
    /// * `Ok(Uuid)` - The ID of the stored memory
    /// * `Err(CaptureError)` - If validation, embedding, or storage fails
    #[instrument(skip(self, chunk), fields(
        file_path = %chunk.metadata.file_path,
        chunk_index = chunk.metadata.chunk_index,
        total_chunks = chunk.metadata.total_chunks
    ))]
    pub async fn capture_md_chunk(
        &amp;self,
        chunk: TextChunk,
        session_id: String,
    ) -> Result&lt;Uuid, CaptureError&gt; {
        info!(
            file_path = %chunk.metadata.file_path,
            chunk = %format!("{}/{}", chunk.metadata.chunk_index + 1, chunk.metadata.total_chunks),
            session_id = %session_id,
            "Capturing MD chunk"
        );

        let source = MemorySource::MDFileChunk {
            file_path: chunk.metadata.file_path.clone(),
            chunk_index: chunk.metadata.chunk_index,
            total_chunks: chunk.metadata.total_chunks,
        };

        // Convert TextChunk metadata to Memory's ChunkMetadata
        let chunk_metadata = ChunkMetadata {
            file_path: chunk.metadata.file_path,
            chunk_index: chunk.metadata.chunk_index,
            total_chunks: chunk.metadata.total_chunks,
            word_offset: chunk.metadata.word_offset,
            char_offset: chunk.metadata.char_offset,
            original_file_hash: chunk.metadata.original_file_hash,
        };

        self.capture_memory(chunk.content, source, session_id, Some(chunk_metadata))
            .await
    }

    /// Internal method to capture memory with validation, embedding, and storage.
    ///
    /// # Fail-Fast Behavior
    ///
    /// 1. Validate content (empty, length) - fail if invalid
    /// 2. Call embedder - fail if embedding fails
    /// 3. Construct Memory - fail if validation fails
    /// 4. Store memory - fail if storage fails
    /// 5. Return UUID only on complete success
    async fn capture_memory(
        &amp;self,
        content: String,
        source: MemorySource,
        session_id: String,
        chunk_metadata: Option&lt;ChunkMetadata&gt;,
    ) -> Result&lt;Uuid, CaptureError&gt; {
        // Step 1: Validate content
        if content.trim().is_empty() {
            error!("Capture rejected: empty content");
            return Err(CaptureError::EmptyContent);
        }

        if content.len() > MAX_CONTENT_LENGTH {
            error!(
                content_len = content.len(),
                max = MAX_CONTENT_LENGTH,
                "Capture rejected: content too long"
            );
            return Err(CaptureError::ContentTooLong {
                max: MAX_CONTENT_LENGTH,
                actual: content.len(),
            });
        }

        // Step 2: Generate embeddings (fail fast on error)
        debug!(content_len = content.len(), "Starting embedding");
        let teleological_array = self.embedder.embed_all(&amp;content).await?;
        debug!(
            storage_size = teleological_array.storage_size(),
            "Embedding complete"
        );

        // Step 3: Construct Memory
        let memory = Memory::new(
            content,
            source,
            session_id.clone(),
            teleological_array,
            chunk_metadata,
        );

        // Step 4: Validate Memory (defensive - Memory::new should produce valid data)
        memory.validate().map_err(|reason| {
            error!(reason = %reason, "Memory validation failed");
            CaptureError::ValidationFailed { reason }
        })?;

        let memory_id = memory.id;

        // Step 5: Store (fail fast on error)
        self.store.store(memory)?;

        info!(
            memory_id = %memory_id,
            session_id = %session_id,
            "Memory stored successfully"
        );

        Ok(memory_id)
    }
}

// ============================================================================
// Test Embedding Provider (test-utils feature only)
// ============================================================================

/// Test embedding provider that returns zeroed TeleologicalArrays.
///
/// # ⚠️ TEST ONLY
///
/// This provider returns zeroed embeddings which:
/// - Pass dimension validation
/// - Have zero magnitude (undefined cosine similarity)
/// - Should NEVER be used in production
///
/// For production, use the GPU embedding pipeline from Phase 2+.
#[cfg(any(test, feature = "test-utils"))]
pub struct TestEmbeddingProvider;

#[cfg(any(test, feature = "test-utils"))]
#[async_trait]
impl EmbeddingProvider for TestEmbeddingProvider {
    async fn embed_all(&amp;self, _content: &amp;str) -> Result&lt;TeleologicalArray, EmbedderError&gt; {
        use crate::types::fingerprint::SemanticFingerprint;
        Ok(SemanticFingerprint::zeroed())
    }
}

/// Test embedding provider that always fails.
///
/// Use this to test error propagation paths.
#[cfg(any(test, feature = "test-utils"))]
pub struct FailingEmbeddingProvider {
    pub error: EmbedderError,
}

#[cfg(any(test, feature = "test-utils"))]
impl FailingEmbeddingProvider {
    pub fn unavailable() -> Self {
        Self {
            error: EmbedderError::Unavailable,
        }
    }

    pub fn computation_failed(message: impl Into&lt;String&gt;) -> Self {
        Self {
            error: EmbedderError::ComputationFailed {
                message: message.into(),
            },
        }
    }
}

#[cfg(any(test, feature = "test-utils"))]
#[async_trait]
impl EmbeddingProvider for FailingEmbeddingProvider {
    async fn embed_all(&amp;self, _content: &amp;str) -> Result&lt;TeleologicalArray, EmbedderError&gt; {
        // Clone the error to return it
        match &amp;self.error {
            EmbedderError::Unavailable => Err(EmbedderError::Unavailable),
            EmbedderError::ComputationFailed { message } => {
                Err(EmbedderError::ComputationFailed {
                    message: message.clone(),
                })
            }
            EmbedderError::InvalidInput { reason } => Err(EmbedderError::InvalidInput {
                reason: reason.clone(),
            }),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // Helper to create test service
    async fn test_service() -> (MemoryCaptureService, Arc&lt;MemoryStore&gt;, tempfile::TempDir) {
        let dir = tempdir().expect("create temp dir");
        let store = Arc::new(MemoryStore::new(dir.path()).expect("create store"));
        let embedder = Arc::new(TestEmbeddingProvider);
        let service = MemoryCaptureService::new(store.clone(), embedder);
        (service, store, dir)
    }

    #[tokio::test]
    async fn test_capture_hook_description_success() {
        let (service, store, _dir) = test_service().await;

        let result = service
            .capture_hook_description(
                "Claude edited the config file".to_string(),
                HookType::PostToolUse,
                "sess-001".to_string(),
                Some("Edit".to_string()),
            )
            .await;

        assert!(result.is_ok());
        let uuid = result.unwrap();

        // Verify persistence
        let memory = store.get(uuid).expect("get memory").expect("memory exists");
        assert_eq!(memory.content, "Claude edited the config file");
        assert!(matches!(
            memory.source,
            MemorySource::HookDescription {
                hook_type: HookType::PostToolUse,
                tool_name: Some(ref name)
            } if name == "Edit"
        ));
        assert_eq!(memory.session_id, "sess-001");
        assert_eq!(memory.word_count, 5);
    }

    #[tokio::test]
    async fn test_capture_claude_response_success() {
        let (service, store, _dir) = test_service().await;

        let result = service
            .capture_claude_response(
                "Session completed successfully with 5 tasks done".to_string(),
                ResponseType::SessionSummary,
                "sess-002".to_string(),
            )
            .await;

        assert!(result.is_ok());
        let uuid = result.unwrap();

        let memory = store.get(uuid).expect("get").expect("exists");
        assert!(matches!(
            memory.source,
            MemorySource::ClaudeResponse {
                response_type: ResponseType::SessionSummary
            }
        ));
    }

    #[tokio::test]
    async fn test_capture_md_chunk_success() {
        let (service, store, _dir) = test_service().await;

        let chunk = TextChunk {
            content: "# Documentation\n\nThis is the documentation content.".to_string(),
            metadata: super::super::chunker::ChunkMetadata {
                file_path: "docs/README.md".to_string(),
                chunk_index: 0,
                total_chunks: 3,
                word_offset: 0,
                char_offset: 0,
                original_file_hash: "sha256:abc123".to_string(),
            },
            content_hash: "sha256:chunk123".to_string(),
        };

        let result = service
            .capture_md_chunk(chunk, "sess-003".to_string())
            .await;

        assert!(result.is_ok());
        let uuid = result.unwrap();

        let memory = store.get(uuid).expect("get").expect("exists");
        assert!(matches!(
            memory.source,
            MemorySource::MDFileChunk {
                ref file_path,
                chunk_index: 0,
                total_chunks: 3
            } if file_path == "docs/README.md"
        ));
        assert!(memory.chunk_metadata.is_some());
        let meta = memory.chunk_metadata.unwrap();
        assert_eq!(meta.file_path, "docs/README.md");
        assert_eq!(meta.chunk_index, 0);
        assert_eq!(meta.total_chunks, 3);
    }

    #[tokio::test]
    async fn test_capture_empty_content_fails() {
        let (service, store, _dir) = test_service().await;
        let count_before = store.count().expect("count");

        let result = service
            .capture_hook_description(
                "".to_string(),
                HookType::SessionStart,
                "sess-empty".to_string(),
                None,
            )
            .await;

        assert!(matches!(result, Err(CaptureError::EmptyContent)));
        assert_eq!(store.count().expect("count"), count_before);
    }

    #[tokio::test]
    async fn test_capture_whitespace_only_fails() {
        let (service, store, _dir) = test_service().await;
        let count_before = store.count().expect("count");

        let result = service
            .capture_hook_description(
                "   \n\t  \r\n   ".to_string(),
                HookType::UserPromptSubmit,
                "sess-ws".to_string(),
                None,
            )
            .await;

        assert!(matches!(result, Err(CaptureError::EmptyContent)));
        assert_eq!(store.count().expect("count"), count_before);
    }

    #[tokio::test]
    async fn test_capture_content_at_max_length_succeeds() {
        let (service, _store, _dir) = test_service().await;

        let content = "x".repeat(MAX_CONTENT_LENGTH);
        let result = service
            .capture_hook_description(
                content,
                HookType::Stop,
                "sess-boundary".to_string(),
                None,
            )
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_capture_content_over_max_length_fails() {
        let (service, store, _dir) = test_service().await;
        let count_before = store.count().expect("count");

        let content = "x".repeat(MAX_CONTENT_LENGTH + 1);
        let result = service
            .capture_hook_description(
                content,
                HookType::Stop,
                "sess-toolong".to_string(),
                None,
            )
            .await;

        assert!(matches!(
            result,
            Err(CaptureError::ContentTooLong { max: 10000, actual: 10001 })
        ));
        assert_eq!(store.count().expect("count"), count_before);
    }

    #[tokio::test]
    async fn test_embedding_error_propagates() {
        let dir = tempdir().expect("create temp dir");
        let store = Arc::new(MemoryStore::new(dir.path()).expect("create store"));
        let embedder = Arc::new(FailingEmbeddingProvider::unavailable());
        let service = MemoryCaptureService::new(store.clone(), embedder);

        let count_before = store.count().expect("count");

        let result = service
            .capture_hook_description(
                "Valid content".to_string(),
                HookType::SessionStart,
                "sess-fail".to_string(),
                None,
            )
            .await;

        assert!(matches!(
            result,
            Err(CaptureError::EmbeddingFailed(EmbedderError::Unavailable))
        ));
        assert_eq!(store.count().expect("count"), count_before);
    }

    #[tokio::test]
    async fn test_memory_indexed_by_session() {
        let (service, store, _dir) = test_service().await;

        let session_id = "sess-index-test";

        // Capture multiple memories for same session
        let uuid1 = service
            .capture_hook_description(
                "First memory".to_string(),
                HookType::SessionStart,
                session_id.to_string(),
                None,
            )
            .await
            .expect("capture 1");

        let uuid2 = service
            .capture_claude_response(
                "Second memory".to_string(),
                ResponseType::StopResponse,
                session_id.to_string(),
            )
            .await
            .expect("capture 2");

        // Verify session index
        let session_memories = store
            .get_by_session(session_id)
            .expect("get by session");

        assert_eq!(session_memories.len(), 2);
        let ids: Vec&lt;_&gt; = session_memories.iter().map(|m| m.id).collect();
        assert!(ids.contains(&amp;uuid1));
        assert!(ids.contains(&amp;uuid2));
    }

    #[tokio::test]
    async fn test_multiple_sessions_isolated() {
        let (service, store, _dir) = test_service().await;

        // Capture to different sessions
        service
            .capture_hook_description(
                "Session A memory".to_string(),
                HookType::SessionStart,
                "sess-A".to_string(),
                None,
            )
            .await
            .expect("capture A");

        service
            .capture_hook_description(
                "Session B memory".to_string(),
                HookType::SessionStart,
                "sess-B".to_string(),
                None,
            )
            .await
            .expect("capture B");

        // Verify isolation
        let a_memories = store.get_by_session("sess-A").expect("get A");
        let b_memories = store.get_by_session("sess-B").expect("get B");

        assert_eq!(a_memories.len(), 1);
        assert_eq!(b_memories.len(), 1);
        assert_eq!(a_memories[0].content, "Session A memory");
        assert_eq!(b_memories[0].content, "Session B memory");
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/memory/capture.rs">MemoryCaptureService implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/memory/mod.rs">
    Add: pub mod capture;
    Add to re-exports: pub use capture::{MemoryCaptureService, CaptureError, EmbedderError, EmbeddingProvider};
    Add (test-utils only): pub use capture::{TestEmbeddingProvider, FailingEmbeddingProvider};
  </file>
</files_to_modify>

<validation_criteria>
  <criterion id="VC-001">capture_hook_description creates memory with MemorySource::HookDescription</criterion>
  <criterion id="VC-002">capture_claude_response creates memory with MemorySource::ClaudeResponse</criterion>
  <criterion id="VC-003">capture_md_chunk creates memory with MemorySource::MDFileChunk and chunk_metadata populated</criterion>
  <criterion id="VC-004">Empty content ("") returns CaptureError::EmptyContent</criterion>
  <criterion id="VC-005">Whitespace-only content returns CaptureError::EmptyContent</criterion>
  <criterion id="VC-006">Content at MAX_CONTENT_LENGTH (10,000) succeeds</criterion>
  <criterion id="VC-007">Content over MAX_CONTENT_LENGTH returns CaptureError::ContentTooLong</criterion>
  <criterion id="VC-008">EmbedderError propagates as CaptureError::EmbeddingFailed</criterion>
  <criterion id="VC-009">StorageError propagates as CaptureError::StorageFailed</criterion>
  <criterion id="VC-010">Memory persists to RocksDB (verified via MemoryStore::get)</criterion>
  <criterion id="VC-011">Memory indexed by session (verified via MemoryStore::get_by_session)</criterion>
  <criterion id="VC-012">No partial storage on error (count unchanged)</criterion>
  <criterion id="VC-013">All tests pass: cargo test --package context-graph-core capture</criterion>
  <criterion id="VC-014">Clippy passes: cargo clippy --package context-graph-core -- -D warnings</criterion>
</validation_criteria>

<test_commands>
  <command description="Run capture module tests">cargo test --package context-graph-core capture -- --nocapture</command>
  <command description="Run with test-utils feature">cargo test --package context-graph-core --features test-utils capture</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
  <command description="Run clippy">cargo clippy --package context-graph-core -- -D warnings</command>
  <command description="Run all memory tests">cargo test --package context-graph-core memory</command>
</test_commands>
</task_spec>
```

## Execution Checklist

### Pre-Implementation Verification
- [ ] Read store.rs to verify MemoryStore API matches expected signatures
- [ ] Read chunker.rs to verify TextChunk structure matches expected fields
- [ ] Read mod.rs to verify Memory struct and ChunkMetadata fields
- [ ] Read source.rs to verify MemorySource, HookType, ResponseType variants
- [ ] Verify TeleologicalArray import path: `crate::types::fingerprint::TeleologicalArray`
- [ ] Verify async-trait is in Cargo.toml dependencies

### Implementation Steps
- [ ] Create capture.rs in crates/context-graph-core/src/memory/
- [ ] Implement EmbedderError enum with thiserror
- [ ] Implement CaptureError enum with thiserror
- [ ] Implement EmbeddingProvider trait with async_trait
- [ ] Implement MemoryCaptureService struct
- [ ] Implement MemoryCaptureService::new()
- [ ] Implement capture_hook_description() with tracing instrumentation
- [ ] Implement capture_claude_response() with tracing instrumentation
- [ ] Implement capture_md_chunk() with tracing instrumentation
- [ ] Implement internal capture_memory() with full validation
- [ ] Implement TestEmbeddingProvider (test-utils feature)
- [ ] Implement FailingEmbeddingProvider (test-utils feature)
- [ ] Add pub mod capture to memory/mod.rs
- [ ] Add re-exports to memory/mod.rs

### Test Implementation
- [ ] test_capture_hook_description_success
- [ ] test_capture_claude_response_success
- [ ] test_capture_md_chunk_success
- [ ] test_capture_empty_content_fails
- [ ] test_capture_whitespace_only_fails
- [ ] test_capture_content_at_max_length_succeeds
- [ ] test_capture_content_over_max_length_fails
- [ ] test_embedding_error_propagates
- [ ] test_memory_indexed_by_session
- [ ] test_multiple_sessions_isolated

### Post-Implementation Verification
- [ ] cargo check --package context-graph-core (no errors)
- [ ] cargo clippy --package context-graph-core -- -D warnings (no warnings)
- [ ] cargo test --package context-graph-core capture (all tests pass)
- [ ] Verify memories persist: MemoryStore::get(uuid) returns stored data
- [ ] Verify session index: MemoryStore::get_by_session returns memories
- [ ] Verify no partial storage on error: count unchanged after failures

### Evidence of Success
- [ ] INFO logs appear for each capture operation
- [ ] DEBUG logs show embedding completion with size
- [ ] ERROR logs appear for validation failures
- [ ] All 14 validation criteria pass

## Notes for Implementing Agent

1. **ChunkMetadata Mapping**: TextChunk uses `chunker::ChunkMetadata`, Memory uses `memory::ChunkMetadata`. They have identical fields - copy values directly.

2. **TeleologicalArray**: Import from `crate::types::fingerprint::TeleologicalArray`. Use `SemanticFingerprint::zeroed()` ONLY in tests with `test-utils` feature.

3. **Error Handling**: Use `?` operator for all fallible operations. Never use `.unwrap()` or `.expect()` in the service code.

4. **Tracing**: Use `tracing` crate with `instrument` attribute and `info!`/`debug!`/`error!` macros. Field names should match the log patterns in evidence_of_success.

5. **Test Isolation**: Each test creates its own tempdir and MemoryStore. Tests are independent and can run in parallel.

6. **Feature Gates**: `TestEmbeddingProvider` and `FailingEmbeddingProvider` are gated behind `#[cfg(any(test, feature = "test-utils"))]`.
