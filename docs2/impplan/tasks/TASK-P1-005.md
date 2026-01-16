# TASK-P1-005: MemoryStore with RocksDB

```xml
<task_spec id="TASK-P1-005" version="2.0">
<metadata>
  <title>MemoryStore with RocksDB</title>
  <status>COMPLETE</status>
  <layer>logic</layer>
  <sequence>10</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-07</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETE">TASK-P1-001</task_ref>
    <task_ref status="COMPLETE">TASK-P1-003</task_ref>
    <task_ref status="COMPLETE">TASK-P1-004</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <last_audit>2026-01-16</last_audit>
</metadata>

<context>
Implements the MemoryStore component for persisting Memory structs to RocksDB.
Provides CRUD operations and indexing by session_id for the Phase 1 Memory Capture system.

This task creates storage infrastructure in context-graph-core for Memory persistence.
Note: context-graph-storage crate exists but handles TeleologicalFingerprint/MemoryNode
storage (different domain). This task creates Memory-specific storage within core crate.

CRITICAL: NO BACKWARDS COMPATIBILITY. Fail fast on errors. No workarounds or fallbacks.
</context>

<current_state_audit date="2026-01-16">
  <verified_facts>
    <fact>Memory struct EXISTS at crates/context-graph-core/src/memory/mod.rs:86-113</fact>
    <fact>Memory has fields: id (Uuid), content (String), source (MemorySource), created_at (DateTime), session_id (String), teleological_array (TeleologicalArray), chunk_metadata (Option), word_count (u32)</fact>
    <fact>MemorySource enum EXISTS at crates/context-graph-core/src/memory/source.rs with HookDescription, ClaudeResponse, MDFileChunk variants</fact>
    <fact>Session struct EXISTS at crates/context-graph-core/src/memory/session.rs:45-82</fact>
    <fact>TextChunker EXISTS at crates/context-graph-core/src/memory/chunker.rs (TASK-P1-004 COMPLETE)</fact>
    <fact>bincode = "1.3" IS in workspace Cargo.toml line 37</fact>
    <fact>bincode.workspace = true IS in context-graph-core/Cargo.toml line 23</fact>
    <fact>rocksdb = "0.22" is used in context-graph-storage/Cargo.toml but NOT in context-graph-core</fact>
    <fact>store.rs does NOT exist in crates/context-graph-core/src/memory/</fact>
    <fact>Memory serialization with bincode WORKS - tested in mod.rs:499-507</fact>
  </verified_facts>

  <files_that_exist>
    <file path="crates/context-graph-core/src/memory/mod.rs" lines="649">Memory struct, ChunkMetadata, re-exports</file>
    <file path="crates/context-graph-core/src/memory/source.rs" lines="265">MemorySource, HookType, ResponseType</file>
    <file path="crates/context-graph-core/src/memory/session.rs" lines="642">Session, SessionStatus</file>
    <file path="crates/context-graph-core/src/memory/chunker.rs" lines="1309">TextChunker, TextChunk, ChunkerError</file>
    <file path="crates/context-graph-core/Cargo.toml" lines="74">Dependencies - HAS bincode, MISSING rocksdb</file>
  </files_that_exist>

  <files_to_create>
    <file path="crates/context-graph-core/src/memory/store.rs">MemoryStore implementation - THIS TASK</file>
  </files_to_create>

  <dependencies_status>
    <dep name="bincode" version="1.3" status="AVAILABLE">workspace = true, already in Cargo.toml</dep>
    <dep name="rocksdb" version="0.22" status="MUST_ADD">NOT in core Cargo.toml, add directly (not workspace)</dep>
    <dep name="thiserror" version="1.0" status="AVAILABLE">workspace = true, already in Cargo.toml</dep>
    <dep name="uuid" version="1.6" status="AVAILABLE">workspace = true, already in Cargo.toml</dep>
  </dependencies_status>
</current_state_audit>

<input_context_files>
  <file purpose="memory_struct" path="crates/context-graph-core/src/memory/mod.rs" lines="86-113">Memory struct definition</file>
  <file purpose="memory_source" path="crates/context-graph-core/src/memory/source.rs">MemorySource enum</file>
  <file purpose="tech_spec" path="docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md" section="component_contracts.MemoryStore">Component contract</file>
  <file purpose="tech_spec" path="docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md" section="database_schema">Column families spec</file>
  <file purpose="tech_spec" path="docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md" section="error_types.StorageError">Error enum spec</file>
  <file purpose="reference_pattern" path="crates/context-graph-storage/src/rocksdb_backend/core.rs">RocksDB initialization pattern</file>
  <file purpose="reference_pattern" path="crates/context-graph-storage/src/rocksdb_backend/error.rs">StorageError pattern</file>
</input_context_files>

<prerequisites>
  <check status="PASS">TASK-P1-001 complete - Memory struct exists at memory/mod.rs:86-113</check>
  <check status="PASS">TASK-P1-003 complete - Session struct exists at memory/session.rs</check>
  <check status="PASS">bincode crate available - workspace = true in Cargo.toml line 23</check>
  <check status="MUST_DO">rocksdb crate - ADD to Cargo.toml: rocksdb = "0.22"</check>
</prerequisites>

<scope>
  <in_scope>
    <item>Add rocksdb = "0.22" to crates/context-graph-core/Cargo.toml</item>
    <item>Create store.rs in crates/context-graph-core/src/memory/</item>
    <item>Implement StorageError enum with thiserror</item>
    <item>Implement MemoryStore struct wrapping Arc&lt;DB&gt;</item>
    <item>Implement new(path) - Initialize RocksDB with 2 column families</item>
    <item>Implement store(memory) - Persist Memory + update session index</item>
    <item>Implement get(id) - Retrieve Memory by UUID</item>
    <item>Implement get_by_session(session_id) - Retrieve all memories for session</item>
    <item>Implement count() - Return total memory count</item>
    <item>Update mod.rs to export store module</item>
    <item>Write integration tests using tempfile for temp directories</item>
  </in_scope>
  <out_of_scope>
    <item>Session storage/SessionStore (TASK-P1-006)</item>
    <item>File hash tracking (part of MDFileWatcher TASK-P1-008)</item>
    <item>Embedding generation (handled by MultiArrayProvider - Phase 2)</item>
    <item>context-graph-storage crate (separate domain - TeleologicalFingerprint storage)</item>
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/memory/store.rs">
// Column family constants
const CF_MEMORIES: &amp;str = "memories";
const CF_SESSION_INDEX: &amp;str = "session_index";

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Database initialization failed: {0}")]
    InitFailed(String),
    #[error("Serialization failed: {0}")]
    SerializationFailed(String),
    #[error("Database write failed: {0}")]
    WriteFailed(String),
    #[error("Database read failed: {0}")]
    ReadFailed(String),
    #[error("Column family not found: {0}")]
    ColumnFamilyNotFound(String),
}

pub struct MemoryStore {
    db: Arc&lt;DB&gt;,
}

impl MemoryStore {
    /// Initialize RocksDB with column families.
    /// FAILS FAST if path invalid or DB cannot be opened.
    pub fn new(path: &amp;Path) -> Result&lt;Self, StorageError&gt;;

    /// Store a Memory and update session index.
    /// NOT async - RocksDB ops are sync, wrap in spawn_blocking if needed.
    pub fn store(&amp;self, memory: &amp;Memory) -> Result&lt;(), StorageError&gt;;

    /// Get Memory by UUID. Returns None if not found.
    pub fn get(&amp;self, id: Uuid) -> Result&lt;Option&lt;Memory&gt;, StorageError&gt;;

    /// Get all memories for a session.
    pub fn get_by_session(&amp;self, session_id: &amp;str) -> Result&lt;Vec&lt;Memory&gt;, StorageError&gt;;

    /// Count total memories in store.
    pub fn count(&amp;self) -> Result&lt;u64, StorageError&gt;;

    /// Delete a memory by ID (for testing/maintenance).
    pub fn delete(&amp;self, id: Uuid) -> Result&lt;bool, StorageError&gt;;
}
    </signature>
  </signatures>

  <constraints>
    <constraint id="C1">Use bincode for serialization (fast, compact) - already tested in mod.rs</constraint>
    <constraint id="C2">Column families: "memories" (primary), "session_index" (secondary)</constraint>
    <constraint id="C3">Key for memories CF: UUID bytes (16 bytes) via id.as_bytes()</constraint>
    <constraint id="C4">Key for session_index CF: session_id string bytes</constraint>
    <constraint id="C5">Value for session_index: bincode-serialized Vec&lt;Uuid&gt;</constraint>
    <constraint id="C6">FAIL FAST on any write error - no partial writes allowed</constraint>
    <constraint id="C7">All operations atomic within single memory operation</constraint>
    <constraint id="C8">NO async methods - RocksDB is sync, caller wraps in spawn_blocking</constraint>
    <constraint id="C9">NO .unwrap() in library code - propagate all errors</constraint>
    <constraint id="C10">rocksdb::Options must set create_if_missing(true) and create_missing_column_families(true)</constraint>
  </constraints>

  <verification>
    <verify id="V1">Store and retrieve round-trips correctly (content, source, session_id match)</verify>
    <verify id="V2">Session index queries return correct memories</verify>
    <verify id="V3">Count returns accurate number after store/delete operations</verify>
    <verify id="V4">Errors propagate correctly (no panics, no silent failures)</verify>
    <verify id="V5">Multiple stores to same session accumulate in index</verify>
    <verify id="V6">Get on non-existent ID returns Ok(None), not error</verify>
    <verify id="V7">Delete removes from both memories CF and session index</verify>
  </verification>
</definition_of_done>

<implementation_steps>
  <step order="1" description="Add rocksdb dependency">
    <action>Edit crates/context-graph-core/Cargo.toml</action>
    <details>Add after line 23 (bincode.workspace = true):
rocksdb = "0.22"</details>
    <verification>Run: cargo check --package context-graph-core</verification>
  </step>

  <step order="2" description="Create store.rs with StorageError">
    <action>Create file crates/context-graph-core/src/memory/store.rs</action>
    <details>Implement StorageError enum using thiserror with variants:
- InitFailed(String) - DB open failures
- SerializationFailed(String) - bincode errors
- WriteFailed(String) - RocksDB write errors
- ReadFailed(String) - RocksDB read errors
- ColumnFamilyNotFound(String) - CF handle missing</details>
    <verification>cargo check --package context-graph-core</verification>
  </step>

  <step order="3" description="Implement MemoryStore::new">
    <action>Implement database initialization</action>
    <details>
1. Create Options with create_if_missing(true), create_missing_column_families(true)
2. Create ColumnFamilyDescriptor for CF_MEMORIES and CF_SESSION_INDEX
3. Call DB::open_cf_descriptors(&amp;opts, path, cf_descriptors)
4. Map RocksDB error to StorageError::InitFailed
5. Return Self { db: Arc::new(db) }</details>
    <verification>cargo test --package context-graph-core store::tests::test_new_creates_db</verification>
  </step>

  <step order="4" description="Implement MemoryStore::store">
    <action>Implement memory persistence with index update</action>
    <details>
1. Get CF handles for memories and session_index (fail if missing)
2. Serialize memory via bincode::serialize - map error to SerializationFailed
3. Put to memories CF with key = memory.id.as_bytes()
4. Read existing session index (or empty Vec if not exists)
5. Append memory.id to Vec (avoid duplicates)
6. Serialize and put updated index
7. All writes must succeed or return error (no partial state)</details>
    <verification>cargo test --package context-graph-core store::tests::test_store_and_get</verification>
  </step>

  <step order="5" description="Implement MemoryStore::get">
    <action>Implement memory retrieval by UUID</action>
    <details>
1. Get CF handle for memories
2. Call db.get_cf with key = id.as_bytes()
3. If None returned, return Ok(None)
4. Deserialize via bincode::deserialize
5. Return Ok(Some(memory))</details>
    <verification>cargo test --package context-graph-core store::tests::test_get_nonexistent</verification>
  </step>

  <step order="6" description="Implement MemoryStore::get_by_session">
    <action>Implement session-based retrieval</action>
    <details>
1. Get CF handle for session_index
2. Read index with key = session_id.as_bytes()
3. If None, return Ok(empty vec)
4. Deserialize Vec&lt;Uuid&gt; from index
5. For each UUID, call self.get(id)
6. Collect Some results into vec (skip None - orphaned references)
7. Return collected memories</details>
    <verification>cargo test --package context-graph-core store::tests::test_get_by_session</verification>
  </step>

  <step order="7" description="Implement MemoryStore::count">
    <action>Implement count via iterator</action>
    <details>
1. Get CF handle for memories
2. Create iterator with IteratorMode::Start
3. Count all entries
4. Return count</details>
    <verification>cargo test --package context-graph-core store::tests::test_count</verification>
  </step>

  <step order="8" description="Implement MemoryStore::delete">
    <action>Implement deletion with index cleanup</action>
    <details>
1. Get memory first to find session_id (return Ok(false) if not found)
2. Delete from memories CF
3. Update session_index: read, remove UUID, write back
4. Return Ok(true) on success</details>
    <verification>cargo test --package context-graph-core store::tests::test_delete</verification>
  </step>

  <step order="9" description="Update mod.rs exports">
    <action>Edit crates/context-graph-core/src/memory/mod.rs</action>
    <details>After line 36 (pub mod source;), add:
pub mod store;

After line 40 (pub use source::...), add:
pub use store::{MemoryStore, StorageError};</details>
    <verification>cargo check --package context-graph-core</verification>
  </step>

  <step order="10" description="Write comprehensive tests">
    <action>Add tests in store.rs</action>
    <details>Tests must use real RocksDB with tempfile::tempdir(), NOT mocks.
Test cases:
- test_new_creates_db: Verify DB opens and CFs exist
- test_store_and_get: Round-trip memory storage
- test_get_nonexistent: Returns Ok(None) for missing ID
- test_get_by_session: Multiple memories for same session
- test_get_by_session_empty: Empty vec for unknown session
- test_count: Accurate count after operations
- test_delete: Removes from both CFs
- test_multiple_sessions: Isolation between sessions
- test_store_duplicate_id: Second store overwrites (idempotent)
- test_serialization_roundtrip: Full Memory with all fields</details>
    <verification>cargo test --package context-graph-core --features test-utils store</verification>
  </step>
</implementation_steps>

<full_state_verification>
  <description>
    After completing implementation, you MUST verify the actual state of the system.
    Do NOT rely on function return values alone.
  </description>

  <source_of_truth>
    <truth id="SOT1">RocksDB database files on disk at the specified path</truth>
    <truth id="SOT2">Column families "memories" and "session_index" exist and are readable</truth>
    <truth id="SOT3">Data persists across MemoryStore instance recreations</truth>
  </source_of_truth>

  <verification_protocol>
    <step order="1" description="Verify DB Creation">
      <action>After new(), check path exists and contains RocksDB files</action>
      <command>ls -la {db_path}</command>
      <expected>Directory contains: CURRENT, MANIFEST-*, OPTIONS-*, *.sst files</expected>
    </step>

    <step order="2" description="Verify Store Persistence">
      <action>Store a memory, drop MemoryStore, reopen, verify data exists</action>
      <code>
let path = tempdir().unwrap();
let store1 = MemoryStore::new(path.path()).unwrap();
store1.store(&amp;memory).unwrap();
drop(store1);

let store2 = MemoryStore::new(path.path()).unwrap();
let retrieved = store2.get(memory.id).unwrap();
assert!(retrieved.is_some());
assert_eq!(retrieved.unwrap().content, memory.content);
      </code>
    </step>

    <step order="3" description="Verify Session Index">
      <action>Store multiple memories to same session, verify index accumulates</action>
      <code>
store.store(&amp;mem1).unwrap(); // session "test-session"
store.store(&amp;mem2).unwrap(); // session "test-session"
store.store(&amp;mem3).unwrap(); // session "other-session"

let test_mems = store.get_by_session("test-session").unwrap();
assert_eq!(test_mems.len(), 2);
assert!(test_mems.iter().any(|m| m.id == mem1.id));
assert!(test_mems.iter().any(|m| m.id == mem2.id));

let other_mems = store.get_by_session("other-session").unwrap();
assert_eq!(other_mems.len(), 1);
      </code>
    </step>

    <step order="4" description="Verify Count Accuracy">
      <action>Count must match actual stored entries</action>
      <code>
assert_eq!(store.count().unwrap(), 0);
store.store(&amp;mem1).unwrap();
assert_eq!(store.count().unwrap(), 1);
store.store(&amp;mem2).unwrap();
assert_eq!(store.count().unwrap(), 2);
store.delete(mem1.id).unwrap();
assert_eq!(store.count().unwrap(), 1);
      </code>
    </step>
  </verification_protocol>

  <edge_case_audit>
    <case id="EC1" description="Empty database">
      <before>Fresh MemoryStore with no data</before>
      <action>get(random_uuid), get_by_session("unknown"), count()</action>
      <expected>Ok(None), Ok(vec![]), Ok(0)</expected>
      <after>Database state unchanged</after>
    </case>

    <case id="EC2" description="Very large content">
      <before>Empty store</before>
      <action>Store Memory with 10,000 char content (MAX_CONTENT_LENGTH)</action>
      <expected>Store succeeds, retrieval returns exact content</expected>
      <after>count() == 1, retrieved.content.len() == 10000</after>
    </case>

    <case id="EC3" description="Special characters in session_id">
      <before>Empty store</before>
      <action>Store with session_id = "test/session:with-special_chars.123"</action>
      <expected>Store succeeds, get_by_session returns memory</expected>
      <after>Verify session_id preserved exactly</after>
    </case>

    <case id="EC4" description="Delete non-existent">
      <before>Empty store</before>
      <action>delete(random_uuid)</action>
      <expected>Ok(false) - not found, no error</expected>
      <after>count() == 0</after>
    </case>

    <case id="EC5" description="Concurrent access (basic)">
      <before>Store with one memory</before>
      <action>Clone Arc, access from multiple threads</action>
      <expected>No panics, data consistency maintained</expected>
      <after>All reads return consistent data</after>
    </case>
  </edge_case_audit>

  <evidence_of_success>
    <log description="After full test run, print verification summary">
      <format>
=== FULL STATE VERIFICATION ===
Database path: {path}
Column families verified: memories, session_index
Total memories stored: {count}
Sessions tracked: {session_count}
Persistence test: PASS/FAIL
Index consistency: PASS/FAIL
Edge cases passed: {n}/5
      </format>
    </log>
  </evidence_of_success>
</full_state_verification>

<manual_testing_protocol>
  <synthetic_test id="MT1" description="Basic CRUD">
    <input>
      Memory {
        content: "Test memory for CRUD verification",
        source: HookDescription { hook_type: SessionStart, tool_name: None },
        session_id: "manual-test-session-001",
        teleological_array: test_fingerprint(), // from test-utils
      }
    </input>
    <steps>
      1. Create MemoryStore at temp path
      2. Store memory
      3. Get by ID - verify all fields match
      4. Get by session - verify returns vec with memory
      5. Count - verify returns 1
      6. Delete - verify returns true
      7. Count - verify returns 0
      8. Get by ID - verify returns None
    </steps>
    <expected_db_state>After step 7: memories CF empty, session_index["manual-test-session-001"] = []</expected_db_state>
  </synthetic_test>

  <synthetic_test id="MT2" description="Multi-session isolation">
    <input>
      mem_a1: session_id = "session-A", content = "A1"
      mem_a2: session_id = "session-A", content = "A2"
      mem_b1: session_id = "session-B", content = "B1"
    </input>
    <steps>
      1. Store all three memories
      2. get_by_session("session-A") - expect [mem_a1, mem_a2]
      3. get_by_session("session-B") - expect [mem_b1]
      4. get_by_session("session-C") - expect []
      5. Delete mem_a1
      6. get_by_session("session-A") - expect [mem_a2] only
    </steps>
    <expected_db_state>
      memories CF: {mem_a2.id, mem_b1.id}
      session_index["session-A"]: [mem_a2.id]
      session_index["session-B"]: [mem_b1.id]
    </expected_db_state>
  </synthetic_test>

  <synthetic_test id="MT3" description="Persistence across restarts">
    <input>
      mem1: content = "Persist test", session_id = "persist-session"
    </input>
    <steps>
      1. Create MemoryStore at known temp path
      2. Store mem1
      3. Drop MemoryStore (closes DB)
      4. Create new MemoryStore at SAME path
      5. get(mem1.id) - must return mem1
      6. count() - must return 1
      7. get_by_session("persist-session") - must return [mem1]
    </steps>
    <expected_db_state>Data persists on disk, survives process restart</expected_db_state>
  </synthetic_test>
</manual_testing_protocol>

<test_commands>
  <command description="Check compilation after adding rocksdb">cargo check --package context-graph-core</command>
  <command description="Run store module tests">cargo test --package context-graph-core --features test-utils store -- --nocapture</command>
  <command description="Run all memory module tests">cargo test --package context-graph-core --features test-utils memory -- --nocapture</command>
  <command description="Check for clippy warnings">cargo clippy --package context-graph-core -- -D warnings</command>
</test_commands>

<error_handling_requirements>
  <requirement id="ERR1">ALL RocksDB errors must map to StorageError variants</requirement>
  <requirement id="ERR2">ALL bincode errors must map to StorageError::SerializationFailed</requirement>
  <requirement id="ERR3">NO .unwrap() or .expect() in non-test code</requirement>
  <requirement id="ERR4">Errors must include context (what operation failed, what key/id)</requirement>
  <requirement id="ERR5">Log errors with tracing::error! before returning</requirement>
  <requirement id="ERR6">Fail immediately on error - no retry logic at this layer</requirement>
</error_handling_requirements>

<anti_patterns_to_avoid>
  <anti_pattern id="AP1">DO NOT use async fn - RocksDB is synchronous</anti_pattern>
  <anti_pattern id="AP2">DO NOT use mock data in tests - use real TeleologicalArray via test-utils</anti_pattern>
  <anti_pattern id="AP3">DO NOT catch and hide errors - propagate everything</anti_pattern>
  <anti_pattern id="AP4">DO NOT create workarounds for missing column families</anti_pattern>
  <anti_pattern id="AP5">DO NOT use in-memory fallbacks if RocksDB fails</anti_pattern>
  <anti_pattern id="AP6">DO NOT batch operations that should be atomic</anti_pattern>
  <anti_pattern id="AP7">DO NOT ignore deserialization errors - they indicate corruption</anti_pattern>
</anti_patterns_to_avoid>

<constitution_compliance>
  <rule ref="AP-14">No .unwrap() in library code - use ? and Result</rule>
  <rule ref="AP-06">All memory ops through tools - MemoryStore is internal implementation</rule>
  <rule ref="rust_standards.error_handling">thiserror for library errors</rule>
  <rule ref="testing.gates.pre-commit">Tests must pass: cargo test --lib</rule>
</constitution_compliance>
</task_spec>
```

## Execution Checklist

- [ ] Add `rocksdb = "0.22"` to crates/context-graph-core/Cargo.toml (after line 23)
- [ ] Create crates/context-graph-core/src/memory/store.rs
- [ ] Implement StorageError enum with 5 variants
- [ ] Implement MemoryStore struct with Arc<DB>
- [ ] Implement new(path) with CF initialization (memories, session_index)
- [ ] Implement store(memory) with session index update
- [ ] Implement get(id) returning Option<Memory>
- [ ] Implement get_by_session(session_id)
- [ ] Implement count()
- [ ] Implement delete(id)
- [ ] Update mod.rs: add `pub mod store;` and `pub use store::{MemoryStore, StorageError};`
- [ ] Write test: test_new_creates_db
- [ ] Write test: test_store_and_get
- [ ] Write test: test_get_nonexistent
- [ ] Write test: test_get_by_session
- [ ] Write test: test_get_by_session_empty
- [ ] Write test: test_count
- [ ] Write test: test_delete
- [ ] Write test: test_persistence_across_reopen
- [ ] Write test: test_multiple_sessions_isolation
- [ ] Run: `cargo test --package context-graph-core --features test-utils store`
- [ ] Run: `cargo clippy --package context-graph-core -- -D warnings`
- [ ] Verify: Database files created on disk
- [ ] Verify: Data persists after MemoryStore drop and reopen
- [ ] Document: Print verification log showing all edge cases pass

## Quick Reference: Key File Locations

| What | Where |
|------|-------|
| Memory struct | `crates/context-graph-core/src/memory/mod.rs:86-113` |
| MemorySource | `crates/context-graph-core/src/memory/source.rs:20-46` |
| HookType | `crates/context-graph-core/src/memory/source.rs:50-62` |
| test_fingerprint() | `crates/context-graph-core/src/memory/mod.rs:299-301` (requires `--features test-utils`) |
| Cargo.toml to edit | `crates/context-graph-core/Cargo.toml` |
| Reference RocksDB pattern | `crates/context-graph-storage/src/rocksdb_backend/core.rs` |
| Reference StorageError | `crates/context-graph-storage/src/rocksdb_backend/error.rs` |

## Next Task

After completing TASK-P1-005, proceed to **TASK-P1-006: SessionManager** which will use MemoryStore for session persistence.
