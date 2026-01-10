# TASK-INTEG-001: Memory MCP Handlers

```xml
<task_spec id="TASK-INTEG-001" version="2.0">
<metadata>
  <title>Implement Memory MCP Tool Handlers</title>
  <status>todo</status>
  <layer>integration</layer>
  <sequence>21</sequence>
  <implements>
    <requirement_ref>REQ-MCP-MEMORY-01</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="done">TASK-LOGIC-008</task_ref>
    <task_ref status="todo">TASK-CORE-006</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <estimated_days>3</estimated_days>
  <last_audit>2026-01-09</last_audit>
</metadata>

<codebase_audit>
  <critical_finding severity="high">
    The task spec references INCORRECT type paths that DO NOT EXIST in the codebase.
    This section provides the ACTUAL current state as of the last audit date.
  </critical_finding>

  <actual_type_paths>
    <!-- ACTUAL paths discovered via codebase audit -->
    <type name="TeleologicalMemoryStore" location="crates/context-graph-core/src/traits/teleological_memory_store.rs">
      TRAIT - The core storage interface. Use: context_graph_core::traits::TeleologicalMemoryStore
    </type>
    <type name="TeleologicalFingerprint" location="crates/context-graph-core/src/types/fingerprint/teleological/types.rs">
      STRUCT - The atomic storage unit (NOT TeleologicalArray). Use: context_graph_core::types::TeleologicalFingerprint
    </type>
    <type name="SemanticFingerprint" location="crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs">
      STRUCT - 13-embedder semantic data. Use: context_graph_core::types::SemanticFingerprint
    </type>
    <type name="TeleologicalComparator" location="crates/context-graph-core/src/teleological/comparator.rs">
      STRUCT - Apples-to-apples comparison. Use: context_graph_core::teleological::TeleologicalComparator
    </type>
    <type name="BatchComparator" location="crates/context-graph-core/src/teleological/comparator.rs">
      STRUCT - Parallel batch comparisons. Use: context_graph_core::teleological::BatchComparator
    </type>
    <type name="PurposeVector" location="crates/context-graph-core/src/types/fingerprint/purpose.rs">
      STRUCT - 13D alignment signature. Use: context_graph_core::types::fingerprint::PurposeVector
    </type>
    <type name="JohariFingerprint" location="crates/context-graph-core/src/types/fingerprint/johari/core.rs">
      STRUCT - Per-embedder awareness. Use: context_graph_core::types::fingerprint::JohariFingerprint
    </type>
  </actual_type_paths>

  <nonexistent_types severity="critical">
    <!-- These types from the original spec DO NOT EXIST -->
    <type name="TeleologicalArrayStore">DOES NOT EXIST - Use TeleologicalMemoryStore trait</type>
    <type name="TeleologicalSearchEngine">DOES NOT EXIST - Use pipeline.rs stages instead</type>
    <type name="TeleologicalArray">DOES NOT EXIST - Use TeleologicalFingerprint</type>
    <type name="EmbedderPipeline">DOES NOT EXIST - Embedding is done via MultiArrayEmbeddingProvider trait</type>
  </nonexistent_types>

  <existing_handlers location="crates/context-graph-mcp/src/handlers/">
    <!-- Already implemented handlers (from memory.rs, ~250 lines) -->
    <handler name="handle_memory_store" method="memory/store" status="IMPLEMENTED">
      Stores TeleologicalFingerprint, computes PurposeVector, returns UUID
    </handler>
    <handler name="handle_memory_retrieve" method="memory/retrieve" status="IMPLEMENTED">
      Retrieves by UUID, returns full TeleologicalFingerprint
    </handler>
    <handler name="handle_memory_search" method="memory/search" status="IMPLEMENTED">
      Uses TeleologicalSearchOptions, returns TeleologicalSearchResult[]
    </handler>
    <handler name="handle_memory_delete" method="memory/delete" status="IMPLEMENTED">
      Soft/hard delete by UUID
    </handler>
  </existing_handlers>

  <missing_handlers>
    <!-- Handlers that need implementation per this task -->
    <handler name="handle_inject" method="memory/inject">
      MISSING - Must wrap store with content → 13-embedder generation
    </handler>
    <handler name="handle_inject_batch" method="memory/inject_batch">
      MISSING - Batch injection with parallel embedding
    </handler>
    <handler name="handle_search_multi_perspective" method="memory/search_multi_perspective">
      MISSING - Multi-embedder perspective search with RRF fusion
    </handler>
    <handler name="handle_compare" method="memory/compare">
      MISSING - Single pair comparison using TeleologicalComparator
    </handler>
    <handler name="handle_batch_compare" method="memory/batch_compare">
      MISSING - 1-to-N comparison using BatchComparator
    </handler>
    <handler name="handle_similarity_matrix" method="memory/similarity_matrix">
      MISSING - N×N matrix using BatchComparator::compare_all_pairs
    </handler>
  </missing_handlers>

  <related_implementations>
    <!-- Key modules that already exist and should be used -->
    <module path="crates/context-graph-storage/src/teleological/search/pipeline.rs" lines="1935">
      5-stage retrieval pipeline (SPLADE→Matryoshka→RRF→Alignment→MaxSim)
      COMPLETED in TASK-LOGIC-008
    </module>
    <module path="crates/context-graph-storage/src/teleological/rocksdb_store.rs">
      RocksDB implementation of TeleologicalMemoryStore trait
    </module>
    <module path="crates/context-graph-core/src/traits/multi_array_embedding.rs">
      MultiArrayEmbeddingProvider trait for 13-embedder generation
    </module>
    <module path="crates/context-graph-mcp/src/handlers/teleological.rs">
      Existing teleological handlers (TELEO-H1 through TELEO-H5)
    </module>
    <module path="crates/context-graph-mcp/src/protocol.rs">
      JSON-RPC types, error codes (STORAGE_ERROR=-32004, EMBEDDING_ERROR=-32005, etc.)
    </module>
  </related_implementations>
</codebase_audit>

<context>
Memory MCP handlers provide the JSON-RPC interface for memory injection, search, and
comparison operations. All handlers work with TeleologicalFingerprint as the atomic unit.

TERMINOLOGY CORRECTION: The codebase uses "TeleologicalFingerprint" NOT "TeleologicalArray".
The fingerprint contains: SemanticFingerprint (13 embedders), PurposeVector (13D),
JohariFingerprint (per-embedder awareness), and purpose_evolution history.
</context>

<objective>
Implement MCP handlers for memory/inject, memory/inject_batch, memory/search_multi_perspective,
memory/compare, memory/batch_compare, and memory/similarity_matrix tools.

NOTE: memory/store, memory/retrieve, memory/search, memory/delete already exist.
</objective>

<rationale>
Memory handlers are the primary interface for Claude Code:
1. memory/inject triggers autonomous 13-embedder fingerprint creation
2. memory/search uses the 5-stage retrieval pipeline (TASK-LOGIC-008)
3. memory/compare enables apples-to-apples comparison via TeleologicalComparator
4. All return per-embedder breakdowns for interpretability
</rationale>

<architectural_rules>
  <!-- From constitution.yaml - MUST BE FOLLOWED -->
  <rule id="ARCH-01">TeleologicalFingerprint is atomic - all 13 embedders created together</rule>
  <rule id="ARCH-02">Apples-to-apples: E1 compares with E1, never cross-embedder</rule>
  <rule id="ARCH-03">Autonomous-first: No manual goal setting, goals emerge from data</rule>
  <rule id="AP-007">FAIL FAST: Return error immediately, no fallback values</rule>
</architectural_rules>

<input_context_files>
  <file purpose="constitution">docs2/constitution.yaml</file>
  <file purpose="task_index">docs2/refactor/specs/tasks/_index.md</file>
  <file purpose="existing_memory_handlers">crates/context-graph-mcp/src/handlers/memory.rs</file>
  <file purpose="trait_definition">crates/context-graph-core/src/traits/teleological_memory_store.rs</file>
  <file purpose="comparator">crates/context-graph-core/src/teleological/comparator.rs</file>
  <file purpose="search_pipeline">crates/context-graph-storage/src/teleological/search/pipeline.rs</file>
  <file purpose="protocol">crates/context-graph-mcp/src/protocol.rs</file>
</input_context_files>

<prerequisites>
  <check status="DONE">TASK-LOGIC-008 complete (5-stage pipeline exists at pipeline.rs:1935 lines)</check>
  <check status="TODO">TASK-CORE-006 complete (storage trait exists at teleological_memory_store.rs)</check>
  <note>TASK-CORE-006 dependency shows TODO in _index.md but TeleologicalMemoryStore trait exists.
        Verify if TASK-CORE-006 requires additional work or should be marked done.</note>
</prerequisites>

<scope>
  <in_scope>
    <item>Implement memory/inject handler (wraps store with embedding generation)</item>
    <item>Implement memory/inject_batch handler (parallel batch injection)</item>
    <item>Implement memory/search_multi_perspective handler (multi-embedder search)</item>
    <item>Implement memory/compare handler (uses TeleologicalComparator)</item>
    <item>Implement memory/batch_compare handler (uses BatchComparator)</item>
    <item>Implement memory/similarity_matrix handler (uses compare_all_pairs)</item>
    <item>JSON-RPC request/response serialization</item>
    <item>FAIL FAST error handling with proper error codes</item>
  </in_scope>
  <out_of_scope>
    <item>memory/store, memory/retrieve, memory/search, memory/delete (ALREADY EXIST)</item>
    <item>Purpose/goal handlers (TASK-INTEG-002)</item>
    <item>Consciousness handlers (TASK-INTEG-003)</item>
    <item>Hook handlers (TASK-INTEG-004)</item>
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-mcp/src/handlers/memory.rs">
      // CORRECTED: Use actual types from codebase
      use crate::protocol::{JsonRpcRequest, JsonRpcResponse, error_codes};
      use context_graph_core::traits::TeleologicalMemoryStore;
      use context_graph_core::types::{TeleologicalFingerprint, SemanticFingerprint, PurposeVector};
      use context_graph_core::teleological::{TeleologicalComparator, BatchComparator};
      use context_graph_core::traits::MultiArrayEmbeddingProvider;

      // Extend existing memory handlers module with new handlers

      /// Handle memory/inject request - generates 13-embedder fingerprint
      pub async fn handle_memory_inject(
          store: &amp;dyn TeleologicalMemoryStore,
          embedder: &amp;dyn MultiArrayEmbeddingProvider,
          params: InjectParams,
      ) -> Result&lt;JsonRpcResponse, HandlerError&gt;;

      /// Handle memory/inject_batch request - parallel batch injection
      pub async fn handle_memory_inject_batch(
          store: &amp;dyn TeleologicalMemoryStore,
          embedder: &amp;dyn MultiArrayEmbeddingProvider,
          params: InjectBatchParams,
      ) -> Result&lt;JsonRpcResponse, HandlerError&gt;;

      /// Handle memory/search_multi_perspective - multi-embedder search with RRF
      pub async fn handle_memory_search_multi_perspective(
          store: &amp;dyn TeleologicalMemoryStore,
          params: MultiPerspectiveSearchParams,
      ) -> Result&lt;JsonRpcResponse, HandlerError&gt;;

      /// Handle memory/compare - single pair comparison
      pub async fn handle_memory_compare(
          store: &amp;dyn TeleologicalMemoryStore,
          comparator: &amp;TeleologicalComparator,
          params: CompareParams,
      ) -> Result&lt;JsonRpcResponse, HandlerError&gt;;

      /// Handle memory/batch_compare - 1-to-N comparison
      pub async fn handle_memory_batch_compare(
          store: &amp;dyn TeleologicalMemoryStore,
          batch_comparator: &amp;BatchComparator,
          params: BatchCompareParams,
      ) -> Result&lt;JsonRpcResponse, HandlerError&gt;;

      /// Handle memory/similarity_matrix - N×N similarity matrix
      pub async fn handle_memory_similarity_matrix(
          store: &amp;dyn TeleologicalMemoryStore,
          batch_comparator: &amp;BatchComparator,
          params: SimilarityMatrixParams,
      ) -> Result&lt;JsonRpcResponse, HandlerError&gt;;
    </signature>
  </signatures>

  <constraints>
    <constraint>All 13 embedders generated atomically on inject (ARCH-01)</constraint>
    <constraint>Compare uses TeleologicalComparator for apples-to-apples (ARCH-02)</constraint>
    <constraint>Per-embedder scores returned when requested</constraint>
    <constraint>FAIL FAST: No fallback values, return error codes immediately (AP-007)</constraint>
    <constraint>Use error codes from protocol.rs (STORAGE_ERROR=-32004, EMBEDDING_ERROR=-32005)</constraint>
  </constraints>

  <verification>
    <command>cargo test -p context-graph-mcp handlers::memory -- --nocapture</command>
  </verification>
</definition_of_done>

<no_backwards_compatibility>
  <rule>DO NOT add fallback values for missing data</rule>
  <rule>DO NOT catch errors and return defaults</rule>
  <rule>DO NOT use unwrap_or, unwrap_or_default, or similar fallback patterns</rule>
  <rule>ALWAYS propagate errors with detailed context via error codes</rule>
  <rule>ALWAYS use ? operator to propagate Result errors</rule>
  <rule>FAIL FAST: If embedding fails, return EMBEDDING_ERROR (-32005) immediately</rule>
  <rule>FAIL FAST: If storage fails, return STORAGE_ERROR (-32004) immediately</rule>
</no_backwards_compatibility>

<full_state_verification>
  <source_of_truth>
    <primary>RocksDB TeleologicalMemoryStore (crates/context-graph-storage/src/teleological/rocksdb_store.rs)</primary>
    <column_families>
      CF_FINGERPRINTS: Primary TeleologicalFingerprint storage by UUID
      CF_MATRYOSHKA: 128D Matryoshka embeddings for ANN
      CF_INVERTED: SPLADE inverted index for Stage 1 recall
      CF_PURPOSE: PurposeVector index for alignment search
    </column_families>
    <verification_method>After each write operation, perform a separate READ to verify data exists</verification_method>
  </source_of_truth>

  <execute_and_inspect>
    <pattern>
      1. Execute the write operation (store fingerprint)
      2. Perform separate retrieve() call with returned UUID
      3. Verify retrieved fingerprint matches input
      4. Verify all 13 embedder dimensions are non-empty
      5. Verify PurposeVector has 13 dimensions
      6. Verify JohariFingerprint has per-embedder classifications
    </pattern>
    <fail_if>Retrieved data is None, empty, or doesn't match expected structure</fail_if>
  </execute_and_inspect>

  <edge_cases count="3">
    <edge_case id="1" name="Empty content injection">
      <input>content: "", memory_type: None</input>
      <expected_before>Store is empty or has N fingerprints</expected_before>
      <expected_after>Return INVALID_PARAMS (-32602) error, store unchanged</expected_after>
      <verification>Count before == count after, error code in response</verification>
    </edge_case>

    <edge_case id="2" name="UUID not found in compare">
      <input>array_a: valid_uuid, array_b: nonexistent_uuid</input>
      <expected_before>Only array_a exists in store</expected_before>
      <expected_after>Return FINGERPRINT_NOT_FOUND (-32010) error</expected_after>
      <verification>Error response contains correct code and message</verification>
    </edge_case>

    <edge_case id="3" name="Large batch injection partial failure">
      <input>100 memories, 5 with invalid content triggering embedding errors</input>
      <expected_before>Store has N fingerprints</expected_before>
      <expected_after>Store has N+95 fingerprints if parallel, error reported for 5</expected_after>
      <verification>succeeded=95, failed=5 in response, verify 95 exist in store</verification>
    </edge_case>
  </edge_cases>

  <evidence_of_success>
    <requirement>After test execution, provide log excerpt showing:</requirement>
    <item>Fingerprint UUID returned from inject</item>
    <item>Retrieved fingerprint with all 13 embedders populated</item>
    <item>Compare result with per-embedder breakdown</item>
    <item>Similarity matrix dimensions matching input count</item>
  </evidence_of_success>
</full_state_verification>

<synthetic_testing>
  <principle>Use REAL data patterns, not mock stubs</principle>

  <synthetic_fingerprint id="SYNTH-001">
    <content>The quick brown fox jumps over the lazy dog</content>
    <expected_properties>
      <e1_semantic>1024D dense vector, L2-normalized</e1_semantic>
      <e6_sparse>Non-zero indices for: quick, brown, fox, jumps, lazy, dog</e6_sparse>
      <e12_late_interaction>8 token vectors (one per word)</e12_late_interaction>
      <e13_splade>SPLADE expanded terms including synonyms</e13_splade>
    </expected_properties>
  </synthetic_fingerprint>

  <synthetic_fingerprint id="SYNTH-002">
    <content>A fast auburn fox leaps above the sleepy hound</content>
    <expected_similarity_to_SYNTH_001>
      <e1_semantic>0.7-0.9 (semantically similar)</e1_semantic>
      <e6_sparse>0.3-0.5 (different keywords)</e6_sparse>
      <e12_late_interaction>0.6-0.8 (token overlap)</e12_late_interaction>
      <overall>0.6-0.8 (weighted average)</overall>
    </expected_similarity_to_SYNTH_001>
  </synthetic_fingerprint>

  <test_matrix>
    <test name="inject_returns_all_13_embedders">
      <input>SYNTH-001 content</input>
      <assertion>Response contains teleological_summary with 13 non-empty dimensions</assertion>
    </test>
    <test name="compare_similar_content">
      <input>SYNTH-001 vs SYNTH-002</input>
      <assertion>overall_similarity in [0.6, 0.8], per_embedder has 13 scores</assertion>
    </test>
    <test name="compare_identical_content">
      <input>SYNTH-001 vs SYNTH-001</input>
      <assertion>overall_similarity >= 0.99, all per_embedder scores >= 0.99</assertion>
    </test>
    <test name="similarity_matrix_symmetric">
      <input>[SYNTH-001, SYNTH-002, SYNTH-001]</input>
      <assertion>matrix[i][j] == matrix[j][i] for all i,j</assertion>
      <assertion>matrix[0][2] >= 0.99 (identical content)</assertion>
    </test>
  </test_matrix>
</synthetic_testing>

<manual_verification>
  <step order="1">Run: cargo test -p context-graph-mcp handlers::memory -- --nocapture</step>
  <step order="2">Check test output for fingerprint UUIDs</step>
  <step order="3">Verify RocksDB directory contains expected column families</step>
  <step order="4">Use rocksdb-tools or custom script to dump CF_FINGERPRINTS and verify count</step>
  <step order="5">Verify no empty embedder vectors in stored fingerprints</step>
  <step order="6">Check that comparison results show all 13 per-embedder scores</step>
</manual_verification>

<implementation_notes>
  <note priority="critical">
    The existing memory.rs handlers (store, retrieve, search, delete) use the
    TeleologicalMemoryStore trait. The new handlers should follow the same pattern.
  </note>

  <note priority="high">
    TeleologicalComparator already implements all comparison logic including:
    - 7 SearchStrategy variants (Cosine, Euclidean, SynergyWeighted, etc.)
    - Coherence computation
    - Dominant embedder detection
    - Breakdown generation
    Use the existing comparator.rs implementation, don't rewrite.
  </note>

  <note priority="high">
    BatchComparator provides:
    - compare_one_to_many: 1-to-N parallel comparison using rayon
    - compare_all_pairs: N×N matrix computation
    - compare_above_threshold: Filtered 1-to-N
    Use these for batch_compare and similarity_matrix handlers.
  </note>

  <note priority="medium">
    MultiArrayEmbeddingProvider trait needs to be integrated for content → fingerprint.
    This is the missing piece between "inject content" and "store fingerprint".
    Check crates/context-graph-core/src/traits/multi_array_embedding.rs for interface.
  </note>

  <note priority="medium">
    Error codes are defined in protocol.rs:
    - EMBEDDING_ERROR: -32005 (embedding generation failed)
    - STORAGE_ERROR: -32004 (storage backend failure)
    - FINGERPRINT_NOT_FOUND: -32010 (UUID lookup failed)
    - INVALID_PARAMS: -32602 (bad request parameters)
  </note>
</implementation_notes>

<pseudo_code>
// crates/context-graph-mcp/src/handlers/memory.rs
// EXTEND existing file - do not replace existing handlers

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::protocol::{JsonRpcResponse, error_codes};
use context_graph_core::traits::{TeleologicalMemoryStore, MultiArrayEmbeddingProvider};
use context_graph_core::types::{TeleologicalFingerprint, SemanticFingerprint, PurposeVector};
use context_graph_core::teleological::{TeleologicalComparator, BatchComparator, ComparisonResult};

// === INJECT HANDLER ===

#[derive(Debug, Deserialize)]
pub struct InjectParams {
    pub content: String,
    pub memory_type: Option<String>,
    pub namespace: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct InjectResponse {
    pub memory_id: Uuid,
    pub embedders_generated: usize,
    pub created_at: DateTime&lt;Utc&gt;,
}

pub async fn handle_memory_inject(
    store: &amp;dyn TeleologicalMemoryStore,
    embedder: &amp;dyn MultiArrayEmbeddingProvider,
    params: InjectParams,
) -> Result&lt;JsonRpcResponse, HandlerError&gt; {
    // FAIL FAST: Validate content is non-empty
    if params.content.trim().is_empty() {
        return Err(HandlerError::new(
            error_codes::INVALID_PARAMS,
            "Content cannot be empty",
        ));
    }

    // Generate semantic fingerprint (all 13 embedders atomically)
    let semantic = embedder
        .embed_all(&amp;params.content)
        .await
        .map_err(|e| HandlerError::new(error_codes::EMBEDDING_ERROR, e.to_string()))?;

    // Compute purpose vector
    let purpose_vector = PurposeVector::from_semantic(&amp;semantic);

    // Create Johari fingerprint (default classification)
    let johari = JohariFingerprint::from_semantic(&amp;semantic);

    // Compute content hash
    let content_hash = sha256_hash(params.content.as_bytes());

    // Create teleological fingerprint
    let fingerprint = TeleologicalFingerprint::new(
        semantic,
        purpose_vector,
        johari,
        content_hash,
    );

    // Store fingerprint
    let memory_id = store
        .store(fingerprint.clone())
        .await
        .map_err(|e| HandlerError::new(error_codes::STORAGE_ERROR, e.to_string()))?;

    // EXECUTE &amp; INSPECT: Verify storage succeeded
    let retrieved = store
        .retrieve(memory_id)
        .await
        .map_err(|e| HandlerError::new(error_codes::STORAGE_ERROR, e.to_string()))?;

    if retrieved.is_none() {
        return Err(HandlerError::new(
            error_codes::STORAGE_ERROR,
            "Verification failed: stored fingerprint not retrievable",
        ));
    }

    Ok(JsonRpcResponse::success(
        None,
        serde_json::to_value(InjectResponse {
            memory_id,
            embedders_generated: 13,
            created_at: Utc::now(),
        })?,
    ))
}

// === COMPARE HANDLER ===

#[derive(Debug, Deserialize)]
pub struct CompareParams {
    pub memory_a: Uuid,
    pub memory_b: Uuid,
    #[serde(default)]
    pub include_per_embedder: bool,
}

#[derive(Debug, Serialize)]
pub struct CompareResponse {
    pub overall_similarity: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_embedder: Option&lt;[Option&lt;f32&gt;; 13]&gt;,
    pub coherence: Option&lt;f32&gt;,
    pub dominant_embedder: Option&lt;String&gt;,
}

pub async fn handle_memory_compare(
    store: &amp;dyn TeleologicalMemoryStore,
    comparator: &amp;TeleologicalComparator,
    params: CompareParams,
) -> Result&lt;JsonRpcResponse, HandlerError&gt; {
    // Retrieve both fingerprints
    let fp_a = store
        .retrieve(params.memory_a)
        .await
        .map_err(|e| HandlerError::new(error_codes::STORAGE_ERROR, e.to_string()))?
        .ok_or_else(|| HandlerError::new(
            error_codes::FINGERPRINT_NOT_FOUND,
            format!("Memory A not found: {}", params.memory_a),
        ))?;

    let fp_b = store
        .retrieve(params.memory_b)
        .await
        .map_err(|e| HandlerError::new(error_codes::STORAGE_ERROR, e.to_string()))?
        .ok_or_else(|| HandlerError::new(
            error_codes::FINGERPRINT_NOT_FOUND,
            format!("Memory B not found: {}", params.memory_b),
        ))?;

    // Compare using TeleologicalComparator (apples-to-apples per ARCH-02)
    let result = comparator
        .compare(&amp;fp_a.semantic, &amp;fp_b.semantic)
        .map_err(|e| HandlerError::new(error_codes::INTERNAL_ERROR, e.to_string()))?;

    let per_embedder = if params.include_per_embedder {
        Some(result.per_embedder)
    } else {
        None
    };

    let dominant = result.dominant_embedder.map(|e| e.name().to_string());

    Ok(JsonRpcResponse::success(
        None,
        serde_json::to_value(CompareResponse {
            overall_similarity: result.overall,
            per_embedder,
            coherence: result.coherence,
            dominant_embedder: dominant,
        })?,
    ))
}

// === BATCH COMPARE HANDLER ===

pub async fn handle_memory_batch_compare(
    store: &amp;dyn TeleologicalMemoryStore,
    batch_comparator: &amp;BatchComparator,
    params: BatchCompareParams,
) -> Result&lt;JsonRpcResponse, HandlerError&gt; {
    // Retrieve reference fingerprint
    let reference = store
        .retrieve(params.reference)
        .await
        .map_err(|e| HandlerError::new(error_codes::STORAGE_ERROR, e.to_string()))?
        .ok_or_else(|| HandlerError::new(
            error_codes::FINGERPRINT_NOT_FOUND,
            format!("Reference not found: {}", params.reference),
        ))?;

    // Retrieve all targets
    let targets: Vec&lt;SemanticFingerprint&gt; = store
        .retrieve_batch(&amp;params.targets)
        .await
        .map_err(|e| HandlerError::new(error_codes::STORAGE_ERROR, e.to_string()))?
        .into_iter()
        .filter_map(|opt| opt.map(|fp| fp.semantic))
        .collect();

    // Use BatchComparator for parallel 1-to-N comparison
    let results = batch_comparator.compare_one_to_many(&amp;reference.semantic, &amp;targets);

    // Build response with rankings
    // ...
}

// === SIMILARITY MATRIX HANDLER ===

pub async fn handle_memory_similarity_matrix(
    store: &amp;dyn TeleologicalMemoryStore,
    batch_comparator: &amp;BatchComparator,
    params: SimilarityMatrixParams,
) -> Result&lt;JsonRpcResponse, HandlerError&gt; {
    // Retrieve all fingerprints
    let fingerprints = store
        .retrieve_batch(&amp;params.memory_ids)
        .await
        .map_err(|e| HandlerError::new(error_codes::STORAGE_ERROR, e.to_string()))?;

    let semantics: Vec&lt;SemanticFingerprint&gt; = fingerprints
        .into_iter()
        .filter_map(|opt| opt.map(|fp| fp.semantic))
        .collect();

    // Use BatchComparator::compare_all_pairs for N×N matrix
    let matrix = batch_comparator.compare_all_pairs(&amp;semantics);

    Ok(JsonRpcResponse::success(
        None,
        serde_json::to_value(SimilarityMatrixResponse {
            matrix,
            memory_ids: params.memory_ids,
        })?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use context_graph_core::stubs::InMemoryTeleologicalStore;

    // Use REAL stub implementation, not mocks

    #[tokio::test]
    async fn test_inject_generates_all_13_embedders() {
        let store = InMemoryTeleologicalStore::new();
        let embedder = StubMultiArrayEmbeddingProvider::new();

        let params = InjectParams {
            content: "The quick brown fox".to_string(),
            memory_type: None,
            namespace: None,
            metadata: None,
        };

        let response = handle_memory_inject(&amp;store, &amp;embedder, params).await.unwrap();

        // EXECUTE &amp; INSPECT: Verify the stored data
        let result: InjectResponse = serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.embedders_generated, 13);

        // Retrieve and verify
        let stored = store.retrieve(result.memory_id).await.unwrap().unwrap();
        assert!(!stored.semantic.e1_semantic.is_empty());
        assert!(!stored.semantic.e2_temporal_recent.is_empty());
        // ... verify all 13
    }

    #[tokio::test]
    async fn test_compare_identical_returns_high_similarity() {
        // SYNTHETIC TEST: Known input, known output
        let store = InMemoryTeleologicalStore::new();
        let comparator = TeleologicalComparator::new();

        // Store two identical fingerprints
        let fp = create_test_fingerprint("identical content");
        let id_a = store.store(fp.clone()).await.unwrap();
        let id_b = store.store(fp.clone()).await.unwrap();

        let params = CompareParams {
            memory_a: id_a,
            memory_b: id_b,
            include_per_embedder: true,
        };

        let response = handle_memory_compare(&amp;store, &amp;comparator, params).await.unwrap();
        let result: CompareResponse = serde_json::from_value(response.result.unwrap()).unwrap();

        assert!(result.overall_similarity >= 0.99, "Identical content should have ~1.0 similarity");
    }

    #[tokio::test]
    async fn test_compare_nonexistent_fails_fast() {
        let store = InMemoryTeleologicalStore::new();
        let comparator = TeleologicalComparator::new();

        let params = CompareParams {
            memory_a: Uuid::new_v4(), // Does not exist
            memory_b: Uuid::new_v4(), // Does not exist
            include_per_embedder: false,
        };

        let result = handle_memory_compare(&amp;store, &amp;comparator, params).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.code, error_codes::FINGERPRINT_NOT_FOUND);
    }

    #[tokio::test]
    async fn test_similarity_matrix_symmetric() {
        let store = InMemoryTeleologicalStore::new();
        let batch_comparator = BatchComparator::new();

        // Store 3 fingerprints
        let fp1 = create_test_fingerprint("content one");
        let fp2 = create_test_fingerprint("content two");
        let fp3 = create_test_fingerprint("content three");

        let id1 = store.store(fp1).await.unwrap();
        let id2 = store.store(fp2).await.unwrap();
        let id3 = store.store(fp3).await.unwrap();

        let params = SimilarityMatrixParams {
            memory_ids: vec![id1, id2, id3],
        };

        let response = handle_memory_similarity_matrix(&amp;store, &amp;batch_comparator, params).await.unwrap();
        let result: SimilarityMatrixResponse = serde_json::from_value(response.result.unwrap()).unwrap();

        // Verify symmetry: matrix[i][j] == matrix[j][i]
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (result.matrix[i][j] - result.matrix[j][i]).abs() &lt; f32::EPSILON,
                    "Matrix must be symmetric"
                );
            }
        }

        // Verify diagonal is 1.0
        for i in 0..3 {
            assert!(
                (result.matrix[i][i] - 1.0).abs() &lt; 0.01,
                "Diagonal must be ~1.0 (self-similarity)"
            );
        }
    }
}
</pseudo_code>

<files_to_modify>
  <file path="crates/context-graph-mcp/src/handlers/memory.rs">
    EXTEND with new handlers: inject, inject_batch, search_multi_perspective,
    compare, batch_compare, similarity_matrix
  </file>
  <file path="crates/context-graph-mcp/src/handlers/mod.rs">
    Verify memory module is already exported (it is)
  </file>
  <file path="crates/context-graph-mcp/src/lib.rs">
    Add method dispatch routes for new memory/* tools
  </file>
</files_to_modify>

<validation_criteria>
  <criterion>memory/inject generates all 13 embedders atomically (ARCH-01)</criterion>
  <criterion>memory/compare uses TeleologicalComparator for apples-to-apples (ARCH-02)</criterion>
  <criterion>memory/batch_compare uses BatchComparator with rayon parallelism</criterion>
  <criterion>memory/similarity_matrix returns symmetric N×N matrix</criterion>
  <criterion>All handlers return proper error codes on failure (FAIL FAST)</criterion>
  <criterion>Per-embedder breakdowns returned when requested</criterion>
  <criterion>JSON-RPC responses match MCP spec format</criterion>
</validation_criteria>

<test_commands>
  <command>cargo test -p context-graph-mcp handlers::memory -- --nocapture</command>
  <command>cargo test -p context-graph-core teleological::comparator -- --nocapture</command>
</test_commands>

<research_recommendations>
  <recommendation priority="high">
    Review the ColBERT MaxSim implementation in similarity/token_level.rs to ensure
    E12 late_interaction comparison is correctly handled in TeleologicalComparator.
  </recommendation>
  <recommendation priority="medium">
    Consider adding metrics collection (via tracing) for inject latency breakdown:
    - Time for each embedder generation
    - Total embedding time
    - Storage write time
  </recommendation>
  <recommendation priority="medium">
    The constitution.yaml specifies inject_context &lt; 25ms budget. Verify that the
    parallel embedding implementation meets this target.
  </recommendation>
</research_recommendations>
</task_spec>
```
