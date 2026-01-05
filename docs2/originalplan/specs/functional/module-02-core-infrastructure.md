# Module 2: Core Infrastructure - Functional Specification

```xml
<functional_spec id="SPEC-CORE" version="1.0">
<metadata>
  <title>Core Infrastructure - Foundational Data Types, Storage, and MCP Tools</title>
  <module>Module 2</module>
  <phase>1</phase>
  <status>draft</status>
  <owner>Lead Architect</owner>
  <created>2025-12-31</created>
  <last_updated>2025-12-31</last_updated>
  <duration>4 weeks</duration>
  <related_specs>
    <spec_ref>SPEC-GHOST (Module 1) - Implements traits from REQ-GHOST-007 through REQ-GHOST-010</spec_ref>
    <spec_ref>SPEC-EMBED (Module 3) - Provides embeddings consumed by storage layer</spec_ref>
    <spec_ref>SPEC-GRAPH (Module 4) - Builds upon storage layer</spec_ref>
  </related_specs>
  <dependencies>
    <dependency>Module 1 (Ghost System) - Trait definitions and workspace structure</dependency>
  </dependencies>
</metadata>

<overview>
Core Infrastructure implements the foundational data types, memory management, and primary MCP tools that all subsequent modules depend upon. This module transforms the Ghost System's stub implementations into functional components with real persistence.

Key deliverables:
1. **Core Data Types**: MemoryNode, JohariQuadrant enum, CognitivePulse, and related domain structures
2. **Memex Storage Layer**: RocksDB-based persistent storage with WAL, Bloom filters, and LRU cache
3. **Primary MCP Tools**: inject_context, store_memory, recall_memory, get_memetic_status, set_verbosity
4. **Verbosity System**: Three-level response detail control (RawOnly, TextAndIds, FullInsights)
5. **Cognitive Pulse Headers**: Entropy, coherence, and suggested_action metadata in every response

This module establishes the data layer contract that all higher-level modules (Embedding Pipeline, Knowledge Graph, UTL Integration) will build upon. Performance target: >10K writes/sec sustained throughput.
</overview>

<!-- ============================================================================ -->
<!-- USER STORIES -->
<!-- ============================================================================ -->

<user_stories>

<story id="US-CORE-01" priority="must-have">
  <narrative>
    As an AI agent
    I want to store memory nodes with content and metadata
    So that I can persist knowledge across sessions
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-CORE-01-01">
      <given>I have content to store with a rationale</given>
      <when>I call store_memory with content, importance, and rationale</when>
      <then>The system returns a UUID for the stored memory node</then>
    </criterion>
    <criterion id="AC-CORE-01-02">
      <given>A memory has been stored</given>
      <when>The MCP server restarts</when>
      <then>The memory is still retrievable with all its metadata intact</then>
    </criterion>
    <criterion id="AC-CORE-01-03">
      <given>I provide invalid or missing rationale</given>
      <when>I call store_memory</when>
      <then>The system returns an error requiring rationale (10-500 chars)</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-CORE-02" priority="must-have">
  <narrative>
    As an AI agent
    I want to retrieve memories by semantic similarity
    So that I can access relevant context for my current task
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-CORE-02-01">
      <given>Multiple memories have been stored</given>
      <when>I call recall_memory with a query string</when>
      <then>The system returns the top-k most relevant memories</then>
    </criterion>
    <criterion id="AC-CORE-02-02">
      <given>Memories exist with varying importance scores</given>
      <when>I call recall_memory with filters</when>
      <then>Results respect the min_importance filter</then>
    </criterion>
    <criterion id="AC-CORE-02-03">
      <given>No memories match the query</given>
      <when>I call recall_memory</when>
      <then>The system returns an empty array, not an error</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-CORE-03" priority="must-have">
  <narrative>
    As an AI agent
    I want to inject context based on a query
    So that I can retrieve distilled, relevant knowledge efficiently
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-CORE-03-01">
      <given>Relevant memories exist in storage</given>
      <when>I call inject_context with a query</when>
      <then>The system returns context, tokens_used, and utl_metrics</then>
    </criterion>
    <criterion id="AC-CORE-03-02">
      <given>The retrieved context exceeds max_tokens</given>
      <when>I call inject_context with distillation_mode=auto</when>
      <then>The system applies distillation and reports compression_ratio</then>
    </criterion>
    <criterion id="AC-CORE-03-03">
      <given>Any inject_context call</given>
      <when>The response is generated</when>
      <then>The response includes Cognitive Pulse header with entropy, coherence, suggested_action</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-CORE-04" priority="must-have">
  <narrative>
    As an AI agent
    I want to check the memetic status of the system
    So that I can understand current entropy/coherence state and pending curation tasks
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-CORE-04-01">
      <given>The system has stored memories</given>
      <when>I call get_memetic_status</when>
      <then>I receive coherence_score, entropy_level, and suggested_action</then>
    </criterion>
    <criterion id="AC-CORE-04-02">
      <given>Curation tasks are pending (duplicate nodes, conflicts)</given>
      <when>I call get_memetic_status</when>
      <then>I receive curation_tasks array with task_type, target_nodes, and priority</then>
    </criterion>
    <criterion id="AC-CORE-04-03">
      <given>The system is idle and healthy</given>
      <when>I call get_memetic_status</when>
      <then>I receive suggested_action = "ready" with empty curation_tasks</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-CORE-05" priority="must-have">
  <narrative>
    As an AI agent
    I want to control the verbosity of responses
    So that I can balance token usage against detail level
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-CORE-05-01">
      <given>Verbosity is set to 0 (RawOnly)</given>
      <when>I call any retrieval tool</when>
      <then>Response contains only essential data (~100 tokens)</then>
    </criterion>
    <criterion id="AC-CORE-05-02">
      <given>Verbosity is set to 1 (TextAndIds) - default</given>
      <when>I call any retrieval tool</when>
      <then>Response contains text content and node IDs (~200 tokens)</then>
    </criterion>
    <criterion id="AC-CORE-05-03">
      <given>Verbosity is set to 2 (FullInsights)</given>
      <when>I call any retrieval tool</when>
      <then>Response includes causal_links, UTL scores, conflict_analysis (~800 tokens)</then>
    </criterion>
    <criterion id="AC-CORE-05-04">
      <given>Session has a verbosity setting</given>
      <when>A tool call includes verbosity_level parameter</when>
      <then>The per-call verbosity overrides the session default</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-CORE-06" priority="must-have">
  <narrative>
    As a developer
    I want the storage layer to survive crashes
    So that no committed data is lost
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-CORE-06-01">
      <given>A memory has been successfully stored (returned UUID)</given>
      <when>The server process is killed and restarted</when>
      <then>The memory is retrievable with identical content and metadata</then>
    </criterion>
    <criterion id="AC-CORE-06-02">
      <given>A store operation is in progress</given>
      <when>The server crashes before completion</when>
      <then>The partial write is rolled back; storage remains consistent</then>
    </criterion>
    <criterion id="AC-CORE-06-03">
      <given>Multiple concurrent write operations</given>
      <when>All complete successfully</when>
      <then>All writes are persisted in the order they were committed</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-CORE-07" priority="must-have">
  <narrative>
    As a developer
    I want the storage layer to achieve high write throughput
    So that the system can handle production workloads
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-CORE-07-01">
      <given>The storage layer is under load</given>
      <when>Benchmark runs 10K sequential write operations</when>
      <then>All writes complete in under 1 second (>10K writes/sec)</then>
    </criterion>
    <criterion id="AC-CORE-07-02">
      <given>The storage layer is under concurrent load</given>
      <when>50 concurrent writers each perform 200 writes</when>
      <then>All 10K writes complete in under 2 seconds with no errors</then>
    </criterion>
    <criterion id="AC-CORE-07-03">
      <given>The LRU cache is warm</given>
      <when>Reading recently accessed nodes</when>
      <then>Read latency is under 100 microseconds</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-CORE-08" priority="must-have">
  <narrative>
    As an AI agent
    I want every response to include a Cognitive Pulse header
    So that I can adapt my behavior based on system state
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-CORE-08-01">
      <given>Any successful MCP tool call</given>
      <when>The response is generated</when>
      <then>Response includes pulse.entropy in range [0.0, 1.0]</then>
    </criterion>
    <criterion id="AC-CORE-08-02">
      <given>Any successful MCP tool call</given>
      <when>The response is generated</when>
      <then>Response includes pulse.coherence in range [0.0, 1.0]</then>
    </criterion>
    <criterion id="AC-CORE-08-03">
      <given>Any successful MCP tool call</given>
      <when>The response is generated</when>
      <then>Response includes pulse.suggested_action from enum: consolidate, explore, clarify, curate, ready, epistemic_action, trigger_dream</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-CORE-09" priority="should-have">
  <narrative>
    As an AI agent
    I want to understand the Johari quadrant of retrieved memories
    So that I can reason about what is known vs unknown
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-CORE-09-01">
      <given>A memory is stored</given>
      <when>The system computes its initial state</when>
      <then>The memory is assigned a JohariQuadrant based on entropy and coherence</then>
    </criterion>
    <criterion id="AC-CORE-09-02">
      <given>Entropy is low (less than 0.5) and coherence is high (greater than 0.5)</given>
      <when>Johari quadrant is computed</when>
      <then>Quadrant is assigned as Open</then>
    </criterion>
    <criterion id="AC-CORE-09-03">
      <given>Entropy is high (greater than 0.5) and coherence is low (less than 0.5)</given>
      <when>Johari quadrant is computed</when>
      <then>Quadrant is assigned as Blind (discovery zone)</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-CORE-10" priority="should-have">
  <narrative>
    As a developer
    I want Bloom filters for fast existence checks
    So that we avoid unnecessary disk I/O for non-existent keys
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-CORE-10-01">
      <given>A Bloom filter is configured for the storage layer</given>
      <when>Checking for a key that definitely does not exist</when>
      <then>The Bloom filter returns false without disk access</then>
    </criterion>
    <criterion id="AC-CORE-10-02">
      <given>The Bloom filter returns "may exist"</given>
      <when>The key actually exists in storage</when>
      <then>Disk lookup confirms existence (no false negatives)</then>
    </criterion>
    <criterion id="AC-CORE-10-03">
      <given>Bloom filter false positive rate configuration</given>
      <when>Testing with 1M keys</when>
      <then>Actual false positive rate is below configured threshold (default 1%)</then>
    </criterion>
  </acceptance_criteria>
</story>

</user_stories>

<!-- ============================================================================ -->
<!-- FUNCTIONAL REQUIREMENTS -->
<!-- ============================================================================ -->

<requirements>

<!-- ==================== Core Data Types ==================== -->

<requirement id="REQ-CORE-001" story_ref="US-CORE-01, US-CORE-02" priority="must">
  <description>The system SHALL define a MemoryNode struct implementing the schema from PRD section 4.1 with all required fields.</description>
  <rationale>MemoryNode is the fundamental data unit; schema must match PRD for compatibility with all modules.</rationale>
  <implements_trait>REQ-GHOST-009 (MemoryStore trait contract)</implements_trait>
  <acceptance_criteria>
    <criterion>MemoryNode includes id: Uuid (primary key)</criterion>
    <criterion>MemoryNode includes content: String (max 65536 chars)</criterion>
    <criterion>MemoryNode includes embedding: Vec&lt;f32&gt; (1536 dimensions until Module 3 upgrades to 4096)</criterion>
    <criterion>MemoryNode includes created_at, last_accessed: DateTime&lt;Utc&gt;</criterion>
    <criterion>MemoryNode includes importance: f32 in range [0, 1]</criterion>
    <criterion>MemoryNode includes access_count: u32</criterion>
    <criterion>MemoryNode includes johari_quadrant: JohariQuadrant</criterion>
    <criterion>MemoryNode includes utl_state: UTLState (delta_s, delta_c, w_e, phi)</criterion>
    <criterion>MemoryNode includes agent_id: Option&lt;String&gt;</criterion>
    <criterion>MemoryNode includes semantic_cluster: Option&lt;Uuid&gt;</criterion>
    <criterion>MemoryNode includes metadata: HashMap&lt;String, serde_json::Value&gt;</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-002" story_ref="US-CORE-09" priority="must">
  <description>The system SHALL define a JohariQuadrant enum with four variants: Open, Blind, Hidden, Unknown.</description>
  <rationale>Johari Window is core to UTL-based knowledge classification per PRD section 2.2.</rationale>
  <acceptance_criteria>
    <criterion>JohariQuadrant::Open - low entropy (less than 0.5), high coherence (greater than 0.5)</criterion>
    <criterion>JohariQuadrant::Blind - high entropy (greater than 0.5), low coherence (less than 0.5)</criterion>
    <criterion>JohariQuadrant::Hidden - low entropy (less than 0.5), low coherence (less than 0.5)</criterion>
    <criterion>JohariQuadrant::Unknown - high entropy (greater than 0.5), high coherence (greater than 0.5)</criterion>
    <criterion>Enum derives Serialize, Deserialize, Clone, Copy, Debug, PartialEq</criterion>
    <criterion>Default value is JohariQuadrant::Unknown for new nodes</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-003" story_ref="US-CORE-08" priority="must">
  <description>The system SHALL define a CognitivePulse struct with entropy, coherence, and suggested_action fields.</description>
  <rationale>Cognitive Pulse is mandatory in every MCP response per PRD section 1.3.</rationale>
  <acceptance_criteria>
    <criterion>CognitivePulse.entropy: f32 in range [0.0, 1.0]</criterion>
    <criterion>CognitivePulse.coherence: f32 in range [0.0, 1.0]</criterion>
    <criterion>CognitivePulse.suggested_action: SuggestedAction enum</criterion>
    <criterion>Struct derives Serialize for JSON output</criterion>
    <criterion>Token cost approximately 30 tokens when serialized</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-004" story_ref="US-CORE-08" priority="must">
  <description>The system SHALL define a SuggestedAction enum with values: consolidate, explore, clarify, curate, ready, epistemic_action, trigger_dream.</description>
  <rationale>SuggestedAction guides agent behavior based on current system state per PRD section 1.3.</rationale>
  <acceptance_criteria>
    <criterion>SuggestedAction::Consolidate when entropy greater than 0.7 and coherence greater than 0.5</criterion>
    <criterion>SuggestedAction::Explore when entropy less than 0.4 and coherence less than 0.4</criterion>
    <criterion>SuggestedAction::Clarify when coherence less than 0.4 regardless of entropy</criterion>
    <criterion>SuggestedAction::Curate when curation_tasks not empty</criterion>
    <criterion>SuggestedAction::Ready when system is healthy and idle</criterion>
    <criterion>SuggestedAction::EpistemicAction when entropy greater than 0.7 and coherence greater than 0.5 (novel, adapting)</criterion>
    <criterion>SuggestedAction::TriggerDream when entropy greater than 0.7 and coherence less than 0.4 (confused)</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-005" story_ref="US-CORE-05" priority="must">
  <description>The system SHALL define a VerbosityLevel enum with three levels: RawOnly(0), TextAndIds(1), FullInsights(2).</description>
  <rationale>Verbosity control enables token economy per PRD section 1.6.</rationale>
  <acceptance_criteria>
    <criterion>VerbosityLevel::RawOnly = 0: approximately 100 tokens response</criterion>
    <criterion>VerbosityLevel::TextAndIds = 1: approximately 200 tokens response (DEFAULT)</criterion>
    <criterion>VerbosityLevel::FullInsights = 2: approximately 800 tokens response</criterion>
    <criterion>VerbosityLevel implements From&lt;u8&gt; with invalid values defaulting to TextAndIds</criterion>
    <criterion>Session-level verbosity can be overridden per-call</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-006" story_ref="US-CORE-01" priority="must">
  <description>The system SHALL define a UTLState struct containing delta_s, delta_c, w_e, and phi fields per UTL equation.</description>
  <rationale>UTL state is required for learning score computation per PRD section 2.1.</rationale>
  <implements_trait>REQ-GHOST-007 (UTLProcessor trait contract)</implements_trait>
  <acceptance_criteria>
    <criterion>UTLState.delta_s: f32 (Entropy/Surprise) in range [0, 1]</criterion>
    <criterion>UTLState.delta_c: f32 (Coherence change) in range [0, 1]</criterion>
    <criterion>UTLState.w_e: f32 (Emotional weight) in range [0.5, 1.5]</criterion>
    <criterion>UTLState.phi: f32 (Phase angle) in range [0, PI]</criterion>
    <criterion>Default values: delta_s=0.5, delta_c=0.5, w_e=1.0, phi=0.0</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-027" story_ref="US-CORE-01" priority="must">
  <description>Edge SHALL support neurotransmitter weight profiles for domain-aware modulation.</description>
  <rationale>Marblestone domain-aware modulation enables adaptive edge weighting based on neuroscience-inspired neurotransmitter profiles for excitatory, inhibitory, and modulatory signals across different knowledge domains.</rationale>
  <acceptance_criteria>
    <criterion>GraphEdge includes neurotransmitter_weights: NeurotransmitterWeights struct</criterion>
    <criterion>NeurotransmitterWeights.excitatory: f32 (Glutamate-like) in range [0, 1]</criterion>
    <criterion>NeurotransmitterWeights.inhibitory: f32 (GABA-like) in range [0, 1]</criterion>
    <criterion>NeurotransmitterWeights.modulatory: f32 (Dopamine-like) in range [0, 1]</criterion>
    <criterion>GraphEdge includes is_amortized_shortcut: bool for sleep replay shortcuts</criterion>
    <criterion>GraphEdge includes steering_reward: f32 in range [-1.0, 1.0] for traversal feedback</criterion>
    <criterion>Domain enum supports Code, Legal, Medical, Creative, Research, General classifications</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Memex Storage Layer ==================== -->

<requirement id="REQ-CORE-007" story_ref="US-CORE-06" priority="must">
  <description>The system SHALL implement the MemoryStore trait using RocksDB as the persistent storage backend.</description>
  <rationale>RocksDB provides proven performance, durability, and LSM-tree efficiency for write-heavy workloads.</rationale>
  <implements_trait>REQ-GHOST-009 (MemoryStore trait contract)</implements_trait>
  <acceptance_criteria>
    <criterion>RocksDB database opens with configurable path</criterion>
    <criterion>Store operation writes to RocksDB with key = UUID bytes</criterion>
    <criterion>Retrieve operation reads from RocksDB by UUID</criterion>
    <criterion>Delete operation removes key with optional soft-delete flag</criterion>
    <criterion>Search operation is available (stub until Module 3 provides real embeddings)</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-008" story_ref="US-CORE-06" priority="must">
  <description>The system SHALL configure RocksDB with Write-Ahead Log (WAL) enabled for crash recovery.</description>
  <rationale>WAL ensures durability by persisting writes before acknowledging to caller.</rationale>
  <acceptance_criteria>
    <criterion>WAL is enabled in RocksDB options</criterion>
    <criterion>WAL directory is configurable via config file</criterion>
    <criterion>After crash, WAL replay recovers uncommitted transactions</criterion>
    <criterion>WAL can be disabled for testing via config flag</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-009" story_ref="US-CORE-10" priority="must">
  <description>The system SHALL configure RocksDB with Bloom filters for fast negative lookups.</description>
  <rationale>Bloom filters reduce disk I/O by filtering non-existent keys before disk access.</rationale>
  <acceptance_criteria>
    <criterion>Bloom filter is configured with 10 bits per key (approximately 1% false positive rate)</criterion>
    <criterion>Bloom filter is applied to SST files</criterion>
    <criterion>False positive rate is configurable via config file</criterion>
    <criterion>Bloom filter statistics are exposed via metrics</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-010" story_ref="US-CORE-07" priority="must">
  <description>The system SHALL implement an LRU cache for frequently accessed MemoryNodes.</description>
  <rationale>LRU cache reduces read latency for hot data per PRD performance targets.</rationale>
  <acceptance_criteria>
    <criterion>LRU cache capacity is configurable (default 10000 nodes)</criterion>
    <criterion>Cache hit updates access order (LRU eviction)</criterion>
    <criterion>Cache is checked before RocksDB read</criterion>
    <criterion>Store/update operations invalidate cache entry</criterion>
    <criterion>Cache hit rate is exposed via metrics</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-011" story_ref="US-CORE-07" priority="must">
  <description>The system SHALL achieve sustained write throughput of greater than 10K writes per second.</description>
  <rationale>High write throughput is critical for real-time memory ingestion per PRD section 13.</rationale>
  <acceptance_criteria>
    <criterion>Benchmark test validates 10K sequential writes in under 1 second</criterion>
    <criterion>Benchmark test validates 10K concurrent writes (50 writers x 200 each) in under 2 seconds</criterion>
    <criterion>Write batching is supported for bulk operations</criterion>
    <criterion>Performance metrics are exposed for monitoring</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-012" story_ref="US-CORE-06" priority="must">
  <description>The system SHALL serialize MemoryNodes using bincode for compact binary storage.</description>
  <rationale>Bincode provides compact serialization with fast encode/decode for Rust structs.</rationale>
  <acceptance_criteria>
    <criterion>MemoryNode serializes to bytes using bincode</criterion>
    <criterion>MemoryNode deserializes from bytes using bincode</criterion>
    <criterion>Serialization format is versioned for future migrations</criterion>
    <criterion>Invalid binary data returns descriptive error, not panic</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-013" story_ref="US-CORE-06" priority="must">
  <description>The system SHALL support soft-delete with 30-day retention before permanent deletion.</description>
  <rationale>Soft delete enables recovery per PRD section 5.5 and SEC-06.</rationale>
  <acceptance_criteria>
    <criterion>delete(id, soft=true) marks node as deleted but retains data</criterion>
    <criterion>delete(id, soft=false, reason="user_requested") permanently removes data</criterion>
    <criterion>Soft-deleted nodes are excluded from search results</criterion>
    <criterion>Background process purges soft-deleted nodes older than 30 days</criterion>
    <criterion>Tombstone metadata includes deletion timestamp and reason</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Primary MCP Tools ==================== -->

<requirement id="REQ-CORE-014" story_ref="US-CORE-01" priority="must">
  <description>The system SHALL implement the store_memory MCP tool per PRD section 5.2 parameter specification.</description>
  <rationale>store_memory is the primary tool for persisting knowledge.</rationale>
  <acceptance_criteria>
    <criterion>Required parameter: content (string, max 65536 chars) OR content_base64 (for binary)</criterion>
    <criterion>Required parameter: rationale (string, 10-500 chars) per AP-010</criterion>
    <criterion>Optional parameter: importance (float 0-1, default 0.5)</criterion>
    <criterion>Optional parameter: modality (text|image|audio|video, default text)</criterion>
    <criterion>Optional parameter: metadata (object)</criterion>
    <criterion>Optional parameter: link_to (array of UUID for graph edges)</criterion>
    <criterion>Returns: node_id (UUID), created_at, johari_quadrant, pulse</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-015" story_ref="US-CORE-02" priority="must">
  <description>The system SHALL implement the recall_memory MCP tool for semantic similarity retrieval.</description>
  <rationale>recall_memory enables retrieval by semantic similarity rather than exact match.</rationale>
  <acceptance_criteria>
    <criterion>Required parameter: query (string, 1-4096 chars)</criterion>
    <criterion>Optional parameter: top_k (int, default 10, max 100)</criterion>
    <criterion>Optional parameter: filters.min_importance (float 0-1)</criterion>
    <criterion>Optional parameter: filters.johari_quadrants (array of quadrant names)</criterion>
    <criterion>Optional parameter: filters.created_after (datetime)</criterion>
    <criterion>Returns: nodes array with id, content, importance, relevance_score</criterion>
    <criterion>Returns: pulse header</criterion>
    <criterion>Note: Uses stub similarity until Module 3 provides real embeddings</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-016" story_ref="US-CORE-03" priority="must">
  <description>The system SHALL implement the inject_context MCP tool per PRD section 5.2 and Appendix 19.</description>
  <rationale>inject_context is the primary retrieval tool for AI agents per PRD section 1.5.</rationale>
  <acceptance_criteria>
    <criterion>Required parameter: query (string, 1-4096 chars)</criterion>
    <criterion>Optional parameter: max_tokens (int, 100-8192, default 2048)</criterion>
    <criterion>Optional parameter: session_id (uuid)</criterion>
    <criterion>Optional parameter: priority (low|normal|high|critical, default normal)</criterion>
    <criterion>Optional parameter: distillation_mode (auto|raw|narrative|structured|code_focused)</criterion>
    <criterion>Optional parameter: include_metadata (array: causal_links, entailment_cones, neighborhood, conflicts)</criterion>
    <criterion>Optional parameter: verbosity_level (0|1|2, default 1)</criterion>
    <criterion>Returns: context, tokens_used, tokens_before_distillation, distillation_applied</criterion>
    <criterion>Returns: compression_ratio, nodes_retrieved, utl_metrics, pulse</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-017" story_ref="US-CORE-04" priority="must">
  <description>The system SHALL implement the get_memetic_status MCP tool per PRD section 5.2.</description>
  <rationale>get_memetic_status is the primary health check tool for AI agents per PRD section 1.2.</rationale>
  <acceptance_criteria>
    <criterion>Optional parameter: session_id (uuid)</criterion>
    <criterion>Returns: coherence_score (float 0-1)</criterion>
    <criterion>Returns: entropy_level (float 0-1)</criterion>
    <criterion>Returns: top_active_concepts (array, max 5 node summaries)</criterion>
    <criterion>Returns: suggested_action (consolidate|explore|clarify|curate|ready)</criterion>
    <criterion>Returns: dream_available (bool)</criterion>
    <criterion>Returns: curation_tasks (array of task objects)</criterion>
    <criterion>Returns: pulse header</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-018" story_ref="US-CORE-05" priority="must">
  <description>The system SHALL implement the set_verbosity MCP tool for controlling response detail level.</description>
  <rationale>Verbosity control enables token economy per PRD section 1.6.</rationale>
  <acceptance_criteria>
    <criterion>Required parameter: level (0|1|2)</criterion>
    <criterion>Optional parameter: session_id (uuid)</criterion>
    <criterion>Level 0 (RawOnly): approximately 100 tokens responses</criterion>
    <criterion>Level 1 (TextAndIds): approximately 200 tokens responses (default)</criterion>
    <criterion>Level 2 (FullInsights): approximately 800 tokens responses</criterion>
    <criterion>Setting persists for session duration</criterion>
    <criterion>Returns: previous_level, new_level, pulse</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-019" story_ref="US-CORE-08" priority="must">
  <description>All MCP tool responses SHALL include the Cognitive Pulse header.</description>
  <rationale>Cognitive Pulse is mandatory per PRD section 1.3 for meta-cognitive state tracking.</rationale>
  <acceptance_criteria>
    <criterion>Every successful response includes pulse.entropy in [0.0, 1.0]</criterion>
    <criterion>Every successful response includes pulse.coherence in [0.0, 1.0]</criterion>
    <criterion>Every successful response includes pulse.suggested_action</criterion>
    <criterion>Pulse values are computed from actual system state, not mocked</criterion>
    <criterion>Pulse computation adds less than 1ms latency</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Verbosity System ==================== -->

<requirement id="REQ-CORE-020" story_ref="US-CORE-05" priority="must">
  <description>VerbosityLevel::RawOnly (0) SHALL return minimal response with only IDs and raw vectors.</description>
  <rationale>RawOnly mode minimizes token usage for high-confidence lookups per PRD section 1.6.</rationale>
  <acceptance_criteria>
    <criterion>Response excludes content text (only node IDs returned)</criterion>
    <criterion>Response excludes UTL analysis</criterion>
    <criterion>Response excludes neighborhood information</criterion>
    <criterion>Response includes pulse header (mandatory)</criterion>
    <criterion>Total response approximately 100 tokens</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-021" story_ref="US-CORE-05" priority="must">
  <description>VerbosityLevel::TextAndIds (1) SHALL return content text and node IDs (default level).</description>
  <rationale>TextAndIds is the balanced default for normal operations per PRD section 1.6.</rationale>
  <acceptance_criteria>
    <criterion>Response includes content text for retrieved nodes</criterion>
    <criterion>Response includes node IDs</criterion>
    <criterion>Response excludes detailed UTL analysis</criterion>
    <criterion>Response excludes causal links</criterion>
    <criterion>Response includes pulse header</criterion>
    <criterion>Total response approximately 200 tokens</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-022" story_ref="US-CORE-05" priority="must">
  <description>VerbosityLevel::FullInsights (2) SHALL return complete analysis including causal links and UTL scores.</description>
  <rationale>FullInsights mode is used when coherence less than 0.4 (confused state) per PRD section 1.6.</rationale>
  <acceptance_criteria>
    <criterion>Response includes all TextAndIds content</criterion>
    <criterion>Response includes causal_links between nodes</criterion>
    <criterion>Response includes entailment_cones</criterion>
    <criterion>Response includes UTL scores per node</criterion>
    <criterion>Response includes conflict_analysis</criterion>
    <criterion>Response includes pulse header</criterion>
    <criterion>Total response approximately 800 tokens</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Session Management ==================== -->

<requirement id="REQ-CORE-023" story_ref="US-CORE-04, US-CORE-05" priority="must">
  <description>The system SHALL maintain session state including verbosity level and working context.</description>
  <rationale>Session state enables personalized behavior and context continuity.</rationale>
  <acceptance_criteria>
    <criterion>Session is created on first tool call with session_id</criterion>
    <criterion>Session stores verbosity_level (default 1)</criterion>
    <criterion>Session stores last_activity timestamp</criterion>
    <criterion>Session expires after configurable timeout (default 24 hours)</criterion>
    <criterion>Session state is stored in memory (not persisted across restarts in Phase 1)</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Entropy and Coherence Computation ==================== -->

<requirement id="REQ-CORE-024" story_ref="US-CORE-08, US-CORE-09" priority="must">
  <description>The system SHALL compute entropy based on information diversity and novelty in the knowledge base.</description>
  <rationale>Entropy is a core UTL metric indicating system uncertainty per PRD section 2.1.</rationale>
  <implements_trait>REQ-GHOST-007 (UTLProcessor.compute_surprise)</implements_trait>
  <acceptance_criteria>
    <criterion>Entropy is computed from embedding distribution statistics</criterion>
    <criterion>High entropy indicates diverse, novel, or conflicting information</criterion>
    <criterion>Low entropy indicates stable, consistent knowledge</criterion>
    <criterion>Entropy value is clamped to [0.0, 1.0] range</criterion>
    <criterion>Computation uses heuristics until Module 5 provides full UTL</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-CORE-025" story_ref="US-CORE-08, US-CORE-09" priority="must">
  <description>The system SHALL compute coherence based on graph connectivity and consistency.</description>
  <rationale>Coherence is a core UTL metric indicating understanding per PRD section 2.1.</rationale>
  <implements_trait>REQ-GHOST-007 (UTLProcessor.compute_coherence_change)</implements_trait>
  <acceptance_criteria>
    <criterion>Coherence considers edge density in retrieved subgraph</criterion>
    <criterion>Coherence penalizes contradictory relationships</criterion>
    <criterion>High coherence indicates well-connected, consistent knowledge</criterion>
    <criterion>Low coherence indicates fragmented or conflicting knowledge</criterion>
    <criterion>Coherence value is clamped to [0.0, 1.0] range</criterion>
    <criterion>Computation uses heuristics until Module 5 provides full UTL</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Error Handling ==================== -->

<requirement id="REQ-CORE-026" story_ref="US-CORE-01, US-CORE-02, US-CORE-03" priority="must">
  <description>The system SHALL return structured errors with JSON-RPC error codes per PRD section 7.8 and constitution.</description>
  <rationale>Structured errors enable programmatic error handling and debugging.</rationale>
  <acceptance_criteria>
    <criterion>Storage errors use code -32002 (StorageError)</criterion>
    <criterion>Session not found uses code -32000 (SessionNotFound)</criterion>
    <criterion>Graph query errors use code -32001 (GraphQueryError)</criterion>
    <criterion>Rate limit exceeded uses code -32004 (RateLimitExceeded)</criterion>
    <criterion>All errors include message and optional data object with details</criterion>
    <criterion>Errors do not expose internal implementation details per SEC requirements</criterion>
  </acceptance_criteria>
</requirement>

</requirements>

<!-- ============================================================================ -->
<!-- EDGE CASES -->
<!-- ============================================================================ -->

<edge_cases>

<edge_case id="EC-CORE-001" req_ref="REQ-CORE-014">
  <scenario>store_memory called with content exceeding 65536 characters</scenario>
  <expected_behavior>Return error -32602 (Invalid params) with message "Content exceeds maximum length of 65536 characters"</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-002" req_ref="REQ-CORE-014">
  <scenario>store_memory called without rationale field</scenario>
  <expected_behavior>Return error -32602 (Invalid params) with message "Rationale is required (10-500 characters)"</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-003" req_ref="REQ-CORE-014">
  <scenario>store_memory called with rationale shorter than 10 characters</scenario>
  <expected_behavior>Return error -32602 (Invalid params) with message "Rationale must be at least 10 characters"</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-004" req_ref="REQ-CORE-015">
  <scenario>recall_memory query returns no matching nodes</scenario>
  <expected_behavior>Return success with empty nodes array, not an error. Include pulse with suggested_action="explore"</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-005" req_ref="REQ-CORE-015">
  <scenario>recall_memory called with top_k=0</scenario>
  <expected_behavior>Return error -32602 (Invalid params) with message "top_k must be between 1 and 100"</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-006" req_ref="REQ-CORE-016">
  <scenario>inject_context called with max_tokens less than minimum (100)</scenario>
  <expected_behavior>Return error -32602 (Invalid params) with message "max_tokens must be between 100 and 8192"</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-007" req_ref="REQ-CORE-007">
  <scenario>RocksDB database file is corrupted</scenario>
  <expected_behavior>Return error -32002 (StorageError) with message "Database integrity check failed. Recovery required." Log detailed corruption info for admin.</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-008" req_ref="REQ-CORE-007">
  <scenario>RocksDB database directory has insufficient permissions</scenario>
  <expected_behavior>Server fails to start with clear error message including path and required permissions</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-009" req_ref="REQ-CORE-007">
  <scenario>Disk is full during write operation</scenario>
  <expected_behavior>Return error -32002 (StorageError) with message "Insufficient disk space". Transaction is rolled back. No partial writes.</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-010" req_ref="REQ-CORE-010">
  <scenario>LRU cache receives more entries than capacity</scenario>
  <expected_behavior>Least recently used entries are evicted. No memory leak. Eviction count is logged at debug level.</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-011" req_ref="REQ-CORE-013">
  <scenario>Retrieve called for soft-deleted node</scenario>
  <expected_behavior>Return Ok(None) as if node does not exist. Soft-deleted nodes are not visible to normal operations.</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-012" req_ref="REQ-CORE-013">
  <scenario>Permanent delete called without reason="user_requested"</scenario>
  <expected_behavior>Return error -32602 (Invalid params) with message "Permanent deletion requires reason='user_requested'"</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-013" req_ref="REQ-CORE-011">
  <scenario>50+ concurrent writers performing bulk inserts</scenario>
  <expected_behavior>All writes complete successfully. No deadlocks. Throughput remains above 5K writes/sec under contention.</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-014" req_ref="REQ-CORE-008">
  <scenario>Server crashes during WAL replay on startup</scenario>
  <expected_behavior>Server logs WAL replay progress. If crash during replay, next startup continues from last checkpoint.</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-015" req_ref="REQ-CORE-023">
  <scenario>Tool call with expired session_id</scenario>
  <expected_behavior>Return error -32000 (SessionNotFound) with message "Session expired or not found". Client should create new session.</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-016" req_ref="REQ-CORE-018">
  <scenario>set_verbosity called with level greater than 2</scenario>
  <expected_behavior>Return error -32602 (Invalid params) with message "Verbosity level must be 0, 1, or 2"</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-017" req_ref="REQ-CORE-014">
  <scenario>store_memory called with both content and content_base64</scenario>
  <expected_behavior>Return error -32602 (Invalid params) with message "Provide either content or content_base64, not both"</expected_behavior>
</edge_case>

<edge_case id="EC-CORE-018" req_ref="REQ-CORE-012">
  <scenario>Deserializing MemoryNode from old format version</scenario>
  <expected_behavior>Migration logic converts old format to current. If conversion impossible, return error with node ID and format version.</expected_behavior>
</edge_case>

</edge_cases>

<!-- ============================================================================ -->
<!-- ERROR STATES -->
<!-- ============================================================================ -->

<error_states>

<error id="ERR-CORE-001" json_rpc_code="-32002">
  <condition>RocksDB write operation fails (disk full, permission denied, corruption)</condition>
  <message>Storage error: {operation} failed - {details}</message>
  <recovery>Check disk space and permissions. If corruption, run database repair tool.</recovery>
  <logging>Log full error context including operation type, key (if safe), and RocksDB error code</logging>
</error>

<error id="ERR-CORE-002" json_rpc_code="-32002">
  <condition>RocksDB read operation fails (file not found, I/O error)</condition>
  <message>Storage error: Read failed for node {node_id}</message>
  <recovery>Verify database integrity. Check for disk errors.</recovery>
  <logging>Log node_id, operation, and underlying error</logging>
</error>

<error id="ERR-CORE-003" json_rpc_code="-32000">
  <condition>Session ID provided but session not found or expired</condition>
  <message>Session not found or expired: {session_id}</message>
  <recovery>Create new session or omit session_id for anonymous operation</recovery>
  <logging>Log session_id and expiration time if known</logging>
</error>

<error id="ERR-CORE-004" json_rpc_code="-32602">
  <condition>Required parameter missing or invalid type</condition>
  <message>Invalid params: {parameter_name} - {validation_error}</message>
  <recovery>Review parameter schema from tools/list and correct request</recovery>
  <logging>Log parameter name, expected type, received value (if safe)</logging>
</error>

<error id="ERR-CORE-005" json_rpc_code="-32602">
  <condition>Rationale field missing or too short for store_memory</condition>
  <message>Invalid params: rationale is required (10-500 characters)</message>
  <recovery>Provide meaningful rationale explaining why this memory should be stored</recovery>
  <logging>Log content length (not content) and rationale length</logging>
</error>

<error id="ERR-CORE-006" json_rpc_code="-32602">
  <condition>Content exceeds maximum length</condition>
  <message>Invalid params: content exceeds maximum length of 65536 characters (received: {length})</message>
  <recovery>Split content into multiple memories or summarize before storing</recovery>
  <logging>Log content length received</logging>
</error>

<error id="ERR-CORE-007" json_rpc_code="-32602">
  <condition>Permanent delete requested without user_requested reason</condition>
  <message>Invalid params: Permanent deletion requires reason='user_requested'</message>
  <recovery>Use soft_delete=true for non-user-requested deletions, or confirm user intent</recovery>
  <logging>Log node_id and provided reason</logging>
</error>

<error id="ERR-CORE-008" json_rpc_code="-32603">
  <condition>Serialization/deserialization failure</condition>
  <message>Internal error: Data serialization failed</message>
  <recovery>Report to administrators with error ID for investigation</recovery>
  <logging>Log full serialization error, data type, and format version</logging>
</error>

<error id="ERR-CORE-009" json_rpc_code="-32002">
  <condition>Database lock contention timeout</condition>
  <message>Storage error: Operation timed out due to lock contention</message>
  <recovery>Retry operation. If persistent, reduce concurrent load.</recovery>
  <logging>Log operation, lock type, wait time, and timeout threshold</logging>
</error>

<error id="ERR-CORE-010" json_rpc_code="-32002">
  <condition>WAL replay failure on startup</condition>
  <message>Storage error: Write-ahead log replay failed</message>
  <recovery>Check WAL directory permissions and integrity. May require manual recovery.</recovery>
  <logging>Log WAL file path, replay position, and underlying error</logging>
</error>

</error_states>

<!-- ============================================================================ -->
<!-- TEST PLAN -->
<!-- ============================================================================ -->

<test_plan>

<!-- Unit Tests: Data Types -->

<test_case id="TC-CORE-001" type="unit" req_ref="REQ-CORE-001">
  <description>MemoryNode serialization round-trip preserves all fields</description>
  <preconditions>None</preconditions>
  <inputs>{"node": MemoryNode with all fields populated}</inputs>
  <expected>bincode::serialize then deserialize returns identical MemoryNode</expected>
  <data_requirements>Use real MemoryNode with varied field values, no mocks</data_requirements>
</test_case>

<test_case id="TC-CORE-002" type="unit" req_ref="REQ-CORE-001">
  <description>MemoryNode content validation enforces 65536 char limit</description>
  <preconditions>None</preconditions>
  <inputs>{"content": "a".repeat(65537)}</inputs>
  <expected>Validation error before storage attempt</expected>
  <data_requirements>Generate real string of exact length</data_requirements>
</test_case>

<test_case id="TC-CORE-003" type="unit" req_ref="REQ-CORE-002">
  <description>JohariQuadrant computation from entropy and coherence values</description>
  <preconditions>None</preconditions>
  <inputs>[{"entropy": 0.3, "coherence": 0.7}, {"entropy": 0.6, "coherence": 0.3}, {"entropy": 0.3, "coherence": 0.3}, {"entropy": 0.6, "coherence": 0.7}]</inputs>
  <expected>[JohariQuadrant::Open, JohariQuadrant::Blind, JohariQuadrant::Hidden, JohariQuadrant::Unknown]</expected>
  <data_requirements>Test boundary values at exactly 0.5</data_requirements>
</test_case>

<test_case id="TC-CORE-004" type="unit" req_ref="REQ-CORE-003">
  <description>CognitivePulse serialization produces valid JSON</description>
  <preconditions>None</preconditions>
  <inputs>{"pulse": CognitivePulse { entropy: 0.5, coherence: 0.7, suggested_action: SuggestedAction::Ready }}</inputs>
  <expected>Valid JSON with entropy, coherence, suggested_action fields</expected>
  <data_requirements>Use real CognitivePulse struct</data_requirements>
</test_case>

<test_case id="TC-CORE-005" type="unit" req_ref="REQ-CORE-004">
  <description>SuggestedAction computed correctly from entropy/coherence matrix</description>
  <preconditions>None</preconditions>
  <inputs>Test all quadrants of entropy/coherence space</inputs>
  <expected>Correct SuggestedAction for each quadrant per PRD section 1.3</expected>
  <data_requirements>Enumerate all entropy/coherence combinations</data_requirements>
</test_case>

<test_case id="TC-CORE-006" type="unit" req_ref="REQ-CORE-005">
  <description>VerbosityLevel From&lt;u8&gt; handles invalid values</description>
  <preconditions>None</preconditions>
  <inputs>[0, 1, 2, 3, 255]</inputs>
  <expected>[RawOnly, TextAndIds, FullInsights, TextAndIds, TextAndIds]</expected>
  <data_requirements>None</data_requirements>
</test_case>

<test_case id="TC-CORE-007" type="unit" req_ref="REQ-CORE-006">
  <description>UTLState default values are correct</description>
  <preconditions>None</preconditions>
  <inputs>UTLState::default()</inputs>
  <expected>delta_s=0.5, delta_c=0.5, w_e=1.0, phi=0.0</expected>
  <data_requirements>None</data_requirements>
</test_case>

<!-- Unit Tests: Storage Layer -->

<test_case id="TC-CORE-008" type="integration" req_ref="REQ-CORE-007">
  <description>RocksDB store and retrieve round-trip</description>
  <preconditions>RocksDB database initialized with temp directory</preconditions>
  <inputs>{"node": MemoryNode with content "test content"}</inputs>
  <expected>store() returns UUID, retrieve(uuid) returns identical node</expected>
  <data_requirements>Use real RocksDB instance, not mock</data_requirements>
</test_case>

<test_case id="TC-CORE-009" type="integration" req_ref="REQ-CORE-008">
  <description>WAL recovery after simulated crash</description>
  <preconditions>RocksDB with WAL enabled</preconditions>
  <inputs>Store 100 nodes, kill process without graceful shutdown, restart</inputs>
  <expected>All 100 nodes are recoverable after restart</expected>
  <data_requirements>Use real RocksDB with actual process termination</data_requirements>
</test_case>

<test_case id="TC-CORE-010" type="integration" req_ref="REQ-CORE-009">
  <description>Bloom filter reduces disk reads for non-existent keys</description>
  <preconditions>RocksDB with Bloom filter, 10K nodes stored</preconditions>
  <inputs>Query 1K non-existent UUIDs</inputs>
  <expected>Bloom filter blocks disk access for majority of queries. Measure via RocksDB statistics.</expected>
  <data_requirements>Use real RocksDB with statistics enabled</data_requirements>
</test_case>

<test_case id="TC-CORE-011" type="integration" req_ref="REQ-CORE-010">
  <description>LRU cache hit improves read latency</description>
  <preconditions>LRU cache with capacity 1000</preconditions>
  <inputs>Store node, retrieve twice in succession</inputs>
  <expected>Second retrieve is 10x+ faster (cache hit). Measure via timing.</expected>
  <data_requirements>Use real LRU cache implementation</data_requirements>
</test_case>

<test_case id="TC-CORE-012" type="benchmark" req_ref="REQ-CORE-011">
  <description>Sequential write throughput exceeds 10K/sec</description>
  <preconditions>RocksDB with production-like configuration</preconditions>
  <inputs>10K sequential store_memory calls</inputs>
  <expected>All writes complete in under 1 second</expected>
  <data_requirements>Use real RocksDB, measure wall clock time</data_requirements>
</test_case>

<test_case id="TC-CORE-013" type="benchmark" req_ref="REQ-CORE-011">
  <description>Concurrent write throughput under contention</description>
  <preconditions>RocksDB with production-like configuration</preconditions>
  <inputs>50 concurrent writers, 200 writes each (10K total)</inputs>
  <expected>All writes complete in under 2 seconds with no errors</expected>
  <data_requirements>Use real RocksDB with actual concurrent threads</data_requirements>
</test_case>

<test_case id="TC-CORE-014" type="integration" req_ref="REQ-CORE-012">
  <description>Bincode serialization handles all field types</description>
  <preconditions>None</preconditions>
  <inputs>MemoryNode with HashMap, Vec, Option fields populated</inputs>
  <expected>Serialization succeeds, deserialization produces identical struct</expected>
  <data_requirements>Use real complex data structures</data_requirements>
</test_case>

<test_case id="TC-CORE-015" type="integration" req_ref="REQ-CORE-013">
  <description>Soft delete makes node invisible but recoverable</description>
  <preconditions>Node stored in RocksDB</preconditions>
  <inputs>delete(node_id, soft=true)</inputs>
  <expected>retrieve(node_id) returns None, but internal tombstone query shows node</expected>
  <data_requirements>Use real RocksDB</data_requirements>
</test_case>

<test_case id="TC-CORE-016" type="integration" req_ref="REQ-CORE-013">
  <description>Permanent delete requires user_requested reason</description>
  <preconditions>Node stored in RocksDB</preconditions>
  <inputs>delete(node_id, soft=false, reason="obsolete")</inputs>
  <expected>Error -32602 requiring reason="user_requested"</expected>
  <data_requirements>Use real RocksDB</data_requirements>
</test_case>

<!-- Integration Tests: MCP Tools -->

<test_case id="TC-CORE-017" type="integration" req_ref="REQ-CORE-014">
  <description>store_memory creates node and returns UUID</description>
  <preconditions>MCP server running with RocksDB backend</preconditions>
  <inputs>{"content": "Test memory content", "rationale": "Testing store_memory functionality", "importance": 0.8}</inputs>
  <expected>Response with valid UUID, created_at timestamp, johari_quadrant, pulse</expected>
  <data_requirements>Use real MCP server, not stubs</data_requirements>
</test_case>

<test_case id="TC-CORE-018" type="integration" req_ref="REQ-CORE-014">
  <description>store_memory rejects missing rationale</description>
  <preconditions>MCP server running</preconditions>
  <inputs>{"content": "Test memory", "importance": 0.5}</inputs>
  <expected>Error -32602 with message about required rationale</expected>
  <data_requirements>Use real MCP server</data_requirements>
</test_case>

<test_case id="TC-CORE-019" type="integration" req_ref="REQ-CORE-015">
  <description>recall_memory returns top-k similar nodes</description>
  <preconditions>10 memory nodes stored with varied content</preconditions>
  <inputs>{"query": "test query", "top_k": 3}</inputs>
  <expected>Array of 3 nodes with relevance_score, sorted by relevance</expected>
  <data_requirements>Use real storage with real nodes, stub similarity until Module 3</data_requirements>
</test_case>

<test_case id="TC-CORE-020" type="integration" req_ref="REQ-CORE-015">
  <description>recall_memory respects min_importance filter</description>
  <preconditions>Nodes with importance 0.2, 0.5, 0.8 stored</preconditions>
  <inputs>{"query": "test", "filters": {"min_importance": 0.6}}</inputs>
  <expected>Only node with importance 0.8 returned</expected>
  <data_requirements>Use real storage</data_requirements>
</test_case>

<test_case id="TC-CORE-021" type="integration" req_ref="REQ-CORE-016">
  <description>inject_context returns context with pulse header</description>
  <preconditions>Memory nodes stored</preconditions>
  <inputs>{"query": "test query", "max_tokens": 500}</inputs>
  <expected>Response with context, tokens_used, utl_metrics, pulse</expected>
  <data_requirements>Use real storage</data_requirements>
</test_case>

<test_case id="TC-CORE-022" type="integration" req_ref="REQ-CORE-017">
  <description>get_memetic_status returns system health metrics</description>
  <preconditions>Several memory nodes stored</preconditions>
  <inputs>{}</inputs>
  <expected>Response with coherence_score, entropy_level, suggested_action, pulse</expected>
  <data_requirements>Use real storage with computed metrics</data_requirements>
</test_case>

<test_case id="TC-CORE-023" type="integration" req_ref="REQ-CORE-018">
  <description>set_verbosity changes session verbosity level</description>
  <preconditions>Session created</preconditions>
  <inputs>{"level": 2, "session_id": "..."}</inputs>
  <expected>Response with previous_level, new_level=2, pulse</expected>
  <data_requirements>Use real session management</data_requirements>
</test_case>

<test_case id="TC-CORE-024" type="integration" req_ref="REQ-CORE-019">
  <description>All tool responses include Cognitive Pulse</description>
  <preconditions>MCP server running</preconditions>
  <inputs>Call each tool: store_memory, recall_memory, inject_context, get_memetic_status, set_verbosity</inputs>
  <expected>Every response includes pulse with entropy, coherence, suggested_action</expected>
  <data_requirements>Use real MCP server</data_requirements>
</test_case>

<!-- Integration Tests: Verbosity -->

<test_case id="TC-CORE-025" type="integration" req_ref="REQ-CORE-020">
  <description>VerbosityLevel 0 returns minimal response</description>
  <preconditions>Verbosity set to 0</preconditions>
  <inputs>inject_context call</inputs>
  <expected>Response approximately 100 tokens, includes IDs but not full content</expected>
  <data_requirements>Measure actual token count</data_requirements>
</test_case>

<test_case id="TC-CORE-026" type="integration" req_ref="REQ-CORE-021">
  <description>VerbosityLevel 1 returns text and IDs</description>
  <preconditions>Verbosity set to 1 (default)</preconditions>
  <inputs>inject_context call</inputs>
  <expected>Response approximately 200 tokens, includes content text and IDs</expected>
  <data_requirements>Measure actual token count</data_requirements>
</test_case>

<test_case id="TC-CORE-027" type="integration" req_ref="REQ-CORE-022">
  <description>VerbosityLevel 2 returns full insights</description>
  <preconditions>Verbosity set to 2</preconditions>
  <inputs>inject_context call</inputs>
  <expected>Response approximately 800 tokens, includes UTL scores and analysis</expected>
  <data_requirements>Measure actual token count</data_requirements>
</test_case>

<!-- Stress Tests -->

<test_case id="TC-CORE-028" type="stress" req_ref="REQ-CORE-011">
  <description>Storage handles 100K nodes without degradation</description>
  <preconditions>Empty RocksDB</preconditions>
  <inputs>Store 100K nodes sequentially</inputs>
  <expected>No errors, final retrieval latency under 10ms</expected>
  <data_requirements>Use real RocksDB with actual data volume</data_requirements>
</test_case>

<test_case id="TC-CORE-029" type="stress" req_ref="REQ-CORE-010">
  <description>LRU cache eviction under memory pressure</description>
  <preconditions>LRU cache with 1000 capacity</preconditions>
  <inputs>Access 10K different nodes</inputs>
  <expected>Cache size stays at 1000, no memory leak, correct eviction order</expected>
  <data_requirements>Use real cache, monitor memory</data_requirements>
</test_case>

<!-- Unit Tests: Marblestone Neuroscience Features -->

<test_case id="TC-CORE-030" type="unit" req_ref="REQ-CORE-027">
  <description>NeurotransmitterWeights domain-specific defaults</description>
  <preconditions>None</preconditions>
  <inputs>All Domain enum variants</inputs>
  <expected>NeurotransmitterWeights::for_domain returns correct defaults per domain (e.g., Creative has high excitatory 0.8, Legal has high inhibitory 0.4)</expected>
  <data_requirements>Verify all six domain types produce distinct weight profiles</data_requirements>
</test_case>

<test_case id="TC-CORE-031" type="unit" req_ref="REQ-CORE-027">
  <description>GraphEdge effective_weight computation with neurotransmitter modulation</description>
  <preconditions>None</preconditions>
  <inputs>GraphEdge with varied neurotransmitter weights and steering rewards</inputs>
  <expected>effective_weight correctly combines base weight, neurotransmitter net activation, and steering reward factor</expected>
  <data_requirements>Test edge cases: high excitatory, high inhibitory, positive/negative steering rewards</data_requirements>
</test_case>

<test_case id="TC-CORE-032" type="unit" req_ref="REQ-CORE-027">
  <description>GraphEdge amortized shortcut identification</description>
  <preconditions>None</preconditions>
  <inputs>GraphEdge with is_amortized_shortcut=true, traversal_count=10, steering_reward=0.5</inputs>
  <expected>is_valuable_shortcut() returns true</expected>
  <data_requirements>Test boundary conditions for traversal_count and steering_reward thresholds</data_requirements>
</test_case>

<test_case id="TC-CORE-033" type="unit" req_ref="REQ-CORE-027">
  <description>GraphEdge serialization with neurotransmitter weights</description>
  <preconditions>None</preconditions>
  <inputs>GraphEdge with populated neurotransmitter_weights, steering_reward, and is_amortized_shortcut fields</inputs>
  <expected>bincode serialize/deserialize round-trip preserves all Marblestone fields</expected>
  <data_requirements>Use real structs, not mocks</data_requirements>
</test_case>

<test_case id="TC-CORE-034" type="unit" req_ref="REQ-CORE-027">
  <description>Steering reward application with decay</description>
  <preconditions>None</preconditions>
  <inputs>GraphEdge.apply_steering with rewards [1.0, -1.0, 0.5, -0.5]</inputs>
  <expected>steering_reward is clamped to [-1.0, 1.0] and decayed by 0.9 factor</expected>
  <data_requirements>Verify boundary clamping and decay multiplication</data_requirements>
</test_case>

</test_plan>

<!-- ============================================================================ -->
<!-- CONSTRAINTS -->
<!-- ============================================================================ -->

<constraints>
  <constraint id="CON-CORE-001">All code must be written in Rust 2021 edition with stable toolchain</constraint>
  <constraint id="CON-CORE-002">NO mock data in tests - tests must verify real RocksDB persistence</constraint>
  <constraint id="CON-CORE-003">NO workarounds for storage failures - errors must propagate with full context</constraint>
  <constraint id="CON-CORE-004">Maximum 500 lines per source file (excluding tests)</constraint>
  <constraint id="CON-CORE-005">All public APIs must have rustdoc comments</constraint>
  <constraint id="CON-CORE-006">Secrets (API keys, passwords) must come from environment variables only</constraint>
  <constraint id="CON-CORE-007">Error messages must not expose internal paths or implementation details</constraint>
  <constraint id="CON-CORE-008">All async functions must be tokio-compatible</constraint>
  <constraint id="CON-CORE-009">Lock acquisition order: always acquire inner lock before faiss_index lock (deadlock prevention)</constraint>
  <constraint id="CON-CORE-010">No use of unwrap() in production code - use expect() with context or propagate Result</constraint>
  <constraint id="CON-CORE-011">Embeddings stored as Vec&lt;f32&gt; with 1536 dimensions until Module 3 upgrade</constraint>
  <constraint id="CON-CORE-012">Rationale field is mandatory for store_memory per AP-010</constraint>
</constraints>

<!-- ============================================================================ -->
<!-- DEPENDENCIES -->
<!-- ============================================================================ -->

<dependencies>
  <dependency type="module" name="Module 1 (Ghost System)" version="1.0">
    <provides>
      <item>UTLProcessor trait (REQ-GHOST-007)</item>
      <item>EmbeddingProvider trait (REQ-GHOST-008)</item>
      <item>MemoryStore trait (REQ-GHOST-009)</item>
      <item>NervousLayer trait (REQ-GHOST-010)</item>
      <item>MCP server infrastructure (REQ-GHOST-011)</item>
      <item>Configuration framework (REQ-GHOST-016)</item>
      <item>Logging infrastructure (REQ-GHOST-020)</item>
    </provides>
  </dependency>
  <dependency type="rust_crate" name="rocksdb" version="0.22+" purpose="Persistent key-value storage"/>
  <dependency type="rust_crate" name="bincode" version="1.3+" purpose="Binary serialization"/>
  <dependency type="rust_crate" name="lru" version="0.12+" purpose="LRU cache implementation"/>
  <dependency type="rust_crate" name="tokio" version="1.35+" purpose="Async runtime"/>
  <dependency type="rust_crate" name="serde" version="1.0+" purpose="Serialization framework"/>
  <dependency type="rust_crate" name="serde_json" version="1.0+" purpose="JSON handling"/>
  <dependency type="rust_crate" name="uuid" version="1.6+" purpose="UUID generation"/>
  <dependency type="rust_crate" name="chrono" version="0.4+" purpose="Timestamps"/>
  <dependency type="rust_crate" name="thiserror" version="1.0+" purpose="Error types"/>
  <dependency type="rust_crate" name="tracing" version="0.1+" purpose="Logging"/>
</dependencies>

</functional_spec>
```

---

## Appendix A: Data Type Definitions

### MemoryNode Structure

```rust
/// A memory node in the knowledge graph.
///
/// Constraint: content.len() <= 65536
/// Constraint: importance in [0.0, 1.0]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    /// Unique identifier (UUID v4)
    pub id: Uuid,

    /// Content text (max 65536 characters)
    pub content: String,

    /// Fused embedding vector (1536D in Phase 1, 4096D after Module 3)
    pub embedding: Vec<f32>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,

    /// Importance score [0, 1]
    pub importance: f32,

    /// Access count for usage tracking
    pub access_count: u32,

    /// Johari quadrant classification
    pub johari_quadrant: JohariQuadrant,

    /// UTL learning state
    pub utl_state: UTLState,

    /// Agent that created this node (for multi-agent isolation)
    pub agent_id: Option<String>,

    /// Semantic cluster assignment
    pub semantic_cluster: Option<Uuid>,

    /// Flexible metadata storage
    pub metadata: HashMap<String, serde_json::Value>,
}
```

### JohariQuadrant Enum

```rust
/// Johari Window quadrant classification based on entropy/coherence.
///
/// | Entropy | Coherence | Quadrant |
/// |---------|-----------|----------|
/// | Low     | High      | Open     |
/// | High    | Low       | Blind    |
/// | Low     | Low       | Hidden   |
/// | High    | High      | Unknown  |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum JohariQuadrant {
    /// Known to self and others - direct recall
    Open,
    /// Unknown to self, known to others - discovery zone
    Blind,
    /// Known to self, hidden from others - private knowledge
    Hidden,
    /// Unknown to both - exploration frontier
    #[default]
    Unknown,
}

impl JohariQuadrant {
    /// Compute quadrant from entropy and coherence values.
    pub fn from_metrics(entropy: f32, coherence: f32) -> Self {
        match (entropy < 0.5, coherence > 0.5) {
            (true, true) => Self::Open,
            (false, false) => Self::Blind,
            (true, false) => Self::Hidden,
            (false, true) => Self::Unknown,
        }
    }
}
```

### UTLState Structure

```rust
/// UTL learning state for a memory node.
///
/// L = f((delta_S x delta_C) . w_e . cos phi)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UTLState {
    /// Entropy change (novelty/surprise) [0, 1]
    pub delta_s: f32,

    /// Coherence change (understanding) [0, 1]
    pub delta_c: f32,

    /// Emotional modulation weight [0.5, 1.5]
    pub w_e: f32,

    /// Phase synchronization angle [0, PI]
    pub phi: f32,
}

impl Default for UTLState {
    fn default() -> Self {
        Self {
            delta_s: 0.5,
            delta_c: 0.5,
            w_e: 1.0,
            phi: 0.0,
        }
    }
}
```

### CognitivePulse Structure

```rust
/// Cognitive Pulse header included in every MCP response.
///
/// Token cost: ~30 tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitivePulse {
    /// System entropy [0, 1]
    pub entropy: f32,

    /// System coherence [0, 1]
    pub coherence: f32,

    /// Suggested next action
    pub suggested_action: SuggestedAction,
}
```

### SuggestedAction Enum

```rust
/// Suggested actions based on system state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuggestedAction {
    /// Process pending consolidation tasks
    Consolidate,
    /// Explore new areas of knowledge
    Explore,
    /// Clarify uncertain information
    Clarify,
    /// Process curation tasks
    Curate,
    /// System healthy and ready
    Ready,
    /// Generate epistemic action (question)
    EpistemicAction,
    /// Trigger dream consolidation
    TriggerDream,
}
```

### VerbosityLevel Enum

```rust
/// Response verbosity levels for token economy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum VerbosityLevel {
    /// Minimal response (~100 tokens)
    RawOnly = 0,
    /// Text and IDs (~200 tokens) - DEFAULT
    TextAndIds = 1,
    /// Full insights (~800 tokens)
    FullInsights = 2,
}

impl Default for VerbosityLevel {
    fn default() -> Self {
        Self::TextAndIds
    }
}

impl From<u8> for VerbosityLevel {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::RawOnly,
            1 => Self::TextAndIds,
            2 => Self::FullInsights,
            _ => Self::TextAndIds, // Default for invalid values
        }
    }
}
```

### Domain Enum (Marblestone)

```rust
/// Domain classification for neurotransmitter profiles.
///
/// Different knowledge domains have characteristic neurotransmitter
/// weight profiles that affect edge traversal behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Domain {
    /// Source code, programming, technical implementation
    Code,
    /// Legal documents, contracts, regulatory compliance
    Legal,
    /// Medical knowledge, clinical data, health information
    Medical,
    /// Creative works, artistic expression, design
    Creative,
    /// Academic research, scientific papers, experimental data
    Research,
    /// General purpose, unclassified domain
    #[default]
    General,
}
```

### NeurotransmitterWeights Structure (Marblestone)

```rust
/// Neurotransmitter-based edge weight modulation (Marblestone).
///
/// Inspired by biological neurotransmitter systems, this structure
/// models excitatory, inhibitory, and modulatory signals that
/// affect edge traversal strength in the knowledge graph.
///
/// Reference: Marblestone et al. - domain-aware modulation for
/// adaptive knowledge retrieval across specialized contexts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeurotransmitterWeights {
    /// Glutamate-like excitatory weight [0, 1]
    /// Higher values strengthen edge activation
    pub excitatory: f32,

    /// GABA-like inhibitory weight [0, 1]
    /// Higher values dampen edge activation
    pub inhibitory: f32,

    /// Dopamine-like modulatory weight [0, 1]
    /// Modulates learning and reward-based edge adjustment
    pub modulatory: f32,
}

impl Default for NeurotransmitterWeights {
    fn default() -> Self {
        Self {
            excitatory: 0.5,
            inhibitory: 0.2,
            modulatory: 0.3,
        }
    }
}

impl NeurotransmitterWeights {
    /// Compute net activation from neurotransmitter balance.
    /// Returns value in [-1, 1] range.
    pub fn net_activation(&self) -> f32 {
        (self.excitatory - self.inhibitory) * (0.5 + 0.5 * self.modulatory)
    }

    /// Create domain-specific default weights.
    pub fn for_domain(domain: Domain) -> Self {
        match domain {
            Domain::Code => Self {
                excitatory: 0.7,  // Strong logical connections
                inhibitory: 0.1,  // Few inhibitory patterns
                modulatory: 0.4,  // Moderate learning rate
            },
            Domain::Legal => Self {
                excitatory: 0.5,  // Balanced activation
                inhibitory: 0.4,  // Strong precedent inhibition
                modulatory: 0.2,  // Conservative learning
            },
            Domain::Medical => Self {
                excitatory: 0.6,  // Strong symptom-diagnosis links
                inhibitory: 0.3,  // Contraindication awareness
                modulatory: 0.5,  // Adaptive to new research
            },
            Domain::Creative => Self {
                excitatory: 0.8,  // High associative activation
                inhibitory: 0.05, // Minimal inhibition
                modulatory: 0.7,  // High plasticity
            },
            Domain::Research => Self {
                excitatory: 0.6,  // Strong citation links
                inhibitory: 0.2,  // Some skeptical inhibition
                modulatory: 0.6,  // Evidence-based learning
            },
            Domain::General => Self::default(),
        }
    }
}
```

### GraphEdge Structure

```rust
/// An edge connecting two MemoryNodes in the knowledge graph.
///
/// Edges represent relationships between knowledge nodes with
/// Marblestone neuroscience-inspired modulation for domain-aware
/// traversal and sleep-based consolidation shortcuts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Unique identifier for this edge
    pub id: Uuid,

    /// Source node UUID
    pub source_id: Uuid,

    /// Target node UUID
    pub target_id: Uuid,

    /// Relationship type (e.g., "causes", "related_to", "contradicts")
    pub relationship_type: String,

    /// Base edge weight [0, 1]
    pub weight: f32,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last traversal timestamp
    pub last_traversed: DateTime<Utc>,

    /// Number of times this edge has been traversed
    pub traversal_count: u32,

    /// Neurotransmitter weight profile for domain-aware modulation
    pub neurotransmitter_weights: NeurotransmitterWeights,

    /// Whether this edge was created as an amortized shortcut during sleep replay
    pub is_amortized_shortcut: bool,

    /// Steering reward signal from last traversal [-1.0, 1.0]
    /// Positive values reinforce the edge, negative values weaken it
    pub steering_reward: f32,

    /// Domain classification for neurotransmitter profile selection
    pub domain: Domain,

    /// Confidence score for this edge relationship [0, 1]
    pub confidence: f32,

    /// Optional metadata for edge-specific properties
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GraphEdge {
    /// Compute effective edge weight considering neurotransmitter modulation.
    pub fn effective_weight(&self) -> f32 {
        let base = self.weight;
        let neuro_mod = self.neurotransmitter_weights.net_activation();
        let reward_factor = 1.0 + (self.steering_reward * 0.2);

        (base * (0.5 + 0.5 * neuro_mod) * reward_factor)
            .clamp(0.0, 1.0)
    }

    /// Apply steering reward from traversal feedback.
    pub fn apply_steering(&mut self, reward: f32) {
        self.steering_reward = reward.clamp(-1.0, 1.0);
        // Decay factor for temporal credit assignment
        let decay = 0.9;
        self.steering_reward *= decay;
    }

    /// Check if this is a high-value shortcut edge.
    pub fn is_valuable_shortcut(&self) -> bool {
        self.is_amortized_shortcut
            && self.traversal_count > 5
            && self.steering_reward > 0.3
    }
}
```

---

## Appendix B: MCP Tool Response Examples

### store_memory Response

```json
{
  "node_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-12-31T10:30:00Z",
  "johari_quadrant": "unknown",
  "utl_state": {
    "delta_s": 0.5,
    "delta_c": 0.5,
    "w_e": 1.0,
    "phi": 0.0
  },
  "pulse": {
    "entropy": 0.52,
    "coherence": 0.68,
    "suggested_action": "ready"
  }
}
```

### recall_memory Response

```json
{
  "nodes": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440001",
      "content": "Relevant memory content...",
      "importance": 0.8,
      "relevance_score": 0.95,
      "johari_quadrant": "open",
      "created_at": "2025-12-30T14:20:00Z"
    },
    {
      "id": "550e8400-e29b-41d4-a716-446655440002",
      "content": "Another relevant memory...",
      "importance": 0.6,
      "relevance_score": 0.82,
      "johari_quadrant": "hidden",
      "created_at": "2025-12-29T09:15:00Z"
    }
  ],
  "pulse": {
    "entropy": 0.45,
    "coherence": 0.75,
    "suggested_action": "ready"
  }
}
```

### inject_context Response

```json
{
  "context": "Based on your query about authentication, here are the relevant findings: [node_abc123] JWT tokens should expire after 24 hours. [node_def456] OAuth2 provides robust third-party authentication...",
  "tokens_used": 180,
  "tokens_before_distillation": 450,
  "distillation_applied": "narrative",
  "compression_ratio": 0.6,
  "nodes_retrieved": [
    "550e8400-e29b-41d4-a716-446655440001",
    "550e8400-e29b-41d4-a716-446655440002"
  ],
  "utl_metrics": {
    "entropy": 0.38,
    "coherence": 0.82,
    "learning_score": 0.65
  },
  "pulse": {
    "entropy": 0.38,
    "coherence": 0.82,
    "suggested_action": "ready"
  }
}
```

### get_memetic_status Response

```json
{
  "coherence_score": 0.75,
  "entropy_level": 0.42,
  "top_active_concepts": [
    {"id": "...", "name": "Authentication", "access_count": 15},
    {"id": "...", "name": "Database Design", "access_count": 12},
    {"id": "...", "name": "API Security", "access_count": 8}
  ],
  "suggested_action": "ready",
  "dream_available": true,
  "curation_tasks": [],
  "pulse": {
    "entropy": 0.42,
    "coherence": 0.75,
    "suggested_action": "ready"
  }
}
```

### set_verbosity Response

```json
{
  "previous_level": 1,
  "new_level": 2,
  "pulse": {
    "entropy": 0.45,
    "coherence": 0.72,
    "suggested_action": "ready"
  }
}
```

---

## Appendix C: Requirement Traceability Matrix

| Requirement ID | User Story | Test Cases | Priority | Status |
|---------------|------------|------------|----------|--------|
| REQ-CORE-001 | US-CORE-01, US-CORE-02 | TC-CORE-001, TC-CORE-002 | must | pending |
| REQ-CORE-002 | US-CORE-09 | TC-CORE-003 | must | pending |
| REQ-CORE-003 | US-CORE-08 | TC-CORE-004 | must | pending |
| REQ-CORE-004 | US-CORE-08 | TC-CORE-005 | must | pending |
| REQ-CORE-005 | US-CORE-05 | TC-CORE-006 | must | pending |
| REQ-CORE-006 | US-CORE-01 | TC-CORE-007 | must | pending |
| REQ-CORE-007 | US-CORE-06 | TC-CORE-008 | must | pending |
| REQ-CORE-008 | US-CORE-06 | TC-CORE-009 | must | pending |
| REQ-CORE-009 | US-CORE-10 | TC-CORE-010 | must | pending |
| REQ-CORE-010 | US-CORE-07 | TC-CORE-011 | must | pending |
| REQ-CORE-011 | US-CORE-07 | TC-CORE-012, TC-CORE-013 | must | pending |
| REQ-CORE-012 | US-CORE-06 | TC-CORE-014 | must | pending |
| REQ-CORE-013 | US-CORE-06 | TC-CORE-015, TC-CORE-016 | must | pending |
| REQ-CORE-014 | US-CORE-01 | TC-CORE-017, TC-CORE-018 | must | pending |
| REQ-CORE-015 | US-CORE-02 | TC-CORE-019, TC-CORE-020 | must | pending |
| REQ-CORE-016 | US-CORE-03 | TC-CORE-021 | must | pending |
| REQ-CORE-017 | US-CORE-04 | TC-CORE-022 | must | pending |
| REQ-CORE-018 | US-CORE-05 | TC-CORE-023 | must | pending |
| REQ-CORE-019 | US-CORE-08 | TC-CORE-024 | must | pending |
| REQ-CORE-020 | US-CORE-05 | TC-CORE-025 | must | pending |
| REQ-CORE-021 | US-CORE-05 | TC-CORE-026 | must | pending |
| REQ-CORE-022 | US-CORE-05 | TC-CORE-027 | must | pending |
| REQ-CORE-023 | US-CORE-04, US-CORE-05 | TC-CORE-023 | must | pending |
| REQ-CORE-024 | US-CORE-08, US-CORE-09 | TC-CORE-024 | must | pending |
| REQ-CORE-025 | US-CORE-08, US-CORE-09 | TC-CORE-024 | must | pending |
| REQ-CORE-026 | US-CORE-01, US-CORE-02, US-CORE-03 | All error tests | must | pending |
| REQ-CORE-027 | US-CORE-01 | TC-CORE-030 | must | pending |

---

## Appendix D: Module 1 Trait References

This module implements the following traits defined in Module 1 (Ghost System):

| Trait | REQ ID | Implementation Notes |
|-------|--------|---------------------|
| UTLProcessor | REQ-GHOST-007 | REQ-CORE-024, REQ-CORE-025 implement compute methods with heuristics |
| EmbeddingProvider | REQ-GHOST-008 | Stub used until Module 3; storage accepts 1536D vectors |
| MemoryStore | REQ-GHOST-009 | REQ-CORE-007 through REQ-CORE-013 implement RocksDB backend |
| NervousLayer | REQ-GHOST-010 | Not implemented in Module 2; used by Module 6 |

---

*Document generated: 2025-12-31*
*Specification Version: 1.0*
*Module: Core Infrastructure (Phase 1)*
