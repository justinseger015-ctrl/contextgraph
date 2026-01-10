# TASK-LOGIC-009: Goal Discovery Pipeline

```xml
<task_spec id="TASK-LOGIC-009" version="2.0">
<metadata>
  <title>Implement Goal Discovery Pipeline</title>
  <status>done</status>
  <layer>logic</layer>
  <sequence>19</sequence>
  <implements>
    <requirement_ref>REQ-GOAL-DISCOVERY-01</requirement_ref>
    <requirement_ref>REQ-AUTONOMOUS-GOALS-01</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="DONE">TASK-LOGIC-004</task_ref>
    <task_ref status="DONE">TASK-LOGIC-008</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
</metadata>

<current_state_audit date="2026-01-09">
  <verification>
    This section documents the ACTUAL state of the codebase as verified by running
    commands and reading source files. ALL paths have been verified to exist.
  </verification>

  <dependencies_status>
    <dependency id="TASK-LOGIC-004" status="DONE">
      TeleologicalComparator exists at:
      crates/context-graph-core/src/teleological/comparator.rs
      - Implements apples-to-apples comparison per ARCH-02
      - Supports dense, sparse, and token-level similarity functions
      - Returns ComparisonResult with per-embedder scores
    </dependency>
    <dependency id="TASK-LOGIC-008" status="DONE">
      5-Stage Retrieval Pipeline exists at:
      crates/context-graph-storage/src/teleological/search/pipeline.rs
      - 21 tests passing (verified: cargo test -p context-graph-storage pipeline)
      - Exports: RetrievalPipeline, PipelineBuilder, PipelineConfig, StageConfig
      - Implements: SPLADE→Matryoshka→RRF→Alignment→MaxSim stages
    </dependency>
  </dependencies_status>

  <existing_code>
    <file path="crates/context-graph-core/src/autonomous/services/subgoal_discovery.rs">
      FOUNDATION CODE EXISTS - Contains:
      - MemoryCluster struct (centroid, members, coherence, label, avg_alignment)
      - DiscoveryConfig struct (min_cluster_size, min_coherence, emergence_threshold)
      - DiscoveryResult struct (candidates, cluster_count, avg_confidence)
      - SubGoalDiscovery service with discover_from_clusters(), extract_candidate()
      - compute_confidence() method (40% coherence, 30% size, 30% alignment)
      - determine_level() for GoalLevel assignment
      NOTE: This uses single-vector centroids, NOT TeleologicalArray centroids.
      MUST BE REFACTORED to use TeleologicalArray for ARCH-01 compliance.
    </file>
    <file path="crates/context-graph-core/src/autonomous/services/mod.rs">
      Module already exports subgoal_discovery
    </file>
  </existing_code>

  <files_to_create>
    <file path="crates/context-graph-core/src/autonomous/discovery.rs">
      NEW FILE - Main GoalDiscoveryPipeline implementation
      This is the primary deliverable of TASK-LOGIC-009
    </file>
  </files_to_create>

  <correct_paths>
    <path purpose="comparator">crates/context-graph-core/src/teleological/comparator.rs</path>
    <path purpose="embedder">crates/context-graph-core/src/teleological/embedder.rs</path>
    <path purpose="types">crates/context-graph-core/src/teleological/types.rs</path>
    <path purpose="pipeline">crates/context-graph-storage/src/teleological/search/pipeline.rs</path>
    <path purpose="search_module">crates/context-graph-storage/src/teleological/search/mod.rs</path>
    <path purpose="existing_subgoal">crates/context-graph-core/src/autonomous/services/subgoal_discovery.rs</path>
  </correct_paths>

  <incorrect_paths_in_previous_version>
    <wrong>crates/context-graph-core/src/teleology/array.rs</wrong>
    <correct>crates/context-graph-core/src/teleological/types.rs (contains TeleologicalArray)</correct>
    <wrong>crates/context-graph-core/src/teleology/comparator.rs</wrong>
    <correct>crates/context-graph-core/src/teleological/comparator.rs</correct>
    <wrong>crates/context-graph-storage/src/teleological/search/engine.rs</wrong>
    <correct>crates/context-graph-storage/src/teleological/search/pipeline.rs</correct>
  </incorrect_paths_in_previous_version>
</current_state_audit>

<context>
Goal discovery replaces manual North Star creation with autonomous clustering-based goal
emergence. Goals are discovered by analyzing teleological arrays, computing cluster
centroids, and assigning appropriate goal levels (NorthStar, Strategic, Tactical, Immediate).

CRITICAL: The existing subgoal_discovery.rs uses single-vector centroids. This task
must implement GoalDiscoveryPipeline that uses FULL TeleologicalArray centroids
(13 embedders) to comply with ARCH-01: "TeleologicalArray is Atomic".
</context>

<objective>
Implement GoalDiscoveryPipeline that uses K-means/HDBSCAN clustering on teleological arrays
to discover emergent goals, compute centroids as goal candidates, and build goal hierarchies.
The pipeline must compare TeleologicalArray-to-TeleologicalArray using TeleologicalComparator.
</objective>

<rationale>
Autonomous goal discovery solves the "apples-to-oranges" problem:
1. Goals are teleological arrays (13 embedders), not single embeddings
2. Goals emerge from data patterns, not manual specification
3. Cluster centroids are valid teleological arrays comparable to memories
4. Goal hierarchies form naturally from cluster relationships

From constitution.yaml:
- ARCH-01: TeleologicalArray is atomic - never split, compare only as unit
- ARCH-02: Apples-to-apples - E1 compares with E1, never cross-embedder
- ARCH-03: Autonomous operation - goals emerge from data patterns
</rationale>

<design_philosophy>
  FAIL FAST. NO FALLBACKS. NO RECOVERY ATTEMPTS.

  All errors are FATAL:
  - InsufficientData → panic with clear message
  - ClusteringFailed → panic with algorithm details
  - NoClustersFound → panic with threshold info
  - InvalidCentroid → panic with embedder state

  This ensures:
  - Bugs caught early in development
  - Data integrity preserved
  - Clear error messages for debugging
  - No silent failures or degraded states
</design_philosophy>

<input_context_files>
  <file purpose="teleological_types" verified="true">
    crates/context-graph-core/src/teleological/types.rs
  </file>
  <file purpose="comparator" verified="true">
    crates/context-graph-core/src/teleological/comparator.rs
  </file>
  <file purpose="pipeline" verified="true">
    crates/context-graph-storage/src/teleological/search/pipeline.rs
  </file>
  <file purpose="existing_subgoal_discovery" verified="true">
    crates/context-graph-core/src/autonomous/services/subgoal_discovery.rs
  </file>
  <file purpose="embedder_definitions" verified="true">
    crates/context-graph-core/src/teleological/embedder.rs
  </file>
</input_context_files>

<prerequisites>
  <check status="VERIFIED">TASK-LOGIC-004 complete (TeleologicalComparator exists at teleological/comparator.rs)</check>
  <check status="VERIFIED">TASK-LOGIC-008 complete (21 tests passing in pipeline.rs)</check>
  <check status="VERIFY_BEFORE_START">TASK-CORE-005 complete (GoalNode uses teleological arrays)</check>
  <check status="VERIFY_BEFORE_START">Embedder enum exists with all() iterator in teleological/embedder.rs</check>
</prerequisites>

<scope>
  <in_scope>
    <item>Create GoalDiscoveryPipeline struct in crates/context-graph-core/src/autonomous/discovery.rs</item>
    <item>Implement K-means clustering for teleological arrays (13-embedder vectors)</item>
    <item>Compute cluster centroids as TeleologicalArray (average each embedder separately)</item>
    <item>Score candidates by coherence, size, and embedder distribution</item>
    <item>Assign goal levels (NorthStar: size≥50/coherence≥0.85, Strategic: ≥20/0.80, Tactical: ≥10/0.75, Immediate: else)</item>
    <item>Build parent-child relationships based on centroid similarity</item>
    <item>Integrate with existing SubGoalDiscovery service (delegate or replace)</item>
  </in_scope>
  <out_of_scope>
    <item>Drift detection (TASK-LOGIC-010)</item>
    <item>MCP handler implementation (TASK-INTEG-*)</item>
    <item>HDBSCAN implementation (K-means is primary, HDBSCAN is stretch goal)</item>
    <item>Spectral clustering (stretch goal)</item>
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/autonomous/discovery.rs">
      use crate::teleological::types::TeleologicalArray;
      use crate::teleological::comparator::{TeleologicalComparator, ComparisonResult};
      use crate::teleological::embedder::Embedder;
      use crate::autonomous::evolution::GoalLevel;

      /// Configuration for goal discovery.
      #[derive(Debug, Clone)]
      pub struct DiscoveryConfig {
          pub sample_size: usize,           // Max arrays to process (default: 500)
          pub min_cluster_size: usize,      // Min members per cluster (default: 5)
          pub min_coherence: f32,           // Min intra-cluster similarity (default: 0.75)
          pub clustering_algorithm: ClusteringAlgorithm,
          pub num_clusters: NumClusters,
      }

      #[derive(Debug, Clone)]
      pub enum ClusteringAlgorithm {
          KMeans,  // PRIMARY - must implement
          HDBSCAN { min_samples: usize },  // STRETCH GOAL
          Spectral { n_neighbors: usize }, // STRETCH GOAL
      }

      #[derive(Debug, Clone)]
      pub enum NumClusters {
          Auto,                         // sqrt(n/2) heuristic
          Fixed(usize),                 // Exact k
          Range { min: usize, max: usize }, // Elbow method
      }

      /// Goal discovery pipeline for autonomous goal emergence.
      pub struct GoalDiscoveryPipeline {
          comparator: TeleologicalComparator,
      }

      impl GoalDiscoveryPipeline {
          pub fn new(comparator: TeleologicalComparator) -> Self;

          /// Discover goals from teleological arrays.
          /// FAILS FAST on any error - no recovery attempts.
          pub fn discover(
              &self,
              arrays: &[TeleologicalArray],
              config: &DiscoveryConfig,
          ) -> DiscoveryResult;  // NO Result wrapper - panics on failure

          /// Cluster arrays using K-means on full teleological arrays.
          fn cluster(
              &self,
              arrays: &[TeleologicalArray],
              config: &DiscoveryConfig,
          ) -> Vec&lt;Cluster&gt;;

          /// Compute centroid for a cluster.
          /// Each embedder's vectors are averaged separately.
          /// Result is a valid TeleologicalArray.
          fn compute_centroid(
              &self,
              members: &[&TeleologicalArray],
          ) -> TeleologicalArray;

          /// Score cluster suitability as a goal.
          fn score_cluster(&self, cluster: &Cluster) -> GoalCandidate;

          /// Assign goal level: NorthStar/Strategic/Tactical/Immediate
          fn assign_level(&self, candidate: &GoalCandidate) -> GoalLevel;

          /// Build parent-child relationships between goals.
          fn build_hierarchy(
              &self,
              candidates: &[GoalCandidate],
          ) -> Vec&lt;GoalRelationship&gt;;
      }

      /// Cluster of teleological arrays.
      #[derive(Debug, Clone)]
      pub struct Cluster {
          pub members: Vec&lt;usize&gt;,         // Indices into source array
          pub centroid: TeleologicalArray,  // FULL 13-embedder centroid
          pub coherence: f32,               // Intra-cluster similarity
      }

      /// Discovered goal from clustering.
      #[derive(Debug)]
      pub struct DiscoveredGoal {
          pub goal_id: String,
          pub description: String,
          pub level: GoalLevel,
          pub confidence: f32,
          pub member_count: usize,
          pub centroid: TeleologicalArray,  // FULL 13-embedder goal vector
          pub dominant_embedders: Vec&lt;Embedder&gt;,  // Top 3 by magnitude
          pub coherence_score: f32,
      }

      /// Result of goal discovery.
      #[derive(Debug)]
      pub struct DiscoveryResult {
          pub discovered_goals: Vec&lt;DiscoveredGoal&gt;,
          pub clusters_found: usize,
          pub total_arrays_analyzed: usize,
          pub hierarchy: Vec&lt;GoalRelationship&gt;,
      }
    </signature>
  </signatures>

  <constraints>
    <constraint id="ARCH-01">Centroids MUST be valid TeleologicalArray (all 13 embedders)</constraint>
    <constraint id="ARCH-02">Comparison uses TeleologicalComparator (apples-to-apples per embedder)</constraint>
    <constraint id="ARCH-03">Goals emerge from data - no manual goal specification</constraint>
    <constraint id="FAIL-FAST">All errors panic with descriptive messages - no Result/Option wrapping</constraint>
    <constraint id="REAL-DATA">Tests use real TeleologicalArray instances, NOT mock data</constraint>
  </constraints>

  <verification>
    <command>cargo test -p context-graph-core autonomous::discovery -- --nocapture</command>
    <expected_output>All tests pass, no warnings</expected_output>
  </verification>
</definition_of_done>

<full_state_verification protocol="MANDATORY">
  <phase name="source_of_truth">
    Before implementing, verify these files exist and contain expected types:
    <check>Read crates/context-graph-core/src/teleological/types.rs - verify TeleologicalArray struct</check>
    <check>Read crates/context-graph-core/src/teleological/embedder.rs - verify Embedder enum with all()</check>
    <check>Read crates/context-graph-core/src/teleological/comparator.rs - verify TeleologicalComparator</check>
    <check>Read crates/context-graph-core/src/autonomous/evolution.rs - verify GoalLevel enum</check>
    <check>Run: cargo test -p context-graph-storage pipeline - must show 21 tests passing</check>
  </phase>

  <phase name="execute_and_inspect">
    After implementation, verify outputs:
    <check>Run: cargo test -p context-graph-core autonomous::discovery -- --nocapture</check>
    <check>Inspect test output for: cluster formation, centroid computation, level assignment</check>
    <check>Verify no warnings about unused code or missing derives</check>
  </phase>

  <phase name="boundary_and_edge_cases">
    Test edge cases with real data:
    <check>Empty input array → must panic with "Insufficient arrays" message</check>
    <check>Single array input → must panic (cannot form clusters)</check>
    <check>Arrays with missing embedders → centroid uses available embedders only</check>
    <check>All arrays identical → single cluster with coherence 1.0</check>
    <check>Widely dispersed arrays → multiple low-coherence clusters</check>
  </phase>

  <phase name="evidence_of_success">
    Collect proof that implementation works:
    <check>Test output shows discovered goals with correct levels</check>
    <check>Centroids are valid TeleologicalArray (log embedder count)</check>
    <check>Hierarchy relationships computed (log parent→child edges)</check>
    <check>No panics in success paths, clear panics in error paths</check>
  </phase>
</full_state_verification>

<testing_requirements>
  <rule id="NO-MOCK-DATA">
    NEVER use mock data in tests. All tests must use:
    - Real TeleologicalArray instances created via TeleologicalArray::new()
    - Real embeddings generated or loaded from test fixtures
    - Real comparisons via TeleologicalComparator
  </rule>

  <rule id="SYNTHETIC-DATA-OK">
    Synthetic test data IS allowed when:
    - Input vectors have known mathematical properties
    - Expected outputs can be computed analytically
    - Example: Three tight clusters of 10 vectors each, with known centroids
  </rule>

  <test name="test_kmeans_three_clusters">
    <description>Create 30 TeleologicalArrays in 3 known clusters, verify K-means finds them</description>
    <synthetic_data>
      Cluster A: 10 arrays with E1 vectors centered at [1,0,0,...]
      Cluster B: 10 arrays with E1 vectors centered at [0,1,0,...]
      Cluster C: 10 arrays with E1 vectors centered at [0,0,1,...]
      Add small noise (±0.1) to each vector
    </synthetic_data>
    <expected>
      3 clusters found
      Each cluster has 10 members
      Centroids close to [1,0,0], [0,1,0], [0,0,1]
      Coherence > 0.9 for each cluster
    </expected>
  </test>

  <test name="test_centroid_is_valid_teleological_array">
    <description>Verify computed centroid has all 13 embedders populated</description>
    <verification>
      For each embedder E1-E13:
        assert!(centroid.get(embedder).is_some())
        assert!(centroid.get(embedder).dimension() == expected_dimension)
    </verification>
  </test>

  <test name="test_goal_level_assignment">
    <description>Verify level assignment thresholds</description>
    <cases>
      size=50, coherence=0.85 → NorthStar
      size=49, coherence=0.85 → Strategic (size threshold not met)
      size=50, coherence=0.84 → Strategic (coherence threshold not met)
      size=20, coherence=0.80 → Strategic
      size=10, coherence=0.75 → Tactical
      size=5, coherence=0.70 → Immediate
    </cases>
  </test>

  <test name="test_hierarchy_construction">
    <description>Verify parent-child relationships based on centroid similarity</description>
    <synthetic_data>
      Parent cluster: Large (50 members), high coherence
      Child cluster: Small (10 members), similar centroid (similarity > 0.7)
    </synthetic_data>
    <expected>
      Hierarchy contains edge from parent → child
      Similarity recorded in GoalRelationship
    </expected>
  </test>

  <test name="test_fail_fast_empty_input">
    <description>Verify panic on empty input</description>
    <input>Empty slice of TeleologicalArray</input>
    <expected>Panic with message containing "Insufficient"</expected>
  </test>

  <test name="test_fail_fast_insufficient_data">
    <description>Verify panic when input smaller than min_cluster_size</description>
    <input>3 TeleologicalArrays with min_cluster_size=5</input>
    <expected>Panic with message containing count and minimum</expected>
  </test>
</testing_requirements>

<manual_testing_protocol>
  <step order="1">
    Create test binary: examples/discover_goals.rs
    Load real TeleologicalArrays from test database or fixtures
    Run discovery with default config
    Print discovered goals, levels, and hierarchy
  </step>
  <step order="2">
    Verify output manually:
    - Goal IDs are unique UUIDs
    - Descriptions are meaningful (not empty)
    - Levels follow size/coherence rules
    - Hierarchy edges are sensible
  </step>
  <step order="3">
    Test with known data:
    Create arrays representing 3 distinct topics
    Verify discovery finds 3 goals corresponding to topics
  </step>
</manual_testing_protocol>

<files_to_create>
  <file path="crates/context-graph-core/src/autonomous/discovery.rs">
    Goal discovery pipeline implementation - PRIMARY DELIVERABLE
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/autonomous/mod.rs">
    Add: pub mod discovery;
    Add: pub use discovery::{GoalDiscoveryPipeline, DiscoveryConfig, DiscoveryResult};
  </file>
  <file path="crates/context-graph-core/Cargo.toml">
    Verify: rand dependency exists (for sampling)
    Verify: uuid dependency exists (for goal IDs)
  </file>
</files_to_modify>

<integration_with_existing_code>
  <note>
    The existing subgoal_discovery.rs provides a simpler interface for cluster-based
    discovery. The new GoalDiscoveryPipeline should:
    1. Use full TeleologicalArray centroids (not single-vector)
    2. Integrate with TeleologicalComparator for similarity
    3. Either replace or wrap SubGoalDiscovery service

    Decision: CREATE NEW discovery.rs alongside subgoal_discovery.rs
    The new pipeline is the "full" implementation; subgoal_discovery.rs
    can later be refactored to delegate to it.
  </note>
</integration_with_existing_code>

<pseudo_code>
See previous version for full implementation pseudocode.
Key changes in this version:
1. All paths corrected to use "teleological" not "teleology"
2. Fail-fast design: discover() returns DiscoveryResult directly, panics on error
3. Tests must use real TeleologicalArray instances, not mocks
4. Full State Verification protocol must be followed
</pseudo_code>

<validation_criteria>
  <criterion>K-means clustering produces valid clusters from real TeleologicalArrays</criterion>
  <criterion>Centroids are valid TeleologicalArray with all 13 embedders populated</criterion>
  <criterion>Goal levels assigned correctly per size/coherence thresholds</criterion>
  <criterion>Hierarchy relationships computed based on centroid similarity</criterion>
  <criterion>Fail-fast behavior on all error conditions</criterion>
  <criterion>No mock data in any test</criterion>
</validation_criteria>

<test_commands>
  <command>cargo test -p context-graph-core autonomous::discovery -- --nocapture</command>
  <command>cargo clippy -p context-graph-core -- -D warnings</command>
  <command>cargo doc -p context-graph-core --no-deps</command>
</test_commands>

<git_history_notes date="2026-01-09">
  Recent commits related to this task:
  - 9652231: feat(TASK-LOGIC-007): implement matrix strategy search with 13x13 correlation weights
  - 632fbc2: feat(TASK-LOGIC-006): implement multi-embedder parallel search
  - d33c70e: feat(TASK-LOGIC-005): implement single embedder HNSW search
  - bd0a278: feat(TASK-LOGIC-004): implement TeleologicalComparator with batch parallel support

  The TeleologicalComparator (bd0a278) is a direct dependency and is COMPLETE.
  Pipeline implementation exists in pipeline.rs (not yet committed separately).
</git_history_notes>

<completion date="2026-01-09">
  <implementation_summary>
    Created crates/context-graph-core/src/autonomous/discovery.rs with:
    - GoalDiscoveryPipeline struct wrapping TeleologicalComparator
    - K-means clustering with k-means++ initialization on TeleologicalArrays
    - compute_centroid() averaging all 13 embedders (dense, sparse, token-level)
    - Goal level assignment per size/coherence thresholds:
      - NorthStar: size>=50, coherence>=0.85
      - Strategic: size>=20, coherence>=0.80
      - Tactical: size>=10, coherence>=0.75
      - Operational: else
    - Hierarchy building based on centroid similarity (>=0.5 threshold)
    - Fail-fast design with panics on all error conditions

    Updated autonomous/mod.rs with discovery module exports.
    Added type alias: pub type GoalCandidate = DiscoveredGoal;

    Tests use real synthetic TeleologicalArray data with known mathematical properties.
    Code reviewed and simplified per code-simplifier recommendations.
  </implementation_summary>

  <test_results>
    cargo test -p context-graph-core autonomous::discovery: 10 passed, 0 failed
    cargo clippy -p context-graph-core --lib -- -D warnings: 0 warnings

    Tests verify:
    - K-means clustering produces correct cluster assignments
    - Centroids are valid TeleologicalArrays with all embedders
    - Goal levels assigned correctly based on thresholds
    - Hierarchy relationships computed from centroid similarity
    - Fail-fast behavior on empty input and insufficient data
  </test_results>

  <files_created_or_modified>
    - crates/context-graph-core/src/autonomous/discovery.rs (NEW - 840 lines)
    - crates/context-graph-core/src/autonomous/mod.rs (MODIFIED - added discovery exports)
  </files_created_or_modified>
</completion>
</task_spec>
```
