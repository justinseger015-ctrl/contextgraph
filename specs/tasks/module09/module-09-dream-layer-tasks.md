# Module 9: Dream Layer - Atomic Tasks

```yaml
metadata:
  module: Module 9 - Dream Layer
  phase: 8
  approach: inside-out-bottom-up
  spec_refs:
    - SPEC-DREAM-009 (Functional)
    - TECH-DREAM-009 (Technical)
  total_tasks: 25
  created: 2025-12-31
  performance_targets:
    nrem_duration: 3 minutes
    rem_duration: 2 minutes
    wake_latency: < 100ms (P95)
    gpu_usage: < 30%
    compression_ratio: 10:1
    blind_spots_per_cycle: > 5
    information_loss: < 15%
  dependencies:
    - Module 8 (GPU Direct Storage)
    - Module 6 (Bio-Nervous System)
    - Module 4 (Knowledge Graph)
  layers:
    foundation: 8 tasks
    logic: 12 tasks
    surface: 5 tasks
```

---

## Foundation Layer Tasks (Build First)

These tasks establish core types, state machine, and configuration structures for the Dream Layer.

```yaml
- id: TASK-DREAM-001
  title: Create dream module structure with feature flags
  type: implementation
  layer: foundation
  requirement_refs: [REQ-DREAM-001]
  dependencies: [Module 6 nervous system infrastructure]
  acceptance_criteria:
    - crates/context-graph-dream/src/lib.rs module entry with public exports
    - Cargo.toml includes all required dependencies (tokio, uuid, chrono, rand)
    - Feature flag "dream" enables Dream Layer compilation
    - Module compiles with and without dream feature
    - Re-exports DreamLayer, NREMPhase, REMPhase, SRCAlgorithm from submodules
    - cargo check -p context-graph-dream succeeds
  estimated_complexity: low
  files_affected:
    - crates/context-graph-dream/Cargo.toml
    - crates/context-graph-dream/src/lib.rs
    - crates/context-graph-dream/src/layer.rs

- id: TASK-DREAM-002
  title: Implement DreamState enum and state machine transitions
  type: implementation
  layer: foundation
  requirement_refs: [REQ-DREAM-001, REQ-DREAM-002]
  dependencies: [TASK-DREAM-001]
  acceptance_criteria:
    - DreamState enum with Awake, EnteringNREM, NREM, Transitioning, REM, Waking variants
    - NREMSubPhase enum with MemorySelection, HebbianReplay, Consolidation, RedundancyElimination, SchemaExtraction
    - WakeReason enum with UserQuery, ResourcePressure, CycleComplete, ManualAbort, Error variants
    - is_dreaming() method returns true for active dream states
    - phase_name() returns string for logging
    - State transitions are validated (no invalid transitions)
    - Unit tests verify all state transitions
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-dream/src/state.rs

- id: TASK-DREAM-003
  title: Implement DreamTrigger condition checker
  type: implementation
  layer: foundation
  requirement_refs: [REQ-DREAM-002]
  dependencies: [TASK-DREAM-002]
  acceptance_criteria:
    - DreamTrigger struct with activity_threshold (0.15), idle_duration (10min), cooldown_period (30min)
    - memory_pressure_threshold (0.8) for forced trigger
    - Default impl sets all specified default values
    - should_trigger() returns TriggerDecision enum
    - TriggerDecision::Trigger(reason) when conditions met
    - TriggerDecision::Wait when conditions not met
    - TriggerDecision::Blocked(reason) when cooldown active or high activity
    - record_completion() updates last_dream_completed timestamp
    - Unit tests verify trigger logic with various conditions
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-dream/src/trigger.rs

- id: TASK-DREAM-004
  title: Implement AbortController for wake-on-query
  type: implementation
  layer: foundation
  requirement_refs: [REQ-DREAM-003, REQ-DREAM-023]
  dependencies: [TASK-DREAM-002]
  acceptance_criteria:
    - AbortController struct with tokio::sync::watch channel
    - wake_latency_budget field (default 100ms)
    - set_sender() stores watch::Sender<bool>
    - abort(WakeReason) sends abort signal and returns AbortResult
    - AbortResult includes latency, checkpoint_saved, reason
    - Abort completes within 100ms budget
    - save_checkpoint() and clear_checkpoint() manage DreamCheckpoint
    - Warn logged if latency exceeds budget
    - Unit tests verify abort propagates within 10ms
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-dream/src/abort.rs

- id: TASK-DREAM-005
  title: Implement DreamError type hierarchy
  type: implementation
  layer: foundation
  requirement_refs: [REQ-DREAM-032]
  dependencies: [TASK-DREAM-001]
  acceptance_criteria:
    - DreamError enum with thiserror derive
    - Variants: DreamInProgress, DreamCooldown, DreamAborted, DreamTimeout
    - Variants: DreamResourceError, DreamCheckpointError, DreamConsolidationError
    - Variants: DreamExplorationError, AbortChannelClosed, InvalidState
    - Error codes match specification (-32100 to -32107)
    - All variants have descriptive #[error()] messages
    - DreamResult<T> type alias defined
    - Error type is Send + Sync
    - Unit tests verify error display formatting
  estimated_complexity: low
  files_affected:
    - crates/context-graph-dream/src/error.rs

- id: TASK-DREAM-006
  title: Implement DreamCheckpoint for state persistence
  type: implementation
  layer: foundation
  requirement_refs: [REQ-DREAM-025, REQ-DREAM-028]
  dependencies: [TASK-DREAM-002, TASK-DREAM-005]
  acceptance_criteria:
    - DreamCheckpoint struct with id, created_at, phase, progress fields
    - src_state field for SRCState serialization
    - pending_operations Vec<PendingOperation>
    - metrics_so_far field for partial metrics
    - Serialize/Deserialize derives for persistence
    - save() async method writes to disk atomically
    - load() async method reads from disk
    - resume() method restores DreamLayer state
    - Checkpoint size < 1MB for typical state
    - Unit tests verify round-trip serialization
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-dream/src/checkpoint.rs

- id: TASK-DREAM-007
  title: Implement ReplayPath and shortcut types (Marblestone)
  type: implementation
  layer: foundation
  requirement_refs: [REQ-DREAM-036, REQ-DREAM-037]
  dependencies: [TASK-DREAM-001]
  acceptance_criteria:
    - ReplayPath struct with nodes Vec<Uuid>, edges Vec<GraphEdge>, total_confidence f32
    - source() returns first node ID
    - target() returns last node ID
    - hop_count() returns edges.len()
    - EdgeMetadata struct includes origin, original_path_hops, creation_cycle
    - EdgeOrigin enum includes AmortizedShortcut variant
    - GraphEdge extended with is_amortized_shortcut bool, shortcut_path_length u32
    - Unit tests verify path confidence calculation
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-dream/src/src/shortcut.rs
    - crates/context-graph-core/src/graph/edge.rs

- id: TASK-DREAM-008
  title: Implement hyperbolic geometry utilities
  type: implementation
  layer: foundation
  requirement_refs: [REQ-DREAM-011]
  dependencies: [TASK-DREAM-001]
  acceptance_criteria:
    - HyperbolicPoint struct for Poincare ball coordinates
    - TangentVector struct for tangent space operations
    - norm(), norm_squared() methods for HyperbolicPoint
    - subtract(), scale(), mobius_add() for Poincare ball operations
    - random_unit() for TangentVector generation
    - blend() for mixing vectors with weight
    - to_embedding() converts HyperbolicPoint to Vector1536
    - project_to_ball() ensures ||x|| < 1
    - hyperbolic_exp_map() implements exponential map
    - Unit tests verify Poincare ball constraints
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/rem/hyperbolic.rs
```

---

## Logic Layer Tasks (Build Second)

These tasks implement NREM phase operations, REM exploration, and SRC algorithm core logic.

```yaml
- id: TASK-DREAM-009
  title: Implement MemoryReplayEngine with recency bias
  type: implementation
  layer: logic
  requirement_refs: [REQ-DREAM-004, REQ-DREAM-005]
  dependencies: [TASK-DREAM-001]
  acceptance_criteria:
    - MemoryReplayEngine struct with recency_bias (0.8), max_replay_count (1000), importance_floor (0.3)
    - select_memories() returns Vec<MemoryNode> with recency-weighted sampling
    - Weight formula: w(t) = recency_bias^(age_hours / 24)
    - Importance filter excludes nodes below floor
    - weighted_sample() function for sampling without replacement
    - replay_memory() activates node and propagates to neighbors
    - ReplayResult includes node_id and activations
    - Unit tests verify recency bias weighting
    - Integration test confirms 24h window selection
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/nrem/replay.rs

- id: TASK-DREAM-010
  title: Implement HebbianUpdater for weight updates
  type: implementation
  layer: logic
  requirement_refs: [REQ-DREAM-006]
  dependencies: [TASK-DREAM-009]
  acceptance_criteria:
    - HebbianUpdater struct with learning_rate (0.01), decay_rate (0.001)
    - weight_cap (1.0), weight_floor (0.05)
    - update_weight() applies: delta_w = eta * pre * post
    - Decay applied: decayed = weight * (1 - decay_rate)
    - Weight clamped to [weight_floor, weight_cap]
    - mark_for_pruning() called when weight <= floor
    - batch_update() processes replay sequence with SRC integration
    - HebbianUpdateStats tracks edges_strengthened, edges_weakened, edges_pruned
    - replay_paths extracted for shortcut creation
    - Unit tests verify Hebbian formula correctness
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/nrem/hebbian.rs

- id: TASK-DREAM-011
  title: Implement CouplingConsolidator for cluster merging
  type: implementation
  layer: logic
  requirement_refs: [REQ-DREAM-007]
  dependencies: [TASK-DREAM-010]
  acceptance_criteria:
    - CouplingConsolidator struct with coupling_threshold (0.7), max_cluster_size (10)
    - merge_similarity (0.9) for redundancy detection
    - find_coupled_clusters() uses community detection algorithm
    - Clusters filtered by size and internal coupling
    - consolidate_cluster() creates merged node with combined embedding
    - check_priors_compatibility() validates before merge
    - fuse_embeddings() combines node embeddings
    - transfer_edges() preserves connections to merged node
    - soft_delete() marks original nodes for recovery
    - ConsolidationResult includes merged_node_id, nodes_consolidated, compression_achieved
    - Unit tests verify cluster detection accuracy
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/nrem/consolidation.rs

- id: TASK-DREAM-012
  title: Implement RedundancyDetector for duplicate elimination
  type: implementation
  layer: logic
  requirement_refs: [REQ-DREAM-008]
  dependencies: [TASK-DREAM-011]
  acceptance_criteria:
    - RedundancyDetector struct with similarity_threshold (0.95), max_candidates (10000)
    - SurvivorStrategy enum: MostRecent, HighestImportance, MostConnected, Oldest, Merge
    - find_redundant_pairs() uses LSH for efficient similarity detection
    - RedundantPair struct with node_a, node_b, similarity
    - eliminate_redundancy() keeps survivor based on strategy
    - Edge connections transferred to survivor
    - RedundancyStats tracks pairs_found, nodes_eliminated, space_recovered_bytes
    - Eliminated nodes soft-deleted with 30-day recovery
    - Parallel processing with rayon for large candidate sets
    - Unit tests verify 0.95 similarity threshold
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/nrem/redundancy.rs

- id: TASK-DREAM-013
  title: Implement SchemaExtractor for pattern abstraction
  type: implementation
  layer: logic
  requirement_refs: [REQ-DREAM-009]
  dependencies: [TASK-DREAM-011]
  acceptance_criteria:
    - SchemaExtractor struct with min_cluster_size (5), abstraction_level (0.7)
    - max_schemas_per_cycle (20)
    - extract_schemas() returns Vec<ExtractedSchema> from clusters
    - ExtractedSchema includes id, name, template, source_nodes, confidence
    - find_common_edge_patterns() identifies structural patterns
    - find_variable_positions() extracts variable slots
    - SchemaTemplate struct for pattern representation
    - generate_schema_name() creates descriptive name
    - Confidence threshold filters low-quality schemas
    - Unit tests verify pattern extraction accuracy
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-dream/src/nrem/schema.rs

- id: TASK-DREAM-014
  title: Implement NREMPhase orchestrator
  type: implementation
  layer: logic
  requirement_refs: [REQ-DREAM-004]
  dependencies: [TASK-DREAM-009, TASK-DREAM-010, TASK-DREAM-011, TASK-DREAM-012, TASK-DREAM-013]
  acceptance_criteria:
    - NREMPhase struct with duration (3min), replay_recency_bias (0.8), learning_rate (0.01)
    - batch_size (64) for consolidation batches
    - execute() runs all sub-phases in sequence with abort checking
    - Phase 1: Memory selection
    - Phase 2: Hebbian replay with SRC
    - Phase 3: Amortized shortcut creation (Marblestone)
    - Phase 4: Tight coupling consolidation
    - Phase 5: Redundancy elimination
    - Phase 6: Schema extraction
    - NREMResult includes completed flag, metrics, replay_paths
    - calculate_compression() computes node ratio
    - Progress tracked through NREMSubPhase enum
    - Integration test verifies 3-minute timeout
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/nrem/mod.rs

- id: TASK-DREAM-015
  title: Implement SyntheticQueryGenerator for REM exploration
  type: implementation
  layer: logic
  requirement_refs: [REQ-DREAM-011]
  dependencies: [TASK-DREAM-008]
  acceptance_criteria:
    - SyntheticQueryGenerator struct with step_size (0.3), walk_length (5)
    - exploration_bias (0.7), curvature (-1.0 Poincare)
    - generate() starts from importance-weighted node
    - Random walk in hyperbolic space with exponential map
    - sample_tangent_vector() blends random with sparsity gradient
    - compute_sparsity_gradient() finds direction toward sparse regions
    - hyperbolic_exp_map() applies Poincare ball geometry
    - project_to_ball() ensures ||x|| < 0.99
    - SyntheticQuery includes embedding, path, origin_node
    - Unit tests verify walk stays in Poincare ball
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/rem/query_gen.rs

- id: TASK-DREAM-016
  title: Implement BlindSpotDetector for novel connection discovery
  type: implementation
  layer: logic
  requirement_refs: [REQ-DREAM-012]
  dependencies: [TASK-DREAM-015]
  acceptance_criteria:
    - BlindSpotDetector struct with min_semantic_distance (0.7), min_causal_overlap (0.3)
    - max_causal_depth (4) for path search
    - detect() finds nodes that are semantically far but causally connected
    - BlindSpot struct with node_a, node_b, semantic_distance, shared_causal_paths
    - discovery_confidence computed as overlap * distance
    - is_significant_blind_spot() checks thresholds
    - cosine_distance() helper for embedding comparison
    - find_shared_causal_paths() traverses graph up to depth
    - compute_path_overlap() calculates confidence overlap
    - Unit tests verify >5 blind spots per cycle target
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/rem/blind_spot.rs

- id: TASK-DREAM-017
  title: Implement EdgeCreator for dream-discovered connections
  type: implementation
  layer: logic
  requirement_refs: [REQ-DREAM-013]
  dependencies: [TASK-DREAM-016]
  acceptance_criteria:
    - EdgeCreator struct with initial_weight (0.3), initial_confidence (0.5)
    - edge_type = EdgeType::DreamDiscovered, max_new_edges (50)
    - create_from_blind_spot() creates GraphEdge with metadata
    - EdgeMetadata includes origin: DreamDiscovery
    - semantic_distance and causal_evidence recorded
    - discovery_cycle tracked for audit
    - Edge creation logged with info level
    - Maximum 50 edges per cycle enforced
    - Unit tests verify edge properties
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-dream/src/rem/edge_creator.rs

- id: TASK-DREAM-018
  title: Implement REMPhase orchestrator
  type: implementation
  layer: logic
  requirement_refs: [REQ-DREAM-010]
  dependencies: [TASK-DREAM-015, TASK-DREAM-016, TASK-DREAM-017]
  acceptance_criteria:
    - REMPhase struct with duration (2min), synthetic_query_count (100)
    - min_semantic_leap (0.7), exploration_temperature (2.0)
    - new_edge_weight (0.3), new_edge_confidence (0.5)
    - execute() generates queries and discovers blind spots
    - Abort signal checked between queries
    - REMResult includes queries_generated, connections_discovered, blind_spots_found
    - REMMetrics tracks synthetic_queries, blind_spots_found, new_edges_created
    - average_semantic_leap and exploration_coverage computed
    - compute_coverage() calculates node visit ratio
    - Integration test verifies 2-minute timeout
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/rem/mod.rs

- id: TASK-DREAM-019
  title: Implement SRCAlgorithm core logic
  type: implementation
  layer: logic
  requirement_refs: [REQ-DREAM-014]
  dependencies: [TASK-DREAM-007]
  acceptance_criteria:
    - SRCAlgorithm struct with sparsity (0.1), randomness (0.5), clustering (0.7)
    - decay_rate (0.9) for activation decay
    - SRCState struct with activations HashMap, step_count, history
    - step() decays activations, selects candidates, activates nodes
    - select_candidates() balances random (exploration) vs clustered (exploitation)
    - compute_activation() combines neighbor activation and importance
    - Prune activations below 0.01
    - SRCStepResult includes nodes_activated, average_activation, clustering_coefficient
    - History capped at 100 entries
    - Unit tests verify sparsity controls activation fraction
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/src/algorithm.rs
    - crates/context-graph-dream/src/src/state.rs

- id: TASK-DREAM-020
  title: Implement amortized_shortcut_creation (Marblestone)
  type: implementation
  layer: logic
  requirement_refs: [REQ-DREAM-036, REQ-DREAM-037]
  dependencies: [TASK-DREAM-019, TASK-DREAM-007]
  acceptance_criteria:
    - amortized_shortcut_creation() async method on SRCAlgorithm
    - Parameters: replay_paths, min_hops (3), confidence_threshold (0.7)
    - Skip paths with < 3 hops (REQ-DREAM-036)
    - Skip paths with confidence < 0.7 (REQ-DREAM-037 quality gate)
    - Path confidence = product of edge confidences
    - Path weight = product of edge weights
    - Created shortcut has is_amortized_shortcut = true
    - shortcut_path_length records original hop count
    - No duplicate shortcuts for same source-target pair
    - EdgeOrigin::AmortizedShortcut in metadata
    - Unit tests verify quality gate rejection
    - Unit tests verify shortcut weight calculation
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/src/shortcut.rs
```

---

## Surface Layer Tasks (Build Last)

These tasks implement high-level orchestration, MCP integration, metrics, and resource management.

```yaml
- id: TASK-DREAM-021
  title: Implement DreamResourceManager for GPU budget enforcement
  type: implementation
  layer: surface
  requirement_refs: [REQ-DREAM-016, REQ-DREAM-024]
  dependencies: [TASK-DREAM-014, TASK-DREAM-018]
  acceptance_criteria:
    - DreamResourceManager struct with max_gpu_usage (0.30), check_interval (100ms)
    - throttle_factor (0.5) when exceeding limit
    - enforce_limits() returns ResourceAction enum
    - ResourceAction: Continue, Warning, Throttle, Pause, Abort
    - Throttle activates when GPU > 30%
    - Warning at 80% of limit
    - check_critical() pauses if external load > 70% or memory > 85%
    - Throttle state tracked and logged
    - Integration test verifies GPU limit enforcement
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-dream/src/resource.rs

- id: TASK-DREAM-022
  title: Implement DreamMetrics and DreamQualityAssessor
  type: implementation
  layer: surface
  requirement_refs: [REQ-DREAM-021, REQ-DREAM-022]
  dependencies: [TASK-DREAM-014, TASK-DREAM-018]
  acceptance_criteria:
    - DreamMetrics struct with nrem, rem, cycle, resources fields
    - NREMMetrics: memories_replayed, edges_*, clusters_consolidated, compression_ratio
    - REMMetrics: synthetic_queries, blind_spots_found, average_semantic_leap
    - CycleMetrics: total_duration_ms, wake_events, information_loss_estimate
    - ResourceMetrics: peak_gpu_usage, throttle_events
    - DreamQualityAssessor with min_compression (5.0), min_blind_spots (3)
    - max_information_loss (0.15)
    - assess() returns DreamQualityReport with overall_score, grade
    - QualityGrade: Excellent (>=0.9), Good (>=0.7), Fair (>=0.5), Poor
    - Compression (30%), exploration (30%), preservation (40%) weighted
    - Recommendations generated for low scores
    - Unit tests verify quality scoring formula
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-dream/src/metrics.rs
    - crates/context-graph-dream/src/quality.rs

- id: TASK-DREAM-023
  title: Implement DreamCycleManager and sleep cycle state machine
  type: implementation
  layer: surface
  requirement_refs: [REQ-DREAM-017]
  dependencies: [TASK-DREAM-021]
  acceptance_criteria:
    - DreamCycleManager struct with cycle_duration (5min), biological_ratio (90min)
    - max_dream_time (30min), min_cycle_gap (10min)
    - plan_cycle() returns DreamCyclePlan enum
    - DreamCyclePlan::Ready with nrem_duration (3min), rem_duration (2min), transition (500ms)
    - DreamCyclePlan::Delayed with wait_until timestamp
    - DreamCyclePlan::Blocked with reason
    - record_completion() updates last_completed and cycle_count
    - Minimum gap enforced between cycles
    - Maximum continuous dream time enforced
    - Unit tests verify timing constraints
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-dream/src/cycle.rs

- id: TASK-DREAM-024
  title: Implement DreamLayer complete orchestration
  type: implementation
  layer: surface
  requirement_refs: [REQ-DREAM-001, REQ-DREAM-020]
  dependencies: [TASK-DREAM-014, TASK-DREAM-018, TASK-DREAM-019, TASK-DREAM-021, TASK-DREAM-022, TASK-DREAM-023]
  acceptance_criteria:
    - DreamLayer struct combining all components
    - Arc<RwLock<>> for thread-safe access to phases, state, metrics
    - execute_cycle() runs NREM -> transition -> REM sequence
    - Neuromodulation levels set per phase (NREM: low DA, REM: high 5-HT)
    - Abort controller integrated with watch channel
    - Resource manager enforces GPU budget throughout
    - Metrics collected and quality assessed
    - DreamCycleResult includes completed, metrics, quality, duration
    - Drop impl cleans up resources
    - Integration test verifies end-to-end cycle
    - Benchmark validates wake latency <100ms P95
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/layer.rs

- id: TASK-DREAM-025
  title: Implement MCP tool integration for dream control
  type: implementation
  layer: surface
  requirement_refs: [REQ-DREAM-018, REQ-DREAM-019, REQ-DREAM-030]
  dependencies: [TASK-DREAM-024]
  acceptance_criteria:
    - TriggerDreamTool struct registered with MCP server
    - TriggerDreamParams: phase (nrem/rem/full_cycle), duration_minutes, synthetic_query_count
    - blocking (false default), abort_on_query (true default)
    - TriggerDreamResult: dream_id, status, started_at, estimated_completion, metrics
    - DreamPhaseSelector enum: NremOnly, RemOnly, FullCycle
    - get_memetic_status extension with DreamStatusExtension
    - ActiveDreamInfo: dream_id, phase, progress, estimated_remaining
    - LastDreamResults: completed_at, nrem_stats, rem_stats, blind_spots_discovered
    - Error codes match specification (-32100 to -32107)
    - Schema matches MCP JSON-RPC 2.0 specification
    - Integration tests verify tool round-trip
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/mcp/tools.rs
    - crates/context-graph-mcp/src/tools/dream.rs
```

---

## Test Tasks

```yaml
- id: TASK-DREAM-TEST-001
  title: Unit tests for dream state machine and triggers
  type: test
  layer: foundation
  requirement_refs: [REQ-DREAM-033]
  dependencies: [TASK-DREAM-002, TASK-DREAM-003, TASK-DREAM-004, TASK-DREAM-005]
  acceptance_criteria:
    - Tests for DreamState transitions (all valid paths)
    - Tests for DreamState.is_dreaming() behavior
    - Tests for DreamTrigger.should_trigger() with various conditions
    - Tests for cooldown enforcement
    - Tests for AbortController abort propagation timing
    - Tests for DreamError display formatting
    - All tests pass with cargo test -p context-graph-dream
  estimated_complexity: medium
  files_affected:
    - crates/context-graph-dream/src/state.rs (test module)
    - crates/context-graph-dream/src/trigger.rs (test module)
    - crates/context-graph-dream/src/abort.rs (test module)

- id: TASK-DREAM-TEST-002
  title: Unit tests for SRC algorithm and Hebbian updates
  type: test
  layer: logic
  requirement_refs: [REQ-DREAM-033]
  dependencies: [TASK-DREAM-010, TASK-DREAM-019, TASK-DREAM-020]
  acceptance_criteria:
    - TC-DREAM-014: SRC sparsity controls activation fraction
    - TC-DREAM-006: Hebbian formula delta_w = eta * pre * post
    - TC-DREAM-015: SRC randomness parameter behavior
    - Tests for weight decay and pruning
    - Tests for amortized shortcut creation quality gate (0.7 confidence)
    - Tests for shortcut path length threshold (3+ hops)
    - All tests pass with cargo test
  estimated_complexity: high
  files_affected:
    - crates/context-graph-dream/src/src/algorithm.rs (test module)
    - crates/context-graph-dream/src/nrem/hebbian.rs (test module)
    - crates/context-graph-dream/src/src/shortcut.rs (test module)

- id: TASK-DREAM-TEST-003
  title: Integration tests for dream cycle performance
  type: test
  layer: surface
  requirement_refs: [REQ-DREAM-034, REQ-DREAM-035]
  dependencies: [TASK-DREAM-024]
  acceptance_criteria:
    - TC-DREAM-001: Full cycle completion with graph modifications
    - TC-DREAM-002: Abort on query during NREM (latency < 100ms)
    - TC-DREAM-003: Abort on query during REM (latency < 100ms)
    - TC-DREAM-004: GPU usage stays < 30%
    - TC-DREAM-005: Compression ratio >= 10:1
    - TC-DREAM-006: Blind spots discovered >= 5
    - TC-DREAM-007: Information loss < 15%
    - TC-DREAM-008: NREM phase completes in ~3 minutes
    - TC-DREAM-009: REM phase completes in ~2 minutes
    - Tests use realistic graph data (100K nodes)
    - All tests in tests/integration/dream_tests.rs pass
  estimated_complexity: high
  files_affected:
    - tests/integration/dream_tests.rs

- id: TASK-DREAM-TEST-004
  title: Integration tests for MCP tool functionality
  type: test
  layer: surface
  requirement_refs: [REQ-DREAM-034]
  dependencies: [TASK-DREAM-025]
  acceptance_criteria:
    - TC-DREAM-018: trigger_dream tool execution
    - TC-DREAM-019: get_memetic_status dream extension
    - TC-DREAM-020: Non-blocking dream trigger returns immediately
    - TC-DREAM-021: Blocking dream trigger waits for completion
    - TC-DREAM-022: Phase selection (NREM only, REM only, full cycle)
    - TC-DREAM-023: Error handling for concurrent dream attempts
    - Tests verify JSON-RPC 2.0 compliance
    - All tests in tests/integration/dream_mcp_tests.rs pass
  estimated_complexity: medium
  files_affected:
    - tests/integration/dream_mcp_tests.rs
```

---

## Dependency Graph

```
TASK-DREAM-001 (module structure)
    |
    +-- TASK-DREAM-002 (state machine) --+-- TASK-DREAM-003 (trigger)
    |       |                             |
    |       +-- TASK-DREAM-004 (abort)    +-- TASK-DREAM-005 (errors)
    |       |
    |       +-- TASK-DREAM-006 (checkpoint)
    |
    +-- TASK-DREAM-007 (replay path types) --+
    |                                         |
    +-- TASK-DREAM-008 (hyperbolic)          |
            |                                 |
            +-- TASK-DREAM-015 (query gen)   |
                    |                         |
    +---------------+---------------+         |
    |               |               |         |
TASK-DREAM-009   TASK-DREAM-016   TASK-DREAM-019 (SRC algorithm)
(replay engine)  (blind spot)            |         |
    |               |                    |         |
    +-- TASK-DREAM-010 (Hebbian)         +-- TASK-DREAM-020 (shortcuts)
            |               |
            +-- TASK-DREAM-011 (consolidation)
            |       |
            +-- TASK-DREAM-012 (redundancy)
            |       |
            +-- TASK-DREAM-013 (schema)
                    |
            TASK-DREAM-014 (NREM phase)
                    |
    +---------------+---------------+
    |               |               |
    +-- TASK-DREAM-017 (edge creator)
            |
    TASK-DREAM-018 (REM phase)
            |
    +-------+-------+-------+-------+
    |       |       |       |       |
TASK-21  TASK-22  TASK-23  TASK-24  TASK-25
(resource)(metrics)(cycle) (layer) (MCP tools)
```

---

## Traceability Matrix

| Task ID | Requirements Covered |
|---------|---------------------|
| TASK-DREAM-001 | REQ-DREAM-001 |
| TASK-DREAM-002 | REQ-DREAM-001, REQ-DREAM-002 |
| TASK-DREAM-003 | REQ-DREAM-002 |
| TASK-DREAM-004 | REQ-DREAM-003, REQ-DREAM-023 |
| TASK-DREAM-005 | REQ-DREAM-032 |
| TASK-DREAM-006 | REQ-DREAM-025, REQ-DREAM-028 |
| TASK-DREAM-007 | REQ-DREAM-036, REQ-DREAM-037 |
| TASK-DREAM-008 | REQ-DREAM-011 |
| TASK-DREAM-009 | REQ-DREAM-004, REQ-DREAM-005 |
| TASK-DREAM-010 | REQ-DREAM-006 |
| TASK-DREAM-011 | REQ-DREAM-007 |
| TASK-DREAM-012 | REQ-DREAM-008 |
| TASK-DREAM-013 | REQ-DREAM-009 |
| TASK-DREAM-014 | REQ-DREAM-004 |
| TASK-DREAM-015 | REQ-DREAM-011 |
| TASK-DREAM-016 | REQ-DREAM-012 |
| TASK-DREAM-017 | REQ-DREAM-013 |
| TASK-DREAM-018 | REQ-DREAM-010 |
| TASK-DREAM-019 | REQ-DREAM-014 |
| TASK-DREAM-020 | REQ-DREAM-036, REQ-DREAM-037 |
| TASK-DREAM-021 | REQ-DREAM-016, REQ-DREAM-024 |
| TASK-DREAM-022 | REQ-DREAM-021, REQ-DREAM-022 |
| TASK-DREAM-023 | REQ-DREAM-017 |
| TASK-DREAM-024 | REQ-DREAM-001, REQ-DREAM-020 |
| TASK-DREAM-025 | REQ-DREAM-018, REQ-DREAM-019, REQ-DREAM-030 |
| TASK-DREAM-TEST-001 | REQ-DREAM-033 |
| TASK-DREAM-TEST-002 | REQ-DREAM-033 |
| TASK-DREAM-TEST-003 | REQ-DREAM-034, REQ-DREAM-035 |
| TASK-DREAM-TEST-004 | REQ-DREAM-034 |

---

## Performance Verification Criteria

| Metric | Target | Verification Task |
|--------|--------|------------------|
| NREM duration | 3 min | TASK-DREAM-TEST-003 (TC-DREAM-008) |
| REM duration | 2 min | TASK-DREAM-TEST-003 (TC-DREAM-009) |
| Wake latency | < 100ms P95 | TASK-DREAM-TEST-003 (TC-DREAM-002, TC-DREAM-003) |
| GPU usage | < 30% | TASK-DREAM-TEST-003 (TC-DREAM-004) |
| Compression ratio | 10:1 | TASK-DREAM-TEST-003 (TC-DREAM-005) |
| Blind spots | > 5/cycle | TASK-DREAM-TEST-003 (TC-DREAM-006) |
| Information loss | < 15% | TASK-DREAM-TEST-003 (TC-DREAM-007) |
| Shortcut confidence gate | >= 0.7 | TASK-DREAM-TEST-002 |
| Shortcut min hops | >= 3 | TASK-DREAM-TEST-002 |
| Unit test coverage | > 90% | All TASK-DREAM-TEST-* |
| Integration test coverage | > 80% | TASK-DREAM-TEST-003, TASK-DREAM-TEST-004 |

---

*Document generated: 2025-12-31*
*Task Specification Version: 1.0*
*Module: Dream Layer (Phase 8)*
*Total Tasks: 29 (25 implementation + 4 test)*
