# Module 12.5: Steering Subsystem - Atomic Tasks

```yaml
metadata:
  module_id: "module-12.5"
  module_name: "Steering Subsystem"
  version: "1.0.0"
  phase: 5
  total_tasks: 22
  approach: "inside-out-bottom-up"
  created: "2025-12-31"
  dependencies:
    - module-04-knowledge-graph
    - module-09-dream-layer
    - module-10-neuromodulation
  estimated_duration: "2 weeks"
  spec_refs:
    - SPEC-STEER-012.5 (Functional)
    - TECH-STEER-012.5 (Technical)
  performance_constraint: "<5ms total evaluation"
  marblestone_integration: true
```

---

## Task Overview

This module implements Adam Marblestone's architectural separation between learning and steering mechanisms. The Steering Subsystem generates reward signals (analogous to dopamine) to guide learning without directly modifying network weights. It comprises three core components: the Gardener for long-term memory curation, the Curator for quality assessment, and the Thought Assessor for real-time evaluation.

### Performance Targets

| Operation | Budget | Notes |
|-----------|--------|-------|
| Gardener evaluation | <2ms | Long-term value assessment |
| Curator evaluation | <2ms | Quality scoring |
| Thought Assessor | <1ms | Real-time coherence |
| **Total evaluation** | **<5ms** | End-to-end pipeline |
| Dopamine feedback | <1ms | Module 10 integration |
| MCP handler | <10ms | Including serialization |

### Weight Distribution

| Component | Weight | Purpose |
|-----------|--------|---------|
| Gardener | 35% | Long-term value |
| Curator | 35% | Quality assessment |
| Assessor | 30% | Immediate relevance |

### Task Organization

1. **Foundation Layer** (Tasks 1-7): Core types, enums, configuration, error handling
2. **Logic Layer** (Tasks 8-15): Gardener, Curator, ThoughtAssessor components
3. **Surface Layer** (Tasks 16-22): Unified evaluation, integrations, MCP tool

---

## Foundation Layer: Core Types & Configuration

```yaml
tasks:
  # ============================================================
  # FOUNDATION: Core Types
  # ============================================================

  - id: "TASK-12.5-001"
    title: "Define SuggestionType Enum and SteeringSuggestion Struct"
    description: |
      Implement SuggestionType enum for steering action recommendations.
      Variants: Consolidate, Prune, StrengthenConnection, EnrichMetadata,
      RequestClarification, DreamReview, PromoteImportance, DemoteImportance.

      Implement SteeringSuggestion struct:
      - suggestion_type: SuggestionType
      - priority: f32 [0.0, 1.0]
      - description: String
      - target_node: Option<Uuid>
      - parameters: HashMap<String, String>

      Derive Clone, Debug, Serialize, PartialEq for enum.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1.5
    file_path: "crates/context-graph-steering/src/suggestion.rs"
    dependencies: []
    acceptance_criteria:
      - "SuggestionType enum with 8 variants compiles"
      - "SteeringSuggestion struct with 5 fields"
      - "Clone, Debug, Serialize derived"
      - "PartialEq implemented for enum"
      - "Default for parameters is empty HashMap"
    test_file: "crates/context-graph-steering/tests/suggestion_tests.rs"
    spec_refs:
      - "REQ-STEER-002"
      - "TECH-STEER-012.5 Section 2"

  - id: "TASK-12.5-002"
    title: "Define SteeringComponents Struct"
    description: |
      Implement SteeringComponents struct for component-wise reward breakdown.
      Fields:
      - gardener_score: f32 [-1.0, 1.0] (Long-term value)
      - curator_score: f32 [-1.0, 1.0] (Quality assessment)
      - assessor_score: f32 [-1.0, 1.0] (Immediate relevance)

      Implement weighted_sum(&self, weights: &SteeringWeights) -> f32.
      Implement dominant_component(&self) -> &'static str returning "gardener", "curator", or "assessor".
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1
    file_path: "crates/context-graph-steering/src/reward.rs"
    dependencies: []
    acceptance_criteria:
      - "SteeringComponents struct with 3 f32 fields"
      - "All scores bounded to [-1.0, 1.0]"
      - "weighted_sum() computes correct weighted combination"
      - "dominant_component() returns name of highest absolute score"
      - "Clone, Debug, Serialize derived"
    test_file: "crates/context-graph-steering/tests/reward_tests.rs"
    spec_refs:
      - "REQ-STEER-002"
      - "TECH-STEER-012.5 Section 2"

  - id: "TASK-12.5-003"
    title: "Define SteeringMetadata and SteeringReward Structs"
    description: |
      Implement SteeringMetadata struct for debugging and analysis:
      - evaluation_id: Uuid
      - timestamp: DateTime<Utc>
      - node_id: Uuid
      - domain: Option<String>
      - novelty: f32 [0.0, 1.0]
      - coherence: f32 [0.0, 1.0]

      Implement SteeringReward struct (main output):
      - reward: f32 [-1.0, 1.0]
      - components: SteeringComponents
      - explanation: String
      - suggestions: Vec<SteeringSuggestion>
      - confidence: f32 [0.0, 1.0]
      - latency: Duration
      - metadata: SteeringMetadata

      Implement neutral() -> Self returning zero reward with no suggestions.
      Implement is_positive() -> bool (reward > 0.3).
      Implement is_negative() -> bool (reward < -0.3).
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-steering/src/reward.rs"
    dependencies:
      - "TASK-12.5-001"
      - "TASK-12.5-002"
    acceptance_criteria:
      - "SteeringMetadata struct with 6 fields"
      - "SteeringReward struct with 7 fields"
      - "neutral() returns reward=0.0, empty suggestions"
      - "is_positive() threshold at 0.3"
      - "is_negative() threshold at -0.3"
      - "Clone, Debug, Serialize derived"
    test_file: "crates/context-graph-steering/tests/reward_tests.rs"
    spec_refs:
      - "REQ-STEER-002"
      - "TECH-STEER-012.5 Section 2"

  - id: "TASK-12.5-004"
    title: "Define SteeringConfig and Component Configs"
    description: |
      Implement SteeringConfig struct:
      - gardener_weight: f32 (Default: 0.35)
      - curator_weight: f32 (Default: 0.35)
      - assessor_weight: f32 (Default: 0.30)
      - positive_threshold: f32 (Default: 0.3)
      - negative_threshold: f32 (Default: -0.3)
      - dopamine_integration_enabled: bool (Default: true)
      - evaluation_timeout: Duration (Default: 5ms)
      - suggestions_enabled: bool (Default: true)
      - max_suggestions: usize (Default: 3)

      Implement validate() -> Result<(), ConfigError> checking weights sum to 1.0.
      Implement from_toml(path) for configuration loading.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-steering/src/config.rs"
    dependencies: []
    acceptance_criteria:
      - "SteeringConfig struct with 9 fields"
      - "Default implementation with all specified values"
      - "Weights sum to 1.0 (0.35 + 0.35 + 0.30)"
      - "validate() checks weight sum tolerance (0.001)"
      - "from_toml() parses TOML configuration"
      - "Clone, Debug, Deserialize derived"
    test_file: "crates/context-graph-steering/tests/config_tests.rs"
    spec_refs:
      - "REQ-STEER-001"
      - "TECH-STEER-012.5 Section 10"

  - id: "TASK-12.5-005"
    title: "Define SteeringError Enum with MCP Codes"
    description: |
      Implement SteeringError enum using thiserror:
      - NodeNotFound(Uuid) -> -32100
      - SessionNotFound(Uuid) -> -32101
      - EvaluationTimeout(u64) -> -32102
      - ComponentFailure(String) -> -32103
      - DopamineFeedbackFailed(String) -> -32104
      - ConfigurationError(String) -> -32105
      - InvalidReward(f32) -> -32106
      - Internal(String) -> -32603

      Implement error_code() -> i32 returning MCP-compliant codes.
      All variants have descriptive #[error()] messages.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1
    file_path: "crates/context-graph-steering/src/error.rs"
    dependencies: []
    acceptance_criteria:
      - "SteeringError enum with 8 variants"
      - "All variants have #[error()] messages"
      - "error_code() returns correct MCP codes"
      - "Error is Send + Sync"
      - "Debug derived"
    test_file: "crates/context-graph-steering/tests/error_tests.rs"
    spec_refs:
      - "TECH-STEER-012.5 Section 4.1"

  - id: "TASK-12.5-006"
    title: "Define SteeringSessionState and RingBuffer"
    description: |
      Implement RingBuffer<T> for efficient history tracking:
      - buffer: Vec<T>
      - head: usize
      - capacity: usize
      - new(capacity) -> Self
      - push(&mut self, item: T)
      - iter(&self) -> impl Iterator<Item = &T>
      - len(&self) -> usize

      Implement SteeringRewardEntry for history:
      - timestamp: Instant
      - node_id: Uuid
      - reward: f32
      - components: SteeringComponents

      Implement SteeringSessionState:
      - session_id: Uuid
      - avg_reward: f32
      - evaluation_count: u64
      - reward_history: Vec<SteeringRewardEntry>
      - domain_stats: HashMap<String, DomainSteeringStats>
      - last_evaluation: Option<Instant>
    layer: "foundation"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-steering/src/session.rs"
    dependencies:
      - "TASK-12.5-002"
    acceptance_criteria:
      - "RingBuffer<T> with FIFO eviction"
      - "push() handles buffer wrap-around"
      - "SteeringRewardEntry struct with 4 fields"
      - "SteeringSessionState struct with 6 fields"
      - "Memory footprint under 100KB per session"
      - "Clone derived for session state"
    test_file: "crates/context-graph-steering/tests/session_tests.rs"
    spec_refs:
      - "REQ-STEER-001"
      - "TECH-STEER-012.5 Section 2"

  - id: "TASK-12.5-007"
    title: "Define SteeringContext for Unified Evaluation"
    description: |
      Implement SteeringContext struct containing all context needed for evaluation:
      - access_stats: HashMap<Uuid, AccessStats>
      - connection_count: HashMap<Uuid, usize>
      - avg_connection_count: f32
      - connected_domains: HashMap<Uuid, Vec<String>>
      - coherence_scores: HashMap<Uuid, f32>
      - reference_count: HashMap<Uuid, usize>
      - domain_consistency: HashMap<Uuid, f32>
      - semantic_similarity: HashMap<Uuid, f32>
      - current_domain: Option<String>
      - reference_validity: HashMap<Uuid, f32>
      - query_similarity: HashMap<Uuid, f32>
      - session_topic: Option<String>
      - recent_context_age_ms: u64

      Implement Default returning empty context.
    layer: "foundation"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-steering/src/context.rs"
    dependencies: []
    acceptance_criteria:
      - "SteeringContext struct with 13 fields"
      - "Default returns empty HashMaps"
      - "Clone derived for context sharing"
      - "Context can be constructed from graph state"
    test_file: "crates/context-graph-steering/tests/context_tests.rs"
    spec_refs:
      - "REQ-STEER-006"
      - "TECH-STEER-012.5 Section 6"

  # ============================================================
  # LOGIC LAYER: Gardener Component
  # ============================================================

  - id: "TASK-12.5-008"
    title: "Implement DecayConfig and UsagePatternAnalyzer"
    description: |
      Implement DecayConfig struct for time-based decay:
      - importance_half_life: f32 (Default: 168.0 hours = 1 week)
      - importance_floor: f32 (Default: 0.1)
      - access_boost: f32 (Default: 0.2)
      - connection_boost: f32 (Default: 0.1)

      Implement AccessStats struct:
      - total_accesses: u64
      - recent_accesses: u32 (last 24 hours)
      - avg_interval_hours: f32
      - last_access: Option<Instant>

      Implement UsagePatternAnalyzer struct:
      - access_history: HashMap<Uuid, Vec<Instant>>
      - frequency_stats: HashMap<Uuid, AccessStats>
      - record_access(node_id: Uuid)
      - get_stats(node_id: Uuid) -> Option<AccessStats>
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-steering/src/gardener/usage.rs"
    dependencies: []
    acceptance_criteria:
      - "DecayConfig struct with 4 fields"
      - "Default matches specification values"
      - "AccessStats struct with 4 fields"
      - "UsagePatternAnalyzer tracks access patterns"
      - "History capped to prevent memory growth"
      - "Clone derived"
    test_file: "crates/context-graph-steering/tests/gardener_tests.rs"
    spec_refs:
      - "REQ-STEER-003"
      - "TECH-STEER-012.5 Section 3"

  - id: "TASK-12.5-009"
    title: "Implement ConsolidationRule and PruningRule Types"
    description: |
      Implement ConsolidationCondition enum:
      - HighAccessFrequency { min_accesses: u32, period_hours: f32 }
      - SemanticCluster { min_similarity: f32, min_cluster_size: usize }
      - TemporalProximity { max_gap_hours: f32, min_sequence_length: usize }
      - HighImportance { threshold: f32 }
      - CrossDomainHub { min_domains: usize, min_connections: usize }

      Implement ConsolidationRule struct:
      - id: String
      - condition: ConsolidationCondition
      - priority: u32
      - description: String

      Implement PruningCondition enum:
      - Stale { min_hours: f32 }
      - LowValue { max_importance: f32, max_connections: usize }
      - Superseded { newer_node_id: Uuid }
      - Redundant { similar_node_id: Uuid, similarity: f32 }
      - Incoherent { coherence_score: f32 }

      Implement PruningSeverity enum: Low, Medium, High, Critical.
      Implement PruningRule struct with id, condition, severity, description.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-steering/src/gardener/rules.rs"
    dependencies: []
    acceptance_criteria:
      - "ConsolidationCondition enum with 5 variants"
      - "ConsolidationRule struct with 4 fields"
      - "PruningCondition enum with 5 variants"
      - "PruningSeverity enum with 4 variants"
      - "PruningRule struct with 4 fields"
      - "Clone, Debug derived"
    test_file: "crates/context-graph-steering/tests/gardener_tests.rs"
    spec_refs:
      - "REQ-STEER-003"
      - "TECH-STEER-012.5 Section 3"

  - id: "TASK-12.5-010"
    title: "Implement Gardener Component with Long-Term Value Assessment"
    description: |
      Implement GardenerConfig struct:
      - min_age_for_pruning: f32 (Default: 24.0 hours)
      - access_frequency_weight: f32 (Default: 0.3)
      - recency_weight: f32 (Default: 0.2)
      - connection_weight: f32 (Default: 0.25)
      - importance_weight: f32 (Default: 0.25)
      - auto_prune_enabled: bool (Default: true)
      - max_prune_suggestions: usize (Default: 5)

      Implement Gardener struct:
      - prune_threshold: f32 (Default: -0.5)
      - consolidation_threshold: f32 (Default: 0.7)
      - consolidation_rules: Vec<ConsolidationRule>
      - pruning_rules: Vec<PruningRule>
      - decay_config: DecayConfig
      - usage_analyzer: UsagePatternAnalyzer
      - config: GardenerConfig

      Methods:
      - evaluate_long_term_value(&self, node, context) -> f32 [Constraint: <2ms]
      - calculate_access_score(&self, node, context) -> f32
      - calculate_recency_score(&self, node) -> f32
      - calculate_connection_score(&self, node, context) -> f32
      - calculate_importance_score(&self, node) -> f32
      - calculate_age_factor(&self, node) -> f32
      - check_consolidation(&self, node, context) -> Option<ConsolidationRule>
      - check_pruning(&self, node, context) -> Option<(PruningRule, PruningSeverity)>
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-steering/src/gardener/mod.rs"
    dependencies:
      - "TASK-12.5-008"
      - "TASK-12.5-009"
    acceptance_criteria:
      - "Gardener struct with 7 fields"
      - "evaluate_long_term_value() completes in <2ms"
      - "Score combines access, recency, connection, importance"
      - "Age decay applied with configurable half-life"
      - "Output clamped to [-1.0, 1.0]"
      - "Pruning at threshold -0.5"
      - "Consolidation at threshold 0.7"
      - "Min age for pruning enforced (24 hours)"
    test_file: "crates/context-graph-steering/tests/gardener_tests.rs"
    spec_refs:
      - "REQ-STEER-003"
      - "TECH-STEER-012.5 Section 3"

  # ============================================================
  # LOGIC LAYER: Curator Component
  # ============================================================

  - id: "TASK-12.5-011"
    title: "Implement QualityModel and Assessment Types"
    description: |
      Implement CompletenessCheck enum:
      - HasMetadata(Vec<String>)
      - HasEmbedding { min_dims: usize }
      - HasConnections { min_count: usize }
      - MinContentLength { min_chars: usize }
      - HasDomain

      Implement CompletenessCriterion struct: name, weight, check.

      Implement AccuracySignal enum:
      - SourceCredibility { min_score: f32 }
      - CrossReferenced { min_references: usize }
      - TemporallyConsistent
      - UserVerified
      - DomainConsistent

      Implement AccuracyIndicator struct: name, weight, indicator.
      Implement ClarityMetrics struct: target_readability, max_complexity, min_structure.
      Implement QualityPattern struct: pattern_id, description, positive_weight, detection_threshold.

      Implement QualityModel struct:
      - completeness_criteria: Vec<CompletenessCriterion>
      - accuracy_indicators: Vec<AccuracyIndicator>
      - clarity_metrics: ClarityMetrics
      - learned_patterns: Vec<QualityPattern>

      Implement QualityThresholds struct:
      - review_threshold: f32 (Default: 0.4)
      - enrichment_threshold: f32 (Default: 0.5)
      - high_quality_threshold: f32 (Default: 0.8)
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-steering/src/curator/model.rs"
    dependencies: []
    acceptance_criteria:
      - "CompletenessCheck enum with 5 variants"
      - "AccuracySignal enum with 5 variants"
      - "QualityModel struct with 4 fields"
      - "QualityThresholds with default values"
      - "Clone, Debug derived"
    test_file: "crates/context-graph-steering/tests/curator_tests.rs"
    spec_refs:
      - "REQ-STEER-004"
      - "TECH-STEER-012.5 Section 4"

  - id: "TASK-12.5-012"
    title: "Implement Curator Component with Quality Assessment"
    description: |
      Implement CuratorConfig struct:
      - min_quality: f32 (Default: 0.3)
      - completeness_weight: f32 (Default: 0.25)
      - accuracy_weight: f32 (Default: 0.30)
      - clarity_weight: f32 (Default: 0.25)
      - relevance_weight: f32 (Default: 0.20)
      - domain_weighting_enabled: bool (Default: true)

      Implement CuratorContext struct:
      - connection_count: HashMap<Uuid, usize>
      - reference_count: HashMap<Uuid, usize>
      - domain_consistency: HashMap<Uuid, f32>
      - semantic_similarity: HashMap<Uuid, f32>
      - current_domain: Option<String>

      Implement Curator struct:
      - quality_model: QualityModel
      - domain_weights: HashMap<Domain, f32>
      - thresholds: QualityThresholds
      - config: CuratorConfig

      Methods:
      - evaluate_quality(&self, node, context) -> f32 [Constraint: <2ms]
      - assess_completeness(&self, node, context) -> f32
      - assess_accuracy(&self, node, context) -> f32
      - assess_clarity(&self, node) -> f32
      - assess_relevance(&self, node, context) -> f32
      - apply_domain_weighting(&self, quality, node, context) -> f32
      - generate_suggestions(&self, node, quality_score, context) -> Vec<SteeringSuggestion>
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-steering/src/curator/mod.rs"
    dependencies:
      - "TASK-12.5-001"
      - "TASK-12.5-011"
    acceptance_criteria:
      - "Curator struct with 4 fields"
      - "evaluate_quality() completes in <2ms"
      - "Weights: completeness 25%, accuracy 30%, clarity 25%, relevance 20%"
      - "Output mapped to [-1.0, 1.0] (0.5 is neutral)"
      - "Domain-specific weighting functional"
      - "Suggestions generated for low quality"
      - "EnrichMetadata suggested for missing embedding"
    test_file: "crates/context-graph-steering/tests/curator_tests.rs"
    spec_refs:
      - "REQ-STEER-004"
      - "TECH-STEER-012.5 Section 4"

  # ============================================================
  # LOGIC LAYER: Thought Assessor Component
  # ============================================================

  - id: "TASK-12.5-013"
    title: "Implement ThoughtSignature and CoherenceModel"
    description: |
      Implement ThoughtSignature struct for novelty detection:
      - id: Uuid
      - embedding_hash: u64
      - domain: Option<String>
      - timestamp: Instant
      - concepts: Vec<String>

      Implement CoherencePatternType enum:
      - SemanticConsistency
      - TemporalFlow
      - CausalValidity
      - ReferenceIntegrity

      Implement CoherencePattern struct: pattern_type, weight.
      Implement CoherenceRule struct: rule_id, pattern, weight.

      Implement CoherenceModel struct:
      - domain_rules: HashMap<String, Vec<CoherenceRule>>
      - global_patterns: Vec<CoherencePattern>

      Implement AssessorContext struct:
      - domain_similarity: HashMap<Uuid, f32>
      - reference_validity: HashMap<Uuid, f32>
      - query_similarity: HashMap<Uuid, f32>
      - session_topic: Option<String>
      - recent_context_age_ms: u64
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-steering/src/assessor/types.rs"
    dependencies:
      - "TASK-12.5-006"
    acceptance_criteria:
      - "ThoughtSignature struct with 5 fields"
      - "CoherencePatternType enum with 4 variants"
      - "CoherenceModel struct with 2 fields"
      - "AssessorContext struct with 5 fields"
      - "Clone derived"
    test_file: "crates/context-graph-steering/tests/assessor_tests.rs"
    spec_refs:
      - "REQ-STEER-005"
      - "TECH-STEER-012.5 Section 5"

  - id: "TASK-12.5-014"
    title: "Implement ThoughtAssessor Component with Real-Time Evaluation"
    description: |
      Implement ThoughtAssessorConfig struct:
      - coherence_weight: f32 (Default: 0.4)
      - novelty_weight: f32 (Default: 0.3)
      - context_weight: f32 (Default: 0.3)
      - min_coherence: f32 (Default: 0.3)
      - novelty_window: usize (Default: 100)
      - high_novelty_threshold: f32 (Default: 0.7)
      - rapid_rejection_enabled: bool (Default: true)

      Implement ThoughtAssessor struct:
      - coherence_threshold: f32 (Default: 0.4)
      - novelty_weight: f32 (Default: 0.3)
      - config: ThoughtAssessorConfig
      - recent_thoughts: RingBuffer<ThoughtSignature>
      - coherence_model: CoherenceModel

      Methods:
      - new(window_size: usize) -> Self
      - evaluate_immediate(&self, node, context) -> f32 [Constraint: <1ms]
      - quick_coherence_check(&self, node) -> Option<f32>
      - assess_coherence(&self, node, context) -> f32
      - assess_structural_coherence(&self, node) -> f32
      - assess_novelty(&self, node) -> f32
      - assess_context_fit(&self, node, context) -> f32
      - compute_signature(&self, node) -> ThoughtSignature
      - compare_signatures(&self, a, b) -> f32
      - record_thought(&mut self, node)
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-steering/src/assessor/mod.rs"
    dependencies:
      - "TASK-12.5-006"
      - "TASK-12.5-013"
    acceptance_criteria:
      - "ThoughtAssessor struct with 5 fields"
      - "evaluate_immediate() completes in <1ms"
      - "Rapid rejection returns -0.5 for empty content"
      - "Rapid rejection returns -0.3 for single-word content"
      - "Novelty score based on similarity to recent thoughts"
      - "First thought gets 0.7 novelty (no history)"
      - "Output mapped to [-1.0, 1.0]"
      - "record_thought() adds to ring buffer"
    test_file: "crates/context-graph-steering/tests/assessor_tests.rs"
    spec_refs:
      - "REQ-STEER-005"
      - "TECH-STEER-012.5 Section 5"

  - id: "TASK-12.5-015"
    title: "Implement GardenerContext and CuratorContext Preparation"
    description: |
      Implement GardenerContext struct:
      - access_stats: HashMap<Uuid, AccessStats>
      - connection_count: HashMap<Uuid, usize>
      - avg_connection_count: f32
      - connected_domains: HashMap<Uuid, Vec<String>>
      - coherence_scores: HashMap<Uuid, f32>

      Implement context preparation methods on SteeringSubsystem:
      - prepare_gardener_context(&self, node, context) -> GardenerContext
      - prepare_curator_context(&self, node, context) -> CuratorContext
      - prepare_assessor_context(&self, node, context) -> AssessorContext

      Each method extracts relevant fields from SteeringContext.
    layer: "logic"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-steering/src/context.rs"
    dependencies:
      - "TASK-12.5-007"
      - "TASK-12.5-008"
    acceptance_criteria:
      - "GardenerContext struct with 5 fields"
      - "All prepare_* methods extract correct fields"
      - "Clone derived for all contexts"
      - "Default implementations available"
    test_file: "crates/context-graph-steering/tests/context_tests.rs"
    spec_refs:
      - "REQ-STEER-006"
      - "TECH-STEER-012.5 Section 6"

  # ============================================================
  # SURFACE LAYER: Unified Evaluation & Integrations
  # ============================================================

  - id: "TASK-12.5-016"
    title: "Implement SteeringDopamineFeedback for Module 10 Integration"
    description: |
      Implement DopamineFeedbackConfig struct:
      - positive_scale: f32 (Default: 0.3)
      - negative_scale: f32 (Default: 0.2)
      - min_magnitude: f32 (Default: 0.1)
      - max_delta: f32 (Default: 0.2)
      - decay_rate: f32 (Default: 0.1)
      - surprise_amplification: bool (Default: true)

      Implement DopamineFeedbackEvent struct:
      - timestamp: Instant
      - steering_reward: f32
      - dopamine_delta: f32
      - components: SteeringComponents
      - surprise_factor: f32

      Implement DopamineFeedbackStats struct:
      - total_events: u64
      - avg_reward: f32
      - avg_delta: f32
      - positive_events: u64
      - negative_events: u64

      Implement SteeringDopamineFeedback struct:
      - neuromod_controller: Option<Arc<RwLock<NeuromodulationController>>>
      - config: DopamineFeedbackConfig
      - history: RingBuffer<DopamineFeedbackEvent>

      Methods:
      - new(config) -> Self
      - connect(&mut self, controller)
      - send_reward(&mut self, reward, components) async [Constraint: <1ms]
      - calculate_surprise(&self, current_reward) -> f32
      - get_stats(&self) -> DopamineFeedbackStats
    layer: "surface"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-steering/src/integration/dopamine.rs"
    dependencies:
      - "TASK-12.5-002"
      - "TASK-12.5-006"
    acceptance_criteria:
      - "SteeringDopamineFeedback struct with 3 fields"
      - "send_reward() completes in <1ms"
      - "Positive reward: delta = reward * 0.3"
      - "Negative reward: delta = reward * 0.2"
      - "Delta clamped to [-0.2, 0.2]"
      - "Surprise amplification multiplies delta"
      - "Dopamine clamped to [0.0, 1.0]"
      - "History tracked in ring buffer"
    test_file: "crates/context-graph-steering/tests/dopamine_tests.rs"
    spec_refs:
      - "REQ-STEER-007"
      - "TECH-STEER-012.5 Section 7"

  - id: "TASK-12.5-017"
    title: "Implement Edge Steering Reward EMA Update for Module 4 Integration"
    description: |
      Implement edge steering_reward update method on SteeringSubsystem:
      - update_edge_steering_reward(&self, node_id: Uuid, reward: f32) async

      Algorithm:
      1. Get all edges for node from KnowledgeGraph
      2. For each edge, apply EMA update:
         edge.steering_reward = edge.steering_reward * (1 - alpha) + reward * alpha
         where alpha = 0.1 (configurable)
      3. Clamp to [-1.0, 1.0]

      Ensure thread-safe access to graph via Arc<RwLock>.
    layer: "surface"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-steering/src/integration/graph.rs"
    dependencies:
      - "TASK-12.5-003"
    acceptance_criteria:
      - "EMA update with alpha=0.1"
      - "steering_reward field updated on all node edges"
      - "Result clamped to [-1.0, 1.0]"
      - "Thread-safe graph access"
      - "Latency <1ms for typical edge count (<100)"
    test_file: "crates/context-graph-steering/tests/graph_tests.rs"
    spec_refs:
      - "REQ-STEER-006"
      - "TECH-STEER-012.5 Section 6"

  - id: "TASK-12.5-018"
    title: "Implement DreamQualityAssessor for Module 9 Integration"
    description: |
      Implement DreamShortcut struct (from Module 9):
      - source_id: Uuid
      - target_id: Uuid
      - value_gain: f32
      - expected_usage: u32

      Implement DreamQualityAssessor struct.

      Methods:
      - assess_shortcut_quality(&self, shortcut, steering) -> f32

      Algorithm:
      1. value_score = if shortcut.value_gain > 0.5 { 0.8 } else { shortcut.value_gain }
      2. usage_score = (shortcut.expected_usage / 10.0).min(1.0)
      3. quality = value_score * 0.6 + usage_score * 0.4
      4. Return quality.clamp(0.0, 1.0)

      Implement on SteeringSubsystem:
      - evaluate_dream_shortcut(&self, shortcut) -> f32
    layer: "surface"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-steering/src/integration/dream.rs"
    dependencies:
      - "TASK-12.5-010"
      - "TASK-12.5-012"
    acceptance_criteria:
      - "DreamShortcut struct with 4 fields"
      - "assess_shortcut_quality() returns [0.0, 1.0]"
      - "Weight: 60% value gain, 40% expected usage"
      - "Value score caps at 0.8 for high value_gain"
      - "evaluate_dream_shortcut() callable from SteeringSubsystem"
    test_file: "crates/context-graph-steering/tests/dream_tests.rs"
    spec_refs:
      - "REQ-STEER-008"
      - "TECH-STEER-012.5 Section 8"

  - id: "TASK-12.5-019"
    title: "Implement SteeringSubsystem Unified Evaluation"
    description: |
      Implement SteeringMetrics struct:
      - total_evaluations: u64
      - avg_latency_us: f64
      - positive_rewards: u64
      - negative_rewards: u64
      - record_evaluation(&mut self, latency, reward)

      Implement SteeringSubsystem struct:
      - gardener: Gardener
      - curator: Curator
      - thought_assessor: ThoughtAssessor
      - dopamine_feedback: SteeringDopamineFeedback
      - config: SteeringConfig
      - session_state: Arc<RwLock<SteeringSessionState>>
      - graph: Arc<RwLock<KnowledgeGraph>>
      - neuromod: Arc<RwLock<NeuromodulationController>>
      - metrics: SteeringMetrics

      Methods:
      - new(config, graph, neuromod) -> Self
      - evaluate(&mut self, node, context) async -> SteeringReward [Constraint: <5ms]
      - generate_explanation(&self, reward, g, c, a) -> String
      - calculate_confidence(&self, g, c, a) -> f32
      - generate_unified_suggestions(&self, node, g, c, a, g_ctx, c_ctx) -> Vec<SteeringSuggestion>
    layer: "surface"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-steering/src/subsystem.rs"
    dependencies:
      - "TASK-12.5-003"
      - "TASK-12.5-004"
      - "TASK-12.5-006"
      - "TASK-12.5-007"
      - "TASK-12.5-010"
      - "TASK-12.5-012"
      - "TASK-12.5-014"
      - "TASK-12.5-015"
      - "TASK-12.5-016"
      - "TASK-12.5-017"
    acceptance_criteria:
      - "SteeringSubsystem struct with 9 fields"
      - "evaluate() completes in <5ms total"
      - "Weighted combination: G*0.35 + C*0.35 + A*0.30"
      - "Reward clamped to [-1.0, 1.0]"
      - "Explanation generated with sentiment and dominant factor"
      - "Confidence based on component sign agreement"
      - "Suggestions sorted by priority, limited to max_suggestions"
      - "record_thought() called after evaluation"
      - "Dopamine feedback sent if enabled"
      - "Edge steering_reward updated via EMA"
      - "Session state updated atomically"
      - "Metrics recorded"
    test_file: "crates/context-graph-steering/tests/subsystem_tests.rs"
    spec_refs:
      - "REQ-STEER-001"
      - "REQ-STEER-006"
      - "TECH-STEER-012.5 Section 6"

  - id: "TASK-12.5-020"
    title: "Implement MCP get_steering_reward Tool Handler"
    description: |
      Implement GetSteeringRewardRequest struct:
      - node_id: Uuid
      - session_id: Option<Uuid>
      - include_components: bool (Default: true)
      - include_suggestions: bool (Default: true)
      - force: bool (Default: false)

      Implement SteeringRewardOutput struct:
      - reward: f32
      - components: Option<SteeringComponentsOutput>
      - explanation: String
      - suggestions: Vec<SteeringSuggestionOutput>
      - confidence: f32

      Implement DopamineFeedbackOutput struct:
      - applied: bool
      - delta: f32
      - new_dopamine_level: f32

      Implement CognitivePulse struct:
      - entropy: f32
      - coherence: f32
      - suggested_action: String ("continue", "review", "monitor")

      Implement GetSteeringRewardResponse struct:
      - reward: SteeringRewardOutput
      - dopamine_feedback: DopamineFeedbackOutput
      - pulse: CognitivePulse
      - latency_ms: f32

      Implement SteeringToolHandler struct:
      - steering: Arc<RwLock<SteeringSubsystem>>
      - graph: Arc<RwLock<KnowledgeGraph>>

      Methods:
      - handle(&self, request) -> Result<GetSteeringRewardResponse, SteeringError>
        [Constraint: <10ms]
      - prepare_context(&self, node, graph) async -> SteeringContext
    layer: "surface"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-steering/src/mcp/tools.rs"
    dependencies:
      - "TASK-12.5-003"
      - "TASK-12.5-005"
      - "TASK-12.5-019"
    acceptance_criteria:
      - "MCP handler completes in <10ms"
      - "Node lookup with proper error handling"
      - "Components optional via include_components"
      - "Suggestions optional via include_suggestions"
      - "Dopamine feedback status included"
      - "CognitivePulse in response"
      - "Correct MCP error codes returned on failure"
      - "Serialize/Deserialize for JSON-RPC 2.0"
    test_file: "crates/context-graph-steering/tests/mcp_tests.rs"
    spec_refs:
      - "REQ-STEER-008"
      - "TECH-STEER-012.5 Section 9"

  - id: "TASK-12.5-021"
    title: "Implement Module Initialization and Public API"
    description: |
      Implement lib.rs module exports:
      - pub mod gardener
      - pub mod curator
      - pub mod assessor
      - pub mod reward
      - pub mod config
      - pub mod error
      - pub mod session
      - pub mod context
      - pub mod integration
      - pub mod mcp
      - pub mod subsystem

      Re-export main types:
      - SteeringSubsystem
      - SteeringReward
      - SteeringComponents
      - SteeringConfig
      - SteeringError
      - Gardener
      - Curator
      - ThoughtAssessor

      Implement builder pattern:
      - SteeringSubsystemBuilder
        - with_config(config) -> Self
        - with_graph(graph) -> Self
        - with_neuromod(controller) -> Self
        - build() -> Result<SteeringSubsystem, SteeringError>

      Implement Cargo.toml with dependencies:
      - tokio (sync, time)
      - uuid (v4, serde)
      - chrono (serde)
      - serde (derive)
      - thiserror
      - context-graph-core
      - context-graph-neuromod
      - context-graph-dream
    layer: "surface"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-steering/src/lib.rs"
    dependencies:
      - "TASK-12.5-019"
      - "TASK-12.5-020"
    acceptance_criteria:
      - "All modules exported"
      - "Main types re-exported at crate root"
      - "Builder pattern for SteeringSubsystem"
      - "cargo check -p context-graph-steering succeeds"
      - "cargo doc generates documentation"
    test_file: "crates/context-graph-steering/tests/lib_tests.rs"
    spec_refs:
      - "REQ-STEER-001"
      - "TECH-STEER-012.5 Section 1.3"

  - id: "TASK-12.5-022"
    title: "Implement Integration Tests and Performance Benchmarks"
    description: |
      Implement integration tests:
      - test_store_memory_with_steering: SteeringReward returned with store_memory
      - test_module_10_dopamine_integration: Dopamine level changes on steering
      - test_mcp_tool_roundtrip: Valid response under 10ms
      - test_session_state_persistence: State maintained across calls
      - test_dream_layer_flagging: Low coherence flagged for review

      Implement unit tests:
      - test_gardener_long_term_value: Score in [-1, 1], latency <2ms
      - test_curator_quality_assessment: Score in [-1, 1], latency <2ms
      - test_thought_assessor_real_time: Score in [-1, 1], latency <1ms
      - test_unified_evaluation: All components combined, latency <5ms
      - test_dopamine_feedback_integration: Delta applied correctly
      - test_pruning_threshold_triggers: Suggestions at -0.5
      - test_consolidation_threshold_triggers: Suggestions at 0.7
      - test_novelty_detection: Duplicates scored lower
      - test_confidence_calculation: Agreement increases confidence
      - test_suggestion_limit: Max suggestions enforced

      Implement benchmarks:
      - bench_single_evaluation: <5ms P99
      - bench_gardener_component: <2ms P99
      - bench_curator_component: <2ms P99
      - bench_assessor_component: <1ms P99
      - bench_dopamine_feedback: <1ms P99
      - bench_throughput: >200 eval/sec sustained
    layer: "surface"
    priority: "high"
    estimated_hours: 4
    file_path: "crates/context-graph-steering/tests/"
    dependencies:
      - "TASK-12.5-021"
    acceptance_criteria:
      - "All unit tests pass"
      - "All integration tests pass"
      - "Benchmark P99 latencies meet targets"
      - "Throughput >200 evaluations/second"
      - "Test coverage >90%"
    test_file: "crates/context-graph-steering/tests/integration_tests.rs"
    spec_refs:
      - "SPEC-STEER-012.5 Section 7"
      - "TECH-STEER-012.5 Section 12"
```

---

## Dependency Graph

```
TASK-12.5-001 (SuggestionType, SteeringSuggestion) ─────────────────────────────────────────┐
TASK-12.5-002 (SteeringComponents) ─────────────────────────────────────────────────────────┤
                                                                                             │
TASK-12.5-001 + TASK-12.5-002 ──► TASK-12.5-003 (SteeringMetadata, SteeringReward) ─────────┤
                                                                                             │
TASK-12.5-004 (SteeringConfig) ─────────────────────────────────────────────────────────────┤
TASK-12.5-005 (SteeringError) ──────────────────────────────────────────────────────────────┤
                                                                                             │
TASK-12.5-002 ──► TASK-12.5-006 (RingBuffer, SteeringSessionState) ─────────────────────────┤
TASK-12.5-007 (SteeringContext) ────────────────────────────────────────────────────────────┤
                                                                                             │
TASK-12.5-008 (DecayConfig, UsagePatternAnalyzer) ──────────────────────────────────────────┤
TASK-12.5-009 (ConsolidationRule, PruningRule) ─────────────────────────────────────────────┤
                                                                                             │
TASK-12.5-008 + TASK-12.5-009 ──► TASK-12.5-010 (Gardener) ─────────────────────────────────┤
                                                                                             │
TASK-12.5-011 (QualityModel) ───────────────────────────────────────────────────────────────┤
                                                                                             │
TASK-12.5-001 + TASK-12.5-011 ──► TASK-12.5-012 (Curator) ──────────────────────────────────┤
                                                                                             │
TASK-12.5-006 ──► TASK-12.5-013 (ThoughtSignature, CoherenceModel) ─────────────────────────┤
                                                                                             │
TASK-12.5-006 + TASK-12.5-013 ──► TASK-12.5-014 (ThoughtAssessor) ──────────────────────────┤
                                                                                             │
TASK-12.5-007 + TASK-12.5-008 ──► TASK-12.5-015 (Context Preparation) ──────────────────────┤
                                                                                             │
TASK-12.5-002 + TASK-12.5-006 ──► TASK-12.5-016 (SteeringDopamineFeedback) ─────────────────┤
TASK-12.5-003 ──► TASK-12.5-017 (Edge EMA Update) ──────────────────────────────────────────┤
TASK-12.5-010 + TASK-12.5-012 ──► TASK-12.5-018 (DreamQualityAssessor) ─────────────────────┤
                                                                                             │
ALL ABOVE ──► TASK-12.5-019 (SteeringSubsystem Unified) ────────────────────────────────────┤
                                                                                             │
TASK-12.5-003 + TASK-12.5-005 + TASK-12.5-019 ──► TASK-12.5-020 (MCP Tool Handler) ─────────┤
                                                                                             │
TASK-12.5-019 + TASK-12.5-020 ──► TASK-12.5-021 (Module Init & Public API) ─────────────────┤
                                                                                             │
TASK-12.5-021 ──► TASK-12.5-022 (Integration Tests & Benchmarks) ◄──────────────────────────┘
```

---

## Implementation Order (Recommended)

### Week 1: Foundation + Components

1. TASK-12.5-001: SuggestionType, SteeringSuggestion
2. TASK-12.5-002: SteeringComponents
3. TASK-12.5-003: SteeringMetadata, SteeringReward
4. TASK-12.5-004: SteeringConfig
5. TASK-12.5-005: SteeringError
6. TASK-12.5-006: RingBuffer, SteeringSessionState
7. TASK-12.5-007: SteeringContext
8. TASK-12.5-008: DecayConfig, UsagePatternAnalyzer
9. TASK-12.5-009: ConsolidationRule, PruningRule
10. TASK-12.5-010: Gardener (<2ms)
11. TASK-12.5-011: QualityModel

### Week 2: Components + Integration + MCP

12. TASK-12.5-012: Curator (<2ms)
13. TASK-12.5-013: ThoughtSignature, CoherenceModel
14. TASK-12.5-014: ThoughtAssessor (<1ms)
15. TASK-12.5-015: Context Preparation
16. TASK-12.5-016: SteeringDopamineFeedback (<1ms)
17. TASK-12.5-017: Edge EMA Update
18. TASK-12.5-018: DreamQualityAssessor
19. TASK-12.5-019: SteeringSubsystem Unified (<5ms)
20. TASK-12.5-020: MCP Tool Handler (<10ms)
21. TASK-12.5-021: Module Init & Public API
22. TASK-12.5-022: Integration Tests & Benchmarks

---

## Quality Gates

| Gate | Criteria | Required For |
|------|----------|--------------|
| Foundation Complete | TASK-12.5-001 through TASK-12.5-007 pass all tests | Week 1.5 start |
| Components Complete | TASK-12.5-010, 12, 14 meet latency targets | Week 2 start |
| Integration Complete | TASK-12.5-016, 17, 18 pass tests | TASK-12.5-019 start |
| Unified Evaluation <5ms | TASK-12.5-019 E2E latency verified | TASK-12.5-020 start |
| MCP Handler <10ms | TASK-12.5-020 latency verified | Module complete |

---

## Performance Targets Summary

| Component | Budget | Metric |
|-----------|--------|--------|
| Gardener.evaluate_long_term_value() | <2ms | P99 latency |
| Curator.evaluate_quality() | <2ms | P99 latency |
| ThoughtAssessor.evaluate_immediate() | <1ms | P99 latency |
| SteeringDopamineFeedback.send_reward() | <1ms | P99 latency |
| SteeringSubsystem.evaluate() | <5ms | Total E2E |
| SteeringToolHandler.handle() | <10ms | MCP response |
| Edge EMA update | <1ms | Per node |
| Throughput | >200/sec | Sustained rate |

---

## Memory Budget

| Component | Budget |
|-----------|--------|
| Session state | <100KB |
| Recent thoughts buffer (100 entries) | <200KB |
| Dopamine feedback history | <50KB |
| Gardener usage analyzer | <100KB |
| **Total per session** | **<500KB** |

---

## Marblestone Integration Summary

This module implements Adam Marblestone's architectural separation between learning and steering mechanisms:

**Core Principle**: Steering influences learning through reward signals, not direct parameter modification.

**Three-Component Architecture**:
- **Gardener**: Long-term memory curation (days to months horizon)
- **Curator**: Quality assessment and domain relevance
- **Thought Assessor**: Real-time coherence and novelty evaluation

**Integration Points**:
- **Module 10 (Neuromodulation)**: Steering rewards modulate dopamine levels
- **Module 4 (Knowledge Graph)**: Edge steering_reward field updated via EMA
- **Module 9 (Dream Layer)**: Shortcut quality assessment for dream consolidation

**Reward Signal Flow**:
```
SteeringReward ──► SteeringDopamineFeedback ──► NeuromodulationController
                                                      │
                                                      v
                                              Hopfield beta modulation
                                              Learning rate adjustment
```

---

## Traceability Matrix

| Task ID | Requirements Covered |
|---------|---------------------|
| TASK-12.5-001 | REQ-STEER-002 |
| TASK-12.5-002 | REQ-STEER-002 |
| TASK-12.5-003 | REQ-STEER-002 |
| TASK-12.5-004 | REQ-STEER-001 |
| TASK-12.5-005 | REQ-STEER-001 |
| TASK-12.5-006 | REQ-STEER-001 |
| TASK-12.5-007 | REQ-STEER-006 |
| TASK-12.5-008 | REQ-STEER-003 |
| TASK-12.5-009 | REQ-STEER-003 |
| TASK-12.5-010 | REQ-STEER-003 |
| TASK-12.5-011 | REQ-STEER-004 |
| TASK-12.5-012 | REQ-STEER-004 |
| TASK-12.5-013 | REQ-STEER-005 |
| TASK-12.5-014 | REQ-STEER-005 |
| TASK-12.5-015 | REQ-STEER-006 |
| TASK-12.5-016 | REQ-STEER-007 |
| TASK-12.5-017 | REQ-STEER-006 |
| TASK-12.5-018 | REQ-STEER-008 |
| TASK-12.5-019 | REQ-STEER-001, REQ-STEER-006 |
| TASK-12.5-020 | REQ-STEER-008 |
| TASK-12.5-021 | REQ-STEER-001 |
| TASK-12.5-022 | All REQ-STEER-* |

---

## Critical Constraints

**LATENCY BUDGETS ARE HARD REQUIREMENTS**:
- Gardener: <2ms (inline, use #[inline])
- Curator: <2ms (inline, use #[inline])
- ThoughtAssessor: <1ms (rapid rejection path)
- Dopamine feedback: <1ms
- Total evaluation: <5ms
- MCP handler: <10ms (including serialization)

**WEIGHT DISTRIBUTION**:
- Gardener: 35%
- Curator: 35%
- Assessor: 30%
- Total: 100%

**THRESHOLDS**:
- Positive reward: >0.3
- Negative reward: <-0.3
- Prune suggestion: <-0.5
- Consolidation suggestion: >0.7
- Min coherence: 0.3 (rapid rejection)
- Min age for pruning: 24 hours

**DOPAMINE FEEDBACK**:
- Positive scale: 0.3
- Negative scale: 0.2 (gentler)
- Max delta: 0.2
- Min magnitude: 0.1

**MEMORY LIMITS**:
- 500KB maximum per session including all buffers
- 100 entries in novelty detection ring buffer
- 100 entries in dopamine feedback history

---

*Generated: 2025-12-31*
*Module: 12.5 - Steering Subsystem*
*Version: 1.0.0*
*Total Tasks: 22*
