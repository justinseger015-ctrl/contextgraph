# Module 12: Active Inference - Atomic Tasks

```yaml
metadata:
  module_id: "module-12"
  module_name: "Active Inference"
  version: "1.0.0"
  phase: 11
  total_tasks: 20
  approach: "inside-out-bottom-up"
  created: "2025-12-31"
  dependencies:
    - module-01-ghost-system
    - module-02-core-infrastructure
    - module-03-embedding-pipeline
    - module-04-knowledge-graph
    - module-05-utl-integration
    - module-06-bio-nervous-system
    - module-09-dream-layer
    - module-10-neuromodulation
    - module-11-immune-system
  estimated_duration: "2 weeks"
  spec_refs:
    - SPEC-ACTINF-012 (Functional)
    - TECH-ACTINF-012 (Technical)
```

---

## Task Overview

This module implements the Free Energy Principle (FEP) for proactive knowledge acquisition. The system transforms from passive storage to active learning by computing Expected Free Energy (EFE), performing variational belief updating, evaluating policies with temporal depth planning, and generating epistemic actions. Marblestone features include OmniInferenceEngine for omnidirectional inference (forward, backward, bidirectional, bridge, abduction) with clamped variable support.

### Key Mathematical Foundations

**Variational Free Energy**:
```
F = D_KL[q(s) || p(s|o)] - ln p(o)
```

**Expected Free Energy (EFE)**:
```
G(pi) = E_q[ln q(s') - ln p(o', s')]
      = Epistemic Value + Pragmatic Value
```

### Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| EFE Computation (single) | <10ms | SIMD-optimized matrix ops |
| Belief Update | <5ms | Variational inference step |
| Policy Evaluation (all) | <10ms | Tree search with pruning |
| Action Generation | <5ms | Template instantiation |
| Total Action Selection | <20ms | End-to-end pipeline |
| MCP Handler | <25ms | Including serialization |
| Omnidirectional Inference | <15ms | All directions |
| Backward Inference | <10ms | Abductive reasoning |

### Task Organization

1. **Foundation Layer** (Tasks 1-7): Core types, belief distribution, session state, configuration
2. **Logic Layer** (Tasks 8-14): Generative model, belief updater, policy evaluator, EFE computation
3. **Surface Layer** (Tasks 15-20): Epistemic actions, OmniInferenceEngine, MCP integration

---

## Foundation Layer: Core Types & Session State

```yaml
tasks:
  # ============================================================
  # FOUNDATION: Belief Distribution Types
  # ============================================================

  - id: "M12-T01"
    title: "Define BeliefDistribution Struct with Entropy and KL Divergence"
    description: |
      Implement BeliefDistribution struct for representing probability distributions
      over hidden states. Use diagonal Gaussian approximation for efficiency with
      1536D vectors aligned with embedding pipeline outputs.

      Fields: mean (DVector<f32>), precision (DVector<f32>), confidence (f32),
      observation_count (u64).

      Methods:
      - uniform_prior(dimension) -> Self
      - entropy() -> f32 [Constraint: <1ms]
      - kl_divergence(&other) -> f32 [Constraint: <1ms]

      For diagonal Gaussian: H = 0.5 * sum(log(2*pi*e/precision))
    layer: "foundation"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-active-inference/src/beliefs/distribution.rs"
    dependencies: []
    acceptance_criteria:
      - "BeliefDistribution struct compiles with all fields"
      - "uniform_prior() returns zero mean with unit precision"
      - "entropy() computes diagonal Gaussian entropy in <1ms"
      - "kl_divergence() computes KL(self || other) in <1ms"
      - "Handles numerical stability with precision floor (1e-10)"
      - "Clone, Debug implemented"
    test_file: "crates/context-graph-active-inference/tests/distribution_tests.rs"
    spec_refs:
      - "REQ-ACTINF-002"
      - "TECH-ACTINF-012 Section 2.2"

  - id: "M12-T02"
    title: "Define JohariQuadrant Enum and GoalPreferences Struct"
    description: |
      Implement JohariQuadrant enum for action prioritization based on
      knowledge state: Open (known to both), Blind (unknown to self),
      Hidden (known to self), Unknown (exploration frontier).

      Implement GoalPreferences struct for prior preferences representing goals:
      - preferred_states: Vec<DVector<f32>>
      - preference_weights: Vec<f32>
      - goal_descriptions: Vec<String>

      Default implementation returns empty preferences.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1.5
    file_path: "crates/context-graph-active-inference/src/beliefs/preferences.rs"
    dependencies: []
    acceptance_criteria:
      - "JohariQuadrant enum with 4 variants"
      - "Default for JohariQuadrant returns Unknown"
      - "GoalPreferences struct with 3 fields"
      - "Default for GoalPreferences returns empty preferences"
      - "Clone, Copy (enum), Debug, PartialEq, Eq implemented"
      - "serde Serialize/Deserialize for JSON output"
    test_file: "crates/context-graph-active-inference/tests/preferences_tests.rs"
    spec_refs:
      - "REQ-ACTINF-002"
      - "TECH-ACTINF-012 Section 2.2"

  - id: "M12-T03"
    title: "Define PredictionError and Observation Types"
    description: |
      Implement PredictionError struct for precision-weighted prediction errors:
      - prediction: DVector<f32>
      - observation: DVector<f32>
      - error: DVector<f32> (observation - prediction)
      - precision: f32
      - timestamp: Instant
      - domain: String

      Implement PredictionErrorBuffer with max_size, running average, and FIFO eviction.

      Implement Observation struct:
      - embedding: DVector<f32>
      - content: String
      - timestamp: Instant
      - source: ObservationSource (enum: UserQuery, MemoryRetrieval, ExternalApi, SystemGenerated)
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-active-inference/src/beliefs/observation.rs"
    dependencies: []
    acceptance_criteria:
      - "PredictionError struct with 6 fields"
      - "PredictionErrorBuffer with FIFO eviction at max_size"
      - "avg_error_magnitude computed and updated on insert"
      - "Observation struct with embedding, content, source"
      - "ObservationSource enum with 4 variants"
      - "Clone, Debug implemented"
    test_file: "crates/context-graph-active-inference/tests/observation_tests.rs"
    spec_refs:
      - "REQ-ACTINF-002"
      - "TECH-ACTINF-012 Section 2.2"

  - id: "M12-T04"
    title: "Define SessionBeliefs Struct"
    description: |
      Implement SessionBeliefs struct for session-level belief state management:
      - session_id: Uuid
      - state_beliefs: BeliefDistribution
      - goal_preferences: GoalPreferences
      - prediction_errors: PredictionErrorBuffer
      - domain_uncertainty: HashMap<String, f32>
      - observation_history: Vec<Observation>
      - johari_quadrant: JohariQuadrant
      - last_update: Instant
      - belief_entropy: f32

      Implement new() returning uniform prior session.
      Memory footprint must be under 500KB per session.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-active-inference/src/beliefs/session.rs"
    dependencies:
      - "M12-T01"
      - "M12-T02"
      - "M12-T03"
    acceptance_criteria:
      - "SessionBeliefs struct with 9 fields"
      - "new() creates session with Uuid::new_v4()"
      - "state_beliefs initialized with uniform_prior(1536)"
      - "prediction_errors buffer sized to 100 max"
      - "Memory footprint verified under 500KB"
      - "Clone implemented"
    test_file: "crates/context-graph-active-inference/tests/session_tests.rs"
    spec_refs:
      - "REQ-ACTINF-002"
      - "TECH-ACTINF-012 Section 2.2"

  # ============================================================
  # FOUNDATION: Configuration Types
  # ============================================================

  - id: "M12-T05"
    title: "Define ActiveInferenceConfig and Related Configurations"
    description: |
      Implement ActiveInferenceConfig struct:
      - uncertainty_threshold: f32 (default: 0.6)
      - max_actions_per_response: usize (default: 3)
      - exploration_budget: u32 (default: 10)
      - planning_depth: usize (default: 5)
      - precision_weight: f32 (default: 1.0)
      - epistemic_pragmatic_balance: f32 (default: 0.5)
      - efe_timeout: Duration (default: 10ms)
      - belief_timeout: Duration (default: 5ms)
      - action_timeout: Duration (default: 20ms)

      Implement Default trait with all specified defaults.
      Implement validate() method for range checking.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1.5
    file_path: "crates/context-graph-active-inference/src/config.rs"
    dependencies: []
    acceptance_criteria:
      - "ActiveInferenceConfig struct with 9 fields"
      - "Default implementation with all specified values"
      - "validate() checks: planning_depth in [1, 10], thresholds in [0, 1]"
      - "Clone, Debug implemented"
      - "serde Serialize/Deserialize for TOML loading"
    test_file: "crates/context-graph-active-inference/tests/config_tests.rs"
    spec_refs:
      - "REQ-ACTINF-001"
      - "TECH-ACTINF-012 Section 2.1"

  - id: "M12-T06"
    title: "Define ActiveInferenceMetrics Struct"
    description: |
      Implement ActiveInferenceMetrics struct for performance tracking:
      - efe_computations: u64
      - avg_efe_latency_us: f64
      - belief_updates: u64
      - avg_belief_latency_us: f64
      - actions_generated: u64
      - actions_helpful: u64
      - total_uncertainty_reduction: f64
      - last_update: Option<Instant>

      Implement helpfulness_ratio() -> f32 returning actions_helpful / actions_generated.
      Implement Default returning zeroed metrics.
    layer: "foundation"
    priority: "high"
    estimated_hours: 1
    file_path: "crates/context-graph-active-inference/src/metrics.rs"
    dependencies: []
    acceptance_criteria:
      - "ActiveInferenceMetrics struct with 8 fields"
      - "helpfulness_ratio() handles division by zero"
      - "Default returns all zeros"
      - "Clone, Debug implemented"
    test_file: "crates/context-graph-active-inference/tests/metrics_tests.rs"
    spec_refs:
      - "REQ-ACTINF-001"
      - "TECH-ACTINF-012 Section 10"

  - id: "M12-T07"
    title: "Define ActiveInferenceError Enum with MCP Codes"
    description: |
      Implement ActiveInferenceError enum using thiserror:
      - SessionNotFound(Uuid) -> -32000
      - BeliefUpdateFailed(String) -> -32010
      - EfeTimeout(u64) -> -32011
      - InvalidModel(String) -> -32012
      - PolicyEvaluationFailed(String) -> -32013
      - ActionGenerationFailed(String) -> -32014
      - NumericalInstability(String) -> -32015
      - ConfigurationError(String) -> -32016
      - Internal(String) -> -32603

      Implement error_code() -> i32 returning MCP-compliant error codes.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1
    file_path: "crates/context-graph-active-inference/src/error.rs"
    dependencies: []
    acceptance_criteria:
      - "ActiveInferenceError enum with 9 variants"
      - "All variants have #[error()] descriptive messages"
      - "error_code() returns correct MCP codes"
      - "Error is Send + Sync"
      - "Debug implemented"
    test_file: "crates/context-graph-active-inference/tests/error_tests.rs"
    spec_refs:
      - "TECH-ACTINF-012 Section 9"

  # ============================================================
  # LOGIC LAYER: Generative Model
  # ============================================================

  - id: "M12-T08"
    title: "Implement GenerativeModelConfig and GenerativeLayer"
    description: |
      Implement GenerativeModelConfig struct:
      - num_layers: usize (default: 5, matching bio-nervous system)
      - layer_dims: Vec<usize> (default: [1536, 768, 384, 192, 96])
      - learning_rate: f32 (default: 0.001)
      - layer_precisions: Vec<f32> (default: [1.0, 0.8, 0.6, 0.4, 0.2])

      Implement GenerativeLayer struct:
      - level: usize (0 = sensory, higher = abstract)
      - dimension: usize
      - state: DVector<f32>
      - precision: f32
      - downward_weights: Option<DMatrix<f32>>
      - upward_weights: Option<DMatrix<f32>>
    layer: "logic"
    priority: "critical"
    estimated_hours: 2.5
    file_path: "crates/context-graph-active-inference/src/generative/layer.rs"
    dependencies:
      - "M12-T01"
    acceptance_criteria:
      - "GenerativeModelConfig struct with 4 fields"
      - "Default matches 5-layer bio-nervous system dimensions"
      - "GenerativeLayer struct with 6 fields"
      - "Weight matrices sized correctly for inter-layer connections"
      - "Clone, Debug implemented"
    test_file: "crates/context-graph-active-inference/tests/generative_layer_tests.rs"
    spec_refs:
      - "REQ-ACTINF-003"
      - "TECH-ACTINF-012 Section 4"

  - id: "M12-T09"
    title: "Implement LikelihoodModel and Prediction Types"
    description: |
      Implement LikelihoodModel struct for p(o|s):
      - state_to_obs: DMatrix<f32> (mapping from hidden states to observations)
      - obs_precision: f32 (observation noise precision)
      - domain_adjustments: HashMap<String, f32>

      Implement new(dimension) initializing identity mapping.

      Implement Prediction struct:
      - observation: DVector<f32>
      - confidence: f32
      - latency: Duration
    layer: "logic"
    priority: "critical"
    estimated_hours: 1.5
    file_path: "crates/context-graph-active-inference/src/generative/likelihood.rs"
    dependencies:
      - "M12-T01"
    acceptance_criteria:
      - "LikelihoodModel struct with 3 fields"
      - "new() initializes identity state_to_obs matrix"
      - "Prediction struct with 3 fields"
      - "Clone implemented"
    test_file: "crates/context-graph-active-inference/tests/likelihood_tests.rs"
    spec_refs:
      - "REQ-ACTINF-003"
      - "TECH-ACTINF-012 Section 4"

  - id: "M12-T10"
    title: "Implement GenerativeModel with Hierarchical Prediction"
    description: |
      Implement GenerativeModel struct:
      - layers: Vec<GenerativeLayer>
      - connections: Vec<LayerConnection>
      - prior: BeliefDistribution
      - likelihood: LikelihoodModel
      - config: GenerativeModelConfig

      Methods:
      - new(config) -> Self: Initialize 5-layer hierarchy
      - predict(&beliefs) -> Prediction [Constraint: <2ms]
      - compute_prediction_error(&predicted, &observed) -> PredictionError [Constraint: <1ms]
      - update_from_error(&error, learning_rate) [Constraint: <3ms]

      predict() implements top-down propagation through hierarchy.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-active-inference/src/generative/model.rs"
    dependencies:
      - "M12-T01"
      - "M12-T03"
      - "M12-T08"
      - "M12-T09"
    acceptance_criteria:
      - "GenerativeModel struct with 5 fields"
      - "new() creates 5-layer hierarchy with connected weights"
      - "predict() completes in <2ms for 1536D input"
      - "compute_prediction_error() returns precision-weighted error in <1ms"
      - "update_from_error() applies gradient descent in <3ms"
      - "Layer dimensions match embedding pipeline outputs"
    test_file: "crates/context-graph-active-inference/tests/generative_model_tests.rs"
    spec_refs:
      - "REQ-ACTINF-003"
      - "TECH-ACTINF-012 Section 4"

  # ============================================================
  # LOGIC LAYER: Belief Updater (Variational Inference)
  # ============================================================

  - id: "M12-T11"
    title: "Implement BeliefUpdaterConfig and IterationMetrics"
    description: |
      Implement BeliefUpdaterConfig struct:
      - max_iterations: usize (default: 10)
      - convergence_threshold: f32 (default: 0.001)
      - learning_rate: f32 (default: 0.1)
      - momentum: f32 (default: 0.9)
      - precision_floor: f32 (default: 1e-6)
      - max_update_magnitude: f32 (default: 1.0)

      Implement IterationMetrics struct:
      - iteration: usize
      - free_energy: f32
      - kl_delta: f32
      - update_magnitude: f32
      - duration: Duration
    layer: "logic"
    priority: "critical"
    estimated_hours: 1
    file_path: "crates/context-graph-active-inference/src/inference/config.rs"
    dependencies: []
    acceptance_criteria:
      - "BeliefUpdaterConfig struct with 6 fields"
      - "Default implementation with all specified values"
      - "IterationMetrics struct with 5 fields"
      - "Clone, Debug implemented"
    test_file: "crates/context-graph-active-inference/tests/updater_config_tests.rs"
    spec_refs:
      - "REQ-ACTINF-004"
      - "TECH-ACTINF-012 Section 5"

  - id: "M12-T12"
    title: "Implement BeliefUpdater with Variational Inference"
    description: |
      Implement BeliefUpdater struct:
      - config: BeliefUpdaterConfig
      - iteration_history: Vec<IterationMetrics>
      - converged: bool

      Methods:
      - new(config) -> Self
      - update_beliefs(&mut beliefs, &observation, &model) -> BeliefUpdateResult
        [Constraint: <5ms]
      - compute_free_energy(&beliefs, &observation, &model) -> f32

      update_beliefs() implements:
      1. Compute prediction from current beliefs
      2. Compute precision-weighted prediction error
      3. Gradient of free energy: dF/dmu = precision * error + prior_precision * (mu - mu_prior)
      4. Apply momentum-based update with gradient clipping
      5. Update precision based on error variance
      6. Check convergence via KL divergence

      BeliefUpdateResult contains: converged, iterations, final_free_energy,
      uncertainty_reduction, latency.
    layer: "logic"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-active-inference/src/inference/variational.rs"
    dependencies:
      - "M12-T01"
      - "M12-T03"
      - "M12-T10"
      - "M12-T11"
    acceptance_criteria:
      - "BeliefUpdater struct with 3 fields"
      - "update_beliefs() completes in <5ms"
      - "Convergence detection via KL divergence threshold"
      - "Momentum-based optimization implemented"
      - "Gradient clipping at max_update_magnitude"
      - "Precision updating from error variance"
      - "BeliefUpdateResult includes all 5 fields"
      - "No NaN/Inf values (precision floor enforced)"
    test_file: "crates/context-graph-active-inference/tests/variational_tests.rs"
    spec_refs:
      - "REQ-ACTINF-004"
      - "TECH-ACTINF-012 Section 5"

  # ============================================================
  # LOGIC LAYER: Policy Evaluator (EFE Computation)
  # ============================================================

  - id: "M12-T13"
    title: "Implement PolicyEvaluatorConfig and PolicyTemplate"
    description: |
      Implement PolicyEvaluatorConfig struct:
      - max_depth: usize (default: 5)
      - branching_factor: usize (default: 4)
      - discount_factor: f32 (default: 0.9)
      - pruning_threshold: f32 (default: 0.01)
      - efe_timeout: Duration (default: 2ms)
      - selection_temperature: f32 (default: 1.0)

      Implement PolicyTemplate struct:
      - id: String
      - actions: Vec<EpistemicActionType>
      - prior_prob: f32
      - description: String

      Implement PolicyCacheKey with policy_id, belief_hash, depth.
      Implement PolicyEvaluationMetrics tracking evaluations and cache hits.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-active-inference/src/policy/config.rs"
    dependencies: []
    acceptance_criteria:
      - "PolicyEvaluatorConfig struct with 6 fields"
      - "Default implementation with all specified values"
      - "PolicyTemplate struct with 4 fields"
      - "PolicyCacheKey implements Hash, Eq for HashMap use"
      - "PolicyEvaluationMetrics tracks policies_evaluated, cache_hits"
      - "Clone, Debug implemented"
    test_file: "crates/context-graph-active-inference/tests/policy_config_tests.rs"
    spec_refs:
      - "REQ-ACTINF-005"
      - "TECH-ACTINF-012 Section 6"

  - id: "M12-T14"
    title: "Implement PolicyEvaluator with EFE Computation"
    description: |
      Implement PolicyEvaluator struct:
      - policy_templates: Vec<PolicyTemplate>
      - efe_cache: HashMap<PolicyCacheKey, f32>
      - config: PolicyEvaluatorConfig
      - metrics: PolicyEvaluationMetrics

      Methods:
      - new(config) -> Self (with default policy templates)
      - evaluate_policies(&beliefs, &model, &goals) -> Vec<PolicyEvaluation>
        [Constraint: Total <10ms]
      - compute_efe(&policy, &beliefs, &model, &goals, depth) -> f32
        [Constraint: Single <2ms]
      - compute_epistemic_value(&beliefs, &model) -> f32
      - compute_pragmatic_value(&beliefs, &goals) -> f32
      - apply_softmax_probabilities(&mut evaluations)

      EFE = Epistemic + Pragmatic
      Epistemic = -E_q[H[p(o|s)]] (negative expected ambiguity)
      Pragmatic = E_q[ln p(o|C)] (expected log preference via cosine similarity)

      PolicyEvaluation contains: policy_id, efe, epistemic_value, pragmatic_value, probability.
    layer: "logic"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-active-inference/src/policy/evaluator.rs"
    dependencies:
      - "M12-T01"
      - "M12-T02"
      - "M12-T04"
      - "M12-T10"
      - "M12-T13"
    acceptance_criteria:
      - "PolicyEvaluator struct with 4 fields"
      - "Default policy templates include explore, clarify, hypothesize, example"
      - "evaluate_policies() completes in <10ms for 4 policies"
      - "compute_efe() completes in <2ms per policy"
      - "EFE correctly decomposes into epistemic + pragmatic"
      - "Softmax probabilities sum to 1.0"
      - "EFE cache reduces redundant computation"
      - "Planning depth of 5 supported with discount factor"
    test_file: "crates/context-graph-active-inference/tests/evaluator_tests.rs"
    spec_refs:
      - "REQ-ACTINF-005"
      - "TECH-ACTINF-012 Section 6"

  # ============================================================
  # SURFACE LAYER: Epistemic Action Generator
  # ============================================================

  - id: "M12-T15"
    title: "Implement EpistemicActionType Enum and KnowledgeGap Struct"
    description: |
      Implement EpistemicActionType enum:
      - SeekClarification
      - RequestExample
      - ProposeHypothesis
      - SuggestExperiment
      - ExploreRelated
      - ConfirmUnderstanding

      Implement KnowledgeGap struct:
      - id: Uuid
      - description: String
      - domain: String
      - severity: f32
      - related_nodes: Vec<Uuid>

      Implement EpistemicAction struct:
      - action_type: EpistemicActionType
      - content: String
      - target_concept: Option<String>
      - expected_info_gain: f32
      - priority: f32
      - relevance: f32
      - knowledge_gap: Option<KnowledgeGap>
    layer: "surface"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-active-inference/src/actions/types.rs"
    dependencies: []
    acceptance_criteria:
      - "EpistemicActionType enum with 6 variants"
      - "Clone, Debug, PartialEq, Eq, Hash implemented for enum"
      - "KnowledgeGap struct with 5 fields"
      - "EpistemicAction struct with 7 fields"
      - "Clone, Debug implemented"
    test_file: "crates/context-graph-active-inference/tests/action_types_tests.rs"
    spec_refs:
      - "REQ-ACTINF-006"
      - "TECH-ACTINF-012 Section 7"

  - id: "M12-T16"
    title: "Implement ActionRateLimiter for Spam Prevention"
    description: |
      Implement ActionRateLimiter struct:
      - recent_actions: HashMap<EpistemicActionType, Vec<Instant>>
      - cooldown: Duration
      - max_per_type: usize

      Methods:
      - new(cooldown, max_per_type) -> Self
      - is_allowed(&action_type) -> bool
        (returns true if fewer than max_per_type actions within cooldown)
      - record_action(action_type)
        (adds current timestamp to recent_actions)

      Constraint: <3 suggestions per response default.
    layer: "surface"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-active-inference/src/actions/rate_limiter.rs"
    dependencies:
      - "M12-T15"
    acceptance_criteria:
      - "ActionRateLimiter struct with 3 fields"
      - "is_allowed() correctly counts actions within cooldown window"
      - "record_action() appends timestamp"
      - "Old timestamps outside cooldown are effectively ignored"
      - "Clone implemented"
    test_file: "crates/context-graph-active-inference/tests/rate_limiter_tests.rs"
    spec_refs:
      - "REQ-ACTINF-006"
      - "TECH-ACTINF-012 Section 7"

  - id: "M12-T17"
    title: "Implement EpistemicActionGenerator"
    description: |
      Implement EpistemicActionConfig struct:
      - uncertainty_threshold: f32 (default: 0.6)
      - max_actions: usize (default: 3)
      - action_cooldown: u64 (default: 60 seconds)
      - min_info_gain: f32 (default: 0.1)
      - require_diversity: bool (default: true)

      Implement EpistemicActionGenerator struct:
      - action_templates: Vec<ActionTemplate>
      - rate_limiter: ActionRateLimiter
      - config: EpistemicActionConfig
      - metrics: ActionGenerationMetrics

      Methods:
      - new(config) -> Self
      - generate_actions(&beliefs, &context) -> Vec<EpistemicAction>
        [Constraint: <5ms]
      - identify_knowledge_gaps(&beliefs, &context) -> Vec<KnowledgeGap>
      - generate_action_for_gap(&gap, &beliefs, &context) -> EpistemicAction
      - ensure_diversity(actions) -> Vec<EpistemicAction>

      ActionContext contains: current_domain, recent_queries, urgency, session_id.
    layer: "surface"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-active-inference/src/actions/generator.rs"
    dependencies:
      - "M12-T04"
      - "M12-T15"
      - "M12-T16"
    acceptance_criteria:
      - "EpistemicActionConfig struct with 5 fields"
      - "EpistemicActionGenerator struct with 4 fields"
      - "generate_actions() completes in <5ms"
      - "Returns empty if belief_entropy < uncertainty_threshold"
      - "Johari quadrant drives action type selection"
      - "Rate limiting enforced (<3 per response)"
      - "Diversity ensured (no duplicate action types)"
      - "Actions sorted by priority"
    test_file: "crates/context-graph-active-inference/tests/generator_tests.rs"
    spec_refs:
      - "REQ-ACTINF-006"
      - "TECH-ACTINF-012 Section 7"

  # ============================================================
  # SURFACE LAYER: Omnidirectional Inference Engine (Marblestone)
  # ============================================================

  - id: "M12-T18"
    title: "Implement InferenceDirection Enum and ClampedValue Struct (Marblestone)"
    description: |
      Implement InferenceDirection enum for omnidirectional inference:
      - Forward: A -> B (cause to effect)
      - Backward: B -> A (effect to cause, abduction)
      - Bidirectional: A <-> B (both directions)
      - Bridge: Cross-domain inference
      - Abduction: Best explanation for observations

      Implement ClampType enum:
      - Hard: Value cannot change during inference
      - Soft { prior_strength: f32 }: Value biased but can change

      Implement ClampedValue struct:
      - node_id: Uuid
      - value: f32
      - is_observation: bool
      - clamp_type: ClampType

      Implement CausalEdge struct for graph traversal:
      - source: Uuid, target: Uuid, edge_type: String
      - strength: f32, prior_probability: f32, depth: u32, domain: String

      Implement InferenceStep for explanation chains:
      - from_node, to_node, edge_type, belief_delta, reasoning
    layer: "surface"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-active-inference/src/inference/omni_types.rs"
    dependencies: []
    acceptance_criteria:
      - "InferenceDirection enum with 5 variants"
      - "ClampType enum with Hard and Soft variants"
      - "ClampedValue struct with 4 fields"
      - "CausalEdge struct with 7 fields"
      - "InferenceStep struct with 5 fields"
      - "Clone, Debug, PartialEq implemented"
    test_file: "crates/context-graph-active-inference/tests/omni_types_tests.rs"
    spec_refs:
      - "REQ-ACTINF-008"
      - "REQ-ACTINF-010"
      - "TECH-ACTINF-012 Section 3"

  - id: "M12-T19"
    title: "Implement OmniInferenceEngine with All Inference Directions (Marblestone)"
    description: |
      Implement OmniInferenceConfig struct:
      - max_depth: u32 (default: 5)
      - step_timeout: Duration (default: 2ms)
      - belief_threshold: f32 (default: 0.1)
      - max_candidates: usize (default: 10)

      Implement OmniInferenceEngine struct:
      - active_inference: Arc<RwLock<ActiveInferenceEngine>>
      - clamped_variables: HashMap<Uuid, ClampedValue>
      - direction: InferenceDirection
      - config: OmniInferenceConfig
      - causal_edge_cache: LruCache<Uuid, Vec<CausalEdge>>
      - metrics: OmniInferenceMetrics

      Methods:
      - new(config) -> Self
      - omni_infer(&query_nodes, direction, &clamped) -> Result<InferenceResult>
        [Constraint: <15ms]
      - forward_inference(&causes) -> Result<InferenceResult>
      - backward_inference(&effects) -> Result<InferenceResult> [Constraint: <10ms]
      - bidirectional_inference(&nodes) -> Result<InferenceResult>
      - bridge_inference(&nodes) -> Result<InferenceResult>
      - abductive_inference(&observations) -> Result<InferenceResult>
      - compute_cause_belief(&edge, effect_id) -> Result<f32> (Bayesian)
      - compute_effect_belief(&edge, cause_id) -> Result<f32>

      backward_inference implements: P(Cause | Effect) = P(Effect | Cause) * P(Cause) / P(Effect)

      InferenceResult contains: inferred_nodes, beliefs, direction, latency, steps, explanation_chain.
    layer: "surface"
    priority: "critical"
    estimated_hours: 6
    file_path: "crates/context-graph-active-inference/src/inference/omni.rs"
    dependencies:
      - "M12-T07"
      - "M12-T18"
    acceptance_criteria:
      - "OmniInferenceEngine struct with 6 fields"
      - "omni_infer() completes in <15ms for all directions"
      - "backward_inference() identifies likely causes in <10ms"
      - "Hard clamps keep values unchanged during inference"
      - "Soft clamps bias values with prior_strength weighting"
      - "Explanation chain provides interpretable reasoning"
      - "bidirectional_inference() runs forward and backward in parallel"
      - "abductive_inference() scores hypotheses by observation coverage"
      - "LRU cache improves repeated edge traversal"
    test_file: "crates/context-graph-active-inference/tests/omni_inference_tests.rs"
    spec_refs:
      - "REQ-ACTINF-008"
      - "REQ-ACTINF-009"
      - "REQ-ACTINF-010"
      - "TECH-ACTINF-012 Section 3"

  # ============================================================
  # SURFACE LAYER: MCP Integration & Engine Assembly
  # ============================================================

  - id: "M12-T20"
    title: "Implement ActiveInferenceEngine and MCP epistemic_action Handler"
    description: |
      Implement ActiveInferenceEngine struct:
      - generative_model: GenerativeModel
      - belief_updater: BeliefUpdater
      - policy_evaluator: PolicyEvaluator
      - action_generator: EpistemicActionGenerator
      - omni_engine: OmniInferenceEngine
      - session_beliefs: Arc<RwLock<SessionBeliefs>>
      - config: ActiveInferenceConfig
      - metrics: ActiveInferenceMetrics

      Methods:
      - new(config) -> Self
      - process_observation(&observation) -> Result<BeliefUpdateResult>
        [Constraint: <10ms]
      - select_action() -> Result<Vec<EpistemicAction>> [Constraint: <20ms]

      Implement MCP types:
      - EpistemicActionRequest: session_id, force, max_actions, min_relevance
      - EpistemicActionResponse: actions, uncertainty_level, johari_quadrant,
        utl_metrics, pulse (CognitivePulse), latency_ms
      - EpistemicActionOutput: action_type, content, target_concept,
        expected_info_gain, relevance, priority
      - CognitivePulse: entropy, coherence, suggested_action

      Implement EpistemicActionHandler:
      - handle(request) -> Result<EpistemicActionResponse> [Constraint: <25ms]

      Suggested actions: "continue", "explore", "trigger_dream" based on entropy/confidence.
    layer: "surface"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-active-inference/src/engine.rs"
    dependencies:
      - "M12-T04"
      - "M12-T05"
      - "M12-T06"
      - "M12-T07"
      - "M12-T10"
      - "M12-T12"
      - "M12-T14"
      - "M12-T17"
      - "M12-T19"
    acceptance_criteria:
      - "ActiveInferenceEngine struct with 8 fields"
      - "new() initializes all components with default configs"
      - "process_observation() updates beliefs in <10ms"
      - "select_action() returns prioritized actions in <20ms"
      - "MCP handler completes in <25ms including serialization"
      - "CognitivePulse header included in all responses"
      - "Force flag bypasses uncertainty threshold"
      - "Relevance filtering applied"
      - "Graceful error handling for missing sessions"
      - "Thread-safe session beliefs via Arc<RwLock>"
      - "Memory footprint under 2MB per session"
    test_file: "crates/context-graph-active-inference/tests/engine_tests.rs"
    spec_refs:
      - "REQ-ACTINF-001"
      - "REQ-ACTINF-007"
      - "TECH-ACTINF-012 Section 2, 8"
```

---

## Dependency Graph

```
M12-T01 (BeliefDistribution) ────────────────────────────────────────────────────┐
M12-T02 (JohariQuadrant, GoalPreferences) ───────────────────────────────────────┤
M12-T03 (PredictionError, Observation) ──────────────────────────────────────────┤
                                                                                  │
M12-T01 + M12-T02 + M12-T03 ──► M12-T04 (SessionBeliefs) ────────────────────────┤
                                                                                  │
M12-T05 (ActiveInferenceConfig) ─────────────────────────────────────────────────┤
M12-T06 (ActiveInferenceMetrics) ────────────────────────────────────────────────┤
M12-T07 (ActiveInferenceError) ──────────────────────────────────────────────────┤
                                                                                  │
M12-T01 ──► M12-T08 (GenerativeModelConfig, GenerativeLayer) ────────────────────┤
M12-T01 ──► M12-T09 (LikelihoodModel, Prediction) ───────────────────────────────┤
                                                                                  │
M12-T01 + M12-T03 + M12-T08 + M12-T09 ──► M12-T10 (GenerativeModel) ─────────────┤
                                                                                  │
M12-T11 (BeliefUpdaterConfig, IterationMetrics) ─────────────────────────────────┤
                                                                                  │
M12-T01 + M12-T03 + M12-T10 + M12-T11 ──► M12-T12 (BeliefUpdater) ────────────────┤
                                                                                  │
M12-T13 (PolicyEvaluatorConfig, PolicyTemplate) ─────────────────────────────────┤
                                                                                  │
M12-T01 + M12-T02 + M12-T04 + M12-T10 + M12-T13 ──► M12-T14 (PolicyEvaluator) ────┤
                                                                                  │
M12-T15 (EpistemicActionType, KnowledgeGap, EpistemicAction) ────────────────────┤
                                                                                  │
M12-T15 ──► M12-T16 (ActionRateLimiter) ─────────────────────────────────────────┤
                                                                                  │
M12-T04 + M12-T15 + M12-T16 ──► M12-T17 (EpistemicActionGenerator) ──────────────┤
                                                                                  │
M12-T18 (InferenceDirection, ClampedValue, CausalEdge) ──────────────────────────┤
                                                                                  │
M12-T07 + M12-T18 ──► M12-T19 (OmniInferenceEngine) ─────────────────────────────┤
                                                                                  │
ALL ABOVE ──► M12-T20 (ActiveInferenceEngine + MCP Handler) ◄────────────────────┘
```

---

## Implementation Order (Recommended)

### Week 1: Foundation + Logic

1. M12-T01: BeliefDistribution (entropy, KL divergence)
2. M12-T02: JohariQuadrant, GoalPreferences
3. M12-T03: PredictionError, Observation
4. M12-T04: SessionBeliefs
5. M12-T05: ActiveInferenceConfig
6. M12-T06: ActiveInferenceMetrics
7. M12-T07: ActiveInferenceError
8. M12-T08: GenerativeModelConfig, GenerativeLayer
9. M12-T09: LikelihoodModel, Prediction
10. M12-T10: GenerativeModel

### Week 2: Logic + Surface + Integration

11. M12-T11: BeliefUpdaterConfig
12. M12-T12: BeliefUpdater (variational inference)
13. M12-T13: PolicyEvaluatorConfig, PolicyTemplate
14. M12-T14: PolicyEvaluator (EFE computation)
15. M12-T15: EpistemicActionType, KnowledgeGap
16. M12-T16: ActionRateLimiter
17. M12-T17: EpistemicActionGenerator
18. M12-T18: InferenceDirection, ClampedValue (Marblestone)
19. M12-T19: OmniInferenceEngine (Marblestone)
20. M12-T20: ActiveInferenceEngine + MCP Handler

---

## Quality Gates

| Gate | Criteria | Required For |
|------|----------|--------------|
| Foundation Complete | M12-T01 through M12-T07 pass all tests | Week 1.5 start |
| Logic Complete | M12-T08 through M12-T14 pass latency tests | Week 2 start |
| Surface Complete | M12-T15 through M12-T20 pass all tests | Module complete |
| E2E MCP Test | Handler responds in <25ms | Module 12.5 start |

---

## Performance Targets Summary

| Component | Budget | Metric |
|-----------|--------|--------|
| BeliefDistribution.entropy() | <1ms | Single computation |
| BeliefDistribution.kl_divergence() | <1ms | Single computation |
| GenerativeModel.predict() | <2ms | Top-down propagation |
| GenerativeModel.compute_prediction_error() | <1ms | Error computation |
| GenerativeModel.update_from_error() | <3ms | Learning step |
| BeliefUpdater.update_beliefs() | <5ms | Variational inference |
| PolicyEvaluator.evaluate_policies() | <10ms | All policies |
| PolicyEvaluator.compute_efe() | <2ms | Single policy |
| EpistemicActionGenerator.generate_actions() | <5ms | Action generation |
| OmniInferenceEngine.omni_infer() | <15ms | All directions |
| OmniInferenceEngine.backward_inference() | <10ms | Abduction |
| ActiveInferenceEngine.process_observation() | <10ms | Observation processing |
| ActiveInferenceEngine.select_action() | <20ms | Action selection |
| EpistemicActionHandler.handle() | <25ms | MCP response |

---

## Memory Budget

| Component | Budget |
|-----------|--------|
| SessionBeliefs (per session) | <500KB |
| GenerativeModel (shared) | <10MB |
| Policy EFE Cache | <1MB |
| Action History (per session) | <100KB |
| OmniInference Edge Cache | <1MB |
| **Total per Session** | **<2MB** |

---

## Marblestone Integration Summary

Tasks with Marblestone features:
- **M12-T18**: InferenceDirection enum (Forward, Backward, Bidirectional, Bridge, Abduction)
- **M12-T18**: ClampType enum (Hard, Soft clamping for constrained inference)
- **M12-T18**: ClampedValue struct for fixing variables during inference
- **M12-T19**: OmniInferenceEngine with all 5 inference directions
- **M12-T19**: Backward inference for causal reasoning (abduction)
- **M12-T19**: Explanation chains for interpretability

---

## Critical Constraints

**LATENCY BUDGETS ARE HARD REQUIREMENTS**:
- Entropy/KL: <1ms (inline, SIMD where possible)
- Prediction: <2ms (vectorized matrix ops)
- Belief Update: <5ms (max 10 iterations)
- Policy Evaluation: <10ms total (cache and prune)
- Action Generation: <5ms (template-based)
- Omni Inference: <15ms (parallel where possible)
- MCP Handler: <25ms (including serialization)

**NUMERICAL STABILITY**:
- Precision floor: 1e-6 minimum
- Gradient clipping: max magnitude 1.0
- Softmax overflow prevention: subtract max EFE

**RATE LIMITING**:
- Maximum 3 epistemic actions per response
- 60 second cooldown per action type

**MEMORY LIMITS**:
- 2MB maximum per session including all buffers
- 500KB for SessionBeliefs struct

---

## Traceability Matrix

| Task ID | Requirements Covered |
|---------|---------------------|
| M12-T01 | REQ-ACTINF-002 |
| M12-T02 | REQ-ACTINF-002 |
| M12-T03 | REQ-ACTINF-002 |
| M12-T04 | REQ-ACTINF-002 |
| M12-T05 | REQ-ACTINF-001 |
| M12-T06 | REQ-ACTINF-001 |
| M12-T07 | REQ-ACTINF-001 |
| M12-T08 | REQ-ACTINF-003 |
| M12-T09 | REQ-ACTINF-003 |
| M12-T10 | REQ-ACTINF-003 |
| M12-T11 | REQ-ACTINF-004 |
| M12-T12 | REQ-ACTINF-004 |
| M12-T13 | REQ-ACTINF-005 |
| M12-T14 | REQ-ACTINF-005 |
| M12-T15 | REQ-ACTINF-006 |
| M12-T16 | REQ-ACTINF-006 |
| M12-T17 | REQ-ACTINF-006 |
| M12-T18 | REQ-ACTINF-008, REQ-ACTINF-010 |
| M12-T19 | REQ-ACTINF-008, REQ-ACTINF-009, REQ-ACTINF-010 |
| M12-T20 | REQ-ACTINF-001, REQ-ACTINF-007 |

---

*Generated: 2025-12-31*
*Module: 12 - Active Inference*
*Version: 1.0.0*
*Total Tasks: 20*
