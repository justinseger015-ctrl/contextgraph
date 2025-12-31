# Module 5: UTL Integration - Atomic Tasks

```yaml
metadata:
  module_id: "module-05"
  module_name: "UTL Integration"
  version: "1.0.0"
  phase: 4
  total_tasks: 25
  approach: "inside-out-bottom-up"
  created: "2025-12-31"
  dependencies:
    - module-02-core-infrastructure
    - module-03-embedding-pipeline
    - module-04-knowledge-graph
  estimated_duration: "4 weeks"
  spec_refs:
    - SPEC-UTL-005 (Functional)
    - TECH-UTL-005 (Technical)
```

---

## Task Overview

This module implements the Unified Theory of Learning (UTL), the core learning equation that governs memory acquisition, prioritization, and consolidation. Tasks are organized in inside-out, bottom-up order:

1. **Foundation Layer** (Tasks 1-8): Core types - Configuration, LifecycleStage, LambdaWeights, JohariQuadrant
2. **Logic Layer** (Tasks 9-17): Surprise computation, Coherence tracking, Emotional weighting, Phase oscillator
3. **Surface Layer** (Tasks 18-25): UTLProcessor, MCP integration, testing, benchmarks

---

## Foundation Layer: Core Types

```yaml
tasks:
  # ============================================================
  # FOUNDATION: Configuration Types
  # ============================================================

  - id: "M05-T01"
    title: "Define UtlConfig and UtlThresholds Structs"
    description: |
      Implement UtlConfig struct containing all UTL configuration parameters.
      Fields: learning_scale_factor (2.0), thresholds (UtlThresholds), salience_update_alpha (0.3),
      surprise (SurpriseConfig), coherence (CoherenceConfig), emotional (EmotionalConfig),
      phase (PhaseConfig), johari (JohariConfig), lifecycle (LifecycleConfig).
      UtlThresholds: consolidation_trigger (0.7), salience_update_min (0.1), surprise_significant (0.6).
      Include Default impl with all specified values.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-utl/src/config.rs"
    dependencies: []
    acceptance_criteria:
      - "UtlConfig struct compiles with all 9 fields"
      - "UtlThresholds struct with 3 threshold values"
      - "Default impl returns spec-defined values"
      - "Serde Serialize/Deserialize implemented"
      - "Clone, Debug traits implemented"
    test_file: "crates/context-graph-utl/tests/config_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 10"
      - "SPEC-UTL-005 Section 15.1"

  - id: "M05-T02"
    title: "Define SurpriseConfig for KL Divergence Parameters"
    description: |
      Implement SurpriseConfig struct for surprise computation configuration.
      Fields: kl_weight (0.6), distance_weight (0.4), kl (KlConfig),
      context_window_size (50), context_decay (0.95), max_surprise_no_context (0.9),
      min_context_for_kl (3).
      KlConfig: epsilon (1e-10), max_kl_value (10.0), temperature (1.0).
      Constraint: kl_weight + distance_weight = 1.0.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1.5
    file_path: "crates/context-graph-utl/src/config.rs"
    dependencies: []
    acceptance_criteria:
      - "SurpriseConfig struct with 7 fields"
      - "KlConfig nested struct with 3 fields"
      - "Default returns spec-defined values"
      - "Weights sum to 1.0"
      - "Serde serialization works"
    test_file: "crates/context-graph-utl/tests/config_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 10"
      - "SPEC-UTL-005 Section 3.3"

  - id: "M05-T03"
    title: "Define CoherenceConfig for Rolling Window Tracking"
    description: |
      Implement CoherenceConfig struct for coherence tracking configuration.
      Fields: window_size (100), recency_decay (0.98), semantic_weight (0.6),
      structural_weight (0.4), default_coherence_empty (0.5),
      default_coherence_no_concepts (0.4).
      Include contradiction detection fields: contradiction_search_k (20),
      contradiction_similarity_threshold (0.85), max_contradiction_penalty (0.5).
    layer: "foundation"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-utl/src/config.rs"
    dependencies: []
    acceptance_criteria:
      - "CoherenceConfig struct with 9 fields"
      - "semantic_weight + structural_weight = 1.0"
      - "Default returns spec-defined values"
      - "Contradiction fields configured correctly"
    test_file: "crates/context-graph-utl/tests/config_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 10"
      - "SPEC-UTL-005 Section 4.3"

  - id: "M05-T04"
    title: "Define EmotionalConfig for Valence/Arousal Parameters"
    description: |
      Implement EmotionalConfig struct for emotional weight calculation.
      Fields: decay_rate (0.1), baseline_weight (1.0), valence_weight (0.6),
      arousal_weight (0.4), intensity_scale (0.5), exclamation_weight (0.3),
      question_weight (0.2), caps_weight (0.2).
      Output range: weight_min (0.5), weight_max (1.5).
    layer: "foundation"
    priority: "high"
    estimated_hours: 1
    file_path: "crates/context-graph-utl/src/config.rs"
    dependencies: []
    acceptance_criteria:
      - "EmotionalConfig struct with 10 fields"
      - "valence_weight + arousal_weight = 1.0"
      - "Default returns spec-defined values"
      - "Output range [0.5, 1.5] enforced"
    test_file: "crates/context-graph-utl/tests/config_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 10"
      - "SPEC-UTL-005 Section 5.3"

  # ============================================================
  # FOUNDATION: Lifecycle Stage (Marblestone)
  # ============================================================

  - id: "M05-T05"
    title: "Define LifecycleStage Enum (Marblestone)"
    description: |
      Implement LifecycleStage enum with Marblestone-inspired dynamic learning rates.
      Variants: Infancy (0-50 interactions), Growth (50-500), Maturity (500+).
      Include methods: get_lambda_weights(), is_novelty_seeking(), is_coherence_preserving(),
      name(), entropy_trigger(), coherence_trigger().
      REQ-UTL-030 through REQ-UTL-035 compliance.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-utl/src/lifecycle/stage.rs"
    dependencies: []
    acceptance_criteria:
      - "LifecycleStage enum with 3 variants"
      - "Default is Infancy"
      - "get_lambda_weights() returns correct weights per stage"
      - "Infancy: lambda_novelty=0.7, lambda_consolidation=0.3"
      - "Growth: lambda_novelty=0.5, lambda_consolidation=0.5"
      - "Maturity: lambda_novelty=0.3, lambda_consolidation=0.7"
      - "Serde serialization with #[repr(u8)]"
    test_file: "crates/context-graph-utl/tests/lifecycle_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 3"
      - "SPEC-UTL-005 Section 8.2"
      - "REQ-UTL-030, REQ-UTL-031, REQ-UTL-032, REQ-UTL-033"

  - id: "M05-T06"
    title: "Define LifecycleLambdaWeights Struct (Marblestone)"
    description: |
      Implement LifecycleLambdaWeights struct for dynamic learning rate modulation.
      Fields: lambda_novelty (f32), lambda_consolidation (f32).
      Invariant: lambda_novelty + lambda_consolidation = 1.0.
      Methods: new(novelty, consolidation) with validation, apply(delta_s, delta_c),
      is_balanced(), is_novelty_dominant(), is_consolidation_dominant().
      REQ-UTL-034 compliance for weight application.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1.5
    file_path: "crates/context-graph-utl/src/lifecycle/lambda.rs"
    dependencies: []
    acceptance_criteria:
      - "LifecycleLambdaWeights struct with 2 f32 fields"
      - "Default: lambda_novelty=0.5, lambda_consolidation=0.5"
      - "new() validates sum = 1.0, returns Result"
      - "apply(delta_s, delta_c) returns weighted tuple"
      - "Invariant check in constructor"
    test_file: "crates/context-graph-utl/tests/lifecycle_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 3.2"
      - "REQ-UTL-034"

  - id: "M05-T07"
    title: "Define LifecycleConfig and StageConfig Structs"
    description: |
      Implement LifecycleConfig struct for lifecycle state machine configuration.
      Fields: infancy_threshold (50), growth_threshold (500),
      infancy (StageConfig), growth (StageConfig), maturity (StageConfig).
      StageConfig: entropy_trigger, coherence_trigger, min_importance_store, consolidation_threshold.
      Infancy: entropy_trigger=0.9, coherence_trigger=0.2, min_importance=0.1, consolidation=0.3.
      Growth: entropy_trigger=0.7, coherence_trigger=0.4, min_importance=0.3, consolidation=0.5.
      Maturity: entropy_trigger=0.6, coherence_trigger=0.5, min_importance=0.4, consolidation=0.6.
    layer: "foundation"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-utl/src/config.rs"
    dependencies:
      - "M05-T05"
    acceptance_criteria:
      - "LifecycleConfig struct with 5 fields"
      - "StageConfig struct with 4 f32 fields"
      - "Default returns spec-defined thresholds per stage"
      - "Serde serialization works"
    test_file: "crates/context-graph-utl/tests/config_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 10"
      - "SPEC-UTL-005 Section 8.4"

  # ============================================================
  # FOUNDATION: Johari Quadrant Types
  # ============================================================

  - id: "M05-T08"
    title: "Define JohariQuadrant Enum and SuggestedAction"
    description: |
      Implement JohariQuadrant enum for memory classification.
      Variants: Open (low entropy, high coherence), Blind (high entropy, low coherence),
      Hidden (low entropy, low coherence), Unknown (high entropy, high coherence).
      SuggestedAction enum: DirectRecall, TriggerDream, GetNeighborhood, EpistemicAction,
      CritiqueContext, Curate.
      Include methods: name(), is_well_understood(), requires_exploration().
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-utl/src/johari/quadrant.rs"
    dependencies: []
    acceptance_criteria:
      - "JohariQuadrant enum with 4 variants"
      - "Default is Open"
      - "SuggestedAction enum with 6 variants"
      - "#[repr(u8)] for efficient storage"
      - "name() returns string representation"
      - "Serde Serialize/Deserialize implemented"
    test_file: "crates/context-graph-utl/tests/johari_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 4.1"
      - "SPEC-UTL-005 Section 7"

  # ============================================================
  # LOGIC LAYER: Surprise Computation
  # ============================================================

  - id: "M05-T09"
    title: "Implement KL Divergence Computation"
    description: |
      Implement kl_divergence(p, q, epsilon) function for distribution comparison.
      Formula: KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
      Include softmax_normalize(values, temperature) for probability distribution.
      Include cosine_similarity(a, b) for embedding comparison.
      Performance target: <1ms for 1536-dimensional vectors.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-utl/src/surprise/kl_divergence.rs"
    dependencies:
      - "M05-T02"
    acceptance_criteria:
      - "kl_divergence() returns non-negative f32"
      - "softmax_normalize() returns probability distribution summing to 1.0"
      - "cosine_similarity() returns [-1, 1] range"
      - "Numerical stability with epsilon parameter"
      - "Performance: <1ms for 1536D vectors"
      - "Unit tests verify mathematical properties"
    test_file: "crates/context-graph-utl/tests/surprise_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 5"
      - "SPEC-UTL-005 Section 3.2.1"

  - id: "M05-T10"
    title: "Implement Surprise Computation Methods"
    description: |
      Implement compute_surprise_kl(observed, context_embeddings, config) function.
      Algorithm:
      1. Compute centroid of context embeddings
      2. Softmax normalize observed and centroid
      3. Compute KL divergence
      4. Normalize to [0, 1] using max_kl_value
      Include compute_surprise_distance() for cosine-based alternative.
      Performance target: <5ms including centroid computation.
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-utl/src/surprise/kl_divergence.rs"
    dependencies:
      - "M05-T09"
    acceptance_criteria:
      - "compute_surprise_kl() returns delta_s in [0, 1]"
      - "Returns max_surprise_no_context when context is empty"
      - "Falls back to distance method when context < min_context_for_kl"
      - "compute_surprise_distance() uses cosine similarity"
      - "Performance: <5ms for 50 context vectors"
    test_file: "crates/context-graph-utl/tests/surprise_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 5"
      - "SPEC-UTL-005 Section 3.2"

  - id: "M05-T11"
    title: "Implement SurpriseCalculator Struct"
    description: |
      Implement SurpriseCalculator struct for ensemble surprise computation.
      Methods: new(config), compute_surprise_ensemble(observed, context),
      compute_surprise_with_decay(observed, weighted_context).
      Ensemble combines KL and distance methods with configurable weights.
      Include recency decay for context importance.
    layer: "logic"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-utl/src/surprise/calculator.rs"
    dependencies:
      - "M05-T10"
    acceptance_criteria:
      - "SurpriseCalculator struct with SurpriseConfig"
      - "compute_surprise_ensemble() combines KL and distance"
      - "Weights: kl_weight * kl_surprise + distance_weight * dist_surprise"
      - "compute_surprise_with_decay() applies recency weighting"
      - "Result clamped to [0, 1]"
    test_file: "crates/context-graph-utl/tests/surprise_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 5.2"
      - "SPEC-UTL-005 Section 3.2.3"

  # ============================================================
  # LOGIC LAYER: Coherence Tracking
  # ============================================================

  - id: "M05-T12"
    title: "Define CoherenceEntry and Rolling Window"
    description: |
      Implement CoherenceEntry struct for coherence window entries.
      Fields: node_id (Uuid), embedding (Vec<f32>), timestamp (DateTime<Utc>), importance (f32).
      Implement rolling window buffer using VecDeque with max_size constraint.
      Include update() method to add entries with automatic eviction.
    layer: "logic"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-utl/src/coherence/tracker.rs"
    dependencies:
      - "M05-T03"
    acceptance_criteria:
      - "CoherenceEntry struct with 4 fields"
      - "VecDeque-based window with capacity"
      - "update() evicts oldest when at capacity"
      - "window_size() returns current count"
      - "clear() empties the window"
      - "Clone, Debug implemented"
    test_file: "crates/context-graph-utl/tests/coherence_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 6.1"
      - "SPEC-UTL-005 Section 4.2"

  - id: "M05-T13"
    title: "Implement CoherenceTracker with Semantic Coherence"
    description: |
      Implement CoherenceTracker struct with semantic coherence computation.
      Methods: new(config, graph), compute_coherence(embedding, content),
      compute_semantic_coherence(embedding), update(entry).
      Semantic coherence: weighted average similarity to window entries with recency decay.
      Formula: sum(sim_i * decay^i * importance_i) / sum(decay^i * importance_i).
      Performance target: <5ms.
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-utl/src/coherence/tracker.rs"
    dependencies:
      - "M05-T12"
    acceptance_criteria:
      - "CoherenceTracker struct with window, config, optional graph"
      - "compute_coherence() returns delta_c in [0, 1]"
      - "compute_semantic_coherence() applies recency decay"
      - "Returns default_coherence_empty when window is empty"
      - "Performance: <5ms for 100 window entries"
    test_file: "crates/context-graph-utl/tests/coherence_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 6"
      - "SPEC-UTL-005 Section 4"

  - id: "M05-T14"
    title: "Implement Structural Coherence and Contradiction Detection"
    description: |
      Extend CoherenceTracker with structural coherence and contradiction detection.
      compute_structural_coherence(): graph connectivity-based coherence.
      compute_contradiction_penalty(): detect conflicts with existing knowledge.
      Final coherence: (semantic_weight * semantic + structural_weight * structural) * (1 - contradiction_penalty).
      Note: Requires KnowledgeGraph integration (can be stubbed initially).
    layer: "logic"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-utl/src/coherence/tracker.rs"
    dependencies:
      - "M05-T13"
    acceptance_criteria:
      - "compute_structural_coherence() uses graph if available"
      - "compute_contradiction_penalty() returns [0, max_penalty]"
      - "Contradiction penalty reduces overall coherence"
      - "Graceful degradation when graph unavailable"
      - "Stub implementation returns defaults"
    test_file: "crates/context-graph-utl/tests/coherence_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 6"
      - "SPEC-UTL-005 Section 4.2"

  # ============================================================
  # LOGIC LAYER: Emotional Weighting
  # ============================================================

  - id: "M05-T15"
    title: "Define EmotionalState Struct with Decay"
    description: |
      Implement EmotionalState struct for tracking valence/arousal.
      Fields: valence (f32 in [-1, 1]), arousal (f32 in [0, 1]), timestamp (DateTime<Utc>).
      Valence: positive (+1) to negative (-1).
      Arousal: calm (0) to excited (1).
      Include Default impl with neutral state (0, 0).
    layer: "logic"
    priority: "high"
    estimated_hours: 1
    file_path: "crates/context-graph-utl/src/emotional/calculator.rs"
    dependencies:
      - "M05-T04"
    acceptance_criteria:
      - "EmotionalState struct with 3 fields"
      - "Default: valence=0.0, arousal=0.0"
      - "Clone, Debug, Default implemented"
      - "Serde serialization works"
    test_file: "crates/context-graph-utl/tests/emotional_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 9"
      - "SPEC-UTL-005 Section 5.2"

  - id: "M05-T16"
    title: "Implement EmotionalWeightCalculator"
    description: |
      Implement EmotionalWeightCalculator struct for computing w_e.
      Methods: new(config), compute_weight(content), get_state().
      Algorithm:
      1. Extract emotion from content (lexicon sentiment + arousal heuristics)
      2. Update state with exponential decay blending
      3. Convert state to weight in [0.5, 1.5]
      Weight formula: baseline + intensity * intensity_scale, where
      intensity = valence_weight * |valence| + arousal_weight * arousal.
      Performance target: <1ms.
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-utl/src/emotional/calculator.rs"
    dependencies:
      - "M05-T15"
    acceptance_criteria:
      - "EmotionalWeightCalculator with config and state"
      - "compute_weight() returns w_e in [0.5, 1.5]"
      - "Lexicon sentiment analysis for valence"
      - "Punctuation/caps heuristics for arousal"
      - "Exponential decay blending for state update"
      - "Performance: <1ms"
    test_file: "crates/context-graph-utl/tests/emotional_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 9"
      - "SPEC-UTL-005 Section 5"

  # ============================================================
  # LOGIC LAYER: Phase Oscillator
  # ============================================================

  - id: "M05-T17"
    title: "Implement PhaseOscillator for Consolidation Timing"
    description: |
      Implement PhaseOscillator struct for theta-inspired memory phase tracking.
      Fields: phi (f32 in [0, PI]), frequency (f32), last_update (Instant), modulation (f32).
      ConsolidationPhase enum: Encoding (phi near 0), Transition, Consolidation (phi near PI).
      Methods: new(config), get_phi(), get_phase(), set_modulation(f32),
      reset_to_encoding(), force_consolidation().
      Phase advances based on elapsed time and modulation factor.
    layer: "logic"
    priority: "high"
    estimated_hours: 2.5
    file_path: "crates/context-graph-utl/src/phase/oscillator.rs"
    dependencies: []
    acceptance_criteria:
      - "PhaseOscillator with phi in [0, PI]"
      - "ConsolidationPhase enum with 3 variants"
      - "get_phi() updates phase based on elapsed time"
      - "Phase oscillates using |sin| * PI formula"
      - "set_modulation() clamps to [modulation_min, modulation_max]"
      - "force_consolidation() sets phi to 0.9 * PI"
      - "Performance: <10us per update"
    test_file: "crates/context-graph-utl/tests/phase_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 8"
      - "SPEC-UTL-005 Section 6"

  # ============================================================
  # SURFACE LAYER: Johari Classifier
  # ============================================================

  - id: "M05-T18"
    title: "Implement JohariClassifier with Retrieval Strategies"
    description: |
      Implement JohariClassifier struct for quadrant classification.
      Methods: new(config), classify(delta_s, delta_c), suggested_action(quadrant),
      retrieval_strategy(quadrant), detect_transition(old_s, old_c, new_s, new_c).
      Classification: low_entropy = delta_s < threshold, high_coherence = delta_c > threshold.
      RetrievalStrategy: search_depth, include_neighbors, confidence_threshold, max_results.
      Per-quadrant defaults as specified in functional spec.
    layer: "surface"
    priority: "critical"
    estimated_hours: 2.5
    file_path: "crates/context-graph-utl/src/johari/classifier.rs"
    dependencies:
      - "M05-T08"
    acceptance_criteria:
      - "JohariClassifier with entropy/coherence thresholds"
      - "classify() returns correct quadrant based on thresholds"
      - "suggested_action() maps quadrant to action"
      - "retrieval_strategy() returns quadrant-specific settings"
      - "Open: depth=1, max_results=5"
      - "Unknown: depth=4, max_results=30"
      - "detect_transition() returns Some when quadrant changes"
    test_file: "crates/context-graph-utl/tests/johari_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 4.2"
      - "SPEC-UTL-005 Section 7"

  # ============================================================
  # SURFACE LAYER: Lifecycle Manager
  # ============================================================

  - id: "M05-T19"
    title: "Implement LifecycleManager State Machine"
    description: |
      Implement LifecycleManager struct for lifecycle state transitions.
      Fields: interaction_count (u64), current_stage (LifecycleStage),
      lambda_weights (LifecycleLambdaWeights), config (LifecycleConfig).
      Methods: new(config), record_interaction(), current_stage(), get_lambda_weights(),
      apply_lambda_weights(delta_s, delta_c), get_thresholds(), storage_stance(),
      should_store(magnitude, importance), restore(count).
      StorageStance enum: CaptureHeavy, Balanced, CurationHeavy.
      REQ-UTL-035: Transitions preserve accumulated knowledge coherence.
    layer: "surface"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-utl/src/lifecycle/manager.rs"
    dependencies:
      - "M05-T05"
      - "M05-T06"
      - "M05-T07"
    acceptance_criteria:
      - "LifecycleManager with all specified fields"
      - "record_interaction() increments count and checks transition"
      - "Transitions at 50 (Infancy->Growth) and 500 (Growth->Maturity)"
      - "Lambda weights update on transition"
      - "get_thresholds() returns stage-specific values"
      - "should_store() varies by storage stance"
      - "Logging on stage transitions"
    test_file: "crates/context-graph-utl/tests/lifecycle_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 3.3"
      - "SPEC-UTL-005 Section 8.3"
      - "REQ-UTL-035"

  # ============================================================
  # SURFACE LAYER: Core UTL Computation
  # ============================================================

  - id: "M05-T20"
    title: "Implement Core UTL Learning Magnitude Function"
    description: |
      Implement compute_learning_magnitude(delta_s, delta_c, w_e, phi) function.
      Formula: L = sigmoid((delta_s * delta_c) * w_e * cos(phi) * LEARNING_SCALE_FACTOR)
      Input clamping: delta_s/delta_c in [0,1], w_e in [0.5,1.5], phi in [0,PI].
      Sigmoid: 1 / (1 + exp(-x)).
      LEARNING_SCALE_FACTOR = 2.0.
      Include compute_learning_magnitude_weighted() for Marblestone integration.
      Performance target: <100us.
      Anti-pattern: Never return NaN or Infinity.
    layer: "surface"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-utl/src/lib.rs"
    dependencies: []
    acceptance_criteria:
      - "compute_learning_magnitude() returns L in [0, 1]"
      - "All inputs clamped before computation"
      - "Result is never NaN or Infinity"
      - "sigmoid() helper function works correctly"
      - "compute_learning_magnitude_weighted() applies lambda weights"
      - "Performance: <100us"
      - "#[inline] annotation for hot path"
    test_file: "crates/context-graph-utl/tests/utl_core_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 2.1"
      - "SPEC-UTL-005 Section 2"
      - "REQ-UTL-001"

  - id: "M05-T21"
    title: "Define LearningSignal and UtlState Structs"
    description: |
      Implement LearningSignal struct for complete UTL output.
      Fields: magnitude (f32), delta_s (f32), delta_c (f32), w_e (f32), phi (f32),
      lambda_weights (Option<LifecycleLambdaWeights>), quadrant (JohariQuadrant),
      suggested_action (SuggestedAction), should_consolidate (bool), should_store (bool),
      timestamp (DateTime<Utc>), latency_us (u64).
      UtlState struct for node storage: delta_s, delta_c, w_e, phi, learning_magnitude,
      quadrant, last_computed.
      Include validate() method checking for NaN/Infinity.
    layer: "surface"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-utl/src/lib.rs"
    dependencies:
      - "M05-T06"
      - "M05-T08"
    acceptance_criteria:
      - "LearningSignal struct with all 12 fields"
      - "UtlState struct with 7 fields for node storage"
      - "validate() returns UtlError if magnitude is NaN/Infinity"
      - "Clone, Debug, Serialize, Deserialize implemented"
      - "Latency tracking in microseconds"
    test_file: "crates/context-graph-utl/tests/utl_core_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 2.2"
      - "SPEC-UTL-005 Section 9.1"

  - id: "M05-T22"
    title: "Implement UtlProcessor Main Orchestrator"
    description: |
      Implement UtlProcessor struct integrating all UTL components.
      Fields: surprise_calculator, coherence_tracker, emotional_calculator,
      phase_oscillator, johari_classifier, lifecycle_manager, config, metrics.
      Methods: new(config), compute_learning(input, embedding, context),
      get_status(), update_coherence_window(entry), set_phase_modulation(f32),
      trigger_consolidation().
      SessionContext struct: session_id, recent_embeddings, interaction_count.
      Performance target: <10ms for full computation.
    layer: "surface"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-utl/src/processor.rs"
    dependencies:
      - "M05-T11"
      - "M05-T13"
      - "M05-T16"
      - "M05-T17"
      - "M05-T18"
      - "M05-T19"
      - "M05-T20"
      - "M05-T21"
    acceptance_criteria:
      - "UtlProcessor orchestrates all 6 components"
      - "compute_learning() executes full pipeline"
      - "Records interaction for lifecycle"
      - "Applies Marblestone lambda weights"
      - "Classifies Johari quadrant"
      - "Determines consolidation and storage decisions"
      - "Validates output for NaN/Infinity"
      - "Updates metrics asynchronously"
      - "Performance: <10ms"
    test_file: "crates/context-graph-utl/tests/processor_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 7"
      - "SPEC-UTL-005 Section 9"

  # ============================================================
  # SURFACE LAYER: Error Handling
  # ============================================================

  - id: "M05-T23"
    title: "Define UtlError Enum"
    description: |
      Implement comprehensive UtlError enum for UTL operations.
      Variants: InvalidComputation {delta_s, delta_c, w_e, phi, reason},
      InvalidLambdaWeights {novelty, consolidation, reason},
      MissingContext, DimensionMismatch {expected, actual},
      GraphError(String), ConfigError(String).
      Use thiserror for derivation.
      All variants have descriptive error messages.
    layer: "surface"
    priority: "high"
    estimated_hours: 1
    file_path: "crates/context-graph-utl/src/error.rs"
    dependencies: []
    acceptance_criteria:
      - "UtlError enum with 6 variants"
      - "All variants have #[error()] messages"
      - "InvalidComputation includes all component values"
      - "Error is Send + Sync"
      - "thiserror derivation works"
    test_file: "crates/context-graph-utl/tests/error_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 11"
      - "SPEC-UTL-005 Section 9.1"

  # ============================================================
  # SURFACE LAYER: Metrics and MCP Integration
  # ============================================================

  - id: "M05-T24"
    title: "Implement UtlMetrics and UtlStatus Structs"
    description: |
      Implement UtlMetrics struct for monitoring.
      Fields: computation_count (u64), avg_learning_magnitude (f32), avg_delta_s (f32),
      avg_delta_c (f32), quadrant_distribution (QuadrantDistribution), lifecycle_stage,
      lambda_weights, avg_latency_us (f64), p99_latency_us (u64).
      QuadrantDistribution: open (u32), blind (u32), hidden (u32), unknown (u32).
      Include percentages() method.
      UtlStatus for get_status(): lifecycle_stage, interaction_count, current_thresholds,
      lambda_weights, phase_angle, consolidation_phase, metrics.
    layer: "surface"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-utl/src/metrics.rs"
    dependencies:
      - "M05-T05"
      - "M05-T06"
      - "M05-T08"
    acceptance_criteria:
      - "UtlMetrics struct with all 9 fields"
      - "QuadrantDistribution with 4 counters"
      - "percentages() returns array of 4 f32"
      - "UtlStatus struct for status reporting"
      - "Default impl for metrics initialization"
      - "Serde serialization works"
    test_file: "crates/context-graph-utl/tests/metrics_tests.rs"
    spec_refs:
      - "TECH-UTL-005 Section 2.2"
      - "SPEC-UTL-005 Section 10"

  - id: "M05-T25"
    title: "Create Module Integration Tests and Benchmarks"
    description: |
      Implement comprehensive integration tests for Module 5:
      - Full UTL pipeline with real data (no mocks)
      - Learning signal correlation with importance (r > 0.7)
      - Lifecycle stage transitions at correct thresholds
      - Lambda weight application verification
      - Johari quadrant classification correctness
      - NaN/Infinity prevention tests
      - Emotional weight bounds [0.5, 1.5]
      Benchmark tests:
      - compute_learning_magnitude: <100us
      - Full UTL computation: <10ms
      - Surprise calculation: <5ms
      - Emotional weight: <1ms
    layer: "surface"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-utl/tests/integration_tests.rs"
    dependencies:
      - "M05-T22"
      - "M05-T24"
    acceptance_criteria:
      - "All unit tests pass"
      - "Integration test covers full pipeline"
      - "Correlation test validates r > 0.7"
      - "Lifecycle transitions verified"
      - "Lambda weights correct per stage"
      - "Johari classification matches spec table"
      - "No NaN/Infinity in any test"
      - "Benchmark meets performance targets"
      - "90%+ code coverage"
    test_file: "crates/context-graph-utl/tests/integration_tests.rs"
    bench_file: "crates/context-graph-utl/benches/utl_bench.rs"
    spec_refs:
      - "TECH-UTL-005 Section 12"
      - "SPEC-UTL-005 Section 12, 13"
```

---

## Dependency Graph

```
M05-T01 (UtlConfig) ─────────────────────────────────────────────────────────────┐
M05-T02 (SurpriseConfig) ──► M05-T09 (KL Divergence) ──► M05-T10 (Surprise Methods) ──► M05-T11 (SurpriseCalculator)
M05-T03 (CoherenceConfig) ──► M05-T12 (CoherenceEntry) ──► M05-T13 (CoherenceTracker) ──► M05-T14 (Structural/Contradiction)
M05-T04 (EmotionalConfig) ──► M05-T15 (EmotionalState) ──► M05-T16 (EmotionalWeightCalculator)
                                                                                  │
M05-T05 (LifecycleStage) ──┬──► M05-T07 (LifecycleConfig) ──────────────────────┐ │
M05-T06 (LambdaWeights) ───┘                                                     │ │
                                                                                  │ │
M05-T05 + M05-T06 + M05-T07 ──► M05-T19 (LifecycleManager) ─────────────────────┼─┤
                                                                                  │ │
M05-T08 (JohariQuadrant) ──► M05-T18 (JohariClassifier) ────────────────────────┼─┤
                                                                                  │ │
M05-T17 (PhaseOscillator) ──────────────────────────────────────────────────────┼─┤
                                                                                  │ │
M05-T20 (Core UTL Function) ────────────────────────────────────────────────────┼─┤
                                                                                  │ │
M05-T21 (LearningSignal) ──────────────────────────────────────────────────────┼─┤
                                                                                  │ │
M05-T23 (UtlError) ─────────────────────────────────────────────────────────────┼─┤
                                                                                  │ │
M05-T11 + M05-T13 + M05-T16 + M05-T17 + M05-T18 + M05-T19 + M05-T20 + M05-T21 ──┴─┴─► M05-T22 (UtlProcessor)
                                                                                          │
M05-T05 + M05-T06 + M05-T08 ──► M05-T24 (UtlMetrics) ─────────────────────────────────────┤
                                                                                          │
M05-T22 + M05-T24 ──► M05-T25 (Integration Tests) ────────────────────────────────────────┘
```

---

## Implementation Order (Recommended)

### Week 1: Foundation Types
1. M05-T01: UtlConfig and UtlThresholds
2. M05-T02: SurpriseConfig for KL divergence
3. M05-T03: CoherenceConfig for rolling window
4. M05-T04: EmotionalConfig for valence/arousal
5. M05-T05: LifecycleStage enum (Marblestone)
6. M05-T06: LifecycleLambdaWeights struct
7. M05-T07: LifecycleConfig and StageConfig
8. M05-T08: JohariQuadrant and SuggestedAction

### Week 2: Component Logic
9. M05-T09: KL Divergence computation
10. M05-T10: Surprise computation methods
11. M05-T11: SurpriseCalculator struct
12. M05-T12: CoherenceEntry and window
13. M05-T13: CoherenceTracker with semantic
14. M05-T14: Structural coherence and contradictions
15. M05-T15: EmotionalState struct
16. M05-T16: EmotionalWeightCalculator
17. M05-T17: PhaseOscillator

### Week 3: Surface Layer
18. M05-T18: JohariClassifier
19. M05-T19: LifecycleManager state machine
20. M05-T20: Core UTL learning magnitude
21. M05-T21: LearningSignal and UtlState
22. M05-T22: UtlProcessor orchestrator
23. M05-T23: UtlError enum

### Week 4: Integration and Testing
24. M05-T24: UtlMetrics and UtlStatus
25. M05-T25: Integration tests and benchmarks

---

## Quality Gates

| Gate | Criteria | Required For |
|------|----------|--------------|
| Foundation Complete | M05-T01 through M05-T08 pass all tests | Week 2 start |
| Logic Complete | M05-T09 through M05-T17 pass all tests | Week 3 start |
| Surface Complete | M05-T18 through M05-T24 pass all tests | Week 4 start |
| Module Complete | All 25 tasks complete, benchmarks pass | Module 6 start |

---

## Performance Targets Summary

| Operation | Target | P99 Target | Conditions |
|-----------|--------|------------|------------|
| `compute_learning_magnitude` | <100us | <500us | Core equation only |
| Full UTL computation | <10ms | <50ms | All components |
| Surprise (KL) | <5ms | <20ms | 1536D, 50 context |
| Surprise (distance) | <1ms | <5ms | 1536D, 50 context |
| Coherence computation | <5ms | <25ms | 100 window entries |
| Emotional weight | <1ms | <5ms | Text analysis |
| Phase update | <10us | <50us | Simple math |
| Johari classification | <1us | <5us | Two comparisons |

---

## Marblestone Integration Summary

Tasks with Marblestone features:
- **M05-T05**: LifecycleStage enum (Infancy/Growth/Maturity)
- **M05-T06**: LifecycleLambdaWeights (lambda_novelty/lambda_consolidation)
- **M05-T07**: LifecycleConfig with per-stage thresholds
- **M05-T19**: LifecycleManager state machine with transitions
- **M05-T20**: compute_learning_magnitude_weighted() with lambda application
- **M05-T22**: UtlProcessor applies lambda weights to all computations

---

## Critical Constraints

**NO NaN/Infinity**: All UTL computations MUST clamp inputs and validate outputs.
- Inputs clamped to valid ranges before computation
- Results validated before return
- InvalidComputation error raised for NaN/Infinity

**Lambda Weight Invariant**: lambda_novelty + lambda_consolidation = 1.0.
- Validated in LifecycleLambdaWeights::new()
- Enforced at stage transitions

**Lifecycle Transitions**:
- Infancy -> Growth at 50 interactions
- Growth -> Maturity at 500 interactions
- Lambda weights update atomically with stage

---

## File Structure

```
crates/context-graph-utl/
├── src/
│   ├── lib.rs                    # Public API, compute_learning_magnitude
│   ├── config.rs                 # All configuration structs
│   ├── error.rs                  # UtlError enum
│   ├── processor.rs              # UtlProcessor main orchestrator
│   ├── metrics.rs                # UtlMetrics, UtlStatus
│   │
│   ├── surprise/
│   │   ├── mod.rs                # Module exports
│   │   ├── kl_divergence.rs      # KL divergence, cosine similarity
│   │   └── calculator.rs         # SurpriseCalculator
│   │
│   ├── coherence/
│   │   ├── mod.rs                # Module exports
│   │   └── tracker.rs            # CoherenceTracker, CoherenceEntry
│   │
│   ├── emotional/
│   │   ├── mod.rs                # Module exports
│   │   └── calculator.rs         # EmotionalWeightCalculator, EmotionalState
│   │
│   ├── phase/
│   │   ├── mod.rs                # Module exports
│   │   └── oscillator.rs         # PhaseOscillator, ConsolidationPhase
│   │
│   ├── johari/
│   │   ├── mod.rs                # Module exports
│   │   ├── quadrant.rs           # JohariQuadrant, SuggestedAction
│   │   └── classifier.rs         # JohariClassifier, RetrievalStrategy
│   │
│   └── lifecycle/
│       ├── mod.rs                # Module exports
│       ├── stage.rs              # LifecycleStage enum
│       ├── lambda.rs             # LifecycleLambdaWeights
│       └── manager.rs            # LifecycleManager
│
├── tests/
│   ├── config_tests.rs           # Configuration tests
│   ├── surprise_tests.rs         # Surprise computation tests
│   ├── coherence_tests.rs        # Coherence tracking tests
│   ├── emotional_tests.rs        # Emotional weight tests
│   ├── phase_tests.rs            # Phase oscillator tests
│   ├── johari_tests.rs           # Johari classification tests
│   ├── lifecycle_tests.rs        # Lifecycle manager tests
│   ├── utl_core_tests.rs         # Core UTL function tests
│   ├── processor_tests.rs        # UtlProcessor tests
│   ├── metrics_tests.rs          # Metrics tests
│   ├── error_tests.rs            # Error handling tests
│   └── integration_tests.rs      # Full integration tests
│
├── benches/
│   └── utl_bench.rs              # Performance benchmarks
│
└── Cargo.toml
```

---

*Generated: 2025-12-31*
*Module: 05 - UTL Integration*
*Version: 1.0.0*
*Total Tasks: 25*
