# Module 6: Bio-Nervous System - Atomic Tasks

```yaml
metadata:
  module_id: "module-06"
  module_name: "Bio-Nervous System"
  version: "1.0.0"
  phase: 5
  total_tasks: 24
  approach: "inside-out-bottom-up"
  created: "2025-12-31"
  dependencies:
    - module-05-utl-integration
    - module-04-knowledge-graph
    - module-03-embedding-pipeline
    - module-02-core-infrastructure
  estimated_duration: "4 weeks"
  spec_refs:
    - SPEC-BIONV-006 (Functional)
    - TECH-BIONV-006 (Technical)
```

---

## Task Overview

This module implements a 5-layer hierarchical processing architecture inspired by biological neural systems. Each layer has specific latency budgets and responsibilities, creating a processing pipeline from raw sensory input to coherent understanding.

### Layer Latency Budgets

| Layer | ID | Budget | Purpose |
|-------|-----|--------|---------|
| L1 Sensing | Sensing | <5ms | Input processing, PII scrubbing, embedding |
| L2 Reflex | Reflex | <100us | Fast pattern-matched cache responses |
| L3 Memory | Memory | <1ms | Associative storage/retrieval (Modern Hopfield) |
| L4 Learning | Learning | <10ms | UTL-driven weight optimization |
| L5 Coherence | Coherence | <10ms | Global state synchronization, predictive coding |

### Task Organization

1. **Foundation Layer** (Tasks 1-8): Core message protocol, NervousLayer trait, configuration types
2. **Logic Layer** (Tasks 9-16): Layer implementations (L1-L5 core logic)
3. **Surface Layer** (Tasks 17-24): Integration, Marblestone verification, system orchestration

---

## Foundation Layer: Protocol & Core Types

```yaml
tasks:
  # ============================================================
  # FOUNDATION: Message Protocol
  # ============================================================

  - id: "M06-T01"
    title: "Define NervousLayerId Enum"
    description: |
      Implement NervousLayerId enum identifying all 5 nervous system layers.
      Variants: Sensing (1), Reflex (2), Memory (3), Learning (4), Coherence (5).
      Include latency_budget() const fn returning Duration for each layer.
      Include name() method returning static string.
      Use #[repr(u8)] for efficient serialization.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1
    file_path: "crates/context-graph-nervous/src/message/layer_id.rs"
    dependencies: []
    acceptance_criteria:
      - "NervousLayerId enum compiles with 5 variants"
      - "latency_budget() returns: Sensing=5ms, Reflex=100us, Memory=1ms, Learning=10ms, Coherence=10ms"
      - "name() returns human-readable layer names"
      - "Clone, Copy, Debug, PartialEq, Eq, Hash derived"
      - "#[repr(u8)] applied for serialization"
    test_file: "crates/context-graph-nervous/tests/layer_id_tests.rs"
    spec_refs:
      - "TECH-BIONV-006 Section 2.1"
      - "SPEC-BIONV-006 Section 2.2"

  - id: "M06-T02"
    title: "Define MessagePriority Enum"
    description: |
      Implement MessagePriority enum for inter-layer message scheduling.
      Variants: Low (0), Normal (1), High (2), Critical (3), Emergency (4).
      Use #[repr(u8)] with ordered discriminants for comparison.
      Implement PartialOrd, Ord for priority comparison.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1
    file_path: "crates/context-graph-nervous/src/message/priority.rs"
    dependencies: []
    acceptance_criteria:
      - "MessagePriority enum with 5 variants"
      - "Ord comparison works: Emergency > Critical > High > Normal > Low"
      - "#[repr(u8)] applied"
      - "Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord derived"
    test_file: "crates/context-graph-nervous/tests/priority_tests.rs"
    spec_refs:
      - "TECH-BIONV-006 Section 2.1"
      - "REQ-BIONV-037"

  - id: "M06-T03"
    title: "Define MessagePayload Enum"
    description: |
      Implement MessagePayload enum for layer-specific data transport.
      Variants: RawInput(RawInputPayload), Embedding(EmbeddingPayload),
      CacheQuery(CacheQueryPayload), MemoryQuery(MemoryQueryPayload),
      LearningSignal(LearningSignalPayload), CoherenceUpdate(CoherenceUpdatePayload),
      Prediction(PredictionPayload), PredictionError(PredictionErrorPayload).
      Each payload struct defined with essential fields per spec.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-nervous/src/message/payload.rs"
    dependencies: []
    acceptance_criteria:
      - "MessagePayload enum with 8 variants"
      - "EmbeddingPayload has: embedding (Vector1536), content (String), delta_s (f32), latency_us (u64)"
      - "LearningSignalPayload has: delta_s, delta_c, w_e, phi, learning_score (all f32)"
      - "All payloads implement Clone, Debug"
      - "Serde Serialize/Deserialize implemented"
    test_file: "crates/context-graph-nervous/tests/payload_tests.rs"
    spec_refs:
      - "TECH-BIONV-006 Section 2.2"
      - "SPEC-BIONV-006 Section 2.2"

  - id: "M06-T04"
    title: "Define LayerMessage Struct"
    description: |
      Implement LayerMessage struct for inter-layer communication.
      Fields: id (Uuid), source (NervousLayerId), target (NervousLayerId),
      payload (MessagePayload), priority (MessagePriority), deadline (Instant),
      created_at (Instant), correlation_id (Option<Uuid>), trace_context (TraceContext).
      Include new(), is_expired(), respond() methods.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-nervous/src/message/layer_message.rs"
    dependencies:
      - "M06-T01"
      - "M06-T02"
      - "M06-T03"
    acceptance_criteria:
      - "LayerMessage struct with 9 fields"
      - "new() creates message with auto-generated Uuid and deadline from budget"
      - "is_expired() returns true if Instant::now() > deadline"
      - "respond() creates response message with swapped source/target and correlation_id"
      - "Clone, Debug implemented"
    test_file: "crates/context-graph-nervous/tests/message_tests.rs"
    spec_refs:
      - "TECH-BIONV-006 Section 2.1"
      - "REQ-BIONV-036 through REQ-BIONV-039"

  # ============================================================
  # FOUNDATION: NervousLayer Trait
  # ============================================================

  - id: "M06-T05"
    title: "Define NervousLayer Base Trait"
    description: |
      Implement NervousLayer trait as base interface for all 5 layers.
      Methods: id() -> NervousLayerId, name() -> &'static str,
      latency_budget() -> Duration, is_healthy() -> bool,
      health_report() -> HealthReport, handle_timeout(&LayerMessage) -> TimeoutAction,
      reset() -> async, metrics() -> LayerMetrics,
      process_message(LayerMessage) -> Result<LayerMessage, LayerError>.
      Use #[async_trait] for async method support.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-nervous/src/layer/trait.rs"
    dependencies:
      - "M06-T04"
    acceptance_criteria:
      - "NervousLayer trait with 9 methods"
      - "Trait is Send + Sync bound"
      - "Default implementations for name() and latency_budget()"
      - "HealthReport, TimeoutAction, LayerMetrics structs defined"
      - "LayerError enum covers all error conditions"
    test_file: "crates/context-graph-nervous/tests/trait_tests.rs"
    spec_refs:
      - "TECH-BIONV-006 Section 3"
      - "SPEC-BIONV-006 Section 4"

  - id: "M06-T06"
    title: "Define HealthReport and HealthStatus Types"
    description: |
      Implement HealthReport struct for layer health monitoring.
      Fields: layer (NervousLayerId), status (HealthStatus), last_success (Option<Instant>),
      recent_errors (u64), latency_p95 (Duration), budget_compliance (f32).
      HealthStatus enum: Healthy, Degraded { reason: String }, Unhealthy { reason: String }.
      Include is_operational() method returning true for Healthy/Degraded.
    layer: "foundation"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-nervous/src/layer/health.rs"
    dependencies:
      - "M06-T01"
    acceptance_criteria:
      - "HealthReport struct with 6 fields"
      - "HealthStatus enum with 3 variants"
      - "is_operational() returns true for Healthy and Degraded"
      - "budget_compliance is percentage in [0.0, 1.0]"
      - "Clone, Debug implemented"
    test_file: "crates/context-graph-nervous/tests/health_tests.rs"
    spec_refs:
      - "TECH-BIONV-006 Section 3"
      - "REQ-BIONV-043"

  - id: "M06-T07"
    title: "Define TimeoutAction and LayerMetrics Types"
    description: |
      Implement TimeoutAction enum for timeout handling strategies.
      Variants: Partial(Box<dyn Any + Send>), Retry { reduced_scope: bool }, Skip, Fail(String).
      Implement LayerMetrics struct for performance tracking.
      Fields: requests (u64), successes (u64), failures (u64), timeouts (u64),
      latency_histogram (LatencyHistogram), throughput (f32).
    layer: "foundation"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-nervous/src/layer/metrics.rs"
    dependencies: []
    acceptance_criteria:
      - "TimeoutAction enum with 4 variants"
      - "LayerMetrics struct with 6 fields"
      - "LatencyHistogram tracks percentiles (p50, p90, p95, p99)"
      - "throughput in requests/second"
      - "All types are Clone, Debug"
    test_file: "crates/context-graph-nervous/tests/metrics_tests.rs"
    spec_refs:
      - "TECH-BIONV-006 Section 3"
      - "REQ-BIONV-040"

  - id: "M06-T08"
    title: "Define LayerError Enum"
    description: |
      Implement comprehensive LayerError enum for nervous system errors.
      Variants: SensingError(SensingErrorKind), ReflexError(ReflexErrorKind),
      MemoryError(MemoryErrorKind), LearningError(LearningErrorKind),
      CoherenceError(CoherenceErrorKind), Timeout { layer: NervousLayerId, elapsed: Duration },
      MessageExpired, InvalidPayload, LayerUnavailable, ConfigError.
      Layer-specific error kinds defined in submodules.
      Use thiserror for derivation.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-nervous/src/error.rs"
    dependencies:
      - "M06-T01"
    acceptance_criteria:
      - "LayerError enum with 10+ variants"
      - "All variants have descriptive #[error()] messages"
      - "SensingErrorKind includes: PIIDetected, AdversarialDetected, TokenizationFailed, EmptyInput"
      - "MemoryErrorKind includes: PatternStoreFailed, RetrievalFailed, CapacityExceeded"
      - "Error is Send + Sync"
    test_file: "crates/context-graph-nervous/tests/error_tests.rs"
    spec_refs:
      - "TECH-BIONV-006 Section 9"

  # ============================================================
  # LOGIC LAYER: L1 Sensing Implementation
  # ============================================================

  - id: "M06-T09"
    title: "Implement PIIScrubber for L1 Sensing"
    description: |
      Implement PIIScrubber struct for PII pattern detection and scrubbing.
      Fields: patterns (Vec<CompiledPattern>), ner_model (Option<NERModel>), stats (PIIScrubberStats).
      Required patterns: API keys, passwords, bearer tokens, SSN, credit cards, email, phone.
      Method: scrub(&str) -> PIIScrubResult with latency <1ms for pattern matching.
      Include PIICategory enum and replacement templates.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-nervous/src/sensing/pii_scrubber.rs"
    dependencies:
      - "M06-T08"
    acceptance_criteria:
      - "PIIScrubber struct with pattern matching"
      - "scrub() replaces PII with [REDACTED:TYPE] tokens"
      - "Pattern matching completes in <1ms for typical input"
      - "PIICategory enum with 8+ variants"
      - "PIIScrubResult includes: content, scrubbed_count, categories_found, latency"
      - "All regex patterns from SPEC-BIONV-006 Section 3.1.2 implemented"
    test_file: "crates/context-graph-nervous/tests/pii_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.1"
      - "REQ-BIONV-002"

  - id: "M06-T10"
    title: "Implement AdversarialDetector for L1 Sensing"
    description: |
      Implement AdversarialDetector for detecting adversarial inputs.
      Fields: anomaly_threshold (f32, default 3.0), alignment_threshold (f32, default 0.4),
      injection_patterns (Vec<Regex>).
      Methods: check(content, embedding) -> AdversarialCheckResult,
      check_injection_patterns(content) -> InjectionCheckResult.
      Required injection patterns: "ignore previous instructions", "you are now", "new instructions:", etc.
    layer: "logic"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-nervous/src/sensing/adversarial.rs"
    dependencies:
      - "M06-T08"
    acceptance_criteria:
      - "AdversarialDetector struct with 3 configuration fields"
      - "check() analyzes content and embedding for anomalies"
      - "check_injection_patterns() detects prompt injection attempts"
      - "All patterns from SPEC-BIONV-006 Section 3.1.2 implemented"
      - ">95% detection accuracy for known injection patterns"
      - "AdversarialCheckResult includes: is_blocked, confidence, detected_patterns"
    test_file: "crates/context-graph-nervous/tests/adversarial_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.1"
      - "REQ-BIONV-003"

  - id: "M06-T11"
    title: "Implement SensingLayer (L1 Complete)"
    description: |
      Implement SensingLayer struct implementing NervousLayer trait.
      Components: pii_scrubber (PIIScrubber), tokenizer (Tokenizer),
      adversarial (AdversarialDetector), embedding_pipeline (Arc<dyn EmbeddingProvider>),
      baseline (RwLock<NoveltyBaseline>).
      Pipeline: PII Scrub -> Adversarial Check -> Tokenize -> Embed -> Delta-S.
      Method: process(RawInput, &ProcessingContext) -> Result<SensingOutput, SensingError>.
      Performance: <5ms P95 latency, 10K inputs/sec throughput.
    layer: "logic"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-nervous/src/sensing/layer.rs"
    dependencies:
      - "M06-T05"
      - "M06-T09"
      - "M06-T10"
    acceptance_criteria:
      - "SensingLayer implements NervousLayer trait"
      - "process() completes full pipeline in <5ms for 95% of inputs"
      - "compute_delta_s() calculates novelty score using baseline centroid"
      - "SensingOutput includes: content, embedding, delta_s, pii_report, adversarial_check, latency"
      - "Throughput: 10K inputs/sec sustained"
      - "tokio::time::timeout enforces 5ms budget"
    test_file: "crates/context-graph-nervous/tests/sensing_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.1"
      - "REQ-BIONV-001 through REQ-BIONV-008"

  # ============================================================
  # LOGIC LAYER: L2 Reflex Implementation
  # ============================================================

  - id: "M06-T12"
    title: "Implement HopfieldQueryCache for L2 Reflex"
    description: |
      Implement HopfieldQueryCache for ultra-fast pattern-matched responses.
      Fields: cache (DashMap<u64, CacheEntry>), hopfield (ModernHopfieldNetwork),
      stats (CacheStats), config (CacheConfig).
      CacheConfig: max_entries (100000), confidence_threshold (0.95), entry_ttl, eviction_policy (LRU).
      Methods: query(embedding) -> Option<ReflexResponse>, store(pattern, response, confidence),
      warm_cache(patterns), should_bypass(confidence) -> bool.
      Performance: <100us P95 latency, >80% cache hit rate.
    layer: "logic"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-nervous/src/reflex/cache.rs"
    dependencies:
      - "M06-T08"
    acceptance_criteria:
      - "HopfieldQueryCache struct with DashMap for thread-safe access"
      - "query() uses locality-sensitive hash for O(1) lookup"
      - "Cache hit returns response in <100us"
      - "should_bypass() returns true when confidence > 0.95"
      - "CacheStats tracks: lookups, hits, misses, avg_latency_us"
      - "LRU eviction when max_entries exceeded"
    test_file: "crates/context-graph-nervous/tests/cache_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.2"
      - "REQ-BIONV-009 through REQ-BIONV-014"

  - id: "M06-T13"
    title: "Implement ReflexLayer (L2 Complete)"
    description: |
      Implement ReflexLayer struct implementing NervousLayer trait.
      Contains HopfieldQueryCache and implements fast bypass logic.
      Methods: query(embedding, context) -> Option<ReflexResponse>,
      store(pattern, response, confidence), stats() -> CacheStats.
      Bypass deeper layers when confidence > 0.95.
      Performance: <100us P95 latency.
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-nervous/src/reflex/layer.rs"
    dependencies:
      - "M06-T05"
      - "M06-T12"
    acceptance_criteria:
      - "ReflexLayer implements NervousLayer trait"
      - "query() returns cached response when hit with >0.95 confidence"
      - "Latency P95 <100us verified by inline measurement"
      - "ReflexResponse includes: response, confidence, latency, match_details"
      - "Thread-safe for concurrent access"
    test_file: "crates/context-graph-nervous/tests/reflex_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.2"
      - "REQ-BIONV-009 through REQ-BIONV-014"

  # ============================================================
  # LOGIC LAYER: L3 Memory Implementation
  # ============================================================

  - id: "M06-T14"
    title: "Implement ModernHopfieldNetwork for L3 Memory"
    description: |
      Implement ModernHopfieldNetwork for associative memory with 2^768 capacity.
      Fields: patterns (Vec<Vector1536>), metadata (Vec<PatternMetadata>),
      beta (f32, range [1.0, 5.0]), max_patterns, gpu_enabled, faiss_index (Option<FAISSGPUIndex>).
      Methods: energy(query) -> f32, retrieve(query, top_k) -> Vec<RetrievedPattern>,
      store(pattern, metadata), set_beta(beta).
      Energy: E(x) = -beta * log(sum_i exp(beta * x^T * xi)).
      Retrieve: p_i = softmax(beta * query^T * patterns).
      Performance: <1ms for top_k <= 100.
    layer: "logic"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-nervous/src/memory/hopfield.rs"
    dependencies:
      - "M06-T08"
    acceptance_criteria:
      - "ModernHopfieldNetwork struct with all fields"
      - "energy() implements Modern Hopfield energy function"
      - "retrieve() uses softmax attention over stored patterns"
      - "GPU path uses FAISS index when available"
      - "set_beta() clamps value to [1.0, 5.0] range"
      - "Capacity supports 2^768 patterns (limited by memory)"
      - "Noise tolerance >20% for retrieval"
    test_file: "crates/context-graph-nervous/tests/hopfield_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.3"
      - "REQ-BIONV-015 through REQ-BIONV-020"

  - id: "M06-T15"
    title: "Implement MemoryLayer (L3 Complete)"
    description: |
      Implement MemoryLayer struct implementing NervousLayer trait.
      Contains ModernHopfieldNetwork and storage integration.
      Methods: store(pattern, metadata) -> Result<Uuid>, retrieve(query, top_k, noise_tolerance),
      update_importance(pattern_id, importance), consolidate() -> ConsolidationReport,
      set_beta(beta).
      Performance: <1ms P95 for retrieval.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-nervous/src/memory/layer.rs"
    dependencies:
      - "M06-T05"
      - "M06-T14"
    acceptance_criteria:
      - "MemoryLayer implements NervousLayer trait"
      - "store() persists pattern and returns unique ID"
      - "retrieve() returns patterns within noise tolerance"
      - "consolidate() performs dream-phase pattern consolidation"
      - "MemoryStats tracks: pattern_count, memory_bytes, avg_retrieval_latency"
      - "Latency P95 <1ms for top_k <= 100"
    test_file: "crates/context-graph-nervous/tests/memory_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.3"
      - "REQ-BIONV-015 through REQ-BIONV-020"

  # ============================================================
  # LOGIC LAYER: L4 Learning Implementation
  # ============================================================

  - id: "M06-T16"
    title: "Implement UTLOptimizer for L4 Learning"
    description: |
      Implement UTLOptimizer for UTL-driven weight optimization.
      Fields: lambda_task (0.4), lambda_semantic (0.3), lambda_dyn (0.3),
      gradient_clip (1.0), update_hz (100.0), neuromod (Arc<RwLock<NeuromodulationController>>).
      Methods: compute_learning_score(UTLState) -> f32, compute_loss(state, task_loss, semantic_loss) -> f32,
      update_weights(signal) -> Result<(), UTLError>.
      Learning score: L = f((delta_s * delta_c) * w_e * cos(phi)).
      Combined loss: J = lambda_task * L_task + lambda_semantic * L_semantic + lambda_dyn * (1 - L).
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-nervous/src/learning/utl_optimizer.rs"
    dependencies:
      - "M06-T08"
    acceptance_criteria:
      - "UTLOptimizer struct with all lambda/config fields"
      - "compute_learning_score() implements UTL equation exactly"
      - "UTLState has: delta_s, delta_c, w_e, phi all in correct ranges"
      - "Gradient clipping applied at 1.0"
      - "Update frequency: 100Hz (10ms period)"
      - "Learning score in [0, 1] after sigmoid"
    test_file: "crates/context-graph-nervous/tests/utl_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.4"
      - "REQ-BIONV-021 through REQ-BIONV-024"

  - id: "M06-T17"
    title: "Implement NeuromodulationController for L4 Learning"
    description: |
      Implement NeuromodulationController for 4-channel neuromodulation.
      Fields: dopamine (-> hopfield.beta [1.0, 5.0]),
      serotonin (-> fuse_moe.top_k [2, 8]),
      noradrenaline (-> attention.temperature [0.5, 2.0]),
      acetylcholine (-> learning_rate [0.001, 0.002]).
      Methods: update(UTLState), get_hopfield_beta(), get_fuse_moe_top_k(),
      get_attention_temperature(), get_learning_rate().
      Performance: <200us per update.
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-nervous/src/learning/neuromodulation.rs"
    dependencies: []
    acceptance_criteria:
      - "NeuromodulationController with 4 modulator channels"
      - "All modulators in [0.0, 1.0] range internally"
      - "update() uses lerp for smooth transitions"
      - "Mapping functions produce correct output ranges"
      - "dopamine=0 -> beta=1.0, dopamine=1 -> beta=5.0"
      - "serotonin=0 -> top_k=2, serotonin=1 -> top_k=8"
      - "Update latency <200us"
    test_file: "crates/context-graph-nervous/tests/neuromod_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.4.2"
      - "REQ-BIONV-025, REQ-BIONV-026"

  - id: "M06-T18"
    title: "Implement LearningLayer (L4 Complete)"
    description: |
      Implement LearningLayer struct implementing NervousLayer trait.
      Contains UTLOptimizer and NeuromodulationController.
      Methods: process(MemoryInput, context) -> Result<LearningOutput>,
      update_importance(node_id, signal), get_neuromodulation() -> NeuromodulationState,
      get_utl_metrics() -> UTLMetrics.
      Performance: <10ms P95 latency, 100Hz update frequency.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-nervous/src/learning/layer.rs"
    dependencies:
      - "M06-T05"
      - "M06-T16"
      - "M06-T17"
    acceptance_criteria:
      - "LearningLayer implements NervousLayer trait"
      - "process() computes learning score and updates neuromodulators"
      - "LearningOutput includes: learning_score, loss, importance_delta, utl_state, neuromod_updates"
      - "UTLMetrics tracks: avg_learning_score, avg_entropy, avg_coherence, johari_distribution"
      - "Latency P95 <10ms"
      - "Update rate: 100Hz verified"
    test_file: "crates/context-graph-nervous/tests/learning_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.4"
      - "REQ-BIONV-021 through REQ-BIONV-027"

  # ============================================================
  # SURFACE LAYER: L5 Coherence Implementation
  # ============================================================

  - id: "M06-T19"
    title: "Implement ThalamicGate for L5 Coherence"
    description: |
      Implement ThalamicGate for cross-layer synchronization and routing.
      Fields: sync_interval (10ms), layer_states (HashMap<NervousLayerId, LayerState>),
      pending_messages (VecDeque<LayerMessage>), routing_table (RoutingTable).
      Methods: synchronize() -> SyncResult, route(message) -> Vec<NervousLayerId>,
      gate(input) -> bool (filter based on state), coherence_score() -> f32.
      Performance: Sync interval = 10ms.
    layer: "surface"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-nervous/src/coherence/thalamic_gate.rs"
    dependencies:
      - "M06-T04"
    acceptance_criteria:
      - "ThalamicGate struct with all fields"
      - "synchronize() updates layer_states every 10ms"
      - "route() determines target layers based on message type and state"
      - "gate() filters expired or invalid messages"
      - "coherence_score() aggregates cross-layer consistency"
      - "Eventual consistency model"
    test_file: "crates/context-graph-nervous/tests/thalamic_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.5.2"
      - "REQ-BIONV-028, REQ-BIONV-029"

  - id: "M06-T20"
    title: "Implement PredictiveCoder for L5 Coherence"
    description: |
      Implement PredictiveCoder for top-down prediction generation.
      Fields: model (PredictionModel), context (ContextState), confidence_threshold (f32).
      Methods: predict(context) -> Prediction, compute_error(prediction, observation) -> PredictionError,
      update(error), get_embedding_priors(domain) -> EmbeddingPriors.
      Prediction error propagates up, reducing token usage ~30%.
      Performance: <5ms for prediction generation.
    layer: "surface"
    priority: "high"
    estimated_hours: 4
    file_path: "crates/context-graph-nervous/src/coherence/predictive_coder.rs"
    dependencies: []
    acceptance_criteria:
      - "PredictiveCoder struct with prediction model"
      - "predict() generates expected input based on context"
      - "compute_error() returns prediction - observation delta"
      - "Only error signal propagates (not raw input)"
      - "EmbeddingPriors provides domain-specific model weights"
      - "Prediction latency <5ms"
    test_file: "crates/context-graph-nervous/tests/predictive_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.5.2"
      - "REQ-BIONV-030, REQ-BIONV-031"

  - id: "M06-T21"
    title: "Implement CoherenceLayer (L5 Complete)"
    description: |
      Implement CoherenceLayer struct implementing NervousLayer trait.
      Contains ThalamicGate, PredictiveCoder, ContextDistiller.
      Methods: process(CoherenceInput, context) -> Result<CoherenceOutput>,
      predict(context) -> Prediction, compute_error(prediction, observation),
      synchronize() -> SyncResult, distill(context, mode) -> DistilledContext, get_phase() -> f32.
      Phase (phi) updates based on coherence score for UTL integration.
      Performance: <10ms P95 latency.
    layer: "surface"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-nervous/src/coherence/layer.rs"
    dependencies:
      - "M06-T05"
      - "M06-T19"
      - "M06-T20"
    acceptance_criteria:
      - "CoherenceLayer implements NervousLayer trait"
      - "process() synchronizes state, generates predictions, updates phase"
      - "get_phase() returns current phi in [0, PI]"
      - "distill() compresses context with <15% information loss, >60% compression"
      - "CoherenceOutput includes: context, phi, coherence_score, distilled, prediction"
      - "Latency P95 <10ms"
    test_file: "crates/context-graph-nervous/tests/coherence_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.5"
      - "REQ-BIONV-028 through REQ-BIONV-035"

  # ============================================================
  # SURFACE LAYER: Marblestone Formal Verification
  # ============================================================

  - id: "M06-T22"
    title: "Implement FormalVerificationLayer (Marblestone SMT)"
    description: |
      Implement FormalVerificationLayer for Lean-inspired code verification.
      Fields: smt_solver (SmtSolver), verification_cache (LruCache<ContentHash, VerificationStatus>),
      timeout_ms (100).
      Methods: verify(content) -> VerificationStatus, is_code_content(content) -> bool,
      extract_conditions(content) -> Vec<VerificationCondition>.
      VerificationCondition types: BoundsCheck, NullSafety, TypeInvariant, Termination, CustomAssertion.
      VerificationStatus: Verified, Failed { counterexample }, Timeout, Unknown, NotApplicable.
    layer: "surface"
    priority: "high"
    estimated_hours: 5
    file_path: "crates/context-graph-nervous/src/verification/formal.rs"
    dependencies:
      - "M06-T08"
    acceptance_criteria:
      - "FormalVerificationLayer struct with SMT solver"
      - "verify() extracts and checks verification conditions"
      - "is_code_content() detects fn/def/function/class patterns"
      - "Timeout enforced at 100ms per verification"
      - "Cache prevents re-verification of identical content"
      - "VerificationCondition includes: id, assertion (SMT-LIB2), source_location"
    test_file: "crates/context-graph-nervous/tests/verification_tests.rs"
    spec_refs:
      - "SPEC-BIONV-006 Section 3.5.3"
      - "REQ-BIONV-056, REQ-BIONV-057"

  - id: "M06-T23"
    title: "Implement SmtSolver with Z3/CVC5 Backend"
    description: |
      Implement SmtSolver struct for SMT verification.
      Fields: backend (SolverBackend), config (SolverConfig).
      SolverBackend enum: Z3, CVC5, Portfolio.
      SolverConfig: default_timeout_ms, incremental, produce_proofs, produce_models.
      Methods: check(condition, timeout_ms) -> SolverResult, build_smt_input(condition) -> String.
      SolverResult: Sat(Model), Unsat, Timeout, Unknown.
      Model: assignments (HashMap<String, Value>).
    layer: "surface"
    priority: "high"
    estimated_hours: 4
    file_path: "crates/context-graph-nervous/src/verification/smt.rs"
    dependencies: []
    acceptance_criteria:
      - "SmtSolver struct with configurable backend"
      - "check() calls Z3/CVC5 with SMT-LIB2 input"
      - "build_smt_input() generates valid SMT-LIB2 from VerificationCondition"
      - "Model includes variable assignments for Sat results"
      - "Value enum: Bool, Int, Real, BitVec, String"
      - "Timeout properly enforced"
    test_file: "crates/context-graph-nervous/tests/smt_tests.rs"
    spec_refs:
      - "TECH-BIONV-006 Section 5.2"

  # ============================================================
  # SURFACE LAYER: System Orchestration
  # ============================================================

  - id: "M06-T24"
    title: "Implement BioNervousSystem Orchestrator"
    description: |
      Implement BioNervousSystem struct orchestrating all 5 layers.
      Fields: sensing (Arc<SensingLayer>), reflex (Arc<ReflexLayer>),
      memory (Arc<MemoryLayer>), learning (Arc<LearningLayer>),
      coherence (Arc<CoherenceLayer>), verification (Arc<RwLock<FormalVerificationLayer>>),
      channels (LayerChannels).
      Methods: process(RawInput) -> Result<NervousSystemOutput>,
      store_memory(content), get_cognitive_pulse() -> CognitivePulse.
      Pipeline: L1 -> L2 (cache check) -> L3 (if miss) -> L4 -> L5.
      Performance: <25ms P95, <50ms P99 end-to-end.
    layer: "surface"
    priority: "critical"
    estimated_hours: 6
    file_path: "crates/context-graph-nervous/src/system.rs"
    dependencies:
      - "M06-T11"
      - "M06-T13"
      - "M06-T15"
      - "M06-T18"
      - "M06-T21"
      - "M06-T22"
    acceptance_criteria:
      - "BioNervousSystem orchestrates all 5 layers"
      - "process() executes full pipeline with bypass on cache hit"
      - "Channel-based inter-layer communication"
      - "Graceful degradation on layer timeout"
      - "NervousSystemOutput includes: learning_score, phi, verification, latency"
      - "E2E latency: <25ms P95, <50ms P99"
      - "CognitivePulse tracks: entropy, coherence, johari quadrant"
    test_file: "crates/context-graph-nervous/tests/system_tests.rs"
    spec_refs:
      - "TECH-BIONV-006 Section 6"
      - "REQ-BIONV-041 through REQ-BIONV-055"
```

---

## Dependency Graph

```
M06-T01 (NervousLayerId) ─────────────────────────────────────────────────┐
M06-T02 (MessagePriority) ────────────────────────────────────────────────┤
M06-T03 (MessagePayload) ─────────────────────────────────────────────────┼──► M06-T04 (LayerMessage) ──┐
                                                                          │                              │
M06-T06 (HealthReport) ◄── M06-T01                                       │                              │
M06-T07 (TimeoutAction/Metrics)                                           │                              │
                                                                          │                              │
M06-T08 (LayerError) ◄── M06-T01                                         │                              │
                                                                          │                              │
M06-T04 + M06-T06 + M06-T07 + M06-T08 ──► M06-T05 (NervousLayer Trait) ◄─┘                              │
                                                   │                                                     │
                                                   │                                                     │
M06-T08 ──► M06-T09 (PIIScrubber) ────────────────┐│                                                    │
M06-T08 ──► M06-T10 (AdversarialDetector) ────────┼┤                                                    │
                                                   ││                                                    │
M06-T05 + M06-T09 + M06-T10 ──► M06-T11 (SensingLayer L1) ──────────────────────────────────────────────┤
                                                                                                         │
M06-T08 ──► M06-T12 (HopfieldQueryCache) ────────────────────────────────────────────────────┐          │
                                                                                              │          │
M06-T05 + M06-T12 ──► M06-T13 (ReflexLayer L2) ──────────────────────────────────────────────┤          │
                                                                                              │          │
M06-T08 ──► M06-T14 (ModernHopfieldNetwork) ─────────────────────────────────────────────────┤          │
                                                                                              │          │
M06-T05 + M06-T14 ──► M06-T15 (MemoryLayer L3) ──────────────────────────────────────────────┤          │
                                                                                              │          │
M06-T08 ──► M06-T16 (UTLOptimizer) ──────────────────────────────────────────────────────────┤          │
M06-T17 (NeuromodulationController) ─────────────────────────────────────────────────────────┤          │
                                                                                              │          │
M06-T05 + M06-T16 + M06-T17 ──► M06-T18 (LearningLayer L4) ──────────────────────────────────┤          │
                                                                                              │          │
M06-T04 ──► M06-T19 (ThalamicGate) ──────────────────────────────────────────────────────────┤          │
M06-T20 (PredictiveCoder) ───────────────────────────────────────────────────────────────────┤          │
                                                                                              │          │
M06-T05 + M06-T19 + M06-T20 ──► M06-T21 (CoherenceLayer L5) ─────────────────────────────────┤          │
                                                                                              │          │
M06-T08 ──► M06-T22 (FormalVerificationLayer) ───────────────────────────────────────────────┤          │
M06-T23 (SmtSolver) ─────────────────────────────────────────────────────────────────────────┤          │
                                                                                              │          │
M06-T11 + M06-T13 + M06-T15 + M06-T18 + M06-T21 + M06-T22 ──► M06-T24 (BioNervousSystem) ◄───┴──────────┘
```

---

## Implementation Order (Recommended)

### Week 1: Foundation & Protocol
1. M06-T01: NervousLayerId enum
2. M06-T02: MessagePriority enum
3. M06-T03: MessagePayload enum
4. M06-T04: LayerMessage struct
5. M06-T06: HealthReport types
6. M06-T07: TimeoutAction/Metrics
7. M06-T08: LayerError enum
8. M06-T05: NervousLayer trait

### Week 2: L1 Sensing & L2 Reflex
9. M06-T09: PIIScrubber (<1ms pattern matching)
10. M06-T10: AdversarialDetector
11. M06-T11: SensingLayer (<5ms complete)
12. M06-T12: HopfieldQueryCache (<100us)
13. M06-T13: ReflexLayer (<100us complete)

### Week 3: L3 Memory & L4 Learning
14. M06-T14: ModernHopfieldNetwork (<1ms)
15. M06-T15: MemoryLayer (<1ms complete)
16. M06-T17: NeuromodulationController (<200us)
17. M06-T16: UTLOptimizer (<10ms)
18. M06-T18: LearningLayer (<10ms complete)

### Week 4: L5 Coherence & System Integration
19. M06-T19: ThalamicGate (10ms sync)
20. M06-T20: PredictiveCoder (<5ms)
21. M06-T21: CoherenceLayer (<10ms complete)
22. M06-T23: SmtSolver
23. M06-T22: FormalVerificationLayer
24. M06-T24: BioNervousSystem (<25ms E2E)

---

## Quality Gates

| Gate | Criteria | Required For |
|------|----------|--------------|
| Protocol Complete | M06-T01 through M06-T08 pass all tests | Week 2 start |
| L1+L2 Operational | M06-T09 through M06-T13 pass latency tests | Week 3 start |
| L3+L4 Operational | M06-T14 through M06-T18 pass latency tests | Week 4 start |
| Module Complete | All 24 tasks complete, E2E <25ms P95 | Module 7 start |

---

## Performance Targets Summary

| Layer | Budget | Metric | Target |
|-------|--------|--------|--------|
| L1 Sensing | <5ms | P95 latency | 5ms |
| L1 Sensing | - | Throughput | 10K/sec |
| L1 PII | <1ms | Pattern matching | 1ms |
| L2 Reflex | <100us | P95 latency | 100us |
| L2 Reflex | >80% | Cache hit rate | 80% |
| L3 Memory | <1ms | P95 retrieval | 1ms |
| L3 Memory | >20% | Noise tolerance | 20% |
| L4 Learning | <10ms | P95 latency | 10ms |
| L4 Learning | 100Hz | Update frequency | 10ms |
| L4 Neuromod | <200us | Update latency | 200us |
| L5 Coherence | <10ms | P95 latency | 10ms |
| L5 Coherence | 10ms | Sync interval | 10ms |
| L5 Distill | >60% | Compression ratio | 60% |
| L5 Distill | <15% | Information loss | 15% |
| E2E Pipeline | <25ms | P95 latency | 25ms |
| E2E Pipeline | <50ms | P99 latency | 50ms |
| Verification | <100ms | Timeout | 100ms |

---

## Memory Budget

| Component | Budget |
|-----------|--------|
| Reflex cache (100K entries) | ~150MB |
| Hopfield patterns (1M) | ~6GB |
| FAISS GPU index | 8GB (shared with Module 4) |
| Layer state buffers | ~512MB |
| Verification cache | ~100MB |
| **Total RAM** | **~7GB** |
| **Total VRAM** | **~8GB (shared)** |

---

## Marblestone Integration Summary

Tasks with Marblestone features:
- **M06-T17**: NeuromodulationController (4-channel modulation)
- **M06-T14**: ModernHopfieldNetwork (beta controlled by dopamine)
- **M06-T22**: FormalVerificationLayer (SMT-based code verification)
- **M06-T23**: SmtSolver (Z3/CVC5 backend)

---

## Critical Constraints

**LATENCY BUDGETS ARE HARD REQUIREMENTS**:
- L1: <5ms (use tokio::time::timeout)
- L2: <100us (inline measurement, no async overhead)
- L3: <1ms (FAISS GPU path required for scale)
- L4: <10ms (100Hz update cycle)
- L5: <10ms (10ms sync interval)
- E2E: <25ms P95, <50ms P99

**BYPASS ON HIGH CONFIDENCE**:
- When L2 cache hit has confidence > 0.95, SKIP L3-L4, go directly to L5
- This path must complete in <200us + L5 budget

**NEUROMODULATOR RANGES**:
- dopamine: [0,1] -> hopfield.beta [1.0, 5.0]
- serotonin: [0,1] -> fuse_moe.top_k [2, 8]
- noradrenaline: [0,1] -> attention.temperature [0.5, 2.0]
- acetylcholine: [0,1] -> learning_rate [0.001, 0.002]

---

*Generated: 2025-12-31*
*Module: 06 - Bio-Nervous System*
*Version: 1.0.0*
*Total Tasks: 24*
