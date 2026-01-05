# Module 11: Immune System - Atomic Tasks

```yaml
metadata:
  module: Module 11 - Immune System
  phase: 10
  approach: inside-out-bottom-up
  spec_refs:
    - SPEC-IMMUNE-011 (Functional)
    - TECH-IMMUNE-011 (Technical)
  total_tasks: 20
  created: 2025-12-31
  performance_targets:
    detection_latency: <50ms
    false_positive_rate: <0.1%
    quarantine_isolation: <100 microseconds
    drift_detection_window: <1 second
    memory_overhead: <5MB per session
  dependencies:
    - Module 1 (Ghost System)
    - Module 2 (Core Infrastructure)
    - Module 3 (Embedding Pipeline)
    - Module 4 (Knowledge Graph)
    - Module 6 (Bio-Nervous System)
    - Module 9 (Dream Layer)
    - Module 10 (Neuromodulation)
  layers:
    foundation: 6 tasks
    logic: 9 tasks
    surface: 5 tasks
```

---

## Foundation Layer Tasks (Build First)

These tasks establish core types, error handling, and configuration structures for the Immune System.

```yaml
- id: TASK-IMM-001
  title: Create Immune System module structure with feature flags
  type: implementation
  layer: foundation
  requirement_refs: [REQ-IMMUNE-001, REQ-IMMUNE-002]
  dependencies: [Module 6 Bio-Nervous infrastructure]
  acceptance_criteria:
    - src/immune/mod.rs module entry with feature-gated exports
    - Cargo.toml includes immune module under features
    - Feature flag "immune" enables Immune System compilation
    - Module compiles with and without immune feature
    - Re-exports all public types from submodules (threat_detector, semantic_immune, drift_monitor, quarantine)
    - cargo check -p context-graph --features immune succeeds
  estimated_complexity: low
  files_affected:
    - crates/context-graph/Cargo.toml
    - crates/context-graph/src/immune/mod.rs
    - crates/context-graph/src/lib.rs

- id: TASK-IMM-002
  title: Implement ThreatDetectorConfig and core type definitions
  type: implementation
  layer: foundation
  requirement_refs: [REQ-IMMUNE-001, REQ-IMMUNE-002]
  dependencies: [TASK-IMM-001]
  acceptance_criteria:
    - ThreatDetectorConfig struct with all fields (embedding_anomaly_threshold, content_alignment_threshold, max_entropy_delta, etc.)
    - Default impl sets embedding_anomaly_threshold = 3.0, content_alignment_threshold = 0.4, timeout = 50ms
    - ThreatLevel enum with NONE, LOW, MEDIUM, HIGH, CRITICAL variants
    - ThreatType enum with INJECTION, POISONING, EXTRACTION, MANIPULATION, ANOMALY, DRIFT, CORRUPTION, OVERFLOW variants
    - AttackType enum with PromptInjection, Jailbreak, EmbeddingPoisoning, SemanticCancer, CircularLogic, PiiExfiltration, PrivilegeEscalation, DenialOfService, Unknown
    - Severity enum with Info, Low, Medium, High, Critical variants (Ord derived)
    - validate() method checks threshold ranges and timeout > 0
    - Unit tests for validation logic and default values
  estimated_complexity: medium
  files_affected:
    - crates/context-graph/src/immune/config.rs
    - crates/context-graph/src/immune/types.rs

- id: TASK-IMM-003
  title: Implement ImmuneSystemError type hierarchy
  type: implementation
  layer: foundation
  requirement_refs: [REQ-IMMUNE-038, REQ-IMMUNE-039]
  dependencies: [TASK-IMM-001]
  acceptance_criteria:
    - ImmuneSystemError enum with thiserror derive
    - Variants: DetectionTimeout, PatternLoadFailed, QuarantineFailed, MemoryExhausted
    - Variants: DriftOverflow, SignatureInvalid, CellUnavailable, ValidationFailed
    - Variants: ConcurrentModification, IntegrityViolation, ConfigurationError
    - From<std::io::Error> impl for file operations
    - All variants have descriptive #[error()] messages with context
    - ImmuneResult<T> type alias defined
    - Error type is Send + Sync
    - ErrorSeverity enum with LOW, MEDIUM, HIGH, CRITICAL variants
    - is_recoverable() method for error classification
  estimated_complexity: low
  files_affected:
    - crates/context-graph/src/immune/error.rs

- id: TASK-IMM-004
  title: Implement ThreatAssessment and ThreatIndicator types
  type: implementation
  layer: foundation
  requirement_refs: [REQ-IMMUNE-001, REQ-IMMUNE-002, REQ-IMMUNE-003]
  dependencies: [TASK-IMM-002]
  acceptance_criteria:
    - ThreatAssessment struct with threatLevel, confidence (0.0-1.0), threatTypes, indicators, recommendation, analysisTimeMs
    - Action enum with ALLOW, MONITOR, QUARANTINE, BLOCK variants
    - ThreatIndicator struct with type, description, severity (0.0-1.0), evidence, location
    - SourceLocation struct with file, line, column, context
    - AssessmentMetadata struct with detectorVersion, signatureDbVersion, timestamp
    - DetectorStats struct with totalScanned, threatsDetected, falsePositives, avgLatencyMs
    - Serialization support (serde Serialize/Deserialize)
    - Unit tests for assessment creation and serialization
  estimated_complexity: medium
  files_affected:
    - crates/context-graph/src/immune/assessment.rs

- id: TASK-IMM-005
  title: Implement SignatureDatabase for threat patterns
  type: implementation
  layer: foundation
  requirement_refs: [REQ-IMMUNE-002, REQ-IMMUNE-012]
  dependencies: [TASK-IMM-002, TASK-IMM-004]
  acceptance_criteria:
    - ThreatSignature struct with id, version, name, description, severity, pattern, matching config, response config
    - SignaturePattern struct with type (EXACT, REGEX, FUZZY, SEMANTIC), value, flags
    - CompiledSignature struct with pre-compiled regex and pattern data
    - SignatureDatabase struct with thread-safe signature storage (Arc<RwLock>)
    - add_signature() adds and compiles new signature
    - remove_signature() removes by ID
    - get_signature() retrieves by ID
    - list_signatures() returns all signatures
    - load_from_file() loads signatures from YAML/JSON
    - Default signatures for prompt injection patterns from constitution.yaml
    - Unit tests for signature CRUD operations
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/signature.rs

- id: TASK-IMM-006
  title: Implement entropy calculation utilities
  type: implementation
  layer: foundation
  requirement_refs: [REQ-IMMUNE-003]
  dependencies: [TASK-IMM-001]
  acceptance_criteria:
    - calculate_shannon_entropy(data: &[u8]) -> f32 with O(n) complexity
    - calculate_content_entropy(content: &str) -> f32 character-based entropy
    - calculate_embedding_entropy(embedding: &[f32]) -> EmbeddingEntropyResult
    - EmbeddingEntropyResult struct with entropy, sparsity, balance, is_anomalous, anomaly_type
    - EmbeddingAnomalyType enum with AllZero, TooSparse, TooDense, Imbalanced
    - EntropyBaseline struct with expected_entropy (4.5), max_entropy (7.0), min_entropy (2.0)
    - Sliding window entropy with configurable window size
    - SIMD optimization for large arrays where available
    - Unit tests for known entropy values
    - Benchmark: entropy calculation < 1ms for 10KB content
  estimated_complexity: medium
  files_affected:
    - crates/context-graph/src/immune/entropy.rs
```

---

## Logic Layer Tasks (Build Second)

These tasks implement detection algorithms, pattern matching, drift monitoring, and quarantine mechanics.

```yaml
- id: TASK-IMM-007
  title: Implement PatternMatcher with Aho-Corasick automaton
  type: implementation
  layer: logic
  requirement_refs: [REQ-IMMUNE-002, REQ-IMMUNE-012, REQ-IMMUNE-013]
  dependencies: [TASK-IMM-005]
  acceptance_criteria:
    - PatternMatcher struct with compiled patterns and LRU cache
    - Aho-Corasick automaton for multi-pattern exact matching
    - matchPatterns(input: &str) -> Vec<PatternMatch> with O(n+m) complexity
    - Regex pattern matching with parallel execution for large pattern sets
    - Fuzzy matching using Levenshtein distance with early termination
    - levenshteinWithCutoff(s1, s2, maxDist) returns -1 if distance > maxDist
    - PatternMatch struct with signatureId, position, matchType (EXACT/REGEX/FUZZY), confidence
    - deduplicateAndRank() combines overlapping matches
    - Cache hit rate tracking
    - Constraint: Match_Latency < 5ms for typical content
    - Unit tests for exact, regex, and fuzzy matching
    - Integration test validates cache effectiveness (>80% hit rate)
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/pattern_matcher.rs

- id: TASK-IMM-008
  title: Implement EntropyAnalyzer with anomaly detection
  type: implementation
  layer: logic
  requirement_refs: [REQ-IMMUNE-003, REQ-IMMUNE-014]
  dependencies: [TASK-IMM-006]
  acceptance_criteria:
    - EntropyAnalyzer struct with domain_stats, global_baseline, threshold_sigma
    - EntropyStats struct with mean, variance, count, sliding window (Welford's algorithm)
    - analyzeEntropyPattern(data) returns EntropyAnalysis with windowEntropies, anomalies, overallEntropy
    - EntropyAnomaly struct with position, entropy, severity
    - is_anomalous(content, domain) checks against baseline and domain statistics
    - Kolmogorov complexity approximation using compression ratio
    - calculateAnomalyScore() using z-score against baseline
    - Domain-specific entropy baselines with auto-calibration
    - Constraint: Entropy_Calc_Latency < 1ms, Anomaly_Check_Latency < 1ms
    - Unit tests for entropy calculation accuracy
    - Integration test validates anomaly detection on adversarial examples
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/entropy_analyzer.rs

- id: TASK-IMM-009
  title: Implement CancerDetector for semantic replication detection
  type: implementation
  layer: logic
  requirement_refs: [REQ-IMMUNE-004, REQ-IMMUNE-015]
  dependencies: [TASK-IMM-004, TASK-IMM-006]
  acceptance_criteria:
    - CancerDetector struct with replicationHistory, thresholds (replication: 10/min, mutation: 0.3, spread: 5 hops)
    - detect(graph) returns CancerAssessment with isCancerous, affectedNodes, replicationRate, mutationScore
    - findReplicationClusters() using Locality-Sensitive Hashing (128 bits, 8 bands)
    - calculateMutationScore() based on semantic drift from cluster centroid
    - analyzeSpreadPattern() returns DENSE_CLUSTER, LINEAR_SPREAD, HUB_AND_SPOKE, or SCATTERED
    - SpreadPattern enum with all four variants
    - identifyOrigin() traces back to original malicious node
    - calculatePriority() determines containment urgency
    - estimateContainmentTime() estimates cleanup duration
    - CancerAssessment struct with all fields from technical spec
    - Constraint: Cancer_Detection_Latency < 100ms
    - Unit tests for cluster detection and spread analysis
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/cancer_detector.rs

- id: TASK-IMM-010
  title: Implement PoisoningDetector for embedding attacks
  type: implementation
  layer: logic
  requirement_refs: [REQ-IMMUNE-005, REQ-IMMUNE-016]
  dependencies: [TASK-IMM-008]
  acceptance_criteria:
    - PoisoningDetector struct with embeddingDistribution (GMM), relationshipPriors, credibilityScores
    - detectPoisoning(node) returns PoisoningAssessment with isPoisoned, confidence, poisonType, affectedDimensions
    - PoisonType enum with EMBEDDING_SHIFT, BACKDOOR, TROJAN, ADVERSARIAL, LABEL_FLIP
    - analyzeEmbedding() using Mahalanobis distance (threshold: 3.0) from learned distribution
    - detectBackdoorPattern() analyzing sparsity, activation entropy, content coherence
    - detectAdversarialPerturbation() using FFT for high-frequency noise detection
    - analyzeRelationships() checking edge consistency with priors
    - findAnomalousDimensions() returns indices of compromised embedding dimensions
    - generateRemediation() creates recovery plan
    - RemediationPlan struct with steps, estimated_time, confidence
    - Constraint: Poisoning_Detection_Latency < 50ms
    - Unit tests with known adversarial examples
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/poisoning_detector.rs

- id: TASK-IMM-011
  title: Implement KL Divergence calculator for drift detection
  type: implementation
  layer: logic
  requirement_refs: [REQ-IMMUNE-006, REQ-IMMUNE-020]
  dependencies: [TASK-IMM-006]
  acceptance_criteria:
    - KLDivergenceCalculator struct with epsilon (1e-10), numBins (100)
    - calculate(p, q) computes D_KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))
    - calculateSymmetric(p, q) computes Jensen-Shannon Divergence (more stable)
    - toHistogram(dist) converts continuous distribution to histogram
    - smooth(hist) applies Laplace smoothing to avoid zero probabilities
    - averageDistribution(p, q) for JS divergence calculation
    - Distribution struct with values, min, max
    - Support for multi-dimensional distributions
    - Constraint: KL_Calc_Latency < 5ms for 1000-sample distributions
    - Unit tests validating KL divergence properties (non-negative, zero for identical)
  estimated_complexity: medium
  files_affected:
    - crates/context-graph/src/immune/kl_divergence.rs

- id: TASK-IMM-012
  title: Implement ADWIN detector for concept drift
  type: implementation
  layer: logic
  requirement_refs: [REQ-IMMUNE-006, REQ-IMMUNE-021]
  dependencies: [TASK-IMM-011]
  acceptance_criteria:
    - ADWINDetector struct with window, buckets, delta (0.002), minWindowSize (5), maxBuckets (5)
    - update(value) inserts element and returns ADWINResult
    - ADWINResult struct with changeDetected, cutPoint, windowSize, mean, variance
    - detectChange() finds optimal cut point comparing sub-window means
    - calculateThreshold() using Hoeffding bound: epsilon = sqrt((1/(2m)) * log(4/delta'))
    - compressWindow() maintains O(log W) buckets with exponential sizing
    - mergeBuckets() combines two buckets into one
    - shrinkWindow() removes old elements before cut point
    - Bucket struct with sum, variance, size
    - getMean(start, end) returns sub-window mean
    - Constraint: Update_Latency < 100us per observation
    - Unit tests with synthetic drift data
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/adwin_detector.rs

- id: TASK-IMM-013
  title: Implement ConceptDriftMonitor with multi-algorithm detection
  type: implementation
  layer: logic
  requirement_refs: [REQ-IMMUNE-006, REQ-IMMUNE-022]
  dependencies: [TASK-IMM-011, TASK-IMM-012]
  acceptance_criteria:
    - ConceptDriftMonitor struct with DistributionTracker, ADWINDetector, PageHinkleyTest, DriftClassifier
    - update(observation) processes through all detectors
    - detectDrift() returns DriftResult with hasDrift, hasWarning, driftType, severity, affectedFeatures, klDivergence
    - DriftType enum with SUDDEN, GRADUAL, INCREMENTAL, RECURRING, OUTLIER
    - DriftSeverity enum with MINOR, MODERATE, SEVERE, CRITICAL
    - getDistribution(feature) returns current distribution for feature
    - compareDistributions(d1, d2) returns DistributionComparison with KL divergence
    - setDriftThreshold() and setWarningThreshold() for configuration
    - resetWindow() clears current window
    - WindowInfo struct with size, mean, variance, samples
    - Constraint: Drift_Detection_Window < 1 second
    - Integration test validates detection of all drift types
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/drift_monitor.rs

- id: TASK-IMM-014
  title: Implement HomeostaticPlasticity regulator
  type: implementation
  layer: logic
  requirement_refs: [REQ-IMMUNE-007, REQ-IMMUNE-023]
  dependencies: [TASK-IMM-013]
  acceptance_criteria:
    - HomeostaticPlasticity struct with parameters, scalingFactor (0.1), smoothingWindow (10)
    - ParameterState struct with targetRange, history, plasticity
    - monitorParameter(name, value) returns StabilityStatus
    - StabilityStatus struct with parameter, currentValue, targetRange, deviation, isStable, trend, requiresIntervention
    - Trend enum with INCREASING, DECREASING, STABLE, OSCILLATING
    - regulate(parameters) returns RegulationResult with adjustments, overallStability, interventionCount
    - RegulationType enum with SCALING, CLAMPING, SMOOTHING, RESET
    - calculateCorrection() using PID-like control (kp=0.5, ki=0.1, kd=0.05)
    - setTargetRange(name, min, max) configures parameter bounds
    - adjustPlasticity(factor) scales adaptation rate
    - getStabilityScore() returns overall system stability (0.0-1.0)
    - Constraint: Homeostatic_Recovery_Time < 10s
    - Unit tests for PID controller behavior
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/homeostatic.rs

- id: TASK-IMM-015
  title: Implement FastIsolator for sub-100 microsecond quarantine
  type: implementation
  layer: logic
  requirement_refs: [REQ-IMMUNE-008, REQ-IMMUNE-024]
  dependencies: [TASK-IMM-003]
  acceptance_criteria:
    - FastIsolator struct with isolationTable, memoryPool (pre-allocated), freeList, cellSize (4KB)
    - isolateImmediate(target) returns IsolationResult with timeUs < 100
    - IsolationResult struct with success, targetId, cellIndex, timeUs, isolationType
    - acquireCell() O(1) cell acquisition from pre-allocated pool using lock-free list
    - getCell(index) returns view into isolation cell
    - serializeTarget() fast serialization of target data
    - invalidateOriginal() sets quarantine flag in node metadata
    - IsolationEntry struct with targetId, cellIndex, timestamp, status
    - IsolationStatus enum with ISOLATED, RELEASED, EXPIRED
    - Memory pool size configurable (default: 1000 cells * 4KB = 4MB)
    - Atomic operations for thread safety
    - Constraint: Isolation_Time < 100 microseconds
    - Benchmark validates sub-100us isolation
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/fast_isolator.rs
```

---

## Surface Layer Tasks (Build Last)

These tasks implement high-level detectors, managers, and the integrated immune context.

```yaml
- id: TASK-IMM-016
  title: Implement ThreatDetector coordinator
  type: implementation
  layer: surface
  requirement_refs: [REQ-IMMUNE-001, REQ-IMMUNE-002, REQ-IMMUNE-003]
  dependencies: [TASK-IMM-007, TASK-IMM-008, TASK-IMM-009, TASK-IMM-010]
  acceptance_criteria:
    - ThreatDetector struct combining PatternMatcher, EntropyAnalyzer, BehavioralAnalyzer, ThreatClassifier
    - analyze(input) returns ThreatAssessment within timeout (default 50ms)
    - analyzeAsync(input) returns Promise<ThreatAssessment>
    - analyzeBatch(inputs) processes multiple inputs with parallel execution
    - addSignature() adds new threat signature
    - removeSignature() removes by ID
    - setThreshold(type, threshold) configures detection sensitivity
    - getStats() returns DetectorStats
    - resetStats() clears statistics
    - Ensemble voting from multiple detection methods
    - RuleEngine for deterministic threat classification
    - MLClassifier integration point (stub for future ML model)
    - Constraint: Detection_Latency < 50ms, P99 < 75ms
    - Integration test validates end-to-end detection
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/threat_detector.rs

- id: TASK-IMM-017
  title: Implement SemanticImmune coordinator
  type: implementation
  layer: surface
  requirement_refs: [REQ-IMMUNE-004, REQ-IMMUNE-005, REQ-IMMUNE-025]
  dependencies: [TASK-IMM-009, TASK-IMM-010]
  acceptance_criteria:
    - SemanticImmune struct with CancerDetector, PoisoningDetector, IntegrityVerifier, AntibodyGenerator
    - detectCancer(graph) returns CancerAssessment
    - monitorReplication(node) returns ReplicationStatus
    - detectPoisoning(node) returns PoisoningAssessment
    - validateEmbedding(embedding) returns EmbeddingValidation
    - verifyRelationship(edge) returns RelationshipValidity
    - computeSemanticHash(node) returns SHA-256 based semantic hash
    - verifyIntegrity(node, expectedHash) validates node integrity
    - trackProvenance(node) returns ProvenanceChain
    - generateAntibody(threat) creates Antibody for future detection
    - deployAntibody(antibody) adds to active defenses
    - Antibody struct with pattern, response, createdAt, effectiveness
    - ProvenanceChain struct with origin, transformations, currentHash
    - Unit tests for each detection capability
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/semantic_immune.rs

- id: TASK-IMM-018
  title: Implement QuarantineManager with zone management
  type: implementation
  layer: surface
  requirement_refs: [REQ-IMMUNE-008, REQ-IMMUNE-026]
  dependencies: [TASK-IMM-015]
  acceptance_criteria:
    - QuarantineManager struct with FastIsolator, QuarantineZone, ReleaseManager, AuditLogger
    - quarantine(target, options) returns QuarantineResult
    - quarantineImmediate(target) uses FastIsolator for <100us isolation
    - quarantineBatch(targets) processes multiple targets
    - createZone(config) returns QuarantineZone
    - QuarantineZone struct with cells, resourceLimits, monitor
    - ContainmentCell struct with id, targetId, level, createdAt, limits, accessLog, violations
    - ContainmentLevel enum with SOFT, MEDIUM, HARD, AIRGAP
    - IsolationType enum with MEMORY, NETWORK, COMPUTE, FULL
    - release(zoneId, targetId) validates and releases target
    - releaseGradual(zoneId, targetId, steps) staged release
    - getQuarantineStatus(targetId) returns QuarantineStatus
    - getZoneMetrics(zoneId) returns ZoneMetrics
    - getAuditLog(filter) returns audit entries
    - AuditEntry struct with timestamp, action, target, user, result
    - Constraint: Quarantine_Isolation < 100us (immediate), < 1ms (full)
    - Integration test validates quarantine lifecycle
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/quarantine_manager.rs

- id: TASK-IMM-019
  title: Implement ImmuneMetrics and monitoring
  type: implementation
  layer: surface
  requirement_refs: [REQ-IMMUNE-027, REQ-IMMUNE-028]
  dependencies: [TASK-IMM-016, TASK-IMM-017, TASK-IMM-018]
  acceptance_criteria:
    - ImmuneMetrics struct with atomic counters for threats, quarantines, drift events, false positives
    - record_detection(assessment) updates detection counters
    - record_quarantine(result) updates quarantine metrics
    - record_drift(result) updates drift detection metrics
    - record_false_positive() increments FP counter
    - detection_rate() returns threats detected per minute
    - false_positive_rate() returns FP / total detections
    - avg_detection_latency() returns average latency in ms
    - quarantine_utilization() returns cells in use / total cells
    - check_targets() returns PerformanceStatus
    - PerformanceStatus struct with latency_target_met, fp_rate_target_met, isolation_target_met
    - Prometheus-compatible metrics export
    - All operations use Ordering::Relaxed for minimal overhead
    - Unit tests verify metric calculations
  estimated_complexity: medium
  files_affected:
    - crates/context-graph/src/immune/metrics.rs

- id: TASK-IMM-020
  title: Implement ImmuneSystem high-level API
  type: implementation
  layer: surface
  requirement_refs: [REQ-IMMUNE-001, REQ-IMMUNE-029, REQ-IMMUNE-030]
  dependencies: [TASK-IMM-013, TASK-IMM-014, TASK-IMM-016, TASK-IMM-017, TASK-IMM-018, TASK-IMM-019]
  acceptance_criteria:
    - ImmuneSystem struct combining all immune components
    - new(config) initializes all subsystems
    - scan(input) performs full threat assessment
    - scan_node(node) analyzes context node for threats
    - scan_query(query) analyzes query for injection
    - enable_realtime() enables real-time monitoring
    - disable_realtime() disables monitoring
    - get_threat_detector() returns Arc to ThreatDetector
    - get_quarantine_manager() returns Arc to QuarantineManager
    - get_drift_monitor() returns Arc to ConceptDriftMonitor
    - metrics() returns Arc<ImmuneMetrics>
    - performance_status() returns current PerformanceStatus
    - Event emission for threat:detected, quarantine:created, drift:warning
    - Drop impl cleanly shuts down all subsystems
    - Integration test verifies end-to-end operation
    - Benchmark validates <50ms detection, <0.1% FP rate, <100us quarantine
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/system.rs
```

---

## Test Tasks

```yaml
- id: TASK-IMM-TEST-001
  title: Unit tests for configuration and core types
  type: test
  layer: foundation
  requirement_refs: [REQ-IMMUNE-049]
  dependencies: [TASK-IMM-002, TASK-IMM-003, TASK-IMM-004, TASK-IMM-006]
  acceptance_criteria:
    - Tests for ThreatDetectorConfig validation (threshold ranges)
    - Tests for ThreatDetectorConfig default values
    - Tests for ThreatLevel and Severity ordering
    - Tests for ImmuneSystemError display formatting
    - Tests for entropy calculation accuracy (known values)
    - Tests for ThreatAssessment serialization/deserialization
    - All tests pass with cargo test -p context-graph --features immune
  estimated_complexity: medium
  files_affected:
    - crates/context-graph/src/immune/config.rs (test module)
    - crates/context-graph/src/immune/error.rs (test module)
    - crates/context-graph/src/immune/entropy.rs (test module)

- id: TASK-IMM-TEST-002
  title: Unit tests for pattern matching and detection algorithms
  type: test
  layer: logic
  requirement_refs: [REQ-IMMUNE-049]
  dependencies: [TASK-IMM-007, TASK-IMM-008, TASK-IMM-009, TASK-IMM-010]
  acceptance_criteria:
    - TC-IMM-001: Pattern matching exact match test
    - TC-IMM-002: Pattern matching regex test
    - TC-IMM-003: Pattern matching fuzzy match test (Levenshtein)
    - TC-IMM-004: Entropy anomaly detection test
    - TC-IMM-005: Cancer detection cluster identification test
    - TC-IMM-006: Poisoning detection adversarial examples test
    - TC-IMM-007: Cache hit rate verification (>80%)
    - Tests verify pattern matching latency < 5ms
    - All tests pass with cargo test
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/pattern_matcher.rs (test module)
    - crates/context-graph/src/immune/entropy_analyzer.rs (test module)
    - crates/context-graph/src/immune/cancer_detector.rs (test module)
    - crates/context-graph/src/immune/poisoning_detector.rs (test module)

- id: TASK-IMM-TEST-003
  title: Unit tests for drift detection and homeostatic regulation
  type: test
  layer: logic
  requirement_refs: [REQ-IMMUNE-049]
  dependencies: [TASK-IMM-011, TASK-IMM-012, TASK-IMM-013, TASK-IMM-014]
  acceptance_criteria:
    - TC-IMM-008: KL divergence properties test (non-negative, zero for identical)
    - TC-IMM-009: ADWIN sudden drift detection test
    - TC-IMM-010: ADWIN gradual drift detection test
    - TC-IMM-011: ConceptDriftMonitor multi-detector test
    - TC-IMM-012: HomeostaticPlasticity PID controller test
    - TC-IMM-013: Parameter stability monitoring test
    - Tests verify drift detection window < 1 second
    - All tests pass with cargo test
  estimated_complexity: high
  files_affected:
    - crates/context-graph/src/immune/kl_divergence.rs (test module)
    - crates/context-graph/src/immune/adwin_detector.rs (test module)
    - crates/context-graph/src/immune/drift_monitor.rs (test module)
    - crates/context-graph/src/immune/homeostatic.rs (test module)

- id: TASK-IMM-TEST-004
  title: Integration tests for immune system performance
  type: test
  layer: surface
  requirement_refs: [REQ-IMMUNE-050]
  dependencies: [TASK-IMM-020]
  acceptance_criteria:
    - TC-IMM-014: Detection latency benchmark (<50ms target)
    - TC-IMM-015: False positive rate validation (<0.1% target)
    - TC-IMM-016: Quarantine isolation timing (<100us target)
    - TC-IMM-017: Cancer detection accuracy test (>98% target)
    - TC-IMM-018: Poisoning detection accuracy test (>97% target)
    - TC-IMM-019: Drift detection accuracy test (>95% target)
    - TC-IMM-020: End-to-end threat detection pipeline test
    - TC-IMM-021: Memory overhead validation (<5MB per session)
    - TC-IMM-022: Homeostatic recovery time test (<10s)
    - All tests in tests/integration/immune_tests.rs pass
  estimated_complexity: high
  files_affected:
    - tests/integration/immune_tests.rs
```

---

## Dependency Graph

```
TASK-IMM-001 (module structure)
    |
    +-- TASK-IMM-002 (config) --+
    |                            |
    +-- TASK-IMM-003 (errors) --+-- TASK-IMM-004 (assessment types)
    |                            |
    +-- TASK-IMM-006 (entropy) --+
            |                    |
            |                    +-- TASK-IMM-005 (signature db)
            |                            |
    +-------+-------+-------------------+
    |               |                   |
TASK-IMM-007    TASK-IMM-008    TASK-IMM-011 (KL divergence)
(pattern)       (entropy anal)       |
    |               |               TASK-IMM-012 (ADWIN)
    |               |                   |
    +-------+-------+               TASK-IMM-013 (drift monitor)
            |                           |
    +-------+-------+               TASK-IMM-014 (homeostatic)
    |               |
TASK-IMM-009    TASK-IMM-010
(cancer)        (poisoning)
    |               |
    +-------+-------+
            |
    +-------+-------+---------------+
    |               |               |
TASK-IMM-016    TASK-IMM-017    TASK-IMM-015 (fast isolator)
(threat detect) (semantic imm)       |
    |               |               |
    |               |           TASK-IMM-018 (quarantine mgr)
    |               |               |
    +---------------+---------------+
            |
    TASK-IMM-019 (metrics)
            |
    TASK-IMM-020 (ImmuneSystem API)
```

---

## Traceability Matrix

| Task ID | Requirements Covered |
|---------|---------------------|
| TASK-IMM-001 | REQ-IMMUNE-001, REQ-IMMUNE-002 |
| TASK-IMM-002 | REQ-IMMUNE-001, REQ-IMMUNE-002 |
| TASK-IMM-003 | REQ-IMMUNE-038, REQ-IMMUNE-039 |
| TASK-IMM-004 | REQ-IMMUNE-001, REQ-IMMUNE-002, REQ-IMMUNE-003 |
| TASK-IMM-005 | REQ-IMMUNE-002, REQ-IMMUNE-012 |
| TASK-IMM-006 | REQ-IMMUNE-003 |
| TASK-IMM-007 | REQ-IMMUNE-002, REQ-IMMUNE-012, REQ-IMMUNE-013 |
| TASK-IMM-008 | REQ-IMMUNE-003, REQ-IMMUNE-014 |
| TASK-IMM-009 | REQ-IMMUNE-004, REQ-IMMUNE-015 |
| TASK-IMM-010 | REQ-IMMUNE-005, REQ-IMMUNE-016 |
| TASK-IMM-011 | REQ-IMMUNE-006, REQ-IMMUNE-020 |
| TASK-IMM-012 | REQ-IMMUNE-006, REQ-IMMUNE-021 |
| TASK-IMM-013 | REQ-IMMUNE-006, REQ-IMMUNE-022 |
| TASK-IMM-014 | REQ-IMMUNE-007, REQ-IMMUNE-023 |
| TASK-IMM-015 | REQ-IMMUNE-008, REQ-IMMUNE-024 |
| TASK-IMM-016 | REQ-IMMUNE-001, REQ-IMMUNE-002, REQ-IMMUNE-003 |
| TASK-IMM-017 | REQ-IMMUNE-004, REQ-IMMUNE-005, REQ-IMMUNE-025 |
| TASK-IMM-018 | REQ-IMMUNE-008, REQ-IMMUNE-026 |
| TASK-IMM-019 | REQ-IMMUNE-027, REQ-IMMUNE-028 |
| TASK-IMM-020 | REQ-IMMUNE-001, REQ-IMMUNE-029, REQ-IMMUNE-030 |
| TASK-IMM-TEST-001 | REQ-IMMUNE-049 |
| TASK-IMM-TEST-002 | REQ-IMMUNE-049 |
| TASK-IMM-TEST-003 | REQ-IMMUNE-049 |
| TASK-IMM-TEST-004 | REQ-IMMUNE-050 |

---

## Performance Verification Criteria

| Metric | Target | Verification Task |
|--------|--------|------------------|
| Detection latency | <50ms | TASK-IMM-TEST-004 (TC-IMM-014) |
| P99 detection latency | <75ms | TASK-IMM-TEST-004 (TC-IMM-014) |
| Pattern matching latency | <5ms | TASK-IMM-TEST-002 (TC-IMM-001-003) |
| Entropy analysis latency | <1ms | TASK-IMM-TEST-002 (TC-IMM-004) |
| False positive rate | <0.1% | TASK-IMM-TEST-004 (TC-IMM-015) |
| False negative rate | <5% | TASK-IMM-TEST-004 (TC-IMM-015) |
| Quarantine isolation | <100us | TASK-IMM-TEST-004 (TC-IMM-016) |
| Cancer detection accuracy | >98% | TASK-IMM-TEST-004 (TC-IMM-017) |
| Poisoning detection accuracy | >97% | TASK-IMM-TEST-004 (TC-IMM-018) |
| Drift detection accuracy | >95% | TASK-IMM-TEST-004 (TC-IMM-019) |
| Drift detection window | <1s | TASK-IMM-TEST-003 (TC-IMM-009-011) |
| Homeostatic recovery | <10s | TASK-IMM-TEST-004 (TC-IMM-022) |
| Memory overhead | <5MB/session | TASK-IMM-TEST-004 (TC-IMM-021) |
| Cache hit rate | >80% | TASK-IMM-TEST-002 (TC-IMM-007) |
| Unit test coverage | >90% | All TASK-IMM-TEST-* |
| Integration test coverage | >80% | TASK-IMM-TEST-004 |

---

*Document generated: 2025-12-31*
*Task Specification Version: 1.0*
*Module: Immune System (Phase 10)*
*Total Tasks: 24 (20 implementation + 4 test)*
