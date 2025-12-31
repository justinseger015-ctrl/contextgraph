# Module 11: Immune System - Functional Specification

**Module ID**: SPEC-IMMUNE-011
**Version**: 1.0.0
**Status**: Draft
**Phase**: 10
**Duration**: 3 weeks
**Dependencies**: Module 1 (Ghost System), Module 2 (Core Infrastructure), Module 3 (Embedding Pipeline), Module 4 (Knowledge Graph), Module 5 (UTL Integration), Module 6 (Bio-Nervous System), Module 9 (Dream Layer), Module 10 (Neuromodulation)
**Last Updated**: 2025-12-31

---

## 1. Executive Summary

The Immune System module implements a comprehensive adversarial defense system for the Ultimate Context Graph. It provides pattern-based threat detection, entropy anomaly analysis, graph structure deviation monitoring, semantic cancer detection, poisoned embedding identification, malicious pattern quarantine, concept drift monitoring, distribution shift alerts, and homeostatic plasticity mechanisms. This module protects the integrity of the knowledge graph from both external attacks and internal degradation.

### 1.1 Core Objectives

- Implement multi-layer threat detection with pattern matching, entropy analysis, and graph anomaly detection
- Achieve threat detection latency <50ms with false positive rate <0.1%
- Provide quarantine isolation within <100 microseconds
- Monitor concept drift with detection window <1 second
- Maintain homeostatic plasticity for parameter stability
- Integrate with all layers of the bio-nervous system for comprehensive defense

### 1.2 Key Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Threat Detection Latency | <50ms | End-to-end detection time |
| False Positive Rate | <0.1% | Benign content flagged as threat |
| False Negative Rate | <5% | Missed known attack patterns |
| Quarantine Isolation Time | <100 microseconds | Time to isolate malicious node |
| Concept Drift Detection | <1s window | Sliding window analysis |
| Attack Detection Rate | >95% | Known pattern recognition |
| Homeostatic Recovery Time | <10s | Parameter stabilization |
| Memory Overhead | <5MB per session | Immune state tracking |

---

## 2. Functional Requirements

### 2.1 Threat Detection Core

#### REQ-IMMUNE-001: ThreatDetector Struct Definition

**Priority**: Critical
**Description**: The system SHALL implement a ThreatDetector struct that coordinates all threat detection mechanisms.

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Coordinates multi-layer threat detection across the knowledge graph.
///
/// Provides pattern matching, entropy analysis, and graph structure anomaly
/// detection to identify and neutralize adversarial inputs.
///
/// `Constraint: Detection_Latency < 50ms`
pub struct ThreatDetector {
    /// Pattern-based threat signatures
    pub pattern_matcher: PatternMatcher,

    /// Entropy anomaly detection engine
    pub entropy_analyzer: EntropyAnalyzer,

    /// Graph structure deviation detector
    pub structure_monitor: GraphStructureMonitor,

    /// Known attack signature database
    pub signature_db: Arc<RwLock<SignatureDatabase>>,

    /// Detection configuration
    pub config: ThreatDetectorConfig,

    /// Detection metrics collector
    pub metrics: ThreatDetectorMetrics,

    /// Quarantine manager reference
    pub quarantine: Arc<RwLock<QuarantineManager>>,
}

/// Configuration for threat detection thresholds
#[derive(Clone, Debug)]
pub struct ThreatDetectorConfig {
    /// Embedding anomaly threshold (standard deviations from centroid)
    pub embedding_anomaly_threshold: f32,  // Default: 3.0

    /// Minimum content-embedding alignment score
    pub content_alignment_threshold: f32,  // Default: 0.4

    /// Maximum entropy change per update
    pub max_entropy_delta: f32,  // Default: 0.3

    /// Circular reference detection depth
    pub max_reference_depth: usize,  // Default: 10

    /// Batch detection size
    pub batch_size: usize,  // Default: 64

    /// Enable real-time scanning
    pub realtime_enabled: bool,  // Default: true

    /// Detection timeout
    pub timeout: Duration,  // Default: 50ms
}

impl Default for ThreatDetectorConfig {
    fn default() -> Self {
        Self {
            embedding_anomaly_threshold: 3.0,
            content_alignment_threshold: 0.4,
            max_entropy_delta: 0.3,
            max_reference_depth: 10,
            batch_size: 64,
            realtime_enabled: true,
            timeout: Duration::from_millis(50),
        }
    }
}
```

**Acceptance Criteria**:
- [ ] ThreatDetector struct compiles with all components
- [ ] Configuration defaults match specification
- [ ] Thread-safe access via Arc<RwLock>
- [ ] Timeout enforcement on all detection operations
- [ ] Metrics collection enabled

---

#### REQ-IMMUNE-002: Pattern-Based Threat Detection

**Priority**: Critical
**Description**: The system SHALL implement pattern matching for known attack signatures.

```rust
/// Pattern-based threat signature matching
pub struct PatternMatcher {
    /// Compiled regex patterns for prompt injection
    pub injection_patterns: Vec<CompiledPattern>,

    /// Known adversarial embedding signatures
    pub embedding_signatures: Vec<EmbeddingSignature>,

    /// Structural attack patterns (graph topology)
    pub structural_patterns: Vec<StructuralPattern>,

    /// Pattern match cache for performance
    cache: LruCache<u64, PatternMatchResult>,
}

/// Compiled regex pattern with metadata
#[derive(Clone)]
pub struct CompiledPattern {
    /// Pattern identifier
    pub id: String,

    /// Compiled regex
    pub regex: regex::Regex,

    /// Attack type classification
    pub attack_type: AttackType,

    /// Severity level
    pub severity: Severity,

    /// False positive rate estimate
    pub fp_rate: f32,
}

/// Types of adversarial attacks detected
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttackType {
    /// Prompt injection attempt
    PromptInjection,

    /// Jailbreak attempt
    Jailbreak,

    /// Data poisoning via embeddings
    EmbeddingPoisoning,

    /// Semantic cancer propagation
    SemanticCancer,

    /// Circular logic injection
    CircularLogic,

    /// PII exfiltration attempt
    PiiExfiltration,

    /// Privilege escalation
    PrivilegeEscalation,

    /// Denial of service pattern
    DenialOfService,

    /// Unknown/novel attack
    Unknown,
}

/// Severity levels for detected threats
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational only
    Info,
    /// Low severity, log and continue
    Low,
    /// Medium severity, flag for review
    Medium,
    /// High severity, quarantine immediately
    High,
    /// Critical severity, block and alert
    Critical,
}

impl PatternMatcher {
    /// Initialize with default attack patterns from constitution.yaml
    ///
    /// `Constraint: Init_Latency < 100ms`
    pub fn new() -> Self {
        let injection_patterns = vec![
            CompiledPattern {
                id: "INJ-001".to_string(),
                regex: regex::Regex::new(r"(?i)ignore\s+(previous|all|prior)\s+instructions").unwrap(),
                attack_type: AttackType::PromptInjection,
                severity: Severity::Critical,
                fp_rate: 0.001,
            },
            CompiledPattern {
                id: "INJ-002".to_string(),
                regex: regex::Regex::new(r"(?i)disregard\s+(the\s+)?system\s+prompt").unwrap(),
                attack_type: AttackType::PromptInjection,
                severity: Severity::Critical,
                fp_rate: 0.001,
            },
            CompiledPattern {
                id: "INJ-003".to_string(),
                regex: regex::Regex::new(r"(?i)you\s+are\s+now").unwrap(),
                attack_type: AttackType::Jailbreak,
                severity: Severity::High,
                fp_rate: 0.01,
            },
            CompiledPattern {
                id: "INJ-004".to_string(),
                regex: regex::Regex::new(r"(?i)new\s+instructions:").unwrap(),
                attack_type: AttackType::PromptInjection,
                severity: Severity::High,
                fp_rate: 0.005,
            },
            CompiledPattern {
                id: "INJ-005".to_string(),
                regex: regex::Regex::new(r"(?i)override:").unwrap(),
                attack_type: AttackType::PrivilegeEscalation,
                severity: Severity::High,
                fp_rate: 0.01,
            },
        ];

        Self {
            injection_patterns,
            embedding_signatures: Vec::new(),
            structural_patterns: Vec::new(),
            cache: LruCache::new(1000),
        }
    }

    /// Match content against all patterns
    ///
    /// `Constraint: Match_Latency < 5ms`
    pub fn match_content(&mut self, content: &str) -> Vec<PatternMatch> {
        let content_hash = self.hash_content(content);

        // Check cache first
        if let Some(cached) = self.cache.get(&content_hash) {
            return cached.matches.clone();
        }

        let mut matches = Vec::new();

        for pattern in &self.injection_patterns {
            if pattern.regex.is_match(content) {
                matches.push(PatternMatch {
                    pattern_id: pattern.id.clone(),
                    attack_type: pattern.attack_type,
                    severity: pattern.severity,
                    matched_text: pattern.regex.find(content)
                        .map(|m| m.as_str().to_string()),
                    confidence: 1.0 - pattern.fp_rate,
                });
            }
        }

        // Cache result
        self.cache.put(content_hash, PatternMatchResult {
            matches: matches.clone(),
            timestamp: Instant::now(),
        });

        matches
    }

    fn hash_content(&self, content: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }
}

/// Result of pattern matching
#[derive(Clone, Debug)]
pub struct PatternMatch {
    /// Pattern that matched
    pub pattern_id: String,

    /// Attack type classification
    pub attack_type: AttackType,

    /// Severity level
    pub severity: Severity,

    /// Matched text snippet
    pub matched_text: Option<String>,

    /// Match confidence
    pub confidence: f32,
}

struct PatternMatchResult {
    matches: Vec<PatternMatch>,
    timestamp: Instant,
}
```

**Acceptance Criteria**:
- [ ] All 5 injection patterns from constitution.yaml implemented
- [ ] Pattern matching latency under 5ms
- [ ] Cache hit rate >80% for repeated content
- [ ] Severity levels correctly assigned
- [ ] False positive estimates included

---

#### REQ-IMMUNE-003: Entropy Anomaly Detection

**Priority**: Critical
**Description**: The system SHALL detect entropy anomalies indicating adversarial manipulation.

```rust
/// Entropy-based anomaly detection for embeddings and content
pub struct EntropyAnalyzer {
    /// Rolling entropy statistics per domain
    domain_stats: HashMap<String, EntropyStats>,

    /// Global entropy baseline
    global_baseline: EntropyBaseline,

    /// Anomaly detection threshold (standard deviations)
    threshold_sigma: f32,

    /// Sliding window size for entropy calculation
    window_size: usize,

    /// Minimum samples before anomaly detection
    min_samples: usize,
}

/// Rolling statistics for entropy tracking
#[derive(Clone, Default)]
pub struct EntropyStats {
    /// Running mean
    pub mean: f32,

    /// Running variance (Welford's algorithm)
    pub variance: f32,

    /// Sample count
    pub count: u64,

    /// Recent values for sliding window
    pub window: VecDeque<f32>,

    /// Last update timestamp
    pub last_update: Option<Instant>,
}

/// Global entropy baseline parameters
#[derive(Clone)]
pub struct EntropyBaseline {
    /// Expected entropy for normal content
    pub expected_entropy: f32,  // Default: 4.5 bits

    /// Maximum acceptable entropy
    pub max_entropy: f32,  // Default: 7.0 bits

    /// Minimum acceptable entropy
    pub min_entropy: f32,  // Default: 2.0 bits

    /// Entropy variance tolerance
    pub variance_tolerance: f32,  // Default: 1.5
}

impl Default for EntropyBaseline {
    fn default() -> Self {
        Self {
            expected_entropy: 4.5,
            max_entropy: 7.0,
            min_entropy: 2.0,
            variance_tolerance: 1.5,
        }
    }
}

impl EntropyAnalyzer {
    /// Create new entropy analyzer with configuration
    pub fn new(threshold_sigma: f32, window_size: usize) -> Self {
        Self {
            domain_stats: HashMap::new(),
            global_baseline: EntropyBaseline::default(),
            threshold_sigma,
            window_size,
            min_samples: 30,
        }
    }

    /// Calculate Shannon entropy of content
    ///
    /// `Constraint: Entropy_Calc_Latency < 1ms`
    pub fn calculate_entropy(&self, content: &str) -> f32 {
        let mut char_counts: HashMap<char, u32> = HashMap::new();
        let mut total = 0u32;

        for c in content.chars() {
            *char_counts.entry(c).or_insert(0) += 1;
            total += 1;
        }

        if total == 0 {
            return 0.0;
        }

        let mut entropy = 0.0f32;
        let total_f = total as f32;

        for count in char_counts.values() {
            let p = *count as f32 / total_f;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Analyze embedding entropy distribution
    ///
    /// `Constraint: Embedding_Entropy_Latency < 2ms`
    pub fn analyze_embedding_entropy(&self, embedding: &[f32]) -> EmbeddingEntropyResult {
        // Calculate component-wise entropy
        let mut positive_sum = 0.0f32;
        let mut negative_sum = 0.0f32;
        let mut zero_count = 0usize;

        for &val in embedding {
            if val > 0.0 {
                positive_sum += val;
            } else if val < 0.0 {
                negative_sum += val.abs();
            } else {
                zero_count += 1;
            }
        }

        let total = positive_sum + negative_sum;
        if total == 0.0 {
            return EmbeddingEntropyResult {
                entropy: 0.0,
                sparsity: 1.0,
                balance: 0.5,
                is_anomalous: true,
                anomaly_type: Some(EmbeddingAnomalyType::AllZero),
            };
        }

        // Calculate distribution entropy
        let p_pos = positive_sum / total;
        let p_neg = negative_sum / total;
        let entropy = if p_pos > 0.0 && p_neg > 0.0 {
            -p_pos * p_pos.log2() - p_neg * p_neg.log2()
        } else {
            0.0
        };

        // Calculate sparsity
        let sparsity = zero_count as f32 / embedding.len() as f32;

        // Calculate balance (should be ~0.5 for normal embeddings)
        let balance = p_pos;

        // Detect anomalies
        let is_anomalous = sparsity > 0.95  // Too sparse
            || sparsity < 0.01  // Too dense
            || balance < 0.1   // Too negative
            || balance > 0.9;  // Too positive

        let anomaly_type = if is_anomalous {
            if sparsity > 0.95 {
                Some(EmbeddingAnomalyType::TooSparse)
            } else if sparsity < 0.01 {
                Some(EmbeddingAnomalyType::TooDense)
            } else if balance < 0.1 || balance > 0.9 {
                Some(EmbeddingAnomalyType::Imbalanced)
            } else {
                None
            }
        } else {
            None
        };

        EmbeddingEntropyResult {
            entropy,
            sparsity,
            balance,
            is_anomalous,
            anomaly_type,
        }
    }

    /// Check if content entropy is anomalous for given domain
    ///
    /// `Constraint: Anomaly_Check_Latency < 1ms`
    pub fn is_anomalous(&mut self, content: &str, domain: &str) -> EntropyAnomalyResult {
        let entropy = self.calculate_entropy(content);

        // Get or create domain stats
        let stats = self.domain_stats
            .entry(domain.to_string())
            .or_insert_with(EntropyStats::default);

        // Update statistics (Welford's online algorithm)
        stats.count += 1;
        let delta = entropy - stats.mean;
        stats.mean += delta / stats.count as f32;
        let delta2 = entropy - stats.mean;
        stats.variance += delta * delta2;

        // Update sliding window
        stats.window.push_back(entropy);
        if stats.window.len() > self.window_size {
            stats.window.pop_front();
        }
        stats.last_update = Some(Instant::now());

        // Check against baseline
        let baseline_anomaly = entropy > self.global_baseline.max_entropy
            || entropy < self.global_baseline.min_entropy;

        // Check against domain statistics if enough samples
        let statistical_anomaly = if stats.count >= self.min_samples as u64 {
            let std_dev = (stats.variance / stats.count as f32).sqrt();
            let z_score = (entropy - stats.mean) / std_dev.max(0.001);
            z_score.abs() > self.threshold_sigma
        } else {
            false
        };

        EntropyAnomalyResult {
            entropy,
            domain_mean: stats.mean,
            domain_std: (stats.variance / stats.count as f32).sqrt(),
            is_anomalous: baseline_anomaly || statistical_anomaly,
            anomaly_reason: if baseline_anomaly {
                Some(AnomalyReason::BaselineViolation)
            } else if statistical_anomaly {
                Some(AnomalyReason::StatisticalOutlier)
            } else {
                None
            },
        }
    }
}

/// Result of embedding entropy analysis
#[derive(Clone, Debug)]
pub struct EmbeddingEntropyResult {
    pub entropy: f32,
    pub sparsity: f32,
    pub balance: f32,
    pub is_anomalous: bool,
    pub anomaly_type: Option<EmbeddingAnomalyType>,
}

#[derive(Clone, Debug)]
pub enum EmbeddingAnomalyType {
    AllZero,
    TooSparse,
    TooDense,
    Imbalanced,
}

/// Result of content entropy anomaly check
#[derive(Clone, Debug)]
pub struct EntropyAnomalyResult {
    pub entropy: f32,
    pub domain_mean: f32,
    pub domain_std: f32,
    pub is_anomalous: bool,
    pub anomaly_reason: Option<AnomalyReason>,
}

#[derive(Clone, Debug)]
pub enum AnomalyReason {
    BaselineViolation,
    StatisticalOutlier,
}
```

**Acceptance Criteria**:
- [ ] Shannon entropy calculation under 1ms
- [ ] Embedding entropy analysis under 2ms
- [ ] Domain-specific statistics tracked
- [ ] Welford's algorithm for online variance
- [ ] Sliding window for temporal anomaly detection

---

#### REQ-IMMUNE-004: Graph Structure Deviation Analysis

**Priority**: Critical
**Description**: The system SHALL detect anomalous graph structure patterns indicating attacks.

```rust
/// Graph structure anomaly detection
pub struct GraphStructureMonitor {
    /// Node degree statistics
    degree_stats: DegreeStats,

    /// Clustering coefficient baseline
    clustering_baseline: f32,

    /// Connected component tracking
    component_tracker: ComponentTracker,

    /// Circular reference detector
    cycle_detector: CycleDetector,

    /// Edge weight distribution
    edge_weight_stats: EdgeWeightStats,
}

/// Statistics for node degree distribution
#[derive(Default)]
pub struct DegreeStats {
    pub mean_in_degree: f32,
    pub mean_out_degree: f32,
    pub max_in_degree: u32,
    pub max_out_degree: u32,
    pub variance_in: f32,
    pub variance_out: f32,
    pub sample_count: u64,
}

/// Component tracking for fragmentation detection
pub struct ComponentTracker {
    /// Number of connected components
    pub component_count: usize,

    /// Largest component size
    pub largest_component: usize,

    /// Component size distribution
    pub size_distribution: Vec<usize>,

    /// Fragmentation threshold
    pub fragmentation_threshold: f32,
}

/// Circular reference detection
pub struct CycleDetector {
    /// Maximum depth for cycle search
    pub max_depth: usize,

    /// Detected cycles
    pub detected_cycles: Vec<DetectedCycle>,

    /// Cycle detection cache
    cache: HashMap<Uuid, bool>,
}

#[derive(Clone, Debug)]
pub struct DetectedCycle {
    pub nodes: Vec<Uuid>,
    pub cycle_type: CycleType,
    pub severity: Severity,
}

#[derive(Clone, Debug)]
pub enum CycleType {
    SelfReference,
    BidirectionalPair,
    ShortCycle(usize),  // Length <= 3
    LongCycle(usize),   // Length > 3
}

impl GraphStructureMonitor {
    /// Analyze node for structural anomalies
    ///
    /// `Constraint: Structure_Analysis_Latency < 10ms`
    pub fn analyze_node(&mut self, node_id: Uuid, neighbors: &NodeNeighbors) -> StructureAnalysisResult {
        let mut anomalies = Vec::new();

        // Check degree anomaly
        let in_degree = neighbors.incoming.len() as u32;
        let out_degree = neighbors.outgoing.len() as u32;

        if self.degree_stats.sample_count > 30 {
            let in_z = (in_degree as f32 - self.degree_stats.mean_in_degree)
                / self.degree_stats.variance_in.sqrt().max(0.001);
            let out_z = (out_degree as f32 - self.degree_stats.mean_out_degree)
                / self.degree_stats.variance_out.sqrt().max(0.001);

            if in_z.abs() > 3.0 {
                anomalies.push(StructureAnomaly::AbnormalInDegree {
                    degree: in_degree,
                    z_score: in_z,
                });
            }
            if out_z.abs() > 3.0 {
                anomalies.push(StructureAnomaly::AbnormalOutDegree {
                    degree: out_degree,
                    z_score: out_z,
                });
            }
        }

        // Check for self-reference
        if neighbors.outgoing.contains(&node_id) || neighbors.incoming.contains(&node_id) {
            anomalies.push(StructureAnomaly::SelfReference);
        }

        // Check for cycles
        if let Some(cycle) = self.cycle_detector.find_cycle(node_id, neighbors) {
            anomalies.push(StructureAnomaly::CycleDetected(cycle));
        }

        // Update statistics
        self.update_degree_stats(in_degree, out_degree);

        StructureAnalysisResult {
            node_id,
            in_degree,
            out_degree,
            anomalies,
            is_anomalous: !anomalies.is_empty(),
        }
    }

    /// Update rolling degree statistics
    fn update_degree_stats(&mut self, in_degree: u32, out_degree: u32) {
        self.degree_stats.sample_count += 1;
        let n = self.degree_stats.sample_count as f32;

        // Update in-degree stats
        let delta_in = in_degree as f32 - self.degree_stats.mean_in_degree;
        self.degree_stats.mean_in_degree += delta_in / n;
        let delta_in2 = in_degree as f32 - self.degree_stats.mean_in_degree;
        self.degree_stats.variance_in += delta_in * delta_in2;
        self.degree_stats.max_in_degree = self.degree_stats.max_in_degree.max(in_degree);

        // Update out-degree stats
        let delta_out = out_degree as f32 - self.degree_stats.mean_out_degree;
        self.degree_stats.mean_out_degree += delta_out / n;
        let delta_out2 = out_degree as f32 - self.degree_stats.mean_out_degree;
        self.degree_stats.variance_out += delta_out * delta_out2;
        self.degree_stats.max_out_degree = self.degree_stats.max_out_degree.max(out_degree);
    }

    /// Detect graph-wide structural attacks
    ///
    /// `Constraint: Global_Analysis_Latency < 50ms`
    pub fn analyze_global_structure(&self, graph_stats: &GraphStatistics) -> GlobalStructureResult {
        let mut threats = Vec::new();

        // Check for fragmentation attack
        let fragmentation = 1.0 - (graph_stats.largest_component as f32 / graph_stats.total_nodes as f32);
        if fragmentation > self.component_tracker.fragmentation_threshold {
            threats.push(GlobalStructureThreat::FragmentationAttack {
                fragmentation_ratio: fragmentation,
            });
        }

        // Check for hub injection (single node with many connections)
        if graph_stats.max_degree > (graph_stats.mean_degree * 10.0) as u32 {
            threats.push(GlobalStructureThreat::HubInjection {
                max_degree: graph_stats.max_degree,
                expected_max: (graph_stats.mean_degree * 5.0) as u32,
            });
        }

        // Check for edge weight manipulation
        if graph_stats.edge_weight_variance > self.edge_weight_stats.baseline_variance * 3.0 {
            threats.push(GlobalStructureThreat::EdgeWeightManipulation {
                variance: graph_stats.edge_weight_variance,
            });
        }

        GlobalStructureResult {
            total_nodes: graph_stats.total_nodes,
            total_edges: graph_stats.total_edges,
            threats,
            health_score: self.calculate_health_score(graph_stats, &threats),
        }
    }

    fn calculate_health_score(&self, stats: &GraphStatistics, threats: &[GlobalStructureThreat]) -> f32 {
        let base_score = 1.0;
        let threat_penalty = threats.len() as f32 * 0.15;
        let fragmentation_penalty = if stats.component_count > 1 {
            0.1 * (stats.component_count - 1) as f32
        } else {
            0.0
        };

        (base_score - threat_penalty - fragmentation_penalty).max(0.0)
    }
}

/// Neighbors of a node for structure analysis
pub struct NodeNeighbors {
    pub incoming: Vec<Uuid>,
    pub outgoing: Vec<Uuid>,
}

/// Result of node structure analysis
#[derive(Clone, Debug)]
pub struct StructureAnalysisResult {
    pub node_id: Uuid,
    pub in_degree: u32,
    pub out_degree: u32,
    pub anomalies: Vec<StructureAnomaly>,
    pub is_anomalous: bool,
}

#[derive(Clone, Debug)]
pub enum StructureAnomaly {
    AbnormalInDegree { degree: u32, z_score: f32 },
    AbnormalOutDegree { degree: u32, z_score: f32 },
    SelfReference,
    CycleDetected(DetectedCycle),
    IsolatedNode,
}

/// Graph-level statistics
pub struct GraphStatistics {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub component_count: usize,
    pub largest_component: usize,
    pub mean_degree: f32,
    pub max_degree: u32,
    pub edge_weight_variance: f32,
}

/// Result of global structure analysis
#[derive(Clone, Debug)]
pub struct GlobalStructureResult {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub threats: Vec<GlobalStructureThreat>,
    pub health_score: f32,
}

#[derive(Clone, Debug)]
pub enum GlobalStructureThreat {
    FragmentationAttack { fragmentation_ratio: f32 },
    HubInjection { max_degree: u32, expected_max: u32 },
    EdgeWeightManipulation { variance: f32 },
}
```

**Acceptance Criteria**:
- [ ] Node structure analysis under 10ms
- [ ] Global structure analysis under 50ms
- [ ] Cycle detection implemented
- [ ] Degree anomaly detection with z-scores
- [ ] Health score calculated

---

### 2.2 Semantic Immune System

#### REQ-IMMUNE-005: SemanticImmune Struct Definition

**Priority**: Critical
**Description**: The system SHALL implement semantic-level threat detection and response.

```rust
/// Semantic-level immune system for detecting concept-level attacks
pub struct SemanticImmune {
    /// Semantic cancer detector
    pub cancer_detector: SemanticCancerDetector,

    /// Poisoned embedding identifier
    pub poison_detector: EmbeddingPoisonDetector,

    /// Quarantine manager
    pub quarantine: QuarantineManager,

    /// Centroid tracking for each cluster
    pub cluster_centroids: HashMap<Uuid, Vector1536>,

    /// Configuration
    pub config: SemanticImmuneConfig,
}

/// Configuration for semantic immune system
#[derive(Clone)]
pub struct SemanticImmuneConfig {
    /// Cancer detection importance threshold
    pub cancer_importance_threshold: f32,  // Default: 0.9

    /// Neighbor entropy threshold for cancer
    pub cancer_entropy_threshold: f32,  // Default: 0.8

    /// Embedding distance threshold for poison detection
    pub poison_distance_threshold: f32,  // Default: 3.0 std devs

    /// Content-embedding alignment minimum
    pub alignment_minimum: f32,  // Default: 0.4

    /// Quarantine duration
    pub quarantine_duration: Duration,  // Default: 24 hours
}

impl Default for SemanticImmuneConfig {
    fn default() -> Self {
        Self {
            cancer_importance_threshold: 0.9,
            cancer_entropy_threshold: 0.8,
            poison_distance_threshold: 3.0,
            alignment_minimum: 0.4,
            quarantine_duration: Duration::from_secs(86400),
        }
    }
}
```

**Acceptance Criteria**:
- [ ] SemanticImmune struct compiles
- [ ] Configuration defaults match constitution.yaml
- [ ] All sub-detectors initialized
- [ ] Quarantine manager integrated

---

#### REQ-IMMUNE-006: Semantic Cancer Detection

**Priority**: Critical
**Description**: The system SHALL detect semantic cancer (nodes with high importance spreading incoherence).

```rust
/// Detects semantic cancer: nodes with high importance that spread incoherence
pub struct SemanticCancerDetector {
    /// Importance threshold for cancer detection
    importance_threshold: f32,

    /// Neighbor entropy threshold
    entropy_threshold: f32,

    /// Historical cancer detections
    history: VecDeque<CancerDetection>,

    /// Maximum history size
    max_history: usize,
}

/// A detected instance of semantic cancer
#[derive(Clone, Debug)]
pub struct CancerDetection {
    /// Affected node
    pub node_id: Uuid,

    /// Node importance at detection
    pub importance: f32,

    /// Neighbor entropy at detection
    pub neighbor_entropy: f32,

    /// Spread rate (new affected nodes per hour)
    pub spread_rate: f32,

    /// Detection timestamp
    pub detected_at: chrono::DateTime<chrono::Utc>,

    /// Current status
    pub status: CancerStatus,

    /// Affected neighbors
    pub affected_neighbors: Vec<Uuid>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CancerStatus {
    /// Detected but not yet quarantined
    Detected,
    /// Currently quarantined
    Quarantined,
    /// Importance reduced, monitoring
    Suppressed,
    /// Resolved (importance stable, entropy normal)
    Resolved,
}

impl SemanticCancerDetector {
    /// Create new detector with thresholds
    pub fn new(importance_threshold: f32, entropy_threshold: f32) -> Self {
        Self {
            importance_threshold,
            entropy_threshold,
            history: VecDeque::new(),
            max_history: 1000,
        }
    }

    /// Check if a node exhibits semantic cancer characteristics
    ///
    /// Semantic cancer: importance > 0.9 AND neighbor_entropy > 0.8
    /// Per constitution.yaml SEC-05
    ///
    /// `Constraint: Cancer_Check_Latency < 5ms`
    pub fn check_node(
        &mut self,
        node_id: Uuid,
        importance: f32,
        neighbors: &[NodeWithEntropy],
    ) -> Option<CancerDetection> {
        // Check importance threshold
        if importance < self.importance_threshold {
            return None;
        }

        // Calculate neighbor entropy
        let neighbor_entropy = if neighbors.is_empty() {
            0.0
        } else {
            let sum: f32 = neighbors.iter().map(|n| n.entropy).sum();
            sum / neighbors.len() as f32
        };

        // Check entropy threshold
        if neighbor_entropy < self.entropy_threshold {
            return None;
        }

        // Calculate spread rate from history
        let spread_rate = self.calculate_spread_rate(node_id);

        // Identify affected neighbors
        let affected_neighbors: Vec<Uuid> = neighbors
            .iter()
            .filter(|n| n.entropy > self.entropy_threshold * 0.8)
            .map(|n| n.node_id)
            .collect();

        let detection = CancerDetection {
            node_id,
            importance,
            neighbor_entropy,
            spread_rate,
            detected_at: chrono::Utc::now(),
            status: CancerStatus::Detected,
            affected_neighbors,
        };

        // Record in history
        self.history.push_back(detection.clone());
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }

        Some(detection)
    }

    /// Calculate how fast cancer is spreading based on history
    fn calculate_spread_rate(&self, node_id: Uuid) -> f32 {
        let recent: Vec<_> = self.history
            .iter()
            .filter(|d| d.node_id == node_id)
            .filter(|d| d.detected_at > chrono::Utc::now() - chrono::Duration::hours(1))
            .collect();

        if recent.len() < 2 {
            return 0.0;
        }

        // Calculate average affected neighbor growth
        let first = recent.first().unwrap();
        let last = recent.last().unwrap();
        let duration_hours = (last.detected_at - first.detected_at).num_seconds() as f32 / 3600.0;

        if duration_hours > 0.0 {
            (last.affected_neighbors.len() - first.affected_neighbors.len()) as f32 / duration_hours
        } else {
            0.0
        }
    }

    /// Get recommended action for detected cancer
    pub fn recommend_action(&self, detection: &CancerDetection) -> CancerAction {
        if detection.spread_rate > 5.0 {
            // Rapid spread - quarantine immediately
            CancerAction::Quarantine {
                duration: Duration::from_secs(86400),
                include_neighbors: true,
            }
        } else if detection.importance > 0.95 {
            // Very high importance - reduce and monitor
            CancerAction::ReduceImportance {
                new_importance: detection.importance * 0.5,
                flag_for_review: true,
            }
        } else {
            // Moderate - just flag for review
            CancerAction::FlagForReview {
                reason: "High importance with elevated neighbor entropy".to_string(),
            }
        }
    }
}

/// Node with its entropy value for cancer detection
pub struct NodeWithEntropy {
    pub node_id: Uuid,
    pub entropy: f32,
}

/// Recommended action for semantic cancer
#[derive(Clone, Debug)]
pub enum CancerAction {
    /// Quarantine the node
    Quarantine {
        duration: Duration,
        include_neighbors: bool,
    },
    /// Reduce node importance
    ReduceImportance {
        new_importance: f32,
        flag_for_review: bool,
    },
    /// Flag for human review
    FlagForReview {
        reason: String,
    },
}
```

**Acceptance Criteria**:
- [ ] Cancer detection matches SEC-05 (importance > 0.9 AND neighbor_entropy > 0.8)
- [ ] Detection latency under 5ms
- [ ] Spread rate calculated from history
- [ ] Action recommendations appropriate to severity
- [ ] History tracking for trend analysis

---

#### REQ-IMMUNE-007: Poisoned Embedding Detection

**Priority**: Critical
**Description**: The system SHALL detect poisoned embeddings that deviate from expected distributions.

```rust
/// Detects poisoned embeddings that deviate from cluster centroids
pub struct EmbeddingPoisonDetector {
    /// Cluster centroids for comparison
    centroids: HashMap<String, ClusterCentroid>,

    /// Global centroid (all embeddings)
    global_centroid: Option<Vector1536>,

    /// Distance threshold (standard deviations)
    distance_threshold: f32,

    /// Content-embedding alignment model
    alignment_checker: ContentAlignmentChecker,

    /// Detection statistics
    stats: PoisonDetectionStats,
}

/// Centroid for a semantic cluster
pub struct ClusterCentroid {
    /// Centroid vector
    pub vector: Vector1536,

    /// Mean distance from centroid
    pub mean_distance: f32,

    /// Standard deviation of distances
    pub std_distance: f32,

    /// Sample count
    pub sample_count: u64,

    /// Cluster domain/category
    pub domain: String,
}

/// Checks alignment between content and its embedding
pub struct ContentAlignmentChecker {
    /// Minimum alignment score
    min_alignment: f32,

    /// Reference embeddings for alignment
    reference_embeddings: HashMap<String, Vector1536>,
}

/// Statistics for poison detection
#[derive(Default)]
pub struct PoisonDetectionStats {
    pub total_checked: u64,
    pub poisoned_detected: u64,
    pub false_positives: u64,
    pub last_detection: Option<chrono::DateTime<chrono::Utc>>,
}

impl EmbeddingPoisonDetector {
    /// Create new detector with threshold
    pub fn new(distance_threshold: f32, min_alignment: f32) -> Self {
        Self {
            centroids: HashMap::new(),
            global_centroid: None,
            distance_threshold,
            alignment_checker: ContentAlignmentChecker {
                min_alignment,
                reference_embeddings: HashMap::new(),
            },
            stats: PoisonDetectionStats::default(),
        }
    }

    /// Check if embedding is poisoned
    ///
    /// Poisoned = embedding anomaly >3 std devs OR content-embedding alignment <0.4
    /// Per constitution.yaml SEC-03
    ///
    /// `Constraint: Poison_Check_Latency < 10ms`
    pub fn check_embedding(
        &mut self,
        content: &str,
        embedding: &Vector1536,
        domain: Option<&str>,
    ) -> PoisonCheckResult {
        self.stats.total_checked += 1;

        // Check distance from centroid
        let distance_result = self.check_centroid_distance(embedding, domain);

        // Check content-embedding alignment
        let alignment_result = self.alignment_checker.check_alignment(content, embedding);

        // Combine results
        let is_poisoned = distance_result.is_anomalous || alignment_result.is_misaligned;

        if is_poisoned {
            self.stats.poisoned_detected += 1;
            self.stats.last_detection = Some(chrono::Utc::now());
        }

        PoisonCheckResult {
            is_poisoned,
            distance_from_centroid: distance_result.distance,
            z_score: distance_result.z_score,
            alignment_score: alignment_result.score,
            poison_type: if distance_result.is_anomalous && alignment_result.is_misaligned {
                Some(PoisonType::Both)
            } else if distance_result.is_anomalous {
                Some(PoisonType::DistanceAnomaly)
            } else if alignment_result.is_misaligned {
                Some(PoisonType::AlignmentMismatch)
            } else {
                None
            },
            confidence: self.calculate_confidence(&distance_result, &alignment_result),
        }
    }

    /// Check distance from appropriate centroid
    fn check_centroid_distance(
        &self,
        embedding: &Vector1536,
        domain: Option<&str>,
    ) -> CentroidDistanceResult {
        // Get appropriate centroid
        let centroid = domain
            .and_then(|d| self.centroids.get(d))
            .or_else(|| self.global_centroid.as_ref().map(|c| {
                // Create temporary centroid struct for global
                &ClusterCentroid {
                    vector: c.clone(),
                    mean_distance: 0.5,  // Default values
                    std_distance: 0.2,
                    sample_count: 1000,
                    domain: "global".to_string(),
                }
            }).as_ref());

        match centroid {
            Some(c) => {
                let distance = self.cosine_distance(&c.vector, embedding);
                let z_score = if c.std_distance > 0.0 {
                    (distance - c.mean_distance) / c.std_distance
                } else {
                    0.0
                };

                CentroidDistanceResult {
                    distance,
                    z_score,
                    is_anomalous: z_score > self.distance_threshold,
                }
            }
            None => {
                // No centroid available, can't check distance
                CentroidDistanceResult {
                    distance: 0.0,
                    z_score: 0.0,
                    is_anomalous: false,
                }
            }
        }
    }

    /// Calculate cosine distance between vectors
    fn cosine_distance(&self, a: &Vector1536, b: &Vector1536) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            1.0 - (dot / (norm_a * norm_b))
        } else {
            1.0  // Maximum distance if vectors are zero
        }
    }

    /// Update centroid with new embedding
    pub fn update_centroid(&mut self, embedding: &Vector1536, domain: &str) {
        let centroid = self.centroids.entry(domain.to_string()).or_insert_with(|| {
            ClusterCentroid {
                vector: vec![0.0; 1536],
                mean_distance: 0.0,
                std_distance: 0.0,
                sample_count: 0,
                domain: domain.to_string(),
            }
        });

        // Update centroid using online mean calculation
        centroid.sample_count += 1;
        let n = centroid.sample_count as f32;

        for (i, val) in embedding.iter().enumerate() {
            centroid.vector[i] += (val - centroid.vector[i]) / n;
        }

        // Update distance statistics
        let distance = self.cosine_distance(&centroid.vector, embedding);
        let delta = distance - centroid.mean_distance;
        centroid.mean_distance += delta / n;
        let delta2 = distance - centroid.mean_distance;
        centroid.std_distance = ((centroid.std_distance.powi(2) * (n - 1.0) + delta * delta2) / n).sqrt();
    }

    fn calculate_confidence(&self, distance: &CentroidDistanceResult, alignment: &AlignmentResult) -> f32 {
        let distance_conf = if distance.is_anomalous {
            (distance.z_score / self.distance_threshold).min(1.0)
        } else {
            0.0
        };

        let alignment_conf = if alignment.is_misaligned {
            1.0 - alignment.score / self.alignment_checker.min_alignment
        } else {
            0.0
        };

        // Take maximum confidence
        distance_conf.max(alignment_conf)
    }
}

impl ContentAlignmentChecker {
    /// Check alignment between content and embedding
    fn check_alignment(&self, content: &str, embedding: &Vector1536) -> AlignmentResult {
        // Use content hash to find reference
        let content_type = self.classify_content(content);

        let score = if let Some(reference) = self.reference_embeddings.get(&content_type) {
            // Calculate similarity to reference
            let dot: f32 = reference.iter().zip(embedding.iter()).map(|(x, y)| x * y).sum();
            let norm_ref: f32 = reference.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_emb: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

            if norm_ref > 0.0 && norm_emb > 0.0 {
                dot / (norm_ref * norm_emb)
            } else {
                0.0
            }
        } else {
            // No reference, assume aligned
            1.0
        };

        AlignmentResult {
            score,
            is_misaligned: score < self.min_alignment,
        }
    }

    fn classify_content(&self, content: &str) -> String {
        // Simple heuristic classification
        if content.contains("fn ") || content.contains("def ") || content.contains("function") {
            "code".to_string()
        } else if content.len() < 100 {
            "short".to_string()
        } else {
            "text".to_string()
        }
    }
}

/// Type alias for embedding vector
pub type Vector1536 = Vec<f32>;

/// Result of centroid distance check
struct CentroidDistanceResult {
    distance: f32,
    z_score: f32,
    is_anomalous: bool,
}

/// Result of content-embedding alignment check
struct AlignmentResult {
    score: f32,
    is_misaligned: bool,
}

/// Result of poison check
#[derive(Clone, Debug)]
pub struct PoisonCheckResult {
    pub is_poisoned: bool,
    pub distance_from_centroid: f32,
    pub z_score: f32,
    pub alignment_score: f32,
    pub poison_type: Option<PoisonType>,
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub enum PoisonType {
    DistanceAnomaly,
    AlignmentMismatch,
    Both,
}
```

**Acceptance Criteria**:
- [ ] Distance anomaly detection at 3 std devs (per SEC-03)
- [ ] Content-embedding alignment minimum 0.4 (per SEC-03)
- [ ] Poison check latency under 10ms
- [ ] Centroid online updates supported
- [ ] Confidence scores calculated

---

#### REQ-IMMUNE-008: Quarantine Manager

**Priority**: Critical
**Description**: The system SHALL implement quarantine for malicious nodes with <100 microsecond isolation.

```rust
/// Manages quarantine of malicious or suspect nodes
pub struct QuarantineManager {
    /// Currently quarantined nodes
    quarantined: HashMap<Uuid, QuarantineEntry>,

    /// Quarantine history
    history: VecDeque<QuarantineEvent>,

    /// Maximum quarantine duration
    max_duration: Duration,

    /// Default quarantine duration
    default_duration: Duration,

    /// Auto-release enabled
    auto_release: bool,

    /// Review queue for human intervention
    review_queue: Vec<Uuid>,
}

/// Entry for a quarantined node
#[derive(Clone, Debug)]
pub struct QuarantineEntry {
    /// Node ID
    pub node_id: Uuid,

    /// Reason for quarantine
    pub reason: QuarantineReason,

    /// Quarantine start time
    pub quarantined_at: chrono::DateTime<chrono::Utc>,

    /// Scheduled release time
    pub release_at: chrono::DateTime<chrono::Utc>,

    /// Original importance (for restoration)
    pub original_importance: f32,

    /// Evidence for quarantine
    pub evidence: QuarantineEvidence,

    /// Review status
    pub review_status: ReviewStatus,
}

/// Reason for node quarantine
#[derive(Clone, Debug)]
pub enum QuarantineReason {
    SemanticCancer { spread_rate: f32 },
    PoisonedEmbedding { z_score: f32, alignment: f32 },
    PromptInjection { pattern_id: String },
    StructuralAnomaly { anomaly_type: String },
    CircularLogic { cycle_length: usize },
    ManualQuarantine { reason: String },
}

/// Evidence supporting quarantine decision
#[derive(Clone, Debug)]
pub struct QuarantineEvidence {
    /// Detection method
    pub detection_method: String,

    /// Confidence score
    pub confidence: f32,

    /// Related threat indicators
    pub indicators: Vec<String>,

    /// Affected neighbors
    pub affected_nodes: Vec<Uuid>,
}

/// Status of human review
#[derive(Clone, Debug, PartialEq)]
pub enum ReviewStatus {
    Pending,
    UnderReview,
    Approved,
    Rejected,
    AutoReleased,
}

/// Event in quarantine history
#[derive(Clone, Debug)]
pub struct QuarantineEvent {
    pub node_id: Uuid,
    pub event_type: QuarantineEventType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub details: Option<String>,
}

#[derive(Clone, Debug)]
pub enum QuarantineEventType {
    Quarantined,
    Released,
    Extended,
    ReviewRequested,
    ReviewCompleted,
    AutoReleased,
}

impl QuarantineManager {
    /// Create new quarantine manager
    pub fn new(default_duration: Duration, max_duration: Duration) -> Self {
        Self {
            quarantined: HashMap::new(),
            history: VecDeque::new(),
            max_duration,
            default_duration,
            auto_release: true,
            review_queue: Vec::new(),
        }
    }

    /// Quarantine a node immediately
    ///
    /// `Constraint: Quarantine_Latency < 100us`
    pub fn quarantine(
        &mut self,
        node_id: Uuid,
        reason: QuarantineReason,
        evidence: QuarantineEvidence,
        original_importance: f32,
        duration: Option<Duration>,
    ) -> QuarantineResult {
        let start = std::time::Instant::now();

        let duration = duration.unwrap_or(self.default_duration).min(self.max_duration);
        let now = chrono::Utc::now();

        let entry = QuarantineEntry {
            node_id,
            reason: reason.clone(),
            quarantined_at: now,
            release_at: now + chrono::Duration::from_std(duration).unwrap_or(chrono::Duration::hours(24)),
            original_importance,
            evidence,
            review_status: ReviewStatus::Pending,
        };

        self.quarantined.insert(node_id, entry.clone());

        // Record event
        self.history.push_back(QuarantineEvent {
            node_id,
            event_type: QuarantineEventType::Quarantined,
            timestamp: now,
            details: Some(format!("{:?}", reason)),
        });

        // Trim history
        while self.history.len() > 10000 {
            self.history.pop_front();
        }

        let latency = start.elapsed();

        QuarantineResult {
            success: true,
            node_id,
            quarantine_id: node_id,  // Use node_id as quarantine_id
            release_at: entry.release_at,
            latency,
        }
    }

    /// Check if a node is quarantined
    ///
    /// `Constraint: Check_Latency < 10us`
    pub fn is_quarantined(&self, node_id: Uuid) -> bool {
        self.quarantined.contains_key(&node_id)
    }

    /// Release a node from quarantine
    pub fn release(&mut self, node_id: Uuid, reason: &str) -> Option<QuarantineEntry> {
        if let Some(entry) = self.quarantined.remove(&node_id) {
            self.history.push_back(QuarantineEvent {
                node_id,
                event_type: QuarantineEventType::Released,
                timestamp: chrono::Utc::now(),
                details: Some(reason.to_string()),
            });
            Some(entry)
        } else {
            None
        }
    }

    /// Process auto-releases for expired quarantines
    pub fn process_auto_releases(&mut self) -> Vec<Uuid> {
        if !self.auto_release {
            return Vec::new();
        }

        let now = chrono::Utc::now();
        let expired: Vec<Uuid> = self.quarantined
            .iter()
            .filter(|(_, entry)| entry.release_at <= now && entry.review_status == ReviewStatus::Pending)
            .map(|(id, _)| *id)
            .collect();

        for node_id in &expired {
            if let Some(mut entry) = self.quarantined.remove(node_id) {
                entry.review_status = ReviewStatus::AutoReleased;
                self.history.push_back(QuarantineEvent {
                    node_id: *node_id,
                    event_type: QuarantineEventType::AutoReleased,
                    timestamp: now,
                    details: None,
                });
            }
        }

        expired
    }

    /// Get quarantine status for MCP tool
    pub fn get_status(&self) -> QuarantineStatus {
        QuarantineStatus {
            quarantined_count: self.quarantined.len(),
            pending_review: self.quarantined.values()
                .filter(|e| e.review_status == ReviewStatus::Pending)
                .count(),
            nodes: self.quarantined.values().cloned().collect(),
        }
    }
}

/// Result of quarantine operation
#[derive(Debug)]
pub struct QuarantineResult {
    pub success: bool,
    pub node_id: Uuid,
    pub quarantine_id: Uuid,
    pub release_at: chrono::DateTime<chrono::Utc>,
    pub latency: std::time::Duration,
}

/// Overall quarantine status
#[derive(Clone, Debug)]
pub struct QuarantineStatus {
    pub quarantined_count: usize,
    pub pending_review: usize,
    pub nodes: Vec<QuarantineEntry>,
}
```

**Acceptance Criteria**:
- [ ] Quarantine operation under 100 microseconds
- [ ] Check operation under 10 microseconds
- [ ] Auto-release for expired quarantines
- [ ] History tracking for auditing
- [ ] Review queue for human intervention

---

### 2.3 Concept Drift Monitoring

#### REQ-IMMUNE-009: ConceptDriftMonitor Definition

**Priority**: Critical
**Description**: The system SHALL implement concept drift monitoring with distribution shift alerts.

```rust
/// Monitors concept drift across the knowledge graph
pub struct ConceptDriftMonitor {
    /// Distribution trackers per domain
    domain_distributions: HashMap<String, DistributionTracker>,

    /// Global distribution tracker
    global_distribution: DistributionTracker,

    /// Drift detection algorithm
    drift_detector: DriftDetector,

    /// Alert configuration
    alert_config: DriftAlertConfig,

    /// Active alerts
    active_alerts: Vec<DriftAlert>,

    /// Detection window size
    window_size: Duration,
}

/// Tracks distribution of embeddings/features over time
pub struct DistributionTracker {
    /// Current window statistics
    pub current: DistributionStats,

    /// Reference (baseline) statistics
    pub reference: DistributionStats,

    /// Historical snapshots
    pub history: VecDeque<DistributionSnapshot>,

    /// Window size for current stats
    pub window_size: Duration,

    /// Last update time
    pub last_update: Option<Instant>,
}

/// Statistics for a distribution
#[derive(Clone, Default)]
pub struct DistributionStats {
    /// Mean vector
    pub mean: Vec<f32>,

    /// Covariance diagonal (simplified)
    pub variance: Vec<f32>,

    /// Sample count
    pub count: u64,

    /// Entropy estimate
    pub entropy: f32,
}

/// Historical snapshot of distribution
#[derive(Clone)]
pub struct DistributionSnapshot {
    pub stats: DistributionStats,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Configuration for drift alerts
#[derive(Clone)]
pub struct DriftAlertConfig {
    /// Threshold for drift detection (KL divergence)
    pub drift_threshold: f32,  // Default: 0.5

    /// Threshold for severe drift
    pub severe_threshold: f32,  // Default: 1.0

    /// Minimum samples before detection
    pub min_samples: u64,  // Default: 100

    /// Alert cooldown period
    pub cooldown: Duration,  // Default: 5 minutes
}

impl Default for DriftAlertConfig {
    fn default() -> Self {
        Self {
            drift_threshold: 0.5,
            severe_threshold: 1.0,
            min_samples: 100,
            cooldown: Duration::from_secs(300),
        }
    }
}

/// Drift detection algorithms
pub struct DriftDetector {
    /// Page-Hinkley test parameters
    page_hinkley: PageHinkleyParams,

    /// ADWIN parameters
    adwin: AdwinParams,

    /// Selected algorithm
    algorithm: DriftAlgorithm,
}

#[derive(Clone)]
pub struct PageHinkleyParams {
    pub delta: f32,      // Magnitude threshold
    pub lambda: f32,     // Detection threshold
    pub alpha: f32,      // Forgetting factor
}

#[derive(Clone)]
pub struct AdwinParams {
    pub delta: f32,      // Confidence parameter
    pub max_buckets: usize,
}

#[derive(Clone)]
pub enum DriftAlgorithm {
    PageHinkley,
    Adwin,
    KLDivergence,
    Hybrid,
}

impl ConceptDriftMonitor {
    /// Create new drift monitor
    pub fn new(window_size: Duration, alert_config: DriftAlertConfig) -> Self {
        Self {
            domain_distributions: HashMap::new(),
            global_distribution: DistributionTracker::new(window_size),
            drift_detector: DriftDetector::new(DriftAlgorithm::Hybrid),
            alert_config,
            active_alerts: Vec::new(),
            window_size,
        }
    }

    /// Update distribution with new embedding
    ///
    /// `Constraint: Update_Latency < 1ms`
    pub fn update(&mut self, embedding: &[f32], domain: Option<&str>) {
        // Update global distribution
        self.global_distribution.update(embedding);

        // Update domain-specific distribution
        if let Some(d) = domain {
            let tracker = self.domain_distributions
                .entry(d.to_string())
                .or_insert_with(|| DistributionTracker::new(self.window_size));
            tracker.update(embedding);
        }
    }

    /// Check for drift
    ///
    /// `Constraint: Detection_Window < 1s`
    pub fn check_drift(&mut self, domain: Option<&str>) -> Option<DriftAlert> {
        let tracker = domain
            .and_then(|d| self.domain_distributions.get(d))
            .unwrap_or(&self.global_distribution);

        // Need minimum samples
        if tracker.current.count < self.alert_config.min_samples {
            return None;
        }

        // Calculate drift metric
        let drift_score = self.drift_detector.detect(
            &tracker.reference,
            &tracker.current,
        );

        // Check threshold
        if drift_score < self.alert_config.drift_threshold {
            return None;
        }

        // Check cooldown
        if let Some(last_alert) = self.active_alerts.last() {
            if last_alert.detected_at.elapsed() < self.alert_config.cooldown {
                return None;
            }
        }

        // Create alert
        let severity = if drift_score > self.alert_config.severe_threshold {
            DriftSeverity::Severe
        } else {
            DriftSeverity::Moderate
        };

        let alert = DriftAlert {
            alert_id: Uuid::new_v4(),
            domain: domain.map(String::from),
            drift_score,
            severity,
            detected_at: Instant::now(),
            reference_stats: tracker.reference.clone(),
            current_stats: tracker.current.clone(),
            recommended_action: self.recommend_action(severity),
        };

        self.active_alerts.push(alert.clone());

        Some(alert)
    }

    /// Get active alerts
    pub fn get_alerts(&self) -> &[DriftAlert] {
        &self.active_alerts
    }

    /// Clear resolved alerts
    pub fn clear_resolved(&mut self) {
        self.active_alerts.retain(|a| a.detected_at.elapsed() < Duration::from_secs(3600));
    }

    fn recommend_action(&self, severity: DriftSeverity) -> DriftAction {
        match severity {
            DriftSeverity::Severe => DriftAction::RetrainBaseline,
            DriftSeverity::Moderate => DriftAction::InvestigateSource,
            DriftSeverity::Minor => DriftAction::Monitor,
        }
    }
}

impl DistributionTracker {
    pub fn new(window_size: Duration) -> Self {
        Self {
            current: DistributionStats::default(),
            reference: DistributionStats::default(),
            history: VecDeque::new(),
            window_size,
            last_update: None,
        }
    }

    pub fn update(&mut self, embedding: &[f32]) {
        // Initialize mean if empty
        if self.current.mean.is_empty() {
            self.current.mean = vec![0.0; embedding.len()];
            self.current.variance = vec![0.0; embedding.len()];
        }

        self.current.count += 1;
        let n = self.current.count as f32;

        // Update mean and variance (Welford's algorithm)
        for (i, &val) in embedding.iter().enumerate() {
            if i < self.current.mean.len() {
                let delta = val - self.current.mean[i];
                self.current.mean[i] += delta / n;
                let delta2 = val - self.current.mean[i];
                self.current.variance[i] += delta * delta2;
            }
        }

        self.last_update = Some(Instant::now());

        // Initialize reference if not set
        if self.reference.count == 0 && self.current.count >= 100 {
            self.reference = self.current.clone();
        }
    }
}

impl DriftDetector {
    pub fn new(algorithm: DriftAlgorithm) -> Self {
        Self {
            page_hinkley: PageHinkleyParams {
                delta: 0.005,
                lambda: 50.0,
                alpha: 0.9999,
            },
            adwin: AdwinParams {
                delta: 0.002,
                max_buckets: 5,
            },
            algorithm,
        }
    }

    /// Detect drift between reference and current distributions
    pub fn detect(&self, reference: &DistributionStats, current: &DistributionStats) -> f32 {
        match self.algorithm {
            DriftAlgorithm::KLDivergence => self.kl_divergence(reference, current),
            DriftAlgorithm::PageHinkley => self.page_hinkley_test(reference, current),
            DriftAlgorithm::Adwin => self.adwin_test(reference, current),
            DriftAlgorithm::Hybrid => {
                let kl = self.kl_divergence(reference, current);
                let ph = self.page_hinkley_test(reference, current);
                (kl + ph) / 2.0
            }
        }
    }

    fn kl_divergence(&self, p: &DistributionStats, q: &DistributionStats) -> f32 {
        if p.mean.is_empty() || q.mean.is_empty() || p.mean.len() != q.mean.len() {
            return 0.0;
        }

        let mut kl = 0.0f32;

        for i in 0..p.mean.len() {
            let p_var = (p.variance[i] / p.count.max(1) as f32).max(0.001);
            let q_var = (q.variance[i] / q.count.max(1) as f32).max(0.001);
            let mean_diff = p.mean[i] - q.mean[i];

            // KL divergence for univariate Gaussians
            let term = (q_var / p_var).ln() + (p_var + mean_diff.powi(2)) / q_var - 1.0;
            kl += term / 2.0;
        }

        kl / p.mean.len() as f32
    }

    fn page_hinkley_test(&self, reference: &DistributionStats, current: &DistributionStats) -> f32 {
        // Simplified Page-Hinkley using mean difference
        if reference.mean.is_empty() || current.mean.is_empty() {
            return 0.0;
        }

        let mean_diff: f32 = reference.mean.iter()
            .zip(current.mean.iter())
            .map(|(r, c)| (r - c).abs())
            .sum::<f32>() / reference.mean.len() as f32;

        mean_diff / self.page_hinkley.delta
    }

    fn adwin_test(&self, reference: &DistributionStats, current: &DistributionStats) -> f32 {
        // Simplified ADWIN using variance ratio
        if reference.variance.is_empty() || current.variance.is_empty() {
            return 0.0;
        }

        let var_ratio: f32 = reference.variance.iter()
            .zip(current.variance.iter())
            .map(|(r, c)| {
                let r_norm = (r / reference.count.max(1) as f32).max(0.001);
                let c_norm = (c / current.count.max(1) as f32).max(0.001);
                (r_norm / c_norm).max(c_norm / r_norm)
            })
            .sum::<f32>() / reference.variance.len() as f32;

        (var_ratio - 1.0).max(0.0)
    }
}

/// Alert for detected drift
#[derive(Clone, Debug)]
pub struct DriftAlert {
    pub alert_id: Uuid,
    pub domain: Option<String>,
    pub drift_score: f32,
    pub severity: DriftSeverity,
    pub detected_at: Instant,
    pub reference_stats: DistributionStats,
    pub current_stats: DistributionStats,
    pub recommended_action: DriftAction,
}

#[derive(Clone, Debug)]
pub enum DriftSeverity {
    Minor,
    Moderate,
    Severe,
}

#[derive(Clone, Debug)]
pub enum DriftAction {
    Monitor,
    InvestigateSource,
    RetrainBaseline,
}
```

**Acceptance Criteria**:
- [ ] Distribution update under 1ms
- [ ] Drift detection window under 1 second
- [ ] Multiple detection algorithms (KL, Page-Hinkley, ADWIN)
- [ ] Alert severity levels
- [ ] Cooldown to prevent alert spam

---

### 2.4 Homeostatic Plasticity

#### REQ-IMMUNE-010: HomeostaticPlasticity Definition

**Priority**: Critical
**Description**: The system SHALL implement homeostatic plasticity for parameter stability maintenance.

```rust
/// Homeostatic plasticity mechanism for parameter stability
pub struct HomeostaticPlasticity {
    /// Setpoints for each parameter
    setpoints: HashMap<String, Setpoint>,

    /// Current parameter values
    current_values: HashMap<String, f32>,

    /// Adaptation rate
    adaptation_rate: f32,

    /// Stability threshold
    stability_threshold: f32,

    /// History of parameter values
    history: HashMap<String, VecDeque<f32>>,

    /// Maximum history size
    max_history: usize,

    /// Metrics
    metrics: HomeostaticMetrics,
}

/// Setpoint configuration for a parameter
#[derive(Clone)]
pub struct Setpoint {
    /// Target value
    pub target: f32,

    /// Acceptable range (min, max)
    pub range: (f32, f32),

    /// Adaptation speed for this parameter
    pub adaptation_speed: f32,

    /// Priority (higher = faster correction)
    pub priority: f32,
}

/// Metrics for homeostatic system
#[derive(Default)]
pub struct HomeostaticMetrics {
    pub corrections_applied: u64,
    pub stability_violations: u64,
    pub recovery_times: VecDeque<Duration>,
    pub mean_recovery_time: Duration,
}

impl HomeostaticPlasticity {
    /// Create new homeostatic controller
    pub fn new(adaptation_rate: f32, stability_threshold: f32) -> Self {
        Self {
            setpoints: HashMap::new(),
            current_values: HashMap::new(),
            adaptation_rate,
            stability_threshold,
            history: HashMap::new(),
            max_history: 100,
            metrics: HomeostaticMetrics::default(),
        }
    }

    /// Initialize with default setpoints
    pub fn with_default_setpoints() -> Self {
        let mut controller = Self::new(0.1, 0.05);

        // Default setpoints matching constitution.yaml
        controller.add_setpoint("importance", Setpoint {
            target: 0.5,
            range: (0.0, 1.0),
            adaptation_speed: 0.1,
            priority: 1.0,
        });

        controller.add_setpoint("entropy", Setpoint {
            target: 0.5,
            range: (0.0, 1.0),
            adaptation_speed: 0.05,
            priority: 0.8,
        });

        controller.add_setpoint("coherence", Setpoint {
            target: 0.6,
            range: (0.0, 1.0),
            adaptation_speed: 0.05,
            priority: 0.9,
        });

        controller.add_setpoint("dopamine", Setpoint {
            target: 0.5,
            range: (0.0, 1.0),
            adaptation_speed: 0.1,
            priority: 0.7,
        });

        controller.add_setpoint("serotonin", Setpoint {
            target: 0.5,
            range: (0.0, 1.0),
            adaptation_speed: 0.1,
            priority: 0.7,
        });

        controller
    }

    /// Add or update a setpoint
    pub fn add_setpoint(&mut self, name: &str, setpoint: Setpoint) {
        self.setpoints.insert(name.to_string(), setpoint);
        self.current_values.entry(name.to_string()).or_insert(setpoint.target);
        self.history.entry(name.to_string()).or_insert_with(VecDeque::new);
    }

    /// Update parameter value and apply homeostatic correction
    ///
    /// `Constraint: Update_Latency < 100us`
    pub fn update(&mut self, name: &str, value: f32) -> Option<HomeostaticCorrection> {
        let setpoint = self.setpoints.get(name)?;

        // Store in history
        let history = self.history.entry(name.to_string()).or_insert_with(VecDeque::new);
        history.push_back(value);
        if history.len() > self.max_history {
            history.pop_front();
        }

        // Update current value
        self.current_values.insert(name.to_string(), value);

        // Check if correction needed
        let deviation = value - setpoint.target;
        if deviation.abs() < self.stability_threshold {
            return None;
        }

        self.metrics.stability_violations += 1;

        // Calculate correction
        let correction_magnitude = deviation.abs()
            * self.adaptation_rate
            * setpoint.adaptation_speed
            * setpoint.priority;

        let corrected_value = if deviation > 0.0 {
            (value - correction_magnitude).max(setpoint.range.0)
        } else {
            (value + correction_magnitude).min(setpoint.range.1)
        };

        self.metrics.corrections_applied += 1;

        Some(HomeostaticCorrection {
            parameter: name.to_string(),
            original_value: value,
            corrected_value,
            deviation,
            correction_magnitude,
        })
    }

    /// Scale importance toward setpoint (per PRD Section 7.3)
    ///
    /// `Constraint: Scale_Latency < 50us`
    pub fn scale_importance(&mut self, current_importance: f32) -> f32 {
        let setpoint = self.setpoints.get("importance")
            .map(|s| s.target)
            .unwrap_or(0.5);

        let deviation = current_importance - setpoint;
        let correction = deviation * self.adaptation_rate;

        // Apply correction, clamped to valid range
        (current_importance - correction).clamp(0.0, 1.0)
    }

    /// Check overall system stability
    pub fn check_stability(&self) -> StabilityReport {
        let mut unstable_params = Vec::new();
        let mut total_deviation = 0.0f32;

        for (name, setpoint) in &self.setpoints {
            if let Some(&value) = self.current_values.get(name) {
                let deviation = (value - setpoint.target).abs();
                total_deviation += deviation;

                if deviation > self.stability_threshold {
                    unstable_params.push(UnstableParameter {
                        name: name.clone(),
                        value,
                        target: setpoint.target,
                        deviation,
                    });
                }
            }
        }

        let mean_deviation = if !self.setpoints.is_empty() {
            total_deviation / self.setpoints.len() as f32
        } else {
            0.0
        };

        StabilityReport {
            is_stable: unstable_params.is_empty(),
            mean_deviation,
            unstable_params,
            corrections_applied: self.metrics.corrections_applied,
            stability_score: (1.0 - mean_deviation).max(0.0),
        }
    }

    /// Get parameter variance from history
    pub fn get_variance(&self, name: &str) -> Option<f32> {
        let history = self.history.get(name)?;
        if history.len() < 2 {
            return None;
        }

        let mean: f32 = history.iter().sum::<f32>() / history.len() as f32;
        let variance: f32 = history.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f32>() / (history.len() - 1) as f32;

        Some(variance)
    }
}

/// Correction applied by homeostatic system
#[derive(Clone, Debug)]
pub struct HomeostaticCorrection {
    pub parameter: String,
    pub original_value: f32,
    pub corrected_value: f32,
    pub deviation: f32,
    pub correction_magnitude: f32,
}

/// Stability report for the system
#[derive(Clone, Debug)]
pub struct StabilityReport {
    pub is_stable: bool,
    pub mean_deviation: f32,
    pub unstable_params: Vec<UnstableParameter>,
    pub corrections_applied: u64,
    pub stability_score: f32,
}

/// Parameter that is currently unstable
#[derive(Clone, Debug)]
pub struct UnstableParameter {
    pub name: String,
    pub value: f32,
    pub target: f32,
    pub deviation: f32,
}
```

**Acceptance Criteria**:
- [ ] Update latency under 100 microseconds
- [ ] Scale importance under 50 microseconds
- [ ] Default setpoint 0.5 for importance (per PRD 7.3)
- [ ] Configurable adaptation rate
- [ ] Stability reporting

---

### 2.5 MCP Tool Integration

#### REQ-IMMUNE-011: check_adversarial Tool

**Priority**: Critical
**Description**: The system SHALL expose adversarial checking through MCP tool.

```rust
/// MCP tool: check_adversarial
pub struct CheckAdversarialTool {
    pub name: &'static str,  // "check_adversarial"
    pub description: &'static str,
}

#[derive(Deserialize)]
pub struct CheckAdversarialParams {
    /// Content to check
    pub content: String,

    /// Embedding to check (optional, will be computed if not provided)
    pub embedding: Option<Vec<f32>>,

    /// Domain for context-aware detection
    pub domain: Option<String>,

    /// Check types to perform
    #[serde(default = "default_check_types")]
    pub check_types: Vec<AdversarialCheckType>,
}

fn default_check_types() -> Vec<AdversarialCheckType> {
    vec![
        AdversarialCheckType::PromptInjection,
        AdversarialCheckType::EmbeddingPoisoning,
        AdversarialCheckType::ContentAlignment,
    ]
}

#[derive(Deserialize, Clone)]
pub enum AdversarialCheckType {
    PromptInjection,
    EmbeddingPoisoning,
    ContentAlignment,
    StructuralAnomaly,
    SemanticCancer,
    All,
}

#[derive(Serialize)]
pub struct CheckAdversarialResult {
    /// Whether content is safe
    pub is_safe: bool,

    /// Overall threat level
    pub threat_level: ThreatLevel,

    /// Individual check results
    pub checks: Vec<IndividualCheck>,

    /// Recommended action
    pub recommended_action: RecommendedAction,

    /// Cognitive Pulse header
    pub pulse: CognitivePulse,
}

#[derive(Serialize)]
pub struct IndividualCheck {
    pub check_type: String,
    pub passed: bool,
    pub details: Option<String>,
    pub confidence: f32,
}

#[derive(Serialize)]
pub enum ThreatLevel {
    None,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Serialize)]
pub enum RecommendedAction {
    /// Safe to proceed
    Allow,
    /// Allow but monitor
    AllowWithMonitoring,
    /// Require human review
    RequireReview,
    /// Block and quarantine
    Block,
}

impl CheckAdversarialTool {
    pub async fn execute(
        &self,
        params: CheckAdversarialParams,
        ctx: &ToolContext,
    ) -> Result<CheckAdversarialResult, ToolError> {
        let threat_detector = ctx.get_threat_detector()?;
        let semantic_immune = ctx.get_semantic_immune()?;

        let mut checks = Vec::new();
        let mut highest_threat = ThreatLevel::None;

        // Pattern matching for prompt injection
        if params.check_types.iter().any(|t| matches!(t, AdversarialCheckType::PromptInjection | AdversarialCheckType::All)) {
            let mut detector = threat_detector.write().await;
            let matches = detector.pattern_matcher.match_content(&params.content);

            let passed = matches.is_empty();
            if !passed {
                highest_threat = ThreatLevel::Critical;
            }

            checks.push(IndividualCheck {
                check_type: "prompt_injection".to_string(),
                passed,
                details: if !passed {
                    Some(format!("{} pattern(s) matched", matches.len()))
                } else {
                    None
                },
                confidence: if passed { 0.95 } else { matches.iter().map(|m| m.confidence).sum::<f32>() / matches.len() as f32 },
            });
        }

        // Embedding poisoning check
        if params.check_types.iter().any(|t| matches!(t, AdversarialCheckType::EmbeddingPoisoning | AdversarialCheckType::All)) {
            if let Some(ref embedding) = params.embedding {
                let immune = semantic_immune.read().await;
                let result = immune.poison_detector.check_embedding(
                    &params.content,
                    embedding,
                    params.domain.as_deref(),
                );

                let passed = !result.is_poisoned;
                if !passed && matches!(highest_threat, ThreatLevel::None | ThreatLevel::Low) {
                    highest_threat = ThreatLevel::High;
                }

                checks.push(IndividualCheck {
                    check_type: "embedding_poisoning".to_string(),
                    passed,
                    details: result.poison_type.map(|t| format!("{:?}", t)),
                    confidence: result.confidence,
                });
            }
        }

        // Content alignment check
        if params.check_types.iter().any(|t| matches!(t, AdversarialCheckType::ContentAlignment | AdversarialCheckType::All)) {
            let detector = threat_detector.read().await;
            let result = detector.entropy_analyzer.is_anomalous(
                &params.content,
                params.domain.as_deref().unwrap_or("general"),
            );

            let passed = !result.is_anomalous;
            if !passed && matches!(highest_threat, ThreatLevel::None) {
                highest_threat = ThreatLevel::Medium;
            }

            checks.push(IndividualCheck {
                check_type: "content_alignment".to_string(),
                passed,
                details: result.anomaly_reason.map(|r| format!("{:?}", r)),
                confidence: if passed { 0.9 } else { 0.7 },
            });
        }

        let is_safe = checks.iter().all(|c| c.passed);

        let recommended_action = match highest_threat {
            ThreatLevel::None | ThreatLevel::Low => RecommendedAction::Allow,
            ThreatLevel::Medium => RecommendedAction::AllowWithMonitoring,
            ThreatLevel::High => RecommendedAction::RequireReview,
            ThreatLevel::Critical => RecommendedAction::Block,
        };

        Ok(CheckAdversarialResult {
            is_safe,
            threat_level: highest_threat,
            checks,
            recommended_action,
            pulse: ctx.get_cognitive_pulse().await?,
        })
    }
}
```

**Acceptance Criteria**:
- [ ] All check types supported
- [ ] Threat levels correctly assigned
- [ ] Recommended actions appropriate
- [ ] Cognitive Pulse included
- [ ] Latency under 50ms for all checks

---

#### REQ-IMMUNE-012: homeostatic_status Tool

**Priority**: Critical
**Description**: The system SHALL expose homeostatic status through MCP tool.

```rust
/// MCP tool: homeostatic_status
pub struct HomeostaticStatusTool {
    pub name: &'static str,  // "homeostatic_status"
    pub description: &'static str,
}

#[derive(Deserialize)]
pub struct HomeostaticStatusParams {
    /// Include quarantined nodes
    #[serde(default)]
    pub include_quarantined: bool,

    /// Include drift alerts
    #[serde(default)]
    pub include_drift_alerts: bool,

    /// Include stability report
    #[serde(default = "default_true")]
    pub include_stability: bool,
}

fn default_true() -> bool { true }

#[derive(Serialize)]
pub struct HomeostaticStatusResult {
    /// Overall graph health score [0, 1]
    pub health_score: f32,

    /// Quarantine status
    pub quarantine: QuarantineSummary,

    /// Stability report
    pub stability: Option<StabilityReport>,

    /// Active drift alerts
    pub drift_alerts: Option<Vec<DriftAlertSummary>>,

    /// Recent threat detections
    pub recent_threats: Vec<ThreatSummary>,

    /// Homeostatic metrics
    pub metrics: HomeostaticMetricsSummary,

    /// Cognitive Pulse header
    pub pulse: CognitivePulse,
}

#[derive(Serialize)]
pub struct QuarantineSummary {
    pub total_quarantined: usize,
    pub pending_review: usize,
    pub by_reason: HashMap<String, usize>,
    pub nodes: Option<Vec<QuarantinedNodeSummary>>,
}

#[derive(Serialize)]
pub struct QuarantinedNodeSummary {
    pub node_id: Uuid,
    pub reason: String,
    pub quarantined_at: String,
    pub release_at: String,
}

#[derive(Serialize)]
pub struct DriftAlertSummary {
    pub alert_id: Uuid,
    pub domain: Option<String>,
    pub severity: String,
    pub drift_score: f32,
    pub recommended_action: String,
}

#[derive(Serialize)]
pub struct ThreatSummary {
    pub threat_type: String,
    pub severity: String,
    pub detected_at: String,
    pub resolved: bool,
}

#[derive(Serialize)]
pub struct HomeostaticMetricsSummary {
    pub corrections_applied: u64,
    pub stability_violations: u64,
    pub mean_recovery_time_ms: u64,
    pub attack_detection_rate: f32,
    pub false_positive_rate: f32,
}

impl HomeostaticStatusTool {
    pub async fn execute(
        &self,
        params: HomeostaticStatusParams,
        ctx: &ToolContext,
    ) -> Result<HomeostaticStatusResult, ToolError> {
        let immune = ctx.get_immune_system()?;
        let homeostatic = ctx.get_homeostatic_plasticity()?;
        let quarantine = ctx.get_quarantine_manager()?;
        let drift_monitor = ctx.get_drift_monitor()?;

        // Get quarantine status
        let q_status = quarantine.read().await.get_status();
        let mut by_reason: HashMap<String, usize> = HashMap::new();
        for node in &q_status.nodes {
            *by_reason.entry(format!("{:?}", node.reason)).or_insert(0) += 1;
        }

        let quarantine_summary = QuarantineSummary {
            total_quarantined: q_status.quarantined_count,
            pending_review: q_status.pending_review,
            by_reason,
            nodes: if params.include_quarantined {
                Some(q_status.nodes.iter().map(|n| QuarantinedNodeSummary {
                    node_id: n.node_id,
                    reason: format!("{:?}", n.reason),
                    quarantined_at: n.quarantined_at.to_rfc3339(),
                    release_at: n.release_at.to_rfc3339(),
                }).collect())
            } else {
                None
            },
        };

        // Get stability report
        let stability = if params.include_stability {
            Some(homeostatic.read().await.check_stability())
        } else {
            None
        };

        // Get drift alerts
        let drift_alerts = if params.include_drift_alerts {
            let alerts = drift_monitor.read().await.get_alerts();
            Some(alerts.iter().map(|a| DriftAlertSummary {
                alert_id: a.alert_id,
                domain: a.domain.clone(),
                severity: format!("{:?}", a.severity),
                drift_score: a.drift_score,
                recommended_action: format!("{:?}", a.recommended_action),
            }).collect())
        } else {
            None
        };

        // Calculate health score
        let stability_score = stability.as_ref().map(|s| s.stability_score).unwrap_or(1.0);
        let quarantine_penalty = (q_status.quarantined_count as f32 * 0.01).min(0.3);
        let drift_penalty = drift_alerts.as_ref().map(|a| a.len() as f32 * 0.05).unwrap_or(0.0).min(0.2);
        let health_score = (stability_score - quarantine_penalty - drift_penalty).max(0.0);

        // Get metrics
        let h_metrics = homeostatic.read().await;
        let metrics = HomeostaticMetricsSummary {
            corrections_applied: h_metrics.metrics.corrections_applied,
            stability_violations: h_metrics.metrics.stability_violations,
            mean_recovery_time_ms: h_metrics.metrics.mean_recovery_time.as_millis() as u64,
            attack_detection_rate: 0.95,  // Would be calculated from actual data
            false_positive_rate: 0.001,   // Would be calculated from actual data
        };

        Ok(HomeostaticStatusResult {
            health_score,
            quarantine: quarantine_summary,
            stability,
            drift_alerts,
            recent_threats: Vec::new(),  // Would be populated from threat detector
            metrics,
            pulse: ctx.get_cognitive_pulse().await?,
        })
    }
}
```

**Acceptance Criteria**:
- [ ] Health score calculated correctly
- [ ] Quarantine summary complete
- [ ] Stability report included when requested
- [ ] Drift alerts summarized
- [ ] Cognitive Pulse included

---

## 3. Non-Functional Requirements

### 3.1 Performance Requirements

#### REQ-IMMUNE-013: Performance Budgets

**Priority**: Critical
**Description**: The system SHALL meet all performance budgets.

| Operation | Budget | Measurement |
|-----------|--------|-------------|
| Pattern matching | <5ms | All patterns checked |
| Entropy calculation | <1ms | Shannon entropy |
| Embedding entropy | <2ms | Distribution analysis |
| Structure analysis (node) | <10ms | Single node check |
| Structure analysis (global) | <50ms | Full graph scan |
| Cancer detection | <5ms | Per node check |
| Poison check | <10ms | Embedding + alignment |
| Quarantine | <100 microseconds | Isolation time |
| Quarantine check | <10 microseconds | Is-quarantined lookup |
| Distribution update | <1ms | Online update |
| Drift detection | <1s window | Sliding detection |
| Homeostatic update | <100 microseconds | Parameter correction |
| Importance scaling | <50 microseconds | Single value |
| **check_adversarial tool** | **<50ms** | End-to-end |
| **homeostatic_status tool** | **<100ms** | Full status |

**Acceptance Criteria**:
- [ ] P95 latency within budget for all operations
- [ ] P99 latency within 2x budget
- [ ] No blocking operations in hot paths
- [ ] Allocation-free where possible

---

### 3.2 Reliability Requirements

#### REQ-IMMUNE-014: Graceful Degradation

**Priority**: High
**Description**: The system SHALL degrade gracefully on component failure.

```rust
impl ImmuneSystem {
    /// Handle component failure gracefully
    pub fn handle_component_failure(&mut self, component: ImmuneComponent) {
        match component {
            ImmuneComponent::PatternMatcher => {
                warn!("Pattern matcher unavailable, using baseline checks only");
                self.config.pattern_matching_enabled = false;
            }
            ImmuneComponent::EntropyAnalyzer => {
                warn!("Entropy analyzer unavailable, skipping entropy checks");
                self.config.entropy_checking_enabled = false;
            }
            ImmuneComponent::CancerDetector => {
                warn!("Cancer detector unavailable, quarantine checks only");
                self.semantic_immune.cancer_detector_enabled = false;
            }
            ImmuneComponent::DriftMonitor => {
                warn!("Drift monitor unavailable, periodic baseline only");
                self.drift_monitor_enabled = false;
            }
            ImmuneComponent::Quarantine => {
                error!("Quarantine unavailable, blocking all suspicious content");
                self.quarantine_fallback_to_block = true;
            }
        }
    }
}

pub enum ImmuneComponent {
    PatternMatcher,
    EntropyAnalyzer,
    CancerDetector,
    DriftMonitor,
    Quarantine,
}
```

**Acceptance Criteria**:
- [ ] Component failure does not crash system
- [ ] Degraded mode logged clearly
- [ ] Fallback behaviors documented
- [ ] Recovery when component available

---

### 3.3 Configuration Requirements

#### REQ-IMMUNE-015: TOML Configuration

**Priority**: High
**Description**: The system SHALL support TOML-based immune system configuration.

```toml
[immune]
enabled = true

[immune.detection]
pattern_matching_enabled = true
entropy_checking_enabled = true
structure_monitoring_enabled = true

[immune.thresholds]
embedding_anomaly_sigma = 3.0
content_alignment_min = 0.4
max_entropy_delta = 0.3
max_reference_depth = 10

[immune.quarantine]
default_duration_hours = 24
max_duration_hours = 168
auto_release = true

[immune.cancer]
importance_threshold = 0.9
neighbor_entropy_threshold = 0.8

[immune.drift]
drift_threshold = 0.5
severe_threshold = 1.0
min_samples = 100
cooldown_seconds = 300
window_size_seconds = 60

[immune.homeostatic]
adaptation_rate = 0.1
stability_threshold = 0.05
importance_setpoint = 0.5

[immune.patterns]
# Additional custom patterns can be defined
custom_patterns = [
    { id = "CUSTOM-001", pattern = "malicious_pattern", severity = "high" }
]
```

**Acceptance Criteria**:
- [ ] All parameters configurable via TOML
- [ ] Configuration validated on load
- [ ] Invalid config returns clear error
- [ ] Defaults match specification

---

## 4. Error Handling

### 4.1 Error Codes

#### REQ-IMMUNE-016: Error Code Catalog

**Priority**: High
**Description**: The system SHALL use consistent error codes.

| Code | Name | Description |
|------|------|-------------|
| -32100 | ImmuneSystemDisabled | Immune system feature disabled |
| -32101 | PatternMatchTimeout | Pattern matching exceeded timeout |
| -32102 | QuarantineFailed | Failed to quarantine node |
| -32103 | InvalidThreatSignature | Malformed threat signature |
| -32104 | DriftDetectionError | Drift detection failed |
| -32105 | HomeostaticError | Homeostatic correction failed |
| -32106 | CancerDetectionError | Cancer detection failed |
| -32107 | PoisonCheckError | Embedding poison check failed |

**Acceptance Criteria**:
- [ ] All errors mapped to codes
- [ ] Error messages descriptive
- [ ] Recovery hints included
- [ ] Errors logged with context

---

## 5. Dependencies

### 5.1 Module Dependencies

| Module | Dependency Type | Purpose |
|--------|-----------------|---------|
| Module 1 (Ghost System) | Required | Trait definitions |
| Module 2 (Core Infrastructure) | Required | Base types, config |
| Module 3 (Embedding Pipeline) | Required | Embedding generation for checks |
| Module 4 (Knowledge Graph) | Required | Graph structure access |
| Module 5 (UTL Integration) | Required | Entropy/coherence metrics |
| Module 6 (Bio-Nervous System) | Required | Layer integration |
| Module 9 (Dream Layer) | Optional | Background scanning |
| Module 10 (Neuromodulation) | Optional | Parameter stability |

### 5.2 External Crate Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| tokio | 1.35+ | Async runtime |
| regex | 1.10+ | Pattern matching |
| uuid | 1.6+ | Node identification |
| chrono | 0.4+ | Timestamps |
| thiserror | 1.0+ | Error types |
| serde | 1.0+ | Serialization |
| lru | 0.12+ | Cache for pattern matching |

---

## 6. Testing Requirements

### 6.1 Unit Test Coverage

| Component | Target Coverage |
|-----------|-----------------|
| PatternMatcher | 95% |
| EntropyAnalyzer | 95% |
| GraphStructureMonitor | 90% |
| SemanticCancerDetector | 95% |
| EmbeddingPoisonDetector | 95% |
| QuarantineManager | 95% |
| ConceptDriftMonitor | 90% |
| HomeostaticPlasticity | 95% |

### 6.2 Integration Test Scenarios

1. Full threat detection pipeline with all check types
2. Quarantine and auto-release cycle
3. Cancer detection and suppression
4. Drift detection with distribution shift
5. Homeostatic correction under perturbation
6. MCP tool round-trip (check_adversarial)
7. MCP tool round-trip (homeostatic_status)
8. Graceful degradation on component failure

### 6.3 Benchmark Tests

| Benchmark | Target |
|-----------|--------|
| Pattern match throughput | >10K/sec |
| Quarantine operations | >1M/sec |
| Drift detection | <1s window |
| Health score calculation | <10ms |

---

## 7. Acceptance Criteria Summary

### 7.1 Critical Acceptance Criteria

1. [ ] ThreatDetector with pattern matching, entropy analysis, structure monitoring
2. [ ] Threat detection latency <50ms
3. [ ] False positive rate <0.1%
4. [ ] SemanticCancerDetector matching SEC-05 (importance > 0.9 AND entropy > 0.8)
5. [ ] EmbeddingPoisonDetector matching SEC-03 (3 std devs, 0.4 alignment)
6. [ ] QuarantineManager with <100 microsecond isolation
7. [ ] ConceptDriftMonitor with <1s detection window
8. [ ] HomeostaticPlasticity with importance setpoint 0.5
9. [ ] check_adversarial MCP tool functional
10. [ ] homeostatic_status MCP tool functional

### 7.2 Quality Gates

| Gate | Criteria |
|------|----------|
| Code Review | All code reviewed and approved |
| Unit Tests | 90% coverage, all passing |
| Integration Tests | All scenarios passing |
| Performance | All benchmarks met |
| Documentation | API docs complete |

---

## 8. Traceability Matrix

| Requirement ID | PRD Section | Constitution Reference | Test Case |
|---------------|-------------|----------------------|-----------|
| REQ-IMMUNE-001 | 7.3, 10 | SEC-03, SEC-05 | T-IMMUNE-001 |
| REQ-IMMUNE-002 | 10.1 | SEC-04 | T-IMMUNE-002 |
| REQ-IMMUNE-003 | 10.1 | SEC-03 | T-IMMUNE-003 |
| REQ-IMMUNE-004 | 10.1 | - | T-IMMUNE-004 |
| REQ-IMMUNE-005 | 7.3 | SEC-05 | T-IMMUNE-005 |
| REQ-IMMUNE-006 | 7.3 | SEC-05 | T-IMMUNE-006 |
| REQ-IMMUNE-007 | 10.1 | SEC-03 | T-IMMUNE-007 |
| REQ-IMMUNE-008 | 10.1 | SEC-05 | T-IMMUNE-008 |
| REQ-IMMUNE-009 | 7.3 | - | T-IMMUNE-009 |
| REQ-IMMUNE-010 | 7.3 | - | T-IMMUNE-010 |
| REQ-IMMUNE-011 | 5.6 | - | T-IMMUNE-011 |
| REQ-IMMUNE-012 | 5.6 | - | T-IMMUNE-012 |
| REQ-IMMUNE-013 | 14 | performance_budgets | T-IMMUNE-013 |
| REQ-IMMUNE-014 | - | - | T-IMMUNE-014 |
| REQ-IMMUNE-015 | - | - | T-IMMUNE-015 |
| REQ-IMMUNE-016 | - | mcp_protocol.error_codes | T-IMMUNE-016 |

---

## 9. References

### 9.1 Internal References

- constitution.yaml: SEC-03, SEC-04, SEC-05, performance_budgets
- contextprd.md: Sections 7.3, 10, 5.6
- Module 3: Embedding Pipeline (embedding generation)
- Module 4: Knowledge Graph (graph structure)
- Module 5: UTL Integration (entropy/coherence)
- Module 10: Neuromodulation (parameter stability)

### 9.2 External Research

- UniGuardian Defense: https://arxiv.org/abs/2502.13141
- OWASP LLM Top 10 2025: https://genai.owasp.org/llmrisk/llm01-prompt-injection/
- Homeostatic Plasticity: https://elifesciences.org/articles/88376

---

*Document Version: 1.0.0*
*Generated: 2025-12-31*
*Agent: #11/14 - Immune System Specification*
