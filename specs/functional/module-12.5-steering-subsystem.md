# Module 12.5: Steering Subsystem - Functional Specification

**Module ID**: SPEC-STEER-012.5
**Version**: 1.0.0
**Status**: Draft
**Phase**: 5
**Duration**: 2 weeks
**Dependencies**: Module 10 (Neuromodulation), Module 12 (Active Inference)
**Last Updated**: 2025-12-31

---

## 1. Executive Summary

The Steering Subsystem module implements Adam Marblestone's architectural concept of separating learning from steering in cognitive systems. This module provides a dedicated system that generates reward signals (analogous to dopamine) to guide learning without directly modifying network weights. The implementation features three core components: the Gardener for long-term memory curation and pruning, the Curator for quality assessment and relevance scoring, and the Thought Assessor for real-time evaluation of thoughts and memories.

### 1.1 Core Objectives

- Implement Marblestone's separation of Learning Subsystem from Steering Subsystem
- Provide SteeringReward feedback with every store_memory operation
- Deploy three-component architecture: Gardener, Curator, Thought Assessor
- Integrate steering rewards with Module 10 dopamine modulation pathway
- Enable reward-driven learning guidance without direct weight modification
- Support adaptive memory curation based on long-term value assessment

### 1.2 Key Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Steering Reward Latency | <5ms per evaluation | End-to-end reward computation |
| Gardener Evaluation | <2ms | Long-term value assessment |
| Curator Assessment | <2ms | Quality and relevance scoring |
| Thought Assessor | <1ms | Real-time coherence check |
| Memory Overhead | <500KB per session | Steering state tracking |
| Reward Accuracy | >80% correlation | User feedback validation |
| Pruning Precision | >90% | False positive rate for valuable content |

---

## 2. Theoretical Background

### 2.1 Marblestone's Steering Subsystem Concept

Adam Marblestone's research proposes a fundamental architectural separation between systems that learn (modify weights/parameters) and systems that steer (provide reward signals to guide learning). This separation mirrors biological neural architectures where dopaminergic pathways modulate learning without directly encoding memories.

**Key Principles**:

1. **Separation of Concerns**: Learning mechanisms should be distinct from reward/steering mechanisms
2. **Indirect Influence**: Steering affects learning through reward signals, not direct parameter modification
3. **Compositional Hierarchy**: Multiple steering components can provide complementary guidance
4. **Temporal Abstraction**: Different components operate at different time scales

### 2.2 Three-Component Architecture

The steering subsystem comprises three specialized components:

**Gardener (Long-term Curation)**:
- Operates on extended time horizons (days to months)
- Evaluates memories for long-term preservation value
- Triggers consolidation and pruning operations
- Maintains memory ecosystem health

**Curator (Quality Assessment)**:
- Evaluates content quality and domain relevance
- Scores information integrity and completeness
- Applies domain-specific weighting
- Ensures knowledge graph coherence

**Thought Assessor (Real-time Evaluation)**:
- Operates in real-time on incoming thoughts/memories
- Evaluates immediate coherence and relevance
- Provides novelty detection and scoring
- Generates instant feedback for learning systems

### 2.3 Integration with Dopamine Modulation

The steering subsystem integrates with Module 10 (Neuromodulation) through a dedicated feedback pathway:

```
SteeringReward --> SteeringDopamineFeedback --> NeuromodulationController
                                                      |
                                                      v
                                              Hopfield beta modulation
                                              Learning rate adjustment
```

This pathway ensures that steering rewards influence system behavior through established neuromodulation channels rather than creating parallel control paths.

---

## 3. Functional Requirements

### 3.1 Steering Subsystem Core

#### REQ-STEER-001: SteeringSubsystem Struct Definition

**Priority**: Critical
**Description**: The system SHALL implement a SteeringSubsystem struct that coordinates all steering operations and generates unified reward signals.

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;
use serde::{Deserialize, Serialize};

/// Core steering subsystem implementing Marblestone's architecture.
///
/// Coordinates Gardener, Curator, and Thought Assessor to generate
/// unified steering rewards that guide learning without direct weight modification.
///
/// `Constraint: Total_Evaluation_Latency < 5ms`
pub struct SteeringSubsystem {
    /// The Gardener: long-term memory curation (Marblestone)
    pub gardener: Gardener,

    /// The Curator: quality and relevance assessment (Marblestone)
    pub curator: Curator,

    /// The Thought Assessor: real-time evaluation (Marblestone)
    pub thought_assessor: ThoughtAssessor,

    /// Dopamine feedback integration
    pub dopamine_feedback: SteeringDopamineFeedback,

    /// Configuration
    pub config: SteeringConfig,

    /// Session-level steering state
    pub session_state: Arc<RwLock<SteeringSessionState>>,

    /// Metrics collection
    pub metrics: SteeringMetrics,
}

/// Configuration for steering subsystem
#[derive(Clone, Debug, Deserialize)]
pub struct SteeringConfig {
    /// Weight for gardener score in final reward
    pub gardener_weight: f32,  // Default: 0.35

    /// Weight for curator score in final reward
    pub curator_weight: f32,   // Default: 0.35

    /// Weight for assessor score in final reward
    pub assessor_weight: f32,  // Default: 0.30

    /// Minimum reward threshold for positive feedback
    pub positive_threshold: f32,  // Default: 0.3

    /// Maximum reward threshold for negative feedback
    pub negative_threshold: f32,  // Default: -0.3

    /// Enable dopamine feedback integration
    pub dopamine_integration_enabled: bool,  // Default: true

    /// Evaluation timeout
    pub evaluation_timeout: Duration,  // Default: 5ms

    /// Enable suggestion generation
    pub suggestions_enabled: bool,  // Default: true

    /// Maximum suggestions per evaluation
    pub max_suggestions: usize,  // Default: 3
}

impl Default for SteeringConfig {
    fn default() -> Self {
        Self {
            gardener_weight: 0.35,
            curator_weight: 0.35,
            assessor_weight: 0.30,
            positive_threshold: 0.3,
            negative_threshold: -0.3,
            dopamine_integration_enabled: true,
            evaluation_timeout: Duration::from_millis(5),
            suggestions_enabled: true,
            max_suggestions: 3,
        }
    }
}

/// Session-level steering state
#[derive(Clone, Default)]
pub struct SteeringSessionState {
    /// Session identifier
    pub session_id: Uuid,

    /// Running average of steering rewards
    pub avg_reward: f32,

    /// Total evaluations performed
    pub evaluation_count: u64,

    /// Recent reward history
    pub reward_history: Vec<SteeringRewardEntry>,

    /// Domain-specific statistics
    pub domain_stats: HashMap<String, DomainSteeringStats>,

    /// Last evaluation timestamp
    pub last_evaluation: Option<Instant>,
}

/// Entry in reward history
#[derive(Clone, Serialize)]
pub struct SteeringRewardEntry {
    pub timestamp: Instant,
    pub node_id: Uuid,
    pub reward: f32,
    pub components: SteeringComponents,
}

/// Per-domain steering statistics
#[derive(Clone, Default, Serialize)]
pub struct DomainSteeringStats {
    pub total_evaluations: u64,
    pub avg_reward: f32,
    pub prune_count: u64,
    pub consolidation_count: u64,
}
```

**Acceptance Criteria**:
- [ ] SteeringSubsystem struct compiles with all components
- [ ] Configuration defaults match specification
- [ ] Thread-safe session state via Arc<RwLock>
- [ ] Timeout enforcement on evaluation path
- [ ] Metrics collection enabled
- [ ] Memory footprint under 500KB per session

---

#### REQ-STEER-002: SteeringReward Output Structure

**Priority**: Critical
**Description**: The system SHALL return a comprehensive SteeringReward with every store_memory operation.

```rust
/// Steering reward returned with every store_memory operation (Marblestone)
#[derive(Clone, Debug, Serialize)]
pub struct SteeringReward {
    /// Overall reward signal [-1.0, 1.0]
    pub reward: f32,

    /// Breakdown by component
    pub components: SteeringComponents,

    /// Explanation for the reward
    pub explanation: String,

    /// Suggested actions based on evaluation
    pub suggestions: Vec<SteeringSuggestion>,

    /// Confidence in the reward assessment [0.0, 1.0]
    pub confidence: f32,

    /// Evaluation latency
    pub latency: Duration,

    /// Metadata for debugging
    pub metadata: SteeringMetadata,
}

/// Component-wise breakdown of steering reward
#[derive(Clone, Debug, Serialize)]
pub struct SteeringComponents {
    /// Gardener score: long-term value [-1.0, 1.0]
    pub gardener_score: f32,

    /// Curator score: quality assessment [-1.0, 1.0]
    pub curator_score: f32,

    /// Assessor score: immediate relevance [-1.0, 1.0]
    pub assessor_score: f32,
}

/// Suggested action from steering evaluation
#[derive(Clone, Debug, Serialize)]
pub struct SteeringSuggestion {
    /// Suggestion type
    pub suggestion_type: SuggestionType,

    /// Priority level [0.0, 1.0]
    pub priority: f32,

    /// Human-readable description
    pub description: String,

    /// Target node ID (if applicable)
    pub target_node: Option<Uuid>,

    /// Suggested action parameters
    pub parameters: HashMap<String, String>,
}

/// Types of suggestions from steering
#[derive(Clone, Debug, Serialize, PartialEq)]
pub enum SuggestionType {
    /// Consolidate with related memories
    Consolidate,
    /// Prune low-value memory
    Prune,
    /// Strengthen connection to existing node
    StrengthenConnection,
    /// Add missing metadata
    EnrichMetadata,
    /// Request clarification from user
    RequestClarification,
    /// Flag for dream-time review
    DreamReview,
    /// Increase importance score
    PromoteImportance,
    /// Decrease importance score
    DemoteImportance,
}

/// Metadata for debugging and analysis
#[derive(Clone, Debug, Serialize)]
pub struct SteeringMetadata {
    /// Evaluation ID for tracing
    pub evaluation_id: Uuid,

    /// Timestamp of evaluation
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Input node ID
    pub node_id: Uuid,

    /// Domain detected
    pub domain: Option<String>,

    /// Novelty score
    pub novelty: f32,

    /// Coherence with existing knowledge
    pub coherence: f32,
}

impl SteeringReward {
    /// Create a neutral reward (no strong signal)
    pub fn neutral() -> Self {
        Self {
            reward: 0.0,
            components: SteeringComponents {
                gardener_score: 0.0,
                curator_score: 0.0,
                assessor_score: 0.0,
            },
            explanation: "Neutral evaluation - no strong positive or negative signals".to_string(),
            suggestions: Vec::new(),
            confidence: 0.5,
            latency: Duration::ZERO,
            metadata: SteeringMetadata::default(),
        }
    }

    /// Check if reward is positive
    pub fn is_positive(&self) -> bool {
        self.reward > 0.3
    }

    /// Check if reward is negative
    pub fn is_negative(&self) -> bool {
        self.reward < -0.3
    }

    /// Get dominant component
    pub fn dominant_component(&self) -> &'static str {
        let abs_scores = [
            (self.components.gardener_score.abs(), "gardener"),
            (self.components.curator_score.abs(), "curator"),
            (self.components.assessor_score.abs(), "assessor"),
        ];
        abs_scores.iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, name)| *name)
            .unwrap_or("none")
    }
}

impl Default for SteeringMetadata {
    fn default() -> Self {
        Self {
            evaluation_id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            node_id: Uuid::nil(),
            domain: None,
            novelty: 0.5,
            coherence: 0.5,
        }
    }
}
```

**Acceptance Criteria**:
- [ ] SteeringReward includes all required fields
- [ ] Reward bounded to [-1.0, 1.0]
- [ ] Component scores individually bounded
- [ ] Suggestions generated based on evaluation
- [ ] Confidence reflects assessment certainty
- [ ] Latency tracked for performance monitoring

---

### 3.2 The Gardener Component

#### REQ-STEER-003: Gardener Long-Term Value Assessment

**Priority**: Critical
**Description**: The system SHALL implement the Gardener component for long-term memory curation and pruning decisions.

```rust
/// The Gardener: long-term memory curation (Marblestone)
///
/// Evaluates memories for their long-term preservation value,
/// triggering consolidation and pruning based on usage patterns,
/// relevance decay, and memory ecosystem health.
pub struct Gardener {
    /// Pruning threshold for low-value memories
    pub prune_threshold: f32,  // Default: -0.5

    /// Consolidation threshold for high-value memories
    pub consolidation_threshold: f32,  // Default: 0.7

    /// Consolidation rules for different memory types
    pub consolidation_rules: Vec<ConsolidationRule>,

    /// Pruning rules for different conditions
    pub pruning_rules: Vec<PruningRule>,

    /// Time decay configuration
    pub decay_config: DecayConfig,

    /// Usage pattern analyzer
    pub usage_analyzer: UsagePatternAnalyzer,

    /// Gardener configuration
    pub config: GardenerConfig,
}

/// Configuration for the Gardener
#[derive(Clone, Debug)]
pub struct GardenerConfig {
    /// Minimum age before pruning consideration (hours)
    pub min_age_for_pruning: f32,  // Default: 24.0

    /// Weight for access frequency in value calculation
    pub access_frequency_weight: f32,  // Default: 0.3

    /// Weight for recency in value calculation
    pub recency_weight: f32,  // Default: 0.2

    /// Weight for connection count in value calculation
    pub connection_weight: f32,  // Default: 0.25

    /// Weight for explicit importance in value calculation
    pub importance_weight: f32,  // Default: 0.25

    /// Enable automatic pruning suggestions
    pub auto_prune_enabled: bool,  // Default: true

    /// Maximum memories to suggest for pruning per evaluation
    pub max_prune_suggestions: usize,  // Default: 5
}

impl Default for GardenerConfig {
    fn default() -> Self {
        Self {
            min_age_for_pruning: 24.0,
            access_frequency_weight: 0.3,
            recency_weight: 0.2,
            connection_weight: 0.25,
            importance_weight: 0.25,
            auto_prune_enabled: true,
            max_prune_suggestions: 5,
        }
    }
}

/// Rule for memory consolidation
#[derive(Clone, Debug)]
pub struct ConsolidationRule {
    /// Rule identifier
    pub id: String,

    /// Condition for triggering consolidation
    pub condition: ConsolidationCondition,

    /// Priority of this rule
    pub priority: u32,

    /// Description for logging
    pub description: String,
}

/// Conditions that trigger consolidation
#[derive(Clone, Debug)]
pub enum ConsolidationCondition {
    /// High access frequency over period
    HighAccessFrequency { min_accesses: u32, period_hours: f32 },

    /// Strong semantic similarity cluster
    SemanticCluster { min_similarity: f32, min_cluster_size: usize },

    /// Temporal proximity pattern
    TemporalProximity { max_gap_hours: f32, min_sequence_length: usize },

    /// Explicit importance above threshold
    HighImportance { threshold: f32 },

    /// Cross-domain connection hub
    CrossDomainHub { min_domains: usize, min_connections: usize },
}

/// Rule for memory pruning
#[derive(Clone, Debug)]
pub struct PruningRule {
    /// Rule identifier
    pub id: String,

    /// Condition for triggering pruning consideration
    pub condition: PruningCondition,

    /// Severity of pruning recommendation
    pub severity: PruningSeverity,

    /// Description for logging
    pub description: String,
}

/// Conditions that suggest pruning
#[derive(Clone, Debug)]
pub enum PruningCondition {
    /// No access for extended period
    Stale { min_hours: f32 },

    /// Low importance and low connections
    LowValue { max_importance: f32, max_connections: usize },

    /// Superseded by newer information
    Superseded { newer_node_id: Uuid },

    /// Redundant with existing content
    Redundant { similar_node_id: Uuid, similarity: f32 },

    /// Failed coherence check
    Incoherent { coherence_score: f32 },
}

/// Severity levels for pruning
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PruningSeverity {
    /// Suggestion only, keep unless space needed
    Low,
    /// Moderate priority for removal
    Medium,
    /// High priority, should be removed
    High,
    /// Critical, immediate removal recommended
    Critical,
}

/// Time-based decay configuration
#[derive(Clone, Debug)]
pub struct DecayConfig {
    /// Half-life for importance decay (hours)
    pub importance_half_life: f32,  // Default: 168.0 (1 week)

    /// Minimum importance floor (never decays below this)
    pub importance_floor: f32,  // Default: 0.1

    /// Access boost factor (how much access counters decay)
    pub access_boost: f32,  // Default: 0.2

    /// Connection boost factor
    pub connection_boost: f32,  // Default: 0.1
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            importance_half_life: 168.0,
            importance_floor: 0.1,
            access_boost: 0.2,
            connection_boost: 0.1,
        }
    }
}

/// Analyzer for memory usage patterns
#[derive(Clone, Default)]
pub struct UsagePatternAnalyzer {
    /// Recent access timestamps per node
    access_history: HashMap<Uuid, Vec<Instant>>,

    /// Access frequency statistics
    frequency_stats: HashMap<Uuid, AccessStats>,
}

/// Access statistics for a node
#[derive(Clone, Default)]
pub struct AccessStats {
    pub total_accesses: u64,
    pub recent_accesses: u32,  // Last 24 hours
    pub avg_interval_hours: f32,
    pub last_access: Option<Instant>,
}

impl Gardener {
    /// Evaluate long-term value of a memory node
    ///
    /// `Constraint: Evaluation_Latency < 2ms`
    pub fn evaluate_long_term_value(&self, node: &KnowledgeNode, context: &GardenerContext) -> f32 {
        let start = Instant::now();

        // Calculate component scores
        let access_score = self.calculate_access_score(node, context);
        let recency_score = self.calculate_recency_score(node);
        let connection_score = self.calculate_connection_score(node, context);
        let importance_score = self.calculate_importance_score(node);

        // Apply decay based on age
        let age_factor = self.calculate_age_factor(node);

        // Weighted combination
        let raw_value =
            self.config.access_frequency_weight * access_score +
            self.config.recency_weight * recency_score +
            self.config.connection_weight * connection_score +
            self.config.importance_weight * importance_score;

        // Apply age decay
        let value = raw_value * age_factor;

        // Clamp to [-1, 1] range
        let clamped = value.clamp(-1.0, 1.0);

        let _elapsed = start.elapsed();
        clamped
    }

    /// Calculate access frequency score
    fn calculate_access_score(&self, node: &KnowledgeNode, context: &GardenerContext) -> f32 {
        if let Some(stats) = context.access_stats.get(&node.id) {
            // Normalize based on expected access patterns
            let expected_daily = 2.0;  // Expected accesses per day
            let actual_daily = stats.recent_accesses as f32;

            // Score: above expected is positive, below is negative
            let ratio = actual_daily / expected_daily;
            (ratio.ln() * 0.5).clamp(-1.0, 1.0)
        } else {
            // No access history - neutral
            0.0
        }
    }

    /// Calculate recency score
    fn calculate_recency_score(&self, node: &KnowledgeNode) -> f32 {
        let age_hours = node.created_at.elapsed().as_secs_f32() / 3600.0;

        // Exponential decay with 72-hour half-life for recency
        let recency_half_life = 72.0;
        let decay = (-age_hours / recency_half_life).exp();

        // Map to [-1, 1]: very recent is positive, very old is negative
        decay * 2.0 - 1.0
    }

    /// Calculate connection score
    fn calculate_connection_score(&self, node: &KnowledgeNode, context: &GardenerContext) -> f32 {
        let connection_count = context.connection_count.get(&node.id).copied().unwrap_or(0);

        // Score based on percentile of connections
        let avg_connections = context.avg_connection_count.max(1.0);
        let ratio = connection_count as f32 / avg_connections;

        // Log scale to handle wide distribution
        (ratio.ln() * 0.3).clamp(-1.0, 1.0)
    }

    /// Calculate importance score (uses explicit metadata)
    fn calculate_importance_score(&self, node: &KnowledgeNode) -> f32 {
        // Map importance [0, 1] to [-1, 1]
        node.importance * 2.0 - 1.0
    }

    /// Calculate age factor for decay
    fn calculate_age_factor(&self, node: &KnowledgeNode) -> f32 {
        let age_hours = node.created_at.elapsed().as_secs_f32() / 3600.0;

        // Exponential decay
        let decay = (-age_hours.ln() / self.decay_config.importance_half_life.ln()).exp();

        // Floor at minimum
        decay.max(self.decay_config.importance_floor)
    }

    /// Check consolidation rules
    pub fn check_consolidation(&self, node: &KnowledgeNode, context: &GardenerContext) -> Option<ConsolidationRule> {
        for rule in &self.consolidation_rules {
            if self.matches_consolidation_condition(&rule.condition, node, context) {
                return Some(rule.clone());
            }
        }
        None
    }

    /// Check pruning rules
    pub fn check_pruning(&self, node: &KnowledgeNode, context: &GardenerContext) -> Option<(PruningRule, PruningSeverity)> {
        // Check minimum age requirement
        let age_hours = node.created_at.elapsed().as_secs_f32() / 3600.0;
        if age_hours < self.config.min_age_for_pruning {
            return None;
        }

        for rule in &self.pruning_rules {
            if self.matches_pruning_condition(&rule.condition, node, context) {
                return Some((rule.clone(), rule.severity));
            }
        }
        None
    }

    fn matches_consolidation_condition(&self, condition: &ConsolidationCondition, node: &KnowledgeNode, context: &GardenerContext) -> bool {
        match condition {
            ConsolidationCondition::HighAccessFrequency { min_accesses, period_hours: _ } => {
                context.access_stats.get(&node.id)
                    .map(|s| s.recent_accesses >= *min_accesses)
                    .unwrap_or(false)
            }
            ConsolidationCondition::HighImportance { threshold } => {
                node.importance >= *threshold
            }
            ConsolidationCondition::CrossDomainHub { min_domains, min_connections } => {
                let domains = context.connected_domains.get(&node.id).map(|d| d.len()).unwrap_or(0);
                let connections = context.connection_count.get(&node.id).copied().unwrap_or(0);
                domains >= *min_domains && connections >= *min_connections
            }
            _ => false
        }
    }

    fn matches_pruning_condition(&self, condition: &PruningCondition, node: &KnowledgeNode, context: &GardenerContext) -> bool {
        match condition {
            PruningCondition::Stale { min_hours } => {
                let last_access = context.access_stats.get(&node.id)
                    .and_then(|s| s.last_access)
                    .map(|t| t.elapsed().as_secs_f32() / 3600.0);
                last_access.map(|h| h >= *min_hours).unwrap_or(true)
            }
            PruningCondition::LowValue { max_importance, max_connections } => {
                let connections = context.connection_count.get(&node.id).copied().unwrap_or(0);
                node.importance <= *max_importance && connections <= *max_connections
            }
            PruningCondition::Incoherent { coherence_score } => {
                context.coherence_scores.get(&node.id)
                    .map(|&c| c < *coherence_score)
                    .unwrap_or(false)
            }
            _ => false
        }
    }
}

/// Context for Gardener evaluation
#[derive(Clone, Default)]
pub struct GardenerContext {
    pub access_stats: HashMap<Uuid, AccessStats>,
    pub connection_count: HashMap<Uuid, usize>,
    pub avg_connection_count: f32,
    pub connected_domains: HashMap<Uuid, Vec<String>>,
    pub coherence_scores: HashMap<Uuid, f32>,
}
```

**Acceptance Criteria**:
- [ ] Gardener evaluates long-term value under 2ms
- [ ] Pruning threshold of -0.5 triggers pruning suggestions
- [ ] Consolidation threshold of 0.7 triggers consolidation
- [ ] Consolidation rules configurable
- [ ] Pruning rules with severity levels
- [ ] Time decay applied to importance
- [ ] Usage pattern analysis functional

---

### 3.3 The Curator Component

#### REQ-STEER-004: Curator Quality Assessment

**Priority**: Critical
**Description**: The system SHALL implement the Curator component for quality assessment and relevance scoring.

```rust
/// The Curator: quality and relevance assessment (Marblestone)
///
/// Evaluates content quality, information integrity, and domain relevance.
/// Ensures knowledge graph maintains high-quality, coherent information.
pub struct Curator {
    /// Quality scoring model
    pub quality_model: QualityModel,

    /// Domain-specific relevance weights
    pub domain_weights: HashMap<Domain, f32>,

    /// Quality thresholds
    pub thresholds: QualityThresholds,

    /// Configuration
    pub config: CuratorConfig,
}

/// Configuration for the Curator
#[derive(Clone, Debug)]
pub struct CuratorConfig {
    /// Minimum acceptable quality score
    pub min_quality: f32,  // Default: 0.3

    /// Weight for completeness in quality
    pub completeness_weight: f32,  // Default: 0.25

    /// Weight for accuracy signals in quality
    pub accuracy_weight: f32,  // Default: 0.30

    /// Weight for clarity in quality
    pub clarity_weight: f32,  // Default: 0.25

    /// Weight for relevance in quality
    pub relevance_weight: f32,  // Default: 0.20

    /// Enable domain-specific weighting
    pub domain_weighting_enabled: bool,  // Default: true
}

impl Default for CuratorConfig {
    fn default() -> Self {
        Self {
            min_quality: 0.3,
            completeness_weight: 0.25,
            accuracy_weight: 0.30,
            clarity_weight: 0.25,
            relevance_weight: 0.20,
            domain_weighting_enabled: true,
        }
    }
}

/// Quality scoring model
#[derive(Clone)]
pub struct QualityModel {
    /// Completeness criteria
    pub completeness_criteria: Vec<CompletenessCriterion>,

    /// Accuracy indicators
    pub accuracy_indicators: Vec<AccuracyIndicator>,

    /// Clarity metrics
    pub clarity_metrics: ClarityMetrics,

    /// Learned quality patterns
    pub learned_patterns: Vec<QualityPattern>,
}

/// Criterion for assessing completeness
#[derive(Clone, Debug)]
pub struct CompletenessCriterion {
    pub name: String,
    pub weight: f32,
    pub check: CompletenessCheck,
}

/// Types of completeness checks
#[derive(Clone, Debug)]
pub enum CompletenessCheck {
    /// Has required metadata fields
    HasMetadata(Vec<String>),
    /// Has minimum embedding dimensions
    HasEmbedding { min_dims: usize },
    /// Has connections to other nodes
    HasConnections { min_count: usize },
    /// Content exceeds minimum length
    MinContentLength { min_chars: usize },
    /// Has domain classification
    HasDomain,
}

/// Indicator of accuracy/reliability
#[derive(Clone, Debug)]
pub struct AccuracyIndicator {
    pub name: String,
    pub weight: f32,
    pub indicator: AccuracySignal,
}

/// Types of accuracy signals
#[derive(Clone, Debug)]
pub enum AccuracySignal {
    /// Source credibility score
    SourceCredibility { min_score: f32 },
    /// Cross-reference validation
    CrossReferenced { min_references: usize },
    /// Temporal consistency
    TemporallyConsistent,
    /// User-verified content
    UserVerified,
    /// Consistency with domain knowledge
    DomainConsistent,
}

/// Metrics for content clarity
#[derive(Clone, Default)]
pub struct ClarityMetrics {
    /// Target readability score
    pub target_readability: f32,
    /// Maximum complexity score
    pub max_complexity: f32,
    /// Minimum structure score
    pub min_structure: f32,
}

/// Learned quality pattern
#[derive(Clone)]
pub struct QualityPattern {
    pub pattern_id: String,
    pub description: String,
    pub positive_weight: f32,  // How much this pattern increases quality
    pub detection_threshold: f32,
}

/// Domain types for relevance weighting
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum Domain {
    Technical,
    Scientific,
    Business,
    Personal,
    Creative,
    Educational,
    General,
    Custom(String),
}

/// Quality thresholds for different actions
#[derive(Clone, Debug)]
pub struct QualityThresholds {
    /// Below this, flag for review
    pub review_threshold: f32,  // Default: 0.4

    /// Below this, suggest enrichment
    pub enrichment_threshold: f32,  // Default: 0.5

    /// Above this, mark as high quality
    pub high_quality_threshold: f32,  // Default: 0.8
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            review_threshold: 0.4,
            enrichment_threshold: 0.5,
            high_quality_threshold: 0.8,
        }
    }
}

impl Curator {
    /// Evaluate quality of a memory node
    ///
    /// `Constraint: Evaluation_Latency < 2ms`
    pub fn evaluate_quality(&self, node: &KnowledgeNode, context: &CuratorContext) -> f32 {
        let start = Instant::now();

        // Calculate component scores
        let completeness = self.assess_completeness(node, context);
        let accuracy = self.assess_accuracy(node, context);
        let clarity = self.assess_clarity(node);
        let relevance = self.assess_relevance(node, context);

        // Weighted combination
        let raw_quality =
            self.config.completeness_weight * completeness +
            self.config.accuracy_weight * accuracy +
            self.config.clarity_weight * clarity +
            self.config.relevance_weight * relevance;

        // Apply domain weighting if enabled
        let quality = if self.config.domain_weighting_enabled {
            self.apply_domain_weighting(raw_quality, node, context)
        } else {
            raw_quality
        };

        // Map to [-1, 1] range (quality 0.5 is neutral)
        let score = (quality - 0.5) * 2.0;

        let _elapsed = start.elapsed();
        score.clamp(-1.0, 1.0)
    }

    /// Assess completeness of node
    fn assess_completeness(&self, node: &KnowledgeNode, context: &CuratorContext) -> f32 {
        let mut total_weight = 0.0;
        let mut weighted_score = 0.0;

        for criterion in &self.quality_model.completeness_criteria {
            let passed = match &criterion.check {
                CompletenessCheck::HasMetadata(fields) => {
                    fields.iter().all(|f| node.metadata.contains_key(f))
                }
                CompletenessCheck::HasEmbedding { min_dims } => {
                    node.embedding.as_ref().map(|e| e.len() >= *min_dims).unwrap_or(false)
                }
                CompletenessCheck::HasConnections { min_count } => {
                    context.connection_count.get(&node.id).copied().unwrap_or(0) >= *min_count
                }
                CompletenessCheck::MinContentLength { min_chars } => {
                    node.content.len() >= *min_chars
                }
                CompletenessCheck::HasDomain => {
                    node.domain.is_some()
                }
            };

            weighted_score += if passed { criterion.weight } else { 0.0 };
            total_weight += criterion.weight;
        }

        if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.5  // Neutral if no criteria
        }
    }

    /// Assess accuracy indicators
    fn assess_accuracy(&self, node: &KnowledgeNode, context: &CuratorContext) -> f32 {
        let mut total_weight = 0.0;
        let mut weighted_score = 0.0;

        for indicator in &self.quality_model.accuracy_indicators {
            let score = match &indicator.indicator {
                AccuracySignal::SourceCredibility { min_score } => {
                    node.source_credibility.map(|c| if c >= *min_score { 1.0 } else { c / min_score }).unwrap_or(0.5)
                }
                AccuracySignal::CrossReferenced { min_references } => {
                    let refs = context.reference_count.get(&node.id).copied().unwrap_or(0);
                    (refs as f32 / *min_references as f32).min(1.0)
                }
                AccuracySignal::UserVerified => {
                    if node.user_verified { 1.0 } else { 0.5 }
                }
                AccuracySignal::DomainConsistent => {
                    context.domain_consistency.get(&node.id).copied().unwrap_or(0.5)
                }
                _ => 0.5
            };

            weighted_score += score * indicator.weight;
            total_weight += indicator.weight;
        }

        if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.5
        }
    }

    /// Assess clarity of content
    fn assess_clarity(&self, node: &KnowledgeNode) -> f32 {
        // Simple heuristic-based clarity assessment
        let content_len = node.content.len();

        // Check for structure indicators
        let has_structure = node.content.contains('\n') || node.content.contains(". ");

        // Check reasonable length
        let length_score = if content_len < 10 {
            0.3
        } else if content_len < 50 {
            0.5
        } else if content_len < 500 {
            0.8
        } else if content_len < 2000 {
            1.0
        } else {
            0.7  // Very long might indicate lack of focus
        };

        // Combine factors
        let structure_bonus = if has_structure { 0.1 } else { 0.0 };
        (length_score + structure_bonus).min(1.0)
    }

    /// Assess relevance to current context
    fn assess_relevance(&self, node: &KnowledgeNode, context: &CuratorContext) -> f32 {
        // Semantic similarity to current context
        let semantic_relevance = context.semantic_similarity.get(&node.id).copied().unwrap_or(0.5);

        // Temporal relevance (recent is more relevant)
        let age_hours = node.created_at.elapsed().as_secs_f32() / 3600.0;
        let temporal_relevance = (-age_hours / 168.0).exp();  // Week half-life

        // Domain match
        let domain_relevance = if let Some(ref ctx_domain) = context.current_domain {
            if node.domain.as_ref() == Some(ctx_domain) { 1.0 } else { 0.5 }
        } else {
            0.7
        };

        // Weighted average
        semantic_relevance * 0.5 + temporal_relevance * 0.25 + domain_relevance * 0.25
    }

    /// Apply domain-specific weighting
    fn apply_domain_weighting(&self, quality: f32, node: &KnowledgeNode, _context: &CuratorContext) -> f32 {
        if let Some(domain) = &node.domain {
            let domain_enum = Domain::Custom(domain.clone());
            let weight = self.domain_weights.get(&domain_enum).copied().unwrap_or(1.0);
            quality * weight
        } else {
            quality
        }
    }

    /// Generate quality improvement suggestions
    pub fn generate_suggestions(&self, node: &KnowledgeNode, quality_score: f32, context: &CuratorContext) -> Vec<SteeringSuggestion> {
        let mut suggestions = Vec::new();

        if quality_score < self.thresholds.enrichment_threshold {
            // Check what's missing
            if node.embedding.is_none() {
                suggestions.push(SteeringSuggestion {
                    suggestion_type: SuggestionType::EnrichMetadata,
                    priority: 0.8,
                    description: "Missing embedding - generate embedding for better retrieval".to_string(),
                    target_node: Some(node.id),
                    parameters: HashMap::new(),
                });
            }

            if context.connection_count.get(&node.id).copied().unwrap_or(0) < 2 {
                suggestions.push(SteeringSuggestion {
                    suggestion_type: SuggestionType::StrengthenConnection,
                    priority: 0.6,
                    description: "Few connections - find related memories to link".to_string(),
                    target_node: Some(node.id),
                    parameters: HashMap::new(),
                });
            }
        }

        if quality_score < self.thresholds.review_threshold {
            suggestions.push(SteeringSuggestion {
                suggestion_type: SuggestionType::RequestClarification,
                priority: 0.9,
                description: "Low quality score - request clarification from user".to_string(),
                target_node: Some(node.id),
                parameters: HashMap::new(),
            });
        }

        suggestions
    }
}

/// Context for Curator evaluation
#[derive(Clone, Default)]
pub struct CuratorContext {
    pub connection_count: HashMap<Uuid, usize>,
    pub reference_count: HashMap<Uuid, usize>,
    pub domain_consistency: HashMap<Uuid, f32>,
    pub semantic_similarity: HashMap<Uuid, f32>,
    pub current_domain: Option<String>,
}
```

**Acceptance Criteria**:
- [ ] Curator evaluates quality under 2ms
- [ ] Completeness criteria configurable
- [ ] Accuracy indicators assessed
- [ ] Clarity metrics applied
- [ ] Domain-specific weighting functional
- [ ] Quality improvement suggestions generated

---

### 3.4 The Thought Assessor Component

#### REQ-STEER-005: Thought Assessor Real-Time Evaluation

**Priority**: Critical
**Description**: The system SHALL implement the Thought Assessor component for real-time evaluation of thoughts and memories.

```rust
/// The Thought Assessor: real-time evaluation (Marblestone)
///
/// Provides immediate evaluation of incoming thoughts/memories
/// for coherence, novelty, and contextual relevance.
pub struct ThoughtAssessor {
    /// Coherence threshold for acceptance
    pub coherence_threshold: f32,  // Default: 0.4

    /// Novelty bonus weight
    pub novelty_weight: f32,  // Default: 0.3

    /// Configuration
    pub config: ThoughtAssessorConfig,

    /// Recent thought cache for novelty detection
    recent_thoughts: RingBuffer<ThoughtSignature>,

    /// Coherence model
    coherence_model: CoherenceModel,
}

/// Configuration for Thought Assessor
#[derive(Clone, Debug)]
pub struct ThoughtAssessorConfig {
    /// Weight for coherence in final score
    pub coherence_weight: f32,  // Default: 0.4

    /// Weight for novelty in final score
    pub novelty_weight: f32,  // Default: 0.3

    /// Weight for contextual fit in final score
    pub context_weight: f32,  // Default: 0.3

    /// Minimum coherence for positive score
    pub min_coherence: f32,  // Default: 0.3

    /// Novelty detection window size
    pub novelty_window: usize,  // Default: 100

    /// High novelty threshold
    pub high_novelty_threshold: f32,  // Default: 0.7

    /// Enable rapid rejection of incoherent thoughts
    pub rapid_rejection_enabled: bool,  // Default: true
}

impl Default for ThoughtAssessorConfig {
    fn default() -> Self {
        Self {
            coherence_weight: 0.4,
            novelty_weight: 0.3,
            context_weight: 0.3,
            min_coherence: 0.3,
            novelty_window: 100,
            high_novelty_threshold: 0.7,
            rapid_rejection_enabled: true,
        }
    }
}

/// Signature of a thought for novelty detection
#[derive(Clone)]
pub struct ThoughtSignature {
    /// Thought ID
    pub id: Uuid,

    /// Embedding (compressed for efficiency)
    pub embedding_hash: u64,

    /// Domain
    pub domain: Option<String>,

    /// Timestamp
    pub timestamp: Instant,

    /// Key concepts
    pub concepts: Vec<String>,
}

/// Model for coherence assessment
#[derive(Clone, Default)]
pub struct CoherenceModel {
    /// Domain coherence rules
    pub domain_rules: HashMap<String, Vec<CoherenceRule>>,

    /// Global coherence patterns
    pub global_patterns: Vec<CoherencePattern>,
}

/// Rule for domain-specific coherence
#[derive(Clone)]
pub struct CoherenceRule {
    pub rule_id: String,
    pub pattern: String,
    pub weight: f32,
}

/// Pattern indicating coherence
#[derive(Clone)]
pub struct CoherencePattern {
    pub pattern_type: CoherencePatternType,
    pub weight: f32,
}

/// Types of coherence patterns
#[derive(Clone, Debug)]
pub enum CoherencePatternType {
    /// Semantic consistency with domain
    SemanticConsistency,
    /// Temporal logical flow
    TemporalFlow,
    /// Causal relationship validity
    CausalValidity,
    /// Reference integrity
    ReferenceIntegrity,
}

/// Ring buffer for recent thoughts
#[derive(Clone)]
pub struct RingBuffer<T> {
    buffer: Vec<T>,
    head: usize,
    capacity: usize,
}

impl<T: Clone> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            head: 0,
            capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(item);
        } else {
            self.buffer[self.head] = item;
        }
        self.head = (self.head + 1) % self.capacity;
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buffer.iter()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl ThoughtAssessor {
    /// Evaluate a thought/memory in real-time
    ///
    /// `Constraint: Evaluation_Latency < 1ms`
    pub fn evaluate_immediate(&self, node: &KnowledgeNode, context: &AssessorContext) -> f32 {
        let start = Instant::now();

        // Rapid rejection for clearly incoherent content
        if self.config.rapid_rejection_enabled {
            if let Some(quick_reject) = self.quick_coherence_check(node) {
                if quick_reject < self.config.min_coherence {
                    return -0.5;  // Negative score for incoherent
                }
            }
        }

        // Full evaluation
        let coherence = self.assess_coherence(node, context);
        let novelty = self.assess_novelty(node);
        let context_fit = self.assess_context_fit(node, context);

        // Weighted combination
        let score =
            self.config.coherence_weight * coherence +
            self.config.novelty_weight * novelty +
            self.config.context_weight * context_fit;

        // Map to [-1, 1] range
        let final_score = (score - 0.5) * 2.0;

        let _elapsed = start.elapsed();
        final_score.clamp(-1.0, 1.0)
    }

    /// Quick coherence check for rapid rejection
    fn quick_coherence_check(&self, node: &KnowledgeNode) -> Option<f32> {
        // Fast heuristic checks
        let content = &node.content;

        // Check for minimal content
        if content.trim().is_empty() {
            return Some(0.0);
        }

        // Check for reasonable structure
        let word_count = content.split_whitespace().count();
        if word_count < 2 {
            return Some(0.2);
        }

        // Check for gibberish patterns
        let alpha_ratio = content.chars().filter(|c| c.is_alphabetic()).count() as f32
            / content.len().max(1) as f32;
        if alpha_ratio < 0.3 {
            return Some(0.1);
        }

        None  // No quick decision, proceed to full evaluation
    }

    /// Assess coherence with existing knowledge
    fn assess_coherence(&self, node: &KnowledgeNode, context: &AssessorContext) -> f32 {
        // Semantic coherence with domain
        let semantic_coherence = context.domain_similarity.get(&node.id).copied().unwrap_or(0.5);

        // Structural coherence
        let structural_coherence = self.assess_structural_coherence(node);

        // Reference coherence (do referenced items exist?)
        let reference_coherence = context.reference_validity.get(&node.id).copied().unwrap_or(1.0);

        // Weighted average
        semantic_coherence * 0.5 + structural_coherence * 0.3 + reference_coherence * 0.2
    }

    /// Assess structural coherence of content
    fn assess_structural_coherence(&self, node: &KnowledgeNode) -> f32 {
        let content = &node.content;

        // Check for balanced structure
        let has_structure = content.contains('.') || content.contains('\n');
        let reasonable_length = content.len() > 10 && content.len() < 10000;

        let mut score = 0.5;
        if has_structure { score += 0.2; }
        if reasonable_length { score += 0.2; }
        if node.embedding.is_some() { score += 0.1; }

        score.min(1.0)
    }

    /// Assess novelty compared to recent thoughts
    fn assess_novelty(&self, node: &KnowledgeNode) -> f32 {
        if self.recent_thoughts.len() == 0 {
            return 0.7;  // First thought gets moderate novelty
        }

        // Compare to recent thoughts
        let node_signature = self.compute_signature(node);

        let mut max_similarity = 0.0f32;
        for recent in self.recent_thoughts.iter() {
            let similarity = self.compare_signatures(&node_signature, recent);
            max_similarity = max_similarity.max(similarity);
        }

        // Novelty is inverse of max similarity
        1.0 - max_similarity
    }

    /// Compute signature for a node
    fn compute_signature(&self, node: &KnowledgeNode) -> ThoughtSignature {
        // Simple hash-based signature
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        node.content.hash(&mut hasher);
        let embedding_hash = hasher.finish();

        // Extract key concepts (simplified)
        let concepts: Vec<String> = node.content
            .split_whitespace()
            .filter(|w| w.len() > 4)
            .take(5)
            .map(|s| s.to_lowercase())
            .collect();

        ThoughtSignature {
            id: node.id,
            embedding_hash,
            domain: node.domain.clone(),
            timestamp: Instant::now(),
            concepts,
        }
    }

    /// Compare two thought signatures
    fn compare_signatures(&self, a: &ThoughtSignature, b: &ThoughtSignature) -> f32 {
        // Hash similarity
        let hash_match = if a.embedding_hash == b.embedding_hash { 0.5 } else { 0.0 };

        // Domain match
        let domain_match = match (&a.domain, &b.domain) {
            (Some(ad), Some(bd)) if ad == bd => 0.3,
            _ => 0.0,
        };

        // Concept overlap
        let concept_overlap = a.concepts.iter()
            .filter(|c| b.concepts.contains(c))
            .count() as f32 / a.concepts.len().max(1) as f32;

        hash_match + domain_match + concept_overlap * 0.2
    }

    /// Assess contextual fit
    fn assess_context_fit(&self, node: &KnowledgeNode, context: &AssessorContext) -> f32 {
        // Query relevance
        let query_relevance = context.query_similarity.get(&node.id).copied().unwrap_or(0.5);

        // Session topic fit
        let session_fit = if let Some(ref session_topic) = context.session_topic {
            if node.domain.as_ref() == Some(session_topic) { 0.9 } else { 0.5 }
        } else {
            0.6
        };

        // Temporal fit (recent context)
        let temporal_fit = if context.recent_context_age_ms < 5000 { 0.8 } else { 0.5 };

        // Weighted average
        query_relevance * 0.5 + session_fit * 0.3 + temporal_fit * 0.2
    }

    /// Record thought for novelty tracking
    pub fn record_thought(&mut self, node: &KnowledgeNode) {
        let signature = self.compute_signature(node);
        self.recent_thoughts.push(signature);
    }
}

/// Context for Thought Assessor evaluation
#[derive(Clone, Default)]
pub struct AssessorContext {
    pub domain_similarity: HashMap<Uuid, f32>,
    pub reference_validity: HashMap<Uuid, f32>,
    pub query_similarity: HashMap<Uuid, f32>,
    pub session_topic: Option<String>,
    pub recent_context_age_ms: u64,
}
```

**Acceptance Criteria**:
- [ ] Thought Assessor evaluates under 1ms
- [ ] Coherence threshold of 0.4 enforced
- [ ] Novelty detection functional
- [ ] Rapid rejection for incoherent content
- [ ] Context fit assessment included
- [ ] Recent thought tracking operational

---

### 3.5 Unified Evaluation and Dopamine Integration

#### REQ-STEER-006: Unified Steering Evaluation

**Priority**: Critical
**Description**: The system SHALL provide a unified evaluation interface that combines all three components and returns SteeringReward.

```rust
impl SteeringSubsystem {
    /// Evaluate a memory/thought and return steering reward
    ///
    /// This is the primary entry point called with every store_memory operation.
    ///
    /// `Constraint: Total_Evaluation_Latency < 5ms`
    pub async fn evaluate(&mut self, node: &KnowledgeNode, context: &SteeringContext) -> SteeringReward {
        let start = Instant::now();
        let evaluation_id = Uuid::new_v4();

        // Prepare component contexts
        let gardener_ctx = self.prepare_gardener_context(node, context);
        let curator_ctx = self.prepare_curator_context(node, context);
        let assessor_ctx = self.prepare_assessor_context(node, context);

        // Execute component evaluations (potentially in parallel for larger systems)
        let gardener_score = self.gardener.evaluate_long_term_value(node, &gardener_ctx);
        let curator_score = self.curator.evaluate_quality(node, &curator_ctx);
        let assessor_score = self.thought_assessor.evaluate_immediate(node, &assessor_ctx);

        // Compute weighted final reward
        let reward =
            self.config.gardener_weight * gardener_score +
            self.config.curator_weight * curator_score +
            self.config.assessor_weight * assessor_score;

        // Clamp to valid range
        let clamped_reward = reward.clamp(-1.0, 1.0);

        // Generate explanation
        let explanation = self.generate_explanation(
            clamped_reward,
            gardener_score,
            curator_score,
            assessor_score,
        );

        // Generate suggestions if enabled
        let suggestions = if self.config.suggestions_enabled {
            self.generate_unified_suggestions(
                node,
                gardener_score,
                curator_score,
                assessor_score,
                &gardener_ctx,
                &curator_ctx,
            )
        } else {
            Vec::new()
        };

        // Calculate confidence based on component agreement
        let confidence = self.calculate_confidence(gardener_score, curator_score, assessor_score);

        // Record thought for novelty tracking
        self.thought_assessor.record_thought(node);

        // Compute latency
        let latency = start.elapsed();

        // Update session state
        {
            let mut state = self.session_state.write().await;
            state.evaluation_count += 1;
            state.avg_reward = (state.avg_reward * (state.evaluation_count - 1) as f32 + clamped_reward)
                / state.evaluation_count as f32;
            state.reward_history.push(SteeringRewardEntry {
                timestamp: Instant::now(),
                node_id: node.id,
                reward: clamped_reward,
                components: SteeringComponents {
                    gardener_score,
                    curator_score,
                    assessor_score,
                },
            });
            state.last_evaluation = Some(Instant::now());
        }

        // Update metrics
        self.metrics.record_evaluation(latency, clamped_reward);

        // Trigger dopamine feedback if enabled
        if self.config.dopamine_integration_enabled {
            self.dopamine_feedback.send_reward(clamped_reward, &SteeringComponents {
                gardener_score,
                curator_score,
                assessor_score,
            }).await;
        }

        SteeringReward {
            reward: clamped_reward,
            components: SteeringComponents {
                gardener_score,
                curator_score,
                assessor_score,
            },
            explanation,
            suggestions,
            confidence,
            latency,
            metadata: SteeringMetadata {
                evaluation_id,
                timestamp: chrono::Utc::now(),
                node_id: node.id,
                domain: node.domain.clone(),
                novelty: (1.0 + assessor_score) / 2.0,  // Map to [0, 1]
                coherence: (1.0 + curator_score) / 2.0,
            },
        }
    }

    /// Generate human-readable explanation for the reward
    fn generate_explanation(
        &self,
        reward: f32,
        gardener: f32,
        curator: f32,
        assessor: f32,
    ) -> String {
        let sentiment = if reward > 0.3 {
            "positive"
        } else if reward < -0.3 {
            "negative"
        } else {
            "neutral"
        };

        let dominant = if gardener.abs() >= curator.abs() && gardener.abs() >= assessor.abs() {
            format!("long-term value ({})", if gardener > 0.0 { "high" } else { "low" })
        } else if curator.abs() >= assessor.abs() {
            format!("quality ({})", if curator > 0.0 { "good" } else { "needs improvement" })
        } else {
            format!("immediate relevance ({})", if assessor > 0.0 { "relevant" } else { "less relevant" })
        };

        format!(
            "Overall {} signal (reward: {:.2}). Dominant factor: {}. \
            Components - Gardener: {:.2}, Curator: {:.2}, Assessor: {:.2}",
            sentiment, reward, dominant, gardener, curator, assessor
        )
    }

    /// Calculate confidence based on component agreement
    fn calculate_confidence(&self, gardener: f32, curator: f32, assessor: f32) -> f32 {
        // Check sign agreement
        let signs = [gardener.signum(), curator.signum(), assessor.signum()];
        let positive_count = signs.iter().filter(|&&s| s > 0.0).count();
        let negative_count = signs.iter().filter(|&&s| s < 0.0).count();

        // High agreement = high confidence
        let agreement = if positive_count >= 2 || negative_count >= 2 {
            0.8
        } else {
            0.5
        };

        // Adjust by magnitude variance
        let mean = (gardener + curator + assessor) / 3.0;
        let variance = ((gardener - mean).powi(2) + (curator - mean).powi(2) + (assessor - mean).powi(2)) / 3.0;

        // Lower variance = higher confidence
        let variance_factor = 1.0 - variance.sqrt();

        (agreement * 0.6 + variance_factor * 0.4).clamp(0.0, 1.0)
    }

    /// Generate unified suggestions from all components
    fn generate_unified_suggestions(
        &self,
        node: &KnowledgeNode,
        gardener_score: f32,
        curator_score: f32,
        assessor_score: f32,
        gardener_ctx: &GardenerContext,
        curator_ctx: &CuratorContext,
    ) -> Vec<SteeringSuggestion> {
        let mut all_suggestions = Vec::new();

        // Gardener suggestions
        if gardener_score < self.gardener.prune_threshold {
            all_suggestions.push(SteeringSuggestion {
                suggestion_type: SuggestionType::Prune,
                priority: (-gardener_score).min(1.0),
                description: "Low long-term value - consider pruning".to_string(),
                target_node: Some(node.id),
                parameters: HashMap::new(),
            });
        } else if gardener_score > self.gardener.consolidation_threshold {
            all_suggestions.push(SteeringSuggestion {
                suggestion_type: SuggestionType::Consolidate,
                priority: gardener_score,
                description: "High long-term value - consider consolidation".to_string(),
                target_node: Some(node.id),
                parameters: HashMap::new(),
            });
        }

        // Curator suggestions
        let curator_suggestions = self.curator.generate_suggestions(node, (curator_score + 1.0) / 2.0, curator_ctx);
        all_suggestions.extend(curator_suggestions);

        // Assessor suggestions
        if assessor_score < -0.3 {
            all_suggestions.push(SteeringSuggestion {
                suggestion_type: SuggestionType::DreamReview,
                priority: 0.7,
                description: "Low coherence - flag for dream-time review".to_string(),
                target_node: Some(node.id),
                parameters: HashMap::new(),
            });
        }

        // Sort by priority and limit
        all_suggestions.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        all_suggestions.truncate(self.config.max_suggestions);

        all_suggestions
    }

    // Context preparation methods
    fn prepare_gardener_context(&self, _node: &KnowledgeNode, context: &SteeringContext) -> GardenerContext {
        GardenerContext {
            access_stats: context.access_stats.clone(),
            connection_count: context.connection_count.clone(),
            avg_connection_count: context.avg_connection_count,
            connected_domains: context.connected_domains.clone(),
            coherence_scores: context.coherence_scores.clone(),
        }
    }

    fn prepare_curator_context(&self, _node: &KnowledgeNode, context: &SteeringContext) -> CuratorContext {
        CuratorContext {
            connection_count: context.connection_count.clone(),
            reference_count: context.reference_count.clone(),
            domain_consistency: context.domain_consistency.clone(),
            semantic_similarity: context.semantic_similarity.clone(),
            current_domain: context.current_domain.clone(),
        }
    }

    fn prepare_assessor_context(&self, _node: &KnowledgeNode, context: &SteeringContext) -> AssessorContext {
        AssessorContext {
            domain_similarity: context.semantic_similarity.clone(),
            reference_validity: context.reference_validity.clone(),
            query_similarity: context.query_similarity.clone(),
            session_topic: context.session_topic.clone(),
            recent_context_age_ms: context.recent_context_age_ms,
        }
    }
}

/// Unified context for steering evaluation
#[derive(Clone, Default)]
pub struct SteeringContext {
    pub access_stats: HashMap<Uuid, AccessStats>,
    pub connection_count: HashMap<Uuid, usize>,
    pub avg_connection_count: f32,
    pub connected_domains: HashMap<Uuid, Vec<String>>,
    pub coherence_scores: HashMap<Uuid, f32>,
    pub reference_count: HashMap<Uuid, usize>,
    pub domain_consistency: HashMap<Uuid, f32>,
    pub semantic_similarity: HashMap<Uuid, f32>,
    pub current_domain: Option<String>,
    pub reference_validity: HashMap<Uuid, f32>,
    pub query_similarity: HashMap<Uuid, f32>,
    pub session_topic: Option<String>,
    pub recent_context_age_ms: u64,
}
```

**Acceptance Criteria**:
- [ ] Unified evaluation completes under 5ms
- [ ] All three component scores combined properly
- [ ] Explanation generated for every reward
- [ ] Suggestions limited to max_suggestions
- [ ] Confidence calculation reflects agreement
- [ ] Session state updated atomically

---

#### REQ-STEER-007: Dopamine Feedback Integration

**Priority**: Critical
**Description**: The system SHALL integrate steering rewards with Module 10 dopamine modulation.

```rust
/// Steering-to-Dopamine feedback pathway (Marblestone integration)
///
/// Converts steering rewards into dopamine modulation signals,
/// implementing the separation of learning from steering.
pub struct SteeringDopamineFeedback {
    /// Reference to neuromodulation controller
    neuromod_controller: Option<Arc<RwLock<NeuromodulationController>>>,

    /// Feedback configuration
    config: DopamineFeedbackConfig,

    /// Feedback history for analysis
    history: RingBuffer<DopamineFeedbackEvent>,
}

/// Configuration for dopamine feedback
#[derive(Clone, Debug)]
pub struct DopamineFeedbackConfig {
    /// Scale factor for positive rewards
    pub positive_scale: f32,  // Default: 0.3

    /// Scale factor for negative rewards
    pub negative_scale: f32,  // Default: 0.2

    /// Minimum reward magnitude to trigger feedback
    pub min_magnitude: f32,  // Default: 0.1

    /// Maximum dopamine change per feedback
    pub max_delta: f32,  // Default: 0.2

    /// Decay rate for sustained rewards
    pub decay_rate: f32,  // Default: 0.1

    /// Enable surprise-based amplification
    pub surprise_amplification: bool,  // Default: true
}

impl Default for DopamineFeedbackConfig {
    fn default() -> Self {
        Self {
            positive_scale: 0.3,
            negative_scale: 0.2,
            min_magnitude: 0.1,
            max_delta: 0.2,
            decay_rate: 0.1,
            surprise_amplification: true,
        }
    }
}

/// Record of dopamine feedback event
#[derive(Clone)]
pub struct DopamineFeedbackEvent {
    pub timestamp: Instant,
    pub steering_reward: f32,
    pub dopamine_delta: f32,
    pub components: SteeringComponents,
    pub surprise_factor: f32,
}

impl SteeringDopamineFeedback {
    /// Create new feedback handler
    pub fn new(config: DopamineFeedbackConfig) -> Self {
        Self {
            neuromod_controller: None,
            config,
            history: RingBuffer::new(100),
        }
    }

    /// Connect to neuromodulation controller
    pub fn connect(&mut self, controller: Arc<RwLock<NeuromodulationController>>) {
        self.neuromod_controller = Some(controller);
    }

    /// Send steering reward as dopamine modulation
    ///
    /// This implements Marblestone's separation: steering provides reward signals
    /// that influence learning through dopamine, without directly modifying weights.
    ///
    /// `Constraint: Feedback_Latency < 1ms`
    pub async fn send_reward(&mut self, reward: f32, components: &SteeringComponents) {
        // Check minimum magnitude
        if reward.abs() < self.config.min_magnitude {
            return;
        }

        // Calculate base dopamine delta
        let scale = if reward > 0.0 {
            self.config.positive_scale
        } else {
            self.config.negative_scale
        };

        let mut delta = reward * scale;

        // Apply surprise amplification if enabled
        let surprise_factor = if self.config.surprise_amplification {
            self.calculate_surprise(reward)
        } else {
            1.0
        };
        delta *= surprise_factor;

        // Clamp to max delta
        delta = delta.clamp(-self.config.max_delta, self.config.max_delta);

        // Send to neuromodulation controller
        if let Some(ref controller) = self.neuromod_controller {
            let mut ctrl = controller.write().await;

            // Modulate dopamine based on reward
            let new_dopamine = (ctrl.dopamine + delta).clamp(0.0, 1.0);
            ctrl.dopamine = new_dopamine;

            // Record event
            self.history.push(DopamineFeedbackEvent {
                timestamp: Instant::now(),
                steering_reward: reward,
                dopamine_delta: delta,
                components: components.clone(),
                surprise_factor,
            });
        }
    }

    /// Calculate surprise factor based on reward history
    fn calculate_surprise(&self, current_reward: f32) -> f32 {
        if self.history.len() < 5 {
            return 1.0;  // Not enough history, no amplification
        }

        // Calculate recent average
        let recent_avg: f32 = self.history.iter()
            .rev()
            .take(10)
            .map(|e| e.steering_reward)
            .sum::<f32>() / 10.0;

        // Surprise is deviation from expectation
        let deviation = (current_reward - recent_avg).abs();

        // Map to surprise factor [1.0, 2.0]
        1.0 + deviation.min(1.0)
    }

    /// Get feedback statistics
    pub fn get_stats(&self) -> DopamineFeedbackStats {
        let events: Vec<_> = self.history.iter().collect();

        DopamineFeedbackStats {
            total_events: events.len() as u64,
            avg_reward: events.iter().map(|e| e.steering_reward).sum::<f32>() / events.len().max(1) as f32,
            avg_delta: events.iter().map(|e| e.dopamine_delta).sum::<f32>() / events.len().max(1) as f32,
            positive_events: events.iter().filter(|e| e.steering_reward > 0.0).count() as u64,
            negative_events: events.iter().filter(|e| e.steering_reward < 0.0).count() as u64,
        }
    }
}

/// Statistics for dopamine feedback
#[derive(Clone, Debug, Serialize)]
pub struct DopamineFeedbackStats {
    pub total_events: u64,
    pub avg_reward: f32,
    pub avg_delta: f32,
    pub positive_events: u64,
    pub negative_events: u64,
}
```

**Acceptance Criteria**:
- [ ] Dopamine feedback under 1ms latency
- [ ] Positive rewards increase dopamine
- [ ] Negative rewards decrease dopamine
- [ ] Surprise amplification functional
- [ ] Maximum delta enforced (0.2)
- [ ] Integration with Module 10 verified

---

### 3.6 MCP Tool Integration

#### REQ-STEER-008: get_steering_reward MCP Tool

**Priority**: High
**Description**: The system SHALL expose steering evaluation through MCP tool.

```rust
use serde::{Deserialize, Serialize};

/// MCP tool request for steering evaluation
#[derive(Clone, Debug, Deserialize)]
pub struct GetSteeringRewardRequest {
    /// Node ID to evaluate
    pub node_id: Uuid,

    /// Session ID for context
    pub session_id: Option<Uuid>,

    /// Include detailed component breakdown
    #[serde(default = "default_include_components")]
    pub include_components: bool,

    /// Include suggestions
    #[serde(default = "default_include_suggestions")]
    pub include_suggestions: bool,

    /// Force evaluation even if recently evaluated
    #[serde(default)]
    pub force: bool,
}

fn default_include_components() -> bool { true }
fn default_include_suggestions() -> bool { true }

/// MCP tool response for steering evaluation
#[derive(Clone, Debug, Serialize)]
pub struct GetSteeringRewardResponse {
    /// Steering reward
    pub reward: SteeringRewardOutput,

    /// Dopamine feedback applied
    pub dopamine_feedback: DopamineFeedbackOutput,

    /// Cognitive pulse header
    pub pulse: CognitivePulse,

    /// Processing latency
    pub latency_ms: f32,
}

/// Serializable steering reward output
#[derive(Clone, Debug, Serialize)]
pub struct SteeringRewardOutput {
    /// Overall reward [-1.0, 1.0]
    pub reward: f32,

    /// Component breakdown (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub components: Option<SteeringComponentsOutput>,

    /// Explanation
    pub explanation: String,

    /// Suggestions (if requested)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<SteeringSuggestionOutput>,

    /// Confidence [0.0, 1.0]
    pub confidence: f32,
}

/// Serializable component output
#[derive(Clone, Debug, Serialize)]
pub struct SteeringComponentsOutput {
    pub gardener: f32,
    pub curator: f32,
    pub assessor: f32,
}

/// Serializable suggestion output
#[derive(Clone, Debug, Serialize)]
pub struct SteeringSuggestionOutput {
    pub suggestion_type: String,
    pub priority: f32,
    pub description: String,
}

/// Dopamine feedback output
#[derive(Clone, Debug, Serialize)]
pub struct DopamineFeedbackOutput {
    /// Whether feedback was applied
    pub applied: bool,

    /// Delta applied to dopamine
    pub delta: f32,

    /// New dopamine level
    pub new_dopamine_level: f32,
}

/// MCP tool handler for steering
pub struct SteeringToolHandler {
    /// Reference to steering subsystem
    steering: Arc<RwLock<SteeringSubsystem>>,

    /// Graph for node lookup
    graph: Arc<RwLock<KnowledgeGraph>>,
}

impl SteeringToolHandler {
    /// Handle get_steering_reward request
    ///
    /// `Constraint: Total_Handler_Latency < 10ms`
    pub async fn handle(&self, request: GetSteeringRewardRequest) -> Result<GetSteeringRewardResponse, SteeringError> {
        let start = Instant::now();

        // Look up node
        let graph = self.graph.read().await;
        let node = graph.get_node(&request.node_id)
            .ok_or(SteeringError::NodeNotFound(request.node_id))?;

        // Prepare context
        let context = self.prepare_context(&node, &graph).await;

        // Evaluate
        let mut steering = self.steering.write().await;
        let reward = steering.evaluate(&node, &context).await;

        // Get dopamine state
        let dopamine_output = if steering.config.dopamine_integration_enabled {
            if let Some(ref controller) = steering.dopamine_feedback.neuromod_controller {
                let ctrl = controller.read().await;
                DopamineFeedbackOutput {
                    applied: true,
                    delta: reward.reward * 0.3,  // Approximate delta
                    new_dopamine_level: ctrl.dopamine,
                }
            } else {
                DopamineFeedbackOutput {
                    applied: false,
                    delta: 0.0,
                    new_dopamine_level: 0.5,
                }
            }
        } else {
            DopamineFeedbackOutput {
                applied: false,
                delta: 0.0,
                new_dopamine_level: 0.5,
            }
        };

        // Build response
        let response = GetSteeringRewardResponse {
            reward: SteeringRewardOutput {
                reward: reward.reward,
                components: if request.include_components {
                    Some(SteeringComponentsOutput {
                        gardener: reward.components.gardener_score,
                        curator: reward.components.curator_score,
                        assessor: reward.components.assessor_score,
                    })
                } else {
                    None
                },
                explanation: reward.explanation,
                suggestions: if request.include_suggestions {
                    reward.suggestions.iter().map(|s| SteeringSuggestionOutput {
                        suggestion_type: format!("{:?}", s.suggestion_type),
                        priority: s.priority,
                        description: s.description.clone(),
                    }).collect()
                } else {
                    Vec::new()
                },
                confidence: reward.confidence,
            },
            dopamine_feedback: dopamine_output,
            pulse: CognitivePulse {
                entropy: reward.metadata.novelty,
                coherence: reward.metadata.coherence,
                suggested_action: if reward.reward > 0.3 {
                    "continue".to_string()
                } else if reward.reward < -0.3 {
                    "review".to_string()
                } else {
                    "monitor".to_string()
                },
            },
            latency_ms: start.elapsed().as_secs_f32() * 1000.0,
        };

        Ok(response)
    }

    async fn prepare_context(&self, _node: &KnowledgeNode, _graph: &KnowledgeGraph) -> SteeringContext {
        // Build context from graph state
        SteeringContext::default()
    }
}

/// Cognitive pulse header
#[derive(Clone, Debug, Serialize)]
pub struct CognitivePulse {
    pub entropy: f32,
    pub coherence: f32,
    pub suggested_action: String,
}
```

**Acceptance Criteria**:
- [ ] MCP handler under 10ms latency
- [ ] Node lookup with error handling
- [ ] Component breakdown optional
- [ ] Suggestions optional
- [ ] Dopamine feedback status included
- [ ] Cognitive pulse in response

---

## 4. Error Handling

### 4.1 Error Types

```rust
use thiserror::Error;

/// Errors for Steering Subsystem
#[derive(Debug, Error)]
pub enum SteeringError {
    /// Node not found in graph
    #[error("Node not found: {0}")]
    NodeNotFound(Uuid),

    /// Session not found
    #[error("Session not found: {0}")]
    SessionNotFound(Uuid),

    /// Evaluation timeout
    #[error("Steering evaluation timed out after {0}ms")]
    EvaluationTimeout(u64),

    /// Component failure
    #[error("Component evaluation failed: {0}")]
    ComponentFailure(String),

    /// Dopamine integration failure
    #[error("Dopamine feedback failed: {0}")]
    DopamineFeedbackFailed(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Invalid reward value
    #[error("Invalid reward value: {0}")]
    InvalidReward(f32),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl SteeringError {
    /// Get MCP error code
    pub fn error_code(&self) -> i32 {
        match self {
            Self::NodeNotFound(_) => -32100,
            Self::SessionNotFound(_) => -32101,
            Self::EvaluationTimeout(_) => -32102,
            Self::ComponentFailure(_) => -32103,
            Self::DopamineFeedbackFailed(_) => -32104,
            Self::ConfigurationError(_) => -32105,
            Self::InvalidReward(_) => -32106,
            Self::Internal(_) => -32603,
        }
    }
}
```

**Acceptance Criteria**:
- [ ] All error types defined
- [ ] MCP error codes assigned
- [ ] Descriptive error messages
- [ ] Error context preserved
- [ ] Graceful degradation possible

---

## 5. User Stories

### 5.1 Developer User Stories

| ID | As a... | I want to... | So that... | Priority |
|----|---------|--------------|------------|----------|
| US-STEER-001 | Developer | Receive SteeringReward with every store_memory call | I can understand how the system evaluates my content | Must |
| US-STEER-002 | Developer | See component breakdown (Gardener/Curator/Assessor) | I can diagnose why content received certain scores | Must |
| US-STEER-003 | Developer | Receive actionable suggestions | I can improve content quality proactively | Should |
| US-STEER-004 | Developer | Configure component weights | I can tune steering for my use case | Should |
| US-STEER-005 | Developer | Access steering history | I can analyze patterns in my content quality | Could |

### 5.2 System User Stories

| ID | As a... | I want to... | So that... | Priority |
|----|---------|--------------|------------|----------|
| US-STEER-006 | Knowledge Graph | Receive pruning recommendations | I can maintain memory health | Must |
| US-STEER-007 | Learning Subsystem | Receive dopamine feedback | I can adjust learning rate appropriately | Must |
| US-STEER-008 | Dream Layer | Receive flagged content for review | I can process low-coherence memories | Should |
| US-STEER-009 | Active Inference | Access steering signals | I can inform epistemic action selection | Should |

---

## 6. API Contracts

### 6.1 Internal API

```rust
/// Primary steering interface
pub trait SteeringInterface {
    /// Evaluate a node and return steering reward
    async fn evaluate(&mut self, node: &KnowledgeNode, context: &SteeringContext) -> SteeringReward;

    /// Get current session state
    async fn get_session_state(&self) -> SteeringSessionState;

    /// Update configuration
    fn update_config(&mut self, config: SteeringConfig);

    /// Get component health status
    fn health_check(&self) -> ComponentHealthStatus;
}

/// Component health status
#[derive(Clone, Debug, Serialize)]
pub struct ComponentHealthStatus {
    pub gardener_healthy: bool,
    pub curator_healthy: bool,
    pub assessor_healthy: bool,
    pub dopamine_connected: bool,
    pub last_evaluation_latency_ms: f32,
}
```

### 6.2 MCP Tool Schema

```json
{
  "name": "get_steering_reward",
  "description": "Evaluate a memory node using Marblestone's Steering Subsystem and return reward signal with component breakdown",
  "inputSchema": {
    "type": "object",
    "properties": {
      "node_id": {
        "type": "string",
        "format": "uuid",
        "description": "UUID of the node to evaluate"
      },
      "session_id": {
        "type": "string",
        "format": "uuid",
        "description": "Optional session ID for context"
      },
      "include_components": {
        "type": "boolean",
        "default": true,
        "description": "Include Gardener/Curator/Assessor breakdown"
      },
      "include_suggestions": {
        "type": "boolean",
        "default": true,
        "description": "Include actionable suggestions"
      }
    },
    "required": ["node_id"]
  }
}
```

---

## 7. Test Plan

### 7.1 Unit Tests

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| T-STEER-001 | Gardener long-term value calculation | Score in [-1, 1], latency <2ms |
| T-STEER-002 | Curator quality assessment | Score in [-1, 1], latency <2ms |
| T-STEER-003 | Thought Assessor real-time eval | Score in [-1, 1], latency <1ms |
| T-STEER-004 | Unified evaluation | All components combined, latency <5ms |
| T-STEER-005 | Dopamine feedback integration | Delta applied correctly |
| T-STEER-006 | Pruning threshold triggers | Suggestions generated at -0.5 |
| T-STEER-007 | Consolidation threshold triggers | Suggestions generated at 0.7 |
| T-STEER-008 | Novelty detection | Recent duplicates scored lower |
| T-STEER-009 | Confidence calculation | Agreement increases confidence |
| T-STEER-010 | Suggestion generation | Max suggestions enforced |

### 7.2 Integration Tests

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| T-STEER-INT-001 | store_memory with steering | SteeringReward returned |
| T-STEER-INT-002 | Module 10 dopamine integration | Dopamine level changes |
| T-STEER-INT-003 | MCP tool round-trip | Valid response under 10ms |
| T-STEER-INT-004 | Session state persistence | State maintained across calls |
| T-STEER-INT-005 | Dream layer flagging | Low-coherence flagged for review |

### 7.3 Performance Benchmarks

| Benchmark | Target | Measurement |
|-----------|--------|-------------|
| Single evaluation | <5ms | P99 latency |
| Gardener component | <2ms | P99 latency |
| Curator component | <2ms | P99 latency |
| Assessor component | <1ms | P99 latency |
| Dopamine feedback | <1ms | P99 latency |
| Throughput | >200 eval/sec | Sustained rate |

---

## 8. Performance Requirements

### 8.1 Latency Budgets

| Operation | Budget | Notes |
|-----------|--------|-------|
| Gardener evaluation | <2ms | Long-term value |
| Curator evaluation | <2ms | Quality assessment |
| Thought Assessor | <1ms | Real-time check |
| Unified evaluation | <5ms | Total including combination |
| Dopamine feedback | <1ms | Modulation signal |
| MCP handler | <10ms | Including serialization |

### 8.2 Memory Constraints

| Component | Budget | Notes |
|-----------|--------|-------|
| Session state | <100KB | Per session |
| Recent thoughts buffer | <200KB | For novelty detection |
| Feedback history | <50KB | Dopamine feedback tracking |
| Total per session | <500KB | Including all buffers |

---

## 9. Dependencies

### 9.1 Internal Module Dependencies

| Dependency | Purpose | Interface |
|------------|---------|-----------|
| Module 2: Core Infrastructure | KnowledgeNode, Uuid | `context-graph-core` |
| Module 4: Knowledge Graph | Node lookup | `KnowledgeGraph` trait |
| Module 10: Neuromodulation | Dopamine modulation | `NeuromodulationController` |
| Module 12: Active Inference | Epistemic action context | `ActiveInferenceEngine` |

### 9.2 External Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| tokio | 1.35+ | Async runtime |
| uuid | 1.6+ | Node identifiers |
| serde | 1.0+ | Serialization |
| chrono | 0.4+ | Timestamps |
| thiserror | 1.0+ | Error handling |

---

## 10. Traceability Matrix

### 10.1 Requirements to PRD

| Requirement | PRD Section | Description |
|-------------|-------------|-------------|
| REQ-STEER-001 | 7.3 (Steering) | SteeringSubsystem struct |
| REQ-STEER-002 | 7.3 (Steering) | SteeringReward output |
| REQ-STEER-003 | 7.3.1 (Gardener) | Long-term value assessment |
| REQ-STEER-004 | 7.3.2 (Curator) | Quality assessment |
| REQ-STEER-005 | 7.3.3 (Assessor) | Real-time evaluation |
| REQ-STEER-006 | 7.3 (Steering) | Unified evaluation |
| REQ-STEER-007 | 7.2/7.3 (Integration) | Dopamine feedback |
| REQ-STEER-008 | 5.2 (MCP Tools) | get_steering_reward tool |

### 10.2 Requirements to Tests

| Requirement | Test Cases |
|-------------|------------|
| REQ-STEER-001 | T-STEER-001, T-STEER-INT-001 |
| REQ-STEER-002 | T-STEER-004, T-STEER-009 |
| REQ-STEER-003 | T-STEER-001, T-STEER-006 |
| REQ-STEER-004 | T-STEER-002, T-STEER-007 |
| REQ-STEER-005 | T-STEER-003, T-STEER-008 |
| REQ-STEER-006 | T-STEER-004, T-STEER-010 |
| REQ-STEER-007 | T-STEER-005, T-STEER-INT-002 |
| REQ-STEER-008 | T-STEER-INT-003 |

---

## 11. Quality Gates

| Gate | Criteria | Verification |
|------|----------|--------------|
| Unit Test Coverage | >90% | cargo tarpaulin |
| Integration Tests | All pass | cargo test --test integration |
| Performance Targets | All met | cargo bench |
| Documentation | All public APIs | cargo doc |
| No Panics | Zero unwrap() in hot path | clippy lint |
| Memory Usage | <500KB/session | Memory profiling |

---

## 12. Appendix

### 12.1 Marblestone Reference

Adam Marblestone's research on neural architectures proposes separating learning mechanisms from steering/reward mechanisms. Key papers:

- Marblestone, A., et al. (2016). "Toward an Integration of Deep Learning and Neuroscience"
- Whittington, J. & Bogacz, R. (2017). "An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity"

### 12.2 Component Interaction Diagram

```
                    +------------------+
                    | store_memory()   |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | SteeringSubsystem|
                    +--------+---------+
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
    +----------+      +----------+      +------------+
    | Gardener |      | Curator  |      | Thought    |
    | (long-   |      | (quality)|      | Assessor   |
    | term)    |      |          |      | (real-time)|
    +----+-----+      +----+-----+      +-----+------+
         |                 |                  |
         +--------+--------+--------+---------+
                  |                 |
                  v                 v
         +----------------+   +--------------+
         | SteeringReward |   | Dopamine     |
         | (output)       |   | Feedback     |
         +----------------+   +------+-------+
                                     |
                                     v
                             +----------------+
                             | Module 10      |
                             | Neuromodulation|
                             +----------------+
```

### 12.3 Configuration Example

```toml
[steering]
enabled = true

[steering.weights]
gardener = 0.35
curator = 0.35
assessor = 0.30

[steering.thresholds]
positive = 0.3
negative = -0.3
prune = -0.5
consolidate = 0.7

[steering.gardener]
min_age_for_pruning = 24.0
access_frequency_weight = 0.3
recency_weight = 0.2
connection_weight = 0.25
importance_weight = 0.25

[steering.curator]
min_quality = 0.3
completeness_weight = 0.25
accuracy_weight = 0.30
clarity_weight = 0.25
relevance_weight = 0.20

[steering.assessor]
coherence_threshold = 0.4
novelty_weight = 0.3
novelty_window = 100

[steering.dopamine]
enabled = true
positive_scale = 0.3
negative_scale = 0.2
min_magnitude = 0.1
max_delta = 0.2
```

---

*Document Version: 1.0.0*
*Generated: 2025-12-31*
*Agent: #12.5/28 - Steering Subsystem Specification*
