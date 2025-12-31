# Module 12.5: Steering Subsystem - Technical Specification

```yaml
metadata:
  id: TECH-STEER-012.5
  version: 1.0.0
  module: Steering Subsystem
  phase: 5
  status: draft
  created: 2025-12-31
  dependencies:
    - TECH-NEURO-010 (Module 10: Neuromodulation)
    - TECH-GRAPH-004 (Module 4: Knowledge Graph)
    - TECH-DREAM-009 (Module 9: Dream Layer)
  functional_spec_ref: SPEC-STEER-012.5
  author: Architecture Agent
```

---

## 1. Architecture Overview

### 1.1 Marblestone Steering Concept

The Steering Subsystem implements Adam Marblestone's architectural separation between learning and steering mechanisms. This module generates reward signals (analogous to dopamine) to guide learning without directly modifying network weights.

**Core Principle**: Learning mechanisms remain distinct from steering mechanisms. Steering influences learning through reward signals, not direct parameter modification.

### 1.2 Performance Budget

| Operation | Budget | Implementation |
|-----------|--------|----------------|
| Gardener evaluation | <2ms | Long-term value assessment |
| Curator evaluation | <2ms | Quality scoring |
| Thought Assessor | <1ms | Real-time coherence |
| **Total** | **<5ms** | End-to-end evaluation |

### 1.3 Module Structure

```
crates/context-graph-steering/src/
├── lib.rs                  # Public API
├── subsystem.rs            # SteeringSubsystem orchestrator
├── reward.rs               # SteeringReward types
├── gardener.rs             # Long-term value assessment
├── curator.rs              # Quality assessment
├── assessor.rs             # ThoughtAssessor
├── integration/
│   ├── dopamine.rs         # Module 10 integration
│   ├── graph.rs            # Module 4 edge updates
│   └── dream.rs            # Module 9 shortcut quality
├── mcp/tools.rs            # get_steering_reward MCP tool
└── config.rs               # Configuration
```

---

## 2. Core Data Structures (REQ-STEER-001, REQ-STEER-002)

```rust
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Core steering subsystem (REQ-STEER-001)
/// `Constraint: Total_Evaluation_Latency < 5ms`
pub struct SteeringSubsystem {
    pub gardener: Gardener,
    pub curator: Curator,
    pub thought_assessor: ThoughtAssessor,
    pub dopamine_feedback: SteeringDopamineFeedback,
    pub config: SteeringConfig,
    pub session_state: Arc<RwLock<SteeringSessionState>>,
    graph: Arc<RwLock<KnowledgeGraph>>,
    neuromod: Arc<RwLock<NeuromodulationController>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SteeringConfig {
    pub gardener_weight: f32,     // Default: 0.35
    pub curator_weight: f32,      // Default: 0.35
    pub assessor_weight: f32,     // Default: 0.30
    pub positive_threshold: f32,  // Default: 0.3
    pub negative_threshold: f32,  // Default: -0.3
    pub dopamine_integration_enabled: bool,
    pub evaluation_timeout: Duration,
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
        }
    }
}

/// Steering reward (REQ-STEER-002)
#[derive(Clone, Debug, Serialize)]
pub struct SteeringReward {
    pub reward: f32,                    // [-1.0, 1.0]
    pub components: SteeringComponents,
    pub explanation: String,
    pub suggestions: Vec<SteeringSuggestion>,
    pub confidence: f32,
    pub latency: Duration,
}

#[derive(Clone, Debug, Serialize)]
pub struct SteeringComponents {
    pub gardener_score: f32,   // Long-term value [-1.0, 1.0]
    pub curator_score: f32,    // Quality [-1.0, 1.0]
    pub assessor_score: f32,   // Immediate relevance [-1.0, 1.0]
}

#[derive(Clone, Debug, Serialize)]
pub struct SteeringSuggestion {
    pub suggestion_type: SuggestionType,
    pub priority: f32,
    pub description: String,
    pub target_node: Option<Uuid>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub enum SuggestionType {
    Consolidate, Prune, StrengthenConnection,
    EnrichMetadata, RequestClarification, DreamReview,
}
```

---

## 3. Gardener Component (REQ-STEER-003)

```rust
/// The Gardener: long-term memory curation (Marblestone)
/// REQ-STEER-003: Gardener long-term value assessment
pub struct Gardener {
    pub prune_threshold: f32,         // Default: -0.5
    pub consolidation_threshold: f32, // Default: 0.7
    pub consolidation_rules: Vec<ConsolidationRule>,
    pub pruning_rules: Vec<PruningRule>,
    pub decay_config: DecayConfig,
}

#[derive(Clone, Debug)]
pub struct DecayConfig {
    pub importance_half_life: f32,  // Default: 168.0 (1 week)
    pub importance_floor: f32,      // Default: 0.1
}

impl Gardener {
    /// Evaluate long-term value `Constraint: <2ms`
    #[inline]
    pub fn evaluate_long_term_value(&self, node: &KnowledgeNode, ctx: &GardenerContext) -> f32 {
        let access_score = self.calculate_access_score(node, ctx);
        let recency_score = self.calculate_recency_score(node);
        let connection_score = self.calculate_connection_score(node, ctx);
        let importance_score = node.importance * 2.0 - 1.0;

        // Weighted combination with decay
        let age_hours = node.created_at.elapsed().as_secs_f32() / 3600.0;
        let decay = (-age_hours * 0.693 / self.decay_config.importance_half_life).exp()
            .max(self.decay_config.importance_floor);

        let raw = access_score * 0.30 + recency_score * 0.20
            + connection_score * 0.25 + importance_score * 0.25;

        (raw * decay).clamp(-1.0, 1.0)
    }

    fn calculate_access_score(&self, node: &KnowledgeNode, ctx: &GardenerContext) -> f32 {
        ctx.access_stats.get(&node.id)
            .map(|s| ((s.recent_accesses as f32 / 2.0).max(0.01).ln() * 0.5).clamp(-1.0, 1.0))
            .unwrap_or(0.0)
    }

    fn calculate_recency_score(&self, node: &KnowledgeNode) -> f32 {
        let age_hours = node.created_at.elapsed().as_secs_f32() / 3600.0;
        (-age_hours / 72.0 * 0.693).exp() * 2.0 - 1.0
    }

    fn calculate_connection_score(&self, node: &KnowledgeNode, ctx: &GardenerContext) -> f32 {
        let count = ctx.connection_count.get(&node.id).copied().unwrap_or(0);
        let ratio = count as f32 / ctx.avg_connection_count.max(1.0);
        (ratio.max(0.01).ln() * 0.3).clamp(-1.0, 1.0)
    }
}

#[derive(Clone, Debug)]
pub struct ConsolidationRule {
    pub id: String,
    pub condition: ConsolidationCondition,
    pub priority: u32,
}

#[derive(Clone, Debug)]
pub enum ConsolidationCondition {
    HighAccessFrequency { min_accesses: u32 },
    HighImportance { threshold: f32 },
    CrossDomainHub { min_domains: usize, min_connections: usize },
}

#[derive(Clone, Debug)]
pub struct PruningRule {
    pub id: String,
    pub condition: PruningCondition,
    pub severity: PruningSeverity,
}

#[derive(Clone, Debug)]
pub enum PruningCondition {
    Stale { min_hours: f32 },
    LowValue { max_importance: f32, max_connections: usize },
    Incoherent { coherence_score: f32 },
}

#[derive(Clone, Copy, Debug)]
pub enum PruningSeverity { Low, Medium, High, Critical }

#[derive(Clone, Default)]
pub struct GardenerContext {
    pub access_stats: HashMap<Uuid, AccessStats>,
    pub connection_count: HashMap<Uuid, usize>,
    pub avg_connection_count: f32,
    pub coherence_scores: HashMap<Uuid, f32>,
}

#[derive(Clone, Default)]
pub struct AccessStats {
    pub recent_accesses: u32,
    pub last_access: Option<Instant>,
}
```

---

## 4. Curator Component (REQ-STEER-004)

```rust
/// The Curator: quality assessment (Marblestone)
/// REQ-STEER-004: Curator quality assessment
pub struct Curator {
    pub quality_model: QualityModel,
    pub thresholds: QualityThresholds,
    pub config: CuratorConfig,
}

#[derive(Clone, Debug)]
pub struct CuratorConfig {
    pub completeness_weight: f32,  // Default: 0.25
    pub accuracy_weight: f32,      // Default: 0.30
    pub clarity_weight: f32,       // Default: 0.25
    pub relevance_weight: f32,     // Default: 0.20
}

impl Curator {
    /// Evaluate quality `Constraint: <2ms`
    #[inline]
    pub fn evaluate_quality(&self, node: &KnowledgeNode, ctx: &CuratorContext) -> f32 {
        let completeness = self.assess_completeness(node, ctx);
        let accuracy = self.assess_accuracy(node, ctx);
        let clarity = self.assess_clarity(node);
        let relevance = self.assess_relevance(node, ctx);

        let quality = self.config.completeness_weight * completeness
            + self.config.accuracy_weight * accuracy
            + self.config.clarity_weight * clarity
            + self.config.relevance_weight * relevance;

        ((quality - 0.5) * 2.0).clamp(-1.0, 1.0)
    }

    fn assess_completeness(&self, node: &KnowledgeNode, ctx: &CuratorContext) -> f32 {
        let has_embedding = node.embedding.is_some();
        let has_connections = ctx.connection_count.get(&node.id).copied().unwrap_or(0) >= 2;
        let sufficient_content = node.content.len() >= 20;

        let mut score = 0.3;
        if has_embedding { score += 0.3; }
        if has_connections { score += 0.2; }
        if sufficient_content { score += 0.2; }
        score
    }

    fn assess_accuracy(&self, node: &KnowledgeNode, ctx: &CuratorContext) -> f32 {
        let source_score = node.source_credibility.unwrap_or(0.5);
        let domain_score = ctx.domain_consistency.get(&node.id).copied().unwrap_or(0.5);
        source_score * 0.6 + domain_score * 0.4
    }

    fn assess_clarity(&self, node: &KnowledgeNode) -> f32 {
        match node.content.len() {
            0..=9 => 0.3,
            10..=499 => 0.7,
            500..=1999 => 1.0,
            _ => 0.7,
        }
    }

    fn assess_relevance(&self, node: &KnowledgeNode, ctx: &CuratorContext) -> f32 {
        ctx.semantic_similarity.get(&node.id).copied().unwrap_or(0.5)
    }
}

#[derive(Clone, Default)]
pub struct QualityModel {
    pub completeness_criteria: Vec<CompletenessCriterion>,
    pub accuracy_indicators: Vec<AccuracyIndicator>,
}

#[derive(Clone)]
pub struct CompletenessCriterion { pub name: String, pub weight: f32 }

#[derive(Clone)]
pub struct AccuracyIndicator { pub name: String, pub weight: f32 }

#[derive(Clone, Default)]
pub struct QualityThresholds {
    pub review_threshold: f32,      // Default: 0.4
    pub high_quality_threshold: f32, // Default: 0.8
}

#[derive(Clone, Default)]
pub struct CuratorContext {
    pub connection_count: HashMap<Uuid, usize>,
    pub domain_consistency: HashMap<Uuid, f32>,
    pub semantic_similarity: HashMap<Uuid, f32>,
}
```

---

## 5. Thought Assessor Component (REQ-STEER-005)

```rust
/// The Thought Assessor: real-time evaluation (Marblestone)
/// REQ-STEER-005: Thought Assessor real-time evaluation
pub struct ThoughtAssessor {
    pub coherence_threshold: f32,  // Default: 0.4
    pub novelty_weight: f32,       // Default: 0.3
    recent_thoughts: RingBuffer<ThoughtSignature>,
}

impl ThoughtAssessor {
    pub fn new(window_size: usize) -> Self {
        Self {
            coherence_threshold: 0.4,
            novelty_weight: 0.3,
            recent_thoughts: RingBuffer::new(window_size),
        }
    }

    /// Evaluate thought `Constraint: <1ms`
    #[inline]
    pub fn evaluate_immediate(&self, node: &KnowledgeNode, ctx: &AssessorContext) -> f32 {
        // Rapid rejection check
        if node.content.trim().is_empty() { return -0.5; }
        if node.content.split_whitespace().count() < 2 { return -0.3; }

        let coherence = self.assess_coherence(node, ctx);
        let novelty = self.assess_novelty(node);
        let context_fit = self.assess_context_fit(node, ctx);

        let score = coherence * 0.4 + novelty * 0.3 + context_fit * 0.3;
        ((score - 0.5) * 2.0).clamp(-1.0, 1.0)
    }

    fn assess_coherence(&self, node: &KnowledgeNode, ctx: &AssessorContext) -> f32 {
        let semantic = ctx.domain_similarity.get(&node.id).copied().unwrap_or(0.5);
        let structural = if node.content.len() > 10 && node.content.len() < 10000 { 0.7 } else { 0.4 };
        semantic * 0.6 + structural * 0.4
    }

    fn assess_novelty(&self, node: &KnowledgeNode) -> f32 {
        if self.recent_thoughts.len() == 0 { return 0.7; }

        let sig = self.compute_signature(node);
        let max_sim = self.recent_thoughts.iter()
            .map(|r| self.compare_signatures(&sig, r))
            .fold(0.0f32, |a, b| a.max(b));
        1.0 - max_sim
    }

    fn assess_context_fit(&self, node: &KnowledgeNode, ctx: &AssessorContext) -> f32 {
        ctx.query_similarity.get(&node.id).copied().unwrap_or(0.5)
    }

    fn compute_signature(&self, node: &KnowledgeNode) -> ThoughtSignature {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut h = DefaultHasher::new();
        node.content.hash(&mut h);
        ThoughtSignature { hash: h.finish(), domain: node.domain.clone() }
    }

    fn compare_signatures(&self, a: &ThoughtSignature, b: &ThoughtSignature) -> f32 {
        let hash_match = if a.hash == b.hash { 0.8 } else { 0.0 };
        let domain_match = if a.domain == b.domain { 0.2 } else { 0.0 };
        hash_match + domain_match
    }

    pub fn record_thought(&mut self, node: &KnowledgeNode) {
        self.recent_thoughts.push(self.compute_signature(node));
    }
}

#[derive(Clone)]
pub struct ThoughtSignature { hash: u64, domain: Option<String> }

/// Ring buffer for novelty tracking
pub struct RingBuffer<T> { buffer: Vec<T>, head: usize, capacity: usize }

impl<T: Clone> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self { Self { buffer: Vec::new(), head: 0, capacity } }
    pub fn push(&mut self, item: T) {
        if self.buffer.len() < self.capacity { self.buffer.push(item); }
        else { self.buffer[self.head] = item; }
        self.head = (self.head + 1) % self.capacity;
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> { self.buffer.iter() }
    pub fn len(&self) -> usize { self.buffer.len() }
}

#[derive(Clone, Default)]
pub struct AssessorContext {
    pub domain_similarity: HashMap<Uuid, f32>,
    pub query_similarity: HashMap<Uuid, f32>,
}
```

---

## 6. Unified Evaluation (REQ-STEER-006)

```rust
impl SteeringSubsystem {
    /// Unified evaluation `Constraint: <5ms total`
    /// REQ-STEER-006: Unified steering evaluation
    pub async fn evaluate(&mut self, node: &KnowledgeNode, ctx: &SteeringContext) -> SteeringReward {
        let start = Instant::now();

        // Component evaluations
        let gardener_score = self.gardener.evaluate_long_term_value(node, &ctx.gardener);
        let curator_score = self.curator.evaluate_quality(node, &ctx.curator);
        let assessor_score = self.thought_assessor.evaluate_immediate(node, &ctx.assessor);

        // Weighted combination
        let reward = (self.config.gardener_weight * gardener_score
            + self.config.curator_weight * curator_score
            + self.config.assessor_weight * assessor_score).clamp(-1.0, 1.0);

        // Record and send dopamine feedback (REQ-STEER-007)
        self.thought_assessor.record_thought(node);
        if self.config.dopamine_integration_enabled {
            self.send_dopamine_feedback(reward, &SteeringComponents {
                gardener_score, curator_score, assessor_score,
            }).await;
        }

        // Update Module 4 edge steering_reward (EMA)
        self.update_edge_steering_reward(node.id, reward).await;

        SteeringReward {
            reward,
            components: SteeringComponents { gardener_score, curator_score, assessor_score },
            explanation: self.generate_explanation(reward, gardener_score, curator_score, assessor_score),
            suggestions: self.generate_suggestions(node, gardener_score, curator_score, assessor_score),
            confidence: self.calculate_confidence(gardener_score, curator_score, assessor_score),
            latency: start.elapsed(),
        }
    }

    fn generate_explanation(&self, r: f32, g: f32, c: f32, a: f32) -> String {
        let sentiment = if r > 0.3 { "positive" } else if r < -0.3 { "negative" } else { "neutral" };
        format!("{} signal ({:.2}): G={:.2}, C={:.2}, A={:.2}", sentiment, r, g, c, a)
    }

    fn calculate_confidence(&self, g: f32, c: f32, a: f32) -> f32 {
        let signs = [g.signum(), c.signum(), a.signum()];
        let agree = signs.iter().filter(|&&s| s > 0.0).count() >= 2
            || signs.iter().filter(|&&s| s < 0.0).count() >= 2;
        if agree { 0.8 } else { 0.5 }
    }

    fn generate_suggestions(&self, node: &KnowledgeNode, g: f32, c: f32, a: f32) -> Vec<SteeringSuggestion> {
        let mut s = Vec::new();
        if g < -0.5 {
            s.push(SteeringSuggestion {
                suggestion_type: SuggestionType::Prune,
                priority: (-g).min(1.0),
                description: "Low long-term value".to_string(),
                target_node: Some(node.id),
            });
        }
        if g > 0.7 {
            s.push(SteeringSuggestion {
                suggestion_type: SuggestionType::Consolidate,
                priority: g,
                description: "High long-term value".to_string(),
                target_node: Some(node.id),
            });
        }
        if a < -0.3 {
            s.push(SteeringSuggestion {
                suggestion_type: SuggestionType::DreamReview,
                priority: 0.7,
                description: "Low coherence - dream review".to_string(),
                target_node: Some(node.id),
            });
        }
        s.truncate(3);
        s
    }

    async fn update_edge_steering_reward(&self, node_id: Uuid, reward: f32) {
        let mut graph = self.graph.write().await;
        if let Some(edges) = graph.get_edges_for_node_mut(node_id) {
            for edge in edges {
                // EMA update (alpha=0.1)
                edge.steering_reward = (edge.steering_reward * 0.9 + reward * 0.1).clamp(-1.0, 1.0);
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct SteeringContext {
    pub gardener: GardenerContext,
    pub curator: CuratorContext,
    pub assessor: AssessorContext,
}
```

---

## 7. Dopamine Feedback Integration (REQ-STEER-007)

```rust
/// Steering-to-Dopamine feedback (Marblestone)
/// REQ-STEER-007: Dopamine feedback integration with Module 10
pub struct SteeringDopamineFeedback {
    pub positive_scale: f32,  // Default: 0.3
    pub negative_scale: f32,  // Default: 0.2
    pub min_magnitude: f32,   // Default: 0.1
    history: RingBuffer<f32>,
}

impl SteeringSubsystem {
    async fn send_dopamine_feedback(&self, reward: f32, components: &SteeringComponents) {
        if reward.abs() < 0.1 { return; }

        let scale = if reward > 0.0 { 0.3 } else { 0.2 };
        let delta = (reward * scale).clamp(-0.2, 0.2);

        // Send to Module 10 apply_steering_feedback
        let mut ctrl = self.neuromod.write().await;
        ctrl.apply_steering_feedback(&SteeringDopamineFeedbackMsg {
            reward,
            signal_strength: delta,
            source: SteeringSource::ThoughtAssessor,
            timestamp: Utc::now(),
        });
    }
}

/// Message to Module 10 neuromodulation
#[derive(Clone)]
pub struct SteeringDopamineFeedbackMsg {
    pub reward: f32,
    pub signal_strength: f32,
    pub source: SteeringSource,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone)]
pub enum SteeringSource { Gardener, Curator, ThoughtAssessor }
```

---

## 8. Dream Layer Integration (REQ-STEER-008)

```rust
/// Module 9 Dream Layer shortcut quality assessment
/// REQ-STEER-008: Dream layer shortcut quality
pub struct DreamQualityAssessor;

impl DreamQualityAssessor {
    /// Assess shortcut quality for dream consolidation
    pub fn assess_shortcut_quality(
        &self,
        shortcut: &DreamShortcut,
        steering: &SteeringSubsystem,
    ) -> f32 {
        // Use steering components to evaluate shortcut
        let value_score = if shortcut.value_gain > 0.5 { 0.8 } else { shortcut.value_gain };
        let usage_score = (shortcut.expected_usage as f32 / 10.0).min(1.0);

        // Weight: 60% value gain, 40% expected usage
        let quality = value_score * 0.6 + usage_score * 0.4;
        quality.clamp(0.0, 1.0)
    }
}

/// Dream shortcut from Module 9
pub struct DreamShortcut {
    pub source_id: Uuid,
    pub target_id: Uuid,
    pub value_gain: f32,
    pub expected_usage: u32,
}

impl SteeringSubsystem {
    /// Integrate with dream layer for shortcut evaluation
    pub fn evaluate_dream_shortcut(&self, shortcut: &DreamShortcut) -> f32 {
        let assessor = DreamQualityAssessor;
        assessor.assess_shortcut_quality(shortcut, self)
    }
}
```

---

## 9. MCP Tool: get_steering_reward

```rust
/// MCP tool for retrieving steering reward
#[tool(name = "get_steering_reward", description = "Get steering reward for a memory")]
pub async fn get_steering_reward(
    node_id: Uuid,
    steering: Arc<RwLock<SteeringSubsystem>>,
    graph: Arc<RwLock<KnowledgeGraph>>,
) -> Result<SteeringReward, McpError> {
    let node = {
        let g = graph.read().await;
        g.get_node(node_id).ok_or(McpError::NotFound)?.clone()
    };

    let ctx = SteeringContext::default();
    let mut s = steering.write().await;
    Ok(s.evaluate(&node, &ctx).await)
}
```

---

## 10. Configuration (TOML)

```toml
[steering]
gardener_weight = 0.35
curator_weight = 0.35
assessor_weight = 0.30
positive_threshold = 0.3
negative_threshold = -0.3
dopamine_integration_enabled = true
evaluation_timeout_ms = 5

[steering.gardener]
prune_threshold = -0.5
consolidation_threshold = 0.7
importance_half_life_hours = 168.0
importance_floor = 0.1

[steering.curator]
completeness_weight = 0.25
accuracy_weight = 0.30
clarity_weight = 0.25
relevance_weight = 0.20

[steering.assessor]
coherence_threshold = 0.4
novelty_weight = 0.3
novelty_window = 100
```

---

## 11. Dependencies

```toml
[dependencies]
tokio = { version = "1.0", features = ["sync", "time"] }
uuid = { version = "1.0", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1.0", features = ["derive"] }
tracing = "0.1"

# Internal crates
context-graph-core = { path = "../context-graph-core" }
context-graph-neuromod = { path = "../context-graph-neuromod" }
context-graph-dream = { path = "../context-graph-dream" }
```

---

## 12. Verification Matrix

| Requirement | Implementation | Test |
|-------------|----------------|------|
| REQ-STEER-001 | `SteeringSubsystem` struct | `test_subsystem_creation` |
| REQ-STEER-002 | `SteeringReward` struct | `test_reward_generation` |
| REQ-STEER-003 | `Gardener` component | `test_gardener_evaluation` |
| REQ-STEER-004 | `Curator` component | `test_curator_evaluation` |
| REQ-STEER-005 | `ThoughtAssessor` component | `test_assessor_evaluation` |
| REQ-STEER-006 | `evaluate()` unified | `test_unified_evaluation` |
| REQ-STEER-007 | `send_dopamine_feedback()` | `test_dopamine_integration` |
| REQ-STEER-008 | `DreamQualityAssessor` | `test_dream_shortcut_quality` |

---

## 13. Acceptance Criteria

1. **Performance**: Total evaluation latency < 5ms (Gardener <2ms, Curator <2ms, Assessor <1ms)
2. **Weights**: Gardener 35%, Curator 35%, Assessor 30% = 100%
3. **Range**: All scores in [-1.0, 1.0]
4. **Integration**: Module 10 dopamine feedback via `apply_steering_feedback()`
5. **Integration**: Module 4 edge `steering_reward` field updated via EMA
6. **Integration**: Module 9 dream shortcut quality assessment
7. **MCP**: `get_steering_reward` tool available
8. **Marblestone**: Learning/steering separation maintained
