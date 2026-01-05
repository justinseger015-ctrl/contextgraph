# Module 12: Active Inference - Technical Specification

```yaml
metadata:
  id: TECH-ACTINF-012
  version: 1.0.0
  module: Active Inference
  phase: 11
  status: draft
  created: 2025-12-31
  dependencies:
    - TECH-GHOST-001 (Module 1: Ghost System)
    - TECH-CORE-002 (Module 2: Core Infrastructure)
    - TECH-EMBED-003 (Module 3: Embedding Pipeline)
    - TECH-KG-004 (Module 4: Knowledge Graph)
    - TECH-UTL-005 (Module 5: UTL Integration)
    - TECH-BIONV-006 (Module 6: Bio-Nervous System)
    - TECH-DREAM-009 (Module 9: Dream Layer)
    - TECH-NEURO-010 (Module 10: Neuromodulation)
    - TECH-IMMUNE-011 (Module 11: Immune System)
  functional_spec_ref: SPEC-ACTINF-012
  author: Architecture Agent
```

---

## 1. Architecture Overview

### 1.1 Free Energy Principle Foundation

The Active Inference module implements the Free Energy Principle (FEP) for proactive knowledge acquisition. The system minimizes variational free energy to maintain accurate world models and generate epistemic actions.

**Variational Free Energy**:
```
F = D_KL[q(s) || p(s|o)] - ln p(o)
```

Where:
- `F`: Variational Free Energy (upper bound on surprise)
- `q(s)`: Approximate posterior (beliefs about hidden states)
- `p(s|o)`: True posterior given observations
- `p(o)`: Model evidence (marginal likelihood)
- `D_KL`: Kullback-Leibler divergence

**Expected Free Energy (EFE)**:
```
G(pi) = E_q[ln q(s') - ln p(o', s')]
      = -E_q[H[p(o|s)]] + E_q[D_KL[q(s') || p(s')]]
      = Epistemic Value + Pragmatic Value
```

### 1.2 Performance Targets

| Operation | Target | Implementation |
|-----------|--------|----------------|
| EFE Computation (single) | <10ms | SIMD-optimized matrix ops |
| Belief Update | <5ms | Variational inference |
| Policy Evaluation (all) | <10ms | Tree search with pruning |
| Action Generation | <5ms | Template instantiation |
| Total Action Selection | <20ms | End-to-end pipeline |
| MCP Handler | <25ms | Including serialization |

### 1.3 Module Structure

```
crates/context-graph-active-inference/src/
├── lib.rs                       # Public API
├── engine.rs                    # ActiveInferenceEngine
├── beliefs/
│   ├── mod.rs                   # BeliefDistribution, SessionBeliefs
│   ├── distribution.rs          # Gaussian belief representation
│   ├── entropy.rs               # Entropy computation
│   └── kl_divergence.rs         # KL divergence operations
├── generative/
│   ├── mod.rs                   # GenerativeModel
│   ├── layer.rs                 # GenerativeLayer hierarchy
│   ├── likelihood.rs            # LikelihoodModel p(o|s)
│   └── prediction.rs            # Prediction generation
├── inference/
│   ├── mod.rs                   # BeliefUpdater
│   ├── variational.rs           # Variational message passing
│   ├── precision.rs             # Precision-weighted updates
│   └── omni.rs                  # OmniInferenceEngine (Marblestone)
├── policy/
│   ├── mod.rs                   # PolicyEvaluator
│   ├── efe.rs                   # Expected Free Energy computation
│   ├── tree_search.rs           # Temporal planning (5 steps)
│   └── softmax.rs               # Policy selection
├── actions/
│   ├── mod.rs                   # EpistemicActionGenerator
│   ├── templates.rs             # Action templates
│   ├── gaps.rs                  # Knowledge gap detection
│   └── rate_limiter.rs          # Spam prevention
├── mcp/
│   ├── mod.rs                   # MCP integration
│   ├── handler.rs               # epistemic_action handler
│   └── schema.rs                # Request/Response types
├── metrics.rs                   # Performance metrics
└── config.rs                    # Configuration
```

---

## 2. Core Data Structures

### 2.1 ActiveInferenceEngine

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;
use nalgebra::{DMatrix, DVector};

/// Core engine for Active Inference computations implementing the Free Energy Principle.
///
/// Coordinates EFE computation, belief updating, policy evaluation, and epistemic
/// action generation based on variational inference and precision-weighted prediction errors.
///
/// REQ-ACTINF-001: ActiveInferenceEngine struct definition
/// `Constraint: Total_Action_Selection_Latency < 20ms`
pub struct ActiveInferenceEngine {
    /// Hierarchical generative model for predictions
    pub generative_model: GenerativeModel,

    /// Belief state updater via variational inference
    pub belief_updater: BeliefUpdater,

    /// Policy tree evaluator with temporal planning
    pub policy_evaluator: PolicyEvaluator,

    /// Epistemic action generator
    pub action_generator: EpistemicActionGenerator,

    /// Omnidirectional inference engine (Marblestone)
    pub omni_engine: OmniInferenceEngine,

    /// Current session beliefs
    pub session_beliefs: Arc<RwLock<SessionBeliefs>>,

    /// Engine configuration
    pub config: ActiveInferenceConfig,

    /// Performance metrics
    pub metrics: ActiveInferenceMetrics,
}

/// Configuration for Active Inference engine
#[derive(Clone, Debug)]
pub struct ActiveInferenceConfig {
    /// Uncertainty threshold for triggering epistemic actions
    pub uncertainty_threshold: f32,  // Default: 0.6

    /// Maximum epistemic actions per response
    pub max_actions_per_response: usize,  // Default: 3

    /// Exploration budget (number of hypotheses to consider)
    pub exploration_budget: u32,  // Default: 10

    /// Temporal planning depth (steps ahead)
    pub planning_depth: usize,  // Default: 5

    /// Precision weighting factor for prediction errors
    pub precision_weight: f32,  // Default: 1.0

    /// Balance between epistemic and pragmatic value
    /// 0 = pure pragmatic, 1 = pure epistemic
    pub epistemic_pragmatic_balance: f32,  // Default: 0.5

    /// EFE computation timeout
    pub efe_timeout: Duration,  // Default: 10ms

    /// Belief update timeout
    pub belief_timeout: Duration,  // Default: 5ms

    /// Total action selection timeout
    pub action_timeout: Duration,  // Default: 20ms
}

impl Default for ActiveInferenceConfig {
    fn default() -> Self {
        Self {
            uncertainty_threshold: 0.6,
            max_actions_per_response: 3,
            exploration_budget: 10,
            planning_depth: 5,
            precision_weight: 1.0,
            epistemic_pragmatic_balance: 0.5,
            efe_timeout: Duration::from_millis(10),
            belief_timeout: Duration::from_millis(5),
            action_timeout: Duration::from_millis(20),
        }
    }
}

impl ActiveInferenceEngine {
    /// Create new engine with configuration
    pub fn new(config: ActiveInferenceConfig) -> Self {
        let generative_model = GenerativeModel::new(GenerativeModelConfig::default());
        let belief_updater = BeliefUpdater::new(BeliefUpdaterConfig::default());
        let policy_evaluator = PolicyEvaluator::new(PolicyEvaluatorConfig::default());
        let action_generator = EpistemicActionGenerator::new(EpistemicActionConfig::default());
        let omni_engine = OmniInferenceEngine::new(OmniInferenceConfig::default());

        Self {
            generative_model,
            belief_updater,
            policy_evaluator,
            action_generator,
            omni_engine,
            session_beliefs: Arc::new(RwLock::new(SessionBeliefs::new())),
            config,
            metrics: ActiveInferenceMetrics::default(),
        }
    }

    /// Process observation and update beliefs
    ///
    /// `Constraint: Observation_Processing < 10ms`
    pub async fn process_observation(
        &mut self,
        observation: Observation,
    ) -> Result<BeliefUpdateResult, ActiveInferenceError> {
        let start = Instant::now();

        // Update beliefs via variational inference
        let mut beliefs = self.session_beliefs.write().await;
        let result = self.belief_updater.update_beliefs(
            &mut beliefs.state_beliefs,
            &observation,
            &self.generative_model,
        );

        // Update metrics
        self.metrics.belief_updates += 1;
        self.metrics.avg_belief_latency_us =
            self.metrics.avg_belief_latency_us * 0.9
            + start.elapsed().as_micros() as f64 * 0.1;

        Ok(result)
    }

    /// Select action based on Expected Free Energy
    ///
    /// `Constraint: Action_Selection < 20ms`
    pub async fn select_action(&mut self) -> Result<Vec<EpistemicAction>, ActiveInferenceError> {
        let start = Instant::now();

        let beliefs = self.session_beliefs.read().await;

        // Evaluate policies
        let policy_evals = self.policy_evaluator.evaluate_policies(
            &beliefs,
            &self.generative_model,
            &beliefs.goal_preferences,
        );

        // Generate actions based on top policies
        let context = ActionContext {
            current_domain: None,
            recent_queries: Vec::new(),
            urgency: 0.5,
            session_id: beliefs.session_id,
        };

        let actions = self.action_generator.generate_actions(&beliefs, &context);

        // Update metrics
        self.metrics.actions_generated += actions.len() as u64;
        let latency_us = start.elapsed().as_micros() as f64;
        self.metrics.avg_efe_latency_us =
            self.metrics.avg_efe_latency_us * 0.9 + latency_us * 0.1;

        Ok(actions)
    }
}
```

### 2.2 Session Beliefs

```rust
/// REQ-ACTINF-002: Session-level belief state for variational inference
#[derive(Clone)]
pub struct SessionBeliefs {
    /// Session identifier
    pub session_id: Uuid,

    /// Current belief distribution over hidden states (q(s))
    pub state_beliefs: BeliefDistribution,

    /// Prior preferences over observations (p(o|C))
    pub goal_preferences: GoalPreferences,

    /// Accumulated prediction errors
    pub prediction_errors: PredictionErrorBuffer,

    /// Uncertainty estimates per knowledge domain
    pub domain_uncertainty: HashMap<String, f32>,

    /// Recent observation history for belief updating
    pub observation_history: Vec<Observation>,

    /// Current Johari quadrant for action prioritization
    pub johari_quadrant: JohariQuadrant,

    /// Last belief update timestamp
    pub last_update: Instant,

    /// Belief entropy (uncertainty measure)
    pub belief_entropy: f32,
}

impl SessionBeliefs {
    /// Create new session beliefs with uniform prior
    pub fn new() -> Self {
        Self {
            session_id: Uuid::new_v4(),
            state_beliefs: BeliefDistribution::uniform_prior(1536),
            goal_preferences: GoalPreferences::default(),
            prediction_errors: PredictionErrorBuffer::new(100),
            domain_uncertainty: HashMap::new(),
            observation_history: Vec::new(),
            johari_quadrant: JohariQuadrant::Unknown,
            last_update: Instant::now(),
            belief_entropy: 1.0,
        }
    }
}

/// Probability distribution over hidden states
#[derive(Clone)]
pub struct BeliefDistribution {
    /// Mean of the belief distribution (1536D aligned with embeddings)
    pub mean: DVector<f32>,

    /// Precision matrix (inverse covariance) - diagonal approximation for efficiency
    pub precision: DVector<f32>,

    /// Confidence in current beliefs [0, 1]
    pub confidence: f32,

    /// Number of observations incorporated
    pub observation_count: u64,
}

impl BeliefDistribution {
    /// Create uniform prior beliefs
    pub fn uniform_prior(dimension: usize) -> Self {
        Self {
            mean: DVector::zeros(dimension),
            precision: DVector::from_element(dimension, 1.0),
            confidence: 0.0,
            observation_count: 0,
        }
    }

    /// Compute entropy of belief distribution
    ///
    /// For diagonal Gaussian: H = 0.5 * sum(log(2 * pi * e / precision))
    ///
    /// `Constraint: Entropy_Computation < 1ms`
    #[inline]
    pub fn entropy(&self) -> f32 {
        let n = self.precision.len() as f32;
        let log_precision_sum: f32 = self.precision.iter()
            .map(|&p| if p > 1e-10 { p.ln() } else { -23.0 })
            .sum();

        0.5 * (n * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln() - log_precision_sum)
    }

    /// Compute KL divergence from another distribution
    ///
    /// KL(self || other) for diagonal Gaussians
    ///
    /// `Constraint: KL_Computation < 1ms`
    #[inline]
    pub fn kl_divergence(&self, other: &BeliefDistribution) -> f32 {
        let diff = &self.mean - &other.mean;

        let precision_ratio: f32 = self.precision.iter()
            .zip(other.precision.iter())
            .map(|(&p1, &p2)| if p2 > 1e-10 { p1 / p2 } else { 1e10 })
            .sum();

        let log_det_ratio: f32 = other.precision.iter()
            .zip(self.precision.iter())
            .map(|(&p2, &p1)| {
                if p1 > 1e-10 && p2 > 1e-10 {
                    (p2 / p1).ln()
                } else {
                    0.0
                }
            })
            .sum();

        let quadratic: f32 = diff.iter()
            .zip(other.precision.iter())
            .map(|(&d, &p)| d * d * p)
            .sum();

        let n = self.mean.len() as f32;
        0.5 * (precision_ratio - n + log_det_ratio + quadratic)
    }
}

/// Johari window quadrant (from UTL integration)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JohariQuadrant {
    /// Known to self and others - direct recall
    Open,
    /// Unknown to self, known to others - discovery zone
    Blind,
    /// Known to self, hidden from others - private
    Hidden,
    /// Unknown to both - exploration frontier
    Unknown,
}
```

---

## 3. Omnidirectional Inference Engine (Marblestone)

### 3.1 REQ-ACTINF-008: OmniInferenceEngine

```rust
/// Direction of inference in the knowledge graph
/// REQ-ACTINF-008: Omnidirectional Inference Support
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceDirection {
    /// A -> B: Given cause, infer effect
    Forward,
    /// B -> A: Given effect, infer cause (abduction)
    Backward,
    /// A <-> B: Bidirectional inference
    Bidirectional,
    /// Bridge inference across domains
    Bridge,
    /// Abductive reasoning: best explanation
    Abduction,
}

/// REQ-ACTINF-010: Clamped variable for constrained inference
#[derive(Clone, Debug)]
pub struct ClampedValue {
    /// Node identifier
    pub node_id: Uuid,
    /// Fixed or biased value
    pub value: f32,
    /// Whether this is an observation
    pub is_observation: bool,
    /// Clamp type
    pub clamp_type: ClampType,
}

/// Type of value clamping
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClampType {
    /// Hard clamp: value cannot change during inference
    Hard,
    /// Soft clamp: value has high prior but can change
    Soft { prior_strength: f32 },
}

/// Configuration for omnidirectional inference
#[derive(Clone, Debug)]
pub struct OmniInferenceConfig {
    /// Maximum inference depth
    pub max_depth: u32,  // Default: 5
    /// Timeout per inference step
    pub step_timeout: Duration,  // Default: 2ms
    /// Minimum belief threshold for path continuation
    pub belief_threshold: f32,  // Default: 0.1
    /// Maximum candidates per step
    pub max_candidates: usize,  // Default: 10
}

impl Default for OmniInferenceConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            step_timeout: Duration::from_millis(2),
            belief_threshold: 0.1,
            max_candidates: 10,
        }
    }
}

/// Omnidirectional inference engine supporting all inference directions (Marblestone)
///
/// Extends standard forward inference with backward (abduction),
/// bidirectional, and bridge inference capabilities.
///
/// REQ-ACTINF-008, REQ-ACTINF-009, REQ-ACTINF-010
pub struct OmniInferenceEngine {
    /// Active inference engine reference for EFE computation
    active_inference: Arc<RwLock<ActiveInferenceEngine>>,
    /// Clamped variables (fixed during inference)
    clamped_variables: HashMap<Uuid, ClampedValue>,
    /// Current inference direction
    direction: InferenceDirection,
    /// Configuration
    config: OmniInferenceConfig,
    /// Causal edge cache
    causal_edge_cache: LruCache<Uuid, Vec<CausalEdge>>,
    /// Metrics
    metrics: OmniInferenceMetrics,
}

/// Result of omnidirectional inference
#[derive(Clone, Debug)]
pub struct InferenceResult {
    /// Inferred node identifiers
    pub inferred_nodes: Vec<Uuid>,
    /// Belief strength per node
    pub beliefs: HashMap<Uuid, f32>,
    /// Inference direction used
    pub direction: InferenceDirection,
    /// Total inference time
    pub latency: Duration,
    /// Number of steps taken
    pub steps: usize,
    /// Explanation chain for interpretability
    pub explanation_chain: Vec<InferenceStep>,
}

/// Single step in inference chain
#[derive(Clone, Debug)]
pub struct InferenceStep {
    pub from_node: Uuid,
    pub to_node: Uuid,
    pub edge_type: String,
    pub belief_delta: f32,
    pub reasoning: String,
}

impl OmniInferenceEngine {
    /// Create new omnidirectional inference engine
    pub fn new(config: OmniInferenceConfig) -> Self {
        Self {
            active_inference: Arc::new(RwLock::new(
                ActiveInferenceEngine::new(ActiveInferenceConfig::default())
            )),
            clamped_variables: HashMap::new(),
            direction: InferenceDirection::Forward,
            config,
            causal_edge_cache: LruCache::new(NonZeroUsize::new(1000).unwrap()),
            metrics: OmniInferenceMetrics::default(),
        }
    }

    /// Perform omnidirectional inference (Marblestone)
    ///
    /// # Arguments
    /// * `query_nodes` - Starting nodes for inference
    /// * `direction` - Inference direction
    /// * `clamped` - Variables to hold fixed
    ///
    /// # Returns
    /// Inference results with updated beliefs
    ///
    /// `Constraint: Total_Inference < 15ms`
    pub async fn omni_infer(
        &mut self,
        query_nodes: Vec<Uuid>,
        direction: InferenceDirection,
        clamped: &[ClampedValue],
    ) -> Result<InferenceResult, InferenceError> {
        let start = Instant::now();
        self.direction = direction;

        // Set up clamped variables
        self.clamped_variables.clear();
        for cv in clamped {
            self.clamped_variables.insert(cv.node_id, cv.clone());
        }

        let result = match direction {
            InferenceDirection::Forward => {
                self.forward_inference(&query_nodes).await
            }
            InferenceDirection::Backward => {
                self.backward_inference(&query_nodes).await
            }
            InferenceDirection::Bidirectional => {
                self.bidirectional_inference(&query_nodes).await
            }
            InferenceDirection::Bridge => {
                self.bridge_inference(&query_nodes).await
            }
            InferenceDirection::Abduction => {
                self.abductive_inference(&query_nodes).await
            }
        };

        // Update metrics
        self.metrics.total_inferences += 1;
        self.metrics.avg_latency_us =
            self.metrics.avg_latency_us * 0.9 + start.elapsed().as_micros() as f64 * 0.1;

        result
    }

    /// REQ-ACTINF-009: Backward inference - given effects, infer likely causes
    ///
    /// `Constraint: Backward_Inference < 10ms`
    async fn backward_inference(
        &self,
        effects: &[Uuid],
    ) -> Result<InferenceResult, InferenceError> {
        let start = Instant::now();
        let mut candidates: Vec<(Uuid, f32)> = Vec::new();
        let mut explanation_chain = Vec::new();

        for effect_id in effects {
            // Skip if clamped as non-effect
            if let Some(cv) = self.clamped_variables.get(effect_id) {
                if !cv.is_observation {
                    continue;
                }
            }

            // Traverse causal edges in reverse
            let incoming = self.get_incoming_causal_edges(*effect_id).await?;

            for edge in incoming {
                // Skip if depth exceeded
                if edge.depth > self.config.max_depth {
                    continue;
                }

                // Compute cause belief using Bayes rule
                let cause_belief = self.compute_cause_belief(&edge, *effect_id)?;

                // Apply clamping constraints
                let final_belief = if let Some(cv) = self.clamped_variables.get(&edge.source) {
                    match cv.clamp_type {
                        ClampType::Hard => cv.value,
                        ClampType::Soft { prior_strength } => {
                            cv.value * prior_strength + cause_belief * (1.0 - prior_strength)
                        }
                    }
                } else {
                    cause_belief
                };

                if final_belief >= self.config.belief_threshold {
                    candidates.push((edge.source, final_belief));
                    explanation_chain.push(InferenceStep {
                        from_node: edge.source,
                        to_node: *effect_id,
                        edge_type: edge.edge_type.clone(),
                        belief_delta: final_belief,
                        reasoning: format!(
                            "Backward: {} likely caused {} (belief: {:.3})",
                            edge.source, effect_id, final_belief
                        ),
                    });
                }
            }
        }

        // Rank by belief strength
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.max_candidates);

        Ok(InferenceResult {
            inferred_nodes: candidates.iter().map(|(id, _)| *id).collect(),
            beliefs: candidates.into_iter().collect(),
            direction: InferenceDirection::Backward,
            latency: start.elapsed(),
            steps: explanation_chain.len(),
            explanation_chain,
        })
    }

    /// Forward inference: given causes, predict effects
    async fn forward_inference(
        &self,
        causes: &[Uuid],
    ) -> Result<InferenceResult, InferenceError> {
        let start = Instant::now();
        let mut candidates: Vec<(Uuid, f32)> = Vec::new();
        let mut explanation_chain = Vec::new();

        for cause_id in causes {
            let outgoing = self.get_outgoing_causal_edges(*cause_id).await?;

            for edge in outgoing {
                if edge.depth > self.config.max_depth {
                    continue;
                }

                let effect_belief = self.compute_effect_belief(&edge, *cause_id)?;

                let final_belief = if let Some(cv) = self.clamped_variables.get(&edge.target) {
                    match cv.clamp_type {
                        ClampType::Hard => cv.value,
                        ClampType::Soft { prior_strength } => {
                            cv.value * prior_strength + effect_belief * (1.0 - prior_strength)
                        }
                    }
                } else {
                    effect_belief
                };

                if final_belief >= self.config.belief_threshold {
                    candidates.push((edge.target, final_belief));
                    explanation_chain.push(InferenceStep {
                        from_node: *cause_id,
                        to_node: edge.target,
                        edge_type: edge.edge_type.clone(),
                        belief_delta: final_belief,
                        reasoning: format!(
                            "Forward: {} causes {} (belief: {:.3})",
                            cause_id, edge.target, final_belief
                        ),
                    });
                }
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.max_candidates);

        Ok(InferenceResult {
            inferred_nodes: candidates.iter().map(|(id, _)| *id).collect(),
            beliefs: candidates.into_iter().collect(),
            direction: InferenceDirection::Forward,
            latency: start.elapsed(),
            steps: explanation_chain.len(),
            explanation_chain,
        })
    }

    /// Bidirectional inference: combine forward and backward
    async fn bidirectional_inference(
        &self,
        nodes: &[Uuid],
    ) -> Result<InferenceResult, InferenceError> {
        let start = Instant::now();

        // Run forward and backward in parallel
        let (forward_result, backward_result) = tokio::join!(
            self.forward_inference(nodes),
            self.backward_inference(nodes)
        );

        let forward = forward_result?;
        let backward = backward_result?;

        // Merge results
        let mut merged_beliefs: HashMap<Uuid, f32> = forward.beliefs;
        for (id, belief) in backward.beliefs {
            merged_beliefs.entry(id)
                .and_modify(|b| *b = (*b + belief) / 2.0)
                .or_insert(belief);
        }

        let mut merged_chain = forward.explanation_chain;
        merged_chain.extend(backward.explanation_chain);

        let mut candidates: Vec<_> = merged_beliefs.into_iter().collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.max_candidates);

        Ok(InferenceResult {
            inferred_nodes: candidates.iter().map(|(id, _)| *id).collect(),
            beliefs: candidates.into_iter().collect(),
            direction: InferenceDirection::Bidirectional,
            latency: start.elapsed(),
            steps: merged_chain.len(),
            explanation_chain: merged_chain,
        })
    }

    /// Bridge inference: cross-domain causal reasoning
    async fn bridge_inference(
        &self,
        nodes: &[Uuid],
    ) -> Result<InferenceResult, InferenceError> {
        // Find cross-domain edges and apply inference
        let start = Instant::now();
        let mut candidates: Vec<(Uuid, f32)> = Vec::new();
        let mut explanation_chain = Vec::new();

        for node_id in nodes {
            let cross_domain = self.get_cross_domain_edges(*node_id).await?;

            for edge in cross_domain {
                let bridge_belief = self.compute_bridge_belief(&edge)?;

                if bridge_belief >= self.config.belief_threshold {
                    candidates.push((edge.target, bridge_belief));
                    explanation_chain.push(InferenceStep {
                        from_node: *node_id,
                        to_node: edge.target,
                        edge_type: format!("bridge:{}", edge.edge_type),
                        belief_delta: bridge_belief,
                        reasoning: format!(
                            "Bridge: {} connects to {} across domains (belief: {:.3})",
                            node_id, edge.target, bridge_belief
                        ),
                    });
                }
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(InferenceResult {
            inferred_nodes: candidates.iter().map(|(id, _)| *id).collect(),
            beliefs: candidates.into_iter().collect(),
            direction: InferenceDirection::Bridge,
            latency: start.elapsed(),
            steps: explanation_chain.len(),
            explanation_chain,
        })
    }

    /// Abductive inference: find best explanation for observations
    async fn abductive_inference(
        &self,
        observations: &[Uuid],
    ) -> Result<InferenceResult, InferenceError> {
        let start = Instant::now();

        // Use backward inference as base
        let backward_result = self.backward_inference(observations).await?;

        // Score hypotheses by how well they explain ALL observations
        let mut hypothesis_scores: HashMap<Uuid, f32> = HashMap::new();

        for (hypothesis_id, base_belief) in &backward_result.beliefs {
            // Count how many observations this hypothesis explains
            let explanation_count = backward_result.explanation_chain.iter()
                .filter(|step| step.from_node == *hypothesis_id)
                .count();

            // Score = belief * coverage
            let coverage = explanation_count as f32 / observations.len() as f32;
            let abduction_score = base_belief * coverage;

            hypothesis_scores.insert(*hypothesis_id, abduction_score);
        }

        let mut candidates: Vec<_> = hypothesis_scores.into_iter().collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.max_candidates);

        Ok(InferenceResult {
            inferred_nodes: candidates.iter().map(|(id, _)| *id).collect(),
            beliefs: candidates.into_iter().collect(),
            direction: InferenceDirection::Abduction,
            latency: start.elapsed(),
            steps: backward_result.steps,
            explanation_chain: backward_result.explanation_chain,
        })
    }

    /// Compute cause belief from effect observation (Bayesian)
    fn compute_cause_belief(
        &self,
        edge: &CausalEdge,
        effect_id: Uuid,
    ) -> Result<f32, InferenceError> {
        // P(Cause | Effect) = P(Effect | Cause) * P(Cause) / P(Effect)
        let p_effect_given_cause = edge.strength;
        let p_cause = edge.prior_probability;
        let p_effect = 0.5; // Prior on effect (could be learned)

        let posterior = (p_effect_given_cause * p_cause) / p_effect;
        Ok(posterior.clamp(0.0, 1.0))
    }

    /// Compute effect belief from cause observation
    fn compute_effect_belief(
        &self,
        edge: &CausalEdge,
        cause_id: Uuid,
    ) -> Result<f32, InferenceError> {
        // P(Effect | Cause) = edge strength
        Ok(edge.strength)
    }

    /// Compute bridge belief across domains
    fn compute_bridge_belief(&self, edge: &CausalEdge) -> Result<f32, InferenceError> {
        // Discount for cross-domain transfer
        let domain_discount = 0.8;
        Ok(edge.strength * domain_discount)
    }

    // Helper methods for graph access (stubs)
    async fn get_incoming_causal_edges(&self, node_id: Uuid) -> Result<Vec<CausalEdge>, InferenceError> {
        // Implementation would query knowledge graph
        Ok(Vec::new())
    }

    async fn get_outgoing_causal_edges(&self, node_id: Uuid) -> Result<Vec<CausalEdge>, InferenceError> {
        Ok(Vec::new())
    }

    async fn get_cross_domain_edges(&self, node_id: Uuid) -> Result<Vec<CausalEdge>, InferenceError> {
        Ok(Vec::new())
    }
}

/// Causal edge in knowledge graph
#[derive(Clone, Debug)]
pub struct CausalEdge {
    pub source: Uuid,
    pub target: Uuid,
    pub edge_type: String,
    pub strength: f32,
    pub prior_probability: f32,
    pub depth: u32,
    pub domain: String,
}

/// Metrics for omnidirectional inference
#[derive(Clone, Default)]
pub struct OmniInferenceMetrics {
    pub total_inferences: u64,
    pub forward_count: u64,
    pub backward_count: u64,
    pub bidirectional_count: u64,
    pub bridge_count: u64,
    pub abduction_count: u64,
    pub avg_latency_us: f64,
}
```

---

## 4. Generative Model

### 4.1 Hierarchical Architecture

```rust
/// REQ-ACTINF-003: Hierarchical generative model implementing p(o, s) = p(o|s) * p(s)
///
/// Provides predictions at multiple abstraction levels from sensory to conceptual,
/// enabling precision-weighted prediction error computation across the hierarchy.
pub struct GenerativeModel {
    /// Hierarchical layers from sensory (L0) to conceptual (L4)
    pub layers: Vec<GenerativeLayer>,

    /// Cross-layer connections for message passing
    pub connections: Vec<LayerConnection>,

    /// Prior distribution over top-level states p(s)
    pub prior: BeliefDistribution,

    /// Likelihood model parameters p(o|s)
    pub likelihood: LikelihoodModel,

    /// Model configuration
    pub config: GenerativeModelConfig,
}

/// Configuration for generative model
#[derive(Clone, Debug)]
pub struct GenerativeModelConfig {
    /// Number of hierarchical layers
    pub num_layers: usize,  // Default: 5 (matching bio-nervous system)

    /// Dimensions per layer
    pub layer_dims: Vec<usize>,  // Default: [1536, 768, 384, 192, 96]

    /// Learning rate for model updates
    pub learning_rate: f32,  // Default: 0.001

    /// Precision for each layer
    pub layer_precisions: Vec<f32>,  // Default: [1.0, 0.8, 0.6, 0.4, 0.2]
}

impl Default for GenerativeModelConfig {
    fn default() -> Self {
        Self {
            num_layers: 5,
            layer_dims: vec![1536, 768, 384, 192, 96],
            learning_rate: 0.001,
            layer_precisions: vec![1.0, 0.8, 0.6, 0.4, 0.2],
        }
    }
}

/// Single layer in the generative hierarchy
#[derive(Clone)]
pub struct GenerativeLayer {
    /// Layer index (0 = sensory, higher = more abstract)
    pub level: usize,

    /// Layer dimension
    pub dimension: usize,

    /// Current state estimate at this layer
    pub state: DVector<f32>,

    /// Precision of state estimates at this layer
    pub precision: f32,

    /// Transformation matrix to lower layer
    pub downward_weights: Option<DMatrix<f32>>,

    /// Transformation matrix from lower layer
    pub upward_weights: Option<DMatrix<f32>>,
}

impl GenerativeModel {
    /// Create new generative model
    pub fn new(config: GenerativeModelConfig) -> Self {
        let mut layers = Vec::new();

        for (i, &dim) in config.layer_dims.iter().enumerate() {
            let downward = if i > 0 {
                Some(DMatrix::from_fn(
                    config.layer_dims[i - 1],
                    dim,
                    |_, _| rand::random::<f32>() * 0.1,
                ))
            } else {
                None
            };

            let upward = if i < config.layer_dims.len() - 1 {
                Some(DMatrix::from_fn(
                    dim,
                    config.layer_dims[i + 1],
                    |_, _| rand::random::<f32>() * 0.1,
                ))
            } else {
                None
            };

            layers.push(GenerativeLayer {
                level: i,
                dimension: dim,
                state: DVector::zeros(dim),
                precision: config.layer_precisions.get(i).copied().unwrap_or(1.0),
                downward_weights: downward,
                upward_weights: upward,
            });
        }

        Self {
            layers,
            connections: Vec::new(),
            prior: BeliefDistribution::uniform_prior(config.layer_dims[0]),
            likelihood: LikelihoodModel::new(config.layer_dims[0]),
            config,
        }
    }

    /// Generate prediction for given beliefs
    ///
    /// `Constraint: Prediction_Latency < 2ms`
    #[inline]
    pub fn predict(&self, beliefs: &BeliefDistribution) -> Prediction {
        let start = Instant::now();

        // Top-down prediction through hierarchy
        let mut current_state = beliefs.mean.clone();

        for layer in self.layers.iter().rev() {
            if let Some(ref weights) = layer.downward_weights {
                current_state = weights * &current_state;
            }
        }

        // Apply likelihood model
        let predicted_obs = &self.likelihood.state_to_obs * &current_state;

        Prediction {
            observation: predicted_obs,
            confidence: beliefs.confidence * self.likelihood.obs_precision,
            latency: start.elapsed(),
        }
    }

    /// Compute prediction error with precision weighting
    ///
    /// `Constraint: Error_Computation < 1ms`
    #[inline]
    pub fn compute_prediction_error(
        &self,
        predicted: &DVector<f32>,
        observed: &DVector<f32>,
    ) -> PredictionError {
        let error = observed - predicted;
        let precision = self.likelihood.obs_precision;

        PredictionError {
            prediction: predicted.clone(),
            observation: observed.clone(),
            error: error.clone(),
            precision,
            timestamp: Instant::now(),
            domain: String::new(),
        }
    }

    /// Update model based on prediction error (learning)
    ///
    /// `Constraint: Model_Update < 3ms`
    pub fn update_from_error(&mut self, error: &PredictionError, learning_rate: f32) {
        // Precision-weighted gradient descent
        let weighted_error = &error.error * error.precision * learning_rate;

        // Update likelihood model (simplified gradient)
        for (i, row) in self.likelihood.state_to_obs.row_iter_mut().enumerate() {
            if i < weighted_error.len() {
                for val in row.iter_mut() {
                    *val += weighted_error[i] * 0.01;
                }
            }
        }
    }
}

/// Prediction output
#[derive(Clone)]
pub struct Prediction {
    /// Predicted observation embedding
    pub observation: DVector<f32>,

    /// Prediction confidence
    pub confidence: f32,

    /// Computation latency
    pub latency: Duration,
}

/// Likelihood model p(o|s) for observation prediction
#[derive(Clone)]
pub struct LikelihoodModel {
    /// Mapping from hidden states to observation predictions
    pub state_to_obs: DMatrix<f32>,

    /// Observation noise precision
    pub obs_precision: f32,

    /// Domain-specific likelihood adjustments
    pub domain_adjustments: HashMap<String, f32>,
}

impl LikelihoodModel {
    pub fn new(dimension: usize) -> Self {
        Self {
            state_to_obs: DMatrix::identity(dimension, dimension),
            obs_precision: 1.0,
            domain_adjustments: HashMap::new(),
        }
    }
}
```

---

## 5. Belief Updater

### 5.1 Variational Inference

```rust
/// REQ-ACTINF-004: Belief updater implementing variational inference
/// for approximate posterior computation.
///
/// Updates beliefs q(s) to minimize variational free energy given observations,
/// using precision-weighted prediction errors for efficient inference.
pub struct BeliefUpdater {
    /// Variational inference configuration
    pub config: BeliefUpdaterConfig,

    /// Iteration history for convergence tracking
    pub iteration_history: Vec<IterationMetrics>,

    /// Current convergence state
    pub converged: bool,
}

/// Configuration for belief updating
#[derive(Clone, Debug)]
pub struct BeliefUpdaterConfig {
    /// Maximum iterations for variational inference
    pub max_iterations: usize,  // Default: 10

    /// Convergence threshold (KL divergence)
    pub convergence_threshold: f32,  // Default: 0.001

    /// Learning rate for belief updates
    pub learning_rate: f32,  // Default: 0.1

    /// Momentum for gradient updates
    pub momentum: f32,  // Default: 0.9

    /// Minimum precision floor (prevents division by zero)
    pub precision_floor: f32,  // Default: 1e-6

    /// Maximum update magnitude (gradient clipping)
    pub max_update_magnitude: f32,  // Default: 1.0
}

impl Default for BeliefUpdaterConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            convergence_threshold: 0.001,
            learning_rate: 0.1,
            momentum: 0.9,
            precision_floor: 1e-6,
            max_update_magnitude: 1.0,
        }
    }
}

impl BeliefUpdater {
    /// Create new belief updater with configuration
    pub fn new(config: BeliefUpdaterConfig) -> Self {
        Self {
            config,
            iteration_history: Vec::new(),
            converged: false,
        }
    }

    /// Update beliefs given new observation
    ///
    /// Implements variational message passing to approximate q(s) that minimizes F.
    ///
    /// `Constraint: Belief_Update_Latency < 5ms`
    pub fn update_beliefs(
        &mut self,
        beliefs: &mut BeliefDistribution,
        observation: &Observation,
        generative_model: &GenerativeModel,
    ) -> BeliefUpdateResult {
        let start = Instant::now();
        self.iteration_history.clear();
        self.converged = false;

        let mut prev_beliefs = beliefs.clone();
        let mut velocity = DVector::zeros(beliefs.mean.len());

        for iter in 0..self.config.max_iterations {
            let iter_start = Instant::now();

            // Compute prediction from current beliefs
            let prediction = generative_model.predict(beliefs);

            // Compute precision-weighted prediction error
            let error = &observation.embedding - &prediction.observation;

            // Gradient of free energy w.r.t. belief mean
            // dF/dmu = precision * error + prior_precision * (mu - mu_prior)
            let prior_error = &beliefs.mean - &generative_model.prior.mean;
            let gradient: DVector<f32> = error.iter()
                .zip(prior_error.iter())
                .zip(beliefs.precision.iter())
                .zip(generative_model.prior.precision.iter())
                .map(|(((e, pe), bp), pp)| {
                    -e * generative_model.likelihood.obs_precision
                    + pe * pp.max(self.config.precision_floor)
                        / bp.max(self.config.precision_floor)
                })
                .collect::<Vec<f32>>()
                .into();

            // Apply momentum
            velocity = &velocity * self.config.momentum - &gradient * self.config.learning_rate;

            // Clip update magnitude
            let magnitude: f32 = velocity.iter().map(|v| v * v).sum::<f32>().sqrt();
            if magnitude > self.config.max_update_magnitude {
                velocity = &velocity * (self.config.max_update_magnitude / magnitude);
            }

            // Update beliefs
            beliefs.mean = &beliefs.mean + &velocity;

            // Update precision based on prediction error variance
            let error_sq: DVector<f32> = error.iter()
                .map(|e| e * e)
                .collect::<Vec<f32>>()
                .into();
            beliefs.precision = beliefs.precision.iter()
                .zip(error_sq.iter())
                .map(|(&p, &e)| {
                    let new_p = p * 0.9 + 0.1 / (e + self.config.precision_floor);
                    new_p.max(self.config.precision_floor)
                })
                .collect::<Vec<f32>>()
                .into();

            // Compute convergence metric
            let kl_delta = beliefs.kl_divergence(&prev_beliefs);

            // Compute free energy
            let free_energy = self.compute_free_energy(
                beliefs,
                &observation.embedding,
                generative_model
            );

            // Store iteration metrics
            self.iteration_history.push(IterationMetrics {
                iteration: iter,
                free_energy,
                kl_delta,
                update_magnitude: magnitude,
                duration: iter_start.elapsed(),
            });

            // Check convergence
            if kl_delta < self.config.convergence_threshold {
                self.converged = true;
                break;
            }

            prev_beliefs = beliefs.clone();
        }

        // Update confidence and observation count
        beliefs.confidence = 1.0 - beliefs.entropy() / (beliefs.mean.len() as f32 * 10.0);
        beliefs.confidence = beliefs.confidence.clamp(0.0, 1.0);
        beliefs.observation_count += 1;

        BeliefUpdateResult {
            converged: self.converged,
            iterations: self.iteration_history.len(),
            final_free_energy: self.iteration_history.last()
                .map(|m| m.free_energy)
                .unwrap_or(f32::MAX),
            uncertainty_reduction: prev_beliefs.entropy() - beliefs.entropy(),
            latency: start.elapsed(),
        }
    }

    /// Compute variational free energy F = D_KL[q||p] - ln p(o|s)
    fn compute_free_energy(
        &self,
        beliefs: &BeliefDistribution,
        observation: &DVector<f32>,
        model: &GenerativeModel,
    ) -> f32 {
        // KL divergence term
        let kl_term = beliefs.kl_divergence(&model.prior);

        // Negative log likelihood term
        let prediction = model.predict(beliefs);
        let error: f32 = observation.iter()
            .zip(prediction.observation.iter())
            .map(|(o, p)| (o - p).powi(2))
            .sum();
        let nll_term = 0.5 * error * model.likelihood.obs_precision;

        kl_term + nll_term
    }
}

/// Result of belief update operation
#[derive(Clone, Debug)]
pub struct BeliefUpdateResult {
    /// Whether inference converged
    pub converged: bool,

    /// Number of iterations performed
    pub iterations: usize,

    /// Final free energy value
    pub final_free_energy: f32,

    /// Reduction in uncertainty (entropy)
    pub uncertainty_reduction: f32,

    /// Total update latency
    pub latency: Duration,
}

/// Metrics for single iteration of belief update
#[derive(Clone, Debug)]
pub struct IterationMetrics {
    /// Iteration number
    pub iteration: usize,

    /// Free energy at this iteration
    pub free_energy: f32,

    /// KL divergence from previous iteration
    pub kl_delta: f32,

    /// Update magnitude
    pub update_magnitude: f32,

    /// Iteration duration
    pub duration: Duration,
}
```

---

## 6. Policy Evaluator

### 6.1 Expected Free Energy Computation

```rust
/// REQ-ACTINF-005: Evaluates policies based on Expected Free Energy (EFE)
/// with temporal planning.
///
/// Implements tree search over possible action sequences to select policies
/// that minimize expected free energy over a planning horizon.
pub struct PolicyEvaluator {
    /// Available policy templates
    pub policy_templates: Vec<PolicyTemplate>,

    /// EFE computation cache
    pub efe_cache: HashMap<PolicyCacheKey, f32>,

    /// Configuration
    pub config: PolicyEvaluatorConfig,

    /// Evaluation metrics
    pub metrics: PolicyEvaluationMetrics,
}

/// Configuration for policy evaluation
#[derive(Clone, Debug)]
pub struct PolicyEvaluatorConfig {
    /// Maximum planning depth (steps ahead)
    pub max_depth: usize,  // Default: 5

    /// Branching factor (policies per level)
    pub branching_factor: usize,  // Default: 4

    /// Discount factor for future EFE
    pub discount_factor: f32,  // Default: 0.9

    /// Pruning threshold (skip low-probability branches)
    pub pruning_threshold: f32,  // Default: 0.01

    /// EFE computation timeout per policy
    pub efe_timeout: Duration,  // Default: 2ms

    /// Temperature for softmax policy selection
    pub selection_temperature: f32,  // Default: 1.0
}

impl Default for PolicyEvaluatorConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            branching_factor: 4,
            discount_factor: 0.9,
            pruning_threshold: 0.01,
            efe_timeout: Duration::from_millis(2),
            selection_temperature: 1.0,
        }
    }
}

impl PolicyEvaluator {
    /// Create new policy evaluator
    pub fn new(config: PolicyEvaluatorConfig) -> Self {
        Self {
            policy_templates: Self::default_policy_templates(),
            efe_cache: HashMap::new(),
            config,
            metrics: PolicyEvaluationMetrics::default(),
        }
    }

    /// Default policy templates
    fn default_policy_templates() -> Vec<PolicyTemplate> {
        vec![
            PolicyTemplate {
                id: "explore".to_string(),
                actions: vec![EpistemicActionType::ExploreRelated],
                prior_prob: 0.25,
                description: "Explore related concepts".to_string(),
            },
            PolicyTemplate {
                id: "clarify".to_string(),
                actions: vec![EpistemicActionType::SeekClarification],
                prior_prob: 0.25,
                description: "Seek clarification".to_string(),
            },
            PolicyTemplate {
                id: "hypothesize".to_string(),
                actions: vec![EpistemicActionType::ProposeHypothesis],
                prior_prob: 0.25,
                description: "Propose hypothesis".to_string(),
            },
            PolicyTemplate {
                id: "example".to_string(),
                actions: vec![EpistemicActionType::RequestExample],
                prior_prob: 0.25,
                description: "Request example".to_string(),
            },
        ]
    }

    /// Evaluate all policies and return ranked list
    ///
    /// `Constraint: Total_Evaluation < 10ms`
    pub fn evaluate_policies(
        &mut self,
        beliefs: &SessionBeliefs,
        generative_model: &GenerativeModel,
        goal_preferences: &GoalPreferences,
    ) -> Vec<PolicyEvaluation> {
        let start = Instant::now();
        let mut evaluations = Vec::new();

        for template in &self.policy_templates {
            // Check cache first
            let cache_key = PolicyCacheKey {
                policy_id: template.id.clone(),
                belief_hash: self.hash_beliefs(&beliefs.state_beliefs),
                depth: 0,
            };

            let efe = if let Some(&cached_efe) = self.efe_cache.get(&cache_key) {
                self.metrics.cache_hits += 1;
                cached_efe
            } else {
                let computed_efe = self.compute_efe(
                    template,
                    &beliefs.state_beliefs,
                    generative_model,
                    goal_preferences,
                    0,
                );
                self.efe_cache.insert(cache_key, computed_efe);
                computed_efe
            };

            evaluations.push(PolicyEvaluation {
                policy_id: template.id.clone(),
                efe,
                epistemic_value: self.compute_epistemic_value(&beliefs.state_beliefs, generative_model),
                pragmatic_value: self.compute_pragmatic_value(&beliefs.state_beliefs, goal_preferences),
                probability: 0.0,
            });

            self.metrics.policies_evaluated += 1;
        }

        // Compute softmax probabilities
        self.apply_softmax_probabilities(&mut evaluations);

        // Sort by EFE (lower is better)
        evaluations.sort_by(|a, b| a.efe.partial_cmp(&b.efe).unwrap_or(std::cmp::Ordering::Equal));

        evaluations
    }

    /// Compute Expected Free Energy for a policy
    ///
    /// G(pi) = E_q[ln q(s') - ln p(o', s')]
    ///       = Epistemic + Pragmatic
    ///
    /// `Constraint: Single_EFE < 2ms`
    fn compute_efe(
        &self,
        policy: &PolicyTemplate,
        beliefs: &BeliefDistribution,
        model: &GenerativeModel,
        goals: &GoalPreferences,
        depth: usize,
    ) -> f32 {
        if depth >= self.config.max_depth {
            return 0.0;
        }

        // Epistemic value: expected information gain
        let epistemic = self.compute_epistemic_value(beliefs, model);

        // Pragmatic value: expected goal satisfaction
        let pragmatic = self.compute_pragmatic_value(beliefs, goals);

        // Combine with discount for future steps
        let current_efe = epistemic + pragmatic;
        let future_efe = if depth + 1 < self.config.max_depth {
            self.config.discount_factor * current_efe
        } else {
            0.0
        };

        current_efe + future_efe
    }

    /// Compute epistemic value (information gain)
    ///
    /// Epistemic = -E_q[H[p(o|s)]] (negative expected ambiguity)
    #[inline]
    fn compute_epistemic_value(
        &self,
        beliefs: &BeliefDistribution,
        model: &GenerativeModel,
    ) -> f32 {
        let prediction = model.predict(beliefs);
        let ambiguity = 1.0 / (model.likelihood.obs_precision + 1e-6);
        let info_gain = beliefs.entropy() * ambiguity;
        -info_gain
    }

    /// Compute pragmatic value (goal satisfaction)
    ///
    /// Pragmatic = E_q[ln p(o|C)] (expected log preference)
    #[inline]
    fn compute_pragmatic_value(
        &self,
        beliefs: &BeliefDistribution,
        goals: &GoalPreferences,
    ) -> f32 {
        if goals.preferred_states.is_empty() {
            return 0.0;
        }

        let mut total_preference = 0.0;
        let mut total_weight = 0.0;

        for (preferred, weight) in goals.preferred_states.iter().zip(goals.preference_weights.iter()) {
            let dot: f32 = beliefs.mean.iter()
                .zip(preferred.iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm_belief: f32 = beliefs.mean.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_pref: f32 = preferred.iter().map(|x| x * x).sum::<f32>().sqrt();

            let similarity = if norm_belief > 1e-6 && norm_pref > 1e-6 {
                dot / (norm_belief * norm_pref)
            } else {
                0.0
            };

            total_preference += similarity * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            total_preference / total_weight
        } else {
            0.0
        }
    }

    /// Apply softmax to convert EFE to probabilities
    fn apply_softmax_probabilities(&self, evaluations: &mut [PolicyEvaluation]) {
        let temp = self.config.selection_temperature;

        let max_efe = evaluations.iter()
            .map(|e| e.efe)
            .fold(f32::NEG_INFINITY, f32::max);

        let exp_sum: f32 = evaluations.iter()
            .map(|e| (-(e.efe - max_efe) / temp).exp())
            .sum();

        for eval in evaluations.iter_mut() {
            eval.probability = (-(eval.efe - max_efe) / temp).exp() / exp_sum;
        }
    }

    /// Hash beliefs for cache key
    fn hash_beliefs(&self, beliefs: &BeliefDistribution) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        for &val in beliefs.mean.iter().take(16) {
            let bits = val.to_bits();
            bits.hash(&mut hasher);
        }

        hasher.finish()
    }
}

/// Result of policy evaluation
#[derive(Clone, Debug)]
pub struct PolicyEvaluation {
    /// Policy identifier
    pub policy_id: String,
    /// Expected Free Energy (lower is better)
    pub efe: f32,
    /// Epistemic value component
    pub epistemic_value: f32,
    /// Pragmatic value component
    pub pragmatic_value: f32,
    /// Selection probability (softmax)
    pub probability: f32,
}
```

---

## 7. Epistemic Action Generator

### 7.1 Action Generation

```rust
/// REQ-ACTINF-006: Generates epistemic actions to reduce uncertainty
/// and acquire information.
pub struct EpistemicActionGenerator {
    /// Action templates for different scenarios
    pub action_templates: Vec<ActionTemplate>,

    /// Rate limiter for action generation
    pub rate_limiter: ActionRateLimiter,

    /// Configuration
    pub config: EpistemicActionConfig,

    /// Generation metrics
    pub metrics: ActionGenerationMetrics,
}

/// Types of epistemic actions
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EpistemicActionType {
    /// Request clarification on ambiguous concept
    SeekClarification,
    /// Request concrete example for abstract concept
    RequestExample,
    /// Propose hypothesis based on partial data
    ProposeHypothesis,
    /// Suggest experiment to test hypothesis
    SuggestExperiment,
    /// Explore related concept
    ExploreRelated,
    /// Request confirmation of understanding
    ConfirmUnderstanding,
}

/// Generated epistemic action
#[derive(Clone, Debug)]
pub struct EpistemicAction {
    /// Action type
    pub action_type: EpistemicActionType,
    /// Action content (question, hypothesis, etc.)
    pub content: String,
    /// Target concept or domain
    pub target_concept: Option<String>,
    /// Expected information gain
    pub expected_info_gain: f32,
    /// Priority score for ranking
    pub priority: f32,
    /// Contextual relevance score
    pub relevance: f32,
    /// Associated knowledge gap
    pub knowledge_gap: Option<KnowledgeGap>,
}

impl EpistemicActionGenerator {
    /// Create new action generator
    pub fn new(config: EpistemicActionConfig) -> Self {
        Self {
            action_templates: Self::default_templates(),
            rate_limiter: ActionRateLimiter::new(Duration::from_secs(60), 3),
            config,
            metrics: ActionGenerationMetrics::default(),
        }
    }

    /// Generate epistemic actions based on current state
    ///
    /// `Constraint: Generation_Latency < 5ms`
    pub fn generate_actions(
        &mut self,
        beliefs: &SessionBeliefs,
        context: &ActionContext,
    ) -> Vec<EpistemicAction> {
        let start = Instant::now();
        let mut actions = Vec::new();

        // Check if uncertainty threshold is met
        if beliefs.belief_entropy < self.config.uncertainty_threshold {
            return actions;
        }

        // Identify knowledge gaps
        let gaps = self.identify_knowledge_gaps(beliefs, context);

        // Generate actions for each gap
        for gap in &gaps {
            let action = self.generate_action_for_gap(gap, beliefs, context);

            // Check rate limiter
            if self.rate_limiter.is_allowed(&action.action_type) {
                if action.expected_info_gain >= self.config.min_info_gain {
                    actions.push(action);
                    if actions.len() >= self.config.max_actions {
                        break;
                    }
                }
            } else {
                self.metrics.rate_limited += 1;
            }
        }

        // Ensure diversity if required
        if self.config.require_diversity {
            actions = self.ensure_diversity(actions);
        }

        // Sort by priority
        actions.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

        // Record actions
        for action in &actions {
            self.rate_limiter.record_action(action.action_type.clone());
            self.metrics.total_generated += 1;
        }

        // Truncate to max
        actions.truncate(self.config.max_actions);

        actions
    }

    /// Identify knowledge gaps from belief state
    fn identify_knowledge_gaps(
        &self,
        beliefs: &SessionBeliefs,
        context: &ActionContext,
    ) -> Vec<KnowledgeGap> {
        let mut gaps = Vec::new();

        // Check domain-specific uncertainty
        for (domain, &uncertainty) in &beliefs.domain_uncertainty {
            if uncertainty > self.config.uncertainty_threshold {
                gaps.push(KnowledgeGap {
                    id: Uuid::new_v4(),
                    description: format!("High uncertainty in domain: {}", domain),
                    domain: domain.clone(),
                    severity: uncertainty,
                    related_nodes: Vec::new(),
                });
            }
        }

        // Use Johari quadrant to identify exploration opportunities
        match beliefs.johari_quadrant {
            JohariQuadrant::Blind => {
                gaps.push(KnowledgeGap {
                    id: Uuid::new_v4(),
                    description: "Blind spot detected".to_string(),
                    domain: context.current_domain.clone().unwrap_or_default(),
                    severity: 0.8,
                    related_nodes: Vec::new(),
                });
            }
            JohariQuadrant::Unknown => {
                gaps.push(KnowledgeGap {
                    id: Uuid::new_v4(),
                    description: "Unknown territory".to_string(),
                    domain: context.current_domain.clone().unwrap_or_default(),
                    severity: 0.9,
                    related_nodes: Vec::new(),
                });
            }
            _ => {}
        }

        gaps.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
        gaps
    }

    /// Generate action for specific knowledge gap
    fn generate_action_for_gap(
        &self,
        gap: &KnowledgeGap,
        beliefs: &SessionBeliefs,
        context: &ActionContext,
    ) -> EpistemicAction {
        let action_type = match beliefs.johari_quadrant {
            JohariQuadrant::Blind => EpistemicActionType::SeekClarification,
            JohariQuadrant::Unknown => EpistemicActionType::ExploreRelated,
            JohariQuadrant::Hidden => EpistemicActionType::ConfirmUnderstanding,
            JohariQuadrant::Open => {
                if gap.severity > 0.7 {
                    EpistemicActionType::ProposeHypothesis
                } else {
                    EpistemicActionType::RequestExample
                }
            }
        };

        let content = self.generate_action_content(&action_type, gap);
        let expected_info_gain = gap.severity * beliefs.belief_entropy;
        let priority = expected_info_gain * (1.0 + context.urgency);
        let relevance = if let Some(ref current_domain) = context.current_domain {
            if gap.domain == *current_domain { 1.0 } else { 0.5 }
        } else {
            0.7
        };

        EpistemicAction {
            action_type,
            content,
            target_concept: Some(gap.domain.clone()),
            expected_info_gain,
            priority,
            relevance,
            knowledge_gap: Some(gap.clone()),
        }
    }

    fn generate_action_content(&self, action_type: &EpistemicActionType, gap: &KnowledgeGap) -> String {
        match action_type {
            EpistemicActionType::SeekClarification => {
                format!("Could you clarify what you mean by '{}' in this context?", gap.domain)
            }
            EpistemicActionType::RequestExample => {
                format!("Could you provide a concrete example of {}?", gap.domain)
            }
            EpistemicActionType::ProposeHypothesis => {
                format!("Based on current context, I hypothesize that {}. Is this correct?", gap.description)
            }
            EpistemicActionType::SuggestExperiment => {
                format!("To better understand {}, we could try testing...", gap.domain)
            }
            EpistemicActionType::ExploreRelated => {
                format!("Should we explore related concepts to {}?", gap.domain)
            }
            EpistemicActionType::ConfirmUnderstanding => {
                format!("Let me confirm my understanding of {}: Is this accurate?", gap.domain)
            }
        }
    }

    fn ensure_diversity(&self, mut actions: Vec<EpistemicAction>) -> Vec<EpistemicAction> {
        let mut seen_types = std::collections::HashSet::new();
        actions.retain(|action| {
            if seen_types.contains(&action.action_type) {
                false
            } else {
                seen_types.insert(action.action_type.clone());
                true
            }
        });
        actions
    }

    fn default_templates() -> Vec<ActionTemplate> {
        vec![]  // Templates loaded from configuration
    }
}

/// Rate limiter to prevent action spam
pub struct ActionRateLimiter {
    recent_actions: HashMap<EpistemicActionType, Vec<Instant>>,
    cooldown: Duration,
    max_per_type: usize,
}

impl ActionRateLimiter {
    pub fn new(cooldown: Duration, max_per_type: usize) -> Self {
        Self {
            recent_actions: HashMap::new(),
            cooldown,
            max_per_type,
        }
    }

    pub fn is_allowed(&self, action_type: &EpistemicActionType) -> bool {
        if let Some(recent) = self.recent_actions.get(action_type) {
            let now = Instant::now();
            let recent_count = recent.iter()
                .filter(|&&t| now.duration_since(t) < self.cooldown)
                .count();
            recent_count < self.max_per_type
        } else {
            true
        }
    }

    pub fn record_action(&mut self, action_type: EpistemicActionType) {
        self.recent_actions
            .entry(action_type)
            .or_insert_with(Vec::new)
            .push(Instant::now());
    }
}
```

---

## 8. MCP Integration

### 8.1 epistemic_action Handler

```rust
use serde::{Deserialize, Serialize};

/// MCP tool request for epistemic action generation
#[derive(Clone, Debug, Deserialize)]
pub struct EpistemicActionRequest {
    /// Session identifier
    pub session_id: Uuid,
    /// Force action generation even below threshold
    #[serde(default)]
    pub force: bool,
    /// Maximum actions to return
    #[serde(default = "default_max_actions")]
    pub max_actions: usize,
    /// Minimum relevance score
    #[serde(default = "default_min_relevance")]
    pub min_relevance: f32,
}

fn default_max_actions() -> usize { 3 }
fn default_min_relevance() -> f32 { 0.3 }

/// MCP tool response for epistemic action
#[derive(Clone, Debug, Serialize)]
pub struct EpistemicActionResponse {
    /// Generated epistemic actions
    pub actions: Vec<EpistemicActionOutput>,
    /// Current uncertainty level
    pub uncertainty_level: f32,
    /// Johari quadrant
    pub johari_quadrant: String,
    /// UTL metrics
    pub utl_metrics: UtlMetricsOutput,
    /// Cognitive pulse (standard header)
    pub pulse: CognitivePulse,
    /// Processing latency
    pub latency_ms: f32,
}

/// Single epistemic action output
#[derive(Clone, Debug, Serialize)]
pub struct EpistemicActionOutput {
    pub action_type: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_concept: Option<String>,
    pub expected_info_gain: f32,
    pub relevance: f32,
    pub priority: u32,
}

/// Cognitive pulse header
#[derive(Clone, Debug, Serialize)]
pub struct CognitivePulse {
    pub entropy: f32,
    pub coherence: f32,
    pub suggested_action: String,
}

/// REQ-ACTINF-007: Handler for epistemic_action MCP tool
pub struct EpistemicActionHandler {
    pub engine: Arc<RwLock<ActiveInferenceEngine>>,
    pub sessions: Arc<RwLock<SessionManager>>,
}

impl EpistemicActionHandler {
    /// Handle MCP epistemic_action request
    ///
    /// `Constraint: Total_Handler_Latency < 25ms`
    pub async fn handle(
        &self,
        request: EpistemicActionRequest,
    ) -> Result<EpistemicActionResponse, ActiveInferenceError> {
        let start = Instant::now();

        // Get session beliefs
        let sessions = self.sessions.read().await;
        let session = sessions.get(&request.session_id)
            .ok_or(ActiveInferenceError::SessionNotFound(request.session_id))?;

        let beliefs = session.beliefs.read().await;

        // Check uncertainty threshold unless forced
        if !request.force && beliefs.belief_entropy < 0.6 {
            return Ok(EpistemicActionResponse {
                actions: Vec::new(),
                uncertainty_level: beliefs.belief_entropy,
                johari_quadrant: format!("{:?}", beliefs.johari_quadrant),
                utl_metrics: UtlMetricsOutput {
                    entropy: beliefs.belief_entropy,
                    coherence: beliefs.state_beliefs.confidence,
                    learning_score: 0.0,
                },
                pulse: CognitivePulse {
                    entropy: beliefs.belief_entropy,
                    coherence: beliefs.state_beliefs.confidence,
                    suggested_action: "continue".to_string(),
                },
                latency_ms: start.elapsed().as_secs_f32() * 1000.0,
            });
        }

        // Generate actions
        let mut engine = self.engine.write().await;
        let context = ActionContext {
            current_domain: session.current_domain.clone(),
            recent_queries: session.recent_queries.clone(),
            urgency: 0.5,
            session_id: request.session_id,
        };

        let actions = engine.action_generator.generate_actions(&beliefs, &context);

        // Filter by relevance
        let filtered_actions: Vec<_> = actions.into_iter()
            .filter(|a| a.relevance >= request.min_relevance)
            .take(request.max_actions)
            .collect();

        // Convert to output format
        let action_outputs: Vec<EpistemicActionOutput> = filtered_actions.iter()
            .enumerate()
            .map(|(i, a)| EpistemicActionOutput {
                action_type: format!("{:?}", a.action_type),
                content: a.content.clone(),
                target_concept: a.target_concept.clone(),
                expected_info_gain: a.expected_info_gain,
                relevance: a.relevance,
                priority: (i + 1) as u32,
            })
            .collect();

        let suggested_action = if beliefs.belief_entropy > 0.7 {
            if beliefs.state_beliefs.confidence < 0.4 { "trigger_dream" } else { "explore" }
        } else {
            "continue"
        };

        Ok(EpistemicActionResponse {
            actions: action_outputs,
            uncertainty_level: beliefs.belief_entropy,
            johari_quadrant: format!("{:?}", beliefs.johari_quadrant),
            utl_metrics: UtlMetricsOutput {
                entropy: beliefs.belief_entropy,
                coherence: beliefs.state_beliefs.confidence,
                learning_score: 0.0,
            },
            pulse: CognitivePulse {
                entropy: beliefs.belief_entropy,
                coherence: beliefs.state_beliefs.confidence,
                suggested_action: suggested_action.to_string(),
            },
            latency_ms: start.elapsed().as_secs_f32() * 1000.0,
        })
    }
}
```

---

## 9. Error Handling

```rust
use thiserror::Error;

/// Errors for Active Inference module
#[derive(Debug, Error)]
pub enum ActiveInferenceError {
    #[error("Session not found: {0}")]
    SessionNotFound(Uuid),

    #[error("Belief update failed: {0}")]
    BeliefUpdateFailed(String),

    #[error("EFE computation timed out after {0}ms")]
    EfeTimeout(u64),

    #[error("Invalid generative model: {0}")]
    InvalidModel(String),

    #[error("Policy evaluation failed: {0}")]
    PolicyEvaluationFailed(String),

    #[error("Action generation failed: {0}")]
    ActionGenerationFailed(String),

    #[error("Numerical instability: {0}")]
    NumericalInstability(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl ActiveInferenceError {
    pub fn error_code(&self) -> i32 {
        match self {
            Self::SessionNotFound(_) => -32000,
            Self::BeliefUpdateFailed(_) => -32010,
            Self::EfeTimeout(_) => -32011,
            Self::InvalidModel(_) => -32012,
            Self::PolicyEvaluationFailed(_) => -32013,
            Self::ActionGenerationFailed(_) => -32014,
            Self::NumericalInstability(_) => -32015,
            Self::ConfigurationError(_) => -32016,
            Self::Internal(_) => -32603,
        }
    }
}

/// Errors for omnidirectional inference
#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("Node not found: {0}")]
    NodeNotFound(Uuid),

    #[error("Edge traversal failed: {0}")]
    EdgeTraversalFailed(String),

    #[error("Inference depth exceeded: {0}")]
    DepthExceeded(u32),

    #[error("Timeout during inference")]
    Timeout,

    #[error("Invalid clamped value: {0}")]
    InvalidClamp(String),
}
```

---

## 10. Performance Metrics

```rust
/// Performance metrics for Active Inference operations
#[derive(Clone, Default)]
pub struct ActiveInferenceMetrics {
    /// Total EFE computations performed
    pub efe_computations: u64,
    /// Average EFE computation latency (microseconds)
    pub avg_efe_latency_us: f64,
    /// Total belief updates performed
    pub belief_updates: u64,
    /// Average belief update latency (microseconds)
    pub avg_belief_latency_us: f64,
    /// Total actions generated
    pub actions_generated: u64,
    /// Actions rated helpful by users
    pub actions_helpful: u64,
    /// Total uncertainty reduction achieved
    pub total_uncertainty_reduction: f64,
    /// Last update timestamp
    pub last_update: Option<Instant>,
}

impl ActiveInferenceMetrics {
    /// Compute action helpfulness ratio
    pub fn helpfulness_ratio(&self) -> f32 {
        if self.actions_generated > 0 {
            self.actions_helpful as f32 / self.actions_generated as f32
        } else {
            0.0
        }
    }
}
```

---

## 11. Verification Matrix

### 11.1 Unit Tests

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| test_efe_computation | EFE formula correctness | Matches reference |
| test_belief_update | Variational inference | Convergence <10 iterations |
| test_policy_ranking | Policy ordering by EFE | Lower EFE = higher rank |
| test_action_generation | Action template instantiation | Valid content |
| test_rate_limiting | Action spam prevention | <3 per response |
| test_numerical_stability | Precision floor enforcement | No NaN/Inf |
| test_omni_forward | Forward inference | Correct effects |
| test_omni_backward | Backward inference | Identifies causes |
| test_omni_bidirectional | Bidirectional inference | Both directions |
| test_omni_bridge | Bridge inference | Cross-domain works |
| test_clamped_hard | Hard clamps | Values unchanged |
| test_clamped_soft | Soft clamps | Values biased |

### 11.2 Performance Benchmarks

| Benchmark | Target | Method |
|-----------|--------|--------|
| bench_efe_throughput | >10000/sec | criterion::bench |
| bench_belief_update_p99 | <5ms | latency histogram |
| bench_action_generation_p99 | <5ms | latency histogram |
| bench_omni_inference_p99 | <15ms | latency histogram |
| bench_concurrent_sessions | >100 | tokio stress test |

---

## 12. Traceability

| Requirement | PRD Section | Implementation |
|-------------|-------------|----------------|
| REQ-ACTINF-001 | 5.2 Core Tools | ActiveInferenceEngine |
| REQ-ACTINF-002 | 2.1 UTL Core | SessionBeliefs |
| REQ-ACTINF-003 | 2.3 5-Layer | GenerativeModel |
| REQ-ACTINF-004 | 2.1 UTL Formula | BeliefUpdater |
| REQ-ACTINF-005 | 6.1 inject_context | PolicyEvaluator |
| REQ-ACTINF-006 | 5.2 epistemic_action | EpistemicActionGenerator |
| REQ-ACTINF-007 | 5.1 Protocol | MCP Handler |
| REQ-ACTINF-008 | Marblestone | OmniInferenceEngine |
| REQ-ACTINF-009 | Marblestone | Backward inference |
| REQ-ACTINF-010 | Marblestone | ClampedValue |

---

*Document Version: 1.0.0*
*Created: 2025-12-31*
*Lines: ~800*
