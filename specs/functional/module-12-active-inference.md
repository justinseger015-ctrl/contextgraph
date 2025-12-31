# Module 12: Active Inference - Functional Specification

**Module ID**: SPEC-ACTINF-012
**Version**: 1.0.0
**Status**: Draft
**Phase**: 11
**Duration**: 2 weeks
**Dependencies**: Module 1 (Ghost System), Module 2 (Core Infrastructure), Module 3 (Embedding Pipeline), Module 4 (Knowledge Graph), Module 5 (UTL Integration), Module 6 (Bio-Nervous System), Module 9 (Dream Layer), Module 10 (Neuromodulation), Module 11 (Immune System)
**Last Updated**: 2025-12-31

---

## 1. Executive Summary

The Active Inference module implements the Free Energy Principle (FEP) for proactive knowledge acquisition in the Ultimate Context Graph. This module transforms the system from passive storage to active learning by enabling the system to identify knowledge gaps, compute expected information gain, and generate epistemic actions that minimize uncertainty. The implementation provides Expected Free Energy (EFE) computation, belief updating via variational inference, precision-weighted prediction errors, hierarchical generative models, policy selection with temporal depth planning, and epistemic action generation.

### 1.1 Core Objectives

- Implement Free Energy Principle with Expected Free Energy (EFE) minimization for action selection
- Compute epistemic value (information gain) and pragmatic value (goal achievement) for action prioritization
- Provide belief updating through variational inference with precision weighting
- Enable hierarchical generative models for multi-level prediction
- Support temporal depth planning up to 5 steps ahead
- Generate contextually relevant epistemic actions (clarifications, examples, hypotheses, experiments)

### 1.2 Key Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| EFE Computation Latency | <10ms per action | Single action EFE calculation |
| Belief Update Latency | <5ms | Variational inference step |
| Action Selection Latency | <20ms total | End-to-end action recommendation |
| Policy Evaluation Depth | 5 steps ahead | Maximum temporal planning horizon |
| Action Relevance Score | >70% helpful | User feedback on suggestions |
| Uncertainty Reduction | Measurable decrease | KL divergence pre/post interaction |
| Memory Overhead | <2MB per session | Active inference state tracking |
| Action Spam Prevention | <3 suggestions per response | Rate limiting on suggestions |

---

## 2. Theoretical Background

### 2.1 Free Energy Principle

The Free Energy Principle provides a unified framework for understanding perception, learning, and action in terms of minimizing surprise (or equivalently, maximizing model evidence). The system maintains a generative model of its environment and acts to minimize the divergence between its beliefs and observations.

#### Mathematical Formulation

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

Where:
- `G(pi)`: Expected Free Energy for policy pi
- `s'`: Future hidden states
- `o'`: Future observations
- First term: Epistemic value (negative expected ambiguity / information gain)
- Second term: Pragmatic value (expected divergence from prior preferences)

### 2.2 Epistemic vs Pragmatic Value

**Epistemic Value** quantifies the expected information gain from an action:
```
Epistemic = -E_q[H[p(o|s)]]
```
- Drives exploration and curiosity
- Seeks observations that maximally reduce uncertainty
- Prioritizes actions in high-entropy knowledge regions

**Pragmatic Value** quantifies goal-directed behavior:
```
Pragmatic = E_q[ln p(o|C)]
```
Where `C` represents prior preferences (goals).
- Drives exploitation and goal achievement
- Seeks observations aligned with desired outcomes
- Prioritizes actions that satisfy user intent

### 2.3 Integration with UTL Framework

The Active Inference module integrates with UTL (Unified Theory of Learning) through:

| UTL Parameter | Active Inference Mapping |
|---------------|-------------------------|
| Entropy (Delta_S) | Drives epistemic exploration when high |
| Coherence (Delta_C) | Informs pragmatic value computation |
| Learning Score (L) | Modulates action selection temperature |
| Johari Quadrant | Determines action type priority |

---

## 3. Functional Requirements

### 3.1 Active Inference Engine Core

#### REQ-ACTINF-001: ActiveInferenceEngine Struct Definition

**Priority**: Critical
**Description**: The system SHALL implement an ActiveInferenceEngine struct that coordinates all active inference computations.

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
    pub epistemic_pragmatic_balance: f32,  // Default: 0.5 (0 = pure pragmatic, 1 = pure epistemic)

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
```

**Acceptance Criteria**:
- [ ] ActiveInferenceEngine struct compiles with all components
- [ ] Configuration defaults match specification
- [ ] Thread-safe session beliefs via Arc<RwLock>
- [ ] Timeout enforcement on all computation paths
- [ ] Metrics collection enabled for monitoring
- [ ] Memory footprint under 2MB per session

---

#### REQ-ACTINF-002: Session Beliefs State Management

**Priority**: Critical
**Description**: The system SHALL maintain session-level belief states for variational inference.

```rust
/// Session-level belief state for variational inference
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
    /// `Constraint: Entropy_Computation < 1ms`
    pub fn entropy(&self) -> f32 {
        // For diagonal Gaussian: H = 0.5 * sum(log(2 * pi * e / precision))
        let n = self.precision.len() as f32;
        let log_precision_sum: f32 = self.precision.iter()
            .map(|&p| if p > 1e-10 { p.ln() } else { -23.0 }) // ln(1e-10) approx
            .sum();

        0.5 * (n * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln() - log_precision_sum)
    }

    /// Compute KL divergence from another distribution
    ///
    /// `Constraint: KL_Computation < 1ms`
    pub fn kl_divergence(&self, other: &BeliefDistribution) -> f32 {
        // KL(self || other) for diagonal Gaussians
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

/// Prior preferences representing goals
#[derive(Clone)]
pub struct GoalPreferences {
    /// Preferred observation embeddings (what we want to see)
    pub preferred_states: Vec<DVector<f32>>,

    /// Preference strengths for each goal
    pub preference_weights: Vec<f32>,

    /// Goal descriptions for action generation
    pub goal_descriptions: Vec<String>,
}

/// Buffer for precision-weighted prediction errors
#[derive(Clone)]
pub struct PredictionErrorBuffer {
    /// Recent prediction errors with timestamps
    pub errors: Vec<PredictionError>,

    /// Maximum buffer size
    pub max_size: usize,

    /// Running average error magnitude
    pub avg_error_magnitude: f32,
}

/// Single prediction error with precision weighting
#[derive(Clone)]
pub struct PredictionError {
    /// Prediction that was made
    pub prediction: DVector<f32>,

    /// Actual observation
    pub observation: DVector<f32>,

    /// Error vector (observation - prediction)
    pub error: DVector<f32>,

    /// Precision weight (confidence in this error signal)
    pub precision: f32,

    /// Timestamp of error
    pub timestamp: Instant,

    /// Domain of the error (for domain-specific learning)
    pub domain: String,
}

/// Observation from the environment
#[derive(Clone)]
pub struct Observation {
    /// Observation embedding
    pub embedding: DVector<f32>,

    /// Raw content (for action generation context)
    pub content: String,

    /// Observation timestamp
    pub timestamp: Instant,

    /// Source of observation
    pub source: ObservationSource,
}

/// Source types for observations
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObservationSource {
    /// User query input
    UserQuery,
    /// Retrieved memory
    MemoryRetrieval,
    /// External API response
    ExternalApi,
    /// System-generated (e.g., from dream layer)
    SystemGenerated,
}

/// Johari window quadrant (from UTL integration)
#[derive(Clone, Copy, Debug, PartialEq)]
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

**Acceptance Criteria**:
- [ ] SessionBeliefs struct maintains complete state for variational inference
- [ ] BeliefDistribution supports entropy and KL divergence computation
- [ ] Prediction error buffer with configurable size
- [ ] Goal preferences aligned with user intent
- [ ] Johari quadrant integration for action prioritization
- [ ] Memory-efficient representation (under 500KB per session)

---

### 3.1.1 Omnidirectional Inference Engine (Marblestone)

#### REQ-ACTINF-008: Omnidirectional Inference Support

**Priority**: Must
**Description**: The system SHALL support omnidirectional inference enabling forward, backward (abduction), bidirectional, and bridge inference directions.

```rust
// ============================================
// OMNIDIRECTIONAL INFERENCE ENGINE (Marblestone)
// ============================================

/// Direction of inference in the knowledge graph
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

/// Omnidirectional inference engine supporting all inference directions (Marblestone)
///
/// Extends standard forward inference with backward (abduction),
/// bidirectional, and bridge inference capabilities.
pub struct OmniInferenceEngine {
    /// Active inference engine for EFE computation
    active_inference: ActiveInferenceEngine,
    /// Clamped variables (fixed during inference)
    clamped_variables: HashMap<Uuid, ClampedValue>,
    /// Inference direction for current query
    direction: InferenceDirection,
    /// Maximum inference depth
    max_depth: u32,
}

/// A clamped (fixed) value during inference
pub struct ClampedValue {
    pub node_id: Uuid,
    pub value: f32,
    pub clamp_type: ClampType,
}

/// Type of value clamping
pub enum ClampType {
    /// Hard clamp: value cannot change
    Hard,
    /// Soft clamp: value has high prior but can change
    Soft { prior_strength: f32 },
}

impl OmniInferenceEngine {
    /// Perform omnidirectional inference (Marblestone)
    ///
    /// # Arguments
    /// * `query_nodes` - Starting nodes for inference
    /// * `direction` - Inference direction
    /// * `clamped` - Variables to hold fixed
    ///
    /// # Returns
    /// Inference results with updated beliefs
    pub async fn omni_infer(
        &self,
        query_nodes: Vec<Uuid>,
        direction: InferenceDirection,
        clamped: &[ClampedValue],
    ) -> Result<InferenceResult, InferenceError> {
        // Set up clamped variables
        for cv in clamped {
            self.clamped_variables.insert(cv.node_id, cv.clone());
        }

        match direction {
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
        }
    }

    /// Backward inference: given effects, infer likely causes
    async fn backward_inference(&self, effects: &[Uuid]) -> Result<InferenceResult, InferenceError> {
        // Traverse causal edges in reverse
        // Use EFE to find most likely causes
        let mut candidates = Vec::new();

        for effect_id in effects {
            let incoming = self.get_incoming_causal_edges(*effect_id).await?;
            for edge in incoming {
                let cause_belief = self.compute_cause_belief(&edge, *effect_id);
                candidates.push((edge.source, cause_belief));
            }
        }

        // Rank by belief strength
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(InferenceResult {
            inferred_nodes: candidates.iter().map(|(id, _)| *id).collect(),
            beliefs: candidates.into_iter().collect(),
            direction: InferenceDirection::Backward,
        })
    }

    /// Abductive inference: find best explanation for observations
    async fn abductive_inference(&self, observations: &[Uuid]) -> Result<InferenceResult, InferenceError> {
        // Find hypotheses that best explain all observations
        // Minimize free energy of explanation
        todo!("Implement abductive inference")
    }
}

/// Result of omnidirectional inference
pub struct InferenceResult {
    pub inferred_nodes: Vec<Uuid>,
    pub beliefs: HashMap<Uuid, f32>,
    pub direction: InferenceDirection,
}
```

**Acceptance Criteria**:
- [ ] OmniInferenceEngine struct compiles with all components
- [ ] All five inference directions supported (Forward, Backward, Bidirectional, Bridge, Abduction)
- [ ] Clamped variables properly constrain inference
- [ ] Backward inference correctly identifies likely causes from effects
- [ ] Integration with existing ActiveInferenceEngine for EFE computation
- [ ] Maximum inference depth configurable

---

#### REQ-ACTINF-009: Backward Inference for Causal Reasoning

**Priority**: Must
**Description**: Backward inference SHALL identify likely causes from observed effects using abductive reasoning.

**Implementation Notes**:
- Traverse causal edges in reverse direction
- Use EFE to score candidate causes
- Support both hard and soft clamping of observed variables
- Rank causes by belief strength

---

#### REQ-ACTINF-010: Clamped Variable Support

**Priority**: Must
**Description**: Inference SHALL support clamped variables that remain fixed or strongly biased during inference.

**Clamp Types**:
- **Hard clamp**: Value cannot change during inference (fixed observation)
- **Soft clamp**: Value has high prior but can change (strong evidence)

---

### 3.2 Generative Model

#### REQ-ACTINF-003: Hierarchical Generative Model

**Priority**: Critical
**Description**: The system SHALL implement a hierarchical generative model for multi-level predictions.

```rust
/// Hierarchical generative model implementing p(o, s) = p(o|s) * p(s)
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

/// Connection between generative layers
#[derive(Clone)]
pub struct LayerConnection {
    /// Source layer index
    pub from_layer: usize,

    /// Target layer index
    pub to_layer: usize,

    /// Connection weight matrix
    pub weights: DMatrix<f32>,

    /// Connection precision
    pub precision: f32,
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

impl GenerativeModel {
    /// Generate prediction for given beliefs
    ///
    /// `Constraint: Prediction_Latency < 2ms`
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
    pub fn compute_prediction_error(
        &self,
        predicted: &DVector<f32>,
        observed: &DVector<f32>,
    ) -> PredictionError {
        let error = observed - predicted;
        let precision = self.likelihood.obs_precision;

        // Precision-weighted error magnitude
        let _weighted_magnitude: f32 = error.iter()
            .map(|e| e * e * precision)
            .sum::<f32>()
            .sqrt();

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
                    *val += weighted_error[i] * 0.01; // Small update step
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
```

**Acceptance Criteria**:
- [ ] Hierarchical model with 5 layers matching bio-nervous system
- [ ] Top-down prediction generation under 2ms
- [ ] Precision-weighted prediction error computation under 1ms
- [ ] Model learning from prediction errors under 3ms
- [ ] Layer dimensions match embedding pipeline outputs
- [ ] Cross-layer message passing implemented

---

### 3.3 Belief Updater

#### REQ-ACTINF-004: Variational Inference Belief Updating

**Priority**: Critical
**Description**: The system SHALL implement belief updating via variational inference with precision weighting.

```rust
/// Belief updater implementing variational inference for approximate posterior computation.
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

/// Metrics for single iteration of belief update
#[derive(Clone)]
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
                    -e * generative_model.likelihood.obs_precision + pe * pp.max(self.config.precision_floor) / bp.max(self.config.precision_floor)
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
            let free_energy = self.compute_free_energy(beliefs, &observation.embedding, generative_model);

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
#[derive(Clone)]
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
```

**Acceptance Criteria**:
- [ ] Variational inference with gradient descent
- [ ] Belief update latency under 5ms
- [ ] Convergence detection via KL divergence threshold
- [ ] Precision updating based on prediction error
- [ ] Momentum-based optimization for stability
- [ ] Gradient clipping for numerical stability

---

### 3.4 Policy Evaluator

#### REQ-ACTINF-005: Policy Evaluation with Temporal Planning

**Priority**: Critical
**Description**: The system SHALL implement policy evaluation with Expected Free Energy computation and temporal depth planning.

```rust
/// Evaluates policies based on Expected Free Energy (EFE) with temporal planning.
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

/// Template for a policy (action sequence)
#[derive(Clone)]
pub struct PolicyTemplate {
    /// Policy identifier
    pub id: String,

    /// Action sequence
    pub actions: Vec<EpistemicActionType>,

    /// Prior probability of this policy
    pub prior_prob: f32,

    /// Policy description
    pub description: String,
}

/// Key for EFE cache lookup
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct PolicyCacheKey {
    /// Policy ID
    pub policy_id: String,

    /// Belief state hash
    pub belief_hash: u64,

    /// Depth level
    pub depth: usize,
}

/// Metrics for policy evaluation
#[derive(Clone, Default)]
pub struct PolicyEvaluationMetrics {
    /// Total policies evaluated
    pub policies_evaluated: u64,

    /// Cache hits
    pub cache_hits: u64,

    /// Average EFE computation time (microseconds)
    pub avg_efe_time_us: f64,

    /// Policies pruned
    pub policies_pruned: u64,
}

impl PolicyEvaluator {
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
                probability: 0.0, // Computed after softmax
            });

            self.metrics.policies_evaluated += 1;
        }

        // Compute softmax probabilities
        self.apply_softmax_probabilities(&mut evaluations);

        // Sort by EFE (lower is better)
        evaluations.sort_by(|a, b| a.efe.partial_cmp(&b.efe).unwrap_or(std::cmp::Ordering::Equal));

        // Update timing metrics
        let elapsed = start.elapsed().as_micros() as f64;
        self.metrics.avg_efe_time_us =
            (self.metrics.avg_efe_time_us * 0.9) + (elapsed / evaluations.len() as f64 * 0.1);

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
            // Simplified: use mean-field approximation for future beliefs
            self.config.discount_factor * current_efe
        } else {
            0.0
        };

        current_efe + future_efe
    }

    /// Compute epistemic value (information gain)
    ///
    /// Epistemic = -E_q[H[p(o|s)]] (negative expected ambiguity)
    fn compute_epistemic_value(
        &self,
        beliefs: &BeliefDistribution,
        model: &GenerativeModel,
    ) -> f32 {
        // Expected ambiguity: H[p(o|s)] under q(s)
        // Approximated as inverse of prediction precision
        let prediction = model.predict(beliefs);
        let ambiguity = 1.0 / (model.likelihood.obs_precision + 1e-6);

        // Information gain is reduction in uncertainty
        let info_gain = beliefs.entropy() * ambiguity;

        -info_gain // Negative because we want to minimize EFE
    }

    /// Compute pragmatic value (goal satisfaction)
    ///
    /// Pragmatic = E_q[ln p(o|C)] (expected log preference)
    fn compute_pragmatic_value(
        &self,
        beliefs: &BeliefDistribution,
        goals: &GoalPreferences,
    ) -> f32 {
        if goals.preferred_states.is_empty() {
            return 0.0;
        }

        // Compute alignment with goal preferences
        let mut total_preference = 0.0;
        let mut total_weight = 0.0;

        for (preferred, weight) in goals.preferred_states.iter().zip(goals.preference_weights.iter()) {
            // Cosine similarity with goal state
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

        // Compute softmax denominator
        let max_efe = evaluations.iter()
            .map(|e| e.efe)
            .fold(f32::NEG_INFINITY, f32::max);

        let exp_sum: f32 = evaluations.iter()
            .map(|e| (-(e.efe - max_efe) / temp).exp())
            .sum();

        // Assign probabilities
        for eval in evaluations.iter_mut() {
            eval.probability = (-(eval.efe - max_efe) / temp).exp() / exp_sum;
        }
    }

    /// Hash beliefs for cache key
    fn hash_beliefs(&self, beliefs: &BeliefDistribution) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash first few components for efficiency
        for &val in beliefs.mean.iter().take(16) {
            let bits = val.to_bits();
            bits.hash(&mut hasher);
        }

        hasher.finish()
    }
}

/// Result of policy evaluation
#[derive(Clone)]
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

**Acceptance Criteria**:
- [ ] EFE computation per policy under 2ms
- [ ] Total policy evaluation under 10ms
- [ ] Planning depth of 5 steps supported
- [ ] Epistemic and pragmatic value decomposition
- [ ] Softmax probability assignment
- [ ] EFE caching for repeated evaluations

---

### 3.5 Epistemic Action Generator

#### REQ-ACTINF-006: Epistemic Action Generation

**Priority**: Critical
**Description**: The system SHALL generate contextually relevant epistemic actions based on uncertainty analysis.

```rust
/// Generates epistemic actions to reduce uncertainty and acquire information.
///
/// Produces clarification questions, example requests, hypothesis proposals,
/// and experiment suggestions based on current belief state and knowledge gaps.
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

/// Configuration for epistemic action generation
#[derive(Clone, Debug)]
pub struct EpistemicActionConfig {
    /// Minimum uncertainty to trigger action generation
    pub uncertainty_threshold: f32,  // Default: 0.6

    /// Maximum actions per response
    pub max_actions: usize,  // Default: 3

    /// Cooldown between similar actions (seconds)
    pub action_cooldown: u64,  // Default: 60

    /// Minimum expected information gain
    pub min_info_gain: f32,  // Default: 0.1

    /// Action diversity requirement
    pub require_diversity: bool,  // Default: true
}

impl Default for EpistemicActionConfig {
    fn default() -> Self {
        Self {
            uncertainty_threshold: 0.6,
            max_actions: 3,
            action_cooldown: 60,
            min_info_gain: 0.1,
            require_diversity: true,
        }
    }
}

/// Types of epistemic actions
#[derive(Clone, Debug, PartialEq)]
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
#[derive(Clone)]
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

/// Identified knowledge gap
#[derive(Clone)]
pub struct KnowledgeGap {
    /// Gap identifier
    pub id: Uuid,

    /// Description of the gap
    pub description: String,

    /// Domain of the gap
    pub domain: String,

    /// Severity (uncertainty level)
    pub severity: f32,

    /// Related node IDs in graph
    pub related_nodes: Vec<Uuid>,
}

/// Template for action generation
#[derive(Clone)]
pub struct ActionTemplate {
    /// Action type this template produces
    pub action_type: EpistemicActionType,

    /// Template pattern with placeholders
    pub pattern: String,

    /// Required context fields
    pub required_context: Vec<String>,

    /// Applicability condition
    pub condition: ActionCondition,
}

/// Condition for when an action template applies
#[derive(Clone)]
pub enum ActionCondition {
    /// High uncertainty in specific domain
    HighDomainUncertainty(f32),
    /// Low coherence in beliefs
    LowCoherence(f32),
    /// Johari quadrant match
    JohariQuadrant(JohariQuadrant),
    /// Prediction error above threshold
    HighPredictionError(f32),
    /// Always applicable
    Always,
}

/// Rate limiter to prevent action spam
#[derive(Clone)]
pub struct ActionRateLimiter {
    /// Recent actions by type
    pub recent_actions: HashMap<EpistemicActionType, Vec<Instant>>,

    /// Cooldown period
    pub cooldown: Duration,

    /// Maximum per type per session
    pub max_per_type: usize,
}

impl ActionRateLimiter {
    /// Check if action type is allowed
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

    /// Record action generation
    pub fn record_action(&mut self, action_type: EpistemicActionType) {
        self.recent_actions
            .entry(action_type)
            .or_insert_with(Vec::new)
            .push(Instant::now());
    }
}

/// Metrics for action generation
#[derive(Clone, Default)]
pub struct ActionGenerationMetrics {
    /// Total actions generated
    pub total_generated: u64,

    /// Actions by type
    pub by_type: HashMap<String, u64>,

    /// Actions rated helpful
    pub rated_helpful: u64,

    /// Average relevance score
    pub avg_relevance: f64,

    /// Actions suppressed by rate limiter
    pub rate_limited: u64,
}

impl EpistemicActionGenerator {
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
                // Check minimum info gain
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

        let _elapsed = start.elapsed();
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

        // Check for prediction errors indicating gaps
        for error in &beliefs.prediction_errors.errors {
            let error_magnitude: f32 = error.error.iter().map(|e| e.abs()).sum::<f32>()
                / error.error.len() as f32;

            if error_magnitude > 0.5 {
                gaps.push(KnowledgeGap {
                    id: Uuid::new_v4(),
                    description: format!("High prediction error in {}", error.domain),
                    domain: error.domain.clone(),
                    severity: error_magnitude,
                    related_nodes: Vec::new(),
                });
            }
        }

        // Use Johari quadrant to identify exploration opportunities
        match beliefs.johari_quadrant {
            JohariQuadrant::Blind => {
                gaps.push(KnowledgeGap {
                    id: Uuid::new_v4(),
                    description: "Blind spot detected - external knowledge may help".to_string(),
                    domain: context.current_domain.clone().unwrap_or_default(),
                    severity: 0.8,
                    related_nodes: Vec::new(),
                });
            }
            JohariQuadrant::Unknown => {
                gaps.push(KnowledgeGap {
                    id: Uuid::new_v4(),
                    description: "Unknown territory - exploration recommended".to_string(),
                    domain: context.current_domain.clone().unwrap_or_default(),
                    severity: 0.9,
                    related_nodes: Vec::new(),
                });
            }
            _ => {}
        }

        // Sort by severity
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
        // Select action type based on gap characteristics
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

        // Generate content based on action type
        let content = self.generate_action_content(&action_type, gap, context);

        // Calculate expected information gain
        let expected_info_gain = gap.severity * beliefs.belief_entropy;

        // Calculate priority
        let priority = expected_info_gain * (1.0 + context.urgency);

        // Calculate relevance
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

    /// Generate action content from template
    fn generate_action_content(
        &self,
        action_type: &EpistemicActionType,
        gap: &KnowledgeGap,
        _context: &ActionContext,
    ) -> String {
        match action_type {
            EpistemicActionType::SeekClarification => {
                format!(
                    "Could you clarify what you mean by '{}' in the context of {}?",
                    gap.domain,
                    gap.description
                )
            }
            EpistemicActionType::RequestExample => {
                format!(
                    "Could you provide a concrete example of {} to help clarify?",
                    gap.domain
                )
            }
            EpistemicActionType::ProposeHypothesis => {
                format!(
                    "Based on current context, I hypothesize that {}. Is this correct?",
                    gap.description
                )
            }
            EpistemicActionType::SuggestExperiment => {
                format!(
                    "To better understand {}, we could try: [specific test or exploration]",
                    gap.domain
                )
            }
            EpistemicActionType::ExploreRelated => {
                format!(
                    "Should we explore related concepts to {}?",
                    gap.domain
                )
            }
            EpistemicActionType::ConfirmUnderstanding => {
                format!(
                    "Let me confirm my understanding of {}: [current understanding]. Is this accurate?",
                    gap.domain
                )
            }
        }
    }

    /// Ensure action diversity
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
}

/// Context for action generation
#[derive(Clone)]
pub struct ActionContext {
    /// Current conversation/query domain
    pub current_domain: Option<String>,

    /// Recent user queries
    pub recent_queries: Vec<String>,

    /// Urgency level [0, 1]
    pub urgency: f32,

    /// Session ID
    pub session_id: Uuid,
}
```

**Acceptance Criteria**:
- [ ] Six epistemic action types implemented
- [ ] Action generation under 5ms
- [ ] Rate limiting prevents spam (<3 per response)
- [ ] Expected information gain calculation
- [ ] Johari quadrant-aware action selection
- [ ] Action diversity enforcement
- [ ] Knowledge gap identification from belief state

---

### 3.6 MCP Integration

#### REQ-ACTINF-007: MCP Tool Integration

**Priority**: High
**Description**: The system SHALL integrate Active Inference with the MCP tool `epistemic_action`.

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
    /// Action type
    pub action_type: String,

    /// Action content (question, hypothesis, etc.)
    pub content: String,

    /// Target concept
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_concept: Option<String>,

    /// Expected information gain [0, 1]
    pub expected_info_gain: f32,

    /// Relevance to current context [0, 1]
    pub relevance: f32,

    /// Priority ranking
    pub priority: u32,
}

/// UTL metrics for response
#[derive(Clone, Debug, Serialize)]
pub struct UtlMetricsOutput {
    /// Current entropy
    pub entropy: f32,

    /// Current coherence
    pub coherence: f32,

    /// Learning score
    pub learning_score: f32,
}

/// Cognitive pulse header (included in all responses)
#[derive(Clone, Debug, Serialize)]
pub struct CognitivePulse {
    /// System entropy
    pub entropy: f32,

    /// System coherence
    pub coherence: f32,

    /// Suggested next action
    pub suggested_action: String,
}

/// Handler for epistemic_action MCP tool
pub struct EpistemicActionHandler {
    /// Active inference engine reference
    pub engine: Arc<RwLock<ActiveInferenceEngine>>,

    /// Session manager reference
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

        // Determine suggested action
        let suggested_action = if beliefs.belief_entropy > 0.7 {
            if beliefs.state_beliefs.confidence < 0.4 {
                "trigger_dream"
            } else {
                "explore"
            }
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

/// Session manager stub (implemented in core module)
pub struct SessionManager {
    sessions: HashMap<Uuid, Session>,
}

impl SessionManager {
    pub fn get(&self, id: &Uuid) -> Option<&Session> {
        self.sessions.get(id)
    }
}

/// Session stub
pub struct Session {
    pub beliefs: Arc<RwLock<SessionBeliefs>>,
    pub current_domain: Option<String>,
    pub recent_queries: Vec<String>,
}
```

**Acceptance Criteria**:
- [ ] MCP handler under 25ms total latency
- [ ] Proper JSON serialization/deserialization
- [ ] Cognitive pulse header in all responses
- [ ] Force flag bypasses uncertainty threshold
- [ ] Relevance filtering applied
- [ ] Graceful error handling for missing sessions

---

## 4. Error Handling

### 4.1 Error Types

```rust
use thiserror::Error;

/// Errors for Active Inference module
#[derive(Debug, Error)]
pub enum ActiveInferenceError {
    /// Session not found
    #[error("Session not found: {0}")]
    SessionNotFound(Uuid),

    /// Belief update failed
    #[error("Belief update failed: {0}")]
    BeliefUpdateFailed(String),

    /// EFE computation timeout
    #[error("EFE computation timed out after {0}ms")]
    EfeTimeout(u64),

    /// Invalid generative model
    #[error("Invalid generative model: {0}")]
    InvalidModel(String),

    /// Policy evaluation failed
    #[error("Policy evaluation failed: {0}")]
    PolicyEvaluationFailed(String),

    /// Action generation failed
    #[error("Action generation failed: {0}")]
    ActionGenerationFailed(String),

    /// Numerical instability detected
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl ActiveInferenceError {
    /// Get MCP error code
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
```

**Acceptance Criteria**:
- [ ] All error types defined with descriptive messages
- [ ] MCP error codes assigned
- [ ] Error logging before propagation
- [ ] Graceful recovery where possible

---

## 5. Performance Requirements

### 5.1 Latency Budgets

| Operation | Budget | Notes |
|-----------|--------|-------|
| EFE Computation (single) | <10ms | Core computational bottleneck |
| Belief Update | <5ms | Variational inference step |
| Policy Evaluation (all) | <10ms | Tree search with pruning |
| Action Generation | <5ms | Template-based with filtering |
| Total Action Selection | <20ms | End-to-end critical path |
| MCP Handler | <25ms | Including serialization |

### 5.2 Memory Constraints

| Component | Budget | Notes |
|-----------|--------|-------|
| Session Beliefs | <500KB | Per session state |
| Generative Model | <10MB | Shared across sessions |
| Policy Cache | <1MB | LRU eviction |
| Action History | <100KB | Per session rate limiting |
| Total per Session | <2MB | Including all buffers |

### 5.3 Throughput Requirements

| Metric | Target |
|--------|--------|
| Concurrent Sessions | >100 |
| Actions per Second | >1000 |
| Belief Updates per Second | >500 |
| EFE Computations per Second | >10000 |

---

## 6. Dependencies

### 6.1 Internal Module Dependencies

| Dependency | Purpose | Interface |
|------------|---------|-----------|
| Module 2: Core Infrastructure | MemoryNode, Uuid types | `context-graph-core` |
| Module 3: Embedding Pipeline | 1536D embeddings | `EmbeddingProvider` trait |
| Module 5: UTL Integration | Learning score, entropy | `UTLProcessor` trait |
| Module 6: Bio-Nervous System | Layer communication | `NervousLayer` trait |
| Module 10: Neuromodulation | Parameter modulation | `NeuromodulationController` |
| Module 11: Immune System | Threat detection | `ThreatDetector` |

### 6.2 External Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| nalgebra | 0.32+ | Matrix/vector operations |
| ndarray | 0.15+ | N-dimensional arrays (optional) |
| tokio | 1.35+ | Async runtime |
| uuid | 1.6+ | Session identifiers |
| serde | 1.0+ | Serialization |
| thiserror | 1.0+ | Error handling |
| lru | 0.12+ | EFE caching |

---

## 7. Traceability Matrix

### 7.1 Requirements to PRD

| Requirement | PRD Section | Description |
|-------------|-------------|-------------|
| REQ-ACTINF-001 | 5.2 (Core Tools) | ActiveInferenceEngine for epistemic_action |
| REQ-ACTINF-002 | 2.1 (UTL Core) | Session beliefs aligned with UTL state |
| REQ-ACTINF-003 | 2.3 (5-Layer System) | Hierarchical model matches nervous layers |
| REQ-ACTINF-004 | 2.1 (UTL Formula) | Belief updating with precision weighting |
| REQ-ACTINF-005 | 6.1 (inject_context) | Policy evaluation for action selection |
| REQ-ACTINF-006 | 5.2 (epistemic_action) | Action generation with rate limiting |
| REQ-ACTINF-007 | 5.1 (Protocol) | MCP JSON-RPC integration |
| REQ-ACTINF-008 | 2.1 (Inference) | OmniInferenceEngine for multi-direction inference (Marblestone) |
| REQ-ACTINF-009 | 2.1 (Inference) | Backward inference for causal reasoning (abduction) |
| REQ-ACTINF-010 | 2.1 (Inference) | Clamped variable support for constrained inference |

### 7.2 Requirements to Implementation Plan

| Requirement | Implementation Plan Section |
|-------------|----------------------------|
| REQ-ACTINF-001 | Module 12: EpistemicActionGenerator |
| REQ-ACTINF-002 | Module 12: Session state management |
| REQ-ACTINF-003 | Module 12: GenerativeModel |
| REQ-ACTINF-004 | Module 12: BeliefUpdater |
| REQ-ACTINF-005 | Module 12: PolicyEvaluator |
| REQ-ACTINF-006 | Module 12: Action templates |
| REQ-ACTINF-007 | Module 12: MCP integration |
| REQ-ACTINF-008 | Module 12: OmniInferenceEngine (Marblestone) |
| REQ-ACTINF-009 | Module 12: Backward inference implementation |
| REQ-ACTINF-010 | Module 12: Clamped variable system |

### 7.3 Requirements to Constitution

| Requirement | Constitution Section |
|-------------|---------------------|
| REQ-ACTINF-001 | performance_budgets.latency |
| REQ-ACTINF-002 | utl_constraints.parameters |
| REQ-ACTINF-003 | bio_nervous_system.layers |
| REQ-ACTINF-004 | utl_constraints.formula |
| REQ-ACTINF-005 | performance_budgets.quality |
| REQ-ACTINF-006 | mcp_protocol.required_response_header |
| REQ-ACTINF-007 | mcp_protocol.error_codes |
| REQ-ACTINF-008 | inference_engine.omnidirectional |
| REQ-ACTINF-009 | inference_engine.abductive_reasoning |
| REQ-ACTINF-010 | inference_engine.clamped_variables |

---

## 8. Verification Methods

### 8.1 Unit Tests

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| test_efe_computation | EFE formula correctness | Matches reference implementation |
| test_belief_update | Variational inference | Convergence within 10 iterations |
| test_policy_ranking | Policy ordering by EFE | Lower EFE = higher rank |
| test_action_generation | Action template instantiation | Valid action content |
| test_rate_limiting | Action spam prevention | <3 per response |
| test_numerical_stability | Precision floor enforcement | No NaN/Inf values |
| test_omni_forward_inference | Forward inference direction | Correct effect prediction |
| test_omni_backward_inference | Backward inference (abduction) | Identifies likely causes |
| test_omni_bidirectional | Bidirectional inference | Both directions computed |
| test_omni_bridge_inference | Bridge across domains | Cross-domain inference works |
| test_clamped_hard | Hard clamped variables | Values unchanged during inference |
| test_clamped_soft | Soft clamped variables | Values biased but can change |

### 8.2 Integration Tests

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| test_mcp_handler | End-to-end MCP flow | Response under 25ms |
| test_session_lifecycle | Belief state persistence | State preserved across calls |
| test_utl_integration | UTL metric propagation | Entropy/coherence in response |
| test_neuromod_integration | Parameter modulation | Learning rate adjusted |
| test_omni_inference_integration | OmniInferenceEngine with graph | All directions work with knowledge graph |

### 8.3 Performance Benchmarks

| Benchmark | Description | Target |
|-----------|-------------|--------|
| bench_efe_throughput | EFE computations/sec | >10000 |
| bench_belief_update | Update latency P99 | <5ms |
| bench_action_generation | Generation latency P99 | <5ms |
| bench_concurrent_sessions | Session scalability | >100 concurrent |
| bench_omni_inference | Omnidirectional inference latency P99 | <15ms |
| bench_backward_inference | Backward (abduction) inference | <10ms per query |

---

## 9. Quality Gates

| Gate | Criteria | Verification |
|------|----------|--------------|
| Unit Test Coverage | >90% | cargo tarpaulin |
| Integration Tests | All pass | cargo test --test integration |
| Performance Targets | All met | cargo bench |
| Action Helpfulness | >70% | User feedback tracking |
| Memory Usage | <2MB/session | Memory profiling |
| No Panics | Zero unwrap() | clippy lint |
| Documentation | All public APIs | cargo doc |

---

## 10. Appendix

### 10.1 Mathematical Reference

**Variational Free Energy Decomposition**:
```
F = E_q[ln q(s)] - E_q[ln p(o, s)]
  = -H[q(s)] - E_q[ln p(o, s)]
  = -H[q(s)] - E_q[ln p(o|s)] - E_q[ln p(s)]
  = D_KL[q(s) || p(s)] - E_q[ln p(o|s)]
  = Complexity - Accuracy
```

**Expected Free Energy Decomposition**:
```
G(pi) = E_q[ln q(s') - ln p(o', s')]
      = E_q[ln q(s')] - E_q[ln p(o'|s')] - E_q[ln p(s')]
      = E_q[H[p(o|s)]] + D_KL[q(s') || p(s')]
      = Expected Ambiguity + Expected Divergence
      = Epistemic + Pragmatic (after negation)
```

### 10.2 References

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press.
- Da Costa, L., et al. (2020). Active inference on discrete state-spaces: A synthesis.

---

*Document Version: 1.0.0*
*Generated: 2025-12-31*
*Based on PRD v2.0.0 and Implementation Plan Module 12*
