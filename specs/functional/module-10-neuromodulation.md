# Module 10: Neuromodulation - Functional Specification

**Version**: 1.0.0
**Status**: Draft
**Phase**: 9
**Duration**: 3 weeks
**Dependencies**: Module 9 (Dream Layer), Module 5 (UTL Integration), Module 6 (Bio-Nervous System)
**Last Updated**: 2025-12-31

---

## 1. Executive Summary

The Neuromodulation module implements a biologically-inspired parameter modulation system that dynamically adjusts system behavior based on cognitive state. Four virtual neuromodulators (Dopamine, Serotonin, Noradrenaline, Acetylcholine) map to core system parameters, enabling adaptive exploration-exploitation tradeoffs, learning rate adjustment, and retrieval sharpness control. This module transforms the system from static parameter tuning to dynamic, context-aware adaptation.

### 1.1 Core Objectives

- Implement 4-channel neuromodulation controller with smooth parameter transitions
- Map neuromodulators to system parameters (Hopfield beta, FuseMoE top_k, attention temperature, UTL learning rate)
- Achieve <200 microsecond update latency per query
- Prevent oscillation through hysteresis mechanisms
- Integrate with UTL state for trigger-based modulation

### 1.2 Key Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Update Latency | <200 microseconds | Per-query timing instrumentation |
| Parameter Stability | <5% oscillation | Variance over 100-query window |
| Trigger Response Time | <50 microseconds | Time from condition to modulation start |
| Memory Overhead | <1KB per session | Session state size |
| Interpolation Smoothness | <0.05 delta per step | Maximum parameter change rate |

---

## 2. Functional Requirements

### 2.1 Neuromodulation Core Structure

#### REQ-NEURO-001: NeuromodulationController Struct Definition

**Priority**: Critical
**Description**: The system SHALL implement a NeuromodulationController struct that manages all neuromodulator channels.

```rust
/// Controls 4 neuromodulator channels that dynamically adjust system parameters
/// based on cognitive state and environmental triggers.
///
/// `Constraint: Update_Latency < 200us`
pub struct NeuromodulationController {
    /// Dopamine level [0.0, 1.0] - maps to retrieval sharpness
    pub dopamine: f32,

    /// Serotonin level [0.0, 1.0] - maps to exploration breadth
    pub serotonin: f32,

    /// Noradrenaline level [0.0, 1.0] - maps to attention distribution
    pub noradrenaline: f32,

    /// Acetylcholine level [0.0, 1.0] - maps to learning speed
    pub acetylcholine: f32,

    /// Rate at which modulators decay/grow toward baseline
    pub update_rate: f32,  // Default: 0.1

    /// Baseline levels for homeostatic return
    baseline: NeuromodulatorBaseline,

    /// Current trigger state
    trigger_state: TriggerState,

    /// Hysteresis configuration to prevent oscillation
    hysteresis: HysteresisConfig,

    /// Parameter mapping configuration
    parameter_map: ParameterMapping,

    /// Metrics for monitoring
    metrics: NeuromodulationMetrics,
}

/// Baseline levels modulators return to over time
#[derive(Clone, Copy)]
pub struct NeuromodulatorBaseline {
    pub dopamine: f32,       // Default: 0.5
    pub serotonin: f32,      // Default: 0.5
    pub noradrenaline: f32,  // Default: 0.3
    pub acetylcholine: f32,  // Default: 0.4
}

impl Default for NeuromodulatorBaseline {
    fn default() -> Self {
        Self {
            dopamine: 0.5,
            serotonin: 0.5,
            noradrenaline: 0.3,
            acetylcholine: 0.4,
        }
    }
}
```

**Acceptance Criteria**:
- [ ] NeuromodulationController struct compiles with all fields
- [ ] All modulator values constrained to [0.0, 1.0] range
- [ ] Default baseline values match specification
- [ ] Thread-safe access pattern implemented
- [ ] Memory footprint under 1KB per instance

---

#### REQ-NEURO-002: Neuromodulator Channel Definition

**Priority**: Critical
**Description**: The system SHALL define individual neuromodulator channels with their biological mappings and system effects.

```rust
/// Individual neuromodulator channel with biological context
pub struct NeuromodulatorChannel {
    /// Current normalized level [0.0, 1.0]
    pub level: f32,

    /// Biological function description
    pub biological_role: &'static str,

    /// System parameter this modulator controls
    pub system_parameter: SystemParameter,

    /// Output range for parameter mapping
    pub output_range: (f32, f32),

    /// Effect description
    pub effect_description: &'static str,

    /// Current mapped value
    pub mapped_value: f32,
}

/// System parameters controlled by neuromodulators
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SystemParameter {
    /// Hopfield network inverse temperature
    HopfieldBeta,

    /// FuseMoE expert selection count
    FuseMoeTopK,

    /// Attention mechanism temperature
    AttentionTemperature,

    /// UTL optimizer learning rate
    UtlLearningRate,
}

/// Channel definitions matching constitution.yaml
pub const NEUROMODULATOR_CHANNELS: [NeuromodulatorChannel; 4] = [
    NeuromodulatorChannel {
        level: 0.5,
        biological_role: "Reward prediction error",
        system_parameter: SystemParameter::HopfieldBeta,
        output_range: (1.0, 5.0),
        effect_description: "High = sharp retrieval (exploitation)",
        mapped_value: 3.0,
    },
    NeuromodulatorChannel {
        level: 0.5,
        biological_role: "Temporal discounting",
        system_parameter: SystemParameter::FuseMoeTopK,
        output_range: (2.0, 8.0),
        effect_description: "High = more experts (exploration)",
        mapped_value: 5.0,
    },
    NeuromodulatorChannel {
        level: 0.3,
        biological_role: "Arousal/Surprise",
        system_parameter: SystemParameter::AttentionTemperature,
        output_range: (0.5, 2.0),
        effect_description: "High = flat attention (exploration)",
        mapped_value: 0.95,
    },
    NeuromodulatorChannel {
        level: 0.4,
        biological_role: "Learning rate modulation",
        system_parameter: SystemParameter::UtlLearningRate,
        output_range: (0.001, 0.002),
        effect_description: "High = faster memory update",
        mapped_value: 0.0014,
    },
];
```

**Acceptance Criteria**:
- [ ] All 4 channels defined with correct parameters
- [ ] Output ranges match constitution.yaml exactly
- [ ] Biological roles documented
- [ ] Effect descriptions clear and accurate
- [ ] Enum variants for all system parameters

---

### 2.2 Parameter Mapping Requirements

#### REQ-NEURO-003: Dopamine to Hopfield Beta Mapping

**Priority**: Critical
**Description**: The system SHALL map dopamine levels to Hopfield network inverse temperature (beta).

```rust
pub struct DopamineMapper {
    /// Input range for dopamine level
    pub input_range: (f32, f32),   // (0.0, 1.0)

    /// Output range for hopfield.beta
    pub output_range: (f32, f32),  // (1.0, 5.0)

    /// Mapping curve type
    pub curve: MappingCurve,
}

impl DopamineMapper {
    /// Map dopamine level [0,1] to hopfield.beta [1.0, 5.0]
    ///
    /// High dopamine -> High beta -> Sharp retrieval (exploitation)
    /// Low dopamine -> Low beta -> Diffuse retrieval (exploration)
    ///
    /// `Constraint: Mapping_Latency < 10us`
    pub fn map(&self, dopamine: f32) -> f32 {
        let clamped = dopamine.clamp(0.0, 1.0);

        match self.curve {
            MappingCurve::Linear => {
                self.output_range.0 + clamped * (self.output_range.1 - self.output_range.0)
            }
            MappingCurve::Sigmoid { steepness } => {
                let sigmoid = 1.0 / (1.0 + (-steepness * (clamped - 0.5)).exp());
                self.output_range.0 + sigmoid * (self.output_range.1 - self.output_range.0)
            }
            MappingCurve::Exponential { base } => {
                let exp = base.powf(clamped);
                let normalized = (exp - 1.0) / (base - 1.0);
                self.output_range.0 + normalized * (self.output_range.1 - self.output_range.0)
            }
        }
    }

    /// Compute effect on retrieval behavior
    pub fn describe_effect(&self, beta: f32) -> RetrievalEffect {
        if beta > 4.0 {
            RetrievalEffect::Sharp { selectivity: 0.9 }
        } else if beta > 2.5 {
            RetrievalEffect::Balanced { selectivity: 0.6 }
        } else {
            RetrievalEffect::Diffuse { selectivity: 0.3 }
        }
    }
}

pub enum RetrievalEffect {
    Sharp { selectivity: f32 },
    Balanced { selectivity: f32 },
    Diffuse { selectivity: f32 },
}
```

**Acceptance Criteria**:
- [ ] Dopamine 0.0 maps to beta 1.0
- [ ] Dopamine 1.0 maps to beta 5.0
- [ ] Linear interpolation by default
- [ ] Sigmoid/exponential curves optional
- [ ] Mapping latency under 10 microseconds

---

#### REQ-NEURO-004: Serotonin to FuseMoE Top-K Mapping

**Priority**: Critical
**Description**: The system SHALL map serotonin levels to FuseMoE expert selection count.

```rust
pub struct SerotoninMapper {
    /// Input range for serotonin level
    pub input_range: (f32, f32),   // (0.0, 1.0)

    /// Output range for fuse_moe.top_k
    pub output_range: (u32, u32),  // (2, 8)

    /// Rounding strategy for discrete output
    pub rounding: RoundingStrategy,
}

pub enum RoundingStrategy {
    Floor,
    Ceiling,
    Nearest,
    Probabilistic,  // Probability proportional to fractional part
}

impl SerotoninMapper {
    /// Map serotonin level [0,1] to fuse_moe.top_k [2, 8]
    ///
    /// High serotonin -> High top_k -> More experts consulted (exploration)
    /// Low serotonin -> Low top_k -> Fewer experts consulted (exploitation)
    ///
    /// `Constraint: Mapping_Latency < 10us`
    pub fn map(&self, serotonin: f32) -> u32 {
        let clamped = serotonin.clamp(0.0, 1.0);

        let continuous = self.output_range.0 as f32
            + clamped * (self.output_range.1 - self.output_range.0) as f32;

        match self.rounding {
            RoundingStrategy::Floor => continuous.floor() as u32,
            RoundingStrategy::Ceiling => continuous.ceil() as u32,
            RoundingStrategy::Nearest => continuous.round() as u32,
            RoundingStrategy::Probabilistic => {
                let base = continuous.floor() as u32;
                let frac = continuous.fract();
                if rand::random::<f32>() < frac {
                    (base + 1).min(self.output_range.1)
                } else {
                    base
                }
            }
        }
        .clamp(self.output_range.0, self.output_range.1)
    }

    /// Describe exploration-exploitation tradeoff
    pub fn describe_effect(&self, top_k: u32) -> ExplorationEffect {
        match top_k {
            2..=3 => ExplorationEffect::Focused { diversity: 0.2 },
            4..=5 => ExplorationEffect::Balanced { diversity: 0.5 },
            6..=8 => ExplorationEffect::Exploratory { diversity: 0.8 },
            _ => ExplorationEffect::Invalid,
        }
    }
}

pub enum ExplorationEffect {
    Focused { diversity: f32 },
    Balanced { diversity: f32 },
    Exploratory { diversity: f32 },
    Invalid,
}
```

**Acceptance Criteria**:
- [ ] Serotonin 0.0 maps to top_k = 2
- [ ] Serotonin 1.0 maps to top_k = 8
- [ ] Integer output with configurable rounding
- [ ] Bounds enforced (never below 2, never above 8)
- [ ] Probabilistic rounding for smoother transitions

---

#### REQ-NEURO-005: Noradrenaline to Attention Temperature Mapping

**Priority**: Critical
**Description**: The system SHALL map noradrenaline levels to attention mechanism temperature.

```rust
pub struct NoradrenalineMapper {
    /// Input range for noradrenaline level
    pub input_range: (f32, f32),   // (0.0, 1.0)

    /// Output range for attention.temperature
    pub output_range: (f32, f32),  // (0.5, 2.0)

    /// Inverse mapping (high noradrenaline = high temp = flat attention)
    pub inverse: bool,
}

impl NoradrenalineMapper {
    /// Map noradrenaline level [0,1] to attention.temperature [0.5, 2.0]
    ///
    /// High noradrenaline -> High temperature -> Flat/uniform attention (exploration)
    /// Low noradrenaline -> Low temperature -> Sharp/focused attention (exploitation)
    ///
    /// `Constraint: Mapping_Latency < 10us`
    pub fn map(&self, noradrenaline: f32) -> f32 {
        let clamped = noradrenaline.clamp(0.0, 1.0);

        // Linear mapping
        self.output_range.0 + clamped * (self.output_range.1 - self.output_range.0)
    }

    /// Describe attention distribution effect
    pub fn describe_effect(&self, temperature: f32) -> AttentionEffect {
        if temperature > 1.5 {
            AttentionEffect::Flat {
                entropy: 0.9,
                description: "Broad attention, high exploration",
            }
        } else if temperature > 0.8 {
            AttentionEffect::Balanced {
                entropy: 0.6,
                description: "Moderate focus with some exploration",
            }
        } else {
            AttentionEffect::Focused {
                entropy: 0.3,
                description: "Sharp attention, high exploitation",
            }
        }
    }
}

pub enum AttentionEffect {
    Focused { entropy: f32, description: &'static str },
    Balanced { entropy: f32, description: &'static str },
    Flat { entropy: f32, description: &'static str },
}
```

**Acceptance Criteria**:
- [ ] Noradrenaline 0.0 maps to temperature 0.5
- [ ] Noradrenaline 1.0 maps to temperature 2.0
- [ ] Linear mapping with smooth transitions
- [ ] Effect descriptions accurate
- [ ] Integration with attention mechanism verified

---

#### REQ-NEURO-006: Acetylcholine to UTL Learning Rate Mapping

**Priority**: Critical
**Description**: The system SHALL map acetylcholine levels to UTL optimizer learning rate.

```rust
pub struct AcetylcholineMapper {
    /// Input range for acetylcholine level
    pub input_range: (f32, f32),   // (0.0, 1.0)

    /// Output range for utl.learning_rate
    pub output_range: (f32, f32),  // (0.001, 0.002)

    /// Logarithmic scaling for learning rate
    pub log_scale: bool,
}

impl AcetylcholineMapper {
    /// Map acetylcholine level [0,1] to utl.learning_rate [0.001, 0.002]
    ///
    /// High acetylcholine -> High learning rate -> Faster memory updates
    /// Low acetylcholine -> Low learning rate -> Slower, more stable updates
    ///
    /// `Constraint: Mapping_Latency < 10us`
    pub fn map(&self, acetylcholine: f32) -> f32 {
        let clamped = acetylcholine.clamp(0.0, 1.0);

        if self.log_scale {
            // Logarithmic interpolation for learning rates
            let log_min = self.output_range.0.ln();
            let log_max = self.output_range.1.ln();
            let log_result = log_min + clamped * (log_max - log_min);
            log_result.exp()
        } else {
            // Linear interpolation
            self.output_range.0 + clamped * (self.output_range.1 - self.output_range.0)
        }
    }

    /// Describe learning rate effect
    pub fn describe_effect(&self, learning_rate: f32) -> LearningEffect {
        let normalized = (learning_rate - 0.001) / 0.001;  // 0 to 1
        if normalized > 0.7 {
            LearningEffect::Fast {
                plasticity: 0.9,
                stability: 0.3,
            }
        } else if normalized > 0.3 {
            LearningEffect::Moderate {
                plasticity: 0.6,
                stability: 0.6,
            }
        } else {
            LearningEffect::Slow {
                plasticity: 0.3,
                stability: 0.9,
            }
        }
    }
}

pub enum LearningEffect {
    Fast { plasticity: f32, stability: f32 },
    Moderate { plasticity: f32, stability: f32 },
    Slow { plasticity: f32, stability: f32 },
}
```

**Acceptance Criteria**:
- [ ] Acetylcholine 0.0 maps to learning rate 0.001
- [ ] Acetylcholine 1.0 maps to learning rate 0.002
- [ ] Logarithmic scaling option available
- [ ] Learning rate stays within valid bounds
- [ ] Stability-plasticity tradeoff documented

---

#### REQ-NEURO-007: Unified Parameter Mapping Interface

**Priority**: High
**Description**: The system SHALL provide a unified interface for all parameter mappings.

```rust
pub struct ParameterMapping {
    pub dopamine: DopamineMapper,
    pub serotonin: SerotoninMapper,
    pub noradrenaline: NoradrenalineMapper,
    pub acetylcholine: AcetylcholineMapper,
}

impl ParameterMapping {
    /// Create default parameter mapping from constitution.yaml
    pub fn from_constitution() -> Self {
        Self {
            dopamine: DopamineMapper {
                input_range: (0.0, 1.0),
                output_range: (1.0, 5.0),
                curve: MappingCurve::Linear,
            },
            serotonin: SerotoninMapper {
                input_range: (0.0, 1.0),
                output_range: (2, 8),
                rounding: RoundingStrategy::Nearest,
            },
            noradrenaline: NoradrenalineMapper {
                input_range: (0.0, 1.0),
                output_range: (0.5, 2.0),
                inverse: false,
            },
            acetylcholine: AcetylcholineMapper {
                input_range: (0.0, 1.0),
                output_range: (0.001, 0.002),
                log_scale: false,
            },
        }
    }

    /// Apply all mappings and return system parameters
    ///
    /// `Constraint: Total_Mapping_Latency < 50us`
    pub fn apply(&self, controller: &NeuromodulationController) -> MappedParameters {
        MappedParameters {
            hopfield_beta: self.dopamine.map(controller.dopamine),
            fuse_moe_top_k: self.serotonin.map(controller.serotonin),
            attention_temperature: self.noradrenaline.map(controller.noradrenaline),
            utl_learning_rate: self.acetylcholine.map(controller.acetylcholine),
        }
    }
}

/// System parameters after neuromodulation mapping
#[derive(Clone, Copy, Debug)]
pub struct MappedParameters {
    pub hopfield_beta: f32,
    pub fuse_moe_top_k: u32,
    pub attention_temperature: f32,
    pub utl_learning_rate: f32,
}
```

**Acceptance Criteria**:
- [ ] All 4 mappings applied in single call
- [ ] Total mapping latency under 50 microseconds
- [ ] Constitution.yaml defaults used
- [ ] Parameters easily overridable
- [ ] Mapped values immediately usable

---

### 2.3 Trigger Conditions Requirements

#### REQ-NEURO-008: High Surprise Dopamine Surge

**Priority**: Critical
**Description**: The system SHALL trigger dopamine surge on high surprise events.

```rust
pub struct SurpriseTrigger {
    /// Surprise threshold for dopamine surge
    pub surprise_threshold: f32,  // Default: 0.7

    /// Magnitude of dopamine increase
    pub surge_magnitude: f32,     // Default: 0.3

    /// Decay rate after surge
    pub decay_rate: f32,          // Default: 0.05 per update

    /// Cooldown between surges
    pub cooldown: Duration,       // Default: 5 seconds
}

impl SurpriseTrigger {
    /// Check if surprise level triggers dopamine surge
    ///
    /// High surprise -> Dopamine surge -> Sharper retrieval for exploitation
    /// of novel information
    ///
    /// `Constraint: Trigger_Check_Latency < 10us`
    pub fn check(&self, utl_state: &UtlState, last_trigger: Instant) -> Option<DopamineSurge> {
        // Surprise correlates with entropy change (delta_s)
        let surprise = utl_state.delta_s;

        // Check cooldown
        if last_trigger.elapsed() < self.cooldown {
            return None;
        }

        // Check threshold
        if surprise >= self.surprise_threshold {
            Some(DopamineSurge {
                magnitude: self.surge_magnitude,
                decay_rate: self.decay_rate,
                triggered_at: Instant::now(),
                trigger_source: TriggerSource::HighSurprise { surprise },
            })
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct DopamineSurge {
    pub magnitude: f32,
    pub decay_rate: f32,
    pub triggered_at: Instant,
    pub trigger_source: TriggerSource,
}
```

**Acceptance Criteria**:
- [ ] Surprise threshold of 0.7 (delta_s) triggers surge
- [ ] Dopamine increases by 0.3 on surge
- [ ] Gradual decay after surge (0.05 per update)
- [ ] Cooldown prevents excessive surges
- [ ] Trigger latency under 10 microseconds

---

#### REQ-NEURO-009: Uncertainty Noradrenaline Increase

**Priority**: Critical
**Description**: The system SHALL increase noradrenaline during high uncertainty states.

```rust
pub struct UncertaintyTrigger {
    /// Entropy threshold for noradrenaline increase
    pub entropy_threshold: f32,    // Default: 0.6

    /// Coherence threshold (low coherence = uncertainty)
    pub coherence_threshold: f32,  // Default: 0.4

    /// Magnitude of noradrenaline increase
    pub increase_magnitude: f32,   // Default: 0.25

    /// Sustained uncertainty duration before trigger
    pub duration_threshold: Duration,  // Default: 2 seconds
}

impl UncertaintyTrigger {
    /// Check if uncertainty triggers noradrenaline increase
    ///
    /// High uncertainty -> Noradrenaline increase -> Flatter attention
    /// for broader exploration
    ///
    /// `Constraint: Trigger_Check_Latency < 10us`
    pub fn check(&self, utl_state: &UtlState, uncertainty_duration: Duration)
        -> Option<NoradrenalineIncrease> {

        // Uncertainty = high entropy + low coherence
        let is_uncertain = utl_state.entropy > self.entropy_threshold
            && utl_state.coherence < self.coherence_threshold;

        // Check sustained duration
        if is_uncertain && uncertainty_duration >= self.duration_threshold {
            let magnitude = self.increase_magnitude
                * (1.0 + (utl_state.entropy - self.entropy_threshold));

            Some(NoradrenalineIncrease {
                magnitude: magnitude.min(0.4),  // Cap at 0.4
                trigger_source: TriggerSource::HighUncertainty {
                    entropy: utl_state.entropy,
                    coherence: utl_state.coherence,
                },
            })
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct NoradrenalineIncrease {
    pub magnitude: f32,
    pub trigger_source: TriggerSource,
}
```

**Acceptance Criteria**:
- [ ] Entropy > 0.6 AND coherence < 0.4 triggers increase
- [ ] 2-second sustained uncertainty before trigger
- [ ] Magnitude proportional to uncertainty level
- [ ] Cap on maximum increase (0.4)
- [ ] Trigger logged for debugging

---

#### REQ-NEURO-010: Exploration Mode Serotonin Boost

**Priority**: Critical
**Description**: The system SHALL boost serotonin when exploration mode is needed.

```rust
pub struct ExplorationTrigger {
    /// Johari quadrants that trigger exploration
    pub exploration_quadrants: Vec<JohariQuadrant>,

    /// Entropy threshold for exploration mode
    pub entropy_threshold: f32,   // Default: 0.7

    /// Serotonin boost magnitude
    pub boost_magnitude: f32,     // Default: 0.3

    /// Sustained duration before boost
    pub duration_threshold: Duration,  // Default: 3 seconds
}

impl ExplorationTrigger {
    /// Check if exploration mode triggers serotonin boost
    ///
    /// Exploration mode -> Serotonin boost -> More experts consulted
    ///
    /// `Constraint: Trigger_Check_Latency < 10us`
    pub fn check(&self, cognitive_state: &CognitiveState) -> Option<SerotoninBoost> {
        // Check if in exploration quadrant (Blind or Unknown)
        let in_exploration_quadrant = self.exploration_quadrants
            .contains(&cognitive_state.johari_quadrant);

        // Check high entropy (novelty)
        let high_entropy = cognitive_state.entropy > self.entropy_threshold;

        // Check suggested action
        let exploring = matches!(
            cognitive_state.suggested_action,
            SuggestedAction::EpistemicAction | SuggestedAction::Explore
        );

        if in_exploration_quadrant || (high_entropy && exploring) {
            Some(SerotoninBoost {
                magnitude: self.boost_magnitude,
                trigger_source: TriggerSource::ExplorationMode {
                    quadrant: cognitive_state.johari_quadrant,
                    entropy: cognitive_state.entropy,
                },
            })
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct SerotoninBoost {
    pub magnitude: f32,
    pub trigger_source: TriggerSource,
}
```

**Acceptance Criteria**:
- [ ] Blind/Unknown quadrants trigger exploration
- [ ] High entropy (> 0.7) with exploration action triggers boost
- [ ] Serotonin increases by 0.3 on boost
- [ ] Multiple triggers can stack (with cap)
- [ ] Exploration state logged

---

#### REQ-NEURO-011: Deep Recall Acetylcholine Rise

**Priority**: Critical
**Description**: The system SHALL increase acetylcholine when deep recall is needed.

```rust
pub struct DeepRecallTrigger {
    /// Coherence threshold indicating need for deeper recall
    pub coherence_threshold: f32,  // Default: 0.3

    /// Importance threshold of target content
    pub importance_threshold: f32, // Default: 0.7

    /// Acetylcholine increase magnitude
    pub increase_magnitude: f32,   // Default: 0.2

    /// Memory age factor (older = more acetylcholine needed)
    pub age_factor: f32,           // Default: 0.1
}

impl DeepRecallTrigger {
    /// Check if deep recall triggers acetylcholine rise
    ///
    /// Deep recall needed -> Acetylcholine rise -> Faster learning rate
    /// for memory consolidation
    ///
    /// `Constraint: Trigger_Check_Latency < 10us`
    pub fn check(&self, recall_context: &RecallContext) -> Option<AcetylcholineRise> {
        // Deep recall needed when:
        // 1. Low coherence (struggling to understand)
        // 2. High importance content
        // 3. Accessing older memories

        let needs_deep_recall = recall_context.coherence < self.coherence_threshold
            && recall_context.target_importance > self.importance_threshold;

        if needs_deep_recall {
            // Calculate magnitude based on memory age
            let age_hours = recall_context.memory_age.as_secs_f32() / 3600.0;
            let age_bonus = (age_hours / 24.0 * self.age_factor).min(0.1);

            Some(AcetylcholineRise {
                magnitude: self.increase_magnitude + age_bonus,
                trigger_source: TriggerSource::DeepRecall {
                    coherence: recall_context.coherence,
                    importance: recall_context.target_importance,
                    age_hours,
                },
            })
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecallContext {
    pub coherence: f32,
    pub target_importance: f32,
    pub memory_age: Duration,
    pub recall_depth: u32,
}

#[derive(Clone, Debug)]
pub struct AcetylcholineRise {
    pub magnitude: f32,
    pub trigger_source: TriggerSource,
}
```

**Acceptance Criteria**:
- [ ] Low coherence (< 0.3) + high importance (> 0.7) triggers rise
- [ ] Older memories receive additional boost
- [ ] Magnitude proportional to recall difficulty
- [ ] Learning rate increases for consolidation
- [ ] Recall context logged

---

#### REQ-NEURO-012: Unified Trigger Source Enum

**Priority**: High
**Description**: The system SHALL track all trigger sources for debugging and analysis.

```rust
/// Source of neuromodulator trigger
#[derive(Clone, Debug, Serialize)]
pub enum TriggerSource {
    /// High surprise/novelty triggered dopamine surge
    HighSurprise { surprise: f32 },

    /// High uncertainty triggered noradrenaline increase
    HighUncertainty { entropy: f32, coherence: f32 },

    /// Exploration mode triggered serotonin boost
    ExplorationMode { quadrant: JohariQuadrant, entropy: f32 },

    /// Deep recall triggered acetylcholine rise
    DeepRecall { coherence: f32, importance: f32, age_hours: f32 },

    /// Baseline return (no active trigger)
    BaselineReturn { elapsed: Duration },

    /// Dream phase modulation
    DreamPhase { phase: DreamPhase },

    /// Manual override (testing/debugging)
    ManualOverride { reason: String },

    /// System initialization
    Initialization,

    /// Steering feedback from Module 12.5 (Marblestone)
    SteeringFeedback {
        reward: f32,
        source: SteeringSource
    },
}

/// Record of a trigger event
#[derive(Clone, Debug, Serialize)]
pub struct TriggerEvent {
    pub timestamp: DateTime<Utc>,
    pub modulator: ModulatorType,
    pub source: TriggerSource,
    pub magnitude: f32,
    pub resulting_level: f32,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub enum ModulatorType {
    Dopamine,
    Serotonin,
    Noradrenaline,
    Acetylcholine,
}
```

**Acceptance Criteria**:
- [ ] All trigger sources enumerated
- [ ] Events recorded with timestamps
- [ ] Resulting levels tracked
- [ ] Events queryable for analysis
- [ ] Serializable for logging/export

---

#### REQ-NEURO-035: Steering Dopamine Feedback (Marblestone)

**Priority**: Critical
**Description**: The system SHALL integrate steering feedback from Module 12.5 (Steering Subsystem) into dopamine modulation per Marblestone's reward signal architecture.

```rust
// ============================================
// STEERING DOPAMINE FEEDBACK (Marblestone)
// ============================================

/// Dopamine feedback from the Steering Subsystem (Marblestone)
///
/// Integrates with Module 12.5 (Steering Subsystem) to receive
/// reward signals that modulate dopamine levels in real-time.
#[derive(Clone, Debug)]
pub struct SteeringDopamineFeedback {
    /// Reward signal from Thought Assessor [-1.0, 1.0]
    pub reward: f32,
    /// Source of the feedback
    pub source: SteeringSource,
    /// Timestamp of feedback
    pub timestamp: DateTime<Utc>,
    /// Associated memory/thought ID
    pub thought_id: Option<Uuid>,
}

/// Source of steering feedback
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum SteeringSource {
    /// Gardener component (long-term curation)
    Gardener,
    /// Curator component (quality assessment)
    Curator,
    /// Thought Assessor (immediate evaluation)
    ThoughtAssessor,
}

impl NeuromodulationController {
    /// Apply steering feedback to dopamine levels (Marblestone)
    ///
    /// Positive rewards increase dopamine (reinforce behavior)
    /// Negative rewards decrease dopamine (discourage behavior)
    ///
    /// # Arguments
    /// * `feedback` - Steering feedback from Module 12.5
    ///
    /// # Effects
    /// - reward > 0: dopamine += reward * 0.2 (capped at 1.0)
    /// - reward < 0: dopamine += reward * 0.1 (floored at 0.0)
    ///
    /// `Constraint: Steering_Feedback_Latency < 20us`
    pub fn apply_steering_feedback(&mut self, feedback: &SteeringDopamineFeedback) {
        let delta = if feedback.reward > 0.0 {
            feedback.reward * 0.2  // Positive reinforcement
        } else {
            feedback.reward * 0.1  // Negative feedback (gentler)
        };

        self.dopamine = (self.dopamine + delta).clamp(0.0, 1.0);

        // Log for metrics
        self.last_steering_feedback = Some(feedback.clone());

        // Record trigger event
        self.metrics.record_steering_feedback(feedback, self.dopamine);
    }

    /// Get steering-modulated dopamine level
    pub fn get_steered_dopamine(&self) -> f32 {
        self.dopamine
    }

    /// Check if recent steering feedback was applied
    pub fn has_recent_steering_feedback(&self, within: Duration) -> bool {
        self.last_steering_feedback
            .as_ref()
            .map(|f| f.timestamp > Utc::now() - chrono::Duration::from_std(within).unwrap())
            .unwrap_or(false)
    }
}

/// Extended controller fields for steering integration
impl NeuromodulationController {
    /// Last steering feedback received (for tracking)
    pub last_steering_feedback: Option<SteeringDopamineFeedback>,
}
```

**Acceptance Criteria**:
- [ ] SteeringDopamineFeedback struct receives reward signals from Module 12.5
- [ ] Positive reward (> 0) increases dopamine by reward * 0.2
- [ ] Negative reward (< 0) decreases dopamine by |reward| * 0.1
- [ ] Dopamine level clamped to [0.0, 1.0] range
- [ ] SteeringSource tracks origin (Gardener, Curator, ThoughtAssessor)
- [ ] Feedback timestamp and thought_id captured for traceability
- [ ] Integration with Marblestone reward signal architecture verified

**Requirements Table (Marblestone Steering Integration)**:

| ID | Requirement | Priority | Rationale |
|----|-------------|----------|-----------|
| REQ-NEURO-035 | System SHALL integrate steering feedback into dopamine modulation | Must | Marblestone reward signal |
| REQ-NEURO-036 | Positive steering reward SHALL increase dopamine by reward * 0.2 | Must | Reinforcement learning |
| REQ-NEURO-037 | Negative steering reward SHALL decrease dopamine by |reward| * 0.1 | Must | Gentler punishment |

---

### 2.4 Smooth Interpolation and Hysteresis

#### REQ-NEURO-013: Smooth Parameter Interpolation

**Priority**: Critical
**Description**: The system SHALL implement smooth interpolation between neuromodulator states.

```rust
pub struct InterpolationEngine {
    /// Maximum change per update step
    pub max_delta: f32,           // Default: 0.05

    /// Interpolation curve type
    pub curve: InterpolationCurve,

    /// Update interval
    pub update_interval: Duration,  // Default: 50ms
}

pub enum InterpolationCurve {
    Linear,
    EaseInOut,
    Exponential { decay: f32 },
    Spring { stiffness: f32, damping: f32 },
}

impl InterpolationEngine {
    /// Interpolate from current to target level
    ///
    /// `Constraint: Interpolation_Step_Latency < 20us`
    pub fn step(&self, current: f32, target: f32) -> f32 {
        let diff = target - current;
        let clamped_diff = diff.clamp(-self.max_delta, self.max_delta);

        match self.curve {
            InterpolationCurve::Linear => {
                current + clamped_diff
            }
            InterpolationCurve::EaseInOut => {
                // Smoother acceleration/deceleration
                let t = (clamped_diff.abs() / self.max_delta).min(1.0);
                let eased = t * t * (3.0 - 2.0 * t);
                current + clamped_diff.signum() * eased * self.max_delta
            }
            InterpolationCurve::Exponential { decay } => {
                current + diff * decay
            }
            InterpolationCurve::Spring { stiffness, damping } => {
                // Spring-damper system for natural feel
                let force = stiffness * diff;
                let damped = force * (1.0 - damping);
                current + damped.clamp(-self.max_delta, self.max_delta)
            }
        }
    }

    /// Interpolate all modulators in controller
    pub fn step_all(&self, controller: &mut NeuromodulationController, targets: &ModulatorTargets) {
        controller.dopamine = self.step(controller.dopamine, targets.dopamine);
        controller.serotonin = self.step(controller.serotonin, targets.serotonin);
        controller.noradrenaline = self.step(controller.noradrenaline, targets.noradrenaline);
        controller.acetylcholine = self.step(controller.acetylcholine, targets.acetylcholine);
    }
}

pub struct ModulatorTargets {
    pub dopamine: f32,
    pub serotonin: f32,
    pub noradrenaline: f32,
    pub acetylcholine: f32,
}
```

**Acceptance Criteria**:
- [ ] Maximum delta of 0.05 per step
- [ ] Multiple interpolation curves supported
- [ ] Smooth transitions between states
- [ ] No sudden parameter jumps
- [ ] Step latency under 20 microseconds

---

#### REQ-NEURO-014: Hysteresis for Oscillation Prevention

**Priority**: Critical
**Description**: The system SHALL implement hysteresis to prevent parameter oscillation.

```rust
pub struct HysteresisConfig {
    /// Dead zone width (no change if within this range)
    pub dead_zone: f32,           // Default: 0.02

    /// Minimum time between direction changes
    pub direction_change_delay: Duration,  // Default: 500ms

    /// Momentum factor (resistance to direction change)
    pub momentum: f32,            // Default: 0.3

    /// Per-modulator state
    state: HysteresisState,
}

#[derive(Default)]
pub struct HysteresisState {
    pub last_direction: Option<Direction>,
    pub last_direction_change: Option<Instant>,
    pub accumulated_pressure: f32,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Direction {
    Increasing,
    Decreasing,
    Stable,
}

impl HysteresisConfig {
    /// Apply hysteresis to proposed change
    ///
    /// `Constraint: Hysteresis_Check_Latency < 10us`
    pub fn apply(&mut self, current: f32, proposed_target: f32) -> f32 {
        let diff = proposed_target - current;

        // Dead zone check
        if diff.abs() < self.dead_zone {
            return current;
        }

        // Determine direction
        let new_direction = if diff > 0.0 {
            Direction::Increasing
        } else if diff < 0.0 {
            Direction::Decreasing
        } else {
            Direction::Stable
        };

        // Check direction change delay
        if let (Some(last_dir), Some(last_change)) = (self.state.last_direction, self.state.last_direction_change) {
            if new_direction != last_dir && last_change.elapsed() < self.direction_change_delay {
                // Accumulate pressure but don't change yet
                self.state.accumulated_pressure += diff.abs();
                if self.state.accumulated_pressure < self.momentum {
                    return current;
                }
            }
        }

        // Update state
        if Some(new_direction) != self.state.last_direction {
            self.state.last_direction = Some(new_direction);
            self.state.last_direction_change = Some(Instant::now());
            self.state.accumulated_pressure = 0.0;
        }

        proposed_target
    }
}
```

**Acceptance Criteria**:
- [ ] Dead zone of 0.02 prevents micro-oscillations
- [ ] 500ms delay between direction changes
- [ ] Momentum resists sudden reversals
- [ ] Pressure accumulation allows breakthrough
- [ ] Oscillation frequency reduced by >90%

---

#### REQ-NEURO-015: Bounds Checking for Stability

**Priority**: Critical
**Description**: The system SHALL enforce strict bounds on all neuromodulator levels.

```rust
pub struct BoundsChecker {
    /// Modulator level bounds (always 0.0 to 1.0)
    pub level_bounds: (f32, f32),

    /// Alert threshold for extreme values
    pub alert_threshold: f32,  // Default: 0.9 or 0.1

    /// Soft limiting vs hard clamping
    pub soft_limit: bool,

    /// Soft limit compression factor
    pub compression_factor: f32,  // Default: 0.8
}

impl BoundsChecker {
    /// Check and enforce bounds on modulator level
    ///
    /// `Constraint: Bounds_Check_Latency < 5us`
    pub fn check(&self, level: f32) -> BoundsCheckResult {
        let bounded_level = if self.soft_limit {
            self.soft_clamp(level)
        } else {
            level.clamp(self.level_bounds.0, self.level_bounds.1)
        };

        let alert = if bounded_level >= self.alert_threshold
            || bounded_level <= (1.0 - self.alert_threshold) {
            Some(BoundsAlert {
                level: bounded_level,
                severity: if bounded_level == self.level_bounds.0
                    || bounded_level == self.level_bounds.1 {
                    Severity::Warning
                } else {
                    Severity::Info
                },
            })
        } else {
            None
        };

        BoundsCheckResult {
            bounded_level,
            original_level: level,
            was_clamped: (level - bounded_level).abs() > f32::EPSILON,
            alert,
        }
    }

    /// Soft clamping using hyperbolic tangent
    fn soft_clamp(&self, level: f32) -> f32 {
        let center = (self.level_bounds.0 + self.level_bounds.1) / 2.0;
        let range = (self.level_bounds.1 - self.level_bounds.0) / 2.0;

        let normalized = (level - center) / range;
        let compressed = normalized.tanh() * self.compression_factor;

        center + compressed * range
    }
}

pub struct BoundsCheckResult {
    pub bounded_level: f32,
    pub original_level: f32,
    pub was_clamped: bool,
    pub alert: Option<BoundsAlert>,
}

pub struct BoundsAlert {
    pub level: f32,
    pub severity: Severity,
}

#[derive(Clone, Copy)]
pub enum Severity {
    Info,
    Warning,
    Error,
}
```

**Acceptance Criteria**:
- [ ] All levels bounded to [0.0, 1.0]
- [ ] Alerts at extreme values (> 0.9 or < 0.1)
- [ ] Soft limiting prevents hard cutoffs
- [ ] Bounds check latency under 5 microseconds
- [ ] Clamping events logged

---

### 2.5 Update Mechanism Requirements

#### REQ-NEURO-016: Per-Query Update Cycle

**Priority**: Critical
**Description**: The system SHALL update neuromodulator levels on each query.

```rust
impl NeuromodulationController {
    /// Update all neuromodulator levels for current query
    ///
    /// This is called once per query/interaction and must complete
    /// within the 200 microsecond budget.
    ///
    /// `Constraint: Total_Update_Latency < 200us`
    pub fn update(&mut self, context: &QueryContext) -> UpdateResult {
        let start = Instant::now();

        // 1. Check trigger conditions (~40us)
        let triggers = self.check_triggers(context);

        // 2. Calculate target levels (~20us)
        let targets = self.calculate_targets(&triggers);

        // 3. Apply hysteresis (~10us)
        let hysteresis_targets = self.apply_hysteresis(&targets);

        // 4. Interpolate toward targets (~20us)
        self.interpolate_step(&hysteresis_targets);

        // 5. Check bounds (~5us)
        self.enforce_bounds();

        // 6. Map to system parameters (~50us)
        let mapped = self.parameter_map.apply(self);

        // 7. Record metrics (~10us)
        let latency = start.elapsed();
        self.metrics.record_update(latency, &triggers);

        if latency > Duration::from_micros(200) {
            warn!("Neuromodulation update exceeded budget: {:?}", latency);
        }

        UpdateResult {
            mapped_parameters: mapped,
            triggers_fired: triggers,
            latency,
            levels: NeuromodulatorLevels {
                dopamine: self.dopamine,
                serotonin: self.serotonin,
                noradrenaline: self.noradrenaline,
                acetylcholine: self.acetylcholine,
            },
        }
    }

    /// Check all trigger conditions
    fn check_triggers(&self, context: &QueryContext) -> Vec<TriggerEvent> {
        let mut triggers = Vec::new();

        if let Some(surge) = self.surprise_trigger.check(&context.utl_state, self.last_dopamine_surge) {
            triggers.push(TriggerEvent::dopamine(surge));
        }

        if let Some(increase) = self.uncertainty_trigger.check(&context.utl_state, context.uncertainty_duration) {
            triggers.push(TriggerEvent::noradrenaline(increase));
        }

        if let Some(boost) = self.exploration_trigger.check(&context.cognitive_state) {
            triggers.push(TriggerEvent::serotonin(boost));
        }

        if let Some(rise) = self.recall_trigger.check(&context.recall_context) {
            triggers.push(TriggerEvent::acetylcholine(rise));
        }

        triggers
    }
}

pub struct UpdateResult {
    pub mapped_parameters: MappedParameters,
    pub triggers_fired: Vec<TriggerEvent>,
    pub latency: Duration,
    pub levels: NeuromodulatorLevels,
}

#[derive(Clone, Copy, Serialize)]
pub struct NeuromodulatorLevels {
    pub dopamine: f32,
    pub serotonin: f32,
    pub noradrenaline: f32,
    pub acetylcholine: f32,
}
```

**Acceptance Criteria**:
- [ ] Update completes within 200 microseconds
- [ ] All trigger conditions checked
- [ ] Hysteresis applied before interpolation
- [ ] Bounds enforced after interpolation
- [ ] Metrics recorded for each update

---

#### REQ-NEURO-017: Baseline Return Mechanism

**Priority**: High
**Description**: The system SHALL return modulator levels to baseline when no triggers are active.

```rust
impl NeuromodulationController {
    /// Calculate target levels including baseline return
    fn calculate_targets(&self, triggers: &[TriggerEvent]) -> ModulatorTargets {
        let mut targets = ModulatorTargets {
            dopamine: self.baseline.dopamine,
            serotonin: self.baseline.serotonin,
            noradrenaline: self.baseline.noradrenaline,
            acetylcholine: self.baseline.acetylcholine,
        };

        // Apply triggers additively
        for trigger in triggers {
            match trigger.modulator {
                ModulatorType::Dopamine => {
                    targets.dopamine = (targets.dopamine + trigger.magnitude).min(1.0);
                }
                ModulatorType::Serotonin => {
                    targets.serotonin = (targets.serotonin + trigger.magnitude).min(1.0);
                }
                ModulatorType::Noradrenaline => {
                    targets.noradrenaline = (targets.noradrenaline + trigger.magnitude).min(1.0);
                }
                ModulatorType::Acetylcholine => {
                    targets.acetylcholine = (targets.acetylcholine + trigger.magnitude).min(1.0);
                }
            }
        }

        targets
    }

    /// Apply decay toward baseline for modulators without active triggers
    fn apply_baseline_decay(&mut self, triggers: &[TriggerEvent]) {
        let decay = self.update_rate;

        // Check which modulators have active triggers
        let has_dopamine_trigger = triggers.iter().any(|t| t.modulator == ModulatorType::Dopamine);
        let has_serotonin_trigger = triggers.iter().any(|t| t.modulator == ModulatorType::Serotonin);
        let has_noradrenaline_trigger = triggers.iter().any(|t| t.modulator == ModulatorType::Noradrenaline);
        let has_acetylcholine_trigger = triggers.iter().any(|t| t.modulator == ModulatorType::Acetylcholine);

        // Decay toward baseline if no trigger
        if !has_dopamine_trigger {
            self.dopamine = self.decay_toward(self.dopamine, self.baseline.dopamine, decay);
        }
        if !has_serotonin_trigger {
            self.serotonin = self.decay_toward(self.serotonin, self.baseline.serotonin, decay);
        }
        if !has_noradrenaline_trigger {
            self.noradrenaline = self.decay_toward(self.noradrenaline, self.baseline.noradrenaline, decay);
        }
        if !has_acetylcholine_trigger {
            self.acetylcholine = self.decay_toward(self.acetylcholine, self.baseline.acetylcholine, decay);
        }
    }

    /// Decay value toward target at given rate
    fn decay_toward(&self, current: f32, target: f32, rate: f32) -> f32 {
        let diff = target - current;
        current + diff * rate
    }
}
```

**Acceptance Criteria**:
- [ ] Modulators return to baseline when triggers inactive
- [ ] Decay rate of 0.1 (10% per update toward baseline)
- [ ] Multiple triggers can be active simultaneously
- [ ] Triggers are additive (with cap at 1.0)
- [ ] Baseline levels configurable

---

### 2.6 Integration Requirements

#### REQ-NEURO-018: Integration with Hopfield Network

**Priority**: Critical
**Description**: The system SHALL integrate dopamine-controlled beta with Hopfield retrieval.

```rust
pub struct HopfieldIntegration {
    /// Reference to neuromodulation controller
    controller: Arc<RwLock<NeuromodulationController>>,
}

impl HopfieldIntegration {
    /// Get current beta value for Hopfield network
    ///
    /// `Constraint: Beta_Query_Latency < 10us`
    pub fn get_beta(&self) -> f32 {
        let controller = self.controller.read().expect("Failed to read controller");
        controller.parameter_map.dopamine.map(controller.dopamine)
    }

    /// Apply beta to Hopfield retrieval
    pub fn apply_to_retrieval(&self, hopfield: &mut HopfieldNetwork) {
        let beta = self.get_beta();
        hopfield.set_beta(beta);
    }
}

/// Hopfield network with neuromodulation support
pub trait NeuromodulatedHopfield {
    /// Set inverse temperature (beta) from neuromodulation
    fn set_beta(&mut self, beta: f32);

    /// Get current beta value
    fn get_beta(&self) -> f32;

    /// Retrieve pattern with current beta
    fn retrieve(&self, query: &[f32], beta: f32) -> Vec<f32>;
}
```

**Acceptance Criteria**:
- [ ] Beta value retrieved in under 10 microseconds
- [ ] Hopfield network uses modulated beta
- [ ] High dopamine produces sharp retrieval
- [ ] Low dopamine produces diffuse retrieval
- [ ] Integration tested with real patterns

---

#### REQ-NEURO-019: Integration with FuseMoE

**Priority**: Critical
**Description**: The system SHALL integrate serotonin-controlled top_k with FuseMoE fusion.

```rust
pub struct FuseMoEIntegration {
    /// Reference to neuromodulation controller
    controller: Arc<RwLock<NeuromodulationController>>,
}

impl FuseMoEIntegration {
    /// Get current top_k value for FuseMoE
    ///
    /// `Constraint: TopK_Query_Latency < 10us`
    pub fn get_top_k(&self) -> u32 {
        let controller = self.controller.read().expect("Failed to read controller");
        controller.parameter_map.serotonin.map(controller.serotonin)
    }

    /// Apply top_k to FuseMoE fusion
    pub fn apply_to_fusion(&self, fuse_moe: &mut FuseMoE) {
        let top_k = self.get_top_k();
        fuse_moe.set_top_k(top_k as usize);
    }
}

/// FuseMoE with neuromodulation support
pub trait NeuromodulatedFuseMoE {
    /// Set number of experts to consult
    fn set_top_k(&mut self, top_k: usize);

    /// Get current top_k value
    fn get_top_k(&self) -> usize;

    /// Fuse embeddings with current top_k
    fn fuse(&self, embeddings: &[Vec<f32>], top_k: usize) -> Vec<f32>;
}
```

**Acceptance Criteria**:
- [ ] top_k value retrieved in under 10 microseconds
- [ ] FuseMoE uses modulated top_k
- [ ] High serotonin consults more experts
- [ ] Low serotonin uses fewer experts
- [ ] Integration tested with real embeddings

---

#### REQ-NEURO-020: Integration with Attention Mechanism

**Priority**: Critical
**Description**: The system SHALL integrate noradrenaline-controlled temperature with attention.

```rust
pub struct AttentionIntegration {
    /// Reference to neuromodulation controller
    controller: Arc<RwLock<NeuromodulationController>>,
}

impl AttentionIntegration {
    /// Get current temperature value for attention
    ///
    /// `Constraint: Temperature_Query_Latency < 10us`
    pub fn get_temperature(&self) -> f32 {
        let controller = self.controller.read().expect("Failed to read controller");
        controller.parameter_map.noradrenaline.map(controller.noradrenaline)
    }

    /// Apply temperature to attention mechanism
    pub fn apply_to_attention(&self, attention: &mut AttentionMechanism) {
        let temperature = self.get_temperature();
        attention.set_temperature(temperature);
    }
}

/// Attention mechanism with neuromodulation support
pub trait NeuromodulatedAttention {
    /// Set softmax temperature
    fn set_temperature(&mut self, temperature: f32);

    /// Get current temperature value
    fn get_temperature(&self) -> f32;

    /// Compute attention with current temperature
    fn attend(&self, query: &[f32], keys: &[Vec<f32>], temperature: f32) -> Vec<f32>;
}
```

**Acceptance Criteria**:
- [ ] Temperature value retrieved in under 10 microseconds
- [ ] Attention mechanism uses modulated temperature
- [ ] High noradrenaline produces flat attention
- [ ] Low noradrenaline produces focused attention
- [ ] Integration tested with real attention computation

---

#### REQ-NEURO-021: Integration with UTL Optimizer

**Priority**: Critical
**Description**: The system SHALL integrate acetylcholine-controlled learning rate with UTL.

```rust
pub struct UTLIntegration {
    /// Reference to neuromodulation controller
    controller: Arc<RwLock<NeuromodulationController>>,
}

impl UTLIntegration {
    /// Get current learning rate for UTL optimizer
    ///
    /// `Constraint: LearningRate_Query_Latency < 10us`
    pub fn get_learning_rate(&self) -> f32 {
        let controller = self.controller.read().expect("Failed to read controller");
        controller.parameter_map.acetylcholine.map(controller.acetylcholine)
    }

    /// Apply learning rate to UTL optimizer
    pub fn apply_to_optimizer(&self, optimizer: &mut UtlOptimizer) {
        let learning_rate = self.get_learning_rate();
        optimizer.set_learning_rate(learning_rate);
    }
}

/// UTL optimizer with neuromodulation support
pub trait NeuromodulatedUtl {
    /// Set learning rate for weight updates
    fn set_learning_rate(&mut self, lr: f32);

    /// Get current learning rate
    fn get_learning_rate(&self) -> f32;

    /// Optimize with current learning rate
    fn optimize(&mut self, gradients: &UtlGradients, lr: f32) -> UtlOptimizeResult;
}
```

**Acceptance Criteria**:
- [ ] Learning rate retrieved in under 10 microseconds
- [ ] UTL optimizer uses modulated learning rate
- [ ] High acetylcholine enables faster learning
- [ ] Low acetylcholine provides stable learning
- [ ] Integration tested with real UTL updates

---

#### REQ-NEURO-022: Integration with Dream Layer

**Priority**: High
**Description**: The system SHALL coordinate neuromodulator levels with dream phases.

```rust
pub struct DreamIntegration {
    /// Reference to neuromodulation controller
    controller: Arc<RwLock<NeuromodulationController>>,

    /// Phase-specific modulator settings
    phase_settings: DreamPhaseSettings,
}

#[derive(Clone)]
pub struct DreamPhaseSettings {
    /// NREM phase modulator targets
    pub nrem: ModulatorTargets,

    /// REM phase modulator targets
    pub rem: ModulatorTargets,

    /// Transition duration
    pub transition_duration: Duration,
}

impl Default for DreamPhaseSettings {
    fn default() -> Self {
        Self {
            nrem: ModulatorTargets {
                dopamine: 0.3,       // Low for broad replay
                serotonin: 0.4,      // Moderate
                noradrenaline: 0.2,  // Very low (consolidation focus)
                acetylcholine: 0.2,  // Low (slow updates)
            },
            rem: ModulatorTargets {
                dopamine: 0.7,       // Higher for exploration
                serotonin: 0.8,      // High for many experts
                noradrenaline: 0.6,  // Elevated for flat attention
                acetylcholine: 0.6,  // Moderate (creative updates)
            },
            transition_duration: Duration::from_millis(500),
        }
    }
}

impl DreamIntegration {
    /// Set modulator levels for dream phase
    pub fn enter_phase(&self, phase: DreamPhase) -> Result<(), NeuromodulationError> {
        let mut controller = self.controller.write().expect("Failed to write controller");

        let targets = match phase {
            DreamPhase::NREM => &self.phase_settings.nrem,
            DreamPhase::REM => &self.phase_settings.rem,
            DreamPhase::Transition => return Ok(()), // Handled by interpolation
            DreamPhase::Awake => {
                // Return to baseline
                &ModulatorTargets {
                    dopamine: controller.baseline.dopamine,
                    serotonin: controller.baseline.serotonin,
                    noradrenaline: controller.baseline.noradrenaline,
                    acetylcholine: controller.baseline.acetylcholine,
                }
            }
        };

        // Mark targets for smooth transition
        controller.set_dream_targets(targets.clone());

        Ok(())
    }

    /// Smoothly transition between phases
    pub fn transition(&self, from: DreamPhase, to: DreamPhase) {
        let start = Instant::now();
        let duration = self.phase_settings.transition_duration;

        // Interpolation handled by normal update cycle
        // This just logs the transition
        info!("Dream phase transition: {:?} -> {:?}", from, to);
    }
}
```

**Acceptance Criteria**:
- [ ] NREM phase has low dopamine/acetylcholine
- [ ] REM phase has high serotonin for exploration
- [ ] Smooth 500ms transitions between phases
- [ ] Wake returns to baseline
- [ ] Integration tested with dream layer

---

### 2.7 MCP Tool Integration

#### REQ-NEURO-023: get_neuromodulation Tool

**Priority**: Critical
**Description**: The system SHALL expose neuromodulation state through MCP tool.

```rust
/// MCP tool: get_neuromodulation
pub struct GetNeuromodulationTool {
    pub name: &'static str,  // "get_neuromodulation"
    pub description: &'static str,
}

#[derive(Deserialize)]
pub struct GetNeuromodulationParams {
    /// Session ID
    pub session_id: Option<Uuid>,

    /// Include history of recent changes
    #[serde(default)]
    pub include_history: bool,

    /// History window size
    #[serde(default = "default_history_size")]
    pub history_size: usize,  // Default: 10
}

#[derive(Serialize)]
pub struct GetNeuromodulationResult {
    /// Current modulator levels [0.0, 1.0]
    pub levels: NeuromodulatorLevels,

    /// Mapped system parameters
    pub mapped_parameters: MappedParameters,

    /// Active triggers
    pub active_triggers: Vec<ActiveTrigger>,

    /// Behavioral state description
    pub behavioral_state: BehavioralState,

    /// Recent history (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history: Option<Vec<NeuromodulationHistoryEntry>>,
}

#[derive(Serialize)]
pub struct ActiveTrigger {
    pub modulator: ModulatorType,
    pub source: TriggerSource,
    pub magnitude: f32,
    pub active_since: DateTime<Utc>,
}

#[derive(Serialize)]
pub struct BehavioralState {
    /// Overall exploration-exploitation balance
    pub exploration_exploitation: f32,  // -1 (exploit) to +1 (explore)

    /// Learning plasticity level
    pub plasticity: f32,

    /// Attention focus level
    pub focus: f32,

    /// State description
    pub description: String,
}

impl GetNeuromodulationTool {
    pub async fn execute(&self, params: GetNeuromodulationParams, ctx: &ToolContext)
        -> Result<GetNeuromodulationResult, ToolError> {

        let controller = ctx.get_neuromodulation_controller()?;
        let controller = controller.read().expect("Failed to read controller");

        let levels = NeuromodulatorLevels {
            dopamine: controller.dopamine,
            serotonin: controller.serotonin,
            noradrenaline: controller.noradrenaline,
            acetylcholine: controller.acetylcholine,
        };

        let mapped = controller.parameter_map.apply(&controller);

        let behavioral_state = self.compute_behavioral_state(&levels);

        let history = if params.include_history {
            Some(controller.metrics.get_history(params.history_size))
        } else {
            None
        };

        Ok(GetNeuromodulationResult {
            levels,
            mapped_parameters: mapped,
            active_triggers: controller.get_active_triggers(),
            behavioral_state,
            history,
        })
    }

    fn compute_behavioral_state(&self, levels: &NeuromodulatorLevels) -> BehavioralState {
        // Exploration-exploitation balance
        // High serotonin + high noradrenaline = explore
        // High dopamine + low noradrenaline = exploit
        let explore = (levels.serotonin + levels.noradrenaline) / 2.0;
        let exploit = levels.dopamine * (1.0 - levels.noradrenaline);
        let ee_balance = explore - exploit;  // -1 to +1

        // Plasticity from acetylcholine
        let plasticity = levels.acetylcholine;

        // Focus from inverse of noradrenaline
        let focus = 1.0 - levels.noradrenaline;

        let description = if ee_balance > 0.3 {
            "Exploratory: Broad attention, consulting multiple experts"
        } else if ee_balance < -0.3 {
            "Exploitative: Focused retrieval, high selectivity"
        } else {
            "Balanced: Moderate focus with some exploration"
        }.to_string();

        BehavioralState {
            exploration_exploitation: ee_balance,
            plasticity,
            focus,
            description,
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Tool returns all 4 modulator levels
- [ ] Mapped parameters included
- [ ] Active triggers shown
- [ ] Behavioral state computed
- [ ] History available on request

---

#### REQ-NEURO-024: Neuromodulation in Cognitive Pulse

**Priority**: High
**Description**: The system SHALL include neuromodulation summary in Cognitive Pulse header.

```rust
/// Extended Cognitive Pulse with neuromodulation
#[derive(Serialize)]
pub struct CognitivePulse {
    /// Current entropy level
    pub entropy: f32,

    /// Current coherence level
    pub coherence: f32,

    /// Suggested action
    pub suggested_action: SuggestedAction,

    /// Neuromodulation summary
    pub neuromod: NeuromodulationSummary,
}

#[derive(Serialize)]
pub struct NeuromodulationSummary {
    /// Dominant modulator (highest deviation from baseline)
    pub dominant: ModulatorType,

    /// Exploration-exploitation balance (-1 to +1)
    pub ee_balance: f32,

    /// Brief state description
    pub state: &'static str,
}

impl NeuromodulationSummary {
    pub fn from_controller(controller: &NeuromodulationController) -> Self {
        let deviations = [
            (ModulatorType::Dopamine, (controller.dopamine - controller.baseline.dopamine).abs()),
            (ModulatorType::Serotonin, (controller.serotonin - controller.baseline.serotonin).abs()),
            (ModulatorType::Noradrenaline, (controller.noradrenaline - controller.baseline.noradrenaline).abs()),
            (ModulatorType::Acetylcholine, (controller.acetylcholine - controller.baseline.acetylcholine).abs()),
        ];

        let dominant = deviations.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(m, _)| *m)
            .unwrap_or(ModulatorType::Dopamine);

        let explore = (controller.serotonin + controller.noradrenaline) / 2.0;
        let exploit = controller.dopamine * (1.0 - controller.noradrenaline);
        let ee_balance = explore - exploit;

        let state = if ee_balance > 0.3 { "exploring" }
            else if ee_balance < -0.3 { "exploiting" }
            else { "balanced" };

        Self { dominant, ee_balance, state }
    }
}
```

**Acceptance Criteria**:
- [ ] Pulse includes neuromodulation summary
- [ ] Dominant modulator identified
- [ ] Exploration-exploitation balance computed
- [ ] State description clear
- [ ] Overhead under 20 tokens

---

### 2.8 Metrics and Monitoring

#### REQ-NEURO-025: Neuromodulation Metrics Collection

**Priority**: High
**Description**: The system SHALL collect comprehensive neuromodulation metrics.

```rust
#[derive(Default)]
pub struct NeuromodulationMetrics {
    /// Update latency histogram
    update_latencies: HistogramVec,

    /// Trigger frequency counters
    trigger_counts: CounterVec,

    /// Modulator level gauges
    level_gauges: GaugeVec,

    /// Mapped parameter gauges
    parameter_gauges: GaugeVec,

    /// History ring buffer
    history: RingBuffer<NeuromodulationHistoryEntry>,
}

#[derive(Clone, Serialize)]
pub struct NeuromodulationHistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub levels: NeuromodulatorLevels,
    pub mapped: MappedParameters,
    pub triggers: Vec<TriggerSource>,
    pub latency_us: u64,
}

impl NeuromodulationMetrics {
    /// Record an update cycle
    pub fn record_update(&mut self, latency: Duration, triggers: &[TriggerEvent]) {
        // Record latency
        self.update_latencies.observe(latency.as_micros() as f64);

        // Count triggers by type
        for trigger in triggers {
            self.trigger_counts.inc(&trigger.modulator.to_string());
        }
    }

    /// Update current level gauges
    pub fn update_levels(&mut self, levels: &NeuromodulatorLevels) {
        self.level_gauges.set("dopamine", levels.dopamine as f64);
        self.level_gauges.set("serotonin", levels.serotonin as f64);
        self.level_gauges.set("noradrenaline", levels.noradrenaline as f64);
        self.level_gauges.set("acetylcholine", levels.acetylcholine as f64);
    }

    /// Update mapped parameter gauges
    pub fn update_parameters(&mut self, params: &MappedParameters) {
        self.parameter_gauges.set("hopfield_beta", params.hopfield_beta as f64);
        self.parameter_gauges.set("fuse_moe_top_k", params.fuse_moe_top_k as f64);
        self.parameter_gauges.set("attention_temperature", params.attention_temperature as f64);
        self.parameter_gauges.set("utl_learning_rate", params.utl_learning_rate as f64);
    }

    /// Get recent history
    pub fn get_history(&self, count: usize) -> Vec<NeuromodulationHistoryEntry> {
        self.history.iter().take(count).cloned().collect()
    }

    /// Export Prometheus metrics
    pub fn export_prometheus(&self) -> String {
        let mut output = String::new();
        // ... format as Prometheus exposition format
        output
    }
}
```

**Acceptance Criteria**:
- [ ] Update latency tracked
- [ ] Trigger frequency counted
- [ ] Current levels gauged
- [ ] Mapped parameters gauged
- [ ] History maintained
- [ ] Prometheus export available

---

#### REQ-NEURO-026: Performance Alerting

**Priority**: Medium
**Description**: The system SHALL alert on neuromodulation performance issues.

```rust
pub struct NeuromodulationAlerts {
    /// Latency threshold for warning
    pub latency_warning_us: u64,  // Default: 150

    /// Latency threshold for critical
    pub latency_critical_us: u64,  // Default: 200

    /// Oscillation detection window
    pub oscillation_window: usize,  // Default: 10

    /// Oscillation threshold
    pub oscillation_threshold: f32,  // Default: 0.3
}

impl NeuromodulationAlerts {
    /// Check for alert conditions
    pub fn check(&self, metrics: &NeuromodulationMetrics) -> Vec<Alert> {
        let mut alerts = Vec::new();

        // Latency check
        let p99_latency = metrics.update_latencies.percentile(0.99);
        if p99_latency > self.latency_critical_us as f64 {
            alerts.push(Alert {
                severity: Severity::Critical,
                message: format!("Neuromodulation P99 latency {}us exceeds 200us", p99_latency),
            });
        } else if p99_latency > self.latency_warning_us as f64 {
            alerts.push(Alert {
                severity: Severity::Warning,
                message: format!("Neuromodulation P99 latency {}us approaching limit", p99_latency),
            });
        }

        // Oscillation check
        if let Some(oscillation) = self.detect_oscillation(metrics) {
            alerts.push(Alert {
                severity: Severity::Warning,
                message: format!("Oscillation detected in {}: amplitude {:.2}", oscillation.modulator, oscillation.amplitude),
            });
        }

        alerts
    }

    fn detect_oscillation(&self, metrics: &NeuromodulationMetrics) -> Option<OscillationInfo> {
        let history = metrics.get_history(self.oscillation_window);
        if history.len() < 3 {
            return None;
        }

        // Check for sign changes in consecutive deltas
        for modulator in [ModulatorType::Dopamine, ModulatorType::Serotonin,
                          ModulatorType::Noradrenaline, ModulatorType::Acetylcholine] {
            let values: Vec<f32> = history.iter().map(|h| match modulator {
                ModulatorType::Dopamine => h.levels.dopamine,
                ModulatorType::Serotonin => h.levels.serotonin,
                ModulatorType::Noradrenaline => h.levels.noradrenaline,
                ModulatorType::Acetylcholine => h.levels.acetylcholine,
            }).collect();

            let mut sign_changes = 0;
            let mut max_amplitude = 0.0f32;
            for i in 2..values.len() {
                let delta1 = values[i-1] - values[i-2];
                let delta2 = values[i] - values[i-1];
                if delta1 * delta2 < 0.0 {
                    sign_changes += 1;
                    max_amplitude = max_amplitude.max(delta1.abs() + delta2.abs());
                }
            }

            if sign_changes >= 3 && max_amplitude > self.oscillation_threshold {
                return Some(OscillationInfo {
                    modulator,
                    amplitude: max_amplitude,
                    frequency: sign_changes,
                });
            }
        }

        None
    }
}

#[derive(Clone)]
pub struct OscillationInfo {
    pub modulator: ModulatorType,
    pub amplitude: f32,
    pub frequency: u32,
}
```

**Acceptance Criteria**:
- [ ] Latency alerts at 150us (warning) and 200us (critical)
- [ ] Oscillation detection implemented
- [ ] Alerts include actionable information
- [ ] Alert history maintained
- [ ] Integration with monitoring system

---

## 3. Non-Functional Requirements

### 3.1 Performance Requirements

#### REQ-NEURO-027: Update Latency Budget

**Priority**: Critical
**Description**: The system SHALL complete neuromodulation updates within 200 microseconds.

| Operation | Budget | Measurement |
|-----------|--------|-------------|
| Trigger checks | <50us | All 4 conditions |
| Target calculation | <20us | Including triggers |
| Hysteresis | <10us | Per modulator |
| Interpolation | <20us | All 4 channels |
| Bounds check | <5us | All 4 channels |
| Parameter mapping | <50us | All 4 mappings |
| Metrics | <10us | Recording only |
| **Total** | **<200us** | End-to-end |

**Acceptance Criteria**:
- [ ] P95 latency under 150 microseconds
- [ ] P99 latency under 200 microseconds
- [ ] No blocking operations in update path
- [ ] Lock contention minimized
- [ ] Allocation-free hot path

---

#### REQ-NEURO-028: Memory Footprint

**Priority**: High
**Description**: The system SHALL minimize memory overhead per session.

| Component | Budget | Notes |
|-----------|--------|-------|
| Controller state | <500 bytes | Core modulator values |
| Trigger state | <200 bytes | Per trigger |
| Hysteresis state | <100 bytes | Direction tracking |
| History buffer | <1KB | Recent updates |
| **Total per session** | **<2KB** | Excluding metrics |

**Acceptance Criteria**:
- [ ] Per-session overhead under 2KB
- [ ] No dynamic allocation in hot path
- [ ] History bounded by ring buffer
- [ ] Metrics use fixed-size structures

---

### 3.2 Reliability Requirements

#### REQ-NEURO-029: Graceful Degradation

**Priority**: High
**Description**: The system SHALL degrade gracefully on component failure.

```rust
impl NeuromodulationController {
    /// Handle component unavailability
    pub fn handle_unavailable_component(&mut self, component: SystemParameter) {
        match component {
            SystemParameter::HopfieldBeta => {
                // Use static beta if Hopfield unavailable
                warn!("Hopfield unavailable, using static beta");
                self.dopamine = 0.5;  // Force to baseline
            }
            SystemParameter::FuseMoeTopK => {
                // Use default top_k if FuseMoE unavailable
                warn!("FuseMoE unavailable, using default top_k");
                self.serotonin = 0.5;  // Force to baseline
            }
            SystemParameter::AttentionTemperature => {
                // Use default temperature if attention unavailable
                warn!("Attention unavailable, using default temperature");
                self.noradrenaline = 0.3;  // Force to baseline
            }
            SystemParameter::UtlLearningRate => {
                // Use default learning rate if UTL unavailable
                warn!("UTL unavailable, using default learning rate");
                self.acetylcholine = 0.4;  // Force to baseline
            }
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Component failure does not crash system
- [ ] Fallback to baseline values on failure
- [ ] Warnings logged for failed components
- [ ] Automatic recovery when component available
- [ ] No data corruption on failure

---

### 3.3 Configuration Requirements

#### REQ-NEURO-030: Configuration Schema

**Priority**: High
**Description**: The system SHALL support TOML-based neuromodulation configuration.

```toml
[neuromodulation]
enabled = true

[neuromodulation.baseline]
dopamine = 0.5
serotonin = 0.5
noradrenaline = 0.3
acetylcholine = 0.4

[neuromodulation.update]
rate = 0.1
interval_ms = 50

[neuromodulation.mapping.dopamine]
output_min = 1.0
output_max = 5.0
curve = "linear"

[neuromodulation.mapping.serotonin]
output_min = 2
output_max = 8
rounding = "nearest"

[neuromodulation.mapping.noradrenaline]
output_min = 0.5
output_max = 2.0

[neuromodulation.mapping.acetylcholine]
output_min = 0.001
output_max = 0.002
log_scale = false

[neuromodulation.triggers.surprise]
threshold = 0.7
surge_magnitude = 0.3
decay_rate = 0.05
cooldown_seconds = 5

[neuromodulation.triggers.uncertainty]
entropy_threshold = 0.6
coherence_threshold = 0.4
increase_magnitude = 0.25
duration_threshold_seconds = 2

[neuromodulation.triggers.exploration]
entropy_threshold = 0.7
boost_magnitude = 0.3
quadrants = ["blind", "unknown"]

[neuromodulation.triggers.recall]
coherence_threshold = 0.3
importance_threshold = 0.7
increase_magnitude = 0.2
age_factor = 0.1

[neuromodulation.hysteresis]
dead_zone = 0.02
direction_change_delay_ms = 500
momentum = 0.3

[neuromodulation.interpolation]
max_delta = 0.05
curve = "linear"

[neuromodulation.dream]
nrem_dopamine = 0.3
nrem_serotonin = 0.4
nrem_noradrenaline = 0.2
nrem_acetylcholine = 0.2
rem_dopamine = 0.7
rem_serotonin = 0.8
rem_noradrenaline = 0.6
rem_acetylcholine = 0.6
transition_duration_ms = 500
```

**Acceptance Criteria**:
- [ ] All parameters configurable via TOML
- [ ] Configuration validated on load
- [ ] Invalid config returns clear error
- [ ] Hot-reload supported for some parameters
- [ ] Defaults match constitution.yaml

---

## 4. Testing Requirements

### 4.1 Unit Tests

#### REQ-NEURO-031: Unit Test Coverage

**Priority**: High
**Description**: The system SHALL have comprehensive unit test coverage.

| Component | Target Coverage |
|-----------|-----------------|
| Parameter Mapping | 95% |
| Trigger Conditions | 95% |
| Interpolation | 90% |
| Hysteresis | 95% |
| Bounds Checking | 100% |
| Update Cycle | 90% |

**Acceptance Criteria**:
- [ ] Coverage targets met
- [ ] Edge cases tested
- [ ] Boundary values tested
- [ ] Error conditions tested
- [ ] Tests run under 10 seconds

---

### 4.2 Integration Tests

#### REQ-NEURO-032: Integration Test Scenarios

**Priority**: High
**Description**: The system SHALL pass integration test scenarios.

**Scenarios**:
1. Full update cycle with all triggers
2. Hopfield integration with dopamine modulation
3. FuseMoE integration with serotonin modulation
4. Attention integration with noradrenaline modulation
5. UTL integration with acetylcholine modulation
6. Dream phase coordination
7. MCP tool round-trip
8. Hysteresis oscillation prevention

**Acceptance Criteria**:
- [ ] All scenarios pass
- [ ] Latency requirements verified
- [ ] Parameter bounds enforced
- [ ] Integration behavior correct
- [ ] No race conditions

---

### 4.3 Benchmark Tests

#### REQ-NEURO-033: Performance Benchmarks

**Priority**: High
**Description**: The system SHALL meet performance benchmarks.

| Benchmark | Target | Measurement |
|-----------|--------|-------------|
| Single update cycle | <200us | P99 latency |
| Parameter mapping | <50us | All 4 mappings |
| Trigger check | <50us | All 4 conditions |
| Interpolation step | <20us | All 4 channels |
| 1000 updates/sec | Sustained | Throughput |

**Acceptance Criteria**:
- [ ] Benchmarks automated in CI
- [ ] Regression detection enabled
- [ ] Historical trends tracked
- [ ] Performance reports generated

---

## 5. Error Handling

### 5.1 Error Codes

#### REQ-NEURO-034: Error Code Catalog

**Priority**: High
**Description**: The system SHALL use consistent error codes.

| Code | Name | Description |
|------|------|-------------|
| -32200 | NeuromodulationDisabled | Neuromodulation feature disabled |
| -32201 | NeuromodulationTimeout | Update exceeded latency budget |
| -32202 | InvalidModulatorLevel | Level outside [0,1] bounds |
| -32203 | ComponentUnavailable | Target component not available |
| -32204 | ConfigurationError | Invalid neuromodulation config |
| -32205 | IntegrationError | Failed to integrate with component |

**Acceptance Criteria**:
- [ ] All errors mapped to codes
- [ ] Error messages descriptive
- [ ] Recovery hints included
- [ ] Errors logged with context

---

## 6. Acceptance Criteria Summary

### 6.1 Critical Acceptance Criteria

1. [ ] 4 neuromodulator channels implemented (dopamine, serotonin, noradrenaline, acetylcholine)
2. [ ] Parameter mapping matches constitution.yaml ranges exactly
3. [ ] Update latency <200 microseconds per query
4. [ ] All 4 trigger conditions implemented and tested
5. [ ] Hysteresis prevents oscillation (>90% reduction)
6. [ ] Smooth interpolation with max delta 0.05
7. [ ] Integration with Hopfield, FuseMoE, Attention, UTL verified
8. [ ] MCP get_neuromodulation tool functional
9. [ ] Dream layer coordination working
10. [ ] Steering dopamine feedback integration with Module 12.5 (Marblestone)

### 6.2 Quality Gates

| Gate | Criteria |
|------|----------|
| Code Review | All code reviewed and approved |
| Unit Tests | 90% coverage, all passing |
| Integration Tests | All scenarios passing |
| Performance | All benchmarks met |
| Documentation | API docs complete |

---

## 7. References

### 7.1 Internal References

- constitution.yaml: neuromodulation section (lines 742-767)
- contextprd.md: Section 7.2 Neuromodulation
- Module 5: UTL Integration (dependency)
- Module 6: Bio-Nervous System (dependency)
- Module 9: Dream Layer (dependency)

### 7.2 External Research

- Neuromodulation in DNNs: https://www.cell.com/trends/neurosciences/abstract/S0166-2236(21)00256-3
- Homeostatic Plasticity: https://elifesciences.org/articles/88376

---

## 8. Appendix

### 8.1 Requirement Traceability Matrix

| Requirement ID | PRD Section | Constitution Reference | Test Case |
|---------------|-------------|----------------------|-----------|
| REQ-NEURO-001 | 7.2 | neuromodulation | T-NEURO-001 |
| REQ-NEURO-002 | 7.2 | neuromodulation.modulators | T-NEURO-002 |
| REQ-NEURO-003 | 7.2 | dopamine -> hopfield.beta | T-NEURO-003 |
| REQ-NEURO-004 | 7.2 | serotonin -> fuse_moe.top_k | T-NEURO-004 |
| REQ-NEURO-005 | 7.2 | noradrenaline -> attention.temperature | T-NEURO-005 |
| REQ-NEURO-006 | 7.2 | acetylcholine -> utl.learning_rate | T-NEURO-006 |
| REQ-NEURO-007 | 7.2 | Unified mapping | T-NEURO-007 |
| REQ-NEURO-008 | 7.2 | High surprise trigger | T-NEURO-008 |
| REQ-NEURO-009 | 7.2 | Uncertainty trigger | T-NEURO-009 |
| REQ-NEURO-010 | 7.2 | Exploration trigger | T-NEURO-010 |
| REQ-NEURO-011 | 7.2 | Deep recall trigger | T-NEURO-011 |
| REQ-NEURO-012 | 7.2 | Trigger source tracking | T-NEURO-012 |
| REQ-NEURO-013 | 7.2 | Smooth interpolation | T-NEURO-013 |
| REQ-NEURO-014 | 7.2 | Hysteresis | T-NEURO-014 |
| REQ-NEURO-015 | 7.2 | Bounds checking | T-NEURO-015 |
| REQ-NEURO-016 | 7.2 | Per-query update | T-NEURO-016 |
| REQ-NEURO-017 | 7.2 | Baseline return | T-NEURO-017 |
| REQ-NEURO-018 | 7.2 | Hopfield integration | T-NEURO-018 |
| REQ-NEURO-019 | 7.2 | FuseMoE integration | T-NEURO-019 |
| REQ-NEURO-020 | 7.2 | Attention integration | T-NEURO-020 |
| REQ-NEURO-021 | 7.2 | UTL integration | T-NEURO-021 |
| REQ-NEURO-022 | 7.2/7.1 | Dream integration | T-NEURO-022 |
| REQ-NEURO-023 | 7.2 | get_neuromodulation tool | T-NEURO-023 |
| REQ-NEURO-024 | 7.2 | Cognitive Pulse integration | T-NEURO-024 |
| REQ-NEURO-025 | 7.2 | Metrics collection | T-NEURO-025 |
| REQ-NEURO-026 | 7.2 | Performance alerting | T-NEURO-026 |
| REQ-NEURO-027 | 7.2 | performance_budgets.latency | T-NEURO-027 |
| REQ-NEURO-028 | - | Memory footprint | T-NEURO-028 |
| REQ-NEURO-029 | - | Graceful degradation | T-NEURO-029 |
| REQ-NEURO-030 | - | Configuration | T-NEURO-030 |
| REQ-NEURO-031 | - | Unit tests | T-NEURO-031 |
| REQ-NEURO-032 | - | Integration tests | T-NEURO-032 |
| REQ-NEURO-033 | - | Benchmarks | T-NEURO-033 |
| REQ-NEURO-034 | - | Error codes | T-NEURO-034 |
| REQ-NEURO-035 | 7.2/12.5 | Steering dopamine feedback (Marblestone) | T-NEURO-035 |
| REQ-NEURO-036 | 7.2/12.5 | Positive steering reward increases dopamine by reward * 0.2 | T-NEURO-036 |
| REQ-NEURO-037 | 7.2/12.5 | Negative steering reward decreases dopamine by |reward| * 0.1 | T-NEURO-037 |

---

*Document Version: 1.1.0*
*Generated: 2025-12-31*
*Agent: #10/28 - Neuromodulation Specification*
