# Module 10: Neuromodulation - Technical Specification

```yaml
metadata:
  id: TECH-NEURO-010
  version: 1.0.0
  module: Neuromodulation
  phase: 9
  status: draft
  created: 2025-12-31
  dependencies:
    - TECH-DREAM-009 (Module 9: Dream Layer)
    - TECH-UTL-005 (Module 5: UTL Integration)
    - TECH-BIONV-006 (Module 6: Bio-Nervous System)
    - SPEC-STEER-012.5 (Module 12.5: Steering Subsystem)
  functional_spec_ref: SPEC-NEURO-010
  author: Architecture Agent
```

---

## 1. Architecture Overview

### 1.1 4-Channel Neuromodulation Architecture

The Neuromodulation module implements a biologically-inspired parameter modulation system with four virtual neuromodulator channels that dynamically adjust system behavior based on cognitive state.

| Channel | Biological Role | System Parameter | Output Range | Baseline |
|---------|----------------|------------------|--------------|----------|
| Dopamine | Reward prediction error | `hopfield.beta` | [1.0, 5.0] | 0.5 |
| Serotonin | Temporal discounting | `fuse_moe.top_k` | [2, 8] | 0.5 |
| Noradrenaline | Arousal/Surprise | `attention.temperature` | [0.5, 2.0] | 0.3 |
| Acetylcholine | Learning rate modulation | `utl.learning_rate` | [0.001, 0.002] | 0.4 |

### 1.2 Performance Budget (<200us Total)

| Operation | Budget | Implementation |
|-----------|--------|----------------|
| Trigger checks | <50us | All 4 conditions |
| Target calculation | <20us | Including triggers |
| Hysteresis | <10us | Per modulator |
| Interpolation | <20us | All 4 channels |
| Bounds check | <5us | All 4 channels |
| Parameter mapping | <50us | All 4 mappings |
| Metrics | <10us | Recording only |
| **Total** | **<200us** | End-to-end |

### 1.3 Module Structure

```
crates/context-graph-neuromod/src/
├── lib.rs                       # Public API
├── controller.rs                # NeuromodulationController
├── channel/
│   ├── mod.rs                   # Channel abstraction
│   ├── dopamine.rs              # Dopamine channel + steering integration
│   ├── serotonin.rs             # Serotonin channel
│   ├── noradrenaline.rs         # Noradrenaline channel
│   └── acetylcholine.rs         # Acetylcholine channel
├── trigger/
│   ├── mod.rs                   # Trigger trait and registry
│   ├── surprise.rs              # High surprise dopamine surge
│   ├── uncertainty.rs           # Uncertainty noradrenaline increase
│   ├── exploration.rs           # Exploration serotonin boost
│   ├── recall.rs                # Deep recall acetylcholine rise
│   └── steering.rs              # Steering dopamine feedback (Marblestone)
├── mapping/
│   ├── mod.rs                   # ParameterMapping unified interface
│   ├── hopfield.rs              # Dopamine -> beta mapping
│   ├── fusemoe.rs               # Serotonin -> top_k mapping
│   ├── attention.rs             # Noradrenaline -> temperature mapping
│   └── utl.rs                   # Acetylcholine -> learning_rate mapping
├── interpolation.rs             # Smooth parameter transitions
├── hysteresis.rs                # Oscillation prevention
├── bounds.rs                    # Bounds checking and soft limiting
├── state_machine.rs             # Modulator state transitions
├── integration/
│   ├── hopfield.rs              # Hopfield network integration
│   ├── fusemoe.rs               # FuseMoE integration
│   ├── attention.rs             # Attention mechanism integration
│   ├── utl.rs                   # UTL optimizer integration
│   └── dream.rs                 # Dream layer coordination
├── metrics.rs                   # NeuromodulationMetrics
├── mcp/
│   └── tools.rs                 # get_neuromodulation MCP tool
└── config.rs                    # TOML configuration
```

---

## 2. Core Data Structures

### 2.1 NeuromodulationController

```rust
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// REQ-NEURO-001: Controls 4 neuromodulator channels that dynamically adjust
/// system parameters based on cognitive state and environmental triggers.
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
    pub baseline: NeuromodulatorBaseline,

    /// Current trigger state
    trigger_state: TriggerState,

    /// Hysteresis configuration to prevent oscillation
    hysteresis: HysteresisConfig,

    /// Parameter mapping configuration
    pub parameter_map: ParameterMapping,

    /// Interpolation engine
    interpolation: InterpolationEngine,

    /// Bounds checker
    bounds_checker: BoundsChecker,

    /// Trigger registry
    triggers: TriggerRegistry,

    /// Last steering feedback (Marblestone)
    pub last_steering_feedback: Option<SteeringDopamineFeedback>,

    /// Last dopamine surge timestamp
    last_dopamine_surge: Instant,

    /// Metrics for monitoring
    pub metrics: NeuromodulationMetrics,

    /// Dream phase targets (when in dream state)
    dream_targets: Option<ModulatorTargets>,
}

impl NeuromodulationController {
    pub fn new(config: NeuromodulationConfig) -> Self {
        Self {
            dopamine: config.baseline.dopamine,
            serotonin: config.baseline.serotonin,
            noradrenaline: config.baseline.noradrenaline,
            acetylcholine: config.baseline.acetylcholine,
            update_rate: config.update_rate,
            baseline: config.baseline,
            trigger_state: TriggerState::default(),
            hysteresis: config.hysteresis,
            parameter_map: ParameterMapping::from_constitution(),
            interpolation: InterpolationEngine::new(config.interpolation),
            bounds_checker: BoundsChecker::default(),
            triggers: TriggerRegistry::default(),
            last_steering_feedback: None,
            last_dopamine_surge: Instant::now(),
            metrics: NeuromodulationMetrics::default(),
            dream_targets: None,
        }
    }
}

/// Baseline levels modulators return to over time
#[derive(Clone, Copy, Debug)]
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

### 2.2 Neuromodulator Channel Definition

```rust
/// REQ-NEURO-002: Individual neuromodulator channel with biological context
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Modulator type enumeration
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum ModulatorType {
    Dopamine,
    Serotonin,
    Noradrenaline,
    Acetylcholine,
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

---

## 3. Parameter Mapping Implementation

### 3.1 Dopamine to Hopfield Beta (REQ-NEURO-003)

```rust
/// Maps dopamine level [0,1] to hopfield.beta [1.0, 5.0]
///
/// High dopamine -> High beta -> Sharp retrieval (exploitation)
/// Low dopamine -> Low beta -> Diffuse retrieval (exploration)
pub struct DopamineMapper {
    pub input_range: (f32, f32),   // (0.0, 1.0)
    pub output_range: (f32, f32),  // (1.0, 5.0)
    pub curve: MappingCurve,
}

pub enum MappingCurve {
    Linear,
    Sigmoid { steepness: f32 },
    Exponential { base: f32 },
}

impl DopamineMapper {
    /// `Constraint: Mapping_Latency < 10us`
    #[inline]
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

### 3.2 Serotonin to FuseMoE Top-K (REQ-NEURO-004)

```rust
/// Maps serotonin level [0,1] to fuse_moe.top_k [2, 8]
///
/// High serotonin -> High top_k -> More experts consulted (exploration)
/// Low serotonin -> Low top_k -> Fewer experts consulted (exploitation)
pub struct SerotoninMapper {
    pub input_range: (f32, f32),   // (0.0, 1.0)
    pub output_range: (u32, u32),  // (2, 8)
    pub rounding: RoundingStrategy,
}

pub enum RoundingStrategy {
    Floor,
    Ceiling,
    Nearest,
    Probabilistic,  // Probability proportional to fractional part
}

impl SerotoninMapper {
    /// `Constraint: Mapping_Latency < 10us`
    #[inline]
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
                if fastrand::f32() < frac {
                    (base + 1).min(self.output_range.1)
                } else {
                    base
                }
            }
        }
        .clamp(self.output_range.0, self.output_range.1)
    }
}
```

### 3.3 Noradrenaline to Attention Temperature (REQ-NEURO-005)

```rust
/// Maps noradrenaline level [0,1] to attention.temperature [0.5, 2.0]
///
/// High noradrenaline -> High temperature -> Flat/uniform attention (exploration)
/// Low noradrenaline -> Low temperature -> Sharp/focused attention (exploitation)
pub struct NoradrenalineMapper {
    pub input_range: (f32, f32),   // (0.0, 1.0)
    pub output_range: (f32, f32),  // (0.5, 2.0)
    pub inverse: bool,
}

impl NoradrenalineMapper {
    /// `Constraint: Mapping_Latency < 10us`
    #[inline]
    pub fn map(&self, noradrenaline: f32) -> f32 {
        let clamped = noradrenaline.clamp(0.0, 1.0);
        self.output_range.0 + clamped * (self.output_range.1 - self.output_range.0)
    }
}
```

### 3.4 Acetylcholine to UTL Learning Rate (REQ-NEURO-006)

```rust
/// Maps acetylcholine level [0,1] to utl.learning_rate [0.001, 0.002]
///
/// High acetylcholine -> High learning rate -> Faster memory updates
/// Low acetylcholine -> Low learning rate -> Slower, more stable updates
pub struct AcetylcholineMapper {
    pub input_range: (f32, f32),   // (0.0, 1.0)
    pub output_range: (f32, f32),  // (0.001, 0.002)
    pub log_scale: bool,
}

impl AcetylcholineMapper {
    /// `Constraint: Mapping_Latency < 10us`
    #[inline]
    pub fn map(&self, acetylcholine: f32) -> f32 {
        let clamped = acetylcholine.clamp(0.0, 1.0);

        if self.log_scale {
            // Logarithmic interpolation for learning rates
            let log_min = self.output_range.0.ln();
            let log_max = self.output_range.1.ln();
            let log_result = log_min + clamped * (log_max - log_min);
            log_result.exp()
        } else {
            self.output_range.0 + clamped * (self.output_range.1 - self.output_range.0)
        }
    }
}
```

### 3.5 Unified Parameter Mapping (REQ-NEURO-007)

```rust
/// Unified interface for all parameter mappings
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
    #[inline]
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
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct MappedParameters {
    pub hopfield_beta: f32,
    pub fuse_moe_top_k: u32,
    pub attention_temperature: f32,
    pub utl_learning_rate: f32,
}
```

---

## 4. Trigger Conditions Implementation

### 4.1 Trigger Source Enumeration (REQ-NEURO-012)

```rust
/// Source of neuromodulator trigger
#[derive(Clone, Debug, serde::Serialize)]
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
    SteeringFeedback { reward: f32, source: SteeringSource },
}

/// Record of a trigger event
#[derive(Clone, Debug, serde::Serialize)]
pub struct TriggerEvent {
    pub timestamp: DateTime<Utc>,
    pub modulator: ModulatorType,
    pub source: TriggerSource,
    pub magnitude: f32,
    pub resulting_level: f32,
}

impl TriggerEvent {
    pub fn dopamine(surge: DopamineSurge) -> Self {
        Self {
            timestamp: Utc::now(),
            modulator: ModulatorType::Dopamine,
            source: surge.trigger_source,
            magnitude: surge.magnitude,
            resulting_level: 0.0, // Set during application
        }
    }

    pub fn serotonin(boost: SerotoninBoost) -> Self {
        Self {
            timestamp: Utc::now(),
            modulator: ModulatorType::Serotonin,
            source: boost.trigger_source,
            magnitude: boost.magnitude,
            resulting_level: 0.0,
        }
    }

    pub fn noradrenaline(increase: NoradrenalineIncrease) -> Self {
        Self {
            timestamp: Utc::now(),
            modulator: ModulatorType::Noradrenaline,
            source: increase.trigger_source,
            magnitude: increase.magnitude,
            resulting_level: 0.0,
        }
    }

    pub fn acetylcholine(rise: AcetylcholineRise) -> Self {
        Self {
            timestamp: Utc::now(),
            modulator: ModulatorType::Acetylcholine,
            source: rise.trigger_source,
            magnitude: rise.magnitude,
            resulting_level: 0.0,
        }
    }
}
```

### 4.2 High Surprise Dopamine Surge (REQ-NEURO-008)

```rust
/// Trigger dopamine surge on high surprise events
pub struct SurpriseTrigger {
    /// Surprise threshold for dopamine surge (delta_s)
    pub surprise_threshold: f32,  // Default: 0.7

    /// Magnitude of dopamine increase
    pub surge_magnitude: f32,     // Default: 0.3

    /// Decay rate after surge
    pub decay_rate: f32,          // Default: 0.05 per update

    /// Cooldown between surges
    pub cooldown: Duration,       // Default: 5 seconds
}

impl Default for SurpriseTrigger {
    fn default() -> Self {
        Self {
            surprise_threshold: 0.7,
            surge_magnitude: 0.3,
            decay_rate: 0.05,
            cooldown: Duration::from_secs(5),
        }
    }
}

impl SurpriseTrigger {
    /// Check if surprise level triggers dopamine surge
    ///
    /// `Constraint: Trigger_Check_Latency < 10us`
    #[inline]
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

### 4.3 Uncertainty Noradrenaline Increase (REQ-NEURO-009)

```rust
/// Trigger noradrenaline increase during high uncertainty
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

impl Default for UncertaintyTrigger {
    fn default() -> Self {
        Self {
            entropy_threshold: 0.6,
            coherence_threshold: 0.4,
            increase_magnitude: 0.25,
            duration_threshold: Duration::from_secs(2),
        }
    }
}

impl UncertaintyTrigger {
    /// `Constraint: Trigger_Check_Latency < 10us`
    #[inline]
    pub fn check(&self, utl_state: &UtlState, uncertainty_duration: Duration) -> Option<NoradrenalineIncrease> {
        let is_uncertain = utl_state.entropy > self.entropy_threshold
            && utl_state.coherence < self.coherence_threshold;

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

### 4.4 Exploration Mode Serotonin Boost (REQ-NEURO-010)

```rust
/// Trigger serotonin boost when exploration mode is needed
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

impl Default for ExplorationTrigger {
    fn default() -> Self {
        Self {
            exploration_quadrants: vec![JohariQuadrant::Blind, JohariQuadrant::Unknown],
            entropy_threshold: 0.7,
            boost_magnitude: 0.3,
            duration_threshold: Duration::from_secs(3),
        }
    }
}

impl ExplorationTrigger {
    /// `Constraint: Trigger_Check_Latency < 10us`
    #[inline]
    pub fn check(&self, cognitive_state: &CognitiveState) -> Option<SerotoninBoost> {
        let in_exploration_quadrant = self.exploration_quadrants
            .contains(&cognitive_state.johari_quadrant);

        let high_entropy = cognitive_state.entropy > self.entropy_threshold;

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

### 4.5 Deep Recall Acetylcholine Rise (REQ-NEURO-011)

```rust
/// Trigger acetylcholine rise when deep recall is needed
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

impl Default for DeepRecallTrigger {
    fn default() -> Self {
        Self {
            coherence_threshold: 0.3,
            importance_threshold: 0.7,
            increase_magnitude: 0.2,
            age_factor: 0.1,
        }
    }
}

impl DeepRecallTrigger {
    /// `Constraint: Trigger_Check_Latency < 10us`
    #[inline]
    pub fn check(&self, recall_context: &RecallContext) -> Option<AcetylcholineRise> {
        let needs_deep_recall = recall_context.coherence < self.coherence_threshold
            && recall_context.target_importance > self.importance_threshold;

        if needs_deep_recall {
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

---

## 5. Steering Dopamine Feedback (Marblestone)

### 5.1 SteeringDopamineFeedback (REQ-NEURO-035/036/037)

```rust
/// Steering dopamine feedback from external sources (Marblestone)
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
    /// Target edges for this feedback (optional)
    pub target_edges: Vec<Uuid>,
    /// Signal strength for modulation [-1.0, 1.0]
    pub signal_strength: f32,
}

/// Source of steering feedback
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub enum SteeringSource {
    /// Gardener component (long-term curation)
    Gardener,
    /// Curator component (quality assessment)
    Curator,
    /// Thought Assessor (immediate evaluation)
    ThoughtAssessor,
    /// External API-provided feedback
    External,
}

impl NeuromodulationController {
    /// Apply steering feedback to dopamine levels (Marblestone)
    ///
    /// Positive rewards increase dopamine (reinforce behavior)
    /// Negative rewards decrease dopamine (discourage behavior)
    ///
    /// # Effects
    /// - reward > 0: dopamine += reward * 0.2 (capped at 1.0)
    /// - reward < 0: dopamine += reward * 0.1 (floored at 0.0)
    ///
    /// `Constraint: Steering_Feedback_Latency < 20us`
    #[inline]
    pub fn apply_steering_feedback(&mut self, feedback: &SteeringDopamineFeedback) {
        let delta = if feedback.reward > 0.0 {
            feedback.reward * 0.2  // Positive reinforcement (REQ-NEURO-036)
        } else {
            feedback.reward * 0.1  // Negative feedback, gentler (REQ-NEURO-037)
        };

        self.dopamine = (self.dopamine + delta).clamp(0.0, 1.0);

        // Store for tracking
        self.last_steering_feedback = Some(feedback.clone());

        // Record trigger event
        self.metrics.record_steering_feedback(feedback, self.dopamine);
    }

    /// Get steering-modulated dopamine level
    #[inline]
    pub fn get_steered_dopamine(&self) -> f32 {
        self.dopamine
    }

    /// Check if recent steering feedback was applied
    pub fn has_recent_steering_feedback(&self, within: Duration) -> bool {
        self.last_steering_feedback
            .as_ref()
            .map(|f| {
                let feedback_time = f.timestamp;
                let now = Utc::now();
                (now - feedback_time).num_milliseconds() < within.as_millis() as i64
            })
            .unwrap_or(false)
    }
}

/// Builder for SteeringDopamineFeedback
impl SteeringDopamineFeedback {
    pub fn new(reward: f32, source: SteeringSource) -> Self {
        Self {
            reward: reward.clamp(-1.0, 1.0),
            source,
            timestamp: Utc::now(),
            thought_id: None,
            target_edges: Vec::new(),
            signal_strength: reward.clamp(-1.0, 1.0),
        }
    }

    pub fn with_thought_id(mut self, id: Uuid) -> Self {
        self.thought_id = Some(id);
        self
    }

    pub fn with_target_edges(mut self, edges: Vec<Uuid>) -> Self {
        self.target_edges = edges;
        self
    }
}
```

---

## 6. Modulator State Machine

### 6.1 State Transitions

```rust
/// Modulator state machine for tracking activation patterns
#[derive(Clone, Debug)]
pub struct ModulatorStateMachine {
    /// Current state per modulator
    states: [ModulatorState; 4],

    /// Transition history
    history: RingBuffer<StateTransition, 100>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModulatorState {
    /// At or near baseline
    Baseline,
    /// Rising toward target
    Rising { target: f32, rate: f32 },
    /// At elevated/depressed level
    Active { level: f32, since: Instant },
    /// Falling toward baseline
    Falling { rate: f32 },
    /// Suppressed by cross-modulator interaction
    Suppressed { by: ModulatorType },
}

#[derive(Clone, Debug)]
pub struct StateTransition {
    pub modulator: ModulatorType,
    pub from: ModulatorState,
    pub to: ModulatorState,
    pub trigger: TriggerSource,
    pub timestamp: Instant,
}

impl ModulatorStateMachine {
    /// Transition modulator to new state
    pub fn transition(&mut self, modulator: ModulatorType, new_state: ModulatorState, trigger: TriggerSource) {
        let idx = modulator as usize;
        let from = self.states[idx];

        self.history.push(StateTransition {
            modulator,
            from,
            to: new_state,
            trigger,
            timestamp: Instant::now(),
        });

        self.states[idx] = new_state;
    }

    /// Get current state for modulator
    pub fn get_state(&self, modulator: ModulatorType) -> ModulatorState {
        self.states[modulator as usize]
    }

    /// Check if modulator is active (above baseline)
    pub fn is_active(&self, modulator: ModulatorType) -> bool {
        matches!(self.states[modulator as usize], ModulatorState::Active { .. } | ModulatorState::Rising { .. })
    }
}
```

---

## 7. Cross-Modulator Interactions

### 7.1 Interaction Rules

```rust
/// Cross-modulator interaction rules
pub struct CrossModulatorInteractions {
    /// Rules for modulator interactions
    rules: Vec<InteractionRule>,
}

#[derive(Clone, Debug)]
pub struct InteractionRule {
    /// Source modulator
    pub source: ModulatorType,
    /// Target modulator
    pub target: ModulatorType,
    /// Interaction type
    pub interaction: InteractionType,
    /// Strength of interaction [0.0, 1.0]
    pub strength: f32,
    /// Threshold for source to trigger interaction
    pub threshold: f32,
}

#[derive(Clone, Debug)]
pub enum InteractionType {
    /// Source inhibits target (dopamine high -> serotonin suppressed)
    Inhibit,
    /// Source excites target
    Excite,
    /// Source modulates target's sensitivity
    ModulateSensitivity { factor: f32 },
    /// Mutual antagonism
    Antagonize,
}

impl Default for CrossModulatorInteractions {
    fn default() -> Self {
        Self {
            rules: vec![
                // High dopamine slightly inhibits serotonin (exploitation vs exploration)
                InteractionRule {
                    source: ModulatorType::Dopamine,
                    target: ModulatorType::Serotonin,
                    interaction: InteractionType::Inhibit,
                    strength: 0.2,
                    threshold: 0.7,
                },
                // High noradrenaline enhances acetylcholine (alertness -> learning)
                InteractionRule {
                    source: ModulatorType::Noradrenaline,
                    target: ModulatorType::Acetylcholine,
                    interaction: InteractionType::Excite,
                    strength: 0.15,
                    threshold: 0.6,
                },
                // High serotonin moderates noradrenaline (calm exploration)
                InteractionRule {
                    source: ModulatorType::Serotonin,
                    target: ModulatorType::Noradrenaline,
                    interaction: InteractionType::ModulateSensitivity { factor: 0.8 },
                    strength: 0.1,
                    threshold: 0.7,
                },
            ],
        }
    }
}

impl CrossModulatorInteractions {
    /// Apply cross-modulator interactions
    ///
    /// `Constraint: Interaction_Check_Latency < 15us`
    #[inline]
    pub fn apply(&self, levels: &mut [f32; 4]) {
        for rule in &self.rules {
            let source_level = levels[rule.source as usize];

            if source_level >= rule.threshold {
                let target_idx = rule.target as usize;
                let effect = (source_level - rule.threshold) * rule.strength;

                match rule.interaction {
                    InteractionType::Inhibit => {
                        levels[target_idx] = (levels[target_idx] - effect).max(0.0);
                    }
                    InteractionType::Excite => {
                        levels[target_idx] = (levels[target_idx] + effect).min(1.0);
                    }
                    InteractionType::ModulateSensitivity { factor } => {
                        // Reduce deviation from baseline
                        let baseline = 0.5; // Simplified
                        let deviation = levels[target_idx] - baseline;
                        levels[target_idx] = baseline + deviation * factor;
                    }
                    InteractionType::Antagonize => {
                        levels[target_idx] = levels[target_idx] * (1.0 - effect);
                    }
                }
            }
        }
    }
}
```

---

## 8. Smooth Interpolation and Hysteresis

### 8.1 InterpolationEngine (REQ-NEURO-013)

```rust
/// Smooth parameter interpolation between states
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
    pub fn new(config: InterpolationConfig) -> Self {
        Self {
            max_delta: config.max_delta,
            curve: config.curve,
            update_interval: config.update_interval,
        }
    }

    /// Interpolate from current to target level
    ///
    /// `Constraint: Interpolation_Step_Latency < 20us`
    #[inline]
    pub fn step(&self, current: f32, target: f32) -> f32 {
        let diff = target - current;
        let clamped_diff = diff.clamp(-self.max_delta, self.max_delta);

        match self.curve {
            InterpolationCurve::Linear => current + clamped_diff,
            InterpolationCurve::EaseInOut => {
                let t = (clamped_diff.abs() / self.max_delta).min(1.0);
                let eased = t * t * (3.0 - 2.0 * t);
                current + clamped_diff.signum() * eased * self.max_delta
            }
            InterpolationCurve::Exponential { decay } => {
                current + diff * decay
            }
            InterpolationCurve::Spring { stiffness, damping } => {
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

#[derive(Clone, Copy, Debug)]
pub struct ModulatorTargets {
    pub dopamine: f32,
    pub serotonin: f32,
    pub noradrenaline: f32,
    pub acetylcholine: f32,
}
```

### 8.2 HysteresisConfig (REQ-NEURO-014)

```rust
/// Hysteresis configuration to prevent oscillation
pub struct HysteresisConfig {
    /// Dead zone width (no change if within this range)
    pub dead_zone: f32,           // Default: 0.02

    /// Minimum time between direction changes
    pub direction_change_delay: Duration,  // Default: 500ms

    /// Momentum factor (resistance to direction change)
    pub momentum: f32,            // Default: 0.3

    /// Per-modulator state
    states: [HysteresisState; 4],
}

#[derive(Clone, Default)]
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
    #[inline]
    pub fn apply(&mut self, modulator: ModulatorType, current: f32, proposed_target: f32) -> f32 {
        let state = &mut self.states[modulator as usize];
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
        if let (Some(last_dir), Some(last_change)) = (state.last_direction, state.last_direction_change) {
            if new_direction != last_dir && last_change.elapsed() < self.direction_change_delay {
                state.accumulated_pressure += diff.abs();
                if state.accumulated_pressure < self.momentum {
                    return current;
                }
            }
        }

        // Update state
        if Some(new_direction) != state.last_direction {
            state.last_direction = Some(new_direction);
            state.last_direction_change = Some(Instant::now());
            state.accumulated_pressure = 0.0;
        }

        proposed_target
    }
}
```

### 8.3 BoundsChecker (REQ-NEURO-015)

```rust
/// Bounds checker for modulator stability
pub struct BoundsChecker {
    /// Modulator level bounds (always 0.0 to 1.0)
    pub level_bounds: (f32, f32),

    /// Alert threshold for extreme values
    pub alert_threshold: f32,  // Default: 0.9

    /// Soft limiting vs hard clamping
    pub soft_limit: bool,

    /// Soft limit compression factor
    pub compression_factor: f32,  // Default: 0.8
}

impl Default for BoundsChecker {
    fn default() -> Self {
        Self {
            level_bounds: (0.0, 1.0),
            alert_threshold: 0.9,
            soft_limit: true,
            compression_factor: 0.8,
        }
    }
}

impl BoundsChecker {
    /// Check and enforce bounds on modulator level
    ///
    /// `Constraint: Bounds_Check_Latency < 5us`
    #[inline]
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
                severity: if (bounded_level - self.level_bounds.0).abs() < f32::EPSILON
                    || (bounded_level - self.level_bounds.1).abs() < f32::EPSILON {
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
    #[inline]
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

#[derive(Clone, Copy, Debug)]
pub enum Severity {
    Info,
    Warning,
    Error,
}
```

---

## 9. Update Cycle Implementation

### 9.1 Per-Query Update (REQ-NEURO-016)

```rust
impl NeuromodulationController {
    /// Update all neuromodulator levels for current query
    ///
    /// `Constraint: Total_Update_Latency < 200us`
    pub fn update(&mut self, context: &QueryContext) -> UpdateResult {
        let start = Instant::now();

        // 1. Check trigger conditions (~40us)
        let triggers = self.check_triggers(context);

        // 2. Calculate target levels (~20us)
        let targets = self.calculate_targets(&triggers);

        // 3. Apply hysteresis (~10us per modulator)
        let hysteresis_targets = self.apply_hysteresis(&targets);

        // 4. Interpolate toward targets (~20us)
        self.interpolation.step_all(self, &hysteresis_targets);

        // 5. Apply cross-modulator interactions (~15us)
        let mut levels = [self.dopamine, self.serotonin, self.noradrenaline, self.acetylcholine];
        CrossModulatorInteractions::default().apply(&mut levels);
        self.dopamine = levels[0];
        self.serotonin = levels[1];
        self.noradrenaline = levels[2];
        self.acetylcholine = levels[3];

        // 6. Check bounds (~5us per modulator)
        self.enforce_bounds();

        // 7. Map to system parameters (~50us)
        let mapped = self.parameter_map.apply(self);

        // 8. Record metrics (~10us)
        let latency = start.elapsed();
        self.metrics.record_update(latency, &triggers);

        if latency > Duration::from_micros(200) {
            tracing::warn!("Neuromodulation update exceeded budget: {:?}", latency);
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
        let mut triggers = Vec::with_capacity(4);

        if let Some(surge) = self.triggers.surprise.check(&context.utl_state, self.last_dopamine_surge) {
            triggers.push(TriggerEvent::dopamine(surge));
        }

        if let Some(increase) = self.triggers.uncertainty.check(&context.utl_state, context.uncertainty_duration) {
            triggers.push(TriggerEvent::noradrenaline(increase));
        }

        if let Some(boost) = self.triggers.exploration.check(&context.cognitive_state) {
            triggers.push(TriggerEvent::serotonin(boost));
        }

        if let Some(rise) = self.triggers.recall.check(&context.recall_context) {
            triggers.push(TriggerEvent::acetylcholine(rise));
        }

        triggers
    }

    /// Calculate target levels including baseline return (REQ-NEURO-017)
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

        // Apply dream targets if in dream state
        if let Some(ref dream_targets) = self.dream_targets {
            targets = *dream_targets;
        }

        targets
    }

    /// Apply hysteresis to all modulators
    fn apply_hysteresis(&mut self, targets: &ModulatorTargets) -> ModulatorTargets {
        ModulatorTargets {
            dopamine: self.hysteresis.apply(ModulatorType::Dopamine, self.dopamine, targets.dopamine),
            serotonin: self.hysteresis.apply(ModulatorType::Serotonin, self.serotonin, targets.serotonin),
            noradrenaline: self.hysteresis.apply(ModulatorType::Noradrenaline, self.noradrenaline, targets.noradrenaline),
            acetylcholine: self.hysteresis.apply(ModulatorType::Acetylcholine, self.acetylcholine, targets.acetylcholine),
        }
    }

    /// Enforce bounds on all modulators
    fn enforce_bounds(&mut self) {
        self.dopamine = self.bounds_checker.check(self.dopamine).bounded_level;
        self.serotonin = self.bounds_checker.check(self.serotonin).bounded_level;
        self.noradrenaline = self.bounds_checker.check(self.noradrenaline).bounded_level;
        self.acetylcholine = self.bounds_checker.check(self.acetylcholine).bounded_level;
    }

    /// Set dream phase targets
    pub fn set_dream_targets(&mut self, targets: ModulatorTargets) {
        self.dream_targets = Some(targets);
    }

    /// Clear dream targets (return to normal operation)
    pub fn clear_dream_targets(&mut self) {
        self.dream_targets = None;
    }
}

pub struct UpdateResult {
    pub mapped_parameters: MappedParameters,
    pub triggers_fired: Vec<TriggerEvent>,
    pub latency: Duration,
    pub levels: NeuromodulatorLevels,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct NeuromodulatorLevels {
    pub dopamine: f32,
    pub serotonin: f32,
    pub noradrenaline: f32,
    pub acetylcholine: f32,
}
```

---

## 10. Dream Layer Integration (REQ-NEURO-022)

```rust
/// Dream layer integration for phase-specific modulation
pub struct DreamIntegration {
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
    pub fn enter_phase(&self, controller: &mut NeuromodulationController, phase: DreamPhase) {
        let targets = match phase {
            DreamPhase::NREM => self.phase_settings.nrem,
            DreamPhase::REM => self.phase_settings.rem,
            DreamPhase::Transition => return,
            DreamPhase::Awake => {
                ModulatorTargets {
                    dopamine: controller.baseline.dopamine,
                    serotonin: controller.baseline.serotonin,
                    noradrenaline: controller.baseline.noradrenaline,
                    acetylcholine: controller.baseline.acetylcholine,
                }
            }
        };

        controller.set_dream_targets(targets);
    }

    /// Exit dream mode
    pub fn exit_dream(&self, controller: &mut NeuromodulationController) {
        controller.clear_dream_targets();
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DreamPhase {
    NREM,
    REM,
    Transition,
    Awake,
}
```

---

## 11. Metrics and Monitoring

### 11.1 NeuromodulationMetrics (REQ-NEURO-025)

```rust
use std::collections::VecDeque;

/// Comprehensive neuromodulation metrics
#[derive(Default)]
pub struct NeuromodulationMetrics {
    /// Update latency histogram (microseconds)
    update_latencies: LatencyHistogram,

    /// Trigger frequency counters
    trigger_counts: TriggerCounters,

    /// Modulator level history
    level_history: VecDeque<NeuromodulationHistoryEntry>,

    /// History buffer size
    history_max_size: usize,

    /// Steering feedback counters
    steering_feedback_count: u64,
    steering_positive_count: u64,
    steering_negative_count: u64,
}

#[derive(Clone, serde::Serialize)]
pub struct NeuromodulationHistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub levels: NeuromodulatorLevels,
    pub mapped: MappedParameters,
    pub triggers: Vec<TriggerSource>,
    pub latency_us: u64,
}

impl NeuromodulationMetrics {
    pub fn new(history_size: usize) -> Self {
        Self {
            update_latencies: LatencyHistogram::new(),
            trigger_counts: TriggerCounters::default(),
            level_history: VecDeque::with_capacity(history_size),
            history_max_size: history_size,
            steering_feedback_count: 0,
            steering_positive_count: 0,
            steering_negative_count: 0,
        }
    }

    /// Record an update cycle
    pub fn record_update(&mut self, latency: Duration, triggers: &[TriggerEvent]) {
        self.update_latencies.record(latency.as_micros() as u64);

        for trigger in triggers {
            self.trigger_counts.increment(trigger.modulator);
        }
    }

    /// Record steering feedback (Marblestone)
    pub fn record_steering_feedback(&mut self, feedback: &SteeringDopamineFeedback, resulting_dopamine: f32) {
        self.steering_feedback_count += 1;
        if feedback.reward > 0.0 {
            self.steering_positive_count += 1;
        } else if feedback.reward < 0.0 {
            self.steering_negative_count += 1;
        }

        // Could add more detailed tracking here
    }

    /// Update current level gauges
    pub fn update_levels(&mut self, levels: &NeuromodulatorLevels, mapped: &MappedParameters, triggers: &[TriggerEvent]) {
        let entry = NeuromodulationHistoryEntry {
            timestamp: Utc::now(),
            levels: *levels,
            mapped: *mapped,
            triggers: triggers.iter().map(|t| t.source.clone()).collect(),
            latency_us: 0,
        };

        if self.level_history.len() >= self.history_max_size {
            self.level_history.pop_front();
        }
        self.level_history.push_back(entry);
    }

    /// Get recent history
    pub fn get_history(&self, count: usize) -> Vec<NeuromodulationHistoryEntry> {
        self.level_history.iter().rev().take(count).cloned().collect()
    }

    /// Get P95 latency
    pub fn p95_latency_us(&self) -> u64 {
        self.update_latencies.percentile(95)
    }

    /// Get P99 latency
    pub fn p99_latency_us(&self) -> u64 {
        self.update_latencies.percentile(99)
    }
}

/// Simple latency histogram
struct LatencyHistogram {
    buckets: [u64; 20],  // 0-10us, 10-20us, ..., 180-190us, 190+us
    count: u64,
    sum: u64,
}

impl LatencyHistogram {
    fn new() -> Self {
        Self { buckets: [0; 20], count: 0, sum: 0 }
    }

    fn record(&mut self, latency_us: u64) {
        let bucket = (latency_us / 10).min(19) as usize;
        self.buckets[bucket] += 1;
        self.count += 1;
        self.sum += latency_us;
    }

    fn percentile(&self, pct: u8) -> u64 {
        if self.count == 0 { return 0; }
        let target = (self.count as f64 * pct as f64 / 100.0) as u64;
        let mut cumulative = 0u64;
        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return (i as u64 + 1) * 10;
            }
        }
        200
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self { Self::new() }
}

#[derive(Default)]
struct TriggerCounters {
    dopamine: u64,
    serotonin: u64,
    noradrenaline: u64,
    acetylcholine: u64,
}

impl TriggerCounters {
    fn increment(&mut self, modulator: ModulatorType) {
        match modulator {
            ModulatorType::Dopamine => self.dopamine += 1,
            ModulatorType::Serotonin => self.serotonin += 1,
            ModulatorType::Noradrenaline => self.noradrenaline += 1,
            ModulatorType::Acetylcholine => self.acetylcholine += 1,
        }
    }
}
```

---

## 12. MCP Tool Integration (REQ-NEURO-023)

```rust
use serde::{Deserialize, Serialize};

/// MCP tool: get_neuromodulation
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

fn default_history_size() -> usize { 10 }

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

    /// Latency of this query
    pub latency_us: u64,
}

#[derive(Serialize)]
pub struct ActiveTrigger {
    pub modulator: ModulatorType,
    pub source: String,
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

impl BehavioralState {
    pub fn from_levels(levels: &NeuromodulatorLevels) -> Self {
        // Exploration-exploitation balance
        let explore = (levels.serotonin + levels.noradrenaline) / 2.0;
        let exploit = levels.dopamine * (1.0 - levels.noradrenaline);
        let ee_balance = explore - exploit;

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

        Self {
            exploration_exploitation: ee_balance,
            plasticity,
            focus,
            description,
        }
    }
}
```

---

## 13. Configuration

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

[neuromodulation.steering]
enabled = true
positive_scale = 0.2
negative_scale = 0.1
```

---

## 14. Performance Targets Summary

| Metric | Target | Measurement |
|--------|--------|-------------|
| Update Latency P95 | <150us | Per-query timing |
| Update Latency P99 | <200us | Per-query timing |
| Trigger Check | <50us | All 4 conditions |
| Parameter Mapping | <50us | All 4 mappings |
| Interpolation Step | <20us | All 4 channels |
| Hysteresis Check | <10us | Per modulator |
| Bounds Check | <5us | Per modulator |
| Steering Feedback | <20us | Marblestone integration |
| Memory per Session | <2KB | Controller state |
| Oscillation Reduction | >90% | Hysteresis effectiveness |

---

## 15. Dependencies

```toml
[dependencies]
context-graph-core = { path = "../context-graph-core" }
context-graph-utl = { path = "../context-graph-utl" }
tokio = { version = "1.35", features = ["sync", "time"] }
uuid = { version = "1.6", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing = "0.1"
thiserror = "1.0"
fastrand = "2.0"
```

---

## 16. Acceptance Criteria

- [ ] 4 neuromodulator channels implemented (dopamine, serotonin, noradrenaline, acetylcholine)
- [ ] Parameter mapping matches constitution.yaml ranges exactly
- [ ] Update latency <200us P99 per query
- [ ] All 4 trigger conditions implemented and tested
- [ ] Hysteresis prevents oscillation (>90% reduction)
- [ ] Smooth interpolation with max delta 0.05
- [ ] Integration with Hopfield, FuseMoE, Attention, UTL verified
- [ ] MCP get_neuromodulation tool functional
- [ ] Dream layer coordination working (NREM/REM phases)
- [ ] Steering dopamine feedback integration with Module 12.5 (Marblestone REQ-NEURO-035/036/037)
- [ ] Cross-modulator interactions implemented
- [ ] Modulator state machine tracking transitions
- [ ] Metrics collection with P95/P99 latency tracking

---

## 17. Traceability Matrix

| Requirement | Implementation | Test Case |
|-------------|----------------|-----------|
| REQ-NEURO-001 | `NeuromodulationController` | T-NEURO-001 |
| REQ-NEURO-002 | `NeuromodulatorChannel` | T-NEURO-002 |
| REQ-NEURO-003 | `DopamineMapper` | T-NEURO-003 |
| REQ-NEURO-004 | `SerotoninMapper` | T-NEURO-004 |
| REQ-NEURO-005 | `NoradrenalineMapper` | T-NEURO-005 |
| REQ-NEURO-006 | `AcetylcholineMapper` | T-NEURO-006 |
| REQ-NEURO-007 | `ParameterMapping` | T-NEURO-007 |
| REQ-NEURO-008 | `SurpriseTrigger` | T-NEURO-008 |
| REQ-NEURO-009 | `UncertaintyTrigger` | T-NEURO-009 |
| REQ-NEURO-010 | `ExplorationTrigger` | T-NEURO-010 |
| REQ-NEURO-011 | `DeepRecallTrigger` | T-NEURO-011 |
| REQ-NEURO-012 | `TriggerSource`, `TriggerEvent` | T-NEURO-012 |
| REQ-NEURO-013 | `InterpolationEngine` | T-NEURO-013 |
| REQ-NEURO-014 | `HysteresisConfig` | T-NEURO-014 |
| REQ-NEURO-015 | `BoundsChecker` | T-NEURO-015 |
| REQ-NEURO-016 | `NeuromodulationController::update` | T-NEURO-016 |
| REQ-NEURO-017 | `calculate_targets` | T-NEURO-017 |
| REQ-NEURO-022 | `DreamIntegration` | T-NEURO-022 |
| REQ-NEURO-023 | MCP `get_neuromodulation` | T-NEURO-023 |
| REQ-NEURO-025 | `NeuromodulationMetrics` | T-NEURO-025 |
| REQ-NEURO-035 | `SteeringDopamineFeedback` | T-NEURO-035 |
| REQ-NEURO-036 | `apply_steering_feedback` (positive) | T-NEURO-036 |
| REQ-NEURO-037 | `apply_steering_feedback` (negative) | T-NEURO-037 |

---

*Document Version: 1.0.0 | Generated: 2025-12-31 | Architecture Agent*
