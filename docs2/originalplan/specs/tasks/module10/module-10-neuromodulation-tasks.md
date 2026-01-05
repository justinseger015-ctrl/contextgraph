# Module 10: Neuromodulation - Atomic Tasks

```yaml
metadata:
  module_id: "module-10"
  module_name: "Neuromodulation"
  version: "1.0.0"
  phase: 9
  total_tasks: 24
  approach: "inside-out-bottom-up"
  created: "2025-12-31"
  dependencies:
    - module-09-dream-layer
    - module-06-bio-nervous-system
    - module-05-utl-integration
    - module-12.5-steering-subsystem
  estimated_duration: "3 weeks"
  spec_refs:
    - SPEC-NEURO-010 (Functional)
    - TECH-NEURO-010 (Technical)
  performance_constraint: "<200us per update"
```

---

## Task Overview

This module implements a 4-channel neuromodulation system that dynamically adjusts system parameters based on cognitive state. Each channel maps to a core system parameter with strict latency budgets (<200us total per update).

### Channel Mapping Summary

| Channel | Parameter | Output Range | Baseline | Effect |
|---------|-----------|--------------|----------|--------|
| Dopamine | hopfield.beta | [1.0, 5.0] | 0.5 | Retrieval sharpness |
| Serotonin | fuse_moe.top_k | [2, 8] | 0.5 | Expert selection |
| Noradrenaline | attention.temperature | [0.5, 2.0] | 0.3 | Attention distribution |
| Acetylcholine | utl.learning_rate | [0.001, 0.002] | 0.4 | Learning speed |

### Task Organization

1. **Foundation Layer** (Tasks 1-8): Core types, enums, channel definitions
2. **Logic Layer** (Tasks 9-16): Triggers, mappers, interpolation, hysteresis
3. **Surface Layer** (Tasks 17-24): Controller, integrations, MCP tools, Marblestone steering

---

## Foundation Layer: Core Types & Enums

```yaml
tasks:
  # ============================================================
  # FOUNDATION: Core Types
  # ============================================================

  - id: "M10-T01"
    title: "Define ModulatorType Enum and SystemParameter Enum"
    description: |
      Implement ModulatorType enum identifying all 4 neuromodulator channels.
      Variants: Dopamine, Serotonin, Noradrenaline, Acetylcholine.
      Implement SystemParameter enum for target system parameters.
      Variants: HopfieldBeta, FuseMoeTopK, AttentionTemperature, UtlLearningRate.
      Both enums use #[repr(u8)] for efficient serialization.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1
    file_path: "crates/context-graph-neuromod/src/channel/mod.rs"
    dependencies: []
    acceptance_criteria:
      - "ModulatorType enum compiles with 4 variants"
      - "SystemParameter enum compiles with 4 variants"
      - "Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize derived"
      - "#[repr(u8)] applied for efficient serialization"
      - "Display trait implemented for logging"
    test_file: "crates/context-graph-neuromod/tests/channel_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 2.2"
      - "REQ-NEURO-002"

  - id: "M10-T02"
    title: "Define NeuromodulatorChannel Struct"
    description: |
      Implement NeuromodulatorChannel struct for individual channel state.
      Fields: level (f32 [0.0, 1.0]), biological_role (&'static str),
      system_parameter (SystemParameter), output_range ((f32, f32)),
      effect_description (&'static str), mapped_value (f32).
      Define NEUROMODULATOR_CHANNELS const array with 4 channel definitions.
      Match constitution.yaml exactly for all default values.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-neuromod/src/channel/definition.rs"
    dependencies:
      - "M10-T01"
    acceptance_criteria:
      - "NeuromodulatorChannel struct with 6 fields"
      - "NEUROMODULATOR_CHANNELS const array with 4 entries"
      - "Dopamine: output_range (1.0, 5.0), baseline 0.5"
      - "Serotonin: output_range (2, 8), baseline 0.5"
      - "Noradrenaline: output_range (0.5, 2.0), baseline 0.3"
      - "Acetylcholine: output_range (0.001, 0.002), baseline 0.4"
    test_file: "crates/context-graph-neuromod/tests/channel_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 2.2"
      - "REQ-NEURO-002"

  - id: "M10-T03"
    title: "Define NeuromodulatorBaseline Struct"
    description: |
      Implement NeuromodulatorBaseline struct for homeostatic return values.
      Fields: dopamine (f32), serotonin (f32), noradrenaline (f32), acetylcholine (f32).
      Implement Default trait with values: 0.5, 0.5, 0.3, 0.4.
      Implement from_config() method for TOML configuration loading.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1
    file_path: "crates/context-graph-neuromod/src/controller.rs"
    dependencies: []
    acceptance_criteria:
      - "NeuromodulatorBaseline struct with 4 f32 fields"
      - "Default::default() returns dopamine=0.5, serotonin=0.5, noradrenaline=0.3, acetylcholine=0.4"
      - "Clone, Copy, Debug derived"
      - "from_config() parses TOML baseline section"
    test_file: "crates/context-graph-neuromod/tests/baseline_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 2.1"
      - "REQ-NEURO-001"

  - id: "M10-T04"
    title: "Define MappedParameters Struct"
    description: |
      Implement MappedParameters struct for system parameter values after mapping.
      Fields: hopfield_beta (f32), fuse_moe_top_k (u32),
      attention_temperature (f32), utl_learning_rate (f32).
      Implement Default with mid-range values.
      Derive Serialize for MCP tool responses.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1
    file_path: "crates/context-graph-neuromod/src/mapping/mod.rs"
    dependencies: []
    acceptance_criteria:
      - "MappedParameters struct with 4 fields"
      - "Default returns: beta=3.0, top_k=5, temp=1.25, lr=0.0015"
      - "Clone, Copy, Debug, Serialize derived"
      - "All fields accessible for integration"
    test_file: "crates/context-graph-neuromod/tests/mapping_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 3.5"
      - "REQ-NEURO-007"

  - id: "M10-T05"
    title: "Define ModulatorTargets Struct"
    description: |
      Implement ModulatorTargets struct for target modulator levels.
      Fields: dopamine (f32), serotonin (f32), noradrenaline (f32), acetylcholine (f32).
      Used for interpolation targets and dream phase settings.
      Include from_baseline() constructor.
    layer: "foundation"
    priority: "high"
    estimated_hours: 1
    file_path: "crates/context-graph-neuromod/src/interpolation.rs"
    dependencies: []
    acceptance_criteria:
      - "ModulatorTargets struct with 4 f32 fields"
      - "from_baseline(NeuromodulatorBaseline) constructor"
      - "Clone, Copy, Debug derived"
      - "All fields in [0.0, 1.0] range"
    test_file: "crates/context-graph-neuromod/tests/interpolation_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 8.1"
      - "REQ-NEURO-013"

  - id: "M10-T06"
    title: "Define NeuromodulatorLevels Struct"
    description: |
      Implement NeuromodulatorLevels struct for current modulator state snapshot.
      Fields: dopamine (f32), serotonin (f32), noradrenaline (f32), acetylcholine (f32).
      Derive Serialize for MCP tool and metrics export.
      Include as_array() method returning [f32; 4].
    layer: "foundation"
    priority: "high"
    estimated_hours: 1
    file_path: "crates/context-graph-neuromod/src/controller.rs"
    dependencies: []
    acceptance_criteria:
      - "NeuromodulatorLevels struct with 4 f32 fields"
      - "Clone, Copy, Debug, Serialize derived"
      - "as_array() returns [dopamine, serotonin, noradrenaline, acetylcholine]"
      - "from_controller() extracts levels from controller"
    test_file: "crates/context-graph-neuromod/tests/levels_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 9.1"
      - "REQ-NEURO-016"

  - id: "M10-T07"
    title: "Define TriggerSource Enum"
    description: |
      Implement TriggerSource enum for all neuromodulator trigger sources.
      Variants: HighSurprise { surprise: f32 }, HighUncertainty { entropy: f32, coherence: f32 },
      ExplorationMode { quadrant: JohariQuadrant, entropy: f32 },
      DeepRecall { coherence: f32, importance: f32, age_hours: f32 },
      BaselineReturn { elapsed: Duration }, DreamPhase { phase: DreamPhase },
      ManualOverride { reason: String }, Initialization,
      SteeringFeedback { reward: f32, source: SteeringSource } (Marblestone).
      Derive Serialize for logging and debugging.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-neuromod/src/trigger/mod.rs"
    dependencies: []
    acceptance_criteria:
      - "TriggerSource enum with 9 variants"
      - "SteeringFeedback variant for Marblestone integration"
      - "Clone, Debug, Serialize derived"
      - "All variants have descriptive fields"
      - "Variant data matches spec requirements"
    test_file: "crates/context-graph-neuromod/tests/trigger_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 4.1"
      - "REQ-NEURO-012"
      - "REQ-NEURO-035"

  - id: "M10-T08"
    title: "Define TriggerEvent Struct and SteeringSource Enum"
    description: |
      Implement TriggerEvent struct for recording trigger occurrences.
      Fields: timestamp (DateTime<Utc>), modulator (ModulatorType),
      source (TriggerSource), magnitude (f32), resulting_level (f32).
      Include factory methods: dopamine(DopamineSurge), serotonin(SerotoninBoost),
      noradrenaline(NoradrenalineIncrease), acetylcholine(AcetylcholineRise).
      Implement SteeringSource enum for Marblestone integration.
      Variants: Gardener, Curator, ThoughtAssessor, External.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-neuromod/src/trigger/event.rs"
    dependencies:
      - "M10-T01"
      - "M10-T07"
    acceptance_criteria:
      - "TriggerEvent struct with 5 fields"
      - "Factory methods for each modulator type"
      - "SteeringSource enum with 4 variants (Marblestone)"
      - "Clone, Debug, Serialize derived"
      - "Timestamp auto-populated with Utc::now()"
    test_file: "crates/context-graph-neuromod/tests/trigger_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 4.1"
      - "REQ-NEURO-012"
      - "REQ-NEURO-035"

  # ============================================================
  # LOGIC LAYER: Parameter Mappers
  # ============================================================

  - id: "M10-T09"
    title: "Implement DopamineMapper (Dopamine -> Hopfield Beta)"
    description: |
      Implement DopamineMapper struct for dopamine to hopfield.beta mapping.
      Fields: input_range ((f32, f32)), output_range ((f32, f32)), curve (MappingCurve).
      MappingCurve enum: Linear, Sigmoid { steepness: f32 }, Exponential { base: f32 }.
      Method: map(dopamine: f32) -> f32 with <10us latency.
      Method: describe_effect(beta: f32) -> RetrievalEffect.
      RetrievalEffect enum: Sharp, Balanced, Diffuse with selectivity field.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-neuromod/src/mapping/hopfield.rs"
    dependencies: []
    acceptance_criteria:
      - "DopamineMapper struct with 3 fields"
      - "map(0.0) returns 1.0, map(1.0) returns 5.0"
      - "Linear interpolation by default"
      - "Sigmoid/exponential curves supported"
      - "Mapping latency <10us (inline)"
      - "RetrievalEffect describes behavior"
    test_file: "crates/context-graph-neuromod/tests/mapper_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 3.1"
      - "REQ-NEURO-003"

  - id: "M10-T10"
    title: "Implement SerotoninMapper (Serotonin -> FuseMoE Top-K)"
    description: |
      Implement SerotoninMapper struct for serotonin to fuse_moe.top_k mapping.
      Fields: input_range ((f32, f32)), output_range ((u32, u32)), rounding (RoundingStrategy).
      RoundingStrategy enum: Floor, Ceiling, Nearest, Probabilistic.
      Method: map(serotonin: f32) -> u32 with <10us latency.
      Probabilistic rounding uses fastrand for smooth transitions.
      Method: describe_effect(top_k: u32) -> ExplorationEffect.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-neuromod/src/mapping/fusemoe.rs"
    dependencies: []
    acceptance_criteria:
      - "SerotoninMapper struct with 3 fields"
      - "map(0.0) returns 2, map(1.0) returns 8"
      - "Integer output with configurable rounding"
      - "Probabilistic rounding for smooth transitions"
      - "Bounds enforced (never <2, never >8)"
      - "Mapping latency <10us"
    test_file: "crates/context-graph-neuromod/tests/mapper_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 3.2"
      - "REQ-NEURO-004"

  - id: "M10-T11"
    title: "Implement NoradrenalineMapper (Noradrenaline -> Attention Temperature)"
    description: |
      Implement NoradrenalineMapper struct for noradrenaline to attention.temperature.
      Fields: input_range ((f32, f32)), output_range ((f32, f32)), inverse (bool).
      Method: map(noradrenaline: f32) -> f32 with <10us latency.
      Method: describe_effect(temperature: f32) -> AttentionEffect.
      AttentionEffect enum: Focused, Balanced, Flat with entropy and description.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-neuromod/src/mapping/attention.rs"
    dependencies: []
    acceptance_criteria:
      - "NoradrenalineMapper struct with 3 fields"
      - "map(0.0) returns 0.5, map(1.0) returns 2.0"
      - "Linear mapping with smooth transitions"
      - "AttentionEffect describes behavior"
      - "Mapping latency <10us"
    test_file: "crates/context-graph-neuromod/tests/mapper_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 3.3"
      - "REQ-NEURO-005"

  - id: "M10-T12"
    title: "Implement AcetylcholineMapper (Acetylcholine -> UTL Learning Rate)"
    description: |
      Implement AcetylcholineMapper struct for acetylcholine to utl.learning_rate.
      Fields: input_range ((f32, f32)), output_range ((f32, f32)), log_scale (bool).
      Method: map(acetylcholine: f32) -> f32 with <10us latency.
      Logarithmic scaling option for learning rates.
      Method: describe_effect(learning_rate: f32) -> LearningEffect.
      LearningEffect enum: Fast, Moderate, Slow with plasticity and stability.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-neuromod/src/mapping/utl.rs"
    dependencies: []
    acceptance_criteria:
      - "AcetylcholineMapper struct with 3 fields"
      - "map(0.0) returns 0.001, map(1.0) returns 0.002"
      - "Logarithmic scaling option available"
      - "Learning rate stays within valid bounds"
      - "Mapping latency <10us"
    test_file: "crates/context-graph-neuromod/tests/mapper_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 3.4"
      - "REQ-NEURO-006"

  - id: "M10-T13"
    title: "Implement ParameterMapping Unified Interface"
    description: |
      Implement ParameterMapping struct unifying all 4 mappers.
      Fields: dopamine (DopamineMapper), serotonin (SerotoninMapper),
      noradrenaline (NoradrenalineMapper), acetylcholine (AcetylcholineMapper).
      Method: from_constitution() -> Self with default configuration.
      Method: apply(&NeuromodulationController) -> MappedParameters with <50us latency.
      All mappings executed in single call.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-neuromod/src/mapping/mod.rs"
    dependencies:
      - "M10-T04"
      - "M10-T09"
      - "M10-T10"
      - "M10-T11"
      - "M10-T12"
    acceptance_criteria:
      - "ParameterMapping struct with 4 mapper fields"
      - "from_constitution() uses constitution.yaml defaults"
      - "apply() returns MappedParameters in <50us"
      - "All 4 mappings applied in single call"
      - "Parameters easily overridable via config"
    test_file: "crates/context-graph-neuromod/tests/mapping_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 3.5"
      - "REQ-NEURO-007"

  # ============================================================
  # LOGIC LAYER: Trigger Conditions
  # ============================================================

  - id: "M10-T14"
    title: "Implement SurpriseTrigger (High Surprise -> Dopamine Surge)"
    description: |
      Implement SurpriseTrigger struct for dopamine surge on high surprise.
      Fields: surprise_threshold (f32, default 0.7), surge_magnitude (f32, default 0.3),
      decay_rate (f32, default 0.05), cooldown (Duration, default 5s).
      Method: check(utl_state, last_trigger) -> Option<DopamineSurge> with <10us latency.
      DopamineSurge struct: magnitude, decay_rate, triggered_at, trigger_source.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-neuromod/src/trigger/surprise.rs"
    dependencies:
      - "M10-T07"
    acceptance_criteria:
      - "SurpriseTrigger struct with 4 config fields"
      - "Surprise threshold of 0.7 (delta_s) triggers surge"
      - "Dopamine increases by 0.3 on surge"
      - "Cooldown prevents excessive surges"
      - "Trigger check latency <10us"
      - "DopamineSurge struct captures all surge data"
    test_file: "crates/context-graph-neuromod/tests/trigger_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 4.2"
      - "REQ-NEURO-008"

  - id: "M10-T15"
    title: "Implement UncertaintyTrigger (High Uncertainty -> Noradrenaline Increase)"
    description: |
      Implement UncertaintyTrigger for noradrenaline increase during uncertainty.
      Fields: entropy_threshold (0.6), coherence_threshold (0.4),
      increase_magnitude (0.25), duration_threshold (2s).
      Method: check(utl_state, uncertainty_duration) -> Option<NoradrenalineIncrease>.
      Uncertainty = high entropy + low coherence sustained for duration.
      NoradrenalineIncrease struct: magnitude, trigger_source.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-neuromod/src/trigger/uncertainty.rs"
    dependencies:
      - "M10-T07"
    acceptance_criteria:
      - "UncertaintyTrigger struct with 4 config fields"
      - "Entropy > 0.6 AND coherence < 0.4 triggers increase"
      - "2-second sustained uncertainty before trigger"
      - "Magnitude proportional to uncertainty level"
      - "Cap on maximum increase (0.4)"
      - "Trigger check latency <10us"
    test_file: "crates/context-graph-neuromod/tests/trigger_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 4.3"
      - "REQ-NEURO-009"

  - id: "M10-T16"
    title: "Implement ExplorationTrigger (Exploration Mode -> Serotonin Boost)"
    description: |
      Implement ExplorationTrigger for serotonin boost in exploration mode.
      Fields: exploration_quadrants (Vec<JohariQuadrant>), entropy_threshold (0.7),
      boost_magnitude (0.3), duration_threshold (3s).
      Method: check(cognitive_state) -> Option<SerotoninBoost>.
      Trigger on Blind/Unknown quadrants or high entropy with exploration action.
      SerotoninBoost struct: magnitude, trigger_source.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-neuromod/src/trigger/exploration.rs"
    dependencies:
      - "M10-T07"
    acceptance_criteria:
      - "ExplorationTrigger struct with 4 config fields"
      - "Blind/Unknown quadrants trigger exploration"
      - "High entropy (> 0.7) with exploration action triggers boost"
      - "Serotonin increases by 0.3 on boost"
      - "Trigger check latency <10us"
    test_file: "crates/context-graph-neuromod/tests/trigger_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 4.4"
      - "REQ-NEURO-010"

  # ============================================================
  # LOGIC LAYER: Interpolation & Hysteresis
  # ============================================================

  - id: "M10-T17"
    title: "Implement DeepRecallTrigger (Deep Recall -> Acetylcholine Rise)"
    description: |
      Implement DeepRecallTrigger for acetylcholine rise on deep recall.
      Fields: coherence_threshold (0.3), importance_threshold (0.7),
      increase_magnitude (0.2), age_factor (0.1).
      Method: check(recall_context) -> Option<AcetylcholineRise>.
      RecallContext struct: coherence, target_importance, memory_age, recall_depth.
      Older memories receive additional boost via age_factor.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-neuromod/src/trigger/recall.rs"
    dependencies:
      - "M10-T07"
    acceptance_criteria:
      - "DeepRecallTrigger struct with 4 config fields"
      - "Low coherence (< 0.3) + high importance (> 0.7) triggers rise"
      - "Older memories receive additional boost"
      - "Magnitude proportional to recall difficulty"
      - "RecallContext struct captures recall state"
      - "Trigger check latency <10us"
    test_file: "crates/context-graph-neuromod/tests/trigger_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 4.5"
      - "REQ-NEURO-011"

  - id: "M10-T18"
    title: "Implement InterpolationEngine"
    description: |
      Implement InterpolationEngine for smooth parameter transitions.
      Fields: max_delta (0.05), curve (InterpolationCurve), update_interval (50ms).
      InterpolationCurve enum: Linear, EaseInOut, Exponential { decay }, Spring { stiffness, damping }.
      Method: step(current, target) -> f32 with <20us latency.
      Method: step_all(&mut controller, &targets) updates all 4 modulators.
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-neuromod/src/interpolation.rs"
    dependencies:
      - "M10-T05"
    acceptance_criteria:
      - "InterpolationEngine struct with 3 config fields"
      - "Maximum delta of 0.05 per step"
      - "4 interpolation curve types supported"
      - "Smooth transitions between states"
      - "No sudden parameter jumps"
      - "Step latency <20us"
    test_file: "crates/context-graph-neuromod/tests/interpolation_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 8.1"
      - "REQ-NEURO-013"

  - id: "M10-T19"
    title: "Implement HysteresisConfig for Oscillation Prevention"
    description: |
      Implement HysteresisConfig to prevent parameter oscillation.
      Fields: dead_zone (0.02), direction_change_delay (500ms), momentum (0.3),
      states ([HysteresisState; 4]).
      HysteresisState: last_direction, last_direction_change, accumulated_pressure.
      Direction enum: Increasing, Decreasing, Stable.
      Method: apply(modulator, current, proposed_target) -> f32 with <10us latency.
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-neuromod/src/hysteresis.rs"
    dependencies:
      - "M10-T01"
    acceptance_criteria:
      - "HysteresisConfig with 4 fields + per-modulator state"
      - "Dead zone of 0.02 prevents micro-oscillations"
      - "500ms delay between direction changes"
      - "Momentum resists sudden reversals"
      - "Pressure accumulation allows breakthrough"
      - "Oscillation frequency reduced by >90%"
    test_file: "crates/context-graph-neuromod/tests/hysteresis_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 8.2"
      - "REQ-NEURO-014"

  - id: "M10-T20"
    title: "Implement BoundsChecker"
    description: |
      Implement BoundsChecker for modulator level stability.
      Fields: level_bounds ((0.0, 1.0)), alert_threshold (0.9),
      soft_limit (bool), compression_factor (0.8).
      Method: check(level) -> BoundsCheckResult with <5us latency.
      BoundsCheckResult: bounded_level, original_level, was_clamped, alert.
      Soft clamping uses hyperbolic tangent for smooth limits.
    layer: "logic"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-neuromod/src/bounds.rs"
    dependencies: []
    acceptance_criteria:
      - "BoundsChecker struct with 4 config fields"
      - "All levels bounded to [0.0, 1.0]"
      - "Alerts at extreme values (> 0.9 or < 0.1)"
      - "Soft limiting prevents hard cutoffs"
      - "Bounds check latency <5us"
      - "Clamping events tracked"
    test_file: "crates/context-graph-neuromod/tests/bounds_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 8.3"
      - "REQ-NEURO-015"

  # ============================================================
  # SURFACE LAYER: Controller & Integration
  # ============================================================

  - id: "M10-T21"
    title: "Implement SteeringDopamineFeedback (Marblestone)"
    description: |
      Implement SteeringDopamineFeedback for Module 12.5 Steering Subsystem integration.
      Fields: reward (f32 [-1.0, 1.0]), source (SteeringSource), timestamp (DateTime<Utc>),
      thought_id (Option<Uuid>), target_edges (Vec<Uuid>), signal_strength (f32).
      Method on controller: apply_steering_feedback(&mut self, feedback) with <20us latency.
      Positive reward: dopamine += reward * 0.2 (REQ-NEURO-036).
      Negative reward: dopamine += reward * 0.1 (gentler, REQ-NEURO-037).
    layer: "surface"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-neuromod/src/trigger/steering.rs"
    dependencies:
      - "M10-T01"
      - "M10-T08"
    acceptance_criteria:
      - "SteeringDopamineFeedback struct receives reward signals"
      - "Positive reward (> 0) increases dopamine by reward * 0.2"
      - "Negative reward (< 0) decreases dopamine by |reward| * 0.1"
      - "Dopamine level clamped to [0.0, 1.0] range"
      - "SteeringSource tracks origin (Gardener, Curator, ThoughtAssessor)"
      - "Steering feedback latency <20us"
    test_file: "crates/context-graph-neuromod/tests/steering_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 5.1"
      - "REQ-NEURO-035"
      - "REQ-NEURO-036"
      - "REQ-NEURO-037"

  - id: "M10-T22"
    title: "Implement NeuromodulationController Core"
    description: |
      Implement NeuromodulationController struct orchestrating all components.
      Fields: dopamine/serotonin/noradrenaline/acetylcholine (f32), update_rate (0.1),
      baseline (NeuromodulatorBaseline), trigger_state, hysteresis, parameter_map,
      interpolation, bounds_checker, triggers, last_steering_feedback, metrics, dream_targets.
      Method: new(config) -> Self with full initialization.
      Method: update(&mut self, context) -> UpdateResult with <200us latency.
      Update pipeline: triggers -> targets -> hysteresis -> interpolate -> bounds -> map.
    layer: "surface"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-neuromod/src/controller.rs"
    dependencies:
      - "M10-T03"
      - "M10-T06"
      - "M10-T13"
      - "M10-T18"
      - "M10-T19"
      - "M10-T20"
      - "M10-T21"
    acceptance_criteria:
      - "NeuromodulationController struct with all fields"
      - "update() completes within 200 microseconds"
      - "All 4 trigger conditions checked"
      - "Hysteresis applied before interpolation"
      - "Bounds enforced after interpolation"
      - "Metrics recorded for each update"
      - "UpdateResult contains mapped_parameters, triggers, latency, levels"
    test_file: "crates/context-graph-neuromod/tests/controller_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 9.1"
      - "REQ-NEURO-016"

  - id: "M10-T23"
    title: "Implement DreamIntegration"
    description: |
      Implement DreamIntegration for dream layer coordination.
      Fields: phase_settings (DreamPhaseSettings).
      DreamPhaseSettings: nrem (ModulatorTargets), rem (ModulatorTargets), transition_duration (500ms).
      NREM targets: dopamine=0.3, serotonin=0.4, noradrenaline=0.2, acetylcholine=0.2.
      REM targets: dopamine=0.7, serotonin=0.8, noradrenaline=0.6, acetylcholine=0.6.
      Methods: enter_phase(controller, phase), exit_dream(controller).
    layer: "surface"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-neuromod/src/integration/dream.rs"
    dependencies:
      - "M10-T05"
      - "M10-T22"
    acceptance_criteria:
      - "DreamIntegration struct with phase settings"
      - "NREM phase has low dopamine/acetylcholine"
      - "REM phase has high serotonin for exploration"
      - "Smooth 500ms transitions between phases"
      - "Wake returns to baseline"
      - "Integration with DreamPhase enum"
    test_file: "crates/context-graph-neuromod/tests/dream_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 10"
      - "REQ-NEURO-022"

  - id: "M10-T24"
    title: "Implement MCP get_neuromodulation Tool and Metrics"
    description: |
      Implement get_neuromodulation MCP tool for state inspection.
      GetNeuromodulationParams: session_id, include_history, history_size.
      GetNeuromodulationResult: levels, mapped_parameters, active_triggers,
      behavioral_state, history (optional), latency_us.
      BehavioralState: exploration_exploitation (-1 to +1), plasticity, focus, description.
      Implement NeuromodulationMetrics: latency histogram, trigger counters,
      level history, steering feedback counters.
    layer: "surface"
    priority: "high"
    estimated_hours: 4
    file_path: "crates/context-graph-neuromod/src/mcp/tools.rs"
    dependencies:
      - "M10-T06"
      - "M10-T04"
      - "M10-T22"
    acceptance_criteria:
      - "MCP tool returns all 4 modulator levels"
      - "Mapped parameters included in response"
      - "Active triggers shown with sources"
      - "Behavioral state computed (exploration-exploitation balance)"
      - "History available on request"
      - "Latency histogram with P95/P99 tracking"
      - "Prometheus export available"
    test_file: "crates/context-graph-neuromod/tests/mcp_tests.rs"
    spec_refs:
      - "TECH-NEURO-010 Section 11, 12"
      - "REQ-NEURO-023"
      - "REQ-NEURO-025"
```

---

## Dependency Graph

```
M10-T01 (ModulatorType/SystemParameter) ─────────────────────────────────────────────────┐
                                                                                          │
M10-T01 ──► M10-T02 (NeuromodulatorChannel) ─────────────────────────────────────────────┤
                                                                                          │
M10-T03 (NeuromodulatorBaseline) ────────────────────────────────────────────────────────┤
M10-T04 (MappedParameters) ──────────────────────────────────────────────────────────────┤
M10-T05 (ModulatorTargets) ──────────────────────────────────────────────────────────────┤
M10-T06 (NeuromodulatorLevels) ──────────────────────────────────────────────────────────┤
M10-T07 (TriggerSource) ─────────────────────────────────────────────────────────────────┤
                                                                                          │
M10-T01 + M10-T07 ──► M10-T08 (TriggerEvent/SteeringSource) ─────────────────────────────┤
                                                                                          │
M10-T09 (DopamineMapper) ────────────────────────────────────────────────────────────────┤
M10-T10 (SerotoninMapper) ───────────────────────────────────────────────────────────────┤
M10-T11 (NoradrenalineMapper) ───────────────────────────────────────────────────────────┤
M10-T12 (AcetylcholineMapper) ───────────────────────────────────────────────────────────┤
                                                                                          │
M10-T04 + M10-T09..12 ──► M10-T13 (ParameterMapping) ────────────────────────────────────┤
                                                                                          │
M10-T07 ──► M10-T14 (SurpriseTrigger) ───────────────────────────────────────────────────┤
M10-T07 ──► M10-T15 (UncertaintyTrigger) ────────────────────────────────────────────────┤
M10-T07 ──► M10-T16 (ExplorationTrigger) ────────────────────────────────────────────────┤
M10-T07 ──► M10-T17 (DeepRecallTrigger) ─────────────────────────────────────────────────┤
                                                                                          │
M10-T05 ──► M10-T18 (InterpolationEngine) ───────────────────────────────────────────────┤
M10-T01 ──► M10-T19 (HysteresisConfig) ──────────────────────────────────────────────────┤
M10-T20 (BoundsChecker) ─────────────────────────────────────────────────────────────────┤
                                                                                          │
M10-T01 + M10-T08 ──► M10-T21 (SteeringDopamineFeedback - Marblestone) ──────────────────┤
                                                                                          │
M10-T03 + M10-T06 + M10-T13..21 ──► M10-T22 (NeuromodulationController) ─────────────────┤
                                                                                          │
M10-T05 + M10-T22 ──► M10-T23 (DreamIntegration) ────────────────────────────────────────┤
                                                                                          │
M10-T04 + M10-T06 + M10-T22 ──► M10-T24 (MCP Tools & Metrics) ◄──────────────────────────┘
```

---

## Implementation Order (Recommended)

### Week 1: Foundation & Mappers
1. M10-T01: ModulatorType/SystemParameter enums
2. M10-T02: NeuromodulatorChannel struct
3. M10-T03: NeuromodulatorBaseline struct
4. M10-T04: MappedParameters struct
5. M10-T05: ModulatorTargets struct
6. M10-T06: NeuromodulatorLevels struct
7. M10-T07: TriggerSource enum
8. M10-T08: TriggerEvent/SteeringSource
9. M10-T09: DopamineMapper (<10us)
10. M10-T10: SerotoninMapper (<10us)
11. M10-T11: NoradrenalineMapper (<10us)
12. M10-T12: AcetylcholineMapper (<10us)
13. M10-T13: ParameterMapping (<50us)

### Week 2: Triggers & Interpolation
14. M10-T14: SurpriseTrigger (<10us)
15. M10-T15: UncertaintyTrigger (<10us)
16. M10-T16: ExplorationTrigger (<10us)
17. M10-T17: DeepRecallTrigger (<10us)
18. M10-T18: InterpolationEngine (<20us)
19. M10-T19: HysteresisConfig (<10us)
20. M10-T20: BoundsChecker (<5us)

### Week 3: Controller & Integration
21. M10-T21: SteeringDopamineFeedback (Marblestone, <20us)
22. M10-T22: NeuromodulationController (<200us E2E)
23. M10-T23: DreamIntegration
24. M10-T24: MCP Tools & Metrics

---

## Quality Gates

| Gate | Criteria | Required For |
|------|----------|--------------|
| Foundation Complete | M10-T01 through M10-T08 pass all tests | Week 2 start |
| Mappers Verified | M10-T09 through M10-T13 all <50us total | Week 2 |
| Triggers Operational | M10-T14 through M10-T17 all <10us each | Week 3 start |
| Controller <200us | M10-T22 E2E latency verified | Module 11 start |
| Marblestone Integration | M10-T21 steering feedback verified | Module 12.5 |

---

## Performance Targets Summary

| Component | Budget | Metric |
|-----------|--------|--------|
| Trigger checks (all 4) | <50us | P95 latency |
| Parameter mapping (all 4) | <50us | P95 latency |
| Interpolation step (all 4) | <20us | P95 latency |
| Hysteresis check (per modulator) | <10us | P95 latency |
| Bounds check (per modulator) | <5us | P95 latency |
| Steering feedback | <20us | Marblestone integration |
| **Total Update Cycle** | **<200us** | **P99 latency** |
| Memory per session | <2KB | Controller state |
| Oscillation reduction | >90% | Hysteresis effectiveness |

---

## Memory Budget

| Component | Budget |
|-----------|--------|
| Controller state | ~500 bytes |
| Trigger state (all 4) | ~200 bytes |
| Hysteresis state | ~100 bytes |
| History buffer | ~1KB |
| **Total per session** | **<2KB** |

---

## Marblestone Integration Summary

Tasks with Marblestone features (Module 12.5 Steering Subsystem):
- **M10-T07**: TriggerSource enum includes SteeringFeedback variant
- **M10-T08**: SteeringSource enum (Gardener, Curator, ThoughtAssessor)
- **M10-T21**: SteeringDopamineFeedback + apply_steering_feedback()
- **M10-T22**: Controller integration with last_steering_feedback

### Steering Feedback Rules (REQ-NEURO-035/036/037)
- Positive reward (> 0): dopamine += reward * 0.2
- Negative reward (< 0): dopamine += reward * 0.1 (gentler)
- Dopamine clamped to [0.0, 1.0]
- Latency constraint: <20us

---

## Critical Constraints

**LATENCY BUDGETS ARE HARD REQUIREMENTS**:
- Each mapper: <10us (use #[inline])
- Each trigger check: <10us (use #[inline])
- Interpolation: <20us
- Bounds check: <5us per modulator
- Total update: <200us P99

**MODULATOR RANGES**:
- dopamine: [0,1] -> hopfield.beta [1.0, 5.0]
- serotonin: [0,1] -> fuse_moe.top_k [2, 8]
- noradrenaline: [0,1] -> attention.temperature [0.5, 2.0]
- acetylcholine: [0,1] -> utl.learning_rate [0.001, 0.002]

**BASELINE VALUES**:
- dopamine: 0.5
- serotonin: 0.5
- noradrenaline: 0.3
- acetylcholine: 0.4

**HYSTERESIS REQUIREMENTS**:
- Dead zone: 0.02
- Direction change delay: 500ms
- Momentum: 0.3
- Oscillation reduction: >90%

---

*Generated: 2025-12-31*
*Module: 10 - Neuromodulation*
*Version: 1.0.0*
*Total Tasks: 24*
