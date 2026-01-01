//! Cognitive Pulse types for meta-cognitive state tracking.
//!
//! Every MCP tool response includes a Cognitive Pulse header to convey
//! the current system state and suggest next actions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::nervous::LayerId;
use super::utl::{EmotionalState, UtlMetrics};

/// Cognitive Pulse header included in all tool responses.
///
/// Provides meta-cognitive state information to help agents
/// understand system state and decide on next actions.
///
/// The 7-field structure captures:
/// - Core metrics: entropy, coherence, coherence_delta
/// - Emotional context: emotional_weight (from EmotionalState)
/// - Action guidance: suggested_action
/// - Source tracking: source_layer
/// - Temporal context: timestamp
///
/// # Example Response
///
/// ```json
/// {
///   "entropy": 0.45,
///   "coherence": 0.72,
///   "coherence_delta": 0.05,
///   "emotional_weight": 1.2,
///   "suggested_action": "continue",
///   "source_layer": "learning",
///   "timestamp": "2025-01-01T12:00:00Z"
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CognitivePulse {
    /// Current entropy level [0.0, 1.0]
    /// Higher values indicate more uncertainty/novelty
    pub entropy: f32,

    /// Current coherence level [0.0, 1.0]
    /// Higher values indicate better integration/understanding
    pub coherence: f32,

    /// Change in coherence from previous measurement [−1.0, 1.0]
    /// Positive values indicate improving understanding
    pub coherence_delta: f32,

    /// Emotional weight modifier from EmotionalState [0.0, 2.0]
    /// Derived from EmotionalState::weight_modifier()
    pub emotional_weight: f32,

    /// Suggested action based on current state
    pub suggested_action: SuggestedAction,

    /// Source layer that generated this pulse (None if computed globally)
    pub source_layer: Option<LayerId>,

    /// UTC timestamp when this pulse was created
    pub timestamp: DateTime<Utc>,
}

impl Default for CognitivePulse {
    fn default() -> Self {
        Self {
            entropy: 0.5,
            coherence: 0.5,
            coherence_delta: 0.0,
            emotional_weight: 1.0,
            suggested_action: SuggestedAction::Continue,
            source_layer: None,
            timestamp: Utc::now(),
        }
    }
}

impl CognitivePulse {
    /// Create a new pulse with explicit values for all 7 fields.
    ///
    /// Values are clamped to valid ranges:
    /// - entropy: [0.0, 1.0]
    /// - coherence: [0.0, 1.0]
    /// - coherence_delta: [-1.0, 1.0]
    /// - emotional_weight: [0.0, 2.0]
    pub fn new(
        entropy: f32,
        coherence: f32,
        coherence_delta: f32,
        emotional_weight: f32,
        suggested_action: SuggestedAction,
        source_layer: Option<LayerId>,
    ) -> Self {
        let entropy = entropy.clamp(0.0, 1.0);
        let coherence = coherence.clamp(0.0, 1.0);
        let coherence_delta = coherence_delta.clamp(-1.0, 1.0);
        let emotional_weight = emotional_weight.clamp(0.0, 2.0);

        Self {
            entropy,
            coherence,
            coherence_delta,
            emotional_weight,
            suggested_action,
            source_layer,
            timestamp: Utc::now(),
        }
    }

    /// Create a new pulse computed from UTL metrics and source layer.
    ///
    /// Derives all 7 fields from the provided metrics:
    /// - entropy: from metrics.entropy
    /// - coherence: from metrics.coherence
    /// - coherence_delta: from metrics.coherence_change
    /// - emotional_weight: from metrics.emotional_weight
    /// - suggested_action: computed from entropy/coherence
    /// - source_layer: from the provided layer parameter
    /// - timestamp: current UTC time
    pub fn computed(metrics: &UtlMetrics, source_layer: Option<LayerId>) -> Self {
        let entropy = metrics.entropy.clamp(0.0, 1.0);
        let coherence = metrics.coherence.clamp(0.0, 1.0);
        let coherence_delta = metrics.coherence_change.clamp(-1.0, 1.0);
        let emotional_weight = metrics.emotional_weight.clamp(0.0, 2.0);
        let suggested_action = Self::compute_action(entropy, coherence);

        Self {
            entropy,
            coherence,
            coherence_delta,
            emotional_weight,
            suggested_action,
            source_layer,
            timestamp: Utc::now(),
        }
    }

    /// Create a simple pulse from entropy and coherence values.
    ///
    /// Uses default values for other fields:
    /// - coherence_delta: 0.0
    /// - emotional_weight: 1.0
    /// - source_layer: None
    /// - suggested_action: computed from entropy/coherence
    pub fn from_values(entropy: f32, coherence: f32) -> Self {
        let entropy = entropy.clamp(0.0, 1.0);
        let coherence = coherence.clamp(0.0, 1.0);
        let suggested_action = Self::compute_action(entropy, coherence);

        Self {
            entropy,
            coherence,
            coherence_delta: 0.0,
            emotional_weight: 1.0,
            suggested_action,
            source_layer: None,
            timestamp: Utc::now(),
        }
    }

    /// Compute the suggested action based on entropy and coherence.
    fn compute_action(entropy: f32, coherence: f32) -> SuggestedAction {
        match (entropy, coherence) {
            // High entropy, low coherence - needs stabilization
            (e, c) if e > 0.7 && c < 0.4 => SuggestedAction::Stabilize,
            // High entropy, high coherence - exploration frontier
            (e, c) if e > 0.6 && c > 0.5 => SuggestedAction::Explore,
            // Low entropy, high coherence - well understood, ready
            (e, c) if e < 0.4 && c > 0.6 => SuggestedAction::Ready,
            // Low coherence - needs consolidation
            (_, c) if c < 0.4 => SuggestedAction::Consolidate,
            // High entropy - consider pruning
            (e, _) if e > 0.8 => SuggestedAction::Prune,
            // Review needed
            (e, c) if e > 0.5 && c < 0.5 => SuggestedAction::Review,
            // Default: continue
            _ => SuggestedAction::Continue,
        }
    }

    /// Returns true if the system is in a healthy state.
    pub fn is_healthy(&self) -> bool {
        self.entropy < 0.8 && self.coherence > 0.3
    }

    /// Create a pulse with a specific emotional state.
    ///
    /// Derives emotional_weight from the provided EmotionalState.
    pub fn with_emotion(
        entropy: f32,
        coherence: f32,
        emotional_state: EmotionalState,
        source_layer: Option<LayerId>,
    ) -> Self {
        let entropy = entropy.clamp(0.0, 1.0);
        let coherence = coherence.clamp(0.0, 1.0);
        let emotional_weight = emotional_state.weight_modifier();
        let suggested_action = Self::compute_action(entropy, coherence);

        Self {
            entropy,
            coherence,
            coherence_delta: 0.0,
            emotional_weight,
            suggested_action,
            source_layer,
            timestamp: Utc::now(),
        }
    }
}

/// Action suggestions based on cognitive state.
///
/// These suggest what the agent should consider doing next
/// based on the current entropy/coherence balance.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SuggestedAction {
    /// System ready for new input - low entropy, high coherence.
    Ready,
    /// Continue current activity - balanced state (DEFAULT).
    #[default]
    Continue,
    /// Explore new knowledge - use epistemic_action or trigger_dream(rem).
    Explore,
    /// Consolidate knowledge - use trigger_dream(nrem) or merge_concepts.
    Consolidate,
    /// Prune redundant information - review curation_tasks.
    Prune,
    /// Stabilize context - use trigger_dream or critique_context.
    Stabilize,
    /// Review context - use critique_context or reflect_on_memory.
    Review,
}

impl SuggestedAction {
    /// Returns a human-readable description with MCP tool guidance.
    ///
    /// Each description includes actionable guidance for which MCP tools
    /// to use based on the current cognitive state.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Ready => "System ready for new input - low entropy, high coherence",
            Self::Continue => "Continue current activity - balanced state",
            Self::Explore => "Explore new knowledge - use epistemic_action or trigger_dream(rem)",
            Self::Consolidate => {
                "Consolidate knowledge - use trigger_dream(nrem) or merge_concepts"
            }
            Self::Prune => "Prune redundant information - review curation_tasks",
            Self::Stabilize => "Stabilize context - use trigger_dream or critique_context",
            Self::Review => "Review context - use critique_context or reflect_on_memory",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulse_default() {
        let pulse = CognitivePulse::default();
        assert_eq!(pulse.entropy, 0.5);
        assert_eq!(pulse.coherence, 0.5);
        assert_eq!(pulse.coherence_delta, 0.0);
        assert_eq!(pulse.emotional_weight, 1.0);
        assert_eq!(pulse.suggested_action, SuggestedAction::Continue);
        assert!(pulse.source_layer.is_none());
        // Timestamp should be recent (within last second)
        let now = Utc::now();
        let diff = now.signed_duration_since(pulse.timestamp);
        assert!(diff.num_seconds().abs() < 2);
    }

    #[test]
    fn test_pulse_new_with_all_fields() {
        let pulse = CognitivePulse::new(
            0.5,
            0.7,
            0.1,
            1.2,
            SuggestedAction::Explore,
            Some(LayerId::Learning),
        );
        assert_eq!(pulse.entropy, 0.5);
        assert_eq!(pulse.coherence, 0.7);
        assert_eq!(pulse.coherence_delta, 0.1);
        assert_eq!(pulse.emotional_weight, 1.2);
        assert_eq!(pulse.suggested_action, SuggestedAction::Explore);
        assert_eq!(pulse.source_layer, Some(LayerId::Learning));
    }

    #[test]
    fn test_pulse_computed_from_metrics() {
        let metrics = UtlMetrics {
            entropy: 0.3,
            coherence: 0.8,
            learning_score: 0.5,
            surprise: 0.4,
            coherence_change: 0.15,
            emotional_weight: 1.3,
            alignment: 0.9,
        };

        let pulse = CognitivePulse::computed(&metrics, Some(LayerId::Coherence));

        assert_eq!(pulse.entropy, 0.3);
        assert_eq!(pulse.coherence, 0.8);
        assert_eq!(pulse.coherence_delta, 0.15);
        assert_eq!(pulse.emotional_weight, 1.3);
        assert_eq!(pulse.suggested_action, SuggestedAction::Ready); // low entropy, high coherence
        assert_eq!(pulse.source_layer, Some(LayerId::Coherence));
    }

    #[test]
    fn test_pulse_from_values() {
        let pulse = CognitivePulse::from_values(0.9, 0.2);
        assert_eq!(pulse.entropy, 0.9);
        assert_eq!(pulse.coherence, 0.2);
        assert_eq!(pulse.coherence_delta, 0.0);
        assert_eq!(pulse.emotional_weight, 1.0);
        assert_eq!(pulse.suggested_action, SuggestedAction::Stabilize);
        assert!(pulse.source_layer.is_none());
    }

    #[test]
    fn test_pulse_with_emotion() {
        let pulse = CognitivePulse::with_emotion(
            0.5,
            0.6,
            EmotionalState::Focused,
            Some(LayerId::Memory),
        );
        assert_eq!(pulse.entropy, 0.5);
        assert_eq!(pulse.coherence, 0.6);
        assert_eq!(pulse.coherence_delta, 0.0);
        assert_eq!(pulse.emotional_weight, 1.3); // Focused weight
        assert_eq!(pulse.source_layer, Some(LayerId::Memory));
    }

    #[test]
    fn test_pulse_computed_stabilize() {
        let pulse = CognitivePulse::from_values(0.9, 0.2);
        assert_eq!(pulse.suggested_action, SuggestedAction::Stabilize);
    }

    #[test]
    fn test_pulse_computed_ready() {
        let pulse = CognitivePulse::from_values(0.3, 0.8);
        assert_eq!(pulse.suggested_action, SuggestedAction::Ready);
    }

    #[test]
    fn test_is_healthy() {
        let healthy = CognitivePulse::from_values(0.5, 0.6);
        assert!(healthy.is_healthy());

        let unhealthy = CognitivePulse::from_values(0.9, 0.2);
        assert!(!unhealthy.is_healthy());
    }

    #[test]
    fn test_pulse_clamps_values() {
        let pulse = CognitivePulse::new(
            1.5,   // should clamp to 1.0
            -0.5,  // should clamp to 0.0
            2.0,   // should clamp to 1.0
            3.0,   // should clamp to 2.0
            SuggestedAction::Continue,
            None,
        );
        assert_eq!(pulse.entropy, 1.0);
        assert_eq!(pulse.coherence, 0.0);
        assert_eq!(pulse.coherence_delta, 1.0);
        assert_eq!(pulse.emotional_weight, 2.0);
    }

    #[test]
    fn test_pulse_clamps_negative_coherence_delta() {
        let pulse = CognitivePulse::new(
            0.5,
            0.5,
            -2.0, // should clamp to -1.0
            1.0,
            SuggestedAction::Continue,
            None,
        );
        assert_eq!(pulse.coherence_delta, -1.0);
    }

    #[test]
    fn test_pulse_timestamp_is_current() {
        let before = Utc::now();
        let pulse = CognitivePulse::default();
        let after = Utc::now();

        assert!(pulse.timestamp >= before);
        assert!(pulse.timestamp <= after);
    }

    #[test]
    fn test_pulse_serde_roundtrip() {
        let pulse = CognitivePulse::new(
            0.5,
            0.7,
            0.1,
            1.2,
            SuggestedAction::Explore,
            Some(LayerId::Learning),
        );

        let json = serde_json::to_string(&pulse).unwrap();
        let parsed: CognitivePulse = serde_json::from_str(&json).unwrap();

        assert_eq!(pulse.entropy, parsed.entropy);
        assert_eq!(pulse.coherence, parsed.coherence);
        assert_eq!(pulse.coherence_delta, parsed.coherence_delta);
        assert_eq!(pulse.emotional_weight, parsed.emotional_weight);
        assert_eq!(pulse.suggested_action, parsed.suggested_action);
        assert_eq!(pulse.source_layer, parsed.source_layer);
        assert_eq!(pulse.timestamp, parsed.timestamp);
    }

    #[test]
    fn test_pulse_all_seven_fields_present() {
        let pulse = CognitivePulse::default();

        // Verify all 7 fields exist and have valid values
        let _entropy: f32 = pulse.entropy;
        let _coherence: f32 = pulse.coherence;
        let _coherence_delta: f32 = pulse.coherence_delta;
        let _emotional_weight: f32 = pulse.emotional_weight;
        let _suggested_action: SuggestedAction = pulse.suggested_action;
        let _source_layer: Option<LayerId> = pulse.source_layer;
        let _timestamp: DateTime<Utc> = pulse.timestamp;

        // All fields are valid
        assert!(pulse.entropy >= 0.0 && pulse.entropy <= 1.0);
        assert!(pulse.coherence >= 0.0 && pulse.coherence <= 1.0);
        assert!(pulse.coherence_delta >= -1.0 && pulse.coherence_delta <= 1.0);
        assert!(pulse.emotional_weight >= 0.0 && pulse.emotional_weight <= 2.0);
    }

    #[test]
    fn test_computed_derives_all_fields_from_metrics() {
        let metrics = UtlMetrics {
            entropy: 0.45,
            coherence: 0.75,
            learning_score: 0.6,
            surprise: 0.5,
            coherence_change: -0.1,
            emotional_weight: 0.8,
            alignment: 0.95,
        };

        let pulse = CognitivePulse::computed(&metrics, Some(LayerId::Sensing));

        // Verify derived values
        assert_eq!(pulse.entropy, metrics.entropy);
        assert_eq!(pulse.coherence, metrics.coherence);
        assert_eq!(pulse.coherence_delta, metrics.coherence_change);
        assert_eq!(pulse.emotional_weight, metrics.emotional_weight);
        assert_eq!(pulse.source_layer, Some(LayerId::Sensing));
    }

    // =======================================================================
    // SuggestedAction Tests (TASK-M02-020)
    // =======================================================================

    #[test]
    fn test_suggested_action_default_is_continue() {
        let action = SuggestedAction::default();
        assert_eq!(action, SuggestedAction::Continue);
    }

    #[test]
    fn test_suggested_action_serde_roundtrip() {
        let actions = [
            SuggestedAction::Ready,
            SuggestedAction::Continue,
            SuggestedAction::Explore,
            SuggestedAction::Consolidate,
            SuggestedAction::Prune,
            SuggestedAction::Stabilize,
            SuggestedAction::Review,
        ];
        for action in actions {
            let json = serde_json::to_string(&action).unwrap();
            let parsed: SuggestedAction = serde_json::from_str(&json).unwrap();
            assert_eq!(action, parsed);
        }
    }

    #[test]
    fn test_suggested_action_serde_snake_case() {
        // Verify snake_case serialization
        let json = serde_json::to_string(&SuggestedAction::Ready).unwrap();
        assert_eq!(json, "\"ready\"");

        let json = serde_json::to_string(&SuggestedAction::Continue).unwrap();
        assert_eq!(json, "\"continue\"");
    }

    #[test]
    fn test_suggested_action_descriptions_not_empty() {
        let actions = [
            SuggestedAction::Ready,
            SuggestedAction::Continue,
            SuggestedAction::Explore,
            SuggestedAction::Consolidate,
            SuggestedAction::Prune,
            SuggestedAction::Stabilize,
            SuggestedAction::Review,
        ];
        for action in actions {
            let desc = action.description();
            assert!(!desc.is_empty(), "{:?} has empty description", action);
            assert!(
                desc.len() > 20,
                "{:?} description too short: {}",
                action,
                desc
            );
        }
    }

    #[test]
    fn test_suggested_action_descriptions_unique() {
        use std::collections::HashSet;
        let actions = [
            SuggestedAction::Ready,
            SuggestedAction::Continue,
            SuggestedAction::Explore,
            SuggestedAction::Consolidate,
            SuggestedAction::Prune,
            SuggestedAction::Stabilize,
            SuggestedAction::Review,
        ];
        let descriptions: HashSet<_> = actions.iter().map(|a| a.description()).collect();
        assert_eq!(
            descriptions.len(),
            actions.len(),
            "Descriptions must be unique"
        );
    }

    #[test]
    fn test_suggested_action_copy_semantics() {
        let action = SuggestedAction::Explore;
        let copied = action; // Copy, not move
        assert_eq!(action, copied);
        assert_eq!(action.description(), copied.description());
    }

    #[test]
    fn test_suggested_action_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SuggestedAction::Ready);
        set.insert(SuggestedAction::Continue);
        set.insert(SuggestedAction::Ready); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_suggested_action_invalid_serde_rejected() {
        // Verify invalid variant is correctly rejected
        let json = "\"unknown_action\"";
        let result: Result<SuggestedAction, _> = serde_json::from_str(json);
        assert!(result.is_err(), "Invalid variant should be rejected");
    }

    #[test]
    fn test_suggested_action_descriptions_contain_mcp_tools() {
        // Verify key actions have MCP tool guidance
        assert!(
            SuggestedAction::Explore
                .description()
                .contains("epistemic_action")
        );
        assert!(SuggestedAction::Explore.description().contains("trigger_dream"));
        assert!(
            SuggestedAction::Consolidate
                .description()
                .contains("trigger_dream")
        );
        assert!(
            SuggestedAction::Consolidate
                .description()
                .contains("merge_concepts")
        );
        assert!(
            SuggestedAction::Stabilize
                .description()
                .contains("critique_context")
        );
        assert!(SuggestedAction::Review.description().contains("reflect_on_memory"));
    }
}
