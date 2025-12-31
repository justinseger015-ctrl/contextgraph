//! Cognitive Pulse types for meta-cognitive state tracking.
//!
//! Every MCP tool response includes a Cognitive Pulse header to convey
//! the current system state and suggest next actions.

use serde::{Deserialize, Serialize};

/// Cognitive Pulse header included in all tool responses.
///
/// Provides meta-cognitive state information to help agents
/// understand system state and decide on next actions.
///
/// # Example Response
///
/// ```json
/// {
///   "entropy": 0.45,
///   "coherence": 0.72,
///   "suggested_action": "continue"
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

    /// Suggested action based on current state
    pub suggested_action: SuggestedAction,
}

impl Default for CognitivePulse {
    fn default() -> Self {
        Self {
            entropy: 0.5,
            coherence: 0.5,
            suggested_action: SuggestedAction::Continue,
        }
    }
}

impl CognitivePulse {
    /// Create a new pulse with the given entropy, coherence, and suggested action.
    ///
    /// If you want the action to be computed automatically based on entropy/coherence,
    /// use `CognitivePulse::computed(entropy, coherence)` instead.
    pub fn new(entropy: f32, coherence: f32, suggested_action: SuggestedAction) -> Self {
        let entropy = entropy.clamp(0.0, 1.0);
        let coherence = coherence.clamp(0.0, 1.0);

        Self {
            entropy,
            coherence,
            suggested_action,
        }
    }

    /// Create a new pulse with the given entropy and coherence.
    /// The suggested action is automatically computed based on the values.
    pub fn computed(entropy: f32, coherence: f32) -> Self {
        let entropy = entropy.clamp(0.0, 1.0);
        let coherence = coherence.clamp(0.0, 1.0);
        let suggested_action = Self::compute_action(entropy, coherence);

        Self {
            entropy,
            coherence,
            suggested_action,
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
}

/// Action suggestions based on cognitive state.
///
/// These suggest what the agent should consider doing next
/// based on the current entropy/coherence balance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SuggestedAction {
    /// System ready, no action needed
    Ready,
    /// Continue current operation
    Continue,
    /// Consider exploring new areas
    Explore,
    /// Focus on consolidating knowledge
    Consolidate,
    /// Reduce complexity, prune low-value nodes
    Prune,
    /// High entropy - needs stabilization
    Stabilize,
    /// Review and verify recent additions
    Review,
}

impl SuggestedAction {
    /// Returns a human-readable description of this action.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Ready => "System is ready and stable",
            Self::Continue => "Continue current operations",
            Self::Explore => "Consider exploring new knowledge areas",
            Self::Consolidate => "Focus on consolidating existing knowledge",
            Self::Prune => "Consider pruning low-value information",
            Self::Stabilize => "System needs stabilization - reduce entropy",
            Self::Review => "Review and verify recent additions",
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
        assert_eq!(pulse.suggested_action, SuggestedAction::Continue);
    }

    #[test]
    fn test_pulse_new_with_action() {
        let pulse = CognitivePulse::new(0.5, 0.7, SuggestedAction::Explore);
        assert_eq!(pulse.entropy, 0.5);
        assert_eq!(pulse.coherence, 0.7);
        assert_eq!(pulse.suggested_action, SuggestedAction::Explore);
    }

    #[test]
    fn test_pulse_computed_stabilize() {
        let pulse = CognitivePulse::computed(0.9, 0.2);
        assert_eq!(pulse.suggested_action, SuggestedAction::Stabilize);
    }

    #[test]
    fn test_pulse_computed_ready() {
        let pulse = CognitivePulse::computed(0.3, 0.8);
        assert_eq!(pulse.suggested_action, SuggestedAction::Ready);
    }

    #[test]
    fn test_is_healthy() {
        let healthy = CognitivePulse::computed(0.5, 0.6);
        assert!(healthy.is_healthy());

        let unhealthy = CognitivePulse::computed(0.9, 0.2);
        assert!(!unhealthy.is_healthy());
    }

    #[test]
    fn test_pulse_clamps_values() {
        let pulse = CognitivePulse::new(1.5, -0.5, SuggestedAction::Continue);
        assert_eq!(pulse.entropy, 1.0);
        assert_eq!(pulse.coherence, 0.0);
    }
}
