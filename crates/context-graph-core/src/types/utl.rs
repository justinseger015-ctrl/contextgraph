//! UTL (Unified Theory of Learning) types and metrics.
//!
//! Implements the learning equation: L = f((ΔS × ΔC) · wₑ · cos φ)

use serde::{Deserialize, Serialize};

/// UTL (Unified Theory of Learning) metrics.
///
/// Captures all components of the UTL equation for measuring learning effectiveness.
///
/// # The UTL Equation
///
/// `L = f((ΔS × ΔC) · wₑ · cos φ)`
///
/// Where:
/// - `ΔS` (delta_s): Surprise/entropy change - information gain
/// - `ΔC` (delta_c): Coherence change - understanding gain
/// - `wₑ`: Emotional weight - attention/motivation
/// - `cos φ`: Goal alignment - how well learning aligns with objectives
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UtlMetrics {
    /// Entropy measure [0.0, 1.0]
    pub entropy: f32,

    /// Coherence measure [0.0, 1.0]
    pub coherence: f32,

    /// Computed learning score
    pub learning_score: f32,

    /// Surprise component (delta_S)
    pub surprise: f32,

    /// Coherence change component (delta_C)
    pub coherence_change: f32,

    /// Emotional weight (w_e)
    pub emotional_weight: f32,

    /// Alignment angle cosine (cos phi)
    pub alignment: f32,
}

impl Default for UtlMetrics {
    fn default() -> Self {
        Self {
            entropy: 0.5,
            coherence: 0.5,
            learning_score: 0.0,
            surprise: 0.0,
            coherence_change: 0.0,
            emotional_weight: 1.0,
            alignment: 1.0,
        }
    }
}

impl UtlMetrics {
    /// Compute the learning score from current components.
    ///
    /// Uses the UTL equation: L = (ΔS × ΔC) · wₑ · cos φ
    pub fn compute_learning_score(&mut self) {
        self.learning_score =
            (self.surprise * self.coherence_change) * self.emotional_weight * self.alignment;
        self.learning_score = self.learning_score.clamp(0.0, 1.0);
    }

    /// Check if this represents an optimal learning state.
    ///
    /// Optimal learning occurs when both surprise and coherence are balanced
    /// (the "Aha!" moment - not too easy, not too confusing).
    pub fn is_optimal(&self) -> bool {
        self.entropy > 0.3 && self.entropy < 0.7 && self.coherence > 0.4 && self.coherence < 0.8
    }
}

/// UTL computation context.
///
/// Provides the contextual information needed to compute UTL metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UtlContext {
    /// Prior beliefs/expectations (prior entropy)
    pub prior_entropy: f32,

    /// Current system coherence
    pub current_coherence: f32,

    /// Emotional state modifier
    pub emotional_state: EmotionalState,

    /// Goal alignment vector
    pub goal_vector: Option<Vec<f32>>,
}

/// Emotional state for UTL weight computation.
///
/// Maps to the `wₑ` component - higher emotional engagement
/// leads to more effective learning.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum EmotionalState {
    #[default]
    Neutral,
    Curious,
    Focused,
    Stressed,
    Fatigued,
}

impl EmotionalState {
    /// Get the emotional weight multiplier for this state.
    pub fn weight(&self) -> f32 {
        match self {
            Self::Neutral => 1.0,
            Self::Curious => 1.3,
            Self::Focused => 1.2,
            Self::Stressed => 0.7,
            Self::Fatigued => 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utl_default() {
        let metrics = UtlMetrics::default();
        assert_eq!(metrics.entropy, 0.5);
        assert_eq!(metrics.coherence, 0.5);
        assert_eq!(metrics.learning_score, 0.0);
    }

    #[test]
    fn test_compute_learning_score() {
        let mut metrics = UtlMetrics {
            surprise: 0.5,
            coherence_change: 0.6,
            emotional_weight: 1.2,
            alignment: 0.9,
            ..Default::default()
        };
        metrics.compute_learning_score();

        // (0.5 * 0.6) * 1.2 * 0.9 = 0.324
        assert!((metrics.learning_score - 0.324).abs() < 0.001);
    }

    #[test]
    fn test_emotional_weights() {
        assert_eq!(EmotionalState::Neutral.weight(), 1.0);
        assert!(EmotionalState::Curious.weight() > 1.0);
        assert!(EmotionalState::Fatigued.weight() < 1.0);
    }
}
