//! Neurotransmitter and steering modulation methods for GraphEdge.

use super::edge::GraphEdge;

impl GraphEdge {
    /// Computes the modulated weight considering NT weights and steering reward.
    ///
    /// This is the primary method for getting an edge's effective weight during
    /// graph traversal. It combines:
    /// 1. Neurotransmitter modulation (domain-specific)
    /// 2. Steering reward feedback (reinforcement learning signal)
    ///
    /// # Formula
    ///
    /// ```text
    /// nt_factor = neurotransmitter_weights.compute_effective_weight(weight)
    /// modulated = (nt_factor * (1.0 + steering_reward * 0.2)).clamp(0.0, 1.0)
    /// ```
    ///
    /// # Returns
    ///
    /// Effective weight in [0.0, 1.0], never NaN or Infinity (per AP-009).
    ///
    /// # Constitution Compliance
    ///
    /// - Dopamine feedback: `pos: "+=r*0.2", neg: "-=|r|*0.1"`
    /// - AP-009: Result always clamped to valid range
    ///
    /// # Examples
    ///
    /// ```rust
    /// use uuid::Uuid;
    /// use context_graph_core::types::GraphEdge;
    /// use context_graph_core::marblestone::{EdgeType, Domain};
    ///
    /// let mut edge = GraphEdge::new(
    ///     Uuid::new_v4(),
    ///     Uuid::new_v4(),
    ///     EdgeType::Semantic,
    ///     Domain::General,
    /// );
    ///
    /// // Initial modulated weight (no steering reward)
    /// let weight = edge.get_modulated_weight();
    /// assert!(weight >= 0.0 && weight <= 1.0);
    ///
    /// // Apply positive steering reward
    /// edge.apply_steering_reward(0.5);
    /// let reinforced = edge.get_modulated_weight();
    /// assert!(reinforced >= weight); // Weight increased
    /// ```
    #[inline]
    pub fn get_modulated_weight(&self) -> f32 {
        let nt_factor = self
            .neurotransmitter_weights
            .compute_effective_weight(self.weight);
        (nt_factor * (1.0 + self.steering_reward * 0.2)).clamp(0.0, 1.0)
    }

    /// Applies a steering reward signal from the Steering Subsystem.
    ///
    /// The steering reward provides reinforcement learning feedback:
    /// - Positive rewards strengthen the edge (encourage traversal)
    /// - Negative rewards weaken the edge (discourage traversal)
    ///
    /// Rewards are accumulated (additive) and clamped to [-1.0, 1.0].
    ///
    /// # Arguments
    ///
    /// * `reward` - Reward signal to add. Positive reinforces, negative discourages.
    ///
    /// # Constitution Compliance
    ///
    /// - steering.reward: `range: "[-1,1]"`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use uuid::Uuid;
    /// use context_graph_core::types::GraphEdge;
    /// use context_graph_core::marblestone::{EdgeType, Domain};
    ///
    /// let mut edge = GraphEdge::new(
    ///     Uuid::new_v4(),
    ///     Uuid::new_v4(),
    ///     EdgeType::Causal,
    ///     Domain::Research,
    /// );
    ///
    /// assert_eq!(edge.steering_reward, 0.0);
    ///
    /// // Positive feedback from successful retrieval
    /// edge.apply_steering_reward(0.3);
    /// assert_eq!(edge.steering_reward, 0.3);
    ///
    /// // Additional positive feedback accumulates
    /// edge.apply_steering_reward(0.5);
    /// assert_eq!(edge.steering_reward, 0.8);
    ///
    /// // Clamped to maximum
    /// edge.apply_steering_reward(0.5);
    /// assert_eq!(edge.steering_reward, 1.0);
    ///
    /// // Negative feedback reduces
    /// edge.apply_steering_reward(-0.3);
    /// assert_eq!(edge.steering_reward, 0.7);
    /// ```
    #[inline]
    pub fn apply_steering_reward(&mut self, reward: f32) {
        self.steering_reward = (self.steering_reward + reward).clamp(-1.0, 1.0);
    }

    /// Decay the steering reward by a factor.
    ///
    /// Used to gradually reduce influence of old rewards over time.
    /// Does NOT clamp - assumes decay_factor is in [0.0, 1.0].
    ///
    /// # Arguments
    /// * `decay_factor` - Multiplicative decay (e.g., 0.9 reduces by 10%)
    #[inline]
    pub fn decay_steering(&mut self, decay_factor: f32) {
        self.steering_reward *= decay_factor;
    }
}
