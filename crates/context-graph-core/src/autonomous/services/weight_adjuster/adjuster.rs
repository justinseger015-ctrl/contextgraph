//! Weight adjuster service implementation.

use std::collections::HashMap;

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::evolution::WeightAdjustment;

use super::types::{AdjustmentReport, WeightAdjusterConfig};

/// Service for optimizing goal section weights based on performance feedback
#[derive(Debug)]
pub struct WeightAdjuster {
    /// Configuration for the adjuster
    config: WeightAdjusterConfig,
    /// Momentum velocities per goal for gradient descent
    velocities: HashMap<GoalId, f32>,
}

impl WeightAdjuster {
    /// Create a new weight adjuster with default configuration
    pub fn new() -> Self {
        Self {
            config: WeightAdjusterConfig::default(),
            velocities: HashMap::new(),
        }
    }

    /// Create a weight adjuster with custom configuration
    ///
    /// # Errors
    /// Returns error if configuration is invalid
    pub fn with_config(config: WeightAdjusterConfig) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            config,
            velocities: HashMap::new(),
        })
    }

    /// Get the current configuration
    pub fn config(&self) -> &WeightAdjusterConfig {
        &self.config
    }

    /// Compute a weight adjustment for a goal based on performance
    ///
    /// Performance is expected to be in [0.0, 1.0] where:
    /// - 1.0 = perfect alignment, weight should increase
    /// - 0.5 = neutral, no change needed
    /// - 0.0 = poor alignment, weight should decrease
    pub fn compute_adjustment(
        &self,
        goal_id: &GoalId,
        performance: f32,
        current_weight: f32,
    ) -> WeightAdjustment {
        // Target weight is current weight adjusted by performance delta from neutral (0.5)
        let performance_delta = performance - 0.5;
        let target_direction = if performance_delta > 0.0 { 1.0 } else { -1.0 };

        // Scale the adjustment by learning rate and performance magnitude
        let adjustment_magnitude = performance_delta.abs() * self.config.learning_rate;
        let raw_new_weight = current_weight + (target_direction * adjustment_magnitude);

        // Clamp to bounds
        let new_weight = self.clamp_weight(raw_new_weight);

        WeightAdjustment {
            goal_id: goal_id.clone(),
            old_weight: current_weight,
            new_weight,
            reason: crate::autonomous::evolution::AdjustmentReason::HighRetrievalActivity,
        }
    }

    /// Apply a batch of weight adjustments and return a report
    pub fn apply_adjustments(&mut self, adjustments: &[WeightAdjustment]) -> AdjustmentReport {
        let mut report = AdjustmentReport::new();

        for adjustment in adjustments {
            if self.validate_adjustment(adjustment) {
                let delta = (adjustment.new_weight - adjustment.old_weight).abs();
                report.total_delta += delta;
                report.max_delta = report.max_delta.max(delta);
                report.adjusted_goals.push(adjustment.goal_id.clone());
                report.adjustments_applied += 1;
            } else {
                let reason = self.validation_failure_reason(adjustment);
                report
                    .skipped_goals
                    .push((adjustment.goal_id.clone(), reason));
                report.adjustments_skipped += 1;
            }
        }

        // Calculate average delta
        if report.adjustments_applied > 0 {
            report.avg_delta = report.total_delta / report.adjustments_applied as f32;
        }

        report
    }

    /// Perform a single gradient step from current weight towards target
    ///
    /// Uses the formula: new_weight = current + lr * (target - current)
    pub fn gradient_step(&self, current: f32, target: f32, lr: f32) -> f32 {
        let gradient = target - current;
        current + lr * gradient
    }

    /// Clamp a weight value to the configured bounds
    pub fn clamp_weight(&self, weight: f32) -> f32 {
        weight.clamp(self.config.min_weight, self.config.max_weight)
    }

    /// Check if a performance delta is significant enough to trigger adjustment
    pub fn should_adjust(&self, performance_delta: f32) -> bool {
        performance_delta.abs() >= self.config.adjustment_threshold
    }

    /// Compute momentum-adjusted gradient for a goal
    ///
    /// Uses exponential moving average: v[t] = momentum * v[t-1] + (1 - momentum) * gradient
    pub fn compute_momentum(&mut self, goal_id: &GoalId, gradient: f32) -> f32 {
        let prev_velocity = self.velocities.get(goal_id).copied().unwrap_or(0.0);
        let new_velocity =
            self.config.momentum * prev_velocity + (1.0 - self.config.momentum) * gradient;
        self.velocities.insert(goal_id.clone(), new_velocity);
        new_velocity
    }

    /// Validate that an adjustment is within acceptable bounds
    pub fn validate_adjustment(&self, adjustment: &WeightAdjustment) -> bool {
        // Check new weight is within bounds
        if adjustment.new_weight < self.config.min_weight
            || adjustment.new_weight > self.config.max_weight
        {
            return false;
        }

        // Check old weight is valid (non-negative)
        if adjustment.old_weight < 0.0 {
            return false;
        }

        // Check for NaN values
        if adjustment.new_weight.is_nan() || adjustment.old_weight.is_nan() {
            return false;
        }

        true
    }

    /// Get the reason why an adjustment failed validation
    fn validation_failure_reason(&self, adjustment: &WeightAdjustment) -> String {
        if adjustment.new_weight.is_nan() {
            return "new_weight is NaN".to_string();
        }
        if adjustment.old_weight.is_nan() {
            return "old_weight is NaN".to_string();
        }
        if adjustment.new_weight < self.config.min_weight {
            return format!(
                "new_weight {} below min {}",
                adjustment.new_weight, self.config.min_weight
            );
        }
        if adjustment.new_weight > self.config.max_weight {
            return format!(
                "new_weight {} above max {}",
                adjustment.new_weight, self.config.max_weight
            );
        }
        if adjustment.old_weight < 0.0 {
            return format!("old_weight {} is negative", adjustment.old_weight);
        }
        "unknown validation failure".to_string()
    }

    /// Reset momentum for a specific goal
    pub fn reset_momentum(&mut self, goal_id: &GoalId) {
        self.velocities.remove(goal_id);
    }

    /// Reset all momentum values
    pub fn reset_all_momentum(&mut self) {
        self.velocities.clear();
    }

    /// Get current velocity for a goal (for debugging/monitoring)
    pub fn get_velocity(&self, goal_id: &GoalId) -> Option<f32> {
        self.velocities.get(goal_id).copied()
    }
}

impl Default for WeightAdjuster {
    fn default() -> Self {
        Self::new()
    }
}
