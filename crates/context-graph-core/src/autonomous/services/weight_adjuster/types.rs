//! Types and configuration for the weight adjuster service.

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::evolution::GoalEvolutionConfig;

/// Reason for weight adjustment with extended variants for the service
#[derive(Clone, Debug, PartialEq)]
pub enum AdjustmentReason {
    /// Adjustment based on alignment performance metrics
    PerformanceBased { performance_delta: f32 },
    /// Correcting drift from target alignment
    DriftCorrection { drift_magnitude: f32 },
    /// User explicitly adjusted the weight
    UserFeedback { magnitude: f32 },
    /// Weight evolved based on goal hierarchy changes
    EvolutionBased { evolution_score: f32 },
    /// Scheduled periodic weight rebalancing
    Scheduled { cycle_id: u32 },
}

impl AdjustmentReason {
    /// Convert to a simple description string
    pub fn description(&self) -> String {
        match self {
            Self::PerformanceBased { performance_delta } => {
                format!("Performance-based (delta: {:.4})", performance_delta)
            }
            Self::DriftCorrection { drift_magnitude } => {
                format!("Drift correction (magnitude: {:.4})", drift_magnitude)
            }
            Self::UserFeedback { magnitude } => {
                format!("User feedback (magnitude: {:.4})", magnitude)
            }
            Self::EvolutionBased { evolution_score } => {
                format!("Evolution-based (score: {:.4})", evolution_score)
            }
            Self::Scheduled { cycle_id } => {
                format!("Scheduled (cycle: {})", cycle_id)
            }
        }
    }
}

/// Configuration for the weight adjuster service
#[derive(Clone, Debug)]
pub struct WeightAdjusterConfig {
    /// Learning rate for gradient updates (default from GoalEvolutionConfig: 0.05)
    pub learning_rate: f32,
    /// Minimum allowed weight (default: 0.3)
    pub min_weight: f32,
    /// Maximum allowed weight (default: 0.95)
    pub max_weight: f32,
    /// Momentum coefficient for gradient updates (default: 0.9)
    pub momentum: f32,
    /// Minimum performance delta to trigger adjustment
    pub adjustment_threshold: f32,
}

impl Default for WeightAdjusterConfig {
    fn default() -> Self {
        let evolution_config = GoalEvolutionConfig::default();
        Self {
            learning_rate: evolution_config.weight_lr,
            min_weight: evolution_config.weight_bounds.0,
            max_weight: evolution_config.weight_bounds.1,
            momentum: 0.9,
            adjustment_threshold: 0.01,
        }
    }
}

impl WeightAdjusterConfig {
    /// Validate configuration values
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err("learning_rate must be in (0.0, 1.0]");
        }
        if self.min_weight < 0.0 {
            return Err("min_weight must be non-negative");
        }
        if self.max_weight > 1.0 {
            return Err("max_weight must not exceed 1.0");
        }
        if self.min_weight >= self.max_weight {
            return Err("min_weight must be less than max_weight");
        }
        if self.momentum < 0.0 || self.momentum >= 1.0 {
            return Err("momentum must be in [0.0, 1.0)");
        }
        if self.adjustment_threshold < 0.0 {
            return Err("adjustment_threshold must be non-negative");
        }
        Ok(())
    }
}

/// Report summarizing weight adjustments applied
#[derive(Clone, Debug)]
pub struct AdjustmentReport {
    /// Number of adjustments applied
    pub adjustments_applied: usize,
    /// Number of adjustments skipped (failed validation)
    pub adjustments_skipped: usize,
    /// Total weight delta (sum of absolute deltas)
    pub total_delta: f32,
    /// Average delta per adjustment
    pub avg_delta: f32,
    /// Maximum absolute delta
    pub max_delta: f32,
    /// Goals that were adjusted
    pub adjusted_goals: Vec<GoalId>,
    /// Goals that were skipped with reasons
    pub skipped_goals: Vec<(GoalId, String)>,
}

impl AdjustmentReport {
    /// Create a new empty report
    pub fn new() -> Self {
        Self {
            adjustments_applied: 0,
            adjustments_skipped: 0,
            total_delta: 0.0,
            avg_delta: 0.0,
            max_delta: 0.0,
            adjusted_goals: Vec::new(),
            skipped_goals: Vec::new(),
        }
    }

    /// Check if any adjustments were made
    pub fn has_adjustments(&self) -> bool {
        self.adjustments_applied > 0
    }

    /// Check if all adjustments were successful
    pub fn all_successful(&self) -> bool {
        self.adjustments_skipped == 0
    }
}

impl Default for AdjustmentReport {
    fn default() -> Self {
        Self::new()
    }
}
