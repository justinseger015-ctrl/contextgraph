//! Goal metrics for coverage analysis.

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::evolution::{GoalActivityMetrics, GoalLevel};

/// Goal paired with its activity metrics for analysis
#[derive(Clone, Debug)]
pub struct GoalWithMetrics {
    pub goal_id: GoalId,
    pub level: GoalLevel,
    pub description: String,
    pub parent_id: Option<GoalId>,
    pub child_ids: Vec<GoalId>,
    pub domains: Vec<String>,
    pub metrics: GoalActivityMetrics,
}

impl GoalWithMetrics {
    /// Calculate coverage score for this goal based on activity metrics
    pub fn coverage_score(&self) -> f32 {
        let activity = self.metrics.activity_score();
        let alignment = self.metrics.avg_child_alignment;
        0.6 * activity + 0.4 * alignment
    }

    /// Check if this goal has weak coverage
    pub fn has_weak_coverage(&self, threshold: f32) -> bool {
        self.coverage_score() < threshold
    }
}
