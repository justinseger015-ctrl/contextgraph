//! Gap type definitions for coverage analysis.

use serde::{Deserialize, Serialize};

use crate::autonomous::bootstrap::GoalId;

/// Types of coverage gaps detected in the goal hierarchy
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GapType {
    /// A domain/topic area with no goal coverage
    UncoveredDomain { domain: String },
    /// A goal with insufficient coverage (below threshold)
    WeakCoverage { goal_id: GoalId, coverage: f32 },
    /// Missing hierarchical or semantic link between goals
    MissingLink { from: GoalId, to: GoalId },
    /// Period without goal activity
    TemporalGap { period: String },
}

impl GapType {
    /// Get the severity of this gap (0.0-1.0, higher is more severe)
    pub fn severity(&self) -> f32 {
        match self {
            GapType::UncoveredDomain { .. } => 0.9,
            GapType::WeakCoverage { coverage, .. } => 1.0 - coverage,
            GapType::MissingLink { .. } => 0.6,
            GapType::TemporalGap { .. } => 0.4,
        }
    }

    /// Get a human-readable description of the gap
    pub fn description(&self) -> String {
        match self {
            GapType::UncoveredDomain { domain } => {
                format!("Domain '{}' has no goal coverage", domain)
            }
            GapType::WeakCoverage { goal_id, coverage } => {
                format!(
                    "Goal {} has weak coverage at {:.1}%",
                    goal_id.0,
                    coverage * 100.0
                )
            }
            GapType::MissingLink { from, to } => {
                format!("Missing link between goals {} and {}", from.0, to.0)
            }
            GapType::TemporalGap { period } => {
                format!("No activity during period: {}", period)
            }
        }
    }
}
