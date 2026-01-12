//! Test helpers for gap detection tests.

use chrono::Utc;

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::evolution::{GoalActivityMetrics, GoalLevel};
use crate::autonomous::services::gap_detection::GoalWithMetrics;

pub fn create_test_metrics(
    goal_id: GoalId,
    memories: u32,
    retrievals: u32,
    alignment: f32,
) -> GoalActivityMetrics {
    GoalActivityMetrics {
        goal_id,
        new_aligned_memories_30d: memories,
        retrievals_14d: retrievals,
        avg_child_alignment: alignment,
        weight_trend: 0.0,
        last_activity: Utc::now(),
    }
}

pub fn create_test_goal(
    level: GoalLevel,
    domains: Vec<&str>,
    memories: u32,
    retrievals: u32,
    alignment: f32,
) -> GoalWithMetrics {
    let goal_id = GoalId::new();
    GoalWithMetrics {
        goal_id: goal_id.clone(),
        level,
        description: "Test goal".into(),
        parent_id: None,
        child_ids: vec![],
        domains: domains.into_iter().map(String::from).collect(),
        metrics: create_test_metrics(goal_id, memories, retrievals, alignment),
    }
}
