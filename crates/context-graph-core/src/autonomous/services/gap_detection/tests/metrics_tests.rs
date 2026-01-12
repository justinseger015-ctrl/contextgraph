//! Tests for GoalWithMetrics struct.

use crate::autonomous::evolution::GoalLevel;

use super::helpers::create_test_goal;

#[test]
fn test_goal_with_metrics_coverage_score() {
    // High activity, high alignment
    let goal = create_test_goal(GoalLevel::Strategic, vec!["domain1"], 50, 25, 0.8);
    let score = goal.coverage_score();
    // activity = 0.5, alignment = 0.8
    // coverage = 0.6 * 0.5 + 0.4 * 0.8 = 0.3 + 0.32 = 0.62
    assert!((score - 0.62).abs() < 0.01);
    println!("[PASS] test_goal_with_metrics_coverage_score");
}

#[test]
fn test_goal_with_metrics_coverage_score_zero() {
    let goal = create_test_goal(GoalLevel::Operational, vec!["domain1"], 0, 0, 0.0);
    let score = goal.coverage_score();
    assert!((score - 0.0).abs() < f32::EPSILON);
    println!("[PASS] test_goal_with_metrics_coverage_score_zero");
}

#[test]
fn test_goal_with_metrics_coverage_score_max() {
    let goal = create_test_goal(GoalLevel::NorthStar, vec!["domain1"], 100, 50, 1.0);
    let score = goal.coverage_score();
    // activity = 1.0, alignment = 1.0
    // coverage = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
    assert!((score - 1.0).abs() < f32::EPSILON);
    println!("[PASS] test_goal_with_metrics_coverage_score_max");
}

#[test]
fn test_goal_with_metrics_has_weak_coverage() {
    let weak_goal = create_test_goal(GoalLevel::Tactical, vec!["domain1"], 10, 5, 0.2);
    assert!(weak_goal.has_weak_coverage(0.4));

    let strong_goal = create_test_goal(GoalLevel::Tactical, vec!["domain1"], 80, 40, 0.9);
    assert!(!strong_goal.has_weak_coverage(0.4));
    println!("[PASS] test_goal_with_metrics_has_weak_coverage");
}
