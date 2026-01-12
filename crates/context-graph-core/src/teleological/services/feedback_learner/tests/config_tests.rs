//! Tests for FeedbackLearnerConfig.

use crate::teleological::services::feedback_learner::FeedbackLearnerConfig;

#[test]
fn test_config_default() {
    let config = FeedbackLearnerConfig::default();

    assert!((config.learning_rate - 0.01).abs() < f32::EPSILON);
    assert!((config.momentum - 0.9).abs() < f32::EPSILON);
    assert!((config.reward_scale - 1.0).abs() < f32::EPSILON);
    assert!((config.penalty_scale - 0.5).abs() < f32::EPSILON);
    assert_eq!(config.min_feedback_count, 10);

    println!("[PASS] FeedbackLearnerConfig::default has correct values");
}
