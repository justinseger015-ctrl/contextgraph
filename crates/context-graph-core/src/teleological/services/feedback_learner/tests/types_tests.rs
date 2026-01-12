//! Tests for FeedbackType, FeedbackEvent, and LearningResult types.

use uuid::Uuid;

use crate::teleological::services::feedback_learner::{
    FeedbackEvent, FeedbackLearnerConfig, FeedbackType, LearningResult,
};
use crate::teleological::types::NUM_EMBEDDERS;

// ===== FeedbackType Tests =====

#[test]
fn test_feedback_type_positive_reward() {
    let config = FeedbackLearnerConfig::default();
    let ft = FeedbackType::Positive { magnitude: 0.8 };

    let reward = ft.reward_value(&config);
    // 0.8 * 1.0 = 0.8
    assert!((reward - 0.8).abs() < f32::EPSILON);
    assert!(ft.is_positive());
    assert!(!ft.is_negative());
    assert!(!ft.is_neutral());

    println!("[PASS] FeedbackType::Positive computes correct reward");
}

#[test]
fn test_feedback_type_negative_reward() {
    let config = FeedbackLearnerConfig::default();
    let ft = FeedbackType::Negative { magnitude: 0.6 };

    let reward = ft.reward_value(&config);
    // -0.6 * 0.5 = -0.3
    assert!((reward - (-0.3)).abs() < f32::EPSILON);
    assert!(!ft.is_positive());
    assert!(ft.is_negative());
    assert!(!ft.is_neutral());

    println!("[PASS] FeedbackType::Negative computes correct penalty");
}

#[test]
fn test_feedback_type_neutral_reward() {
    let config = FeedbackLearnerConfig::default();
    let ft = FeedbackType::Neutral;

    let reward = ft.reward_value(&config);
    assert!((reward - 0.0).abs() < f32::EPSILON);
    assert!(!ft.is_positive());
    assert!(!ft.is_negative());
    assert!(ft.is_neutral());

    println!("[PASS] FeedbackType::Neutral returns zero reward");
}

// ===== FeedbackEvent Tests =====

#[test]
fn test_feedback_event_new() {
    let id = Uuid::new_v4();
    let event = FeedbackEvent::new(id, FeedbackType::Positive { magnitude: 0.9 }, 12345);

    assert_eq!(event.vector_id, id);
    assert_eq!(event.timestamp, 12345);
    assert!(event.context.is_none());

    // Default contributions should be uniform
    let expected = 1.0 / NUM_EMBEDDERS as f32;
    for contrib in &event.embedder_contributions {
        assert!((*contrib - expected).abs() < 0.001);
    }

    println!("[PASS] FeedbackEvent::new creates event with uniform contributions");
}

#[test]
fn test_feedback_event_with_context() {
    let event = FeedbackEvent::new(Uuid::new_v4(), FeedbackType::Neutral, 1000)
        .with_context("test context");

    assert_eq!(event.context, Some("test context".to_string()));

    println!("[PASS] FeedbackEvent::with_context sets context");
}

#[test]
fn test_feedback_event_with_contributions() {
    let mut contributions = [0.0f32; NUM_EMBEDDERS];
    contributions[0] = 0.5;
    contributions[5] = 0.3;
    contributions[12] = 0.2;

    let event = FeedbackEvent::new(
        Uuid::new_v4(),
        FeedbackType::Positive { magnitude: 1.0 },
        2000,
    )
    .with_contributions(contributions);

    assert!((event.embedder_contributions[0] - 0.5).abs() < f32::EPSILON);
    assert!((event.embedder_contributions[5] - 0.3).abs() < f32::EPSILON);
    assert!((event.embedder_contributions[12] - 0.2).abs() < f32::EPSILON);

    println!("[PASS] FeedbackEvent::with_contributions sets custom contributions");
}

#[test]
#[should_panic(expected = "FAIL FAST")]
fn test_feedback_event_contributions_must_sum_to_one() {
    let contributions = [0.1f32; NUM_EMBEDDERS]; // Sums to 1.3, not 1.0

    let _ = FeedbackEvent::new(Uuid::new_v4(), FeedbackType::Neutral, 0)
        .with_contributions(contributions);
}

// ===== LearningResult Tests =====

#[test]
fn test_learning_result_new() {
    let adjustments = vec![0.1; NUM_EMBEDDERS];
    let result = LearningResult::new(adjustments.clone(), 0.05, 20);

    assert_eq!(result.adjustments, adjustments);
    assert!((result.confidence_delta - 0.05).abs() < f32::EPSILON);
    assert_eq!(result.events_processed, 20);
    assert!(result.has_adjustments());

    println!("[PASS] LearningResult::new creates valid result");
}

#[test]
fn test_learning_result_no_adjustments() {
    let result = LearningResult::new(vec![0.0; NUM_EMBEDDERS], 0.0, 0);
    assert!(!result.has_adjustments());

    println!("[PASS] LearningResult::has_adjustments returns false for zero adjustments");
}

#[test]
fn test_learning_result_total_magnitude() {
    let mut adjustments = vec![0.0; NUM_EMBEDDERS];
    adjustments[0] = 0.3;
    adjustments[1] = -0.2;
    adjustments[5] = 0.1;

    let result = LearningResult::new(adjustments, 0.0, 3);
    let magnitude = result.total_adjustment_magnitude();

    // |0.3| + |-0.2| + |0.1| = 0.6
    assert!((magnitude - 0.6).abs() < f32::EPSILON);

    println!("[PASS] LearningResult::total_adjustment_magnitude computes L1 norm");
}

#[test]
#[should_panic(expected = "FAIL FAST")]
fn test_learning_result_wrong_length() {
    let _ = LearningResult::new(vec![0.1; 5], 0.0, 0); // Wrong length
}
