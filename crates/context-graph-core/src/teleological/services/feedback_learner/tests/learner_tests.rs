//! Tests for FeedbackLearner core functionality.

use uuid::Uuid;

use crate::teleological::services::feedback_learner::{
    FeedbackEvent, FeedbackLearner, FeedbackLearnerConfig, FeedbackType,
};
use crate::teleological::types::NUM_EMBEDDERS;

#[test]
fn test_feedback_learner_new() {
    let learner = FeedbackLearner::new();

    assert_eq!(learner.feedback_buffer_size(), 0);
    assert_eq!(learner.total_events_processed(), 0);
    assert!(!learner.should_learn());

    for i in 0..NUM_EMBEDDERS {
        assert!((learner.get_adjustment_for_embedder(i) - 0.0).abs() < f32::EPSILON);
    }

    println!("[PASS] FeedbackLearner::new creates empty learner");
}

#[test]
fn test_feedback_learner_with_config() {
    let config = FeedbackLearnerConfig {
        learning_rate: 0.05,
        momentum: 0.8,
        reward_scale: 2.0,
        penalty_scale: 1.0,
        min_feedback_count: 5,
    };

    let learner = FeedbackLearner::with_config(config.clone());

    assert!((learner.config().learning_rate - 0.05).abs() < f32::EPSILON);
    assert!((learner.config().momentum - 0.8).abs() < f32::EPSILON);
    assert_eq!(learner.config().min_feedback_count, 5);

    println!("[PASS] FeedbackLearner::with_config uses custom config");
}

#[test]
#[should_panic(expected = "FAIL FAST")]
fn test_feedback_learner_invalid_learning_rate() {
    let config = FeedbackLearnerConfig {
        learning_rate: 0.0, // Invalid
        ..Default::default()
    };
    let _ = FeedbackLearner::with_config(config);
}

#[test]
#[should_panic(expected = "FAIL FAST")]
fn test_feedback_learner_invalid_momentum() {
    let config = FeedbackLearnerConfig {
        momentum: 1.0, // Invalid (must be < 1.0)
        ..Default::default()
    };
    let _ = FeedbackLearner::with_config(config);
}

#[test]
fn test_record_feedback() {
    let mut learner = FeedbackLearner::new();

    for i in 0..5 {
        let event = FeedbackEvent::new(
            Uuid::new_v4(),
            FeedbackType::Positive { magnitude: 0.7 },
            i as u64,
        );
        learner.record_feedback(event);
    }

    assert_eq!(learner.feedback_buffer_size(), 5);
    assert!(!learner.should_learn()); // Need 10 by default

    println!("[PASS] record_feedback adds events to buffer");
}

#[test]
fn test_should_learn_threshold() {
    let mut learner = FeedbackLearner::new();

    for i in 0..9 {
        let event = FeedbackEvent::new(Uuid::new_v4(), FeedbackType::Neutral, i as u64);
        learner.record_feedback(event);
    }

    assert!(!learner.should_learn()); // 9 < 10

    let event = FeedbackEvent::new(Uuid::new_v4(), FeedbackType::Neutral, 9);
    learner.record_feedback(event);

    assert!(learner.should_learn()); // 10 >= 10

    println!("[PASS] should_learn returns true when buffer >= min_feedback_count");
}

#[test]
fn test_apply_gradient() {
    let mut learner = FeedbackLearner::new();

    let gradient = vec![0.5; NUM_EMBEDDERS];

    learner.apply_gradient(&gradient);

    // After first application: m = (1 - 0.9) * 0.5 = 0.05
    // adjustment += 0.01 * 0.05 = 0.0005
    let adj = learner.get_adjustment_for_embedder(0);
    assert!(adj > 0.0, "Adjustment should be positive");
    assert!((adj - 0.0005).abs() < 0.0001);

    println!("[PASS] apply_gradient updates adjustments with momentum");
    println!("  - adjustment[0] = {:.6}", adj);
}

#[test]
fn test_apply_gradient_momentum_accumulation() {
    let mut learner = FeedbackLearner::new();

    let gradient = vec![1.0; NUM_EMBEDDERS];

    // Apply same gradient multiple times
    for _ in 0..10 {
        learner.apply_gradient(&gradient);
    }

    let adj = learner.get_adjustment_for_embedder(0);

    // Momentum should cause adjustments to grow faster than without momentum
    assert!(adj > 0.005, "Adjustment should accumulate, got {}", adj);

    println!("[PASS] apply_gradient accumulates momentum over iterations");
    println!("  - adjustment[0] after 10 iterations = {:.6}", adj);
}

#[test]
fn test_learn_empty_buffer() {
    let mut learner = FeedbackLearner::new();

    let result = learner.learn();

    assert_eq!(result.events_processed, 0);
    assert!(!result.has_adjustments());

    println!("[PASS] learn() handles empty buffer gracefully");
}

#[test]
fn test_reset_feedback_buffer() {
    let mut learner = FeedbackLearner::new();

    for i in 0..5 {
        let event = FeedbackEvent::new(Uuid::new_v4(), FeedbackType::Neutral, i as u64);
        learner.record_feedback(event);
    }

    assert_eq!(learner.feedback_buffer_size(), 5);

    learner.reset_feedback_buffer();

    assert_eq!(learner.feedback_buffer_size(), 0);

    println!("[PASS] reset_feedback_buffer clears buffer");
}

#[test]
fn test_reset_all_state() {
    let mut learner = FeedbackLearner::new();

    // Add some events and learn
    let uniform = [1.0 / NUM_EMBEDDERS as f32; NUM_EMBEDDERS];
    for i in 0..15 {
        let event = FeedbackEvent::new(
            Uuid::new_v4(),
            FeedbackType::Positive { magnitude: 1.0 },
            i as u64,
        )
        .with_contributions(uniform);
        learner.record_feedback(event);
    }
    learner.learn();

    assert!(learner.total_events_processed() > 0);
    assert!(learner.get_adjustment_for_embedder(0) != 0.0);

    learner.reset();

    assert_eq!(learner.feedback_buffer_size(), 0);
    assert_eq!(learner.total_events_processed(), 0);
    assert!((learner.cumulative_confidence_delta() - 0.0).abs() < f32::EPSILON);
    for i in 0..NUM_EMBEDDERS {
        assert!((learner.get_adjustment_for_embedder(i) - 0.0).abs() < f32::EPSILON);
    }

    println!("[PASS] reset() clears all state");
}

#[test]
fn test_adjustment_clamping() {
    let config = FeedbackLearnerConfig {
        learning_rate: 1.0, // High learning rate to force clamping
        momentum: 0.0,
        min_feedback_count: 1,
        ..Default::default()
    };
    let mut learner = FeedbackLearner::with_config(config);

    // Apply extreme positive gradient
    let gradient = vec![10.0; NUM_EMBEDDERS];
    for _ in 0..10 {
        learner.apply_gradient(&gradient);
    }

    let adj = learner.get_adjustment_for_embedder(0);
    assert!(
        adj <= 1.0,
        "Adjustment should be clamped to <= 1.0, got {}",
        adj
    );
    assert!((adj - 1.0).abs() < f32::EPSILON);

    // Apply extreme negative gradient
    let gradient = vec![-20.0; NUM_EMBEDDERS];
    for _ in 0..10 {
        learner.apply_gradient(&gradient);
    }

    let adj = learner.get_adjustment_for_embedder(0);
    assert!(
        adj >= -1.0,
        "Adjustment should be clamped to >= -1.0, got {}",
        adj
    );
    assert!((adj - (-1.0)).abs() < f32::EPSILON);

    println!("[PASS] Adjustments are clamped to [-1.0, 1.0]");
}

#[test]
fn test_total_events_tracking() {
    let config = FeedbackLearnerConfig {
        min_feedback_count: 3,
        ..Default::default()
    };
    let mut learner = FeedbackLearner::with_config(config);

    // First batch
    for i in 0..5 {
        let event = FeedbackEvent::new(Uuid::new_v4(), FeedbackType::Neutral, i as u64);
        learner.record_feedback(event);
    }
    learner.learn();
    assert_eq!(learner.total_events_processed(), 5);

    // Second batch
    for i in 0..7 {
        let event = FeedbackEvent::new(Uuid::new_v4(), FeedbackType::Neutral, i as u64);
        learner.record_feedback(event);
    }
    learner.learn();
    assert_eq!(learner.total_events_processed(), 12);

    println!("[PASS] total_events_processed accumulates across learn() calls");
}

#[test]
fn test_cumulative_confidence_delta() {
    let config = FeedbackLearnerConfig {
        min_feedback_count: 2,
        ..Default::default()
    };
    let mut learner = FeedbackLearner::with_config(config);

    // Positive feedback increases confidence
    let uniform = [1.0 / NUM_EMBEDDERS as f32; NUM_EMBEDDERS];
    for _ in 0..5 {
        let event =
            FeedbackEvent::new(Uuid::new_v4(), FeedbackType::Positive { magnitude: 1.0 }, 0)
                .with_contributions(uniform);
        learner.record_feedback(event);
    }
    learner.learn();

    let delta1 = learner.cumulative_confidence_delta();
    assert!(delta1 > 0.0, "Positive feedback should increase confidence");

    // Negative feedback decreases confidence
    for _ in 0..5 {
        let event =
            FeedbackEvent::new(Uuid::new_v4(), FeedbackType::Negative { magnitude: 1.0 }, 0)
                .with_contributions(uniform);
        learner.record_feedback(event);
    }
    learner.learn();

    let delta2 = learner.cumulative_confidence_delta();
    assert!(
        delta2 < delta1,
        "Negative feedback should decrease cumulative delta"
    );

    println!("[PASS] cumulative_confidence_delta tracks confidence changes");
    println!("  - After positive: {:.4}", delta1);
    println!("  - After negative: {:.4}", delta2);
}

#[test]
#[should_panic(expected = "FAIL FAST")]
fn test_get_adjustment_out_of_bounds() {
    let learner = FeedbackLearner::new();
    let _ = learner.get_adjustment_for_embedder(13);
}

#[test]
fn test_default_impl() {
    let learner = FeedbackLearner::default();
    assert_eq!(learner.feedback_buffer_size(), 0);

    println!("[PASS] FeedbackLearner::default() works");
}
