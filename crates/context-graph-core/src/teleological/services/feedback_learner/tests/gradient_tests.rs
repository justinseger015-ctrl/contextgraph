//! Tests for gradient computation and learning cycles.

use uuid::Uuid;

use crate::teleological::services::feedback_learner::{
    FeedbackEvent, FeedbackLearner, FeedbackLearnerConfig, FeedbackType,
};
use crate::teleological::types::NUM_EMBEDDERS;

#[test]
fn test_compute_gradient_positive_feedback() {
    let learner = FeedbackLearner::new();

    // Create events with known contributions
    let mut contributions = [0.0f32; NUM_EMBEDDERS];
    contributions[0] = 1.0; // All contribution to embedder 0

    let events: Vec<FeedbackEvent> = (0..5)
        .map(|i| {
            FeedbackEvent::new(
                Uuid::new_v4(),
                FeedbackType::Positive { magnitude: 1.0 },
                i as u64,
            )
            .with_contributions(contributions)
        })
        .collect();

    let gradient = learner.compute_gradient(&events);

    // Gradient[0] should be positive (reward = 1.0 * 1.0 * 1.0 = 1.0, avg = 1.0)
    assert!(
        gradient[0] > 0.0,
        "Gradient[0] = {} should be positive",
        gradient[0]
    );
    // Other embedders should have zero gradient
    for grad in gradient.iter().skip(1) {
        assert!((*grad - 0.0).abs() < f32::EPSILON);
    }

    println!("[PASS] compute_gradient produces positive gradient for positive feedback");
    println!("  - gradient[0] = {:.4}", gradient[0]);
}

#[test]
fn test_compute_gradient_negative_feedback() {
    let learner = FeedbackLearner::new();

    let mut contributions = [0.0f32; NUM_EMBEDDERS];
    contributions[5] = 1.0; // All contribution to embedder 5

    let events: Vec<FeedbackEvent> = (0..3)
        .map(|i| {
            FeedbackEvent::new(
                Uuid::new_v4(),
                FeedbackType::Negative { magnitude: 0.8 },
                i as u64,
            )
            .with_contributions(contributions)
        })
        .collect();

    let gradient = learner.compute_gradient(&events);

    // Gradient[5] should be negative (penalty = -0.8 * 0.5 * 1.0 = -0.4)
    assert!(
        gradient[5] < 0.0,
        "Gradient[5] = {} should be negative",
        gradient[5]
    );

    println!("[PASS] compute_gradient produces negative gradient for negative feedback");
    println!("  - gradient[5] = {:.4}", gradient[5]);
}

#[test]
fn test_compute_gradient_mixed_feedback() {
    let learner = FeedbackLearner::new();

    // Uniform contributions
    let uniform = [1.0 / NUM_EMBEDDERS as f32; NUM_EMBEDDERS];

    let mut events = Vec::new();
    // 3 positive
    for i in 0..3 {
        events.push(
            FeedbackEvent::new(
                Uuid::new_v4(),
                FeedbackType::Positive { magnitude: 1.0 },
                i as u64,
            )
            .with_contributions(uniform),
        );
    }
    // 2 negative
    for i in 3..5 {
        events.push(
            FeedbackEvent::new(
                Uuid::new_v4(),
                FeedbackType::Negative { magnitude: 1.0 },
                i as u64,
            )
            .with_contributions(uniform),
        );
    }

    let gradient = learner.compute_gradient(&events);

    // Net reward per embedder: (3 * 1.0 - 2 * 0.5) / 5 / 13 = (3 - 1) / 5 / 13 = 0.031
    // All embedders should have same small positive gradient
    for &g in &gradient {
        assert!(g > 0.0, "Gradient should be positive, got {}", g);
    }

    println!("[PASS] compute_gradient handles mixed feedback correctly");
}

#[test]
fn test_learn_full_cycle() {
    let config = FeedbackLearnerConfig {
        min_feedback_count: 5,
        ..Default::default()
    };
    let mut learner = FeedbackLearner::with_config(config);

    // Add positive feedback with bias toward embedder 0
    let mut contributions = [0.0f32; NUM_EMBEDDERS];
    contributions[0] = 0.8;
    contributions[1] = 0.2;

    for i in 0..10 {
        let event = FeedbackEvent::new(
            Uuid::new_v4(),
            FeedbackType::Positive { magnitude: 0.9 },
            i as u64,
        )
        .with_contributions(contributions);
        learner.record_feedback(event);
    }

    assert!(learner.should_learn());

    let result = learner.learn();

    assert_eq!(result.events_processed, 10);
    assert!(result.has_adjustments());
    assert!(result.confidence_delta > 0.0); // Positive feedback = positive delta
    assert_eq!(learner.feedback_buffer_size(), 0); // Buffer cleared

    // Embedder 0 should have higher adjustment than embedder 1
    let adj0 = learner.get_adjustment_for_embedder(0);
    let adj1 = learner.get_adjustment_for_embedder(1);
    assert!(adj0 > adj1, "adj0={} should be > adj1={}", adj0, adj1);

    println!("[PASS] learn() performs full learning cycle");
    println!("  - events_processed: {}", result.events_processed);
    println!("  - confidence_delta: {:.4}", result.confidence_delta);
    println!("  - adjustment[0]: {:.6}", adj0);
    println!("  - adjustment[1]: {:.6}", adj1);
}

#[test]
fn test_gradient_calculation_real_scenario() {
    let learner = FeedbackLearner::new();

    // Simulate retrieval where embedders 0, 3, 5 contributed most
    let mut contributions = [0.0f32; NUM_EMBEDDERS];
    contributions[0] = 0.4;
    contributions[3] = 0.35;
    contributions[5] = 0.25;

    // Create 20 events: 15 positive, 5 negative
    let mut events = Vec::new();
    for i in 0..15 {
        events.push(
            FeedbackEvent::new(
                Uuid::new_v4(),
                FeedbackType::Positive { magnitude: 0.8 },
                i as u64,
            )
            .with_contributions(contributions),
        );
    }
    for i in 15..20 {
        events.push(
            FeedbackEvent::new(
                Uuid::new_v4(),
                FeedbackType::Negative { magnitude: 0.6 },
                i as u64,
            )
            .with_contributions(contributions),
        );
    }

    let gradient = learner.compute_gradient(&events);

    // Net reward per event:
    // Positive: 0.8 * 1.0 = 0.8, 15 events
    // Negative: -0.6 * 0.5 = -0.3, 5 events
    // Total reward: 15 * 0.8 + 5 * (-0.3) = 12 - 1.5 = 10.5
    // Average reward per event: 10.5 / 20 = 0.525

    // Expected gradient for embedder 0: 0.4 * 0.525 = 0.21
    let expected_g0 = 0.4 * 0.525;
    assert!(
        (gradient[0] - expected_g0).abs() < 0.01,
        "gradient[0] = {} (expected ~{})",
        gradient[0],
        expected_g0
    );

    // Expected gradient for embedder 3: 0.35 * 0.525 = 0.18375
    let expected_g3 = 0.35 * 0.525;
    assert!(
        (gradient[3] - expected_g3).abs() < 0.01,
        "gradient[3] = {} (expected ~{})",
        gradient[3],
        expected_g3
    );

    // Embedders with zero contribution should have zero gradient
    assert!(
        gradient[1].abs() < f32::EPSILON,
        "gradient[1] should be 0, got {}",
        gradient[1]
    );

    println!("[PASS] Real gradient calculation matches expected values");
    println!(
        "  - gradient[0] = {:.6} (expected {:.6})",
        gradient[0], expected_g0
    );
    println!(
        "  - gradient[3] = {:.6} (expected {:.6})",
        gradient[3], expected_g3
    );
    println!("  - gradient[5] = {:.6}", gradient[5]);
}

#[test]
fn test_multiple_learn_cycles_convergence() {
    let config = FeedbackLearnerConfig {
        min_feedback_count: 5,
        learning_rate: 0.1,
        momentum: 0.9,
        ..Default::default()
    };
    let mut learner = FeedbackLearner::with_config(config);

    // Simulate consistent positive feedback for embedder 0
    let mut contributions = [0.0f32; NUM_EMBEDDERS];
    contributions[0] = 1.0;

    for cycle in 0..10 {
        for _ in 0..10 {
            let event = FeedbackEvent::new(
                Uuid::new_v4(),
                FeedbackType::Positive { magnitude: 1.0 },
                cycle as u64,
            )
            .with_contributions(contributions);
            learner.record_feedback(event);
        }

        learner.learn();
    }

    let final_adj = learner.get_adjustment_for_embedder(0);

    // With consistent positive feedback (lr=0.1, momentum=0.9, 10 cycles of 10 events),
    // mathematical convergence: momentum converges to 1.0, adjustment accumulates as
    // sum of lr * m_i where m_i follows geometric series. After 10 cycles: ~0.41
    // Expected range: [0.35, 0.50] based on momentum-SGD convergence properties
    assert!(
        final_adj > 0.35,
        "After 10 cycles, adjustment should be substantial (>0.35), got {}",
        final_adj
    );
    assert!(
        final_adj < 0.50,
        "After only 10 cycles, adjustment should not yet exceed 0.50, got {}",
        final_adj
    );

    println!("[PASS] Multiple learning cycles show convergence");
    println!("  - Final adjustment[0] = {:.4}", final_adj);
    println!(
        "  - Total events processed = {}",
        learner.total_events_processed()
    );
}
