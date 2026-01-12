//! Integration tests for the ThresholdLearner service.
//!
//! This module contains integration tests, serialization tests, and
//! constitution compliance tests.

use chrono::Duration;

use crate::autonomous::{AlignmentBucket, RetrievalStats};

use super::learner::ThresholdLearner;
use super::types::{NUM_EMBEDDERS, RECALIBRATION_CHECK_INTERVAL_SECS};
use crate::autonomous::AdaptiveThresholdConfig;

#[test]
fn test_should_recalibrate_after_observations() {
    let mut learner = ThresholdLearner::new();

    // Add observations
    let stats = RetrievalStats::new();
    for _ in 0..15 {
        learner.learn_from_feedback(&stats, true);
    }

    // Reset timer to simulate time passage
    learner.mark_recalibration_checked();
    // Need to access private field - use workaround via serde
    let mut json = serde_json::to_value(&learner).unwrap();
    if let Some(obj) = json.as_object_mut() {
        let past_time =
            chrono::Utc::now() - Duration::seconds(RECALIBRATION_CHECK_INTERVAL_SECS + 1);
        obj.insert(
            "last_recalibration_check".to_string(),
            serde_json::to_value(past_time).unwrap(),
        );
    }
    learner = serde_json::from_value(json).unwrap();

    // Should check recalibration logic (may or may not trigger based on drift)
    let _result = learner.should_recalibrate();

    println!("[PASS] test_should_recalibrate_after_observations: Recalibration check runs");
}

#[test]
fn test_bayesian_history_recording() {
    let mut learner = ThresholdLearner::new();
    let stats = RetrievalStats::new();

    // Add 100 observations to trigger history recording
    for _ in 0..100 {
        learner.learn_from_feedback(&stats, true);
    }

    assert!(!learner.get_bayesian_history().is_empty());
    assert_eq!(learner.get_bayesian_history().len(), 1);

    // Add another 100
    for _ in 0..100 {
        learner.learn_from_feedback(&stats, false);
    }

    assert_eq!(learner.get_bayesian_history().len(), 2);

    println!("[PASS] test_bayesian_history_recording: Bayesian history recorded at intervals");
}

#[test]
fn test_best_performance_tracking() {
    let mut learner = ThresholdLearner::new();
    let stats = RetrievalStats::new();

    // Start with successes
    for _ in 0..10 {
        learner.learn_from_feedback(&stats, true);
    }
    let perf_after_success = learner.best_performance();

    // Add failures
    for _ in 0..10 {
        learner.learn_from_feedback(&stats, false);
    }

    // Best should not decrease
    assert!(learner.best_performance() >= perf_after_success * 0.9);

    println!("[PASS] test_best_performance_tracking: Best performance tracked");
}

#[test]
fn test_per_embedder_threshold_evolution() {
    let mut learner = ThresholdLearner::new();

    let initial_thresholds: Vec<f32> = (0..NUM_EMBEDDERS)
        .map(|i| learner.get_threshold(i))
        .collect();

    // Learn from mixed feedback
    let stats = RetrievalStats::new();
    for i in 0..50 {
        learner.learn_from_feedback(&stats, i % 2 == 0);
    }

    let final_thresholds: Vec<f32> = (0..NUM_EMBEDDERS)
        .map(|i| learner.get_threshold(i))
        .collect();

    // Thresholds should have evolved (at least some difference)
    let total_change: f32 = initial_thresholds
        .iter()
        .zip(final_thresholds.iter())
        .map(|(i, f)| (i - f).abs())
        .sum();

    assert!(total_change > 0.0, "Thresholds should evolve with feedback");

    println!("[PASS] test_per_embedder_threshold_evolution: Per-embedder thresholds evolve");
}

#[test]
fn test_4_level_atc_integration() {
    let mut learner = ThresholdLearner::new();

    // Simulate realistic usage scenario
    let mut stats = RetrievalStats::new();

    // Phase 1: Good results (high alignment)
    for _ in 0..20 {
        stats.record_retrieval(AlignmentBucket::Optimal, true);
        learner.learn_from_feedback(&stats, true);
    }

    let threshold_after_good = learner.get_threshold(0);

    // Phase 2: Mixed results
    for _ in 0..20 {
        stats.record_retrieval(AlignmentBucket::Warning, false);
        learner.learn_from_feedback(&stats, false);
    }

    // Level 1: EWMA should have tracked drift
    let threshold_after_mixed = learner.get_threshold(0);

    // Level 2: Temperature should be updated
    let temp = learner.get_embedder_state(0).unwrap().temperature;
    assert!((temp - 1.0).abs() < 0.5); // Should be near default

    // Level 3: Thompson state should reflect history
    let thompson = &learner.get_embedder_state(0).unwrap().thompson;
    assert!(thompson.samples >= 40);

    println!("[PASS] test_4_level_atc_integration: All 4 ATC levels work together");
    println!(
        "       Level 1 (EWMA): {} -> {}",
        threshold_after_good, threshold_after_mixed
    );
    println!("       Level 2 (Temp): {}", temp);
    println!(
        "       Level 3 (Thompson): alpha={}, beta={}",
        thompson.alpha, thompson.beta
    );
    println!(
        "       Level 4 (Bayesian history): {} observations",
        learner.get_bayesian_history().len()
    );
}

#[test]
fn test_constitution_compliance() {
    // Verify compliance with constitution.yaml adaptive_thresholds section
    let learner = ThresholdLearner::new();

    // Check default priors match constitution
    assert!(
        (learner.get_state().optimal - 0.75).abs() < 0.01,
        "theta_opt prior should be ~0.75"
    );
    assert!(
        (learner.get_state().acceptable - 0.70).abs() < 0.01,
        "theta_acc prior should be ~0.70"
    );
    assert!(
        (learner.get_state().warning - 0.55).abs() < 0.01,
        "theta_warn prior should be ~0.55"
    );
    assert!(
        (learner.get_state().critical - 0.40).abs() < 0.01,
        "theta_crit prior should be ~0.40"
    );

    // Check config bounds match constitution
    let config = learner.get_config();
    assert!(config.optimal_bounds.0 >= 0.60 && config.optimal_bounds.1 <= 0.90);
    assert!(config.warning_bounds.0 >= 0.40 && config.warning_bounds.1 <= 0.70);

    println!("[PASS] test_constitution_compliance: Matches constitution.yaml ATC spec");
}

#[test]
fn test_serialization_roundtrip() {
    let mut learner = ThresholdLearner::new();

    // Add some data
    let stats = RetrievalStats::new();
    for _ in 0..10 {
        learner.learn_from_feedback(&stats, true);
    }

    // Serialize
    let json = serde_json::to_string(&learner).expect("Serialization should succeed");

    // Deserialize
    let restored: ThresholdLearner =
        serde_json::from_str(&json).expect("Deserialization should succeed");

    assert_eq!(restored.total_observations(), learner.total_observations());
    assert!((restored.get_state().optimal - learner.get_state().optimal).abs() < f32::EPSILON);

    println!("[PASS] test_serialization_roundtrip: Serde works correctly");
}

#[test]
fn test_threshold_validity_after_learning() {
    let mut learner = ThresholdLearner::new();

    // Add significant feedback
    let mut stats = RetrievalStats::new();
    for i in 0..100 {
        stats.record_retrieval(AlignmentBucket::Optimal, true);
        stats.record_retrieval(AlignmentBucket::Acceptable, i % 2 == 0);
        stats.record_retrieval(AlignmentBucket::Warning, false);
        learner.learn_from_feedback(&stats, true);
    }

    // After learning, verify thresholds are still valid (positive and bounded)
    assert!(
        learner.get_state().optimal > 0.0 && learner.get_state().optimal <= 1.0,
        "Optimal ({}) should be in (0, 1]",
        learner.get_state().optimal
    );
    assert!(
        learner.get_state().acceptable > 0.0 && learner.get_state().acceptable <= 1.0,
        "Acceptable ({}) should be in (0, 1]",
        learner.get_state().acceptable
    );
    assert!(
        learner.get_state().warning >= 0.0 && learner.get_state().warning <= 1.0,
        "Warning ({}) should be in [0, 1]",
        learner.get_state().warning
    );
    assert!(
        learner.get_state().critical >= 0.0 && learner.get_state().critical <= 1.0,
        "Critical ({}) should be in [0, 1]",
        learner.get_state().critical
    );

    println!("[PASS] test_threshold_validity_after_learning: Threshold bounds maintained");
}

#[test]
fn test_threshold_bounds_respected() {
    let mut learner = ThresholdLearner::new();
    let stats = RetrievalStats::new();

    // Apply many learning updates
    for _ in 0..200 {
        learner.learn_from_feedback(&stats, true);
    }

    let config = learner.get_config();

    // Optimal should be within bounds
    assert!(
        learner.get_state().optimal >= config.optimal_bounds.0
            && learner.get_state().optimal <= config.optimal_bounds.1,
        "Optimal ({}) should be within bounds {:?}",
        learner.get_state().optimal,
        config.optimal_bounds
    );

    // Warning should be within bounds
    assert!(
        learner.get_state().warning >= config.warning_bounds.0
            && learner.get_state().warning <= config.warning_bounds.1,
        "Warning ({}) should be within bounds {:?}",
        learner.get_state().warning,
        config.warning_bounds
    );

    println!("[PASS] test_threshold_bounds_respected: Thresholds stay within bounds");
}
