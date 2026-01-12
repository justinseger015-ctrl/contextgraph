//! Unit tests for weight adjuster config, construction, and gradient operations.

use crate::autonomous::services::weight_adjuster::{WeightAdjuster, WeightAdjusterConfig};

// === WeightAdjusterConfig Tests ===

#[test]
fn test_config_default_values() {
    let config = WeightAdjusterConfig::default();

    assert!((config.learning_rate - 0.05).abs() < f32::EPSILON);
    assert!((config.min_weight - 0.3).abs() < f32::EPSILON);
    assert!((config.max_weight - 0.95).abs() < f32::EPSILON);
    assert!((config.momentum - 0.9).abs() < f32::EPSILON);
    assert!((config.adjustment_threshold - 0.01).abs() < f32::EPSILON);

    println!("[PASS] test_config_default_values");
}

#[test]
fn test_config_validation_valid() {
    let config = WeightAdjusterConfig {
        learning_rate: 0.1,
        min_weight: 0.1,
        max_weight: 0.9,
        momentum: 0.5,
        adjustment_threshold: 0.05,
    };

    assert!(config.validate().is_ok());
    println!("[PASS] test_config_validation_valid");
}

#[test]
fn test_config_validation_invalid_learning_rate() {
    let config = WeightAdjusterConfig {
        learning_rate: 0.0,
        ..WeightAdjusterConfig::default()
    };
    assert!(config.validate().is_err());

    let config = WeightAdjusterConfig {
        learning_rate: 1.5,
        ..WeightAdjusterConfig::default()
    };
    assert!(config.validate().is_err());

    let config = WeightAdjusterConfig {
        learning_rate: -0.1,
        ..WeightAdjusterConfig::default()
    };
    assert!(config.validate().is_err());

    println!("[PASS] test_config_validation_invalid_learning_rate");
}

#[test]
fn test_config_validation_invalid_weight_bounds() {
    // min_weight negative
    let config = WeightAdjusterConfig {
        min_weight: -0.1,
        ..WeightAdjusterConfig::default()
    };
    assert!(config.validate().is_err());

    // max_weight > 1.0
    let config = WeightAdjusterConfig {
        max_weight: 1.1,
        ..WeightAdjusterConfig::default()
    };
    assert!(config.validate().is_err());

    // min >= max
    let config = WeightAdjusterConfig {
        min_weight: 0.8,
        max_weight: 0.5,
        ..WeightAdjusterConfig::default()
    };
    assert!(config.validate().is_err());

    println!("[PASS] test_config_validation_invalid_weight_bounds");
}

#[test]
fn test_config_validation_invalid_momentum() {
    let config = WeightAdjusterConfig {
        momentum: 1.0,
        ..WeightAdjusterConfig::default()
    };
    assert!(config.validate().is_err());

    let config = WeightAdjusterConfig {
        momentum: -0.1,
        ..WeightAdjusterConfig::default()
    };
    assert!(config.validate().is_err());

    println!("[PASS] test_config_validation_invalid_momentum");
}

// === WeightAdjuster Construction Tests ===

#[test]
fn test_new_default() {
    let adjuster = WeightAdjuster::new();

    assert!((adjuster.config().learning_rate - 0.05).abs() < f32::EPSILON);

    println!("[PASS] test_new_default");
}

#[test]
fn test_with_config_valid() {
    let config = WeightAdjusterConfig {
        learning_rate: 0.1,
        min_weight: 0.2,
        max_weight: 0.8,
        momentum: 0.85,
        adjustment_threshold: 0.02,
    };

    let adjuster = WeightAdjuster::with_config(config.clone());
    assert!(adjuster.is_ok());

    let adjuster = adjuster.unwrap();
    assert!((adjuster.config().learning_rate - 0.1).abs() < f32::EPSILON);
    assert!((adjuster.config().min_weight - 0.2).abs() < f32::EPSILON);

    println!("[PASS] test_with_config_valid");
}

#[test]
fn test_with_config_invalid() {
    let config = WeightAdjusterConfig {
        learning_rate: 0.0, // Invalid
        ..WeightAdjusterConfig::default()
    };

    let result = WeightAdjuster::with_config(config);
    assert!(result.is_err());

    println!("[PASS] test_with_config_invalid");
}

// === Gradient Step Tests ===

#[test]
fn test_gradient_step_towards_target() {
    let adjuster = WeightAdjuster::new();

    // Moving from 0.5 towards 0.8 with lr=0.1
    // Expected: 0.5 + 0.1 * (0.8 - 0.5) = 0.5 + 0.03 = 0.53
    let result = adjuster.gradient_step(0.5, 0.8, 0.1);
    assert!((result - 0.53).abs() < 1e-6);

    // Moving from 0.8 towards 0.5 with lr=0.1
    // Expected: 0.8 + 0.1 * (0.5 - 0.8) = 0.8 - 0.03 = 0.77
    let result = adjuster.gradient_step(0.8, 0.5, 0.1);
    assert!((result - 0.77).abs() < 1e-6);

    println!("[PASS] test_gradient_step_towards_target");
}

#[test]
fn test_gradient_step_at_target() {
    let adjuster = WeightAdjuster::new();

    // Already at target, no change
    let result = adjuster.gradient_step(0.6, 0.6, 0.1);
    assert!((result - 0.6).abs() < f32::EPSILON);

    println!("[PASS] test_gradient_step_at_target");
}

#[test]
fn test_gradient_step_zero_learning_rate() {
    let adjuster = WeightAdjuster::new();

    // Zero lr means no change
    let result = adjuster.gradient_step(0.5, 0.9, 0.0);
    assert!((result - 0.5).abs() < f32::EPSILON);

    println!("[PASS] test_gradient_step_zero_learning_rate");
}

#[test]
fn test_gradient_step_full_learning_rate() {
    let adjuster = WeightAdjuster::new();

    // lr=1.0 means jump directly to target
    let result = adjuster.gradient_step(0.5, 0.9, 1.0);
    assert!((result - 0.9).abs() < 1e-6);

    println!("[PASS] test_gradient_step_full_learning_rate");
}

// === Clamp Weight Tests ===

#[test]
fn test_clamp_weight_within_bounds() {
    let adjuster = WeightAdjuster::new();

    let result = adjuster.clamp_weight(0.6);
    assert!((result - 0.6).abs() < f32::EPSILON);

    println!("[PASS] test_clamp_weight_within_bounds");
}

#[test]
fn test_clamp_weight_below_min() {
    let adjuster = WeightAdjuster::new();

    let result = adjuster.clamp_weight(0.1);
    assert!((result - 0.3).abs() < f32::EPSILON); // min is 0.3

    let result = adjuster.clamp_weight(-0.5);
    assert!((result - 0.3).abs() < f32::EPSILON);

    println!("[PASS] test_clamp_weight_below_min");
}

#[test]
fn test_clamp_weight_above_max() {
    let adjuster = WeightAdjuster::new();

    let result = adjuster.clamp_weight(0.99);
    assert!((result - 0.95).abs() < f32::EPSILON); // max is 0.95

    let result = adjuster.clamp_weight(1.5);
    assert!((result - 0.95).abs() < f32::EPSILON);

    println!("[PASS] test_clamp_weight_above_max");
}

#[test]
fn test_clamp_weight_at_boundaries() {
    let adjuster = WeightAdjuster::new();

    let result = adjuster.clamp_weight(0.3);
    assert!((result - 0.3).abs() < f32::EPSILON);

    let result = adjuster.clamp_weight(0.95);
    assert!((result - 0.95).abs() < f32::EPSILON);

    println!("[PASS] test_clamp_weight_at_boundaries");
}

// === Should Adjust Tests ===

#[test]
fn test_should_adjust_above_threshold() {
    let adjuster = WeightAdjuster::new();

    assert!(adjuster.should_adjust(0.05)); // Above 0.01 threshold
    assert!(adjuster.should_adjust(-0.05)); // Negative also above threshold
    assert!(adjuster.should_adjust(0.01)); // Exactly at threshold

    println!("[PASS] test_should_adjust_above_threshold");
}

#[test]
fn test_should_adjust_below_threshold() {
    let adjuster = WeightAdjuster::new();

    assert!(!adjuster.should_adjust(0.005)); // Below 0.01 threshold
    assert!(!adjuster.should_adjust(-0.005));
    assert!(!adjuster.should_adjust(0.0));

    println!("[PASS] test_should_adjust_below_threshold");
}
