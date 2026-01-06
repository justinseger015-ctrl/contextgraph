//! Construction and initialization tests for TemporalPeriodicModel.

use crate::error::EmbeddingError;
use crate::models::custom::temporal_periodic::{
    periods, TemporalPeriodicModel, TEMPORAL_PERIODIC_DIMENSION,
};
use crate::traits::EmbeddingModel;
use crate::types::ModelId;

#[test]
fn test_new_creates_initialized_model() {
    let model = TemporalPeriodicModel::new();

    println!("BEFORE: model created");
    println!("AFTER: is_initialized = {}", model.is_initialized());

    assert!(
        model.is_initialized(),
        "Custom model must be initialized immediately"
    );
    assert_eq!(model.periods.len(), 5, "Must have 5 periods");
}

#[test]
fn test_default_periods() {
    let model = TemporalPeriodicModel::new();

    println!("Periods: {:?}", model.periods);

    assert_eq!(model.periods[0], periods::HOUR, "Hour period");
    assert_eq!(model.periods[1], periods::DAY, "Day period");
    assert_eq!(model.periods[2], periods::WEEK, "Week period");
    assert_eq!(model.periods[3], periods::MONTH, "Month period");
    assert_eq!(model.periods[4], periods::YEAR, "Year period");
}

#[test]
fn test_custom_periods_valid() {
    let custom = vec![3600, 86400, 604800, 2592000, 31536000];
    let model = TemporalPeriodicModel::with_periods(custom.clone()).expect("Should succeed");

    assert_eq!(model.periods, custom);
}

#[test]
fn test_custom_periods_wrong_count() {
    let periods = vec![3600, 86400, 604800]; // Only 3

    let result = TemporalPeriodicModel::with_periods(periods);

    assert!(result.is_err(), "Should fail with wrong count");
    match result {
        Err(EmbeddingError::ConfigError { message }) => {
            assert!(
                message.contains("exactly 5"),
                "Error should mention 5 periods"
            );
        }
        Err(other) => panic!("Expected ConfigError, got {:?}", other),
        Ok(_) => panic!("Expected ConfigError, got Ok"),
    }
}

#[test]
fn test_custom_periods_zero_invalid() {
    let periods = vec![3600, 0, 604800, 2592000, 31536000]; // Zero period

    let result = TemporalPeriodicModel::with_periods(periods);

    assert!(result.is_err(), "Zero period should fail");
}

#[test]
fn test_custom_config_valid() {
    let periods = vec![3600, 86400];
    let model =
        TemporalPeriodicModel::with_custom_config(periods.clone(), 25).expect("Should succeed");

    assert_eq!(model.periods, periods);
    assert_eq!(model.num_harmonics, 25);
}

#[test]
fn test_custom_config_empty_periods_invalid() {
    let result = TemporalPeriodicModel::with_custom_config(vec![], 51);

    assert!(result.is_err(), "Empty periods should fail");
}

#[test]
fn test_custom_config_zero_harmonics_invalid() {
    let periods = vec![3600];
    let result = TemporalPeriodicModel::with_custom_config(periods, 0);

    assert!(result.is_err(), "Zero harmonics should fail");
}

#[test]
fn test_default_impl() {
    let model = TemporalPeriodicModel::default();

    assert!(model.is_initialized());
    assert_eq!(model.periods.len(), 5);
}

#[test]
fn test_model_id_is_temporal_periodic() {
    let model = TemporalPeriodicModel::new();

    assert_eq!(model.model_id(), ModelId::TemporalPeriodic);
}

#[test]
fn test_supported_input_types() {
    use crate::types::InputType;

    let model = TemporalPeriodicModel::new();
    let types = model.supported_input_types();

    assert_eq!(types.len(), 1, "Should support exactly 1 input type");
    assert_eq!(types[0], InputType::Text, "Should support Text input");
}

#[test]
fn test_dimension_is_512() {
    let model = TemporalPeriodicModel::new();

    // dimension() uses default impl that delegates to model_id().dimension()
    assert_eq!(model.dimension(), TEMPORAL_PERIODIC_DIMENSION);
    assert_eq!(model.dimension(), 512);
}

#[test]
fn test_is_pretrained_returns_false() {
    let model = TemporalPeriodicModel::new();

    // is_pretrained() uses default impl that delegates to model_id().is_pretrained()
    // ModelId::TemporalPeriodic.is_pretrained() should return false
    assert!(!model.is_pretrained(), "Custom models are not pretrained");
}

#[test]
fn test_latency_budget_is_2ms() {
    let model = TemporalPeriodicModel::new();

    assert_eq!(model.latency_budget_ms(), 2, "Latency budget should be 2ms");
}
