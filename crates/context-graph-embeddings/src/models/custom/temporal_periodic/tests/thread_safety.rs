//! Thread safety and constants tests for TemporalPeriodicModel.

use crate::models::custom::temporal_periodic::{
    periods, TemporalPeriodicModel, DEFAULT_PERIODS, FEATURES_PER_PERIOD, HARMONICS_PER_PERIOD,
    TEMPORAL_PERIODIC_DIMENSION,
};

#[test]
fn test_model_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<TemporalPeriodicModel>();
}

#[test]
fn test_model_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<TemporalPeriodicModel>();
}

#[test]
fn test_constants_are_correct() {
    assert_eq!(TEMPORAL_PERIODIC_DIMENSION, 512);
    assert_eq!(DEFAULT_PERIODS.len(), 5);
    assert_eq!(HARMONICS_PER_PERIOD, 51);
    assert_eq!(FEATURES_PER_PERIOD, 102);

    // Verify period constants
    assert_eq!(periods::HOUR, 3600);
    assert_eq!(periods::DAY, 86400);
    assert_eq!(periods::WEEK, 604800);
    assert_eq!(periods::MONTH, 2592000);
    assert_eq!(periods::YEAR, 31536000);
}

#[test]
fn test_dimension_calculation() {
    // 5 periods * 51 harmonics * 2 (sin/cos) = 510 features + 2 padding = 512
    let expected = DEFAULT_PERIODS.len() * HARMONICS_PER_PERIOD * 2;
    assert_eq!(expected, 510, "Raw features should be 510");
    assert!(
        expected <= TEMPORAL_PERIODIC_DIMENSION,
        "Should fit in 512D"
    );
}
