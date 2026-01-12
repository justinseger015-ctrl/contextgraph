//! Tests for GapDetectionConfig struct.

use crate::autonomous::services::gap_detection::GapDetectionConfig;

#[test]
fn test_gap_detection_config_default() {
    let config = GapDetectionConfig::default();
    assert!((config.coverage_threshold - 0.4).abs() < f32::EPSILON);
    assert!((config.activity_threshold - 0.2).abs() < f32::EPSILON);
    assert_eq!(config.min_goals_per_domain, 1);
    assert_eq!(config.inactivity_days, 14);
    assert!(config.detect_missing_links);
    assert!((config.link_similarity_threshold - 0.7).abs() < f32::EPSILON);
    println!("[PASS] test_gap_detection_config_default");
}

#[test]
fn test_gap_detection_config_custom() {
    let config = GapDetectionConfig {
        coverage_threshold: 0.5,
        activity_threshold: 0.3,
        min_goals_per_domain: 2,
        inactivity_days: 7,
        detect_missing_links: false,
        link_similarity_threshold: 0.8,
    };
    assert!((config.coverage_threshold - 0.5).abs() < f32::EPSILON);
    assert!((config.activity_threshold - 0.3).abs() < f32::EPSILON);
    assert_eq!(config.min_goals_per_domain, 2);
    assert_eq!(config.inactivity_days, 7);
    assert!(!config.detect_missing_links);
    println!("[PASS] test_gap_detection_config_custom");
}

#[test]
fn test_gap_detection_config_serialization() {
    let config = GapDetectionConfig::default();
    let json = serde_json::to_string(&config).expect("serialize");
    let deserialized: GapDetectionConfig = serde_json::from_str(&json).expect("deserialize");
    assert!((config.coverage_threshold - deserialized.coverage_threshold).abs() < f32::EPSILON);
    assert_eq!(
        config.min_goals_per_domain,
        deserialized.min_goals_per_domain
    );
    println!("[PASS] test_gap_detection_config_serialization");
}
