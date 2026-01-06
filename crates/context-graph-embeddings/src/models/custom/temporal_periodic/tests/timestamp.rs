//! Timestamp parsing and extraction tests for TemporalPeriodicModel.

use chrono::{Datelike, Utc};

use crate::models::custom::temporal_periodic::TemporalPeriodicModel;
use crate::types::ModelInput;

#[test]
fn test_parse_timestamp_iso8601() {
    let instruction = "timestamp:2024-01-15T10:30:00Z";
    let result = TemporalPeriodicModel::parse_timestamp(instruction);

    assert!(result.is_some(), "Should parse ISO 8601");
    let dt = result.unwrap();
    assert_eq!(dt.year(), 2024);
    assert_eq!(dt.month(), 1);
    assert_eq!(dt.day(), 15);
}

#[test]
fn test_parse_timestamp_unix_epoch() {
    let instruction = "epoch:1705315800";
    let result = TemporalPeriodicModel::parse_timestamp(instruction);

    assert!(result.is_some(), "Should parse Unix epoch");
}

#[test]
fn test_parse_timestamp_invalid() {
    let invalid_inputs = vec![
        "not a timestamp",
        "timestamp:invalid",
        "epoch:notanumber",
        "random text",
        "",
    ];

    for input in invalid_inputs {
        let result = TemporalPeriodicModel::parse_timestamp(input);
        assert!(result.is_none(), "Should return None for '{}'", input);
    }
}

#[tokio::test]
async fn test_extract_timestamp_with_iso8601() {
    let model = TemporalPeriodicModel::new();
    let input = ModelInput::text_with_instruction("content", "timestamp:2024-01-15T10:30:00Z")
        .expect("Failed to create input");

    let timestamp = model.extract_timestamp(&input);

    assert_eq!(timestamp.year(), 2024);
    assert_eq!(timestamp.month(), 1);
    assert_eq!(timestamp.day(), 15);
}

#[tokio::test]
async fn test_extract_timestamp_fallback_to_now() {
    let model = TemporalPeriodicModel::new();
    let input = ModelInput::text("no timestamp").expect("Failed to create input");

    let before = Utc::now();
    let timestamp = model.extract_timestamp(&input);
    let after = Utc::now();

    assert!(timestamp >= before, "Fallback should be >= before");
    assert!(timestamp <= after, "Fallback should be <= after");
}
