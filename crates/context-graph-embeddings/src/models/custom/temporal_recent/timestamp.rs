//! Timestamp parsing and extraction for the Temporal-Recent model.

use chrono::{DateTime, Utc};

use crate::types::ModelInput;

/// Parse timestamp from instruction string.
///
/// Supports formats:
/// - ISO 8601: "timestamp:2024-01-15T10:30:00Z"
/// - Unix epoch: "epoch:1705315800"
pub fn parse_timestamp(instruction: &str) -> Option<DateTime<Utc>> {
    // Try ISO 8601 format: "timestamp:2024-01-15T10:30:00Z"
    if let Some(ts_str) = instruction.strip_prefix("timestamp:") {
        if let Ok(dt) = DateTime::parse_from_rfc3339(ts_str.trim()) {
            return Some(dt.with_timezone(&Utc));
        }
    }

    // Try Unix epoch: "epoch:1705315800"
    if let Some(epoch_str) = instruction.strip_prefix("epoch:") {
        if let Ok(secs) = epoch_str.trim().parse::<i64>() {
            return DateTime::from_timestamp(secs, 0);
        }
    }

    None
}

/// Extract timestamp from ModelInput.
///
/// Attempts to parse timestamp from the instruction field:
/// - ISO 8601 format: "timestamp:2024-01-15T10:30:00Z"
/// - Unix epoch: "epoch:1705315800"
///
/// Falls back to current time if no valid timestamp found.
pub fn extract_timestamp(input: &ModelInput) -> DateTime<Utc> {
    match input {
        ModelInput::Text { instruction, .. } => instruction
            .as_ref()
            .and_then(|inst| parse_timestamp(inst))
            .unwrap_or_else(Utc::now),
        // For non-text inputs, use current time
        _ => Utc::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Datelike;

    #[test]
    fn test_parse_timestamp_iso8601() {
        let instruction = "timestamp:2024-01-15T10:30:00Z";
        let result = parse_timestamp(instruction);

        assert!(result.is_some(), "Should parse ISO 8601");
        let dt = result.unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 15);
    }

    #[test]
    fn test_parse_timestamp_unix_epoch() {
        let instruction = "epoch:1705315800";
        let result = parse_timestamp(instruction);

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
            let result = parse_timestamp(input);
            assert!(result.is_none(), "Should return None for '{}'", input);
        }
    }

    #[tokio::test]
    async fn test_extract_timestamp_with_iso8601() {
        let input = ModelInput::text_with_instruction("content", "timestamp:2024-01-15T10:30:00Z")
            .expect("Failed to create input");

        let timestamp = extract_timestamp(&input);

        assert_eq!(timestamp.year(), 2024);
        assert_eq!(timestamp.month(), 1);
        assert_eq!(timestamp.day(), 15);
    }

    #[tokio::test]
    async fn test_extract_timestamp_fallback_to_now() {
        let input = ModelInput::text("no timestamp").expect("Failed to create input");

        let before = Utc::now();
        let timestamp = extract_timestamp(&input);
        let after = Utc::now();

        assert!(timestamp >= before, "Fallback should be >= before");
        assert!(timestamp <= after, "Fallback should be <= after");
    }
}
