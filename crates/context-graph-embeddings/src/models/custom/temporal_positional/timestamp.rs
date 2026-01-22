//! Timestamp and sequence position parsing utilities for the Temporal-Positional model (E4).
//!
//! E4 encodes session sequence positions to enable "before/after" queries within a session.
//! This module supports both sequence-based positions (preferred) and timestamp-based positions
//! (fallback for backward compatibility).

use chrono::{DateTime, Utc};

use crate::types::ModelInput;

/// Position information extracted from input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionInfo {
    /// The position value (sequence number or Unix timestamp)
    pub position: i64,
    /// True if position is a session sequence number, false if Unix timestamp
    pub is_sequence: bool,
}

impl PositionInfo {
    /// Create a new sequence-based position.
    #[must_use]
    pub fn sequence(seq: u64) -> Self {
        Self {
            position: seq as i64,
            is_sequence: true,
        }
    }

    /// Create a new timestamp-based position.
    #[must_use]
    pub fn timestamp(ts: i64) -> Self {
        Self {
            position: ts,
            is_sequence: false,
        }
    }

    /// Create a position from current time.
    #[must_use]
    pub fn now() -> Self {
        Self::timestamp(Utc::now().timestamp())
    }
}

/// Extract position from ModelInput for E4 embedding.
///
/// Priority order:
/// 1. "sequence:N" - Session sequence number (preferred for E4)
/// 2. "timestamp:ISO8601" - ISO 8601 timestamp
/// 3. "epoch:N" - Unix epoch seconds
/// 4. Falls back to current time
///
/// Returns `PositionInfo` indicating whether position is sequence-based or timestamp-based.
pub fn extract_position(input: &ModelInput) -> PositionInfo {
    match input {
        ModelInput::Text { instruction, .. } => instruction
            .as_ref()
            .and_then(|inst| parse_position(inst))
            .unwrap_or_else(PositionInfo::now),
        // For non-text inputs, use current time
        _ => PositionInfo::now(),
    }
}

/// Parse position from instruction string.
///
/// Priority order (first match wins):
/// 1. "sequence:N" -> (N, is_sequence=true) - Session sequence number
/// 2. "timestamp:ISO8601" -> (unix_secs, is_sequence=false)
/// 3. "epoch:N" -> (N, is_sequence=false)
///
/// # Returns
/// `Some(PositionInfo)` if parsing succeeded, `None` otherwise.
pub fn parse_position(instruction: &str) -> Option<PositionInfo> {
    // Priority 1: Sequence number (e.g., "sequence:123")
    if let Some(seq_str) = instruction.strip_prefix("sequence:") {
        if let Ok(seq) = seq_str.trim().parse::<u64>() {
            return Some(PositionInfo::sequence(seq));
        }
    }

    // Priority 2: ISO 8601 timestamp (e.g., "timestamp:2024-01-15T10:30:00Z")
    if let Some(ts_str) = instruction.strip_prefix("timestamp:") {
        if let Ok(dt) = DateTime::parse_from_rfc3339(ts_str.trim()) {
            return Some(PositionInfo::timestamp(dt.with_timezone(&Utc).timestamp()));
        }
    }

    // Priority 3: Unix epoch (e.g., "epoch:1705315800")
    if let Some(epoch_str) = instruction.strip_prefix("epoch:") {
        if let Ok(secs) = epoch_str.trim().parse::<i64>() {
            return Some(PositionInfo::timestamp(secs));
        }
    }

    None
}

/// Extract timestamp from ModelInput (legacy API for backward compatibility).
///
/// Attempts to parse timestamp from the instruction field:
/// - ISO 8601 format: "timestamp:2024-01-15T10:30:00Z"
/// - Unix epoch: "epoch:1705315800"
///
/// Falls back to current time if no valid timestamp found.
///
/// Note: For new code, prefer `extract_position()` which also supports sequence numbers.
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

/// Parse timestamp from instruction string (legacy API for backward compatibility).
///
/// Supports formats:
/// - ISO 8601: "timestamp:2024-01-15T10:30:00Z"
/// - Unix epoch: "epoch:1705315800"
///
/// Note: For new code, prefer `parse_position()` which also supports sequence numbers.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sequence() {
        let pos = parse_position("sequence:42").unwrap();
        assert_eq!(pos.position, 42);
        assert!(pos.is_sequence);
    }

    #[test]
    fn test_parse_sequence_large() {
        let pos = parse_position("sequence:9999999").unwrap();
        assert_eq!(pos.position, 9999999);
        assert!(pos.is_sequence);
    }

    #[test]
    fn test_parse_epoch() {
        let pos = parse_position("epoch:1705315800").unwrap();
        assert_eq!(pos.position, 1705315800);
        assert!(!pos.is_sequence);
    }

    #[test]
    fn test_parse_timestamp_iso() {
        let pos = parse_position("timestamp:2024-01-15T10:30:00Z").unwrap();
        // 2024-01-15T10:30:00Z = 1705314600 Unix seconds
        assert_eq!(pos.position, 1705314600);
        assert!(!pos.is_sequence);
    }

    #[test]
    fn test_sequence_priority_over_timestamp() {
        // If both formats could match (they can't syntactically, but test priority)
        let pos = parse_position("sequence:100").unwrap();
        assert!(pos.is_sequence);
    }

    #[test]
    fn test_invalid_instruction() {
        assert!(parse_position("invalid").is_none());
        assert!(parse_position("sequence:").is_none());
        assert!(parse_position("sequence:abc").is_none());
    }

    #[test]
    fn test_position_info_constructors() {
        let seq = PositionInfo::sequence(100);
        assert_eq!(seq.position, 100);
        assert!(seq.is_sequence);

        let ts = PositionInfo::timestamp(1705315800);
        assert_eq!(ts.position, 1705315800);
        assert!(!ts.is_sequence);
    }
}
