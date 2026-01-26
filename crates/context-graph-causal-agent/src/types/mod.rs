//! Type definitions for the Causal Discovery Agent.
//!
//! Contains structures for causal analysis results, candidate pairs,
//! and configuration options.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Result of LLM causal relationship analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalAnalysisResult {
    /// Whether a causal link was detected.
    pub has_causal_link: bool,

    /// Direction of the causal relationship.
    pub direction: CausalLinkDirection,

    /// Confidence score [0.0, 1.0].
    pub confidence: f32,

    /// Description of the causal mechanism.
    pub mechanism: String,

    /// Raw LLM response (for debugging).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<String>,
}

impl Default for CausalAnalysisResult {
    fn default() -> Self {
        Self {
            has_causal_link: false,
            direction: CausalLinkDirection::NoCausalLink,
            confidence: 0.0,
            mechanism: String::new(),
            raw_response: None,
        }
    }
}

/// Direction of a causal relationship between two memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLinkDirection {
    /// Memory A causes Memory B (A → B).
    ACausesB,

    /// Memory B causes Memory A (B → A).
    BCausesA,

    /// Bidirectional causation (A ↔ B).
    Bidirectional,

    /// No causal relationship detected.
    NoCausalLink,
}

impl CausalLinkDirection {
    /// Parse from LLM response string.
    pub fn from_str(s: &str) -> Self {
        let lower = s.to_lowercase();
        if lower.contains("a_causes_b") || lower.contains("a causes b") || lower == "forward" {
            Self::ACausesB
        } else if lower.contains("b_causes_a") || lower.contains("b causes a") || lower == "backward"
        {
            Self::BCausesA
        } else if lower.contains("bidirectional") || lower.contains("mutual") || lower == "both" {
            Self::Bidirectional
        } else {
            Self::NoCausalLink
        }
    }

    /// Whether this represents an actual causal link.
    pub fn is_causal(&self) -> bool {
        !matches!(self, Self::NoCausalLink)
    }
}

/// A candidate pair of memories for causal analysis.
#[derive(Debug, Clone)]
pub struct CausalCandidate {
    /// Potential cause memory ID.
    pub cause_memory_id: Uuid,

    /// Potential cause memory content.
    pub cause_content: String,

    /// Potential effect memory ID.
    pub effect_memory_id: Uuid,

    /// Potential effect memory content.
    pub effect_content: String,

    /// Initial score based on heuristics (causal markers, temporal order).
    pub initial_score: f32,

    /// Timestamp of the earlier memory.
    pub earlier_timestamp: DateTime<Utc>,

    /// Timestamp of the later memory.
    pub later_timestamp: DateTime<Utc>,
}

impl CausalCandidate {
    /// Create a new causal candidate.
    pub fn new(
        cause_id: Uuid,
        cause_content: String,
        effect_id: Uuid,
        effect_content: String,
        score: f32,
        earlier: DateTime<Utc>,
        later: DateTime<Utc>,
    ) -> Self {
        Self {
            cause_memory_id: cause_id,
            cause_content,
            effect_memory_id: effect_id,
            effect_content,
            initial_score: score,
            earlier_timestamp: earlier,
            later_timestamp: later,
        }
    }
}

/// Memory representation for causal analysis.
#[derive(Debug, Clone)]
pub struct MemoryForAnalysis {
    /// Memory UUID.
    pub id: Uuid,

    /// Text content.
    pub content: String,

    /// Creation timestamp.
    pub created_at: DateTime<Utc>,

    /// Session ID (if available).
    pub session_id: Option<String>,

    /// E1 semantic embedding (for clustering).
    pub e1_embedding: Vec<f32>,
}

/// Causal markers used for initial scoring.
pub struct CausalMarkers;

impl CausalMarkers {
    /// Words/phrases that indicate causation.
    pub const CAUSE_MARKERS: &'static [&'static str] = &[
        "because",
        "due to",
        "caused by",
        "result of",
        "led to",
        "since",
        "as a result of",
        "owing to",
        "thanks to",
        "on account of",
        "stems from",
        "arises from",
        "originates from",
        "triggered by",
    ];

    /// Words/phrases that indicate effects.
    pub const EFFECT_MARKERS: &'static [&'static str] = &[
        "therefore",
        "consequently",
        "as a result",
        "thus",
        "hence",
        "so",
        "accordingly",
        "resulting in",
        "leading to",
        "causing",
        "produces",
        "yields",
        "generates",
        "brings about",
    ];

    /// Check if text contains any cause markers.
    pub fn has_cause_marker(text: &str) -> bool {
        let lower = text.to_lowercase();
        Self::CAUSE_MARKERS.iter().any(|m| lower.contains(m))
    }

    /// Check if text contains any effect markers.
    pub fn has_effect_marker(text: &str) -> bool {
        let lower = text.to_lowercase();
        Self::EFFECT_MARKERS.iter().any(|m| lower.contains(m))
    }

    /// Count causal markers in text.
    pub fn count_markers(text: &str) -> usize {
        let lower = text.to_lowercase();
        let cause_count = Self::CAUSE_MARKERS
            .iter()
            .filter(|m| lower.contains(*m))
            .count();
        let effect_count = Self::EFFECT_MARKERS
            .iter()
            .filter(|m| lower.contains(*m))
            .count();
        cause_count + effect_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_direction_parsing() {
        assert_eq!(
            CausalLinkDirection::from_str("A_causes_B"),
            CausalLinkDirection::ACausesB
        );
        assert_eq!(
            CausalLinkDirection::from_str("B_causes_A"),
            CausalLinkDirection::BCausesA
        );
        assert_eq!(
            CausalLinkDirection::from_str("bidirectional"),
            CausalLinkDirection::Bidirectional
        );
        assert_eq!(
            CausalLinkDirection::from_str("none"),
            CausalLinkDirection::NoCausalLink
        );
    }

    #[test]
    fn test_causal_markers() {
        assert!(CausalMarkers::has_cause_marker("This happened because of X"));
        assert!(CausalMarkers::has_effect_marker("Therefore, Y occurred"));
        assert!(!CausalMarkers::has_cause_marker("Hello world"));
        assert_eq!(
            CausalMarkers::count_markers("Because of X, therefore Y"),
            2
        );
    }
}
