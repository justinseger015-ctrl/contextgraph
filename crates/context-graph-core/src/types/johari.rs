//! Johari Window quadrant classification and modality types.

use serde::{Deserialize, Serialize};

/// Johari Window quadrant classification.
///
/// The Johari Window is a psychological model for understanding self-awareness
/// and interpersonal relationships. In the context graph, it classifies memories
/// based on their visibility and awareness states.
///
/// # Quadrants
///
/// - **Open**: Known to self and others - readily accessible knowledge
/// - **Blind**: Known to others, unknown to self - insights from external feedback
/// - **Hidden**: Known to self, hidden from others - private knowledge
/// - **Unknown**: Unknown to both self and others - exploration frontier
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum JohariQuadrant {
    /// Known to self and others
    Open,
    /// Known to others, unknown to self
    Blind,
    /// Known to self, hidden from others
    Hidden,
    /// Unknown to both self and others
    #[default]
    Unknown,
}

impl JohariQuadrant {
    /// Returns true if this quadrant represents known information.
    pub fn is_known(&self) -> bool {
        matches!(self, Self::Open | Self::Hidden)
    }

    /// Returns true if this quadrant is visible to others.
    pub fn is_visible(&self) -> bool {
        matches!(self, Self::Open | Self::Blind)
    }
}

/// Input modality classification.
///
/// Classifies the type of content stored in a memory node.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Plain text content
    #[default]
    Text,
    /// Source code
    Code,
    /// Image data
    Image,
    /// Audio data
    Audio,
    /// Structured data (JSON, XML, etc.)
    Structured,
    /// Mixed modalities
    Mixed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_johari_default() {
        assert_eq!(JohariQuadrant::default(), JohariQuadrant::Unknown);
    }

    #[test]
    fn test_johari_is_known() {
        assert!(JohariQuadrant::Open.is_known());
        assert!(JohariQuadrant::Hidden.is_known());
        assert!(!JohariQuadrant::Blind.is_known());
        assert!(!JohariQuadrant::Unknown.is_known());
    }

    #[test]
    fn test_modality_default() {
        assert_eq!(Modality::default(), Modality::Text);
    }
}
