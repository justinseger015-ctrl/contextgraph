//! Johari transition types: TransitionTrigger and JohariTransition.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

use super::quadrant::JohariQuadrant;

/// Triggers that cause Johari quadrant transitions.
///
/// Each trigger represents a specific event that moves knowledge
/// between quadrants in the Johari Window model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionTrigger {
    /// User explicitly shares hidden knowledge (Hidden → Open).
    ExplicitShare,
    /// Agent recognizes pattern in blind spot (Blind → Open/Hidden).
    SelfRecognition,
    /// Dream consolidation discovers new patterns (Unknown → Open).
    PatternDiscovery,
    /// User marks knowledge as private (Open → Hidden).
    Privatize,
    /// External observation reveals blind spot (Unknown → Blind).
    ExternalObservation,
    /// Dream consolidation surfaces unknown knowledge (Unknown → Open/Hidden).
    DreamConsolidation,
}

impl TransitionTrigger {
    /// Returns a human-readable description of this trigger.
    pub fn description(&self) -> &'static str {
        match self {
            Self::ExplicitShare => "User explicitly shares hidden knowledge",
            Self::SelfRecognition => "Agent recognizes pattern in blind spot",
            Self::PatternDiscovery => "Dream consolidation discovers new patterns",
            Self::Privatize => "User marks knowledge as private",
            Self::ExternalObservation => "External observation reveals blind spot",
            Self::DreamConsolidation => "Dream consolidation surfaces unknown knowledge",
        }
    }

    /// Returns all trigger variants as a fixed-size array.
    pub fn all() -> [TransitionTrigger; 6] {
        [
            Self::ExplicitShare,
            Self::SelfRecognition,
            Self::PatternDiscovery,
            Self::Privatize,
            Self::ExternalObservation,
            Self::DreamConsolidation,
        ]
    }
}

impl fmt::Display for TransitionTrigger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExplicitShare => write!(f, "ExplicitShare"),
            Self::SelfRecognition => write!(f, "SelfRecognition"),
            Self::PatternDiscovery => write!(f, "PatternDiscovery"),
            Self::Privatize => write!(f, "Privatize"),
            Self::ExternalObservation => write!(f, "ExternalObservation"),
            Self::DreamConsolidation => write!(f, "DreamConsolidation"),
        }
    }
}

/// Record of a Johari quadrant transition.
///
/// Captures the complete context of a knowledge reclassification event,
/// including source/target quadrants, trigger, timestamp, memory ID, and embedder index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JohariTransition {
    /// Memory ID this transition belongs to.
    pub memory_id: Uuid,
    /// Embedder index (0-12) that transitioned.
    pub embedder_idx: usize,
    /// Starting quadrant.
    pub from: JohariQuadrant,
    /// Ending quadrant.
    pub to: JohariQuadrant,
    /// What triggered this transition.
    pub trigger: TransitionTrigger,
    /// When this transition occurred.
    pub timestamp: DateTime<Utc>,
}

impl JohariTransition {
    /// Create a new transition record with current UTC timestamp.
    ///
    /// # Arguments
    /// * `memory_id` - UUID of the memory this transition belongs to
    /// * `embedder_idx` - Embedder index (0-12) that transitioned
    /// * `from` - Source quadrant
    /// * `to` - Target quadrant
    /// * `trigger` - Event that caused the transition
    ///
    /// # Example
    /// ```
    /// use uuid::Uuid;
    /// use context_graph_core::types::{JohariQuadrant, TransitionTrigger, JohariTransition};
    /// let memory_id = Uuid::new_v4();
    /// let t = JohariTransition::new(
    ///     memory_id,
    ///     0,
    ///     JohariQuadrant::Hidden,
    ///     JohariQuadrant::Open,
    ///     TransitionTrigger::ExplicitShare
    /// );
    /// assert_eq!(t.memory_id, memory_id);
    /// assert_eq!(t.embedder_idx, 0);
    /// assert_eq!(t.from, JohariQuadrant::Hidden);
    /// assert_eq!(t.to, JohariQuadrant::Open);
    /// ```
    pub fn new(
        memory_id: Uuid,
        embedder_idx: usize,
        from: JohariQuadrant,
        to: JohariQuadrant,
        trigger: TransitionTrigger,
    ) -> Self {
        Self {
            memory_id,
            embedder_idx,
            from,
            to,
            trigger,
            timestamp: Utc::now(),
        }
    }
}
