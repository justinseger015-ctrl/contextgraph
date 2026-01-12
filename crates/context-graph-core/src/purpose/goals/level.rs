//! Goal hierarchy level definition.

use serde::{Deserialize, Serialize};

/// Goal hierarchy level.
///
/// Defines the position of a goal in the hierarchical tree structure.
/// Each level has a different propagation weight for alignment computation.
///
/// From constitution.yaml:
/// - NorthStar: 1.0 weight (global goal)
/// - Strategic: 0.7 weight (mid-level)
/// - Tactical: 0.4 weight (short-term)
/// - Immediate: 0.2 weight (per-operation)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum GoalLevel {
    /// Top-level aspirational goal (the "North Star").
    /// Only one allowed per hierarchy.
    NorthStar = 0,

    /// Mid-term strategic objectives.
    /// Children of NorthStar.
    Strategic = 1,

    /// Short-term tactical goals.
    /// Children of Strategic goals.
    Tactical = 2,

    /// Immediate context goals.
    /// Lowest level, most specific.
    Immediate = 3,
}

impl GoalLevel {
    /// Weight factor for hierarchical propagation.
    ///
    /// From constitution.yaml:
    /// - NorthStar: 1.0
    /// - Strategic: 0.7
    /// - Tactical: 0.4
    /// - Immediate: 0.2
    #[inline]
    pub fn propagation_weight(&self) -> f32 {
        match self {
            GoalLevel::NorthStar => 1.0,
            GoalLevel::Strategic => 0.7,
            GoalLevel::Tactical => 0.4,
            GoalLevel::Immediate => 0.2,
        }
    }

    /// Get numeric depth (0 = NorthStar, 3 = Immediate).
    #[inline]
    pub fn depth(&self) -> u8 {
        *self as u8
    }
}
