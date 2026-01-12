//! Alignment result types.

use super::super::misalignment::MisalignmentFlags;
use super::super::pattern::{AlignmentPattern, EmbedderBreakdown};
use super::super::score::GoalAlignmentScore;

/// Result of alignment computation.
///
/// Contains the full alignment score plus optional extras
/// (patterns, embedder breakdown) based on config.
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// The computed alignment score.
    pub score: GoalAlignmentScore,

    /// Detected misalignment flags.
    pub flags: MisalignmentFlags,

    /// Detected patterns (if pattern detection enabled).
    pub patterns: Vec<AlignmentPattern>,

    /// Per-embedder breakdown (if enabled in config).
    pub embedder_breakdown: Option<EmbedderBreakdown>,

    /// Computation time in microseconds.
    pub computation_time_us: u64,
}

impl AlignmentResult {
    /// Check if alignment is healthy (no critical issues).
    #[inline]
    pub fn is_healthy(&self) -> bool {
        !self.flags.needs_intervention() && !self.score.has_critical()
    }

    /// Check if alignment needs attention (warnings present).
    #[inline]
    pub fn needs_attention(&self) -> bool {
        self.flags.has_any() || self.score.has_misalignment()
    }

    /// Get overall severity (0 = healthy, 1 = warning, 2 = critical).
    pub fn severity(&self) -> u8 {
        if self.flags.needs_intervention() || self.score.has_critical() {
            2
        } else if self.flags.has_any() || self.score.has_misalignment() {
            1
        } else {
            0
        }
    }
}
