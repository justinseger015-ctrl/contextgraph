//! Misalignment detection for DefaultAlignmentCalculator.

use std::collections::HashMap;
use tracing::{debug, warn};
use uuid::Uuid;

use super::DefaultAlignmentCalculator;
use crate::alignment::misalignment::{MisalignmentFlags, MisalignmentThresholds};
use crate::alignment::pattern::EmbedderBreakdown;
use crate::alignment::score::GoalAlignmentScore;
use crate::purpose::GoalHierarchy;
use crate::types::fingerprint::AlignmentThreshold;

impl DefaultAlignmentCalculator {
    /// Detect misalignment flags from scores.
    pub(crate) fn detect_misalignment_flags(
        &self,
        score: &GoalAlignmentScore,
        thresholds: &MisalignmentThresholds,
        hierarchy: &GoalHierarchy,
    ) -> MisalignmentFlags {
        let mut flags = MisalignmentFlags::empty();

        // Check tactical without strategic
        if thresholds
            .is_tactical_without_strategic(score.tactical_alignment, score.strategic_alignment)
        {
            flags.tactical_without_strategic = true;
            warn!(
                tactical = score.tactical_alignment,
                strategic = score.strategic_alignment,
                "Tactical without strategic pattern detected"
            );
        }

        // Check for critical/warning goals
        for goal_score in &score.goal_scores {
            match goal_score.threshold {
                AlignmentThreshold::Critical => {
                    flags.mark_below_threshold(goal_score.goal_id);
                }
                AlignmentThreshold::Warning => {
                    flags.mark_warning(goal_score.goal_id);
                }
                _ => {}
            }
        }

        // Check divergent hierarchy
        self.check_divergent_hierarchy(&mut flags, score, hierarchy, thresholds);

        flags
    }

    /// Check for divergent parent-child alignment.
    fn check_divergent_hierarchy(
        &self,
        flags: &mut MisalignmentFlags,
        score: &GoalAlignmentScore,
        hierarchy: &GoalHierarchy,
        thresholds: &MisalignmentThresholds,
    ) {
        // Build a map of goal_id -> alignment
        let alignment_map: HashMap<Uuid, f32> = score
            .goal_scores
            .iter()
            .map(|s| (s.goal_id, s.alignment))
            .collect();

        // Check each goal against its parent
        for goal_score in &score.goal_scores {
            if let Some(goal) = hierarchy.get(&goal_score.goal_id) {
                if let Some(parent_id) = goal.parent_id {
                    if let Some(&parent_alignment) = alignment_map.get(&parent_id) {
                        if thresholds.is_divergent(parent_alignment, goal_score.alignment) {
                            flags.mark_divergent(parent_id, goal_score.goal_id);
                            warn!(
                                parent = %parent_id,
                                child = %goal_score.goal_id,
                                parent_alignment = parent_alignment,
                                child_alignment = goal_score.alignment,
                                "Divergent hierarchy detected"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Check for inconsistent alignment across embedders.
    pub(crate) fn check_inconsistent_alignment(
        &self,
        flags: &mut MisalignmentFlags,
        breakdown: &EmbedderBreakdown,
        thresholds: &MisalignmentThresholds,
    ) {
        let variance = breakdown.std_dev.powi(2);
        if thresholds.is_inconsistent(variance) {
            flags.mark_inconsistent(variance);
            debug!(
                variance = variance,
                std_dev = breakdown.std_dev,
                "Inconsistent alignment detected across embedders"
            );
        }
    }
}
