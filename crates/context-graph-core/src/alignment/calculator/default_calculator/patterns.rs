//! Pattern detection for DefaultAlignmentCalculator.

use std::collections::HashSet;

use super::DefaultAlignmentCalculator;
use crate::alignment::config::AlignmentConfig;
use crate::alignment::misalignment::MisalignmentFlags;
use crate::alignment::pattern::{AlignmentPattern, PatternType};
use crate::alignment::score::GoalAlignmentScore;
use crate::config::constants::alignment as thresholds;
use crate::types::fingerprint::AlignmentThreshold;

/// Detect patterns from alignment score and flags.
pub(crate) fn detect_patterns(
    _calculator: &DefaultAlignmentCalculator,
    score: &GoalAlignmentScore,
    flags: &MisalignmentFlags,
    _config: &AlignmentConfig,
) -> Vec<AlignmentPattern> {
    let mut patterns = Vec::new();

    // Check for North Star drift (WARNING_THRESHOLD per constitution)
    if score.north_star_alignment < thresholds::WARNING {
        let pattern = AlignmentPattern::new(
            PatternType::NorthStarDrift,
            format!(
                "North Star alignment at {:.1}% is below warning threshold",
                score.north_star_alignment * 100.0
            ),
            "Review and realign content with North Star goal",
        )
        .with_severity(2);
        patterns.push(pattern);
    }

    // Check for tactical without strategic
    if flags.tactical_without_strategic {
        let pattern = AlignmentPattern::new(
            PatternType::TacticalWithoutStrategic,
            format!(
                "High tactical alignment ({:.1}%) without strategic direction ({:.1}%)",
                score.tactical_alignment * 100.0,
                score.strategic_alignment * 100.0
            ),
            "Develop strategic goals to guide tactical activities",
        )
        .with_severity(1);
        patterns.push(pattern);
    }

    // Check for critical misalignment
    if flags.below_threshold {
        let pattern = AlignmentPattern::new(
            PatternType::CriticalMisalignment,
            format!(
                "{} goal(s) below critical threshold",
                flags.critical_goals.len()
            ),
            "Immediate attention required for critically misaligned goals",
        )
        .with_affected_goals(flags.critical_goals.clone())
        .with_severity(2);
        patterns.push(pattern);
    }

    // Check for divergent hierarchy
    if flags.divergent_hierarchy {
        let pattern = AlignmentPattern::new(
            PatternType::DivergentHierarchy,
            format!(
                "{} parent-child pair(s) show divergent alignment",
                flags.divergent_pairs.len()
            ),
            "Review child goals to ensure they support parent goals",
        )
        .with_severity(2);
        patterns.push(pattern);
    }

    // Check for inconsistent alignment
    if flags.inconsistent_alignment {
        let pattern = AlignmentPattern::new(
            PatternType::InconsistentAlignment,
            format!(
                "High variance ({:.3}) in alignment across embedding spaces",
                flags.alignment_variance
            ),
            "Content may have inconsistent interpretation across domains",
        )
        .with_severity(1);
        patterns.push(pattern);
    }

    // Check for positive patterns
    if !flags.has_any() && matches!(score.threshold, AlignmentThreshold::Optimal) {
        patterns.push(AlignmentPattern::new(
            PatternType::OptimalAlignment,
            "All goals optimally aligned".to_string(),
            "Maintain current alignment practices",
        ));
    }

    // Check hierarchical coherence (ACCEPTABLE_THRESHOLD per constitution)
    if !flags.divergent_hierarchy
        && score.goal_count() > 1
        && score.composite_score >= thresholds::ACCEPTABLE
    {
        let has_multiple_levels = {
            let mut levels = HashSet::new();
            for gs in &score.goal_scores {
                levels.insert(gs.level);
            }
            levels.len() > 1
        };

        if has_multiple_levels {
            patterns.push(AlignmentPattern::new(
                PatternType::HierarchicalCoherence,
                "Goal hierarchy shows coherent alignment across levels",
                "Good hierarchical structure maintained",
            ));
        }
    }

    patterns
}
