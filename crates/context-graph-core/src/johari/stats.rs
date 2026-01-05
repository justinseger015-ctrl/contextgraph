//! Transition statistics types for Johari operations.
//!
//! This module provides types for tracking and aggregating statistics about
//! Johari quadrant transitions over time.

use std::collections::HashMap;

use crate::types::{JohariQuadrant, TransitionTrigger};

/// Number of embedders in the system (E1-E13).
const NUM_EMBEDDERS: usize = 13;

/// Transition path (from → to).
///
/// Represents a specific state transition in the Johari state machine.
/// Used as a key for aggregating transition counts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TransitionPath {
    /// Source quadrant
    pub from: JohariQuadrant,
    /// Target quadrant
    pub to: JohariQuadrant,
}

impl TransitionPath {
    /// Create a new transition path.
    #[inline]
    pub fn new(from: JohariQuadrant, to: JohariQuadrant) -> Self {
        Self { from, to }
    }

    /// Get a human-readable representation of this transition.
    pub fn display_name(&self) -> String {
        format!("{:?} → {:?}", self.from, self.to)
    }
}

/// Statistics about Johari transitions over a time period.
///
/// Aggregates various metrics about transition patterns to enable
/// analysis of memory evolution behavior.
#[derive(Debug, Clone, Default)]
pub struct TransitionStats {
    /// Total transitions in the period
    pub total_transitions: u64,

    /// Unique memories with at least one transition
    pub memories_affected: u64,

    /// Counts per transition path (e.g., Hidden→Open: 42)
    pub path_counts: HashMap<TransitionPath, u64>,

    /// Counts per trigger type
    pub trigger_counts: HashMap<TransitionTrigger, u64>,

    /// Counts per embedder index (0-12)
    pub embedder_counts: [u64; NUM_EMBEDDERS],

    /// Average transitions per affected memory
    pub avg_transitions_per_memory: f32,

    /// Top 5 most common paths
    pub top_paths: Vec<(TransitionPath, u64)>,
}

impl TransitionStats {
    /// Create new empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single transition.
    ///
    /// # Arguments
    /// * `from` - Source quadrant
    /// * `to` - Target quadrant
    /// * `trigger` - What triggered the transition
    /// * `embedder_idx` - Which embedder space transitioned
    pub fn record(
        &mut self,
        from: JohariQuadrant,
        to: JohariQuadrant,
        trigger: TransitionTrigger,
        embedder_idx: usize,
    ) {
        self.total_transitions += 1;

        let path = TransitionPath::new(from, to);
        *self.path_counts.entry(path).or_insert(0) += 1;

        *self.trigger_counts.entry(trigger).or_insert(0) += 1;

        if embedder_idx < NUM_EMBEDDERS {
            self.embedder_counts[embedder_idx] += 1;
        }
    }

    /// Record that a memory was affected by transitions.
    pub fn record_affected_memory(&mut self) {
        self.memories_affected += 1;
    }

    /// Finalize stats by computing derived fields.
    ///
    /// Call this after recording all transitions to compute:
    /// - `avg_transitions_per_memory`
    /// - `top_paths` (top 5 most common transitions)
    pub fn finalize(&mut self) {
        // Compute average
        if self.memories_affected > 0 {
            self.avg_transitions_per_memory =
                self.total_transitions as f32 / self.memories_affected as f32;
        }

        // Compute top paths
        let mut paths: Vec<_> = self.path_counts.iter().collect();
        paths.sort_by(|a, b| b.1.cmp(a.1));

        self.top_paths = paths
            .into_iter()
            .take(5)
            .map(|(p, c)| (p.clone(), *c))
            .collect();
    }

    /// Get the most common transition path.
    pub fn most_common_path(&self) -> Option<&TransitionPath> {
        self.top_paths.first().map(|(p, _)| p)
    }

    /// Get the most common trigger.
    pub fn most_common_trigger(&self) -> Option<TransitionTrigger> {
        self.trigger_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(trigger, _)| *trigger)
    }

    /// Get the most active embedder (most transitions).
    pub fn most_active_embedder(&self) -> usize {
        self.embedder_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| *count)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Get transition count for a specific path.
    pub fn count_for_path(&self, from: JohariQuadrant, to: JohariQuadrant) -> u64 {
        let path = TransitionPath::new(from, to);
        *self.path_counts.get(&path).unwrap_or(&0)
    }

    /// Get transition count for a specific trigger.
    pub fn count_for_trigger(&self, trigger: TransitionTrigger) -> u64 {
        *self.trigger_counts.get(&trigger).unwrap_or(&0)
    }

    /// Get transition count for a specific embedder.
    pub fn count_for_embedder(&self, embedder_idx: usize) -> u64 {
        if embedder_idx < NUM_EMBEDDERS {
            self.embedder_counts[embedder_idx]
        } else {
            0
        }
    }

    /// Merge another TransitionStats into this one.
    pub fn merge(&mut self, other: &TransitionStats) {
        self.total_transitions += other.total_transitions;
        self.memories_affected += other.memories_affected;

        for (path, count) in &other.path_counts {
            *self.path_counts.entry(path.clone()).or_insert(0) += count;
        }

        for (trigger, count) in &other.trigger_counts {
            *self.trigger_counts.entry(*trigger).or_insert(0) += count;
        }

        for (i, count) in other.embedder_counts.iter().enumerate() {
            self.embedder_counts[i] += count;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_path() {
        let path = TransitionPath::new(JohariQuadrant::Hidden, JohariQuadrant::Open);
        assert_eq!(path.from, JohariQuadrant::Hidden);
        assert_eq!(path.to, JohariQuadrant::Open);
        assert_eq!(path.display_name(), "Hidden → Open");

        println!("[VERIFIED] test_transition_path: Path creation and display work correctly");
    }

    #[test]
    fn test_transition_stats_record() {
        let mut stats = TransitionStats::new();

        stats.record(
            JohariQuadrant::Hidden,
            JohariQuadrant::Open,
            TransitionTrigger::ExplicitShare,
            0,
        );

        stats.record(
            JohariQuadrant::Hidden,
            JohariQuadrant::Open,
            TransitionTrigger::ExplicitShare,
            0,
        );

        stats.record(
            JohariQuadrant::Unknown,
            JohariQuadrant::Open,
            TransitionTrigger::DreamConsolidation,
            5,
        );

        assert_eq!(stats.total_transitions, 3);
        assert_eq!(
            stats.count_for_path(JohariQuadrant::Hidden, JohariQuadrant::Open),
            2
        );
        assert_eq!(stats.count_for_trigger(TransitionTrigger::ExplicitShare), 2);
        assert_eq!(stats.count_for_embedder(0), 2);
        assert_eq!(stats.count_for_embedder(5), 1);

        println!("[VERIFIED] test_transition_stats_record: Recording transitions works correctly");
    }

    #[test]
    fn test_transition_stats_finalize() {
        let mut stats = TransitionStats::new();

        // Record multiple transitions
        for _ in 0..5 {
            stats.record(
                JohariQuadrant::Hidden,
                JohariQuadrant::Open,
                TransitionTrigger::ExplicitShare,
                0,
            );
        }
        for _ in 0..3 {
            stats.record(
                JohariQuadrant::Unknown,
                JohariQuadrant::Open,
                TransitionTrigger::DreamConsolidation,
                1,
            );
        }

        stats.record_affected_memory();
        stats.record_affected_memory();
        stats.finalize();

        assert_eq!(stats.total_transitions, 8);
        assert_eq!(stats.memories_affected, 2);
        assert!((stats.avg_transitions_per_memory - 4.0).abs() < f32::EPSILON);
        assert!(!stats.top_paths.is_empty());

        // Most common path should be Hidden→Open (5 occurrences)
        let most_common = stats.most_common_path().unwrap();
        assert_eq!(most_common.from, JohariQuadrant::Hidden);
        assert_eq!(most_common.to, JohariQuadrant::Open);

        println!("[VERIFIED] test_transition_stats_finalize: Finalization computes derived fields correctly");
    }

    #[test]
    fn test_transition_stats_merge() {
        let mut stats1 = TransitionStats::new();
        stats1.record(
            JohariQuadrant::Hidden,
            JohariQuadrant::Open,
            TransitionTrigger::ExplicitShare,
            0,
        );
        stats1.record_affected_memory();

        let mut stats2 = TransitionStats::new();
        stats2.record(
            JohariQuadrant::Hidden,
            JohariQuadrant::Open,
            TransitionTrigger::ExplicitShare,
            0,
        );
        stats2.record(
            JohariQuadrant::Unknown,
            JohariQuadrant::Blind,
            TransitionTrigger::ExternalObservation,
            5,
        );
        stats2.record_affected_memory();

        stats1.merge(&stats2);

        assert_eq!(stats1.total_transitions, 3);
        assert_eq!(stats1.memories_affected, 2);
        assert_eq!(
            stats1.count_for_path(JohariQuadrant::Hidden, JohariQuadrant::Open),
            2
        );

        println!("[VERIFIED] test_transition_stats_merge: Merging stats works correctly");
    }

    #[test]
    fn test_most_active_embedder() {
        let mut stats = TransitionStats::new();

        // E5 (causal) gets the most transitions
        for _ in 0..5 {
            stats.record(
                JohariQuadrant::Unknown,
                JohariQuadrant::Blind,
                TransitionTrigger::ExternalObservation,
                4, // E5
            );
        }
        for _ in 0..2 {
            stats.record(
                JohariQuadrant::Hidden,
                JohariQuadrant::Open,
                TransitionTrigger::ExplicitShare,
                0, // E1
            );
        }

        assert_eq!(stats.most_active_embedder(), 4);

        println!("[VERIFIED] test_most_active_embedder: Identifies most active embedder correctly");
    }
}
