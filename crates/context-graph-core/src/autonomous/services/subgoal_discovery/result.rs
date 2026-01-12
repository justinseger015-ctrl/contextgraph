//! Result types for sub-goal discovery.

use crate::autonomous::evolution::SubGoalCandidate;

/// Result of sub-goal discovery process
#[derive(Clone, Debug)]
pub struct DiscoveryResult {
    /// Discovered sub-goal candidates
    pub candidates: Vec<SubGoalCandidate>,
    /// Number of clusters analyzed
    pub cluster_count: usize,
    /// Average confidence across all candidates
    pub avg_confidence: f32,
    /// Number of clusters that passed minimum size threshold
    pub viable_clusters: usize,
    /// Number of candidates filtered out
    pub filtered_count: usize,
}

impl DiscoveryResult {
    /// Create a new discovery result
    pub(crate) fn new(candidates: Vec<SubGoalCandidate>, cluster_count: usize) -> Self {
        let avg_confidence = if candidates.is_empty() {
            0.0
        } else {
            candidates.iter().map(|c| c.confidence).sum::<f32>() / candidates.len() as f32
        };

        Self {
            candidates,
            cluster_count,
            avg_confidence,
            viable_clusters: 0,
            filtered_count: 0,
        }
    }

    /// Check if any candidates were discovered
    pub fn has_candidates(&self) -> bool {
        !self.candidates.is_empty()
    }

    /// Get candidates that should be promoted
    pub fn promotable_candidates(&self, threshold: f32) -> Vec<&SubGoalCandidate> {
        self.candidates
            .iter()
            .filter(|c| c.confidence >= threshold)
            .collect()
    }
}
