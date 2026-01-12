//! Configuration for sub-goal discovery.

use crate::autonomous::evolution::GoalEvolutionConfig;

/// Configuration for sub-goal discovery
#[derive(Clone, Debug)]
pub struct DiscoveryConfig {
    /// Minimum cluster size to consider for sub-goal extraction
    pub min_cluster_size: usize,
    /// Minimum coherence for a cluster to be viable
    pub min_coherence: f32,
    /// Threshold for emergence (confidence above which to promote)
    pub emergence_threshold: f32,
    /// Maximum candidates to return
    pub max_candidates: usize,
    /// Minimum confidence for a candidate to be viable
    pub min_confidence: f32,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 10,
            min_coherence: 0.6,
            emergence_threshold: 0.7,
            max_candidates: 20,
            min_confidence: 0.5,
        }
    }
}

impl From<&GoalEvolutionConfig> for DiscoveryConfig {
    fn from(config: &GoalEvolutionConfig) -> Self {
        Self {
            min_cluster_size: config.min_cluster_size,
            ..Default::default()
        }
    }
}
