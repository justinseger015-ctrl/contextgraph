//! Configuration for gap detection.

use serde::{Deserialize, Serialize};

/// Configuration for gap detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GapDetectionConfig {
    /// Minimum coverage score threshold (0.0-1.0)
    pub coverage_threshold: f32,
    /// Minimum activity score for a goal to be considered active
    pub activity_threshold: f32,
    /// Minimum number of goals per domain
    pub min_goals_per_domain: usize,
    /// Days without activity to flag as temporal gap
    pub inactivity_days: u32,
    /// Whether to detect missing links between related goals
    pub detect_missing_links: bool,
    /// Similarity threshold for detecting potentially linked goals
    pub link_similarity_threshold: f32,
}

impl Default for GapDetectionConfig {
    fn default() -> Self {
        Self {
            coverage_threshold: 0.4,
            activity_threshold: 0.2,
            min_goals_per_domain: 1,
            inactivity_days: 14,
            detect_missing_links: true,
            link_similarity_threshold: 0.7,
        }
    }
}
