//! Sub-goal discovery service implementation.

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::evolution::{GoalEvolutionConfig, GoalLevel, SubGoalCandidate};

use super::cluster::MemoryCluster;
use super::config::DiscoveryConfig;
use super::result::DiscoveryResult;

/// Sub-goal discovery service
///
/// Discovers emergent sub-goals by analyzing memory clusters for patterns
/// that suggest coherent new goal areas.
#[derive(Clone, Debug)]
pub struct SubGoalDiscovery {
    config: DiscoveryConfig,
}

impl SubGoalDiscovery {
    /// Create a new discovery service with default configuration
    pub fn new() -> Self {
        Self {
            config: DiscoveryConfig::default(),
        }
    }

    /// Create a discovery service with custom configuration
    pub fn with_config(config: DiscoveryConfig) -> Self {
        Self { config }
    }

    /// Create from goal evolution config
    pub fn from_evolution_config(config: &GoalEvolutionConfig) -> Self {
        Self {
            config: DiscoveryConfig::from(config),
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &DiscoveryConfig {
        &self.config
    }

    /// Discover sub-goals from a set of memory clusters
    ///
    /// Analyzes each cluster and extracts candidate sub-goals that meet
    /// the minimum requirements.
    pub fn discover_from_clusters(&self, clusters: &[MemoryCluster]) -> DiscoveryResult {
        let cluster_count = clusters.len();

        if clusters.is_empty() {
            return DiscoveryResult::new(vec![], 0);
        }

        let mut candidates = Vec::new();
        let mut viable_clusters = 0;
        let mut filtered_count = 0;

        for cluster in clusters {
            // Skip clusters that don't meet minimum size
            if cluster.size() < self.config.min_cluster_size {
                filtered_count += 1;
                continue;
            }

            // Skip clusters with low coherence
            if cluster.coherence < self.config.min_coherence {
                filtered_count += 1;
                continue;
            }

            viable_clusters += 1;

            if let Some(candidate) = self.extract_candidate(cluster) {
                if candidate.confidence >= self.config.min_confidence {
                    candidates.push(candidate);
                } else {
                    filtered_count += 1;
                }
            }
        }

        // Rank candidates by confidence
        self.rank_candidates(&mut candidates);

        // Limit to max candidates
        if candidates.len() > self.config.max_candidates {
            filtered_count += candidates.len() - self.config.max_candidates;
            candidates.truncate(self.config.max_candidates);
        }

        let mut result = DiscoveryResult::new(candidates, cluster_count);
        result.viable_clusters = viable_clusters;
        result.filtered_count = filtered_count;
        result
    }

    /// Extract a sub-goal candidate from a memory cluster
    ///
    /// Returns None if the cluster doesn't meet requirements or lacks
    /// sufficient signal for goal extraction.
    pub fn extract_candidate(&self, cluster: &MemoryCluster) -> Option<SubGoalCandidate> {
        // Validate cluster
        if cluster.is_empty() {
            return None;
        }

        if cluster.size() < self.config.min_cluster_size {
            return None;
        }

        if cluster.coherence < self.config.min_coherence {
            return None;
        }

        let confidence = self.compute_confidence(cluster);
        let level = self.determine_level(confidence, cluster.size());

        // Generate description from label or placeholder
        let description = cluster
            .label
            .clone()
            .unwrap_or_else(|| format!("Emergent goal from {} memories", cluster.size()));

        Some(SubGoalCandidate {
            suggested_description: description,
            level,
            parent_id: GoalId::new(), // Placeholder, will be assigned by find_parent_goal
            cluster_size: cluster.size(),
            centroid_alignment: cluster.avg_alignment,
            confidence,
            supporting_memories: cluster.members.clone(),
        })
    }

    /// Compute confidence score for a cluster
    ///
    /// Confidence is based on:
    /// - Cluster coherence (40%)
    /// - Cluster size (30%)
    /// - Average alignment (30%)
    pub fn compute_confidence(&self, cluster: &MemoryCluster) -> f32 {
        if cluster.is_empty() {
            return 0.0;
        }

        // Coherence contribution (40%)
        let coherence_score = cluster.coherence * 0.4;

        // Size contribution (30%) - logarithmic scaling, max at ~100 members
        let size_score = (cluster.size() as f32).ln().min(4.6) / 4.6 * 0.3;

        // Alignment contribution (30%)
        let alignment_score = cluster.avg_alignment * 0.3;

        (coherence_score + size_score + alignment_score).clamp(0.0, 1.0)
    }

    /// Find the best parent goal for a candidate
    ///
    /// Uses the candidate's centroid alignment and level to find
    /// the most appropriate parent in the existing hierarchy.
    pub fn find_parent_goal(
        &self,
        candidate: &SubGoalCandidate,
        existing_goals: &[GoalId],
    ) -> Option<GoalId> {
        if existing_goals.is_empty() {
            return None;
        }

        // For now, return the first goal as a simple heuristic
        // In production, this would compute similarity between
        // the candidate's centroid and each goal's embedding
        match candidate.level {
            GoalLevel::Strategic | GoalLevel::Tactical => Some(existing_goals[0].clone()),
            GoalLevel::Operational => {
                // Prefer non-first goal if available (assuming hierarchy)
                if existing_goals.len() > 1 {
                    Some(existing_goals[1].clone())
                } else {
                    Some(existing_goals[0].clone())
                }
            }
            GoalLevel::NorthStar => None, // NorthStar has no parent
        }
    }

    /// Determine the appropriate goal level based on confidence and evidence
    ///
    /// Higher confidence and more evidence suggest higher-level goals.
    pub fn determine_level(&self, confidence: f32, evidence_count: usize) -> GoalLevel {
        // Strong signal with lots of evidence -> Strategic
        if confidence >= 0.85 && evidence_count >= 50 {
            return GoalLevel::Strategic;
        }

        // Good signal with moderate evidence -> Tactical
        if confidence >= 0.7 && evidence_count >= 20 {
            return GoalLevel::Tactical;
        }

        // Default to Operational for weaker signals
        GoalLevel::Operational
    }

    /// Check if a candidate should be promoted to an actual goal
    ///
    /// Candidates should be promoted if they exceed the emergence threshold
    /// and have sufficient supporting evidence.
    pub fn should_promote(&self, candidate: &SubGoalCandidate) -> bool {
        // Must meet emergence threshold
        if candidate.confidence < self.config.emergence_threshold {
            return false;
        }

        // Must have minimum cluster size
        if candidate.cluster_size < self.config.min_cluster_size {
            return false;
        }

        // Must have reasonable alignment
        if candidate.centroid_alignment < 0.3 {
            return false;
        }

        true
    }

    /// Rank candidates by priority for promotion
    ///
    /// Ranking considers confidence, cluster size, and alignment.
    pub fn rank_candidates(&self, candidates: &mut [SubGoalCandidate]) {
        candidates.sort_by(|a, b| {
            // Primary sort by confidence (descending)
            let conf_cmp = b
                .confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal);

            if conf_cmp != std::cmp::Ordering::Equal {
                return conf_cmp;
            }

            // Secondary sort by cluster size (descending)
            let size_cmp = b.cluster_size.cmp(&a.cluster_size);

            if size_cmp != std::cmp::Ordering::Equal {
                return size_cmp;
            }

            // Tertiary sort by alignment (descending)
            b.centroid_alignment
                .partial_cmp(&a.centroid_alignment)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

impl Default for SubGoalDiscovery {
    fn default() -> Self {
        Self::new()
    }
}
