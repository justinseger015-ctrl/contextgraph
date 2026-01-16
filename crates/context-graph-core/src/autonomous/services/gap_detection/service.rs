//! Gap detection service implementation.

use std::collections::{HashMap, HashSet};

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::evolution::GoalLevel;

use super::config::GapDetectionConfig;
use super::metrics::GoalWithMetrics;
use super::report::GapReport;
use super::types::GapType;

/// Service for detecting gaps in goal coverage
#[derive(Clone, Debug)]
pub struct GapDetectionService {
    pub(crate) config: GapDetectionConfig,
}

impl GapDetectionService {
    /// Create a new gap detection service with default configuration
    pub fn new() -> Self {
        Self {
            config: GapDetectionConfig::default(),
        }
    }

    /// Create a new gap detection service with custom configuration
    pub fn with_config(config: GapDetectionConfig) -> Self {
        Self { config }
    }

    /// Analyze coverage and detect all gaps
    pub fn analyze_coverage(&self, goals: &[GoalWithMetrics]) -> GapReport {
        if goals.is_empty() {
            return GapReport {
                gaps: vec![],
                coverage_score: 0.0,
                recommendations: vec![
                    "No goals to analyze. Consider bootstrapping initial goals.".into()
                ],
                goals_analyzed: 0,
                domains_detected: 0,
            };
        }

        let mut all_gaps = Vec::new();

        // Detect domain gaps
        let domain_gaps = self.detect_domain_gaps(goals);
        all_gaps.extend(domain_gaps);

        // Detect weak coverage
        let weak_coverage_gaps = self.detect_weak_coverage(goals);
        all_gaps.extend(weak_coverage_gaps);

        // Detect missing links
        if self.config.detect_missing_links {
            let link_gaps = self.detect_missing_links(goals);
            all_gaps.extend(link_gaps);
        }

        // Compute overall coverage
        let coverage_score = self.compute_coverage_score(goals);

        // Collect all domains
        let domains: HashSet<_> = goals.iter().flat_map(|g| g.domains.iter()).collect();

        // Generate recommendations
        let recommendations = self.generate_recommendations(&all_gaps);

        GapReport {
            gaps: all_gaps,
            coverage_score,
            recommendations,
            goals_analyzed: goals.len(),
            domains_detected: domains.len(),
        }
    }

    /// Detect domains with no or insufficient goal coverage
    pub fn detect_domain_gaps(&self, goals: &[GoalWithMetrics]) -> Vec<GapType> {
        let mut gaps = Vec::new();

        // Build domain coverage map
        let mut domain_goals: HashMap<&String, Vec<&GoalWithMetrics>> = HashMap::new();
        for goal in goals {
            for domain in &goal.domains {
                domain_goals.entry(domain).or_default().push(goal);
            }
        }

        // Check each domain for sufficient coverage
        for (domain, domain_goal_list) in &domain_goals {
            let active_goals: Vec<_> = domain_goal_list
                .iter()
                .filter(|g| g.metrics.activity_score() >= self.config.activity_threshold)
                .collect();

            if active_goals.len() < self.config.min_goals_per_domain {
                gaps.push(GapType::UncoveredDomain {
                    domain: (*domain).clone(),
                });
            }
        }

        gaps
    }

    /// Detect goals with weak coverage (low activity/alignment)
    pub fn detect_weak_coverage(&self, goals: &[GoalWithMetrics]) -> Vec<GapType> {
        let mut gaps = Vec::new();

        for goal in goals {
            if goal.has_weak_coverage(self.config.coverage_threshold) {
                gaps.push(GapType::WeakCoverage {
                    goal_id: goal.goal_id.clone(),
                    coverage: goal.coverage_score(),
                });
            }
        }

        gaps
    }

    /// Detect missing links between related goals
    pub fn detect_missing_links(&self, goals: &[GoalWithMetrics]) -> Vec<GapType> {
        let mut gaps = Vec::new();

        // Build parent-child relationship set
        let mut linked_pairs: HashSet<(GoalId, GoalId)> = HashSet::new();
        for goal in goals {
            if let Some(ref parent_id) = goal.parent_id {
                linked_pairs.insert((parent_id.clone(), goal.goal_id.clone()));
                linked_pairs.insert((goal.goal_id.clone(), parent_id.clone()));
            }
            for child_id in &goal.child_ids {
                linked_pairs.insert((goal.goal_id.clone(), child_id.clone()));
                linked_pairs.insert((child_id.clone(), goal.goal_id.clone()));
            }
        }

        // Check for goals that share domains but are not linked
        for i in 0..goals.len() {
            for j in (i + 1)..goals.len() {
                let goal_a = &goals[i];
                let goal_b = &goals[j];

                // Check if they share domains
                let shared_domains: Vec<_> = goal_a
                    .domains
                    .iter()
                    .filter(|d| goal_b.domains.contains(d))
                    .collect();

                if !shared_domains.is_empty() {
                    // They share domains - check if linked
                    let pair = (goal_a.goal_id.clone(), goal_b.goal_id.clone());
                    if !linked_pairs.contains(&pair) {
                        // Not linked but share domains - potential missing link
                        // Only flag if both are active
                        if goal_a.metrics.is_active() && goal_b.metrics.is_active() {
                            gaps.push(GapType::MissingLink {
                                from: goal_a.goal_id.clone(),
                                to: goal_b.goal_id.clone(),
                            });
                        }
                    }
                }
            }
        }

        gaps
    }

    /// Compute overall coverage score across all goals
    pub fn compute_coverage_score(&self, goals: &[GoalWithMetrics]) -> f32 {
        if goals.is_empty() {
            return 0.0;
        }

        let total: f32 = goals.iter().map(|g| g.coverage_score()).sum();
        let average = total / goals.len() as f32;

        // Penalize for inactive goals
        let active_ratio =
            goals.iter().filter(|g| g.metrics.is_active()).count() as f32 / goals.len() as f32;

        // TASK-P0-001: Weight by level (Strategic goals matter most now)
        let level_weighted: f32 = goals
            .iter()
            .map(|g| {
                let level_weight = match g.level {
                    GoalLevel::Strategic => 2.0,
                    GoalLevel::Tactical => 1.0,
                    GoalLevel::Operational => 0.75,
                };
                g.coverage_score() * level_weight
            })
            .sum::<f32>()
            / goals
                .iter()
                .map(|g| match g.level {
                    GoalLevel::Strategic => 2.0,
                    GoalLevel::Tactical => 1.0,
                    GoalLevel::Operational => 0.75,
                })
                .sum::<f32>();

        // Combine metrics
        (0.4 * average + 0.3 * active_ratio + 0.3 * level_weighted).clamp(0.0, 1.0)
    }

    /// Generate recommendations based on detected gaps
    pub fn generate_recommendations(&self, gaps: &[GapType]) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Count gap types
        let mut uncovered_domains = Vec::new();
        let mut weak_coverage_count = 0;
        let mut missing_link_count = 0;
        let mut temporal_gap_count = 0;

        for gap in gaps {
            match gap {
                GapType::UncoveredDomain { domain } => uncovered_domains.push(domain.clone()),
                GapType::WeakCoverage { .. } => weak_coverage_count += 1,
                GapType::MissingLink { .. } => missing_link_count += 1,
                GapType::TemporalGap { .. } => temporal_gap_count += 1,
            }
        }

        // Generate recommendations based on gaps found
        if !uncovered_domains.is_empty() {
            if uncovered_domains.len() == 1 {
                recommendations.push(format!(
                    "Create a goal to cover the '{}' domain",
                    uncovered_domains[0]
                ));
            } else {
                recommendations.push(format!(
                    "Create goals to cover {} uncovered domains: {}",
                    uncovered_domains.len(),
                    uncovered_domains.join(", ")
                ));
            }
        }

        if weak_coverage_count > 0 {
            if weak_coverage_count == 1 {
                recommendations.push("Review and strengthen the goal with weak coverage".into());
            } else {
                recommendations.push(format!(
                    "Review and strengthen {} goals with weak coverage",
                    weak_coverage_count
                ));
            }
        }

        if missing_link_count > 0 {
            if missing_link_count == 1 {
                recommendations.push("Consider establishing a link between related goals".into());
            } else {
                recommendations.push(format!(
                    "Consider establishing {} links between related goals",
                    missing_link_count
                ));
            }
        }

        if temporal_gap_count > 0 {
            recommendations.push(format!(
                "Address {} period(s) of inactivity by scheduling regular goal reviews",
                temporal_gap_count
            ));
        }

        if recommendations.is_empty() && gaps.is_empty() {
            recommendations.push("Goal coverage is healthy. Continue monitoring.".into());
        }

        recommendations
    }
}

impl Default for GapDetectionService {
    fn default() -> Self {
        Self::new()
    }
}
