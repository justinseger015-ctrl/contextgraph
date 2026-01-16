//! Integration tests for sub-goal discovery.

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::evolution::GoalLevel;

use super::super::config::DiscoveryConfig;
use super::super::service::SubGoalDiscovery;
use super::make_labeled_cluster;

#[test]
fn test_full_discovery_workflow() {
    let config = DiscoveryConfig {
        min_cluster_size: 5,
        min_coherence: 0.5,
        min_confidence: 0.4,
        emergence_threshold: 0.6,
        max_candidates: 10,
    };
    let discovery = SubGoalDiscovery::with_config(config);

    let clusters = vec![
        make_labeled_cluster(20, 0.9, 0.8, "Primary Focus Area"),
        make_labeled_cluster(15, 0.8, 0.7, "Secondary Focus"),
        make_labeled_cluster(3, 0.9, 0.9, "Too Small"), // Filtered
        make_labeled_cluster(10, 0.3, 0.8, "Low Coherence"), // Filtered
        make_labeled_cluster(8, 0.7, 0.6, "Moderate Focus"),
    ];

    let result = discovery.discover_from_clusters(&clusters);

    assert_eq!(result.cluster_count, 5);
    assert_eq!(result.viable_clusters, 3);
    assert!(result.has_candidates());

    // Check candidates are ranked by confidence
    for i in 0..result.candidates.len() - 1 {
        assert!(result.candidates[i].confidence >= result.candidates[i + 1].confidence);
    }

    // Check promotable candidates
    let promotable = result.promotable_candidates(0.6);
    for c in &promotable {
        assert!(discovery.should_promote(c));
    }

    // Test parent assignment
    let goals = vec![GoalId::new(), GoalId::new(), GoalId::new()];
    for candidate in &result.candidates {
        let parent = discovery.find_parent_goal(candidate, &goals);
        if candidate.level != GoalLevel::Strategic {
            assert!(parent.is_some());
        }
    }

    println!("[PASS] test_full_discovery_workflow");
}
