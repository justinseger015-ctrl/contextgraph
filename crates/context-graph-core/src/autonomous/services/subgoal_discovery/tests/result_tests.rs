//! Tests for DiscoveryResult.

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::evolution::{GoalLevel, SubGoalCandidate};

use super::super::result::DiscoveryResult;

#[test]
fn test_discovery_result_empty() {
    let result = DiscoveryResult::new(vec![], 0);

    assert!(!result.has_candidates());
    assert_eq!(result.cluster_count, 0);
    assert!((result.avg_confidence - 0.0).abs() < f32::EPSILON);

    println!("[PASS] test_discovery_result_empty");
}

#[test]
fn test_discovery_result_with_candidates() {
    let candidates = vec![
        SubGoalCandidate {
            suggested_description: "Test 1".into(),
            level: GoalLevel::Tactical,
            parent_id: GoalId::new(),
            cluster_size: 10,
            centroid_alignment: 0.8,
            confidence: 0.7,
            supporting_memories: vec![],
        },
        SubGoalCandidate {
            suggested_description: "Test 2".into(),
            level: GoalLevel::Operational,
            parent_id: GoalId::new(),
            cluster_size: 15,
            centroid_alignment: 0.6,
            confidence: 0.9,
            supporting_memories: vec![],
        },
    ];

    let result = DiscoveryResult::new(candidates, 5);

    assert!(result.has_candidates());
    assert_eq!(result.cluster_count, 5);
    assert_eq!(result.candidates.len(), 2);
    // Average confidence: (0.7 + 0.9) / 2 = 0.8
    assert!((result.avg_confidence - 0.8).abs() < f32::EPSILON);

    println!("[PASS] test_discovery_result_with_candidates");
}

#[test]
fn test_discovery_result_promotable_candidates() {
    let candidates = vec![
        SubGoalCandidate {
            suggested_description: "High".into(),
            level: GoalLevel::Strategic,
            parent_id: GoalId::new(),
            cluster_size: 20,
            centroid_alignment: 0.9,
            confidence: 0.85,
            supporting_memories: vec![],
        },
        SubGoalCandidate {
            suggested_description: "Low".into(),
            level: GoalLevel::Operational,
            parent_id: GoalId::new(),
            cluster_size: 10,
            centroid_alignment: 0.5,
            confidence: 0.55,
            supporting_memories: vec![],
        },
    ];

    let result = DiscoveryResult::new(candidates, 2);
    let promotable = result.promotable_candidates(0.7);

    assert_eq!(promotable.len(), 1);
    assert_eq!(promotable[0].suggested_description, "High");

    println!("[PASS] test_discovery_result_promotable_candidates");
}
