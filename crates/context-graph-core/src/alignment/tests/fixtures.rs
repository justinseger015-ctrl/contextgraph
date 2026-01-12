//! Test fixtures and helpers for alignment module tests.
//!
//! Provides real data types - no mocks.

use crate::alignment::*;
use crate::purpose::{GoalDiscoveryMetadata, GoalHierarchy, GoalLevel, GoalNode};
use crate::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, TeleologicalFingerprint, NUM_EMBEDDERS,
};

use chrono::Utc;
use uuid::Uuid;

/// Create a real TeleologicalFingerprint with specified alignment characteristics.
pub fn create_real_fingerprint(alignment_factor: f32) -> TeleologicalFingerprint {
    // Create REAL SemanticFingerprint (not mocked)
    let mut semantic = SemanticFingerprint::zeroed();

    // Populate E1 with a realistic pattern
    for i in 0..semantic.e1_semantic.len() {
        // Use deterministic pattern based on alignment factor
        semantic.e1_semantic[i] = ((i as f32 / 128.0).sin() * alignment_factor).clamp(-1.0, 1.0);
    }

    // Create REAL PurposeVector
    let alignments = [alignment_factor; NUM_EMBEDDERS];
    let purpose_vector = PurposeVector::new(alignments);

    // Create REAL JohariFingerprint
    let johari = JohariFingerprint::zeroed();

    TeleologicalFingerprint {
        id: Uuid::new_v4(),
        semantic,
        purpose_vector,
        johari,
        purpose_evolution: Vec::new(),
        theta_to_north_star: alignment_factor,
        content_hash: [0u8; 32],
        created_at: Utc::now(),
        last_updated: Utc::now(),
        access_count: 0,
    }
}

/// Helper to create bootstrap discovery metadata for tests.
pub fn test_discovery() -> GoalDiscoveryMetadata {
    GoalDiscoveryMetadata::bootstrap()
}

/// Helper to create a SemanticFingerprint with deterministic pattern.
pub fn create_test_fingerprint(seed: f32) -> SemanticFingerprint {
    let mut fp = SemanticFingerprint::zeroed();
    // Populate E1 with deterministic pattern based on seed
    for i in 0..fp.e1_semantic.len() {
        fp.e1_semantic[i] = ((i as f32 / 128.0 + seed).sin()).clamp(-1.0, 1.0);
    }
    fp
}

/// Create a real GoalHierarchy with all four levels.
pub fn create_real_hierarchy() -> GoalHierarchy {
    let mut hierarchy = GoalHierarchy::new();

    // North Star - primary goal
    let ns = GoalNode::autonomous_goal(
        "Revolutionize knowledge management through AI".into(),
        GoalLevel::NorthStar,
        create_test_fingerprint(0.0),
        test_discovery(),
    )
    .expect("FAIL: Could not create North Star goal");
    let ns_id = ns.id;
    hierarchy
        .add_goal(ns)
        .expect("FAIL: Could not add North Star goal");

    // Strategic goals
    let s1 = GoalNode::child_goal(
        "Build intelligent retrieval system".into(),
        GoalLevel::Strategic,
        ns_id,
        create_test_fingerprint(0.1),
        test_discovery(),
    )
    .expect("FAIL: Could not create Strategic goal 1");
    let s1_id = s1.id;
    hierarchy
        .add_goal(s1)
        .expect("FAIL: Could not add Strategic goal 1");

    let s2 = GoalNode::child_goal(
        "Enable semantic understanding".into(),
        GoalLevel::Strategic,
        ns_id,
        create_test_fingerprint(0.2),
        test_discovery(),
    )
    .expect("FAIL: Could not create Strategic goal 2");
    let s2_id = s2.id;
    hierarchy
        .add_goal(s2)
        .expect("FAIL: Could not add Strategic goal 2");

    // Tactical goals
    let t1 = GoalNode::child_goal(
        "Implement vector search".into(),
        GoalLevel::Tactical,
        s1_id,
        create_test_fingerprint(0.3),
        test_discovery(),
    )
    .expect("FAIL: Could not create Tactical goal 1");
    let t1_id = t1.id;
    hierarchy
        .add_goal(t1)
        .expect("FAIL: Could not add Tactical goal 1");

    let t2 = GoalNode::child_goal(
        "Build embedding pipeline".into(),
        GoalLevel::Tactical,
        s2_id,
        create_test_fingerprint(0.4),
        test_discovery(),
    )
    .expect("FAIL: Could not create Tactical goal 2");
    hierarchy
        .add_goal(t2)
        .expect("FAIL: Could not add Tactical goal 2");

    // Immediate goals
    let i1 = GoalNode::child_goal(
        "Optimize query latency".into(),
        GoalLevel::Immediate,
        t1_id,
        create_test_fingerprint(0.5),
        test_discovery(),
    )
    .expect("FAIL: Could not create Immediate goal 1");
    hierarchy
        .add_goal(i1)
        .expect("FAIL: Could not add Immediate goal 1");

    hierarchy
}

/// Create a hierarchy with intentional misalignment for testing.
pub fn create_misaligned_hierarchy() -> GoalHierarchy {
    let mut hierarchy = GoalHierarchy::new();

    // North Star
    let ns = GoalNode::autonomous_goal(
        "North Star Goal".into(),
        GoalLevel::NorthStar,
        create_test_fingerprint(0.0),
        test_discovery(),
    )
    .expect("FAIL: North Star");
    let ns_id = ns.id;
    hierarchy.add_goal(ns).expect("FAIL: North Star");

    // Strategic with different embedding (will diverge)
    // Use PI offset to create different embedding pattern
    let mut divergent_fp = SemanticFingerprint::zeroed();
    for i in 0..divergent_fp.e1_semantic.len() {
        divergent_fp.e1_semantic[i] = ((i as f32 / 128.0) + std::f32::consts::PI).sin();
    }

    let s1 = GoalNode::child_goal(
        "Divergent Strategic".into(),
        GoalLevel::Strategic,
        ns_id,
        divergent_fp,
        test_discovery(),
    )
    .expect("FAIL: Strategic");
    hierarchy.add_goal(s1).expect("FAIL: Strategic");

    hierarchy
}
