//! Tests for goal hierarchy types.

#[cfg(test)]
mod tests {
    use crate::purpose::goals::{
        DiscoveryMethod, GoalDiscoveryMetadata, GoalHierarchy, GoalHierarchyError, GoalLevel,
        GoalNode, GoalNodeError,
    };
    use crate::types::fingerprint::{SemanticFingerprint, ValidationError};
    use chrono::Utc;
    use uuid::Uuid;

    // Helper function to create a valid zeroed fingerprint for testing
    fn test_fingerprint() -> SemanticFingerprint {
        SemanticFingerprint::zeroed()
    }

    // Helper function to create bootstrap discovery metadata
    fn test_discovery() -> GoalDiscoveryMetadata {
        GoalDiscoveryMetadata::bootstrap()
    }

    // Helper function to create clustering discovery metadata
    fn clustering_discovery(
        confidence: f32,
        cluster_size: usize,
        coherence: f32,
    ) -> Result<GoalDiscoveryMetadata, GoalNodeError> {
        GoalDiscoveryMetadata::new(
            DiscoveryMethod::Clustering,
            confidence,
            cluster_size,
            coherence,
        )
    }

    #[test]
    fn test_goal_level_propagation_weights() {
        assert_eq!(GoalLevel::NorthStar.propagation_weight(), 1.0);
        assert_eq!(GoalLevel::Strategic.propagation_weight(), 0.7);
        assert_eq!(GoalLevel::Tactical.propagation_weight(), 0.4);
        assert_eq!(GoalLevel::Immediate.propagation_weight(), 0.2);
        println!("[VERIFIED] GoalLevel propagation weights match constitution.yaml");
    }

    #[test]
    fn test_goal_level_depth() {
        assert_eq!(GoalLevel::NorthStar.depth(), 0);
        assert_eq!(GoalLevel::Strategic.depth(), 1);
        assert_eq!(GoalLevel::Tactical.depth(), 2);
        assert_eq!(GoalLevel::Immediate.depth(), 3);
        println!("[VERIFIED] GoalLevel depth values are correct");
    }

    #[test]
    fn test_discovery_metadata_valid() {
        let discovery = clustering_discovery(0.85, 42, 0.78).unwrap();
        assert_eq!(discovery.method, DiscoveryMethod::Clustering);
        assert_eq!(discovery.confidence, 0.85);
        assert_eq!(discovery.cluster_size, 42);
        assert_eq!(discovery.coherence, 0.78);
        println!("[VERIFIED] GoalDiscoveryMetadata::new creates valid metadata");
    }

    #[test]
    fn test_discovery_metadata_bootstrap() {
        let discovery = GoalDiscoveryMetadata::bootstrap();
        assert_eq!(discovery.method, DiscoveryMethod::Bootstrap);
        assert_eq!(discovery.confidence, 0.0);
        assert_eq!(discovery.cluster_size, 0);
        assert_eq!(discovery.coherence, 0.0);
        println!("[VERIFIED] GoalDiscoveryMetadata::bootstrap creates correct defaults");
    }

    #[test]
    fn test_discovery_metadata_invalid_confidence() {
        let result = clustering_discovery(1.5, 10, 0.8);
        assert!(matches!(result, Err(GoalNodeError::InvalidConfidence(1.5))));

        let result = clustering_discovery(-0.1, 10, 0.8);
        assert!(matches!(result, Err(GoalNodeError::InvalidConfidence(_))));
        println!("[VERIFIED] GoalDiscoveryMetadata rejects invalid confidence");
    }

    #[test]
    fn test_discovery_metadata_invalid_coherence() {
        let result = clustering_discovery(0.8, 10, 1.5);
        assert!(matches!(result, Err(GoalNodeError::InvalidCoherence(1.5))));

        let result = clustering_discovery(0.8, 10, -0.1);
        assert!(matches!(result, Err(GoalNodeError::InvalidCoherence(_))));
        println!("[VERIFIED] GoalDiscoveryMetadata rejects invalid coherence");
    }

    #[test]
    fn test_discovery_metadata_empty_cluster() {
        let result = clustering_discovery(0.8, 0, 0.7);
        assert!(matches!(result, Err(GoalNodeError::EmptyCluster)));
        println!("[VERIFIED] GoalDiscoveryMetadata rejects empty cluster for non-Bootstrap");
    }

    #[test]
    fn test_goal_node_autonomous_creation() {
        let fp = test_fingerprint();
        let discovery = test_discovery();

        let goal = GoalNode::autonomous_goal(
            "Test North Star".into(),
            GoalLevel::NorthStar,
            fp,
            discovery,
        )
        .expect("Should create goal");

        assert_eq!(goal.level, GoalLevel::NorthStar);
        assert!(goal.parent_id.is_none());
        assert!(goal.child_ids.is_empty());
        assert!(goal.is_north_star());
        println!("[VERIFIED] GoalNode::autonomous_goal creates correct structure");
    }

    #[test]
    fn test_goal_node_child_creation() {
        let fp = test_fingerprint();
        let discovery = test_discovery();
        let parent_id = Uuid::new_v4();

        let child = GoalNode::child_goal(
            "Test Strategic Goal".into(),
            GoalLevel::Strategic,
            parent_id,
            fp,
            discovery,
        )
        .expect("Should create child goal");

        assert_eq!(child.level, GoalLevel::Strategic);
        assert_eq!(child.parent_id, Some(parent_id));
        assert!(!child.is_north_star());
        println!("[VERIFIED] GoalNode::child_goal creates correct structure");
    }

    #[test]
    #[should_panic(expected = "Child goal cannot be NorthStar")]
    fn test_child_goal_cannot_be_north_star() {
        let fp = test_fingerprint();
        let discovery = test_discovery();
        let parent_id = Uuid::new_v4();

        let _ = GoalNode::child_goal(
            "Bad goal".into(),
            GoalLevel::NorthStar, // Should panic
            parent_id,
            fp,
            discovery,
        );
    }

    #[test]
    fn test_goal_node_invalid_fingerprint() {
        let mut fp = test_fingerprint();
        fp.e1_semantic = vec![]; // Invalid - empty

        let discovery = test_discovery();
        let result = GoalNode::autonomous_goal("Test".into(), GoalLevel::NorthStar, fp, discovery);

        assert!(result.is_err());
        assert!(matches!(result, Err(GoalNodeError::InvalidArray(_))));
        println!("[VERIFIED] GoalNode rejects invalid teleological array");
    }

    #[test]
    fn test_goal_node_array_access() {
        let fp = test_fingerprint();
        let discovery = test_discovery();

        let goal =
            GoalNode::autonomous_goal("Test".into(), GoalLevel::NorthStar, fp, discovery).unwrap();

        let array = goal.array();
        assert_eq!(array.e1_semantic.len(), 1024);
        println!("[VERIFIED] GoalNode::array() provides access to teleological array");
    }

    #[test]
    fn test_goal_node_child_management() {
        let fp = test_fingerprint();
        let discovery = test_discovery();

        let mut goal =
            GoalNode::autonomous_goal("Test".into(), GoalLevel::NorthStar, fp, discovery).unwrap();

        let child_id = Uuid::new_v4();
        goal.add_child(child_id);
        assert!(goal.child_ids.contains(&child_id));
        assert_eq!(goal.child_ids.len(), 1);

        // Adding same child again should not duplicate
        goal.add_child(child_id);
        assert_eq!(goal.child_ids.len(), 1);

        goal.remove_child(child_id);
        assert!(!goal.child_ids.contains(&child_id));
        println!("[VERIFIED] GoalNode child management works correctly");
    }

    #[test]
    fn test_goal_hierarchy_single_north_star() {
        let mut hierarchy = GoalHierarchy::new();

        let ns1 = GoalNode::autonomous_goal(
            "NS1".into(),
            GoalLevel::NorthStar,
            test_fingerprint(),
            test_discovery(),
        )
        .unwrap();

        let ns2 = GoalNode::autonomous_goal(
            "NS2".into(),
            GoalLevel::NorthStar,
            test_fingerprint(),
            test_discovery(),
        )
        .unwrap();

        assert!(hierarchy.add_goal(ns1).is_ok());
        let result = hierarchy.add_goal(ns2);
        assert!(matches!(
            result,
            Err(GoalHierarchyError::MultipleNorthStars)
        ));
        println!("[VERIFIED] GoalHierarchy enforces single North Star");
    }

    #[test]
    fn test_goal_hierarchy_parent_validation() {
        let mut hierarchy = GoalHierarchy::new();

        let fake_parent = Uuid::new_v4();
        let child = GoalNode::child_goal(
            "Orphan".into(),
            GoalLevel::Strategic,
            fake_parent,
            test_fingerprint(),
            test_discovery(),
        )
        .unwrap();

        let result = hierarchy.add_goal(child);
        assert!(matches!(result, Err(GoalHierarchyError::ParentNotFound(_))));
        println!("[VERIFIED] GoalHierarchy validates parent existence");
    }

    #[test]
    fn test_goal_hierarchy_full_tree() {
        let mut hierarchy = GoalHierarchy::new();

        // Add North Star
        let ns = GoalNode::autonomous_goal(
            "Master ML".into(),
            GoalLevel::NorthStar,
            test_fingerprint(),
            clustering_discovery(0.9, 100, 0.85).unwrap(),
        )
        .unwrap();
        let ns_id = ns.id;
        hierarchy.add_goal(ns).unwrap();

        // Add Strategic child
        let strategic = GoalNode::child_goal(
            "Learn PyTorch".into(),
            GoalLevel::Strategic,
            ns_id,
            test_fingerprint(),
            clustering_discovery(0.8, 50, 0.75).unwrap(),
        )
        .unwrap();
        let strategic_id = strategic.id;
        hierarchy.add_goal(strategic).unwrap();

        // Add Tactical child
        let tactical = GoalNode::child_goal(
            "Complete tutorial".into(),
            GoalLevel::Tactical,
            strategic_id,
            test_fingerprint(),
            clustering_discovery(0.7, 20, 0.65).unwrap(),
        )
        .unwrap();
        let tactical_id = tactical.id;
        hierarchy.add_goal(tactical).unwrap();

        assert_eq!(hierarchy.len(), 3);
        assert!(!hierarchy.is_empty());
        assert!(hierarchy.has_north_star());
        assert!(hierarchy.north_star().is_some());
        assert_eq!(hierarchy.at_level(GoalLevel::Strategic).len(), 1);
        assert_eq!(hierarchy.at_level(GoalLevel::Tactical).len(), 1);
        assert_eq!(hierarchy.children(&ns_id).len(), 1);
        assert!(hierarchy.validate().is_ok());

        // Verify path to north star
        let path = hierarchy.path_to_north_star(&tactical_id);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], tactical_id);
        assert_eq!(path[1], strategic_id);
        assert_eq!(path[2], ns_id);

        println!("[VERIFIED] GoalHierarchy full tree structure works correctly");
    }

    #[test]
    fn test_goal_hierarchy_validate_no_north_star() {
        let mut hierarchy = GoalHierarchy::new();

        // Manually insert a node without North Star (bypass add_goal validation)
        let goal = GoalNode {
            id: Uuid::new_v4(),
            description: "Orphan".into(),
            level: GoalLevel::Strategic,
            teleological_array: test_fingerprint(),
            parent_id: None,
            child_ids: vec![],
            discovery: test_discovery(),
            created_at: Utc::now(),
        };
        hierarchy.nodes.insert(goal.id, goal);

        let result = hierarchy.validate();
        assert!(matches!(result, Err(GoalHierarchyError::NoNorthStar)));
        println!("[VERIFIED] validate detects missing North Star");
    }

    #[test]
    fn test_goal_hierarchy_iter() {
        let mut hierarchy = GoalHierarchy::new();

        let ns = GoalNode::autonomous_goal(
            "NS".into(),
            GoalLevel::NorthStar,
            test_fingerprint(),
            test_discovery(),
        )
        .unwrap();
        let ns_id = ns.id;
        hierarchy.add_goal(ns).unwrap();

        let child = GoalNode::child_goal(
            "C1".into(),
            GoalLevel::Strategic,
            ns_id,
            test_fingerprint(),
            test_discovery(),
        )
        .unwrap();
        hierarchy.add_goal(child).unwrap();

        let count = hierarchy.iter().count();
        assert_eq!(count, 2);
        println!("[VERIFIED] GoalHierarchy iter works correctly");
    }

    #[test]
    fn test_goal_serialization_roundtrip() {
        let fp = test_fingerprint();
        let discovery = clustering_discovery(0.85, 42, 0.78).unwrap();

        let goal =
            GoalNode::autonomous_goal("Test goal".into(), GoalLevel::Strategic, fp, discovery)
                .unwrap();

        // Serialize
        let json = serde_json::to_string(&goal).expect("Serialize");

        // Deserialize
        let restored: GoalNode = serde_json::from_str(&json).expect("Deserialize");

        // Verify
        assert_eq!(goal.id, restored.id);
        assert_eq!(goal.level, restored.level);
        assert_eq!(goal.description, restored.description);
        assert_eq!(
            goal.teleological_array.e1_semantic.len(),
            restored.teleological_array.e1_semantic.len()
        );
        println!("[VERIFIED] GoalNode survives JSON serialization roundtrip");
    }

    #[test]
    fn test_discovery_method_serialization() {
        let methods = vec![
            DiscoveryMethod::Clustering,
            DiscoveryMethod::PatternRecognition,
            DiscoveryMethod::Decomposition,
            DiscoveryMethod::Bootstrap,
        ];

        for method in methods {
            let json = serde_json::to_string(&method).expect("Serialize");
            let restored: DiscoveryMethod = serde_json::from_str(&json).expect("Deserialize");
            assert_eq!(method, restored);
        }
        println!("[VERIFIED] DiscoveryMethod serialization works correctly");
    }

    // Edge Case Tests from Task Spec

    #[test]
    fn test_edge_case_incomplete_fingerprint_rejected() {
        let mut fp = test_fingerprint();
        fp.e1_semantic = vec![]; // Invalid - empty

        let discovery = test_discovery();
        let result = GoalNode::autonomous_goal("Test".into(), GoalLevel::NorthStar, fp, discovery);

        assert!(result.is_err());
        match result {
            Err(GoalNodeError::InvalidArray(ValidationError::DimensionMismatch { .. })) => {
                println!("[EDGE CASE 1 PASSED] Incomplete fingerprint rejected");
            }
            _ => panic!("Wrong error type: {:?}", result),
        }
    }

    #[test]
    fn test_edge_case_invalid_confidence_rejected() {
        let result = GoalDiscoveryMetadata::new(
            DiscoveryMethod::Clustering,
            1.5, // Invalid
            10,
            0.8,
        );

        assert!(matches!(result, Err(GoalNodeError::InvalidConfidence(1.5))));
        println!("[EDGE CASE 2 PASSED] Invalid confidence rejected");
    }

    #[test]
    fn test_edge_case_multiple_north_stars_rejected() {
        let mut hierarchy = GoalHierarchy::new();
        let fp = test_fingerprint();
        let discovery = test_discovery();

        let ns1 = GoalNode::autonomous_goal(
            "NS1".into(),
            GoalLevel::NorthStar,
            fp.clone(),
            discovery.clone(),
        )
        .unwrap();

        let ns2 =
            GoalNode::autonomous_goal("NS2".into(), GoalLevel::NorthStar, fp, discovery).unwrap();

        hierarchy.add_goal(ns1).unwrap();
        let result = hierarchy.add_goal(ns2);

        assert!(matches!(
            result,
            Err(GoalHierarchyError::MultipleNorthStars)
        ));
        println!("[EDGE CASE 3 PASSED] Multiple North Stars rejected");
    }

    #[test]
    fn test_edge_case_orphan_child_rejected() {
        let mut hierarchy = GoalHierarchy::new();
        let fp = test_fingerprint();
        let discovery = test_discovery();

        let fake_parent = Uuid::new_v4();
        let child = GoalNode::child_goal(
            "Orphan".into(),
            GoalLevel::Strategic,
            fake_parent,
            fp,
            discovery,
        )
        .unwrap();

        let result = hierarchy.add_goal(child);
        assert!(matches!(result, Err(GoalHierarchyError::ParentNotFound(_))));
        println!("[EDGE CASE 4 PASSED] Orphan child rejected");
    }

    #[test]
    fn test_edge_case_empty_cluster_rejected() {
        let result = GoalDiscoveryMetadata::new(
            DiscoveryMethod::Clustering,
            0.8,
            0, // Invalid for Clustering
            0.7,
        );

        assert!(matches!(result, Err(GoalNodeError::EmptyCluster)));
        println!("[EDGE CASE 5 PASSED] Empty cluster rejected");
    }
}
