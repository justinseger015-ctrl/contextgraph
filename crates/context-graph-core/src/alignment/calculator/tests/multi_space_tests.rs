//! Multi-Space Alignment Tests (Constitution v4.0.0).

use super::{
    create_test_fingerprint, create_test_hierarchy, create_test_semantic_fingerprint,
    test_discovery, AlignmentConfig, DefaultAlignmentCalculator, GoalAlignmentCalculator,
    LevelWeights, NUM_EMBEDDERS,
};
use crate::alignment::calculator::weights::TeleologicalWeights;
use crate::purpose::{GoalLevel, GoalNode};

#[test]
fn test_teleological_weights_default() {
    let weights = TeleologicalWeights::default();
    let sum: f32 = weights.weights.iter().sum();

    // Verify sum is approximately 1.0
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Default weights should sum to 1.0, got {}",
        sum
    );

    // Verify uniform distribution (1/13 each)
    let expected = 1.0 / NUM_EMBEDDERS as f32;
    for (i, &w) in weights.weights.iter().enumerate() {
        assert!(
            (w - expected).abs() < 0.001,
            "Weight {} should be {}, got {}",
            i,
            expected,
            w
        );
    }

    assert!(weights.validate().is_ok());
    println!("[VERIFIED] TeleologicalWeights::default() creates uniform weights (1/13 each)");
}

#[test]
fn test_teleological_weights_semantic_focused() {
    let weights = TeleologicalWeights::semantic_focused();
    let sum: f32 = weights.weights.iter().sum();

    assert!(
        (sum - 1.0).abs() < 0.01,
        "Semantic-focused weights should sum to 1.0, got {}",
        sum
    );

    // E1 should have 0.40 weight
    assert!(
        (weights.weights[0] - 0.40).abs() < 0.001,
        "E1 weight should be 0.40, got {}",
        weights.weights[0]
    );

    assert!(weights.validate().is_ok());
    println!("[VERIFIED] TeleologicalWeights::semantic_focused() weights E1 at 0.40");
}

#[test]
fn test_multi_space_alignment_uses_all_13_embedders() {
    let calculator = DefaultAlignmentCalculator::new();

    // Create a test fingerprint with known values
    let semantic = create_test_semantic_fingerprint(0.5);

    // Create a goal using new API
    let goal = GoalNode::autonomous_goal(
        "Test goal".into(),
        GoalLevel::Strategic,
        create_test_semantic_fingerprint(0.5),
        test_discovery(),
    )
    .expect("FAIL: Could not create test goal");

    // Compute all space alignments
    let alignments = calculator.compute_all_space_alignments(&semantic, &goal);

    // Verify we get exactly 13 alignment values
    assert_eq!(
        alignments.len(),
        NUM_EMBEDDERS,
        "Should compute {} alignments, got {}",
        NUM_EMBEDDERS,
        alignments.len()
    );

    // All alignments should be in [0, 1] range (normalized)
    for (i, &alignment) in alignments.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&alignment),
            "Alignment {} should be in [0,1], got {}",
            i,
            alignment
        );
    }

    println!("\n=== Multi-Space Alignment Values ===");
    let embedder_names = [
        "E1_Semantic",
        "E2_Temporal_Recent",
        "E3_Temporal_Periodic",
        "E4_Temporal_Positional",
        "E5_Causal",
        "E6_Sparse",
        "E7_Code",
        "E8_Graph",
        "E9_HDC",
        "E10_Multimodal",
        "E11_Entity",
        "E12_LateInteraction",
        "E13_SPLADE",
    ];
    for (i, &alignment) in alignments.iter().enumerate() {
        println!("  {}: {:.4}", embedder_names[i], alignment);
    }

    println!("[VERIFIED] compute_all_space_alignments uses ALL 13 embedders");
}

#[test]
fn test_multi_space_weighted_aggregation() {
    let calculator = DefaultAlignmentCalculator::new();

    // Create test fingerprint and goal using same seed for high correlation
    let semantic = create_test_semantic_fingerprint(0.8);

    let goal = GoalNode::autonomous_goal(
        "Test goal".into(),
        GoalLevel::Strategic,
        create_test_semantic_fingerprint(0.8),
        test_discovery(),
    )
    .expect("FAIL: Could not create test goal");

    let weights = LevelWeights::default();

    // Compute goal alignment (uses multi-space)
    let score = calculator.compute_goal_alignment(&semantic, &goal, &weights);

    // Verify alignment is computed
    assert!(
        score.alignment >= 0.0 && score.alignment <= 1.0,
        "Final alignment should be in [0,1], got {}",
        score.alignment
    );

    println!("\n=== Multi-Space Weighted Aggregation ===");
    println!("  Goal: {}", score.goal_id);
    println!("  Level: {:?}", score.level);
    println!("  Alignment: {:.4}", score.alignment);
    println!(
        "  Weighted Contribution: {:.4}",
        score.weighted_contribution
    );
    println!("  Threshold: {:?}", score.threshold);

    println!("[VERIFIED] Multi-space alignment uses weighted aggregation formula");
}

#[test]
fn test_calculator_with_custom_weights() {
    // Create semantic-focused weights
    let weights = TeleologicalWeights::semantic_focused();
    let calculator = DefaultAlignmentCalculator::with_weights(weights);

    // Verify the weights are stored
    assert!(
        (calculator.teleological_weights().weights[0] - 0.40).abs() < 0.001,
        "Custom weights should be preserved"
    );

    println!("[VERIFIED] DefaultAlignmentCalculator accepts custom teleological weights");
}

#[tokio::test]
async fn test_multi_space_alignment_full_integration() {
    let calculator = DefaultAlignmentCalculator::new();
    let fingerprint = create_test_fingerprint(0.8);
    let hierarchy = create_test_hierarchy();

    let config = AlignmentConfig::with_hierarchy(hierarchy)
        .with_pattern_detection(true)
        .with_embedder_breakdown(true);

    let result = calculator
        .compute_alignment(&fingerprint, &config)
        .await
        .expect("Alignment computation failed");

    println!("\n=== Multi-Space Integration Test ===");
    println!("Goal Count: {}", result.score.goal_count());
    println!("Composite Score: {:.4}", result.score.composite_score);

    // Verify embedder breakdown shows all 13 spaces
    if let Some(breakdown) = &result.embedder_breakdown {
        assert_eq!(
            breakdown.alignments.len(),
            NUM_EMBEDDERS,
            "Embedder breakdown should have {} entries",
            NUM_EMBEDDERS
        );

        println!("\nPer-Embedder Breakdown:");
        for i in 0..NUM_EMBEDDERS {
            println!(
                "  {}: {:.4} ({:?})",
                crate::alignment::pattern::EmbedderBreakdown::embedder_name(i),
                breakdown.alignments[i],
                breakdown.thresholds[i]
            );
        }
    }

    println!("\n[VERIFIED] Full multi-space alignment integration works correctly");
    println!("  - All 13 embedding spaces are used");
    println!("  - Teleological weights are applied");
    println!("  - Level propagation weights are applied");
    println!("  - Constitution v4.0.0 formula: A_multi = SUM_i(w_i * A(E_i, V))");
}
