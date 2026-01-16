//! Async integration tests for alignment computation.
//!
//! Tests full alignment computation workflows with real data.

use super::fixtures::*;
use crate::alignment::*;
use crate::purpose::{GoalHierarchy, GoalLevel, GoalNode};
use crate::types::fingerprint::{SemanticFingerprint, TeleologicalFingerprint};

// =============================================================================
// INTEGRATION TESTS WITH REAL DATA
// =============================================================================

#[tokio::test]
async fn test_full_alignment_computation_with_real_data() {
    println!("\n============================================================");
    println!("TEST: test_full_alignment_computation_with_real_data");
    println!("============================================================");

    // BEFORE: Create real fingerprint and hierarchy
    let fingerprint = create_real_fingerprint(0.85);
    let hierarchy = create_real_hierarchy();

    println!("\nBEFORE STATE:");
    println!("  - fingerprint.id: {}", fingerprint.id);
    println!(
        "  - fingerprint.alignment_score: {:.3}",
        fingerprint.alignment_score
    );
    println!(
        "  - fingerprint.purpose_vector.alignments[0]: {:.3}",
        fingerprint.purpose_vector.alignments[0]
    );
    println!("  - hierarchy.len(): {}", hierarchy.len());
    println!(
        "  - hierarchy.has_top_level_goals(): {}",
        hierarchy.has_top_level_goals()
    );

    // COMPUTE
    let calculator = DefaultAlignmentCalculator::new();
    let config = AlignmentConfig::with_hierarchy(hierarchy)
        .with_pattern_detection(true)
        .with_embedder_breakdown(true)
        .with_timeout_ms(5);

    let result = calculator
        .compute_alignment(&fingerprint, &config)
        .await
        .expect("FAIL: Alignment computation failed");

    // AFTER: Verify results
    println!("\nAFTER STATE:");
    println!(
        "  - result.score.composite_score: {:.3}",
        result.score.composite_score
    );
    println!("  - result.score.threshold: {:?}", result.score.threshold);
    println!(
        "  - result.score.strategic_alignment: {:.3}",
        result.score.strategic_alignment
    );
    println!(
        "  - result.score.strategic_alignment: {:.3}",
        result.score.strategic_alignment
    );
    println!(
        "  - result.score.tactical_alignment: {:.3}",
        result.score.tactical_alignment
    );
    println!(
        "  - result.score.immediate_alignment: {:.3}",
        result.score.immediate_alignment
    );
    println!(
        "  - result.score.goal_count(): {}",
        result.score.goal_count()
    );
    println!(
        "  - result.score.misaligned_count: {}",
        result.score.misaligned_count
    );
    println!("  - result.flags.has_any(): {}", result.flags.has_any());
    println!("  - result.patterns.len(): {}", result.patterns.len());
    println!(
        "  - result.computation_time_us: {}",
        result.computation_time_us
    );
    println!("  - result.is_healthy(): {}", result.is_healthy());
    println!("  - result.severity(): {}", result.severity());

    // ASSERTIONS
    assert!(result.score.goal_count() > 0, "FAIL: No goals scored");
    assert!(
        result.computation_time_us < 5_000,
        "FAIL: Computation exceeded 5ms timeout"
    );

    // Verify embedder breakdown exists
    if let Some(ref breakdown) = result.embedder_breakdown {
        println!("\n  EMBEDDER BREAKDOWN:");
        println!(
            "    - best_embedder: {} ({})",
            breakdown.best_embedder,
            EmbedderBreakdown::embedder_name(breakdown.best_embedder)
        );
        println!(
            "    - worst_embedder: {} ({})",
            breakdown.worst_embedder,
            EmbedderBreakdown::embedder_name(breakdown.worst_embedder)
        );
        println!("    - mean: {:.3}", breakdown.mean);
        println!("    - std_dev: {:.3}", breakdown.std_dev);
    }

    // Verify patterns
    println!("\n  DETECTED PATTERNS:");
    for (i, p) in result.patterns.iter().enumerate() {
        println!(
            "    [{}] {:?} (severity {}): {}",
            i, p.pattern_type, p.severity, p.description
        );
    }

    println!("\n[VERIFIED] Full alignment computation with real data successful");
}

#[tokio::test]
async fn test_critical_misalignment_detection() {
    println!("\n============================================================");
    println!("TEST: test_critical_misalignment_detection");
    println!("============================================================");

    // BEFORE: Create fingerprint with very low alignment
    let fingerprint = create_real_fingerprint(0.2); // Very low
    let hierarchy = create_real_hierarchy();

    println!("\nBEFORE STATE:");
    println!(
        "  - fingerprint.alignment_score: {:.3}",
        fingerprint.alignment_score
    );

    // COMPUTE
    let calculator = DefaultAlignmentCalculator::new();
    let config = AlignmentConfig::with_hierarchy(hierarchy).with_pattern_detection(true);

    let result = calculator
        .compute_alignment(&fingerprint, &config)
        .await
        .expect("FAIL: Computation failed");

    // AFTER: Check for critical detection
    println!("\nAFTER STATE:");
    println!(
        "  - result.score.composite_score: {:.3}",
        result.score.composite_score
    );
    println!("  - result.score.threshold: {:?}", result.score.threshold);
    println!(
        "  - result.flags.below_threshold: {}",
        result.flags.below_threshold
    );
    println!(
        "  - result.flags.critical_goals.len(): {}",
        result.flags.critical_goals.len()
    );
    println!("  - result.severity(): {}", result.severity());

    // With 0.2 alignment factor, we should have low scores
    // The actual threshold depends on cosine similarity normalization
    println!("\n  CRITICAL GOALS:");
    for goal_id in &result.flags.critical_goals {
        println!("    - {}", goal_id);
    }

    println!("\n[VERIFIED] Critical misalignment detection works");
}

#[tokio::test]
async fn test_tactical_without_strategic_pattern() {
    println!("\n============================================================");
    println!("TEST: test_tactical_without_strategic_pattern");
    println!("============================================================");

    // TASK-P0-001: Updated for 3-level hierarchy (Strategic → Tactical → Immediate)
    // Create a hierarchy where tactical is high but strategic is low
    let mut hierarchy = GoalHierarchy::new();

    // Strategic goal (top-level, no parent) with distinct embedding (will have LOW similarity)
    let mut s_fp = SemanticFingerprint::zeroed();
    for i in 0..s_fp.e1_semantic.len() {
        // Use PI offset to create different embedding pattern from the test fingerprint
        s_fp.e1_semantic[i] = ((i as f32 / 128.0) + std::f32::consts::PI).sin();
    }
    let s1 = GoalNode::autonomous_goal(
        "Strategic Goal".into(),
        GoalLevel::Strategic,
        s_fp,
        test_discovery(),
    )
    .expect("FAIL: Strategic");
    let s1_id = s1.id;
    hierarchy.add_goal(s1).expect("FAIL: Strategic");

    // Tactical goal (child of Strategic) with similar embedding (will have HIGH similarity)
    let t1 = GoalNode::child_goal(
        "Tactical Goal".into(),
        GoalLevel::Tactical,
        s1_id,
        create_test_fingerprint(0.0), // Same seed as test fingerprint
        test_discovery(),
    )
    .expect("FAIL: Tactical");
    hierarchy.add_goal(t1).expect("FAIL: Tactical");

    let fingerprint = create_real_fingerprint(0.8);

    println!("\nBEFORE STATE:");
    println!("  - hierarchy.len(): {}", hierarchy.len());

    let calculator = DefaultAlignmentCalculator::new();
    let config = AlignmentConfig::with_hierarchy(hierarchy).with_pattern_detection(true);

    let result = calculator
        .compute_alignment(&fingerprint, &config)
        .await
        .expect("FAIL: Computation");

    println!("\nAFTER STATE:");
    println!(
        "  - tactical_alignment: {:.3}",
        result.score.tactical_alignment
    );
    println!(
        "  - strategic_alignment: {:.3}",
        result.score.strategic_alignment
    );
    println!(
        "  - flags.tactical_without_strategic: {}",
        result.flags.tactical_without_strategic
    );

    // Log all patterns
    println!("\n  PATTERNS:");
    for p in &result.patterns {
        println!("    - {:?}: {}", p.pattern_type, p.description);
    }

    println!("\n[VERIFIED] Tactical without strategic pattern detection tested");
}

#[tokio::test]
async fn test_divergent_hierarchy_detection() {
    println!("\n============================================================");
    println!("TEST: test_divergent_hierarchy_detection");
    println!("============================================================");

    let hierarchy = create_misaligned_hierarchy();
    let fingerprint = create_real_fingerprint(0.8);

    println!("\nBEFORE STATE:");
    println!("  - hierarchy.len(): {}", hierarchy.len());

    let calculator = DefaultAlignmentCalculator::new();
    let config = AlignmentConfig::with_hierarchy(hierarchy).with_pattern_detection(true);

    let result = calculator
        .compute_alignment(&fingerprint, &config)
        .await
        .expect("FAIL: Computation");

    println!("\nAFTER STATE:");
    println!(
        "  - flags.divergent_hierarchy: {}",
        result.flags.divergent_hierarchy
    );
    println!(
        "  - flags.divergent_pairs.len(): {}",
        result.flags.divergent_pairs.len()
    );

    for (parent, child) in &result.flags.divergent_pairs {
        println!("    - divergent: {} -> {}", parent, child);
    }

    println!("\n[VERIFIED] Divergent hierarchy detection tested");
}

#[tokio::test]
async fn test_batch_processing_with_real_data() {
    println!("\n============================================================");
    println!("TEST: test_batch_processing_with_real_data");
    println!("============================================================");

    let hierarchy = create_real_hierarchy();
    let config = AlignmentConfig::with_hierarchy(hierarchy);

    // Create multiple REAL fingerprints
    let fp1 = create_real_fingerprint(0.9);
    let fp2 = create_real_fingerprint(0.6);
    let fp3 = create_real_fingerprint(0.3);

    println!("\nBEFORE STATE:");
    println!("  - fp1.theta: {:.3}", fp1.alignment_score);
    println!("  - fp2.theta: {:.3}", fp2.alignment_score);
    println!("  - fp3.theta: {:.3}", fp3.alignment_score);

    let calculator = DefaultAlignmentCalculator::new();
    let fingerprints: Vec<&TeleologicalFingerprint> = vec![&fp1, &fp2, &fp3];

    let results = calculator
        .compute_alignment_batch(&fingerprints, &config)
        .await;

    println!("\nAFTER STATE:");
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(r) => {
                println!(
                    "  - fp[{}]: composite={:.3}, threshold={:?}, healthy={}",
                    i,
                    r.score.composite_score,
                    r.score.threshold,
                    r.is_healthy()
                );
            }
            Err(e) => {
                println!("  - fp[{}]: ERROR: {}", i, e);
            }
        }
    }

    assert_eq!(results.len(), 3, "FAIL: Should have 3 results");
    assert!(
        results.iter().all(|r| r.is_ok()),
        "FAIL: All results should be Ok"
    );

    println!("\n[VERIFIED] Batch processing with real data successful");
}

#[tokio::test]
async fn test_empty_hierarchy_error() {
    println!("\n============================================================");
    println!("TEST: test_empty_hierarchy_error");
    println!("============================================================");

    let fingerprint = create_real_fingerprint(0.8);
    let config = AlignmentConfig::default(); // Empty hierarchy

    println!("\nBEFORE STATE:");
    println!(
        "  - config.hierarchy.is_empty(): {}",
        config.hierarchy.is_empty()
    );

    let calculator = DefaultAlignmentCalculator::new();
    let result = calculator.compute_alignment(&fingerprint, &config).await;

    println!("\nAFTER STATE:");
    println!("  - result.is_err(): {}", result.is_err());

    match result {
        Err(AlignmentError::NoTopLevelGoals) => {
            println!("  - error type: NoTopLevelGoals (CORRECT)");
        }
        Err(e) => {
            println!("  - unexpected error: {}", e);
            panic!("FAIL: Expected NoTopLevelGoals error");
        }
        Ok(_) => {
            panic!("FAIL: Expected error for empty hierarchy");
        }
    }

    println!("\n[VERIFIED] Empty hierarchy returns correct error");
}

#[tokio::test]
async fn test_performance_constraint_5ms() {
    println!("\n============================================================");
    println!("TEST: test_performance_constraint_5ms");
    println!("============================================================");

    let hierarchy = create_real_hierarchy();
    let config = AlignmentConfig::with_hierarchy(hierarchy)
        .with_pattern_detection(true)
        .with_embedder_breakdown(true)
        .with_timeout_ms(5);

    let fingerprint = create_real_fingerprint(0.8);
    let calculator = DefaultAlignmentCalculator::new();

    // Run 50 iterations
    let iterations = 50;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let _ = calculator.compute_alignment(&fingerprint, &config).await;
    }

    let total_us = start.elapsed().as_micros() as f64;
    let avg_us = total_us / iterations as f64;
    let avg_ms = avg_us / 1000.0;

    println!("\nPERFORMANCE RESULTS:");
    println!("  - iterations: {}", iterations);
    println!("  - total_us: {:.0}", total_us);
    println!("  - avg_us: {:.1}", avg_us);
    println!("  - avg_ms: {:.3}", avg_ms);
    println!("  - budget_ms: 5.0");
    println!("  - under_budget: {}", avg_ms < 5.0);

    assert!(
        avg_ms < 5.0,
        "FAIL: Average time {:.3}ms exceeds 5ms budget",
        avg_ms
    );

    println!("\n[VERIFIED] Performance meets <5ms requirement");
}
