//! Pattern detection tests.

use uuid::Uuid;

use super::{
    create_test_hierarchy, AlignmentConfig, DefaultAlignmentCalculator, GoalAlignmentScore,
    GoalScore, LevelWeights, MisalignmentFlags,
};
use crate::alignment::calculator::GoalAlignmentCalculator;
use crate::alignment::pattern::PatternType;
use crate::purpose::GoalLevel;

#[test]
fn test_detect_patterns_optimal() {
    let calculator = DefaultAlignmentCalculator::new();
    let hierarchy = create_test_hierarchy();

    // Create optimal score using UUIDs
    let scores = vec![
        GoalScore::new(Uuid::new_v4(), GoalLevel::NorthStar, 0.85, 0.4),
        GoalScore::new(Uuid::new_v4(), GoalLevel::Strategic, 0.80, 0.3),
        GoalScore::new(Uuid::new_v4(), GoalLevel::Tactical, 0.78, 0.2),
        GoalScore::new(Uuid::new_v4(), GoalLevel::Immediate, 0.76, 0.1),
    ];
    let score = GoalAlignmentScore::compute(scores, LevelWeights::default());
    let flags = MisalignmentFlags::empty();
    let config = AlignmentConfig::with_hierarchy(hierarchy);

    let patterns = calculator.detect_patterns(&score, &flags, &config);

    println!("\n=== Detected Patterns ===");
    for p in &patterns {
        println!(
            "  - {:?}: {} (severity {})",
            p.pattern_type, p.description, p.severity
        );
    }

    // Should detect OptimalAlignment and HierarchicalCoherence
    let has_optimal = patterns
        .iter()
        .any(|p| p.pattern_type == PatternType::OptimalAlignment);
    let has_coherence = patterns
        .iter()
        .any(|p| p.pattern_type == PatternType::HierarchicalCoherence);

    assert!(
        has_optimal || has_coherence,
        "Should detect positive patterns for optimal alignment"
    );
    println!("[VERIFIED] detect_patterns identifies positive patterns");
}

#[test]
fn test_detect_patterns_north_star_drift() {
    let calculator = DefaultAlignmentCalculator::new();
    let hierarchy = create_test_hierarchy();

    // Create score with low North Star alignment using UUIDs
    let scores = vec![
        GoalScore::new(Uuid::new_v4(), GoalLevel::NorthStar, 0.40, 0.4), // Below warning
        GoalScore::new(Uuid::new_v4(), GoalLevel::Strategic, 0.80, 0.3),
    ];
    let score = GoalAlignmentScore::compute(scores, LevelWeights::default());
    let flags = MisalignmentFlags::empty();
    let config = AlignmentConfig::with_hierarchy(hierarchy);

    let patterns = calculator.detect_patterns(&score, &flags, &config);

    println!("\n=== North Star Drift Detection ===");
    println!("BEFORE: north_star_alignment = 0.40");
    for p in &patterns {
        println!(
            "AFTER: pattern = {:?}, severity = {}",
            p.pattern_type, p.severity
        );
    }

    let has_drift = patterns
        .iter()
        .any(|p| p.pattern_type == PatternType::NorthStarDrift);
    assert!(has_drift, "Should detect NorthStarDrift pattern");
    println!("[VERIFIED] detect_patterns identifies NorthStarDrift");
}
