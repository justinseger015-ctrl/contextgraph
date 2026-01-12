//! Tests for GapType enum.

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::services::gap_detection::GapType;

#[test]
fn test_gap_type_uncovered_domain_severity() {
    let gap = GapType::UncoveredDomain {
        domain: "security".into(),
    };
    assert!((gap.severity() - 0.9).abs() < f32::EPSILON);
    println!("[PASS] test_gap_type_uncovered_domain_severity");
}

#[test]
fn test_gap_type_weak_coverage_severity() {
    let gap = GapType::WeakCoverage {
        goal_id: GoalId::new(),
        coverage: 0.3,
    };
    assert!((gap.severity() - 0.7).abs() < f32::EPSILON);

    let gap_high = GapType::WeakCoverage {
        goal_id: GoalId::new(),
        coverage: 0.8,
    };
    assert!((gap_high.severity() - 0.2).abs() < f32::EPSILON);
    println!("[PASS] test_gap_type_weak_coverage_severity");
}

#[test]
fn test_gap_type_missing_link_severity() {
    let gap = GapType::MissingLink {
        from: GoalId::new(),
        to: GoalId::new(),
    };
    assert!((gap.severity() - 0.6).abs() < f32::EPSILON);
    println!("[PASS] test_gap_type_missing_link_severity");
}

#[test]
fn test_gap_type_temporal_gap_severity() {
    let gap = GapType::TemporalGap {
        period: "2024-01".into(),
    };
    assert!((gap.severity() - 0.4).abs() < f32::EPSILON);
    println!("[PASS] test_gap_type_temporal_gap_severity");
}

#[test]
fn test_gap_type_description() {
    let domain_gap = GapType::UncoveredDomain {
        domain: "security".into(),
    };
    assert!(domain_gap.description().contains("security"));
    assert!(domain_gap.description().contains("no goal coverage"));

    let goal_id = GoalId::new();
    let weak_gap = GapType::WeakCoverage {
        goal_id: goal_id.clone(),
        coverage: 0.25,
    };
    assert!(weak_gap.description().contains("weak coverage"));
    assert!(weak_gap.description().contains("25.0%"));

    let from = GoalId::new();
    let to = GoalId::new();
    let link_gap = GapType::MissingLink {
        from: from.clone(),
        to: to.clone(),
    };
    assert!(link_gap.description().contains("Missing link"));

    let temporal_gap = GapType::TemporalGap {
        period: "Q1 2024".into(),
    };
    assert!(temporal_gap.description().contains("Q1 2024"));
    println!("[PASS] test_gap_type_description");
}

#[test]
fn test_gap_type_equality() {
    let gap1 = GapType::UncoveredDomain {
        domain: "security".into(),
    };
    let gap2 = GapType::UncoveredDomain {
        domain: "security".into(),
    };
    let gap3 = GapType::UncoveredDomain {
        domain: "performance".into(),
    };

    assert_eq!(gap1, gap2);
    assert_ne!(gap1, gap3);
    println!("[PASS] test_gap_type_equality");
}

#[test]
fn test_gap_type_serialization() {
    let gap = GapType::UncoveredDomain {
        domain: "testing".into(),
    };
    let json = serde_json::to_string(&gap).expect("serialize");
    let deserialized: GapType = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(gap, deserialized);
    println!("[PASS] test_gap_type_serialization");
}
