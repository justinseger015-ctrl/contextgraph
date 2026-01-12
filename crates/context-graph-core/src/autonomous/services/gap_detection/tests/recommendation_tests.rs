//! Tests for recommendation generation.

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::services::gap_detection::{GapDetectionService, GapType};

#[test]
fn test_generate_recommendations_no_gaps() {
    let service = GapDetectionService::new();
    let recommendations = service.generate_recommendations(&[]);

    assert_eq!(recommendations.len(), 1);
    assert!(recommendations[0].contains("healthy"));
    println!("[PASS] test_generate_recommendations_no_gaps");
}

#[test]
fn test_generate_recommendations_single_uncovered_domain() {
    let service = GapDetectionService::new();
    let gaps = vec![GapType::UncoveredDomain {
        domain: "security".into(),
    }];

    let recommendations = service.generate_recommendations(&gaps);
    assert!(!recommendations.is_empty());
    assert!(recommendations[0].contains("security"));
    println!("[PASS] test_generate_recommendations_single_uncovered_domain");
}

#[test]
fn test_generate_recommendations_multiple_uncovered_domains() {
    let service = GapDetectionService::new();
    let gaps = vec![
        GapType::UncoveredDomain {
            domain: "security".into(),
        },
        GapType::UncoveredDomain {
            domain: "performance".into(),
        },
    ];

    let recommendations = service.generate_recommendations(&gaps);
    assert!(!recommendations.is_empty());
    assert!(recommendations[0].contains("2 uncovered domains"));
    println!("[PASS] test_generate_recommendations_multiple_uncovered_domains");
}

#[test]
fn test_generate_recommendations_weak_coverage() {
    let service = GapDetectionService::new();
    let gaps = vec![GapType::WeakCoverage {
        goal_id: GoalId::new(),
        coverage: 0.2,
    }];

    let recommendations = service.generate_recommendations(&gaps);
    assert!(!recommendations.is_empty());
    assert!(recommendations[0].contains("weak coverage"));
    println!("[PASS] test_generate_recommendations_weak_coverage");
}

#[test]
fn test_generate_recommendations_missing_links() {
    let service = GapDetectionService::new();
    let gaps = vec![GapType::MissingLink {
        from: GoalId::new(),
        to: GoalId::new(),
    }];

    let recommendations = service.generate_recommendations(&gaps);
    assert!(!recommendations.is_empty());
    assert!(recommendations[0].contains("link"));
    println!("[PASS] test_generate_recommendations_missing_links");
}

#[test]
fn test_generate_recommendations_temporal_gaps() {
    let service = GapDetectionService::new();
    let gaps = vec![
        GapType::TemporalGap {
            period: "Q1 2024".into(),
        },
        GapType::TemporalGap {
            period: "Q2 2024".into(),
        },
    ];

    let recommendations = service.generate_recommendations(&gaps);
    assert!(!recommendations.is_empty());
    assert!(recommendations[0].contains("2 period(s)"));
    println!("[PASS] test_generate_recommendations_temporal_gaps");
}

#[test]
fn test_generate_recommendations_mixed_gaps() {
    let service = GapDetectionService::new();
    let gaps = vec![
        GapType::UncoveredDomain {
            domain: "security".into(),
        },
        GapType::WeakCoverage {
            goal_id: GoalId::new(),
            coverage: 0.2,
        },
        GapType::MissingLink {
            from: GoalId::new(),
            to: GoalId::new(),
        },
    ];

    let recommendations = service.generate_recommendations(&gaps);
    assert!(recommendations.len() >= 3);
    println!("[PASS] test_generate_recommendations_mixed_gaps");
}
