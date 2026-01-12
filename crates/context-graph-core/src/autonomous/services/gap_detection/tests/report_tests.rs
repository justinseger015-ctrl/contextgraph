//! Tests for GapReport struct.

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::services::gap_detection::{GapReport, GapType};

#[test]
fn test_gap_report_has_gaps() {
    let empty_report = GapReport {
        gaps: vec![],
        coverage_score: 0.8,
        recommendations: vec![],
        goals_analyzed: 5,
        domains_detected: 3,
    };
    assert!(!empty_report.has_gaps());

    let report_with_gaps = GapReport {
        gaps: vec![GapType::UncoveredDomain {
            domain: "security".into(),
        }],
        coverage_score: 0.6,
        recommendations: vec![],
        goals_analyzed: 5,
        domains_detected: 3,
    };
    assert!(report_with_gaps.has_gaps());
    println!("[PASS] test_gap_report_has_gaps");
}

#[test]
fn test_gap_report_gaps_by_severity() {
    let report = GapReport {
        gaps: vec![
            GapType::TemporalGap {
                period: "Q1".into(),
            }, // 0.4
            GapType::UncoveredDomain {
                domain: "security".into(),
            }, // 0.9
            GapType::MissingLink {
                from: GoalId::new(),
                to: GoalId::new(),
            }, // 0.6
        ],
        coverage_score: 0.5,
        recommendations: vec![],
        goals_analyzed: 5,
        domains_detected: 3,
    };

    let sorted = report.gaps_by_severity();
    assert_eq!(sorted.len(), 3);
    assert!((sorted[0].severity() - 0.9).abs() < f32::EPSILON); // UncoveredDomain first
    assert!((sorted[1].severity() - 0.6).abs() < f32::EPSILON); // MissingLink second
    assert!((sorted[2].severity() - 0.4).abs() < f32::EPSILON); // TemporalGap last
    println!("[PASS] test_gap_report_gaps_by_severity");
}

#[test]
fn test_gap_report_gap_counts() {
    let report = GapReport {
        gaps: vec![
            GapType::UncoveredDomain {
                domain: "security".into(),
            },
            GapType::UncoveredDomain {
                domain: "performance".into(),
            },
            GapType::WeakCoverage {
                goal_id: GoalId::new(),
                coverage: 0.3,
            },
        ],
        coverage_score: 0.5,
        recommendations: vec![],
        goals_analyzed: 5,
        domains_detected: 3,
    };

    let counts = report.gap_counts();
    assert_eq!(counts.get("uncovered_domain"), Some(&2));
    assert_eq!(counts.get("weak_coverage"), Some(&1));
    assert_eq!(counts.get("missing_link"), None);
    println!("[PASS] test_gap_report_gap_counts");
}

#[test]
fn test_gap_report_most_severe_gap() {
    let report = GapReport {
        gaps: vec![
            GapType::TemporalGap {
                period: "Q1".into(),
            },
            GapType::WeakCoverage {
                goal_id: GoalId::new(),
                coverage: 0.2,
            }, // severity 0.8
            GapType::MissingLink {
                from: GoalId::new(),
                to: GoalId::new(),
            },
        ],
        coverage_score: 0.5,
        recommendations: vec![],
        goals_analyzed: 5,
        domains_detected: 3,
    };

    let most_severe = report.most_severe_gap();
    assert!(most_severe.is_some());
    assert!((most_severe.unwrap().severity() - 0.8).abs() < f32::EPSILON);

    let empty_report = GapReport {
        gaps: vec![],
        coverage_score: 0.9,
        recommendations: vec![],
        goals_analyzed: 0,
        domains_detected: 0,
    };
    assert!(empty_report.most_severe_gap().is_none());
    println!("[PASS] test_gap_report_most_severe_gap");
}
