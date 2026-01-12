//! Tests for PruningService implementation

use chrono::{Duration, Utc};

use crate::autonomous::curation::{MemoryId, PruningConfig};
use crate::autonomous::services::pruning_service::types::{
    ExtendedPruningConfig, PruneReason, PruningCandidate,
};
use crate::autonomous::services::pruning_service::PruningService;

use super::helpers::{make_metadata, make_metadata_with_access};

// ============================================================================
// PruningService construction tests
// ============================================================================

#[test]
fn test_pruning_service_new() {
    let service = PruningService::new();
    assert_eq!(service.config().max_daily_prunes, 100);
    println!("[PASS] PruningService::new");
}

#[test]
fn test_pruning_service_with_config() {
    let config = ExtendedPruningConfig {
        max_daily_prunes: 50,
        ..Default::default()
    };
    let service = PruningService::with_config(config);
    assert_eq!(service.config().max_daily_prunes, 50);
    println!("[PASS] PruningService::with_config");
}

#[test]
fn test_pruning_service_default() {
    let service = PruningService::default();
    assert_eq!(service.config().max_daily_prunes, 100);
    println!("[PASS] PruningService::default");
}

// ============================================================================
// identify_candidates tests
// ============================================================================

#[test]
fn test_identify_candidates_empty() {
    let service = PruningService::new();
    let candidates = service.identify_candidates(&[]);
    assert!(candidates.is_empty());
    println!("[PASS] identify_candidates with empty input");
}

#[test]
fn test_identify_candidates_low_alignment() {
    let service = PruningService::new();
    let memories = vec![
        make_metadata(0.20, 60, 1, 1024), // Low alignment, should be candidate
        make_metadata(0.80, 60, 1, 1024), // High alignment, should not
    ];

    let candidates = service.identify_candidates(&memories);
    assert_eq!(candidates.len(), 1);
    assert!((candidates[0].alignment - 0.20).abs() < f32::EPSILON);
    assert_eq!(candidates[0].reason, PruneReason::LowAlignment);
    println!("[PASS] identify_candidates finds low alignment");
}

#[test]
fn test_identify_candidates_orphaned() {
    let service = PruningService::new();
    let memories = vec![
        make_metadata(0.80, 60, 0, 1024), // No connections = orphaned
        make_metadata(0.80, 60, 5, 1024), // Has connections, should not
    ];

    let candidates = service.identify_candidates(&memories);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].connections, 0);
    assert_eq!(candidates[0].reason, PruneReason::Orphaned);
    println!("[PASS] identify_candidates finds orphaned");
}

#[test]
fn test_identify_candidates_too_young() {
    let service = PruningService::new();
    // 10 days old, below min_age_days of 30
    let memories = vec![make_metadata(0.20, 10, 0, 1024)];

    let candidates = service.identify_candidates(&memories);
    assert!(candidates.is_empty());
    println!("[PASS] identify_candidates ignores young memories");
}

#[test]
fn test_identify_candidates_stale() {
    let service = PruningService::new();
    // Stale: 100 days since access (> 90 day threshold)
    let memories = vec![make_metadata_with_access(0.80, 120, 2, 100)];

    let candidates = service.identify_candidates(&memories);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].reason, PruneReason::Stale);
    println!("[PASS] identify_candidates finds stale memories");
}

#[test]
fn test_identify_candidates_low_quality() {
    let service = PruningService::new();
    let mut meta = make_metadata(0.80, 60, 2, 1024);
    meta.quality_score = Some(0.10); // Below 0.30 threshold

    let candidates = service.identify_candidates(&[meta]);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].reason, PruneReason::LowQuality);
    println!("[PASS] identify_candidates finds low quality");
}

#[test]
fn test_identify_candidates_redundant() {
    let service = PruningService::new();

    let mut meta1 = make_metadata(0.80, 60, 2, 1024);
    meta1.content_hash = Some(12345);

    let mut meta2 = make_metadata(0.50, 60, 2, 1024);
    meta2.content_hash = Some(12345); // Same hash, lower alignment

    let candidates = service.identify_candidates(&[meta1, meta2]);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].reason, PruneReason::Redundant);
    println!("[PASS] identify_candidates finds redundant memories");
}

#[test]
fn test_identify_candidates_preserves_connected() {
    let config = ExtendedPruningConfig {
        base: PruningConfig {
            preserve_connected: true,
            min_connections: 3,
            ..Default::default()
        },
        ..Default::default()
    };
    let service = PruningService::with_config(config);

    // Low alignment but well connected - should be preserved
    let memories = vec![make_metadata(0.20, 60, 5, 1024)];

    let candidates = service.identify_candidates(&memories);
    assert!(candidates.is_empty());
    println!("[PASS] identify_candidates preserves well-connected memories");
}

#[test]
fn test_identify_candidates_sorted_by_priority() {
    let service = PruningService::new();
    let memories = vec![
        make_metadata(0.35, 60, 1, 1024), // Higher alignment, lower priority
        make_metadata(0.10, 60, 0, 1024), // Lower alignment, higher priority
        make_metadata(0.25, 60, 1, 1024), // Medium
    ];

    let candidates = service.identify_candidates(&memories);
    assert_eq!(candidates.len(), 3);
    // Should be sorted by priority (highest first)
    assert!(candidates[0].priority_score >= candidates[1].priority_score);
    assert!(candidates[1].priority_score >= candidates[2].priority_score);
    println!("[PASS] identify_candidates sorted by priority");
}

// ============================================================================
// evaluate_candidate tests
// ============================================================================

#[test]
fn test_evaluate_candidate_prune_disabled() {
    let config = ExtendedPruningConfig {
        base: PruningConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let service = PruningService::with_config(config);
    let meta = make_metadata(0.10, 60, 0, 1024);

    let candidate = service.evaluate_candidate(&meta);
    assert!(candidate.is_none());
    println!("[PASS] evaluate_candidate returns None when disabled");
}

#[test]
fn test_evaluate_candidate_healthy_memory() {
    let service = PruningService::new();
    // Good alignment, has connections, not stale
    let mut meta = make_metadata(0.80, 60, 5, 1024);
    meta.last_accessed = Some(Utc::now() - Duration::days(5));
    meta.quality_score = Some(0.90);

    let candidate = service.evaluate_candidate(&meta);
    assert!(candidate.is_none());
    println!("[PASS] evaluate_candidate returns None for healthy memory");
}

// ============================================================================
// should_prune tests
// ============================================================================

#[test]
fn test_should_prune_basic() {
    let service = PruningService::new();
    let candidate = PruningCandidate::new(
        MemoryId::new(),
        60,
        0.30,
        1,
        PruneReason::LowAlignment,
        1024,
    );

    assert!(service.should_prune(&candidate));
    println!("[PASS] should_prune returns true for basic candidate");
}

#[test]
fn test_should_prune_preserves_connected() {
    let config = ExtendedPruningConfig {
        base: PruningConfig {
            preserve_connected: true,
            min_connections: 3,
            ..Default::default()
        },
        ..Default::default()
    };
    let service = PruningService::with_config(config);

    let candidate = PruningCandidate::new(
        MemoryId::new(),
        60,
        0.30,
        5,
        PruneReason::LowAlignment,
        1024,
    );

    assert!(!service.should_prune(&candidate));
    println!("[PASS] should_prune preserves connected memories");
}

// ============================================================================
// get_prune_reason tests
// ============================================================================

#[test]
fn test_get_prune_reason_none() {
    let service = PruningService::new();
    let mut meta = make_metadata(0.80, 60, 5, 1024);
    meta.last_accessed = Some(Utc::now());
    meta.quality_score = Some(0.90);

    let reason = service.get_prune_reason(&meta);
    assert!(reason.is_none());
    println!("[PASS] get_prune_reason returns None for healthy memory");
}

#[test]
fn test_get_prune_reason_priority() {
    let service = PruningService::new();

    // Orphaned takes priority over low alignment
    let meta = make_metadata(0.30, 60, 0, 1024);
    let reason = service.get_prune_reason(&meta);
    assert_eq!(reason, Some(PruneReason::Orphaned));
    println!("[PASS] get_prune_reason: Orphaned > LowAlignment");

    // Low alignment
    let meta = make_metadata(0.30, 60, 1, 1024);
    let reason = service.get_prune_reason(&meta);
    assert_eq!(reason, Some(PruneReason::LowAlignment));
    println!("[PASS] get_prune_reason: LowAlignment");
}

// ============================================================================
// Hash tracking tests
// ============================================================================

#[test]
fn test_register_and_check_hash() {
    let mut service = PruningService::new();
    let id = MemoryId::new();

    assert!(!service.is_hash_known(12345));

    service.register_hash(12345, id.clone());

    assert!(service.is_hash_known(12345));
    assert_eq!(service.get_hash_owner(12345), Some(&id));
    println!("[PASS] register_hash and is_hash_known");
}
