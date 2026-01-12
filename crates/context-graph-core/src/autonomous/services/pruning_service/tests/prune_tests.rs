//! Tests for PruningService prune operations

use crate::autonomous::curation::{MemoryId, PruningConfig};
use crate::autonomous::services::pruning_service::types::{
    ExtendedPruningConfig, PruneReason, PruningCandidate,
};
use crate::autonomous::services::pruning_service::PruningService;

use super::helpers::make_metadata;

// ============================================================================
// prune tests
// ============================================================================

#[test]
fn test_prune_empty() {
    let mut service = PruningService::new();
    let report = service.prune(&[]);

    assert_eq!(report.candidates_evaluated, 0);
    assert_eq!(report.pruned_count, 0);
    assert_eq!(report.bytes_freed, 0);
    println!("[PASS] prune with empty input");
}

#[test]
fn test_prune_basic() {
    let mut service = PruningService::new();
    let candidates = vec![
        PruningCandidate::new(
            MemoryId::new(),
            60,
            0.30,
            1,
            PruneReason::LowAlignment,
            1024,
        ),
        PruningCandidate::new(MemoryId::new(), 60, 0.20, 0, PruneReason::Orphaned, 2048),
    ];

    let report = service.prune(&candidates);
    assert_eq!(report.candidates_evaluated, 2);
    assert_eq!(report.pruned_count, 2);
    assert_eq!(report.bytes_freed, 3072);
    println!("[PASS] prune basic operation");
}

#[test]
fn test_prune_respects_daily_limit() {
    let config = ExtendedPruningConfig {
        max_daily_prunes: 2,
        ..Default::default()
    };
    let mut service = PruningService::with_config(config);

    let candidates: Vec<PruningCandidate> = (0..5)
        .map(|_| {
            PruningCandidate::new(MemoryId::new(), 60, 0.30, 1, PruneReason::LowAlignment, 100)
        })
        .collect();

    let report = service.prune(&candidates);
    assert_eq!(report.candidates_evaluated, 5);
    assert_eq!(report.pruned_count, 2);
    assert!(report.daily_limit_reached);
    println!("[PASS] prune respects daily limit");
}

#[test]
fn test_prune_preserves_connected() {
    let config = ExtendedPruningConfig {
        base: PruningConfig {
            preserve_connected: true,
            min_connections: 3,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut service = PruningService::with_config(config);

    let candidates = vec![
        PruningCandidate::new(
            MemoryId::new(),
            60,
            0.30,
            5,
            PruneReason::LowAlignment,
            1024,
        ), // Preserved
        PruningCandidate::new(
            MemoryId::new(),
            60,
            0.30,
            1,
            PruneReason::LowAlignment,
            1024,
        ), // Pruned
    ];

    let report = service.prune(&candidates);
    assert_eq!(report.pruned_count, 1);
    assert_eq!(report.preserved_count, 1);
    println!("[PASS] prune preserves connected memories");
}

// ============================================================================
// estimate_bytes_freed tests
// ============================================================================

#[test]
fn test_estimate_bytes_freed_empty() {
    let service = PruningService::new();
    let estimate = service.estimate_bytes_freed(&[]);
    assert_eq!(estimate, 0);
    println!("[PASS] estimate_bytes_freed with empty input");
}

#[test]
fn test_estimate_bytes_freed_basic() {
    let service = PruningService::new();
    let candidates = vec![
        PruningCandidate::new(
            MemoryId::new(),
            60,
            0.30,
            1,
            PruneReason::LowAlignment,
            1000,
        ),
        PruningCandidate::new(
            MemoryId::new(),
            60,
            0.30,
            1,
            PruneReason::LowAlignment,
            2000,
        ),
        PruningCandidate::new(
            MemoryId::new(),
            60,
            0.30,
            1,
            PruneReason::LowAlignment,
            3000,
        ),
    ];

    let estimate = service.estimate_bytes_freed(&candidates);
    assert_eq!(estimate, 6000);
    println!("[PASS] estimate_bytes_freed = 6000");
}

#[test]
fn test_estimate_bytes_freed_respects_limit() {
    let config = ExtendedPruningConfig {
        max_daily_prunes: 2,
        ..Default::default()
    };
    let service = PruningService::with_config(config);

    let candidates = vec![
        PruningCandidate::new(
            MemoryId::new(),
            60,
            0.30,
            1,
            PruneReason::LowAlignment,
            1000,
        ),
        PruningCandidate::new(
            MemoryId::new(),
            60,
            0.30,
            1,
            PruneReason::LowAlignment,
            2000,
        ),
        PruningCandidate::new(
            MemoryId::new(),
            60,
            0.30,
            1,
            PruneReason::LowAlignment,
            3000,
        ),
    ];

    let estimate = service.estimate_bytes_freed(&candidates);
    // Only first 2 count due to limit
    assert_eq!(estimate, 3000);
    println!("[PASS] estimate_bytes_freed respects daily limit");
}

// ============================================================================
// Daily rollover tests
// ============================================================================

#[test]
fn test_remaining_daily_prunes() {
    let config = ExtendedPruningConfig {
        max_daily_prunes: 100,
        ..Default::default()
    };
    let mut service = PruningService::with_config(config);

    assert_eq!(service.remaining_daily_prunes(), 100);

    // Prune some
    let candidates = vec![
        PruningCandidate::new(MemoryId::new(), 60, 0.30, 1, PruneReason::LowAlignment, 100),
        PruningCandidate::new(MemoryId::new(), 60, 0.30, 1, PruneReason::LowAlignment, 100),
    ];
    service.prune(&candidates);

    assert_eq!(service.remaining_daily_prunes(), 98);
    println!("[PASS] remaining_daily_prunes tracks correctly");
}

// ============================================================================
// Integration test
// ============================================================================

#[test]
fn test_full_pruning_workflow() {
    let mut service = PruningService::new();

    // Create a diverse set of memories
    let memories = vec![
        // Healthy - should not be pruned
        make_metadata(0.85, 60, 5, 1000),
        // Low alignment
        make_metadata(0.20, 60, 2, 2000),
        // Orphaned
        make_metadata(0.75, 60, 0, 3000),
        // Too young
        make_metadata(0.10, 10, 0, 4000),
        // Well connected despite low alignment
        make_metadata(0.25, 60, 10, 5000),
    ];

    // Identify candidates
    let candidates = service.identify_candidates(&memories);

    // Should find: low alignment (0.20) and orphaned (0.75)
    // The well-connected one should be excluded
    // The too-young one should be excluded
    assert_eq!(candidates.len(), 2);
    println!(
        "[PASS] Full workflow: identified {} candidates",
        candidates.len()
    );

    // Prune
    let report = service.prune(&candidates);

    assert_eq!(report.pruned_count, 2);
    assert_eq!(report.bytes_freed, 5000); // 2000 + 3000
    assert!(!report.daily_limit_reached);
    println!(
        "[PASS] Full workflow: pruned {} memories, freed {} bytes",
        report.pruned_count, report.bytes_freed
    );

    // Verify reasons breakdown
    assert!(
        report
            .reasons_breakdown
            .contains_key(&PruneReason::LowAlignment)
            || report
                .reasons_breakdown
                .contains_key(&PruneReason::Orphaned)
    );
    println!("[PASS] Full workflow: reasons breakdown populated");
}
