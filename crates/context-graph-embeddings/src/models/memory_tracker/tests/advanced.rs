//! Advanced tests for MemoryTracker.
//!
//! Tests covering:
//! - Memory deallocation
//! - Usage tracking
//! - Edge cases

use crate::error::EmbeddingError;
use crate::models::memory_tracker::MemoryTracker;
use crate::types::ModelId;

// =========================================================================
// DEALLOCATE TESTS
// =========================================================================

#[test]
fn test_deallocate_success() {
    let mut tracker = MemoryTracker::new(2_000_000_000);
    tracker.allocate(ModelId::Semantic, 1_400_000_000).unwrap();

    let freed = tracker.deallocate(ModelId::Semantic).unwrap();

    assert_eq!(freed, 1_400_000_000);
    assert_eq!(tracker.current_usage(), 0);
    assert!(!tracker.is_allocated(ModelId::Semantic));
    assert_eq!(tracker.allocation_count(), 0);
}

#[test]
fn test_deallocate_fails_when_not_loaded() {
    let mut tracker = MemoryTracker::new(1_000_000_000);

    let result = tracker.deallocate(ModelId::Semantic);
    assert!(result.is_err());
    match result {
        Err(EmbeddingError::ModelNotLoaded { model_id }) => {
            assert_eq!(model_id, ModelId::Semantic);
        }
        _ => panic!("Expected ModelNotLoaded error"),
    }
}

#[test]
fn test_deallocate_one_of_multiple() {
    let mut tracker = MemoryTracker::new(3_000_000_000);
    tracker.allocate(ModelId::Semantic, 1_400_000_000).unwrap();
    tracker.allocate(ModelId::Code, 550_000_000).unwrap();

    let freed = tracker.deallocate(ModelId::Semantic).unwrap();

    assert_eq!(freed, 1_400_000_000);
    assert_eq!(tracker.current_usage(), 550_000_000);
    assert!(!tracker.is_allocated(ModelId::Semantic));
    assert!(tracker.is_allocated(ModelId::Code));
}

#[test]
fn test_deallocate_then_reallocate() {
    let mut tracker = MemoryTracker::new(1_500_000_000);
    tracker.allocate(ModelId::Semantic, 1_400_000_000).unwrap();
    tracker.deallocate(ModelId::Semantic).unwrap();

    // Should be able to allocate again
    let result = tracker.allocate(ModelId::Semantic, 1_400_000_000);
    assert!(result.is_ok());
    assert_eq!(tracker.current_usage(), 1_400_000_000);
}

// =========================================================================
// USAGE TRACKING TESTS
// =========================================================================

#[test]
fn test_current_usage_accurate() {
    let mut tracker = MemoryTracker::new(10_000_000_000);

    tracker.allocate(ModelId::Semantic, 1_400_000_000).unwrap();
    assert_eq!(tracker.current_usage(), 1_400_000_000);

    tracker.allocate(ModelId::Code, 550_000_000).unwrap();
    assert_eq!(tracker.current_usage(), 1_950_000_000);

    tracker.deallocate(ModelId::Semantic).unwrap();
    assert_eq!(tracker.current_usage(), 550_000_000);
}

#[test]
fn test_remaining_accurate() {
    let mut tracker = MemoryTracker::new(2_000_000_000);

    assert_eq!(tracker.remaining(), 2_000_000_000);

    tracker.allocate(ModelId::Graph, 120_000_000).unwrap();
    assert_eq!(tracker.remaining(), 1_880_000_000);
}

#[test]
fn test_allocation_for_returns_correct_value() {
    let mut tracker = MemoryTracker::new(5_000_000_000);
    tracker.allocate(ModelId::Semantic, 1_400_000_000).unwrap();
    tracker
        .allocate(ModelId::Contextual, 1_600_000_000)
        .unwrap();

    assert_eq!(tracker.allocation_for(ModelId::Semantic), 1_400_000_000);
    assert_eq!(tracker.allocation_for(ModelId::Contextual), 1_600_000_000);
    assert_eq!(tracker.allocation_for(ModelId::Code), 0);
}

#[test]
fn test_allocated_models_returns_all() {
    let mut tracker = MemoryTracker::new(5_000_000_000);
    tracker.allocate(ModelId::Semantic, 1_400_000_000).unwrap();
    tracker.allocate(ModelId::Code, 550_000_000).unwrap();
    tracker.allocate(ModelId::Graph, 120_000_000).unwrap();

    let models = tracker.allocated_models();
    assert_eq!(models.len(), 3);
    assert!(models.contains(&ModelId::Semantic));
    assert!(models.contains(&ModelId::Code));
    assert!(models.contains(&ModelId::Graph));
}

// =========================================================================
// EDGE CASE TESTS
// =========================================================================

#[test]
fn test_allocate_exact_budget() {
    let mut tracker = MemoryTracker::new(1_400_000_000);
    let result = tracker.allocate(ModelId::Semantic, 1_400_000_000);

    assert!(result.is_ok());
    assert_eq!(tracker.remaining(), 0);
    assert!(!tracker.can_allocate(1));
}

#[test]
fn test_overflow_protection_on_saturating_add() {
    // Use a smaller budget that we can actually test overflow with
    let budget = 1_000_000_000;
    let mut tracker = MemoryTracker::new(budget);
    tracker.allocate(ModelId::Semantic, budget - 100).unwrap();

    // Can allocate remaining 100 bytes, but not 101
    assert!(tracker.can_allocate(100));
    assert!(!tracker.can_allocate(101));

    // Test that saturating_add doesn't overflow when checking very large values
    assert!(!tracker.can_allocate(usize::MAX));
}

#[test]
fn test_underflow_protection_on_saturating_sub() {
    let mut tracker = MemoryTracker::new(1_000_000_000);
    // Manually corrupt state (shouldn't happen in practice)
    tracker.set_current_bytes_for_test(100);
    tracker.insert_allocation_for_test(ModelId::Semantic, 200);

    // Should NOT underflow
    let freed = tracker.deallocate(ModelId::Semantic).unwrap();
    assert_eq!(freed, 200);
    assert_eq!(tracker.current_usage(), 0); // Saturated to 0
}

#[test]
fn test_budget_of_1_byte() {
    let mut tracker = MemoryTracker::new(1);

    assert!(tracker.can_allocate(1));
    assert!(!tracker.can_allocate(2));

    tracker.allocate(ModelId::Hdc, 1).unwrap();
    assert_eq!(tracker.current_usage(), 1);
    assert!(!tracker.can_allocate(1));
}
