//! Error Tests - All PurposeIndexError variants

use crate::index::error::IndexError;
use crate::index::purpose::error::{PurposeIndexError, PurposeIndexResult};
use uuid::Uuid;

#[test]
fn test_error_not_found_has_descriptive_message() {
    let id = Uuid::new_v4();
    let err = PurposeIndexError::not_found(id);
    let msg = err.to_string();

    assert!(
        msg.contains(&id.to_string()),
        "Error should contain memory ID"
    );
    assert!(
        msg.contains("not found"),
        "Error should contain 'not found'"
    );

    println!("[VERIFIED] NotFound error contains memory ID: {}", msg);
}

#[test]
fn test_error_invalid_confidence_has_descriptive_message() {
    let err = PurposeIndexError::invalid_confidence(1.5, "test context");
    let msg = err.to_string();

    assert!(msg.contains("1.5"), "Error should contain invalid value");
    assert!(msg.contains("test context"), "Error should contain context");

    println!(
        "[VERIFIED] InvalidConfidence error contains value and context: {}",
        msg
    );
}

#[test]
fn test_error_invalid_query_has_descriptive_message() {
    let err = PurposeIndexError::invalid_query("limit must be positive");
    let msg = err.to_string();

    assert!(
        msg.contains("limit must be positive"),
        "Error should contain reason"
    );

    println!("[VERIFIED] InvalidQuery error contains reason: {}", msg);
}

#[test]
fn test_error_dimension_mismatch_has_descriptive_message() {
    let err = PurposeIndexError::dimension_mismatch(13, 10);
    let msg = err.to_string();

    assert!(
        msg.contains("13"),
        "Error should contain expected dimension"
    );
    assert!(msg.contains("10"), "Error should contain actual dimension");

    println!(
        "[VERIFIED] DimensionMismatch error contains dimensions: {}",
        msg
    );
}

#[test]
fn test_error_clustering_has_descriptive_message() {
    let err = PurposeIndexError::clustering("insufficient data points");
    let msg = err.to_string();

    assert!(
        msg.contains("insufficient data points"),
        "Error should contain reason"
    );

    println!("[VERIFIED] ClusteringError contains reason: {}", msg);
}

#[test]
fn test_error_hnsw_wrapping() {
    let index_err = IndexError::NotFound {
        memory_id: Uuid::new_v4(),
    };
    let purpose_err: PurposeIndexError = index_err.into();
    let msg = purpose_err.to_string();

    assert!(msg.contains("HNSW"), "Error should indicate HNSW source");

    println!("[VERIFIED] HnswError wraps IndexError correctly: {}", msg);
}

#[test]
fn test_error_persistence_has_descriptive_message() {
    let err = PurposeIndexError::persistence("saving index", "disk full");
    let msg = err.to_string();

    assert!(msg.contains("saving index"), "Error should contain context");
    assert!(msg.contains("disk full"), "Error should contain message");

    println!(
        "[VERIFIED] PersistenceError contains context and message: {}",
        msg
    );
}

#[test]
fn test_error_propagation_fail_fast() {
    // Test that errors propagate properly through the result type
    fn inner_operation() -> PurposeIndexResult<()> {
        Err(PurposeIndexError::not_found(Uuid::new_v4()))
    }

    fn outer_operation() -> PurposeIndexResult<String> {
        inner_operation()?; // Should propagate
        Ok("success".to_string())
    }

    let result = outer_operation();
    assert!(result.is_err());

    println!("[VERIFIED] Errors propagate through Result (fail-fast semantics)");
}
