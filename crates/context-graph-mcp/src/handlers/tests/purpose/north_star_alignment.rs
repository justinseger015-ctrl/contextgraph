//! purpose/north_star_alignment Tests - TASK-CORE-001: DEPRECATED
//!
//! NOTE: purpose/north_star_alignment was removed per ARCH-03 (autonomous-first).
//! All calls now return METHOD_NOT_FOUND (-32601).
//! Use auto_bootstrap_north_star tool for autonomous goal discovery instead.

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::super::{create_test_handlers, create_test_handlers_no_north_star, make_request};

/// TASK-CORE-001: Test purpose/north_star_alignment returns METHOD_NOT_FOUND.
#[tokio::test]
async fn test_north_star_alignment_valid_fingerprint() {
    let handlers = create_test_handlers();

    // Store content first
    let store_params = json!({
        "content": "Building the best ML learning system for education",
        "importance": 0.9
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    let store_response = handlers.dispatch(store_request).await;
    let fingerprint_id = store_response
        .result
        .unwrap()
        .get("fingerprintId")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    // Check alignment - should return METHOD_NOT_FOUND (deprecated)
    let align_params = json!({
        "fingerprint_id": fingerprint_id,
        "include_breakdown": true,
        "include_patterns": true
    });
    let align_request = make_request(
        "purpose/north_star_alignment",
        Some(JsonRpcId::Number(2)),
        Some(align_params),
    );
    let response = handlers.dispatch(align_request).await;

    // TASK-CORE-001: Deprecated method returns METHOD_NOT_FOUND
    assert!(
        response.error.is_some(),
        "purpose/north_star_alignment must return error (deprecated per TASK-CORE-001)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601), got {}",
        error.code
    );
}

/// TASK-CORE-001: Test purpose/north_star_alignment returns METHOD_NOT_FOUND for missing params.
#[tokio::test]
async fn test_north_star_alignment_missing_id_fails() {
    let handlers = create_test_handlers();

    let align_params = json!({
        "include_breakdown": true
    });
    let align_request = make_request(
        "purpose/north_star_alignment",
        Some(JsonRpcId::Number(1)),
        Some(align_params),
    );
    let response = handlers.dispatch(align_request).await;

    // TASK-CORE-001: Deprecated method returns METHOD_NOT_FOUND (not INVALID_PARAMS)
    assert!(
        response.error.is_some(),
        "purpose/north_star_alignment must return error (deprecated per TASK-CORE-001)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601), got {}",
        error.code
    );
}

/// TASK-CORE-001: Test purpose/north_star_alignment returns METHOD_NOT_FOUND for invalid UUID.
#[tokio::test]
async fn test_north_star_alignment_invalid_uuid_fails() {
    let handlers = create_test_handlers();

    let align_params = json!({
        "fingerprint_id": "not-a-valid-uuid"
    });
    let align_request = make_request(
        "purpose/north_star_alignment",
        Some(JsonRpcId::Number(1)),
        Some(align_params),
    );
    let response = handlers.dispatch(align_request).await;

    // TASK-CORE-001: Deprecated method returns METHOD_NOT_FOUND (not INVALID_PARAMS)
    assert!(
        response.error.is_some(),
        "purpose/north_star_alignment must return error (deprecated per TASK-CORE-001)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601), got {}",
        error.code
    );
}

/// TASK-CORE-001: Test purpose/north_star_alignment returns METHOD_NOT_FOUND for non-existent ID.
#[tokio::test]
async fn test_north_star_alignment_not_found_fails() {
    let handlers = create_test_handlers();

    let align_params = json!({
        "fingerprint_id": "00000000-0000-0000-0000-000000000000"
    });
    let align_request = make_request(
        "purpose/north_star_alignment",
        Some(JsonRpcId::Number(1)),
        Some(align_params),
    );
    let response = handlers.dispatch(align_request).await;

    // TASK-CORE-001: Deprecated method returns METHOD_NOT_FOUND (not FINGERPRINT_NOT_FOUND)
    assert!(
        response.error.is_some(),
        "purpose/north_star_alignment must return error (deprecated per TASK-CORE-001)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601), got {}",
        error.code
    );
}

/// Test autonomous operation: store succeeds without North Star.
///
/// AUTONOMOUS OPERATION: Per contextprd.md, memory storage works without North Star
/// by using a default purpose vector [0.0; 13]. The 13-embedding array IS the
/// teleological vector - purpose alignment is secondary metadata.
///
/// TASK-CORE-001: purpose/north_star_alignment is deprecated, returns METHOD_NOT_FOUND.
#[tokio::test]
async fn test_north_star_alignment_autonomous_operation() {
    let handlers = create_test_handlers_no_north_star();

    // Store content - should SUCCEED without North Star (AUTONOMOUS OPERATION)
    let store_params = json!({
        "content": "Test content for autonomous operation",
        "importance": 0.8
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    let store_response = handlers.dispatch(store_request).await;

    // Store should succeed with default purpose vector
    assert!(
        store_response.error.is_none(),
        "memory/store must succeed without North Star (AUTONOMOUS OPERATION). Error: {:?}",
        store_response.error
    );
    let result = store_response.result.expect("Should have result");
    assert!(
        result.get("fingerprintId").is_some(),
        "Must return fingerprintId"
    );

    // TASK-CORE-001: purpose/north_star_alignment is deprecated
    let align_params = json!({
        "fingerprint_id": "00000000-0000-0000-0000-000000000001"
    });
    let align_request = make_request(
        "purpose/north_star_alignment",
        Some(JsonRpcId::Number(2)),
        Some(align_params),
    );
    let response = handlers.dispatch(align_request).await;

    // Should fail with METHOD_NOT_FOUND (deprecated per TASK-CORE-001)
    assert!(
        response.error.is_some(),
        "purpose/north_star_alignment must return error (deprecated per TASK-CORE-001)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601), got {}",
        error.code
    );
}
