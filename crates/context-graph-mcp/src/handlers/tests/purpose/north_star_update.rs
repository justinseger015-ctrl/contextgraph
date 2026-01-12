//! purpose/north_star_update Tests (DEPRECATED - TASK-CORE-001)
//!
//! NOTE: purpose/north_star_update is REMOVED per ARCH-03 (autonomous-first).
//! All tests now verify METHOD_NOT_FOUND (-32601) is returned.
//! Use auto_bootstrap_north_star for autonomous goal discovery instead.

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::super::{create_test_handlers, create_test_handlers_no_north_star, make_request};

/// TASK-CORE-001: Test purpose/north_star_update returns METHOD_NOT_FOUND.
/// Previously tested creating new North Star, now verifies deprecation.
#[tokio::test]
async fn test_north_star_update_create_returns_method_not_found() {
    let handlers = create_test_handlers_no_north_star();

    let update_params = json!({
        "description": "Build the best AI assistant for developers",
        "keywords": ["ai", "assistant", "developer"]
    });
    let update_request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(update_params),
    );
    let response = handlers.dispatch(update_request).await;

    // TASK-CORE-001: Must return METHOD_NOT_FOUND for deprecated method
    assert!(
        response.error.is_some(),
        "purpose/north_star_update must return error (deprecated per ARCH-03)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601) for deprecated method"
    );
}

/// TASK-CORE-001: Test purpose/north_star_update with replace=false returns METHOD_NOT_FOUND.
#[tokio::test]
async fn test_north_star_update_exists_no_replace_returns_method_not_found() {
    let handlers = create_test_handlers();

    let update_params = json!({
        "description": "New North Star goal",
        "replace": false
    });
    let update_request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(update_params),
    );
    let response = handlers.dispatch(update_request).await;

    // TASK-CORE-001: Must return METHOD_NOT_FOUND for deprecated method
    assert!(
        response.error.is_some(),
        "purpose/north_star_update must return error (deprecated per ARCH-03)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601) for deprecated method"
    );
}

/// TASK-CORE-001: Test purpose/north_star_update with replace=true returns METHOD_NOT_FOUND.
#[tokio::test]
async fn test_north_star_update_replace_returns_method_not_found() {
    let handlers = create_test_handlers();

    let update_params = json!({
        "description": "New improved North Star goal",
        "keywords": ["improved", "goal"],
        "replace": true
    });
    let update_request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(update_params),
    );
    let response = handlers.dispatch(update_request).await;

    // TASK-CORE-001: Must return METHOD_NOT_FOUND for deprecated method
    assert!(
        response.error.is_some(),
        "purpose/north_star_update must return error (deprecated per ARCH-03)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601) for deprecated method"
    );
}

/// TASK-CORE-001: Test purpose/north_star_update without description returns METHOD_NOT_FOUND.
/// Note: METHOD_NOT_FOUND takes precedence over INVALID_PARAMS since method is removed.
#[tokio::test]
async fn test_north_star_update_missing_description_returns_method_not_found() {
    let handlers = create_test_handlers_no_north_star();

    let update_params = json!({
        "keywords": ["test"]
    });
    let update_request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(update_params),
    );
    let response = handlers.dispatch(update_request).await;

    // TASK-CORE-001: Must return METHOD_NOT_FOUND for deprecated method
    assert!(
        response.error.is_some(),
        "purpose/north_star_update must return error (deprecated per ARCH-03)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601) for deprecated method"
    );
}

/// TASK-CORE-001: Test purpose/north_star_update with empty description returns METHOD_NOT_FOUND.
#[tokio::test]
async fn test_north_star_update_empty_description_returns_method_not_found() {
    let handlers = create_test_handlers_no_north_star();

    let update_params = json!({
        "description": ""
    });
    let update_request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(update_params),
    );
    let response = handlers.dispatch(update_request).await;

    // TASK-CORE-001: Must return METHOD_NOT_FOUND for deprecated method
    assert!(
        response.error.is_some(),
        "purpose/north_star_update must return error (deprecated per ARCH-03)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601) for deprecated method"
    );
}

/// TASK-CORE-001: Test purpose/north_star_update with embedding returns METHOD_NOT_FOUND.
/// Note: METHOD_NOT_FOUND takes precedence over INVALID_PARAMS since method is removed.
#[tokio::test]
async fn test_north_star_update_with_embedding_returns_method_not_found() {
    let handlers = create_test_handlers_no_north_star();

    // 768 dimensions instead of 1024 - doesn't matter since method is removed
    let wrong_embedding: Vec<f64> = vec![0.5; 768];

    let update_params = json!({
        "description": "Test North Star",
        "embedding": wrong_embedding
    });
    let update_request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(update_params),
    );
    let response = handlers.dispatch(update_request).await;

    // TASK-CORE-001: Must return METHOD_NOT_FOUND for deprecated method
    assert!(
        response.error.is_some(),
        "purpose/north_star_update must return error (deprecated per ARCH-03)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601) for deprecated method"
    );
}
