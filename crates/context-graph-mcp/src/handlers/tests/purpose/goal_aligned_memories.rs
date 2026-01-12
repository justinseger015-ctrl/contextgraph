//! goal/aligned_memories Tests
//!
//! Tests for the goal/aligned_memories MCP endpoint.
//! Finds memories aligned to specific goals in the hierarchy.

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::super::{create_test_handlers, make_request};
use super::helpers::get_goal_ids_from_hierarchy;

/// Test goal/aligned_memories with valid goal.
/// TASK-CORE-005: Updated to use UUID-based goal IDs instead of hardcoded strings.
#[tokio::test]
async fn test_goal_aligned_memories_valid() {
    let handlers = create_test_handlers();

    // First, get the actual strategic goal ID from the hierarchy
    let (_, strategic_ids, _, _) = get_goal_ids_from_hierarchy(&handlers).await;
    assert!(
        !strategic_ids.is_empty(),
        "Should have at least one Strategic goal"
    );
    let strategic_id = &strategic_ids[0];

    // Store content
    let store_params = json!({
        "content": "Improving retrieval accuracy through semantic search",
        "importance": 0.9
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    // Find memories aligned to strategic goal
    let aligned_params = json!({
        "goal_id": strategic_id,
        "topK": 10,
        "minAlignment": 0.0  // P1-FIX-1: Required parameter for fail-fast
    });
    let aligned_request = make_request(
        "goal/aligned_memories",
        Some(JsonRpcId::Number(2)),
        Some(aligned_params),
    );
    let response = handlers.dispatch(aligned_request).await;

    assert!(
        response.error.is_none(),
        "goal/aligned_memories should succeed"
    );
    let result = response.result.expect("Should have result");

    let goal = result.get("goal").expect("Should have goal");
    assert_eq!(
        goal.get("id").and_then(|v| v.as_str()),
        Some(strategic_id.as_str()),
        "Should return correct goal"
    );

    assert!(result.get("results").is_some(), "Should have results array");
    assert!(result.get("count").is_some(), "Should have count");
    assert!(
        result.get("search_time_ms").is_some(),
        "Should report search_time_ms"
    );
}

/// Test goal/aligned_memories fails with missing goal_id.
#[tokio::test]
async fn test_goal_aligned_memories_missing_id_fails() {
    let handlers = create_test_handlers();

    let aligned_params = json!({
        "topK": 10
    });
    let aligned_request = make_request(
        "goal/aligned_memories",
        Some(JsonRpcId::Number(1)),
        Some(aligned_params),
    );
    let response = handlers.dispatch(aligned_request).await;

    assert!(
        response.error.is_some(),
        "goal/aligned_memories must fail without goal_id"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
    assert!(
        error.message.contains("goal_id"),
        "Error should mention missing goal_id"
    );
}

/// Test goal/aligned_memories fails with non-existent goal.
/// TASK-CORE-005: Updated to use valid UUID format for non-existent goal.
#[tokio::test]
async fn test_goal_aligned_memories_goal_not_found_fails() {
    let handlers = create_test_handlers();

    // Use a valid UUID format that doesn't exist in the hierarchy
    let aligned_params = json!({
        "goal_id": "00000000-0000-0000-0000-000000000000",
        "minAlignment": 0.0  // P1-FIX-1: Required parameter (test expects goal not found error)
    });
    let aligned_request = make_request(
        "goal/aligned_memories",
        Some(JsonRpcId::Number(1)),
        Some(aligned_params),
    );
    let response = handlers.dispatch(aligned_request).await;

    assert!(
        response.error.is_some(),
        "goal/aligned_memories must fail with non-existent goal"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32020,
        "Should return GOAL_NOT_FOUND error code"
    );
}
