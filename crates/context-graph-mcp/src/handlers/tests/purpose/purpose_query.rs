//! purpose/query Tests
//!
//! Tests for the purpose/query MCP endpoint with 13D purpose vector similarity.

use serde_json::json;

use crate::protocol::JsonRpcId;
use context_graph_core::types::fingerprint::NUM_EMBEDDERS;

use super::super::{create_test_handlers, make_request};

/// Test purpose/query with valid 13-element purpose vector.
#[tokio::test]
async fn test_purpose_query_valid_vector() {
    let handlers = create_test_handlers();

    // First store some content to search
    let store_params = json!({
        "content": "Machine learning enables computers to learn from data",
        "importance": 0.9
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    // Query with 13D purpose vector
    let purpose_vector: Vec<f64> = vec![
        0.8, 0.5, 0.3, 0.3, 0.6, 0.2, 0.7, 0.5, 0.4, 0.3, 0.5, 0.3, 0.2,
    ];

    let query_params = json!({
        "purpose_vector": purpose_vector,
        "topK": 10
    });
    let query_request = make_request(
        "purpose/query",
        Some(JsonRpcId::Number(2)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    assert!(response.error.is_none(), "purpose/query should succeed");
    let result = response.result.expect("Should have result");

    // Verify response structure
    assert!(result.get("results").is_some(), "Should have results array");
    assert!(result.get("count").is_some(), "Should have count");
    assert!(
        result.get("query_metadata").is_some(),
        "Should have query_metadata"
    );

    // Verify query_metadata structure
    let metadata = result.get("query_metadata").unwrap();
    assert!(
        metadata.get("purpose_vector_used").is_some(),
        "Should include purpose_vector_used"
    );
    assert!(
        metadata.get("dominant_embedder").is_some(),
        "Should include dominant_embedder"
    );
    assert!(
        metadata.get("query_coherence").is_some(),
        "Should include query_coherence"
    );
    assert!(
        metadata.get("search_time_ms").is_some(),
        "Should report search_time_ms"
    );

    // Verify purpose_vector_used has 13 elements
    let pv_used = metadata
        .get("purpose_vector_used")
        .and_then(|v| v.as_array());
    assert!(pv_used.is_some(), "purpose_vector_used must be array");
    assert_eq!(
        pv_used.unwrap().len(),
        NUM_EMBEDDERS,
        "purpose_vector_used must have 13 elements"
    );
}

/// Test purpose/query fails with missing purpose_vector.
#[tokio::test]
async fn test_purpose_query_missing_vector_fails() {
    let handlers = create_test_handlers();

    let query_params = json!({
        "topK": 10
    });
    let query_request = make_request(
        "purpose/query",
        Some(JsonRpcId::Number(1)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    assert!(
        response.error.is_some(),
        "purpose/query must fail without purpose_vector"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
    assert!(
        error.message.contains("purpose_vector"),
        "Error should mention missing purpose_vector"
    );
}

/// Test purpose/query fails with 12-element purpose vector (must be 13).
#[tokio::test]
async fn test_purpose_query_wrong_vector_size_fails() {
    let handlers = create_test_handlers();

    // Only 12 elements (WRONG - must be 13!)
    let invalid_vector: Vec<f64> = vec![0.5; 12];

    let query_params = json!({
        "purpose_vector": invalid_vector
    });
    let query_request = make_request(
        "purpose/query",
        Some(JsonRpcId::Number(1)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    assert!(
        response.error.is_some(),
        "purpose/query must fail with 12-element vector"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
    assert!(
        error.message.contains("13") || error.message.contains("elements"),
        "Error should mention 13 elements required"
    );
}

/// Test purpose/query fails with out-of-range values.
#[tokio::test]
async fn test_purpose_query_out_of_range_values_fails() {
    let handlers = create_test_handlers();

    // Value > 1.0 is invalid
    let mut invalid_vector: Vec<f64> = vec![0.5; 13];
    invalid_vector[5] = 1.5; // Out of range!

    let query_params = json!({
        "purpose_vector": invalid_vector
    });
    let query_request = make_request(
        "purpose/query",
        Some(JsonRpcId::Number(1)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    assert!(
        response.error.is_some(),
        "purpose/query must fail with out-of-range value"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
    assert!(
        error.message.contains("range")
            || error.message.contains("0.0")
            || error.message.contains("1.0"),
        "Error should mention valid range [0.0, 1.0]"
    );
}

/// Test purpose/query with min_alignment filter.
#[tokio::test]
async fn test_purpose_query_min_alignment_filter() {
    let handlers = create_test_handlers();

    // Store content
    let store_params = json!({
        "content": "Purpose alignment testing content",
        "importance": 0.8
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    let purpose_vector: Vec<f64> = vec![0.7; 13];

    let query_params = json!({
        "purpose_vector": purpose_vector,
        "minAlignment": 0.5,
        "topK": 5
    });
    let query_request = make_request(
        "purpose/query",
        Some(JsonRpcId::Number(2)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    assert!(response.error.is_none(), "purpose/query should succeed");
    let result = response.result.expect("Should have result");

    // Verify min_alignment_filter is reported
    let metadata = result.get("query_metadata").unwrap();
    let min_filter = metadata
        .get("min_alignment_filter")
        .and_then(|v| v.as_f64());
    assert_eq!(min_filter, Some(0.5), "Should report min_alignment_filter");
}
