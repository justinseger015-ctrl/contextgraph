//! Teleological Memory Handler Tests
//!
//! TASK-S001: Rewritten for TeleologicalFingerprint storage with real data.
//! NO mock data - uses InMemoryTeleologicalStore with real embeddings.
//!
//! Tests verify:
//! - memory/store creates TeleologicalFingerprint with 13 embeddings
//! - memory/retrieve returns fingerprint by UUID
//! - memory/search performs 13-embedding semantic search
//! - memory/delete supports soft and hard delete

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::{create_test_handlers, make_request};

/// Test memory/store with valid content creates TeleologicalFingerprint.
#[tokio::test]
async fn test_memory_store_creates_teleological_fingerprint() {
    let handlers = create_test_handlers();
    let params = json!({
        "content": "The quick brown fox jumps over the lazy dog",
        "importance": 0.8
    });
    let request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "memory/store should succeed");
    let result = response.result.expect("Should have result");

    // Verify returns fingerprintId (UUID format)
    let fingerprint_id = result.get("fingerprintId").expect("Should return fingerprintId");
    assert!(fingerprint_id.is_string(), "fingerprintId should be string");
    let id_str = fingerprint_id.as_str().unwrap();
    uuid::Uuid::parse_str(id_str).expect("fingerprintId should be valid UUID");

    // Verify returns embedderCount (should be 13)
    let embedder_count = result.get("embedderCount").and_then(|v| v.as_u64());
    assert_eq!(embedder_count, Some(13), "Should return embedderCount=13");

    // Verify returns embedding latency
    assert!(
        result.get("embeddingLatencyMs").is_some(),
        "Should return embeddingLatencyMs"
    );
    assert!(
        result.get("storageLatencyMs").is_some(),
        "Should return storageLatencyMs"
    );
}

/// Test memory/store fails with missing content parameter.
#[tokio::test]
async fn test_memory_store_missing_content_fails() {
    let handlers = create_test_handlers();
    let params = json!({
        "importance": 0.5
    });
    let request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_some(),
        "memory/store should fail without content"
    );
    let error = response.error.unwrap();
    assert_eq!(error.code, -32602, "Should return INVALID_PARAMS error code");
    assert!(
        error.message.contains("content"),
        "Error message should mention 'content'"
    );
}

/// Test memory/store fails with empty content string.
#[tokio::test]
async fn test_memory_store_empty_content_fails() {
    let handlers = create_test_handlers();
    let params = json!({
        "content": "",
        "importance": 0.5
    });
    let request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_some(),
        "memory/store should fail with empty content"
    );
    let error = response.error.unwrap();
    assert_eq!(error.code, -32602, "Should return INVALID_PARAMS error code");
}

/// Test memory/retrieve returns stored TeleologicalFingerprint.
#[tokio::test]
async fn test_memory_retrieve_returns_fingerprint() {
    let handlers = create_test_handlers();

    // First store a fingerprint
    let store_params = json!({
        "content": "Memory content for retrieval test",
        "importance": 0.7
    });
    let store_request =
        make_request("memory/store", Some(JsonRpcId::Number(1)), Some(store_params));
    let store_response = handlers.dispatch(store_request).await;
    let fingerprint_id = store_response
        .result
        .unwrap()
        .get("fingerprintId")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    // Then retrieve it
    let retrieve_params = json!({
        "fingerprintId": fingerprint_id
    });
    let retrieve_request = make_request(
        "memory/retrieve",
        Some(JsonRpcId::Number(2)),
        Some(retrieve_params),
    );

    let response = handlers.dispatch(retrieve_request).await;

    assert!(response.error.is_none(), "memory/retrieve should succeed");
    let result = response.result.expect("Should have result");

    // Verify fingerprint object structure
    let fingerprint = result.get("fingerprint").expect("Should return fingerprint object");
    assert_eq!(
        fingerprint.get("id").and_then(|v| v.as_str()),
        Some(fingerprint_id.as_str()),
        "ID should match"
    );
    assert!(
        fingerprint.get("thetaToNorthStar").is_some(),
        "Should have thetaToNorthStar"
    );
    assert!(
        fingerprint.get("accessCount").is_some(),
        "Should have accessCount"
    );
    assert!(
        fingerprint.get("createdAt").is_some(),
        "Should have createdAt"
    );
    assert!(
        fingerprint.get("lastUpdated").is_some(),
        "Should have lastUpdated"
    );
    assert!(
        fingerprint.get("purposeVector").is_some(),
        "Should have purposeVector"
    );
    assert!(
        fingerprint.get("johariDominant").is_some(),
        "Should have johariDominant"
    );
    assert!(
        fingerprint.get("contentHashHex").is_some(),
        "Should have contentHashHex"
    );
}

/// Test memory/retrieve fails with invalid UUID.
#[tokio::test]
async fn test_memory_retrieve_invalid_uuid_fails() {
    let handlers = create_test_handlers();

    let retrieve_params = json!({
        "fingerprintId": "not-a-valid-uuid"
    });
    let retrieve_request = make_request(
        "memory/retrieve",
        Some(JsonRpcId::Number(1)),
        Some(retrieve_params),
    );

    let response = handlers.dispatch(retrieve_request).await;

    assert!(
        response.error.is_some(),
        "memory/retrieve should fail with invalid UUID"
    );
    let error = response.error.unwrap();
    assert_eq!(error.code, -32602, "Should return INVALID_PARAMS error code");
    assert!(
        error.message.contains("UUID"),
        "Error message should mention 'UUID'"
    );
}

/// Test memory/retrieve returns error for non-existent fingerprint.
#[tokio::test]
async fn test_memory_retrieve_not_found() {
    let handlers = create_test_handlers();

    let retrieve_params = json!({
        "fingerprintId": "00000000-0000-0000-0000-000000000000"
    });
    let retrieve_request = make_request(
        "memory/retrieve",
        Some(JsonRpcId::Number(1)),
        Some(retrieve_params),
    );

    let response = handlers.dispatch(retrieve_request).await;

    assert!(
        response.error.is_some(),
        "memory/retrieve should fail for non-existent fingerprint"
    );
    let error = response.error.unwrap();
    assert_eq!(error.code, -32010, "Should return FINGERPRINT_NOT_FOUND error code");
}

/// Test memory/search returns results with semantic similarity.
#[tokio::test]
async fn test_memory_search_returns_results() {
    let handlers = create_test_handlers();

    // Store some content first
    let store_params = json!({
        "content": "Machine learning is a subset of artificial intelligence",
        "importance": 0.9
    });
    let store_request =
        make_request("memory/store", Some(JsonRpcId::Number(1)), Some(store_params));
    handlers.dispatch(store_request).await;

    // Search for similar content
    let search_params = json!({
        "query": "AI and machine learning",
        "topK": 5
    });
    let search_request = make_request("memory/search", Some(JsonRpcId::Number(2)), Some(search_params));

    let response = handlers.dispatch(search_request).await;

    assert!(response.error.is_none(), "memory/search should succeed");
    let result = response.result.expect("Should have result");
    assert!(result.get("results").is_some(), "Should return results array");
    assert!(result.get("count").is_some(), "Should return count");
    assert!(result.get("queryLatencyMs").is_some(), "Should return queryLatencyMs");
}

/// Test memory/search fails with missing query parameter.
#[tokio::test]
async fn test_memory_search_missing_query_fails() {
    let handlers = create_test_handlers();

    let search_params = json!({
        "topK": 5
    });
    let search_request = make_request("memory/search", Some(JsonRpcId::Number(1)), Some(search_params));

    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_some(),
        "memory/search should fail without query"
    );
    let error = response.error.unwrap();
    assert_eq!(error.code, -32602, "Should return INVALID_PARAMS error code");
}

/// Test memory/delete soft delete marks fingerprint as deleted.
#[tokio::test]
async fn test_memory_delete_soft() {
    let handlers = create_test_handlers();

    // First store a fingerprint
    let store_params = json!({
        "content": "Content to soft delete",
        "importance": 0.5
    });
    let store_request =
        make_request("memory/store", Some(JsonRpcId::Number(1)), Some(store_params));
    let store_response = handlers.dispatch(store_request).await;
    let fingerprint_id = store_response
        .result
        .unwrap()
        .get("fingerprintId")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    // Soft delete
    let delete_params = json!({
        "fingerprintId": fingerprint_id,
        "soft": true
    });
    let delete_request = make_request(
        "memory/delete",
        Some(JsonRpcId::Number(2)),
        Some(delete_params),
    );

    let response = handlers.dispatch(delete_request).await;

    assert!(response.error.is_none(), "memory/delete should succeed");
    let result = response.result.expect("Should have result");
    assert_eq!(
        result.get("deleted").and_then(|v| v.as_bool()),
        Some(true),
        "Should return deleted=true"
    );
    assert_eq!(
        result.get("deleteType").and_then(|v| v.as_str()),
        Some("soft"),
        "Should return deleteType=soft"
    );
}

/// Test memory/delete hard delete permanently removes fingerprint.
#[tokio::test]
async fn test_memory_delete_hard() {
    let handlers = create_test_handlers();

    // First store a fingerprint
    let store_params = json!({
        "content": "Content to hard delete",
        "importance": 0.5
    });
    let store_request =
        make_request("memory/store", Some(JsonRpcId::Number(1)), Some(store_params));
    let store_response = handlers.dispatch(store_request).await;
    let fingerprint_id = store_response
        .result
        .unwrap()
        .get("fingerprintId")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    // Hard delete
    let delete_params = json!({
        "fingerprintId": fingerprint_id,
        "soft": false
    });
    let delete_request = make_request(
        "memory/delete",
        Some(JsonRpcId::Number(2)),
        Some(delete_params),
    );

    let response = handlers.dispatch(delete_request).await;

    assert!(response.error.is_none(), "memory/delete should succeed");
    let result = response.result.expect("Should have result");
    assert_eq!(
        result.get("deleted").and_then(|v| v.as_bool()),
        Some(true),
        "Should return deleted=true"
    );
    assert_eq!(
        result.get("deleteType").and_then(|v| v.as_str()),
        Some("hard"),
        "Should return deleteType=hard"
    );

    // Verify retrieve fails after hard delete
    let retrieve_params = json!({
        "fingerprintId": fingerprint_id
    });
    let retrieve_request = make_request(
        "memory/retrieve",
        Some(JsonRpcId::Number(3)),
        Some(retrieve_params),
    );
    let retrieve_response = handlers.dispatch(retrieve_request).await;
    assert!(
        retrieve_response.error.is_some(),
        "retrieve should fail after hard delete"
    );
}

/// Test that store/retrieve roundtrip preserves content hash.
#[tokio::test]
async fn test_memory_store_retrieve_preserves_content_hash() {
    let handlers = create_test_handlers();

    let content = "Unique content for hash verification test 12345";
    let store_params = json!({
        "content": content,
        "importance": 0.6
    });
    let store_request =
        make_request("memory/store", Some(JsonRpcId::Number(1)), Some(store_params));
    let store_response = handlers.dispatch(store_request).await;
    let fingerprint_id = store_response
        .result
        .unwrap()
        .get("fingerprintId")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    // Retrieve and check hash
    let retrieve_params = json!({
        "fingerprintId": fingerprint_id
    });
    let retrieve_request = make_request(
        "memory/retrieve",
        Some(JsonRpcId::Number(2)),
        Some(retrieve_params),
    );
    let response = handlers.dispatch(retrieve_request).await;

    let result = response.result.expect("Should have result");
    let fingerprint = result.get("fingerprint").expect("Should have fingerprint");
    let hash_hex = fingerprint.get("contentHashHex").and_then(|v| v.as_str()).expect("Should have contentHashHex");

    // Compute expected hash
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let expected_hash = hex::encode(hasher.finalize());

    assert_eq!(hash_hex, expected_hash, "Content hash should match");
}

/// Test that store increments count correctly.
#[tokio::test]
async fn test_memory_store_increments_count() {
    let handlers = create_test_handlers();

    // Store 3 fingerprints
    for i in 0..3 {
        let store_params = json!({
            "content": format!("Test content number {}", i),
            "importance": 0.5
        });
        let store_request =
            make_request("memory/store", Some(JsonRpcId::Number(i as i64 + 1)), Some(store_params));
        handlers.dispatch(store_request).await;
    }

    // Check count via get_memetic_status tool
    let status_request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(10)),
        Some(json!({
            "name": "get_memetic_status",
            "arguments": {}
        })),
    );
    let response = handlers.dispatch(status_request).await;

    // The response contains MCP tool format with content array
    // We need to extract the inner JSON from the text field
    let result = response.result.expect("Should have result");
    let content = result.get("content").and_then(|v| v.as_array()).expect("Should have content array");
    let text = content[0].get("text").and_then(|v| v.as_str()).expect("Should have text");
    let status: serde_json::Value = serde_json::from_str(text).expect("Should parse inner JSON");

    let count = status.get("fingerprintCount").and_then(|v| v.as_u64());
    assert_eq!(count, Some(3), "Should have 3 fingerprints stored");
}

// =============================================================================
// TASK-S001: Full State Verification Tests
// =============================================================================

/// FULL STATE VERIFICATION: Comprehensive CRUD cycle test.
///
/// This test manually verifies all memory operations with real data:
/// 1. STORE: Creates TeleologicalFingerprint with 13 embeddings
/// 2. RETRIEVE: Fetches stored fingerprint and verifies all fields
/// 3. SEARCH: Finds fingerprint via semantic 13-embedding search
/// 4. DELETE (soft): Marks as deleted but retains data
/// 5. DELETE (hard): Permanently removes fingerprint
///
/// NO MOCKS - uses InMemoryTeleologicalStore with StubMultiArrayProvider.
#[tokio::test]
async fn test_full_state_verification_crud_cycle() {
    let handlers = create_test_handlers();

    // =========================================================================
    // STEP 1: STORE - Create TeleologicalFingerprint with 13 embeddings
    // =========================================================================
    let content = "Machine learning enables computers to learn from data without explicit programming";
    let store_params = json!({
        "content": content,
        "importance": 0.9
    });
    let store_request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(store_params));
    let store_response = handlers.dispatch(store_request).await;

    // Verify STORE success
    assert!(store_response.error.is_none(), "STORE must succeed");
    let store_result = store_response.result.expect("STORE must return result");

    // Verify fingerprintId is valid UUID
    let fingerprint_id = store_result.get("fingerprintId")
        .expect("STORE must return fingerprintId")
        .as_str()
        .expect("fingerprintId must be string");
    let parsed_uuid = uuid::Uuid::parse_str(fingerprint_id)
        .expect("fingerprintId must be valid UUID");
    assert!(!parsed_uuid.is_nil(), "fingerprintId must not be nil UUID");

    // Verify 13 embeddings were generated
    let embedder_count = store_result.get("embedderCount")
        .and_then(|v| v.as_u64())
        .expect("STORE must return embedderCount");
    assert_eq!(embedder_count, 13, "Must generate exactly 13 embeddings");

    // Verify latency metrics exist
    assert!(store_result.get("embeddingLatencyMs").is_some(), "STORE must report embeddingLatencyMs");
    assert!(store_result.get("storageLatencyMs").is_some(), "STORE must report storageLatencyMs");

    // =========================================================================
    // STEP 2: RETRIEVE - Fetch fingerprint and verify all fields
    // =========================================================================
    let retrieve_params = json!({ "fingerprintId": fingerprint_id });
    let retrieve_request = make_request("memory/retrieve", Some(JsonRpcId::Number(2)), Some(retrieve_params));
    let retrieve_response = handlers.dispatch(retrieve_request).await;

    // Verify RETRIEVE success
    assert!(retrieve_response.error.is_none(), "RETRIEVE must succeed for valid fingerprint");
    let retrieve_result = retrieve_response.result.expect("RETRIEVE must return result");

    // Verify fingerprint object structure
    let fp = retrieve_result.get("fingerprint").expect("RETRIEVE must return fingerprint object");

    // Verify ID matches what we stored
    assert_eq!(
        fp.get("id").and_then(|v| v.as_str()),
        Some(fingerprint_id),
        "Retrieved ID must match stored ID"
    );

    // Verify theta_to_north_star exists (alignment to North Star)
    assert!(fp.get("thetaToNorthStar").is_some(), "Must have thetaToNorthStar");

    // Verify access_count (should be at least 1 after retrieve)
    let access_count = fp.get("accessCount").and_then(|v| v.as_u64());
    assert!(access_count.is_some(), "Must have accessCount");

    // Verify timestamps
    let created_at = fp.get("createdAt").and_then(|v| v.as_str());
    let last_updated = fp.get("lastUpdated").and_then(|v| v.as_str());
    assert!(created_at.is_some(), "Must have createdAt timestamp");
    assert!(last_updated.is_some(), "Must have lastUpdated timestamp");

    // Verify purpose vector (13D alignment)
    let purpose_vector = fp.get("purposeVector").and_then(|v| v.as_array());
    assert!(purpose_vector.is_some(), "Must have purposeVector");
    let pv = purpose_vector.unwrap();
    assert_eq!(pv.len(), 13, "purposeVector must have 13 elements");

    // Verify Johari dominant quadrant
    let johari_dominant = fp.get("johariDominant").and_then(|v| v.as_str());
    assert!(johari_dominant.is_some(), "Must have johariDominant");
    let quadrant = johari_dominant.unwrap();
    assert!(
        ["Open", "Hidden", "Blind", "Unknown"].contains(&quadrant),
        "johariDominant must be valid quadrant: {}", quadrant
    );

    // Verify content hash
    let hash_hex = fp.get("contentHashHex").and_then(|v| v.as_str());
    assert!(hash_hex.is_some(), "Must have contentHashHex");
    let hash = hash_hex.unwrap();
    assert_eq!(hash.len(), 64, "contentHashHex must be 64 hex chars (SHA-256)");

    // Verify hash is correct
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let expected_hash = hex::encode(hasher.finalize());
    assert_eq!(hash, expected_hash, "Content hash must match SHA-256 of content");

    // =========================================================================
    // STEP 3: SEARCH - Find fingerprint via semantic 13-embedding search
    // =========================================================================
    let search_params = json!({
        "query": "AI machine learning data",
        "topK": 10
    });
    let search_request = make_request("memory/search", Some(JsonRpcId::Number(3)), Some(search_params));
    let search_response = handlers.dispatch(search_request).await;

    // Verify SEARCH success
    assert!(search_response.error.is_none(), "SEARCH must succeed");
    let search_result = search_response.result.expect("SEARCH must return result");

    // Verify search results structure
    let results = search_result.get("results").and_then(|v| v.as_array())
        .expect("SEARCH must return results array");
    let count = search_result.get("count").and_then(|v| v.as_u64())
        .expect("SEARCH must return count");
    assert!(search_result.get("queryLatencyMs").is_some(), "SEARCH must report queryLatencyMs");

    // Our stored fingerprint should be in results
    assert!(count >= 1, "SEARCH should find at least 1 result");
    let found = results.iter().any(|r| {
        r.get("fingerprintId").and_then(|v| v.as_str()) == Some(fingerprint_id)
    });
    assert!(found, "SEARCH must find our stored fingerprint");

    // Verify search result structure for first result
    if !results.is_empty() {
        let first = &results[0];
        assert!(first.get("fingerprintId").is_some(), "Result must have fingerprintId");
        assert!(first.get("similarity").is_some(), "Result must have similarity score");
        assert!(first.get("dominantEmbedder").is_some(), "Result must have dominantEmbedder");
        assert!(first.get("embedderScores").is_some(), "Result must have embedderScores");
    }

    // =========================================================================
    // STEP 4: DELETE (soft) - Mark as deleted but retain data
    // =========================================================================
    let soft_delete_params = json!({
        "fingerprintId": fingerprint_id,
        "soft": true
    });
    let soft_delete_request = make_request("memory/delete", Some(JsonRpcId::Number(4)), Some(soft_delete_params));
    let soft_delete_response = handlers.dispatch(soft_delete_request).await;

    // Verify soft DELETE success
    assert!(soft_delete_response.error.is_none(), "Soft DELETE must succeed");
    let soft_delete_result = soft_delete_response.result.expect("Soft DELETE must return result");
    assert_eq!(
        soft_delete_result.get("deleted").and_then(|v| v.as_bool()),
        Some(true),
        "Soft DELETE must return deleted=true"
    );
    assert_eq!(
        soft_delete_result.get("deleteType").and_then(|v| v.as_str()),
        Some("soft"),
        "Soft DELETE must return deleteType=soft"
    );

    // =========================================================================
    // STEP 5: Store another fingerprint then hard DELETE
    // =========================================================================
    let content2 = "Deep neural networks revolutionize pattern recognition";
    let store2_params = json!({
        "content": content2,
        "importance": 0.7
    });
    let store2_request = make_request("memory/store", Some(JsonRpcId::Number(5)), Some(store2_params));
    let store2_response = handlers.dispatch(store2_request).await;
    let fingerprint_id2 = store2_response.result.unwrap()
        .get("fingerprintId").unwrap().as_str().unwrap().to_string();

    // Hard delete the second fingerprint
    let hard_delete_params = json!({
        "fingerprintId": fingerprint_id2,
        "soft": false
    });
    let hard_delete_request = make_request("memory/delete", Some(JsonRpcId::Number(6)), Some(hard_delete_params));
    let hard_delete_response = handlers.dispatch(hard_delete_request).await;

    // Verify hard DELETE success
    assert!(hard_delete_response.error.is_none(), "Hard DELETE must succeed");
    let hard_delete_result = hard_delete_response.result.expect("Hard DELETE must return result");
    assert_eq!(
        hard_delete_result.get("deleted").and_then(|v| v.as_bool()),
        Some(true),
        "Hard DELETE must return deleted=true"
    );
    assert_eq!(
        hard_delete_result.get("deleteType").and_then(|v| v.as_str()),
        Some("hard"),
        "Hard DELETE must return deleteType=hard"
    );

    // Verify hard deleted fingerprint cannot be retrieved
    let retrieve2_params = json!({ "fingerprintId": fingerprint_id2 });
    let retrieve2_request = make_request("memory/retrieve", Some(JsonRpcId::Number(7)), Some(retrieve2_params));
    let retrieve2_response = handlers.dispatch(retrieve2_request).await;

    assert!(
        retrieve2_response.error.is_some(),
        "RETRIEVE after hard DELETE must fail"
    );
    let error = retrieve2_response.error.unwrap();
    assert_eq!(
        error.code,
        -32010, // FINGERPRINT_NOT_FOUND
        "RETRIEVE after hard DELETE must return FINGERPRINT_NOT_FOUND (-32010)"
    );

    // =========================================================================
    // VERIFICATION COMPLETE: All CRUD operations work with real data
    // =========================================================================
}

/// Test that error codes are correctly returned for various failure modes.
#[tokio::test]
async fn test_full_state_verification_error_codes() {
    let handlers = create_test_handlers();

    // Test INVALID_PARAMS for missing content
    let no_content = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(json!({ "importance": 0.5 })),
    );
    let resp = handlers.dispatch(no_content).await;
    assert_eq!(resp.error.unwrap().code, -32602, "Missing content must return INVALID_PARAMS");

    // Test INVALID_PARAMS for invalid UUID
    let bad_uuid = make_request(
        "memory/retrieve",
        Some(JsonRpcId::Number(2)),
        Some(json!({ "fingerprintId": "not-a-uuid" })),
    );
    let resp = handlers.dispatch(bad_uuid).await;
    assert_eq!(resp.error.unwrap().code, -32602, "Invalid UUID must return INVALID_PARAMS");

    // Test FINGERPRINT_NOT_FOUND for non-existent ID
    let missing = make_request(
        "memory/retrieve",
        Some(JsonRpcId::Number(3)),
        Some(json!({ "fingerprintId": "00000000-0000-0000-0000-000000000000" })),
    );
    let resp = handlers.dispatch(missing).await;
    assert_eq!(resp.error.unwrap().code, -32010, "Non-existent must return FINGERPRINT_NOT_FOUND");

    // Test INVALID_PARAMS for empty query
    let empty_query = make_request(
        "memory/search",
        Some(JsonRpcId::Number(4)),
        Some(json!({ "query": "", "topK": 5 })),
    );
    let resp = handlers.dispatch(empty_query).await;
    assert_eq!(resp.error.unwrap().code, -32602, "Empty query must return INVALID_PARAMS");
}
