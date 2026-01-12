//! Scenario 4: Ego State Verification Tests
//!
//! Tests get_ego_state tool:
//! - Valid identity and purpose vector
//! - 13-element purpose vector verification
//! - Warm state non-zero values

use serde_json::json;

use crate::handlers::tests::{create_test_handlers_with_warm_gwt, extract_mcp_tool_data};
use crate::protocol::{JsonRpcId, JsonRpcRequest};
use crate::tools::tool_names;

/// FSV Test: get_ego_state returns valid identity and purpose vector.
///
/// Source of Truth: SelfEgoProvider
/// Expected: Response contains purpose_vector (13D), identity_coherence, identity_status
#[tokio::test]
async fn test_get_ego_state_returns_valid_data() {
    let handlers = create_test_handlers_with_warm_gwt();

    // EXECUTE: Call get_ego_state
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_EGO_STATE,
            "arguments": {}
        })),
    };
    let response = handlers.dispatch(request).await;

    // VERIFY: Response is successful
    assert!(
        response.error.is_none(),
        "[FSV] Expected success, got error: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // FSV: CRITICAL - Verify purpose_vector has exactly 13 elements
    let pv = data
        .get("purpose_vector")
        .and_then(|v| v.as_array())
        .expect("purpose_vector must exist");
    assert_eq!(
        pv.len(),
        13,
        "[FSV] CRITICAL: Purpose vector must have 13 elements (one per embedder), got {}",
        pv.len()
    );

    // FSV: Verify all purpose vector elements are floats in [-1, 1]
    // (Purpose alignments are cosine similarities)
    for (i, val) in pv.iter().enumerate() {
        let v = val.as_f64().expect("purpose_vector elements must be f64");
        assert!(
            v >= -1.0 && v <= 1.0,
            "[FSV] Purpose vector[{}] must be in [-1, 1], got {}",
            i,
            v
        );
    }

    // FSV: Verify identity_coherence is in [0, 1]
    let identity_coherence = data
        .get("identity_coherence")
        .and_then(|v| v.as_f64())
        .expect("identity_coherence must exist");
    assert!(
        (0.0..=1.0).contains(&identity_coherence),
        "[FSV] identity_coherence must be in [0, 1], got {}",
        identity_coherence
    );

    // FSV: Verify identity_status is valid
    let status = data
        .get("identity_status")
        .and_then(|v| v.as_str())
        .expect("identity_status must exist");
    let valid_statuses = ["Healthy", "Warning", "Degraded", "Critical"];
    // Status might be Debug formatted (e.g., "Healthy" or "IdentityStatus::Healthy")
    let status_valid = valid_statuses.iter().any(|s| status.contains(s));
    assert!(
        status_valid,
        "[FSV] Invalid identity_status: {}, expected one containing {:?}",
        status,
        valid_statuses
    );

    // FSV: Verify coherence_with_actions is in [0, 1]
    let coherence_with_actions = data
        .get("coherence_with_actions")
        .and_then(|v| v.as_f64())
        .expect("coherence_with_actions must exist");
    assert!(
        (0.0..=1.0).contains(&coherence_with_actions),
        "[FSV] coherence_with_actions must be in [0, 1], got {}",
        coherence_with_actions
    );

    // FSV: Verify trajectory_length is non-negative
    let trajectory_length = data
        .get("trajectory_length")
        .and_then(|v| v.as_u64())
        .expect("trajectory_length must exist");
    assert!(
        trajectory_length >= 0,
        "[FSV] trajectory_length must be non-negative"
    );

    // FSV: Verify thresholds are present
    let thresholds = data.get("thresholds").expect("thresholds must exist");
    assert_eq!(
        thresholds.get("healthy").and_then(|v| v.as_f64()),
        Some(0.9),
        "[FSV] thresholds.healthy must be 0.9"
    );
    assert_eq!(
        thresholds.get("warning").and_then(|v| v.as_f64()),
        Some(0.7),
        "[FSV] thresholds.warning must be 0.7"
    );

    println!("[FSV] Phase 3 - get_ego_state verification PASSED");
    println!(
        "[FSV]   purpose_vector.len={}, identity_coherence={:.4}, status={}",
        pv.len(),
        identity_coherence,
        status
    );
    println!(
        "[FSV]   trajectory_length={}, coherence_with_actions={:.4}",
        trajectory_length, coherence_with_actions
    );
}

/// FSV Test: get_ego_state with WARM state has non-zero purpose vector.
#[tokio::test]
async fn test_get_ego_state_warm_has_non_zero_purpose_vector() {
    // Warm GWT state includes a pre-initialized purpose vector
    let handlers = create_test_handlers_with_warm_gwt();

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_EGO_STATE,
            "arguments": {}
        })),
    };
    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none());

    let data = extract_mcp_tool_data(&response.result.unwrap());
    let pv = data
        .get("purpose_vector")
        .and_then(|v| v.as_array())
        .expect("purpose_vector must exist");

    // FSV: At least some elements should be non-zero in warm state
    let non_zero_count = pv
        .iter()
        .filter(|v| {
            let val = v.as_f64().unwrap_or(0.0);
            val.abs() > 0.001
        })
        .count();

    assert!(
        non_zero_count > 0,
        "[FSV] WARM state should have non-zero purpose vector elements, got {} non-zero",
        non_zero_count
    );

    println!("[FSV] Phase 3 - get_ego_state WARM state verification PASSED");
    println!("[FSV]   Non-zero purpose vector elements: {}/13", non_zero_count);
}
