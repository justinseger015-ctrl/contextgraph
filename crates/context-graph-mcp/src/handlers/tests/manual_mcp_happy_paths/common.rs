//! Common utilities for manual MCP happy path tests

use crate::protocol::{JsonRpcId, JsonRpcRequest};

/// Helper function to create a JSON-RPC request
pub(crate) fn make_request(
    method: &str,
    id: i64,
    params: Option<serde_json::Value>,
) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(id)),
        method: method.to_string(),
        params,
    }
}

/// Re-export the test helper from parent module (crate visibility)
pub(crate) use super::super::create_test_handlers_with_rocksdb_store_access;
