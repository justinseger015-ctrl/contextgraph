//! MCP lifecycle handlers.

use serde_json::json;
use tracing::info;

use context_graph_core::types::{CognitivePulse, SuggestedAction};

use crate::protocol::{JsonRpcId, JsonRpcResponse};

use super::Handlers;

impl Handlers {
    /// Handle MCP initialize request.
    ///
    /// Returns server capabilities following MCP 2024-11-05 protocol specification.
    pub(super) async fn handle_initialize(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        info!("Handling initialize request");

        // Initialize defaults - system hasn't processed any data yet
        // These will be updated with real values after first operations
        // Note: parameters are (entropy, coherence, coherence_delta, emotional_weight, action, source_layer)
        let pulse = CognitivePulse::new(
            0.5,  // entropy: neutral starting point
            0.5,  // coherence: neutral starting point (not fake 0.8)
            0.0,  // coherence_delta: no change yet (not fake 1.0)
            0.5,  // emotional_weight: neutral starting point
            SuggestedAction::Ready,
            None,  // source_layer: None during initialization (no layer has processed yet)
        );

        // MCP-compliant initialize response
        JsonRpcResponse::success(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": { "listChanged": true }
                },
                "serverInfo": {
                    "name": "context-graph-mcp",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        )
        .with_pulse(pulse)
    }

    /// Handle notifications/initialized - this is a notification, not a request.
    ///
    /// Notifications don't require a response per JSON-RPC 2.0 spec.
    pub(super) fn handle_initialized_notification(&self) -> JsonRpcResponse {
        info!("Client initialized notification received");

        // Return a response with no id, result, or error to signal "no response needed"
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: None,
            result: None,
            error: None,
            cognitive_pulse: None,
        }
    }

    /// Handle MCP shutdown request.
    pub(super) async fn handle_shutdown(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        info!("Handling shutdown request");
        JsonRpcResponse::success(id, json!(null))
    }
}
