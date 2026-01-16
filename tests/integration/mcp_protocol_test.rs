//! MCP Protocol Compliance Integration Tests
//!
//! Tests TC-GHOST-009 through TC-GHOST-015 for MCP protocol compliance.
//! These tests verify the MCP server properly implements JSON-RPC 2.0
//! and the Model Context Protocol specification.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// JSON-RPC 2.0 Request structure
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    id: Option<Value>,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

impl JsonRpcRequest {
    fn new(id: impl Into<Value>, method: &str, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0",
            id: Some(id.into()),
            method: method.to_string(),
            params,
        }
    }

    fn notification(method: &str, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0",
            id: None,
            method: method.to_string(),
            params,
        }
    }
}

/// JSON-RPC 2.0 Response structure
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Option<Value>,
    result: Option<Value>,
    error: Option<JsonRpcError>,
    #[serde(rename = "X-Cognitive-Pulse")]
    cognitive_pulse: Option<CognitivePulse>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    data: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct CognitivePulse {
    entropy: f32,
    coherence: f32,
    suggested_action: String,
}

/// MCP Server test harness
struct McpTestServer {
    process: Child,
}

impl McpTestServer {
    /// Spawn the MCP server process
    fn spawn() -> Result<Self, std::io::Error> {
        let process = Command::new("cargo")
            .args(["run", "-p", "context-graph-mcp", "--quiet"])
            .current_dir(env!("CARGO_MANIFEST_DIR").to_string() + "/..")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;

        // Give the server time to initialize
        std::thread::sleep(Duration::from_millis(500));

        Ok(Self { process })
    }

    /// Send a request and receive a response
    fn send_request(&mut self, request: &JsonRpcRequest) -> Result<JsonRpcResponse, String> {
        let stdin = self.process.stdin.as_mut().ok_or("Failed to get stdin")?;
        let stdout = self.process.stdout.as_mut().ok_or("Failed to get stdout")?;

        // Send request
        let request_json = serde_json::to_string(request).map_err(|e| e.to_string())?;
        writeln!(stdin, "{}", request_json).map_err(|e| e.to_string())?;
        stdin.flush().map_err(|e| e.to_string())?;

        // Read response
        let mut reader = BufReader::new(stdout);
        let mut response_line = String::new();

        // Use a timeout-like approach with multiple read attempts
        for _ in 0..10 {
            match reader.read_line(&mut response_line) {
                Ok(0) => {
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
                Ok(_) => break,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
                Err(e) => return Err(e.to_string()),
            }
        }

        if response_line.is_empty() {
            return Err("No response received".to_string());
        }

        serde_json::from_str(&response_line).map_err(|e| format!("Parse error: {} for: {}", e, response_line))
    }

    /// Send a notification (no response expected)
    fn send_notification(&mut self, request: &JsonRpcRequest) -> Result<(), String> {
        let stdin = self.process.stdin.as_mut().ok_or("Failed to get stdin")?;

        let request_json = serde_json::to_string(request).map_err(|e| e.to_string())?;
        writeln!(stdin, "{}", request_json).map_err(|e| e.to_string())?;
        stdin.flush().map_err(|e| e.to_string())?;

        Ok(())
    }
}

impl Drop for McpTestServer {
    fn drop(&mut self) {
        // First send SIGKILL to terminate the process
        let _ = self.process.kill();
        // CRITICAL: Must call wait() to reap the zombie process.
        // Without this, the process remains in the kernel's process table
        // as a zombie (state 'Z') until the parent process collects its
        // exit status via wait(). This was causing zombie accumulation
        // during test runs.
        let _ = self.process.wait();
    }
}

// =============================================================================
// TC-GHOST-009: Initialize Handshake Test
// =============================================================================

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn tc_ghost_009_initialize_handshake() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    // Send initialize request
    let request = JsonRpcRequest::new(1, "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {
            "name": "test-client",
            "version": "1.0.0"
        }
    })));

    let response = server.send_request(&request).expect("Failed to get response");

    // Verify JSON-RPC 2.0 compliance
    assert_eq!(response.jsonrpc, "2.0");
    assert_eq!(response.id, Some(json!(1)));
    assert!(response.error.is_none(), "Expected success response");

    // Verify MCP initialize response
    let result = response.result.expect("Expected result");
    assert_eq!(result["protocolVersion"], "2024-11-05");
    assert!(result["capabilities"].is_object());
    assert!(result["serverInfo"].is_object());
    assert_eq!(result["serverInfo"]["name"], "context-graph-mcp");
}

// =============================================================================
// TC-GHOST-010: Tools List Response Validation
// =============================================================================

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn tc_ghost_010_tools_list_response() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    // Initialize first
    let init_request = JsonRpcRequest::new(1, "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "test", "version": "1.0" }
    })));
    server.send_request(&init_request).expect("Initialize failed");

    // Request tools list
    let request = JsonRpcRequest::new(2, "tools/list", None);
    let response = server.send_request(&request).expect("Failed to get response");

    assert!(response.error.is_none(), "Expected success response");

    let result = response.result.expect("Expected result");
    let tools = result["tools"].as_array().expect("Expected tools array");

    // Verify all 5 required tools are present
    let tool_names: Vec<&str> = tools
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();

    assert!(tool_names.contains(&"inject_context"), "Missing inject_context tool");
    assert!(tool_names.contains(&"store_memory"), "Missing store_memory tool");
    assert!(tool_names.contains(&"get_memetic_status"), "Missing get_memetic_status tool");
    assert!(tool_names.contains(&"get_graph_manifest"), "Missing get_graph_manifest tool");
    assert!(tool_names.contains(&"search_graph"), "Missing search_graph tool");

    // Verify each tool has required schema
    for tool in tools {
        assert!(tool["name"].is_string(), "Tool must have name");
        assert!(tool["description"].is_string(), "Tool must have description");
        assert!(tool["inputSchema"].is_object(), "Tool must have inputSchema");
    }
}

// =============================================================================
// TC-GHOST-011: Tools Call with Valid Parameters
// =============================================================================

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn tc_ghost_011_tools_call_inject_context() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    // Initialize
    let init_request = JsonRpcRequest::new(1, "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "test", "version": "1.0" }
    })));
    server.send_request(&init_request).expect("Initialize failed");

    // Call inject_context tool
    let request = JsonRpcRequest::new(3, "tools/call", Some(json!({
        "name": "inject_context",
        "arguments": {
            "content": "Test context content for integration test",
            "rationale": "Testing MCP protocol compliance",
            "importance": 0.8
        }
    })));

    let response = server.send_request(&request).expect("Failed to get response");

    assert!(response.error.is_none(), "Expected success response, got: {:?}", response.error);

    // Verify MCP tool result format
    let result = response.result.expect("Expected result");
    assert!(result["content"].is_array(), "Tool result must have content array");
    assert_eq!(result["isError"], false, "Tool should succeed");

    // Parse the text content
    let content = result["content"][0]["text"]
        .as_str()
        .expect("Content must have text");
    let data: Value = serde_json::from_str(content).expect("Content must be valid JSON");

    assert!(data["nodeId"].is_string(), "Result must include nodeId");
    assert!(data["utl"].is_object(), "Result must include UTL metrics");
}

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn tc_ghost_011_tools_call_store_memory() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    // Initialize
    let init_request = JsonRpcRequest::new(1, "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "test", "version": "1.0" }
    })));
    server.send_request(&init_request).expect("Initialize failed");

    // Call store_memory tool
    let request = JsonRpcRequest::new(4, "tools/call", Some(json!({
        "name": "store_memory",
        "arguments": {
            "content": "Memory content for test",
            "importance": 0.7
        }
    })));

    let response = server.send_request(&request).expect("Failed to get response");

    assert!(response.error.is_none(), "Expected success response");

    let result = response.result.expect("Expected result");
    assert_eq!(result["isError"], false);
}

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn tc_ghost_011_tools_call_get_memetic_status() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    // Initialize
    let init_request = JsonRpcRequest::new(1, "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "test", "version": "1.0" }
    })));
    server.send_request(&init_request).expect("Initialize failed");

    // Call get_memetic_status tool
    let request = JsonRpcRequest::new(5, "tools/call", Some(json!({
        "name": "get_memetic_status",
        "arguments": {}
    })));

    let response = server.send_request(&request).expect("Failed to get response");

    assert!(response.error.is_none(), "Expected success response");

    let result = response.result.expect("Expected result");
    let content = result["content"][0]["text"].as_str().unwrap();
    let data: Value = serde_json::from_str(content).expect("Content must be valid JSON");

    assert!(data["phase"].is_string(), "Must include phase");
    assert!(data["nodeCount"].is_number(), "Must include nodeCount");
    assert!(data["utl"].is_object(), "Must include UTL metrics");
}

// =============================================================================
// TC-GHOST-012: Cognitive Pulse in Responses
// =============================================================================

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn tc_ghost_012_cognitive_pulse_in_initialize() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    let request = JsonRpcRequest::new(1, "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "test", "version": "1.0" }
    })));

    let response = server.send_request(&request).expect("Failed to get response");

    // Initialize response should include Cognitive Pulse
    if let Some(pulse) = response.cognitive_pulse {
        assert!(pulse.entropy >= 0.0 && pulse.entropy <= 1.0, "Entropy must be in [0,1]");
        assert!(pulse.coherence >= 0.0 && pulse.coherence <= 1.0, "Coherence must be in [0,1]");
        assert!(!pulse.suggested_action.is_empty(), "Must have suggested action");
    }
    // Note: Cognitive Pulse is optional in initialize response
}

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn tc_ghost_012_cognitive_pulse_in_memory_store() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    // Initialize
    let init_request = JsonRpcRequest::new(1, "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "test", "version": "1.0" }
    })));
    server.send_request(&init_request).expect("Initialize failed");

    // Memory store should include Cognitive Pulse
    let request = JsonRpcRequest::new(2, "memory/store", Some(json!({
        "content": "Test content with pulse"
    })));

    let response = server.send_request(&request).expect("Failed to get response");

    if let Some(pulse) = response.cognitive_pulse {
        assert!(pulse.entropy >= 0.0 && pulse.entropy <= 1.0);
        assert!(pulse.coherence >= 0.0 && pulse.coherence <= 1.0);
    }
}

// =============================================================================
// TC-GHOST-013: Parse Error Handling
// =============================================================================

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn tc_ghost_013_parse_error() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    // Send invalid JSON
    let stdin = server.process.stdin.as_mut().unwrap();
    writeln!(stdin, "{{ invalid json }}").unwrap();
    stdin.flush().unwrap();

    // Read response
    let stdout = server.process.stdout.as_mut().unwrap();
    let mut reader = BufReader::new(stdout);
    let mut response_line = String::new();

    std::thread::sleep(Duration::from_millis(200));
    let _ = reader.read_line(&mut response_line);

    if !response_line.is_empty() {
        let response: JsonRpcResponse = serde_json::from_str(&response_line)
            .expect("Error response should be valid JSON");

        assert!(response.error.is_some(), "Should return error");
        let error = response.error.unwrap();
        assert_eq!(error.code, -32700, "Parse error code should be -32700");
    }
}

// =============================================================================
// TC-GHOST-014: Method Not Found Error
// =============================================================================

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn tc_ghost_014_method_not_found() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    // Initialize first
    let init_request = JsonRpcRequest::new(1, "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "test", "version": "1.0" }
    })));
    server.send_request(&init_request).expect("Initialize failed");

    // Call unknown method
    let request = JsonRpcRequest::new(2, "unknown/method", None);
    let response = server.send_request(&request).expect("Failed to get response");

    assert!(response.error.is_some(), "Should return error");
    let error = response.error.unwrap();
    assert_eq!(error.code, -32601, "Method not found code should be -32601");
}

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn tc_ghost_014_invalid_params() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    // Initialize
    let init_request = JsonRpcRequest::new(1, "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "test", "version": "1.0" }
    })));
    server.send_request(&init_request).expect("Initialize failed");

    // Call tools/call without required params
    let request = JsonRpcRequest::new(2, "tools/call", None);
    let response = server.send_request(&request).expect("Failed to get response");

    assert!(response.error.is_some(), "Should return error");
    let error = response.error.unwrap();
    assert_eq!(error.code, -32602, "Invalid params code should be -32602");
}

// =============================================================================
// TC-GHOST-015: Tool Not Found Error
// =============================================================================

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn tc_ghost_015_tool_not_found() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    // Initialize
    let init_request = JsonRpcRequest::new(1, "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "test", "version": "1.0" }
    })));
    server.send_request(&init_request).expect("Initialize failed");

    // Call unknown tool
    let request = JsonRpcRequest::new(2, "tools/call", Some(json!({
        "name": "nonexistent_tool",
        "arguments": {}
    })));

    let response = server.send_request(&request).expect("Failed to get response");

    assert!(response.error.is_some(), "Should return error");
    let error = response.error.unwrap();
    assert_eq!(error.code, -32006, "Tool not found code should be -32006");
}

// =============================================================================
// Additional Protocol Compliance Tests
// =============================================================================

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn test_shutdown_handling() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    // Initialize
    let init_request = JsonRpcRequest::new(1, "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "test", "version": "1.0" }
    })));
    server.send_request(&init_request).expect("Initialize failed");

    // Send shutdown
    let request = JsonRpcRequest::new(99, "shutdown", None);
    let response = server.send_request(&request).expect("Failed to get response");

    assert!(response.error.is_none(), "Shutdown should succeed");
    assert_eq!(response.result, Some(Value::Null), "Shutdown result should be null");
}

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn test_string_id_handling() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    // Send with string ID
    let request = JsonRpcRequest::new("request-abc-123", "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "test", "version": "1.0" }
    })));

    let response = server.send_request(&request).expect("Failed to get response");

    // ID should be echoed back as string
    assert_eq!(response.id, Some(json!("request-abc-123")));
}

#[test]
#[ignore = "requires built binary - run with cargo test -- --ignored"]
fn test_get_graph_manifest() {
    let mut server = McpTestServer::spawn().expect("Failed to spawn server");

    // Initialize
    let init_request = JsonRpcRequest::new(1, "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "test", "version": "1.0" }
    })));
    server.send_request(&init_request).expect("Initialize failed");

    // Get manifest
    let request = JsonRpcRequest::new(2, "tools/call", Some(json!({
        "name": "get_graph_manifest",
        "arguments": {}
    })));

    let response = server.send_request(&request).expect("Failed to get response");

    assert!(response.error.is_none());

    let result = response.result.expect("Expected result");
    let content = result["content"][0]["text"].as_str().unwrap();
    let data: Value = serde_json::from_str(content).expect("Content must be valid JSON");

    // Verify 5-layer architecture is documented
    assert_eq!(data["architecture"], "5-layer-bio-nervous");
    let layers = data["layers"].as_array().expect("Must have layers");
    assert_eq!(layers.len(), 5, "Must have exactly 5 layers");
}
