# TASK-TEST-005: End-to-End MCP Protocol Tests

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-TEST-005 |
| **Title** | End-to-End MCP Protocol Tests |
| **Status** | :white_circle: todo |
| **Layer** | Testing |
| **Sequence** | 45 |
| **Estimated Days** | 1.5 |
| **Complexity** | Medium |

## Implements

- MCP protocol compliance
- Claude Code integration verification
- Real-world usage scenarios

## Dependencies

| Task | Reason |
|------|--------|
| TASK-INTEG-001 | MCP handlers |
| TASK-INTEG-002 | inject_context implementation |
| TASK-INTEG-003 | store_memory implementation |
| TASK-INTEG-006 | search_graph implementation |
| TASK-INTEG-013 | consolidate_memories implementation |

## Objective

Create end-to-end tests that verify the MCP server works correctly when called via the MCP protocol, simulating how Claude Code would interact with the context graph.

## Context

End-to-end tests verify the complete request/response cycle through the MCP protocol layer. This includes:
- JSON-RPC message parsing
- Tool discovery
- Parameter validation
- Response formatting
- Error handling

## Scope

### In Scope

- MCP protocol compliance tests
- Tool discovery tests
- All 5 MCP tools tested via protocol
- Error response formatting
- Concurrent request handling
- Session management via MCP

### Out of Scope

- Performance testing (see PERF tasks)
- Network transport testing
- Real Claude Code integration

## Definition of Done

### MCP Test Client

```rust
// tests/e2e/mcp_client.rs

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::process::{Child, Command};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// MCP test client for e2e testing
pub struct McpTestClient {
    process: Child,
    request_id: u64,
}

#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    id: u64,
    method: String,
    params: Value,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: u64,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
    data: Option<Value>,
}

impl McpTestClient {
    /// Start MCP server and connect
    pub async fn connect() -> Result<Self, Box<dyn std::error::Error>> {
        let process = Command::new("cargo")
            .args(["run", "--bin", "context-graph-mcp"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        Ok(Self {
            process,
            request_id: 0,
        })
    }

    /// Send request and get response
    pub async fn call(
        &mut self,
        method: &str,
        params: Value,
    ) -> Result<Value, McpTestError> {
        self.request_id += 1;

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id: self.request_id,
            method: method.to_string(),
            params,
        };

        let request_json = serde_json::to_string(&request)?;

        // Write to stdin
        let stdin = self.process.stdin.as_mut().unwrap();
        stdin.write_all(request_json.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        stdin.flush().await?;

        // Read from stdout
        let stdout = self.process.stdout.as_mut().unwrap();
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        reader.read_line(&mut line).await?;

        let response: JsonRpcResponse = serde_json::from_str(&line)?;

        if let Some(error) = response.error {
            return Err(McpTestError::RpcError {
                code: error.code,
                message: error.message,
            });
        }

        response.result.ok_or(McpTestError::NoResult)
    }

    /// Call tools/list to discover available tools
    pub async fn list_tools(&mut self) -> Result<Vec<ToolInfo>, McpTestError> {
        let result = self.call("tools/list", json!({})).await?;
        let tools: ToolListResponse = serde_json::from_value(result)?;
        Ok(tools.tools)
    }

    /// Call a specific tool
    pub async fn call_tool(
        &mut self,
        name: &str,
        arguments: Value,
    ) -> Result<Value, McpTestError> {
        self.call("tools/call", json!({
            "name": name,
            "arguments": arguments
        })).await
    }

    /// Shutdown gracefully
    pub async fn shutdown(mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.call("shutdown", json!({})).await.ok();
        self.process.kill().await?;
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct ToolListResponse {
    tools: Vec<ToolInfo>,
}

#[derive(Debug, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, thiserror::Error)]
pub enum McpTestError {
    #[error("JSON-RPC error {code}: {message}")]
    RpcError { code: i64, message: String },
    #[error("No result in response")]
    NoResult,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
```

### Tool Discovery Tests

```rust
// tests/e2e/tool_discovery_tests.rs

mod mcp_client;
use mcp_client::McpTestClient;

#[tokio::test]
async fn test_discovers_all_required_tools() {
    let mut client = McpTestClient::connect().await.unwrap();

    let tools = client.list_tools().await.unwrap();
    let tool_names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();

    // Constitution requires these 5 tools
    assert!(tool_names.contains(&"inject_context"));
    assert!(tool_names.contains(&"store_memory"));
    assert!(tool_names.contains(&"search_graph"));
    assert!(tool_names.contains(&"discover_goals"));
    assert!(tool_names.contains(&"consolidate_memories"));

    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_tool_schemas_valid() {
    let mut client = McpTestClient::connect().await.unwrap();

    let tools = client.list_tools().await.unwrap();

    for tool in tools {
        // Each tool should have valid JSON schema
        assert!(tool.input_schema.is_object());
        let schema = tool.input_schema.as_object().unwrap();

        // Should have type and properties
        assert!(schema.contains_key("type"));
        assert!(schema.contains_key("properties"));
    }

    client.shutdown().await.unwrap();
}
```

### inject_context E2E Tests

```rust
// tests/e2e/inject_context_e2e.rs

#[tokio::test]
async fn test_inject_context_success() {
    let mut client = McpTestClient::connect().await.unwrap();

    let result = client.call_tool("inject_context", json!({
        "content": "Test context for injection",
        "importance": 0.8
    })).await.unwrap();

    assert!(result["success"].as_bool().unwrap());
    assert!(result["array_id"].is_string());

    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_inject_context_validates_importance() {
    let mut client = McpTestClient::connect().await.unwrap();

    // Invalid importance (>1.0)
    let result = client.call_tool("inject_context", json!({
        "content": "Test",
        "importance": 1.5
    })).await;

    assert!(result.is_err());

    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_inject_context_rejects_empty_content() {
    let mut client = McpTestClient::connect().await.unwrap();

    let result = client.call_tool("inject_context", json!({
        "content": "",
        "importance": 0.5
    })).await;

    assert!(result.is_err());

    client.shutdown().await.unwrap();
}
```

### store_memory E2E Tests

```rust
// tests/e2e/store_memory_e2e.rs

#[tokio::test]
async fn test_store_memory_success() {
    let mut client = McpTestClient::connect().await.unwrap();

    let result = client.call_tool("store_memory", json!({
        "content": "Important information to remember",
        "purpose": "knowledge-capture"
    })).await.unwrap();

    assert!(result["success"].as_bool().unwrap());
    assert!(result["memory_id"].is_string());

    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_store_memory_with_metadata() {
    let mut client = McpTestClient::connect().await.unwrap();

    let result = client.call_tool("store_memory", json!({
        "content": "Memory with metadata",
        "purpose": "test",
        "metadata": {
            "source": "e2e-test",
            "tags": ["test", "metadata"]
        }
    })).await.unwrap();

    assert!(result["success"].as_bool().unwrap());

    client.shutdown().await.unwrap();
}
```

### search_graph E2E Tests

```rust
// tests/e2e/search_graph_e2e.rs

#[tokio::test]
async fn test_search_graph_empty_results() {
    let mut client = McpTestClient::connect().await.unwrap();

    let result = client.call_tool("search_graph", json!({
        "query": "nonexistent query xyz123",
        "limit": 10
    })).await.unwrap();

    assert!(result["success"].as_bool().unwrap());
    assert!(result["results"].as_array().unwrap().is_empty());

    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_search_graph_after_store() {
    let mut client = McpTestClient::connect().await.unwrap();

    // Store first
    client.call_tool("store_memory", json!({
        "content": "Rust programming language features",
        "purpose": "knowledge"
    })).await.unwrap();

    // Search
    let result = client.call_tool("search_graph", json!({
        "query": "Rust programming",
        "limit": 10
    })).await.unwrap();

    let results = result["results"].as_array().unwrap();
    assert!(!results.is_empty());
    assert!(results[0]["content"].as_str().unwrap().contains("Rust"));

    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_search_graph_respects_limit() {
    let mut client = McpTestClient::connect().await.unwrap();

    // Store multiple
    for i in 0..20 {
        client.call_tool("store_memory", json!({
            "content": format!("Test memory number {}", i),
            "purpose": "test"
        })).await.unwrap();
    }

    // Search with limit
    let result = client.call_tool("search_graph", json!({
        "query": "test memory",
        "limit": 5
    })).await.unwrap();

    let results = result["results"].as_array().unwrap();
    assert!(results.len() <= 5);

    client.shutdown().await.unwrap();
}
```

### discover_goals E2E Tests

```rust
// tests/e2e/discover_goals_e2e.rs

#[tokio::test]
async fn test_discover_goals_empty() {
    let mut client = McpTestClient::connect().await.unwrap();

    let result = client.call_tool("discover_goals", json!({
        "limit": 10
    })).await.unwrap();

    assert!(result["success"].as_bool().unwrap());
    // May or may not have goals depending on state

    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_discover_goals_after_activity() {
    let mut client = McpTestClient::connect().await.unwrap();

    // Generate activity
    for _ in 0..10 {
        client.call_tool("store_memory", json!({
            "content": "Working on implementing authentication system",
            "purpose": "task-progress"
        })).await.unwrap();
    }

    // Discover goals
    let result = client.call_tool("discover_goals", json!({
        "limit": 10
    })).await.unwrap();

    assert!(result["success"].as_bool().unwrap());
    // Should discover some patterns

    client.shutdown().await.unwrap();
}
```

### consolidate_memories E2E Tests

```rust
// tests/e2e/consolidate_e2e.rs

#[tokio::test]
async fn test_consolidate_light_mode() {
    let mut client = McpTestClient::connect().await.unwrap();

    let result = client.call_tool("consolidate_memories", json!({
        "mode": "light",
        "dry_run": true
    })).await.unwrap();

    assert!(result["success"].as_bool().unwrap());
    assert_eq!(result["mode"].as_str().unwrap(), "light");
    assert!(result["dry_run"].as_bool().unwrap());

    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_consolidate_rate_limited() {
    let mut client = McpTestClient::connect().await.unwrap();

    // First call should succeed
    client.call_tool("consolidate_memories", json!({
        "mode": "light"
    })).await.unwrap();

    // Second call should be rate limited (1/min)
    let result = client.call_tool("consolidate_memories", json!({
        "mode": "light"
    })).await;

    assert!(result.is_err());

    client.shutdown().await.unwrap();
}
```

### Concurrent Request Tests

```rust
// tests/e2e/concurrent_e2e.rs

#[tokio::test]
async fn test_concurrent_stores() {
    let mut client = McpTestClient::connect().await.unwrap();

    let mut handles = vec![];

    // Spawn concurrent store requests
    for i in 0..10 {
        let handle = tokio::spawn(async move {
            // Note: Would need shared client or separate connections
            // Simplified for illustration
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }

    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_store_and_search_concurrent() {
    let mut client = McpTestClient::connect().await.unwrap();

    // Store while searching
    let store_handle = tokio::spawn(async move {
        // Store operations
    });

    let search_handle = tokio::spawn(async move {
        // Search operations
    });

    store_handle.await.unwrap();
    search_handle.await.unwrap();

    client.shutdown().await.unwrap();
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| E2E test timeout | 60s per test |
| Server startup | < 5s |
| Response time | < 1s per tool call |

## Verification

- [ ] Tool discovery returns all 5 required tools
- [ ] inject_context works via MCP protocol
- [ ] store_memory works via MCP protocol
- [ ] search_graph works via MCP protocol
- [ ] discover_goals works via MCP protocol
- [ ] consolidate_memories works via MCP protocol
- [ ] Rate limiting enforced at protocol level
- [ ] Error responses properly formatted

## Files to Create

| File | Purpose |
|------|---------|
| `tests/e2e/mod.rs` | E2E test module |
| `tests/e2e/mcp_client.rs` | MCP test client |
| `tests/e2e/tool_discovery_tests.rs` | Discovery tests |
| `tests/e2e/inject_context_e2e.rs` | inject_context tests |
| `tests/e2e/store_memory_e2e.rs` | store_memory tests |
| `tests/e2e/search_graph_e2e.rs` | search_graph tests |
| `tests/e2e/discover_goals_e2e.rs` | discover_goals tests |
| `tests/e2e/consolidate_e2e.rs` | consolidate tests |
| `tests/e2e/concurrent_e2e.rs` | Concurrency tests |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Flaky due to process | Medium | Medium | Retry logic |
| Port conflicts | Low | Low | Dynamic ports |
| Slow startup | Medium | Low | Warm-up phase |

## Traceability

- Source: Constitution MCP tool specifications
- Related: All INTEG-* handler tasks
