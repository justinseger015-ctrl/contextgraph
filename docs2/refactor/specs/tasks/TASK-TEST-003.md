# TASK-TEST-003: Integration Test Suite

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-TEST-003 |
| **Title** | Integration Test Suite |
| **Status** | :white_circle: todo |
| **Layer** | Testing |
| **Sequence** | 43 |
| **Estimated Days** | 2.5 |
| **Complexity** | Medium |

## Implements

- End-to-end system verification
- MCP protocol compliance testing
- Hook lifecycle verification
- Subagent coordination testing

## Dependencies

| Task | Reason |
|------|--------|
| TASK-INTEG-001 | MCP handlers to test |
| TASK-INTEG-004 | Hook system to test |
| TASK-INTEG-014 | Subagents to test |
| All LOGIC tasks | Pipeline components to test |

## Objective

Create comprehensive integration tests that verify:
1. Full MCP request/response cycles
2. Hook lifecycle (SessionStart → PreToolUse → PostToolUse → SessionEnd)
3. Subagent spawn/stop/merge cycles
4. End-to-end memory storage and retrieval
5. Search pipeline with real embeddings

## Context

Integration tests verify that components work together correctly. Unlike unit tests (isolated) or property tests (invariants), integration tests exercise full system paths.

## Scope

### In Scope

- MCP protocol integration tests
- Hook lifecycle integration tests
- Subagent coordination tests
- Full pipeline tests (store → search → retrieve)
- Error propagation tests
- Timeout behavior tests

### Out of Scope

- Performance benchmarks (see PERF tasks)
- UI/CLI integration tests
- External service mocking

## Definition of Done

### Test Infrastructure

```rust
// tests/common/mod.rs

use context_graph_core::teleology::array::TeleologicalArray;
use context_graph_mcp::McpServer;
use context_graph_storage::TeleologicalArrayStore;
use std::sync::Arc;
use tempfile::TempDir;

/// Test harness for integration tests
pub struct TestHarness {
    pub temp_dir: TempDir,
    pub store: Arc<TeleologicalArrayStore>,
    pub mcp_server: Arc<McpServer>,
    pub session_id: String,
}

impl TestHarness {
    /// Create new test harness with isolated environment
    pub async fn new() -> Self {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let store = Arc::new(
            TeleologicalArrayStore::new(&db_path)
                .await
                .unwrap()
        );

        let mcp_server = Arc::new(
            McpServer::new(store.clone())
                .await
                .unwrap()
        );

        let session_id = uuid::Uuid::new_v4().to_string();

        Self {
            temp_dir,
            store,
            mcp_server,
            session_id,
        }
    }

    /// Start a session (triggers SessionStart hook)
    pub async fn start_session(&self) -> SessionStartOutput {
        self.mcp_server
            .dispatch_session_start(&self.session_id)
            .await
            .unwrap()
    }

    /// End session (triggers SessionEnd hook)
    pub async fn end_session(&self) -> SessionEndOutput {
        self.mcp_server
            .dispatch_session_end(&self.session_id)
            .await
            .unwrap()
    }

    /// Call inject_context MCP tool
    pub async fn inject_context(
        &self,
        content: &str,
        importance: f32,
    ) -> InjectContextOutput {
        let params = InjectContextParams {
            content: content.to_string(),
            metadata: Default::default(),
            importance: Some(importance),
        };

        self.mcp_server
            .handle_inject_context(&self.session_id, params)
            .await
            .unwrap()
    }

    /// Call store_memory MCP tool
    pub async fn store_memory(
        &self,
        content: &str,
        purpose: &str,
    ) -> StoreMemoryOutput {
        let params = StoreMemoryParams {
            content: content.to_string(),
            purpose: purpose.to_string(),
            metadata: Default::default(),
        };

        self.mcp_server
            .handle_store_memory(&self.session_id, params)
            .await
            .unwrap()
    }

    /// Call search_graph MCP tool
    pub async fn search_graph(
        &self,
        query: &str,
        limit: usize,
    ) -> SearchGraphOutput {
        let params = SearchGraphParams {
            query: query.to_string(),
            limit: Some(limit),
            filters: Default::default(),
        };

        self.mcp_server
            .handle_search_graph(&self.session_id, params)
            .await
            .unwrap()
    }

    /// Create test array with content
    pub fn create_test_array(content: &str) -> TeleologicalArray {
        let mut array = TeleologicalArray::new();
        array.content = content.to_string();
        array
    }
}

impl Drop for TestHarness {
    fn drop(&mut self) {
        // temp_dir is automatically cleaned up
    }
}

/// Helper macro for async integration tests
#[macro_export]
macro_rules! integration_test {
    ($name:ident, $body:expr) => {
        #[tokio::test]
        async fn $name() {
            let harness = TestHarness::new().await;
            $body(harness).await;
        }
    };
}
```

### MCP Protocol Tests

```rust
// tests/mcp_integration.rs

mod common;
use common::TestHarness;

#[tokio::test]
async fn test_full_session_lifecycle() {
    let harness = TestHarness::new().await;

    // 1. Start session
    let start_output = harness.start_session().await;
    assert!(start_output.success);
    assert!(start_output.session_count >= 1);

    // 2. Inject context
    let inject_output = harness.inject_context("Test context", 0.8).await;
    assert!(inject_output.success);
    assert!(inject_output.array_id.is_some());

    // 3. Store memory
    let store_output = harness.store_memory(
        "Important fact to remember",
        "knowledge-capture",
    ).await;
    assert!(store_output.success);

    // 4. Search for stored memory
    let search_output = harness.search_graph("important fact", 10).await;
    assert!(!search_output.results.is_empty());
    assert!(search_output.results[0].content.contains("Important"));

    // 5. End session
    let end_output = harness.end_session().await;
    assert!(end_output.success);
    assert!(end_output.memories_consolidated >= 0);
}

#[tokio::test]
async fn test_inject_context_with_hooks() {
    let harness = TestHarness::new().await;
    harness.start_session().await;

    // Inject should trigger PreToolUse and PostToolUse hooks
    let output = harness.inject_context("Hook test content", 0.5).await;

    assert!(output.success);
    // Hooks should have processed
    assert!(output.hooks_executed.contains(&"PreToolUse"));
    assert!(output.hooks_executed.contains(&"PostToolUse"));

    harness.end_session().await;
}

#[tokio::test]
async fn test_search_with_no_results() {
    let harness = TestHarness::new().await;
    harness.start_session().await;

    // Search empty store
    let output = harness.search_graph("nonexistent query xyz", 10).await;

    assert!(output.success);
    assert!(output.results.is_empty());

    harness.end_session().await;
}

#[tokio::test]
async fn test_store_then_search_consistency() {
    let harness = TestHarness::new().await;
    harness.start_session().await;

    // Store multiple memories
    let contents = vec![
        "Rust programming language",
        "Python data science",
        "TypeScript web development",
    ];

    for content in &contents {
        harness.store_memory(content, "knowledge").await;
    }

    // Search should find relevant results
    let output = harness.search_graph("programming language", 10).await;

    assert!(!output.results.is_empty());
    // Rust should be most relevant
    assert!(output.results[0].content.contains("Rust")
        || output.results[0].content.contains("programming"));

    harness.end_session().await;
}

#[tokio::test]
async fn test_rate_limiting() {
    let harness = TestHarness::new().await;
    harness.start_session().await;

    // Exhaust rate limit for consolidate_memories (1/min)
    let result1 = harness.mcp_server
        .handle_consolidate_memories(&harness.session_id, Default::default())
        .await;
    assert!(result1.is_ok());

    // Second call should be rate limited
    let result2 = harness.mcp_server
        .handle_consolidate_memories(&harness.session_id, Default::default())
        .await;
    assert!(matches!(result2, Err(McpError::RateLimited(_))));

    harness.end_session().await;
}
```

### Hook Lifecycle Tests

```rust
// tests/hook_integration.rs

mod common;
use common::TestHarness;

#[tokio::test]
async fn test_hook_execution_order() {
    let harness = TestHarness::new().await;

    // Track hook execution order
    let hook_log = Arc::new(RwLock::new(Vec::new()));

    // Configure hooks to log execution
    harness.mcp_server.set_hook_logger(hook_log.clone());

    // Run full lifecycle
    harness.start_session().await;
    harness.inject_context("Test", 0.5).await;
    harness.store_memory("Memory", "test").await;
    harness.search_graph("query", 10).await;
    harness.end_session().await;

    // Verify order
    let log = hook_log.read().await;
    assert_eq!(log[0], "SessionStart");
    assert!(log.contains(&"PreToolUse".to_string()));
    assert!(log.contains(&"PostToolUse".to_string()));
    assert_eq!(log[log.len() - 1], "SessionEnd");
}

#[tokio::test]
async fn test_hook_timeout_handling() {
    let harness = TestHarness::new().await;

    // Configure slow hook that exceeds timeout
    harness.mcp_server.set_hook_delay(
        HookEvent::PreToolUse,
        Duration::from_secs(10),
    );

    harness.start_session().await;

    // PreToolUse has 3000ms timeout, should fail
    let result = harness.inject_context("Timeout test", 0.5).await;

    // Hook timeout should not fail the operation, just log warning
    assert!(result.success);
    assert!(result.warnings.iter().any(|w| w.contains("timeout")));

    harness.end_session().await;
}

#[tokio::test]
async fn test_session_end_consolidation() {
    let harness = TestHarness::new().await;
    harness.start_session().await;

    // Store many memories
    for i in 0..100 {
        harness.store_memory(&format!("Memory {}", i), "test").await;
    }

    // End session should trigger consolidation
    let output = harness.end_session().await;

    // Consolidation should have run (60s timeout allows it)
    assert!(output.consolidation_result.is_some());
}
```

### Subagent Integration Tests

```rust
// tests/subagent_integration.rs

mod common;
use common::TestHarness;

#[tokio::test]
async fn test_subagent_spawn_and_stop() {
    let harness = TestHarness::new().await;
    harness.start_session().await;

    // Spawn embedding subagent
    let subagent = harness.mcp_server
        .spawn_subagent(SubagentType::Embedding)
        .await
        .unwrap();

    assert_eq!(subagent.status(), SubagentStatus::Running);

    // Stop subagent
    let output = harness.mcp_server
        .stop_subagent(subagent.id())
        .await
        .unwrap();

    assert!(output.success);
    // SubagentStop hook should have merged learnings
    assert!(output.merged_count >= 0);

    harness.end_session().await;
}

#[tokio::test]
async fn test_subagent_learning_merge() {
    let harness = TestHarness::new().await;
    harness.start_session().await;

    let initial_count = harness.store.count().await.unwrap();

    // Spawn goal subagent
    let subagent = harness.mcp_server
        .spawn_subagent(SubagentType::Goal)
        .await
        .unwrap();

    // Subagent discovers patterns/goals internally
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Stop and merge
    let output = harness.mcp_server
        .stop_subagent(subagent.id())
        .await
        .unwrap();

    // Learnings should be merged into main store
    let final_count = harness.store.count().await.unwrap();
    assert!(final_count >= initial_count);

    harness.end_session().await;
}

#[tokio::test]
async fn test_all_subagents_health_check() {
    let harness = TestHarness::new().await;
    harness.start_session().await;

    // Spawn all subagents
    let types = vec![
        SubagentType::Embedding,
        SubagentType::Search,
        SubagentType::Goal,
        SubagentType::Dream,
    ];

    for agent_type in &types {
        let subagent = harness.mcp_server
            .spawn_subagent(*agent_type)
            .await
            .unwrap();

        let health = subagent.health_check().await;
        assert_eq!(health.status, SubagentStatus::Running);
    }

    // All subagents should be healthy
    let all_health = harness.mcp_server.subagent_manager()
        .health_check_all()
        .await;

    assert_eq!(all_health.len(), 4);
    for (_, health) in all_health {
        assert_eq!(health.status, SubagentStatus::Running);
    }

    harness.end_session().await;
}
```

### Pipeline Integration Tests

```rust
// tests/pipeline_integration.rs

mod common;
use common::TestHarness;

#[tokio::test]
async fn test_full_pipeline_store_to_search() {
    let harness = TestHarness::new().await;
    harness.start_session().await;

    // Store memories with embeddings
    let memories = vec![
        ("Machine learning algorithms", 0.9),
        ("Database indexing strategies", 0.8),
        ("Web application security", 0.7),
        ("Cloud infrastructure design", 0.6),
    ];

    for (content, importance) in memories {
        harness.inject_context(content, importance).await;
    }

    // Wait for indexing
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Search should use entry-point selection
    let output = harness.search_graph("machine learning", 10).await;

    assert!(!output.results.is_empty());
    // ML should be top result due to semantic similarity
    assert!(output.results[0].content.contains("Machine learning"));
    // Verify entry point was selected (not hardcoded)
    assert!(output.entry_point_used.is_some());

    harness.end_session().await;
}

#[tokio::test]
async fn test_cache_hit_on_repeated_search() {
    let harness = TestHarness::new().await;
    harness.start_session().await;

    harness.store_memory("Test content for caching", "test").await;

    // First search - cache miss
    let output1 = harness.search_graph("test content", 10).await;
    assert!(!output1.cache_hit);

    // Second identical search - cache hit
    let output2 = harness.search_graph("test content", 10).await;
    assert!(output2.cache_hit);

    // Results should be identical
    assert_eq!(output1.results.len(), output2.results.len());

    harness.end_session().await;
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Test timeout | 30s per test |
| Parallel test execution | Supported (isolated harness) |
| Test cleanup | Automatic (temp dirs) |

## Verification

- [ ] All integration tests pass
- [ ] Tests run in < 5 minutes total
- [ ] No flaky tests
- [ ] Tests isolated (can run in parallel)
- [ ] Coverage of all MCP tools
- [ ] Coverage of all hooks
- [ ] Coverage of all subagents

## Files to Create

| File | Purpose |
|------|---------|
| `tests/common/mod.rs` | Test harness |
| `tests/mcp_integration.rs` | MCP protocol tests |
| `tests/hook_integration.rs` | Hook lifecycle tests |
| `tests/subagent_integration.rs` | Subagent tests |
| `tests/pipeline_integration.rs` | Pipeline tests |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Flaky tests | Medium | Medium | Proper isolation |
| Slow tests | Medium | Low | Parallel execution |
| Test maintenance | Medium | Medium | Good abstractions |

## Traceability

- Source: System integration requirements
- Related: All INTEG and LOGIC tasks
