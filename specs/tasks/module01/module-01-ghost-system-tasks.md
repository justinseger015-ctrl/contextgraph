# Module 1: Ghost System - Atomic Task Specification

```yaml
metadata:
  module: Module 1 - Ghost System
  phase: 0
  approach: inside-out-bottom-up
  status: IMPLEMENTATION COMPLETE - VALIDATION REQUIRED
  spec_refs:
    - /home/cabdru/contextgraph/docs2/constitution.yaml
    - /home/cabdru/contextgraph/docs2/contextprd.md
  created: 2025-12-31
  last_audit: 2025-12-31
  auditor: sherlock-holmes
  verification: ALL CODE IMPLEMENTED - 52 UNIT TESTS PASS
```

---

## ⚠️ CRITICAL INSTRUCTIONS FOR AI AGENT

### GOLDEN RULES (ABSOLUTE)
1. **FAIL FAST**: No workarounds, no fallbacks. If broken, ERROR OUT with full diagnostics
2. **REAL DATA ONLY**: Tests must use actual implementations, NO MOCKS in validation
3. **VERIFY STATE**: Don't trust return values - ALWAYS query source of truth after operations
4. **NO BACKWARDS COMPATIBILITY**: System works or fails - no shims, no compatibility hacks

### BEFORE YOU START
1. Run `cargo check --workspace` - must pass
2. Run `cargo test --workspace` - expect 52 unit tests to pass
3. Run `cargo test -- --ignored` - for integration tests (requires MCP server running)
4. Read `/home/cabdru/contextgraph/docs2/constitution.yaml` for coding standards

---

## CURRENT PROJECT STATE (AUDITED 2025-12-31)

### Workspace Structure
```
/home/cabdru/contextgraph/
├── Cargo.toml                          # Workspace root - resolver = "2"
├── rust-toolchain.toml                 # Rust 1.75+
├── config/
│   ├── default.toml                    # Base config
│   ├── development.toml
│   ├── test.toml
│   └── production.toml
├── crates/
│   ├── context-graph-core/             # Domain logic
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── error.rs                # CoreError enum with thiserror
│   │   │   ├── config.rs               # Config loading with TOML + env override
│   │   │   ├── types/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── memory_node.rs      # MemoryNode, NodeId, EdgeId
│   │   │   │   ├── graph_edge.rs       # GraphEdge, EdgeType
│   │   │   │   ├── johari.rs           # JohariQuadrant, Modality enums
│   │   │   │   ├── utl.rs              # UtlMetrics, UtlContext, EmotionalState
│   │   │   │   ├── pulse.rs            # CognitivePulse, SuggestedAction
│   │   │   │   └── nervous.rs          # LayerId, LayerInput, LayerOutput
│   │   │   ├── traits/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── memory_store.rs     # MemoryStore trait + SearchOptions
│   │   │   │   ├── utl_processor.rs    # UTLProcessor trait
│   │   │   │   ├── nervous_layer.rs    # NervousLayer trait
│   │   │   │   └── graph_index.rs      # GraphIndex trait
│   │   │   └── stubs/
│   │   │       ├── mod.rs
│   │   │       ├── memory_stub.rs      # InMemoryStore (thread-safe)
│   │   │       ├── utl_stub.rs         # StubUtlProcessor
│   │   │       ├── layers.rs           # 5 stub layers (Sensing→Coherence)
│   │   │       └── graph_index.rs      # InMemoryGraphIndex
│   │   └── tests/
│   │       └── edge_case_tests.rs      # 10 edge case tests with state verification
│   ├── context-graph-embeddings/       # Embedding provider
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── error.rs                # EmbeddingError enum
│   │       ├── provider.rs             # EmbeddingProvider trait (async)
│   │       └── stub.rs                 # StubEmbedder (deterministic hashing)
│   ├── context-graph-cuda/             # GPU ops (stubs for Phase 0)
│   │   └── src/
│   │       ├── lib.rs                  # cuda_available() returns false
│   │       ├── error.rs
│   │       ├── ops.rs                  # VectorOps trait
│   │       └── stub.rs                 # StubVectorOps
│   └── context-graph-mcp/              # MCP server binary
│       └── src/
│           ├── main.rs                 # tokio::main entry point
│           ├── protocol.rs             # JsonRpc types, McpError codes
│           ├── server.rs               # McpServer with stdio transport
│           ├── tools.rs                # 5 tool definitions with JSON schemas
│           └── handlers.rs             # Tool handlers with Cognitive Pulse
├── tests/
│   ├── integration/
│   │   └── mcp_protocol_test.rs        # TC-GHOST-009 to TC-GHOST-015
│   ├── benchmarks/
│   ├── chaos/
│   ├── fixtures/
│   └── validation/
└── .github/
    └── workflows/
        └── ci.yml                      # GitHub Actions CI pipeline
```

### Implementation Status Matrix

| Component | Status | Files | Tests |
|-----------|--------|-------|-------|
| Workspace Cargo.toml | ✅ COMPLETE | Cargo.toml | N/A |
| CoreError type | ✅ COMPLETE | error.rs | Unit tests pass |
| JohariQuadrant/Modality | ✅ COMPLETE | johari.rs | Serde tests pass |
| MemoryNode + NodeId | ✅ COMPLETE | memory_node.rs | Round-trip tests |
| GraphEdge + EdgeType | ✅ COMPLETE | graph_edge.rs | Unit tests pass |
| UtlMetrics/UtlContext | ✅ COMPLETE | utl.rs | Default value tests |
| CognitivePulse | ✅ COMPLETE | pulse.rs | Boundary tests |
| LayerId + nervous types | ✅ COMPLETE | nervous.rs | Latency budget tests |
| EmbeddingProvider trait | ✅ COMPLETE | provider.rs | N/A (trait only) |
| StubEmbedder | ✅ COMPLETE | stub.rs | Determinism tests |
| MemoryStore trait | ✅ COMPLETE | memory_store.rs | N/A (trait only) |
| InMemoryStore | ✅ COMPLETE | memory_stub.rs | CRUD tests |
| UTLProcessor trait | ✅ COMPLETE | utl_processor.rs | N/A (trait only) |
| StubUtlProcessor | ✅ COMPLETE | utl_stub.rs | Range tests |
| NervousLayer trait | ✅ COMPLETE | nervous_layer.rs | N/A (trait only) |
| 5 Stub Layers | ✅ COMPLETE | layers.rs | Latency tests |
| GraphIndex trait | ✅ COMPLETE | graph_index.rs | N/A (trait only) |
| InMemoryGraphIndex | ✅ COMPLETE | graph_index.rs | Cosine similarity tests |
| Config loading | ✅ COMPLETE | config.rs | Env override tests |
| MCP Protocol types | ✅ COMPLETE | protocol.rs | Error code tests |
| MCP Server + initialize | ✅ COMPLETE | server.rs | Integration tests |
| 5 Tool definitions | ✅ COMPLETE | tools.rs | Schema tests |
| Tool handlers + Pulse | ✅ COMPLETE | handlers.rs | Response tests |
| MCP Error handling | ✅ COMPLETE | protocol.rs | -32700 to -32603 |
| CI pipeline | ✅ COMPLETE | ci.yml | GitHub Actions |
| Edge case tests | ✅ COMPLETE | edge_case_tests.rs | 10 tests |
| Integration tests | ✅ COMPLETE | mcp_protocol_test.rs | 12 tests (ignored) |

---

## VALIDATION TASKS (AGENT: EXECUTE THESE)

### TASK-VAL-001: Full Build Verification
```yaml
id: TASK-VAL-001
title: Verify full workspace compiles without warnings
type: validation
priority: critical
steps:
  - Run: cargo fmt --check --all
  - Run: cargo clippy --workspace -- -D warnings
  - Run: cargo build --workspace
  - Run: cargo doc --workspace --no-deps
expected_outcome: All commands exit 0 with no warnings
source_of_truth: Compiler output
```

### TASK-VAL-002: Unit Test Verification
```yaml
id: TASK-VAL-002
title: Verify all 52 unit tests pass with real data
type: validation
priority: critical
steps:
  - Run: cargo test --workspace --lib -- --nocapture
  - Verify: Output shows "52 passed"
  - Run: cargo test --workspace -- edge_case --nocapture
  - Verify: All 10 edge case tests print STATE BEFORE/AFTER/EVIDENCE
expected_outcome: All tests pass, no mocks detected
source_of_truth: Test output
failure_action: If any test fails, FIX IT. Do not skip or disable.
```

### TASK-VAL-003: Integration Test Verification
```yaml
id: TASK-VAL-003
title: Run MCP protocol integration tests
type: validation
priority: high
steps:
  - Build server: cargo build -p context-graph-mcp --release
  - Run ignored tests: cargo test -p context-graph-mcp -- --ignored --nocapture
  - Verify: TC-GHOST-009 through TC-GHOST-015 pass
expected_outcome: All 12 integration tests pass
source_of_truth: Test harness spawns actual server, reads real responses
note: Integration tests require binary to be built first
```

### TASK-VAL-004: MCP Server Manual Test
```yaml
id: TASK-VAL-004
title: Manual MCP server protocol test
type: validation
priority: high
steps:
  1. Start server: cargo run -p context-graph-mcp
  2. Send to stdin: {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
  3. Read stdout and verify response contains:
     - protocolVersion: "2024-11-05"
     - capabilities.tools
     - serverInfo.name
  4. Send: {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
  5. Verify 5 tools: inject_context, store_memory, get_memetic_status, get_graph_manifest, search_graph
source_of_truth: Actual JSON responses from stdout
```

---

## FULL STATE VERIFICATION PROTOCOL

After ANY operation that modifies state, you MUST:

### 1. Define Source of Truth
| Operation | Source of Truth |
|-----------|-----------------|
| store() | InMemoryStore internal HashMap via retrieve() |
| delete() | Verify retrieve() returns None (hard) or deleted=true (soft) |
| update() | retrieve() shows updated values |
| add() to GraphIndex | search() returns the added vector |
| MCP tool call | Parsed JSON response matches expected schema |

### 2. Execute & Inspect Pattern
```rust
// WRONG: Trust return value
let id = store.store(node).await?;
// Assume success...

// CORRECT: Verify state
let id = store.store(node).await?;
let stored = store.retrieve(id).await?
    .expect("FAILURE: Node not found after store");
assert_eq!(stored.content, original_content, "FAILURE: Content mismatch");
```

### 3. Edge Case Audit (Minimum 3)
For each new feature, test these edge cases:

**Empty Input:**
- STATE BEFORE: log current system state
- ACTION: call with empty string/vec
- STATE AFTER: log result and system state
- EVIDENCE: prove correct handling

**Maximum Limit:**
- STATE BEFORE: log state
- ACTION: call with max size (e.g., 65536 chars for content)
- STATE AFTER: verify stored correctly
- EVIDENCE: retrieve and compare

**Invalid Format:**
- STATE BEFORE: log state
- ACTION: call with wrong type/dimension
- STATE AFTER: verify error returned
- EVIDENCE: error type matches expected

### 4. Evidence of Success Format
```
=== OPERATION: store_memory ===
STATE BEFORE:
  - Store count: 0
  - Input content length: 100
EXECUTION:
  - store() called
  - Returned ID: 550e8400-e29b-41d4-a716-446655440000
STATE AFTER (Source of Truth Query):
  - retrieve(id) returned: Some(MemoryNode)
  - Content matches: true
  - Store count: 1
EVIDENCE: Node successfully stored and verified
```

---

## ERROR HANDLING REQUIREMENTS

### Anti-Patterns (FORBIDDEN per constitution.yaml)
- AP-001: `unwrap()` in production code → use `expect("descriptive message")`
- AP-002: Broad exception catching → use specific error types
- AP-003: Silent failures → always log errors before returning
- AP-004: Empty catch blocks → must log and propagate

### Required Error Propagation
```rust
// FORBIDDEN: Swallowing errors
fn process() -> Option<Data> {
    match risky_op() {
        Ok(d) => Some(d),
        Err(_) => None  // ❌ Silent failure
    }
}

// REQUIRED: Log and propagate
fn process() -> Result<Data, CoreError> {
    risky_op().map_err(|e| {
        tracing::error!("risky_op failed: {}", e);
        CoreError::from(e)
    })
}
```

### MCP Error Codes (Implemented in protocol.rs)
| Code | Meaning | When Used |
|------|---------|-----------|
| -32700 | Parse error | Invalid JSON |
| -32600 | Invalid Request | Missing jsonrpc/method |
| -32601 | Method not found | Unknown method |
| -32602 | Invalid params | Missing required params |
| -32603 | Internal error | Server crash/panic |
| -32006 | Tool not found | Unknown tool name |

---

## CODING STANDARDS (from constitution.yaml)

### Naming Conventions
- Files: `snake_case.rs`
- Types: `PascalCase`
- Functions: `snake_case_verb_first()` (e.g., `compute_learning_score`)
- Constants: `SCREAMING_SNAKE`
- JSON: `snake_case`

### Module Rules
- Max 500 lines per module (excluding tests)
- Co-locate unit tests via `#[cfg(test)]`
- Import order: std → external → workspace → super/self
- Use `Result<T, E>` for fallible operations
- Use `thiserror` for error derivation

### Concurrency
- Use `tokio` for async
- Use `Arc<RwLock<T>>` for shared state
- Lock order: `inner` → `faiss_index` (prevents deadlock)

---

## TEST VALIDATION REQUIREMENTS

### What Tests MUST Do
1. Use REAL implementations (InMemoryStore, StubUtlProcessor, etc.)
2. Print STATE BEFORE/AFTER for all stateful operations
3. Query source of truth after operations
4. Test edge cases (empty, max, invalid)
5. Fail loudly with descriptive messages

### What Tests MUST NOT Do
1. Use mock objects that fake behavior
2. Trust return values without verification
3. Skip error cases
4. Use `#[should_panic]` without explicit reason

### Test File Locations
- Unit tests: In same file as implementation via `#[cfg(test)]`
- Edge case tests: `crates/context-graph-core/tests/edge_case_tests.rs`
- Integration tests: `tests/integration/mcp_protocol_test.rs`

---

## KNOWN DEVIATIONS FROM ORIGINAL SPEC

These are intentional simplifications for Phase 0:

| Original Spec | Actual Implementation | Reason |
|--------------|----------------------|--------|
| `src/protocol/mod.rs` (directory) | `src/protocol.rs` (flat file) | Simpler for Ghost phase |
| `src/handlers/*.rs` (many files) | `src/handlers.rs` (consolidated) | All handlers in one file |
| `src/tools/definitions.rs` | `src/tools.rs` | Combined tools module |
| `memory_store.rs` stub name | `memory_stub.rs` | Clearer naming |
| `utl_processor.rs` stub name | `utl_stub.rs` | Clearer naming |
| `.ai/` directory | Not created | Agent protocol files not needed yet |

---

## QUICK REFERENCE: Running Tests

```bash
# Full workspace check
cargo fmt --check --all && cargo clippy --workspace -- -D warnings

# All unit tests
cargo test --workspace --lib

# Edge case tests with output
cargo test --workspace -- edge_case --nocapture

# Integration tests (requires build first)
cargo build -p context-graph-mcp
cargo test -- --ignored --nocapture

# Doc tests
cargo test --workspace --doc
```

---

## COMPLETION CRITERIA

Module 1 Ghost System is COMPLETE when:

- [ ] `cargo check --workspace` passes
- [ ] `cargo clippy -- -D warnings` passes (no warnings)
- [ ] `cargo test --workspace` shows 52+ tests passing
- [ ] Edge case tests print STATE BEFORE/AFTER/EVIDENCE
- [ ] Integration tests pass when run with `--ignored`
- [ ] MCP server responds correctly to initialize + tools/list
- [ ] All 5 tools return Cognitive Pulse header
- [ ] No `unwrap()` in production code paths

---

*Document Version: 2.0 (Post-Audit)*
*Last Updated: 2025-12-31*
*Audit Status: PASSED - Implementation Complete*
*Agent Instructions: Execute TASK-VAL-001 through TASK-VAL-004*
