# TASK-HOOKS-016: Create Integration Tests for Hook Lifecycle

```xml
<task_spec id="TASK-HOOKS-016" version="2.0">
<metadata>
  <title>Create Integration Tests for Hook Lifecycle</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>16</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-43</requirement_ref>
    <requirement_ref>REQ-HOOKS-44</requirement_ref>
    <requirement_ref>REQ-HOOKS-47</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-006</task_ref>
    <task_ref>TASK-HOOKS-007</task_ref>
    <task_ref>TASK-HOOKS-008</task_ref>
    <task_ref>TASK-HOOKS-009</task_ref>
    <task_ref>TASK-HOOKS-012</task_ref>
    <task_ref>TASK-HOOKS-013</task_ref>
    <task_ref>TASK-HOOKS-014</task_ref>
    <task_ref>TASK-HOOKS-015</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <estimated_hours>4.0</estimated_hours>
</metadata>

<context>
Integration tests verify that individual hook components work together correctly
through the COMPLETE lifecycle: session_start → tool_use → prompt_submit → session_end.

These tests use REAL CLI binary execution and REAL database operations - NO MOCKS.
All tests verify physical state in RocksDB after each operation.

Per constitution AP-50: Only native Claude Code hooks via settings.json.
No custom hook infrastructure.

## CRITICAL: NO BACKWARDS COMPATIBILITY
The hook system MUST fail fast with explicit error messages. Unknown hook types,
malformed JSON, or missing required fields MUST result in immediate failure with
exit code 4 (ERR_INVALID_INPUT). No silent fallbacks, no default values for
required fields, no graceful degradation.
</context>

<current_state>
## VERIFIED FILE INVENTORY (2026-01-15)

### CLI Binary Location
- ./target/release/context-graph-cli (PRIMARY)
- ./target/debug/context-graph-cli (FALLBACK)

### Hook Handlers (crates/context-graph-cli/src/commands/hooks/)
| File | Handler | Timeout | Status |
|------|---------|---------|--------|
| session_start.rs | SessionStart | 5000ms | IMPLEMENTED |
| pre_tool_use.rs | PreToolUse | 100ms (FAST PATH) | IMPLEMENTED |
| post_tool_use.rs | PostToolUse | 3000ms | IMPLEMENTED |
| user_prompt_submit.rs | UserPromptSubmit | 2000ms | IMPLEMENTED |
| session_end.rs | SessionEnd | 30000ms | IMPLEMENTED |

### Shell Scripts (.claude/hooks/) - ALL EXIST
- session_start.sh (5000ms timeout)
- pre_tool_use.sh (100ms FAST PATH timeout)
- post_tool_use.sh (3000ms timeout)
- user_prompt_submit.sh (2000ms timeout)
- session_end.sh (30000ms timeout)

### Type System (types.rs)
- HookEventType: SessionStart, PreToolUse, PostToolUse, UserPromptSubmit, SessionEnd
- HookInput: hook_type, session_id, timestamp_ms, payload (ALL REQUIRED)
- HookOutput: success, error?, consciousness_state?, ic_classification?, context_injection?, drift_metrics?, execution_time_ms
- HookPayload: Typed variants for each hook type (internally tagged enum)
- ICLevel: Healthy (>=0.9), Normal (>=0.7), Warning (>=0.5), Critical (<0.5)
- SessionEndStatus: Normal, Timeout, Error, UserAbort, Clear
- DriftMetrics: session_id, previous_session_id, identity_distance, restoration_confidence

### Existing Test Infrastructure
- tests/integration/mcp_protocol_test.rs - MCP protocol tests
- crates/context-graph-storage/tests/full_state_verification/ - REFERENCE for testing pattern
- crates/context-graph-storage/tests/storage_integration/ - Storage tests

### Database Backend
- RocksDB at ./data/context-graph.db (or CONTEXT_GRAPH_DB_PATH)
- Sessions stored in sessions column family
- Snapshots stored in snapshots column family
</current_state>

<input_context_files>
  <file purpose="hook_handlers" exists="true">crates/context-graph-cli/src/commands/hooks/mod.rs</file>
  <file purpose="hook_types" exists="true">crates/context-graph-cli/src/commands/hooks/types.rs</file>
  <file purpose="hook_args" exists="true">crates/context-graph-cli/src/commands/hooks/args.rs</file>
  <file purpose="session_start" exists="true">crates/context-graph-cli/src/commands/hooks/session_start.rs</file>
  <file purpose="session_end" exists="true">crates/context-graph-cli/src/commands/hooks/session_end.rs</file>
  <file purpose="pre_tool_use" exists="true">crates/context-graph-cli/src/commands/hooks/pre_tool_use.rs</file>
  <file purpose="post_tool_use" exists="true">crates/context-graph-cli/src/commands/hooks/post_tool_use.rs</file>
  <file purpose="user_prompt_submit" exists="true">crates/context-graph-cli/src/commands/hooks/user_prompt_submit.rs</file>
  <file purpose="shell_scripts" exists="true">.claude/hooks/</file>
  <file purpose="settings" exists="true">.claude/settings.json</file>
  <file purpose="test_pattern_reference" exists="true">crates/context-graph-storage/tests/full_state_verification/</file>
  <file purpose="constitution" exists="true">docs2/constitution.yaml</file>
</input_context_files>

<prerequisites>
  <check>CLI binary compiled: cargo build --release -p context-graph-cli</check>
  <check>RocksDB path writable: $CONTEXT_GRAPH_DB_PATH or ./data/context-graph.db</check>
  <check>Shell scripts executable: chmod +x .claude/hooks/*.sh</check>
  <check>All hook handlers return valid HookOutput JSON</check>
  <check>tempfile crate available in dev-dependencies</check>
</prerequisites>

<scope>
  <in_scope>
    - Session lifecycle integration tests (start → tools → end)
    - Identity snapshot/restore integration tests with drift verification
    - Hook timeout verification tests per constitution budgets
    - Exit code verification tests (0, 2, 3, 4)
    - Error propagation tests with stderr capture
    - Concurrent tool hook execution tests
    - Physical database verification after operations
    - Test fixtures with REAL data (not mocks)
  </in_scope>
  <out_of_scope>
    - End-to-end Claude Code tests (TASK-HOOKS-017)
    - Performance benchmarks (separate task)
    - MCP server tests (use existing tests/integration/mcp_protocol_test.rs)
  </out_of_scope>
</scope>

<hook_input_contract>
## HookInput JSON Format (REQUIRED FIELDS - NO DEFAULTS)
All fields are REQUIRED. Missing fields MUST cause exit code 4.

```json
{
  "hook_type": "session_start|pre_tool_use|post_tool_use|user_prompt_submit|session_end",
  "session_id": "string (non-empty)",
  "timestamp_ms": 1705312345678,
  "payload": { ... }
}
```

## Payload Variants (internally tagged with "type" field)

### SessionStart Payload
```json
{
  "type": "session_start",
  "cwd": "/home/user/project",
  "source": "cli|ide|resume",
  "previous_session_id": "optional-string"
}
```

### PreToolUse Payload (FAST PATH - 100ms max, NO DB ACCESS)
```json
{
  "type": "pre_tool_use",
  "tool_name": "Write|Edit|Bash|Read|etc",
  "tool_input": {},
  "tool_use_id": "unique-id"
}
```

### PostToolUse Payload
```json
{
  "type": "post_tool_use",
  "tool_name": "Write",
  "tool_input": {},
  "tool_response": "string result",
  "tool_use_id": "unique-id"
}
```

### UserPromptSubmit Payload
```json
{
  "type": "user_prompt_submit",
  "prompt": "user's input text",
  "context": []
}
```

### SessionEnd Payload
```json
{
  "type": "session_end",
  "duration_ms": 3600000,
  "status": "normal|timeout|error|user_abort|clear",
  "reason": "optional-string"
}
```
</hook_input_contract>

<exit_codes>
## Exit Code Specification (AP-26)
| Code | Meaning | When |
|------|---------|------|
| 0 | Success | Hook executed correctly |
| 1 | General error | Unexpected runtime error |
| 2 | Timeout exceeded | Hook exceeded budget |
| 3 | Database error | Connection or query failure |
| 4 | Invalid input | Malformed JSON, missing required fields |

## FAIL FAST REQUIREMENTS
- Missing hook_type: exit 4 immediately
- Empty session_id: exit 4 immediately
- timestamp_ms <= 0: exit 4 immediately
- Unknown hook_type value: exit 4 immediately
- Malformed JSON: exit 4 immediately
- NO SILENT FAILURES - all errors must be logged to stderr as JSON
</exit_codes>

<timeout_budgets>
## Timeout Requirements (Constitution + TECH-HOOKS.md)
| Hook | Timeout | Notes |
|------|---------|-------|
| PreToolUse | 100ms | FAST PATH - uses IdentityCache only, NO DB |
| PostToolUse | 3000ms | Async learning allowed |
| UserPromptSubmit | 2000ms | Context injection |
| SessionStart | 5000ms | Identity restoration with drift |
| SessionEnd | 30000ms | Full persistence with consolidation |
</timeout_budgets>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/tests/integration/mod.rs">
      mod hook_lifecycle_test;
      mod identity_integration_test;
      mod exit_code_test;
      mod timeout_test;

      // Re-exports for test discovery
      pub use hook_lifecycle_test::*;
      pub use identity_integration_test::*;
    </signature>
    <signature file="crates/context-graph-cli/tests/integration/hook_lifecycle_test.rs">
      use std::process::{Command, Stdio};
      use tempfile::TempDir;

      /// Full session lifecycle: start → pre_tool → post_tool → prompt → end
      #[tokio::test]
      async fn test_session_lifecycle_full_flow();

      /// Multiple tool uses in single session with state accumulation
      #[tokio::test]
      async fn test_multiple_tool_uses_in_session();

      /// Consciousness state injection verification
      #[tokio::test]
      async fn test_consciousness_state_injection();

      /// Concurrent tool hooks don't interfere
      #[tokio::test]
      async fn test_concurrent_tool_hooks();

      // Test helpers
      fn invoke_hook(hook_type: &str, input: &str) -> (i32, String, String);
      fn create_session_start_input(session_id: &str, cwd: &str) -> String;
      fn create_pre_tool_input(session_id: &str, tool_name: &str) -> String;
      fn create_post_tool_input(session_id: &str, tool_name: &str, response: &str) -> String;
      fn create_session_end_input(session_id: &str, duration_ms: u64) -> String;
    </signature>
    <signature file="crates/context-graph-cli/tests/integration/identity_integration_test.rs">
      use tempfile::TempDir;
      use serde_json::Value;

      /// Identity snapshot is created on session_end
      #[tokio::test]
      async fn test_identity_snapshot_created();

      /// Identity restored on session_start with previous_session_id
      #[tokio::test]
      async fn test_identity_restoration_with_drift();

      /// Drift metrics computed correctly
      #[tokio::test]
      async fn test_drift_metrics_computation();

      /// IC classification matches thresholds
      #[tokio::test]
      async fn test_ic_classification_thresholds();

      // Helpers
      fn verify_snapshot_in_db(db_path: &Path, session_id: &str) -> bool;
      fn get_drift_metrics(output: &str) -> Option<DriftMetrics>;
    </signature>
    <signature file="crates/context-graph-cli/tests/integration/exit_code_test.rs">
      /// Exit code 4 on missing hook_type
      #[test]
      fn test_exit_code_4_missing_hook_type();

      /// Exit code 4 on empty session_id
      #[test]
      fn test_exit_code_4_empty_session_id();

      /// Exit code 4 on invalid timestamp
      #[test]
      fn test_exit_code_4_invalid_timestamp();

      /// Exit code 4 on unknown hook_type
      #[test]
      fn test_exit_code_4_unknown_hook_type();

      /// Exit code 4 on malformed JSON
      #[test]
      fn test_exit_code_4_malformed_json();

      /// Exit code 0 on valid input
      #[test]
      fn test_exit_code_0_valid_input();
    </signature>
    <signature file="crates/context-graph-cli/tests/integration/timeout_test.rs">
      /// PreToolUse completes within 100ms budget
      #[tokio::test]
      async fn test_pre_tool_use_fast_path_timing();

      /// SessionEnd can use full 30000ms budget
      #[tokio::test]
      async fn test_session_end_long_timeout();

      /// Timeout produces exit code 2
      #[tokio::test]
      async fn test_timeout_exit_code();
    </signature>
  </signatures>

  <constraints>
    - NO MOCK DATA - all tests use real CLI binary execution
    - NO MOCK DATABASES - use tempfile::TempDir for isolated RocksDB instances
    - Tests must clean up after themselves (automatic with TempDir)
    - Tests must be independent (no shared state between tests)
    - Tests must complete within 60 seconds each (allows for SessionEnd's 30s budget)
    - Failed tests MUST provide: exit code, stdout, stderr, input JSON
    - Tests must run in CI environment (no external dependencies)
    - All assertions must verify PHYSICAL state in database
  </constraints>

  <verification>
    - cargo test --package context-graph-cli --test integration
    - All tests pass with release binary
    - No test pollution (can run in any order with cargo test --test-threads=1)
    - Exit codes match specification
    - Stderr contains valid JSON error messages
  </verification>
</definition_of_done>

<full_state_verification>
## Source of Truth
- Constitution Reference: AP-50, AP-26, IDENTITY-002, GWT-003
- Type Definitions: crates/context-graph-cli/src/commands/hooks/types.rs
- Shell Scripts: .claude/hooks/*.sh
- Database: RocksDB at $CONTEXT_GRAPH_DB_PATH

## Execute & Inspect Pattern
For each test operation:
1. Execute hook via CLI (echo JSON | ./target/release/context-graph-cli hooks <cmd> --stdin)
2. Capture exit code, stdout, stderr
3. Parse stdout as HookOutput JSON
4. Open RocksDB with separate reader and verify physical bytes
5. Compare actual state vs expected state
6. Log detailed diagnostics on failure

## Boundary & Edge Case Audit (3 minimum REQUIRED)

### Edge Case 1: Empty session_id (Exit Code Verification)
SYNTHETIC INPUT:
```json
{"hook_type":"session_start","session_id":"","timestamp_ms":1705312345678,"payload":{"type":"session_start","cwd":"/tmp","source":"cli"}}
```
EXPECTED EXIT CODE: 4
EXPECTED STDERR: Contains "session_id cannot be empty"
BEFORE STATE: No session exists
AFTER STATE: No session created, error logged to stderr

### Edge Case 2: Session Restoration with Maximum Drift
SYNTHETIC INPUT (first session):
```json
{"hook_type":"session_start","session_id":"test-old-001","timestamp_ms":1705312345678,"payload":{"type":"session_start","cwd":"/tmp","source":"cli"}}
```
Then end session with:
```json
{"hook_type":"session_end","session_id":"test-old-001","timestamp_ms":1705312355678,"payload":{"type":"session_end","duration_ms":10000,"status":"normal"}}
```
Then new session with previous_session_id:
```json
{"hook_type":"session_start","session_id":"test-new-001","timestamp_ms":1705400000000,"payload":{"type":"session_start","cwd":"/different","source":"resume","previous_session_id":"test-old-001"}}
```
EXPECTED: drift_metrics present in output with identity_distance > 0

### Edge Case 3: PreToolUse Fast Path (No Database Access)
SYNTHETIC INPUT:
```json
{"hook_type":"pre_tool_use","session_id":"test-fast-001","timestamp_ms":1705312345678,"payload":{"type":"pre_tool_use","tool_name":"Read","tool_input":{"file_path":"/tmp/test.rs"},"tool_use_id":"tool-001"}}
```
EXPECTED: Completes in < 100ms
EXPECTED: execution_time_ms < 100 in output
VERIFICATION: Time the actual execution, assert < 100ms

### Edge Case 4: Concurrent Tool Hooks (Race Condition Test)
Run 10 parallel PreToolUse hooks with same session_id but different tool_use_id
EXPECTED: All 10 complete successfully (exit code 0)
EXPECTED: No database corruption
VERIFICATION: After all complete, session state is consistent

## Evidence of Success Logging
Each test MUST log structured output:
```json
{"test":"test_name","hook_type":"session_start","session_id":"xxx","exit_code":0,"execution_time_ms":42,"stdout_size":256,"stderr_size":0,"db_verified":true}
```
</full_state_verification>

<manual_verification>
## Database Verification
After SessionStart + SessionEnd sequence:
```bash
# Verify session exists in RocksDB
./target/release/context-graph-cli db inspect --column-family sessions --key "test-session-001"
```

## Snapshot Verification
After SessionEnd with identity persistence:
```bash
# Verify snapshot stored
./target/release/context-graph-cli db inspect --column-family snapshots --prefix "test-session-001"
```

## Output Verification
After each hook:
```bash
# Parse and validate output JSON
echo '<input>' | ./target/release/context-graph-cli hooks session-start --stdin --format json | jq .
```

## Drift Metrics Verification
After session restoration:
```bash
# Check drift_metrics in output
echo '<session_start_with_prev>' | ./target/release/context-graph-cli hooks session-start --stdin | jq '.drift_metrics'
```
Expected fields: session_id, previous_session_id, identity_distance, restoration_confidence
</manual_verification>

<test_commands>
  <command desc="Run all integration tests">cargo test --package context-graph-cli --test integration -- --test-threads=1</command>
  <command desc="Run hook lifecycle tests">cargo test --package context-graph-cli --test integration hook_lifecycle</command>
  <command desc="Run identity tests">cargo test --package context-graph-cli --test integration identity</command>
  <command desc="Run exit code tests">cargo test --package context-graph-cli --test integration exit_code</command>
  <command desc="Run timeout tests">cargo test --package context-graph-cli --test integration timeout</command>
  <command desc="Run with verbose output">cargo test --package context-graph-cli --test integration -- --nocapture</command>
  <command desc="Build release binary first">cargo build --release -p context-graph-cli</command>
</test_commands>

<files_to_create>
  <file path="crates/context-graph-cli/tests/integration/mod.rs">
    Integration test module root - declares submodules
  </file>
  <file path="crates/context-graph-cli/tests/integration/hook_lifecycle_test.rs">
    Full session lifecycle tests with physical DB verification
  </file>
  <file path="crates/context-graph-cli/tests/integration/identity_integration_test.rs">
    Identity snapshot/restore tests with drift metrics
  </file>
  <file path="crates/context-graph-cli/tests/integration/exit_code_test.rs">
    Exit code verification for all error conditions
  </file>
  <file path="crates/context-graph-cli/tests/integration/timeout_test.rs">
    Timeout budget verification tests
  </file>
  <file path="crates/context-graph-cli/tests/integration/helpers.rs">
    Shared test utilities: invoke_hook, create_*_input, verify_db_state
  </file>
</files_to_create>

<files_to_modify>
  <!-- None - all new test files -->
</files_to_modify>

<implementation_notes>
## Test Execution Pattern
Each test should follow this pattern:
```rust
#[tokio::test]
async fn test_example() {
    // 1. Create temp directory for isolated DB
    let temp_dir = TempDir::new().unwrap();
    std::env::set_var("CONTEXT_GRAPH_DB_PATH", temp_dir.path());

    // 2. Create synthetic input JSON
    let input = create_session_start_input("test-001", "/tmp");

    // 3. Invoke CLI via process
    let (exit_code, stdout, stderr) = invoke_hook("session-start", &input);

    // 4. Verify exit code
    assert_eq!(exit_code, 0, "Expected success, stderr: {}", stderr);

    // 5. Parse output
    let output: HookOutput = serde_json::from_str(&stdout).unwrap();
    assert!(output.success);

    // 6. Verify physical state in DB
    let db = RocksDbMemex::open(temp_dir.path()).unwrap();
    let session = db.get_session("test-001").unwrap();
    assert!(session.is_some());

    // 7. Log evidence
    println!("{{\"test\":\"test_example\",\"exit_code\":{},\"db_verified\":true}}", exit_code);
}
```

## Helper Function: invoke_hook
```rust
fn invoke_hook(hook_cmd: &str, input: &str) -> (i32, String, String) {
    let output = Command::new("./target/release/context-graph-cli")
        .args(["hooks", hook_cmd, "--stdin", "--format", "json"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            child.stdin.take().unwrap().write_all(input.as_bytes())?;
            child.wait_with_output()
        })
        .expect("Failed to execute CLI");

    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}
```

## Synthetic Test Data Requirements
ALL test inputs must be:
- Deterministic (same input = same output for pure functions)
- Isolated (use unique session_id per test)
- Verifiable (known expected outputs documented)
- Real format (valid HookInput JSON per types.rs)

## NO MOCK DATA RULE
- Use real CLI binary (./target/release/context-graph-cli)
- Use real RocksDB in temp directories
- Use real JSON serialization/deserialization
- DO NOT stub database operations
- DO NOT mock hook handlers
</implementation_notes>

<no_mock_data>
## Real Data Requirements
- session_id: Use unique test-prefixed IDs (test-001, test-lifecycle-001, etc.)
- timestamp_ms: Use real Unix timestamps (chrono::Utc::now().timestamp_millis())
- tool_name: Use actual tool names from Claude Code (Write, Edit, Bash, Read, Glob, Grep)
- tool_input: Use realistic tool parameters (actual file paths in temp dirs)
- NO synthetic/mock data that bypasses real code paths
- Tests MUST use real CLI binary, not mocks
- Tests MUST verify state in real RocksDB, not mock storage
</no_mock_data>

<reference_patterns>
## Existing Test Pattern to Follow
See: crates/context-graph-storage/tests/full_state_verification/

Key patterns:
1. Create temp directory with TempDir::new()
2. Initialize real store pointing to temp dir
3. Execute operations
4. Verify physical bytes with get_raw_bytes()
5. Log structured output with [PASS]/[FAIL] markers
6. Print detailed diagnostics on failure

Example from edge_case_tests.rs:
```rust
println!("\n================================================================================");
println!("EDGE CASE TEST: Minimal Valid Fingerprint");
println!("================================================================================\n");

let temp_dir = TempDir::new().expect("Failed to create temp dir");
let store = create_test_store(&temp_dir);

// ... test operations ...

// Physical verification
let raw = store.get_raw_bytes(CF_FINGERPRINTS, &key).expect("Failed to read");
assert!(raw.is_some(), "Fingerprint not stored!");

println!("\n[PASS] Minimal fingerprint edge case successful");
```
</reference_patterns>
</task_spec>
```
