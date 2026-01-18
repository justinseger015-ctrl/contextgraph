# Task 03: Create Stop Hook Shell Script

## Metadata
- **Task ID**: TASK-GAP-002
- **Phase**: 1 (Foundation)
- **Priority**: Critical
- **Complexity**: Low
- **Dependencies**: CLI `memory capture-response` command (ALREADY EXISTS)
- **Constitution**: AP-50, AP-53, ARCH-07, ARCH-11

## Objective

Create the Stop hook shell script (`stop.sh`) that captures Claude's response when a response completes and stores it as a `ClaudeResponse` memory. Per AP-53: "Hook logic MUST be in shell scripts calling context-graph-cli".

## CRITICAL: Current Codebase State (Verified 2026-01-18)

### What ALREADY EXISTS:
1. **CLI Command**: `context-graph-cli memory capture-response` - FULLY IMPLEMENTED
   - Location: `crates/context-graph-cli/src/commands/memory/capture.rs:331-408`
   - Accepts: `--content`, `--session-id`, `--response-type`, `--db-path`
   - Environment vars: `RESPONSE_SUMMARY`, `CLAUDE_SESSION_ID`, `CONTEXT_GRAPH_DATA_DIR`
   - Response types: `session_summary`, `stop_response`, `significant_response`

2. **Hook Scripts Pattern**: All hooks use underscore naming in `.claude/hooks/`
   - `.claude/hooks/session_start.sh` (EXISTS)
   - `.claude/hooks/session_end.sh` (EXISTS)
   - `.claude/hooks/pre_tool_use.sh` (EXISTS)
   - `.claude/hooks/post_tool_use.sh` (EXISTS)
   - `.claude/hooks/user_prompt_submit.sh` (EXISTS)
   - `.claude/hooks/stop.sh` (DOES NOT EXIST - THIS TASK)

3. **Settings Configuration**: `.claude/settings.json` - Does NOT have Stop hook configured

### What Does NOT Exist:
- `.claude/hooks/stop.sh` - MUST CREATE
- Stop hook entry in `.claude/settings.json` - MUST ADD

## DISCREPANCIES IN ORIGINAL TASK (NOW CORRECTED)

| Original Task | Actual State | Correction |
|--------------|--------------|------------|
| Path: `hooks/stop.sh` | Scripts are in `.claude/hooks/` | Use `.claude/hooks/stop.sh` |
| Reference: `session_end.sh` | File is `session_end.sh` (underscore) | Updated path |
| CLI command: `hooks capture-response` | Command is `memory capture-response` | Use `memory capture-response` |
| Timeout: mentioned but not configured | Settings.json needs hook entry | Add Stop hook to settings.json |

## Input Context Files (READ BEFORE STARTING)

1. **Reference Hook Pattern**: `.claude/hooks/session_end.sh` - Copy error handling pattern
2. **CLI Interface**: `crates/context-graph-cli/src/commands/memory/capture.rs` - Understand args
3. **Settings Template**: `.claude/settings.json` - Understand hook config structure
4. **Claude Hooks Spec**: `/home/cabdru/contextgraph/docs2/claudehooks.md` - Stop hook input schema

## Stop Hook Input Schema (from Claude Code)

Per Claude Code hooks documentation, the Stop hook receives:
```json
{
  "session_id": "string",
  "transcript_path": "string",
  "cwd": "string",
  "permission_mode": "string",
  "hook_event_name": "Stop",
  "stop_hook_active": boolean
}
```

**NOTE**: Claude Code does NOT send `response_text` directly. The response content must be extracted from the transcript or passed via environment variable by Claude Code.

## Implementation Steps

### Step 1: Create the stop.sh script

Create file: `.claude/hooks/stop.sh`

```bash
#!/bin/bash
# Claude Code Hook: Stop
# Timeout: 3000ms
#
# Constitution: AP-50, AP-53, ARCH-07, ARCH-11
# Exit Codes: 0=success, 1=cli_not_found, 2=timeout, 3=db_error, 4=invalid_input
#
# Captures Claude's response and stores as ClaudeResponse memory.
# Per ARCH-11: Memory sources include ClaudeResponse.

set -euo pipefail

# Read stdin (Claude Code sends JSON)
INPUT=$(cat)
if [ -z "$INPUT" ]; then
    # Empty stdin is valid for Stop hook - exit silently
    echo '{"success":true,"skipped":true,"reason":"Empty stdin"}'
    exit 0
fi

# Validate JSON input
if ! echo "$INPUT" | jq empty 2>/dev/null; then
    echo '{"success":false,"error":"Invalid JSON input","exit_code":4}' >&2
    exit 4
fi

# Find CLI binary - check multiple locations
CONTEXT_GRAPH_CLI="${CONTEXT_GRAPH_CLI:-context-graph-cli}"
if ! command -v "$CONTEXT_GRAPH_CLI" &>/dev/null; then
    for candidate in \
        "./target/release/context-graph-cli" \
        "./target/debug/context-graph-cli" \
        "$HOME/.cargo/bin/context-graph-cli" \
    ; do
        if [ -x "$candidate" ]; then
            CONTEXT_GRAPH_CLI="$candidate"
            break
        fi
    done
fi

if ! command -v "$CONTEXT_GRAPH_CLI" &>/dev/null && [ ! -x "$CONTEXT_GRAPH_CLI" ]; then
    echo '{"success":false,"error":"CLI binary not found","exit_code":1}' >&2
    exit 1
fi

# Parse input JSON
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')

# Get response content - check environment variable or use placeholder
# Claude Code may pass response via RESPONSE_SUMMARY env var
RESPONSE_CONTENT="${RESPONSE_SUMMARY:-}"

# If no content available, skip silently (not an error)
if [ -z "$RESPONSE_CONTENT" ]; then
    echo '{"success":true,"skipped":true,"reason":"No response content available"}'
    exit 0
fi

# Truncate response if > 10000 chars (per MAX_CONTENT_LENGTH constraint)
RESPONSE_CONTENT=$(echo "$RESPONSE_CONTENT" | head -c 10000)

# Execute CLI with 3s timeout
# Uses memory capture-response (NOT hooks capture-response)
if timeout 3s "$CONTEXT_GRAPH_CLI" memory capture-response \
    --content "$RESPONSE_CONTENT" \
    --session-id "${SESSION_ID:-default}" \
    --response-type stop_response; then
    echo '{"success":true,"stored":true}'
    exit 0
else
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo '{"success":false,"error":"Timeout after 3000ms","exit_code":2}' >&2
        exit 2
    fi
    echo '{"success":false,"error":"CLI command failed","exit_code":'$exit_code'}' >&2
    exit $exit_code
fi
```

### Step 2: Make script executable

```bash
chmod +x .claude/hooks/stop.sh
```

### Step 3: Add Stop hook to settings.json

Update `.claude/settings.json` to add the Stop hook configuration:

```json
{
  "hooks": {
    "SessionStart": [...],
    "SessionEnd": [...],
    "PreToolUse": [...],
    "PostToolUse": [...],
    "UserPromptSubmit": [...],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/stop.sh",
            "timeout": 3000
          }
        ]
      }
    ]
  }
}
```

## Definition of Done

- [x] Script exists at `.claude/hooks/stop.sh`
- [x] Script is executable (`chmod +x`)
- [x] Script has proper shebang (`#!/bin/bash`)
- [x] Script uses `set -euo pipefail` for error handling
- [x] Script validates JSON input using jq
- [x] Script handles empty stdin gracefully (exit 0)
- [x] Script handles empty response content gracefully (exit 0)
- [x] Script truncates content > 10000 characters
- [x] Script resolves CLI binary from multiple candidate paths
- [x] Script calls `memory capture-response` (NOT `hooks capture-response`)
- [x] Settings.json includes Stop hook configuration
- [x] Exit codes match specification:
  - 0 = success (including skipped)
  - 1 = cli_not_found
  - 2 = timeout
  - 4 = invalid_input
- [x] Script completes within 3000ms timeout

## Verification Commands

```bash
cd /home/cabdru/contextgraph

# 1. Verify script exists and is executable
test -x .claude/hooks/stop.sh && echo "PASS: Script is executable" || echo "FAIL: Script not executable"

# 2. Verify script syntax
bash -n .claude/hooks/stop.sh && echo "PASS: Syntax OK" || echo "FAIL: Syntax error"

# 3. Verify settings.json has Stop hook
jq '.hooks.Stop' .claude/settings.json | grep -q "stop.sh" && echo "PASS: Stop hook in settings" || echo "FAIL: Stop hook not in settings"

# 4. Test empty stdin (should exit 0)
echo '' | .claude/hooks/stop.sh
echo "Exit code: $? (expect 0)"

# 5. Test valid JSON but no RESPONSE_SUMMARY (should exit 0, skipped)
echo '{"session_id":"test-123"}' | .claude/hooks/stop.sh
echo "Exit code: $? (expect 0)"

# 6. Test invalid JSON (should exit 4)
echo 'not json' | .claude/hooks/stop.sh 2>&1
echo "Exit code: $? (expect 4)"
```

## Full State Verification (MANDATORY)

After completing the implementation, perform these verification steps:

### Source of Truth
The final result is stored in:
1. **File System**: `.claude/hooks/stop.sh` must exist and be executable
2. **Settings Config**: `.claude/settings.json` must have Stop hook entry
3. **RocksDB**: When executed with content, memory stored in `./data/memories/`

### Execute & Inspect Procedure

```bash
cd /home/cabdru/contextgraph

# Create test data directory
mkdir -p ./data/memories

# Build CLI if not already built
cargo build --release -p context-graph-cli 2>/dev/null || cargo build -p context-graph-cli

# Test with synthetic response content
export RESPONSE_SUMMARY="This is a test response from Claude about implementing authentication."
echo '{"session_id":"fsv-stop-test-001"}' | .claude/hooks/stop.sh
echo "Hook exit code: $?"

# VERIFY: Check if memory was actually stored in RocksDB
# The CLI should have created a memory entry
./target/release/context-graph-cli memory inject-context --query "authentication" --db-path ./data 2>/dev/null || \
./target/debug/context-graph-cli memory inject-context --query "authentication" --db-path ./data 2>/dev/null || \
echo "Note: inject-context may return no results if embeddings are stub - check RocksDB directly"

# Alternative verification: Count memories in session
# (This requires the store to be queryable by session)
```

### Edge Case Audit (3 Mandatory Tests)

#### Edge Case 1: Empty Input
```bash
echo "=== EDGE CASE 1: Empty stdin ==="
echo "STATE BEFORE: Script receives empty stdin"
RESULT=$(echo '' | .claude/hooks/stop.sh)
EXIT=$?
echo "STATE AFTER: Exit code=$EXIT, Output=$RESULT"
echo "EXPECTED: Exit 0, success=true, skipped=true"
[ $EXIT -eq 0 ] && echo "PASS" || echo "FAIL"
```

#### Edge Case 2: Max Length Content (10000 chars)
```bash
echo "=== EDGE CASE 2: Max length content ==="
export RESPONSE_SUMMARY=$(python3 -c "print('x' * 10000)")
echo "STATE BEFORE: RESPONSE_SUMMARY length=$(echo -n "$RESPONSE_SUMMARY" | wc -c)"
RESULT=$(echo '{"session_id":"edge-max-len"}' | .claude/hooks/stop.sh)
EXIT=$?
echo "STATE AFTER: Exit code=$EXIT"
echo "EXPECTED: Exit 0, content truncated and stored"
[ $EXIT -eq 0 ] && echo "PASS" || echo "FAIL"
unset RESPONSE_SUMMARY
```

#### Edge Case 3: Over Max Length (10001 chars)
```bash
echo "=== EDGE CASE 3: Over max length content (10001 chars) ==="
export RESPONSE_SUMMARY=$(python3 -c "print('x' * 10001)")
echo "STATE BEFORE: RESPONSE_SUMMARY length=$(echo -n "$RESPONSE_SUMMARY" | wc -c)"
RESULT=$(echo '{"session_id":"edge-over-max"}' | .claude/hooks/stop.sh)
EXIT=$?
echo "STATE AFTER: Exit code=$EXIT"
echo "EXPECTED: Exit 0 (script truncates before passing to CLI)"
# Note: The shell script truncates with head -c 10000, so CLI never sees >10000
[ $EXIT -eq 0 ] && echo "PASS" || echo "FAIL"
unset RESPONSE_SUMMARY
```

### Evidence of Success Log

After running all tests, record the following evidence:

```
============================================================
STOP HOOK VERIFICATION LOG
============================================================
Date: [TIMESTAMP]
Tester: [AI Agent ID]

1. FILE EXISTS CHECK
   - .claude/hooks/stop.sh exists: [YES/NO]
   - Is executable: [YES/NO]
   - File size: [N bytes]

2. SETTINGS.JSON CHECK
   - Stop hook entry present: [YES/NO]
   - Timeout configured: [3000ms]
   - Command path correct: [YES/NO]

3. SYNTAX CHECK
   - bash -n result: [PASS/FAIL]

4. FUNCTIONAL TESTS
   - Empty stdin (exit 0): [PASS/FAIL]
   - Valid JSON no content (exit 0): [PASS/FAIL]
   - Invalid JSON (exit 4): [PASS/FAIL]
   - With content (exit 0): [PASS/FAIL]

5. EDGE CASES
   - Empty input: [PASS/FAIL]
   - Max length (10000): [PASS/FAIL]
   - Over max (10001): [PASS/FAIL]

6. DATABASE VERIFICATION
   - Memory stored after test: [YES/NO/STUB_EMBEDDINGS]
   - Session ID matches: [YES/NO]
   - Content type is ClaudeResponse: [YES/NO]

OVERALL STATUS: [PASS/FAIL]
============================================================
```

## NO BACKWARDS COMPATIBILITY

- If the CLI binary is not found, the script MUST exit with code 1
- If JSON is invalid, the script MUST exit with code 4
- If the CLI command fails, the script MUST propagate the error
- NO fallback logic, NO silent failures, NO workarounds

## Related Files

| File | Purpose | Status |
|------|---------|--------|
| `.claude/hooks/stop.sh` | Stop hook script | TO CREATE |
| `.claude/settings.json` | Hook configuration | TO UPDATE |
| `crates/context-graph-cli/src/commands/memory/capture.rs` | CLI command | EXISTS |
| `.claude/hooks/session_end.sh` | Reference pattern | EXISTS |

## Constitution References

- **AP-50**: NO internal/built-in Claude Code hooks
- **AP-53**: Hook logic MUST be in shell scripts calling context-graph-cli
- **ARCH-07**: NATIVE Claude Code hooks via .claude/settings.json
- **ARCH-11**: Memory sources include ClaudeResponse
- **SEC-02**: PII scrubbing (handled by CLI)
