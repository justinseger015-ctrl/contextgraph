# Claude Code Hooks: The Complete Guide

> Comprehensive reference for Claude Code hooks system - lifecycle events, JSON schemas, tool input formats, exit codes, matchers, environment variables, and advanced patterns. (January 2026)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Hook Events Reference](#2-hook-events-reference)
3. [Configuration](#3-configuration)
4. [JSON Input Schemas](#4-json-input-schemas)
5. [JSON Output Schemas](#5-json-output-schemas)
6. [Tool Input Schemas](#6-tool-input-schemas)
7. [Matchers](#7-matchers)
8. [Exit Codes & Control Flow](#8-exit-codes--control-flow)
9. [Environment Variables](#9-environment-variables)
10. [Hook Types: Command vs Prompt](#10-hook-types-command-vs-prompt)
11. [Advanced Patterns](#11-advanced-patterns)
12. [Security Best Practices](#12-security-best-practices)
13. [Debugging](#13-debugging)
14. [Complete Examples](#14-complete-examples)
15. [Known Issues & Workarounds](#15-known-issues--workarounds)
16. [Version History](#16-version-history)
17. [Sources](#17-sources)

---

## 1. Overview

Hooks are shell commands or LLM prompts that execute automatically at specific lifecycle points in Claude Code. They enable deterministic automation without relying on LLM decisions.

### Key Characteristics

| Feature | Description |
|---------|-------------|
| **Execution** | Shell commands or LLM prompts |
| **Trigger** | Lifecycle events (tool use, session, etc.) |
| **Control** | Can block, allow, or modify actions |
| **Input** | JSON via stdin |
| **Output** | JSON via stdout, exit codes |
| **Timeout** | 60 seconds default (configurable) |
| **Parallelization** | Multiple matching hooks run in parallel |
| **Deduplication** | Identical commands auto-deduplicated |

### When to Use Hooks

| Use Case | Hook Type |
|----------|-----------|
| Validate/block dangerous operations | PreToolUse |
| Auto-approve safe commands | PermissionRequest |
| Format code after edits | PostToolUse |
| Inject session context | SessionStart |
| Backup transcripts | PreCompact |
| Force task completion | Stop |
| Validate subagent work | SubagentStop |
| Log all activity | Any hook type |

---

## 2. Hook Events Reference

Claude Code provides **10 hook events** covering the complete session lifecycle:

### 2.1 PreToolUse

**When**: Before any tool executes
**Blocking**: Yes (exit code 2)
**Matchers**: Supported
**Version**: Original release

**Capabilities**:
- Block dangerous operations
- Auto-approve safe operations
- Modify tool inputs before execution (v2.0.10+)
- Log all tool calls

**Input Fields**:
```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/directory",
  "permission_mode": "default",
  "hook_event_name": "PreToolUse",
  "tool_name": "Bash",
  "tool_input": {
    "command": "npm test",
    "description": "Run tests"
  }
}
```

**Decision Options**:
- `"allow"` - Bypass permission dialog, execute tool
- `"deny"` - Block tool, show reason to Claude
- `"ask"` - Show permission dialog to user

---

### 2.2 PostToolUse

**When**: After tool completes successfully
**Blocking**: No (tool already executed)
**Matchers**: Supported
**Version**: Original release

**Capabilities**:
- Format/lint generated code
- Validate tool results
- Log outcomes
- Provide feedback to Claude

**Input Fields**:
```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/directory",
  "permission_mode": "default",
  "hook_event_name": "PostToolUse",
  "tool_name": "Write",
  "tool_input": {
    "file_path": "/path/to/file.ts",
    "content": "..."
  },
  "tool_response": {
    "success": true,
    "filePath": "/path/to/file.ts"
  }
}
```

**Note**: PostToolUse cannot prevent execution (it already happened), but can provide feedback to Claude via `{"decision": "block", "reason": "explanation"}`.

---

### 2.3 PermissionRequest (v2.0.45+)

**When**: Permission dialog about to be shown
**Blocking**: Yes
**Matchers**: Supported
**Version**: 2.0.45

**Capabilities**:
- Auto-approve safe operations
- Auto-deny dangerous operations
- Modify inputs before approval

**Input Fields**:
```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/directory",
  "permission_mode": "default",
  "hook_event_name": "PermissionRequest",
  "tool_name": "Bash",
  "tool_input": {
    "command": "npm test"
  }
}
```

**Decision Options**:
- `"allow"` - Auto-approve, skip user dialog
- `"deny"` - Auto-deny, skip user dialog
- `"ask"` - Show permission dialog (default)

---

### 2.4 UserPromptSubmit

**When**: User submits a prompt (before Claude processes)
**Blocking**: Yes (exit code 2)
**Matchers**: Not applicable
**Version**: Original release

**Capabilities**:
- Validate user input
- Add context to prompts
- Log all user messages
- Block inappropriate content

**Input Fields**:
```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/directory",
  "permission_mode": "default",
  "hook_event_name": "UserPromptSubmit",
  "prompt": "User's message text"
}
```

**Special Behavior**: stdout is added to context (visible to Claude).

---

### 2.5 Stop

**When**: Claude finishes responding
**Blocking**: Yes (can force continuation)
**Matchers**: Not applicable
**Version**: Original release

**Capabilities**:
- Verify task completion
- Force continuation if incomplete
- Log session activity
- Trigger notifications

**Input Fields**:
```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/directory",
  "permission_mode": "default",
  "hook_event_name": "Stop",
  "stop_hook_active": false
}
```

**`stop_hook_active`**: `true` if Claude is already continuing due to a previous Stop hook (prevents infinite loops).

**Decision Options**:
- `"block"` with `"continue": true` - Force Claude to continue
- `"approve"` with `"continue": false` - Allow stop

---

### 2.6 SubagentStop (v1.0.41+)

**When**: Subagent (Task tool) finishes
**Blocking**: Yes
**Matchers**: Not applicable
**Version**: 1.0.41

**Capabilities**:
- Validate subagent output quality
- Force subagent to continue
- Log subagent activity

**Input Fields**:
```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/directory",
  "permission_mode": "default",
  "hook_event_name": "SubagentStop",
  "stop_hook_active": false
}
```

**Usage**: Identical to Stop hook but triggers for subagents.

---

### 2.7 SessionStart

**When**: New session begins or existing session resumes
**Blocking**: No
**Matchers**: Not applicable
**Version**: Original release

**Capabilities**:
- Load dynamic context
- Initialize environment
- Set up persistent variables
- Log session start

**Input Fields**:
```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/directory",
  "permission_mode": "default",
  "hook_event_name": "SessionStart",
  "source": "startup"
}
```

**`source` Values**:
- `"startup"` - New session started
- `"resume"` - Existing session resumed
- `"clear"` - Session started after /clear

**Special Variables**:
- `CLAUDE_ENV_FILE` - Path for persisting environment variables

**Special Behavior**: stdout is added to context (visible to Claude).

---

### 2.8 SessionEnd (v1.0.85+)

**When**: Session terminates
**Blocking**: No
**Matchers**: Not applicable
**Version**: 1.0.85

**Capabilities**:
- Cleanup temporary files
- Archive session transcripts
- Stop background processes
- Release shared resources
- Log session end

**Input Fields**:
```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/directory",
  "permission_mode": "default",
  "hook_event_name": "SessionEnd",
  "reason": "exit"
}
```

**`reason` Values**:
- `"exit"` - Normal exit
- `"clear"` - Session cleared with /clear
- `"logout"` - User logged out
- `"prompt_input_exit"` - User exited during prompt input
- `"other"` - Other exit reasons

---

### 2.9 PreCompact

**When**: Before context compaction (summarization)
**Blocking**: No
**Matchers**: Supported (`"auto"` or `"manual"`)
**Version**: Original release

**Capabilities**:
- Backup full transcripts
- Archive important context
- Log compaction events

**Input Fields**:
```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/directory",
  "permission_mode": "default",
  "hook_event_name": "PreCompact",
  "trigger": "auto",
  "custom_instructions": ""
}
```

**`trigger` Values**:
- `"auto"` - Automatic compaction (95% context usage)
- `"manual"` - User triggered /compact

**`custom_instructions`**: Only populated for manual compaction (user's input to /compact).

---

### 2.10 Notification

**When**: Claude Code sends a notification
**Blocking**: No
**Matchers**: Not applicable
**Version**: Original release

**Capabilities**:
- Custom desktop notifications
- Text-to-speech announcements
- Sound effects
- External integrations

**Input Fields**:
```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/directory",
  "permission_mode": "default",
  "hook_event_name": "Notification",
  "message": "Waiting for your input..."
}
```

---

## 3. Configuration

### 3.1 Settings File Locations

| Location | Scope | Git |
|----------|-------|-----|
| `~/.claude/settings.json` | User (all projects) | Not tracked |
| `.claude/settings.json` | Project (shared) | Committed |
| `.claude/settings.local.json` | Project (personal) | Gitignored |

**Priority**: Local > Project > User

### 3.2 Basic Structure

```json
{
  "hooks": {
    "HookEventName": [
      {
        "matcher": "ToolPattern",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/script.sh",
            "timeout": 30000
          }
        ]
      }
    ]
  }
}
```

### 3.3 Complete Example

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude/hooks/validate_bash.py",
            "timeout": 10000
          }
        ]
      },
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/validate_file_edit.sh"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "prettier --write \"$CLAUDE_FILE_PATHS\" 2>/dev/null || true"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "prompt",
            "prompt": "Evaluate if all tasks are complete. Context: $ARGUMENTS"
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "echo '## Context'; git status --short; echo '---'"
          }
        ]
      }
    ]
  }
}
```

### 3.4 Hook Security Review

Direct edits to hook settings require review in the `/hooks` menu before they take effect. This prevents malicious modifications from affecting your current session.

---

## 4. JSON Input Schemas

### 4.1 Common Fields (All Hooks)

```typescript
interface BaseHookInput {
  session_id: string;           // Unique session identifier
  transcript_path: string;      // Path to JSONL transcript file
  cwd: string;                  // Current working directory
  hook_event_name: string;      // Name of the triggering event
  permission_mode: "default" | "plan" | "acceptEdits" | "dontAsk" | "bypassPermissions";
}
```

### 4.2 PreToolUse Input

```typescript
interface PreToolUseInput extends BaseHookInput {
  hook_event_name: "PreToolUse";
  tool_name: string;            // Name of tool being called
  tool_input: Record<string, unknown>;  // Tool-specific parameters
  tool_use_id?: string;         // Unique tool call ID
}
```

### 4.3 PostToolUse Input

```typescript
interface PostToolUseInput extends BaseHookInput {
  hook_event_name: "PostToolUse";
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_response: Record<string, unknown>;  // Tool execution result
  tool_use_id?: string;
}
```

### 4.4 PermissionRequest Input

```typescript
interface PermissionRequestInput extends BaseHookInput {
  hook_event_name: "PermissionRequest";
  tool_name: string;
  tool_input: Record<string, unknown>;
}
```

### 4.5 UserPromptSubmit Input

```typescript
interface UserPromptSubmitInput extends BaseHookInput {
  hook_event_name: "UserPromptSubmit";
  prompt: string;               // User's message text
}
```

### 4.6 Stop Input

```typescript
interface StopInput extends BaseHookInput {
  hook_event_name: "Stop";
  stop_hook_active: boolean;    // True if already continuing from hook
}
```

### 4.7 SubagentStop Input

```typescript
interface SubagentStopInput extends BaseHookInput {
  hook_event_name: "SubagentStop";
  stop_hook_active: boolean;
}
```

### 4.8 SessionStart Input

```typescript
interface SessionStartInput extends BaseHookInput {
  hook_event_name: "SessionStart";
  source: "startup" | "resume" | "clear";
}
```

### 4.9 SessionEnd Input

```typescript
interface SessionEndInput extends BaseHookInput {
  hook_event_name: "SessionEnd";
  reason: "exit" | "clear" | "logout" | "prompt_input_exit" | "other";
}
```

### 4.10 PreCompact Input

```typescript
interface PreCompactInput extends BaseHookInput {
  hook_event_name: "PreCompact";
  trigger: "auto" | "manual";
  custom_instructions: string;  // Only for manual trigger
}
```

### 4.11 Notification Input

```typescript
interface NotificationInput extends BaseHookInput {
  hook_event_name: "Notification";
  message: string;              // Notification content
}
```

---

## 5. JSON Output Schemas

### 5.1 Common Output Fields

```typescript
interface BaseHookOutput {
  continue?: boolean;           // False to halt processing (default: true)
  stopReason?: string;          // Message when continue=false
  suppressOutput?: boolean;     // Hide stdout from transcript
  systemMessage?: string;       // Warning shown to user
}
```

### 5.2 PreToolUse Output

```typescript
interface PreToolUseOutput extends BaseHookOutput {
  hookSpecificOutput?: {
    hookEventName: "PreToolUse";
    permissionDecision?: "allow" | "deny" | "ask";
    permissionDecisionReason?: string;
    updatedInput?: Record<string, unknown>;  // Modified tool input (v2.0.10+)
  };
}
```

**Example - Allow with modification**:
```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "permissionDecisionReason": "Auto-approved test command with safety flag",
    "updatedInput": {
      "command": "npm test --dry-run"
    }
  }
}
```

### 5.3 PostToolUse Output

```typescript
interface PostToolUseOutput extends BaseHookOutput {
  decision?: "block";           // Provide feedback to Claude
  reason?: string;              // Explanation for Claude
}
```

### 5.4 PermissionRequest Output

```typescript
interface PermissionRequestOutput extends BaseHookOutput {
  hookSpecificOutput?: {
    hookEventName: "PermissionRequest";
    decision?: {
      behavior: "allow" | "deny";
      message?: string;
      updatedInput?: Record<string, unknown>;
      interrupt?: boolean;
    };
  };
}
```

### 5.5 UserPromptSubmit Output

```typescript
interface UserPromptSubmitOutput extends BaseHookOutput {
  decision?: "block";
  reason?: string;
  hookSpecificOutput?: {
    hookEventName: "UserPromptSubmit";
    additionalContext?: string;  // Added to context
  };
}
```

### 5.6 Stop/SubagentStop Output

```typescript
interface StopOutput extends BaseHookOutput {
  decision?: "approve" | "block";
  reason?: string;              // Required if decision="block"
}
```

### 5.7 SessionStart Output

```typescript
interface SessionStartOutput extends BaseHookOutput {
  hookSpecificOutput?: {
    hookEventName: "SessionStart";
    additionalContext?: string;  // Added to context
  };
}
```

---

## 6. Tool Input Schemas

### 6.1 File Operations

**Read**:
```typescript
{
  file_path: string;     // Required: absolute path
  offset?: number;       // Starting line (1-based)
  limit?: number;        // Max lines (default: 2000)
}
```

**Write**:
```typescript
{
  file_path: string;     // Required: absolute path
  content: string;       // Required: complete file content
}
```

**Edit**:
```typescript
{
  file_path: string;     // Required: absolute path
  old_string: string;    // Required: text to replace
  new_string: string;    // Required: replacement text
  replace_all?: boolean; // Replace all occurrences (default: false)
}
```

**Glob**:
```typescript
{
  pattern: string;       // Required: glob pattern (e.g., "**/*.ts")
  path?: string;         // Directory to search (default: cwd)
}
```

**Grep**:
```typescript
{
  pattern: string;           // Required: regex pattern
  path?: string;             // File or directory (default: cwd)
  output_mode?: "content" | "files_with_matches" | "count";
  glob?: string;             // File filter (e.g., "*.js")
  type?: string;             // File type (e.g., "js", "py")
  "-i"?: boolean;            // Case insensitive
  "-n"?: boolean;            // Show line numbers
  "-A"?: number;             // Lines after match
  "-B"?: number;             // Lines before match
  "-C"?: number;             // Lines around match
  multiline?: boolean;       // Multiline mode
  head_limit?: number;       // Limit results
}
```

### 6.2 Execution

**Bash**:
```typescript
{
  command: string;           // Required: shell command
  description?: string;      // 5-10 word description
  timeout?: number;          // Max 600000ms (default: 120000)
  run_in_background?: boolean;
}
```

**KillShell**:
```typescript
{
  shell_id: string;          // Required: shell to terminate
}
```

### 6.3 Notebook

**NotebookEdit**:
```typescript
{
  notebook_path: string;     // Required: absolute path
  new_source: string;        // Required: cell content
  cell_id?: string;          // Cell identifier
  cell_type?: "code" | "markdown";
  edit_mode?: "replace" | "insert" | "delete";
}
```

### 6.4 Web

**WebFetch**:
```typescript
{
  url: string;               // Required: valid URL
  prompt: string;            // Required: extraction prompt
}
```

**WebSearch**:
```typescript
{
  query: string;             // Required: search query (min 2 chars)
  allowed_domains?: string[];
  blocked_domains?: string[];
}
```

### 6.5 Task Management

**Task**:
```typescript
{
  description: string;       // Required: 3-5 word summary
  prompt: string;            // Required: detailed task
  subagent_type: string;     // Required: agent type
  model?: "sonnet" | "opus" | "haiku" | "inherit";
  run_in_background?: boolean;
  resume?: string;           // Agent ID to continue
}
```

**TodoWrite**:
```typescript
{
  todos: Array<{
    content: string;         // Required: imperative form
    status: "pending" | "in_progress" | "completed";
    activeForm: string;      // Required: present continuous form
  }>;
}
```

### 6.6 User Interaction

**AskUserQuestion**:
```typescript
{
  questions: Array<{         // 1-4 questions
    question: string;        // Required: the question
    header: string;          // Required: max 12 chars
    multiSelect: boolean;    // Required
    options: Array<{         // 2-4 options
      label: string;         // 1-5 words
      description: string;   // Required
    }>;
  }>;
}
```

---

## 7. Matchers

Matchers filter which tools trigger hooks. Only applicable to **PreToolUse**, **PostToolUse**, and **PermissionRequest**.

### 7.1 Matcher Patterns

| Pattern | Matches |
|---------|---------|
| `"Write"` | Write tool only |
| `"Write\|Edit"` | Write OR Edit |
| `"*"` | All tools |
| `""` (empty) | All tools |
| `"Bash(npm test*)"` | Bash with specific args |
| `"Bash(git:*)"` | Bash starting with "git" |
| `"mcp__github__.*"` | MCP tool regex |
| `"Notebook.*"` | Regex pattern |

### 7.2 Case Sensitivity

Matchers are **case-sensitive**:
- `"Bash"` - Matches Bash tool
- `"bash"` - Does NOT match

### 7.3 Argument Patterns

For Bash commands, you can match specific arguments:

```json
{
  "matcher": "Bash(npm test*)",
  "hooks": [...]
}
```

This matches:
- `npm test`
- `npm test --coverage`
- `npm test:unit`

### 7.4 MCP Tool Patterns

For MCP server tools:

```json
{
  "matcher": "mcp__memory__.*",
  "hooks": [...]
}
```

### 7.5 PreCompact Matchers

PreCompact supports trigger-based matching:

```json
{
  "hooks": {
    "PreCompact": [
      {
        "matcher": "auto",
        "hooks": [{ "type": "command", "command": "..." }]
      },
      {
        "matcher": "manual",
        "hooks": [{ "type": "command", "command": "..." }]
      }
    ]
  }
}
```

---

## 8. Exit Codes & Control Flow

### 8.1 Exit Code Meanings

| Code | Meaning | Behavior |
|------|---------|----------|
| **0** | Success | stdout processed; action continues |
| **2** | Blocking error | stderr fed to Claude; action prevented |
| **Other** | Non-blocking error | stderr shown to user; action continues |

### 8.2 Control Flow Priority

1. `"continue": false` - Highest priority, stops all processing
2. JSON `"decision"` field - Hook-specific blocking
3. Exit code 2 - stderr-based blocking
4. Exit code != 0 - Non-blocking errors

### 8.3 Which Hooks Can Block

| Hook | Can Block? | Method |
|------|------------|--------|
| PreToolUse | Yes | Exit 2 or `permissionDecision: "deny"` |
| PostToolUse | No | Tool already executed |
| PermissionRequest | Yes | `decision.behavior: "deny"` |
| UserPromptSubmit | Yes | Exit 2 or `decision: "block"` |
| Stop | Yes | `decision: "block"` with `continue: true` |
| SubagentStop | Yes | `decision: "block"` with `continue: true` |
| SessionStart | No | Informational only |
| SessionEnd | No | Informational only |
| PreCompact | No | Informational only |
| Notification | No | Informational only |

### 8.4 Blocking Examples

**PreToolUse - Block with exit code**:
```bash
#!/bin/bash
echo "Dangerous command blocked" >&2
exit 2
```

**PreToolUse - Block with JSON**:
```python
import json
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": "Command contains sudo"
    }
}))
```

**Stop - Force continuation**:
```python
import json
print(json.dumps({
    "decision": "block",
    "reason": "Tests have not been run yet",
    "continue": True
}))
```

---

## 9. Environment Variables

### 9.1 Available Variables

| Variable | Scope | Description |
|----------|-------|-------------|
| `CLAUDE_PROJECT_DIR` | All hooks | Absolute path to project root |
| `CLAUDE_CODE_REMOTE` | All hooks | `"true"` if running in web environment |
| `CLAUDE_ENV_FILE` | SessionStart | Path for persistent env vars |
| `CLAUDE_FILE_PATHS` | PostToolUse | File paths from tool (for formatters) |

### 9.2 Tool Input Variables

For some tools, specific input fields are available as environment variables:

| Tool | Variable | Content |
|------|----------|---------|
| Write/Edit | `CLAUDE_TOOL_INPUT_FILE_PATH` | Target file path |
| Bash | `CLAUDE_TOOL_INPUT_COMMAND` | Command being run |

### 9.3 Using Environment Variables

**In shell commands**:
```json
{
  "command": "prettier --write \"$CLAUDE_FILE_PATHS\" 2>/dev/null || true"
}
```

**In Python scripts**:
```python
import os
project_dir = os.environ.get('CLAUDE_PROJECT_DIR', os.getcwd())
```

### 9.4 Known Issues

Some environment variables (`CLAUDE_TOOL_INPUT_*`) have been reported as empty in certain versions. The recommended approach is to read the JSON from stdin instead.

---

## 10. Hook Types: Command vs Prompt

### 10.1 Command Hooks

**Type**: `"command"`
**Execution**: Shell command
**Use Case**: Deterministic, fast operations

```json
{
  "type": "command",
  "command": "/path/to/script.sh",
  "timeout": 30000
}
```

**Characteristics**:
- Execute any shell command
- Receive JSON via stdin
- Return JSON via stdout
- Control via exit codes
- Default timeout: 60 seconds

### 10.2 Prompt Hooks

**Type**: `"prompt"`
**Execution**: LLM evaluation (Claude Haiku)
**Use Case**: Context-aware decisions
**Supported Events**: Stop, SubagentStop only

```json
{
  "type": "prompt",
  "prompt": "Evaluate if Claude completed all tasks. Context: $ARGUMENTS",
  "timeout": 30000
}
```

**Characteristics**:
- Uses Claude Haiku for evaluation
- `$ARGUMENTS` replaced with hook input
- Returns structured JSON with decision
- Best for judgment calls

### 10.3 When to Use Each

| Scenario | Hook Type | Reason |
|----------|-----------|--------|
| Block dangerous commands | Command | Deterministic, fast |
| Format code | Command | Specific tool execution |
| Verify task completion | Prompt | Requires judgment |
| Validate subagent quality | Prompt | Context-aware |
| Log all activity | Command | Simple append |
| Desktop notifications | Command | System integration |

### 10.4 Prompt Hook Response Format

```json
{
  "decision": "approve" | "block",
  "reason": "Explanation for decision",
  "continue": true | false,
  "stopReason": "Message shown to user",
  "systemMessage": "Optional warning"
}
```

---

## 11. Advanced Patterns

### 11.1 Input Modification (v2.0.10+)

Modify tool inputs transparently before execution:

```python
#!/usr/bin/env python3
import json
import sys

payload = json.loads(sys.stdin.read())
tool_input = payload.get("tool_input", {})

# Add safety flag to commands
if payload.get("tool_name") == "Bash":
    command = tool_input.get("command", "")
    if command.startswith("rm "):
        tool_input["command"] = command.replace("rm ", "rm -i ")

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "permissionDecisionReason": "Auto-approved with safety flag",
        "updatedInput": tool_input
    }
}))
```

### 11.2 Parallel Hook Execution

Multiple hooks matching the same event run in parallel:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          { "type": "command", "command": "prettier --write $CLAUDE_FILE_PATHS" },
          { "type": "command", "command": "eslint --fix $CLAUDE_FILE_PATHS" }
        ]
      }
    ]
  }
}
```

### 11.3 Context Injection (SessionStart)

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "echo '## Context'; git status --short; echo '---'; cat TODO.md 2>/dev/null || true"
      }]
    }]
  }
}
```

### 11.4 Transcript Backup (PreCompact)

```bash
#!/bin/bash
# backup-transcript.sh
PAYLOAD=$(cat)
TRANSCRIPT_PATH=$(echo "$PAYLOAD" | jq -r '.transcript_path')
BACKUP_DIR="$HOME/.claude/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"
cp "$TRANSCRIPT_PATH" "$BACKUP_DIR/transcript_${TIMESTAMP}.jsonl"

echo '{"continue": true}'
```

### 11.5 Task Completion Verification (Stop)

```json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "prompt",
        "prompt": "Review the conversation and determine if ALL tasks are complete. Check: 1) All code changes implemented 2) All tests pass 3) No TODO items remain. If incomplete, respond with {\"decision\":\"block\",\"reason\":\"[specific incomplete items]\",\"continue\":true}. Context: $ARGUMENTS"
      }]
    }]
  }
}
```

### 11.6 TTS Notifications

```json
{
  "hooks": {
    "Notification": [{
      "hooks": [{
        "type": "command",
        "command": "python3 ~/.claude/hooks/speak_notification.py"
      }]
    }]
  }
}
```

```python
#!/usr/bin/env python3
import json
import sys
import subprocess

payload = json.loads(sys.stdin.read())
message = payload.get("message", "Notification from Claude Code")

# macOS say command
subprocess.run(["say", message], capture_output=True)
```

### 11.7 UV Single-File Scripts

Use `uv` for dependency management:

```python
#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.9"
# dependencies = ["requests", "python-dotenv"]
# ///

import json
import sys
import requests

payload = json.loads(sys.stdin.read())
# ... process with dependencies
```

---

## 12. Security Best Practices

### 12.1 Input Validation

```python
import json
import re
import sys

payload = json.loads(sys.stdin.read())
tool_input = payload.get("tool_input", {})

# Validate file paths
file_path = tool_input.get("file_path", "")
if ".." in file_path:
    print("Path traversal detected", file=sys.stderr)
    sys.exit(2)

# Validate commands
command = tool_input.get("command", "")
dangerous = ["rm -rf /", "sudo", "chmod 777", "mkfs"]
if any(d in command for d in dangerous):
    print("Dangerous command blocked", file=sys.stderr)
    sys.exit(2)
```

### 12.2 Quote Shell Variables

```bash
# GOOD
if echo "$CLAUDE_TOOL_INPUT_FILE_PATH" | grep -q "\.env"; then
    echo "Blocked: sensitive file" >&2
    exit 2
fi

# BAD - vulnerable to injection
if echo $CLAUDE_TOOL_INPUT_FILE_PATH | grep -q "\.env"; then
```

### 12.3 Sensitive File Protection

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Write|Edit|Read",
      "hooks": [{
        "type": "command",
        "command": "if echo \"$CLAUDE_TOOL_INPUT_FILE_PATH\" | grep -qE '\\.(env|pem|key|secrets)$|credentials'; then echo '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"Blocked: sensitive file\"}}'; else echo '{}'; fi"
      }]
    }]
  }
}
```

### 12.4 Don't Log Secrets

Never log:
- API keys
- Passwords
- Private keys
- Auth tokens
- `.env` file contents

### 12.5 Use Absolute Paths

```bash
# GOOD
"$CLAUDE_PROJECT_DIR/.claude/hooks/validate.sh"

# BAD - may execute wrong script
"./hooks/validate.sh"
```

---

## 13. Debugging

### 13.1 Debug Mode

```bash
claude --debug
```

Shows hook loading, matching, and execution details.

### 13.2 Hook Status

```bash
/hooks
```

Interactive menu to view, test, and manage hooks.

### 13.3 Transcript Monitoring

```bash
tail -f /path/to/transcript.jsonl | jq
```

### 13.4 Common Issues

| Issue | Solution |
|-------|----------|
| Hook not executing | Check path, permissions, test manually |
| Hook executing twice | Known bug in home directory; use project directory |
| Environment vars empty | Use stdin JSON instead |
| Timeout errors | Increase timeout or optimize script |
| Permission denied | Run `chmod +x script.sh` |

### 13.5 Wrapper Script for Debugging

```bash
#!/bin/bash
# debug-wrapper.sh
LOG="$HOME/.claude/hooks/debug.log"
echo "=== $(date) ===" >> "$LOG"
echo "Args: $@" >> "$LOG"
cat >> "$LOG"  # Log stdin

# Run actual hook
exec /path/to/actual-hook.sh
```

---

## 14. Complete Examples

### 14.1 Auto-Format After Edits

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Write|Edit",
      "hooks": [{
        "type": "command",
        "command": "prettier --write \"$CLAUDE_FILE_PATHS\" 2>/dev/null; eslint --fix \"$CLAUDE_FILE_PATHS\" 2>/dev/null || true"
      }]
    }]
  }
}
```

### 14.2 Auto-Approve Test Commands

```json
{
  "hooks": {
    "PermissionRequest": [{
      "matcher": "Bash(npm test*)|Bash(pytest*)|Bash(cargo test*)",
      "hooks": [{
        "type": "command",
        "command": "echo '{\"hookSpecificOutput\":{\"hookEventName\":\"PermissionRequest\",\"decision\":{\"behavior\":\"allow\",\"message\":\"Auto-approved test command\"}}}'"
      }]
    }]
  }
}
```

### 14.3 Block Dangerous Operations

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "python3 ~/.claude/hooks/validate_bash.py"
      }]
    }]
  }
}
```

**validate_bash.py**:
```python
#!/usr/bin/env python3
import json
import re
import sys

DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"sudo\s+",
    r"chmod\s+777",
    r">\s*/dev/sd",
    r"mkfs\.",
    r"dd\s+if=",
    r":(){ :|:& };:",
]

payload = json.loads(sys.stdin.read())
command = payload.get("tool_input", {}).get("command", "")

for pattern in DANGEROUS_PATTERNS:
    if re.search(pattern, command):
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Blocked: matches dangerous pattern '{pattern}'"
            }
        }))
        sys.exit(0)

print("{}")
```

### 14.4 Session Context Injection

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "~/.claude/hooks/session_context.sh"
      }]
    }]
  }
}
```

**session_context.sh**:
```bash
#!/bin/bash
echo "## Project Context"
echo ""
echo "### Git Status"
git status --short 2>/dev/null || echo "Not a git repository"
echo ""
echo "### Recent Commits"
git log --oneline -5 2>/dev/null || echo "No commits"
echo ""
echo "### TODO Items"
cat TODO.md 2>/dev/null || echo "No TODO.md found"
echo ""
echo "---"
```

### 14.5 Task Completion Enforcement

```json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "prompt",
        "prompt": "You are evaluating if Claude Code should stop.\n\nCheck ALL of these conditions:\n1. Were ALL requested changes implemented?\n2. Do all tests pass (if tests were required)?\n3. Are there any unresolved TODO comments in modified code?\n4. Were all files saved correctly?\n\nIf ANY condition is not met, respond:\n{\"decision\":\"block\",\"reason\":\"[Specific incomplete items]\",\"continue\":true}\n\nIf ALL conditions are met, respond:\n{\"decision\":\"approve\",\"continue\":false}\n\nContext: $ARGUMENTS"
      }]
    }]
  }
}
```

### 14.6 Desktop Notifications (macOS)

```json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "osascript -e 'display notification \"Claude Code finished\" with title \"Claude Code\" sound name \"Glass\"'"
      }]
    }],
    "Notification": [{
      "hooks": [{
        "type": "command",
        "command": "osascript -e 'display notification \"Waiting for input\" with title \"Claude Code\" sound name \"Ping\"'"
      }]
    }]
  }
}
```

### 14.7 Logging All Commands

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "echo \"$(date '+%Y-%m-%d %H:%M:%S'): $CLAUDE_TOOL_INPUT_COMMAND\" >> ~/.claude/bash.log"
      }]
    }]
  }
}
```

### 14.8 Subagent Quality Validation

```json
{
  "hooks": {
    "SubagentStop": [{
      "hooks": [{
        "type": "prompt",
        "prompt": "Evaluate if this subagent completed its task properly.\n\nCheck:\n1. Did it address all requested items?\n2. Is the output quality acceptable?\n3. Were any errors or warnings generated?\n\nIf work is incomplete or low quality, respond:\n{\"decision\":\"block\",\"reason\":\"[What needs improvement]\",\"continue\":true}\n\nIf satisfactory, respond:\n{\"decision\":\"approve\",\"continue\":false}\n\nContext: $ARGUMENTS"
      }]
    }]
  }
}
```

---

## 15. Known Issues & Workarounds

### 15.1 Duplicate Hook Execution

**Issue**: Hooks fire twice when Claude Code runs from home directory.
**Workaround**: Always run from project directory, not `~`.

### 15.2 Empty Environment Variables

**Issue**: `CLAUDE_TOOL_INPUT_*` variables sometimes empty.
**Workaround**: Parse JSON from stdin instead:

```python
import json
import sys
payload = json.loads(sys.stdin.read())
file_path = payload.get("tool_input", {}).get("file_path")
```

### 15.3 SessionEnd Not Firing on /clear

**Issue**: SessionEnd doesn't fire when using `/clear`.
**Status**: Known bug, may be fixed in future versions.

### 15.4 PreCompact Empty transcript_path

**Issue**: transcript_path may be empty in PreCompact hooks.
**Workaround**: Store transcript path from SessionStart hook.

### 15.5 PermissionRequest Triggers for AskUserQuestion

**Issue**: PermissionRequest fires for AskUserQuestion tool.
**Workaround**: Add specific check in hook to skip AskUserQuestion.

---

## 16. Version History

| Version | Feature |
|---------|---------|
| Original | PreToolUse, PostToolUse, Stop, SessionStart, PreCompact, Notification, UserPromptSubmit |
| 1.0.41 | SubagentStop hook |
| 1.0.85 | SessionEnd hook |
| 2.0.10 | PreToolUse `updatedInput` for input modification |
| 2.0.41 | Prompt-based hooks for Stop/SubagentStop |
| 2.0.45 | PermissionRequest hook |

---

## 17. Sources

### Official Documentation
- [Hooks Reference - Claude Code Docs](https://code.claude.com/docs/en/hooks)
- [Get Started with Hooks Guide](https://code.claude.com/docs/en/hooks-guide)
- [How to Configure Hooks - Claude Blog](https://claude.com/blog/how-to-configure-hooks)
- [Agent SDK Hooks Reference](https://platform.claude.com/docs/en/agent-sdk/hooks)

### Community Resources
- [Claude Code Hooks Mastery - GitHub](https://github.com/disler/claude-code-hooks-mastery)
- [Claude Code Hooks JSON Schemas - Gist](https://gist.github.com/FrancisBourre/50dca37124ecc43eaf08328cdcccdb34)
- [Claude Code Hook Control Flow](https://stevekinney.com/courses/ai-development/claude-code-hook-control-flow)
- [ClaudeLog Hooks Reference](https://claudelog.com/mechanics/hooks/)

### Tools & Integrations
- [macOS Notification Hook](https://github.com/wyattjoh/claude-code-notification)
- [TTS Audio Feedback Plugin](https://github.com/husniadil/cc-hooks)
- [Voice Notification Handler](https://github.com/markhilton/claude-code-voice-handler)

### Issue Tracker
- [Claude Code GitHub Issues](https://github.com/anthropics/claude-code/issues)

---

*Last updated: January 2026*
