# Claude Code Complete Reference Guide

> **For AI Agents**: This comprehensive guide contains everything you need to optimally use Claude Code - the agentic coding tool from Anthropic. Read this document to understand all features, tools, hooks, subagents, skills, and commands available.

---

## Table of Contents

1. [Overview & Core Concepts](#1-overview--core-concepts)
2. [Installation & Setup](#2-installation--setup)
3. [Core CLI Commands & Options](#3-core-cli-commands--options)
4. [Configuration & Settings](#4-configuration--settings)
5. [Memory System (CLAUDE.md)](#5-memory-system-claudemd)
6. [Available Tools](#6-available-tools)
7. [Model Selection](#7-model-selection)
8. [Built-in Slash Commands](#8-built-in-slash-commands)
9. [Custom Slash Commands](#9-custom-slash-commands)
10. [Skills System](#10-skills-system)
11. [Subagents & Task Tool](#11-subagents--task-tool)
12. [Hooks System](#12-hooks-system)
13. [MCP Integration](#13-mcp-integration)
14. [IDE Integrations](#14-ide-integrations)
15. [Keyboard Shortcuts](#15-keyboard-shortcuts)
16. [Permissions & Security](#16-permissions--security)
17. [Best Practices](#17-best-practices)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. Overview & Core Concepts

### What is Claude Code?

Claude Code is Anthropic's **agentic coding tool** that operates in your terminal and IDE. It understands your entire codebase and can:

- **Build features** from natural language descriptions
- **Debug and fix** issues automatically
- **Navigate and understand** entire codebases
- **Automate** tedious development tasks
- **Execute actions directly** (edit files, run commands, create commits)

### Key Principles

1. **Agentic Execution**: Claude Code can take autonomous actions, not just suggest code
2. **Context-Aware**: Understands your project structure, dependencies, and patterns
3. **Tool-Based**: Uses specific tools for different operations (Read, Write, Edit, Bash, etc.)
4. **Permission-Controlled**: Asks permission before potentially destructive actions
5. **Extensible**: Custom hooks, skills, subagents, and MCP integrations

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Claude Code                         │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │  Tools  │  │  Hooks  │  │ Skills  │  │Subagents│    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Memory System   │  │    MCP Servers              │  │
│  │ (CLAUDE.md)     │  │    (External Integrations)  │  │
│  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Installation & Setup

### System Requirements

- **macOS** 10.15+ / **Ubuntu** 20.04+ / **Windows** 10+ (WSL/Git Bash)
- **RAM**: 4 GB+ recommended
- **Node.js**: 18+ (for npm installation)
- **Internet**: Required for authentication

### Installation Methods

```bash
# Native install (macOS/Linux) - Recommended
curl -fsSL https://claude.ai/install.sh | bash

# Homebrew (macOS)
brew install --cask claude-code

# NPM (all platforms)
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version
```

### Initial Setup

```bash
# Start Claude Code (triggers authentication)
claude

# Run diagnostics
claude doctor

# Initialize project memory
claude /init
```

### Authentication Methods

| Method | Best For | Setup |
|--------|----------|-------|
| **Claude Pro/Max** | Individual developers | Browser-based OAuth |
| **Claude Console** | API billing | `ANTHROPIC_API_KEY` env var |
| **Claude for Teams** | Team collaboration | Organization SSO |
| **AWS Bedrock** | AWS users | `AWS_BEARER_TOKEN_BEDROCK` |
| **Google Vertex AI** | GCP users | `GOOGLE_APPLICATION_CREDENTIALS` |

---

## 3. Core CLI Commands & Options

### Essential Commands

```bash
# Start interactive REPL
claude

# Execute single query and exit
claude -p "explain this codebase"

# Continue previous session
claude -c

# Resume specific session by ID
claude -r "session-id"

# Update to latest version
claude update

# System health check
claude doctor
```

### Key Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--model <name>` | Specify model | `claude --model opus` |
| `-p, --prompt` | Non-interactive prompt | `claude -p "fix the bug"` |
| `-c, --continue` | Continue last session | `claude -c` |
| `-r, --resume` | Resume specific session | `claude -r abc123` |
| `--system-prompt` | Custom system prompt | `claude --system-prompt "Be concise"` |
| `--tools` | Restrict available tools | `claude --tools "Read,Grep,Glob"` |
| `--permission-mode` | Set permission mode | `claude --permission-mode plan` |
| `--output-format` | Output format | `claude --output-format json` |
| `--agents` | Define session agents | `claude --agents '{...}'` |
| `--verbose` | Enable verbose output | `claude --verbose` |

### Permission Modes

| Mode | Description |
|------|-------------|
| `default` | Ask for permission on sensitive actions |
| `acceptEdits` | Auto-accept file edits, ask for others |
| `bypassPermissions` | Skip all permission prompts (use carefully) |
| `plan` | Read-only analysis mode (no modifications) |

---

## 4. Configuration & Settings

### Settings Hierarchy (Highest to Lowest Priority)

1. **Managed settings** - IT/organization deployed (system-wide)
2. **User settings** - `~/.claude/settings.json`
3. **Project settings** - `.claude/settings.json` (shared with team)
4. **Local settings** - `.claude/settings.local.json` (gitignored)

### Configuration File Structure

```json
{
  "model": "claude-sonnet-4-20250514",
  "permissions": {
    "allow": ["Read", "Glob", "Grep"],
    "deny": ["Bash(rm:*)", "Write(.env)"],
    "ask": ["Edit", "Write", "Bash"]
  },
  "hooks": {
    "PreToolUse": [...],
    "PostToolUse": [...]
  },
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    }
  },
  "env": {
    "CUSTOM_VAR": "value"
  }
}
```

### Key Settings

| Setting | Purpose | Example |
|---------|---------|---------|
| `model` | Default model | `"claude-sonnet-4-20250514"` |
| `permissions` | Tool access rules | `{"allow": [...], "deny": [...]}` |
| `hooks` | Lifecycle hooks | See [Hooks System](#12-hooks-system) |
| `mcpServers` | MCP server config | See [MCP Integration](#13-mcp-integration) |
| `env` | Environment variables | `{"API_KEY": "..."}` |
| `theme` | UI theme | `"dark"` or `"light"` |

### Managing Settings

```bash
# Open settings UI
/config

# Edit specific setting file
/config user     # ~/.claude/settings.json
/config project  # .claude/settings.json
/config local    # .claude/settings.local.json
```

---

## 5. Memory System (CLAUDE.md)

### Memory Hierarchy

Claude Code reads instructions from multiple sources in priority order:

1. **Enterprise CLAUDE.md** - Organization-wide rules (managed)
2. **Project CLAUDE.md** - `./CLAUDE.md` (shared with team)
3. **Modular rules** - `.claude/rules/*.md` (topic-specific)
4. **User CLAUDE.md** - `~/.claude/CLAUDE.md` (personal)
5. **Local CLAUDE.md** - `./CLAUDE.local.md` (gitignored)

### CLAUDE.md Best Practices

```markdown
# Project: MyApp

## Build Commands
- `npm run build` - Build production
- `npm test` - Run tests
- `npm run lint` - Lint code

## Code Style
- Use TypeScript strict mode
- Prefer functional components
- Use named exports

## Architecture
- `/src/components` - React components
- `/src/services` - API services
- `/src/utils` - Utility functions

## Important Notes
- Never commit .env files
- Run tests before committing
- Use conventional commits
```

### File Imports

Reference other files within CLAUDE.md:

```markdown
See @docs/architecture.md for system design.
Review @src/types/index.ts for type definitions.
```

### Path-Specific Rules

Create rules that apply only to certain paths using YAML frontmatter:

```markdown
---
globs: src/components/**/*.tsx
---

# Component Guidelines
- Use React.FC type
- Include PropTypes
- Add JSDoc comments
```

### Managing Memory

```bash
# Initialize project CLAUDE.md
/init

# Edit memory files
/memory

# View current memory context
/context
```

---

## 6. Available Tools

### Core Tools

| Tool | Permission | Purpose |
|------|------------|---------|
| **Read** | None | Read file contents |
| **Write** | Ask | Create new files |
| **Edit** | Ask | Modify existing files |
| **MultiEdit** | Ask | Edit multiple files |
| **Glob** | None | Find files by pattern |
| **Grep** | None | Search file contents |
| **Bash** | Ask | Execute shell commands |
| **WebFetch** | Ask | Fetch URL contents |
| **WebSearch** | Ask | Search the web |
| **Task** | None | Spawn subagents |
| **TodoWrite** | None | Manage task lists |
| **NotebookEdit** | Ask | Edit Jupyter notebooks |
| **LSP** | None | Language server queries |

### Tool Usage Examples

```bash
# Reading files
Read("/src/index.ts")
Read("/package.json", offset=10, limit=50)

# Finding files
Glob("**/*.tsx", path="/src")
Glob("*.test.ts")

# Searching content
Grep("function authenticate", type="ts")
Grep("TODO|FIXME", glob="**/*.js")

# Editing files
Edit(file_path="/src/app.ts", old_string="...", new_string="...")

# Running commands
Bash("npm test")
Bash("git status")

# Spawning subagents
Task(prompt="Analyze security", subagent_type="Explore")
```

### Tool Restrictions

Configure tool access in settings:

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Bash(npm:*)",
      "Bash(git:*)"
    ],
    "deny": [
      "Bash(rm:*)",
      "Bash(sudo:*)",
      "Write(.env)",
      "Edit(.git/*)"
    ]
  }
}
```

---

## 7. Model Selection

### Available Models

| Alias | Model | Best For |
|-------|-------|----------|
| `default` | Recommended | General use |
| `sonnet` | Claude Sonnet 4.5 | Daily tasks, balanced |
| `opus` | Claude Opus 4.5 | Complex reasoning |
| `haiku` | Claude Haiku | Fast, simple tasks |
| `sonnet[1m]` | Sonnet + 1M context | Large codebases |
| `opusplan` | Opus for planning | Plan + execute split |

### Setting the Model

```bash
# During session
/model opus

# At startup
claude --model opus

# Environment variable
export ANTHROPIC_MODEL=opus

# In settings.json
{
  "model": "claude-opus-4-5-20250514"
}
```

### Model Selection Strategy

```
┌─────────────────────────────────────────────────────────┐
│                When to Use Each Model                    │
├─────────────────────────────────────────────────────────┤
│  SONNET (Default)                                       │
│  • General development tasks                            │
│  • Code reviews                                         │
│  • Bug fixes                                            │
│  • Feature implementation                               │
├─────────────────────────────────────────────────────────┤
│  OPUS (Complex)                                         │
│  • Architectural decisions                              │
│  • Security analysis                                    │
│  • Complex refactoring                                  │
│  • When accuracy is critical                            │
├─────────────────────────────────────────────────────────┤
│  HAIKU (Fast)                                           │
│  • Quick searches                                       │
│  • Simple file reads                                    │
│  • Cost-sensitive operations                            │
│  • Exploration tasks                                    │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Built-in Slash Commands

### Session Management

| Command | Description |
|---------|-------------|
| `/help` | Show help and available commands |
| `/clear` | Clear conversation history |
| `/compact` | Compress context (keep summary) |
| `/resume` | Resume previous session |
| `/rewind` | Undo recent changes/messages |
| `/export` | Export conversation |

### Configuration

| Command | Description |
|---------|-------------|
| `/config` | Open settings interface |
| `/model <name>` | Change model |
| `/permissions` | Manage tool permissions |
| `/theme` | Change UI theme |
| `/vim` | Enable vim mode |

### Project & Memory

| Command | Description |
|---------|-------------|
| `/init` | Initialize CLAUDE.md |
| `/memory` | Edit memory files |
| `/context` | Show current context |
| `/cost` | Show token usage |
| `/stats` | Session statistics |

### Code Quality

| Command | Description |
|---------|-------------|
| `/review` | Code review |
| `/security-review` | Security audit |
| `/explain` | Explain code |
| `/think` | Enable extended thinking |

### Extensions

| Command | Description |
|---------|-------------|
| `/agents` | Manage subagents |
| `/mcp` | MCP server management |
| `/hooks` | Manage hooks |
| `/plugin` | Plugin management |

### Account

| Command | Description |
|---------|-------------|
| `/login` | Login to different account |
| `/logout` | Sign out |
| `/doctor` | System diagnostics |

---

## 9. Custom Slash Commands

### Creating Custom Commands

Store in `.claude/commands/` (project) or `~/.claude/commands/` (personal).

### Basic Command

```markdown
<!-- .claude/commands/review-pr.md -->
---
description: Review a pull request
argument-hint: <pr-number>
---

Review pull request #$ARGUMENTS focusing on:
1. Code quality
2. Security issues
3. Performance concerns
4. Test coverage
```

### Command with Arguments

```markdown
<!-- .claude/commands/fix-issue.md -->
---
description: Fix a GitHub issue
argument-hint: <issue-number> [priority]
---

Fix GitHub issue #$1 with priority $2.

1. Read the issue details
2. Analyze the codebase
3. Implement the fix
4. Write tests
5. Create commit
```

### Command with Bash Execution

```markdown
<!-- .claude/commands/branch-status.md -->
---
description: Show branch status
allowed-tools: Bash(git:*)
---

## Current Branch Status

- Branch: !`git branch --show-current`
- Status: !`git status --short`
- Recent commits: !`git log --oneline -5`
```

### Command with File References

```markdown
<!-- .claude/commands/component.md -->
---
description: Create a React component
argument-hint: <ComponentName>
---

Create component $ARGUMENTS following the pattern in:
@src/components/Button/Button.tsx

Use the types from:
@src/types/components.ts
```

### Command with Tool Restrictions

```markdown
<!-- .claude/commands/analyze.md -->
---
description: Read-only code analysis
allowed-tools: Read, Grep, Glob
---

Analyze the codebase structure without making changes.
```

---

## 10. Skills System

### What Are Skills?

Skills are **model-invoked** capabilities that Claude automatically uses based on context. Unlike slash commands (explicitly invoked), Skills are discovered and used automatically.

### Skill vs Slash Command

| Aspect | Skills | Slash Commands |
|--------|--------|----------------|
| Invocation | Automatic | Manual (`/command`) |
| Structure | Directory + SKILL.md | Single .md file |
| Complexity | Multi-file workflows | Simple prompts |
| Discovery | Description-based | Listed in `/help` |

### Creating a Skill

```bash
# Create skill directory
mkdir -p .claude/skills/code-analyzer

# Create SKILL.md
cat > .claude/skills/code-analyzer/SKILL.md << 'EOF'
---
name: code-analyzer
description: Analyzes code quality, complexity, and suggests improvements. Use when asked to review or analyze code.
allowed-tools: Read, Grep, Glob
model: sonnet
---

# Code Analyzer

## Purpose
Analyze code for quality, complexity, and maintainability.

## Process
1. Read the target files
2. Analyze complexity metrics
3. Identify code smells
4. Suggest improvements

## Output Format
- Summary of findings
- Detailed issues list
- Prioritized recommendations
EOF
```

### Skill YAML Fields

| Field | Required | Purpose |
|-------|----------|---------|
| `name` | Yes | Identifier (lowercase, hyphens) |
| `description` | Yes | When to use (crucial for discovery) |
| `allowed-tools` | No | Restrict tool access |
| `model` | No | Specific Claude model |
| `context` | No | `fork` for isolated execution |
| `user-invocable` | No | Show in slash menu (default: true) |
| `disable-model-invocation` | No | Block Skill tool access |

### Multi-File Skills

```
.claude/skills/pdf-processor/
├── SKILL.md           # Main skill definition
├── REFERENCE.md       # Detailed reference
├── EXAMPLES.md        # Usage examples
└── scripts/
    └── validate.py    # Helper scripts
```

### The Skill Tool

Claude can programmatically invoke skills:

```python
# Claude uses the Skill tool internally
Skill(skill="code-analyzer", args="src/components/")
```

### Skill Discovery

Claude discovers skills by matching your request against skill descriptions. Write descriptions that include:

- **What** the skill does
- **When** to use it
- **Keywords** users might mention

**Good description**:
> "Analyzes code quality, complexity metrics, and suggests refactoring. Use when reviewing code, checking quality, or looking for improvements."

**Bad description**:
> "Helps with code"

---

## 11. Subagents & Task Tool

### What Are Subagents?

Subagents are **specialized AI instances** spawned via the Task tool. They:

- Run in isolated context windows
- Have specific tool access
- Can execute concurrently
- Return results to the main agent

### Built-in Subagents

| Subagent | Purpose | Tools | Model |
|----------|---------|-------|-------|
| **general-purpose** | Complex multi-step tasks | All | Sonnet |
| **Plan** | Read-only analysis | Read, Glob, Grep | Sonnet |
| **Explore** | Fast codebase search | Read, Glob, Grep | Haiku |

### Using the Task Tool

```python
# Spawn an exploration agent
Task(
    prompt="Find all authentication-related files",
    subagent_type="Explore",
    description="Search for auth files"
)

# Spawn a general-purpose agent
Task(
    prompt="Refactor the user service",
    subagent_type="general-purpose",
    description="Refactor user service"
)

# Run agent in background
Task(
    prompt="Analyze security vulnerabilities",
    subagent_type="security-analyzer",
    run_in_background=True
)

# Resume a previous agent
Task(
    prompt="Continue the analysis",
    resume="agent-id-abc123"
)
```

### Creating Custom Subagents

**Method 1: Using /agents command**
```bash
/agents
# Select "Create New Agent"
# Follow interactive prompts
```

**Method 2: Manual file creation**
```markdown
<!-- .claude/agents/security-reviewer.md -->
---
name: security-reviewer
description: Security expert. Use PROACTIVELY for security analysis.
tools: Read, Grep, Glob
model: opus
---

You are a security specialist focusing on:

## Authentication & Authorization
- Proper auth mechanisms
- Authorization checks
- Privilege escalation risks

## Data Protection
- Sensitive data exposure
- Encryption usage
- Injection vulnerabilities

For each finding, provide:
- Clear description
- Risk level
- Specific fix
```

**Method 3: CLI flag**
```bash
claude --agents '{
  "security-reviewer": {
    "description": "Security expert",
    "prompt": "You are a security reviewer...",
    "tools": ["Read", "Grep", "Glob"],
    "model": "opus"
  }
}'
```

### Parallel Agent Execution

Spawn multiple agents concurrently:

```bash
# Request parallel execution
> Analyze this codebase using agents in parallel:
> Agent 1: analyze src/services/
> Agent 2: analyze src/components/
> Agent 3: analyze src/utils/
```

### Subagent Communication

- Subagents communicate through the Task tool
- Main agent orchestrates all communication
- No direct peer-to-peer channels
- Results integrated into main conversation

### Best Practices

1. **Single responsibility** - One focused task per agent
2. **Detailed prompts** - Clear instructions in system prompt
3. **Action-oriented descriptions** - Use "PROACTIVELY" and "MUST BE USED"
4. **Minimal tools** - Grant only necessary tools
5. **Version control** - Store in `.claude/agents/` for team sharing

---

## 12. Hooks System

### What Are Hooks?

Hooks are **user-defined shell commands** that execute at specific lifecycle points. They provide **deterministic control** over Claude's behavior.

### Available Hook Events

| Event | When | Can Block? | Purpose |
|-------|------|------------|---------|
| `PreToolUse` | Before tool execution | Yes | Validate/modify/block tools |
| `PostToolUse` | After tool completion | No | Post-processing, logging |
| `PermissionRequest` | Permission dialog shown | Yes | Auto-allow/deny |
| `UserPromptSubmit` | User submits prompt | Yes | Validate/add context |
| `Notification` | Notification sent | No | Custom notifications |
| `Stop` | Main agent finishes | Yes | Control continuation |
| `SubagentStop` | Subagent finishes | Yes | Control subagent completion |
| `PreCompact` | Before context compression | No | Prepare for compacting |
| `SessionStart` | Session startup | No | Load context, setup |
| `SessionEnd` | Session termination | No | Cleanup, logging |

### Hook Configuration

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'File modification: $TOOL_INPUT'",
            "timeout": 30
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/log-command.sh"
          }
        ]
      }
    ]
  }
}
```

### Hook Matchers

| Pattern | Description |
|---------|-------------|
| `"Write"` | Exact match |
| `"Edit\|Write\|MultiEdit"` | Any of these |
| `"Notebook.*"` | Regex pattern |
| `"*"` or `""` | Match all |
| `"mcp__github__.*"` | MCP tool pattern |

### Environment Variables in Hooks

| Variable | Available In | Value |
|----------|--------------|-------|
| `CLAUDE_PROJECT_DIR` | All hooks | Project root path |
| `CLAUDE_CODE_REMOTE` | All hooks | `"true"` if web |
| `CLAUDE_ENV_FILE` | SessionStart | Env file path |
| `${CLAUDE_PLUGIN_ROOT}` | Plugin hooks | Plugin directory |

### Hook Input (JSON via stdin)

```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/transcript",
  "cwd": "/current/directory",
  "permission_mode": "default",
  "hook_event_name": "PreToolUse",
  "tool_name": "Bash",
  "tool_input": {"command": "npm test"}
}
```

### Exit Codes

| Code | Behavior |
|------|----------|
| `0` | Success - continue normally |
| `1` | Non-blocking error - show stderr, continue |
| `2` | Blocking error - prevent action, show to Claude |

### Practical Examples

**Auto-format on edit:**
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "prettier --write \"$FILE_PATH\" 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

**Block sensitive files:**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$FILE_PATH\" | grep -qE '\\.(env|pem|key)$'; then echo 'Blocked: sensitive file' >&2; exit 2; fi"
          }
        ]
      }
    ]
  }
}
```

**Log all bash commands:**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "echo \"$(date): $COMMAND\" >> ~/.claude/bash.log"
          }
        ]
      }
    ]
  }
}
```

### Security Considerations

1. **Validate inputs** - Sanitize all hook inputs
2. **Quote variables** - Always use `"$VAR"`
3. **Block path traversal** - Detect `..` patterns
4. **Use absolute paths** - Combine with `$CLAUDE_PROJECT_DIR`
5. **Don't log sensitive data** - Avoid logging credentials
6. **Set timeouts** - Prevent resource exhaustion

---

## 13. MCP Integration

### What is MCP?

Model Context Protocol (MCP) allows Claude Code to connect to external services and tools through standardized servers.

### Configuring MCP Servers

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxxx"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://..."
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
    }
  }
}
```

### Popular MCP Servers

| Server | Purpose | Package |
|--------|---------|---------|
| GitHub | GitHub API access | `@modelcontextprotocol/server-github` |
| PostgreSQL | Database queries | `@modelcontextprotocol/server-postgres` |
| Filesystem | Extended file access | `@modelcontextprotocol/server-filesystem` |
| Slack | Slack integration | `@modelcontextprotocol/server-slack` |
| Google Drive | Drive access | `@anthropic/server-gdrive` |
| Jira | Issue tracking | Community servers |

### Managing MCP

```bash
# List MCP servers
/mcp

# Add MCP server
/mcp add github npx @modelcontextprotocol/server-github

# Remove MCP server
/mcp remove github

# Test MCP connection
/mcp test github
```

### Using MCP Tools

MCP tools appear as `mcp__<server>__<tool>`:

```bash
# GitHub operations
mcp__github__create_issue(repo="owner/repo", title="Bug fix")
mcp__github__create_pull_request(...)

# Database queries
mcp__postgres__query(sql="SELECT * FROM users")
```

---

## 14. IDE Integrations

### VS Code

**Installation:**
```bash
# Install from marketplace
code --install-extension anthropic.claude-code

# Or via URL
vscode:extension/anthropic.claude-code
```

**Features:**
- Inline diffs with accept/reject
- @-mentions for files
- Plan review interface
- Multiple conversation tabs
- Integrated terminal

**Shortcuts:**
| Shortcut | Action |
|----------|--------|
| `Alt+K` | Insert @-mention |
| `Cmd/Ctrl+N` | New conversation |
| `Cmd/Ctrl+Shift+P` | Command palette |

### JetBrains IDEs

**Supported:**
- IntelliJ IDEA
- PyCharm
- WebStorm
- GoLand
- CLion
- Rider

**Installation:**
1. Open Settings → Plugins
2. Search "Claude Code"
3. Install and restart

**Features:**
- Code completion
- Inline suggestions
- Remote development support
- WSL integration

### Web Version

Access Claude Code at `claude.ai`:
- Real-time terminal access
- Cloud execution environment
- No local installation needed
- External service integration

---

## 15. Keyboard Shortcuts

### Essential Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Cancel current operation |
| `Ctrl+L` | Clear screen |
| `Ctrl+R` | Reverse history search |
| `Ctrl+B` | Background tasks |
| `Esc+Esc` | Rewind code/conversation |
| `Alt+P` | Switch model |
| `Alt+M` / `Shift+Tab` | Toggle permission modes |

### Text Editing

| Shortcut | Action |
|----------|--------|
| `Ctrl+A` | Beginning of line |
| `Ctrl+E` | End of line |
| `Ctrl+K` | Delete to end of line |
| `Ctrl+U` | Delete entire line |
| `Ctrl+W` | Delete word backward |
| `Alt+B` | Move word backward |
| `Alt+F` | Move word forward |

### Multiline Input

| Method | Works In |
|--------|----------|
| `Shift+Enter` | Most terminals |
| `Option+Enter` | macOS Terminal |
| `\` + `Enter` | All terminals |

### Vim Mode

Enable with `/vim`:
- Full vim keybindings
- Normal, Insert, Visual modes
- Text objects (iw, aw, i", etc.)
- Common motions (w, b, e, 0, $)

---

## 16. Permissions & Security

### Permission Rules

Configure in settings.json:

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Bash(npm:*)",
      "Bash(git:*)"
    ],
    "deny": [
      "Bash(rm -rf:*)",
      "Bash(sudo:*)",
      "Write(.env)",
      "Write(.git/*)"
    ],
    "ask": [
      "Edit",
      "Write",
      "Bash"
    ]
  }
}
```

### Permission Patterns

| Pattern | Description |
|---------|-------------|
| `"Read"` | All Read operations |
| `"Bash(npm:*)"` | Bash commands starting with npm |
| `"Write(.env)"` | Writing .env files |
| `"Edit(src/**)"` | Editing files in src/ |
| `"mcp__github__*"` | All GitHub MCP tools |

### Security Best Practices

1. **Principle of least privilege** - Grant minimal necessary permissions
2. **Deny dangerous commands** - Block `rm -rf`, `sudo`, etc.
3. **Protect sensitive files** - Block `.env`, keys, credentials
4. **Review before allowing** - Use `ask` for sensitive operations
5. **Use hooks for validation** - Add PreToolUse checks
6. **Audit with PostToolUse** - Log sensitive operations
7. **Sandbox when possible** - Use `/sandbox` for untrusted code

### Enterprise Security

| Feature | Purpose |
|---------|---------|
| Managed settings | Organization-wide policies |
| SSO integration | Team authentication |
| Audit logging | Compliance and tracking |
| Allowed hooks only | `allowManagedHooksOnly` |
| IP restrictions | Network-level security |

---

## 17. Best Practices

### For AI Agents Using Claude Code

#### Tool Selection
```
┌─────────────────────────────────────────────────────────┐
│                  Tool Selection Guide                    │
├─────────────────────────────────────────────────────────┤
│  ALWAYS prefer specialized tools:                       │
│  • Read file → Use Read tool (not cat)                 │
│  • Find files → Use Glob tool (not find)               │
│  • Search content → Use Grep tool (not grep/rg)        │
│  • Edit file → Use Edit tool (not sed/awk)             │
│  • Write file → Use Write tool (not echo >)            │
├─────────────────────────────────────────────────────────┤
│  Use Bash ONLY for:                                     │
│  • Git operations                                       │
│  • Package managers (npm, pip, cargo)                   │
│  • Build/test commands                                  │
│  • System commands (docker, make)                       │
└─────────────────────────────────────────────────────────┘
```

#### Task Management
- Use TodoWrite for multi-step tasks
- Mark tasks in_progress before starting
- Mark completed immediately after finishing
- One task in_progress at a time

#### Context Efficiency
- Use `/compact` when context grows large
- Spawn subagents for isolated exploration
- Use Explore agent for codebase searches
- Avoid reading entire large files

#### Code Changes
- Read files before editing
- Make minimal necessary changes
- Don't over-engineer solutions
- Run tests after changes
- Use Edit for modifications (not Write)

### For Developers Customizing Claude Code

#### Memory (CLAUDE.md)
- Keep instructions concise
- Include build/test commands
- Document code style preferences
- List important file locations

#### Hooks
- Keep hooks focused and fast
- Handle errors gracefully
- Use proper exit codes
- Don't block on slow operations

#### Skills
- Write clear descriptions for discovery
- Use progressive disclosure
- Restrict tools appropriately
- Test before deploying

#### Subagents
- Single responsibility per agent
- Action-oriented descriptions
- Version control agent definitions
- Start simple, iterate

---

## 18. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Tool permission denied | Check `/permissions`, add to allow list |
| Hook not executing | Verify path, check timeout, test manually |
| Skill not discovered | Improve description keywords |
| Subagent not invoking | Use explicit "Use the X agent" |
| MCP connection failed | Check server config, test with `/mcp test` |
| Context too large | Use `/compact` or spawn subagents |
| Model rate limited | Switch to different model or wait |

### Diagnostic Commands

```bash
# System health check
claude doctor

# Check configuration
/config

# View permissions
/permissions

# Test MCP servers
/mcp test <server-name>

# View hook status
/hooks

# Check memory files
/memory

# Export session for debugging
/export
```

### Getting Help

- **In-session**: `/help`
- **Documentation**: https://code.claude.com/docs
- **Issues**: https://github.com/anthropics/claude-code/issues
- **Community**: Discord and forums

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│                 Claude Code Quick Reference              │
├─────────────────────────────────────────────────────────┤
│  START SESSION                                          │
│  claude                    Start interactive REPL       │
│  claude -p "query"         One-shot query               │
│  claude -c                 Continue last session        │
├─────────────────────────────────────────────────────────┤
│  ESSENTIAL COMMANDS                                     │
│  /help                     Show all commands            │
│  /model opus               Switch to Opus               │
│  /compact                  Compress context             │
│  /init                     Create CLAUDE.md             │
│  /config                   Open settings                │
├─────────────────────────────────────────────────────────┤
│  KEYBOARD SHORTCUTS                                     │
│  Ctrl+C                    Cancel operation             │
│  Ctrl+L                    Clear screen                 │
│  Esc+Esc                   Rewind changes               │
│  Alt+P                     Switch model                 │
├─────────────────────────────────────────────────────────┤
│  KEY DIRECTORIES                                        │
│  ~/.claude/                User config & commands       │
│  .claude/                  Project config & agents      │
│  ./CLAUDE.md               Project memory               │
├─────────────────────────────────────────────────────────┤
│  SUBAGENT TYPES                                         │
│  Explore                   Fast codebase search         │
│  general-purpose           Complex multi-step tasks     │
│  Plan                      Read-only analysis           │
├─────────────────────────────────────────────────────────┤
│  MODELS                                                 │
│  sonnet                    Balanced (default)           │
│  opus                      Complex reasoning            │
│  haiku                     Fast & cheap                 │
└─────────────────────────────────────────────────────────┘
```

---

## Summary for AI Agents

When operating as an AI agent in Claude Code:

1. **Read before writing** - Always read files before modifying
2. **Use specialized tools** - Prefer Read/Write/Edit over Bash
3. **Manage tasks** - Use TodoWrite for complex work
4. **Spawn subagents** - Delegate exploration to Explore agent
5. **Respect permissions** - Follow configured allow/deny rules
6. **Follow CLAUDE.md** - Read and follow project instructions
7. **Be efficient** - Don't over-engineer, make minimal changes
8. **Test changes** - Run tests after modifications
9. **Use hooks** - Leverage hooks for automation
10. **Compress context** - Use /compact when needed

This guide covers everything needed to optimally use Claude Code. Refer to specific sections for detailed information on each feature.
