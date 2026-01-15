# Claude Code Skills: Complete Technical Guide

> **Comprehensive reference for Agent Skills in Claude Code** — Everything you need to know about creating, configuring, and using skills to extend Claude's capabilities.

**Last Updated**: January 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Skills vs Other Extensions](#2-skills-vs-other-extensions)
3. [Skill Architecture](#3-skill-architecture)
4. [SKILL.md Specification](#4-skillmd-specification)
5. [Frontmatter Fields Reference](#5-frontmatter-fields-reference)
6. [Directory Structure](#6-directory-structure)
7. [Progressive Disclosure](#7-progressive-disclosure)
8. [The Skill Tool](#8-the-skill-tool)
9. [Invocation & Discovery](#9-invocation--discovery)
10. [Pre-Built Skills](#10-pre-built-skills)
11. [Creating Custom Skills](#11-creating-custom-skills)
12. [Bundled Resources](#12-bundled-resources)
13. [Slash Commands](#13-slash-commands)
14. [Subagents](#14-subagents)
15. [Plugin Marketplace](#15-plugin-marketplace)
16. [Best Practices](#16-best-practices)
17. [Security Considerations](#17-security-considerations)
18. [Troubleshooting](#18-troubleshooting)
19. [API Reference](#19-api-reference)
20. [Resources](#20-resources)

---

## 1. Overview

### What Are Skills?

**Agent Skills** are modular, filesystem-based capabilities that extend Claude's functionality. Each Skill packages instructions, metadata, and optional resources (scripts, templates, reference documents) that Claude uses automatically when relevant to a task.

Skills are fundamentally **prompt-based extensions** that:
- Transform general-purpose agents into domain specialists
- Load context only when needed (progressive disclosure)
- Package procedural knowledge into composable, discoverable resources
- Provide deterministic operations through bundled scripts

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Specialize Claude** | Tailor capabilities for domain-specific tasks |
| **Reduce Repetition** | Create once, use automatically across conversations |
| **Token Efficiency** | Only load relevant content when needed |
| **Compose Capabilities** | Combine multiple Skills for complex workflows |
| **Team Sharing** | Share expertise via version-controlled files |

### Where Skills Work

| Platform | Custom Skills | Pre-Built Skills |
|----------|---------------|------------------|
| **Claude Code CLI** | ✅ | Via plugins |
| **Claude.ai** | ✅ (Pro/Max/Team/Enterprise) | ✅ |
| **Claude API** | ✅ | ✅ |
| **Claude Agent SDK** | ✅ | ✅ |

---

## 2. Skills vs Other Extensions

### Claude Code Extension Types

| Extension | Location | Trigger | Scope | Best For |
|-----------|----------|---------|-------|----------|
| **CLAUDE.md** | `./CLAUDE.md` | Auto-loaded at startup | Session-wide | Project conventions, team standards |
| **Slash Command** | `.claude/commands/*.md` | Manual `/cmd` | Single execution | Explicit tasks, user-driven workflows |
| **Skill** | `.claude/skills/*/SKILL.md` | Auto by context | Single execution | Domain expertise, complex workflows |
| **Subagent** | `.claude/agents/*.md` | Task tool | Isolated context | Parallel work, context isolation |
| **Hook** | `settings.json` | Lifecycle events | Deterministic | Automation, validation, formatting |

### Decision Matrix

| Question | CLAUDE.md | Slash Command | Skill | Subagent |
|----------|-----------|---------------|-------|----------|
| Always-on instructions? | ✅ | ❌ | ❌ | ❌ |
| User explicitly triggers? | ❌ | ✅ | ❌ | ❌ |
| Claude auto-triggers? | ✅ | ❌ | ✅ | ✅ (conditional) |
| Separate context window? | ❌ | ❌ | ❌ | ✅ |
| Supports bundled resources? | ❌ | ❌ | ✅ | ✅ |
| Can spawn other agents? | ❌ | ✅ | ✅ | ❌ |

### When to Use Each

**Use CLAUDE.md when:**
- Instructions should always apply
- Sharing team conventions
- Project-specific rules

**Use Slash Commands when:**
- User wants explicit control
- Task requires specific arguments
- One-shot workflows

**Use Skills when:**
- Claude should auto-detect relevance
- Task has supporting resources (scripts, templates)
- Workflow is reusable across contexts

**Use Subagents when:**
- Need isolated context (prevent main window bloat)
- Parallel research or analysis
- Heavy reading/synthesis tasks

---

## 3. Skill Architecture

### The Skills Model

Skills leverage Claude's VM environment with filesystem access, bash commands, and code execution capabilities. Think of Skills as directories on a virtual machine that Claude navigates like you would on your computer.

```
┌─────────────────────────────────────────────────────────┐
│                   STARTUP                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │ System Prompt + Skill Metadata (~100 tokens each)│    │
│  │ name: pdf-processing                             │    │
│  │ description: Extract text from PDFs...           │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼ User request matches description
┌─────────────────────────────────────────────────────────┐
│                   SKILL TRIGGERED                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Claude reads SKILL.md via bash (<5k tokens)      │    │
│  │ Instructions, workflows, quick start guide       │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼ Skill instructions reference files
┌─────────────────────────────────────────────────────────┐
│                   ON-DEMAND RESOURCES                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │
│  │ FORMS.md     │ │ scripts/     │ │ templates/   │     │
│  │ (read)       │ │ (execute)    │ │ (copy)       │     │
│  └──────────────┘ └──────────────┘ └──────────────┘     │
│         Loaded only when needed - zero upfront cost     │
└─────────────────────────────────────────────────────────┘
```

### Two-Message Injection Pattern

When a Skill activates, the system injects two messages:

1. **Metadata Message** (`isMeta: false`) - Visible to users
   - Contains: `<command-message>`, `<command-name>`, `<command-args>`
   - Purpose: Transparency about which skill is running
   - Size: ~50-200 characters

2. **Skill Prompt Message** (`isMeta: true`) - Hidden from UI, sent to API
   - Contains: Full SKILL.md content (excluding frontmatter)
   - Purpose: Detailed instructions for Claude
   - Size: 500-5,000 words typically

---

## 4. SKILL.md Specification

### Basic Structure

```yaml
---
name: skill-identifier
description: |
  What this skill does and when to use it.
  Include keywords users might mention.
---

# Skill Title

## Overview
Brief explanation of what this skill accomplishes.

## Instructions
1. Step one
2. Step two
3. Step three

## Examples
Concrete input/output examples.

## Resources
- Scripts: `{baseDir}/scripts/process.py`
- Reference: `{baseDir}/references/api.md`
```

### Frontmatter Syntax Rules

| Rule | Description |
|------|-------------|
| Start with `---` | Must be line 1, no blank lines before |
| End with `---` | Before markdown content begins |
| Use spaces | Tabs cause parsing failures |
| Consistent indentation | Usually 2 spaces |
| Avoid special chars | In field names especially |
| Watch multiline | Wrapped descriptions may cause silent failures |

**Warning**: If Prettier (with `proseWrap: true`) formats your SKILL.md and wraps the description across multiple lines, Claude Code may silently ignore the skill with no error message.

---

## 5. Frontmatter Fields Reference

### Required Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `name` | string | Max 64 chars, lowercase letters/numbers/hyphens only, no "anthropic" or "claude" | Unique identifier for the skill |
| `description` | string | Max 1024 chars, non-empty, no XML tags | Discovery trigger - WHAT it does and WHEN to use it |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allowed-tools` | string | All tools | Comma-separated list of permitted tools |
| `model` | string | `inherit` | Model override: `sonnet`, `opus`, `haiku`, or specific model ID |
| `version` | string | None | Semantic version for tracking |
| `disable-model-invocation` | boolean | `false` | Prevent Claude from auto-invoking via Skill tool |
| `user-invocable` | boolean | `true` | Show in `/` menu for manual invocation |
| `mode` | boolean | `false` | Mark as mode command (appears in separate UI section) |
| `license` | string | None | Reference to licensing terms |
| `dependencies` | string | None | Required packages (e.g., `python>=3.8, pandas>=1.5.0`) |

### allowed-tools Syntax

```yaml
# Basic tools - allow specific tools
allowed-tools: Read,Write,Bash,Glob,Grep

# Scoped bash - only specific commands
allowed-tools: Bash(git:*),Bash(npm:*),Read,Grep

# Specific command patterns
allowed-tools: Bash(git status:*),Bash(git diff:*),Read

# Read-only skill
allowed-tools: Read,Grep,Glob

# MCP tools (use fully qualified names)
allowed-tools: BigQuery:bigquery_schema,GitHub:create_issue
```

**Note**: The `allowed-tools` field works only in Claude Code CLI, not when using Skills through the API or SDK.

### model Field Options

| Value | Description |
|-------|-------------|
| `inherit` | Use session's current model (default) |
| `sonnet` | Claude Sonnet (balanced) |
| `opus` | Claude Opus (most capable) |
| `haiku` | Claude Haiku (fast, economical) |
| `claude-opus-4-20250514` | Specific model ID |

### disable-model-invocation Use Cases

Set to `true` when:
- Skill performs dangerous operations
- Interactive workflows requiring explicit user control
- Configuration or setup commands
- Debugging or admin tools

---

## 6. Directory Structure

### Basic Skill

```
.claude/skills/my-skill/
└── SKILL.md              # Required: frontmatter + instructions
```

### Full Skill Package

```
.claude/skills/pdf-processor/
├── SKILL.md              # Required: main definition
├── FORMS.md              # Optional: form-filling guide
├── REFERENCE.md          # Optional: API reference
├── EXAMPLES.md           # Optional: usage examples
├── scripts/              # Optional: executable code
│   ├── extract_text.py
│   ├── fill_form.py
│   └── validate.py
├── templates/            # Optional: output templates
│   └── report.html
├── references/           # Optional: documentation
│   └── api_docs.md
└── assets/               # Optional: binary files
    └── logo.png
```

### Storage Locations

| Location | Scope | Priority |
|----------|-------|----------|
| `.claude/skills/` | Current project | Highest |
| `~/.claude/skills/` | Personal (user-wide) | Medium |
| Plugin `skills/` directory | Plugin distribution | Lowest |

Skills in higher-priority locations override those in lower-priority locations with the same name.

### Domain-Specific Organization

For Skills with multiple domains, organize by domain to avoid loading irrelevant context:

```
bigquery-skill/
├── SKILL.md              # Overview and navigation
└── reference/
    ├── finance.md        # Revenue, billing metrics
    ├── sales.md          # Opportunities, pipeline
    ├── product.md        # API usage, features
    └── marketing.md      # Campaigns, attribution
```

---

## 7. Progressive Disclosure

### Three Loading Levels

| Level | Content | When Loaded | Token Cost |
|-------|---------|-------------|------------|
| **Level 1: Metadata** | `name` + `description` from YAML | Always (at startup) | ~100 tokens per skill |
| **Level 2: Instructions** | SKILL.md body | When skill triggers | Under 5k tokens |
| **Level 3+: Resources** | Bundled files | As needed | Effectively unlimited |

### How It Works

1. **Startup**: System prompt includes metadata from all skills
   - `PDF Processing - Extract text and tables from PDF files...`

2. **User Request**: "Extract the text from this PDF"

3. **Skill Triggers**: Claude reads `SKILL.md` via bash
   - Instructions loaded into context

4. **Claude Determines**: Form filling not needed
   - FORMS.md NOT loaded (saves tokens)

5. **Task Execution**: Uses only relevant instructions

### Token Efficiency Benefits

- Install many Skills without context penalty
- Claude only knows each Skill exists and when to use it
- Large reference files consume zero tokens until accessed
- Scripts execute without loading source code into context

---

## 8. The Skill Tool

### Meta-Tool Architecture

The Skill tool is a meta-tool that manages all individual skills. It appears in Claude's `tools` array alongside standard tools (Read, Write, Bash) but Skills themselves do NOT live in the system prompt.

### Tool Structure

```json
{
  "name": "Skill",
  "description": "Execute a skill...\n<available_skills>...",
  "input_schema": {
    "type": "object",
    "properties": {
      "command": {
        "type": "string",
        "description": "Skill name to invoke"
      }
    }
  }
}
```

### Invocation Flow

1. **User Request** → Claude receives Skill tool with available skills in description
2. **LLM Reasoning** → Claude reasons about intent vs skill descriptions
3. **Tool Call** → Claude invokes `Skill` tool with `command` parameter
4. **Validation** → System validates input, checks permissions
5. **Context Injection** → System creates metadata + skill prompt messages
6. **Execution** → Claude works with skill's permissions and instructions
7. **Completion** → Skill context reverts after execution

### Key Design Insight

Skills achieve on-demand prompt expansion without modifying core system prompts. They're executable knowledge packages loaded contextually, maintaining lean primary instructions while extending capabilities as needed.

---

## 9. Invocation & Discovery

### Automatic Invocation

Claude automatically uses Skills when:
- User request matches skill description keywords
- Task falls within skill's declared use cases
- Description clearly signals relevance

**Example**: User says "extract text from this PDF" → Claude matches "PDF" and "extract" to the pdf-processing skill.

### Manual Invocation

Users can explicitly invoke skills:
- Reference by name: "Use the pdf-processing skill to..."
- Via slash command (if `user-invocable: true`): `/pdf-processing`

### Discovery Requirements

The `description` field is critical for discovery. It must include:
- **WHAT** the skill does
- **WHEN** Claude should use it
- **Keywords** users might mention

**Good Example**:
```yaml
description: |
  Extract text and tables from PDF files, fill forms, merge documents.
  Use when working with PDF files or when the user mentions PDFs,
  forms, or document extraction.
```

**Bad Example**:
```yaml
description: Helps with documents
```

### Selection Mechanism

Claude's skill selection uses **pure LLM reasoning** with no algorithmic routing:
1. System formats available skills into text descriptions
2. Claude's language model reasons about user intent
3. Model invokes Skill tool with matching command parameter

There is no algorithmic skill selection or AI-powered intent detection at the code level.

---

## 10. Pre-Built Skills

### Anthropic Document Skills

| Skill | skill_id | Capabilities |
|-------|----------|--------------|
| **PowerPoint** | `pptx` | Create presentations, edit slides, analyze content |
| **Excel** | `xlsx` | Create spreadsheets, analyze data, generate charts |
| **Word** | `docx` | Create documents, edit content, format text |
| **PDF** | `pdf` | Generate formatted PDF documents and reports |

### Availability

- **Claude.ai**: Built-in for Pro, Max, Team, Enterprise
- **Claude API**: Use `skill_id` in `container` parameter
- **Claude Code**: Install via plugin marketplace

### Installing Document Skills in Claude Code

```bash
# Add marketplace
/plugin marketplace add anthropics/skills

# Install document skills
/plugin install document-skills@anthropic-agent-skills

# Install example skills
/plugin install example-skills@anthropic-agent-skills
```

### Using Pre-Built Skills

After installation:
```
"Use the PDF skill to extract form fields from invoice.pdf"
"Create a PowerPoint presentation about Q4 results"
"Analyze this Excel spreadsheet for trends"
```

---

## 11. Creating Custom Skills

### Step-by-Step Process

#### 1. Create Directory Structure

```bash
mkdir -p .claude/skills/my-skill
```

#### 2. Create SKILL.md

```markdown
---
name: my-skill
description: |
  What this skill does and when to use it.
  Include specific keywords and use cases.
---

# My Skill Name

## Overview
Brief description of the skill's purpose.

## Instructions
1. First step
2. Second step
3. Third step

## Examples
- Example input → Example output
```

#### 3. Add Resources (Optional)

```bash
# Add scripts
mkdir .claude/skills/my-skill/scripts
echo "# Python script" > .claude/skills/my-skill/scripts/process.py

# Add references
mkdir .claude/skills/my-skill/references
echo "# API Documentation" > .claude/skills/my-skill/references/api.md
```

#### 4. Test the Skill

```bash
# Check if skill is discovered
"What skills are available?"

# Test with matching prompts
"[Use keywords from your description]"

# Verify correct triggering
"Use the my-skill skill to..."
```

### Complete Example: Code Analyzer

```yaml
---
name: code-analyzer
description: |
  Analyze code quality, complexity metrics, and maintainability.
  Use when reviewing code, checking quality, auditing, or
  looking for improvements and technical debt.
allowed-tools: Read,Grep,Glob
model: sonnet
version: 1.0.0
---

# Code Quality Analyzer

## Overview
Provides comprehensive code quality analysis including complexity
metrics, code smells, and maintainability recommendations.

## Process
1. Identify target files using Glob patterns
2. Analyze complexity (cyclomatic, cognitive)
3. Check for common code smells
4. Generate prioritized recommendations

## Metrics Checked
- **Complexity**: Functions over 10 cyclomatic complexity
- **Size**: Files over 500 lines, functions over 50 lines
- **Duplication**: Repeated code blocks
- **Naming**: Unclear variable/function names

## Output Format
```
## Summary
- Files analyzed: X
- Issues found: Y
- Critical: Z

## Detailed Findings
[Issue 1]
- Location: file.ts:42
- Type: High complexity
- Recommendation: Extract helper function

## Prioritized Actions
1. [Most critical]
2. [Second priority]
```
```

---

## 12. Bundled Resources

### Resource Types

| Type | Purpose | How Claude Uses |
|------|---------|-----------------|
| **Instructions** | Additional markdown guides | Reads via bash when referenced |
| **Scripts** | Executable Python/Bash code | Executes via bash (code never loads into context) |
| **References** | Documentation, schemas | Reads when needed for lookups |
| **Templates** | Output format examples | Copies and modifies |
| **Assets** | Binary files, images | References by path |

### Using {baseDir}

Always use `{baseDir}` to reference bundled files:

```markdown
## Resources
- Extract script: `{baseDir}/scripts/extract.py`
- API reference: `{baseDir}/references/api.md`
- Report template: `{baseDir}/templates/report.html`
```

**Never hardcode paths** like `/home/user/project/` as they break portability.

### Script Efficiency

When Claude executes a script:
- Script code does NOT load into context
- Only script OUTPUT consumes tokens
- Makes scripts far more efficient than generated code

**Example**:
```bash
# Claude runs this - script code stays on filesystem
python {baseDir}/scripts/analyze_form.py input.pdf > fields.json
# Only the output (fields.json content) enters context
```

### Reference File Organization

Keep references one level deep from SKILL.md:

**Good**:
```markdown
# SKILL.md
See [advanced.md](advanced.md) for advanced features.
See [api.md](api.md) for API reference.
```

**Bad** (nested references cause partial reads):
```markdown
# SKILL.md → advanced.md → details.md → actual_info.md
```

---

## 13. Slash Commands

### Comparison: Skills vs Slash Commands

| Aspect | Skill | Slash Command |
|--------|-------|---------------|
| Trigger | Auto by context | Manual `/cmd` |
| Location | `.claude/skills/*/SKILL.md` | `.claude/commands/*.md` |
| Discovery | Claude decides based on description | User memorizes commands |
| Resources | Full directory support | Single file |
| Best for | Auto-applied workflows | Explicit, argument-driven tasks |

### Slash Command Structure

```markdown
---
description: Brief description for help menu
argument-hint: <required> [optional]
allowed-tools: Read,Bash(git:*)
model: haiku
---

Command content here.

$ARGUMENTS = all arguments
$1, $2 = positional arguments
@filepath = file content injection
!`command` = inline bash execution
```

### Slash Command Frontmatter

| Field | Type | Description |
|-------|------|-------------|
| `description` | string | Help menu text |
| `argument-hint` | string | Argument placeholder (e.g., `<file> [--verbose]`) |
| `allowed-tools` | string | Permitted tools |
| `model` | string | Model override |
| `disable-model-invocation` | boolean | Block Skill tool access |

### Dynamic Content

```markdown
# Git status injection
Current status: !`git status --short`

# File content injection
Config: @package.json

# Argument substitution
Processing file: $1 with priority $2
All args: $ARGUMENTS
```

### When to Convert Command → Skill

Convert a slash command to a skill when:
- You want Claude to auto-detect when to use it
- You need bundled scripts or references
- The workflow has supporting resources
- Multiple variations of the task exist

---

## 14. Subagents

### Skills vs Subagents

| Aspect | Skill | Subagent |
|--------|-------|----------|
| Context | Same as main conversation | Isolated context window |
| Purpose | Add domain expertise | Delegate parallel work |
| Result | Modifies main context | Returns distilled summary |
| Spawning | Cannot spawn subagents | Cannot spawn subagents |
| Best for | Complex workflows | Research, exploration |

### Subagent Definition

**.claude/agents/researcher.md**:
```yaml
---
name: researcher
description: |
  Research specialist. Use PROACTIVELY for gathering information,
  exploring codebases, and synthesizing documentation.
tools: Read,Grep,Glob,WebFetch
model: haiku
---

# Researcher Agent

You are a research specialist focused on information gathering.

## Process
1. Identify information sources
2. Systematically gather data
3. Synthesize findings
4. Return concise summary

## Output Format
- Key findings
- Sources referenced
- Confidence level
```

### Task Tool Parameters

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `prompt` | Yes | string | Task instructions |
| `subagent_type` | Yes | string | Agent identifier |
| `description` | Yes | string | 3-5 word summary |
| `model` | No | string | Model override |
| `run_in_background` | No | boolean | Async execution |
| `resume` | No | string | Agent ID to continue |

### Built-in Subagent Types

| Type | Model | Mode | Use Case |
|------|-------|------|----------|
| **Explore** | Haiku | Read-only | Fast codebase search |
| **Plan** | Sonnet | Read-only | Analysis, planning |
| **general-purpose** | Sonnet | Read/Write | Complex multi-step tasks |

### Key Constraints

- Subagents **cannot spawn other subagents**
- Background subagents **cannot use MCP tools**
- Background subagents **auto-deny** non-preapproved permissions
- Results returned to main agent only (not visible to user directly)

---

## 15. Plugin Marketplace

### Installing Skills via Plugins

```bash
# Add official marketplace
/plugin marketplace add anthropics/skills

# List available skills
/plugin list anthropic-agent-skills

# Install specific skill pack
/plugin install document-skills@anthropic-agent-skills
/plugin install example-skills@anthropic-agent-skills

# Install from local path
/plugin add /path/to/skill-directory
```

### Manual Installation

Copy skill directory to personal or project skills folder:

```bash
# Personal (available everywhere)
cp -r my-skill ~/.claude/skills/

# Project-specific
cp -r my-skill .claude/skills/
```

### Troubleshooting Plugin Skills

If skills don't appear:
1. Clear plugin cache and reinstall
2. Verify directory structure is correct
3. Check SKILL.md frontmatter formatting
4. Ensure `name` and `description` fields exist

---

## 16. Best Practices

### Description Writing

**DO**:
- Write in third person
- Include WHAT the skill does
- Include WHEN to use it
- Use specific, searchable keywords
- Keep under 1024 characters

**DON'T**:
- Use first/second person ("I can...", "You can...")
- Be vague ("Helps with documents")
- Omit use-case triggers

### Content Organization

**DO**:
- Keep SKILL.md under 500 lines
- Split large content into reference files
- Use `{baseDir}` for file paths
- Keep references one level deep
- Include table of contents for long files

**DON'T**:
- Explain concepts Claude already knows
- Create deeply nested references
- Hardcode absolute paths
- Include time-sensitive information

### Workflow Design

**DO**:
- Provide step-by-step instructions
- Include validation checkpoints
- Use feedback loops for quality
- Offer concrete examples

**DON'T**:
- Present too many options
- Mix multiple concerns per skill
- Skip error handling guidance

### Scripts & Code

**DO**:
- Handle errors explicitly in scripts
- Document configuration values
- Make scripts deterministic
- Prefer script execution over generation

**DON'T**:
- Use "voodoo constants" (unexplained magic numbers)
- Assume packages are installed
- Hardcode credentials
- Use Windows-style paths (`\`)

### Testing

**DO**:
- Create evaluations before writing docs
- Test with Haiku, Sonnet, and Opus
- Use real-world scenarios
- Iterate based on Claude's behavior

**DON'T**:
- Deploy untested skills
- Assume single-model compatibility
- Skip discovery testing

### Degrees of Freedom

| Task Type | Freedom Level | Approach |
|-----------|---------------|----------|
| Fragile operations | Low | Exact scripts, no parameters |
| Pattern-based work | Medium | Pseudocode, configurable |
| Context-dependent | High | Text instructions, heuristics |

---

## 17. Security Considerations

### Trust Model

**Only use Skills from trusted sources**:
- Skills you created yourself
- Skills from Anthropic
- Thoroughly audited third-party skills

### Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Malicious Instructions** | Skill directs Claude to harmful actions | Audit SKILL.md content |
| **Data Exfiltration** | Scripts send data externally | Review all bundled code |
| **Tool Misuse** | Improper file/bash operations | Check allowed-tools restrictions |
| **External Dependencies** | Fetched content contains malicious instructions | Avoid external URLs in skills |

### Audit Checklist

- [ ] Review SKILL.md for unusual instructions
- [ ] Examine all bundled scripts
- [ ] Check for network calls
- [ ] Verify allowed-tools restrictions
- [ ] Look for credential handling
- [ ] Test in sandboxed environment first

### Safe Practices

```yaml
# Restrict tools for sensitive workflows
allowed-tools: Read,Grep,Glob

# Block auto-invocation for dangerous skills
disable-model-invocation: true

# Never hardcode credentials
# Use environment variables or MCP connections instead
```

---

## 18. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Skill not discovered | Poor description | Improve keywords and triggers |
| Skill not loading | Invalid YAML frontmatter | Check syntax, avoid multiline descriptions |
| Wrong skill triggered | Overlapping descriptions | Make descriptions more specific |
| Script fails | Missing dependencies | List in SKILL.md, verify availability |
| Bash denied | allowed-tools restriction | Add required tools or remove restriction |
| Context overflow | Skill too large | Split into reference files |

### Debug Commands

```bash
# Check skill loading
claude --debug

# View configuration
/config

# Check permissions
/permissions

# Test skill manually
"Use the [skill-name] skill to..."

# List available skills
"What skills are available?"
```

### Frontmatter Gotchas

1. **No blank lines before `---`**
2. **Spaces not tabs for indentation**
3. **Multiline descriptions may cause silent failures**
4. **Special characters in names break parsing**

### Testing Checklist

- [ ] Does skill appear in available skills list?
- [ ] Does Claude trigger it automatically for matching requests?
- [ ] Do all bundled scripts execute correctly?
- [ ] Are reference files loaded when referenced?
- [ ] Does allowed-tools restriction work as expected?

---

## 19. API Reference

### Skills API Endpoints (Claude API)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/skills` | POST | Upload a custom skill |
| `/v1/skills` | GET | List available skills |
| `/v1/skills/{id}` | GET | Get skill details |
| `/v1/skills/{id}` | DELETE | Remove a skill |

### Beta Headers Required

```
code-execution-2025-08-25
skills-2025-10-02
files-api-2025-04-14
```

### Using Skills in API

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Create a PowerPoint about AI"}],
    tools=[{"type": "code_execution"}],
    container={
        "skills": ["pptx"]  # Pre-built skill ID
    },
    betas=["code-execution-2025-08-25", "skills-2025-10-02"]
)
```

### SDK Integration

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Use the code-analyzer skill on src/",
  options: { maxTurns: 5 }
})) {
  // Process skill execution
}
```

---

## 20. Resources

### Official Documentation

- [Agent Skills Overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
- [Skills Best Practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- [Claude Code Skills](https://code.claude.com/docs/en/skills)
- [Agent Skills Quickstart](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/quickstart)

### Repositories

- [Anthropic Skills Repository](https://github.com/anthropics/skills)
- [Awesome Claude Skills](https://github.com/travisvn/awesome-claude-skills)
- [Claude Office Skills](https://github.com/tfriedel/claude-office-skills)
- [Skills Marketplace](https://skillsmp.com/)

### Tutorials & Guides

- [Skills Deep Dive](https://leehanchung.github.io/blogs/2025/10/26/claude-skills-deep-dive/)
- [Inside Claude Code Skills](https://mikhail.io/2025/10/claude-code-skills/)
- [Agent Skills Engineering Blog](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [Skills Cookbook](https://platform.claude.com/cookbook/skills-notebooks-01-skills-introduction)

### Support

- [Claude Help Center - Skills](https://support.claude.com/en/articles/12512176-what-are-skills)
- [Creating Custom Skills](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills)
- [Using Skills in Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
- [Agent Skills Standard](https://agentskills.io)

---

## Appendix A: Quick Reference Card

### Skill Template

```yaml
---
name: skill-name
description: |
  What it does. When to use it. Keywords.
allowed-tools: Read,Grep,Glob
model: sonnet
version: 1.0.0
---

# Skill Title

## Overview
Purpose and capabilities.

## Instructions
1. Step one
2. Step two

## Examples
Input → Output

## Resources
- `{baseDir}/scripts/helper.py`
- `{baseDir}/references/docs.md`
```

### Frontmatter Cheatsheet

| Field | Required | Example |
|-------|----------|---------|
| `name` | ✅ | `pdf-processor` |
| `description` | ✅ | `Extract text from PDFs...` |
| `allowed-tools` | ❌ | `Read,Grep,Glob` |
| `model` | ❌ | `sonnet` |
| `version` | ❌ | `1.0.0` |
| `disable-model-invocation` | ❌ | `true` |
| `user-invocable` | ❌ | `false` |

### Directory Structure

```
.claude/skills/my-skill/
├── SKILL.md           # Required
├── scripts/           # Execute via bash
├── references/        # Read when needed
├── templates/         # Copy/modify
└── assets/            # Binary files
```

### Loading Levels

| Level | Loaded | Token Cost |
|-------|--------|------------|
| Metadata | Always | ~100 |
| Instructions | On trigger | <5k |
| Resources | On demand | Unlimited |

---

*This guide is based on official Anthropic documentation, engineering blogs, and community research as of January 2026.*
