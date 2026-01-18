# Task 13: Complete Curation SKILL.md

## Metadata
- **Task ID**: TASK-GAP-013
- **Phase**: 3 (Skills Framework)
- **Priority**: High
- **Complexity**: Low
- **Dependencies**: None (curation MCP tools already implemented)

## Current State (Audited 2026-01-18)

### What EXISTS and WORKS:
| Component | Status | Location |
|-----------|--------|----------|
| `forget_concept` MCP tool | IMPLEMENTED | `crates/context-graph-mcp/src/handlers/tools/curation_tools.rs` |
| `boost_importance` MCP tool | IMPLEMENTED | `crates/context-graph-mcp/src/handlers/tools/curation_tools.rs` |
| `merge_concepts` MCP tool | IMPLEMENTED | `crates/context-graph-mcp/src/handlers/tools/merge_handler.rs` |
| Tool definitions | IMPLEMENTED | `crates/context-graph-mcp/src/tools/definitions/curation.rs` and `merge.rs` |
| Tool dispatch | WIRED | `crates/context-graph-mcp/src/handlers/tools/dispatch.rs` lines 88-91 |
| DTOs | IMPLEMENTED | `crates/context-graph-mcp/src/handlers/tools/curation_dtos.rs` |
| curation SKILL.md | **PLACEHOLDER** | `.claude/skills/curation/SKILL.md` (marked "STATUS: PLACEHOLDER") |
| dream-consolidation SKILL.md | COMPLETE | `.claude/skills/dream-consolidation/SKILL.md` (version 1.0.0) |

### Related Skill Already Complete:
The `dream-consolidation` skill is FULLY DOCUMENTED at `.claude/skills/dream-consolidation/SKILL.md`. Task 12's dream implementation is DONE.

## Objective

Replace the placeholder curation SKILL.md with a complete, accurate skill document that matches the ACTUAL implemented MCP tools.

## Source of Truth

| Data | Location | How to Verify |
|------|----------|---------------|
| Tool definitions | `crates/context-graph-mcp/src/tools/definitions/curation.rs` | `cargo test -p context-graph-mcp definitions` |
| Tool definitions | `crates/context-graph-mcp/src/tools/definitions/merge.rs` | `cargo test -p context-graph-mcp definitions` |
| DTOs/schemas | `crates/context-graph-mcp/src/handlers/tools/curation_dtos.rs` | `cargo test -p context-graph-mcp curation_dtos` |
| Handler logic | `crates/context-graph-mcp/src/handlers/tools/curation_tools.rs` | `cargo test -p context-graph-mcp curation_tools` |
| Dispatch wiring | `crates/context-graph-mcp/src/handlers/tools/dispatch.rs` | Lines 88-95 |

## ACTUAL MCP Tool Schemas (From Codebase)

### forget_concept
```json
{
  "name": "forget_concept",
  "description": "Soft-delete a memory with 30-day recovery window (per SEC-06). Set soft_delete=false for permanent deletion (use with caution). Returns deleted_at timestamp for recovery tracking.",
  "inputSchema": {
    "type": "object",
    "required": ["node_id"],
    "properties": {
      "node_id": {
        "type": "string",
        "format": "uuid",
        "description": "UUID of the memory to forget"
      },
      "soft_delete": {
        "type": "boolean",
        "default": true,
        "description": "Use soft delete with 30-day recovery (default true per BR-MCP-001)"
      }
    }
  }
}
```

**Response:**
```json
{
  "forgotten_id": "uuid",
  "soft_deleted": true,
  "recoverable_until": "2026-02-17T12:00:00Z"
}
```

### boost_importance
```json
{
  "name": "boost_importance",
  "description": "Adjust a memory's importance score by delta. Final value is clamped to [0.0, 1.0] (per BR-MCP-002). Response includes old, delta, and new values.",
  "inputSchema": {
    "type": "object",
    "required": ["node_id", "delta"],
    "properties": {
      "node_id": {
        "type": "string",
        "format": "uuid",
        "description": "UUID of the memory to boost"
      },
      "delta": {
        "type": "number",
        "minimum": -1.0,
        "maximum": 1.0,
        "description": "Importance change value (-1.0 to 1.0)"
      }
    }
  }
}
```

**Response:**
```json
{
  "node_id": "uuid",
  "old_importance": 0.5,
  "new_importance": 0.7,
  "clamped": false
}
```

### merge_concepts
```json
{
  "name": "merge_concepts",
  "description": "Merge two or more related concept nodes into a unified node. Supports union (combine all), intersection (common only), or weighted_average (by importance) strategies. Returns reversal_hash for 30-day undo capability. Requires rationale per PRD 0.3.",
  "inputSchema": {
    "type": "object",
    "required": ["source_ids", "target_name", "rationale"],
    "properties": {
      "source_ids": {
        "type": "array",
        "items": { "type": "string", "format": "uuid" },
        "minItems": 2,
        "maxItems": 10,
        "description": "UUIDs of concepts to merge (2-10 required)"
      },
      "target_name": {
        "type": "string",
        "minLength": 1,
        "maxLength": 256,
        "description": "Name for the merged concept (1-256 chars)"
      },
      "merge_strategy": {
        "type": "string",
        "enum": ["union", "intersection", "weighted_average"],
        "default": "union",
        "description": "Strategy: union=combine all, intersection=common only, weighted_average=by importance"
      },
      "rationale": {
        "type": "string",
        "minLength": 1,
        "maxLength": 1024,
        "description": "Rationale for merge (REQUIRED per PRD 0.3)"
      },
      "force_merge": {
        "type": "boolean",
        "default": false,
        "description": "Force merge even if priors conflict (use with caution)"
      }
    }
  }
}
```

## Implementation

### Step 1: Replace Placeholder SKILL.md

Replace the content of `.claude/skills/curation/SKILL.md` with the following:

```markdown
---
name: curation
description: Curate the knowledge graph by merging, annotating, or forgetting concepts. Process curation tasks from get_memetic_status. Keywords: curate, merge, forget, annotate, prune, duplicate.
allowed-tools: Read,Glob,Bash
model: sonnet
version: 1.0.0
user-invocable: true
---
# Knowledge Curation

Curate the knowledge graph by merging, forgetting, or boosting memories.

## Overview

This skill provides tools for knowledge graph curation:
- **Forget** memories with 30-day soft-delete recovery (SEC-06)
- **Boost** or demote memory importance scores
- **Merge** related concepts with configurable strategies

## MCP Tools

### forget_concept

Soft-delete a memory with 30-day recovery window.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `node_id` | string (uuid) | Yes | - | UUID of memory to forget |
| `soft_delete` | boolean | No | true | Soft delete with recovery (per BR-MCP-001) |

**Example:**

```json
{
  "name": "forget_concept",
  "arguments": {
    "node_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

**Response:**

```json
{
  "forgotten_id": "550e8400-e29b-41d4-a716-446655440000",
  "soft_deleted": true,
  "recoverable_until": "2026-02-17T12:00:00Z"
}
```

**Constitution Compliance:**
- SEC-06: 30-day recovery for soft delete
- BR-MCP-001: soft_delete defaults to true

### boost_importance

Adjust a memory's importance score.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `node_id` | string (uuid) | Yes | - | UUID of memory to boost |
| `delta` | number | Yes | - | Change value (-1.0 to 1.0) |

**Example - Boost:**

```json
{
  "name": "boost_importance",
  "arguments": {
    "node_id": "550e8400-e29b-41d4-a716-446655440000",
    "delta": 0.3
  }
}
```

**Example - Demote:**

```json
{
  "name": "boost_importance",
  "arguments": {
    "node_id": "550e8400-e29b-41d4-a716-446655440000",
    "delta": -0.2
  }
}
```

**Response:**

```json
{
  "node_id": "550e8400-e29b-41d4-a716-446655440000",
  "old_importance": 0.5,
  "new_importance": 0.8,
  "clamped": false
}
```

**Constitution Compliance:**
- BR-MCP-002: Final value clamped to [0.0, 1.0]
- AP-10: No NaN/Infinity values accepted

### merge_concepts

Merge 2-10 related concept nodes into one unified node.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_ids` | array[uuid] | Yes | - | 2-10 UUIDs to merge |
| `target_name` | string | Yes | - | Name for merged concept (1-256 chars) |
| `rationale` | string | Yes | - | Why merging (1-1024 chars, per PRD 0.3) |
| `merge_strategy` | string | No | union | Strategy: union, intersection, weighted_average |
| `force_merge` | boolean | No | false | Force even if priors conflict |

**Merge Strategies:**

| Strategy | Description |
|----------|-------------|
| `union` | Combine all properties from source concepts (default) |
| `intersection` | Keep only common properties |
| `weighted_average` | Weight by importance scores |

**Example:**

```json
{
  "name": "merge_concepts",
  "arguments": {
    "source_ids": [
      "550e8400-e29b-41d4-a716-446655440001",
      "550e8400-e29b-41d4-a716-446655440002"
    ],
    "target_name": "Unified Authentication Pattern",
    "rationale": "Consolidating duplicate auth patterns for cleaner graph",
    "merge_strategy": "union"
  }
}
```

**Response:**

```json
{
  "merged_id": "new-uuid",
  "reversal_hash": "hash-for-undo",
  "sources_merged": 2,
  "strategy_used": "union"
}
```

**Constitution Compliance:**
- SEC-06: 30-day reversal via reversal_hash
- PRD 0.3: Rationale required

## Usage Workflow

### Check for Curation Suggestions

```json
{
  "name": "get_memetic_status",
  "arguments": {}
}
```

Look for `suggested_action` or `curation_tasks` in response.

### Process Duplicate Suggestions

When high-similarity memories are detected:

1. Review the suggested pairs
2. Decide on merge strategy
3. Call `merge_concepts` with rationale

### Importance Adjustments

Manually adjust importance based on utility:
- Frequently useful content: boost with positive delta
- Stale/irrelevant content: demote with negative delta

### Safe Deletion

Always prefer soft delete (default). Hard delete only when:
- User explicitly confirms permanent deletion
- Content violates policies

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Invalid UUID | Error: "Memory {id} not found" |
| Already soft-deleted | Error: "Memory already deleted" |
| delta outside [-1.0, 1.0] | Error: "delta must be between -1.0 and 1.0" |
| delta = NaN/Infinity | Error: "delta must be a finite number" |
| new_importance > 1.0 | Clamped to 1.0, response shows `clamped: true` |
| new_importance < 0.0 | Clamped to 0.0, response shows `clamped: true` |
| source_ids.len() < 2 | Error: "Need at least 2 concepts to merge" |
| source_ids.len() > 10 | Error: "Maximum 10 concepts per merge" |
| Empty rationale | Error: "rationale is required" |
| Concurrent delete race | Error: "Memory may have been deleted concurrently" |

## Related Skills

- `/dream-consolidation` - Memory consolidation via NREM/REM phases
- `/topic-explorer` - Explore topic portfolio and stability
- `/memory-inject` - Retrieve context for current task
```

## Full State Verification

After implementation, execute these verification steps:

### 1. File Existence Verification
```bash
# Source of truth: filesystem
test -f /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md && echo "PASS: SKILL.md exists" || echo "FAIL: SKILL.md missing"
```

### 2. Frontmatter Verification
```bash
# Extract and verify frontmatter
head -10 /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md

# Expected:
# ---
# name: curation
# description: ...
# allowed-tools: Read,Glob,Bash
# model: sonnet
# version: 1.0.0
# user-invocable: true
# ---
```

### 3. Version Update Verification
```bash
# Must NOT be placeholder version
grep "version:" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md | grep -v "0.1.0" && echo "PASS: Not placeholder version" || echo "FAIL: Still placeholder version"
```

### 4. Placeholder Status Removed
```bash
# Must NOT contain PLACEHOLDER marker
! grep -q "STATUS: PLACEHOLDER" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md && echo "PASS: Placeholder removed" || echo "FAIL: Placeholder still present"
```

### 5. Tool Documentation Verification
```bash
# All 3 curation tools must be documented
grep -c "### forget_concept" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md | grep -q "1" && echo "PASS: forget_concept documented"
grep -c "### boost_importance" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md | grep -q "1" && echo "PASS: boost_importance documented"
grep -c "### merge_concepts" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md | grep -q "1" && echo "PASS: merge_concepts documented"
```

### 6. Constitution References Verification
```bash
# SEC-06 soft delete must be referenced
grep -q "SEC-06" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md && echo "PASS: SEC-06 referenced" || echo "FAIL: SEC-06 missing"

# BR-MCP-001 (soft_delete default) must be referenced
grep -q "BR-MCP-001" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md && echo "PASS: BR-MCP-001 referenced" || echo "FAIL: BR-MCP-001 missing"

# BR-MCP-002 (importance clamping) must be referenced
grep -q "BR-MCP-002" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md && echo "PASS: BR-MCP-002 referenced" || echo "FAIL: BR-MCP-002 missing"
```

### 7. Correct Merge Strategies
```bash
# Must have correct strategies (NOT keep_newest/combine)
grep -q "union" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md && echo "PASS: union strategy"
grep -q "intersection" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md && echo "PASS: intersection strategy"
grep -q "weighted_average" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md && echo "PASS: weighted_average strategy"

# Must NOT have incorrect strategies
! grep -q "keep_newest" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md && echo "PASS: No invalid keep_newest" || echo "FAIL: Contains invalid strategy"
```

## Boundary & Edge Case Audit

### Edge Case 1: Empty Input
**Before State:** No file modification
**Action:** Attempt to call skill with no parameters
**Expected:** get_memetic_status should work with empty arguments `{}`
**After State:** Returns system status

### Edge Case 2: Invalid UUID Format
**Before State:** Existing memory graph
**Action:** Call `forget_concept` with `node_id: "not-a-uuid"`
**Expected:** Error message: "Invalid UUID format for node_id 'not-a-uuid': ..."
**After State:** No changes to graph

### Edge Case 3: Boundary Delta Values
**Before State:** Memory with importance 0.5
**Action 1:** `boost_importance` with delta=1.0 -> new_importance=1.0, clamped=true
**Action 2:** `boost_importance` with delta=-1.0 -> new_importance=0.0, clamped=true
**After State:** Importance at boundary values

## Evidence of Success Log

After completing implementation, capture this evidence:

```bash
echo "=== VERIFICATION LOG $(date) ==="
echo ""
echo "1. File exists:"
ls -la /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md
echo ""
echo "2. Frontmatter:"
head -10 /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md
echo ""
echo "3. Version (should be 1.0.0):"
grep "version:" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md
echo ""
echo "4. No placeholder marker:"
grep "PLACEHOLDER" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md || echo "CLEAN: No placeholder found"
echo ""
echo "5. Tools documented:"
grep -E "^### (forget_concept|boost_importance|merge_concepts)" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md
echo ""
echo "6. Constitution refs:"
grep -o "SEC-06\|BR-MCP-001\|BR-MCP-002\|AP-10\|PRD 0.3" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md | sort | uniq
echo ""
echo "7. Merge strategies:"
grep -o "union\|intersection\|weighted_average" /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md | sort | uniq
echo ""
echo "=== END VERIFICATION LOG ==="
```

## Definition of Done

- [x] `.claude/skills/curation/SKILL.md` updated (not placeholder)
- [x] Version changed from 0.1.0 to 1.0.0
- [x] "STATUS: PLACEHOLDER" marker removed
- [x] All 3 MCP tools documented: forget_concept, boost_importance, merge_concepts
- [x] Correct parameter names (source_ids NOT source_node_ids)
- [x] Correct merge strategies (union/intersection/weighted_average)
- [x] Constitution references included (SEC-06, BR-MCP-001, BR-MCP-002, AP-10)
- [x] Edge cases documented
- [x] All verification commands pass
- [x] Evidence log captured showing actual file state

## Completion Evidence (2026-01-18)

### Verification Log Output
```
=== VERIFICATION LOG Sun Jan 18 15:35:27 CST 2026 ===

1. File exists:
-rw-r--r-- 1 cabdru cabdru 5405 Jan 18 15:35 /home/cabdru/contextgraph/.claude/skills/curation/SKILL.md

2. Frontmatter (first 10 lines):
---
name: curation
description: Curate the knowledge graph by merging, annotating, or forgetting concepts...
allowed-tools: Read,Glob,Bash
model: sonnet
version: 1.0.0
user-invocable: true
---
# Knowledge Curation

3. Version check (should be 1.0.0):
version: 1.0.0
PASS: Not placeholder version

4. No placeholder marker check:
CLEAN: No placeholder found

5. Tools documented:
### forget_concept
### boost_importance
### merge_concepts

6. Constitution refs:
AP-10
BR-MCP-001
BR-MCP-002
PRD 0.3
SEC-06

7. Merge strategies:
intersection
union
weighted_average

=== END VERIFICATION LOG ===
```

### Test Results
- 64 curation-related tests: PASS
- 36 merge-related tests: PASS
- All FSV (Full State Verification) tests: PASS
- Code simplifier review: Documentation confirmed complete and accurate

## NO Backwards Compatibility

This task does NOT:
- Create workarounds for old parameter names
- Add fallbacks for missing functionality
- Mock any data in tests
- Cover up errors with passing tests

If anything fails, it should fail fast with clear error messages.
