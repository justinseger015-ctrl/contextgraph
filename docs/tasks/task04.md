# Task 04: Add MCP Tool Name Constants

## Metadata
- **Task ID**: TASK-GAP-004
- **Phase**: 2 (MCP Infrastructure)
- **Priority**: High
- **Complexity**: Low
- **Dependencies**: None (standalone - `cargo check` already passes)

## Critical Context for AI Agent

### CURRENT STATE (Verified 2026-01-18)

**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs`

**EXISTING Constants (6)**:
```rust
pub const INJECT_CONTEXT: &str = "inject_context";
pub const STORE_MEMORY: &str = "store_memory";
pub const GET_MEMETIC_STATUS: &str = "get_memetic_status";
pub const SEARCH_GRAPH: &str = "search_graph";
pub const TRIGGER_CONSOLIDATION: &str = "trigger_consolidation";
pub const MERGE_CONCEPTS: &str = "merge_concepts";
```

**COMMENTED OUT (6 - Need to be uncommented)**:
```rust
// pub const GET_TOPIC_PORTFOLIO: &str = "get_topic_portfolio";
// pub const GET_TOPIC_STABILITY: &str = "get_topic_stability";
// pub const DETECT_TOPICS: &str = "detect_topics";
// pub const GET_DIVERGENCE_ALERTS: &str = "get_divergence_alerts";
// pub const FORGET_CONCEPT: &str = "forget_concept";
// pub const BOOST_IMPORTANCE: &str = "boost_importance";
```

### WHAT THIS TASK IS

This task is ONLY about uncommenting 6 string constants. These constants are used for tool dispatch matching (string comparisons). The tool handlers do NOT exist yet - this is just adding the constant definitions.

### WHAT THIS TASK IS NOT

- NOT implementing tool handlers
- NOT adding new tool definitions to `registry.rs` (which currently expects exactly 6 tools)
- NOT modifying tool schemas

## Objective

Uncomment the 6 TODO tool name constants in `names.rs`. This enables future handler implementations to reference these constants for dispatch matching.

## Input Context Files

**READ BEFORE STARTING:**
1. `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs` - Current file with TODOs
2. `/home/cabdru/contextgraph/docs2/constitution.yaml` - Section `mcp.core_tools` for tool listing

## Implementation

### Step 1: Open names.rs and locate TODO section (lines 21-27)

Current content:
```rust
// TODO: Add these PRD-required tools:
// pub const GET_TOPIC_PORTFOLIO: &str = "get_topic_portfolio";
// pub const GET_TOPIC_STABILITY: &str = "get_topic_stability";
// pub const DETECT_TOPICS: &str = "detect_topics";
// pub const GET_DIVERGENCE_ALERTS: &str = "get_divergence_alerts";
// pub const FORGET_CONCEPT: &str = "forget_concept";
// pub const BOOST_IMPORTANCE: &str = "boost_importance";
```

### Step 2: Replace with uncommented constants

Remove the TODO comment and uncomment all 6 constants. Add section headers for organization.

### Final File Content

Replace entire `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs` with:

```rust
//! Tool names as constants for dispatch matching.
//!
//! Per PRD v6 Section 10, these MCP tools should be exposed:
//! - Core: inject_context, search_graph, store_memory, get_memetic_status
//! - Topic: get_topic_portfolio, get_topic_stability, detect_topics, get_divergence_alerts
//! - Consolidation: trigger_consolidation
//! - Curation: merge_concepts, forget_concept, boost_importance

// ========== CORE TOOLS (PRD Section 10.1) ==========
pub const INJECT_CONTEXT: &str = "inject_context";
pub const STORE_MEMORY: &str = "store_memory";
pub const GET_MEMETIC_STATUS: &str = "get_memetic_status";
pub const SEARCH_GRAPH: &str = "search_graph";

// ========== CONSOLIDATION TOOLS (PRD Section 10.1) ==========
pub const TRIGGER_CONSOLIDATION: &str = "trigger_consolidation";

// ========== TOPIC TOOLS (PRD Section 10.2) ==========
pub const GET_TOPIC_PORTFOLIO: &str = "get_topic_portfolio";
pub const GET_TOPIC_STABILITY: &str = "get_topic_stability";
pub const DETECT_TOPICS: &str = "detect_topics";
pub const GET_DIVERGENCE_ALERTS: &str = "get_divergence_alerts";

// ========== CURATION TOOLS (PRD Section 10.3) ==========
pub const MERGE_CONCEPTS: &str = "merge_concepts";
pub const FORGET_CONCEPT: &str = "forget_concept";
pub const BOOST_IMPORTANCE: &str = "boost_importance";
```

## Definition of Done

- [ ] File contains all 12 tool name constants (6 existing + 6 new)
- [ ] No TODO comments remain for tool definitions
- [ ] Constants organized by category with section headers
- [ ] Constant names use SCREAMING_SNAKE_CASE (per constitution naming.vars.const)
- [ ] String values use snake_case (per constitution)
- [ ] `cargo check -p context-graph-mcp` passes
- [ ] `cargo clippy -p context-graph-mcp -- -D warnings` passes

## Full State Verification (MANDATORY)

### Source of Truth
The source of truth for this task is the file `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs`. After modification, the file must contain exactly 12 `pub const` declarations.

### Execute & Inspect

After making changes, run these verification commands:

```bash
cd /home/cabdru/contextgraph

# 1. Verify file compiles
cargo check -p context-graph-mcp
# Expected: "Finished" with no errors

# 2. Verify no clippy warnings
cargo clippy -p context-graph-mcp -- -D warnings
# Expected: "Finished" with no errors

# 3. Count constants in file (SOURCE OF TRUTH VERIFICATION)
grep "^pub const" crates/context-graph-mcp/src/tools/names.rs | wc -l
# EXPECTED OUTPUT: 12

# 4. List all constant names to verify
grep "^pub const" crates/context-graph-mcp/src/tools/names.rs
# EXPECTED: 12 lines, one for each tool constant

# 5. Verify topic tools are defined
grep -E "^pub const (GET_TOPIC_PORTFOLIO|GET_TOPIC_STABILITY|DETECT_TOPICS|GET_DIVERGENCE_ALERTS)" \
    crates/context-graph-mcp/src/tools/names.rs | wc -l
# EXPECTED OUTPUT: 4

# 6. Verify curation tools are defined
grep -E "^pub const (FORGET_CONCEPT|BOOST_IMPORTANCE)" \
    crates/context-graph-mcp/src/tools/names.rs | wc -l
# EXPECTED OUTPUT: 2

# 7. Verify no TODO comments remain
grep -i "TODO" crates/context-graph-mcp/src/tools/names.rs
# EXPECTED OUTPUT: empty (no output)

# 8. Verify no commented-out constants
grep "^// pub const" crates/context-graph-mcp/src/tools/names.rs
# EXPECTED OUTPUT: empty (no output)
```

### Evidence of Success

After running verification commands, capture this evidence:

```
VERIFICATION LOG:
================
Date: [timestamp]
cargo check: [PASS/FAIL]
cargo clippy: [PASS/FAIL]
Constant count: [number] (expected: 12)
Topic tools found: [number] (expected: 4)
Curation tools found: [number] (expected: 2)
TODO comments: [count] (expected: 0)
Commented constants: [count] (expected: 0)
```

### Boundary & Edge Case Audit

**Edge Case 1: Empty input (N/A for this task)**
- This task modifies a Rust source file, not runtime input
- No empty input handling needed

**Edge Case 2: Duplicate constant names**
- Before: Check that no constant name appears twice
- After: `grep "^pub const" names.rs | sort | uniq -d` should return empty
- State verification: `rustc` will fail compilation on duplicate definitions

**Edge Case 3: Invalid constant naming**
- Constant names must be SCREAMING_SNAKE_CASE (e.g., `GET_TOPIC_PORTFOLIO`)
- String values must be snake_case (e.g., `"get_topic_portfolio"`)
- Clippy will catch naming violations

## WARNING: Registry Will NOT Match

The `registry.rs` file asserts exactly 6 tools:
```rust
assert_eq!(actual_count, 6, "Expected 6 tools per PRD v6...");
```

Adding these constants does NOT break this assertion because:
1. Constants are just string definitions
2. The registry loads tools from `definitions/core.rs` and `definitions/merge.rs`
3. No new tool definitions are added by this task

A FUTURE TASK will need to:
1. Add tool definitions to `definitions/topic.rs` (new file)
2. Add tool definitions to `definitions/curation.rs` (extend existing)
3. Update `registry.rs` assertion to expect 12 tools
4. Implement handlers for the new tools

## Fail Fast Behavior

If any verification step fails:
1. DO NOT add workarounds or fallbacks
2. DO NOT modify registry.rs to silence errors
3. STOP and report the exact error
4. The system must compile or fail - no partial states

## Related Files (Read-Only Context)

| File | Purpose | Note |
|------|---------|------|
| `crates/context-graph-mcp/src/tools/registry.rs` | Tool registration | Expects 6 tools currently |
| `crates/context-graph-mcp/src/tools/definitions/core.rs` | Core tool schemas | 5 tools defined |
| `crates/context-graph-mcp/src/tools/definitions/merge.rs` | Merge tool schema | 1 tool defined |
| `crates/context-graph-mcp/src/tools/mod.rs` | Module exports | No changes needed |
| `docs2/constitution.yaml` | PRD v6 tool list | Reference for tool names |

## Naming Convention Reference (from constitution)

```yaml
naming:
  vars:
    const: SCREAMING_SNAKE  # Constant names
  # String values follow snake_case per rust convention
```

Example pattern:
```rust
pub const MY_TOOL_NAME: &str = "my_tool_name";
//         ↑ SCREAMING_SNAKE    ↑ snake_case
```
