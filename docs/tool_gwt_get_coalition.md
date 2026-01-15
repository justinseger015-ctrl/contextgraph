# FORENSIC INVESTIGATION: gwt_get_coalition

## CASE SUMMARY

**Case ID**: TOOL-GWT-COALITION-001
**Date**: 2026-01-14
**Subject**: Investigation of `gwt_get_coalition` MCP tool
**Investigator**: Sherlock Holmes (Forensic Code Agent)

---

## VERDICT: TOOL DOES NOT EXIST

**GUILTY OF NON-EXISTENCE**

After exhaustive forensic analysis of the entire codebase at `/home/cabdru/contextgraph`, I must report that the tool named `gwt_get_coalition` **DOES NOT EXIST**.

---

## EVIDENCE LOG

### Search Results

| Search Pattern | Location Searched | Matches Found |
|----------------|-------------------|---------------|
| `gwt_get_coalition` | Full codebase | **0** |
| `get_coalition` | Full codebase | **0** |
| `coalition` | Full codebase | **0** |
| `Coalition` | Full codebase | **0** |

### Chain of Custody

```
TIMESTAMP: 2026-01-14T00:00:00Z
ACTION: grep -r "gwt_get_coalition|coalition" /home/cabdru/contextgraph
RESULT: No matches found
VERIFIED BY: Holmes
```

### Source Files Examined

1. `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/gwt.rs`
   - Contains 9 GWT tools (NONE named `gwt_get_coalition`)

2. `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tests/tools_list.rs`
   - Declares exactly 59 tools in the system
   - "coalition" appears nowhere in tool names

3. `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/workspace/global.rs`
   - GlobalWorkspace uses "candidates", NOT "coalition"
   - Winner-Take-All selection uses `WorkspaceCandidate` struct

---

## THE "COALITION" CONCEPT IN GWT LITERATURE

The user may be referring to the Global Workspace Theory concept of "coalitions" - competing groups of neurons/processes that vie for access to conscious workspace.

**In this codebase**, this concept is implemented differently:

| GWT Literature Term | Codebase Implementation |
|---------------------|-------------------------|
| Coalition | `WorkspaceCandidate` |
| Coalition competition | Winner-Take-All (WTA) selection |
| Coalition strength | Kuramoto order parameter `r` |
| Winning coalition | `active_memory` in GlobalWorkspace |

---

## CLOSEST EQUIVALENT TOOLS

The following tools provide the functionality the user may be seeking:

### 1. `get_workspace_status`

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/gwt.rs:49`

**Description**: Get Global Workspace status including active memory, competing candidates, broadcast state, and coherence threshold.

**What it returns**:
```json
{
  "active_memory": "uuid-123",
  "is_broadcasting": true,
  "has_conflict": false,
  "coherence_threshold": 0.8,
  "conflict_memories": null,
  "broadcast_duration_ms": 100
}
```

**Inputs**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | No | Session ID (optional) |

### 2. `get_consciousness_state`

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/gwt.rs:14`

**Description**: Get current consciousness state including Kuramoto sync (r), consciousness level (C), meta-cognitive score, differentiation, workspace status, and identity coherence.

**What it returns**:
```json
{
  "C": 0.72,
  "r": 0.85,
  "psi": 1.2,
  "meta_score": 0.88,
  "differentiation": 0.79,
  "state": "CONSCIOUS",
  "workspace": {
    "active_memory": "uuid-123",
    "is_broadcasting": true,
    "has_conflict": false
  },
  "identity": {
    "coherence": 0.94,
    "status": "Healthy"
  }
}
```

### 3. `get_coherence_state`

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/gwt.rs:147`

**Description**: High-level coherence summary. Returns Kuramoto order parameter, coherence level classification (High/Medium/Low), workspace broadcasting status, and conflict detection status.

---

## HOW THE GLOBAL WORKSPACE WORKS (PRD Context)

From `/home/cabdru/contextgraph/docs2/contextprd.md` (Section 2.5.3):

```
GlobalWorkspace:
  active_memory: Option<MemoryId>      - Currently "conscious" memory
  coherence_threshold: 0.8             - Minimum r for broadcast
  broadcast_duration: 100ms            - How long memory stays active
  competing_memories: PriorityQueue    - Sorted by r x importance

Broadcast Selection Algorithm:
1. Compute r for all candidate memories
2. Filter: candidates where r >= coherence_threshold
3. Rank: score = r x importance x north_star_alignment
4. Select: top-1 becomes active_memory
5. Broadcast: active_memory visible to all subsystems
6. Inhibit: losing candidates get dopamine reduction
```

---

## RECOMMENDATION

If you need to understand what memories are competing for workspace access (i.e., the "coalition" in GWT terms), use:

```bash
# Get workspace status including competing candidates
call_tool("get_workspace_status", {})

# Get full consciousness state with workspace details
call_tool("get_consciousness_state", {})
```

---

## FORENSIC CONCLUSION

**The tool `gwt_get_coalition` does not exist.**

The user may be:
1. Referencing GWT theoretical literature terminology ("coalition")
2. Expecting a tool that was planned but not implemented
3. Confusing this with another system

The equivalent functionality is provided by `get_workspace_status` and `get_consciousness_state`.

---

**CASE STATUS**: CLOSED - Tool confirmed non-existent
**EVIDENCE PRESERVED**: Yes, in this document
**REMEDIATION**: Use `get_workspace_status` or `get_consciousness_state` instead
