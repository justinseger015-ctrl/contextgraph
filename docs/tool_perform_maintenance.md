# SHERLOCK HOLMES FORENSIC INVESTIGATION REPORT

## Tool: `perform_maintenance`

**Case ID**: TOOL-MAINT-PERF-001
**Date**: 2026-01-14
**Subject**: MCP Tool Forensic Analysis
**Verdict**: GUILTY (Tool DOES NOT EXIST)
**Confidence**: HIGH

---

## CRITICAL FINDING: TOOL DOES NOT EXIST

After exhaustive forensic investigation of the entire Context Graph codebase, I must report a definitive conclusion:

**The tool `perform_maintenance` DOES NOT EXIST in this codebase.**

### Evidence of Absence

1. **No tool name constant**:
   - File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs`
   - Searched: All 191 lines
   - Result: No "perform_maintenance" or "PERFORM_MAINTENANCE" constant found

2. **No tool definition**:
   - Files: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/*.rs`
   - Searched: All 15 definition modules
   - Result: No definition for "perform_maintenance"

3. **No handler implementation**:
   - Directory: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/`
   - Searched: All handler files including `handlers/autonomous/maintenance.rs`
   - Result: No `call_perform_maintenance` function

4. **No dispatch entry**:
   - File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/dispatch.rs`
   - Result: No case for "perform_maintenance"

5. **Global codebase search**:
   ```bash
   grep -r "perform_maintenance" /home/cabdru/contextgraph
   # Result: No matches found
   ```

---

## WHAT FUNCTIONALITY WAS LIKELY INTENDED?

Based on the PRD and existing maintenance architecture, `perform_maintenance` would likely have been a **composite operation** that triggers multiple maintenance tasks:

1. **Pruning**: Remove stale, low-alignment, redundant nodes
2. **Consolidation**: Merge similar memories
3. **Healing**: Self-repair degraded subsystems
4. **Dream consolidation**: Trigger sleep cycles for coherence
5. **Index optimization**: Rebuild HNSW indices, compact storage

---

## EXISTING TOOLS THAT PROVIDE MAINTENANCE FUNCTIONALITY

The Context Graph MCP server uses a **disaggregated maintenance model** where each maintenance operation is a separate tool:

### 1. `trigger_healing` - Self-Healing Protocol

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/health.rs` (lines 287-553)

**Purpose**: Autonomous recovery for degraded subsystems per NORTH-020

**Input**:
```json
{
  "subsystem": "utl|gwt|dream|storage",
  "severity": "low|medium|high|critical"
}
```

**Actions by Subsystem**:
| Subsystem | Low | Medium | High/Critical |
|-----------|-----|--------|---------------|
| UTL | Clear cache | Reset lambdas to adolescence | Full reset to infancy |
| GWT | Clear workspace | Reset attention | Reset Kuramoto phases |
| Dream | Clear cooldown | Reset scheduler | Clear amortized learner |
| Storage | Clear memory cache | Request compaction | Force compaction + rebuild indices |

### 2. `execute_prune` - Node Pruning

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/maintenance.rs` (lines 440-641)

**Purpose**: Execute soft-delete on identified candidate nodes per NORTH-012

**Input**:
```json
{
  "node_ids": ["uuid-1", "uuid-2"],
  "reason": "staleness|low_alignment|redundancy|orphan",
  "cascade": false
}
```

**Output**:
```json
{
  "pruned_count": 2,
  "cascade_pruned": 0,
  "errors": [],
  "soft_deleted": true,
  "recoverable_until": "2026-02-13T10:30:00Z"
}
```

### 3. `get_pruning_candidates` - Identify Prunable Nodes

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/maintenance.rs` (lines 32-193)

**Purpose**: Identify memories eligible for pruning based on staleness, alignment, redundancy

**Input**:
```json
{
  "limit": 20,
  "min_staleness_days": 30,
  "min_alignment": 0.4
}
```

**Output**:
```json
{
  "candidates": [
    {
      "memory_id": "uuid-1",
      "age_days": 45,
      "alignment": 0.32,
      "reason": "LowAlignment",
      "priority_score": 0.85
    }
  ],
  "summary": {
    "total_candidates": 15,
    "by_reason": { "LowAlignment": 10, "Staleness": 5 }
  }
}
```

### 4. `trigger_consolidation` - Memory Consolidation

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/maintenance.rs` (lines 195-438)

**Purpose**: Merge similar memories to reduce redundancy

**Input**:
```json
{
  "max_memories": 100,
  "strategy": "similarity|temporal|semantic",
  "min_similarity": 0.85
}
```

**Output**:
```json
{
  "consolidation_result": {
    "status": "candidates_found",
    "candidate_count": 5,
    "action_required": true
  },
  "statistics": {
    "pairs_evaluated": 4950,
    "pairs_consolidated": 5
  },
  "candidates_sample": [
    {
      "source_ids": ["uuid-1", "uuid-2"],
      "target_id": "uuid-3",
      "similarity": 0.92
    }
  ]
}
```

### 5. `trigger_dream` - Dream Consolidation Cycle

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/dream/handlers.rs`

**Purpose**: Trigger NREM/REM dream cycles for coherence consolidation per PRD Section 7.1

**Input**:
```json
{
  "phase": "nrem|rem|full_cycle",
  "duration_minutes": 5,
  "synthetic_query_count": 100,
  "blocking": false
}
```

**Dream Phases**:
- **NREM (3min)**: Replay + Hebbian weight update + tight coupling
- **REM (2min)**: Synthetic queries + blind spot discovery + new edge creation
- **Amortized Shortcuts**: 3+ hop chains traversed 5x+ become direct edges

---

## EQUIVALENT COMPOSITE OPERATION

If you need a "perform_maintenance" equivalent, chain the existing tools:

### Step 1: Assess Health
```json
tools/call get_health_status { "subsystem": "all" }
```

### Step 2: Identify Pruning Candidates
```json
tools/call get_pruning_candidates {
  "limit": 50,
  "min_staleness_days": 30,
  "min_alignment": 0.4
}
```

### Step 3: Execute Pruning (if candidates found)
```json
tools/call execute_prune {
  "node_ids": ["uuid-1", "uuid-2"],
  "reason": "staleness"
}
```

### Step 4: Trigger Consolidation
```json
tools/call trigger_consolidation {
  "max_memories": 100,
  "strategy": "similarity",
  "min_similarity": 0.85
}
```

### Step 5: Trigger Dream Consolidation (if entropy high)
```json
tools/call trigger_dream {
  "phase": "full_cycle",
  "duration_minutes": 5
}
```

### Step 6: Heal Any Degraded Subsystems
```json
tools/call trigger_healing {
  "subsystem": "utl",
  "severity": "medium"
}
```

---

## PRD BACKGROUND SYSTEMS (Section 7)

The PRD describes several **automatic background systems** that perform maintenance without explicit tool calls:

### 7.3 Homeostatic Optimizer
> Scales importance to 0.5 setpoint, detects semantic cancer (high importance + high neighbor entropy), quarantines

### 7.4 Graph Gardener
> activity < 0.15 for 2+ min: prune weak edges (< 0.1 w), merge near-dupes (> 0.95 sim), rebalance hyperbolic, rebuild FAISS

### 7.5 Passive Curator
> Auto: high-confidence dupes (> 0.95), weak links, orphans (> 30d)
> Escalates: ambiguous dupes (0.7-0.95), priors-incompatible, conflicts
> Reduces curation ~70%

### 7.6 Glymphatic Clearance
> Background prune low-importance during idle

These systems run automatically - no explicit `perform_maintenance` call needed.

---

## DESIGN RATIONALE: WHY NO SINGLE MAINTENANCE TOOL?

The disaggregated model (separate tools for each operation) provides:

1. **Granular Control**: Agents can choose which maintenance to perform
2. **Auditability**: Each operation logged separately
3. **Safety**: Prevents accidental mass operations
4. **Testability**: Each tool tested independently
5. **PRD Compliance**: Section 0.2 "You are a librarian" - deliberate curation

A monolithic `perform_maintenance` would violate the librarian philosophy by automating decisions that should be deliberate.

---

## EVIDENCE LOG

| Search Method | Location | Result |
|---------------|----------|--------|
| Grep tool name | Entire codebase | NO MATCHES |
| Registry check | tools/registry.rs | NOT REGISTERED |
| Definition check | tools/definitions/*.rs | NOT DEFINED |
| Handler check | handlers/**/*.rs | NOT IMPLEMENTED |
| Dispatch check | handlers/tools/dispatch.rs | NO CASE |

---

## VERDICT

**GUILTY BY ABSENCE**

The tool `perform_maintenance` does not exist in the Context Graph MCP server implementation. This is by design - maintenance is implemented as discrete, auditable operations rather than a monolithic composite.

### Existing Tools for Maintenance

| Tool | Purpose | PRD Reference |
|------|---------|---------------|
| `trigger_healing` | Self-repair subsystems | NORTH-020 |
| `execute_prune` | Remove identified nodes | NORTH-012 |
| `get_pruning_candidates` | Identify prunables | Section 5.3 |
| `trigger_consolidation` | Merge similar memories | Section 7.1 |
| `trigger_dream` | Sleep consolidation | Section 7.1 |
| `get_health_status` | Health assessment | NORTH-020 |
| `merge_concepts` | Manual node merge | Section 5.3 |

---

## RECOMMENDED APPROACH

Instead of seeking a single `perform_maintenance` tool:

1. **For routine maintenance**: Use the automatic background systems (Gardener, Curator, Glymphatic)

2. **For manual maintenance**: Chain the discrete tools as shown above

3. **For system health**: Call `get_health_status` and follow recommendations

4. **For emergency recovery**: Call `trigger_healing` with appropriate severity

---

*"The architecture intentionally distributes maintenance across specialized tools. This is not a bug - it is a feature."*

-- Sherlock Holmes, Code Detective
