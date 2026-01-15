# Forensic Investigation Report: execute_prune Tool

## SHERLOCK HOLMES CASE FILE

**Case ID**: TOOL-EXECUTE-PRUNE-001
**Date**: 2026-01-14
**Subject**: MCP Tool execute_prune
**Investigator**: Sherlock Holmes (Forensic Code Investigation Agent)

---

## 1. Tool Name and Category

**Tool Name**: `execute_prune`
**Category**: Autonomous North Star System / Memory Maintenance
**Specification Reference**: SPEC-AUTONOMOUS-001, NORTH-012, SEC-06

---

## 2. Core Functionality

The `execute_prune` tool executes soft deletion on identified memory nodes, completing the pruning workflow that begins with `get_pruning_candidates`.

**What It Does**:
1. Accepts a list of node UUIDs to prune and a reason for pruning
2. Validates each UUID and checks for protected nodes (SELF_EGO_NODE)
3. Performs soft delete on each node via TeleologicalMemoryStore
4. Returns comprehensive results including success count, errors, and recovery information
5. Optionally cascades deletion to dependent nodes (not yet fully implemented)

**Safety Guarantees**:
- Uses SOFT DELETE only (30-day recovery per SEC-06)
- Protects SELF_EGO_NODE from deletion (system identity preservation)
- FAIL FAST on invalid UUIDs or storage errors
- Returns detailed error information for each failed node

---

## 3. Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `node_ids` | array[string] | YES | - | Array of node UUIDs to prune |
| `reason` | string (enum) | YES | - | Reason for pruning (audit logging) |
| `cascade` | boolean | NO | false | Also prune dependent nodes and edges |

**Valid Reason Values**:
- `staleness` - Node is too old and unused
- `low_alignment` - Node has drifted from North Star goals
- `redundancy` - Node duplicates other content
- `orphan` - Node has no connections (isolated)

**Evidence - Parameter Definition** (from `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/params.rs:229-242`):
```rust
pub struct ExecutePruneParams {
    pub node_ids: Vec<String>,
    pub reason: String,
    #[serde(default)]
    pub cascade: bool,
}
```

---

## 4. Output Format

**Successful Response**:
```json
{
  "pruned_count": 3,
  "cascade_pruned": 0,
  "errors": [],
  "soft_deleted": true,
  "recoverable_until": "2026-02-13T14:30:00Z",
  "reason": "staleness",
  "note": "Nodes soft-deleted via TeleologicalMemoryStore.delete(id, soft=true). Recoverable for 30 days per SEC-06."
}
```

**Response with Errors**:
```json
{
  "pruned_count": 2,
  "cascade_pruned": 0,
  "errors": [
    {
      "node_id": "invalid-uuid",
      "error": "Invalid UUID: invalid character"
    },
    {
      "node_id": "self_ego_node_xxx",
      "error": "Cannot prune SELF_EGO_NODE - protected system identity node"
    }
  ],
  "soft_deleted": true,
  "recoverable_until": "2026-02-13T14:30:00Z",
  "reason": "low_alignment"
}
```

**Empty Node List Response** (valid per EC-AUTO-08):
```json
{
  "pruned_count": 0,
  "cascade_pruned": 0,
  "errors": [],
  "soft_deleted": true,
  "recoverable_until": "2026-02-13T14:30:00Z",
  "message": "No nodes provided for pruning"
}
```

**Output Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `pruned_count` | integer | Number of nodes successfully soft-deleted |
| `cascade_pruned` | integer | Number of dependent nodes pruned (if cascade=true) |
| `errors` | array | List of nodes that failed to prune with reasons |
| `soft_deleted` | boolean | Always true (per SEC-06 mandate) |
| `recoverable_until` | string (ISO 8601) | Date until which nodes can be recovered |
| `reason` | string | The pruning reason provided |
| `note` | string | Implementation details |

---

## 5. Purpose - Why This Tool Exists

The `execute_prune` tool is essential for memory hygiene in the Context Graph system. It exists to:

### 5.1 Complete the Pruning Workflow
The autonomous system identifies pruning candidates via `get_pruning_candidates`. This tool provides the execution step to actually remove them.

```
[get_pruning_candidates] --> [AI reviews candidates]
                                      |
                                      v
[AI selects subset] --> [execute_prune] --> [Soft delete executed]
```

### 5.2 Prevent Memory Bloat
As memories accumulate, the graph can become:
- **Bloated** with stale content
- **Incoherent** from low-alignment nodes
- **Redundant** with duplicate information
- **Fragmented** with orphan nodes

Pruning maintains graph health and retrieval quality.

### 5.3 Ensure Safe Deletion
By mandating soft delete with 30-day recovery:
- Accidental deletions can be reversed
- Audit trails are maintained
- System identity (SELF_EGO_NODE) is protected

### 5.4 Enable Autonomous Maintenance
Per NORTH-012, the system should be able to self-maintain without human intervention. This tool enables autonomous pruning workflows.

---

## 6. PRD Alignment - Global Workspace Theory Goals

### 6.1 Graph Gardener (PRD Section 7.4)
From PRD 7.4:
> "prune weak edges (<0.1 w, no access), merge near-dupes (>0.95 sim, priors OK), rebalance hyperbolic, rebuild FAISS"

The `execute_prune` tool is the execution mechanism for the Graph Gardener's pruning decisions.

### 6.2 Glymphatic Clearance (PRD Section 7.6)
From PRD 7.6:
> "Background prune low-importance during idle"

This tool enables the glymphatic (brain waste clearance) metaphor by removing low-value memories.

### 6.3 Soft Delete Protocol (PRD Section 17)
From PRD 17:
> "Soft delete default (30d recovery), permanent only: reason='user_requested'+soft_delete=false"

The tool enforces this by ALWAYS using soft delete (hard-coded `soft=true`).

### 6.4 SELF_EGO_NODE Protection (PRD Section 2.5.4)
From PRD 2.5.4, the SELF_EGO_NODE is:
> "A persistent, special node representing the system itself"

The tool protects this node from deletion, preserving system identity.

### 6.5 Memory Lifecycle (PRD Section 2.4)
The Marblestone lambda weights evolve through lifecycle stages:
- **Infancy**: Capture (high novelty tolerance)
- **Growth**: Balanced
- **Maturity**: Curation (high coherence requirement)

Pruning becomes more aggressive in maturity, removing low-coherence memories.

---

## 7. Usage by AI Agents in MCP System

### 7.1 When to Use This Tool

An AI agent should call `execute_prune` when:
1. `get_pruning_candidates` has identified memories for removal
2. The agent has reviewed and approved the candidates
3. Memory hygiene is needed (high entropy, low coherence)
4. The `get_memetic_status` tool returns `suggested_action: "curate"`

### 7.2 Typical Workflow

```
1. Agent calls get_memetic_status
   --> Returns { suggested_action: "curate", curation_tasks: [...] }

2. Agent calls get_pruning_candidates
   --> Returns list of candidates with reasons:
   {
     "candidates": [
       { "memory_id": "abc-123", "reason": "Staleness", "age_days": 45 },
       { "memory_id": "def-456", "reason": "LowAlignment", "alignment": 0.35 }
     ]
   }

3. Agent reviews candidates (may filter based on importance)

4. Agent calls execute_prune:
   {
     "node_ids": ["abc-123", "def-456"],
     "reason": "staleness"
   }

5. System soft-deletes nodes, returns results
```

### 7.3 Integration with Other Tools

| Related Tool | Relationship |
|--------------|--------------|
| `get_pruning_candidates` | Identifies candidates; execute_prune acts on them |
| `get_memetic_status` | Suggests when curation (including pruning) is needed |
| `trigger_consolidation` | Alternative to pruning - merges similar memories |
| `forget_concept` | Similar function but for user-initiated deletion |
| `restore_from_hash` | Reverses soft deletions within 30-day window |
| `get_autonomous_status` | Shows overall pruning service health |

### 7.4 Best Practices

**DO**:
- Review candidates before pruning (don't blindly prune all)
- Use the correct reason for audit logging
- Check `errors` array in response for partial failures
- Consider `trigger_consolidation` as an alternative for redundancy

**DO NOT**:
- Attempt to prune SELF_EGO_NODE
- Use cascade=true unless you understand the implications
- Ignore errors in the response
- Assume all nodes were pruned without checking `pruned_count`

---

## 8. Implementation Details - Key Code Paths

### 8.1 Tool Definition Location
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/autonomous.rs:325-353`

### 8.2 Handler Implementation Location
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/maintenance.rs:440-641`

### 8.3 Parameter Struct Location
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/params.rs:229-242`

### 8.4 Dispatch Registration
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/dispatch.rs:160`
```rust
tool_names::EXECUTE_PRUNE => self.call_execute_prune(id, arguments).await,
```

### 8.5 Tool Name Constant
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs:117`
```rust
pub const EXECUTE_PRUNE: &str = "execute_prune";
```

### 8.6 Critical Code Flow

```rust
// 1. Validate reason is one of valid values
let valid_reasons = ["staleness", "low_alignment", "redundancy", "orphan"];
if !valid_reasons.contains(&params.reason.as_str()) {...}

// 2. Handle empty node_ids (valid no-op per EC-AUTO-08)
if params.node_ids.is_empty() {
    return json!({ "pruned_count": 0, ... });
}

// 3. For each node_id:
for node_id_str in &params.node_ids {
    // 3a. Check SELF_EGO_NODE protection (EC-AUTO-07)
    if node_id_str.to_lowercase().contains("self_ego_node") {
        errors.push(...);
        continue;
    }

    // 3b. Parse UUID
    let uuid = match Uuid::parse_str(node_id_str) {...}

    // 3c. Verify node exists
    match self.teleological_store.retrieve(uuid).await {...}

    // 3d. Perform soft delete
    match self.teleological_store.delete(uuid, true).await {
        Ok(true) => pruned_count += 1,
        Ok(false) => errors.push("may be already deleted"),
        Err(e) => errors.push(e),
    }
}

// 4. Calculate recovery date (30 days per SEC-06)
let recoverable_until = chrono::Utc::now() + chrono::Duration::days(30);
```

### 8.7 SELF_EGO_NODE Protection Implementation

```rust
// From maintenance.rs:514-529
const SELF_EGO_NODE_MARKER: &str = "self_ego_node";

// Check for SELF_EGO_NODE protection
if node_id_str.to_lowercase().contains(SELF_EGO_NODE_MARKER) {
    warn!(
        node_id = %node_id_str,
        "execute_prune: Cannot prune SELF_EGO_NODE - protected"
    );
    errors.push(json!({
        "node_id": node_id_str,
        "error": "Cannot prune SELF_EGO_NODE - protected system identity node"
    }));
    continue;
}
```

### 8.8 Soft Delete Enforcement

The tool ALWAYS passes `soft=true` to the store:
```rust
match self.teleological_store.delete(uuid, true).await {...}
//                                           ^^^^
//                                    HARD-CODED soft=true
```

---

## 9. Forensic Evidence Summary

| Evidence Item | Location | Verified |
|---------------|----------|----------|
| Tool definition with JSON schema | autonomous.rs:325-353 | YES |
| Handler implementation | maintenance.rs:440-641 | YES |
| Parameter structs | params.rs:229-242 | YES |
| Dispatch routing | dispatch.rs:160 | YES |
| Tool name constant | names.rs:117 | YES |
| Valid reason enum | maintenance.rs:481 | YES |
| SELF_EGO_NODE protection | maintenance.rs:514-529 | YES |
| Soft delete enforcement | maintenance.rs:553 | YES |
| 30-day recovery calculation | maintenance.rs:618 | YES |
| Empty node_ids handling | maintenance.rs:494-507 | YES |

---

## 10. Verdict

**INNOCENT**: The `execute_prune` tool is correctly implemented according to the PRD specifications. It:
- Enforces soft delete only (never hard delete)
- Protects SELF_EGO_NODE from deletion
- Validates reason values for audit logging
- Handles empty node_ids gracefully (valid no-op)
- Returns comprehensive error information for partial failures
- Calculates correct 30-day recovery window per SEC-06
- Integrates properly with TeleologicalMemoryStore

The implementation aligns with Global Workspace Theory goals by enabling autonomous memory maintenance while preserving system identity and providing safe recovery options.

**OBSERVATION**: The `cascade` parameter is accepted but not fully implemented. Cascade deletion of dependent nodes and edges is noted as a TODO in the code (maintenance.rs:563-571). This is not a bug - it's a documented limitation.

---

*"Eliminate all other factors, and the one which remains must be the truth."* - Sherlock Holmes

**Case Status**: CLOSED
**Confidence Level**: HIGH
