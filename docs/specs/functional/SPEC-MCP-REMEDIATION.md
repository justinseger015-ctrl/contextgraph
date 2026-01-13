# Functional Specification: MCP Tools Remediation

## Metadata

| Field | Value |
|-------|-------|
| **ID** | SPEC-MCP-001 |
| **Title** | MCP Tools Remediation - Missing Tools, Parameter Validation, SSE Transport |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Created** | 2026-01-12 |
| **Author** | Specification Agent |
| **Related Issues** | ISS-006, ISS-012, ISS-013 |
| **PRD Sections** | 0.2, 5.1-5.10, 26 |
| **Constitution Sections** | mcp.core_tools, AP-32 |

---

## 1. Overview

### 1.1 Purpose

This specification defines the requirements for completing the MCP (Model Context Protocol) tool implementation to achieve PRD compliance. The current system implements only **39 of ~75 tools** (52%), leaving critical functionality gaps that prevent users from fulfilling their "librarian" role as defined in PRD Section 0.2.

### 1.2 Problem Statement

The Ultimate Context Graph PRD establishes a clear paradigm: **"You are a librarian, not an archivist."** This means the agent's role is to:
- Curate quality (merge, annotate, forget)
- Ensure findability and coherence
- Respond to Pulse suggestions
- Generate epistemic questions when confused

Without the full MCP tool suite, agents cannot:
1. **Curate memories** - No merge_concepts, forget_concept, annotate_node
2. **Navigate the graph** - No get_neighborhood, find_causal_path
3. **Self-reflect** - No reflect_on_memory, critique_context
4. **Generate questions** - No epistemic_action tool
5. **Diagnose issues** - No test_recall_accuracy, search_tombstones

### 1.3 Current State Analysis

**Currently Implemented (39 tools):**

| Category | Tools | Count |
|----------|-------|-------|
| Core | inject_context, store_memory, get_memetic_status, get_graph_manifest, search_graph, utl_status | 6 |
| GWT | get_consciousness_state, get_kuramoto_sync, get_workspace_status, get_ego_state, trigger_workspace_broadcast, adjust_coupling | 6 |
| UTL | gwt/compute_delta_sc | 1 |
| ATC | get_threshold_status, get_calibration_metrics, trigger_recalibration | 3 |
| Dream | trigger_dream, get_dream_status, abort_dream, get_amortized_shortcuts | 4 |
| Neuromod | get_neuromodulation_state, adjust_neuromodulator | 2 |
| Steering | get_steering_feedback | 1 |
| Causal | omni_infer | 1 |
| Teleological | search_teleological, compute_teleological_vector, fuse_embeddings, update_synergy_matrix, manage_teleological_profile | 5 |
| Autonomous | auto_bootstrap_north_star, get_alignment_drift, trigger_drift_correction, get_pruning_candidates, trigger_consolidation, discover_sub_goals, get_autonomous_status | 7 |
| Meta-UTL | get_meta_learning_status, trigger_lambda_recalibration, get_meta_learning_log | 3 |

**Missing (36+ tools):**

| Category | Missing Tools | PRD Section |
|----------|--------------|-------------|
| Curation | merge_concepts, annotate_node, forget_concept, boost_importance, restore_from_hash | 5.3 |
| Navigation | get_neighborhood, get_recent_context, find_causal_path, entailment_query | 5.4 |
| Meta-Cognitive | reflect_on_memory, generate_search_plan, critique_context, hydrate_citation, get_system_instructions, get_system_logs, get_node_lineage | 5.5 |
| Diagnostic | homeostatic_status, check_adversarial, test_recall_accuracy, debug_compare_retrieval, search_tombstones | 5.6 |
| Admin | reload_manifest, temporary_scratchpad | 5.7 |
| GWT Extension | get_johari_classification, epistemic_action | 5.10 |
| Core (query) | query_causal, get_neuromodulation | 5.2 |

### 1.4 Transport Gap

PRD Section 5.1 specifies: `"JSON-RPC 2.0, stdio/SSE"`

Current implementation:
- **stdio**: Implemented (server.rs)
- **TCP**: Implemented (TASK-INTEG-018)
- **SSE (Server-Sent Events)**: NOT IMPLEMENTED

---

## 2. User Stories

### US-MCP-001: Agent Curation Workflow (P0)

**As** an AI agent operating as a "librarian"
**I want** to merge duplicate concepts, annotate nodes with metadata, and soft-delete obsolete memories
**So that** I can maintain a clean, coherent knowledge graph without information duplication

**Acceptance Criteria:**
- AC-01: Given two similar nodes (>0.9 similarity), when I call `merge_concepts`, then the nodes are merged using the specified strategy (summarize|keep_highest)
- AC-02: Given a merge operation, when I do not provide a rationale, then the operation fails with error code -32009
- AC-03: Given a soft-deleted node, when I call `restore_from_hash` within 30 days, then the node is restored
- AC-04: Given a node, when I call `annotate_node`, then metadata is attached without modifying the embedding
- AC-05: Given a call to `forget_concept`, when soft_delete=true (default), then the node is marked deleted but not purged

### US-MCP-002: Epistemic Action Generation (P0)

**As** an AI agent with low coherence (< 0.4)
**I want** the system to generate clarifying questions I should ask the user
**So that** I can reduce entropy and improve understanding

**Acceptance Criteria:**
- AC-01: Given coherence < 0.4 for 3+ exchanges, when I call `epistemic_action`, then a question is returned with expected_entropy_reduction
- AC-02: Given `epistemic_action(force=true)`, when called regardless of coherence, then a question is generated
- AC-03: Given the epistemic action response, then it includes focal_nodes identifying which memories need clarification
- AC-04: Given workspace_empty for 5 seconds (per GWT spec), then `epistemic_action` is automatically suggested in Pulse

### US-MCP-003: Graph Navigation (P1)

**As** an AI agent exploring causal relationships
**I want** to find paths between concepts, get neighborhood context, and query entailment cones
**So that** I can understand WHY things are connected, not just THAT they are connected

**Acceptance Criteria:**
- AC-01: Given a node ID, when I call `get_neighborhood`, then I receive immediate neighbors with edge weights and types
- AC-02: Given two node IDs, when I call `find_causal_path`, then I receive the causal chain (e.g., "UserAuth→JWT→Middleware→RateLimiting")
- AC-03: Given a node in hyperbolic space, when I call `entailment_query`, then I receive ancestors (cones containing node) and descendants (within node's cone)
- AC-04: Given `get_recent_context`, then I receive temporally-recent memories sorted by access time

### US-MCP-004: Meta-Cognitive Reflection (P1)

**As** an AI agent needing to understand my own memory
**I want** to reflect on memory quality, generate search plans, and critique my context
**So that** I can improve retrieval quality and identify blind spots

**Acceptance Criteria:**
- AC-01: Given a goal, when I call `generate_search_plan`, then I receive 3 optimized queries (semantic/causal/code)
- AC-02: Given recent context, when I call `critique_context`, then I receive fact-check results and conflict detection
- AC-03: Given a citation tag like `[node_abc123]`, when I call `hydrate_citation`, then I receive the full node content
- AC-04: Given `reflect_on_memory(goal)`, then I receive a recommended tool sequence to achieve the goal
- AC-05: Given `get_system_instructions`, then I receive the ~300 token mental model from PRD Section 1

### US-MCP-005: Diagnostic and Admin (P2)

**As** a system administrator debugging memory issues
**I want** to check adversarial content, test recall accuracy, and search tombstones
**So that** I can maintain system health and audit deletions

**Acceptance Criteria:**
- AC-01: Given content, when I call `check_adversarial`, then I receive detection results for outliers, prompt injection, circular logic
- AC-02: Given a query and expected results, when I call `test_recall_accuracy`, then I receive precision/recall metrics
- AC-03: Given `search_tombstones`, then I receive soft-deleted nodes within the 30-day recovery window
- AC-04: Given `homeostatic_status`, then I receive importance scaling status, semantic cancer detection, quarantine status

### US-MCP-006: SSE Transport (P2)

**As** a web-based MCP client
**I want** to connect via Server-Sent Events
**So that** I can receive streaming updates without WebSocket complexity

**Acceptance Criteria:**
- AC-01: Given a client connecting to `/sse` endpoint, when establishing connection, then the server sends an initial heartbeat
- AC-02: Given a streaming operation (e.g., long search), when results are ready, then they are streamed as SSE events
- AC-03: Given connection drop, when client reconnects, then session state is preserved for 60 seconds

### US-MCP-007: Parameter Validation (P1)

**As** a developer calling MCP tools
**I want** parameters to be validated against PRD Section 26 constraints
**So that** I receive clear errors instead of undefined behavior

**Acceptance Criteria:**
- AC-01: Given `inject_context` with empty query, then error -32002 (INVALID_PARAMS) is returned
- AC-02: Given `store_memory` without rationale, then error -32009 (MISSING_RATIONALE) is returned
- AC-03: Given `trigger_dream` with invalid phase, then error -32002 with allowed enum values is returned
- AC-04: Given content > 65536 characters, then error -32002 with max length constraint is returned

---

## 3. Requirements

### 3.1 P0 (Critical) - Required for Cognitive Pulse

#### REQ-MCP-P0-001: epistemic_action Tool

**Description:** Implement tool to generate clarifying questions when coherence is low.

**Rationale:** PRD Section 1.8 defines epistemic actions as system-generated questions. The Cognitive Pulse (Section 1.3) requires suggesting `epistemic_action` when entropy > 0.7 and coherence > 0.5 (Unknown quadrant).

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "session_id": {
      "type": "string",
      "description": "Session identifier for context"
    },
    "force": {
      "type": "boolean",
      "default": false,
      "description": "Generate question regardless of coherence state"
    }
  },
  "required": []
}
```

**Output Schema:**
```json
{
  "action_type": "ask_user",
  "question": "string",
  "expected_entropy_reduction": "number [0,1]",
  "focal_nodes": ["uuid[]"],
  "coherence_before": "number",
  "suggested_follow_up": "string | null"
}
```

**Implementation Notes:**
- Must integrate with GWT workspace (empty workspace for 5s triggers this)
- Uses teleological fingerprint to identify highest-entropy embedding spaces
- Question generation should target focal_nodes with lowest coherence

### 3.2 P0 (Critical) - Curation Tools

#### REQ-MCP-P0-002: merge_concepts Tool

**Description:** Merge duplicate or related concept nodes.

**Rationale:** PRD Section 0.2 emphasizes curation. Section 5.3 specifies merge_concepts with strategies.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "source_node_ids": {
      "type": "array",
      "items": { "type": "string", "format": "uuid" },
      "minItems": 2,
      "description": "Node IDs to merge"
    },
    "target_name": {
      "type": "string",
      "minLength": 1,
      "maxLength": 256,
      "description": "Name for merged node"
    },
    "merge_strategy": {
      "type": "string",
      "enum": ["summarize", "keep_highest", "concatenate"],
      "default": "summarize",
      "description": "How to combine content"
    },
    "rationale": {
      "type": "string",
      "minLength": 10,
      "maxLength": 1000,
      "description": "Why these nodes should be merged"
    },
    "force_merge": {
      "type": "boolean",
      "default": false,
      "description": "Merge even if priors incompatible"
    }
  },
  "required": ["source_node_ids", "target_name", "rationale"]
}
```

**Behavior:**
- If priors incompatible (cos_sim < 0.7) and force_merge=false, return error with suggestion to check priors
- summarize: Use LLM to create merged summary
- keep_highest: Keep content from highest-importance source
- concatenate: Join with separator
- Must create reversal_hash for 30-day undo

#### REQ-MCP-P0-003: forget_concept Tool

**Description:** Soft-delete (default) or hard-delete a concept node.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "node_id": {
      "type": "string",
      "format": "uuid"
    },
    "soft_delete": {
      "type": "boolean",
      "default": true,
      "description": "Soft delete allows 30-day recovery"
    },
    "reason": {
      "type": "string",
      "enum": ["obsolete", "duplicate", "incorrect", "user_requested", "semantic_cancer"],
      "description": "Reason for deletion"
    },
    "rationale": {
      "type": "string",
      "minLength": 10,
      "description": "Detailed explanation"
    }
  },
  "required": ["node_id", "reason", "rationale"]
}
```

**Behavior:**
- soft_delete=true: Mark tombstone, retain 30 days
- soft_delete=false: Permanent only if reason='user_requested'
- Return reversal_hash for recovery

#### REQ-MCP-P0-004: annotate_node Tool

**Description:** Add metadata annotations to a node without modifying embeddings.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "node_id": { "type": "string", "format": "uuid" },
    "annotations": {
      "type": "object",
      "properties": {
        "tags": { "type": "array", "items": { "type": "string" } },
        "domain": { "type": "string", "enum": ["Code", "Medical", "Legal", "Creative", "Research", "General"] },
        "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
        "notes": { "type": "string", "maxLength": 2000 },
        "related_concepts": { "type": "array", "items": { "type": "string" } }
      }
    }
  },
  "required": ["node_id", "annotations"]
}
```

#### REQ-MCP-P0-005: boost_importance Tool

**Description:** Manually adjust node importance within bounds.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "node_id": { "type": "string", "format": "uuid" },
    "delta": {
      "type": "number",
      "minimum": -0.5,
      "maximum": 0.5,
      "description": "Importance adjustment"
    },
    "rationale": { "type": "string", "minLength": 10 }
  },
  "required": ["node_id", "delta", "rationale"]
}
```

#### REQ-MCP-P0-006: restore_from_hash Tool

**Description:** Restore soft-deleted node using reversal hash.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "reversal_hash": {
      "type": "string",
      "description": "Hash from original delete/merge operation"
    }
  },
  "required": ["reversal_hash"]
}
```

**Behavior:**
- Must fail if > 30 days since deletion
- Restores node to exact pre-deletion state
- Re-indexes in all HNSW indexes

### 3.3 P1 (High) - Navigation Tools

#### REQ-MCP-P1-001: get_neighborhood Tool

**Description:** Get immediate neighbors of a node with edge information.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "node_id": { "type": "string", "format": "uuid" },
    "depth": {
      "type": "integer",
      "minimum": 1,
      "maximum": 3,
      "default": 1,
      "description": "Traversal depth"
    },
    "edge_types": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["Semantic", "Temporal", "Causal", "Hierarchical", "Relational"]
      },
      "description": "Filter by edge type"
    },
    "min_weight": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "default": 0.3,
      "description": "Minimum edge weight"
    }
  },
  "required": ["node_id"]
}
```

**Output:** Neighbors with edge_type, weight, confidence, direction.

#### REQ-MCP-P1-002: find_causal_path Tool

**Description:** Find causal chain between two nodes.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "source_id": { "type": "string", "format": "uuid" },
    "target_id": { "type": "string", "format": "uuid" },
    "max_hops": {
      "type": "integer",
      "minimum": 1,
      "maximum": 10,
      "default": 5
    },
    "direction": {
      "type": "string",
      "enum": ["forward", "backward", "bidirectional"],
      "default": "forward"
    }
  },
  "required": ["source_id", "target_id"]
}
```

**Output:** Ordered path with each hop's causal relationship.

#### REQ-MCP-P1-003: entailment_query Tool

**Description:** Query hyperbolic entailment cones for hierarchical relationships.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "node_id": { "type": "string", "format": "uuid" },
    "direction": {
      "type": "string",
      "enum": ["ancestors", "descendants", "both"],
      "default": "both"
    },
    "embedding_space": {
      "type": "string",
      "enum": ["E1", "E5", "E7"],
      "default": "E1",
      "description": "Which space to query (typically E1 semantic)"
    }
  },
  "required": ["node_id"]
}
```

**Output:** Ancestors (cones containing node), descendants (within node's cone).

#### REQ-MCP-P1-004: get_recent_context Tool

**Description:** Get temporally-recent memories.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "limit": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100,
      "default": 20
    },
    "since": {
      "type": "string",
      "format": "date-time",
      "description": "Only memories after this time"
    },
    "session_only": {
      "type": "boolean",
      "default": false,
      "description": "Limit to current session"
    }
  },
  "required": []
}
```

### 3.4 P1 (High) - Meta-Cognitive Tools

#### REQ-MCP-P1-005: reflect_on_memory Tool

**Description:** Get recommended tool sequence for a goal.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "goal": {
      "type": "string",
      "minLength": 10,
      "maxLength": 1000,
      "description": "What you're trying to achieve"
    }
  },
  "required": ["goal"]
}
```

**Output:** Ordered list of recommended MCP tools to achieve goal.

#### REQ-MCP-P1-006: generate_search_plan Tool

**Description:** Generate optimized queries for multi-space search.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "goal": {
      "type": "string",
      "minLength": 10,
      "maxLength": 1000
    },
    "focus_spaces": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["semantic", "causal", "code", "temporal", "entity"]
      },
      "description": "Preferred embedding spaces"
    }
  },
  "required": ["goal"]
}
```

**Output:** 3 optimized queries targeting different embedding spaces.

#### REQ-MCP-P1-007: critique_context Tool

**Description:** Fact-check and conflict-detect in recent context.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "context_window": {
      "type": "integer",
      "minimum": 1,
      "maximum": 50,
      "default": 10,
      "description": "Number of recent memories to critique"
    },
    "check_types": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["conflicts", "duplicates", "outdated", "low_confidence"]
      },
      "default": ["conflicts", "duplicates"]
    }
  },
  "required": []
}
```

#### REQ-MCP-P1-008: hydrate_citation Tool

**Description:** Expand citation tags to full content.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "citation_tag": {
      "type": "string",
      "pattern": "^\\[node_[a-f0-9]+\\]$",
      "description": "Citation tag like [node_abc123]"
    }
  },
  "required": ["citation_tag"]
}
```

#### REQ-MCP-P1-009: get_system_instructions Tool

**Description:** Get the ~300 token mental model from PRD Section 1.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {},
  "required": []
}
```

**Output:** Static mental model text that should be kept in agent context.

#### REQ-MCP-P1-010: get_system_logs Tool

**Description:** Get recent system logs for debugging "why don't you remember X?"

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "level": {
      "type": "string",
      "enum": ["error", "warn", "info", "debug"],
      "default": "info"
    },
    "limit": {
      "type": "integer",
      "minimum": 1,
      "maximum": 1000,
      "default": 100
    },
    "component": {
      "type": "string",
      "enum": ["all", "storage", "retrieval", "curation", "dream", "gwt"],
      "default": "all"
    }
  },
  "required": []
}
```

#### REQ-MCP-P1-011: get_node_lineage Tool

**Description:** Get the history of a node (creation, modifications, merges).

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "node_id": { "type": "string", "format": "uuid" }
  },
  "required": ["node_id"]
}
```

#### REQ-MCP-P1-012: get_johari_classification Tool (GWT Extension)

**Description:** Get per-embedder Johari quadrant classification.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "node_id": { "type": "string", "format": "uuid" }
  },
  "required": ["node_id"]
}
```

**Output:**
```json
{
  "quadrants": ["Open", "Blind", ...], // 13 values
  "confidence": [0.9, 0.7, ...], // 13 confidence scores
  "insights": ["Open(E1) but Blind(E5): knows WHAT but not WHY"]
}
```

### 3.5 P2 (Medium) - Diagnostic Tools

#### REQ-MCP-P2-001: homeostatic_status Tool

**Description:** Get homeostatic optimizer status.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {},
  "required": []
}
```

**Output:** Importance scaling, semantic cancer detection, quarantine list.

#### REQ-MCP-P2-002: check_adversarial Tool

**Description:** Check content for adversarial patterns.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "content": {
      "type": "string",
      "maxLength": 65536
    },
    "checks": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["embedding_outlier", "content_misalign", "prompt_injection", "circular_logic"]
      },
      "default": ["embedding_outlier", "prompt_injection"]
    }
  },
  "required": ["content"]
}
```

#### REQ-MCP-P2-003: test_recall_accuracy Tool

**Description:** Test retrieval precision/recall with known queries.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "expected_node_ids": {
      "type": "array",
      "items": { "type": "string", "format": "uuid" }
    },
    "top_k": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100,
      "default": 10
    }
  },
  "required": ["query", "expected_node_ids"]
}
```

**Output:** precision, recall, F1, missing_expected, unexpected_results.

#### REQ-MCP-P2-004: debug_compare_retrieval Tool

**Description:** Compare retrieval across different embedding spaces.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "spaces": {
      "type": "array",
      "items": { "type": "string", "enum": ["E1", "E5", "E7", "E13"] }
    },
    "top_k": { "type": "integer", "default": 10 }
  },
  "required": ["query", "spaces"]
}
```

#### REQ-MCP-P2-005: search_tombstones Tool

**Description:** Search soft-deleted nodes within recovery window.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "deleted_after": { "type": "string", "format": "date-time" },
    "limit": { "type": "integer", "default": 20 }
  },
  "required": []
}
```

### 3.6 P2 (Medium) - Admin Tools

#### REQ-MCP-P2-006: reload_manifest Tool

**Description:** Reload the user manifest file after external edits.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "default": "~/.context-graph/manifest.md"
    }
  },
  "required": []
}
```

#### REQ-MCP-P2-007: temporary_scratchpad Tool

**Description:** Temporary working memory that doesn't persist to graph.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "action": {
      "type": "string",
      "enum": ["write", "read", "clear"]
    },
    "key": { "type": "string" },
    "value": { "type": "string" }
  },
  "required": ["action"]
}
```

### 3.7 Parameter Validation Requirements

#### REQ-MCP-VAL-001: Content Length Constraints

All content fields must enforce:
- `content`: minLength: 1, maxLength: 65536 (PRD Section 4.1)
- `query`: minLength: 1, maxLength: 4096
- `rationale`: minLength: 10, maxLength: 1000

#### REQ-MCP-VAL-002: Required Rationale

Per PRD Section 0.3, missing rationale incurs -0.5 penalty. Tools that modify data MUST require rationale:
- `store_memory`
- `merge_concepts`
- `forget_concept`
- `boost_importance`

#### REQ-MCP-VAL-003: Enum Validation

All enum parameters must return error -32002 with allowed values when invalid:
- `trigger_dream.phase`: nrem|rem|full
- `forget_concept.reason`: obsolete|duplicate|incorrect|user_requested|semantic_cancer
- `merge_concepts.merge_strategy`: summarize|keep_highest|concatenate

#### REQ-MCP-VAL-004: Numeric Bounds

Enforce bounds with clear error messages:
- `importance`: [0.0, 1.0]
- `min_similarity`: [0.0, 1.0]
- `top_k`: [1, 100]
- `boost_importance.delta`: [-0.5, 0.5]

### 3.8 SSE Transport Requirements

#### REQ-MCP-SSE-001: SSE Endpoint

Implement `/sse` endpoint for Server-Sent Events transport.

**Protocol:**
```
GET /sse HTTP/1.1
Accept: text/event-stream

event: message
data: {"jsonrpc":"2.0","method":"initialize",...}

event: heartbeat
data: {"type":"ping","timestamp":1234567890}
```

#### REQ-MCP-SSE-002: Session Persistence

SSE connections must maintain session state:
- Connection ID assigned on connect
- 60-second grace period for reconnection
- Session state preserved across reconnect

#### REQ-MCP-SSE-003: Streaming Results

Long-running operations should stream partial results:
- `search_graph`: Stream results as found
- `trigger_dream`: Stream phase transitions

---

## 4. Edge Cases

### EC-MCP-001: Merge with Incompatible Priors

**Scenario:** User attempts to merge nodes with priors cosine_similarity < 0.7.

**Expected Behavior:**
- If force_merge=false: Return error -32010 (PRIORS_INCOMPATIBLE) with suggestion to review priors
- If force_merge=true: Proceed with merge, create Relational edge instead of merging priors

### EC-MCP-002: Restore After 30-Day Window

**Scenario:** User calls restore_from_hash for a node deleted > 30 days ago.

**Expected Behavior:** Return error -32011 (RECOVERY_WINDOW_EXPIRED) with deletion timestamp.

### EC-MCP-003: Epistemic Action Without Focal Nodes

**Scenario:** System has very low memory count (< 10 nodes).

**Expected Behavior:** Return generic question about user's domain/goals, with empty focal_nodes array.

### EC-MCP-004: Circular Causal Path

**Scenario:** find_causal_path detects a cycle (A→B→C→A).

**Expected Behavior:** Return path up to cycle detection, include cycle_detected=true flag.

### EC-MCP-005: SSE Reconnection During Dream

**Scenario:** Client disconnects during dream phase, reconnects within 60s.

**Expected Behavior:** Resume streaming dream status updates from current phase.

---

## 5. Error States

| Error Code | Name | Condition | Recovery |
|------------|------|-----------|----------|
| -32002 | INVALID_PARAMS | Parameter validation failed | Check parameter constraints |
| -32009 | MISSING_RATIONALE | Required rationale not provided | Add rationale to request |
| -32010 | PRIORS_INCOMPATIBLE | Merge attempted with incompatible priors | Use force_merge=true or check priors |
| -32011 | RECOVERY_WINDOW_EXPIRED | Restore attempted after 30 days | Node is permanently deleted |
| -32012 | NODE_NOT_FOUND | Requested node doesn't exist | Verify node ID |
| -32013 | CIRCULAR_DEPENDENCY | Circular path detected | Review graph structure |
| -32014 | TOMBSTONE_EXISTS | Node already soft-deleted | Use restore_from_hash first |
| -32015 | SEMANTIC_CANCER_DETECTED | Operation would create semantic cancer | Review importance distribution |
| -32016 | SSE_SESSION_EXPIRED | SSE session not found | Reconnect with new session |

---

## 6. Test Plan

### 6.1 Unit Tests - Tool Definitions

| Test ID | Tool | Test Description | Expected |
|---------|------|------------------|----------|
| TC-DEF-001 | All | Tool count matches expected | 75 tools total |
| TC-DEF-002 | All | All tools have valid JSON schemas | Schema validation passes |
| TC-DEF-003 | All | Required fields enforced | Error on missing required |
| TC-DEF-004 | All | Enum values validated | Error on invalid enum |

### 6.2 Integration Tests - Curation Tools

| Test ID | Tool | Test Description | Expected |
|---------|------|------------------|----------|
| TC-CUR-001 | merge_concepts | Merge two similar nodes | Single merged node created |
| TC-CUR-002 | merge_concepts | Merge without rationale | Error -32009 |
| TC-CUR-003 | merge_concepts | Merge incompatible priors | Error -32010 or success with force |
| TC-CUR-004 | forget_concept | Soft delete node | Node tombstoned |
| TC-CUR-005 | forget_concept | Hard delete non-user-requested | Error |
| TC-CUR-006 | restore_from_hash | Restore within window | Node restored |
| TC-CUR-007 | restore_from_hash | Restore after window | Error -32011 |
| TC-CUR-008 | annotate_node | Add tags | Tags attached |
| TC-CUR-009 | boost_importance | Adjust within bounds | Importance updated |
| TC-CUR-010 | boost_importance | Adjust beyond bounds | Error -32002 |

### 6.3 Integration Tests - Navigation Tools

| Test ID | Tool | Test Description | Expected |
|---------|------|------------------|----------|
| TC-NAV-001 | get_neighborhood | Get direct neighbors | Neighbors returned |
| TC-NAV-002 | get_neighborhood | Get depth=3 | 3-hop neighbors |
| TC-NAV-003 | find_causal_path | Find existing path | Path returned |
| TC-NAV-004 | find_causal_path | No path exists | Empty result |
| TC-NAV-005 | find_causal_path | Circular path | Cycle detected |
| TC-NAV-006 | entailment_query | Get ancestors | Containing cones |
| TC-NAV-007 | get_recent_context | Session only | Session memories only |

### 6.4 Integration Tests - Meta-Cognitive Tools

| Test ID | Tool | Test Description | Expected |
|---------|------|------------------|----------|
| TC-META-001 | epistemic_action | Low coherence | Question generated |
| TC-META-002 | epistemic_action | force=true | Question always generated |
| TC-META-003 | generate_search_plan | Valid goal | 3 queries returned |
| TC-META-004 | critique_context | Conflicts present | Conflicts detected |
| TC-META-005 | hydrate_citation | Valid tag | Full content returned |
| TC-META-006 | hydrate_citation | Invalid tag | Error -32012 |
| TC-META-007 | reflect_on_memory | Valid goal | Tool sequence returned |

### 6.5 Integration Tests - GWT Extension

| Test ID | Tool | Test Description | Expected |
|---------|------|------------------|----------|
| TC-GWT-001 | get_johari_classification | Node with mixed quadrants | 13 quadrants returned |
| TC-GWT-002 | get_johari_classification | Insights for cross-space patterns | Insights include patterns |

### 6.6 Integration Tests - Diagnostic Tools

| Test ID | Tool | Test Description | Expected |
|---------|------|------------------|----------|
| TC-DIAG-001 | check_adversarial | Prompt injection | Detected |
| TC-DIAG-002 | check_adversarial | Normal content | Not detected |
| TC-DIAG-003 | test_recall_accuracy | Known query | Precision/recall computed |
| TC-DIAG-004 | search_tombstones | Deleted nodes exist | Tombstones returned |

### 6.7 Integration Tests - Parameter Validation

| Test ID | Tool | Test Description | Expected |
|---------|------|------------------|----------|
| TC-VAL-001 | store_memory | Missing rationale | Error -32009 |
| TC-VAL-002 | store_memory | Content > 65536 | Error -32002 |
| TC-VAL-003 | trigger_dream | Invalid phase | Error with enum values |
| TC-VAL-004 | search_graph | top_k > 100 | Error with bounds |

### 6.8 Integration Tests - SSE Transport

| Test ID | Test Description | Expected |
|---------|------------------|----------|
| TC-SSE-001 | Connect to /sse | Heartbeat received |
| TC-SSE-002 | Send request via SSE | Response received as event |
| TC-SSE-003 | Disconnect and reconnect | Session preserved |
| TC-SSE-004 | Streaming search results | Results streamed |

---

## 7. Implementation Priority

### Phase 1: P0 Critical (Week 1-2)

| Tool | Effort | Dependency |
|------|--------|------------|
| epistemic_action | 8h | GWT workspace integration |
| merge_concepts | 12h | Reversal hash system |
| forget_concept | 6h | Tombstone table |
| restore_from_hash | 4h | forget_concept |
| annotate_node | 4h | None |
| boost_importance | 2h | None |

**Total: ~36 hours**

### Phase 2: P1 High (Week 3-4)

| Tool | Effort | Dependency |
|------|--------|------------|
| get_neighborhood | 4h | Graph traversal |
| find_causal_path | 8h | Causal edge type |
| entailment_query | 6h | Hyperbolic embeddings |
| get_recent_context | 2h | Temporal index |
| reflect_on_memory | 6h | Tool recommendation |
| generate_search_plan | 4h | Multi-space queries |
| critique_context | 8h | Conflict detection |
| hydrate_citation | 2h | Node lookup |
| get_system_instructions | 1h | Static content |
| get_system_logs | 4h | Log aggregation |
| get_node_lineage | 4h | Event history |
| get_johari_classification | 4h | Per-embedder ΔS/ΔC |
| Parameter validation updates | 8h | Schema updates |

**Total: ~61 hours**

### Phase 3: P2 Medium (Week 5-6)

| Tool | Effort | Dependency |
|------|--------|------------|
| homeostatic_status | 4h | Homeostatic optimizer |
| check_adversarial | 6h | Adversarial detection |
| test_recall_accuracy | 4h | Evaluation metrics |
| debug_compare_retrieval | 4h | Multi-space search |
| search_tombstones | 2h | Tombstone table |
| reload_manifest | 4h | File parsing |
| temporary_scratchpad | 2h | Session memory |
| SSE transport | 16h | HTTP server |

**Total: ~42 hours**

---

## 8. Dependencies

- **ISS-001**: Kuramoto 13 oscillators (required for get_johari_classification)
- **ISS-002**: IC dream trigger (related to epistemic_action workspace empty trigger)
- **ISS-003**: KuramotoStepper wiring (required for GWT tools)

---

## 9. Non-Functional Requirements

### NFR-001: Latency

All tool calls must complete within:
- Status/query tools: < 25ms p95
- Curation tools: < 100ms p95
- Navigation tools: < 50ms p95
- SSE heartbeat: Every 30 seconds

### NFR-002: Error Handling

All tools must:
- Return structured JSON-RPC errors
- Include error code, message, and recovery suggestion
- Log errors with context for debugging

### NFR-003: Backwards Compatibility

New tools must not break existing clients:
- New optional parameters have defaults
- Error codes in reserved range (-32100 to -32199)

---

## Appendix A: Tool Name Constants

Add to `crates/context-graph-mcp/src/tools/names.rs`:

```rust
// ========== CURATION TOOLS (TASK-MCP-P0) ==========
pub const MERGE_CONCEPTS: &str = "merge_concepts";
pub const FORGET_CONCEPT: &str = "forget_concept";
pub const ANNOTATE_NODE: &str = "annotate_node";
pub const BOOST_IMPORTANCE: &str = "boost_importance";
pub const RESTORE_FROM_HASH: &str = "restore_from_hash";

// ========== NAVIGATION TOOLS (TASK-MCP-P1) ==========
pub const GET_NEIGHBORHOOD: &str = "get_neighborhood";
pub const GET_RECENT_CONTEXT: &str = "get_recent_context";
pub const FIND_CAUSAL_PATH: &str = "find_causal_path";
pub const ENTAILMENT_QUERY: &str = "entailment_query";

// ========== META-COGNITIVE TOOLS (TASK-MCP-P1) ==========
pub const REFLECT_ON_MEMORY: &str = "reflect_on_memory";
pub const GENERATE_SEARCH_PLAN: &str = "generate_search_plan";
pub const CRITIQUE_CONTEXT: &str = "critique_context";
pub const HYDRATE_CITATION: &str = "hydrate_citation";
pub const GET_SYSTEM_INSTRUCTIONS: &str = "get_system_instructions";
pub const GET_SYSTEM_LOGS: &str = "get_system_logs";
pub const GET_NODE_LINEAGE: &str = "get_node_lineage";

// ========== GWT EXTENSION TOOLS (TASK-MCP-P1) ==========
pub const GET_JOHARI_CLASSIFICATION: &str = "get_johari_classification";
pub const EPISTEMIC_ACTION: &str = "epistemic_action";

// ========== DIAGNOSTIC TOOLS (TASK-MCP-P2) ==========
pub const HOMEOSTATIC_STATUS: &str = "homeostatic_status";
pub const CHECK_ADVERSARIAL: &str = "check_adversarial";
pub const TEST_RECALL_ACCURACY: &str = "test_recall_accuracy";
pub const DEBUG_COMPARE_RETRIEVAL: &str = "debug_compare_retrieval";
pub const SEARCH_TOMBSTONES: &str = "search_tombstones";

// ========== ADMIN TOOLS (TASK-MCP-P2) ==========
pub const RELOAD_MANIFEST: &str = "reload_manifest";
pub const TEMPORARY_SCRATCHPAD: &str = "temporary_scratchpad";
```

---

## Appendix B: Error Code Registry

Add to `crates/context-graph-mcp/src/protocol.rs`:

```rust
// MCP Tool-specific error codes (-32100 to -32199)
pub const MISSING_RATIONALE: i32 = -32009;
pub const PRIORS_INCOMPATIBLE: i32 = -32010;
pub const RECOVERY_WINDOW_EXPIRED: i32 = -32011;
pub const NODE_NOT_FOUND: i32 = -32012;
pub const CIRCULAR_DEPENDENCY: i32 = -32013;
pub const TOMBSTONE_EXISTS: i32 = -32014;
pub const SEMANTIC_CANCER_DETECTED: i32 = -32015;
pub const SSE_SESSION_EXPIRED: i32 = -32016;
```

---

**END OF SPECIFICATION**
