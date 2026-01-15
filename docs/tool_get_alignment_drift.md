# MCP Tool Forensic Report: get_alignment_drift

## SHERLOCK HOLMES CASE FILE

**Case ID**: TOOL-DRIFT-001
**Date**: 2026-01-14
**Subject**: Forensic Analysis of `get_alignment_drift` MCP Tool
**Verdict**: VERIFIED - Tool operates correctly with per-embedder analysis

---

## 1. Tool Name and Category

| Attribute | Value |
|-----------|-------|
| **Tool Name** | `get_alignment_drift` |
| **Category** | Autonomous North Star System - Drift Detection |
| **Module** | `context-graph-mcp/src/tools/definitions/autonomous.rs` |
| **Handler** | `context-graph-mcp/src/handlers/autonomous/drift.rs` |
| **Specification** | TASK-AUTONOMOUS-MCP, TASK-INTEG-002, TASK-LOGIC-010 |
| **Core Engine** | `TeleologicalDriftDetector` |

---

## 2. Core Functionality

### 2.1 What It Does

The `get_alignment_drift` tool measures how far stored memories have deviated from the system's primary purpose (North Star goal). It provides a comprehensive **5-level per-embedder drift analysis** across all 13 embedding spaces.

**Key Operations:**

1. **Parses comparison strategy** (cosine, euclidean, synergy, group, cross_correlation)
2. **Retrieves specified memory fingerprints** or provides usage guidance if none specified
3. **Determines reference point**:
   - If North Star exists: compares against North Star teleological array
   - If no North Star (ARCH-03 compliant): computes centroid of stored memories
4. **Executes TeleologicalDriftDetector** for per-embedder drift analysis
5. **Classifies drift levels** (Critical, High, Medium, Low, None) per embedder
6. **Generates recommendations** for addressing drift in specific embedding spaces
7. **Computes trend analysis** if history is available

### 2.2 The 5-Level Drift Classification

| Level | Similarity Range | Interpretation |
|-------|------------------|----------------|
| **Critical** | < 0.40 | Severe misalignment - emergency intervention needed |
| **High** | 0.40 - 0.55 | Significant drift - requires attention |
| **Medium** | 0.55 - 0.70 | Moderate drift - monitoring recommended |
| **Low** | 0.70 - 0.85 | Minor drift - within acceptable range |
| **None** | >= 0.85 | No significant drift detected |

### 2.3 ARCH-03 Compliance

The tool works **WITHOUT a North Star** by computing drift relative to the stored fingerprints' computed centroid. This ensures the system can operate autonomously from first use.

---

## 3. Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `timeframe` | `string` | No | `"24h"` | Timeframe to analyze. Options: `"1h"`, `"24h"`, `"7d"`, `"30d"` |
| `include_history` | `boolean` | No | `false` | Include full drift history in response |
| `memory_ids` | `array[string]` | No | - | Specific memory UUIDs to analyze (required for per-embedder analysis) |
| `strategy` | `string` | No | `"cosine"` | Comparison strategy: `"cosine"`, `"euclidean"`, `"synergy"`, `"group"`, `"cross_correlation"` |

### 3.1 Parameter Struct

```rust
pub struct GetAlignmentDriftParams {
    #[serde(default = "default_timeframe")]
    pub timeframe: String,  // default: "24h"

    #[serde(default)]
    pub include_history: bool,  // default: false
}
```

**Source**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/params.rs` (lines 76-85)

### 3.2 Strategy Options

| Strategy | Description |
|----------|-------------|
| `cosine` | Standard cosine similarity (default) |
| `euclidean` | Euclidean distance-based comparison |
| `synergy` / `synergy_weighted` | Weighted synergy across embedders |
| `group` / `hierarchical` | Hierarchical group-based comparison |
| `cross_correlation` / `dominant` | Cross-correlation with dominant embedder detection |

---

## 4. Output Format

### 4.1 Full Response (with memory_ids)

```json
{
  "overall_drift": {
    "level": "Medium",
    "similarity": 0.62,
    "drift_score": 0.38,
    "has_drifted": true
  },
  "per_embedder_drift": [
    {
      "embedder": "Semantic",
      "embedder_index": 0,
      "similarity": 0.75,
      "drift_score": 0.25,
      "drift_level": "Low"
    },
    {
      "embedder": "Causal",
      "embedder_index": 4,
      "similarity": 0.48,
      "drift_score": 0.52,
      "drift_level": "High"
    }
    // ... 13 total entries
  ],
  "most_drifted_embedders": [
    {
      "embedder": "Causal",
      "embedder_index": 4,
      "similarity": 0.48,
      "drift_score": 0.52,
      "drift_level": "High"
    }
    // ... up to 5 entries, sorted worst-first
  ],
  "recommendations": [
    {
      "embedder": "Causal",
      "priority": "High",
      "issue": "Causal reasoning drift at high level (sim: 0.48)",
      "suggestion": "Strengthen cause-effect relationship tracking"
    }
  ],
  "trend": {
    "direction": "Worsening",
    "velocity": 0.02,
    "samples": 5,
    "projected_critical_in": "10.5 checks at current rate"
  },
  "analyzed_count": 50,
  "timestamp": "2026-01-14T12:00:00Z",
  "legacy_state": {
    "severity": "Moderate",
    "goal_id": "uuid-or-centroid"
  },
  "timeframe": "24h",
  "reference_type": "north_star",
  "reference_id": "uuid-string",
  "arch03_compliant": true,
  "note": "Drift computed relative to North Star goal"
}
```

### 4.2 Minimal Response (without memory_ids)

When no `memory_ids` are provided:

```json
{
  "legacy_state": {
    "severity": "None",
    "trend": "Stable",
    "observation_count": 0,
    "goal_id": "uuid-or-none",
    "reference_type": "north_star",
    "note": "No memory_ids provided. Pass memory_ids array for per-embedder TeleologicalDriftDetector analysis."
  },
  "timeframe": "24h",
  "reference_type": "north_star",
  "north_star_id": "uuid-string",
  "usage_hint": "Provide 'memory_ids' parameter with fingerprint UUIDs for per-embedder drift analysis"
}
```

### 4.3 Error Responses

| Error Code | Condition | Message |
|------------|-----------|---------|
| `-32602` (INVALID_PARAMS) | Invalid strategy | "Unknown search strategy 'xyz'. Valid: cosine, euclidean, synergy, group, cross_correlation" |
| `-32602` (INVALID_PARAMS) | Empty memory_ids | "memory_ids array cannot be empty" |
| `-32101` (FINGERPRINT_NOT_FOUND) | Memory not found | "Memory {id} not found" |
| `-32603` (INTERNAL_ERROR) | Storage error | "Storage error retrieving {id}: {error}" |
| `-32105` (ALIGNMENT_COMPUTATION_ERROR) | Comparison failed | "Comparison failed for {embedder}: {reason}" |

---

## 5. Purpose - Why This Tool Exists

### 5.1 The Problem It Solves

AI memory systems suffer from **drift** - the gradual misalignment of stored knowledge with the system's intended purpose. Without drift detection:

1. Memories accumulate without coherence checks
2. Retrieval returns increasingly irrelevant results
3. System behavior diverges from user intent
4. No early warning of misalignment

### 5.2 Per-Embedder Analysis

From PRD Section 21 "Per-Embedder Johari Classification":

> Memory can be Open(semantic) but Blind(causal) - enables targeted learning

The tool provides **targeted drift detection** for each of the 13 embedding spaces:

| Embedder | What Drift Means |
|----------|------------------|
| E1 Semantic | Core meaning divergence from purpose |
| E5 Causal | Cause-effect reasoning misalignment |
| E7 Code | Technical/code content drift |
| E9 HDC | Holographic pattern inconsistency |
| E13 SPLADE | Keyword/lexical alignment issues |

### 5.3 Early Warning System

The tool serves as an **early warning system** for alignment degradation:

- **Trend analysis** projects when critical drift will occur
- **Per-embedder breakdown** identifies which semantic spaces are drifting
- **Recommendations** provide actionable guidance

---

## 6. PRD Alignment - Global Workspace Theory Goals

### 6.1 Identity Continuity Metric

From PRD Section 2.5.4:

```
IdentityContinuity = cosine(PV_t, PV_{t-1}) x r(t)

Thresholds:
  IC > 0.9  -> Strong continuity (healthy)
  IC < 0.7  -> Identity drift warning
  IC < 0.5  -> Trigger dream consolidation
```

The `get_alignment_drift` tool is the primary mechanism for detecting identity drift.

### 6.2 Teleological Alignment Framework

From PRD Section 4.5:

```
A(v, V) = cos(v, V) = (v . V) / (||v|| x ||V||)

Thresholds:
  theta >= 0.75     -> Optimal alignment
  theta in [0.70, 0.75) -> Acceptable
  theta in [0.55, 0.70) -> Warning
  theta < 0.55     -> Critical misalignment
  Delta_A < -0.15   -> Predicts failure 30-60s ahead
```

The tool's 5-level classification maps directly to these thresholds.

### 6.3 Kuramoto Synchronization Connection

From PRD Section 2.5.2:

```
r > 0.8 -> Memory is coherent ("conscious")
r < 0.5 -> Memory fragmentation alert
```

Drift detection helps maintain the Kuramoto order parameter by identifying memories that break synchronization.

---

## 7. Usage by AI Agents - MCP Integration

### 7.1 When to Call This Tool

| Scenario | Action |
|----------|--------|
| After storing multiple memories | Check overall drift |
| Before important retrieval | Verify alignment status |
| Pulse shows high entropy | Diagnose with drift analysis |
| After dream consolidation | Verify improvement |
| System health monitoring | Periodic drift checks |

### 7.2 Typical Agent Workflow

```
1. Store memories over time
2. Periodically call get_alignment_drift
   a. If drift < 0.70: Continue normal operation
   b. If drift 0.55-0.70: Monitor closely
   c. If drift < 0.55: Call trigger_drift_correction
3. Review per_embedder_drift for targeted issues
4. Follow recommendations for specific embedders
5. Re-check drift after corrections
```

### 7.3 Integration with Other Tools

| Tool | Relationship |
|------|-------------|
| `auto_bootstrap_north_star` | Provides reference for drift calculations |
| `trigger_drift_correction` | Called when drift exceeds thresholds |
| `get_drift_history` | Detailed historical analysis |
| `trigger_dream` | May be called when drift > 0.7 |
| `store_memory` | Source of new fingerprints to analyze |
| `get_memetic_status` | Complementary system health view |

---

## 8. Implementation Details - Key Code Paths

### 8.1 Entry Point

**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/drift.rs`

**Function**: `Handlers::call_get_alignment_drift`

### 8.2 Critical Code Flow

```
call_get_alignment_drift
    |
    +-> Parse GetAlignmentDriftParams
    |
    +-> Parse optional memory_ids from arguments
    |
    +-> Validate strategy parameter
    |   |
    |   +-> FAIL FAST on unknown strategy
    |
    +-> If memory_ids provided:
    |   |
    |   +-> Load each SemanticFingerprint
    |   |   +-> FAIL FAST on not found / storage error
    |   |
    |   +-> Create TeleologicalDriftDetector
    |   |
    |   +-> Determine reference fingerprint:
    |   |   |
    |   |   +-> If North Star exists: Use North Star array
    |   |   +-> Else (ARCH-03): Compute centroid of memories
    |   |
    |   +-> Execute detector.check_drift()
    |   |   +-> FAIL FAST on DriftError
    |   |
    |   +-> Build per_embedder_drift (13 entries)
    |   |
    |   +-> Build most_drifted_embedders (top 5)
    |   |
    |   +-> Generate recommendations
    |   |
    |   +-> Build trend response (if available)
    |   |
    |   +-> Map to legacy_state for backwards compatibility
    |
    +-> Else (no memory_ids):
        |
        +-> Return usage guidance with legacy_state
```

### 8.3 Core Engine: TeleologicalDriftDetector

**File**: `/home/cabdru/contextgraph/crates/context-graph-core/src/autonomous/drift/detector.rs`

```rust
pub struct TeleologicalDriftDetector {
    comparator: TeleologicalComparator,
    history: DriftHistory,
    thresholds: DriftThresholds,
}
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `check_drift()` | Stateless drift analysis |
| `check_drift_with_history()` | Stateful analysis with trend tracking |
| `get_trend()` | Retrieve trend analysis from history |
| `validate_fingerprint()` | FAIL FAST on NaN/Inf values |
| `compute_per_embedder_similarities()` | Per-embedder comparison |
| `generate_recommendations()` | Actionable guidance generation |

### 8.4 Drift Threshold Configuration

```rust
impl Default for DriftThresholds {
    fn default() -> Self {
        Self {
            none_min: 0.85,     // >= 0.85 = No drift
            low_min: 0.70,      // >= 0.70 = Low drift
            medium_min: 0.55,   // >= 0.55 = Medium drift
            high_min: 0.40,     // >= 0.40 = High drift
            // < 0.40 = Critical drift
        }
    }
}
```

### 8.5 Per-Embedder Recommendation Generation

The tool generates targeted recommendations based on which embedder is drifting:

| Embedder | Issue Format | Suggestion |
|----------|--------------|------------|
| Semantic | "Semantic meaning drift at {level} level" | "Review core semantic content alignment with goals" |
| Causal | "Causal reasoning drift at {level} level" | "Strengthen cause-effect relationship tracking" |
| Code | "Code structure drift at {level} level" | "Ensure code-related memories align with technical goals" |
| Entity | "Entity recognition drift at {level} level" | "Ensure named entities are consistently identified" |
| KeywordSplade | "Keyword expansion drift at {level} level" | "Check learned keyword expansion coverage" |

### 8.6 FAIL FAST Points

| Check | Error Type | Location |
|-------|------------|----------|
| Unknown strategy | INVALID_PARAMS | Line 100-111 |
| Memory not found | FINGERPRINT_NOT_FOUND | Line 124-129 |
| Storage retrieval error | INTERNAL_ERROR | Line 130-139 |
| Empty memories array | INVALID_PARAMS | Line 178-184 |
| Comparison failed | ALIGNMENT_COMPUTATION_ERROR | Line 217-240 |

---

## 9. Evidence Summary

### Physical Evidence Examined

| File | Purpose | Lines Examined |
|------|---------|----------------|
| `autonomous.rs` (definitions) | Tool schema | 43-66 |
| `drift.rs` (handler) | Implementation | Lines 1-354 |
| `params.rs` | Parameter structs | 76-85 |
| `detector.rs` | Core drift engine | Full file (425 lines) |
| `types.rs` | Drift types/levels | Full file (423 lines) |
| `error.rs` | Error definitions | Full file |
| `history.rs` | Trend analysis | Full file (157 lines) |
| `contextprd.md` | PRD specification | Sections 2.5, 4.5, 21, 22 |

### Verification Matrix

| Check | Expected | Actual | Verdict |
|-------|----------|--------|---------|
| 5-level drift classification | Critical/High/Medium/Low/None | Implemented | VERIFIED |
| 13-embedder support | All embedders analyzed | 13-element arrays | VERIFIED |
| ARCH-03 compliance | Works without North Star | Computes centroid | VERIFIED |
| Fail-fast behavior | No fallbacks | DriftError propagated | VERIFIED |
| Trend analysis | Linear regression | Implemented | VERIFIED |
| Recommendations | Per-embedder guidance | 13 recommendation types | VERIFIED |

---

## 10. VERDICT

**INNOCENT** - The `get_alignment_drift` tool correctly implements per-embedder drift detection per TASK-LOGIC-010. It:

1. Analyzes all 13 embedding spaces independently
2. Provides actionable recommendations per embedder
3. Works autonomously with or without North Star (ARCH-03)
4. Computes trend projections for proactive intervention
5. Maintains backwards compatibility via legacy_state
6. Fails fast with robust error handling

**Confidence Level**: HIGH

**The case is CLOSED.**

---

*"The little things are infinitely the most important." - Sherlock Holmes*
