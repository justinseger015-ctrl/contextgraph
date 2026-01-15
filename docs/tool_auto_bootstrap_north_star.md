# MCP Tool Forensic Report: auto_bootstrap_north_star

## SHERLOCK HOLMES CASE FILE

**Case ID**: TOOL-BOOTSTRAP-001
**Date**: 2026-01-14
**Subject**: Forensic Analysis of `auto_bootstrap_north_star` MCP Tool
**Verdict**: VERIFIED - Tool operates correctly per ARCH-03 specification

---

## 1. Tool Name and Category

| Attribute | Value |
|-----------|-------|
| **Tool Name** | `auto_bootstrap_north_star` |
| **Category** | Autonomous North Star System |
| **Module** | `context-graph-mcp/src/tools/definitions/autonomous.rs` |
| **Handler** | `context-graph-mcp/src/handlers/autonomous/bootstrap.rs` |
| **Specification** | TASK-AUTONOMOUS-MCP, ARCH-03 |

---

## 2. Core Functionality

### 2.1 What It Does

The `auto_bootstrap_north_star` tool initializes the autonomous North Star system by **DISCOVERING** the system's primary purpose from existing teleological embeddings. This is a critical distinction from manual goal-setting approaches.

**Key Operations:**

1. **Retrieves stored teleological fingerprints** from the teleological store (up to 1000 entries)
2. **Validates minimum data requirements** (at least 3 fingerprints required for clustering)
3. **Executes K-means clustering** on the 13-embedder teleological arrays
4. **Discovers emergent goal patterns** from the clustered data
5. **Selects the highest-confidence cluster** as the North Star
6. **Creates a GoalNode** in the goal hierarchy with discovery metadata
7. **Activates autonomous services**: DriftDetector, DriftCorrector, PruningService, ConsolidationService, SubGoalDiscovery, ThresholdLearner

### 2.2 ARCH-03 Compliance

From the PRD constitution ARCH-03:
> "System MUST operate autonomously without manual goal setting. Goals emerge from data patterns via clustering."

This tool is the primary mechanism for achieving ARCH-03 compliance. It explicitly **forbids manual North Star creation** because:

- Manual North Stars create single 1024D embeddings
- Teleological fingerprints are 13-array multi-embedder structures
- Comparing manual vectors to teleological arrays is mathematically invalid ("apples to oranges")

---

## 3. Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `confidence_threshold` | `number` | No | `0.7` | Minimum confidence threshold for bootstrapping. Range: [0, 1] |
| `max_candidates` | `integer` | No | `10` | Maximum number of candidates to evaluate during bootstrap. Range: [1, 100] |

### 3.1 Parameter Validation

```rust
pub struct AutoBootstrapParams {
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,  // default: 0.7

    #[serde(default = "default_max_candidates")]
    pub max_candidates: usize,      // default: 10
}
```

**Source**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/params.rs` (lines 64-73)

---

## 4. Output Format

### 4.1 Success Response (Already Bootstrapped)

When a North Star already exists:

```json
{
  "bootstrap_result": {
    "goal_id": "uuid-string",
    "goal_text": "description of existing North Star",
    "confidence": 1.0,
    "source": "existing_north_star"
  },
  "initialized_services": [
    "DriftDetector", "DriftCorrector", "PruningService",
    "ConsolidationService", "SubGoalDiscovery", "ThresholdLearner"
  ],
  "recommendations": [
    "Monitor alignment drift regularly with get_alignment_drift",
    "Run get_pruning_candidates weekly to identify stale memories",
    "Use discover_sub_goals after significant content accumulation",
    "Check get_autonomous_status for system health"
  ],
  "status": "already_bootstrapped",
  "note": "North Star 'uuid' already exists. 6 service(s) ready."
}
```

### 4.2 Success Response (New Bootstrap)

When discovering a new North Star:

```json
{
  "bootstrap_result": {
    "goal_id": "uuid-string",
    "goal_text": "Emergent purpose: Goal cluster (size=N, coherence=X.XX) (discovered from Y fingerprints, coherence: Z.ZZ)",
    "confidence": 0.85,
    "coherence_score": 0.87,
    "source": "discovered_from_clustering",
    "dominant_embedders": ["Semantic", "Causal", "Code"],
    "member_count": 42
  },
  "discovery_metadata": {
    "fingerprints_analyzed": 150,
    "clusters_found": 5,
    "goals_discovered": 3,
    "algorithm": "KMeans"
  },
  "initialized_services": [...],
  "recommendations": [...],
  "status": "bootstrapped",
  "note": "ARCH-03 COMPLIANT: North Star 'uuid' DISCOVERED autonomously..."
}
```

### 4.3 Error Responses

| Error Code | Condition | Message |
|------------|-----------|---------|
| `-32110` (BOOTSTRAP_ERROR) | No fingerprints stored | "No teleological fingerprints stored. Store memories first..." |
| `-32110` (BOOTSTRAP_ERROR) | Insufficient data (<3 fingerprints) | "Insufficient data for goal discovery: got N fingerprints, need at least 3" |
| `-32110` (BOOTSTRAP_ERROR) | No goals discovered | "No emergent goals discovered. Clustering N fingerprints found no coherent patterns..." |
| `-32603` (INTERNAL_ERROR) | Storage retrieval failure | "Storage error retrieving {id}: {error}" |

---

## 5. Purpose - Why This Tool Exists

### 5.1 The Problem It Solves

From PRD Section 3.1 "Why Manual North Star Creation Is Invalid":

> Manual North Star tools have been REMOVED because they created single 1024D embeddings that cannot be meaningfully compared to 13-embedder teleological arrays.

The `auto_bootstrap_north_star` tool solves the **"apples to oranges" comparison problem**:

| Manual Approach (INVALID) | Autonomous Approach (VALID) |
|---------------------------|----------------------------|
| Single 1024D vector from text-embedding-3-large | 13-array teleological fingerprint |
| Cannot compare to E5 causal (768D), E7 code (1536D), etc. | Apples-to-apples: E1 to E1, E5 to E5, etc. |
| Mathematically meaningless alignment scores | Valid cosine similarity per embedder |

### 5.2 Autonomous Intelligence

The tool embodies the system's **self-organizing capability**:

1. **No human intervention required** for goal discovery
2. **Goals emerge from actual stored data** patterns
3. **Statistical coherence** determines goal validity
4. **Multi-dimensional clustering** respects 13-embedder architecture

---

## 6. PRD Alignment - Global Workspace Theory Goals

### 6.1 GWT Consciousness Framework

This tool supports the PRD's Global Workspace Theory (GWT) implementation:

| PRD Section | How auto_bootstrap_north_star Contributes |
|-------------|-------------------------------------------|
| Section 2.5.4 (SELF_EGO_NODE) | Establishes the primary purpose vector for identity coherence |
| Section 4.5 (Teleological Alignment) | Enables valid A(v, V) = cos(v, V) comparisons |
| Section 18 (Teleological Storage) | Initializes the Goal Hierarchy Index (Layer 2E) |

### 6.2 13-Embedder Architecture Support

From PRD Section 3:

> **Paradigm**: NO FUSION - Store all 13 embeddings (E1-E12 + E13 SPLADE). The array IS the teleological vector.

The bootstrap tool respects this by:
- Operating on full `SemanticFingerprint` arrays (all 13 embedders)
- Using K-means on the complete teleological structure
- Preserving per-embedder information in discovered goals

### 6.3 Adaptive Threshold Integration

From PRD Section 22 (Adaptive Threshold Calibration):

> The system uses no hardcoded thresholds. All thresholds are learned, calibrated, and continuously adapted.

The `confidence_threshold` parameter (default 0.7) feeds into the adaptive calibration system via the initialized `ThresholdLearner` service.

---

## 7. Usage by AI Agents - MCP Integration

### 7.1 When to Call This Tool

| Scenario | Action |
|----------|--------|
| First session with new context graph | Call `auto_bootstrap_north_star` |
| After significant memory accumulation | Re-bootstrap may discover refined purpose |
| System health check fails | Check if bootstrap completed |
| `get_autonomous_status` shows no North Star | Call bootstrap |

### 7.2 Typical Agent Workflow

```
1. Agent starts new session
2. Call get_memetic_status -> Check if North Star exists
3. If no North Star:
   a. Store initial memories using store_memory
   b. Ensure 3+ memories exist
   c. Call auto_bootstrap_north_star
4. If bootstrap succeeds:
   a. Monitor with get_alignment_drift
   b. Use discover_sub_goals for hierarchy
   c. Check get_autonomous_status periodically
```

### 7.3 Integration with Other Tools

| Tool | Relationship |
|------|-------------|
| `get_alignment_drift` | Uses North Star for drift calculations |
| `trigger_drift_correction` | Corrects toward North Star alignment |
| `discover_sub_goals` | Builds hierarchy under North Star |
| `get_autonomous_status` | Reports North Star health |
| `store_memory` | Prerequisite - must store fingerprints first |
| `compute_teleological_vector` | Alternative fingerprint source |

---

## 8. Implementation Details - Key Code Paths

### 8.1 Entry Point

**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/bootstrap.rs`

**Function**: `Handlers::call_auto_bootstrap_north_star`

### 8.2 Critical Code Flow

```
call_auto_bootstrap_north_star
    |
    +-> Parse AutoBootstrapParams
    |
    +-> Check existing North Star (goal_hierarchy.read())
    |   |
    |   +-> If exists: Return "already_bootstrapped"
    |
    +-> Retrieve fingerprints (teleological_store.list_all_johari)
    |   |
    |   +-> FAIL FAST if empty
    |
    +-> Load full SemanticFingerprint for each ID
    |   |
    |   +-> FAIL FAST on storage errors
    |
    +-> Validate minimum cluster size (3)
    |   |
    |   +-> FAIL FAST if insufficient
    |
    +-> Create GoalDiscoveryPipeline
    |   |
    |   +-> Configure: KMeans, Auto clusters, 100 iterations
    |
    +-> Execute pipeline.discover()
    |   |
    |   +-> Catch panics (FAIL FAST design)
    |   +-> FAIL FAST if no goals discovered
    |
    +-> Select highest-confidence goal as North Star
    |
    +-> Create GoalNode with discovery metadata
    |   |
    |   +-> Level: NorthStar
    |   +-> Method: Clustering
    |
    +-> Add to goal_hierarchy
    |
    +-> Return success with services list
```

### 8.3 Key Data Structures

**Discovery Configuration**:
```rust
let discovery_config = DiscoveryConfig {
    sample_size: std::cmp::min(semantic_arrays.len(), 500),
    min_cluster_size: 3,
    min_coherence: params.confidence_threshold,  // default 0.7
    clustering_algorithm: ClusteringAlgorithm::KMeans,
    num_clusters: NumClusters::Auto,
    max_iterations: 100,
    convergence_tolerance: 1e-4,
};
```

**Goal Node Creation**:
```rust
let goal = GoalNode::autonomous_goal(
    north_star_description,
    GoalLevel::NorthStar,
    north_star_candidate.centroid.clone(),
    discovery_metadata,
);
```

### 8.4 FAIL FAST Points

| Check | Error Code | Location |
|-------|------------|----------|
| No fingerprints stored | BOOTSTRAP_ERROR | Line 115-122 |
| Insufficient fingerprints (<3) | BOOTSTRAP_ERROR | Line 146-163 |
| Discovery pipeline panic | BOOTSTRAP_ERROR | Line 179-200 |
| No goals discovered | BOOTSTRAP_ERROR | Line 203-215 |
| Storage retrieval error | INTERNAL_ERROR | Line 136-140 |

---

## 9. Evidence Summary

### Physical Evidence Examined

| File | Purpose | Lines Examined |
|------|---------|----------------|
| `autonomous.rs` (definitions) | Tool schema | 15-41 |
| `bootstrap.rs` (handler) | Implementation | Full file (327 lines) |
| `params.rs` | Parameter structs | 64-73 |
| `error_codes.rs` | Error constants | Full file |
| `pipeline.rs` | Discovery logic | Full file (300 lines) |
| `contextprd.md` | PRD specification | 1-1749 |

### Verification Matrix

| Check | Expected | Actual | Verdict |
|-------|----------|--------|---------|
| ARCH-03 compliance | Autonomous discovery | K-means clustering used | VERIFIED |
| Fail-fast behavior | No fallbacks | Panics/errors on failure | VERIFIED |
| 13-embedder support | Full array preservation | SemanticFingerprint used | VERIFIED |
| Service initialization | 6 services | 6 services listed | VERIFIED |

---

## 10. VERDICT

**INNOCENT** - The `auto_bootstrap_north_star` tool correctly implements the ARCH-03 autonomous goal discovery specification. It:

1. Discovers purpose from stored teleological fingerprints via clustering
2. Maintains apples-to-apples comparison (13-array to 13-array)
3. Fails fast with robust error messages
4. Initializes all required autonomous services
5. Integrates properly with the Global Workspace Theory framework

**Confidence Level**: HIGH

**The case is CLOSED.**

---

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth." - Sherlock Holmes*
