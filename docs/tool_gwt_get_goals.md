# Forensic Investigation Report: gwt_get_goals

## SHERLOCK HOLMES CASE FILE

**Case ID**: GWT-GOALS-001
**Date**: 2026-01-14
**Subject**: gwt_get_goals MCP Tool
**Investigator**: Sherlock Holmes, Forensic Code Detective

---

## EXECUTIVE SUMMARY

The tool `gwt_get_goals` as named **does not exist** in the codebase. After exhaustive forensic investigation, I have determined that the conceptual functionality maps to the canonical tool **`discover_sub_goals`**.

**VERDICT**: The requested tool name is a conceptual alias. The actual implementation is `discover_sub_goals`.

---

## 1. WHAT DOES THIS TOOL DO?

The `discover_sub_goals` tool discovers emergent goals from memory clusters using K-means clustering on teleological fingerprints. It analyzes stored memories to find patterns that could become strategic or tactical goals in the goal hierarchy.

### Core Functionality

1. **Goal Discovery Pipeline**: Uses `GoalDiscoveryPipeline` with K-means clustering
2. **Hierarchical Goal Detection**: Discovers goals at different levels (North Star -> Strategic -> Tactical)
3. **Autonomous Operation**: Works WITHOUT requiring a North Star (ARCH-03 compliant)
4. **Cluster Analysis**: Groups similar memories to identify emergent themes
5. **Confidence Scoring**: Each discovered goal has a confidence score and coherence rating

---

## 2. HOW DOES IT WORK INTERNALLY?

### Implementation Location
- **Handler**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/discovery.rs`
- **Definition**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/autonomous.rs`
- **Name Constant**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs` (line 109)
- **Alias**: `discover_goals` -> `discover_sub_goals` (in aliases.rs)

### Internal Flow

```rust
pub(crate) async fn call_discover_sub_goals(
    &self,
    id: Option<JsonRpcId>,
    arguments: serde_json::Value,
) -> JsonRpcResponse {
    // 1. Parse parameters (min_confidence, max_goals, parent_goal_id, memory_ids, algorithm)
    let params: DiscoverSubGoalsParams = serde_json::from_value(arguments)?;

    // 2. ARCH-03 COMPLIANT: Check for North Star, but don't require it
    let (parent_goal, discovery_mode) = {
        let hierarchy = self.goal_hierarchy.read();
        match &params.parent_goal_id {
            Some(goal_id_str) => /* Use specified parent */,
            None => match hierarchy.north_star() {
                Some(ns) => (Some(ns), "under_north_star"),
                None => (None, "autonomous") // ARCH-03: Works without North Star
            }
        }
    };

    // 3. Collect memories to analyze (from memory_ids or all recent)
    let arrays: Vec<SemanticFingerprint> = /* Load fingerprints */;

    // 4. FAIL FAST: Require minimum 3 arrays for clustering
    if arrays.len() < 3 {
        return error("Insufficient data for clustering");
    }

    // 5. Create GoalDiscoveryPipeline (TASK-LOGIC-009)
    let comparator = TeleologicalComparator::new();
    let pipeline = GoalDiscoveryPipeline::new(comparator);

    // 6. Configure and execute discovery
    let config = DiscoveryConfig {
        clustering_algorithm: ClusteringAlgorithm::KMeans, // or HDBSCAN, Spectral
        num_clusters: NumClusters::Auto,
        min_coherence: params.min_confidence,
        ...
    };
    let result = pipeline.discover(&arrays, &config);

    // 7. Build response with discovered goals, hierarchy, cluster analysis
    self.tool_result_with_pulse(id, json!({
        "discovered_goals": [...],
        "hierarchy_relationships": [...],
        "cluster_analysis": {...},
        "discovery_mode": discovery_mode,
        "arch03_compliant": true
    }))
}
```

### Discovery Pipeline Architecture

```
Memory Fingerprints (13-embedder arrays)
    -> TeleologicalComparator (similarity computation)
        -> GoalDiscoveryPipeline (K-means/HDBSCAN/Spectral)
            -> Cluster Formation
                -> Goal Extraction (centroid analysis)
                    -> Hierarchy Construction
                        -> discover_sub_goals (THIS TOOL)
```

### Clustering Algorithms Supported

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `kmeans` (default) | K-means clustering | Well-separated clusters |
| `hdbscan` | Hierarchical DBSCAN (min_samples=5) | Noise-tolerant discovery |
| `spectral` | Spectral clustering (n_neighbors=10) | Complex manifolds |

---

## 3. WHY DOES IT EXIST?

### Goal Hierarchy Context (PRD Section 3.1)

The PRD describes an autonomous goal system:

> "The autonomous services (NORTH-008 to NORTH-020) work entirely within teleological space, ensuring all comparisons are mathematically valid and semantically meaningful."

The tool exists to:

1. **Emergent Goal Discovery**: Find patterns in stored memories that represent implicit goals
2. **Goal Hierarchy Evolution**: Build North Star -> Strategic -> Tactical goal structure
3. **Autonomous Operation**: Discover purpose WITHOUT manual North Star configuration
4. **Sub-goal Decomposition**: Break down high-level goals into actionable sub-goals

### ARCH-03 Compliance

The tool is explicitly designed to work **without a North Star**:

```rust
// ARCH-03: Try North Star first, but work autonomously if none exists
match hierarchy.north_star() {
    Some(ns) => (Some(ns.clone()), "under_north_star"),
    None => {
        info!("No North Star - discovering goals autonomously (ARCH-03)");
        (None, "autonomous")
    }
}
```

---

## 4. PURPOSE IN REACHING PRD END GOAL

### The Problem Being Solved

Per PRD Section 0.1:

> "AI agents fail because: no persistent memory, poor retrieval, no learning loop, context bloat."

### How Goal Discovery Solves This

1. **Purpose Alignment**: Goals provide direction for memory storage decisions
2. **Retrieval Guidance**: Goals help prioritize which memories to retrieve
3. **Learning Focus**: Goals focus the learning system on what matters
4. **Context Compression**: Goals enable intelligent context distillation

### Integration with GWT

The Global Workspace Theory requires understanding what the system is "trying to do":

```
GWT Consciousness Equation:
C(t) = I(t) x R(t) x D(t)

Where D(t) = Differentiation = Purpose Vector Entropy
```

Goals directly impact the **Differentiation (D)** component of consciousness by providing a clear purpose vector.

### Goal Hierarchy Structure (PRD Section 2.5.4)

```
North Star (Level 0): Overarching purpose
    -> Strategic Goals (Level 1): discover_sub_goals discovers these
        -> Tactical Goals (Level 2): discover_sub_goals discovers these
            -> Memory Alignment: memories align to nearest goal
```

---

## 5. INPUTS (Parameters)

### JSON Schema

```json
{
  "type": "object",
  "properties": {
    "min_confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "default": 0.6,
      "description": "Minimum confidence for a discovered sub-goal"
    },
    "max_goals": {
      "type": "integer",
      "minimum": 1,
      "maximum": 20,
      "default": 5,
      "description": "Maximum number of sub-goals to discover"
    },
    "parent_goal_id": {
      "type": "string",
      "description": "Parent goal ID to discover sub-goals for (defaults to North Star)"
    },
    "memory_ids": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Specific memory UUIDs to analyze (optional)"
    },
    "algorithm": {
      "type": "string",
      "enum": ["kmeans", "hdbscan", "spectral"],
      "default": "kmeans",
      "description": "Clustering algorithm to use"
    }
  },
  "required": []
}
```

### Parameter Details

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_confidence` | number [0,1] | No | 0.6 | Minimum coherence score for goal |
| `max_goals` | integer [1,20] | No | 5 | Maximum goals to discover |
| `parent_goal_id` | string (UUID) | No | North Star or null | Parent goal for sub-goal discovery |
| `memory_ids` | array of UUIDs | No | All recent | Specific memories to analyze |
| `algorithm` | string enum | No | "kmeans" | Clustering algorithm |

---

## 6. OUTPUTS

### Success Response

```json
{
  "content": [{
    "type": "text",
    "text": {
      "discovered_goals": [
        {
          "goal_id": "550e8400-e29b-41d4-a716-446655440001",
          "description": "Authentication and Security Management",
          "level": "Strategic",
          "confidence": 0.85,
          "member_count": 12,
          "coherence_score": 0.78,
          "dominant_embedders": ["E1_semantic", "E5_causal", "E7_code"]
        }
      ],
      "hierarchy_relationships": [
        {
          "parent_id": "north-star-uuid",
          "child_id": "550e8400-e29b-41d4-a716-446655440001",
          "similarity": 0.82
        }
      ],
      "cluster_analysis": {
        "parent_goal_id": "north-star-uuid",
        "parent_goal_description": "System Purpose",
        "clusters_found": 3,
        "total_arrays_analyzed": 50,
        "discovery_mode": "under_north_star",
        "discovery_parameters": {
          "min_confidence": 0.6,
          "max_goals": 5,
          "algorithm": "KMeans"
        }
      },
      "discovery_metadata": {
        "total_arrays_analyzed": 50,
        "clusters_found": 3,
        "algorithm_used": "KMeans",
        "num_clusters_setting": "Auto",
        "discovery_mode": "under_north_star"
      },
      "discovery_count": 3,
      "arch03_compliant": true,
      "note": "Goals discovered as sub-goals under parent goal"
    }
  }],
  "isError": false,
  "_pulse": {...}
}
```

### Autonomous Mode Response (No North Star)

```json
{
  "discovery_mode": "autonomous",
  "arch03_compliant": true,
  "note": "ARCH-03 COMPLIANT: Goals discovered autonomously from clustering (no North Star required)"
}
```

### Output Field Details

| Field | Type | Description |
|-------|------|-------------|
| `discovered_goals` | array | List of discovered goal objects |
| `discovered_goals[].goal_id` | string (UUID) | Unique goal identifier |
| `discovered_goals[].description` | string | Human-readable goal description |
| `discovered_goals[].level` | string | "NorthStar", "Strategic", or "Tactical" |
| `discovered_goals[].confidence` | number | Discovery confidence [0,1] |
| `discovered_goals[].member_count` | number | Memories in this cluster |
| `discovered_goals[].coherence_score` | number | Cluster coherence [0,1] |
| `discovered_goals[].dominant_embedders` | array | Which embedding spaces dominate |
| `hierarchy_relationships` | array | Parent-child goal relationships |
| `cluster_analysis` | object | Detailed clustering information |
| `discovery_mode` | string | "autonomous", "under_north_star", or "under_parent" |
| `arch03_compliant` | boolean | Always true (ARCH-03 compliance) |

### Error Response (Insufficient Data)

```json
{
  "error": {
    "code": -32602,
    "message": "Insufficient data for clustering: got 2 arrays, need at least 3 - FAIL FAST"
  }
}
```

---

## 7. CORE FUNCTIONALITY SUMMARY

### What It Does
Discovers emergent goals from memory clusters using machine learning clustering algorithms on teleological fingerprints.

### Why It Exists
The system needs a way to autonomously evolve its goal hierarchy based on actual content, not just manual configuration. Goals emerge from data patterns.

### Key Features
- **ARCH-03 Compliant**: Works without North Star
- **Multi-Algorithm**: K-means, HDBSCAN, Spectral clustering
- **Hierarchical**: Discovers goals at Strategic and Tactical levels
- **Confidence-Based**: Each goal has a coherence score
- **Embedder-Aware**: Identifies which embedding spaces dominate each goal

---

## 8. HOW AN AI AGENT USES THIS TOOL

### Typical Agent Workflow

```
1. Agent stores memories via store_memory over time
2. Agent calls discover_sub_goals to find emergent patterns
3. System returns discovered goals with coherence scores
4. Agent can then:
   - Use goals to guide future memory storage (rationale linking)
   - Check alignment of new content against discovered goals
   - Build understanding of system's implicit purpose
```

### Integration with Goal Hierarchy

```
auto_bootstrap_north_star (creates North Star from fingerprints)
    |
    v
discover_sub_goals (THIS TOOL - creates sub-goals)
    |
    v
get_autonomous_status (monitors goal hierarchy health)
    |
    v
get_alignment_drift (measures drift from goals)
```

### Agent Decision Tree

```
When to call discover_sub_goals:
  |
  +-- After storing 20+ memories -> Discover initial structure
  |
  +-- get_autonomous_status shows "not_configured" -> Bootstrap goals
  |
  +-- High entropy in specific domain -> Discover domain-specific goals
  |
  +-- User asks "what have you learned?" -> Discover emergent patterns
```

### Usage with GWT

The discovered goals feed into the GWT consciousness equation via the **Purpose Vector**:

```
Goals discovered by discover_sub_goals
    -> Stored in goal_hierarchy
        -> Feed into SelfEgoNode.purpose_vector
            -> Used in C(t) = I(t) x R(t) x D(t)
                -> D(t) = Purpose Vector Entropy
```

---

## EVIDENCE CHAIN OF CUSTODY

| Timestamp | Action | File | Verified |
|-----------|--------|------|----------|
| 2026-01-14 | Handler implementation examined | autonomous/discovery.rs:1-356 | HOLMES |
| 2026-01-14 | Tool definition examined | definitions/autonomous.rs:204-234 | HOLMES |
| 2026-01-14 | Name constant verified | names.rs:109 | HOLMES |
| 2026-01-14 | Alias verified | aliases.rs:33 (discover_goals -> discover_sub_goals) | HOLMES |
| 2026-01-14 | PRD ARCH-03 compliance verified | discovery.rs:97-108 | HOLMES |
| 2026-01-14 | Dispatch routing confirmed | dispatch.rs:153 | HOLMES |

---

## VERDICT

**INNOCENT** - The tool is correctly implemented per PRD specification with full ARCH-03 compliance.

**CONFIDENCE**: HIGH

**NOTE**: The name `gwt_get_goals` is a conceptual alias. The canonical tool name is `discover_sub_goals`. There is also a legacy alias `discover_goals`.

---

*"Data! Data! Data! I can't make bricks without clay." - Sherlock Holmes*
