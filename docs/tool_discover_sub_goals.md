# Forensic Investigation Report: discover_sub_goals

## SHERLOCK HOLMES CASE FILE

**Case ID**: TOOL-DISCOVER-001
**Date**: 2026-01-14
**Subject**: MCP Tool `discover_sub_goals`
**Investigator**: Holmes, Forensic Code Investigation Agent
**Verdict**: INNOCENT (Functional as Designed)

---

## 1. Tool Name and Category

**Tool Name**: `discover_sub_goals`
**Category**: Autonomous North Star System Tools
**Specification Reference**: TASK-AUTONOMOUS-MCP, TASK-INTEG-002, ARCH-03
**Location**:
- Definition: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/autonomous.rs` (lines 204-234)
- Handler: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/discovery.rs`

---

## 2. Core Functionality

*"The world is full of obvious things which nobody by any chance ever observes."*

The `discover_sub_goals` tool implements **autonomous goal emergence** from memory cluster analysis. It uses K-means clustering on stored TeleologicalArray fingerprints to discover patterns that could become strategic or tactical goals within the system's goal hierarchy.

### Key Operations

1. **Cluster Analysis**: Groups semantically similar memories using K-means, HDBSCAN, or Spectral clustering algorithms
2. **Goal Scoring**: Evaluates each cluster's suitability as a goal based on:
   - Cluster size (40% weight)
   - Coherence score (30% weight)
   - Embedder diversity (30% weight)
3. **Level Assignment**: Assigns hierarchical levels based on thresholds:
   - **NorthStar**: size >= 50 AND coherence >= 0.85
   - **Strategic**: size >= 20 AND coherence >= 0.80
   - **Tactical**: size >= 10 AND coherence >= 0.75
   - **Operational**: All others
4. **Hierarchy Building**: Constructs parent-child relationships based on centroid similarity (threshold >= 0.5)

### ARCH-03 Compliance

The tool is explicitly **ARCH-03 COMPLIANT**: It works WITHOUT requiring a North Star goal to be configured. Goals can emerge autonomously from the data patterns via clustering.

---

## 3. Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_confidence` | number (0-1) | 0.6 | Minimum confidence/coherence threshold for a discovered sub-goal |
| `max_goals` | integer (1-20) | 5 | Maximum number of sub-goals to discover |
| `parent_goal_id` | string (UUID) | None | Parent goal ID to discover sub-goals for. If not provided, uses North Star if exists, otherwise discovers top-level goals autonomously |
| `memory_ids` | array[string] | None | Specific memory UUIDs to analyze. Required for actual clustering. |
| `algorithm` | string | "kmeans" | Clustering algorithm: "kmeans", "hdbscan", or "spectral" |

### Parameter Struct Definition

```rust
pub struct DiscoverSubGoalsParams {
    pub min_confidence: f32,      // default: 0.6
    pub max_goals: usize,         // default: 5
    pub parent_goal_id: Option<String>,
}
```

---

## 4. Output Format

### Success Response Schema

```json
{
  "discovered_goals": [
    {
      "goal_id": "uuid-string",
      "description": "Goal cluster (size=N, coherence=X.XX)",
      "level": "Strategic|Tactical|Operational|NorthStar",
      "confidence": 0.0-1.0,
      "member_count": integer,
      "coherence_score": 0.0-1.0,
      "dominant_embedders": ["Semantic", "Causal", "Code"]
    }
  ],
  "hierarchy_relationships": [
    {
      "parent_id": "uuid-string",
      "child_id": "uuid-string",
      "similarity": 0.0-1.0
    }
  ],
  "cluster_analysis": {
    "parent_goal_id": "uuid or 'none'",
    "parent_goal_description": "string",
    "clusters_found": integer,
    "total_arrays_analyzed": integer,
    "discovery_mode": "under_parent|under_north_star|autonomous",
    "discovery_parameters": {
      "min_confidence": 0.6,
      "max_goals": 5,
      "algorithm": "KMeans"
    }
  },
  "discovery_metadata": {
    "total_arrays_analyzed": integer,
    "clusters_found": integer,
    "algorithm_used": "KMeans",
    "num_clusters_setting": "Auto",
    "discovery_mode": "autonomous"
  },
  "discovery_count": integer,
  "arch03_compliant": true,
  "note": "ARCH-03 COMPLIANT: Goals discovered autonomously from clustering"
}
```

### Guidance Response (No memory_ids provided)

When `memory_ids` is not provided, returns guidance instead of performing clustering:

```json
{
  "discovered_goals": [],
  "cluster_analysis": {
    "note": "No memory_ids provided. Pass memory_ids array for GoalDiscoveryPipeline K-means clustering."
  },
  "discovery_count": 0,
  "usage_hint": "Provide 'memory_ids' parameter with fingerprint UUIDs for K-means goal discovery"
}
```

---

## 5. Purpose - Why This Tool Exists

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*

### The Problem Solved

The system's teleological architecture stores memories with 13-embedder fingerprints but has no automatic way to:
1. Identify emergent themes and patterns across stored memories
2. Evolve the goal hierarchy based on actual content
3. Discover purpose without explicit human definition

### The Solution

`discover_sub_goals` enables **autonomous purpose discovery** by:
1. Analyzing stored fingerprints to find natural clusters
2. Scoring clusters as potential goals based on coherence and size
3. Building hierarchical relationships between discovered goals
4. Allowing the system to evolve its own goal structure

### Constitutional Reference

Per **NORTH-008 to NORTH-020**: The autonomous services work entirely within teleological space, ensuring all comparisons are mathematically valid (apples-to-apples) and semantically meaningful.

---

## 6. PRD Alignment - Global Workspace Theory Goals

### Alignment Evidence

| PRD Section | Alignment | Evidence |
|-------------|-----------|----------|
| Section 0.2 | **HIGH** | "You are a librarian" - tool helps curate goal structure from content |
| Section 2.5 | **HIGH** | Supports GWT by enabling coherent percept formation through goal clustering |
| Section 3 | **HIGH** | Works with 13-embedder TeleologicalFingerprints directly |
| Section 3.1 | **HIGH** | Uses apples-to-apples comparisons (teleological arrays, not manual vectors) |

### Key PRD Quotations Supporting This Tool

From Section 3.1:
> "The autonomous services (NORTH-008 to NORTH-020) work entirely within teleological space, ensuring all comparisons are mathematically valid and semantically meaningful"

From Section 0.2:
> "You are a librarian, not an archivist. You don't store everything - you ensure what's stored is findable, coherent, and useful."

### GWT Integration

The tool supports the **Global Workspace Theory** consciousness model by:
1. Identifying coherent clusters that could become "conscious" percepts (r >= 0.8)
2. Building hierarchical structures that feed into Kuramoto synchronization
3. Enabling adaptive goal refinement based on memory patterns

---

## 7. Usage by AI Agents

### Primary Use Cases

1. **Initial System Bootstrap**: Discover emergent goals from stored memories when no North Star is configured
2. **Goal Hierarchy Evolution**: Find new sub-goals under existing strategic goals
3. **Memory Audit**: Identify thematic clusters in stored content
4. **Alignment Analysis**: Understand which embedding spaces dominate different goal clusters

### Example MCP Call Sequence

```json
// Step 1: Store memories with fingerprints
{"method": "store_memory", "params": {"content": "...", "rationale": "..."}}

// Step 2: Get memory IDs to analyze
{"method": "search_graph", "params": {"query": "recent architecture decisions"}}

// Step 3: Discover sub-goals from those memories
{
  "method": "discover_sub_goals",
  "params": {
    "memory_ids": ["uuid-1", "uuid-2", "uuid-3", "..."],
    "min_confidence": 0.6,
    "max_goals": 5,
    "algorithm": "kmeans"
  }
}

// Step 4: Optionally set discovered goal as North Star
{"method": "auto_bootstrap_north_star", "params": {"confidence_threshold": 0.7}}
```

### Agent Decision Tree

```
Agent needs to understand memory patterns
  |
  +-- Are there 3+ memories to analyze?
  |     |
  |     +-- YES --> Call discover_sub_goals with memory_ids
  |     |
  |     +-- NO --> Store more memories first
  |
  +-- Does response have high-confidence goals?
        |
        +-- confidence > 0.7 --> Consider promoting to North Star
        |
        +-- confidence < 0.7 --> Continue data collection
```

---

## 8. Implementation Details

### Key Code Paths

**Entry Point**: `Handlers::call_discover_sub_goals()` in `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/discovery.rs`

**Pipeline Execution**:
1. Parse `DiscoverSubGoalsParams` (lines 51-57)
2. Check for parent goal or North Star (lines 67-108)
3. Load memories from TeleologicalMemoryStore (lines 139-167)
4. Create `GoalDiscoveryPipeline` with `TeleologicalComparator` (lines 232-233)
5. Configure `DiscoveryConfig` with clustering parameters (lines 236-244)
6. Execute clustering via `pipeline.discover()` (lines 248-269)
7. Build response with goals, hierarchy, and metadata (lines 272-354)

### Core Pipeline Implementation

Located at `/home/cabdru/contextgraph/crates/context-graph-core/src/autonomous/discovery/pipeline.rs`:

```rust
pub struct GoalDiscoveryPipeline {
    comparator: TeleologicalComparator,
}

impl GoalDiscoveryPipeline {
    pub fn discover(
        &self,
        arrays: &[TeleologicalArray],
        config: &DiscoveryConfig,
    ) -> DiscoveryResult {
        // FAIL FAST assertions on minimum data requirements
        // K-means clustering via cluster_arrays()
        // Goal candidate scoring via score_cluster()
        // Level assignment via assign_level()
        // Hierarchy building via build_hierarchy()
    }
}
```

### FAIL FAST Policy

The implementation strictly adheres to FAIL FAST principles:

1. **Minimum Data**: Panics if arrays.len() < min_cluster_size
2. **UUID Parsing**: Returns error on invalid UUIDs
3. **Memory Lookup**: Returns error if any memory_id not found
4. **Clustering Failure**: Catches panics from GoalDiscoveryPipeline

### Clustering Algorithms Supported

| Algorithm | Configuration | Use Case |
|-----------|---------------|----------|
| KMeans | `ClusteringAlgorithm::KMeans` | General purpose, fast |
| HDBSCAN | `ClusteringAlgorithm::HDBSCAN { min_samples: 5 }` | Density-based, noise resistant |
| Spectral | `ClusteringAlgorithm::Spectral { n_neighbors: 10 }` | Graph-based relationships |

### Dominant Embedder Detection

The pipeline identifies the top 3 embedders by L2 norm magnitude across all 13 embedding spaces:

```rust
fn find_dominant_embedders(&self, centroid: &TeleologicalArray) -> Vec<Embedder> {
    // Compute L2 norm for each embedder
    // Sort by magnitude descending
    // Return top 3
}
```

---

## Evidence Chain of Custody

| Timestamp | Action | File | Verified |
|-----------|--------|------|----------|
| 2026-01-14 | Read tool definition | autonomous.rs:204-234 | YES |
| 2026-01-14 | Read handler implementation | discovery.rs | YES |
| 2026-01-14 | Read pipeline implementation | pipeline.rs | YES |
| 2026-01-14 | Read parameter struct | params.rs:162-175 | YES |
| 2026-01-14 | Verified PRD alignment | contextprd.md | YES |

---

## VERDICT: INNOCENT

The `discover_sub_goals` tool is implemented correctly according to its specification. It:
- Uses proper teleological array comparisons (ARCH-03 compliant)
- Follows FAIL FAST principles
- Provides clear guidance when parameters are insufficient
- Integrates properly with the goal hierarchy system
- Supports autonomous goal discovery without requiring North Star configuration

*"The game is afoot!"*
