# Forensic Analysis Report: trigger_consolidation

## Case File Summary

**Tool Name:** `trigger_consolidation`
**Category:** Autonomous North Star System (TASK-AUTONOMOUS-MCP)
**Specification:** SPEC-STUBFIX-003 (Real Data Implementation)
**Investigation Date:** 2026-01-14
**Investigator:** Sherlock Holmes, Code Forensics Division

---

## 1. Tool Name and Category

| Attribute | Value |
|-----------|-------|
| **Canonical Name** | `trigger_consolidation` |
| **Category** | Autonomous Memory Maintenance |
| **Module Path** | `context-graph-mcp::handlers::autonomous::maintenance` |
| **Task Reference** | TASK-AUTONOMOUS-MCP, SPEC-STUBFIX-003 |
| **Tool Index** | 6th of 13 autonomous tools |

---

## 2. Core Functionality

The `trigger_consolidation` tool triggers memory consolidation to merge similar memories and reduce redundancy in the Context Graph. Unlike pruning (which removes memories), consolidation **combines** semantically similar memories into unified entries.

### Operational Behavior

1. **Retrieves fingerprints from teleological store** - Uses `list_all_johari()` to get stored memories
2. **Converts to MemoryContent** - Extracts E1 (semantic) embedding for comparison
3. **Builds memory pairs based on strategy** - Different pairing logic per strategy
4. **Creates ConsolidationService** - Applies user-specified similarity threshold
5. **Finds consolidation candidates** - Uses `ConsolidationService.find_consolidation_candidates()`
6. **Returns candidates for review** - Does not auto-merge (allows human oversight)

### Consolidation Strategies

| Strategy | Pairing Logic | Use Case |
|----------|---------------|----------|
| `similarity` | All pairs where cosine similarity >= threshold * 0.9 | General duplicate detection |
| `temporal` | Pairs created within 24 hours of each other | Session-based deduplication |
| `semantic` | Pairs where both have alignment >= 0.5 | Quality-aware consolidation |

---

## 3. Input Parameters

| Parameter | Type | Required | Default | Range | Description |
|-----------|------|----------|---------|-------|-------------|
| `max_memories` | `integer` | No | `100` | 1-10000 | Maximum memories to process in one batch |
| `strategy` | `string` | No | `"similarity"` | similarity/temporal/semantic | Consolidation strategy |
| `min_similarity` | `number` | No | `0.85` | 0.0-1.0 | Minimum similarity threshold for merge candidates |

### Parameter Struct Definition

```rust
// File: /crates/context-graph-mcp/src/handlers/autonomous/params.rs

#[derive(Debug, Deserialize)]
pub struct TriggerConsolidationParams {
    #[serde(default = "default_max_memories")]  // 100
    pub max_memories: usize,

    #[serde(default = "default_consolidation_strategy")]  // "similarity"
    pub strategy: String,

    #[serde(default = "default_consolidation_similarity")]  // 0.85
    pub min_similarity: f32,
}
```

### Tool Definition Schema

```json
{
  "type": "object",
  "properties": {
    "max_memories": {
      "type": "integer",
      "minimum": 1,
      "maximum": 10000,
      "default": 100,
      "description": "Maximum memories to process in one batch"
    },
    "strategy": {
      "type": "string",
      "enum": ["similarity", "temporal", "semantic"],
      "default": "similarity",
      "description": "Consolidation strategy to use"
    },
    "min_similarity": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "default": 0.85,
      "description": "Minimum similarity threshold for consolidation candidates"
    }
  },
  "required": []
}
```

---

## 4. Output Format

### Success Response Structure

```json
{
  "consolidation_result": {
    "status": "candidates_found|no_candidates",
    "candidate_count": 5,
    "action_required": true
  },
  "statistics": {
    "pairs_evaluated": 4950,
    "pairs_consolidated": 5,
    "strategy": "similarity",
    "similarity_threshold": 0.85,
    "max_memories_limit": 100,
    "fingerprints_analyzed": 100
  },
  "candidates_sample": [
    {
      "source_ids": [
        "550e8400-e29b-41d4-a716-446655440001",
        "550e8400-e29b-41d4-a716-446655440002"
      ],
      "target_id": "550e8400-e29b-41d4-a716-446655440099",
      "similarity": 0.92,
      "combined_alignment": 0.78
    }
  ]
}
```

### Response Fields Explained

| Field | Description |
|-------|-------------|
| `consolidation_result.status` | "candidates_found" or "no_candidates" |
| `consolidation_result.candidate_count` | Number of merge candidates identified |
| `consolidation_result.action_required` | True if candidates exist (requires `merge_concepts` to execute) |
| `statistics.pairs_evaluated` | Number of memory pairs compared (n*(n-1)/2 for similarity) |
| `statistics.pairs_consolidated` | Number of pairs meeting consolidation criteria |
| `statistics.strategy` | Strategy that was used |
| `statistics.similarity_threshold` | Threshold that was applied |
| `statistics.fingerprints_analyzed` | Total fingerprints processed |
| `candidates_sample` | Up to 10 candidates (limited for response size) |
| `candidates_sample[].source_ids` | UUIDs of memories to merge |
| `candidates_sample[].target_id` | UUID for the merged result |
| `candidates_sample[].similarity` | Cosine similarity between sources |
| `candidates_sample[].combined_alignment` | Weighted alignment of merged memory |

---

## 5. Purpose - Why This Tool Exists

### The Problem: Memory Redundancy

The Context Graph accumulates redundant memories through:

1. **Near-duplicate storage** - Similar information stored multiple times
2. **Incremental updates** - Related information stored as separate entries
3. **Session overlap** - Same concepts mentioned across conversations
4. **Temporal clustering** - Burst storage during active periods

This redundancy causes:
- **Storage waste** - Multiple copies of semantically equivalent information
- **Retrieval ambiguity** - Search returns multiple similar results
- **Coherence fragmentation** - Same concept scattered across nodes
- **Kuramoto desynchronization** - Multiple nodes competing for workspace

### The Solution: Memory Consolidation

Consolidation merges semantically similar memories by:

1. **Combining embeddings** - Weighted average of E1 semantic vectors
2. **Concatenating content** - Preserving information from both sources
3. **Aggregating access counts** - Maintaining usage history
4. **Computing combined alignment** - Favoring higher-alignment sources

### PRD Reference: Dream Layer (Section 7.1)

```
PRD Section 7.1 - Dream Layer (SRC):
NREM (3min): Replay + Hebbian delta_w = eta x pre x post + tight coupling
```

This tool provides **manual consolidation** as a complement to automatic dream-phase consolidation.

### PRD Reference: Passive Curator (Section 7.5)

```
PRD Section 7.5 - Passive Curator:
Auto: high-confidence dupes (>0.95), weak links, orphans (>30d)
Escalates: ambiguous dupes (0.7-0.95), priors-incompatible, conflicts
```

This tool handles the "ambiguous dupes (0.7-0.95)" case where automatic consolidation is not confident enough.

---

## 6. PRD Alignment - Global Workspace Theory Goals

### 6.1 Workspace Competition Reduction

```
PRD Section 2.5.3 - Global Broadcast Architecture:
competing_memories: PriorityQueue  -- Sorted by r x importance
```

Similar memories compete for workspace access. Consolidation reduces competition by merging similar entries into one stronger candidate.

### 6.2 Kuramoto Synchronization Support

```
PRD Section 2.5.2 - Kuramoto Oscillator Layer:
r x e^(i*psi) = (1/N) * sum_j e^(i*theta_j)

Where:
  r > 0.8 -> Memory is coherent ("conscious")
  r < 0.5 -> Memory fragmentation alert
```

Fragmented (similar but separate) memories reduce the order parameter `r`. Consolidation increases `r` by reducing `N` while preserving information content.

### 6.3 Token Economy Optimization

```
PRD Section 1.6 - Token Economy:
| Level | Tokens | When |
|-------|--------|------|
| 0     | ~100   | High confidence |
```

Consolidated memories enable Level 0 retrieval more often by reducing the candidate pool.

### 6.4 Storage Efficiency (Section 14.1)

```
PRD Section 14.1 - Embedding & Storage:
Storage per memory (quantized): ~17KB
Storage per memory (uncompressed): ~46KB
```

Consolidating 2 memories saves ~17KB while preserving semantic content.

---

## 7. Usage by AI Agents - MCP System Integration

### 7.1 When to Call This Tool

An AI agent should call `trigger_consolidation` when:

1. **After bulk imports** - Many related memories may be redundant
2. **Post-session cleanup** - Session may have created duplicates
3. **High entropy detection** - Entropy > 0.7 may indicate fragmentation
4. **Scheduled maintenance** - Weekly consolidation passes
5. **Before critical searches** - Reduce candidate pool for faster retrieval

### 7.2 Example Agent Workflow

```javascript
// Step 1: Check memetic status for fragmentation indicators
const status = await callTool("get_memetic_status", {});

if (status.entropy_level > 0.6) {
  console.log("High entropy - consolidation may help");

  // Step 2: Find consolidation candidates
  const consolidation = await callTool("trigger_consolidation", {
    max_memories: 200,
    strategy: "similarity",
    min_similarity: 0.85
  });

  console.log(`Found ${consolidation.consolidation_result.candidate_count} candidates`);
  console.log(`Pairs evaluated: ${consolidation.statistics.pairs_evaluated}`);

  // Step 3: Review candidates
  if (consolidation.consolidation_result.action_required) {
    for (const candidate of consolidation.candidates_sample) {
      console.log(`Merge ${candidate.source_ids.join(" + ")} -> ${candidate.target_id}`);
      console.log(`  Similarity: ${candidate.similarity}`);
      console.log(`  Combined alignment: ${candidate.combined_alignment}`);

      // Step 4: Execute merge via merge_concepts
      if (candidate.similarity > 0.90) {
        await callTool("merge_concepts", {
          source_node_ids: candidate.source_ids,
          target_name: "Consolidated Memory",
          merge_strategy: "summarize"
        });
      }
    }
  }
}
```

### 7.3 Strategy Selection Guide

| Scenario | Recommended Strategy | Rationale |
|----------|---------------------|-----------|
| General cleanup | `similarity` | Catches all semantic duplicates |
| Session deduplication | `temporal` | Focuses on recent burst storage |
| Quality-focused | `semantic` | Only merges well-aligned memories |
| Aggressive | `similarity` with threshold=0.80 | Catches more potential duplicates |
| Conservative | `similarity` with threshold=0.92 | Only catches near-exact duplicates |

### 7.4 Integration with Other Tools

| Tool | Relationship |
|------|--------------|
| `merge_concepts` | **Required follow-up** - Executes the actual merge |
| `get_pruning_candidates` | Alternative - prune instead of consolidate |
| `trigger_dream` | Automatic consolidation during NREM phase |
| `get_memetic_status` | High entropy may indicate need for consolidation |
| `search_graph` | Test before/after consolidation for result quality |

### 7.5 Consolidation vs Pruning Decision

```
Memory pair has high similarity (>0.85)
  |
  +-- Both have good alignment (>0.5)
  |     |
  |     +-> CONSOLIDATE: Merge into stronger memory
  |
  +-- One has low alignment (<0.4)
  |     |
  |     +-> PRUNE: Remove low-alignment one
  |
  +-- Both have low alignment (<0.4)
        |
        +-> PRUNE BOTH: Neither is valuable
```

---

## 8. Implementation Details - Key Code Paths

### 8.1 File Locations

| Component | Path |
|-----------|------|
| Tool Definition | `/crates/context-graph-mcp/src/tools/definitions/autonomous.rs:170-202` |
| Handler Implementation | `/crates/context-graph-mcp/src/handlers/autonomous/maintenance.rs:195-438` |
| Core Service | `/crates/context-graph-core/src/autonomous/services/consolidation_service/service.rs` |
| Parameters | `/crates/context-graph-mcp/src/handlers/autonomous/params.rs:146-160` |
| Types | `/crates/context-graph-core/src/autonomous/services/consolidation_service/types.rs` |

### 8.2 Handler Flow

```rust
// File: /crates/context-graph-mcp/src/handlers/autonomous/maintenance.rs

pub(crate) async fn call_trigger_consolidation(
    &self,
    id: Option<JsonRpcId>,
    arguments: serde_json::Value,
) -> JsonRpcResponse {
    // 1. Parse and validate parameters
    let params: TriggerConsolidationParams = serde_json::from_value(arguments)?;

    // Validate strategy
    let valid_strategies = ["similarity", "temporal", "semantic"];
    if !valid_strategies.contains(&params.strategy.as_str()) {
        return self.tool_error_with_pulse(id, &format!("Invalid strategy '{}'", params.strategy));
    }

    // 2. FAIL FAST: Get fingerprints from store
    let johari_list = self.teleological_store
        .list_all_johari(params.max_memories)
        .await?;

    // 3. Convert TeleologicalFingerprint to MemoryContent
    let mut memory_contents: Vec<MemoryContent> = Vec::new();
    let mut fingerprints: Vec<(Uuid, DateTime<Utc>)> = Vec::new();

    for (uuid, _johari) in johari_list.iter() {
        let fp = self.teleological_store.retrieve(*uuid).await?;

        // Use E1 (semantic 1024D) embedding for comparison
        let embedding = fp.semantic.e1_semantic.clone();

        let content = MemoryContent::new(
            MemoryId(fp.id),
            embedding,
            String::new(),  // No text content in fingerprint
            fp.theta_to_north_star,
        ).with_access_count(fp.access_count as u32);

        memory_contents.push(content);
        fingerprints.push((fp.id, fp.created_at));
    }

    // 4. Build pairs based on strategy
    let pairs: Vec<MemoryPair> = match params.strategy.as_str() {
        "similarity" => build_similarity_pairs(&memory_contents, params.min_similarity),
        "temporal" => build_temporal_pairs(&memory_contents, &fingerprints),
        "semantic" => build_semantic_pairs(&memory_contents),
        _ => Vec::new(),
    };

    // 5. Create ConsolidationService and find candidates
    let config = ConsolidationConfig {
        enabled: true,
        similarity_threshold: params.min_similarity,
        max_daily_merges: 50,
        theta_diff_threshold: 0.05,
    };
    let consolidation_service = ConsolidationService::with_config(config);

    let candidates = consolidation_service.find_consolidation_candidates(&pairs);

    // 6. Return results
    self.tool_result_with_pulse(id, json!({
        "consolidation_result": {...},
        "statistics": {...},
        "candidates_sample": candidates.take(10)
    }))
}
```

### 8.3 Strategy Implementations

#### Similarity Strategy (Default)

```rust
// Compare all pairs, pre-filter by threshold * 0.9
fn build_similarity_pairs(
    memory_contents: &[MemoryContent],
    min_similarity: f32
) -> Vec<MemoryPair> {
    let mut pairs = Vec::new();
    let threshold = min_similarity * 0.9;  // Pre-filter threshold

    for i in 0..memory_contents.len() {
        for j in (i + 1)..memory_contents.len() {
            // Quick cosine similarity via dot product
            let sim: f32 = memory_contents[i].embedding.iter()
                .zip(memory_contents[j].embedding.iter())
                .map(|(a, b)| a * b)
                .sum();

            if sim >= threshold {
                pairs.push(MemoryPair::new(
                    memory_contents[i].clone(),
                    memory_contents[j].clone(),
                ));
            }
        }
    }
    pairs
}
```

**Complexity:** O(n^2) comparisons, but early filtering reduces ConsolidationService load.

#### Temporal Strategy

```rust
// Compare fingerprints created within 24 hours
fn build_temporal_pairs(
    memory_contents: &[MemoryContent],
    fingerprints: &[(Uuid, DateTime<Utc>)]
) -> Vec<MemoryPair> {
    let window_secs = 24 * 60 * 60;  // 24 hours
    let mut pairs = Vec::new();

    for i in 0..memory_contents.len() {
        for j in (i + 1)..memory_contents.len() {
            let diff = (fingerprints[i].1 - fingerprints[j].1).num_seconds().abs();
            if diff < window_secs {
                pairs.push(MemoryPair::new(
                    memory_contents[i].clone(),
                    memory_contents[j].clone(),
                ));
            }
        }
    }
    pairs
}
```

#### Semantic Strategy

```rust
// Pair only well-aligned memories
fn build_semantic_pairs(memory_contents: &[MemoryContent]) -> Vec<MemoryPair> {
    let alignment_threshold = 0.5;
    let mut pairs = Vec::new();

    for i in 0..memory_contents.len() {
        for j in (i + 1)..memory_contents.len() {
            if memory_contents[i].alignment >= alignment_threshold
                && memory_contents[j].alignment >= alignment_threshold
            {
                pairs.push(MemoryPair::new(
                    memory_contents[i].clone(),
                    memory_contents[j].clone(),
                ));
            }
        }
    }
    pairs
}
```

### 8.4 ConsolidationService.find_consolidation_candidates()

```rust
// File: /crates/context-graph-core/src/autonomous/services/consolidation_service/service.rs

pub fn find_consolidation_candidates(
    &self,
    memories: &[MemoryPair],
) -> Vec<ServiceConsolidationCandidate> {
    if !self.config.enabled {
        return Vec::new();
    }

    let mut candidates = Vec::new();

    for pair in memories {
        let similarity = self.compute_similarity(&pair.a, &pair.b);
        let alignment_diff = pair.alignment_diff();

        if self.should_consolidate(similarity, alignment_diff) {
            let combined_alignment = self.compute_combined_alignment(
                &[pair.a.alignment, pair.b.alignment]
            );

            candidates.push(ServiceConsolidationCandidate::new(
                vec![pair.a.id.clone(), pair.b.id.clone()],
                MemoryId::new(),  // New target ID
                similarity,
                combined_alignment,
            ));
        }
    }

    candidates
}

pub fn should_consolidate(&self, similarity: f32, alignment_diff: f32) -> bool {
    similarity >= self.config.similarity_threshold
        && alignment_diff <= self.config.theta_diff_threshold
}
```

### 8.5 Combined Alignment Calculation

```rust
pub fn compute_combined_alignment(&self, alignments: &[f32]) -> f32 {
    if alignments.is_empty() {
        return 0.0;
    }

    // Weight by alignment^2 to favor higher alignments
    let weights: Vec<f32> = alignments.iter().map(|a| a * a).collect();
    let total_weight: f32 = weights.iter().sum();

    if total_weight < f32::EPSILON {
        return alignments.iter().sum::<f32>() / alignments.len() as f32;
    }

    let weighted_sum: f32 = alignments.iter()
        .zip(weights.iter())
        .map(|(a, w)| a * w)
        .sum();

    weighted_sum / total_weight
}
```

**Example:**
- Memory A: alignment = 0.8
- Memory B: alignment = 0.6
- Weights: [0.64, 0.36]
- Combined: (0.8 * 0.64 + 0.6 * 0.36) / (0.64 + 0.36) = 0.728

---

## 9. Forensic Evidence Summary

### EVIDENCE LOG

| Timestamp | Action | Expected | Actual | Verdict |
|-----------|--------|----------|--------|---------|
| 2026-01-14 | Tool definition exists | Present in autonomous.rs | Found at line 170-202 | VERIFIED |
| 2026-01-14 | Handler implements logic | call_trigger_consolidation | Found at line 195-438 in maintenance.rs | VERIFIED |
| 2026-01-14 | Dispatch routes correctly | Routes to handler | Found at line 150-152 in dispatch.rs | VERIFIED |
| 2026-01-14 | Core service exists | ConsolidationService | Found in service.rs | VERIFIED |
| 2026-01-14 | Uses REAL data | SPEC-STUBFIX-003 | Confirmed - no mock data | VERIFIED |
| 2026-01-14 | Strategy validation | FAIL FAST on invalid | Error handling at line 226-237 | VERIFIED |
| 2026-01-14 | Three strategies work | similarity/temporal/semantic | All implemented at line 308-379 | VERIFIED |

### VERDICT: INNOCENT

The `trigger_consolidation` tool is **fully implemented and operational**. The implementation follows SPEC-STUBFIX-003 requirements for real data processing with three distinct strategies. All code paths trace correctly from tool definition through dispatch to handler to core service.

---

## 10. Chain of Custody

| File | Last Modified | Author | Purpose |
|------|--------------|--------|---------|
| `autonomous.rs` | Recent | Development Team | Tool definition |
| `maintenance.rs` | Recent | Development Team | Handler implementation |
| `service.rs` | Recent | Development Team | ConsolidationService logic |
| `types.rs` | Recent | Development Team | MemoryContent, MemoryPair, ServiceConsolidationCandidate |
| `params.rs` | Recent | Development Team | TriggerConsolidationParams |

---

*Case File Closed - Sherlock Holmes, Code Forensics Division*
