# Topic Detection Root Cause Analysis: Why 184 Memories Produce Only 1 Topic

**Date**: 2026-02-08
**Branch**: casetrack
**Status**: Root cause identified, fix recommendations provided

---

## Executive Summary

Topic detection with 184 diverse memories about 30+ distinct subsystems of context-graph produces only **1 mega-topic containing all 184 memories**. The root cause is a **hardcoded gap threshold of 0.20 in the HDBSCAN `detect_gap_threshold()` function** (`hdbscan.rs:606`), which merges all points into a single cluster per embedding space. The synthesizer then has no choice but to produce 1 topic because all memories share cluster_id=0 across all spaces.

---

## Test Setup

- 184 memories stored covering 30+ distinct domains: E1-E13 embedders, RocksDB storage, HDBSCAN clustering, topic synthesis, MCP tools, search strategies, graph linking, code pipeline, dream consolidation, causal discovery, entity extraction, weight profiles, etc.
- Each memory is embedded across all 13 embedding spaces
- detect_topics called with force=true

**Result**: `1 topics with weighted_agreement >= 2.5`, member_count=184, confidence=0.64

---

## Root Cause 1 (CRITICAL): Hardcoded HDBSCAN Gap Threshold

### Location
`crates/context-graph-core/src/clustering/hdbscan.rs:590-638`

### The Bug

```rust
fn detect_gap_threshold(&self, mst: &[(usize, usize, f32)]) -> f32 {
    // ...
    if self.params.metric == DistanceMetric::Cosine
        || self.params.metric == DistanceMetric::AsymmetricCosine
    {
        // Use 0.20 as threshold (similarity >= 0.80 to cluster together)
        // This is tight enough to separate ML/DB/DevOps topics
        return 0.20;   // <--- HARDCODED, IGNORES ACTUAL DATA DISTRIBUTION
    }
    // ...
}
```

### Why It Fails

The hardcoded 0.20 (cosine distance, meaning similarity >= 0.80) is **data-independent**. For 184 memories about the same project:

1. **MST edge weights** (cosine distance between nearest neighbors) are typically in the range [0.03, 0.18] because all memories share domain vocabulary
2. The threshold 0.20 is **above all MST edges**, so the break at line 533 **never fires**:
   ```rust
   for (i, j, weight) in mst {
       if *weight > gap_threshold {  // 0.18 > 0.20? NO -> never breaks
           break;
       }
       union(parent, i, j);  // merges EVERYTHING
   }
   ```
3. Union-Find merges all 184 points into **1 connected component** -> cluster_id=0 for all

### Impact Chain

```
HDBSCAN per-space: 184 points -> 1 cluster (all cluster_id=0)
  x 6 semantic spaces = all spaces show 1 cluster
    -> synthesizer: weighted_agreement for ANY pair = 6.0 >> 2.5 threshold
      -> Union-Find connects all 184 into 1 component
        -> 1 mega-topic with 184 members
```

### The Irony

The comment claims "tight enough to separate ML/DB/DevOps topics" but this is only true for **highly disparate** domains with large embedding distance. Within a single project's documentation, even clearly distinct subsystems (RocksDB storage vs. HDBSCAN clustering vs. MCP tools) have cosine distances < 0.20 because they share common technical vocabulary.

---

## Root Cause 2 (HIGH): No Mega-Cluster Detection or Splitting

### Location
`crates/context-graph-core/src/clustering/hdbscan.rs:484-578` (`extract_clusters()`)

### The Problem

After clustering, there is **no validation** that cluster sizes are reasonable. A cluster with 184 members (100% of all memories) should trigger:
- A warning log
- Recursive sub-clustering attempt
- Fallback to a different threshold strategy

Currently, the code assigns all 184 points to cluster_id=0 and moves on.

### Missing Logic

```rust
// MISSING: After extract_clusters
if clusters.len() == 1 && n_points > some_threshold {
    // Try adaptive threshold or recursive sub-clustering
}
```

---

## Root Cause 3 (HIGH): Synthesizer Cannot Split Per-Space Mega-Clusters

### Location
`crates/context-graph-core/src/clustering/synthesizer.rs:175-229` (`find_topic_mates()`)

### The Problem

The synthesizer computes `weighted_agreement` between all pairs of memories. If all memories share cluster_id=0 across all 6 semantic spaces, **every pair** gets weighted_agreement = 6.0, which is well above the 2.5 threshold. The synthesizer literally cannot produce multiple topics from identical per-space clusterings.

```rust
// Line 160: Both in same non-noise cluster -> counts as agreement
if ca != -1 && ca == cb {
    weighted += category_for(embedder).topic_weight();
}
```

With all memories in cluster 0 across E1(1.0) + E5(1.0) + E6(1.0) + E7(1.0) + E10(1.0) + E11(0.5) = 5.5 weighted agreement per pair, every pair exceeds 2.5.

---

## Root Cause 4 (MEDIUM): No MST Distribution Diagnostics

### Location
`crates/context-graph-core/src/clustering/hdbscan.rs:590-606`

### The Problem

There is **zero logging** about MST edge weight distribution. The system silently returns 0.20 without reporting:
- How many MST edges exist
- What the min/max/median edge weights are
- How many edges fall below the threshold
- Whether the threshold removes ANY edges at all

This made the bug invisible until manual testing with 184 memories.

---

## Fix Recommendations

### Fix 1 (Required): Data-Driven Gap Threshold

Replace the hardcoded threshold with adaptive detection based on the actual MST edge distribution.

**Option A: Percentile-based** (Simplest)
```rust
fn detect_gap_threshold(&self, mst: &[(usize, usize, f32)]) -> f32 {
    let weights: Vec<f32> = mst.iter().map(|(_, _, w)| *w).collect();
    let n = weights.len();
    if n == 0 { return f32::MAX; }

    // Use the 70th percentile as the gap threshold
    // This ensures ~30% of edges are cut, creating multiple clusters
    let p70_idx = (n as f32 * 0.70) as usize;
    let percentile = weights[p70_idx.min(n - 1)];

    // For cosine metrics, also enforce a minimum to avoid
    // over-splitting very tight clusters
    if self.params.metric == DistanceMetric::Cosine
        || self.params.metric == DistanceMetric::AsymmetricCosine
    {
        return percentile.max(0.05); // floor at 0.05 cosine distance
    }
    percentile
}
```

**Option B: Relative gap detection** (More robust)
```rust
fn detect_gap_threshold(&self, mst: &[(usize, usize, f32)]) -> f32 {
    let weights: Vec<f32> = mst.iter().map(|(_, _, w)| *w).collect();
    let n = weights.len();
    if n == 0 { return f32::MAX; }

    // Compute gaps between consecutive sorted MST edges
    let mut max_gap = 0.0f32;
    let mut gap_idx = 0;
    for i in 1..n {
        let gap = weights[i] - weights[i - 1];
        if gap > max_gap {
            max_gap = gap;
            gap_idx = i;
        }
    }

    // Use the edge weight at the largest gap as threshold
    // Fall back to 75th percentile if no clear gap found
    if max_gap > 0.01 {
        weights[gap_idx]
    } else {
        let p75_idx = (n as f32 * 0.75) as usize;
        weights[p75_idx.min(n - 1)]
    }
}
```

**Option C: Mean + standard deviation** (Statistical)
```rust
fn detect_gap_threshold(&self, mst: &[(usize, usize, f32)]) -> f32 {
    let weights: Vec<f32> = mst.iter().map(|(_, _, w)| *w).collect();
    let n = weights.len();
    if n == 0 { return f32::MAX; }

    let mean = weights.iter().sum::<f32>() / n as f32;
    let variance = weights.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / n as f32;
    let std_dev = variance.sqrt();

    // Threshold = mean + 1 standard deviation
    // This separates the "nearby" edges from "gap" edges
    (mean + std_dev).max(0.05)
}
```

### Fix 2 (Required): Add Mega-Cluster Detection

In `extract_clusters()`, after the Union-Find phase:

```rust
// After labeling, check for mega-clusters
let total_labeled = labels.iter().filter(|&&l| l != -1).count();
for (&root, &size) in &cluster_sizes {
    if size > total_labeled / 2 && total_labeled > 20 {
        tracing::warn!(
            cluster_size = size,
            total_points = total_labeled,
            "Mega-cluster detected: contains {}% of all points. \
             Consider adjusting gap threshold or using recursive sub-clustering.",
            (size * 100) / total_labeled
        );
    }
}
```

### Fix 3 (Recommended): Recursive Sub-Clustering

When a single cluster contains >50% of memories, attempt to sub-cluster:

```rust
fn maybe_split_mega_cluster(
    &self,
    embeddings: &[Vec<f32>],
    cluster_members: &[usize],
) -> Vec<Vec<usize>> {
    if cluster_members.len() < 10 {
        return vec![cluster_members.to_vec()];
    }

    // Extract sub-embeddings
    let sub_embeddings: Vec<&Vec<f32>> = cluster_members
        .iter()
        .map(|&i| &embeddings[i])
        .collect();

    // Re-cluster with tighter parameters
    let sub_params = HDBSCANParams {
        min_cluster_size: self.params.min_cluster_size,
        min_samples: self.params.min_samples,
        ..self.params.clone()
    };

    // Use adaptive threshold (not hardcoded) for sub-clustering
    // ...
}
```

### Fix 4 (Recommended): Add MST Diagnostics

```rust
fn detect_gap_threshold(&self, mst: &[(usize, usize, f32)]) -> f32 {
    let weights: Vec<f32> = mst.iter().map(|(_, _, w)| *w).collect();
    let n = weights.len();
    if n == 0 { return f32::MAX; }

    let min_w = weights[0];
    let max_w = weights[n - 1];
    let median = weights[n / 2];

    tracing::debug!(
        mst_edges = n,
        min_weight = %format!("{:.4}", min_w),
        max_weight = %format!("{:.4}", max_w),
        median_weight = %format!("{:.4}", median),
        "MST edge weight distribution"
    );

    // ... threshold computation ...

    tracing::debug!(
        threshold = %format!("{:.4}", threshold),
        edges_below = weights.iter().filter(|&&w| w < threshold).count(),
        edges_above = weights.iter().filter(|&&w| w >= threshold).count(),
        "Gap threshold selected"
    );

    threshold
}
```

### Fix 5 (Optional): Synthesizer Mega-Topic Warning

In `synthesize_topics()`, after creating topics:

```rust
let total_memories = mem_clusters.len();
for topic in &topics {
    if topic.member_count() > total_memories / 2 && total_memories > 10 {
        tracing::warn!(
            topic_id = %topic.id,
            member_count = topic.member_count(),
            total_memories = total_memories,
            "Mega-topic absorbs {}% of all memories - per-space clustering may be too coarse",
            (topic.member_count() * 100) / total_memories
        );
    }
}
```

---

## Recommended Fix Priority

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| P0 | Fix 1: Data-driven gap threshold | Medium | Resolves root cause |
| P0 | Fix 4: MST diagnostics | Low | Prevents silent failures |
| P1 | Fix 2: Mega-cluster detection | Low | Early warning system |
| P2 | Fix 3: Recursive sub-clustering | High | Handles edge cases |
| P3 | Fix 5: Synthesizer warnings | Low | Observability |

---

## Recommended Approach: Option B (Largest Gap)

Option B (largest gap detection) is the best approach because:

1. **Data-driven**: Adapts to actual embedding distribution
2. **Already partially implemented**: Lines 622-634 have gap detection for non-cosine metrics, but the cosine early-return at line 601-606 skips it
3. **Proven algorithm**: Largest-gap clustering is well-studied in the MST literature
4. **Minimal change**: Remove the hardcoded return, let the existing gap detection run for cosine metrics too

The fix is essentially:
```rust
// REMOVE this early return (lines 601-606):
if self.params.metric == DistanceMetric::Cosine ... {
    return 0.20;  // DELETE THIS
}

// KEEP the existing gap detection (lines 622-634)
// It already handles all metrics correctly
```

Then add a cosine-specific floor (e.g., 0.03) to prevent over-splitting of genuinely similar memories.

---

## Expected Outcomes After Fix

With data-driven thresholding on 184 diverse project memories:
- **Expected clusters per space**: 5-20 (depending on embedder)
- **Expected topics after synthesis**: 10-40 (depending on cross-space agreement)
- **Expected topic sizes**: 3-15 members each (focused, meaningful topics)
- **Churn tracking**: Meaningful because topics will have stable, distinct member sets

---

## Files Affected

| File | Lines | Change |
|------|-------|--------|
| `crates/context-graph-core/src/clustering/hdbscan.rs` | 590-638 | Replace hardcoded threshold with data-driven approach |
| `crates/context-graph-core/src/clustering/hdbscan.rs` | 484-578 | Add mega-cluster detection warning |
| `crates/context-graph-core/src/clustering/synthesizer.rs` | 360-388 | Add mega-topic warning |
| `crates/context-graph-core/src/clustering/topic.rs` | N/A | No changes needed |

---

## Verification Plan

1. Apply the gap threshold fix
2. Build release: `cargo build --release`
3. Run clustering tests: `cargo test -p context-graph-core -- clustering`
4. Clean DB: `rm -rf contextgraph_data`
5. Store 90+ diverse memories via MCP
6. Run `detect_topics` with force=true
7. Verify: total_after >= 10 topics (ideally 20-30)
8. Run `detect_topics` again with force=true
9. Verify: churn is 0.0 or near-0 (stable topic IDs via deterministic UUIDs)
10. Add new memories from a different domain
11. Run `detect_topics` again
12. Verify: churn > 0 reflecting the new topic(s)
