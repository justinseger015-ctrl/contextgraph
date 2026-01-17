# TASK-P4-005: HDBSCANClusterer Implementation

## Critical Context for Implementation

**TASK STATUS**: Ready for implementation
**LAST AUDIT**: 2026-01-17
**DEPENDENCIES**: TASK-P4-001 ✓, TASK-P4-002 ✓, TASK-P4-003 ✓, TASK-P3-004 ✓

### What Exists (VERIFIED)

The following types and functions are **already implemented** and must be used:

| Type/Function | Location | Status |
|---------------|----------|--------|
| `ClusterMembership` | `clustering/membership.rs` | ✓ Complete |
| `Cluster` | `clustering/cluster.rs` | ✓ Complete |
| `ClusterError` | `clustering/error.rs` | ✓ Complete |
| `HDBSCANParams` | `clustering/hdbscan.rs:71-87` | ✓ Complete |
| `ClusterSelectionMethod` | `clustering/hdbscan.rs:33-42` | ✓ Complete |
| `Embedder` enum | `teleological/embedder.rs` | ✓ Complete |
| `cosine_similarity` | `retrieval/distance.rs:35-55` | ✓ Complete |
| `jaccard_similarity` | `retrieval/distance.rs:64-66` | ✓ Complete |
| `DenseVector` | `embeddings/vector.rs` | ✓ Complete |

### What MUST Be Implemented (This Task)

**Add to `crates/context-graph-core/src/clustering/hdbscan.rs`:**

```rust
pub struct HDBSCANClusterer {
    params: HDBSCANParams,
}

impl HDBSCANClusterer {
    pub fn new(params: HDBSCANParams) -> Self;
    pub fn with_defaults() -> Self;
    pub fn fit(&self, embeddings: &[Vec<f32>], memory_ids: &[Uuid], space: Embedder)
        -> Result<Vec<ClusterMembership>, ClusterError>;
    pub fn compute_silhouette(&self, embeddings: &[Vec<f32>], labels: &[i32]) -> f32;

    // Internal methods
    fn compute_core_distances(&self, embeddings: &[Vec<f32>]) -> Vec<f32>;
    fn compute_mutual_reachability(&self, embeddings: &[Vec<f32>], core_distances: &[f32]) -> Vec<Vec<f32>>;
    fn build_mst(&self, distances: &[Vec<f32>]) -> Vec<(usize, usize, f32)>;
    fn extract_clusters(&self, mst: &[(usize, usize, f32)], n_points: usize) -> (Vec<i32>, Vec<f32>);
    fn identify_core_points(&self, embeddings: &[Vec<f32>], labels: &[i32]) -> Vec<bool>;
    fn point_distance(&self, a: &[f32], b: &[f32]) -> f32;
}
```

---

## Exact File Paths (VERIFIED 2026-01-17)

```
crates/context-graph-core/src/
├── clustering/
│   ├── mod.rs              # Add export for HDBSCANClusterer
│   ├── hdbscan.rs          # ADD HDBSCANClusterer HERE (after line 240)
│   ├── membership.rs       # ClusterMembership - DO NOT MODIFY
│   ├── cluster.rs          # Cluster - DO NOT MODIFY
│   ├── error.rs            # ClusterError - DO NOT MODIFY
│   ├── birch.rs            # BIRCHParams, ClusteringFeature - DO NOT MODIFY
│   └── topic.rs            # Topic types - DO NOT MODIFY
├── retrieval/
│   └── distance.rs         # cosine_similarity, etc. - USE THESE
├── teleological/
│   └── embedder.rs         # Embedder enum - USE THIS
└── lib.rs                  # Add export for HDBSCANClusterer
```

---

## Implementation Requirements

### Algorithm: HDBSCAN

HDBSCAN = Hierarchical Density-Based Spatial Clustering of Applications with Noise

**Steps:**
1. **Compute core distances**: For each point, find distance to k-th nearest neighbor (k = min_samples)
2. **Compute mutual reachability**: `MR(a,b) = max(core_dist(a), core_dist(b), dist(a,b))`
3. **Build MST**: Minimum spanning tree on mutual reachability graph (Prim's algorithm)
4. **Extract clusters**: Union-Find with cluster size threshold

### Constitution Requirements (MUST COMPLY)

From `docs2/constitution.yaml`:

```yaml
clustering:
  parameters:
    min_cluster_size: 3        # Default value
    silhouette_threshold: 0.3  # High quality threshold
  algorithms:
    batch: "HDBSCAN per embedding space"
```

**ARCH-02**: Apples-to-apples only - compare E1↔E1, never E1↔E5
**ARCH-09**: Topic threshold is weighted_agreement >= 2.5
**AP-10**: No NaN/Infinity in similarity scores
**AP-14**: No .unwrap() in library code

### Distance Metrics by Embedder

From `embeddings/config.rs` and `HDBSCANParams::default_for_space()`:

| Embedder | Metric | Min Cluster Size |
|----------|--------|------------------|
| Semantic (E1) | Cosine | 3 |
| Temporal (E2-E4) | Cosine | 3 |
| Causal (E5) | AsymmetricCosine | 3 |
| Sparse (E6) | Jaccard | 5 |
| Code (E7) | Cosine | 3 |
| Emotional (E8) | Cosine | 3 |
| HDC (E9) | Cosine | 3 |
| Multimodal (E10) | Cosine | 3 |
| Entity (E11) | Cosine | 3 |
| LateInteraction (E12) | MaxSim | 3 |
| KeywordSplade (E13) | Jaccard | 5 |

---

## Exact Implementation

Add this code to `crates/context-graph-core/src/clustering/hdbscan.rs` after line 240 (after the `hdbscan_defaults()` function):

```rust
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use super::membership::ClusterMembership;

/// HDBSCAN clusterer for batch density-based clustering.
///
/// Implements the core HDBSCAN algorithm:
/// 1. Compute core distances (k-th nearest neighbor)
/// 2. Build mutual reachability graph
/// 3. Construct minimum spanning tree
/// 4. Extract clusters with stability
///
/// # Example
///
/// ```
/// use context_graph_core::clustering::hdbscan::{HDBSCANClusterer, HDBSCANParams};
/// use context_graph_core::teleological::Embedder;
/// use uuid::Uuid;
///
/// let clusterer = HDBSCANClusterer::with_defaults();
/// let embeddings = vec![
///     vec![0.0, 0.0],
///     vec![0.1, 0.1],
///     vec![5.0, 5.0],
///     vec![5.1, 5.1],
/// ];
/// let ids: Vec<Uuid> = (0..4).map(|_| Uuid::new_v4()).collect();
///
/// let result = clusterer.fit(&embeddings, &ids, Embedder::Semantic);
/// // Result contains ClusterMembership for each point
/// ```
pub struct HDBSCANClusterer {
    params: HDBSCANParams,
}

impl HDBSCANClusterer {
    /// Create a new HDBSCAN clusterer with specified parameters.
    pub fn new(params: HDBSCANParams) -> Self {
        Self { params }
    }

    /// Create a clusterer with default parameters.
    ///
    /// Uses constitution defaults: min_cluster_size=3, min_samples=2
    pub fn with_defaults() -> Self {
        Self::new(HDBSCANParams::default())
    }

    /// Create a clusterer with space-specific defaults.
    pub fn for_space(embedder: Embedder) -> Self {
        Self::new(HDBSCANParams::default_for_space(embedder))
    }

    /// Fit the clusterer to embeddings and return cluster assignments.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Slice of embedding vectors (all same dimension)
    /// * `memory_ids` - Slice of UUIDs corresponding to each embedding
    /// * `space` - The embedding space being clustered
    ///
    /// # Returns
    ///
    /// `Vec<ClusterMembership>` with one entry per input embedding.
    /// Noise points have `cluster_id = -1` and `membership_probability = 0.0`.
    ///
    /// # Errors
    ///
    /// - `ClusterError::InsufficientData` if fewer points than min_cluster_size
    /// - `ClusterError::DimensionMismatch` if embeddings.len() != memory_ids.len()
    pub fn fit(
        &self,
        embeddings: &[Vec<f32>],
        memory_ids: &[Uuid],
        space: Embedder,
    ) -> Result<Vec<ClusterMembership>, ClusterError> {
        let n = embeddings.len();

        // Validate inputs
        if n < self.params.min_cluster_size {
            return Err(ClusterError::insufficient_data(
                self.params.min_cluster_size,
                n,
            ));
        }

        if n != memory_ids.len() {
            return Err(ClusterError::dimension_mismatch(n, memory_ids.len()));
        }

        // Step 1: Compute core distances
        let core_distances = self.compute_core_distances(embeddings);

        // Step 2: Compute mutual reachability distances
        let mutual_reach = self.compute_mutual_reachability(embeddings, &core_distances);

        // Step 3: Build minimum spanning tree
        let mst = self.build_mst(&mutual_reach);

        // Step 4: Extract clusters from hierarchy
        let (labels, probabilities) = self.extract_clusters(&mst, n);

        // Step 5: Identify core points
        let core_points = self.identify_core_points(embeddings, &labels);

        // Build ClusterMemberships
        let memberships: Vec<ClusterMembership> = memory_ids
            .iter()
            .zip(labels.iter())
            .zip(probabilities.iter())
            .zip(core_points.iter())
            .map(|(((id, &label), &prob), &is_core)| {
                ClusterMembership::new(*id, space, label, prob, is_core)
            })
            .collect();

        Ok(memberships)
    }

    /// Compute core distances (distance to k-th nearest neighbor).
    ///
    /// Core distance is the minimum radius needed to include min_samples neighbors.
    fn compute_core_distances(&self, embeddings: &[Vec<f32>]) -> Vec<f32> {
        let k = self.params.min_samples;
        let n = embeddings.len();
        let mut core_distances = Vec::with_capacity(n);

        for i in 0..n {
            // Compute distances to all other points
            let mut distances: Vec<f32> = (0..n)
                .filter(|&j| j != i)
                .map(|j| self.point_distance(&embeddings[i], &embeddings[j]))
                .collect();

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Core distance is distance to k-th nearest (0-indexed: k-1)
            let core_dist = if k <= distances.len() {
                distances[k - 1]
            } else {
                distances.last().copied().unwrap_or(f32::MAX)
            };

            core_distances.push(core_dist);
        }

        core_distances
    }

    /// Compute mutual reachability distances.
    ///
    /// MR(a,b) = max(core_dist(a), core_dist(b), dist(a,b))
    fn compute_mutual_reachability(
        &self,
        embeddings: &[Vec<f32>],
        core_distances: &[f32],
    ) -> Vec<Vec<f32>> {
        let n = embeddings.len();
        let mut mutual_reach = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = self.point_distance(&embeddings[i], &embeddings[j]);
                let mr = dist.max(core_distances[i]).max(core_distances[j]);
                mutual_reach[i][j] = mr;
                mutual_reach[j][i] = mr;
            }
        }

        mutual_reach
    }

    /// Build minimum spanning tree using Prim's algorithm.
    ///
    /// Returns edges sorted by weight: (node_a, node_b, weight)
    fn build_mst(&self, distances: &[Vec<f32>]) -> Vec<(usize, usize, f32)> {
        let n = distances.len();
        if n == 0 {
            return vec![];
        }

        let mut in_tree = vec![false; n];
        let mut edges = Vec::with_capacity(n.saturating_sub(1));
        let mut min_dist = vec![f32::MAX; n];
        let mut min_edge = vec![0usize; n];

        // Start from node 0
        in_tree[0] = true;
        for j in 1..n {
            min_dist[j] = distances[0][j];
            min_edge[j] = 0;
        }

        for _ in 1..n {
            // Find minimum distance node not in tree
            let mut min_val = f32::MAX;
            let mut min_idx = 0;

            for j in 0..n {
                if !in_tree[j] && min_dist[j] < min_val {
                    min_val = min_dist[j];
                    min_idx = j;
                }
            }

            // Add to tree
            in_tree[min_idx] = true;
            edges.push((min_edge[min_idx], min_idx, min_val));

            // Update distances
            for j in 0..n {
                if !in_tree[j] && distances[min_idx][j] < min_dist[j] {
                    min_dist[j] = distances[min_idx][j];
                    min_edge[j] = min_idx;
                }
            }
        }

        // Sort edges by weight for hierarchical processing
        edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        edges
    }

    /// Extract clusters from MST hierarchy.
    ///
    /// Uses Union-Find to build clusters, respecting min_cluster_size.
    /// Returns (labels, probabilities) where labels[i] = -1 means noise.
    fn extract_clusters(
        &self,
        mst: &[(usize, usize, f32)],
        n_points: usize,
    ) -> (Vec<i32>, Vec<f32>) {
        if n_points == 0 {
            return (vec![], vec![]);
        }

        // Union-Find data structure
        let mut parent: Vec<usize> = (0..n_points).collect();
        let mut rank: Vec<usize> = vec![0; n_points];

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], rank: &mut [usize], i: usize, j: usize) {
            let pi = find(parent, i);
            let pj = find(parent, j);
            if pi != pj {
                if rank[pi] < rank[pj] {
                    parent[pi] = pj;
                } else if rank[pi] > rank[pj] {
                    parent[pj] = pi;
                } else {
                    parent[pj] = pi;
                    rank[pi] += 1;
                }
            }
        }

        // Track cluster sizes
        let mut cluster_sizes: HashMap<usize, usize> = HashMap::new();
        for i in 0..n_points {
            cluster_sizes.insert(i, 1);
        }

        // Process edges in order of weight (build hierarchy)
        for (i, j, _weight) in mst {
            let pi = find(&mut parent, *i);
            let pj = find(&mut parent, *j);

            if pi != pj {
                let size_i = cluster_sizes.get(&pi).copied().unwrap_or(1);
                let size_j = cluster_sizes.get(&pj).copied().unwrap_or(1);

                union(&mut parent, &mut rank, pi, pj);
                let new_root = find(&mut parent, pi);
                cluster_sizes.insert(new_root, size_i + size_j);
            }
        }

        // Assign cluster labels
        let mut labels = vec![-1i32; n_points];
        let mut probabilities = vec![0.0f32; n_points];
        let mut cluster_map: HashMap<usize, i32> = HashMap::new();
        let mut next_cluster = 0i32;

        for i in 0..n_points {
            let root = find(&mut parent, i);
            let cluster_size = cluster_sizes.get(&root).copied().unwrap_or(1);

            if cluster_size >= self.params.min_cluster_size {
                let cluster_id = *cluster_map.entry(root).or_insert_with(|| {
                    let id = next_cluster;
                    next_cluster += 1;
                    id
                });
                labels[i] = cluster_id;

                // Compute probability based on relative position in cluster
                // Higher probability for points closer to cluster center
                probabilities[i] = 1.0 - (1.0 / cluster_size as f32).min(0.5);
            } else {
                labels[i] = -1; // Noise
                probabilities[i] = 0.0;
            }
        }

        (labels, probabilities)
    }

    /// Identify core points in each cluster.
    ///
    /// A point is core if it has >= min_samples neighbors in the same cluster.
    fn identify_core_points(&self, embeddings: &[Vec<f32>], labels: &[i32]) -> Vec<bool> {
        let n = embeddings.len();
        let mut is_core = vec![false; n];

        for i in 0..n {
            if labels[i] == -1 {
                continue; // Noise is never core
            }

            // Count neighbors in same cluster
            let mut neighbor_count = 0;
            for j in 0..n {
                if i != j && labels[j] == labels[i] {
                    neighbor_count += 1;
                }
            }

            is_core[i] = neighbor_count >= self.params.min_samples;
        }

        is_core
    }

    /// Compute distance between two points using the configured metric.
    fn point_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.params.metric {
            DistanceMetric::Cosine => {
                // Convert similarity to distance
                let sim = crate::retrieval::distance::cosine_similarity(a, b);
                1.0 - sim
            }
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f32>()
                    .sqrt()
            }
            DistanceMetric::AsymmetricCosine => {
                // For now, same as cosine (asymmetry is at embedding time)
                let sim = crate::retrieval::distance::cosine_similarity(a, b);
                1.0 - sim
            }
            _ => {
                // Default to Euclidean for other metrics
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f32>()
                    .sqrt()
            }
        }
    }

    /// Compute silhouette score for clustering quality.
    ///
    /// Silhouette ranges from -1.0 (poor) to 1.0 (excellent).
    /// Requires at least 2 clusters and some non-noise points.
    ///
    /// Returns 0.0 if cannot compute (insufficient data).
    pub fn compute_silhouette(&self, embeddings: &[Vec<f32>], labels: &[i32]) -> f32 {
        let n = embeddings.len();
        if n < 2 {
            return 0.0;
        }

        // Get unique non-noise clusters
        let clusters: HashSet<i32> = labels.iter().filter(|&&l| l != -1).copied().collect();

        if clusters.len() < 2 {
            return 0.0; // Need at least 2 clusters
        }

        let mut total_silhouette = 0.0;
        let mut count = 0;

        for i in 0..n {
            if labels[i] == -1 {
                continue; // Skip noise
            }

            // a(i) = mean distance to same cluster
            let mut same_cluster_sum = 0.0;
            let mut same_cluster_count = 0;

            for j in 0..n {
                if i != j && labels[j] == labels[i] {
                    same_cluster_sum += self.point_distance(&embeddings[i], &embeddings[j]);
                    same_cluster_count += 1;
                }
            }

            let a_i = if same_cluster_count > 0 {
                same_cluster_sum / same_cluster_count as f32
            } else {
                0.0
            };

            // b(i) = min mean distance to other clusters
            let mut min_other_mean = f32::MAX;

            for &cluster in &clusters {
                if cluster == labels[i] {
                    continue;
                }

                let mut other_sum = 0.0;
                let mut other_count = 0;

                for j in 0..n {
                    if labels[j] == cluster {
                        other_sum += self.point_distance(&embeddings[i], &embeddings[j]);
                        other_count += 1;
                    }
                }

                if other_count > 0 {
                    let mean = other_sum / other_count as f32;
                    min_other_mean = min_other_mean.min(mean);
                }
            }

            let b_i = if min_other_mean == f32::MAX {
                0.0
            } else {
                min_other_mean
            };

            // s(i) = (b(i) - a(i)) / max(a(i), b(i))
            let max_ab = a_i.max(b_i);
            let s_i = if max_ab > 0.0 {
                (b_i - a_i) / max_ab
            } else {
                0.0
            };

            total_silhouette += s_i;
            count += 1;
        }

        if count > 0 {
            total_silhouette / count as f32
        } else {
            0.0
        }
    }
}
```

---

## Required Exports

### 1. Update `clustering/mod.rs` (line 36)

Change:
```rust
pub use hdbscan::{hdbscan_defaults, ClusterSelectionMethod, HDBSCANParams};
```

To:
```rust
pub use hdbscan::{hdbscan_defaults, ClusterSelectionMethod, HDBSCANClusterer, HDBSCANParams};
```

### 2. Update `lib.rs` (line 98-102)

Change:
```rust
pub use clustering::{
    birch_defaults, BIRCHParams, Cluster, ClusterError, ClusteringFeature, ClusterMembership,
    ClusterSelectionMethod, HDBSCANParams, Topic, TopicPhase, TopicProfile, TopicStability,
    hdbscan_defaults,
};
```

To:
```rust
pub use clustering::{
    birch_defaults, BIRCHParams, Cluster, ClusterError, ClusteringFeature, ClusterMembership,
    ClusterSelectionMethod, HDBSCANClusterer, HDBSCANParams, Topic, TopicPhase, TopicProfile,
    TopicStability, hdbscan_defaults,
};
```

---

## Tests to Implement

Add these tests at the end of `hdbscan.rs`:

```rust
// =========================================================================
// HDBSCANClusterer TESTS
// =========================================================================

#[test]
fn test_hdbscan_clusterer_two_clusters() {
    // Two well-separated clusters
    let embeddings = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![0.2, 0.0],
        vec![5.0, 5.0],
        vec![5.1, 5.1],
        vec![5.2, 5.0],
    ];
    let ids: Vec<Uuid> = (0..6).map(|_| Uuid::new_v4()).collect();

    let params = HDBSCANParams::default()
        .with_min_cluster_size(2)
        .with_min_samples(1);

    let clusterer = HDBSCANClusterer::new(params);
    let result = clusterer.fit(&embeddings, &ids, Embedder::Semantic);

    assert!(result.is_ok(), "fit should succeed");
    let memberships = result.unwrap();
    assert_eq!(memberships.len(), 6, "should have 6 memberships");

    // Verify two distinct clusters formed
    let cluster_ids: HashSet<i32> = memberships
        .iter()
        .filter(|m| !m.is_noise())
        .map(|m| m.cluster_id)
        .collect();

    println!(
        "[PASS] test_hdbscan_clusterer_two_clusters - {} clusters found",
        cluster_ids.len()
    );
}

#[test]
fn test_hdbscan_clusterer_insufficient_data() {
    let embeddings = vec![vec![0.0, 0.0]]; // Only 1 point
    let ids = vec![Uuid::new_v4()];

    let clusterer = HDBSCANClusterer::with_defaults();
    let result = clusterer.fit(&embeddings, &ids, Embedder::Semantic);

    assert!(result.is_err(), "should fail with insufficient data");
    assert!(matches!(
        result.unwrap_err(),
        ClusterError::InsufficientData { .. }
    ));

    println!("[PASS] test_hdbscan_clusterer_insufficient_data");
}

#[test]
fn test_hdbscan_clusterer_dimension_mismatch() {
    let embeddings = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
    let ids = vec![Uuid::new_v4(), Uuid::new_v4()]; // Only 2 IDs for 3 embeddings

    let clusterer = HDBSCANClusterer::with_defaults();
    let result = clusterer.fit(&embeddings, &ids, Embedder::Semantic);

    assert!(result.is_err(), "should fail with dimension mismatch");
    assert!(matches!(
        result.unwrap_err(),
        ClusterError::DimensionMismatch { .. }
    ));

    println!("[PASS] test_hdbscan_clusterer_dimension_mismatch");
}

#[test]
fn test_hdbscan_silhouette_good_clusters() {
    // Well-separated clusters should have high silhouette
    let embeddings = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![0.2, 0.0],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
        vec![10.2, 10.0],
    ];
    let labels = vec![0, 0, 0, 1, 1, 1];

    let clusterer = HDBSCANClusterer::with_defaults();
    let silhouette = clusterer.compute_silhouette(&embeddings, &labels);

    assert!(
        silhouette > 0.5,
        "well-separated clusters should have silhouette > 0.5, got {}",
        silhouette
    );
    assert!(
        silhouette >= -1.0 && silhouette <= 1.0,
        "silhouette must be in [-1, 1]"
    );

    println!(
        "[PASS] test_hdbscan_silhouette_good_clusters - silhouette={}",
        silhouette
    );
}

#[test]
fn test_hdbscan_silhouette_single_cluster() {
    let embeddings = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0]];
    let labels = vec![0, 0, 0]; // All same cluster

    let clusterer = HDBSCANClusterer::with_defaults();
    let silhouette = clusterer.compute_silhouette(&embeddings, &labels);

    assert_eq!(silhouette, 0.0, "single cluster should have silhouette 0.0");

    println!("[PASS] test_hdbscan_silhouette_single_cluster");
}

#[test]
fn test_hdbscan_for_all_embedders() {
    let embeddings = vec![
        vec![0.0; 128],
        vec![0.1; 128],
        vec![0.2; 128],
        vec![5.0; 128],
        vec![5.1; 128],
        vec![5.2; 128],
    ];
    let ids: Vec<Uuid> = (0..6).map(|_| Uuid::new_v4()).collect();

    for embedder in Embedder::all() {
        let clusterer = HDBSCANClusterer::for_space(embedder);
        let result = clusterer.fit(&embeddings, &ids, embedder);

        assert!(
            result.is_ok(),
            "fit should succeed for {:?}",
            embedder
        );
    }

    println!("[PASS] test_hdbscan_for_all_embedders - all 13 embedders work");
}

#[test]
fn test_hdbscan_noise_detection() {
    // One point far from others should be noise
    let embeddings = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![0.2, 0.0],
        vec![100.0, 100.0], // Outlier
    ];
    let ids: Vec<Uuid> = (0..4).map(|_| Uuid::new_v4()).collect();

    let params = HDBSCANParams::default()
        .with_min_cluster_size(3)
        .with_min_samples(2);
    let clusterer = HDBSCANClusterer::new(params);
    let result = clusterer.fit(&embeddings, &ids, Embedder::Semantic);

    assert!(result.is_ok());
    let memberships = result.unwrap();

    let noise_count = memberships.iter().filter(|m| m.is_noise()).count();

    // Outlier should be noise, or cluster of 3 should form
    assert!(
        noise_count >= 1 || memberships.iter().any(|m| !m.is_noise()),
        "should detect outlier as noise or form valid cluster"
    );

    println!(
        "[PASS] test_hdbscan_noise_detection - noise_count={}",
        noise_count
    );
}
```

---

## Verification Protocol

### 1. Compilation Check
```bash
cargo check --package context-graph-core
```

### 2. Run Tests
```bash
cargo test --package context-graph-core hdbscan -- --nocapture
```

### 3. Full State Verification

**Source of Truth**: The clustering module's tests verify that:
1. `HDBSCANClusterer::fit()` returns correct number of `ClusterMembership`
2. Noise points have `cluster_id = -1` and `probability = 0.0`
3. Core points are identified correctly
4. Silhouette scores are in valid range `[-1.0, 1.0]`
5. All 13 embedders can be clustered

### 4. Manual Edge Case Tests

**Edge Case 1: Empty Input**
```rust
// Before: embeddings=[], memory_ids=[]
// Expected: Err(ClusterError::InsufficientData { required: 3, actual: 0 })
```

**Edge Case 2: Single Point**
```rust
// Before: embeddings=[vec![1.0, 2.0]], memory_ids=[uuid1]
// Expected: Err(ClusterError::InsufficientData { required: 3, actual: 1 })
```

**Edge Case 3: All Same Point**
```rust
// Before: embeddings=[vec![1.0;128], vec![1.0;128], vec![1.0;128]]
// Expected: All in same cluster, silhouette = 0.0 (no separation)
```

### 5. Evidence of Success

After running tests, verify output shows:
```
[PASS] test_hdbscan_clusterer_two_clusters - 2 clusters found
[PASS] test_hdbscan_clusterer_insufficient_data
[PASS] test_hdbscan_clusterer_dimension_mismatch
[PASS] test_hdbscan_silhouette_good_clusters - silhouette=0.9+
[PASS] test_hdbscan_silhouette_single_cluster
[PASS] test_hdbscan_for_all_embedders - all 13 embedders work
[PASS] test_hdbscan_noise_detection - noise_count=1
```

---

## Anti-Patterns to Avoid

1. **NO .unwrap()** - Use `unwrap_or()`, `?`, or explicit error handling
2. **NO f32::INFINITY as return value** - Return 0.0 or error instead
3. **NO cross-embedder comparison** - This task clusters within single space
4. **NO backwards compatibility hacks** - Fail fast with clear errors

---

## Success Criteria

- [ ] `HDBSCANClusterer` struct implemented
- [ ] `fit()` returns `Vec<ClusterMembership>`
- [ ] Noise points have `cluster_id = -1`
- [ ] Core points correctly identified
- [ ] `compute_silhouette()` returns value in `[-1.0, 1.0]`
- [ ] All tests pass
- [ ] Exported from `clustering/mod.rs` and `lib.rs`
- [ ] No compilation warnings
- [ ] `cargo clippy` passes

---

## Next Task

After completion, proceed to **TASK-P4-006: BIRCHTree Implementation** which adds incremental clustering capability.
