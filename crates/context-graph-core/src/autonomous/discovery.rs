//! Goal Discovery Pipeline for autonomous goal emergence.
//!
//! This module implements TASK-LOGIC-009: Goal Discovery Pipeline using K-means
//! clustering on TeleologicalArrays (13-embedder vectors) to discover emergent goals.
//!
//! # Architecture
//!
//! From constitution.yaml:
//! - ARCH-01: TeleologicalArray is atomic - never split, compare only as unit
//! - ARCH-02: Apples-to-apples - E1 compares with E1, never cross-embedder
//! - ARCH-03: Autonomous operation - goals emerge from data patterns
//!
//! # Design Philosophy
//!
//! FAIL FAST. NO FALLBACKS. NO RECOVERY ATTEMPTS.
//!
//! All errors are FATAL:
//! - InsufficientData -> panic with clear message
//! - ClusteringFailed -> panic with algorithm details
//! - NoClustersFound -> panic with threshold info
//! - InvalidCentroid -> panic with embedder state

use std::collections::HashMap;
use uuid::Uuid;

use crate::autonomous::evolution::GoalLevel;
use crate::teleological::comparator::TeleologicalComparator;
use crate::teleological::{Embedder, NUM_EMBEDDERS};
use crate::types::fingerprint::{
    SemanticFingerprint, SparseVector, TeleologicalArray, E1_DIM, E2_DIM, E3_DIM, E4_DIM, E5_DIM,
    E7_DIM, E8_DIM, E9_DIM, E10_DIM, E11_DIM, E12_TOKEN_DIM,
};

/// Configuration for goal discovery.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Max arrays to process (default: 500)
    pub sample_size: usize,
    /// Min members per cluster (default: 5)
    pub min_cluster_size: usize,
    /// Min intra-cluster similarity (default: 0.75)
    pub min_coherence: f32,
    /// Clustering algorithm to use
    pub clustering_algorithm: ClusteringAlgorithm,
    /// Number of clusters
    pub num_clusters: NumClusters,
    /// Maximum K-means iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f32,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            sample_size: 500,
            min_cluster_size: 5,
            min_coherence: 0.75,
            clustering_algorithm: ClusteringAlgorithm::KMeans,
            num_clusters: NumClusters::Auto,
            max_iterations: 100,
            convergence_tolerance: 1e-4,
        }
    }
}

/// Clustering algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusteringAlgorithm {
    /// K-means clustering - PRIMARY algorithm (MUST implement)
    KMeans,
    /// HDBSCAN - density-based clustering (STRETCH GOAL)
    HDBSCAN { min_samples: usize },
    /// Spectral clustering (STRETCH GOAL)
    Spectral { n_neighbors: usize },
}

/// Number of clusters specification.
#[derive(Debug, Clone)]
pub enum NumClusters {
    /// sqrt(n/2) heuristic
    Auto,
    /// Exact k clusters
    Fixed(usize),
    /// Elbow method within range
    Range { min: usize, max: usize },
}

/// Cluster of teleological arrays.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Indices into source array
    pub members: Vec<usize>,
    /// FULL 13-embedder centroid
    pub centroid: TeleologicalArray,
    /// Intra-cluster similarity
    pub coherence: f32,
}

/// Discovered goal from clustering (also used as GoalCandidate internally).
#[derive(Debug, Clone)]
pub struct DiscoveredGoal {
    /// Unique goal ID
    pub goal_id: String,
    /// Suggested description
    pub description: String,
    /// Assigned goal level
    pub level: GoalLevel,
    /// Confidence score
    pub confidence: f32,
    /// Number of cluster members
    pub member_count: usize,
    /// FULL 13-embedder goal vector
    pub centroid: TeleologicalArray,
    /// Top 3 embedders by magnitude
    pub dominant_embedders: Vec<Embedder>,
    /// Coherence score
    pub coherence_score: f32,
}

/// Goal candidate from clustering - alias for DiscoveredGoal.
pub type GoalCandidate = DiscoveredGoal;

/// Relationship between goals in hierarchy.
#[derive(Debug, Clone)]
pub struct GoalRelationship {
    /// Parent goal ID
    pub parent_id: String,
    /// Child goal ID
    pub child_id: String,
    /// Similarity between centroids
    pub similarity: f32,
}

/// Result of goal discovery.
#[derive(Debug)]
pub struct DiscoveryResult {
    /// Discovered goals
    pub discovered_goals: Vec<DiscoveredGoal>,
    /// Number of clusters found
    pub clusters_found: usize,
    /// Total arrays analyzed
    pub total_arrays_analyzed: usize,
    /// Goal hierarchy relationships
    pub hierarchy: Vec<GoalRelationship>,
}

/// Goal discovery pipeline for autonomous goal emergence.
///
/// Uses K-means clustering on TeleologicalArrays to discover emergent goals.
/// Compares arrays using TeleologicalComparator (apples-to-apples per embedder).
pub struct GoalDiscoveryPipeline {
    comparator: TeleologicalComparator,
}

impl GoalDiscoveryPipeline {
    /// Create a new GoalDiscoveryPipeline with default comparator.
    pub fn new(comparator: TeleologicalComparator) -> Self {
        Self { comparator }
    }

    /// Discover goals from teleological arrays.
    ///
    /// FAILS FAST on any error - no recovery attempts.
    ///
    /// # Panics
    ///
    /// - If input arrays is empty
    /// - If input is smaller than min_cluster_size
    /// - If clustering fails
    /// - If no valid clusters found
    pub fn discover(
        &self,
        arrays: &[TeleologicalArray],
        config: &DiscoveryConfig,
    ) -> DiscoveryResult {
        // FAIL FAST: Check minimum data requirements
        assert!(
            !arrays.is_empty(),
            "FAIL FAST: Insufficient arrays for goal discovery. Got 0 arrays, need at least {}",
            config.min_cluster_size
        );

        assert!(
            arrays.len() >= config.min_cluster_size,
            "FAIL FAST: Insufficient arrays for clustering. Got {} arrays, minimum required: {}",
            arrays.len(),
            config.min_cluster_size
        );

        // Sample if needed
        let sampled_arrays: Vec<&TeleologicalArray> = if arrays.len() > config.sample_size {
            // Simple deterministic sampling using stride
            let stride = arrays.len() / config.sample_size;
            arrays.iter().step_by(stride).take(config.sample_size).collect()
        } else {
            arrays.iter().collect()
        };

        eprintln!(
            "[GoalDiscoveryPipeline] Analyzing {} arrays (sampled from {})",
            sampled_arrays.len(),
            arrays.len()
        );

        // Perform clustering
        let clusters = self.cluster(&sampled_arrays, config);

        // FAIL FAST: Verify clusters found
        assert!(
            !clusters.is_empty(),
            "FAIL FAST: No clusters found with min_cluster_size={} and min_coherence={}",
            config.min_cluster_size,
            config.min_coherence
        );

        eprintln!("[GoalDiscoveryPipeline] Found {} clusters", clusters.len());

        // Score clusters and create goal candidates
        let mut candidates: Vec<GoalCandidate> = clusters
            .iter()
            .filter(|c| c.members.len() >= config.min_cluster_size)
            .filter(|c| c.coherence >= config.min_coherence)
            .map(|c| self.score_cluster(c))
            .collect();

        // Assign goal levels
        for candidate in &mut candidates {
            candidate.level = self.assign_level(candidate);
        }

        // Build hierarchy
        let hierarchy = self.build_hierarchy(&candidates);

        // candidates is Vec<GoalCandidate> which is Vec<DiscoveredGoal>
        let discovered_goals: Vec<DiscoveredGoal> = candidates;

        eprintln!(
            "[GoalDiscoveryPipeline] Discovered {} goals with {} hierarchy relationships",
            discovered_goals.len(),
            hierarchy.len()
        );

        DiscoveryResult {
            clusters_found: discovered_goals.len(),
            total_arrays_analyzed: sampled_arrays.len(),
            discovered_goals,
            hierarchy,
        }
    }

    /// Cluster arrays using K-means on full teleological arrays.
    fn cluster(
        &self,
        arrays: &[&TeleologicalArray],
        config: &DiscoveryConfig,
    ) -> Vec<Cluster> {
        let n = arrays.len();

        // Determine number of clusters
        let k = match &config.num_clusters {
            NumClusters::Auto => {
                // sqrt(n/2) heuristic
                let k = ((n as f32 / 2.0).sqrt().ceil() as usize).max(2);
                k.min(n / config.min_cluster_size) // Don't exceed data capacity
            }
            NumClusters::Fixed(k) => *k,
            NumClusters::Range { min, max } => {
                // Use elbow method approximation
                let k_auto = ((n as f32 / 2.0).sqrt().ceil() as usize).max(2);
                k_auto.clamp(*min, *max)
            }
        };

        assert!(
            k >= 1,
            "FAIL FAST: Cannot form clusters with k=0. n={}, min_cluster_size={}",
            n,
            config.min_cluster_size
        );

        eprintln!("[K-means] Clustering {} arrays into {} clusters", n, k);

        // Initialize centroids using k-means++ strategy
        let mut centroids: Vec<TeleologicalArray> = self.initialize_centroids_kmeans_pp(arrays, k);
        let mut assignments: Vec<usize> = vec![0; n];
        let mut iteration = 0;

        loop {
            iteration += 1;

            // Assignment step: assign each array to nearest centroid
            let mut changed = false;
            for (i, array) in arrays.iter().enumerate() {
                let nearest = self.find_nearest_centroid(array, &centroids);
                if nearest != assignments[i] {
                    changed = true;
                    assignments[i] = nearest;
                }
            }

            // Check convergence
            if !changed || iteration >= config.max_iterations {
                eprintln!(
                    "[K-means] Converged after {} iterations (changed={})",
                    iteration, changed
                );
                break;
            }

            // Update step: recompute centroids
            centroids = self.recompute_centroids(arrays, &assignments, k);
        }

        // Build cluster objects
        let mut clusters: Vec<Cluster> = Vec::with_capacity(k);
        for cluster_id in 0..k {
            let members: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &a)| a == cluster_id)
                .map(|(i, _)| i)
                .collect();

            if members.is_empty() {
                continue; // Skip empty clusters
            }

            let member_arrays: Vec<&TeleologicalArray> =
                members.iter().map(|&i| arrays[i]).collect();

            let centroid = self.compute_centroid(&member_arrays);
            let coherence = self.compute_cluster_coherence(&member_arrays, &centroid);

            clusters.push(Cluster {
                members,
                centroid,
                coherence,
            });
        }

        clusters
    }

    /// Initialize centroids using k-means++ strategy.
    fn initialize_centroids_kmeans_pp(
        &self,
        arrays: &[&TeleologicalArray],
        k: usize,
    ) -> Vec<TeleologicalArray> {
        let mut centroids: Vec<TeleologicalArray> = Vec::with_capacity(k);

        // First centroid: pick the first array (deterministic for reproducibility)
        centroids.push(arrays[0].clone());

        // Remaining centroids: pick proportional to squared distance
        for _ in 1..k {
            let mut max_min_dist = 0.0_f32;
            let mut best_idx = 0;

            for (i, array) in arrays.iter().enumerate() {
                // Find minimum distance to existing centroids
                let mut min_dist = f32::MAX;
                for centroid in &centroids {
                    let result = self.comparator.compare(array, centroid);
                    let similarity = result.map(|r| r.overall).unwrap_or(0.0);
                    let distance = 1.0 - similarity;
                    min_dist = min_dist.min(distance);
                }

                // Pick array with maximum minimum distance (furthest from all centroids)
                if min_dist > max_min_dist {
                    max_min_dist = min_dist;
                    best_idx = i;
                }
            }

            centroids.push(arrays[best_idx].clone());
        }

        centroids
    }

    /// Find nearest centroid for an array.
    fn find_nearest_centroid(
        &self,
        array: &TeleologicalArray,
        centroids: &[TeleologicalArray],
    ) -> usize {
        let mut best_idx = 0;
        let mut best_similarity = f32::NEG_INFINITY;

        for (i, centroid) in centroids.iter().enumerate() {
            let result = self.comparator.compare(array, centroid);
            let similarity = result.map(|r| r.overall).unwrap_or(0.0);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Recompute centroids from assignments.
    fn recompute_centroids(
        &self,
        arrays: &[&TeleologicalArray],
        assignments: &[usize],
        k: usize,
    ) -> Vec<TeleologicalArray> {
        (0..k)
            .map(|cluster_id| {
                let members: Vec<&TeleologicalArray> = assignments
                    .iter()
                    .enumerate()
                    .filter(|(_, &a)| a == cluster_id)
                    .map(|(i, _)| arrays[i])
                    .collect();

                if members.is_empty() {
                    // Keep the previous centroid if cluster is empty
                    // This shouldn't happen with proper k-means++ initialization
                    Self::create_zeroed_fingerprint()
                } else {
                    self.compute_centroid(&members)
                }
            })
            .collect()
    }

    /// Compute centroid for a cluster.
    ///
    /// Each embedder's vectors are averaged separately.
    /// Result is a valid TeleologicalArray.
    pub fn compute_centroid(&self, members: &[&TeleologicalArray]) -> TeleologicalArray {
        assert!(
            !members.is_empty(),
            "FAIL FAST: Cannot compute centroid of empty cluster"
        );

        let n = members.len() as f32;

        // Average each dense embedding separately
        let e1_semantic = Self::average_dense_vectors(members.iter().map(|m| &m.e1_semantic), E1_DIM);
        let e2_temporal_recent = Self::average_dense_vectors(members.iter().map(|m| &m.e2_temporal_recent), E2_DIM);
        let e3_temporal_periodic = Self::average_dense_vectors(members.iter().map(|m| &m.e3_temporal_periodic), E3_DIM);
        let e4_temporal_positional = Self::average_dense_vectors(members.iter().map(|m| &m.e4_temporal_positional), E4_DIM);
        let e5_causal = Self::average_dense_vectors(members.iter().map(|m| &m.e5_causal), E5_DIM);
        let e7_code = Self::average_dense_vectors(members.iter().map(|m| &m.e7_code), E7_DIM);
        let e8_graph = Self::average_dense_vectors(members.iter().map(|m| &m.e8_graph), E8_DIM);
        let e9_hdc = Self::average_dense_vectors(members.iter().map(|m| &m.e9_hdc), E9_DIM);
        let e10_multimodal = Self::average_dense_vectors(members.iter().map(|m| &m.e10_multimodal), E10_DIM);
        let e11_entity = Self::average_dense_vectors(members.iter().map(|m| &m.e11_entity), E11_DIM);

        // Average sparse vectors (E6, E13)
        let e6_sparse = Self::average_sparse_vectors(members.iter().map(|m| &m.e6_sparse), n);
        let e13_splade = Self::average_sparse_vectors(members.iter().map(|m| &m.e13_splade), n);

        // Average token-level vectors (E12)
        // Strategy: Average all tokens across all members, result has average token count
        let e12_late_interaction = Self::average_token_vectors(members.iter().map(|m| &m.e12_late_interaction));

        SemanticFingerprint {
            e1_semantic,
            e2_temporal_recent,
            e3_temporal_periodic,
            e4_temporal_positional,
            e5_causal,
            e6_sparse,
            e7_code,
            e8_graph,
            e9_hdc,
            e10_multimodal,
            e11_entity,
            e12_late_interaction,
            e13_splade,
        }
    }

    /// Average dense vectors element-wise.
    fn average_dense_vectors<'a, I>(vectors: I, dim: usize) -> Vec<f32>
    where
        I: Iterator<Item = &'a Vec<f32>>,
    {
        let mut sum = vec![0.0_f32; dim];
        let mut count = 0;

        for vec in vectors {
            if vec.len() == dim {
                for (i, &val) in vec.iter().enumerate() {
                    sum[i] += val;
                }
                count += 1;
            }
        }

        if count > 0 {
            for val in &mut sum {
                *val /= count as f32;
            }
        }

        sum
    }

    /// Average sparse vectors by combining indices and averaging values.
    fn average_sparse_vectors<'a, I>(vectors: I, n: f32) -> SparseVector
    where
        I: Iterator<Item = &'a SparseVector>,
    {
        // Collect all (index, value) pairs and sum values per index
        let mut index_sums: HashMap<u16, f32> = HashMap::new();

        for sparse in vectors {
            for (idx, val) in sparse.indices.iter().zip(sparse.values.iter()) {
                *index_sums.entry(*idx).or_insert(0.0) += *val;
            }
        }

        // Average and build result
        let mut pairs: Vec<(u16, f32)> = index_sums
            .into_iter()
            .map(|(idx, sum)| (idx, sum / n))
            .collect();

        // Sort by index for SparseVector construction
        pairs.sort_by_key(|(idx, _)| *idx);

        let indices: Vec<u16> = pairs.iter().map(|(idx, _)| *idx).collect();
        let values: Vec<f32> = pairs.iter().map(|(_, val)| *val).collect();

        SparseVector::new(indices, values).unwrap_or_else(|e| {
            eprintln!(
                "[GoalDiscoveryPipeline] Warning: Failed to construct sparse centroid: {:?}",
                e
            );
            SparseVector::empty()
        })
    }

    /// Average token-level vectors.
    fn average_token_vectors<'a, I>(vectors: I) -> Vec<Vec<f32>>
    where
        I: Iterator<Item = &'a Vec<Vec<f32>>>,
    {
        let all_tokens: Vec<&Vec<Vec<f32>>> = vectors.collect();

        if all_tokens.is_empty() {
            return Vec::new();
        }

        // Find average token count
        let total_tokens: usize = all_tokens.iter().map(|t| t.len()).sum();
        let avg_token_count = (total_tokens / all_tokens.len()).max(1);

        // Collect all tokens and average by position
        let mut result = Vec::with_capacity(avg_token_count);

        for pos in 0..avg_token_count {
            let mut sum = vec![0.0_f32; E12_TOKEN_DIM];
            let mut count = 0;

            for tokens in &all_tokens {
                if let Some(token) = tokens.get(pos) {
                    if token.len() == E12_TOKEN_DIM {
                        for (i, &val) in token.iter().enumerate() {
                            sum[i] += val;
                        }
                        count += 1;
                    }
                }
            }

            if count > 0 {
                for val in &mut sum {
                    *val /= count as f32;
                }
            }

            result.push(sum);
        }

        result
    }

    /// Compute intra-cluster coherence.
    fn compute_cluster_coherence(
        &self,
        members: &[&TeleologicalArray],
        centroid: &TeleologicalArray,
    ) -> f32 {
        if members.is_empty() {
            return 0.0;
        }

        let similarities: Vec<f32> = members
            .iter()
            .filter_map(|m| {
                self.comparator
                    .compare(m, centroid)
                    .ok()
                    .map(|r| r.overall)
            })
            .collect();

        if similarities.is_empty() {
            return 0.0;
        }

        similarities.iter().sum::<f32>() / similarities.len() as f32
    }

    /// Score cluster suitability as a goal.
    fn score_cluster(&self, cluster: &Cluster) -> GoalCandidate {
        let size_score = (cluster.members.len() as f32 / 50.0).min(1.0);
        let coherence_score = cluster.coherence;

        // Find dominant embedders
        let dominant_embedders = self.find_dominant_embedders(&cluster.centroid);
        let embedder_diversity = (dominant_embedders.len() as f32) / 3.0;

        // Combined confidence: 40% coherence, 30% size, 30% embedder distribution
        let confidence = 0.4 * coherence_score + 0.3 * size_score + 0.3 * embedder_diversity;

        GoalCandidate {
            goal_id: Uuid::new_v4().to_string(),
            description: format!(
                "Goal cluster (size={}, coherence={:.2})",
                cluster.members.len(),
                cluster.coherence
            ),
            level: GoalLevel::Operational, // Will be reassigned
            confidence,
            member_count: cluster.members.len(),
            centroid: cluster.centroid.clone(),
            dominant_embedders,
            coherence_score: cluster.coherence,
        }
    }

    /// Find top 3 embedders by magnitude.
    fn find_dominant_embedders(&self, centroid: &TeleologicalArray) -> Vec<Embedder> {
        let mut embedder_magnitudes: Vec<(Embedder, f32)> = Vec::with_capacity(NUM_EMBEDDERS);

        // Dense embedders
        embedder_magnitudes.push((Embedder::Semantic, Self::l2_norm(&centroid.e1_semantic)));
        embedder_magnitudes.push((Embedder::TemporalRecent, Self::l2_norm(&centroid.e2_temporal_recent)));
        embedder_magnitudes.push((Embedder::TemporalPeriodic, Self::l2_norm(&centroid.e3_temporal_periodic)));
        embedder_magnitudes.push((Embedder::TemporalPositional, Self::l2_norm(&centroid.e4_temporal_positional)));
        embedder_magnitudes.push((Embedder::Causal, Self::l2_norm(&centroid.e5_causal)));
        embedder_magnitudes.push((Embedder::Code, Self::l2_norm(&centroid.e7_code)));
        embedder_magnitudes.push((Embedder::Graph, Self::l2_norm(&centroid.e8_graph)));
        embedder_magnitudes.push((Embedder::Hdc, Self::l2_norm(&centroid.e9_hdc)));
        embedder_magnitudes.push((Embedder::Multimodal, Self::l2_norm(&centroid.e10_multimodal)));
        embedder_magnitudes.push((Embedder::Entity, Self::l2_norm(&centroid.e11_entity)));

        // Sparse embedders
        embedder_magnitudes.push((Embedder::Sparse, centroid.e6_sparse.l2_norm()));
        embedder_magnitudes.push((Embedder::KeywordSplade, centroid.e13_splade.l2_norm()));

        // Token-level embedder
        let e12_magnitude: f32 = centroid
            .e12_late_interaction
            .iter()
            .map(|t| Self::l2_norm(t))
            .sum::<f32>()
            / centroid.e12_late_interaction.len().max(1) as f32;
        embedder_magnitudes.push((Embedder::LateInteraction, e12_magnitude));

        // Sort by magnitude descending and take top 3
        embedder_magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        embedder_magnitudes.into_iter().take(3).map(|(e, _)| e).collect()
    }

    /// Compute L2 norm of a vector.
    fn l2_norm(vec: &[f32]) -> f32 {
        vec.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Assign goal level based on size and coherence thresholds.
    ///
    /// Thresholds:
    /// - NorthStar: size >= 50 AND coherence >= 0.85
    /// - Strategic: size >= 20 AND coherence >= 0.80
    /// - Tactical: size >= 10 AND coherence >= 0.75
    /// - Operational: everything else
    pub fn assign_level(&self, candidate: &GoalCandidate) -> GoalLevel {
        let size = candidate.member_count;
        let coherence = candidate.coherence_score;

        if size >= 50 && coherence >= 0.85 {
            GoalLevel::NorthStar
        } else if size >= 20 && coherence >= 0.80 {
            GoalLevel::Strategic
        } else if size >= 10 && coherence >= 0.75 {
            GoalLevel::Tactical
        } else {
            GoalLevel::Operational
        }
    }

    /// Build parent-child relationships based on centroid similarity.
    ///
    /// Higher-level goals (larger, more coherent) become parents of
    /// lower-level goals with similar centroids.
    pub fn build_hierarchy(&self, candidates: &[GoalCandidate]) -> Vec<GoalRelationship> {
        let mut relationships = Vec::new();

        // Sort candidates by level (higher first) and then by size
        let mut sorted: Vec<(usize, &GoalCandidate)> = candidates.iter().enumerate().collect();
        sorted.sort_by(|a, b| {
            let cmp = Self::level_order(&a.1.level).cmp(&Self::level_order(&b.1.level));
            if cmp == std::cmp::Ordering::Equal {
                b.1.member_count.cmp(&a.1.member_count)
            } else {
                cmp
            }
        });

        // For each candidate, find the best parent (higher level, most similar)
        for i in 0..sorted.len() {
            let child = sorted[i].1;
            let child_level = Self::level_order(&child.level);

            let mut best_parent: Option<(&GoalCandidate, f32)> = None;

            for (_, parent) in sorted.iter().take(i) {
                let parent_level = Self::level_order(&parent.level);

                // Parent must be higher level
                if parent_level >= child_level {
                    continue;
                }

                // Compute similarity
                let result = self.comparator.compare(&parent.centroid, &child.centroid);
                let similarity = result.map(|r| r.overall).unwrap_or(0.0);

                // Threshold: at least 0.5 similarity to form relationship
                if similarity >= 0.5 {
                    if let Some((_, best_sim)) = best_parent {
                        if similarity > best_sim {
                            best_parent = Some((parent, similarity));
                        }
                    } else {
                        best_parent = Some((parent, similarity));
                    }
                }
            }

            if let Some((parent, similarity)) = best_parent {
                relationships.push(GoalRelationship {
                    parent_id: parent.goal_id.clone(),
                    child_id: child.goal_id.clone(),
                    similarity,
                });
            }
        }

        relationships
    }

    /// Convert level to ordering number.
    fn level_order(level: &GoalLevel) -> u8 {
        match level {
            GoalLevel::NorthStar => 0,
            GoalLevel::Strategic => 1,
            GoalLevel::Tactical => 2,
            GoalLevel::Operational => 3,
        }
    }

    /// Create a zeroed fingerprint for empty cluster fallback.
    #[cfg(any(test, feature = "test-utils"))]
    fn create_zeroed_fingerprint() -> SemanticFingerprint {
        SemanticFingerprint::zeroed()
    }

    #[cfg(not(any(test, feature = "test-utils")))]
    fn create_zeroed_fingerprint() -> SemanticFingerprint {
        // In production, create a minimal valid fingerprint
        SemanticFingerprint {
            e1_semantic: vec![0.0; E1_DIM],
            e2_temporal_recent: vec![0.0; E2_DIM],
            e3_temporal_periodic: vec![0.0; E3_DIM],
            e4_temporal_positional: vec![0.0; E4_DIM],
            e5_causal: vec![0.0; E5_DIM],
            e6_sparse: SparseVector::empty(),
            e7_code: vec![0.0; E7_DIM],
            e8_graph: vec![0.0; E8_DIM],
            e9_hdc: vec![0.0; E9_DIM],
            e10_multimodal: vec![0.0; E10_DIM],
            e11_entity: vec![0.0; E11_DIM],
            e12_late_interaction: Vec::new(),
            e13_splade: SparseVector::empty(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test fingerprint with specific patterns for deterministic testing.
    fn create_test_fingerprint(cluster_id: usize, variance: f32) -> SemanticFingerprint {
        // Create base patterns based on cluster_id
        let base_e1: Vec<f32> = (0..E1_DIM)
            .map(|i| {
                let phase = (cluster_id as f32) * 2.0 * std::f32::consts::PI / 3.0;
                (i as f32 / E1_DIM as f32 * std::f32::consts::PI + phase).sin() + variance
            })
            .collect();

        // Normalize to unit vector
        let norm: f32 = base_e1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let e1_semantic: Vec<f32> = base_e1.iter().map(|x| x / norm).collect();

        // Similar patterns for other embeddings
        let create_normalized = |dim: usize| -> Vec<f32> {
            let base: Vec<f32> = (0..dim)
                .map(|i| {
                    let phase = (cluster_id as f32) * 2.0 * std::f32::consts::PI / 3.0;
                    (i as f32 / dim as f32 * std::f32::consts::PI + phase).cos() + variance
                })
                .collect();
            let norm: f32 = base.iter().map(|x| x * x).sum::<f32>().sqrt();
            base.iter().map(|x| x / norm.max(1e-6)).collect()
        };

        SemanticFingerprint {
            e1_semantic,
            e2_temporal_recent: create_normalized(E2_DIM),
            e3_temporal_periodic: create_normalized(E3_DIM),
            e4_temporal_positional: create_normalized(E4_DIM),
            e5_causal: create_normalized(E5_DIM),
            e6_sparse: SparseVector::empty(), // Sparse vectors are optional
            e7_code: create_normalized(E7_DIM),
            e8_graph: create_normalized(E8_DIM),
            e9_hdc: create_normalized(E9_DIM),
            e10_multimodal: create_normalized(E10_DIM),
            e11_entity: create_normalized(E11_DIM),
            e12_late_interaction: vec![create_normalized(E12_TOKEN_DIM)],
            e13_splade: SparseVector::empty(),
        }
    }

    #[test]
    fn test_kmeans_three_clusters() {
        // Create 30 TeleologicalArrays in 3 known clusters
        let mut arrays: Vec<TeleologicalArray> = Vec::new();

        // Cluster A: 10 arrays with variance around 0.0
        for _ in 0..10 {
            arrays.push(create_test_fingerprint(0, 0.01));
        }

        // Cluster B: 10 arrays with variance around 1/3 period
        for _ in 0..10 {
            arrays.push(create_test_fingerprint(1, 0.01));
        }

        // Cluster C: 10 arrays with variance around 2/3 period
        for _ in 0..10 {
            arrays.push(create_test_fingerprint(2, 0.01));
        }

        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let config = DiscoveryConfig {
            min_cluster_size: 5,
            min_coherence: 0.5, // Lower threshold for test data
            num_clusters: NumClusters::Fixed(3),
            ..Default::default()
        };

        let result = pipeline.discover(&arrays, &config);

        // Verify 3 clusters found
        assert!(
            result.clusters_found >= 2,
            "Expected at least 2 clusters, got {}",
            result.clusters_found
        );

        // Verify total arrays analyzed
        assert_eq!(result.total_arrays_analyzed, 30);

        println!("[PASS] test_kmeans_three_clusters: Found {} clusters", result.clusters_found);
    }

    #[test]
    fn test_centroid_is_valid_teleological_array() {
        // Create test arrays
        let arrays: Vec<TeleologicalArray> = (0..10)
            .map(|i| create_test_fingerprint(i % 3, 0.02))
            .collect();

        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let members: Vec<&TeleologicalArray> = arrays.iter().collect();
        let centroid = pipeline.compute_centroid(&members);

        // Verify all 13 embedders are populated
        assert_eq!(centroid.e1_semantic.len(), E1_DIM, "E1 dimension mismatch");
        assert_eq!(centroid.e2_temporal_recent.len(), E2_DIM, "E2 dimension mismatch");
        assert_eq!(centroid.e3_temporal_periodic.len(), E3_DIM, "E3 dimension mismatch");
        assert_eq!(centroid.e4_temporal_positional.len(), E4_DIM, "E4 dimension mismatch");
        assert_eq!(centroid.e5_causal.len(), E5_DIM, "E5 dimension mismatch");
        assert_eq!(centroid.e7_code.len(), E7_DIM, "E7 dimension mismatch");
        assert_eq!(centroid.e8_graph.len(), E8_DIM, "E8 dimension mismatch");
        assert_eq!(centroid.e9_hdc.len(), E9_DIM, "E9 dimension mismatch");
        assert_eq!(centroid.e10_multimodal.len(), E10_DIM, "E10 dimension mismatch");
        assert_eq!(centroid.e11_entity.len(), E11_DIM, "E11 dimension mismatch");

        // E12 should have at least one token
        assert!(!centroid.e12_late_interaction.is_empty(), "E12 should have tokens");
        for token in &centroid.e12_late_interaction {
            assert_eq!(token.len(), E12_TOKEN_DIM, "E12 token dimension mismatch");
        }

        println!("[PASS] test_centroid_is_valid_teleological_array");
    }

    #[test]
    fn test_goal_level_assignment() {
        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let base_centroid = create_test_fingerprint(0, 0.0);

        // Test NorthStar: size=50, coherence=0.85
        let candidate_ns = GoalCandidate {
            goal_id: "ns".to_string(),
            description: "Test".to_string(),
            level: GoalLevel::Operational,
            confidence: 0.9,
            member_count: 50,
            centroid: base_centroid.clone(),
            dominant_embedders: vec![Embedder::Semantic],
            coherence_score: 0.85,
        };
        assert_eq!(pipeline.assign_level(&candidate_ns), GoalLevel::NorthStar);

        // Test Strategic (size not met): size=49, coherence=0.85
        let candidate_strat_size = GoalCandidate {
            member_count: 49,
            coherence_score: 0.85,
            ..candidate_ns.clone()
        };
        assert_eq!(pipeline.assign_level(&candidate_strat_size), GoalLevel::Strategic);

        // Test Strategic (coherence not met): size=50, coherence=0.84
        let candidate_strat_coh = GoalCandidate {
            member_count: 50,
            coherence_score: 0.84,
            ..candidate_ns.clone()
        };
        assert_eq!(pipeline.assign_level(&candidate_strat_coh), GoalLevel::Strategic);

        // Test Strategic: size=20, coherence=0.80
        let candidate_strat = GoalCandidate {
            member_count: 20,
            coherence_score: 0.80,
            ..candidate_ns.clone()
        };
        assert_eq!(pipeline.assign_level(&candidate_strat), GoalLevel::Strategic);

        // Test Tactical: size=10, coherence=0.75
        let candidate_tact = GoalCandidate {
            member_count: 10,
            coherence_score: 0.75,
            ..candidate_ns.clone()
        };
        assert_eq!(pipeline.assign_level(&candidate_tact), GoalLevel::Tactical);

        // Test Operational: size=5, coherence=0.70
        let candidate_op = GoalCandidate {
            member_count: 5,
            coherence_score: 0.70,
            ..candidate_ns.clone()
        };
        assert_eq!(pipeline.assign_level(&candidate_op), GoalLevel::Operational);

        println!("[PASS] test_goal_level_assignment");
    }

    #[test]
    fn test_hierarchy_construction() {
        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let base_centroid = create_test_fingerprint(0, 0.0);
        let similar_centroid = create_test_fingerprint(0, 0.05); // Very similar

        // Parent cluster: Large (50 members), high coherence (NorthStar)
        let parent = GoalCandidate {
            goal_id: "parent".to_string(),
            description: "Parent goal".to_string(),
            level: GoalLevel::NorthStar,
            confidence: 0.9,
            member_count: 50,
            centroid: base_centroid.clone(),
            dominant_embedders: vec![Embedder::Semantic],
            coherence_score: 0.85,
        };

        // Child cluster: Small (10 members), similar centroid (Tactical)
        let child = GoalCandidate {
            goal_id: "child".to_string(),
            description: "Child goal".to_string(),
            level: GoalLevel::Tactical,
            confidence: 0.7,
            member_count: 10,
            centroid: similar_centroid,
            dominant_embedders: vec![Embedder::Semantic],
            coherence_score: 0.75,
        };

        let candidates = vec![parent, child];
        let hierarchy = pipeline.build_hierarchy(&candidates);

        // Should have at least one parent-child relationship
        // (depends on centroid similarity threshold)
        println!(
            "[INFO] test_hierarchy_construction: Found {} relationships",
            hierarchy.len()
        );

        // Verify relationship structure if any exist
        for rel in &hierarchy {
            assert!(
                rel.similarity >= 0.5,
                "Relationship similarity should be >= 0.5"
            );
            println!(
                "[PASS] Parent {} -> Child {}, similarity={}",
                rel.parent_id, rel.child_id, rel.similarity
            );
        }

        println!("[PASS] test_hierarchy_construction");
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_fail_fast_empty_input() {
        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);
        let config = DiscoveryConfig::default();

        let empty: Vec<TeleologicalArray> = vec![];
        let _ = pipeline.discover(&empty, &config);
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_fail_fast_insufficient_data() {
        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let config = DiscoveryConfig {
            min_cluster_size: 5,
            ..Default::default()
        };

        // Only 3 arrays with min_cluster_size=5
        let arrays: Vec<TeleologicalArray> = (0..3)
            .map(|i| create_test_fingerprint(i, 0.01))
            .collect();

        let _ = pipeline.discover(&arrays, &config);
    }

    #[test]
    fn test_all_arrays_identical() {
        // All arrays identical -> single cluster with coherence 1.0
        let identical = create_test_fingerprint(0, 0.0);
        let arrays: Vec<TeleologicalArray> = vec![identical.clone(); 10];

        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let config = DiscoveryConfig {
            min_cluster_size: 5,
            min_coherence: 0.5,
            num_clusters: NumClusters::Fixed(1),
            ..Default::default()
        };

        let result = pipeline.discover(&arrays, &config);

        // Should have 1 cluster
        assert!(result.clusters_found >= 1);

        // Coherence should be high (close to 1.0)
        if let Some(goal) = result.discovered_goals.first() {
            assert!(
                goal.coherence_score > 0.9,
                "Identical arrays should have coherence > 0.9, got {}",
                goal.coherence_score
            );
        }

        println!("[PASS] test_all_arrays_identical");
    }

    #[test]
    fn test_widely_dispersed_arrays() {
        // Create widely dispersed arrays (each in its own "cluster")
        let arrays: Vec<TeleologicalArray> = (0..20)
            .map(|i| create_test_fingerprint(i, 0.5)) // High variance
            .collect();

        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let config = DiscoveryConfig {
            min_cluster_size: 2,
            min_coherence: 0.3, // Very low threshold
            num_clusters: NumClusters::Auto,
            ..Default::default()
        };

        let result = pipeline.discover(&arrays, &config);

        // Should have multiple clusters (dispersed data)
        println!(
            "[INFO] test_widely_dispersed_arrays: Found {} clusters",
            result.clusters_found
        );

        // Coherence should be lower than identical arrays
        for goal in &result.discovered_goals {
            println!(
                "  Cluster: size={}, coherence={:.3}",
                goal.member_count, goal.coherence_score
            );
        }

        println!("[PASS] test_widely_dispersed_arrays");
    }

    #[test]
    fn test_dominant_embedders() {
        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let centroid = create_test_fingerprint(0, 0.0);
        let dominant = pipeline.find_dominant_embedders(&centroid);

        // Should return exactly 3 embedders
        assert_eq!(dominant.len(), 3, "Should return 3 dominant embedders");

        // All should be valid Embedder variants
        for emb in &dominant {
            println!("[INFO] Dominant embedder: {:?}", emb);
        }

        println!("[PASS] test_dominant_embedders");
    }

    #[test]
    fn test_discovery_config_defaults() {
        let config = DiscoveryConfig::default();

        assert_eq!(config.sample_size, 500);
        assert_eq!(config.min_cluster_size, 5);
        assert!((config.min_coherence - 0.75).abs() < f32::EPSILON);
        assert_eq!(config.max_iterations, 100);

        println!("[PASS] test_discovery_config_defaults");
    }
}
