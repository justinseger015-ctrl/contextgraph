//! Type definitions for goal discovery pipeline.
//!
//! This module contains all configuration types, clustering types, and result types
//! used by the goal discovery system.

use crate::autonomous::evolution::GoalLevel;
use crate::teleological::Embedder;
use crate::types::fingerprint::TeleologicalArray;

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
