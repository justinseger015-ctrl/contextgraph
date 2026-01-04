//! Contradiction detection algorithm.
//!
//! # M04-T21: Implement Contradiction Detection
//!
//! Combines semantic similarity search with explicit CONTRADICTS edges
//! to identify conflicting knowledge in the graph.
//!
//! # FAIL FAST
//!
//! All errors are explicit - no graceful degradation.
//! Invalid inputs fail immediately with clear error messages.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::info;
use uuid::Uuid;

use crate::error::{GraphError, GraphResult};
use crate::index::FaissGpuIndex;
use crate::search::{semantic_search, NodeMetadataProvider, SearchFilters};
use crate::storage::edges::GraphEdge;
use crate::storage::GraphStorage;
use crate::traversal::bfs::{bfs_traverse, BfsParams};

// Re-export core types
pub use context_graph_core::marblestone::{ContradictionType, Domain, EdgeType, NeurotransmitterWeights};

/// Result from contradiction detection.
///
/// Contains information about a detected contradiction between two nodes.
#[derive(Debug, Clone)]
pub struct ContradictionResult {
    /// The node that contradicts the query node (UUID).
    pub contradicting_node_id: Uuid,

    /// Type of contradiction detected.
    pub contradiction_type: ContradictionType,

    /// Overall confidence score [0, 1].
    /// Higher values indicate stronger contradiction evidence.
    pub confidence: f32,

    /// Semantic similarity to query node [0, 1].
    /// High similarity with contradiction indicates direct opposition.
    pub semantic_similarity: f32,

    /// Weight of explicit CONTRADICTS edge (if exists).
    pub edge_weight: Option<f32>,

    /// Whether there's an explicit contradiction edge.
    pub has_explicit_edge: bool,

    /// Evidence supporting the contradiction.
    pub evidence: Vec<String>,
}

impl ContradictionResult {
    /// Check if this is a high-confidence contradiction.
    ///
    /// # Arguments
    /// * `threshold` - Minimum confidence to consider high
    #[inline]
    pub fn is_high_confidence(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Get severity based on type and confidence.
    ///
    /// Severity = confidence * type_severity
    /// DirectOpposition has highest type severity (1.0)
    #[inline]
    pub fn severity(&self) -> f32 {
        self.confidence * self.contradiction_type.severity()
    }
}

/// Parameters for contradiction detection.
///
/// Controls sensitivity, search depth, and evidence weighting.
#[derive(Debug, Clone)]
pub struct ContradictionParams {
    /// Minimum confidence threshold [0, 1].
    /// Contradictions below this are not returned.
    pub threshold: f32,

    /// Number of semantic similarity candidates to consider.
    pub semantic_k: usize,

    /// Minimum semantic similarity to consider [0, 1].
    pub min_similarity: f32,

    /// BFS depth for graph exploration.
    pub graph_depth: usize,

    /// Weight given to explicit edges vs semantic similarity.
    /// Higher = more weight to explicit edges.
    /// Range [0, 1].
    pub explicit_edge_weight: f32,
}

impl Default for ContradictionParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            semantic_k: 50,
            min_similarity: 0.3,
            graph_depth: 2,
            explicit_edge_weight: 0.6,
        }
    }
}

impl ContradictionParams {
    /// Builder: set confidence threshold.
    #[must_use]
    pub fn threshold(mut self, t: f32) -> Self {
        self.threshold = t.clamp(0.0, 1.0);
        self
    }

    /// Builder: set semantic k.
    #[must_use]
    pub fn semantic_k(mut self, k: usize) -> Self {
        self.semantic_k = k;
        self
    }

    /// Builder: set minimum similarity.
    #[must_use]
    pub fn min_similarity(mut self, s: f32) -> Self {
        self.min_similarity = s.clamp(0.0, 1.0);
        self
    }

    /// Builder: set graph depth.
    #[must_use]
    pub fn graph_depth(mut self, d: usize) -> Self {
        self.graph_depth = d;
        self
    }

    /// Builder: high sensitivity (lower threshold, more candidates).
    #[must_use]
    pub fn high_sensitivity(self) -> Self {
        self.threshold(0.3).semantic_k(100).min_similarity(0.2)
    }

    /// Builder: low sensitivity (higher threshold, fewer candidates).
    #[must_use]
    pub fn low_sensitivity(self) -> Self {
        self.threshold(0.7).semantic_k(20).min_similarity(0.5)
    }

    /// Validate parameters - FAIL FAST.
    pub fn validate(&self) -> GraphResult<()> {
        if self.semantic_k == 0 {
            return Err(GraphError::InvalidInput(
                "semantic_k must be > 0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.threshold) {
            return Err(GraphError::InvalidInput(format!(
                "threshold must be in [0, 1], got {}",
                self.threshold
            )));
        }
        if !(0.0..=1.0).contains(&self.min_similarity) {
            return Err(GraphError::InvalidInput(format!(
                "min_similarity must be in [0, 1], got {}",
                self.min_similarity
            )));
        }
        if !(0.0..=1.0).contains(&self.explicit_edge_weight) {
            return Err(GraphError::InvalidInput(format!(
                "explicit_edge_weight must be in [0, 1], got {}",
                self.explicit_edge_weight
            )));
        }
        Ok(())
    }
}

/// Internal candidate info for scoring.
struct CandidateInfo {
    semantic_similarity: f32,
    has_explicit_edge: bool,
    edge_weight: Option<f32>,
    edge_type: Option<ContradictionType>,
}

/// Detect contradictions for a given node.
///
/// Combines semantic similarity search with explicit CONTRADICTS edges
/// to find potentially conflicting knowledge.
///
/// # Algorithm
///
/// 1. Validate inputs (FAIL FAST)
/// 2. Semantic search for similar nodes (k candidates)
/// 3. BFS to find nodes with CONTRADICTS edges
/// 4. Combine and score contradictions
/// 5. Classify contradiction types
/// 6. Filter by threshold
/// 7. Sort by confidence descending
///
/// # Arguments
///
/// * `index` - FAISS GPU index for semantic search
/// * `storage` - Graph storage backend
/// * `node_id` - Node to check for contradictions (UUID)
/// * `node_embedding` - Embedding as raw f32 slice
/// * `params` - Detection parameters
/// * `metadata` - Optional metadata provider for filtering
///
/// # Returns
///
/// Vector of contradictions above threshold, sorted by confidence descending.
///
/// # Errors
///
/// * `GraphError::InvalidInput` - Invalid parameters (FAIL FAST)
/// * `GraphError::FaissSearchFailed` - FAISS search failed
/// * `GraphError::Storage` - Storage access failed
///
/// # Example
///
/// ```ignore
/// use context_graph_graph::contradiction::{contradiction_detect, ContradictionParams};
///
/// let params = ContradictionParams::default().high_sensitivity();
/// let results = contradiction_detect(
///     &index,
///     &storage,
///     node_id,
///     &embedding,
///     params,
///     None,
/// )?;
///
/// for result in results {
///     println!("Contradiction: {} (confidence: {})",
///         result.contradicting_node_id,
///         result.confidence);
/// }
/// ```
pub fn contradiction_detect<M: NodeMetadataProvider>(
    index: &FaissGpuIndex,
    storage: &GraphStorage,
    node_id: Uuid,
    node_embedding: &[f32],
    params: ContradictionParams,
    metadata: Option<&M>,
) -> GraphResult<Vec<ContradictionResult>> {
    // FAIL FAST validation
    if node_embedding.is_empty() {
        return Err(GraphError::InvalidInput(
            "node_embedding cannot be empty".to_string(),
        ));
    }

    params.validate()?;

    let mut candidates: HashMap<Uuid, CandidateInfo> = HashMap::new();

    // Step 1: Semantic search for similar nodes
    let filters = SearchFilters::new().with_min_similarity(params.min_similarity);

    let semantic_results = semantic_search(
        index,
        node_embedding,
        params.semantic_k,
        Some(filters),
        metadata,
    )?;

    for item in semantic_results.items.iter() {
        // Skip if no node_id resolved or if it's the query node itself
        if let Some(item_node_id) = item.node_id {
            if item_node_id != node_id {
                candidates.insert(
                    item_node_id,
                    CandidateInfo {
                        semantic_similarity: item.similarity,
                        has_explicit_edge: false,
                        edge_weight: None,
                        edge_type: None,
                    },
                );
            }
        }
    }

    // Step 2: BFS to find CONTRADICTS edges
    // Convert UUID to i64 for storage operations
    let node_id_i64 = uuid_to_i64(&node_id);

    let bfs_params = BfsParams::default()
        .max_depth(params.graph_depth)
        .edge_types(vec![EdgeType::Contradicts]);

    let bfs_result = bfs_traverse(storage, node_id_i64, bfs_params)?;

    for edge in bfs_result.edges.iter() {
        if edge.edge_type == EdgeType::Contradicts {
            let target = if edge.source == node_id {
                edge.target
            } else {
                edge.source
            };

            candidates
                .entry(target)
                .and_modify(|info| {
                    info.has_explicit_edge = true;
                    info.edge_weight = Some(edge.weight);
                    info.edge_type = Some(infer_contradiction_type_from_edge(edge));
                })
                .or_insert(CandidateInfo {
                    semantic_similarity: 0.0,
                    has_explicit_edge: true,
                    edge_weight: Some(edge.weight),
                    edge_type: Some(infer_contradiction_type_from_edge(edge)),
                });
        }
    }

    // Step 3: Score and classify contradictions
    let mut results: Vec<ContradictionResult> = Vec::with_capacity(candidates.len());

    for (candidate_id, info) in candidates {
        let confidence = compute_confidence(&info, &params);

        if confidence >= params.threshold {
            let contradiction_type = info
                .edge_type
                .unwrap_or_else(|| infer_type_from_similarity(info.semantic_similarity));

            results.push(ContradictionResult {
                contradicting_node_id: candidate_id,
                contradiction_type,
                confidence,
                semantic_similarity: info.semantic_similarity,
                edge_weight: info.edge_weight,
                has_explicit_edge: info.has_explicit_edge,
                evidence: Vec::new(),
            });
        }
    }

    // Sort by confidence descending
    results.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    info!(
        node_id = %node_id,
        candidates_found = results.len(),
        threshold = params.threshold,
        "Contradiction detection complete"
    );

    Ok(results)
}

/// Check for contradiction between two specific nodes.
///
/// Looks for explicit CONTRADICTS edge between the nodes.
///
/// # Arguments
///
/// * `storage` - Graph storage backend
/// * `node_a` - First node UUID
/// * `node_b` - Second node UUID
///
/// # Returns
///
/// * `Some(ContradictionResult)` if explicit contradiction exists
/// * `None` if no explicit contradiction edge found
pub fn check_contradiction(
    storage: &GraphStorage,
    node_a: Uuid,
    node_b: Uuid,
) -> GraphResult<Option<ContradictionResult>> {
    let node_a_i64 = uuid_to_i64(&node_a);
    let edges = storage.get_outgoing_edges(node_a_i64)?;

    for edge in edges {
        if edge.edge_type == EdgeType::Contradicts && edge.target == node_b {
            return Ok(Some(ContradictionResult {
                contradicting_node_id: node_b,
                contradiction_type: infer_contradiction_type_from_edge(&edge),
                confidence: edge.confidence,
                semantic_similarity: 0.0,
                edge_weight: Some(edge.weight),
                has_explicit_edge: true,
                evidence: Vec::new(),
            }));
        }
    }

    Ok(None)
}

/// Mark two nodes as contradicting.
///
/// Creates bidirectional CONTRADICTS edges with inhibitory-heavy NT modulation.
///
/// # Arguments
///
/// * `storage` - Mutable graph storage backend
/// * `node_a` - First node UUID
/// * `node_b` - Second node UUID
/// * `contradiction_type` - Type of contradiction
/// * `confidence` - Confidence score [0, 1]
///
/// # Errors
///
/// * `GraphError::InvalidInput` - Self-contradiction or invalid confidence (FAIL FAST)
/// * `GraphError::Storage` - Storage write failed
pub fn mark_contradiction(
    storage: &GraphStorage,
    node_a: Uuid,
    node_b: Uuid,
    _contradiction_type: ContradictionType,
    confidence: f32,
) -> GraphResult<()> {
    // FAIL FAST validation
    if node_a == node_b {
        return Err(GraphError::InvalidInput(
            "Cannot create self-contradiction".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&confidence) {
        return Err(GraphError::InvalidInput(format!(
            "Confidence must be in [0, 1], got {}",
            confidence
        )));
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Create bidirectional contradiction edges using the factory method
    let edge_a_to_b = GraphEdge {
        id: generate_edge_id(),
        source: node_a,
        target: node_b,
        edge_type: EdgeType::Contradicts,
        weight: EdgeType::Contradicts.default_weight(),
        confidence,
        domain: Domain::General,
        // Inhibitory-heavy NT profile per M04-T26
        neurotransmitter_weights: NeurotransmitterWeights {
            excitatory: 0.2,
            inhibitory: 0.7,
            modulatory: 0.1,
        },
        is_amortized_shortcut: false,
        steering_reward: 0.5,
        traversal_count: 0,
        created_at: now,
        last_traversed_at: 0,
    };

    let edge_b_to_a = GraphEdge {
        id: generate_edge_id(),
        source: node_b,
        target: node_a,
        edge_type: EdgeType::Contradicts,
        weight: EdgeType::Contradicts.default_weight(),
        confidence,
        domain: Domain::General,
        neurotransmitter_weights: NeurotransmitterWeights {
            excitatory: 0.2,
            inhibitory: 0.7,
            modulatory: 0.1,
        },
        is_amortized_shortcut: false,
        steering_reward: 0.5,
        traversal_count: 0,
        created_at: now,
        last_traversed_at: 0,
    };

    storage.put_edge(&edge_a_to_b)?;
    storage.put_edge(&edge_b_to_a)?;

    info!(
        node_a = %node_a,
        node_b = %node_b,
        confidence = confidence,
        "Marked contradiction between nodes"
    );

    Ok(())
}

/// Get all contradictions for a node from storage.
///
/// Returns all explicit CONTRADICTS edges from the given node.
///
/// # Arguments
///
/// * `storage` - Graph storage backend
/// * `node_id` - Node UUID to get contradictions for
pub fn get_contradictions(
    storage: &GraphStorage,
    node_id: Uuid,
) -> GraphResult<Vec<ContradictionResult>> {
    let node_id_i64 = uuid_to_i64(&node_id);
    let edges = storage.get_outgoing_edges(node_id_i64)?;

    let results: Vec<ContradictionResult> = edges
        .iter()
        .filter(|e| e.edge_type == EdgeType::Contradicts)
        .map(|e| ContradictionResult {
            contradicting_node_id: e.target,
            contradiction_type: infer_contradiction_type_from_edge(e),
            confidence: e.confidence,
            semantic_similarity: 0.0,
            edge_weight: Some(e.weight),
            has_explicit_edge: true,
            evidence: Vec::new(),
        })
        .collect();

    Ok(results)
}

// ========== Helper Functions ==========

/// Convert UUID to i64 for storage key operations.
#[inline]
fn uuid_to_i64(uuid: &Uuid) -> i64 {
    let bytes = uuid.as_bytes();
    i64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Generate unique edge ID from current time.
fn generate_edge_id() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

/// Compute confidence score from candidate info.
fn compute_confidence(info: &CandidateInfo, params: &ContradictionParams) -> f32 {
    let semantic_component = info.semantic_similarity * (1.0 - params.explicit_edge_weight);

    let edge_component = if info.has_explicit_edge {
        info.edge_weight.unwrap_or(0.5) * params.explicit_edge_weight
    } else {
        0.0
    };

    // Boost if both semantic and explicit evidence
    let combined = semantic_component + edge_component;
    let boost = if info.has_explicit_edge && info.semantic_similarity > 0.5 {
        1.2 // 20% boost for corroborating evidence
    } else {
        1.0
    };

    (combined * boost).clamp(0.0, 1.0)
}

/// Infer contradiction type from edge metadata.
fn infer_contradiction_type_from_edge(edge: &GraphEdge) -> ContradictionType {
    // Use domain hint for classification
    match edge.domain {
        Domain::Code => ContradictionType::LogicalInconsistency,
        _ => ContradictionType::DirectOpposition,
    }
}

/// Infer type from semantic similarity pattern.
fn infer_type_from_similarity(similarity: f32) -> ContradictionType {
    if similarity > 0.9 {
        ContradictionType::DirectOpposition
    } else if similarity > 0.7 {
        ContradictionType::LogicalInconsistency
    } else if similarity > 0.5 {
        ContradictionType::TemporalConflict
    } else {
        ContradictionType::CausalConflict
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== ContradictionParams Tests ==========

    #[test]
    fn test_contradiction_params_default() {
        let params = ContradictionParams::default();

        assert!((params.threshold - 0.5).abs() < 1e-6);
        assert_eq!(params.semantic_k, 50);
        assert!((params.min_similarity - 0.3).abs() < 1e-6);
        assert_eq!(params.graph_depth, 2);
        assert!((params.explicit_edge_weight - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_contradiction_params_builder() {
        let params = ContradictionParams::default()
            .threshold(0.7)
            .semantic_k(100)
            .min_similarity(0.4)
            .graph_depth(3);

        assert!((params.threshold - 0.7).abs() < 1e-6);
        assert_eq!(params.semantic_k, 100);
        assert!((params.min_similarity - 0.4).abs() < 1e-6);
        assert_eq!(params.graph_depth, 3);
    }

    #[test]
    fn test_high_sensitivity() {
        let params = ContradictionParams::default().high_sensitivity();

        assert!(params.threshold < 0.5);
        assert!(params.semantic_k > 50);
        assert!(params.min_similarity < 0.3);
    }

    #[test]
    fn test_low_sensitivity() {
        let params = ContradictionParams::default().low_sensitivity();

        assert!(params.threshold > 0.5);
        assert!(params.semantic_k < 50);
        assert!(params.min_similarity > 0.3);
    }

    #[test]
    fn test_threshold_clamping() {
        let params = ContradictionParams::default().threshold(1.5);
        assert!((params.threshold - 1.0).abs() < 1e-6);

        let params = ContradictionParams::default().threshold(-0.5);
        assert!((params.threshold - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_validate_params_valid() {
        let params = ContradictionParams::default();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_validate_params_zero_k() {
        let mut params = ContradictionParams::default();
        params.semantic_k = 0;

        let result = params.validate();
        assert!(result.is_err());
        match result {
            Err(GraphError::InvalidInput(msg)) => {
                assert!(msg.contains("semantic_k"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    // ========== ContradictionResult Tests ==========

    #[test]
    fn test_contradiction_result_severity() {
        let result = ContradictionResult {
            contradicting_node_id: Uuid::new_v4(),
            contradiction_type: ContradictionType::DirectOpposition,
            confidence: 0.8,
            semantic_similarity: 0.9,
            edge_weight: Some(0.85),
            has_explicit_edge: true,
            evidence: vec![],
        };

        // DirectOpposition has severity 1.0
        // severity = 0.8 * 1.0 = 0.8
        assert!((result.severity() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_is_high_confidence() {
        let result = ContradictionResult {
            contradicting_node_id: Uuid::new_v4(),
            contradiction_type: ContradictionType::DirectOpposition,
            confidence: 0.7,
            semantic_similarity: 0.0,
            edge_weight: None,
            has_explicit_edge: false,
            evidence: vec![],
        };

        assert!(result.is_high_confidence(0.5));
        assert!(result.is_high_confidence(0.7));
        assert!(!result.is_high_confidence(0.8));
    }

    // ========== Confidence Calculation Tests ==========

    #[test]
    fn test_compute_confidence_semantic_only() {
        let info = CandidateInfo {
            semantic_similarity: 0.8,
            has_explicit_edge: false,
            edge_weight: None,
            edge_type: None,
        };

        let params = ContradictionParams::default();
        let confidence = compute_confidence(&info, &params);

        // semantic_component = 0.8 * (1 - 0.6) = 0.8 * 0.4 = 0.32
        // No edge component, no boost
        assert!((confidence - 0.32).abs() < 1e-6);
    }

    #[test]
    fn test_compute_confidence_explicit_only() {
        let info = CandidateInfo {
            semantic_similarity: 0.0,
            has_explicit_edge: true,
            edge_weight: Some(0.8),
            edge_type: None,
        };

        let params = ContradictionParams::default();
        let confidence = compute_confidence(&info, &params);

        // edge_component = 0.8 * 0.6 = 0.48
        // No boost (semantic_similarity <= 0.5)
        assert!((confidence - 0.48).abs() < 1e-6);
    }

    #[test]
    fn test_compute_confidence_dual_evidence_boost() {
        let info = CandidateInfo {
            semantic_similarity: 0.8,
            has_explicit_edge: true,
            edge_weight: Some(0.8),
            edge_type: None,
        };

        let params = ContradictionParams::default();
        let confidence = compute_confidence(&info, &params);

        // semantic_component = 0.8 * 0.4 = 0.32
        // edge_component = 0.8 * 0.6 = 0.48
        // combined = 0.32 + 0.48 = 0.80
        // boost = 1.2 (dual evidence)
        // result = 0.80 * 1.2 = 0.96
        assert!((confidence - 0.96).abs() < 1e-6);
    }

    // ========== Contradiction Type Tests ==========

    #[test]
    fn test_infer_type_from_high_similarity() {
        let t = infer_type_from_similarity(0.95);
        assert_eq!(t, ContradictionType::DirectOpposition);
    }

    #[test]
    fn test_infer_type_from_medium_similarity() {
        let t = infer_type_from_similarity(0.75);
        assert_eq!(t, ContradictionType::LogicalInconsistency);
    }

    #[test]
    fn test_infer_type_from_low_similarity() {
        let t = infer_type_from_similarity(0.4);
        assert_eq!(t, ContradictionType::CausalConflict);
    }

    // ========== Helper Function Tests ==========

    #[test]
    fn test_uuid_to_i64_roundtrip() {
        // Create a UUID from an i64
        let original: i64 = 12345678;
        let uuid = Uuid::from_u64_pair(original as u64, 0);

        // Convert back
        let recovered = uuid_to_i64(&uuid);
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_generate_edge_id_unique() {
        let id1 = generate_edge_id();
        std::thread::sleep(std::time::Duration::from_nanos(1));
        let id2 = generate_edge_id();

        // IDs should be different (time-based)
        assert_ne!(id1, id2);
    }

    // ========== Contradiction Type Severity Tests ==========

    #[test]
    fn test_contradiction_type_severity_ordering() {
        assert!(ContradictionType::DirectOpposition.severity() > ContradictionType::LogicalInconsistency.severity());
        assert!(ContradictionType::LogicalInconsistency.severity() > ContradictionType::TemporalConflict.severity());
        assert!(ContradictionType::TemporalConflict.severity() > ContradictionType::CausalConflict.severity());
    }

    #[test]
    fn test_contradiction_type_severity_values() {
        assert!((ContradictionType::DirectOpposition.severity() - 1.0).abs() < 1e-6);
        assert!((ContradictionType::LogicalInconsistency.severity() - 0.8).abs() < 1e-6);
        assert!((ContradictionType::TemporalConflict.severity() - 0.7).abs() < 1e-6);
        assert!((ContradictionType::CausalConflict.severity() - 0.6).abs() < 1e-6);
    }
}
