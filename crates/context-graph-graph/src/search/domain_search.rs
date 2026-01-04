//! Domain-aware search with Marblestone neurotransmitter modulation.
//!
//! # CANONICAL FORMULA
//!
//! ```text
//! net_activation = excitatory - inhibitory + (modulatory * 0.5)
//! domain_bonus = 0.1 if node_domain == query_domain else 0.0
//! modulated_score = base_similarity * (1.0 + net_activation + domain_bonus)
//! ```
//!
//! # Constitution Reference
//!
//! - edge_model.nt_weights: Definition and formula
//! - edge_model.nt_weights.domain: Code|Legal|Medical|Creative|Research|General
//! - AP-001: Never unwrap() in prod - all errors properly typed
//! - AP-009: NaN/Infinity clamped to valid range

use crate::error::GraphResult;
use crate::index::FaissGpuIndex;
use crate::search::{
    semantic_search, Domain, NodeMetadataProvider, SearchFilters, SemanticSearchResultItem,
};

// Re-export from core - DO NOT REDEFINE
use context_graph_core::marblestone::NeurotransmitterWeights;

use tracing::{debug, warn};
use uuid::Uuid;

/// Domain bonus for matching domains (from constitution)
const DOMAIN_MATCH_BONUS: f32 = 0.1;

/// Over-fetch multiplier for re-ranking
const OVERFETCH_MULTIPLIER: usize = 3;

/// Result from domain-aware search with modulation metadata.
#[derive(Debug, Clone)]
pub struct DomainSearchResult {
    /// FAISS internal ID
    pub faiss_id: i64,

    /// Node UUID (if resolved via metadata provider)
    pub node_id: Option<Uuid>,

    /// Base similarity score (before modulation)
    pub base_similarity: f32,

    /// Modulated score after NT adjustment
    pub modulated_score: f32,

    /// L2 distance from query
    pub distance: f32,

    /// Rank in result set (0 = best match)
    pub rank: usize,

    /// Domain of the result node
    pub node_domain: Option<Domain>,

    /// Query domain used for modulation
    pub query_domain: Domain,

    /// Whether domain matched (bonus applied)
    pub domain_matched: bool,

    /// Net activation from NT weights (for debugging)
    pub net_activation: f32,
}

impl DomainSearchResult {
    /// Create from semantic search result item with modulation.
    pub fn from_semantic_item(
        item: &SemanticSearchResultItem,
        modulated_score: f32,
        net_activation: f32,
        query_domain: Domain,
    ) -> Self {
        let domain_matched = item.domain.map(|d| d == query_domain).unwrap_or(false);

        Self {
            faiss_id: item.faiss_id,
            node_id: item.node_id,
            base_similarity: item.similarity,
            modulated_score,
            distance: item.distance,
            rank: 0, // Will be set after re-ranking
            node_domain: item.domain,
            query_domain,
            domain_matched,
            net_activation,
        }
    }

    /// Get the boost/penalty applied (modulated - base).
    #[inline]
    pub fn modulation_delta(&self) -> f32 {
        self.modulated_score - self.base_similarity
    }

    /// Get boost ratio (modulated / base).
    #[inline]
    pub fn boost_ratio(&self) -> f32 {
        if self.base_similarity > 1e-6 {
            self.modulated_score / self.base_similarity
        } else {
            1.0
        }
    }
}

/// Container for domain search results with metadata.
#[derive(Debug, Clone)]
pub struct DomainSearchResults {
    /// Search results ordered by modulated score
    pub items: Vec<DomainSearchResult>,

    /// Number of candidates fetched before filtering
    pub candidates_fetched: usize,

    /// Number of results after filtering
    pub results_returned: usize,

    /// Query domain used
    pub query_domain: Domain,

    /// Search latency in microseconds
    pub latency_us: u64,
}

impl DomainSearchResults {
    /// Create empty results.
    pub fn empty(query_domain: Domain) -> Self {
        Self {
            items: Vec::new(),
            candidates_fetched: 0,
            results_returned: 0,
            query_domain,
            latency_us: 0,
        }
    }

    /// Iterate over results.
    pub fn iter(&self) -> impl Iterator<Item = &DomainSearchResult> {
        self.items.iter()
    }

    /// Check if results are empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get number of results.
    pub fn len(&self) -> usize {
        self.items.len()
    }
}

/// Compute net activation from NT weights using CANONICAL formula.
///
/// # CANONICAL FORMULA
///
/// ```text
/// net_activation = excitatory - inhibitory + (modulatory * 0.5)
/// ```
///
/// IMPORTANT: This is inline computation because NeurotransmitterWeights
/// does NOT have a net_activation() method.
#[inline]
fn compute_net_activation(nt: &NeurotransmitterWeights) -> f32 {
    nt.excitatory - nt.inhibitory + (nt.modulatory * 0.5)
}

/// Perform domain-aware search with neurotransmitter modulation.
///
/// Uses Marblestone-inspired NT modulation to adjust search relevance
/// based on the query domain. Over-fetches 3x candidates, applies
/// NT modulation, then re-ranks by modulated score.
///
/// # CANONICAL FORMULA
///
/// ```text
/// net_activation = excitatory - inhibitory + (modulatory * 0.5)
/// domain_bonus = 0.1 if node_domain == query_domain else 0.0
/// modulated_score = base_similarity * (1.0 + net_activation + domain_bonus)
/// ```
///
/// # Arguments
///
/// * `index` - FAISS GPU index (must be trained)
/// * `query` - Query embedding as f32 slice (1536 dimensions)
/// * `query_domain` - Domain for NT profile selection
/// * `k` - Number of results to return
/// * `filters` - Optional additional filters
/// * `metadata` - Metadata provider for node UUID/domain resolution
///
/// # Returns
///
/// * Top-k results ranked by modulated score
///
/// # Errors
///
/// * `GraphError::IndexNotTrained` - If FAISS index not trained
/// * `GraphError::DimensionMismatch` - If query dimension wrong
/// * `GraphError::FaissSearchFailed` - If FAISS search fails
/// * `GraphError::InvalidConfig` - If filters invalid
///
/// # Performance
///
/// Target: <10ms for k=10 on 10M vectors
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_graph::search::domain_search::{domain_aware_search, DomainSearchResults};
/// use context_graph_graph::search::Domain;
///
/// let results = domain_aware_search(
///     &index,
///     &query_embedding,
///     Domain::Code,
///     10,
///     None,
///     Some(&storage),
/// )?;
///
/// for result in results.iter() {
///     println!("Node {:?} base: {:.3} modulated: {:.3} (delta: {:+.3})",
///         result.node_id,
///         result.base_similarity,
///         result.modulated_score,
///         result.modulation_delta()
///     );
/// }
/// ```
#[tracing::instrument(skip(index, query, metadata), fields(domain = ?query_domain, k = k))]
pub fn domain_aware_search<M: NodeMetadataProvider>(
    index: &FaissGpuIndex,
    query: &[f32],
    query_domain: Domain,
    k: usize,
    filters: Option<SearchFilters>,
    metadata: Option<&M>,
) -> GraphResult<DomainSearchResults> {
    let start = std::time::Instant::now();

    // Validate k
    if k == 0 {
        warn!("domain_aware_search called with k=0, returning empty results");
        return Ok(DomainSearchResults::empty(query_domain));
    }

    // Over-fetch 3x candidates for re-ranking
    let fetch_k = k.saturating_mul(OVERFETCH_MULTIPLIER);
    debug!(fetch_k, "Over-fetching candidates for re-ranking");

    // Get base semantic results
    let semantic_results = semantic_search(index, query, fetch_k, filters, metadata)?;

    if semantic_results.items.is_empty() {
        debug!("No semantic search results found");
        return Ok(DomainSearchResults::empty(query_domain));
    }

    // Get domain-specific NT profile
    let domain_nt = NeurotransmitterWeights::for_domain(query_domain);
    // CANONICAL FORMULA: net_activation computed inline (no method on NeurotransmitterWeights)
    let base_net_activation = compute_net_activation(&domain_nt);
    debug!(
        net_activation = base_net_activation,
        "Using NT profile for domain"
    );

    // Apply modulation to each result
    let mut modulated_results: Vec<DomainSearchResult> =
        Vec::with_capacity(semantic_results.items.len());

    for item in &semantic_results.items {
        // Calculate domain bonus
        let domain_bonus = match item.domain {
            Some(node_domain) if node_domain == query_domain => DOMAIN_MATCH_BONUS,
            _ => 0.0,
        };

        // CANONICAL FORMULA: modulated_score = base * (1.0 + net_activation + domain_bonus)
        let modulated_score = item.similarity * (1.0 + base_net_activation + domain_bonus);

        // Clamp to [0.0, 1.0] per AP-009
        let modulated_score = modulated_score.clamp(0.0, 1.0);

        modulated_results.push(DomainSearchResult::from_semantic_item(
            item,
            modulated_score,
            base_net_activation,
            query_domain,
        ));
    }

    // Re-rank by modulated score (descending)
    modulated_results.sort_by(|a, b| {
        b.modulated_score
            .partial_cmp(&a.modulated_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Update ranks and truncate to k
    for (i, result) in modulated_results.iter_mut().enumerate() {
        result.rank = i;
    }

    let candidates_fetched = modulated_results.len();
    modulated_results.truncate(k);
    let results_returned = modulated_results.len();

    let latency = start.elapsed();
    debug!(
        latency_us = latency.as_micros(),
        candidates = candidates_fetched,
        returned = results_returned,
        "Domain search complete"
    );

    Ok(DomainSearchResults {
        items: modulated_results,
        candidates_fetched,
        results_returned,
        query_domain,
        latency_us: latency.as_micros() as u64,
    })
}

/// Get expected boost ratio for a domain (matching nodes).
///
/// Returns the expected modulation multiplier for nodes matching the query domain.
/// Useful for testing and validation.
#[inline]
pub fn expected_domain_boost(domain: Domain) -> f32 {
    let nt = NeurotransmitterWeights::for_domain(domain);
    let net_activation = compute_net_activation(&nt);
    1.0 + net_activation + DOMAIN_MATCH_BONUS
}

/// Get NT profile summary for a domain.
///
/// Returns a human-readable summary of the NT profile for debugging.
pub fn domain_nt_summary(domain: Domain) -> String {
    let nt = NeurotransmitterWeights::for_domain(domain);
    let net_activation = compute_net_activation(&nt);
    format!(
        "{:?}: exc={:.2} inh={:.2} mod={:.2} net={:+.3} boost={:.2}x",
        domain,
        nt.excitatory,
        nt.inhibitory,
        nt.modulatory,
        net_activation,
        expected_domain_boost(domain)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Unit Tests (No GPU Required) ==========

    #[test]
    fn test_canonical_formula_net_activation() {
        // Verify net_activation formula: excitatory - inhibitory + (modulatory * 0.5)
        // These values come from NeurotransmitterWeights::for_domain() in context-graph-core

        // Code: e=0.6, i=0.3, m=0.4 -> 0.6 - 0.3 + (0.4 * 0.5) = 0.5
        let code_nt = NeurotransmitterWeights::for_domain(Domain::Code);
        let net = compute_net_activation(&code_nt);
        assert!(
            (net - 0.5).abs() < 0.01,
            "Code net_activation should be 0.5, got {}",
            net
        );

        // Creative: e=0.8, i=0.1, m=0.6 -> 0.8 - 0.1 + (0.6 * 0.5) = 0.7 + 0.3 = 1.0
        let creative_nt = NeurotransmitterWeights::for_domain(Domain::Creative);
        let net = compute_net_activation(&creative_nt);
        assert!(
            (net - 1.0).abs() < 0.01,
            "Creative net_activation should be 1.0, got {}",
            net
        );

        // General: e=0.5, i=0.2, m=0.3 -> 0.5 - 0.2 + (0.3 * 0.5) = 0.3 + 0.15 = 0.45
        let general_nt = NeurotransmitterWeights::for_domain(Domain::General);
        let net = compute_net_activation(&general_nt);
        assert!(
            (net - 0.45).abs() < 0.01,
            "General net_activation should be 0.45, got {}",
            net
        );

        // Legal: e=0.4, i=0.4, m=0.2 -> 0.4 - 0.4 + (0.2 * 0.5) = 0 + 0.1 = 0.1
        let legal_nt = NeurotransmitterWeights::for_domain(Domain::Legal);
        let net = compute_net_activation(&legal_nt);
        assert!(
            (net - 0.1).abs() < 0.01,
            "Legal net_activation should be 0.1, got {}",
            net
        );

        // Medical: e=0.5, i=0.3, m=0.5 -> 0.5 - 0.3 + (0.5 * 0.5) = 0.2 + 0.25 = 0.45
        let medical_nt = NeurotransmitterWeights::for_domain(Domain::Medical);
        let net = compute_net_activation(&medical_nt);
        assert!(
            (net - 0.45).abs() < 0.01,
            "Medical net_activation should be 0.45, got {}",
            net
        );

        // Research: e=0.6, i=0.2, m=0.5 -> 0.6 - 0.2 + (0.5 * 0.5) = 0.4 + 0.25 = 0.65
        let research_nt = NeurotransmitterWeights::for_domain(Domain::Research);
        let net = compute_net_activation(&research_nt);
        assert!(
            (net - 0.65).abs() < 0.01,
            "Research net_activation should be 0.65, got {}",
            net
        );
    }

    #[test]
    fn test_modulation_formula_application() {
        // Test the modulation formula: base * (1.0 + net_activation + domain_bonus)

        let base = 0.8f32;
        let net_activation = 0.5f32;
        let domain_bonus = 0.1f32;

        let modulated = base * (1.0 + net_activation + domain_bonus);
        let modulated = modulated.clamp(0.0, 1.0);

        // 0.8 * 1.6 = 1.28, clamped to 1.0
        assert!(
            (modulated - 1.0).abs() < 1e-6,
            "Modulated should be clamped to 1.0"
        );
    }

    #[test]
    fn test_modulation_no_domain_bonus() {
        let base = 0.5f32;
        let net_activation = 0.3f32;
        let domain_bonus = 0.0f32; // No match

        let modulated = base * (1.0 + net_activation + domain_bonus);
        // 0.5 * 1.3 = 0.65
        assert!((modulated - 0.65).abs() < 1e-6);
    }

    #[test]
    fn test_modulation_clamping_high() {
        // Test that high modulation values get clamped to 1.0
        let base = 0.9f32;
        let net_activation = 1.0f32; // Creative domain
        let domain_bonus = 0.1f32;

        let modulated = base * (1.0 + net_activation + domain_bonus);
        // 0.9 * 2.1 = 1.89, clamped to 1.0
        let clamped = modulated.clamp(0.0, 1.0);
        assert!(
            (clamped - 1.0).abs() < 1e-6,
            "Should be clamped to 1.0, got {}",
            clamped
        );
    }

    #[test]
    fn test_domain_search_result_from_semantic_item() {
        let item = SemanticSearchResultItem {
            faiss_id: 42,
            node_id: Some(Uuid::from_u128(1)),
            distance: 0.4,
            similarity: 0.7,
            domain: Some(Domain::Code),
            relevance_score: None,
        };

        let result = DomainSearchResult::from_semantic_item(
            &item,
            0.9, // Modulated up
            0.5, // net_activation
            Domain::Code,
        );

        assert_eq!(result.faiss_id, 42);
        assert_eq!(result.base_similarity, 0.7);
        assert_eq!(result.modulated_score, 0.9);
        assert!(result.domain_matched);
        assert!((result.modulation_delta() - 0.2).abs() < 1e-6);
        assert!((result.boost_ratio() - (0.9 / 0.7)).abs() < 1e-6);
    }

    #[test]
    fn test_domain_search_result_no_domain_match() {
        let item = SemanticSearchResultItem {
            faiss_id: 42,
            node_id: None,
            distance: 0.4,
            similarity: 0.7,
            domain: Some(Domain::Legal),
            relevance_score: None,
        };

        let result = DomainSearchResult::from_semantic_item(
            &item,
            0.8,
            0.5,
            Domain::Code, // Different from Legal
        );

        assert!(!result.domain_matched);
    }

    #[test]
    fn test_expected_domain_boost() {
        // Code: net_activation = 0.5
        // boost = 1.0 + 0.5 + 0.1 = 1.6
        let code_boost = expected_domain_boost(Domain::Code);
        assert!(
            (code_boost - 1.6).abs() < 0.01,
            "Code boost should be 1.6, got {}",
            code_boost
        );

        // Creative: net_activation = 1.0
        // boost = 1.0 + 1.0 + 0.1 = 2.1
        let creative_boost = expected_domain_boost(Domain::Creative);
        assert!(
            (creative_boost - 2.1).abs() < 0.01,
            "Creative boost should be 2.1, got {}",
            creative_boost
        );

        // Legal: net_activation = 0.1
        // boost = 1.0 + 0.1 + 0.1 = 1.2
        let legal_boost = expected_domain_boost(Domain::Legal);
        assert!(
            (legal_boost - 1.2).abs() < 0.01,
            "Legal boost should be 1.2, got {}",
            legal_boost
        );

        // General: net_activation = 0.45
        // boost = 1.0 + 0.45 + 0.1 = 1.55
        let general_boost = expected_domain_boost(Domain::General);
        assert!(
            (general_boost - 1.55).abs() < 0.01,
            "General boost should be 1.55, got {}",
            general_boost
        );

        // General should have lower boost than Code
        assert!(
            general_boost < code_boost,
            "General boost should be lower than Code"
        );
    }

    #[test]
    fn test_domain_nt_summary() {
        let summary = domain_nt_summary(Domain::Code);
        assert!(summary.contains("Code"));
        assert!(summary.contains("exc="));
        assert!(summary.contains("net="));
        assert!(summary.contains("boost="));
    }

    #[test]
    fn test_domain_search_results_empty() {
        let results = DomainSearchResults::empty(Domain::Code);
        assert!(results.is_empty());
        assert_eq!(results.len(), 0);
        assert_eq!(results.query_domain, Domain::Code);
    }

    #[test]
    fn test_boost_ratio_zero_base() {
        let item = SemanticSearchResultItem {
            faiss_id: 1,
            node_id: None,
            distance: 10.0,
            similarity: 0.0, // Zero base
            domain: None,
            relevance_score: None,
        };

        let result = DomainSearchResult::from_semantic_item(&item, 0.0, 0.0, Domain::General);

        // Should return 1.0 to avoid division by zero
        assert_eq!(result.boost_ratio(), 1.0);
    }

    #[test]
    fn test_domain_search_results_iter() {
        let item1 = SemanticSearchResultItem {
            faiss_id: 1,
            node_id: None,
            distance: 0.1,
            similarity: 0.9,
            domain: Some(Domain::Code),
            relevance_score: None,
        };
        let item2 = SemanticSearchResultItem {
            faiss_id: 2,
            node_id: None,
            distance: 0.2,
            similarity: 0.8,
            domain: Some(Domain::Code),
            relevance_score: None,
        };

        let results = DomainSearchResults {
            items: vec![
                DomainSearchResult::from_semantic_item(&item1, 1.0, 0.5, Domain::Code),
                DomainSearchResult::from_semantic_item(&item2, 0.9, 0.5, Domain::Code),
            ],
            candidates_fetched: 6,
            results_returned: 2,
            query_domain: Domain::Code,
            latency_us: 100,
        };

        assert_eq!(results.len(), 2);
        assert!(!results.is_empty());
        assert_eq!(results.iter().count(), 2);
    }

    #[test]
    fn test_all_domains_have_valid_boost() {
        // Ensure all domains produce valid boost ratios
        for domain in Domain::all() {
            let boost = expected_domain_boost(domain);
            assert!(boost > 1.0, "{:?} should have boost > 1.0, got {}", domain, boost);
            assert!(boost < 3.0, "{:?} should have boost < 3.0, got {}", domain, boost);
        }
    }

    #[test]
    fn test_modulation_delta_positive_when_boosted() {
        let item = SemanticSearchResultItem {
            faiss_id: 1,
            node_id: None,
            distance: 0.1,
            similarity: 0.5,
            domain: Some(Domain::Code),
            relevance_score: None,
        };

        // With boost
        let result = DomainSearchResult::from_semantic_item(&item, 0.8, 0.5, Domain::Code);
        assert!(
            result.modulation_delta() > 0.0,
            "Modulation delta should be positive when boosted"
        );
    }

    // ========== Integration Tests (GPU Required) ==========

    #[test]
    #[ignore] // Requires GPU
    #[cfg(feature = "faiss-gpu")]
    fn test_domain_aware_search_with_real_index() {
        // This test requires:
        // 1. Real FAISS GPU index (trained)
        // 2. Real metadata provider implementation
        // 3. Real embeddings from context-graph-embeddings
        //
        // See Full State Verification section below for implementation
        todo!("Implement with real FAISS index and storage")
    }

    #[test]
    #[ignore] // Requires GPU
    #[cfg(feature = "faiss-gpu")]
    fn test_domain_search_reranks_correctly() {
        // Verify that domain-matching nodes get boosted above non-matching
        // even if their base similarity is slightly lower
        todo!("Implement with real FAISS index")
    }

    #[test]
    #[ignore] // Requires GPU
    #[cfg(feature = "faiss-gpu")]
    fn test_domain_search_performance_10ms() {
        // Verify <10ms latency for k=10 on 10M vectors
        todo!("Implement performance test")
    }
}
