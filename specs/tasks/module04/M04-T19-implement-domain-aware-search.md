---
id: "M04-T19"
title: "Implement Domain-Aware Search (Marblestone)"
description: |
  Implement domain_aware_search(query, domain, k) with neurotransmitter modulation.
  Algorithm:
  1. FAISS k-NN search fetching 3x candidates
  2. Apply NeurotransmitterWeights modulation per domain
  3. Re-rank by modulated score
  4. Return top-k results

  CANONICAL FORMULA for modulation:
  net_activation = excitatory - inhibitory + (modulatory * 0.5)
  modulated_score = base_similarity * (1.0 + net_activation + domain_bonus)

  Performance: <10ms for k=10 on 10M vectors.
layer: "surface"
status: "complete"
priority: "critical"
estimated_hours: 3
sequence: 27
depends_on:
  - "M04-T14a"  # NT weight validation (COMPLETE)
  - "M04-T15"   # GraphEdge storage (COMPLETE)
  - "M04-T18"   # Semantic search (COMPLETE)
spec_refs:
  - "TECH-GRAPH-004 Section 8"
  - "REQ-KG-065"
  - "constitution.yaml: edge_model.nt_weights"
files_to_create:
  - path: "crates/context-graph-graph/src/search/domain_search.rs"
    description: "Domain-aware search with NT modulation"
files_to_modify:
  - path: "crates/context-graph-graph/src/search/mod.rs"
    description: "Add domain_search module export"
  - path: "crates/context-graph-graph/src/marblestone/mod.rs"
    description: "Add domain_search re-export"
test_file: "crates/context-graph-graph/src/search/domain_search.rs"
constitution_refs:
  - "edge_model.nt_weights: w_eff = base × (1 + excitatory - inhibitory + 0.5×modulatory)"
  - "edge_model.nt_weights.domain: Code|Legal|Medical|Creative|Research|General"
  - "AP-001: Never unwrap() in prod - all errors properly typed"
  - "AP-009: NaN/Infinity clamped to valid range"
---

## CRITICAL POLICIES

### NO BACKWARDS COMPATIBILITY
- System MUST work or FAIL FAST with robust error logging
- All errors return `GraphResult<T>` - NO `unwrap()` or `expect()`
- Use `GraphError::InvalidConfig`, `GraphError::SearchFailed`, etc.
- Log all failures with context via tracing crate

### NO MOCK DATA IN TESTS
- ALL tests MUST use REAL FAISS GPU index
- ALL tests MUST use REAL GraphStorage (RocksDB)
- ALL tests MUST use REAL NeurotransmitterWeights from context-graph-core
- Mark GPU tests with `#[ignore]` and `#[cfg(feature = "cuda")]`

---

## Current Codebase State (VERIFIED 2026-01-04)

### EXISTING Types and Locations (USE THESE EXACTLY)

#### Types from `context-graph-graph` crate:

| Type | Location | Import Path |
|------|----------|-------------|
| `SemanticSearchResult` | `src/search/result.rs` | `crate::search::SemanticSearchResult` |
| `SemanticSearchResultItem` | `src/search/result.rs` | `crate::search::SemanticSearchResultItem` |
| `SearchFilters` | `src/search/filters.rs` | `crate::search::SearchFilters` |
| `Domain` | re-export from core | `crate::search::Domain` or `crate::storage::edges::Domain` |
| `FaissGpuIndex` | `src/index/gpu_index.rs` | `crate::index::FaissGpuIndex` |
| `GraphError` | `src/error.rs` | `crate::error::GraphError` |
| `GraphResult<T>` | `src/error.rs` | `crate::error::GraphResult` |
| `NodeMetadataProvider` | `src/search/mod.rs` | `crate::search::NodeMetadataProvider` |

#### Types from `context-graph-core` crate (DO NOT REDEFINE):

| Type | Location | Import Path |
|------|----------|-------------|
| `Domain` | `context_graph_core::marblestone::Domain` | enum with Code, Legal, Medical, Creative, Research, General |
| `NeurotransmitterWeights` | `context_graph_core::marblestone::NeurotransmitterWeights` | struct with excitatory, inhibitory, modulatory |
| `NodeId` | `context_graph_core::types::NodeId` | `type NodeId = Uuid` |

### EXISTING Functions (USE THESE):

```rust
// In crates/context-graph-graph/src/search/mod.rs (lines 156-195)
pub fn semantic_search<M: NodeMetadataProvider>(
    index: &FaissGpuIndex,
    query: &[f32],           // NOT Vector1536, just &[f32] slice
    k: usize,
    filters: Option<SearchFilters>,
    metadata: Option<&M>,
) -> GraphResult<SemanticSearchResult>

// In context_graph_core::marblestone::NeurotransmitterWeights
impl NeurotransmitterWeights {
    pub fn for_domain(domain: Domain) -> Self;
    pub fn compute_effective_weight(&self, base_weight: f32) -> f32;
    pub fn validate(&self) -> bool;
}

// IMPORTANT: NeurotransmitterWeights does NOT have a net_activation() method!
// Implementation uses compute_net_activation() helper function:
//   fn compute_net_activation(nt: &NeurotransmitterWeights) -> f32 {
//       nt.excitatory - nt.inhibitory + (nt.modulatory * 0.5)
//   }
```

### EXISTING SemanticSearchResultItem Fields (lines 18-36 in result.rs):

```rust
pub struct SemanticSearchResultItem {
    pub faiss_id: i64,              // NOT NodeId - use i64 for FAISS internal ID
    pub node_id: Option<Uuid>,      // Optional - resolved via NodeMetadataProvider
    pub distance: f32,              // L2 distance from query
    pub similarity: f32,            // Converted to similarity score (1.0 / (1.0 + distance))
    pub domain: Option<Domain>,     // Optional - resolved via NodeMetadataProvider
    pub relevance_score: Option<f32>, // For future use
}
```

### NodeMetadataProvider Trait (MUST IMPLEMENT for storage):

```rust
// In crates/context-graph-graph/src/search/mod.rs (lines 71-97)
pub trait NodeMetadataProvider {
    fn get_node_uuid(&self, faiss_id: i64) -> Option<Uuid>;
    fn get_node_domain(&self, faiss_id: i64) -> Option<Domain>;
    fn get_node_uuids(&self, faiss_ids: &[i64]) -> Vec<Option<Uuid>>;
    fn get_node_domains(&self, faiss_ids: &[i64]) -> Vec<Option<Domain>>;
}
```

---

## Context

Domain-aware search implements the Marblestone brain-inspired modulation system for context-sensitive retrieval. Different knowledge domains (Code, Legal, Medical, Creative, Research, General) have distinct neurotransmitter profiles that modulate edge weights and search relevance.

### CANONICAL FORMULA (from constitution.yaml):
```
net_activation = excitatory - inhibitory + (modulatory * 0.5)
modulated_score = base_similarity * (1.0 + net_activation + domain_bonus)
```

Where `domain_bonus = 0.1` if node_domain == query_domain, else 0.0

---

## Scope

### In Scope
- `domain_aware_search()` function in `src/search/domain_search.rs`
- `DomainSearchResult` struct with base and modulated scores
- Over-fetch 3x candidates for re-ranking
- Apply domain-specific NT profiles
- Re-rank by modulated score

### Out of Scope
- NT profile learning/training
- Cross-domain fusion
- Hierarchical domain matching
- CUDA-accelerated NT modulation

---

## Definition of Done

### File: `crates/context-graph-graph/src/search/domain_search.rs`

```rust
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

use crate::error::{GraphError, GraphResult};
use crate::index::FaissGpuIndex;
use crate::search::{semantic_search, NodeMetadataProvider, SemanticSearchResult, SemanticSearchResultItem, SearchFilters, Domain};

// Re-export from core - DO NOT REDEFINE
use context_graph_core::marblestone::NeurotransmitterWeights;

use uuid::Uuid;
use tracing::{debug, warn, instrument};

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
            rank: 0,  // Will be set after re-ranking
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
/// * `GraphError::SearchFailed` - If FAISS search fails
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
#[instrument(skip(index, query, metadata), fields(domain = ?query_domain, k = k))]
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
    debug!(net_activation = base_net_activation, "Using NT profile for domain");

    // Apply modulation to each result
    let mut modulated_results: Vec<DomainSearchResult> = Vec::with_capacity(semantic_results.items.len());

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
    debug!(latency_us = latency.as_micros(), candidates = candidates_fetched, returned = results_returned, "Domain search complete");

    Ok(DomainSearchResults {
        items: modulated_results,
        candidates_fetched,
        results_returned,
        query_domain,
        latency_us: latency.as_micros() as u64,
    })
}

/// Compute net activation from NT weights.
///
/// CANONICAL FORMULA: net_activation = excitatory - inhibitory + (modulatory * 0.5)
/// NOTE: NeurotransmitterWeights does NOT have a net_activation() method.
#[inline]
fn compute_net_activation(nt: &NeurotransmitterWeights) -> f32 {
    nt.excitatory - nt.inhibitory + (nt.modulatory * 0.5)
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
        assert!((net - 0.5).abs() < 0.01, "Code net_activation should be 0.5, got {}", net);

        // Creative: e=0.8, i=0.1, m=0.6 -> 0.8 - 0.1 + (0.6 * 0.5) = 1.0
        let creative_nt = NeurotransmitterWeights::for_domain(Domain::Creative);
        let net = compute_net_activation(&creative_nt);
        assert!((net - 1.0).abs() < 0.01, "Creative net_activation should be 1.0, got {}", net);

        // General: e=0.5, i=0.2, m=0.3 -> 0.5 - 0.2 + (0.3 * 0.5) = 0.45
        let general_nt = NeurotransmitterWeights::for_domain(Domain::General);
        let net = compute_net_activation(&general_nt);
        assert!((net - 0.45).abs() < 0.01, "General net_activation should be 0.45, got {}", net);

        // Legal: e=0.4, i=0.4, m=0.2 -> 0.4 - 0.4 + (0.2 * 0.5) = 0.1
        let legal_nt = NeurotransmitterWeights::for_domain(Domain::Legal);
        let net = compute_net_activation(&legal_nt);
        assert!((net - 0.1).abs() < 0.01, "Legal net_activation should be 0.1, got {}", net);

        // Medical: e=0.5, i=0.3, m=0.5 -> 0.5 - 0.3 + (0.5 * 0.5) = 0.45
        let medical_nt = NeurotransmitterWeights::for_domain(Domain::Medical);
        let net = compute_net_activation(&medical_nt);
        assert!((net - 0.45).abs() < 0.01, "Medical net_activation should be 0.45, got {}", net);

        // Research: e=0.6, i=0.2, m=0.5 -> 0.6 - 0.2 + (0.5 * 0.5) = 0.65
        let research_nt = NeurotransmitterWeights::for_domain(Domain::Research);
        let net = compute_net_activation(&research_nt);
        assert!((net - 0.65).abs() < 0.01, "Research net_activation should be 0.65, got {}", net);
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
        assert!((modulated - 1.0).abs() < 1e-6, "Modulated should be clamped to 1.0");
    }

    #[test]
    fn test_modulation_no_domain_bonus() {
        let base = 0.5f32;
        let net_activation = 0.3f32;
        let domain_bonus = 0.0f32;  // No match

        let modulated = base * (1.0 + net_activation + domain_bonus);
        // 0.5 * 1.3 = 0.65
        assert!((modulated - 0.65).abs() < 1e-6);
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
            0.9,  // Modulated up
            0.5,  // net_activation
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
            Domain::Code,  // Different from Legal
        );

        assert!(!result.domain_matched);
    }

    #[test]
    fn test_expected_domain_boost() {
        // Code: 1.0 + net_activation + 0.1
        let code_boost = expected_domain_boost(Domain::Code);
        assert!(code_boost > 1.4, "Code boost should be > 1.4, got {}", code_boost);
        assert!(code_boost < 2.0, "Code boost should be < 2.0, got {}", code_boost);

        // General should have lower boost
        let general_boost = expected_domain_boost(Domain::General);
        assert!(general_boost < code_boost, "General boost should be lower than Code");
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
            similarity: 0.0,  // Zero base
            domain: None,
            relevance_score: None,
        };

        let result = DomainSearchResult::from_semantic_item(&item, 0.0, 0.0, Domain::General);

        // Should return 1.0 to avoid division by zero
        assert_eq!(result.boost_ratio(), 1.0);
    }

    // ========== Integration Tests (GPU Required) ==========

    #[test]
    #[ignore]  // Requires GPU
    #[cfg(feature = "cuda")]
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
    #[ignore]  // Requires GPU
    #[cfg(feature = "cuda")]
    fn test_domain_search_reranks_correctly() {
        // Verify that domain-matching nodes get boosted above non-matching
        // even if their base similarity is slightly lower
        todo!("Implement with real FAISS index")
    }

    #[test]
    #[ignore]  // Requires GPU
    #[cfg(feature = "cuda")]
    fn test_domain_search_performance_10ms() {
        // Verify <10ms latency for k=10 on 10M vectors
        todo!("Implement performance test")
    }
}
```

### Modifications to `crates/context-graph-graph/src/search/mod.rs`:

Add after line 37:
```rust
pub mod domain_search;
```

Add to re-exports (around line 43):
```rust
pub use domain_search::{domain_aware_search, DomainSearchResult, DomainSearchResults};
```

### Modifications to `crates/context-graph-graph/src/marblestone/mod.rs`:

Add re-export:
```rust
// Domain search (M04-T19)
pub use crate::search::domain_search::{domain_aware_search, DomainSearchResult, DomainSearchResults};
```

---

## Constraints

- MUST use CANONICAL formula for net_activation: `excitatory - inhibitory + (modulatory * 0.5)`
- MUST use types from existing codebase (see Current Codebase State)
- MUST over-fetch 3x candidates before modulation
- MUST re-rank by modulated_score, not base_similarity
- Domain match bonus = 0.1 (DOMAIN_MATCH_BONUS constant)
- Modulated score clamped to [0.0, 1.0] per AP-009
- Performance: <10ms for k=10 on 10M vectors
- NO `unwrap()` or `expect()` - use `GraphResult<T>` everywhere
- Use `tracing` crate for logging (debug!, warn!, etc.)

---

## Acceptance Criteria

- [ ] `domain_aware_search()` over-fetches candidates (3x)
- [ ] Applies NT modulation using CANONICAL formula
- [ ] Re-ranks results by modulated score
- [ ] Truncates to requested k
- [ ] `DomainSearchResult` includes faiss_id, node_id, base_similarity, modulated_score
- [ ] All errors return `GraphResult<T>` - no panics
- [ ] Uses existing types from `crate::search` module
- [ ] Compiles with `cargo build -p context-graph-graph`
- [ ] Unit tests pass with `cargo test -p context-graph-graph domain_search`
- [ ] Integration tests pass with REAL GPU index (no mocks)
- [ ] No clippy warnings: `cargo clippy -p context-graph-graph -- -D warnings`
- [ ] Performance verified: <10ms for k=10 on 10M vectors

---

## Full State Verification Requirements

### 1. Source of Truth Identification

| Component | Source of Truth | Location |
|-----------|-----------------|----------|
| NeurotransmitterWeights | context-graph-core | `context_graph_core::marblestone::NeurotransmitterWeights` |
| Domain enum | context-graph-core | `context_graph_core::marblestone::Domain` |
| SemanticSearchResult | context-graph-graph | `crate::search::result::SemanticSearchResult` |
| NodeMetadataProvider | context-graph-graph | `crate::search::mod::NodeMetadataProvider` |
| CANONICAL formula | constitution.yaml | `edge_model.nt_weights` |

### 2. Execute & Inspect

Before marking complete, execute and verify:

```bash
# Build the crate
cargo build -p context-graph-graph 2>&1

# Run unit tests
cargo test -p context-graph-graph domain_search -- --nocapture 2>&1

# Run clippy
cargo clippy -p context-graph-graph -- -D warnings 2>&1

# Verify the module is exported correctly
cargo doc -p context-graph-graph --no-deps 2>&1 | grep domain_search
```

### 3. Boundary/Edge Case Audit

| Edge Case | Expected Behavior | Test |
|-----------|-------------------|------|
| k=0 | Return empty results, log warning | `test_k_zero` |
| Empty semantic results | Return empty DomainSearchResults | `test_empty_results` |
| All results filtered | Return empty DomainSearchResults | `test_all_filtered` |
| node_domain is None | domain_bonus = 0.0 | `test_no_domain_match` |
| modulated_score > 1.0 | Clamp to 1.0 | `test_modulation_clamping` |
| modulated_score < 0.0 | Clamp to 0.0 | `test_modulation_clamping_negative` |
| base_similarity = 0.0 | boost_ratio returns 1.0 | `test_boost_ratio_zero_base` |
| NaN in similarity | Handled by clamp() | `test_nan_handling` |

### 4. Evidence of Success Logging

All verification must log evidence to `./verification-logs/M04-T19-domain-search-YYYYMMDD-HHMMSS.log`:

```
[PASS] Compilation successful
[PASS] Unit tests: 12/12 passed
[PASS] Clippy: 0 warnings
[PASS] Module exports verified
[PASS] Integration test with real GPU: PASSED
[PASS] Performance: 8.2ms for k=10 on 10M vectors (target: <10ms)
```

---

## Manual Output Verification

After implementation, manually verify the following exist and are correct:

### 1. File System Verification

```bash
# Verify files exist
ls -la crates/context-graph-graph/src/search/domain_search.rs
```

### 2. Code Structure Verification

```bash
# Verify function signature in source
grep -n "pub fn domain_aware_search" crates/context-graph-graph/src/search/domain_search.rs

# Verify struct exists
grep -n "pub struct DomainSearchResult" crates/context-graph-graph/src/search/domain_search.rs

# Verify export in search/mod.rs
grep -n "domain_search" crates/context-graph-graph/src/search/mod.rs
```

### 3. Type Verification

Ensure the following types are used EXACTLY as specified (no redefinitions):

```bash
# Verify Domain import from core (not redefined)
grep -n "use context_graph_core::marblestone" crates/context-graph-graph/src/search/domain_search.rs

# Verify NO local Domain enum definition
grep -n "enum Domain" crates/context-graph-graph/src/search/domain_search.rs
# Should return NO MATCHES
```

### 4. Integration Test Verification (Requires GPU)

```bash
# Run with GPU feature
cargo test -p context-graph-graph domain_search --features cuda -- --ignored --nocapture
```

---

## Sherlock-Holmes Verification Step

After implementation is complete, run the sherlock-holmes subagent for forensic verification:

```
Use Task tool with subagent_type="sherlock-holmes" and prompt:

"Verify M04-T19 domain-aware search implementation is COMPLETE and CORRECT:

1. FILE EXISTENCE:
   - crates/context-graph-graph/src/search/domain_search.rs EXISTS
   - Module exported in search/mod.rs
   - Re-exported in marblestone/mod.rs

2. TYPE VERIFICATION:
   - Domain enum imported from context_graph_core (NOT redefined)
   - NeurotransmitterWeights imported from context_graph_core (NOT redefined)
   - SemanticSearchResultItem used (NOT SemanticSearchResult directly)
   - DomainSearchResult struct has: faiss_id (i64), node_id (Option<Uuid>), base_similarity, modulated_score

3. FORMULA VERIFICATION:
   - net_activation = excitatory - inhibitory + (modulatory * 0.5)
   - domain_bonus = 0.1 if matching, 0.0 otherwise
   - modulated_score = base_similarity * (1.0 + net_activation + domain_bonus)
   - Result clamped to [0.0, 1.0]

4. FUNCTION SIGNATURE:
   - domain_aware_search<M: NodeMetadataProvider>(index, query, query_domain, k, filters, metadata)
   - Returns GraphResult<DomainSearchResults>

5. BUILD VERIFICATION:
   - cargo build -p context-graph-graph compiles without errors
   - cargo test -p context-graph-graph domain_search passes
   - cargo clippy -p context-graph-graph has no warnings

6. NO BACKWARDS COMPATIBILITY HACKS:
   - No deprecated functions
   - No legacy type aliases
   - No commented-out old code

Report: GUILTY (incomplete/incorrect) or INNOCENT (fully verified) with evidence."
```

---

## Domain NT Profiles Reference (VERIFIED from context-graph-core)

Values from `context_graph_core::marblestone::neurotransmitter_weights.rs`:

| Domain | Excitatory | Inhibitory | Modulatory | Net Activation | Boost (with match) |
|--------|------------|------------|------------|----------------|-------------------|
| Code | 0.6 | 0.3 | 0.4 | +0.50 | 1.60x |
| Creative | 0.8 | 0.1 | 0.6 | +1.00 | 2.10x |
| Legal | 0.4 | 0.4 | 0.2 | +0.10 | 1.20x |
| Medical | 0.5 | 0.3 | 0.5 | +0.45 | 1.55x |
| Research | 0.6 | 0.2 | 0.5 | +0.65 | 1.75x |
| General | 0.5 | 0.2 | 0.3 | +0.45 | 1.55x |

Formula: `net_activation = excitatory - inhibitory + (modulatory * 0.5)`
Formula: `boost = 1.0 + net_activation + domain_bonus` (domain_bonus = 0.1 if matching)

*Source of Truth: `NeurotransmitterWeights::for_domain()` in context-graph-core*

---

## Verification Commands

```bash
# Full verification sequence
cargo build -p context-graph-graph
cargo test -p context-graph-graph domain_search
cargo clippy -p context-graph-graph -- -D warnings
cargo doc -p context-graph-graph --no-deps

# Integration tests with GPU
cargo test -p context-graph-graph domain_search --features cuda -- --ignored
```

---

## Completion Verification (VERIFIED 2026-01-04)

### Sherlock-Holmes Forensic Verification

**VERDICT: INNOCENT (COMPLETE AND CORRECT)**

Evidence gathered by sherlock-holmes subagent:

1. **File Existence**: ✅
   - `crates/context-graph-graph/src/search/domain_search.rs` EXISTS (693 lines)
   - Module exported in `search/mod.rs`
   - Re-exported in `marblestone/mod.rs`

2. **Type Verification**: ✅
   - `Domain` enum imported from `context_graph_core` (NOT redefined)
   - `NeurotransmitterWeights` imported from `context_graph_core` (NOT redefined)
   - `DomainSearchResult` struct has: faiss_id (i64), node_id (Option<Uuid>), base_similarity, modulated_score

3. **Formula Implementation**: ✅
   - Uses `compute_net_activation()` helper function (NOT method call)
   - `net_activation = excitatory - inhibitory + (modulatory * 0.5)`
   - `domain_bonus = 0.1` if matching, `0.0` otherwise
   - `modulated_score = base_similarity * (1.0 + net_activation + domain_bonus)`
   - Result clamped to [0.0, 1.0]

4. **Build Verification**: ✅
   - `cargo build -p context-graph-graph` compiles without errors
   - `cargo test -p context-graph-graph domain_search` passes all 13 tests
   - `cargo clippy -p context-graph-graph` has no warnings

5. **NT Values Verified Against Source**: ✅
   - All domain NT values verified against `context_graph_core::marblestone::neurotransmitter_weights.rs`
   - Code: 0.6, 0.3, 0.4 → net=0.50
   - Creative: 0.8, 0.1, 0.6 → net=1.00
   - Legal: 0.4, 0.4, 0.2 → net=0.10
   - Medical: 0.5, 0.3, 0.5 → net=0.45
   - Research: 0.6, 0.2, 0.5 → net=0.65
   - General: 0.5, 0.2, 0.3 → net=0.45

---

*Generated: 2026-01-04*
*Task: M04-T19 - Implement Domain-Aware Search*
*Status: COMPLETE (Verified by sherlock-holmes 2026-01-04)*
*Dependencies: M04-T14a (COMPLETE), M04-T15 (COMPLETE), M04-T18 (COMPLETE)*
