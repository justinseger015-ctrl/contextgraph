//! Search options for teleological memory queries.
//!
//! # Search Strategies
//!
//! Three search strategies are available:
//!
//! - **E1Only** (default): Uses only E1 Semantic HNSW index. Fastest, backward compatible.
//! - **MultiSpace**: Weighted fusion of semantic embedders (E1, E5, E7, E10).
//!   Temporal embedders (E2-E4) are excluded from scoring per research findings.
//! - **Pipeline**: Full 3-stage retrieval: Recall → Score → Re-rank.
//!
//! # Fusion Strategies (ARCH-18)
//!
//! When using MultiSpace or Pipeline strategies, score fusion can use:
//!
//! - **WeightedSum** (legacy): Simple weighted sum of similarity scores
//! - **WeightedRRF** (default per ARCH-18): Weighted Reciprocal Rank Fusion
//!
//! RRF formula: `RRF_score(d) = Sum(weight_i / (rank_i + k))`
//!
//! RRF is more robust to score distribution differences between embedders.
//!
//! # Research References
//!
//! - [Cascading Retrieval](https://www.pinecone.io/blog/cascading-retrieval/) - 48% improvement
//! - [Fusion Analysis](https://dl.acm.org/doi/10.1145/3596512) - Convex combination beats RRF
//! - [Elastic Weighted RRF](https://www.elastic.co/blog/weighted-reciprocal-rank-fusion-rrf)
//! - [ColBERT Late Interaction](https://weaviate.io/blog/late-interaction-overview)

use serde::{Deserialize, Serialize};

use crate::code::CodeQueryType;
use crate::fusion::FusionStrategy;
use crate::types::fingerprint::SemanticFingerprint;

/// Search strategy for semantic queries.
///
/// Controls how the 13-embedder multi-space index is used for ranking.
///
/// # Key Insight
///
/// Temporal embedders (E2-E4) measure TIME proximity, not TOPIC similarity.
/// They are excluded from similarity scoring and applied as post-retrieval boosts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SearchStrategy {
    /// E1 HNSW only. Backward compatible, fastest.
    #[default]
    E1Only,

    /// Weighted fusion of semantic embedders (E1, E5, E7, E10).
    /// Temporal embedders (E2-E4) have weight 0.0 per AP-71.
    MultiSpace,

    /// Full 3-stage pipeline: Recall → Score → Re-rank.
    /// Stage 1: E13 SPLADE + E1 for broad recall.
    /// Stage 2: Multi-space scoring with semantic embedders.
    /// Stage 3: Optional E12 ColBERT re-ranking.
    Pipeline,
}

/// Score normalization strategy for multi-space fusion.
///
/// Applied before combining scores from multiple embedders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationStrategyOption {
    /// No normalization - use raw similarity scores.
    None,

    /// Min-max normalization to [0, 1] range.
    #[default]
    MinMax,

    /// Z-score normalization (mean=0, std=1), scaled to [0, 1].
    ZScore,

    /// Convex combination (research-backed best practice).
    /// See: https://dl.acm.org/doi/10.1145/3596512
    Convex,
}

/// Search options for teleological memory queries.
///
/// Controls filtering, pagination, and result formatting for
/// semantic and purpose-based searches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleologicalSearchOptions {
    /// Maximum number of results to return.
    /// Default: 10, Max: 1000
    pub top_k: usize,

    /// Minimum similarity threshold [0.0, 1.0].
    /// Results below this threshold are filtered out.
    /// Default: 0.0 (no filtering)
    pub min_similarity: f32,

    /// Include soft-deleted items in results.
    /// Default: false
    pub include_deleted: bool,

    /// Embedder indices to use for search (0-12).
    /// Empty = use all embedders.
    pub embedder_indices: Vec<usize>,

    /// Optional semantic fingerprint for computing per-embedder scores.
    /// When provided, enables computation of actual cosine similarity scores
    /// for each embedder instead of returning zeros.
    #[serde(skip)]
    pub semantic_query: Option<SemanticFingerprint>,

    /// Whether to include original content text in search results.
    ///
    /// When `true`, the `content` field of `TeleologicalSearchResult` will be
    /// populated with the original text (if available). When `false` (default),
    /// the `content` field will be `None` for better performance.
    ///
    /// Default: `false` (opt-in for performance reasons)
    ///
    /// TASK-CONTENT-005: Added for content hydration in search results.
    #[serde(default)]
    pub include_content: bool,

    // =========================================================================
    // Multi-Space Search Options (TASK-MULTISPACE)
    // =========================================================================

    /// Search strategy: E1Only (default), MultiSpace, or Pipeline.
    ///
    /// - `E1Only`: Backward compatible, uses only E1 Semantic HNSW.
    /// - `MultiSpace`: Weighted fusion of semantic embedders.
    /// - `Pipeline`: Full 3-stage retrieval with optional re-ranking.
    ///
    /// Default: `E1Only` for backward compatibility.
    #[serde(default)]
    pub strategy: SearchStrategy,

    /// Weight profile name for multi-space scoring.
    ///
    /// Available profiles:
    /// - `"semantic_search"`: General queries (E1: 35%, E7: 20%, E5/E10: 15%)
    /// - `"code_search"`: Programming queries (E7: 40%, E1: 20%)
    /// - `"causal_reasoning"`: "Why" questions (E5: 45%, E1: 20%)
    /// - `"fact_checking"`: Entity/fact queries (E11: 40%, E6: 15%)
    /// - `"category_weighted"`: Constitution-compliant category weights
    ///
    /// All profiles have E2-E4 (temporal) = 0.0 per research findings.
    /// Default: `"semantic_search"`.
    #[serde(default)]
    pub weight_profile: Option<String>,

    /// Recency boost factor [0.0, 1.0].
    ///
    /// Applied POST-retrieval as: `final = semantic * (1.0 - boost) + temporal * boost`.
    /// Uses E2 temporal embedding similarity for recency scoring.
    ///
    /// - `0.0`: No recency boost (default)
    /// - `0.5`: Balance semantic and recency
    /// - `1.0`: Strong recency preference
    ///
    /// Per ARCH-14: Temporal is a POST-retrieval boost, not similarity.
    #[serde(default)]
    pub recency_boost: f32,

    /// Enable E12 ColBERT re-ranking (Stage 3 in Pipeline strategy).
    ///
    /// More accurate but slower. Per AP-73: ColBERT is for re-ranking only.
    /// Default: `false`.
    #[serde(default)]
    pub enable_rerank: bool,

    /// Normalization strategy for score fusion.
    ///
    /// Applied before combining scores from multiple embedders.
    /// Default: `MinMax`.
    #[serde(default)]
    pub normalization: NormalizationStrategyOption,

    /// Fusion strategy for combining multi-embedder results (ARCH-18).
    ///
    /// - `WeightedSum`: Legacy weighted sum of similarity scores
    /// - `WeightedRRF`: Weighted Reciprocal Rank Fusion (default per ARCH-18)
    ///
    /// RRF formula: `RRF_score(d) = Sum(weight_i / (rank_i + k))`
    ///
    /// RRF is recommended because it:
    /// - Preserves individual embedder rankings
    /// - Is robust to score distribution differences between embedders
    /// - Works well with varying numbers of results per embedder
    ///
    /// Default: `WeightedRRF` per ARCH-18.
    #[serde(default)]
    pub fusion_strategy: FusionStrategy,

    // =========================================================================
    // Code Query Type Detection (ARCH-16)
    // =========================================================================

    /// Original query text for code query type detection.
    ///
    /// When provided, enables E7 Code embedder similarity adjustment
    /// based on detected query type (Code2Code, Text2Code, NonCode).
    ///
    /// Per ARCH-16: E7 Code MUST detect query type and use appropriate
    /// similarity computation.
    #[serde(default)]
    pub query_text: Option<String>,

    /// Pre-computed code query type.
    ///
    /// If `None` and `query_text` is provided, the type will be
    /// auto-detected. If explicitly set, skips auto-detection.
    ///
    /// - `Code2Code`: Query is actual code syntax (e.g., "fn process<T>")
    /// - `Text2Code`: Query is natural language about code (e.g., "batch function")
    /// - `NonCode`: Query is not code-related
    #[serde(default)]
    pub code_query_type: Option<CodeQueryType>,
}

impl Default for TeleologicalSearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_similarity: 0.0,
            include_deleted: false,
            embedder_indices: Vec::new(),
            semantic_query: None,
            include_content: false, // TASK-CONTENT-005: Opt-in for performance
            // Multi-space options (TASK-MULTISPACE)
            strategy: SearchStrategy::default(),
            weight_profile: None,
            recency_boost: 0.0,
            enable_rerank: false,
            normalization: NormalizationStrategyOption::default(),
            // Fusion strategy (ARCH-18) - WeightedRRF by default
            fusion_strategy: FusionStrategy::default(),
            // Code query type detection (ARCH-16)
            query_text: None,
            code_query_type: None,
        }
    }
}

impl TeleologicalSearchOptions {
    /// Create options for a quick top-k search.
    #[inline]
    pub fn quick(top_k: usize) -> Self {
        Self {
            top_k,
            ..Default::default()
        }
    }

    /// Create options with minimum similarity threshold.
    #[inline]
    pub fn with_min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = threshold;
        self
    }

    /// Create options filtering by specific embedders.
    #[inline]
    pub fn with_embedders(mut self, indices: Vec<usize>) -> Self {
        self.embedder_indices = indices;
        self
    }

    /// Attach semantic fingerprint for computing per-embedder similarity scores.
    /// When provided, computes actual cosine similarities between query and
    /// stored semantic fingerprints instead of returning zeros.
    #[inline]
    pub fn with_semantic_query(mut self, semantic: SemanticFingerprint) -> Self {
        self.semantic_query = Some(semantic);
        self
    }

    /// Set whether to include original content text in search results.
    ///
    /// When `true`, content will be fetched and included in results.
    /// Default is `false` for better performance.
    ///
    /// TASK-CONTENT-005: Builder method for content inclusion.
    #[inline]
    pub fn with_include_content(mut self, include: bool) -> Self {
        self.include_content = include;
        self
    }

    // =========================================================================
    // Multi-Space Search Builder Methods (TASK-MULTISPACE)
    // =========================================================================

    /// Set the search strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - One of `E1Only`, `MultiSpace`, or `Pipeline`
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::traits::{TeleologicalSearchOptions, SearchStrategy};
    ///
    /// let opts = TeleologicalSearchOptions::quick(10)
    ///     .with_strategy(SearchStrategy::MultiSpace);
    /// ```
    #[inline]
    pub fn with_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the weight profile for multi-space scoring.
    ///
    /// # Arguments
    ///
    /// * `profile` - Profile name (e.g., "semantic_search", "code_search")
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::traits::TeleologicalSearchOptions;
    ///
    /// let opts = TeleologicalSearchOptions::quick(10)
    ///     .with_weight_profile("code_search");
    /// ```
    #[inline]
    pub fn with_weight_profile(mut self, profile: &str) -> Self {
        self.weight_profile = Some(profile.to_string());
        self
    }

    /// Set the recency boost factor.
    ///
    /// Applied POST-retrieval as: `final = semantic * (1.0 - boost) + temporal * boost`.
    ///
    /// # Arguments
    ///
    /// * `factor` - Boost factor [0.0, 1.0]. Clamped to valid range.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::traits::TeleologicalSearchOptions;
    ///
    /// let opts = TeleologicalSearchOptions::quick(10)
    ///     .with_recency_boost(0.3);
    /// ```
    #[inline]
    pub fn with_recency_boost(mut self, factor: f32) -> Self {
        self.recency_boost = factor.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable E12 ColBERT re-ranking.
    ///
    /// Only effective with `SearchStrategy::Pipeline`.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable re-ranking
    #[inline]
    pub fn with_rerank(mut self, enable: bool) -> Self {
        self.enable_rerank = enable;
        self
    }

    /// Set the normalization strategy for score fusion.
    ///
    /// # Arguments
    ///
    /// * `strategy` - Normalization strategy
    #[inline]
    pub fn with_normalization(mut self, strategy: NormalizationStrategyOption) -> Self {
        self.normalization = strategy;
        self
    }

    /// Set the fusion strategy for combining multi-embedder results (ARCH-18).
    ///
    /// # Arguments
    ///
    /// * `strategy` - Fusion strategy (`WeightedSum` or `WeightedRRF`)
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::traits::TeleologicalSearchOptions;
    /// use context_graph_core::fusion::FusionStrategy;
    ///
    /// let opts = TeleologicalSearchOptions::quick(10)
    ///     .with_fusion_strategy(FusionStrategy::WeightedRRF);
    /// ```
    #[inline]
    pub fn with_fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }

    // =========================================================================
    // Code Query Type Builder Methods (ARCH-16)
    // =========================================================================

    /// Set the query text for E7 Code query type detection.
    ///
    /// When provided, enables automatic detection of whether the query is:
    /// - Code2Code: Actual code syntax (e.g., "fn process<T>()")
    /// - Text2Code: Natural language about code (e.g., "batch processing function")
    /// - NonCode: Not code-related
    ///
    /// E7 similarity computation is adjusted based on detected type.
    ///
    /// # Arguments
    ///
    /// * `query` - The original query text
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::traits::TeleologicalSearchOptions;
    ///
    /// let opts = TeleologicalSearchOptions::quick(10)
    ///     .with_query_text("impl Iterator for Counter");
    /// ```
    #[inline]
    pub fn with_query_text(mut self, query: &str) -> Self {
        self.query_text = Some(query.to_string());
        self
    }

    /// Explicitly set the code query type.
    ///
    /// Use this to override auto-detection when you know the query type.
    ///
    /// # Arguments
    ///
    /// * `query_type` - The code query type
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::traits::TeleologicalSearchOptions;
    /// use context_graph_core::code::CodeQueryType;
    ///
    /// let opts = TeleologicalSearchOptions::quick(10)
    ///     .with_code_query_type(CodeQueryType::Code2Code);
    /// ```
    #[inline]
    pub fn with_code_query_type(mut self, query_type: CodeQueryType) -> Self {
        self.code_query_type = Some(query_type);
        self
    }

    /// Get the effective code query type, detecting if necessary.
    ///
    /// Returns:
    /// - The explicitly set `code_query_type` if present
    /// - Auto-detected type from `query_text` if present
    /// - `None` if neither is available
    pub fn effective_code_query_type(&self) -> Option<CodeQueryType> {
        if let Some(explicit) = self.code_query_type {
            return Some(explicit);
        }
        if let Some(ref text) = self.query_text {
            return Some(crate::code::detect_code_query_type(text));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_options_default() {
        let opts = TeleologicalSearchOptions::default();
        assert_eq!(opts.top_k, 10);
        assert_eq!(opts.min_similarity, 0.0);
        assert!(!opts.include_deleted);
        assert!(opts.embedder_indices.is_empty());
        // ARCH-18: Default fusion strategy should be WeightedRRF
        assert_eq!(opts.fusion_strategy, crate::fusion::FusionStrategy::WeightedRRF);
    }

    #[test]
    fn test_search_options_quick() {
        let opts = TeleologicalSearchOptions::quick(50);
        assert_eq!(opts.top_k, 50);
    }

    #[test]
    fn test_search_options_builder() {
        let opts = TeleologicalSearchOptions::quick(20)
            .with_min_similarity(0.5)
            .with_embedders(vec![0, 1, 2]);

        assert_eq!(opts.top_k, 20);
        assert_eq!(opts.min_similarity, 0.5);
        assert_eq!(opts.embedder_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_search_options_fusion_strategy() {
        // Test default is WeightedRRF per ARCH-18
        let opts = TeleologicalSearchOptions::default();
        assert_eq!(opts.fusion_strategy, crate::fusion::FusionStrategy::WeightedRRF);

        // Test builder method
        let opts = TeleologicalSearchOptions::quick(10)
            .with_fusion_strategy(crate::fusion::FusionStrategy::WeightedSum);
        assert_eq!(opts.fusion_strategy, crate::fusion::FusionStrategy::WeightedSum);

        let opts = TeleologicalSearchOptions::quick(10)
            .with_fusion_strategy(crate::fusion::FusionStrategy::WeightedRRF);
        assert_eq!(opts.fusion_strategy, crate::fusion::FusionStrategy::WeightedRRF);
    }
}
