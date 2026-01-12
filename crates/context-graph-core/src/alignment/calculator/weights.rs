//! Teleological weights for multi-space alignment.

use crate::types::fingerprint::NUM_EMBEDDERS;

/// Teleological weights for each of the 13 embedding spaces.
///
/// From constitution.yaml embedder_purposes, each space has a specific
/// teleological goal. Default weights are equal (1/13 each) but can be
/// customized for domain-specific alignment.
///
/// # Embedder Purposes (from constitution.yaml)
/// - E1_Semantic: V_meaning
/// - E2_Temporal_Recent: V_freshness
/// - E3_Temporal_Periodic: V_periodicity
/// - E4_Temporal_Positional: V_ordering
/// - E5_Causal: V_causality
/// - E6_Sparse: V_selectivity
/// - E7_Code: V_correctness
/// - E8_Graph: V_connectivity
/// - E9_HDC: V_robustness
/// - E10_Multimodal: V_multimodality
/// - E11_Entity: V_factuality
/// - E12_LateInteraction: V_precision
/// - E13_SPLADE: V_keyword_precision
#[derive(Debug, Clone, Copy)]
pub struct TeleologicalWeights {
    /// Weights for each of the 13 embedders.
    /// Must sum to 1.0 for proper normalization.
    pub weights: [f32; NUM_EMBEDDERS],
}

impl Default for TeleologicalWeights {
    /// Default: equal weights for all 13 embedders (1/13 each).
    fn default() -> Self {
        Self {
            weights: [1.0 / NUM_EMBEDDERS as f32; NUM_EMBEDDERS],
        }
    }
}

impl TeleologicalWeights {
    /// Create with equal weights for all embedders.
    pub fn uniform() -> Self {
        Self::default()
    }

    /// Create with custom weights. Weights should sum to 1.0.
    ///
    /// # Panics
    /// Panics if weights don't sum to approximately 1.0 (within 0.01 tolerance).
    pub fn new(weights: [f32; NUM_EMBEDDERS]) -> Self {
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "TeleologicalWeights must sum to 1.0, got {sum}"
        );
        Self { weights }
    }

    /// Create semantic-focused weights (E1 weighted higher).
    pub fn semantic_focused() -> Self {
        let mut weights = [0.05; NUM_EMBEDDERS];
        weights[0] = 0.40; // E1_Semantic
                           // Redistribute remaining 0.60 across other 12 embedders
        for w in weights.iter_mut().skip(1) {
            *w = 0.60 / 12.0;
        }
        Self { weights }
    }

    /// Get the weight for a specific embedder index (0-12).
    #[inline]
    pub fn weight(&self, idx: usize) -> f32 {
        self.weights.get(idx).copied().unwrap_or(0.0)
    }

    /// Validate weights sum to 1.0.
    pub fn validate(&self) -> Result<(), &'static str> {
        let sum: f32 = self.weights.iter().sum();
        if (sum - 1.0).abs() > 0.01 {
            return Err("TeleologicalWeights must sum to 1.0");
        }
        for &w in &self.weights {
            if w < 0.0 {
                return Err("TeleologicalWeights cannot be negative");
            }
        }
        Ok(())
    }
}
