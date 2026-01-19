//! Bio-Nervous Layer Threshold Management
//!
//! Provides domain-aware thresholds for Memory (L3) and Learning (L4) layers.
//! Replaces hardcoded constants with adaptive threshold calibration (ATC).
//!
//! # Constitution Reference
//!
//! From `docs2/constitution.yaml`:
//! - layers.L3_Memory: <1ms latency, MHN
//! - layers.L4_Learning: 100Hz frequency
//!
//! # Legacy Values (MUST preserve for backwards compatibility)
//!
//! - MIN_MEMORY_SIMILARITY = 0.50 (memory retrieval)
//! - DEFAULT_CONSOLIDATION_THRESHOLD = 0.10 (learning consolidation)
//!
//! # ATC Domain Thresholds
//!
//! From `atc/domain.rs`:
//! - theta_memory_sim: [0.35, 0.75] Memory similarity
//! - theta_consolidation: [0.05, 0.30] Consolidation trigger

use crate::atc::{AdaptiveThresholdCalibration, Domain};
use crate::error::{CoreError, CoreResult};

/// Layer thresholds for bio-nervous system (L3, L4).
///
/// These thresholds control behavior across two layers:
/// - `memory_similarity`: Minimum similarity for memory retrieval (L3)
/// - `consolidation`: Threshold for consolidation trigger (L4)
///
/// # Constitution Reference
///
/// - theta_memory_sim: [0.35, 0.75] Memory similarity
/// - theta_consolidation: [0.05, 0.30] Consolidation trigger
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerThresholds {
    /// Minimum similarity for memory to be considered relevant (L3).
    /// Below this, memory results are filtered out.
    pub memory_similarity: f32,

    /// Threshold for consolidation trigger (L4).
    /// When UTL score is below this, consolidation is triggered.
    pub consolidation: f32,
}

impl LayerThresholds {
    /// Create from ATC for a specific domain.
    ///
    /// Retrieves domain-specific thresholds from the Adaptive Threshold Calibration system.
    /// Domain strictness affects threshold values:
    /// - Stricter domains (Medical, Code) have higher thresholds
    /// - Looser domains (Creative) have lower thresholds
    ///
    /// # Arguments
    ///
    /// * `atc` - Reference to the AdaptiveThresholdCalibration system
    /// * `domain` - The domain to retrieve thresholds for
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - ATC doesn't have the requested domain
    /// - Retrieved thresholds fail validation (out of range)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use context_graph_core::atc::{AdaptiveThresholdCalibration, Domain};
    /// use context_graph_core::layers::LayerThresholds;
    ///
    /// let atc = AdaptiveThresholdCalibration::new();
    /// let thresholds = LayerThresholds::from_atc(&atc, Domain::Code)?;
    /// assert!(thresholds.memory_similarity > 0.50); // Code domain is strict
    /// ```
    pub fn from_atc(atc: &AdaptiveThresholdCalibration, domain: Domain) -> CoreResult<Self> {
        let domain_thresholds = atc.get_domain_thresholds(domain).ok_or_else(|| {
            CoreError::ConfigError(format!(
                "ATC missing domain thresholds for {:?}. \
                Ensure AdaptiveThresholdCalibration is properly initialized.",
                domain
            ))
        })?;

        let layer = Self {
            memory_similarity: domain_thresholds.theta_memory_sim,
            consolidation: domain_thresholds.theta_consolidation,
        };

        if !layer.is_valid() {
            return Err(CoreError::ValidationError {
                field: "LayerThresholds".to_string(),
                message: format!(
                    "Invalid thresholds from ATC domain {:?}: memory_sim={}, consolidation={}. \
                    Required: values within constitution ranges.",
                    domain, layer.memory_similarity, layer.consolidation
                ),
            });
        }

        Ok(layer)
    }

    /// Create with legacy General domain defaults.
    ///
    /// These values MUST match the old hardcoded constants for backwards compatibility:
    /// - MIN_MEMORY_SIMILARITY = 0.50
    /// - DEFAULT_CONSOLIDATION_THRESHOLD = 0.10
    ///
    /// # Important
    ///
    /// Use this method when:
    /// - No ATC is available
    /// - Domain context is unknown
    /// - Legacy behavior must be preserved
    ///
    /// For domain-aware behavior, use [`from_atc`](Self::from_atc) instead.
    #[inline]
    pub fn default_general() -> Self {
        Self {
            memory_similarity: 0.50,
            consolidation: 0.10,
        }
    }

    /// Validate thresholds are within constitution ranges.
    ///
    /// # Validation Rules
    ///
    /// Per constitution `atc/domain.rs`:
    /// - memory_similarity: [0.35, 0.75]
    /// - consolidation: [0.05, 0.30]
    ///
    /// # Returns
    ///
    /// `true` if all constraints are satisfied, `false` otherwise.
    pub fn is_valid(&self) -> bool {
        (0.35..=0.75).contains(&self.memory_similarity)
            && (0.05..=0.30).contains(&self.consolidation)
    }

    /// Check if UTL score should trigger consolidation.
    ///
    /// Consolidation is triggered when UTL score FALLS BELOW the threshold.
    /// Lower UTL scores indicate the system needs to consolidate learning.
    ///
    /// # Arguments
    ///
    /// * `utl_score` - Current UTL learning score [0, 1]
    ///
    /// # Returns
    ///
    /// `true` if consolidation should be triggered, `false` otherwise.
    #[inline]
    pub fn should_consolidate(&self, utl_score: f32) -> bool {
        utl_score < self.consolidation
    }

    /// Check if similarity is sufficient for memory retrieval.
    ///
    /// # Arguments
    ///
    /// * `similarity` - Cosine similarity score [0, 1]
    ///
    /// # Returns
    ///
    /// `true` if similarity is above threshold, `false` otherwise.
    #[inline]
    pub fn is_memory_relevant(&self, similarity: f32) -> bool {
        similarity >= self.memory_similarity
    }
}

impl Default for LayerThresholds {
    /// Returns legacy General domain defaults.
    ///
    /// Equivalent to [`default_general()`](Self::default_general).
    fn default() -> Self {
        Self::default_general()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================
    // LEGACY VALUE COMPATIBILITY TESTS
    // ========================================================

    #[test]
    fn test_default_matches_legacy_constants() {
        let t = LayerThresholds::default_general();

        // These MUST match the old hardcoded values EXACTLY
        assert_eq!(
            t.memory_similarity, 0.50,
            "memory_similarity must match MIN_MEMORY_SIMILARITY (0.50), got {}",
            t.memory_similarity
        );
        assert_eq!(
            t.consolidation, 0.10,
            "consolidation must match DEFAULT_CONSOLIDATION_THRESHOLD (0.10), got {}",
            t.consolidation
        );

        println!("[VERIFIED] default_general() matches legacy constants:");
        println!(
            "  memory_similarity: {} == 0.50 (MIN_MEMORY_SIMILARITY)",
            t.memory_similarity
        );
        println!(
            "  consolidation: {} == 0.10 (DEFAULT_CONSOLIDATION_THRESHOLD)",
            t.consolidation
        );
    }

    #[test]
    fn test_default_is_valid() {
        let t = LayerThresholds::default_general();
        assert!(
            t.is_valid(),
            "default_general() must produce valid thresholds"
        );
        println!("[VERIFIED] default_general() produces valid thresholds");
    }

    #[test]
    fn test_default_trait_matches_default_general() {
        let default_trait = LayerThresholds::default();
        let default_general = LayerThresholds::default_general();

        assert_eq!(
            default_trait, default_general,
            "Default trait must match default_general()"
        );
        println!("[VERIFIED] Default trait == default_general()");
    }

    // ========================================================
    // ATC INTEGRATION TESTS
    // ========================================================

    #[test]
    fn test_from_atc_all_domains() {
        let atc = AdaptiveThresholdCalibration::new();

        for domain in [
            Domain::Code,
            Domain::Medical,
            Domain::Legal,
            Domain::Creative,
            Domain::Research,
            Domain::General,
        ] {
            let result = LayerThresholds::from_atc(&atc, domain);
            assert!(
                result.is_ok(),
                "Domain {:?} should produce valid thresholds, got error: {:?}",
                domain,
                result.err()
            );

            let t = result.unwrap();
            assert!(
                t.is_valid(),
                "Domain {:?} thresholds should be valid: memory_sim={}, consolidation={}",
                domain,
                t.memory_similarity,
                t.consolidation
            );
        }
        println!("[VERIFIED] All 6 domains produce valid LayerThresholds from ATC");
    }

    #[test]
    fn test_domain_strictness_ordering() {
        let atc = AdaptiveThresholdCalibration::new();

        let code = LayerThresholds::from_atc(&atc, Domain::Code).unwrap();
        let creative = LayerThresholds::from_atc(&atc, Domain::Creative).unwrap();
        let medical = LayerThresholds::from_atc(&atc, Domain::Medical).unwrap();
        let general = LayerThresholds::from_atc(&atc, Domain::General).unwrap();

        // Stricter domains have HIGHER memory similarity (harder to match)
        assert!(
            code.memory_similarity > creative.memory_similarity,
            "Code memory_similarity ({}) should be > Creative memory_similarity ({})",
            code.memory_similarity,
            creative.memory_similarity
        );
        assert!(
            medical.memory_similarity > general.memory_similarity,
            "Medical memory_similarity ({}) should be > General memory_similarity ({})",
            medical.memory_similarity,
            general.memory_similarity
        );

        // Medical is strictest (strictness=1.0)
        assert!(
            medical.memory_similarity >= code.memory_similarity,
            "Medical memory_similarity ({}) should be >= Code memory_similarity ({})",
            medical.memory_similarity,
            code.memory_similarity
        );

        println!("[VERIFIED] Domain strictness ordering:");
        println!(
            "  memory_sim: Medical({}) >= Code({}) > General({}) > Creative({})",
            medical.memory_similarity,
            code.memory_similarity,
            general.memory_similarity,
            creative.memory_similarity
        );
    }

    #[test]
    fn test_print_all_domain_thresholds() {
        println!("\n=== ATC Domain Layer Thresholds ===\n");

        let atc = AdaptiveThresholdCalibration::new();

        for domain in [
            Domain::Medical,
            Domain::Code,
            Domain::Legal,
            Domain::General,
            Domain::Research,
            Domain::Creative,
        ] {
            let t = LayerThresholds::from_atc(&atc, domain).unwrap();
            println!(
                "{:?} (strictness={:.1}): memory_sim={:.3}, consolidation={:.3}",
                domain,
                domain.strictness(),
                t.memory_similarity,
                t.consolidation
            );
        }

        println!("\nLegacy defaults: memory_sim=0.50, consolidation=0.10");
    }

    // ========================================================
    // HELPER METHOD TESTS
    // ========================================================

    #[test]
    fn test_should_consolidate() {
        let t = LayerThresholds::default_general();

        // Score BELOW threshold triggers consolidation
        assert!(
            t.should_consolidate(0.05),
            "UTL score 0.05 should trigger consolidation (threshold=0.10)"
        );
        assert!(
            t.should_consolidate(0.09),
            "UTL score 0.09 should trigger consolidation (threshold=0.10)"
        );

        // Score AT or ABOVE threshold does NOT trigger
        assert!(
            !t.should_consolidate(0.10),
            "UTL score 0.10 should NOT trigger consolidation (threshold=0.10)"
        );
        assert!(
            !t.should_consolidate(0.15),
            "UTL score 0.15 should NOT trigger consolidation (threshold=0.10)"
        );

        println!("[VERIFIED] should_consolidate boundary at threshold=0.10");
    }

    #[test]
    fn test_is_memory_relevant() {
        let t = LayerThresholds::default_general();

        // Below threshold = not relevant
        assert!(
            !t.is_memory_relevant(0.49),
            "Similarity 0.49 should NOT be relevant (threshold=0.50)"
        );

        // At threshold = relevant
        assert!(
            t.is_memory_relevant(0.50),
            "Similarity 0.50 SHOULD be relevant (threshold=0.50)"
        );

        // Above threshold = relevant
        assert!(
            t.is_memory_relevant(0.80),
            "Similarity 0.80 SHOULD be relevant (threshold=0.50)"
        );

        println!("[VERIFIED] is_memory_relevant boundary at threshold=0.50");
    }

    // ========================================================
    // VALIDATION TESTS
    // ========================================================

    #[test]
    fn test_invalid_memory_sim_out_of_range() {
        // Below minimum (0.35)
        let t1 = LayerThresholds {
            memory_similarity: 0.30,
            consolidation: 0.10,
        };
        assert!(
            !t1.is_valid(),
            "memory_similarity=0.30 below min 0.35 should fail"
        );

        // Above maximum (0.75)
        let t2 = LayerThresholds {
            memory_similarity: 0.80,
            consolidation: 0.10,
        };
        assert!(
            !t2.is_valid(),
            "memory_similarity=0.80 above max 0.75 should fail"
        );
    }

    #[test]
    fn test_invalid_consolidation_out_of_range() {
        // Below minimum (0.05)
        let t1 = LayerThresholds {
            memory_similarity: 0.50,
            consolidation: 0.03,
        };
        assert!(
            !t1.is_valid(),
            "consolidation=0.03 below min 0.05 should fail"
        );

        // Above maximum (0.30)
        let t2 = LayerThresholds {
            memory_similarity: 0.50,
            consolidation: 0.35,
        };
        assert!(
            !t2.is_valid(),
            "consolidation=0.35 above max 0.30 should fail"
        );
    }

    // ========================================================
    // FULL STATE VERIFICATION (FSV) TEST
    // ========================================================

    #[test]
    fn test_fsv_layer_threshold_verification() {
        println!("\n=== FSV: Layer Threshold Verification ===\n");

        // 1. Verify default_general matches legacy
        let default = LayerThresholds::default_general();
        println!("Default General Thresholds:");
        println!(
            "  memory_similarity: {} (expected: 0.50)",
            default.memory_similarity
        );
        println!(
            "  consolidation: {} (expected: 0.10)",
            default.consolidation
        );
        assert_eq!(default.memory_similarity, 0.50);
        assert_eq!(default.consolidation, 0.10);
        println!("  [VERIFIED] Default matches legacy constants\n");

        // 2. Verify ATC retrieval for all domains
        let atc = AdaptiveThresholdCalibration::new();
        println!("ATC Domain Thresholds:");
        for domain in [
            Domain::Code,
            Domain::Medical,
            Domain::Creative,
            Domain::General,
        ] {
            let t = LayerThresholds::from_atc(&atc, domain).unwrap();
            println!(
                "  {:?}: memory_sim={:.3}, consolidation={:.3}",
                domain, t.memory_similarity, t.consolidation
            );
            assert!(t.is_valid());
        }
        println!("  [VERIFIED] All domains produce valid thresholds\n");

        // 3. Helper method boundary tests with state printout
        println!("Helper Method Boundary Tests:");
        let t = LayerThresholds::default_general();

        let test_cases: [(f32, &str, bool, bool); 4] = [
            (0.09, "should_consolidate", t.should_consolidate(0.09), true),
            (
                0.10,
                "should_consolidate",
                t.should_consolidate(0.10),
                false,
            ),
            (
                0.49,
                "is_memory_relevant",
                t.is_memory_relevant(0.49),
                false,
            ),
            (0.50, "is_memory_relevant", t.is_memory_relevant(0.50), true),
        ];

        for (value, method, actual, expected) in test_cases {
            println!(
                "  value={:.2}, {}() = {} (expected: {})",
                value, method, actual, expected
            );
            assert_eq!(actual, expected, "Failed for value={} on {}", value, method);
        }
        println!("  [VERIFIED] All boundary conditions correct\n");

        println!("=== FSV COMPLETE: All verifications passed ===\n");
    }
}
