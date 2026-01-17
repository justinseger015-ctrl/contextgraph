//! BIRCH clustering parameters and ClusteringFeature.
//!
//! Provides configuration types for BIRCH (Balanced Iterative Reducing and
//! Clustering using Hierarchies) algorithm. BIRCH enables O(log n) incremental
//! clustering for real-time memory insertion.
//!
//! # Constitution Defaults
//!
//! Per constitution BIRCH_DEFAULTS:
//! - branching_factor: 50
//! - threshold: 0.3 (adaptive)
//! - max_node_entries: 50
//!
//! # Clustering Feature (CF)
//!
//! The CF is a triple (n, LS, SS) that summarizes a set of points:
//! - n: number of data points
//! - LS: linear sum (vector sum of all points)
//! - SS: squared sum (scalar sum of squared norms)
//!
//! Key property: CFs are additive. CF(A ∪ B) = CF(A) + CF(B)
//!
//! # BIRCHTree
//!
//! The BIRCH CF-tree for incremental clustering:
//! - O(log n) insertion via tree traversal
//! - Automatic node splitting when exceeding max_node_entries
//! - Memory ID tracking for cluster membership queries
//! - Threshold adaptation for target cluster count

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::teleological::Embedder;

use super::error::ClusterError;

// =============================================================================
// BIRCHParams
// =============================================================================

/// Parameters for BIRCH clustering algorithm.
///
/// Per constitution: branching_factor=50, threshold=0.3, max_node_entries=50
///
/// # Example
///
/// ```
/// use context_graph_core::clustering::birch::{BIRCHParams, birch_defaults};
/// use context_graph_core::teleological::Embedder;
///
/// // Use defaults
/// let params = birch_defaults();
/// assert_eq!(params.branching_factor, 50);
///
/// // Or space-specific
/// let code_params = BIRCHParams::default_for_space(Embedder::Code);
/// assert!(code_params.threshold < 0.3); // Code embeddings more specific
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BIRCHParams {
    /// Maximum number of children per non-leaf node.
    /// Controls tree width. Higher = flatter tree but more work per node.
    pub branching_factor: usize,

    /// Threshold for cluster radius.
    /// Points within this radius merge into same CF. Adaptive in practice.
    pub threshold: f32,

    /// Maximum entries per leaf node.
    /// When exceeded, node splits.
    pub max_node_entries: usize,
}

impl Default for BIRCHParams {
    fn default() -> Self {
        Self {
            branching_factor: 50, // Per constitution
            threshold: 0.3,       // Per constitution
            max_node_entries: 50, // Per constitution
        }
    }
}

impl BIRCHParams {
    /// Create new BIRCH params.
    ///
    /// Values are NOT automatically validated - call validate() to check.
    pub fn new(branching_factor: usize, threshold: f32, max_entries: usize) -> Self {
        Self {
            branching_factor,
            threshold,
            max_node_entries: max_entries,
        }
    }

    /// Create params for a specific embedding space.
    ///
    /// Adjusts threshold based on space characteristics:
    /// - Sparse spaces (Sparse, KeywordSplade): 0.4 (looser for high dimensionality)
    /// - Code embeddings: 0.25 (tighter for specificity)
    /// - All other spaces: 0.3 (constitution default)
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::clustering::birch::BIRCHParams;
    /// use context_graph_core::teleological::Embedder;
    ///
    /// let sparse_params = BIRCHParams::default_for_space(Embedder::Sparse);
    /// assert_eq!(sparse_params.threshold, 0.4);
    ///
    /// let code_params = BIRCHParams::default_for_space(Embedder::Code);
    /// assert_eq!(code_params.threshold, 0.25);
    /// ```
    pub fn default_for_space(embedder: Embedder) -> Self {
        let threshold = match embedder {
            // Sparse spaces need looser threshold due to high dimensionality
            Embedder::Sparse | Embedder::KeywordSplade => 0.4,
            // Code embeddings are more specific, need tighter threshold
            Embedder::Code => 0.25,
            // All other spaces use constitution default
            _ => 0.3,
        };

        Self {
            branching_factor: 50,
            threshold,
            max_node_entries: 50,
        }
    }

    /// Set branching factor.
    ///
    /// Value is NOT automatically clamped - use validate() to check.
    #[must_use]
    pub fn with_branching_factor(mut self, bf: usize) -> Self {
        self.branching_factor = bf;
        self
    }

    /// Set threshold.
    ///
    /// Value is NOT automatically clamped - use validate() to check.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set max node entries.
    ///
    /// Value is NOT automatically clamped - use validate() to check.
    #[must_use]
    pub fn with_max_node_entries(mut self, entries: usize) -> Self {
        self.max_node_entries = entries;
        self
    }

    /// Validate parameters.
    ///
    /// Fails fast with descriptive error messages.
    ///
    /// # Errors
    ///
    /// Returns `ClusterError::InvalidParameter` if:
    /// - branching_factor < 2
    /// - threshold <= 0.0 or threshold is NaN/Infinity
    /// - max_node_entries < branching_factor
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::clustering::birch::BIRCHParams;
    ///
    /// let invalid = BIRCHParams::new(1, 0.3, 50);
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), ClusterError> {
        if self.branching_factor < 2 {
            return Err(ClusterError::invalid_parameter(format!(
                "branching_factor must be >= 2, got {}. BIRCH tree nodes need at least 2 children.",
                self.branching_factor
            )));
        }

        if self.threshold <= 0.0 || self.threshold.is_nan() || self.threshold.is_infinite() {
            return Err(ClusterError::invalid_parameter(format!(
                "threshold must be > 0.0 and finite, got {}. Threshold controls cluster compactness.",
                self.threshold
            )));
        }

        if self.max_node_entries < self.branching_factor {
            return Err(ClusterError::invalid_parameter(format!(
                "max_node_entries ({}) must be >= branching_factor ({}). Leaf nodes must hold at least branching_factor entries.",
                self.max_node_entries, self.branching_factor
            )));
        }

        Ok(())
    }
}

/// Get default BIRCH parameters.
///
/// Returns params matching constitution defaults:
/// - branching_factor: 50
/// - threshold: 0.3
/// - max_node_entries: 50
pub fn birch_defaults() -> BIRCHParams {
    BIRCHParams::default()
}

// =============================================================================
// ClusteringFeature
// =============================================================================

/// Clustering Feature - statistical summary for BIRCH.
///
/// A CF is a triple (n, LS, SS) that summarizes a set of d-dimensional points:
/// - n: number of data points
/// - LS: linear sum, d-dimensional vector = Σ Xi
/// - SS: squared sum, scalar = Σ ||Xi||²
///
/// # Key Properties
///
/// 1. **Additivity**: CF(A ∪ B) = CF(A) + CF(B)
/// 2. **Sufficient Statistics**: Can compute centroid, radius, diameter
/// 3. **Compact**: O(d) space regardless of n
///
/// # Example
///
/// ```
/// use context_graph_core::clustering::birch::ClusteringFeature;
///
/// let mut cf = ClusteringFeature::from_point(&[1.0, 2.0, 3.0]);
/// cf.add_point(&[2.0, 3.0, 4.0]).unwrap();
///
/// assert_eq!(cf.n, 2);
/// assert_eq!(cf.centroid(), vec![1.5, 2.5, 3.5]);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusteringFeature {
    /// Number of data points summarized.
    pub n: u32,
    /// Linear sum: Σ Xi (d-dimensional vector).
    pub ls: Vec<f32>,
    /// Squared sum: Σ ||Xi||² (scalar).
    pub ss: f32,
}

impl ClusteringFeature {
    /// Create empty CF with given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            n: 0,
            ls: vec![0.0; dimension],
            ss: 0.0,
        }
    }

    /// Create CF from a single point.
    pub fn from_point(point: &[f32]) -> Self {
        let ss: f32 = point.iter().map(|x| x * x).sum();
        Self {
            n: 1,
            ls: point.to_vec(),
            ss,
        }
    }

    /// Get dimension of this CF.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.ls.len()
    }

    /// Check if CF is empty (no points).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Compute centroid (mean point).
    ///
    /// centroid = LS / n
    ///
    /// Returns zero vector if n=0.
    pub fn centroid(&self) -> Vec<f32> {
        if self.n == 0 {
            return self.ls.clone(); // Zero vector
        }
        let n_f32 = self.n as f32;
        self.ls.iter().map(|x| x / n_f32).collect()
    }

    /// Compute radius (RMS distance from centroid to points).
    ///
    /// radius = sqrt(SS/n - ||centroid||²)
    ///
    /// Returns 0.0 if n=0 or if variance is negative (numerical precision).
    pub fn radius(&self) -> f32 {
        if self.n == 0 {
            return 0.0;
        }

        let centroid = self.centroid();
        let centroid_norm_sq: f32 = centroid.iter().map(|x| x * x).sum();
        let variance = (self.ss / self.n as f32) - centroid_norm_sq;

        // Handle numerical precision issues
        if variance < 0.0 || variance.is_nan() {
            0.0
        } else {
            variance.sqrt()
        }
    }

    /// Compute diameter (average pairwise distance approximation).
    ///
    /// diameter ≈ 2 * radius (approximation for spherical clusters)
    ///
    /// Returns 0.0 if n <= 1.
    pub fn diameter(&self) -> f32 {
        if self.n <= 1 {
            return 0.0;
        }
        2.0 * self.radius()
    }

    /// Merge another CF into this one.
    ///
    /// CF(A ∪ B) = (n_A + n_B, LS_A + LS_B, SS_A + SS_B)
    ///
    /// # Errors
    ///
    /// Returns `ClusterError::DimensionMismatch` if dimensions differ.
    pub fn merge(&mut self, other: &ClusteringFeature) -> Result<(), ClusterError> {
        if other.n == 0 {
            return Ok(()); // Merging empty CF is no-op
        }

        if self.n == 0 {
            // Self is empty, just copy other
            self.n = other.n;
            self.ls = other.ls.clone();
            self.ss = other.ss;
            return Ok(());
        }

        // Check dimension match
        if self.ls.len() != other.ls.len() {
            return Err(ClusterError::dimension_mismatch(self.ls.len(), other.ls.len()));
        }

        // Additive merge
        self.n += other.n;
        for (a, b) in self.ls.iter_mut().zip(other.ls.iter()) {
            *a += b;
        }
        self.ss += other.ss;

        Ok(())
    }

    /// Add a single point to this CF.
    ///
    /// # Errors
    ///
    /// Returns `ClusterError::DimensionMismatch` if point dimension differs.
    pub fn add_point(&mut self, point: &[f32]) -> Result<(), ClusterError> {
        // Initialize dimension if empty
        if self.ls.is_empty() {
            self.ls = vec![0.0; point.len()];
        }

        // Check dimension match
        if self.ls.len() != point.len() {
            return Err(ClusterError::dimension_mismatch(self.ls.len(), point.len()));
        }

        self.n += 1;
        for (a, b) in self.ls.iter_mut().zip(point.iter()) {
            *a += b;
        }
        self.ss += point.iter().map(|x| x * x).sum::<f32>();

        Ok(())
    }

    /// Compute Euclidean distance between centroids of two CFs.
    ///
    /// # Errors
    ///
    /// Returns `ClusterError::DimensionMismatch` if dimensions differ.
    pub fn distance(&self, other: &ClusteringFeature) -> Result<f32, ClusterError> {
        if self.ls.len() != other.ls.len() {
            return Err(ClusterError::dimension_mismatch(self.ls.len(), other.ls.len()));
        }

        let c1 = self.centroid();
        let c2 = other.centroid();

        let dist_sq: f32 = c1
            .iter()
            .zip(c2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        Ok(dist_sq.sqrt())
    }

    /// Check if a point would fit within threshold after merging.
    ///
    /// Computes the hypothetical radius without allocating a new CF.
    pub fn would_fit(&self, point: &[f32], threshold: f32) -> bool {
        if self.n == 0 {
            return true; // Empty CF accepts anything
        }

        if self.ls.len() != point.len() {
            return false; // Dimension mismatch
        }

        // Compute merged statistics inline without cloning
        let new_n = self.n + 1;
        let new_n_f32 = new_n as f32;

        // Compute new centroid and check radius
        let point_ss: f32 = point.iter().map(|x| x * x).sum();
        let new_ss = self.ss + point_ss;

        // centroid_norm_sq = ||new_ls / new_n||^2 = (1/new_n^2) * ||new_ls||^2
        let new_centroid_norm_sq: f32 = self
            .ls
            .iter()
            .zip(point.iter())
            .map(|(a, b)| {
                let sum = a + b;
                (sum / new_n_f32) * (sum / new_n_f32)
            })
            .sum();

        let variance = (new_ss / new_n_f32) - new_centroid_norm_sq;

        if variance < 0.0 || variance.is_nan() {
            return true; // Numerical precision: treat as zero radius
        }

        variance.sqrt() <= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // BIRCHParams DEFAULT VALUES TESTS
    // =========================================================================

    #[test]
    fn test_birch_defaults_match_constitution() {
        let params = birch_defaults();

        // Per constitution: BIRCH_DEFAULTS
        assert_eq!(
            params.branching_factor, 50,
            "branching_factor must be 50 per constitution"
        );
        assert!(
            (params.threshold - 0.3).abs() < f32::EPSILON,
            "threshold must be 0.3 per constitution"
        );
        assert_eq!(
            params.max_node_entries, 50,
            "max_node_entries must be 50 per constitution"
        );

        // Validate should pass for defaults
        assert!(params.validate().is_ok(), "Default params must be valid");

        println!("[PASS] test_birch_defaults_match_constitution - defaults verified");
    }

    // =========================================================================
    // BIRCHParams SPACE-SPECIFIC TESTS
    // =========================================================================

    #[test]
    fn test_default_for_space_sparse() {
        let params = BIRCHParams::default_for_space(Embedder::Sparse);
        assert!(
            (params.threshold - 0.4).abs() < f32::EPSILON,
            "Sparse should use 0.4 threshold"
        );
        assert!(params.validate().is_ok());
        println!("[PASS] test_default_for_space_sparse - threshold=0.4");
    }

    #[test]
    fn test_default_for_space_keyword_splade() {
        let params = BIRCHParams::default_for_space(Embedder::KeywordSplade);
        assert!(
            (params.threshold - 0.4).abs() < f32::EPSILON,
            "KeywordSplade should use 0.4 threshold"
        );
        assert!(params.validate().is_ok());
        println!("[PASS] test_default_for_space_keyword_splade - threshold=0.4");
    }

    #[test]
    fn test_default_for_space_code() {
        let params = BIRCHParams::default_for_space(Embedder::Code);
        assert!(
            (params.threshold - 0.25).abs() < f32::EPSILON,
            "Code should use 0.25 threshold"
        );
        assert!(params.validate().is_ok());
        println!("[PASS] test_default_for_space_code - threshold=0.25");
    }

    #[test]
    fn test_default_for_space_semantic() {
        let params = BIRCHParams::default_for_space(Embedder::Semantic);
        assert!(
            (params.threshold - 0.3).abs() < f32::EPSILON,
            "Semantic should use 0.3 threshold"
        );
        assert!(params.validate().is_ok());
        println!("[PASS] test_default_for_space_semantic - threshold=0.3");
    }

    #[test]
    fn test_default_for_all_embedders() {
        // Verify all 13 embedder variants produce valid params
        for embedder in Embedder::all() {
            let params = BIRCHParams::default_for_space(embedder);
            assert!(
                params.validate().is_ok(),
                "default_for_space({:?}) must produce valid params",
                embedder
            );
        }
        println!("[PASS] test_default_for_all_embedders - all 13 variants produce valid params");
    }

    // =========================================================================
    // BIRCHParams VALIDATION TESTS - FAIL FAST
    // =========================================================================

    #[test]
    fn test_validation_rejects_branching_factor_below_2() {
        let params = BIRCHParams::new(1, 0.3, 50);
        let result = params.validate();
        assert!(result.is_err(), "branching_factor=1 must be rejected");

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("branching_factor"),
            "Error must mention field name"
        );
        assert!(err_msg.contains("2"), "Error must mention minimum value");

        println!(
            "[PASS] test_validation_rejects_branching_factor_below_2 - error: {}",
            err_msg
        );
    }

    #[test]
    fn test_validation_rejects_zero_threshold() {
        let params = BIRCHParams::new(50, 0.0, 50);
        let result = params.validate();
        assert!(result.is_err(), "threshold=0.0 must be rejected");

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("threshold"),
            "Error must mention field name"
        );

        println!(
            "[PASS] test_validation_rejects_zero_threshold - error: {}",
            err_msg
        );
    }

    #[test]
    fn test_validation_rejects_negative_threshold() {
        let params = BIRCHParams::new(50, -0.1, 50);
        let result = params.validate();
        assert!(result.is_err(), "threshold=-0.1 must be rejected");
        println!("[PASS] test_validation_rejects_negative_threshold");
    }

    #[test]
    fn test_validation_rejects_nan_threshold() {
        let params = BIRCHParams::new(50, f32::NAN, 50);
        let result = params.validate();
        assert!(result.is_err(), "threshold=NaN must be rejected");
        println!("[PASS] test_validation_rejects_nan_threshold");
    }

    #[test]
    fn test_validation_rejects_infinite_threshold() {
        let params = BIRCHParams::new(50, f32::INFINITY, 50);
        let result = params.validate();
        assert!(result.is_err(), "threshold=INFINITY must be rejected");
        println!("[PASS] test_validation_rejects_infinite_threshold");
    }

    #[test]
    fn test_validation_rejects_max_entries_below_branching() {
        let params = BIRCHParams::new(50, 0.3, 40);
        let result = params.validate();
        assert!(
            result.is_err(),
            "max_node_entries < branching_factor must be rejected"
        );

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("max_node_entries"),
            "Error must mention max_node_entries"
        );
        assert!(
            err_msg.contains("branching_factor"),
            "Error must mention branching_factor"
        );

        println!(
            "[PASS] test_validation_rejects_max_entries_below_branching - error: {}",
            err_msg
        );
    }

    #[test]
    fn test_validation_accepts_boundary_values() {
        // Minimum valid: branching_factor=2, threshold=0.001, max_entries=2
        let minimal = BIRCHParams::new(2, 0.001, 2);
        assert!(minimal.validate().is_ok(), "Minimal valid params must pass");

        // Equal values: max_entries == branching_factor
        let equal = BIRCHParams::new(100, 1.0, 100);
        assert!(
            equal.validate().is_ok(),
            "Equal max_entries and branching_factor must pass"
        );

        println!("[PASS] test_validation_accepts_boundary_values");
    }

    // =========================================================================
    // BIRCHParams BUILDER TESTS
    // =========================================================================

    #[test]
    fn test_builder_pattern() {
        let params = BIRCHParams::default()
            .with_branching_factor(100)
            .with_threshold(0.5)
            .with_max_node_entries(200);

        assert_eq!(params.branching_factor, 100);
        assert!((params.threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(params.max_node_entries, 200);
        assert!(params.validate().is_ok());

        println!("[PASS] test_builder_pattern - all builder methods work");
    }

    #[test]
    fn test_builder_does_not_auto_clamp() {
        // Builder should NOT auto-clamp - validation is explicit
        let params = BIRCHParams::default()
            .with_branching_factor(1) // Invalid
            .with_threshold(0.0); // Invalid

        assert_eq!(params.branching_factor, 1, "Builder must not modify value");
        assert!(
            (params.threshold - 0.0).abs() < f32::EPSILON,
            "Builder must not modify value"
        );
        assert!(
            params.validate().is_err(),
            "Validation must catch invalid values"
        );

        println!("[PASS] test_builder_does_not_auto_clamp - explicit validation required");
    }

    // =========================================================================
    // BIRCHParams SERIALIZATION TESTS
    // =========================================================================

    #[test]
    fn test_birch_params_serialization_roundtrip() {
        let params = BIRCHParams::default_for_space(Embedder::Code)
            .with_branching_factor(75)
            .with_max_node_entries(100);

        let json = serde_json::to_string(&params).expect("serialize must succeed");
        let restored: BIRCHParams = serde_json::from_str(&json).expect("deserialize must succeed");

        assert_eq!(params.branching_factor, restored.branching_factor);
        assert!((params.threshold - restored.threshold).abs() < f32::EPSILON);
        assert_eq!(params.max_node_entries, restored.max_node_entries);

        println!(
            "[PASS] test_birch_params_serialization_roundtrip - JSON: {}",
            json
        );
    }

    // =========================================================================
    // ClusteringFeature CREATION TESTS
    // =========================================================================

    #[test]
    fn test_cf_new_empty() {
        let cf = ClusteringFeature::new(128);
        assert_eq!(cf.n, 0);
        assert_eq!(cf.dimension(), 128);
        assert_eq!(cf.ls.len(), 128);
        assert!(cf.ls.iter().all(|&x| x == 0.0));
        assert_eq!(cf.ss, 0.0);
        assert!(cf.is_empty());

        println!("[PASS] test_cf_new_empty - dimension={}", cf.dimension());
    }

    #[test]
    fn test_cf_from_point() {
        let point = vec![1.0, 2.0, 3.0];
        let cf = ClusteringFeature::from_point(&point);

        assert_eq!(cf.n, 1);
        assert_eq!(cf.ls, point);
        assert_eq!(cf.ss, 14.0); // 1 + 4 + 9 = 14
        assert_eq!(cf.dimension(), 3);
        assert!(!cf.is_empty());
        assert_eq!(cf.centroid(), point);

        println!("[PASS] test_cf_from_point - n={}, ss={}", cf.n, cf.ss);
    }

    // =========================================================================
    // ClusteringFeature CENTROID TESTS
    // =========================================================================

    #[test]
    fn test_cf_centroid_single_point() {
        let point = vec![1.0, 2.0, 3.0];
        let cf = ClusteringFeature::from_point(&point);
        assert_eq!(cf.centroid(), point);
        println!("[PASS] test_cf_centroid_single_point");
    }

    #[test]
    fn test_cf_centroid_two_points() {
        let mut cf = ClusteringFeature::new(3);
        cf.add_point(&[1.0, 0.0, 0.0]).unwrap();
        cf.add_point(&[3.0, 0.0, 0.0]).unwrap();

        let centroid = cf.centroid();
        assert_eq!(centroid, vec![2.0, 0.0, 0.0]);

        println!(
            "[PASS] test_cf_centroid_two_points - centroid={:?}",
            centroid
        );
    }

    #[test]
    fn test_cf_centroid_empty() {
        let cf = ClusteringFeature::new(3);
        let centroid = cf.centroid();
        assert_eq!(
            centroid,
            vec![0.0, 0.0, 0.0],
            "Empty CF should return zero vector"
        );

        println!("[PASS] test_cf_centroid_empty - returns zero vector");
    }

    // =========================================================================
    // ClusteringFeature RADIUS TESTS
    // =========================================================================

    #[test]
    fn test_cf_radius_single_point() {
        let cf = ClusteringFeature::from_point(&[1.0, 2.0, 3.0]);
        assert_eq!(cf.radius(), 0.0, "Single point has zero radius");

        println!("[PASS] test_cf_radius_single_point - radius=0");
    }

    #[test]
    fn test_cf_radius_two_symmetric_points() {
        let mut cf = ClusteringFeature::new(2);
        cf.add_point(&[-1.0, 0.0]).unwrap();
        cf.add_point(&[1.0, 0.0]).unwrap();

        // Centroid is (0, 0), each point is distance 1 away
        let radius = cf.radius();
        assert!((radius - 1.0).abs() < 1e-5);

        println!(
            "[PASS] test_cf_radius_two_symmetric_points - radius={}",
            radius
        );
    }

    #[test]
    fn test_cf_radius_empty() {
        let cf = ClusteringFeature::new(3);
        assert_eq!(cf.radius(), 0.0, "Empty CF should have zero radius");

        println!("[PASS] test_cf_radius_empty - radius=0");
    }

    // =========================================================================
    // ClusteringFeature DIAMETER TESTS
    // =========================================================================

    #[test]
    fn test_cf_diameter_two_points() {
        let mut cf = ClusteringFeature::new(2);
        cf.add_point(&[-1.0, 0.0]).unwrap();
        cf.add_point(&[1.0, 0.0]).unwrap();

        let diameter = cf.diameter();
        // diameter approx 2 * radius = 2 * 1.0 = 2.0
        assert!((diameter - 2.0).abs() < 1e-5);

        println!("[PASS] test_cf_diameter_two_points - diameter={}", diameter);
    }

    #[test]
    fn test_cf_diameter_single_point() {
        let cf = ClusteringFeature::from_point(&[1.0, 2.0]);
        assert_eq!(cf.diameter(), 0.0, "Single point has zero diameter");

        println!("[PASS] test_cf_diameter_single_point - diameter=0");
    }

    // =========================================================================
    // ClusteringFeature MERGE TESTS
    // =========================================================================

    #[test]
    fn test_cf_merge_two_cfs() {
        let mut cf1 = ClusteringFeature::from_point(&[1.0, 2.0]);
        let cf2 = ClusteringFeature::from_point(&[3.0, 4.0]);

        cf1.merge(&cf2).unwrap();

        assert_eq!(cf1.n, 2);
        assert_eq!(cf1.ls, vec![4.0, 6.0]);
        assert_eq!(cf1.ss, 5.0 + 25.0); // (1+4) + (9+16) = 30
        assert_eq!(cf1.centroid(), vec![2.0, 3.0]);

        println!(
            "[PASS] test_cf_merge_two_cfs - n={}, ls={:?}",
            cf1.n, cf1.ls
        );
    }

    #[test]
    fn test_cf_merge_with_empty() {
        let mut cf1 = ClusteringFeature::from_point(&[1.0, 2.0]);
        let cf2 = ClusteringFeature::new(2);

        cf1.merge(&cf2).unwrap();

        assert_eq!(cf1.n, 1, "Merging empty CF should not change count");
        assert_eq!(
            cf1.ls,
            vec![1.0, 2.0],
            "Merging empty CF should not change LS"
        );

        println!("[PASS] test_cf_merge_with_empty - no change");
    }

    #[test]
    fn test_cf_merge_into_empty() {
        let mut cf1 = ClusteringFeature::new(2);
        let cf2 = ClusteringFeature::from_point(&[1.0, 2.0]);

        cf1.merge(&cf2).unwrap();

        assert_eq!(cf1.n, 1);
        assert_eq!(cf1.ls, vec![1.0, 2.0]);

        println!("[PASS] test_cf_merge_into_empty - copied from other");
    }

    #[test]
    fn test_cf_merge_dimension_mismatch() {
        let mut cf1 = ClusteringFeature::from_point(&[1.0, 2.0]);
        let cf2 = ClusteringFeature::from_point(&[1.0, 2.0, 3.0]);

        let result = cf1.merge(&cf2);
        assert!(result.is_err(), "Dimension mismatch must fail");

        let err = result.unwrap_err();
        assert!(matches!(err, ClusterError::DimensionMismatch { .. }));

        println!("[PASS] test_cf_merge_dimension_mismatch - correctly rejected");
    }

    // =========================================================================
    // ClusteringFeature ADD_POINT TESTS
    // =========================================================================

    #[test]
    fn test_cf_add_point() {
        let mut cf = ClusteringFeature::new(3);
        cf.add_point(&[1.0, 2.0, 3.0]).unwrap();
        cf.add_point(&[4.0, 5.0, 6.0]).unwrap();

        assert_eq!(cf.n, 2);
        assert_eq!(cf.ls, vec![5.0, 7.0, 9.0]);

        println!("[PASS] test_cf_add_point - n={}", cf.n);
    }

    #[test]
    fn test_cf_add_point_dimension_mismatch() {
        let mut cf = ClusteringFeature::from_point(&[1.0, 2.0]);

        let result = cf.add_point(&[1.0, 2.0, 3.0]);
        assert!(result.is_err(), "Dimension mismatch must fail");

        println!("[PASS] test_cf_add_point_dimension_mismatch - correctly rejected");
    }

    // =========================================================================
    // ClusteringFeature DISTANCE TESTS
    // =========================================================================

    #[test]
    fn test_cf_distance_3_4_5_triangle() {
        let cf1 = ClusteringFeature::from_point(&[0.0, 0.0]);
        let cf2 = ClusteringFeature::from_point(&[3.0, 4.0]);

        let dist = cf1.distance(&cf2).unwrap();
        assert!((dist - 5.0).abs() < 1e-5, "3-4-5 triangle: distance should be 5");

        println!("[PASS] test_cf_distance_3_4_5_triangle - dist={}", dist);
    }

    #[test]
    fn test_cf_distance_same_point() {
        let cf1 = ClusteringFeature::from_point(&[1.0, 2.0, 3.0]);
        let cf2 = ClusteringFeature::from_point(&[1.0, 2.0, 3.0]);

        let dist = cf1.distance(&cf2).unwrap();
        assert!(dist.abs() < 1e-5, "Same point distance should be 0");

        println!("[PASS] test_cf_distance_same_point - dist={}", dist);
    }

    #[test]
    fn test_cf_distance_dimension_mismatch() {
        let cf1 = ClusteringFeature::from_point(&[1.0, 2.0]);
        let cf2 = ClusteringFeature::from_point(&[1.0, 2.0, 3.0]);

        let result = cf1.distance(&cf2);
        assert!(result.is_err(), "Dimension mismatch must fail");

        println!("[PASS] test_cf_distance_dimension_mismatch - correctly rejected");
    }

    // =========================================================================
    // ClusteringFeature WOULD_FIT TESTS
    // =========================================================================

    #[test]
    fn test_cf_would_fit_close_point() {
        let cf = ClusteringFeature::from_point(&[0.0, 0.0]);
        assert!(cf.would_fit(&[0.1, 0.1], 1.0), "Close point should fit");

        println!("[PASS] test_cf_would_fit_close_point");
    }

    #[test]
    fn test_cf_would_fit_far_point() {
        let cf = ClusteringFeature::from_point(&[0.0, 0.0]);
        assert!(!cf.would_fit(&[10.0, 10.0], 1.0), "Far point should not fit");

        println!("[PASS] test_cf_would_fit_far_point");
    }

    #[test]
    fn test_cf_would_fit_empty_cf() {
        let cf = ClusteringFeature::new(2);
        assert!(
            cf.would_fit(&[100.0, 100.0], 0.1),
            "Empty CF should accept any point"
        );

        println!("[PASS] test_cf_would_fit_empty_cf");
    }

    #[test]
    fn test_cf_would_fit_dimension_mismatch() {
        let cf = ClusteringFeature::from_point(&[0.0, 0.0]);
        assert!(
            !cf.would_fit(&[1.0, 2.0, 3.0], 10.0),
            "Dimension mismatch should not fit"
        );

        println!("[PASS] test_cf_would_fit_dimension_mismatch");
    }

    // =========================================================================
    // ClusteringFeature SERIALIZATION TESTS
    // =========================================================================

    #[test]
    fn test_cf_serialization_roundtrip() {
        let mut cf = ClusteringFeature::from_point(&[1.0, 2.0, 3.0]);
        cf.add_point(&[4.0, 5.0, 6.0]).unwrap();

        let json = serde_json::to_string(&cf).expect("serialize must succeed");
        let restored: ClusteringFeature =
            serde_json::from_str(&json).expect("deserialize must succeed");

        assert_eq!(cf.n, restored.n);
        assert_eq!(cf.ls, restored.ls);
        assert!((cf.ss - restored.ss).abs() < f32::EPSILON);

        println!("[PASS] test_cf_serialization_roundtrip - JSON: {}", json);
    }

    // =========================================================================
    // EDGE CASE TESTS
    // =========================================================================

    #[test]
    fn test_cf_high_dimensional() {
        // Test with 1024 dimensions (like E1 semantic embeddings)
        let dim = 1024;
        let mut cf = ClusteringFeature::new(dim);

        let point: Vec<f32> = (0..dim).map(|i| i as f32 * 0.001).collect();
        cf.add_point(&point).unwrap();

        assert_eq!(cf.dimension(), dim);
        assert_eq!(cf.n, 1);

        println!("[PASS] test_cf_high_dimensional - dim={}", dim);
    }

    #[test]
    fn test_cf_numerical_precision_radius() {
        // Create scenario that could cause negative variance due to precision
        let mut cf = ClusteringFeature::new(2);
        // Add same point multiple times
        for _ in 0..1000 {
            cf.add_point(&[1e-10, 1e-10]).unwrap();
        }

        let radius = cf.radius();
        assert!(!radius.is_nan(), "Radius should not be NaN");
        assert!(radius >= 0.0, "Radius should not be negative");

        println!(
            "[PASS] test_cf_numerical_precision_radius - radius={}",
            radius
        );
    }

    #[test]
    fn test_cf_large_values() {
        let cf = ClusteringFeature::from_point(&[1e6, 1e6, 1e6]);

        let centroid = cf.centroid();
        assert!((centroid[0] - 1e6).abs() < 1.0, "Large values should work");

        println!("[PASS] test_cf_large_values");
    }

    #[test]
    fn test_cf_zero_dimension() {
        // Edge case: zero-dimension CF
        let cf = ClusteringFeature::new(0);
        assert_eq!(cf.dimension(), 0);
        assert!(cf.is_empty());
        assert_eq!(cf.centroid(), Vec::<f32>::new());
        assert_eq!(cf.radius(), 0.0);
        assert_eq!(cf.diameter(), 0.0);

        println!("[PASS] test_cf_zero_dimension - handles zero dimension gracefully");
    }

    #[test]
    fn test_cf_merge_both_empty() {
        let mut cf1 = ClusteringFeature::new(3);
        let cf2 = ClusteringFeature::new(3);

        cf1.merge(&cf2).unwrap();

        assert_eq!(cf1.n, 0);
        assert!(cf1.is_empty());

        println!("[PASS] test_cf_merge_both_empty - merging two empty CFs works");
    }

    #[test]
    fn test_cf_add_point_to_uninitialized() {
        // Test adding a point when ls is empty (dimension set by first point)
        let mut cf = ClusteringFeature {
            n: 0,
            ls: Vec::new(),
            ss: 0.0,
        };

        cf.add_point(&[1.0, 2.0, 3.0]).unwrap();

        assert_eq!(cf.n, 1);
        assert_eq!(cf.dimension(), 3);
        assert_eq!(cf.ls, vec![1.0, 2.0, 3.0]);

        println!("[PASS] test_cf_add_point_to_uninitialized - dimension set by first point");
    }

    #[test]
    fn test_cf_centroid_many_points() {
        // Test centroid calculation with many points
        let mut cf = ClusteringFeature::new(2);

        for i in 0..100 {
            cf.add_point(&[i as f32, (100 - i) as f32]).unwrap();
        }

        let centroid = cf.centroid();
        // Average of 0..99 = 49.5, average of 100..1 = 50.5
        assert!((centroid[0] - 49.5).abs() < 1e-4);
        assert!((centroid[1] - 50.5).abs() < 1e-4);

        println!(
            "[PASS] test_cf_centroid_many_points - centroid={:?}",
            centroid
        );
    }

    #[test]
    fn test_cf_radius_uniform_distribution() {
        // Test radius with uniformly distributed points around origin
        let mut cf = ClusteringFeature::new(2);

        // Add points at (1,0), (0,1), (-1,0), (0,-1)
        cf.add_point(&[1.0, 0.0]).unwrap();
        cf.add_point(&[0.0, 1.0]).unwrap();
        cf.add_point(&[-1.0, 0.0]).unwrap();
        cf.add_point(&[0.0, -1.0]).unwrap();

        let centroid = cf.centroid();
        assert!((centroid[0]).abs() < 1e-5, "Centroid x should be 0");
        assert!((centroid[1]).abs() < 1e-5, "Centroid y should be 0");

        let radius = cf.radius();
        // All points are distance 1 from origin
        assert!((radius - 1.0).abs() < 1e-5, "Radius should be 1");

        println!(
            "[PASS] test_cf_radius_uniform_distribution - radius={}",
            radius
        );
    }
}
