//! Configuration types for Knowledge Graph components.
//!
//! This module provides configuration structures for:
//! - FAISS IVF-PQ vector index (IndexConfig)
//! - Hyperbolic/Poincare ball geometry (HyperbolicConfig)
//! - Entailment cones for IS-A queries (ConeConfig)
//!
//! # Constitution Reference
//!
//! - perf.latency.faiss_1M_k100: <2ms (drives nlist/nprobe defaults)
//! - embeddings.models.E7_Code: 1536D (default dimension)
//!
//! TODO: Full implementation in M04-T01, M04-T02, M04-T03

use serde::{Deserialize, Serialize};

use crate::error::GraphError;

/// Configuration for FAISS IVF-PQ GPU index.
///
/// Configures the FAISS GPU index for 10M+ vector search with <5ms latency.
///
/// # Performance Targets
/// - 10M vectors, k=10: <5ms latency
/// - 10M vectors, k=100: <10ms latency
/// - Memory: ~8GB VRAM for 10M 1536D vectors with PQ64x8
///
/// # Constitution Reference
/// - perf.latency.faiss_1M_k100: <2ms
/// - stack.deps: faiss@0.12+gpu
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexConfig {
    /// Vector dimension (must match embedding dimension).
    /// Default: 1536 per constitution embeddings.models.E7_Code
    pub dimension: usize,

    /// Number of inverted lists (clusters).
    /// Default: 16384 = 4 * sqrt(10M) for optimal recall/speed tradeoff
    pub nlist: usize,

    /// Number of clusters to probe during search.
    /// Default: 128 balances accuracy vs search time
    pub nprobe: usize,

    /// Number of product quantization segments.
    /// Must evenly divide dimension. Default: 64 (1536/64 = 24 bytes per segment)
    pub pq_segments: usize,

    /// Bits per quantization code.
    /// Valid values: 4, 8, 12, 16. Default: 8
    pub pq_bits: u8,

    /// GPU device ID.
    /// Default: 0 (primary GPU)
    pub gpu_id: i32,

    /// Use float16 for reduced memory.
    /// Default: true (halves VRAM usage)
    pub use_float16: bool,

    /// Minimum vectors required for training (256 * nlist).
    /// Default: 4,194,304 (256 * 16384)
    pub min_train_vectors: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimension: 1536,
            nlist: 16384,
            nprobe: 128,
            pq_segments: 64,
            pq_bits: 8,
            gpu_id: 0,
            use_float16: true,
            min_train_vectors: 4_194_304, // 256 * 16384
        }
    }
}

impl IndexConfig {
    /// Generate FAISS factory string for index creation.
    ///
    /// Returns format: "IVF{nlist},PQ{pq_segments}x{pq_bits}"
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::IndexConfig;
    /// let config = IndexConfig::default();
    /// assert_eq!(config.factory_string(), "IVF16384,PQ64x8");
    /// ```
    pub fn factory_string(&self) -> String {
        format!("IVF{},PQ{}x{}", self.nlist, self.pq_segments, self.pq_bits)
    }

    /// Calculate minimum training vectors based on nlist.
    ///
    /// FAISS requires at least 256 vectors per cluster for quality training.
    ///
    /// # Returns
    /// 256 * nlist
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::IndexConfig;
    /// let config = IndexConfig::default();
    /// assert_eq!(config.calculate_min_train_vectors(), 4_194_304);
    /// ```
    pub fn calculate_min_train_vectors(&self) -> usize {
        256 * self.nlist
    }
}

/// Hyperbolic (Poincare ball) configuration.
///
/// Configures the Poincare ball model for representing hierarchical
/// relationships in hyperbolic space.
///
/// # Mathematics
/// - d(x,y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
/// - Curvature must be negative (typically -1.0)
/// - All points must have norm < 1.0
///
/// # Constitution Reference
/// - edge_model.nt_weights: Neurotransmitter weighting in hyperbolic space
/// - perf.latency.entailment_check: <1ms
///
/// # Example
/// ```
/// use context_graph_graph::config::HyperbolicConfig;
///
/// let config = HyperbolicConfig::default();
/// assert_eq!(config.dim, 64);
/// assert_eq!(config.curvature, -1.0);
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HyperbolicConfig {
    /// Dimension of hyperbolic space (typically 64 for knowledge graphs).
    /// Must be positive.
    pub dim: usize,

    /// Curvature of hyperbolic space. MUST be negative.
    /// Default: -1.0 (unit hyperbolic space)
    /// Validated in validate().
    pub curvature: f32,

    /// Epsilon for numerical stability in hyperbolic operations.
    /// Prevents division by zero and NaN in distance calculations.
    /// Default: 1e-7
    pub eps: f32,

    /// Maximum norm for points (keeps points strictly inside ball boundary).
    /// Points with norm >= max_norm will be projected back inside.
    /// Must be in open interval (0, 1). Default: 1.0 - 1e-5 = 0.99999
    pub max_norm: f32,
}

impl Default for HyperbolicConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            curvature: -1.0,
            eps: 1e-7,
            max_norm: 1.0 - 1e-5, // 0.99999
        }
    }
}

impl HyperbolicConfig {
    /// Create config with custom curvature.
    ///
    /// # Arguments
    /// * `curvature` - Must be negative. Use validate() to check.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    /// let config = HyperbolicConfig::with_curvature(-0.5);
    /// assert_eq!(config.curvature, -0.5);
    /// assert_eq!(config.dim, 64); // other fields use defaults
    /// ```
    pub fn with_curvature(curvature: f32) -> Self {
        Self {
            curvature,
            ..Default::default()
        }
    }

    /// Get absolute value of curvature.
    ///
    /// Useful for formulas that need |c| rather than c.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    /// let config = HyperbolicConfig::default();
    /// assert_eq!(config.abs_curvature(), 1.0);
    /// ```
    #[inline]
    pub fn abs_curvature(&self) -> f32 {
        self.curvature.abs()
    }

    /// Scale factor derived from curvature: sqrt(|c|)
    ///
    /// Used in Mobius operations and distance calculations.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    /// let config = HyperbolicConfig::default();
    /// assert_eq!(config.scale(), 1.0); // sqrt(|-1.0|) = 1.0
    /// ```
    #[inline]
    pub fn scale(&self) -> f32 {
        self.abs_curvature().sqrt()
    }

    /// Validate that all configuration parameters are mathematically valid
    /// for the Poincare ball model.
    ///
    /// # Validation Rules
    /// - `dim` > 0: Dimension must be positive
    /// - `curvature` < 0: Must be negative for hyperbolic space
    /// - `eps` > 0: Must be positive for numerical stability
    /// - `max_norm` in (0, 1): Must be strictly between 0 and 1
    ///
    /// # Errors
    /// Returns `GraphError::InvalidConfig` with descriptive message if any
    /// parameter is invalid. Returns the FIRST error encountered (fail-fast).
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// // Valid config passes
    /// let valid = HyperbolicConfig::default();
    /// assert!(valid.validate().is_ok());
    ///
    /// // Invalid curvature fails
    /// let mut invalid = HyperbolicConfig::default();
    /// invalid.curvature = 1.0; // positive is invalid
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), GraphError> {
        // Check dimension
        if self.dim == 0 {
            return Err(GraphError::InvalidConfig(
                "dim must be positive (got 0)".to_string()
            ));
        }

        // Check curvature - MUST be negative for hyperbolic space
        if self.curvature >= 0.0 {
            return Err(GraphError::InvalidConfig(
                format!(
                    "curvature must be negative for hyperbolic space (got {})",
                    self.curvature
                )
            ));
        }

        // Check for NaN curvature
        if self.curvature.is_nan() {
            return Err(GraphError::InvalidConfig(
                "curvature cannot be NaN".to_string()
            ));
        }

        // Check epsilon
        if self.eps <= 0.0 {
            return Err(GraphError::InvalidConfig(
                format!(
                    "eps must be positive for numerical stability (got {})",
                    self.eps
                )
            ));
        }

        // Check for NaN eps
        if self.eps.is_nan() {
            return Err(GraphError::InvalidConfig(
                "eps cannot be NaN".to_string()
            ));
        }

        // Check max_norm - must be in open interval (0, 1)
        if self.max_norm <= 0.0 || self.max_norm >= 1.0 {
            return Err(GraphError::InvalidConfig(
                format!(
                    "max_norm must be in open interval (0, 1), got {}",
                    self.max_norm
                )
            ));
        }

        // Check for NaN max_norm
        if self.max_norm.is_nan() {
            return Err(GraphError::InvalidConfig(
                "max_norm cannot be NaN".to_string()
            ));
        }

        Ok(())
    }

    /// Create a validated config with custom curvature.
    ///
    /// Returns error if curvature is invalid (>= 0 or NaN).
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let config = HyperbolicConfig::try_with_curvature(-0.5).unwrap();
    /// assert_eq!(config.curvature, -0.5);
    ///
    /// // Invalid curvature returns error
    /// assert!(HyperbolicConfig::try_with_curvature(1.0).is_err());
    /// ```
    pub fn try_with_curvature(curvature: f32) -> Result<Self, GraphError> {
        let config = Self {
            curvature,
            ..Default::default()
        };
        config.validate()?;
        Ok(config)
    }
}

/// Configuration for entailment cones in hyperbolic space.
///
/// Entailment cones enable O(1) IS-A hierarchy queries. A concept's cone
/// contains all concepts it subsumes. Aperture narrows with depth,
/// creating increasingly specific cones for child concepts.
///
/// # Mathematics
///
/// - Aperture at depth d: `aperture(d) = base_aperture * decay^d`
/// - Result clamped to `[min_aperture, max_aperture]`
/// - Cone A contains point P iff angle(P - apex, axis) <= aperture
///
/// # Constitution Reference
///
/// - perf.latency.entailment_check: <1ms
/// - Section 9 "HYPERBOLIC ENTAILMENT CONES" in contextprd.md
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConeConfig {
    /// Minimum cone aperture in radians.
    /// Prevents cones from becoming too narrow at deep levels.
    /// Default: 0.1 rad (~5.7 degrees)
    pub min_aperture: f32,

    /// Maximum cone aperture in radians.
    /// Prevents cones from becoming too wide at root level.
    /// Default: 1.5 rad (~85.9 degrees)
    pub max_aperture: f32,

    /// Base aperture for depth 0 nodes (root concepts).
    /// This is the starting aperture before decay is applied.
    /// Default: 1.0 rad (~57.3 degrees)
    pub base_aperture: f32,

    /// Decay factor applied per hierarchy level.
    /// Must be in open interval (0, 1).
    /// Default: 0.85 (15% narrower per level)
    pub aperture_decay: f32,

    /// Threshold for soft membership scoring.
    /// Points with membership score >= threshold are considered contained.
    /// Must be in open interval (0, 1).
    /// Default: 0.7
    pub membership_threshold: f32,
}

impl Default for ConeConfig {
    fn default() -> Self {
        Self {
            min_aperture: 0.1,      // ~5.7 degrees
            max_aperture: 1.5,      // ~85.9 degrees
            base_aperture: 1.0,     // ~57.3 degrees
            aperture_decay: 0.85,   // 15% narrower per level
            membership_threshold: 0.7,
        }
    }
}

impl ConeConfig {
    /// Compute aperture for a node at given depth.
    ///
    /// # Formula
    /// `aperture = base_aperture * aperture_decay^depth`
    /// Result is clamped to `[min_aperture, max_aperture]`.
    ///
    /// # Arguments
    /// * `depth` - Depth in hierarchy (0 = root)
    ///
    /// # Returns
    /// Aperture in radians, clamped to valid range.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::ConeConfig;
    ///
    /// let config = ConeConfig::default();
    /// assert_eq!(config.compute_aperture(0), 1.0);  // base at root
    /// assert!((config.compute_aperture(1) - 0.85).abs() < 1e-6);  // 1.0 * 0.85
    /// assert_eq!(config.compute_aperture(100), 0.1);  // clamped to min
    /// ```
    pub fn compute_aperture(&self, depth: u32) -> f32 {
        let raw = self.base_aperture * self.aperture_decay.powi(depth as i32);
        raw.clamp(self.min_aperture, self.max_aperture)
    }

    /// Validate configuration parameters.
    ///
    /// # Validation Rules
    /// - `min_aperture` > 0: Must be positive
    /// - `max_aperture` > `min_aperture`: Max must exceed min
    /// - `base_aperture` in [`min_aperture`, `max_aperture`]: Base must be in valid range
    /// - `aperture_decay` in (0, 1): Must be strictly between 0 and 1
    /// - `membership_threshold` in (0, 1): Must be strictly between 0 and 1
    ///
    /// # Errors
    /// Returns `GraphError::InvalidConfig` with descriptive message if any
    /// parameter is invalid. Fails fast on first error.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::ConeConfig;
    ///
    /// let valid = ConeConfig::default();
    /// assert!(valid.validate().is_ok());
    ///
    /// let mut invalid = ConeConfig::default();
    /// invalid.aperture_decay = 1.5;  // must be < 1
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), GraphError> {
        // Check for NaN in min_aperture
        if self.min_aperture.is_nan() {
            return Err(GraphError::InvalidConfig(
                "min_aperture cannot be NaN".to_string()
            ));
        }

        // Check min_aperture is positive
        if self.min_aperture <= 0.0 {
            return Err(GraphError::InvalidConfig(
                format!("min_aperture must be positive (got {})", self.min_aperture)
            ));
        }

        // Check for NaN in max_aperture
        if self.max_aperture.is_nan() {
            return Err(GraphError::InvalidConfig(
                "max_aperture cannot be NaN".to_string()
            ));
        }

        // Check max_aperture > min_aperture
        if self.max_aperture <= self.min_aperture {
            return Err(GraphError::InvalidConfig(
                format!(
                    "max_aperture ({}) must be greater than min_aperture ({})",
                    self.max_aperture, self.min_aperture
                )
            ));
        }

        // Check for NaN in base_aperture
        if self.base_aperture.is_nan() {
            return Err(GraphError::InvalidConfig(
                "base_aperture cannot be NaN".to_string()
            ));
        }

        // Check base_aperture is in valid range
        if self.base_aperture < self.min_aperture || self.base_aperture > self.max_aperture {
            return Err(GraphError::InvalidConfig(
                format!(
                    "base_aperture ({}) must be in range [{}, {}]",
                    self.base_aperture, self.min_aperture, self.max_aperture
                )
            ));
        }

        // Check for NaN in aperture_decay
        if self.aperture_decay.is_nan() {
            return Err(GraphError::InvalidConfig(
                "aperture_decay cannot be NaN".to_string()
            ));
        }

        // Check aperture_decay in (0, 1)
        if self.aperture_decay <= 0.0 || self.aperture_decay >= 1.0 {
            return Err(GraphError::InvalidConfig(
                format!(
                    "aperture_decay must be in open interval (0, 1), got {}",
                    self.aperture_decay
                )
            ));
        }

        // Check for NaN in membership_threshold
        if self.membership_threshold.is_nan() {
            return Err(GraphError::InvalidConfig(
                "membership_threshold cannot be NaN".to_string()
            ));
        }

        // Check membership_threshold in (0, 1)
        if self.membership_threshold <= 0.0 || self.membership_threshold >= 1.0 {
            return Err(GraphError::InvalidConfig(
                format!(
                    "membership_threshold must be in open interval (0, 1), got {}",
                    self.membership_threshold
                )
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_config_default_values() {
        let config = IndexConfig::default();
        assert_eq!(config.dimension, 1536);
        assert_eq!(config.nlist, 16384);
        assert_eq!(config.nprobe, 128);
        assert_eq!(config.pq_segments, 64);
        assert_eq!(config.pq_bits, 8);
        assert_eq!(config.gpu_id, 0);
        assert!(config.use_float16);
        assert_eq!(config.min_train_vectors, 4_194_304);
    }

    #[test]
    fn test_index_config_pq_segments_divides_dimension() {
        let config = IndexConfig::default();
        assert_eq!(
            config.dimension % config.pq_segments,
            0,
            "PQ segments must divide dimension evenly"
        );
    }

    #[test]
    fn test_index_config_min_train_vectors_formula() {
        let config = IndexConfig::default();
        assert_eq!(
            config.min_train_vectors,
            256 * config.nlist,
            "min_train_vectors must equal 256 * nlist"
        );
    }

    #[test]
    fn test_factory_string_default() {
        let config = IndexConfig::default();
        assert_eq!(config.factory_string(), "IVF16384,PQ64x8");
    }

    #[test]
    fn test_factory_string_custom() {
        let config = IndexConfig {
            dimension: 768,
            nlist: 4096,
            nprobe: 64,
            pq_segments: 32,
            pq_bits: 4,
            gpu_id: 1,
            use_float16: false,
            min_train_vectors: 256 * 4096,
        };
        assert_eq!(config.factory_string(), "IVF4096,PQ32x4");
    }

    #[test]
    fn test_calculate_min_train_vectors() {
        let config = IndexConfig::default();
        assert_eq!(config.calculate_min_train_vectors(), 4_194_304);

        let custom = IndexConfig {
            nlist: 1024,
            ..Default::default()
        };
        assert_eq!(custom.calculate_min_train_vectors(), 256 * 1024);
    }

    #[test]
    fn test_index_config_serialization_roundtrip() {
        let config = IndexConfig::default();
        let json = serde_json::to_string(&config).expect("Serialization failed");
        let deserialized: IndexConfig =
            serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_index_config_json_format() {
        let config = IndexConfig::default();
        let json = serde_json::to_string_pretty(&config).expect("Serialization failed");
        assert!(json.contains("\"dimension\": 1536"));
        assert!(json.contains("\"nlist\": 16384"));
        assert!(json.contains("\"nprobe\": 128"));
        assert!(json.contains("\"pq_segments\": 64"));
        assert!(json.contains("\"pq_bits\": 8"));
        assert!(json.contains("\"gpu_id\": 0"));
        assert!(json.contains("\"use_float16\": true"));
        assert!(json.contains("\"min_train_vectors\": 4194304"));
    }

    #[test]
    fn test_pq_bits_type_is_u8() {
        let config = IndexConfig::default();
        // This is a compile-time check - if pq_bits is not u8, this won't compile
        let _: u8 = config.pq_bits;
    }

    #[test]
    fn test_hyperbolic_config_default() {
        let config = HyperbolicConfig::default();

        // Verify all 4 fields
        assert_eq!(config.dim, 64, "Default dim must be 64");
        assert_eq!(config.curvature, -1.0, "Default curvature must be -1.0");
        assert_eq!(config.eps, 1e-7, "Default eps must be 1e-7");
        assert!((config.max_norm - 0.99999).abs() < 1e-10, "Default max_norm must be 1.0 - 1e-5");

        // Invariants
        assert!(config.curvature < 0.0, "Curvature must be negative");
        assert!(config.max_norm < 1.0, "Max norm must be < 1.0");
        assert!(config.max_norm > 0.0, "Max norm must be positive");
        assert!(config.eps > 0.0, "Eps must be positive");
    }

    #[test]
    fn test_hyperbolic_config_with_curvature() {
        let config = HyperbolicConfig::with_curvature(-0.5);
        assert_eq!(config.curvature, -0.5);
        assert_eq!(config.dim, 64); // defaults preserved
        assert_eq!(config.eps, 1e-7);
    }

    #[test]
    fn test_hyperbolic_config_abs_curvature() {
        let config = HyperbolicConfig::default();
        assert_eq!(config.abs_curvature(), 1.0);

        let config2 = HyperbolicConfig::with_curvature(-2.5);
        assert_eq!(config2.abs_curvature(), 2.5);
    }

    #[test]
    fn test_hyperbolic_config_scale() {
        let config = HyperbolicConfig::default();
        assert_eq!(config.scale(), 1.0); // sqrt(|-1.0|) = 1.0

        let config2 = HyperbolicConfig::with_curvature(-4.0);
        assert_eq!(config2.scale(), 2.0); // sqrt(|-4.0|) = 2.0
    }

    #[test]
    fn test_hyperbolic_config_serialization_roundtrip() {
        let config = HyperbolicConfig::default();
        let json = serde_json::to_string(&config).expect("Serialization failed");
        let deserialized: HyperbolicConfig = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_hyperbolic_config_json_fields() {
        let config = HyperbolicConfig::default();
        let json = serde_json::to_string_pretty(&config).expect("Serialization failed");

        // Verify all 4 fields appear in JSON
        assert!(json.contains("\"dim\":"), "JSON must contain dim field");
        assert!(json.contains("\"curvature\":"), "JSON must contain curvature field");
        assert!(json.contains("\"eps\":"), "JSON must contain eps field");
        assert!(json.contains("\"max_norm\":"), "JSON must contain max_norm field");
    }

    // ============ Validation Tests ============

    #[test]
    fn test_validate_default_passes() {
        let config = HyperbolicConfig::default();
        assert!(config.validate().is_ok(), "Default config must be valid");
    }

    #[test]
    fn test_validate_dim_zero_fails() {
        let config = HyperbolicConfig {
            dim: 0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("dim"), "Error should mention 'dim'");
        assert!(err_msg.contains("positive"), "Error should mention 'positive'");
    }

    #[test]
    fn test_validate_curvature_zero_fails() {
        let config = HyperbolicConfig {
            curvature: 0.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("curvature"), "Error should mention 'curvature'");
        assert!(err_msg.contains("negative"), "Error should mention 'negative'");
    }

    #[test]
    fn test_validate_curvature_positive_fails() {
        let config = HyperbolicConfig {
            curvature: 1.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("1"), "Error should include the actual value");
    }

    #[test]
    fn test_validate_curvature_nan_fails() {
        let config = HyperbolicConfig {
            curvature: f32::NAN,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("NaN"), "Error should mention 'NaN'");
    }

    #[test]
    fn test_validate_eps_zero_fails() {
        let config = HyperbolicConfig {
            eps: 0.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("eps"), "Error should mention 'eps'");
    }

    #[test]
    fn test_validate_eps_negative_fails() {
        let config = HyperbolicConfig {
            eps: -1e-7,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_max_norm_zero_fails() {
        let config = HyperbolicConfig {
            max_norm: 0.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("max_norm"), "Error should mention 'max_norm'");
    }

    #[test]
    fn test_validate_max_norm_one_fails() {
        let config = HyperbolicConfig {
            max_norm: 1.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err(), "max_norm=1.0 is ON boundary, not inside ball");
    }

    #[test]
    fn test_validate_max_norm_greater_than_one_fails() {
        let config = HyperbolicConfig {
            max_norm: 1.5,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_max_norm_negative_fails() {
        let config = HyperbolicConfig {
            max_norm: -0.5,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_custom_valid_curvature() {
        // Various valid negative curvatures
        for c in [-0.1, -0.5, -1.0, -2.0, -10.0] {
            let config = HyperbolicConfig::with_curvature(c);
            assert!(config.validate().is_ok(), "curvature {} should be valid", c);
        }
    }

    #[test]
    fn test_try_with_curvature_valid() {
        let config = HyperbolicConfig::try_with_curvature(-0.5).unwrap();
        assert_eq!(config.curvature, -0.5);
        assert_eq!(config.dim, 64); // default
    }

    #[test]
    fn test_try_with_curvature_invalid() {
        assert!(HyperbolicConfig::try_with_curvature(0.0).is_err());
        assert!(HyperbolicConfig::try_with_curvature(1.0).is_err());
        assert!(HyperbolicConfig::try_with_curvature(f32::NAN).is_err());
    }

    #[test]
    fn test_validate_fail_fast_order() {
        // When multiple fields are invalid, should fail on first check (dim)
        let config = HyperbolicConfig {
            dim: 0,
            curvature: 1.0,  // also invalid
            eps: -1.0,       // also invalid
            max_norm: 2.0,   // also invalid
        };
        let err_msg = config.validate().unwrap_err().to_string();
        assert!(err_msg.contains("dim"), "Should fail on dim first");
    }

    #[test]
    fn test_validate_boundary_values() {
        // Test values very close to boundaries
        let barely_valid = HyperbolicConfig {
            dim: 1,
            curvature: -1e-10,  // tiny but negative
            eps: 1e-10,         // tiny but positive
            max_norm: 0.9999999, // close to 1 but not 1
        };
        assert!(barely_valid.validate().is_ok());
    }

    // ============ ConeConfig Tests ============

    #[test]
    fn test_cone_config_default_values() {
        let config = ConeConfig::default();

        // Verify all 5 fields match spec
        assert_eq!(config.min_aperture, 0.1, "min_aperture must be 0.1");
        assert_eq!(config.max_aperture, 1.5, "max_aperture must be 1.5");
        assert_eq!(config.base_aperture, 1.0, "base_aperture must be 1.0");
        assert_eq!(config.aperture_decay, 0.85, "aperture_decay must be 0.85");
        assert_eq!(config.membership_threshold, 0.7, "membership_threshold must be 0.7");
    }

    #[test]
    fn test_cone_config_field_constraints() {
        let config = ConeConfig::default();

        // Verify logical relationships
        assert!(config.min_aperture > 0.0, "min_aperture must be positive");
        assert!(config.max_aperture > config.min_aperture, "max > min");
        assert!(config.base_aperture >= config.min_aperture, "base >= min");
        assert!(config.base_aperture <= config.max_aperture, "base <= max");
        assert!(config.aperture_decay > 0.0 && config.aperture_decay < 1.0, "decay in (0,1)");
        assert!(config.membership_threshold > 0.0 && config.membership_threshold < 1.0, "threshold in (0,1)");
    }

    #[test]
    fn test_compute_aperture_depth_zero() {
        let config = ConeConfig::default();
        // depth=0: base_aperture * 0.85^0 = 1.0 * 1 = 1.0
        assert_eq!(config.compute_aperture(0), 1.0);
    }

    #[test]
    fn test_compute_aperture_depth_one() {
        let config = ConeConfig::default();
        // depth=1: 1.0 * 0.85^1 = 0.85
        let result = config.compute_aperture(1);
        assert!((result - 0.85).abs() < 1e-6, "Expected 0.85, got {}", result);
    }

    #[test]
    fn test_compute_aperture_depth_two() {
        let config = ConeConfig::default();
        // depth=2: 1.0 * 0.85^2 = 0.7225
        let result = config.compute_aperture(2);
        assert!((result - 0.7225).abs() < 1e-6, "Expected 0.7225, got {}", result);
    }

    #[test]
    fn test_compute_aperture_clamps_to_min() {
        let config = ConeConfig::default();
        // Very deep: should clamp to min_aperture = 0.1
        // 1.0 * 0.85^100 ≈ 3.6e-8, clamped to 0.1
        assert_eq!(config.compute_aperture(100), 0.1);
    }

    #[test]
    fn test_compute_aperture_clamps_to_max() {
        // Config where base > max (shouldn't happen in practice but test clamping)
        let config = ConeConfig {
            min_aperture: 0.1,
            max_aperture: 0.5,
            base_aperture: 1.0,  // Exceeds max
            aperture_decay: 0.85,
            membership_threshold: 0.7,
        };
        // depth=0: raw=1.0, clamped to max=0.5
        assert_eq!(config.compute_aperture(0), 0.5);
    }

    #[test]
    fn test_cone_validate_default_passes() {
        let config = ConeConfig::default();
        assert!(config.validate().is_ok(), "Default config must be valid");
    }

    #[test]
    fn test_cone_validate_min_aperture_zero_fails() {
        let config = ConeConfig {
            min_aperture: 0.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("min_aperture"));
    }

    #[test]
    fn test_cone_validate_min_aperture_negative_fails() {
        let config = ConeConfig {
            min_aperture: -0.1,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cone_validate_max_less_than_min_fails() {
        let config = ConeConfig {
            min_aperture: 1.0,
            max_aperture: 0.5,  // Less than min
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_aperture"));
    }

    #[test]
    fn test_cone_validate_max_equals_min_fails() {
        let config = ConeConfig {
            min_aperture: 0.5,
            max_aperture: 0.5,  // Equal, should fail (must be greater)
            base_aperture: 0.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cone_validate_decay_zero_fails() {
        let config = ConeConfig {
            aperture_decay: 0.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("aperture_decay"));
    }

    #[test]
    fn test_cone_validate_decay_one_fails() {
        let config = ConeConfig {
            aperture_decay: 1.0,  // Boundary, excluded
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cone_validate_decay_greater_than_one_fails() {
        let config = ConeConfig {
            aperture_decay: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cone_validate_threshold_zero_fails() {
        let config = ConeConfig {
            membership_threshold: 0.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("membership_threshold"));
    }

    #[test]
    fn test_cone_validate_threshold_one_fails() {
        let config = ConeConfig {
            membership_threshold: 1.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cone_validate_nan_fields_fail() {
        // Test each field with NaN
        let configs = [
            ConeConfig { min_aperture: f32::NAN, ..Default::default() },
            ConeConfig { max_aperture: f32::NAN, ..Default::default() },
            ConeConfig { base_aperture: f32::NAN, ..Default::default() },
            ConeConfig { aperture_decay: f32::NAN, ..Default::default() },
            ConeConfig { membership_threshold: f32::NAN, ..Default::default() },
        ];

        for (i, config) in configs.iter().enumerate() {
            assert!(
                config.validate().is_err(),
                "Config {} with NaN should fail validation", i
            );
        }
    }

    #[test]
    fn test_cone_config_serialization_roundtrip() {
        let config = ConeConfig::default();
        let json = serde_json::to_string(&config).expect("Serialization failed");
        let deserialized: ConeConfig = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_cone_config_json_has_all_fields() {
        let config = ConeConfig::default();
        let json = serde_json::to_string_pretty(&config).expect("Serialization failed");

        // Verify all 5 fields appear in JSON
        assert!(json.contains("\"min_aperture\":"), "JSON must contain min_aperture");
        assert!(json.contains("\"max_aperture\":"), "JSON must contain max_aperture");
        assert!(json.contains("\"base_aperture\":"), "JSON must contain base_aperture");
        assert!(json.contains("\"aperture_decay\":"), "JSON must contain aperture_decay");
        assert!(json.contains("\"membership_threshold\":"), "JSON must contain membership_threshold");
    }

    #[test]
    fn test_cone_config_equality() {
        let a = ConeConfig::default();
        let b = ConeConfig::default();
        assert_eq!(a, b, "Two default configs must be equal");

        let c = ConeConfig {
            min_aperture: 0.2,  // Different
            ..Default::default()
        };
        assert_ne!(a, c, "Different configs must not be equal");
    }

    #[test]
    fn test_cone_validate_base_below_min_fails() {
        let config = ConeConfig {
            min_aperture: 0.5,
            max_aperture: 1.5,
            base_aperture: 0.3,  // Below min
            aperture_decay: 0.85,
            membership_threshold: 0.7,
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("base_aperture"));
    }

    #[test]
    fn test_cone_validate_base_above_max_fails() {
        let config = ConeConfig {
            min_aperture: 0.1,
            max_aperture: 0.8,
            base_aperture: 1.0,  // Above max
            aperture_decay: 0.85,
            membership_threshold: 0.7,
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("base_aperture"));
    }
}
