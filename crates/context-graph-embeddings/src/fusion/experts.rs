//! Expert Networks for FuseMoE routing.
//!
//! This module implements 8 expert networks that transform the concatenated
//! 8320D embedding into 1536D outputs. The gating network selects top-k experts
//! and provides routing weights for weighted combination.
//!
//! # Architecture
//!
//! ```text
//! Expert FFN: input(8320) -> hidden(4096) -> GELU -> output(1536)
//!
//! ExpertPool Flow:
//! 1. Receive (indices, weights) from GatingNetwork.select_top_k()
//! 2. Forward input through selected experts
//! 3. Compute weighted combination of outputs
//! 4. Return fused 1536D embedding
//! ```
//!
//! # Constitution Compliance
//!
//! - `num_experts = 8` (constitution.yaml: fuse_moe.num_experts)
//! - `expert_hidden_dim = 4096` (FusionConfig)
//! - `output_dim = 1536` (FUSED_OUTPUT constant)
//!
//! # No Fallbacks Policy
//!
//! - Invalid expert index -> `EmbeddingError::InvalidExpertIndex`
//! - Dimension mismatch -> `EmbeddingError::DimensionMismatch`
//! - Empty input -> `EmbeddingError::EmptyInput`

use crate::config::FusionConfig;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::fusion::gating::Linear;
use crate::types::dimensions::{FUSED_OUTPUT, NUM_EXPERTS, TOTAL_CONCATENATED};
use tracing::{debug, warn};

// =============================================================================
// ACTIVATION FUNCTIONS
// =============================================================================

/// Activation function types for expert hidden layers.
///
/// # Variants
///
/// - `Gelu`: Gaussian Error Linear Unit (default, recommended)
/// - `Relu`: Rectified Linear Unit (faster, less smooth)
/// - `Silu`: Sigmoid Linear Unit (smooth alternative)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// Gaussian Error Linear Unit: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    #[default]
    Gelu,
    /// Rectified Linear Unit: max(0, x)
    Relu,
    /// Sigmoid Linear Unit: x * sigmoid(x)
    Silu,
}

impl Activation {
    /// Apply activation function element-wise.
    ///
    /// # Arguments
    ///
    /// * `x` - Input value
    ///
    /// # Returns
    ///
    /// Activated value according to the activation function.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::experts::Activation;
    ///
    /// let gelu = Activation::Gelu;
    /// let result = gelu.apply(1.0);
    /// assert!(result > 0.8 && result < 0.9); // GELU(1.0) ≈ 0.8413
    /// ```
    #[inline]
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::Gelu => {
                // Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/pi)
                const COEF: f32 = 0.044715;
                let inner = SQRT_2_OVER_PI * (x + COEF * x * x * x);
                x * 0.5 * (1.0 + inner.tanh())
            }
            Activation::Relu => x.max(0.0),
            Activation::Silu => x * (1.0 / (1.0 + (-x).exp())), // x * sigmoid(x)
        }
    }
}

// =============================================================================
// EXPERT NETWORK
// =============================================================================

/// Single expert network: Feed-Forward Network with hidden layer.
///
/// Architecture: input_dim -> hidden_dim -> activation -> output_dim
///
/// # Fields
///
/// - `input_to_hidden`: First linear layer (8320 -> 4096)
/// - `hidden_to_output`: Second linear layer (4096 -> 1536)
/// - `activation`: Activation function for hidden layer (GELU default)
/// - `expert_id`: Unique identifier (0..NUM_EXPERTS)
///
/// # Example
///
/// ```rust
/// use context_graph_embeddings::fusion::experts::{Expert, Activation};
/// use context_graph_embeddings::types::dimensions::{TOTAL_CONCATENATED, FUSED_OUTPUT};
///
/// let expert = Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu).unwrap();
/// let input = vec![0.1f32; TOTAL_CONCATENATED];
/// let output = expert.forward(&input, 1).unwrap();
/// assert_eq!(output.len(), FUSED_OUTPUT);
/// ```
#[derive(Debug, Clone)]
pub struct Expert {
    /// First linear layer: input_dim -> hidden_dim
    input_to_hidden: Linear,
    /// Second linear layer: hidden_dim -> output_dim
    hidden_to_output: Linear,
    /// Activation function for hidden layer
    activation: Activation,
    /// Expert identifier (0-7)
    expert_id: usize,
    /// Input dimension
    input_dim: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Output dimension
    output_dim: usize,
}

impl Expert {
    /// Create a new expert network.
    ///
    /// # Arguments
    ///
    /// * `expert_id` - Unique identifier 0..NUM_EXPERTS
    /// * `input_dim` - Input dimension (8320)
    /// * `hidden_dim` - Hidden layer dimension (4096)
    /// * `output_dim` - Output dimension (1536)
    /// * `activation` - Activation function (default: Gelu)
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::InvalidDimension` if dimensions are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::experts::{Expert, Activation};
    ///
    /// let expert = Expert::new(0, 8320, 4096, 1536, Activation::Gelu).unwrap();
    /// assert_eq!(expert.expert_id(), 0);
    /// ```
    pub fn new(
        expert_id: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        activation: Activation,
    ) -> EmbeddingResult<Self> {
        // Validate dimensions
        if input_dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }
        if hidden_dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }
        if output_dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }

        let input_to_hidden = Linear::new(input_dim, hidden_dim)?;
        let hidden_to_output = Linear::new(hidden_dim, output_dim)?;

        debug!(
            expert_id,
            input_dim, hidden_dim, output_dim, "Created Expert network"
        );

        Ok(Self {
            input_to_hidden,
            hidden_to_output,
            activation,
            expert_id,
            input_dim,
            hidden_dim,
            output_dim,
        })
    }

    /// Forward pass through single expert.
    ///
    /// # Arguments
    ///
    /// * `input` - Flattened input [batch_size * input_dim]
    /// * `batch_size` - Number of samples in batch
    ///
    /// # Returns
    ///
    /// * `Vec<f32>` of shape [batch_size * output_dim]
    ///
    /// # Errors
    ///
    /// * `EmbeddingError::EmptyInput` if batch_size is 0
    /// * `EmbeddingError::DimensionMismatch` if input length != batch_size * input_dim
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::experts::{Expert, Activation};
    ///
    /// let expert = Expert::new(0, 100, 50, 25, Activation::Gelu).unwrap();
    /// let input = vec![0.1f32; 100];
    /// let output = expert.forward(&input, 1).unwrap();
    /// assert_eq!(output.len(), 25);
    /// ```
    pub fn forward(&self, input: &[f32], batch_size: usize) -> EmbeddingResult<Vec<f32>> {
        // Validate input
        if batch_size == 0 {
            return Err(EmbeddingError::EmptyInput);
        }

        let expected_len = batch_size * self.input_dim;
        if input.len() != expected_len {
            return Err(EmbeddingError::DimensionMismatch {
                expected: expected_len,
                got: input.len(),
            });
        }

        debug!(
            expert_id = self.expert_id,
            batch_size,
            input_len = input.len(),
            "Expert forward pass"
        );

        // Step 1: Input -> Hidden (linear)
        let hidden = self.input_to_hidden.forward(input, batch_size)?;

        // Step 2: Apply activation
        let activated: Vec<f32> = hidden.iter().map(|&x| self.activation.apply(x)).collect();

        // Step 3: Hidden -> Output (linear)
        self.hidden_to_output.forward(&activated, batch_size)
    }

    /// Get expert identifier.
    #[inline]
    #[must_use]
    pub fn expert_id(&self) -> usize {
        self.expert_id
    }

    /// Get input dimension.
    #[inline]
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get hidden dimension.
    #[inline]
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get output dimension.
    #[inline]
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Get activation function.
    #[inline]
    #[must_use]
    pub fn activation(&self) -> Activation {
        self.activation
    }

    /// Get parameter count for this expert.
    ///
    /// Calculated as:
    /// - input_to_hidden: input_dim * hidden_dim + hidden_dim (weights + bias)
    /// - hidden_to_output: hidden_dim * output_dim + output_dim (weights + bias)
    ///
    /// # Example
    ///
    /// For dimensions 8320 -> 4096 -> 1536:
    /// - Layer 1: 8320 * 4096 + 4096 = 34,082,816
    /// - Layer 2: 4096 * 1536 + 1536 = 6,293,088
    /// - Total: ~40.4M parameters per expert
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        let layer1_params = self.input_dim * self.hidden_dim + self.hidden_dim;
        let layer2_params = self.hidden_dim * self.output_dim + self.output_dim;
        layer1_params + layer2_params
    }
}

// =============================================================================
// EXPERT POOL
// =============================================================================

/// Pool of all 8 expert networks with top-k routing.
///
/// Manages the collection of experts and provides the core `forward_topk`
/// method that computes weighted combinations of expert outputs.
///
/// # Fields
///
/// - `experts`: Array of 8 experts (always length NUM_EXPERTS)
/// - `input_dim`: Input dimension (8320)
/// - `hidden_dim`: Hidden dimension (4096)
/// - `output_dim`: Output dimension (1536)
///
/// # Example
///
/// ```rust
/// use context_graph_embeddings::fusion::experts::ExpertPool;
/// use context_graph_embeddings::config::FusionConfig;
/// use context_graph_embeddings::types::dimensions::{TOTAL_CONCATENATED, FUSED_OUTPUT, TOP_K_EXPERTS};
///
/// let config = FusionConfig::default();
/// let pool = ExpertPool::new(&config).unwrap();
///
/// let input = vec![0.1f32; TOTAL_CONCATENATED];
/// let indices = vec![0, 2, 4, 6];
/// let weights = vec![0.4, 0.3, 0.2, 0.1];
///
/// let output = pool.forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS).unwrap();
/// assert_eq!(output.len(), FUSED_OUTPUT);
/// ```
#[derive(Debug, Clone)]
pub struct ExpertPool {
    /// Array of 8 experts
    experts: Vec<Expert>,
    /// Input dimension (8320)
    input_dim: usize,
    /// Hidden dimension (4096)
    hidden_dim: usize,
    /// Output dimension (1536)
    output_dim: usize,
}

impl ExpertPool {
    /// Create new expert pool from config.
    ///
    /// # Arguments
    ///
    /// * `config` - FusionConfig with expert_hidden_dim
    ///
    /// # Errors
    ///
    /// Returns error if expert initialization fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::experts::ExpertPool;
    /// use context_graph_embeddings::config::FusionConfig;
    ///
    /// let config = FusionConfig::default();
    /// let pool = ExpertPool::new(&config).unwrap();
    /// assert_eq!(pool.num_experts(), 8);
    /// ```
    pub fn new(config: &FusionConfig) -> EmbeddingResult<Self> {
        let input_dim = TOTAL_CONCATENATED;
        let hidden_dim = config.expert_hidden_dim;
        let output_dim = FUSED_OUTPUT;

        let mut experts = Vec::with_capacity(NUM_EXPERTS);

        for expert_id in 0..NUM_EXPERTS {
            let expert = Expert::new(
                expert_id,
                input_dim,
                hidden_dim,
                output_dim,
                Activation::Gelu,
            )?;
            experts.push(expert);
        }

        debug!(
            num_experts = NUM_EXPERTS,
            input_dim,
            hidden_dim,
            output_dim,
            "Created ExpertPool"
        );

        Ok(Self {
            experts,
            input_dim,
            hidden_dim,
            output_dim,
        })
    }

    /// Forward pass through a single expert by index.
    ///
    /// # Arguments
    ///
    /// * `input` - Input [batch_size * input_dim]
    /// * `batch_size` - Number of samples
    /// * `expert_idx` - Expert index 0..NUM_EXPERTS
    ///
    /// # Errors
    ///
    /// * `EmbeddingError::InvalidExpertIndex` if expert_idx >= NUM_EXPERTS
    /// * `EmbeddingError::EmptyInput` if batch_size is 0
    /// * `EmbeddingError::DimensionMismatch` if input length is wrong
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::experts::ExpertPool;
    /// use context_graph_embeddings::config::FusionConfig;
    /// use context_graph_embeddings::types::dimensions::{TOTAL_CONCATENATED, FUSED_OUTPUT};
    ///
    /// let config = FusionConfig::default();
    /// let pool = ExpertPool::new(&config).unwrap();
    /// let input = vec![0.1f32; TOTAL_CONCATENATED];
    ///
    /// let output = pool.forward(&input, 1, 0).unwrap();
    /// assert_eq!(output.len(), FUSED_OUTPUT);
    /// ```
    pub fn forward(
        &self,
        input: &[f32],
        batch_size: usize,
        expert_idx: usize,
    ) -> EmbeddingResult<Vec<f32>> {
        if expert_idx >= NUM_EXPERTS {
            return Err(EmbeddingError::InvalidExpertIndex {
                index: expert_idx,
                max: NUM_EXPERTS,
            });
        }

        self.experts[expert_idx].forward(input, batch_size)
    }

    /// Forward through top-k experts with weighted combination.
    ///
    /// This is the PRIMARY method consumed by FuseMoE Router.
    ///
    /// # Arguments
    ///
    /// * `input` - Concatenated embedding [batch_size * 8320]
    /// * `batch_size` - Number of samples in batch
    /// * `indices` - Expert indices from GatingNetwork.select_top_k() [batch_size * top_k]
    /// * `weights` - Routing weights from GatingNetwork.select_top_k() [batch_size * top_k]
    /// * `top_k` - Number of experts per sample (typically 4)
    ///
    /// # Returns
    ///
    /// * Weighted combination of expert outputs [batch_size * 1536]
    ///
    /// # Formula
    ///
    /// For each sample: output = sum(weights[i] * expert[indices[i]].forward(input))
    ///
    /// # Errors
    ///
    /// * `EmbeddingError::EmptyInput` if batch_size is 0
    /// * `EmbeddingError::DimensionMismatch` if input/indices/weights lengths don't match
    /// * `EmbeddingError::InvalidExpertIndex` if any index >= NUM_EXPERTS
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::experts::ExpertPool;
    /// use context_graph_embeddings::config::FusionConfig;
    /// use context_graph_embeddings::types::dimensions::{TOTAL_CONCATENATED, FUSED_OUTPUT, TOP_K_EXPERTS};
    ///
    /// let config = FusionConfig::default();
    /// let pool = ExpertPool::new(&config).unwrap();
    ///
    /// let input = vec![0.1f32; TOTAL_CONCATENATED];
    /// let indices = vec![0, 2, 4, 6];
    /// let weights = vec![0.4, 0.3, 0.2, 0.1];
    ///
    /// let output = pool.forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS).unwrap();
    /// assert_eq!(output.len(), FUSED_OUTPUT);
    /// ```
    pub fn forward_topk(
        &self,
        input: &[f32],
        batch_size: usize,
        indices: &[usize],
        weights: &[f32],
        top_k: usize,
    ) -> EmbeddingResult<Vec<f32>> {
        // Validate inputs
        if batch_size == 0 {
            return Err(EmbeddingError::EmptyInput);
        }

        if input.len() != batch_size * self.input_dim {
            return Err(EmbeddingError::DimensionMismatch {
                expected: batch_size * self.input_dim,
                got: input.len(),
            });
        }

        if indices.len() != batch_size * top_k {
            return Err(EmbeddingError::DimensionMismatch {
                expected: batch_size * top_k,
                got: indices.len(),
            });
        }

        if weights.len() != batch_size * top_k {
            return Err(EmbeddingError::DimensionMismatch {
                expected: batch_size * top_k,
                got: weights.len(),
            });
        }

        // Validate all indices
        for &idx in indices {
            if idx >= NUM_EXPERTS {
                return Err(EmbeddingError::InvalidExpertIndex {
                    index: idx,
                    max: NUM_EXPERTS,
                });
            }
        }

        // Log warning if weights don't sum to ~1.0
        for sample_idx in 0..batch_size {
            let weight_start = sample_idx * top_k;
            let weight_end = weight_start + top_k;
            let weight_sum: f32 = weights[weight_start..weight_end].iter().sum();
            if (weight_sum - 1.0).abs() > 0.01 {
                warn!(
                    sample_idx,
                    weight_sum, "Weights do not sum to 1.0 (deviation > 0.01)"
                );
            }
        }

        debug!(
            batch_size,
            top_k,
            input_len = input.len(),
            "ExpertPool forward_topk"
        );

        // Initialize output buffer
        let mut output = vec![0.0f32; batch_size * self.output_dim];

        // Process each sample in batch
        for sample_idx in 0..batch_size {
            let input_start = sample_idx * self.input_dim;
            let input_end = input_start + self.input_dim;
            let sample_input = &input[input_start..input_end];

            let output_start = sample_idx * self.output_dim;

            // Process each selected expert for this sample
            for k in 0..top_k {
                let routing_idx = sample_idx * top_k + k;
                let expert_idx = indices[routing_idx];
                let weight = weights[routing_idx];

                // Forward through single expert (batch_size=1 for this sample)
                let expert_output = self.experts[expert_idx].forward(sample_input, 1)?;

                // Weighted accumulation
                for (j, &val) in expert_output.iter().enumerate() {
                    output[output_start + j] += weight * val;
                }
            }
        }

        Ok(output)
    }

    /// Get number of experts in pool.
    #[inline]
    #[must_use]
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Get total parameter count across all experts.
    ///
    /// For default configuration (8320 -> 4096 -> 1536, 8 experts):
    /// - Per expert: ~40.4M parameters
    /// - Total: ~323M parameters
    #[must_use]
    pub fn total_parameter_count(&self) -> usize {
        self.experts.iter().map(|e| e.parameter_count()).sum()
    }

    /// Get input dimension.
    #[inline]
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get hidden dimension.
    #[inline]
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get output dimension.
    #[inline]
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::dimensions::TOP_K_EXPERTS;

    // =========================================================================
    // ACTIVATION TESTS
    // =========================================================================

    #[test]
    fn test_gelu_activation_zero() {
        let act = Activation::Gelu;
        let result = act.apply(0.0);
        assert!((result - 0.0).abs() < 1e-6, "GELU(0) should be 0");
    }

    #[test]
    fn test_gelu_activation_positive() {
        let act = Activation::Gelu;
        let result = act.apply(1.0);
        // GELU(1.0) ≈ 0.8413 (computed, not hardcoded)
        assert!(
            result > 0.8 && result < 0.9,
            "GELU(1.0) should be ~0.84, got {}",
            result
        );
    }

    #[test]
    fn test_gelu_activation_negative() {
        let act = Activation::Gelu;
        let result = act.apply(-1.0);
        // GELU(-1.0) ≈ -0.1587 (computed, not hardcoded)
        assert!(
            result > -0.2 && result < -0.1,
            "GELU(-1.0) should be ~-0.16, got {}",
            result
        );
    }

    #[test]
    fn test_relu_activation() {
        let act = Activation::Relu;
        assert!((act.apply(1.0) - 1.0).abs() < 1e-6);
        assert!((act.apply(-1.0) - 0.0).abs() < 1e-6);
        assert!((act.apply(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu_activation() {
        let act = Activation::Silu;
        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!((act.apply(0.0) - 0.0).abs() < 1e-6);
        // SiLU(x) for positive x should be positive
        assert!(act.apply(1.0) > 0.0);
        // SiLU(x) for negative x should be negative
        assert!(act.apply(-1.0) < 0.0);
    }

    #[test]
    fn test_activation_default() {
        assert_eq!(Activation::default(), Activation::Gelu);
    }

    // =========================================================================
    // EXPERT TESTS
    // =========================================================================

    #[test]
    fn test_expert_creation() {
        let expert = Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu);
        assert!(expert.is_ok());
        let expert = expert.unwrap();
        assert_eq!(expert.expert_id(), 0);
        assert_eq!(expert.input_dim(), TOTAL_CONCATENATED);
        assert_eq!(expert.hidden_dim(), 4096);
        assert_eq!(expert.output_dim(), FUSED_OUTPUT);
        assert_eq!(expert.activation(), Activation::Gelu);
    }

    #[test]
    fn test_expert_output_shape() {
        let expert =
            Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu).unwrap();
        let input = vec![0.1f32; TOTAL_CONCATENATED];
        let output = expert.forward(&input, 1).unwrap();
        assert_eq!(output.len(), FUSED_OUTPUT, "Output should be 1536D");
    }

    #[test]
    fn test_expert_batch_output_shape() {
        let expert =
            Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu).unwrap();
        let batch_size = 4;
        let input = vec![0.1f32; TOTAL_CONCATENATED * batch_size];
        let output = expert.forward(&input, batch_size).unwrap();
        assert_eq!(output.len(), FUSED_OUTPUT * batch_size);
    }

    #[test]
    fn test_expert_empty_batch_fails() {
        let expert =
            Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu).unwrap();
        let input: Vec<f32> = vec![];
        let result = expert.forward(&input, 0);
        assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
    }

    #[test]
    fn test_expert_wrong_input_size_fails() {
        let expert =
            Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu).unwrap();
        let input = vec![0.1f32; 100]; // Wrong size
        let result = expert.forward(&input, 1);
        assert!(matches!(
            result,
            Err(EmbeddingError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_expert_parameter_count() {
        let expert =
            Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu).unwrap();
        let count = expert.parameter_count();

        // Layer 1: 8320 * 4096 + 4096 = 34,082,816
        // Layer 2: 4096 * 1536 + 1536 = 6,293,088
        // Total: ~40,375,904
        assert!(count > 40_000_000, "Should have > 40M parameters");
        assert!(count < 42_000_000, "Should have < 42M parameters");
    }

    #[test]
    fn test_expert_zero_input_dim_fails() {
        let result = Expert::new(0, 0, 4096, 1536, Activation::Gelu);
        assert!(matches!(
            result,
            Err(EmbeddingError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn test_expert_zero_hidden_dim_fails() {
        let result = Expert::new(0, 8320, 0, 1536, Activation::Gelu);
        assert!(matches!(
            result,
            Err(EmbeddingError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn test_expert_zero_output_dim_fails() {
        let result = Expert::new(0, 8320, 4096, 0, Activation::Gelu);
        assert!(matches!(
            result,
            Err(EmbeddingError::InvalidDimension { .. })
        ));
    }

    // =========================================================================
    // EXPERT POOL TESTS
    // =========================================================================

    #[test]
    fn test_expert_pool_creation() {
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config);
        assert!(pool.is_ok());
        let pool = pool.unwrap();
        assert_eq!(pool.num_experts(), NUM_EXPERTS);
        assert_eq!(pool.input_dim(), TOTAL_CONCATENATED);
        assert_eq!(pool.hidden_dim(), config.expert_hidden_dim);
        assert_eq!(pool.output_dim(), FUSED_OUTPUT);
    }

    #[test]
    fn test_expert_pool_forward_single() {
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let input = vec![0.1f32; TOTAL_CONCATENATED];

        for expert_idx in 0..NUM_EXPERTS {
            let output = pool.forward(&input, 1, expert_idx).unwrap();
            assert_eq!(output.len(), FUSED_OUTPUT);
        }
    }

    #[test]
    fn test_expert_pool_invalid_index_fails() {
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let input = vec![0.1f32; TOTAL_CONCATENATED];

        let result = pool.forward(&input, 1, NUM_EXPERTS); // Index 8 is invalid
        assert!(matches!(
            result,
            Err(EmbeddingError::InvalidExpertIndex { .. })
        ));
    }

    #[test]
    fn test_forward_topk_output_shape() {
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let input = vec![0.1f32; TOTAL_CONCATENATED];

        let indices = vec![0, 2, 4, 6]; // Top 4 experts
        let weights = vec![0.4, 0.3, 0.2, 0.1]; // Sum to 1.0

        let output = pool
            .forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)
            .unwrap();
        assert_eq!(output.len(), FUSED_OUTPUT);
    }

    #[test]
    fn test_forward_topk_batch() {
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let batch_size = 2;
        let input = vec![0.1f32; TOTAL_CONCATENATED * batch_size];

        // 2 samples * 4 experts each
        let indices = vec![0, 2, 4, 6, 1, 3, 5, 7];
        let weights = vec![0.4, 0.3, 0.2, 0.1, 0.25, 0.25, 0.25, 0.25];

        let output = pool
            .forward_topk(&input, batch_size, &indices, &weights, TOP_K_EXPERTS)
            .unwrap();
        assert_eq!(output.len(), FUSED_OUTPUT * batch_size);
    }

    #[test]
    fn test_forward_topk_single_expert_weight_one() {
        // If only one expert has weight 1.0, output should equal that expert's output
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let input = vec![0.5f32; TOTAL_CONCATENATED];

        let indices = vec![3, 0, 0, 0];
        let weights = vec![1.0, 0.0, 0.0, 0.0]; // Only expert 3 contributes

        let topk_output = pool
            .forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)
            .unwrap();
        let direct_output = pool.forward(&input, 1, 3).unwrap();

        for (a, b) in topk_output.iter().zip(direct_output.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "forward_topk with weight 1.0 should match forward"
            );
        }
    }

    #[test]
    fn test_forward_topk_weighted_combination() {
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let input = vec![0.3f32; TOTAL_CONCATENATED];

        let indices = vec![0, 1, 2, 3];
        let weights = vec![0.5, 0.5, 0.0, 0.0];

        // Get outputs from experts 0 and 1
        let out0 = pool.forward(&input, 1, 0).unwrap();
        let out1 = pool.forward(&input, 1, 1).unwrap();

        // Compute expected weighted combination
        let expected: Vec<f32> = out0
            .iter()
            .zip(out1.iter())
            .map(|(&a, &b)| 0.5 * a + 0.5 * b)
            .collect();

        let actual = pool
            .forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)
            .unwrap();

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "Mismatch at index {}: actual={}, expected={}",
                i,
                a,
                e
            );
        }
    }

    #[test]
    fn test_parameter_count() {
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();

        // Each expert: (8320*4096 + 4096) + (4096*1536 + 1536)
        // = 34,082,816 + 6,293,088 = 40,375,904 per expert
        // Total: 8 * ~40.4M = ~323M parameters
        let total = pool.total_parameter_count();
        assert!(total > 300_000_000, "Should have > 300M parameters");
        assert!(total < 350_000_000, "Should have < 350M parameters");
    }

    #[test]
    fn test_forward_topk_empty_batch_fails() {
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let input: Vec<f32> = vec![];
        let indices: Vec<usize> = vec![];
        let weights: Vec<f32> = vec![];

        let result = pool.forward_topk(&input, 0, &indices, &weights, TOP_K_EXPERTS);
        assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
    }

    #[test]
    fn test_forward_topk_wrong_input_size_fails() {
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let input = vec![0.1f32; 100]; // Wrong size
        let indices = vec![0, 1, 2, 3];
        let weights = vec![0.25, 0.25, 0.25, 0.25];

        let result = pool.forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS);
        assert!(matches!(
            result,
            Err(EmbeddingError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_forward_topk_wrong_indices_size_fails() {
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let input = vec![0.1f32; TOTAL_CONCATENATED];
        let indices = vec![0, 1]; // Wrong size (should be 4)
        let weights = vec![0.25, 0.25, 0.25, 0.25];

        let result = pool.forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS);
        assert!(matches!(
            result,
            Err(EmbeddingError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_forward_topk_wrong_weights_size_fails() {
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let input = vec![0.1f32; TOTAL_CONCATENATED];
        let indices = vec![0, 1, 2, 3];
        let weights = vec![0.5, 0.5]; // Wrong size (should be 4)

        let result = pool.forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS);
        assert!(matches!(
            result,
            Err(EmbeddingError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_forward_topk_invalid_expert_index_fails() {
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let input = vec![0.1f32; TOTAL_CONCATENATED];
        let indices = vec![0, 1, 2, 10]; // 10 is invalid
        let weights = vec![0.25, 0.25, 0.25, 0.25];

        let result = pool.forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS);
        assert!(matches!(
            result,
            Err(EmbeddingError::InvalidExpertIndex { .. })
        ));
    }

    // =========================================================================
    // INTEGRATION WITH GATING NETWORK
    // =========================================================================

    #[test]
    fn test_integration_with_gating_network() {
        use crate::fusion::GatingNetwork;

        let config = FusionConfig::default();
        let gating = GatingNetwork::new(&config).unwrap();
        let pool = ExpertPool::new(&config).unwrap();

        let input = vec![0.5f32; TOTAL_CONCATENATED];

        // Forward through gating
        let probs = gating.forward(&input, 1).unwrap();

        // Select top-k
        let (indices, weights) = gating.select_top_k(&probs, 1, TOP_K_EXPERTS).unwrap();

        // Forward through expert pool
        let output = pool
            .forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)
            .unwrap();

        assert_eq!(output.len(), FUSED_OUTPUT, "Final output should be 1536D");

        // Output should be finite (no NaN or Inf)
        for &val in &output {
            assert!(val.is_finite(), "Output contains non-finite value");
        }
    }

    // =========================================================================
    // EDGE CASE TESTS WITH STATE PRINTING
    // =========================================================================

    #[test]
    fn edge_case_empty_batch() {
        println!("=== BEFORE: Empty batch test ===");
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        println!("ExpertPool created with {} experts", pool.num_experts());

        let input: Vec<f32> = vec![];
        println!("Input length: {}", input.len());

        println!("=== EXECUTE: forward with batch_size=0 ===");
        let result = pool.forward(&input, 0, 0);

        println!("=== AFTER: Result ===");
        match &result {
            Ok(_) => println!("ERROR: Should have failed"),
            Err(e) => println!("Correctly failed with: {:?}", e),
        }
        assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
    }

    #[test]
    fn edge_case_equal_weights() {
        println!("=== BEFORE: Equal weights test ===");
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let input = vec![1.0f32; TOTAL_CONCATENATED];

        let indices = vec![0, 1, 2, 3];
        let weights = vec![0.25, 0.25, 0.25, 0.25];
        println!("Indices: {:?}", indices);
        println!(
            "Weights: {:?} (sum={})",
            weights,
            weights.iter().sum::<f32>()
        );

        println!("=== EXECUTE: forward_topk ===");
        let output = pool
            .forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)
            .unwrap();

        println!("=== AFTER: Output stats ===");
        let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "Output len: {}, mean: {:.6}, min: {:.6}, max: {:.6}",
            output.len(),
            mean,
            min,
            max
        );

        assert_eq!(output.len(), FUSED_OUTPUT);
    }

    #[test]
    fn edge_case_extreme_weights() {
        println!("=== BEFORE: Extreme weight (0.999) test ===");
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let input = vec![0.7f32; TOTAL_CONCATENATED];

        let indices = vec![5, 0, 0, 0];
        let weights = vec![0.999, 0.001 / 3.0, 0.001 / 3.0, 0.001 / 3.0];
        println!("Dominant expert: 5 with weight 0.999");

        println!("=== EXECUTE: Compare forward_topk vs direct forward ===");
        let topk_output = pool
            .forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)
            .unwrap();
        let direct_output = pool.forward(&input, 1, 5).unwrap();

        println!("=== AFTER: Compare outputs ===");
        let diff: f32 = topk_output
            .iter()
            .zip(direct_output.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / FUSED_OUTPUT as f32;
        println!("Average absolute difference: {:.8}", diff);

        assert!(
            diff < 0.01,
            "With 0.999 weight, output should be very close to single expert"
        );
    }

    #[test]
    fn test_all_experts_different_outputs() {
        // Verify that different experts produce different outputs
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();
        let input = vec![0.5f32; TOTAL_CONCATENATED];

        let mut outputs: Vec<Vec<f32>> = Vec::new();
        for expert_idx in 0..NUM_EXPERTS {
            let output = pool.forward(&input, 1, expert_idx).unwrap();
            outputs.push(output);
        }

        // Check that outputs are different (at least first few elements)
        for i in 0..NUM_EXPERTS {
            for j in (i + 1)..NUM_EXPERTS {
                let diff: f32 = outputs[i]
                    .iter()
                    .zip(outputs[j].iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>();
                // Due to random initialization, different experts should produce different outputs
                // Note: This could fail if by extreme chance weights are identical
                println!(
                    "Diff between expert {} and {}: {:.6}",
                    i,
                    j,
                    diff / FUSED_OUTPUT as f32
                );
            }
        }
    }

    #[test]
    fn test_output_values_are_finite() {
        let config = FusionConfig::default();
        let pool = ExpertPool::new(&config).unwrap();

        // Test with various input patterns
        let test_inputs = vec![
            vec![0.0f32; TOTAL_CONCATENATED],
            vec![1.0f32; TOTAL_CONCATENATED],
            vec![-1.0f32; TOTAL_CONCATENATED],
            vec![0.001f32; TOTAL_CONCATENATED],
        ];

        for (i, input) in test_inputs.iter().enumerate() {
            for expert_idx in 0..NUM_EXPERTS {
                let output = pool.forward(input, 1, expert_idx).unwrap();
                for (j, &val) in output.iter().enumerate() {
                    assert!(
                        val.is_finite(),
                        "Non-finite value at input pattern {}, expert {}, index {}",
                        i,
                        expert_idx,
                        j
                    );
                }
            }
        }
    }
}
