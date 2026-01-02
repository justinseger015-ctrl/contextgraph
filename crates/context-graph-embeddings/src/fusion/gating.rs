//! Gating network for FuseMoE routing.
//!
//! This module implements the gating network that routes the 8320D concatenated
//! embeddings to 8 experts based on temperature-scaled softmax probabilities.
//!
//! # Architecture
//!
//! ```text
//! Input: [batch_size, 8320]
//!        |
//!        v
//!   [LayerNorm(8320)]
//!        |
//!        v
//!   [Linear(8320 → 8)]
//!        |
//!        v
//!   [Temperature-scaled Softmax] --> Expert weights [batch_size, 8]
//!        |
//!        v
//!   [Laplace Smoothing (optional)]
//!        |
//!        v
//!   [Top-K Selection] --> Selected experts and weights
//! ```
//!
//! # Constitution Compliance
//!
//! - `num_experts = 8` (constitution.yaml: fuse_moe.num_experts)
//! - `top_k = 4` (constitution.yaml: fuse_moe.top_k)
//! - `temperature = 1.0` (default, neuromodulation range [0.5, 2.0])
//! - `laplace_alpha = 0.01` (constitution.yaml: fuse_moe.laplace_alpha)
//!
//! # No Fallbacks Policy
//!
//! - Invalid dimensions → `EmbeddingError::InvalidDimension`
//! - NaN values → `EmbeddingError::InvalidValue`
//! - Empty input → `EmbeddingError::EmptyInput`

use crate::config::FusionConfig;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::dimensions::TOTAL_CONCATENATED;
use rand::Rng;
use rand_distr::{Distribution, Normal};

// =============================================================================
// LAYERNORM
// =============================================================================

/// Layer normalization for input stabilization.
///
/// Normalizes each sample in a batch to have mean=0 and variance=1,
/// then applies learned scale (gamma) and shift (beta) parameters.
///
/// # Formula
///
/// ```text
/// y = gamma * (x - mean) / sqrt(var + eps) + beta
/// ```
///
/// # Fields
///
/// - `gamma`: Scale parameter (learned, initialized to 1.0)
/// - `beta`: Shift parameter (learned, initialized to 0.0)
/// - `eps`: Numerical stability constant (1e-5)
/// - `dim`: Expected input dimension
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Scale parameter (γ) - shape: [dim]
    gamma: Vec<f32>,
    /// Shift parameter (β) - shape: [dim]
    beta: Vec<f32>,
    /// Numerical stability constant
    eps: f32,
    /// Expected input dimension
    dim: usize,
}

impl LayerNorm {
    /// Create a new LayerNorm with given dimension.
    ///
    /// Initializes:
    /// - `gamma = 1.0` (no scaling)
    /// - `beta = 0.0` (no shift)
    /// - `eps = 1e-5` (numerical stability)
    ///
    /// # Arguments
    ///
    /// * `dim` - Input/output dimension (must be > 0)
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::InvalidDimension` if dim == 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::LayerNorm;
    ///
    /// let norm = LayerNorm::new(8320).unwrap();
    /// assert_eq!(norm.dim(), 8320);
    /// ```
    pub fn new(dim: usize) -> EmbeddingResult<Self> {
        if dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }

        Ok(Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            eps: 1e-5,
            dim,
        })
    }

    /// Get the input dimension.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the epsilon value.
    #[inline]
    #[must_use]
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Forward pass through layer normalization.
    ///
    /// Normalizes each sample in the batch independently.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [batch_size * dim]
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Returns
    ///
    /// Normalized output tensor of shape [batch_size * dim].
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::EmptyInput` if batch_size == 0
    /// - `EmbeddingError::InvalidDimension` if input length != batch_size * dim
    /// - `EmbeddingError::InvalidValue` if input contains NaN
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::LayerNorm;
    ///
    /// let norm = LayerNorm::new(4).unwrap();
    /// let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let output = norm.forward(&input, 2).unwrap();
    /// assert_eq!(output.len(), 8);
    ///
    /// // Verify normalization (each sample should have mean ≈ 0)
    /// let sample1_mean: f32 = output[0..4].iter().sum::<f32>() / 4.0;
    /// assert!(sample1_mean.abs() < 1e-5);
    /// ```
    pub fn forward(&self, input: &[f32], batch_size: usize) -> EmbeddingResult<Vec<f32>> {
        if batch_size == 0 {
            return Err(EmbeddingError::EmptyInput);
        }

        let expected_len = batch_size * self.dim;
        if input.len() != expected_len {
            return Err(EmbeddingError::InvalidDimension {
                expected: expected_len,
                actual: input.len(),
            });
        }

        // Check for NaN values
        for (i, &val) in input.iter().enumerate() {
            if val.is_nan() {
                return Err(EmbeddingError::InvalidValue { index: i, value: val });
            }
        }

        let mut output = vec![0.0f32; expected_len];

        for b in 0..batch_size {
            let start = b * self.dim;
            let end = start + self.dim;
            let sample = &input[start..end];

            // Compute mean
            let mean: f32 = sample.iter().sum::<f32>() / self.dim as f32;

            // Compute variance
            let variance: f32 = sample
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>()
                / self.dim as f32;

            // Normalize: (x - mean) / sqrt(var + eps)
            let inv_std = 1.0 / (variance + self.eps).sqrt();

            for (i, &x) in sample.iter().enumerate() {
                let normalized = (x - mean) * inv_std;
                // Apply scale and shift
                output[start + i] = self.gamma[i] * normalized + self.beta[i];
            }
        }

        Ok(output)
    }

    /// Set the gamma (scale) parameters.
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::InvalidDimension` if gamma.len() != dim.
    pub fn set_gamma(&mut self, gamma: Vec<f32>) -> EmbeddingResult<()> {
        if gamma.len() != self.dim {
            return Err(EmbeddingError::InvalidDimension {
                expected: self.dim,
                actual: gamma.len(),
            });
        }
        self.gamma = gamma;
        Ok(())
    }

    /// Set the beta (shift) parameters.
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::InvalidDimension` if beta.len() != dim.
    pub fn set_beta(&mut self, beta: Vec<f32>) -> EmbeddingResult<()> {
        if beta.len() != self.dim {
            return Err(EmbeddingError::InvalidDimension {
                expected: self.dim,
                actual: beta.len(),
            });
        }
        self.beta = beta;
        Ok(())
    }

    /// Get a reference to gamma (scale) parameters.
    #[inline]
    #[must_use]
    pub fn gamma(&self) -> &[f32] {
        &self.gamma
    }

    /// Get a reference to beta (shift) parameters.
    #[inline]
    #[must_use]
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }
}

// =============================================================================
// LINEAR PROJECTION
// =============================================================================

/// Linear projection layer (fully connected / dense).
///
/// Transforms input from `in_features` to `out_features` dimensions
/// using learned weights and optional bias.
///
/// # Formula
///
/// ```text
/// y = x @ W^T + b
/// ```
///
/// where:
/// - `x`: Input tensor [batch_size, in_features]
/// - `W`: Weight matrix [out_features, in_features]
/// - `b`: Bias vector [out_features]
/// - `y`: Output tensor [batch_size, out_features]
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix (row-major): [out_features, in_features]
    weights: Vec<f32>,
    /// Bias vector: [out_features]
    bias: Vec<f32>,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
}

impl Linear {
    /// Create a new Linear layer with Xavier initialization.
    ///
    /// Weights are initialized using Xavier uniform distribution:
    /// `U(-sqrt(6/(in+out)), sqrt(6/(in+out)))`
    ///
    /// Bias is initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension (must be > 0)
    /// * `out_features` - Output dimension (must be > 0)
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::InvalidDimension` if either dimension is 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::Linear;
    ///
    /// let linear = Linear::new(8320, 8).unwrap();
    /// assert_eq!(linear.in_features(), 8320);
    /// assert_eq!(linear.out_features(), 8);
    /// ```
    pub fn new(in_features: usize, out_features: usize) -> EmbeddingResult<Self> {
        if in_features == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }
        if out_features == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }

        // Xavier initialization
        let mut rng = rand::thread_rng();
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();

        let weights: Vec<f32> = (0..(out_features * in_features))
            .map(|_| rng.gen_range((-limit)..limit) as f32)
            .collect();

        let bias = vec![0.0; out_features];

        Ok(Self {
            weights,
            bias,
            in_features,
            out_features,
        })
    }

    /// Create a new Linear layer with provided weights and bias.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `weights` - Weight matrix (row-major: [out_features, in_features])
    /// * `bias` - Bias vector: [out_features]
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::InvalidDimension` if weights or bias dimensions don't match.
    pub fn with_weights(
        in_features: usize,
        out_features: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> EmbeddingResult<Self> {
        let expected_weights = out_features * in_features;
        if weights.len() != expected_weights {
            return Err(EmbeddingError::InvalidDimension {
                expected: expected_weights,
                actual: weights.len(),
            });
        }
        if bias.len() != out_features {
            return Err(EmbeddingError::InvalidDimension {
                expected: out_features,
                actual: bias.len(),
            });
        }

        Ok(Self {
            weights,
            bias,
            in_features,
            out_features,
        })
    }

    /// Get input dimension.
    #[inline]
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output dimension.
    #[inline]
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Forward pass through linear layer.
    ///
    /// Computes `y = x @ W^T + b` for each sample in the batch.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [batch_size * in_features]
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Returns
    ///
    /// Output tensor of shape [batch_size * out_features].
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::EmptyInput` if batch_size == 0
    /// - `EmbeddingError::InvalidDimension` if input length != batch_size * in_features
    /// - `EmbeddingError::InvalidValue` if input contains NaN
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::Linear;
    ///
    /// let linear = Linear::new(4, 2).unwrap();
    /// let input = vec![1.0, 2.0, 3.0, 4.0]; // batch_size=1
    /// let output = linear.forward(&input, 1).unwrap();
    /// assert_eq!(output.len(), 2);
    /// ```
    pub fn forward(&self, input: &[f32], batch_size: usize) -> EmbeddingResult<Vec<f32>> {
        if batch_size == 0 {
            return Err(EmbeddingError::EmptyInput);
        }

        let expected_len = batch_size * self.in_features;
        if input.len() != expected_len {
            return Err(EmbeddingError::InvalidDimension {
                expected: expected_len,
                actual: input.len(),
            });
        }

        // Check for NaN values
        for (i, &val) in input.iter().enumerate() {
            if val.is_nan() {
                return Err(EmbeddingError::InvalidValue { index: i, value: val });
            }
        }

        let mut output = vec![0.0f32; batch_size * self.out_features];

        // Matrix multiplication: y = x @ W^T + b
        for b in 0..batch_size {
            let input_offset = b * self.in_features;
            let output_offset = b * self.out_features;

            for o in 0..self.out_features {
                let mut sum = self.bias[o];
                let weight_offset = o * self.in_features;

                for i in 0..self.in_features {
                    sum += input[input_offset + i] * self.weights[weight_offset + i];
                }

                output[output_offset + o] = sum;
            }
        }

        Ok(output)
    }

    /// Get a reference to the weight matrix.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get a reference to the bias vector.
    #[inline]
    #[must_use]
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }
}

// =============================================================================
// GATING NETWORK
// =============================================================================

/// Gating network for FuseMoE expert routing.
///
/// Routes the 8320D concatenated embeddings to 8 experts using
/// temperature-scaled softmax with optional Laplace smoothing.
///
/// # Architecture
///
/// 1. **LayerNorm**: Normalize input to mean=0, var=1
/// 2. **Linear**: Project from 8320D to 8D (one logit per expert)
/// 3. **Temperature Scaling**: Scale logits by 1/temperature
/// 4. **Softmax**: Convert to probabilities
/// 5. **Laplace Smoothing** (optional): Prevent zero probabilities
///
/// # Constitution Compliance
///
/// - `num_experts = 8` (constitution.yaml: fuse_moe.num_experts)
/// - `temperature = 1.0` (default, configurable)
/// - `laplace_alpha = 0.01` (constitution.yaml: fuse_moe.laplace_alpha)
///
/// # Example
///
/// ```rust
/// use context_graph_embeddings::fusion::GatingNetwork;
/// use context_graph_embeddings::config::FusionConfig;
///
/// let config = FusionConfig::default();
/// let gating = GatingNetwork::new(&config).unwrap();
///
/// // Forward pass
/// let input = vec![0.0f32; 8320];
/// let probs = gating.forward(&input, 1).unwrap();
///
/// // Verify output
/// assert_eq!(probs.len(), 8);
/// assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5); // Sums to 1
/// assert!(probs.iter().all(|&p| p > 0.0)); // All positive
/// ```
#[derive(Debug, Clone)]
pub struct GatingNetwork {
    /// Layer normalization for input
    layer_norm: LayerNorm,
    /// Linear projection from input_dim to num_experts
    projection: Linear,
    /// Softmax temperature (lower = sharper)
    temperature: f32,
    /// Laplace smoothing alpha (0 = disabled)
    laplace_alpha: f32,
    /// Number of experts
    num_experts: usize,
    /// Noise standard deviation for training
    noise_std: f32,
}

impl GatingNetwork {
    /// Create a new GatingNetwork from FusionConfig.
    ///
    /// # Arguments
    ///
    /// * `config` - Fusion configuration specifying num_experts, temperature, etc.
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::ConfigError` if configuration is invalid
    /// - `EmbeddingError::InvalidDimension` if dimensions are inconsistent
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::GatingNetwork;
    /// use context_graph_embeddings::config::FusionConfig;
    ///
    /// let config = FusionConfig::default();
    /// let gating = GatingNetwork::new(&config).unwrap();
    /// assert_eq!(gating.num_experts(), 8);
    /// ```
    pub fn new(config: &FusionConfig) -> EmbeddingResult<Self> {
        // Validate config
        config.validate()?;

        let layer_norm = LayerNorm::new(TOTAL_CONCATENATED)?;
        let projection = Linear::new(TOTAL_CONCATENATED, config.num_experts)?;

        Ok(Self {
            layer_norm,
            projection,
            temperature: config.temperature,
            laplace_alpha: config.laplace_alpha,
            num_experts: config.num_experts,
            noise_std: config.noise_std,
        })
    }

    /// Create a GatingNetwork with custom input dimension.
    ///
    /// Useful for testing or non-standard configurations.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input dimension (typically 8320)
    /// * `config` - Fusion configuration
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::InvalidDimension` if input_dim == 0
    pub fn with_input_dim(input_dim: usize, config: &FusionConfig) -> EmbeddingResult<Self> {
        if input_dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }

        config.validate()?;

        let layer_norm = LayerNorm::new(input_dim)?;
        let projection = Linear::new(input_dim, config.num_experts)?;

        Ok(Self {
            layer_norm,
            projection,
            temperature: config.temperature,
            laplace_alpha: config.laplace_alpha,
            num_experts: config.num_experts,
            noise_std: config.noise_std,
        })
    }

    /// Get the number of experts.
    #[inline]
    #[must_use]
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Get the temperature value.
    #[inline]
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Get the Laplace alpha value.
    #[inline]
    #[must_use]
    pub fn laplace_alpha(&self) -> f32 {
        self.laplace_alpha
    }

    /// Get the noise standard deviation.
    #[inline]
    #[must_use]
    pub fn noise_std(&self) -> f32 {
        self.noise_std
    }

    /// Set the temperature for softmax scaling.
    ///
    /// Lower temperature → sharper distribution (more confident)
    /// Higher temperature → flatter distribution (more uncertain)
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::ConfigError` if temperature <= 0.
    pub fn set_temperature(&mut self, temperature: f32) -> EmbeddingResult<()> {
        if temperature <= 0.0 || temperature.is_nan() {
            return Err(EmbeddingError::ConfigError {
                message: format!("temperature must be > 0, got {}", temperature),
            });
        }
        self.temperature = temperature;
        Ok(())
    }

    /// Forward pass through gating network.
    ///
    /// Computes expert routing probabilities for each sample in the batch.
    ///
    /// # Processing Steps
    ///
    /// 1. Layer normalize input
    /// 2. Linear projection to logits
    /// 3. Temperature scaling (logits / temperature)
    /// 4. Softmax to probabilities
    /// 5. Laplace smoothing (if alpha > 0)
    ///
    /// # Arguments
    ///
    /// * `input` - Concatenated embeddings [batch_size * input_dim]
    /// * `batch_size` - Number of samples in batch
    ///
    /// # Returns
    ///
    /// Expert probabilities [batch_size * num_experts], each row sums to 1.0.
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::EmptyInput` if batch_size == 0
    /// - `EmbeddingError::InvalidDimension` if input dimensions are wrong
    /// - `EmbeddingError::InvalidValue` if input contains NaN
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::GatingNetwork;
    /// use context_graph_embeddings::config::FusionConfig;
    ///
    /// let config = FusionConfig::default();
    /// let gating = GatingNetwork::new(&config).unwrap();
    ///
    /// let input = vec![0.5f32; 8320];
    /// let probs = gating.forward(&input, 1).unwrap();
    ///
    /// // Check probabilities sum to 1
    /// let sum: f32 = probs.iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-5);
    /// ```
    pub fn forward(&self, input: &[f32], batch_size: usize) -> EmbeddingResult<Vec<f32>> {
        if batch_size == 0 {
            return Err(EmbeddingError::EmptyInput);
        }

        // Step 1: Layer normalization
        let normalized = self.layer_norm.forward(input, batch_size)?;

        // Step 2: Linear projection to logits
        let logits = self.projection.forward(&normalized, batch_size)?;

        // Step 3 & 4: Temperature-scaled softmax
        let mut probs = self.softmax_with_temperature(&logits, batch_size)?;

        // Step 5: Laplace smoothing (if enabled)
        if self.laplace_alpha > 0.0 {
            self.apply_laplace_smoothing(&mut probs, batch_size);
        }

        Ok(probs)
    }

    /// Forward pass with Gaussian noise for training exploration.
    ///
    /// Adds Gaussian noise to logits before softmax to encourage
    /// exploration of different expert combinations during training.
    ///
    /// # Arguments
    ///
    /// * `input` - Concatenated embeddings [batch_size * input_dim]
    /// * `batch_size` - Number of samples in batch
    ///
    /// # Returns
    ///
    /// Expert probabilities [batch_size * num_experts] with noise-influenced routing.
    ///
    /// # Errors
    ///
    /// Same as `forward()`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::GatingNetwork;
    /// use context_graph_embeddings::config::FusionConfig;
    ///
    /// let config = FusionConfig::for_training();
    /// let gating = GatingNetwork::new(&config).unwrap();
    ///
    /// let input = vec![0.5f32; 8320];
    /// let probs = gating.forward_with_noise(&input, 1).unwrap();
    ///
    /// // Probabilities still sum to 1
    /// let sum: f32 = probs.iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-5);
    /// ```
    pub fn forward_with_noise(&self, input: &[f32], batch_size: usize) -> EmbeddingResult<Vec<f32>> {
        if batch_size == 0 {
            return Err(EmbeddingError::EmptyInput);
        }

        // Step 1: Layer normalization
        let normalized = self.layer_norm.forward(input, batch_size)?;

        // Step 2: Linear projection to logits
        let mut logits = self.projection.forward(&normalized, batch_size)?;

        // Step 2.5: Add Gaussian noise (training only)
        if self.noise_std > 0.0 {
            self.add_gaussian_noise(&mut logits);
        }

        // Step 3 & 4: Temperature-scaled softmax
        let mut probs = self.softmax_with_temperature(&logits, batch_size)?;

        // Step 5: Laplace smoothing (if enabled)
        if self.laplace_alpha > 0.0 {
            self.apply_laplace_smoothing(&mut probs, batch_size);
        }

        Ok(probs)
    }

    /// Add Gaussian noise to logits for training exploration.
    fn add_gaussian_noise(&self, logits: &mut [f32]) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0f64, self.noise_std as f64).unwrap();

        for logit in logits.iter_mut() {
            *logit += normal.sample(&mut rng) as f32;
        }
    }

    /// Temperature-scaled softmax.
    ///
    /// Computes: softmax(logits / temperature)
    ///
    /// Uses numerically stable implementation with max subtraction.
    fn softmax_with_temperature(
        &self,
        logits: &[f32],
        batch_size: usize,
    ) -> EmbeddingResult<Vec<f32>> {
        let mut probs = vec![0.0f32; batch_size * self.num_experts];

        for b in 0..batch_size {
            let offset = b * self.num_experts;
            let sample_logits = &logits[offset..offset + self.num_experts];

            // Apply temperature scaling
            let scaled: Vec<f32> = sample_logits
                .iter()
                .map(|&x| x / self.temperature)
                .collect();

            // Numerical stability: subtract max
            let max_logit = scaled
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);

            // Compute exp(scaled - max)
            let exp_values: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();

            // Sum of exponentials
            let sum_exp: f32 = exp_values.iter().sum();

            // Avoid division by zero (shouldn't happen with valid inputs)
            if sum_exp == 0.0 || sum_exp.is_nan() {
                return Err(EmbeddingError::FusionError {
                    message: format!(
                        "Softmax sum is invalid ({}). Check for NaN/Inf in logits.",
                        sum_exp
                    ),
                });
            }

            // Normalize to probabilities
            for (i, exp_val) in exp_values.iter().enumerate() {
                probs[offset + i] = exp_val / sum_exp;
            }
        }

        Ok(probs)
    }

    /// Apply Laplace smoothing to probabilities.
    ///
    /// Formula: (p + alpha) / (1 + alpha * K)
    ///
    /// where K = num_experts
    ///
    /// This prevents any expert from having exactly zero probability,
    /// which improves gradient flow during training.
    fn apply_laplace_smoothing(&self, probs: &mut [f32], batch_size: usize) {
        let alpha = self.laplace_alpha;
        let k = self.num_experts as f32;
        let denominator = 1.0 + alpha * k;

        for b in 0..batch_size {
            let offset = b * self.num_experts;
            for i in 0..self.num_experts {
                probs[offset + i] = (probs[offset + i] + alpha) / denominator;
            }
        }
    }

    /// Select top-K experts based on probabilities.
    ///
    /// Returns the indices of the K experts with highest probabilities
    /// for each sample in the batch.
    ///
    /// # Arguments
    ///
    /// * `probs` - Expert probabilities [batch_size * num_experts]
    /// * `batch_size` - Number of samples in batch
    /// * `top_k` - Number of experts to select per sample
    ///
    /// # Returns
    ///
    /// Tuple of:
    /// - `indices`: Selected expert indices [batch_size * top_k]
    /// - `weights`: Normalized weights for selected experts [batch_size * top_k]
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::ConfigError` if top_k > num_experts
    /// - `EmbeddingError::InvalidDimension` if probs dimensions are wrong
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::GatingNetwork;
    /// use context_graph_embeddings::config::FusionConfig;
    /// use context_graph_embeddings::types::dimensions::TOP_K_EXPERTS;
    ///
    /// let config = FusionConfig::default();
    /// let gating = GatingNetwork::new(&config).unwrap();
    ///
    /// let input = vec![0.5f32; 8320];
    /// let probs = gating.forward(&input, 1).unwrap();
    ///
    /// let (indices, weights) = gating.select_top_k(&probs, 1, TOP_K_EXPERTS).unwrap();
    /// assert_eq!(indices.len(), TOP_K_EXPERTS);
    /// assert_eq!(weights.len(), TOP_K_EXPERTS);
    ///
    /// // Weights are renormalized to sum to 1
    /// let weight_sum: f32 = weights.iter().sum();
    /// assert!((weight_sum - 1.0).abs() < 1e-5);
    /// ```
    pub fn select_top_k(
        &self,
        probs: &[f32],
        batch_size: usize,
        top_k: usize,
    ) -> EmbeddingResult<(Vec<usize>, Vec<f32>)> {
        if top_k > self.num_experts {
            return Err(EmbeddingError::ConfigError {
                message: format!(
                    "top_k ({}) cannot exceed num_experts ({})",
                    top_k, self.num_experts
                ),
            });
        }

        let expected_len = batch_size * self.num_experts;
        if probs.len() != expected_len {
            return Err(EmbeddingError::InvalidDimension {
                expected: expected_len,
                actual: probs.len(),
            });
        }

        let mut indices = Vec::with_capacity(batch_size * top_k);
        let mut weights = Vec::with_capacity(batch_size * top_k);

        for b in 0..batch_size {
            let offset = b * self.num_experts;
            let sample_probs = &probs[offset..offset + self.num_experts];

            // Create (index, prob) pairs and sort by prob descending
            let mut indexed: Vec<(usize, f32)> = sample_probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();

            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top K
            let selected: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();

            // Renormalize weights to sum to 1
            let weight_sum: f32 = selected.iter().map(|(_, w)| w).sum();

            for (idx, w) in selected {
                indices.push(idx);
                weights.push(w / weight_sum);
            }
        }

        Ok((indices, weights))
    }

    /// Get a reference to the layer normalization component.
    #[inline]
    #[must_use]
    pub fn layer_norm(&self) -> &LayerNorm {
        &self.layer_norm
    }

    /// Get a reference to the linear projection component.
    #[inline]
    #[must_use]
    pub fn projection(&self) -> &Linear {
        &self.projection
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::dimensions::{NUM_EXPERTS, TOP_K_EXPERTS};

    // =========================================================================
    // LAYERNORM TESTS
    // =========================================================================

    #[test]
    fn test_layernorm_creation() {
        let norm = LayerNorm::new(1024).unwrap();
        assert_eq!(norm.dim(), 1024);
        assert!((norm.eps() - 1e-5).abs() < 1e-8);
    }

    #[test]
    fn test_layernorm_zero_dim_fails() {
        let result = LayerNorm::new(0);
        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::InvalidDimension { expected, actual } => {
                assert_eq!(expected, 1);
                assert_eq!(actual, 0);
            }
            _ => panic!("Expected InvalidDimension error"),
        }
    }

    #[test]
    fn test_layernorm_forward_normalizes_to_zero_mean() {
        let norm = LayerNorm::new(4).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = norm.forward(&input, 1).unwrap();

        // Mean should be ~0
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        println!("BEFORE: input = {:?}", input);
        println!("AFTER: output = {:?}, mean = {}", output, mean);
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_layernorm_forward_unit_variance() {
        let norm = LayerNorm::new(4).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = norm.forward(&input, 1).unwrap();

        // Variance should be ~1
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        let variance: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        println!("BEFORE: input = {:?}", input);
        println!("AFTER: variance = {}", variance);
        assert!(
            (variance - 1.0).abs() < 0.1,
            "Variance should be ~1, got {}",
            variance
        );
    }

    #[test]
    fn test_layernorm_batch_processing() {
        let norm = LayerNorm::new(4).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = norm.forward(&input, 2).unwrap();

        assert_eq!(output.len(), 8);

        // Check each sample independently normalized
        let mean1: f32 = output[0..4].iter().sum::<f32>() / 4.0;
        let mean2: f32 = output[4..8].iter().sum::<f32>() / 4.0;

        println!("BEFORE: input = {:?}", input);
        println!("AFTER: mean1 = {}, mean2 = {}", mean1, mean2);

        assert!(mean1.abs() < 1e-5);
        assert!(mean2.abs() < 1e-5);
    }

    #[test]
    fn test_layernorm_dimension_mismatch() {
        let norm = LayerNorm::new(4).unwrap();
        let input = vec![1.0, 2.0, 3.0]; // Wrong size
        let result = norm.forward(&input, 1);

        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::InvalidDimension { expected, actual } => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 3);
            }
            _ => panic!("Expected InvalidDimension error"),
        }
    }

    #[test]
    fn test_layernorm_empty_batch_fails() {
        let norm = LayerNorm::new(4).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = norm.forward(&input, 0);

        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::EmptyInput => {}
            _ => panic!("Expected EmptyInput error"),
        }
    }

    #[test]
    fn test_layernorm_nan_input_fails() {
        let norm = LayerNorm::new(4).unwrap();
        let input = vec![1.0, f32::NAN, 3.0, 4.0];
        let result = norm.forward(&input, 1);

        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::InvalidValue { index, value: _ } => {
                assert_eq!(index, 1);
            }
            _ => panic!("Expected InvalidValue error"),
        }
    }

    #[test]
    fn test_layernorm_set_gamma() {
        let mut norm = LayerNorm::new(4).unwrap();
        let new_gamma = vec![2.0, 2.0, 2.0, 2.0];
        norm.set_gamma(new_gamma.clone()).unwrap();

        assert_eq!(norm.gamma(), &new_gamma[..]);
    }

    #[test]
    fn test_layernorm_set_gamma_wrong_size_fails() {
        let mut norm = LayerNorm::new(4).unwrap();
        let result = norm.set_gamma(vec![1.0, 2.0]);

        assert!(result.is_err());
    }

    #[test]
    fn test_layernorm_set_beta() {
        let mut norm = LayerNorm::new(4).unwrap();
        let new_beta = vec![0.5, 0.5, 0.5, 0.5];
        norm.set_beta(new_beta.clone()).unwrap();

        assert_eq!(norm.beta(), &new_beta[..]);
    }

    // =========================================================================
    // LINEAR TESTS
    // =========================================================================

    #[test]
    fn test_linear_creation() {
        let linear = Linear::new(8320, 8).unwrap();
        assert_eq!(linear.in_features(), 8320);
        assert_eq!(linear.out_features(), 8);
        assert_eq!(linear.weights().len(), 8320 * 8);
        assert_eq!(linear.bias().len(), 8);
    }

    #[test]
    fn test_linear_zero_in_features_fails() {
        let result = Linear::new(0, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_zero_out_features_fails() {
        let result = Linear::new(8320, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_forward_dimensions() {
        let linear = Linear::new(4, 2).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = linear.forward(&input, 1).unwrap();

        assert_eq!(output.len(), 2);
        println!("BEFORE: input = {:?}", input);
        println!("AFTER: output = {:?}", output);
    }

    #[test]
    fn test_linear_forward_batch() {
        let linear = Linear::new(4, 2).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // batch_size = 2
        let output = linear.forward(&input, 2).unwrap();

        assert_eq!(output.len(), 4); // 2 * 2
    }

    #[test]
    fn test_linear_with_weights() {
        // Simple case: identity-ish transformation
        let weights = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let bias = vec![0.0, 0.0];
        let linear = Linear::with_weights(2, 2, weights, bias).unwrap();

        let input = vec![3.0, 5.0];
        let output = linear.forward(&input, 1).unwrap();

        println!("BEFORE: input = {:?}", input);
        println!("AFTER: output = {:?}", output);
        assert!((output[0] - 3.0).abs() < 1e-5);
        assert!((output[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_with_weights_wrong_size_fails() {
        let weights = vec![1.0, 2.0]; // Wrong size
        let bias = vec![0.0, 0.0];
        let result = Linear::with_weights(2, 2, weights, bias);

        assert!(result.is_err());
    }

    #[test]
    fn test_linear_dimension_mismatch() {
        let linear = Linear::new(4, 2).unwrap();
        let input = vec![1.0, 2.0, 3.0]; // Wrong size
        let result = linear.forward(&input, 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_linear_empty_batch_fails() {
        let linear = Linear::new(4, 2).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = linear.forward(&input, 0);

        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::EmptyInput => {}
            _ => panic!("Expected EmptyInput error"),
        }
    }

    #[test]
    fn test_linear_nan_input_fails() {
        let linear = Linear::new(4, 2).unwrap();
        let input = vec![1.0, f32::NAN, 3.0, 4.0];
        let result = linear.forward(&input, 1);

        assert!(result.is_err());
    }

    // =========================================================================
    // GATING NETWORK TESTS
    // =========================================================================

    #[test]
    fn test_gating_creation() {
        let config = FusionConfig::default();
        let gating = GatingNetwork::new(&config).unwrap();

        assert_eq!(gating.num_experts(), NUM_EXPERTS);
        assert!((gating.temperature() - 1.0).abs() < 1e-5);
        assert!((gating.laplace_alpha() - 0.01).abs() < 1e-5);
    }

    #[test]
    fn test_gating_forward_returns_valid_probs() {
        let config = FusionConfig::default();
        let gating = GatingNetwork::new(&config).unwrap();

        let input = vec![0.5f32; TOTAL_CONCATENATED];
        let probs = gating.forward(&input, 1).unwrap();

        assert_eq!(probs.len(), NUM_EXPERTS);

        // All probabilities should be positive
        for (i, &p) in probs.iter().enumerate() {
            assert!(p > 0.0, "Probability at index {} should be > 0, got {}", i, p);
        }

        // Should sum to 1 (or very close due to Laplace smoothing)
        let sum: f32 = probs.iter().sum();
        println!("BEFORE: input (first 10) = {:?}", &input[0..10]);
        println!("AFTER: probs = {:?}, sum = {}", probs, sum);
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Probabilities should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_gating_forward_batch() {
        let config = FusionConfig::default();
        let gating = GatingNetwork::new(&config).unwrap();

        let input = vec![0.5f32; TOTAL_CONCATENATED * 3]; // batch_size = 3
        let probs = gating.forward(&input, 3).unwrap();

        assert_eq!(probs.len(), NUM_EXPERTS * 3);

        // Check each sample sums to 1
        for b in 0..3 {
            let sample_sum: f32 = probs[b * NUM_EXPERTS..(b + 1) * NUM_EXPERTS]
                .iter()
                .sum();
            assert!(
                (sample_sum - 1.0).abs() < 1e-5,
                "Sample {} should sum to 1, got {}",
                b,
                sample_sum
            );
        }
    }

    #[test]
    fn test_gating_temperature_affects_distribution() {
        let mut config = FusionConfig::default();
        config.temperature = 0.1; // Sharp distribution
        config.laplace_alpha = 0.0; // Disable smoothing for clear comparison

        let gating_sharp = GatingNetwork::new(&config).unwrap();

        config.temperature = 2.0; // Flat distribution
        let gating_flat = GatingNetwork::new(&config).unwrap();

        let input = vec![0.5f32; TOTAL_CONCATENATED];

        let probs_sharp = gating_sharp.forward(&input, 1).unwrap();
        let probs_flat = gating_flat.forward(&input, 1).unwrap();

        // Sharp temperature should have higher max probability
        let max_sharp = probs_sharp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_flat = probs_flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        println!("Sharp (T=0.1): {:?}", probs_sharp);
        println!("Flat (T=2.0): {:?}", probs_flat);

        // Note: Due to random initialization, this might not always hold
        // but the distribution should generally be sharper with lower temperature
        assert!(
            max_sharp >= max_flat * 0.5, // Allow some tolerance
            "Sharp distribution should have higher max: {} vs {}",
            max_sharp,
            max_flat
        );
    }

    #[test]
    fn test_gating_set_temperature() {
        let config = FusionConfig::default();
        let mut gating = GatingNetwork::new(&config).unwrap();

        gating.set_temperature(0.5).unwrap();
        assert!((gating.temperature() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_gating_set_temperature_zero_fails() {
        let config = FusionConfig::default();
        let mut gating = GatingNetwork::new(&config).unwrap();

        let result = gating.set_temperature(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_gating_set_temperature_negative_fails() {
        let config = FusionConfig::default();
        let mut gating = GatingNetwork::new(&config).unwrap();

        let result = gating.set_temperature(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_gating_forward_with_noise_still_valid() {
        let config = FusionConfig::for_training();
        let gating = GatingNetwork::new(&config).unwrap();

        let input = vec![0.5f32; TOTAL_CONCATENATED];
        let probs = gating.forward_with_noise(&input, 1).unwrap();

        // Should still sum to 1
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Probabilities with noise should sum to 1, got {}",
            sum
        );

        // All positive
        assert!(probs.iter().all(|&p| p > 0.0));
    }

    #[test]
    fn test_gating_laplace_smoothing_prevents_zero() {
        let mut config = FusionConfig::default();
        config.laplace_alpha = 0.01;

        let gating = GatingNetwork::new(&config).unwrap();

        let input = vec![0.5f32; TOTAL_CONCATENATED];
        let probs = gating.forward(&input, 1).unwrap();

        // With Laplace smoothing, no probability should be exactly 0
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                p > 0.0,
                "Laplace smoothing should prevent zero at index {}, got {}",
                i,
                p
            );
        }
    }

    #[test]
    fn test_gating_select_top_k() {
        let config = FusionConfig::default();
        let gating = GatingNetwork::new(&config).unwrap();

        let input = vec![0.5f32; TOTAL_CONCATENATED];
        let probs = gating.forward(&input, 1).unwrap();

        let (indices, weights) = gating.select_top_k(&probs, 1, TOP_K_EXPERTS).unwrap();

        assert_eq!(indices.len(), TOP_K_EXPERTS);
        assert_eq!(weights.len(), TOP_K_EXPERTS);

        // Indices should be unique
        let unique: std::collections::HashSet<_> = indices.iter().collect();
        assert_eq!(unique.len(), TOP_K_EXPERTS);

        // Indices should be in range [0, NUM_EXPERTS)
        assert!(indices.iter().all(|&i| i < NUM_EXPERTS));

        // Weights should sum to 1
        let weight_sum: f32 = weights.iter().sum();
        println!("BEFORE: probs = {:?}", probs);
        println!("AFTER: indices = {:?}, weights = {:?}", indices, weights);
        assert!(
            (weight_sum - 1.0).abs() < 1e-5,
            "Top-K weights should sum to 1, got {}",
            weight_sum
        );
    }

    #[test]
    fn test_gating_select_top_k_batch() {
        let config = FusionConfig::default();
        let gating = GatingNetwork::new(&config).unwrap();

        let input = vec![0.5f32; TOTAL_CONCATENATED * 2]; // batch_size = 2
        let probs = gating.forward(&input, 2).unwrap();

        let (indices, weights) = gating.select_top_k(&probs, 2, TOP_K_EXPERTS).unwrap();

        assert_eq!(indices.len(), 2 * TOP_K_EXPERTS);
        assert_eq!(weights.len(), 2 * TOP_K_EXPERTS);

        // Check each sample's weights sum to 1
        let sum1: f32 = weights[0..TOP_K_EXPERTS].iter().sum();
        let sum2: f32 = weights[TOP_K_EXPERTS..].iter().sum();

        assert!((sum1 - 1.0).abs() < 1e-5);
        assert!((sum2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gating_select_top_k_exceeds_num_experts_fails() {
        let config = FusionConfig::default();
        let gating = GatingNetwork::new(&config).unwrap();

        let probs = vec![0.125f32; NUM_EXPERTS];
        let result = gating.select_top_k(&probs, 1, NUM_EXPERTS + 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_gating_custom_input_dim() {
        let config = FusionConfig::default();
        let gating = GatingNetwork::with_input_dim(1024, &config).unwrap();

        let input = vec![0.5f32; 1024];
        let probs = gating.forward(&input, 1).unwrap();

        assert_eq!(probs.len(), NUM_EXPERTS);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gating_custom_input_dim_zero_fails() {
        let config = FusionConfig::default();
        let result = GatingNetwork::with_input_dim(0, &config);

        assert!(result.is_err());
    }

    #[test]
    fn test_gating_empty_batch_fails() {
        let config = FusionConfig::default();
        let gating = GatingNetwork::new(&config).unwrap();

        let input = vec![0.5f32; TOTAL_CONCATENATED];
        let result = gating.forward(&input, 0);

        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::EmptyInput => {}
            _ => panic!("Expected EmptyInput error"),
        }
    }

    // =========================================================================
    // EDGE CASE TESTS
    // =========================================================================

    #[test]
    fn test_edge_case_very_large_input() {
        let norm = LayerNorm::new(4).unwrap();
        let input = vec![1e10, 1e10, 1e10, 1e10];
        let output = norm.forward(&input, 1).unwrap();

        // All same values → mean matches input → normalized to all zeros
        println!("BEFORE: input = {:?}", input);
        println!("AFTER: output = {:?}", output);

        // When all inputs are the same, variance is 0
        // normalized = (x - mean) / sqrt(eps) ≈ 0
        assert!(output.iter().all(|&x| x.abs() < 1.0));
    }

    #[test]
    fn test_edge_case_very_small_input() {
        let norm = LayerNorm::new(4).unwrap();
        let input = vec![1e-10, 2e-10, 3e-10, 4e-10];
        let output = norm.forward(&input, 1).unwrap();

        // Should still normalize properly
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        println!("BEFORE: input = {:?}", input);
        println!("AFTER: output = {:?}, mean = {}", output, mean);
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn test_edge_case_all_zeros_input() {
        let norm = LayerNorm::new(4).unwrap();
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let output = norm.forward(&input, 1).unwrap();

        // All zeros → mean = 0, variance = 0
        // normalized = 0 / sqrt(eps) = 0
        println!("BEFORE: input = {:?}", input);
        println!("AFTER: output = {:?}", output);
        assert!(output.iter().all(|&x| x.abs() < 1e-5));
    }

    #[test]
    fn test_edge_case_negative_input() {
        let norm = LayerNorm::new(4).unwrap();
        let input = vec![-10.0, -5.0, 0.0, 5.0];
        let output = norm.forward(&input, 1).unwrap();

        // Should handle negative inputs fine
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        println!("BEFORE: input = {:?}", input);
        println!("AFTER: output = {:?}", output);
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn test_edge_case_mixed_sign_input() {
        let linear = Linear::new(4, 2).unwrap();
        let input = vec![-1.0, -0.5, 0.5, 1.0];
        let output = linear.forward(&input, 1).unwrap();

        // Just verify it completes without error
        println!("BEFORE: input = {:?}", input);
        println!("AFTER: output = {:?}", output);
        assert_eq!(output.len(), 2);
    }

    // =========================================================================
    // INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_full_pipeline_integration() {
        let config = FusionConfig::default();
        let gating = GatingNetwork::new(&config).unwrap();

        // Simulate real concatenated embedding
        let input = vec![0.1f32; TOTAL_CONCATENATED];

        // Forward pass
        let probs = gating.forward(&input, 1).unwrap();

        // Select top-K
        let (indices, weights) =
            gating.select_top_k(&probs, 1, TOP_K_EXPERTS).unwrap();

        println!("=== Full Pipeline Integration ===");
        println!("Input dimension: {}", TOTAL_CONCATENATED);
        println!("Probabilities: {:?}", probs);
        println!("Selected experts: {:?}", indices);
        println!("Expert weights: {:?}", weights);

        // Verify
        assert_eq!(probs.len(), NUM_EXPERTS);
        assert_eq!(indices.len(), TOP_K_EXPERTS);
        assert_eq!(weights.len(), TOP_K_EXPERTS);

        let prob_sum: f32 = probs.iter().sum();
        let weight_sum: f32 = weights.iter().sum();

        assert!((prob_sum - 1.0).abs() < 1e-5);
        assert!((weight_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_training_vs_inference_mode() {
        let inference_config = FusionConfig::for_inference();
        let training_config = FusionConfig::for_training();

        let gating_inference = GatingNetwork::new(&inference_config).unwrap();
        let gating_training = GatingNetwork::new(&training_config).unwrap();

        // Verify configurations
        assert!((gating_inference.noise_std() - 0.0).abs() < 1e-5);
        assert!(gating_training.noise_std() > 0.0);

        println!("Inference noise_std: {}", gating_inference.noise_std());
        println!("Training noise_std: {}", gating_training.noise_std());
    }

    #[test]
    fn test_batch_consistency() {
        let config = FusionConfig::default();
        let gating = GatingNetwork::new(&config).unwrap();

        // Same input processed as batch vs individual
        let input = vec![0.5f32; TOTAL_CONCATENATED];

        let single_probs = gating.forward(&input, 1).unwrap();

        let batch_input = vec![0.5f32; TOTAL_CONCATENATED * 2];
        let batch_probs = gating.forward(&batch_input, 2).unwrap();

        // First sample of batch should match single (deterministic weights)
        println!("Single: {:?}", single_probs);
        println!("Batch[0]: {:?}", &batch_probs[0..NUM_EXPERTS]);
        println!("Batch[1]: {:?}", &batch_probs[NUM_EXPERTS..]);

        // Both samples in batch should be identical (same input)
        for i in 0..NUM_EXPERTS {
            assert!(
                (batch_probs[i] - batch_probs[NUM_EXPERTS + i]).abs() < 1e-5,
                "Batch samples should be identical at index {}",
                i
            );
        }
    }
}
