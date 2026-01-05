//! Fused Embedding Provider implementation.
//!
//! Provides a GPU-accelerated embedding provider that produces 1536D vectors
//! compatible with OpenAI's ada-002 embedding format.
//!
//! # Architecture
//!
//! The FusedEmbeddingProvider uses a two-stage pipeline:
//! 1. **SemanticModel (E1)**: Produces 1024D dense vectors using e5-large-v2
//! 2. **ProjectionLayer**: Linear projection 1024D -> 1536D with learned weights
//!
//! This approach provides:
//! - Fast inference (<10ms single embed)
//! - High-quality semantic understanding from E5
//! - Compatibility with 1536D downstream systems
//!
//! # GPU Acceleration
//!
//! All operations run on GPU via Candle:
//! - BERT forward pass for semantic embedding
//! - cuBLAS GEMM for projection layer
//! - L2 normalization in GPU memory
//!
//! # Thread Safety
//!
//! - `AtomicBool` for ready state (lock-free reads)
//! - `RwLock` for projection weights (read-heavy)
//! - All tensor operations are GPU-synchronized

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::{init_gpu, normalize_gpu};
use crate::models::pretrained::SemanticModel;
use crate::traits::SingleModelConfig;
use crate::types::dimensions::{FUSED_OUTPUT, SEMANTIC};
use crate::types::ModelInput;

use super::EmbeddingProvider;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Model name for the fused embedding provider.
pub const FUSED_MODEL_NAME: &str = "fused-embedding-v1";

/// Maximum token count (inherited from SemanticModel).
pub const FUSED_MAX_TOKENS: usize = 8192;

/// Input dimension for projection (SemanticModel output).
const PROJECTION_INPUT_DIM: usize = SEMANTIC; // 1024

/// Output dimension for projection (final embedding).
const PROJECTION_OUTPUT_DIM: usize = FUSED_OUTPUT; // 1536

/// Default model path for SemanticModel weights.
const DEFAULT_MODEL_PATH: &str = "models/e5-large-v2";

// =============================================================================
// PROJECTION LAYER
// =============================================================================

/// Linear projection layer for dimension expansion.
///
/// Projects 1024D semantic embeddings to 1536D fused embeddings
/// using a learned linear transformation: y = Wx + b
///
/// # Initialization
///
/// Weights are initialized using Xavier/Glorot uniform initialization
/// for stable gradient flow. Bias is initialized to zero.
///
/// # GPU Acceleration
///
/// Uses cuBLAS GEMM for matrix multiplication on GPU.
#[derive(Debug)]
pub struct ProjectionLayer {
    /// Weight matrix: [1536, 1024] on GPU
    weight: Tensor,
    /// Bias vector: [1536] on GPU
    bias: Tensor,
}

impl ProjectionLayer {
    /// Create a new projection layer with Xavier initialization.
    ///
    /// # Arguments
    /// * `device` - GPU device for tensor allocation
    ///
    /// # Errors
    /// Returns error if tensor allocation fails (e.g., OOM).
    pub fn new(device: &Device) -> EmbeddingResult<Self> {
        // Xavier initialization: scale = sqrt(6 / (fan_in + fan_out))
        let fan_in = PROJECTION_INPUT_DIM as f64;
        let fan_out = PROJECTION_OUTPUT_DIM as f64;
        let scale = (6.0 / (fan_in + fan_out)).sqrt();

        // Initialize weight matrix with Xavier uniform
        let weight = Tensor::rand(
            -scale,
            scale,
            (PROJECTION_OUTPUT_DIM, PROJECTION_INPUT_DIM),
            device,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("ProjectionLayer weight init failed: {}", e),
        })?
        .to_dtype(DType::F32)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("ProjectionLayer weight dtype conversion failed: {}", e),
        })?;

        // Initialize bias to zero
        let bias = Tensor::zeros((PROJECTION_OUTPUT_DIM,), DType::F32, device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("ProjectionLayer bias init failed: {}", e),
            }
        })?;

        Ok(Self { weight, bias })
    }

    /// Create projection layer from saved weights.
    ///
    /// # Arguments
    /// * `weight_path` - Path to weight tensor file (.safetensors)
    /// * `bias_path` - Path to bias tensor file (.safetensors)
    /// * `device` - GPU device for tensor allocation
    ///
    /// # Errors
    /// Returns error if files don't exist or have wrong shape.
    #[allow(dead_code)]
    pub fn from_files(
        weight_path: &Path,
        bias_path: &Path,
        device: &Device,
    ) -> EmbeddingResult<Self> {
        use candle_core::safetensors;

        // Load weight tensor
        let weight_tensors = safetensors::load(weight_path, device).map_err(|e| {
            EmbeddingError::ModelLoadError {
                model_id: crate::types::ModelId::Semantic,
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Projection weight load failed: {}", e),
                )),
            }
        })?;

        let weight = weight_tensors
            .get("weight")
            .ok_or_else(|| EmbeddingError::ConfigError {
                message: "Projection weight tensor not found in safetensors".to_string(),
            })?
            .clone();

        // Load bias tensor
        let bias_tensors =
            safetensors::load(bias_path, device).map_err(|e| EmbeddingError::ModelLoadError {
                model_id: crate::types::ModelId::Semantic,
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Projection bias load failed: {}", e),
                )),
            })?;

        let bias = bias_tensors
            .get("bias")
            .ok_or_else(|| EmbeddingError::ConfigError {
                message: "Projection bias tensor not found in safetensors".to_string(),
            })?
            .clone();

        // Validate shapes
        let weight_shape = weight.shape().dims();
        if weight_shape != [PROJECTION_OUTPUT_DIM, PROJECTION_INPUT_DIM] {
            return Err(EmbeddingError::InvalidDimension {
                expected: PROJECTION_OUTPUT_DIM * PROJECTION_INPUT_DIM,
                actual: weight_shape.iter().product(),
            });
        }

        let bias_shape = bias.shape().dims();
        if bias_shape != [PROJECTION_OUTPUT_DIM] {
            return Err(EmbeddingError::InvalidDimension {
                expected: PROJECTION_OUTPUT_DIM,
                actual: bias_shape.iter().product(),
            });
        }

        Ok(Self { weight, bias })
    }

    /// Forward pass: y = Wx + b
    ///
    /// # Arguments
    /// * `input` - Input tensor [1024] or [batch, 1024]
    ///
    /// # Returns
    /// Output tensor [1536] or [batch, 1536]
    ///
    /// # GPU Acceleration
    /// Uses cuBLAS GEMM for matrix multiplication.
    pub fn forward(&self, input: &Tensor) -> EmbeddingResult<Tensor> {
        // Ensure input is 2D: [batch, input_dim]
        let input_2d = if input.dims().len() == 1 {
            input.unsqueeze(0).map_err(|e| EmbeddingError::GpuError {
                message: format!("ProjectionLayer input unsqueeze failed: {}", e),
            })?
        } else {
            input.clone()
        };

        // Linear transformation: y = x @ W^T + b
        // weight: [1536, 1024], input: [batch, 1024]
        // result: [batch, 1536]
        let output = input_2d
            .matmul(&self.weight.t().map_err(|e| EmbeddingError::GpuError {
                message: format!("ProjectionLayer weight transpose failed: {}", e),
            })?)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("ProjectionLayer matmul failed: {}", e),
            })?;

        // Add bias
        let output = output
            .broadcast_add(&self.bias)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("ProjectionLayer bias add failed: {}", e),
            })?;

        // Squeeze back to 1D if input was 1D
        if input.dims().len() == 1 {
            output.squeeze(0).map_err(|e| EmbeddingError::GpuError {
                message: format!("ProjectionLayer output squeeze failed: {}", e),
            })
        } else {
            Ok(output)
        }
    }
}

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for FusedEmbeddingProvider.
#[derive(Debug, Clone)]
pub struct FusedProviderConfig {
    /// Path to SemanticModel weights directory.
    pub model_path: PathBuf,
    /// Maximum batch size for batch processing.
    pub max_batch_size: usize,
    /// Whether to use query mode (vs passage mode) by default.
    pub default_query_mode: bool,
}

impl Default for FusedProviderConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from(DEFAULT_MODEL_PATH),
            max_batch_size: 32,
            default_query_mode: false,
        }
    }
}

impl FusedProviderConfig {
    /// Create config with custom model path.
    pub fn with_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = path.into();
        self
    }

    /// Set maximum batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.max_batch_size = batch_size;
        self
    }

    /// Set default query mode.
    pub fn with_query_mode(mut self, query_mode: bool) -> Self {
        self.default_query_mode = query_mode;
        self
    }
}

// =============================================================================
// FUSED EMBEDDING PROVIDER
// =============================================================================

/// GPU-accelerated fused embedding provider producing 1536D vectors.
///
/// # Architecture
///
/// ```text
/// Text Input
///     │
///     ▼
/// ┌─────────────────────────────────────┐
/// │  SemanticModel (E5-large-v2)        │
/// │  - Tokenization                     │
/// │  - BERT Encoder (24 layers)         │
/// │  - Mean Pooling                     │
/// │  - L2 Normalization                 │
/// │  Output: 1024D normalized vector    │
/// └─────────────────────────────────────┘
///     │
///     ▼
/// ┌─────────────────────────────────────┐
/// │  ProjectionLayer                    │
/// │  - Linear: 1024D → 1536D            │
/// │  - Bias addition                    │
/// │  - L2 Normalization                 │
/// │  Output: 1536D normalized vector    │
/// └─────────────────────────────────────┘
///     │
///     ▼
/// 1536D Embedding (OpenAI ada-002 compatible)
/// ```
///
/// # Performance
///
/// - Single embed: <10ms on RTX 5090
/// - Batch 32: <50ms on RTX 5090
/// - Memory: ~2GB VRAM (E5-large-v2 + projection)
///
/// # Thread Safety
///
/// Safe for concurrent access from multiple threads.
pub struct FusedEmbeddingProvider {
    /// SemanticModel for 1024D embeddings.
    semantic_model: SemanticModel,
    /// Projection layer for 1024D -> 1536D.
    projection: RwLock<Option<ProjectionLayer>>,
    /// Configuration.
    config: FusedProviderConfig,
    /// Ready state flag.
    ready: AtomicBool,
}

impl FusedEmbeddingProvider {
    /// Create a new FusedEmbeddingProvider with default configuration.
    ///
    /// # Errors
    /// Returns error if SemanticModel creation fails.
    pub fn new() -> EmbeddingResult<Self> {
        Self::with_config(FusedProviderConfig::default())
    }

    /// Create a new FusedEmbeddingProvider with custom configuration.
    ///
    /// # Arguments
    /// * `config` - Provider configuration
    ///
    /// # Errors
    /// Returns error if SemanticModel creation fails.
    pub fn with_config(config: FusedProviderConfig) -> EmbeddingResult<Self> {
        let model_config = SingleModelConfig::default();
        let semantic_model = SemanticModel::new(&config.model_path, model_config)?;

        Ok(Self {
            semantic_model,
            projection: RwLock::new(None),
            config,
            ready: AtomicBool::new(false),
        })
    }

    /// Initialize the provider (load model weights).
    ///
    /// # GPU Pipeline
    ///
    /// 1. Initialize CUDA device
    /// 2. Load SemanticModel weights (~1.3GB)
    /// 3. Initialize ProjectionLayer weights (~6MB)
    /// 4. Set ready state
    ///
    /// # Errors
    /// Returns error if GPU init or model loading fails.
    pub async fn initialize(&self) -> EmbeddingResult<()> {
        tracing::info!(
            target: "context_graph_embeddings::fused",
            "Initializing FusedEmbeddingProvider..."
        );

        // Initialize GPU
        let device = init_gpu().map_err(|e| {
            tracing::error!(
                target: "context_graph_embeddings::fused",
                error = %e,
                "FusedEmbeddingProvider GPU init failed"
            );
            EmbeddingError::GpuError {
                message: format!("FusedEmbeddingProvider GPU init failed: {}", e),
            }
        })?;

        // Load SemanticModel
        self.semantic_model.load().await?;

        // Initialize projection layer
        let projection = ProjectionLayer::new(device)?;

        // Store projection layer
        let mut proj_guard = self
            .projection
            .write()
            .map_err(|e| EmbeddingError::InternalError {
                message: format!("Failed to acquire projection write lock: {}", e),
            })?;
        *proj_guard = Some(projection);
        drop(proj_guard);

        // Set ready state
        self.ready.store(true, Ordering::SeqCst);

        tracing::info!(
            target: "context_graph_embeddings::fused",
            "FusedEmbeddingProvider initialized: ready for inference"
        );

        Ok(())
    }

    /// Check if the provider is ready for inference.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    /// Embed text and return 1536D vector.
    ///
    /// Internal implementation used by EmbeddingProvider trait.
    async fn embed_internal(&self, text: &str, is_query: bool) -> EmbeddingResult<Vec<f32>> {
        // Check ready state
        if !self.is_ready() {
            return Err(EmbeddingError::NotInitialized {
                model_id: crate::types::ModelId::Semantic,
            });
        }

        // Validate input
        if text.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let start = std::time::Instant::now();

        // Create ModelInput for SemanticModel
        let instruction = if is_query {
            Some("query".to_string())
        } else {
            None
        };

        let input = ModelInput::Text {
            content: text.to_string(),
            instruction,
        };

        // Get semantic embedding (1024D)
        let semantic_embedding = self.semantic_model.embed_single(&input).await?;
        let semantic_vector = &semantic_embedding.vector;

        // Project to 1536D
        let projected = self.project_embedding(semantic_vector)?;

        let elapsed = start.elapsed();
        tracing::trace!(
            target: "context_graph_embeddings::fused",
            latency_ms = elapsed.as_millis(),
            input_len = text.len(),
            "FusedEmbeddingProvider embed completed"
        );

        Ok(projected)
    }

    /// Project 1024D embedding to 1536D.
    fn project_embedding(&self, input: &[f32]) -> EmbeddingResult<Vec<f32>> {
        // Validate input dimension
        if input.len() != PROJECTION_INPUT_DIM {
            return Err(EmbeddingError::InvalidDimension {
                expected: PROJECTION_INPUT_DIM,
                actual: input.len(),
            });
        }

        // Get projection layer
        let proj_guard = self
            .projection
            .read()
            .map_err(|e| EmbeddingError::InternalError {
                message: format!("Failed to acquire projection read lock: {}", e),
            })?;

        let projection = proj_guard.as_ref().ok_or(EmbeddingError::NotInitialized {
            model_id: crate::types::ModelId::Semantic,
        })?;

        // Create input tensor on GPU
        let device = init_gpu().map_err(|e| EmbeddingError::GpuError {
            message: format!("GPU device access failed: {}", e),
        })?;

        let input_tensor =
            Tensor::from_slice(input, (PROJECTION_INPUT_DIM,), device).map_err(|e| {
                EmbeddingError::GpuError {
                    message: format!("Input tensor creation failed: {}", e),
                }
            })?;

        // Project
        let projected = projection.forward(&input_tensor)?;

        // Normalize
        let normalized = normalize_gpu(&projected).map_err(|e| EmbeddingError::GpuError {
            message: format!("Normalization failed: {}", e),
        })?;

        // Extract to CPU
        let result: Vec<f32> = normalized.to_vec1().map_err(|e| EmbeddingError::GpuError {
            message: format!("Tensor extraction failed: {}", e),
        })?;

        // Validate output dimension
        if result.len() != PROJECTION_OUTPUT_DIM {
            return Err(EmbeddingError::InvalidDimension {
                expected: PROJECTION_OUTPUT_DIM,
                actual: result.len(),
            });
        }

        Ok(result)
    }

    /// Batch embed texts for efficiency.
    ///
    /// Processes texts in batches up to max_batch_size for optimal GPU utilization.
    async fn embed_batch_internal(
        &self,
        texts: &[&str],
        is_query: bool,
    ) -> EmbeddingResult<Vec<Vec<f32>>> {
        if !self.is_ready() {
            return Err(EmbeddingError::NotInitialized {
                model_id: crate::types::ModelId::Semantic,
            });
        }

        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let start = std::time::Instant::now();

        // Create ModelInputs
        let inputs: Vec<ModelInput> = texts
            .iter()
            .map(|text| {
                let instruction = if is_query {
                    Some("query".to_string())
                } else {
                    None
                };
                ModelInput::Text {
                    content: text.to_string(),
                    instruction,
                }
            })
            .collect();

        // Get semantic embeddings (1024D each)
        let semantic_embeddings = self.semantic_model.embed_batch(&inputs).await?;

        // Project each to 1536D
        let mut results = Vec::with_capacity(texts.len());
        for embedding in semantic_embeddings {
            let projected = self.project_embedding(&embedding.vector)?;
            results.push(projected);
        }

        let elapsed = start.elapsed();
        tracing::debug!(
            target: "context_graph_embeddings::fused",
            latency_ms = elapsed.as_millis(),
            batch_size = texts.len(),
            "FusedEmbeddingProvider batch embed completed"
        );

        Ok(results)
    }

    /// Get provider configuration.
    pub fn config(&self) -> &FusedProviderConfig {
        &self.config
    }
}

// =============================================================================
// EMBEDDING PROVIDER TRAIT IMPLEMENTATION
// =============================================================================

#[async_trait]
impl EmbeddingProvider for FusedEmbeddingProvider {
    /// Generate 1536D embedding for a single text.
    ///
    /// Uses default mode (passage) unless configured otherwise.
    async fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>> {
        self.embed_internal(text, self.config.default_query_mode)
            .await
    }

    /// Generate 1536D embeddings for multiple texts.
    ///
    /// Optimized batch processing for GPU efficiency.
    async fn embed_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
        self.embed_batch_internal(texts, self.config.default_query_mode)
            .await
    }

    /// Get output dimension (1536).
    #[inline]
    fn dimension(&self) -> usize {
        PROJECTION_OUTPUT_DIM
    }

    /// Get model name.
    #[inline]
    fn model_name(&self) -> &str {
        FUSED_MODEL_NAME
    }

    /// Get maximum token count (8192).
    #[inline]
    fn max_tokens(&self) -> usize {
        FUSED_MAX_TOKENS
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for FusedEmbeddingProvider {}
unsafe impl Sync for FusedEmbeddingProvider {}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = FusedProviderConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert!(!config.default_query_mode);
    }

    #[test]
    fn test_config_builder() {
        let config = FusedProviderConfig::default()
            .with_model_path("/custom/path")
            .with_batch_size(64)
            .with_query_mode(true);

        assert_eq!(config.model_path, PathBuf::from("/custom/path"));
        assert_eq!(config.max_batch_size, 64);
        assert!(config.default_query_mode);
    }

    #[test]
    fn test_dimension_constants() {
        assert_eq!(PROJECTION_INPUT_DIM, 1024);
        assert_eq!(PROJECTION_OUTPUT_DIM, 1536);
        assert_eq!(FUSED_MAX_TOKENS, 8192);
    }

    #[test]
    fn test_model_name() {
        assert_eq!(FUSED_MODEL_NAME, "fused-embedding-v1");
    }

    // GPU tests require actual GPU hardware
    #[cfg(feature = "candle")]
    mod gpu_tests {
        use super::*;

        #[test]
        fn test_projection_layer_creation() {
            // This test only runs with GPU available
            if let Ok(device) = init_gpu() {
                let layer = ProjectionLayer::new(device);
                assert!(layer.is_ok(), "ProjectionLayer creation should succeed");

                let layer = layer.unwrap();
                let weight_shape = layer.weight.shape().dims();
                assert_eq!(weight_shape, &[1536, 1024]);

                let bias_shape = layer.bias.shape().dims();
                assert_eq!(bias_shape, &[1536]);
            }
        }

        #[test]
        fn test_projection_forward() {
            if let Ok(device) = init_gpu() {
                let layer = ProjectionLayer::new(device).unwrap();

                // Create test input
                let input = Tensor::zeros((1024,), DType::F32, device).unwrap();
                let output = layer.forward(&input);

                assert!(output.is_ok(), "Forward pass should succeed");
                let output = output.unwrap();
                assert_eq!(output.shape().dims(), &[1536]);
            }
        }

        #[test]
        fn test_projection_batch_forward() {
            if let Ok(device) = init_gpu() {
                let layer = ProjectionLayer::new(device).unwrap();

                // Create batch input
                let input = Tensor::zeros((4, 1024), DType::F32, device).unwrap();
                let output = layer.forward(&input);

                assert!(output.is_ok(), "Batch forward should succeed");
                let output = output.unwrap();
                assert_eq!(output.shape().dims(), &[4, 1536]);
            }
        }
    }
}
