//! CodeModel struct and core implementation.
//!
//! Contains the main model struct, construction, and embedding methods.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use tokenizers::Tokenizer;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::init_gpu;
use crate::traits::{EmbeddingModel, SingleModelConfig};
use crate::types::{InputType, ModelEmbedding, ModelId, ModelInput};

use super::forward::gpu_forward;
use super::weights::CodeT5pWeights;

/// Internal state that varies based on feature flags.
#[allow(dead_code)]
pub(super) enum ModelState {
    /// Unloaded - no weights in memory.
    Unloaded,

    /// Loaded with candle model and tokenizer (GPU-accelerated).
    Loaded {
        /// CodeT5p model weights on GPU.
        weights: CodeT5pWeights,
        /// HuggingFace tokenizer for text encoding (boxed to reduce enum size).
        tokenizer: Box<Tokenizer>,
    },
}

/// Code embedding model using Salesforce/codet5p-110m-embedding.
///
/// This model produces 256D native vectors (768D projected) optimized for
/// code understanding and semantic similarity in source code.
///
/// # Architecture
///
/// CodeT5+ is an encoder-decoder model based on T5 with code-specific
/// pretraining objectives:
/// - **Span denoising**: Reconstruct masked spans in code
/// - **Identifier prediction**: Predict masked variable/function names
/// - **Cross-modal contrastive learning**: Align code and natural language
///
/// The embedding model uses the encoder with a learned projection head.
///
/// # Supported Languages
///
/// Trained on code from multiple programming languages:
/// - Python, JavaScript, Java, Go, Ruby, PHP, C#, C, C++, TypeScript
///
/// # Construction
///
/// ```rust,no_run
/// use context_graph_embeddings::models::CodeModel;
/// use context_graph_embeddings::traits::SingleModelConfig;
/// use context_graph_embeddings::error::EmbeddingResult;
/// use std::path::Path;
///
/// async fn example() -> EmbeddingResult<()> {
///     let model = CodeModel::new(
///         Path::new("models/code"),
///         SingleModelConfig::default(),
///     )?;
///     model.load().await?;  // Must load before embed
///     Ok(())
/// }
/// ```
pub struct CodeModel {
    /// Model weights and inference engine.
    #[allow(dead_code)]
    pub(super) model_state: std::sync::RwLock<ModelState>,

    /// Path to model weights directory.
    #[allow(dead_code)]
    pub(super) model_path: PathBuf,

    /// Configuration for this model instance.
    #[allow(dead_code)]
    pub(super) config: SingleModelConfig,

    /// Whether model weights are loaded and ready.
    pub(super) loaded: AtomicBool,

    /// Memory used by model weights (bytes).
    #[allow(dead_code)]
    pub(super) memory_size: usize,
}

impl CodeModel {
    /// Create a new CodeModel instance.
    ///
    /// Model is NOT loaded after construction. Call `load()` before `embed()`.
    ///
    /// # Arguments
    /// * `model_path` - Path to directory containing model weights
    /// * `config` - Device placement and quantization settings
    ///
    /// # Errors
    /// - `EmbeddingError::ConfigError` if config validation fails
    pub fn new(model_path: &Path, config: SingleModelConfig) -> EmbeddingResult<Self> {
        if config.max_batch_size == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "max_batch_size cannot be zero".to_string(),
            });
        }

        Ok(Self {
            model_state: std::sync::RwLock::new(ModelState::Unloaded),
            model_path: model_path.to_path_buf(),
            config,
            loaded: AtomicBool::new(false),
            memory_size: 0,
        })
    }

    /// Load model weights into memory.
    ///
    /// # GPU Pipeline
    ///
    /// 1. Initialize CUDA device
    /// 2. Load config.json and tokenizer.json
    /// 3. Load model.safetensors
    /// 4. Transfer all weight tensors to GPU VRAM
    pub async fn load(&self) -> EmbeddingResult<()> {
        // Initialize GPU device
        let device = init_gpu().map_err(|e| EmbeddingError::GpuError {
            message: format!("CodeModel GPU init failed: {}", e),
        })?;

        // Load tokenizer from model directory
        let tokenizer_path = self.model_path.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| EmbeddingError::ModelLoadError {
                model_id: ModelId::Code,
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "Tokenizer load failed at {}: {}",
                        tokenizer_path.display(),
                        e
                    ),
                )),
            })?;

        // Load weights from safetensors
        let weights = CodeT5pWeights::from_path(&self.model_path, device)?;

        tracing::info!(
            "CodeModel loaded: {} layers, d_model={}, embed_dim={}",
            weights.config.num_layers,
            weights.config.d_model,
            weights.config.embed_dim
        );

        // Update state
        let mut state = self
            .model_state
            .write()
            .map_err(|e| EmbeddingError::InternalError {
                message: format!("Failed to acquire write lock: {}", e),
            })?;

        *state = ModelState::Loaded {
            weights,
            tokenizer: Box::new(tokenizer),
        };
        self.loaded.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Unload model weights from memory.
    pub async fn unload(&self) -> EmbeddingResult<()> {
        if !self.is_initialized() {
            return Err(EmbeddingError::NotInitialized {
                model_id: self.model_id(),
            });
        }

        let mut state = self
            .model_state
            .write()
            .map_err(|e| EmbeddingError::InternalError {
                message: format!("Failed to acquire write lock: {}", e),
            })?;

        *state = ModelState::Unloaded;
        self.loaded.store(false, Ordering::SeqCst);
        tracing::info!("CodeModel unloaded");
        Ok(())
    }

    /// Embed a batch of inputs (more efficient than single embed).
    pub async fn embed_batch(&self, inputs: &[ModelInput]) -> EmbeddingResult<Vec<ModelEmbedding>> {
        if !self.is_initialized() {
            return Err(EmbeddingError::NotInitialized {
                model_id: self.model_id(),
            });
        }

        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.embed(input).await?);
        }
        Ok(results)
    }

    /// Extract text content from model input for embedding.
    fn extract_content(input: &ModelInput) -> EmbeddingResult<String> {
        match input {
            ModelInput::Text {
                content,
                instruction,
            } => {
                let mut full = content.clone();
                if let Some(inst) = instruction {
                    full = format!("{} {}", inst, full);
                }
                Ok(full)
            }
            ModelInput::Code { content, language } => {
                // Prepend language hint for better code understanding
                Ok(format!("// Language: {}\n{}", language, content))
            }
            _ => Err(EmbeddingError::UnsupportedModality {
                model_id: ModelId::Code,
                input_type: InputType::from(input),
            }),
        }
    }
}

#[async_trait]
impl EmbeddingModel for CodeModel {
    fn model_id(&self) -> ModelId {
        ModelId::Code
    }

    fn supported_input_types(&self) -> &[InputType] {
        &[InputType::Code, InputType::Text]
    }

    fn is_initialized(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
        if !self.is_initialized() {
            return Err(EmbeddingError::NotInitialized {
                model_id: self.model_id(),
            });
        }

        self.validate_input(input)?;

        let start = std::time::Instant::now();

        // Extract content for embedding
        let content = Self::extract_content(input)?;

        let state = self
            .model_state
            .read()
            .map_err(|e| EmbeddingError::InternalError {
                message: format!("CodeModel failed to acquire read lock: {}", e),
            })?;

        match &*state {
            ModelState::Loaded { weights, tokenizer } => {
                let vector = gpu_forward(&content, weights, tokenizer)?;
                let latency_us = start.elapsed().as_micros() as u64;
                Ok(ModelEmbedding::new(ModelId::Code, vector, latency_us))
            }
            _ => Err(EmbeddingError::NotInitialized {
                model_id: ModelId::Code,
            }),
        }
    }
}

// Implement Send and Sync manually since RwLock is involved
unsafe impl Send for CodeModel {}
unsafe impl Sync for CodeModel {}
