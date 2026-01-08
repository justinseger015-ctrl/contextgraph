//! CausalModel struct and implementation.
//!
//! This module contains the main CausalModel struct, its constructor,
//! load/unload methods, embed methods, and the EmbeddingModel trait implementation.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use tokenizers::Tokenizer;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::init_gpu;
use crate::traits::{EmbeddingModel, SingleModelConfig};
use crate::types::{InputType, ModelEmbedding, ModelId, ModelInput};

use super::config::DEFAULT_ATTENTION_WINDOW;
use super::forward::gpu_forward;
use super::loader::load_longformer_weights;
use super::weights::LongformerWeights;

/// Internal state that varies based on feature flags.
#[allow(dead_code)]
pub(crate) enum ModelState {
    /// Unloaded - no weights in memory.
    Unloaded,

    /// Loaded with candle model and tokenizer (GPU-accelerated).
    Loaded {
        /// Longformer model weights on GPU.
        weights: LongformerWeights,
        /// HuggingFace tokenizer for text encoding (boxed to reduce enum size).
        tokenizer: Box<Tokenizer>,
    },
}

/// Causal embedding model using allenai/longformer-base-4096.
///
/// This model produces 768D vectors optimized for causal reasoning.
/// Uses sliding window attention + global attention for efficient
/// processing of documents up to 4096 tokens.
///
/// # Attention Mechanism
///
/// Longformer uses a combination of:
/// - **Sliding window attention**: Each token attends to `window_size` neighbors
/// - **Global attention**: Selected tokens (e.g., [CLS]) attend to all tokens
///
/// This allows O(n x w) complexity instead of O(n^2) for long sequences.
///
/// # Construction
///
/// ```rust,no_run
/// use context_graph_embeddings::models::CausalModel;
/// use context_graph_embeddings::traits::SingleModelConfig;
/// use context_graph_embeddings::error::EmbeddingResult;
/// use std::path::Path;
///
/// async fn example() -> EmbeddingResult<()> {
///     let model = CausalModel::new(
///         Path::new("models/causal"),
///         SingleModelConfig::default(),
///     )?;
///     model.load().await?;  // Must load before embed
///     Ok(())
/// }
/// ```
pub struct CausalModel {
    /// Model weights and inference engine.
    #[allow(dead_code)]
    model_state: std::sync::RwLock<ModelState>,

    /// Path to model weights directory.
    #[allow(dead_code)]
    model_path: PathBuf,

    /// Configuration for this model instance.
    #[allow(dead_code)]
    config: SingleModelConfig,

    /// Whether model weights are loaded and ready.
    loaded: AtomicBool,

    /// Memory used by model weights (bytes).
    #[allow(dead_code)]
    memory_size: usize,

    /// Attention window size for sliding window attention.
    attention_window: usize,

    /// Token indices that receive global attention (e.g., [CLS] at 0).
    global_attention_tokens: Vec<usize>,
}

impl CausalModel {
    /// Create a new CausalModel instance.
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
            attention_window: DEFAULT_ATTENTION_WINDOW,
            global_attention_tokens: vec![0], // [CLS] token by default
        })
    }

    /// Configure which token indices receive global attention.
    ///
    /// By default, only the [CLS] token (index 0) has global attention.
    /// Additional tokens can be added for task-specific needs.
    ///
    /// # Arguments
    /// * `tokens` - Indices of tokens that should attend globally
    pub fn set_global_attention_tokens(&mut self, tokens: &[usize]) {
        self.global_attention_tokens = tokens.to_vec();
    }

    /// Get the attention window size for sliding attention.
    #[inline]
    #[must_use]
    pub fn attention_window(&self) -> usize {
        self.attention_window
    }

    /// Get the current global attention token indices.
    #[inline]
    #[must_use]
    pub fn global_attention_tokens(&self) -> &[usize] {
        &self.global_attention_tokens
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
            message: format!("CausalModel GPU init failed: {}", e),
        })?;

        // Load tokenizer from model directory
        let tokenizer_path = self.model_path.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| EmbeddingError::ModelLoadError {
                model_id: ModelId::Causal,
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
        let weights = load_longformer_weights(&self.model_path, device)?;

        tracing::info!(
            "CausalModel loaded: {} layers, hidden_size={}",
            weights.config.num_hidden_layers,
            weights.config.hidden_size
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
        tracing::info!("CausalModel unloaded");
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
}

#[async_trait]
impl EmbeddingModel for CausalModel {
    fn model_id(&self) -> ModelId {
        ModelId::Causal
    }

    fn supported_input_types(&self) -> &[InputType] {
        &[InputType::Text]
    }

    fn is_initialized(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    async fn load(&self) -> EmbeddingResult<()> {
        CausalModel::load(self).await
    }

    async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
        if !self.is_initialized() {
            return Err(EmbeddingError::NotInitialized {
                model_id: self.model_id(),
            });
        }

        self.validate_input(input)?;

        let start = std::time::Instant::now();

        // Extract text content
        let text_content = match input {
            ModelInput::Text {
                content,
                instruction,
            } => {
                let mut full = content.clone();
                if let Some(inst) = instruction {
                    full = format!("{} {}", inst, full);
                }
                full
            }
            _ => {
                return Err(EmbeddingError::UnsupportedModality {
                    model_id: ModelId::Causal,
                    input_type: InputType::from(input),
                });
            }
        };

        let state = self
            .model_state
            .read()
            .map_err(|e| EmbeddingError::InternalError {
                message: format!("CausalModel failed to acquire read lock: {}", e),
            })?;

        match &*state {
            ModelState::Loaded { weights, tokenizer } => {
                let vector = gpu_forward(&text_content, weights, tokenizer)?;
                let latency_us = start.elapsed().as_micros() as u64;
                Ok(ModelEmbedding::new(ModelId::Causal, vector, latency_us))
            }
            _ => Err(EmbeddingError::NotInitialized {
                model_id: ModelId::Causal,
            }),
        }
    }
}

// Implement Send and Sync manually since RwLock is involved
unsafe impl Send for CausalModel {}
unsafe impl Sync for CausalModel {}
