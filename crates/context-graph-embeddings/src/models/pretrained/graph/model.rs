//! Core GraphModel struct and lifecycle management.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::{init_gpu, GpuModelLoader};
use crate::traits::{EmbeddingModel, SingleModelConfig};
use crate::types::{InputType, ModelEmbedding, ModelId, ModelInput};

use super::constants::GRAPH_DIMENSION;
use super::encoding::{encode_context, encode_relation};
use super::forward::gpu_forward;
use super::state::ModelState;

/// Graph embedding model using sentence-transformers/paraphrase-MiniLM-L6-v2.
///
/// Produces 384D vectors optimized for knowledge graph embeddings,
/// relation encoding, and graph structure understanding.
///
/// MiniLM is a distilled BERT model optimized for speed while maintaining
/// good semantic understanding. The paraphrase variant is trained on
/// paraphrase and semantic similarity data.
///
/// # Example
/// ```rust,no_run
/// use context_graph_embeddings::models::GraphModel;
/// use context_graph_embeddings::traits::SingleModelConfig;
/// use context_graph_embeddings::error::EmbeddingResult;
/// use std::path::Path;
///
/// async fn example() -> EmbeddingResult<()> {
///     let model = GraphModel::new(Path::new("models/graph"), SingleModelConfig::default())?;
///     model.load().await?;
///     let relation_text = GraphModel::encode_relation("Alice", "works_at", "Anthropic");
///     Ok(())
/// }
/// ```
pub struct GraphModel {
    #[allow(dead_code)]
    pub(crate) model_state: std::sync::RwLock<ModelState>,
    #[allow(dead_code)]
    pub(crate) model_path: PathBuf,
    #[allow(dead_code)]
    pub(crate) config: SingleModelConfig,
    pub(crate) loaded: AtomicBool,
    #[allow(dead_code)]
    pub(crate) memory_size: usize,
}

impl GraphModel {
    /// Create a new GraphModel instance. Call `load()` before `embed()`.
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

    /// Encode a relation triple into a text string for embedding.
    pub fn encode_relation(subject: &str, predicate: &str, object: &str) -> String {
        encode_relation(subject, predicate, object)
    }

    /// Encode a node with its neighboring relations into a context string.
    pub fn encode_context(node: &str, neighbors: &[(String, String)]) -> String {
        encode_context(node, neighbors)
    }

    /// Load model weights into memory.
    ///
    /// Initializes CUDA device, loads tokenizer.json and model.safetensors,
    /// and transfers weight tensors to GPU VRAM.
    pub async fn load(&self) -> EmbeddingResult<()> {
        let _device = init_gpu().map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel GPU init failed: {}", e),
        })?;

        let tokenizer_path = self.model_path.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            EmbeddingError::ModelLoadError {
                model_id: ModelId::Graph,
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "GraphModel tokenizer load failed at {}: {}",
                        tokenizer_path.display(),
                        e
                    ),
                )),
            }
        })?;

        let loader = GpuModelLoader::new().map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel loader init failed: {}", e),
        })?;

        let weights = loader.load_bert_weights(&self.model_path).map_err(|e| {
            EmbeddingError::ModelLoadError {
                model_id: ModelId::Graph,
                source: Box::new(std::io::Error::other(format!(
                    "GraphModel weight load failed: {}",
                    e
                ))),
            }
        })?;

        if weights.config.hidden_size != GRAPH_DIMENSION {
            return Err(EmbeddingError::InvalidDimension {
                expected: GRAPH_DIMENSION,
                actual: weights.config.hidden_size,
            });
        }

        tracing::info!(
            "GraphModel loaded: {} params, {:.2} MB VRAM, hidden_size={}",
            weights.param_count(),
            weights.vram_bytes() as f64 / (1024.0 * 1024.0),
            weights.config.hidden_size
        );

        let mut state = self
            .model_state
            .write()
            .map_err(|e| EmbeddingError::InternalError {
                message: format!("GraphModel failed to acquire write lock: {}", e),
            })?;
        *state = ModelState::Loaded {
            weights: Box::new(weights),
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
        tracing::info!("GraphModel unloaded");
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
            _ => Err(EmbeddingError::UnsupportedModality {
                model_id: ModelId::Graph,
                input_type: InputType::from(input),
            }),
        }
    }
}

#[async_trait]
impl EmbeddingModel for GraphModel {
    fn model_id(&self) -> ModelId {
        ModelId::Graph
    }

    fn supported_input_types(&self) -> &[InputType] {
        &[InputType::Text]
    }

    fn is_initialized(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
        if !self.is_initialized() {
            return Err(EmbeddingError::NotInitialized {
                model_id: ModelId::Graph,
            });
        }
        self.validate_input(input)?;
        let content = Self::extract_content(input)?;
        let start = std::time::Instant::now();

        let state = self
            .model_state
            .read()
            .map_err(|e| EmbeddingError::InternalError {
                message: format!("GraphModel failed to acquire read lock: {}", e),
            })?;

        let (weights, tokenizer) = match &*state {
            ModelState::Loaded { weights, tokenizer } => (weights, tokenizer),
            _ => {
                return Err(EmbeddingError::NotInitialized {
                    model_id: ModelId::Graph,
                })
            }
        };

        let vector = gpu_forward(&content, weights, tokenizer)?;
        let latency_us = start.elapsed().as_micros() as u64;
        Ok(ModelEmbedding::new(ModelId::Graph, vector, latency_us))
    }
}

unsafe impl Send for GraphModel {}
unsafe impl Sync for GraphModel {}
