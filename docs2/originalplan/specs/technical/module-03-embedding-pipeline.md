# Module 3: Embedding Pipeline - Technical Specification

```yaml
metadata:
  id: TECH-EMBED-003
  version: 1.0.0
  module: Embedding Pipeline
  phase: 1
  status: draft
  created: 2025-12-31
  functional_spec_ref: SPEC-EMBED-PIPELINE
  author: Architecture Agent
  dependencies:
    - TECH-GHOST-001 (Module 1: Ghost System)
    - TECH-CORE-002 (Module 2: Core Infrastructure)
```

---

## 1. Architecture Overview

### 1.1 Module Dependency Graph

```
                    +---------------------------+
                    |   context-graph-mcp       |
                    |      (binary crate)       |
                    +---------------------------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
+------------------------+        +---------------------------+
| context-graph-core     |        | context-graph-embeddings  |
|  (MemoryNode, Types)   |        |  (12-Model Pipeline)      |
+------------------------+        +---------------------------+
                                              |
              +---------------+---------------+---------------+
              |               |               |               |
              v               v               v               v
      +-----------+   +-----------+   +-----------+   +-----------+
      | Model     |   | Batch     |   | FuseMoE   |   | Cache     |
      | Registry  |   | Processor |   | Fusion    |   | Manager   |
      +-----------+   +-----------+   +-----------+   +-----------+
              |               |               |               |
              v               v               v               v
+------------------------------------------------------------------+
|                    context-graph-cuda                             |
|              (CUDA Kernels for GPU Acceleration)                  |
+------------------------------------------------------------------+
```

### 1.2 Module Structure

```
context-graph/
├── crates/
│   └── context-graph-embeddings/
│       ├── src/
│       │   ├── lib.rs                    # Public API exports
│       │   ├── pipeline.rs               # Main EmbeddingPipeline struct
│       │   ├── config.rs                 # Configuration types
│       │   │
│       │   ├── models/
│       │   │   ├── mod.rs                # Model module exports
│       │   │   ├── registry.rs           # ModelRegistry implementation
│       │   │   ├── loader.rs             # Model loading utilities
│       │   │   ├── base.rs               # EmbeddingModel trait
│       │   │   │
│       │   │   ├── text/
│       │   │   │   ├── mod.rs
│       │   │   │   ├── minilm.rs         # E1: all-MiniLM-L6
│       │   │   │   ├── bge.rs            # E2: bge-large-en
│       │   │   │   ├── instructor.rs     # E3: instructor-xl
│       │   │   │   ├── sentence_t5.rs    # E7: SentenceT5-XXL
│       │   │   │   ├── mpnet.rs          # E8: MPNet-base
│       │   │   │   ├── contriever.rs     # E9: Contriever
│       │   │   │   ├── dragon.rs         # E10: DRAGON+
│       │   │   │   ├── gte.rs            # E11: GTE-large
│       │   │   │   └── e5_mistral.rs     # E12: E5-mistral-7b
│       │   │   │
│       │   │   ├── multimodal/
│       │   │   │   ├── mod.rs
│       │   │   │   ├── clip.rs           # E4: CLIP-ViT-L
│       │   │   │   └── whisper.rs        # E5: Whisper-large
│       │   │   │
│       │   │   └── code/
│       │   │       ├── mod.rs
│       │   │       └── codebert.rs       # E6: CodeBERT
│       │   │
│       │   ├── batch/
│       │   │   ├── mod.rs                # Batch module exports
│       │   │   ├── processor.rs          # BatchProcessor implementation
│       │   │   ├── scheduler.rs          # Dynamic batch scheduling
│       │   │   ├── padding.rs            # Token padding utilities
│       │   │   └── queue.rs              # Async batch queue
│       │   │
│       │   ├── fusion/
│       │   │   ├── mod.rs                # Fusion module exports
│       │   │   ├── fusemoe.rs            # FuseMoE implementation
│       │   │   ├── gating.rs             # Gating network
│       │   │   ├── experts.rs            # Expert networks
│       │   │   └── router.rs             # Top-k routing
│       │   │
│       │   ├── cache/
│       │   │   ├── mod.rs                # Cache module exports
│       │   │   ├── manager.rs            # CacheManager implementation
│       │   │   ├── lru.rs                # LRU eviction policy
│       │   │   ├── content_hash.rs       # Content-based hashing
│       │   │   └── persistence.rs        # Disk persistence
│       │   │
│       │   └── error.rs                  # Error types
│       │
│       ├── benches/
│       │   ├── embedding_bench.rs        # Embedding benchmarks
│       │   └── fusion_bench.rs           # FuseMoE benchmarks
│       │
│       └── Cargo.toml
│
├── crates/context-graph-cuda/
│   ├── src/
│   │   ├── lib.rs                        # CUDA bindings
│   │   ├── kernels/
│   │   │   ├── mod.rs
│   │   │   ├── embedding.rs              # Embedding CUDA ops
│   │   │   ├── fusion.rs                 # FuseMoE CUDA ops
│   │   │   └── similarity.rs             # Similarity CUDA ops
│   │   └── memory.rs                     # GPU memory management
│   │
│   ├── kernels/
│   │   ├── embedding.cu                  # CUDA embedding kernels
│   │   ├── fusemoe.cu                    # CUDA FuseMoE kernels
│   │   └── similarity.cu                 # CUDA similarity kernels
│   │
│   └── Cargo.toml
```

---

## 2. Data Structures

### 2.1 Embedding Types

```rust
// crates/context-graph-embeddings/src/lib.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model identifier for the 12-model ensemble
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ModelId {
    /// E1: all-MiniLM-L6 (384D) - Fast semantic
    MiniLM = 1,
    /// E2: bge-large-en (1024D) - Deep semantic
    BgeLarge = 2,
    /// E3: instructor-xl (768D) - Task-specific
    InstructorXL = 3,
    /// E4: CLIP-ViT-L (768D) - Visual
    ClipViT = 4,
    /// E5: Whisper-large (1280D) - Audio
    WhisperLarge = 5,
    /// E6: CodeBERT (768D) - Code
    CodeBERT = 6,
    /// E7: SentenceT5-XXL (768D) - Sentence
    SentenceT5 = 7,
    /// E8: MPNet-base (768D) - Paraphrase
    MPNet = 8,
    /// E9: Contriever (768D) - Retrieval
    Contriever = 9,
    /// E10: DRAGON+ (768D) - Dense retrieval
    DragonPlus = 10,
    /// E11: GTE-large (1024D) - General
    GteLarge = 11,
    /// E12: E5-mistral-7b (4096D) - LLM-based
    E5Mistral = 12,
}

impl ModelId {
    /// Get the output dimension for this model
    pub const fn dimension(&self) -> usize {
        match self {
            Self::MiniLM => 384,
            Self::BgeLarge => 1024,
            Self::InstructorXL => 768,
            Self::ClipViT => 768,
            Self::WhisperLarge => 1280,
            Self::CodeBERT => 768,
            Self::SentenceT5 => 768,
            Self::MPNet => 768,
            Self::Contriever => 768,
            Self::DragonPlus => 768,
            Self::GteLarge => 1024,
            Self::E5Mistral => 4096,
        }
    }

    /// Get the total dimension across all models (for multi-array storage sizing).
    /// NOTE: This is NOT used for fusion/concatenation. Each embedding is stored separately.
    pub const fn total_dimension() -> usize {
        384 + 1024 + 768 + 768 + 1280 + 768 + 768 + 768 + 768 + 768 + 1024 + 4096
        // Total: 12,954 dimensions across all 12 models
        // Each model's embedding is stored SEPARATELY in multi-array storage
    }

    /// Get all model IDs in order
    pub fn all() -> &'static [ModelId] {
        &[
            Self::MiniLM,
            Self::BgeLarge,
            Self::InstructorXL,
            Self::ClipViT,
            Self::WhisperLarge,
            Self::CodeBERT,
            Self::SentenceT5,
            Self::MPNet,
            Self::Contriever,
            Self::DragonPlus,
            Self::GteLarge,
            Self::E5Mistral,
        ]
    }

    /// Get model name for logging and display
    pub const fn name(&self) -> &'static str {
        match self {
            Self::MiniLM => "all-MiniLM-L6-v2",
            Self::BgeLarge => "bge-large-en-v1.5",
            Self::InstructorXL => "instructor-xl",
            Self::ClipViT => "clip-vit-large-patch14",
            Self::WhisperLarge => "whisper-large-v3",
            Self::CodeBERT => "codebert-base",
            Self::SentenceT5 => "sentence-t5-xxl",
            Self::MPNet => "all-mpnet-base-v2",
            Self::Contriever => "contriever-msmarco",
            Self::DragonPlus => "dragon-plus-context-encoder",
            Self::GteLarge => "gte-large-en-v1.5",
            Self::E5Mistral => "e5-mistral-7b-instruct",
        }
    }

    /// Get recommended max sequence length
    pub const fn max_seq_length(&self) -> usize {
        match self {
            Self::MiniLM => 256,
            Self::BgeLarge => 512,
            Self::InstructorXL => 512,
            Self::ClipViT => 77,        // CLIP text tokens
            Self::WhisperLarge => 448,  // 30 seconds at 16kHz
            Self::CodeBERT => 512,
            Self::SentenceT5 => 512,
            Self::MPNet => 384,
            Self::Contriever => 512,
            Self::DragonPlus => 512,
            Self::GteLarge => 8192,
            Self::E5Mistral => 4096,
        }
    }
}

/// Raw embedding output from a single model
#[derive(Debug, Clone)]
pub struct ModelEmbedding {
    /// Source model identifier
    pub model_id: ModelId,
    /// Embedding vector (dimension varies by model)
    pub vector: Vec<f32>,
    /// Processing time in microseconds
    pub latency_us: u64,
    /// Optional attention weights (for debugging)
    pub attention_weights: Option<Vec<f32>>,
}

/// Concatenated embeddings from all models
#[derive(Debug, Clone)]
pub struct ConcatenatedEmbedding {
    /// Individual model embeddings in order
    pub embeddings: Vec<ModelEmbedding>,
    /// Concatenated vector (12,954D)
    pub concatenated: Vec<f32>,
    /// Total processing time in microseconds
    pub total_latency_us: u64,
}

impl ConcatenatedEmbedding {
    /// Validate that all expected embeddings are present
    pub fn is_complete(&self) -> bool {
        self.embeddings.len() == 12 &&
        self.concatenated.len() == ModelId::total_dimension()
    }
}

/// Final fused embedding output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedEmbedding {
    /// Unified embedding vector (1536D)
    pub vector: Vec<f32>,
    /// Expert routing weights (8 experts)
    pub expert_weights: [f32; 8],
    /// Selected experts (top-k=2)
    pub selected_experts: [u8; 2],
    /// Total pipeline latency in microseconds
    pub pipeline_latency_us: u64,
    /// Input content hash for cache lookup
    pub content_hash: u64,
}

impl FusedEmbedding {
    /// Dimension of the fused embedding
    pub const DIMENSION: usize = 1536;

    /// Number of experts in FuseMoE
    pub const NUM_EXPERTS: usize = 8;

    /// Top-k routing parameter
    pub const TOP_K: usize = 2;

    /// Validate the fused embedding
    pub fn validate(&self) -> Result<(), EmbeddingError> {
        if self.vector.len() != Self::DIMENSION {
            return Err(EmbeddingError::InvalidDimension {
                expected: Self::DIMENSION,
                actual: self.vector.len(),
            });
        }

        // Check for NaN/Inf values
        for (i, &v) in self.vector.iter().enumerate() {
            if !v.is_finite() {
                return Err(EmbeddingError::InvalidValue {
                    index: i,
                    value: v,
                });
            }
        }

        Ok(())
    }

    /// Normalize the embedding to unit length
    pub fn normalize(&mut self) {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut self.vector {
                *v /= norm;
            }
        }
    }
}
```

### 2.2 Configuration Types

```rust
// crates/context-graph-embeddings/src/config.rs

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use crate::ModelId;

/// Configuration for the embedding pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model registry configuration
    pub models: ModelRegistryConfig,
    /// Batch processing configuration
    pub batch: BatchConfig,
    /// FuseMoE fusion configuration
    pub fusion: FusionConfig,
    /// Cache configuration
    pub cache: CacheConfig,
    /// GPU configuration
    pub gpu: GpuConfig,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            models: ModelRegistryConfig::default(),
            batch: BatchConfig::default(),
            fusion: FusionConfig::default(),
            cache: CacheConfig::default(),
            gpu: GpuConfig::default(),
        }
    }
}

/// Model registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistryConfig {
    /// Directory for model weights
    pub models_dir: PathBuf,
    /// Enable lazy loading of models
    pub lazy_loading: bool,
    /// Models to preload at startup
    pub preload_models: Vec<ModelId>,
    /// Model-specific configurations
    pub model_configs: Vec<SingleModelConfig>,
    /// Download models if not present
    pub auto_download: bool,
    /// Hugging Face hub cache directory
    pub hf_cache_dir: Option<PathBuf>,
}

impl Default for ModelRegistryConfig {
    fn default() -> Self {
        Self {
            models_dir: PathBuf::from("~/.contextgraph/models"),
            lazy_loading: true,
            preload_models: vec![ModelId::MiniLM, ModelId::BgeLarge],
            model_configs: vec![],
            auto_download: true,
            hf_cache_dir: None,
        }
    }
}

/// Configuration for a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleModelConfig {
    /// Model identifier
    pub model_id: ModelId,
    /// Device placement (gpu:0, gpu:1, cpu)
    pub device: DevicePlacement,
    /// Quantization mode
    pub quantization: QuantizationMode,
    /// Override max sequence length
    pub max_seq_length: Option<usize>,
    /// Enable flash attention
    pub use_flash_attention: bool,
    /// Memory pool size in MB
    pub memory_pool_mb: Option<usize>,
}

/// Device placement for model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DevicePlacement {
    /// Run on CPU
    Cpu,
    /// Run on specific GPU
    Gpu(usize),
    /// Auto-place based on memory availability
    Auto,
}

impl Default for DevicePlacement {
    fn default() -> Self {
        Self::Auto
    }
}

/// Quantization mode for model weights
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QuantizationMode {
    /// Full precision (FP32)
    None,
    /// Half precision (FP16)
    Fp16,
    /// Brain float (BF16)
    Bf16,
    /// 8-bit integer quantization
    Int8,
    /// 4-bit quantization (GPTQ/AWQ)
    Int4,
}

impl Default for QuantizationMode {
    fn default() -> Self {
        Self::Fp16
    }
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Minimum batch size before processing
    pub min_batch_size: usize,
    /// Maximum wait time for batch accumulation (ms)
    pub max_wait_ms: u64,
    /// Enable dynamic batching
    pub dynamic_batching: bool,
    /// Sort by sequence length for efficiency
    pub sort_by_length: bool,
    /// Padding strategy
    pub padding: PaddingStrategy,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            min_batch_size: 1,
            max_wait_ms: 50,
            dynamic_batching: true,
            sort_by_length: true,
            padding: PaddingStrategy::DynamicMax,
        }
    }
}

/// Token padding strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// Pad to maximum model sequence length
    MaxLength,
    /// Pad to longest sequence in batch
    DynamicMax,
    /// Pad to nearest power of 2
    PowerOfTwo,
    /// Bucket into fixed sizes
    Bucket { sizes: [usize; 4] },
}

impl Default for PaddingStrategy {
    fn default() -> Self {
        Self::DynamicMax
    }
}

/// FuseMoE fusion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Top-k routing parameter
    pub top_k: usize,
    /// Output dimension (1536)
    pub output_dim: usize,
    /// Hidden dimension in experts
    pub expert_hidden_dim: usize,
    /// Enable auxiliary load balancing loss
    pub load_balance_loss: bool,
    /// Load balancing coefficient
    pub load_balance_coef: f32,
    /// Dropout rate for training
    pub dropout: f32,
    /// Enable expert capacity limiting
    pub capacity_factor: Option<f32>,
    /// Gating network configuration
    pub gating: GatingConfig,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            top_k: 2,
            output_dim: 1536,
            expert_hidden_dim: 4096,
            load_balance_loss: true,
            load_balance_coef: 0.01,
            dropout: 0.0,
            capacity_factor: Some(1.25),
            gating: GatingConfig::default(),
        }
    }
}

/// Gating network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingConfig {
    /// Input projection dimension
    pub projection_dim: usize,
    /// Noise factor for exploration
    pub noise_std: f32,
    /// Temperature for softmax
    pub temperature: f32,
    /// Enable layer normalization
    pub layer_norm: bool,
}

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            projection_dim: 2048,
            noise_std: 0.0,
            temperature: 1.0,
            layer_norm: true,
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable embedding cache
    pub enabled: bool,
    /// Maximum cache size in entries
    pub max_entries: usize,
    /// Maximum cache size in bytes
    pub max_bytes: usize,
    /// Time-to-live in seconds (0 = no expiry)
    pub ttl_seconds: u64,
    /// Enable disk persistence
    pub persist_to_disk: bool,
    /// Disk cache directory
    pub disk_cache_dir: Option<PathBuf>,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 100_000,
            max_bytes: 1024 * 1024 * 1024, // 1GB
            ttl_seconds: 0,
            persist_to_disk: true,
            disk_cache_dir: Some(PathBuf::from("~/.contextgraph/cache/embeddings")),
            eviction_policy: EvictionPolicy::Lru,
        }
    }
}

/// Cache eviction policy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// Time-based with LRU fallback
    TtlLru,
    /// Adaptive Replacement Cache
    Arc,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self::Lru
    }
}

/// GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    pub enabled: bool,
    /// GPU device indices to use
    pub device_ids: Vec<usize>,
    /// Memory fraction to use per GPU
    pub memory_fraction: f32,
    /// Enable CUDA graphs for repeated operations
    pub cuda_graphs: bool,
    /// Enable TensorRT optimization
    pub tensorrt: bool,
    /// Maximum GPU memory in bytes (0 = no limit)
    pub max_memory_bytes: usize,
    /// Enable mixed precision
    pub mixed_precision: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device_ids: vec![0],
            memory_fraction: 0.9,
            cuda_graphs: true,
            tensorrt: false,
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            mixed_precision: true,
        }
    }
}
```

### 2.3 Error Types

```rust
// crates/context-graph-embeddings/src/error.rs

use thiserror::Error;
use crate::ModelId;

/// Errors that can occur in the embedding pipeline
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Model {model_id:?} not found in registry")]
    ModelNotFound { model_id: ModelId },

    #[error("Model {model_id:?} failed to load: {reason}")]
    ModelLoadError { model_id: ModelId, reason: String },

    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    #[error("Invalid embedding value at index {index}: {value}")]
    InvalidValue { index: usize, value: f32 },

    #[error("Input exceeds maximum length {max_length} for model {model_id:?}")]
    InputTooLong { model_id: ModelId, max_length: usize },

    #[error("Empty input provided")]
    EmptyInput,

    #[error("Batch processing error: {0}")]
    BatchError(String),

    #[error("Fusion error: {0}")]
    FusionError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("GPU error: {0}")]
    GpuError(String),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    #[error("Timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("Pipeline not initialized")]
    NotInitialized,

    #[error("Unsupported modality {modality} for model {model_id:?}")]
    UnsupportedModality { model_id: ModelId, modality: String },
}

/// Result type alias for embedding operations
pub type EmbeddingResult<T> = Result<T, EmbeddingError>;
```

---

## 3. Core Components

### 3.1 EmbeddingModel Trait

```rust
// crates/context-graph-embeddings/src/models/base.rs

use async_trait::async_trait;
use crate::{EmbeddingResult, ModelEmbedding, ModelId};

/// Input types supported by embedding models
#[derive(Debug, Clone)]
pub enum ModelInput {
    /// Text input with optional instruction prefix
    Text {
        content: String,
        instruction: Option<String>,
    },
    /// Image input as raw bytes
    Image { bytes: Vec<u8>, format: ImageFormat },
    /// Audio input as raw bytes
    Audio {
        bytes: Vec<u8>,
        sample_rate: u32,
        channels: u8,
    },
    /// Code input with language hint
    Code { content: String, language: String },
}

#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    Png,
    Jpeg,
    WebP,
}

/// Trait for embedding models
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Get the model identifier
    fn model_id(&self) -> ModelId;

    /// Get the output embedding dimension
    fn dimension(&self) -> usize;

    /// Get the maximum sequence length
    fn max_sequence_length(&self) -> usize;

    /// Check if model is loaded and ready
    fn is_loaded(&self) -> bool;

    /// Load model weights into memory/GPU
    async fn load(&mut self) -> EmbeddingResult<()>;

    /// Unload model weights from memory
    async fn unload(&mut self) -> EmbeddingResult<()>;

    /// Generate embedding for a single input
    async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding>;

    /// Generate embeddings for a batch of inputs
    async fn embed_batch(&self, inputs: &[ModelInput]) -> EmbeddingResult<Vec<ModelEmbedding>>;

    /// Get current memory usage in bytes
    fn memory_usage(&self) -> usize;

    /// Get supported input types
    fn supported_inputs(&self) -> &[InputType];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputType {
    Text,
    Image,
    Audio,
    Code,
}
```

### 3.2 Model Registry

```rust
// crates/context-graph-embeddings/src/models/registry.rs

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::Semaphore;
use crate::{
    EmbeddingError, EmbeddingResult, EmbeddingModel, ModelId,
    ModelRegistryConfig, SingleModelConfig, DevicePlacement,
};

/// Registry for managing embedding models
pub struct ModelRegistry {
    /// Loaded models
    models: RwLock<HashMap<ModelId, Arc<dyn EmbeddingModel>>>,
    /// Configuration
    config: ModelRegistryConfig,
    /// Loading semaphore (prevent concurrent loads of same model)
    loading_locks: HashMap<ModelId, Semaphore>,
    /// Total GPU memory tracker
    gpu_memory_tracker: RwLock<GpuMemoryTracker>,
}

struct GpuMemoryTracker {
    /// Memory used per GPU device
    per_device: HashMap<usize, usize>,
    /// Maximum memory per device
    max_per_device: usize,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new(config: ModelRegistryConfig) -> Self {
        let loading_locks = ModelId::all()
            .iter()
            .map(|&id| (id, Semaphore::new(1)))
            .collect();

        Self {
            models: RwLock::new(HashMap::new()),
            config,
            loading_locks,
            gpu_memory_tracker: RwLock::new(GpuMemoryTracker {
                per_device: HashMap::new(),
                max_per_device: 8 * 1024 * 1024 * 1024, // 8GB default
            }),
        }
    }

    /// Initialize the registry, preloading configured models
    pub async fn initialize(&self) -> EmbeddingResult<()> {
        // Preload models in parallel with controlled concurrency
        let preload_semaphore = Semaphore::new(3); // Max 3 concurrent loads

        let futures: Vec<_> = self.config.preload_models
            .iter()
            .map(|&model_id| {
                let semaphore = &preload_semaphore;
                async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    self.load_model(model_id).await
                }
            })
            .collect();

        let results = futures::future::join_all(futures).await;

        for result in results {
            result?;
        }

        Ok(())
    }

    /// Get a model, loading it if necessary (lazy loading)
    pub async fn get_model(&self, model_id: ModelId) -> EmbeddingResult<Arc<dyn EmbeddingModel>> {
        // Fast path: check if already loaded
        if let Some(model) = self.models.read().get(&model_id) {
            return Ok(Arc::clone(model));
        }

        // Slow path: load the model
        if self.config.lazy_loading {
            self.load_model(model_id).await?;
            self.models.read()
                .get(&model_id)
                .map(Arc::clone)
                .ok_or(EmbeddingError::ModelNotFound { model_id })
        } else {
            Err(EmbeddingError::ModelNotFound { model_id })
        }
    }

    /// Load a specific model
    pub async fn load_model(&self, model_id: ModelId) -> EmbeddingResult<()> {
        // Acquire loading lock for this model
        let lock = self.loading_locks.get(&model_id)
            .ok_or(EmbeddingError::ModelNotFound { model_id })?;
        let _permit = lock.acquire().await.unwrap();

        // Double-check after acquiring lock
        if self.models.read().contains_key(&model_id) {
            return Ok(());
        }

        // Determine device placement
        let device = self.determine_device_placement(model_id)?;

        // Create and load the model
        let model = self.create_model(model_id, device).await?;

        // Register the model
        self.models.write().insert(model_id, model);

        tracing::info!("Loaded model {:?} on {:?}", model_id, device);

        Ok(())
    }

    /// Unload a specific model
    pub async fn unload_model(&self, model_id: ModelId) -> EmbeddingResult<()> {
        let mut models = self.models.write();
        if let Some(model) = models.remove(&model_id) {
            // Update GPU memory tracker
            let memory = model.memory_usage();
            self.release_gpu_memory(model_id, memory);

            tracing::info!("Unloaded model {:?}, freed {} bytes", model_id, memory);
        }
        Ok(())
    }

    /// Determine optimal device placement for a model
    fn determine_device_placement(&self, model_id: ModelId) -> EmbeddingResult<DevicePlacement> {
        // Check for explicit configuration
        if let Some(cfg) = self.config.model_configs
            .iter()
            .find(|c| c.model_id == model_id)
        {
            return Ok(cfg.device.clone());
        }

        // Auto-placement based on model size and available memory
        let estimated_memory = self.estimate_model_memory(model_id);
        let tracker = self.gpu_memory_tracker.read();

        for (device_id, &used) in &tracker.per_device {
            let available = tracker.max_per_device.saturating_sub(used);
            if available >= estimated_memory {
                return Ok(DevicePlacement::Gpu(*device_id));
            }
        }

        // Fallback to CPU
        Ok(DevicePlacement::Cpu)
    }

    /// Estimate memory requirements for a model
    fn estimate_model_memory(&self, model_id: ModelId) -> usize {
        match model_id {
            ModelId::MiniLM => 90 * 1024 * 1024,        // ~90MB
            ModelId::BgeLarge => 1300 * 1024 * 1024,   // ~1.3GB
            ModelId::InstructorXL => 5000 * 1024 * 1024, // ~5GB
            ModelId::ClipViT => 1500 * 1024 * 1024,    // ~1.5GB
            ModelId::WhisperLarge => 3000 * 1024 * 1024, // ~3GB
            ModelId::CodeBERT => 500 * 1024 * 1024,    // ~500MB
            ModelId::SentenceT5 => 5000 * 1024 * 1024, // ~5GB
            ModelId::MPNet => 420 * 1024 * 1024,       // ~420MB
            ModelId::Contriever => 440 * 1024 * 1024,  // ~440MB
            ModelId::DragonPlus => 440 * 1024 * 1024,  // ~440MB
            ModelId::GteLarge => 1400 * 1024 * 1024,   // ~1.4GB
            ModelId::E5Mistral => 14000 * 1024 * 1024, // ~14GB
        }
    }

    /// Create a model instance
    async fn create_model(
        &self,
        model_id: ModelId,
        device: DevicePlacement,
    ) -> EmbeddingResult<Arc<dyn EmbeddingModel>> {
        // Implementation delegates to specific model factories
        match model_id {
            ModelId::MiniLM => self.create_minilm(device).await,
            ModelId::BgeLarge => self.create_bge_large(device).await,
            ModelId::InstructorXL => self.create_instructor_xl(device).await,
            ModelId::ClipViT => self.create_clip_vit(device).await,
            ModelId::WhisperLarge => self.create_whisper_large(device).await,
            ModelId::CodeBERT => self.create_codebert(device).await,
            ModelId::SentenceT5 => self.create_sentence_t5(device).await,
            ModelId::MPNet => self.create_mpnet(device).await,
            ModelId::Contriever => self.create_contriever(device).await,
            ModelId::DragonPlus => self.create_dragon_plus(device).await,
            ModelId::GteLarge => self.create_gte_large(device).await,
            ModelId::E5Mistral => self.create_e5_mistral(device).await,
        }
    }

    /// Release GPU memory when unloading a model
    fn release_gpu_memory(&self, _model_id: ModelId, memory: usize) {
        // Update tracker (implementation detail)
        let _ = memory;
    }

    /// Get statistics about loaded models
    pub fn stats(&self) -> RegistryStats {
        let models = self.models.read();
        RegistryStats {
            loaded_models: models.len(),
            total_memory: models.values().map(|m| m.memory_usage()).sum(),
            model_ids: models.keys().copied().collect(),
        }
    }
}

/// Statistics about the model registry
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub loaded_models: usize,
    pub total_memory: usize,
    pub model_ids: Vec<ModelId>,
}
```

### 3.3 Batch Processor

```rust
// crates/context-graph-embeddings/src/batch/processor.rs

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;
use crate::{
    BatchConfig, EmbeddingError, EmbeddingResult, ModelEmbedding,
    ModelId, ModelInput, ModelRegistry, PaddingStrategy,
};

/// Request for batch processing
pub struct BatchRequest {
    /// Input to embed
    pub input: ModelInput,
    /// Target model
    pub model_id: ModelId,
    /// Response channel
    pub response_tx: oneshot::Sender<EmbeddingResult<ModelEmbedding>>,
    /// Request timestamp for latency tracking
    pub submitted_at: Instant,
}

/// Batch processor for efficient batched inference
pub struct BatchProcessor {
    /// Configuration
    config: BatchConfig,
    /// Model registry reference
    registry: Arc<ModelRegistry>,
    /// Per-model batch queues
    queues: HashMap<ModelId, BatchQueue>,
}

struct BatchQueue {
    /// Pending requests
    requests: Vec<BatchRequest>,
    /// Last batch trigger time
    last_batch_time: Instant,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(config: BatchConfig, registry: Arc<ModelRegistry>) -> Self {
        let queues = ModelId::all()
            .iter()
            .map(|&id| (id, BatchQueue {
                requests: Vec::with_capacity(config.max_batch_size),
                last_batch_time: Instant::now(),
            }))
            .collect();

        Self { config, registry, queues }
    }

    /// Submit a single embedding request
    pub async fn submit(&self, input: ModelInput, model_id: ModelId) -> EmbeddingResult<ModelEmbedding> {
        let (response_tx, response_rx) = oneshot::channel();

        let request = BatchRequest {
            input,
            model_id,
            response_tx,
            submitted_at: Instant::now(),
        };

        // Add to queue and potentially trigger batch
        self.add_to_queue(request).await?;

        // Wait for response
        response_rx.await.map_err(|_| EmbeddingError::BatchError(
            "Response channel closed".to_string()
        ))?
    }

    /// Add request to appropriate queue
    async fn add_to_queue(&self, request: BatchRequest) -> EmbeddingResult<()> {
        let model_id = request.model_id;
        let queue = self.queues.get_mut(&model_id)
            .ok_or(EmbeddingError::ModelNotFound { model_id })?;

        queue.requests.push(request);

        // Check if we should process the batch
        let should_process = queue.requests.len() >= self.config.max_batch_size
            || (queue.requests.len() >= self.config.min_batch_size
                && queue.last_batch_time.elapsed() >= Duration::from_millis(self.config.max_wait_ms));

        if should_process {
            self.process_batch(model_id).await?;
        }

        Ok(())
    }

    /// Process a batch of requests for a specific model
    async fn process_batch(&self, model_id: ModelId) -> EmbeddingResult<()> {
        let queue = self.queues.get_mut(&model_id)
            .ok_or(EmbeddingError::ModelNotFound { model_id })?;

        if queue.requests.is_empty() {
            return Ok(());
        }

        // Take requests from queue
        let requests: Vec<_> = queue.requests.drain(..).collect();
        queue.last_batch_time = Instant::now();

        let batch_size = requests.len();
        tracing::debug!("Processing batch of {} for {:?}", batch_size, model_id);

        // Sort by sequence length if configured
        let sorted_requests = if self.config.sort_by_length {
            self.sort_by_sequence_length(requests)
        } else {
            requests
        };

        // Prepare inputs
        let inputs: Vec<_> = sorted_requests.iter()
            .map(|r| r.input.clone())
            .collect();

        // Pad inputs according to strategy
        let padded_inputs = self.pad_inputs(&inputs, model_id)?;

        // Get model and run batch inference
        let model = self.registry.get_model(model_id).await?;
        let results = model.embed_batch(&padded_inputs).await;

        // Distribute results to requesters
        match results {
            Ok(embeddings) => {
                for (request, embedding) in sorted_requests.into_iter().zip(embeddings) {
                    let _ = request.response_tx.send(Ok(embedding));
                }
            }
            Err(e) => {
                for request in sorted_requests {
                    let _ = request.response_tx.send(Err(EmbeddingError::BatchError(
                        e.to_string()
                    )));
                }
            }
        }

        Ok(())
    }

    /// Sort requests by sequence length for efficient batching
    fn sort_by_sequence_length(&self, mut requests: Vec<BatchRequest>) -> Vec<BatchRequest> {
        requests.sort_by_key(|r| self.estimate_sequence_length(&r.input));
        requests
    }

    /// Estimate sequence length for an input
    fn estimate_sequence_length(&self, input: &ModelInput) -> usize {
        match input {
            ModelInput::Text { content, instruction } => {
                content.len() + instruction.as_ref().map_or(0, |i| i.len())
            }
            ModelInput::Code { content, .. } => content.len(),
            ModelInput::Image { .. } => 77, // Fixed for CLIP
            ModelInput::Audio { bytes, .. } => bytes.len() / 32000, // Rough estimate
        }
    }

    /// Pad inputs according to configured strategy
    fn pad_inputs(&self, inputs: &[ModelInput], model_id: ModelId) -> EmbeddingResult<Vec<ModelInput>> {
        let max_len = match self.config.padding {
            PaddingStrategy::MaxLength => model_id.max_seq_length(),
            PaddingStrategy::DynamicMax => {
                inputs.iter()
                    .map(|i| self.estimate_sequence_length(i))
                    .max()
                    .unwrap_or(0)
            }
            PaddingStrategy::PowerOfTwo => {
                let max_seq = inputs.iter()
                    .map(|i| self.estimate_sequence_length(i))
                    .max()
                    .unwrap_or(0);
                max_seq.next_power_of_two()
            }
            PaddingStrategy::Bucket { sizes } => {
                let max_seq = inputs.iter()
                    .map(|i| self.estimate_sequence_length(i))
                    .max()
                    .unwrap_or(0);
                *sizes.iter().find(|&&s| s >= max_seq).unwrap_or(&sizes[3])
            }
        };

        // Clone inputs (actual padding done by tokenizer)
        Ok(inputs.to_vec())
    }

    /// Start background batch processing loop
    pub async fn start_background_processor(self: Arc<Self>) {
        let processor = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(10)).await;

                // Check each queue for pending batches
                for &model_id in ModelId::all() {
                    if let Ok(queue) = processor.queues.get(&model_id) {
                        let should_flush = !queue.requests.is_empty()
                            && queue.last_batch_time.elapsed() >= Duration::from_millis(
                                processor.config.max_wait_ms
                            );

                        if should_flush {
                            let _ = processor.process_batch(model_id).await;
                        }
                    }
                }
            }
        });
    }
}
```

### 3.4 FuseMoE Implementation

```rust
// crates/context-graph-embeddings/src/fusion/fusemoe.rs

use ndarray::{Array1, Array2};
use crate::{
    ConcatenatedEmbedding, EmbeddingError, EmbeddingResult, FusedEmbedding,
    FusionConfig, GatingConfig, ModelId,
};

/// FuseMoE (Fused Mixture of Experts) for combining multiple embeddings
///
/// Architecture:
/// ```
/// Input (12,954D) -> Gating Network -> Expert Selection (top-k=2)
///                                   |
///                                   v
///                    +--------+--------+--------+--------+
///                    |Expert 0|Expert 1|  ...   |Expert 7|
///                    +--------+--------+--------+--------+
///                                   |
///                                   v
///                          Weighted Sum -> Output (1536D)
/// ```
pub struct FuseMoE {
    /// Configuration
    config: FusionConfig,
    /// Gating network weights
    gating: GatingNetwork,
    /// Expert networks
    experts: Vec<ExpertNetwork>,
    /// Is initialized
    initialized: bool,
}

/// Gating network for expert selection
struct GatingNetwork {
    /// Input projection: 12954 -> projection_dim
    w_gate: Array2<f32>,
    /// Gate output: projection_dim -> num_experts
    w_out: Array2<f32>,
    /// Layer normalization parameters
    layer_norm: Option<LayerNorm>,
    /// Configuration
    config: GatingConfig,
}

/// Single expert network
struct ExpertNetwork {
    /// Expert index
    index: usize,
    /// Input to hidden: 12954 -> hidden_dim
    w1: Array2<f32>,
    /// Hidden to output: hidden_dim -> output_dim
    w2: Array2<f32>,
    /// Bias for hidden layer
    b1: Array1<f32>,
    /// Bias for output layer
    b2: Array1<f32>,
}

/// Layer normalization parameters
struct LayerNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    eps: f32,
}

impl FuseMoE {
    /// Total input dimension (sum of all model embeddings)
    const INPUT_DIM: usize = 12_954;

    /// Create a new FuseMoE instance
    pub fn new(config: FusionConfig) -> Self {
        Self {
            config,
            gating: GatingNetwork::new(&config.gating, Self::INPUT_DIM, config.num_experts),
            experts: (0..config.num_experts)
                .map(|i| ExpertNetwork::new(
                    i,
                    Self::INPUT_DIM,
                    config.expert_hidden_dim,
                    config.output_dim,
                ))
                .collect(),
            initialized: false,
        }
    }

    /// Initialize with pre-trained weights
    pub fn load_weights(&mut self, weights_path: &std::path::Path) -> EmbeddingResult<()> {
        // Load weights from file (implementation detail)
        self.initialized = true;
        Ok(())
    }

    /// Fuse concatenated embeddings into unified representation
    pub fn fuse(&self, input: &ConcatenatedEmbedding) -> EmbeddingResult<FusedEmbedding> {
        if !self.initialized {
            return Err(EmbeddingError::NotInitialized);
        }

        if input.concatenated.len() != Self::INPUT_DIM {
            return Err(EmbeddingError::InvalidDimension {
                expected: Self::INPUT_DIM,
                actual: input.concatenated.len(),
            });
        }

        let start = std::time::Instant::now();

        // Convert to ndarray for computation
        let x = Array1::from_vec(input.concatenated.clone());

        // 1. Compute gating scores
        let (expert_weights, selected_experts) = self.gating.forward(&x)?;

        // 2. Run selected experts
        let mut output = Array1::zeros(self.config.output_dim);

        for &expert_idx in &selected_experts {
            let expert = &self.experts[expert_idx as usize];
            let expert_weight = expert_weights[expert_idx as usize];

            // Skip experts with negligible weight
            if expert_weight < 1e-6 {
                continue;
            }

            let expert_output = expert.forward(&x)?;
            output = output + expert_output * expert_weight;
        }

        // 3. Normalize output
        let norm = output.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            output.mapv_inplace(|v| v / norm);
        }

        let elapsed = start.elapsed();

        Ok(FusedEmbedding {
            vector: output.to_vec(),
            expert_weights: expert_weights.try_into().unwrap_or([0.0; 8]),
            selected_experts: selected_experts.try_into().unwrap_or([0; 2]),
            pipeline_latency_us: elapsed.as_micros() as u64,
            content_hash: input.embeddings.get(0)
                .map(|e| crate::cache::content_hash::hash_embedding(&e.vector))
                .unwrap_or(0),
        })
    }

    /// Batch fusion for multiple inputs
    pub fn fuse_batch(&self, inputs: &[ConcatenatedEmbedding]) -> EmbeddingResult<Vec<FusedEmbedding>> {
        inputs.iter().map(|input| self.fuse(input)).collect()
    }

    /// Get load balancing loss for training
    pub fn load_balance_loss(&self, batch_expert_weights: &[Array1<f32>]) -> f32 {
        if !self.config.load_balance_loss || batch_expert_weights.is_empty() {
            return 0.0;
        }

        let batch_size = batch_expert_weights.len();
        let num_experts = self.config.num_experts;

        // Compute mean expert utilization
        let mut expert_counts = vec![0.0f32; num_experts];
        for weights in batch_expert_weights {
            for (i, &w) in weights.iter().enumerate() {
                if w > 1e-6 {
                    expert_counts[i] += 1.0;
                }
            }
        }

        // Normalize
        for count in &mut expert_counts {
            *count /= batch_size as f32;
        }

        // Compute load balancing loss (encourage uniform distribution)
        let mean = expert_counts.iter().sum::<f32>() / num_experts as f32;
        let variance: f32 = expert_counts.iter()
            .map(|&c| (c - mean).powi(2))
            .sum::<f32>() / num_experts as f32;

        self.config.load_balance_coef * variance
    }
}

impl GatingNetwork {
    fn new(config: &GatingConfig, input_dim: usize, num_experts: usize) -> Self {
        // Initialize with small random values (in practice, load from checkpoint)
        let w_gate = Array2::zeros((input_dim, config.projection_dim));
        let w_out = Array2::zeros((config.projection_dim, num_experts));

        let layer_norm = if config.layer_norm {
            Some(LayerNorm {
                gamma: Array1::ones(input_dim),
                beta: Array1::zeros(input_dim),
                eps: 1e-5,
            })
        } else {
            None
        };

        Self { w_gate, w_out, layer_norm, config: config.clone() }
    }

    fn forward(&self, x: &Array1<f32>) -> EmbeddingResult<(Array1<f32>, Vec<u8>)> {
        // Apply layer normalization
        let normalized = if let Some(ref ln) = self.layer_norm {
            let mean = x.mean().unwrap_or(0.0);
            let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
            let std = (var + ln.eps).sqrt();
            (&ln.gamma * (x - mean) / std) + &ln.beta
        } else {
            x.clone()
        };

        // Project to gate dimension
        let hidden = normalized.dot(&self.w_gate);

        // Apply GELU activation
        let activated = hidden.mapv(|v| v * 0.5 * (1.0 + (v / std::f32::consts::SQRT_2).tanh()));

        // Compute gate logits
        let logits = activated.dot(&self.w_out);

        // Apply softmax with temperature
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter()
            .map(|&v| ((v - max_logit) / self.config.temperature).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Array1<f32> = Array1::from_vec(
            exp_logits.iter().map(|&v| v / sum_exp).collect()
        );

        // Select top-k experts
        let mut indexed: Vec<_> = probs.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected: Vec<u8> = indexed.iter()
            .take(2) // TOP_K = 2
            .map(|&(i, _)| i as u8)
            .collect();

        Ok((probs, selected))
    }
}

impl ExpertNetwork {
    fn new(index: usize, input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        // Initialize with small random values (in practice, load from checkpoint)
        Self {
            index,
            w1: Array2::zeros((input_dim, hidden_dim)),
            w2: Array2::zeros((hidden_dim, output_dim)),
            b1: Array1::zeros(hidden_dim),
            b2: Array1::zeros(output_dim),
        }
    }

    fn forward(&self, x: &Array1<f32>) -> EmbeddingResult<Array1<f32>> {
        // Hidden layer with SwiGLU activation
        let hidden = x.dot(&self.w1) + &self.b1;
        let gate = hidden.mapv(|v| 1.0 / (1.0 + (-v).exp())); // Sigmoid
        let activated = &hidden * &gate;

        // Output projection
        let output = activated.dot(&self.w2) + &self.b2;

        Ok(output)
    }
}
```

### 3.5 Cache Manager

```rust
// crates/context-graph-embeddings/src/cache/manager.rs

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::RwLock;
use crate::{CacheConfig, EmbeddingError, EmbeddingResult, EvictionPolicy, FusedEmbedding};

/// Cache manager for embedding caching
pub struct CacheManager {
    /// Configuration
    config: CacheConfig,
    /// In-memory cache
    memory_cache: RwLock<MemoryCache>,
    /// Disk cache (if enabled)
    disk_cache: Option<DiskCache>,
    /// Statistics
    stats: RwLock<CacheStats>,
}

struct MemoryCache {
    /// Content hash -> embedding
    entries: HashMap<u64, CacheEntry>,
    /// LRU order (most recent at back)
    lru_order: Vec<u64>,
    /// Current memory usage in bytes
    current_bytes: usize,
}

struct CacheEntry {
    embedding: FusedEmbedding,
    size_bytes: usize,
    created_at: std::time::Instant,
    access_count: u64,
    last_accessed: std::time::Instant,
}

struct DiskCache {
    cache_dir: PathBuf,
    max_size_bytes: usize,
}

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub current_entries: usize,
    pub current_bytes: usize,
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new(config: CacheConfig) -> EmbeddingResult<Self> {
        let disk_cache = if config.persist_to_disk {
            config.disk_cache_dir.as_ref().map(|dir| {
                std::fs::create_dir_all(dir).ok();
                DiskCache {
                    cache_dir: dir.clone(),
                    max_size_bytes: config.max_bytes / 2, // Half on disk
                }
            })
        } else {
            None
        };

        Ok(Self {
            config,
            memory_cache: RwLock::new(MemoryCache {
                entries: HashMap::new(),
                lru_order: Vec::new(),
                current_bytes: 0,
            }),
            disk_cache,
            stats: RwLock::new(CacheStats::default()),
        })
    }

    /// Get embedding from cache
    pub fn get(&self, content_hash: u64) -> Option<FusedEmbedding> {
        if !self.config.enabled {
            return None;
        }

        // Try memory cache first
        {
            let mut cache = self.memory_cache.write();
            if let Some(entry) = cache.entries.get_mut(&content_hash) {
                // Check TTL
                if self.config.ttl_seconds > 0
                    && entry.created_at.elapsed().as_secs() > self.config.ttl_seconds
                {
                    cache.entries.remove(&content_hash);
                    cache.lru_order.retain(|&h| h != content_hash);
                    return None;
                }

                // Update access statistics
                entry.access_count += 1;
                entry.last_accessed = std::time::Instant::now();

                // Update LRU order
                cache.lru_order.retain(|&h| h != content_hash);
                cache.lru_order.push(content_hash);

                self.stats.write().hits += 1;
                return Some(entry.embedding.clone());
            }
        }

        // Try disk cache
        if let Some(ref disk) = self.disk_cache {
            if let Some(embedding) = disk.get(content_hash) {
                // Promote to memory cache
                self.put_internal(content_hash, embedding.clone(), false);
                self.stats.write().hits += 1;
                return Some(embedding);
            }
        }

        self.stats.write().misses += 1;
        None
    }

    /// Put embedding in cache
    pub fn put(&self, content_hash: u64, embedding: FusedEmbedding) {
        if !self.config.enabled {
            return;
        }

        self.put_internal(content_hash, embedding, true);
    }

    fn put_internal(&self, content_hash: u64, embedding: FusedEmbedding, persist: bool) {
        let entry_size = std::mem::size_of::<FusedEmbedding>()
            + embedding.vector.len() * std::mem::size_of::<f32>();

        let mut cache = self.memory_cache.write();

        // Evict if necessary
        while cache.current_bytes + entry_size > self.config.max_bytes
            || cache.entries.len() >= self.config.max_entries
        {
            if let Some(evict_hash) = self.select_for_eviction(&cache) {
                if let Some(evicted) = cache.entries.remove(&evict_hash) {
                    cache.current_bytes -= evicted.size_bytes;
                    cache.lru_order.retain(|&h| h != evict_hash);

                    // Optionally persist to disk before eviction
                    if persist {
                        if let Some(ref disk) = self.disk_cache {
                            disk.put(evict_hash, &evicted.embedding);
                        }
                    }

                    self.stats.write().evictions += 1;
                }
            } else {
                break;
            }
        }

        // Insert new entry
        let entry = CacheEntry {
            embedding,
            size_bytes: entry_size,
            created_at: std::time::Instant::now(),
            access_count: 0,
            last_accessed: std::time::Instant::now(),
        };

        cache.entries.insert(content_hash, entry);
        cache.lru_order.push(content_hash);
        cache.current_bytes += entry_size;

        let mut stats = self.stats.write();
        stats.current_entries = cache.entries.len();
        stats.current_bytes = cache.current_bytes;
    }

    /// Select an entry for eviction based on policy
    fn select_for_eviction(&self, cache: &MemoryCache) -> Option<u64> {
        match self.config.eviction_policy {
            EvictionPolicy::Lru => {
                cache.lru_order.first().copied()
            }
            EvictionPolicy::Lfu => {
                cache.entries.iter()
                    .min_by_key(|(_, e)| e.access_count)
                    .map(|(&h, _)| h)
            }
            EvictionPolicy::TtlLru => {
                // First try expired entries
                let now = std::time::Instant::now();
                for (&hash, entry) in &cache.entries {
                    if self.config.ttl_seconds > 0
                        && entry.created_at.elapsed().as_secs() > self.config.ttl_seconds
                    {
                        return Some(hash);
                    }
                }
                // Fall back to LRU
                cache.lru_order.first().copied()
            }
            EvictionPolicy::Arc => {
                // Simplified ARC: use LRU for now
                cache.lru_order.first().copied()
            }
        }
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        let mut cache = self.memory_cache.write();
        cache.entries.clear();
        cache.lru_order.clear();
        cache.current_bytes = 0;

        if let Some(ref disk) = self.disk_cache {
            disk.clear();
        }

        let mut stats = self.stats.write();
        stats.current_entries = 0;
        stats.current_bytes = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.read();
        let total = stats.hits + stats.misses;
        if total == 0 {
            0.0
        } else {
            stats.hits as f64 / total as f64
        }
    }
}

impl DiskCache {
    fn get(&self, content_hash: u64) -> Option<FusedEmbedding> {
        let path = self.cache_dir.join(format!("{:016x}.bin", content_hash));
        if path.exists() {
            std::fs::read(&path)
                .ok()
                .and_then(|bytes| bincode::deserialize(&bytes).ok())
        } else {
            None
        }
    }

    fn put(&self, content_hash: u64, embedding: &FusedEmbedding) {
        let path = self.cache_dir.join(format!("{:016x}.bin", content_hash));
        if let Ok(bytes) = bincode::serialize(embedding) {
            let _ = std::fs::write(&path, bytes);
        }
    }

    fn clear(&self) {
        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}
```

---

## 4. Main Pipeline

### 4.1 EmbeddingPipeline

```rust
// crates/context-graph-embeddings/src/pipeline.rs

use std::sync::Arc;
use std::time::Instant;
use crate::{
    BatchProcessor, CacheManager, ConcatenatedEmbedding, EmbeddingConfig,
    EmbeddingError, EmbeddingResult, FusedEmbedding, FuseMoE,
    ModelEmbedding, ModelId, ModelInput, ModelRegistry,
};

/// Main embedding pipeline orchestrating all components
pub struct EmbeddingPipeline {
    /// Configuration
    config: EmbeddingConfig,
    /// Model registry
    registry: Arc<ModelRegistry>,
    /// Batch processor
    batch_processor: Arc<BatchProcessor>,
    /// FuseMoE fusion layer
    fusion: Arc<FuseMoE>,
    /// Cache manager
    cache: Arc<CacheManager>,
    /// Pipeline statistics
    stats: parking_lot::RwLock<PipelineStats>,
}

#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_latency_us: u64,
    pub model_latencies_us: std::collections::HashMap<ModelId, u64>,
    pub fusion_latency_us: u64,
    pub errors: u64,
}

impl EmbeddingPipeline {
    /// Create a new embedding pipeline
    pub async fn new(config: EmbeddingConfig) -> EmbeddingResult<Self> {
        // Initialize model registry
        let registry = Arc::new(ModelRegistry::new(config.models.clone()));
        registry.initialize().await?;

        // Initialize batch processor
        let batch_processor = Arc::new(BatchProcessor::new(
            config.batch.clone(),
            Arc::clone(&registry),
        ));

        // Initialize FuseMoE
        let mut fusion = FuseMoE::new(config.fusion.clone());
        let weights_path = config.models.models_dir.join("fusemoe_weights.bin");
        if weights_path.exists() {
            fusion.load_weights(&weights_path)?;
        }

        // Initialize cache
        let cache = Arc::new(CacheManager::new(config.cache.clone())?);

        Ok(Self {
            config,
            registry,
            batch_processor,
            fusion: Arc::new(fusion),
            cache,
            stats: parking_lot::RwLock::new(PipelineStats::default()),
        })
    }

    /// Generate a fused embedding for text input
    pub async fn embed_text(&self, text: &str) -> EmbeddingResult<FusedEmbedding> {
        self.embed(ModelInput::Text {
            content: text.to_string(),
            instruction: None,
        }).await
    }

    /// Generate a fused embedding for any input type
    pub async fn embed(&self, input: ModelInput) -> EmbeddingResult<FusedEmbedding> {
        let start = Instant::now();
        self.stats.write().total_requests += 1;

        // Compute content hash for caching
        let content_hash = self.compute_content_hash(&input);

        // Check cache first
        if let Some(cached) = self.cache.get(content_hash) {
            self.stats.write().cache_hits += 1;
            return Ok(cached);
        }

        self.stats.write().cache_misses += 1;

        // Generate embeddings from all models
        let concatenated = self.generate_all_embeddings(&input).await?;

        // Fuse embeddings
        let mut fused = self.fusion.fuse(&concatenated)?;
        fused.content_hash = content_hash;

        // Update total pipeline latency
        fused.pipeline_latency_us = start.elapsed().as_micros() as u64;

        // Cache the result
        self.cache.put(content_hash, fused.clone());

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.total_latency_us += fused.pipeline_latency_us;
        }

        Ok(fused)
    }

    /// Generate embeddings for a batch of inputs
    pub async fn embed_batch(&self, inputs: Vec<ModelInput>) -> EmbeddingResult<Vec<FusedEmbedding>> {
        let mut results = Vec::with_capacity(inputs.len());
        let mut uncached_inputs = Vec::new();
        let mut uncached_indices = Vec::new();

        // Check cache for each input
        for (i, input) in inputs.iter().enumerate() {
            let hash = self.compute_content_hash(input);
            if let Some(cached) = self.cache.get(hash) {
                results.push((i, cached));
            } else {
                uncached_inputs.push(input.clone());
                uncached_indices.push(i);
            }
        }

        // Process uncached inputs
        if !uncached_inputs.is_empty() {
            // Generate all embeddings in parallel
            let all_concatenated = self.generate_batch_embeddings(&uncached_inputs).await?;

            // Fuse embeddings
            let fused = self.fusion.fuse_batch(&all_concatenated)?;

            // Cache results and collect
            for (fused_embedding, &idx) in fused.into_iter().zip(&uncached_indices) {
                self.cache.put(fused_embedding.content_hash, fused_embedding.clone());
                results.push((idx, fused_embedding));
            }
        }

        // Sort by original index and return
        results.sort_by_key(|(i, _)| *i);
        Ok(results.into_iter().map(|(_, e)| e).collect())
    }

    /// Generate embeddings from all 12 models
    async fn generate_all_embeddings(&self, input: &ModelInput) -> EmbeddingResult<ConcatenatedEmbedding> {
        let start = Instant::now();

        // Determine which models to use based on input type
        let model_ids = self.select_models_for_input(input);

        // Run all models in parallel
        let futures: Vec<_> = model_ids.iter()
            .map(|&model_id| {
                let input = input.clone();
                let processor = Arc::clone(&self.batch_processor);
                async move {
                    processor.submit(input, model_id).await
                }
            })
            .collect();

        let results = futures::future::join_all(futures).await;

        // Collect embeddings
        let mut embeddings: Vec<ModelEmbedding> = Vec::with_capacity(12);
        let mut concatenated: Vec<f32> = Vec::with_capacity(ModelId::total_dimension());

        for (model_id, result) in model_ids.iter().zip(results) {
            match result {
                Ok(embedding) => {
                    concatenated.extend(&embedding.vector);
                    embeddings.push(embedding);
                }
                Err(e) => {
                    // Use zero vector for failed models
                    tracing::warn!("Model {:?} failed: {}", model_id, e);
                    concatenated.extend(std::iter::repeat(0.0).take(model_id.dimension()));
                    embeddings.push(ModelEmbedding {
                        model_id: *model_id,
                        vector: vec![0.0; model_id.dimension()],
                        latency_us: 0,
                        attention_weights: None,
                    });
                }
            }
        }

        Ok(ConcatenatedEmbedding {
            embeddings,
            concatenated,
            total_latency_us: start.elapsed().as_micros() as u64,
        })
    }

    /// Generate batch embeddings from all models
    async fn generate_batch_embeddings(
        &self,
        inputs: &[ModelInput],
    ) -> EmbeddingResult<Vec<ConcatenatedEmbedding>> {
        // For each model, batch all inputs
        let model_ids = ModelId::all();
        let mut all_model_embeddings: Vec<Vec<ModelEmbedding>> = vec![Vec::new(); inputs.len()];

        for &model_id in model_ids {
            // Get model and run batch
            let model = self.registry.get_model(model_id).await?;
            let embeddings = model.embed_batch(inputs).await?;

            for (i, emb) in embeddings.into_iter().enumerate() {
                all_model_embeddings[i].push(emb);
            }
        }

        // Concatenate embeddings for each input
        let concatenated: Vec<_> = all_model_embeddings
            .into_iter()
            .map(|embeddings| {
                let mut concat = Vec::with_capacity(ModelId::total_dimension());
                for emb in &embeddings {
                    concat.extend(&emb.vector);
                }
                ConcatenatedEmbedding {
                    embeddings,
                    concatenated: concat,
                    total_latency_us: 0,
                }
            })
            .collect();

        Ok(concatenated)
    }

    /// Select appropriate models based on input type
    fn select_models_for_input(&self, input: &ModelInput) -> Vec<ModelId> {
        match input {
            ModelInput::Text { .. } => ModelId::all().to_vec(),
            ModelInput::Image { .. } => vec![ModelId::ClipViT],
            ModelInput::Audio { .. } => vec![ModelId::WhisperLarge],
            ModelInput::Code { .. } => vec![
                ModelId::CodeBERT,
                ModelId::MiniLM,
                ModelId::BgeLarge,
                ModelId::GteLarge,
                ModelId::E5Mistral,
            ],
        }
    }

    /// Compute content hash for caching
    fn compute_content_hash(&self, input: &ModelInput) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();

        match input {
            ModelInput::Text { content, instruction } => {
                "text".hash(&mut hasher);
                content.hash(&mut hasher);
                instruction.hash(&mut hasher);
            }
            ModelInput::Image { bytes, format } => {
                "image".hash(&mut hasher);
                bytes.hash(&mut hasher);
                (*format as u8).hash(&mut hasher);
            }
            ModelInput::Audio { bytes, sample_rate, channels } => {
                "audio".hash(&mut hasher);
                bytes.hash(&mut hasher);
                sample_rate.hash(&mut hasher);
                channels.hash(&mut hasher);
            }
            ModelInput::Code { content, language } => {
                "code".hash(&mut hasher);
                content.hash(&mut hasher);
                language.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> PipelineStats {
        self.stats.read().clone()
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> crate::cache::manager::CacheStats {
        self.cache.stats()
    }

    /// Get model registry statistics
    pub fn registry_stats(&self) -> crate::models::registry::RegistryStats {
        self.registry.stats()
    }
}
```

---

## 5. CUDA Kernel Interfaces

### 5.1 CUDA Bindings

```rust
// crates/context-graph-cuda/src/lib.rs

#![cfg(feature = "cuda")]

use std::ffi::c_void;

/// CUDA error type
#[derive(Debug)]
pub struct CudaError {
    pub code: i32,
    pub message: String,
}

pub type CudaResult<T> = Result<T, CudaError>;

/// Initialize CUDA runtime
pub fn cuda_init() -> CudaResult<()>;

/// Get number of available CUDA devices
pub fn cuda_device_count() -> CudaResult<usize>;

/// Set active CUDA device
pub fn cuda_set_device(device_id: usize) -> CudaResult<()>;

/// Allocate GPU memory
pub fn cuda_malloc(size: usize) -> CudaResult<*mut c_void>;

/// Free GPU memory
pub fn cuda_free(ptr: *mut c_void) -> CudaResult<()>;

/// Copy data from host to device
pub fn cuda_memcpy_h2d(dst: *mut c_void, src: *const c_void, size: usize) -> CudaResult<()>;

/// Copy data from device to host
pub fn cuda_memcpy_d2h(dst: *mut c_void, src: *const c_void, size: usize) -> CudaResult<()>;

/// Synchronize CUDA stream
pub fn cuda_synchronize() -> CudaResult<()>;
```

### 5.2 Embedding CUDA Operations

```rust
// crates/context-graph-cuda/src/kernels/embedding.rs

use crate::{CudaResult, cuda_malloc, cuda_free, cuda_memcpy_h2d, cuda_memcpy_d2h};

/// GPU-accelerated embedding operations
pub struct GpuEmbeddingOps {
    device_id: usize,
    workspace: *mut std::ffi::c_void,
    workspace_size: usize,
}

impl GpuEmbeddingOps {
    /// Create new GPU embedding operations
    pub fn new(device_id: usize) -> CudaResult<Self> {
        let workspace_size = 256 * 1024 * 1024; // 256MB workspace
        let workspace = cuda_malloc(workspace_size)?;

        Ok(Self {
            device_id,
            workspace,
            workspace_size,
        })
    }

    /// Batch embedding normalization on GPU
    pub fn normalize_batch(
        &self,
        embeddings: &mut [f32],
        batch_size: usize,
        embedding_dim: usize,
    ) -> CudaResult<()> {
        extern "C" {
            fn cuda_normalize_embeddings(
                data: *mut f32,
                batch_size: usize,
                dim: usize,
            ) -> i32;
        }

        // Copy to device
        let data_size = embeddings.len() * std::mem::size_of::<f32>();
        let device_data = cuda_malloc(data_size)?;
        cuda_memcpy_h2d(device_data, embeddings.as_ptr() as *const _, data_size)?;

        // Run kernel
        unsafe {
            let result = cuda_normalize_embeddings(
                device_data as *mut f32,
                batch_size,
                embedding_dim,
            );
            if result != 0 {
                cuda_free(device_data)?;
                return Err(crate::CudaError {
                    code: result,
                    message: "Normalization kernel failed".to_string(),
                });
            }
        }

        // Copy back to host
        cuda_memcpy_d2h(embeddings.as_mut_ptr() as *mut _, device_data, data_size)?;
        cuda_free(device_data)?;

        Ok(())
    }

    /// Batch cosine similarity computation on GPU
    pub fn cosine_similarity_batch(
        &self,
        query: &[f32],
        candidates: &[f32],
        num_candidates: usize,
        dim: usize,
    ) -> CudaResult<Vec<f32>> {
        extern "C" {
            fn cuda_cosine_similarity(
                query: *const f32,
                candidates: *const f32,
                results: *mut f32,
                num_candidates: usize,
                dim: usize,
            ) -> i32;
        }

        let query_size = query.len() * std::mem::size_of::<f32>();
        let candidates_size = candidates.len() * std::mem::size_of::<f32>();
        let results_size = num_candidates * std::mem::size_of::<f32>();

        // Allocate device memory
        let d_query = cuda_malloc(query_size)?;
        let d_candidates = cuda_malloc(candidates_size)?;
        let d_results = cuda_malloc(results_size)?;

        // Copy inputs to device
        cuda_memcpy_h2d(d_query, query.as_ptr() as *const _, query_size)?;
        cuda_memcpy_h2d(d_candidates, candidates.as_ptr() as *const _, candidates_size)?;

        // Run kernel
        unsafe {
            let result = cuda_cosine_similarity(
                d_query as *const f32,
                d_candidates as *const f32,
                d_results as *mut f32,
                num_candidates,
                dim,
            );
            if result != 0 {
                cuda_free(d_query)?;
                cuda_free(d_candidates)?;
                cuda_free(d_results)?;
                return Err(crate::CudaError {
                    code: result,
                    message: "Cosine similarity kernel failed".to_string(),
                });
            }
        }

        // Copy results back
        let mut results = vec![0.0f32; num_candidates];
        cuda_memcpy_d2h(results.as_mut_ptr() as *mut _, d_results, results_size)?;

        // Cleanup
        cuda_free(d_query)?;
        cuda_free(d_candidates)?;
        cuda_free(d_results)?;

        Ok(results)
    }
}

impl Drop for GpuEmbeddingOps {
    fn drop(&mut self) {
        let _ = cuda_free(self.workspace);
    }
}
```

### 5.3 FuseMoE CUDA Operations

```rust
// crates/context-graph-cuda/src/kernels/fusion.rs

use crate::CudaResult;

/// GPU-accelerated FuseMoE operations
pub struct GpuFuseMoEOps {
    device_id: usize,
    /// Pre-allocated expert buffers
    expert_buffers: Vec<*mut std::ffi::c_void>,
    /// Gating network weights on GPU
    gating_weights: *mut std::ffi::c_void,
}

impl GpuFuseMoEOps {
    /// Create new GPU FuseMoE operations
    pub fn new(
        device_id: usize,
        num_experts: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> CudaResult<Self>;

    /// Load gating network weights to GPU
    pub fn load_gating_weights(&mut self, weights: &[f32]) -> CudaResult<()>;

    /// Load expert weights to GPU
    pub fn load_expert_weights(&mut self, expert_idx: usize, weights: &[f32]) -> CudaResult<()>;

    /// Run FuseMoE forward pass on GPU
    pub fn forward(
        &self,
        input: &[f32],          // [batch_size, input_dim]
        batch_size: usize,
    ) -> CudaResult<FuseMoEOutput>;

    /// Batch forward pass with fused attention
    pub fn forward_batch_fused(
        &self,
        inputs: &[f32],         // [batch_size, input_dim]
        batch_size: usize,
    ) -> CudaResult<Vec<FuseMoEOutput>>;
}

pub struct FuseMoEOutput {
    /// Output embeddings [batch_size, output_dim]
    pub embeddings: Vec<f32>,
    /// Expert weights [batch_size, num_experts]
    pub expert_weights: Vec<f32>,
    /// Selected experts [batch_size, top_k]
    pub selected_experts: Vec<u8>,
}
```

---

## 6. Performance Optimizations

### 6.1 Memory Pool

```rust
// crates/context-graph-embeddings/src/memory_pool.rs

use std::collections::VecDeque;
use parking_lot::Mutex;

/// Thread-safe memory pool for embedding vectors
pub struct EmbeddingPool {
    /// Pools by dimension
    pools: Vec<Mutex<VecDeque<Vec<f32>>>>,
    /// Supported dimensions
    dimensions: Vec<usize>,
    /// Max buffers per dimension
    max_buffers: usize,
}

impl EmbeddingPool {
    pub fn new(dimensions: Vec<usize>, max_buffers: usize) -> Self {
        let pools = dimensions.iter()
            .map(|_| Mutex::new(VecDeque::with_capacity(max_buffers)))
            .collect();

        Self { pools, dimensions, max_buffers }
    }

    /// Acquire a buffer of the specified dimension
    pub fn acquire(&self, dim: usize) -> Vec<f32> {
        if let Some(pool_idx) = self.dimensions.iter().position(|&d| d == dim) {
            let mut pool = self.pools[pool_idx].lock();
            if let Some(buf) = pool.pop_front() {
                return buf;
            }
        }

        // Allocate new buffer
        vec![0.0; dim]
    }

    /// Return a buffer to the pool
    pub fn release(&self, mut buf: Vec<f32>) {
        let dim = buf.len();
        if let Some(pool_idx) = self.dimensions.iter().position(|&d| d == dim) {
            let mut pool = self.pools[pool_idx].lock();
            if pool.len() < self.max_buffers {
                // Clear and return to pool
                buf.fill(0.0);
                pool.push_back(buf);
            }
        }
        // Otherwise, let buffer be dropped
    }
}
```

### 6.2 SIMD Operations

```rust
// crates/context-graph-embeddings/src/simd.rs

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-accelerated vector operations
pub struct SimdOps;

impl SimdOps {
    /// Compute dot product using AVX-512 if available
    #[cfg(target_arch = "x86_64")]
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());

        if is_x86_feature_detected!("avx512f") {
            unsafe { Self::dot_product_avx512(a, b) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { Self::dot_product_avx2(a, b) }
        } else {
            Self::dot_product_scalar(a, b)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = _mm512_setzero_ps();
        let chunks = a.len() / 16;

        for i in 0..chunks {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            sum = _mm512_fmadd_ps(a_vec, b_vec, sum);
        }

        let mut result = _mm512_reduce_add_ps(sum);

        // Handle remainder
        for i in (chunks * 16)..a.len() {
            result += a[i] * b[i];
        }

        result
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;

        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        }

        // Horizontal sum
        let low = _mm256_extractf128_ps(sum, 0);
        let high = _mm256_extractf128_ps(sum, 1);
        let sum128 = _mm_add_ps(low, high);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remainder
        for i in (chunks * 8)..a.len() {
            result += a[i] * b[i];
        }

        result
    }

    fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Normalize vector to unit length using SIMD
    pub fn normalize(v: &mut [f32]) {
        let norm = Self::dot_product(v, v).sqrt();
        if norm > 1e-8 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }
}
```

---

## 7. Error Handling Strategy

### 7.1 Graceful Degradation

```rust
// crates/context-graph-embeddings/src/fallback.rs

use crate::{EmbeddingResult, FusedEmbedding, ModelId, EmbeddingError};

/// Fallback strategy for handling model failures
pub enum FallbackStrategy {
    /// Use zero vector for failed models
    ZeroVector,
    /// Use cached embedding if available
    UseCached,
    /// Retry with exponential backoff
    RetryWithBackoff { max_retries: u32, base_delay_ms: u64 },
    /// Use alternative model
    AlternativeModel { alternatives: Vec<ModelId> },
    /// Fail the entire request
    FailFast,
}

/// Handle model failure with configured strategy
pub async fn handle_model_failure(
    model_id: ModelId,
    error: EmbeddingError,
    strategy: &FallbackStrategy,
) -> EmbeddingResult<Vec<f32>> {
    tracing::warn!("Model {:?} failed: {}, applying fallback", model_id, error);

    match strategy {
        FallbackStrategy::ZeroVector => {
            Ok(vec![0.0; model_id.dimension()])
        }
        FallbackStrategy::UseCached => {
            // Implementation would check cache
            Err(error)
        }
        FallbackStrategy::RetryWithBackoff { max_retries, base_delay_ms } => {
            // Implementation would retry with backoff
            Err(error)
        }
        FallbackStrategy::AlternativeModel { alternatives } => {
            // Implementation would try alternative models
            Err(error)
        }
        FallbackStrategy::FailFast => {
            Err(error)
        }
    }
}
```

### 7.2 Circuit Breaker

```rust
// crates/context-graph-embeddings/src/circuit_breaker.rs

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use parking_lot::RwLock;

/// Circuit breaker for model health management
pub struct CircuitBreaker {
    /// Model identifier
    model_id: crate::ModelId,
    /// Current state
    state: RwLock<CircuitState>,
    /// Failure count
    failure_count: AtomicU64,
    /// Success count since last failure
    success_count: AtomicU64,
    /// Configuration
    config: CircuitBreakerConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,
    Open { until: Instant },
    HalfOpen,
}

pub struct CircuitBreakerConfig {
    /// Failures before opening circuit
    pub failure_threshold: u64,
    /// Successes in half-open to close
    pub success_threshold: u64,
    /// Time before attempting recovery
    pub recovery_timeout: Duration,
}

impl CircuitBreaker {
    pub fn new(model_id: crate::ModelId, config: CircuitBreakerConfig) -> Self {
        Self {
            model_id,
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            config,
        }
    }

    /// Check if request should be allowed
    pub fn should_allow(&self) -> bool {
        let state = *self.state.read();

        match state {
            CircuitState::Closed => true,
            CircuitState::Open { until } => {
                if Instant::now() >= until {
                    *self.state.write() = CircuitState::HalfOpen;
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful operation
    pub fn record_success(&self) {
        self.failure_count.store(0, Ordering::SeqCst);

        let state = *self.state.read();
        if state == CircuitState::HalfOpen {
            let count = self.success_count.fetch_add(1, Ordering::SeqCst) + 1;
            if count >= self.config.success_threshold {
                *self.state.write() = CircuitState::Closed;
                self.success_count.store(0, Ordering::SeqCst);
                tracing::info!("Circuit closed for {:?}", self.model_id);
            }
        }
    }

    /// Record a failed operation
    pub fn record_failure(&self) {
        self.success_count.store(0, Ordering::SeqCst);

        let count = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
        if count >= self.config.failure_threshold {
            let until = Instant::now() + self.config.recovery_timeout;
            *self.state.write() = CircuitState::Open { until };
            tracing::warn!("Circuit opened for {:?}", self.model_id);
        }
    }
}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```rust
// crates/context-graph-embeddings/src/tests/mod.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_dimensions() {
        assert_eq!(ModelId::MiniLM.dimension(), 384);
        assert_eq!(ModelId::BgeLarge.dimension(), 1024);
        assert_eq!(ModelId::E5Mistral.dimension(), 4096);
        assert_eq!(ModelId::total_dimension(), 12_954);
    }

    #[test]
    fn test_fused_embedding_validation() {
        let mut embedding = FusedEmbedding {
            vector: vec![0.1; 1536],
            expert_weights: [0.125; 8],
            selected_experts: [0, 1],
            pipeline_latency_us: 1000,
            content_hash: 12345,
        };

        assert!(embedding.validate().is_ok());

        embedding.vector[0] = f32::NAN;
        assert!(embedding.validate().is_err());
    }

    #[test]
    fn test_content_hash_consistency() {
        let input1 = ModelInput::Text {
            content: "hello world".to_string(),
            instruction: None,
        };
        let input2 = ModelInput::Text {
            content: "hello world".to_string(),
            instruction: None,
        };

        // Same input should produce same hash
        let hash1 = compute_content_hash(&input1);
        let hash2 = compute_content_hash(&input2);
        assert_eq!(hash1, hash2);
    }

    #[tokio::test]
    async fn test_cache_hit_miss() {
        let config = CacheConfig::default();
        let cache = CacheManager::new(config).unwrap();

        let embedding = FusedEmbedding {
            vector: vec![0.1; 1536],
            expert_weights: [0.125; 8],
            selected_experts: [0, 1],
            pipeline_latency_us: 1000,
            content_hash: 12345,
        };

        // Should miss initially
        assert!(cache.get(12345).is_none());

        // Put and get should hit
        cache.put(12345, embedding.clone());
        assert!(cache.get(12345).is_some());

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }
}
```

### 8.2 Integration Tests

```rust
// tests/integration/embedding_tests.rs

use context_graph_embeddings::{EmbeddingPipeline, EmbeddingConfig, ModelInput};

#[tokio::test]
async fn test_full_pipeline() {
    let config = EmbeddingConfig::default();
    let pipeline = EmbeddingPipeline::new(config).await.unwrap();

    let text = "This is a test sentence for embedding.";
    let embedding = pipeline.embed_text(text).await.unwrap();

    assert_eq!(embedding.vector.len(), 1536);
    assert!(embedding.validate().is_ok());
}

#[tokio::test]
async fn test_batch_embedding() {
    let config = EmbeddingConfig::default();
    let pipeline = EmbeddingPipeline::new(config).await.unwrap();

    let inputs: Vec<ModelInput> = (0..32)
        .map(|i| ModelInput::Text {
            content: format!("Test sentence number {}", i),
            instruction: None,
        })
        .collect();

    let embeddings = pipeline.embed_batch(inputs).await.unwrap();

    assert_eq!(embeddings.len(), 32);
    for emb in &embeddings {
        assert_eq!(emb.vector.len(), 1536);
    }
}

#[tokio::test]
async fn test_latency_targets() {
    let config = EmbeddingConfig::default();
    let pipeline = EmbeddingPipeline::new(config).await.unwrap();

    // Single embedding should be < 50ms
    let start = std::time::Instant::now();
    let _ = pipeline.embed_text("Test sentence").await.unwrap();
    let single_latency = start.elapsed();
    assert!(single_latency.as_millis() < 50, "Single latency: {:?}", single_latency);

    // Batch of 32 should be < 200ms
    let inputs: Vec<ModelInput> = (0..32)
        .map(|i| ModelInput::Text {
            content: format!("Test {}", i),
            instruction: None,
        })
        .collect();

    let start = std::time::Instant::now();
    let _ = pipeline.embed_batch(inputs).await.unwrap();
    let batch_latency = start.elapsed();
    assert!(batch_latency.as_millis() < 200, "Batch latency: {:?}", batch_latency);
}
```

---

## 9. Deployment Configuration

### 9.1 Docker Configuration

```yaml
# config/docker/embedding-service.yaml
version: '3.8'

services:
  embedding-service:
    build:
      context: .
      dockerfile: Dockerfile.embeddings
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - RUST_LOG=info
      - MODEL_CACHE_DIR=/models
      - EMBEDDING_CACHE_DIR=/cache
    volumes:
      - model-weights:/models:ro
      - embedding-cache:/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 16G

volumes:
  model-weights:
  embedding-cache:
```

### 9.2 Kubernetes Configuration

```yaml
# config/k8s/embedding-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-pipeline
spec:
  replicas: 2
  selector:
    matchLabels:
      app: embedding-pipeline
  template:
    metadata:
      labels:
        app: embedding-pipeline
    spec:
      containers:
      - name: embedding
        image: contextgraph/embedding:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            memory: "8Gi"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: MODEL_CACHE_DIR
          value: "/models"
        volumeMounts:
        - name: model-weights
          mountPath: /models
          readOnly: true
        - name: embedding-cache
          mountPath: /cache
      volumes:
      - name: model-weights
        persistentVolumeClaim:
          claimName: model-weights-pvc
      - name: embedding-cache
        emptyDir:
          medium: Memory
          sizeLimit: 4Gi
```

---

## 10. Performance Benchmarks

### 10.1 Target Metrics

| Metric | Target | Condition |
|--------|--------|-----------|
| Single embedding latency | < 50ms | Cold cache, GPU |
| Batch embedding (32) | < 200ms | Cold cache, GPU |
| Cache hit latency | < 1ms | Memory cache |
| Cache miss + compute | < 50ms | GPU |
| GPU memory usage | < 8GB | All models loaded |
| Throughput | > 500 emb/s | Sustained, batched |
| Model swap time | < 500ms | GPU to GPU |

### 10.2 Benchmark Suite

```rust
// benches/embedding_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use context_graph_embeddings::{EmbeddingPipeline, EmbeddingConfig, ModelInput};
use tokio::runtime::Runtime;

fn benchmark_single_embedding(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let pipeline = rt.block_on(async {
        EmbeddingPipeline::new(EmbeddingConfig::default()).await.unwrap()
    });

    c.bench_function("single_embedding", |b| {
        b.to_async(&rt).iter(|| async {
            pipeline.embed_text("Test sentence for benchmarking").await.unwrap()
        });
    });
}

fn benchmark_batch_embedding(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let pipeline = rt.block_on(async {
        EmbeddingPipeline::new(EmbeddingConfig::default()).await.unwrap()
    });

    let mut group = c.benchmark_group("batch_embedding");

    for batch_size in [8, 16, 32, 64].iter() {
        let inputs: Vec<ModelInput> = (0..*batch_size)
            .map(|i| ModelInput::Text {
                content: format!("Benchmark sentence number {}", i),
                instruction: None,
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &inputs,
            |b, inputs| {
                b.to_async(&rt).iter(|| async {
                    pipeline.embed_batch(inputs.clone()).await.unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_single_embedding, benchmark_batch_embedding);
criterion_main!(benches);
```

---

## 11. Appendix

### 11.1 Model Weight Sources

| Model ID | Source | Size | License |
|----------|--------|------|---------|
| E1 | sentence-transformers/all-MiniLM-L6-v2 | 90MB | Apache 2.0 |
| E2 | BAAI/bge-large-en-v1.5 | 1.3GB | MIT |
| E3 | hkunlp/instructor-xl | 5GB | Apache 2.0 |
| E4 | openai/clip-vit-large-patch14 | 1.5GB | MIT |
| E5 | openai/whisper-large-v3 | 3GB | MIT |
| E6 | microsoft/codebert-base | 500MB | MIT |
| E7 | sentence-transformers/sentence-t5-xxl | 5GB | Apache 2.0 |
| E8 | sentence-transformers/all-mpnet-base-v2 | 420MB | Apache 2.0 |
| E9 | facebook/contriever-msmarco | 440MB | CC-BY-NC |
| E10 | facebook/dragon-plus-context-encoder | 440MB | CC-BY-NC |
| E11 | thenlper/gte-large | 1.4GB | MIT |
| E12 | intfloat/e5-mistral-7b-instruct | 14GB | MIT |

### 11.2 Dimension Summary

```
Total Input Dimension: 12,954
├── E1  (MiniLM):      384
├── E2  (BGE):       1,024
├── E3  (Instructor):  768
├── E4  (CLIP):        768
├── E5  (Whisper):   1,280
├── E6  (CodeBERT):    768
├── E7  (SentenceT5):  768
├── E8  (MPNet):       768
├── E9  (Contriever):  768
├── E10 (DRAGON+):     768
├── E11 (GTE):       1,024
└── E12 (E5-Mistral):4,096

Output Dimension: 1,536 (FuseMoE unified)
```

---

*End of Technical Specification - Module 3: Embedding Pipeline*
