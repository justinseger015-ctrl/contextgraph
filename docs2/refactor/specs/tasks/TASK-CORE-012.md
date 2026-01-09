# TASK-CORE-012: Embedding Model Loading Infrastructure

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-CORE-012 |
| **Title** | Embedding Model Loading Infrastructure |
| **Status** | :white_circle: todo |
| **Layer** | Foundation |
| **Sequence** | 12 |
| **Estimated Days** | 4 |
| **Complexity** | High |

## Implements

- **REQ-EMBEDDINGS-01**: All 13 embedding models must be loadable
- **Performance Budget**: Cold start < 5s, warm embedding < 500ms

## Dependencies

| Task | Reason |
|------|--------|
| TASK-CORE-011 | Requires GPU memory pool for model allocation |
| TASK-CORE-002 | Uses Embedder enum definitions |

## Objective

Create model loading infrastructure for all 13 embedding models using Candle framework with ONNX support and warm-up capabilities.

## Context

The system requires 13 different embedding models to generate TeleologicalArrays. Each model must be:
- Loadable from disk (ONNX format)
- Allocated on GPU via GpuMemoryPool
- Warmed up before first real inference
- Managed for memory efficiency (lazy loading optional)

Constitution specifies:
- Cold start embedding latency < 5s (p95)
- Full array embedding latency < 500ms (p95)

## Scope

### In Scope

- `ModelLoader` trait for loading embedding models
- `EmbedderModelRegistry` for managing 13 models
- Model warm-up strategy (single dummy inference)
- Lazy loading support for memory optimization
- Model file path configuration
- Version validation and compatibility checks
- Model unloading for memory reclamation

### Out of Scope

- Actual embedding model implementations (separate per-model tasks)
- Quantization (see TASK-CORE-013)
- Batch embedding orchestration

## Definition of Done

### Signatures

```rust
// crates/context-graph-embeddings/src/loader.rs

use context_graph_core::teleology::embedder::Embedder;
use context_graph_cuda::GpuMemoryPool;

/// Trait for loading embedding models
#[async_trait]
pub trait ModelLoader: Send + Sync {
    /// Load model for specific embedder
    async fn load(&self, embedder: Embedder) -> EmbeddingResult<Box<dyn EmbeddingModel>>;

    /// Warm up loaded model with dummy inference
    async fn warm_up(&self, model: &dyn EmbeddingModel) -> EmbeddingResult<()>;

    /// Unload model and reclaim memory
    async fn unload(&self, embedder: Embedder) -> EmbeddingResult<()>;

    /// Check if model file exists and is valid
    fn validate_model_file(&self, embedder: Embedder) -> EmbeddingResult<ModelMetadata>;
}

/// Registry managing all 13 embedding models
pub struct EmbedderModelRegistry {
    models: [Option<Box<dyn EmbeddingModel>>; 13],
    loader: Box<dyn ModelLoader>,
    gpu_pool: Arc<GpuMemoryPool>,
    config: ModelConfig,
}

impl EmbedderModelRegistry {
    /// Create registry with loader and GPU pool
    pub fn new(
        loader: Box<dyn ModelLoader>,
        gpu_pool: Arc<GpuMemoryPool>,
        config: ModelConfig,
    ) -> Self;

    /// Load all 13 models
    pub async fn load_all(&mut self) -> EmbeddingResult<()>;

    /// Load specific embedder's model
    pub async fn load(&mut self, embedder: Embedder) -> EmbeddingResult<()>;

    /// Get loaded model reference
    pub fn get(&self, embedder: Embedder) -> Option<&dyn EmbeddingModel>;

    /// Ensure model is loaded (lazy load if not)
    pub async fn ensure_loaded(&mut self, embedder: Embedder) -> EmbeddingResult<&dyn EmbeddingModel>;

    /// Unload model to free memory
    pub async fn unload(&mut self, embedder: Embedder) -> EmbeddingResult<()>;

    /// Get total memory usage across all models
    pub fn total_memory_usage(&self) -> u64;

    /// Check which models are loaded
    pub fn loaded_models(&self) -> Vec<Embedder>;
}

/// Trait for individual embedding models
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Which embedder this model serves
    fn embedder(&self) -> Embedder;

    /// Embed single text
    async fn embed(&self, text: &str) -> EmbeddingResult<EmbedderOutput>;

    /// Embed batch of texts
    async fn embed_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<EmbedderOutput>>;

    /// Get model's GPU memory usage
    fn memory_usage(&self) -> u64;

    /// Get model version/metadata
    fn metadata(&self) -> &ModelMetadata;
}

/// Configuration for model loading
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Base directory for model files
    pub model_dir: PathBuf,
    /// Whether to use lazy loading
    pub lazy_load: bool,
    /// Maximum batch size for inference
    pub max_batch_size: usize,
    /// Model file patterns per embedder
    pub model_paths: HashMap<Embedder, PathBuf>,
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub embedder: Embedder,
    pub version: String,
    pub dimensions: usize,
    pub file_size: u64,
    pub checksum: String,
}

// crates/context-graph-embeddings/src/candle_loader.rs

/// Candle-based model loader implementation
pub struct CandleModelLoader {
    model_dir: PathBuf,
    gpu_pool: Arc<GpuMemoryPool>,
}

impl CandleModelLoader {
    pub fn new(model_dir: PathBuf, gpu_pool: Arc<GpuMemoryPool>) -> Self;
}

#[async_trait]
impl ModelLoader for CandleModelLoader {
    // ... implementation
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Cold start (all 13 models) | < 5s |
| Per-model load time | < 500ms |
| Warm-up latency | < 100ms per model |
| Memory validation | Reject if > 8GB total |

## Verification

- [ ] All 13 models can be loaded successfully
- [ ] Cold start completes within 5s
- [ ] Warm-up runs before first real inference
- [ ] Lazy loading defers load until first use
- [ ] Model unloading reclaims GPU memory
- [ ] Invalid model files produce clear errors
- [ ] Memory limit enforced (fails if > 8GB)

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-embeddings/src/loader.rs` | ModelLoader trait and registry |
| `crates/context-graph-embeddings/src/candle_loader.rs` | Candle implementation |
| `crates/context-graph-embeddings/src/config.rs` | Configuration types |
| `crates/context-graph-embeddings/src/metadata.rs` | Model metadata |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model file corruption | Low | High | Checksum validation |
| OOM during load | Medium | High | Memory limit check before load |
| Version incompatibility | Low | Medium | Version validation |

## Traceability

- Source: Constitution embedder_specification (lines 779-844)
- Performance Budget: Cold start < 5s (line 530)
