# Module 3: Embedding Pipeline - Atomic Tasks

```yaml
metadata:
  module_id: "module-03"
  module_name: "Embedding Pipeline"
  version: "1.0.0"
  phase: 2
  total_tasks: 24
  approach: "inside-out-bottom-up"
  created: "2025-12-31"
  dependencies:
    - module-01-ghost-system
    - module-02-core-infrastructure
  estimated_duration: "4 weeks"
```

---

## Task Overview

This module implements the 12-model embedding pipeline with FuseMoE fusion for producing unified 1536D semantic representations. Tasks are organized in inside-out, bottom-up order:

1. **Foundation Layer** (Tasks 1-8): Core types, ModelId enum, EmbeddingModel trait, configuration
2. **Logic Layer** (Tasks 9-17): Model implementations, BatchProcessor, CacheManager, FuseMoE
3. **Surface Layer** (Tasks 18-24): EmbeddingPipeline, CUDA integration, MCP handlers

---

## Foundation Layer: Core Types and Interfaces

```yaml
tasks:
  # ============================================================
  # FOUNDATION: Core Type Definitions
  # ============================================================

  - id: "M03-T01"
    title: "Define ModelId Enum and Dimension Constants"
    description: |
      Implement the ModelId enum for the 12-model ensemble (E1-E12):
      MiniLM (384D), BgeLarge (1024D), InstructorXL (768D), ClipViT (768D),
      WhisperLarge (1280D), CodeBERT (768D), SentenceT5 (768D), MPNet (768D),
      Contriever (768D), DragonPlus (768D), GteLarge (1024D), E5Mistral (4096D).
      Include methods: dimension(), total_dimension() (12954D), all(), name(), max_seq_length().
      Use #[repr(u8)] for compact storage.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-embeddings/src/lib.rs"
    dependencies: []
    acceptance_criteria:
      - "ModelId enum compiles with 12 variants (E1-E12)"
      - "dimension() returns correct output size for each model"
      - "total_dimension() returns 12954"
      - "name() returns Hugging Face model identifiers"
      - "max_seq_length() returns correct limits (e.g., CLIP=77, GTE=8192)"
      - "Serde serialization/deserialization works"
      - "Copy, Clone, Hash, PartialEq, Eq traits implemented"
    test_file: "crates/context-graph-embeddings/tests/model_id_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 2.1"
      - "REQ-EMBED-001 to REQ-EMBED-012"

  - id: "M03-T02"
    title: "Define ModelEmbedding and ConcatenatedEmbedding Structs"
    description: |
      Implement ModelEmbedding struct with fields: model_id (ModelId), vector (Vec<f32>),
      latency_us (u64), attention_weights (Option<Vec<f32>>).
      Implement ConcatenatedEmbedding struct with fields: embeddings (Vec<ModelEmbedding>),
      concatenated (Vec<f32>), total_latency_us (u64).
      Include is_complete() method verifying all 12 models present with total dim 12954.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-embeddings/src/lib.rs"
    dependencies:
      - "M03-T01"
    acceptance_criteria:
      - "ModelEmbedding struct compiles with 4 fields"
      - "ConcatenatedEmbedding struct compiles with 3 fields"
      - "is_complete() validates 12 embeddings and 12954D concatenated vector"
      - "Clone, Debug traits implemented"
      - "Efficient memory layout verified"
    test_file: "crates/context-graph-embeddings/tests/embedding_types_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 2.1"

  - id: "M03-T03"
    title: "Define FusedEmbedding Struct"
    description: |
      Implement FusedEmbedding struct with fields: vector (Vec<f32> 1536D),
      expert_weights ([f32; 8]), selected_experts ([u8; 2]),
      pipeline_latency_us (u64), content_hash (u64).
      Include constants: DIMENSION=1536, NUM_EXPERTS=8, TOP_K=2.
      Include validate() method checking dimension and NaN/Inf values.
      Include normalize() method for unit vector normalization.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-embeddings/src/lib.rs"
    dependencies:
      - "M03-T01"
    acceptance_criteria:
      - "FusedEmbedding struct compiles with 5 fields"
      - "DIMENSION constant is 1536"
      - "NUM_EXPERTS constant is 8"
      - "TOP_K constant is 2"
      - "validate() rejects wrong dimension and non-finite values"
      - "normalize() produces unit-length vectors (L2 norm = 1.0)"
      - "Serde Serialize/Deserialize implemented"
    test_file: "crates/context-graph-embeddings/tests/fused_embedding_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 2.1"
      - "REQ-EMBED-013"

  - id: "M03-T04"
    title: "Define EmbeddingError Enum"
    description: |
      Implement EmbeddingError enum with variants: ModelNotFound, ModelLoadError,
      InvalidDimension, InvalidValue, InputTooLong, EmptyInput, BatchError,
      FusionError, CacheError, GpuError, TokenizationError, IoError,
      SerializationError, Timeout, NotInitialized, UnsupportedModality.
      Use thiserror for derivation. Include contextual information in all variants.
      Define EmbeddingResult<T> type alias.
    layer: "foundation"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-embeddings/src/error.rs"
    dependencies: []
    acceptance_criteria:
      - "EmbeddingError enum compiles with 16 variants"
      - "All variants include meaningful context (model_id, expected/actual values)"
      - "thiserror derive works correctly"
      - "From<std::io::Error> and From<bincode::Error> implemented"
      - "Display trait produces human-readable messages"
      - "EmbeddingResult<T> alias defined"
    test_file: "crates/context-graph-embeddings/tests/error_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 2.3"

  - id: "M03-T05"
    title: "Define ModelInput Enum and InputType"
    description: |
      Implement ModelInput enum with variants:
      - Text { content: String, instruction: Option<String> }
      - Image { bytes: Vec<u8>, format: ImageFormat }
      - Audio { bytes: Vec<u8>, sample_rate: u32, channels: u8 }
      - Code { content: String, language: String }
      Implement ImageFormat enum: Png, Jpeg, WebP.
      Implement InputType enum: Text, Image, Audio, Code.
    layer: "foundation"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-embeddings/src/models/base.rs"
    dependencies: []
    acceptance_criteria:
      - "ModelInput enum compiles with 4 variants"
      - "ImageFormat enum compiles with 3 variants"
      - "InputType enum compiles with 4 variants"
      - "Clone, Debug traits implemented for all types"
      - "Text variant supports optional instruction prefix"
    test_file: "crates/context-graph-embeddings/tests/input_types_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 3.1"

  - id: "M03-T06"
    title: "Define EmbeddingModel Trait"
    description: |
      Implement the EmbeddingModel async trait with methods:
      - model_id() -> ModelId
      - dimension() -> usize
      - max_sequence_length() -> usize
      - is_loaded() -> bool
      - load() -> EmbeddingResult<()>
      - unload() -> EmbeddingResult<()>
      - embed(input: &ModelInput) -> EmbeddingResult<ModelEmbedding>
      - embed_batch(inputs: &[ModelInput]) -> EmbeddingResult<Vec<ModelEmbedding>>
      - memory_usage() -> usize
      - supported_inputs() -> &[InputType]
      Use async_trait crate. Trait must be Send + Sync.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-embeddings/src/models/base.rs"
    dependencies:
      - "M03-T01"
      - "M03-T04"
      - "M03-T05"
    acceptance_criteria:
      - "EmbeddingModel trait compiles with 10 methods"
      - "Trait is async-safe (async_trait)"
      - "Trait is object-safe for dyn dispatch"
      - "Send + Sync bounds enforced"
      - "Doc comments document all methods with constraints"
    test_file: "crates/context-graph-embeddings/tests/embedding_trait_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 3.1"
      - "REQ-EMBED-029"

  - id: "M03-T07"
    title: "Define Configuration Types"
    description: |
      Implement configuration structs:
      - EmbeddingConfig (models, batch, fusion, cache, gpu)
      - ModelRegistryConfig (models_dir, lazy_loading, preload_models, auto_download)
      - SingleModelConfig (model_id, device, quantization, max_seq_length, flash_attention)
      - BatchConfig (max_batch_size=32, min_batch_size=1, max_wait_ms=50, dynamic_batching)
      - FusionConfig (num_experts=8, top_k=2, output_dim=1536, expert_hidden_dim=4096)
      - CacheConfig (enabled, max_entries=100000, max_bytes=1GB, ttl_seconds, eviction_policy)
      - GpuConfig (enabled, device_ids, memory_fraction=0.9, cuda_graphs, mixed_precision)
      Implement DevicePlacement, QuantizationMode, PaddingStrategy, EvictionPolicy enums.
      All types must implement Default with sensible values.
    layer: "foundation"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-embeddings/src/config.rs"
    dependencies:
      - "M03-T01"
    acceptance_criteria:
      - "All 7 config structs compile with documented fields"
      - "Default implementations provide sensible production values"
      - "Serde Serialize/Deserialize for all types"
      - "Clone, Debug traits implemented"
      - "Config can be loaded from TOML/JSON file"
    test_file: "crates/context-graph-embeddings/tests/config_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 2.2"

  - id: "M03-T08"
    title: "Create Crate Structure and Module Exports"
    description: |
      Create context-graph-embeddings crate with Cargo.toml dependencies:
      tokio, async-trait, parking_lot, tracing, serde, bincode, thiserror,
      candle-core, candle-nn, candle-transformers, tokenizers, safetensors.
      Create module structure: lib.rs, config.rs, error.rs, pipeline.rs.
      Create submodules: models/, batch/, fusion/, cache/.
      Export all public types from lib.rs.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-embeddings/Cargo.toml"
    dependencies:
      - "M03-T01"
      - "M03-T02"
      - "M03-T03"
      - "M03-T04"
    acceptance_criteria:
      - "Cargo.toml with all required dependencies"
      - "lib.rs exports ModelId, FusedEmbedding, EmbeddingError, etc."
      - "Module structure matches technical spec"
      - "Crate compiles without errors"
      - "cargo doc generates clean documentation"
    test_file: "crates/context-graph-embeddings/tests/lib_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 1.2"

  # ============================================================
  # LOGIC LAYER: Core Implementations
  # ============================================================

  - id: "M03-T09"
    title: "Implement ModelRegistry with Lazy Loading"
    description: |
      Implement ModelRegistry struct with fields: models (RwLock<HashMap>),
      config (ModelRegistryConfig), loading_locks (per-model Semaphore),
      gpu_memory_tracker (RwLock<GpuMemoryTracker>).
      Include methods: new(), initialize(), get_model() (lazy load), load_model(),
      unload_model(), determine_device_placement(), estimate_model_memory(), stats().
      Support concurrent model loading with controlled parallelism (max 3 concurrent).
      Track GPU memory usage per device.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-embeddings/src/models/registry.rs"
    dependencies:
      - "M03-T06"
      - "M03-T07"
    acceptance_criteria:
      - "ModelRegistry compiles with all fields"
      - "get_model() loads on first access when lazy_loading enabled"
      - "Concurrent load requests for same model are serialized via Semaphore"
      - "estimate_model_memory() returns reasonable estimates per model"
      - "GPU memory tracking prevents OOM"
      - "stats() returns RegistryStats with loaded_models, total_memory"
    test_file: "crates/context-graph-embeddings/tests/registry_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 3.2"

  - id: "M03-T10"
    title: "Implement Base Text Embedding Model (MiniLM)"
    description: |
      Implement MiniLMModel struct implementing EmbeddingModel trait.
      Use candle-transformers for model loading and inference.
      Load tokenizer from tokenizers crate.
      Support batch inference with dynamic padding.
      Output dimension: 384D.
      Latency target: <10ms per input.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-embeddings/src/models/text/minilm.rs"
    dependencies:
      - "M03-T06"
      - "M03-T08"
    acceptance_criteria:
      - "MiniLMModel implements EmbeddingModel trait"
      - "embed() produces 384D vectors"
      - "embed_batch() processes multiple inputs efficiently"
      - "load() downloads model from HF Hub if not present"
      - "Memory footprint ~90MB"
      - "Latency <10ms for single input"
    test_file: "crates/context-graph-embeddings/tests/minilm_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 3"
      - "REQ-EMBED-001"

  - id: "M03-T11"
    title: "Implement Model Factory for All 12 Models"
    description: |
      Implement factory methods in ModelRegistry for all 12 models:
      create_minilm(), create_bge_large(), create_instructor_xl(), create_clip_vit(),
      create_whisper_large(), create_codebert(), create_sentence_t5(), create_mpnet(),
      create_contriever(), create_dragon_plus(), create_gte_large(), create_e5_mistral().
      Each factory creates model with appropriate device placement and quantization.
      Models E4 (CLIP) and E5 (Whisper) are multimodal; E6 (CodeBERT) is code-specific.
    layer: "logic"
    priority: "high"
    estimated_hours: 8
    file_path: "crates/context-graph-embeddings/src/models/loader.rs"
    dependencies:
      - "M03-T09"
      - "M03-T10"
    acceptance_criteria:
      - "All 12 factory methods implemented"
      - "Each model loads with correct architecture"
      - "Quantization applied per configuration"
      - "Device placement respected (GPU/CPU)"
      - "All models pass basic smoke test (embed single input)"
    test_file: "crates/context-graph-embeddings/tests/model_factory_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 3.2"
      - "REQ-EMBED-001 to REQ-EMBED-012"

  - id: "M03-T12"
    title: "Implement BatchProcessor with Dynamic Batching"
    description: |
      Implement BatchProcessor struct with fields: config (BatchConfig),
      registry (Arc<ModelRegistry>), queues (HashMap<ModelId, BatchQueue>).
      Implement BatchRequest struct with: input, model_id, response_tx, submitted_at.
      Include methods: new(), submit() (async), add_to_queue(), process_batch(),
      sort_by_sequence_length(), pad_inputs().
      Batch triggers: max_batch_size reached OR min_batch_size + max_wait_ms elapsed.
      Support PaddingStrategy: MaxLength, DynamicMax, PowerOfTwo, Bucket.
    layer: "logic"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-embeddings/src/batch/processor.rs"
    dependencies:
      - "M03-T07"
      - "M03-T09"
    acceptance_criteria:
      - "BatchProcessor compiles with per-model queues"
      - "submit() queues request and returns future"
      - "Batch triggered at max_batch_size (32) or timeout (50ms)"
      - "sort_by_length groups similar-length inputs"
      - "pad_inputs() implements all padding strategies"
      - "Throughput >100 items/sec at batch size 32"
    test_file: "crates/context-graph-embeddings/tests/batch_processor_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 3.3"
      - "REQ-EMBED-018, REQ-EMBED-019"

  - id: "M03-T13"
    title: "Implement CacheManager with LRU Eviction"
    description: |
      Implement CacheManager struct with fields: config (CacheConfig),
      cache (RwLock<LruCache>), metrics (CacheMetrics), disk_backend (Option<DiskCache>).
      Implement CacheEntry struct: embedding, created_at, last_accessed, access_count.
      Include methods: new(), get(), put(), contains(), remove(), clear(),
      compute_content_hash(), evict_lru(), persist_to_disk(), load_from_disk().
      Cache key: xxhash64 of content.
      Eviction policies: LRU (default), LFU, TtlLru, ARC.
    layer: "logic"
    priority: "high"
    estimated_hours: 4
    file_path: "crates/context-graph-embeddings/src/cache/manager.rs"
    dependencies:
      - "M03-T03"
      - "M03-T07"
    acceptance_criteria:
      - "CacheManager compiles with LRU cache backing"
      - "get() returns cached FusedEmbedding if present"
      - "put() stores embedding with content hash key"
      - "LRU eviction triggers at max_entries (100000)"
      - "Cache hit rate >80% on repeated queries"
      - "Disk persistence saves/loads cache on shutdown/startup"
    test_file: "crates/context-graph-embeddings/tests/cache_manager_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 3.4"
      - "REQ-EMBED-020, REQ-EMBED-021"

  - id: "M03-T14"
    title: "Implement Gating Network for FuseMoE"
    description: |
      Implement GatingNetwork struct with fields: projection (Linear),
      layer_norm (LayerNorm), config (GatingConfig).
      Include methods: new(), forward(input) -> expert_weights.
      Input: 12954D concatenated embedding.
      Output: 8D probability distribution over experts (softmax).
      Support configurable noise injection for exploration during training.
      Temperature scaling for softmax.
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-embeddings/src/fusion/gating.rs"
    dependencies:
      - "M03-T02"
      - "M03-T07"
      - "M03-T08"
    acceptance_criteria:
      - "GatingNetwork compiles with Linear projection layer"
      - "forward() produces 8D probability vector"
      - "Probabilities sum to 1.0"
      - "Temperature parameter affects distribution sharpness"
      - "Noise injection works during training mode"
      - "Layer normalization applied to input"
    test_file: "crates/context-graph-embeddings/tests/gating_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 3.4"
      - "REQ-EMBED-014"

  - id: "M03-T15"
    title: "Implement Expert Networks for FuseMoE"
    description: |
      Implement Expert struct with fields: input_proj (Linear), hidden (Linear),
      output_proj (Linear), activation (GELU), dropout (Dropout).
      Architecture: 12954D -> 4096D -> 4096D -> 1536D.
      Implement ExpertPool struct managing 8 Expert instances.
      Include methods: new(), forward(input, expert_idx) -> output.
      Shared input projection across experts for efficiency.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-embeddings/src/fusion/experts.rs"
    dependencies:
      - "M03-T08"
    acceptance_criteria:
      - "Expert struct compiles with MLP architecture"
      - "ExpertPool manages 8 Expert instances"
      - "forward() produces 1536D output from 12954D input"
      - "GELU activation used in hidden layer"
      - "Dropout configurable (0.0 for inference)"
      - "Memory-efficient shared projections"
    test_file: "crates/context-graph-embeddings/tests/experts_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 3.4"
      - "REQ-EMBED-013"

  - id: "M03-T16"
    title: "Implement FuseMoE Router and Top-K Selection"
    description: |
      Implement Router struct with fields: gating (GatingNetwork), experts (ExpertPool),
      top_k (usize = 2), load_balance_coef (f32 = 0.01).
      Include methods: new(), route(input) -> (selected_experts, weights),
      forward(input) -> fused_output, compute_load_balance_loss().
      Top-k selection: Select 2 experts with highest gating weights.
      Load balancing: Auxiliary loss to prevent expert collapse.
      Capacity factor limits tokens per expert (1.25x).
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-embeddings/src/fusion/router.rs"
    dependencies:
      - "M03-T14"
      - "M03-T15"
    acceptance_criteria:
      - "Router compiles with gating network and expert pool"
      - "route() returns top-2 expert indices and normalized weights"
      - "Weights for selected experts sum to 1.0"
      - "forward() produces 1536D fused embedding"
      - "compute_load_balance_loss() returns auxiliary loss"
      - "Capacity factor limits prevent expert overload"
    test_file: "crates/context-graph-embeddings/tests/router_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 3.4"
      - "REQ-EMBED-013, REQ-EMBED-014"

  - id: "M03-T17"
    title: "Implement FuseMoE Fusion Layer"
    description: |
      Implement FuseMoE struct combining all fusion components.
      Fields: router (Router), config (FusionConfig), output_norm (LayerNorm).
      Include methods: new(), fuse(concatenated_embedding) -> FusedEmbedding,
      fuse_batch(batch) -> Vec<FusedEmbedding>, get_routing_weights().
      Final output: 1536D normalized embedding.
      Latency target: <5ms fusion time.
      Support debug mode exposing routing weights.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-embeddings/src/fusion/fusemoe.rs"
    dependencies:
      - "M03-T03"
      - "M03-T16"
    acceptance_criteria:
      - "FuseMoE struct compiles with router and config"
      - "fuse() produces valid FusedEmbedding (1536D)"
      - "expert_weights populated with 8 values"
      - "selected_experts populated with top-2 indices"
      - "fuse_batch() processes multiple inputs efficiently"
      - "Latency <5ms for single fusion"
    test_file: "crates/context-graph-embeddings/tests/fusemoe_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 3.4"
      - "REQ-EMBED-013 to REQ-EMBED-015"

  # ============================================================
  # SURFACE LAYER: Pipeline and Integration
  # ============================================================

  - id: "M03-T18"
    title: "Implement EmbeddingPipeline Main Entry Point"
    description: |
      Implement EmbeddingPipeline struct with fields: config (EmbeddingConfig),
      registry (Arc<ModelRegistry>), batch_processor (Arc<BatchProcessor>),
      fusemoe (Arc<FuseMoE>), cache (Arc<CacheManager>), initialized (AtomicBool).
      Include methods: new(), initialize(), embed(content) -> FusedEmbedding,
      embed_batch(contents) -> Vec<FusedEmbedding>, embed_multimodal(input) -> FusedEmbedding,
      shutdown(), health_check().
      Full pipeline: cache check -> tokenize -> 12 models -> concatenate -> FuseMoE -> cache store.
      E2E latency target: <200ms single input, >100 items/sec batch.
    layer: "surface"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-embeddings/src/pipeline.rs"
    dependencies:
      - "M03-T09"
      - "M03-T12"
      - "M03-T13"
      - "M03-T17"
    acceptance_criteria:
      - "EmbeddingPipeline compiles with all components"
      - "embed() returns FusedEmbedding (1536D)"
      - "Cache hit returns in <1ms"
      - "Cache miss runs full pipeline <200ms"
      - "embed_batch() throughput >100 items/sec"
      - "health_check() verifies all components operational"
    test_file: "crates/context-graph-embeddings/tests/pipeline_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 3"
      - "REQ-EMBED-027"

  - id: "M03-T19"
    title: "Define CUDA Kernel Interfaces"
    description: |
      Create context-graph-cuda crate structure with module exports.
      Define CudaDevice trait with methods: allocate(), free(), copy_to_device(),
      copy_from_device(), synchronize().
      Define kernel interfaces for: embedding_forward(), fusemoe_gating(),
      fusemoe_expert_forward(), batch_normalize(), cosine_similarity().
      Include GpuBuffer struct for managing device memory.
      FFI bindings to actual CUDA kernels (placeholder implementations for now).
    layer: "surface"
    priority: "high"
    estimated_hours: 4
    file_path: "crates/context-graph-cuda/src/lib.rs"
    dependencies:
      - "M03-T08"
    acceptance_criteria:
      - "context-graph-cuda crate compiles"
      - "CudaDevice trait defined with memory operations"
      - "Kernel interfaces defined for embedding and fusion operations"
      - "GpuBuffer provides safe device memory management"
      - "FFI boundary clearly documented"
      - "CPU fallback path available when CUDA unavailable"
    test_file: "crates/context-graph-cuda/tests/interface_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 1.2"
      - "REQ-EMBED-024 to REQ-EMBED-026"

  - id: "M03-T20"
    title: "Implement GPU Memory Manager"
    description: |
      Implement GpuMemoryManager struct with fields: device_id, total_memory,
      allocated_memory, memory_pool (BTreeMap<size, Vec<GpuBuffer>>).
      Include methods: new(), allocate(size) -> GpuBuffer, free(buffer),
      get_available(), get_used(), defragment(), set_memory_limit().
      Memory pool for efficient reuse of common buffer sizes.
      Automatic defragmentation when fragmentation exceeds threshold.
      Memory limit enforcement (<16GB for all models per spec).
    layer: "surface"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-cuda/src/memory.rs"
    dependencies:
      - "M03-T19"
    acceptance_criteria:
      - "GpuMemoryManager compiles with memory tracking"
      - "allocate() returns pooled buffer when available"
      - "free() returns buffer to pool for reuse"
      - "Memory limit enforced (default 16GB)"
      - "Allocation failure returns GpuError"
      - "Defragmentation reduces fragmentation"
    test_file: "crates/context-graph-cuda/tests/memory_tests.rs"
    spec_refs:
      - "TECH-EMBED-003 Section 3"
      - "REQ-EMBED-024"

  - id: "M03-T21"
    title: "Implement EmbeddingProvider Trait from Ghost System"
    description: |
      Implement the EmbeddingProvider trait from Module 1 Ghost System.
      Bridge EmbeddingPipeline to EmbeddingProvider interface:
      - embed(content: &str) -> Vec<f32> (1536D)
      - embed_batch(contents: &[&str]) -> Vec<Vec<f32>>
      - dimension() -> usize (1536)
      Replace stub provider from Module 1 with real implementation.
      Thread-safe via Arc<EmbeddingPipeline>.
    layer: "surface"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-embeddings/src/provider.rs"
    dependencies:
      - "M03-T18"
    acceptance_criteria:
      - "EmbeddingProvider trait implemented"
      - "embed() returns 1536D Vec<f32>"
      - "embed_batch() processes multiple strings"
      - "dimension() returns 1536"
      - "Thread-safe implementation"
      - "Compatible with Module 1 EmbeddingProvider interface"
    test_file: "crates/context-graph-embeddings/tests/provider_tests.rs"
    spec_refs:
      - "REQ-EMBED-029"
      - "Module 1 Ghost System dependency"

  - id: "M03-T22"
    title: "Implement Hot-Swap Model Loading"
    description: |
      Extend ModelRegistry with hot-swap capability:
      - swap_model(model_id, new_config) -> EmbeddingResult<()>
      - validate_model_compatibility(model_id, new_model) -> bool
      Request queue during swap (no drops).
      Atomic traffic redirection after load complete.
      Rollback if new model fails validation.
      Maximum swap duration: 60 seconds.
    layer: "surface"
    priority: "medium"
    estimated_hours: 3
    file_path: "crates/context-graph-embeddings/src/models/registry.rs"
    dependencies:
      - "M03-T09"
    acceptance_criteria:
      - "swap_model() loads new model in background"
      - "Requests queue during swap, no errors"
      - "Atomic switchover after validation"
      - "validate_model_compatibility() checks dimension match"
      - "Rollback preserves previous model on failure"
      - "Swap completes in <60 seconds"
    test_file: "crates/context-graph-embeddings/tests/hot_swap_tests.rs"
    spec_refs:
      - "REQ-EMBED-022, REQ-EMBED-023"

  - id: "M03-T23"
    title: "Create Module Integration Tests"
    description: |
      Implement comprehensive integration tests for Module 3:
      - End-to-end embedding (text -> 12 models -> FuseMoE -> 1536D)
      - Batch processing throughput (>100 items/sec)
      - Cache hit/miss behavior
      - Model hot-swap without request drops
      - GPU memory stays within 16GB budget
      - Latency targets met (<200ms single, >100/sec batch)
      - All 12 models produce valid embeddings
    layer: "surface"
    priority: "critical"
    estimated_hours: 6
    file_path: "crates/context-graph-embeddings/tests/integration_tests.rs"
    dependencies:
      - "M03-T18"
      - "M03-T21"
      - "M03-T22"
    acceptance_criteria:
      - "E2E test embeds text and gets 1536D output"
      - "Batch test achieves >100 items/sec throughput"
      - "Cache test achieves >80% hit rate on repeated content"
      - "Hot-swap test completes without request errors"
      - "Memory test stays under 16GB GPU allocation"
      - "Test coverage >80% for embeddings crate"
    test_file: "crates/context-graph-embeddings/tests/integration_tests.rs"
    spec_refs:
      - "All REQ-EMBED-* requirements"

  - id: "M03-T24"
    title: "Implement Benchmarks and Performance Validation"
    description: |
      Create benchmark suite using criterion:
      - Single embedding latency (target <200ms P95)
      - Batch embedding throughput (target >100/sec)
      - FuseMoE fusion latency (target <5ms)
      - Cache lookup latency (target <1ms)
      - Per-model latency profiling
      Generate benchmark reports with percentile breakdowns.
      Add CI integration for performance regression detection.
    layer: "surface"
    priority: "high"
    estimated_hours: 4
    file_path: "crates/context-graph-embeddings/benches/embedding_bench.rs"
    dependencies:
      - "M03-T18"
    acceptance_criteria:
      - "Criterion benchmarks for all performance targets"
      - "Single embedding P95 <200ms verified"
      - "Batch throughput >100/sec verified"
      - "FuseMoE fusion <5ms verified"
      - "Benchmark results saved for comparison"
      - "CI job runs benchmarks on merge"
    test_file: "N/A (benchmarks)"
    spec_refs:
      - "REQ-EMBED-026, REQ-EMBED-027"
```

---

## Dependency Graph

```
M03-T01 (ModelId) ──────────────────────────┐
M03-T04 (EmbeddingError) ───────────────────┤
M03-T05 (ModelInput) ───────────────────────┼─► M03-T06 (EmbeddingModel Trait)
                                            │
M03-T01 ────────────────────────────────────┼─► M03-T02 (ModelEmbedding)
M03-T01 ────────────────────────────────────┼─► M03-T03 (FusedEmbedding)
M03-T01 ────────────────────────────────────┼─► M03-T07 (Configuration)
                                            │
M03-T01,T02,T03,T04 ────────────────────────┴─► M03-T08 (Crate Structure)

M03-T06,T07 ─► M03-T09 (ModelRegistry) ─┬─► M03-T10 (MiniLM Model)
                                        │
                                        └─► M03-T11 (All 12 Model Factories)

M03-T07,T09 ─► M03-T12 (BatchProcessor)

M03-T03,T07 ─► M03-T13 (CacheManager)

M03-T02,T07,T08 ─► M03-T14 (Gating Network) ─┐
M03-T08 ──────► M03-T15 (Expert Networks) ───┼─► M03-T16 (Router) ─► M03-T17 (FuseMoE)
                                             │
M03-T03 ─────────────────────────────────────┘

M03-T09,T12,T13,T17 ─► M03-T18 (EmbeddingPipeline)

M03-T08 ─► M03-T19 (CUDA Interfaces) ─► M03-T20 (GPU Memory Manager)

M03-T18 ─► M03-T21 (EmbeddingProvider Trait)
M03-T09 ─► M03-T22 (Hot-Swap)

M03-T18,T21,T22 ─► M03-T23 (Integration Tests)
M03-T18 ─► M03-T24 (Benchmarks)
```

---

## Implementation Order (Recommended)

### Week 1: Foundation
1. M03-T01: ModelId enum and dimension constants
2. M03-T04: EmbeddingError enum
3. M03-T05: ModelInput and InputType enums
4. M03-T02: ModelEmbedding and ConcatenatedEmbedding structs
5. M03-T03: FusedEmbedding struct
6. M03-T06: EmbeddingModel trait
7. M03-T07: Configuration types
8. M03-T08: Crate structure and module exports

### Week 2: Model Registry and Base Models
9. M03-T09: ModelRegistry with lazy loading
10. M03-T10: Base MiniLM model implementation
11. M03-T11: All 12 model factory methods
12. M03-T12: BatchProcessor with dynamic batching

### Week 3: Fusion and Cache
13. M03-T13: CacheManager with LRU eviction
14. M03-T14: Gating network for FuseMoE
15. M03-T15: Expert networks for FuseMoE
16. M03-T16: FuseMoE router and top-k selection
17. M03-T17: FuseMoE fusion layer

### Week 4: Pipeline Integration and Testing
18. M03-T18: EmbeddingPipeline main entry point
19. M03-T19: CUDA kernel interfaces
20. M03-T20: GPU memory manager
21. M03-T21: EmbeddingProvider trait implementation
22. M03-T22: Hot-swap model loading
23. M03-T23: Integration tests
24. M03-T24: Benchmarks and performance validation

---

## Quality Gates

| Gate | Criteria | Required For |
|------|----------|--------------|
| Foundation Complete | M03-T01 through M03-T08 pass all tests | Week 2 start |
| Models Operational | M03-T09 through M03-T12 pass all tests | Week 3 start |
| Fusion Complete | M03-T13 through M03-T17 pass all tests | Week 4 start |
| Module Complete | All 24 tasks complete, >80% coverage | Module 4 start |

---

## Performance Targets Summary

| Component | Target | Metric |
|-----------|--------|--------|
| Single embedding E2E | <200ms | P95 latency |
| Batch throughput | >100 items/sec | At batch size 32 |
| Cache hit | <1ms | Lookup latency |
| FuseMoE fusion | <5ms | Single fusion |
| GPU memory | <16GB | All models loaded |
| Cache hit rate | >80% | Under normal load |

---

## FuseMoE Configuration Summary

| Parameter | Value |
|-----------|-------|
| Number of experts | 8 |
| Top-k routing | 2 |
| Input dimension | 12,954D (concatenated) |
| Output dimension | 1,536D |
| Expert hidden dim | 4,096D |
| Load balance coef | 0.01 |
| Capacity factor | 1.25 |

---

*Generated: 2025-12-31*
*Module: 03 - Embedding Pipeline*
*Version: 1.0.0*
