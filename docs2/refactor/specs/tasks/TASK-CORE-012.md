# TASK-CORE-012: Embedding Model Loading Infrastructure

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-CORE-012 |
| **Title** | Embedding Model Loading Infrastructure |
| **Status** | :white_check_mark: DONE |
| **Layer** | Foundation |
| **Sequence** | 12 |
| **Complexity** | High |
| **Completed** | 2026-01-09 |
| **Verified By** | 25 tests passing (6 conversions + 6 unified + 13 slots) |

## Dependencies

| Task | Status | What It Provides |
|------|--------|------------------|
| TASK-CORE-011 | :white_check_mark: DONE | `ModelSlotManager`, `MemoryPressure`, 8GB budget enforcement at `crates/context-graph-embeddings/src/gpu/memory/slots.rs` |
| TASK-CORE-002 | :white_check_mark: DONE | `Embedder` enum (13 variants) at `crates/context-graph-core/src/teleological/embedder.rs` |

---

## Objective (COMPLETED)

Created the bridge between `ModelSlotManager` (TASK-CORE-011) and actual pretrained model loading via Candle.

**What was implemented:**
1. Bidirectional conversion between `Embedder` (core crate) and `ModelId` (embeddings crate)
2. `UnifiedModelLoader` struct that combines `ModelSlotManager` with `GpuModelLoader`
3. Memory-aware model loading with automatic LRU eviction
4. Comprehensive error types following Constitution AP-007 (no fallbacks)

---

## Implementation Summary (VERIFIED 2026-01-09)

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `crates/context-graph-embeddings/src/types/model_id/conversions.rs` | Embedder ↔ ModelId conversion with exhaustive match | ~170 new lines |
| `crates/context-graph-embeddings/src/gpu/model_loader/unified.rs` | `UnifiedModelLoader` with 8GB budget, LRU eviction | 687 lines |

### Files Modified

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/gpu/model_loader/mod.rs` | Added unified module exports, updated docs |
| `crates/context-graph-embeddings/src/gpu/mod.rs` | Re-exported `UnifiedModelLoader`, `LoaderConfig`, etc. |

### Actual File Locations (VERIFIED)

```
crates/context-graph-embeddings/
├── src/
│   ├── gpu/
│   │   ├── memory/
│   │   │   └── slots.rs          # ModelSlotManager (from TASK-CORE-011)
│   │   ├── model_loader/
│   │   │   ├── mod.rs            # Module exports
│   │   │   ├── unified.rs        # NEW: UnifiedModelLoader
│   │   │   ├── loader.rs         # GpuModelLoader (Candle)
│   │   │   ├── config.rs         # BertConfig parsing
│   │   │   ├── weights.rs        # BertWeights struct
│   │   │   └── error.rs          # ModelLoadError
│   │   └── mod.rs                # GPU module exports
│   └── types/
│       └── model_id/
│           ├── core.rs           # ModelId enum
│           └── conversions.rs    # MODIFIED: Embedder ↔ ModelId
└── tests/
    └── gpu_memory_slots_test.rs  # Integration tests
```

---

## Key Types Implemented

### 1. Embedder ↔ ModelId Conversion

**File**: `crates/context-graph-embeddings/src/types/model_id/conversions.rs`

```rust
// From Embedder (core) to ModelId (embeddings)
impl From<Embedder> for ModelId {
    fn from(embedder: Embedder) -> Self {
        match embedder {
            Embedder::Semantic => ModelId::Semantic,
            Embedder::TemporalRecent => ModelId::TemporalRecent,
            Embedder::TemporalPeriodic => ModelId::TemporalPeriodic,
            Embedder::TemporalPositional => ModelId::TemporalPositional,
            Embedder::Causal => ModelId::Causal,
            Embedder::Sparse => ModelId::Sparse,
            Embedder::Code => ModelId::Code,
            Embedder::Graph => ModelId::Graph,
            Embedder::Hdc => ModelId::Hdc,
            Embedder::Multimodal => ModelId::Multimodal,
            Embedder::Entity => ModelId::Entity,
            Embedder::LateInteraction => ModelId::LateInteraction,
            Embedder::KeywordSplade => ModelId::Splade,  // NOTE: E13 naming difference
        }
    }
}

// From ModelId (embeddings) to Embedder (core)
impl From<ModelId> for Embedder {
    // ... reverse mapping
}
```

**Key Point**: E13 is `KeywordSplade` in core crate, `Splade` in embeddings crate.

### 2. UnifiedModelLoader

**File**: `crates/context-graph-embeddings/src/gpu/model_loader/unified.rs`

```rust
pub struct UnifiedModelLoader {
    config: LoaderConfig,
    slot_manager: Arc<RwLock<ModelSlotManager>>,
    gpu_loader: GpuModelLoader,
    loaded_weights: Arc<RwLock<HashMap<Embedder, BertWeights>>>,
}

pub struct LoaderConfig {
    pub models_dir: PathBuf,              // e.g., "./models"
    pub memory_budget: usize,             // Default: 8GB
    pub enable_auto_eviction: bool,       // Default: true
    pub preload_models: Vec<ModelId>,     // Models to load on init
}
```

### 3. Error Types

```rust
pub enum UnifiedLoaderError {
    ConfigError { source: LoaderConfigError },
    GpuInitFailed { message: String },
    ModelNotFound { model_id: ModelId, path: PathBuf },
    OutOfMemory { requested: usize, available: usize, budget: usize },
    ModelLoadFailed { model_id: ModelId, source: ModelLoadError },
    LockPoisoned,
}

pub enum LoaderConfigError {
    ZeroBudget,
    ModelsDirectoryNotFound { path: PathBuf },
}
```

---

## Test Results (VERIFIED)

```bash
# All tests pass:
cargo test -p context-graph-embeddings --features cuda --lib conversions
# test result: ok. 6 passed

cargo test -p context-graph-embeddings --features cuda --lib unified
# test result: ok. 6 passed

cargo test -p context-graph-embeddings --features cuda --test gpu_memory_slots_test
# test result: ok. 13 passed
```

### Test Coverage

| Category | Tests | Status |
|----------|-------|--------|
| Embedder → ModelId conversion (all 13) | 1 | :white_check_mark: |
| ModelId → Embedder conversion (all 13) | 1 | :white_check_mark: |
| Roundtrip preservation | 1 | :white_check_mark: |
| Index preservation | 1 | :white_check_mark: |
| LoaderConfig defaults | 1 | :white_check_mark: |
| LoaderConfig builder | 1 | :white_check_mark: |
| LoaderConfig model_path | 1 | :white_check_mark: |
| LoaderConfig validation (zero budget) | 1 | :white_check_mark: |
| LoaderConfig validation (missing dir) | 1 | :white_check_mark: |
| MemoryStatsSnapshot Debug | 1 | :white_check_mark: |
| GPU memory slots (13 integration tests) | 13 | :white_check_mark: |

---

## How to Use

### Basic Usage

```rust
use context_graph_embeddings::gpu::{UnifiedModelLoader, LoaderConfig};
use context_graph_embeddings::types::ModelId;

// 1. Configure
let config = LoaderConfig::with_models_dir("/path/to/models")
    .with_budget(8 * 1024 * 1024 * 1024)  // 8GB
    .with_auto_eviction(true);

// 2. Create loader (initializes GPU)
let loader = UnifiedModelLoader::new(config)?;

// 3. Load a model (manages memory automatically)
loader.load_model(ModelId::Semantic)?;

// 4. Check memory usage
let stats = loader.memory_stats()?;
println!("Allocated: {} MB", stats.allocated / (1024 * 1024));

// 5. Access loaded weights
loader.with_weights(ModelId::Semantic, |weights| {
    println!("Parameters: {}", weights.param_count());
})?;
```

### Converting Between Crates

```rust
use context_graph_core::teleological::embedder::Embedder;
use context_graph_embeddings::types::ModelId;

// Core → Embeddings
let embedder = Embedder::Semantic;
let model_id: ModelId = embedder.into();

// Embeddings → Core
let model_id = ModelId::Code;
let embedder: Embedder = model_id.into();

// Index is preserved
assert_eq!(embedder.index(), model_id as usize);
```

---

## Constitution Compliance

| Rule | How Implemented |
|------|-----------------|
| ARCH-08 (GPU required) | `GpuModelLoader::new()` fails if no GPU; no CPU fallback |
| AP-007 (no fallbacks) | All errors propagate; no degraded modes |
| 8GB budget | `LoaderConfig::memory_budget` defaults to 8GB |
| LRU eviction | `allocate_with_eviction()` in ModelSlotManager |

---

## NO BACKWARDS COMPATIBILITY

Per Constitution AP-007, this implementation has NO fallbacks:

1. **GPU not available** → `UnifiedLoaderError::GpuInitFailed` (process should exit)
2. **Model file missing** → `UnifiedLoaderError::ModelNotFound` (no mock models)
3. **Out of memory** → `UnifiedLoaderError::OutOfMemory` (LRU eviction attempted first)
4. **Config invalid** → `UnifiedLoaderError::ConfigError` (no defaults for invalid config)

---

## Full State Verification Protocol

### Source of Truth

| Data | Location | How to Verify |
|------|----------|---------------|
| Allocated slots | `ModelSlotManager.slots` HashMap | `slot_manager.allocated_embedders()` |
| Loaded weights | `UnifiedModelLoader.loaded_weights` HashMap | `loader.loaded_models()` |
| Memory usage | `ModelSlotManager.total_allocated` | `loader.memory_stats()` |

### Verification Commands

```bash
# Run all TASK-CORE-012 tests
cargo test -p context-graph-embeddings --features cuda --lib conversions -- --nocapture
cargo test -p context-graph-embeddings --features cuda --lib unified -- --nocapture
cargo test -p context-graph-embeddings --features cuda --test gpu_memory_slots_test -- --nocapture

# Check memory allocation (requires GPU + models)
RUST_LOG=debug cargo test test_allocate_all_13_embedders -- --nocapture 2>&1 | grep -E "(Allocated|PASS|FAIL)"
```

### Evidence of Success (From Tests)

```
[PASS] All 13 Embedder -> ModelId conversions correct
[PASS] All 13 ModelId -> Embedder conversions correct
[PASS] Embedder <-> ModelId roundtrip preserves all 13 variants
[PASS] Index values preserved across Embedder <-> ModelId conversion
[PASS] LoaderConfig::default() has correct values
[PASS] LoaderConfig builder methods work correctly
[PASS] LoaderConfig::model_path() generates correct paths
[PASS] LoaderConfig validation rejects zero budget
[PASS] LoaderConfig validation rejects missing directory
[PASS] MemoryStatsSnapshot Debug impl works
```

---

## Edge Cases Tested

### Case 1: Zero Budget Rejection

```rust
let config = LoaderConfig::default().with_budget(0);
let result = config.validate();
assert!(matches!(result, Err(LoaderConfigError::ZeroBudget)));
```

### Case 2: 8GB Boundary Allocation

```rust
// Allocate all 13 models within 8GB budget
// See test: test_allocate_all_13_embedders_within_budget
// Expected: All 13 slots allocated, pressure = High (not Critical)
```

### Case 3: LRU Eviction Under Pressure

```rust
// Small budget forces eviction
// See test: test_fsv_edge_case_3_lru_eviction_selection
// Expected: Oldest accessed model evicted first
```

---

## What Remains (Out of Scope)

These items are for future tasks:

| Item | Future Task |
|------|-------------|
| Model quantization (FP8, PQ-8) | TASK-CORE-013 |
| Warm-up inference | Future integration |
| All 13 models actually loaded | Integration test with real models directory |

---

## Traceability

| Requirement | Source | Implementation |
|-------------|--------|----------------|
| 13 embedders | ARCH-05 | Exhaustive match in conversions.rs |
| GPU required | ARCH-08 | `GpuModelLoader::new()` failure propagates |
| 8GB budget | ARCH-08, slots.rs | `MODEL_BUDGET_BYTES = 8GB` default |
| No fallbacks | AP-07 | All errors are fatal; no degraded modes |
| Type safety | ARCH-02 | Explicit `From` impls, no index guessing |

---

## Summary for Future AI Agents

**This task is COMPLETE.** The following was implemented:

1. **Embedder ↔ ModelId conversion** in `conversions.rs` - exhaustive match for all 13 variants
2. **UnifiedModelLoader** in `unified.rs` - combines slot management with Candle loading
3. **LoaderConfig** - builder pattern for configuration
4. **Error types** - comprehensive, propagating errors per AP-007

**To verify the implementation:**

```bash
cargo test -p context-graph-embeddings --features cuda conversions unified gpu_memory_slots
```

**All 25 tests must pass.** If they don't, something has regressed.

**Next task in sequence:** TASK-CORE-013 (Embedding Quantization Infrastructure)
