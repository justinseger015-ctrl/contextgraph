# Automatic Causal Discovery Background Loop

## Implementation Task for AI Agent

**Date**: 2026-02-08
**Branch**: casetrack
**Target Hardware**: RTX 5090 32GB (Blackwell GB202) + CUDA 13.1
**Constraint**: ABSOLUTELY NO BACKWARDS COMPATIBILITY. Fail fast or work. No mocks in tests.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Context You Must Understand](#2-system-context-you-must-understand)
3. [What Exists Today (Audited 2026-02-08)](#3-what-exists-today-audited-2026-02-08)
4. [What You Must Build](#4-what-you-must-build)
5. [Exact Implementation Specification](#5-exact-implementation-specification)
6. [Resource Budget (Verified Against Codebase)](#6-resource-budget-verified-against-codebase)
7. [Research Findings & Corrections to Prior Assumptions](#7-research-findings--corrections-to-prior-assumptions)
8. [Configuration & Presets](#8-configuration--presets)
9. [Testing Requirements](#9-testing-requirements)
10. [Full State Verification Protocol](#10-full-state-verification-protocol)
11. [Risk Analysis & Plan Options](#11-risk-analysis--plan-options)
12. [Implementation Checklist](#12-implementation-checklist)

---

## 1. Executive Summary

**The task**: Replace the placeholder `warn!()` message in `CausalDiscoveryService::start()` (file: `crates/context-graph-causal-agent/src/service/mod.rs`, lines 480-488) with a real background loop that automatically discovers causal relationships between memories.

**What the placeholder currently does** (NOTHING):
```rust
warn!(
    "Causal discovery cycle tick: NOT IMPLEMENTED. \
     Background loop is running but no discovery occurs. \
     Use trigger_causal_discovery MCP tool for on-demand analysis."
);
```

**What the real loop must do**:
1. Fetch recent memories from RocksDB via `TeleologicalMemoryStore`
2. Run `MemoryScanner` to find candidate pairs (O(n²) E1 cosine clustering)
3. Run `CausalDiscoveryLLM` (Hermes 2 Pro 7B, Q5_K_M, via llama-cpp-2) to confirm causality
4. Run `E5EmbedderActivator` to generate dual cause/effect embeddings for confirmed relationships
5. Store `CausalRelationship` records in RocksDB + index in HNSW
6. Emit audit records for provenance
7. Persist a cursor so the loop resumes on restart
8. Adapt its interval based on discovery rate

**All infrastructure exists** — scanner, LLM, activator, store methods are all production-tested. Only the loop itself is a stub.

---

## 2. System Context You Must Understand

### 2.1 Constitution Rules (from `docs2/constitution.yaml`)

These are IMMUTABLE constraints:

- **ARCH-GPU-01**: GPU is mandatory — no CPU fallback for embeddings
- **ARCH-GPU-02**: All 13 embedders warm-loaded into VRAM at MCP server startup
- **ARCH-05**: All 13 embedders required — missing = fatal
- **ARCH-PROV-01**: Audit writes are non-fatal — `warn!` on failure, never block main operation
- **ARCH-PROV-02**: AuditRecord `target_id` uses `Uuid::nil()` for operations on CFs/files
- **ARCH-PROV-04**: All destructive ops MUST emit audit records
- **AP-77**: E5 MUST NOT use symmetric cosine — causal is directional
- **ARCH-LLM-01**: LLM inference is local-only via llama.cpp — no external API calls
- **ARCH-LLM-02**: All LLM output constrained by GBNF grammar for structured JSON
- **Serialization**: JSON (NOT bincode) for all provenance and SourceMetadata

### 2.2 The 13 Embedders (All GPU-Resident, FP16)

| # | Name | Model | Params | Dim | FP16 VRAM | GPU? |
|---|------|-------|--------|-----|-----------|------|
| E1 | Semantic | intfloat/e5-large-v2 | 335M | 1024 | ~700 MB | Yes |
| E2 | Temporal Recent | Custom algorithmic | — | 512 | 0 | CPU-only |
| E3 | Temporal Periodic | Custom algorithmic | — | 512 | 0 | CPU-only |
| E4 | Temporal Positional | Custom algorithmic | — | 512 | 0 | CPU-only |
| E5 | Causal | allenai/longformer-base-4096 | 125M | 768 | ~325 MB | Yes |
| E6 | Sparse | naver/splade-cocondenser | 110M | 30522→1536 | ~275 MB | Yes |
| E7 | Code | **Qodo/Qodo-Embed-1-1.5B** | **1,500M** | **1536** | **~3,000 MB** | Yes |
| E8 | Graph | intfloat/e5-large-v2 (shared w/ E1) | 0* | 1024 | ~60 MB | Yes |
| E9 | HDC | Custom algorithmic | — | 10000→1024 | 0 | CPU-only |
| E10 | Multimodal | openai/clip-vit-large-patch14 | 350M | 768 | ~800 MB | Yes |
| E11 | Entity | THU-KEG/KEPLER-Wiki5M-KE | 125M | 768 | ~60 MB | Yes |
| E12 | LateInteraction | colbert-ir/colbertv2.0 | 110M | 128/tok | ~225 MB | Yes |
| E13 | SPLADE v3 | prithivida/Splade_PP_en_v1 | 110M | 30522→1536 | ~275 MB | Yes |
| | | | | | **~5,720 MB** | |

**WARNING**: The codebase `MEMORY_ESTIMATES` in `crates/context-graph-embeddings/src/traits/model_factory/memory.rs` lists E7 at 550 MB FP32. This is **WRONG** for 1.5B parameters (real: ~6 GB FP32, ~3 GB FP16). Do not rely on that constant.

### 2.3 Concurrent Background Services (Already Running)

Your background loop will share GPU with these:

| Service | Interval | Embedders Used | File |
|---------|----------|---------------|------|
| **Graph Builder** | 60s | E1, E5, E7, E8, E10, E11 | `crates/context-graph-storage/src/graph_edges/builder.rs` |
| **File Watcher** | 500ms poll | All 13 (triggers `embed_all`) | `crates/context-graph-mcp/src/server.rs:898` |
| **Code Watcher** | 5s (optional) | Code embedders | `server.rs:1048` (env `CODE_PIPELINE_ENABLED=true`) |
| **TCP MCP Server** | On-demand | All 13 | `server.rs` (32 max connections) |

**ONNX sessions are thread-safe** — concurrent inference on the same session is fine. But the graph builder's 1000-item K-NN batch can saturate GPU for 5-10s, delaying your loop's Phase 4.

### 2.4 VRAM Budget (RTX 5090 32GB)

| Component | VRAM | Persistent? |
|-----------|------|-------------|
| 13 Embedders (FP16) | ~5.7 GB | Yes — always warm |
| ONNX Runtime overhead | ~0.4 GB | Yes |
| CUDA context + driver | ~0.3 GB | Yes |
| Hermes 2 Pro LLM (Q5_K_M) | ~5.0 GB | Yes — stays loaded |
| KV cache (4096 ctx) | ~1.0 GB | Per-inference |
| Scratch buffers | ~0.3 GB | Transient |
| **Total** | **~12.7 GB** | |
| **Usable (memory_fraction=0.9)** | **28.8 GB** | |
| **Free** | **~16.1 GB** | |

`GpuConfig` defaults: `memory_fraction: 0.9` (file: `crates/context-graph-embeddings/src/config/gpu.rs:98`).

---

## 3. What Exists Today (Audited 2026-02-08)

### 3.1 Causal Agent Crate: `crates/context-graph-causal-agent/`

| File | Status | Key Contents |
|------|--------|-------------|
| `src/lib.rs` (lines 54-70) | Complete | Re-exports all public types |
| `src/error.rs` (lines 11-76) | Complete | `CausalAgentError` with 14 variants, `CausalAgentResult<T>` |
| `src/types/mod.rs` (lines 1-708) | Complete | `CausalAnalysisResult`, `CausalCandidate`, `MemoryForAnalysis`, `DirectionalEmbeddings`, `CausalLinkDirection`, `ExtractedCausalRelationship`, `SourceSpan`, `CausalMarkers` |
| `src/scanner/mod.rs` (lines 1-434) | Complete | `MemoryScanner`, `ScannerConfig`. O(n²) E1 clustering. Scoring: markers +0.15, temporal +0.1, session +0.1, length +0.05, similarity +0.1. AGT-01 fix: does NOT mark_analyzed before LLM confirmation. |
| `src/activator/mod.rs` (lines 1-783) | Complete | `E5EmbedderActivator`. `activate_relationship()` → `(Vec<f32>, Vec<f32>)`. Uses `CausalModel::embed_dual_guided()` with LLM guidance. **Production mode requires `with_model()` constructor** — without `CausalModel`, fails fast (no `test-mode` feature). |
| `src/llm/mod.rs` (lines 1-1352) | Complete | `CausalDiscoveryLLM`. Hermes 2 Pro Mistral 7B via llama-cpp-2. Q5_K_M quantized. GBNF grammars: Causal, Graph, Validation, SingleText, MultiRelationship. `analyze_causal_relationship(&str, &str) → CausalAnalysisResult`. `extract_causal_relationships(&str) → MultiRelationshipResult`. Temperature 0.0, max_tokens 512, context_size 4096, n_gpu_layers u32::MAX. |
| `src/llm/prompt.rs` (lines 1-551) | Complete | `CausalPromptBuilder`. System prompts, few-shot examples. |
| **`src/service/mod.rs` (lines 1-671)** | **STUB at line 480** | `CausalDiscoveryService`. `run_discovery_cycle(&[MemoryForAnalysis])` works. `start()` is the **PLACEHOLDER** you must replace. `stop()` works (CRIT-04/05 fixes applied). |

### 3.2 The Stub You Must Replace

**File**: `crates/context-graph-causal-agent/src/service/mod.rs`
**Lines**: 439-500

```rust
pub async fn start(self: Arc<Self>) -> CausalAgentResult<()> {
    if self.running.swap(true, Ordering::SeqCst) {
        return Err(CausalAgentError::ServiceAlreadyRunning);
    }
    *self.status.write() = ServiceStatus::Starting;

    // Load model if not already loaded
    if !self.llm.is_loaded() {
        self.llm.load().await?;
    }
    *self.status.write() = ServiceStatus::Running;

    let (tx, mut rx) = mpsc::channel::<()>(1);
    *self.shutdown_tx.write() = Some(tx);

    let service = Arc::clone(&self);
    let handle = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = rx.recv() => {
                    info!("Causal discovery loop: shutdown signal received");
                    break;
                }
                _ = tokio::time::sleep(service.config.interval) => {
                    // ╔══════════════════════════════════════════════╗
                    // ║  THIS IS THE STUB — REPLACE THIS BLOCK      ║
                    // ╚══════════════════════════════════════════════╝
                    warn!(
                        "Causal discovery cycle tick: NOT IMPLEMENTED. \
                         Background loop is running but no discovery occurs. \
                         Use trigger_causal_discovery MCP tool for on-demand analysis."
                    );
                }
            }
        }
    });

    *self.join_handle.write() = Some(handle);
    Ok(())
}
```

### 3.3 Existing `run_discovery_cycle` (WORKS — Use It)

**File**: `crates/context-graph-causal-agent/src/service/mod.rs`, lines 311-398

This method already does:
1. Ensures LLM is loaded
2. Runs `scanner.find_candidates(memories)` to get `Vec<CausalCandidate>`
3. Calls `process_candidate()` for each (LLM analysis + E5 activation)
4. Returns `DiscoveryCycleResult` with counts

**BUT** it takes `&[MemoryForAnalysis]` as input — it does NOT fetch memories from the store. And it does NOT persist relationships to RocksDB. The in-memory `CausalGraph` is updated, but not the persistent store.

### 3.4 Store Trait Methods Available

**File**: `crates/context-graph-core/src/traits/teleological_memory_store/store.rs`

| Method | Signature | Exists? |
|--------|-----------|---------|
| `scan_fingerprints_for_clustering` | `(limit: Option<usize>) → Vec<(Uuid, [Vec<f32>; 13])>` | **YES** (line 545) |
| `get_content` | `(id: Uuid) → Option<String>` | **YES** (line 312) |
| `store_causal_relationship` | `(relationship: &CausalRelationship) → Uuid` | **YES** (line 584) |
| `append_audit_record` | `(record: &AuditRecord) → ()` | **YES** (line 750) |
| `get_source_metadata` | `(id: Uuid) → Option<SourceMetadata>` | **YES** (line 352) |
| `put_system_metadata` / `get_system_metadata` | — | **NO — DOES NOT EXIST** |

**CRITICAL**: There is no `put_system_metadata` method. For cursor persistence, you must either:
- **Option A**: Add `put_system_metadata`/`get_system_metadata` to the `TeleologicalMemoryStore` trait and implement in RocksDB store
- **Option B**: Store the cursor in an existing column family using a well-known key (e.g., key `b"causal_discovery_cursor"` in `CF_AUDIT_LOG` or a new CF)
- **Option C** (SIMPLEST): Add two methods to the trait: `store_processing_cursor(&self, key: &str, data: &[u8])` and `get_processing_cursor(&self, key: &str) → Option<Vec<u8>>`

### 3.5 How Server Creates the Service

**File**: `crates/context-graph-mcp/src/server.rs`, lines 516-606

The server creates `GraphDiscoveryService` (NOT `CausalDiscoveryService`) with:
```rust
let shared_llm = Arc::new(CausalDiscoveryLLM::new()?);
shared_llm.load().await?; // FAIL FAST — no fallback
let graph_discovery_service = Arc::new(GraphDiscoveryService::with_models(
    Arc::clone(&shared_llm),
    graph_model,
    graph_discovery_config,
));
```

**IMPORTANT**: The server currently does NOT create a `CausalDiscoveryService`. You must wire it in. The `CausalDiscoveryService` needs:
- The same `shared_llm: Arc<CausalDiscoveryLLM>` (already created)
- A `CausalModel` from the warm embedder provider (for E5 dual embeddings)
- A `TeleologicalMemoryStore` (already available as `teleological_store`)

### 3.6 CausalDiscoveryConfig Defaults

**File**: `crates/context-graph-causal-agent/src/service/mod.rs`, lines 46-82

```rust
interval: Duration::from_secs(3600),  // 1 hour — TOO SLOW, change to 120s
batch_size: 50,
min_confidence: 0.7,
skip_analyzed: true,
```

### 3.7 Git History (Key Commits)

| Commit | Change |
|--------|--------|
| `9a1b4a1` | Eliminated test stubs, real GPU tests everywhere |
| `35f65b8` | Forensic remediation: 62 fixes across 7 batches |
| `3a321c5` | LLM-guided E5 causal embedding enhancement |
| `eba2305` | E5 dual embeddings for asymmetric causal search |
| `d26845d` | CausalRelationship storage + search_causal_relationships MCP tool |
| `581c04b` | Initial causal discovery agent + MCP tools |

---

## 4. What You Must Build

### 4.1 Changes Required

| # | File | Action | Description |
|---|------|--------|-------------|
| 1 | `causal-agent/src/service/mod.rs` | **MODIFY** | Replace placeholder loop (lines 480-488) with real discovery logic. Modify `start()` to accept `store: Arc<dyn TeleologicalMemoryStore>`. Add cursor and adaptive interval. |
| 2 | `causal-agent/src/service/mod.rs` | **ADD** | Add `harvest_memories()`, `compute_next_interval()`, `save_cursor()`, `load_cursor()` methods. Add `DiscoveryCursor` struct. |
| 3 | `core/src/traits/teleological_memory_store/store.rs` | **MODIFY** | Add `store_processing_cursor(&self, key: &str, data: &[u8]) → CoreResult<()>` and `get_processing_cursor(&self, key: &str) → CoreResult<Option<Vec<u8>>>` to trait. |
| 4 | `core/src/traits/teleological_memory_store/defaults.rs` | **MODIFY** | Add default impls that return `Err(CoreError::NotImplemented)` |
| 5 | `storage/src/teleological/rocksdb_store/crud.rs` | **MODIFY** | Implement `store_processing_cursor` and `get_processing_cursor` using an existing CF (e.g., `CF_AUDIT_LOG` with a `cursor::` prefix key, or add `CF_PROCESSING_CURSORS`). |
| 6 | `mcp/src/server.rs` | **MODIFY** | Create `CausalDiscoveryService`, wire `shared_llm` + `CausalModel` + store, call `start()`. |
| 7 | `causal-agent/src/service/mod.rs` | **MODIFY** | Modify `CausalDiscoveryConfig` to add: `enable_background: bool`, `rescan_window_secs: u64`, `min_interval: Duration`, `max_interval: Duration`, `max_consecutive_errors: u32`. |

### 4.2 What You Must NOT Build

- **NO Green Context FFI wrappers** — No Rust bindings exist, experimental, no forward progress guarantees. Use CUDA stream priorities if needed.
- **NO FP4 quantization** — llama.cpp does NOT support FP4 yet (mxfp4 is experimental, not merged). Use Q5_K_M (current default).
- **NO backwards compatibility** — If the new cursor CF doesn't exist in an old database, fail fast with a clear error message.
- **NO mock data in tests** — Use real RocksDB + real GPU embeddings.
- **NO workarounds** — If something doesn't work, it errors out with descriptive logging.

---

## 5. Exact Implementation Specification

### 5.1 New Type: DiscoveryCursor

```rust
// Add to crates/context-graph-causal-agent/src/service/mod.rs

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiscoveryCursor {
    pub last_timestamp: Option<chrono::DateTime<chrono::Utc>>,
    pub last_fingerprint_id: Option<Uuid>,
    pub cycles_completed: u64,
    pub total_relationships: u64,
}
```

### 5.2 Modified `start()` Signature

```rust
pub async fn start(
    self: Arc<Self>,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> CausalAgentResult<()>
```

**Breaking change**: `start()` now requires a `store` parameter. All callers must be updated.

### 5.3 The Real Background Loop Logic

Replace the `warn!()` at line 484 with:

```rust
// Inside the tokio::select! sleep branch:
match service.run_background_tick(&store, &mut cursor, &mut consecutive_errors).await {
    Ok(metrics) => {
        consecutive_errors = 0;
        current_interval = service.compute_next_interval(&metrics);
        service.save_cursor(&store, &cursor).await.ok(); // non-fatal
        info!(
            cycle = metrics.cycle_number,
            harvested = metrics.memories_harvested,
            discovered = metrics.relationships_discovered,
            duration_ms = metrics.total_duration.as_millis(),
            next_interval_s = current_interval.as_secs(),
            "Causal discovery cycle complete"
        );
    }
    Err(e) => {
        consecutive_errors += 1;
        error!(error = %e, consecutive = consecutive_errors, "Causal discovery cycle failed");
        if consecutive_errors >= service.config.max_consecutive_errors {
            error!("Too many consecutive failures, pausing background discovery");
            current_interval = Duration::from_secs(3600); // 1 hour backoff
        } else {
            current_interval = Duration::from_secs(60 * 2u64.pow(consecutive_errors - 1));
        }
    }
}
```

### 5.4 `run_background_tick()` Method

```rust
async fn run_background_tick(
    &self,
    store: &Arc<dyn TeleologicalMemoryStore>,
    cursor: &mut DiscoveryCursor,
    consecutive_errors: &mut u32,
) -> CausalAgentResult<CycleMetrics> {
    let cycle_start = Instant::now();
    let cycle_number = cursor.cycles_completed + 1;

    // Phase 1: Harvest — fetch memories from store
    let memories = self.harvest_memories(store).await?;
    if memories.is_empty() {
        cursor.cycles_completed = cycle_number;
        return Ok(CycleMetrics { cycle_number, memories_harvested: 0, ..Default::default() });
    }

    // Phase 2+3: Scan + Analyze + Activate — use existing run_discovery_cycle
    let result = self.run_discovery_cycle(&memories).await?;

    // Phase 4: Persist confirmed relationships to RocksDB
    // NOTE: run_discovery_cycle only updates in-memory CausalGraph.
    // You must also store to RocksDB. The activate_relationship() in
    // process_candidate() generates embeddings but doesn't call
    // store_causal_relationship(). You need to add this.
    //
    // OPTION A: Modify process_candidate() to accept &store and persist
    // OPTION B: After run_discovery_cycle, iterate the in-memory graph for new edges and persist
    // RECOMMENDED: Option A — modify process_candidate() to accept store

    // Phase 5: Emit audit
    if let Err(e) = store.append_audit_record(&AuditRecord::new(
        AuditOperation::RelationshipDiscovered,
        Uuid::nil(), // targets the CF, not a single entity
    ).with_rationale("background_discovery_loop")
     .with_parameters(&serde_json::json!({
        "cycle": cycle_number,
        "discovered": result.relationships_confirmed,
        "rejected": result.relationships_rejected,
    }).to_string())).await {
        warn!(error = %e, "Audit write failed (non-fatal per ARCH-PROV-01)");
    }

    // Update cursor
    if let Some(last) = memories.last() {
        cursor.last_timestamp = Some(last.created_at);
        cursor.last_fingerprint_id = Some(last.id);
    }
    cursor.cycles_completed = cycle_number;
    cursor.total_relationships += result.relationships_confirmed as u64;

    Ok(CycleMetrics {
        cycle_number,
        memories_harvested: memories.len(),
        relationships_discovered: result.relationships_confirmed,
        relationships_rejected: result.relationships_rejected,
        total_duration: cycle_start.elapsed(),
        ..Default::default()
    })
}
```

### 5.5 `harvest_memories()` Method

```rust
async fn harvest_memories(
    &self,
    store: &Arc<dyn TeleologicalMemoryStore>,
) -> CausalAgentResult<Vec<MemoryForAnalysis>> {
    let fingerprints = store
        .scan_fingerprints_for_clustering(Some(self.config.batch_size))
        .await
        .map_err(|e| CausalAgentError::StorageError { message: e.to_string() })?;

    let mut memories = Vec::with_capacity(fingerprints.len());
    for (id, embeddings) in fingerprints {
        let content = match store.get_content(id).await {
            Ok(Some(c)) => c,
            Ok(None) => {
                warn!(id = %id, "Fingerprint has no content, skipping");
                continue;
            }
            Err(e) => {
                warn!(id = %id, error = %e, "Failed to get content, skipping");
                continue;
            }
        };

        memories.push(MemoryForAnalysis {
            id,
            content,
            e1_embedding: embeddings[0].clone(), // E1 is index 0
            created_at: chrono::Utc::now(), // TODO: get real created_at from SourceMetadata
            session_id: None,
        });
    }

    info!(count = memories.len(), "Harvested memories for causal discovery");
    Ok(memories)
}
```

### 5.6 Adaptive Interval

```rust
fn compute_next_interval(&self, metrics: &CycleMetrics) -> Duration {
    if metrics.memories_harvested == 0 {
        Duration::from_secs(600) // 10 min: nothing to process
    } else if metrics.relationships_discovered == 0 {
        Duration::from_secs(300) // 5 min: content but no causation
    } else if metrics.relationships_discovered <= 5 {
        Duration::from_secs(120) // 2 min: moderate discovery
    } else {
        Duration::from_secs(30) // 30s: heavy causal content
    }
    .max(self.config.min_interval)
    .min(self.config.max_interval)
}
```

### 5.7 Cursor Persistence

```rust
const CURSOR_KEY: &str = "causal_discovery_cursor";

async fn save_cursor(
    &self,
    store: &Arc<dyn TeleologicalMemoryStore>,
    cursor: &DiscoveryCursor,
) -> CausalAgentResult<()> {
    let json = serde_json::to_vec(cursor)
        .map_err(|e| CausalAgentError::ParseError { message: e.to_string() })?;
    store.store_processing_cursor(CURSOR_KEY, &json).await
        .map_err(|e| CausalAgentError::StorageError { message: e.to_string() })?;
    Ok(())
}

async fn load_cursor(
    &self,
    store: &Arc<dyn TeleologicalMemoryStore>,
) -> DiscoveryCursor {
    match store.get_processing_cursor(CURSOR_KEY).await {
        Ok(Some(bytes)) => serde_json::from_slice(&bytes).unwrap_or_default(),
        Ok(None) => DiscoveryCursor::default(),
        Err(e) => {
            warn!(error = %e, "Failed to load cursor, starting fresh");
            DiscoveryCursor::default()
        }
    }
}
```

### 5.8 Wiring in server.rs

Add after the existing `GraphDiscoveryService` creation (around line 606):

```rust
// Create CausalDiscoveryService for background loop
let causal_model = {
    // Get the CausalModel from warm provider (same pattern as graph_model)
    let causal_model_arc = get_warm_causal_model()?; // You need to find the actual accessor
    causal_model_arc
};

let causal_discovery_config = CausalDiscoveryConfig {
    interval: Duration::from_secs(120), // Start at 2 min
    batch_size: 100,
    min_confidence: 0.7,
    enable_background: true,
    ..Default::default()
};

let causal_discovery_service = Arc::new(CausalDiscoveryService::with_models(
    Arc::clone(&shared_llm),
    causal_model,
    causal_discovery_config,
));

// Start background loop
let bg_store = Arc::clone(&teleological_store);
let bg_service = Arc::clone(&causal_discovery_service);
tokio::spawn(async move {
    if let Err(e) = bg_service.start(bg_store).await {
        error!(error = %e, "Failed to start causal discovery background loop");
    }
});
```

### 5.9 Relationship Persistence (Missing Piece)

The existing `process_candidate()` (line 401) calls `activate_relationship()` which generates embeddings and updates the in-memory `CausalGraph`, but does NOT call `store.store_causal_relationship()`.

You must modify `process_candidate()` to also persist to RocksDB, OR add a new method that wraps `run_discovery_cycle()` and persists results.

**Recommended approach**: Add a `store` parameter to `run_discovery_cycle()`:

```rust
pub async fn run_discovery_cycle(
    &self,
    memories: &[MemoryForAnalysis],
    store: Option<&Arc<dyn TeleologicalMemoryStore>>, // NEW
) -> CausalAgentResult<DiscoveryCycleResult>
```

Inside `process_candidate()`, after `activate_relationship()` succeeds, if `store` is `Some`:
```rust
let relationship = CausalRelationship {
    id: Uuid::new_v4(),
    source_fingerprint_id: candidate.cause_memory_id,
    cause_description: candidate.cause_content.clone(),
    effect_description: candidate.effect_content.clone(),
    confidence: analysis.confidence,
    mechanism: analysis.mechanism.clone(),
    mechanism_type: analysis.mechanism_type.as_ref().map(|m| m.as_str().to_string()),
    direction: analysis.direction.to_string(),
    cause_embedding: Some(cause_vec.clone()),
    effect_embedding: Some(effect_vec.clone()),
    llm_provenance: analysis.llm_provenance.clone(),
    created_at: chrono::Utc::now(),
    ..Default::default()
};

store.store_causal_relationship(&relationship).await
    .map_err(|e| CausalAgentError::StorageError { message: e.to_string() })?;
```

---

## 6. Resource Budget (Verified Against Codebase)

### 6.1 CPU Per Cycle

| Operation | 50 memories | 200 memories |
|-----------|-------------|--------------|
| E1 clustering (O(n²)) | ~1ms | ~15ms |
| Candidate scoring | ~1ms | ~5ms |
| LLM analysis (Q5_K_M, ~100 tok/s) | ~5-25s | ~5-25s* |
| E5 dual embedding | ~1.5s | ~4.5s |
| RocksDB writes | ~10ms | ~30ms |
| **Total** | **~8-28s** | **~10-30s** |

*Batch capped at `batch_size` (default 100 pairs) regardless of memory count.

### 6.2 Memory

- Discovery loop RAM: ~25-75 MB
- HNSW indices (16 total): ~200 MB at 10K items, ~2-4 GB at 100K (CPU RAM, not VRAM)
- RocksDB block cache: 256 MB (`DEFAULT_CACHE_SIZE` in `rocksdb_backend/config.rs:51`)
- 52 column families total

### 6.3 Power

- Phase 2+3 (LLM active): ~400W for ~5-25s
- Phase 4 (Embedders): ~250W for ~1.5-4.5s
- Idle between cycles: ~30W

---

## 7. Research Findings & Corrections to Prior Assumptions

### 7.1 FP4 Quantization — NOT AVAILABLE

**Prior assumption**: FP4 via llama.cpp for Blackwell gives 3x throughput, 70% VRAM reduction.

**Reality**: As of early 2026, llama.cpp does NOT support hardware FP4. An experimental `mxfp4` PR exists but is NOT merged or stable. The "Q4_K_M" format in llama.cpp is INT4-based k-quantization, NOT native FP4 tensor core acceleration.

**Action**: Use Q5_K_M (current default, ~5 GB, ~100 tok/s on RTX 5090). Do NOT write any FP4 code paths.

### 7.2 Green Contexts — NOT READY FOR PRODUCTION

**Prior assumption**: Green Contexts provide hardware SM isolation with Rust FFI wrappers.

**Reality**:
- No mature Rust bindings for Green Contexts exist
- Green Contexts do NOT guarantee forward progress even with disjoint SM partitions
- Workloads can use MORE SMs than provisioned (not less, but upper bound not guaranteed)
- Requires Compute 9.0+ and minimum 8-SM granularity

**Action**: Do NOT implement Green Context FFI. Use standard CUDA stream priorities for workload isolation. The background loop should use cooperative scheduling (yield when graph builder is active).

### 7.3 llama.cpp Concurrency — Limited

**Prior assumption**: Can batch LLM inference across multiple candidate pairs simultaneously.

**Reality**: llama.cpp's continuous batching supports `--parallel N` slots, but throughput is flat — it doesn't scale linearly with concurrency. Non-deterministic with `--parallel > 1`.

**Action**: Process candidate pairs sequentially. The LLM is already fast enough (~100 tok/s, ~5s per pair at max_tokens=512).

### 7.4 ONNX Runtime — Thread-Safe, No Priority Queue

**Reality**: ONNX sessions are thread-safe for concurrent `Run()` calls. But there's no built-in request prioritization — all callers are equal.

**Action**: Accept that graph builder batches may delay discovery loop embedding generation by 5-10s. Stagger the discovery loop timing to avoid the graph builder's 60s cycle.

### 7.5 Recommended Plan Options

| Option | Description | Pros | Cons | Recommended? |
|--------|-------------|------|------|-------------|
| **A: Minimal** | Replace stub with `harvest_memories() + run_discovery_cycle()` + cursor. No Green Contexts, no FP4, no parallel phases. | Simple, uses existing code, low risk | Sequential phases, no staggering with graph builder | **YES — Start here** |
| **B: Staggered** | Option A + offset loop timing by 30s from graph builder's 60s cycle | Avoids GPU contention | Couples to graph builder internals | Yes, as Phase 2 |
| **C: Parallel Phases** | Option B + mpsc channel between LLM analysis and embedding activation | 30-40% faster cycles | More complex, harder to debug | Optional optimization |
| **D: Green Contexts** | Option C + CUDA Green Context SM partitioning | Hardware isolation | No Rust bindings, experimental | **NO — Not ready** |

**Recommendation**: Implement Option A first. It's the Occam's razor answer. If performance is insufficient, add Option B staggering. Option C parallel phases can be added later if needed.

---

## 8. Configuration & Presets

### 8.1 Extended CausalDiscoveryConfig

Add these fields to the existing struct in `service/mod.rs`:

```rust
pub enable_background: bool,       // Default: false (explicit opt-in)
pub rescan_window_secs: u64,       // Default: 3600 (1 hour)
pub min_interval: Duration,        // Default: 30s
pub max_interval: Duration,        // Default: 600s (10 min)
pub max_consecutive_errors: u32,   // Default: 3
pub intra_batch_size: usize,       // Default: 20 (memories for intra-extraction)
```

### 8.2 Environment Variable Overrides

```bash
CAUSAL_DISCOVERY_ENABLED=true       # Master switch
CAUSAL_DISCOVERY_INTERVAL_SECS=120  # Initial interval
CAUSAL_DISCOVERY_BATCH_SIZE=100     # Max pairs per cycle
CAUSAL_DISCOVERY_MIN_CONFIDENCE=0.7 # LLM threshold
CAUSAL_DISCOVERY_RESCAN_WINDOW=3600 # Re-scan window in seconds
```

### 8.3 Presets

| Preset | Interval | Batch | Confidence | VRAM Peak |
|--------|----------|-------|------------|-----------|
| **Balanced** | 2 min | 100 | 0.7 | ~12.7 GB |
| **Quality** | 5 min | 50 | 0.8 | ~12.7 GB |
| **Background** | 10 min | 30 | 0.8 | ~12.7 GB |

---

## 9. Testing Requirements

### 9.1 Rules

- **NO MOCKS** — All tests use real RocksDB + real GPU embeddings (`ProductionMultiArrayProvider`)
- Use `create_test_handlers()` which returns `(Handlers, TempDir)` with real stores
- Tests must **fail fast** if GPU is unavailable
- Tests must verify **physical proof** that data was persisted (read it back from RocksDB)

### 9.2 Synthetic Test Data

Use these exact inputs so you know the expected outputs:

**Causal pair (should discover relationship)**:
```
Memory A: "Deploying version 2.3 of the authentication service introduced a breaking change
in the session token format, causing all existing sessions to become invalid."

Memory B: "After the v2.3 deployment, the monitoring dashboard showed a 500% spike in
HTTP 401 errors across all microservices that depend on the authentication service."
```
- **Expected**: `has_causal_link: true`, `direction: ACausesB`, `confidence >= 0.7`
- **Expected**: `mechanism` mentions "session token format change" or similar
- **Verify**: `CausalRelationship` stored in RocksDB with `cause_embedding` (768D, non-zero) and `effect_embedding` (768D, non-zero, different from cause)

**Non-causal pair (should reject)**:
```
Memory C: "The weather in Seattle was particularly rainy during November 2025."

Memory D: "The quarterly revenue report for Q4 2025 exceeded analyst expectations by 12%."
```
- **Expected**: `has_causal_link: false` OR `confidence < 0.7`
- **Verify**: NO `CausalRelationship` stored for this pair

**Intra-memory extraction**:
```
Memory E: "The database index on the users table was dropped during migration, which caused
query latency to spike from 5ms to 2000ms. This triggered circuit breakers in the API
gateway, which prevented all downstream services from functioning. The incident lasted
4 hours until the index was manually recreated."
```
- **Expected**: At least 2 relationships extracted:
  1. index dropped → query latency spike
  2. latency spike → circuit breakers triggered
- **Verify**: Multiple `CausalRelationship` records in RocksDB with `source_fingerprint_id` = Memory E's UUID

### 9.3 Test Functions to Write

| Test | What It Verifies |
|------|-----------------|
| `test_background_loop_discovers_causal_pair` | Store memories A+B, start loop, wait for 1 cycle, verify CausalRelationship in RocksDB |
| `test_background_loop_rejects_non_causal` | Store memories C+D, start loop, verify NO CausalRelationship created |
| `test_background_loop_extracts_intra_memory` | Store memory E, start loop, verify multiple relationships extracted |
| `test_cursor_persists_across_restart` | Run 1 cycle, stop, read cursor from store, verify `cycles_completed == 1` |
| `test_cursor_prevents_reprocessing` | Run 2 cycles with same data, verify relationships not duplicated |
| `test_adaptive_interval_speeds_up` | Run cycle that discovers 6+ relationships, verify next interval = 30s |
| `test_adaptive_interval_slows_down` | Run cycle with 0 memories, verify next interval = 600s |
| `test_consecutive_errors_pause` | Force 3 LLM failures, verify interval jumps to 3600s |
| `test_stop_graceful_shutdown` | Start loop, call stop(), verify JoinHandle resolves within 10s |
| `test_relationship_has_embeddings` | Verify stored CausalRelationship has non-zero 768D cause_embedding and effect_embedding |
| `test_audit_record_emitted` | Verify AuditRecord with `RelationshipDiscovered` operation exists after cycle |

### 9.4 Edge Cases to Test

| Edge Case | Input | Expected Behavior |
|-----------|-------|-------------------|
| Empty database | 0 memories | Cycle completes immediately, interval = 600s, no errors |
| Single memory | 1 memory | Scanner finds 0 pairs (need ≥2), no relationships |
| Duplicate content | 2 identical memories | Scanner may find pair but LLM should detect no causal link |
| Very long content | Memory with 50,000 chars | LLM truncates to `max_content_length` (default 4000 chars), no crash |
| LLM not loaded | `is_loaded() == false` | `run_discovery_cycle` calls `load()` first, then proceeds |
| Store write failure | Simulate by dropping DB | Error logged, consecutive_errors incremented, no panic |

---

## 10. Full State Verification Protocol

### 10.1 Source of Truth

The **Source of Truth** for causal relationships is:
- **RocksDB column family**: `CF_CAUSAL_RELATIONSHIPS` (in `crates/context-graph-storage/src/teleological/rocksdb_store/`)
- **HNSW index**: E5 causal HNSW index (cause vectors) and E5 causal HNSW index (effect vectors)
- **Audit log**: `CF_AUDIT_LOG` entries with `AuditOperation::RelationshipDiscovered`

### 10.2 Execute & Inspect Protocol

After each test:
1. **Execute**: Run the background loop for exactly 1 cycle
2. **Read-back**: Call `store.get_causal_relationship(id)` on the returned UUID
3. **Verify fields**: Check that `cause_embedding.len() == 768`, `effect_embedding.len() == 768`, `confidence >= min_confidence`, `cause_description` matches input
4. **Verify HNSW**: Call `store.search_causal_relationships(cause_embedding, 5, None)` and confirm the stored relationship appears in results
5. **Verify audit**: Call `store.get_audit_trail(Uuid::nil(), time_range)` and confirm an `AuditOperation::RelationshipDiscovered` entry exists

### 10.3 Boundary & Edge Case Audit

For each of these 3 edge cases, print system state BEFORE and AFTER:

**Edge Case 1: Empty database**
```
BEFORE: scan_fingerprints_for_clustering(Some(100)) → []
ACTION: run_background_tick()
AFTER:  cursor.cycles_completed == 1, cursor.total_relationships == 0
VERIFY: No new entries in CF_CAUSAL_RELATIONSHIPS
```

**Edge Case 2: Maximum batch size**
```
BEFORE: Store 200 memories, scan_fingerprints_for_clustering(Some(100)) → [100 items]
ACTION: run_background_tick()
AFTER:  cursor.total_relationships >= 0 (depends on content)
VERIFY: scanner processed at most batch_size pairs
```

**Edge Case 3: Invalid content (empty strings)**
```
BEFORE: Store memory with empty content ""
ACTION: run_background_tick() → harvest_memories() skips empty content
AFTER:  Memory skipped, no CausalRelationship created, no error
VERIFY: Warning logged for skipped memory
```

### 10.4 Evidence of Success

After all tests pass, produce a log showing:
```
=== CAUSAL DISCOVERY BACKGROUND LOOP — VERIFICATION LOG ===

1. Stored 3 test memories (A, B, E)
2. Started background loop
3. Cycle 1 completed in X.XXs
4. Discovered Y relationships
5. READ-BACK VERIFICATION:
   - Relationship 1: cause="Deploying version 2.3..." effect="500% spike in HTTP 401..."
     cause_embedding: [0.123, -0.456, ...] (768 dims, L2 norm=X.XX)
     effect_embedding: [0.789, -0.012, ...] (768 dims, L2 norm=X.XX)
     confidence: 0.XX
     mechanism: "session token format change caused authentication failures"
     STORED IN ROCKSDB: ✓ (read back matches)
     FOUND IN HNSW SEARCH: ✓ (rank 1, similarity 0.XX)
     AUDIT RECORD EXISTS: ✓ (operation=RelationshipDiscovered, timestamp=...)

6. Cursor state: cycles_completed=1, total_relationships=Y
7. CURSOR PERSISTED: ✓ (read back from store matches)

8. Rejected pair (C, D): No CausalRelationship found — ✓
9. Edge case (empty DB): 0 relationships, no errors — ✓
10. Edge case (empty content): Skipped with warning — ✓

ALL VERIFICATIONS PASSED
```

### 10.5 Trigger → Process → Outcome Chain

For every operation, trace the full chain:

```
TRIGGER: Timer fires (tokio::time::sleep expires)
  → PROCESS X: run_background_tick() called
    → harvest_memories() fetches from CF_FINGERPRINTS via scan_fingerprints_for_clustering()
    → scanner.find_candidates() clusters by E1 cosine similarity
    → llm.analyze_causal_relationship() sends prompt to Hermes 2 Pro via llama-cpp-2
    → activator.activate_relationship() generates E5 dual embeddings via CausalModel
    → store.store_causal_relationship() writes to CF_CAUSAL_RELATIONSHIPS
    → store.append_audit_record() writes to CF_AUDIT_LOG
  → OUTCOME Y: CausalRelationship exists in RocksDB with non-zero embeddings
    → VERIFY Y: Read back from CF_CAUSAL_RELATIONSHIPS, search via HNSW, check audit log
```

**YOU MUST MANUALLY VERIFY OUTCOME Y EXISTS** — Do not rely on return values alone. Read back from the database. Search the HNSW index. Check the audit log. If the output should exist somewhere, go look for it.

---

## 11. Risk Analysis & Plan Options

### 11.1 Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Graph builder and discovery loop collide on GPU | Medium | Stagger by 30s offset (Option B). Graph builder runs at t=0,60,120s; discovery at t=30,90,150s. |
| LLM inference fails (OOM, timeout) | Medium | Consecutive error counter with exponential backoff. After 3 failures, pause for 1 hour. |
| O(n²) scanning grows too large | Medium | `batch_size` cap. At 100 memories, 4,950 pairs max. Scanner prunes by similarity threshold. |
| False positive relationships pollute graph | Medium | `min_confidence: 0.7` threshold. Tag all background relationships with source for easy filtering. |
| Cursor data corrupted | Low | JSON serialization. On parse failure, log warning and start fresh. |
| Store trait changes break other implementations | Low | Only adding 2 new methods with default impls that error. Existing code unaffected. |
| LLM model files missing | High | `CausalDiscoveryLLM::load()` fails fast with descriptive error. Server does not start. |

### 11.2 E7 Memory Estimate Bug

The codebase `MEMORY_ESTIMATES` array in `traits/model_factory/memory.rs` lists E7 at 550 MB FP32. For 1.5B parameters, the real value is ~6 GB FP32 / ~3 GB FP16. This should be corrected but is OUT OF SCOPE for this task. Note it in a code comment if you encounter it.

---

## 12. Implementation Checklist

Complete these in order:

- [ ] 1. Add `store_processing_cursor` and `get_processing_cursor` to `TeleologicalMemoryStore` trait
- [ ] 2. Add default impls in `defaults.rs` that return `CoreError::NotImplemented`
- [ ] 3. Implement in RocksDB store (`crud.rs`) — store as raw bytes in a system metadata key
- [ ] 4. Add `DiscoveryCursor` struct to `service/mod.rs`
- [ ] 5. Add new config fields to `CausalDiscoveryConfig` (`enable_background`, `min_interval`, etc.)
- [ ] 6. Add `CycleMetrics` struct
- [ ] 7. Implement `harvest_memories()` method
- [ ] 8. Implement `compute_next_interval()` method
- [ ] 9. Implement `save_cursor()` and `load_cursor()` methods
- [ ] 10. Modify `start()` to accept `store: Arc<dyn TeleologicalMemoryStore>`
- [ ] 11. Replace placeholder `warn!()` with real `run_background_tick()` call
- [ ] 12. Implement `run_background_tick()` method
- [ ] 13. Modify `process_candidate()` or `run_discovery_cycle()` to persist `CausalRelationship` to store
- [ ] 14. Wire `CausalDiscoveryService` creation in `server.rs`
- [ ] 15. Add environment variable parsing for config overrides
- [ ] 16. Write test `test_background_loop_discovers_causal_pair` with synthetic data
- [ ] 17. Write test `test_background_loop_rejects_non_causal`
- [ ] 18. Write test `test_cursor_persists_across_restart`
- [ ] 19. Write test `test_adaptive_interval_speeds_up`
- [ ] 20. Write test `test_stop_graceful_shutdown`
- [ ] 21. Write edge case tests (empty DB, max batch, empty content)
- [ ] 22. Run Full State Verification Protocol — read back from RocksDB, search HNSW, check audit
- [ ] 23. Verify `cargo build --release` succeeds with zero warnings
- [ ] 24. Verify all existing tests still pass (`cargo test --workspace`)
- [ ] 25. Run manual verification with synthetic data and produce Evidence of Success log

---

## Sources

- [ONNX Runtime Thread Safety](https://github.com/microsoft/onnxruntime/discussions/10107) — Sessions are thread-safe for concurrent Run()
- [llama.cpp Continuous Batching](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) — Flat throughput scaling
- [llama.cpp FP4 Discussion](https://github.com/ggml-org/llama.cpp/discussions/11517) — "FP4 hardware currently not used at any point"
- [CUDA Green Contexts Docs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html) — No forward progress guarantees
- [RocksDB Iterator Patterns](https://github.com/facebook/rocksdb/wiki/Iterator) — Seek/Next for cursor-based scanning
- [Amortized Causal Discovery (TRACE)](https://www.arxiv.org/pdf/2602.01135) — Incremental discovery without retraining
- [CausalFusion Algorithm](https://www.nature.com/articles/s41598-025-34507-0) — Adaptive weight learning for KG integration
- [Production KG ROI](https://medium.com/@claudiubranzan/from-llms-to-knowledge-graphs-building-production-ready-graph-systems-in-2025-2b4aff1ec99a) — 300-320% ROI proven
