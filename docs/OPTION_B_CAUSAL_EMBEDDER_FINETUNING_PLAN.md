# Option B: Fine-Tuned Causal Embedder — Implementation Task

**Updated**: 2026-02-10
**Branch**: casetrack
**Crate**: context-graph-embeddings (workspace member)
**Language**: Rust 2021, MSRV 1.75
**GPU Framework**: Candle 0.9.2-alpha (HuggingFace) + CUDA
**Constitution**: `docs2/constitution.yaml` v7.0.0

---

## 0. Executive Summary

Replace the E5 causal embedder's base model (`allenai/longformer-base-4096`) with a contrastively fine-tuned model (`nomic-ai/nomic-embed-text-v1.5`). The current Longformer produces anisotropic 768D vectors — all content scores 0.93-0.98 cosine regardless of topic, giving E5 zero ranking discrimination. The fine-tuned model must achieve >0.10 spread per query (vs current 0.026) and >=4/6 standalone top-1 accuracy (vs current 0/6).

**ABSOLUTELY NO BACKWARDS COMPATIBILITY.** The system must work after changes or fail fast with robust error logging. No workarounds, no fallbacks, no mock data in tests.

---

## 1. Current State of E5 (Source of Truth)

### 1.1 Proven Problem (Benchmark Data)

| Metric | E1 (Semantic, 1024D) | E5 (Causal, 768D) |
|--------|---------------------|-------------------|
| Score range per query | 0.79 - 0.91 | 0.93 - 0.98 (compressed) |
| Spread (top-1 minus rank-5) | 0.051 avg | 0.026 avg |
| Standalone top-1 accuracy | 6/6 | 0/6 |
| Discrimination ratio vs E1 | 1.0x (baseline) | 0.51x |
| Ablation delta when removed | N/A | 0.0% (removing E5 changes ZERO rankings) |

Source: `docs/ACCURACY_BENCHMARK_REPORT.md` — 7-phase benchmark, 24+ queries, 2026-02-09.

### 1.2 Root Cause: Anisotropy

`allenai/longformer-base-4096` was pre-trained for MLM only, never for sentence similarity. Its pooled representations occupy a narrow cone where random sentence pairs average 0.6-0.9 cosine instead of ~0.0 (Ethayarajh, EMNLP 2019). The `CausalProjectionWeights` (perturbed identity matrices) apply linear transformations to already-anisotropic vectors — a linear projection of a narrow cone is still a narrow cone.

### 1.3 Current Mitigations (Band-Aids, Not Fixes)

These were applied as interim fixes (commit `5308ad1`):
- E5 weight demoted from 0.45 to 0.10 in `causal_reasoning` profile
- E1 promoted to 0.40 as primary signal
- Direction modifiers (1.2x cause→effect, 0.8x effect→cause) work mathematically but don't change rankings because E5 spread is too narrow

These mitigations will be **removed** when the fine-tuned model is integrated, since a discriminative E5 makes them unnecessary.

### 1.4 Current E5 File Inventory (VERIFIED)

All files under `crates/context-graph-embeddings/src/models/pretrained/causal/`:

| File | Purpose | Lines |
|------|---------|-------|
| `mod.rs` | Module exports, public API | ~40 |
| `model.rs` | `CausalModel` struct, `embed_dual()`, `embed_as_cause()`, `embed_as_effect()`, `EmbeddingModel` trait impl | 482 |
| `weights.rs` | `LongformerWeights`, `CausalProjectionWeights`, `TrainableProjection`, `CAUSAL_PROJECTION_SEED` | ~576 |
| `config.rs` | `LongformerConfig`, `DEFAULT_ATTENTION_WINDOW=512`, `CAUSAL_MAX_TOKENS=4096`, `CAUSAL_DIMENSION=768` | ~53 |
| `loader.rs` | `load_longformer_weights()` from safetensors + config.json | ~114 |
| `embeddings_loader.rs` | Embedding layer weight loading (prefix `longformer.embeddings`) | ~80 |
| `encoder_loader.rs` | Encoder layer loading (prefix `longformer.encoder.layer.N`) | ~120 |
| `marker_detection.rs` | `detect_causal_markers_with_hints()`, 80+ causal patterns, `MARKER_BOOST=2.5` | ~783 |
| `forward/mod.rs` | `gpu_forward()`, `gpu_forward_dual()` — the inference pipeline | ~414 |
| `forward/ops.rs` | `layer_norm()`, `mean_pooling()`, `marker_weighted_pooling()`, `l2_normalize()` | ~200 |
| `forward/attention.rs` | Multi-head self-attention (standard, not sliding window despite Longformer name) | ~150 |
| `forward/encoder.rs` | Encoder layer loop | ~80 |
| `tests.rs` | Unit tests | ~100 |

### 1.5 Current Inference Pipeline (What `embed_dual()` Does)

```
Input text
  → Tokenize (HF tokenizer, RoBERTa vocab, max 4096 tokens)
  → Detect causal markers (80+ patterns, optional LLM hints)
  → Create GPU tensors (input_ids, attention_mask, position_ids, token_type_ids)
  → Compute embeddings (word + position + token_type, LayerNorm)
  → Run encoder (12 Longformer layers, standard attention NOT sliding-window)
  → Marker-weighted pooling × 2 (cause weights: boost "because"/"due to" 2.5x; effect weights: boost "therefore"/"results" 2.5x)
  → Apply W_cause projection → cause_vec
  → Apply W_effect projection → effect_vec
  → L2 normalize both
  → Return (cause_vec: Vec<f32>[768], effect_vec: Vec<f32>[768])
```

### 1.6 Key Types and Signatures

```rust
// Model state
pub(crate) enum ModelState {
    Unloaded,
    Loaded { weights: LongformerWeights, projection: CausalProjectionWeights, tokenizer: Box<Tokenizer> },
}

// Struct
pub struct CausalModel { model_state: RwLock<ModelState>, model_path: PathBuf, ... }

// Public API (KEEP THESE SIGNATURES UNCHANGED)
impl CausalModel {
    pub fn new(model_path: &Path, config: SingleModelConfig) -> EmbeddingResult<Self>
    pub async fn load(&self) -> EmbeddingResult<()>
    pub async fn embed_dual(&self, content: &str) -> EmbeddingResult<(Vec<f32>, Vec<f32>)>
    pub async fn embed_as_cause(&self, content: &str) -> EmbeddingResult<Vec<f32>>
    pub async fn embed_as_effect(&self, content: &str) -> EmbeddingResult<Vec<f32>>
}

// Forward functions
pub fn gpu_forward(text: &str, weights: &LongformerWeights, tokenizer: &Tokenizer) -> EmbeddingResult<Vec<f32>>
pub fn gpu_forward_dual(text, weights, projection, tokenizer, guidance) -> EmbeddingResult<(Vec<f32>, Vec<f32>)>
```

### 1.7 Model Identity References

| Location | Current Value | Must Change To |
|----------|--------------|----------------|
| `types/model_id/repository.rs:17` | `"allenai/longformer-base-4096"` | `"nomic-ai/nomic-embed-text-v1.5"` |
| `types/model_id/core.rs:48` | `/// E5: Causal embedding using allenai/longformer-base-4096 (768D)` | Update doc comment |
| `types/model_id/repository.rs:41` | `Self::Causal => "causal"` (directory name) | Keep as `"causal"` |

### 1.8 Weight Profile (causal_reasoning)

File: `crates/context-graph-core/src/weights/mod.rs` lines 143-160

Current: `[0.40, 0.0, 0.0, 0.0, 0.10, 0.05, 0.15, 0.10, 0.0, 0.10, 0.10, 0.0, 0.0]`

After fine-tuning success, E5 (index 4) should increase to 0.25-0.35 and E1 (index 0) decrease proportionally.

### 1.9 Existing Training Infrastructure

Files under `crates/context-graph-embeddings/src/training/`:

| File | Contents | Reusable? |
|------|----------|-----------|
| `data.rs` | `CausalTrainingPair`, `TrainingBatch`, `TrainingDirection` | YES — adapt for CausalBank |
| `loss.rs` | `DirectionalContrastiveLoss` with InfoNCE + directional margin + separation + soft-label | YES — core of training |
| `trainer.rs` | Training loop with momentum encoder | YES — adapt for new model |
| `optimizer.rs` | AdamW with warmup + cosine decay | YES |
| `evaluation.rs` | Directional accuracy, MRR, AUC | YES — add spread/anisotropy |
| `distillation.rs` | LLM→embedder online teaching | PARTIAL |
| `lora.rs` | LoRA adapters for Longformer | REPLACE — new architecture |
| `multitask.rs` | Direction classification heads | PARTIAL |

### 1.10 Asymmetric Search Integration

File: `crates/context-graph-core/src/causal/asymmetric.rs`

- `detect_causal_query_intent(query) -> CausalDirection` — 100+ keyword patterns, score-based classification
- Direction modifiers: `CAUSE_TO_EFFECT=1.2`, `EFFECT_TO_CAUSE=0.8`
- Used by `search_causes`, `search_effects` MCP tools
- This code is INDEPENDENT of the model — it only applies direction modifiers to E5 scores at search time

---

## 2. What Must Change (Scope)

### 2.1 REPLACE (Longformer → Nomic-embed)

| Component | Remove | Add |
|-----------|--------|-----|
| Base model | `allenai/longformer-base-4096` (149M params, MLM-only, anisotropic) | `nomic-ai/nomic-embed-text-v1.5` (137M params, contrastive pre-trained, isotropic) |
| Weight struct | `LongformerWeights` + `LongformerEmbeddingWeights` + `LongformerAttentionWeights` + `LongformerFfnWeights` + `LongformerEncoderLayerWeights` | `NomicWeights` (BERT-like with rotary embeddings) |
| Config struct | `LongformerConfig` | `NomicConfig` (from config.json) |
| Loader | `load_longformer_weights()`, `embeddings_loader.rs`, `encoder_loader.rs` | `load_nomic_weights()` (single safetensors loader) |
| Forward pass | `forward/attention.rs` (standard attention), `forward/encoder.rs` | New attention with rotary position embeddings |
| Projection | `CausalProjectionWeights` (perturbed identity) | **REMOVE** — asymmetry from instruction prefixes |
| Marker detection | `marker_detection.rs` (80+ patterns, weighted pooling) | **REMOVE** — fine-tuned model handles internally |
| Pooling | `marker_weighted_pooling()` | Standard `mean_pooling()` (model already fine-tuned for it) |

### 2.2 KEEP (Unchanged)

- `CausalModel` public API signatures (`embed_dual`, `embed_as_cause`, `embed_as_effect`)
- `ModelId::Causal = 4` enum variant and 768D dimension
- `detect_causal_query_intent()` in `asymmetric.rs` (search-time, model-independent)
- Direction modifiers (1.2x/0.8x) in `asymmetric.rs`
- All MCP tools (`search_causes`, `search_effects`, `search_causal_relationships`, etc.)
- Column family `emb_4` (768D quantized E5 vectors in RocksDB)
- `TrainableProjection` struct (still useful for future fine-tuning of projection heads)

### 2.3 New Asymmetric Encoding (Instruction Prefixes)

Instead of `marker_weighted_pooling()` + `CausalProjectionWeights`, use instruction prefixes:

```rust
// BEFORE: Single encoder pass + two projection matrices
let (cause_pooled, effect_pooled) = marker_weighted_pooling(hidden, mask, cause_wts, effect_wts);
let cause_vec = projection.project_cause(&cause_pooled)?;
let effect_vec = projection.project_effect(&effect_pooled)?;

// AFTER: Two encoder passes with different instruction prefixes
let cause_input = format!("search_query: Identify the cause in: {}", content);
let effect_input = format!("search_query: Identify the effect of: {}", content);
let cause_vec = gpu_forward(&cause_input, weights, tokenizer)?;
let effect_vec = gpu_forward(&effect_input, weights, tokenizer)?;
```

This produces genuinely different representations because the model attends differently based on the instruction prefix. nomic-embed-text-v1.5 natively supports `search_query:` / `search_document:` prefixes.

---

## 3. Model Selection: Options Analysis

### Option 1: nomic-ai/nomic-embed-text-v1.5 (RECOMMENDED)

| Property | Value |
|----------|-------|
| Params | 137M |
| Output dim | 768 (matches E5 exactly) |
| Architecture | BERT-like + rotary embeddings |
| Anisotropy | LOW (contrastive pre-training) |
| Fine-tune support | Excellent (`contrastors` library, 149+ community fine-tunes) |
| GGUF support | Native (nomic-ai provides GGUF) |
| Instruction prefixes | `search_query:` / `search_document:` |
| Matryoshka dims | 768, 512, 256 |
| VRAM | ~500MB FP16 |

**Pros**: Already isotropic, 768D native, rotary embeddings for better extrapolation, proven fine-tuning ecosystem, instruction-following capability for asymmetric cause/effect encoding.

**Cons**: Requires custom candle forward pass (rotary PE differs from absolute PE). No sliding-window attention (not needed — causal sentences are 10-50 tokens).

### Option 2: nomic-ai/nomic-embed-text-v2-moe (ALTERNATIVE)

| Property | Value |
|----------|-------|
| Params | 475M total, 305M active (MoE 8x2) |
| Output dim | 768 (Matryoshka) |
| Multilingual | ~100 languages |
| GGUF | Available |
| VRAM | ~1.2GB (more than v1.5) |

**Pros**: Newer, MoE architecture routes different inputs to specialized experts. Multilingual.

**Cons**: MoE routing adds implementation complexity in candle. 2x VRAM. MoE not supported in standard candle BERT forward pass — would require custom expert routing. Overkill for English-only causal embedding.

**Verdict**: v1.5 for this task. v2-moe is a future upgrade path.

### Option 3: mxbai-embed-large-v1 (FALLBACK)

| Property | Value |
|----------|-------|
| Params | 335M |
| Output dim | 1024 (truncate to 768 via Matryoshka) |
| VRAM | ~1.3GB |

**Pros**: Higher capacity. **Cons**: Dimension mismatch requires Matryoshka truncation. 2.5x larger. Fewer fine-tuning examples.

---

## 4. Dataset Construction

### 4.1 Primary: CausalBank (314M pairs)

Source: Li et al., "Guided Generation of Cause and Effect", IJCAI 2020
- GitHub: https://github.com/eecrazy/CausalBank
- Official: https://nlp.jhu.edu/causalbank/
- 133M effect→cause + 181M cause→effect pairs from Common Crawl
- CSV format, ~15GB compressed

**Preprocessing Pipeline**:
1. Filter: both sentences 10-200 tokens
2. Deduplicate by normalized text (~40% reduction expected)
3. Cluster by domain using existing E1 embeddings (top-20 domains)
4. Balanced sample: 500K pairs/domain = 10M training pairs
5. Hold out 50K for evaluation (stratified by domain)

### 4.2 Evaluation: SemEval-2010 Task 8

- 10,717 annotated sentences with 9 relation types including Cause-Effect
- Available: HuggingFace `SemEvalWorkshop/sem_eval_2010_task_8`, Kaggle
- Use for evaluation ONLY (too small for training)
- Filter to 1,461 Cause-Effect pairs

### 4.3 Hard Negatives (3 Types)

**Type 1 — Same domain, different relationship**: "Smoking causes lung cancer" vs "Lung cancer treatment includes chemotherapy"
**Type 2 — Reversed causation**: "Deforestation causes flooding" vs "Flooding causes deforestation"
**Type 3 — Correlation, not causation**: "Ice cream sales increase drowning deaths" (confound)

Mine with `sentence_transformers.mine_hard_negatives()` using E1 as base retriever. Top-20 nearest neighbors NOT in positive set → "semantically close but causally unrelated".

### 4.4 Dataset Splits

| Split | Size | Source | Purpose |
|-------|------|--------|---------|
| Train | 10M triplets | CausalBank + mined negatives | Contrastive learning |
| Hard negatives | 2M triplets | Synthetic Types 1-3 | Direction discrimination |
| Validation | 50K triplets | CausalBank holdout | Loss monitoring |
| Eval-SemEval | 1,461 pairs | SemEval-2010 | Benchmark |
| Eval-internal | 100 pairs | Synthetic (known ground truth) | Integration testing |

---

## 5. Training Pipeline

### 5.1 Three-Stage Progressive Training

**Stage 1: Domain Adaptation (1 epoch on 10M pairs)**
- Loss: MNR only (in-batch negatives)
- Goal: Adapt embedding space to causal language
- Expected: Anisotropy drops from ~0.75 to ~0.40

**Stage 2: Hard Negative Mining (2 epochs)**
- Use Stage 1 model to mine hard negatives (rank 50-200)
- Loss: MNR (0.7) + Directional Triplet (0.3)
- Expected: Spread increases to ~0.06

**Stage 3: Direction Fine-Tuning (1 epoch on 2M directional triplets)**
- Loss: MNR (0.5) + Directional Triplet (0.4) + Soft-ZCA Whitening (0.1)
- Goal: Asymmetric cause/effect encoding
- Expected: Spread > 0.10, direction accuracy > 80%

### 5.2 Training Config

```yaml
base_model: nomic-ai/nomic-embed-text-v1.5
max_seq_length: 256
effective_batch_size: 2048  # via CachedMNRL
mini_batch_size: 64
learning_rate: 2e-5
warmup_ratio: 0.1
epochs: 3 (across stages)
weight_decay: 0.01
fp16: true
gradient_accumulation: 4
early_stopping: 3 eval steps without spread improvement
```

### 5.3 Anti-Anisotropy: Soft-ZCA Whitening

Recent research (ESANN 2025) confirms Soft-ZCA whitening improves embedding isotropy:
- Compute batch covariance matrix
- Apply: `z = α * Σ^(-1/2) @ (x - μ) + (1-α) * x` with α=0.1
- Regularization weight: 0.01-0.1
- Moderate whitening (ε ∈ {0.1, 0.01}) works best for base models

### 5.4 Existing Loss Functions (REUSE)

The codebase already has `DirectionalContrastiveLoss` in `training/loss.rs` with:
- `info_nce_loss()` — InfoNCE contrastive (τ=0.05)
- `directional_margin_loss()` — Forward > Reverse (margin=0.2)
- `separation_loss()` — cause_vec ≠ effect_vec for same text
- `soft_label_loss()` — LLM confidence distillation

Default weights: `lambda_contrastive=1.0, lambda_directional=0.3, lambda_separation=0.1, lambda_soft=0.2`

These are implemented in candle tensors and run on GPU. They are directly reusable.

---

## 6. Integration: Code Changes Required

### 6.1 Files to MODIFY

**File 1: `crates/context-graph-embeddings/src/models/pretrained/causal/model.rs`**
- Replace `ModelState::Loaded` to hold `NomicWeights` instead of `LongformerWeights`
- Remove `CausalProjectionWeights` from state
- Update `embed_dual()` to use two `gpu_forward()` calls with instruction prefixes
- Update `load()` to call `load_nomic_weights()` instead of `load_longformer_weights()`
- Keep ALL public API signatures unchanged
- Instruction prefixes already defined as constants: `CAUSE_INSTRUCTION`, `EFFECT_INSTRUCTION` — update text to use `search_query:` prefix

**File 2: `crates/context-graph-embeddings/src/models/pretrained/causal/weights.rs`**
- Replace `LongformerWeights`, `LongformerEmbeddingWeights`, `LongformerAttentionWeights`, `LongformerFfnWeights`, `LongformerEncoderLayerWeights` with `NomicWeights` (BERT-like with rotary position embeddings)
- Keep `CausalProjectionWeights` and `TrainableProjection` (useful for optional projection head training later)
- Keep `CAUSAL_PROJECTION_SEED`

**File 3: `crates/context-graph-embeddings/src/models/pretrained/causal/config.rs`**
- Replace `LongformerConfig` with `NomicConfig`
- Keep `CAUSAL_DIMENSION = 768` (unchanged)
- Update `CAUSAL_MAX_TOKENS` from 4096 to 512 (nomic-embed supports 8192 but causal sentences are short)
- Remove `DEFAULT_ATTENTION_WINDOW` (no sliding window)

**File 4: `crates/context-graph-embeddings/src/models/pretrained/causal/loader.rs`**
- Replace `load_longformer_weights()` with `load_nomic_weights()`
- Parse nomic config.json format (rotary_emb_base, rotary_emb_fraction, etc.)
- Load from safetensors (same format, different tensor names)

**File 5: `crates/context-graph-embeddings/src/models/pretrained/causal/forward/mod.rs`**
- Update `gpu_forward()` to use new weight structs
- Update `gpu_forward_dual()`:
  - Remove `projection` parameter
  - Remove `marker_detection` calls
  - Instead: prepend cause/effect instruction prefixes, call `gpu_forward()` twice
- Remove marker-related imports

**File 6: `crates/context-graph-embeddings/src/models/pretrained/causal/forward/attention.rs`**
- Replace standard multi-head attention with rotary position embedding attention
- nomic-embed uses rotary embeddings (RoPE), not absolute position embeddings

**File 7: `crates/context-graph-embeddings/src/models/pretrained/causal/forward/encoder.rs`**
- Update to match nomic encoder layer structure

**File 8: `crates/context-graph-embeddings/src/types/model_id/repository.rs`**
- Line 17: Change `"allenai/longformer-base-4096"` to `"nomic-ai/nomic-embed-text-v1.5"`

**File 9: `crates/context-graph-embeddings/src/types/model_id/core.rs`**
- Line 48: Update doc comment

### 6.2 Files to DELETE

| File | Reason |
|------|--------|
| `embeddings_loader.rs` | Longformer-specific embedding layer loader |
| `encoder_loader.rs` | Longformer-specific encoder loader |
| `marker_detection.rs` | Replaced by instruction prefix approach |
| `forward/ops.rs:marker_weighted_pooling()` | Remove function only; keep `layer_norm`, `mean_pooling`, `l2_normalize` |

### 6.3 Files to ADD

| File | Contents |
|------|----------|
| `nomic_loader.rs` | Load nomic-embed weights from safetensors, parse config |

### 6.4 Weight File Changes

Model weights directory: `models/causal/` (same path, different contents)

**Remove**: `model.safetensors` (Longformer), `config.json` (Longformer), `tokenizer.json` (RoBERTa)
**Add**: `model.safetensors` (fine-tuned nomic-embed), `config.json` (nomic), `tokenizer.json` (nomic/SentencePiece)

### 6.5 After Training: Update Weight Profile

File: `crates/context-graph-core/src/weights/mod.rs` line 144-160

Change E5 from 0.10 back to 0.25-0.35 once benchmarks pass. E1 correspondingly from 0.40 to ~0.30.

---

## 7. Migration Path

1. Train model externally (Python, Sentence Transformers) → export as safetensors
2. Place weights in `models/causal/` directory (overwrite Longformer files)
3. Implement code changes (Section 6)
4. `cargo build --release` — MUST compile with zero errors
5. **Delete existing RocksDB database** — ALL E5 vectors are invalidated
6. Re-store memories via `store_memory` MCP tool (re-embeds with new model)
7. Run full verification suite (Section 8)

**Breaking change**: All stored E5 vectors become invalid. No migration path — must re-embed. This is acceptable: current E5 vectors provide zero ranking value.

---

## 8. Full State Verification (MANDATORY)

After completing ANY logic change, you MUST perform Full State Verification. Do not rely on return values alone.

### 8.1 Define Source of Truth

| What | Source of Truth | How to Inspect |
|------|----------------|----------------|
| E5 vectors | RocksDB CF `emb_4` (quantized) and CF `fingerprints` (full 768D) | `get_memory_fingerprint` MCP tool |
| Cause/effect dual vectors | Fingerprint's E5 slot contains both cause and effect variants | `get_memory_fingerprint` returns `e5_causal_cause_vec` and `e5_causal_effect_vec` |
| Model loaded | `CausalModel.is_initialized()` returns true | Log output on startup |
| Weight profile | `crates/context-graph-core/src/weights/mod.rs` | `create_weight_profile` MCP tool to read |
| Search results | RRF fusion output from `search_graph` | `includeEmbedderBreakdown=true` parameter |

### 8.2 Execute & Inspect Protocol

For every change, run this sequence:

1. **Build**: `cargo build --release` — zero errors, zero warnings in changed files
2. **Unit tests**: `cargo test -p context-graph-embeddings` — all pass
3. **MCP tests**: `cargo test -p context-graph-mcp` — all 653+ pass
4. **Store synthetic memories**: Use `store_memory` to store 5 memories (3 causal + 2 neutral):
   - "Smoking causes lung cancer through DNA damage"
   - "Deforestation leads to soil erosion and flooding"
   - "Increased CO2 causes global temperature rise"
   - "The Rust programming language was created in 2010" (neutral)
   - "Photosynthesis converts CO2 into oxygen" (neutral)
5. **Verify physical vectors**: Call `get_memory_fingerprint` for each stored memory. Confirm:
   - E5 vectors exist (non-null, 768D)
   - Cause and effect variants are different (cosine < 0.95)
   - Vectors are NOT degenerate (not all 0.93-0.98 cosine with each other)
6. **Search happy path**:
   - Query: "What causes lung cancer?" → top-1 should be smoking memory
   - Query: "What are the effects of deforestation?" → top-1 should be deforestation memory
   - Query: "What programming language was created in 2010?" → should NOT return causal memories as top-1
7. **Verify in database**: Read the RocksDB CF `emb_4` entry for each stored memory. Confirm the quantized vector exists and has 768 dimensions.

### 8.3 Boundary & Edge Case Audit

For each edge case, print system state BEFORE and AFTER:

**Edge Case 1: Empty input**
- Input: `embed_dual("")`
- Expected: `EmbeddingError::TokenizationError` or graceful empty handling
- Verify: No crash, no corrupt state, error logged with model name

**Edge Case 2: Maximum length input**
- Input: `embed_dual("word ".repeat(512))` (fills context window)
- Expected: Truncation to `CAUSAL_MAX_TOKENS`, valid 768D output
- Verify: Output dimensions are exactly 768, no NaN/Inf values

**Edge Case 3: Non-causal input**
- Input: `embed_dual("The quick brown fox jumps over the lazy dog")`
- Expected: Valid vectors, low causal signal
- Verify: cause_vec and effect_vec are still different (instruction prefix effect)

**Edge Case 4: Unicode and special characters**
- Input: `embed_dual("CO₂ → temperature ↑ → ice ↓")`
- Expected: Valid embedding (tokenizer handles unicode)
- Verify: 768D output, no tokenization error

**Edge Case 5: Identical cause/effect text**
- Input: store two memories with same text, query both
- Expected: Same E5 scores for identical content
- Verify: scores are identical (floating point tolerance 1e-6)

### 8.4 Evidence of Success

Provide a log showing:
1. `cargo build --release` exit code 0
2. Test counts: X passed, 0 failed
3. For each of the 5 stored memories: UUID, E5 vector first 5 values, cosine between cause/effect variants
4. For each of the 3 search queries: top-3 results with E5 scores and overall RRF scores
5. For each edge case: before state, action, after state, error message (if any)
6. Anisotropy measurement: average cosine of 10 random pairs of stored memories' E5 vectors (must be < 0.30 for fine-tuned model)

---

## 9. Success Criteria

The fine-tuned model is ready for production when ALL of these are met:

| # | Criterion | Metric | Threshold | How to Measure |
|---|-----------|--------|-----------|----------------|
| 1 | Score discrimination | Avg spread per query | > 0.10 | Top-1 E5 score minus rank-5 E5 score across 10 queries |
| 2 | Standalone accuracy | Top-1 on 6 standard queries (E5-only search) | >= 4/6 | `search_graph` with E5-only custom weights |
| 3 | Ablation value | Accuracy delta when E5 removed | > 5% | Compare multi_space with/without E5 |
| 4 | Direction awareness | Correct direction preference | >= 80% | For 30 pairs: does cause-query prefer cause-embed? |
| 5 | Isotropy | Avg random pair cosine | < 0.30 | Embed 100 random sentences, compute pairwise cosine |
| 6 | Integration | `cargo build --release` | Zero errors | Build command |
| 7 | Regression | Existing MCP tests | All pass | `cargo test -p context-graph-mcp` |
| 8 | Full benchmark | 7-phase accuracy suite | >= 100% top-1 | `docs/ACCURACY_BENCHMARK_REPORT.md` protocol |

---

## 10. Testing Requirements

### 10.1 NO MOCK DATA

All tests MUST use:
- Real RocksDB (via `create_test_handlers()` which provides `(Handlers, TempDir)`)
- Real GPU embeddings (`ProductionMultiArrayProvider`)
- Real model inference (not stubbed)

Constitution rule: `testing.golden_rule: "ALL MCP tests use real RocksDB + real GPU embeddings. NO STUBS."`

### 10.2 New Tests to Add

```rust
#[tokio::test]
async fn test_e5_spread_minimum() {
    // Store 10 diverse memories, verify E5 spread > 0.10
    // FAIL FAST if spread below threshold
}

#[tokio::test]
async fn test_e5_cause_effect_asymmetry() {
    // embed_dual("Smoking causes cancer") → cause_vec ≠ effect_vec
    // cosine(cause_vec, effect_vec) must be < 0.95 AND > 0.50
}

#[tokio::test]
async fn test_e5_anisotropy_bound() {
    // Embed 100 random sentences, avg pairwise cosine < 0.30
}

#[tokio::test]
async fn test_e5_standalone_accuracy() {
    // 6 standard queries with E5-only weights
    // At least 4/6 correct top-1
}

#[tokio::test]
async fn test_e5_direction_discrimination() {
    // 30 causal pairs: cause-query should prefer cause-embed
    // >= 24/30 correct
}

#[tokio::test]
async fn test_e5_instruction_prefix_produces_different_vectors() {
    // Same text with cause vs effect prefix → different vectors
    // This validates the instruction-prefix asymmetry works
}
```

### 10.3 Existing Tests Must Pass

- `cargo test -p context-graph-mcp` (653+ tests)
- `cargo test -p context-graph-core` (132+ tests)
- `cargo test -p context-graph-embeddings`
- `cargo test -p context-graph-causal-agent` (65+ tests)

---

## 11. Error Handling Requirements

Per constitution: `error_handling.rule: "FAIL FAST — no silent degradation"`

Every error must:
1. Use `thiserror #[derive(Error)]` types
2. Include the model name (`CausalModel`) in error messages
3. Include the operation that failed (e.g., "tokenization", "forward pass", "weight loading")
4. Include dimensional expectations (e.g., "expected 768D, got 512D")
5. Propagate via `EmbeddingResult<T>` (not `.unwrap()` or `.expect()`)
6. Log at `tracing::error!` level for unrecoverable failures
7. Return JSON-RPC error codes -32001 to -32009 at MCP boundary

Example:
```rust
if vector.len() != 768 {
    return Err(EmbeddingError::InternalError {
        message: format!(
            "CausalModel embed_dual: E5 cause vector dimension error: got {}, expected 768. \
             Model: nomic-embed-text-v1.5, input_len: {} tokens",
            vector.len(), seq_len
        ),
    });
}
```

---

## 12. Timeline & Phases

### Phase 1: Data Preparation (1-2 weeks)
- Download CausalBank (15GB)
- Preprocess: filter, dedup, cluster by domain
- Mine hard negatives with E1
- Create eval sets

### Phase 2: Training (1-2 weeks)
- Stage 1: Domain adaptation (4-6 hours on A100)
- Stage 2: Hard negative training (8-12 hours)
- Stage 3: Direction fine-tuning (4-6 hours)
- Full evaluation after each stage

### Phase 3: Integration (1 week)
- Export trained model as safetensors
- Implement code changes (Section 6)
- Build and test

### Phase 4: Validation (3-5 days)
- Full State Verification (Section 8)
- Run all benchmarks
- Update weight profiles
- Final ablation

---

## 13. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| CausalBank quality insufficient | HIGH | Evaluate after Stage 1; pivot to NLI + STS data |
| nomic-embed rotary PE not supported in candle | HIGH | candle supports BERT variants; rotary PE may need custom impl. Fallback: use `candle-transformers::models::bert` with custom RoPE layer |
| Fine-tuned model loses general embedding quality | MEDIUM | Include 10% general STS data in training |
| VRAM increase from two encoder passes in embed_dual | MEDIUM | Profile; batch cause/effect passes. nomic is 137M params (~500MB) vs Longformer 149M (~600MB), so net smaller |
| Direction asymmetry doesn't emerge | MEDIUM | Increase directional triplet loss weight in Stage 3 |
| All stored E5 vectors invalidated | LOW | Acceptable — current vectors are valueless (0.0% ablation delta) |

---

## 14. References

1. Ethayarajh (2019). "How Contextual are Contextualized Word Representations?" EMNLP.
2. Li et al. (2020). "Guided Generation of Cause and Effect." IJCAI. https://github.com/eecrazy/CausalBank
3. Nussbaum et al. (2024). "nomic-embed: Training a Reproducible Long Context Text Embedder." https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
4. SemEval-2010 Task 8. https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8
5. Soft-ZCA Whitening (ESANN 2025). https://arxiv.org/html/2411.17538
6. Sentence Transformers CachedMNRL. https://sbert.net/docs/package_reference/sentence_transformer/losses.html
7. nomic-embed-text-v2-moe (future). https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe
8. Candle ML framework. https://github.com/huggingface/candle
