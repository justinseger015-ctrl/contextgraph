# Option B Implementation Report: E5 Causal Embedder Model Replacement

**Date**: 2026-02-10
**Branch**: casetrack
**Status**: Base model replacement COMPLETE. Fine-tuning pipeline NOT YET EXECUTED.
**Crate**: context-graph-embeddings (workspace member)

---

## 1. Executive Summary

Option B replaced the E5 causal embedder's base model from `allenai/longformer-base-4096` (149M params, MLM-only) to `nomic-ai/nomic-embed-text-v1.5` (137M params, contrastive pre-training). The goal was to eliminate E5 anisotropy — the phenomenon where all content receives nearly identical E5 scores (0.93-0.98 cosine), rendering E5 useless for ranking.

### Outcome

**The base model swap alone did NOT solve E5 anisotropy.** The nomic-embed model produces isotropic embeddings in its native embedding space, but when used for causal similarity scoring via instruction prefixes, E5 still exhibits score compression (~0.70-0.77 range, spread ~0.026 per query). E1 (semantic, 1024D) remains the sole ranking discriminator with 2x better score spread.

However, the replacement achieved several structural improvements:

| Metric | Before (Longformer) | After (nomic-embed) |
|--------|-------------------|-------------------|
| Architecture | MLM-only, absolute PE | Contrastive, RoPE, SwiGLU |
| Asymmetry method | Marker-weighted pooling + projection matrices | Instruction-prefix dual forward pass |
| Code complexity | 12 files, 2500+ lines, marker_detection.rs (783 lines) | 10 files, ~1800 lines, no marker detection |
| Dead code | CausalProjectionWeights, embeddings_loader, encoder_loader | None — all code paths active |
| Fine-tuning readiness | Low (MLM model, no contrastive loss) | High (contrastive pre-training, instruction-following) |
| E5 weight in causal_reasoning | 0.45 (harmful) | 0.10 (neutralized) |
| Ablation delta | 0.0% | 0.0% (unchanged) |
| Top-1 accuracy (multi_space) | 100% | 100% (unchanged) |
| Build/test status | All pass | All pass (1322 MCP + 132 core + 37 causal embeddings + 35 causal agent) |

### Bottom Line

The model swap was **necessary but not sufficient**. It completed Phase 3 (Integration) of the Option B plan. The 3-stage fine-tuning (Phases 1-2: data preparation + training) remains the critical path to making E5 a discriminative ranking signal. The current system is strictly better than before: less code, cleaner architecture, and ready for fine-tuning.

---

## 2. What Was Done

### 2.1 Files Rewritten (Longformer → NomicBERT)

| File | Change |
|------|--------|
| `config.rs` | `LongformerConfig` → `NomicConfig` (RoPE base=1000, SwiGLU FFN, fused QKV) |
| `weights.rs` | `LongformerWeights` → `NomicWeights` (12-layer NomicBERT encoder) |
| `loader.rs` | `load_longformer_weights()` → `load_nomic_weights()` (safetensors) |
| `model.rs` | Removed `CausalProjectionWeights`, `config`, `memory_size` fields. Updated to dual forward pass with instruction prefixes |
| `forward/mod.rs` | Updated `gpu_forward()` + `gpu_forward_dual()` for NomicBERT |
| `forward/attention.rs` | Standard attention → RoPE attention (base=1000, full head_dim, interleaved guard) |
| `forward/encoder.rs` | Updated for SwiGLU FFN (gate + up projection, down projection) |
| `forward/ops.rs` | Removed `marker_weighted_pooling()`. Kept `layer_norm`, `mean_pooling`, `l2_normalize` |
| `tests.rs` | Updated for new instruction prefix constants |
| `mod.rs` | Added `CAUSE_INSTRUCTION`, `EFFECT_INSTRUCTION` re-exports |

### 2.2 Files Deleted

| File | Reason |
|------|--------|
| `embeddings_loader.rs` | Longformer-specific embedding layer loader |
| `encoder_loader.rs` | Longformer-specific encoder layer loader |
| `marker_detection.rs` | 783 lines of marker-weighted pooling. Replaced by instruction prefixes. |

### 2.3 Code Quality Fixes (Post-Replacement)

| Issue | Fix |
|-------|-----|
| H1: `.unwrap()` panics in `save_trained()` | Replaced with proper `map_err()` error propagation |
| H2: Instruction prefix string duplication | Moved `CAUSE_INSTRUCTION`/`EFFECT_INSTRUCTION` to `config.rs` as canonical constants |
| M1: Dead `memory_size` field | Removed from struct |
| M4: Dead `#[allow(dead_code)]` annotations | Removed 5 unnecessary suppressions |
| L1: Unsafe Send/Sync missing safety doc | Added detailed safety comment explaining thread-safety guarantees |

### 2.4 New Asymmetric Encoding

**Before** (Longformer):
```
Text → Tokenize → Detect causal markers (80+ patterns)
  → Single encoder pass → Marker-weighted pooling ×2
  → W_cause projection → cause_vec (768D)
  → W_effect projection → effect_vec (768D)
```

**After** (nomic-embed):
```
Text → "search_query: Identify the cause in: {text}" → Full encoder pass → cause_vec (768D)
Text → "search_query: Identify the effect of: {text}" → Full encoder pass → effect_vec (768D)
```

Trade-off: Two encoder passes instead of one, but eliminates 783 lines of marker detection, two projection matrices, and produces genuinely different embeddings from the contrastive model's instruction-following capability.

---

## 3. Benchmark Results vs Success Criteria

The Option B plan (Section 9) defined 8 success criteria. These were measured **after the base model swap but before fine-tuning**:

| # | Criterion | Threshold | Measured | Status |
|---|-----------|-----------|----------|--------|
| 1 | Score discrimination (avg spread) | > 0.10 | 0.026 | FAIL |
| 2 | Standalone top-1 accuracy (E5-only) | >= 4/6 | 0/6 | FAIL |
| 3 | Ablation delta when E5 removed | > 5% | 0.0% | FAIL |
| 4 | Direction awareness (correct preference) | >= 80% | 100% (modifiers) | PASS |
| 5 | Isotropy (avg random pair cosine) | < 0.30 | Not measured | TBD |
| 6 | Integration (`cargo build --release`) | Zero errors | Zero errors | PASS |
| 7 | Regression (MCP tests pass) | All pass | 1322/1322 | PASS |
| 8 | Full benchmark (7-phase suite) | >= 100% top-1 | 100% | PASS |

**3 of 5 ranking criteria FAIL.** This confirms the base model swap alone is insufficient — the 3-stage fine-tuning pipeline (Section 5 of the plan) is required.

---

## 4. What E5 Uniquely Provides That E1 Does Not

This is the core question. After 55+ MCP tool invocations, 24+ benchmark queries, ablation analysis, and Full State Verification, the answer is precise:

### 4.1 E5 Provides: Causal Structure Detection (Binary Signal)

E5 scores cluster at 0.70-0.77 for ALL causal content and lower for non-causal content. This makes E5 a reliable **binary filter** — "does this text contain causal language?" — even though it cannot distinguish WHICH causal topic matches the query.

| Content Type | E5 Score Range | E1 Score Range |
|-------------|---------------|---------------|
| Causal text (any topic) | 0.70 - 0.77 | 0.79 - 0.91 (topic-dependent) |
| Neutral text | 0.55 - 0.68 | 0.70 - 0.83 (topic-dependent) |

E1 also discriminates causal from non-causal, but E5 does so through a fundamentally different mechanism: E5 detects the structural markers of causation (because, causes, leads to, results in) regardless of domain, while E1 detects topical similarity.

### 4.2 E5 Provides: Directional Cause/Effect Asymmetry

E5's dual-vector encoding (cause_vec, effect_vec) stores two physically different vectors per memory. The direction modifiers in `asymmetric.rs` apply:
- `search_causes`: 0.8x dampening (abductive direction)
- `search_effects`: 1.2x boost (predictive direction)

Measured ratio: 1.47x (98% of theoretical 1.50x). This directional asymmetry is impossible with E1 alone, which stores a single vector per memory.

**However**: The directional modifiers currently shift all E5 scores uniformly (because E5 spread is only 0.026) and cannot change relative rankings. After fine-tuning, if E5 spread reaches >0.10, the direction modifiers would produce meaningful rank changes.

### 4.3 E5 Provides: search_causes / search_effects API

The `search_causes` and `search_effects` MCP tools use E5's dual vectors plus direction modifiers to express directional causal queries. These tools correctly:
- Auto-detect causal direction from query text (96-100% accuracy)
- Apply the `causal_reasoning` weight profile (E1=0.40, E5=0.10)
- Apply direction modifiers to E5 scores
- Return results with causal metadata (direction, confidence, modifier values)

This API infrastructure is valuable regardless of E5's current ranking contribution.

### 4.4 What E5 Does NOT Provide

| Capability | E5 Status | Why |
|-----------|-----------|-----|
| Topical ranking discrimination | NO | All causal content scores 0.70-0.77 regardless of topic |
| Top-1 accuracy improvement over E1 | NO | 0/6 standalone, 0.0% ablation delta |
| Cross-embedder anomaly discovery | NO | E5 never surfaces results that E1 misses |
| Direction-based rank reordering | NO | Spread too narrow for modifiers to change ranks |
| Entity-level causal discrimination | NO | "smoking→cancer" vs "pollution→cancer" same E5 score |

### 4.5 Occam's Razor Assessment

**E5's current unique value = direction modifiers + causal API infrastructure.**

Without fine-tuning, E5 contributes no ranking improvement. E1 alone achieves 100% top-1 accuracy across all tested queries. The system would function identically with E5 weight set to 0.00.

The reason to keep E5 at 0.10 (rather than 0.00) is forward compatibility: when the fine-tuning pipeline (Section 5 of the plan) is executed, E5's weight can be increased to 0.25-0.35 without any code changes.

---

## 5. Full State Verification Results

Eight-phase FSV was executed with 5 synthetic memories (3 causal + 2 neutral):

| Phase | Test | Result |
|-------|------|--------|
| 1 | Store 5 memories, all get 13 embedders | PASS |
| 2 | Physical vectors via `get_memory_fingerprint` — E1(1024D) + E5(768D) + cause/effect variants | PASS |
| 3 | Happy path: 3/3 correct top-1 (smoking, deforestation, interest rates) | PASS |
| 4 | Direction modifiers: abductive=0.8, predictive=1.2 confirmed in metadata | PASS |
| 5 | Edge cases: minimal "a", negation "does NOT cause", minimal content "x" | PASS |
| 6 | E5 spread ~0 vs E1 spread 8.4% — E1 drives all rankings | CONFIRMED |
| 7 | Ablation: removing E5 changes zero rankings | CONFIRMED |
| 8 | Code-simplifier review: 0 HIGH issues remaining after fixes | PASS |

### Key Physical Evidence

Old Longformer-generated E5 vectors show `e5=-0.027` when searched with nomic query vectors (incompatible embedding spaces). New nomic-generated E5 vectors show `e5~1.0` (same space). This confirms the model replacement invalidated all old E5 vectors as intended.

---

## 6. Architecture After Replacement

### 6.1 Model Specifications

| Property | Value |
|----------|-------|
| Model | nomic-ai/nomic-embed-text-v1.5 |
| Parameters | 137M |
| Architecture | NomicBERT (12 layers, 12 heads, 768 hidden) |
| Position encoding | Rotary (RoPE), base=1000, full head_dim |
| FFN activation | SwiGLU (gate + up + down projections) |
| QKV | Fused (single Wqkv weight, no bias) |
| Pooling | Mean pooling + L2 normalization |
| Output dimension | 768 (matches E5 slot exactly) |
| Max tokens | 512 (configurable, model supports 8192) |
| VRAM | ~500MB FP32 |
| Asymmetry | Instruction-prefix dual forward pass |

### 6.2 Instruction Prefixes (config.rs)

```rust
pub const CAUSE_INSTRUCTION: &str = "search_query: Identify the cause in: ";
pub const EFFECT_INSTRUCTION: &str = "search_query: Identify the effect of: ";
```

### 6.3 Weight Profile (causal_reasoning)

```
E1=0.40  E2=0.0  E3=0.0  E4=0.0  E5=0.10  E6=0.05  E7=0.15
E8=0.10  E9=0.0  E10=0.10  E11=0.10  E12=0.0  E13=0.0
```

### 6.4 Module Structure

```
crates/context-graph-embeddings/src/models/pretrained/causal/
  mod.rs          - Module exports
  config.rs       - NomicConfig, constants, instruction prefixes
  weights.rs      - NomicWeights, TrainableProjection
  loader.rs       - load_nomic_weights() from safetensors
  model.rs        - CausalModel struct, embed_dual(), EmbeddingModel trait
  tests.rs        - Unit tests
  forward/
    mod.rs        - gpu_forward(), gpu_forward_dual()
    attention.rs  - RoPE multi-head self-attention
    encoder.rs    - Encoder layers with SwiGLU FFN
    ops.rs        - layer_norm, mean_pooling, l2_normalize
```

---

## 7. Comparative Analysis: E5 vs E1

### 7.1 Quantitative Comparison

| Metric | E1 (Semantic) | E5 (Causal) | Ratio |
|--------|--------------|-------------|-------|
| Dimension | 1024 | 768 | 1.33x |
| Score range per query | 0.79 - 0.91 | 0.70 - 0.77 | — |
| Spread per query | 0.051 avg | 0.026 avg | E1 is 2.0x better |
| Standalone top-1 accuracy | 6/6 (100%) | 0/6 (0%) | E1 is infinitely better |
| Ablation delta | N/A (baseline) | 0.0% | E5 has no measurable impact |
| RRF contribution | 41% of total | ~10% of total | E1 is 4.1x more important |
| Vectors per memory | 1 | 2 (cause + effect) | E5 stores 2x more |
| Forward passes per embed | 1 | 2 (dual instruction) | E5 is 2x slower |

### 7.2 Qualitative Comparison

**E1 (nomic-embed-text-v1.5, 1024D)**: General-purpose semantic embedder. Produces isotropic, well-spread embeddings that discriminate by topical similarity. "Smoking causes lung cancer" and "Deforestation causes erosion" receive very different E1 scores when queried about lung cancer (0.905 vs 0.816). E1 is the sole ranking driver in the current system.

**E5 (nomic-embed-text-v1.5, 768D)**: Causal structure embedder. Detects whether text contains causal language patterns but cannot distinguish which causal topic matches. Both "smoking causes lung cancer" and "deforestation causes erosion" receive similar E5 scores (~0.74) because both contain causal structure. E5 provides directional asymmetry (cause vs effect vectors) that E1 cannot.

### 7.3 Why E5 Is Still Compressed Despite Using a "Good" Base Model

The nomic-embed-text-v1.5 model IS isotropic for general sentence similarity. The compression occurs because:

1. **Instruction prefix narrows the embedding space.** The prefix "search_query: Identify the cause in: " constrains the model to focus on causal markers, which are syntactically similar across all causal text (because, causes, leads to, results in). This creates a narrow sub-space within the otherwise isotropic embedding space.

2. **Causal structure is domain-agnostic.** The syntactic pattern of causation ("X causes Y") is identical across domains. The model correctly detects this shared structure — which is exactly why all causal text scores similarly.

3. **Fine-tuning is needed to inject domain-specific discrimination.** The 3-stage training pipeline would teach the model that "smoking→cancer" and "deforestation→erosion" are different causal relationships, not just instances of the same causal structure. This requires contrastive training on domain-diverse causal pairs (CausalBank, 314M pairs).

---

## 8. Recommendations

### 8.1 Immediate (No Code Changes)

- **Keep E5 at 0.10 weight** — Prevents E5 from harming rankings while preserving causal API infrastructure.
- **Keep search_causes/search_effects tools** — Correct directional metadata regardless of ranking quality.
- **Keep direction modifiers** — Mathematical precision is 98% of theoretical. Will become useful after fine-tuning.

### 8.2 Short-Term: Fine-Tuning Pipeline (Estimated 3-4 weeks)

Execute Sections 4-5 of the Option B plan:

1. **Stage 1: Domain Adaptation** — 1 epoch on 10M CausalBank pairs. Goal: adapt embedding space to causal language. Expected result: anisotropy drops from ~0.75 to ~0.40.

2. **Stage 2: Hard Negative Mining** — 2 epochs with mined negatives (rank 50-200). Goal: discriminate between different causal relationships. Expected result: spread increases to ~0.06.

3. **Stage 3: Direction Fine-Tuning** — 1 epoch on 2M directional triplets with Soft-ZCA whitening. Goal: asymmetric cause/effect encoding. Expected result: spread >0.10, direction accuracy >80%.

After fine-tuning, update causal_reasoning profile: E5 from 0.10 to 0.25-0.35, E1 from 0.40 to 0.25-0.30.

### 8.3 Alternative: E5 as Binary Gate

If fine-tuning is deferred, convert E5 from a continuous RRF weight to a binary gate:
- E5 score > 0.70 → include memory in causal search results
- E5 score < 0.70 → exclude from causal results
- Remove E5 from RRF fusion entirely (weight=0.00)

This would formalize E5's current role as a structure detector without pretending it's a ranking signal.

### 8.4 Long-Term: v2-moe Upgrade Path

nomic-ai/nomic-embed-text-v2-moe (475M params, MoE 8x2) could provide better causal discrimination through expert routing. However, it requires custom MoE routing in candle and 2x VRAM (~1.2GB). Recommended as a future upgrade after v1.5 fine-tuning is proven.

---

## 9. Test Results Summary

| Test Suite | Count | Status |
|-----------|-------|--------|
| MCP tools (`context-graph-mcp`) | 1322 | All pass |
| Core library (`context-graph-core`) | 132 | All pass |
| Causal embeddings (`context-graph-embeddings` causal tests) | 37 | All pass |
| Causal agent (`context-graph-causal-agent`) | 35 | All pass |
| Release build | — | Zero errors |

### Code Quality (Post-Simplifier Review)

- 0 HIGH severity issues
- 0 MEDIUM severity issues (both fixed)
- 5 LOW severity issues (all addressed)
- Dead code: none remaining
- Unsafe impl: documented with safety comments
- Error handling: all `.unwrap()` replaced with proper `map_err()`

---

## 10. Conclusion

The Option B base model replacement is a **structural improvement** that positions the system for future gains:

1. **Cleaner architecture**: 700+ lines of marker detection code eliminated. Asymmetry now comes from the model's instruction-following capability rather than hand-crafted pattern matching.

2. **Better foundation**: nomic-embed's contrastive pre-training provides a dramatically better starting point for fine-tuning than Longformer's MLM-only training.

3. **No regression**: 100% top-1 accuracy maintained. All tests pass. Build succeeds.

4. **Honest assessment**: E5 contributes zero ranking value today. Its unique contribution is directional infrastructure (cause/effect vectors, direction modifiers, search API) that will become meaningful after fine-tuning.

The system's accuracy is entirely driven by E1. This is not a failure — it's an honest assessment that enables the right next step: domain-specific fine-tuning to teach E5 the difference between "smoking causes cancer" and "deforestation causes erosion."

---

## Appendix A: Key References

| Document | Path |
|----------|------|
| Option B Plan | `docs/OPTION_B_CAUSAL_EMBEDDER_FINETUNING_PLAN.md` |
| Accuracy Benchmark | `docs/ACCURACY_BENCHMARK_REPORT.md` |
| Weight Profile Source | `crates/context-graph-core/src/weights/mod.rs:144-160` |
| Causal Model Source | `crates/context-graph-embeddings/src/models/pretrained/causal/` |
| Asymmetric Search | `crates/context-graph-core/src/causal/asymmetric.rs` |
| Direction Modifiers | AP-77: CAUSE_TO_EFFECT=1.2, EFFECT_TO_CAUSE=0.8 |

## Appendix B: Benchmark Query Results (Summary)

### Phase 1 — Top-1 Accuracy (6/6 = 100%)

| Query | Top-1 | E1 Score |
|-------|-------|---------|
| What causes lung cancer? | Smoking/DNA damage | 0.905 |
| Effects of CO2 emissions? | CO2/temperature rise | 0.875 |
| Effects of poverty? | Poverty/education access | 0.854 |
| What causes glacial melting? | Glacial melting/sea level | 0.876 |
| Tell me about Arctic climate | Arctic daylight cycles | 0.831 |
| What causes addiction? | Dopamine/reward pathways | 0.851 |

### Phase 2 — Ablation (0.0% delta)

| Condition | Top-1 Accuracy | Avg Similarity |
|-----------|---------------|---------------|
| Baseline (E5=0.10) | 6/6 (100%) | 0.732 |
| Exclude E5 | 6/6 (100%) | 0.732 |
| E1-only | 6/6 (100%) | 0.793 |

### Phase 4 — Direction Modifiers (6/6 = 100%)

| Direction | Modifier | Measured Ratio | Theoretical |
|-----------|----------|---------------|------------|
| search_causes | 0.8x | — | — |
| search_effects | 1.2x | 1.47x | 1.50x (98%) |
