# Functional Specification: Embedding Pipeline Integrity Restoration

<functional_spec id="SPEC-EMB-001" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Spec ID** | SPEC-EMB-001 |
| **Title** | Embedding Pipeline Integrity Restoration |
| **Status** | Draft |
| **Version** | 1.0 |
| **Owner** | Context Graph Team |
| **Created** | 2026-01-06 |
| **Last Updated** | 2026-01-06 |
| **Related Issues** | ISSUE-001, ISSUE-002, ISSUE-003, ISSUE-004, ISSUE-005, ISSUE-006 |
| **Related Investigation** | SHERLOCK-03-EMBEDDINGS |
| **Constitution Version** | 4.0.0 |

---

## Overview

### Purpose

This specification defines the functional requirements for restoring integrity to the Context Graph embedding pipeline. The investigation (SHERLOCK-03-EMBEDDINGS) revealed critical issues where the system claims to provide learned neural projections, GPU-accelerated warm loading, and quantized embeddings, but actually provides hash-based approximations, simulated operations, and stub implementations.

### Problem Statement

The Context Graph embedding system violates the Constitution v4.0.0 in multiple ways:

1. **AP-007 Violation**: "Stub data in prod" - The warm loading pipeline is entirely simulated
2. **Dimension Mismatch**: Constitution specifies E6_Sparse as `dim: "~30K 5%active"` with 1536D projection, implementation produces 768D
3. **False Semantic Claims**: "Learned projection" is actually `idx % projected_dim` hash modulo
4. **Silent Failures**: Preflight returns fake GPU info instead of failing fast

### Who Benefits

| Stakeholder | Benefit |
|-------------|---------|
| **System Operators** | Reliable, predictable embedding behavior with clear error states |
| **Data Scientists** | True learned projections preserve semantic information |
| **End Users** | Higher quality retrieval from proper multi-space embeddings |
| **Developers** | Clear contracts between specification and implementation |

### Success Criteria

- All 13 embedders produce embeddings matching their Constitutional dimensions
- No simulated operations in production code paths
- System fails fast with clear errors when GPU unavailable
- Quantization actually reduces memory per Constitutional strategy
- Storage module provides real multi-array persistence

---

## User Stories

### US-001: Real Sparse Projection

**As a** data scientist using the Context Graph system,
**I want** sparse embeddings (E6/E13) to use learned neural projection,
**So that** semantic information is preserved rather than destroyed by hash collisions.

**Acceptance Criteria:**

```gherkin
Scenario: Sparse to dense projection preserves semantic similarity
  Given two semantically similar texts "machine learning" and "deep learning"
  When I compute their E6_Sparse embeddings
  And I compute cosine similarity of the projected dense vectors
  Then the similarity should be greater than 0.7
  And the projection should use a learned weight matrix (not hash modulo)

Scenario: Projection dimensions match Constitution
  Given the Constitution specifies E6_Sparse as "~30K 5%active" with 1536D output
  When I embed any text with E6_Sparse
  Then the output vector dimension should be exactly 1536
  And the sparse input should have ~30K vocabulary dimension
  And approximately 5% of sparse dimensions should be active
```

### US-002: Dimension Consistency

**As a** system integrator,
**I want** embedding dimensions to match between specification and implementation,
**So that** downstream systems receive vectors of expected dimensions without runtime panics.

**Acceptance Criteria:**

```gherkin
Scenario: ModelId reports correct dimension
  Given the model E6_Sparse (ModelId::Sparse)
  When I call projected_dimension() on the model
  Then it should return 1536
  And the actual embedding output should have dimension 1536

Scenario: All 13 embedders match Constitution dimensions
  Given the Constitution embedding specifications
  When I embed text with each of E1 through E13
  Then each output dimension should match:
    | Embedder | Expected Dimension |
    | E1_Semantic | 1024 |
    | E2_Temporal_Recent | 512 |
    | E3_Temporal_Periodic | 512 |
    | E4_Temporal_Positional | 512 |
    | E5_Causal | 768 |
    | E6_Sparse | 1536 |
    | E7_Code | 1536 |
    | E8_Graph | 384 |
    | E9_HDC | 1024 |
    | E10_Multimodal | 768 |
    | E11_Entity | 384 |
    | E12_LateInteraction | 128 per token |
    | E13_SPLADE | sparse ~30K |
```

### US-003: Real Warm Loading

**As a** system operator deploying Context Graph,
**I want** warm loading to actually load model weights into GPU VRAM,
**So that** models are ready for inference when the system reports ready.

**Acceptance Criteria:**

```gherkin
Scenario: Weight loading reads actual SafeTensors files
  Given a model configuration pointing to weight files
  When I call warm_load for the model
  Then the system should read actual bytes from the SafeTensors file
  And compute a real SHA256 checksum of the weights
  And the checksum should change if the weight file changes

Scenario: VRAM allocation uses real cudaMalloc
  Given CUDA feature is enabled and GPU is available
  When I warm load a model
  Then cudaMalloc should be called with the model size
  And the returned pointer should be a valid CUDA device pointer
  And nvidia-smi should show increased VRAM usage

Scenario: Validation runs real inference
  Given a warm-loaded model
  When validation runs test inference
  Then actual model forward pass should execute
  And output should vary based on input (not sin(i * 0.001))
  And output dimension should match model specification
```

### US-004: No Stub Modes

**As a** reliability engineer,
**I want** the system to fail immediately when GPU is unavailable,
**So that** we detect deployment issues rather than running with fake GPU.

**Acceptance Criteria:**

```gherkin
Scenario: Missing CUDA feature causes compile-time error
  Given the cuda feature is not enabled
  When I attempt to compile the embeddings crate
  Then compilation should fail with error "CUDA feature required"

Scenario: Missing GPU hardware causes runtime panic
  Given the cuda feature is enabled
  But no CUDA-capable GPU is present
  When I attempt to initialize the embedding pipeline
  Then the system should panic with message containing "No CUDA GPU"
  And no fake "Simulated RTX 5090" should appear

Scenario: Insufficient VRAM causes clear error
  Given a GPU with less than 32GB VRAM
  When I attempt to warm load all 13 models
  Then the system should error with specific VRAM requirements
  And should report actual available VRAM vs required VRAM
```

### US-005: Real Quantization

**As a** system operator optimizing memory,
**I want** quantization to actually reduce memory usage,
**So that** I get the Constitutional 63% storage reduction.

**Acceptance Criteria:**

```gherkin
Scenario: PQ-8 quantization reduces memory
  Given an E1_Semantic embedding (1024D float32 = 4096 bytes)
  When I apply PQ-8 quantization per Constitution
  Then storage should be approximately 128 bytes (32x reduction)
  And recall loss should be less than 5%

Scenario: Quantization is applied during storage
  Given a TeleologicalFingerprint to store
  When I store it with quantization enabled
  Then the stored size should be approximately 17KB
  And should NOT be the uncompressed 46KB
  And each embedder should use its Constitutional quantization:
    | Embedders | Method | Compression |
    | E1, E5, E7, E10 | PQ-8 | 32x |
    | E2, E3, E4, E8, E11 | Float8 | 4x |
    | E9 | Binary | 32x |
    | E6, E13 | Sparse | native |
    | E12 | TokenPruning | ~50% |
```

### US-006: Complete Storage Module

**As a** developer integrating with Context Graph,
**I want** the storage module to actually store and retrieve TeleologicalFingerprints,
**So that** the 13-embedding arrays persist across sessions.

**Acceptance Criteria:**

```gherkin
Scenario: Store and retrieve TeleologicalFingerprint
  Given a complete TeleologicalFingerprint with all 13 embeddings
  When I store it with a UUID key
  And I retrieve it by the same UUID
  Then all 13 embeddings should match exactly
  And the purpose_vector should be preserved
  And johari_quadrants should be preserved

Scenario: Per-embedder indexes are created
  Given stored TeleologicalFingerprints
  When I query by E1_Semantic similarity
  Then the query should use the E1 HNSW index
  And should NOT require loading all embeddings
```

---

## Requirements

### REQ-EMB-001: Learned Sparse Projection Matrix

**Priority:** CRITICAL
**Related Issues:** ISSUE-001, ISSUE-002
**Constitution Reference:** `E6_Sparse: { dim: "~30K 5%active" }`, `AP-007`

**Description:**
Replace the hash-based sparse-to-dense projection with a learned neural projection layer.

**Functional Requirements:**

| ID | Requirement |
|----|-------------|
| REQ-EMB-001.1 | System SHALL load a pre-trained projection weight matrix of size [30522 x 1536] (BERT vocab to output dim) |
| REQ-EMB-001.2 | System SHALL compute projection as `dense = W^T @ sparse` (matrix multiplication, not hash modulo) |
| REQ-EMB-001.3 | System SHALL L2-normalize the projected output |
| REQ-EMB-001.4 | Projection matrix SHALL be stored as SafeTensors file at `models/sparse_projection.safetensors` |
| REQ-EMB-001.5 | System SHALL FAIL if projection matrix file is missing (no fallback to hash) |
| REQ-EMB-001.6 | Projection SHALL execute on GPU via cuBLAS SpMM (sparse matrix-matrix multiply) |

**Verification:**

- Unit test: Semantic similarity preservation (>0.7 for related terms)
- Unit test: Projection matrix multiplication correctness
- Integration test: E6 output dimension = 1536
- Benchmark: Projection latency < 3ms per batch of 64

---

### REQ-EMB-002: Dimension Alignment

**Priority:** CRITICAL
**Related Issues:** ISSUE-002
**Constitution Reference:** `embeddings.models`

**Description:**
Align all dimension constants and outputs to match Constitution specifications.

**Functional Requirements:**

| ID | Requirement |
|----|-------------|
| REQ-EMB-002.1 | `SPARSE_PROJECTED_DIMENSION` SHALL be 1536 (not 768) |
| REQ-EMB-002.2 | `ModelId::Sparse.projected_dimension()` SHALL return 1536 |
| REQ-EMB-002.3 | Actual E6/E13 output vectors SHALL have dimension 1536 |
| REQ-EMB-002.4 | System SHALL panic at startup if any embedder output dimension mismatches its ModelId specification |
| REQ-EMB-002.5 | All dimension constants SHALL be defined in a single source of truth: `types/dimensions/constants.rs` |

**Verification:**

- Compile-time: Static assertion that SPARSE_PROJECTED_DIMENSION == ModelId::Sparse.projected_dimension()
- Unit test: Each embedder output matches Constitutional dimension
- Integration test: TeleologicalFingerprint storage/retrieval dimension consistency

---

### REQ-EMB-003: Real Weight Loading

**Priority:** CRITICAL
**Related Issues:** ISSUE-003
**Constitution Reference:** `stack.gpu`, `AP-007`

**Description:**
Replace simulated weight loading with actual SafeTensors file reading and GPU memory allocation.

**Functional Requirements:**

| ID | Requirement |
|----|-------------|
| REQ-EMB-003.1 | System SHALL read weight files using `safetensors::SafeTensors::deserialize()` |
| REQ-EMB-003.2 | System SHALL compute SHA256 checksum of actual weight bytes |
| REQ-EMB-003.3 | System SHALL call `cuMemAlloc` (via cudarc) for VRAM allocation |
| REQ-EMB-003.4 | System SHALL call `cuMemcpyHtoD` to transfer weights to GPU |
| REQ-EMB-003.5 | System SHALL verify loaded weights by running a deterministic test inference |
| REQ-EMB-003.6 | Test inference output SHALL be compared against a golden reference (not sin(i * 0.001)) |
| REQ-EMB-003.7 | System SHALL track actual VRAM usage and report it accurately |
| REQ-EMB-003.8 | `simulate_weight_loading` function SHALL be DELETED from codebase |

**Verification:**

- Integration test: nvidia-smi VRAM increase after warm load
- Integration test: Checksum changes when weights modified
- Integration test: Test inference produces model-specific output

---

### REQ-EMB-004: CUDA Requirement Enforcement

**Priority:** HIGH
**Related Issues:** ISSUE-004
**Constitution Reference:** `stack.gpu.target: "RTX 5090"`, `AP-007`

**Description:**
Enforce CUDA availability with compile-time and runtime checks. No fallback to simulation.

**Functional Requirements:**

| ID | Requirement |
|----|-------------|
| REQ-EMB-004.1 | `#[cfg(not(feature = "cuda"))]` blocks SHALL compile_error!, not return fake data |
| REQ-EMB-004.2 | Runtime SHALL panic with descriptive message if CUDA driver unavailable |
| REQ-EMB-004.3 | Runtime SHALL panic if no GPU with compute capability >= 8.0 found |
| REQ-EMB-004.4 | Runtime SHALL panic if GPU VRAM < 24GB (8GB headroom per Constitution) |
| REQ-EMB-004.5 | All references to "Simulated RTX 5090" SHALL be DELETED |
| REQ-EMB-004.6 | Error messages SHALL include: found hardware, required hardware, remediation steps |

**Verification:**

- Compile test: Build without cuda feature fails with compile_error
- Integration test: System panics with clear message on CPU-only machine
- Integration test: Error includes actual vs required specs

---

### REQ-EMB-005: Quantization Implementation

**Priority:** MEDIUM
**Related Issues:** ISSUE-006
**Constitution Reference:** `embeddings.quantization`

**Description:**
Implement actual quantization during weight loading and embedding storage.

**Functional Requirements:**

| ID | Requirement |
|----|-------------|
| REQ-EMB-005.1 | PQ-8 (Product Quantization, 8 centroids) SHALL be applied to E1, E5, E7, E10 |
| REQ-EMB-005.2 | Float8 (E4M3) SHALL be applied to E2, E3, E4, E8, E11 |
| REQ-EMB-005.3 | Binary quantization SHALL be applied to E9 (HDC) |
| REQ-EMB-005.4 | Sparse native format SHALL be used for E6, E13 |
| REQ-EMB-005.5 | Token pruning (~50%) SHALL be applied to E12 |
| REQ-EMB-005.6 | Total TeleologicalFingerprint size SHALL be ~17KB after quantization |
| REQ-EMB-005.7 | System SHALL provide dequantization for query-time similarity computation |
| REQ-EMB-005.8 | Recall loss from quantization SHALL be measured and logged |

**Verification:**

- Unit test: Quantized size matches expected compression ratio
- Unit test: Dequantized vectors have acceptable error (< 5% for PQ-8)
- Benchmark: Storage per memory < 20KB

---

### REQ-EMB-006: Storage Module Implementation

**Priority:** MEDIUM
**Related Issues:** ISSUE-005
**Constitution Reference:** `storage.layer1_primary`, `storage.layer2c_per_embedder`

**Description:**
Complete the placeholder storage module with real multi-array storage.

**Functional Requirements:**

| ID | Requirement |
|----|-------------|
| REQ-EMB-006.1 | Storage module SHALL implement `store(id: Uuid, fingerprint: TeleologicalFingerprint)` |
| REQ-EMB-006.2 | Storage module SHALL implement `retrieve(id: Uuid) -> Option<TeleologicalFingerprint>` |
| REQ-EMB-006.3 | Storage SHALL use RocksDB in dev, ScyllaDB in prod (per Constitution) |
| REQ-EMB-006.4 | Each embedder's vector SHALL be stored in its own column for per-space indexing |
| REQ-EMB-006.5 | 13 HNSW indexes SHALL be created, one per embedder |
| REQ-EMB-006.6 | Purpose vector (13D) SHALL have its own searchable index |
| REQ-EMB-006.7 | Johari quadrants and confidence SHALL be stored and retrievable |

**Verification:**

- Integration test: Store and retrieve round-trip preserves all fields
- Integration test: Per-embedder search uses correct index
- Benchmark: Storage latency < 10ms per fingerprint

---

### REQ-EMB-007: Performance Compliance

**Priority:** HIGH
**Related Issues:** All
**Constitution Reference:** `perf.latency`

**Description:**
All embedding operations must meet Constitutional latency budgets.

**Functional Requirements:**

| ID | Requirement |
|----|-------------|
| REQ-EMB-007.1 | single_embed SHALL complete in < 10ms (p95) |
| REQ-EMB-007.2 | batch_embed_64 SHALL complete in < 50ms (p95) |
| REQ-EMB-007.3 | Sparse projection SHALL complete in < 3ms per embedding |
| REQ-EMB-007.4 | Model warm loading SHALL complete in < 30s total for all 13 models |
| REQ-EMB-007.5 | Test inference validation SHALL complete in < 100ms per model |

**Verification:**

- Benchmark: Latency percentiles measured under load
- CI gate: bench regression < 5% per Constitution

---

## Edge Cases

### EC-001: Empty Input Text

| Scenario | Expected Behavior |
|----------|-------------------|
| Empty string "" | Return zero vector (all dimensions = 0.0) |
| Whitespace only | Treat as empty, return zero vector |
| Single character | Process normally, may have high entropy |

### EC-002: Extremely Long Text

| Scenario | Expected Behavior |
|----------|-------------------|
| Text > 512 tokens (BERT limit) | Truncate to max_tokens with warning logged |
| Text > 10MB | Reject with InputTooLarge error |

### EC-003: GPU Memory Pressure

| Scenario | Expected Behavior |
|----------|-------------------|
| VRAM nearly full | Log warning, attempt batch size reduction |
| VRAM exhausted during batch | FAIL with OOM error, do NOT return partial results |
| Concurrent embedding requests | Queue with backpressure, do NOT overcommit VRAM |

### EC-004: Weight File Issues

| Scenario | Expected Behavior |
|----------|-------------------|
| Weight file missing | PANIC with file path and download instructions |
| Weight file corrupted (checksum mismatch) | PANIC with expected vs actual checksum |
| Weight file wrong version | PANIC with version mismatch details |

### EC-005: Sparse Embedding Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| All sparse weights = 0 | Return zero vector (degenerate case) |
| Single active dimension | Project to dense, normalize (will have low magnitude) |
| > 10% active dimensions | Process normally (unusually dense for sparse model) |

### EC-006: Dimension Mismatches at Runtime

| Scenario | Expected Behavior |
|----------|-------------------|
| Model outputs wrong dimension | PANIC immediately, do NOT truncate or pad |
| Stored embedding has wrong dimension | FAIL retrieval with corruption error |

---

## Error States

**CRITICAL PRINCIPLE: FAIL FAST, NO FALLBACKS**

The system SHALL NOT:
- Return fake/simulated data
- Silently fall back to CPU
- Truncate or pad dimensions
- Use hash approximations when neural projection fails
- Return partial results on OOM

### Error Taxonomy

| Error Code | Category | Recoverable | Action |
|------------|----------|-------------|--------|
| EMB-E001 | CUDA_UNAVAILABLE | NO | Panic with hardware requirements |
| EMB-E002 | INSUFFICIENT_VRAM | NO | Panic with VRAM requirements |
| EMB-E003 | WEIGHT_FILE_MISSING | NO | Panic with file path |
| EMB-E004 | WEIGHT_CHECKSUM_MISMATCH | NO | Panic with checksum details |
| EMB-E005 | DIMENSION_MISMATCH | NO | Panic with expected vs actual |
| EMB-E006 | PROJECTION_MATRIX_MISSING | NO | Panic, do NOT use hash fallback |
| EMB-E007 | OOM_DURING_BATCH | NO | Panic, do NOT return partial |
| EMB-E008 | INFERENCE_VALIDATION_FAILED | NO | Panic with model name |
| EMB-E009 | INPUT_TOO_LARGE | YES | Return error to caller |
| EMB-E010 | STORAGE_CORRUPTION | NO | Panic with corruption details |

### Error Message Format

All error messages SHALL follow this format:

```
[EMB-EXXX] {Category}: {Description}
  Expected: {expected_value}
  Actual: {actual_value}
  Location: {file}:{line}
  Remediation: {steps_to_fix}
```

---

## Test Plan

**CRITICAL: NO MOCK DATA IN TESTS**

All tests SHALL use:
- Real embedding models (can be smaller test variants)
- Real weight files (from tests/fixtures/)
- Real GPU (tests requiring GPU marked with `#[cfg(feature = "cuda")]`)
- Real quantization (verify actual compression)

### Unit Tests

| Test ID | Requirement | Description |
|---------|-------------|-------------|
| UT-001 | REQ-EMB-001 | Sparse projection uses matrix multiplication |
| UT-002 | REQ-EMB-001 | Projection preserves semantic similarity |
| UT-003 | REQ-EMB-002 | All dimension constants consistent |
| UT-004 | REQ-EMB-002 | ModelId.projected_dimension() matches output |
| UT-005 | REQ-EMB-005 | PQ-8 achieves 32x compression |
| UT-006 | REQ-EMB-005 | Float8 achieves 4x compression |
| UT-007 | REQ-EMB-005 | Dequantization error < threshold |

### Integration Tests

| Test ID | Requirement | Description |
|---------|-------------|-------------|
| IT-001 | REQ-EMB-003 | Weight loading increases VRAM (nvidia-smi) |
| IT-002 | REQ-EMB-003 | Checksum changes with weight file changes |
| IT-003 | REQ-EMB-003 | Test inference produces non-trivial output |
| IT-004 | REQ-EMB-004 | Missing GPU causes panic |
| IT-005 | REQ-EMB-004 | Insufficient VRAM causes panic |
| IT-006 | REQ-EMB-006 | Store/retrieve roundtrip preserves data |
| IT-007 | REQ-EMB-006 | Per-embedder index queries work |
| IT-008 | REQ-EMB-007 | End-to-end latency within budget |

### Benchmark Tests

| Test ID | Requirement | Metric | Target |
|---------|-------------|--------|--------|
| BM-001 | REQ-EMB-007 | single_embed latency | < 10ms p95 |
| BM-002 | REQ-EMB-007 | batch_embed_64 latency | < 50ms p95 |
| BM-003 | REQ-EMB-005 | storage size per fingerprint | < 20KB |
| BM-004 | REQ-EMB-001 | sparse projection latency | < 3ms |

### Chaos Tests

| Test ID | Description | Expected Behavior |
|---------|-------------|-------------------|
| CT-001 | Kill GPU process during embedding | Clean panic, no corruption |
| CT-002 | Fill VRAM with other process | OOM panic, no partial results |
| CT-003 | Corrupt weight file mid-load | Checksum panic |
| CT-004 | Network partition during weight download | Clear error, retry instructions |

---

## Appendix A: Issue Summary

| Issue ID | Severity | Problem | Solution Requirement |
|----------|----------|---------|---------------------|
| ISSUE-001 | CRITICAL | Hash-based sparse projection | REQ-EMB-001 |
| ISSUE-002 | CRITICAL | Dimension mismatch (768 vs 1536) | REQ-EMB-002 |
| ISSUE-003 | CRITICAL | Simulated warm loading | REQ-EMB-003 |
| ISSUE-004 | HIGH | Stub mode with fake GPU | REQ-EMB-004 |
| ISSUE-005 | MEDIUM | Empty storage module | REQ-EMB-006 |
| ISSUE-006 | MEDIUM | Quantization not applied | REQ-EMB-005 |

---

## Appendix B: Constitution References

Key Constitution sections relevant to this specification:

- `stack.gpu`: RTX 5090, 32GB VRAM, CUDA 13.1
- `forbidden.AP-007`: "Stub data in prod -> use tests/fixtures/"
- `embeddings.models.E6_Sparse`: `dim: "~30K 5%active"`
- `embeddings.quantization`: Per-embedder quantization strategy
- `perf.latency.single_embed`: < 10ms
- `perf.latency.batch_embed_64`: < 50ms
- `storage.layer1_primary`: RocksDB/ScyllaDB with TeleologicalFingerprint schema

---

## Appendix C: File Locations

Files requiring modification:

| File | Issue | Change Required |
|------|-------|-----------------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` | ISSUE-001, ISSUE-002 | Replace hash projection with learned matrix |
| `crates/context-graph-embeddings/src/types/model_id/core.rs` | ISSUE-002 | Verify dimension alignment |
| `crates/context-graph-embeddings/src/warm/loader/operations.rs` | ISSUE-003 | Delete simulate functions, implement real loading |
| `crates/context-graph-embeddings/src/warm/loader/preflight.rs` | ISSUE-004 | Remove stub mode, add panic |
| `crates/context-graph-embeddings/src/storage/mod.rs` | ISSUE-005 | Implement complete storage |
| `crates/context-graph-embeddings/src/traits/model_factory/quantization.rs` | ISSUE-006 | Implement actual quantization |

---

## Appendix D: Requirement Traceability Matrix

| Requirement | User Story | Issue | Test Cases |
|-------------|------------|-------|------------|
| REQ-EMB-001 | US-001 | ISSUE-001 | UT-001, UT-002, BM-004 |
| REQ-EMB-002 | US-002 | ISSUE-002 | UT-003, UT-004 |
| REQ-EMB-003 | US-003 | ISSUE-003 | IT-001, IT-002, IT-003 |
| REQ-EMB-004 | US-004 | ISSUE-004 | IT-004, IT-005, CT-001, CT-002 |
| REQ-EMB-005 | US-005 | ISSUE-006 | UT-005, UT-006, UT-007, BM-003 |
| REQ-EMB-006 | US-006 | ISSUE-005 | IT-006, IT-007 |
| REQ-EMB-007 | All | All | BM-001, BM-002, IT-008 |

</functional_spec>

---

## Next Steps

This specification serves as the foundation for subsequent technical specifications:

1. **TECH-EMB-001**: Sparse Projection Architecture (REQ-EMB-001)
2. **TECH-EMB-002**: Warm Loading Implementation (REQ-EMB-003)
3. **TECH-EMB-003**: Quantization Implementation (REQ-EMB-005)
4. **TECH-EMB-004**: Storage Module Design (REQ-EMB-006)

Each technical specification will detail the implementation approach for its respective requirements.
