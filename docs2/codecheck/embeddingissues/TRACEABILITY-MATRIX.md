# Embedding Pipeline Traceability Matrix

<traceability_matrix id="TRACE-EMB-001" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Document ID** | TRACE-EMB-001 |
| **Title** | Embedding Pipeline Traceability Matrix |
| **Version** | 1.0 |
| **Created** | 2026-01-06 |
| **Status** | Complete |
| **Spec Reference** | SPEC-EMB-001 |
| **Constitution Reference** | v4.0.0 |

---

## Purpose

This matrix provides full traceability from:
1. Constitutional requirements to functional requirements
2. Functional requirements to technical specifications
3. Technical specifications to implementation tasks
4. Implementation tasks to test cases

---

## Requirement to Task Mapping

### REQ-EMB-001: Learned Sparse Projection Matrix

| Attribute | Value |
|-----------|-------|
| **Priority** | CRITICAL |
| **Constitution Ref** | `E6_Sparse: { dim: "~30K 5%active" }`, `AP-007` |
| **Issue Refs** | ISSUE-001, ISSUE-002 |
| **Tech Spec** | TECH-EMB-001 |

| Task ID | Task Title | Layer | Status |
|---------|------------|-------|--------|
| TASK-EMB-001 | Fix Dimension Constants | Foundation | Ready |
| TASK-EMB-002 | Create ProjectionMatrix Struct | Foundation | Ready |
| TASK-EMB-003 | Create ProjectionError Enum | Foundation | Ready |
| TASK-EMB-008 | Update SparseVector Struct | Foundation | Ready |
| TASK-EMB-011 | Implement ProjectionMatrix::load() | Logic | Ready |
| TASK-EMB-012 | Implement ProjectionMatrix::project() | Logic | Ready |
| TASK-EMB-021 | Integrate into SparseModel | Surface | Ready |

**Verification Tests:**
- UT-001: Sparse projection uses matrix multiplication
- UT-002: Projection preserves semantic similarity
- BM-004: Sparse projection latency < 3ms

---

### REQ-EMB-002: Dimension Alignment

| Attribute | Value |
|-----------|-------|
| **Priority** | CRITICAL |
| **Constitution Ref** | `embeddings.models` |
| **Issue Refs** | ISSUE-002 |
| **Tech Spec** | TECH-EMB-001 (section on dimensions) |

| Task ID | Task Title | Layer | Status |
|---------|------------|-------|--------|
| TASK-EMB-001 | Fix Dimension Constants | Foundation | Ready |
| TASK-EMB-008 | Update SparseVector Struct | Foundation | Ready |

**Verification Tests:**
- UT-003: All dimension constants consistent
- UT-004: ModelId.projected_dimension() matches output

---

### REQ-EMB-003: Real Weight Loading

| Attribute | Value |
|-----------|-------|
| **Priority** | CRITICAL |
| **Constitution Ref** | `stack.gpu`, `AP-007` |
| **Issue Refs** | ISSUE-003 |
| **Tech Spec** | TECH-EMB-002 |

| Task ID | Task Title | Layer | Status |
|---------|------------|-------|--------|
| TASK-EMB-006 | Create WarmLoadResult Struct | Foundation | Ready |
| TASK-EMB-010 | Create Golden Reference Fixtures | Foundation | Ready |
| TASK-EMB-013 | Implement Real Weight Loading | Logic | Ready |
| TASK-EMB-014 | Implement Real VRAM Allocation | Logic | Ready |
| TASK-EMB-015 | Implement Real Inference Validation | Logic | Ready |

**Verification Tests:**
- IT-001: Weight loading increases VRAM (nvidia-smi)
- IT-002: Checksum changes with weight file changes
- IT-003: Test inference produces non-trivial output

---

### REQ-EMB-004: CUDA Requirement Enforcement

| Attribute | Value |
|-----------|-------|
| **Priority** | HIGH |
| **Constitution Ref** | `stack.gpu.target: "RTX 5090"`, `AP-007` |
| **Issue Refs** | ISSUE-004 |
| **Tech Spec** | TECH-EMB-002 (section on preflight) |

| Task ID | Task Title | Layer | Status |
|---------|------------|-------|--------|
| TASK-EMB-019 | Remove Stub Mode from Preflight | Logic | Ready |

**Verification Tests:**
- Compile test: Build without cuda feature fails
- IT-004: Missing GPU causes panic
- IT-005: Insufficient VRAM causes panic

---

### REQ-EMB-005: Quantization Implementation

| Attribute | Value |
|-----------|-------|
| **Priority** | MEDIUM |
| **Constitution Ref** | `embeddings.quantization` |
| **Issue Refs** | ISSUE-006 |
| **Tech Spec** | TECH-EMB-003 |

| Task ID | Task Title | Layer | Status |
|---------|------------|-------|--------|
| TASK-EMB-004 | Create Quantization Structs | Foundation | Ready |
| TASK-EMB-016 | Implement PQ-8 Quantization | Logic | Ready |
| TASK-EMB-017 | Implement Float8 Quantization | Logic | Ready |
| TASK-EMB-018 | Implement Binary Quantization | Logic | Ready |
| TASK-EMB-020 | Implement QuantizationRouter | Logic | Ready |

**Verification Tests:**
- UT-005: PQ-8 achieves 32x compression
- UT-006: Float8 achieves 4x compression
- UT-007: Dequantization error < threshold
- BM-003: Storage per fingerprint < 20KB

---

### REQ-EMB-006: Storage Module Implementation

| Attribute | Value |
|-----------|-------|
| **Priority** | MEDIUM |
| **Constitution Ref** | `storage.layer1_primary`, `storage.layer2c_per_embedder` |
| **Issue Refs** | ISSUE-005 |
| **Tech Spec** | TECH-EMB-004 |

| Task ID | Task Title | Layer | Status |
|---------|------------|-------|--------|
| TASK-EMB-005 | Create Storage Types | Foundation | Ready |
| TASK-EMB-022 | Implement Storage Backend | Surface | Ready |
| TASK-EMB-023 | Implement Multi-Space Search | Surface | Ready |

**Verification Tests:**
- IT-006: Store/retrieve roundtrip preserves data
- IT-007: Per-embedder index queries work

---

### REQ-EMB-007: Performance Compliance

| Attribute | Value |
|-----------|-------|
| **Priority** | HIGH |
| **Constitution Ref** | `perf.latency` |
| **Issue Refs** | All |
| **Tech Spec** | All |

| Task ID | Task Title | Layer | Status |
|---------|------------|-------|--------|
| TASK-EMB-025 | Integration Tests | Surface | Ready |

**Verification Tests:**
- BM-001: single_embed latency < 10ms p95
- BM-002: batch_embed_64 latency < 50ms p95
- IT-008: End-to-end latency within budget

---

## Full Matrix View

```
+-------------+------------------+-------------------+-------------------+
| Requirement | Tech Spec        | Foundation Tasks  | Logic Tasks       | Surface Tasks     |
+-------------+------------------+-------------------+-------------------+-------------------+
| REQ-EMB-001 | TECH-EMB-001     | 001,002,003,008   | 011,012           | 021               |
| REQ-EMB-002 | TECH-EMB-001     | 001,008           | -                 | -                 |
| REQ-EMB-003 | TECH-EMB-002     | 006,010           | 013,014,015       | -                 |
| REQ-EMB-004 | TECH-EMB-002     | -                 | 019               | -                 |
| REQ-EMB-005 | TECH-EMB-003     | 004               | 016,017,018,020   | -                 |
| REQ-EMB-006 | TECH-EMB-004     | 005               | -                 | 022,023           |
| REQ-EMB-007 | All              | -                 | -                 | 025               |
+-------------+------------------+-------------------+-------------------+-------------------+
```

---

## Task to Requirement Reverse Mapping

| Task ID | Requirement(s) | Tech Spec |
|---------|----------------|-----------|
| TASK-EMB-001 | REQ-EMB-001, REQ-EMB-002 | TECH-EMB-001 |
| TASK-EMB-002 | REQ-EMB-001 | TECH-EMB-001 |
| TASK-EMB-003 | REQ-EMB-001 | TECH-EMB-001 |
| TASK-EMB-004 | REQ-EMB-005 | TECH-EMB-003 |
| TASK-EMB-005 | REQ-EMB-006 | TECH-EMB-004 |
| TASK-EMB-006 | REQ-EMB-003 | TECH-EMB-002 |
| TASK-EMB-007 | All | All |
| TASK-EMB-008 | REQ-EMB-001, REQ-EMB-002 | TECH-EMB-001 |
| TASK-EMB-009 | REQ-EMB-001 | TECH-EMB-001 |
| TASK-EMB-010 | REQ-EMB-003 | TECH-EMB-002 |
| TASK-EMB-011 | REQ-EMB-001 | TECH-EMB-001 |
| TASK-EMB-012 | REQ-EMB-001 | TECH-EMB-001 |
| TASK-EMB-013 | REQ-EMB-003 | TECH-EMB-002 |
| TASK-EMB-014 | REQ-EMB-003 | TECH-EMB-002 |
| TASK-EMB-015 | REQ-EMB-003 | TECH-EMB-002 |
| TASK-EMB-016 | REQ-EMB-005 | TECH-EMB-003 |
| TASK-EMB-017 | REQ-EMB-005 | TECH-EMB-003 |
| TASK-EMB-018 | REQ-EMB-005 | TECH-EMB-003 |
| TASK-EMB-019 | REQ-EMB-004 | TECH-EMB-002 |
| TASK-EMB-020 | REQ-EMB-005 | TECH-EMB-003 |
| TASK-EMB-021 | REQ-EMB-001 | TECH-EMB-001 |
| TASK-EMB-022 | REQ-EMB-006 | TECH-EMB-004 |
| TASK-EMB-023 | REQ-EMB-006 | TECH-EMB-004 |
| TASK-EMB-024 | All | All |
| TASK-EMB-025 | REQ-EMB-007 | All |

---

## Issue to Task Mapping

| Issue ID | Description | Resolving Tasks |
|----------|-------------|-----------------|
| ISSUE-001 | Hash-based sparse projection | TASK-EMB-001, 002, 003, 008, 011, 012, 021 |
| ISSUE-002 | Dimension mismatch (768 vs 1536) | TASK-EMB-001, 008 |
| ISSUE-003 | Simulated warm loading | TASK-EMB-006, 010, 013, 014, 015 |
| ISSUE-004 | Stub mode with fake GPU | TASK-EMB-019 |
| ISSUE-005 | Empty storage module | TASK-EMB-005, 022, 023 |
| ISSUE-006 | Quantization not applied | TASK-EMB-004, 016, 017, 018, 020 |

---

## Test Case Coverage

### Unit Tests

| Test ID | Requirement | Tasks Verified | Status |
|---------|-------------|----------------|--------|
| UT-001 | REQ-EMB-001 | TASK-EMB-011, 012 | Planned |
| UT-002 | REQ-EMB-001 | TASK-EMB-012, 021 | Planned |
| UT-003 | REQ-EMB-002 | TASK-EMB-001 | Planned |
| UT-004 | REQ-EMB-002 | TASK-EMB-001, 008 | Planned |
| UT-005 | REQ-EMB-005 | TASK-EMB-016 | Planned |
| UT-006 | REQ-EMB-005 | TASK-EMB-017 | Planned |
| UT-007 | REQ-EMB-005 | TASK-EMB-016, 017, 018 | Planned |

### Integration Tests

| Test ID | Requirement | Tasks Verified | Status |
|---------|-------------|----------------|--------|
| IT-001 | REQ-EMB-003 | TASK-EMB-014 | Planned |
| IT-002 | REQ-EMB-003 | TASK-EMB-013 | Planned |
| IT-003 | REQ-EMB-003 | TASK-EMB-015 | Planned |
| IT-004 | REQ-EMB-004 | TASK-EMB-019 | Planned |
| IT-005 | REQ-EMB-004 | TASK-EMB-019 | Planned |
| IT-006 | REQ-EMB-006 | TASK-EMB-022 | Planned |
| IT-007 | REQ-EMB-006 | TASK-EMB-023 | Planned |
| IT-008 | REQ-EMB-007 | TASK-EMB-025 | Planned |

### Benchmark Tests

| Test ID | Requirement | Metric | Target | Status |
|---------|-------------|--------|--------|--------|
| BM-001 | REQ-EMB-007 | single_embed latency | < 10ms p95 | Planned |
| BM-002 | REQ-EMB-007 | batch_embed_64 latency | < 50ms p95 | Planned |
| BM-003 | REQ-EMB-005 | storage per fingerprint | < 20KB | Planned |
| BM-004 | REQ-EMB-001 | sparse projection latency | < 3ms | Planned |

---

## Code to Delete (Anti-Patterns)

| Pattern | Location | Resolving Task |
|---------|----------|----------------|
| `idx % projected_dim` (hash projection) | `sparse/types.rs` | TASK-EMB-008 |
| `simulate_weight_loading()` | `warm/loader/operations.rs` | TASK-EMB-013 |
| `0x7f80_0000_0000` (fake pointer) | `warm/loader/operations.rs` | TASK-EMB-014 |
| `(i * 0.001).sin()` (fake output) | `warm/loader/operations.rs` | TASK-EMB-015 |
| `"Simulated RTX 5090"` | `warm/loader/preflight.rs` | TASK-EMB-019 |
| `0xDEAD_BEEF_CAFE_BABE` (fake checksum) | `warm/loader/operations.rs` | TASK-EMB-013 |

---

## Completion Checklist

### Foundation Layer (TASK-EMB-001 to TASK-EMB-010)

- [ ] TASK-EMB-001: Dimension constants fixed to 1536
- [ ] TASK-EMB-002: ProjectionMatrix struct created
- [ ] TASK-EMB-003: ProjectionError enum created
- [ ] TASK-EMB-004: Quantization structs created
- [ ] TASK-EMB-005: Storage types created
- [ ] TASK-EMB-006: WarmLoadResult struct created
- [ ] TASK-EMB-007: Consolidated error types created
- [ ] TASK-EMB-008: SparseVector updated
- [ ] TASK-EMB-009: Weight file spec documented
- [ ] TASK-EMB-010: Golden reference fixtures created

### Logic Layer (TASK-EMB-011 to TASK-EMB-020)

- [ ] TASK-EMB-011: ProjectionMatrix::load() implemented
- [ ] TASK-EMB-012: ProjectionMatrix::project() implemented
- [ ] TASK-EMB-013: Real weight loading implemented
- [ ] TASK-EMB-014: Real VRAM allocation implemented
- [ ] TASK-EMB-015: Real inference validation implemented
- [ ] TASK-EMB-016: PQ-8 quantization implemented
- [ ] TASK-EMB-017: Float8 quantization implemented
- [ ] TASK-EMB-018: Binary quantization implemented
- [ ] TASK-EMB-019: Stub mode removed
- [ ] TASK-EMB-020: QuantizationRouter implemented

### Surface Layer (TASK-EMB-021 to TASK-EMB-025)

- [ ] TASK-EMB-021: SparseModel integrated with projection
- [ ] TASK-EMB-022: Storage backend implemented
- [ ] TASK-EMB-023: Multi-space search implemented
- [ ] TASK-EMB-024: MCP handlers updated
- [ ] TASK-EMB-025: Integration tests complete

---

## Summary Statistics

| Category | Count |
|----------|-------|
| **Requirements** | 7 |
| **Tech Specs** | 4 |
| **Foundation Tasks** | 10 |
| **Logic Tasks** | 10 |
| **Surface Tasks** | 5 |
| **Total Tasks** | 25 |
| **Unit Tests** | 7 |
| **Integration Tests** | 8 |
| **Benchmark Tests** | 4 |
| **Total Tests** | 19 |

---

## Memory Key

Store completion status:
```
contextgraph/embedding-issues/traceability-complete
```

</traceability_matrix>
