# TASK-EMB-009: Create Weight File Specification

<task_spec id="TASK-EMB-009" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-009 |
| **Title** | Document Weight File Specification |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 9 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-002 |
| **Estimated Complexity** | low |
| **Constitution Ref** | `AP-007` (no stub data in production) |

---

## Context

The sparse projection weight file needs formal specification for validation during loading. This includes expected shape, checksum verification, and download instructions.

**Why This Matters:**
- Clear specification enables automated validation
- Users need download instructions when weights are missing
- Checksum verification ensures file integrity

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |
| ProjectionMatrix | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` |

---

## Prerequisites

- [ ] TASK-EMB-002 completed (ProjectionMatrix struct exists)

---

## Scope

### In Scope
- Create weight file specification in models/README.md
- Document expected tensor shape [30522, 1536]
- Document checksum verification process
- Document download URL

### Out of Scope
- Actual weight training (separate process)
- Weight file creation (separate process)

---

## Definition of Done

### Exact File Content

```markdown
# Model Weight Files

This directory contains pre-trained model weights required for the embedding pipeline.

## Sparse Projection Matrix

### File Details

| Property | Value |
|----------|-------|
| **File Name** | `sparse_projection.safetensors` |
| **Format** | SafeTensors v0.4+ |
| **Tensor Name** | `projection.weight` |
| **Shape** | [30522, 1536] |
| **Data Type** | float32 |
| **Size** | ~187 MB (30522 * 1536 * 4 bytes) |
| **Download URL** | https://huggingface.co/contextgraph/sparse-projection |

### Checksum Verification

The embedding system verifies weight file integrity via SHA-256 checksum.

Expected checksum (placeholder until training complete):
```
SHA256: <TBD_AFTER_TRAINING>
```

### Training Details

- **Trained On**: MS MARCO passages
- **Training Objective**: Contrastive learning
- **Semantic Preservation Score**: >0.85
- **Constitution Version**: 4.0.0

### Usage

The projection matrix is loaded automatically during model initialization:

```rust
// Automatic loading in SparseModel::load()
let projection = ProjectionMatrix::load(&model_path)?;

// Project sparse to dense
let dense = projection.project(&sparse_vector)?;
```

### CRITICAL: No Hash Fallback

If this file is missing, the system WILL PANIC. There is NO fallback to hash-based projection.

The hash-based approach (`idx % projected_dim`) destroys semantic information and is
explicitly forbidden by Constitution AP-007 (no stub data in production).

## Other Model Weights

### E1-E13 Embedders

Each embedder requires its own weight files. See individual model documentation for details.

| Embedder | Weight File | Size | Source |
|----------|-------------|------|--------|
| E1_Semantic | `semantic.safetensors` | ~2GB | Nomic |
| E5_Causal | `causal.safetensors` | ~1.5GB | Custom |
| E7_Code | `code.safetensors` | ~2GB | CodeBERT |
| ... | ... | ... | ... |

## Directory Structure

```
models/
├── README.md                    # This file
├── sparse_projection.safetensors  # Sparse projection weights
├── semantic/                    # E1 weights
├── temporal/                    # E2-E4 weights
├── causal/                      # E5 weights
├── sparse/                      # E6, E13 weights
├── code/                        # E7 weights
├── graph/                       # E8 weights
├── hdc/                         # E9 weights
├── multimodal/                  # E10 weights
├── entity/                      # E11 weights
└── late_interaction/            # E12 weights
```
```

### Constraints
- Must be placed in models/ directory
- Checksum placeholder is acceptable (TBD)
- Must emphasize no fallback policy

### Verification
- README exists
- Correct tensor shape documented
- Download URL present

---

## Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/models/README.md` | Weight file specification |

---

## Validation Criteria

- [ ] Shape documented as [30522, 1536]
- [ ] File size documented as ~187 MB
- [ ] Download URL present
- [ ] No fallback policy emphasized

---

## Test Commands

```bash
ls -la /home/cabdru/contextgraph/crates/context-graph-embeddings/models/
cat /home/cabdru/contextgraph/crates/context-graph-embeddings/models/README.md
```

---

## Traceability

| Requirement | Tech Spec | Issue |
|-------------|-----------|-------|
| REQ-EMB-001 | TECH-EMB-001 | ISSUE-001 |

---

## Notes

- The checksum will be filled in after actual weight training
- This documentation enables developers to understand weight requirements
- The "no fallback" policy prevents silent degradation

</task_spec>
