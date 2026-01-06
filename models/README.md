# Model Weight Files

This directory contains pre-trained model weights required for the Context Graph embedding pipeline.

## Directory Structure

```
models/
├── README.md                      # This file
├── models_config.toml             # Auto-generated model paths
├── sparse_projection.safetensors  # REQUIRED: Sparse projection weights [30522, 1536]
├── sparse/                        # SPLADE backbone (naver/splade-cocondenser-ensembledistil)
├── semantic/                      # E1: intfloat/e5-large-v2
├── code/                          # E7: microsoft/codebert-base
├── causal/                        # E5: allenai/longformer-base-4096
├── entity/                        # E11: sentence-transformers/all-MiniLM-L6-v2
├── graph/                         # E8: sentence-transformers/paraphrase-MiniLM-L6-v2
├── multimodal/                    # E10: openai/clip-vit-large-patch14
├── late-interaction/              # E12: colbert-ir/colbertv2.0
├── contextual/                    # sentence-transformers/all-mpnet-base-v2
└── temporal/                      # E2-E4: Custom temporal embeddings
```

## Sparse Projection Matrix

### CRITICAL: This File is REQUIRED

The sparse projection matrix converts 30522-dimensional sparse vectors (from SPLADE)
to 1536-dimensional dense vectors for multi-array storage.

**If this file is missing, the system WILL FAIL. There is NO fallback.**

### File Specification

| Property | Value |
|----------|-------|
| **File Name** | `sparse_projection.safetensors` |
| **Location** | `models/sparse_projection.safetensors` |
| **Format** | SafeTensors v0.4+ |
| **Tensor Name** | `projection.weight` |
| **Shape** | [30522, 1536] |
| **Data Type** | float32 |
| **Size** | ~187 MB (30522 × 1536 × 4 bytes = 187,527,168 bytes) |
| **Download URL** | https://huggingface.co/contextgraph/sparse-projection |

### Shape Breakdown

```
Input:  30522 (BERT vocabulary size - sparse dimension)
Output: 1536  (Constitution E6_Sparse projected dimension)

Matrix: [30522 rows × 1536 columns]
        Each row i contains the dense representation for vocabulary token i
```

### Checksum Verification

The embedding system verifies weight file integrity via SHA-256 checksum.

```
Expected checksum: <TBD_AFTER_TRAINING>
Verification: SHA256(sparse_projection.safetensors) must match
```

**Verification command:**
```bash
sha256sum models/sparse_projection.safetensors
```

### Training Details

| Property | Value |
|----------|-------|
| **Training Dataset** | MS MARCO passages |
| **Training Objective** | Contrastive learning with semantic preservation |
| **Semantic Preservation Score** | >0.85 (required) |
| **Constitution Version** | 4.0.0 |
| **Training Script** | `scripts/train_sparse_projection.py` (TBD) |

### Usage in Code

The projection matrix is loaded automatically during model initialization:

```rust
use crate::models::pretrained::sparse::{ProjectionMatrix, PROJECTION_WEIGHT_FILE};

// Load projection matrix (TASK-EMB-011)
let model_path = PathBuf::from("models");
let projection = ProjectionMatrix::load(&model_path)?;

// Project sparse to dense (TASK-EMB-012)
let sparse_vector = model.embed_sparse(&input).await?;
let dense_vector = projection.project(&sparse_vector)?;

// Result: 1536D dense vector for multi-array storage
assert_eq!(dense_vector.len(), 1536);
```

### CRITICAL: No Hash Fallback

**Constitution AP-007 prohibits stub data in production.**

The previous hash-based projection (`idx % projected_dim`) has been **REMOVED** because:

1. **Hash collisions destroy semantics**:
   - Token "machine" (idx 3057) and "learning" (idx 4593) could map to the same dimension
   - `3057 % 1536 = 481` and `4593 % 1536 = 481` (collision!)

2. **No learned representation**:
   - Hash modulo is random noise, not semantic structure

3. **Violates AP-007**:
   - Hash fallback is stub/mock behavior masquerading as real functionality

**If `sparse_projection.safetensors` is missing, the system MUST panic with:**
```
[EMB-E006] PROJECTION_MATRIX_MISSING: Weight file not found at models/sparse_projection.safetensors
  Expected: models/sparse_projection.safetensors
  Remediation: Download projection weights from https://huggingface.co/contextgraph/sparse-projection
```

## Other Model Weights

### Pre-trained Models (Downloaded from HuggingFace)

| Embedder | Directory | HuggingFace Repo | Size |
|----------|-----------|------------------|------|
| E1 Semantic | `semantic/` | intfloat/e5-large-v2 | ~1.3GB |
| E5 Causal | `causal/` | allenai/longformer-base-4096 | ~717MB |
| E6 SPLADE | `sparse/` | naver/splade-cocondenser-ensembledistil | ~508MB |
| E7 Code | `code/` | microsoft/codebert-base | ~513MB |
| E8 Graph | `graph/` | sentence-transformers/paraphrase-MiniLM-L6-v2 | ~87MB |
| E10 Multimodal | `multimodal/` | openai/clip-vit-large-patch14 | ~1.6GB |
| E11 Entity | `entity/` | sentence-transformers/all-MiniLM-L6-v2 | ~87MB |
| E12 Late Interaction | `late-interaction/` | colbert-ir/colbertv2.0 | ~419MB |

### Custom Models (Require Training)

| Embedder | Directory | Status |
|----------|-----------|--------|
| E2-E4 Temporal | `temporal/` | Custom implementation |
| E9 HDC | `hdc/` | Custom hyperdimensional computing |
| E13 SPLADE (v3) | `splade-v3/` | Alternative SPLADE version |

## Downloading Models

### Using the Download Script

```bash
# Download all pre-trained models
python scripts/download_models.py

# Download specific model
python scripts/download_models.py --model sparse
```

### Manual Download

Each model can be downloaded manually from HuggingFace:

```bash
# Example: Download SPLADE model
git lfs install
git clone https://huggingface.co/naver/splade-cocondenser-ensembledistil models/sparse
```

## Validation

### Verify All Required Weights Exist

```bash
#!/bin/bash
# Check all required weight files

MODELS_DIR="models"
REQUIRED_FILES=(
    "sparse_projection.safetensors"
    "sparse/model.safetensors"
    "semantic/model.safetensors"
    "code/model.safetensors"
    "causal/model.safetensors"
    "entity/model.safetensors"
    "graph/model.safetensors"
    "multimodal/model.safetensors"
    "late-interaction/model.safetensors"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$MODELS_DIR/$file" ]]; then
        echo "✓ $file exists"
    else
        echo "✗ $file MISSING"
    fi
done
```

### Verify Sparse Projection Shape

```bash
# Using Python to verify safetensors file
python -c "
from safetensors import safe_open
with safe_open('models/sparse_projection.safetensors', framework='pt') as f:
    tensor = f.get_tensor('projection.weight')
    print(f'Shape: {tensor.shape}')
    assert tensor.shape == (30522, 1536), f'Wrong shape: {tensor.shape}'
    print('✓ Shape verified: [30522, 1536]')
"
```

## Troubleshooting

### Error: PROJECTION_MATRIX_MISSING

**Symptom:**
```
[EMB-E006] PROJECTION_MATRIX_MISSING: Weight file not found
```

**Solution:**
1. Download the projection weights:
   ```bash
   wget https://huggingface.co/contextgraph/sparse-projection/resolve/main/sparse_projection.safetensors -O models/sparse_projection.safetensors
   ```
2. Or train your own (see training documentation)

### Error: WEIGHT_CHECKSUM_MISMATCH

**Symptom:**
```
[EMB-E004] WEIGHT_CHECKSUM_MISMATCH: Corrupted weight file
```

**Solution:**
1. Delete the corrupted file
2. Re-download from trusted source
3. Verify checksum matches

### Error: DIMENSION_MISMATCH

**Symptom:**
```
[EMB-E005] DIMENSION_MISMATCH: Projection matrix has wrong shape
```

**Solution:**
1. Verify you downloaded the correct file
2. Shape MUST be [30522, 1536]
3. Older versions may have [30522, 768] - these are INCOMPATIBLE

## Constitution References

- **E6_Sparse**: `dim: "~30K 5%active"` → 1536D output via learned projection
- **AP-007**: No stub data in production - hash fallback is FORBIDDEN
- **E13_SPLADE**: Same projection architecture as E6

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-06 | Initial specification |
