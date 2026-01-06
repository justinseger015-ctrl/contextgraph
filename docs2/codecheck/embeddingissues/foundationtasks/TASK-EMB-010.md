# TASK-EMB-010: Create Golden Reference Fixtures

<task_spec id="TASK-EMB-010" version="2.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-010 |
| **Title** | Create Golden Reference Test Fixtures |
| **Status** | completed |
| **Layer** | foundation |
| **Sequence** | 10 (parallel with others) |
| **Implements** | REQ-EMB-003 |
| **Depends On** | None (can run in parallel) |
| **Estimated Complexity** | low |
| **Constitution Ref** | `AP-007` (no stub data in production) |
| **Last Updated** | 2026-01-06 |

---

## Context

TECH-EMB-002 specifies golden reference outputs for inference validation. This task creates:
1. Directory structure for golden reference fixtures
2. Placeholder files for each of the 13 embedding models
3. Documentation explaining the binary file format
4. A README.md documenting format, generation, and validation tolerance

**Why This Matters:**
- Golden references enable automated validation of model loading
- Ensures inference produces correct output after weight loading
- Catches silent failures in weight loading or model configuration
- Required by `InferenceValidator` trait in TECH-EMB-002

**Critical Constitution Reference (AP-007):**
> "Stub data in prod → use tests/fixtures/"

The directory created here is where real golden reference data will be stored after models are trained.

---

## Current State Analysis

### Existing Directory Structure
```
/home/cabdru/contextgraph/tests/fixtures/
├── .  (EMPTY - no golden subdirectory exists)
```

The `tests/fixtures/` directory exists but is empty. There is no `golden/` subdirectory.

### Related Implemented Components

1. **Warm Loading Types** (`crates/context-graph-embeddings/src/warm/loader/types.rs`):
   - `TensorMetadata` - shapes, dtype, total_params
   - `WarmLoadResult` - gpu_ptr, checksum (32-byte SHA256), size_bytes, load_duration
   - `LoadedModelWeights` - model_id, tensors, file_checksum (32-byte SHA256)
   - All types enforce fail-fast validation per Constitution AP-007

2. **Validation Module** (`crates/context-graph-embeddings/src/warm/validation/`):
   - `WarmValidator` with `DEFAULT_TOLERANCE: f32 = 1e-5`
   - `validate_dimensions()` - checks expected vs actual dimensions
   - `validate_weights_finite()` - checks for NaN/Inf
   - `compare_output()` - compares against reference with tolerance

3. **Error Module** (`crates/context-graph-embeddings/src/error/mod.rs`):
   - `WarmError::ModelDimensionMismatch { model_id, expected, actual }`
   - `WarmError::ModelValidationFailed { model_id, reason, expected_output, actual_output }`

### 13 Embedding Models (Constitution Reference)

From `constitution.yaml` § `embeddings.models`:

| ID | Model | Dimension | Quantization |
|----|-------|-----------|--------------|
| E1_Semantic | nomic-embed-text-v1.5 | 1024 | PQ-8 |
| E2_TemporalRecent | Temporal Recent | 512 | Float8 |
| E3_TemporalPeriodic | Temporal Periodic | 512 | Float8 |
| E4_TemporalPositional | Temporal Positional | 512 | Float8 |
| E5_Causal | Causal SCM | 768 | PQ-8 |
| E6_Sparse | Sparse Projection | ~30K sparse | Sparse |
| E7_Code | CodeBERT | 1536 | PQ-8 |
| E8_Graph | MiniLM Graph | 384 | Float8 |
| E9_Hdc | HDC Binary | 1024 | Binary |
| E10_Multimodal | CLIP | 768 | PQ-8 |
| E11_Entity | MiniLM Entity | 384 | Float8 |
| E12_LateInteraction | ColBERT | 128/tok | TokenPruning |
| E13_Splade | SPLADE v2 | ~30K sparse | Sparse |

---

## Scope

### In Scope
1. Create `tests/fixtures/golden/` directory structure
2. Create 13 model subdirectories (E1_Semantic through E13_Splade)
3. Create placeholder files in each directory (`test_input.bin`, `golden_output.bin`, `checksum.txt`)
4. Create `README.md` documenting the binary file format
5. Create `.gitkeep` files to preserve empty directories in git

### Out of Scope
- Actual golden data generation (requires model inference)
- Test implementation using golden data (Logic Layer - separate task)
- Weight file creation
- Model training

---

## Definition of Done

### Required Directory Structure

```
/home/cabdru/contextgraph/tests/fixtures/golden/
├── README.md                          # Format documentation
├── E1_Semantic/
│   ├── test_input.bin                 # Placeholder (0 bytes)
│   ├── golden_output.bin              # Placeholder (0 bytes)
│   ├── checksum.txt                   # Placeholder (0 bytes)
│   └── .gitkeep                       # Preserve directory in git
├── E2_TemporalRecent/
│   ├── test_input.bin
│   ├── golden_output.bin
│   ├── checksum.txt
│   └── .gitkeep
├── E3_TemporalPeriodic/
│   ├── test_input.bin
│   ├── golden_output.bin
│   ├── checksum.txt
│   └── .gitkeep
├── E4_TemporalPositional/
│   ├── test_input.bin
│   ├── golden_output.bin
│   ├── checksum.txt
│   └── .gitkeep
├── E5_Causal/
│   ├── test_input.bin
│   ├── golden_output.bin
│   ├── checksum.txt
│   └── .gitkeep
├── E6_Sparse/
│   ├── test_input.bin
│   ├── golden_output.bin
│   ├── checksum.txt
│   └── .gitkeep
├── E7_Code/
│   ├── test_input.bin
│   ├── golden_output.bin
│   ├── checksum.txt
│   └── .gitkeep
├── E8_Graph/
│   ├── test_input.bin
│   ├── golden_output.bin
│   ├── checksum.txt
│   └── .gitkeep
├── E9_Hdc/
│   ├── test_input.bin
│   ├── golden_output.bin
│   ├── checksum.txt
│   └── .gitkeep
├── E10_Multimodal/
│   ├── test_input.bin
│   ├── golden_output.bin
│   ├── checksum.txt
│   └── .gitkeep
├── E11_Entity/
│   ├── test_input.bin
│   ├── golden_output.bin
│   ├── checksum.txt
│   └── .gitkeep
├── E12_LateInteraction/
│   ├── test_input.bin
│   ├── golden_output.bin
│   ├── checksum.txt
│   └── .gitkeep
└── E13_Splade/
    ├── test_input.bin
    ├── golden_output.bin
    ├── checksum.txt
    └── .gitkeep
```

### README.md Content

Create `tests/fixtures/golden/README.md` with this exact content:

```markdown
# Golden Reference Test Fixtures

## Purpose

These fixtures contain golden reference outputs for inference validation.
They ensure that model loading produces correct inference results.

## Constitution Alignment

- **AP-007**: No stub data in production - these fixtures hold REAL validation data
- **REQ-EMB-003**: Inference validation against golden references
- **TECH-EMB-002**: Warm loading validation specification

## File Format

### test_input.bin

Binary format for dense embeddings (E1-E5, E7-E11):
- 4 bytes: dimension count (u32, little-endian)
- N * 4 bytes: f32 values (little-endian)

Example for 768-dim input:
```
Bytes 0-3:   0x00 0x03 0x00 0x00  (768 in little-endian)
Bytes 4-7:   f32 value #0 (little-endian)
Bytes 8-11:  f32 value #1 (little-endian)
...
```

### golden_output.bin

Same binary format as test_input.bin for dense models.

For sparse models (E6_Sparse, E13_Splade):
- 4 bytes: number of non-zero elements (u32, little-endian)
- For each non-zero element:
  - 4 bytes: index (u32, little-endian)
  - 4 bytes: value (f32, little-endian)

For late interaction (E12_LateInteraction):
- 4 bytes: number of tokens (u32, little-endian)
- 4 bytes: dimension per token (u32, little-endian = 128)
- For each token: 128 * 4 bytes: f32 values

### checksum.txt

SHA-256 checksum (64 hex characters) of the model weight file that produced this golden output.
Used to verify that golden data matches the current model weights.

Example content:
```
a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890
```

## Model Directory Mapping

| Directory | Model | Input Dim | Output Dim | Notes |
|-----------|-------|-----------|------------|-------|
| E1_Semantic | nomic-embed-text-v1.5 | 768 | 1024 | Dense, Matryoshka |
| E2_TemporalRecent | Temporal Recent | 512 | 512 | Dense |
| E3_TemporalPeriodic | Temporal Periodic | 512 | 512 | Dense |
| E4_TemporalPositional | Temporal Positional | 512 | 512 | Dense |
| E5_Causal | Causal SCM | 768 | 768 | Dense, Asymmetric |
| E6_Sparse | Sparse Projection | 1536 | ~30K | Sparse (~5% active) |
| E7_Code | CodeBERT | 768 | 1536 | Dense |
| E8_Graph | MiniLM Graph | 384 | 384 | Dense |
| E9_Hdc | HDC Binary | 1024 | 1024 | Binary (packed bits) |
| E10_Multimodal | CLIP | 768 | 768 | Dense |
| E11_Entity | MiniLM Entity | 384 | 384 | Dense |
| E12_LateInteraction | ColBERT | Variable | 128/tok | Token-level |
| E13_Splade | SPLADE v2 | Variable | ~30K | Sparse |

## Validation Tolerance

Per `WarmValidator::DEFAULT_TOLERANCE` in `warm/validation/validator.rs`:
- **FP32**: 1e-5 absolute difference
- **Mixed precision**: 1e-3 absolute difference (for Float8/PQ-8 quantized models)

## Generating Golden Data

Golden data is generated by running real inference with verified model weights:

```bash
# After model weights are available:
cargo run --bin generate-golden -- --model E1_Semantic --output tests/fixtures/golden/E1_Semantic/
```

Generation steps:
1. Load model weights (verified SHA256 checksum)
2. Run forward pass with standard test input
3. Save output as golden_output.bin
4. Record weight file checksum in checksum.txt

## Placeholder Status

Until weights are trained and inference is operational, directories contain
placeholder files (0 bytes).

Run this command to list all placeholder files:
```bash
find tests/fixtures/golden -size 0 -type f
```

Run this command to verify directory structure:
```bash
find tests/fixtures/golden -type d | sort
```

## Related Files

- `crates/context-graph-embeddings/src/warm/validation/validator.rs` - Validation logic
- `crates/context-graph-embeddings/src/warm/validation/comparisons.rs` - Output comparison
- `crates/context-graph-embeddings/src/warm/loader/types.rs` - Data types
- `docs2/codecheck/embeddingissues/TECH-EMB-002-warm-loading.md` - Technical spec
```

---

## Implementation Steps

### Step 1: Create Root Golden Directory

```bash
mkdir -p /home/cabdru/contextgraph/tests/fixtures/golden
```

### Step 2: Create Model Subdirectories with Placeholder Files

For each model, create directory with placeholder files:

```bash
MODELS="E1_Semantic E2_TemporalRecent E3_TemporalPeriodic E4_TemporalPositional E5_Causal E6_Sparse E7_Code E8_Graph E9_Hdc E10_Multimodal E11_Entity E12_LateInteraction E13_Splade"

for model in $MODELS; do
  mkdir -p "/home/cabdru/contextgraph/tests/fixtures/golden/$model"
  touch "/home/cabdru/contextgraph/tests/fixtures/golden/$model/test_input.bin"
  touch "/home/cabdru/contextgraph/tests/fixtures/golden/$model/golden_output.bin"
  touch "/home/cabdru/contextgraph/tests/fixtures/golden/$model/checksum.txt"
  touch "/home/cabdru/contextgraph/tests/fixtures/golden/$model/.gitkeep"
done
```

### Step 3: Create README.md

Write the README.md content from the "README.md Content" section above to:
```
/home/cabdru/contextgraph/tests/fixtures/golden/README.md
```

---

## Verification Criteria

### Automated Checks (Run These Commands)

```bash
# 1. Verify root directory exists
ls -la /home/cabdru/contextgraph/tests/fixtures/golden/

# 2. Count model directories (should be 13)
find /home/cabdru/contextgraph/tests/fixtures/golden/ -maxdepth 1 -type d | grep -v "^/home/cabdru/contextgraph/tests/fixtures/golden/$" | wc -l

# 3. List all directories (should show 13 model directories)
find /home/cabdru/contextgraph/tests/fixtures/golden/ -type d | sort

# 4. Count total files per directory (should be 4 each: test_input.bin, golden_output.bin, checksum.txt, .gitkeep)
for dir in /home/cabdru/contextgraph/tests/fixtures/golden/E*/; do
  count=$(ls -la "$dir" | grep -v "^total" | grep -v "^\." | wc -l)
  echo "$dir: $count files"
done

# 5. Verify README.md exists and has content
test -s /home/cabdru/contextgraph/tests/fixtures/golden/README.md && echo "README.md exists with content" || echo "FAIL: README.md missing or empty"

# 6. Verify all placeholder files exist (should be 39 files: 13 models * 3 placeholders)
find /home/cabdru/contextgraph/tests/fixtures/golden -name "*.bin" -o -name "checksum.txt" | wc -l

# 7. Verify all placeholders are 0 bytes
find /home/cabdru/contextgraph/tests/fixtures/golden \( -name "*.bin" -o -name "checksum.txt" \) -size 0 | wc -l
```

### Expected Results

| Check | Expected Result |
|-------|-----------------|
| Model directory count | 13 |
| Files per model directory | 4 (test_input.bin, golden_output.bin, checksum.txt, .gitkeep) |
| README.md | Exists with content |
| Placeholder files (bin + txt) | 39 total |
| Zero-byte placeholders | 39 (all bin and txt files should be 0 bytes) |

---

## Full State Verification

After completing the implementation, you MUST perform Full State Verification.

### Source of Truth

The source of truth is the filesystem at `/home/cabdru/contextgraph/tests/fixtures/golden/`.

### Execute & Inspect

Run these commands to verify the final state:

```bash
# Verify directory tree
tree /home/cabdru/contextgraph/tests/fixtures/golden/ 2>/dev/null || find /home/cabdru/contextgraph/tests/fixtures/golden/ -type f -o -type d | sort

# Verify file sizes (all placeholders should be 0 bytes)
find /home/cabdru/contextgraph/tests/fixtures/golden/ -type f -exec ls -l {} \;

# Verify README content starts with expected header
head -5 /home/cabdru/contextgraph/tests/fixtures/golden/README.md
```

### Boundary & Edge Case Audit

Manually verify these 3 edge cases:

#### Edge Case 1: Empty Directory Handling
```bash
# Before: Verify tests/fixtures/golden does NOT exist
ls /home/cabdru/contextgraph/tests/fixtures/golden/ 2>&1
# Expected: "No such file or directory" OR empty directory

# After: Verify golden/ exists with 13 subdirectories
ls -la /home/cabdru/contextgraph/tests/fixtures/golden/
# Expected: 13 E* directories + README.md
```

#### Edge Case 2: Git Preservation
```bash
# Verify .gitkeep files exist to preserve empty directories
find /home/cabdru/contextgraph/tests/fixtures/golden -name ".gitkeep" | wc -l
# Expected: 13

# Stage directories and verify they appear in git
git add /home/cabdru/contextgraph/tests/fixtures/golden/
git status | grep "new file.*golden"
# Expected: Shows new files in golden/
```

#### Edge Case 3: File Permissions
```bash
# Verify all files are readable
find /home/cabdru/contextgraph/tests/fixtures/golden/ -type f ! -readable 2>/dev/null | wc -l
# Expected: 0 (no unreadable files)
```

### Evidence of Success

After implementation, provide this log output:

```bash
# Final verification log
echo "=== TASK-EMB-010 Full State Verification ==="
echo ""
echo "1. Directory count:"
find /home/cabdru/contextgraph/tests/fixtures/golden/ -maxdepth 1 -type d | wc -l

echo ""
echo "2. All directories:"
find /home/cabdru/contextgraph/tests/fixtures/golden/ -type d | sort

echo ""
echo "3. File count per model:"
for dir in /home/cabdru/contextgraph/tests/fixtures/golden/E*/; do
  echo "  $dir: $(ls -1 "$dir" | wc -l) files"
done

echo ""
echo "4. README.md exists:"
ls -la /home/cabdru/contextgraph/tests/fixtures/golden/README.md

echo ""
echo "5. Total placeholder files:"
find /home/cabdru/contextgraph/tests/fixtures/golden \( -name "*.bin" -o -name "checksum.txt" \) | wc -l

echo ""
echo "6. Zero-byte files (should match total placeholders):"
find /home/cabdru/contextgraph/tests/fixtures/golden \( -name "*.bin" -o -name "checksum.txt" \) -size 0 | wc -l

echo ""
echo "=== Verification Complete ==="
```

---

## Traceability

| Requirement | Tech Spec | Constitution |
|-------------|-----------|--------------|
| REQ-EMB-003 | TECH-EMB-002 | AP-007 |

---

## Notes

- This task can run in parallel with all other Foundation tasks
- Golden data will be generated after model weights are available
- Placeholder files are 0 bytes by design (per Constitution AP-007, no fake data)
- The tolerance values (1e-5, 1e-3) are defined in `WarmValidator::DEFAULT_TOLERANCE`
- The 32-byte SHA256 checksum format matches `WarmLoadResult.checksum` and `LoadedModelWeights.file_checksum`

---

## Critical Reminders

1. **NO MOCK DATA**: All `.bin` and `.txt` files must be 0-byte placeholders until real golden data is generated
2. **NO BACKWARDS COMPATIBILITY**: System must work after changes or fail fast
3. **FAIL FAST**: If golden reference loading fails, the system should error with clear logs
4. **VERIFY OUTPUTS EXIST**: After creating files, manually verify they exist on disk

</task_spec>
