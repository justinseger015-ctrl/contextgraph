# TASK-CORE-003: SemanticFingerprint Type-Safe Access & Validation

```xml
<task_spec id="TASK-CORE-003" version="4.0">
<metadata>
  <title>SemanticFingerprint Type-Safe Access & Validation</title>
  <status>COMPLETED</status>
  <completed_date>2026-01-09</completed_date>
  <layer>foundation</layer>
  <sequence>3</sequence>
  <implements>
    <requirement_ref>REQ-TELEOLOGICAL-02</requirement_ref>
    <requirement_ref>REQ-STORAGE-ATOMIC-01</requirement_ref>
    <requirement_ref>ARCH-01: TeleologicalArray is atomic (all 13 embeddings)</requirement_ref>
    <requirement_ref>ARCH-05: All 13 Embedders Must Be Present</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETED">TASK-CORE-002</task_ref>
  </depends_on>
</metadata>

<decision_made>
## ARCHITECTURAL DECISION: COMPLETED

**Decision**: SemanticFingerprint IS TeleologicalArray (Option 4 from original spec).

The existing `SemanticFingerprint` struct already correctly implements the 13-embedder
storage architecture. A type alias `TeleologicalArray = SemanticFingerprint` provides
spec alignment without code duplication.

**Implemented Features**:
1. ✅ `pub type TeleologicalArray = SemanticFingerprint;`
2. ✅ `SemanticFingerprint::get(&self, embedder: Embedder) -> EmbeddingRef<'_>`
3. ✅ `SemanticFingerprint::is_complete(&self) -> bool`
4. ✅ `SemanticFingerprint::validate_strict(&self) -> Result<(), ValidationError>`
5. ✅ `SemanticFingerprint::storage_bytes(&self) -> usize`
6. ✅ `EmbeddingRef<'a>` enum (Dense, Sparse, TokenLevel variants)
7. ✅ `ValidationError` enum with detailed error context
8. ✅ 24 comprehensive tests passing
</decision_made>

<implementation_summary>
## Files Modified

### Primary Implementation
**File**: `crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs`
- Line 26: `pub type TeleologicalArray = SemanticFingerprint;`
- Lines 40-47: `EmbeddingRef<'a>` enum definition
- Lines 53-99: `ValidationError` enum with 4 variants
- Lines 323-338: `get(&self, embedder: Embedder) -> EmbeddingRef<'_>`
- Lines 350-361: `is_complete(&self) -> bool`
- Lines 367-370: `storage_bytes(&self) -> usize`
- Lines 387-497: `validate_strict(&self) -> Result<(), ValidationError>`

### Re-exports
**File**: `crates/context-graph-core/src/types/fingerprint/mod.rs`
- Line 42-44: Added re-exports for `EmbeddingRef`, `TeleologicalArray`, `ValidationError`

### Tests
**File**: `crates/context-graph-core/src/types/fingerprint/semantic/tests/task_core_003_tests.rs`
- 24 tests covering all edge cases

### Module Declaration
**File**: `crates/context-graph-core/src/types/fingerprint/semantic/tests/mod.rs`
- Added `mod task_core_003_tests;`
</implementation_summary>

<source_of_truth>
## Source of Truth (For Verification)

| Component | Location | What to Verify |
|-----------|----------|----------------|
| Embedder enum | `src/teleological/embedder.rs` | 13 variants, `index()`, `from_index()`, `expected_dims()`, `all()` |
| Dimension constants | `src/types/fingerprint/semantic/constants.rs` | E1_DIM=1024, E2_DIM=512, etc. |
| SparseVector | `src/types/fingerprint/sparse.rs` | `SparseVectorError` exported |
| SemanticFingerprint | `src/types/fingerprint/semantic/fingerprint.rs` | All 5 methods implemented |
| TeleologicalArray alias | `src/types/fingerprint/semantic/fingerprint.rs:26` | Type alias exists |
| Re-exports | `src/types/fingerprint/mod.rs:42-44` | All types re-exported |
| Tests | `src/types/fingerprint/semantic/tests/task_core_003_tests.rs` | 24 tests passing |

**Note**: All paths are relative to `crates/context-graph-core/`
</source_of_truth>

<verification_commands>
## Full State Verification Commands

Run these commands to verify the implementation is correct:

```bash
# 1. Verify compilation (MUST pass with no errors)
cargo check -p context-graph-core
# Expected: Finished dev profile

# 2. Verify no duplicate Embedder definitions
rg "pub enum Embedder\b" --type rust crates/context-graph-core/
# Expected: ONLY teleological/embedder.rs

# 3. Run TASK-CORE-003 specific tests
cargo test -p context-graph-core task_core_003 -- --nocapture
# Expected: 24 passed; 0 failed

# 4. Verify type alias exists
rg "pub type TeleologicalArray = SemanticFingerprint" --type rust crates/context-graph-core/
# Expected: One match at semantic/fingerprint.rs

# 5. Verify re-exports
rg "TeleologicalArray" --type rust crates/context-graph-core/src/types/fingerprint/mod.rs
# Expected: TeleologicalArray in re-export list

# 6. Verify ValidationError re-export
rg "ValidationError" --type rust crates/context-graph-core/src/types/fingerprint/mod.rs
# Expected: ValidationError in re-export list

# 7. Verify EmbeddingRef re-export
rg "EmbeddingRef" --type rust crates/context-graph-core/src/types/fingerprint/mod.rs
# Expected: EmbeddingRef in re-export list
```
</verification_commands>

<test_evidence>
## Test Evidence (2026-01-09)

```
running 24 tests
test types::fingerprint::semantic::tests::task_core_003_tests::test_all_embedders_via_get ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_embedder_dims_match_get_type ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_empty_sparse_valid ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_empty_token_level_valid ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_embedding_ref_categorization ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_get_returns_correct_dimensions ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_is_complete_invalid_dimensions ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_bincode_serialization_roundtrip ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_max_sparse_index_valid ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_is_complete_zeroed ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_storage_bytes ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_storage_bytes_with_sparse ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_storage_size_bytes_consistency ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_teleological_array_alias ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_bincode_serialization_with_data ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_validate_strict_sparse_duplicate ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_validate_strict_sparse_out_of_bounds ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_validate_strict_sparse_length_mismatch ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_validate_strict_sparse_unsorted ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_validate_strict_valid ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_validate_strict_wrong_e1_dimension ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_validate_strict_wrong_token_dimension ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_validation_error_display ... ok
test types::fingerprint::semantic::tests::task_core_003_tests::test_json_serialization_roundtrip ... ok
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 2712 filtered out
```
</test_evidence>

<edge_cases_verified>
## Edge Cases Verified

| Scenario | Test Name | Result |
|----------|-----------|--------|
| Empty dense embedding | `test_validate_strict_wrong_e1_dimension` | ✅ Returns `DimensionMismatch` |
| Wrong E1 dimension (512 instead of 1024) | `test_validate_strict_wrong_e1_dimension` | ✅ Returns `DimensionMismatch` |
| Empty sparse vector (nnz=0) | `test_empty_sparse_valid` | ✅ VALID (sparse can be empty) |
| Sparse index out of bounds (50000) | `test_validate_strict_sparse_out_of_bounds` | ✅ Returns `SparseVectorError` |
| E12 with 0 tokens | `test_empty_token_level_valid` | ✅ VALID (empty content) |
| E12 token with wrong dim (64 instead of 128) | `test_validate_strict_wrong_token_dimension` | ✅ Returns `TokenDimensionMismatch` |
| Maximum sparse index (30521) | `test_max_sparse_index_valid` | ✅ VALID |
| Unsorted sparse indices | `test_validate_strict_sparse_unsorted` | ✅ Returns `SparseVectorError` |
| Duplicate sparse indices | `test_validate_strict_sparse_duplicate` | ✅ Returns `SparseVectorError` |
| Mismatched indices/values lengths | `test_validate_strict_sparse_length_mismatch` | ✅ Returns `SparseVectorError` |
| bincode serialization roundtrip | `test_bincode_serialization_roundtrip` | ✅ PASS |
| JSON serialization roundtrip | `test_json_serialization_roundtrip` | ✅ PASS |
</edge_cases_verified>

<downstream_tasks>
## Downstream Tasks Unblocked

This task completion unblocks:

1. **TASK-CORE-004: Define Comparison Types**
   - Can use `SemanticFingerprint`/`TeleologicalArray` for comparison operands
   - Can use `EmbeddingRef` for per-embedder comparison

2. **TASK-LOGIC-001: Dense Similarity Functions**
   - Can use `EmbeddingRef::Dense` for similarity computation
   - Can use `Embedder::is_dense()` to filter

3. **TASK-CORE-006: Storage Implementation**
   - Can serialize/deserialize `SemanticFingerprint`
   - Uses `storage_bytes()` for allocation sizing

4. **TASK-LOGIC-012: Entry-Point Selection**
   - Can use `SemanticFingerprint::get(embedder)` for single-space search
</downstream_tasks>

<api_reference>
## API Reference

### Types

```rust
/// Type alias: SemanticFingerprint IS TeleologicalArray
pub type TeleologicalArray = SemanticFingerprint;

/// Reference to embedding data (no-copy access)
pub enum EmbeddingRef<'a> {
    Dense(&'a [f32]),           // E1, E2-E5, E7-E11
    Sparse(&'a SparseVector),   // E6, E13
    TokenLevel(&'a [Vec<f32>]), // E12
}

/// Validation error with full context
pub enum ValidationError {
    DimensionMismatch { embedder: Embedder, expected: usize, actual: usize },
    EmptyDenseEmbedding { embedder: Embedder, expected: usize },
    SparseVectorError { embedder: Embedder, source: SparseVectorError },
    TokenDimensionMismatch { embedder: Embedder, token_index: usize, expected: usize, actual: usize },
}
```

### Methods

```rust
impl SemanticFingerprint {
    /// Type-safe access by Embedder enum
    pub fn get(&self, embedder: Embedder) -> EmbeddingRef<'_>;

    /// Check all dense embeddings have correct dimensions
    pub fn is_complete(&self) -> bool;

    /// Total heap allocation in bytes
    pub fn storage_bytes(&self) -> usize;

    /// Comprehensive validation with detailed errors
    pub fn validate_strict(&self) -> Result<(), ValidationError>;
}
```

### Usage Example

```rust
use context_graph_core::types::fingerprint::{SemanticFingerprint, EmbeddingRef, ValidationError};
use context_graph_core::teleological::Embedder;

fn process_fingerprint(fp: &SemanticFingerprint) -> Result<(), ValidationError> {
    // Validate before use
    fp.validate_strict()?;

    // Type-safe access
    match fp.get(Embedder::Semantic) {
        EmbeddingRef::Dense(data) => {
            println!("E1 has {} dimensions", data.len());
        }
        _ => unreachable!("E1 is always dense"),
    }

    // Check memory usage
    println!("Fingerprint uses {} bytes", fp.storage_bytes());

    Ok(())
}
```
</api_reference>

<critical_constraints>
## Critical Constraints (From constitution.yaml)

| Constraint | Severity | Enforcement |
|------------|----------|-------------|
| ARCH-01: All 13 embeddings must be stored atomically | critical | `validate_strict()` checks all 13 |
| ARCH-05: Missing embedders are fatal | critical | `validate_strict()` returns error |
| AP-14: No `.unwrap()` in library code | medium | All validation returns `Result` |
| NO Default impl | high | Intentionally omitted (see comment at line 546) |
| NO backwards compatibility | critical | No deprecated shims |
</critical_constraints>

</task_spec>
```
