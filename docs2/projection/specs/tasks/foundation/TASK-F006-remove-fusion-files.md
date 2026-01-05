# Task: TASK-F006 - Remove All Fusion-Related Files

## Metadata
- **ID**: TASK-F006
- **Layer**: Foundation
- **Priority**: P0 (Critical Path - Blocking)
- **Estimated Effort**: L (Large - 29 files + 80+ references)
- **Dependencies**: None (Can run in parallel with TASK-F001 through TASK-F005)
- **Traces To**: TS-601, FR-601, FR-602
- **Last Audited**: 2026-01-05

## Current Codebase State (VERIFIED 2026-01-05)

### Fusion Files CONFIRMED TO EXIST (29 files in fusion/ directory):

```
crates/context-graph-embeddings/src/fusion/
├── mod.rs                                          # Root module - exports fusion types
├── experts/
│   ├── mod.rs                                      # Expert module exports
│   ├── activation.rs                               # Activation functions (GELU, ReLU, SiLU)
│   ├── expert.rs                                   # Expert FFN (8320→4096→1536)
│   ├── pool.rs                                     # ExpertPool with top-k routing
│   └── tests.rs                                    # Expert unit tests
├── gating/
│   ├── mod.rs                                      # Gating module exports
│   ├── layer_norm.rs                               # LayerNorm implementation
│   ├── linear.rs                                   # Linear projection layer
│   ├── routing.rs                                  # Top-k expert routing
│   ├── softmax.rs                                  # Softmax implementation
│   ├── network/
│   │   ├── mod.rs                                  # GatingNetwork module
│   │   ├── core.rs                                 # GatingNetwork struct
│   │   └── forward.rs                              # Forward pass implementation
│   └── tests/
│       ├── mod.rs                                  # Test module
│       ├── gating_tests.rs                         # Core gating tests
│       ├── edge_case_tests.rs                      # Edge case tests
│       └── integration_tests.rs                    # Integration tests
└── gpu_fusion/
    ├── mod.rs                                      # GPU module exports
    ├── activation.rs                               # GPU activation kernels
    ├── expert.rs                                   # GpuExpert implementation
    ├── expert_pool.rs                              # GpuExpertPool implementation
    ├── fusemoe.rs                                  # GpuFuseMoE (main fusion layer)
    ├── gating.rs                                   # GpuGatingNetwork
    ├── layer_norm.rs                               # GPU LayerNorm
    ├── linear.rs                                   # GPU Linear (cuBLAS GEMM)
    ├── tests.rs                                    # GPU fusion tests
    ├── tests_expert.rs                             # GPU expert tests
    └── tests_pool.rs                               # GPU pool tests
```

### Additional Fusion-Related Files (VERIFIED):

```
crates/context-graph-embeddings/src/
├── config/fusion.rs                                # FusionConfig struct (8266 bytes)
├── provider/fused.rs                               # FusedEmbeddingProvider
└── types/dimensions/fusemoe.rs                     # FuseMoE dimension constants

crates/context-graph-mcp/src/adapters/
├── embedding_adapter.rs                            # Uses FusedEmbeddingProvider (modify)
└── mod.rs                                          # Exports embedding_adapter (modify)
```

### Files That Import Fusion Types (112 files reference fusion patterns):

**Critical files requiring modification:**
1. `crates/context-graph-embeddings/src/lib.rs` - Exports fusion module and types
2. `crates/context-graph-embeddings/src/config/mod.rs` - Exports FusionConfig
3. `crates/context-graph-mcp/src/adapters/embedding_adapter.rs` - Uses FusedEmbeddingProvider
4. `crates/context-graph-mcp/src/adapters/mod.rs` - Exports adapter

**Storage/binary files with fusion references:**
- `crates/context-graph-embeddings/src/storage/binary/*.rs` (7 files)
- `crates/context-graph-embeddings/src/storage/gds/*.rs` (4 files)
- `crates/context-graph-embeddings/src/storage/batch/*.rs` (2 files)

## Description

**CRITICAL CONTEXT**: The Multi-Array Teleological Fingerprint architecture stores ALL 13 embeddings (E1-E13) without fusion. FuseMoE with top-k=4 loses 67% of information. The array IS the representation.

**WHAT THIS TASK DOES**: Completely remove the FuseMoE architecture that combines 12 embeddings into a single 1536D vector. The new architecture stores all 13 embeddings separately (~17KB quantized per memory).

**NO BACKWARDS COMPATIBILITY. NO WORKAROUNDS. NO FALLBACKS.**

If something breaks after removal, it must error with clear diagnostics - do not create compatibility shims.

## Acceptance Criteria

- [ ] All 29 files in `fusion/` directory deleted
- [ ] `config/fusion.rs` deleted
- [ ] `provider/fused.rs` deleted
- [ ] `types/dimensions/fusemoe.rs` deleted
- [ ] `lib.rs` updated: remove `pub mod fusion;` and all fusion exports
- [ ] `config/mod.rs` updated: remove FusionConfig export
- [ ] All `use.*fusion` imports removed from remaining files
- [ ] EmbeddingProviderAdapter in MCP crate updated to use new multi-array provider
- [ ] `cargo check --all` passes with ZERO fusion references
- [ ] `cargo test --all` passes
- [ ] `cargo clippy --all -- -D warnings` passes
- [ ] Git commit with full removal for audit trail

## Implementation Steps (EXACT ORDER)

### Phase 1: Pre-Flight Verification
```bash
# Document current state BEFORE any changes
git status
cargo check --all  # Must pass before starting
cargo test --all   # Must pass before starting

# Count fusion files (should be 29 in fusion/ + 3 additional)
find crates/context-graph-embeddings/src/fusion -type f -name "*.rs" | wc -l
# Expected: 29

# Count files with fusion references
rg -l "fuse|fusion|gating|FuseMoE|GatingNetwork|ExpertPool|FusedEmbedding" --type rust | wc -l
# Document this number - it's your baseline
```

### Phase 2: Delete Fusion Directory (29 files)
```bash
# Delete entire fusion directory tree
rm -rf crates/context-graph-embeddings/src/fusion/

# Verify deletion
ls crates/context-graph-embeddings/src/fusion/ 2>/dev/null && echo "ERROR: fusion dir still exists" || echo "OK: fusion dir deleted"
```

### Phase 3: Delete Additional Fusion Files (3 files)
```bash
# Delete FusionConfig
rm crates/context-graph-embeddings/src/config/fusion.rs

# Delete FusedEmbeddingProvider
rm crates/context-graph-embeddings/src/provider/fused.rs

# Delete FuseMoE dimensions
rm crates/context-graph-embeddings/src/types/dimensions/fusemoe.rs
```

### Phase 4: Update lib.rs
Edit `crates/context-graph-embeddings/src/lib.rs`:

**REMOVE these lines:**
```rust
pub mod fusion;  // Line ~58

// Remove from pub use config:
FusionConfig,

// Remove from pub use provider:
FusedEmbeddingProvider, FusedProviderConfig, ProjectionLayer,

// Remove from pub use fusion:
pub use fusion::{Activation, Expert, ExpertPool, GatingNetwork, LayerNorm, Linear};

// Remove these constants:
pub const DEFAULT_DIMENSION: usize = 1536;
pub const CONCATENATED_DIMENSION: usize = 8320;

// Remove doc examples mentioning FuseMoE
```

### Phase 5: Update config/mod.rs
Edit `crates/context-graph-embeddings/src/config/mod.rs`:

**REMOVE:**
```rust
mod fusion;  // Remove this module declaration
pub use fusion::FusionConfig;  // Remove this export
```

### Phase 6: Update types/dimensions/mod.rs
Edit `crates/context-graph-embeddings/src/types/dimensions/mod.rs`:

**REMOVE:**
```rust
mod fusemoe;  // Remove module declaration
pub use fusemoe::*;  // Remove re-export
// Remove any TOTAL_CONCATENATED, FUSED_OUTPUT, TOP_K_EXPERTS references
```

### Phase 7: Update provider/mod.rs
Edit `crates/context-graph-embeddings/src/provider/mod.rs`:

**REMOVE:**
```rust
mod fused;
pub use fused::{FusedEmbeddingProvider, FusedProviderConfig, ProjectionLayer};
```

### Phase 8: Fix Storage Module References
Files to modify (remove FusedEmbedding imports/usage):
- `crates/context-graph-embeddings/src/storage/binary/types.rs`
- `crates/context-graph-embeddings/src/storage/binary/encode.rs`
- `crates/context-graph-embeddings/src/storage/binary/decode.rs`
- `crates/context-graph-embeddings/src/storage/binary/mod.rs`
- `crates/context-graph-embeddings/src/storage/binary/reference.rs`
- `crates/context-graph-embeddings/src/storage/binary/tests.rs`
- `crates/context-graph-embeddings/src/storage/gds/*.rs`
- `crates/context-graph-embeddings/src/storage/batch/*.rs`

### Phase 9: Update MCP Adapter
Edit `crates/context-graph-mcp/src/adapters/embedding_adapter.rs`:

**This file currently wraps FusedEmbeddingProvider.** It must be:
1. Either deleted entirely (if MCP doesn't need embeddings yet)
2. Or updated to use the new MultiArrayEmbeddingProvider (TASK-F007)

**DECISION REQUIRED**: If MultiArrayEmbeddingProvider doesn't exist yet, delete the adapter and mark MCP embedding integration as blocked until TASK-F007 completes.

### Phase 10: Update error/types.rs
Remove fusion-related error variants if they exist.

### Phase 11: Update warm/ Module
Check and fix:
- `crates/context-graph-embeddings/src/warm/loader.rs`
- `crates/context-graph-embeddings/src/warm/memory_pool.rs`
- `crates/context-graph-embeddings/src/warm/cuda_alloc.rs`

### Phase 12: Iterative Compilation Fix
```bash
# First check - will show all errors
cargo check -p context-graph-embeddings 2>&1 | tee /tmp/fusion-errors.txt

# Count errors
grep -c "^error" /tmp/fusion-errors.txt

# Fix each error. Do NOT create stubs or workarounds.
# If something depends on fusion, that dependency must be removed or the dependent code deleted.
```

### Phase 13: Clean Up Tests
```bash
# Find and delete any remaining fusion tests outside fusion/ directory
rg -l "fusion|FuseMoE|GatingNetwork" crates/context-graph-embeddings/src/**/*test*.rs --type rust

# Delete any test files that only test fusion functionality
```

## Code Signature (Definition of Done)

After completion, ALL of these commands must succeed:

```bash
# 1. No fusion files exist
test ! -d crates/context-graph-embeddings/src/fusion && echo "PASS: fusion dir gone"

# 2. No fusion patterns in Rust files
rg -l "fuse|fusion|gating|FuseMoE|GatingNetwork|ExpertPool|FusedEmbedding|expert_select|Vector1536" --type rust
# Expected: NO OUTPUT (exit code 1)

# 3. No fusion module declarations
rg "pub mod fusion|mod fusion" --type rust
# Expected: NO OUTPUT

# 4. No fusion imports
rg "use.*fusion|use.*fuse|use.*FuseMoE|use.*Gating" --type rust
# Expected: NO OUTPUT

# 5. Compilation succeeds
cargo check --all
# Expected: exit code 0

# 6. All tests pass
cargo test --all
# Expected: exit code 0

# 7. No warnings
cargo clippy --all -- -D warnings
# Expected: exit code 0

# 8. Documentation builds without fusion references
cargo doc -p context-graph-embeddings --no-deps 2>&1 | grep -i fusion
# Expected: NO OUTPUT
```

## Full State Verification (MANDATORY)

### Source of Truth Definition
After removal, the "source of truth" is:
1. **File System**: No fusion files exist in `crates/context-graph-embeddings/src/fusion/`
2. **Cargo Check**: Compilation succeeds without fusion modules
3. **Grep Verification**: Zero matches for fusion patterns

### Execute & Inspect Protocol
```bash
# After making changes, immediately verify:
echo "=== SOURCE OF TRUTH VERIFICATION ==="

# Check 1: File system state
echo "--- File System Check ---"
ls -la crates/context-graph-embeddings/src/fusion/ 2>/dev/null
echo "Exit code: $? (should be 2 = directory not found)"

# Check 2: Pattern search
echo "--- Pattern Search ---"
FUSION_COUNT=$(rg -c "FuseMoE|GatingNetwork|FusedEmbedding|ExpertPool" --type rust 2>/dev/null | awk -F: '{sum+=$2} END{print sum}')
echo "Fusion pattern occurrences: $FUSION_COUNT (should be 0)"

# Check 3: Compilation state
echo "--- Compilation Check ---"
cargo check -p context-graph-embeddings 2>&1 | tail -5

# Check 4: Module resolution
echo "--- Module Resolution ---"
cargo metadata --format-version=1 | jq '.packages[] | select(.name=="context-graph-embeddings") | .targets[].name'
```

### Boundary & Edge Case Audit (MANDATORY - Execute These 3 Tests)

**Edge Case 1: Empty fusion directory (simulate incomplete deletion)**
```bash
# Before: Create edge case
mkdir -p /tmp/test-edge1
echo "If this test fails, deletion was incomplete"

# Test: Verify rm -rf handles nested directories
rm -rf crates/context-graph-embeddings/src/fusion/
RESULT=$(find crates/context-graph-embeddings/src -name "*fusion*" -type f 2>/dev/null | wc -l)

# After: Log state
echo "Edge Case 1 - Files remaining: $RESULT (expected: 0)"
```

**Edge Case 2: Dangling imports after deletion**
```bash
# Before: Document import count
BEFORE_IMPORTS=$(rg -c "use.*fusion" --type rust 2>/dev/null | awk -F: '{sum+=$2} END{print sum}')
echo "Before: $BEFORE_IMPORTS fusion imports"

# Test: After deletion, check for unresolved imports
cargo check -p context-graph-embeddings 2>&1 | grep -c "unresolved import"

# After: Log state
AFTER_IMPORTS=$(rg -c "use.*fusion" --type rust 2>/dev/null | awk -F: '{sum+=$2} END{print sum}')
echo "After: $AFTER_IMPORTS fusion imports (expected: 0)"
```

**Edge Case 3: Config deserialization without FusionConfig**
```bash
# Before: Check if any config files reference fusion
FUSION_CONFIGS=$(rg -l "fusion" config/ tests/fixtures/ 2>/dev/null | wc -l)
echo "Before: $FUSION_CONFIGS config files reference fusion"

# Test: Verify config loading doesn't fail
cargo test -p context-graph-embeddings config:: 2>&1 | tail -10

# After: Log state
echo "Config tests exit code: $? (expected: 0)"
```

### Evidence of Success (MANDATORY LOG)
Create this file after completion: `docs2/projection/specs/tasks/foundation/TASK-F006-completion-evidence.log`

```bash
# Generate evidence log
{
  echo "=== TASK-F006 COMPLETION EVIDENCE ==="
  echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Git Commit: $(git rev-parse HEAD)"
  echo ""
  echo "=== FILE SYSTEM STATE ==="
  echo "Fusion directory exists: $(test -d crates/context-graph-embeddings/src/fusion && echo YES || echo NO)"
  echo "fusion.rs exists: $(test -f crates/context-graph-embeddings/src/config/fusion.rs && echo YES || echo NO)"
  echo "fused.rs exists: $(test -f crates/context-graph-embeddings/src/provider/fused.rs && echo YES || echo NO)"
  echo "fusemoe.rs exists: $(test -f crates/context-graph-embeddings/src/types/dimensions/fusemoe.rs && echo YES || echo NO)"
  echo ""
  echo "=== PATTERN SEARCH ==="
  echo "FuseMoE occurrences: $(rg -c FuseMoE --type rust 2>/dev/null | awk -F: '{sum+=$2} END{print sum+0}')"
  echo "GatingNetwork occurrences: $(rg -c GatingNetwork --type rust 2>/dev/null | awk -F: '{sum+=$2} END{print sum+0}')"
  echo "FusedEmbedding occurrences: $(rg -c FusedEmbedding --type rust 2>/dev/null | awk -F: '{sum+=$2} END{print sum+0}')"
  echo "fusion import occurrences: $(rg -c 'use.*fusion' --type rust 2>/dev/null | awk -F: '{sum+=$2} END{print sum+0}')"
  echo ""
  echo "=== COMPILATION ==="
  cargo check --all 2>&1 | tail -3
  echo "Exit code: $?"
  echo ""
  echo "=== TESTS ==="
  cargo test --all 2>&1 | tail -5
  echo "Exit code: $?"
  echo ""
  echo "=== CLIPPY ==="
  cargo clippy --all -- -D warnings 2>&1 | tail -3
  echo "Exit code: $?"
} | tee docs2/projection/specs/tasks/foundation/TASK-F006-completion-evidence.log
```

## Sherlock-Holmes Final Verification (MANDATORY)

After completing all implementation steps, spawn a `sherlock-holmes` agent with this prompt:

```
FORENSIC INVESTIGATION: TASK-F006 Fusion Removal Verification

CONTEXT: TASK-F006 required removing ALL fusion-related code from context-graph-embeddings.
The removal should have deleted 29+ files and updated 80+ file references.

YOUR MISSION:
1. Verify the fusion/ directory no longer exists
2. Search for ANY remaining fusion patterns: FuseMoE, GatingNetwork, ExpertPool, FusedEmbedding, gating, expert_select, Vector1536
3. Verify cargo check --all passes
4. Verify cargo test --all passes
5. Check for orphaned imports or dead code
6. Verify no fusion types are exported from lib.rs
7. Check that the MCP adapter was properly updated or removed

EVIDENCE REQUIRED:
- List every file you checked
- For each potential issue, provide file path and line number
- Confirm or deny completion with specific evidence

GUILTY UNTIL PROVEN INNOCENT: Assume the removal was incomplete until you prove otherwise.
```

Any issues identified by sherlock-holmes MUST be fixed before marking this task complete.

## Constraints (NON-NEGOTIABLE)

- **NO BACKWARDS COMPATIBILITY** - per projectionplan1.md and constitution.yaml
- **NO COMPATIBILITY SHIMS** - delete, don't deprecate
- **NO STUB IMPLEMENTATIONS** - if code needs fusion, delete that code
- **NO WORKAROUNDS** - if something breaks, it breaks loudly with clear errors
- **FAIL FAST** - any remaining fusion reference should cause compilation failure
- Remove files in dependency order to minimize intermediate errors
- Commit as single atomic commit for easy revert if needed

## Git Commit Message

```
refactor!: remove all fusion-related code (TASK-F006)

BREAKING CHANGE: Complete removal of FuseMoE, gating networks, and
single-vector fusion. The Multi-Array Teleological Fingerprint
architecture stores all 13 embeddings without fusion.

Removed:
- crates/context-graph-embeddings/src/fusion/ (29 files)
- crates/context-graph-embeddings/src/config/fusion.rs
- crates/context-graph-embeddings/src/provider/fused.rs
- crates/context-graph-embeddings/src/types/dimensions/fusemoe.rs

Modified:
- lib.rs: removed fusion exports
- config/mod.rs: removed FusionConfig
- provider/mod.rs: removed FusedEmbeddingProvider
- storage/*: removed fusion type references
- MCP adapter: [deleted|updated] for new architecture

No migration path provided (per specification).
All 13 embeddings now stored as SemanticFingerprint array.

Traces To: TS-601, FR-601, FR-602

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## Why This Removal is Correct

| Old Architecture | New Architecture | Benefit |
|-----------------|------------------|---------|
| FuseMoE top-k=4 | Store all 13 embeddings | 100% vs 33% information retention |
| 1536D single vector | 13× separate vectors | Cross-space pattern visibility |
| Gating complexity | Simple array storage | Reduced failure modes |
| Single similarity metric | RRF across 13 spaces | Better retrieval quality |

**"The pattern across embedding spaces reveals purpose"** - you can't see the pattern with fusion.

## Risk Mitigation

- Removal is reversible via git
- No production data depends on fusion (new architecture)
- Tests verify remaining code still works
- Compilation will fail if any fusion code remains (desired behavior)

## Related Tasks

- **TASK-F001**: SemanticFingerprint (13 embeddings) - COMPLETED
- **TASK-F003**: JohariFingerprint (13 embedders) - COMPLETED
- **TASK-F004**: Storage Schema (8 CFs) - COMPLETED
- **TASK-F007**: MultiArrayEmbeddingProvider trait - PENDING (needed to replace FusedProvider)
- **TASK-S007**: Remove Fused MCP Handlers - PENDING (depends on this task)

Reference: projectionplan1.md Section 15.1, constitution.yaml embeddings.paradigm
