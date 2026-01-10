# TASK-CORE-002: Define Canonical Embedder Enumeration

```xml
<task_spec id="TASK-CORE-002" version="3.0">
<metadata>
  <title>Define Canonical Embedder Enumeration with Dimension Metadata</title>
  <status>completed</status>
  <completed_date>2026-01-09</completed_date>
  <layer>foundation</layer>
  <sequence>2</sequence>
  <implements>
    <requirement_ref>REQ-TELEOLOGICAL-01</requirement_ref>
    <requirement_ref>REQ-EMBEDDER-TYPES-01</requirement_ref>
    <requirement_ref>ARCH-02: Apples-to-apples comparison only</requirement_ref>
    <requirement_ref>ARCH-05: All 13 Embedders Must Be Present</requirement_ref>
  </implements>
  <depends_on><!-- None - can start immediately --></depends_on>
  <blocks>
    <task_ref>TASK-CORE-003</task_ref><!-- TeleologicalArray uses Embedder -->
    <task_ref>TASK-CORE-004</task_ref><!-- Comparison types use Embedder -->
    <task_ref>TASK-LOGIC-001</task_ref><!-- Similarity functions reference Embedder -->
  </blocks>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<completion_summary>
## STATUS: COMPLETED

This task has been fully implemented. The canonical `Embedder` enum exists at:
**`crates/context-graph-core/src/teleological/embedder.rs`** (552 lines)

### What Was Done
1. Created `Embedder` enum with all 13 variants (E1-E13)
2. Created `EmbedderDims` enum (Dense, Sparse, TokenLevel)
3. Created `EmbedderMask` bitmask type for embedder subsets
4. Created `EmbedderGroup` enum for predefined groupings
5. Implemented all required methods: `index()`, `from_index()`, `expected_dims()`, `all()`, `name()`, `short_name()`
6. Removed duplicate Embedder enum from `atc/level2_temperature.rs`
7. Updated `atc/level2_temperature.rs` to import from teleological module
8. Updated `teleological/mod.rs` with `pub mod embedder` and re-exports
9. Updated `lib.rs` with crate-level re-exports
10. All 53 tests pass

### Verification Commands (Run These to Confirm)
```bash
# 1. Verify compilation
cargo check -p context-graph-core

# 2. Verify single definition (expect ONLY teleological/embedder.rs)
rg "pub enum Embedder\b" --type rust crates/context-graph-core/

# 3. Run tests
cargo test -p context-graph-core embedder -- --nocapture

# 4. Temperature tests still pass
cargo test -p context-graph-core temperature -- --nocapture
```

### Test Results (2026-01-09)
```
test result: ok. 53 passed; 0 failed
test teleological::embedder::tests::test_embedder_count ... ok
test teleological::embedder::tests::test_index_roundtrip ... ok
test teleological::embedder::tests::test_index_bounds ... ok
test teleological::embedder::tests::test_expected_dims_match_constants ... ok
test teleological::embedder::tests::test_names ... ok
test teleological::embedder::tests::test_embedder_mask_operations ... ok
test teleological::embedder::tests::test_embedder_mask_all ... ok
test teleological::embedder::tests::test_embedder_mask_iter ... ok
test teleological::embedder::tests::test_embedder_group_temporal ... ok
test teleological::embedder::tests::test_embedder_group_dense ... ok
test teleological::embedder::tests::test_embedder_serde ... ok
test teleological::embedder::tests::test_embedder_mask_serde ... ok
test teleological::embedder::tests::test_type_classification ... ok
test teleological::embedder::tests::test_default_temperature ... ok
test teleological::embedder::tests::test_display ... ok
test teleological::embedder::tests::test_embedder_dims_primary_dim ... ok
```
</completion_summary>

<current_codebase_state>
## File Locations (Verified 2026-01-09)

### Primary Implementation
| File | Purpose | Lines |
|------|---------|-------|
| `crates/context-graph-core/src/teleological/embedder.rs` | **CANONICAL Embedder enum** | 552 |
| `crates/context-graph-core/src/teleological/mod.rs` | Re-exports embedder module | 108 |
| `crates/context-graph-core/src/lib.rs` | Crate-level re-exports | ~75 |
| `crates/context-graph-core/src/atc/level2_temperature.rs` | Uses Embedder via import | 324 |

### Related Types (NOT duplicates - different purposes)
| File | Type | Purpose |
|------|------|---------|
| `crates/context-graph-core/src/index/config.rs` | `EmbedderIndex` | Storage layer HNSW index identifiers (includes E1Matryoshka128) |

### Dimension Constants (Source of Truth)
| File | Contains |
|------|----------|
| `crates/context-graph-core/src/types/fingerprint/mod.rs` | Re-exports: E1_DIM through E13_SPLADE_VOCAB |
| `crates/context-graph-core/src/types/fingerprint/semantic/constants.rs` | Original definitions |

### Dimension Values
| Embedder | Constant | Value | Type |
|----------|----------|-------|------|
| E1 | E1_DIM | 1024 | Dense |
| E2 | E2_DIM | 512 | Dense |
| E3 | E3_DIM | 512 | Dense |
| E4 | E4_DIM | 512 | Dense |
| E5 | E5_DIM | 768 | Dense |
| E6 | E6_SPARSE_VOCAB | 30522 | Sparse |
| E7 | E7_DIM | 1536 | Dense |
| E8 | E8_DIM | 384 | Dense |
| E9 | E9_DIM | 1024 | Dense (projected from 10K binary) |
| E10 | E10_DIM | 768 | Dense |
| E11 | E11_DIM | 384 | Dense |
| E12 | E12_TOKEN_DIM | 128 | TokenLevel (per token) |
| E13 | E13_SPLADE_VOCAB | 30522 | Sparse |
</current_codebase_state>

<how_to_use>
## Usage Examples

### Import the Embedder
```rust
// From within context-graph-core crate
use crate::teleological::Embedder;
use crate::teleological::{EmbedderDims, EmbedderMask, EmbedderGroup};

// From other crates
use context_graph_core::Embedder;
use context_graph_core::{EmbedderDims, EmbedderMask, EmbedderGroup};
```

### Index Operations
```rust
// Get index (0-12)
let idx = Embedder::Semantic.index(); // 0
let idx = Embedder::KeywordSplade.index(); // 12

// Create from index
let e = Embedder::from_index(4); // Some(Embedder::Causal)
let e = Embedder::from_index(13); // None

// Iterate all
for embedder in Embedder::all() {
    println!("{}: {}", embedder.short_name(), embedder.name());
}
```

### Dimension Queries
```rust
let dims = Embedder::Semantic.expected_dims();
assert_eq!(dims, EmbedderDims::Dense(1024));

let dims = Embedder::Sparse.expected_dims();
assert_eq!(dims, EmbedderDims::Sparse { vocab_size: 30522 });

// Type classification
assert!(Embedder::Semantic.is_dense());
assert!(Embedder::Sparse.is_sparse());
assert!(Embedder::LateInteraction.is_token_level());
```

### Embedder Masks
```rust
let mut mask = EmbedderMask::new();
mask.set(Embedder::Semantic);
mask.set(Embedder::Causal);
assert_eq!(mask.count(), 2);
assert!(mask.contains(Embedder::Semantic));

// From predefined groups
let temporal = EmbedderGroup::Temporal.embedders();
assert!(temporal.contains(Embedder::TemporalRecent));
assert!(temporal.contains(Embedder::TemporalPeriodic));
assert!(temporal.contains(Embedder::TemporalPositional));
```

### Temperature (ATC)
```rust
// Default temperatures from constitution.yaml
let t = Embedder::Causal.default_temperature(); // 1.2
let t = Embedder::Code.default_temperature(); // 0.9

// Temperature ranges (ATC-specific function)
use crate::atc::level2_temperature::embedder_temperature_range;
let (min, max) = embedder_temperature_range(Embedder::Causal); // (0.8, 2.5)
```
</how_to_use>

<api_reference>
## Complete API

### Embedder Enum
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Embedder {
    Semantic = 0,        // E1: 1024D dense
    TemporalRecent = 1,  // E2: 512D dense
    TemporalPeriodic = 2, // E3: 512D dense
    TemporalPositional = 3, // E4: 512D dense
    Causal = 4,          // E5: 768D dense (ASYMMETRIC)
    Sparse = 5,          // E6: ~30K sparse
    Code = 6,            // E7: 1536D dense
    Graph = 7,           // E8: 384D dense
    Hdc = 8,             // E9: 1024D dense (projected from 10K binary)
    Multimodal = 9,      // E10: 768D dense
    Entity = 10,         // E11: 384D dense
    LateInteraction = 11, // E12: 128D per token
    KeywordSplade = 12,  // E13: ~30K sparse
}

impl Embedder {
    pub const COUNT: usize = 13;
    pub fn index(self) -> usize;
    pub fn from_index(idx: usize) -> Option<Self>;
    pub fn expected_dims(self) -> EmbedderDims;
    pub fn all() -> impl Iterator<Item = Embedder> + ExactSizeIterator;
    pub fn name(self) -> &'static str;
    pub fn short_name(self) -> &'static str;
    pub fn is_dense(self) -> bool;
    pub fn is_sparse(self) -> bool;
    pub fn is_token_level(self) -> bool;
    pub fn default_temperature(self) -> f32;
}

impl Display for Embedder { ... } // Uses name()
```

### EmbedderDims Enum
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderDims {
    Dense(usize),
    Sparse { vocab_size: usize },
    TokenLevel { per_token: usize },
}

impl EmbedderDims {
    pub fn primary_dim(&self) -> usize;
}
```

### EmbedderMask Struct
```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbedderMask(u16);

impl EmbedderMask {
    pub fn new() -> Self;
    pub fn all() -> Self;
    pub fn from_slice(embedders: &[Embedder]) -> Self;
    pub fn set(&mut self, embedder: Embedder);
    pub fn unset(&mut self, embedder: Embedder);
    pub fn contains(self, embedder: Embedder) -> bool;
    pub fn iter(self) -> impl Iterator<Item = Embedder>;
    pub fn count(self) -> usize;
    pub fn is_empty(self) -> bool;
    pub fn as_u16(self) -> u16;
}
```

### EmbedderGroup Enum
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbedderGroup {
    Temporal,       // E2, E3, E4
    Relational,     // E4, E5, E11
    Lexical,        // E6, E12, E13
    Dense,          // All except E6, E12, E13
    Factual,        // E1, E12, E13
    Implementation, // E7 only
    All,            // All 13
}

impl EmbedderGroup {
    pub fn embedders(self) -> EmbedderMask;
}
```
</api_reference>

<downstream_tasks>
## Tasks That Depend on This (Now Unblocked)

1. **TASK-CORE-003: Define TeleologicalArray Type**
   - Can now use `Embedder::COUNT` for array sizing
   - Can use `EmbedderMask` for sparse operations

2. **TASK-CORE-004: Define Comparison Types**
   - Can use `Embedder` to constrain comparison operands
   - `EmbedderDims` helps validate compatible comparisons

3. **TASK-LOGIC-001: Dense Similarity Functions**
   - Can use `Embedder::is_dense()` to filter
   - Can use `Embedder::expected_dims()` for dimension validation

4. **TASK-LOGIC-013: Search Result Caching**
   - Can use `EmbedderGroup` for cache partitioning
</downstream_tasks>

<no_action_required>
## No Further Action Required

This task is COMPLETE. The implementation:
- Passes all 53 tests
- Has no duplicate definitions in context-graph-core (only one `pub enum Embedder`)
- Is properly re-exported from teleological/mod.rs and lib.rs
- Is used by atc/level2_temperature.rs via import (no local definition)
- Has comprehensive inline documentation
- Follows NO BACKWARDS COMPATIBILITY requirement (no deprecated shims)

### To Verify Completion
```bash
cd /home/cabdru/contextgraph
cargo test -p context-graph-core embedder -- --nocapture
rg "pub enum Embedder\b" --type rust crates/context-graph-core/
# Should only show: crates/context-graph-core/src/teleological/embedder.rs
```
</no_action_required>

</task_spec>
```
