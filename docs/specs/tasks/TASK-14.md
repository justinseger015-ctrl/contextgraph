# TASK-14: Implement TokenPruningConfig types

```xml
<task_spec id="TASK-EMBED-002" version="2.0">
<metadata>
  <title>Implement TokenPruningConfig types</title>
  <status>complete</status>
  <layer>logic</layer>
  <sequence>14</sequence>
  <implements><requirement_ref>REQ-EMBED-002</requirement_ref></implements>
  <depends_on></depends_on>
  <blocks>TASK-15</blocks>
  <estimated_hours>2</estimated_hours>
</metadata>

<context>
Token pruning reduces E12 Late Interaction (ColBERT) embedding size by ~50% while maintaining
recall quality. This task defines the configuration types and scoring methods before
implementing the actual quantizer in TASK-15.

E12 produces 128D vectors PER TOKEN (not a single vector). For a 512-token input,
this is 512 × 128 × 4 bytes = 262KB per document. Token pruning reduces this to ~131KB
by removing low-importance tokens while retaining semantic recall.
</context>

<critical_context_for_implementation>
## CURRENT CODEBASE STATE (Verified 2026-01-13)

### E12 Model Location and Structure
The E12 (Late Interaction/ColBERT) model is at:
`crates/context-graph-embeddings/src/models/pretrained/late_interaction/`

Key files:
- `types.rs` - Defines `TokenEmbeddings` struct with `vectors: Vec<Vec<f32>>`, `tokens: Vec<String>`, `mask: Vec<bool>`
- `mod.rs` - Re-exports `LateInteractionModel`, `TokenEmbeddings`, constants
- Constants: `LATE_INTERACTION_DIMENSION = 128`, `LATE_INTERACTION_MAX_TOKENS = 512`

### TokenEmbeddings Structure (SOURCE OF TRUTH)
```rust
// From crates/context-graph-embeddings/src/models/pretrained/late_interaction/types.rs
pub struct TokenEmbeddings {
    pub vectors: Vec<Vec<f32>>,  // [num_tokens, 128]
    pub tokens: Vec<String>,     // Token strings
    pub mask: Vec<bool>,         // Valid (non-padding) tokens
}
```

### Existing Error Types (DO NOT DUPLICATE)
`crates/context-graph-embeddings/src/error/types.rs` already defines:
- `EmbeddingError::ConfigError { message: String }` - USE THIS for validation errors
- `EmbeddingError::InvalidDimension { expected, actual }` - USE THIS for dimension errors
- `EmbeddingError::EmptyInput` - USE THIS for empty input errors

DO NOT create a new `ConfigError` type. Use the existing `EmbeddingError` variants.

### Module Structure Pattern
The crate uses a module directory pattern. Look at existing examples:
- `src/quantization/mod.rs` exports from `src/quantization/pq8/`, `src/quantization/router/`
- `src/error/mod.rs` exports from `src/error/types.rs`

Follow this pattern: Create `src/pruning/mod.rs` that exports from `src/pruning/config.rs`.

### Cargo.toml - NO CHANGES NEEDED
The existing dependencies are sufficient. Do not add new dependencies.
</critical_context_for_implementation>

<scope>
<in_scope>
- Create pruning module directory: `crates/context-graph-embeddings/src/pruning/`
- Define `TokenPruningConfig` struct with validation
- Define `ImportanceScoringMethod` enum
- Define `PrunedEmbeddings` result struct
- Add pruning module export to `lib.rs`
- Add unit tests for config validation
</in_scope>
<out_of_scope>
- Actual pruning implementation (TASK-15)
- Integration with E12 model
- Attention weight extraction from transformer
</out_of_scope>
</scope>

<definition_of_done>
<signatures>
```rust
// crates/context-graph-embeddings/src/pruning/config.rs

use crate::error::{EmbeddingError, EmbeddingResult};

/// Configuration for token pruning of E12 (ColBERT) embeddings.
///
/// Token pruning reduces the number of per-token embeddings while
/// maintaining semantic recall quality. Constitution target: ~50% compression.
#[derive(Debug, Clone)]
pub struct TokenPruningConfig {
    /// Target compression ratio (default: 0.5 = 50% compression)
    /// Range: (0.0, 1.0) exclusive - 0.0 means no compression, 1.0 means remove all
    pub target_compression: f32,

    /// Minimum tokens to retain (default: 64)
    /// Prevents over-pruning short sequences
    pub min_tokens: usize,

    /// Importance scoring method for ranking tokens
    pub scoring_method: ImportanceScoringMethod,
}

impl Default for TokenPruningConfig {
    fn default() -> Self {
        Self {
            target_compression: 0.5,
            min_tokens: 64,
            scoring_method: ImportanceScoringMethod::AttentionBased,
        }
    }
}

impl TokenPruningConfig {
    /// Validate configuration.
    ///
    /// # Errors
    /// Returns `EmbeddingError::ConfigError` if:
    /// - target_compression not in (0.0, 1.0) exclusive
    /// - min_tokens is 0
    pub fn validate(&self) -> EmbeddingResult<()> {
        if self.target_compression <= 0.0 || self.target_compression >= 1.0 {
            return Err(EmbeddingError::ConfigError {
                message: format!(
                    "target_compression must be in (0.0, 1.0), got {}",
                    self.target_compression
                ),
            });
        }
        if self.min_tokens == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "min_tokens must be at least 1".to_string(),
            });
        }
        Ok(())
    }

    /// Create a new config with custom compression ratio.
    /// Validates immediately - fails fast if invalid.
    pub fn with_compression(target_compression: f32) -> EmbeddingResult<Self> {
        let config = Self {
            target_compression,
            ..Default::default()
        };
        config.validate()?;
        Ok(config)
    }
}

/// Method for scoring token importance during pruning.
///
/// Different methods have different accuracy/performance tradeoffs:
/// - AttentionBased: Best accuracy, requires attention weights
/// - EmbeddingMagnitude: Fast, no additional data needed
/// - Entropy: Good for diverse token selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImportanceScoringMethod {
    /// Use attention weights from transformer layers (most accurate)
    #[default]
    AttentionBased,

    /// Use L2 norm of token embeddings (fastest, no extra data)
    EmbeddingMagnitude,

    /// Use entropy of token probability distribution (moderate)
    Entropy,
}

/// Result of token pruning operation.
///
/// Contains the pruned embeddings and metadata about what was retained.
#[derive(Debug, Clone)]
pub struct PrunedEmbeddings {
    /// Pruned token embeddings - each inner Vec is 128D
    /// Length: retained_indices.len()
    pub embeddings: Vec<Vec<f32>>,

    /// Indices of retained tokens in original sequence
    /// Sorted in ascending order to preserve positional information
    pub retained_indices: Vec<usize>,

    /// Achieved compression ratio [0, 1]
    /// 0.0 = no compression, 1.0 = all tokens removed (impossible)
    pub compression_ratio: f32,
}

impl PrunedEmbeddings {
    /// Number of tokens after pruning.
    pub fn token_count(&self) -> usize {
        self.embeddings.len()
    }

    /// Memory size in bytes (for monitoring).
    pub fn memory_bytes(&self) -> usize {
        self.embeddings.len() * 128 * std::mem::size_of::<f32>()
    }
}
```
</signatures>

<constraints>
- target_compression MUST be in (0.0, 1.0) exclusive - enforce via validate()
- min_tokens MUST be at least 1 - enforce via validate()
- Default compression MUST be 0.5 (50%)
- Default min_tokens MUST be 64
- Default scoring_method MUST be AttentionBased
- Use existing EmbeddingError::ConfigError - DO NOT create new error types
- Validation MUST fail fast with descriptive error messages
</constraints>

<files_to_create>
- crates/context-graph-embeddings/src/pruning/mod.rs
- crates/context-graph-embeddings/src/pruning/config.rs
</files_to_create>

<files_to_modify>
- crates/context-graph-embeddings/src/lib.rs (add `pub mod pruning;` after line 64)
</files_to_modify>

<verification>
```bash
# 1. Check compilation
cargo check -p context-graph-embeddings

# 2. Run pruning config tests
cargo test -p context-graph-embeddings pruning --no-fail-fast

# 3. Verify no new warnings
cargo clippy -p context-graph-embeddings -- -D warnings
```
</verification>
</definition_of_done>

<test_requirements>
## Required Test Cases

### File: crates/context-graph-embeddings/src/pruning/config.rs (at bottom)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // === TokenPruningConfig Tests ===

    #[test]
    fn test_default_config_is_valid() {
        let config = TokenPruningConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.target_compression, 0.5);
        assert_eq!(config.min_tokens, 64);
        assert_eq!(config.scoring_method, ImportanceScoringMethod::AttentionBased);
    }

    #[test]
    fn test_compression_at_zero_fails() {
        let config = TokenPruningConfig {
            target_compression: 0.0,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, EmbeddingError::ConfigError { .. }));
    }

    #[test]
    fn test_compression_at_one_fails() {
        let config = TokenPruningConfig {
            target_compression: 1.0,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, EmbeddingError::ConfigError { .. }));
    }

    #[test]
    fn test_compression_negative_fails() {
        let config = TokenPruningConfig {
            target_compression: -0.1,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, EmbeddingError::ConfigError { .. }));
    }

    #[test]
    fn test_compression_above_one_fails() {
        let config = TokenPruningConfig {
            target_compression: 1.5,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, EmbeddingError::ConfigError { .. }));
    }

    #[test]
    fn test_min_tokens_zero_fails() {
        let config = TokenPruningConfig {
            min_tokens: 0,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, EmbeddingError::ConfigError { .. }));
    }

    #[test]
    fn test_valid_edge_cases() {
        // Just above 0.0
        let config = TokenPruningConfig {
            target_compression: 0.001,
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        // Just below 1.0
        let config = TokenPruningConfig {
            target_compression: 0.999,
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        // Min tokens = 1
        let config = TokenPruningConfig {
            min_tokens: 1,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_with_compression_valid() {
        let config = TokenPruningConfig::with_compression(0.7).unwrap();
        assert_eq!(config.target_compression, 0.7);
    }

    #[test]
    fn test_with_compression_invalid() {
        assert!(TokenPruningConfig::with_compression(0.0).is_err());
        assert!(TokenPruningConfig::with_compression(1.0).is_err());
        assert!(TokenPruningConfig::with_compression(-0.5).is_err());
    }

    // === ImportanceScoringMethod Tests ===

    #[test]
    fn test_scoring_method_default() {
        assert_eq!(
            ImportanceScoringMethod::default(),
            ImportanceScoringMethod::AttentionBased
        );
    }

    #[test]
    fn test_scoring_method_equality() {
        assert_eq!(
            ImportanceScoringMethod::AttentionBased,
            ImportanceScoringMethod::AttentionBased
        );
        assert_ne!(
            ImportanceScoringMethod::AttentionBased,
            ImportanceScoringMethod::EmbeddingMagnitude
        );
    }

    // === PrunedEmbeddings Tests ===

    #[test]
    fn test_pruned_embeddings_token_count() {
        let pruned = PrunedEmbeddings {
            embeddings: vec![vec![0.0; 128]; 10],
            retained_indices: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            compression_ratio: 0.5,
        };
        assert_eq!(pruned.token_count(), 10);
    }

    #[test]
    fn test_pruned_embeddings_memory_bytes() {
        let pruned = PrunedEmbeddings {
            embeddings: vec![vec![0.0; 128]; 10],
            retained_indices: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            compression_ratio: 0.5,
        };
        // 10 tokens × 128 dims × 4 bytes = 5120 bytes
        assert_eq!(pruned.memory_bytes(), 5120);
    }

    #[test]
    fn test_pruned_embeddings_empty() {
        let pruned = PrunedEmbeddings {
            embeddings: vec![],
            retained_indices: vec![],
            compression_ratio: 1.0,
        };
        assert_eq!(pruned.token_count(), 0);
        assert_eq!(pruned.memory_bytes(), 0);
    }
}
```
</test_requirements>

<full_state_verification>
## MANDATORY: Full State Verification Protocol

After completing the implementation, you MUST perform these verification steps:

### 1. Source of Truth Identification
The source of truth for this task is:
- **File existence**: `crates/context-graph-embeddings/src/pruning/mod.rs` and `config.rs` MUST exist
- **Module export**: `pruning` module MUST be exported from `lib.rs`
- **Compilation**: `cargo check -p context-graph-embeddings` MUST succeed with zero errors

### 2. Execute & Inspect Protocol
```bash
# Step 1: Verify files exist
ls -la crates/context-graph-embeddings/src/pruning/

# Expected output:
# mod.rs
# config.rs

# Step 2: Verify module is exported
grep -n "pub mod pruning" crates/context-graph-embeddings/src/lib.rs

# Expected: Should find "pub mod pruning;" on some line

# Step 3: Verify compilation
cargo check -p context-graph-embeddings 2>&1 | grep -E "(error|warning)"

# Expected: No errors, warnings acceptable

# Step 4: Run all tests
cargo test -p context-graph-embeddings pruning -- --nocapture

# Expected: All tests pass (13+ tests)
```

### 3. Boundary & Edge Case Audit
Execute these manual tests and record before/after state:

**Test Case 1: Compression at boundary 0.0**
```rust
// Input state: config with target_compression = 0.0
let config = TokenPruningConfig { target_compression: 0.0, ..Default::default() };
println!("Before validate(): target_compression = {}", config.target_compression);
let result = config.validate();
println!("After validate(): result = {:?}", result);
// EXPECTED: Err(ConfigError) with message about (0.0, 1.0) range
```

**Test Case 2: Compression at boundary 1.0**
```rust
// Input state: config with target_compression = 1.0
let config = TokenPruningConfig { target_compression: 1.0, ..Default::default() };
println!("Before validate(): target_compression = {}", config.target_compression);
let result = config.validate();
println!("After validate(): result = {:?}", result);
// EXPECTED: Err(ConfigError) with message about (0.0, 1.0) range
```

**Test Case 3: min_tokens = 0**
```rust
// Input state: config with min_tokens = 0
let config = TokenPruningConfig { min_tokens: 0, ..Default::default() };
println!("Before validate(): min_tokens = {}", config.min_tokens);
let result = config.validate();
println!("After validate(): result = {:?}", result);
// EXPECTED: Err(ConfigError) with message about min_tokens >= 1
```

### 4. Evidence of Success
After implementation, provide a log showing:
```bash
# Final verification command
cargo test -p context-graph-embeddings pruning -- --nocapture 2>&1 | tail -30

# Expected output format:
# running 13 tests
# test pruning::config::tests::test_default_config_is_valid ... ok
# test pruning::config::tests::test_compression_at_zero_fails ... ok
# test pruning::config::tests::test_compression_at_one_fails ... ok
# ... (all tests)
# test result: ok. 13 passed; 0 failed; 0 ignored
```
</full_state_verification>

<anti_patterns>
## DO NOT DO THESE THINGS

1. **DO NOT create new error types** - Use existing `EmbeddingError::ConfigError`
2. **DO NOT use mock data in tests** - Use real values and verify actual behavior
3. **DO NOT add fallbacks** - Validation failures MUST return errors, not default values
4. **DO NOT modify E12 model** - This task is types-only, implementation is TASK-15
5. **DO NOT add dependencies** - Existing Cargo.toml has everything needed
6. **DO NOT create workarounds** - If something doesn't work, fix the root cause
7. **DO NOT use unwrap()** - Use `?` operator or explicit error handling
</anti_patterns>

<implementation_checklist>
## Step-by-Step Implementation

### Step 1: Create pruning module directory
```bash
mkdir -p crates/context-graph-embeddings/src/pruning
```

### Step 2: Create mod.rs
Create `crates/context-graph-embeddings/src/pruning/mod.rs`:
```rust
//! Token pruning for E12 (ColBERT) late-interaction embeddings.
//!
//! This module provides configuration types for token pruning,
//! which reduces embedding size by ~50% while maintaining recall quality.

mod config;

pub use config::{ImportanceScoringMethod, PrunedEmbeddings, TokenPruningConfig};
```

### Step 3: Create config.rs
Create `crates/context-graph-embeddings/src/pruning/config.rs` with the signatures and tests from this spec.

### Step 4: Export from lib.rs
Add to `crates/context-graph-embeddings/src/lib.rs` after line 64 (after `pub mod warm;`):
```rust
pub mod pruning;
```

And add to re-exports section:
```rust
// Pruning re-exports
pub use pruning::{ImportanceScoringMethod, PrunedEmbeddings, TokenPruningConfig};
```

### Step 5: Verify compilation
```bash
cargo check -p context-graph-embeddings
```

### Step 6: Run tests
```bash
cargo test -p context-graph-embeddings pruning
```

### Step 7: Run clippy
```bash
cargo clippy -p context-graph-embeddings -- -D warnings
```
</implementation_checklist>

<constitution_references>
- embeddings.models.E12_LateInteraction: "128D/tok, dense_per_token"
- embeddings.paradigm: "NO FUSION - Store all 13 embeddings"
- perf.quality.info_loss: "<15%"
- rules: "Result<T,E>, thiserror derivation"
- rules: "Never unwrap() in prod"
</constitution_references>
</task_spec>
```

## Historical Context

This task was originally TASK-EMBED-002 and is now TASK-14 in the sequential numbering system. It is part of the ISS-009 remediation (TokenPruning for E12 missing).

## Related Tasks
- **TASK-13** (TASK-EMBED-001): Green Contexts auto-enable - Independent, can run in parallel
- **TASK-15** (TASK-EMBED-003): TokenPruningQuantizer implementation - DEPENDS ON THIS TASK

## Git Context
Recent commits show the project has completed:
- TASK-12: KuramotoStepper wired to MCP server lifecycle
- TASK-10/11: 13-oscillator Kuramoto network implementation
- TASK-07/08/09: Async provider conversions and Johari fixes

The pruning module does NOT exist in git history - this is new code.
