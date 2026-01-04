---
id: "M04-T14a"
title: "Add Result-Returning NT Weight Validation Wrapper"
description: |
  context-graph-core has validate() -> bool.
  This task adds a Graph-layer wrapper: validate_or_error() -> GraphResult<()>
  Uses existing GraphError::InvalidNtWeights for detailed error messages.
  NO BACKWARDS COMPATIBILITY - fail fast with robust error logging.
layer: "logic"
status: "pending"
priority: "high"
estimated_hours: 1
sequence: 20
depends_on:
  - "M04-T14"
spec_refs:
  - "TECH-GRAPH-004 Section 4.1"
  - "REQ-KG-065"
  - "constitution.yaml AP-001"
files_to_create:
  - path: "crates/context-graph-graph/src/marblestone/validation.rs"
    description: "NT weight validation wrapper with GraphError integration"
files_to_modify:
  - path: "crates/context-graph-graph/src/marblestone/mod.rs"
    description: "Add validation module"
test_file: "crates/context-graph-graph/tests/nt_validation_tests.rs"
---

## ⚠️ CRITICAL: Current Codebase State

### What EXISTS (DO NOT RECREATE)

| Component | Location | Returns |
|-----------|----------|---------|
| `NeurotransmitterWeights::validate()` | `context-graph-core/.../neurotransmitter_weights.rs:148` | `bool` |
| `GraphError::InvalidNtWeights` | `context-graph-graph/src/error.rs:148` | Error variant |
| `NeurotransmitterWeights` re-export | `context-graph-graph/src/lib.rs:60` | - |

### What This Task ADDS

| Component | Location | Returns |
|-----------|----------|---------|
| `validate_or_error()` function | `context-graph-graph/src/marblestone/validation.rs` | `GraphResult<()>` |

### Existing Error Variant (USE THIS)
```rust
// In crates/context-graph-graph/src/error.rs line 148
#[error("Invalid NT weights: {field} = {value} (must be in [0.0, 1.0])")]
InvalidNtWeights { field: String, value: f32 },
```

## Context

The core crate provides `validate() -> bool` for simple validation checks. The graph layer needs `GraphResult<()>` for proper error propagation through the ? operator. This task bridges the gap by creating a validation wrapper that:
1. Uses existing `validate()` logic
2. Returns `GraphError::InvalidNtWeights` with field-specific details
3. Follows fail-fast principle (AP-001)

### Constitution Reference
- `AP-001`: Never unwrap() in prod - all errors properly typed
- `edge_model.nt_weights`: All weights in [0.0, 1.0]

## Scope

### In Scope
- Create `validation.rs` in `context-graph-graph/src/marblestone/`
- Implement `validate_or_error(weights: &NeurotransmitterWeights) -> GraphResult<()>`
- Use `GraphError::InvalidNtWeights` with specific field names
- NO MOCK DATA in tests - use real domain profiles

### Out of Scope
- Modifying `context-graph-core` (existing code)
- Auto-clamping or normalization
- GraphEdge integration (M04-T15)

## Definition of Done

### Signatures

```rust
// In crates/context-graph-graph/src/marblestone/validation.rs

use crate::error::{GraphError, GraphResult};
use context_graph_core::marblestone::NeurotransmitterWeights;

/// Validate NT weights and return detailed error on failure.
///
/// Wraps `NeurotransmitterWeights::validate()` with GraphError for
/// proper error propagation in graph operations.
///
/// # Arguments
/// * `weights` - NT weights to validate
///
/// # Returns
/// * `Ok(())` - All weights in [0.0, 1.0]
/// * `Err(GraphError::InvalidNtWeights)` - First invalid weight with details
///
/// # Example
/// ```rust
/// use context_graph_graph::marblestone::validate_or_error;
/// use context_graph_graph::NeurotransmitterWeights;
///
/// let valid = NeurotransmitterWeights::new(0.5, 0.3, 0.4);
/// assert!(validate_or_error(&valid).is_ok());
///
/// let invalid = NeurotransmitterWeights::new(1.5, 0.0, 0.0);
/// let err = validate_or_error(&invalid).unwrap_err();
/// // err contains field="excitatory", value=1.5
/// ```
pub fn validate_or_error(weights: &NeurotransmitterWeights) -> GraphResult<()> {
    // Check excitatory
    if weights.excitatory < 0.0 || weights.excitatory > 1.0 ||
       weights.excitatory.is_nan() || weights.excitatory.is_infinite() {
        log::error!(
            "VALIDATION FAILED: excitatory={} (must be in [0.0, 1.0])",
            weights.excitatory
        );
        return Err(GraphError::InvalidNtWeights {
            field: "excitatory".to_string(),
            value: weights.excitatory,
        });
    }

    // Check inhibitory
    if weights.inhibitory < 0.0 || weights.inhibitory > 1.0 ||
       weights.inhibitory.is_nan() || weights.inhibitory.is_infinite() {
        log::error!(
            "VALIDATION FAILED: inhibitory={} (must be in [0.0, 1.0])",
            weights.inhibitory
        );
        return Err(GraphError::InvalidNtWeights {
            field: "inhibitory".to_string(),
            value: weights.inhibitory,
        });
    }

    // Check modulatory
    if weights.modulatory < 0.0 || weights.modulatory > 1.0 ||
       weights.modulatory.is_nan() || weights.modulatory.is_infinite() {
        log::error!(
            "VALIDATION FAILED: modulatory={} (must be in [0.0, 1.0])",
            weights.modulatory
        );
        return Err(GraphError::InvalidNtWeights {
            field: "modulatory".to_string(),
            value: weights.modulatory,
        });
    }

    log::debug!(
        "Validation passed: e={}, i={}, m={}",
        weights.excitatory, weights.inhibitory, weights.modulatory
    );
    Ok(())
}

/// Validate and compute effective weight in one call.
///
/// Fails fast if weights are invalid, otherwise computes effective weight.
///
/// # Arguments
/// * `weights` - NT weights to validate and use
/// * `base_weight` - Base edge weight [0.0, 1.0]
///
/// # Returns
/// * `Ok(f32)` - Effective weight after modulation
/// * `Err(GraphError::InvalidNtWeights)` - Validation failed
pub fn compute_effective_validated(
    weights: &NeurotransmitterWeights,
    base_weight: f32,
) -> GraphResult<f32> {
    validate_or_error(weights)?;
    Ok(weights.compute_effective_weight(base_weight))
}
```

### Module Registration

```rust
// In crates/context-graph-graph/src/marblestone/mod.rs

mod validation;

pub use validation::{validate_or_error, compute_effective_validated};
```

### Constraints
- NO auto-clamping - fail fast on invalid weights
- Check order: excitatory → inhibitory → modulatory (first failure wins)
- Handle NaN and Infinity as invalid
- Use `log::error!` for failures, `log::debug!` for success

### Acceptance Criteria
- [ ] `validate_or_error(&valid_weights).is_ok()` returns true
- [ ] `validate_or_error(&invalid_weights)` returns `Err(InvalidNtWeights)`
- [ ] Error contains correct field name and value
- [ ] NaN/Infinity treated as invalid
- [ ] Boundary values (0.0, 1.0) accepted
- [ ] Compiles with `cargo build`
- [ ] Tests pass with `cargo test`
- [ ] No clippy warnings

### Verification Commands
```bash
cargo build -p context-graph-graph
cargo test -p context-graph-graph nt_validation -- --nocapture
cargo clippy -p context-graph-graph -- -D warnings
```

## Implementation Approach

### Step 1: Create validation.rs
Create new file at `crates/context-graph-graph/src/marblestone/validation.rs` with the signature above.

### Step 2: Update mod.rs
Add module declaration and re-export in `crates/context-graph-graph/src/marblestone/mod.rs`.

### Step 3: Write Tests (NO MOCK DATA)
```rust
// In crates/context-graph-graph/tests/nt_validation_tests.rs

use context_graph_graph::marblestone::{validate_or_error, compute_effective_validated};
use context_graph_graph::{NeurotransmitterWeights, Domain, GraphError};

#[test]
fn test_validate_real_domain_code() {
    // Use REAL domain profile, not mock data
    let weights = NeurotransmitterWeights::for_domain(Domain::Code);
    println!("BEFORE: Validating Code domain weights: e={}, i={}, m={}",
        weights.excitatory, weights.inhibitory, weights.modulatory);
    let result = validate_or_error(&weights);
    println!("AFTER: result = {:?}", result);
    assert!(result.is_ok());
}

#[test]
fn test_validate_all_real_domains() {
    for domain in Domain::all() {
        let weights = NeurotransmitterWeights::for_domain(domain);
        println!("BEFORE: Validating {:?} domain", domain);
        let result = validate_or_error(&weights);
        println!("AFTER: {:?} -> {:?}", domain, result);
        assert!(result.is_ok(), "Domain {:?} should produce valid weights", domain);
    }
}

#[test]
fn test_invalid_excitatory_returns_field_name() {
    let invalid = NeurotransmitterWeights::new(1.5, 0.5, 0.5);
    println!("BEFORE: Validating invalid excitatory=1.5");
    let result = validate_or_error(&invalid);
    println!("AFTER: result = {:?}", result);

    match result {
        Err(GraphError::InvalidNtWeights { field, value }) => {
            assert_eq!(field, "excitatory");
            assert!((value - 1.5).abs() < 0.001);
        }
        _ => panic!("Expected InvalidNtWeights error"),
    }
}

#[test]
fn test_compute_effective_validated_success() {
    let weights = NeurotransmitterWeights::for_domain(Domain::General);
    println!("BEFORE: compute_effective_validated with General domain, base=1.0");
    let result = compute_effective_validated(&weights, 1.0);
    println!("AFTER: result = {:?}", result);

    let effective = result.expect("Should succeed");
    // General: ((1.0*0.5 - 1.0*0.2) * (1 + (0.3-0.5)*0.4)) = 0.3 * 0.92 = 0.276
    assert!((effective - 0.276).abs() < 0.001);
}

#[test]
fn test_compute_effective_validated_fails_on_invalid() {
    let invalid = NeurotransmitterWeights::new(-0.1, 0.5, 0.5);
    println!("BEFORE: compute_effective_validated with invalid weights");
    let result = compute_effective_validated(&invalid, 1.0);
    println!("AFTER: result = {:?}", result);
    assert!(result.is_err());
}
```

## Full State Verification Requirements

### Source of Truth
- **Core validate()**: `crates/context-graph-core/src/marblestone/neurotransmitter_weights.rs:148`
- **Error type**: `crates/context-graph-graph/src/error.rs:148`
- **New wrapper**: `crates/context-graph-graph/src/marblestone/validation.rs`

### Execute & Inspect
```bash
# 1. Verify error type exists
grep -n "InvalidNtWeights" crates/context-graph-graph/src/error.rs

# 2. Verify core validate exists
grep -n "pub fn validate" crates/context-graph-core/src/marblestone/neurotransmitter_weights.rs

# 3. Run validation tests with logging
RUST_LOG=debug cargo test -p context-graph-graph nt_validation -- --nocapture

# 4. Check for clippy warnings
cargo clippy -p context-graph-graph -- -D warnings
```

### Edge Cases (3 Required - BEFORE/AFTER logging)

#### Edge Case 1: NaN Value
```rust
#[test]
fn test_nan_is_invalid() {
    println!("BEFORE: Testing NaN excitatory weight");
    let invalid = NeurotransmitterWeights::new(f32::NAN, 0.5, 0.5);
    let result = validate_or_error(&invalid);
    println!("AFTER: result = {:?}", result);

    match result {
        Err(GraphError::InvalidNtWeights { field, value }) => {
            assert_eq!(field, "excitatory");
            assert!(value.is_nan());
        }
        _ => panic!("NaN should be rejected"),
    }
}
```

#### Edge Case 2: Infinity Value
```rust
#[test]
fn test_infinity_is_invalid() {
    println!("BEFORE: Testing +Infinity inhibitory weight");
    let invalid = NeurotransmitterWeights::new(0.5, f32::INFINITY, 0.5);
    let result = validate_or_error(&invalid);
    println!("AFTER: result = {:?}", result);

    match result {
        Err(GraphError::InvalidNtWeights { field, value }) => {
            assert_eq!(field, "inhibitory");
            assert!(value.is_infinite());
        }
        _ => panic!("Infinity should be rejected"),
    }
}
```

#### Edge Case 3: Boundary Values (0.0 and 1.0)
```rust
#[test]
fn test_boundary_values_accepted() {
    println!("BEFORE: Testing boundary min (0.0, 0.0, 0.0)");
    let min = NeurotransmitterWeights::new(0.0, 0.0, 0.0);
    let result = validate_or_error(&min);
    println!("AFTER: min boundary result = {:?}", result);
    assert!(result.is_ok());

    println!("BEFORE: Testing boundary max (1.0, 1.0, 1.0)");
    let max = NeurotransmitterWeights::new(1.0, 1.0, 1.0);
    let result = validate_or_error(&max);
    println!("AFTER: max boundary result = {:?}", result);
    assert!(result.is_ok());
}
```

### Evidence of Success
1. `cargo test -p context-graph-graph nt_validation -- --nocapture` shows all tests pass
2. Log output shows BEFORE/AFTER for each edge case
3. `validate_or_error` returns `Err(GraphError::InvalidNtWeights)` with correct field/value
4. No clippy warnings

## Sherlock-Holmes Verification

After implementation, spawn `sherlock-holmes` subagent to verify:

```
Investigate and verify M04-T14a implementation:

1. VERIFY: validation.rs exists at correct location
   - Run: ls -la crates/context-graph-graph/src/marblestone/validation.rs

2. VERIFY: mod.rs exports validation functions
   - Run: grep -n "validation" crates/context-graph-graph/src/marblestone/mod.rs
   - Should see: mod validation; pub use validation::{...};

3. VERIFY: Tests use REAL data not mocks
   - Run: grep -n "for_domain" crates/context-graph-graph/tests/nt_validation_tests.rs
   - Should use Domain::Code, Domain::General, etc.

4. VERIFY: Error handling is correct
   - Run: RUST_LOG=debug cargo test -p context-graph-graph test_nan_is_invalid -- --nocapture
   - Should see log::error! output for validation failure

5. VERIFY: Question mark operator works
   - Create test function that uses ? with validate_or_error
   - Must propagate GraphError correctly

6. EVIDENCE REQUIRED:
   - Test output showing BEFORE/AFTER logs
   - Proof that NaN/Infinity are rejected
   - Proof that boundary values (0.0, 1.0) are accepted
   - Error field names match weight names exactly
```

## Related Tasks
- M04-T14: Verify NT weight re-exports (prerequisite)
- M04-T15: Integrate NT weights into GraphEdge (uses this validation)
