---
id: "M04-T14"
title: "Integrate NeurotransmitterWeights with Graph Layer"
description: |
  NeurotransmitterWeights and Domain ALREADY EXIST in context-graph-core.
  This task verifies integration with context-graph-graph and documents the API.
  NO NEW TYPES TO CREATE - verify re-exports and write integration tests.

  CANONICAL FORMULA (from context-graph-core):
  w_eff = ((base * excitatory - base * inhibitory) * (1 + (modulatory - 0.5) * 0.4)).clamp(0.0, 1.0)
layer: "logic"
status: "pending"
priority: "high"
estimated_hours: 1
sequence: 19
depends_on: []
spec_refs:
  - "TECH-GRAPH-004 Section 4.1"
  - "REQ-KG-065"
  - "constitution.yaml edge_model.nt_weights"
files_to_create: []
files_to_modify: []
test_file: "crates/context-graph-graph/tests/nt_integration_tests.rs"
---

## ⚠️ CRITICAL: Current Codebase State

**NeurotransmitterWeights and Domain ALREADY EXIST. DO NOT RECREATE.**

### Existing Implementation Locations

| Type | Location | Status |
|------|----------|--------|
| `NeurotransmitterWeights` | `context-graph-core/src/marblestone/neurotransmitter_weights.rs` | ✅ COMPLETE |
| `Domain` | `context-graph-core/src/marblestone/domain.rs` | ✅ COMPLETE |
| Re-export | `context-graph-graph/src/lib.rs` line 60 | ✅ COMPLETE |
| Error type | `context-graph-graph/src/error.rs` `InvalidNtWeights` | ✅ COMPLETE |

### Existing API (DO NOT MODIFY)

```rust
// In context-graph-core/src/marblestone/neurotransmitter_weights.rs
pub struct NeurotransmitterWeights {
    pub excitatory: f32,  // [0.0, 1.0]
    pub inhibitory: f32,  // [0.0, 1.0]
    pub modulatory: f32,  // [0.0, 1.0]
}

impl NeurotransmitterWeights {
    pub fn new(excitatory: f32, inhibitory: f32, modulatory: f32) -> Self;
    pub fn for_domain(domain: Domain) -> Self;
    pub fn compute_effective_weight(&self, base_weight: f32) -> f32;
    pub fn validate(&self) -> bool;  // Returns bool, NOT Result
}

// In context-graph-core/src/marblestone/domain.rs
pub enum Domain { Code, Legal, Medical, Creative, Research, General }
```

### Domain-Specific Profiles (EXISTING)

| Domain | excitatory | inhibitory | modulatory |
|--------|------------|------------|------------|
| Code | 0.6 | 0.3 | 0.4 |
| Legal | 0.4 | 0.4 | 0.2 |
| Medical | 0.5 | 0.3 | 0.5 |
| Creative | 0.8 | 0.1 | 0.6 |
| Research | 0.6 | 0.2 | 0.5 |
| General | 0.5 | 0.2 | 0.3 |

## Context

The Marblestone-inspired neurotransmitter system is already implemented in `context-graph-core`. This task verifies the integration with `context-graph-graph` works correctly through re-exports.

### Constitution Reference
- `edge_model.nt_weights`: Defines weight structure and domain profiles
- `AP-009`: All weights must be in [0.0, 1.0]

## Scope

### In Scope
- Verify re-exports work from `context-graph-graph::NeurotransmitterWeights`
- Verify re-exports work from `context-graph-graph::Domain`
- Write integration tests in context-graph-graph
- Document the API for downstream consumers

### Out of Scope
- Creating new types (ALREADY EXIST)
- Modifying existing implementations
- GraphEdge integration (M04-T15)
- Result-returning validation (M04-T14a)

## Definition of Done

### Acceptance Criteria
- [ ] `use context_graph_graph::NeurotransmitterWeights;` compiles
- [ ] `use context_graph_graph::Domain;` compiles
- [ ] `NeurotransmitterWeights::for_domain(Domain::Code)` returns correct values
- [ ] `weights.compute_effective_weight(1.0)` computes correctly
- [ ] `weights.validate()` returns true for valid weights
- [ ] Integration tests pass in context-graph-graph
- [ ] No clippy warnings

### Verification Commands
```bash
cargo build -p context-graph-graph
cargo test -p context-graph-graph nt_integration
cargo clippy -p context-graph-graph -- -D warnings
```

## Implementation Approach

### Step 1: Verify Re-exports
```rust
// In crates/context-graph-graph/tests/nt_integration_tests.rs
use context_graph_graph::{NeurotransmitterWeights, Domain};

#[test]
fn test_nt_weights_reexport() {
    let weights = NeurotransmitterWeights::for_domain(Domain::Code);
    assert_eq!(weights.excitatory, 0.6);
    assert_eq!(weights.inhibitory, 0.3);
    assert_eq!(weights.modulatory, 0.4);
}
```

### Step 2: Write Integration Tests
```rust
#[test]
fn test_compute_effective_weight_general_domain() {
    let weights = NeurotransmitterWeights::for_domain(Domain::General);
    // General: e=0.5, i=0.2, m=0.3
    // w_eff = ((1.0*0.5 - 1.0*0.2) * (1 + (0.3-0.5)*0.4)).clamp(0,1)
    // w_eff = (0.3 * 0.92).clamp(0,1) = 0.276
    let effective = weights.compute_effective_weight(1.0);
    assert!((effective - 0.276).abs() < 0.001);
}

#[test]
fn test_validate_returns_bool() {
    let valid = NeurotransmitterWeights::new(0.5, 0.3, 0.4);
    assert!(valid.validate());  // Returns bool, not Result

    let invalid = NeurotransmitterWeights::new(1.5, 0.0, 0.0);
    assert!(!invalid.validate());
}

#[test]
fn test_all_domains_produce_valid_weights() {
    for domain in Domain::all() {
        let weights = NeurotransmitterWeights::for_domain(domain);
        assert!(weights.validate(), "Domain {:?} produces invalid weights", domain);
    }
}
```

## Full State Verification Requirements

### Source of Truth
- **File**: `crates/context-graph-core/src/marblestone/neurotransmitter_weights.rs`
- **Re-export**: `crates/context-graph-graph/src/lib.rs` line 60
- **Verify**: `grep -n "NeurotransmitterWeights" crates/context-graph-graph/src/lib.rs`

### Execute & Inspect
```bash
# 1. Verify re-export exists
grep -n "pub use context_graph_core::marblestone" crates/context-graph-graph/src/lib.rs

# 2. Run integration tests
cargo test -p context-graph-graph nt_integration -- --nocapture

# 3. Verify compilation
cargo build -p context-graph-graph 2>&1 | head -20
```

### Edge Cases (3 Required - BEFORE/AFTER logging)

#### Edge Case 1: Boundary Values
```rust
#[test]
fn test_boundary_values() {
    println!("BEFORE: Testing boundary value weights (0.0, 0.0, 0.0)");
    let min = NeurotransmitterWeights::new(0.0, 0.0, 0.0);
    println!("AFTER: validate() = {}", min.validate());
    assert!(min.validate());

    println!("BEFORE: Testing boundary value weights (1.0, 1.0, 1.0)");
    let max = NeurotransmitterWeights::new(1.0, 1.0, 1.0);
    println!("AFTER: validate() = {}", max.validate());
    assert!(max.validate());
}
```

#### Edge Case 2: Out of Range (Negative)
```rust
#[test]
fn test_negative_weight_invalid() {
    println!("BEFORE: Testing negative excitatory weight -0.1");
    let invalid = NeurotransmitterWeights::new(-0.1, 0.5, 0.5);
    println!("AFTER: validate() = {}", invalid.validate());
    assert!(!invalid.validate());
}
```

#### Edge Case 3: Out of Range (Above 1.0)
```rust
#[test]
fn test_above_one_weight_invalid() {
    println!("BEFORE: Testing excitatory weight 1.1 (above max)");
    let invalid = NeurotransmitterWeights::new(1.1, 0.5, 0.5);
    println!("AFTER: validate() = {}", invalid.validate());
    assert!(!invalid.validate());
}
```

### Evidence of Success
1. `cargo test -p context-graph-graph nt_integration` shows all tests pass
2. No warnings from `cargo clippy -p context-graph-graph`
3. `use context_graph_graph::NeurotransmitterWeights` compiles in downstream code

## Sherlock-Holmes Verification

After implementation, spawn `sherlock-holmes` subagent to verify:

```
Investigate and verify M04-T14 implementation:

1. VERIFY: Re-exports exist in crates/context-graph-graph/src/lib.rs
   - Check line 60: `pub use context_graph_core::marblestone::{Domain, EdgeType, NeurotransmitterWeights};`

2. VERIFY: Integration tests exist and pass
   - Run: cargo test -p context-graph-graph nt_integration -- --nocapture
   - All 6 domains must produce valid weights

3. VERIFY: No duplicate implementations
   - Run: grep -r "struct NeurotransmitterWeights" crates/
   - Should only find ONE definition in context-graph-core

4. VERIFY: Formula correctness
   - compute_effective_weight(1.0) for General domain = 0.276 ± 0.001

5. EVIDENCE REQUIRED:
   - Screenshot/log of passing tests
   - grep output showing single definition
   - Formula verification calculation
```

## Related Tasks
- M04-T14a: Add Result-returning validation wrapper in graph layer
- M04-T15: Integrate NT weights into GraphEdge struct
