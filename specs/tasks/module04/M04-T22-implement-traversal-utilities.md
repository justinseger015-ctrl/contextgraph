---
id: "M04-T22"
title: "Implement Standalone Modulation Utility Functions (Marblestone)"
description: |
  Add standalone utility functions in marblestone module that wrap GraphEdge.get_modulated_weight()
  and provide additional traversal helpers for domain-aware operations.

  CRITICAL: GraphEdge.get_modulated_weight() ALREADY EXISTS in storage/edges.rs.
  This task adds STANDALONE functions that wrap and extend that functionality.
layer: "surface"
status: "ready"
priority: "high"
estimated_hours: 1
sequence: 30
depends_on:
  - "M04-T15"  # GraphEdge with get_modulated_weight (COMPLETE)
  - "M04-T16"  # BFS (COMPLETE - already uses get_modulated_weight)
  - "M04-T17"  # DFS (COMPLETE)
  - "M04-T17a" # A* (COMPLETE)
spec_refs:
  - "TECH-GRAPH-004 Section 8"
  - "REQ-KG-065"
files_to_create: []
files_to_modify:
  - path: "crates/context-graph-graph/src/marblestone/mod.rs"
    description: "Add standalone utility functions wrapping GraphEdge.get_modulated_weight()"
test_file: "crates/context-graph-graph/src/marblestone/mod.rs (inline #[cfg(test)])"
audited_against: "commit 4fd5052 (M04-T20 complete)"
last_updated: "2026-01-04"
---

# M04-T22: Implement Standalone Modulation Utility Functions

## CRITICAL: Read This First

### What ALREADY EXISTS (DO NOT RECREATE)

**1. `GraphEdge.get_modulated_weight(query_domain: Domain) -> f32`**
- **File**: `crates/context-graph-graph/src/storage/edges.rs` (lines 258-276)
- **Status**: FULLY IMPLEMENTED AND TESTED
- **Formula** (CANONICAL - this is the source of truth):

```rust
// EXISTING CODE - DO NOT REIMPLEMENT
let nt = &self.neurotransmitter_weights;
let net_activation = nt.excitatory - nt.inhibitory + (nt.modulatory * 0.5);
let domain_bonus = if self.domain == query_domain { 0.1 } else { 0.0 };
let steering_factor = 0.5 + self.steering_reward;  // Range [0.5, 1.5]
let w_eff = self.weight * (1.0 + net_activation + domain_bonus) * steering_factor;
w_eff.clamp(0.0, 1.0)
```

**2. `NeurotransmitterWeights::compute_effective_weight(base: f32) -> f32`**
- **File**: `crates/context-graph-core/src/marblestone/neurotransmitter_weights.rs`
- Uses a DIFFERENT formula (does NOT include domain_bonus or steering_factor)
- Do NOT confuse with GraphEdge method

**3. Actual Domain NT Profiles (from neurotransmitter_weights.rs lines 83-91):**

| Domain | excitatory | inhibitory | modulatory | net_activation |
|--------|------------|------------|------------|----------------|
| Code | 0.6 | 0.3 | 0.4 | 0.5 |
| Legal | 0.4 | 0.4 | 0.2 | 0.1 |
| Medical | 0.5 | 0.3 | 0.5 | 0.45 |
| Creative | 0.8 | 0.1 | 0.6 | 1.0 |
| Research | 0.6 | 0.2 | 0.5 | 0.65 |
| General | 0.5 | 0.2 | 0.3 | 0.45 |

**WARNING**: The OLD task document had WRONG NT values. The above are CORRECT.

**4. Existing marblestone/mod.rs exports:**
- `Domain`, `EdgeType`, `NeurotransmitterWeights` (from core)
- `validate_or_error`, `compute_effective_validated` (from validation.rs)
- `domain_aware_search`, `domain_nt_summary`, `expected_domain_boost`, etc. (from search)

---

## Task Objective

Add **standalone utility functions** to `crates/context-graph-graph/src/marblestone/mod.rs` that:
1. Wrap `GraphEdge.get_modulated_weight()` for functional-style API
2. Provide `traversal_cost()` (inverted weight for pathfinding)
3. Provide `modulation_ratio()` to show boost/suppress as multiplier
4. Provide batch operations
5. Provide analysis/summary types for debugging

These are **convenience wrappers** that delegate to the existing GraphEdge implementation.

---

## Exact Implementation

### Update: `crates/context-graph-graph/src/marblestone/mod.rs`

Add the following to the existing file (after the current exports):

```rust
// =============================================================================
// STANDALONE UTILITY FUNCTIONS (M04-T22)
// =============================================================================

use crate::storage::edges::GraphEdge;

/// Domain match bonus constant (matches GraphEdge implementation).
pub const DOMAIN_MATCH_BONUS: f32 = 0.1;

/// Get effective edge weight with Marblestone modulation.
///
/// This is a **standalone wrapper** around `GraphEdge.get_modulated_weight()`.
/// Use when you have an edge reference and want functional-style API.
///
/// # Canonical Formula (delegated to GraphEdge)
///
/// ```text
/// net_activation = excitatory - inhibitory + (modulatory * 0.5)
/// domain_bonus = 0.1 if edge_domain == query_domain else 0.0
/// steering_factor = 0.5 + steering_reward  // Range [0.5, 1.5]
/// w_eff = base_weight * (1.0 + net_activation + domain_bonus) * steering_factor
/// Result clamped to [0.0, 1.0]
/// ```
///
/// # Arguments
/// * `edge` - The graph edge to compute weight for
/// * `query_domain` - The domain context for the query
///
/// # Returns
/// Effective weight in [0.0, 1.0]
#[inline]
pub fn get_modulated_weight(edge: &GraphEdge, query_domain: Domain) -> f32 {
    edge.get_modulated_weight(query_domain)
}

/// Compute traversal cost from edge weight.
///
/// For pathfinding algorithms (BFS, DFS, A*), lower cost = preferred path.
/// This inverts the modulated weight: high weight = low cost.
///
/// # Formula
/// ```text
/// cost = 1.0 - get_modulated_weight(edge, query_domain)
/// ```
#[inline]
pub fn traversal_cost(edge: &GraphEdge, query_domain: Domain) -> f32 {
    1.0 - edge.get_modulated_weight(query_domain)
}

/// Calculate the modulation ratio (effective / base).
///
/// Shows how much the modulation changes the weight:
/// - ratio > 1.0: Weight boosted
/// - ratio < 1.0: Weight suppressed
/// - ratio = 1.0: No change
///
/// Returns 1.0 if base_weight is zero (avoid division by zero).
#[inline]
pub fn modulation_ratio(edge: &GraphEdge, query_domain: Domain) -> f32 {
    let effective = edge.get_modulated_weight(query_domain);
    if edge.weight > 1e-6 {
        effective / edge.weight
    } else {
        1.0
    }
}

/// Batch compute modulated weights for multiple edges.
pub fn get_modulated_weights_batch(edges: &[GraphEdge], query_domain: Domain) -> Vec<f32> {
    edges
        .iter()
        .map(|e| e.get_modulated_weight(query_domain))
        .collect()
}

/// Batch compute traversal costs for multiple edges.
pub fn traversal_costs_batch(edges: &[GraphEdge], query_domain: Domain) -> Vec<f32> {
    edges.iter().map(|e| traversal_cost(e, query_domain)).collect()
}

/// Modulation effect classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModulationEffect {
    /// Weight increased (ratio > 1.05)
    Boosted,
    /// Weight unchanged (ratio in [0.95, 1.05])
    Neutral,
    /// Weight decreased (ratio < 0.95)
    Suppressed,
}

/// Determine if edge is boosted or suppressed for a domain.
#[inline]
pub fn modulation_effect(edge: &GraphEdge, query_domain: Domain) -> ModulationEffect {
    let ratio = modulation_ratio(edge, query_domain);
    if ratio > 1.05 {
        ModulationEffect::Boosted
    } else if ratio < 0.95 {
        ModulationEffect::Suppressed
    } else {
        ModulationEffect::Neutral
    }
}

/// Expected modulation multiplier for a domain (assuming match + neutral steering).
///
/// # Formula
/// ```text
/// nt = NeurotransmitterWeights::for_domain(domain)
/// net_activation = nt.excitatory - nt.inhibitory + (nt.modulatory * 0.5)
/// multiplier = 1.0 + net_activation + DOMAIN_MATCH_BONUS
/// ```
/// Note: Assumes neutral steering (steering_factor = 1.0).
pub fn expected_domain_modulation(domain: Domain) -> f32 {
    let nt = NeurotransmitterWeights::for_domain(domain);
    let net_activation = nt.excitatory - nt.inhibitory + (nt.modulatory * 0.5);
    1.0 + net_activation + DOMAIN_MATCH_BONUS
}

/// Detailed modulation summary for debugging.
#[derive(Debug, Clone)]
pub struct ModulationSummary {
    pub base_weight: f32,
    pub effective_weight: f32,
    pub net_activation: f32,
    pub domain_bonus: f32,
    pub steering_factor: f32,
    pub ratio: f32,
    pub effect: ModulationEffect,
}

impl ModulationSummary {
    /// Create summary from edge and query domain.
    pub fn from_edge(edge: &GraphEdge, query_domain: Domain) -> Self {
        let nt = &edge.neurotransmitter_weights;
        let net_activation = nt.excitatory - nt.inhibitory + (nt.modulatory * 0.5);
        let domain_bonus = if edge.domain == query_domain {
            DOMAIN_MATCH_BONUS
        } else {
            0.0
        };
        let steering_factor = 0.5 + edge.steering_reward;
        let effective = edge.get_modulated_weight(query_domain);
        let ratio = if edge.weight > 1e-6 {
            effective / edge.weight
        } else {
            1.0
        };

        Self {
            base_weight: edge.weight,
            effective_weight: effective,
            net_activation,
            domain_bonus,
            steering_factor,
            ratio,
            effect: modulation_effect(edge, query_domain),
        }
    }
}

impl std::fmt::Display for ModulationSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "base={:.3} eff={:.3} (net_act={:+.3} dom={:.2} steer={:.2}) ratio={:.2}x {:?}",
            self.base_weight,
            self.effective_weight,
            self.net_activation,
            self.domain_bonus,
            self.steering_factor,
            self.ratio,
            self.effect,
        )
    }
}

// =============================================================================
// TESTS (M04-T22)
// =============================================================================

#[cfg(test)]
mod modulation_tests {
    use super::*;
    use uuid::Uuid;

    fn make_test_edge(weight: f32, domain: Domain) -> GraphEdge {
        let mut edge = GraphEdge::new(
            1,
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            weight,
            domain,
        );
        // Set steering_reward to 0.5 so steering_factor = 1.0 (neutral)
        edge.steering_reward = 0.5;
        edge
    }

    #[test]
    fn test_standalone_matches_method() {
        let edge = make_test_edge(0.8, Domain::Code);
        let standalone = get_modulated_weight(&edge, Domain::Code);
        let method = edge.get_modulated_weight(Domain::Code);
        assert!(
            (standalone - method).abs() < 1e-6,
            "Standalone should match method: {} vs {}",
            standalone,
            method
        );
    }

    #[test]
    fn test_code_domain_modulation() {
        let edge = make_test_edge(0.5, Domain::Code);
        // Code: e=0.6, i=0.3, m=0.4
        // net_activation = 0.6 - 0.3 + 0.2 = 0.5
        // domain_bonus = 0.1 (matching)
        // steering_factor = 0.5 + 0.5 = 1.0
        // w_eff = 0.5 * (1.0 + 0.5 + 0.1) * 1.0 = 0.5 * 1.6 = 0.8
        let w = get_modulated_weight(&edge, Domain::Code);
        assert!((w - 0.8).abs() < 0.01, "Expected ~0.8, got {}", w);
    }

    #[test]
    fn test_domain_mismatch_no_bonus() {
        let edge = make_test_edge(0.5, Domain::Code);
        // Query Legal, edge is Code
        // net_activation = 0.5 (Code profile still applies)
        // domain_bonus = 0.0 (no match)
        // w_eff = 0.5 * (1.0 + 0.5 + 0.0) * 1.0 = 0.5 * 1.5 = 0.75
        let w = get_modulated_weight(&edge, Domain::Legal);
        assert!((w - 0.75).abs() < 0.01, "Expected ~0.75, got {}", w);
    }

    #[test]
    fn test_traversal_cost_inversion() {
        let edge = make_test_edge(0.8, Domain::Code);
        let effective = get_modulated_weight(&edge, Domain::Code);
        let cost = traversal_cost(&edge, Domain::Code);
        assert!(
            (cost + effective - 1.0).abs() < 1e-6,
            "cost + effective should = 1.0"
        );
    }

    #[test]
    fn test_modulation_ratio() {
        let edge = make_test_edge(0.5, Domain::Code);
        let ratio = modulation_ratio(&edge, Domain::Code);
        // effective = 0.8, base = 0.5, ratio = 1.6
        assert!((ratio - 1.6).abs() < 0.01, "Expected ratio ~1.6, got {}", ratio);
    }

    #[test]
    fn test_modulation_effect_boosted() {
        let edge = make_test_edge(0.5, Domain::Code);
        let effect = modulation_effect(&edge, Domain::Code);
        assert_eq!(effect, ModulationEffect::Boosted);
    }

    #[test]
    fn test_zero_base_weight() {
        let edge = make_test_edge(0.0, Domain::Code);
        let w = get_modulated_weight(&edge, Domain::Code);
        assert!((w - 0.0).abs() < 1e-6, "Zero base should give zero effective");
        let ratio = modulation_ratio(&edge, Domain::Code);
        assert!((ratio - 1.0).abs() < 1e-6, "Zero base should give ratio 1.0");
    }

    #[test]
    fn test_batch_modulation() {
        let edges = vec![
            make_test_edge(0.3, Domain::Code),
            make_test_edge(0.5, Domain::Legal),
            make_test_edge(0.7, Domain::General),
        ];
        let weights = get_modulated_weights_batch(&edges, Domain::Code);
        assert_eq!(weights.len(), 3);
        // First edge (Code) has domain match, should be highest ratio
    }

    #[test]
    fn test_expected_domain_modulation() {
        // Code: net_act = 0.5, mult = 1.0 + 0.5 + 0.1 = 1.6
        let code_mod = expected_domain_modulation(Domain::Code);
        assert!((code_mod - 1.6).abs() < 0.01, "Code should be ~1.6, got {}", code_mod);

        // General: net_act = 0.5 - 0.2 + 0.15 = 0.45, mult = 1.55
        let general_mod = expected_domain_modulation(Domain::General);
        assert!(
            (general_mod - 1.55).abs() < 0.01,
            "General should be ~1.55, got {}",
            general_mod
        );
    }

    #[test]
    fn test_modulation_summary() {
        let edge = make_test_edge(0.5, Domain::Code);
        let summary = ModulationSummary::from_edge(&edge, Domain::Code);

        assert!((summary.base_weight - 0.5).abs() < 1e-6);
        assert!((summary.net_activation - 0.5).abs() < 0.01);
        assert!((summary.domain_bonus - 0.1).abs() < 1e-6);
        assert!((summary.steering_factor - 1.0).abs() < 1e-6);
        assert_eq!(summary.effect, ModulationEffect::Boosted);
    }

    #[test]
    fn test_clamping_high_values() {
        let mut edge = make_test_edge(1.0, Domain::Creative);
        edge.steering_reward = 1.0; // steering_factor = 1.5
        // Creative: net_act = 0.8 - 0.1 + 0.3 = 1.0
        // w_eff = 1.0 * (1.0 + 1.0 + 0.1) * 1.5 = 3.15 -> clamped to 1.0
        let w = get_modulated_weight(&edge, Domain::Creative);
        assert!((w - 1.0).abs() < 1e-6, "Should clamp to 1.0");
    }

    #[test]
    fn test_steering_affects_output() {
        let mut edge1 = make_test_edge(0.5, Domain::General);
        edge1.steering_reward = 0.0; // steering_factor = 0.5

        let mut edge2 = make_test_edge(0.5, Domain::General);
        edge2.steering_reward = 1.0; // steering_factor = 1.5

        let w1 = get_modulated_weight(&edge1, Domain::General);
        let w2 = get_modulated_weight(&edge2, Domain::General);

        // w2 should be 3x w1 (1.5 / 0.5 = 3)
        assert!(w2 > w1, "Higher steering should give higher weight");
        assert!((w2 / w1 - 3.0).abs() < 0.1, "Ratio should be ~3x");
    }
}
```

---

## Acceptance Criteria

- [ ] `get_modulated_weight()` standalone function exists and delegates to `GraphEdge` method
- [ ] `traversal_cost()` returns `1.0 - get_modulated_weight()`
- [ ] `modulation_ratio()` returns `effective / base` (or 1.0 if base is zero)
- [ ] `get_modulated_weights_batch()` works for `&[GraphEdge]`
- [ ] `traversal_costs_batch()` works for `&[GraphEdge]`
- [ ] `ModulationEffect` enum with `Boosted`/`Neutral`/`Suppressed`
- [ ] `modulation_effect()` classifies based on ratio thresholds (1.05, 0.95)
- [ ] `expected_domain_modulation()` returns expected multiplier
- [ ] `ModulationSummary` struct with `Display` impl
- [ ] `DOMAIN_MATCH_BONUS` constant equals `0.1`
- [ ] All tests pass
- [ ] No clippy warnings
- [ ] Compiles successfully

---

## Build & Test Commands

```bash
# Build
cargo build -p context-graph-graph

# Run specific tests
cargo test -p context-graph-graph modulation_tests -- --nocapture

# Run all marblestone tests
cargo test -p context-graph-graph marblestone -- --nocapture

# Clippy
cargo clippy -p context-graph-graph -- -D warnings

# Verify exports in docs
cargo doc -p context-graph-graph --no-deps
```

---

## Full State Verification Requirements

### 1. Source of Truth

- **Primary File**: `crates/context-graph-graph/src/marblestone/mod.rs`
- **Delegate**: `crates/context-graph-graph/src/storage/edges.rs` (GraphEdge.get_modulated_weight)
- **Verification**: Standalone functions must delegate to GraphEdge method exactly

### 2. Execute & Inspect

After implementation, run these commands and capture output:

```bash
# Step 1: Verify compilation
cargo build -p context-graph-graph 2>&1

# Step 2: Run tests with output
cargo test -p context-graph-graph modulation_tests -- --nocapture 2>&1

# Step 3: Verify no clippy warnings
cargo clippy -p context-graph-graph -- -D warnings 2>&1

# Step 4: Verify functions are exported
grep -n "pub fn get_modulated_weight\|pub fn traversal_cost\|pub fn modulation_ratio" \
    crates/context-graph-graph/src/marblestone/mod.rs
```

### 3. Edge Case Audit (MUST PRINT BEFORE/AFTER STATE)

Manually verify these 3 edge cases in tests with print statements:

**Edge Case 1: Zero Base Weight**
```rust
#[test]
fn test_edge_case_zero_weight() {
    let edge = make_test_edge(0.0, Domain::Code);
    println!("BEFORE: base_weight={}, domain={:?}", edge.weight, edge.domain);
    let w = get_modulated_weight(&edge, Domain::Code);
    println!("AFTER: effective_weight={}", w);
    assert!((w - 0.0).abs() < 1e-6, "Zero base must give zero effective");
}
```

**Edge Case 2: Maximum Values (Clamp Test)**
```rust
#[test]
fn test_edge_case_clamp() {
    let mut edge = make_test_edge(1.0, Domain::Creative);
    edge.steering_reward = 1.0;
    println!("BEFORE: base={}, steering_reward={}, domain=Creative", edge.weight, edge.steering_reward);
    let w = get_modulated_weight(&edge, Domain::Creative);
    println!("AFTER: effective={} (expected: clamped to 1.0)", w);
    assert!((w - 1.0).abs() < 1e-6, "Must clamp to 1.0");
}
```

**Edge Case 3: Domain Mismatch**
```rust
#[test]
fn test_edge_case_domain_mismatch() {
    let edge = make_test_edge(0.5, Domain::Code);
    println!("BEFORE: edge_domain=Code, query_domain=Legal");
    let w_match = get_modulated_weight(&edge, Domain::Code);
    let w_mismatch = get_modulated_weight(&edge, Domain::Legal);
    println!("AFTER: match_weight={}, mismatch_weight={}", w_match, w_mismatch);
    println!("Difference (domain bonus): {}", w_match - w_mismatch);
    assert!(w_match > w_mismatch, "Domain match must give higher weight");
}
```

### 4. Evidence of Success

Provide:
1. Full test output showing all tests pass
2. Clippy output showing no warnings
3. Grep output showing functions are exported

---

## Sherlock-Holmes Final Verification (MANDATORY)

After completing implementation, you **MUST** spawn a `sherlock-holmes` subagent to verify:

1. **Existence Check**: All functions exist in `marblestone/mod.rs`
2. **Delegation Check**: `get_modulated_weight()` delegates to `GraphEdge.get_modulated_weight()`
3. **Formula Consistency**: No formula discrepancies between standalone and method
4. **Test Coverage**: All edge cases covered
5. **No Regressions**: Existing tests still pass

The sherlock-holmes agent will provide a forensic verification report. **Any issues identified MUST be fixed before marking task complete.**

---

## Related Tasks

| Task | Status | Relevance |
|------|--------|-----------|
| M04-T15 | COMPLETE | GraphEdge with `get_modulated_weight()` method |
| M04-T16 | COMPLETE | BFS uses `edge.get_modulated_weight(domain)` |
| M04-T17 | COMPLETE | DFS traversal |
| M04-T17a | COMPLETE | A* with hyperbolic heuristic |
| M04-T14a | COMPLETE | `validate_or_error()` validation wrapper |
| M04-T19 | COMPLETE | Domain-aware search uses modulation |

---

## Important Notes for AI Agent

1. **DO NOT reimplement `get_modulated_weight` logic** - GraphEdge method is source of truth
2. The standalone function is a **WRAPPER**, not a reimplementation
3. Use `edge.get_modulated_weight(query_domain)` internally
4. **CORRECT STEERING FORMULA**: `steering_factor = 0.5 + steering_reward` (NOT clamped to [0.1, 2.0])
5. **CORRECT NT VALUES**: Use the table in "What ALREADY EXISTS" section
6. All tests must verify wrapper matches method exactly
7. **NO BACKWARDS COMPATIBILITY HACKS** - code works or fails fast with clear errors
8. **NO MOCK DATA** - all tests use real GraphEdge instances

---

*Task Version: 2.0*
*Last Updated: 2026-01-04*
*Audited Against: commit 4fd5052 (M04-T20 complete)*
