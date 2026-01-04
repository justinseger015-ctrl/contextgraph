//! Marblestone neurotransmitter integration.
//!
//! This module re-exports types from context-graph-core and adds
//! graph-specific operations for NT-weighted edge traversal.
//!
//! # Neurotransmitter Model
//!
//! Edges in the graph have NT weights that modulate effective edge weight
//! based on the current domain context:
//!
//! ```text
//! w_eff = ((base * excitatory - base * inhibitory) * (1 + (modulatory - 0.5) * 0.4)).clamp(0.0, 1.0)
//! ```
//!
//! # Domain-Specific Modulation
//!
//! Different domains activate different NT profiles:
//! - Code: High modulatory for pattern matching
//! - Legal: High inhibitory for precise boundaries
//! - Medical: High excitatory for causal reasoning
//! - Creative: Balanced for exploration
//!
//! # Components
//!
//! - Re-exports from context-graph-core
//! - Validation: `validate_or_error()` for Result-returning validation (M04-T14a)
//! - Domain-aware search (M04-T19 COMPLETE)
//! - NT modulation functions (TODO: M04-T26)
//!
//! # Constitution Reference
//!
//! - edge_model.nt_weights: Definition and formula
//! - edge_model.nt_weights.domain: Code|Legal|Medical|Creative|Research|General
//! - AP-001: Never unwrap() in prod - all errors properly typed

mod validation;

// Re-export from core for convenience
pub use context_graph_core::marblestone::{Domain, EdgeType, NeurotransmitterWeights};

// Re-export validation functions (M04-T14a)
pub use validation::{compute_effective_validated, validate_or_error};

// Re-export domain-aware search (M04-T19)
pub use crate::search::domain_search::{
    domain_aware_search, domain_nt_summary, expected_domain_boost, DomainSearchResult,
    DomainSearchResults,
};

// TODO: M04-T26 - Implement NT modulation
// pub fn compute_effective_weight(
//     base_weight: f32,
//     nt_weights: &NeurotransmitterWeights,
// ) -> f32 {
//     base_weight * (1.0 + nt_weights.excitatory - nt_weights.inhibitory + 0.5 * nt_weights.modulatory)
// }
