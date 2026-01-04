//! Contradiction detection module for knowledge graph.
//!
//! Identifies conflicting information in the knowledge graph by combining:
//! - Semantic similarity search (find semantically related nodes)
//! - Explicit CONTRADICTS edges (known contradictions from M04-T26)
//!
//! # Algorithm
//!
//! 1. Semantic search for similar nodes (k=50 candidates)
//! 2. BFS to find nodes with CONTRADICTS edges
//! 3. Combine and score contradictions
//! 4. Classify contradiction types
//! 5. Filter by threshold
//!
//! # Constitution Reference
//!
//! - edge_model.attrs: type:Semantic|Temporal|Causal|Hierarchical|Contradicts
//! - edge_model.nt_weights.formula: w_eff = base × (1 + excitatory - inhibitory + 0.5×modulatory)
//! - AP-001: Never unwrap() in prod - FAIL FAST with explicit errors
//!
//! # M04-T21: Contradiction Detection Implementation
//!
//! This module depends on M04-T26 (EdgeType::Contradicts).

pub mod detector;

pub use detector::{
    check_contradiction, contradiction_detect, get_contradictions, mark_contradiction,
    ContradictionParams, ContradictionResult,
};

// Re-export ContradictionType from core for convenience
pub use context_graph_core::marblestone::ContradictionType;
