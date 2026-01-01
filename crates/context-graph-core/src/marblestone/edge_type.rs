//! Edge type definitions for the Marblestone graph model.
//!
//! # Constitution Reference
//! - edge_model.attrs: type:Semantic|Temporal|Causal|Hierarchical

use serde::{Deserialize, Serialize};
use std::fmt;

/// Type of relationship between two nodes in the graph.
///
/// Each edge type represents a distinct semantic relationship with
/// different traversal and weighting characteristics:
/// - Semantic: Similarity-based connections (weight: 0.5)
/// - Temporal: Time-ordered sequences (weight: 0.7)
/// - Causal: Cause-effect relationships (weight: 0.8)
/// - Hierarchical: Parent-child taxonomies (weight: 0.9)
///
/// # Constitution Reference
///
/// - edge_model.attrs: type:Semantic|Temporal|Causal|Hierarchical
///
/// # Examples
///
/// ## Creating and Inspecting Edge Types
///
/// ```rust
/// use context_graph_core::marblestone::EdgeType;
///
/// // Create different edge types
/// let semantic = EdgeType::Semantic;
/// let causal = EdgeType::Causal;
///
/// // Check string representation
/// assert_eq!(semantic.to_string(), "semantic");
/// assert_eq!(causal.to_string(), "causal");
///
/// // Get default weights
/// assert_eq!(semantic.default_weight(), 0.5);
/// assert_eq!(causal.default_weight(), 0.8);
/// ```
///
/// ## Iterating Over All Edge Types
///
/// ```rust
/// use context_graph_core::marblestone::EdgeType;
///
/// for edge_type in EdgeType::all() {
///     println!("{}: weight={}, desc={}",
///         edge_type,
///         edge_type.default_weight(),
///         edge_type.description()
///     );
/// }
/// ```
///
/// ## Using with GraphEdge
///
/// ```rust
/// use uuid::Uuid;
/// use context_graph_core::types::GraphEdge;
/// use context_graph_core::marblestone::{EdgeType, Domain};
///
/// // Hierarchical edges have highest default weight
/// let edge = GraphEdge::new(
///     Uuid::new_v4(),
///     Uuid::new_v4(),
///     EdgeType::Hierarchical,
///     Domain::Code,
/// );
/// assert_eq!(edge.weight, 0.9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    /// Semantic similarity relationship.
    ///
    /// Nodes share similar meaning, topic, or conceptual space.
    /// Default weight: 0.5 (variable based on embedding similarity).
    Semantic,

    /// Temporal sequence relationship.
    ///
    /// Source node occurred before target node in time.
    /// Default weight: 0.7 (time relationships are usually reliable).
    Temporal,

    /// Causal relationship.
    ///
    /// Source node causes, influences, or triggers target node.
    /// Default weight: 0.8 (strong evidence when established).
    Causal,

    /// Hierarchical relationship.
    ///
    /// Source node is a parent, category, or ancestor of target node.
    /// Default weight: 0.9 (taxonomy relationships are very strong).
    Hierarchical,
}

impl EdgeType {
    /// Returns a human-readable description of this edge type.
    #[inline]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Semantic => "Semantic similarity - nodes share similar meaning or topic",
            Self::Temporal => "Temporal sequence - source precedes target in time",
            Self::Causal => "Causal relationship - source causes or influences target",
            Self::Hierarchical => "Hierarchical - source is parent or ancestor of target",
        }
    }

    /// Returns all edge type variants as an array.
    #[inline]
    pub fn all() -> [EdgeType; 4] {
        [
            Self::Semantic,
            Self::Temporal,
            Self::Causal,
            Self::Hierarchical,
        ]
    }

    /// Returns the default base weight for this edge type.
    ///
    /// These weights reflect the inherent reliability of each relationship type:
    /// - Semantic (0.5): Variable based on embedding similarity
    /// - Temporal (0.7): Time relationships are usually reliable
    /// - Causal (0.8): Strong evidence when established
    /// - Hierarchical (0.9): Taxonomy relationships are very strong
    #[inline]
    pub fn default_weight(&self) -> f32 {
        match self {
            Self::Semantic => 0.5,
            Self::Temporal => 0.7,
            Self::Causal => 0.8,
            Self::Hierarchical => 0.9,
        }
    }
}

impl Default for EdgeType {
    /// Returns `EdgeType::Semantic` as the default.
    /// Semantic is the most common edge type in knowledge graphs.
    #[inline]
    fn default() -> Self {
        Self::Semantic
    }
}

impl fmt::Display for EdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Semantic => "semantic",
            Self::Temporal => "temporal",
            Self::Causal => "causal",
            Self::Hierarchical => "hierarchical",
        };
        write!(f, "{}", s)
    }
}
