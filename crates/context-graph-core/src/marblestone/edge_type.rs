//! Edge type definitions for the Marblestone graph model.
//!
//! # Constitution Reference
//! - edge_model.attrs: type:Semantic|Temporal|Causal|Hierarchical|Contradicts
//!
//! # M04-T26: NT Modulation for Contradiction Detection
//! Added Contradicts variant for M04-T21 contradiction detection support.

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
/// - Contradicts: Logical contradiction relationship (weight: 0.3, inhibitory-heavy NT)
///
/// # Constitution Reference
///
/// - edge_model.attrs: type:Semantic|Temporal|Causal|Hierarchical|Contradicts
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
/// let contradicts = EdgeType::Contradicts;
///
/// // Check string representation
/// assert_eq!(semantic.to_string(), "semantic");
/// assert_eq!(causal.to_string(), "causal");
/// assert_eq!(contradicts.to_string(), "contradicts");
///
/// // Get default weights
/// assert_eq!(semantic.default_weight(), 0.5);
/// assert_eq!(causal.default_weight(), 0.8);
/// assert_eq!(contradicts.default_weight(), 0.3);
///
/// // Check contradiction status
/// assert!(!semantic.is_contradiction());
/// assert!(contradicts.is_contradiction());
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

    /// Contradiction relationship.
    ///
    /// Source node logically contradicts or conflicts with target node.
    /// Default weight: 0.3 (low base weight with inhibitory-heavy NT modulation).
    /// This edge type is symmetric: if A contradicts B, then B contradicts A.
    ///
    /// # M04-T26: NT Modulation
    /// Uses inhibitory-heavy NT profile: excitatory=0.2, inhibitory=0.7, modulatory=0.1
    /// This ensures contradictions suppress related content during retrieval.
    Contradicts,
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
            Self::Contradicts => "Contradiction - source logically contradicts target",
        }
    }

    /// Returns all edge type variants as an array.
    #[inline]
    pub fn all() -> [EdgeType; 5] {
        [
            Self::Semantic,
            Self::Temporal,
            Self::Causal,
            Self::Hierarchical,
            Self::Contradicts,
        ]
    }

    /// Returns the default base weight for this edge type.
    ///
    /// These weights reflect the inherent reliability of each relationship type:
    /// - Semantic (0.5): Variable based on embedding similarity
    /// - Temporal (0.7): Time relationships are usually reliable
    /// - Causal (0.8): Strong evidence when established
    /// - Hierarchical (0.9): Taxonomy relationships are very strong
    /// - Contradicts (0.3): Low base weight, inhibitory NT modulation applied
    #[inline]
    pub fn default_weight(&self) -> f32 {
        match self {
            Self::Semantic => 0.5,
            Self::Temporal => 0.7,
            Self::Causal => 0.8,
            Self::Hierarchical => 0.9,
            Self::Contradicts => 0.3,
        }
    }

    /// Returns whether this edge type represents a contradiction relationship.
    ///
    /// # M04-T26: NT Modulation for Contradiction Detection
    ///
    /// Contradiction edges require special NT modulation (inhibitory-heavy)
    /// to suppress contradicting content during retrieval.
    #[inline]
    pub fn is_contradiction(&self) -> bool {
        matches!(self, Self::Contradicts)
    }

    /// Returns whether this edge type is symmetric.
    ///
    /// # M04-T26: Contradiction Edge Symmetry
    ///
    /// Symmetric edges work in both directions:
    /// - Semantic: A similar to B implies B similar to A
    /// - Contradicts: A contradicts B implies B contradicts A
    ///
    /// Non-symmetric (directed) edges:
    /// - Temporal: A before B does NOT imply B before A
    /// - Causal: A causes B does NOT imply B causes A
    /// - Hierarchical: A parent of B does NOT imply B parent of A
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        matches!(self, Self::Semantic | Self::Contradicts)
    }

    /// Converts this edge type to its u8 representation for storage.
    ///
    /// # M04-T15: Edge Storage Integration
    ///
    /// Used for compact storage in RocksDB CF_EDGES column family.
    ///
    /// # Returns
    ///
    /// - Semantic: 0
    /// - Temporal: 1
    /// - Causal: 2
    /// - Hierarchical: 3
    /// - Contradicts: 4
    #[inline]
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::Semantic => 0,
            Self::Temporal => 1,
            Self::Causal => 2,
            Self::Hierarchical => 3,
            Self::Contradicts => 4,
        }
    }

    /// Converts a u8 value to an EdgeType.
    ///
    /// # M04-T15: Edge Storage Integration
    ///
    /// Used for deserializing from RocksDB CF_EDGES column family.
    ///
    /// # Arguments
    ///
    /// * `value` - The u8 representation (0-4)
    ///
    /// # Returns
    ///
    /// - `Some(EdgeType)` if value is 0-4
    /// - `None` if value is out of range
    #[inline]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Semantic),
            1 => Some(Self::Temporal),
            2 => Some(Self::Causal),
            3 => Some(Self::Hierarchical),
            4 => Some(Self::Contradicts),
            _ => None,
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
            Self::Contradicts => "contradicts",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== as_u8 / from_u8 Tests (M04-T15) ==========

    #[test]
    fn test_edge_type_as_u8() {
        assert_eq!(EdgeType::Semantic.as_u8(), 0);
        assert_eq!(EdgeType::Temporal.as_u8(), 1);
        assert_eq!(EdgeType::Causal.as_u8(), 2);
        assert_eq!(EdgeType::Hierarchical.as_u8(), 3);
        assert_eq!(EdgeType::Contradicts.as_u8(), 4);
    }

    #[test]
    fn test_edge_type_from_u8_valid() {
        assert_eq!(EdgeType::from_u8(0), Some(EdgeType::Semantic));
        assert_eq!(EdgeType::from_u8(1), Some(EdgeType::Temporal));
        assert_eq!(EdgeType::from_u8(2), Some(EdgeType::Causal));
        assert_eq!(EdgeType::from_u8(3), Some(EdgeType::Hierarchical));
        assert_eq!(EdgeType::from_u8(4), Some(EdgeType::Contradicts));
    }

    #[test]
    fn test_edge_type_from_u8_contradicts() {
        assert_eq!(EdgeType::from_u8(4), Some(EdgeType::Contradicts));
    }

    #[test]
    fn test_edge_type_from_u8_invalid() {
        assert_eq!(EdgeType::from_u8(5), None);
        assert_eq!(EdgeType::from_u8(255), None);
    }

    #[test]
    fn test_edge_type_roundtrip() {
        for edge_type in EdgeType::all() {
            let u8_val = edge_type.as_u8();
            let recovered = EdgeType::from_u8(u8_val).expect("valid u8 should convert");
            assert_eq!(recovered, edge_type);
        }
    }

    // ========== Existing Tests ==========

    #[test]
    fn test_default_weight() {
        assert_eq!(EdgeType::Semantic.default_weight(), 0.5);
        assert_eq!(EdgeType::Temporal.default_weight(), 0.7);
        assert_eq!(EdgeType::Causal.default_weight(), 0.8);
        assert_eq!(EdgeType::Hierarchical.default_weight(), 0.9);
        assert_eq!(EdgeType::Contradicts.default_weight(), 0.3);
    }

    #[test]
    fn test_display() {
        assert_eq!(EdgeType::Semantic.to_string(), "semantic");
        assert_eq!(EdgeType::Temporal.to_string(), "temporal");
        assert_eq!(EdgeType::Causal.to_string(), "causal");
        assert_eq!(EdgeType::Hierarchical.to_string(), "hierarchical");
        assert_eq!(EdgeType::Contradicts.to_string(), "contradicts");
    }

    #[test]
    fn test_default() {
        assert_eq!(EdgeType::default(), EdgeType::Semantic);
    }

    #[test]
    fn test_all() {
        let all = EdgeType::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&EdgeType::Semantic));
        assert!(all.contains(&EdgeType::Temporal));
        assert!(all.contains(&EdgeType::Causal));
        assert!(all.contains(&EdgeType::Hierarchical));
        assert!(all.contains(&EdgeType::Contradicts));
    }

    // ========== M04-T26: Contradicts Tests ==========

    #[test]
    fn test_contradicts_is_contradiction() {
        assert!(EdgeType::Contradicts.is_contradiction());
        assert!(!EdgeType::Semantic.is_contradiction());
        assert!(!EdgeType::Temporal.is_contradiction());
        assert!(!EdgeType::Causal.is_contradiction());
        assert!(!EdgeType::Hierarchical.is_contradiction());
    }

    #[test]
    fn test_is_symmetric() {
        // Symmetric: Semantic and Contradicts
        assert!(EdgeType::Semantic.is_symmetric());
        assert!(EdgeType::Contradicts.is_symmetric());
        // Directional: Temporal, Causal, Hierarchical
        assert!(!EdgeType::Temporal.is_symmetric());
        assert!(!EdgeType::Causal.is_symmetric());
        assert!(!EdgeType::Hierarchical.is_symmetric());
    }

    #[test]
    fn test_contradicts_has_low_weight() {
        // Contradicts should have lowest weight
        assert!(EdgeType::Contradicts.default_weight() < EdgeType::Semantic.default_weight());
    }

    #[test]
    fn test_contradicts_serde() {
        let json = serde_json::to_string(&EdgeType::Contradicts).unwrap();
        assert_eq!(json, r#""contradicts""#);
        let restored: EdgeType = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, EdgeType::Contradicts);
    }
}
