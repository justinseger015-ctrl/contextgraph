//! Marblestone architecture integration for context-aware neurotransmitter weighting.
//!
//! This module provides domain classification for the Marblestone edge model,
//! enabling context-specific retrieval behavior in the knowledge graph.
//!
//! # Constitution Reference
//! - edge_model.nt_weights.domain: Code|Legal|Medical|Creative|Research|General
//! - Formula: w_eff = base × (1 + excitatory - inhibitory + 0.5×modulatory)

use serde::{Deserialize, Serialize};
use std::fmt;

/// Knowledge domain for context-aware neurotransmitter weighting.
///
/// Different domains have different optimal retrieval characteristics:
/// - Code: High precision, structured relationships
/// - Legal: High inhibition, careful reasoning
/// - Medical: High causal awareness, evidence-based
/// - Creative: High exploration, associative connections
/// - Research: Balanced exploration and precision
/// - General: Default balanced profile
///
/// # Constitution Compliance
/// - Naming: PascalCase enum per constitution.yaml
/// - Serde: snake_case serialization per JSON naming rules
///
/// # Example
/// ```rust
/// use context_graph_core::marblestone::Domain;
///
/// let domain = Domain::Code;
/// assert_eq!(domain.to_string(), "code");
/// assert!(domain.description().contains("precision"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    /// Programming and software development context.
    /// Characteristics: High precision, structured relationships, strong type awareness.
    Code,
    /// Legal documents and reasoning context.
    /// Characteristics: High inhibition, careful reasoning, precedent-based.
    Legal,
    /// Medical and healthcare context.
    /// Characteristics: High causal awareness, evidence-based, risk-conscious.
    Medical,
    /// Creative writing and artistic context.
    /// Characteristics: High exploration, associative connections, novelty-seeking.
    Creative,
    /// Academic research context.
    /// Characteristics: Balanced exploration and precision, citation-aware.
    Research,
    /// General purpose context.
    /// Characteristics: Default balanced profile for mixed contexts.
    General,
}

impl Domain {
    /// Returns a human-readable description of this domain's characteristics.
    ///
    /// # Returns
    /// Static string describing the domain's retrieval behavior.
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::marblestone::Domain;
    ///
    /// let desc = Domain::Medical.description();
    /// assert!(desc.contains("causal"));
    /// ```
    #[inline]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Code => "High precision, structured relationships, strong type awareness",
            Self::Legal => "High inhibition, careful reasoning, precedent-based",
            Self::Medical => "High causal awareness, evidence-based, risk-conscious",
            Self::Creative => "High exploration, associative connections, novelty-seeking",
            Self::Research => "Balanced exploration and precision, citation-aware",
            Self::General => "Default balanced profile for mixed contexts",
        }
    }

    /// Returns all domain variants as an array.
    ///
    /// # Returns
    /// Array containing all 6 Domain variants in definition order.
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::marblestone::Domain;
    ///
    /// let all = Domain::all();
    /// assert_eq!(all.len(), 6);
    /// assert_eq!(all[0], Domain::Code);
    /// assert_eq!(all[5], Domain::General);
    /// ```
    #[inline]
    pub fn all() -> [Domain; 6] {
        [
            Self::Code,
            Self::Legal,
            Self::Medical,
            Self::Creative,
            Self::Research,
            Self::General,
        ]
    }
}

impl Default for Domain {
    /// Returns `Domain::General` as the default.
    ///
    /// General is the most balanced profile, suitable for mixed contexts.
    #[inline]
    fn default() -> Self {
        Self::General
    }
}

impl fmt::Display for Domain {
    /// Formats the domain as a lowercase string.
    ///
    /// # Output
    /// - Code → "code"
    /// - Legal → "legal"
    /// - Medical → "medical"
    /// - Creative → "creative"
    /// - Research → "research"
    /// - General → "general"
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Code => "code",
            Self::Legal => "legal",
            Self::Medical => "medical",
            Self::Creative => "creative",
            Self::Research => "research",
            Self::General => "general",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Default Implementation Tests
    // =========================================================================

    #[test]
    fn test_default_is_general() {
        let domain = Domain::default();
        assert_eq!(domain, Domain::General, "Default domain must be General");
    }

    // =========================================================================
    // Description Method Tests
    // =========================================================================

    #[test]
    fn test_description_non_empty_for_all_variants() {
        for domain in Domain::all() {
            let desc = domain.description();
            assert!(!desc.is_empty(), "Description for {:?} must not be empty", domain);
            assert!(desc.len() > 10, "Description for {:?} should be meaningful", domain);
        }
    }

    #[test]
    fn test_code_description_mentions_precision() {
        assert!(Domain::Code.description().to_lowercase().contains("precision"));
    }

    #[test]
    fn test_legal_description_mentions_reasoning() {
        assert!(Domain::Legal.description().to_lowercase().contains("reasoning"));
    }

    #[test]
    fn test_medical_description_mentions_causal() {
        assert!(Domain::Medical.description().to_lowercase().contains("causal"));
    }

    #[test]
    fn test_creative_description_mentions_exploration() {
        assert!(Domain::Creative.description().to_lowercase().contains("exploration"));
    }

    #[test]
    fn test_research_description_mentions_balanced() {
        assert!(Domain::Research.description().to_lowercase().contains("balanced"));
    }

    #[test]
    fn test_general_description_mentions_default() {
        assert!(Domain::General.description().to_lowercase().contains("default"));
    }

    // =========================================================================
    // all() Method Tests
    // =========================================================================

    #[test]
    fn test_all_returns_6_variants() {
        let all = Domain::all();
        assert_eq!(all.len(), 6, "Domain::all() must return exactly 6 variants");
    }

    #[test]
    fn test_all_contains_all_variants() {
        let all = Domain::all();
        assert!(all.contains(&Domain::Code));
        assert!(all.contains(&Domain::Legal));
        assert!(all.contains(&Domain::Medical));
        assert!(all.contains(&Domain::Creative));
        assert!(all.contains(&Domain::Research));
        assert!(all.contains(&Domain::General));
    }

    #[test]
    fn test_all_order_matches_definition() {
        let all = Domain::all();
        assert_eq!(all[0], Domain::Code);
        assert_eq!(all[1], Domain::Legal);
        assert_eq!(all[2], Domain::Medical);
        assert_eq!(all[3], Domain::Creative);
        assert_eq!(all[4], Domain::Research);
        assert_eq!(all[5], Domain::General);
    }

    // =========================================================================
    // Display Trait Tests
    // =========================================================================

    #[test]
    fn test_display_code() {
        assert_eq!(Domain::Code.to_string(), "code");
    }

    #[test]
    fn test_display_legal() {
        assert_eq!(Domain::Legal.to_string(), "legal");
    }

    #[test]
    fn test_display_medical() {
        assert_eq!(Domain::Medical.to_string(), "medical");
    }

    #[test]
    fn test_display_creative() {
        assert_eq!(Domain::Creative.to_string(), "creative");
    }

    #[test]
    fn test_display_research() {
        assert_eq!(Domain::Research.to_string(), "research");
    }

    #[test]
    fn test_display_general() {
        assert_eq!(Domain::General.to_string(), "general");
    }

    #[test]
    fn test_display_all_lowercase() {
        for domain in Domain::all() {
            let s = domain.to_string();
            assert_eq!(s, s.to_lowercase(), "Display for {:?} must be lowercase", domain);
        }
    }

    // =========================================================================
    // Serde Serialization Tests
    // =========================================================================

    #[test]
    fn test_serde_serializes_to_lowercase() {
        let domain = Domain::Code;
        let json = serde_json::to_string(&domain).expect("serialize failed");
        assert_eq!(json, r#""code""#, "Serde must serialize to lowercase");
    }

    #[test]
    fn test_serde_deserializes_from_lowercase() {
        let domain: Domain = serde_json::from_str(r#""legal""#).expect("deserialize failed");
        assert_eq!(domain, Domain::Legal);
    }

    #[test]
    fn test_serde_roundtrip_all_variants() {
        for domain in Domain::all() {
            let json = serde_json::to_string(&domain).expect("serialize failed");
            let restored: Domain = serde_json::from_str(&json).expect("deserialize failed");
            assert_eq!(restored, domain, "Serde roundtrip failed for {:?}", domain);
        }
    }

    #[test]
    fn test_serde_snake_case_format() {
        // All variants should serialize to snake_case (which for single words is just lowercase)
        for domain in Domain::all() {
            let json = serde_json::to_string(&domain).unwrap();
            // Remove quotes
            let value = json.trim_matches('"');
            assert!(value.chars().all(|c| c.is_lowercase() || c == '_'),
                "Serde output for {:?} must be snake_case: {}", domain, value);
        }
    }

    // =========================================================================
    // Derive Trait Tests
    // =========================================================================

    #[test]
    fn test_clone() {
        let domain = Domain::Medical;
        let cloned = domain.clone();
        assert_eq!(domain, cloned);
    }

    #[test]
    fn test_copy() {
        let domain = Domain::Creative;
        let copied = domain; // Copy, not move
        assert_eq!(domain, copied);
        let _still_valid = domain; // Can still use original
    }

    #[test]
    fn test_debug_format() {
        let debug = format!("{:?}", Domain::Research);
        assert!(debug.contains("Research"), "Debug should show variant name");
    }

    #[test]
    fn test_partial_eq() {
        assert_eq!(Domain::Code, Domain::Code);
        assert_ne!(Domain::Code, Domain::Legal);
    }

    #[test]
    fn test_hash_in_collection() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Domain::Code);
        set.insert(Domain::Legal);
        set.insert(Domain::Code); // Duplicate
        assert_eq!(set.len(), 2, "Hash must properly deduplicate");
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_all_variants_unique() {
        use std::collections::HashSet;
        let all = Domain::all();
        let unique: HashSet<_> = all.iter().collect();
        assert_eq!(unique.len(), 6, "All variants must be unique");
    }

    #[test]
    fn test_default_is_in_all() {
        let default = Domain::default();
        assert!(Domain::all().contains(&default), "Default must be in all()");
    }
}
