//! Entity types for E11 KEPLER embeddings.
//!
//! KEPLER (Knowledge Embedding and Pre-training for Language Representation) is trained
//! on Wikidata5M (4.8M entities, 20M triples) with TransE objective. The model's
//! embeddings ALREADY encode entity knowledge - no keyword lookup needed.
//!
//! # Philosophy
//!
//! KEPLER embeddings ARE the entity detection:
//! - Query "database" → KEPLER embedding
//! - Memory about "Diesel" → KEPLER embedding
//! - KEPLER similarity finds relationship (Diesel IS a database ORM)
//!
//! We don't need a keyword lookup table because:
//! 1. KEPLER was trained on 4.8M entities from Wikidata
//! 2. TransE training encodes relationships (h + r ≈ t)
//! 3. Embedding similarity captures entity relationships
//!
//! # What This Module Provides
//!
//! - `EntityType`: Categories for display/grouping only
//! - `EntityLink`: Surface form with optional type annotation
//! - `EntityMetadata`: Collection of entities (for API responses)
//!
//! Entity "detection" is now simply: embed with KEPLER, search by similarity.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Categories of entities for display/grouping purposes.
///
/// Note: KEPLER doesn't need these for search - its embeddings already
/// encode entity relationships. These are for UI/API display only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EntityType {
    /// Programming languages
    ProgrammingLanguage,
    /// Frameworks and libraries
    Framework,
    /// Databases
    Database,
    /// Cloud providers and services
    Cloud,
    /// Companies and products
    Company,
    /// Technical terms and protocols
    TechnicalTerm,
    /// Unknown/general entity (default)
    #[default]
    Unknown,
}

/// An entity reference with surface form.
///
/// The canonical_id is simply the lowercase normalized form.
/// KEPLER embeddings handle actual entity resolution.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityLink {
    /// Raw entity text as provided
    pub surface_form: String,
    /// Lowercase normalized form
    pub canonical_id: String,
    /// Entity type (for display only - KEPLER doesn't need this)
    pub entity_type: EntityType,
}

impl EntityLink {
    /// Create a new entity link from surface form.
    /// Canonical ID is simply the lowercase form.
    pub fn new(surface_form: &str) -> Self {
        Self {
            surface_form: surface_form.to_string(),
            canonical_id: surface_form.to_lowercase(),
            entity_type: EntityType::Unknown,
        }
    }

    /// Create with explicit type annotation.
    pub fn with_type(surface_form: &str, entity_type: EntityType) -> Self {
        Self {
            surface_form: surface_form.to_string(),
            canonical_id: surface_form.to_lowercase(),
            entity_type,
        }
    }
}

/// Collection of entities.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityMetadata {
    /// All entities
    pub entities: Vec<EntityLink>,
}

impl EntityMetadata {
    /// Create empty metadata
    pub fn empty() -> Self {
        Self {
            entities: Vec::new(),
        }
    }

    /// Create from a list of entities
    pub fn from_entities(entities: Vec<EntityLink>) -> Self {
        Self { entities }
    }

    /// Create from a list of surface forms (all Unknown type)
    pub fn from_surface_forms(forms: &[&str]) -> Self {
        Self {
            entities: forms.iter().map(|s| EntityLink::new(s)).collect(),
        }
    }

    /// Get set of canonical IDs
    pub fn canonical_ids(&self) -> HashSet<&str> {
        self.entities
            .iter()
            .map(|e| e.canonical_id.as_str())
            .collect()
    }

    /// Check if metadata is empty
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Get entity count
    pub fn len(&self) -> usize {
        self.entities.len()
    }
}

/// Compute Jaccard similarity between two entity sets.
///
/// Jaccard(A, B) = |A ∩ B| / |A ∪ B|
///
/// Note: This is for string-level overlap only. KEPLER embedding
/// similarity is the primary entity relationship measure.
pub fn entity_jaccard_similarity(a: &EntityMetadata, b: &EntityMetadata) -> f32 {
    let set_a = a.canonical_ids();
    let set_b = b.canonical_ids();

    if set_a.is_empty() && set_b.is_empty() {
        return 0.0;
    }

    let intersection: HashSet<_> = set_a.intersection(&set_b).collect();
    let union: HashSet<_> = set_a.union(&set_b).collect();

    intersection.len() as f32 / union.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_link_new() {
        let link = EntityLink::new("PostgreSQL");
        assert_eq!(link.surface_form, "PostgreSQL");
        assert_eq!(link.canonical_id, "postgresql");
        assert_eq!(link.entity_type, EntityType::Unknown);
    }

    #[test]
    fn test_entity_link_with_type() {
        let link = EntityLink::with_type("Rust", EntityType::ProgrammingLanguage);
        assert_eq!(link.surface_form, "Rust");
        assert_eq!(link.canonical_id, "rust");
        assert_eq!(link.entity_type, EntityType::ProgrammingLanguage);
    }

    #[test]
    fn test_entity_metadata_from_surface_forms() {
        let meta = EntityMetadata::from_surface_forms(&["Rust", "PostgreSQL"]);
        assert_eq!(meta.len(), 2);
        let ids = meta.canonical_ids();
        assert!(ids.contains("rust"));
        assert!(ids.contains("postgresql"));
    }

    #[test]
    fn test_entity_jaccard_identical() {
        let a = EntityMetadata::from_surface_forms(&["Rust", "Python"]);
        let b = EntityMetadata::from_surface_forms(&["rust", "python"]); // Same canonical
        assert!((entity_jaccard_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_entity_jaccard_disjoint() {
        let a = EntityMetadata::from_surface_forms(&["Rust"]);
        let b = EntityMetadata::from_surface_forms(&["Python"]);
        assert!(entity_jaccard_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_entity_jaccard_partial() {
        let a = EntityMetadata::from_surface_forms(&["Rust", "PostgreSQL"]);
        let b = EntityMetadata::from_surface_forms(&["Rust", "Redis"]);
        // Intersection: {rust}, Union: {rust, postgresql, redis}
        let jaccard = entity_jaccard_similarity(&a, &b);
        assert!((jaccard - 0.333).abs() < 0.01);
    }
}
