//! Entity detection and linking module for E11 Entity embeddings.
//!
//! Per ARCH-17: "E11 Entity SHOULD use entity linking for disambiguation when possible,
//! not just embedding similarity."
//!
//! This module provides:
//! 1. Entity detection from text using pattern matching
//! 2. Entity linking to canonical forms (disambiguation)
//! 3. Entity-aware similarity computation combining embedding + entity overlap
//!
//! # Entity Types
//!
//! - **Programming**: Language names (Rust, Python), frameworks (React, Django)
//! - **Database**: PostgreSQL, MySQL, MongoDB, Redis
//! - **Cloud**: AWS, GCP, Azure services
//! - **Company/Product**: Anthropic, OpenAI, Google, Microsoft
//! - **Technical Terms**: API, REST, GraphQL, gRPC
//!
//! # Similarity Computation
//!
//! Entity-aware similarity combines:
//! - Embedding similarity (E11 cosine): semantic meaning
//! - Entity Jaccard overlap: exact entity matching
//!
//! Formula: `final = embedding_weight * cos_sim + entity_weight * jaccard`
//! Default weights: embedding=0.7, entity=0.3 (per research on entity matching)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// An extracted entity with optional canonical link.
///
/// Entity linking resolves variations to canonical forms:
/// - "Rust lang" → "rust_language"
/// - "rustlang" → "rust_language"
/// - "postgresql" → "postgresql"
/// - "postgres" → "postgresql"
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityLink {
    /// Raw entity text as extracted from content
    pub surface_form: String,
    /// Canonical entity identifier (lowercase, normalized)
    pub canonical_id: String,
    /// Entity type category
    pub entity_type: EntityType,
}

/// Categories of entities for domain-aware processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// Programming languages (Rust, Python, JavaScript)
    ProgrammingLanguage,
    /// Frameworks and libraries (React, Django, FastAPI)
    Framework,
    /// Databases (PostgreSQL, MySQL, Redis)
    Database,
    /// Cloud providers and services (AWS S3, GCP BigQuery)
    Cloud,
    /// Companies and products (Anthropic, OpenAI, Claude)
    Company,
    /// Technical terms and protocols (REST, GraphQL, gRPC)
    TechnicalTerm,
    /// Unknown/general entity
    Unknown,
}

/// Collection of entities extracted from text.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityMetadata {
    /// All extracted entities with canonical links
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

    /// Get set of canonical IDs for Jaccard similarity
    pub fn canonical_ids(&self) -> HashSet<&str> {
        self.entities.iter().map(|e| e.canonical_id.as_str()).collect()
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

/// Entity disambiguation knowledge base.
///
/// Maps surface forms to canonical identifiers for disambiguation.
/// e.g., "postgres" → "postgresql", "pg" → "postgresql"
pub fn get_entity_kb() -> HashMap<&'static str, (&'static str, EntityType)> {
    let mut kb = HashMap::new();

    // Programming Languages
    for (surface, canonical) in [
        ("rust", "rust_language"),
        ("rustlang", "rust_language"),
        ("rust lang", "rust_language"),
        ("python", "python"),
        ("py", "python"),
        ("python3", "python"),
        ("javascript", "javascript"),
        ("js", "javascript"),
        ("typescript", "typescript"),
        ("ts", "typescript"),
        ("golang", "go_language"),
        ("go lang", "go_language"),
        ("java", "java"),
        ("c++", "cpp"),
        ("cpp", "cpp"),
        ("csharp", "csharp"),
        ("c#", "csharp"),
    ] {
        kb.insert(surface, (canonical, EntityType::ProgrammingLanguage));
    }

    // Databases
    for (surface, canonical) in [
        ("postgresql", "postgresql"),
        ("postgres", "postgresql"),
        ("pg", "postgresql"),
        ("mysql", "mysql"),
        ("mongodb", "mongodb"),
        ("mongo", "mongodb"),
        ("redis", "redis"),
        ("sqlite", "sqlite"),
        ("rocksdb", "rocksdb"),
        ("dynamodb", "dynamodb"),
        ("cassandra", "cassandra"),
    ] {
        kb.insert(surface, (canonical, EntityType::Database));
    }

    // Frameworks
    for (surface, canonical) in [
        ("react", "react"),
        ("reactjs", "react"),
        ("react.js", "react"),
        ("angular", "angular"),
        ("vue", "vuejs"),
        ("vuejs", "vuejs"),
        ("vue.js", "vuejs"),
        ("django", "django"),
        ("fastapi", "fastapi"),
        ("flask", "flask"),
        ("express", "expressjs"),
        ("expressjs", "expressjs"),
        ("nextjs", "nextjs"),
        ("next.js", "nextjs"),
        ("tokio", "tokio"),
        ("actix", "actix"),
        ("axum", "axum"),
    ] {
        kb.insert(surface, (canonical, EntityType::Framework));
    }

    // Cloud Services
    for (surface, canonical) in [
        ("aws", "aws"),
        ("amazon web services", "aws"),
        ("s3", "aws_s3"),
        ("ec2", "aws_ec2"),
        ("lambda", "aws_lambda"),
        ("gcp", "gcp"),
        ("google cloud", "gcp"),
        ("bigquery", "gcp_bigquery"),
        ("azure", "azure"),
        ("kubernetes", "kubernetes"),
        ("k8s", "kubernetes"),
        ("docker", "docker"),
    ] {
        kb.insert(surface, (canonical, EntityType::Cloud));
    }

    // Companies/Products
    for (surface, canonical) in [
        ("anthropic", "anthropic"),
        ("claude", "anthropic_claude"),
        ("openai", "openai"),
        ("gpt", "openai_gpt"),
        ("chatgpt", "openai_chatgpt"),
        ("google", "google"),
        ("microsoft", "microsoft"),
        ("github", "github"),
        ("gitlab", "gitlab"),
    ] {
        kb.insert(surface, (canonical, EntityType::Company));
    }

    // Technical Terms
    for (surface, canonical) in [
        ("rest", "rest_api"),
        ("rest api", "rest_api"),
        ("restful", "rest_api"),
        ("graphql", "graphql"),
        ("grpc", "grpc"),
        ("websocket", "websocket"),
        ("websockets", "websocket"),
        ("http", "http"),
        ("https", "https"),
        ("json", "json"),
        ("yaml", "yaml"),
        ("toml", "toml"),
        ("hnsw", "hnsw"),
        ("faiss", "faiss"),
        ("colbert", "colbert"),
        ("splade", "splade"),
    ] {
        kb.insert(surface, (canonical, EntityType::TechnicalTerm));
    }

    kb
}

/// Detect entities in text using pattern matching and KB lookup.
///
/// This is a lightweight entity detection that:
/// 1. Extracts potential entity spans (capitalized words, technical terms)
/// 2. Looks up canonical forms in the knowledge base
/// 3. Returns linked entities with canonical IDs
///
/// # Arguments
/// * `text` - Input text to extract entities from
///
/// # Returns
/// EntityMetadata containing all detected entities with canonical links
pub fn detect_entities(text: &str) -> EntityMetadata {
    let kb = get_entity_kb();
    let mut entities = Vec::new();
    let mut seen_canonical: HashSet<String> = HashSet::new();

    // Normalize text for matching
    let lower = text.to_lowercase();

    // Check each KB entry against the text
    for (surface, (canonical, entity_type)) in &kb {
        if lower.contains(surface) && !seen_canonical.contains(*canonical) {
            entities.push(EntityLink {
                surface_form: surface.to_string(),
                canonical_id: canonical.to_string(),
                entity_type: *entity_type,
            });
            seen_canonical.insert(canonical.to_string());
        }
    }

    // Also detect capitalized sequences that might be entities
    // (heuristic for proper nouns not in KB)
    let words: Vec<&str> = text.split_whitespace().collect();
    for word in words {
        // Check if word is capitalized and not a sentence start
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        if clean.len() >= 2
            && clean.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
        {
            // Check if it looks like a proper noun (not common English word)
            let lower_clean = clean.to_lowercase();
            if !is_common_word(&lower_clean) && !seen_canonical.contains(&lower_clean) {
                entities.push(EntityLink {
                    surface_form: clean.to_string(),
                    canonical_id: lower_clean.clone(),
                    entity_type: EntityType::Unknown,
                });
                seen_canonical.insert(lower_clean);
            }
        }
    }

    EntityMetadata::from_entities(entities)
}

/// Check if a word is a common English word (not an entity).
fn is_common_word(word: &str) -> bool {
    const COMMON_WORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "up", "about", "into", "over", "after", "this", "that",
        "these", "those", "then", "than", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "no", "not", "only", "same", "so", "and", "but",
        "if", "or", "because", "as", "until", "while", "although", "though",
        "after", "before", "since", "when", "where", "why", "how", "what",
        "which", "who", "whom", "whose", "it", "its", "they", "them", "their",
        "he", "she", "him", "her", "his", "i", "me", "my", "we", "us", "our",
        "you", "your", "here", "there", "now", "today", "use", "using", "used",
        "function", "method", "class", "struct", "type", "error", "result",
        "value", "data", "code", "file", "module", "test", "return", "new",
    ];
    COMMON_WORDS.contains(&word)
}

/// Compute Jaccard similarity between two entity sets.
///
/// Jaccard(A, B) = |A ∩ B| / |A ∪ B|
///
/// Returns 0.0 if both sets are empty, 1.0 if identical.
pub fn entity_jaccard_similarity(a: &EntityMetadata, b: &EntityMetadata) -> f32 {
    let set_a = a.canonical_ids();
    let set_b = b.canonical_ids();

    if set_a.is_empty() && set_b.is_empty() {
        return 0.0; // No entities in either - no entity-based similarity
    }

    let intersection: HashSet<_> = set_a.intersection(&set_b).collect();
    let union: HashSet<_> = set_a.union(&set_b).collect();

    intersection.len() as f32 / union.len() as f32
}

/// Compute combined E11 similarity with entity linking.
///
/// Combines embedding cosine similarity with entity Jaccard overlap.
///
/// # Arguments
/// * `e11_a` - E11 embedding vector for document A
/// * `e11_b` - E11 embedding vector for document B
/// * `entities_a` - Entity metadata for document A
/// * `entities_b` - Entity metadata for document B
/// * `embedding_weight` - Weight for embedding similarity (default 0.7)
/// * `entity_weight` - Weight for entity overlap (default 0.3)
///
/// # Returns
/// Combined similarity score [0.0, 1.0]
pub fn compute_e11_similarity_with_entities(
    e11_a: &[f32],
    e11_b: &[f32],
    entities_a: &EntityMetadata,
    entities_b: &EntityMetadata,
    embedding_weight: f32,
    entity_weight: f32,
) -> f32 {
    // Compute embedding cosine similarity
    let embedding_sim = cosine_similarity(e11_a, e11_b);

    // Compute entity Jaccard similarity
    let entity_sim = entity_jaccard_similarity(entities_a, entities_b);

    // Combine with weights
    let total_weight = embedding_weight + entity_weight;
    if total_weight <= 0.0 {
        return embedding_sim;
    }

    (embedding_weight * embedding_sim + entity_weight * entity_sim) / total_weight
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a.sqrt()) * (norm_b.sqrt());
    if denom < f32::EPSILON {
        return 0.0;
    }

    (dot / denom).clamp(0.0, 1.0)
}

/// Default weight for embedding similarity in combined score
pub const DEFAULT_EMBEDDING_WEIGHT: f32 = 0.7;

/// Default weight for entity overlap in combined score
pub const DEFAULT_ENTITY_WEIGHT: f32 = 0.3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_entities_programming_languages() {
        let text = "I'm learning Rust and Python for systems programming";
        let entities = detect_entities(text);

        let canonical_ids: HashSet<_> = entities.canonical_ids().into_iter().collect();
        assert!(
            canonical_ids.contains("rust_language"),
            "Should detect Rust -> rust_language"
        );
        assert!(
            canonical_ids.contains("python"),
            "Should detect Python -> python"
        );
    }

    #[test]
    fn test_detect_entities_databases() {
        let text = "Migrating from PostgreSQL to MongoDB";
        let entities = detect_entities(text);

        let canonical_ids: HashSet<_> = entities.canonical_ids().into_iter().collect();
        assert!(
            canonical_ids.contains("postgresql"),
            "Should detect PostgreSQL"
        );
        assert!(canonical_ids.contains("mongodb"), "Should detect MongoDB");
    }

    #[test]
    fn test_entity_disambiguation() {
        // "postgres" and "postgresql" should map to same canonical ID
        let text1 = "Using postgres for the backend";
        let text2 = "Using postgresql for the backend";

        let entities1 = detect_entities(text1);
        let entities2 = detect_entities(text2);

        assert_eq!(
            entities1.canonical_ids(),
            entities2.canonical_ids(),
            "postgres and postgresql should have same canonical ID"
        );
    }

    #[test]
    fn test_entity_jaccard_similarity() {
        let entities_a = EntityMetadata::from_entities(vec![
            EntityLink {
                surface_form: "Rust".to_string(),
                canonical_id: "rust_language".to_string(),
                entity_type: EntityType::ProgrammingLanguage,
            },
            EntityLink {
                surface_form: "PostgreSQL".to_string(),
                canonical_id: "postgresql".to_string(),
                entity_type: EntityType::Database,
            },
        ]);

        let entities_b = EntityMetadata::from_entities(vec![
            EntityLink {
                surface_form: "Rust".to_string(),
                canonical_id: "rust_language".to_string(),
                entity_type: EntityType::ProgrammingLanguage,
            },
            EntityLink {
                surface_form: "Redis".to_string(),
                canonical_id: "redis".to_string(),
                entity_type: EntityType::Database,
            },
        ]);

        let jaccard = entity_jaccard_similarity(&entities_a, &entities_b);
        // Intersection: {rust_language}, Union: {rust_language, postgresql, redis}
        // Jaccard = 1/3 ≈ 0.333
        assert!(
            (jaccard - 0.333).abs() < 0.01,
            "Jaccard should be ~0.333, got {}",
            jaccard
        );
    }

    #[test]
    fn test_entity_jaccard_identical_sets() {
        let entities = EntityMetadata::from_entities(vec![EntityLink {
            surface_form: "Rust".to_string(),
            canonical_id: "rust_language".to_string(),
            entity_type: EntityType::ProgrammingLanguage,
        }]);

        let jaccard = entity_jaccard_similarity(&entities, &entities);
        assert!(
            (jaccard - 1.0).abs() < 0.001,
            "Identical sets should have Jaccard = 1.0"
        );
    }

    #[test]
    fn test_entity_jaccard_disjoint_sets() {
        let entities_a = EntityMetadata::from_entities(vec![EntityLink {
            surface_form: "Rust".to_string(),
            canonical_id: "rust_language".to_string(),
            entity_type: EntityType::ProgrammingLanguage,
        }]);

        let entities_b = EntityMetadata::from_entities(vec![EntityLink {
            surface_form: "Python".to_string(),
            canonical_id: "python".to_string(),
            entity_type: EntityType::ProgrammingLanguage,
        }]);

        let jaccard = entity_jaccard_similarity(&entities_a, &entities_b);
        assert!(
            jaccard.abs() < 0.001,
            "Disjoint sets should have Jaccard = 0.0"
        );
    }

    #[test]
    fn test_combined_e11_similarity() {
        // Two similar embeddings with different entities
        let e11_a = vec![1.0, 0.5, 0.3];
        let e11_b = vec![0.9, 0.5, 0.4];

        let entities_a = EntityMetadata::from_entities(vec![EntityLink {
            surface_form: "Rust".to_string(),
            canonical_id: "rust_language".to_string(),
            entity_type: EntityType::ProgrammingLanguage,
        }]);

        let entities_b = EntityMetadata::from_entities(vec![EntityLink {
            surface_form: "Rust".to_string(),
            canonical_id: "rust_language".to_string(),
            entity_type: EntityType::ProgrammingLanguage,
        }]);

        let sim = compute_e11_similarity_with_entities(
            &e11_a,
            &e11_b,
            &entities_a,
            &entities_b,
            DEFAULT_EMBEDDING_WEIGHT,
            DEFAULT_ENTITY_WEIGHT,
        );

        // Both have same entity (Jaccard = 1.0) and similar embeddings
        // Combined should be high
        assert!(sim > 0.9, "Combined similarity should be high, got {}", sim);
    }

    #[test]
    fn test_combined_e11_entity_mismatch() {
        // Similar embeddings but different entities
        let e11_a = vec![1.0, 0.5, 0.3];
        let e11_b = vec![0.95, 0.5, 0.35]; // Very similar embedding

        let entities_a = EntityMetadata::from_entities(vec![EntityLink {
            surface_form: "PostgreSQL".to_string(),
            canonical_id: "postgresql".to_string(),
            entity_type: EntityType::Database,
        }]);

        let entities_b = EntityMetadata::from_entities(vec![EntityLink {
            surface_form: "MySQL".to_string(),
            canonical_id: "mysql".to_string(),
            entity_type: EntityType::Database,
        }]);

        let sim_with_entities = compute_e11_similarity_with_entities(
            &e11_a,
            &e11_b,
            &entities_a,
            &entities_b,
            DEFAULT_EMBEDDING_WEIGHT,
            DEFAULT_ENTITY_WEIGHT,
        );

        let sim_embedding_only = cosine_similarity(&e11_a, &e11_b);

        // Entity mismatch should reduce combined similarity below pure embedding
        assert!(
            sim_with_entities < sim_embedding_only,
            "Entity mismatch should reduce similarity: {} vs {}",
            sim_with_entities,
            sim_embedding_only
        );
    }

    #[test]
    fn test_detect_technical_terms() {
        let text = "Implementing a REST API with GraphQL federation";
        let entities = detect_entities(text);

        let canonical_ids: HashSet<_> = entities.canonical_ids().into_iter().collect();
        assert!(canonical_ids.contains("rest_api"), "Should detect REST API");
        assert!(canonical_ids.contains("graphql"), "Should detect GraphQL");
    }

    #[test]
    fn test_detect_cloud_services() {
        let text = "Deploying to AWS Lambda with S3 storage and Kubernetes";
        let entities = detect_entities(text);

        let canonical_ids: HashSet<_> = entities.canonical_ids().into_iter().collect();
        assert!(canonical_ids.contains("aws"), "Should detect AWS");
        assert!(canonical_ids.contains("aws_lambda"), "Should detect Lambda");
        assert!(canonical_ids.contains("aws_s3"), "Should detect S3");
        assert!(
            canonical_ids.contains("kubernetes"),
            "Should detect Kubernetes"
        );
    }
}
