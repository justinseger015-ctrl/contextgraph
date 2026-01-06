//! Entity and relation encoding utilities.
//!
//! Provides methods to format entity names and relations for embedding.

use super::EntityModel;

impl EntityModel {
    /// Encode an entity with optional type context.
    ///
    /// Creates a text representation suitable for embedding with the MiniLM model.
    /// The entity type is uppercased and wrapped in brackets.
    ///
    /// # Arguments
    /// * `name` - The entity name (e.g., "Alice", "Anthropic")
    /// * `entity_type` - Optional entity type (e.g., "PERSON", "ORG")
    ///
    /// # Returns
    /// A string formatted as "[TYPE] name" if type provided, otherwise just "name".
    ///
    /// # Examples
    /// ```rust
    /// use context_graph_embeddings::models::EntityModel;
    ///
    /// let text = EntityModel::encode_entity("Alice", Some("person"));
    /// assert_eq!(text, "[PERSON] Alice");
    ///
    /// let text = EntityModel::encode_entity("Paris", None);
    /// assert_eq!(text, "Paris");
    /// ```
    pub fn encode_entity(name: &str, entity_type: Option<&str>) -> String {
        match entity_type {
            Some(etype) => format!("[{}] {}", etype.to_uppercase(), name),
            None => name.to_string(),
        }
    }

    /// Encode a relation for TransE-style operations.
    ///
    /// Converts relation predicates into natural language form by replacing
    /// underscores with spaces.
    ///
    /// # Arguments
    /// * `relation` - The relation predicate (e.g., "works_at", "is_friend_of")
    ///
    /// # Returns
    /// A string with underscores replaced by spaces.
    ///
    /// # Examples
    /// ```rust
    /// use context_graph_embeddings::models::EntityModel;
    ///
    /// let text = EntityModel::encode_relation("works_at");
    /// assert_eq!(text, "works at");
    ///
    /// let text = EntityModel::encode_relation("is_friend_of");
    /// assert_eq!(text, "is friend of");
    /// ```
    pub fn encode_relation(relation: &str) -> String {
        relation.replace('_', " ")
    }
}
