//! Memory node representing a stored memory unit in the knowledge graph.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{JohariQuadrant, Modality};

/// Unique identifier for memory nodes
pub type NodeId = Uuid;

/// Embedding vector type (1536 dimensions for OpenAI-compatible)
pub type EmbeddingVector = Vec<f32>;

/// Memory node representing a stored memory unit.
///
/// Each node contains content, its embedding vector, importance scores,
/// and metadata for graph operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryNode {
    /// Unique identifier
    pub id: NodeId,

    /// Raw content stored in this node
    pub content: String,

    /// Embedding vector (1536D)
    pub embedding: EmbeddingVector,

    /// Semantic importance score [0.0, 1.0]
    pub importance: f32,

    /// Access count for decay calculations
    pub access_count: u64,

    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Soft deletion marker
    pub deleted: bool,

    /// Johari quadrant classification
    pub johari_quadrant: JohariQuadrant,

    /// Source modality
    pub modality: Modality,

    /// Additional metadata
    pub metadata: NodeMetadata,
}

impl MemoryNode {
    /// Create a new memory node with the given content and embedding.
    pub fn new(content: String, embedding: EmbeddingVector) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            content,
            embedding,
            importance: 0.5,
            access_count: 0,
            last_accessed: now,
            created_at: now,
            deleted: false,
            johari_quadrant: JohariQuadrant::default(),
            modality: Modality::default(),
            metadata: NodeMetadata::default(),
        }
    }

    /// Mark this node as accessed, updating access count and timestamp.
    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
        self.last_accessed = Utc::now();
    }
}

/// Additional node metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct NodeMetadata {
    /// Source identifier (conversation, file, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Content language
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Custom tags
    #[serde(default)]
    pub tags: Vec<String>,

    /// UTL learning score at creation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utl_score: Option<f32>,

    /// Consolidation status
    #[serde(default)]
    pub consolidated: bool,

    /// Rationale for storing this memory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_node_creation() {
        let embedding = vec![0.1; 1536];
        let node = MemoryNode::new("test content".to_string(), embedding.clone());

        assert_eq!(node.content, "test content");
        assert_eq!(node.embedding.len(), 1536);
        assert_eq!(node.importance, 0.5);
        assert_eq!(node.access_count, 0);
        assert!(!node.deleted);
    }

    #[test]
    fn test_mark_accessed() {
        let embedding = vec![0.1; 1536];
        let mut node = MemoryNode::new("test".to_string(), embedding);
        let initial_accessed = node.last_accessed;

        std::thread::sleep(std::time::Duration::from_millis(10));
        node.mark_accessed();

        assert_eq!(node.access_count, 1);
        assert!(node.last_accessed > initial_accessed);
    }

    // =========================================================================
    // TC-GHOST-006: Serialization Safety Tests
    // =========================================================================

    #[test]
    fn test_memory_node_json_serialization_round_trip() {
        // TC-GHOST-006: MemoryNode must serialize and deserialize exactly through JSON
        let embedding = vec![0.5; 1536];
        let mut node = MemoryNode::new("Test content for serialization".to_string(), embedding);
        node.importance = 0.85;
        node.access_count = 42;
        node.metadata.source = Some("test-source".to_string());
        node.metadata.language = Some("en".to_string());
        node.metadata.tags = vec!["tag1".to_string(), "tag2".to_string(), "tag3".to_string()];
        node.metadata.utl_score = Some(0.75);
        node.metadata.consolidated = true;
        node.metadata.rationale = Some("Testing serialization round-trip".to_string());

        // Serialize to JSON
        let json_str = serde_json::to_string(&node).expect("MemoryNode must serialize to JSON");

        // Deserialize back
        let restored: MemoryNode = serde_json::from_str(&json_str)
            .expect("MemoryNode must deserialize from JSON");

        // Verify exact match using PartialEq
        assert_eq!(restored, node, "Deserialized node must match original exactly");
    }

    #[test]
    fn test_memory_node_complex_metadata_serialization() {
        // TC-GHOST-006: Complex metadata fields must survive serialization
        let embedding = vec![0.1, 0.2, 0.3]; // Small embedding for test
        let mut node = MemoryNode::new("Complex metadata test".to_string(), embedding);

        // Set all metadata fields
        node.metadata.source = Some("conversation:abc123".to_string());
        node.metadata.language = Some("en-US".to_string());
        node.metadata.tags = vec![
            "important".to_string(),
            "technical".to_string(),
            "machine-learning".to_string(),
            "neural-networks".to_string(),
        ];
        node.metadata.utl_score = Some(0.9876543);
        node.metadata.consolidated = true;
        node.metadata.rationale = Some("This is a complex test case with special chars: @#$%^&*()".to_string());

        // Round-trip through JSON
        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        // Verify all metadata fields
        assert_eq!(restored.metadata.source, Some("conversation:abc123".to_string()));
        assert_eq!(restored.metadata.language, Some("en-US".to_string()));
        assert_eq!(restored.metadata.tags.len(), 4);
        assert!(restored.metadata.tags.contains(&"machine-learning".to_string()));
        assert_eq!(restored.metadata.utl_score, Some(0.9876543));
        assert!(restored.metadata.consolidated);
        assert!(restored.metadata.rationale.as_ref().unwrap().contains("special chars"));
    }

    #[test]
    fn test_memory_node_embedding_precision_preserved() {
        // TC-GHOST-006: Embedding float precision must be preserved
        let mut embedding = Vec::with_capacity(1536);
        for i in 0..1536 {
            // Use values that might have precision issues
            let value = (i as f32 / 1536.0) * std::f32::consts::PI;
            embedding.push(value);
        }

        let node = MemoryNode::new("Precision test".to_string(), embedding.clone());

        // Round-trip through JSON
        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        // Verify embedding values are exactly preserved
        assert_eq!(restored.embedding.len(), 1536);
        for (i, (original, restored_val)) in embedding.iter().zip(restored.embedding.iter()).enumerate() {
            assert_eq!(
                original, restored_val,
                "Embedding value at index {} must be exactly preserved: {} vs {}",
                i, original, restored_val
            );
        }
    }

    #[test]
    fn test_memory_node_timestamps_preserved() {
        // TC-GHOST-006: Timestamps must be preserved through serialization
        let embedding = vec![0.1; 10];
        let node = MemoryNode::new("Timestamp test".to_string(), embedding);
        let original_created_at = node.created_at;
        let original_last_accessed = node.last_accessed;

        // Round-trip through JSON
        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        assert_eq!(restored.created_at, original_created_at, "created_at must be preserved");
        assert_eq!(restored.last_accessed, original_last_accessed, "last_accessed must be preserved");
    }

    #[test]
    fn test_memory_node_uuid_preserved() {
        // TC-GHOST-006: UUID must be preserved through serialization
        let embedding = vec![0.1; 10];
        let node = MemoryNode::new("UUID test".to_string(), embedding);
        let original_id = node.id;

        // Round-trip through JSON
        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        assert_eq!(restored.id, original_id, "UUID must be exactly preserved");
    }

    #[test]
    fn test_memory_node_optional_fields_none_serialization() {
        // TC-GHOST-006: Optional None fields must round-trip correctly
        let embedding = vec![0.1; 10];
        let node = MemoryNode::new("Optional fields test".to_string(), embedding);

        // Ensure optional fields are None
        assert!(node.metadata.source.is_none());
        assert!(node.metadata.language.is_none());
        assert!(node.metadata.utl_score.is_none());
        assert!(node.metadata.rationale.is_none());

        // Round-trip
        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        assert!(restored.metadata.source.is_none(), "None source must remain None");
        assert!(restored.metadata.language.is_none(), "None language must remain None");
        assert!(restored.metadata.utl_score.is_none(), "None utl_score must remain None");
        assert!(restored.metadata.rationale.is_none(), "None rationale must remain None");
    }

    #[test]
    fn test_node_metadata_serialization_isolated() {
        // TC-GHOST-006: NodeMetadata must serialize independently
        let mut metadata = NodeMetadata::default();
        metadata.source = Some("isolated-test".to_string());
        metadata.tags = vec!["a".to_string(), "b".to_string()];
        metadata.utl_score = Some(0.5);

        let json_str = serde_json::to_string(&metadata).unwrap();
        let restored: NodeMetadata = serde_json::from_str(&json_str).unwrap();

        assert_eq!(restored.source, Some("isolated-test".to_string()));
        assert_eq!(restored.tags, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(restored.utl_score, Some(0.5));
    }

    #[test]
    fn test_memory_node_binary_json_equivalence() {
        // TC-GHOST-006: Both pretty and compact JSON must deserialize identically
        let embedding = vec![0.5; 100];
        let mut node = MemoryNode::new("Binary equivalence test".to_string(), embedding);
        node.metadata.tags = vec!["test".to_string()];

        // Compact JSON
        let compact_json = serde_json::to_string(&node).unwrap();
        let from_compact: MemoryNode = serde_json::from_str(&compact_json).unwrap();

        // Pretty JSON
        let pretty_json = serde_json::to_string_pretty(&node).unwrap();
        let from_pretty: MemoryNode = serde_json::from_str(&pretty_json).unwrap();

        // Both must produce identical results
        assert_eq!(from_compact, from_pretty, "Compact and pretty JSON must deserialize identically");
        assert_eq!(from_compact, node, "Both must match original");
    }

    #[test]
    fn test_memory_node_special_content_serialization() {
        // TC-GHOST-006: Special characters in content must be preserved
        let special_content = r#"Content with "quotes", 'apostrophes', \backslashes\, and
newlines, plus unicode: 日本語 🎉 émojis"#;

        let embedding = vec![0.1; 10];
        let node = MemoryNode::new(special_content.to_string(), embedding);

        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        assert_eq!(restored.content, special_content, "Special characters must be preserved");
    }

    #[test]
    fn test_memory_node_extreme_values() {
        // TC-GHOST-006: Extreme float values must be handled
        let mut embedding = vec![0.0; 10];
        embedding[0] = f32::MIN_POSITIVE;
        embedding[1] = f32::MAX;
        embedding[2] = f32::MIN;
        embedding[3] = 1e-38;
        embedding[4] = 1e38;

        let mut node = MemoryNode::new("Extreme values test".to_string(), embedding.clone());
        node.importance = 0.0;
        node.access_count = u64::MAX;

        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        assert_eq!(restored.importance, 0.0);
        assert_eq!(restored.access_count, u64::MAX);
        assert_eq!(restored.embedding[0], f32::MIN_POSITIVE);
    }
}
