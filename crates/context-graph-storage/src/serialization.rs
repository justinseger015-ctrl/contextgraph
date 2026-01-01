//! Binary serialization utilities for storage operations.
//!
//! This module provides efficient binary serialization for `MemoryNode`,
//! `GraphEdge`, `EmbeddingVector`, and `UUID` types. It is optimized for
//! both space efficiency and speed.
//!
//! # Serialization Strategy
//!
//! | Type | Format | Rationale |
//! |------|--------|-----------|
//! | `MemoryNode` | MessagePack | Handles `skip_serializing_if` correctly |
//! | `GraphEdge` | bincode | Optimal for fixed-layout structs |
//! | `EmbeddingVector` | Raw LE f32 | Maximum performance, no overhead |
//! | `UUID` | Raw bytes | Exactly 16 bytes, no overhead |
//!
//! # Performance Characteristics
//!
//! | Operation | Size | Latency |
//! |-----------|------|---------|
//! | MemoryNode (1536D) | ~6.5KB | < 100us |
//! | GraphEdge | ~200 bytes | < 50us |
//! | Embedding (1536D) | 6144 bytes | < 10us |
//! | UUID | 16 bytes | < 1us |
//!
//! # Constitution Compliance
//!
//! - AP-009: All functions preserve exact f32 values (no NaN/Infinity manipulation)
//! - Naming: `snake_case` functions, `PascalCase` types
//!
//! # Example: Round-trip Serialization
//!
//! ```rust
//! use context_graph_storage::serialization::{
//!     serialize_embedding, deserialize_embedding,
//!     serialize_uuid, deserialize_uuid,
//! };
//! use uuid::Uuid;
//!
//! // Embedding round-trip
//! let embedding = vec![0.5_f32; 100];
//! let bytes = serialize_embedding(&embedding);
//! let restored = deserialize_embedding(&bytes).unwrap();
//! assert_eq!(embedding, restored);
//!
//! // UUID round-trip
//! let id = Uuid::new_v4();
//! let bytes = serialize_uuid(&id);
//! let restored = deserialize_uuid(&bytes);
//! assert_eq!(id, restored);
//! ```
//!
//! # Why MessagePack for MemoryNode?
//!
//! `MemoryNode` contains `NodeMetadata` which uses serde attributes like
//! `#[serde(skip_serializing_if = "Option::is_none")]`. Bincode requires
//! fixed-layout serialization and doesn't support these attributes.
//! MessagePack properly handles serde attributes while still providing
//! compact binary output (smaller than JSON, similar to bincode).

use thiserror::Error;
use uuid::Uuid;

use context_graph_core::types::{EmbeddingVector, GraphEdge, MemoryNode};

/// Errors that can occur during serialization/deserialization operations.
///
/// These errors indicate data format issues or corruption. They are converted
/// to [`StorageError::Serialization`](crate::StorageError::Serialization)
/// when propagated from storage operations.
///
/// # Design Notes
///
/// - `bincode::Error` does not implement `Clone`, so we store error messages as `String`
/// - All variants include enough context for debugging
/// - Implements `Clone` and `PartialEq` for testing
///
/// # Example: Error Construction
///
/// ```rust
/// use context_graph_storage::SerializationError;
///
/// // Invalid embedding size
/// let error = SerializationError::InvalidEmbeddingSize {
///     expected: 6144,
///     actual: 100,
/// };
/// assert!(error.to_string().contains("expected 6144"));
///
/// // Serialization failure
/// let error = SerializationError::SerializeFailed("corrupt data".to_string());
/// assert!(error.to_string().contains("Serialization failed"));
/// ```
///
/// # Example: Error Matching
///
/// ```rust
/// use context_graph_storage::serialization::{SerializationError, deserialize_embedding};
///
/// let bad_bytes = vec![0u8; 13]; // Not divisible by 4
/// match deserialize_embedding(&bad_bytes) {
///     Ok(_) => unreachable!(),
///     Err(SerializationError::InvalidEmbeddingSize { expected, actual }) => {
///         println!("Expected {} bytes, got {}", expected, actual);
///     }
///     Err(_) => unreachable!(),
/// }
/// ```
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SerializationError {
    /// Serialization operation failed.
    ///
    /// Contains the underlying error message from bincode or MessagePack.
    /// Common causes:
    /// - Data structure incompatible with serialization format
    /// - Out of memory during serialization
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::SerializationError;
    ///
    /// let error = SerializationError::SerializeFailed("unexpected type".to_string());
    /// assert!(error.to_string().contains("Serialization failed"));
    /// ```
    #[error("Serialization failed: {0}")]
    SerializeFailed(String),

    /// Deserialization operation failed.
    ///
    /// Contains the underlying error message from bincode or MessagePack.
    /// Common causes:
    /// - Corrupted data in database
    /// - Schema version mismatch
    /// - Truncated data (incomplete write)
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::SerializationError;
    ///
    /// let error = SerializationError::DeserializeFailed("invalid format".to_string());
    /// assert!(error.to_string().contains("Deserialization failed"));
    /// ```
    #[error("Deserialization failed: {0}")]
    DeserializeFailed(String),

    /// Embedding byte array has invalid size.
    ///
    /// Embedding bytes must be divisible by 4 (size of f32).
    /// This error occurs when deserializing raw embedding bytes.
    ///
    /// # Fields
    ///
    /// - `expected`: The next valid byte count (rounded up to multiple of 4)
    /// - `actual`: The actual byte count received
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::SerializationError;
    ///
    /// let error = SerializationError::InvalidEmbeddingSize {
    ///     expected: 16, // Next multiple of 4
    ///     actual: 13,
    /// };
    /// assert!(error.to_string().contains("expected 16"));
    /// ```
    #[error("Invalid embedding size: expected {expected} bytes, got {actual}")]
    InvalidEmbeddingSize {
        /// Expected byte count (must be divisible by 4)
        expected: usize,
        /// Actual byte count received
        actual: usize,
    },

    /// UUID byte array has invalid size.
    ///
    /// UUID requires exactly 16 bytes. This error is included for
    /// completeness but is unlikely in practice since UUIDs use
    /// fixed-size arrays.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::SerializationError;
    ///
    /// let error = SerializationError::InvalidUuidSize { actual: 10 };
    /// assert!(error.to_string().contains("expected 16 bytes"));
    /// ```
    #[error("Invalid UUID bytes: expected 16 bytes, got {actual}")]
    InvalidUuidSize {
        /// Actual byte count received
        actual: usize,
    },
}

/// Serializes a `MemoryNode` to MessagePack bytes.
///
/// Uses MessagePack (rmp-serde) format which properly handles serde's
/// `skip_serializing_if` attributes on `NodeMetadata`.
///
/// # Arguments
///
/// * `node` - The `MemoryNode` to serialize
///
/// # Returns
///
/// * `Ok(Vec<u8>)` - Serialized bytes (~6.5KB for node with 1536D embedding)
/// * `Err(SerializationError::SerializeFailed)` - If serialization fails
///
/// # Errors
///
/// * `SerializationError::SerializeFailed` - MessagePack serialization error
///
/// # Performance
///
/// `Constraint: latency < 100us for typical node`
///
/// # Example
///
/// ```rust
/// use context_graph_storage::serialization::{serialize_node, deserialize_node};
/// use context_graph_storage::MemoryNode;
///
/// // Create a valid normalized embedding
/// let dim = 1536;
/// let val = 1.0_f32 / (dim as f32).sqrt();
/// let embedding = vec![val; dim];
///
/// // Create and serialize node
/// let node = MemoryNode::new("Test content".to_string(), embedding);
/// let bytes = serialize_node(&node).expect("serialize failed");
///
/// // Serialized size should be ~6.5KB
/// assert!(bytes.len() > 6000);
/// assert!(bytes.len() < 10000);
///
/// // Round-trip verification
/// let restored = deserialize_node(&bytes).unwrap();
/// assert_eq!(node.id, restored.id);
/// assert_eq!(node.content, restored.content);
/// ```
pub fn serialize_node(node: &MemoryNode) -> Result<Vec<u8>, SerializationError> {
    // Use named format for proper field handling with skip_serializing_if attrs
    rmp_serde::to_vec_named(node).map_err(|e| SerializationError::SerializeFailed(e.to_string()))
}

/// Deserializes MessagePack bytes to a `MemoryNode`.
///
/// # Arguments
///
/// * `bytes` - The MessagePack bytes to deserialize
///
/// # Returns
///
/// * `Ok(MemoryNode)` - The deserialized node
/// * `Err(SerializationError::DeserializeFailed)` - If deserialization fails
///
/// # Errors
///
/// * `SerializationError::DeserializeFailed` - Corrupt data, truncated bytes,
///   or schema mismatch
///
/// # Example
///
/// ```rust
/// use context_graph_storage::serialization::{serialize_node, deserialize_node};
/// use context_graph_storage::MemoryNode;
///
/// // Create a node with normalized embedding
/// let dim = 1536;
/// let val = 1.0_f32 / (dim as f32).sqrt();
/// let node = MemoryNode::new("Test".to_string(), vec![val; dim]);
///
/// // Round-trip
/// let bytes = serialize_node(&node).unwrap();
/// let restored = deserialize_node(&bytes).unwrap();
/// assert_eq!(node, restored);
/// ```
pub fn deserialize_node(bytes: &[u8]) -> Result<MemoryNode, SerializationError> {
    rmp_serde::from_slice(bytes).map_err(|e| SerializationError::DeserializeFailed(e.to_string()))
}

/// Serializes a `GraphEdge` to bincode bytes.
///
/// Uses bincode format which is optimal for fixed-layout structs like `GraphEdge`.
///
/// # Arguments
///
/// * `edge` - The `GraphEdge` to serialize
///
/// # Returns
///
/// * `Ok(Vec<u8>)` - Serialized bytes (~200 bytes)
/// * `Err(SerializationError::SerializeFailed)` - If serialization fails
///
/// # Errors
///
/// * `SerializationError::SerializeFailed` - Bincode serialization error
///
/// # Performance
///
/// `Constraint: latency < 50us for typical edge`
///
/// # Example
///
/// ```rust
/// use context_graph_storage::serialization::{serialize_edge, deserialize_edge};
/// use context_graph_storage::{GraphEdge, EdgeType, Domain};
/// use uuid::Uuid;
///
/// // Create edge
/// let edge = GraphEdge::new(
///     Uuid::new_v4(),
///     Uuid::new_v4(),
///     EdgeType::Semantic,
///     Domain::Code,
/// );
///
/// // Serialize
/// let bytes = serialize_edge(&edge).expect("serialize failed");
/// assert!(bytes.len() > 100 && bytes.len() < 500);
///
/// // Round-trip
/// let restored = deserialize_edge(&bytes).unwrap();
/// assert_eq!(edge, restored);
/// ```
pub fn serialize_edge(edge: &GraphEdge) -> Result<Vec<u8>, SerializationError> {
    bincode::serialize(edge).map_err(|e| SerializationError::SerializeFailed(e.to_string()))
}

/// Deserializes bincode bytes to a `GraphEdge`.
///
/// # Arguments
///
/// * `bytes` - The bincode bytes to deserialize
///
/// # Returns
///
/// * `Ok(GraphEdge)` - The deserialized edge
/// * `Err(SerializationError::DeserializeFailed)` - If deserialization fails
///
/// # Errors
///
/// * `SerializationError::DeserializeFailed` - Corrupt data, truncated bytes,
///   or schema mismatch
///
/// # Example
///
/// ```rust
/// use context_graph_storage::serialization::{serialize_edge, deserialize_edge};
/// use context_graph_storage::{GraphEdge, EdgeType, Domain};
/// use uuid::Uuid;
///
/// let edge = GraphEdge::new(
///     Uuid::new_v4(),
///     Uuid::new_v4(),
///     EdgeType::Causal,
///     Domain::Medical,
/// );
///
/// let bytes = serialize_edge(&edge).unwrap();
/// let restored = deserialize_edge(&bytes).unwrap();
/// assert_eq!(edge.edge_type, restored.edge_type);
/// assert_eq!(edge.domain, restored.domain);
/// ```
pub fn deserialize_edge(bytes: &[u8]) -> Result<GraphEdge, SerializationError> {
    bincode::deserialize(bytes).map_err(|e| SerializationError::DeserializeFailed(e.to_string()))
}

/// Serialize embedding to raw f32 bytes (little-endian).
///
/// This uses raw bytes rather than bincode for maximum efficiency.
/// Each f32 becomes exactly 4 bytes in little-endian format.
///
/// # Arguments
/// * `embedding` - The embedding vector to serialize
///
/// # Returns
/// * `Vec<u8>` - Raw bytes (dim * 4 bytes, e.g., 6144 bytes for 1536D)
///
/// # Infallible
/// This function cannot fail - no Result wrapper needed.
///
/// # Example
/// ```rust
/// use context_graph_storage::serialization::serialize_embedding;
///
/// let embedding = vec![0.5_f32; 1536];
/// let bytes = serialize_embedding(&embedding);
/// assert_eq!(bytes.len(), 1536 * 4); // 6144 bytes
/// ```
pub fn serialize_embedding(embedding: &EmbeddingVector) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Deserialize raw f32 bytes to embedding vector.
///
/// Converts little-endian bytes back to f32 values.
///
/// # Arguments
/// * `bytes` - Raw bytes (must be divisible by 4)
///
/// # Returns
/// * `Ok(EmbeddingVector)` - The deserialized embedding
/// * `Err(SerializationError::InvalidEmbeddingSize)` - If bytes.len() % 4 != 0
///
/// # Example
/// ```rust
/// use context_graph_storage::serialization::{serialize_embedding, deserialize_embedding};
///
/// let embedding = vec![0.25_f32; 100];
/// let bytes = serialize_embedding(&embedding);
/// let restored = deserialize_embedding(&bytes).unwrap();
/// assert_eq!(embedding, restored);
/// ```
pub fn deserialize_embedding(bytes: &[u8]) -> Result<EmbeddingVector, SerializationError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(SerializationError::InvalidEmbeddingSize {
            expected: ((bytes.len() / 4) + 1) * 4,
            actual: bytes.len(),
        });
    }

    let mut embedding = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        // SAFETY: chunks_exact(4) guarantees exactly 4 bytes
        let arr: [u8; 4] = chunk.try_into().expect("chunk is exactly 4 bytes");
        embedding.push(f32::from_le_bytes(arr));
    }
    Ok(embedding)
}

/// Serialize UUID to exactly 16 bytes.
///
/// # Arguments
/// * `id` - The UUID to serialize
///
/// # Returns
/// * `[u8; 16]` - The UUID bytes
///
/// # Infallible
/// This function cannot fail.
///
/// # Example
/// ```rust
/// use context_graph_storage::serialization::serialize_uuid;
/// use uuid::Uuid;
///
/// let id = Uuid::new_v4();
/// let bytes = serialize_uuid(&id);
/// assert_eq!(bytes.len(), 16);
/// ```
#[inline]
pub fn serialize_uuid(id: &Uuid) -> [u8; 16] {
    *id.as_bytes()
}

/// Deserialize 16 bytes to UUID.
///
/// # Arguments
/// * `bytes` - Exactly 16 bytes
///
/// # Returns
/// * `Uuid` - The deserialized UUID
///
/// # Infallible
/// This function cannot fail when given a valid [u8; 16] array.
///
/// # Example
/// ```rust
/// use context_graph_storage::serialization::{serialize_uuid, deserialize_uuid};
/// use uuid::Uuid;
///
/// let id = Uuid::new_v4();
/// let bytes = serialize_uuid(&id);
/// let restored = deserialize_uuid(&bytes);
/// assert_eq!(id, restored);
/// ```
#[inline]
pub fn deserialize_uuid(bytes: &[u8; 16]) -> Uuid {
    Uuid::from_bytes(*bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use context_graph_core::marblestone::{Domain, EdgeType, NeurotransmitterWeights};
    use context_graph_core::types::{
        JohariQuadrant, NodeMetadata, DEFAULT_EMBEDDING_DIM, MAX_CONTENT_SIZE,
    };
    use serde_json::json;
    use uuid::Uuid;

    // =========================================================================
    // Helper Functions - Create REAL data (no mocks)
    // =========================================================================

    /// Create a valid normalized embedding vector.
    /// Normalization ensures magnitude ~= 1.0 (validates per MemoryNode::validate).
    fn create_normalized_embedding(dim: usize) -> EmbeddingVector {
        let val = 1.0 / (dim as f32).sqrt();
        vec![val; dim]
    }

    /// Create a valid MemoryNode with real data.
    fn create_test_node() -> MemoryNode {
        let embedding = create_normalized_embedding(DEFAULT_EMBEDDING_DIM);
        let mut node = MemoryNode::new("Test content for serialization".to_string(), embedding);
        node.importance = 0.75;
        node.emotional_valence = 0.5;
        node.quadrant = JohariQuadrant::Open;
        node.metadata = NodeMetadata::new()
            .with_source("test-source")
            .with_language("en");
        node
    }

    /// Create a valid GraphEdge with real data.
    fn create_test_edge() -> GraphEdge {
        GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::Code,
        )
    }

    // =========================================================================
    // SerializationError Tests
    // =========================================================================

    #[test]
    fn test_serialization_error_serialize_failed() {
        let error = SerializationError::SerializeFailed("test error".to_string());
        let msg = error.to_string();
        assert!(msg.contains("Serialization failed"));
        assert!(msg.contains("test error"));
    }

    #[test]
    fn test_serialization_error_deserialize_failed() {
        let error = SerializationError::DeserializeFailed("corrupt data".to_string());
        let msg = error.to_string();
        assert!(msg.contains("Deserialization failed"));
        assert!(msg.contains("corrupt data"));
    }

    #[test]
    fn test_serialization_error_invalid_embedding_size() {
        let error = SerializationError::InvalidEmbeddingSize {
            expected: 6144,
            actual: 100,
        };
        let msg = error.to_string();
        assert!(msg.contains("expected 6144"));
        assert!(msg.contains("got 100"));
    }

    #[test]
    fn test_serialization_error_invalid_uuid_size() {
        let error = SerializationError::InvalidUuidSize { actual: 10 };
        let msg = error.to_string();
        assert!(msg.contains("expected 16"));
        assert!(msg.contains("got 10"));
    }

    #[test]
    fn test_serialization_error_clone() {
        let original = SerializationError::SerializeFailed("test".to_string());
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_serialization_error_partial_eq() {
        let a = SerializationError::InvalidEmbeddingSize {
            expected: 100,
            actual: 50,
        };
        let b = SerializationError::InvalidEmbeddingSize {
            expected: 100,
            actual: 50,
        };
        let c = SerializationError::InvalidEmbeddingSize {
            expected: 100,
            actual: 60,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_serialization_error_debug() {
        let error = SerializationError::InvalidUuidSize { actual: 8 };
        let debug = format!("{:?}", error);
        assert!(debug.contains("InvalidUuidSize"));
        assert!(debug.contains("8"));
    }

    // =========================================================================
    // Node Serialization Tests
    // =========================================================================

    #[test]
    fn test_node_roundtrip() {
        let node = create_test_node();
        let bytes = serialize_node(&node).expect("serialize failed");
        let restored = deserialize_node(&bytes).expect("deserialize failed");
        assert_eq!(node, restored, "Round-trip must preserve all fields");
    }

    #[test]
    fn test_node_size_reasonable() {
        let node = create_test_node();
        let bytes = serialize_node(&node).unwrap();
        // 1536 * 4 = 6144 bytes for embedding alone
        // Total should be ~6.5-8KB with other fields
        // MessagePack with named format includes field names adding ~500 bytes overhead
        assert!(
            bytes.len() > 6000,
            "Node should be at least 6KB, got {}",
            bytes.len()
        );
        assert!(
            bytes.len() < 10000,
            "Node should be less than 10KB, got {}",
            bytes.len()
        );
    }

    #[test]
    fn test_node_preserves_all_fields() {
        let mut node = create_test_node();
        node.importance = 0.999;
        node.emotional_valence = -0.555;
        node.access_count = 12345;
        node.quadrant = JohariQuadrant::Hidden;
        node.metadata.add_tag("important");
        node.metadata.add_tag("verified");
        node.metadata.set_custom("priority", json!(5));

        let bytes = serialize_node(&node).unwrap();
        let restored = deserialize_node(&bytes).unwrap();

        assert_eq!(node.id, restored.id);
        assert_eq!(node.content, restored.content);
        assert_eq!(node.embedding, restored.embedding);
        assert_eq!(node.quadrant, restored.quadrant);
        assert_eq!(node.importance, restored.importance);
        assert_eq!(node.emotional_valence, restored.emotional_valence);
        assert_eq!(node.created_at, restored.created_at);
        assert_eq!(node.accessed_at, restored.accessed_at);
        assert_eq!(node.access_count, restored.access_count);
        assert_eq!(node.metadata, restored.metadata);
    }

    #[test]
    fn test_node_with_all_metadata() {
        let mut node = create_test_node();
        node.metadata.add_tag("important");
        node.metadata.add_tag("verified");
        node.metadata.set_custom("priority", json!(5));
        node.metadata.mark_consolidated();
        node.metadata.rationale = Some("Testing serialization".to_string());

        let bytes = serialize_node(&node).unwrap();
        let restored = deserialize_node(&bytes).unwrap();

        assert_eq!(node.metadata.tags, restored.metadata.tags);
        assert_eq!(
            node.metadata.get_custom("priority"),
            restored.metadata.get_custom("priority")
        );
        assert!(restored.metadata.consolidated);
        assert_eq!(node.metadata.rationale, restored.metadata.rationale);
    }

    // =========================================================================
    // Edge Serialization Tests
    // =========================================================================

    #[test]
    fn test_edge_roundtrip() {
        let edge = create_test_edge();
        let bytes = serialize_edge(&edge).expect("serialize failed");
        let restored = deserialize_edge(&bytes).expect("deserialize failed");
        assert_eq!(edge, restored, "Round-trip must preserve all fields");
    }

    #[test]
    fn test_edge_size_reasonable() {
        let edge = create_test_edge();
        let bytes = serialize_edge(&edge).unwrap();
        assert!(
            bytes.len() > 100,
            "Edge should be at least 100 bytes, got {}",
            bytes.len()
        );
        assert!(
            bytes.len() < 500,
            "Edge should be less than 500 bytes, got {}",
            bytes.len()
        );
    }

    #[test]
    fn test_edge_preserves_all_13_fields() {
        let mut edge = create_test_edge();
        edge.weight = 0.85;
        edge.confidence = 0.95;
        edge.is_amortized_shortcut = true;
        edge.steering_reward = 0.75;
        edge.traversal_count = 42;
        edge.neurotransmitter_weights = NeurotransmitterWeights::for_domain(Domain::Medical);
        edge.record_traversal();

        let bytes = serialize_edge(&edge).unwrap();
        let restored = deserialize_edge(&bytes).unwrap();

        assert_eq!(edge.id, restored.id);
        assert_eq!(edge.source_id, restored.source_id);
        assert_eq!(edge.target_id, restored.target_id);
        assert_eq!(edge.edge_type, restored.edge_type);
        assert_eq!(edge.weight, restored.weight);
        assert_eq!(edge.confidence, restored.confidence);
        assert_eq!(edge.domain, restored.domain);
        assert_eq!(
            edge.neurotransmitter_weights,
            restored.neurotransmitter_weights
        );
        assert_eq!(edge.is_amortized_shortcut, restored.is_amortized_shortcut);
        assert_eq!(edge.steering_reward, restored.steering_reward);
        assert_eq!(edge.traversal_count, restored.traversal_count);
        assert_eq!(edge.created_at, restored.created_at);
        assert!(restored.last_traversed_at.is_some());
    }

    #[test]
    fn test_edge_with_all_marblestone_fields() {
        let mut edge = create_test_edge();
        edge.is_amortized_shortcut = true;
        edge.steering_reward = 0.75;
        edge.traversal_count = 42;
        edge.confidence = 0.9;
        edge.neurotransmitter_weights = NeurotransmitterWeights::for_domain(Domain::Medical);
        edge.record_traversal();

        let bytes = serialize_edge(&edge).unwrap();
        let restored = deserialize_edge(&bytes).unwrap();

        assert_eq!(edge.is_amortized_shortcut, restored.is_amortized_shortcut);
        assert_eq!(edge.steering_reward, restored.steering_reward);
        assert_eq!(edge.traversal_count, restored.traversal_count);
        assert_eq!(
            edge.neurotransmitter_weights,
            restored.neurotransmitter_weights
        );
        assert!(restored.last_traversed_at.is_some());
    }

    #[test]
    fn test_edge_all_types() {
        for edge_type in EdgeType::all() {
            for domain in Domain::all() {
                let edge = GraphEdge::new(Uuid::new_v4(), Uuid::new_v4(), edge_type, domain);
                let bytes = serialize_edge(&edge).unwrap();
                let restored = deserialize_edge(&bytes).unwrap();
                assert_eq!(edge, restored, "Failed for {:?} / {:?}", edge_type, domain);
            }
        }
    }

    // =========================================================================
    // Embedding Serialization Tests
    // =========================================================================

    #[test]
    fn test_embedding_roundtrip() {
        let embedding = create_normalized_embedding(DEFAULT_EMBEDDING_DIM);
        let bytes = serialize_embedding(&embedding);
        let restored = deserialize_embedding(&bytes).expect("deserialize failed");

        assert_eq!(embedding.len(), restored.len());
        for (orig, rest) in embedding.iter().zip(restored.iter()) {
            assert_eq!(*orig, *rest, "f32 values must be exactly preserved");
        }
    }

    #[test]
    fn test_embedding_size_exact() {
        let embedding = vec![0.5_f32; 1536];
        let bytes = serialize_embedding(&embedding);
        assert_eq!(bytes.len(), 1536 * 4, "Embedding should be dim * 4 bytes");
    }

    #[test]
    fn test_embedding_invalid_size() {
        let bytes = vec![0u8; 13]; // Not divisible by 4
        let result = deserialize_embedding(&bytes);
        assert!(matches!(
            result,
            Err(SerializationError::InvalidEmbeddingSize { .. })
        ));
    }

    #[test]
    fn test_embedding_empty() {
        let embedding: EmbeddingVector = vec![];
        let bytes = serialize_embedding(&embedding);
        assert!(bytes.is_empty());
        let restored = deserialize_embedding(&bytes).unwrap();
        assert!(restored.is_empty());
    }

    #[test]
    fn test_embedding_various_dimensions() {
        for dim in [1, 10, 128, 512, 768, 1024, 1536] {
            let embedding = create_normalized_embedding(dim);
            let bytes = serialize_embedding(&embedding);
            assert_eq!(bytes.len(), dim * 4);
            let restored = deserialize_embedding(&bytes).unwrap();
            assert_eq!(restored.len(), dim);
            assert_eq!(embedding, restored);
        }
    }

    // =========================================================================
    // UUID Serialization Tests
    // =========================================================================

    #[test]
    fn test_uuid_roundtrip() {
        let id = Uuid::new_v4();
        let bytes = serialize_uuid(&id);
        assert_eq!(bytes.len(), 16);
        let restored = deserialize_uuid(&bytes);
        assert_eq!(id, restored);
    }

    #[test]
    fn test_uuid_nil() {
        let nil = Uuid::nil();
        let bytes = serialize_uuid(&nil);
        let restored = deserialize_uuid(&bytes);
        assert_eq!(nil, restored);
        assert!(restored.is_nil());
    }

    #[test]
    fn test_uuid_max() {
        let max = Uuid::max();
        let bytes = serialize_uuid(&max);
        let restored = deserialize_uuid(&bytes);
        assert_eq!(max, restored);
    }

    #[test]
    fn test_uuid_multiple_roundtrips() {
        for _ in 0..100 {
            let id = Uuid::new_v4();
            let bytes = serialize_uuid(&id);
            let restored = deserialize_uuid(&bytes);
            assert_eq!(id, restored);
        }
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_deserialization_invalid_bytes() {
        let garbage = vec![0xFF, 0x00, 0xAB, 0xCD];
        let node_result = deserialize_node(&garbage);
        assert!(node_result.is_err());

        let edge_result = deserialize_edge(&garbage);
        assert!(edge_result.is_err());
    }

    #[test]
    fn test_deserialization_empty_bytes() {
        let empty: Vec<u8> = vec![];
        let node_result = deserialize_node(&empty);
        assert!(node_result.is_err());

        let edge_result = deserialize_edge(&empty);
        assert!(edge_result.is_err());
    }

    #[test]
    fn test_deserialization_truncated_node() {
        let node = create_test_node();
        let bytes = serialize_node(&node).unwrap();
        let truncated = &bytes[..bytes.len() / 2];
        let result = deserialize_node(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialization_truncated_edge() {
        let edge = create_test_edge();
        let bytes = serialize_edge(&edge).unwrap();
        let truncated = &bytes[..bytes.len() / 2];
        let result = deserialize_edge(truncated);
        assert!(result.is_err());
    }

    // =========================================================================
    // Edge Case Tests (REQUIRED - with before/after state printing)
    // =========================================================================

    #[test]
    fn edge_case_empty_content() {
        let mut node = create_test_node();
        node.content = String::new();

        println!("=== EDGE CASE 1: Empty Content ===");
        println!("BEFORE: content.len() = {}", node.content.len());

        let bytes = serialize_node(&node).unwrap();
        println!("SERIALIZED: bytes.len() = {}", bytes.len());

        let restored = deserialize_node(&bytes).unwrap();
        println!("AFTER: content.len() = {}", restored.content.len());

        assert_eq!(node.content, restored.content);
        println!("RESULT: PASS - Empty content preserved");
    }

    #[test]
    fn edge_case_max_content() {
        let mut node = create_test_node();
        node.content = "x".repeat(MAX_CONTENT_SIZE);

        println!("=== EDGE CASE 2: Maximum Content Size ===");
        println!(
            "BEFORE: content.len() = {} (MAX_CONTENT_SIZE = {})",
            node.content.len(),
            MAX_CONTENT_SIZE
        );

        let bytes = serialize_node(&node).unwrap();
        println!(
            "SERIALIZED: bytes.len() = {} (~{:.2}MB)",
            bytes.len(),
            bytes.len() as f64 / 1_048_576.0
        );

        let restored = deserialize_node(&bytes).unwrap();
        println!("AFTER: content.len() = {}", restored.content.len());

        assert_eq!(node.content.len(), restored.content.len());
        println!("RESULT: PASS - Max content size preserved");
    }

    #[test]
    fn edge_case_extreme_floats() {
        let extremes = vec![
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            1e-38_f32,
            1e38_f32,
            0.0_f32,
            -0.0_f32,
        ];

        println!("=== EDGE CASE 3: Extreme Float Values ===");
        println!("BEFORE: {:?}", extremes);

        let bytes = serialize_embedding(&extremes);
        println!("SERIALIZED: {} bytes", bytes.len());

        let restored = deserialize_embedding(&bytes).unwrap();
        println!("AFTER: {:?}", restored);

        for (i, (orig, rest)) in extremes.iter().zip(restored.iter()).enumerate() {
            assert_eq!(orig.to_bits(), rest.to_bits(), "Value {} differs", i);
        }
        println!("RESULT: PASS - All extreme float values preserved exactly");
    }

    #[test]
    fn edge_case_special_unicode_content() {
        let mut node = create_test_node();
        node.content = "Unicode: 日本語 🎉 émojis λ α β γ δ ε ζ".to_string();

        println!("=== EDGE CASE: Special Unicode Content ===");
        println!("BEFORE: content = {:?}", node.content);
        println!("BEFORE: content.len() = {} bytes", node.content.len());

        let bytes = serialize_node(&node).unwrap();
        let restored = deserialize_node(&bytes).unwrap();

        println!("AFTER: content = {:?}", restored.content);
        println!("AFTER: content.len() = {} bytes", restored.content.len());

        assert_eq!(node.content, restored.content);
        println!("RESULT: PASS - Unicode content preserved");
    }

    #[test]
    fn edge_case_boundary_values() {
        let mut edge = create_test_edge();
        edge.weight = 0.0;
        edge.confidence = 1.0;
        edge.steering_reward = -1.0;
        edge.traversal_count = u64::MAX;

        println!("=== EDGE CASE: Boundary Values ===");
        println!(
            "BEFORE: weight={}, confidence={}, steering_reward={}, traversal_count={}",
            edge.weight, edge.confidence, edge.steering_reward, edge.traversal_count
        );

        let bytes = serialize_edge(&edge).unwrap();
        let restored = deserialize_edge(&bytes).unwrap();

        println!(
            "AFTER: weight={}, confidence={}, steering_reward={}, traversal_count={}",
            restored.weight,
            restored.confidence,
            restored.steering_reward,
            restored.traversal_count
        );

        assert_eq!(edge.weight, restored.weight);
        assert_eq!(edge.confidence, restored.confidence);
        assert_eq!(edge.steering_reward, restored.steering_reward);
        assert_eq!(edge.traversal_count, restored.traversal_count);
        println!("RESULT: PASS - All boundary values preserved");
    }

    // =========================================================================
    // Comprehensive Field Preservation Tests
    // =========================================================================

    #[test]
    fn test_node_embedding_precision_preserved() {
        // Use specific float values that might have precision issues
        let mut embedding = Vec::with_capacity(1536);
        for i in 0..1536 {
            let value = (i as f32 / 1536.0) * std::f32::consts::PI;
            embedding.push(value);
        }

        let node = MemoryNode::new("Precision test".to_string(), embedding.clone());
        let bytes = serialize_node(&node).unwrap();
        let restored = deserialize_node(&bytes).unwrap();

        for (i, (orig, rest)) in embedding.iter().zip(restored.embedding.iter()).enumerate() {
            assert_eq!(
                orig.to_bits(),
                rest.to_bits(),
                "Embedding value at index {} differs: {} vs {}",
                i,
                orig,
                rest
            );
        }
    }

    #[test]
    fn test_timestamps_preserved() {
        let node = create_test_node();
        let original_created = node.created_at;
        let original_accessed = node.accessed_at;

        let bytes = serialize_node(&node).unwrap();
        let restored = deserialize_node(&bytes).unwrap();

        assert_eq!(restored.created_at, original_created);
        assert_eq!(restored.accessed_at, original_accessed);
    }

    #[test]
    fn test_edge_timestamps_preserved() {
        let mut edge = create_test_edge();
        edge.record_traversal();
        let original_created = edge.created_at;
        let original_traversed = edge.last_traversed_at;

        let bytes = serialize_edge(&edge).unwrap();
        let restored = deserialize_edge(&bytes).unwrap();

        assert_eq!(restored.created_at, original_created);
        assert_eq!(restored.last_traversed_at, original_traversed);
    }

    // =========================================================================
    // Additional Tests for Full Coverage
    // =========================================================================

    #[test]
    fn test_embedding_invalid_sizes() {
        for invalid_len in [1, 2, 3, 5, 7, 9, 11, 13, 15, 17] {
            let bytes = vec![0u8; invalid_len];
            let result = deserialize_embedding(&bytes);
            assert!(
                matches!(result, Err(SerializationError::InvalidEmbeddingSize { .. })),
                "Should fail for length {}",
                invalid_len
            );
        }
    }

    #[test]
    fn test_embedding_valid_sizes() {
        for valid_len in [0, 4, 8, 12, 16, 20, 100, 400, 6144] {
            let bytes = vec![0u8; valid_len];
            let result = deserialize_embedding(&bytes);
            assert!(result.is_ok(), "Should succeed for length {}", valid_len);
            assert_eq!(result.unwrap().len(), valid_len / 4);
        }
    }

    #[test]
    fn test_node_default_can_serialize() {
        let node = MemoryNode::default();
        let bytes = serialize_node(&node).unwrap();
        let restored = deserialize_node(&bytes).unwrap();
        assert_eq!(node, restored);
    }

    #[test]
    fn test_edge_default_can_serialize() {
        let edge = GraphEdge::default();
        let bytes = serialize_edge(&edge).unwrap();
        let restored = deserialize_edge(&bytes).unwrap();
        assert_eq!(edge, restored);
    }
}
