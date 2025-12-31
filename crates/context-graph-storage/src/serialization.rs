//! Bincode serialization utilities.
//!
//! Provides efficient binary serialization for MemoryNode, GraphEdge,
//! and EmbeddingVector using bincode.
//!
//! # Performance
//! - MemoryNode serialized: ~6.5KB average (with 1536D embedding)
//! - GraphEdge serialized: ~200 bytes
//! - Round-trip overhead: < 100μs

// TODO: Implement in TASK-M02-014
// Required functions:
// - serialize_node(node: &MemoryNode) -> Result<Vec<u8>, StorageError>
// - deserialize_node(bytes: &[u8]) -> Result<MemoryNode, StorageError>
// - serialize_edge(edge: &GraphEdge) -> Result<Vec<u8>, StorageError>
// - deserialize_edge(bytes: &[u8]) -> Result<GraphEdge, StorageError>
// - serialize_embedding(embedding: &EmbeddingVector) -> Result<Vec<u8>, StorageError>
// - deserialize_embedding(bytes: &[u8]) -> Result<EmbeddingVector, StorageError>
