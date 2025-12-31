//! Memex storage trait abstraction.
//!
//! The Memex trait defines the storage contract for MemoryNode and GraphEdge
//! persistence. Named after Vannevar Bush's conceptual memory machine.
//!
//! # Implementors
//! - `RocksDbMemex`: Production RocksDB implementation (TASK-M02-016)
//!
//! # Constitution Reference
//! - SEC-06: All delete operations must be soft deletes with 30-day recovery
//! - AP-010: store_memory requires rationale

// TODO: Implement Memex trait in TASK-M02-026
// Required methods:
// - store_node(node: &MemoryNode) -> Result<NodeId, StorageError>
// - get_node(id: NodeId) -> Result<Option<MemoryNode>, StorageError>
// - update_node(node: &MemoryNode) -> Result<(), StorageError>
// - delete_node(id: NodeId, soft: bool) -> Result<(), StorageError>
// - store_edge(edge: &GraphEdge) -> Result<EdgeId, StorageError>
// - get_edge(id: EdgeId) -> Result<Option<GraphEdge>, StorageError>
// - get_edges_from(node_id: NodeId) -> Result<Vec<GraphEdge>, StorageError>
// - get_edges_to(node_id: NodeId) -> Result<Vec<GraphEdge>, StorageError>
