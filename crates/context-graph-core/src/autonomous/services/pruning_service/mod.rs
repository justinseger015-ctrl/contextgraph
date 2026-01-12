//! NORTH-012: PruningService
//!
//! Service for identifying and removing low-value memories from the knowledge graph.
//! Uses alignment scores, age, connection count, and quality metrics to determine
//! which memories should be pruned.
//!
//! # Architecture
//!
//! The PruningService operates in three phases:
//! 1. **Identification**: Scan memories and identify pruning candidates
//! 2. **Evaluation**: Score each candidate and determine prune reason
//! 3. **Execution**: Remove candidates up to daily limit
//!
//! # Fail-Fast Behavior
//!
//! All methods fail immediately on invalid input rather than returning
//! partial results or silently ignoring errors.
//!
//! # Module Structure
//!
//! - `types`: Data structures (PruneReason, MemoryMetadata, PruningCandidate, etc.)
//! - `service`: Core PruningService implementation

pub mod service;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use service::PruningService;
pub use types::{
    ExtendedPruningConfig, MemoryMetadata, PruneReason, PruningCandidate, PruningReport,
};
