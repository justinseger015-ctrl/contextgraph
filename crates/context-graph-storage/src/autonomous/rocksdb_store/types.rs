//! Storage-local types for autonomous system.
//!
//! These types are defined in the storage crate (not in context-graph-core)
//! because they are specific to the storage layer's requirements.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use context_graph_core::autonomous::{GoalId, MemoryId};

// ============================================================================
// LineageEvent
// ============================================================================

/// Lineage event for traceability of autonomous operations.
///
/// Records significant system events for audit and debugging.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineageEvent {
    /// Unique event identifier.
    pub id: Uuid,
    /// Event timestamp.
    pub timestamp: DateTime<Utc>,
    /// Event type (e.g., "bootstrap", "drift_correction", "goal_evolution").
    pub event_type: String,
    /// Human-readable description.
    pub description: String,
    /// Associated goal ID, if applicable.
    pub goal_id: Option<GoalId>,
    /// Associated memory ID, if applicable.
    pub memory_id: Option<MemoryId>,
    /// Additional metadata as JSON string.
    pub metadata: Option<String>,
}

impl LineageEvent {
    /// Create a new lineage event with auto-generated ID and current timestamp.
    pub fn new(event_type: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: event_type.into(),
            description: description.into(),
            goal_id: None,
            memory_id: None,
            metadata: None,
        }
    }

    /// Create a lineage event with goal association.
    pub fn with_goal(mut self, goal_id: GoalId) -> Self {
        self.goal_id = Some(goal_id);
        self
    }

    /// Create a lineage event with memory association.
    pub fn with_memory(mut self, memory_id: MemoryId) -> Self {
        self.memory_id = Some(memory_id);
        self
    }

    /// Create a lineage event with metadata.
    pub fn with_metadata(mut self, metadata: impl Into<String>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }

    /// Get the timestamp in milliseconds for key generation.
    pub fn timestamp_ms(&self) -> i64 {
        self.timestamp.timestamp_millis()
    }
}

// ============================================================================
// ConsolidationRecord
// ============================================================================

/// Record of a memory consolidation operation.
///
/// Tracks when and how memories were merged or consolidated.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsolidationRecord {
    /// Unique record identifier.
    pub id: Uuid,
    /// Consolidation timestamp.
    pub timestamp: DateTime<Utc>,
    /// Source memory IDs that were consolidated.
    pub source_memories: Vec<MemoryId>,
    /// Target memory ID (merged result).
    pub target_memory: MemoryId,
    /// Similarity score that triggered consolidation.
    pub similarity_score: f32,
    /// Alignment difference (theta_diff) at consolidation.
    pub theta_diff: f32,
    /// Whether consolidation was successful.
    pub success: bool,
    /// Error message if consolidation failed.
    pub error_message: Option<String>,
}

impl ConsolidationRecord {
    /// Create a new successful consolidation record.
    pub fn success(
        source_memories: Vec<MemoryId>,
        target_memory: MemoryId,
        similarity_score: f32,
        theta_diff: f32,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            source_memories,
            target_memory,
            similarity_score,
            theta_diff,
            success: true,
            error_message: None,
        }
    }

    /// Create a new failed consolidation record.
    pub fn failure(
        source_memories: Vec<MemoryId>,
        target_memory: MemoryId,
        similarity_score: f32,
        theta_diff: f32,
        error: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            source_memories,
            target_memory,
            similarity_score,
            theta_diff,
            success: false,
            error_message: Some(error.into()),
        }
    }

    /// Get the timestamp in milliseconds for key generation.
    pub fn timestamp_ms(&self) -> i64 {
        self.timestamp.timestamp_millis()
    }
}
