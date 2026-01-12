//! Memory cluster types for sub-goal discovery.

use crate::autonomous::curation::MemoryId;

/// Memory cluster representing a group of related memories
#[derive(Clone, Debug)]
pub struct MemoryCluster {
    /// Centroid embedding of the cluster (normalized)
    pub centroid: Vec<f32>,
    /// Member memory IDs in this cluster
    pub members: Vec<MemoryId>,
    /// Coherence score of the cluster (0.0 to 1.0)
    pub coherence: f32,
    /// Optional label or description extracted from members
    pub label: Option<String>,
    /// Average alignment of members to current goals
    pub avg_alignment: f32,
}

impl MemoryCluster {
    /// Create a new memory cluster
    pub fn new(centroid: Vec<f32>, members: Vec<MemoryId>, coherence: f32) -> Self {
        assert!(!centroid.is_empty(), "Centroid cannot be empty");
        assert!(
            (0.0..=1.0).contains(&coherence),
            "Coherence must be in [0.0, 1.0]"
        );

        Self {
            centroid,
            members,
            coherence,
            label: None,
            avg_alignment: 0.0,
        }
    }

    /// Create a cluster with a label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the average alignment
    pub fn with_avg_alignment(mut self, alignment: f32) -> Self {
        self.avg_alignment = alignment.clamp(0.0, 1.0);
        self
    }

    /// Get the cluster size
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Check if cluster is empty
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}
