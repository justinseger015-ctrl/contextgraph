//! Purpose query target types.
//!
//! This module provides [`PurposeQueryTarget`] which specifies what to search for
//! in a purpose-based query.

use uuid::Uuid;

use crate::types::fingerprint::PurposeVector;

use super::super::error::{PurposeIndexError, PurposeIndexResult};

/// Specifies the target for a purpose-based query.
///
/// # Variants
///
/// - `Vector`: Query with an existing `PurposeVector`
/// - `Pattern`: Query for pattern clusters with constraints
/// - `FromMemory`: Find memories with similar purpose to a given memory
///
/// # Fail-Fast
///
/// The `Pattern` variant validates its parameters at construction:
/// - `coherence_threshold` must be in [0.0, 1.0]
#[derive(Clone, Debug)]
pub enum PurposeQueryTarget {
    /// Query with a purpose vector.
    ///
    /// Searches for memories with similar 13D alignment profiles.
    Vector(PurposeVector),

    /// Query for pattern clusters.
    ///
    /// Finds clusters of memories with similar purpose patterns.
    ///
    /// # Fields
    ///
    /// - `min_cluster_size`: Minimum number of members in a cluster
    /// - `coherence_threshold`: Minimum coherence score [0.0, 1.0]
    Pattern {
        /// Minimum number of memories in a cluster to be returned.
        min_cluster_size: usize,
        /// Minimum coherence threshold [0.0, 1.0].
        coherence_threshold: f32,
    },

    /// Find memories with similar purpose to a given memory.
    ///
    /// The target memory must exist in the index.
    FromMemory(Uuid),
}

impl PurposeQueryTarget {
    /// Create a Vector target from a PurposeVector.
    #[inline]
    pub fn vector(pv: PurposeVector) -> Self {
        Self::Vector(pv)
    }

    /// Create a Pattern target with validation.
    ///
    /// # Errors
    ///
    /// Returns `PurposeIndexError::InvalidQuery` if:
    /// - `min_cluster_size` is 0
    /// - `coherence_threshold` is not in [0.0, 1.0]
    pub fn pattern(min_cluster_size: usize, coherence_threshold: f32) -> PurposeIndexResult<Self> {
        if min_cluster_size == 0 {
            return Err(PurposeIndexError::invalid_query(
                "min_cluster_size must be > 0",
            ));
        }
        if !(0.0..=1.0).contains(&coherence_threshold) {
            return Err(PurposeIndexError::invalid_query(format!(
                "coherence_threshold {} must be in [0.0, 1.0]",
                coherence_threshold
            )));
        }
        Ok(Self::Pattern {
            min_cluster_size,
            coherence_threshold,
        })
    }

    /// Create a FromMemory target.
    #[inline]
    pub fn from_memory(memory_id: Uuid) -> Self {
        Self::FromMemory(memory_id)
    }

    /// Check if this target requires looking up an existing memory.
    #[inline]
    pub fn requires_memory_lookup(&self) -> bool {
        matches!(self, Self::FromMemory(_))
    }
}
