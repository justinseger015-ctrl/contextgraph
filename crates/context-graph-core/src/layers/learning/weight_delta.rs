//! Weight Delta - Result of UTL computation.

use serde::{Deserialize, Serialize};

/// Weight delta from UTL computation: Δw = η*(S⊗C_w)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightDelta {
    /// The computed weight change: η*(S⊗C_w)
    pub value: f32,

    /// Surprise component (S) used in computation
    pub surprise: f32,

    /// Weighted coherence component (C_w) used in computation
    pub coherence_w: f32,

    /// Learning rate (η) used in computation
    pub learning_rate: f32,

    /// Whether gradient clipping was applied
    pub was_clipped: bool,
}

impl WeightDelta {
    /// Get the absolute magnitude of the weight delta.
    pub fn magnitude(&self) -> f32 {
        self.value.abs()
    }

    /// Check if this delta should trigger consolidation.
    pub fn should_consolidate(&self, threshold: f32) -> bool {
        self.magnitude() > threshold
    }
}
