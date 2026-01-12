//! UTL Weight Computer - Core computation engine.
//!
//! Implements the canonical weight update formula: W' = W + η*(S⊗C_w)

use crate::error::{CoreError, CoreResult};

use super::constants::{DEFAULT_LEARNING_RATE, GRADIENT_CLIP};
use super::weight_delta::WeightDelta;

/// UTL Weight Computer - implements W' = W + η*(S⊗C_w)
///
/// This is the core computation engine for L4 Learning.
/// It computes weight updates based on surprise and coherence signals.
///
/// # Formula
///
/// W' = W + η*(S⊗C_w)
///
/// Where:
/// - η = learning rate (default 0.0005)
/// - S = surprise signal [0, 1]
/// - C_w = weighted coherence [0, 1]
/// - ⊗ = element-wise product (scalar here for global signal)
///
/// # Gradient Clipping
///
/// The delta is clipped to [-1.0, 1.0] per constitution.
#[derive(Debug, Clone)]
pub struct UtlWeightComputer {
    /// Learning rate (η)
    learning_rate: f32,

    /// Gradient clipping bound
    grad_clip: f32,
}

impl UtlWeightComputer {
    /// Create a new UTL weight computer with specified learning rate.
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            grad_clip: GRADIENT_CLIP,
        }
    }

    /// Create with custom gradient clipping bound.
    pub fn with_grad_clip(mut self, clip: f32) -> Self {
        self.grad_clip = clip.abs().max(0.01); // Ensure positive minimum
        self
    }

    /// Compute weight update: Δw = η*(S⊗C_w)
    ///
    /// # Arguments
    ///
    /// * `surprise` - Surprise signal S from sensing/memory layers
    /// * `coherence_w` - Weighted coherence C_w from pulse
    ///
    /// # Returns
    ///
    /// WeightDelta containing the computed delta and metadata.
    ///
    /// # Errors
    ///
    /// Returns error for invalid (NaN/Infinity) inputs per AP-009.
    pub fn compute_update(&self, surprise: f32, coherence_w: f32) -> CoreResult<WeightDelta> {
        // Validate inputs - NO silent fallbacks per AP-009
        if surprise.is_nan() || surprise.is_infinite() {
            return Err(CoreError::UtlError(format!(
                "Invalid surprise value: {} - NaN/Infinity not allowed per AP-009",
                surprise
            )));
        }
        if coherence_w.is_nan() || coherence_w.is_infinite() {
            return Err(CoreError::UtlError(format!(
                "Invalid coherence value: {} - NaN/Infinity not allowed per AP-009",
                coherence_w
            )));
        }

        // Clamp inputs to valid range
        let s = surprise.clamp(0.0, 1.0);
        let c = coherence_w.clamp(0.0, 1.0);

        // S⊗C_w (element-wise product, scalar for global signal)
        let learning_signal = s * c;

        // η*(S⊗C_w)
        let raw_delta = self.learning_rate * learning_signal;

        // Apply gradient clipping
        let (delta, was_clipped) = if raw_delta.abs() > self.grad_clip {
            (raw_delta.signum() * self.grad_clip, true)
        } else {
            (raw_delta, false)
        };

        Ok(WeightDelta {
            value: delta,
            surprise: s,
            coherence_w: c,
            learning_rate: self.learning_rate,
            was_clipped,
        })
    }

    /// Get the current learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Get the gradient clip bound.
    pub fn grad_clip(&self) -> f32 {
        self.grad_clip
    }
}

impl Default for UtlWeightComputer {
    fn default() -> Self {
        Self::new(DEFAULT_LEARNING_RATE)
    }
}
