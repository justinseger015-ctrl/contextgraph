//! L4 Learning Layer - UTL-driven weight optimization.
//!
//! This layer computes weight updates based on surprise and coherence signals
//! following the UTL formula: W' = W + η*(S⊗C_w)

use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::error::CoreResult;
use crate::traits::NervousLayer;
use crate::types::{LayerId, LayerInput, LayerOutput, LayerResult};

#[allow(deprecated)]
use super::constants::DEFAULT_CONSOLIDATION_THRESHOLD;
use super::utl_computer::UtlWeightComputer;

/// L4 Learning Layer - UTL-driven weight optimization.
///
/// This layer computes weight updates based on surprise and coherence signals
/// following the UTL formula: W' = W + η*(S⊗C_w)
///
/// # Constitution Compliance
///
/// - Latency: <10ms (CRITICAL)
/// - Frequency: 100Hz
/// - Grad clip: 1.0
/// - Components: UTL optimizer (this), neuromod controller (external)
///
/// # No Fallbacks
///
/// Per AP-007: If UTL computation fails, this layer returns an error.
/// Invalid inputs (NaN/Infinity) are rejected per AP-009.
///
/// # Consolidation Trigger
///
/// When weight delta magnitude exceeds the consolidation threshold,
/// the layer signals that consolidation should occur (for L5/dream).
#[derive(Debug)]
pub struct LearningLayer {
    /// UTL weight computation engine
    weight_computer: UtlWeightComputer,

    /// Consolidation threshold - signal consolidation when exceeded
    consolidation_threshold: f32,

    /// Total layer processing time in microseconds
    total_processing_us: AtomicU64,

    /// Total layer invocations
    invocation_count: AtomicU64,

    /// Total consolidation triggers
    consolidation_triggers: AtomicU64,
}

impl LearningLayer {
    /// Create a new Learning layer with default configuration.
    #[allow(deprecated)]
    pub fn new() -> Self {
        Self {
            weight_computer: UtlWeightComputer::default(),
            consolidation_threshold: DEFAULT_CONSOLIDATION_THRESHOLD,
            total_processing_us: AtomicU64::new(0),
            invocation_count: AtomicU64::new(0),
            consolidation_triggers: AtomicU64::new(0),
        }
    }

    /// Create with custom learning rate.
    pub fn with_learning_rate(mut self, rate: f32) -> Self {
        self.weight_computer = UtlWeightComputer::new(rate);
        self
    }

    /// Create with custom consolidation threshold.
    pub fn with_consolidation_threshold(mut self, threshold: f32) -> Self {
        self.consolidation_threshold = threshold.clamp(0.01, 1.0);
        self
    }

    /// Get the current learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.weight_computer.learning_rate()
    }

    /// Get the consolidation threshold.
    pub fn consolidation_threshold(&self) -> f32 {
        self.consolidation_threshold
    }

    /// Get the number of consolidation triggers.
    pub fn consolidation_trigger_count(&self) -> u64 {
        self.consolidation_triggers.load(Ordering::Relaxed)
    }

    /// Get average processing time in microseconds.
    pub fn avg_processing_us(&self) -> f64 {
        let count = self.invocation_count.load(Ordering::Relaxed);
        let total = self.total_processing_us.load(Ordering::Relaxed);
        if count > 0 {
            total as f64 / count as f64
        } else {
            0.0
        }
    }

    /// Extract surprise signal from layer context.
    ///
    /// Computes surprise from:
    /// - L1 Sensing: delta_s (entropy change)
    /// - L3 Memory: novelty (if available)
    ///
    /// Combined: S = delta_s × novelty
    fn compute_surprise(&self, context: &crate::types::LayerContext) -> CoreResult<f32> {
        // Get delta_s from L1 Sensing result
        let delta_s = context
            .layer_results
            .iter()
            .find(|r| r.layer == LayerId::Sensing)
            .and_then(|r| r.data.get("delta_s"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);

        // Get novelty from L3 Memory if available
        let memory_novelty = context
            .layer_results
            .iter()
            .find(|r| r.layer == LayerId::Memory)
            .and_then(|r| r.data.get("novelty"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);

        // If we have retrieval count from memory, compute novelty from that
        let retrieval_novelty = context
            .layer_results
            .iter()
            .find(|r| r.layer == LayerId::Memory)
            .and_then(|r| r.data.get("retrieval_count"))
            .and_then(|v| v.as_u64())
            .map(|count| {
                // More retrievals = less novel (inverse relationship)
                // 0 retrievals = high novelty (1.0)
                // 10+ retrievals = low novelty (~0.1)
                1.0 / (1.0 + count as f32 * 0.1)
            });

        // Combine available signals
        let surprise = match (delta_s, memory_novelty.or(retrieval_novelty)) {
            (Some(ds), Some(nov)) => ds * nov,
            (Some(ds), None) => ds,
            (None, Some(nov)) => nov,
            (None, None) => {
                // Use pulse entropy as fallback surprise indicator
                context.pulse.entropy
            }
        };

        Ok(surprise.clamp(0.0, 1.0))
    }
}

impl Default for LearningLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl NervousLayer for LearningLayer {
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput> {
        let start = Instant::now();

        // Compute surprise from context
        let surprise = self.compute_surprise(&input.context)?;

        // Get coherence from pulse (weighted coherence C_w)
        let coherence_w = input.context.pulse.coherence;

        // Compute weight update: Δw = η*(S⊗C_w)
        let weight_delta = self.weight_computer.compute_update(surprise, coherence_w)?;

        // Check consolidation trigger
        let should_consolidate = weight_delta.should_consolidate(self.consolidation_threshold);

        if should_consolidate {
            self.consolidation_triggers.fetch_add(1, Ordering::Relaxed);
        }

        let duration = start.elapsed();
        let duration_us = duration.as_micros() as u64;

        // Record metrics
        self.total_processing_us
            .fetch_add(duration_us, Ordering::Relaxed);
        self.invocation_count.fetch_add(1, Ordering::Relaxed);

        // Check latency budget
        let budget = self.latency_budget();
        if duration > budget {
            tracing::warn!(
                "LearningLayer exceeded latency budget: {:?} > {:?}",
                duration,
                budget
            );
        }

        // Update pulse with learning information
        let mut updated_pulse = input.context.pulse.clone();
        // Coherence improves slightly with positive learning signal
        if weight_delta.value > 0.0 {
            updated_pulse.coherence =
                (updated_pulse.coherence + weight_delta.value * 0.1).clamp(0.0, 1.0);
        }
        // Update coherence delta to reflect the learning
        updated_pulse.coherence_delta = weight_delta.value;

        // Build result data
        let result_data = serde_json::json!({
            "weight_delta": weight_delta.value,
            "surprise": weight_delta.surprise,
            "coherence_w": weight_delta.coherence_w,
            "learning_rate": weight_delta.learning_rate,
            "was_clipped": weight_delta.was_clipped,
            "should_consolidate": should_consolidate,
            "consolidation_threshold": self.consolidation_threshold,
            "duration_us": duration_us,
            "within_budget": duration <= budget,
        });

        Ok(LayerOutput {
            layer: LayerId::Learning,
            result: LayerResult::success(LayerId::Learning, result_data),
            pulse: updated_pulse,
            duration_us,
        })
    }

    fn latency_budget(&self) -> Duration {
        Duration::from_millis(10) // 10ms budget per constitution
    }

    fn layer_id(&self) -> LayerId {
        LayerId::Learning
    }

    fn layer_name(&self) -> &'static str {
        "Learning Layer"
    }

    async fn health_check(&self) -> CoreResult<bool> {
        // Verify UTL computation works with valid inputs
        let test = self.weight_computer.compute_update(0.5, 0.5);
        if test.is_err() {
            return Ok(false);
        }

        // Verify invalid inputs are rejected
        let nan_test = self.weight_computer.compute_update(f32::NAN, 0.5);
        if nan_test.is_ok() {
            return Ok(false); // Should have failed
        }

        Ok(true)
    }
}
