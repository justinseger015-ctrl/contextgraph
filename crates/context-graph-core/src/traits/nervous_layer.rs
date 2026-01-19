//! Nervous layer trait for bio-nervous system architecture.

use async_trait::async_trait;
use std::time::Duration;

use crate::error::CoreResult;
use crate::types::{LayerId, LayerInput, LayerOutput};

/// Bio-nervous system layer abstraction.
///
/// Each layer in the 4-layer architecture implements this trait.
/// Layers process input within their latency budget and pass results downstream.
///
/// # Layers
///
/// 1. **Sensing** (5ms): Multi-modal input processing
/// 2. **Memory** (1ms): Associative memory storage
/// 3. **Learning** (10ms): UTL-driven weight optimization
/// 4. **Coherence** (10ms): Global state synchronization
///
/// # Example
///
/// ```
/// use context_graph_core::types::LayerId;
/// use std::time::Duration;
///
/// // Nervous layer latency budgets from constitution
/// let sensing_budget = Duration::from_millis(5);   // Sensing: 5ms
/// let memory_budget = Duration::from_millis(1);    // Memory: 1ms
/// let learning_budget = Duration::from_millis(10); // Learning: 10ms
///
/// assert!(memory_budget < sensing_budget);
/// assert!(sensing_budget < learning_budget);
/// ```
#[async_trait]
pub trait NervousLayer: Send + Sync {
    /// Process input through this layer.
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput>;

    /// Get the latency budget for this layer.
    fn latency_budget(&self) -> Duration;

    /// Get the layer identifier.
    fn layer_id(&self) -> LayerId;

    /// Get human-readable layer name.
    fn layer_name(&self) -> &'static str;

    /// Check if layer is healthy and ready.
    async fn health_check(&self) -> CoreResult<bool>;
}
