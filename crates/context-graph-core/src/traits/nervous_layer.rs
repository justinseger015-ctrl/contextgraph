//! Nervous layer trait for bio-nervous system architecture.

use async_trait::async_trait;
use std::time::Duration;

use crate::error::CoreResult;
use crate::types::{LayerId, LayerInput, LayerOutput};

/// Bio-nervous system layer abstraction.
///
/// Each layer in the 5-layer architecture implements this trait.
/// Layers process input within their latency budget and pass results downstream.
///
/// # Layers
///
/// 1. **Sensing** (5ms): Multi-modal input processing
/// 2. **Reflex** (100μs): Pattern-matched fast responses
/// 3. **Memory** (1ms): Modern Hopfield associative storage
/// 4. **Learning** (10ms): UTL-driven weight optimization
/// 5. **Coherence** (10ms): Global state synchronization
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_core::traits::NervousLayer;
///
/// let layer = ReflexLayer::new();
/// let output = layer.process(input).await?;
/// assert!(output.duration_us < layer.latency_budget().as_micros() as u64);
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
