//! Bio-nervous system layer types.
//!
//! Defines the 5-layer architecture: Sensing, Reflex, Memory, Learning, Coherence

use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::CognitivePulse;

/// Bio-nervous system layer identifier.
///
/// The 5-layer architecture models the nervous system with distinct
/// processing layers, each with its own latency budget and function.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum LayerId {
    /// Multi-modal input processing (5ms budget)
    Sensing,
    /// Pattern-matched fast responses (100μs budget)
    Reflex,
    /// Modern Hopfield associative storage (1ms budget)
    Memory,
    /// UTL-driven weight optimization (10ms budget)
    Learning,
    /// Global state synchronization (10ms budget)
    Coherence,
}

impl LayerId {
    /// Get the latency budget for this layer.
    pub fn latency_budget(&self) -> Duration {
        match self {
            LayerId::Sensing => Duration::from_millis(5),
            LayerId::Reflex => Duration::from_micros(100),
            LayerId::Memory => Duration::from_millis(1),
            LayerId::Learning => Duration::from_millis(10),
            LayerId::Coherence => Duration::from_millis(10),
        }
    }

    /// Get human-readable layer name.
    pub fn display_name(&self) -> &'static str {
        match self {
            LayerId::Sensing => "Sensing Layer",
            LayerId::Reflex => "Reflex Layer",
            LayerId::Memory => "Memory Layer",
            LayerId::Learning => "Learning Layer",
            LayerId::Coherence => "Coherence Layer",
        }
    }

    /// Get the layer's function description.
    pub fn function(&self) -> &'static str {
        match self {
            LayerId::Sensing => "Multi-modal input processing",
            LayerId::Reflex => "Pattern-matched fast responses",
            LayerId::Memory => "Modern Hopfield associative storage",
            LayerId::Learning => "UTL-driven weight optimization",
            LayerId::Coherence => "Global state synchronization",
        }
    }

    /// Get all layers in processing order.
    pub fn all() -> &'static [LayerId] {
        &[
            LayerId::Sensing,
            LayerId::Reflex,
            LayerId::Memory,
            LayerId::Learning,
            LayerId::Coherence,
        ]
    }
}

/// Input to a nervous system layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInput {
    /// Request identifier for tracing
    pub request_id: String,

    /// Input content
    pub content: String,

    /// Optional embedding (if pre-computed)
    pub embedding: Option<Vec<f32>>,

    /// Context from previous layers
    pub context: LayerContext,
}

impl LayerInput {
    /// Create a new layer input.
    pub fn new(request_id: String, content: String) -> Self {
        Self {
            request_id,
            content,
            embedding: None,
            context: LayerContext::default(),
        }
    }
}

/// Context passed between layers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerContext {
    /// Accumulated latency in microseconds
    pub accumulated_latency_us: u64,

    /// Results from previous layers
    pub layer_results: Vec<LayerResult>,

    /// Current pulse state
    pub pulse: CognitivePulse,
}

impl LayerContext {
    /// Add latency from a layer processing step.
    pub fn add_latency(&mut self, duration: Duration) {
        self.accumulated_latency_us += duration.as_micros() as u64;
    }

    /// Get total accumulated latency.
    pub fn total_latency(&self) -> Duration {
        Duration::from_micros(self.accumulated_latency_us)
    }
}

/// Output from a nervous system layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerOutput {
    /// Layer that produced this output
    pub layer: LayerId,

    /// Processing result
    pub result: LayerResult,

    /// Updated pulse
    pub pulse: CognitivePulse,

    /// Processing duration in microseconds
    pub duration_us: u64,
}

/// Result data from layer processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerResult {
    /// Layer identifier
    pub layer: LayerId,

    /// Whether processing was successful
    pub success: bool,

    /// Result data (layer-specific)
    pub data: serde_json::Value,

    /// Optional error message
    pub error: Option<String>,
}

impl LayerResult {
    /// Create a successful result.
    pub fn success(layer: LayerId, data: serde_json::Value) -> Self {
        Self {
            layer,
            success: true,
            data,
            error: None,
        }
    }

    /// Create a failed result.
    pub fn failure(layer: LayerId, error: String) -> Self {
        Self {
            layer,
            success: false,
            data: serde_json::Value::Null,
            error: Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_latency_budgets() {
        assert_eq!(LayerId::Sensing.latency_budget(), Duration::from_millis(5));
        assert_eq!(LayerId::Reflex.latency_budget(), Duration::from_micros(100));
        assert_eq!(LayerId::Memory.latency_budget(), Duration::from_millis(1));
    }

    #[test]
    fn test_all_layers() {
        let layers = LayerId::all();
        assert_eq!(layers.len(), 5);
        assert_eq!(layers[0], LayerId::Sensing);
        assert_eq!(layers[4], LayerId::Coherence);
    }

    #[test]
    fn test_layer_context_latency() {
        let mut ctx = LayerContext::default();
        ctx.add_latency(Duration::from_millis(5));
        ctx.add_latency(Duration::from_millis(3));
        assert_eq!(ctx.total_latency(), Duration::from_millis(8));
    }
}
