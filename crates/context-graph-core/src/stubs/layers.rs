//! Stub implementations of NervousLayer for all 5 bio-nervous system layers.
//!
//! These implementations provide deterministic, instant responses for the
//! Ghost System phase (Phase 0). Production implementations will replace
//! these with real processing logic.
//!
//! Each stub:
//! - Returns immediately (no sleep)
//! - Reports duration well within latency budget
//! - Returns deterministic output based on input
//! - Always passes health_check()

use async_trait::async_trait;
use serde_json::json;
use std::time::Duration;

use crate::error::CoreResult;
use crate::traits::NervousLayer;
use crate::types::{
    CognitivePulse, LayerId, LayerInput, LayerOutput, LayerResult, SuggestedAction,
};

// =============================================================================
// 1. StubSensingLayer - 5ms latency budget
// =============================================================================

/// Stub implementation of the Sensing layer.
///
/// The Sensing layer handles multi-modal input processing.
/// This stub immediately returns a processed result with sensible defaults.
///
/// # Latency Budget
/// Real implementation: 5ms max
/// Stub implementation: <1us (instant return)
#[derive(Debug, Clone, Default)]
pub struct StubSensingLayer {
    /// Configuration flag (unused in stub, for future compatibility)
    _config: StubLayerConfig,
}

impl StubSensingLayer {
    /// Create a new stub sensing layer.
    pub fn new() -> Self {
        Self {
            _config: StubLayerConfig::default(),
        }
    }
}

#[async_trait]
impl NervousLayer for StubSensingLayer {
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput> {
        // Compute deterministic hash from input for reproducibility
        let input_hash = compute_input_hash(&input.content);

        // Generate deterministic entropy/coherence based on input
        let entropy = (input_hash % 100) as f32 / 200.0 + 0.2; // Range: 0.2-0.7
        let coherence = 0.6 + (input_hash % 50) as f32 / 200.0; // Range: 0.6-0.85

        let result = LayerResult::success(
            LayerId::Sensing,
            json!({
                "input_processed": true,
                "content_length": input.content.len(),
                "request_id": input.request_id,
                "modality": "text",
                "tokenized": true
            }),
        );

        Ok(LayerOutput {
            layer: LayerId::Sensing,
            result,
            pulse: CognitivePulse::new(entropy, coherence, SuggestedAction::Continue),
            duration_us: 500, // 500us stub value, well under 5ms budget
        })
    }

    fn latency_budget(&self) -> Duration {
        Duration::from_millis(5)
    }

    fn layer_id(&self) -> LayerId {
        LayerId::Sensing
    }

    fn layer_name(&self) -> &'static str {
        "Sensing Layer (Stub)"
    }

    async fn health_check(&self) -> CoreResult<bool> {
        Ok(true)
    }
}

// =============================================================================
// 2. StubReflexLayer - 100us latency budget
// =============================================================================

/// Stub implementation of the Reflex layer.
///
/// The Reflex layer provides pattern-matched fast responses.
/// This stub immediately returns a pattern match result.
///
/// # Latency Budget
/// Real implementation: 100us max (very fast)
/// Stub implementation: <1us (instant return)
#[derive(Debug, Clone, Default)]
pub struct StubReflexLayer {
    _config: StubLayerConfig,
}

impl StubReflexLayer {
    /// Create a new stub reflex layer.
    pub fn new() -> Self {
        Self {
            _config: StubLayerConfig::default(),
        }
    }
}

#[async_trait]
impl NervousLayer for StubReflexLayer {
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput> {
        let input_hash = compute_input_hash(&input.content);

        // Reflex layer has very low latency, so entropy should be low (fast decisions)
        let entropy = 0.2 + (input_hash % 30) as f32 / 200.0; // Range: 0.2-0.35
        let coherence = 0.7 + (input_hash % 40) as f32 / 200.0; // Range: 0.7-0.9

        // Deterministic pattern match simulation
        let pattern_found = input.content.len() > 10;

        let result = LayerResult::success(
            LayerId::Reflex,
            json!({
                "pattern_matched": pattern_found,
                "match_confidence": if pattern_found { 0.85 } else { 0.0 },
                "reflex_triggered": pattern_found,
                "patterns_checked": 5
            }),
        );

        Ok(LayerOutput {
            layer: LayerId::Reflex,
            result,
            pulse: CognitivePulse::new(entropy, coherence, SuggestedAction::Continue),
            duration_us: 50, // 50us stub value, well under 100us budget
        })
    }

    fn latency_budget(&self) -> Duration {
        Duration::from_micros(100)
    }

    fn layer_id(&self) -> LayerId {
        LayerId::Reflex
    }

    fn layer_name(&self) -> &'static str {
        "Reflex Layer (Stub)"
    }

    async fn health_check(&self) -> CoreResult<bool> {
        Ok(true)
    }
}

// =============================================================================
// 3. StubMemoryLayer - 1ms latency budget
// =============================================================================

/// Stub implementation of the Memory layer.
///
/// The Memory layer handles Modern Hopfield associative storage.
/// This stub simulates memory retrieval operations.
///
/// # Latency Budget
/// Real implementation: 1ms max
/// Stub implementation: <1us (instant return)
#[derive(Debug, Clone, Default)]
pub struct StubMemoryLayer {
    _config: StubLayerConfig,
}

impl StubMemoryLayer {
    /// Create a new stub memory layer.
    pub fn new() -> Self {
        Self {
            _config: StubLayerConfig::default(),
        }
    }
}

#[async_trait]
impl NervousLayer for StubMemoryLayer {
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput> {
        let input_hash = compute_input_hash(&input.content);

        // Memory layer has moderate entropy (some uncertainty in retrieval)
        let entropy = 0.3 + (input_hash % 40) as f32 / 200.0; // Range: 0.3-0.5
        let coherence = 0.65 + (input_hash % 35) as f32 / 200.0; // Range: 0.65-0.825

        // Deterministic memory retrieval simulation
        let memories_found = (input_hash % 5) + 1; // 1-5 memories

        let result = LayerResult::success(
            LayerId::Memory,
            json!({
                "memories_retrieved": memories_found,
                "retrieval_scores": vec![0.95, 0.87, 0.76, 0.68, 0.55][..memories_found as usize].to_vec(),
                "hopfield_energy": -0.85,
                "cache_hit": input_hash % 3 == 0
            }),
        );

        Ok(LayerOutput {
            layer: LayerId::Memory,
            result,
            pulse: CognitivePulse::new(entropy, coherence, SuggestedAction::Continue),
            duration_us: 200, // 200us stub value, well under 1ms budget
        })
    }

    fn latency_budget(&self) -> Duration {
        Duration::from_millis(1)
    }

    fn layer_id(&self) -> LayerId {
        LayerId::Memory
    }

    fn layer_name(&self) -> &'static str {
        "Memory Layer (Stub)"
    }

    async fn health_check(&self) -> CoreResult<bool> {
        Ok(true)
    }
}

// =============================================================================
// 4. StubLearningLayer - 10ms latency budget
// =============================================================================

/// Stub implementation of the Learning layer.
///
/// The Learning layer handles UTL-driven weight optimization.
/// This stub simulates learning computations.
///
/// # Latency Budget
/// Real implementation: 10ms max
/// Stub implementation: <1us (instant return)
#[derive(Debug, Clone, Default)]
pub struct StubLearningLayer {
    _config: StubLayerConfig,
}

impl StubLearningLayer {
    /// Create a new stub learning layer.
    pub fn new() -> Self {
        Self {
            _config: StubLayerConfig::default(),
        }
    }
}

#[async_trait]
impl NervousLayer for StubLearningLayer {
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput> {
        let input_hash = compute_input_hash(&input.content);

        // Learning layer can have higher entropy (exploring new patterns)
        let entropy = 0.4 + (input_hash % 50) as f32 / 200.0; // Range: 0.4-0.65
        let coherence = 0.55 + (input_hash % 45) as f32 / 200.0; // Range: 0.55-0.775

        // Deterministic UTL score simulation
        let utl_score = 0.6 + (input_hash % 40) as f32 / 100.0; // Range: 0.6-1.0 (capped)
        let utl_score = utl_score.min(1.0);

        let result = LayerResult::success(
            LayerId::Learning,
            json!({
                "utl_score": utl_score,
                "learning_applied": true,
                "weights_updated": (input_hash % 10) + 1,
                "gradient_norm": 0.05,
                "convergence_metric": 0.92
            }),
        );

        // Learning might suggest exploration if UTL score is high
        let action = if utl_score > 0.8 {
            SuggestedAction::Explore
        } else {
            SuggestedAction::Continue
        };

        Ok(LayerOutput {
            layer: LayerId::Learning,
            result,
            pulse: CognitivePulse::new(entropy, coherence, action),
            duration_us: 2000, // 2ms stub value, well under 10ms budget
        })
    }

    fn latency_budget(&self) -> Duration {
        Duration::from_millis(10)
    }

    fn layer_id(&self) -> LayerId {
        LayerId::Learning
    }

    fn layer_name(&self) -> &'static str {
        "Learning Layer (Stub)"
    }

    async fn health_check(&self) -> CoreResult<bool> {
        Ok(true)
    }
}

// =============================================================================
// 5. StubCoherenceLayer - 10ms latency budget
// =============================================================================

/// Stub implementation of the Coherence layer.
///
/// The Coherence layer handles global state synchronization.
/// This stub simulates coherence computations.
///
/// # Latency Budget
/// Real implementation: 10ms max
/// Stub implementation: <1us (instant return)
#[derive(Debug, Clone, Default)]
pub struct StubCoherenceLayer {
    _config: StubLayerConfig,
}

impl StubCoherenceLayer {
    /// Create a new stub coherence layer.
    pub fn new() -> Self {
        Self {
            _config: StubLayerConfig::default(),
        }
    }
}

#[async_trait]
impl NervousLayer for StubCoherenceLayer {
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput> {
        let input_hash = compute_input_hash(&input.content);

        // Coherence layer should report high coherence (its job is to improve it)
        let entropy = 0.25 + (input_hash % 35) as f32 / 200.0; // Range: 0.25-0.425
        let coherence = 0.75 + (input_hash % 25) as f32 / 200.0; // Range: 0.75-0.875

        // Global coherence score
        let global_coherence = 0.8 + (input_hash % 20) as f32 / 100.0;
        let global_coherence = global_coherence.min(1.0);

        let result = LayerResult::success(
            LayerId::Coherence,
            json!({
                "global_coherence": global_coherence,
                "state_synchronized": true,
                "conflicts_resolved": (input_hash % 3) as u32,
                "coherence_delta": 0.05,
                "integration_complete": true
            }),
        );

        // Coherence layer typically signals ready state
        let action = if global_coherence > 0.85 {
            SuggestedAction::Ready
        } else {
            SuggestedAction::Continue
        };

        Ok(LayerOutput {
            layer: LayerId::Coherence,
            result,
            pulse: CognitivePulse::new(entropy, coherence, action),
            duration_us: 1500, // 1.5ms stub value, well under 10ms budget
        })
    }

    fn latency_budget(&self) -> Duration {
        Duration::from_millis(10)
    }

    fn layer_id(&self) -> LayerId {
        LayerId::Coherence
    }

    fn layer_name(&self) -> &'static str {
        "Coherence Layer (Stub)"
    }

    async fn health_check(&self) -> CoreResult<bool> {
        Ok(true)
    }
}

// =============================================================================
// Helper types and functions
// =============================================================================

/// Stub layer configuration (placeholder for future extension).
#[derive(Debug, Clone, Default)]
struct StubLayerConfig {
    // Reserved for future configuration options
}

/// Compute a deterministic hash from input string for reproducible stub behavior.
/// Same input always produces same output.
fn compute_input_hash(input: &str) -> u64 {
    // Simple deterministic hash function
    let mut hash: u64 = 0;
    for (i, byte) in input.bytes().enumerate() {
        hash = hash.wrapping_add((byte as u64).wrapping_mul((i as u64).wrapping_add(1)));
    }
    hash
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create test input
    fn test_input(content: &str) -> LayerInput {
        LayerInput::new("test-request-123".to_string(), content.to_string())
    }

    // -------------------------------------------------------------------------
    // StubSensingLayer Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_sensing_layer_output_within_budget() {
        let layer = StubSensingLayer::new();
        let input = test_input("test sensing input");

        let output = layer.process(input).await.expect("process should succeed");

        // Verify output is within latency budget
        let budget_us = layer.latency_budget().as_micros() as u64;
        assert!(
            output.duration_us < budget_us,
            "duration {}us should be < budget {}us",
            output.duration_us,
            budget_us
        );

        // Verify correct layer ID
        assert_eq!(output.layer, LayerId::Sensing);
        assert!(output.result.success);
    }

    #[tokio::test]
    async fn test_sensing_layer_determinism() {
        let layer = StubSensingLayer::new();
        let input1 = test_input("same input");
        let input2 = test_input("same input");

        let output1 = layer.process(input1).await.unwrap();
        let output2 = layer.process(input2).await.unwrap();

        // Same input should produce same pulse values
        assert_eq!(output1.pulse.entropy, output2.pulse.entropy);
        assert_eq!(output1.pulse.coherence, output2.pulse.coherence);
    }

    #[tokio::test]
    async fn test_sensing_layer_health_check() {
        let layer = StubSensingLayer::new();
        assert!(layer.health_check().await.unwrap());
    }

    #[test]
    fn test_sensing_layer_properties() {
        let layer = StubSensingLayer::new();
        assert_eq!(layer.layer_id(), LayerId::Sensing);
        assert_eq!(layer.latency_budget(), Duration::from_millis(5));
        assert!(layer.layer_name().contains("Sensing"));
    }

    // -------------------------------------------------------------------------
    // StubReflexLayer Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_reflex_layer_output_within_budget() {
        let layer = StubReflexLayer::new();
        let input = test_input("reflex pattern input");

        let output = layer.process(input).await.expect("process should succeed");

        let budget_us = layer.latency_budget().as_micros() as u64;
        assert!(
            output.duration_us < budget_us,
            "duration {}us should be < budget {}us",
            output.duration_us,
            budget_us
        );

        assert_eq!(output.layer, LayerId::Reflex);
        assert!(output.result.success);
    }

    #[tokio::test]
    async fn test_reflex_layer_determinism() {
        let layer = StubReflexLayer::new();
        let input1 = test_input("reflex test");
        let input2 = test_input("reflex test");

        let output1 = layer.process(input1).await.unwrap();
        let output2 = layer.process(input2).await.unwrap();

        assert_eq!(output1.pulse.entropy, output2.pulse.entropy);
        assert_eq!(output1.pulse.coherence, output2.pulse.coherence);
    }

    #[tokio::test]
    async fn test_reflex_layer_health_check() {
        let layer = StubReflexLayer::new();
        assert!(layer.health_check().await.unwrap());
    }

    #[test]
    fn test_reflex_layer_properties() {
        let layer = StubReflexLayer::new();
        assert_eq!(layer.layer_id(), LayerId::Reflex);
        assert_eq!(layer.latency_budget(), Duration::from_micros(100));
        assert!(layer.layer_name().contains("Reflex"));
    }

    // -------------------------------------------------------------------------
    // StubMemoryLayer Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_memory_layer_output_within_budget() {
        let layer = StubMemoryLayer::new();
        let input = test_input("memory retrieval query");

        let output = layer.process(input).await.expect("process should succeed");

        let budget_us = layer.latency_budget().as_micros() as u64;
        assert!(
            output.duration_us < budget_us,
            "duration {}us should be < budget {}us",
            output.duration_us,
            budget_us
        );

        assert_eq!(output.layer, LayerId::Memory);
        assert!(output.result.success);
    }

    #[tokio::test]
    async fn test_memory_layer_determinism() {
        let layer = StubMemoryLayer::new();
        let input1 = test_input("memory test");
        let input2 = test_input("memory test");

        let output1 = layer.process(input1).await.unwrap();
        let output2 = layer.process(input2).await.unwrap();

        assert_eq!(output1.pulse.entropy, output2.pulse.entropy);
        assert_eq!(output1.pulse.coherence, output2.pulse.coherence);
    }

    #[tokio::test]
    async fn test_memory_layer_health_check() {
        let layer = StubMemoryLayer::new();
        assert!(layer.health_check().await.unwrap());
    }

    #[test]
    fn test_memory_layer_properties() {
        let layer = StubMemoryLayer::new();
        assert_eq!(layer.layer_id(), LayerId::Memory);
        assert_eq!(layer.latency_budget(), Duration::from_millis(1));
        assert!(layer.layer_name().contains("Memory"));
    }

    // -------------------------------------------------------------------------
    // StubLearningLayer Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_learning_layer_output_within_budget() {
        let layer = StubLearningLayer::new();
        let input = test_input("learning optimization task");

        let output = layer.process(input).await.expect("process should succeed");

        let budget_us = layer.latency_budget().as_micros() as u64;
        assert!(
            output.duration_us < budget_us,
            "duration {}us should be < budget {}us",
            output.duration_us,
            budget_us
        );

        assert_eq!(output.layer, LayerId::Learning);
        assert!(output.result.success);
    }

    #[tokio::test]
    async fn test_learning_layer_determinism() {
        let layer = StubLearningLayer::new();
        let input1 = test_input("learning test");
        let input2 = test_input("learning test");

        let output1 = layer.process(input1).await.unwrap();
        let output2 = layer.process(input2).await.unwrap();

        assert_eq!(output1.pulse.entropy, output2.pulse.entropy);
        assert_eq!(output1.pulse.coherence, output2.pulse.coherence);
    }

    #[tokio::test]
    async fn test_learning_layer_health_check() {
        let layer = StubLearningLayer::new();
        assert!(layer.health_check().await.unwrap());
    }

    #[test]
    fn test_learning_layer_properties() {
        let layer = StubLearningLayer::new();
        assert_eq!(layer.layer_id(), LayerId::Learning);
        assert_eq!(layer.latency_budget(), Duration::from_millis(10));
        assert!(layer.layer_name().contains("Learning"));
    }

    // -------------------------------------------------------------------------
    // StubCoherenceLayer Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_coherence_layer_output_within_budget() {
        let layer = StubCoherenceLayer::new();
        let input = test_input("coherence synchronization");

        let output = layer.process(input).await.expect("process should succeed");

        let budget_us = layer.latency_budget().as_micros() as u64;
        assert!(
            output.duration_us < budget_us,
            "duration {}us should be < budget {}us",
            output.duration_us,
            budget_us
        );

        assert_eq!(output.layer, LayerId::Coherence);
        assert!(output.result.success);
    }

    #[tokio::test]
    async fn test_coherence_layer_determinism() {
        let layer = StubCoherenceLayer::new();
        let input1 = test_input("coherence test");
        let input2 = test_input("coherence test");

        let output1 = layer.process(input1).await.unwrap();
        let output2 = layer.process(input2).await.unwrap();

        assert_eq!(output1.pulse.entropy, output2.pulse.entropy);
        assert_eq!(output1.pulse.coherence, output2.pulse.coherence);
    }

    #[tokio::test]
    async fn test_coherence_layer_health_check() {
        let layer = StubCoherenceLayer::new();
        assert!(layer.health_check().await.unwrap());
    }

    #[test]
    fn test_coherence_layer_properties() {
        let layer = StubCoherenceLayer::new();
        assert_eq!(layer.layer_id(), LayerId::Coherence);
        assert_eq!(layer.latency_budget(), Duration::from_millis(10));
        assert!(layer.layer_name().contains("Coherence"));
    }

    // -------------------------------------------------------------------------
    // Cross-layer Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_all_layers_healthy() {
        let sensing = StubSensingLayer::new();
        let reflex = StubReflexLayer::new();
        let memory = StubMemoryLayer::new();
        let learning = StubLearningLayer::new();
        let coherence = StubCoherenceLayer::new();

        assert!(sensing.health_check().await.unwrap());
        assert!(reflex.health_check().await.unwrap());
        assert!(memory.health_check().await.unwrap());
        assert!(learning.health_check().await.unwrap());
        assert!(coherence.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_pipeline_processing() {
        // Test that layers can be chained (context accumulates)
        let sensing = StubSensingLayer::new();
        let reflex = StubReflexLayer::new();
        let memory = StubMemoryLayer::new();
        let learning = StubLearningLayer::new();
        let coherence = StubCoherenceLayer::new();

        let mut input = test_input("pipeline test");

        let out1 = sensing.process(input.clone()).await.unwrap();
        input
            .context
            .add_latency(std::time::Duration::from_micros(out1.duration_us));
        input.context.layer_results.push(out1.result.clone());

        let out2 = reflex.process(input.clone()).await.unwrap();
        input
            .context
            .add_latency(std::time::Duration::from_micros(out2.duration_us));
        input.context.layer_results.push(out2.result.clone());

        let out3 = memory.process(input.clone()).await.unwrap();
        input
            .context
            .add_latency(std::time::Duration::from_micros(out3.duration_us));
        input.context.layer_results.push(out3.result.clone());

        let out4 = learning.process(input.clone()).await.unwrap();
        input
            .context
            .add_latency(std::time::Duration::from_micros(out4.duration_us));
        input.context.layer_results.push(out4.result.clone());

        let out5 = coherence.process(input.clone()).await.unwrap();
        input
            .context
            .add_latency(std::time::Duration::from_micros(out5.duration_us));
        input.context.layer_results.push(out5.result.clone());

        // All layers should have run successfully
        assert_eq!(input.context.layer_results.len(), 5);
        for result in &input.context.layer_results {
            assert!(result.success);
        }
    }

    #[test]
    fn test_compute_input_hash_determinism() {
        let hash1 = compute_input_hash("test string");
        let hash2 = compute_input_hash("test string");
        let hash3 = compute_input_hash("different string");

        assert_eq!(hash1, hash2, "Same input should produce same hash");
        assert_ne!(
            hash1, hash3,
            "Different input should produce different hash"
        );
    }

    // =========================================================================
    // TC-GHOST-007: Layer Latency Budget Enforcement Tests
    // =========================================================================

    #[tokio::test]
    async fn test_layer_execution_within_latency_budget_sensing() {
        // TC-GHOST-007: Sensing layer must complete within its 5ms latency budget
        use std::time::Instant;

        let layer = StubSensingLayer::new();
        let input = test_input("test content for sensing layer latency budget");

        let budget = layer.latency_budget();
        let start = Instant::now();
        let result = layer.process(input).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Sensing layer must process successfully");
        assert!(
            elapsed <= budget,
            "Sensing layer took {:?} but budget is {:?}",
            elapsed, budget
        );
    }

    #[tokio::test]
    async fn test_layer_execution_within_latency_budget_reflex() {
        // TC-GHOST-007: Reflex layer must complete within its 100us latency budget
        use std::time::Instant;

        let layer = StubReflexLayer::new();
        let input = test_input("test content for reflex layer latency budget");

        let budget = layer.latency_budget();
        let start = Instant::now();
        let result = layer.process(input).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Reflex layer must process successfully");
        assert!(
            elapsed <= budget,
            "Reflex layer took {:?} but budget is {:?}",
            elapsed, budget
        );
    }

    #[tokio::test]
    async fn test_layer_execution_within_latency_budget_memory() {
        // TC-GHOST-007: Memory layer must complete within its 1ms latency budget
        use std::time::Instant;

        let layer = StubMemoryLayer::new();
        let input = test_input("test content for memory layer latency budget");

        let budget = layer.latency_budget();
        let start = Instant::now();
        let result = layer.process(input).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Memory layer must process successfully");
        assert!(
            elapsed <= budget,
            "Memory layer took {:?} but budget is {:?}",
            elapsed, budget
        );
    }

    #[tokio::test]
    async fn test_layer_execution_within_latency_budget_learning() {
        // TC-GHOST-007: Learning layer must complete within its 10ms latency budget
        use std::time::Instant;

        let layer = StubLearningLayer::new();
        let input = test_input("test content for learning layer latency budget");

        let budget = layer.latency_budget();
        let start = Instant::now();
        let result = layer.process(input).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Learning layer must process successfully");
        assert!(
            elapsed <= budget,
            "Learning layer took {:?} but budget is {:?}",
            elapsed, budget
        );
    }

    #[tokio::test]
    async fn test_layer_execution_within_latency_budget_coherence() {
        // TC-GHOST-007: Coherence layer must complete within its 10ms latency budget
        use std::time::Instant;

        let layer = StubCoherenceLayer::new();
        let input = test_input("test content for coherence layer latency budget");

        let budget = layer.latency_budget();
        let start = Instant::now();
        let result = layer.process(input).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Coherence layer must process successfully");
        assert!(
            elapsed <= budget,
            "Coherence layer took {:?} but budget is {:?}",
            elapsed, budget
        );
    }

    #[tokio::test]
    async fn test_all_layers_within_latency_budget() {
        // TC-GHOST-007: All layers must complete within their respective budgets
        use std::time::Instant;

        let layers: Vec<(Box<dyn NervousLayer>, &str)> = vec![
            (Box::new(StubSensingLayer::new()), "Sensing"),
            (Box::new(StubReflexLayer::new()), "Reflex"),
            (Box::new(StubMemoryLayer::new()), "Memory"),
            (Box::new(StubLearningLayer::new()), "Learning"),
            (Box::new(StubCoherenceLayer::new()), "Coherence"),
        ];

        for (layer, name) in layers {
            let input = LayerInput::new("test-request".to_string(), format!("test content for {}", name));

            let budget = layer.latency_budget();
            let start = Instant::now();
            let result = layer.process(input).await;
            let elapsed = start.elapsed();

            assert!(
                result.is_ok(),
                "{} layer must process successfully",
                name
            );
            assert!(
                elapsed <= budget,
                "{} layer took {:?} but budget is {:?}",
                name, elapsed, budget
            );
        }
    }

    #[tokio::test]
    async fn test_layer_reported_duration_within_budget() {
        // TC-GHOST-007: Layer's reported duration_us must be within budget
        let layers: Vec<Box<dyn NervousLayer>> = vec![
            Box::new(StubSensingLayer::new()),
            Box::new(StubReflexLayer::new()),
            Box::new(StubMemoryLayer::new()),
            Box::new(StubLearningLayer::new()),
            Box::new(StubCoherenceLayer::new()),
        ];

        for layer in layers {
            let input = LayerInput::new("test-request".to_string(), "test content".to_string());
            let output = layer.process(input).await.unwrap();

            let budget_us = layer.latency_budget().as_micros() as u64;
            assert!(
                output.duration_us < budget_us,
                "{} reported duration {}us exceeds budget {}us",
                layer.layer_name(), output.duration_us, budget_us
            );
        }
    }

    #[tokio::test]
    async fn test_layer_output_coherence_ranges() {
        // TC-GHOST-007: Layer outputs must have entropy/coherence in valid [0.0, 1.0] range
        let layers: Vec<Box<dyn NervousLayer>> = vec![
            Box::new(StubSensingLayer::new()),
            Box::new(StubReflexLayer::new()),
            Box::new(StubMemoryLayer::new()),
            Box::new(StubLearningLayer::new()),
            Box::new(StubCoherenceLayer::new()),
        ];

        for layer in layers {
            let input = LayerInput::new("test-request".to_string(), "test content for range validation".to_string());
            let output = layer.process(input).await.unwrap();

            assert!(
                output.pulse.entropy >= 0.0 && output.pulse.entropy <= 1.0,
                "{} entropy {} must be in [0.0, 1.0]",
                layer.layer_name(), output.pulse.entropy
            );
            assert!(
                output.pulse.coherence >= 0.0 && output.pulse.coherence <= 1.0,
                "{} coherence {} must be in [0.0, 1.0]",
                layer.layer_name(), output.pulse.coherence
            );
        }
    }

    #[tokio::test]
    async fn test_layer_output_correct_layer_id() {
        // TC-GHOST-007: Each layer must report its correct LayerId in output
        let test_cases = vec![
            (Box::new(StubSensingLayer::new()) as Box<dyn NervousLayer>, LayerId::Sensing),
            (Box::new(StubReflexLayer::new()), LayerId::Reflex),
            (Box::new(StubMemoryLayer::new()), LayerId::Memory),
            (Box::new(StubLearningLayer::new()), LayerId::Learning),
            (Box::new(StubCoherenceLayer::new()), LayerId::Coherence),
        ];

        for (layer, expected_id) in test_cases {
            let input = LayerInput::new("test-request".to_string(), "test content".to_string());
            let output = layer.process(input).await.unwrap();

            assert_eq!(
                output.layer, expected_id,
                "{} must report correct LayerId {:?}, got {:?}",
                layer.layer_name(), expected_id, output.layer
            );
            assert_eq!(
                output.result.layer, expected_id,
                "{} result must report correct LayerId {:?}, got {:?}",
                layer.layer_name(), expected_id, output.result.layer
            );
        }
    }

    #[tokio::test]
    async fn test_full_pipeline_within_total_budget() {
        // TC-GHOST-007: Full pipeline must complete within sum of all budgets (26.1ms)
        use std::time::Instant;

        let sensing = StubSensingLayer::new();
        let reflex = StubReflexLayer::new();
        let memory = StubMemoryLayer::new();
        let learning = StubLearningLayer::new();
        let coherence = StubCoherenceLayer::new();

        // Total budget: 5ms + 100us + 1ms + 10ms + 10ms = 26.1ms
        let total_budget = Duration::from_micros(26100);

        let input = test_input("full pipeline test content");
        let start = Instant::now();

        let _ = sensing.process(input.clone()).await.unwrap();
        let _ = reflex.process(input.clone()).await.unwrap();
        let _ = memory.process(input.clone()).await.unwrap();
        let _ = learning.process(input.clone()).await.unwrap();
        let _ = coherence.process(input).await.unwrap();

        let elapsed = start.elapsed();

        assert!(
            elapsed <= total_budget,
            "Full pipeline took {:?} but total budget is {:?}",
            elapsed, total_budget
        );
    }
}
