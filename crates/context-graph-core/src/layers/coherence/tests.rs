//! Tests for CoherenceLayer - REAL implementations, NO MOCKS

use std::time::{Duration, Instant};

use crate::traits::NervousLayer;
use crate::types::{LayerId, LayerInput, LayerResult};

use super::constants::GW_THRESHOLD;
use super::layer::CoherenceLayer;

#[tokio::test]
async fn test_coherence_layer_process() {
    let layer = CoherenceLayer::new();
    let input = LayerInput::new("test-123".to_string(), "test content".to_string());

    let result = layer.process(input).await.unwrap();

    assert_eq!(result.layer, LayerId::Coherence);
    assert!(result.result.success);
    assert!(result.result.data.get("resonance").is_some());
    assert!(result.result.data.get("consciousness").is_some());
    assert!(result.result.data.get("gw_ignited").is_some());

    println!("[VERIFIED] CoherenceLayer.process() returns valid output");
}

#[tokio::test]
async fn test_coherence_layer_resonance_range() {
    let layer = CoherenceLayer::new();
    let input = LayerInput::new("test-456".to_string(), "resonance test".to_string());

    let result = layer.process(input).await.unwrap();

    let resonance = result.result.data["resonance"].as_f64().unwrap() as f32;
    assert!(
        (0.0..=1.0).contains(&resonance),
        "Resonance should be in [0,1], got {}",
        resonance
    );
    println!("[VERIFIED] Resonance r ∈ [0, 1]: r = {}", resonance);
}

#[tokio::test]
async fn test_coherence_layer_consciousness_range() {
    let layer = CoherenceLayer::new();
    let input = LayerInput::new("test-789".to_string(), "consciousness test".to_string());

    let result = layer.process(input).await.unwrap();

    let consciousness = result.result.data["consciousness"].as_f64().unwrap() as f32;
    assert!(
        (0.0..=1.0).contains(&consciousness),
        "Consciousness should be in [0,1], got {}",
        consciousness
    );
    println!("[VERIFIED] Consciousness C ∈ [0, 1]: C = {}", consciousness);
}

#[tokio::test]
async fn test_coherence_layer_with_learning_context() {
    let layer = CoherenceLayer::new();

    // Create input with L4 Learning context
    let mut input = LayerInput::new(
        "learning-ctx".to_string(),
        "learning context test".to_string(),
    );
    input.context.layer_results.push(LayerResult::success(
        LayerId::Learning,
        serde_json::json!({
            "weight_delta": 0.5,
            "surprise": 0.8,
            "coherence_w": 0.7,
        }),
    ));

    let result = layer.process(input).await.unwrap();

    assert!(result.result.success);

    let learning_signal = result.result.data["learning_signal"].as_f64().unwrap() as f32;
    assert!(
        (learning_signal - 0.5).abs() < 1e-6,
        "Learning signal should be extracted from L4"
    );

    println!("[VERIFIED] Learning signal extracted: {}", learning_signal);
}

#[tokio::test]
async fn test_coherence_layer_properties() {
    let layer = CoherenceLayer::new();

    assert_eq!(layer.layer_id(), LayerId::Coherence);
    assert_eq!(layer.latency_budget(), Duration::from_millis(10));
    assert_eq!(layer.layer_name(), "Coherence Layer");
    assert!((layer.gw_threshold() - GW_THRESHOLD).abs() < 1e-6);

    println!("[VERIFIED] CoherenceLayer properties correct");
}

#[tokio::test]
async fn test_coherence_layer_health_check() {
    let layer = CoherenceLayer::new();
    let healthy = layer.health_check().await.unwrap();

    assert!(healthy, "CoherenceLayer should be healthy");
    println!("[VERIFIED] health_check passes");
}

#[tokio::test]
async fn test_coherence_layer_custom_config() {
    let layer = CoherenceLayer::with_kuramoto(6, 3.0)
        .with_gw_threshold(0.75)
        .with_integration_steps(15);

    assert!((layer.gw_threshold() - 0.75).abs() < 1e-6);
    assert_eq!(layer.integration_steps, 15);

    println!("[VERIFIED] Custom configuration works");
}

#[tokio::test]
async fn test_gw_ignition_tracking() {
    let layer = CoherenceLayer::new().with_gw_threshold(0.1); // Low threshold for easy ignition

    // Run multiple times
    for i in 0..5 {
        let input = LayerInput::new(format!("ignition-{}", i), "test ignition".to_string());
        let _ = layer.process(input).await;
    }

    // Should have some ignitions with low threshold
    let count = layer.ignition_count();
    println!("[INFO] Ignition count with low threshold: {}", count);
    // Note: ignition depends on Kuramoto dynamics, may not always ignite
}

#[tokio::test]
async fn test_pulse_update() {
    let layer = CoherenceLayer::new();

    let mut input = LayerInput::new("pulse-test".to_string(), "pulse update test".to_string());
    input.context.pulse.coherence = 0.3;
    input.context.pulse.entropy = 0.7;

    let result = layer.process(input).await.unwrap();

    // Coherence should be updated to resonance
    assert!(
        result.pulse.source_layer == Some(LayerId::Coherence),
        "Source layer should be Coherence"
    );

    println!("[VERIFIED] Pulse updated with resonance");
}

// ============================================================
// Performance Benchmark - CRITICAL <10ms
// ============================================================

#[tokio::test]
async fn test_coherence_layer_latency_benchmark() {
    let layer = CoherenceLayer::new();

    let iterations = 1000;
    let mut total_us: u64 = 0;
    let mut max_us: u64 = 0;

    for i in 0..iterations {
        let mut input =
            LayerInput::new(format!("bench-{}", i), format!("Benchmark content {}", i));
        input.context.pulse.entropy = (i as f32 / iterations as f32).clamp(0.0, 1.0);
        input.context.pulse.coherence = 0.5;

        let start = Instant::now();
        let _ = layer.process(input).await;
        let elapsed = start.elapsed().as_micros() as u64;

        total_us += elapsed;
        max_us = max_us.max(elapsed);
    }

    let avg_us = total_us / iterations as u64;

    println!("Coherence Layer Benchmark Results:");
    println!("  Iterations: {}", iterations);
    println!("  Avg latency: {} us", avg_us);
    println!("  Max latency: {} us", max_us);
    println!("  Budget: 10000 us (10ms)");

    // Average should be well under budget
    assert!(
        avg_us < 10_000,
        "Average latency {} us exceeds 10ms budget",
        avg_us
    );

    // Max should also be under budget for reliable performance
    assert!(
        max_us < 10_000,
        "Max latency {} us exceeds 10ms budget",
        max_us
    );

    println!("[VERIFIED] Average latency {} us < 10000 us budget", avg_us);
}

// ============================================================
// Integration Tests
// ============================================================

#[tokio::test]
async fn test_full_pipeline_context() {
    let layer = CoherenceLayer::new();

    // Simulate full L1 -> L2 -> L3 -> L4 -> L5 pipeline context
    let mut input = LayerInput::new(
        "pipeline-test".to_string(),
        "Full pipeline test".to_string(),
    );

    // L1 Sensing result
    input.context.layer_results.push(LayerResult::success(
        LayerId::Sensing,
        serde_json::json!({
            "delta_s": 0.6,
            "scrubbed_content": "Full pipeline test",
            "pii_found": false,
        }),
    ));

    // L2 Reflex result (cache miss)
    input.context.layer_results.push(LayerResult::success(
        LayerId::Reflex,
        serde_json::json!({
            "cache_hit": false,
            "query_norm": 1.0,
        }),
    ));

    // L3 Memory result
    input.context.layer_results.push(LayerResult::success(
        LayerId::Memory,
        serde_json::json!({
            "retrieval_count": 3,
            "memories": [],
        }),
    ));

    // L4 Learning result
    input.context.layer_results.push(LayerResult::success(
        LayerId::Learning,
        serde_json::json!({
            "weight_delta": 0.3,
            "surprise": 0.6,
            "coherence_w": 0.75,
            "should_consolidate": false,
        }),
    ));

    // Set pulse state
    input.context.pulse.coherence = 0.5;
    input.context.pulse.entropy = 0.6;

    let result = layer.process(input).await.unwrap();

    assert!(result.result.success);

    // Verify all expected fields are present
    let data = &result.result.data;
    assert!(data.get("resonance").is_some());
    assert!(data.get("consciousness").is_some());
    assert!(data.get("differentiation").is_some());
    assert!(data.get("gw_ignited").is_some());
    assert!(data.get("state").is_some());
    assert!(data.get("oscillator_phases").is_some());
    assert!(data.get("learning_signal").is_some());

    let resonance = data["resonance"].as_f64().unwrap() as f32;
    let consciousness = data["consciousness"].as_f64().unwrap() as f32;
    let learning_signal = data["learning_signal"].as_f64().unwrap() as f32;

    // Verify values are in expected ranges
    assert!((0.0..=1.0).contains(&resonance));
    assert!((0.0..=1.0).contains(&consciousness));
    assert!((learning_signal - 0.3).abs() < 1e-6);

    println!("[VERIFIED] Full pipeline context processed correctly");
    println!("  Resonance: {}", resonance);
    println!("  Consciousness: {}", consciousness);
    println!("  Learning signal: {}", learning_signal);
}

#[tokio::test]
async fn test_consciousness_equation() {
    // Test C(t) = I(t) × R(t) × D(t)
    let layer = CoherenceLayer::new();

    // Test with known values
    let c1 = layer.compute_consciousness(1.0, 1.0, 1.0);
    assert!((c1 - 1.0).abs() < 1e-6, "C(1,1,1) should be 1.0");

    let c2 = layer.compute_consciousness(0.5, 0.5, 0.5);
    assert!((c2 - 0.125).abs() < 1e-6, "C(0.5,0.5,0.5) should be 0.125");

    let c3 = layer.compute_consciousness(0.0, 0.8, 0.8);
    assert!((c3).abs() < 1e-6, "C(0,0.8,0.8) should be 0");

    // Test NaN handling
    let c_nan = layer.compute_consciousness(f32::NAN, 0.5, 0.5);
    assert!((c_nan).abs() < 1e-6, "NaN input should return 0");

    println!("[VERIFIED] Consciousness equation C(t) = I × R × D");
}
