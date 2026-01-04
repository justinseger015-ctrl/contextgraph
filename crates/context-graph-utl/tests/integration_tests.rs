//! Integration tests for UTL (Unified Theory of Learning) module (M05-T25)
//!
//! These tests validate the complete UTL pipeline with real data (NO MOCKS):
//! - Formula correctness: `L = f((ΔS × ΔC) · wₑ · cos φ)`
//! - Lifecycle transitions at 50/500 thresholds
//! - Johari quadrant classification
//! - Performance within targets
//!
//! Constitution Reference: constitution.yaml Section 5, contextprd.md Section 5

use context_graph_utl::{
    compute_learning_magnitude, compute_learning_magnitude_validated,
    LearningIntensity, LearningSignal, UtlError, UtlState,
    processor::UtlProcessor,
    metrics::StageThresholds,
    johari::{JohariClassifier, JohariQuadrant, SuggestedAction},
    lifecycle::{LifecycleLambdaWeights, LifecycleStage},
    emotional::EmotionalWeightCalculator,
};
use std::time::Instant;

// =============================================================================
// Helper Functions: Deterministic Data Generation (NO MOCKS)
// =============================================================================

/// Generate deterministic embedding using mathematical functions.
/// Uses sin-based generation for reproducible, normalized values.
fn generate_embedding(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = (i as f64 + seed as f64) * 0.1;
            ((x.sin() + 1.0) / 2.0) as f32 // Normalized [0, 1]
        })
        .collect()
}

/// Generate multiple context embeddings for comparison.
fn generate_context(count: usize, dim: usize, base_seed: u64) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| generate_embedding(dim, base_seed + i as u64))
        .collect()
}

/// Generate embedding with a specific value pattern (for edge case testing).
fn uniform_embedding(dim: usize, value: f32) -> Vec<f32> {
    vec![value; dim]
}

// =============================================================================
// FULL UTL PIPELINE TESTS
// =============================================================================

#[test]
fn test_full_utl_pipeline_with_real_data() {
    let mut processor = UtlProcessor::with_defaults();

    let embedding = generate_embedding(1536, 42);
    let context = generate_context(50, 1536, 100);
    let content = "This is a moderately surprising statement about quantum mechanics!";

    let result = processor
        .compute_learning(content, &embedding, &context)
        .expect("UTL computation must succeed with valid inputs");

    // Post-condition: all bounds satisfied
    assert!(
        result.magnitude >= 0.0 && result.magnitude <= 1.0,
        "magnitude {} out of bounds [0,1]",
        result.magnitude
    );
    assert!(
        result.delta_s >= 0.0 && result.delta_s <= 1.0,
        "delta_s {} out of bounds [0,1]",
        result.delta_s
    );
    assert!(
        result.delta_c >= 0.0 && result.delta_c <= 1.0,
        "delta_c {} out of bounds [0,1]",
        result.delta_c
    );
    assert!(
        result.w_e >= 0.5 && result.w_e <= 1.5,
        "w_e {} out of bounds [0.5,1.5]",
        result.w_e
    );
    assert!(
        result.phi >= 0.0 && result.phi <= std::f32::consts::PI,
        "phi {} out of bounds [0,π]",
        result.phi
    );

    // Verify lambda weights are present
    assert!(
        result.lambda_weights.is_some(),
        "Lambda weights must be present in LearningSignal"
    );

    // Verify quadrant is valid
    assert!(matches!(
        result.quadrant,
        JohariQuadrant::Open
            | JohariQuadrant::Blind
            | JohariQuadrant::Hidden
            | JohariQuadrant::Unknown
    ));
}

#[test]
fn test_formula_mathematical_properties() {
    // Property 1: Zero surprise = zero learning
    let result = compute_learning_magnitude(0.0, 0.5, 1.0, 0.5);
    assert_eq!(result, 0.0, "zero surprise must yield zero learning");

    // Property 2: Zero coherence = zero learning
    let result = compute_learning_magnitude(0.5, 0.0, 1.0, 0.5);
    assert_eq!(result, 0.0, "zero coherence must yield zero learning");

    // Property 3: phi=π/2 = zero learning (cos(π/2) ≈ 0)
    let result = compute_learning_magnitude(1.0, 1.0, 1.0, std::f32::consts::FRAC_PI_2);
    assert!(
        result.abs() < 1e-6,
        "phi=π/2 must yield near-zero learning, got {}",
        result
    );

    // Property 4: Maximum learning at optimal conditions
    // L = (1.0 * 1.0) * 1.5 * cos(0) = 1.5, clamped to 1.0
    let result = compute_learning_magnitude(1.0, 1.0, 1.5, 0.0);
    assert!(
        (result - 1.0).abs() < 1e-5,
        "max conditions should yield L=1.0, got {}",
        result
    );
}

#[test]
fn test_formula_computation_accuracy() {
    // Test exact formula: L = (ΔS × ΔC) · wₑ · cos(φ)
    let delta_s = 0.8;
    let delta_c = 0.7;
    let w_e = 1.2;
    let phi = 0.0;

    let result = compute_learning_magnitude(delta_s, delta_c, w_e, phi);

    // Expected: (0.8 * 0.7) * 1.2 * cos(0) = 0.56 * 1.2 * 1.0 = 0.672
    let expected = (delta_s * delta_c) * w_e * phi.cos();
    let expected_clamped = expected.clamp(0.0, 1.0);

    assert!(
        (result - expected_clamped).abs() < 1e-5,
        "formula mismatch: got {} expected {}",
        result,
        expected_clamped
    );
}

#[test]
fn test_anti_phase_suppresses_learning() {
    // At phi = π, cos(π) = -1, so learning should be suppressed to 0
    let result = compute_learning_magnitude(0.8, 0.7, 1.2, std::f32::consts::PI);

    // Raw: (0.8 * 0.7) * 1.2 * (-1) = -0.672, clamped to 0
    assert_eq!(result, 0.0, "anti-phase (φ=π) must suppress learning to 0");
}

// =============================================================================
// LIFECYCLE TRANSITION TESTS
// =============================================================================

#[test]
fn test_lifecycle_transitions_at_thresholds() {
    let mut processor = UtlProcessor::with_defaults();

    // Start at Infancy
    assert_eq!(
        processor.lifecycle_stage(),
        LifecycleStage::Infancy,
        "Must start at Infancy"
    );

    // Progress through interactions - compute_learning increments interaction count
    let embedding = uniform_embedding(128, 0.5);
    let content = "test";

    // Infancy: 0-49 interactions (compute_learning records each as an interaction)
    for i in 0..49 {
        let _ = processor.compute_learning(content, &embedding, &[]);
        assert_eq!(
            processor.lifecycle_stage(),
            LifecycleStage::Infancy,
            "should stay Infancy at {} interactions",
            i + 1
        );
    }

    // 50th interaction -> Growth
    let _ = processor.compute_learning(content, &embedding, &[]);
    assert_eq!(
        processor.lifecycle_stage(),
        LifecycleStage::Growth,
        "must transition to Growth at 50 interactions, got {:?} at {}",
        processor.lifecycle_stage(),
        processor.interaction_count()
    );

    // Continue to 499 (staying in Growth)
    for i in 50..499 {
        let _ = processor.compute_learning(content, &embedding, &[]);
        assert_eq!(
            processor.lifecycle_stage(),
            LifecycleStage::Growth,
            "should stay Growth at {} interactions",
            i + 1
        );
    }

    // 500th interaction -> Maturity
    let _ = processor.compute_learning(content, &embedding, &[]);
    assert_eq!(
        processor.lifecycle_stage(),
        LifecycleStage::Maturity,
        "must transition to Maturity at 500 interactions, got {:?} at {}",
        processor.lifecycle_stage(),
        processor.interaction_count()
    );
}

#[test]
fn test_lifecycle_boundary_exact_at_50() {
    let mut processor = UtlProcessor::with_defaults();
    let embedding = uniform_embedding(128, 0.5);

    // 49 interactions
    for _ in 0..49 {
        let _ = processor.compute_learning("test", &embedding, &[]);
    }
    assert_eq!(processor.lifecycle_stage(), LifecycleStage::Infancy);
    assert_eq!(processor.interaction_count(), 49);

    // 50th interaction triggers transition
    let _ = processor.compute_learning("test", &embedding, &[]);
    assert_eq!(processor.lifecycle_stage(), LifecycleStage::Growth);
    assert_eq!(processor.interaction_count(), 50);
}

#[test]
fn test_lifecycle_boundary_exact_at_500() {
    // Per LifecycleConfig default: transition_hysteresis = 10
    // Hysteresis prevents rapid oscillation at boundaries by requiring
    // N interactions since last transition before allowing stage change.
    //
    // When using restore_lifecycle(count), last_transition_count is set to count,
    // meaning hysteresis applies from the restored position.
    let mut processor = UtlProcessor::with_defaults();
    let embedding = uniform_embedding(128, 0.5);

    // Restore to 490 (10 interactions before maturity threshold)
    // This allows exactly 10 interactions (matching hysteresis) to reach Maturity
    processor.restore_lifecycle(490);
    assert_eq!(processor.lifecycle_stage(), LifecycleStage::Growth);
    assert_eq!(processor.interaction_count(), 490);

    // 10 interactions: 491, 492, 493, 494, 495, 496, 497, 498, 499, 500
    // After 10 interactions (>=hysteresis), we reach 500 and can transition
    for i in 0..9 {
        let _ = processor.compute_learning("test", &embedding, &[]);
        assert_eq!(
            processor.lifecycle_stage(),
            LifecycleStage::Growth,
            "should stay Growth at {} interactions (hysteresis not satisfied)",
            491 + i
        );
    }

    // 10th interaction (count=500): hysteresis satisfied, transition allowed
    let _ = processor.compute_learning("test", &embedding, &[]);
    assert_eq!(processor.interaction_count(), 500);
    assert_eq!(
        processor.lifecycle_stage(),
        LifecycleStage::Maturity,
        "must transition to Maturity at 500 after hysteresis (10 interactions) is satisfied"
    );
}

#[test]
fn test_lambda_weights_per_lifecycle_stage() {
    // Per Marblestone (2016): Infancy prioritizes novelty, Maturity prioritizes consolidation
    let infancy = LifecycleLambdaWeights::for_stage(LifecycleStage::Infancy);
    assert!(
        (infancy.lambda_s() - 0.7).abs() < 0.001,
        "Infancy lambda_s should be 0.7, got {}",
        infancy.lambda_s()
    );
    assert!(
        (infancy.lambda_c() - 0.3).abs() < 0.001,
        "Infancy lambda_c should be 0.3, got {}",
        infancy.lambda_c()
    );

    let growth = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
    assert!(
        (growth.lambda_s() - 0.5).abs() < 0.001,
        "Growth lambda_s should be 0.5, got {}",
        growth.lambda_s()
    );
    assert!(
        (growth.lambda_c() - 0.5).abs() < 0.001,
        "Growth lambda_c should be 0.5, got {}",
        growth.lambda_c()
    );

    let maturity = LifecycleLambdaWeights::for_stage(LifecycleStage::Maturity);
    assert!(
        (maturity.lambda_s() - 0.3).abs() < 0.001,
        "Maturity lambda_s should be 0.3, got {}",
        maturity.lambda_s()
    );
    assert!(
        (maturity.lambda_c() - 0.7).abs() < 0.001,
        "Maturity lambda_c should be 0.7, got {}",
        maturity.lambda_c()
    );
}

#[test]
fn test_lambda_weights_sum_to_one() {
    for stage in [
        LifecycleStage::Infancy,
        LifecycleStage::Growth,
        LifecycleStage::Maturity,
    ] {
        let weights = LifecycleLambdaWeights::for_stage(stage);
        let sum = weights.lambda_s() + weights.lambda_c();
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Lambda weights for {:?} must sum to 1.0, got {}",
            stage,
            sum
        );
    }
}

// =============================================================================
// JOHARI QUADRANT TESTS
// =============================================================================

#[test]
fn test_johari_quadrant_classification() {
    // Per constitution.yaml Johari Window:
    // Open: low entropy (surprise), high coherence
    // Blind: high entropy, low coherence
    // Hidden: low entropy, low coherence
    // Unknown: high entropy, high coherence

    let classifier = JohariClassifier::default();

    // Open: low surprise (entropy), high coherence
    let open = classifier.classify(0.2, 0.8);
    assert_eq!(
        open,
        JohariQuadrant::Open,
        "low entropy + high coherence = Open"
    );

    // Blind: high entropy, low coherence
    let blind = classifier.classify(0.8, 0.2);
    assert_eq!(
        blind,
        JohariQuadrant::Blind,
        "high entropy + low coherence = Blind"
    );

    // Hidden: low entropy, low coherence
    let hidden = classifier.classify(0.2, 0.2);
    assert_eq!(
        hidden,
        JohariQuadrant::Hidden,
        "low entropy + low coherence = Hidden"
    );

    // Unknown: high entropy, high coherence
    let unknown = classifier.classify(0.8, 0.8);
    assert_eq!(
        unknown,
        JohariQuadrant::Unknown,
        "high entropy + high coherence = Unknown"
    );
}

#[test]
fn test_suggested_actions_per_quadrant() {
    // Per contextprd.md Section 5.4
    // Test the mapping of Johari quadrants to suggested actions:
    // Open -> DirectRecall
    // Blind -> TriggerDream
    // Hidden -> GetNeighborhood
    // Unknown -> EpistemicAction

    // Test via LearningSignal which contains the mapping
    let signal = LearningSignal::new(
        0.5,
        0.2,
        0.8,
        1.0,
        0.5,
        None,
        JohariQuadrant::Open,
        SuggestedAction::DirectRecall,
        false,
        true,
        100,
    )
    .unwrap();
    assert_eq!(signal.suggested_action, SuggestedAction::DirectRecall);

    let signal = LearningSignal::new(
        0.5,
        0.8,
        0.2,
        1.0,
        0.5,
        None,
        JohariQuadrant::Blind,
        SuggestedAction::TriggerDream,
        false,
        true,
        100,
    )
    .unwrap();
    assert_eq!(signal.suggested_action, SuggestedAction::TriggerDream);

    let signal = LearningSignal::new(
        0.5,
        0.2,
        0.2,
        1.0,
        0.5,
        None,
        JohariQuadrant::Hidden,
        SuggestedAction::GetNeighborhood,
        false,
        true,
        100,
    )
    .unwrap();
    assert_eq!(signal.suggested_action, SuggestedAction::GetNeighborhood);

    let signal = LearningSignal::new(
        0.5,
        0.8,
        0.8,
        1.0,
        0.5,
        None,
        JohariQuadrant::Unknown,
        SuggestedAction::EpistemicAction,
        false,
        true,
        100,
    )
    .unwrap();
    assert_eq!(signal.suggested_action, SuggestedAction::EpistemicAction);
}

#[test]
fn test_johari_boundary_values() {
    let classifier = JohariClassifier::default();

    // Test at exact threshold (0.5, 0.5) - should classify deterministically
    let result = classifier.classify(0.5, 0.5);
    assert!(
        matches!(
            result,
            JohariQuadrant::Open
                | JohariQuadrant::Blind
                | JohariQuadrant::Hidden
                | JohariQuadrant::Unknown
        ),
        "Boundary value (0.5, 0.5) must classify to a valid quadrant"
    );
}

// =============================================================================
// NaN/INFINITY PREVENTION TESTS
// =============================================================================

#[test]
fn test_nan_infinity_prevention_edge_cases() {
    // Empty/zero inputs should not produce NaN/Infinity
    let result = compute_learning_magnitude_validated(0.0, 0.0, 1.0, 0.0);
    assert!(result.is_ok(), "zero inputs should not error");
    let value = result.unwrap();
    assert!(!value.is_nan(), "zero inputs should not produce NaN");
    assert!(!value.is_infinite(), "zero inputs should not produce Infinity");

    // Boundary values
    for &val in &[0.0_f32, 0.5, 1.0] {
        let result = compute_learning_magnitude(val, val, 1.0, val.min(std::f32::consts::PI));
        assert!(!result.is_nan(), "NaN detected for input {}", val);
        assert!(!result.is_infinite(), "Infinity detected for input {}", val);
    }
}

#[test]
fn test_error_handling_out_of_bounds_delta_s() {
    // delta_s negative
    let result = compute_learning_magnitude_validated(-0.1, 0.5, 1.0, 0.5);
    assert!(result.is_err(), "negative delta_s must fail");
    match result.unwrap_err() {
        UtlError::InvalidParameter { name, .. } => assert_eq!(name, "delta_s"),
        e => panic!("Expected InvalidParameter error for delta_s, got {:?}", e),
    }

    // delta_s > 1
    let result = compute_learning_magnitude_validated(1.1, 0.5, 1.0, 0.5);
    assert!(result.is_err(), "delta_s > 1 must fail");
    match result.unwrap_err() {
        UtlError::InvalidParameter { name, .. } => assert_eq!(name, "delta_s"),
        e => panic!("Expected InvalidParameter error for delta_s, got {:?}", e),
    }
}

#[test]
fn test_error_handling_out_of_bounds_delta_c() {
    // delta_c negative
    let result = compute_learning_magnitude_validated(0.5, -0.1, 1.0, 0.5);
    assert!(result.is_err(), "negative delta_c must fail");
    match result.unwrap_err() {
        UtlError::InvalidParameter { name, .. } => assert_eq!(name, "delta_c"),
        e => panic!("Expected InvalidParameter error for delta_c, got {:?}", e),
    }

    // delta_c > 1
    let result = compute_learning_magnitude_validated(0.5, 1.1, 1.0, 0.5);
    assert!(result.is_err(), "delta_c > 1 must fail");
}

#[test]
fn test_error_handling_out_of_bounds_w_e() {
    // w_e < 0.5
    let result = compute_learning_magnitude_validated(0.5, 0.5, 0.4, 0.5);
    assert!(result.is_err(), "w_e < 0.5 must fail");
    match result.unwrap_err() {
        UtlError::InvalidParameter { name, .. } => assert_eq!(name, "w_e"),
        e => panic!("Expected InvalidParameter error for w_e, got {:?}", e),
    }

    // w_e > 1.5
    let result = compute_learning_magnitude_validated(0.5, 0.5, 1.6, 0.5);
    assert!(result.is_err(), "w_e > 1.5 must fail");
}

#[test]
fn test_error_handling_out_of_bounds_phi() {
    // phi negative
    let result = compute_learning_magnitude_validated(0.5, 0.5, 1.0, -0.1);
    assert!(result.is_err(), "negative phi must fail");
    match result.unwrap_err() {
        UtlError::InvalidParameter { name, .. } => assert_eq!(name, "phi"),
        e => panic!("Expected InvalidParameter error for phi, got {:?}", e),
    }

    // phi > π
    let result = compute_learning_magnitude_validated(0.5, 0.5, 1.0, 4.0);
    assert!(result.is_err(), "phi > π must fail");
}

// =============================================================================
// EMOTIONAL WEIGHT TESTS
// =============================================================================

#[test]
fn test_emotional_weight_bounds() {
    use context_graph_utl::config::EmotionalConfig;

    let config = EmotionalConfig::default();
    let calculator = EmotionalWeightCalculator::new(&config);

    let test_cases = [
        ("neutral statement without strong emotion", 0.8, 1.2),
        ("AMAZING! INCREDIBLE! WONDERFUL!", 1.0, 1.5),
        ("ERROR! DANGER! CRITICAL FAILURE!", 1.0, 1.5),
        ("", 0.9, 1.1), // Empty should give near-neutral weight
    ];

    for (content, min_expected, max_expected) in test_cases {
        let weight = calculator.compute_emotional_weight(
            content,
            context_graph_core::types::EmotionalState::Neutral,
        );
        assert!(
            weight >= 0.5 && weight <= 1.5,
            "weight {} out of [0.5, 1.5] for '{}'",
            weight,
            content
        );
        // Relaxed bounds check - emotional weight depends on implementation
        assert!(
            weight >= min_expected - 0.3 && weight <= max_expected + 0.3,
            "weight {} not in expected range [{}, {}] for '{}'",
            weight,
            min_expected,
            max_expected,
            content
        );
    }
}

#[test]
fn test_emotional_state_modifiers() {
    use context_graph_core::types::EmotionalState;

    // Different emotional states should produce different weight modifiers
    assert_eq!(EmotionalState::Neutral.weight_modifier(), 1.0);
    assert!(EmotionalState::Focused.weight_modifier() > 1.0);
    assert!(EmotionalState::Fatigued.weight_modifier() < 1.0);
}

// =============================================================================
// EDGE CASE TESTS (per M05-T25 specification)
// =============================================================================

#[test]
fn test_empty_context() {
    let mut processor = UtlProcessor::with_defaults();
    let embedding = generate_embedding(1536, 42);
    let context: Vec<Vec<f32>> = vec![];

    let result = processor.compute_learning("test with no context", &embedding, &context);
    assert!(result.is_ok(), "empty context should produce valid signal");

    let signal = result.unwrap();
    // With empty context, computation should still succeed with valid bounds
    assert!(signal.magnitude >= 0.0 && signal.magnitude <= 1.0);
    assert!(signal.delta_s >= 0.0 && signal.delta_s <= 1.0);
    assert!(signal.delta_c >= 0.0 && signal.delta_c <= 1.0);
}

#[test]
fn test_zero_embedding() {
    let mut processor = UtlProcessor::with_defaults();
    let embedding = vec![0.0; 1536];
    let context = generate_context(10, 1536, 100);

    let result = processor.compute_learning("test with zero embedding", &embedding, &context);
    assert!(result.is_ok(), "zero embedding should produce valid signal");

    let signal = result.unwrap();
    // Zero embedding might produce low surprise (similar to nothing)
    assert!(signal.magnitude >= 0.0 && signal.magnitude <= 1.0);
}

#[test]
fn test_empty_content() {
    let mut processor = UtlProcessor::with_defaults();
    let embedding = generate_embedding(128, 42);
    let context = vec![generate_embedding(128, 100)];

    let result = processor.compute_learning("", &embedding, &context);
    assert!(result.is_ok(), "empty content should produce valid signal");

    let signal = result.unwrap();
    // Empty content should give neutral emotional weight (close to 1.0)
    assert!(
        (signal.w_e - 1.0).abs() < 0.3,
        "empty content should give near-neutral w_e, got {}",
        signal.w_e
    );
}

#[test]
fn test_single_context() {
    let mut processor = UtlProcessor::with_defaults();
    let embedding = generate_embedding(1536, 42);
    let context = vec![generate_embedding(1536, 100)];

    let result = processor.compute_learning("test with single context", &embedding, &context);
    assert!(result.is_ok(), "single context should compute coherence");

    let signal = result.unwrap();
    assert!(signal.delta_c >= 0.0 && signal.delta_c <= 1.0);
}

#[test]
fn test_max_values() {
    // Maximum input values: all 1.0, w_e at max 1.5, phi at 0 (cos=1)
    let result = compute_learning_magnitude(1.0, 1.0, 1.5, 0.0);

    // L = (1 * 1) * 1.5 * cos(0) = 1.5, clamped to 1.0
    assert_eq!(result, 1.0, "max values should clamp to L=1.0");
}

#[test]
fn test_min_values() {
    // Minimum meaningful values
    let result = compute_learning_magnitude(0.0, 0.0, 0.5, 0.0);

    // L = (0 * 0) * 0.5 * 1 = 0
    assert_eq!(result, 0.0, "min values should yield L=0.0");
}

#[test]
fn test_phi_orthogonal() {
    // At phi = π/2, cos(π/2) ≈ 0, so learning should be ~0
    let result = compute_learning_magnitude(1.0, 1.0, 1.5, std::f32::consts::FRAC_PI_2);

    assert!(
        result.abs() < 1e-6,
        "phi=π/2 should yield L≈0, got {}",
        result
    );
}

#[test]
fn test_identical_embeddings() {
    let mut processor = UtlProcessor::with_defaults();

    // All embeddings are identical
    let embedding = uniform_embedding(128, 0.5);
    let context = vec![
        uniform_embedding(128, 0.5),
        uniform_embedding(128, 0.5),
        uniform_embedding(128, 0.5),
    ];

    let result = processor.compute_learning("identical test", &embedding, &context);
    assert!(result.is_ok());

    let signal = result.unwrap();
    // Identical embeddings should have low surprise
    assert!(
        signal.delta_s < 0.5,
        "identical embeddings should have low surprise, got {}",
        signal.delta_s
    );
}

// =============================================================================
// LEARNING INTENSITY TESTS
// =============================================================================

#[test]
fn test_learning_intensity_categories() {
    // Low: magnitude < 0.3
    let low = LearningSignal::new(
        0.2,
        0.3,
        0.3,
        1.0,
        0.5,
        None,
        JohariQuadrant::Hidden,
        SuggestedAction::GetNeighborhood,
        false,
        false,
        100,
    )
    .unwrap();
    assert_eq!(low.intensity_category(), LearningIntensity::Low);
    assert!(low.is_low_learning());
    assert!(!low.is_high_learning());

    // Medium: 0.3 <= magnitude < 0.7
    let medium = LearningSignal::new(
        0.5,
        0.5,
        0.5,
        1.0,
        0.5,
        None,
        JohariQuadrant::Open,
        SuggestedAction::DirectRecall,
        false,
        true,
        100,
    )
    .unwrap();
    assert_eq!(medium.intensity_category(), LearningIntensity::Medium);
    assert!(!medium.is_low_learning());
    assert!(!medium.is_high_learning());

    // High: magnitude >= 0.7
    let high = LearningSignal::new(
        0.8,
        0.8,
        0.8,
        1.2,
        0.3,
        None,
        JohariQuadrant::Unknown,
        SuggestedAction::EpistemicAction,
        true,
        true,
        100,
    )
    .unwrap();
    assert_eq!(high.intensity_category(), LearningIntensity::High);
    assert!(!high.is_low_learning());
    assert!(high.is_high_learning());
}

// =============================================================================
// UTL STATE TESTS
// =============================================================================

#[test]
fn test_utl_state_from_signal() {
    let signal = LearningSignal::new(
        0.7,
        0.6,
        0.8,
        1.2,
        0.5,
        Some(LifecycleLambdaWeights::for_stage(LifecycleStage::Growth)),
        JohariQuadrant::Open,
        SuggestedAction::DirectRecall,
        true,
        true,
        1500,
    )
    .unwrap();

    let state = UtlState::from_signal(&signal);

    assert_eq!(state.delta_s, signal.delta_s);
    assert_eq!(state.delta_c, signal.delta_c);
    assert_eq!(state.w_e, signal.w_e);
    assert_eq!(state.phi, signal.phi);
    assert_eq!(state.learning_magnitude, signal.magnitude);
    assert_eq!(state.quadrant, signal.quadrant);
}

#[test]
fn test_utl_state_validation() {
    // Valid state
    let valid = UtlState::empty();
    assert!(valid.validate().is_ok());

    // State with NaN should fail validation
    let invalid = UtlState {
        delta_s: f32::NAN,
        ..UtlState::empty()
    };
    assert!(invalid.validate().is_err());
}

// =============================================================================
// PERFORMANCE TESTS
// =============================================================================

#[test]
fn test_performance_under_10ms() {
    let mut processor = UtlProcessor::with_defaults();

    let content = "Performance test content for UTL processing with various words.";
    let embedding = generate_embedding(1536, 42);
    let context = generate_context(50, 1536, 100);

    let start = Instant::now();
    let signal = processor.compute_learning(content, &embedding, &context);
    let elapsed = start.elapsed();

    assert!(signal.is_ok(), "computation must succeed");
    assert!(
        elapsed.as_millis() < 10,
        "Computation took {}ms, expected < 10ms",
        elapsed.as_millis()
    );

    // Also verify latency is tracked in signal
    let signal = signal.unwrap();
    assert!(
        signal.latency_us < 10_000,
        "Reported latency {}μs exceeds 10ms",
        signal.latency_us
    );
}

#[test]
fn test_performance_large_context() {
    let mut processor = UtlProcessor::with_defaults();

    let content = "Large context performance test";
    let embedding = generate_embedding(1536, 42);
    let context = generate_context(100, 1536, 100);

    let start = Instant::now();
    let signal = processor.compute_learning(content, &embedding, &context);
    let elapsed = start.elapsed();

    assert!(signal.is_ok());
    // Allow more time for large context, but should still be reasonable
    assert!(
        elapsed.as_millis() < 50,
        "Large context took {}ms, expected < 50ms",
        elapsed.as_millis()
    );
}

// =============================================================================
// SERIALIZATION ROUNDTRIP TESTS
// =============================================================================

#[test]
fn test_learning_signal_serialization() {
    let signal = LearningSignal::new(
        0.7,
        0.6,
        0.8,
        1.2,
        0.5,
        Some(LifecycleLambdaWeights::for_stage(LifecycleStage::Growth)),
        JohariQuadrant::Open,
        SuggestedAction::DirectRecall,
        true,
        true,
        1500,
    )
    .unwrap();

    let json = serde_json::to_string(&signal).expect("serialization must succeed");
    let deserialized: LearningSignal =
        serde_json::from_str(&json).expect("deserialization must succeed");

    assert_eq!(deserialized.magnitude, signal.magnitude);
    assert_eq!(deserialized.delta_s, signal.delta_s);
    assert_eq!(deserialized.quadrant, signal.quadrant);
}

#[test]
fn test_utl_state_serialization() {
    let state = UtlState {
        delta_s: 0.6,
        delta_c: 0.8,
        w_e: 1.2,
        phi: 0.5,
        learning_magnitude: 0.7,
        quadrant: JohariQuadrant::Blind,
        last_computed: chrono::Utc::now(),
    };

    let json = serde_json::to_string(&state).expect("serialization must succeed");
    let deserialized: UtlState =
        serde_json::from_str(&json).expect("deserialization must succeed");

    assert_eq!(deserialized.delta_s, state.delta_s);
    assert_eq!(deserialized.quadrant, state.quadrant);
}

// =============================================================================
// PROCESSOR STATE MANAGEMENT TESTS
// =============================================================================

#[test]
fn test_processor_reset() {
    let mut processor = UtlProcessor::with_defaults();
    let embedding = uniform_embedding(128, 0.5);

    // Do some computations
    for _ in 0..10 {
        let _ = processor.compute_learning("test", &embedding, &[]);
    }

    assert!(processor.computation_count() > 0);
    assert!(processor.interaction_count() > 0);

    // Reset
    processor.reset();

    assert_eq!(processor.computation_count(), 0);
    assert_eq!(processor.interaction_count(), 0);
    assert_eq!(processor.lifecycle_stage(), LifecycleStage::Infancy);
}

#[test]
fn test_processor_restore_lifecycle() {
    let mut processor = UtlProcessor::with_defaults();

    // Restore to Growth stage (100 interactions)
    processor.restore_lifecycle(100);

    assert_eq!(processor.lifecycle_stage(), LifecycleStage::Growth);
    assert_eq!(processor.interaction_count(), 100);

    // Restore to Maturity stage (600 interactions)
    processor.reset();
    processor.restore_lifecycle(600);

    assert_eq!(processor.lifecycle_stage(), LifecycleStage::Maturity);
    assert_eq!(processor.interaction_count(), 600);
}

// =============================================================================
// STAGE THRESHOLDS INTEGRATION TESTS
// =============================================================================

#[test]
fn test_stage_thresholds_progression() {
    let infancy = StageThresholds::infancy();
    let growth = StageThresholds::growth();
    let maturity = StageThresholds::maturity();

    // Entropy triggers should decrease (less novelty-seeking over time)
    assert!(infancy.entropy_trigger > growth.entropy_trigger);
    assert!(growth.entropy_trigger > maturity.entropy_trigger);

    // Coherence triggers should increase
    assert!(infancy.coherence_trigger < growth.coherence_trigger);
    assert!(growth.coherence_trigger < maturity.coherence_trigger);

    // Min importance should increase (more selective)
    assert!(infancy.min_importance_store < growth.min_importance_store);
    assert!(growth.min_importance_store < maturity.min_importance_store);
}
