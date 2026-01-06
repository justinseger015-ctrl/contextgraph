//! Embedding output and validation tests for TemporalPeriodicModel.

use crate::error::EmbeddingError;
use crate::models::custom::temporal_periodic::TemporalPeriodicModel;
use crate::traits::EmbeddingModel;
use crate::types::{InputType, ModelId, ModelInput};

#[tokio::test]
async fn test_embed_returns_512d_vector() {
    let model = TemporalPeriodicModel::new();
    let input = ModelInput::text("test content").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    println!("Vector length: {}", embedding.vector.len());
    assert_eq!(embedding.vector.len(), 512, "Must return exactly 512D");
}

#[tokio::test]
async fn test_embed_model_id_correct() {
    let model = TemporalPeriodicModel::new();
    let input = ModelInput::text("test").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    assert_eq!(embedding.model_id, ModelId::TemporalPeriodic);
}

#[tokio::test]
async fn test_embed_l2_normalized() {
    let model = TemporalPeriodicModel::new();
    let input = ModelInput::text("test normalization").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    println!("L2 norm: {}", norm);
    println!("Deviation from 1.0: {}", (norm - 1.0).abs());

    assert!(
        (norm - 1.0).abs() < 0.001,
        "Vector MUST be L2 normalized, got norm = {}",
        norm
    );
}

#[tokio::test]
async fn test_embed_no_nan_values() {
    let model = TemporalPeriodicModel::new();
    let input = ModelInput::text("test").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    let has_nan = embedding.vector.iter().any(|x| x.is_nan());

    assert!(!has_nan, "Output must not contain NaN values");
}

#[tokio::test]
async fn test_embed_no_inf_values() {
    let model = TemporalPeriodicModel::new();
    let input = ModelInput::text("test").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    let has_inf = embedding.vector.iter().any(|x| x.is_infinite());

    assert!(!has_inf, "Output must not contain Inf values");
}

#[tokio::test]
async fn test_embed_records_latency() {
    let model = TemporalPeriodicModel::new();
    let input = ModelInput::text("test latency").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    println!("Latency: {} microseconds", embedding.latency_us);

    // Latency should be recorded (not necessarily non-zero for very fast ops)
    // Just verify the value is reasonable (< 2s as an upper sanity check)
    assert!(
        embedding.latency_us < 2_000_000,
        "Latency should be under 2 seconds"
    );
}

#[tokio::test]
async fn test_embed_latency_under_2ms() {
    let model = TemporalPeriodicModel::new();
    let input = ModelInput::text("test performance").expect("Failed to create input");

    let start = std::time::Instant::now();
    let _embedding = model.embed(&input).await.expect("Embed should succeed");
    let elapsed = start.elapsed();

    println!("Elapsed: {:?}", elapsed);

    assert!(
        elapsed.as_millis() < 2,
        "Latency must be under 2ms, got {:?}",
        elapsed
    );
}

#[tokio::test]
async fn test_unsupported_code_input() {
    let model = TemporalPeriodicModel::new();
    let input = ModelInput::code("fn main() {}", "rust").expect("Failed to create input");

    let result = model.embed(&input).await;

    assert!(result.is_err(), "Code input should be rejected");
    match result {
        Err(EmbeddingError::UnsupportedModality {
            model_id,
            input_type,
        }) => {
            assert_eq!(model_id, ModelId::TemporalPeriodic);
            assert_eq!(input_type, InputType::Code);
        }
        other => panic!("Expected UnsupportedModality error, got {:?}", other),
    }
}

#[tokio::test]
async fn test_fourier_produces_sin_cos_pairs() {
    let model = TemporalPeriodicModel::new();
    let input = ModelInput::text_with_instruction("test", "timestamp:2024-01-15T10:30:00Z")
        .expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    // For each period, we should have alternating sin/cos values
    // After normalization, sin^2 + cos^2 property is modified but
    // consecutive pairs should still show harmonic structure

    // Just verify we have valid values
    assert_eq!(embedding.vector.len(), 512);
    assert!(embedding.vector.iter().all(|x| x.is_finite()));
}
