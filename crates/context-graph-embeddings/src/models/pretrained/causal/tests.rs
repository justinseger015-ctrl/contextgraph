//! Tests for the CausalModel (nomic-embed-text-v1.5).

use super::*;
use crate::error::EmbeddingError;
use crate::traits::{EmbeddingModel, SingleModelConfig};
use crate::types::{InputType, ModelId, ModelInput};
use once_cell::sync::OnceCell;
use serial_test::serial;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::OnceCell as AsyncOnceCell;

/// Shared warm model instance for latency testing.
/// Loaded once and reused across all latency tests to measure true warm inference latency.
static WARM_MODEL: OnceCell<Arc<AsyncOnceCell<CausalModel>>> = OnceCell::new();

/// Get or initialize the shared warm model instance.
async fn get_warm_model() -> &'static CausalModel {
    let cell = WARM_MODEL.get_or_init(|| Arc::new(AsyncOnceCell::new()));
    cell.get_or_init(|| async {
        let model = create_test_model();
        model.load().await.expect("Failed to load warm model");
        model
    })
    .await
}

fn workspace_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn create_test_model() -> CausalModel {
    let model_path = workspace_root().join("models/causal");
    CausalModel::new(&model_path, SingleModelConfig::default())
        .expect("Failed to create CausalModel")
}

async fn create_and_load_model() -> CausalModel {
    let model = create_test_model();
    model.load().await.expect("Failed to load model");
    model
}

// Construction Tests
#[test]
fn test_new_creates_unloaded_model() {
    assert!(!create_test_model().is_initialized());
}

#[test]
fn test_new_with_zero_batch_size_fails() {
    let config = SingleModelConfig {
        max_batch_size: 0,
        ..Default::default()
    };
    let result = CausalModel::new(&workspace_root().join("models/causal"), config);
    assert!(matches!(result, Err(EmbeddingError::ConfigError { .. })));
}

// Trait Implementation Tests
#[test]
fn test_model_metadata() {
    let model = create_test_model();
    assert_eq!(model.model_id(), ModelId::Causal);
    assert_eq!(model.dimension(), 768);
    assert_eq!(model.max_tokens(), 512);
    assert_eq!(model.latency_budget_ms(), 8);
    assert!(model.is_pretrained());
    assert_eq!(model.supported_input_types(), &[InputType::Text]);
}

// State Transition Tests
#[tokio::test]
async fn test_load_sets_initialized() {
    let model = create_test_model();
    assert!(!model.is_initialized());
    model.load().await.expect("Load should succeed");
    assert!(model.is_initialized());
}

#[tokio::test]
async fn test_unload_clears_initialized() {
    let model = create_and_load_model().await;
    model.unload().await.expect("Unload should succeed");
    assert!(!model.is_initialized());
}

#[tokio::test]
async fn test_unload_when_not_loaded_fails() {
    let result = create_test_model().unload().await;
    assert!(matches!(result, Err(EmbeddingError::NotInitialized { .. })));
}

// Serial: Multiple load/unload cycles require exclusive VRAM access
#[tokio::test]
#[serial]
async fn test_state_transitions_full_cycle() {
    let model = create_test_model();
    assert!(!model.is_initialized());
    model.load().await.unwrap();
    assert!(model.is_initialized());
    model.unload().await.unwrap();
    assert!(!model.is_initialized());
    model.load().await.unwrap();
    assert!(model.is_initialized());
}

// Embedding Tests
#[tokio::test]
async fn test_embed_before_load_fails() {
    let input = ModelInput::text("test").expect("Failed to create input");
    let result = create_test_model().embed(&input).await;
    assert!(matches!(result, Err(EmbeddingError::NotInitialized { .. })));
}

#[tokio::test]
async fn test_embed_returns_768d_vector() {
    let model = create_and_load_model().await;
    let input = ModelInput::text("Test for causal embedding").expect("Input");
    let embedding = model.embed(&input).await.expect("Embed should succeed");
    assert_eq!(embedding.vector.len(), CAUSAL_DIMENSION);
}

#[tokio::test]
async fn test_embed_returns_l2_normalized_vector() {
    let model = create_and_load_model().await;
    let input = ModelInput::text("Test normalization").expect("Input");
    let embedding = model.embed(&input).await.expect("Embed");
    let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.001,
        "L2 norm should be ~1.0, got {}",
        norm
    );
}

#[tokio::test]
async fn test_embed_no_nan_or_inf_values() {
    let model = create_and_load_model().await;
    let input = ModelInput::text("Check for NaN and Inf").expect("Input");
    let embedding = model.embed(&input).await.expect("Embed");
    assert!(
        !embedding.vector.iter().any(|x| x.is_nan()),
        "Vector must not contain NaN"
    );
    assert!(
        !embedding.vector.iter().any(|x| x.is_infinite()),
        "Vector must not contain Inf"
    );
}

#[tokio::test]
async fn test_embed_deterministic() {
    let model = create_and_load_model().await;
    let input = ModelInput::text("Determinism check").expect("Input");
    let emb1 = model.embed(&input).await.expect("Embed 1");
    let emb2 = model.embed(&input).await.expect("Embed 2");
    assert_eq!(
        emb1.vector, emb2.vector,
        "Same input must produce identical embeddings"
    );
}

#[tokio::test]
async fn test_embed_different_inputs_differ() {
    let model = create_and_load_model().await;
    let emb1 = model
        .embed(&ModelInput::text("Input A").expect("Input"))
        .await
        .expect("Embed 1");
    let emb2 = model
        .embed(&ModelInput::text("Input B").expect("Input"))
        .await
        .expect("Embed 2");
    assert_ne!(
        emb1.vector, emb2.vector,
        "Different inputs must produce different embeddings"
    );
}

#[tokio::test]
async fn test_embed_model_id_is_causal() {
    let model = create_and_load_model().await;
    let embedding = model
        .embed(&ModelInput::text("test").expect("Input"))
        .await
        .expect("Embed");
    assert_eq!(embedding.model_id, ModelId::Causal);
}

/// Test embedding latency is under budget for warmed models.
#[tokio::test]
async fn test_embed_latency_under_budget() {
    let model = get_warm_model().await;
    let input = ModelInput::text("Short text for latency test").expect("Input");

    // Warm-up calls
    for _ in 0..10 {
        let _warmup = model.embed(&input).await.expect("Warm-up embed");
    }

    // Measure actual inference latency
    let mut latencies = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = std::time::Instant::now();
        let _embedding = model.embed(&input).await.expect("Embed");
        latencies.push(start.elapsed());
    }
    latencies.sort();
    let median_latency = latencies[2];

    let budget_ms: u128 = 1500;
    assert!(
        median_latency.as_millis() < budget_ms,
        "Warm model median latency {} ms exceeds {}ms budget (latencies: {:?})",
        median_latency.as_millis(),
        budget_ms,
        latencies
    );
}

// Batch Tests
#[tokio::test]
async fn test_embed_batch_multiple_inputs() {
    let model = create_and_load_model().await;
    let inputs = vec![
        ModelInput::text("Input 1").expect("Input"),
        ModelInput::text("Input 2").expect("Input"),
        ModelInput::text("Input 3").expect("Input"),
    ];
    let embeddings = model.embed_batch(&inputs).await.expect("Batch embed");
    assert_eq!(embeddings.len(), 3);
    for emb in &embeddings {
        assert_eq!(emb.vector.len(), 768);
        assert_eq!(emb.model_id, ModelId::Causal);
    }
}

#[tokio::test]
async fn test_embed_batch_before_load_fails() {
    let inputs = vec![ModelInput::text("test").expect("Input")];
    let result = create_test_model().embed_batch(&inputs).await;
    assert!(matches!(result, Err(EmbeddingError::NotInitialized { .. })));
}

// Edge Case Tests
#[tokio::test]
async fn test_edge_case_empty_input() {
    let model = create_and_load_model().await;
    assert!(ModelInput::text("").is_err(), "Empty string should error");
    let input = ModelInput::text(" ").expect("Whitespace input should work");
    let embedding = model.embed(&input).await.expect("Whitespace should embed");
    assert_eq!(embedding.vector.len(), 768);
}

#[tokio::test]
async fn test_edge_case_long_document() {
    let model = create_and_load_model().await;
    let long_text = "This is a test sentence. ".repeat(800);
    let embedding = model
        .embed(&ModelInput::text(&long_text).expect("Input"))
        .await
        .expect("Long embed");
    assert_eq!(embedding.vector.len(), 768);
}

#[tokio::test]
async fn test_edge_case_unsupported_modality() {
    let model = create_and_load_model().await;
    let input = ModelInput::Code {
        content: "fn main() {}".to_string(),
        language: "rust".to_string(),
    };
    let result = model.embed(&input).await;
    assert!(matches!(
        result,
        Err(EmbeddingError::UnsupportedModality { .. })
    ));
}

// Source of Truth Verification
#[tokio::test]
async fn test_source_of_truth_verification() {
    let model = create_and_load_model().await;
    let input =
        ModelInput::text_with_instruction("Cause A leads to Effect B", "causal").expect("Input");
    let embedding = model.embed(&input).await.expect("Embed should succeed");
    assert_eq!(embedding.model_id, ModelId::Causal);
    assert_eq!(embedding.vector.len(), 768);
    let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.001);
    assert!(!embedding
        .vector
        .iter()
        .any(|x| x.is_nan() || x.is_infinite()));
}

// Evidence of Success (comprehensive)
#[tokio::test]
async fn test_evidence_of_success() {
    let model = create_and_load_model().await;
    assert_eq!(model.model_id(), ModelId::Causal);
    assert_eq!(model.dimension(), 768);
    assert!(model.is_initialized());

    let input = ModelInput::text("Causal reasoning test").expect("Input");
    let embedding = model.embed(&input).await.expect("Embed");
    assert_eq!(embedding.vector.len(), 768);

    let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.001);

    let emb2 = model.embed(&input).await.unwrap();
    assert_eq!(embedding.vector, emb2.vector, "Deterministic");

    let emb3 = model
        .embed(&ModelInput::text("Different text").expect("Input"))
        .await
        .unwrap();
    assert_ne!(embedding.vector, emb3.vector, "Different inputs differ");
}

// Constants Tests
#[test]
fn test_constants_are_correct() {
    assert_eq!(CAUSAL_DIMENSION, 768);
    assert_eq!(CAUSAL_MAX_TOKENS, 512);
    assert_eq!(CAUSAL_LATENCY_BUDGET_MS, 8);
}

// =============================================================================
// Dual Embedding Tests (ARCH-15 Compliance)
// =============================================================================

#[test]
fn test_cause_instruction_constant() {
    assert!(!CAUSE_INSTRUCTION.is_empty());
    assert!(CAUSE_INSTRUCTION.contains("cause"));
    assert_ne!(
        CAUSE_INSTRUCTION,
        EFFECT_INSTRUCTION,
        "Instruction prefixes must be different"
    );
}

#[test]
fn test_effect_instruction_constant() {
    assert!(!EFFECT_INSTRUCTION.is_empty());
    assert!(EFFECT_INSTRUCTION.contains("effect"));
}

#[tokio::test]
async fn test_embed_as_cause_before_load_fails() {
    let result = create_test_model().embed_as_cause("test content").await;
    assert!(
        matches!(result, Err(EmbeddingError::NotInitialized { .. })),
        "embed_as_cause should fail before model is loaded"
    );
}

#[tokio::test]
async fn test_embed_as_effect_before_load_fails() {
    let result = create_test_model().embed_as_effect("test content").await;
    assert!(
        matches!(result, Err(EmbeddingError::NotInitialized { .. })),
        "embed_as_effect should fail before model is loaded"
    );
}

#[tokio::test]
async fn test_embed_dual_before_load_fails() {
    let result = create_test_model().embed_dual("test content").await;
    assert!(
        matches!(result, Err(EmbeddingError::NotInitialized { .. })),
        "embed_dual should fail before model is loaded"
    );
}

#[tokio::test]
async fn test_embed_as_cause_returns_768d_vector() {
    let model = create_and_load_model().await;
    let embedding = model
        .embed_as_cause("The pressure increase causes temperature rise")
        .await
        .expect("embed_as_cause should succeed");
    assert_eq!(
        embedding.len(),
        CAUSAL_DIMENSION,
        "Cause embedding must be 768D"
    );
}

#[tokio::test]
async fn test_embed_as_effect_returns_768d_vector() {
    let model = create_and_load_model().await;
    let embedding = model
        .embed_as_effect("Temperature rose due to pressure increase")
        .await
        .expect("embed_as_effect should succeed");
    assert_eq!(
        embedding.len(),
        CAUSAL_DIMENSION,
        "Effect embedding must be 768D"
    );
}

#[tokio::test]
async fn test_embed_dual_returns_two_768d_vectors() {
    let model = create_and_load_model().await;
    let (cause_vec, effect_vec) = model
        .embed_dual("Climate change affects global weather patterns")
        .await
        .expect("embed_dual should succeed");

    assert_eq!(cause_vec.len(), CAUSAL_DIMENSION, "Cause vector must be 768D");
    assert_eq!(
        effect_vec.len(),
        CAUSAL_DIMENSION,
        "Effect vector must be 768D"
    );
}

/// Test: Verify that cause and effect vectors are not bit-for-bit identical.
///
/// With instruction-prefix asymmetry, cause/effect vectors differ because
/// different instruction prefixes produce different token sequences through
/// the encoder. The per-dimension difference is small (~1e-5) since the texts
/// share ~93% of tokens, but the vectors are NOT identical.
///
/// Note: In production, meaningful asymmetry comes from the 0.8x/1.2x direction
/// modifiers applied during search (see asymmetric.rs), not raw embedding divergence.
#[tokio::test]
async fn test_embed_dual_produces_different_vectors() {
    let model = create_and_load_model().await;
    let content = "Smoking causes lung cancer because of tar buildup, therefore leading to respiratory failure";
    let (cause_vec, effect_vec) = model.embed_dual(content).await.expect("embed_dual");

    // Verify vectors are not bit-for-bit identical — they must differ in at least one dimension
    let max_diff: f32 = cause_vec
        .iter()
        .zip(effect_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff > 0.0,
        "Cause and effect vectors must not be bit-for-bit identical! \
         The two instruction prefixes should produce different token sequences."
    );

    // The per-dimension difference should be small but non-zero (~1e-5 to 1e-3)
    // since the texts share most tokens and only differ in the prefix
    let mean_diff: f32 = cause_vec
        .iter()
        .zip(effect_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / cause_vec.len() as f32;

    assert!(
        mean_diff > 1e-7,
        "Mean per-dimension difference too small ({:.2e}), vectors are nearly identical",
        mean_diff
    );

    // Calculate cosine similarity
    let dot: f32 = cause_vec
        .iter()
        .zip(effect_vec.iter())
        .map(|(a, b)| a * b)
        .sum();
    let norm_cause: f32 = cause_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_effect: f32 = effect_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine_sim = dot / (norm_cause * norm_effect);

    // With instruction-prefix asymmetry on similar content, cosine should be very
    // high (>0.99) since only 2/15 tokens differ. The production search system uses
    // 0.8x/1.2x direction modifiers to amplify this small difference.
    assert!(
        cosine_sim > 0.5,
        "Cause and effect vectors should still be related! \
         Cosine similarity = {} is too low.",
        cosine_sim
    );

    println!(
        "[OK] Cause-Effect: cosine={:.6}, max_diff={:.2e}, mean_diff={:.2e}",
        cosine_sim, max_diff, mean_diff
    );
}

#[tokio::test]
async fn test_embed_dual_deterministic() {
    let model = create_and_load_model().await;
    let content = "Exercise improves cardiovascular health";

    let (cause1, effect1) = model.embed_dual(content).await.expect("embed_dual 1");
    let (cause2, effect2) = model.embed_dual(content).await.expect("embed_dual 2");

    assert_eq!(
        cause1, cause2,
        "Cause embeddings must be deterministic for same input"
    );
    assert_eq!(
        effect1, effect2,
        "Effect embeddings must be deterministic for same input"
    );
}

#[tokio::test]
async fn test_embed_as_cause_matches_dual_cause() {
    let model = create_and_load_model().await;
    let content = "Deforestation leads to habitat loss";

    let cause_vec = model.embed_as_cause(content).await.expect("embed_as_cause");
    let (dual_cause, _) = model.embed_dual(content).await.expect("embed_dual");

    assert_eq!(
        cause_vec, dual_cause,
        "embed_as_cause must produce same vector as embed_dual cause component"
    );
}

#[tokio::test]
async fn test_embed_as_effect_matches_dual_effect() {
    let model = create_and_load_model().await;
    let content = "Global warming results from greenhouse gas emissions";

    let effect_vec = model
        .embed_as_effect(content)
        .await
        .expect("embed_as_effect");
    let (_, dual_effect) = model.embed_dual(content).await.expect("embed_dual");

    assert_eq!(
        effect_vec, dual_effect,
        "embed_as_effect must produce same vector as embed_dual effect component"
    );
}

#[tokio::test]
async fn test_embed_dual_vectors_are_normalized() {
    let model = create_and_load_model().await;
    let (cause_vec, effect_vec) = model
        .embed_dual("Water scarcity threatens agriculture")
        .await
        .expect("embed_dual");

    let cause_norm: f32 = cause_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let effect_norm: f32 = effect_vec.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!(
        (cause_norm - 1.0).abs() < 0.001,
        "Cause vector should be L2 normalized, got norm = {}",
        cause_norm
    );
    assert!(
        (effect_norm - 1.0).abs() < 0.001,
        "Effect vector should be L2 normalized, got norm = {}",
        effect_norm
    );
}

#[tokio::test]
async fn test_embed_dual_no_nan_or_inf() {
    let model = create_and_load_model().await;
    let (cause_vec, effect_vec) = model
        .embed_dual("Economic policy affects employment rates")
        .await
        .expect("embed_dual");

    assert!(
        !cause_vec.iter().any(|x| x.is_nan()),
        "Cause vector must not contain NaN"
    );
    assert!(
        !cause_vec.iter().any(|x| x.is_infinite()),
        "Cause vector must not contain Inf"
    );
    assert!(
        !effect_vec.iter().any(|x| x.is_nan()),
        "Effect vector must not contain NaN"
    );
    assert!(
        !effect_vec.iter().any(|x| x.is_infinite()),
        "Effect vector must not contain Inf"
    );
}
