//! Text and image embedding tests for MultimodalModel.

use crate::traits::EmbeddingModel;
use crate::types::{ImageFormat, ModelId, ModelInput};

use super::super::MULTIMODAL_DIMENSION;
use super::{create_and_load_model, create_minimal_jpeg, create_minimal_png};

// ==================== Text Embedding Tests ====================

#[tokio::test]
async fn test_embed_text_returns_768d() {
    let model = create_and_load_model().await;
    let input = ModelInput::text("a photo of a cat").expect("Input");
    let embedding = model.embed(&input).await.expect("Embed should succeed");
    assert_eq!(embedding.vector.len(), MULTIMODAL_DIMENSION);
    assert_eq!(embedding.vector.len(), 768);
}

#[tokio::test]
async fn test_embed_text_returns_l2_normalized_vector() {
    let model = create_and_load_model().await;
    let input = ModelInput::text("a dog playing in the park").expect("Input");
    let embedding = model.embed(&input).await.expect("Embed");
    let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.001,
        "L2 norm should be ~1.0, got {}",
        norm
    );
}

#[tokio::test]
async fn test_embed_text_no_nan_values() {
    let model = create_and_load_model().await;
    let input = ModelInput::text("CLIP text embedding test").expect("Input");
    let embedding = model.embed(&input).await.expect("Embed");
    let has_nan = embedding.vector.iter().any(|x| x.is_nan());
    assert!(!has_nan, "Vector must not contain NaN values");
}

#[tokio::test]
async fn test_embed_text_deterministic() {
    let model = create_and_load_model().await;
    let input = ModelInput::text("a photo of a cat").expect("Input");
    let emb1 = model.embed(&input).await.expect("Embed 1");
    let emb2 = model.embed(&input).await.expect("Embed 2");
    assert_eq!(
        emb1.vector, emb2.vector,
        "Same input must produce identical embeddings"
    );
}

#[tokio::test]
async fn test_embed_text_different_inputs_differ() {
    let model = create_and_load_model().await;
    let input1 = ModelInput::text("a photo of a cat").expect("Input");
    let input2 = ModelInput::text("a photo of a dog").expect("Input");
    let emb1 = model.embed(&input1).await.expect("Embed 1");
    let emb2 = model.embed(&input2).await.expect("Embed 2");
    assert_ne!(
        emb1.vector, emb2.vector,
        "Different inputs must produce different embeddings"
    );
}

// ==================== Image Embedding Tests ====================

#[tokio::test]
async fn test_embed_image_returns_768d() {
    let model = create_and_load_model().await;
    let png_bytes = create_minimal_png();
    let input = ModelInput::image(png_bytes, ImageFormat::Png).expect("Image input");
    let embedding = model.embed(&input).await.expect("Embed should succeed");
    assert_eq!(embedding.vector.len(), MULTIMODAL_DIMENSION);
}

#[tokio::test]
async fn test_embed_jpeg_image() {
    let model = create_and_load_model().await;
    let jpeg_bytes = create_minimal_jpeg();
    let input = ModelInput::image(jpeg_bytes, ImageFormat::Jpeg).expect("JPEG input");
    let embedding = model.embed(&input).await.expect("Embed should succeed");
    assert_eq!(embedding.vector.len(), 768);
    assert_eq!(embedding.model_id, ModelId::Contextual);
}

// ==================== Cross-Modal Tests ====================

#[tokio::test]
async fn test_text_and_image_in_same_embedding_space() {
    let model = create_and_load_model().await;
    let text_input = ModelInput::text("a photo of a cat").expect("Text");
    let jpeg_bytes = create_minimal_jpeg();
    let image_input = ModelInput::image(jpeg_bytes, ImageFormat::Jpeg).expect("Image");
    let text_emb = model.embed(&text_input).await.expect("Text embed");
    let image_emb = model.embed(&image_input).await.expect("Image embed");
    assert_eq!(text_emb.vector.len(), 768);
    assert_eq!(image_emb.vector.len(), 768);
    let text_norm: f32 = text_emb.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    let image_norm: f32 = image_emb.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((text_norm - 1.0).abs() < 0.001);
    assert!((image_norm - 1.0).abs() < 0.001);
    let similarity: f32 = text_emb
        .vector
        .iter()
        .zip(&image_emb.vector)
        .map(|(a, b)| a * b)
        .sum();
    assert!(similarity.is_finite());
}

// ==================== Thread Safety Tests ====================

#[tokio::test]
async fn test_concurrent_embed_calls() {
    let model = std::sync::Arc::new(create_and_load_model().await);
    let mut handles = Vec::new();
    for i in 0..10 {
        let model = model.clone();
        let handle = tokio::spawn(async move {
            let text = format!("a photo of animal number {}", i);
            let input = ModelInput::text(&text).expect("Input");
            model.embed(&input).await
        });
        handles.push(handle);
    }
    for handle in handles {
        let result = handle.await;
        let embedding = result
            .expect("Task should not panic")
            .expect("Embed should succeed");
        assert_eq!(embedding.vector.len(), 768);
    }
}
