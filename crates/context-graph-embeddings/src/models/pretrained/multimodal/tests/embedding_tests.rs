//! Text and image embedding tests for MultimodalModel.

use crate::traits::EmbeddingModel;
use crate::types::ModelInput;

use super::super::MULTIMODAL_DIMENSION;
use super::create_and_load_model;

// ==================== Text Embedding Tests ====================

#[tokio::test]
async fn test_embed_text_returns_768d() {
    let model = create_and_load_model().await;
    let input = ModelInput::text("a photo of a cat").expect("Input");
    let embedding = model.embed(&input).await.expect("Embed should succeed");
    assert_eq!(embedding.vector.len(), MULTIMODAL_DIMENSION);
    assert_eq!(embedding.vector.len(), 768);
}
