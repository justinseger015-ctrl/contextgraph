//! EmbeddingModel trait implementation tests.

use super::*;

#[tokio::test]
async fn test_embed_text_succeeds() {
    let model = HdcModel::default_model();
    let input = ModelInput::text("Hello, HDC world!").unwrap();
    let result = model.embed(&input).await;
    assert!(result.is_ok());
    let embedding = result.unwrap();
    assert_eq!(embedding.model_id, ModelId::Hdc);
    assert_eq!(embedding.dimension(), HDC_PROJECTED_DIMENSION);
}
