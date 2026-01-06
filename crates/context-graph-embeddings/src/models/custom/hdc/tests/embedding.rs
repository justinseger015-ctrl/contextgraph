//! EmbeddingModel trait implementation tests.

use super::*;

#[test]
fn test_supported_input_types() {
    let model = HdcModel::default_model();
    let types = model.supported_input_types();
    assert!(types.contains(&InputType::Text));
    assert!(types.contains(&InputType::Code));
    assert!(!types.contains(&InputType::Image));
    assert!(!types.contains(&InputType::Audio));
}

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

#[tokio::test]
async fn test_embed_code_succeeds() {
    let model = HdcModel::default_model();
    let input = ModelInput::code("fn main() { println!(\"Hello\"); }", "rust").unwrap();
    let result = model.embed(&input).await;
    assert!(result.is_ok());
    let embedding = result.unwrap();
    assert_eq!(embedding.model_id, ModelId::Hdc);
}

#[tokio::test]
async fn test_embed_image_fails() {
    let model = HdcModel::default_model();
    let input = ModelInput::image(vec![1, 2, 3, 4], crate::types::ImageFormat::Png).unwrap();
    let result = model.embed(&input).await;
    assert!(result.is_err());
    match result {
        Err(EmbeddingError::UnsupportedModality {
            model_id,
            input_type,
        }) => {
            assert_eq!(model_id, ModelId::Hdc);
            assert_eq!(input_type, InputType::Image);
        }
        _ => panic!("Expected UnsupportedModality error"),
    }
}

#[tokio::test]
async fn test_embed_empty_text_fails() {
    let model = HdcModel::default_model();
    // Create text with only whitespace
    let input = ModelInput::Text {
        content: "   ".to_string(),
        instruction: None,
    };
    let result = model.embed(&input).await;
    assert!(result.is_err());
    assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
}

#[tokio::test]
async fn test_embed_deterministic() {
    let model = HdcModel::default_model();
    let input = ModelInput::text("Deterministic test").unwrap();

    let emb1 = model.embed(&input).await.unwrap();
    let emb2 = model.embed(&input).await.unwrap();

    assert_eq!(emb1.vector, emb2.vector);
}
