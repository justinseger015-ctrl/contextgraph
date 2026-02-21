//! Embedding and trait implementation tests for TemporalPositionalModel.

use crate::traits::EmbeddingModel;
use crate::types::ModelInput;

use super::super::TemporalPositionalModel;

#[tokio::test]
async fn test_embed_returns_512d_vector() {
    let model = TemporalPositionalModel::new();
    let input = ModelInput::text("test content").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    println!("Vector length: {}", embedding.vector.len());
    assert_eq!(embedding.vector.len(), 512, "Must return exactly 512D");
}
