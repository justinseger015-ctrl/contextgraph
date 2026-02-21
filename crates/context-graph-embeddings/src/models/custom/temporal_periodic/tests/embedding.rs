//! Embedding output and validation tests for TemporalPeriodicModel.

use crate::models::custom::temporal_periodic::TemporalPeriodicModel;
use crate::traits::EmbeddingModel;
use crate::types::ModelInput;

#[tokio::test]
async fn test_embed_returns_512d_vector() {
    let model = TemporalPeriodicModel::new();
    let input = ModelInput::text("test content").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    assert_eq!(embedding.vector.len(), 512, "Must return exactly 512D");
}
