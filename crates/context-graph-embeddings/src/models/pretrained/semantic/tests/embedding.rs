//! Embedding behavior tests for the semantic embedding model.

use crate::types::ModelInput;

use super::super::SEMANTIC_DIMENSION;
use super::helpers::create_and_load_model;

#[tokio::test]
async fn test_embed_batch() {
    let model = create_and_load_model().await;
    let inputs = vec![
        ModelInput::text("First").expect("Failed to create input"),
        ModelInput::text("Second").expect("Failed to create input"),
        ModelInput::text("Third").expect("Failed to create input"),
    ];

    let embeddings = model.embed_batch(&inputs).await.unwrap();

    assert_eq!(embeddings.len(), 3);
    for embedding in &embeddings {
        assert_eq!(embedding.vector.len(), SEMANTIC_DIMENSION);
        let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.001,
            "Each embedding must be L2 normalized"
        );
    }
}
