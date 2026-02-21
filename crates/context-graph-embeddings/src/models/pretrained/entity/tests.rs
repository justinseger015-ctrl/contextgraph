//! Tests for EntityModel.

use super::*;
use crate::traits::{EmbeddingModel, SingleModelConfig};
use crate::types::ModelInput;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn create_test_model() -> EntityModel {
    let model_path = workspace_root().join("models/entity");
    EntityModel::new(&model_path, SingleModelConfig::default())
        .expect("Failed to create EntityModel")
}

#[test]
fn test_new_creates_unloaded_model() {
    let model = create_test_model();
    assert!(!model.is_initialized());
}

#[tokio::test]
async fn test_embed_text_returns_384d() {
    let model = create_test_model();
    model.load().await.expect("Failed to load model");
    let input = ModelInput::text("[PERSON] Alice").expect("Input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    assert_eq!(embedding.vector.len(), ENTITY_DIMENSION);
    assert_eq!(embedding.vector.len(), 384);
}
