//! Tests for the CausalModel (nomic-embed-text-v1.5).

use super::*;
use crate::traits::{EmbeddingModel, SingleModelConfig};
use std::path::PathBuf;

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

#[test]
fn test_new_creates_unloaded_model() {
    assert!(!create_test_model().is_initialized());
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
