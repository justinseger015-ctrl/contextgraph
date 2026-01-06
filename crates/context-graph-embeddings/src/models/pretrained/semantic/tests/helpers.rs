//! Test helpers for the semantic embedding model.

use std::path::PathBuf;

use crate::traits::SingleModelConfig;

use super::super::SemanticModel;

/// Get the workspace root directory for test model paths.
pub fn workspace_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Helper to create a test model.
pub async fn create_test_model() -> SemanticModel {
    let model_path = workspace_root().join("models/semantic");
    SemanticModel::new(&model_path, SingleModelConfig::default()).expect("Failed to create model")
}

/// Helper to create and load a test model.
pub async fn create_and_load_model() -> SemanticModel {
    let model = create_test_model().await;
    model.load().await.expect("Failed to load model");
    model
}
