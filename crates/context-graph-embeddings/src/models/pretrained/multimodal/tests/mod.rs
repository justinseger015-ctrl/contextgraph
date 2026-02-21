//! Tests for the multimodal CLIP embedding model.

mod construction_tests;
mod embedding_tests;
mod image_processor_tests;

use std::path::PathBuf;

use crate::traits::SingleModelConfig;

use super::MultimodalModel;

/// Get the workspace root directory for test model paths.
pub fn workspace_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn create_test_model() -> MultimodalModel {
    let model_path = workspace_root().join("models/multimodal");
    MultimodalModel::new(&model_path, SingleModelConfig::default())
        .expect("Failed to create MultimodalModel")
}

pub async fn create_and_load_model() -> MultimodalModel {
    let model = create_test_model();
    model.load().await.expect("Failed to load model");
    model
}
