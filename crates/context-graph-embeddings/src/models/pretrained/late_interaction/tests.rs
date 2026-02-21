//! Core tests for LateInteractionModel.

use super::*;
use crate::traits::SingleModelConfig;
use std::path::PathBuf;

/// Get the workspace root directory for test model paths.
fn workspace_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

pub(crate) fn create_test_model() -> LateInteractionModel {
    let model_path = workspace_root().join("models/late-interaction");
    LateInteractionModel::new(&model_path, SingleModelConfig::default())
        .expect("Failed to create LateInteractionModel")
}

pub(crate) async fn create_and_load_model() -> LateInteractionModel {
    let model = create_test_model();
    model.load().await.expect("Failed to load model");
    model
}

#[test]
fn test_new_creates_unloaded_model() {
    let model = create_test_model();
    assert!(!model.is_initialized());
}
