//! Core tests for the CodeModel.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use std::path::PathBuf;

    use crate::models::pretrained::code::CodeModel;
    use crate::traits::{EmbeddingModel, SingleModelConfig};

    /// Get the workspace root directory for test model paths.
    pub(crate) fn workspace_root() -> PathBuf {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(manifest_dir)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    pub(crate) fn create_test_model() -> CodeModel {
        let model_path = workspace_root().join("models/code-1536");
        CodeModel::new(&model_path, SingleModelConfig::default())
            .expect("Failed to create CodeModel")
    }

    #[test]
    fn test_new_creates_unloaded_model() {
        let model = create_test_model();
        assert!(!model.is_initialized());
    }
}
