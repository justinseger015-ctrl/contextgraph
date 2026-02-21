//! Edge case tests for the CodeModel.

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::models::pretrained::code::{CodeModel, CODE_PROJECTED_DIMENSION};
    use crate::traits::{EmbeddingModel, SingleModelConfig};
    use crate::types::ModelInput;

    fn workspace_root() -> PathBuf {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(manifest_dir)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    async fn create_and_load_model() -> CodeModel {
        let model_path = workspace_root().join("models/code-1536");
        let model =
            CodeModel::new(&model_path, SingleModelConfig::default()).expect("Failed to create");
        model.load().await.expect("Failed to load model");
        model
    }

    #[tokio::test]
    async fn test_edge_case_1_empty_code_content() {
        let model = create_and_load_model().await;

        // Empty string should error on ModelInput::code()
        let result = ModelInput::code("", "rust");
        assert!(
            result.is_err(),
            "Empty code string should error on ModelInput::code"
        );

        // Test with whitespace code
        let input = ModelInput::code(" ", "rust").expect("Whitespace input should work");
        let result = model.embed(&input).await;

        assert!(result.is_ok(), "Whitespace input should not error");
        let embedding = result.unwrap();
        assert_eq!(embedding.vector.len(), CODE_PROJECTED_DIMENSION);
    }
}
