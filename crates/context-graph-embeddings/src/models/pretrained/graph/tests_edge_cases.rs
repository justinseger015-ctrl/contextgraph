//! Edge case tests for GraphModel.

#[cfg(test)]
mod tests {
    use crate::models::pretrained::graph::GraphModel;
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

    async fn create_and_load_model() -> GraphModel {
        let model_path = workspace_root().join("models/semantic");
        let model = GraphModel::new(&model_path, SingleModelConfig::default())
            .expect("Failed to create GraphModel");
        model.load().await.expect("Failed to load model");
        model
    }

    #[tokio::test]
    async fn test_edge_case_1_empty_text_content() {
        let model = create_and_load_model().await;

        // Empty string should error on ModelInput::text()
        let result = ModelInput::text("");
        assert!(
            result.is_err(),
            "Empty text string should error on ModelInput::text"
        );

        // Test with whitespace text
        let input = ModelInput::text(" ").expect("Whitespace input should work");
        let result = model.embed(&input).await;

        assert!(result.is_ok(), "Whitespace input should not error");
        let embedding = result.unwrap();
        assert_eq!(embedding.vector.len(), 1024);
    }
}
