//! Batch and language-specific tests for the CodeModel.

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::models::pretrained::code::{CodeModel, CODE_PROJECTED_DIMENSION};
    use crate::traits::SingleModelConfig;
    use crate::types::{ModelId, ModelInput};

    fn workspace_root() -> PathBuf {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(manifest_dir)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    fn create_test_model() -> CodeModel {
        let model_path = workspace_root().join("models/code-1536");
        CodeModel::new(&model_path, SingleModelConfig::default())
            .expect("Failed to create CodeModel")
    }

    async fn create_and_load_model() -> CodeModel {
        let model = create_test_model();
        model.load().await.expect("Failed to load model");
        model
    }

    #[tokio::test]
    async fn test_embed_batch_multiple_inputs() {
        let model = create_and_load_model().await;
        let inputs = vec![
            ModelInput::code("fn one() {}", "rust").expect("Input"),
            ModelInput::code("fn two() {}", "rust").expect("Input"),
            ModelInput::code("fn three() {}", "rust").expect("Input"),
        ];
        let embeddings = model.embed_batch(&inputs).await.expect("Batch embed");
        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.vector.len(), CODE_PROJECTED_DIMENSION);
            assert_eq!(emb.model_id, ModelId::Code);
        }
    }
}
