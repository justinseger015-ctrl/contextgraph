//! Tests for the sparse embedding model.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use std::path::PathBuf;

    use crate::models::pretrained::sparse::{SparseModel, SparseVector, SPARSE_VOCAB_SIZE};
    use crate::traits::SingleModelConfig;

    /// Get the workspace root directory for test model paths.
    fn workspace_root() -> PathBuf {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(manifest_dir)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    fn create_test_model() -> SparseModel {
        let model_path = workspace_root().join("models/sparse");
        SparseModel::new(&model_path, SingleModelConfig::default())
            .expect("Failed to create SparseModel")
    }

    // ==================== Construction Tests ====================

    #[test]
    fn test_new_creates_unloaded_model() {
        let model = create_test_model();
        assert!(!model.is_initialized());
    }

    // ==================== Sparse Vector Tests ====================

    #[test]
    fn test_sparse_vector_new() {
        let indices = vec![10, 100, 500];
        let weights = vec![0.5, 0.3, 0.8];
        let sparse = SparseVector::new(indices.clone(), weights.clone());

        assert_eq!(sparse.indices, indices);
        assert_eq!(sparse.weights, weights);
        assert_eq!(sparse.dimension, SPARSE_VOCAB_SIZE);
        assert_eq!(sparse.dimension, 30522);
    }
}
