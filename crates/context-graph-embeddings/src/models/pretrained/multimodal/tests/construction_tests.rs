//! Construction and trait implementation tests for MultimodalModel.

use super::create_test_model;

use crate::traits::EmbeddingModel;

// ==================== Construction Tests ====================

#[test]
fn test_new_creates_unloaded_model() {
    let model = create_test_model();
    assert!(!model.is_initialized());
}
