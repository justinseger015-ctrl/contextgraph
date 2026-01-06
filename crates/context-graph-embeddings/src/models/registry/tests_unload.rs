//! Unload and get model tests for ModelRegistry.

use std::sync::Arc;

use crate::error::EmbeddingError;
use crate::types::{ModelId, ModelInput};

use super::config::ModelRegistryConfig;
use super::core::ModelRegistry;
use super::tests::TestFactory;

// =========================================================================
// UNLOAD MODEL TESTS (4 tests)
// =========================================================================

#[tokio::test]
async fn test_unload_model_success() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = ModelRegistry::new(config, factory).await.unwrap();
    registry.load_model(ModelId::Semantic).await.unwrap();

    let result = registry.unload_model(ModelId::Semantic).await;

    assert!(result.is_ok());
    assert!(!registry.is_loaded(ModelId::Semantic).await);
}

#[tokio::test]
async fn test_unload_model_frees_memory() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = ModelRegistry::new(config, factory).await.unwrap();
    registry.load_model(ModelId::Semantic).await.unwrap();

    let before = registry.total_memory_usage().await;
    assert!(before > 0);

    registry.unload_model(ModelId::Semantic).await.unwrap();

    assert_eq!(registry.total_memory_usage().await, 0);
}

#[tokio::test]
async fn test_unload_model_updates_stats() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = ModelRegistry::new(config, factory).await.unwrap();
    registry.load_model(ModelId::Code).await.unwrap();
    registry.unload_model(ModelId::Code).await.unwrap();

    let stats = registry.stats().await;
    assert_eq!(stats.unload_count, 1);
    assert_eq!(stats.loaded_count, 0);
}

#[tokio::test]
async fn test_unload_model_fails_when_not_loaded() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = ModelRegistry::new(config, factory).await.unwrap();
    let result = registry.unload_model(ModelId::Semantic).await;

    assert!(result.is_err());
    match result {
        Err(EmbeddingError::ModelNotLoaded { model_id }) => {
            assert_eq!(model_id, ModelId::Semantic);
        }
        _ => panic!("Expected ModelNotLoaded error"),
    }
}

// =========================================================================
// GET MODEL TESTS (4 tests)
// =========================================================================

#[tokio::test]
async fn test_get_model_lazy_loads() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = ModelRegistry::new(config, factory.clone()).await.unwrap();

    assert!(!registry.is_loaded(ModelId::Semantic).await);

    let model = registry.get_model(ModelId::Semantic).await.unwrap();

    assert!(registry.is_loaded(ModelId::Semantic).await);
    assert_eq!(model.model_id(), ModelId::Semantic);
    assert_eq!(factory.create_count(), 1);
}

#[tokio::test]
async fn test_get_model_returns_cached() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = ModelRegistry::new(config, factory.clone()).await.unwrap();

    let model1 = registry.get_model(ModelId::Semantic).await.unwrap();
    let model2 = registry.get_model(ModelId::Semantic).await.unwrap();

    // Should be same Arc instance
    assert!(Arc::ptr_eq(&model1, &model2));
    assert_eq!(factory.create_count(), 1);

    let stats = registry.stats().await;
    assert_eq!(stats.cache_hits, 1);
}

#[tokio::test]
async fn test_get_model_records_cache_hit() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = ModelRegistry::new(config, factory).await.unwrap();
    registry.load_model(ModelId::Code).await.unwrap();

    // Get the already-loaded model
    let _model = registry.get_model(ModelId::Code).await.unwrap();

    let stats = registry.stats().await;
    assert_eq!(stats.cache_hits, 1);
}

#[tokio::test]
async fn test_get_model_usable_for_embedding() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = ModelRegistry::new(config, factory).await.unwrap();
    let model = registry.get_model(ModelId::Semantic).await.unwrap();

    let input = ModelInput::text("Hello, world!").unwrap();
    let embedding = model.embed(&input).await.unwrap();

    assert_eq!(embedding.model_id, ModelId::Semantic);
    assert_eq!(embedding.dimension(), 1024);
}
