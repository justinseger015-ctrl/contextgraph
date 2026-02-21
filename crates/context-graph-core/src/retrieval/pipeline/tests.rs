//! Tests for the teleological retrieval pipeline.

use std::sync::Arc;

use crate::retrieval::InMemoryMultiEmbeddingExecutor;
use crate::stubs::{InMemoryTeleologicalStore, StubMultiArrayProvider};

use super::super::teleological_query::TeleologicalQuery;
use super::{DefaultTeleologicalPipeline, TeleologicalRetrievalPipeline};
use crate::error::CoreError;

async fn create_test_pipeline(
) -> DefaultTeleologicalPipeline<InMemoryMultiEmbeddingExecutor, InMemoryTeleologicalStore> {
    let store = InMemoryTeleologicalStore::new();
    let provider = StubMultiArrayProvider::new();

    let store_arc = Arc::new(store);

    let executor = Arc::new(InMemoryMultiEmbeddingExecutor::with_arcs(
        store_arc.clone(),
        Arc::new(provider),
    ));

    DefaultTeleologicalPipeline::new(executor, store_arc)
}

#[tokio::test]
async fn test_pipeline_creation() {
    let pipeline = create_test_pipeline().await;
    let health = pipeline.health_check().await.unwrap();

    assert_eq!(health.spaces_available, 13);
}

#[tokio::test]
async fn test_execute_basic_query() {
    let pipeline = create_test_pipeline().await;

    let query = TeleologicalQuery::from_text("authentication patterns");
    let result = pipeline.execute(&query).await.unwrap();

    assert!(result.total_time.as_millis() < 1000);
    assert!(result.spaces_searched > 0);
}

#[tokio::test]
async fn test_execute_fails_empty_query() {
    let pipeline = create_test_pipeline().await;

    let query = TeleologicalQuery::default();
    let result = pipeline.execute(&query).await;

    assert!(result.is_err());
    match result {
        Err(CoreError::ValidationError { field, .. }) => {
            assert_eq!(field, "text");
        }
        _ => panic!("Expected ValidationError"),
    }
}
