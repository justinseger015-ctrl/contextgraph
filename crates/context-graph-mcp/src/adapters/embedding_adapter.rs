//! Adapter bridging embeddings crate to core trait interface.
//!
//! This module provides `EmbeddingProviderAdapter` which wraps the
//! `context_graph_embeddings::FusedEmbeddingProvider` and implements the async
//! `context_graph_core::traits::EmbeddingProvider` trait.
//!
//! # Architecture
//!
//! The embeddings crate provides `FusedEmbeddingProvider` which produces 1536D
//! vectors via GPU-accelerated inference. This adapter bridges to the core trait,
//! converting `EmbeddingResult<Vec<f32>>` to `CoreResult<EmbeddingOutput>`.
//!
//! # Error Mapping
//!
//! | Embeddings Error | Core Error |
//! |-----------------|------------|
//! | `EmbeddingError::*` | `CoreError::Embedding(message)` |
//!
//! # Example
//!
//! ```ignore
//! use context_graph_mcp::adapters::EmbeddingProviderAdapter;
//! use std::path::Path;
//!
//! // Create with default configuration
//! let adapter = EmbeddingProviderAdapter::with_defaults(Path::new("models/e5-large-v2"))?;
//!
//! // Initialize (loads model weights)
//! adapter.initialize().await?;
//!
//! // Generate embedding
//! let output = adapter.embed("Hello, world!").await?;
//! assert_eq!(output.dimensions, 1536);
//! ```

use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use context_graph_core::error::{CoreError, CoreResult};
use context_graph_core::traits::{EmbeddingOutput, EmbeddingProvider as CoreEmbeddingProvider};

use context_graph_embeddings::{
    EmbeddingError, EmbeddingProvider as EmbeddingsProvider, FusedEmbeddingProvider,
    FusedProviderConfig,
};

/// Adapter bridging FusedEmbeddingProvider to core EmbeddingProvider trait.
///
/// Wraps the GPU-accelerated `FusedEmbeddingProvider` from the embeddings crate
/// and implements the `context_graph_core::traits::EmbeddingProvider` trait.
///
/// # Thread Safety
///
/// Uses `Arc<FusedEmbeddingProvider>` for shared ownership across async tasks.
/// The inner provider is `Send + Sync`.
///
/// # Performance
///
/// - Single embed: <10ms (constitution.yaml requirement)
/// - Batch embed (64 items): <50ms (constitution.yaml requirement)
/// - Output dimension: 1536D (OpenAI ada-002 compatible)
#[derive(Clone)]
pub struct EmbeddingProviderAdapter {
    /// The wrapped fused embedding provider
    inner: Arc<FusedEmbeddingProvider>,
}

impl EmbeddingProviderAdapter {
    /// Create adapter from an existing FusedEmbeddingProvider.
    ///
    /// # Arguments
    /// * `provider` - The fused embedding provider to wrap
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_embeddings::FusedEmbeddingProvider;
    /// use context_graph_mcp::adapters::EmbeddingProviderAdapter;
    ///
    /// let provider = FusedEmbeddingProvider::new()?;
    /// let adapter = EmbeddingProviderAdapter::new(provider);
    /// ```
    pub fn new(provider: FusedEmbeddingProvider) -> Self {
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create adapter with custom configuration.
    ///
    /// # Arguments
    /// * `config` - Configuration for the fused provider
    ///
    /// # Errors
    /// Returns `CoreError::Embedding` if provider creation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_embeddings::FusedProviderConfig;
    /// use context_graph_mcp::adapters::EmbeddingProviderAdapter;
    ///
    /// let config = FusedProviderConfig::default()
    ///     .with_model_path("/path/to/models")
    ///     .with_batch_size(64);
    /// let adapter = EmbeddingProviderAdapter::with_config(config)?;
    /// ```
    pub fn with_config(config: FusedProviderConfig) -> CoreResult<Self> {
        let provider =
            FusedEmbeddingProvider::with_config(config).map_err(|e| map_embedding_error(&e))?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Create adapter with default configuration using the given model path.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model weights directory
    ///
    /// # Errors
    /// Returns `CoreError::Embedding` if provider creation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_mcp::adapters::EmbeddingProviderAdapter;
    /// use std::path::Path;
    ///
    /// let adapter = EmbeddingProviderAdapter::with_defaults(Path::new("models/e5-large-v2"))?;
    /// ```
    pub fn with_defaults(model_path: &Path) -> CoreResult<Self> {
        let config = FusedProviderConfig::default().with_model_path(model_path);
        Self::with_config(config)
    }

    /// Initialize the provider (load model weights).
    ///
    /// Must be called before generating embeddings. Loads the SemanticModel
    /// weights and initializes the projection layer on GPU.
    ///
    /// # Errors
    /// Returns `CoreError::Embedding` if initialization fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let adapter = EmbeddingProviderAdapter::with_defaults(model_path)?;
    /// adapter.initialize().await?;
    /// assert!(adapter.is_ready());
    /// ```
    pub async fn initialize(&self) -> CoreResult<()> {
        self.inner
            .initialize()
            .await
            .map_err(|e| map_embedding_error(&e))
    }

    /// Get the inner provider reference.
    ///
    /// Useful for accessing provider-specific functionality.
    pub fn inner(&self) -> &FusedEmbeddingProvider {
        &self.inner
    }
}

impl std::fmt::Debug for EmbeddingProviderAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingProviderAdapter")
            .field("model_id", &self.inner.model_name())
            .field("dimensions", &self.inner.dimension())
            .field("is_ready", &self.inner.is_ready())
            .finish()
    }
}

#[async_trait]
impl CoreEmbeddingProvider for EmbeddingProviderAdapter {
    /// Generate an embedding for a single piece of content.
    ///
    /// # Arguments
    /// * `content` - The text content to embed
    ///
    /// # Returns
    /// An [`EmbeddingOutput`] containing the 1536D embedding vector.
    ///
    /// # Errors
    /// Returns `CoreError::Embedding` if:
    /// - Provider is not initialized
    /// - Content is empty
    /// - Embedding generation fails
    ///
    /// # Performance
    /// Target latency: <10ms (constitution.yaml requirement)
    async fn embed(&self, content: &str) -> CoreResult<EmbeddingOutput> {
        let start = Instant::now();

        // Call the embeddings crate's embed method
        let vector = self
            .inner
            .embed(content)
            .await
            .map_err(|e| map_embedding_error(&e))?;

        let latency = start.elapsed();

        Ok(EmbeddingOutput::new(
            vector,
            self.inner.model_name().to_string(),
            latency,
        ))
    }

    /// Generate embeddings for multiple pieces of content in batch.
    ///
    /// Batch processing is more efficient than individual calls,
    /// amortizing GPU overhead across the batch.
    ///
    /// # Arguments
    /// * `contents` - Slice of text content to embed
    ///
    /// # Returns
    /// A vector of [`EmbeddingOutput`] in the same order as input contents.
    ///
    /// # Errors
    /// Returns `CoreError::Embedding` if:
    /// - Provider is not initialized
    /// - Any content is empty
    /// - Batch processing fails
    ///
    /// # Performance
    /// Target latency for 64 items: <50ms (constitution.yaml requirement)
    async fn embed_batch(&self, contents: &[String]) -> CoreResult<Vec<EmbeddingOutput>> {
        let start = Instant::now();

        // Convert &[String] to &[&str] for the embeddings crate
        let content_refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();

        // Call the embeddings crate's batch embed method
        let vectors = self
            .inner
            .embed_batch(&content_refs)
            .await
            .map_err(|e| map_embedding_error(&e))?;

        let total_latency = start.elapsed();
        let per_item_latency = total_latency / contents.len().max(1) as u32;
        let model_id = self.inner.model_name().to_string();

        // Convert Vec<Vec<f32>> to Vec<EmbeddingOutput>
        let outputs = vectors
            .into_iter()
            .map(|vector| EmbeddingOutput::new(vector, model_id.clone(), per_item_latency))
            .collect();

        Ok(outputs)
    }

    /// Get the dimensionality of embeddings produced by this provider.
    ///
    /// # Returns
    /// The number of dimensions (1536 for FusedEmbeddingProvider).
    fn dimensions(&self) -> usize {
        self.inner.dimension()
    }

    /// Get the model identifier for this provider.
    ///
    /// # Returns
    /// The model ID string (e.g., "fused-embedding-v1").
    fn model_id(&self) -> &str {
        self.inner.model_name()
    }

    /// Check if the provider is ready to generate embeddings.
    ///
    /// # Returns
    /// `true` if the provider has been initialized and is ready for inference.
    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }
}

/// Map an EmbeddingError to a CoreError.
///
/// All embedding errors are mapped to `CoreError::Embedding` with a descriptive message.
fn map_embedding_error(err: &EmbeddingError) -> CoreError {
    CoreError::Embedding(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that adapter construction succeeds with default config.
    /// Note: This test may fail without GPU/model files in CI.
    #[test]
    fn test_adapter_with_config_creation() {
        // Use a non-existent path to test error handling
        let config = FusedProviderConfig::default().with_model_path("/nonexistent/path");
        let result = EmbeddingProviderAdapter::with_config(config);

        // Should fail because model path doesn't exist
        // This verifies error mapping works
        match result {
            Ok(_) => {
                // If it succeeds (unlikely without real model), that's also fine
            }
            Err(CoreError::Embedding(msg)) => {
                assert!(!msg.is_empty(), "Error message should not be empty");
            }
            Err(other) => {
                panic!("Expected CoreError::Embedding, got {:?}", other);
            }
        }
    }

    /// Test that error mapping produces CoreError::Embedding.
    #[test]
    fn test_error_mapping() {
        let embedding_error = EmbeddingError::EmptyInput;
        let core_error = map_embedding_error(&embedding_error);

        match core_error {
            CoreError::Embedding(msg) => {
                assert!(
                    msg.contains("Empty input"),
                    "Error message should describe the issue"
                );
            }
            _ => panic!("Expected CoreError::Embedding"),
        }
    }

    /// Test error mapping for dimension mismatch.
    #[test]
    fn test_error_mapping_dimension_mismatch() {
        let embedding_error = EmbeddingError::InvalidDimension {
            expected: 1536,
            actual: 768,
        };
        let core_error = map_embedding_error(&embedding_error);

        match core_error {
            CoreError::Embedding(msg) => {
                assert!(msg.contains("1536"), "Should contain expected dimension");
                assert!(msg.contains("768"), "Should contain actual dimension");
            }
            _ => panic!("Expected CoreError::Embedding"),
        }
    }

    /// Test error mapping for GPU errors.
    #[test]
    fn test_error_mapping_gpu_error() {
        let embedding_error = EmbeddingError::GpuError {
            message: "CUDA out of memory".to_string(),
        };
        let core_error = map_embedding_error(&embedding_error);

        match core_error {
            CoreError::Embedding(msg) => {
                assert!(
                    msg.contains("CUDA out of memory"),
                    "Should preserve error message"
                );
            }
            _ => panic!("Expected CoreError::Embedding"),
        }
    }

    /// Test error mapping for not initialized.
    #[test]
    fn test_error_mapping_not_initialized() {
        let embedding_error = EmbeddingError::NotInitialized {
            model_id: context_graph_embeddings::ModelId::Semantic,
        };
        let core_error = map_embedding_error(&embedding_error);

        match core_error {
            CoreError::Embedding(msg) => {
                assert!(
                    msg.contains("not initialized"),
                    "Should indicate initialization issue"
                );
            }
            _ => panic!("Expected CoreError::Embedding"),
        }
    }

    /// Test Debug implementation.
    #[test]
    fn test_debug_formatting() {
        // Create a minimal config for testing debug output
        let config = FusedProviderConfig::default().with_model_path("/test/path");

        // Try to create adapter - may fail without real model
        if let Ok(adapter) = EmbeddingProviderAdapter::with_config(config) {
            let debug_str = format!("{:?}", adapter);
            assert!(
                debug_str.contains("EmbeddingProviderAdapter"),
                "Debug output should contain type name"
            );
        }
    }

    // =========================================================================
    // Integration tests requiring GPU/model files
    // =========================================================================

    #[cfg(feature = "integration")]
    mod integration {
        use super::*;
        use std::path::PathBuf;

        fn model_path() -> PathBuf {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("models/e5-large-v2")
        }

        #[tokio::test]
        async fn test_adapter_initialization() {
            let adapter = EmbeddingProviderAdapter::with_defaults(&model_path())
                .expect("Adapter creation should succeed");

            assert!(!adapter.is_ready(), "Should not be ready before init");

            adapter
                .initialize()
                .await
                .expect("Initialization should succeed");

            assert!(adapter.is_ready(), "Should be ready after init");
        }

        #[tokio::test]
        async fn test_embed_produces_1536d() {
            let adapter = EmbeddingProviderAdapter::with_defaults(&model_path())
                .expect("Adapter creation should succeed");
            adapter
                .initialize()
                .await
                .expect("Initialization should succeed");

            let output = adapter
                .embed("Hello, world!")
                .await
                .expect("Embed should succeed");

            assert_eq!(output.dimensions, 1536, "Should produce 1536D embeddings");
            assert_eq!(
                output.vector.len(),
                1536,
                "Vector length should match dimensions"
            );
            assert!(output.latency.as_millis() < 100, "Should complete quickly");
        }

        #[tokio::test]
        async fn test_embed_batch_produces_correct_count() {
            let adapter = EmbeddingProviderAdapter::with_defaults(&model_path())
                .expect("Adapter creation should succeed");
            adapter
                .initialize()
                .await
                .expect("Initialization should succeed");

            let contents = vec![
                "First document".to_string(),
                "Second document".to_string(),
                "Third document".to_string(),
            ];

            let outputs = adapter
                .embed_batch(&contents)
                .await
                .expect("Batch embed should succeed");

            assert_eq!(outputs.len(), 3, "Should produce 3 embeddings");
            for output in &outputs {
                assert_eq!(output.dimensions, 1536, "Each should be 1536D");
            }
        }

        #[tokio::test]
        async fn test_embed_latency_within_target() {
            let adapter = EmbeddingProviderAdapter::with_defaults(&model_path())
                .expect("Adapter creation should succeed");
            adapter
                .initialize()
                .await
                .expect("Initialization should succeed");

            // Warm up
            let _ = adapter.embed("warmup").await;

            // Measure
            let start = std::time::Instant::now();
            let _ = adapter.embed("test input").await;
            let latency = start.elapsed();

            assert!(
                latency.as_millis() < 10,
                "Single embed should be <10ms, got {}ms",
                latency.as_millis()
            );
        }
    }
}
