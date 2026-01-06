//! Model loading operations for ModelRegistry.
//!
//! This module contains get_model and load_model implementations.

use std::sync::Arc;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::traits::{EmbeddingModel, SingleModelConfig};
use crate::types::ModelId;

use super::core::ModelRegistry;

impl ModelRegistry {
    /// Get model, loading lazily if needed.
    ///
    /// Thread-safe: concurrent calls for same model only load once.
    /// The per-model semaphore serializes concurrent load requests.
    ///
    /// # Arguments
    /// * `model_id` - The model to retrieve
    ///
    /// # Returns
    /// - `Ok(Arc<dyn EmbeddingModel>)` - The loaded model
    /// - `Err(EmbeddingError)` if load fails
    pub async fn get_model(&self, model_id: ModelId) -> EmbeddingResult<Arc<dyn EmbeddingModel>> {
        // Fast path: check if already loaded
        {
            let models = self.models.read().await;
            if let Some(model) = models.get(&model_id) {
                // Cache hit
                let mut stats = self.stats.write().await;
                stats.cache_hits += 1;

                if self.config.enable_debug_logging {
                    tracing::debug!(model_id = ?model_id, "Cache hit");
                }

                return Ok(Arc::clone(model));
            }
        }

        // Slow path: need to load
        self.load_model(model_id).await?;

        // Now get from cache
        let models = self.models.read().await;
        models.get(&model_id).cloned().ok_or_else(|| {
            // This should never happen if load_model succeeded
            tracing::error!(model_id = ?model_id, "Model missing after successful load");
            EmbeddingError::InternalError {
                message: format!("Model {:?} missing after load", model_id),
            }
        })
    }

    /// Explicitly load a model.
    ///
    /// Uses per-model semaphore to prevent concurrent loads of the same model.
    /// Checks memory budget before loading.
    ///
    /// # Arguments
    /// * `model_id` - The model to load
    ///
    /// # Returns
    /// - `Ok(())` if load succeeds
    /// - `Err(EmbeddingError::MemoryBudgetExceeded)` if insufficient memory
    /// - `Err(EmbeddingError::ModelLoadError)` if factory/load fails
    /// - `Err(EmbeddingError::ModelAlreadyLoaded)` if already loaded
    pub async fn load_model(&self, model_id: ModelId) -> EmbeddingResult<()> {
        // Get the per-model lock
        let lock = self
            .loading_locks
            .get(&model_id)
            .ok_or_else(|| EmbeddingError::ModelNotFound { model_id })?;

        // Acquire per-model semaphore (serializes concurrent load requests)
        let _permit = lock
            .acquire()
            .await
            .map_err(|_| EmbeddingError::InternalError {
                message: format!("Semaphore closed for model {:?}", model_id),
            })?;

        // Double-check if already loaded (another task may have loaded while we waited)
        {
            let models = self.models.read().await;
            if models.contains_key(&model_id) {
                // Already loaded - count as cache hit
                let mut stats = self.stats.write().await;
                stats.cache_hits += 1;

                if self.config.enable_debug_logging {
                    tracing::debug!(model_id = ?model_id, "Model already loaded (race avoided)");
                }

                return Ok(());
            }
        }

        // Get memory estimate
        let memory_needed = self.factory.estimate_memory(model_id);

        // Check memory budget
        {
            let tracker = self.memory_tracker.read().await;
            if !tracker.can_allocate(memory_needed) {
                let mut stats = self.stats.write().await;
                stats.load_failures += 1;

                tracing::warn!(
                    model_id = ?model_id,
                    required_bytes = memory_needed,
                    available_bytes = tracker.remaining(),
                    "Memory budget check FAILED"
                );

                return Err(EmbeddingError::MemoryBudgetExceeded {
                    requested_bytes: memory_needed,
                    available_bytes: tracker.remaining(),
                    budget_bytes: tracker.budget(),
                });
            }
        }

        // Create model via factory
        let config = SingleModelConfig::default();
        let model = match self.factory.create_model(model_id, &config) {
            Ok(m) => m,
            Err(e) => {
                let mut stats = self.stats.write().await;
                stats.load_failures += 1;
                tracing::error!(
                    model_id = ?model_id,
                    error = ?e,
                    "Model creation FAILED"
                );
                return Err(e);
            }
        };

        // Convert to Arc<dyn EmbeddingModel>
        let model: Arc<dyn EmbeddingModel> = Arc::from(model);

        // Allocate memory
        {
            let mut tracker = self.memory_tracker.write().await;
            if let Err(e) = tracker.allocate(model_id, memory_needed) {
                let mut stats = self.stats.write().await;
                stats.load_failures += 1;
                return Err(e);
            }
        }

        // Insert into cache
        {
            let mut models = self.models.write().await;
            models.insert(model_id, model);
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.load_count += 1;
        }

        tracing::info!(
            model_id = ?model_id,
            memory_bytes = memory_needed,
            "Model loaded successfully"
        );

        Ok(())
    }
}
