//! EmbeddingModel trait implementation for EntityModel.

use std::sync::atomic::Ordering;

use async_trait::async_trait;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::traits::EmbeddingModel;
use crate::types::{InputType, ModelEmbedding, ModelId, ModelInput};

use super::types::{EntityModel, ModelState};

#[async_trait]
impl EmbeddingModel for EntityModel {
    fn model_id(&self) -> ModelId {
        ModelId::Entity
    }

    fn supported_input_types(&self) -> &[InputType] {
        &[InputType::Text]
    }

    fn is_initialized(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    async fn load(&self) -> EmbeddingResult<()> {
        EntityModel::load(self).await
    }

    async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
        // 1. Check initialized
        if !self.is_initialized() {
            return Err(EmbeddingError::NotInitialized {
                model_id: ModelId::Entity,
            });
        }

        // 2. Validate input type
        self.validate_input(input)?;

        // 3. Extract text content
        let content = Self::extract_content(input)?;

        let start = std::time::Instant::now();

        // 4. Get loaded weights and tokenizer
        let state = self
            .model_state
            .read()
            .map_err(|e| EmbeddingError::InternalError {
                message: format!("EntityModel failed to acquire read lock: {}", e),
            })?;

        let (weights, tokenizer) = match &*state {
            ModelState::Loaded { weights, tokenizer } => (weights, tokenizer),
            _ => {
                return Err(EmbeddingError::NotInitialized {
                    model_id: ModelId::Entity,
                });
            }
        };

        // 5. Run GPU-accelerated BERT forward pass
        let vector = Self::gpu_forward(&content, weights, tokenizer)?;

        let latency_us = start.elapsed().as_micros() as u64;

        Ok(ModelEmbedding::new(ModelId::Entity, vector, latency_us))
    }
}
