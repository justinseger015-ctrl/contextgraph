//! Core types for the 12-model embedding pipeline.

mod concatenated;
mod embedding;
mod fused;
mod model_id;
pub mod dimensions;

pub use concatenated::ConcatenatedEmbedding;
pub use embedding::ModelEmbedding;
pub use fused::{AuxiliaryEmbeddingData, FusedEmbedding};
pub use model_id::ModelId;
pub use model_id::TokenizerFamily;
