//! FuseMoE fusion layer components.
//!
//! This module implements the Mixture-of-Experts fusion for combining
//! 12 model embeddings into a unified 1536D representation.
//!
//! # Components
//!
//! - [`GatingNetwork`]: Routes 8320D concatenated embeddings to 8 experts
//! - [`LayerNorm`]: Input normalization for stability
//! - [`Linear`]: Projection layer for the gating network
//! - [`Expert`]: Single expert FFN (8320 -> 4096 -> 1536)
//! - [`ExpertPool`]: Pool of 8 experts with top-k routing
//! - [`Activation`]: Activation functions for experts (GELU, ReLU, SiLU)
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_embeddings::fusion::{GatingNetwork, ExpertPool};
//! use context_graph_embeddings::config::FusionConfig;
//! use context_graph_embeddings::types::dimensions::TOP_K_EXPERTS;
//!
//! let config = FusionConfig::default();
//! let gating = GatingNetwork::new(&config)?;
//! let experts = ExpertPool::new(&config)?;
//!
//! // Forward pass with batch of 1
//! let input = vec![0.5f32; 8320];
//! let probs = gating.forward(&input, 1)?;
//! let (indices, weights) = gating.select_top_k(&probs, 1, TOP_K_EXPERTS)?;
//! let output = experts.forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)?;
//! assert_eq!(output.len(), 1536);  // Fused output
//! ```

pub mod experts;
pub mod gating;

pub use experts::{Activation, Expert, ExpertPool};
pub use gating::{GatingNetwork, LayerNorm, Linear};
