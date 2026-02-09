//! Longformer weight structures.
//!
//! This module contains the tensor weight structures for embedding layers,
//! attention mechanisms, feed-forward networks, and complete model weights.
//!
//! # Asymmetric Causal Projections
//!
//! The `CausalProjectionWeights` struct provides learned projection matrices
//! for creating asymmetric cause/effect embeddings. These are initialized as
//! perturbed identity matrices to create immediate asymmetry without training.

use candle_core::{DType, Device, Tensor};
use rand::Rng;

use crate::error::{EmbeddingError, EmbeddingResult};

use super::config::LongformerConfig;

/// Seed for causal projection initialization (deterministic).
pub const CAUSAL_PROJECTION_SEED: u64 = 0xCA05A1;

/// Longformer embedding weights.
#[derive(Debug)]
pub struct LongformerEmbeddingWeights {
    /// Word embeddings: [vocab_size, hidden_size]
    pub word_embeddings: Tensor,
    /// Position embeddings: [max_position, hidden_size]
    pub position_embeddings: Tensor,
    /// Token type embeddings: [type_vocab_size, hidden_size]
    pub token_type_embeddings: Tensor,
    /// LayerNorm weight: [hidden_size]
    pub layer_norm_weight: Tensor,
    /// LayerNorm bias: [hidden_size]
    pub layer_norm_bias: Tensor,
}

/// Longformer self-attention weights (includes global attention).
#[derive(Debug)]
pub struct LongformerAttentionWeights {
    /// Query projection: [hidden_size, hidden_size]
    pub query_weight: Tensor,
    /// Query bias: [hidden_size]
    pub query_bias: Tensor,
    /// Key projection: [hidden_size, hidden_size]
    pub key_weight: Tensor,
    /// Key bias: [hidden_size]
    pub key_bias: Tensor,
    /// Value projection: [hidden_size, hidden_size]
    pub value_weight: Tensor,
    /// Value bias: [hidden_size]
    pub value_bias: Tensor,
    /// Output projection: [hidden_size, hidden_size]
    pub output_weight: Tensor,
    /// Output bias: [hidden_size]
    pub output_bias: Tensor,
    /// Attention output LayerNorm weight: [hidden_size]
    pub layer_norm_weight: Tensor,
    /// Attention output LayerNorm bias: [hidden_size]
    pub layer_norm_bias: Tensor,
    // Note: Global attention weights omitted for simplicity in initial implementation.
    // For full sliding window + global attention, we would include:
    // query_global_weight, key_global_weight, value_global_weight, etc.
}

/// Longformer FFN weights.
#[derive(Debug)]
pub struct LongformerFfnWeights {
    /// Intermediate projection: [intermediate_size, hidden_size]
    pub intermediate_weight: Tensor,
    /// Intermediate bias: [intermediate_size]
    pub intermediate_bias: Tensor,
    /// Output projection: [hidden_size, intermediate_size]
    pub output_weight: Tensor,
    /// Output bias: [hidden_size]
    pub output_bias: Tensor,
    /// Output LayerNorm weight: [hidden_size]
    pub layer_norm_weight: Tensor,
    /// Output LayerNorm bias: [hidden_size]
    pub layer_norm_bias: Tensor,
}

/// Longformer encoder layer weights.
#[derive(Debug)]
pub struct LongformerEncoderLayerWeights {
    /// Self-attention weights.
    pub attention: LongformerAttentionWeights,
    /// FFN weights.
    pub ffn: LongformerFfnWeights,
}

/// Complete Longformer model weights.
#[derive(Debug)]
pub struct LongformerWeights {
    /// Model configuration.
    pub config: LongformerConfig,
    /// Embedding layer weights.
    pub embeddings: LongformerEmbeddingWeights,
    /// Encoder layer weights.
    pub encoder_layers: Vec<LongformerEncoderLayerWeights>,
    /// GPU device reference.
    pub(crate) device: &'static Device,
}

// =============================================================================
// Causal Projection Weights for Asymmetric Embeddings
// =============================================================================

/// Standard deviation for initializing projection weight perturbations.
const PROJECTION_INIT_STD: f64 = 0.02;

/// Learned projection weights for asymmetric cause/effect embeddings.
///
/// These projections transform the base Longformer embedding into
/// cause-role and effect-role vectors. The projections are initialized
/// as perturbed identity matrices (I + N(0, 0.02)) to create immediate
/// asymmetry without requiring fine-tuning.
///
/// # Architecture
///
/// ```text
/// base_embedding [768D]
///        |
///    +---+---+
///    |       |
/// W_cause  W_effect
///    |       |
/// cause_vec effect_vec
/// [768D]    [768D]
/// ```
///
/// # Why Perturbed Identity?
///
/// 1. **Immediate asymmetry**: Different random perturbations create distinct
///    projections from the start
/// 2. **Preserved semantics**: Identity component ensures the base meaning is retained
/// 3. **No training required**: Works out-of-the-box without fine-tuning
/// 4. **Deterministic**: Same seed produces same weights across runs
#[derive(Debug)]
pub struct CausalProjectionWeights {
    /// Cause projection matrix: [hidden_size, hidden_size]
    pub cause_projection: Tensor,
    /// Cause projection bias: [hidden_size]
    pub cause_bias: Tensor,
    /// Effect projection matrix: [hidden_size, hidden_size]
    pub effect_projection: Tensor,
    /// Effect projection bias: [hidden_size]
    pub effect_bias: Tensor,
}

// =============================================================================
// Trainable Projection Weights (for fine-tuning)
// =============================================================================

/// Trainable projection weights that wrap Candle `Var` for autograd support.
///
/// Holds both trainable `Var` tensors (for gradient computation) and inference
/// `Tensor` views. During training, gradients flow through the `Var` tensors.
/// For inference, `as_inference()` returns the standard `CausalProjectionWeights`.
///
/// # Loading Priority
///
/// `trained/projection_v1.safetensors` > perturbed identity fallback
#[derive(Debug)]
pub struct TrainableProjection {
    /// Trainable cause projection [hidden_size, hidden_size].
    pub cause_projection_var: candle_core::Var,
    /// Trainable cause bias [hidden_size].
    pub cause_bias_var: candle_core::Var,
    /// Trainable effect projection [hidden_size, hidden_size].
    pub effect_projection_var: candle_core::Var,
    /// Trainable effect bias [hidden_size].
    pub effect_bias_var: candle_core::Var,
    /// Hidden size (768 for Longformer-base).
    pub hidden_size: usize,
}

impl TrainableProjection {
    /// Create from existing (static) projection weights by wrapping in Var.
    pub fn from_inference(weights: &CausalProjectionWeights) -> crate::error::EmbeddingResult<Self> {
        let hidden_size = weights.cause_projection.dim(0)
            .map_err(|e| crate::error::EmbeddingError::GpuError {
                message: format!("Failed to get projection dim: {}", e),
            })?;

        let cause_projection_var = candle_core::Var::from_tensor(&weights.cause_projection)
            .map_err(|e| crate::error::EmbeddingError::GpuError {
                message: format!("Failed to create trainable cause projection: {}", e),
            })?;
        let cause_bias_var = candle_core::Var::from_tensor(&weights.cause_bias)
            .map_err(|e| crate::error::EmbeddingError::GpuError {
                message: format!("Failed to create trainable cause bias: {}", e),
            })?;
        let effect_projection_var = candle_core::Var::from_tensor(&weights.effect_projection)
            .map_err(|e| crate::error::EmbeddingError::GpuError {
                message: format!("Failed to create trainable effect projection: {}", e),
            })?;
        let effect_bias_var = candle_core::Var::from_tensor(&weights.effect_bias)
            .map_err(|e| crate::error::EmbeddingError::GpuError {
                message: format!("Failed to create trainable effect bias: {}", e),
            })?;

        Ok(Self {
            cause_projection_var,
            cause_bias_var,
            effect_projection_var,
            effect_bias_var,
            hidden_size,
        })
    }

    /// Create a new trainable projection with perturbed identity initialization.
    pub fn new(hidden_size: usize, device: &Device) -> crate::error::EmbeddingResult<Self> {
        let weights = CausalProjectionWeights::initialize(hidden_size, device, CAUSAL_PROJECTION_SEED)?;
        Self::from_inference(&weights)
    }

    /// Create inference weights view (borrows tensor data from Var).
    pub fn as_inference(&self) -> CausalProjectionWeights {
        CausalProjectionWeights {
            cause_projection: self.cause_projection_var.as_tensor().clone(),
            cause_bias: self.cause_bias_var.as_tensor().clone(),
            effect_projection: self.effect_projection_var.as_tensor().clone(),
            effect_bias: self.effect_bias_var.as_tensor().clone(),
        }
    }

    /// Get all trainable Var tensors (for optimizer registration).
    pub fn trainable_vars(&self) -> Vec<&candle_core::Var> {
        vec![
            &self.cause_projection_var,
            &self.cause_bias_var,
            &self.effect_projection_var,
            &self.effect_bias_var,
        ]
    }

    /// Save trained weights to safetensors format.
    pub fn save_trained(&self, path: &std::path::Path) -> crate::error::EmbeddingResult<()> {
        use std::collections::HashMap;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "cause_projection".to_string(),
            self.cause_projection_var.as_tensor().clone(),
        );
        tensors.insert(
            "cause_bias".to_string(),
            self.cause_bias_var.as_tensor().clone(),
        );
        tensors.insert(
            "effect_projection".to_string(),
            self.effect_projection_var.as_tensor().clone(),
        );
        tensors.insert(
            "effect_bias".to_string(),
            self.effect_bias_var.as_tensor().clone(),
        );

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| crate::error::EmbeddingError::InternalError {
                message: format!("Failed to create checkpoint dir: {}", e),
            })?;
        }

        // Build views with owned data kept alive
        let tensor_data: Vec<(String, Vec<f32>, Vec<usize>)> = tensors
            .iter()
            .map(|(k, v)| {
                let data: Vec<f32> = v.flatten_all().unwrap().to_vec1().unwrap();
                let shape: Vec<usize> = v.shape().dims().to_vec();
                (k.clone(), data, shape)
            })
            .collect();

        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_data
            .iter()
            .map(|(k, data, shape)| {
                (
                    k.clone(),
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        shape.clone(),
                        bytemuck::cast_slice(data.as_slice()),
                    )
                    .unwrap(),
                )
            })
            .collect();

        safetensors::tensor::serialize_to_file(
            views.iter().map(|(k, v)| (k.clone(), v.clone())),
            &None::<HashMap<String, String>>,
            path,
        )
        .map_err(|e| crate::error::EmbeddingError::InternalError {
            message: format!("Failed to save trained weights: {}", e),
        })?;

        tracing::info!("Saved trained projection weights to {}", path.display());
        Ok(())
    }

    /// Load trained weights from safetensors format.
    pub fn load_trained(
        path: &std::path::Path,
        device: &Device,
    ) -> crate::error::EmbeddingResult<Self> {
        let data = std::fs::read(path).map_err(|e| crate::error::EmbeddingError::InternalError {
            message: format!("Failed to read checkpoint: {}", e),
        })?;

        let safetensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| crate::error::EmbeddingError::InternalError {
                message: format!("Failed to deserialize checkpoint: {}", e),
            })?;

        let load_tensor = |name: &str| -> crate::error::EmbeddingResult<Tensor> {
            let view = safetensors.tensor(name).map_err(|e| {
                crate::error::EmbeddingError::InternalError {
                    message: format!("Missing tensor '{}': {}", name, e),
                }
            })?;
            let shape: Vec<usize> = view.shape().to_vec();
            let float_data: &[f32] = bytemuck::cast_slice(view.data());
            Tensor::from_slice(float_data, shape, device).map_err(|e| {
                crate::error::EmbeddingError::GpuError {
                    message: format!("Failed to create tensor '{}': {}", name, e),
                }
            })
        };

        let cause_proj = load_tensor("cause_projection")?;
        let cause_bias = load_tensor("cause_bias")?;
        let effect_proj = load_tensor("effect_projection")?;
        let effect_bias = load_tensor("effect_bias")?;

        let hidden_size = cause_proj.dim(0).map_err(|e| {
            crate::error::EmbeddingError::GpuError {
                message: format!("Failed to get dim: {}", e),
            }
        })?;

        Ok(Self {
            cause_projection_var: candle_core::Var::from_tensor(&cause_proj).map_err(|e| {
                crate::error::EmbeddingError::GpuError {
                    message: format!("Var from tensor: {}", e),
                }
            })?,
            cause_bias_var: candle_core::Var::from_tensor(&cause_bias).map_err(|e| {
                crate::error::EmbeddingError::GpuError {
                    message: format!("Var from tensor: {}", e),
                }
            })?,
            effect_projection_var: candle_core::Var::from_tensor(&effect_proj).map_err(|e| {
                crate::error::EmbeddingError::GpuError {
                    message: format!("Var from tensor: {}", e),
                }
            })?,
            effect_bias_var: candle_core::Var::from_tensor(&effect_bias).map_err(|e| {
                crate::error::EmbeddingError::GpuError {
                    message: format!("Var from tensor: {}", e),
                }
            })?,
            hidden_size,
        })
    }

    /// Apply cause projection using trainable Var (gradients flow through).
    pub fn project_cause_trainable(&self, embedding: &Tensor) -> crate::error::EmbeddingResult<Tensor> {
        let projected = embedding
            .matmul(
                &self.cause_projection_var.as_tensor()
                    .t()
                    .map_err(|e| crate::error::EmbeddingError::GpuError {
                        message: format!("Cause projection transpose: {}", e),
                    })?,
            )
            .map_err(|e| crate::error::EmbeddingError::GpuError {
                message: format!("Cause projection matmul: {}", e),
            })?;

        projected
            .broadcast_add(self.cause_bias_var.as_tensor())
            .map_err(|e| crate::error::EmbeddingError::GpuError {
                message: format!("Cause bias add: {}", e),
            })
    }

    /// Apply effect projection using trainable Var (gradients flow through).
    pub fn project_effect_trainable(&self, embedding: &Tensor) -> crate::error::EmbeddingResult<Tensor> {
        let projected = embedding
            .matmul(
                &self.effect_projection_var.as_tensor()
                    .t()
                    .map_err(|e| crate::error::EmbeddingError::GpuError {
                        message: format!("Effect projection transpose: {}", e),
                    })?,
            )
            .map_err(|e| crate::error::EmbeddingError::GpuError {
                message: format!("Effect projection matmul: {}", e),
            })?;

        projected
            .broadcast_add(self.effect_bias_var.as_tensor())
            .map_err(|e| crate::error::EmbeddingError::GpuError {
                message: format!("Effect bias add: {}", e),
            })
    }
}

impl CausalProjectionWeights {
    /// Initialize projection weights as perturbed identity matrices.
    ///
    /// Creates W_cause = I + N(0, 0.02) and W_effect = I + N(0, 0.02) with
    /// different random perturbations to create asymmetry.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Dimension of embeddings (768 for Longformer-base)
    /// * `device` - Device to create tensors on
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Initialized CausalProjectionWeights
    pub fn initialize(
        hidden_size: usize,
        device: &Device,
        seed: u64,
    ) -> EmbeddingResult<Self> {
        // Use seeded RNG for reproducibility
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Create perturbed identity for cause projection
        let cause_data = create_perturbed_identity(hidden_size, &mut rng, PROJECTION_INIT_STD);
        let cause_projection =
            Tensor::from_slice(&cause_data, (hidden_size, hidden_size), device)
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("Failed to create cause projection: {}", e),
                })?
                .to_dtype(DType::F32)
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("Failed to convert cause projection dtype: {}", e),
                })?;

        // Create perturbed identity for effect projection (different perturbation)
        let effect_data = create_perturbed_identity(hidden_size, &mut rng, PROJECTION_INIT_STD);
        let effect_projection =
            Tensor::from_slice(&effect_data, (hidden_size, hidden_size), device)
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("Failed to create effect projection: {}", e),
                })?
                .to_dtype(DType::F32)
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("Failed to convert effect projection dtype: {}", e),
                })?;

        // Initialize biases to small random values (not zero, to add asymmetry)
        let cause_bias_data: Vec<f32> = (0..hidden_size)
            .map(|_| rng.gen_range(-0.01f32..0.01f32))
            .collect();
        let cause_bias =
            Tensor::from_slice(&cause_bias_data, hidden_size, device).map_err(|e| {
                EmbeddingError::GpuError {
                    message: format!("Failed to create cause bias: {}", e),
                }
            })?;

        let effect_bias_data: Vec<f32> = (0..hidden_size)
            .map(|_| rng.gen_range(-0.01f32..0.01f32))
            .collect();
        let effect_bias =
            Tensor::from_slice(&effect_bias_data, hidden_size, device).map_err(|e| {
                EmbeddingError::GpuError {
                    message: format!("Failed to create effect bias: {}", e),
                }
            })?;

        Ok(Self {
            cause_projection,
            cause_bias,
            effect_projection,
            effect_bias,
        })
    }

    /// Apply cause projection to an embedding.
    ///
    /// Computes: cause_vec = base_embedding @ W_cause^T + b_cause
    ///
    /// # Arguments
    ///
    /// * `embedding` - Input embedding tensor [1, hidden_size]
    ///
    /// # Returns
    ///
    /// Projected cause embedding [1, hidden_size]
    pub fn project_cause(&self, embedding: &Tensor) -> EmbeddingResult<Tensor> {
        let projected = embedding
            .matmul(
                &self
                    .cause_projection
                    .t()
                    .map_err(|e| EmbeddingError::GpuError {
                        message: format!("Cause projection transpose failed: {}", e),
                    })?,
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Cause projection matmul failed: {}", e),
            })?;

        projected
            .broadcast_add(&self.cause_bias)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Cause projection bias add failed: {}", e),
            })
    }

    /// Apply effect projection to an embedding.
    ///
    /// Computes: effect_vec = base_embedding @ W_effect^T + b_effect
    ///
    /// # Arguments
    ///
    /// * `embedding` - Input embedding tensor [1, hidden_size]
    ///
    /// # Returns
    ///
    /// Projected effect embedding [1, hidden_size]
    pub fn project_effect(&self, embedding: &Tensor) -> EmbeddingResult<Tensor> {
        let projected = embedding
            .matmul(
                &self
                    .effect_projection
                    .t()
                    .map_err(|e| EmbeddingError::GpuError {
                        message: format!("Effect projection transpose failed: {}", e),
                    })?,
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Effect projection matmul failed: {}", e),
            })?;

        projected
            .broadcast_add(&self.effect_bias)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Effect projection bias add failed: {}", e),
            })
    }
}

/// Create a perturbed identity matrix: I + N(0, std)
fn create_perturbed_identity<R: Rng>(size: usize, rng: &mut R, std: f64) -> Vec<f32> {
    let mut data = vec![0.0f32; size * size];

    for i in 0..size {
        for j in 0..size {
            let idx = i * size + j;
            // Identity component
            let identity: f32 = if i == j { 1.0 } else { 0.0 };
            // Random perturbation from normal distribution
            // Using Box-Muller transform for normal distribution
            let u1: f64 = rng.gen_range(0.0001f64..1.0f64);
            let u2: f64 = rng.gen_range(0.0f64..1.0f64);
            let normal: f64 = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
            let perturbation = (normal * std) as f32;

            data[idx] = identity + perturbation;
        }
    }

    data
}
