//! Loss functions for causal embedder fine-tuning.
//!
//! Four loss components informed by Causal2Vec, A-InfoNCE, and contrastive learning literature:
//! 1. InfoNCE contrastive (τ=0.05)
//! 2. Directional margin (margin=0.2)
//! 3. Causal separation
//! 4. Soft label distillation

use candle_core::{DType, Tensor};

use crate::error::{EmbeddingError, EmbeddingResult};

/// Configuration for the combined loss function.
#[derive(Debug, Clone)]
pub struct LossConfig {
    /// Temperature for InfoNCE (default: 0.05 per Causal2Vec).
    pub temperature: f32,
    /// Margin for directional loss (default: 0.2).
    pub margin: f32,
    /// Weight for contrastive loss (default: 1.0).
    pub lambda_contrastive: f32,
    /// Weight for directional loss (default: 0.3).
    pub lambda_directional: f32,
    /// Weight for separation loss (default: 0.1).
    pub lambda_separation: f32,
    /// Weight for soft label loss (default: 0.2).
    pub lambda_soft: f32,
    /// Minimum E1 similarity threshold to exclude potential false negatives.
    pub false_negative_threshold: f32,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            temperature: 0.05,
            margin: 0.2,
            lambda_contrastive: 1.0,
            lambda_directional: 0.3,
            lambda_separation: 0.1,
            lambda_soft: 0.2,
            false_negative_threshold: 0.8,
        }
    }
}

/// Combined loss function for causal embedder training.
pub struct DirectionalContrastiveLoss {
    config: LossConfig,
}

/// Per-component loss values for logging.
#[derive(Debug, Clone, Default)]
pub struct LossComponents {
    /// InfoNCE contrastive loss value.
    pub contrastive: f32,
    /// Directional margin loss value.
    pub directional: f32,
    /// Causal separation loss value.
    pub separation: f32,
    /// Soft label distillation loss value.
    pub soft_label: f32,
    /// Total combined loss.
    pub total: f32,
}

impl DirectionalContrastiveLoss {
    /// Create a new loss function with the given configuration.
    pub fn new(config: LossConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(LossConfig::default())
    }

    /// Compute InfoNCE contrastive loss.
    ///
    /// Pulls cause→effect pairs together, pushes non-causal pairs apart.
    /// Uses in-batch negatives: each batch of N pairs provides N*(N-1) free negatives.
    ///
    /// L = -log(exp(sim(cause_i, effect_i)/τ) / Σ_j exp(sim(cause_i, effect_j)/τ))
    ///
    /// # Arguments
    /// * `cause_vecs` - Cause embeddings [N, D] (L2-normalized)
    /// * `effect_vecs` - Effect embeddings [N, D] (L2-normalized)
    pub fn info_nce_loss(
        &self,
        cause_vecs: &Tensor,
        effect_vecs: &Tensor,
    ) -> EmbeddingResult<Tensor> {
        let tau = self.config.temperature as f64;

        // Similarity matrix: [N, N] where sim[i,j] = cos(cause_i, effect_j) / τ
        let logits = cause_vecs
            .matmul(&effect_vecs.t().map_err(map_candle)?)
            .map_err(map_candle)?
            .affine(1.0 / tau, 0.0)
            .map_err(map_candle)?;

        // Labels: diagonal (each cause_i pairs with effect_i)
        let n = cause_vecs.dim(0).map_err(map_candle)?;
        let labels = Tensor::arange(0u32, n as u32, cause_vecs.device())
            .map_err(map_candle)?;

        // Cross-entropy loss over rows
        cross_entropy_loss(&logits, &labels)
    }

    /// Compute directional margin loss.
    ///
    /// Enforces that forward direction (cause→effect) scores higher than reverse.
    /// L = max(0, margin - sim(cause, effect) + sim(effect, cause))
    ///
    /// # Arguments
    /// * `cause_vecs` - Cause embeddings [N, D]
    /// * `effect_vecs` - Effect embeddings [N, D]
    pub fn directional_margin_loss(
        &self,
        cause_vecs: &Tensor,
        effect_vecs: &Tensor,
    ) -> EmbeddingResult<Tensor> {
        // Forward similarity: sim(cause_i, effect_i) - diagonal of matmul
        let forward_sim = batch_cosine_similarity(cause_vecs, effect_vecs)?;

        // Reverse similarity: sim(effect_i, cause_i) - should be lower
        let reverse_sim = batch_cosine_similarity(effect_vecs, cause_vecs)?;

        // margin - forward + reverse, clamped to >= 0
        let margin_tensor = Tensor::ones_like(&forward_sim)
            .map_err(map_candle)?
            .affine(self.config.margin as f64, 0.0)
            .map_err(map_candle)?;

        let loss = margin_tensor
            .sub(&forward_sim)
            .map_err(map_candle)?
            .add(&reverse_sim)
            .map_err(map_candle)?;

        // ReLU: max(0, x)
        let zeros = Tensor::zeros_like(&loss).map_err(map_candle)?;
        let clamped = loss.maximum(&zeros).map_err(map_candle)?;

        clamped.mean_all().map_err(map_candle)
    }

    /// Compute causal separation loss.
    ///
    /// Same text's cause-vector and effect-vector should differ.
    /// L = -distance(cause_vec, effect_vec) for same input text.
    ///
    /// # Arguments
    /// * `cause_vecs` - Cause embeddings [N, D] (from same texts)
    /// * `effect_vecs` - Effect embeddings [N, D] (from same texts)
    pub fn separation_loss(
        &self,
        cause_vecs: &Tensor,
        effect_vecs: &Tensor,
    ) -> EmbeddingResult<Tensor> {
        // Cosine similarity between paired cause/effect vectors
        let sim = batch_cosine_similarity(cause_vecs, effect_vecs)?;

        // We want to minimize similarity → loss = mean(sim)
        // (Equivalent to maximizing distance)
        sim.mean_all().map_err(map_candle)
    }

    /// Compute soft label distillation loss.
    ///
    /// Uses LLM confidence as soft target instead of hard binary labels.
    /// L_soft = MSE(sim(cause, effect), confidence)
    ///
    /// # Arguments
    /// * `cause_vecs` - Cause embeddings [N, D]
    /// * `effect_vecs` - Effect embeddings [N, D]
    /// * `confidences` - LLM confidence scores [N] as soft labels
    pub fn soft_label_loss(
        &self,
        cause_vecs: &Tensor,
        effect_vecs: &Tensor,
        confidences: &Tensor,
    ) -> EmbeddingResult<Tensor> {
        // Predicted similarity
        let sim = batch_cosine_similarity(cause_vecs, effect_vecs)?;

        // MSE: mean((sim - confidence)^2)
        let diff = sim.sub(confidences).map_err(map_candle)?;
        let sq = diff.sqr().map_err(map_candle)?;
        sq.mean_all().map_err(map_candle)
    }

    /// Compute the combined loss.
    ///
    /// L = λ_c * L_contrastive + λ_d * L_directional + λ_s * L_separation + λ_soft * L_soft
    ///
    /// # Arguments
    /// * `cause_vecs` - Cause embeddings [N, D] (L2-normalized)
    /// * `effect_vecs` - Effect embeddings [N, D] (L2-normalized)
    /// * `confidences` - LLM confidence scores [N]
    ///
    /// # Returns
    /// (total_loss_tensor, LossComponents for logging)
    pub fn compute(
        &self,
        cause_vecs: &Tensor,
        effect_vecs: &Tensor,
        confidences: &Tensor,
    ) -> EmbeddingResult<(Tensor, LossComponents)> {
        let l_contrastive = self.info_nce_loss(cause_vecs, effect_vecs)?;
        let l_directional = self.directional_margin_loss(cause_vecs, effect_vecs)?;
        let l_separation = self.separation_loss(cause_vecs, effect_vecs)?;
        let l_soft = self.soft_label_loss(cause_vecs, effect_vecs, confidences)?;

        // Weighted combination
        let total = l_contrastive
            .affine(self.config.lambda_contrastive as f64, 0.0)
            .map_err(map_candle)?
            .add(
                &l_directional
                    .affine(self.config.lambda_directional as f64, 0.0)
                    .map_err(map_candle)?,
            )
            .map_err(map_candle)?
            .add(
                &l_separation
                    .affine(self.config.lambda_separation as f64, 0.0)
                    .map_err(map_candle)?,
            )
            .map_err(map_candle)?
            .add(
                &l_soft
                    .affine(self.config.lambda_soft as f64, 0.0)
                    .map_err(map_candle)?,
            )
            .map_err(map_candle)?;

        // Extract scalar values for logging
        let components = LossComponents {
            contrastive: tensor_to_f32(&l_contrastive)?,
            directional: tensor_to_f32(&l_directional)?,
            separation: tensor_to_f32(&l_separation)?,
            soft_label: tensor_to_f32(&l_soft)?,
            total: tensor_to_f32(&total)?,
        };

        Ok((total, components))
    }

    /// Get the loss configuration.
    pub fn config(&self) -> &LossConfig {
        &self.config
    }
}

/// Compute batch cosine similarity between paired vectors.
/// Returns [N] where result[i] = cos(a[i], b[i]).
fn batch_cosine_similarity(a: &Tensor, b: &Tensor) -> EmbeddingResult<Tensor> {
    // Element-wise multiply and sum over last dimension
    let dot = (a * b).map_err(map_candle)?.sum(1).map_err(map_candle)?;

    // Norms
    let norm_a = a.sqr().map_err(map_candle)?.sum(1).map_err(map_candle)?.sqrt().map_err(map_candle)?;
    let norm_b = b.sqr().map_err(map_candle)?.sum(1).map_err(map_candle)?.sqrt().map_err(map_candle)?;

    let denom = (norm_a * norm_b).map_err(map_candle)?;

    // Add epsilon to avoid division by zero
    let eps = Tensor::ones_like(&denom)
        .map_err(map_candle)?
        .affine(1e-8, 0.0)
        .map_err(map_candle)?;
    let safe_denom = denom.add(&eps).map_err(map_candle)?;

    dot.div(&safe_denom).map_err(map_candle)
}

/// Cross-entropy loss over rows of logits with integer labels.
fn cross_entropy_loss(logits: &Tensor, labels: &Tensor) -> EmbeddingResult<Tensor> {
    let n = logits.dim(0).map_err(map_candle)?;
    let device = logits.device();

    // log_softmax over last dimension
    let max_logits = logits
        .max_keepdim(1)
        .map_err(map_candle)?;
    let shifted = logits
        .broadcast_sub(&max_logits)
        .map_err(map_candle)?;
    let exp = shifted.exp().map_err(map_candle)?;
    let sum_exp = exp.sum_keepdim(1).map_err(map_candle)?;
    let log_softmax = shifted
        .broadcast_sub(&sum_exp.log().map_err(map_candle)?)
        .map_err(map_candle)?;

    // Gather log probabilities at label indices
    let labels_u32 = labels.to_dtype(DType::U32).map_err(map_candle)?;
    let label_vec: Vec<u32> = labels_u32.to_vec1().map_err(map_candle)?;

    let mut nll_sum = 0.0f64;
    for i in 0..n {
        let row = log_softmax.get(i).map_err(map_candle)?;
        let label_idx = label_vec[i] as usize;
        let log_prob: f32 = row
            .get(label_idx)
            .map_err(map_candle)?
            .to_scalar()
            .map_err(map_candle)?;
        nll_sum -= log_prob as f64;
    }

    let loss_val = (nll_sum / n as f64) as f32;
    // Return scalar tensor matching mean_all() output shape
    Tensor::new(&[loss_val], device)
        .map_err(map_candle)?
        .squeeze(0)
        .map_err(map_candle)
}

/// Extract a scalar f32 from a 0-dim or 1-element tensor.
fn tensor_to_f32(t: &Tensor) -> EmbeddingResult<f32> {
    let flat = t.flatten_all().map_err(map_candle)?;
    let val: f32 = flat.to_vec1::<f32>().map_err(map_candle)?[0];
    Ok(val)
}

/// Map candle errors to EmbeddingError.
fn map_candle(e: candle_core::Error) -> EmbeddingError {
    EmbeddingError::GpuError {
        message: format!("Loss computation error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn make_test_vecs(n: usize, d: usize) -> (Tensor, Tensor) {
        let device = Device::Cpu;
        // Create normalized random vectors
        let cause_data: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let effect_data: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2 + 1.0).cos()).collect();

        let cause = Tensor::from_slice(&cause_data, (n, d), &device).unwrap();
        let effect = Tensor::from_slice(&effect_data, (n, d), &device).unwrap();

        // L2 normalize
        let cause_norm = cause.sqr().unwrap().sum(1).unwrap().sqrt().unwrap().unsqueeze(1).unwrap();
        let effect_norm = effect.sqr().unwrap().sum(1).unwrap().sqrt().unwrap().unsqueeze(1).unwrap();

        let cause = cause.broadcast_div(&cause_norm).unwrap();
        let effect = effect.broadcast_div(&effect_norm).unwrap();

        (cause, effect)
    }

    #[test]
    fn test_info_nce_loss_positive() {
        let loss_fn = DirectionalContrastiveLoss::default_config();
        let (cause, effect) = make_test_vecs(4, 16);

        let loss = loss_fn.info_nce_loss(&cause, &effect).unwrap();
        let val: f32 = loss.flatten_all().unwrap().to_vec1().unwrap()[0];

        // Loss should be positive (it's a cross-entropy)
        assert!(val > 0.0, "InfoNCE loss should be positive, got {}", val);
    }

    #[test]
    fn test_directional_margin_loss() {
        let loss_fn = DirectionalContrastiveLoss::default_config();
        let (cause, effect) = make_test_vecs(4, 16);

        let loss = loss_fn.directional_margin_loss(&cause, &effect).unwrap();
        let val: f32 = loss.flatten_all().unwrap().to_vec1().unwrap()[0];

        // Loss should be non-negative (ReLU)
        assert!(val >= 0.0, "Directional loss should be >= 0, got {}", val);
    }

    #[test]
    fn test_separation_loss() {
        let loss_fn = DirectionalContrastiveLoss::default_config();
        let (cause, effect) = make_test_vecs(4, 16);

        let loss = loss_fn.separation_loss(&cause, &effect).unwrap();
        let val: f32 = loss.flatten_all().unwrap().to_vec1().unwrap()[0];

        // Separation loss is mean cosine similarity, should be in [-1, 1]
        assert!(val >= -1.0 && val <= 1.0, "Separation loss should be in [-1,1], got {}", val);
    }

    #[test]
    fn test_soft_label_loss() {
        let loss_fn = DirectionalContrastiveLoss::default_config();
        let (cause, effect) = make_test_vecs(4, 16);
        let confidences = Tensor::from_slice(&[0.9f32, 0.8, 0.7, 0.1], 4, &Device::Cpu).unwrap();

        let loss = loss_fn.soft_label_loss(&cause, &effect, &confidences).unwrap();
        let val: f32 = loss.flatten_all().unwrap().to_vec1().unwrap()[0];

        // MSE should be non-negative
        assert!(val >= 0.0, "Soft label loss should be >= 0, got {}", val);
    }

    #[test]
    fn test_combined_loss() {
        let loss_fn = DirectionalContrastiveLoss::default_config();
        let (cause, effect) = make_test_vecs(4, 16);
        let confidences = Tensor::from_slice(&[0.9f32, 0.8, 0.7, 0.1], 4, &Device::Cpu).unwrap();

        let (total, components) = loss_fn.compute(&cause, &effect, &confidences).unwrap();
        let total_val: f32 = total.flatten_all().unwrap().to_vec1().unwrap()[0];

        assert!(total_val > 0.0, "Total loss should be positive");
        assert!(components.contrastive > 0.0, "Contrastive component should be positive");
        assert!(components.total > 0.0, "Logged total should match");
    }

    #[test]
    fn test_loss_config_default() {
        let config = LossConfig::default();
        assert_eq!(config.temperature, 0.05);
        assert_eq!(config.margin, 0.2);
        assert_eq!(config.lambda_contrastive, 1.0);
        assert_eq!(config.lambda_directional, 0.3);
        assert_eq!(config.lambda_separation, 0.1);
        assert_eq!(config.lambda_soft, 0.2);
    }
}
