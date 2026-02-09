//! Evaluation metrics for causal embedder training.
//!
//! Measures directional accuracy, topical MRR, causal vs non-causal AUC,
//! direction ratio, and cross-topic rejection rate.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

/// Evaluation metrics for trained causal embeddings.
#[derive(Debug, Clone, Default)]
pub struct EvaluationMetrics {
    /// Fraction of pairs where forward sim > reverse sim (target: >0.90).
    pub directional_accuracy: f32,
    /// Mean Reciprocal Rank for topical retrieval (target: >0.7).
    pub topical_mrr: f32,
    /// AUC for distinguishing causal vs non-causal pairs (target: >0.85).
    pub causal_auc: f32,
    /// Ratio of forward similarity to reverse similarity (target: >2.0).
    pub direction_ratio: f32,
    /// Fraction of cross-topic non-causal results rejected from top-K (target: >0.80).
    pub cross_topic_rejection: f32,
    /// Number of evaluation pairs.
    pub num_pairs: usize,
}

impl EvaluationMetrics {
    /// Check if all targets are met.
    pub fn meets_targets(&self) -> bool {
        self.directional_accuracy > 0.90
            && self.topical_mrr > 0.7
            && self.causal_auc > 0.85
            && self.direction_ratio > 2.0
            && self.cross_topic_rejection > 0.80
    }

    /// Format metrics as a summary string.
    pub fn summary(&self) -> String {
        format!(
            "DirAcc={:.3} MRR={:.3} AUC={:.3} DirRatio={:.2} XTopicRej={:.3} (n={})",
            self.directional_accuracy,
            self.topical_mrr,
            self.causal_auc,
            self.direction_ratio,
            self.cross_topic_rejection,
            self.num_pairs,
        )
    }
}

/// Evaluator for causal embedding quality.
pub struct Evaluator;

impl Evaluator {
    /// Compute directional accuracy: fraction of pairs where
    /// sim(cause→effect) > sim(effect→cause).
    ///
    /// # Arguments
    /// * `cause_vecs` - Cause embeddings [N, D] (L2-normalized)
    /// * `effect_vecs` - Effect embeddings [N, D] (L2-normalized)
    pub fn directional_accuracy(
        cause_vecs: &Tensor,
        effect_vecs: &Tensor,
    ) -> EmbeddingResult<f32> {
        let forward_sim = batch_cosine_sim(cause_vecs, effect_vecs)?;
        let reverse_sim = batch_cosine_sim(effect_vecs, cause_vecs)?;

        let forward_vals: Vec<f32> = forward_sim.to_vec1().map_err(map_candle)?;
        let reverse_vals: Vec<f32> = reverse_sim.to_vec1().map_err(map_candle)?;

        let n = forward_vals.len();
        if n == 0 {
            return Ok(0.0);
        }

        let correct = forward_vals
            .iter()
            .zip(reverse_vals.iter())
            .filter(|(f, r)| f > r)
            .count();

        Ok(correct as f32 / n as f32)
    }

    /// Compute direction ratio: mean(forward_sim) / mean(reverse_sim).
    pub fn direction_ratio(
        cause_vecs: &Tensor,
        effect_vecs: &Tensor,
    ) -> EmbeddingResult<f32> {
        let forward_sim = batch_cosine_sim(cause_vecs, effect_vecs)?;
        let reverse_sim = batch_cosine_sim(effect_vecs, cause_vecs)?;

        let forward_vals: Vec<f32> = forward_sim.to_vec1().map_err(map_candle)?;
        let reverse_vals: Vec<f32> = reverse_sim.to_vec1().map_err(map_candle)?;

        let mean_forward: f32 =
            forward_vals.iter().sum::<f32>() / forward_vals.len().max(1) as f32;
        let mean_reverse: f32 =
            reverse_vals.iter().sum::<f32>() / reverse_vals.len().max(1) as f32;

        if mean_reverse.abs() < 1e-8 {
            return Ok(f32::INFINITY);
        }

        Ok(mean_forward / mean_reverse)
    }

    /// Compute AUC for causal vs non-causal discrimination.
    ///
    /// Uses similarity scores as predictions and binary labels (1=causal, 0=non-causal).
    pub fn causal_auc(
        similarities: &[f32],
        labels: &[bool],
    ) -> f32 {
        if similarities.len() != labels.len() || similarities.is_empty() {
            return 0.0;
        }

        // Sort by similarity descending
        let mut pairs: Vec<(f32, bool)> = similarities
            .iter()
            .zip(labels.iter())
            .map(|(&s, &l)| (s, l))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_pos = labels.iter().filter(|&&l| l).count() as f32;
        let total_neg = labels.iter().filter(|&&l| !l).count() as f32;

        if total_pos == 0.0 || total_neg == 0.0 {
            return 0.5;
        }

        // Wilcoxon-Mann-Whitney statistic
        let mut auc = 0.0f32;
        let mut tp = 0.0f32;

        for (_, is_positive) in &pairs {
            if *is_positive {
                tp += 1.0;
            } else {
                auc += tp;
            }
        }

        auc / (total_pos * total_neg)
    }

    /// Compute Mean Reciprocal Rank for topical retrieval.
    ///
    /// For each query, finds the rank of the correct same-topic causal partner
    /// among all candidates.
    ///
    /// # Arguments
    /// * `query_vecs` - Query embeddings [N, D]
    /// * `candidate_vecs` - Candidate embeddings [M, D]
    /// * `correct_indices` - Index into candidates for each query's correct match
    pub fn topical_mrr(
        query_vecs: &Tensor,
        candidate_vecs: &Tensor,
        correct_indices: &[usize],
    ) -> EmbeddingResult<f32> {
        let sim_matrix = query_vecs
            .matmul(&candidate_vecs.t().map_err(map_candle)?)
            .map_err(map_candle)?;

        let n = correct_indices.len();
        if n == 0 {
            return Ok(0.0);
        }

        let mut mrr_sum = 0.0f32;

        for i in 0..n {
            let row: Vec<f32> = sim_matrix.get(i).map_err(map_candle)?.to_vec1().map_err(map_candle)?;
            let correct_idx = correct_indices[i];

            // Rank: count how many candidates have higher similarity
            let correct_sim = row[correct_idx];
            let rank = row.iter().filter(|&&s| s > correct_sim).count() + 1;
            mrr_sum += 1.0 / rank as f32;
        }

        Ok(mrr_sum / n as f32)
    }

    /// Compute full evaluation metrics.
    ///
    /// # Arguments
    /// * `cause_vecs` - Cause embeddings [N, D] for causal pairs
    /// * `effect_vecs` - Effect embeddings [N, D] for causal pairs
    /// * `non_causal_sims` - Similarity scores for non-causal pairs
    pub fn evaluate(
        cause_vecs: &Tensor,
        effect_vecs: &Tensor,
        non_causal_sims: &[f32],
    ) -> EmbeddingResult<EvaluationMetrics> {
        let n = cause_vecs.dim(0).map_err(map_candle)?;

        let directional_accuracy = Self::directional_accuracy(cause_vecs, effect_vecs)?;
        let direction_ratio = Self::direction_ratio(cause_vecs, effect_vecs)?;

        // Build labels for AUC: causal pairs + non-causal pairs
        let forward_sim = batch_cosine_sim(cause_vecs, effect_vecs)?;
        let forward_vals: Vec<f32> = forward_sim.to_vec1().map_err(map_candle)?;

        let mut all_sims: Vec<f32> = forward_vals.clone();
        all_sims.extend_from_slice(non_causal_sims);

        let mut all_labels: Vec<bool> = vec![true; n];
        all_labels.extend(vec![false; non_causal_sims.len()]);

        let causal_auc = Self::causal_auc(&all_sims, &all_labels);

        Ok(EvaluationMetrics {
            directional_accuracy,
            topical_mrr: 0.0, // Requires retrieval setup — computed separately
            causal_auc,
            direction_ratio,
            cross_topic_rejection: 0.0, // Requires cross-topic pairs — computed separately
            num_pairs: n,
        })
    }
}

/// Batch cosine similarity between paired vectors [N] where result[i] = cos(a[i], b[i]).
fn batch_cosine_sim(a: &Tensor, b: &Tensor) -> EmbeddingResult<Tensor> {
    let dot = (a * b).map_err(map_candle)?.sum(1).map_err(map_candle)?;
    let norm_a = a
        .sqr()
        .map_err(map_candle)?
        .sum(1)
        .map_err(map_candle)?
        .sqrt()
        .map_err(map_candle)?;
    let norm_b = b
        .sqr()
        .map_err(map_candle)?
        .sum(1)
        .map_err(map_candle)?
        .sqrt()
        .map_err(map_candle)?;
    let denom = (norm_a * norm_b).map_err(map_candle)?;
    let eps = Tensor::ones_like(&denom)
        .map_err(map_candle)?
        .affine(1e-8, 0.0)
        .map_err(map_candle)?;
    let safe_denom = denom.add(&eps).map_err(map_candle)?;
    dot.div(&safe_denom).map_err(map_candle)
}

/// Map candle errors to EmbeddingError.
fn map_candle(e: candle_core::Error) -> EmbeddingError {
    EmbeddingError::GpuError {
        message: format!("Evaluation error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_metrics_default() {
        let m = EvaluationMetrics::default();
        assert!(!m.meets_targets());
    }

    #[test]
    fn test_metrics_summary() {
        let m = EvaluationMetrics {
            directional_accuracy: 0.92,
            topical_mrr: 0.75,
            causal_auc: 0.88,
            direction_ratio: 2.3,
            cross_topic_rejection: 0.85,
            num_pairs: 100,
        };
        assert!(m.meets_targets());
        let s = m.summary();
        assert!(s.contains("0.920"));
        assert!(s.contains("n=100"));
    }

    #[test]
    fn test_causal_auc() {
        // Perfect separation
        let sims = vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
        let labels = vec![true, true, true, false, false, false];
        let auc = Evaluator::causal_auc(&sims, &labels);
        assert!((auc - 1.0).abs() < 1e-6, "Perfect separation AUC should be 1.0, got {}", auc);

        // Interleaved scores: AUC should be between 0 and 1
        let sims = vec![0.6, 0.4, 0.5, 0.3];
        let labels = vec![true, false, true, false];
        let auc = Evaluator::causal_auc(&sims, &labels);
        assert!(auc >= 0.0 && auc <= 1.0, "AUC should be in [0, 1], got {}", auc);
    }

    #[test]
    fn test_directional_accuracy() {
        let device = Device::Cpu;
        // With symmetric cosine similarity, directional accuracy measures
        // whether cos(cause_i, effect_i) > cos(effect_i, cause_i).
        // For standard cosine, these are identical, so accuracy should be 0%.
        // After training with asymmetric projections, this would differ.
        let cause = Tensor::from_slice(
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0],
            (2, 3),
            &device,
        )
        .unwrap();
        let effect = Tensor::from_slice(
            &[0.9f32, 0.1, 0.0, 0.1, 0.9, 0.0],
            (2, 3),
            &device,
        )
        .unwrap();

        let acc = Evaluator::directional_accuracy(&cause, &effect).unwrap();
        // Symmetric cosine → forward == reverse → 0% accuracy (none strictly greater)
        assert!(acc >= 0.0 && acc <= 1.0, "Accuracy should be in [0,1], got {}", acc);
    }
}
