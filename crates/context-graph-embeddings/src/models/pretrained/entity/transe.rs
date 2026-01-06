//! TransE-style knowledge graph operations.
//!
//! TransE models relationships as translations in embedding space:
//! - For a valid triple (h, r, t): h + r ≈ t
//! - Score = -||h + r - t||₂ (higher is better, 0 is perfect)

use super::types::ENTITY_DIMENSION;
use super::EntityModel;

impl EntityModel {
    /// TransE scoring: score = -||h + r - t||₂
    ///
    /// Computes the TransE score for a (head, relation, tail) triple.
    /// Higher score (closer to 0) indicates a more likely valid triple.
    ///
    /// # Arguments
    /// * `head` - Head entity embedding (384D)
    /// * `relation` - Relation embedding (384D)
    /// * `tail` - Tail entity embedding (384D)
    ///
    /// # Returns
    /// Negative L2 distance: 0 = perfect, more negative = worse.
    ///
    /// # Panics
    /// Panics if any input vector is not exactly ENTITY_DIMENSION (384) elements.
    ///
    /// # Examples
    /// ```rust
    /// use context_graph_embeddings::models::EntityModel;
    ///
    /// // Perfect triple: h + r = t
    /// let h: Vec<f32> = vec![1.0; 384];
    /// let r: Vec<f32> = vec![0.5; 384];
    /// let t: Vec<f32> = vec![1.5; 384];
    /// let score = EntityModel::transe_score(&h, &r, &t);
    /// assert!(score.abs() < 1e-5); // Should be ~0
    /// ```
    pub fn transe_score(head: &[f32], relation: &[f32], tail: &[f32]) -> f32 {
        assert_eq!(head.len(), ENTITY_DIMENSION);
        assert_eq!(relation.len(), ENTITY_DIMENSION);
        assert_eq!(tail.len(), ENTITY_DIMENSION);

        let sum_sq: f32 = head
            .iter()
            .zip(relation.iter())
            .zip(tail.iter())
            .map(|((h, r), t)| {
                let diff = h + r - t;
                diff * diff
            })
            .sum();

        -sum_sq.sqrt()
    }

    /// Predict tail entity embedding: t_hat = h + r
    ///
    /// Given a head entity and relation, predicts what the tail entity
    /// embedding should be according to the TransE model.
    ///
    /// # Arguments
    /// * `head` - Head entity embedding (384D)
    /// * `relation` - Relation embedding (384D)
    ///
    /// # Returns
    /// Predicted tail embedding (384D).
    ///
    /// # Panics
    /// Panics if any input vector is not exactly ENTITY_DIMENSION (384) elements.
    ///
    /// # Examples
    /// ```rust
    /// use context_graph_embeddings::models::EntityModel;
    ///
    /// let h: Vec<f32> = vec![1.0; 384];
    /// let r: Vec<f32> = vec![0.5; 384];
    /// let predicted_t = EntityModel::predict_tail(&h, &r);
    /// assert_eq!(predicted_t, vec![1.5; 384]);
    /// ```
    pub fn predict_tail(head: &[f32], relation: &[f32]) -> Vec<f32> {
        assert_eq!(head.len(), ENTITY_DIMENSION);
        assert_eq!(relation.len(), ENTITY_DIMENSION);

        head.iter()
            .zip(relation.iter())
            .map(|(h, r)| h + r)
            .collect()
    }

    /// Predict relation embedding: r_hat = t - h
    ///
    /// Given head and tail entity embeddings, predicts what the relation
    /// embedding should be according to the TransE model.
    ///
    /// # Arguments
    /// * `head` - Head entity embedding (384D)
    /// * `tail` - Tail entity embedding (384D)
    ///
    /// # Returns
    /// Predicted relation embedding (384D).
    ///
    /// # Panics
    /// Panics if any input vector is not exactly ENTITY_DIMENSION (384) elements.
    ///
    /// # Examples
    /// ```rust
    /// use context_graph_embeddings::models::EntityModel;
    ///
    /// let h: Vec<f32> = vec![1.0; 384];
    /// let t: Vec<f32> = vec![1.5; 384];
    /// let predicted_r = EntityModel::predict_relation(&h, &t);
    /// assert_eq!(predicted_r, vec![0.5; 384]);
    /// ```
    pub fn predict_relation(head: &[f32], tail: &[f32]) -> Vec<f32> {
        assert_eq!(head.len(), ENTITY_DIMENSION);
        assert_eq!(tail.len(), ENTITY_DIMENSION);

        tail.iter().zip(head.iter()).map(|(t, h)| t - h).collect()
    }
}
