//! Multi-space alignment computation for DefaultAlignmentCalculator.

use super::DefaultAlignmentCalculator;
use crate::alignment::calculator::similarity::{
    compute_dense_alignment, compute_late_interaction_vectors, compute_sparse_vector_alignment,
};
use crate::alignment::score::{GoalScore, LevelWeights};
use crate::purpose::GoalNode;
use crate::types::fingerprint::{SemanticFingerprint, NUM_EMBEDDERS};

impl DefaultAlignmentCalculator {
    /// Compute alignment between fingerprint and a single goal.
    ///
    /// Computes cosine similarity across ALL 13 embedding spaces as required
    /// by constitution.yaml and returns the weighted average.
    ///
    /// # Multi-Space Alignment (Constitution v4.0.0)
    ///
    /// Per constitution.yaml, alignment MUST use ALL 13 embedding spaces:
    /// ```text
    /// A_multi = SUM_i(w_i * A(E_i, V)) where SUM(w_i) = 1
    /// ```
    pub(crate) fn compute_goal_alignment(
        &self,
        fingerprint: &SemanticFingerprint,
        goal: &GoalNode,
        weights: &LevelWeights,
    ) -> GoalScore {
        // Get propagation weight based on goal level
        let level_weight = Self::get_propagation_weight(goal.level);
        let config_weight = weights.for_level(goal.level);

        // Compute multi-space alignment using ALL 13 embedders
        // Formula: A_multi = SUM_i(w_i * A(E_i, V)) where SUM(w_i) = 1
        let alignments = self.compute_all_space_alignments(fingerprint, goal);

        // Aggregate using teleological weights
        let mut weighted_alignment = 0.0f32;
        for (i, &alignment) in alignments.iter().enumerate() {
            weighted_alignment += self.teleological_weights.weight(i) * alignment;
        }

        // Apply level propagation weight
        let final_alignment = weighted_alignment * level_weight;

        GoalScore::new(goal.id, goal.level, final_alignment, config_weight)
    }

    /// Compute cosine similarity for ALL 13 embedding spaces using apples-to-apples comparison.
    ///
    /// ARCH-02: Each embedding space in the fingerprint is compared with the corresponding
    /// embedding space in the goal's teleological array. E1 vs E1, E2 vs E2, etc.
    ///
    /// Returns an array of 13 alignment values, one for each embedding space.
    /// Each value is normalized to [0, 1] range.
    pub(crate) fn compute_all_space_alignments(
        &self,
        fingerprint: &SemanticFingerprint,
        goal: &GoalNode,
    ) -> [f32; NUM_EMBEDDERS] {
        let mut alignments = [0.0f32; NUM_EMBEDDERS];
        let goal_array = goal.array();

        // ARCH-02: Apples-to-apples comparison - same embedder to same embedder
        // E1: Semantic (1024D) - E1 vs E1
        alignments[0] = compute_dense_alignment(&fingerprint.e1_semantic, &goal_array.e1_semantic);

        // E2: Temporal Recent (512D) - E2 vs E2
        alignments[1] = compute_dense_alignment(
            &fingerprint.e2_temporal_recent,
            &goal_array.e2_temporal_recent,
        );

        // E3: Temporal Periodic (512D) - E3 vs E3
        alignments[2] = compute_dense_alignment(
            &fingerprint.e3_temporal_periodic,
            &goal_array.e3_temporal_periodic,
        );

        // E4: Temporal Positional (512D) - E4 vs E4
        alignments[3] = compute_dense_alignment(
            &fingerprint.e4_temporal_positional,
            &goal_array.e4_temporal_positional,
        );

        // E5: Causal (768D) - E5 vs E5
        alignments[4] = compute_dense_alignment(&fingerprint.e5_causal, &goal_array.e5_causal);

        // E6: Sparse (SPLADE) - E6 vs E6
        alignments[5] =
            compute_sparse_vector_alignment(&fingerprint.e6_sparse, &goal_array.e6_sparse);

        // E7: Code (1536D - Qodo-Embed) - E7 vs E7
        alignments[6] = compute_dense_alignment(&fingerprint.e7_code, &goal_array.e7_code);

        // E8: Graph (384D) - E8 vs E8
        alignments[7] = compute_dense_alignment(&fingerprint.e8_graph, &goal_array.e8_graph);

        // E9: HDC (1024D projected) - E9 vs E9
        alignments[8] = compute_dense_alignment(&fingerprint.e9_hdc, &goal_array.e9_hdc);

        // E10: Multimodal (768D) - E10 vs E10
        alignments[9] =
            compute_dense_alignment(&fingerprint.e10_multimodal, &goal_array.e10_multimodal);

        // E11: Entity (384D) - E11 vs E11
        alignments[10] = compute_dense_alignment(&fingerprint.e11_entity, &goal_array.e11_entity);

        // E12: Late Interaction (ColBERT) - E12 vs E12 (max-sim over tokens)
        alignments[11] = compute_late_interaction_vectors(
            &fingerprint.e12_late_interaction,
            &goal_array.e12_late_interaction,
        );

        // E13: SPLADE v3 - E13 vs E13
        alignments[12] =
            compute_sparse_vector_alignment(&fingerprint.e13_splade, &goal_array.e13_splade);

        alignments
    }
}
