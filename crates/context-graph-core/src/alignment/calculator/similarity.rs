//! Similarity computation utilities for alignment.

/// Compute cosine similarity between two embedding vectors.
///
/// # Performance
/// O(n) where n is the embedding dimension (typically 1024).
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a.sqrt()) * (norm_b.sqrt());
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

/// Compute dense embedding alignment with normalization to [0, 1].
#[inline]
pub fn compute_dense_alignment(embedding: &[f32], goal_projected: &[f32]) -> f32 {
    if embedding.is_empty() || goal_projected.is_empty() {
        return 0.5; // Neutral alignment for missing embeddings
    }
    let cosine = cosine_similarity(embedding, goal_projected);
    // Normalize cosine [-1, 1] to [0, 1]
    (cosine + 1.0) / 2.0
}

/// Compute sparse vector alignment using cosine similarity between sparse vectors.
///
/// ARCH-02: Apples-to-apples comparison between sparse vectors in the same embedding space.
pub fn compute_sparse_vector_alignment(
    fp_sparse: &crate::types::fingerprint::SparseVector,
    goal_sparse: &crate::types::fingerprint::SparseVector,
) -> f32 {
    if fp_sparse.is_empty() && goal_sparse.is_empty() {
        return 0.5; // Both empty = neutral alignment
    }
    if fp_sparse.is_empty() || goal_sparse.is_empty() {
        return 0.25; // One empty = low alignment
    }

    // Compute sparse vector dot product over shared indices
    let mut dot = 0.0f32;
    let mut fp_i = 0;
    let mut goal_i = 0;

    while fp_i < fp_sparse.indices.len() && goal_i < goal_sparse.indices.len() {
        let fp_idx = fp_sparse.indices[fp_i];
        let goal_idx = goal_sparse.indices[goal_i];

        if fp_idx == goal_idx {
            dot += fp_sparse.values[fp_i] * goal_sparse.values[goal_i];
            fp_i += 1;
            goal_i += 1;
        } else if fp_idx < goal_idx {
            fp_i += 1;
        } else {
            goal_i += 1;
        }
    }

    let fp_norm = fp_sparse.l2_norm();
    let goal_norm = goal_sparse.l2_norm();
    let denom = fp_norm * goal_norm;

    if denom < f32::EPSILON {
        return 0.5; // Neutral for zero-norm vectors
    }

    let cosine = dot / denom;
    // Normalize cosine [-1, 1] to [0, 1]
    (cosine + 1.0) / 2.0
}

/// Compute late interaction alignment (ColBERT style) using max-sim between token vectors.
///
/// ARCH-02: Apples-to-apples comparison between late interaction token vectors.
pub fn compute_late_interaction_vectors(fp_tokens: &[Vec<f32>], goal_tokens: &[Vec<f32>]) -> f32 {
    if fp_tokens.is_empty() && goal_tokens.is_empty() {
        return 0.5; // Both empty = neutral alignment
    }
    if fp_tokens.is_empty() || goal_tokens.is_empty() {
        return 0.25; // One empty = low alignment
    }

    // MaxSim: for each fingerprint token, find max similarity across goal tokens
    let mut total_max_sim = 0.0f32;
    for fp_token in fp_tokens {
        let mut max_sim = -1.0f32;
        for goal_token in goal_tokens {
            let sim = cosine_similarity(fp_token, goal_token);
            if sim > max_sim {
                max_sim = sim;
            }
        }
        total_max_sim += max_sim;
    }

    // Average max similarity and normalize to [0, 1]
    let avg_max_sim = total_max_sim / fp_tokens.len() as f32;
    (avg_max_sim + 1.0) / 2.0
}
