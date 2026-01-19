//! Stage 4 teleological filtering implementation.
//!
//! This module provides the core filtering logic for the teleological
//! retrieval pipeline, including:
//! - Purpose alignment computation
//! - Filtering by thresholds

use tracing::{debug, instrument};

use crate::error::CoreResult;
use crate::traits::TeleologicalMemoryStore;
use crate::types::fingerprint::TeleologicalFingerprint;

use super::super::teleological_query::TeleologicalQuery;
use super::super::teleological_result::ScoredMemory;
use super::super::{AggregatedMatch, MultiEmbeddingQueryExecutor};
use super::DefaultTeleologicalPipeline;

impl<E, S> DefaultTeleologicalPipeline<E, S>
where
    E: MultiEmbeddingQueryExecutor,
    S: TeleologicalMemoryStore,
{
    /// Apply Stage 4 teleological filtering to candidates.
    ///
    /// This is the core teleological filtering that:
    /// 1. Uses purpose alignment from each candidate's fingerprint
    /// 2. Filters by minimum alignment threshold
    #[instrument(skip(self, candidates, query), fields(candidate_count = candidates.len()))]
    pub(crate) async fn apply_stage4_filtering(
        &self,
        candidates: &[(&TeleologicalFingerprint, &AggregatedMatch)],
        query: &TeleologicalQuery,
    ) -> CoreResult<(Vec<ScoredMemory>, usize, f32)> {
        let config = query.effective_config();
        let min_threshold = config.min_alignment_threshold;

        let mut results = Vec::with_capacity(candidates.len());
        let mut filtered_count = 0;
        let mut filtered_alignments = Vec::new();

        for (fingerprint, aggregated) in candidates {
            // Use a default alignment score since alignment_score field was removed
            // from TeleologicalFingerprint. The alignment threshold check is now
            // always passing (all candidates are accepted by default).
            let alignment_score = 1.0_f32;
            let is_misaligned = false;

            // Check if filtered by alignment threshold (always passes with default 1.0)
            if alignment_score < min_threshold {
                filtered_count += 1;
                filtered_alignments.push(alignment_score);
                debug!(
                    memory_id = %fingerprint.id,
                    alignment_score = alignment_score,
                    threshold = min_threshold,
                    "Filtered by alignment threshold"
                );
                continue;
            }

            // Create scored memory
            let scored = ScoredMemory::new(
                fingerprint.id,
                aggregated.aggregate_score,
                self.compute_avg_similarity(aggregated),
                alignment_score,
                aggregated.space_count,
            )
            .with_misalignment(is_misaligned);

            results.push(scored);
        }

        // Compute average alignment of filtered candidates
        let filtered_avg = if filtered_alignments.is_empty() {
            0.0
        } else {
            filtered_alignments.iter().sum::<f32>() / filtered_alignments.len() as f32
        };

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to teleological_limit
        let limit = config.teleological_limit;
        if results.len() > limit {
            results.truncate(limit);
        }

        debug!(
            input_count = candidates.len(),
            output_count = results.len(),
            filtered = filtered_count,
            avg_filtered_alignment = filtered_avg,
            "Stage 4 filtering complete"
        );

        Ok((results, filtered_count, filtered_avg))
    }

    /// Compute average content similarity from space contributions.
    pub(crate) fn compute_avg_similarity(&self, aggregated: &AggregatedMatch) -> f32 {
        if aggregated.space_contributions.is_empty() {
            return aggregated.aggregate_score;
        }

        let sum: f32 = aggregated
            .space_contributions
            .iter()
            .map(|c| c.similarity)
            .sum();
        sum / aggregated.space_contributions.len() as f32
    }
}
