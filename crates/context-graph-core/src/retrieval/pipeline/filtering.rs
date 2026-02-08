//! Stage 4 score-based filtering implementation.
//!
//! This module provides the core filtering logic for the teleological
//! retrieval pipeline:
//! - Score-based ranking and truncation

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
    /// Apply Stage 4 score-based filtering to candidates.
    ///
    /// Scores candidates by multi-space aggregation and ranks/truncates
    /// to the configured limit.
    #[instrument(skip(self, candidates, query), fields(candidate_count = candidates.len()))]
    pub(crate) async fn apply_stage4_filtering(
        &self,
        candidates: &[(&TeleologicalFingerprint, &AggregatedMatch)],
        query: &TeleologicalQuery,
    ) -> CoreResult<Vec<ScoredMemory>> {
        let config = query.effective_config();

        let mut results = Vec::with_capacity(candidates.len());

        for (fingerprint, aggregated) in candidates {
            let scored = ScoredMemory::new(
                fingerprint.id,
                aggregated.aggregate_score,
                self.compute_avg_similarity(aggregated),
                aggregated.space_count,
            );

            results.push(scored);
        }

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
            "Stage 4 filtering complete"
        );

        Ok(results)
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
