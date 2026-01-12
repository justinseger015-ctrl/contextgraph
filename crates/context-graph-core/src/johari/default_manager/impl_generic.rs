//! Generic implementation of JohariTransitionManager for DefaultJohariManager<S>.

use std::collections::HashSet;

use async_trait::async_trait;

use crate::traits::TeleologicalMemoryStore;
use crate::types::fingerprint::{JohariFingerprint, SemanticFingerprint, NUM_EMBEDDERS};
use crate::types::{JohariQuadrant, JohariTransition, TransitionTrigger};

use super::super::error::JohariError;
use super::super::external_signal::{BlindSpotCandidate, ExternalSignal};
use super::super::manager::{
    ClassificationContext, JohariTransitionManager, MemoryId, QuadrantPattern, TimeRange,
};
use super::super::stats::TransitionStats;
use super::helpers::{matches_pattern, set_quadrant_weights};
use super::types::DefaultJohariManager;

#[async_trait]
impl<S: TeleologicalMemoryStore + 'static> JohariTransitionManager for DefaultJohariManager<S> {
    async fn classify(
        &self,
        _semantic: &SemanticFingerprint,
        context: &ClassificationContext,
    ) -> Result<JohariFingerprint, JohariError> {
        let mut fingerprint = JohariFingerprint::zeroed();

        for embedder_idx in 0..NUM_EMBEDDERS {
            let delta_s = context.delta_s[embedder_idx];
            let delta_c = context.delta_c[embedder_idx];

            // Use existing classification logic from JohariFingerprint
            let quadrant = JohariFingerprint::classify_quadrant(delta_s, delta_c);

            // Apply disclosure intent override: if hidden intent, force Hidden
            let final_quadrant =
                if !context.disclosure_intent[embedder_idx] && quadrant == JohariQuadrant::Open {
                    JohariQuadrant::Hidden
                } else {
                    quadrant
                };

            // Set hard classification (100% weight to one quadrant)
            set_quadrant_weights(&mut fingerprint, embedder_idx, final_quadrant);
        }

        Ok(fingerprint)
    }

    async fn transition(
        &self,
        memory_id: MemoryId,
        embedder_idx: usize,
        to_quadrant: JohariQuadrant,
        trigger: TransitionTrigger,
    ) -> Result<JohariFingerprint, JohariError> {
        // Validate embedder index (FAIL FAST)
        if embedder_idx >= NUM_EMBEDDERS {
            return Err(JohariError::InvalidEmbedderIndex(embedder_idx));
        }

        // Retrieve current state
        let current = self
            .store
            .retrieve(memory_id)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?
            .ok_or(JohariError::NotFound(memory_id))?;

        let mut johari = current.johari.clone();
        let current_quadrant = johari.dominant_quadrant(embedder_idx);

        // Validate transition using existing state machine
        if !current_quadrant.can_transition_to(to_quadrant) {
            return Err(JohariError::InvalidTransition {
                from: current_quadrant,
                to: to_quadrant,
                embedder_idx,
            });
        }

        // Validate trigger is valid for this transition
        if current_quadrant
            .transition_to(to_quadrant, trigger)
            .is_err()
        {
            return Err(JohariError::InvalidTrigger {
                from: current_quadrant,
                to: to_quadrant,
                trigger,
            });
        }

        // Apply transition (set 100% weight to new quadrant)
        set_quadrant_weights(&mut johari, embedder_idx, to_quadrant);

        // Update stored fingerprint
        let mut updated = current;
        updated.johari = johari.clone();

        self.store
            .update(updated)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?;

        // Record the transition in history
        let transition_record = JohariTransition::new(
            memory_id,
            embedder_idx,
            current_quadrant,
            to_quadrant,
            trigger,
        );
        self.record_transition(transition_record).await;

        Ok(johari)
    }

    async fn transition_batch(
        &self,
        memory_id: MemoryId,
        transitions: Vec<(usize, JohariQuadrant, TransitionTrigger)>,
    ) -> Result<JohariFingerprint, JohariError> {
        // Retrieve current state
        let current = self
            .store
            .retrieve(memory_id)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?
            .ok_or(JohariError::NotFound(memory_id))?;

        let mut johari = current.johari.clone();

        // Validate ALL transitions first (all-or-nothing)
        for (idx, (embedder_idx, to_quadrant, trigger)) in transitions.iter().enumerate() {
            // Check embedder index bounds
            if *embedder_idx >= NUM_EMBEDDERS {
                return Err(JohariError::BatchValidationFailed {
                    idx,
                    reason: format!("Invalid embedder index: {}", embedder_idx),
                });
            }

            let current_quadrant = johari.dominant_quadrant(*embedder_idx);

            // Check transition validity
            if !current_quadrant.can_transition_to(*to_quadrant) {
                return Err(JohariError::BatchValidationFailed {
                    idx,
                    reason: format!(
                        "Invalid transition {:?} → {:?} for embedder {}",
                        current_quadrant, to_quadrant, embedder_idx
                    ),
                });
            }

            // Check trigger validity
            if current_quadrant
                .transition_to(*to_quadrant, *trigger)
                .is_err()
            {
                return Err(JohariError::BatchValidationFailed {
                    idx,
                    reason: format!(
                        "Invalid trigger {:?} for {:?} → {:?}",
                        trigger, current_quadrant, to_quadrant
                    ),
                });
            }
        }

        // Collect transition records before applying
        // We need to capture current quadrants before applying the transitions
        let original_johari = current.johari.clone();
        let mut transition_records = Vec::with_capacity(transitions.len());

        // Apply all transitions (all validated) and record them
        for (embedder_idx, to_quadrant, trigger) in transitions {
            let from_quadrant = original_johari.dominant_quadrant(embedder_idx);
            set_quadrant_weights(&mut johari, embedder_idx, to_quadrant);

            // Create transition record
            transition_records.push(JohariTransition::new(
                memory_id,
                embedder_idx,
                from_quadrant,
                to_quadrant,
                trigger,
            ));
        }

        // Persist
        let mut updated = current;
        updated.johari = johari.clone();

        self.store
            .update(updated)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?;

        // Record all transitions in history
        for record in transition_records {
            self.record_transition(record).await;
        }

        Ok(johari)
    }

    async fn find_by_quadrant(
        &self,
        pattern: QuadrantPattern,
        limit: usize,
    ) -> Result<Vec<(MemoryId, JohariFingerprint)>, JohariError> {
        // AP-007: Use proper list_all_johari scan instead of broken zeroed query
        // Zeroed semantic queries have undefined cosine similarity and return wrong results.
        let all_johari = self
            .store
            .list_all_johari(limit * 10) // Fetch extra to allow for filtering
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?;

        let matches: Vec<_> = all_johari
            .into_iter()
            .filter(|(_, johari)| matches_pattern(johari, &pattern))
            .take(limit)
            .collect();

        Ok(matches)
    }

    async fn discover_blind_spots(
        &self,
        memory_id: MemoryId,
        external_signals: &[ExternalSignal],
    ) -> Result<Vec<BlindSpotCandidate>, JohariError> {
        let current = self
            .store
            .retrieve(memory_id)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?
            .ok_or(JohariError::NotFound(memory_id))?;

        let johari = &current.johari;
        let mut candidates = Vec::new();

        for embedder_idx in 0..NUM_EMBEDDERS {
            let current_quadrant = johari.dominant_quadrant(embedder_idx);

            // Only consider Unknown or Hidden as potential blind spots
            if current_quadrant != JohariQuadrant::Unknown
                && current_quadrant != JohariQuadrant::Hidden
            {
                continue;
            }

            // Aggregate signal strength for this embedder
            let mut signal_strength = 0.0f32;
            let mut sources = Vec::new();

            for signal in external_signals {
                if signal.embedder_idx == embedder_idx {
                    signal_strength += signal.strength;
                    sources.push(signal.source.clone());
                }
            }

            if signal_strength > self.blind_spot_threshold {
                candidates.push(BlindSpotCandidate::new(
                    embedder_idx,
                    current_quadrant,
                    signal_strength,
                    sources,
                ));
            }
        }

        // Sort by signal strength descending
        candidates.sort_by(|a, b| {
            b.signal_strength
                .partial_cmp(&a.signal_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(candidates)
    }

    async fn get_transition_stats(
        &self,
        time_range: TimeRange,
    ) -> Result<TransitionStats, JohariError> {
        let transitions = self.transitions.read().await;
        let mut stats = TransitionStats::new();
        let mut affected_memories: HashSet<MemoryId> = HashSet::new();

        for t in transitions.iter() {
            // Filter by time range
            if t.timestamp >= time_range.start && t.timestamp < time_range.end {
                stats.record(t.from, t.to, t.trigger, t.embedder_idx);
                affected_memories.insert(t.memory_id);
            }
        }

        stats.memories_affected = affected_memories.len() as u64;
        stats.finalize();

        Ok(stats)
    }

    async fn get_transition_history(
        &self,
        memory_id: MemoryId,
        limit: usize,
    ) -> Result<Vec<JohariTransition>, JohariError> {
        let transitions = self.transitions.read().await;

        // Filter by memory_id, take up to limit (already sorted by newest first)
        let history: Vec<JohariTransition> = transitions
            .iter()
            .filter(|t| t.memory_id == memory_id)
            .take(limit)
            .cloned()
            .collect();

        Ok(history)
    }
}
