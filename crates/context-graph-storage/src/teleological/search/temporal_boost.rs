//! Temporal boost functions for E2/E3/E4 POST-retrieval scoring.
//!
//! Per ARCH-14: Temporal embedders are applied POST-retrieval, not in similarity scoring.
//!
//! # Embedder Roles
//!
//! - **E2 (V_freshness)**: Recency scoring with configurable decay functions
//! - **E3 (V_periodicity)**: Periodic pattern matching (hour-of-day, day-of-week)
//! - **E4 (V_ordering)**: Sequence understanding (before/after relationships)
//!
//! # Design Philosophy
//!
//! Documents created at the same time are NOT necessarily on the same topic.
//! Temporal embedders measure TIME proximity, not TOPIC similarity.
//! Therefore, temporal scores are applied as POST-retrieval boosts.
//!
//! # Research References
//!
//! - [Cascading Retrieval](https://www.pinecone.io/blog/cascading-retrieval/)
//! - [ACM TOIS Fusion](https://dl.acm.org/doi/10.1145/3596512)

use std::collections::HashMap;

use chrono::{DateTime, Datelike, Timelike, Utc};
use context_graph_core::traits::{
    DecayFunction, SequenceDirection, TemporalSearchOptions,
    TimeWindow, TeleologicalSearchResult,
};
use context_graph_core::types::fingerprint::SemanticFingerprint;
use tracing::debug;
use uuid::Uuid;

// =============================================================================
// SHARED UTILITY FUNCTIONS
// =============================================================================

/// Compute cosine similarity between two vectors.
///
/// Returns a similarity score in [0.0, 1.0] where 1.0 is identical.
/// Handles edge cases: empty vectors, mismatched lengths.
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

    let norm = (norm_a.sqrt() * norm_b.sqrt()).max(1e-8);
    (dot / norm).clamp(0.0, 1.0)
}

/// Extract hour and day-of-week from a timestamp in milliseconds.
///
/// Returns (hour: 0-23, day_of_week: 0-6 where 0=Sunday).
fn extract_temporal_components(timestamp_ms: i64) -> (u8, u8) {
    let datetime = DateTime::from_timestamp_millis(timestamp_ms)
        .unwrap_or_else(Utc::now);
    let hour = datetime.hour() as u8;
    // chrono weekday: Mon=0, Sun=6, but we use Sun=0, Sat=6
    let dow = datetime.weekday().num_days_from_sunday() as u8;
    (hour, dow)
}

// =============================================================================
// E2 RECENCY FUNCTIONS
// =============================================================================

/// Compute E2 recency score with configurable decay function.
///
/// # Arguments
///
/// * `memory_timestamp_ms` - Timestamp of the memory in milliseconds
/// * `query_timestamp_ms` - Current time in milliseconds
/// * `options` - Temporal search options with decay configuration
///
/// # Returns
///
/// Recency score [0.0, 1.0] where 1.0 is most recent
pub fn compute_e2_recency_score(
    memory_timestamp_ms: i64,
    query_timestamp_ms: i64,
    options: &TemporalSearchOptions,
) -> f32 {
    // Age in seconds
    let age_secs = ((query_timestamp_ms - memory_timestamp_ms).max(0) / 1000) as f64;

    match options.decay_function {
        DecayFunction::Linear => {
            // Linear decay: score = 1.0 - (age / max_age)
            // Max age is based on temporal scale
            let max_age_secs = options.temporal_scale.horizon_seconds() as f64;
            let score = 1.0 - (age_secs / max_age_secs).min(1.0);
            score.max(0.0) as f32
        }
        DecayFunction::Exponential => {
            // Exponential decay: score = exp(-age * ln(2) / half_life)
            // This gives score = 0.5 at half_life
            let half_life_secs = options.effective_half_life() as f64;
            let lambda = 0.693147 / half_life_secs; // ln(2) / half_life
            let score = (-age_secs * lambda).exp();
            score as f32
        }
        DecayFunction::Step => {
            // Step function with configurable time buckets
            let age_secs_u64 = age_secs as u64;
            for &(threshold, score) in &options.step_buckets {
                if age_secs_u64 <= threshold {
                    return score;
                }
            }
            0.1 // Default for items older than all buckets
        }
        DecayFunction::NoDecay => {
            // No decay - all memories have equal recency score
            1.0
        }
    }
}

/// Compute E2 recency score with adaptive half-life based on corpus size.
///
/// E2 decay accuracy degrades at larger corpus sizes with fixed half-life.
/// This variant scales the half-life to maintain accuracy across scales.
///
/// # Arguments
///
/// * `memory_timestamp_ms` - Timestamp of the memory in milliseconds
/// * `query_timestamp_ms` - Current time in milliseconds
/// * `options` - Temporal search options with decay configuration
/// * `corpus_size` - Number of memories in the corpus
///
/// # Returns
///
/// Recency score [0.0, 1.0] where 1.0 is most recent
pub fn compute_e2_recency_score_adaptive(
    memory_timestamp_ms: i64,
    query_timestamp_ms: i64,
    options: &TemporalSearchOptions,
    corpus_size: usize,
) -> f32 {
    // Age in seconds
    let age_secs = ((query_timestamp_ms - memory_timestamp_ms).max(0) / 1000) as f64;

    match options.decay_function {
        DecayFunction::Linear => {
            // Linear decay uses temporal scale, not half-life
            let max_age_secs = options.temporal_scale.horizon_seconds() as f64;
            let score = 1.0 - (age_secs / max_age_secs).min(1.0);
            score.max(0.0) as f32
        }
        DecayFunction::Exponential => {
            // Use adaptive half-life based on corpus size
            let half_life_secs = options.adaptive_half_life(corpus_size) as f64;
            let lambda = 0.693147 / half_life_secs; // ln(2) / half_life
            let score = (-age_secs * lambda).exp();
            score as f32
        }
        DecayFunction::Step => {
            // Step function doesn't use half-life
            let age_secs_u64 = age_secs as u64;
            for &(threshold, score) in &options.step_buckets {
                if age_secs_u64 <= threshold {
                    return score;
                }
            }
            0.1
        }
        DecayFunction::NoDecay => 1.0,
    }
}

/// Compute E2 recency score using the E2 embedding similarity.
///
/// Uses cosine similarity between query E2 and memory E2 embeddings.
/// This captures learned temporal patterns, not just raw timestamps.
///
/// # Arguments
///
/// * `query_e2` - Query E2 temporal embedding
/// * `memory_e2` - Memory E2 temporal embedding
///
/// # Returns
///
/// Similarity score [0.0, 1.0]
#[inline]
pub fn compute_e2_embedding_similarity(query_e2: &[f32], memory_e2: &[f32]) -> f32 {
    cosine_similarity(query_e2, memory_e2)
}

// =============================================================================
// E3 PERIODIC FUNCTIONS
// =============================================================================

/// Compute E3 periodic pattern similarity.
///
/// Uses cosine similarity between query E3 and memory E3 embeddings.
/// E3 embeddings capture hour-of-day and day-of-week patterns.
///
/// # Arguments
///
/// * `query_e3` - Query E3 periodic embedding (or generated from target time)
/// * `memory_e3` - Memory E3 periodic embedding
///
/// # Returns
///
/// Similarity score [0.0, 1.0]
#[inline]
pub fn compute_e3_periodic_score(query_e3: &[f32], memory_e3: &[f32]) -> f32 {
    cosine_similarity(query_e3, memory_e3)
}

/// Compute periodic match score based on hour and day of week.
///
/// This is a fallback when E3 embeddings are not available.
/// Uses simple hour/day matching with configurable tolerances.
///
/// # Arguments
///
/// * `target_hour` - Target hour of day (0-23)
/// * `memory_hour` - Memory's creation hour
/// * `target_dow` - Target day of week (0=Sun, 6=Sat)
/// * `memory_dow` - Memory's creation day of week
///
/// # Returns
///
/// Match score [0.0, 1.0]
pub fn compute_periodic_match_fallback(
    target_hour: Option<u8>,
    memory_hour: u8,
    target_dow: Option<u8>,
    memory_dow: u8,
) -> f32 {
    let mut score = 0.0f32;
    let mut factors = 0;

    // Hour matching with tolerance
    if let Some(th) = target_hour {
        factors += 1;
        let hour_diff = ((th as i16 - memory_hour as i16).abs() % 24).min(
            24 - (th as i16 - memory_hour as i16).abs() % 24
        ) as f32;
        // Score: 1.0 for exact match, 0.0 for 12 hours apart
        score += (1.0 - hour_diff / 12.0).max(0.0);
    }

    // Day of week matching
    if let Some(td) = target_dow {
        factors += 1;
        let dow_diff = ((td as i16 - memory_dow as i16).abs() % 7).min(
            7 - (td as i16 - memory_dow as i16).abs() % 7
        ) as f32;
        // Score: 1.0 for exact match, 0.0 for 3.5 days apart
        score += (1.0 - dow_diff / 3.5).max(0.0);
    }

    if factors > 0 {
        score / factors as f32
    } else {
        0.5 // Neutral if no targets specified
    }
}

// =============================================================================
// E4 SEQUENCE FUNCTIONS
// =============================================================================

/// Compute E4 sequence proximity score.
///
/// Uses cosine similarity between anchor E4 and memory E4 embeddings.
/// E4 embeddings capture positional/sequence information.
///
/// # Arguments
///
/// * `anchor_e4` - Anchor memory's E4 positional embedding
/// * `memory_e4` - Memory E4 positional embedding
/// * `memory_ts` - Memory timestamp
/// * `anchor_ts` - Anchor timestamp
/// * `direction` - Search direction (Before, After, Both)
///
/// # Returns
///
/// Similarity score [0.0, 1.0], 0.0 if direction constraint not met
pub fn compute_e4_sequence_score(
    anchor_e4: &[f32],
    memory_e4: &[f32],
    memory_ts: i64,
    anchor_ts: i64,
    direction: SequenceDirection,
) -> f32 {
    // Check direction constraint first
    match direction {
        SequenceDirection::Before => {
            if memory_ts >= anchor_ts {
                return 0.0;
            }
        }
        SequenceDirection::After => {
            if memory_ts <= anchor_ts {
                return 0.0;
            }
        }
        SequenceDirection::Both => {
            // No constraint - include all
        }
    }

    // Compute E4 cosine similarity using shared function
    cosine_similarity(anchor_e4, memory_e4)
}

/// Compute sequence proximity score based on timestamp distance.
///
/// This is a fallback when E4 embeddings are not available.
///
/// # Arguments
///
/// * `memory_ts` - Memory timestamp
/// * `anchor_ts` - Anchor timestamp
/// * `direction` - Search direction
/// * `max_distance_secs` - Maximum temporal distance for scoring
///
/// # Returns
///
/// Proximity score [0.0, 1.0], 0.0 if direction constraint not met
pub fn compute_sequence_proximity_fallback(
    memory_ts: i64,
    anchor_ts: i64,
    direction: SequenceDirection,
    max_distance_secs: u64,
) -> f32 {
    // Check direction constraint
    match direction {
        SequenceDirection::Before => {
            if memory_ts >= anchor_ts {
                return 0.0;
            }
        }
        SequenceDirection::After => {
            if memory_ts <= anchor_ts {
                return 0.0;
            }
        }
        SequenceDirection::Both => {}
    }

    // Compute proximity (closer = higher score) using linear decay
    let distance_ms = (memory_ts - anchor_ts).abs();
    let distance_secs = distance_ms / 1000;
    let max_distance = max_distance_secs as i64;

    let proximity = 1.0 - (distance_secs.min(max_distance) as f32 / max_distance as f32);
    proximity.max(0.0)
}

/// Compute sequence proximity with exponential decay.
///
/// This provides better fallback behavior that matches E4's learned distance semantics.
/// Uses exponential decay: score = exp(-distance / characteristic)
/// where characteristic = max_distance / 3 for reasonable falloff.
///
/// # Arguments
///
/// * `memory_ts` - Memory timestamp (milliseconds)
/// * `anchor_ts` - Anchor timestamp (milliseconds)
/// * `direction` - Search direction (Before, After, Both)
/// * `max_distance_secs` - Maximum temporal distance for scoring
///
/// # Returns
///
/// Proximity score [0.0, 1.0], 0.0 if direction constraint not met
pub fn compute_sequence_proximity_exponential(
    memory_ts: i64,
    anchor_ts: i64,
    direction: SequenceDirection,
    max_distance_secs: u64,
) -> f32 {
    // Check direction constraint
    match direction {
        SequenceDirection::Before => {
            if memory_ts >= anchor_ts {
                return 0.0;
            }
        }
        SequenceDirection::After => {
            if memory_ts <= anchor_ts {
                return 0.0;
            }
        }
        SequenceDirection::Both => {}
    }

    // Compute proximity with exponential decay
    let distance_secs = (memory_ts - anchor_ts).abs() as f64 / 1000.0;

    // Characteristic decay constant: items at max_distance/3 get ~37% score
    let characteristic = max_distance_secs as f64 / 3.0;

    // Exponential decay: closer items score higher
    ((-distance_secs / characteristic).exp() as f32).clamp(0.0, 1.0)
}

// =============================================================================
// TIME WINDOW FILTERING
// =============================================================================

/// Filter results by time window.
///
/// Removes results with timestamps outside the specified window.
///
/// # Arguments
///
/// * `results` - Search results to filter (modified in place)
/// * `window` - Time window specification
/// * `get_timestamp` - Function to extract timestamp from result
pub fn filter_by_time_window<F>(
    results: &mut Vec<TeleologicalSearchResult>,
    window: &TimeWindow,
    get_timestamp: F,
) where
    F: Fn(&TeleologicalSearchResult) -> i64,
{
    if !window.is_defined() {
        return;
    }

    let original_len = results.len();
    results.retain(|r| window.contains(get_timestamp(r)));

    debug!(
        "Time window filter: {} -> {} results",
        original_len,
        results.len()
    );
}

/// Filter results by session ID.
///
/// Removes results that don't match the specified session.
///
/// # Arguments
///
/// * `results` - Search results to filter (modified in place)
/// * `session_id` - Target session ID
/// * `get_session` - Function to extract session ID from result
pub fn filter_by_session<F>(
    results: &mut Vec<TeleologicalSearchResult>,
    session_id: &str,
    get_session: F,
) where
    F: Fn(&TeleologicalSearchResult) -> Option<&str>,
{
    let original_len = results.len();
    results.retain(|r| {
        get_session(r).map_or(false, |sid| sid == session_id)
    });

    debug!(
        "Session filter '{}': {} -> {} results",
        session_id,
        original_len,
        results.len()
    );
}

// =============================================================================
// COMBINED TEMPORAL BOOST
// =============================================================================

/// Combined temporal boost data for a single memory.
#[derive(Debug, Clone)]
pub struct TemporalBoostData {
    /// E2 recency score [0.0, 1.0]
    pub recency_score: f32,
    /// E3 periodic score [0.0, 1.0]
    pub periodic_score: f32,
    /// E4 sequence score [0.0, 1.0]
    pub sequence_score: f32,
    /// Combined temporal score [0.0, 1.0]
    pub combined_score: f32,
}

impl Default for TemporalBoostData {
    fn default() -> Self {
        Self {
            recency_score: 1.0,
            periodic_score: 0.5,
            sequence_score: 0.5,
            combined_score: 0.5,
        }
    }
}

/// Apply all temporal boosts POST-retrieval per ARCH-14.
///
/// This function:
/// 1. Computes E2 recency scores (if decay function is active)
/// 2. Computes E3 periodic scores (if periodic options are set)
/// 3. Computes E4 sequence scores (if sequence options are set)
/// 4. Combines boosts with configurable weights
/// 5. Re-sorts results by final score
///
/// # Arguments
///
/// * `results` - Search results to boost (modified in place)
/// * `query_fp` - Query semantic fingerprint (for embedding comparisons)
/// * `options` - Temporal search options
/// * `fingerprints` - Map of memory IDs to their fingerprints
/// * `timestamps` - Map of memory IDs to their timestamps (ms)
/// * `anchor_fp` - Optional anchor fingerprint for sequence queries
/// * `anchor_ts` - Optional anchor timestamp for sequence queries
///
/// # Returns
///
/// Map of memory IDs to their temporal boost data (for debugging/logging)
pub fn apply_temporal_boosts(
    results: &mut Vec<TeleologicalSearchResult>,
    query_fp: &SemanticFingerprint,
    options: &TemporalSearchOptions,
    fingerprints: &HashMap<Uuid, SemanticFingerprint>,
    timestamps: &HashMap<Uuid, i64>,
    anchor_fp: Option<&SemanticFingerprint>,
    anchor_ts: Option<i64>,
) -> HashMap<Uuid, TemporalBoostData> {
    let now_ms = chrono::Utc::now().timestamp_millis();
    let temporal_weight = options.temporal_weight;

    // If no temporal weight, skip all processing
    if temporal_weight <= 0.0 {
        return HashMap::new();
    }

    let mut boost_data: HashMap<Uuid, TemporalBoostData> = HashMap::new();

    // Compute individual component weights from configured weights
    // Only include weights for active components, then normalize
    let has_recency = options.decay_function.is_active();
    let has_periodic = options.periodic_options.is_some();
    let has_sequence = options.sequence_options.is_some();

    if !has_recency && !has_periodic && !has_sequence {
        return HashMap::new();
    }

    // Use configured weights (default: 0.50/0.15/0.35 from benchmark optimization)
    let (w_recency, w_periodic, w_sequence) = options.component_weights;

    // Zero out weights for inactive components
    let raw_recency = if has_recency { w_recency } else { 0.0 };
    let raw_periodic = if has_periodic { w_periodic } else { 0.0 };
    let raw_sequence = if has_sequence { w_sequence } else { 0.0 };

    // Normalize active weights to sum to 1.0
    let total_weight = raw_recency + raw_periodic + raw_sequence;
    let (recency_weight, periodic_weight, sequence_weight) = if total_weight > f32::EPSILON {
        (
            raw_recency / total_weight,
            raw_periodic / total_weight,
            raw_sequence / total_weight,
        )
    } else {
        // Fallback: equal among active
        let active_count = has_recency as u8 + has_periodic as u8 + has_sequence as u8;
        let w = 1.0 / active_count.max(1) as f32;
        (
            if has_recency { w } else { 0.0 },
            if has_periodic { w } else { 0.0 },
            if has_sequence { w } else { 0.0 },
        )
    };

    debug!(
        "Temporal boost weights: recency={:.2}, periodic={:.2}, sequence={:.2}, master={:.2}",
        recency_weight, periodic_weight, sequence_weight, temporal_weight
    );

    for result in results.iter_mut() {
        let id = result.fingerprint.id;
        let memory_fp = fingerprints.get(&id);
        let memory_ts = timestamps.get(&id).copied().unwrap_or(0);

        let mut data = TemporalBoostData::default();

        // E2 Recency
        if has_recency {
            if let Some(fp) = memory_fp {
                // Prefer embedding-based similarity if query has E2
                if !query_fp.e2_temporal_recent.is_empty() && !fp.e2_temporal_recent.is_empty() {
                    data.recency_score = compute_e2_embedding_similarity(
                        &query_fp.e2_temporal_recent,
                        &fp.e2_temporal_recent,
                    );
                } else {
                    // Fall back to timestamp-based decay
                    data.recency_score = compute_e2_recency_score(memory_ts, now_ms, options);
                }
            } else {
                data.recency_score = compute_e2_recency_score(memory_ts, now_ms, options);
            }
        }

        // E3 Periodic
        if let Some(ref periodic_opts) = options.periodic_options {
            if let Some(fp) = memory_fp {
                // Prefer embedding-based similarity
                if !query_fp.e3_temporal_periodic.is_empty() && !fp.e3_temporal_periodic.is_empty() {
                    data.periodic_score = compute_e3_periodic_score(
                        &query_fp.e3_temporal_periodic,
                        &fp.e3_temporal_periodic,
                    );
                } else {
                    // Fall back to hour/day matching using chrono
                    let (memory_hour, memory_dow) = extract_temporal_components(memory_ts);
                    data.periodic_score = compute_periodic_match_fallback(
                        periodic_opts.effective_hour(),
                        memory_hour,
                        periodic_opts.effective_day_of_week(),
                        memory_dow,
                    );
                }
            }
        }

        // E4 Sequence
        if let Some(ref sequence_opts) = options.sequence_options {
            if let (Some(anchor), Some(anchor_time)) = (anchor_fp, anchor_ts) {
                if let Some(fp) = memory_fp {
                    // Prefer embedding-based similarity
                    if !anchor.e4_temporal_positional.is_empty() && !fp.e4_temporal_positional.is_empty() {
                        data.sequence_score = compute_e4_sequence_score(
                            &anchor.e4_temporal_positional,
                            &fp.e4_temporal_positional,
                            memory_ts,
                            anchor_time,
                            sequence_opts.direction,
                        );
                    } else {
                        // Fall back to timestamp proximity
                        let max_distance = sequence_opts.max_distance as u64 * 60; // Convert positions to seconds
                        data.sequence_score = if sequence_opts.use_exponential_fallback {
                            compute_sequence_proximity_exponential(
                                memory_ts,
                                anchor_time,
                                sequence_opts.direction,
                                max_distance,
                            )
                        } else {
                            compute_sequence_proximity_fallback(
                                memory_ts,
                                anchor_time,
                                sequence_opts.direction,
                                max_distance,
                            )
                        };
                    }
                }
            }
        }

        // Combine component scores
        data.combined_score =
            data.recency_score * recency_weight +
            data.periodic_score * periodic_weight +
            data.sequence_score * sequence_weight;

        // Apply temporal boost to final similarity
        // formula: final = semantic * (1 - weight) + temporal * weight
        let original = result.similarity;
        result.similarity = original * (1.0 - temporal_weight) + data.combined_score * temporal_weight;

        debug!(
            "Temporal boost for {}: {} -> {} (recency={:.3}, periodic={:.3}, sequence={:.3})",
            id, original, result.similarity,
            data.recency_score, data.periodic_score, data.sequence_score
        );

        boost_data.insert(id, data);
    }

    // Re-sort results by boosted similarity
    results.sort_by(|a, b| {
        b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal)
    });

    boost_data
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use context_graph_core::traits::TemporalScale;

    #[test]
    fn test_linear_decay() {
        let options = TemporalSearchOptions::default()
            .with_decay_function(DecayFunction::Linear)
            .with_temporal_scale(TemporalScale::Meso); // 1 hour horizon

        let now = 1000000000i64;

        // Fresh (0 age) = 1.0
        assert!((compute_e2_recency_score(now, now, &options) - 1.0).abs() < 0.01);

        // Half age = 0.5
        let half_horizon = (TemporalScale::Meso.horizon_seconds() / 2) as i64 * 1000;
        let score = compute_e2_recency_score(now - half_horizon, now, &options);
        assert!((score - 0.5).abs() < 0.01, "Expected ~0.5, got {}", score);

        // Full age = 0.0
        let full_horizon = TemporalScale::Meso.horizon_seconds() as i64 * 1000;
        let score = compute_e2_recency_score(now - full_horizon, now, &options);
        assert!(score < 0.01, "Expected ~0.0, got {}", score);
    }

    #[test]
    fn test_exponential_decay() {
        let options = TemporalSearchOptions::default()
            .with_decay_function(DecayFunction::Exponential)
            .with_decay_half_life(3600); // 1 hour half-life

        let now = 1000000000i64;

        // Fresh (0 age) = 1.0
        assert!((compute_e2_recency_score(now, now, &options) - 1.0).abs() < 0.01);

        // At half-life = 0.5
        let half_life_ms = 3600 * 1000;
        let score = compute_e2_recency_score(now - half_life_ms, now, &options);
        assert!((score - 0.5).abs() < 0.05, "Expected ~0.5, got {}", score);

        // At 2x half-life = 0.25
        let score = compute_e2_recency_score(now - 2 * half_life_ms, now, &options);
        assert!((score - 0.25).abs() < 0.05, "Expected ~0.25, got {}", score);
    }

    #[test]
    fn test_step_decay() {
        let options = TemporalSearchOptions::default()
            .with_decay_function(DecayFunction::Step);

        let now = 1000000000i64;

        // < 5 min = 1.0
        assert_eq!(compute_e2_recency_score(now - 60_000, now, &options), 1.0);

        // < 1 hour = 0.8
        assert_eq!(compute_e2_recency_score(now - 1800_000, now, &options), 0.8);

        // < 1 day = 0.5
        assert_eq!(compute_e2_recency_score(now - 43200_000, now, &options), 0.5);

        // > 1 week = 0.1
        assert_eq!(compute_e2_recency_score(now - 1000000_000, now, &options), 0.1);
    }

    #[test]
    fn test_no_decay() {
        let options = TemporalSearchOptions::default()
            .with_decay_function(DecayFunction::NoDecay);

        let now = 1000000000i64;

        // All timestamps return 1.0
        assert_eq!(compute_e2_recency_score(now, now, &options), 1.0);
        assert_eq!(compute_e2_recency_score(now - 1000000_000, now, &options), 1.0);
    }

    #[test]
    fn test_time_window_contains() {
        let window = TimeWindow {
            start_ms: Some(1000),
            end_ms: Some(2000),
        };

        assert!(!window.contains(999));
        assert!(window.contains(1000));
        assert!(window.contains(1500));
        assert!(!window.contains(2000));
    }

    #[test]
    fn test_sequence_direction_filter() {
        let anchor_ts = 1000i64;
        let anchor_e4 = vec![1.0; 512];
        let memory_e4 = vec![1.0; 512]; // Perfect match

        // Before: memory must be before anchor
        let score = compute_e4_sequence_score(&anchor_e4, &memory_e4, 500, anchor_ts, SequenceDirection::Before);
        assert!(score > 0.9, "Before direction should match");

        let score = compute_e4_sequence_score(&anchor_e4, &memory_e4, 1500, anchor_ts, SequenceDirection::Before);
        assert_eq!(score, 0.0, "Before direction should reject after");

        // After: memory must be after anchor
        let score = compute_e4_sequence_score(&anchor_e4, &memory_e4, 1500, anchor_ts, SequenceDirection::After);
        assert!(score > 0.9, "After direction should match");

        let score = compute_e4_sequence_score(&anchor_e4, &memory_e4, 500, anchor_ts, SequenceDirection::After);
        assert_eq!(score, 0.0, "After direction should reject before");

        // Both: accept all
        let score = compute_e4_sequence_score(&anchor_e4, &memory_e4, 500, anchor_ts, SequenceDirection::Both);
        assert!(score > 0.9, "Both direction should match before");

        let score = compute_e4_sequence_score(&anchor_e4, &memory_e4, 1500, anchor_ts, SequenceDirection::Both);
        assert!(score > 0.9, "Both direction should match after");
    }

    #[test]
    fn test_periodic_match_fallback() {
        // Exact hour match
        let score = compute_periodic_match_fallback(Some(14), 14, None, 0);
        assert!((score - 1.0).abs() < 0.01);

        // 6 hours apart
        let score = compute_periodic_match_fallback(Some(14), 8, None, 0);
        assert!((score - 0.5).abs() < 0.01);

        // 12 hours apart (opposite)
        let score = compute_periodic_match_fallback(Some(14), 2, None, 0);
        assert!(score < 0.1);
    }

    #[test]
    fn test_e2_embedding_similarity() {
        let query = vec![1.0, 0.0, 0.0];
        let same = vec![1.0, 0.0, 0.0];
        let orthogonal = vec![0.0, 1.0, 0.0];

        assert!((compute_e2_embedding_similarity(&query, &same) - 1.0).abs() < 0.01);
        assert!(compute_e2_embedding_similarity(&query, &orthogonal) < 0.01);
    }
}
