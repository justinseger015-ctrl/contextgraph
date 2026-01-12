//! PruningService implementation
//!
//! Core service for identifying and removing low-value memories from the knowledge graph.
//! Uses alignment scores, age, connection count, and quality metrics to determine
//! which memories should be pruned.

use chrono::{DateTime, Utc};
use std::collections::HashMap;

use crate::autonomous::curation::MemoryId;

use super::types::{
    ExtendedPruningConfig, MemoryMetadata, PruneReason, PruningCandidate, PruningReport,
};

// ============================================================================
// PruningService
// ============================================================================

/// Service for identifying and removing low-value memories
#[derive(Clone, Debug)]
pub struct PruningService {
    /// Configuration for pruning behavior
    config: ExtendedPruningConfig,
    /// Count of prunes today
    daily_prune_count: u32,
    /// Date of current day for daily limit tracking
    current_day: DateTime<Utc>,
    /// Known content hashes for redundancy detection
    known_hashes: HashMap<u64, MemoryId>,
}

impl PruningService {
    /// Create a new PruningService with default configuration
    pub fn new() -> Self {
        Self {
            config: ExtendedPruningConfig::default(),
            daily_prune_count: 0,
            current_day: Utc::now()
                .date_naive()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc(),
            known_hashes: HashMap::new(),
        }
    }

    /// Create a new PruningService with custom configuration
    pub fn with_config(config: ExtendedPruningConfig) -> Self {
        Self {
            config,
            daily_prune_count: 0,
            current_day: Utc::now()
                .date_naive()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc(),
            known_hashes: HashMap::new(),
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &ExtendedPruningConfig {
        &self.config
    }

    /// Reset daily counter if day has changed
    fn check_day_rollover(&mut self) {
        let today = Utc::now()
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        if today > self.current_day {
            self.daily_prune_count = 0;
            self.current_day = today;
        }
    }

    /// Get remaining prunes allowed today
    pub fn remaining_daily_prunes(&mut self) -> u32 {
        self.check_day_rollover();
        self.config
            .max_daily_prunes
            .saturating_sub(self.daily_prune_count)
    }

    /// Identify all pruning candidates from a set of memories
    ///
    /// # Fail-Fast
    /// Returns empty vec if memories is empty (not an error condition).
    pub fn identify_candidates(&self, memories: &[MemoryMetadata]) -> Vec<PruningCandidate> {
        if memories.is_empty() {
            return Vec::new();
        }

        // Build hash map for redundancy detection
        let mut hash_to_first: HashMap<u64, &MemoryMetadata> = HashMap::new();
        let mut redundant_ids: HashMap<MemoryId, MemoryId> = HashMap::new();

        for memory in memories {
            if let Some(hash) = memory.content_hash {
                if let Some(first) = hash_to_first.get(&hash) {
                    // This is a duplicate - the one with lower alignment is redundant
                    if memory.alignment < first.alignment {
                        redundant_ids.insert(memory.id.clone(), first.id.clone());
                    }
                } else {
                    hash_to_first.insert(hash, memory);
                }
            }
        }

        let mut candidates = Vec::new();

        for memory in memories {
            if let Some(candidate) = self.evaluate_candidate_internal(memory, &redundant_ids) {
                candidates.push(candidate);
            }
        }

        // Sort by priority score (highest first)
        candidates.sort_by(|a, b| {
            b.priority_score
                .partial_cmp(&a.priority_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
    }

    /// Evaluate a single memory for pruning
    ///
    /// Returns Some(PruningCandidate) if the memory should be pruned,
    /// None if it should be kept.
    pub fn evaluate_candidate(&self, metadata: &MemoryMetadata) -> Option<PruningCandidate> {
        self.evaluate_candidate_internal(metadata, &HashMap::new())
    }

    /// Internal evaluation with redundancy context
    fn evaluate_candidate_internal(
        &self,
        metadata: &MemoryMetadata,
        redundant_ids: &HashMap<MemoryId, MemoryId>,
    ) -> Option<PruningCandidate> {
        if !self.config.base.enabled {
            return None;
        }

        let age_days = metadata.age_days();

        // Must be old enough to prune
        if age_days < self.config.base.min_age_days {
            return None;
        }

        // Check for prune reason
        if let Some(reason) = self.get_prune_reason_internal(metadata, redundant_ids) {
            // Check if should be preserved due to connections
            if self.config.base.preserve_connected
                && metadata.connection_count >= self.config.base.min_connections
            {
                return None;
            }

            return Some(PruningCandidate::new(
                metadata.id.clone(),
                age_days,
                metadata.alignment,
                metadata.connection_count,
                reason,
                metadata.byte_size,
            ));
        }

        None
    }

    /// Determine if a candidate should be pruned
    ///
    /// Takes into account daily limits and preservation rules.
    pub fn should_prune(&self, candidate: &PruningCandidate) -> bool {
        // Check if preserved by connection count
        if self.config.base.preserve_connected
            && candidate.connections >= self.config.base.min_connections
        {
            return false;
        }

        // Check daily limit (must be mutable to check, so we're conservative here)
        // The actual limit enforcement happens in prune()
        true
    }

    /// Execute pruning on a set of candidates
    ///
    /// Returns a report of what was pruned.
    ///
    /// # Fail-Fast
    /// - Respects daily limit strictly
    /// - Preserves candidates meeting protection criteria
    pub fn prune(&mut self, candidates: &[PruningCandidate]) -> PruningReport {
        self.check_day_rollover();

        let mut report = PruningReport::new();
        report.candidates_evaluated = candidates.len();

        for candidate in candidates {
            // Check daily limit
            if self.daily_prune_count >= self.config.max_daily_prunes {
                report.daily_limit_reached = true;
                break;
            }

            // Check preservation rules
            if self.config.base.preserve_connected
                && candidate.connections >= self.config.base.min_connections
            {
                report.record_preserved();
                continue;
            }

            // Prune this candidate
            report.record_prune(candidate);
            self.daily_prune_count += 1;
        }

        report
    }

    /// Get the prune reason for a memory
    pub fn get_prune_reason(&self, metadata: &MemoryMetadata) -> Option<PruneReason> {
        self.get_prune_reason_internal(metadata, &HashMap::new())
    }

    /// Internal prune reason with redundancy context
    fn get_prune_reason_internal(
        &self,
        metadata: &MemoryMetadata,
        redundant_ids: &HashMap<MemoryId, MemoryId>,
    ) -> Option<PruneReason> {
        // Priority order of reasons (first match wins)

        // 1. Check for redundancy
        if redundant_ids.contains_key(&metadata.id) {
            return Some(PruneReason::Redundant);
        }

        // 2. Check for orphaned (no connections)
        if metadata.connection_count == 0 {
            return Some(PruneReason::Orphaned);
        }

        // 3. Check for low alignment
        if metadata.alignment < self.config.base.min_alignment {
            return Some(PruneReason::LowAlignment);
        }

        // 4. Check for staleness
        if let Some(days) = metadata.days_since_access() {
            if days >= self.config.stale_days {
                return Some(PruneReason::Stale);
            }
        }

        // 5. Check for low quality
        if let Some(quality) = metadata.quality_score {
            if quality < self.config.min_quality {
                return Some(PruneReason::LowQuality);
            }
        }

        None
    }

    /// Estimate total bytes that would be freed by pruning candidates
    pub fn estimate_bytes_freed(&self, candidates: &[PruningCandidate]) -> u64 {
        let remaining = self
            .config
            .max_daily_prunes
            .saturating_sub(self.daily_prune_count) as usize;

        candidates
            .iter()
            .take(remaining)
            .filter(|c| self.should_prune(c))
            .map(|c| c.byte_size)
            .sum()
    }

    /// Register a content hash for redundancy detection
    pub fn register_hash(&mut self, hash: u64, memory_id: MemoryId) {
        self.known_hashes.insert(hash, memory_id);
    }

    /// Check if a hash is already known (potential duplicate)
    pub fn is_hash_known(&self, hash: u64) -> bool {
        self.known_hashes.contains_key(&hash)
    }

    /// Get the memory id for a known hash
    pub fn get_hash_owner(&self, hash: u64) -> Option<&MemoryId> {
        self.known_hashes.get(&hash)
    }
}

impl Default for PruningService {
    fn default() -> Self {
        Self::new()
    }
}
