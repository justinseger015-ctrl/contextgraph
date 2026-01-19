//! Pipeline types, configuration, and result structures.
//!
//! This module contains all type definitions for the 4-stage retrieval pipeline.

use uuid::Uuid;

use super::super::super::indexes::EmbedderIndex;
use super::super::error::SearchError;

// ============================================================================
// PIPELINE ERRORS
// ============================================================================

/// Pipeline-specific errors. FAIL FAST - no recovery.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// Stage execution error.
    #[error("FAIL FAST: Stage {stage:?} error: {error}")]
    Stage { stage: PipelineStage, error: String },

    /// Stage exceeded maximum latency.
    #[error("FAIL FAST: Stage {stage:?} timeout after {elapsed_ms}ms (max: {max_ms}ms)")]
    Timeout {
        stage: PipelineStage,
        elapsed_ms: u64,
        max_ms: u64,
    },

    /// Required query missing for stage.
    #[error("FAIL FAST: Missing query for stage {stage:?}")]
    MissingQuery { stage: PipelineStage },

    /// Empty candidates at stage (when not expected).
    #[error("FAIL FAST: Empty candidates at stage {stage:?}")]
    EmptyCandidates { stage: PipelineStage },

    /// Wrapped search error.
    #[error("FAIL FAST: Search error: {0}")]
    Search(#[from] SearchError),
}

// ============================================================================
// STAGE CONFIGURATION
// ============================================================================

/// Configuration for a single pipeline stage.
#[derive(Debug, Clone)]
pub struct StageConfig {
    /// Whether this stage is enabled.
    pub enabled: bool,
    /// Candidate multiplier: target = k * multiplier.
    pub candidate_multiplier: f32,
    /// Minimum score threshold to pass this stage.
    pub min_score_threshold: f32,
    /// Maximum allowed latency in milliseconds.
    pub max_latency_ms: u64,
}

impl Default for StageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            candidate_multiplier: 3.0,
            min_score_threshold: 0.3,
            max_latency_ms: 20,
        }
    }
}

// ============================================================================
// PIPELINE STAGE ENUM
// ============================================================================

/// The 4 pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    /// Stage 1: SPLADE sparse pre-filter (inverted index).
    SpladeFilter,
    /// Stage 2: Matryoshka 128D fast ANN.
    MatryoshkaAnn,
    /// Stage 3: Multi-space RRF rerank.
    RrfRerank,
    /// Stage 4: Late interaction MaxSim.
    MaxSimRerank,
}

impl PipelineStage {
    /// Get the stage index (0-3).
    #[inline]
    pub fn index(&self) -> usize {
        match self {
            Self::SpladeFilter => 0,
            Self::MatryoshkaAnn => 1,
            Self::RrfRerank => 2,
            Self::MaxSimRerank => 3,
        }
    }

    /// Get all stages in order.
    pub fn all() -> [Self; 4] {
        [
            Self::SpladeFilter,
            Self::MatryoshkaAnn,
            Self::RrfRerank,
            Self::MaxSimRerank,
        ]
    }
}

// ============================================================================
// PIPELINE CANDIDATE
// ============================================================================

/// A candidate moving through the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineCandidate {
    /// Memory ID.
    pub id: Uuid,
    /// Current aggregated score.
    pub score: f32,
    /// Stage-by-stage scores for debugging.
    pub stage_scores: Vec<(PipelineStage, f32)>,
}

impl PipelineCandidate {
    /// Create a new candidate with initial score.
    #[inline]
    pub fn new(id: Uuid, score: f32) -> Self {
        Self {
            id,
            score,
            stage_scores: Vec::with_capacity(4),
        }
    }

    /// Add a stage score.
    #[inline]
    pub fn add_stage_score(&mut self, stage: PipelineStage, score: f32) {
        self.stage_scores.push((stage, score));
        self.score = score;
    }
}

// ============================================================================
// STAGE RESULT
// ============================================================================

/// Result from a single pipeline stage.
#[derive(Debug)]
pub struct StageResult {
    /// Candidates that passed this stage.
    pub candidates: Vec<PipelineCandidate>,
    /// Stage execution latency in microseconds.
    pub latency_us: u64,
    /// Number of candidates entering this stage.
    pub candidates_in: usize,
    /// Number of candidates exiting this stage.
    pub candidates_out: usize,
    /// Stage that produced this result.
    pub stage: PipelineStage,
}

// ============================================================================
// PIPELINE CONFIGURATION
// ============================================================================

/// Configuration for the full pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Per-stage configurations (indexed by PipelineStage as usize).
    pub stages: [StageConfig; 4],
    /// Final result limit.
    pub k: usize,
    /// RRF constant (default 60.0).
    pub rrf_k: f32,
    /// RRF embedders to use in Stage 3.
    pub rrf_embedders: Vec<EmbedderIndex>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            stages: [
                StageConfig {
                    candidate_multiplier: 10.0,
                    max_latency_ms: 5,
                    ..Default::default()
                }, // Stage 1: wide
                StageConfig {
                    candidate_multiplier: 5.0,
                    max_latency_ms: 10,
                    ..Default::default()
                }, // Stage 2: narrower
                StageConfig {
                    candidate_multiplier: 3.0,
                    max_latency_ms: 20,
                    ..Default::default()
                }, // Stage 3: RRF
                StageConfig {
                    candidate_multiplier: 1.0,
                    max_latency_ms: 15,
                    ..Default::default()
                }, // Stage 4: final
            ],
            k: 10,
            rrf_k: 60.0,
            rrf_embedders: vec![
                EmbedderIndex::E1Semantic,
                EmbedderIndex::E8Graph,
                EmbedderIndex::E5Causal,
            ],
        }
    }
}

// ============================================================================
// PIPELINE RESULT
// ============================================================================

/// Final pipeline result.
#[derive(Debug)]
pub struct PipelineResult {
    /// Final ranked results.
    pub results: Vec<PipelineCandidate>,
    /// Per-stage results for debugging and analysis.
    pub stage_results: Vec<StageResult>,
    /// Total pipeline latency in microseconds.
    pub total_latency_us: u64,
    /// Stages that were executed.
    pub stages_executed: Vec<PipelineStage>,
}

impl PipelineResult {
    /// Check if results are empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get number of results.
    #[inline]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Get top result.
    #[inline]
    pub fn top(&self) -> Option<&PipelineCandidate> {
        self.results.first()
    }

    /// Get total latency in milliseconds.
    #[inline]
    pub fn latency_ms(&self) -> f64 {
        self.total_latency_us as f64 / 1000.0
    }
}
