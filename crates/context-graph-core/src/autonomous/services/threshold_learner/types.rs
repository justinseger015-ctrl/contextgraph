//! Types and constants for the ThresholdLearner service.
//!
//! Contains the core data structures used by the 4-level ATC system:
//! - ThompsonState: Thompson sampling state for exploration
//! - EmbedderLearningState: Per-embedder learning state
//! - BayesianObservation: Bayesian optimization observation

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::autonomous::AdaptiveThresholdConfig;

/// Type alias for backwards compatibility
pub type ThresholdLearnerConfig = AdaptiveThresholdConfig;

/// Number of embedders in the system (E1-E13)
pub const NUM_EMBEDDERS: usize = 13;

/// Default EWMA smoothing factor
pub(crate) const DEFAULT_ALPHA: f32 = 0.2;

/// Minimum observations before recalibration is considered
pub(crate) const MIN_OBSERVATIONS_FOR_RECALIBRATION: u32 = 10;

/// Recalibration check interval
pub(crate) const RECALIBRATION_CHECK_INTERVAL_SECS: i64 = 3600; // 1 hour

/// Thompson sampling state for a single embedder threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThompsonState {
    /// Beta distribution alpha parameter (successes + 1)
    pub alpha: f32,
    /// Beta distribution beta parameter (failures + 1)
    pub beta: f32,
    /// Total samples taken
    pub samples: u32,
}

impl Default for ThompsonState {
    fn default() -> Self {
        Self {
            alpha: 1.0, // Uniform prior
            beta: 1.0,
            samples: 0,
        }
    }
}

/// Per-embedder learning state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderLearningState {
    /// Current EWMA threshold value
    pub ewma_threshold: f32,
    /// Temperature scaling factor for calibration
    pub temperature: f32,
    /// Thompson sampling state
    pub thompson: ThompsonState,
    /// Observation count for this embedder
    pub observation_count: u32,
    /// Cumulative success count
    pub success_count: u32,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

impl Default for EmbedderLearningState {
    fn default() -> Self {
        Self {
            ewma_threshold: 0.75,
            temperature: 1.0,
            thompson: ThompsonState::default(),
            observation_count: 0,
            success_count: 0,
            last_updated: Utc::now(),
        }
    }
}

/// Bayesian optimization observation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianObservation {
    /// Threshold configuration tried
    pub thresholds: [f32; NUM_EMBEDDERS],
    /// Performance score achieved
    pub performance: f32,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}
