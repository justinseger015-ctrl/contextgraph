//! Type definitions for core handlers.
//!
//! TASK-S005: Prediction types for meta-UTL tracking.
//! TASK-METAUTL-P0-001: Domain and meta-learning event types.

use std::time::Instant;
use uuid::Uuid;

/// Prediction type for tracking
/// TASK-S005: Used to distinguish storage vs retrieval predictions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionType {
    Storage,
    Retrieval,
}

/// Domain enum for domain-specific accuracy tracking.
/// TASK-METAUTL-P0-001: Enables per-domain meta-learning optimization.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Domain {
    /// Source code, programming-related content
    Code,
    /// Medical and healthcare content
    Medical,
    /// Legal documents and regulations
    Legal,
    /// Creative writing, art, design
    Creative,
    /// Research papers, scientific content
    Research,
    /// General purpose, unclassified
    General,
}

impl Default for Domain {
    fn default() -> Self {
        Self::General
    }
}

/// Meta-learning event types for logging and auditing.
/// TASK-METAUTL-P0-001: Used to track significant meta-learning state changes.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaLearningEventType {
    /// Lambda weight adjustment occurred
    LambdaAdjustment,
    /// Bayesian optimization escalation triggered
    BayesianEscalation,
    /// Accuracy dropped below threshold
    AccuracyAlert,
    /// Recovery from low accuracy period
    AccuracyRecovery,
    /// Weight clamping applied (exceeded bounds)
    WeightClamped,
}

/// Meta-learning event for logging significant state changes.
/// TASK-METAUTL-P0-001: Provides audit trail for meta-learning behavior.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct MetaLearningEvent {
    /// Type of event
    pub event_type: MetaLearningEventType,
    /// When the event occurred
    pub timestamp: Instant,
    /// Embedder index affected (if applicable)
    pub embedder_index: Option<usize>,
    /// Previous value (if applicable)
    pub previous_value: Option<f32>,
    /// New value (if applicable)
    pub new_value: Option<f32>,
    /// Optional description
    pub description: Option<String>,
}

#[allow(dead_code)]
impl MetaLearningEvent {
    /// Create a lambda adjustment event.
    pub fn lambda_adjustment(embedder_idx: usize, previous: f32, new: f32) -> Self {
        Self {
            event_type: MetaLearningEventType::LambdaAdjustment,
            timestamp: Instant::now(),
            embedder_index: Some(embedder_idx),
            previous_value: Some(previous),
            new_value: Some(new),
            description: None,
        }
    }

    /// Create a bayesian escalation event.
    pub fn bayesian_escalation(consecutive_low: usize) -> Self {
        Self {
            event_type: MetaLearningEventType::BayesianEscalation,
            timestamp: Instant::now(),
            embedder_index: None,
            previous_value: None,
            new_value: Some(consecutive_low as f32),
            description: Some(format!(
                "Escalation triggered after {} consecutive low accuracy cycles",
                consecutive_low
            )),
        }
    }

    /// Create a weight clamped event.
    pub fn weight_clamped(embedder_idx: usize, original: f32, clamped: f32) -> Self {
        Self {
            event_type: MetaLearningEventType::WeightClamped,
            timestamp: Instant::now(),
            embedder_index: Some(embedder_idx),
            previous_value: Some(original),
            new_value: Some(clamped),
            description: None,
        }
    }
}

/// Configuration for self-correction behavior.
/// TASK-METAUTL-P0-001: Constitution-defined parameters for meta-learning.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct SelfCorrectionConfig {
    /// Whether self-correction is enabled
    pub enabled: bool,
    /// Prediction error threshold (constitution: 0.2)
    pub error_threshold: f32,
    /// Maximum consecutive failures before escalation (constitution: 10)
    pub max_consecutive_failures: usize,
    /// Accuracy threshold below which is considered "low" (constitution: 0.7)
    pub low_accuracy_threshold: f32,
    /// Minimum weight bound (constitution NORTH-016: 0.05)
    /// Note: 13 × 0.05 = 0.65 < 1.0, so sum=1.0 is achievable
    pub min_weight: f32,
    /// Maximum weight bound (constitution: 0.9)
    pub max_weight: f32,
    /// Escalation strategy
    pub escalation_strategy: String,
}

impl Default for SelfCorrectionConfig {
    /// Creates config with constitution-mandated defaults.
    ///
    /// From docs2/constitution.yaml:
    /// - threshold: 0.2
    /// - max_consecutive_failures: 10
    /// - escalation_strategy: "bayesian_optimization"
    /// - NORTH-016_WeightAdjuster: min=0.05, max_delta=0.10
    fn default() -> Self {
        Self {
            enabled: true,
            error_threshold: 0.2,
            max_consecutive_failures: 10,
            low_accuracy_threshold: 0.7,
            min_weight: 0.05, // NORTH-016: min=0.05 (13×0.05=0.65 < 1.0, sum is achievable)
            max_weight: 0.9,
            escalation_strategy: "bayesian_optimization".to_string(),
        }
    }
}

/// Stored prediction for validation
/// TASK-S005: Stores predicted values for later validation against actual outcomes.
#[derive(Clone, Debug)]
pub struct StoredPrediction {
    pub _created_at: Instant,
    pub prediction_type: PredictionType,
    pub predicted_values: serde_json::Value,
    #[allow(dead_code)]
    pub fingerprint_id: Uuid,
}
