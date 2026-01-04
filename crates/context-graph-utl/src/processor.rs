//! UTL Processor - Main orchestrator for UTL computation pipeline.
//!
//! The `UtlProcessor` integrates all UTL components into a unified pipeline:
//! - SurpriseCalculator for delta_s
//! - CoherenceTracker for delta_c
//! - EmotionalWeightCalculator for w_e
//! - PhaseOscillator for phi
//! - LifecycleManager for lambda weights
//! - JohariClassifier for quadrant classification
//!
//! # Constitution Reference
//! - UTL formula: `L = f((ΔS × ΔC) · wₑ · cos φ)` (constitution.yaml:152)
//! - Lifecycle stages: Infancy/Growth/Maturity (constitution.yaml:165-167)

use std::time::Instant;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::UtlConfig;
use crate::error::{UtlError, UtlResult};
use crate::surprise::SurpriseCalculator;
use crate::coherence::CoherenceTracker;
use crate::emotional::EmotionalWeightCalculator;
use crate::phase::PhaseOscillator;
use crate::johari::JohariClassifier;
use crate::lifecycle::LifecycleManager;
use crate::{
    compute_learning_magnitude_validated, LearningSignal, LifecycleLambdaWeights,
    LifecycleStage, JohariQuadrant, SuggestedAction,
};
use context_graph_core::types::EmotionalState;

/// Main UTL computation orchestrator.
///
/// Integrates all 6 UTL components into a unified pipeline:
/// 1. Computes surprise (delta_s) from embedding comparison
/// 2. Computes coherence (delta_c) from semantic consistency
/// 3. Computes emotional weight (w_e) from content sentiment
/// 4. Gets phase angle (phi) from oscillator
/// 5. Applies lifecycle lambda weights
/// 6. Computes learning magnitude and classifies Johari quadrant
///
/// # Performance Budget
/// - Full `compute_learning()`: < 10ms (constitution perf.latency)
///
/// # Example
/// ```
/// use context_graph_utl::processor::UtlProcessor;
/// use context_graph_utl::config::UtlConfig;
///
/// let config = UtlConfig::default();
/// let mut processor = UtlProcessor::new(config);
///
/// let content = "New information about machine learning";
/// let embedding = vec![0.1; 1536];
/// let context = vec![vec![0.15; 1536], vec![0.12; 1536]];
///
/// let signal = processor.compute_learning(content, &embedding, &context)
///     .expect("Valid computation");
///
/// assert!(signal.magnitude >= 0.0 && signal.magnitude <= 1.0);
/// ```
#[derive(Debug)]
pub struct UtlProcessor {
    /// Surprise calculator (delta_s)
    surprise_calculator: SurpriseCalculator,

    /// Coherence tracker (delta_c)
    coherence_tracker: CoherenceTracker,

    /// Emotional weight calculator (w_e)
    emotional_calculator: EmotionalWeightCalculator,

    /// Phase oscillator (phi)
    phase_oscillator: PhaseOscillator,

    /// Johari quadrant classifier
    johari_classifier: JohariClassifier,

    /// Lifecycle state manager
    lifecycle_manager: LifecycleManager,

    /// Configuration
    config: UtlConfig,

    /// Total computation count for metrics
    computation_count: u64,
}

impl UtlProcessor {
    /// Create a new UtlProcessor with the given configuration.
    ///
    /// # Arguments
    /// * `config` - UTL configuration (validated on construction)
    ///
    /// # Panics
    /// Panics if config validation fails. Use `try_new()` for fallible construction.
    pub fn new(config: UtlConfig) -> Self {
        config.validate().expect("UtlConfig validation failed");

        Self {
            surprise_calculator: SurpriseCalculator::new(&config.surprise),
            coherence_tracker: CoherenceTracker::new(&config.coherence),
            emotional_calculator: EmotionalWeightCalculator::new(&config.emotional),
            phase_oscillator: PhaseOscillator::new(&config.phase),
            johari_classifier: JohariClassifier::new(&config.johari),
            lifecycle_manager: LifecycleManager::new(&config.lifecycle),
            config,
            computation_count: 0,
        }
    }

    /// Try to create a new UtlProcessor, returning an error if config is invalid.
    pub fn try_new(config: UtlConfig) -> UtlResult<Self> {
        config.validate().map_err(UtlError::ConfigError)?;

        Ok(Self {
            surprise_calculator: SurpriseCalculator::new(&config.surprise),
            coherence_tracker: CoherenceTracker::new(&config.coherence),
            emotional_calculator: EmotionalWeightCalculator::new(&config.emotional),
            phase_oscillator: PhaseOscillator::new(&config.phase),
            johari_classifier: JohariClassifier::new(&config.johari),
            lifecycle_manager: LifecycleManager::new(&config.lifecycle),
            config,
            computation_count: 0,
        })
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(UtlConfig::default())
    }

    /// Compute full UTL learning signal.
    ///
    /// This is the main entry point for UTL computation. It:
    /// 1. Records interaction for lifecycle tracking
    /// 2. Computes all 4 UTL components
    /// 3. Applies Marblestone lambda weights
    /// 4. Classifies Johari quadrant
    /// 5. Determines storage/consolidation decisions
    ///
    /// # Arguments
    /// * `content` - Text content for emotional analysis
    /// * `embedding` - Vector embedding of the content (typically 1536D)
    /// * `context_embeddings` - Recent context embeddings for comparison
    ///
    /// # Returns
    /// `Ok(LearningSignal)` with all computed values, or `Err(UtlError)` on failure.
    ///
    /// # Errors
    /// - `UtlError::InvalidComputation` if result is NaN/Infinity
    /// - `UtlError::InvalidParameter` if inputs are out of range
    ///
    /// # Performance
    /// Target: < 10ms (per constitution perf.latency.inject_context)
    pub fn compute_learning(
        &mut self,
        content: &str,
        embedding: &[f32],
        context_embeddings: &[Vec<f32>],
    ) -> UtlResult<LearningSignal> {
        let start = Instant::now();

        // 1. Record interaction for lifecycle tracking (may trigger stage transition)
        let _transitioned = self.lifecycle_manager.increment();

        // 2. Compute surprise (delta_s) [0, 1]
        let delta_s = self.surprise_calculator.compute_surprise(embedding, context_embeddings);

        // 3. Compute coherence (delta_c) [0, 1]
        let delta_c = self.coherence_tracker.compute_coherence(embedding, context_embeddings);

        // 4. Compute emotional weight (w_e) [0.5, 1.5]
        // Use Neutral state as default - can be overridden via compute_learning_with_state
        let w_e = self.emotional_calculator.compute_emotional_weight(content, EmotionalState::Neutral);

        // 5. Get phase angle (phi) [0, PI]
        let phi = self.phase_oscillator.phase();

        // 6. Get lifecycle lambda weights
        let lambda_weights = self.lifecycle_manager.current_weights();

        // 7. Compute weighted learning magnitude
        // Apply Marblestone lambda weights: L_weighted = lambda_s * delta_s + lambda_c * delta_c
        let weighted_delta_s = delta_s * lambda_weights.lambda_s();
        let weighted_delta_c = delta_c * lambda_weights.lambda_c();

        // Compute magnitude with validated inputs
        let magnitude = compute_learning_magnitude_validated(
            weighted_delta_s.clamp(0.0, 1.0),
            weighted_delta_c.clamp(0.0, 1.0),
            w_e.clamp(0.5, 1.5),
            phi.clamp(0.0, std::f32::consts::PI),
        )?;

        // 8. Classify Johari quadrant using raw (unweighted) values
        let quadrant = self.johari_classifier.classify(delta_s, delta_c);
        let suggested_action = suggested_action_for_quadrant(quadrant);

        // 9. Determine consolidation and storage decisions
        // Using high_quality threshold for consolidation, low_quality for storage
        let should_consolidate = magnitude > self.config.thresholds.high_quality;
        let should_store = magnitude > self.config.thresholds.low_quality;

        // Calculate latency
        let latency_us = start.elapsed().as_micros() as u64;

        // Update computation count
        self.computation_count += 1;

        // Create and validate learning signal
        LearningSignal::new(
            magnitude,
            delta_s,
            delta_c,
            w_e,
            phi,
            Some(lambda_weights),
            quadrant,
            suggested_action,
            should_consolidate,
            should_store,
            latency_us,
        )
    }

    /// Compute learning with explicit emotional state.
    pub fn compute_learning_with_state(
        &mut self,
        content: &str,
        embedding: &[f32],
        context_embeddings: &[Vec<f32>],
        emotional_state: EmotionalState,
    ) -> UtlResult<LearningSignal> {
        let start = Instant::now();

        let _transitioned = self.lifecycle_manager.increment();
        let delta_s = self.surprise_calculator.compute_surprise(embedding, context_embeddings);
        let delta_c = self.coherence_tracker.compute_coherence(embedding, context_embeddings);
        let w_e = self.emotional_calculator.compute_emotional_weight(content, emotional_state);
        let phi = self.phase_oscillator.phase();
        let lambda_weights = self.lifecycle_manager.current_weights();

        let weighted_delta_s = delta_s * lambda_weights.lambda_s();
        let weighted_delta_c = delta_c * lambda_weights.lambda_c();

        let magnitude = compute_learning_magnitude_validated(
            weighted_delta_s.clamp(0.0, 1.0),
            weighted_delta_c.clamp(0.0, 1.0),
            w_e.clamp(0.5, 1.5),
            phi.clamp(0.0, std::f32::consts::PI),
        )?;

        let quadrant = self.johari_classifier.classify(delta_s, delta_c);
        let suggested_action = suggested_action_for_quadrant(quadrant);
        // Using high_quality threshold for consolidation, low_quality for storage
        let should_consolidate = magnitude > self.config.thresholds.high_quality;
        let should_store = magnitude > self.config.thresholds.low_quality;
        let latency_us = start.elapsed().as_micros() as u64;

        self.computation_count += 1;

        LearningSignal::new(
            magnitude, delta_s, delta_c, w_e, phi,
            Some(lambda_weights), quadrant, suggested_action,
            should_consolidate, should_store, latency_us,
        )
    }

    /// Get current lifecycle stage.
    #[inline]
    pub fn lifecycle_stage(&self) -> LifecycleStage {
        self.lifecycle_manager.current_stage()
    }

    /// Get current lifecycle lambda weights.
    #[inline]
    pub fn lambda_weights(&self) -> LifecycleLambdaWeights {
        self.lifecycle_manager.current_weights()
    }

    /// Get total interaction count.
    #[inline]
    pub fn interaction_count(&self) -> u64 {
        self.lifecycle_manager.interaction_count()
    }

    /// Get computation count.
    #[inline]
    pub fn computation_count(&self) -> u64 {
        self.computation_count
    }

    /// Get current phase angle.
    #[inline]
    pub fn current_phase(&self) -> f32 {
        self.phase_oscillator.phase()
    }

    /// Update phase oscillator with elapsed time.
    pub fn update_phase(&mut self, elapsed: std::time::Duration) {
        self.phase_oscillator.update(elapsed);
    }

    /// Reset the processor to initial state.
    pub fn reset(&mut self) {
        self.lifecycle_manager.reset();
        self.coherence_tracker.clear();
        self.phase_oscillator.reset();
        self.computation_count = 0;
    }

    /// Restore lifecycle state from persistence.
    pub fn restore_lifecycle(&mut self, interaction_count: u64) {
        // Use increment_by to restore count and trigger stage transitions
        self.lifecycle_manager.increment_by(interaction_count);
    }
}

/// Map Johari quadrant to suggested action.
///
/// Per constitution.yaml:159-163 and contextprd.md Section 2.2:
/// - Open (low ΔS, high ΔC): Direct recall
/// - Blind (high ΔS, low ΔC): Discovery via epistemic_action or dream
/// - Hidden (low ΔS, low ΔC): Private - use get_neighborhood
/// - Unknown (high ΔS, high ΔC): Frontier exploration
fn suggested_action_for_quadrant(quadrant: JohariQuadrant) -> SuggestedAction {
    match quadrant {
        JohariQuadrant::Open => SuggestedAction::DirectRecall,
        JohariQuadrant::Blind => SuggestedAction::TriggerDream,
        JohariQuadrant::Hidden => SuggestedAction::GetNeighborhood,
        JohariQuadrant::Unknown => SuggestedAction::EpistemicAction,
    }
}

// ============================================================================
// SessionContext
// ============================================================================

/// Context for a single UTL computation session.
///
/// Maintains state for multiple related computations within a session,
/// including recent embeddings for context comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContext {
    /// Unique session identifier.
    pub session_id: Uuid,

    /// Recent embeddings for context (sliding window).
    pub recent_embeddings: Vec<Vec<f32>>,

    /// Maximum context window size.
    pub max_window_size: usize,

    /// Number of interactions in this session.
    pub interaction_count: u64,

    /// Session start time.
    pub started_at: DateTime<Utc>,

    /// Last activity timestamp.
    pub last_activity: DateTime<Utc>,
}

impl SessionContext {
    /// Create a new session context.
    pub fn new(session_id: Uuid, max_window_size: usize) -> Self {
        let now = Utc::now();
        Self {
            session_id,
            recent_embeddings: Vec::with_capacity(max_window_size),
            max_window_size,
            interaction_count: 0,
            started_at: now,
            last_activity: now,
        }
    }

    /// Create a new session with auto-generated ID.
    pub fn new_with_generated_id(max_window_size: usize) -> Self {
        Self::new(Uuid::new_v4(), max_window_size)
    }

    /// Default session with 50-embedding window.
    pub fn default_session() -> Self {
        Self::new_with_generated_id(50)
    }

    /// Add an embedding to the context window.
    ///
    /// Maintains a sliding window - oldest embeddings are removed when
    /// the window exceeds `max_window_size`.
    pub fn add_embedding(&mut self, embedding: Vec<f32>) {
        if self.recent_embeddings.len() >= self.max_window_size {
            self.recent_embeddings.remove(0);
        }
        self.recent_embeddings.push(embedding);
        self.interaction_count += 1;
        self.last_activity = Utc::now();
    }

    /// Get context embeddings as slice.
    #[inline]
    pub fn context_embeddings(&self) -> &[Vec<f32>] {
        &self.recent_embeddings
    }

    /// Check if session has sufficient context for meaningful comparison.
    pub fn has_sufficient_context(&self) -> bool {
        self.recent_embeddings.len() >= 2
    }

    /// Check if session is stale.
    pub fn is_stale(&self, max_age_seconds: i64) -> bool {
        let age = Utc::now() - self.last_activity;
        age.num_seconds() > max_age_seconds
    }

    /// Get session age in seconds.
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.started_at).num_seconds()
    }

    /// Clear context but preserve session metadata.
    pub fn clear_context(&mut self) {
        self.recent_embeddings.clear();
        self.interaction_count = 0;
        self.last_activity = Utc::now();
    }
}

impl Default for SessionContext {
    fn default() -> Self {
        Self::default_session()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_embedding(dim: usize, value: f32) -> Vec<f32> {
        vec![value; dim]
    }

    #[test]
    fn test_processor_creation() {
        let processor = UtlProcessor::with_defaults();
        assert_eq!(processor.lifecycle_stage(), LifecycleStage::Infancy);
        assert_eq!(processor.interaction_count(), 0);
        assert_eq!(processor.computation_count(), 0);
    }

    #[test]
    fn test_processor_try_new_with_valid_config() {
        let config = UtlConfig::default();
        let result = UtlProcessor::try_new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compute_learning_basic() {
        let mut processor = UtlProcessor::with_defaults();

        let content = "This is a test message for UTL computation.";
        let embedding = test_embedding(128, 0.1);
        let context = vec![
            test_embedding(128, 0.15),
            test_embedding(128, 0.12),
        ];

        let signal = processor.compute_learning(content, &embedding, &context);
        assert!(signal.is_ok());

        let signal = signal.unwrap();
        assert!(signal.magnitude >= 0.0 && signal.magnitude <= 1.0);
        assert!(signal.delta_s >= 0.0 && signal.delta_s <= 1.0);
        assert!(signal.delta_c >= 0.0 && signal.delta_c <= 1.0);
        assert!(signal.w_e >= 0.5 && signal.w_e <= 1.5);
        assert!(signal.phi >= 0.0 && signal.phi <= std::f32::consts::PI);
        assert!(signal.lambda_weights.is_some());
    }

    #[test]
    fn test_compute_learning_empty_context() {
        let mut processor = UtlProcessor::with_defaults();

        let content = "Test with no context.";
        let embedding = test_embedding(128, 0.1);
        let context: Vec<Vec<f32>> = vec![];

        let signal = processor.compute_learning(content, &embedding, &context);
        assert!(signal.is_ok());

        // With empty context, surprise should be baseline
        let signal = signal.unwrap();
        assert!(signal.magnitude >= 0.0);
    }

    #[test]
    fn test_lifecycle_progression() {
        let mut processor = UtlProcessor::with_defaults();
        assert_eq!(processor.lifecycle_stage(), LifecycleStage::Infancy);

        // Simulate 50 interactions to trigger Infancy -> Growth
        for i in 0..50 {
            let content = format!("Message {}", i);
            let embedding = test_embedding(128, 0.1 + (i as f32 * 0.01));
            let _ = processor.compute_learning(&content, &embedding, &[]);
        }

        assert_eq!(processor.lifecycle_stage(), LifecycleStage::Growth);
        assert_eq!(processor.interaction_count(), 50);
    }

    #[test]
    fn test_lambda_weights_change_with_stage() {
        let mut processor = UtlProcessor::with_defaults();

        // Infancy weights: lambda_s = 0.7, lambda_c = 0.3
        let infancy_weights = processor.lambda_weights();
        assert!((infancy_weights.lambda_s() - 0.7).abs() < 0.01);
        assert!((infancy_weights.lambda_c() - 0.3).abs() < 0.01);

        // Progress to Growth
        for i in 0..50 {
            let _ = processor.compute_learning(
                &format!("msg {}", i),
                &test_embedding(128, 0.1),
                &[],
            );
        }

        // Growth weights: lambda_s = 0.5, lambda_c = 0.5
        let growth_weights = processor.lambda_weights();
        assert!((growth_weights.lambda_s() - 0.5).abs() < 0.01);
        assert!((growth_weights.lambda_c() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_johari_quadrant_classification() {
        let _processor = UtlProcessor::with_defaults();

        // Open: low surprise, high coherence
        let open_action = suggested_action_for_quadrant(JohariQuadrant::Open);
        assert_eq!(open_action, SuggestedAction::DirectRecall);

        // Blind: high surprise, low coherence
        let blind_action = suggested_action_for_quadrant(JohariQuadrant::Blind);
        assert_eq!(blind_action, SuggestedAction::TriggerDream);

        // Hidden: low surprise, low coherence
        let hidden_action = suggested_action_for_quadrant(JohariQuadrant::Hidden);
        assert_eq!(hidden_action, SuggestedAction::GetNeighborhood);

        // Unknown: high surprise, high coherence
        let unknown_action = suggested_action_for_quadrant(JohariQuadrant::Unknown);
        assert_eq!(unknown_action, SuggestedAction::EpistemicAction);
    }

    #[test]
    fn test_computation_count_tracking() {
        let mut processor = UtlProcessor::with_defaults();
        assert_eq!(processor.computation_count(), 0);

        for _ in 0..5 {
            let _ = processor.compute_learning("test", &test_embedding(128, 0.1), &[]);
        }

        assert_eq!(processor.computation_count(), 5);
    }

    #[test]
    fn test_processor_reset() {
        let mut processor = UtlProcessor::with_defaults();

        // Do some computations
        for _ in 0..10 {
            let _ = processor.compute_learning("test", &test_embedding(128, 0.1), &[]);
        }

        assert!(processor.computation_count() > 0);
        assert!(processor.interaction_count() > 0);

        // Reset
        processor.reset();

        assert_eq!(processor.computation_count(), 0);
        assert_eq!(processor.interaction_count(), 0);
        assert_eq!(processor.lifecycle_stage(), LifecycleStage::Infancy);
    }

    #[test]
    fn test_restore_lifecycle() {
        let mut processor = UtlProcessor::with_defaults();
        assert_eq!(processor.lifecycle_stage(), LifecycleStage::Infancy);

        // Restore to 100 interactions (Growth stage)
        processor.restore_lifecycle(100);

        assert_eq!(processor.lifecycle_stage(), LifecycleStage::Growth);
        assert_eq!(processor.interaction_count(), 100);
    }

    #[test]
    fn test_performance_under_10ms() {
        let mut processor = UtlProcessor::with_defaults();

        let content = "Performance test content for UTL processing.";
        let embedding = test_embedding(1536, 0.5);
        let context: Vec<Vec<f32>> = (0..50)
            .map(|i| test_embedding(1536, 0.1 * (i as f32 % 10.0)))
            .collect();

        let start = Instant::now();
        let signal = processor.compute_learning(content, &embedding, &context);
        let elapsed = start.elapsed();

        assert!(signal.is_ok());
        assert!(
            elapsed.as_millis() < 10,
            "Computation took {}ms, expected < 10ms",
            elapsed.as_millis()
        );
    }

    // ========================================================================
    // SessionContext Tests
    // ========================================================================

    #[test]
    fn test_session_context_creation() {
        let session = SessionContext::default_session();

        assert!(!session.session_id.is_nil());
        assert_eq!(session.max_window_size, 50);
        assert_eq!(session.interaction_count, 0);
        assert!(session.recent_embeddings.is_empty());
    }

    #[test]
    fn test_session_add_embedding() {
        let mut session = SessionContext::new_with_generated_id(5);

        session.add_embedding(vec![1.0; 128]);
        session.add_embedding(vec![2.0; 128]);
        session.add_embedding(vec![3.0; 128]);

        assert_eq!(session.interaction_count, 3);
        assert_eq!(session.recent_embeddings.len(), 3);
        assert!(session.has_sufficient_context());
    }

    #[test]
    fn test_session_window_sliding() {
        let mut session = SessionContext::new_with_generated_id(3);

        // Add 5 embeddings to window of size 3
        for i in 1..=5 {
            session.add_embedding(vec![i as f32; 128]);
        }

        // Window should only contain last 3
        assert_eq!(session.recent_embeddings.len(), 3);
        assert_eq!(session.recent_embeddings[0][0], 3.0);
        assert_eq!(session.recent_embeddings[1][0], 4.0);
        assert_eq!(session.recent_embeddings[2][0], 5.0);
        assert_eq!(session.interaction_count, 5);
    }

    #[test]
    fn test_session_staleness() {
        let session = SessionContext::default_session();

        // Fresh session should not be stale
        assert!(!session.is_stale(60));
    }

    #[test]
    fn test_session_clear_context() {
        let mut session = SessionContext::default_session();

        session.add_embedding(vec![1.0; 128]);
        session.add_embedding(vec![2.0; 128]);
        assert_eq!(session.interaction_count, 2);

        session.clear_context();

        assert_eq!(session.interaction_count, 0);
        assert!(session.recent_embeddings.is_empty());
        // Session ID should be preserved
        assert!(!session.session_id.is_nil());
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_single_dimension_embedding() {
        let mut processor = UtlProcessor::with_defaults();

        let content = "Minimal test";
        let embedding = vec![0.5];
        let context = vec![vec![0.3], vec![0.7]];

        let signal = processor.compute_learning(content, &embedding, &context);
        assert!(signal.is_ok());
    }

    #[test]
    fn test_large_context_window() {
        let mut processor = UtlProcessor::with_defaults();

        let content = "Large context test";
        let embedding = test_embedding(128, 0.5);
        let context: Vec<Vec<f32>> = (0..1000)
            .map(|i| test_embedding(128, (i as f32 % 100.0) / 100.0))
            .collect();

        let signal = processor.compute_learning(content, &embedding, &context);
        assert!(signal.is_ok());
    }

    #[test]
    fn test_empty_content() {
        let mut processor = UtlProcessor::with_defaults();

        let content = "";
        let embedding = test_embedding(128, 0.1);
        let context = vec![test_embedding(128, 0.2)];

        let signal = processor.compute_learning(content, &embedding, &context);
        assert!(signal.is_ok());

        // Empty content should give neutral emotional weight
        let signal = signal.unwrap();
        assert!((signal.w_e - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_identical_embeddings() {
        let mut processor = UtlProcessor::with_defaults();

        let content = "Same embedding test";
        let embedding = test_embedding(128, 0.5);
        let context = vec![
            test_embedding(128, 0.5),
            test_embedding(128, 0.5),
            test_embedding(128, 0.5),
        ];

        let signal = processor.compute_learning(content, &embedding, &context);
        assert!(signal.is_ok());

        // Identical embeddings = low surprise, high coherence = Open quadrant
        let signal = signal.unwrap();
        // Surprise should be low (similar to context)
        assert!(signal.delta_s < 0.5);
    }
}
