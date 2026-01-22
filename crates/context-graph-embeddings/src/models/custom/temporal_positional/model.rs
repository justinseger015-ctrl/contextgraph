//! Core Temporal-Positional embedding model implementation (E4).
//!
//! E4 encodes session sequence positions to enable "before/after" queries within a session.
//! This is distinct from E2 (V_freshness) which encodes Unix timestamps.

use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::traits::EmbeddingModel;
use crate::types::{InputType, ModelEmbedding, ModelId, ModelInput};

use super::constants::{DEFAULT_BASE, MAX_BASE, MIN_BASE, TEMPORAL_POSITIONAL_DIMENSION};
use super::encoding::{compute_positional_encoding, compute_positional_encoding_from_position};
use super::timestamp::{extract_position, extract_timestamp, parse_position, parse_timestamp};

/// Temporal-Positional embedding model (E4 - V_ordering).
///
/// Encodes **session sequence positions** to enable "before/after" queries within a session.
/// This is distinct from E2 (V_freshness) which encodes Unix timestamps for recency.
///
/// # Position Types (Priority Order)
///
/// 1. **Sequence number** (preferred): `sequence:N` - Session-local ordering (0, 1, 2, ...)
/// 2. **ISO timestamp**: `timestamp:2024-01-15T10:30:00Z`
/// 3. **Unix epoch**: `epoch:1705315800`
/// 4. **Current time**: Falls back to `Utc::now()` if no instruction provided
///
/// # Algorithm
///
/// Uses transformer-style sinusoidal positional encoding:
///   - PE(pos, 2i) = sin(pos / base^(2i/d_model))
///   - PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
///
/// For sequence mode (small position values), uses a smaller effective base
/// to ensure consecutive positions have distinct encodings.
///
/// # Purpose
///
/// E4 enables queries like:
/// - "What happened before X in this session?"
/// - "What came after Y?"
/// - "Order these memories by when they occurred in the session"
///
/// # Construction
///
/// ```rust,no_run
/// use context_graph_embeddings::models::TemporalPositionalModel;
/// use context_graph_embeddings::error::EmbeddingResult;
/// use context_graph_embeddings::EmbeddingModel; // For is_initialized() trait method
///
/// fn example() -> EmbeddingResult<()> {
///     // Default base (10000.0)
///     let model = TemporalPositionalModel::new();
///     assert!(model.is_initialized());
///     assert_eq!(model.base(), 10000.0);
///
///     // Custom base frequency
///     let model = TemporalPositionalModel::with_base(5000.0)?;
///     assert_eq!(model.base(), 5000.0);
///     Ok(())
/// }
/// ```
pub struct TemporalPositionalModel {
    /// Base frequency for positional encoding (default 10000.0).
    base: f32,

    /// d_model dimension (always 512).
    d_model: usize,

    /// Always true for custom models (no weights to load).
    initialized: AtomicBool,
}

impl TemporalPositionalModel {
    /// Create a new TemporalPositionalModel with default base frequency (10000.0).
    ///
    /// Uses the standard transformer positional encoding base.
    /// Model is immediately ready for use (no loading required).
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: DEFAULT_BASE,
            d_model: TEMPORAL_POSITIONAL_DIMENSION,
            initialized: AtomicBool::new(true),
        }
    }

    /// Create a model with custom base frequency.
    ///
    /// # Arguments
    /// * `base` - Base frequency for positional encoding. Must be > 1.0.
    ///   Larger values create slower-varying frequencies.
    ///
    /// # Errors
    /// Returns `EmbeddingError::ConfigError` if base is not in valid range (1.0, 1e10).
    pub fn with_base(base: f32) -> EmbeddingResult<Self> {
        if base <= MIN_BASE || !base.is_finite() || base > MAX_BASE {
            return Err(EmbeddingError::ConfigError {
                message: format!(
                    "TemporalPositionalModel base must be in range ({}, {}], got {}",
                    MIN_BASE, MAX_BASE, base
                ),
            });
        }

        Ok(Self {
            base,
            d_model: TEMPORAL_POSITIONAL_DIMENSION,
            initialized: AtomicBool::new(true),
        })
    }

    /// Get the base frequency used by this model.
    #[must_use]
    pub fn base(&self) -> f32 {
        self.base
    }

    /// Compute the positional encoding for a given position.
    ///
    /// # Arguments
    /// * `position` - The position value (sequence number or Unix timestamp)
    /// * `is_sequence` - True if position is a session sequence number
    fn compute_positional_encoding_from_pos(
        &self,
        position: i64,
        is_sequence: bool,
    ) -> Vec<f32> {
        compute_positional_encoding_from_position(position, self.base, self.d_model, is_sequence)
    }

    /// Compute the transformer-style positional encoding for a given timestamp.
    ///
    /// Legacy method for backward compatibility.
    fn compute_positional_encoding(&self, timestamp: DateTime<Utc>) -> Vec<f32> {
        compute_positional_encoding(timestamp, self.base, self.d_model)
    }

    /// Extract timestamp from ModelInput.
    ///
    /// Legacy method for backward compatibility.
    fn extract_timestamp(&self, input: &ModelInput) -> DateTime<Utc> {
        extract_timestamp(input)
    }

    /// Parse timestamp from instruction string (legacy API).
    ///
    /// Supports formats:
    /// - ISO 8601: "timestamp:2024-01-15T10:30:00Z"
    /// - Unix epoch: "epoch:1705315800"
    ///
    /// For new code, prefer `parse_position()` which also supports sequence numbers.
    pub fn parse_timestamp(instruction: &str) -> Option<DateTime<Utc>> {
        parse_timestamp(instruction)
    }

    /// Parse position from instruction string.
    ///
    /// Supports formats (priority order):
    /// - Sequence: "sequence:123" (preferred for E4)
    /// - ISO 8601: "timestamp:2024-01-15T10:30:00Z"
    /// - Unix epoch: "epoch:1705315800"
    pub fn parse_position(instruction: &str) -> Option<super::timestamp::PositionInfo> {
        parse_position(instruction)
    }
}

impl Default for TemporalPositionalModel {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EmbeddingModel for TemporalPositionalModel {
    fn model_id(&self) -> ModelId {
        ModelId::TemporalPositional
    }

    fn supported_input_types(&self) -> &[InputType] {
        // TemporalPositional supports Text input (timestamp via instruction field)
        &[InputType::Text]
    }

    fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
        // 1. Validate input type
        self.validate_input(input)?;

        let start = std::time::Instant::now();

        // 2. Extract position from input (prefers sequence: over timestamp:/epoch:)
        let position_info = extract_position(input);

        // 3. Compute positional encoding using position value and mode
        let vector = self.compute_positional_encoding_from_pos(
            position_info.position,
            position_info.is_sequence,
        );

        let latency_us = start.elapsed().as_micros() as u64;

        // 4. Create and return ModelEmbedding
        let embedding = ModelEmbedding::new(ModelId::TemporalPositional, vector, latency_us);

        // Validate output (checks dimension, NaN, Inf)
        embedding.validate()?;

        Ok(embedding)
    }
}

// Implement Send and Sync explicitly (safe due to AtomicBool usage)
unsafe impl Send for TemporalPositionalModel {}
unsafe impl Sync for TemporalPositionalModel {}
