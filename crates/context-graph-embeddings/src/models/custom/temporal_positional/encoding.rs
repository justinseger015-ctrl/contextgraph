//! Positional encoding computation for the Temporal-Positional model (E4).
//!
//! Supports both session sequence positions (preferred) and Unix timestamps (fallback).

use chrono::{DateTime, Utc};

use super::constants::TEMPORAL_POSITIONAL_DIMENSION;

/// Compute the transformer-style positional encoding for a given position.
///
/// The embedding uses the standard formula from "Attention Is All You Need":
/// - PE(pos, 2i) = sin(pos / base^(2i/d_model))
/// - PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
///
/// # Arguments
/// * `position` - The position value (session sequence number or Unix timestamp)
/// * `base` - Base frequency for positional encoding
/// * `d_model` - Model dimension (always 512)
/// * `is_sequence` - If true, position is a session sequence number (smaller values),
///                   if false, position is a Unix timestamp (larger values).
///                   This affects the scaling to ensure good gradient distribution.
///
/// # Returns
/// A 512-dimensional L2-normalized vector
///
/// # Scaling
/// - Sequence mode: Uses position directly (typically 0-10000 range within a session)
/// - Timestamp mode: Uses Unix seconds (large values like 1700000000)
///
/// The sinusoidal encoding naturally handles both ranges, but sequence mode
/// will have more distinct encodings for small position differences.
pub fn compute_positional_encoding_from_position(
    position: i64,
    base: f32,
    d_model: usize,
    is_sequence: bool,
) -> Vec<f32> {
    // For sequence mode, use a smaller base to get more distinct encodings
    // for consecutive positions (0, 1, 2, 3...). For timestamp mode, keep
    // the standard base for larger position values.
    let effective_base = if is_sequence {
        // Sequence numbers are small (0-10000), so use smaller base for
        // better differentiation between consecutive positions
        (base / 100.0).max(10.0)
    } else {
        base
    };

    let pos = position as f64;
    let d_model_f64 = d_model as f64;
    let base_f64 = effective_base as f64;

    let mut vector = Vec::with_capacity(TEMPORAL_POSITIONAL_DIMENSION);

    // Transformer PE formula:
    // PE(pos, 2i) = sin(pos / base^(2i/d_model))
    // PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
    for i in 0..(d_model / 2) {
        let i_f64 = i as f64;
        let exponent = 2.0 * i_f64 / d_model_f64;
        let div_term = base_f64.powf(exponent);

        let angle = pos / div_term;
        let sin_val = angle.sin() as f32;
        let cos_val = angle.cos() as f32;

        vector.push(sin_val);
        vector.push(cos_val);
    }

    // L2 normalize
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for v in &mut vector {
            *v /= norm;
        }
    }

    vector
}

/// Compute the transformer-style positional encoding for a given timestamp.
///
/// Legacy API for backward compatibility. Internally delegates to
/// `compute_positional_encoding_from_position` with is_sequence=false.
///
/// The embedding uses the standard formula from "Attention Is All You Need":
/// - PE(pos, 2i) = sin(pos / base^(2i/d_model))
/// - PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
///
/// # Arguments
/// * `timestamp` - The timestamp to encode
/// * `base` - Base frequency for positional encoding
/// * `d_model` - Model dimension (always 512)
///
/// # Returns
/// A 512-dimensional L2-normalized vector
pub fn compute_positional_encoding(
    timestamp: DateTime<Utc>,
    base: f32,
    d_model: usize,
) -> Vec<f32> {
    compute_positional_encoding_from_position(timestamp.timestamp(), base, d_model, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_encoding_distinct_consecutive() {
        // Consecutive sequence positions should have distinct encodings
        let enc0 = compute_positional_encoding_from_position(0, 10000.0, 512, true);
        let enc1 = compute_positional_encoding_from_position(1, 10000.0, 512, true);
        let enc2 = compute_positional_encoding_from_position(2, 10000.0, 512, true);

        // Compute cosine similarities
        let sim_01: f32 = enc0.iter().zip(&enc1).map(|(a, b)| a * b).sum();
        let sim_12: f32 = enc1.iter().zip(&enc2).map(|(a, b)| a * b).sum();
        let sim_02: f32 = enc0.iter().zip(&enc2).map(|(a, b)| a * b).sum();

        // Adjacent positions should be more similar than non-adjacent
        assert!(sim_01 > sim_02, "Adjacent positions should be more similar");
        assert!(sim_12 > sim_02, "Adjacent positions should be more similar");

        // But all should be distinct (not identical)
        assert!(sim_01 < 0.999, "Consecutive positions should be distinct");
        assert!(sim_12 < 0.999, "Consecutive positions should be distinct");
    }

    #[test]
    fn test_timestamp_encoding_backward_compatible() {
        use chrono::TimeZone;

        let ts = Utc.timestamp_opt(1705315800, 0).unwrap();
        let enc_legacy = compute_positional_encoding(ts, 10000.0, 512);
        let enc_new = compute_positional_encoding_from_position(1705315800, 10000.0, 512, false);

        // Should produce identical results
        assert_eq!(enc_legacy.len(), enc_new.len());
        for (a, b) in enc_legacy.iter().zip(&enc_new) {
            assert!((a - b).abs() < 1e-6, "Legacy and new APIs should match");
        }
    }

    #[test]
    fn test_encoding_is_normalized() {
        let enc_seq = compute_positional_encoding_from_position(42, 10000.0, 512, true);
        let enc_ts = compute_positional_encoding_from_position(1705315800, 10000.0, 512, false);

        let norm_seq: f32 = enc_seq.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_ts: f32 = enc_ts.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!((norm_seq - 1.0).abs() < 1e-5, "Sequence encoding should be normalized");
        assert!((norm_ts - 1.0).abs() < 1e-5, "Timestamp encoding should be normalized");
    }

    #[test]
    fn test_dimension_is_correct() {
        let enc = compute_positional_encoding_from_position(100, 10000.0, 512, true);
        assert_eq!(enc.len(), 512);
    }
}
