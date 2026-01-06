//! Decay embedding computation for the Temporal-Recent model.

use chrono::{DateTime, Utc};

use super::constants::{FEATURES_PER_SCALE, MAX_TIME_DELTA_SECS, TEMPORAL_RECENT_DIMENSION};

/// Compute the decay embedding for a given timestamp.
///
/// The embedding encodes temporal recency across multiple time scales
/// using exponential decay with phase variations.
///
/// # Arguments
/// * `timestamp` - The timestamp to encode
/// * `reference_time` - Reference time for computing time delta (None = use current time)
/// * `decay_rates` - Decay rates for each time scale
///
/// # Returns
/// A 512-dimensional L2-normalized vector encoding temporal recency.
pub fn compute_decay_embedding(
    timestamp: DateTime<Utc>,
    reference_time: Option<DateTime<Utc>>,
    decay_rates: &[f32],
) -> Vec<f32> {
    let reference = reference_time.unwrap_or_else(Utc::now);
    let time_delta_secs = (reference - timestamp).num_seconds() as f32;

    let mut vector = Vec::with_capacity(TEMPORAL_RECENT_DIMENSION);

    for &decay_rate in decay_rates {
        // Clamp to prevent numerical issues with very old/future timestamps
        let clamped_delta = time_delta_secs.clamp(0.0, MAX_TIME_DELTA_SECS);
        let base_decay = (-decay_rate * clamped_delta).exp();

        // Generate 128 features for this time scale with phase variations
        for i in 0..FEATURES_PER_SCALE {
            let phase = (i as f32) * std::f32::consts::PI / 64.0;
            let value = base_decay * (phase + clamped_delta * decay_rate * 0.001).cos();
            vector.push(value);
        }
    }

    // L2 normalize
    l2_normalize(&mut vector);

    vector
}

/// L2 normalize a vector in place.
///
/// If the vector has zero magnitude (within epsilon), leaves it unchanged.
#[inline]
fn l2_normalize(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for v in vector.iter_mut() {
            *v /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_compute_decay_embedding_dimension() {
        let ref_time = Utc::now();
        let timestamp = ref_time - Duration::hours(1);
        let decay_rates = vec![1.0 / 3600.0, 1.0 / 86400.0, 1.0 / 604800.0, 1.0 / 2592000.0];

        let embedding = compute_decay_embedding(timestamp, Some(ref_time), &decay_rates);

        assert_eq!(embedding.len(), 512, "Must produce 512D vector");
    }

    #[test]
    fn test_compute_decay_embedding_normalized() {
        let ref_time = Utc::now();
        let timestamp = ref_time - Duration::hours(1);
        let decay_rates = vec![1.0 / 3600.0, 1.0 / 86400.0, 1.0 / 604800.0, 1.0 / 2592000.0];

        let embedding = compute_decay_embedding(timestamp, Some(ref_time), &decay_rates);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (norm - 1.0).abs() < 0.001,
            "Vector must be L2 normalized, got norm = {}",
            norm
        );
    }

    #[test]
    fn test_compute_decay_embedding_no_nan() {
        let ref_time = Utc::now();
        let timestamp = ref_time - Duration::days(365); // Very old
        let decay_rates = vec![1.0 / 3600.0, 1.0 / 86400.0, 1.0 / 604800.0, 1.0 / 2592000.0];

        let embedding = compute_decay_embedding(timestamp, Some(ref_time), &decay_rates);

        assert!(
            embedding.iter().all(|x| x.is_finite()),
            "Must not contain NaN or Inf values"
        );
    }

    #[test]
    fn test_compute_decay_embedding_future_timestamp() {
        let ref_time = Utc::now();
        let timestamp = ref_time + Duration::days(30); // Future
        let decay_rates = vec![1.0 / 3600.0, 1.0 / 86400.0, 1.0 / 604800.0, 1.0 / 2592000.0];

        let embedding = compute_decay_embedding(timestamp, Some(ref_time), &decay_rates);

        assert!(
            embedding.iter().all(|x| x.is_finite()),
            "Future timestamps must produce valid output"
        );
    }

    #[test]
    fn test_l2_normalize() {
        let mut vector = vec![3.0, 4.0];
        l2_normalize(&mut vector);

        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
        assert!((vector[0] - 0.6).abs() < 0.001);
        assert!((vector[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut vector = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut vector);

        // Zero vector should remain unchanged
        assert!(vector.iter().all(|&x| x == 0.0));
    }
}
