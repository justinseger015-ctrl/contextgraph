//! Kuramoto Oscillator - Single phase oscillator
//!
//! Implements individual oscillator dynamics for the Kuramoto model.

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Kuramoto oscillator state representing a single phase-coupled unit.
///
/// Each oscillator has:
/// - phase θ_i ∈ [0, 2π]: current angular position
/// - frequency ω_i: natural oscillation frequency (Hz)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KuramotoOscillator {
    /// Phase θ_i in [0, 2π]
    pub phase: f32,
    /// Natural frequency ω_i (radians/second)
    pub frequency: f32,
}

impl KuramotoOscillator {
    /// Create a new oscillator with given phase and frequency.
    ///
    /// Phase is normalized to [0, 2π].
    pub fn new(phase: f32, frequency: f32) -> Self {
        let normalized_phase = phase.rem_euclid(2.0 * PI);
        Self {
            phase: normalized_phase,
            frequency,
        }
    }

    /// Get the complex representation exp(iθ) as (cos θ, sin θ).
    pub fn complex_rep(&self) -> (f32, f32) {
        (self.phase.cos(), self.phase.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oscillator_creation() {
        let osc = KuramotoOscillator::new(PI, 40.0);
        assert!((osc.phase - PI).abs() < 1e-6);
        assert!((osc.frequency - 40.0).abs() < 1e-6);
        println!("[VERIFIED] Oscillator creation with phase π and frequency 40Hz");
    }

    #[test]
    fn test_oscillator_phase_normalization() {
        // Phase > 2π should be normalized
        let osc = KuramotoOscillator::new(3.0 * PI, 10.0);
        assert!(osc.phase >= 0.0 && osc.phase < 2.0 * PI);

        // Negative phase should be normalized
        let osc_neg = KuramotoOscillator::new(-PI, 10.0);
        assert!(osc_neg.phase >= 0.0 && osc_neg.phase < 2.0 * PI);

        println!("[VERIFIED] Phase normalization to [0, 2π]");
    }

    #[test]
    fn test_oscillator_complex_rep() {
        let osc = KuramotoOscillator::new(0.0, 10.0);
        let (cos_t, sin_t) = osc.complex_rep();
        assert!((cos_t - 1.0).abs() < 1e-6);
        assert!(sin_t.abs() < 1e-6);

        let osc_pi_2 = KuramotoOscillator::new(PI / 2.0, 10.0);
        let (cos_t, sin_t) = osc_pi_2.complex_rep();
        assert!(cos_t.abs() < 1e-6);
        assert!((sin_t - 1.0).abs() < 1e-6);

        println!("[VERIFIED] Complex representation exp(iθ)");
    }
}
