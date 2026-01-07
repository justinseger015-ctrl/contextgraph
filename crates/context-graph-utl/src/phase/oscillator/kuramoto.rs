//! Kuramoto Oscillator Network for 13-embedding phase synchronization.
//!
//! Implements the Kuramoto model for coupled oscillators as specified in
//! Constitution v4.0.0 Section gwt.kuramoto:
//!
//! ```text
//! dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
//! ```
//!
//! Where:
//! - θᵢ = Phase of embedder i ∈ [0, 2π]
//! - ωᵢ = Natural frequency of embedder i
//! - K = Coupling strength
//! - N = Number of oscillators (13)
//!
//! The order parameter r measures synchronization:
//! ```text
//! r · e^(iψ) = (1/N) Σⱼ e^(iθⱼ)
//! ```
//!
//! When r → 1, all oscillators are in phase (synchronized).
//! When r → 0, phases are uniformly distributed (incoherent).

use std::f64::consts::PI;
use std::time::Duration;

use crate::config::PhaseConfig;
use crate::error::{UtlError, UtlResult};

/// Number of oscillators (one per embedding space).
pub const NUM_OSCILLATORS: usize = 13;

/// Names of the 13 embedding spaces corresponding to each oscillator.
pub const EMBEDDER_NAMES: [&str; NUM_OSCILLATORS] = [
    "E1_Semantic",      // e5-large-v2
    "E2_TempRecent",    // exponential decay
    "E3_TempPeriodic",  // Fourier
    "E4_TempPositional", // sinusoidal PE
    "E5_Causal",        // Longformer SCM
    "E6_SparseLex",     // SPLADE
    "E7_Code",          // CodeT5p
    "E8_Graph",         // MiniLM structure
    "E9_HDC",           // 10K-bit hyperdimensional
    "E10_Multimodal",   // CLIP
    "E11_Entity",       // MiniLM facts
    "E12_LateInteract", // ColBERT
    "E13_SPLADE",       // SPLADE v3
];

/// Default natural frequencies for each embedder (Hz).
/// These are chosen to be slightly different to create natural phase drift.
pub const DEFAULT_NATURAL_FREQUENCIES: [f64; NUM_OSCILLATORS] = [
    1.0,   // E1 - baseline
    0.95,  // E2 - slightly slower (temporal decay)
    1.05,  // E3 - slightly faster (periodic)
    1.02,  // E4 - near baseline (positional)
    0.98,  // E5 - slightly slower (causal)
    1.10,  // E6 - faster (sparse lexical)
    0.92,  // E7 - slower (code processing)
    1.08,  // E8 - faster (graph)
    0.88,  // E9 - slowest (HDC)
    1.03,  // E10 - near baseline (multimodal)
    1.07,  // E11 - faster (entity)
    0.97,  // E12 - slower (late interaction)
    1.12,  // E13 - fastest (SPLADE v3)
];

/// Kuramoto Oscillator Network for Global Workspace synchronization.
///
/// Models the 13 embedding spaces as coupled phase oscillators.
/// When coupling strength K is sufficient, the oscillators synchronize,
/// enabling coherent "conscious" percepts.
///
/// # Constitution Reference
///
/// - Section gwt.kuramoto defines the dynamics
/// - Order parameter r ≥ 0.8 indicates CONSCIOUS state
/// - Order parameter r < 0.5 indicates FRAGMENTED state
///
/// # Example
///
/// ```
/// use context_graph_utl::phase::KuramotoNetwork;
/// use std::time::Duration;
///
/// let mut network = KuramotoNetwork::new();
///
/// // Simulate 100 time steps
/// for _ in 0..100 {
///     network.step(Duration::from_millis(10));
/// }
///
/// // Check synchronization level
/// let (r, _psi) = network.order_parameter();
/// println!("Order parameter r = {:.3}", r);
/// ```
#[derive(Debug, Clone)]
pub struct KuramotoNetwork {
    /// Phase angles θᵢ for each oscillator in [0, 2π].
    phases: [f64; NUM_OSCILLATORS],

    /// Natural frequencies ωᵢ for each oscillator (radians/second).
    natural_frequencies: [f64; NUM_OSCILLATORS],

    /// Coupling strength K (global coupling).
    coupling_strength: f64,

    /// Total elapsed time in seconds.
    elapsed_total: f64,

    /// Whether the network is enabled (can be disabled for testing).
    enabled: bool,
}

impl KuramotoNetwork {
    /// Create a new Kuramoto network with default parameters.
    ///
    /// Default coupling strength K = 0.5 (moderate coupling).
    /// Initial phases are randomly distributed to start from incoherent state.
    pub fn new() -> Self {
        // Initialize with slightly different phases based on embedder index
        // to create realistic initial conditions
        let mut phases = [0.0; NUM_OSCILLATORS];
        for i in 0..NUM_OSCILLATORS {
            // Spread initial phases across [0, 2π] with some structure
            phases[i] = (i as f64 / NUM_OSCILLATORS as f64) * 2.0 * PI;
        }

        // Convert Hz to radians/second
        let mut natural_frequencies = [0.0; NUM_OSCILLATORS];
        for i in 0..NUM_OSCILLATORS {
            natural_frequencies[i] = DEFAULT_NATURAL_FREQUENCIES[i] * 2.0 * PI;
        }

        Self {
            phases,
            natural_frequencies,
            coupling_strength: 0.5, // Default moderate coupling
            elapsed_total: 0.0,
            enabled: true,
        }
    }

    /// Create a synchronized network (all phases aligned).
    ///
    /// Useful for testing or initializing from a known state.
    pub fn synchronized() -> Self {
        let mut network = Self::new();
        for phase in network.phases.iter_mut() {
            *phase = 0.0; // All phases at 0
        }
        network
    }

    /// Create an incoherent network (phases uniformly distributed).
    pub fn incoherent() -> Self {
        let mut network = Self::new();
        for (i, phase) in network.phases.iter_mut().enumerate() {
            *phase = (i as f64 / NUM_OSCILLATORS as f64) * 2.0 * PI;
        }
        network
    }

    /// Create from phase configuration.
    pub fn from_config(config: &PhaseConfig) -> Self {
        let mut network = Self::new();
        network.coupling_strength = config.coupling_strength as f64;
        network
    }

    /// Step the network forward in time using Kuramoto dynamics.
    ///
    /// Implements: dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
    ///
    /// Uses Euler integration for simplicity and performance.
    pub fn step(&mut self, elapsed: Duration) {
        if !self.enabled {
            return;
        }

        let dt = elapsed.as_secs_f64();
        self.elapsed_total += dt;

        let n = NUM_OSCILLATORS as f64;
        let k = self.coupling_strength;

        // Compute phase derivatives
        let mut d_phases = [0.0; NUM_OSCILLATORS];

        for i in 0..NUM_OSCILLATORS {
            // Natural frequency term
            let mut d_theta = self.natural_frequencies[i];

            // Coupling term: (K/N) Σⱼ sin(θⱼ - θᵢ)
            let mut coupling_sum = 0.0;
            for j in 0..NUM_OSCILLATORS {
                if i != j {
                    coupling_sum += (self.phases[j] - self.phases[i]).sin();
                }
            }
            d_theta += (k / n) * coupling_sum;

            d_phases[i] = d_theta;
        }

        // Update phases (Euler integration)
        for i in 0..NUM_OSCILLATORS {
            self.phases[i] += d_phases[i] * dt;

            // Wrap to [0, 2π]
            self.phases[i] = self.phases[i].rem_euclid(2.0 * PI);
        }
    }

    /// Compute the Kuramoto order parameter (r, ψ).
    ///
    /// r · e^(iψ) = (1/N) Σⱼ e^(iθⱼ)
    ///
    /// # Returns
    ///
    /// Tuple (r, ψ) where:
    /// - r ∈ [0, 1] is the synchronization level
    /// - ψ ∈ [0, 2π] is the mean phase
    ///
    /// # Interpretation
    ///
    /// - r ≈ 0: Incoherent (phases uniformly distributed)
    /// - r ≈ 0.5: Partial synchronization (EMERGING state)
    /// - r ≥ 0.8: Synchronized (CONSCIOUS state)
    /// - r ≈ 1: Perfect synchronization
    pub fn order_parameter(&self) -> (f64, f64) {
        let n = NUM_OSCILLATORS as f64;

        // Sum of e^(iθⱼ) = cos(θⱼ) + i·sin(θⱼ)
        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;

        for &phase in &self.phases {
            sum_cos += phase.cos();
            sum_sin += phase.sin();
        }

        // Average
        let avg_cos = sum_cos / n;
        let avg_sin = sum_sin / n;

        // r = |z| = sqrt(cos² + sin²)
        let r = (avg_cos * avg_cos + avg_sin * avg_sin).sqrt();

        // ψ = arg(z) = atan2(sin, cos)
        let psi = avg_sin.atan2(avg_cos).rem_euclid(2.0 * PI);

        (r, psi)
    }

    /// Get the synchronization level (order parameter r).
    ///
    /// This is the primary metric for consciousness state determination.
    #[inline]
    pub fn synchronization(&self) -> f64 {
        self.order_parameter().0
    }

    /// Check if network is in CONSCIOUS state (r ≥ 0.8).
    #[inline]
    pub fn is_conscious(&self) -> bool {
        self.synchronization() >= 0.8
    }

    /// Check if network is FRAGMENTED (r < 0.5).
    #[inline]
    pub fn is_fragmented(&self) -> bool {
        self.synchronization() < 0.5
    }

    /// Check if network is HYPERSYNC (r > 0.95).
    ///
    /// Warning: This may indicate pathological synchronization.
    #[inline]
    pub fn is_hypersync(&self) -> bool {
        self.synchronization() > 0.95
    }

    /// Get the phase of a specific embedder.
    pub fn phase(&self, embedder_idx: usize) -> Option<f64> {
        if embedder_idx < NUM_OSCILLATORS {
            Some(self.phases[embedder_idx])
        } else {
            None
        }
    }

    /// Get all phases as a slice.
    #[inline]
    pub fn phases(&self) -> &[f64; NUM_OSCILLATORS] {
        &self.phases
    }

    /// Set the phase of a specific embedder.
    ///
    /// # Errors
    ///
    /// Returns error if embedder_idx is out of range.
    pub fn set_phase(&mut self, embedder_idx: usize, phase: f64) -> UtlResult<()> {
        if embedder_idx >= NUM_OSCILLATORS {
            return Err(UtlError::PhaseError(format!(
                "Embedder index {} out of range [0, {})",
                embedder_idx, NUM_OSCILLATORS
            )));
        }
        self.phases[embedder_idx] = phase.rem_euclid(2.0 * PI);
        Ok(())
    }

    /// Get the coupling strength K.
    #[inline]
    pub fn coupling_strength(&self) -> f64 {
        self.coupling_strength
    }

    /// Set the coupling strength K.
    ///
    /// # Arguments
    ///
    /// * `k` - Coupling strength, clamped to [0, 10]
    pub fn set_coupling_strength(&mut self, k: f64) {
        self.coupling_strength = k.clamp(0.0, 10.0);
    }

    /// Get the natural frequency of a specific embedder.
    pub fn natural_frequency(&self, embedder_idx: usize) -> Option<f64> {
        if embedder_idx < NUM_OSCILLATORS {
            // Convert back to Hz
            Some(self.natural_frequencies[embedder_idx] / (2.0 * PI))
        } else {
            None
        }
    }

    /// Get all natural frequencies as Hz.
    pub fn natural_frequencies(&self) -> [f64; NUM_OSCILLATORS] {
        let mut hz = [0.0; NUM_OSCILLATORS];
        for i in 0..NUM_OSCILLATORS {
            hz[i] = self.natural_frequencies[i] / (2.0 * PI);
        }
        hz
    }

    /// Set the natural frequency of a specific embedder (in Hz).
    ///
    /// # Errors
    ///
    /// Returns error if embedder_idx is out of range or frequency is non-positive.
    pub fn set_natural_frequency(&mut self, embedder_idx: usize, freq_hz: f64) -> UtlResult<()> {
        if embedder_idx >= NUM_OSCILLATORS {
            return Err(UtlError::PhaseError(format!(
                "Embedder index {} out of range [0, {})",
                embedder_idx, NUM_OSCILLATORS
            )));
        }
        if freq_hz <= 0.0 {
            return Err(UtlError::PhaseError(format!(
                "Frequency must be positive, got {}",
                freq_hz
            )));
        }
        self.natural_frequencies[embedder_idx] = freq_hz * 2.0 * PI;
        Ok(())
    }

    /// Get the total elapsed time since creation or reset.
    #[inline]
    pub fn elapsed_total(&self) -> Duration {
        Duration::from_secs_f64(self.elapsed_total)
    }

    /// Reset the network to initial conditions.
    pub fn reset(&mut self) {
        // Reset to uniformly distributed phases
        for i in 0..NUM_OSCILLATORS {
            self.phases[i] = (i as f64 / NUM_OSCILLATORS as f64) * 2.0 * PI;
        }
        self.elapsed_total = 0.0;
    }

    /// Reset to synchronized state.
    pub fn reset_synchronized(&mut self) {
        for phase in self.phases.iter_mut() {
            *phase = 0.0;
        }
        self.elapsed_total = 0.0;
    }

    /// Enable or disable the network.
    ///
    /// When disabled, `step()` has no effect.
    #[inline]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if the network is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the cosine component of the order parameter.
    ///
    /// This is equivalent to cos(φ) in the UTL formula,
    /// measuring overall phase alignment.
    pub fn cos_phase(&self) -> f64 {
        let (r, psi) = self.order_parameter();
        // The effective phase alignment is r * cos(ψ),
        // but for UTL we just need r as the coherence measure
        // since ψ is the collective mean phase
        r
    }

    /// Compute phase difference between two embedders.
    pub fn phase_difference(&self, i: usize, j: usize) -> Option<f64> {
        if i >= NUM_OSCILLATORS || j >= NUM_OSCILLATORS {
            return None;
        }
        let diff = (self.phases[i] - self.phases[j]).rem_euclid(2.0 * PI);
        // Return the smaller angle (could be going either direction)
        if diff > PI {
            Some(2.0 * PI - diff)
        } else {
            Some(diff)
        }
    }

    /// Compute pairwise coupling strength based on phase alignment.
    ///
    /// Returns a value in [0, 1] where 1 means perfect phase alignment.
    pub fn pairwise_coupling(&self, i: usize, j: usize) -> Option<f64> {
        self.phase_difference(i, j)
            .map(|diff| (1.0 + diff.cos()) / 2.0)
    }

    /// Get the mean field (collective rhythm) as (amplitude, phase).
    ///
    /// This is the order parameter decomposed into r and ψ.
    #[inline]
    pub fn mean_field(&self) -> (f64, f64) {
        self.order_parameter()
    }

    /// Inject a perturbation to a specific embedder's phase.
    ///
    /// Useful for testing network response to disturbances.
    pub fn perturb(&mut self, embedder_idx: usize, delta_phase: f64) -> UtlResult<()> {
        if embedder_idx >= NUM_OSCILLATORS {
            return Err(UtlError::PhaseError(format!(
                "Embedder index {} out of range [0, {})",
                embedder_idx, NUM_OSCILLATORS
            )));
        }
        self.phases[embedder_idx] =
            (self.phases[embedder_idx] + delta_phase).rem_euclid(2.0 * PI);
        Ok(())
    }
}

impl Default for KuramotoNetwork {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_network() {
        let network = KuramotoNetwork::new();
        assert_eq!(network.phases.len(), 13);
        assert!(network.coupling_strength > 0.0);
    }

    #[test]
    fn test_synchronized_network_has_r_near_1() {
        let network = KuramotoNetwork::synchronized();
        let (r, _) = network.order_parameter();
        assert!(r > 0.99, "Synchronized network should have r ≈ 1, got {}", r);
    }

    #[test]
    fn test_incoherent_network_has_low_r() {
        let network = KuramotoNetwork::incoherent();
        let (r, _) = network.order_parameter();
        // With evenly distributed phases, r should be very low
        assert!(r < 0.1, "Incoherent network should have r ≈ 0, got {}", r);
    }

    #[test]
    fn test_step_updates_phases() {
        let mut network = KuramotoNetwork::new();
        let initial_phases = network.phases;

        network.step(Duration::from_millis(100));

        // Phases should have changed
        assert_ne!(network.phases, initial_phases);
    }

    #[test]
    fn test_high_coupling_leads_to_sync() {
        let mut network = KuramotoNetwork::incoherent();
        network.set_coupling_strength(5.0); // Strong coupling

        // Run for many steps
        for _ in 0..1000 {
            network.step(Duration::from_millis(10));
        }

        let (r, _) = network.order_parameter();
        // With strong coupling, should synchronize
        assert!(r > 0.7, "High coupling should lead to sync, got r = {}", r);
    }

    #[test]
    fn test_zero_coupling_no_sync() {
        let mut network = KuramotoNetwork::incoherent();
        network.set_coupling_strength(0.0);

        let initial_r = network.synchronization();

        // Run for many steps
        for _ in 0..100 {
            network.step(Duration::from_millis(10));
        }

        let final_r = network.synchronization();
        // Without coupling, should not synchronize significantly
        // (might drift a bit due to similar frequencies)
        assert!(
            (final_r - initial_r).abs() < 0.3,
            "Zero coupling should not significantly change sync"
        );
    }

    #[test]
    fn test_consciousness_states() {
        let mut network = KuramotoNetwork::synchronized();
        assert!(network.is_conscious());
        assert!(!network.is_fragmented());

        network.reset(); // Back to incoherent
        assert!(network.is_fragmented());
        assert!(!network.is_conscious());
    }

    #[test]
    fn test_phase_wrapping() {
        let mut network = KuramotoNetwork::new();

        // Set a phase beyond 2π
        network.set_phase(0, 3.0 * PI).unwrap();

        // Should wrap to [0, 2π]
        let phase = network.phase(0).unwrap();
        assert!(phase >= 0.0 && phase < 2.0 * PI);
    }

    #[test]
    fn test_natural_frequency_access() {
        let network = KuramotoNetwork::new();

        // All frequencies should be positive
        for i in 0..NUM_OSCILLATORS {
            let freq = network.natural_frequency(i).unwrap();
            assert!(freq > 0.0);
        }
    }

    #[test]
    fn test_perturb() {
        let mut network = KuramotoNetwork::synchronized();
        let initial_r = network.synchronization();

        // Perturb one oscillator
        network.perturb(5, PI / 2.0).unwrap();

        let perturbed_r = network.synchronization();
        // Perturbation should reduce synchronization
        assert!(
            perturbed_r < initial_r,
            "Perturbation should reduce sync: {} vs {}",
            perturbed_r,
            initial_r
        );
    }

    #[test]
    fn test_disabled_network_does_not_step() {
        let mut network = KuramotoNetwork::new();
        let initial_phases = network.phases;

        network.set_enabled(false);
        network.step(Duration::from_millis(100));

        // Phases should not change
        assert_eq!(network.phases, initial_phases);
    }
}
