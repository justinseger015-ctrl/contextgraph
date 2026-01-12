//! Constants from Constitution (gwt.kuramoto)
//!
//! These constants define the Kuramoto synchronization parameters
//! and Global Workspace Theory thresholds.

/// Kuramoto coupling strength K from constitution (kuramoto_K: 2.0)
pub const KURAMOTO_K: f32 = 2.0;

/// Number of oscillators N for layer-level synchronization
pub const KURAMOTO_N: usize = 8;

/// Global workspace ignition threshold from constitution (coherence_threshold: 0.8)
/// Using 0.7 as per task spec for GW_THRESHOLD
pub const GW_THRESHOLD: f32 = 0.7;

/// Time step for Kuramoto integration (dt)
pub const KURAMOTO_DT: f32 = 0.01;

/// Number of integration steps per process call
pub const INTEGRATION_STEPS: usize = 10;

/// Hypersync threshold (r > 0.95 is pathological)
pub const HYPERSYNC_THRESHOLD: f32 = 0.95;

/// Fragmentation threshold (r < 0.5)
pub const FRAGMENTATION_THRESHOLD: f32 = 0.5;
