//! Constants from Constitution for L4 Learning Layer.

/// Default learning rate (η) from constitution utl.constants.eta
pub const DEFAULT_LEARNING_RATE: f32 = 0.0005;

/// Consolidation threshold - trigger when weight delta exceeds this
pub const DEFAULT_CONSOLIDATION_THRESHOLD: f32 = 0.1;

/// Gradient clipping value from constitution (L4_Learning.grad_clip)
pub const GRADIENT_CLIP: f32 = 1.0;

/// Target frequency in Hz (100Hz = 10ms period)
pub const TARGET_FREQUENCY_HZ: u32 = 100;
