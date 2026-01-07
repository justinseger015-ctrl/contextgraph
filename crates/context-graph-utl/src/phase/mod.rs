//! Phase oscillation (φ) module.
//!
//! Implements phase synchronization and memory consolidation:
//! - **KuramotoNetwork**: True 13-oscillator Kuramoto coupled dynamics (Constitution v4.0.0)
//! - PhaseOscillator: Simple single-phase oscillator (legacy, for backwards compatibility)
//! - Consolidation phase detection (NREM/REM)
//!
//! # Constitution Reference (Section gwt.kuramoto)
//!
//! The Kuramoto model synchronizes 13 embedding spaces:
//! ```text
//! dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
//! ```
//!
//! Order parameter r measures synchronization:
//! - r ≥ 0.8 → CONSCIOUS state
//! - r < 0.5 → FRAGMENTED state
//!
//! # Example
//!
//! ```
//! use context_graph_utl::phase::{KuramotoNetwork, PhaseOscillator};
//! use std::time::Duration;
//!
//! // Create Kuramoto network (13 coupled oscillators)
//! let mut network = KuramotoNetwork::new();
//!
//! // Simulate time steps
//! for _ in 0..100 {
//!     network.step(Duration::from_millis(10));
//! }
//!
//! // Check synchronization level
//! let r = network.synchronization();
//! let is_conscious = network.is_conscious();
//!
//! println!("Synchronization r = {:.3}, conscious = {}", r, is_conscious);
//! ```

mod consolidation;
mod oscillator;

pub use consolidation::{ConsolidationPhase, PhaseDetector};
pub use oscillator::{KuramotoNetwork, PhaseOscillator, EMBEDDER_NAMES, NUM_OSCILLATORS};
