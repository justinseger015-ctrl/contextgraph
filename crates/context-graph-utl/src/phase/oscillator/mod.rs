//! Phase oscillator for learning rhythms.
//!
//! Provides smooth phase oscillation with configurable frequency and coupling
//! strength for the UTL formula phase component `cos(φ)`.
//!
//! # Constitution Compliance
//!
//! This module implements the Kuramoto oscillator network as specified in
//! Constitution v4.0.0 Section gwt.kuramoto. The `KuramotoNetwork` provides
//! true 13-oscillator coupled dynamics for Global Workspace synchronization.
//!
//! The simple `PhaseOscillator` is retained for backwards compatibility but
//! should be deprecated in favor of `KuramotoNetwork` for consciousness-aware
//! applications.

mod core;
mod coupling;
mod kuramoto;
mod types;

#[cfg(test)]
mod tests;

// Re-export the main types
pub use kuramoto::{KuramotoNetwork, EMBEDDER_NAMES, NUM_OSCILLATORS};
pub use types::PhaseOscillator;
