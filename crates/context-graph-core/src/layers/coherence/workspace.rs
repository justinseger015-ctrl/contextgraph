//! Global Workspace State and Consciousness State Machine
//!
//! Implements Global Workspace Theory (GWT) state management.

use serde::{Deserialize, Serialize};

use super::constants::{FRAGMENTATION_THRESHOLD, HYPERSYNC_THRESHOLD};

/// Global Workspace state for GWT implementation.
///
/// The Global Workspace represents the currently "conscious" content
/// that is broadcast to all subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalWorkspace {
    /// Whether the workspace is active (ignited)
    pub active: bool,
    /// Current ignition level (r from Kuramoto)
    pub ignition_level: f32,
    /// Broadcast content when ignited
    pub broadcast_content: Option<serde_json::Value>,
    /// Current consciousness state
    pub state: ConsciousnessState,
}

impl Default for GlobalWorkspace {
    fn default() -> Self {
        Self {
            active: false,
            ignition_level: 0.0,
            broadcast_content: None,
            state: ConsciousnessState::Dormant,
        }
    }
}

/// Consciousness state from GWT state machine.
///
/// From constitution gwt.state_machine.states
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConsciousnessState {
    /// r < 0.3, no active workspace
    Dormant,
    /// 0.3 ≤ r < 0.5, partial sync
    Fragmented,
    /// 0.5 ≤ r < 0.8, approaching coherence
    Emerging,
    /// r ≥ 0.8, unified percept active
    Conscious,
    /// r > 0.95, possibly pathological
    Hypersync,
}

impl ConsciousnessState {
    /// Determine state from order parameter r.
    pub fn from_order_parameter(r: f32) -> Self {
        if r > HYPERSYNC_THRESHOLD {
            Self::Hypersync
        } else if r >= 0.8 {
            Self::Conscious
        } else if r >= FRAGMENTATION_THRESHOLD {
            Self::Emerging
        } else if r >= 0.3 {
            Self::Fragmented
        } else {
            Self::Dormant
        }
    }

    /// Check if this is a healthy state (not Dormant or Hypersync).
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Fragmented | Self::Emerging | Self::Conscious)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_state_from_r() {
        assert_eq!(
            ConsciousnessState::from_order_parameter(0.1),
            ConsciousnessState::Dormant
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter(0.35),
            ConsciousnessState::Fragmented
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter(0.6),
            ConsciousnessState::Emerging
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter(0.85),
            ConsciousnessState::Conscious
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter(0.98),
            ConsciousnessState::Hypersync
        );
        println!("[VERIFIED] Consciousness state classification from r");
    }

    #[test]
    fn test_consciousness_state_health() {
        assert!(!ConsciousnessState::Dormant.is_healthy());
        assert!(ConsciousnessState::Fragmented.is_healthy());
        assert!(ConsciousnessState::Emerging.is_healthy());
        assert!(ConsciousnessState::Conscious.is_healthy());
        assert!(!ConsciousnessState::Hypersync.is_healthy());
        println!("[VERIFIED] Consciousness state health check");
    }
}
