//! Phase 3: GWT (Global Workspace Theory) Consciousness Tools Manual Testing
//!
//! This module tests all 6 GWT consciousness tools:
//! 1. get_consciousness_state - Returns Psi (consciousness level)
//! 2. get_kuramoto_sync - Returns order parameter r and oscillator phases
//! 3. get_workspace_status - Returns active memory, coherence threshold, broadcasting status
//! 4. get_ego_state - Returns identity coherence, purpose vector, trajectory length
//! 5. trigger_workspace_broadcast - Performs winner-take-all selection
//! 6. adjust_coupling - Modifies Kuramoto coupling constant K
//!
//! # Test Scenarios
//!
//! ## Scenario 1: Consciousness State Verification
//! - Call get_consciousness_state
//! - Verify psi value is in valid range [0, 1]
//! - Verify layers (perception, memory, reasoning, action, meta) have valid status
//!
//! ## Scenario 2: Kuramoto Synchronization
//! - Call get_kuramoto_sync
//! - Verify r (order parameter) is in [0, 1]
//! - Verify phases array has 13 oscillators (one per embedder)
//! - Test adjust_coupling and verify K changes
//!
//! ## Scenario 3: Workspace Broadcast
//! - Store a memory first
//! - Call trigger_workspace_broadcast with the memory_id
//! - Verify workspace state changes (is_broadcasting, active_memory)
//!
//! ## Scenario 4: Ego State Verification
//! - Call get_ego_state
//! - Verify identity_status, purpose_vector (13D), trajectory_length
//!
//! # Critical Verification
//! - State values must be within valid ranges
//! - State changes must persist across calls
//! - Kuramoto must have 13 oscillators (one per embedder per constitution)

mod consciousness_state;
mod ego_state;
mod integration;
mod kuramoto_sync;
mod workspace_broadcast;
