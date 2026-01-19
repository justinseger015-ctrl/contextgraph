//! Common test utilities and helper functions for GWT tests

use crate::types::fingerprint::{SemanticFingerprint, TeleologicalFingerprint};

/// Helper to create a test TeleologicalFingerprint with known semantic values
pub fn create_test_fingerprint(_seed_values: [f32; 13]) -> TeleologicalFingerprint {
    let semantic = SemanticFingerprint::zeroed();
    TeleologicalFingerprint::new(semantic, [0u8; 32])
}
