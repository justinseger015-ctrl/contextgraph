//! Core implementation for TeleologicalFingerprint.
//!
//! This module contains constructors, constants, and core methods.

use chrono::Utc;
use uuid::Uuid;

use crate::types::fingerprint::SemanticFingerprint;

use super::types::TeleologicalFingerprint;

impl TeleologicalFingerprint {
    /// Expected size in bytes for a complete teleological fingerprint.
    /// From constitution.yaml: ~46KB per node.
    pub const EXPECTED_SIZE_BYTES: usize = 46_000;

    /// Create a new TeleologicalFingerprint.
    ///
    /// Automatically:
    /// - Generates a new UUID v4
    /// - Sets timestamps to now
    ///
    /// # Arguments
    /// * `semantic` - The semantic fingerprint (13 embeddings)
    /// * `content_hash` - SHA-256 hash of source content
    pub fn new(semantic: SemanticFingerprint, content_hash: [u8; 32]) -> Self {
        let now = Utc::now();

        Self {
            id: Uuid::new_v4(),
            semantic,
            content_hash,
            created_at: now,
            last_updated: now,
            access_count: 0,
        }
    }

    /// Create a TeleologicalFingerprint with a specific ID (for testing/import).
    pub fn with_id(id: Uuid, semantic: SemanticFingerprint, content_hash: [u8; 32]) -> Self {
        let mut fp = Self::new(semantic, content_hash);
        fp.id = id;
        fp
    }

    /// Record an access event.
    ///
    /// Increments access_count and updates last_updated timestamp.
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_updated = Utc::now();
    }

    /// Get the age of this fingerprint (time since creation).
    pub fn age(&self) -> chrono::Duration {
        Utc::now() - self.created_at
    }
}
