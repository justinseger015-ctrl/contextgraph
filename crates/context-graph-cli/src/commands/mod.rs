//! CLI command handlers
//!
//! # Modules
//!
//! - `consciousness`: Identity continuity and dream trigger commands (TASK-SESSION-08)
//! - `session`: Session identity persistence commands (TASK-SESSION-12, TASK-SESSION-13)
//! - `hooks`: Hook types for Claude Code native integration (TASK-HOOKS-001)

pub mod consciousness;
pub mod hooks;
pub mod session;

/// Test utilities for CLI tests
///
/// Provides a global test lock to serialize tests that access the global IdentityCache.
/// All tests modifying IdentityCache MUST acquire this lock.
#[cfg(test)]
pub mod test_utils {
    use std::sync::Mutex;

    /// Global test lock for serializing tests that access IdentityCache.
    ///
    /// Since IdentityCache is a process-global singleton, tests that modify it
    /// must be serialized to avoid race conditions.
    pub static GLOBAL_IDENTITY_LOCK: Mutex<()> = Mutex::new(());
}
