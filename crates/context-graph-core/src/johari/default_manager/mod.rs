//! Default implementation of JohariTransitionManager using TeleologicalMemoryStore.
//!
//! This module provides `DefaultJohariManager`, a concrete implementation that:
//! - Uses TeleologicalMemoryStore for persistence
//! - Validates all transitions via JohariQuadrant state machine
//! - Implements blind spot discovery algorithm
//! - Supports batch operations with all-or-nothing semantics
//! - Persists transition history in-memory for stats and history queries
//!
//! # Module Structure
//!
//! - `types` - Core struct definitions (`DefaultJohariManager`, `DynDefaultJohariManager`)
//! - `impl_generic` - Generic implementation for `DefaultJohariManager<S>`
//! - `impl_dyn` - Dynamic (type-erased) implementation for `DynDefaultJohariManager`
//! - `helpers` - Internal helper functions

mod helpers;
mod impl_dyn;
mod impl_generic;
mod types;

#[cfg(test)]
mod tests;

// Re-export public types
pub use types::{DefaultJohariManager, DynDefaultJohariManager};
