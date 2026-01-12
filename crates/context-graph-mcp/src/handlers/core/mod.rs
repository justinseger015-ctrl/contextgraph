//! Core Handlers struct and dispatch logic.
//!
//! TASK-S001: Updated to use TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
//! TASK-S003: Added GoalAlignmentCalculator for purpose/goal operations.
//! TASK-S004: Added JohariTransitionManager for johari/* handlers.
//! TASK-S005: Added MetaUtlTracker for meta_utl/* handlers.
//! TASK-GWT-001: Added GWT/Kuramoto provider traits for consciousness operations.
//! NO BACKWARDS COMPATIBILITY with legacy MemoryStore trait.
//!
//! # Module Organization
//!
//! This module is split into submodules for better maintainability:
//! - `types`: Type definitions (PredictionType, Domain, MetaLearningEvent, etc.)
//! - `meta_utl_tracker`: MetaUtlTracker struct and implementation
//! - `handlers`: Handlers struct definition and constructors
//! - `dispatch`: Request dispatch logic

mod dispatch;
mod handlers;
mod meta_utl_tracker;
mod types;

// Re-export all public types for backwards compatibility
pub use self::handlers::Handlers;
pub use self::meta_utl_tracker::MetaUtlTracker;
pub use self::types::{PredictionType, StoredPrediction};
