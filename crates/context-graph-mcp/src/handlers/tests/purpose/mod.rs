//! Purpose Handler Tests
//!
//! TASK-S003: Tests for purpose/query, goal/hierarchy_query,
//! goal/aligned_memories, and purpose/drift_check handlers.
//!
//! TASK-CORE-001: Removed tests for deprecated methods per ARCH-03:
//! - purpose/north_star_alignment - Returns METHOD_NOT_FOUND (-32601)
//! - purpose/north_star_update - Returns METHOD_NOT_FOUND (-32601)
//!
//! Uses STUBS (InMemoryTeleologicalStore, StubMultiArrayProvider) with real GoalHierarchy.
//!
//! Tests verify:
//! - purpose/query with 13D purpose vector similarity
//! - goal/hierarchy_query operations (get_all, get_goal, get_children, get_ancestors, get_subtree)
//! - goal/aligned_memories for finding memories aligned to specific goals
//! - purpose/drift_check for detecting alignment drift
//! - Deprecated methods return METHOD_NOT_FOUND
//! - Error handling for invalid parameters
//!
//! # Module Organization
//!
//! Tests are split by handler endpoint:
//! - `helpers` - Shared helper functions for UUID-based goal tests
//! - `purpose_query` - Tests for purpose/query endpoint
//! - `north_star_alignment` - Tests for deprecated purpose/north_star_alignment
//! - `goal_hierarchy` - Tests for goal/hierarchy_query endpoint
//! - `goal_aligned_memories` - Tests for goal/aligned_memories endpoint
//! - `drift_check` - Tests for purpose/drift_check endpoint
//! - `north_star_update` - Tests for deprecated purpose/north_star_update
//! - `full_state_verification` - End-to-end purpose workflow test

mod helpers;
mod purpose_query;
mod north_star_alignment;
mod goal_hierarchy;
mod goal_aligned_memories;
mod drift_check;
mod north_star_update;
mod full_state_verification;

// Re-export helpers for use by other test modules
pub(crate) use helpers::get_goal_ids_from_hierarchy;
