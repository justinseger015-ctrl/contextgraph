//! Comprehensive integration tests for the Purpose Pattern Index.
//!
//! # Test Philosophy
//!
//! - NO mock data - all tests use real values
//! - Every test prints "[VERIFIED]" on success
//! - Tests verify fail-fast semantics
//! - Full State Verification after complex operations
//!
//! # CRITICAL: NO FALLBACKS
//!
//! All operations are fail-fast. Missing entries cause immediate errors.
//! Invalid queries rejected at construction time.
//!
//! # Test Categories
//!
//! 1. Error Tests - All PurposeIndexError variants
//! 2. Entry Tests - PurposeMetadata and PurposeIndexEntry
//! 3. Query Tests - PurposeQuery builder and validation
//! 4. Clustering Tests - K-means with real 13D vectors
//! 5. HNSW Index Tests - Insert/remove/search cycle
//! 6. Full State Verification Tests - Complete workflow

#[cfg(test)]
mod helpers;

#[cfg(test)]
mod error_tests;

#[cfg(test)]
mod entry_tests;

#[cfg(test)]
mod query_tests;

#[cfg(test)]
mod clustering_tests;

#[cfg(test)]
mod hnsw_tests;

#[cfg(test)]
mod state_verification_tests;

#[cfg(test)]
mod edge_case_tests;
