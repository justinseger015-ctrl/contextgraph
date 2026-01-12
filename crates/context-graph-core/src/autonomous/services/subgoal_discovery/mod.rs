//! NORTH-015: Sub-Goal Discovery Service
//!
//! Discovers emergent sub-goals from memory clusters. This service analyzes
//! clusters of related memories to identify patterns that suggest new goals
//! should be added to the goal hierarchy.
//!
//! # Architecture
//!
//! The discovery process:
//! 1. Analyze memory clusters for coherent themes
//! 2. Extract candidate sub-goals with confidence scores
//! 3. Find appropriate parent goals in the hierarchy
//! 4. Determine goal level based on evidence
//! 5. Rank and filter candidates for promotion
//!
//! # Module Organization
//!
//! - [`cluster`]: Memory cluster types
//! - [`config`]: Configuration for discovery
//! - [`result`]: Discovery result types
//! - [`service`]: Main discovery service implementation

pub mod cluster;
pub mod config;
pub mod result;
pub mod service;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use cluster::MemoryCluster;
pub use config::DiscoveryConfig;
pub use result::DiscoveryResult;
pub use service::SubGoalDiscovery;
