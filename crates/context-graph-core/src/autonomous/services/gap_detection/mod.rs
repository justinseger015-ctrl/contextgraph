//! NORTH-014: Gap Detection Service
//!
//! Identifies gaps in goal coverage including uncovered domains,
//! weak coverage areas, missing links between goals, and temporal gaps.

mod config;
mod metrics;
mod report;
mod service;
#[cfg(test)]
mod tests;
mod types;

// Re-export all public items for backwards compatibility
pub use config::GapDetectionConfig;
pub use metrics::GoalWithMetrics;
pub use report::GapReport;
pub use service::GapDetectionService;
pub use types::GapType;
