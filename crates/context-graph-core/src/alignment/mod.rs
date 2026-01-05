//! Goal alignment computation module.
//!
//! Provides types and algorithms for computing alignment between
//! teleological fingerprints and goal hierarchies.
//!
//! # Overview
//!
//! The alignment module computes how well a piece of content (represented
//! as a TeleologicalFingerprint) aligns with an organization's goal hierarchy.
//!
//! # Key Types
//!
//! - [`GoalAlignmentCalculator`]: Trait for computing alignment
//! - [`DefaultAlignmentCalculator`]: Default implementation
//! - [`GoalAlignmentScore`]: Complete alignment result
//! - [`GoalScore`]: Per-goal alignment
//! - [`MisalignmentFlags`]: Detected alignment issues
//! - [`AlignmentPattern`]: Specific pattern detections
//! - [`AlignmentConfig`]: Configuration options
//! - [`AlignmentError`]: Error types
//!
//! # Usage
//!
//! ```ignore
//! use context_graph_core::alignment::{
//!     DefaultAlignmentCalculator, GoalAlignmentCalculator, AlignmentConfig
//! };
//! use context_graph_core::purpose::goals::GoalHierarchy;
//!
//! let calculator = DefaultAlignmentCalculator::new();
//! let hierarchy = GoalHierarchy::new(); // Add goals...
//! let config = AlignmentConfig::with_hierarchy(hierarchy);
//!
//! let result = calculator.compute_alignment(&fingerprint, &config).await?;
//! println!("Alignment: {:.1}%", result.score.composite_score * 100.0);
//! ```
//!
//! # Performance
//!
//! From constitution.yaml: alignment computation must complete in <5ms.
//! The default implementation meets this requirement for hierarchies
//! of up to 100 goals.
//!
//! # Thresholds (from constitution.yaml)
//!
//! - Optimal: θ ≥ 0.75
//! - Acceptable: θ ∈ [0.70, 0.75)
//! - Warning: θ ∈ [0.55, 0.70)
//! - Critical: θ < 0.55

mod calculator;
mod config;
mod error;
mod misalignment;
mod pattern;
mod score;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use calculator::{AlignmentResult, DefaultAlignmentCalculator, GoalAlignmentCalculator};
pub use config::AlignmentConfig;
pub use error::AlignmentError;
pub use misalignment::{MisalignmentFlags, MisalignmentThresholds};
pub use pattern::{AlignmentPattern, EmbedderBreakdown, PatternType};
pub use score::{GoalAlignmentScore, GoalScore, LevelWeights};
