//! NORTH-009: ThresholdLearner Service
//!
//! Adaptive threshold learning service implementing the 4-level ATC (Adaptive
//! Threshold Calibration) from constitution.yaml. Learns optimal alignment
//! thresholds based on retrieval feedback without hardcoded values.
//!
//! # 4-Level Architecture
//!
//! 1. **Level 1 - EWMA Drift Adjustment** (per-query)
//!    - Formula: `θ_ewma(t) = α × θ_observed(t) + (1 - α) × θ_ewma(t-1)`
//!    - Detects distribution drift; triggers higher levels when drift > 2σ/3σ
//!
//! 2. **Level 2 - Temperature Scaling Calibration** (hourly)
//!    - Formula: `calibrated = σ(logit(raw) / T)`
//!    - Per-embedder temperatures for confidence calibration
//!
//! 3. **Level 3 - Thompson Sampling Exploration** (session)
//!    - Samples from `Beta(α, β)` per threshold arm
//!    - Balances exploration vs exploitation with decaying violation budget
//!
//! 4. **Level 4 - Bayesian Meta-Optimization** (weekly)
//!    - Gaussian Process surrogate with Expected Improvement acquisition
//!    - Constrained optimization respecting monotonicity bounds
//!
//! # Constitution Reference
//!
//! Lines 1016-1133 define the ATC system with:
//! - Threshold priors and ranges (θ_opt, θ_acc, θ_warn, etc.)
//! - Calibration metrics (ECE < 0.05, MCE < 0.10, Brier < 0.10)
//! - Self-correction protocol (minor/moderate/major/critical)

mod learner;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_integration;
mod types;

// Re-export all public items for backwards compatibility
pub use learner::ThresholdLearner;
pub use types::{
    BayesianObservation, EmbedderLearningState, ThompsonState, ThresholdLearnerConfig,
    NUM_EMBEDDERS,
};
