//! Real implementations of bio-nervous system layers.
//!
//! These are PRODUCTION implementations that replace the stubs.
//! Each layer implements the `NervousLayer` trait with real processing logic.
//!
//! # Layers (4-Layer System)
//!
//! - [`SensingLayer`] - L1 Multi-modal input processing with PII scrubbing
//! - [`MemoryLayer`] - L3 Associative Network associative memory with decay scoring
//! - [`LearningLayer`] - L4 UTL-driven weight optimization with consolidation triggers
//! - [`CoherenceLayer`] - L5 Per-space clustering coordination and Global Workspace broadcast
//!
//! # Constitution Compliance
//!
//! Per SEC-01/SEC-02: All input is validated and sanitized via PII scrubber.
//! Per AP-007: No mock data, no fallbacks - errors fail fast.
//! Per AP-009: NaN/Infinity rejected in UTL computations.
//! Per Perf: Memory layer <1ms, Learning/Coherence <10ms.

mod coherence;
mod learning;
mod memory;
mod sensing;
mod thresholds;

pub use coherence::{
    CoherenceLayer, CoherenceState, GlobalWorkspace, GwtThresholds, INTEGRATION_STEPS,
};
// Re-export deprecated constants with warning suppression for backwards compatibility
#[allow(deprecated)]
pub use coherence::{FRAGMENTATION_THRESHOLD, GW_THRESHOLD, HYPERSYNC_THRESHOLD};
#[allow(deprecated)]
pub use learning::DEFAULT_CONSOLIDATION_THRESHOLD;
pub use learning::{
    LearningLayer, UtlWeightComputer, WeightDelta, DEFAULT_LEARNING_RATE, GRADIENT_CLIP,
    TARGET_FREQUENCY_HZ,
};
#[allow(deprecated)]
pub use memory::MIN_MEMORY_SIMILARITY;
pub use memory::{
    AssociativeMemory, MemoryContent, MemoryLayer, ScoredMemory, StoredMemory,
    DECAY_HALF_LIFE_HOURS, DEFAULT_MAX_RETRIEVE, DEFAULT_MHN_BETA, MEMORY_PATTERN_DIM,
};
pub use sensing::{PiiPattern, PiiScrubber, ScrubbedContent, SensingLayer, SensingMetrics};
pub use thresholds::LayerThresholds;
