# Module 5: UTL Integration - Technical Specification

```yaml
metadata:
  id: TECH-UTL-005
  version: 2.0.0
  module: UTL Integration
  phase: 4
  status: draft
  created: 2025-12-31
  dependencies:
    - TECH-CORE-002 (Module 2: Core Infrastructure)
    - TECH-EMBED-003 (Module 3: Embedding Pipeline)
    - TECH-GRAPH-004 (Module 4: Knowledge Graph)
  functional_spec_ref: SPEC-UTL-005
  author: Architecture Agent
```

---

## 1. Architecture Overview

### 1.1 Module Dependency Graph

```
                    +---------------------------+
                    |   context-graph-mcp       |
                    |      (binary crate)       |
                    +---------------------------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
+------------------------+        +---------------------------+
| context-graph-core     |        | context-graph-utl         |
|  (MemoryNode, Types)   |        |  (UTL Computation Engine) |
+------------------------+        +---------------------------+
              |                                 |
              |              +------------------+------------------+
              |              |                  |                  |
              |              v                  v                  v
              |     +----------------+  +----------------+  +----------------+
              |     | Surprise       |  | Coherence      |  | Johari         |
              |     | Calculator     |  | Tracker        |  | Classifier     |
              |     +----------------+  +----------------+  +----------------+
              |              |                  |                  |
              v              v                  v                  v
+------------------------------------------------------------------+
|                    context-graph-graph                            |
|              (Knowledge Graph for Coherence)                      |
+------------------------------------------------------------------+
```

### 1.2 Module Structure

```
context-graph/
├── crates/
│   └── context-graph-utl/
│       ├── src/
│       │   ├── lib.rs                    # Public API exports
│       │   ├── processor.rs              # UtlProcessor implementation
│       │   ├── config.rs                 # Configuration types
│       │   │
│       │   ├── surprise/
│       │   │   ├── mod.rs                # Surprise module exports
│       │   │   ├── calculator.rs         # SurpriseCalculator
│       │   │   ├── kl_divergence.rs      # KL divergence computation
│       │   │   └── embedding_distance.rs # Cosine-based surprise
│       │   │
│       │   ├── coherence/
│       │   │   ├── mod.rs                # Coherence module exports
│       │   │   ├── tracker.rs            # CoherenceTracker
│       │   │   ├── window.rs             # Rolling window buffer
│       │   │   └── structural.rs         # Graph-based coherence
│       │   │
│       │   ├── emotional/
│       │   │   ├── mod.rs                # Emotional module exports
│       │   │   ├── calculator.rs         # EmotionalWeightCalculator
│       │   │   ├── lexicon.rs            # Sentiment lexicon
│       │   │   └── state.rs              # EmotionalState with decay
│       │   │
│       │   ├── phase/
│       │   │   ├── mod.rs                # Phase module exports
│       │   │   ├── oscillator.rs         # PhaseOscillator
│       │   │   └── consolidation.rs      # ConsolidationPhase logic
│       │   │
│       │   ├── johari/
│       │   │   ├── mod.rs                # Johari module exports
│       │   │   ├── classifier.rs         # JohariClassifier
│       │   │   ├── quadrant.rs           # JohariQuadrant enum
│       │   │   └── retrieval.rs          # Quadrant-aware retrieval
│       │   │
│       │   ├── lifecycle/
│       │   │   ├── mod.rs                # Lifecycle module exports
│       │   │   ├── manager.rs            # LifecycleManager
│       │   │   ├── stage.rs              # LifecycleStage enum (Marblestone)
│       │   │   └── lambda.rs             # LifecycleLambdaWeights
│       │   │
│       │   ├── metrics.rs                # UtlMetrics struct
│       │   └── error.rs                  # Error types
│       │
│       ├── benches/
│       │   ├── utl_bench.rs              # UTL computation benchmarks
│       │   └── surprise_bench.rs         # Surprise calculation benchmarks
│       │
│       └── Cargo.toml
```

---

## 2. Core Data Structures

### 2.1 UTL Core Equation Types

```rust
// crates/context-graph-utl/src/lib.rs

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Learning magnitude computation constant
pub const LEARNING_SCALE_FACTOR: f32 = 2.0;

/// UTL Core Equation: L = f((delta_S x delta_C) . w_e . cos(phi))
///
/// This is the fundamental learning equation that governs memory acquisition.
///
/// # Components
/// - delta_S: Surprise (KL divergence-based novelty) in [0, 1]
/// - delta_C: Coherence (consistency with existing knowledge) in [0, 1]
/// - w_e: Emotional weight (valence/arousal modulation) in [0.5, 1.5]
/// - phi: Phase alignment angle in [0, PI]
///
/// # Output
/// Learning magnitude L in [0, 1]
///
/// # Performance Target
/// <100us per computation (REQ-UTL-001)
#[inline]
pub fn compute_learning_magnitude(
    delta_s: f32,
    delta_c: f32,
    w_e: f32,
    phi: f32,
) -> f32 {
    // Clamp inputs to valid ranges (prevents NaN/Infinity)
    let delta_s = delta_s.clamp(0.0, 1.0);
    let delta_c = delta_c.clamp(0.0, 1.0);
    let w_e = w_e.clamp(0.5, 1.5);
    let phi = phi.clamp(0.0, PI);

    // Core UTL computation
    let raw_signal = (delta_s * delta_c) * w_e * phi.cos();

    // Sigmoid normalization to [0, 1]
    sigmoid(raw_signal * LEARNING_SCALE_FACTOR)
}

/// Sigmoid activation for normalization
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute learning magnitude with Marblestone lambda weights
/// REQ-UTL-030 through REQ-UTL-035
///
/// # Arguments
/// - delta_s: Raw surprise component
/// - delta_c: Raw coherence component
/// - lambda_s: Lambda weight for surprise (lifecycle-dependent)
/// - lambda_c: Lambda weight for coherence (lifecycle-dependent)
/// - w_e: Emotional weight
/// - phi: Phase angle
#[inline]
pub fn compute_learning_magnitude_weighted(
    delta_s: f32,
    delta_c: f32,
    lambda_s: f32,
    lambda_c: f32,
    w_e: f32,
    phi: f32,
) -> f32 {
    // Apply Marblestone lambda weights before core computation
    let weighted_s = delta_s * lambda_s;
    let weighted_c = delta_c * lambda_c;

    compute_learning_magnitude(weighted_s, weighted_c, w_e, phi)
}
```

### 2.2 Learning Signal Types

```rust
// crates/context-graph-utl/src/lib.rs (continued)

use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Complete learning signal output from UTL computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSignal {
    /// Learning magnitude L in [0, 1]
    pub magnitude: f32,

    /// Surprise component delta_s in [0, 1]
    pub delta_s: f32,

    /// Coherence component delta_c in [0, 1]
    pub delta_c: f32,

    /// Emotional weight w_e in [0.5, 1.5]
    pub w_e: f32,

    /// Phase angle phi in [0, PI]
    pub phi: f32,

    /// Marblestone lambda weights applied
    pub lambda_weights: Option<LifecycleLambdaWeights>,

    /// Johari quadrant classification
    pub quadrant: JohariQuadrant,

    /// Suggested action based on quadrant
    pub suggested_action: SuggestedAction,

    /// Whether this should trigger consolidation
    pub should_consolidate: bool,

    /// Whether this should be stored
    pub should_store: bool,

    /// Computation timestamp
    pub timestamp: DateTime<Utc>,

    /// Computation latency in microseconds
    pub latency_us: u64,
}

impl LearningSignal {
    /// Validate the learning signal for NaN/Infinity
    pub fn validate(&self) -> Result<(), UtlError> {
        if !self.magnitude.is_finite() {
            return Err(UtlError::InvalidComputation {
                delta_s: self.delta_s,
                delta_c: self.delta_c,
                w_e: self.w_e,
                phi: self.phi,
                reason: "Learning magnitude is NaN or Infinity".to_string(),
            });
        }
        Ok(())
    }
}

/// UTL state stored on MemoryNode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtlState {
    pub delta_s: f32,
    pub delta_c: f32,
    pub w_e: f32,
    pub phi: f32,
    pub learning_magnitude: f32,
    pub quadrant: JohariQuadrant,
    pub last_computed: DateTime<Utc>,
}

/// Aggregated UTL metrics for monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UtlMetrics {
    /// Total computations performed
    pub computation_count: u64,

    /// Average learning magnitude
    pub avg_learning_magnitude: f32,

    /// Average surprise level
    pub avg_delta_s: f32,

    /// Average coherence level
    pub avg_delta_c: f32,

    /// Distribution across Johari quadrants
    pub quadrant_distribution: QuadrantDistribution,

    /// Current lifecycle stage
    pub lifecycle_stage: LifecycleStage,

    /// Current lambda weights
    pub lambda_weights: LifecycleLambdaWeights,

    /// Average computation latency in microseconds
    pub avg_latency_us: f64,

    /// P99 computation latency in microseconds
    pub p99_latency_us: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuadrantDistribution {
    pub open: u32,
    pub blind: u32,
    pub hidden: u32,
    pub unknown: u32,
}

impl QuadrantDistribution {
    pub fn total(&self) -> u32 {
        self.open + self.blind + self.hidden + self.unknown
    }

    pub fn percentages(&self) -> [f32; 4] {
        let total = self.total() as f32;
        if total == 0.0 {
            return [0.0; 4];
        }
        [
            self.open as f32 / total * 100.0,
            self.blind as f32 / total * 100.0,
            self.hidden as f32 / total * 100.0,
            self.unknown as f32 / total * 100.0,
        ]
    }
}
```

---

## 3. Lifecycle Stage (Marblestone)

### 3.1 LifecycleStage Enum

```rust
// crates/context-graph-utl/src/lifecycle/stage.rs

use serde::{Deserialize, Serialize};

/// Lifecycle stages with Marblestone-inspired dynamic learning rates
///
/// REQ-UTL-030: System SHALL adjust lambda_delta_S/lambda_delta_C based on stage
/// REQ-UTL-031: Infancy stage uses lambda_delta_S=0.7, lambda_delta_C=0.3
/// REQ-UTL-032: Growth stage uses lambda_delta_S=0.5, lambda_delta_C=0.5
/// REQ-UTL-033: Maturity stage uses lambda_delta_S=0.3, lambda_delta_C=0.7
///
/// # Design Rationale
/// Marblestone's dynamic learning principle states that:
/// - Early learning (Infancy): Prioritize novelty acquisition over coherence
/// - Active learning (Growth): Balance exploration and consolidation
/// - Stable operation (Maturity): Prioritize coherence preservation over novelty
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum LifecycleStage {
    /// Early learning phase: 0-50 interactions
    /// High novelty seeking (lambda_novelty=0.7, lambda_consolidation=0.3)
    #[default]
    Infancy = 0,

    /// Active learning phase: 50-500 interactions
    /// Balanced exploration and consolidation (lambda_novelty=0.5, lambda_consolidation=0.5)
    Growth = 1,

    /// Stable operation phase: 500+ interactions
    /// Coherence preservation (lambda_novelty=0.3, lambda_consolidation=0.7)
    Maturity = 2,
}

impl LifecycleStage {
    /// Get Marblestone lambda weights for current lifecycle stage
    ///
    /// REQ-UTL-034: Lambda weights SHALL be applied to UTL learning magnitude
    pub fn get_lambda_weights(&self) -> LifecycleLambdaWeights {
        match self {
            LifecycleStage::Infancy => LifecycleLambdaWeights {
                lambda_novelty: 0.7,
                lambda_consolidation: 0.3,
            },
            LifecycleStage::Growth => LifecycleLambdaWeights {
                lambda_novelty: 0.5,
                lambda_consolidation: 0.5,
            },
            LifecycleStage::Maturity => LifecycleLambdaWeights {
                lambda_novelty: 0.3,
                lambda_consolidation: 0.7,
            },
        }
    }

    /// Check if this stage prioritizes novelty seeking
    pub fn is_novelty_seeking(&self) -> bool {
        self.get_lambda_weights().lambda_novelty > 0.5
    }

    /// Check if this stage prioritizes coherence preservation
    pub fn is_coherence_preserving(&self) -> bool {
        self.get_lambda_weights().lambda_consolidation > 0.5
    }

    /// Get stage name for logging
    pub const fn name(&self) -> &'static str {
        match self {
            LifecycleStage::Infancy => "Infancy",
            LifecycleStage::Growth => "Growth",
            LifecycleStage::Maturity => "Maturity",
        }
    }

    /// Get entropy trigger threshold for this stage
    pub fn entropy_trigger(&self) -> f32 {
        match self {
            LifecycleStage::Infancy => 0.9,  // Capture-Heavy
            LifecycleStage::Growth => 0.7,   // Balanced
            LifecycleStage::Maturity => 0.6, // Curation-Heavy
        }
    }

    /// Get coherence trigger threshold for this stage
    pub fn coherence_trigger(&self) -> f32 {
        match self {
            LifecycleStage::Infancy => 0.2,
            LifecycleStage::Growth => 0.4,
            LifecycleStage::Maturity => 0.5,
        }
    }
}

/// Backward compatibility alias
pub type LifecyclePhase = LifecycleStage;
```

### 3.2 LifecycleLambdaWeights

```rust
// crates/context-graph-utl/src/lifecycle/lambda.rs

use serde::{Deserialize, Serialize};

/// Lambda weights for lifecycle-dependent learning (Marblestone)
///
/// These weights modulate the balance between:
/// - Novelty/surprise (delta_S) seeking
/// - Coherence/consolidation (delta_C) preservation
///
/// # Invariant
/// lambda_novelty + lambda_consolidation = 1.0
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LifecycleLambdaWeights {
    /// Weight for entropy/surprise component (lambda_delta_S)
    /// Higher values = more novelty seeking
    pub lambda_novelty: f32,

    /// Weight for coherence component (lambda_delta_C)
    /// Higher values = more coherence preservation
    pub lambda_consolidation: f32,
}

impl Default for LifecycleLambdaWeights {
    fn default() -> Self {
        // Default to balanced (Growth stage)
        Self {
            lambda_novelty: 0.5,
            lambda_consolidation: 0.5,
        }
    }
}

impl LifecycleLambdaWeights {
    /// Create new lambda weights with validation
    pub fn new(lambda_novelty: f32, lambda_consolidation: f32) -> Result<Self, UtlError> {
        let sum = lambda_novelty + lambda_consolidation;
        if (sum - 1.0).abs() > f32::EPSILON * 10.0 {
            return Err(UtlError::InvalidLambdaWeights {
                novelty: lambda_novelty,
                consolidation: lambda_consolidation,
                reason: format!("Weights must sum to 1.0, got {}", sum),
            });
        }

        Ok(Self {
            lambda_novelty: lambda_novelty.clamp(0.0, 1.0),
            lambda_consolidation: lambda_consolidation.clamp(0.0, 1.0),
        })
    }

    /// Apply weights to surprise and coherence components
    ///
    /// REQ-UTL-034: Lambda weights SHALL be applied to UTL computation
    #[inline]
    pub fn apply(&self, delta_s: f32, delta_c: f32) -> (f32, f32) {
        (
            delta_s * self.lambda_novelty,
            delta_c * self.lambda_consolidation,
        )
    }

    /// Check if weights are balanced (Growth stage)
    pub fn is_balanced(&self) -> bool {
        (self.lambda_novelty - 0.5).abs() < f32::EPSILON
    }

    /// Check if novelty-dominant (Infancy stage)
    pub fn is_novelty_dominant(&self) -> bool {
        self.lambda_novelty > self.lambda_consolidation
    }

    /// Check if consolidation-dominant (Maturity stage)
    pub fn is_consolidation_dominant(&self) -> bool {
        self.lambda_consolidation > self.lambda_novelty
    }
}
```

### 3.3 LifecycleManager

```rust
// crates/context-graph-utl/src/lifecycle/manager.rs

use crate::config::LifecycleConfig;
use crate::lifecycle::{LifecycleLambdaWeights, LifecycleStage};
use tracing::{info, debug};

/// Thresholds for lifecycle stage entry
#[derive(Debug, Clone)]
pub struct StageThresholds {
    pub entropy_trigger: f32,
    pub coherence_trigger: f32,
    pub min_importance_store: f32,
    pub consolidation_threshold: f32,
}

/// Storage stance based on lifecycle stage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageStance {
    /// Infancy: Store almost everything
    CaptureHeavy,
    /// Growth: Standard threshold
    Balanced,
    /// Maturity: High bar for storage
    CurationHeavy,
}

/// Lifecycle state machine manager
///
/// Manages transitions between Infancy, Growth, and Maturity stages
/// based on interaction count, with Marblestone lambda weight updates.
pub struct LifecycleManager {
    /// Current interaction count
    interaction_count: u64,

    /// Current lifecycle stage
    current_stage: LifecycleStage,

    /// Cached lambda weights for current stage
    lambda_weights: LifecycleLambdaWeights,

    /// Configuration
    config: LifecycleConfig,
}

impl LifecycleManager {
    /// Create new lifecycle manager
    pub fn new(config: LifecycleConfig) -> Self {
        let initial_stage = LifecycleStage::Infancy;
        Self {
            interaction_count: 0,
            current_stage: initial_stage,
            lambda_weights: initial_stage.get_lambda_weights(),
            config,
        }
    }

    /// Record an interaction and check for stage transition
    ///
    /// REQ-UTL-035: Transitions SHALL preserve accumulated knowledge coherence
    pub fn record_interaction(&mut self) -> Option<LifecycleStage> {
        self.interaction_count += 1;
        self.check_stage_transition()
    }

    /// Get current lifecycle stage
    pub fn current_stage(&self) -> LifecycleStage {
        self.current_stage
    }

    /// Get current lambda weights
    pub fn get_lambda_weights(&self) -> LifecycleLambdaWeights {
        self.lambda_weights
    }

    /// Get current interaction count
    pub fn interaction_count(&self) -> u64 {
        self.interaction_count
    }

    /// Apply lambda weights to surprise and coherence
    #[inline]
    pub fn apply_lambda_weights(&self, delta_s: f32, delta_c: f32) -> (f32, f32) {
        self.lambda_weights.apply(delta_s, delta_c)
    }

    /// Get current stage thresholds
    pub fn get_thresholds(&self) -> StageThresholds {
        match self.current_stage {
            LifecycleStage::Infancy => StageThresholds {
                entropy_trigger: self.config.infancy.entropy_trigger,
                coherence_trigger: self.config.infancy.coherence_trigger,
                min_importance_store: self.config.infancy.min_importance_store,
                consolidation_threshold: self.config.infancy.consolidation_threshold,
            },
            LifecycleStage::Growth => StageThresholds {
                entropy_trigger: self.config.growth.entropy_trigger,
                coherence_trigger: self.config.growth.coherence_trigger,
                min_importance_store: self.config.growth.min_importance_store,
                consolidation_threshold: self.config.growth.consolidation_threshold,
            },
            LifecycleStage::Maturity => StageThresholds {
                entropy_trigger: self.config.maturity.entropy_trigger,
                coherence_trigger: self.config.maturity.coherence_trigger,
                min_importance_store: self.config.maturity.min_importance_store,
                consolidation_threshold: self.config.maturity.consolidation_threshold,
            },
        }
    }

    /// Get storage stance for current stage
    pub fn storage_stance(&self) -> StorageStance {
        match self.current_stage {
            LifecycleStage::Infancy => StorageStance::CaptureHeavy,
            LifecycleStage::Growth => StorageStance::Balanced,
            LifecycleStage::Maturity => StorageStance::CurationHeavy,
        }
    }

    /// Check if content should be stored based on lifecycle stage
    pub fn should_store(&self, learning_magnitude: f32, importance: f32) -> bool {
        let thresholds = self.get_thresholds();
        match self.storage_stance() {
            StorageStance::CaptureHeavy => {
                learning_magnitude > 0.1 || importance > thresholds.min_importance_store
            }
            StorageStance::Balanced => {
                learning_magnitude > thresholds.consolidation_threshold
                    || importance > thresholds.min_importance_store
            }
            StorageStance::CurationHeavy => {
                learning_magnitude > thresholds.consolidation_threshold
                    && importance > thresholds.min_importance_store
            }
        }
    }

    /// Check for stage transition and update if needed
    fn check_stage_transition(&mut self) -> Option<LifecycleStage> {
        let new_stage = if self.interaction_count < self.config.infancy_threshold {
            LifecycleStage::Infancy
        } else if self.interaction_count < self.config.growth_threshold {
            LifecycleStage::Growth
        } else {
            LifecycleStage::Maturity
        };

        if new_stage != self.current_stage {
            let old_stage = self.current_stage;
            let old_weights = self.lambda_weights;
            let new_weights = new_stage.get_lambda_weights();

            info!(
                old_stage = %old_stage.name(),
                new_stage = %new_stage.name(),
                interactions = self.interaction_count,
                old_lambda_s = old_weights.lambda_novelty,
                old_lambda_c = old_weights.lambda_consolidation,
                new_lambda_s = new_weights.lambda_novelty,
                new_lambda_c = new_weights.lambda_consolidation,
                "Lifecycle stage transition"
            );

            self.current_stage = new_stage;
            self.lambda_weights = new_weights;
            Some(new_stage)
        } else {
            None
        }
    }

    /// Restore from persisted state
    pub fn restore(&mut self, interaction_count: u64) {
        self.interaction_count = interaction_count;
        let _ = self.check_stage_transition();
    }
}
```

---

## 4. Johari Quadrant System

### 4.1 JohariQuadrant Enum

```rust
// crates/context-graph-utl/src/johari/quadrant.rs

use serde::{Deserialize, Serialize};

/// Johari Window quadrant for memory classification
///
/// | Quadrant | delta_s | delta_c | Meaning              | Behavior           |
/// |----------|---------|---------|----------------------|--------------------|
/// | Open     | Low     | High    | Known to all         | Direct recall      |
/// | Blind    | High    | Low     | Unknown to self      | Discovery zone     |
/// | Hidden   | Low     | Low     | Known to self only   | get_neighborhood   |
/// | Unknown  | High    | High    | Unknown to all       | Epistemic action   |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum JohariQuadrant {
    /// Low entropy (<0.5), High coherence (>0.5)
    /// Known to self and others - direct recall, fast retrieval
    #[default]
    Open = 0,

    /// High entropy (>0.5), Low coherence (<0.5)
    /// Unknown to self - discovery zone, trigger exploration
    Blind = 1,

    /// Low entropy (<0.5), Low coherence (<0.5)
    /// Known to self only - private knowledge, use get_neighborhood
    Hidden = 2,

    /// High entropy (>0.5), High coherence (>0.5)
    /// Unknown to all - exploration frontier, epistemic action
    Unknown = 3,
}

impl JohariQuadrant {
    /// Get quadrant name
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Open => "Open",
            Self::Blind => "Blind",
            Self::Hidden => "Hidden",
            Self::Unknown => "Unknown",
        }
    }

    /// Check if this quadrant indicates well-understood knowledge
    pub fn is_well_understood(&self) -> bool {
        matches!(self, Self::Open)
    }

    /// Check if this quadrant requires exploration
    pub fn requires_exploration(&self) -> bool {
        matches!(self, Self::Blind | Self::Unknown)
    }
}

/// Suggested action based on Johari quadrant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestedAction {
    /// Open quadrant: Use direct retrieval
    DirectRecall,

    /// Blind quadrant: Trigger dream/consolidation layer
    TriggerDream,

    /// Hidden quadrant: Explore neighborhood graph
    GetNeighborhood,

    /// Unknown quadrant: Take epistemic action (query, explore)
    EpistemicAction,

    /// Low confidence: Request critique
    CritiqueContext,

    /// Maturity stage: Curate/prune
    Curate,
}

impl SuggestedAction {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::DirectRecall => "direct_recall",
            Self::TriggerDream => "trigger_dream",
            Self::GetNeighborhood => "get_neighborhood",
            Self::EpistemicAction => "epistemic_action",
            Self::CritiqueContext => "critique_context",
            Self::Curate => "curate",
        }
    }
}

/// Retrieval strategy configuration for each quadrant
#[derive(Debug, Clone)]
pub struct RetrievalStrategy {
    /// Search depth for graph traversal
    pub search_depth: u32,

    /// Include neighbor nodes in results
    pub include_neighbors: bool,

    /// Minimum confidence threshold
    pub confidence_threshold: f32,

    /// Maximum results to return
    pub max_results: usize,
}

impl Default for RetrievalStrategy {
    fn default() -> Self {
        Self {
            search_depth: 2,
            include_neighbors: true,
            confidence_threshold: 0.5,
            max_results: 10,
        }
    }
}
```

### 4.2 JohariClassifier

```rust
// crates/context-graph-utl/src/johari/classifier.rs

use crate::config::JohariConfig;
use crate::johari::{JohariQuadrant, RetrievalStrategy, SuggestedAction};

/// Johari quadrant classifier
///
/// Classifies memories based on entropy (surprise) and coherence levels.
pub struct JohariClassifier {
    /// Threshold for entropy classification (<threshold = low)
    entropy_threshold: f32,

    /// Threshold for coherence classification (>threshold = high)
    coherence_threshold: f32,

    /// Retrieval strategies per quadrant
    strategies: QuadrantStrategies,
}

#[derive(Debug, Clone)]
pub struct QuadrantStrategies {
    pub open: RetrievalStrategy,
    pub blind: RetrievalStrategy,
    pub hidden: RetrievalStrategy,
    pub unknown: RetrievalStrategy,
}

impl Default for QuadrantStrategies {
    fn default() -> Self {
        Self {
            open: RetrievalStrategy {
                search_depth: 1,
                include_neighbors: false,
                confidence_threshold: 0.8,
                max_results: 5,
            },
            blind: RetrievalStrategy {
                search_depth: 3,
                include_neighbors: true,
                confidence_threshold: 0.5,
                max_results: 20,
            },
            hidden: RetrievalStrategy {
                search_depth: 2,
                include_neighbors: true,
                confidence_threshold: 0.6,
                max_results: 10,
            },
            unknown: RetrievalStrategy {
                search_depth: 4,
                include_neighbors: true,
                confidence_threshold: 0.4,
                max_results: 30,
            },
        }
    }
}

impl JohariClassifier {
    /// Create new classifier from config
    pub fn new(config: &JohariConfig) -> Self {
        Self {
            entropy_threshold: config.entropy_threshold,
            coherence_threshold: config.coherence_threshold,
            strategies: QuadrantStrategies::default(),
        }
    }

    /// Classify memory into Johari quadrant
    ///
    /// | Low Entropy | High Entropy |
    /// |-------------|--------------|
    /// | Low Coh: Hidden | Low Coh: Blind |
    /// | High Coh: Open | High Coh: Unknown |
    #[inline]
    pub fn classify(&self, delta_s: f32, delta_c: f32) -> JohariQuadrant {
        let low_entropy = delta_s < self.entropy_threshold;
        let high_coherence = delta_c > self.coherence_threshold;

        match (low_entropy, high_coherence) {
            (true, true) => JohariQuadrant::Open,
            (false, false) => JohariQuadrant::Blind,
            (true, false) => JohariQuadrant::Hidden,
            (false, true) => JohariQuadrant::Unknown,
        }
    }

    /// Get suggested action for quadrant
    pub fn suggested_action(&self, quadrant: JohariQuadrant) -> SuggestedAction {
        match quadrant {
            JohariQuadrant::Open => SuggestedAction::DirectRecall,
            JohariQuadrant::Blind => SuggestedAction::TriggerDream,
            JohariQuadrant::Hidden => SuggestedAction::GetNeighborhood,
            JohariQuadrant::Unknown => SuggestedAction::EpistemicAction,
        }
    }

    /// Get retrieval strategy for quadrant
    pub fn retrieval_strategy(&self, quadrant: JohariQuadrant) -> &RetrievalStrategy {
        match quadrant {
            JohariQuadrant::Open => &self.strategies.open,
            JohariQuadrant::Blind => &self.strategies.blind,
            JohariQuadrant::Hidden => &self.strategies.hidden,
            JohariQuadrant::Unknown => &self.strategies.unknown,
        }
    }

    /// Check if quadrant transition occurred
    pub fn detect_transition(
        &self,
        old_s: f32,
        old_c: f32,
        new_s: f32,
        new_c: f32,
    ) -> Option<(JohariQuadrant, JohariQuadrant)> {
        let old_q = self.classify(old_s, old_c);
        let new_q = self.classify(new_s, new_c);
        if old_q != new_q {
            Some((old_q, new_q))
        } else {
            None
        }
    }
}
```

---

## 5. Surprise Computation

### 5.1 KL Divergence Method

```rust
// crates/context-graph-utl/src/surprise/kl_divergence.rs

use crate::config::SurpriseConfig;

/// Compute KL divergence between two probability distributions
///
/// KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
///
/// # Performance Target
/// <1ms for 1536-dimensional vectors
#[inline]
pub fn kl_divergence(p: &[f32], q: &[f32], epsilon: f32) -> f32 {
    debug_assert_eq!(p.len(), q.len(), "Distributions must have same length");

    p.iter()
        .zip(q.iter())
        .map(|(pi, qi)| {
            let qi_safe = qi.max(epsilon);
            let pi_safe = pi.max(epsilon);
            pi_safe * (pi_safe / qi_safe).ln()
        })
        .sum::<f32>()
        .max(0.0) // KL divergence is non-negative
}

/// Softmax normalization for probability distribution
#[inline]
pub fn softmax_normalize(values: &[f32], temperature: f32) -> Vec<f32> {
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = values
        .iter()
        .map(|x| ((x - max_val) / temperature).exp())
        .collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.into_iter().map(|x| x / sum).collect()
}

/// Compute surprise via KL divergence between observed and predicted
///
/// # Arguments
/// - observed: Observed embedding vector (1536D)
/// - context_embeddings: Recent context embeddings for prediction
/// - config: Surprise computation configuration
///
/// # Returns
/// Surprise score delta_s in [0, 1]
///
/// # Performance Target
/// <5ms including centroid computation
pub fn compute_surprise_kl(
    observed: &[f32],
    context_embeddings: &[&[f32]],
    config: &SurpriseConfig,
) -> f32 {
    if context_embeddings.is_empty() {
        return config.max_surprise_no_context;
    }

    if context_embeddings.len() < config.min_context_for_kl {
        // Fall back to distance-based surprise for insufficient context
        return compute_surprise_distance(observed, context_embeddings, config);
    }

    // Compute predicted distribution (centroid of context)
    let predicted = compute_centroid(context_embeddings);

    // Softmax normalization for probability distributions
    let p_obs = softmax_normalize(observed, config.kl.temperature);
    let p_pred = softmax_normalize(&predicted, config.kl.temperature);

    // KL divergence with numerical stability
    let kl_div = kl_divergence(&p_obs, &p_pred, config.kl.epsilon);

    // Normalize to [0, 1]
    (kl_div / config.kl.max_kl_value).clamp(0.0, 1.0)
}

/// Compute centroid of embedding vectors
fn compute_centroid(embeddings: &[&[f32]]) -> Vec<f32> {
    if embeddings.is_empty() {
        return vec![];
    }

    let dim = embeddings[0].len();
    let mut centroid = vec![0.0f32; dim];
    let n = embeddings.len() as f32;

    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            centroid[i] += v / n;
        }
    }

    centroid
}

/// Compute surprise via cosine distance from centroid
///
/// Lower similarity = higher surprise
pub fn compute_surprise_distance(
    observed: &[f32],
    context_embeddings: &[&[f32]],
    config: &SurpriseConfig,
) -> f32 {
    if context_embeddings.is_empty() {
        return config.max_surprise_no_context;
    }

    let centroid = compute_centroid(context_embeddings);
    let similarity = cosine_similarity(observed, &centroid);

    // Convert similarity [-1, 1] to surprise [0, 1]
    ((1.0 - similarity) / 2.0).clamp(0.0, 1.0)
}

/// Cosine similarity between two vectors
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}
```

### 5.2 SurpriseCalculator

```rust
// crates/context-graph-utl/src/surprise/calculator.rs

use crate::config::SurpriseConfig;
use crate::surprise::{compute_surprise_kl, compute_surprise_distance};

/// Surprise computation engine
pub struct SurpriseCalculator {
    config: SurpriseConfig,
}

impl SurpriseCalculator {
    pub fn new(config: &SurpriseConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Compute ensemble surprise combining KL and distance methods
    ///
    /// # Performance Target
    /// <5ms total
    pub fn compute_surprise_ensemble(
        &self,
        observed: &[f32],
        context: &[&[f32]],
    ) -> f32 {
        let kl_surprise = compute_surprise_kl(observed, context, &self.config);
        let dist_surprise = compute_surprise_distance(observed, context, &self.config);

        // Weighted combination
        let surprise = self.config.kl_weight * kl_surprise
            + self.config.distance_weight * dist_surprise;

        surprise.clamp(0.0, 1.0)
    }

    /// Compute surprise with recency decay
    pub fn compute_surprise_with_decay(
        &self,
        observed: &[f32],
        context: &[(f32, &[f32])], // (age_weight, embedding)
    ) -> f32 {
        if context.is_empty() {
            return self.config.max_surprise_no_context;
        }

        // Compute weighted centroid
        let dim = observed.len();
        let mut weighted_centroid = vec![0.0f32; dim];
        let mut total_weight = 0.0f32;

        for (weight, emb) in context {
            let decayed_weight = weight * self.config.context_decay;
            total_weight += decayed_weight;
            for (i, &v) in emb.iter().enumerate() {
                weighted_centroid[i] += v * decayed_weight;
            }
        }

        if total_weight > 0.0 {
            for v in &mut weighted_centroid {
                *v /= total_weight;
            }
        }

        let similarity = crate::surprise::cosine_similarity(observed, &weighted_centroid);
        ((1.0 - similarity) / 2.0).clamp(0.0, 1.0)
    }
}
```

---

## 6. Coherence Tracking

### 6.1 CoherenceTracker

```rust
// crates/context-graph-utl/src/coherence/tracker.rs

use crate::config::CoherenceConfig;
use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Entry in the coherence tracking window
#[derive(Debug, Clone)]
pub struct CoherenceEntry {
    pub node_id: Uuid,
    pub embedding: Vec<f32>,
    pub timestamp: DateTime<Utc>,
    pub importance: f32,
}

/// Rolling window coherence tracker
///
/// Tracks semantic and structural coherence of new information
/// relative to recent memory context.
pub struct CoherenceTracker {
    /// Circular buffer of recent node embeddings
    window: VecDeque<CoherenceEntry>,

    /// Maximum window size
    max_size: usize,

    /// Graph reference for structural coherence
    graph: Option<Arc<RwLock<KnowledgeGraph>>>,

    /// Configuration
    config: CoherenceConfig,
}

impl CoherenceTracker {
    pub fn new(config: &CoherenceConfig, graph: Option<Arc<RwLock<KnowledgeGraph>>>) -> Self {
        Self {
            window: VecDeque::with_capacity(config.window_size),
            max_size: config.window_size,
            graph,
            config: config.clone(),
        }
    }

    /// Compute coherence change for new input
    ///
    /// # Returns
    /// delta_c in [0, 1] where:
    /// - 1.0 = Perfectly coherent with context
    /// - 0.0 = Completely incoherent
    ///
    /// # Performance Target
    /// <5ms
    pub fn compute_coherence(&self, new_embedding: &[f32], _new_content: &str) -> f32 {
        let semantic_coherence = self.compute_semantic_coherence(new_embedding);
        let structural_coherence = self.compute_structural_coherence();
        let contradiction_penalty = self.compute_contradiction_penalty(new_embedding);

        // Weighted combination with penalty
        let raw_coherence = self.config.semantic_weight * semantic_coherence
            + self.config.structural_weight * structural_coherence;

        // Apply contradiction penalty
        (raw_coherence * (1.0 - contradiction_penalty)).clamp(0.0, 1.0)
    }

    /// Semantic coherence via weighted similarity to window
    fn compute_semantic_coherence(&self, embedding: &[f32]) -> f32 {
        if self.window.is_empty() {
            return self.config.default_coherence_empty;
        }

        let mut weighted_sum = 0.0f32;
        let mut weight_total = 0.0f32;

        for (i, entry) in self.window.iter().enumerate() {
            let recency_idx = self.window.len() - i;
            let decay = self.config.recency_decay.powi(recency_idx as i32);
            let weight = decay * entry.importance;

            let sim = crate::surprise::cosine_similarity(embedding, &entry.embedding);
            weighted_sum += sim * weight;
            weight_total += weight;
        }

        if weight_total > 0.0 {
            (weighted_sum / weight_total).clamp(0.0, 1.0)
        } else {
            self.config.default_coherence_empty
        }
    }

    /// Structural coherence via graph connectivity (placeholder)
    fn compute_structural_coherence(&self) -> f32 {
        // TODO: Implement full structural coherence with KnowledgeGraph
        // For now, return default value
        self.config.default_coherence_no_concepts
    }

    /// Detect contradictions with existing knowledge
    fn compute_contradiction_penalty(&self, _embedding: &[f32]) -> f32 {
        // TODO: Implement contradiction detection via graph edges
        // For now, return 0 (no penalty)
        0.0
    }

    /// Update window with new entry
    pub fn update(&mut self, entry: CoherenceEntry) {
        if self.window.len() >= self.max_size {
            self.window.pop_front();
        }
        self.window.push_back(entry);
    }

    /// Get current window size
    pub fn window_size(&self) -> usize {
        self.window.len()
    }

    /// Clear the window
    pub fn clear(&mut self) {
        self.window.clear();
    }
}

// Placeholder for KnowledgeGraph type
pub struct KnowledgeGraph;
```

---

## 7. UtlProcessor Implementation

```rust
// crates/context-graph-utl/src/processor.rs

use crate::coherence::CoherenceTracker;
use crate::config::UtlConfig;
use crate::emotional::EmotionalWeightCalculator;
use crate::johari::JohariClassifier;
use crate::lifecycle::LifecycleManager;
use crate::phase::PhaseOscillator;
use crate::surprise::SurpriseCalculator;
use crate::{
    compute_learning_magnitude_weighted, LearningSignal, UtlError, UtlMetrics, UtlState,
};
use chrono::Utc;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Session context for UTL computation
#[derive(Debug, Clone)]
pub struct SessionContext {
    pub session_id: uuid::Uuid,
    pub recent_embeddings: Vec<Vec<f32>>,
    pub interaction_count: u64,
}

/// Main UTL processor integrating all components
///
/// Orchestrates:
/// - Surprise computation (delta_s)
/// - Coherence tracking (delta_c)
/// - Emotional weighting (w_e)
/// - Phase oscillation (phi)
/// - Johari classification
/// - Lifecycle management (Marblestone)
pub struct UtlProcessor {
    /// Surprise computation
    surprise_calculator: SurpriseCalculator,

    /// Coherence tracking
    coherence_tracker: CoherenceTracker,

    /// Emotional weighting
    emotional_calculator: EmotionalWeightCalculator,

    /// Phase oscillator
    phase_oscillator: PhaseOscillator,

    /// Johari classifier
    johari_classifier: JohariClassifier,

    /// Lifecycle manager (Marblestone)
    lifecycle_manager: LifecycleManager,

    /// Configuration
    config: UtlConfig,

    /// Metrics accumulator
    metrics: Arc<RwLock<UtlMetrics>>,
}

impl UtlProcessor {
    /// Create new UTL processor
    pub fn new(config: UtlConfig) -> Self {
        Self {
            surprise_calculator: SurpriseCalculator::new(&config.surprise),
            coherence_tracker: CoherenceTracker::new(&config.coherence, None),
            emotional_calculator: EmotionalWeightCalculator::new(&config.emotional),
            phase_oscillator: PhaseOscillator::new(&config.phase),
            johari_classifier: JohariClassifier::new(&config.johari),
            lifecycle_manager: LifecycleManager::new(config.lifecycle.clone()),
            config,
            metrics: Arc::new(RwLock::new(UtlMetrics::default())),
        }
    }

    /// Compute learning signal for input
    ///
    /// # Arguments
    /// - input: Text content for emotional analysis
    /// - embedding: Pre-computed embedding (1536D)
    /// - context: Session context with recent embeddings
    ///
    /// # Returns
    /// Complete learning signal with all UTL components
    ///
    /// # Performance Target
    /// <10ms total (REQ-UTL-001)
    pub async fn compute_learning(
        &mut self,
        input: &str,
        embedding: &[f32],
        context: &SessionContext,
    ) -> Result<LearningSignal, UtlError> {
        let start = Instant::now();

        // Record interaction for lifecycle
        self.lifecycle_manager.record_interaction();

        // Get lifecycle lambda weights (Marblestone)
        let lambda_weights = self.lifecycle_manager.get_lambda_weights();
        let thresholds = self.lifecycle_manager.get_thresholds();

        // 1. Compute raw surprise (delta_s)
        let context_refs: Vec<&[f32]> = context
            .recent_embeddings
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let raw_delta_s = self.surprise_calculator.compute_surprise_ensemble(embedding, &context_refs);

        // 2. Compute raw coherence (delta_c)
        let raw_delta_c = self.coherence_tracker.compute_coherence(embedding, input);

        // 3. Apply Marblestone lambda weights
        let (weighted_delta_s, weighted_delta_c) = lambda_weights.apply(raw_delta_s, raw_delta_c);

        // 4. Compute emotional weight (w_e)
        let w_e = self.emotional_calculator.compute_weight(input);

        // 5. Get phase angle (phi)
        let phi = self.phase_oscillator.get_phi();

        // 6. Compute learning magnitude with weighted components
        let magnitude = compute_learning_magnitude_weighted(
            raw_delta_s,
            raw_delta_c,
            lambda_weights.lambda_novelty,
            lambda_weights.lambda_consolidation,
            w_e,
            phi,
        );

        // 7. Classify Johari quadrant (using raw values for classification)
        let quadrant = self.johari_classifier.classify(raw_delta_s, raw_delta_c);
        let suggested_action = self.johari_classifier.suggested_action(quadrant);

        // 8. Determine consolidation and storage
        let should_consolidate = magnitude > thresholds.consolidation_threshold;
        let should_store = self.lifecycle_manager.should_store(magnitude, raw_delta_s);

        let latency_us = start.elapsed().as_micros() as u64;

        let signal = LearningSignal {
            magnitude,
            delta_s: raw_delta_s,
            delta_c: raw_delta_c,
            w_e,
            phi,
            lambda_weights: Some(lambda_weights),
            quadrant,
            suggested_action,
            should_consolidate,
            should_store,
            timestamp: Utc::now(),
            latency_us,
        };

        // Validate result
        signal.validate()?;

        // Update metrics
        self.update_metrics(&signal).await;

        Ok(signal)
    }

    /// Get current UTL status
    pub async fn get_status(&self) -> UtlStatus {
        let metrics = self.metrics.read().await;
        UtlStatus {
            lifecycle_stage: self.lifecycle_manager.current_stage(),
            interaction_count: self.lifecycle_manager.interaction_count(),
            current_thresholds: self.lifecycle_manager.get_thresholds(),
            lambda_weights: self.lifecycle_manager.get_lambda_weights(),
            phase_angle: self.phase_oscillator.get_phi(),
            consolidation_phase: self.phase_oscillator.get_phase(),
            metrics: metrics.clone(),
        }
    }

    /// Update coherence window
    pub fn update_coherence_window(&mut self, entry: crate::coherence::CoherenceEntry) {
        self.coherence_tracker.update(entry);
    }

    /// Set phase modulation (from neuromodulation)
    pub fn set_phase_modulation(&mut self, modulation: f32) {
        self.phase_oscillator.set_modulation(modulation);
    }

    /// Force consolidation phase
    pub fn trigger_consolidation(&mut self) {
        self.phase_oscillator.force_consolidation();
    }

    /// Update metrics with new signal
    async fn update_metrics(&self, signal: &LearningSignal) {
        let mut metrics = self.metrics.write().await;

        metrics.computation_count += 1;
        let n = metrics.computation_count as f32;

        // Exponential moving average
        let alpha = 0.1;
        metrics.avg_learning_magnitude =
            alpha * signal.magnitude + (1.0 - alpha) * metrics.avg_learning_magnitude;
        metrics.avg_delta_s = alpha * signal.delta_s + (1.0 - alpha) * metrics.avg_delta_s;
        metrics.avg_delta_c = alpha * signal.delta_c + (1.0 - alpha) * metrics.avg_delta_c;

        // Update quadrant distribution
        match signal.quadrant {
            crate::JohariQuadrant::Open => metrics.quadrant_distribution.open += 1,
            crate::JohariQuadrant::Blind => metrics.quadrant_distribution.blind += 1,
            crate::JohariQuadrant::Hidden => metrics.quadrant_distribution.hidden += 1,
            crate::JohariQuadrant::Unknown => metrics.quadrant_distribution.unknown += 1,
        }

        // Update lifecycle info
        metrics.lifecycle_stage = self.lifecycle_manager.current_stage();
        metrics.lambda_weights = self.lifecycle_manager.get_lambda_weights();

        // Update latency
        metrics.avg_latency_us =
            alpha as f64 * signal.latency_us as f64 + (1.0 - alpha as f64) * metrics.avg_latency_us;
    }
}

/// UTL status report
#[derive(Debug, Clone)]
pub struct UtlStatus {
    pub lifecycle_stage: crate::LifecycleStage,
    pub interaction_count: u64,
    pub current_thresholds: crate::lifecycle::StageThresholds,
    pub lambda_weights: crate::LifecycleLambdaWeights,
    pub phase_angle: f32,
    pub consolidation_phase: crate::phase::ConsolidationPhase,
    pub metrics: UtlMetrics,
}
```

---

## 8. Phase Oscillator

```rust
// crates/context-graph-utl/src/phase/oscillator.rs

use crate::config::PhaseConfig;
use std::f32::consts::PI;
use std::time::Instant;

/// Consolidation phase of the memory system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsolidationPhase {
    /// Encoding: optimal for new memories (phi near 0)
    Encoding,
    /// Transition: between states
    Transition,
    /// Consolidation: optimal for strengthening (phi near PI)
    Consolidation,
}

/// Phase oscillator for memory consolidation timing
///
/// Inspired by biological theta rhythms, cycles between
/// encoding (phi ~ 0) and consolidation (phi ~ PI) phases.
pub struct PhaseOscillator {
    /// Current phase angle in [0, PI]
    phi: f32,

    /// Oscillation frequency (cycles per second)
    frequency: f32,

    /// Last update timestamp
    last_update: Instant,

    /// External modulation factor
    modulation: f32,

    /// Configuration
    config: PhaseConfig,
}

impl PhaseOscillator {
    pub fn new(config: &PhaseConfig) -> Self {
        Self {
            phi: 0.0,
            frequency: config.base_frequency,
            last_update: Instant::now(),
            modulation: 1.0,
            config: config.clone(),
        }
    }

    /// Get current phase angle
    ///
    /// # Returns
    /// phi in [0, PI]
    pub fn get_phi(&mut self) -> f32 {
        self.update();
        self.phi
    }

    /// Get current consolidation phase
    pub fn get_phase(&mut self) -> ConsolidationPhase {
        self.update();

        let encoding_threshold = PI * self.config.encoding_threshold;
        let consolidation_threshold = PI * self.config.consolidation_threshold;

        if self.phi < encoding_threshold {
            ConsolidationPhase::Encoding
        } else if self.phi > consolidation_threshold {
            ConsolidationPhase::Consolidation
        } else {
            ConsolidationPhase::Transition
        }
    }

    /// Update phase based on elapsed time
    fn update(&mut self) {
        let now = Instant::now();
        let elapsed = (now - self.last_update).as_secs_f32();
        self.last_update = now;

        // Phase advancement with modulation
        let delta_phi = 2.0 * PI * self.frequency * self.modulation * elapsed;

        // Oscillate between 0 and PI (half cycle)
        let raw_phase = (self.phi + delta_phi) % (2.0 * PI);
        self.phi = raw_phase.sin().abs() * PI;

        // Ensure valid range
        self.phi = self.phi.clamp(0.0, PI);
    }

    /// Apply external modulation
    pub fn set_modulation(&mut self, modulation: f32) {
        self.modulation = modulation.clamp(
            self.config.modulation_min,
            self.config.modulation_max,
        );
    }

    /// Reset to encoding phase
    pub fn reset_to_encoding(&mut self) {
        self.phi = 0.0;
    }

    /// Jump to consolidation phase
    pub fn force_consolidation(&mut self) {
        self.phi = PI * 0.9;
    }
}
```

---

## 9. Emotional Weight Calculator

```rust
// crates/context-graph-utl/src/emotional/calculator.rs

use crate::config::EmotionalConfig;
use chrono::{DateTime, Utc};

/// Emotional state for learning modulation
#[derive(Debug, Clone)]
pub struct EmotionalState {
    /// Valence: positive (1.0) to negative (-1.0)
    pub valence: f32,
    /// Arousal: calm (0.0) to excited (1.0)
    pub arousal: f32,
    /// Timestamp for decay calculation
    pub timestamp: DateTime<Utc>,
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.0,
            timestamp: Utc::now(),
        }
    }
}

/// Emotional weight calculator with decay
pub struct EmotionalWeightCalculator {
    /// Current emotional state
    current_state: EmotionalState,

    /// Configuration
    config: EmotionalConfig,
}

impl EmotionalWeightCalculator {
    pub fn new(config: &EmotionalConfig) -> Self {
        Self {
            current_state: EmotionalState::default(),
            config: config.clone(),
        }
    }

    /// Compute emotional weight w_e from content
    ///
    /// # Returns
    /// w_e in [0.5, 1.5]
    ///
    /// # Performance Target
    /// <1ms
    pub fn compute_weight(&mut self, content: &str) -> f32 {
        let (valence, arousal) = self.extract_emotion(content);
        self.update_state(valence, arousal);
        self.state_to_weight()
    }

    /// Extract valence and arousal from content
    fn extract_emotion(&self, content: &str) -> (f32, f32) {
        let valence = self.compute_lexicon_sentiment(content);
        let arousal = self.compute_arousal_heuristics(content);

        (valence.clamp(-1.0, 1.0), arousal.clamp(0.0, 1.0))
    }

    /// Simple lexicon-based sentiment
    fn compute_lexicon_sentiment(&self, content: &str) -> f32 {
        let content_lower = content.to_lowercase();

        // Simple keyword matching (extend with full lexicon)
        let positive = ["success", "excellent", "good", "great", "wonderful"];
        let negative = ["error", "failed", "bad", "danger", "critical"];

        let mut score = 0.0f32;
        for word in positive {
            if content_lower.contains(word) {
                score += 0.2;
            }
        }
        for word in negative {
            if content_lower.contains(word) {
                score -= 0.2;
            }
        }

        score.clamp(-1.0, 1.0)
    }

    /// Arousal heuristics from punctuation/formatting
    fn compute_arousal_heuristics(&self, content: &str) -> f32 {
        let exclamation_count = content.matches('!').count() as f32;
        let question_count = content.matches('?').count() as f32;
        let caps_ratio = content.chars().filter(|c| c.is_uppercase()).count() as f32
            / content.len().max(1) as f32;

        let arousal = self.config.exclamation_weight * (exclamation_count / 10.0).min(1.0)
            + self.config.question_weight * (question_count / 5.0).min(1.0)
            + self.config.caps_weight * caps_ratio;

        arousal.clamp(0.0, 1.0)
    }

    /// Update emotional state with decay
    fn update_state(&mut self, new_valence: f32, new_arousal: f32) {
        let now = Utc::now();
        let elapsed = (now - self.current_state.timestamp).num_milliseconds() as f32 / 1000.0;

        // Exponential decay
        let decay_factor = (-self.config.decay_rate * elapsed).exp();

        // Blend decayed state with new input
        self.current_state.valence = decay_factor * self.current_state.valence
            + (1.0 - decay_factor) * new_valence;
        self.current_state.arousal = decay_factor * self.current_state.arousal
            + (1.0 - decay_factor) * new_arousal;
        self.current_state.timestamp = now;

        // Clamp to valid ranges
        self.current_state.valence = self.current_state.valence.clamp(-1.0, 1.0);
        self.current_state.arousal = self.current_state.arousal.clamp(0.0, 1.0);
    }

    /// Convert emotional state to weight
    fn state_to_weight(&self) -> f32 {
        // Absolute valence matters (strong emotions boost learning)
        let valence_magnitude = self.current_state.valence.abs();

        // Combined emotional intensity
        let intensity = self.config.valence_weight * valence_magnitude
            + self.config.arousal_weight * self.current_state.arousal;

        // Map intensity [0, 1] to weight [0.5, 1.5]
        let weight = self.config.baseline_weight + intensity * self.config.intensity_scale;

        weight.clamp(0.5, 1.5)
    }

    /// Get current emotional state
    pub fn get_state(&self) -> &EmotionalState {
        &self.current_state
    }
}
```

---

## 10. Configuration Types

```rust
// crates/context-graph-utl/src/config.rs

use serde::{Deserialize, Serialize};

/// Complete UTL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtlConfig {
    /// Core equation parameters
    pub learning_scale_factor: f32,

    /// Thresholds
    pub thresholds: UtlThresholds,

    /// Salience update parameters
    pub salience_update_alpha: f32,

    /// Surprise configuration
    pub surprise: SurpriseConfig,

    /// Coherence configuration
    pub coherence: CoherenceConfig,

    /// Emotional configuration
    pub emotional: EmotionalConfig,

    /// Phase configuration
    pub phase: PhaseConfig,

    /// Johari configuration
    pub johari: JohariConfig,

    /// Lifecycle configuration (Marblestone)
    pub lifecycle: LifecycleConfig,
}

impl Default for UtlConfig {
    fn default() -> Self {
        Self {
            learning_scale_factor: 2.0,
            thresholds: UtlThresholds::default(),
            salience_update_alpha: 0.3,
            surprise: SurpriseConfig::default(),
            coherence: CoherenceConfig::default(),
            emotional: EmotionalConfig::default(),
            phase: PhaseConfig::default(),
            johari: JohariConfig::default(),
            lifecycle: LifecycleConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtlThresholds {
    pub consolidation_trigger: f32,
    pub salience_update_min: f32,
    pub surprise_significant: f32,
}

impl Default for UtlThresholds {
    fn default() -> Self {
        Self {
            consolidation_trigger: 0.7,
            salience_update_min: 0.1,
            surprise_significant: 0.6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseConfig {
    pub kl_weight: f32,
    pub distance_weight: f32,
    pub kl: KlConfig,
    pub context_window_size: usize,
    pub context_decay: f32,
    pub max_surprise_no_context: f32,
    pub min_context_for_kl: usize,
}

impl Default for SurpriseConfig {
    fn default() -> Self {
        Self {
            kl_weight: 0.6,
            distance_weight: 0.4,
            kl: KlConfig::default(),
            context_window_size: 50,
            context_decay: 0.95,
            max_surprise_no_context: 0.9,
            min_context_for_kl: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlConfig {
    pub epsilon: f32,
    pub max_kl_value: f32,
    pub temperature: f32,
}

impl Default for KlConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-10,
            max_kl_value: 10.0,
            temperature: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceConfig {
    pub window_size: usize,
    pub recency_decay: f32,
    pub semantic_weight: f32,
    pub structural_weight: f32,
    pub default_coherence_empty: f32,
    pub default_coherence_no_concepts: f32,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            recency_decay: 0.98,
            semantic_weight: 0.6,
            structural_weight: 0.4,
            default_coherence_empty: 0.5,
            default_coherence_no_concepts: 0.4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalConfig {
    pub decay_rate: f32,
    pub baseline_weight: f32,
    pub valence_weight: f32,
    pub arousal_weight: f32,
    pub intensity_scale: f32,
    pub exclamation_weight: f32,
    pub question_weight: f32,
    pub caps_weight: f32,
}

impl Default for EmotionalConfig {
    fn default() -> Self {
        Self {
            decay_rate: 0.1,
            baseline_weight: 1.0,
            valence_weight: 0.6,
            arousal_weight: 0.4,
            intensity_scale: 0.5,
            exclamation_weight: 0.3,
            question_weight: 0.2,
            caps_weight: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConfig {
    pub base_frequency: f32,
    pub modulation_min: f32,
    pub modulation_max: f32,
    pub encoding_threshold: f32,
    pub consolidation_threshold: f32,
}

impl Default for PhaseConfig {
    fn default() -> Self {
        Self {
            base_frequency: 0.1,
            modulation_min: 0.1,
            modulation_max: 3.0,
            encoding_threshold: 0.33,
            consolidation_threshold: 0.67,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JohariConfig {
    pub entropy_threshold: f32,
    pub coherence_threshold: f32,
}

impl Default for JohariConfig {
    fn default() -> Self {
        Self {
            entropy_threshold: 0.5,
            coherence_threshold: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleConfig {
    pub infancy_threshold: u64,
    pub growth_threshold: u64,
    pub infancy: StageConfig,
    pub growth: StageConfig,
    pub maturity: StageConfig,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            infancy_threshold: 50,
            growth_threshold: 500,
            infancy: StageConfig {
                entropy_trigger: 0.9,
                coherence_trigger: 0.2,
                min_importance_store: 0.1,
                consolidation_threshold: 0.3,
            },
            growth: StageConfig {
                entropy_trigger: 0.7,
                coherence_trigger: 0.4,
                min_importance_store: 0.3,
                consolidation_threshold: 0.5,
            },
            maturity: StageConfig {
                entropy_trigger: 0.6,
                coherence_trigger: 0.5,
                min_importance_store: 0.4,
                consolidation_threshold: 0.6,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageConfig {
    pub entropy_trigger: f32,
    pub coherence_trigger: f32,
    pub min_importance_store: f32,
    pub consolidation_threshold: f32,
}
```

---

## 11. Error Types

```rust
// crates/context-graph-utl/src/error.rs

use thiserror::Error;

#[derive(Debug, Error)]
pub enum UtlError {
    #[error("Invalid UTL computation: delta_s={delta_s}, delta_c={delta_c}, w_e={w_e}, phi={phi}. {reason}")]
    InvalidComputation {
        delta_s: f32,
        delta_c: f32,
        w_e: f32,
        phi: f32,
        reason: String,
    },

    #[error("Invalid lambda weights: novelty={novelty}, consolidation={consolidation}. {reason}")]
    InvalidLambdaWeights {
        novelty: f32,
        consolidation: f32,
        reason: String,
    },

    #[error("Missing context for computation")]
    MissingContext,

    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Graph access error: {0}")]
    GraphError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}
```

---

## 12. Performance Targets

| Operation | Target | P99 Target | Conditions |
|-----------|--------|------------|------------|
| `compute_learning_magnitude` | <100us | <500us | Core equation only |
| `compute_learning` (full) | <10ms | <50ms | All components |
| Surprise (KL) | <5ms | <20ms | 1536D, 50 context |
| Surprise (distance) | <1ms | <5ms | 1536D, 50 context |
| Coherence computation | <5ms | <25ms | 100 window |
| Emotional weight | <1ms | <5ms | Text analysis |
| Phase update | <10us | <50us | Simple math |
| Johari classification | <1us | <5us | Two comparisons |

---

## 13. Cargo.toml

```toml
[package]
name = "context-graph-utl"
version = "0.1.0"
edition = "2021"

[dependencies]
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1.6", features = ["v4", "serde"] }
tokio = { version = "1.35", features = ["sync", "time"] }
tracing = "0.1"

context-graph-core = { path = "../context-graph-core" }

[dev-dependencies]
criterion = "0.5"
tokio = { version = "1.35", features = ["full", "test-util"] }

[[bench]]
name = "utl_bench"
harness = false
```

---

## 14. Summary

Module 5 (UTL Integration) provides:

1. **Core UTL Equation**: L = f((delta_S x delta_C) . w_e . cos(phi))
   - Mathematically sound learning signal computation
   - NaN/Infinity protection via input clamping
   - <100us latency for core equation

2. **Marblestone Lifecycle (REQ-UTL-030-035)**:
   - `LifecycleStage`: Infancy, Growth, Maturity
   - `LifecycleLambdaWeights`: Dynamic novelty vs coherence balance
   - Auto-detection of stage transitions at 50/500 interactions

3. **Johari Quadrants**:
   - Open/Blind/Hidden/Unknown classification
   - Quadrant-aware retrieval strategies
   - Suggested actions per quadrant

4. **Component Computations**:
   - Surprise (delta_s): KL divergence + cosine distance ensemble
   - Coherence (delta_c): Semantic + structural + contradiction detection
   - Emotional weight (w_e): Lexicon sentiment + arousal heuristics
   - Phase angle (phi): Theta-inspired consolidation oscillator

5. **Performance**: <10ms total UTL computation
