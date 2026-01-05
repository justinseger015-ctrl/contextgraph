# TASK-L003: Goal Alignment Calculator

```yaml
metadata:
  id: "TASK-L003"
  title: "Goal Alignment Calculator"
  layer: "logic"
  priority: "P0"
  estimated_hours: 6
  created: "2026-01-04"
  updated: "2026-01-05"
  status: "COMPLETED"
  completed_date: "2026-01-05"
  dependencies:
    - "TASK-L002"  # Purpose Vector Computation (COMPLETE)
    - "TASK-F002"  # TeleologicalFingerprint struct (COMPLETE)
  spec_refs:
    - "constitution.yaml:teleological.goal_hierarchy"
    - "constitution.yaml:teleological.thresholds"
```

---

## ✅ TASK STATUS: COMPLETED

This task has been fully implemented and tested. **All 108 alignment-related tests pass.**

### Implementation Location

```
crates/context-graph-core/src/alignment/
├── mod.rs              # Module exports
├── calculator.rs       # GoalAlignmentCalculator trait + DefaultAlignmentCalculator
├── config.rs           # AlignmentConfig
├── error.rs            # AlignmentError enum
├── misalignment.rs     # MisalignmentFlags, MisalignmentThresholds
├── pattern.rs          # AlignmentPattern, PatternType, EmbedderBreakdown
├── score.rs            # GoalAlignmentScore, GoalScore, LevelWeights
└── tests.rs            # 30+ comprehensive tests
```

### Module Export in lib.rs

```rust
// crates/context-graph-core/src/lib.rs line 26
pub mod alignment;
```

### Verification Commands

```bash
# Build (passes)
cargo build -p context-graph-core

# Tests (108 pass)
cargo test -p context-graph-core alignment -- --nocapture
# Output: test result: ok. 108 passed; 0 failed
```

### Performance Verified

```
PERFORMANCE RESULTS:
  - avg_ms: 0.040
  - budget_ms: 5.0
  - under_budget: true
```

---

## What Was Implemented

### 1. GoalAlignmentScore (`score.rs`)

```rust
pub struct GoalAlignmentScore {
    pub north_star_alignment: f32,
    pub strategic_alignment: f32,
    pub tactical_alignment: f32,
    pub immediate_alignment: f32,
    pub composite_score: f32,
    pub goal_breakdown: Vec<GoalScore>,
    pub embedder_breakdown: Vec<EmbedderBreakdown>,
}
```

### 2. MisalignmentFlags (`misalignment.rs`)

```rust
pub struct MisalignmentFlags {
    pub tactical_without_strategic: bool,
    pub divergent_hierarchy: bool,
    pub below_threshold: bool,
    pub inconsistent_alignment: bool,
}
```

### 3. AlignmentConfig (`config.rs`)

```rust
pub struct AlignmentConfig {
    pub hierarchy: GoalHierarchy,
    pub level_weights: LevelWeights,
    pub alignment_threshold: f32,  // Default: 0.55
    pub detect_misalignment: bool,
    pub space_weights: Option<[f32; 13]>,
    pub misalignment_thresholds: MisalignmentThresholds,
}
```

### 4. AlignmentError (`error.rs`)

```rust
pub enum AlignmentError {
    NoNorthStar,
    EmptyHierarchy,
    GoalNotFound(GoalId),
    InvalidFingerprint(String),
    HierarchyError(GoalHierarchyError),
    ComputationError(String),
}
```

### 5. GoalAlignmentCalculator Trait (`calculator.rs`)

```rust
#[async_trait]
pub trait GoalAlignmentCalculator: Send + Sync {
    async fn compute_alignment(
        &self,
        fingerprint: &TeleologicalFingerprint,
        config: &AlignmentConfig,
    ) -> Result<AlignmentResult, AlignmentError>;

    async fn compute_alignment_batch(
        &self,
        fingerprints: &[TeleologicalFingerprint],
        config: &AlignmentConfig,
    ) -> Result<Vec<AlignmentResult>, AlignmentError>;
}

pub struct DefaultAlignmentCalculator;
```

### 6. AlignmentPattern (`pattern.rs`)

```rust
pub enum AlignmentPattern {
    WellAligned,
    NearThreshold,
    Misaligned,
    Critical,
}

pub enum PatternType {
    TacticalWithoutStrategic,
    DivergentHierarchy,
    BelowThreshold,
    InconsistentAlignment,
    WellAligned,
}
```

---

## Alignment Thresholds (from constitution.yaml)

| Threshold | Value | Description |
|-----------|-------|-------------|
| Optimal | θ ≥ 0.75 | Strong alignment with North Star goal |
| Acceptable | θ ∈ [0.70, 0.75) | Acceptable, monitor for drift |
| Warning | θ ∈ [0.55, 0.70) | Alignment degrading, intervention recommended |
| Critical | θ < 0.55 | Immediate action required |

---

## Usage Example

```rust
use context_graph_core::alignment::{
    DefaultAlignmentCalculator, GoalAlignmentCalculator, AlignmentConfig
};
use context_graph_core::purpose::goals::{GoalHierarchy, GoalNode, GoalLevel};
use context_graph_core::types::fingerprint::TeleologicalFingerprint;

// Create hierarchy
let mut hierarchy = GoalHierarchy::new();
let ns = GoalNode::north_star("learn_ml", "Master ML", vec![0.5; 1024], vec![]);
hierarchy.add_goal(ns)?;

// Configure
let config = AlignmentConfig::with_hierarchy(hierarchy);

// Calculate
let calculator = DefaultAlignmentCalculator::new();
let result = calculator.compute_alignment(&fingerprint, &config).await?;

// Inspect results
println!("Composite: {:.2}", result.score.composite_score);
println!("Misaligned: {}", result.score.misalignment_flags.has_issues());
```

---

## Dependencies Verified

| Dependency | Status | Location |
|------------|--------|----------|
| `PurposeVector` | ✅ | `crates/context-graph-core/src/types/fingerprint/purpose.rs` |
| `TeleologicalFingerprint` | ✅ | `crates/context-graph-core/src/types/fingerprint/teleological/` |
| `GoalHierarchy` | ✅ | `crates/context-graph-core/src/purpose/goals.rs` |
| `GoalNode` | ✅ | `crates/context-graph-core/src/purpose/goals.rs` |
| `GoalLevel` | ✅ | `crates/context-graph-core/src/purpose/goals.rs` |
| `GoalId` | ✅ | `crates/context-graph-core/src/purpose/goals.rs` |

---

## Test Coverage

The implementation includes tests for:

- ✅ Alignment threshold classification
- ✅ Config validation
- ✅ Batch processing with real data
- ✅ Divergent hierarchy detection
- ✅ Critical misalignment detection
- ✅ Embedder breakdown statistics
- ✅ Goal alignment score composite computation
- ✅ Empty hierarchy error
- ✅ Full alignment computation with real data
- ✅ Level weights invariant
- ✅ Goal score weighted contribution
- ✅ Misalignment flags severity levels
- ✅ Pattern type classification
- ✅ Tactical without strategic pattern
- ✅ Performance constraint <5ms
- ✅ Edge case: Zero purpose vector
- ✅ Edge case: Maximum alignment
- ✅ Edge case: Negative alignment

---

*Task completed: 2026-01-05*
*Layer: Logic*
*Status: ✅ COMPLETED*
