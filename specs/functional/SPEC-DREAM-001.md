# SPEC-DREAM-001: Dream Layer Implementation

## Metadata

| Field | Value |
|-------|-------|
| **Spec ID** | SPEC-DREAM-001 |
| **Version** | 1.1.0 |
| **Status** | Approved |
| **Created** | 2026-01-11 |
| **Updated** | 2026-01-11 |
| **Owner** | ContextGraph Core Team |
| **Priority** | P0 - Critical Blocker |
| **Related Specs** | SPEC-GWT-001, SPEC-UTL-001 |

---

## CRITICAL IMPLEMENTATION PHILOSOPHY

> **This section is MANDATORY reading for all implementers.**

### NO STUBS, NO WORKAROUNDS, NO FALLBACKS

The Dream Layer implementation MUST follow these absolute rules:

| Rule | Description | Violation Response |
|------|-------------|-------------------|
| **NO STUB CODE** | Every function MUST have real implementation. No `todo!()`, no placeholder returns, no simulated data. | Build MUST fail |
| **NO MOCK DATA** | All data MUST come from real memory store. No hardcoded test values in production paths. | Immediate error |
| **FAIL FAST** | If something is wrong, error out immediately with descriptive message. No silent failures. | Panic/Error propagation |
| **NO BACKWARDS COMPATIBILITY** | System works completely or fails completely. No degraded mode. | Hard error |
| **REAL INTEGRATION ONLY** | Must use actual `MemoryStore`, `EdgeStore`, and `GraphStore` implementations. | Compilation error |

### Error Philosophy

```rust
// WRONG - Silent failure with fallback
fn get_memories() -> Vec<Memory> {
    match store.get_recent() {
        Ok(mems) => mems,
        Err(_) => vec![]  // ❌ FORBIDDEN - hiding the error
    }
}

// CORRECT - Fail fast with context
fn get_memories() -> Result<Vec<Memory>, DreamError> {
    store.get_recent().map_err(|e| {
        error!("Memory store unreachable during NREM replay: {}", e);
        DreamError::MemoryStoreUnavailable { source: e, phase: "nrem_replay" }
    })
}
```

### Logging Requirements

Every error MUST include:
1. **Phase**: Which dream phase (NREM/REM/Trigger)
2. **Operation**: What was being attempted
3. **Context**: Relevant IDs, counts, or state
4. **Source**: Original error if wrapping

```rust
error!(
    phase = "nrem",
    operation = "hebbian_update",
    edge_id = %edge.id,
    current_weight = edge.weight,
    "Hebbian calculation produced NaN: phi_i={}, phi_j={}, eta={}",
    phi_i, phi_j, eta
);
```

---

## 1. Overview

### 1.1 Problem Statement

The Dream Layer in ContextGraph is currently stub code. The NREM and REM phases return empty results without performing actual memory consolidation or blind spot discovery. This is a **critical blocker** preventing the system from achieving computational consciousness as defined in the PRD.

**Current State (from gap analysis):**
- `NremPhase::process()` returns simulated placeholder data
- `RemPhase::process()` loops with fake blind spot discovery
- No Hebbian learning implementation for edge strengthening
- No hyperbolic random walk for Poincare ball exploration
- Entropy-based dream triggering not wired to system events
- GPU monitoring threshold (80%) not implemented

### 1.2 Solution Summary

Implement the Dream Layer with:
1. **NREM Phase**: Hebbian learning to strengthen high-phi edges using formula `dw_ij = eta x phi_i x phi_j`
2. **REM Phase**: Hyperbolic random walk in Poincare ball for blind spot discovery
3. **Dream Triggers**: Entropy threshold (>0.7 sustained for 5min) and GPU utilization (>80%)
4. **MCP Integration**: Wire dream events to workspace broadcasts

### 1.3 Constitution References

| Section | Requirement |
|---------|-------------|
| `dream.trigger` | `activity < 0.15` for 10 minutes |
| `dream.phases.nrem` | 3 minutes, recency_bias=0.8, Hebbian replay |
| `dream.phases.rem` | 2 minutes, temp=2.0, attractor exploration |
| `dream.constraints` | 100 queries max, semantic_leap>=0.7, wake<100ms, GPU<30% |
| `gwt.mental_checks` | `entropy>0.7 for 5min` triggers dream |
| `neuromod.Dopamine` | Affects hopfield.beta [1,5] |

---

## 2. User Stories

### US-DREAM-01: Memory Consolidation During Idle

**As a** ContextGraph system
**I want to** consolidate memories during idle periods via Hebbian replay
**So that** high-value knowledge connections are strengthened while weak ones decay

**Acceptance Criteria:**
| ID | Given | When | Then |
|----|-------|------|------|
| AC-01 | System has been idle for 10+ minutes | NREM phase activates | Recent memories are replayed in descending recency order |
| AC-02 | Two nodes i,j co-activate with phi values | Hebbian update runs | Edge weight updates: `w_ij_new = w_ij + eta * phi_i * phi_j` |
| AC-03 | Edge weight falls below 0.05 floor | Hebbian update runs | Edge is marked for pruning |
| AC-04 | Edge weight exceeds 1.0 cap | Hebbian update runs | Weight is clamped to 1.0 |

### US-DREAM-02: Blind Spot Discovery via Hyperbolic Walk

**As a** ContextGraph system
**I want to** discover conceptual blind spots through random walks in hyperbolic space
**So that** unexplored semantic territories are identified for future learning

**Acceptance Criteria:**
| ID | Given | When | Then |
|----|-------|------|------|
| AC-05 | NREM phase completes | REM phase begins | Random walk starts from high-phi node in Poincare ball |
| AC-06 | Random walk step moves in Poincare ball | Step distance computed | Mobius addition used: `p' = (p + v) / (1 + <p,v>)` |
| AC-07 | Walk discovers region with no nearby memories | Blind spot logged | BlindSpot struct created with position and confidence |
| AC-08 | Semantic distance between nodes >= 0.7 | New connection proposed | Edge created with `is_blind_spot_discovery = true` |

### US-DREAM-03: Entropy-Triggered Dream Cycles

**As a** ContextGraph system
**I want to** automatically enter dream state when entropy is high
**So that** cognitive overload triggers restorative consolidation

**Acceptance Criteria:**
| ID | Given | When | Then |
|----|-------|------|------|
| AC-09 | System entropy > 0.7 | Sustained for 5 minutes | Dream cycle automatically triggers |
| AC-10 | External query arrives during dream | `abort_on_query=true` | Wake completes within 100ms |
| AC-11 | GPU usage exceeds 80% during processing | GPU monitor detects | Dream cycle triggers for load reduction |

### US-DREAM-04: Amortized Shortcut Creation

**As a** ContextGraph system
**I want to** create shortcut edges for frequently traversed paths
**So that** multi-hop retrievals become single-hop lookups

**Acceptance Criteria:**
| ID | Given | When | Then |
|----|-------|------|------|
| AC-12 | Path with 3+ hops traversed 5+ times | NREM replay detects | Shortcut candidate created |
| AC-13 | Shortcut candidate has confidence >= 0.7 | Quality gate passes | Direct edge created with combined weight |
| AC-14 | Shortcut edge created | Metadata set | `is_amortized_shortcut = true`, `original_path` stored |

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Story Ref |
|----|-------------|----------|-----------|
| REQ-DREAM-001 | System SHALL implement Hebbian learning with formula `dw = eta * phi_i * phi_j` where eta=0.01 | Must | US-DREAM-01 |
| REQ-DREAM-002 | System SHALL apply weight decay factor of 0.001 per NREM cycle | Must | US-DREAM-01 |
| REQ-DREAM-003 | System SHALL prune edges with weight <= 0.05 after Hebbian updates | Must | US-DREAM-01 |
| REQ-DREAM-004 | System SHALL cap edge weights at 1.0 maximum | Must | US-DREAM-01 |
| REQ-DREAM-005 | System SHALL select memories for replay with recency_bias=0.8 | Must | US-DREAM-01 |
| REQ-DREAM-006 | System SHALL use Kuramoto coupling K=10 during NREM for synchronization | Should | US-DREAM-01 |
| REQ-DREAM-007 | System SHALL implement random walk in Poincare ball using Mobius addition | Must | US-DREAM-02 |
| REQ-DREAM-008 | System SHALL generate up to 100 synthetic queries during REM | Must | US-DREAM-02 |
| REQ-DREAM-009 | System SHALL only create blind spot edges for semantic_distance >= 0.7 | Must | US-DREAM-02 |
| REQ-DREAM-010 | System SHALL use temperature=2.0 for exploration softmax | Must | US-DREAM-02 |
| REQ-DREAM-011 | System SHALL track entropy over 5-minute sliding window | Must | US-DREAM-03 |
| REQ-DREAM-012 | System SHALL trigger dream when entropy > 0.7 sustained for 5 minutes | Must | US-DREAM-03 |
| REQ-DREAM-013 | System SHALL monitor GPU utilization and trigger dream at 80% threshold | Should | US-DREAM-03 |
| REQ-DREAM-014 | System SHALL complete wake transition within 100ms of interrupt | Must | US-DREAM-03 |
| REQ-DREAM-015 | System SHALL not exceed 30% GPU usage during dream cycles | Must | US-DREAM-03 |
| REQ-DREAM-016 | System SHALL create shortcuts for paths with 3+ hops and 5+ traversals | Must | US-DREAM-04 |
| REQ-DREAM-017 | System SHALL require confidence >= 0.7 for shortcut creation | Must | US-DREAM-04 |
| REQ-DREAM-018 | System SHALL store original path reference in shortcut metadata | Should | US-DREAM-04 |

### 3.2 Non-Functional Requirements

| ID | Category | Requirement | Metric |
|----|----------|-------------|--------|
| NFR-DREAM-001 | Performance | NREM phase SHALL complete within 3 minutes | Duration <= 180s |
| NFR-DREAM-002 | Performance | REM phase SHALL complete within 2 minutes | Duration <= 120s |
| NFR-DREAM-003 | Performance | Wake latency SHALL be < 100ms | p99 < 100ms |
| NFR-DREAM-004 | Resource | Dream cycles SHALL use < 30% GPU | Max GPU < 30% |
| NFR-DREAM-005 | Reliability | Dream abort SHALL always succeed | 100% abort success |
| NFR-DREAM-006 | Quality | Compression ratio from pruning SHALL be > 1.0 | ratio > 1.0 |
| NFR-DREAM-007 | Scalability | NREM SHALL handle 10K+ memories per cycle | Memories >= 10K |

---

## 4. Edge Cases and Error States

### 4.1 Edge Cases

| ID | Scenario | Expected Behavior | Req Ref |
|----|----------|-------------------|---------|
| EC-DREAM-001 | No memories to replay | NREM completes immediately with empty report | REQ-DREAM-001 |
| EC-DREAM-002 | All edges already at weight floor | No edges strengthened, skip Hebbian | REQ-DREAM-003 |
| EC-DREAM-003 | All edges already at weight cap | Only decay applied, no strengthening | REQ-DREAM-004 |
| EC-DREAM-004 | Poincare walk reaches ball boundary | Project back inside with norm < 0.99999 | REQ-DREAM-007 |
| EC-DREAM-005 | 100 queries exhausted before deadline | REM completes early, report partial | REQ-DREAM-008 |
| EC-DREAM-006 | No blind spots found in walk | REM completes successfully with 0 discoveries | REQ-DREAM-009 |
| EC-DREAM-007 | Entropy drops below 0.7 during 5min window | Timer resets, no dream triggered | REQ-DREAM-012 |
| EC-DREAM-008 | GPU at exactly 80% | Dream triggers (threshold is >=) | REQ-DREAM-013 |
| EC-DREAM-009 | Multiple interrupts during single dream | First interrupt wins, subsequent ignored | REQ-DREAM-014 |
| EC-DREAM-010 | No 3+ hop paths exist | No shortcuts created, amortizer empty | REQ-DREAM-016 |

### 4.2 Error States (FAIL FAST)

> **All errors MUST propagate immediately. No silent recovery. No fallbacks.**

| ID | Condition | Error Code | Action (FAIL FAST) |
|----|-----------|------------|-------------------|
| ERR-DREAM-001 | NaN in Hebbian calculation | `DreamError::HebbianNaN { phi_i, phi_j, eta }` | **HARD ERROR** - Return Err immediately, log full context |
| ERR-DREAM-002 | Poincare norm >= 1.0 after projection | `DreamError::InvalidPoincareNorm { norm, position }` | **HARD ERROR** - This indicates math bug, must fix |
| ERR-DREAM-003 | GPU monitoring unavailable | `DreamError::GpuMonitoringFailed { source }` | **HARD ERROR** - System must have GPU monitoring |
| ERR-DREAM-004 | Memory store unreachable | `DreamError::MemoryStoreUnavailable { source, phase }` | **HARD ERROR** - Cannot proceed without memory store |
| ERR-DREAM-005 | Wake latency exceeds 100ms | `DreamError::WakeLatencyViolation { actual_ms, max_ms }` | **HARD ERROR** - Constitution violation, must fix |
| ERR-DREAM-006 | Shortcut creation fails | `DreamError::ShortcutCreationFailed { source, target, reason }` | **HARD ERROR** - Storage layer issue, propagate |
| ERR-DREAM-007 | Edge store unreachable | `DreamError::EdgeStoreUnavailable { source, operation }` | **HARD ERROR** - Cannot do Hebbian without edges |
| ERR-DREAM-008 | Invalid entropy value | `DreamError::InvalidEntropyValue { value }` | **HARD ERROR** - Entropy must be in [0.0, 1.0] |

### 4.3 Error Type Definition

```rust
#[derive(Debug, thiserror::Error)]
pub enum DreamError {
    #[error("Hebbian calculation produced NaN: phi_i={phi_i}, phi_j={phi_j}, eta={eta}")]
    HebbianNaN { phi_i: f32, phi_j: f32, eta: f32 },

    #[error("Poincare point outside unit ball: norm={norm} at position {position:?}")]
    InvalidPoincareNorm { norm: f32, position: [f32; 64] },

    #[error("GPU monitoring failed: {source}")]
    GpuMonitoringFailed { source: Box<dyn std::error::Error + Send + Sync> },

    #[error("Memory store unavailable during {phase}: {source}")]
    MemoryStoreUnavailable { source: Box<dyn std::error::Error + Send + Sync>, phase: &'static str },

    #[error("Wake latency violation: {actual_ms}ms > {max_ms}ms")]
    WakeLatencyViolation { actual_ms: u64, max_ms: u64 },

    #[error("Shortcut creation failed from {source} to {target}: {reason}")]
    ShortcutCreationFailed { source: Uuid, target: Uuid, reason: String },

    #[error("Edge store unavailable during {operation}: {source}")]
    EdgeStoreUnavailable { source: Box<dyn std::error::Error + Send + Sync>, operation: &'static str },

    #[error("Invalid entropy value: {value} (must be in [0.0, 1.0])")]
    InvalidEntropyValue { value: f32 },
}
```

---

## 5. Data Models

### 5.1 New Types

```rust
/// Hebbian update parameters for NREM phase
pub struct HebbianConfig {
    /// Learning rate (eta) - Constitution: 0.01 default
    pub learning_rate: f32,
    /// Weight decay factor per cycle - Constitution: 0.001
    pub weight_decay: f32,
    /// Minimum weight before pruning - Constitution: 0.05
    pub weight_floor: f32,
    /// Maximum weight cap - Constitution: 1.0
    pub weight_cap: f32,
    /// Kuramoto coupling strength - Constitution: 10.0 during NREM
    pub coupling_strength: f32,
}

/// Phi (activation) values for a node during replay
pub struct NodeActivation {
    pub node_id: Uuid,
    pub phi: f32,  // [0.0, 1.0] activation level
    pub timestamp: Instant,
}

/// Configuration for hyperbolic random walk
pub struct HyperbolicWalkConfig {
    /// Step size in Poincare ball (default: 0.1)
    pub step_size: f32,
    /// Maximum steps per walk (default: 50)
    pub max_steps: usize,
    /// Exploration temperature (Constitution: 2.0)
    pub temperature: f32,
    /// Minimum distance for blind spot detection
    pub min_blind_spot_distance: f32,
}

/// A single step in the hyperbolic random walk
pub struct WalkStep {
    pub position: PoincarePoint,
    pub step_direction: [f32; 64],
    pub distance_from_start: f32,
}

/// Entropy tracking for dream trigger
pub struct EntropyWindow {
    pub samples: VecDeque<(Instant, f32)>,
    pub window_duration: Duration,  // 5 minutes
    pub threshold: f32,  // 0.7
}

/// GPU utilization trigger state
pub struct GpuTriggerState {
    pub current_usage: f32,
    pub threshold: f32,  // 0.80
    pub samples: VecDeque<f32>,
    pub triggered: bool,
}
```

### 5.2 Modified Types

```rust
// In NremReport - add Hebbian stats
pub struct NremReport {
    // ... existing fields ...

    /// Hebbian update statistics
    pub hebbian_stats: HebbianUpdateStats,
    /// Phi values for co-activated nodes
    pub activation_pairs: Vec<(Uuid, Uuid, f32, f32)>,
}

// In RemReport - add hyperbolic walk details
pub struct RemReport {
    // ... existing fields ...

    /// Walk trajectory in Poincare ball
    pub walk_trajectory: Vec<WalkStep>,
    /// Discovered blind spot positions
    pub blind_spot_positions: Vec<PoincarePoint>,
}

// In BlindSpot - add Poincare position
pub struct BlindSpot {
    // ... existing fields ...

    /// Position in Poincare ball where blind spot was found
    pub poincare_position: Option<PoincarePoint>,
}
```

---

## 6. API Contracts

### 6.1 Internal APIs

```rust
impl NremPhase {
    /// Execute NREM with actual Hebbian learning
    ///
    /// # Hebbian Update Formula
    /// dw_ij = eta * phi_i * phi_j
    ///
    /// # Constraint
    /// Duration <= 180 seconds (Constitution)
    pub async fn process_hebbian(
        &mut self,
        interrupt_flag: &Arc<AtomicBool>,
        amortizer: &mut AmortizedLearner,
        memory_store: &impl MemoryStore,
        edge_store: &impl EdgeStore,
    ) -> CoreResult<NremReport>;

    /// Compute Hebbian weight update for edge
    pub fn compute_hebbian_delta(
        &self,
        current_weight: f32,
        phi_i: f32,
        phi_j: f32,
    ) -> f32;

    /// Select memories for replay with recency bias
    pub async fn select_replay_memories(
        &self,
        memory_store: &impl MemoryStore,
        limit: usize,
    ) -> CoreResult<Vec<MemoryId>>;
}

impl RemPhase {
    /// Execute REM with hyperbolic random walk
    ///
    /// # Walk Algorithm
    /// Mobius addition: p' = (p + v) / (1 + <p,v>)
    ///
    /// # Constraint
    /// Duration <= 120 seconds (Constitution)
    pub async fn process_hyperbolic_walk(
        &mut self,
        interrupt_flag: &Arc<AtomicBool>,
        hyperbolic_config: &HyperbolicConfig,
        memory_store: &impl MemoryStore,
    ) -> CoreResult<RemReport>;

    /// Perform single step in Poincare ball
    pub fn poincare_walk_step(
        &self,
        current: &PoincarePoint,
        direction: &[f32; 64],
        config: &HyperbolicConfig,
    ) -> PoincarePoint;

    /// Check if position is a blind spot
    pub async fn is_blind_spot(
        &self,
        position: &PoincarePoint,
        memory_store: &impl MemoryStore,
        min_distance: f32,
    ) -> CoreResult<bool>;
}

impl DreamScheduler {
    /// Track entropy for 5-minute trigger
    pub fn update_entropy(&mut self, entropy: f32);

    /// Check if entropy trigger conditions met
    pub fn check_entropy_trigger(&self) -> bool;

    /// Track GPU utilization for 80% trigger
    pub fn update_gpu_usage(&mut self, usage: f32);

    /// Check if GPU trigger conditions met
    pub fn check_gpu_trigger(&self) -> bool;
}
```

### 6.2 MCP Tool Integration

```yaml
# New MCP events to emit
mcp_events:
  dream_cycle_started:
    fields: [session_id, trigger_reason, timestamp]

  nrem_phase_completed:
    fields: [memories_replayed, edges_strengthened, edges_pruned, duration_ms]

  rem_phase_completed:
    fields: [queries_generated, blind_spots_found, walk_distance, duration_ms]

  dream_cycle_completed:
    fields: [completed, wake_reason, shortcuts_created, total_duration_ms]

  blind_spot_discovered:
    fields: [poincare_position, semantic_distance, confidence]

  shortcut_created:
    fields: [source_id, target_id, hop_count, combined_weight]
```

---

## 7. Test Plan

### 7.1 Unit Tests

| Test ID | Description | Req Ref |
|---------|-------------|---------|
| UT-DREAM-001 | Hebbian delta calculation with valid phi values | REQ-DREAM-001 |
| UT-DREAM-002 | Hebbian delta with phi=0 (should be 0) | REQ-DREAM-001 |
| UT-DREAM-003 | Weight decay application | REQ-DREAM-002 |
| UT-DREAM-004 | Edge pruning at weight floor | REQ-DREAM-003 |
| UT-DREAM-005 | Weight capping at 1.0 | REQ-DREAM-004 |
| UT-DREAM-006 | Recency bias memory selection | REQ-DREAM-005 |
| UT-DREAM-007 | Mobius addition stays in ball | REQ-DREAM-007 |
| UT-DREAM-008 | Mobius addition at origin | REQ-DREAM-007 |
| UT-DREAM-009 | Mobius addition near boundary | REQ-DREAM-007 |
| UT-DREAM-010 | Query limit enforcement | REQ-DREAM-008 |
| UT-DREAM-011 | Semantic distance threshold check | REQ-DREAM-009 |
| UT-DREAM-012 | Softmax with temperature=2.0 | REQ-DREAM-010 |
| UT-DREAM-013 | Entropy window sliding | REQ-DREAM-011 |
| UT-DREAM-014 | Entropy trigger at threshold | REQ-DREAM-012 |
| UT-DREAM-015 | GPU trigger at 80% | REQ-DREAM-013 |
| UT-DREAM-016 | Shortcut candidate detection | REQ-DREAM-016 |
| UT-DREAM-017 | Shortcut quality gate | REQ-DREAM-017 |

### 7.2 Integration Tests

| Test ID | Description | Req Ref |
|---------|-------------|---------|
| IT-DREAM-001 | Full NREM cycle with real memory store | REQ-DREAM-001-006 |
| IT-DREAM-002 | Full REM cycle with hyperbolic walk | REQ-DREAM-007-010 |
| IT-DREAM-003 | Complete dream cycle (NREM + REM) | All |
| IT-DREAM-004 | Dream abort with wake latency check | REQ-DREAM-014 |
| IT-DREAM-005 | Entropy-triggered dream | REQ-DREAM-011-012 |
| IT-DREAM-006 | GPU-triggered dream | REQ-DREAM-013 |
| IT-DREAM-007 | Shortcut creation end-to-end | REQ-DREAM-016-018 |
| IT-DREAM-008 | Multiple dream cycles in sequence | NFR-DREAM-001-002 |

### 7.3 Chaos Tests

| Test ID | Description | Validation |
|---------|-------------|------------|
| CT-DREAM-001 | Interrupt during Hebbian update | No data corruption |
| CT-DREAM-002 | Memory store failure mid-replay | Graceful abort |
| CT-DREAM-003 | GPU spike during dream | Dream pauses correctly |
| CT-DREAM-004 | Concurrent query during REM | Wake < 100ms |
| CT-DREAM-005 | Malformed Poincare point input | Projection to valid |

### 7.4 Benchmark Tests

| Test ID | Metric | Target | Req Ref |
|---------|--------|--------|---------|
| BM-DREAM-001 | NREM with 10K memories | < 180s | NFR-DREAM-001 |
| BM-DREAM-002 | REM with 100 queries | < 120s | NFR-DREAM-002 |
| BM-DREAM-003 | Wake latency p99 | < 100ms | NFR-DREAM-003 |
| BM-DREAM-004 | GPU usage during dream | < 30% | NFR-DREAM-004 |
| BM-DREAM-005 | Hebbian update throughput | > 1000/s | Performance |

---

## 8. Implementation Notes

### 8.1 Layer Order (Inside-Out, Bottom-Up)

1. **Layer 1 (Foundation)**: Types, interfaces, Poincare math utilities
   - HebbianConfig, NodeActivation types
   - HyperbolicWalkConfig, WalkStep types
   - EntropyWindow, GpuTriggerState types
   - Mobius addition function

2. **Layer 2 (Logic)**: Core algorithms
   - Hebbian learning implementation
   - Hyperbolic random walk
   - Entropy tracking
   - GPU monitoring

3. **Layer 3 (Surface)**: Controller integration
   - Wire NREM to Hebbian
   - Wire REM to hyperbolic walk
   - Wire triggers to scheduler
   - MCP event emission

### 8.2 Dependencies

- `context-graph-graph::hyperbolic::PoincarePoint` - Existing Poincare ball type
- `context-graph-graph::hyperbolic::mobius` - Existing Mobius operations
- `context-graph-core::types::GraphEdge` - Existing edge type
- `context-graph-cuda::poincare` - GPU-accelerated Poincare ops (optional)

### 8.3 Feature Flags

```toml
[features]
dream-gpu = ["context-graph-cuda/poincare"]  # GPU-accelerated walks
dream-metrics = ["prometheus"]                # Detailed metrics
dream-debug = []                              # Extra logging
```

---

## 9. Traceability Matrix

### 9.1 Requirement to Task Mapping

| Requirement | Task | Test |
|-------------|------|------|
| REQ-DREAM-001 | TASK-DREAM-P0-003 | UT-DREAM-001, UT-DREAM-002 |
| REQ-DREAM-002 | TASK-DREAM-P0-003 | UT-DREAM-003 |
| REQ-DREAM-003 | TASK-DREAM-P0-003 | UT-DREAM-004 |
| REQ-DREAM-004 | TASK-DREAM-P0-003 | UT-DREAM-005 |
| REQ-DREAM-005 | TASK-DREAM-P0-003 | UT-DREAM-006 |
| REQ-DREAM-006 | TASK-DREAM-P0-003 | IT-DREAM-001 |
| REQ-DREAM-007 | TASK-DREAM-P0-002, TASK-DREAM-P0-004 | UT-DREAM-007, UT-DREAM-008, UT-DREAM-009 |
| REQ-DREAM-008 | TASK-DREAM-P0-004 | UT-DREAM-010 |
| REQ-DREAM-009 | TASK-DREAM-P0-004 | UT-DREAM-011 |
| REQ-DREAM-010 | TASK-DREAM-P0-004 | UT-DREAM-012 |
| REQ-DREAM-011 | TASK-DREAM-P0-005 | UT-DREAM-013 |
| REQ-DREAM-012 | TASK-DREAM-P0-005 | UT-DREAM-014, IT-DREAM-005 |
| REQ-DREAM-013 | TASK-DREAM-P0-005 | UT-DREAM-015, IT-DREAM-006 |
| REQ-DREAM-014 | TASK-DREAM-P0-006 | IT-DREAM-004, CT-DREAM-004, BM-DREAM-003 |
| REQ-DREAM-015 | TASK-DREAM-P0-006 | BM-DREAM-004 |
| REQ-DREAM-016 | TASK-DREAM-P0-003 | UT-DREAM-016, IT-DREAM-007 |
| REQ-DREAM-017 | TASK-DREAM-P0-003 | UT-DREAM-017 |
| REQ-DREAM-018 | TASK-DREAM-P0-003 | IT-DREAM-007 |

### 9.2 Task Dependency Graph

```
TASK-DREAM-P0-001 (Types & Interfaces)
        |
        +---> TASK-DREAM-P0-002 (Poincare Math)
        |            |
        |            +---> TASK-DREAM-P0-004 (Hyperbolic Walk)
        |                          |
        +---> TASK-DREAM-P0-003 (Hebbian Learning)
        |            |             |
        |            +-------------+
        |                          |
        +---> TASK-DREAM-P0-005 (Dream Triggers)
                     |             |
                     +-------------+
                                   |
                                   v
                     TASK-DREAM-P0-006 (Wake Controller & Integration)
```

### 9.3 NFR to Test Mapping

| NFR | Benchmark Test | Acceptance Threshold |
|-----|---------------|---------------------|
| NFR-DREAM-001 | BM-DREAM-001 | NREM duration <= 180s |
| NFR-DREAM-002 | BM-DREAM-002 | REM duration <= 120s |
| NFR-DREAM-003 | BM-DREAM-003 | Wake latency p99 < 100ms |
| NFR-DREAM-004 | BM-DREAM-004 | GPU usage < 30% |
| NFR-DREAM-005 | CT-DREAM-001, CT-DREAM-002 | 100% abort success |
| NFR-DREAM-006 | IT-DREAM-001 | Compression ratio > 1.0 |
| NFR-DREAM-007 | BM-DREAM-001 | Handles 10K+ memories |

### 9.4 Task Summary Table

| Task ID | Title | Layer | Effort | Status |
|---------|-------|-------|--------|--------|
| TASK-DREAM-P0-001 | Types and Interfaces | 1 (Foundation) | 2h | Specified |
| TASK-DREAM-P0-002 | Poincare Ball Math Utilities | 1 (Foundation) | 3h | Specified |
| TASK-DREAM-P0-003 | Hebbian Learning Implementation | 2 (Logic) | 4h | Specified |
| TASK-DREAM-P0-004 | Hyperbolic Random Walk | 2 (Logic) | 4h | Specified |
| TASK-DREAM-P0-005 | Dream Trigger Implementation | 2 (Logic) | 3h | Specified |
| TASK-DREAM-P0-006 | Wake Controller & MCP Integration | 3 (Surface) | 4h | Specified |
| **TOTAL** | | | **20h** | |

---

## 10. GWT Workspace Integration

### 10.1 Event Flow

```
                    GWT Workspace
                         |
    +--------------------+--------------------+
    |                    |                    |
    v                    v                    v
query_event        entropy_update       gpu_usage_update
    |                    |                    |
    v                    v                    v
WakeController    EntropyCalculator    GpuMonitor
    |                    |                    |
    |                    +--------------------+
    |                              |
    v                              v
abort_dream              TriggerManager
    |                              |
    +------------------------------+
                   |
                   v
            DreamScheduler
                   |
    +--------------+---------------+
    |                              |
    v                              v
NREM Phase                    REM Phase
    |                              |
    +------------------------------+
                   |
                   v
         MCP Event Broadcaster
                   |
    +--------------+--------------+
    |              |              |
    v              v              v
dream_started  blind_spot   dream_completed
               discovered
```

### 10.2 MCP Event Schemas

```yaml
dream_cycle_started:
  type: object
  required: [session_id, trigger_reason, timestamp_ms]
  properties:
    session_id: { type: string, format: uuid }
    trigger_reason: { type: string, enum: [idle_timeout, high_entropy, gpu_overload, manual] }
    timestamp_ms: { type: integer }
    expected_nrem_duration_ms: { type: integer, default: 180000 }
    expected_rem_duration_ms: { type: integer, default: 120000 }

nrem_phase_completed:
  type: object
  required: [session_id, timestamp_ms, memories_replayed, edges_strengthened, duration_ms]
  properties:
    session_id: { type: string, format: uuid }
    timestamp_ms: { type: integer }
    memories_replayed: { type: integer }
    edges_strengthened: { type: integer }
    edges_pruned: { type: integer }
    duration_ms: { type: integer }
    compression_ratio: { type: number }

rem_phase_completed:
  type: object
  required: [session_id, timestamp_ms, queries_generated, blind_spots_found, duration_ms]
  properties:
    session_id: { type: string, format: uuid }
    timestamp_ms: { type: integer }
    queries_generated: { type: integer }
    blind_spots_found: { type: integer }
    walk_distance: { type: number }
    duration_ms: { type: integer }
    average_semantic_leap: { type: number }

blind_spot_discovered:
  type: object
  required: [session_id, timestamp_ms, blind_spot_id, poincare_position, semantic_distance]
  properties:
    session_id: { type: string, format: uuid }
    timestamp_ms: { type: integer }
    blind_spot_id: { type: string, format: uuid }
    poincare_position: { type: array, items: { type: number }, minItems: 64, maxItems: 64 }
    semantic_distance: { type: number, minimum: 0.7 }
    confidence: { type: number, minimum: 0, maximum: 1 }

dream_cycle_completed:
  type: object
  required: [session_id, timestamp_ms, completed, wake_reason, total_duration_ms]
  properties:
    session_id: { type: string, format: uuid }
    timestamp_ms: { type: integer }
    completed: { type: boolean }
    wake_reason: { type: string }
    shortcuts_created: { type: integer }
    total_duration_ms: { type: integer }
    wake_latency_ms: { type: integer }

wake_triggered:
  type: object
  required: [session_id, timestamp_ms, reason, phase]
  properties:
    session_id: { type: string, format: uuid }
    timestamp_ms: { type: integer }
    reason: { type: string, enum: [external_query, gpu_over_budget, manual_abort, cycle_complete] }
    phase: { type: string, enum: [nrem, rem, idle] }
    latency_ms: { type: integer }
```

### 10.3 Subscription Topics

| Topic | Subscribers | Purpose |
|-------|------------|---------|
| `dream.cycle.started` | GWT Workspace, Logging | Notify workspace of dream entry |
| `dream.cycle.completed` | GWT Workspace, Metrics | Notify workspace of dream exit |
| `dream.phase.nrem.completed` | Metrics, Analytics | NREM statistics |
| `dream.phase.rem.completed` | Metrics, Analytics | REM statistics |
| `dream.blind_spot.discovered` | Knowledge Graph, Metrics | New exploration target |
| `dream.shortcut.created` | Knowledge Graph | New edge for faster retrieval |
| `dream.wake.triggered` | GWT Workspace, Alerting | Immediate wake notification |
| `dream.gpu.warning` | Alerting, Auto-scaling | GPU budget concern |

---

## 11. Appendix

### A. Formula Reference

**Hebbian Learning:**
```
dw_ij = eta * phi_i * phi_j
w_new = (w_old * (1 - decay)) + dw_ij
w_final = clamp(w_new, floor, cap)

Where:
- eta = 0.01 (learning rate)
- phi_i, phi_j in [0.0, 1.0] (activation levels)
- decay = 0.001 (weight decay per cycle)
- floor = 0.05 (minimum weight before pruning)
- cap = 1.0 (maximum weight)
```

**Mobius Addition (Poincare Ball):**
```
Full formula:
p' = ((1 + 2<p,v> + ||v||^2)p + (1 - ||p||^2)v) / (1 + 2<p,v> + ||p||^2 ||v||^2)

Simplified (for small v):
p' ~ (p + v) / (1 + <p,v>)

Where:
- p is current position (norm < 1)
- v is velocity/direction vector
- <p,v> is inner product
- ||.|| is Euclidean norm
```

**Geodesic Distance (Poincare Ball):**
```
d(p, q) = acosh(1 + 2||p - q||^2 / ((1 - ||p||^2)(1 - ||q||^2)))

Where:
- p, q are points in the Poincare ball (norm < 1)
- acosh(x) = ln(x + sqrt(x^2 - 1))
```

**Kuramoto Synchronization:**
```
d(theta_i)/dt = omega_i + (K/N) * sum_j(sin(theta_j - theta_i))
r * e^(i*psi) = (1/N) * sum_j(e^(i*theta_j))

Where:
- K = 10.0 (coupling strength during NREM)
- theta_i is phase of oscillator i
- omega_i is natural frequency
- r is order parameter (synchronization measure)
```

**Exploration Softmax:**
```
P(i) = exp(score_i / T) / sum_j(exp(score_j / T))

Where:
- T = 2.0 (temperature, Constitution mandate)
- Higher T = more uniform distribution (exploration)
- Lower T = more peaked distribution (exploitation)
```

### B. Related Constitution Sections

- Section `dream` (lines 446-453)
- Section `gwt.mental_checks`
- Section `neuromod`
- Section `layer.L5_Coherence`

### C. Implementation Priority Order

1. **TASK-DREAM-P0-001** (Types) - Foundation, no dependencies
2. **TASK-DREAM-P0-002** (Poincare Math) - Depends on 001
3. **TASK-DREAM-P0-003** (Hebbian) - Depends on 001
4. **TASK-DREAM-P0-004** (Hyperbolic Walk) - Depends on 001, 002
5. **TASK-DREAM-P0-005** (Triggers) - Depends on 001
6. **TASK-DREAM-P0-006** (Integration) - Depends on all above

### D. Constitution Constraint Summary

| Constraint | Value | Enforcement Point |
|------------|-------|-------------------|
| NREM duration | 3 min | DreamController deadline |
| REM duration | 2 min | DreamController deadline |
| Activity threshold | 0.15 | DreamScheduler trigger |
| Idle trigger duration | 10 min | DreamScheduler timer |
| Entropy threshold | 0.7 | EntropyWindow check |
| Entropy window | 5 min | EntropyWindow duration |
| GPU trigger | 80% | GpuMonitor threshold |
| GPU budget (dream) | 30% | WakeController check |
| Wake latency | < 100ms | WakeController measurement |
| REM queries | 100 max | RemPhase counter |
| Semantic leap | >= 0.7 | BlindSpot filter |
| Shortcut hops | 3+ | AmortizedLearner filter |
| Shortcut traversals | 5+ | AmortizedLearner counter |
| Shortcut confidence | >= 0.7 | AmortizedLearner gate |
| Learning rate | 0.01 | HebbianConfig default |
| Weight decay | 0.001 | HebbianConfig default |
| Weight floor | 0.05 | HebbianEngine pruning |
| Weight cap | 1.0 | HebbianEngine clamping |
| Coupling strength | 10.0 | HebbianConfig (Kuramoto K) |
| Temperature | 2.0 | HyperbolicWalkConfig default |
| Recency bias | 0.8 | NremPhase memory selection |
