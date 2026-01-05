# TASK-L004: Johari Transition Manager

```yaml
metadata:
  id: "TASK-L004"
  title: "Johari Transition Manager"
  layer: "logic"
  priority: "P0"
  estimated_hours: 8
  created: "2026-01-04"
  updated: "2026-01-05"
  status: "COMPLETED"
  completed_commit: "HEAD (pending push)"
  dependencies:
    - "TASK-F003"  # JohariFingerprint - COMPLETED (b059c91)
    - "TASK-F002"  # TeleologicalFingerprint - COMPLETED (303e203, 6e6406b)
  spec_refs:
    - "constitution.yaml:177-184 (Johari quadrants)"
    - "contextprd.md:sections 2.2, 2.4"
    - "learntheory.md (L = f(DS x DC) foundation)"
```

---

## STATUS: COMPLETED

**Implementation verified 2026-01-05**:
- All source files created and functional
- 116 tests passing (`cargo test -p context-graph-core johari`)
- Code compiles without errors
- Full State Verification Protocol executed successfully

---

## IMPLEMENTATION SUMMARY

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `crates/context-graph-core/src/johari/mod.rs` | Module exports + integration tests | ~265 |
| `crates/context-graph-core/src/johari/manager.rs` | `JohariTransitionManager` trait | ~413 |
| `crates/context-graph-core/src/johari/default_manager.rs` | `DefaultJohariManager` impl | ~700+ |
| `crates/context-graph-core/src/johari/error.rs` | `JohariError` enum | ~100 |
| `crates/context-graph-core/src/johari/external_signal.rs` | `ExternalSignal`, `BlindSpotCandidate` | ~200 |
| `crates/context-graph-core/src/johari/stats.rs` | `TransitionStats`, `TransitionPath` | ~250 |

### Files Modified

| File | Change |
|------|--------|
| `crates/context-graph-core/src/lib.rs` | Added `pub mod johari;` |

---

## PUBLIC API

### Core Types (from `context_graph_core::johari`)

```rust
// Trait
pub trait JohariTransitionManager: Send + Sync {
    async fn classify(&self, semantic: &SemanticFingerprint, context: &ClassificationContext) -> Result<JohariFingerprint, JohariError>;
    async fn transition(&self, memory_id: MemoryId, embedder_idx: usize, to_quadrant: JohariQuadrant, trigger: TransitionTrigger) -> Result<JohariFingerprint, JohariError>;
    async fn transition_batch(&self, memory_id: MemoryId, transitions: Vec<(usize, JohariQuadrant, TransitionTrigger)>) -> Result<JohariFingerprint, JohariError>;
    async fn find_by_quadrant(&self, pattern: QuadrantPattern, limit: usize) -> Result<Vec<(MemoryId, JohariFingerprint)>, JohariError>;
    async fn discover_blind_spots(&self, memory_id: MemoryId, external_signals: &[ExternalSignal]) -> Result<Vec<BlindSpotCandidate>, JohariError>;
    async fn get_transition_stats(&self, time_range: TimeRange) -> Result<TransitionStats, JohariError>;
    async fn get_transition_history(&self, memory_id: MemoryId, limit: usize) -> Result<Vec<JohariTransition>, JohariError>;
}

// Default implementation
pub struct DefaultJohariManager<S: TeleologicalMemoryStore>;

// Supporting types
pub type MemoryId = Uuid;
pub struct ClassificationContext { delta_s: [f32; 13], delta_c: [f32; 13], disclosure_intent: [bool; 13], access_counts: [u32; 13] }
pub struct TimeRange { start: DateTime<Utc>, end: DateTime<Utc> }
pub enum QuadrantPattern { AllIn(JohariQuadrant), AtLeast { quadrant, count }, Exact([JohariQuadrant; 13]), Mixed { min_open, max_unknown } }
pub struct ExternalSignal { source: String, embedder_idx: usize, strength: f32, description: Option<String>, timestamp: DateTime<Utc> }
pub struct BlindSpotCandidate { embedder_idx: usize, current_quadrant: JohariQuadrant, signal_strength: f32, suggested_transition: JohariQuadrant, sources: Vec<String> }
pub struct TransitionStats { total_transitions: u64, memories_affected: u64, path_counts: HashMap<TransitionPath, u64>, trigger_counts: HashMap<TransitionTrigger, u64>, embedder_counts: [u64; 13], avg_transitions_per_memory: f32, top_paths: Vec<(TransitionPath, u64)> }
pub struct TransitionPath { from: JohariQuadrant, to: JohariQuadrant }
pub enum JohariError { NotFound(Uuid), InvalidTransition { from, to, embedder_idx }, InvalidEmbedderIndex(usize), StorageError(String), BatchValidationFailed { idx, reason }, ClassificationError(String) }
```

---

## UTL CLASSIFICATION RULES

From `constitution.yaml:177-184` and `learntheory.md`:

| Quadrant | Entropy (DS) | Coherence (DC) | Meaning |
|----------|--------------|----------------|---------|
| **Open** | < 0.5 (low) | > 0.5 (high) | Known to self AND others |
| **Hidden** | < 0.5 (low) | <= 0.5 (low) | Known to self, NOT others |
| **Blind** | >= 0.5 (high) | <= 0.5 (low) | NOT known to self, known to others |
| **Unknown** | >= 0.5 (high) | > 0.5 (high) | NOT known to self OR others |

**Boundary behavior** (at exactly 0.5):
- `entropy = 0.5` is treated as HIGH entropy (>= threshold)
- `coherence = 0.5` is treated as LOW coherence (<= threshold)
- Therefore `(0.5, 0.5)` classifies as **Blind**

---

## STATE MACHINE

Valid transitions from `JohariQuadrant::valid_transitions()`:

```
Open    -> Hidden (Privatize)
Hidden  -> Open (ExplicitShare)
Blind   -> Open (SelfRecognition), Hidden (SelfRecognition)
Unknown -> Open (DreamConsolidation, PatternDiscovery), Hidden (DreamConsolidation), Blind (ExternalObservation)
```

**Constraints**:
- Self-transitions are INVALID (from == to returns error)
- Invalid trigger for target returns error
- Embedder index must be 0-12 (13 embedders total)

---

## VERIFICATION COMMANDS

```bash
# Run all Johari tests with output
cargo test -p context-graph-core johari -- --nocapture

# Verify no compilation errors
cargo check -p context-graph-core

# Run clippy (expect only deprecation warnings for legacy EmbeddingProvider)
cargo clippy -p context-graph-core -- -D warnings

# Run specific integration test
cargo test -p context-graph-core full_state_verification_test -- --nocapture
```

---

## EVIDENCE OF COMPLETION

Test output from `full_state_verification_test`:

```
========== TASK-L004 FULL STATE VERIFICATION ==========

[TEST 1] Classification from UTL state
  E1: Open (weights: [1.0, 0.0, 0.0, 0.0])
  ...
  E13: Open (weights: [1.0, 0.0, 0.0, 0.0])
[TEST 1 PASSED] All embedders classified as Open

[TEST 2] Valid transition: Hidden -> Open
  [STATE BEFORE] E1: Hidden
  [STATE AFTER] E1: Open
  [VERIFIED] Transition persisted to store
[TEST 2 PASSED] Transition succeeded and persisted

[TEST 3] Invalid transition: Open -> Blind (should fail)
  [ERROR] InvalidTransition { from: Open, to: Blind, embedder_idx: 0 }
[TEST 3 PASSED] Invalid transition correctly rejected

[TEST 4] Blind spot discovery from external signals
  Found 1 blind spot candidates
    E6: Unknown -> Blind (strength: 0.70)
[TEST 4 PASSED] Blind spot correctly discovered

[TEST 5] Batch transitions (all-or-nothing)
  [STATE BEFORE] E1-E3: Unknown
  [STATE AFTER] E1: Open, E2: Hidden, E3: Blind
  [VERIFIED] All batch transitions persisted
[TEST 5 PASSED] Batch transitions succeeded

========== EVIDENCE OF SUCCESS ==========
[EVIDENCE] TASK-L004 Verification Complete
  - Total tests run: 5
  - Tests passed: 5
  - Storage operations verified: 4
  - Edge cases validated: 3/3 (in unit tests)
  - State transitions verified: 5
==========================================
```

---

## DEPENDENCIES FOR DOWNSTREAM TASKS

Tasks that depend on TASK-L004:

| Task | Dependency Usage |
|------|------------------|
| TASK-L008 | Uses `JohariTransitionManager` for awareness-based filtering in retrieval pipeline |

---

## RELATED FILES

### Existing Types (DO NOT MODIFY)

| Type | Location | Purpose |
|------|----------|---------|
| `JohariQuadrant` | `src/types/johari/quadrant.rs` | Enum with transitions |
| `JohariTransition` | `src/types/johari/transition.rs` | Transition record |
| `TransitionTrigger` | `src/types/johari/transition.rs` | Trigger enum (6 variants) |
| `JohariFingerprint` | `src/types/fingerprint/johari/core.rs` | 13-embedder soft classification |
| `TeleologicalMemoryStore` | `src/traits/teleological_memory_store.rs` | Storage trait |
| `InMemoryTeleologicalStore` | `src/stubs/teleological_store_stub.rs` | Test stub |

---

*Task completed: 2026-01-05*
*Verified: All 116 johari tests passing*
*Layer: Logic*
*Priority: P0 - Core awareness tracking*
