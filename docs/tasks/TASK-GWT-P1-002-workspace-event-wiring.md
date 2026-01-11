# TASK-GWT-P1-002: Workspace Event Wiring to Subsystems

## ✅ STATUS: COMPLETED

**Completed:** 2026-01-11
**Last Updated:** 2026-01-10
**Audited Against:** Current codebase (commit 115b1f6)
**All file paths and structures verified against git HEAD**

### Implementation Summary

All requirements implemented:
- ✅ Added `register_listener()` and `listener_count()` to `WorkspaceEventBroadcaster`
- ✅ Created `listeners.rs` with `DreamEventListener`, `NeuromodulationEventListener`, `MetaCognitiveEventListener`
- ✅ Added `inhibit_losers()` to `GlobalWorkspace` using `neuromod.adjust(ModulatorType::Dopamine, -magnitude)`
- ✅ Wired all 3 listeners in `GwtSystem::new()` with shared state via `Arc<RwLock<>>`
- ✅ Added helper methods: `is_epistemic_action_triggered()`, `reset_epistemic_action()`, `dream_queue_len()`, `drain_dream_queue()`
- ✅ All 92 GWT tests pass
- ✅ Clippy passes with no warnings
- ✅ Code review by code-simplifier agent: no issues found

---

## Quick Reference

| Item | Value |
|------|-------|
| Task Type | Event Wiring / Integration |
| Priority | P1 (Surface Layer) |
| Complexity | Medium |
| Estimated Files | 2 modified, 1 created |
| Dependencies | TASK-GWT-P0-001 ✅, TASK-GWT-P0-002 ✅, TASK-GWT-P0-003 ✅ |

---

## Context

The `WorkspaceEventBroadcaster` exists in the GWT system but has **NO listeners registered**.
Events like `MemoryEnters`, `MemoryExits`, `WorkspaceConflict`, and `WorkspaceEmpty` fire into
the void. This task wires these events to their designated handlers:

- **DreamController** - Receives `MemoryExits` for offline replay consolidation
- **NeuromodulationManager** - Boosts dopamine on `MemoryEnters`
- **MetaCognitiveLoop** - Triggers epistemic action on `WorkspaceEmpty`
- **Losers** - Receive dopamine reduction (completing WTA algorithm step 6)

### Constitution Reference

From `docs2/constitution.yaml`:

```yaml
# neuromod.Dopamine (lines 162-170)
dopamine:
  trigger: "memory_enters_workspace"
  increment: 0.2

# gwt.global_workspace (lines 352-369)
# Step 6: "Inhibit: losing candidates receive dopamine reduction"

# gwt.workspace_events
# - memory_exits_workspace → dream replay logging
# - workspace_empty → epistemic action trigger
```

---

## Source Files (Verified 2026-01-10)

### Input Context Files

| File | Lines | Purpose | Key Symbols |
|------|-------|---------|-------------|
| `crates/context-graph-core/src/gwt/workspace.rs` | 398 | WorkspaceEventBroadcaster, WorkspaceEvent enum | `WorkspaceEvent` (L230), `WorkspaceEventListener` (L262), `WorkspaceEventBroadcaster` (L267) |
| `crates/context-graph-core/src/gwt/mod.rs` | 946 | GwtSystem struct that owns event_broadcaster | `GwtSystem` (L71), `event_broadcaster` (L88) |
| `crates/context-graph-core/src/dream/controller.rs` | ~620 | DreamController for MemoryExits handling | `set_interrupt()` (L480), `should_trigger_dream()` (L495) |
| `crates/context-graph-core/src/neuromod/state.rs` | ~500 | NeuromodulationManager for dopamine boost | `on_workspace_entry()` (L259), `on_negative_event()` (L285) |
| `crates/context-graph-core/src/neuromod/dopamine.rs` | 248 | DopamineModulator methods | `on_workspace_entry()` (L91), `on_negative_event()` (L101), `DA_WORKSPACE_INCREMENT = 0.2` (L35) |
| `crates/context-graph-core/src/gwt/meta_cognitive.rs` | 453 | MetaCognitiveLoop for WorkspaceEmpty | `acetylcholine()` (L197), `monitoring_frequency()` (L202) |

### Current Code State

**WorkspaceEvent enum** (workspace.rs:230-260):
```rust
pub enum WorkspaceEvent {
    MemoryEnters { id: Uuid, order_parameter: f32, timestamp: DateTime<Utc> },
    MemoryExits { id: Uuid, order_parameter: f32, timestamp: DateTime<Utc> },
    WorkspaceConflict { memories: Vec<Uuid>, timestamp: DateTime<Utc> },
    WorkspaceEmpty { duration_ms: u64, timestamp: DateTime<Utc> },
    IdentityCritical { identity_coherence: f32, reason: String, timestamp: DateTime<Utc> },
}
```

**WorkspaceEventListener trait** (workspace.rs:262-264):
```rust
pub trait WorkspaceEventListener: Send + Sync {
    fn on_event(&self, event: &WorkspaceEvent);
}
```

**WorkspaceEventBroadcaster** (workspace.rs:267-296):
```rust
pub struct WorkspaceEventBroadcaster {
    listeners: std::sync::Arc<tokio::sync::RwLock<Vec<Box<dyn WorkspaceEventListener>>>>,
}

impl WorkspaceEventBroadcaster {
    pub fn new() -> Self { ... }
    pub async fn broadcast(&self, event: WorkspaceEvent) { ... }
    // MISSING: register_listener() method
}
```

**NeuromodulationManager.on_workspace_entry()** (state.rs:259-261):
```rust
pub fn on_workspace_entry(&mut self) {
    self.dopamine.on_workspace_entry();
}
```

**DopamineModulator.on_workspace_entry()** (dopamine.rs:91-98):
```rust
pub fn on_workspace_entry(&mut self) {
    self.level.value = (self.level.value + DA_WORKSPACE_INCREMENT).clamp(DA_MIN, DA_MAX);
    self.level.last_trigger = Some(Utc::now());
}
```

**DopamineModulator.on_negative_event()** (dopamine.rs:101-108):
```rust
pub fn on_negative_event(&mut self, magnitude: f32) {
    let delta = magnitude.abs() * 0.1;
    self.level.value = (self.level.value - delta).clamp(DA_MIN, DA_MAX);
}
```

---

## Scope

### In Scope

1. Add `register_listener()` method to `WorkspaceEventBroadcaster`
2. Create `DreamEventListener` implementing `WorkspaceEventListener`
3. Create `NeuromodulationEventListener` implementing `WorkspaceEventListener`
4. Create `MetaCognitiveEventListener` implementing `WorkspaceEventListener`
5. Wire `MemoryExits` → DreamController queue for replay
6. Wire `MemoryEnters` → NeuromodulationManager::on_workspace_entry()
7. Wire `WorkspaceEmpty` → MetaCognitiveLoop epistemic action flag
8. Implement dopamine reduction for losing WTA candidates
9. Register all listeners during GwtSystem initialization
10. Add integration tests for event flow

### Out of Scope

- KuramotoNetwork integration (TASK-GWT-P0-001 ✅)
- Background Kuramoto stepping (TASK-GWT-P0-002 ✅)
- SelfAwarenessLoop activation (TASK-GWT-P0-003 ✅)
- Dream cycle execution logic (already implemented)
- Neuromodulator decay logic (already implemented)

---

## Implementation Requirements

### CRITICAL: NO BACKWARDS COMPATIBILITY

**FAIL FAST with robust error logging:**
- Panics are acceptable for invariant violations
- No try-catch recovery patterns
- Log detailed context before panic
- Use `tracing::error!` for critical failures

```rust
// CORRECT - Fail fast with context
pub async fn register_listener(&self, listener: Box<dyn WorkspaceEventListener>) {
    let mut listeners = self.listeners.write().await;
    tracing::debug!("Registering workspace event listener (total: {})", listeners.len() + 1);
    listeners.push(listener);
}

// CORRECT - Fail fast on invariant violation
fn on_event(&self, event: &WorkspaceEvent) {
    match self.neuromod_manager.try_write() {
        Ok(mut guard) => guard.on_workspace_entry(),
        Err(e) => {
            tracing::error!("CRITICAL: Failed to acquire neuromod lock: {:?}", e);
            panic!("NeuromodulationEventListener: Lock poisoned or deadlocked");
        }
    }
}
```

### CRITICAL: NO MOCK DATA IN TESTS

All tests must use **real data structures** with known values:

```rust
// WRONG - Mock/stub
let mock_neuromod = MockNeuromodulationManager::new();

// CORRECT - Real data with synthetic known values
let neuromod = Arc::new(RwLock::new(NeuromodulationManager::new()));
{
    let mut mgr = neuromod.write().await;
    mgr.dopamine.set_value(3.0); // Known baseline
}
```

---

## Definition of Done

### Required Signatures

**workspace.rs:**
```rust
impl WorkspaceEventBroadcaster {
    pub async fn register_listener(&self, listener: Box<dyn WorkspaceEventListener>);
}

impl GlobalWorkspace {
    /// Apply dopamine reduction to losing WTA candidates
    pub async fn inhibit_losers(
        &self,
        winner_id: Uuid,
        neuromod: &mut NeuromodulationManager,
    ) -> CoreResult<usize>;
}
```

**listeners.rs (NEW FILE):**
```rust
/// Listener that queues exiting memories for dream replay
pub struct DreamEventListener {
    dream_queue: Arc<RwLock<Vec<Uuid>>>,
}

impl WorkspaceEventListener for DreamEventListener {
    fn on_event(&self, event: &WorkspaceEvent);
}

/// Listener that boosts dopamine on memory entry
pub struct NeuromodulationEventListener {
    neuromod_manager: Arc<RwLock<NeuromodulationManager>>,
}

impl WorkspaceEventListener for NeuromodulationEventListener {
    fn on_event(&self, event: &WorkspaceEvent);
}

/// Listener that triggers epistemic action on workspace empty
pub struct MetaCognitiveEventListener {
    meta_cognitive: Arc<RwLock<MetaCognitiveLoop>>,
    epistemic_action_triggered: Arc<AtomicBool>,
}

impl WorkspaceEventListener for MetaCognitiveEventListener {
    fn on_event(&self, event: &WorkspaceEvent);
}

pub const DA_INHIBITION_FACTOR: f32 = 0.1;
```

### Constraints

- Listeners MUST be `Send + Sync` to work with async broadcaster
- `DreamEventListener` MUST NOT block - queue operation only
- `NeuromodulationEventListener` dopamine boost MUST use `on_workspace_entry()`
- Loser inhibition MUST use `DopamineModulator::on_negative_event()`
- `MetaCognitiveEventListener` MUST set flag, not block on action
- All listeners MUST handle all event variants (even if no-op)
- Event handling MUST panic on lock failures (fail fast)
- Memory IDs from events MUST be propagated correctly

---

## Pseudo Code

### WorkspaceEventBroadcaster::register_listener (workspace.rs)
```
fn register_listener(listener):
  Acquire write lock on listeners vector
  Push listener box into vector
  Log registration with count
```

### DreamEventListener::on_event (listeners.rs)
```
fn on_event(event):
  Match event variant:
    MemoryExits { id, order_parameter, timestamp } =>
      Acquire write lock on dream_queue (panic on failure)
      Push id to dream_queue
      Log: "Queued memory {} for dream replay (r={:.3})"
    IdentityCritical { identity_coherence, reason, .. } =>
      Log: "Identity critical (IC={:.3}): {}"
      // DreamController will be notified separately
    _ => no-op
```

### NeuromodulationEventListener::on_event (listeners.rs)
```
fn on_event(event):
  Match event variant:
    MemoryEnters { id, order_parameter, timestamp } =>
      Acquire write lock on neuromod_manager (panic on failure)
      Call neuromod_manager.on_workspace_entry()
      Log: "Dopamine boosted for memory {} entering workspace (r={:.3})"
    _ => no-op
```

### MetaCognitiveEventListener::on_event (listeners.rs)
```
fn on_event(event):
  Match event variant:
    WorkspaceEmpty { duration_ms, timestamp } =>
      Set epistemic_action_triggered flag to true
      Log: "Workspace empty for {}ms - epistemic action triggered"
    _ => no-op
```

### GlobalWorkspace::inhibit_losers (workspace.rs)
```
fn inhibit_losers(winner_id, neuromod):
  Count = 0
  For each candidate in self.candidates:
    If candidate.id != winner_id:
      inhibition_magnitude = 1.0 - candidate.score
      neuromod.on_negative_event(inhibition_magnitude * DA_INHIBITION_FACTOR)
      Count += 1
  Return count of inhibited candidates
```

### GwtSystem Integration (mod.rs)
```
// In GwtSystem::new() - register listeners after broadcaster creation
fn new():
  ... existing code ...

  // Create listeners
  let dream_listener = DreamEventListener::new(dream_queue.clone());
  let neuromod_listener = NeuromodulationEventListener::new(neuromod_manager.clone());
  let meta_listener = MetaCognitiveEventListener::new(meta_cognitive.clone());

  // Register all listeners
  event_broadcaster.register_listener(Box::new(dream_listener)).await;
  event_broadcaster.register_listener(Box::new(neuromod_listener)).await;
  event_broadcaster.register_listener(Box::new(meta_listener)).await;
```

---

## Files to Create

### `crates/context-graph-core/src/gwt/listeners.rs`

New module containing:
- `DreamEventListener` struct and impl
- `NeuromodulationEventListener` struct and impl
- `MetaCognitiveEventListener` struct and impl
- `DA_INHIBITION_FACTOR` constant (0.1 per PRD)
- Helper types for async listener operations
- Unit tests (in same file)

## Files to Modify

### `crates/context-graph-core/src/gwt/workspace.rs`

- Add `register_listener()` method to `WorkspaceEventBroadcaster`
- Add `inhibit_losers()` method to `GlobalWorkspace`
- Add tests for new functionality

### `crates/context-graph-core/src/gwt/mod.rs`

- Add `pub mod listeners;` declaration
- Re-export listener types: `DreamEventListener`, `NeuromodulationEventListener`, `MetaCognitiveEventListener`
- Modify `GwtSystem::new()` to create and register listeners
- Add integration test for full event flow

---

## Full State Verification (FSV) Requirements

### 1. Source of Truth

| Component | Source of Truth | Verification Method |
|-----------|-----------------|---------------------|
| Listener Registration | `WorkspaceEventBroadcaster.listeners.len()` | Read lock, check count |
| Dopamine Value | `NeuromodulationManager.dopamine.value()` | Read before/after event |
| Dream Queue | `DreamEventListener.dream_queue.len()` | Read lock, check count |
| Epistemic Flag | `MetaCognitiveEventListener.epistemic_action_triggered` | AtomicBool load |
| Loser Inhibition | `DopamineModulator.value()` before/after | Compare pre/post values |

### 2. Execute & Inspect Protocol

For each test:
1. **SETUP**: Create real instances with known initial values
2. **BEFORE**: Read and log all relevant state
3. **EXECUTE**: Perform the operation
4. **AFTER**: Read state via SEPARATE read (not same lock)
5. **VERIFY**: Assert state changes match expected
6. **EVIDENCE**: Print confirmation message

```rust
#[tokio::test]
async fn test_fsv_neuromod_listener_dopamine_boost() {
    println!("=== FSV: NeuromodulationEventListener ===");

    // SETUP
    let neuromod = Arc::new(RwLock::new(NeuromodulationManager::new()));
    let listener = NeuromodulationEventListener::new(neuromod.clone());

    // BEFORE - Read via separate lock
    let before_da = {
        let mgr = neuromod.read().await;
        mgr.dopamine.value()
    };
    println!("BEFORE: dopamine = {:.3}", before_da);

    // EXECUTE
    let event = WorkspaceEvent::MemoryEnters {
        id: Uuid::new_v4(),
        order_parameter: 0.85,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    // AFTER - Read via SEPARATE lock
    let after_da = {
        let mgr = neuromod.read().await;
        mgr.dopamine.value()
    };
    println!("AFTER: dopamine = {:.3}", after_da);

    // VERIFY
    let expected_da = before_da + DA_WORKSPACE_INCREMENT;
    assert!((after_da - expected_da).abs() < f32::EPSILON,
        "Expected dopamine {:.3}, got {:.3}", expected_da, after_da);

    // EVIDENCE
    println!("EVIDENCE: Dopamine correctly increased by {:.3}", DA_WORKSPACE_INCREMENT);
}
```

### 3. Boundary & Edge Case Audit

| Edge Case | Test Name | Expected Behavior |
|-----------|-----------|-------------------|
| Empty listeners vector | `test_broadcast_empty_listeners` | No panic, no-op |
| All events to DreamEventListener | `test_dream_listener_all_events` | Only MemoryExits queued |
| Dopamine at max | `test_neuromod_listener_at_max` | Clamped to DA_MAX (5.0) |
| Dopamine at min | `test_neuromod_listener_at_min` | Clamped to DA_MIN (1.0) |
| WorkspaceEmpty duration=0 | `test_meta_listener_zero_duration` | Flag set, no panic |
| inhibit_losers no losers | `test_inhibit_losers_single_winner` | Returns 0 |
| inhibit_losers all losers | `test_inhibit_losers_all_below_threshold` | All inhibited |
| IdentityCritical event | `test_dream_listener_identity_critical` | Logs, no queue |
| Concurrent listener access | `test_concurrent_event_broadcast` | No deadlock |

### 4. Evidence of Success

Each test MUST output explicit evidence:
```
=== TEST: [Test Name] ===
BEFORE: [state before]
EXECUTE: [action performed]
AFTER: [state after]
EVIDENCE: [proof of correct behavior]
```

---

## Test Commands

```bash
# Run all listener tests
cargo test -p context-graph-core gwt::listeners -- --nocapture

# Run workspace registration tests
cargo test -p context-graph-core gwt::workspace::tests::test_register_listener -- --nocapture

# Run inhibit_losers tests
cargo test -p context-graph-core gwt::workspace::tests::test_inhibit_losers -- --nocapture

# Run integration tests
cargo test -p context-graph-core gwt::tests::test_event_flow_integration -- --nocapture

# Run all GWT tests
cargo test -p context-graph-core gwt -- --nocapture

# Clippy check
cargo clippy -p context-graph-core -- -D warnings

# Build verification
cargo build -p context-graph-core
```

---

## Synthetic Test Data

### Known Inputs for Testing

```rust
// Workspace candidates with known scores
const TEST_CANDIDATES: [(Uuid, f32, f32, f32); 4] = [
    (uuid!("11111111-1111-1111-1111-111111111111"), 0.90, 0.85, 0.88), // score = 0.6732 (winner)
    (uuid!("22222222-2222-2222-2222-222222222222"), 0.85, 0.80, 0.82), // score = 0.5576 (loser)
    (uuid!("33333333-3333-3333-3333-333333333333"), 0.82, 0.75, 0.78), // score = 0.4797 (loser)
    (uuid!("44444444-4444-4444-4444-444444444444"), 0.75, 0.70, 0.72), // score = 0.378 (filtered out)
];

// Expected dopamine changes
const DA_BASELINE: f32 = 3.0;
const DA_WORKSPACE_INCREMENT: f32 = 0.2;
const DA_INHIBITION_FACTOR: f32 = 0.1;

// After winner enters: DA = 3.0 + 0.2 = 3.2
// Loser 1 inhibition: (1.0 - 0.5576) * 0.1 = 0.04424 → DA = 3.2 - 0.0044 = 3.1956
// Loser 2 inhibition: (1.0 - 0.4797) * 0.1 = 0.05203 → DA = 3.1956 - 0.0052 = 3.1904
```

---

## Event Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    GwtSystem::select_workspace_memory()          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  GlobalWorkspace::select_winning_memory()          │
              │  - Filters by coherence (r >= 0.8)                 │
              │  - Ranks by score = r × importance × alignment     │
              │  - Selects top-1 winner                            │
              └───────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
       Winner Selected                  No Winner
              │                               │
              │                               ▼
              │                  ┌─────────────────────────┐
              │                  │ Broadcast: WorkspaceEmpty        │
              │                  └─────────────────────────┘
              │                               │
              │                               ▼
              │                  ┌─────────────────────────┐
              │                  │ MetaCognitiveEventListener       │
              │                  │ → Set epistemic_action flag      │
              │                  └─────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Broadcast: MemoryEnters (winner)                │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│             NeuromodulationEventListener                         │
│             → NeuromodulationManager::on_workspace_entry()       │
│             → Dopamine += 0.2 (DA_WORKSPACE_INCREMENT)           │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│         GlobalWorkspace::inhibit_losers(winner_id, neuromod)     │
│         → For each non-winner candidate:                         │
│             - inhibition = (1.0 - score) * DA_INHIBITION_FACTOR  │
│             - DopamineModulator::on_negative_event(inhibition)   │
│             - Broadcast: MemoryExits                             │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DreamEventListener (for each loser)             │
│                  → Queue memory ID for dream replay              │
│                  → dream_queue.push(loser_id)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Related Tasks

| Task | Status | Relationship |
|------|--------|--------------|
| TASK-GWT-P0-001 | ✅ COMPLETED | Prerequisite - Kuramoto integration |
| TASK-GWT-P0-002 | ✅ COMPLETED | Prerequisite - Background stepper |
| TASK-GWT-P0-003 | ✅ COMPLETED | Prerequisite - SelfAwarenessLoop activation |
| TASK-GWT-P1-001 | ✅ COMPLETED | Prerequisite - EgoNode persistence |
| TASK-GWT-P1-003 | Ready | Depends on this task - Dream queue integration |

---

## Traceability

| Requirement | Source | Implementation |
|-------------|--------|----------------|
| Dopamine boost on workspace entry | Constitution neuromod.Dopamine.trigger | NeuromodulationEventListener |
| DA increment = 0.2 | dopamine.rs:35 `DA_WORKSPACE_INCREMENT` | Uses existing constant |
| Dream replay on workspace exit | PRD gwt.workspace_events | DreamEventListener |
| Epistemic action on workspace empty | PRD gwt.workspace_events | MetaCognitiveEventListener |
| Dopamine reduction for WTA losers | PRD gwt.global_workspace step 6 | GlobalWorkspace::inhibit_losers |
| Event broadcaster has listeners | Sherlock-01 GAP 4 fix | register_listener() + GwtSystem wiring |
| on_negative_event for serotonin | state.rs:285-286 | Already implemented |

---

## Git History Context

Recent commits (for context):
```
115b1f6 feat(TASK-UTL-P1-001): implement per-embedder ΔS entropy methods
710c7c7 feat(TASK-STORAGE-P1-001): replace HNSW brute force with usearch
39609b0 feat(TASK-GWT-P0-003): implement SelfAwarenessLoop activation
5e10c5e feat(TASK-GWT-P0-002): implement Kuramoto background stepper
c139abf feat(TASK-GWT-P0-001): integrate KuramotoNetwork into GwtSystem
```

---

## Agent Instructions

### Before Starting

1. Read this entire document
2. Run `cargo test -p context-graph-core gwt` to verify baseline (67 tests should pass)
3. Read the source files listed in "Source Files" section
4. Understand the existing code patterns

### Implementation Order

1. **First**: Add `register_listener()` to `WorkspaceEventBroadcaster` (workspace.rs)
2. **Second**: Create `listeners.rs` with all three listener structs
3. **Third**: Add `inhibit_losers()` to `GlobalWorkspace` (workspace.rs)
4. **Fourth**: Wire listeners in `GwtSystem::new()` (mod.rs)
5. **Fifth**: Add comprehensive tests with FSV pattern
6. **Last**: Run full test suite and clippy

### If You Encounter Issues

- Check git history for similar patterns in completed tasks
- Look at TASK-GWT-P0-003 for wiring patterns
- Look at TASK-GWT-P1-001 for FSV test patterns
- Search for existing `on_workspace_entry` usage in state.rs
- If unsure about constitution reference, check `docs2/constitution.yaml`

### Commit Message Format

```
feat(TASK-GWT-P1-002): wire workspace events to subsystem listeners

- Add register_listener() to WorkspaceEventBroadcaster
- Create DreamEventListener, NeuromodulationEventListener, MetaCognitiveEventListener
- Add inhibit_losers() for WTA loser dopamine reduction
- Wire all listeners in GwtSystem::new()
- Add FSV tests for all event flows
```
