# Functional Specification: Performance Domain Remediation

## Metadata

| Field | Value |
|-------|-------|
| **ID** | SPEC-PERF-001 |
| **Title** | Performance Domain Remediation - Async Safety and Memory Optimization |
| **Status** | Draft |
| **Owner** | Performance Engineering |
| **Version** | 1.0.0 |
| **Created** | 2026-01-12 |
| **Last Updated** | 2026-01-12 |
| **Related Issues** | ISS-004 (CRITICAL), ISS-015 (LOW), ISS-016 (LOW) |
| **Related Specs** | PRD-ANALYSIS-REMEDIATION.md, MASTER-ISSUES-REMEDIATION-PLAN.md |
| **Constitution Rules** | AP-08 |

---

## Overview

### Problem Statement

The Performance Domain contains three issues that impact system reliability and efficiency:

1. **ISS-004 (CRITICAL)**: Eight instances of `futures::executor::block_on()` in async contexts within `gwt_providers.rs` violate Constitution rule AP-08 ("No sync I/O in async context"). These calls can cause **deadlocks** when executed on single-threaded async runtimes, as `block_on()` blocks the executor thread while waiting for an async lock that can only be released by the blocked executor.

2. **ISS-015 (LOW)**: The `WakeController` in `wake_controller.rs` uses `std::sync::RwLock` instead of `parking_lot::RwLock` or `tokio::sync::RwLock`. While not causing deadlocks, this creates suboptimal performance due to OS-level synchronization overhead.

3. **ISS-016 (LOW)**: Multiple `HashMap::new()` calls throughout the codebase lack capacity hints, causing unnecessary allocations and rehashing during growth.

### Why This Matters

- **Deadlock Risk**: The `block_on()` pattern is fundamentally unsafe in async contexts. When running on a single-threaded tokio runtime (common in tests and some deployments), calling `block_on()` while holding or waiting for an async lock can permanently freeze the system.
- **Correctness over Performance**: Constitution AP-08 exists specifically to prevent these subtle, hard-to-debug runtime failures. This is not an optimization - it's a correctness requirement.
- **Production Impact**: Any MCP tool call that invokes `WorkspaceProvider` methods like `get_active_memory()` or `is_broadcasting()` from an async context risks deadlock.

### Affected Code Locations

#### ISS-004: block_on() Violations (8 instances)

| # | File | Line | Method | Lock Target |
|---|------|------|--------|-------------|
| 1 | `crates/context-graph-mcp/src/handlers/gwt_providers.rs` | 343 | `get_active_memory()` | `self.workspace.read()` |
| 2 | `crates/context-graph-mcp/src/handlers/gwt_providers.rs` | 348 | `is_broadcasting()` | `self.workspace.read()` |
| 3 | `crates/context-graph-mcp/src/handlers/gwt_providers.rs` | 353 | `has_conflict()` | `self.workspace.read()` |
| 4 | `crates/context-graph-mcp/src/handlers/gwt_providers.rs` | 358 | `get_conflict_details()` | `self.workspace.read()` |
| 5 | `crates/context-graph-mcp/src/handlers/gwt_providers.rs` | 363 | `coherence_threshold()` | `self.workspace.read()` |
| 6 | `crates/context-graph-mcp/src/handlers/gwt_providers.rs` | 415 | `acetylcholine()` | `self.meta_cognitive.read()` |
| 7 | `crates/context-graph-mcp/src/handlers/gwt_providers.rs` | 420 | `monitoring_frequency()` | `self.meta_cognitive.read()` |
| 8 | `crates/context-graph-mcp/src/handlers/gwt_providers.rs` | 425 | `get_recent_scores()` | `self.meta_cognitive.read()` |

#### ISS-015: std::sync::RwLock in WakeController

| File | Lines | Fields |
|------|-------|--------|
| `crates/context-graph-core/src/dream/wake_controller.rs` | 90, 100, 103, 109 | `state`, `wake_start`, `wake_complete`, `gpu_monitor` |

#### ISS-016: HashMap without Capacity

Approximately 15+ instances across multiple crates, including:
- `crates/context-graph-core/src/retrieval/aggregation.rs:143,165`
- `crates/context-graph-graph/src/traversal/astar/*.rs`
- `crates/context-graph-storage/src/teleological/*.rs`

---

## User Stories

### US-PERF-001: Safe Async MCP Tool Execution

**Priority**: Must-Have

**Narrative**:
> As the **MCP Server**,
> I want all GWT provider methods to be safe to call from any async context,
> So that MCP tool handlers never deadlock regardless of runtime configuration.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-001-01 | MCP server running on single-threaded tokio runtime | Handler calls `workspace_provider.get_active_memory()` | Method returns immediately without blocking executor |
| AC-001-02 | Multiple concurrent MCP requests | Each invokes `WorkspaceProvider` methods | No deadlock, all requests complete |
| AC-001-03 | CI test suite with `current_thread` runtime | All GWT provider tests execute | All pass without hanging |

### US-PERF-002: Wake Controller Performance

**Priority**: Nice-to-Have

**Narrative**:
> As the **Dream Subsystem**,
> I want the WakeController to use efficient synchronization primitives,
> So that wake latency remains under the 100ms constitution limit.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-002-01 | Dream in progress | External query triggers wake | Wake completes in <100ms (measured) |
| AC-002-02 | High concurrency wake scenario | 10 concurrent wake attempts | No lock contention issues, first wake wins |

### US-PERF-003: Efficient HashMap Allocation

**Priority**: Nice-to-Have

**Narrative**:
> As a **Performance Engineer**,
> I want HashMaps to be pre-allocated when size is predictable,
> So that unnecessary reallocations are avoided during hot paths.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-003-01 | Aggregation with known candidate count | Processing 1000 candidates | HashMap allocated once with correct capacity |
| AC-003-02 | A* search on known graph | Pathfinding on 10K node graph | g_scores HashMap pre-allocated |

---

## Requirements

### REQ-PERF-001: No block_on() in Async Context

**Severity**: CRITICAL
**Source Issue**: ISS-004
**Constitution Rule**: AP-08
**Story Ref**: US-PERF-001

**Description**:
No `futures::executor::block_on()` calls SHALL exist in code paths that may be invoked from async contexts.

**Rationale**:
`block_on()` blocks the current thread until the future completes. In async contexts, this thread IS the executor - blocking it prevents the awaited future from making progress, causing deadlock.

**Implementation Options**:

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A | Make trait methods `async` | Clean async composition | Requires trait consumers to be async |
| B | Use `tokio::sync::RwLock::blocking_read()` | Sync interface preserved | Only safe when called from sync context |
| C | Cache values, avoid lock on read path | Best performance | Complexity, potential staleness |

**Recommended**: Option A (async methods) for `WorkspaceProvider` and `MetaCognitiveProvider` traits.

### REQ-PERF-002: WorkspaceProviderImpl Async Methods

**Severity**: CRITICAL
**Source Issue**: ISS-004
**Story Ref**: US-PERF-001

**Description**:
`WorkspaceProviderImpl` synchronous methods (`get_active_memory`, `is_broadcasting`, `has_conflict`, `get_conflict_details`, `coherence_threshold`) SHALL be converted to async OR use sync-safe access patterns.

**Technical Constraints**:
- The `WorkspaceProvider` trait in `gwt_traits.rs` defines sync methods
- Changing to async requires updating the trait definition
- All trait implementors and consumers must be updated

**Fix Pattern (Option A - Async)**:
```rust
// In gwt_traits.rs - change trait to fully async
#[async_trait]
pub trait WorkspaceProvider: Send + Sync {
    async fn get_active_memory(&self) -> Option<Uuid>;
    async fn is_broadcasting(&self) -> bool;
    async fn has_conflict(&self) -> bool;
    async fn get_conflict_details(&self) -> Option<Vec<Uuid>>;
    async fn coherence_threshold(&self) -> f32;
    // ... select_winning_memory already async
}

// In gwt_providers.rs - proper async implementation
async fn get_active_memory(&self) -> Option<Uuid> {
    let workspace = self.workspace.read().await;
    workspace.get_active_memory()
}
```

### REQ-PERF-003: MetaCognitiveProviderImpl Async Methods

**Severity**: CRITICAL
**Source Issue**: ISS-004
**Story Ref**: US-PERF-001

**Description**:
`MetaCognitiveProviderImpl` synchronous methods (`acetylcholine`, `monitoring_frequency`, `get_recent_scores`) SHALL be converted to async OR use sync-safe access patterns.

**Fix Pattern**:
```rust
// In gwt_traits.rs
#[async_trait]
pub trait MetaCognitiveProvider: Send + Sync {
    async fn acetylcholine(&self) -> f32;
    async fn monitoring_frequency(&self) -> f32;
    async fn get_recent_scores(&self) -> Vec<f32>;
    // ... evaluate already async
}

// In gwt_providers.rs
async fn acetylcholine(&self) -> f32 {
    let meta_cognitive = self.meta_cognitive.read().await;
    meta_cognitive.acetylcholine()
}
```

### REQ-PERF-004: WakeController Lock Optimization

**Severity**: LOW
**Source Issue**: ISS-015
**Story Ref**: US-PERF-002

**Description**:
`WakeController` SHOULD use `parking_lot::RwLock` instead of `std::sync::RwLock` for improved performance characteristics.

**Rationale**:
- `parking_lot` locks are faster for uncontended cases
- No lock poisoning (simpler error handling)
- Fair scheduling under contention
- Smaller memory footprint

**Current Code** (wake_controller.rs:90,100,103,109):
```rust
state: Arc<std::sync::RwLock<WakeState>>,
wake_start: Arc<std::sync::RwLock<Option<Instant>>>,
wake_complete: Arc<std::sync::RwLock<Option<Instant>>>,
gpu_monitor: Arc<std::sync::RwLock<GpuMonitor>>,
```

**Target Code**:
```rust
use parking_lot::RwLock;

state: Arc<RwLock<WakeState>>,
wake_start: Arc<RwLock<Option<Instant>>>,
wake_complete: Arc<RwLock<Option<Instant>>>,
gpu_monitor: Arc<RwLock<GpuMonitor>>,
```

### REQ-PERF-005: HashMap Capacity Hints

**Severity**: LOW
**Source Issue**: ISS-016
**Story Ref**: US-PERF-003

**Description**:
`HashMap::new()` calls SHOULD be replaced with `HashMap::with_capacity(n)` where the expected size is known or estimable.

**Examples**:

| File | Current | Improved |
|------|---------|----------|
| `aggregation.rs:143` | `HashMap::new()` | `HashMap::with_capacity(candidates.len())` |
| `bidirectional.rs:58` | `HashMap::new()` | `HashMap::with_capacity(expected_nodes)` |
| `algorithm.rs:83` | `HashMap::new()` | `HashMap::with_capacity(graph.node_count())` |

---

## Edge Cases

### EC-PERF-001: Single-Threaded Runtime Deadlock Prevention

**Related Requirement**: REQ-PERF-001

**Scenario**: System running on `tokio::runtime::Builder::new_current_thread()` with a single executor thread.

**Current Behavior**: `block_on()` blocks the executor thread, preventing the async lock from being released, causing permanent deadlock.

**Expected Behavior**: Async methods properly await the lock without blocking the executor, allowing progress.

### EC-PERF-002: Concurrent Lock Access

**Related Requirement**: REQ-PERF-001

**Scenario**: Multiple MCP handlers simultaneously accessing `workspace.read()` while one handler holds a write lock.

**Current Behavior**: With `block_on()`, readers block their executor threads waiting for the write lock, potentially exhausting the thread pool.

**Expected Behavior**: Async `.await` properly yields execution, allowing the write operation to complete.

### EC-PERF-003: Wake Controller State Race

**Related Requirement**: REQ-PERF-004

**Scenario**: Wake signal arrives exactly as GPU budget check triggers wake.

**Current Behavior**: Works correctly (first signal wins due to state machine).

**Expected Behavior**: No change - behavior remains correct but with lower latency using `parking_lot`.

### EC-PERF-004: HashMap Growth Under Load

**Related Requirement**: REQ-PERF-005

**Scenario**: Processing 10,000 search candidates with HashMap starting at default capacity (0).

**Current Behavior**: Multiple reallocations as HashMap grows: 0 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2048 -> 4096 -> 8192 -> 16384.

**Expected Behavior**: Single allocation with `with_capacity(10000)`.

---

## Error States

### ERR-PERF-001: Async Lock Timeout

**HTTP/Error Code**: N/A (internal)
**Condition**: Async lock acquisition takes longer than expected (>1s)
**Message**: "Lock acquisition timeout - potential deadlock detected"
**Recovery**: Log warning, continue with timeout. If persists, trigger system health alert.

### ERR-PERF-002: Wake Latency Violation

**HTTP/Error Code**: `WakeError::LatencyViolation`
**Condition**: Wake latency exceeds 100ms constitution limit
**Message**: "CONSTITUTION VIOLATION: Wake latency {actual_ms}ms > {max_ms}ms"
**Recovery**: Log error, increment violation counter, complete wake anyway.

### ERR-PERF-003: Lock Poisoning (std::sync only)

**HTTP/Error Code**: Panic
**Condition**: Thread panicked while holding std::sync lock
**Message**: "Lock poisoned - FATAL ERROR"
**Recovery**: With parking_lot, this error is eliminated. With std::sync, system restart required.

---

## Test Plan

### Test Strategy

1. **Unit Tests**: Verify each fixed method works in isolation
2. **Integration Tests**: Verify MCP handler flow with real providers
3. **Concurrency Tests**: Stress test with multiple concurrent callers
4. **Deadlock Detection Tests**: Use single-threaded runtime to verify no deadlocks

### Test Cases

#### TC-PERF-001: Async WorkspaceProvider Methods

**Type**: Unit
**Requirement Ref**: REQ-PERF-002

**Description**: Verify WorkspaceProvider async methods complete without blocking.

```rust
#[tokio::test(flavor = "current_thread")]
async fn test_workspace_provider_no_deadlock_single_thread() {
    let provider = WorkspaceProviderImpl::new();

    // These calls would deadlock with block_on() on single-thread runtime
    let active = provider.get_active_memory().await;
    let broadcasting = provider.is_broadcasting().await;
    let conflict = provider.has_conflict().await;
    let details = provider.get_conflict_details().await;
    let threshold = provider.coherence_threshold().await;

    assert!(active.is_none()); // No selection yet
    assert!(!broadcasting);
    assert!(!conflict);
    assert!(details.is_none());
    assert!((threshold - 0.8).abs() < 0.01);
}
```

**Inputs**: Fresh WorkspaceProviderImpl
**Expected**: All methods complete without deadlock on single-threaded runtime

#### TC-PERF-002: Async MetaCognitiveProvider Methods

**Type**: Unit
**Requirement Ref**: REQ-PERF-003

**Description**: Verify MetaCognitiveProvider async methods complete without blocking.

```rust
#[tokio::test(flavor = "current_thread")]
async fn test_meta_cognitive_provider_no_deadlock_single_thread() {
    let provider = MetaCognitiveProviderImpl::new();

    let ach = provider.acetylcholine().await;
    let freq = provider.monitoring_frequency().await;
    let scores = provider.get_recent_scores().await;

    assert!((ach - 0.001).abs() < 0.0001);
    assert!((freq - 1.0).abs() < 0.01);
    assert!(scores.is_empty());
}
```

**Inputs**: Fresh MetaCognitiveProviderImpl
**Expected**: All methods complete without deadlock

#### TC-PERF-003: Concurrent Provider Access

**Type**: Integration
**Requirement Ref**: REQ-PERF-001

**Description**: Verify providers handle concurrent access without deadlock.

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_provider_access() {
    let provider = Arc::new(WorkspaceProviderImpl::new());
    let mut handles = Vec::new();

    for _ in 0..100 {
        let p = Arc::clone(&provider);
        handles.push(tokio::spawn(async move {
            for _ in 0..10 {
                let _ = p.get_active_memory().await;
                let _ = p.is_broadcasting().await;
                let _ = p.coherence_threshold().await;
            }
        }));
    }

    // All tasks should complete within timeout
    let results = tokio::time::timeout(
        Duration::from_secs(5),
        futures::future::join_all(handles)
    ).await;

    assert!(results.is_ok(), "Concurrent access timed out - possible deadlock");
}
```

**Inputs**: 100 concurrent tasks, each making 10 calls
**Expected**: All complete within 5 seconds

#### TC-PERF-004: Wake Controller Latency

**Type**: Unit
**Requirement Ref**: REQ-PERF-004

**Description**: Verify wake latency stays under 100ms with parking_lot locks.

```rust
#[test]
fn test_wake_latency_with_parking_lot() {
    let controller = WakeController::new();
    controller.prepare_for_dream();

    let start = Instant::now();
    controller.signal_wake(WakeReason::ExternalQuery).unwrap();
    let latency = controller.complete_wake().unwrap();
    let total = start.elapsed();

    assert!(latency < Duration::from_millis(100));
    assert!(total < Duration::from_millis(100));
}
```

**Inputs**: Fresh WakeController
**Expected**: Wake completes in <100ms

#### TC-PERF-005: HashMap Capacity Efficiency

**Type**: Unit
**Requirement Ref**: REQ-PERF-005

**Description**: Verify HashMap pre-allocation reduces allocations.

```rust
#[test]
fn test_hashmap_capacity_aggregation() {
    let candidates: Vec<(Uuid, f32)> = (0..1000)
        .map(|_| (Uuid::new_v4(), rand::random()))
        .collect();

    // With capacity - should have 0 reallocations
    let mut scores: HashMap<Uuid, f32> = HashMap::with_capacity(candidates.len());
    for (id, score) in &candidates {
        scores.insert(*id, *score);
    }

    assert_eq!(scores.len(), 1000);
    assert!(scores.capacity() >= 1000);
}
```

**Inputs**: 1000 candidates
**Expected**: Single allocation, no rehashing

#### TC-PERF-006: No block_on in Codebase Grep Test

**Type**: CI Gate
**Requirement Ref**: REQ-PERF-001

**Description**: Verify no block_on() exists in non-test async code paths.

```bash
# CI script
#!/bin/bash
# Fail if block_on found in non-test source files
count=$(grep -r "block_on" crates/*/src --include="*.rs" | grep -v "_test.rs" | grep -v "/tests/" | wc -l)
if [ "$count" -gt 0 ]; then
    echo "ERROR: Found $count instances of block_on() in source files"
    grep -r "block_on" crates/*/src --include="*.rs" | grep -v "_test.rs" | grep -v "/tests/"
    exit 1
fi
echo "OK: No block_on() found in source files"
```

**Inputs**: Full codebase
**Expected**: Exit 0 (no block_on in source)

---

## Implementation Notes

### Phase 1: Critical Fixes (REQ-PERF-001 through REQ-PERF-003)

**Estimated Effort**: 4-8 hours

1. Update `WorkspaceProvider` trait in `gwt_traits.rs`:
   - Convert 5 sync methods to async
   - Ensure `#[async_trait]` attribute present

2. Update `WorkspaceProviderImpl` in `gwt_providers.rs`:
   - Remove `futures::executor::block_on()` from 5 methods
   - Use `.await` for lock acquisition

3. Update `MetaCognitiveProvider` trait:
   - Convert 3 sync methods to async

4. Update `MetaCognitiveProviderImpl`:
   - Remove `block_on()` from 3 methods
   - Use `.await` for lock acquisition

5. Update all trait consumers:
   - Search for `WorkspaceProvider` and `MetaCognitiveProvider` usage
   - Add `.await` to method calls
   - Ensure calling contexts are async

### Phase 2: Optimization (REQ-PERF-004, REQ-PERF-005)

**Estimated Effort**: 2-4 hours

1. Add `parking_lot` to `Cargo.toml` dependencies

2. Update `WakeController`:
   - Replace `std::sync::RwLock` with `parking_lot::RwLock`
   - Remove `.expect("Lock poisoned")` calls (parking_lot doesn't poison)

3. Update HashMap allocations:
   - Audit all `HashMap::new()` calls
   - Add capacity hints where size is known

### Verification Checklist

Post-implementation verification:

- [ ] `cargo test --all` passes
- [ ] No `block_on()` in non-test source files (grep check)
- [ ] TC-PERF-001 through TC-PERF-006 pass
- [ ] Single-threaded runtime tests pass
- [ ] No lock poisoning panics possible (parking_lot)
- [ ] Wake latency < 100ms verified

---

## Appendix A: Constitution Reference

### AP-08: No Sync I/O in Async Context

From `docs2/constitution.yaml`:

```yaml
forbidden:
  AP-08: "No sync I/O in async context"
```

This rule prohibits any blocking I/O or synchronization primitives in async code paths. The `block_on()` function explicitly violates this by blocking the current thread to wait for an async operation.

### Performance Budgets

From `docs2/constitution.yaml`:

```yaml
perf:
  latency:
    dream_wake: "<100ms"
```

The WakeController must complete wake transitions within 100ms. Using efficient locks (parking_lot) helps ensure this budget is met.

---

## Appendix B: Deadlock Scenario Illustration

### Single-Threaded Runtime Deadlock

```
Thread: tokio-runtime-worker

1. MCP Handler calls workspace_provider.get_active_memory()
2. get_active_memory() calls block_on(self.workspace.read())
3. block_on() parks the executor thread waiting for the read lock
4. BUT: The executor thread IS the only thread that can make progress
5. If any writer is pending, it can never run to release
6. DEADLOCK: Thread parked forever

Timeline:
t0: Handler enters get_active_memory()
t1: block_on() starts polling self.workspace.read()
t2: Poll returns Pending (writer holding lock)
t3: block_on() blocks thread waiting
t4: Writer needs executor to progress... but executor is blocked
t5: DEADLOCK
```

### Fixed Async Pattern

```
Thread: tokio-runtime-worker

1. MCP Handler calls workspace_provider.get_active_memory().await
2. get_active_memory() calls self.workspace.read().await
3. .await returns Pending, yields control back to executor
4. Executor runs other tasks, including the writer
5. Writer completes, releases lock
6. Executor polls get_active_memory() again
7. Lock acquired, method completes
8. No deadlock!

Timeline:
t0: Handler enters get_active_memory()
t1: .await on self.workspace.read()
t2: Poll returns Pending, control yields to executor
t3: Executor runs writer task
t4: Writer completes, signals waiters
t5: Executor polls read() again, lock acquired
t6: Method completes successfully
```

---

**END OF SPECIFICATION**
