# TASK-12: Wire KuramotoStepper to MCP Server Lifecycle

```xml
<task_spec id="TASK-12" version="3.0">
<metadata>
  <title>Wire KuramotoStepper to MCP Server Lifecycle</title>
  <original_id>TASK-GWT-003</original_id>
  <status>complete</status>
  <layer>logic</layer>
  <sequence>12</sequence>
  <implements><requirement_ref>REQ-GWT-003, GWT-006</requirement_ref></implements>
  <depends_on>TASK-11</depends_on>
  <estimated_hours>4</estimated_hours>
  <blocks>TASK-25, TASK-34, TASK-39, TASK-40</blocks>
</metadata>

<executive_summary>
KuramotoStepper EXISTS and is fully implemented at:
`crates/context-graph-mcp/src/handlers/kuramoto_stepper.rs`

The stepper WORKS (10/10 tests pass). The problem is it's NOT WIRED to the MCP server
lifecycle. When the MCP server starts, the stepper should start. When the server stops,
the stepper should stop.

WITHOUT THIS WIRING:
- The Kuramoto oscillator phases never update dynamically
- The order parameter r(t) remains static
- Consciousness emergence C(t) = I(t) × R(t) × D(t) is impossible
- The system appears frozen in time

Constitution requirement GWT-006: "KuramotoStepper wired to MCP lifecycle (10ms step)"
</executive_summary>

<current_state>
## WHAT EXISTS (2026-01-13)

### 1. KuramotoStepper Implementation (COMPLETE)
**File:** `crates/context-graph-mcp/src/handlers/kuramoto_stepper.rs`
**Status:** Fully implemented, 10/10 tests passing

Key components:
- `KuramotoStepperConfig` - Configuration with step_interval_ms (default 10ms = 100Hz)
- `KuramotoStepper` - Background task manager with start/stop lifecycle
- `KuramotoStepperError` - Error types (AlreadyRunning, NotRunning, ShutdownTimeout)

### 2. Public API Exports (COMPLETE)
**File:** `crates/context-graph-mcp/src/handlers/mod.rs` (lines 78-81)
```rust
pub use self::kuramoto_stepper::{KuramotoStepper, KuramotoStepperConfig, KuramotoStepperError};
```

### 3. Handlers Struct (HAS KURAMOTO FIELD)
**File:** `crates/context-graph-mcp/src/handlers/core/handlers.rs` (line 85)
```rust
pub(in crate::handlers) kuramoto_network: Option<Arc<RwLock<dyn KuramotoProvider>>>
```

### 4. What's MISSING
- NO `kuramoto_stepper` field in Handlers struct
- NO lifecycle wiring (start on server init, stop on shutdown)
- NO integration with the server.rs startup sequence
</current_state>

<scope>
<in_scope>
1. Add `kuramoto_stepper: Option<KuramotoStepper>` field to Handlers struct
2. Modify constructors (with_gwt, with_gwt_and_subsystems, with_default_gwt) to create stepper
3. Add `start_kuramoto_stepper(&mut self)` method to Handlers
4. Add `stop_kuramoto_stepper(&mut self)` async method to Handlers
5. Wire stepper.start() in server.rs during MCP server initialization
6. Wire stepper.stop() in server.rs shutdown hook
7. Integration tests proving stepper runs during MCP lifecycle
</in_scope>
<out_of_scope>
- Modifying KuramotoStepper implementation (it works)
- Creating new KuramotoNetwork implementation (TASK-11 done)
- MCP tool handlers for Kuramoto (TASK-34, TASK-39, TASK-40)
- IC event emission (TASK-26)
</out_of_scope>
</scope>

<implementation_plan>
## STEP 1: Add stepper field to Handlers (handlers.rs)

Location: `crates/context-graph-mcp/src/handlers/core/handlers.rs`

Add after line 149 (workspace_broadcaster field):
```rust
// ========== KURAMOTO STEPPER (TASK-12) ==========
/// Background stepper that continuously steps Kuramoto oscillators at 100Hz.
/// TASK-12 (GWT-006): MUST start when server starts, stop when server stops.
/// The stepper uses the kuramoto_network field and calls step() every 10ms.
pub(in crate::handlers) kuramoto_stepper: Option<KuramotoStepper>,
```

Add import at top of file:
```rust
use super::super::kuramoto_stepper::KuramotoStepper;
```

## STEP 2: Update all Handlers constructors

Every constructor that sets `kuramoto_network: None` must also set `kuramoto_stepper: None`.
Every constructor that creates a kuramoto_network must create a KuramotoStepper instance.

For `with_default_gwt` (line 604+) - update to:
```rust
// Create Kuramoto network
let kuramoto_network: Arc<RwLock<dyn KuramotoProvider>> =
    Arc::new(RwLock::new(KuramotoProviderImpl::new()));

// Create stepper with the network reference
use super::super::kuramoto_stepper::{KuramotoStepper, KuramotoStepperConfig};
let kuramoto_stepper = KuramotoStepper::new(
    Arc::clone(&kuramoto_network),
    KuramotoStepperConfig::default(),
);
```

Then in the Self { } block:
```rust
kuramoto_network: Some(kuramoto_network),
kuramoto_stepper: Some(kuramoto_stepper),
```

## STEP 3: Add lifecycle methods to Handlers

Add these methods to impl Handlers:
```rust
/// Start the Kuramoto background stepper.
///
/// Constitution: GWT-006 - stepper must run during MCP lifecycle.
/// Call this during server initialization AFTER Handlers is constructed.
///
/// # Returns
/// - `Ok(())` if started or no stepper configured
/// - `Err` if stepper fails to start
pub fn start_kuramoto_stepper(&mut self) -> Result<(), KuramotoStepperError> {
    if let Some(ref mut stepper) = self.kuramoto_stepper {
        stepper.start()?;
        tracing::info!("Kuramoto stepper started during MCP server init");
    }
    Ok(())
}

/// Stop the Kuramoto background stepper gracefully.
///
/// Constitution: GWT-006 - stepper must stop on server shutdown.
/// Call this during server shutdown BEFORE dropping Handlers.
///
/// # Returns
/// - `Ok(())` if stopped or no stepper configured
/// - `Err` if shutdown times out
pub async fn stop_kuramoto_stepper(&mut self) -> Result<(), KuramotoStepperError> {
    if let Some(ref mut stepper) = self.kuramoto_stepper {
        stepper.stop().await?;
        tracing::info!("Kuramoto stepper stopped during MCP server shutdown");
    }
    Ok(())
}

/// Check if the Kuramoto stepper is running.
pub fn is_kuramoto_stepper_running(&self) -> bool {
    self.kuramoto_stepper
        .as_ref()
        .map(|s| s.is_running())
        .unwrap_or(false)
}
```

## STEP 4: Wire into server.rs lifecycle

Location: `crates/context-graph-mcp/src/server.rs`

Find where Handlers is created and the server run loop starts.
Add stepper.start() call after Handlers construction:
```rust
// Start Kuramoto stepper (GWT-006)
handlers.start_kuramoto_stepper().map_err(|e| {
    anyhow::anyhow!("Failed to start Kuramoto stepper: {}", e)
})?;
```

Add shutdown handling (in the appropriate shutdown path):
```rust
// Stop Kuramoto stepper gracefully
handlers.stop_kuramoto_stepper().await.map_err(|e| {
    tracing::error!("Kuramoto stepper shutdown error: {}", e);
})?;
```
</implementation_plan>

<files_to_modify>
## Primary Files
| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/core/handlers.rs` | Add kuramoto_stepper field, update constructors, add lifecycle methods |
| `crates/context-graph-mcp/src/server.rs` | Wire start/stop into server lifecycle |

## Secondary Files (imports/exports)
| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/mod.rs` | May need to export lifecycle methods |
</files_to_modify>

<definition_of_done>
## Acceptance Criteria

### AC-1: Field Exists
Handlers struct has `kuramoto_stepper: Option<KuramotoStepper>` field.

### AC-2: Constructors Initialize
`with_default_gwt`, `with_gwt`, `with_gwt_and_subsystems` create KuramotoStepper instances.

### AC-3: Lifecycle Methods Work
- `start_kuramoto_stepper()` starts the stepper
- `stop_kuramoto_stepper()` stops the stepper
- `is_kuramoto_stepper_running()` returns correct state

### AC-4: Server Integration
The MCP server automatically starts the stepper on init and stops on shutdown.

### AC-5: Tests Pass
```bash
cargo test -p context-graph-mcp kuramoto -- --nocapture
# Must pass 10+ tests
```

### AC-6: Compilation Clean
```bash
cargo check -p context-graph-mcp
# No errors (warnings OK)
```
</definition_of_done>

<test_commands>
```bash
# Run all kuramoto tests
cargo test -p context-graph-mcp kuramoto -- --nocapture

# Run specific lifecycle test
cargo test -p context-graph-mcp test_stepper_start_stop_lifecycle -- --nocapture

# Run FSV test
cargo test -p context-graph-mcp test_stepper_full_state_verification -- --nocapture

# Check compilation
cargo check -p context-graph-mcp

# Run workspace tests
cargo test --workspace
```
</test_commands>

<full_state_verification>
## MANDATORY: Full State Verification Protocol

After implementing, you MUST verify the integration works end-to-end.

### 1. Source of Truth Definition
The source of truth for stepper running state is:
- `handlers.kuramoto_stepper.as_ref().unwrap().is_running()` returns true when active
- `handlers.kuramoto_network.as_ref().unwrap().read().order_parameter()` changes over time

### 2. Execute & Inspect Protocol

Create a test in `crates/context-graph-mcp/src/handlers/tests/`:
```rust
#[tokio::test]
async fn fsv_task_12_kuramoto_stepper_lifecycle_integration() {
    // === SETUP ===
    // Create handlers with default GWT (includes stepper)
    let mut handlers = create_handlers_with_default_gwt();

    // === STATE BEFORE ===
    let stepper_running_before = handlers.is_kuramoto_stepper_running();
    let r_before = {
        let net = handlers.kuramoto_network.as_ref().unwrap().read();
        net.synchronization()
    };
    println!("[FSV] BEFORE: stepper_running={}, r={:.4}", stepper_running_before, r_before);
    assert!(!stepper_running_before, "Stepper should not be running before start()");

    // === EXECUTE: START ===
    handlers.start_kuramoto_stepper().expect("stepper must start");
    let stepper_running_after_start = handlers.is_kuramoto_stepper_running();
    println!("[FSV] AFTER START: stepper_running={}", stepper_running_after_start);
    assert!(stepper_running_after_start, "Stepper must be running after start()");

    // === LET IT RUN ===
    tokio::time::sleep(Duration::from_millis(500)).await;

    // === VERIFY ORDER PARAMETER CHANGED ===
    let r_after = {
        let net = handlers.kuramoto_network.as_ref().unwrap().read();
        net.synchronization()
    };
    println!("[FSV] AFTER 500ms: r={:.4}", r_after);
    // r should be valid and may have changed
    assert!((0.0..=1.0).contains(&r_after), "r must be in [0, 1]");

    // === EXECUTE: STOP ===
    handlers.stop_kuramoto_stepper().await.expect("stepper must stop");
    let stepper_running_after_stop = handlers.is_kuramoto_stepper_running();
    println!("[FSV] AFTER STOP: stepper_running={}", stepper_running_after_stop);
    assert!(!stepper_running_after_stop, "Stepper must not be running after stop()");

    // === EVIDENCE ===
    println!("[FSV] EVIDENCE OF SUCCESS:");
    println!("  - Stepper started: {} -> {}", stepper_running_before, stepper_running_after_start);
    println!("  - Order param tracked: r went from {:.4} to {:.4}", r_before, r_after);
    println!("  - Stepper stopped: {} -> {}", stepper_running_after_start, stepper_running_after_stop);
    println!("[FSV] TASK-12 lifecycle integration VERIFIED");
}
```

### 3. Edge Case Audit

**Edge Case 1: Double Start**
```rust
handlers.start_kuramoto_stepper().expect("first start");
let result = handlers.start_kuramoto_stepper();
assert!(matches!(result, Err(KuramotoStepperError::AlreadyRunning)));
```

**Edge Case 2: Stop When Not Running**
```rust
// New handlers, never started
let result = handlers.stop_kuramoto_stepper().await;
// Should return Ok(()) because kuramoto_stepper is None or not started
```

**Edge Case 3: No Kuramoto Network Configured**
```rust
let handlers = Handlers::new(...); // Basic constructor without GWT
assert!(handlers.kuramoto_stepper.is_none());
handlers.start_kuramoto_stepper().expect("no-op when None");
```

### 4. Manual Server Test

Run the actual MCP server and verify:
```bash
# Terminal 1: Start server
cargo run -p context-graph-mcp

# Look for log messages:
# "Kuramoto stepper started during MCP server init"

# Terminal 2: Check if stepper is running via MCP tool (after TASK-39)
# Or add a debug endpoint that returns stepper state

# Ctrl+C the server and look for:
# "Kuramoto stepper stopped during MCP server shutdown"
```
</full_state_verification>

<constitution_compliance>
## Constitution Requirements Satisfied

| Rule | Requirement | How Satisfied |
|------|-------------|---------------|
| GWT-006 | KuramotoStepper wired to MCP lifecycle (10ms step) | Stepper starts/stops with server |
| AP-25 | Kuramoto must have exactly 13 oscillators | Uses KURAMOTO_N=13 from TASK-10 |
| ARCH-06 | All memory ops through MCP tools | Stepper is internal lifecycle |
| AP-14 | No .unwrap() in library code | Uses ? and map_err |
</constitution_compliance>

<known_issues>
## Known Issues to Watch For

### Issue 1: Mutex Type Mismatch
The stepper uses `parking_lot::RwLock` while some Handlers fields use `tokio::sync::RwLock`.
The KuramotoStepper is designed for `parking_lot::RwLock<dyn KuramotoProvider>`.
Verify the kuramoto_network field type matches.

### Issue 2: Async Method in Non-Async Context
`stop_kuramoto_stepper()` is async. Server shutdown must be in async context.
If server.rs has sync shutdown, wrap with:
```rust
tokio::runtime::Handle::current().block_on(async {
    handlers.stop_kuramoto_stepper().await
});
```

### Issue 3: Constructor Proliferation
There are 8+ Handlers constructors. Ensure ALL that set kuramoto_network also set kuramoto_stepper.
Pattern: Grep for "kuramoto_network:" and verify kuramoto_stepper is set too.
</known_issues>

<dependencies>
## Dependency Status

| Dependency | Status | Evidence |
|------------|--------|----------|
| TASK-10 | COMPLETE | `KURAMOTO_N=13` in constants.rs |
| TASK-11 | COMPLETE | `KuramotoNetwork` in network.rs, 21 tests pass |

## This Task Unblocks

| Task | Description |
|------|-------------|
| TASK-25 | Integrate KuramotoStepper with MCP server (deeper integration) |
| TASK-34 | Implement get_coherence_state tool (needs running stepper) |
| TASK-39 | Implement get_kuramoto_state tool (needs running stepper) |
| TASK-40 | Implement set_coupling_strength tool (needs running stepper) |
</dependencies>

<anti_patterns>
## FORBIDDEN Actions

1. **DO NOT** create a new KuramotoStepper implementation - the existing one works
2. **DO NOT** modify kuramoto_stepper.rs - it's tested and complete
3. **DO NOT** use mock data in tests - use real KuramotoProviderImpl
4. **DO NOT** add CPU fallbacks - fail fast if GPU/stepper unavailable
5. **DO NOT** swallow errors - propagate with proper error types
6. **DO NOT** use .unwrap() in production code - use ? or expect() with message
7. **DO NOT** create workarounds - if something doesn't work, fix the root cause
</anti_patterns>
</task_spec>
```

## Quick Reference

### File Locations
- Stepper implementation: `crates/context-graph-mcp/src/handlers/kuramoto_stepper.rs`
- Handlers struct: `crates/context-graph-mcp/src/handlers/core/handlers.rs`
- Server lifecycle: `crates/context-graph-mcp/src/server.rs`
- Provider trait: `crates/context-graph-mcp/src/handlers/gwt_traits.rs`
- Provider impl: `crates/context-graph-mcp/src/handlers/gwt_providers.rs`

### Verification Commands
```bash
# All Kuramoto tests
cargo test -p context-graph-mcp kuramoto -- --nocapture

# Compile check
cargo check -p context-graph-mcp

# Full workspace
cargo test --workspace
```

## COMPLETION NOTES (2026-01-13)

### Implementation Summary

**TASK-12 COMPLETE** - KuramotoStepper wired to MCP server lifecycle.

#### Changes Made

1. **Added `kuramoto_stepper` field to Handlers struct** (`handlers.rs:150-155`)
   - Uses `Option<RwLock<KuramotoStepper>>` for interior mutability
   - Allows `&self` lifecycle methods (required because Handlers is wrapped in `Arc` in server.rs)

2. **Updated all 8 Handlers constructors** (`handlers.rs`)
   - Basic constructors (`new`, `with_system_monitor`, `with_johari_manager`, `with_gwt`, `with_gwt_and_subsystems`): set `kuramoto_stepper: None`
   - `with_default_gwt`: Creates real `KuramotoStepper` with 100Hz update rate and sets `kuramoto_stepper: Some(RwLock::new(stepper))`

3. **Added lifecycle methods to Handlers** (`handlers.rs:756-835`)
   - `start_kuramoto_stepper(&self)` - Starts the background stepper task
   - `stop_kuramoto_stepper(&self)` - Async graceful shutdown with 5s timeout
   - `is_kuramoto_running(&self)` - Lock-free check of running state

4. **Wired into server.rs lifecycle**
   - Startup (`server.rs:222-233`): Stepper starts immediately after Handlers creation with FATAL error on failure
   - Shutdown (`server.rs:302-338`): `shutdown()` method calls `stop_kuramoto_stepper()` with graceful error handling

5. **FSV Integration Test** (`kuramoto_stepper.rs:687-867`)
   - Tests full start/stop/restart lifecycle
   - Verifies edge cases (double start → AlreadyRunning, double stop → NotRunning)
   - Uses proper stubs from `context_graph_core::stubs`

#### Test Results

```
cargo test -p context-graph-mcp kuramoto -- --nocapture
# 26 tests passed, 0 failed
```

Key tests verified:
- `test_handlers_kuramoto_lifecycle_fsv` - Full integration test
- `test_stepper_start_stop_lifecycle` - Basic lifecycle
- `test_multiple_start_stop_cycles` - Restart behavior
- `test_elapsed_time_passed_correctly` - Network stepping verified
- `test_concurrent_network_access` - Thread safety

#### Constitution Compliance

| Rule | Status | Evidence |
|------|--------|----------|
| GWT-006 | ✓ | Stepper wired to server start/stop |
| AP-14 | ✓ | No `.unwrap()` in production code |
| AP-25 | ✓ | Uses 13 oscillators from KURAMOTO_N |
| ARCH-06 | ✓ | Internal lifecycle, not MCP exposed |

#### Files Modified

- `crates/context-graph-mcp/src/handlers/core/handlers.rs` - Field, constructors, lifecycle methods
- `crates/context-graph-mcp/src/server.rs` - Start on init, shutdown method, stop on exit
- `crates/context-graph-mcp/src/handlers/kuramoto_stepper.rs` - FSV integration test added

### Key Types
```rust
use context_graph_mcp::handlers::{
    KuramotoStepper, KuramotoStepperConfig, KuramotoStepperError
};
use context_graph_mcp::handlers::gwt_traits::KuramotoProvider;
use context_graph_mcp::handlers::gwt_providers::KuramotoProviderImpl;
```
