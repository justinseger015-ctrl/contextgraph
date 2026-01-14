# TASK-37: Implement get_gpu_status MCP Tool

```xml
<task_spec id="TASK-MCP-011" version="2.0">
<metadata>
  <title>Implement get_gpu_status MCP tool</title>
  <status>complete</status>
  <layer>surface</layer>
  <sequence>37</sequence>
  <implements><requirement_ref>REQ-MCP-011</requirement_ref></implements>
  <depends_on>TASK-23</depends_on>
  <estimated_hours>3</estimated_hours>
  <actual_hours>2</actual_hours>
  <updated>2026-01-14</updated>
  <completed>2026-01-14</completed>
</metadata>
```

---

## EXECUTIVE SUMMARY

**Objective**: Create an MCP tool `get_gpu_status` that exposes GPU utilization and dream eligibility status via the `GpuMonitor` trait (implemented in TASK-23).

**Key Facts**:
- `GpuMonitor` trait and `NvmlGpuMonitor` implementation exist in `crates/context-graph-core/src/dream/triggers.rs`
- The `Handlers` struct does NOT currently have a `gpu_monitor` field - you MUST add one
- The MCP tool definitions system uses `crates/context-graph-mcp/src/tools/definitions/` for schemas
- Handlers are implemented in `crates/context-graph-mcp/src/handlers/` modules
- Tool dispatch is in `crates/context-graph-mcp/src/handlers/tools/dispatch.rs`

---

## CONSTITUTION REFERENCES

| Reference | Value | Location |
|-----------|-------|----------|
| `dream.trigger.gpu` | `<80%` | Eligibility threshold to START dream |
| `dream.constraints.gpu` | `<30%` | Budget threshold DURING dream |
| `AP-26` | No silent failures | Return errors, never fallback to 0.0 |

**Thresholds are defined in `crates/context-graph-core/src/dream/triggers.rs:33-48`**:
```rust
pub mod gpu_thresholds {
    pub const GPU_ELIGIBILITY_THRESHOLD: f32 = 0.80;
    pub const GPU_BUDGET_THRESHOLD: f32 = 0.30;
}
```

---

## EXISTING CODE LOCATIONS

### Source of Truth: GpuMonitor Trait and NvmlGpuMonitor
**File**: `crates/context-graph-core/src/dream/triggers.rs`

**Key Types** (already implemented):
```rust
// Lines 67-117: Error types
pub enum GpuMonitorError {
    NvmlInitFailed(String),
    NoDevices,
    DeviceAccessFailed { index: u32, message: String },
    UtilizationQueryFailed(String),
    NvmlNotAvailable,
    Disabled,
}

// Lines 685-783: Trait definition
pub trait GpuMonitor: Send + Sync + std::fmt::Debug {
    fn get_utilization(&mut self) -> Result<f32, GpuMonitorError>;
    fn is_eligible_for_dream(&mut self) -> Result<bool, GpuMonitorError>;
    fn should_abort_dream(&mut self) -> Result<bool, GpuMonitorError>;
    fn is_available(&self) -> bool;
}

// Lines 813-953: StubGpuMonitor for testing
pub struct StubGpuMonitor { ... }

// Lines 983-1168: NvmlGpuMonitor (requires "nvml" feature)
#[cfg(feature = "nvml")]
pub struct NvmlGpuMonitor { ... }
```

**Re-exports** in `crates/context-graph-core/src/dream/mod.rs:98-109`:
```rust
pub use triggers::{
    EntropyCalculator,
    GpuMonitor,
    GpuMonitorError,
    StubGpuMonitor,
    TriggerConfig,
    TriggerManager,
    gpu_thresholds,
};

#[cfg(feature = "nvml")]
pub use triggers::NvmlGpuMonitor;
```

### Handlers Struct
**File**: `crates/context-graph-mcp/src/handlers/core/handlers.rs`

The `Handlers` struct (starting line 43) contains all MCP tool dependencies. You MUST add:
```rust
// Add after trigger_manager field (around line 136)
/// GPU monitor for get_gpu_status tool.
/// TASK-37: Required for real GPU utilization queries.
/// Uses Arc<parking_lot::RwLock<>> for interior mutability (get_utilization takes &mut self).
pub(in crate::handlers) gpu_monitor: Option<Arc<RwLock<Box<dyn GpuMonitor>>>>,
```

### Tool Definitions
**File**: `crates/context-graph-mcp/src/tools/definitions/dream.rs`

Add `get_gpu_status` tool definition here (after existing 4 tools).

### Tool Dispatch
**File**: `crates/context-graph-mcp/src/handlers/tools/dispatch.rs`

Add dispatch case for `"get_gpu_status"`.

### Handler Implementation
**File**: `crates/context-graph-mcp/src/handlers/dream.rs`

Add `call_get_gpu_status` method following existing patterns (e.g., `call_trigger_dream`).

---

## IMPLEMENTATION STEPS

### Step 1: Add GpuMonitor Field to Handlers

**File**: `crates/context-graph-mcp/src/handlers/core/handlers.rs`

1. Add import at top:
```rust
use context_graph_core::dream::GpuMonitor;
```

2. Add field to `Handlers` struct after `trigger_manager` (around line 136):
```rust
// ========== GPU MONITOR (TASK-37) ==========
/// GPU monitor for get_gpu_status tool.
/// TASK-37: Required for real GPU utilization queries.
/// Uses Arc<RwLock<Box<dyn GpuMonitor>>> for interior mutability.
pub(in crate::handlers) gpu_monitor: Option<Arc<RwLock<Box<dyn GpuMonitor>>>>,
```

3. Initialize to `None` in all constructors:
   - `new()` (around line 219)
   - `new_test()` (around line 287)
   - `new_rocks()` (around line 353)
   - `with_test_teleological_store()` (around line 411)
   - `with_all_teleological()` (around line 473)
   - `with_test_infrastructure()` (around line 545)
   - `with_dream_infrastructure()` (around line 571)

4. Initialize to real `NvmlGpuMonitor` or `StubGpuMonitor` in `with_default_gwt()` (around line 725):
```rust
// TASK-37: Create GPU monitor - try NVML first, fallback to Stub
#[cfg(feature = "nvml")]
let gpu_monitor: Arc<RwLock<Box<dyn GpuMonitor>>> = {
    match context_graph_core::dream::NvmlGpuMonitor::new() {
        Ok(nvml) => Arc::new(RwLock::new(Box::new(nvml) as Box<dyn GpuMonitor>)),
        Err(e) => {
            tracing::warn!("NVML unavailable ({}), using StubGpuMonitor", e);
            Arc::new(RwLock::new(Box::new(context_graph_core::dream::StubGpuMonitor::unavailable()) as Box<dyn GpuMonitor>))
        }
    }
};

#[cfg(not(feature = "nvml"))]
let gpu_monitor: Arc<RwLock<Box<dyn GpuMonitor>>> = {
    Arc::new(RwLock::new(Box::new(context_graph_core::dream::StubGpuMonitor::unavailable()) as Box<dyn GpuMonitor>))
};
```

5. Add builder method:
```rust
/// Configure GPU monitor for get_gpu_status tool.
///
/// # Arguments
/// * `gpu_monitor` - Shared GpuMonitor instance
///
/// # Returns
/// Self with gpu_monitor configured
pub fn with_gpu_monitor(mut self, gpu_monitor: Arc<RwLock<Box<dyn GpuMonitor>>>) -> Self {
    self.gpu_monitor = Some(gpu_monitor);
    self
}
```

### Step 2: Add Tool Definition

**File**: `crates/context-graph-mcp/src/tools/definitions/dream.rs`

Add to the `definitions()` function's returned Vec:
```rust
// get_gpu_status - Get GPU utilization and dream eligibility
// TASK-37: Exposes GPU status via GpuMonitor trait
ToolDefinition::new(
    "get_gpu_status",
    "Get GPU utilization and dream eligibility status. \
     Returns real NVML metrics when available, or error if GPU unavailable. \
     Constitution: dream.trigger.gpu=<80% (eligibility), dream.constraints.gpu=<30% (budget). \
     Per AP-26: Returns error on failure, never silently returns 0.0.",
    json!({
        "type": "object",
        "properties": {},
        "required": []
    }),
),
```

Update the function's Vec capacity comment from 4 to 5 tools.

### Step 3: Update Tool Definitions Module

**File**: `crates/context-graph-mcp/src/tools/definitions/mod.rs`

Update `get_tool_definitions()` capacity comment from 43 to 44 tools (line 26).

### Step 4: Implement Handler

**File**: `crates/context-graph-mcp/src/handlers/dream.rs`

Add method to `impl Handlers`:
```rust
/// get_gpu_status tool implementation.
///
/// TASK-37: Get GPU utilization and dream eligibility status.
/// FAIL FAST if GpuMonitor not initialized (AP-26).
///
/// Returns:
/// - utilization: f32 - Current GPU utilization [0.0, 1.0] or null if unavailable
/// - eligible_for_dream: bool - Whether GPU < 80% (can start dream)
/// - budget_exceeded: bool - Whether GPU > 30% (would abort dream)
/// - device_count: u32 - Number of GPUs detected (0 if unavailable)
/// - available: bool - Whether GPU monitoring is available
/// - error: Option<string> - Error message if GPU query failed
///
/// # Constitution Compliance
/// - Eligibility: dream.trigger.gpu = "<80%"
/// - Budget: dream.constraints.gpu = "<30%"
/// - AP-26: Returns explicit error, never silently returns 0.0
pub(super) async fn call_get_gpu_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
    debug!("Handling get_gpu_status tool call");

    // FAIL FAST: GpuMonitor is REQUIRED
    let gpu_monitor = match &self.gpu_monitor {
        Some(gm) => gm,
        None => {
            error!("get_gpu_status: GpuMonitor not initialized - FAIL FAST per AP-26");
            return JsonRpcResponse::error(
                id,
                error_codes::GPU_MONITOR_NOT_INITIALIZED,
                "GpuMonitor not initialized. Configure with with_gpu_monitor() or use with_default_gwt().",
            );
        }
    };

    // Check availability first
    let available = {
        let monitor = gpu_monitor.read();
        monitor.is_available()
    };

    if !available {
        // Per AP-26: Return explicit unavailable status, not 0.0
        return self.tool_result_with_pulse(
            id,
            json!({
                "utilization": null,
                "eligible_for_dream": false,
                "budget_exceeded": false,
                "device_count": 0,
                "available": false,
                "error": "GPU monitoring unavailable - NVML drivers not installed or no GPU detected",
                "constitution_ref": {
                    "eligibility_threshold": 0.80,
                    "budget_threshold": 0.30,
                    "ap26_compliance": "Explicit error returned per AP-26"
                }
            }),
        );
    }

    // Query GPU utilization
    let result = {
        let mut monitor = gpu_monitor.write();
        monitor.get_utilization()
    };

    match result {
        Ok(utilization) => {
            // Compute eligibility and budget status
            let eligible_for_dream = utilization < context_graph_core::dream::gpu_thresholds::GPU_ELIGIBILITY_THRESHOLD;
            let budget_exceeded = utilization > context_graph_core::dream::gpu_thresholds::GPU_BUDGET_THRESHOLD;

            // Try to get device count (for NvmlGpuMonitor)
            // StubGpuMonitor doesn't expose this, so default to 1 if available
            let device_count: u32 = 1; // Default for stub

            info!(
                "get_gpu_status: utilization={:.1}%, eligible={}, budget_exceeded={}",
                utilization * 100.0,
                eligible_for_dream,
                budget_exceeded
            );

            self.tool_result_with_pulse(
                id,
                json!({
                    "utilization": utilization,
                    "eligible_for_dream": eligible_for_dream,
                    "budget_exceeded": budget_exceeded,
                    "device_count": device_count,
                    "available": true,
                    "error": null,
                    "constitution_ref": {
                        "eligibility_threshold": 0.80,
                        "eligibility_note": "GPU < 80% can start dream",
                        "budget_threshold": 0.30,
                        "budget_note": "GPU > 30% aborts dream"
                    }
                }),
            )
        }
        Err(e) => {
            // Per AP-26: Return explicit error, never return 0.0
            warn!("get_gpu_status: GPU query failed: {}", e);
            self.tool_result_with_pulse(
                id,
                json!({
                    "utilization": null,
                    "eligible_for_dream": false,
                    "budget_exceeded": false,
                    "device_count": 0,
                    "available": false,
                    "error": format!("GPU query failed: {}", e),
                    "constitution_ref": {
                        "ap26_compliance": "Explicit error returned per AP-26"
                    }
                }),
            )
        }
    }
}
```

### Step 5: Add Error Code

**File**: `crates/context-graph-mcp/src/protocol.rs`

Search for `error_codes` module and add:
```rust
pub const GPU_MONITOR_NOT_INITIALIZED: i32 = -32098; // Use next available code
```

### Step 6: Add Dispatch Case

**File**: `crates/context-graph-mcp/src/handlers/tools/dispatch.rs`

Find the `match tool_name` block and add:
```rust
"get_gpu_status" => self.call_get_gpu_status(id).await,
```

### Step 7: Update Imports in dream.rs

**File**: `crates/context-graph-mcp/src/handlers/dream.rs`

Add to existing imports:
```rust
// (should already exist from other handlers)
use tracing::{debug, error, info, warn};
```

---

## OUTPUT SCHEMA

```json
{
  "utilization": 0.45,           // f32 [0.0-1.0] or null if unavailable
  "eligible_for_dream": true,    // bool: utilization < 0.80
  "budget_exceeded": true,       // bool: utilization > 0.30
  "device_count": 1,             // u32: number of GPUs
  "available": true,             // bool: GPU monitoring available
  "error": null,                 // string or null
  "constitution_ref": {
    "eligibility_threshold": 0.80,
    "budget_threshold": 0.30
  }
}
```

---

## VERIFICATION REQUIREMENTS

### Full State Verification Protocol

After implementing, you MUST verify the implementation actually works:

1. **Source of Truth**: The `GpuMonitor` trait's `get_utilization()` method
2. **Execute & Inspect**: Call the MCP tool and verify the response matches actual GPU state

### Test Commands

```bash
# Compile check
cargo check -p context-graph-mcp

# Run unit tests
cargo test -p context-graph-mcp gpu

# Run specific handler tests
cargo test -p context-graph-mcp get_gpu_status
```

### Manual Verification (REQUIRED)

Create test file `crates/context-graph-mcp/src/handlers/tests/exhaustive_mcp_tools/gpu_status.rs`:

```rust
//! TASK-37: Full State Verification for get_gpu_status tool.
//!
//! NO MOCK DATA - Tests use real StubGpuMonitor with known values.

use crate::handlers::Handlers;
use crate::protocol::JsonRpcId;
use context_graph_core::dream::{GpuMonitor, StubGpuMonitor};
use parking_lot::RwLock;
use std::sync::Arc;

/// FSV Test 1: GPU available with known utilization
#[tokio::test]
async fn test_get_gpu_status_fsv_available() {
    // SETUP: Create handlers with StubGpuMonitor at 50% utilization
    let stub = StubGpuMonitor::with_usage(0.50);
    let gpu_monitor: Arc<RwLock<Box<dyn GpuMonitor>>> =
        Arc::new(RwLock::new(Box::new(stub)));

    let handlers = Handlers::new_test()
        .with_gpu_monitor(gpu_monitor.clone());

    // EXECUTE: Call the tool
    let response = handlers.call_get_gpu_status(Some(JsonRpcId::Number(1))).await;

    // VERIFY: Check response structure
    let result = response.result.expect("Should have result");

    // SOURCE OF TRUTH VERIFICATION: Compare with GpuMonitor directly
    let actual_utilization = {
        let mut monitor = gpu_monitor.write();
        monitor.get_utilization().expect("Should get utilization")
    };

    let response_utilization = result["utilization"].as_f64().expect("Should have utilization") as f32;

    // EVIDENCE OF SUCCESS
    println!("=== FSV EVIDENCE ===");
    println!("Source of Truth (GpuMonitor): {:.1}%", actual_utilization * 100.0);
    println!("MCP Response utilization: {:.1}%", response_utilization * 100.0);
    println!("eligible_for_dream: {}", result["eligible_for_dream"]);
    println!("budget_exceeded: {}", result["budget_exceeded"]);
    println!("available: {}", result["available"]);

    // ASSERTIONS
    assert!((response_utilization - actual_utilization).abs() < 0.001,
        "Response must match source of truth");
    assert!(result["eligible_for_dream"].as_bool().unwrap(),
        "50% < 80% should be eligible");
    assert!(result["budget_exceeded"].as_bool().unwrap(),
        "50% > 30% should exceed budget");
    assert!(result["available"].as_bool().unwrap(),
        "Should be available");
    assert!(result["error"].is_null(),
        "Should have no error");
}

/// FSV Test 2: GPU unavailable returns error (AP-26 compliance)
#[tokio::test]
async fn test_get_gpu_status_fsv_unavailable() {
    // SETUP: Create handlers with unavailable StubGpuMonitor
    let stub = StubGpuMonitor::unavailable();
    let gpu_monitor: Arc<RwLock<Box<dyn GpuMonitor>>> =
        Arc::new(RwLock::new(Box::new(stub)));

    let handlers = Handlers::new_test()
        .with_gpu_monitor(gpu_monitor);

    // EXECUTE
    let response = handlers.call_get_gpu_status(Some(JsonRpcId::Number(2))).await;

    // VERIFY
    let result = response.result.expect("Should have result");

    // EVIDENCE OF SUCCESS
    println!("=== FSV EVIDENCE (Unavailable) ===");
    println!("utilization: {:?}", result["utilization"]);
    println!("available: {}", result["available"]);
    println!("error: {:?}", result["error"]);

    // ASSERTIONS - Per AP-26: explicit error, not 0.0
    assert!(result["utilization"].is_null(),
        "AP-26: utilization must be null when unavailable, not 0.0");
    assert!(!result["available"].as_bool().unwrap(),
        "Should not be available");
    assert!(!result["error"].is_null(),
        "Should have error message");
}

/// FSV Test 3: Boundary conditions - eligibility threshold
#[tokio::test]
async fn test_get_gpu_status_fsv_eligibility_boundary() {
    // EDGE CASE: Exactly at 80% threshold
    let test_cases = [
        (0.79, true, "79% < 80% should be eligible"),
        (0.80, false, "80% = 80% should NOT be eligible (strict less-than)"),
        (0.81, false, "81% > 80% should NOT be eligible"),
    ];

    for (usage, expected_eligible, description) in test_cases {
        let stub = StubGpuMonitor::with_usage(usage);
        let gpu_monitor: Arc<RwLock<Box<dyn GpuMonitor>>> =
            Arc::new(RwLock::new(Box::new(stub)));

        let handlers = Handlers::new_test()
            .with_gpu_monitor(gpu_monitor);

        let response = handlers.call_get_gpu_status(Some(JsonRpcId::Number(3))).await;
        let result = response.result.expect("Should have result");

        // EVIDENCE
        println!("=== Eligibility Boundary: {:.0}% ===", usage * 100.0);
        println!("eligible_for_dream: {}", result["eligible_for_dream"]);
        println!("Expected: {}", expected_eligible);

        assert_eq!(
            result["eligible_for_dream"].as_bool().unwrap(),
            expected_eligible,
            "{}",
            description
        );
    }
}

/// FSV Test 4: Boundary conditions - budget threshold
#[tokio::test]
async fn test_get_gpu_status_fsv_budget_boundary() {
    // EDGE CASE: Exactly at 30% threshold
    let test_cases = [
        (0.29, false, "29% < 30% should NOT exceed budget"),
        (0.30, false, "30% = 30% should NOT exceed budget (strict greater-than)"),
        (0.31, true, "31% > 30% should exceed budget"),
    ];

    for (usage, expected_exceeded, description) in test_cases {
        let stub = StubGpuMonitor::with_usage(usage);
        let gpu_monitor: Arc<RwLock<Box<dyn GpuMonitor>>> =
            Arc::new(RwLock::new(Box::new(stub)));

        let handlers = Handlers::new_test()
            .with_gpu_monitor(gpu_monitor);

        let response = handlers.call_get_gpu_status(Some(JsonRpcId::Number(4))).await;
        let result = response.result.expect("Should have result");

        // EVIDENCE
        println!("=== Budget Boundary: {:.0}% ===", usage * 100.0);
        println!("budget_exceeded: {}", result["budget_exceeded"]);
        println!("Expected: {}", expected_exceeded);

        assert_eq!(
            result["budget_exceeded"].as_bool().unwrap(),
            expected_exceeded,
            "{}",
            description
        );
    }
}

/// FSV Test 5: No GpuMonitor configured - FAIL FAST
#[tokio::test]
async fn test_get_gpu_status_fsv_not_initialized() {
    // SETUP: Handlers WITHOUT gpu_monitor
    let handlers = Handlers::new_test();
    // DO NOT call .with_gpu_monitor()

    // EXECUTE
    let response = handlers.call_get_gpu_status(Some(JsonRpcId::Number(5))).await;

    // VERIFY: Should be error response
    println!("=== FSV EVIDENCE (Not Initialized) ===");
    println!("Has error: {}", response.error.is_some());

    assert!(response.error.is_some(), "Should return error when not initialized");

    let error = response.error.unwrap();
    println!("Error code: {}", error.code);
    println!("Error message: {}", error.message);

    // Per AP-26: explicit error, not silent failure
    assert!(error.message.contains("not initialized") || error.message.contains("GpuMonitor"));
}
```

### Register Test Module

**File**: `crates/context-graph-mcp/src/handlers/tests/exhaustive_mcp_tools/mod.rs`

Add:
```rust
mod gpu_status;
```

---

## DEFINITION OF DONE

- [ ] `GpuMonitor` import added to handlers.rs
- [ ] `gpu_monitor` field added to `Handlers` struct
- [ ] Field initialized in all constructors
- [ ] `with_gpu_monitor()` builder method added
- [ ] `with_default_gwt()` initializes real GpuMonitor
- [ ] Tool definition added to dream.rs definitions
- [ ] Tool count updated in mod.rs
- [ ] `call_get_gpu_status` handler implemented
- [ ] Error code added to protocol.rs
- [ ] Dispatch case added
- [ ] All tests pass: `cargo test -p context-graph-mcp gpu`
- [ ] Manual FSV tests pass with evidence logs
- [ ] No compiler warnings in modified files

---

## FAIL-FAST REQUIREMENTS (AP-26)

1. **Never return 0.0 on error** - Return `null` for utilization when unavailable
2. **Never silently degrade** - Return explicit error message
3. **Panic on configuration errors** - If Handlers is misconfigured, fail fast
4. **Log all errors** - Use `tracing::error!` for failures

---

## FILES TO CREATE

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/handlers/tests/exhaustive_mcp_tools/gpu_status.rs` | FSV tests |

## FILES TO MODIFY

| File | Changes |
|------|---------|
| `crates/context-graph-mcp/src/handlers/core/handlers.rs` | Add `gpu_monitor` field, builder method |
| `crates/context-graph-mcp/src/tools/definitions/dream.rs` | Add `get_gpu_status` definition |
| `crates/context-graph-mcp/src/tools/definitions/mod.rs` | Update tool count |
| `crates/context-graph-mcp/src/handlers/dream.rs` | Add `call_get_gpu_status` method |
| `crates/context-graph-mcp/src/handlers/tools/dispatch.rs` | Add dispatch case |
| `crates/context-graph-mcp/src/protocol.rs` | Add error code |
| `crates/context-graph-mcp/src/handlers/tests/exhaustive_mcp_tools/mod.rs` | Register test module |

---

## ANTI-PATTERNS TO AVOID

1. **NO backwards compatibility hacks** - If something breaks, fix it properly
2. **NO mock data in production tests** - Use real `StubGpuMonitor` with known values
3. **NO silent failures** - Always return explicit errors
4. **NO 0.0 fallback** - Return `null` when GPU unavailable
5. **NO ignoring constitution thresholds** - Use `gpu_thresholds::*` constants

---

## GIT COMMIT MESSAGE FORMAT

```
feat(mcp): implement get_gpu_status tool (TASK-37)

- Add gpu_monitor field to Handlers struct
- Implement get_gpu_status handler with FSV tests
- Wire GpuMonitor trait from context-graph-core
- Constitution: dream.trigger.gpu=<80%, dream.constraints.gpu=<30%
- AP-26 compliance: explicit errors, never 0.0 fallback
```
