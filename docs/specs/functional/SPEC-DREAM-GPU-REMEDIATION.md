# Functional Specification: Dream Domain GPU Monitoring Remediation

## Metadata

| Field | Value |
|-------|-------|
| **Spec ID** | SPEC-DREAM-002 |
| **Title** | Dream Domain GPU Monitoring Remediation |
| **Status** | Draft |
| **Version** | 1.0.0 |
| **Created** | 2026-01-12 |
| **Author** | Specification Agent |
| **Related Specs** | SPEC-IDENTITY-001 (Dream Triggers), SPEC-GWT-REMEDIATION |
| **Source Issues** | ISS-007, ISS-014 |
| **Constitution Version** | v5.0.0 |

---

## Overview

### Problem Statement

The Dream Domain's GPU monitoring system has two critical issues preventing proper dream cycle management:

1. **ISS-007 (HIGH)**: `GpuMonitor` is a stub that always returns `0.0`, making GPU-based dream triggers and budget enforcement non-functional in production.

2. **ISS-014 (MEDIUM)**: Confusion between two GPU threshold values in the codebase - 30% and 80% - with unclear documentation about when each applies.

### Why This Matters

GPU monitoring serves two distinct purposes in the Dream Layer:

| Purpose | Threshold | Constitution Reference | Current State |
|---------|-----------|----------------------|---------------|
| **Dream Eligibility** | `<80%` | `dream.trigger.gpu` | Not implemented |
| **Dream Constraint** | `<30%` | `dream.constraints.gpu` | Hardcoded but relies on stub |

Without real GPU monitoring:
- Dreams cannot trigger based on GPU availability (when system is under 80% load)
- Dreams cannot abort when GPU usage exceeds 30% budget during execution
- Resource pressure detection is completely disabled
- Constitution requirements AP-26 (no silent failures) are violated

### Constitution Requirements

From `docs2/constitution.yaml`:

```yaml
dream:
  trigger:
    gpu: "<80%"  # Dream can START when GPU usage is below 80%

  constraints:
    gpu: "<30%"  # Dream must ABORT if GPU usage exceeds 30% during execution
```

---

## User Stories

### US-DREAM-GPU-001: System Operator Needs Real GPU Monitoring

**Priority**: Must-Have

**Narrative**:
> As the autonomous dream system,
> I want to monitor actual GPU utilization via NVML,
> So that I can make informed decisions about when to start dreams and when to abort them.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-001-01 | NVIDIA GPU is present with NVML drivers | `GpuMonitor::get_usage()` is called | Returns actual GPU utilization as f32 in [0.0, 1.0] |
| AC-001-02 | NVIDIA GPU is present | `GpuMonitor::is_available()` is called | Returns `true` |
| AC-001-03 | No NVIDIA GPU present (or NVML unavailable) | `GpuMonitor::get_usage()` is called | Returns `Err(GpuMonitorError::NvmlUnavailable)` - NOT 0.0 |
| AC-001-04 | Multiple GPUs present | `GpuMonitor::get_usage()` is called | Returns maximum utilization across all GPUs |

### US-DREAM-GPU-002: Dream Controller Needs Eligibility Check

**Priority**: Must-Have

**Narrative**:
> As the dream trigger manager,
> I want to check if GPU usage is below 80%,
> So that I only start dreams when the system has GPU capacity available.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-002-01 | GPU at 75% utilization | `TriggerManager::check_triggers()` is called | GPU does NOT block dream eligibility |
| AC-002-02 | GPU at 85% utilization | `TriggerManager::check_triggers()` is called | GPU blocks dream eligibility (too busy) |
| AC-002-03 | GPU exactly at 80% | `TriggerManager::check_triggers()` is called | Dream NOT eligible (threshold is `<80%`, strict) |

### US-DREAM-GPU-003: Wake Controller Needs Budget Enforcement

**Priority**: Must-Have

**Narrative**:
> As the wake controller,
> I want to monitor GPU usage during dream execution,
> So that I can abort the dream if GPU usage exceeds 30%.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-003-01 | Dream running, GPU at 25% | `WakeController::check_gpu_budget()` is called | Returns `Ok(())`, dream continues |
| AC-003-02 | Dream running, GPU at 35% | `WakeController::check_gpu_budget()` is called | Returns `Err(GpuBudgetExceeded)`, wake signaled |
| AC-003-03 | Dream running, GPU exactly at 30% | `WakeController::check_gpu_budget()` is called | Returns `Ok(())` (threshold is `>30%`, not `>=30%`) |

---

## Requirements

### REQ-DREAM-001: GpuMonitor MUST Return Real GPU Values via NVML

**ID**: REQ-DREAM-001
**Priority**: CRITICAL
**Source**: ISS-007
**Constitution**: `stack.gpu`, `dream.constraints.gpu`

**Description**:
The `GpuMonitor` struct MUST use NVIDIA Management Library (NVML) to query actual GPU utilization instead of returning hardcoded `0.0`.

**Rationale**:
Without real GPU monitoring, the system cannot:
1. Determine if GPU is available for dream processing
2. Enforce the 30% GPU budget during dreams
3. Trigger dreams based on GPU availability

**Implementation Requirements**:

1. Add `nvml-wrapper` crate as dependency (or use raw NVML FFI via `context-graph-cuda`)
2. Implement `NvmlGpuMonitor` struct with:
   - `fn new() -> Result<Self, GpuMonitorError>`
   - `fn get_usage(&self) -> Result<f32, GpuMonitorError>`
   - `fn is_available(&self) -> bool`
   - `fn device_count(&self) -> usize`
   - `fn get_device_usage(&self, device_idx: u32) -> Result<f32, GpuMonitorError>`

3. Query `nvmlDeviceGetUtilizationRates()` for GPU utilization
4. Return maximum utilization across all GPUs for multi-GPU systems
5. Cache NVML handle to avoid repeated initialization

**Error Handling (Fail-Fast per AP-26)**:

```rust
pub enum GpuMonitorError {
    /// NVML library not available - MUST error, not return 0.0
    NvmlUnavailable,
    /// Failed to initialize NVML
    NvmlInitFailed(String),
    /// No GPU devices found
    NoDevices,
    /// Failed to query specific device
    DeviceQueryFailed { device: u32, reason: String },
}
```

**NO WORKAROUNDS**: If NVML is unavailable:
- Do NOT return 0.0 (silent failure)
- Do NOT fall back to CPU estimation
- MUST return `Err(GpuMonitorError::NvmlUnavailable)` with clear error message

**Current Location**: `crates/context-graph-core/src/dream/triggers.rs:291-345`

**TODO to Remove**:
```rust
// TODO(FUTURE): Implement real GPU monitoring via NVML
// For now, return 0.0 (no GPU usage)
0.0
```

### REQ-DREAM-002: Clarify and Document GPU Threshold Semantics

**ID**: REQ-DREAM-002
**Priority**: HIGH
**Source**: ISS-014
**Constitution**: `dream.trigger.gpu`, `dream.constraints.gpu`

**Description**:
Clearly distinguish and document the two different GPU thresholds:

| Threshold | Value | Meaning | Usage |
|-----------|-------|---------|-------|
| **Eligibility** | 80% | GPU must be below 80% to START a dream | `TriggerManager` |
| **Budget** | 30% | GPU must stay below 30% DURING a dream | `WakeController` |

**Rationale**:
Current code conflates these thresholds:
- `GpuTriggerState.threshold` is 0.30 (correct for budget)
- Constitution `dream.trigger.gpu: "<80%"` is for eligibility (not implemented)
- Comments reference "30% vs 80%" confusion

**Implementation Requirements**:

1. Add `GPU_ELIGIBILITY_THRESHOLD: f32 = 0.80` constant to `dream/mod.rs::constants`
2. Update `TriggerManager` to check eligibility threshold (80%) for dream start
3. Keep `WakeController` using budget threshold (30%) for abort
4. Add doc comments clarifying:
   ```rust
   /// GPU eligibility threshold for dream START.
   /// Constitution: dream.trigger.gpu: "<80%"
   /// Dream can START when GPU < 80% (system has capacity)
   pub const GPU_ELIGIBILITY_THRESHOLD: f32 = 0.80;

   /// GPU budget threshold DURING dream.
   /// Constitution: dream.constraints.gpu: "<30%"
   /// Dream must ABORT if GPU > 30% (using too much)
   pub const MAX_GPU_USAGE: f32 = 0.30;
   ```

### REQ-DREAM-003: AMD GPU Fallback (Optional Enhancement)

**ID**: REQ-DREAM-003
**Priority**: LOW
**Source**: Best Practice
**Constitution**: None (enhancement)

**Description**:
Optionally support AMD GPUs via ROCm-SMI for systems without NVIDIA hardware.

**Implementation Notes**:
- Only attempt ROCm-SMI if NVML is unavailable
- If neither NVML nor ROCm available, return error (not 0.0)
- This is an OPTIONAL enhancement, not required for remediation

**Error Behavior**:
```rust
match (nvml_available, rocm_available) {
    (true, _) => use_nvml(),
    (false, true) => use_rocm(),
    (false, false) => Err(GpuMonitorError::NoGpuBackend),
}
```

---

## Edge Cases

### EC-DREAM-001: No GPU Present

**Related Requirement**: REQ-DREAM-001
**Scenario**: System has no dedicated GPU (e.g., development laptop)
**Expected Behavior**:
- `GpuMonitor::is_available()` returns `false`
- `GpuMonitor::get_usage()` returns `Err(GpuMonitorError::NoDevices)`
- `TriggerManager` skips GPU eligibility check (uses other triggers)
- `WakeController` skips GPU budget check (cannot enforce)
- System logs warning: "GPU monitoring unavailable - budget enforcement disabled"

### EC-DREAM-002: NVML Driver Not Installed

**Related Requirement**: REQ-DREAM-001
**Scenario**: NVIDIA GPU present but NVML drivers not installed
**Expected Behavior**:
- `NvmlGpuMonitor::new()` returns `Err(GpuMonitorError::NvmlInitFailed)`
- Same fallback behavior as EC-DREAM-001
- System logs error: "NVML initialization failed: {reason}"

### EC-DREAM-003: GPU Usage Exactly at Threshold

**Related Requirement**: REQ-DREAM-002
**Scenario**: GPU usage is exactly 30% or exactly 80%
**Expected Behavior**:
- At 30%: Dream continues (threshold is `>30%`, not `>=30%`)
- At 80%: Dream NOT eligible (threshold is `<80%`, strict inequality)
- These behaviors are tested and documented

### EC-DREAM-004: GPU Usage Spikes During Dream

**Related Requirement**: REQ-DREAM-002
**Scenario**: GPU was at 25% when dream started, spikes to 35% mid-dream
**Expected Behavior**:
- `WakeController::check_gpu_budget()` detects spike
- Wake signaled with `WakeReason::GpuOverBudget`
- Dream aborts within <100ms (Constitution requirement)
- Partial progress saved, dream can resume later

### EC-DREAM-005: Multi-GPU System

**Related Requirement**: REQ-DREAM-001
**Scenario**: System has 2+ GPUs with different utilization
**Expected Behavior**:
- `GpuMonitor` queries all devices
- Returns MAXIMUM utilization across all GPUs
- If any GPU exceeds threshold, condition is met
- Example: GPU0 at 20%, GPU1 at 40% -> returns 0.40

### EC-DREAM-006: NVML Query Fails Mid-Operation

**Related Requirement**: REQ-DREAM-001
**Scenario**: NVML was working but query fails during dream
**Expected Behavior**:
- Log error: "GPU query failed: {reason}"
- Return last known value with staleness flag
- If staleness > 5 seconds, treat as unavailable
- Do NOT silently return 0.0

---

## Error States

### ERR-DREAM-001: NVML Unavailable

**Condition**: NVML library cannot be loaded
**HTTP Code**: N/A (internal)
**Error Type**: `GpuMonitorError::NvmlUnavailable`
**User Message**: "GPU monitoring unavailable: NVML library not found"
**Recovery**: System operates without GPU monitoring; logs warning

### ERR-DREAM-002: NVML Initialization Failed

**Condition**: NVML library loaded but `nvmlInit()` fails
**HTTP Code**: N/A (internal)
**Error Type**: `GpuMonitorError::NvmlInitFailed(String)`
**User Message**: "GPU monitoring failed to initialize: {reason}"
**Recovery**: Same as ERR-DREAM-001

### ERR-DREAM-003: GPU Budget Exceeded

**Condition**: GPU usage > 30% during dream execution
**HTTP Code**: N/A (internal)
**Error Type**: `WakeError::GpuBudgetExceeded { usage, max }`
**User Message**: "Dream aborted: GPU usage {usage}% exceeded budget {max}%"
**Recovery**: Dream aborts, system returns to normal operation

### ERR-DREAM-004: Device Query Failed

**Condition**: Cannot query specific GPU device
**HTTP Code**: N/A (internal)
**Error Type**: `GpuMonitorError::DeviceQueryFailed { device, reason }`
**User Message**: "Failed to query GPU {device}: {reason}"
**Recovery**: Skip device, use other GPUs; if all fail, treat as unavailable

---

## Test Plan

### Unit Tests (Hardware-Conditional)

> **Note**: Per project requirements, NO MOCK DATA is used in tests. All GPU tests
> are conditional and only run on systems with real NVIDIA hardware. Tests are
> skipped with `#[ignore]` on systems without GPU support.

#### TC-DREAM-001: NvmlGpuMonitor Real Initialization

**Type**: Unit (Hardware-Conditional)
**Related Requirement**: REQ-DREAM-001
**Description**: Test `NvmlGpuMonitor::new()` with real NVML driver
**Preconditions**: NVIDIA GPU with NVML drivers installed
**Skip Condition**: `#[ignore = "Requires NVIDIA GPU with NVML"]` - skipped if no GPU
**Implementation**:
```rust
#[test]
#[ignore = "Requires NVIDIA GPU with NVML"]
fn test_nvml_real_initialization() {
    // Only runs on GPU systems, auto-skipped otherwise
    let monitor = NvmlGpuMonitor::new();
    assert!(monitor.is_ok(), "NVML should initialize on GPU system");
    assert!(monitor.unwrap().is_available());
}
```
**Expected**: Monitor initializes successfully on GPU systems; test skipped on non-GPU systems

#### TC-DREAM-002: NvmlGpuMonitor Graceful Error Without GPU

**Type**: Unit (Error Path)
**Related Requirement**: REQ-DREAM-001
**Description**: Verify error handling when NVML unavailable (real environment check)
**Preconditions**: System without NVIDIA GPU or NVML drivers
**Implementation**:
```rust
#[test]
fn test_nvml_unavailable_returns_error() {
    // This test validates error type when NVML unavailable
    // Runs on ALL systems - verifies error path on non-GPU, success path on GPU
    let result = NvmlGpuMonitor::new();
    match result {
        Ok(monitor) => {
            // On GPU systems: verify real values returned
            let usage = monitor.get_usage();
            assert!(usage.is_ok(), "GPU systems should return real usage");
            let val = usage.unwrap();
            assert!(val >= 0.0 && val <= 1.0, "Usage must be in [0.0, 1.0]");
        }
        Err(GpuMonitorError::NvmlUnavailable) => {
            // Expected on non-GPU systems - this is the correct behavior
        }
        Err(e) => panic!("Unexpected error type: {:?}", e),
    }
}
```
**Expected**: Returns `Err(NvmlUnavailable)` on non-GPU systems; returns real values on GPU systems

#### TC-DREAM-003: GPU Usage Query (Real Hardware)

**Type**: Unit (Hardware-Conditional)
**Related Requirement**: REQ-DREAM-001
**Description**: Test `get_usage()` returns real GPU utilization value
**Preconditions**: NVIDIA GPU with NVML drivers installed
**Skip Condition**: `#[ignore = "Requires NVIDIA GPU"]` - skipped if no GPU
**Implementation**:
```rust
#[test]
#[ignore = "Requires NVIDIA GPU"]
fn test_gpu_usage_returns_real_value() {
    let monitor = NvmlGpuMonitor::new().expect("NVML should initialize");
    let usage = monitor.get_usage().expect("Should get real usage");

    // Real GPU usage MUST be in valid range
    assert!(usage >= 0.0, "Usage cannot be negative");
    assert!(usage <= 1.0, "Usage cannot exceed 100%");

    // Verify it's not hardcoded by checking multiple samples
    let samples: Vec<f32> = (0..5)
        .map(|_| {
            std::thread::sleep(std::time::Duration::from_millis(100));
            monitor.get_usage().unwrap()
        })
        .collect();

    // At least verify we get valid readings (may or may not vary)
    for s in &samples {
        assert!(*s >= 0.0 && *s <= 1.0);
    }
}
```
**Expected**: `get_usage()` returns real value in [0.0, 1.0] from NVML

#### TC-DREAM-004: Multi-GPU Maximum Selection (Real Hardware)

**Type**: Unit (Hardware-Conditional)
**Related Requirement**: REQ-DREAM-001
**Description**: Test that maximum GPU usage is returned on multi-GPU systems
**Preconditions**: System with 2+ NVIDIA GPUs
**Skip Condition**: `#[ignore = "Requires multiple NVIDIA GPUs"]` - skipped if <2 GPUs
**Implementation**:
```rust
#[test]
#[ignore = "Requires multiple NVIDIA GPUs"]
fn test_multi_gpu_returns_maximum() {
    let monitor = NvmlGpuMonitor::new().expect("NVML should initialize");

    // Verify we have multiple GPUs
    let device_count = monitor.device_count();
    assert!(device_count >= 2, "Test requires 2+ GPUs, found {}", device_count);

    // Get individual device usages
    let individual: Vec<f32> = (0..device_count as u32)
        .filter_map(|i| monitor.get_device_usage(i).ok())
        .collect();

    // Get aggregated usage (should be maximum)
    let aggregated = monitor.get_usage().expect("Should get usage");
    let max_individual = individual.iter().cloned().fold(0.0_f32, f32::max);

    // Aggregated should equal or be very close to max individual
    assert!((aggregated - max_individual).abs() < 0.01,
        "Aggregated {} should equal max individual {}", aggregated, max_individual);
}
```
**Expected**: `get_usage()` returns maximum value across all GPUs

#### TC-DREAM-005: Eligibility Threshold 80%

**Type**: Unit
**Related Requirement**: REQ-DREAM-002
**Description**: Test dream eligibility at various GPU levels
**Test Cases**:
- GPU at 75%: Dream eligible
- GPU at 80%: Dream NOT eligible
- GPU at 85%: Dream NOT eligible

#### TC-DREAM-006: Budget Threshold 30%

**Type**: Unit
**Related Requirement**: REQ-DREAM-002
**Description**: Test GPU budget enforcement during dream
**Test Cases**:
- GPU at 25%: No abort
- GPU at 30%: No abort (exact threshold)
- GPU at 31%: Abort triggered

#### TC-DREAM-007: Error Returns Error (Not Zero)

**Type**: Unit (Error Path Validation)
**Related Requirement**: REQ-DREAM-001
**Description**: Verify that errors return Error type, never 0.0 (fail-fast per AP-26)
**Implementation**:
```rust
#[test]
fn test_error_returns_error_not_zero() {
    // Validates the API contract: errors MUST return Err, never Ok(0.0)
    // This runs on all systems and validates the type signature

    let result = NvmlGpuMonitor::new();
    match result {
        Ok(monitor) => {
            // On GPU systems: query an invalid device index
            let invalid_idx = u32::MAX;
            let err_result = monitor.get_device_usage(invalid_idx);
            assert!(err_result.is_err(), "Invalid device MUST return Err, not Ok(0.0)");
            match err_result {
                Err(GpuMonitorError::DeviceQueryFailed { .. }) => { /* correct */ }
                Err(e) => panic!("Expected DeviceQueryFailed, got {:?}", e),
                Ok(v) => panic!("Expected Err, got Ok({})", v),
            }
        }
        Err(_) => {
            // On non-GPU systems: the new() error validates fail-fast
            // This is the expected path - we don't get Ok(0.0), we get Err
        }
    }
}
```
**Expected**: Invalid operations return `Err(...)`, never silent `Ok(0.0)`

### Integration Tests

#### TC-DREAM-INT-001: Real NVML Integration

**Type**: Integration
**Related Requirement**: REQ-DREAM-001
**Description**: Test with actual NVML on GPU-equipped system
**Preconditions**: NVIDIA GPU with NVML drivers installed
**Steps**:
1. Create `NvmlGpuMonitor`
2. Call `get_usage()`
3. Verify value is in [0.0, 1.0]
4. Verify value changes over time (not hardcoded)
**Note**: Skip on systems without GPU

#### TC-DREAM-INT-002: Dream Cycle with GPU Monitoring

**Type**: Integration
**Related Requirement**: REQ-DREAM-001, REQ-DREAM-002
**Description**: Full dream cycle with real GPU monitoring
**Steps**:
1. Set up TriggerManager with real GpuMonitor
2. Verify GPU eligibility check (80% threshold)
3. Start dream cycle
4. Verify WakeController checks 30% budget
5. Complete cycle or abort on budget exceeded

#### TC-DREAM-INT-003: Graceful Degradation Without GPU

**Type**: Integration
**Related Requirement**: REQ-DREAM-001
**Description**: System operates correctly without GPU
**Preconditions**: No NVIDIA GPU available
**Steps**:
1. Create GpuMonitor
2. Verify `is_available()` returns `false`
3. Start dream cycle
4. Verify cycle completes without GPU checks
5. Verify warning logged about missing GPU

### Constitution Compliance Tests

#### TC-CONST-001: Dream Trigger GPU < 80%

**Type**: Constitution
**Related**: `dream.trigger.gpu`
**Description**: Verify 80% eligibility threshold per constitution
**Expected**: Dream eligible when GPU < 80%, not eligible when >= 80%

#### TC-CONST-002: Dream Constraint GPU < 30%

**Type**: Constitution
**Related**: `dream.constraints.gpu`
**Description**: Verify 30% budget threshold per constitution
**Expected**: Dream aborts when GPU > 30%, continues when <= 30%

#### TC-CONST-003: No Silent Failures (AP-26)

**Type**: Constitution
**Related**: `forbidden.AP-26`
**Description**: Verify GPU errors are not silently swallowed
**Expected**: Errors return Error type, never 0.0

---

## Dependency Graph

```
REQ-DREAM-001 (GpuMonitor NVML)
    |
    +-- REQ-DREAM-002 (Threshold Clarification)
    |       |
    |       +-- TriggerManager (80% eligibility)
    |       |
    |       +-- WakeController (30% budget)
    |
    +-- REQ-DREAM-003 (AMD ROCm - optional)
```

### Implementation Order

1. **REQ-DREAM-001**: Implement real NVML monitoring (foundation)
2. **REQ-DREAM-002**: Add eligibility threshold, clarify documentation
3. *REQ-DREAM-003*: (Optional) Add AMD ROCm fallback

### External Dependencies

| Dependency | Source | Required |
|------------|--------|----------|
| `nvml-wrapper` crate | crates.io | Yes |
| NVIDIA GPU | Hardware | No (graceful degradation) |
| NVML drivers | System | No (returns error if missing) |
| ROCm-SMI | AMD | No (optional enhancement) |

---

## Files to Modify

| File | Changes |
|------|---------|
| `crates/context-graph-core/src/dream/triggers.rs` | Replace `GpuMonitor` stub with `NvmlGpuMonitor` |
| `crates/context-graph-core/src/dream/mod.rs` | Add `GPU_ELIGIBILITY_THRESHOLD` constant |
| `crates/context-graph-core/src/dream/types.rs` | Update `GpuTriggerState` documentation |
| `crates/context-graph-core/src/dream/wake_controller.rs` | Add documentation clarifying 30% budget |
| `crates/context-graph-core/Cargo.toml` | Add `nvml-wrapper` dependency |

---

## Estimated Effort

| Task | Effort | Notes |
|------|--------|-------|
| Add nvml-wrapper dependency | 0.5h | Cargo.toml update, feature flags |
| Implement NvmlGpuMonitor | 3-4h | Core NVML integration |
| Add error types | 1h | GpuMonitorError enum |
| Update TriggerManager | 1h | Add 80% eligibility check |
| Update documentation | 1h | Clarify thresholds |
| Unit tests | 2h | Hardware-conditional tests (no mocks) |
| Integration tests | 2h | Real GPU tests (where available) |
| **Total** | **10-12h** | |

---

## Success Criteria

1. `GpuMonitor::get_usage()` returns real NVML values on GPU systems
2. `GpuMonitor::get_usage()` returns `Err(...)` on non-GPU systems (not 0.0)
3. Dreams trigger when GPU < 80% (eligibility)
4. Dreams abort when GPU > 30% (budget)
5. All constitution compliance tests pass
6. No silent failures (AP-26 compliance)

---

## Appendix: Current Code Analysis

### Current GpuMonitor Implementation (Stub)

**Location**: `crates/context-graph-core/src/dream/triggers.rs:291-345`

```rust
/// GPU utilization monitor stub.
///
/// Provides a placeholder for actual GPU monitoring.
/// Real implementation would use NVML (NVIDIA) or ROCm (AMD).
///
/// # Note
/// This is a STUB. Production requires actual GPU monitoring integration.
#[derive(Debug, Clone)]
pub struct GpuMonitor {
    simulated_usage: f32,
    use_simulated: bool,
}

impl GpuMonitor {
    pub fn get_usage(&self) -> f32 {
        if self.use_simulated {
            self.simulated_usage
        } else {
            // TODO(FUTURE): Implement real GPU monitoring via NVML
            // For now, return 0.0 (no GPU usage)
            0.0  // <-- VIOLATION: Silent failure
        }
    }

    pub fn is_available(&self) -> bool {
        // TODO(FUTURE): Check for actual GPU
        false
    }
}
```

### Current Threshold Definitions

**GpuTriggerState (30% budget)**: `crates/context-graph-core/src/dream/types.rs:470-481`
```rust
pub fn new() -> Self {
    Self {
        current_usage: 0.0,
        threshold: 0.30,  // Constitution says <30%
        // ...
    }
}
```

**MAX_GPU_USAGE constant**: `crates/context-graph-core/src/dream/mod.rs:163-164`
```rust
/// Maximum GPU usage during dream (Constitution: <30%)
pub const MAX_GPU_USAGE: f32 = 0.30;
```

**Missing: 80% Eligibility Threshold** - Not currently defined in code.

### Constitution Reference (docs2/constitution.yaml)

```yaml
dream:
  trigger: { activity: "<0.15", idle: "10min", entropy: ">0.7 for 5min", gpu: "<80%" }
  #                                                                        ^^^^^^
  #                                   Dream can START when GPU under 80% (has capacity)

  constraints: { queries: 100, semantic_leap: 0.7, abort_on_query: true, wake: "<100ms", gpu: "<30%" }
  #                                                                                        ^^^^^^
  #                                   Dream must use UNDER 30% GPU (budget during execution)
```
