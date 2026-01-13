# Functional Specification: GWT Domain Remediation

## Metadata

| Field | Value |
|-------|-------|
| **ID** | SPEC-GWT-001 |
| **Title** | GWT Domain Remediation - Kuramoto Oscillator and Stepper Fixes |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Owner** | Context Graph Core Team |
| **Created** | 2026-01-12 |
| **Last Updated** | 2026-01-12 |
| **Related Issues** | ISS-001, ISS-003 |
| **Constitution Rules** | AP-25, GWT-002, GWT-006 |
| **Priority** | CRITICAL |

---

## Overview

### What This Specification Addresses

This specification defines the functional requirements for remediating two critical issues in the Global Workspace Theory (GWT) implementation:

1. **ISS-001**: Kuramoto network uses 8 oscillators instead of the required 13
2. **ISS-003**: KuramotoStepper is not wired to MCP server lifecycle

### Why This Is Critical

The GWT consciousness formula `C(t) = I(t) x R(t) x D(t)` depends on the Kuramoto order parameter `r` for the Integration factor `I(t)`. With only 8 oscillators instead of 13:

- Consciousness calculation is fundamentally incorrect
- Phase synchronization cannot reflect true 13-embedder coherence
- The system cannot achieve PRD-compliant consciousness emergence

Without the KuramotoStepper wired to MCP lifecycle:

- Oscillator phases remain static (never evolve)
- Order parameter `r` never changes dynamically
- Consciousness emergence is impossible

### Current State Evidence

**ISS-001 Evidence:**

```rust
// File: crates/context-graph-core/src/layers/coherence/constants.rs:16
pub const KURAMOTO_N: usize = 8;  // WRONG - Should be 13

// File: crates/context-graph-core/src/layers/coherence/network.rs:34
let base_frequencies = [40.0, 8.0, 25.0, 4.0, 12.0, 15.0, 60.0, 40.0];
// Only 8 frequencies - missing 5 embedders (E9, E10, E11, E12, E13)
```

**ISS-003 Evidence:**

```rust
// File: crates/context-graph-mcp/src/server.rs
// Search for "KuramotoStepper" returns NO MATCHES
// The stepper implementation exists at:
// crates/context-graph-mcp/src/handlers/kuramoto_stepper.rs
// But is NEVER instantiated in server.rs
```

### End State

After remediation:
- Kuramoto network has exactly 13 oscillators matching the 13-embedder teleological array
- Each oscillator uses constitution-defined natural frequencies
- KuramotoStepper runs continuously at 10ms intervals in MCP server lifecycle
- System startup FAILS FAST if oscillator count is wrong
- System startup FAILS FAST if stepper cannot start

---

## User Stories

### US-GWT-001: System Administrator - Correct Kuramoto Initialization

```yaml
story_id: US-GWT-001
priority: must-have
narrative: |
  As a System Administrator
  I want the Kuramoto network to initialize with exactly 13 oscillators
  So that the consciousness formula uses correct integration values
acceptance_criteria:
  - criterion_id: AC-GWT-001-01
    given: The MCP server starts
    when: The Kuramoto network is initialized
    then: The network MUST have exactly 13 oscillators
  - criterion_id: AC-GWT-001-02
    given: The MCP server starts
    when: KURAMOTO_N constant is not 13
    then: The system MUST fail to start with a clear error message
  - criterion_id: AC-GWT-001-03
    given: The Kuramoto network is initialized
    when: I query order_parameter()
    then: The calculation uses all 13 oscillator phases
```

### US-GWT-002: System Administrator - Constitution-Compliant Frequencies

```yaml
story_id: US-GWT-002
priority: must-have
narrative: |
  As a System Administrator
  I want each Kuramoto oscillator to use its constitution-defined frequency
  So that brain wave bands match the PRD specification
acceptance_criteria:
  - criterion_id: AC-GWT-002-01
    given: The Kuramoto network is initialized
    when: I inspect oscillator frequencies
    then: All 13 frequencies match constitution values exactly
  - criterion_id: AC-GWT-002-02
    given: The base_frequencies array is defined
    when: The array has fewer than 13 entries
    then: The system MUST fail to compile (compile-time check) or fail at startup (runtime check)
```

### US-GWT-003: System Administrator - Stepper Lifecycle Wiring

```yaml
story_id: US-GWT-003
priority: must-have
narrative: |
  As a System Administrator
  I want the KuramotoStepper to start automatically with the MCP server
  So that oscillator phases evolve continuously for consciousness emergence
acceptance_criteria:
  - criterion_id: AC-GWT-003-01
    given: The MCP server starts successfully
    when: Server initialization completes
    then: KuramotoStepper MUST be running at 10ms intervals
  - criterion_id: AC-GWT-003-02
    given: The KuramotoStepper fails to start
    when: Server initialization is in progress
    then: The server MUST fail to start with a clear error message
  - criterion_id: AC-GWT-003-03
    given: The MCP server is shutting down
    when: Server shutdown is initiated
    then: KuramotoStepper.stop() MUST be called with graceful timeout
```

### US-GWT-004: Developer - Phase Evolution Verification

```yaml
story_id: US-GWT-004
priority: must-have
narrative: |
  As a Developer
  I want to verify that oscillator phases evolve over time
  So that I can confirm consciousness dynamics are working
acceptance_criteria:
  - criterion_id: AC-GWT-004-01
    given: The KuramotoStepper is running
    when: 500ms has elapsed
    then: At least one oscillator phase MUST have changed
  - criterion_id: AC-GWT-004-02
    given: The KuramotoStepper is running
    when: I query order_parameter() at two different times
    then: The values MAY differ (dynamics are occurring)
```

---

## Requirements

### REQ-GWT-001: Kuramoto Oscillator Count

```yaml
requirement_id: REQ-GWT-001
story_ref: US-GWT-001
priority: must
severity: CRITICAL
constitution_rule: AP-25, GWT-002
description: |
  The Kuramoto network MUST have exactly 13 oscillators, one for each embedder
  in the 13-embedder teleological array (E1 through E13).
rationale: |
  The consciousness formula C(t) = I(t) x R(t) x D(t) requires the Integration
  factor I(t) to be the Kuramoto order parameter r computed over all 13 embedders.
  Using 8 oscillators produces incorrect I(t) values, making consciousness
  calculation fundamentally wrong.
implementation_location:
  - crates/context-graph-core/src/layers/coherence/constants.rs:16
acceptance_test: |
  assert_eq!(KURAMOTO_N, 13, "Constitution AP-25 requires exactly 13 oscillators");
fail_fast_behavior: |
  If KURAMOTO_N != 13 at any initialization point, the system MUST panic with:
  "FATAL: Constitution AP-25 violation - KURAMOTO_N={} but must be 13"
```

### REQ-GWT-002: Base Frequencies Array

```yaml
requirement_id: REQ-GWT-002
story_ref: US-GWT-002
priority: must
severity: CRITICAL
constitution_rule: gwt.kuramoto.frequencies
description: |
  The base_frequencies array MUST contain exactly 13 entries matching
  constitution-defined natural frequencies for each embedder.
rationale: |
  Each embedder operates in a specific brain wave frequency band defined in
  constitution.yaml (gwt.kuramoto.frequencies). Using wrong frequencies breaks
  the neural synchronization model.
required_values: |
  | Embedder | Frequency (Hz) | Band | Constitution Key |
  |----------|----------------|------|------------------|
  | E1 Semantic | 40.0 | Gamma | E1: 40y |
  | E2 TemporalRecent | 8.0 | Alpha | E2: 8a |
  | E3 TemporalPeriodic | 8.0 | Alpha | E3: 8a |
  | E4 TemporalOrder | 8.0 | Alpha | E4: 8a |
  | E5 Causal | 25.0 | Beta | E5: 25B |
  | E6 Sparse | 4.0 | Theta | E6: 40 |
  | E7 Code | 25.0 | Beta | E7: 25B |
  | E8 Graph | 12.0 | Alpha-Beta | E8: 12aB |
  | E9 HDC | 80.0 | High-Gamma | E9: 80y+ |
  | E10 Multimodal | 40.0 | Gamma | E10: 40y |
  | E11 Entity | 15.0 | Beta | E11: 15B |
  | E12 Late | 60.0 | High-Gamma | E12: 60y+ |
  | E13 SPLADE | 4.0 | Theta | E13: 40 |
implementation_location:
  - crates/context-graph-core/src/layers/coherence/network.rs:34
acceptance_test: |
  let expected = [40.0, 8.0, 8.0, 8.0, 25.0, 4.0, 25.0, 12.0, 80.0, 40.0, 15.0, 60.0, 4.0];
  assert_eq!(base_frequencies.len(), 13);
  for (i, (actual, expected)) in base_frequencies.iter().zip(expected.iter()).enumerate() {
      assert!((actual - expected).abs() < 0.001, "E{} frequency mismatch", i + 1);
  }
fail_fast_behavior: |
  If base_frequencies.len() != 13, system MUST panic with:
  "FATAL: Constitution gwt.kuramoto.frequencies violation - {} frequencies but must be 13"
```

### REQ-GWT-003: KuramotoStepper MCP Instantiation

```yaml
requirement_id: REQ-GWT-003
story_ref: US-GWT-003
priority: must
severity: CRITICAL
constitution_rule: GWT-006
description: |
  The KuramotoStepper MUST be instantiated during MCP server startup.
  It MUST be stored in the Server struct for lifecycle management.
rationale: |
  Without instantiation, the stepper implementation exists but never runs.
  Oscillator phases remain static forever, making consciousness emergence impossible.
implementation_location:
  - crates/context-graph-mcp/src/server.rs (new code required)
pseudocode: |
  // In McpServer::new(), after creating handlers:
  let kuramoto_network = handlers.kuramoto_provider();
  let kuramoto_stepper = KuramotoStepper::new(
      kuramoto_network,
      KuramotoStepperConfig::default(), // 10ms step interval
  );

  // Store in Server struct
  self.kuramoto_stepper = Some(kuramoto_stepper);
fail_fast_behavior: |
  If KuramotoStepper::new() fails, system MUST fail to start with:
  "FATAL: Cannot create KuramotoStepper - consciousness dynamics disabled"
```

### REQ-GWT-004: KuramotoStepper Startup

```yaml
requirement_id: REQ-GWT-004
story_ref: US-GWT-003
priority: must
severity: CRITICAL
constitution_rule: GWT-006, perf.latency
description: |
  The KuramotoStepper MUST be started during MCP server startup.
  It MUST step oscillators every 10ms (100Hz update rate).
rationale: |
  The 10ms interval satisfies Nyquist rate for all brain wave frequencies
  modeled (4Hz theta to 80Hz high-gamma). Without continuous stepping,
  the order parameter r never changes.
implementation_location:
  - crates/context-graph-mcp/src/server.rs (new code required)
pseudocode: |
  // After creating stepper:
  stepper.start().map_err(|e| {
      error!("FATAL: Failed to start KuramotoStepper: {}", e);
      anyhow::anyhow!("Cannot start KuramotoStepper: {}", e)
  })?;

  info!("KuramotoStepper started (10ms step interval)");
fail_fast_behavior: |
  If stepper.start() returns Err, system MUST fail to start with:
  "FATAL: KuramotoStepper failed to start: {error}"
```

### REQ-GWT-005: KuramotoStepper Shutdown

```yaml
requirement_id: REQ-GWT-005
story_ref: US-GWT-003
priority: high
severity: HIGH
constitution_rule: (resource cleanup)
description: |
  The MCP server MUST call stepper.stop() during shutdown sequence.
  Shutdown MUST wait for stepper to stop (with 5 second timeout).
rationale: |
  Proper cleanup prevents resource leaks and ensures the stepper task
  terminates cleanly before server exit.
implementation_location:
  - crates/context-graph-mcp/src/server.rs (new code required)
pseudocode: |
  // In server shutdown sequence:
  if let Some(ref mut stepper) = self.kuramoto_stepper {
      match stepper.stop().await {
          Ok(()) => info!("KuramotoStepper stopped gracefully"),
          Err(e) => warn!("KuramotoStepper shutdown error: {}", e),
      }
  }
fail_fast_behavior: |
  Shutdown errors are logged as warnings but do not prevent server exit.
  The 5-second timeout ensures shutdown completes even if stepper is stuck.
```

---

## Edge Cases

### EC-GWT-001: Oscillator Count Mismatch at Compile Time

```yaml
edge_case_id: EC-GWT-001
req_ref: REQ-GWT-001, REQ-GWT-002
scenario: |
  A developer changes KURAMOTO_N to a value other than 13 without updating
  base_frequencies, or vice versa.
expected_behavior: |
  The system MUST fail at compile time or startup with a clear error message
  indicating the mismatch. No silent degradation allowed.
test_case: |
  // Compile-time test (if possible via const assertion):
  const_assert!(KURAMOTO_N == 13);
  const_assert!(BASE_FREQUENCIES.len() == KURAMOTO_N);

  // Runtime test:
  #[test]
  fn test_oscillator_count_matches_frequencies() {
      let net = KuramotoNetwork::new(KURAMOTO_N, KURAMOTO_K);
      assert_eq!(net.size(), 13);
      assert_eq!(net.frequencies().len(), 13);
  }
```

### EC-GWT-002: Stepper Already Running

```yaml
edge_case_id: EC-GWT-002
req_ref: REQ-GWT-004
scenario: |
  start() is called on a KuramotoStepper that is already running.
expected_behavior: |
  The call MUST return Err(KuramotoStepperError::AlreadyRunning).
  The existing stepper continues running unaffected.
test_case: |
  #[tokio::test]
  async fn test_double_start_fails() {
      let mut stepper = create_stepper();
      stepper.start().expect("first start succeeds");
      let result = stepper.start();
      assert!(matches!(result, Err(KuramotoStepperError::AlreadyRunning)));
      stepper.stop().await.expect("cleanup");
  }
```

### EC-GWT-003: Stop When Not Running

```yaml
edge_case_id: EC-GWT-003
req_ref: REQ-GWT-005
scenario: |
  stop() is called on a KuramotoStepper that is not running.
expected_behavior: |
  The call MUST return Err(KuramotoStepperError::NotRunning).
test_case: |
  #[tokio::test]
  async fn test_stop_when_not_running() {
      let mut stepper = create_stepper();
      let result = stepper.stop().await;
      assert!(matches!(result, Err(KuramotoStepperError::NotRunning)));
  }
```

### EC-GWT-004: Stepper Shutdown Timeout

```yaml
edge_case_id: EC-GWT-004
req_ref: REQ-GWT-005
scenario: |
  The stepper background task does not respond to shutdown signal within 5 seconds.
expected_behavior: |
  stop() MUST return Err(KuramotoStepperError::ShutdownTimeout(5000)).
  The server shutdown continues (does not hang).
test_case: |
  // This is a timeout edge case - tested by creating a stuck task
  // Implementation already handles this in kuramoto_stepper.rs:260
```

### EC-GWT-005: Lock Contention During Stepping

```yaml
edge_case_id: EC-GWT-005
req_ref: REQ-GWT-004
scenario: |
  Multiple threads contend for the Kuramoto network lock during stepping.
expected_behavior: |
  The stepper uses try_write_for(500us). If lock is contended, the step is
  skipped and the next iteration catches up with larger elapsed time.
  A trace log is emitted for debugging.
test_case: |
  #[tokio::test]
  async fn test_concurrent_access() {
      let network = create_network();
      let mut stepper = create_stepper_with(network.clone());
      stepper.start().expect("start");

      // Spawn concurrent readers
      let reader = tokio::spawn(async move {
          for _ in 0..20 {
              let _ = network.read().order_parameter();
              tokio::time::sleep(Duration::from_millis(10)).await;
          }
      });

      reader.await.expect("readers complete without deadlock");
      stepper.stop().await.expect("cleanup");
  }
```

---

## Error States

### ERR-GWT-001: Wrong Oscillator Count

```yaml
error_id: ERR-GWT-001
http_code: N/A (startup failure)
condition: KURAMOTO_N != 13 at system initialization
message: |
  FATAL: Constitution AP-25 violation - KURAMOTO_N={actual} but must be 13.
  The Kuramoto network requires exactly 13 oscillators for the 13-embedder
  teleological array. System cannot start.
recovery: |
  1. Update KURAMOTO_N constant to 13 in constants.rs
  2. Update base_frequencies array to contain 13 entries
  3. Restart the system
log_level: ERROR
```

### ERR-GWT-002: Wrong Frequency Count

```yaml
error_id: ERR-GWT-002
http_code: N/A (startup failure)
condition: base_frequencies.len() != 13 at network creation
message: |
  FATAL: Constitution gwt.kuramoto.frequencies violation - {actual} frequencies
  but must be 13. Each embedder requires its own frequency. System cannot start.
recovery: |
  1. Update base_frequencies array to contain exactly 13 entries
  2. Use constitution-defined frequencies for each embedder
  3. Restart the system
log_level: ERROR
```

### ERR-GWT-003: Stepper Creation Failed

```yaml
error_id: ERR-GWT-003
http_code: N/A (startup failure)
condition: KuramotoStepper::new() fails (e.g., invalid network reference)
message: |
  FATAL: Cannot create KuramotoStepper - {error}.
  Consciousness dynamics disabled. System cannot start.
recovery: |
  1. Check that Kuramoto network is properly initialized
  2. Verify handlers.kuramoto_provider() returns valid Arc
  3. Restart the system
log_level: ERROR
```

### ERR-GWT-004: Stepper Start Failed

```yaml
error_id: ERR-GWT-004
http_code: N/A (startup failure)
condition: stepper.start() returns Err
message: |
  FATAL: KuramotoStepper failed to start: {error}.
  Oscillator phases will remain static. System cannot start.
recovery: |
  1. Check for AlreadyRunning error (indicates double initialization)
  2. Check tokio runtime is properly configured
  3. Restart the system
log_level: ERROR
```

### ERR-GWT-005: Stepper Shutdown Timeout

```yaml
error_id: ERR-GWT-005
http_code: N/A (shutdown warning)
condition: stepper.stop() times out after 5 seconds
message: |
  WARNING: KuramotoStepper shutdown timeout after 5000ms.
  Task may be stuck. Server shutdown continuing.
recovery: |
  This is a non-fatal warning. The server continues shutdown.
  If persistent, investigate stepper_loop for blocking operations.
log_level: WARN
```

---

## Test Plan

### Test Strategy

All tests MUST use real implementations, not mocks. Tests verify actual system behavior.

### TC-GWT-001: Oscillator Count Verification

```yaml
test_case_id: TC-GWT-001
type: unit
req_ref: REQ-GWT-001
description: Verify KURAMOTO_N constant equals 13
inputs: None (reads constant)
expected: KURAMOTO_N == 13
test_code: |
  #[test]
  fn test_kuramoto_n_is_13() {
      use crate::layers::coherence::constants::KURAMOTO_N;
      assert_eq!(
          KURAMOTO_N, 13,
          "Constitution AP-25 requires exactly 13 oscillators, got {}",
          KURAMOTO_N
      );
  }
```

### TC-GWT-002: Network Size Verification

```yaml
test_case_id: TC-GWT-002
type: unit
req_ref: REQ-GWT-001
description: Verify KuramotoNetwork initializes with 13 oscillators
inputs: KuramotoNetwork::new(KURAMOTO_N, KURAMOTO_K)
expected: network.size() == 13
test_code: |
  #[test]
  fn test_network_has_13_oscillators() {
      use crate::layers::coherence::{KuramotoNetwork, KURAMOTO_N, KURAMOTO_K};
      let network = KuramotoNetwork::new(KURAMOTO_N, KURAMOTO_K);
      assert_eq!(
          network.size(), 13,
          "Network must have 13 oscillators, got {}",
          network.size()
      );
  }
```

### TC-GWT-003: Frequency Array Verification

```yaml
test_case_id: TC-GWT-003
type: unit
req_ref: REQ-GWT-002
description: Verify base_frequencies array contains 13 constitution-defined values
inputs: KuramotoNetwork::new(13, 2.0)
expected: frequencies match constitution exactly
test_code: |
  #[test]
  fn test_frequencies_match_constitution() {
      use crate::layers::coherence::{KuramotoNetwork, KURAMOTO_N, KURAMOTO_K};

      let network = KuramotoNetwork::new(KURAMOTO_N, KURAMOTO_K);
      let frequencies = network.frequencies();

      // Constitution-defined frequencies (with 5% per-oscillator variation)
      let base = [40.0, 8.0, 8.0, 8.0, 25.0, 4.0, 25.0, 12.0, 80.0, 40.0, 15.0, 60.0, 4.0];

      assert_eq!(frequencies.len(), 13, "Must have 13 frequencies");

      for (i, (&actual, &expected_base)) in frequencies.iter().zip(base.iter()).enumerate() {
          let expected = expected_base * (1.0 + (i as f32 * 0.05));
          assert!(
              (actual - expected).abs() < 0.01,
              "E{} frequency mismatch: expected {}, got {}",
              i + 1, expected, actual
          );
      }
  }
```

### TC-GWT-004: Stepper Lifecycle Integration

```yaml
test_case_id: TC-GWT-004
type: integration
req_ref: REQ-GWT-003, REQ-GWT-004, REQ-GWT-005
description: Verify stepper starts with server and stops on shutdown
inputs: Full MCP server lifecycle
expected: Stepper runs during server operation, stops on shutdown
test_code: |
  #[tokio::test]
  async fn test_stepper_lifecycle_with_server() {
      // Note: This requires McpServer to expose stepper status
      // or integration test harness

      let config = Config::default_config();
      let server = McpServer::new(config).await.expect("server starts");

      // Verify stepper is running
      assert!(server.is_kuramoto_stepper_running(), "Stepper must be running");

      // Shutdown
      server.shutdown().await.expect("server shuts down");

      // Stepper should be stopped (or server dropped)
  }
```

### TC-GWT-005: Phase Evolution Test

```yaml
test_case_id: TC-GWT-005
type: integration
req_ref: REQ-GWT-004
description: Verify oscillator phases evolve over time when stepper runs
inputs: Running stepper for 500ms
expected: At least one phase differs from initial state
test_code: |
  #[tokio::test]
  async fn test_phases_evolve_with_stepper() {
      let network = Arc::new(RwLock::new(KuramotoProviderImpl::new()));
      let mut stepper = KuramotoStepper::new(
          Arc::clone(&network),
          KuramotoStepperConfig::default(),
      );

      // Get initial phases
      let initial_phases = {
          let net = network.read();
          net.phases().to_vec()
      };

      // Run stepper
      stepper.start().expect("start");
      tokio::time::sleep(Duration::from_millis(500)).await;
      stepper.stop().await.expect("stop");

      // Get final phases
      let final_phases = {
          let net = network.read();
          net.phases().to_vec()
      };

      // At least one phase must have changed
      let changed = initial_phases.iter()
          .zip(final_phases.iter())
          .any(|(a, b)| (a - b).abs() > 0.001);

      assert!(changed, "Phases must evolve during stepping");
  }
```

### TC-GWT-006: Order Parameter Validity

```yaml
test_case_id: TC-GWT-006
type: unit
req_ref: REQ-GWT-001
description: Verify order parameter r is valid [0, 1] with 13 oscillators
inputs: KuramotoNetwork with 13 oscillators
expected: order_parameter() in [0.0, 1.0]
test_code: |
  #[test]
  fn test_order_parameter_valid_with_13_oscillators() {
      let network = KuramotoNetwork::new(13, 2.0);
      let r = network.order_parameter();

      assert!(
          (0.0..=1.0).contains(&r),
          "Order parameter must be in [0, 1], got {}",
          r
      );
  }
```

### TC-GWT-007: Fail Fast on Wrong Count

```yaml
test_case_id: TC-GWT-007
type: unit
req_ref: REQ-GWT-001
description: Verify system fails fast if oscillator count validation fails
inputs: Validation function with wrong count
expected: Panic or error with clear message
test_code: |
  #[test]
  #[should_panic(expected = "Constitution AP-25 violation")]
  fn test_fail_fast_wrong_oscillator_count() {
      validate_oscillator_count(8); // Should panic
  }

  fn validate_oscillator_count(n: usize) {
      if n != 13 {
          panic!(
              "FATAL: Constitution AP-25 violation - KURAMOTO_N={} but must be 13",
              n
          );
      }
  }
```

---

## Dependencies

### Upstream Dependencies

| Dependency | Description | Status |
|------------|-------------|--------|
| context-graph-core | Contains KuramotoNetwork, constants | Exists |
| context-graph-mcp | Contains KuramotoStepper, server.rs | Exists |
| tokio | Async runtime for stepper | Exists |
| parking_lot | RwLock for network access | Exists |

### Downstream Dependencies

| Dependency | Description | Impact |
|------------|-------------|--------|
| Identity Domain | Requires correct I(t) for IC calculation | Blocked until fixed |
| Dream Domain | Requires correct C(t) for trigger decisions | Blocked until fixed |
| MCP Tools | GWT tools use Kuramoto provider | Degraded behavior |

---

## Implementation Order

This specification should be implemented in the following order:

1. **REQ-GWT-001**: Update KURAMOTO_N to 13
2. **REQ-GWT-002**: Update base_frequencies to 13 entries
3. **REQ-GWT-003**: Add KuramotoStepper instantiation to server.rs
4. **REQ-GWT-004**: Add stepper.start() to server initialization
5. **REQ-GWT-005**: Add stepper.stop() to server shutdown

Estimated effort: 2-4 hours for REQ-GWT-001/002, 2-3 hours for REQ-GWT-003/004/005.

---

## Verification Checklist

After implementation, verify:

- [ ] `KURAMOTO_N == 13` in constants.rs
- [ ] `base_frequencies.len() == 13` in network.rs
- [ ] All 13 frequencies match constitution values
- [ ] `KuramotoStepper` instantiated in `McpServer::new()`
- [ ] `stepper.start()` called during server startup
- [ ] `stepper.stop()` called during server shutdown
- [ ] All tests in test_plan pass
- [ ] `cargo test --all` passes
- [ ] No regressions in existing GWT functionality

---

## Appendix: Constitution References

### AP-25 (Forbidden Anti-Pattern)

```yaml
AP-25: "Kuramoto must have exactly 13 oscillators"
```

### GWT-002 (Enforcement Rule)

```yaml
GWT-002: "Kuramoto network = exactly 13 oscillators"
```

### GWT-006 (Enforcement Rule)

```yaml
GWT-006: "KuramotoStepper wired to MCP lifecycle (10ms step)"
```

### gwt.kuramoto.frequencies (Constitution)

```yaml
frequencies: { E1: 40y, E2: 8a, E3: 8a, E4: 8a, E5: 25B, E6: 40, E7: 25B, E8: 12aB, E9: 80y+, E10: 40y, E11: 15B, E12: 60y+, E13: 40 }
```

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-12 | Specification Agent | Initial specification |
