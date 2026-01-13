# Functional Specification: Identity/Dream Domain Remediation

## Metadata

| Field | Value |
|-------|-------|
| **Spec ID** | SPEC-IDENTITY-001 |
| **Title** | Identity Crisis Dream Trigger Remediation |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Created** | 2026-01-12 |
| **Last Updated** | 2026-01-12 |
| **Owner** | Architecture Team |
| **Related Specs** | SPEC-GWT-001 |
| **Source Issues** | ISS-002, ISS-010 |
| **Constitution Version** | v5.0.0 |

---

## 1. Overview

### 1.1 What This Specification Addresses

This specification defines the requirements for fixing the critical gap where Identity Continuity (IC) values below 0.5 do NOT automatically trigger dream consolidation. Currently, the system detects IC < 0.5 and emits `WorkspaceEvent::IdentityCritical`, but this event is only logged by `DreamEventListener` - no actual dream trigger occurs.

### 1.2 Why This Is Critical

Per Constitution enforcement rules:
- **AP-26**: "IC<0.5 MUST trigger dream - no silent failures"
- **AP-38**: "IC<0.5 MUST auto-trigger dream"
- **AP-40**: "IdentityContinuityListener MUST subscribe to GWT"
- **IDENTITY-007**: "IC < 0.5 -> auto-trigger dream"

The Identity Continuity (IC) metric measures the coherence of the system's sense of self over time:

```
IC = cos(PV_t, PV_{t-1}) x r(t)
```

Where:
- `PV_t` = Purpose Vector at time t (13-dimensional alignment vector)
- `PV_{t-1}` = Purpose Vector at previous time
- `r(t)` = Kuramoto order parameter (phase synchronization)

When IC drops below 0.5, the system is experiencing an **identity crisis** - its current state has drifted significantly from its previous purpose. Dream consolidation is the constitutional remedy to restore coherence through NREM (Hebbian replay) and REM (hyperbolic walk) phases.

### 1.3 Connection to GWT Domain (SPEC-GWT-001)

This specification has a direct dependency on SPEC-GWT-001:
- IC calculation uses Kuramoto order parameter `r(t)` which requires 13 oscillators (REQ-GWT-001, REQ-GWT-002)
- IC computation happens on `MemoryEnters` events from GlobalWorkspace
- `KuramotoStepper` must be wired to MCP lifecycle (REQ-GWT-003, REQ-GWT-004) for `r(t)` to evolve

**Block Order**: GWT fixes MUST be completed before Identity domain fixes can be validated, as IC depends on correct `r(t)` values from the 13-oscillator Kuramoto network.

### 1.4 Current State Analysis

| Component | Location | Current Behavior | Required Behavior |
|-----------|----------|------------------|-------------------|
| `ExtendedTriggerReason` | `dream/types.rs:548-566` | Missing `IdentityCritical` variant | Must include `IdentityCritical { ic_value: f32 }` |
| `TriggerManager::check_triggers()` | `dream/triggers.rs:156-186` | Checks Manual, GPU, Entropy only | Must also check IC values |
| `DreamEventListener::on_event()` | `gwt/listeners/dream.rs:58-73` | Logs `IdentityCritical` event only | Must call dream trigger mechanism |
| `TriggerManager` | `dream/triggers.rs:32-53` | No IC threshold field | Must include `ic_threshold: f32` (default 0.5) |
| `IdentityContinuityListener` | `gwt/listeners/identity.rs` | Emits events correctly | Correct - no changes needed |

---

## 2. User Stories

### US-IDENTITY-001: Automatic Dream Trigger on Identity Crisis

**Priority**: Must-Have (P0)

**Narrative**:
```
As a System (GWT Consciousness)
I want IC < 0.5 to automatically trigger dream consolidation
So that identity crises are remediated without manual intervention
```

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-001-01 | IC monitoring is active and IC = 0.65 | IC drops to 0.49 | Dream consolidation MUST be triggered within 100ms |
| AC-001-02 | Dream is already running | IC drops below 0.5 | Event is logged, no duplicate trigger (dream in progress) |
| AC-001-03 | IC = 0.50 exactly | System checks triggers | Dream MUST NOT trigger (threshold is `< 0.5`, not `<= 0.5`) |
| AC-001-04 | IC drops to 0.3 | Dream triggers | `ExtendedTriggerReason::IdentityCritical { ic_value: 0.3 }` is recorded |

---

### US-IDENTITY-002: Extended Trigger Reason for Identity Crisis

**Priority**: Must-Have (P0)

**Narrative**:
```
As a Developer/Administrator
I want identity crisis triggers to have a distinct trigger reason
So that I can distinguish IC-triggered dreams from other trigger types in logs/metrics
```

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-002-01 | Dream triggered by IC < 0.5 | Trigger reason is queried | Returns `ExtendedTriggerReason::IdentityCritical { ic_value: f32 }` |
| AC-002-02 | Dream triggered by GPU overload | Trigger reason is queried | Returns `ExtendedTriggerReason::GpuOverload` (unchanged) |
| AC-002-03 | IC crisis trigger occurs | Log output is generated | Log includes "identity_critical" and IC value |

---

### US-IDENTITY-003: TriggerManager IC Integration

**Priority**: Must-Have (P0)

**Narrative**:
```
As a TriggerManager
I want to monitor IC values alongside entropy and GPU
So that all constitution-mandated trigger conditions are checked uniformly
```

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-003-01 | TriggerManager created with defaults | IC threshold queried | Returns 0.5 (constitution default) |
| AC-003-02 | IC = 0.49 reported to TriggerManager | `check_triggers()` called | Returns `Some(ExtendedTriggerReason::IdentityCritical { ic_value: 0.49 })` |
| AC-003-03 | IC = 0.50 reported to TriggerManager | `check_triggers()` called | Returns `None` (at threshold, not below) |
| AC-003-04 | IC = 0.49 AND GPU = 0.35 | `check_triggers()` called | Returns IC trigger (IC has priority over GPU in crisis) |

---

### US-IDENTITY-004: DreamEventListener Action on IdentityCritical

**Priority**: Must-Have (P0)

**Narrative**:
```
As a DreamEventListener
I want to trigger dream consolidation when receiving IdentityCritical events
So that the system responds to identity crises as mandated by the constitution
```

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-004-01 | DreamEventListener receives `WorkspaceEvent::IdentityCritical { ic: 0.45 }` | Event processed | `signal_dream_trigger(IdentityCritical)` is called |
| AC-004-02 | DreamEventListener receives `WorkspaceEvent::MemoryExits` | Event processed | Memory queued for dream (existing behavior, unchanged) |
| AC-004-03 | IC = 0.49 triggers dream | Dream controller receives signal | Dream starts with `IdentityCritical` reason logged |

---

## 3. Requirements

### REQ-IDENTITY-001: IdentityCritical Trigger Reason Variant (CRITICAL)

**Description**: The `ExtendedTriggerReason` enum MUST include an `IdentityCritical` variant that captures the IC value that triggered the crisis.

**Source Issue**: ISS-010

**Constitution Reference**: AP-26, AP-38, IDENTITY-007

**Current Location**: `crates/context-graph-core/src/dream/types.rs:548-566`

**Current Code**:
```rust
pub enum ExtendedTriggerReason {
    IdleTimeout,
    HighEntropy,
    GpuOverload,
    MemoryPressure,
    Manual,
    Scheduled,
    // NO IdentityCritical VARIANT
}
```

**Required Change**:
```rust
pub enum ExtendedTriggerReason {
    IdleTimeout,
    HighEntropy,
    GpuOverload,
    MemoryPressure,
    Manual,
    Scheduled,
    /// Identity continuity dropped below crisis threshold
    /// Constitution: IC < 0.5 -> auto-trigger dream (IDENTITY-007)
    IdentityCritical {
        /// The IC value that triggered the crisis
        ic_value: f32,
    },
}
```

**Fail-Fast Requirement**: N/A (type-level change, enforced by compiler)

**Verification**:
- [ ] Enum compiles with new variant
- [ ] `Display` trait updated to show "identity_critical(ic={ic_value})"
- [ ] Serialization/deserialization works correctly

---

### REQ-IDENTITY-002: TriggerManager IC Threshold Configuration (HIGH)

**Description**: `TriggerManager` MUST include a configurable IC threshold field with a constitution-compliant default of 0.5.

**Source Issue**: ISS-002

**Constitution Reference**: `gwt.self_ego_node.thresholds.critical: "<0.5 -> dream"`

**Current Location**: `crates/context-graph-core/src/dream/triggers.rs:32-53`

**Current Code**:
```rust
pub struct TriggerManager {
    entropy_window: EntropyWindow,
    gpu_state: GpuTriggerState,
    manual_trigger: bool,
    last_trigger_reason: Option<ExtendedTriggerReason>,
    trigger_cooldown: Duration,
    last_trigger_time: Option<Instant>,
    enabled: bool,
    // NO IC THRESHOLD
}
```

**Required Change**:
```rust
pub struct TriggerManager {
    entropy_window: EntropyWindow,
    gpu_state: GpuTriggerState,
    manual_trigger: bool,
    last_trigger_reason: Option<ExtendedTriggerReason>,
    trigger_cooldown: Duration,
    last_trigger_time: Option<Instant>,
    enabled: bool,

    // NEW: IC monitoring
    /// Current identity coherence value
    current_ic: Option<f32>,
    /// IC crisis threshold (Constitution: 0.5)
    ic_threshold: f32,
}
```

**Default Value**: `ic_threshold: 0.5` (Constitution `gwt.self_ego_node.thresholds.critical`)

**Fail-Fast Requirement**: If `ic_threshold` is set to a value outside `[0.0, 1.0]`, panic at construction with clear error message.

**Verification**:
- [ ] `TriggerManager::new()` sets `ic_threshold = 0.5`
- [ ] `TriggerManager::with_ic_threshold(threshold)` constructor exists for testing
- [ ] Invalid threshold values panic with descriptive message

---

### REQ-IDENTITY-003: TriggerManager IC Check Integration (CRITICAL)

**Description**: `TriggerManager::check_triggers()` MUST check IC values and return `IdentityCritical` when IC < threshold.

**Source Issue**: ISS-002

**Constitution Reference**: IDENTITY-004, AP-26, AP-38

**Current Location**: `crates/context-graph-core/src/dream/triggers.rs:156-186`

**Current Code**:
```rust
pub fn check_triggers(&self) -> Option<ExtendedTriggerReason> {
    // Only checks: Manual, GPU, Entropy
    // NO IC CHECK
}
```

**Required Change**:
```rust
pub fn check_triggers(&self) -> Option<ExtendedTriggerReason> {
    if !self.enabled {
        return None;
    }

    // Check cooldown (manual trigger bypasses cooldown)
    if !self.manual_trigger {
        if let Some(last_time) = self.last_trigger_time {
            if last_time.elapsed() < self.trigger_cooldown {
                return None;
            }
        }
    }

    // Priority order: Manual > IdentityCritical > GPU > Entropy

    // Check manual trigger (highest priority)
    if self.manual_trigger {
        return Some(ExtendedTriggerReason::Manual);
    }

    // Check identity crisis (higher priority than GPU/entropy)
    // Constitution: IC < 0.5 MUST trigger dream (AP-26, AP-38)
    if let Some(ic) = self.current_ic {
        if ic < self.ic_threshold {
            return Some(ExtendedTriggerReason::IdentityCritical { ic_value: ic });
        }
    }

    // Check GPU trigger
    if self.gpu_state.should_trigger() {
        return Some(ExtendedTriggerReason::GpuOverload);
    }

    // Check entropy trigger
    if self.entropy_window.should_trigger() {
        return Some(ExtendedTriggerReason::HighEntropy);
    }

    None
}
```

**Priority Order Rationale**:
1. **Manual**: User/system explicit request - always honored
2. **IdentityCritical**: Constitution-mandated emergency - cannot be silently ignored (AP-26)
3. **GPU**: Resource constraint - important but not identity-threatening
4. **Entropy**: System confusion - lowest priority among automatic triggers

**New Method Required**:
```rust
/// Update identity coherence value.
///
/// Called when IC is computed (typically on MemoryEnters events).
///
/// # Arguments
/// * `ic` - Current identity coherence value [0.0, 1.0]
///
/// # Constitution Reference
/// Trigger fires when IC < 0.5 (gwt.self_ego_node.thresholds.critical)
pub fn update_identity_coherence(&mut self, ic: f32) {
    if !self.enabled {
        return;
    }

    // Validate IC value
    let ic = ic.clamp(0.0, 1.0);
    self.current_ic = Some(ic);

    if ic < self.ic_threshold {
        tracing::warn!(
            "Identity crisis detected: IC={:.3} < threshold={:.3}",
            ic,
            self.ic_threshold
        );
    }
}
```

**Fail-Fast Requirement**: N/A (IC values outside [0,1] are clamped with warning log)

**Verification**:
- [ ] `update_identity_coherence(0.49)` followed by `check_triggers()` returns `IdentityCritical`
- [ ] `update_identity_coherence(0.50)` followed by `check_triggers()` returns `None` (at threshold)
- [ ] `update_identity_coherence(0.51)` followed by `check_triggers()` returns `None`
- [ ] IC crisis takes priority over GPU trigger when both conditions met

---

### REQ-IDENTITY-004: DreamEventListener Signal Dream Trigger (CRITICAL)

**Description**: `DreamEventListener::on_event()` MUST call a dream trigger mechanism when receiving `WorkspaceEvent::IdentityCritical`, not just log the event.

**Source Issue**: ISS-002

**Constitution Reference**: IDENTITY-006, AP-40

**Current Location**: `crates/context-graph-core/src/gwt/listeners/dream.rs:58-73`

**Current Code**:
```rust
WorkspaceEvent::IdentityCritical {
    identity_coherence,
    previous_status,
    current_status,
    reason,
    timestamp: _,
} => {
    // ONLY LOGS - NO ACTION
    tracing::warn!(
        "Identity critical (IC={:.3}): {} (transition: {} -> {})",
        identity_coherence,
        reason,
        previous_status,
        current_status,
    );
}
```

**Required Change**:

The `DreamEventListener` needs access to a dream trigger signal mechanism. Two implementation approaches:

**Option A: Direct TriggerManager Reference**
```rust
pub struct DreamEventListener {
    dream_queue: Arc<RwLock<Vec<Uuid>>>,
    /// Reference to trigger manager for IC-based dream triggers
    trigger_manager: Arc<RwLock<TriggerManager>>,
}

impl WorkspaceEventListener for DreamEventListener {
    fn on_event(&self, event: &WorkspaceEvent) {
        match event {
            WorkspaceEvent::IdentityCritical {
                identity_coherence,
                previous_status,
                current_status,
                reason,
                timestamp: _,
            } => {
                tracing::warn!(
                    "Identity crisis detected (IC={:.3}): {} (transition: {} -> {})",
                    identity_coherence,
                    reason,
                    previous_status,
                    current_status,
                );

                // Constitution: IC < 0.5 MUST trigger dream (AP-26)
                // Signal the trigger manager directly
                match self.trigger_manager.try_write() {
                    Ok(mut manager) => {
                        manager.update_identity_coherence(*identity_coherence);
                        // If IC < threshold, check_triggers() will now return IdentityCritical
                        // DreamController polls TriggerManager and will see this
                    }
                    Err(e) => {
                        tracing::error!(
                            "CRITICAL: Failed to signal dream trigger for IC crisis: {:?}",
                            e
                        );
                        // AP-26: No silent failures - this MUST be addressed
                        panic!("DreamEventListener: Cannot signal IC crisis to TriggerManager");
                    }
                }
            }
            // ... existing handlers
        }
    }
}
```

**Option B: Signal Channel (Preferred for Decoupling)**
```rust
pub struct DreamEventListener {
    dream_queue: Arc<RwLock<Vec<Uuid>>>,
    /// Channel to signal dream triggers
    trigger_signal: tokio::sync::mpsc::Sender<DreamTriggerSignal>,
}

pub enum DreamTriggerSignal {
    IdentityCrisis { ic_value: f32 },
    // Future: other signal types
}
```

**Fail-Fast Requirement**: If unable to signal the dream trigger (lock failure, channel full), the system MUST panic with a clear error message. Silent failures are forbidden per AP-26.

**Verification**:
- [ ] `IdentityCritical` event with `ic: 0.45` results in dream trigger signal
- [ ] Lock acquisition failure causes panic (not silent failure)
- [ ] Existing `MemoryExits` handling is unchanged

---

### REQ-IDENTITY-005: TriggerConfig IC Threshold Field (HIGH)

**Description**: If a `TriggerConfig` struct exists for serialization/configuration, it MUST include an `ic_threshold` field with default 0.5.

**Source Issue**: ISS-002

**Constitution Reference**: `gwt.self_ego_node.thresholds`

**Required Change** (if TriggerConfig exists):
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerConfig {
    pub entropy_threshold: f32,
    pub entropy_window_secs: u64,
    pub gpu_threshold: f32,
    pub cooldown_secs: u64,
    /// IC crisis threshold (Constitution default: 0.5)
    #[serde(default = "default_ic_threshold")]
    pub ic_threshold: f32,
}

fn default_ic_threshold() -> f32 {
    0.5 // Constitution: gwt.self_ego_node.thresholds.critical
}
```

**Verification**:
- [ ] Config deserializes with default `ic_threshold = 0.5` when field is omitted
- [ ] Config with custom `ic_threshold` is respected
- [ ] Invalid threshold values (outside [0,1]) cause validation error

---

## 4. Edge Cases

| ID | Related Req | Scenario | Expected Behavior |
|----|-------------|----------|-------------------|
| EC-001 | REQ-IDENTITY-003 | IC exactly 0.5 | Dream MUST NOT trigger (threshold is `< 0.5`, strict inequality) |
| EC-002 | REQ-IDENTITY-003 | IC drops from 0.6 to 0.4 in single update | Dream triggers once with IC=0.4 |
| EC-003 | REQ-IDENTITY-003 | IC fluctuates: 0.49, 0.51, 0.49 rapidly | Cooldown prevents rapid re-triggering; only first 0.49 triggers |
| EC-004 | REQ-IDENTITY-003 | IC and GPU both in trigger state | IC trigger takes priority (identity crisis is more severe) |
| EC-005 | REQ-IDENTITY-004 | Multiple `IdentityCritical` events before dream starts | Only first event triggers dream; subsequent events are queued/logged |
| EC-006 | REQ-IDENTITY-003 | IC never set (None), other triggers met | Other triggers function normally; IC check is skipped |
| EC-007 | REQ-IDENTITY-003 | IC = NaN or Infinity | Value is clamped to [0, 1]; NaN becomes 0.0 with warning log |
| EC-008 | REQ-IDENTITY-004 | Dream already in progress when IC crisis occurs | Log event, do not interrupt current dream; crisis will be addressed |
| EC-009 | REQ-IDENTITY-001 | `IdentityCritical` display/serialization | Displays as "identity_critical(ic=0.45)"; serializes with ic_value field |
| EC-010 | REQ-IDENTITY-003 | TriggerManager disabled when IC drops below threshold | No trigger (disabled state is respected), but warn log is emitted |

---

## 5. Error States

| ID | HTTP Code | Condition | User-Visible Message | Recovery Action |
|----|-----------|-----------|---------------------|-----------------|
| ERR-IDENTITY-001 | N/A | Lock poisoned on TriggerManager | N/A (system internal) | Panic with "TriggerManager lock poisoned during IC update" |
| ERR-IDENTITY-002 | N/A | Lock poisoned on dream_queue | N/A (system internal) | Panic with "DreamEventListener: Lock poisoned or deadlocked" |
| ERR-IDENTITY-003 | N/A | Unable to signal dream trigger for IC crisis | N/A (system internal) | Panic with "Cannot signal IC crisis - AP-26 violation" (fail-fast) |
| ERR-IDENTITY-004 | N/A | IC threshold outside [0, 1] at construction | N/A (system internal) | Panic with "IC threshold must be in [0.0, 1.0], got: {value}" |
| ERR-IDENTITY-005 | N/A | IC value is NaN after computation | N/A (system internal) | Clamp to 0.0, emit warning "IC computation returned NaN, treating as 0.0" |

**Fail-Fast Philosophy**: Per AP-26, silent failures on IC crisis are FORBIDDEN. If the system cannot properly handle an IC < 0.5 condition, it MUST fail loudly rather than silently ignore the crisis.

---

## 6. Test Plan

### 6.1 Unit Tests

| ID | Type | Req Ref | Description | Inputs | Expected Output |
|----|------|---------|-------------|--------|-----------------|
| TC-IDENTITY-001 | unit | REQ-IDENTITY-001 | `ExtendedTriggerReason::IdentityCritical` serialization | `IdentityCritical { ic_value: 0.45 }` | Serializes to JSON with ic_value field |
| TC-IDENTITY-002 | unit | REQ-IDENTITY-001 | `ExtendedTriggerReason::IdentityCritical` display | `IdentityCritical { ic_value: 0.45 }` | "identity_critical(ic=0.45)" |
| TC-IDENTITY-003 | unit | REQ-IDENTITY-002 | `TriggerManager::new()` IC threshold default | N/A | `ic_threshold == 0.5` |
| TC-IDENTITY-004 | unit | REQ-IDENTITY-003 | IC below threshold triggers | `update_identity_coherence(0.49)` | `check_triggers() == Some(IdentityCritical { ic_value: 0.49 })` |
| TC-IDENTITY-005 | unit | REQ-IDENTITY-003 | IC at threshold does not trigger | `update_identity_coherence(0.50)` | `check_triggers() == None` |
| TC-IDENTITY-006 | unit | REQ-IDENTITY-003 | IC above threshold does not trigger | `update_identity_coherence(0.51)` | `check_triggers() == None` |
| TC-IDENTITY-007 | unit | REQ-IDENTITY-003 | IC priority over GPU | `update_identity_coherence(0.49); update_gpu_usage(0.35)` | `check_triggers() == Some(IdentityCritical)` |
| TC-IDENTITY-008 | unit | REQ-IDENTITY-003 | Manual priority over IC | `update_identity_coherence(0.49); request_manual_trigger()` | `check_triggers() == Some(Manual)` |
| TC-IDENTITY-009 | unit | REQ-IDENTITY-003 | IC None, GPU triggers | `update_gpu_usage(0.35)` (no IC set) | `check_triggers() == Some(GpuOverload)` |
| TC-IDENTITY-010 | unit | REQ-IDENTITY-003 | IC NaN handling | `update_identity_coherence(f32::NAN)` | IC clamped to 0.0, triggers crisis |

### 6.2 Integration Tests

| ID | Type | Req Ref | Description | Setup | Steps | Expected Result |
|----|------|---------|-------------|-------|-------|-----------------|
| TC-IDENTITY-011 | integration | REQ-IDENTITY-004 | Full IC crisis to dream trigger flow | GWTSystem with DreamEventListener | 1. Emit `IdentityCritical { ic: 0.45 }` event 2. Wait 100ms 3. Check dream state | Dream is triggered with `IdentityCritical` reason |
| TC-IDENTITY-012 | integration | REQ-IDENTITY-003, REQ-IDENTITY-004 | End-to-end IC monitoring | GWTSystem with IdentityContinuityListener | 1. Insert memory with low alignment PV 2. Wait for IC computation 3. Check if dream triggered | If IC < 0.5, dream is automatically triggered |
| TC-IDENTITY-013 | integration | REQ-IDENTITY-003 | Cooldown prevents rapid re-trigger | TriggerManager with 100ms cooldown | 1. `update_identity_coherence(0.49)` 2. `mark_triggered()` 3. Wait 50ms 4. `update_identity_coherence(0.48)` 5. `check_triggers()` | Returns None (cooldown active) |
| TC-IDENTITY-014 | integration | REQ-IDENTITY-003 | Cooldown allows re-trigger after expiry | TriggerManager with 100ms cooldown | 1. `update_identity_coherence(0.49)` 2. `mark_triggered()` 3. Wait 150ms 4. `update_identity_coherence(0.48)` 5. `check_triggers()` | Returns `IdentityCritical { ic_value: 0.48 }` |

### 6.3 Fail-Fast Tests

| ID | Type | Req Ref | Description | Setup | Steps | Expected Result |
|----|------|---------|-------------|-------|-------|-----------------|
| TC-IDENTITY-015 | unit | REQ-IDENTITY-002 | Invalid IC threshold panics | N/A | `TriggerManager::with_ic_threshold(1.5)` | Panic with clear error message |
| TC-IDENTITY-016 | unit | REQ-IDENTITY-002 | Negative IC threshold panics | N/A | `TriggerManager::with_ic_threshold(-0.1)` | Panic with clear error message |

### 6.4 Constitution Compliance Tests

| ID | Type | Req Ref | Constitution Rule | Test Description |
|----|------|---------|-------------------|------------------|
| TC-CONST-001 | integration | REQ-IDENTITY-003 | AP-26 | "IC<0.5 MUST trigger dream - no silent failures" - IC=0.49 MUST result in dream trigger |
| TC-CONST-002 | integration | REQ-IDENTITY-003 | AP-38 | "IC<0.5 MUST auto-trigger dream" - No manual intervention required |
| TC-CONST-003 | integration | REQ-IDENTITY-001 | IDENTITY-007 | "IC < 0.5 -> auto-trigger dream" - Verify automatic triggering |
| TC-CONST-004 | unit | REQ-IDENTITY-002 | gwt.self_ego_node.thresholds | Default threshold is 0.5 per constitution |

---

## 7. Implementation Dependencies

### 7.1 Prerequisite Specs

| Spec ID | Requirement | Reason |
|---------|-------------|--------|
| SPEC-GWT-001 | REQ-GWT-001 (13 oscillators) | IC calculation uses `r(t)` from Kuramoto network |
| SPEC-GWT-001 | REQ-GWT-002 (13 frequencies) | Correct frequencies needed for accurate `r(t)` |
| SPEC-GWT-001 | REQ-GWT-003, REQ-GWT-004 (stepper wired) | Oscillators must evolve for `r(t)` to change |

### 7.2 Implementation Order

1. **REQ-IDENTITY-001**: Add `IdentityCritical` variant to `ExtendedTriggerReason`
2. **REQ-IDENTITY-002**: Add IC threshold field to `TriggerManager`
3. **REQ-IDENTITY-003**: Implement `update_identity_coherence()` and modify `check_triggers()`
4. **REQ-IDENTITY-004**: Wire `DreamEventListener` to signal dream trigger
5. **REQ-IDENTITY-005**: Update `TriggerConfig` if it exists

### 7.3 Files to Modify

| File | Changes |
|------|---------|
| `crates/context-graph-core/src/dream/types.rs` | Add `IdentityCritical` variant, update `Display` impl |
| `crates/context-graph-core/src/dream/triggers.rs` | Add IC fields, `update_identity_coherence()`, modify `check_triggers()` |
| `crates/context-graph-core/src/gwt/listeners/dream.rs` | Add trigger signal mechanism, handle `IdentityCritical` event |
| `crates/context-graph-core/src/dream/mod.rs` | Export new types if needed |

---

## 8. Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| IC crisis triggers dream | Test pass rate | 100% (TC-CONST-001, TC-CONST-002, TC-CONST-003) |
| No silent IC failures | Static analysis + tests | Zero occurrences of ignored IC < 0.5 |
| Trigger latency | p95 latency from IC detection to dream trigger | < 100ms |
| Constitution compliance | All AP-26, AP-38, IDENTITY-007 tests | Pass |

---

## 9. Appendix: Constitution References

### Relevant Constitution Sections

```yaml
# From constitution.yaml

forbidden:
  AP-26: "IC<0.5 MUST trigger dream - no silent failures"
  AP-38: "IC<0.5 MUST auto-trigger dream"
  AP-40: "IdentityContinuityListener MUST subscribe to GWT"

gwt:
  self_ego_node:
    identity_continuity: "IC = cos(PV_t, PV_{t-1}) x r(t)"
    thresholds: { healthy: ">0.9", warning: "<0.7", critical: "<0.5 -> dream" }

enforcement:
  identity:
    IDENTITY-001: "IC = cos(PV_t, PV_{t-1}) x r(t)"
    IDENTITY-002: "Thresholds: Healthy>0.9, Warning[0.7,0.9], Degraded[0.5,0.7), Critical<0.5"
    IDENTITY-004: "IdentityContinuityMonitor struct required"
    IDENTITY-006: "IdentityContinuityListener subscribes to workspace events"
    IDENTITY-007: "IC < 0.5 -> auto-trigger dream"
```

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-12 | Architecture Team | Initial specification |

---

**END OF SPECIFICATION**
