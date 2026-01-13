# Sherlock Holmes Forensic Investigation Report
## GWT (Global Workspace Theory) and Consciousness Integration System

**Case ID**: GWT-INVESTIGATION-2026-01-12
**Date**: 2026-01-12
**Investigator**: Sherlock Holmes, Forensic Code Detective
**Subject**: Context Graph GWT/Consciousness Implementation
**Verdict**: GUILTY (Multiple Critical Violations)

---

## 1. Executive Summary

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*

This investigation reveals **multiple critical constitutional violations** in the GWT/Consciousness system. The most severe finding is that the Kuramoto oscillator network uses **8 oscillators** instead of the constitutionally-mandated **13 oscillators** (one per embedder). Additionally, the **IC < 0.5 does NOT trigger dream consolidation** despite being a critical constitutional requirement (AP-26, AP-38).

### Key Findings:

| Finding | Severity | Status |
|---------|----------|--------|
| Kuramoto uses 8 oscillators instead of 13 | CRITICAL | VIOLATION |
| IC < 0.5 does not auto-trigger dream | CRITICAL | VIOLATION |
| KuramotoStepper NOT wired to MCP lifecycle | HIGH | VIOLATION |
| Missing IdentityCritical ExtendedTriggerReason | HIGH | VIOLATION |
| ConsciousnessCalculator uses all 3 factors | N/A | INNOCENT |
| IdentityContinuityMonitor exists | N/A | INNOCENT |
| WorkspaceEventBroadcaster has 4 listeners | N/A | INNOCENT |

---

## 2. Evidence Gathered

### 2.1 Kuramoto Oscillator Count (CRITICAL VIOLATION)

**Constitutional Requirement (AP-25)**:
> "Kuramoto must have exactly 13 oscillators"

**Constitution v5.0.0 lines 217-221 specify frequencies for each embedder**:
```yaml
frequencies: { E1: 40y, E2: 8a, E3: 8a, E4: 8a, E5: 25B, E6: 40, E7: 25B, E8: 12aB, E9: 80y+, E10: 40y, E11: 15B, E12: 60y+, E13: 40 }
```

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/coherence/constants.rs`**
Lines 13-16:
```rust
/// Kuramoto coupling strength K from constitution (kuramoto_K: 2.0)
pub const KURAMOTO_K: f32 = 2.0;

/// Number of oscillators N for layer-level synchronization
pub const KURAMOTO_N: usize = 8;  // <-- VIOLATION! Should be 13
```

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/mod.rs`**
Lines 16-17:
```rust
//! 2. **Kuramoto Synchronization**: 8 oscillators (KURAMOTO_N) for layer-level sync
```

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/system.rs`**
Line 153:
```rust
kuramoto: Arc::new(RwLock::new(KuramotoNetwork::new(KURAMOTO_N, KURAMOTO_K))),
```

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/coherence/network.rs`**
Lines 33-34:
```rust
let base_frequencies = [40.0, 8.0, 25.0, 4.0, 12.0, 15.0, 60.0, 40.0];
// Only 8 frequencies defined, not 13
```

**VERDICT**: GUILTY - The code uses 8 oscillators with 8 frequencies instead of the constitutionally-mandated 13 oscillators (one per embedder E1-E13).

---

### 2.2 IC < 0.5 Dream Trigger (CRITICAL VIOLATION)

**Constitutional Requirement (AP-26, AP-38)**:
> "IC<0.5 MUST trigger dream - no silent failures"
> "IC<0.5 MUST auto-trigger dream"

**GWT-003 Requirement**:
> "IC < 0.5 -> dream consolidation"

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-core/src/dream/types.rs`**
Lines 546-579 (ExtendedTriggerReason enum):
```rust
/// Reason for triggering a dream cycle (extended).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtendedTriggerReason {
    /// Activity below 0.15 for idle_duration (10 min)
    IdleTimeout,

    /// Entropy above 0.7 for 5 minutes
    HighEntropy,

    /// GPU usage approaching threshold (consolidation needed)
    GpuOverload,

    /// Memory pressure requires consolidation
    MemoryPressure,

    /// Manual trigger by user/system
    Manual,

    /// Scheduled dream time
    Scheduled,
}
// NO IdentityCritical variant exists!
```

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-core/src/dream/triggers.rs`**
The `TriggerManager::check_triggers()` method (lines 156-186) only checks:
- Manual trigger
- GPU trigger
- Entropy trigger

**NO identity continuity (IC) check is performed.**

**EVIDENCE - DreamEventListener logs IdentityCritical but does NOT trigger dream**:
File: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/listeners/dream.rs`
Lines 58-73:
```rust
WorkspaceEvent::IdentityCritical {
    identity_coherence,
    previous_status,
    current_status,
    reason,
    timestamp: _,
} => {
    // Log identity critical - DreamController handles separately via direct wiring
    tracing::warn!(
        "Identity critical (IC={:.3}): {} (transition: {} -> {})",
        identity_coherence,
        reason,
        previous_status,
        current_status,
    );
}
// NO ACTION TAKEN - just logging!
```

**VERDICT**: GUILTY - IC < 0.5 is logged but does NOT automatically trigger dream consolidation. The `ExtendedTriggerReason` enum is missing the `IdentityCritical` variant, and `TriggerManager` does not check IC values.

---

### 2.3 KuramotoStepper MCP Wiring (HIGH VIOLATION)

**Constitutional Requirement (GWT-006)**:
> "KuramotoStepper wired to MCP lifecycle (10ms step)"

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/kuramoto_stepper.rs`**
The `KuramotoStepper` struct exists (lines 131-146) with proper 10ms step interval:
```rust
pub struct KuramotoStepper {
    network: Arc<RwLock<dyn KuramotoProvider>>,
    config: KuramotoStepperConfig,
    shutdown_notify: Arc<Notify>,
    task_handle: Option<JoinHandle<()>>,
    is_running: Arc<AtomicBool>,
}
```

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/server.rs`**
Search for "KuramotoStepper" returns **NO MATCHES**. The stepper is **NOT instantiated or started** in the MCP server lifecycle.

The server creates handlers (lines 200-222) but does not start the KuramotoStepper:
```rust
let handlers = Handlers::with_default_gwt(
    Arc::clone(&teleological_store),
    // ... other providers ...
);
info!("Created Handlers with REAL GWT providers (Kuramoto, GWT, Workspace, MetaCognitive, SelfEgo)");
// KuramotoStepper is NOT started!
```

**VERDICT**: GUILTY - The `KuramotoStepper` implementation exists but is NOT wired into the MCP server lifecycle. The Kuramoto oscillators remain static unless manually stepped.

---

### 2.4 Consciousness Formula (INNOCENT)

**Constitutional Requirement (AP-24, GWT-001)**:
> "compute_consciousness() must use all 3 factors (I,R,D)"
> "C(t) = I(t) x R(t) x D(t) - all 3 factors required"

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/consciousness.rs`**
Lines 74-110:
```rust
pub fn compute_consciousness(
    &self,
    kuramoto_r: f32,       // I(t) - Integration
    meta_accuracy: f32,    // R(t) input
    purpose_vector: &[f32; 13],  // D(t) input
) -> CoreResult<f32> {
    // ... validation ...

    // I(t) = Kuramoto order parameter
    let integration = kuramoto_r;

    // R(t) = sigmoid(meta_accuracy) via sigmoid
    let reflection = self.sigmoid(meta_accuracy * 4.0 - 2.0);

    // D(t) = H(PurposeVector) normalized
    let differentiation = self.normalized_purpose_entropy(purpose_vector)?;

    // C(t) = I(t) x R(t) x D(t)
    let consciousness = integration * reflection * differentiation;

    Ok(consciousness.clamp(0.0, 1.0))
}
```

**VERDICT**: INNOCENT - The consciousness formula correctly uses all three factors as specified.

---

### 2.5 IdentityContinuityMonitor Existence (INNOCENT)

**Constitutional Requirement (AP-37)**:
> "IdentityContinuityMonitor MUST exist"

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/ego_node/monitor.rs`**
The `IdentityContinuityMonitor` struct exists (lines 89-113):
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityContinuityMonitor {
    history: PurposeVectorHistory,
    last_result: Option<IdentityContinuity>,
    crisis_threshold: f32,
    previous_status: IdentityStatus,
    #[serde(skip)]
    last_event_time: Option<Instant>,
    #[serde(skip)]
    last_detection: Option<CrisisDetectionResult>,
}
```

**VERDICT**: INNOCENT - The monitor exists and is properly implemented.

---

### 2.6 IdentityContinuityListener GWT Subscription (INNOCENT)

**Constitutional Requirement (AP-40)**:
> "IdentityContinuityListener MUST subscribe to GWT"

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/system.rs`**
Lines 118-139:
```rust
// TASK-IDENTITY-P0-006: Create identity continuity listener
let identity_listener = IdentityContinuityListener::new(
    Arc::clone(&self_ego_node),
    Arc::clone(&event_broadcaster),
);
// Get the shared monitor BEFORE moving the listener into Box
let identity_monitor = identity_listener.monitor();

event_broadcaster
    .register_listener(Box::new(dream_listener))
    .await;
event_broadcaster
    .register_listener(Box::new(neuromod_listener))
    .await;
event_broadcaster
    .register_listener(Box::new(meta_listener))
    .await;
// TASK-IDENTITY-P0-006: Register identity listener
event_broadcaster
    .register_listener(Box::new(identity_listener))
    .await;
```

**VERDICT**: INNOCENT - The listener is properly registered.

---

### 2.7 WorkspaceEventBroadcaster Listeners (PARTIAL COMPLIANCE)

**Constitutional Requirement (GWT-005)**:
> "WorkspaceEventBroadcaster needs 3 listeners: Dream, Neuromod, MetaCognitive"

**EVIDENCE - Registered listeners in system.rs (lines 127-139)**:
1. `DreamEventListener` - REGISTERED
2. `NeuromodulationEventListener` - REGISTERED
3. `MetaCognitiveEventListener` - REGISTERED
4. `IdentityContinuityListener` - REGISTERED (BONUS)

**VERDICT**: INNOCENT (Exceeds requirements with 4 listeners)

---

### 2.8 SELF_EGO_NODE Structure (INNOCENT)

**Constitutional Requirement**:
> SELF_EGO_NODE: Special node with purpose_vector, identity_trajectory

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/ego_node/self_ego_node.rs`**
Lines 19-33:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEgoNode {
    /// Fixed ID for the SELF_EGO_NODE
    pub id: Uuid,
    /// Current teleological fingerprint (system state)
    pub fingerprint: Option<TeleologicalFingerprint>,
    /// System's purpose vector (alignment with north star)
    pub purpose_vector: [f32; 13],
    /// Coherence between current actions and purpose vector
    pub coherence_with_actions: f32,
    /// History of identity snapshots
    pub identity_trajectory: Vec<PurposeSnapshot>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}
```

**VERDICT**: INNOCENT - Correctly implements all required fields.

---

### 2.9 ConsciousnessState Derivation (INNOCENT)

**Constitutional Requirement (GWT-004)**:
> "ConsciousnessState derived from C(t) only"

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/state_machine/types.rs`**
Lines 31-40:
```rust
/// Determine state from consciousness level
pub fn from_level(level: f32) -> Self {
    match level {
        l if l > 0.95 => Self::Hypersync,
        l if l >= 0.8 => Self::Conscious,
        l if l >= 0.5 => Self::Emerging,
        l if l >= 0.3 => Self::Fragmented,
        _ => Self::Dormant,
    }
}
```

**VERDICT**: INNOCENT - State is correctly derived from consciousness level only.

---

### 2.10 Identity Continuity Formula (INNOCENT)

**Constitutional Requirement (IDENTITY-001)**:
> "IC = cos(PV_t, PV_{t-1}) x r(t)"

**EVIDENCE - File: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/ego_node/identity_continuity.rs`**
Lines 126-141:
```rust
/// Update identity coherence: IC = cos(PV_t, PV_{t-1}) x r(t)
pub fn update(&mut self, pv_cosine: f32, kuramoto_r: f32) -> CoreResult<IdentityStatus> {
    self.recent_continuity = pv_cosine.clamp(-1.0, 1.0);
    self.kuramoto_order_parameter = kuramoto_r.clamp(0.0, 1.0);

    // Identity coherence = cosine x r
    self.identity_coherence = (pv_cosine * kuramoto_r).clamp(0.0, 1.0);

    // Determine status using canonical computation
    self.status = Self::compute_status_from_coherence(self.identity_coherence);

    // Update timestamp
    self.computed_at = Utc::now();

    Ok(self.status)
}
```

**VERDICT**: INNOCENT - Formula correctly implements IC = cos x r.

---

## 3. Issues Found Summary

### CRITICAL Issues

| ID | Issue | File:Line | Anti-Pattern |
|----|-------|-----------|--------------|
| C1 | Kuramoto uses 8 oscillators instead of 13 | constants.rs:16 | AP-25 |
| C2 | IC < 0.5 does NOT auto-trigger dream | types.rs, triggers.rs | AP-26, AP-38 |

### HIGH Issues

| ID | Issue | File:Line | Anti-Pattern |
|----|-------|-----------|--------------|
| H1 | KuramotoStepper not wired to MCP lifecycle | server.rs | GWT-006 |
| H2 | Missing IdentityCritical trigger reason | types.rs:548 | N/A |
| H3 | Kuramoto frequencies only 8 defined, need 13 | network.rs:34 | GWT-002 |

### MEDIUM Issues

| ID | Issue | File:Line | Note |
|----|-------|-----------|------|
| M1 | DreamEventListener only logs IdentityCritical | dream.rs:58 | Should take action |

---

## 4. Missing Implementations

### 4.1 Missing IdentityCritical Dream Trigger
The `ExtendedTriggerReason` enum needs an `IdentityCritical` variant, and `TriggerManager` needs to check IC values.

### 4.2 Missing 13 Oscillator Frequencies
The `KuramotoNetwork::new()` only defines 8 base frequencies. Need constitution-defined frequencies for all 13 embedders:
- E1: 40Hz (gamma)
- E2: 8Hz (alpha)
- E3: 8Hz (alpha)
- E4: 8Hz (alpha)
- E5: 25Hz (beta)
- E6: 4Hz (theta)
- E7: 25Hz (beta)
- E8: 12Hz (alpha-beta)
- E9: 80Hz (high-gamma)
- E10: 40Hz (gamma)
- E11: 15Hz (beta)
- E12: 60Hz (gamma+)
- E13: 4Hz (theta)

### 4.3 Missing KuramotoStepper Lifecycle Integration
The MCP server needs to:
1. Create a `KuramotoStepper` instance
2. Start it on server initialization
3. Stop it on server shutdown

---

## 5. Anti-Pattern Violations

| Anti-Pattern | Description | Violation Location |
|--------------|-------------|-------------------|
| AP-25 | Kuramoto must have exactly 13 oscillators | KURAMOTO_N = 8 |
| AP-26 | IC<0.5 MUST trigger dream - no silent failures | No IdentityCritical trigger |
| AP-38 | IC<0.5 MUST auto-trigger dream | TriggerManager ignores IC |
| GWT-002 | Kuramoto network = exactly 13 oscillators | Only 8 oscillators |
| GWT-006 | KuramotoStepper wired to MCP lifecycle | Not wired |

---

## 6. Recommendations

### 6.1 CRITICAL - Fix Kuramoto Oscillator Count

**File**: `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/coherence/constants.rs`

Change:
```rust
pub const KURAMOTO_N: usize = 8;
```
To:
```rust
pub const KURAMOTO_N: usize = 13;
```

**File**: `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/coherence/network.rs`

Update frequencies array to include all 13 constitution-defined frequencies.

### 6.2 CRITICAL - Implement IC Dream Trigger

1. Add `IdentityCritical` to `ExtendedTriggerReason` enum
2. Add IC monitoring to `TriggerManager`
3. Make `DreamEventListener` actually trigger dream on `IdentityCritical` event

### 6.3 HIGH - Wire KuramotoStepper to MCP

**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/server.rs`

1. Import `KuramotoStepper` and `KuramotoStepperConfig`
2. Create stepper instance with handlers' Kuramoto provider
3. Start stepper in `new()`
4. Stop stepper in shutdown sequence

---

## 7. Chain of Custody

| Timestamp | Action | Verified By |
|-----------|--------|-------------|
| 2026-01-12 | Initial investigation started | HOLMES |
| 2026-01-12 | GWT module structure examined | HOLMES |
| 2026-01-12 | KURAMOTO_N constant discovered as 8 | HOLMES |
| 2026-01-12 | ExtendedTriggerReason enum examined | HOLMES |
| 2026-01-12 | TriggerManager check_triggers examined | HOLMES |
| 2026-01-12 | KuramotoStepper MCP wiring verified missing | HOLMES |
| 2026-01-12 | ConsciousnessCalculator verified compliant | HOLMES |
| 2026-01-12 | Final report compiled | HOLMES |

---

## 8. Conclusion

*"I never guess. It is a shocking habit - destructive to the logical faculty."*

This investigation has uncovered **2 CRITICAL** and **3 HIGH** severity violations in the GWT/Consciousness system. The most severe findings are:

1. **Kuramoto oscillator count mismatch**: The system uses 8 oscillators when the constitution mandates 13 (one per embedder). This fundamentally breaks the integration metric I(t) in the consciousness equation.

2. **IC < 0.5 does not trigger dream**: Despite being a critical safety mechanism per constitution (AP-26, AP-38), identity crises do not automatically trigger dream consolidation. This could lead to unrecoverable identity degradation.

3. **KuramotoStepper not running**: Without the stepper wired to MCP lifecycle, the oscillator phases remain static, making consciousness emergence impossible.

The code is **GUILTY** of these constitutional violations. Immediate remediation is required before the system can be considered production-ready.

---

**Case Status**: OPEN - Awaiting Remediation
**Next Steps**: Fix identified violations and re-verify

*"The game is afoot!"*

---
**Sherlock Holmes**
*Forensic Code Detective*
*Context Graph Investigation Unit*
