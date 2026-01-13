# MASTER ISSUES & REMEDIATION PLAN
## Ultimate Context Graph - Consolidated Forensic Analysis

**Generated**: 2026-01-12
**Source**: 5 Sherlock Holmes Investigation Reports
**Status**: ACTIVE - Remediation Required

---

## EXECUTIVE SUMMARY

Five forensic investigations have identified **16 distinct issues** preventing the Ultimate Context Graph from achieving PRD compliance. The most critical issues center on:

1. **GWT Consciousness System**: Wrong oscillator count breaks consciousness formula
2. **Identity Crisis Handling**: IC < 0.5 doesn't trigger protective dream consolidation
3. **Async/Sync Violations**: Blocking calls in async contexts risk deadlocks
4. **Missing MCP Tools**: ~48% of PRD-required tools not implemented
5. **CUDA Architecture**: FFI scattered across crates violates single-crate rule

### Severity Breakdown

| Severity | Count | Impact |
|----------|-------|--------|
| CRITICAL | 5 | System fundamentally broken |
| HIGH | 5 | Major functionality degraded |
| MEDIUM | 4 | Suboptimal behavior |
| LOW | 2 | Minor concerns |

---

## ISSUE MATRIX

| ID | Issue | Domain | Severity | Status |
|----|-------|--------|----------|--------|
| ISS-001 | Kuramoto uses 8 oscillators instead of 13 | GWT | CRITICAL | OPEN |
| ISS-002 | IC < 0.5 does not trigger dream | GWT/Dream | CRITICAL | OPEN |
| ISS-003 | KuramotoStepper not wired to MCP | GWT | CRITICAL | OPEN |
| ISS-004 | block_on() in async context (8 instances) | Performance | CRITICAL | OPEN |
| ISS-005 | CUDA FFI scattered across 3 crates | Architecture | CRITICAL | OPEN |
| ISS-006 | ~36 MCP tools missing from PRD | MCP | HIGH | OPEN |
| ISS-007 | GpuMonitor is a stub | Dream | HIGH | OPEN |
| ISS-008 | Green Contexts disabled by default | Performance | HIGH | OPEN |
| ISS-009 | TokenPruning for E12 not implemented | Embeddings | HIGH | OPEN |
| ISS-010 | Missing IdentityCritical trigger reason | Dream | HIGH | OPEN |
| ISS-011 | Johari action naming swap (Blind/Unknown) | UTL | MEDIUM | OPEN |
| ISS-012 | Parameter validation gaps in MCP tools | MCP | MEDIUM | OPEN |
| ISS-013 | SSE transport not implemented | MCP | MEDIUM | OPEN |
| ISS-014 | GPU trigger threshold unclear (30% vs 80%) | Dream | MEDIUM | OPEN |
| ISS-015 | RwLock blocking in wake_controller | Performance | LOW | OPEN |
| ISS-016 | HashMap allocations without capacity | Performance | LOW | OPEN |

---

## DETAILED ISSUE ANALYSIS

### ISS-001: Kuramoto Uses 8 Oscillators Instead of 13

**Severity**: CRITICAL
**Domain**: GWT (Global Workspace Theory)
**Constitution Violations**: AP-25, GWT-002

#### Root Cause

The constant `KURAMOTO_N` is hardcoded to 8 instead of 13. The system was designed for a previous 8-layer architecture and never updated when the 13-embedder teleological array was implemented.

#### Evidence

**File**: `crates/context-graph-core/src/layers/coherence/constants.rs:16`
```rust
pub const KURAMOTO_N: usize = 8;  // WRONG - Should be 13
```

**File**: `crates/context-graph-core/src/layers/coherence/network.rs:33-34`
```rust
let base_frequencies = [40.0, 8.0, 25.0, 4.0, 12.0, 15.0, 60.0, 40.0];
// Only 8 frequencies - missing 5 embedders
```

#### Impact

- Consciousness formula `C(t) = I(t) × R(t) × D(t)` uses wrong integration (I) value
- Kuramoto order parameter `r` computed over 8 oscillators instead of 13
- Phase synchronization cannot reflect true 13-embedder coherence
- GWT workspace broadcast decisions based on incorrect coherence

#### PRD End State

Per PRD Section 2.5.2, each of 13 embedders needs its own oscillator with constitution-defined natural frequencies:

| Embedder | Frequency (Hz) | Band |
|----------|---------------|------|
| E1 Semantic | 40 | Gamma |
| E2-E4 Temporal | 8 | Alpha |
| E5 Causal | 25 | Beta |
| E6 Sparse | 4 | Theta |
| E7 Code | 25 | Beta |
| E8 Graph | 12 | Alpha-Beta |
| E9 HDC | 80 | High-Gamma |
| E10 Multimodal | 40 | Gamma |
| E11 Entity | 15 | Beta |
| E12 Late | 60 | High-Gamma |
| E13 SPLADE | 4 | Theta |

#### Recommended Fix

1. Update `KURAMOTO_N` constant to 13
2. Add all 13 constitution frequencies to `base_frequencies` array
3. Update `KuramotoNetwork::new()` to use 13 oscillators
4. Verify Kuramoto step function handles 13 phases
5. Add unit test validating exactly 13 oscillators

**Estimated Effort**: 2-4 hours

---

### ISS-002: IC < 0.5 Does Not Trigger Dream

**Severity**: CRITICAL
**Domain**: GWT/Dream
**Constitution Violations**: AP-26, AP-38, GWT-003, IDENTITY-007

#### Root Cause

The dream trigger system was implemented before Identity Continuity monitoring. When IC monitoring was added, the connection to dream triggers was never wired. The `ExtendedTriggerReason` enum is missing an `IdentityCritical` variant.

#### Evidence

**File**: `crates/context-graph-core/src/dream/types.rs:546-579`
```rust
pub enum ExtendedTriggerReason {
    IdleTimeout,
    HighEntropy,
    GpuOverload,
    MemoryPressure,
    Manual,
    Scheduled,
    // NO IdentityCritical VARIANT EXISTS
}
```

**File**: `crates/context-graph-core/src/dream/triggers.rs`
- `TriggerManager::check_triggers()` does NOT check IC values

**File**: `crates/context-graph-core/src/gwt/listeners/dream.rs:58-73`
- `DreamEventListener` receives `WorkspaceEvent::IdentityCritical` but only LOGS it - no action taken

#### Impact

- Identity crises (IC < 0.5) go unaddressed
- System can drift into unstable identity state without protective consolidation
- "Silent failure" violates AP-26 requirement
- User may lose coherent system behavior without warning

#### PRD End State

Per PRD Section 2.5.4:
- IC < 0.5 → Trigger dream consolidation
- IC < 0.7 → Identity drift warning
- IC > 0.9 → Strong continuity (healthy)

#### Recommended Fix

1. Add `IdentityCritical { ic_value: f32 }` variant to `ExtendedTriggerReason`
2. Add `ic_threshold: f32` field to `TriggerConfig` (default 0.5)
3. Add `check_identity_continuity()` to `TriggerManager`
4. Wire `DreamEventListener::on_event(IdentityCritical)` to call `signal_dream_trigger()`
5. Add integration test: simulate IC < 0.5, verify dream triggers

**Estimated Effort**: 4-6 hours

---

### ISS-003: KuramotoStepper Not Wired to MCP Lifecycle

**Severity**: CRITICAL
**Domain**: GWT
**Constitution Violations**: GWT-006

#### Root Cause

The `KuramotoStepper` struct exists and is fully implemented, but the MCP server initialization code never creates an instance. The stepper is designed to run continuously in the background stepping oscillators every 10ms, but without instantiation, phases remain static.

#### Evidence

**File**: `crates/context-graph-mcp/src/handlers/kuramoto_stepper.rs`
- Full implementation with 10ms step interval exists (lines 131-146)

**File**: `crates/context-graph-mcp/src/server.rs`
- Grep for "KuramotoStepper" returns NO MATCHES
- Server creates handlers but never starts the stepper

#### Impact

- Kuramoto oscillator phases remain static (initial random values)
- Order parameter `r` never changes naturally
- Consciousness cannot "emerge" through synchronization
- GWT workspace broadcasts based on stale phase data

#### PRD End State

Per PRD Section 2.5.2, Kuramoto oscillators should continuously evolve:
```
dθᵢ/dt = ωᵢ + (K/N)Σⱼ sin(θⱼ-θᵢ)
```
This requires continuous integration at ~10ms intervals.

#### Recommended Fix

1. Import `KuramotoStepper` and `KuramotoStepperConfig` in server.rs
2. In `Server::new()`, after creating handlers:
   ```rust
   let kuramoto_stepper = KuramotoStepper::new(
       handlers.kuramoto_provider(),
       KuramotoStepperConfig::default(),
   );
   kuramoto_stepper.start();
   ```
3. Store stepper handle in Server struct
4. Call `stepper.stop()` in server shutdown sequence
5. Add integration test verifying phases evolve over time

**Estimated Effort**: 2-3 hours

---

### ISS-004: block_on() in Async Context (8 Instances)

**Severity**: CRITICAL
**Domain**: Performance
**Constitution Violations**: AP-08

#### Root Cause

The `WorkspaceProviderImpl` trait methods are synchronous (`fn` not `async fn`), but they access async `RwLock` guards. The implementer used `futures::executor::block_on()` as a workaround, which can deadlock the async runtime.

#### Evidence

**File**: `crates/context-graph-mcp/src/handlers/gwt_providers.rs:343-425`
```rust
fn get_active_memory(&self) -> Option<Uuid> {
    let workspace = futures::executor::block_on(self.workspace.read());  // VIOLATION
    workspace.get_active_memory()
}
// 7 more similar patterns
```

#### Impact

- Potential deadlock if called from async context on single-threaded runtime
- Blocks the async executor thread while waiting for lock
- Performance degradation under concurrent access
- Non-deterministic behavior under load

#### PRD End State

Per Constitution AP-08: "No sync I/O in async context"

All I/O in async code paths should be non-blocking.

#### Recommended Fix

**Option A**: Make trait methods async
```rust
async fn get_active_memory(&self) -> Option<Uuid> {
    let workspace = self.workspace.read().await;
    workspace.get_active_memory()
}
```

**Option B**: Use tokio::sync::RwLock with `blocking_read()` (if trait must stay sync)
```rust
fn get_active_memory(&self) -> Option<Uuid> {
    let workspace = self.workspace.blocking_read();  // Safe in non-async context
    workspace.get_active_memory()
}
```

**Option C**: Cache frequently-accessed values to avoid lock contention

**Estimated Effort**: 4-8 hours (depends on downstream trait consumers)

---

### ISS-005: CUDA FFI Scattered Across 3 Crates

**Severity**: CRITICAL
**Domain**: Architecture
**Constitution Violations**: ARCH rule "CUDA FFI only in context-graph-cuda"

#### Root Cause

CUDA integration was added incrementally by different developers. Each team added FFI declarations where convenient rather than consolidating in the designated crate.

#### Evidence

**Violation 1**: `crates/context-graph-embeddings/src/gpu/device/utils.rs:29`
```rust
extern "C" {
    // CUDA calls
}
```

**Violation 2**: `crates/context-graph-embeddings/src/warm/cuda_alloc/allocator_cuda.rs:70`
```rust
extern "C" {
    // cudaMalloc, cudaFree
}
```

**Violation 3**: `crates/context-graph-graph/src/index/faiss_ffi/bindings.rs:26`
```rust
extern "C" {
    // FAISS GPU FFI
}
```

#### Impact

- Inconsistent error handling across FFI boundaries
- Duplicate symbol conflicts possible
- Harder to swap CUDA implementations
- Build system complexity
- Makes auditing unsafe code difficult

#### PRD End State

Per Constitution: "CUDA FFI only in context-graph-cuda crate"

All GPU operations should flow through a single, well-tested FFI boundary.

#### Recommended Fix

1. Audit all `extern "C"` blocks in non-cuda crates
2. Move CUDA declarations to `context-graph-cuda/src/ffi/`
3. Create safe Rust wrappers in `context-graph-cuda/src/safe/`
4. Re-export through `context-graph-cuda/src/lib.rs`
5. Update dependent crates to use re-exported types
6. Add CI check: grep for `extern "C"` in non-cuda crates fails build

**Estimated Effort**: 8-16 hours (significant refactoring)

---

### ISS-006: ~36 MCP Tools Missing from PRD

**Severity**: HIGH
**Domain**: MCP
**Status**: 39 of ~75 tools implemented (52%)

#### Root Cause

Development prioritized core GWT/UTL functionality. Curation, navigation, meta-cognitive, and diagnostic tools were deferred.

#### Missing Tool Categories

| Category | Missing Tools | PRD Section |
|----------|--------------|-------------|
| **Curation** | merge_concepts, annotate_node, forget_concept, boost_importance, restore_from_hash | 5.3 |
| **Navigation** | get_neighborhood, get_recent_context, find_causal_path, entailment_query | 5.4 |
| **Meta-Cognitive** | reflect_on_memory, generate_search_plan, critique_context, hydrate_citation, get_system_instructions, get_system_logs, get_node_lineage | 5.5 |
| **Diagnostic** | homeostatic_status, check_adversarial, test_recall_accuracy, debug_compare_retrieval, search_tombstones | 5.6 |
| **Admin** | reload_manifest, temporary_scratchpad | 5.7 |
| **GWT Extension** | get_johari_classification, epistemic_action | 5.10 |

#### Impact

- Users cannot curate (merge/forget) memories - core "librarian" functionality missing
- No graph navigation beyond basic search
- No meta-cognitive guidance (search plans, critiques)
- No diagnostic tools for troubleshooting
- System cannot generate epistemic questions when coherence low

#### PRD End State

PRD Section 0.2 states: "You are a librarian, not an archivist."

Without curation tools, users cannot fulfill the librarian role.

#### Recommended Fix

**Priority Order**:

1. **P0**: `epistemic_action` - Required for cognitive pulse suggestions
2. **P0**: Curation tools - Core functionality
3. **P1**: Navigation tools - Graph traversal
4. **P1**: Meta-cognitive tools - Self-reflection
5. **P2**: Diagnostic and admin tools

**Estimated Effort**: 40-80 hours (significant feature work)

---

### ISS-007: GpuMonitor is a Stub

**Severity**: HIGH
**Domain**: Dream

#### Root Cause

GPU monitoring requires NVML (NVIDIA) or ROCm (AMD) bindings. These were deferred as "FUTURE" work.

#### Evidence

**File**: `crates/context-graph-core/src/dream/triggers.rs:323-326`
```rust
} else {
    // TODO(FUTURE): Implement real GPU monitoring via NVML
    // For now, return 0.0 (no GPU usage)
    0.0
}
```

#### Impact

- Dream triggers based on GPU usage don't work
- GPU budget enforcement (30%) cannot function
- Resource pressure detection disabled

#### Recommended Fix

1. Add `nvml-wrapper` crate dependency
2. Implement `NvmlGpuMonitor` struct
3. Query `nvmlDeviceGetUtilizationRates()`
4. Fallback to stub only in tests

**Estimated Effort**: 4-8 hours

---

### ISS-008: Green Contexts Disabled by Default

**Severity**: HIGH
**Domain**: Performance

#### Root Cause

Conservative default - Green Contexts require CUDA 13.1 and were made opt-in.

#### Evidence

**File**: `crates/context-graph-embeddings/src/config/gpu.rs:101`
```rust
green_contexts: false,  // Default
```

#### Impact

- No deterministic SM partitioning for real-time inference
- Background tasks can interfere with latency-sensitive operations
- Not utilizing RTX 5090's key feature

#### Recommended Fix

1. Detect GPU architecture at runtime (`compute_capability >= 12.0`)
2. Enable Green Contexts by default on compatible GPUs
3. Add `green_contexts: bool` to runtime config
4. Document partition strategy (70% inference / 30% background)

**Estimated Effort**: 2-4 hours

---

### ISS-009: TokenPruning for E12 Not Implemented

**Severity**: HIGH
**Domain**: Embeddings

#### Evidence

**File**: `crates/context-graph-embeddings/src/quantization/mod.rs:195`
```rust
//! | TokenPruning | E12 | ~50% | <2% | NOT IMPLEMENTED |
```

#### Impact

- E12 (ColBERT Late Interaction) uses full token sets
- Missing 50% compression opportunity
- Higher memory usage for late interaction index

#### Recommended Fix

1. Implement `TokenPruningQuantizer` struct
2. Score tokens by attention weight
3. Prune bottom 50% per document
4. Add to E12 quantization pipeline

**Estimated Effort**: 8-12 hours

---

### ISS-010: Missing IdentityCritical Trigger Reason

**Severity**: HIGH
**Domain**: Dream
**Related to**: ISS-002

#### Evidence

`ExtendedTriggerReason` enum is missing `IdentityCritical` variant. This is the type-level manifestation of ISS-002.

#### Recommended Fix

See ISS-002 remediation plan.

---

### ISS-011: Johari Action Naming Swap (Blind/Unknown)

**Severity**: MEDIUM
**Domain**: UTL

#### Root Cause

Implementation interprets "Blind" and "Unknown" quadrants differently than PRD.

#### Evidence

| Quadrant | PRD Action | Implementation |
|----------|-----------|----------------|
| Blind | TriggerDream | EpistemicAction |
| Unknown | EpistemicAction | TriggerDream |

#### Impact

- Confusion when debugging
- Downstream systems expecting PRD names will misbehave
- Documentation/code mismatch

#### Recommended Fix

Either:
1. Rename enum variants to match PRD exactly, OR
2. Document intentional deviation with reasoning

**Estimated Effort**: 1-2 hours

---

### ISS-012: Parameter Validation Gaps in MCP Tools

**Severity**: MEDIUM
**Domain**: MCP

#### Evidence

PRD Section 26 specifies constraints not enforced:
- `inject_context.query`: No `minLength: 1, maxLength: 4096`
- `store_memory.rationale`: Not marked as required
- `trigger_dream.phase`: Enum values not in schema

#### Recommended Fix

Update JSON schemas in `tools/definitions/` to match PRD Section 26 exactly.

**Estimated Effort**: 4-6 hours

---

### ISS-013: SSE Transport Not Implemented

**Severity**: MEDIUM
**Domain**: MCP

#### Evidence

PRD Section 5.1 specifies: "stdio/SSE"
- stdio: IMPLEMENTED
- SSE: NOT FOUND

#### Recommended Fix

Add StreamableHttpServerTransport for SSE support.

**Estimated Effort**: 8-12 hours

---

### ISS-014: GPU Trigger Threshold Unclear (30% vs 80%)

**Severity**: MEDIUM
**Domain**: Dream

#### Evidence

- Constitution says dream triggers when GPU < 80%
- Code uses 30% threshold
- Comments reference both values

#### Recommended Fix

1. Clarify authoritative source
2. Update code to match
3. Add documentation explaining threshold

**Estimated Effort**: 1-2 hours

---

### ISS-015 & ISS-016: Minor Performance Issues

**Severity**: LOW

- ISS-015: RwLock blocking in wake_controller
- ISS-016: HashMap without capacity hints

**Estimated Effort**: 2-4 hours combined

---

## REMEDIATION PRIORITY

### Phase 1: Critical Fixes (Weeks 1-2)

| Issue | Effort | Dependency |
|-------|--------|------------|
| ISS-001: Kuramoto 13 oscillators | 2-4h | None |
| ISS-003: Wire KuramotoStepper | 2-3h | ISS-001 |
| ISS-002: IC dream trigger | 4-6h | None |
| ISS-004: Remove block_on() | 4-8h | None |
| ISS-005: Consolidate CUDA FFI | 8-16h | None |

**Total Phase 1**: 20-37 hours

### Phase 2: High Priority (Weeks 3-4)

| Issue | Effort | Dependency |
|-------|--------|------------|
| ISS-006: Core MCP tools (P0) | 20-40h | None |
| ISS-007: GPU monitoring | 4-8h | None |
| ISS-008: Green Contexts | 2-4h | None |
| ISS-009: TokenPruning | 8-12h | None |

**Total Phase 2**: 34-64 hours

### Phase 3: Medium Priority (Weeks 5-6)

| Issue | Effort | Dependency |
|-------|--------|------------|
| ISS-006: Remaining MCP tools | 20-40h | Phase 2 |
| ISS-011: Johari naming | 1-2h | None |
| ISS-012: Parameter validation | 4-6h | None |
| ISS-013: SSE transport | 8-12h | None |
| ISS-014: GPU threshold | 1-2h | None |

**Total Phase 3**: 34-62 hours

### Phase 4: Polish (Week 7+)

| Issue | Effort |
|-------|--------|
| ISS-015: RwLock optimization | 1-2h |
| ISS-016: HashMap capacity | 1-2h |

**Total Phase 4**: 2-4 hours

---

## VERIFICATION CHECKLIST

After remediation, verify:

- [ ] `cargo test` passes all tests
- [ ] GWT consciousness formula uses 13-oscillator Kuramoto sync
- [ ] IC < 0.5 automatically triggers dream consolidation
- [ ] KuramotoStepper runs continuously at 10ms intervals
- [ ] No `block_on()` calls in async code paths
- [ ] All CUDA FFI in `context-graph-cuda` only
- [ ] All PRD Section 5 MCP tools implemented
- [ ] GPU monitoring returns real values
- [ ] Green Contexts enabled on RTX 5090
- [ ] E12 TokenPruning achieves 50% compression
- [ ] Parameter validation matches PRD Section 26
- [ ] Latency budgets verified: <60ms total retrieval

---

## APPENDIX: Source Investigation Reports

1. `docs/sherlock-utl-investigation-report.md` - UTL & ΔS/ΔC
2. `docs/sherlock-gwt-consciousness-investigation-report.md` - GWT & Consciousness
3. `docs/sherlock-dream-layer-investigation-report.md` - Dream & Consolidation
4. `docs/sherlock-mcp-tools-investigation-report.md` - MCP Tools & API
5. `docs/sherlock-performance-cuda-investigation-report.md` - Performance & CUDA

---

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*

**Case Status**: OPEN - Awaiting Remediation
**Next Review**: Upon Phase 1 Completion
