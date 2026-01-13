# PRD Analysis: Ultimate Context Graph Remediation
## Analysis of MASTER-ISSUES-REMEDIATION-PLAN

**Generated**: 2026-01-12
**Source**: MASTER-ISSUES-REMEDIATION-PLAN.md + 5 Sherlock Investigation Reports
**Constitution**: v5.0.0
**PRD**: v4.0.0 (Global Workspace)

---

## User Types

| Type | Description | Permission |
|------|-------------|------------|
| System | Internal autonomous processes (Dream, Neuromod, GWT) | Full system access |
| MCP Client | External AI agents via MCP protocol | Tool-scoped access |
| Developer | Human developers maintaining the system | Full codebase access |
| Administrator | System administrators | Configuration + monitoring |

---

## User Journeys

1. **GWT Consciousness Calculation**: System computes C(t) = I(t) × R(t) × D(t) using 13-oscillator Kuramoto sync
2. **Identity Crisis Recovery**: When IC < 0.5, system automatically triggers dream consolidation
3. **Memory Storage**: MCP client stores memory, system computes TeleologicalFingerprint with all 13 embedders
4. **Dream Consolidation**: System runs NREM (Hebbian replay) + REM (hyperbolic walk) during idle
5. **Async MCP Operations**: All MCP tool calls execute without blocking the async runtime

---

## Functional Domains

- [x] **GWT Domain** - Global Workspace Theory consciousness implementation
- [x] **Identity Domain** - SELF_EGO_NODE and Identity Continuity monitoring
- [x] **Dream Domain** - NREM/REM consolidation and trigger management
- [x] **Performance Domain** - Async runtime, lock management, memory allocation
- [x] **Architecture Domain** - CUDA FFI consolidation, crate organization
- [x] **MCP Domain** - Tool implementation, parameter validation, transport
- [x] **Embeddings Domain** - Green Contexts, TokenPruning quantization
- [x] **UTL Domain** - Johari quadrant classification and actions

---

## Requirements Matrix

### GWT Domain Requirements

| ID | Requirement | Source Issue | Constitution Rule | Severity |
|----|-------------|--------------|-------------------|----------|
| REQ-GWT-001 | Kuramoto network MUST have exactly 13 oscillators | ISS-001 | AP-25, GWT-002 | CRITICAL |
| REQ-GWT-002 | Base frequencies array MUST contain all 13 embedder frequencies | ISS-001 | gwt.kuramoto.frequencies | CRITICAL |
| REQ-GWT-003 | KuramotoStepper MUST be instantiated in MCP server startup | ISS-003 | GWT-006 | CRITICAL |
| REQ-GWT-004 | KuramotoStepper MUST step oscillators every 10ms | ISS-003 | perf.latency | CRITICAL |
| REQ-GWT-005 | Server MUST call stepper.stop() on shutdown | ISS-003 | - | HIGH |

### Identity Domain Requirements

| ID | Requirement | Source Issue | Constitution Rule | Severity |
|----|-------------|--------------|-------------------|----------|
| REQ-IDENTITY-001 | IC < 0.5 MUST trigger dream consolidation | ISS-002 | AP-26, AP-38, IDENTITY-007 | CRITICAL |
| REQ-IDENTITY-002 | ExtendedTriggerReason MUST include IdentityCritical variant | ISS-010 | - | HIGH |
| REQ-IDENTITY-003 | TriggerManager.check_triggers() MUST check IC values | ISS-002 | IDENTITY-004 | CRITICAL |
| REQ-IDENTITY-004 | DreamEventListener MUST call signal_dream_trigger() on IdentityCritical | ISS-002 | IDENTITY-006 | CRITICAL |
| REQ-IDENTITY-005 | TriggerConfig MUST include ic_threshold field (default 0.5) | ISS-002 | gwt.self_ego_node.thresholds | HIGH |

### Dream Domain Requirements

| ID | Requirement | Source Issue | Constitution Rule | Severity |
|----|-------------|--------------|-------------------|----------|
| REQ-DREAM-001 | GpuMonitor MUST return real GPU utilization values | ISS-007 | - | HIGH |
| REQ-DREAM-002 | GpuMonitor MUST use NVML for NVIDIA GPUs | ISS-007 | stack.gpu | HIGH |
| REQ-DREAM-003 | GPU trigger threshold MUST be clarified and consistent | ISS-014 | dream.constraints.gpu | MEDIUM |
| REQ-DREAM-004 | Dream trigger threshold MUST match constitution (80% or 30%) | ISS-014 | dream.trigger.gpu | MEDIUM |

### Performance Domain Requirements

| ID | Requirement | Source Issue | Constitution Rule | Severity |
|----|-------------|--------------|-------------------|----------|
| REQ-PERF-001 | No block_on() calls in async context | ISS-004 | AP-08 | CRITICAL |
| REQ-PERF-002 | WorkspaceProviderImpl methods MUST be async or use blocking_read() | ISS-004 | rust_standards.async_patterns | CRITICAL |
| REQ-PERF-003 | All 8 block_on() instances in gwt_providers.rs MUST be removed | ISS-004 | - | CRITICAL |
| REQ-PERF-004 | RwLock in wake_controller should use parking_lot or tokio::sync | ISS-015 | - | LOW |
| REQ-PERF-005 | HashMap allocations MUST use with_capacity() where size known | ISS-016 | - | LOW |

### Architecture Domain Requirements

| ID | Requirement | Source Issue | Constitution Rule | Severity |
|----|-------------|--------------|-------------------|----------|
| REQ-ARCH-001 | ALL CUDA FFI declarations MUST be in context-graph-cuda crate | ISS-005 | rules: "CUDA FFI only in context-graph-cuda" | CRITICAL |
| REQ-ARCH-002 | No extern "C" blocks in context-graph-embeddings | ISS-005 | - | CRITICAL |
| REQ-ARCH-003 | No extern "C" blocks in context-graph-graph | ISS-005 | - | CRITICAL |
| REQ-ARCH-004 | CI MUST fail if extern "C" found in non-cuda crates | ISS-005 | testing.gates.pre-merge | HIGH |
| REQ-ARCH-005 | Safe Rust wrappers MUST be in context-graph-cuda/src/safe/ | ISS-005 | - | HIGH |

### MCP Domain Requirements

| ID | Requirement | Source Issue | Constitution Rule | Severity |
|----|-------------|--------------|-------------------|----------|
| REQ-MCP-001 | All PRD Section 5 tools MUST be implemented | ISS-006 | mcp.core_tools | HIGH |
| REQ-MCP-002 | Curation tools (merge_concepts, forget_concept, etc.) MUST exist | ISS-006 | - | HIGH |
| REQ-MCP-003 | Navigation tools MUST exist | ISS-006 | - | HIGH |
| REQ-MCP-004 | Meta-cognitive tools MUST exist | ISS-006 | - | HIGH |
| REQ-MCP-005 | epistemic_action tool MUST be P0 priority | ISS-006 | - | CRITICAL |
| REQ-MCP-006 | Parameter validation MUST match PRD Section 26 | ISS-012 | - | MEDIUM |
| REQ-MCP-007 | inject_context.query MUST have minLength:1, maxLength:4096 | ISS-012 | - | MEDIUM |
| REQ-MCP-008 | store_memory.rationale MUST be marked as required | ISS-012 | - | MEDIUM |
| REQ-MCP-009 | SSE transport MUST be implemented | ISS-013 | mcp.transport | MEDIUM |

### Embeddings Domain Requirements

| ID | Requirement | Source Issue | Constitution Rule | Severity |
|----|-------------|--------------|-------------------|----------|
| REQ-EMBED-001 | Green Contexts MUST be enabled by default on compatible GPUs | ISS-008 | stack.gpu | HIGH |
| REQ-EMBED-002 | GPU architecture MUST be detected at runtime | ISS-008 | - | HIGH |
| REQ-EMBED-003 | TokenPruning for E12 MUST be implemented | ISS-009 | embeddings.models.E12 | HIGH |
| REQ-EMBED-004 | TokenPruning MUST achieve ~50% compression | ISS-009 | - | HIGH |

### UTL Domain Requirements

| ID | Requirement | Source Issue | Constitution Rule | Severity |
|----|-------------|--------------|-------------------|----------|
| REQ-UTL-001 | Johari action naming MUST match PRD or deviation documented | ISS-011 | utl.johari | MEDIUM |
| REQ-UTL-002 | Blind quadrant action MUST be TriggerDream (per PRD) | ISS-011 | - | MEDIUM |
| REQ-UTL-003 | Unknown quadrant action MUST be EpistemicAction (per PRD) | ISS-011 | - | MEDIUM |

---

## Non-Functional Requirements

| ID | Category | Requirement | Metric | Source |
|----|----------|-------------|--------|--------|
| NFR-001 | Performance | Kuramoto step interval | 10ms | GWT-006 |
| NFR-002 | Performance | Dream wake latency | <100ms | dream.constraints.wake |
| NFR-003 | Performance | No async blocking | 0 block_on() calls | AP-08 |
| NFR-004 | Reliability | No silent failures for IC<0.5 | 100% trigger rate | AP-26 |
| NFR-005 | Security | CUDA FFI isolation | Single crate | rules |
| NFR-006 | Compatibility | Green Contexts on RTX 5090 | compute_cap >= 12.0 | stack.gpu |

---

## Edge Cases

| Related Req | Scenario | Expected Behavior |
|-------------|----------|-------------------|
| REQ-GWT-001 | Kuramoto network initialized with wrong count | Fatal error, system refuses to start |
| REQ-IDENTITY-001 | IC drops to 0.49 during active query | Dream triggered immediately, query continues |
| REQ-IDENTITY-001 | IC exactly 0.5 | No trigger (threshold is <0.5, not <=0.5) |
| REQ-PERF-001 | Async context calls sync method | Method uses blocking_read() safely |
| REQ-ARCH-001 | Developer adds extern "C" to embeddings | CI fails with explicit error message |
| REQ-MCP-009 | Client connects via SSE | Connection established, events streamed |
| REQ-EMBED-001 | GPU is RTX 4090 (compute 8.9) | Green Contexts disabled, warning logged |
| REQ-DREAM-003 | GPU at exactly 30% | Dream NOT triggered (threshold is <30%) |

---

## Open Questions (Resolved by Constitution)

| # | Question | Resolution |
|---|----------|------------|
| 1 | GPU trigger threshold: 30% or 80%? | Constitution says dream.trigger.gpu: "<80%", dream.constraints.gpu: "<30%". CLARIFICATION NEEDED: 80% triggers eligibility, 30% is max usage during dream |
| 2 | Should IC=0.5 exactly trigger dream? | Constitution says "<0.5", so IC=0.5 does NOT trigger |
| 3 | Is Johari naming swap intentional? | No documentation of intentional deviation, treat as bug |

---

## Domain Dependency Graph

```
Architecture (ISS-005)
    └── No downstream dependencies (can be done first)

GWT (ISS-001, ISS-003)
    └── Depends on: Nothing
    └── Blocks: Identity Domain

Identity (ISS-002, ISS-010)
    └── Depends on: GWT Domain (Kuramoto for IC calculation)
    └── Blocks: Dream Domain (trigger wiring)

Dream (ISS-007, ISS-014)
    └── Depends on: Identity Domain (trigger reasons)
    └── Blocks: Nothing

Performance (ISS-004, ISS-015, ISS-016)
    └── Depends on: Nothing (parallel with Architecture)
    └── Blocks: Nothing

MCP (ISS-006, ISS-012, ISS-013)
    └── Depends on: GWT, Identity, Dream (tools expose these)
    └── Blocks: Nothing

Embeddings (ISS-008, ISS-009)
    └── Depends on: Architecture (CUDA FFI)
    └── Blocks: Nothing

UTL (ISS-011)
    └── Depends on: Nothing
    └── Blocks: Nothing
```

---

## Recommended Phase Order

### Phase 1: Foundation (No Dependencies)
1. **Architecture Domain** - CUDA FFI consolidation (ISS-005)
2. **Performance Domain** - Remove block_on() (ISS-004, ISS-015, ISS-016)
3. **UTL Domain** - Fix Johari naming (ISS-011)

### Phase 2: Core Systems (Depends on Phase 1)
4. **GWT Domain** - Fix Kuramoto oscillators (ISS-001, ISS-003)
5. **Embeddings Domain** - Green Contexts + TokenPruning (ISS-008, ISS-009)

### Phase 3: Integration (Depends on Phase 2)
6. **Identity Domain** - IC dream trigger (ISS-002, ISS-010)
7. **Dream Domain** - GpuMonitor + threshold (ISS-007, ISS-014)

### Phase 4: Surface (Depends on Phase 3)
8. **MCP Domain** - Missing tools + validation + SSE (ISS-006, ISS-012, ISS-013)

---

## Verification Checklist

After all remediation:

- [ ] `cargo test --all` passes
- [ ] No `extern "C"` in non-cuda crates
- [ ] Kuramoto network has exactly 13 oscillators
- [ ] IC < 0.5 triggers dream consolidation (integration test)
- [ ] KuramotoStepper runs at 10ms intervals
- [ ] Zero `block_on()` calls in async code paths
- [ ] All PRD Section 5 MCP tools implemented
- [ ] GpuMonitor returns real GPU utilization
- [ ] Green Contexts enabled on RTX 5090
- [ ] E12 TokenPruning achieves 50% compression
- [ ] Parameter validation matches PRD Section 26
- [ ] SSE transport functional
- [ ] Johari actions match PRD exactly

---

**Next Step**: Generate Functional Specifications per domain
