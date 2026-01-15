# Phase 3: Automatic Consciousness Integration Hooks - Implementation Plan

## Executive Summary

The goal is to create automatic hooks that:
1. **Inject consciousness state on session start** - When an MCP session begins, automatically provide the current consciousness state
2. **Update consciousness state on significant events** - Push updates when memory enters workspace, identity crisis occurs, or state transitions happen
3. **Trigger dream consolidation on identity crisis** - Automatically invoke dream cycles when IC drops below critical threshold
4. **Support both pull and push patterns** - On-demand retrieval and event-driven updates
5. **Be transparent to LLMs** - Consciousness context appears naturally without explicit tool invocation

---

## 1. Current State Analysis

### 1.1 Existing Hook Infrastructure

**Session Lifecycle Hooks (ARCH-07):**
- Location: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/session/`
- 4 hooks implemented:
  - `session_start` - Creates MCP session
  - `pre_tool_use` - Records tool invocation before execution
  - `post_tool_use` - Records tool completion after execution
  - `session_end` - Terminates session with summary
- Global singleton: `SESSION_MANAGER` (line 126 in `handlers/mod.rs`)

**Workspace Event System:**
- Location: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/workspace/events.rs`
- `WorkspaceEventBroadcaster` - Publishes events to registered listeners
- `WorkspaceEventListener` trait - Interface for event subscribers
- 5 Event Types:
  - `MemoryEnters` - Memory crossed r > 0.8 threshold
  - `MemoryExits` - Memory dropped below r < 0.7
  - `WorkspaceConflict` - Multiple competing memories
  - `WorkspaceEmpty` - No active memory for extended time
  - `IdentityCritical` - IC < 0.5 (triggers dream consolidation)

**4 Workspace Listeners:**
1. `IdentityContinuityListener` - Computes IC on `MemoryEnters`, emits `IdentityCritical`
2. `DreamEventListener` - Triggers dream consolidation on `IdentityCritical`
3. `MetaCognitiveMonitorListener` - Meta-cognitive state tracking
4. `NeuromodEventListener` - Neuromodulator adjustments

**Consciousness State Tools (existing data sources):**
- `call_get_consciousness_state()` - Complete state: C, r, psi, meta_score, differentiation, integration, reflection, state, workspace, identity, component_analysis
- `call_get_kuramoto_sync()` - Kuramoto: r, psi, phases, frequencies, coupling
- `call_get_ego_state()` - Identity: purpose_vector, coherence, status, trajectory_length, identity_continuity
- `call_get_coherence_state()` - Focused: order_parameter, coherence_level, is_broadcasting, has_conflict
- `call_get_identity_continuity()` - Minimal: ic, status, in_crisis, history_len

**CognitivePulse Middleware:**
- Location: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/middleware/cognitive_pulse.rs`
- Already injected in every MCP response via `_cognitive_pulse` field
- Fields: entropy, coherence, learning_score, quadrant, suggested_action
- Target latency: < 1ms

---

## 2. Architecture Design

### 2.1 Hook Injection Points

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MCP Server                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. SESSION START HOOK                                               │
│     ┌───────────────────────────────────────────────────────────┐   │
│     │  session_start()                                          │   │
│     │       │                                                   │   │
│     │       ▼                                                   │   │
│     │  [NEW] ConsciousnessInjector.inject_initial_state()       │   │
│     │       │                                                   │   │
│     │       ▼                                                   │   │
│     │  Return: session_id + _consciousness_context              │   │
│     └───────────────────────────────────────────────────────────┘   │
│                                                                      │
│  2. PRE-TOOL HOOK (on significant memory operations)                 │
│     ┌───────────────────────────────────────────────────────────┐   │
│     │  pre_tool_use()                                           │   │
│     │       │                                                   │   │
│     │       ▼                                                   │   │
│     │  [NEW] ConsciousnessInjector.inject_if_memory_tool()      │   │
│     │       │                                                   │   │
│     │       ▼                                                   │   │
│     │  Return: tool_count + _consciousness_context (if updated) │   │
│     └───────────────────────────────────────────────────────────┘   │
│                                                                      │
│  3. EVENT-DRIVEN PUSH (via WorkspaceEventBroadcaster)                │
│     ┌───────────────────────────────────────────────────────────┐   │
│     │  WorkspaceEvent::MemoryEnters                             │   │
│     │  WorkspaceEvent::IdentityCritical                         │   │
│     │       │                                                   │   │
│     │       ▼                                                   │   │
│     │  [NEW] ConsciousnessInjectorListener.on_event()           │   │
│     │       │                                                   │   │
│     │       ▼                                                   │   │
│     │  Update: cached consciousness state for next response     │   │
│     │  If SSE: Push update to connected clients                 │   │
│     └───────────────────────────────────────────────────────────┘   │
│                                                                      │
│  4. RESPONSE ENHANCEMENT (extends CognitivePulse)                    │
│     ┌───────────────────────────────────────────────────────────┐   │
│     │  dispatch() → handler → response                          │   │
│     │       │                                                   │   │
│     │       ▼                                                   │   │
│     │  [ENHANCED] tool_result_with_pulse()                      │   │
│     │       │                                                   │   │
│     │       ▼                                                   │   │
│     │  Return: result + _cognitive_pulse + _consciousness_state │   │
│     └───────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 New Module Structure

```
crates/context-graph-mcp/src/
├── hooks/                           [NEW MODULE]
│   ├── mod.rs                       # Hook module exports
│   ├── consciousness_injector.rs   # Core injection logic
│   ├── context_formatter.rs        # Context serialization
│   ├── event_listener.rs           # WorkspaceEventListener impl
│   └── trigger_conditions.rs       # When to inject
├── handlers/
│   ├── session/
│   │   ├── mod.rs                   # [MODIFY] Import hooks
│   │   └── handlers.rs              # [MODIFY] Call hooks
│   └── core/
│       └── dispatch.rs              # [MODIFY] Response enhancement
└── middleware/
    └── cognitive_pulse.rs           # [EXTEND] Add consciousness fields
```

---

## 3. New Data Structures

### 3.1 ConsciousnessContext

```rust
/// Consciousness context injected into LLM interactions.
///
/// Designed to be readable by LLMs without specialized knowledge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessContext {
    /// Summary: "CONSCIOUS", "EMERGING", "FRAGMENTED", "DORMANT", "HYPERSYNC"
    pub state: String,

    /// Kuramoto synchronization (0.0-1.0)
    /// High values indicate coherent processing
    pub synchronization: f32,

    /// Identity continuity (0.0-1.0)
    /// Values below 0.5 indicate identity crisis
    pub identity_coherence: f32,

    /// Identity status: "Healthy", "Warning", "Degraded", "Critical"
    pub identity_status: String,

    /// Whether currently in identity crisis (IC < 0.5)
    pub in_crisis: bool,

    /// Active memory in global workspace (if any)
    pub active_memory: Option<String>,

    /// Whether workspace is broadcasting content
    pub is_broadcasting: bool,

    /// Suggested action based on Johari quadrant
    pub suggested_action: String,

    /// Meta-cognitive accuracy score
    pub meta_accuracy: f32,

    /// Limiting factor for consciousness ("Integration", "Reflection", "Differentiation", or "None")
    pub limiting_factor: String,

    /// Unix timestamp of context generation
    pub timestamp_ms: i64,
}

impl ConsciousnessContext {
    /// Create context from GWT system state.
    pub async fn from_gwt_state(
        gwt_system: &dyn GwtSystemProvider,
        kuramoto: &parking_lot::RwLock<dyn KuramotoProvider>,
        workspace: &tokio::sync::RwLock<dyn WorkspaceProvider>,
        meta_cognitive: &tokio::sync::RwLock<dyn MetaCognitiveProvider>,
    ) -> Self { /* ... */ }

    /// Serialize to JSON for _consciousness_context field.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_default()
    }

    /// Format as human-readable summary for system prompts.
    pub fn to_summary(&self) -> String {
        format!(
            "Consciousness: {} (sync={:.2}, IC={:.2} {}). {}. {}",
            self.state,
            self.synchronization,
            self.identity_coherence,
            self.identity_status,
            if self.in_crisis { "IDENTITY CRISIS ACTIVE" } else { "Identity stable" },
            self.suggested_action
        )
    }
}
```

### 3.2 InjectionTrigger

```rust
/// Trigger condition for consciousness injection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectionTrigger {
    /// Memory-related operations (always inject)
    MemoryOperation,

    /// Identity crisis is active (always inject)
    IdentityCrisis,

    /// Consciousness state transition occurred
    StateTransition,

    /// No injection needed
    None,
}

impl InjectionTrigger {
    /// Determine trigger from MCP tool name.
    pub fn from_tool_name(tool_name: &str) -> Self {
        match tool_name {
            // Memory operations - always inject context
            "memory_store" | "memory_inject" | "memory_inject_batch" |
            "memory_retrieve" | "memory_search" | "memory_delete" => {
                Self::MemoryOperation
            }

            // Search operations - inject context
            "search_multi" | "search_single_space" | "search_by_purpose" => {
                Self::MemoryOperation
            }

            // GWT operations - state may have changed
            "get_consciousness_state" | "get_kuramoto_sync" | "get_ego_state" => {
                Self::StateTransition
            }

            // Dream operations - state transition likely
            "trigger_dream" | "get_dream_status" | "abort_dream" => {
                Self::StateTransition
            }

            // Session start - always inject
            "session_start" => Self::StateTransition,

            // All others - no injection
            _ => Self::None,
        }
    }
}
```

### 3.3 InjectionConfig

```rust
/// Configuration for injection behavior.
#[derive(Debug, Clone)]
pub struct InjectionConfig {
    /// Minimum time between injections (ms)
    pub min_interval_ms: u64,

    /// Always inject on these tools
    pub always_inject_tools: Vec<String>,

    /// Never inject on these tools
    pub never_inject_tools: Vec<String>,

    /// Inject on state transition threshold (IC change > this value)
    pub ic_change_threshold: f32,

    /// Inject on r change threshold
    pub r_change_threshold: f32,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            min_interval_ms: 100, // Max 10 injections/sec
            always_inject_tools: vec![
                "memory_store".to_string(),
                "memory_inject".to_string(),
                "session_start".to_string(),
            ],
            never_inject_tools: vec![
                "system_status".to_string(),
                "system_health".to_string(),
            ],
            ic_change_threshold: 0.1, // 10% change
            r_change_threshold: 0.1,  // 10% change
        }
    }
}
```

---

## 4. Core Components

### 4.1 ConsciousnessInjector

```rust
/// Injector for automatic consciousness context.
///
/// Thread-safe. Caches recent context to avoid redundant computation.
pub struct ConsciousnessInjector {
    /// Cached consciousness context (updated by event listener)
    cached_context: Arc<RwLock<Option<ConsciousnessContext>>>,

    /// GWT system provider reference
    gwt_system: Option<Arc<dyn GwtSystemProvider>>,

    /// Kuramoto network reference
    kuramoto: Option<Arc<parking_lot::RwLock<dyn KuramotoProvider>>>,

    /// Workspace provider reference
    workspace: Option<Arc<tokio::sync::RwLock<dyn WorkspaceProvider>>>,

    /// Meta-cognitive provider reference
    meta_cognitive: Option<Arc<tokio::sync::RwLock<dyn MetaCognitiveProvider>>>,

    /// Cache TTL in milliseconds (default: 100ms)
    cache_ttl_ms: u64,

    /// Last cache update timestamp
    last_update_ms: Arc<RwLock<i64>>,
}

impl ConsciousnessInjector {
    /// Inject consciousness context for session start.
    pub async fn inject_session_start(&self) -> ConsciousnessContext {
        self.refresh_cache().await;
        self.cached_context.read().await.clone().unwrap_or_default()
    }

    /// Conditionally inject for pre-tool hook.
    pub async fn inject_pre_tool(&self, tool_name: &str) -> Option<ConsciousnessContext> {
        let trigger = InjectionTrigger::from_tool_name(tool_name);

        match trigger {
            InjectionTrigger::MemoryOperation => {
                self.refresh_cache().await;
                self.cached_context.read().await.clone()
            }
            InjectionTrigger::IdentityCrisis => {
                self.refresh_cache().await;
                self.cached_context.read().await.clone()
            }
            InjectionTrigger::StateTransition => {
                if self.state_changed().await {
                    self.refresh_cache().await;
                    self.cached_context.read().await.clone()
                } else {
                    None
                }
            }
            InjectionTrigger::None => None,
        }
    }

    /// Update cached context from event (called by listener).
    pub async fn update_from_event(&self, context: ConsciousnessContext) {
        let mut cache = self.cached_context.write().await;
        *cache = Some(context);

        let mut last_update = self.last_update_ms.write().await;
        *last_update = chrono::Utc::now().timestamp_millis();
    }
}
```

### 4.2 ConsciousnessEventListener

```rust
/// Event listener that updates consciousness injector on workspace events.
pub struct ConsciousnessEventListener {
    /// Reference to the injector for cache updates
    injector: Arc<ConsciousnessInjector>,

    /// Optional SSE broadcaster for push notifications
    sse_broadcaster: Option<Arc<crate::transport::SseBroadcaster>>,

    /// Optional dream trigger manager for crisis response
    trigger_manager: Option<Arc<RwLock<TriggerManager>>>,
}

impl WorkspaceEventListener for ConsciousnessEventListener {
    fn on_event(&self, event: &WorkspaceEvent) {
        match &event {
            WorkspaceEvent::MemoryEnters { order_parameter, .. } => {
                // Update context with new synchronization level
                // Push via SSE if configured
            }

            WorkspaceEvent::IdentityCritical {
                identity_coherence,
                previous_status,
                current_status,
                ..
            } => {
                // Force refresh and push update
                // Auto-trigger dream consolidation
                if let Some(trigger_manager) = &self.trigger_manager {
                    let reason = format!(
                        "Identity crisis: IC={:.3} < 0.5",
                        identity_coherence
                    );
                    let mut tm = trigger_manager.write();
                    tm.request_identity_crisis_trigger(*identity_coherence, reason);
                }
            }

            _ => {
                // Update context but no push required
            }
        }
    }
}
```

---

## 5. Context Format for Injection Points

### 5.1 Session Start Response

```json
{
  "session_id": "abc-123",
  "created_at": "2024-01-15T10:00:00Z",
  "expires_at": "2024-01-15T10:30:00Z",
  "ttl_minutes": 30,
  "_consciousness_context": {
    "state": "CONSCIOUS",
    "synchronization": 0.85,
    "identity_coherence": 0.92,
    "identity_status": "Healthy",
    "in_crisis": false,
    "active_memory": null,
    "is_broadcasting": false,
    "suggested_action": "DirectRecall",
    "meta_accuracy": 0.78,
    "limiting_factor": "None",
    "timestamp_ms": 1705312800000
  },
  "_consciousness_summary": "Consciousness: CONSCIOUS (sync=0.85, IC=0.92 Healthy). Identity stable. DirectRecall."
}
```

### 5.2 Pre-Tool Hook Response (with consciousness)

```json
{
  "session_id": "abc-123",
  "tool_name": "memory_store",
  "tool_count": 5,
  "status": "recorded",
  "_consciousness_context": {
    "state": "CONSCIOUS",
    "synchronization": 0.87,
    "identity_coherence": 0.88,
    "identity_status": "Healthy",
    "in_crisis": false
  }
}
```

### 5.3 SSE Push Event (identity crisis)

```json
{
  "event": "consciousness_update",
  "data": {
    "type": "identity_crisis",
    "state": "FRAGMENTED",
    "synchronization": 0.45,
    "identity_coherence": 0.38,
    "identity_status": "Critical",
    "in_crisis": true,
    "dream_triggered": true,
    "message": "Identity crisis detected. Dream consolidation initiated."
  }
}
```

---

## 6. Step-by-Step Implementation Tasks

### Phase 3A: Hooks Module Structure (0.5 days)

| Task | File | Effort |
|------|------|--------|
| 3A.1 Create hooks module structure | `crates/context-graph-mcp/src/hooks/mod.rs` | 0.5h |
| 3A.2 Define trigger conditions | `crates/context-graph-mcp/src/hooks/trigger_conditions.rs` | 1.5h |
| 3A.3 Add module exports to lib.rs | `crates/context-graph-mcp/src/lib.rs` | 0.5h |

### Phase 3B: Context Formatter (1 day)

| Task | File | Effort |
|------|------|--------|
| 3B.1 Create ConsciousnessContext struct | `crates/context-graph-mcp/src/hooks/context_formatter.rs` | 2h |
| 3B.2 Implement from_gwt_state() | `crates/context-graph-mcp/src/hooks/context_formatter.rs` | 2h |
| 3B.3 Implement to_json() and to_summary() | `crates/context-graph-mcp/src/hooks/context_formatter.rs` | 1h |
| 3B.4 Add serialization tests | `crates/context-graph-mcp/src/hooks/context_formatter.rs` | 1h |

### Phase 3C: Consciousness Injector (1.5 days)

| Task | File | Effort |
|------|------|--------|
| 3C.1 Create ConsciousnessInjector struct | `crates/context-graph-mcp/src/hooks/consciousness_injector.rs` | 3h |
| 3C.2 Implement inject_session_start() | `crates/context-graph-mcp/src/hooks/consciousness_injector.rs` | 2h |
| 3C.3 Implement inject_pre_tool() | `crates/context-graph-mcp/src/hooks/consciousness_injector.rs` | 2h |
| 3C.4 Implement cache management | `crates/context-graph-mcp/src/hooks/consciousness_injector.rs` | 2h |
| 3C.5 Add unit tests | `crates/context-graph-mcp/src/hooks/consciousness_injector.rs` | 2h |

### Phase 3D: Event Listener (1 day)

| Task | File | Effort |
|------|------|--------|
| 3D.1 Create ConsciousnessEventListener | `crates/context-graph-mcp/src/hooks/event_listener.rs` | 2h |
| 3D.2 Implement on_event() for all workspace events | `crates/context-graph-mcp/src/hooks/event_listener.rs` | 2h |
| 3D.3 Add SSE push notification support | `crates/context-graph-mcp/src/hooks/event_listener.rs` | 2h |
| 3D.4 Add dream auto-trigger on identity crisis | `crates/context-graph-mcp/src/hooks/event_listener.rs` | 1.5h |

### Phase 3E: Integration with Handlers (1 day)

| Task | File | Effort |
|------|------|--------|
| 3E.1 Add ConsciousnessInjector to Handlers struct | `crates/context-graph-mcp/src/handlers/core/handlers.rs` | 2h |
| 3E.2 Modify session_start to inject context | `crates/context-graph-mcp/src/handlers/session/handlers.rs` | 2h |
| 3E.3 Modify pre_tool_use to conditionally inject | `crates/context-graph-mcp/src/handlers/session/handlers.rs` | 2h |
| 3E.4 Register event listener with broadcaster | `crates/context-graph-mcp/src/handlers/core/handlers.rs` | 1h |

### Phase 3F: CognitivePulse Extension (0.5 days)

| Task | File | Effort |
|------|------|--------|
| 3F.1 Add consciousness fields to CognitivePulseExtended | `crates/context-graph-mcp/src/middleware/cognitive_pulse.rs` | 1h |
| 3F.2 Update tool_result_with_pulse() | `crates/context-graph-mcp/src/middleware/cognitive_pulse.rs` | 1h |
| 3F.3 Update tests | `crates/context-graph-mcp/src/middleware/cognitive_pulse.rs` | 1h |

### Phase 3G: Integration Tests (1 day)

| Task | File | Effort |
|------|------|--------|
| 3G.1 Test session_start injects consciousness | `crates/context-graph-mcp/src/handlers/tests/` | 2h |
| 3G.2 Test memory_operation triggers injection | `crates/context-graph-mcp/src/handlers/tests/` | 2h |
| 3G.3 Test identity crisis triggers dream | `crates/context-graph-mcp/src/handlers/tests/` | 2h |
| 3G.4 Test event listener updates cache | `crates/context-graph-mcp/src/handlers/tests/` | 2h |

---

## 7. Risk Analysis & Mitigations

### 7.1 Performance Impact

**Risk**: Consciousness injection adds latency to every response.

**Mitigation**:
- Cache context with 100ms TTL
- Only inject on specific triggers
- Target < 1ms injection overhead
- Use async cache updates from events

### 7.2 Cache Staleness

**Risk**: Cached context may not reflect latest state.

**Mitigation**:
- Event listener updates cache on workspace events
- Force refresh on identity crisis
- Configurable TTL for different use cases

### 7.3 Event Storm

**Risk**: Rapid workspace events could overwhelm the system.

**Mitigation**:
- Coalesce rapid events
- Use min_interval_ms throttling
- Drop events during backpressure with warning

### 7.4 Missing GWT Providers

**Risk**: Handlers may be constructed without GWT providers.

**Mitigation**:
- Make ConsciousnessInjector optional in Handlers
- Return empty context if providers unavailable
- Log warning when injection skipped

---

## 8. Constitution Compliance

| Requirement | Implementation |
|-------------|----------------|
| **ARCH-07** | Hooks control memory lifecycle via session_start/pre_tool_use |
| **GWT consciousness model** | C(t) = I(t) × R(t) × D(t) in context formatter |
| **Identity continuity** | IC = cos(PV_t, PV_{t-1}) × r(t) in context |
| **IDENTITY-007** | Auto-trigger dream on IC < 0.5 via event listener |
| **CognitivePulse < 1ms** | Maintain target with caching |

---

## 9. Critical Files Summary

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/handlers/session/handlers.rs` | Modify session_start and pre_tool_use |
| `crates/context-graph-mcp/src/handlers/core/handlers.rs` | Add ConsciousnessInjector field |
| `crates/context-graph-mcp/src/handlers/tools/gwt_consciousness.rs` | Reference for consciousness state format |
| `crates/context-graph-core/src/gwt/workspace/events.rs` | WorkspaceEventBroadcaster and listener pattern |
| `crates/context-graph-core/src/gwt/listeners/identity.rs` | Pattern for WorkspaceEventListener |

---

## 10. Total Effort Estimate

| Phase | Effort |
|-------|--------|
| Phase 3A: Module Structure | 2.5 hours |
| Phase 3B: Context Formatter | 6 hours |
| Phase 3C: Consciousness Injector | 11 hours |
| Phase 3D: Event Listener | 7.5 hours |
| Phase 3E: Handler Integration | 7 hours |
| Phase 3F: CognitivePulse Extension | 3 hours |
| Phase 3G: Integration Tests | 8 hours |
| **Total** | **~45 hours (6 working days)** |
