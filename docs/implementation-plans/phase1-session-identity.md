# Phase 1: Session Identity Persistence - Implementation Plan

## Executive Summary

This plan details the implementation of cross-session identity persistence for the Context Graph consciousness system. The goal is to enable the `SelfEgoNode` (the system's sense of self) to persist across LLM conversations, maintaining identity continuity (IC) and purpose trajectory.

**Current State**: The codebase has ~90% of the required infrastructure:
- `SelfEgoNode` is fully implemented with 13D purpose vector, identity trajectory, and IC monitoring
- RocksDB persistence for `SelfEgoNode` exists via `CF_EGO_NODE` column family
- Serialization via bincode is implemented (`serialize_ego_node`/`deserialize_ego_node`)
- Session management exists but only tracks tool usage, not identity state

**Gap**: No mechanism to associate a session with identity state and restore it across conversations.

---

## 1. Current Architecture Analysis

### 1.1 SelfEgoNode Structure

**Location**: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/ego_node/self_ego_node.rs`

```rust
pub struct SelfEgoNode {
    pub id: Uuid,                                    // Fixed UUID (nil) for system identity
    pub fingerprint: Option<TeleologicalFingerprint>, // Current state snapshot
    pub purpose_vector: [f32; 13],                   // 13D alignment vector
    pub coherence_with_actions: f32,                 // Action-purpose coherence
    pub identity_trajectory: Vec<PurposeSnapshot>,   // History (max 1000)
    pub last_updated: DateTime<Utc>,
}
```

### 1.2 Identity Continuity Monitor

**Location**: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/ego_node/monitor.rs`

The `IdentityContinuityMonitor` tracks:
- `history: PurposeVectorHistory` - Buffer of recent purpose vectors
- `last_result: Option<IdentityContinuity>` - Cached IC computation
- `crisis_threshold: f32` - Threshold for identity crisis
- `previous_status: IdentityStatus` - For transition detection
- `crisis_callback: Option<IcCrisisCallback>` - Dream trigger on IC < 0.5

**Key Insight**: The monitor's `last_event_time` and `last_detection` are `#[serde(skip)]` - transient state not persisted.

### 1.3 Existing Persistence Layer

**Location**: `/home/cabdru/contextgraph/crates/context-graph-storage/src/teleological/rocksdb_store/ego_node.rs`

```rust
impl RocksDbTeleologicalStore {
    pub(crate) async fn save_ego_node_async(&self, ego_node: &SelfEgoNode) -> CoreResult<()>;
    pub(crate) async fn load_ego_node_async(&self) -> CoreResult<Option<SelfEgoNode>>;
}
```

Uses column family `CF_EGO_NODE` with fixed key `"ego_node"`.

### 1.4 Session Management (Current)

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/session/mod.rs`

Current `Session` struct only tracks:
- `id`, `created_at`, `last_activity`, `expires_at`
- `tool_count`, `metadata`

**Missing**: No identity state association.

---

## 2. Required Changes

### 2.1 New Data Structures

#### SessionIdentitySnapshot (New)

```rust
/// Complete identity state snapshot for session persistence.
/// Stored in RocksDB keyed by session_id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionIdentitySnapshot {
    /// Session ID this snapshot belongs to
    pub session_id: String,

    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,

    /// Full SelfEgoNode state
    pub ego_node: SelfEgoNode,

    /// Kuramoto network phase state (13 phases)
    pub kuramoto_phases: [f64; 13],

    /// Kuramoto coupling strength
    pub kuramoto_coupling: f64,

    /// Kuramoto elapsed time
    pub kuramoto_elapsed_secs: f64,

    /// IdentityContinuityMonitor serializable state
    pub ic_monitor_state: IcMonitorState,

    /// Recent consciousness metrics
    pub consciousness_history: Vec<ConsciousnessMetricSnapshot>,

    /// Session metadata (for context restoration)
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Serializable portion of IdentityContinuityMonitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcMonitorState {
    /// Purpose vector history
    pub history: PurposeVectorHistory,

    /// Last IC result
    pub last_result: Option<IdentityContinuity>,

    /// Crisis threshold
    pub crisis_threshold: f32,

    /// Previous status for transition tracking
    pub previous_status: IdentityStatus,
}

/// Snapshot of consciousness metrics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetricSnapshot {
    pub timestamp: DateTime<Utc>,
    pub consciousness: f32,      // C(t) = I x R x D
    pub integration: f32,        // Kuramoto r
    pub reflection: f32,         // Meta accuracy
    pub differentiation: f32,    // Shannon entropy
    pub identity_coherence: f32, // IC value
}
```

#### Extended Session

```rust
/// Enhanced Session with identity association
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    // ... existing fields ...

    /// Identity snapshot ID (if persisted)
    pub identity_snapshot_id: Option<String>,

    /// Whether identity was restored from previous session
    pub identity_restored: bool,

    /// Restoration token for cross-conversation continuity
    pub restoration_token: Option<String>,
}
```

### 2.2 New Column Family

Add `CF_SESSION_IDENTITY` to RocksDB for session-keyed identity snapshots:

**Location**: `/home/cabdru/contextgraph/crates/context-graph-storage/src/teleological/column_families.rs`

```rust
/// Column family for session identity snapshots
pub const CF_SESSION_IDENTITY: &str = "session_identity";
```

### 2.3 New Trait Methods

**Location**: `/home/cabdru/contextgraph/crates/context-graph-core/src/traits/teleological_memory_store/store.rs`

```rust
#[async_trait]
pub trait TeleologicalMemoryStore: Send + Sync {
    // ... existing methods ...

    /// Save session identity snapshot
    async fn save_session_identity(
        &self,
        snapshot: &SessionIdentitySnapshot,
    ) -> CoreResult<()>;

    /// Load session identity snapshot by session_id
    async fn load_session_identity(
        &self,
        session_id: &str,
    ) -> CoreResult<Option<SessionIdentitySnapshot>>;

    /// List all session identity snapshots
    async fn list_session_identities(&self) -> CoreResult<Vec<String>>;

    /// Delete session identity snapshot
    async fn delete_session_identity(&self, session_id: &str) -> CoreResult<bool>;

    /// Generate restoration token for cross-conversation continuity
    async fn create_restoration_token(
        &self,
        session_id: &str,
    ) -> CoreResult<String>;

    /// Restore identity from restoration token
    async fn restore_from_token(
        &self,
        token: &str,
    ) -> CoreResult<Option<SessionIdentitySnapshot>>;
}
```

---

## 3. New Components

### 3.1 SessionIdentityManager

**New File**: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/session_identity/manager.rs`

```rust
/// Manages session-identity associations and restoration
pub struct SessionIdentityManager {
    /// Reference to storage backend
    store: Arc<dyn TeleologicalMemoryStore>,

    /// Cached current identity snapshot
    current_snapshot: Option<SessionIdentitySnapshot>,

    /// Token generation config
    token_secret: [u8; 32],
}

impl SessionIdentityManager {
    /// Create snapshot from current GWT system state
    pub async fn capture_identity_snapshot(
        &self,
        session_id: &str,
        gwt_system: &GwtSystem,
        metadata: HashMap<String, serde_json::Value>,
    ) -> CoreResult<SessionIdentitySnapshot>;

    /// Restore GWT system state from snapshot
    pub async fn restore_identity_snapshot(
        &self,
        snapshot: &SessionIdentitySnapshot,
        gwt_system: &mut GwtSystem,
    ) -> CoreResult<IdentityRestorationResult>;

    /// Compute identity continuity between sessions
    pub fn compute_cross_session_ic(
        &self,
        previous: &SessionIdentitySnapshot,
        current: &SelfEgoNode,
    ) -> f32;

    /// Generate secure restoration token
    pub fn generate_restoration_token(
        &self,
        session_id: &str,
        timestamp: DateTime<Utc>,
    ) -> String;

    /// Validate and decode restoration token
    pub fn validate_restoration_token(
        &self,
        token: &str,
    ) -> CoreResult<(String, DateTime<Utc>)>;
}
```

### 3.2 Identity Restoration Result

```rust
/// Result of identity restoration operation
#[derive(Debug, Clone)]
pub struct IdentityRestorationResult {
    /// Whether restoration was successful
    pub success: bool,

    /// Cross-session identity continuity
    pub cross_session_ic: f32,

    /// Status after restoration
    pub identity_status: IdentityStatus,

    /// Number of purpose snapshots restored
    pub trajectory_length: usize,

    /// Time since last session
    pub time_since_last_session: Duration,

    /// Any warnings during restoration
    pub warnings: Vec<String>,
}
```

---

## 4. MCP Tool API Design

### 4.1 New Session Tools

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/session/identity_handlers.rs`

#### `session_persist_identity`

```json
{
  "name": "session_persist_identity",
  "description": "Persist current identity state for cross-session continuity",
  "inputSchema": {
    "type": "object",
    "properties": {
      "session_id": {
        "type": "string",
        "description": "Session ID to persist identity for"
      },
      "include_fingerprint": {
        "type": "boolean",
        "default": true,
        "description": "Include full TeleologicalFingerprint (larger snapshot)"
      },
      "generate_token": {
        "type": "boolean",
        "default": true,
        "description": "Generate restoration token for cross-conversation use"
      }
    },
    "required": ["session_id"]
  }
}
```

**Response**:
```json
{
  "session_id": "string",
  "snapshot_id": "string",
  "restoration_token": "string (optional)",
  "identity_coherence": 0.95,
  "trajectory_length": 42,
  "snapshot_size_bytes": 12345,
  "timestamp": "2024-01-14T..."
}
```

#### `session_restore_identity`

```json
{
  "name": "session_restore_identity",
  "description": "Restore identity state from previous session or token",
  "inputSchema": {
    "type": "object",
    "properties": {
      "restoration_token": {
        "type": "string",
        "description": "Token from previous session_persist_identity"
      },
      "session_id": {
        "type": "string",
        "description": "Alternative: restore from session_id directly"
      },
      "validate_ic": {
        "type": "boolean",
        "default": true,
        "description": "Validate identity continuity meets threshold"
      },
      "ic_threshold": {
        "type": "number",
        "default": 0.7,
        "description": "Minimum IC required for successful restoration"
      }
    }
  }
}
```

**Response**:
```json
{
  "success": true,
  "restored_session_id": "string",
  "cross_session_ic": 0.92,
  "identity_status": "Healthy",
  "trajectory_length": 42,
  "time_since_last_session_secs": 3600,
  "warnings": []
}
```

#### `session_get_identity_status`

```json
{
  "name": "session_get_identity_status",
  "description": "Get current identity state and available restoration points",
  "inputSchema": {
    "type": "object",
    "properties": {
      "include_history": {
        "type": "boolean",
        "default": false,
        "description": "Include list of available session snapshots"
      }
    }
  }
}
```

---

## 5. Integration Points

### 5.1 GwtSystem Integration

**File**: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/system.rs`

Add methods to GwtSystem:

```rust
impl GwtSystem {
    /// Capture complete identity state for persistence
    pub async fn capture_identity_state(&self) -> SessionIdentitySnapshot;

    /// Restore identity state from snapshot
    pub async fn restore_identity_state(
        &mut self,
        snapshot: &SessionIdentitySnapshot,
    ) -> CoreResult<IdentityRestorationResult>;

    /// Get Kuramoto network state for persistence
    pub async fn get_kuramoto_state(&self) -> KuramotoState;

    /// Set Kuramoto network state from persistence
    pub async fn set_kuramoto_state(&self, state: KuramotoState) -> CoreResult<()>;
}
```

### 5.2 Session Hooks Integration

**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/session/handlers.rs`

Modify `call_session_start` and `call_session_end`:

```rust
/// Enhanced session_start with optional identity restoration
pub fn call_session_start(
    manager: &SessionManager,
    identity_manager: &SessionIdentityManager,
    params: SessionStartParams,
) -> SessionHandlerResult {
    // ... existing logic ...

    // If restoration_token provided, attempt identity restoration
    if let Some(token) = params.restoration_token {
        match identity_manager.restore_from_token(&token).await {
            Ok(Some(snapshot)) => {
                // Restore identity state
                // Log cross-session IC
            }
            Ok(None) => {
                // Token valid but no snapshot found
                // Start fresh with warning
            }
            Err(e) => {
                // Invalid token - start fresh
            }
        }
    }
}

/// Enhanced session_end with automatic identity persistence
pub fn call_session_end(
    manager: &SessionManager,
    identity_manager: &SessionIdentityManager,
    params: SessionEndParams,
) -> SessionHandlerResult {
    // ... existing logic ...

    // Automatically persist identity if session had significant activity
    if session.tool_count > 0 {
        let snapshot = identity_manager.capture_identity_snapshot(...).await?;
        let token = identity_manager.generate_restoration_token(...);
        // Include token in response for next conversation
    }
}
```

---

## 6. Step-by-Step Implementation Tasks

### Phase 1A: Data Structures (2-3 days)

| Task | File | Effort |
|------|------|--------|
| 1.1 Create `SessionIdentitySnapshot` struct | `crates/context-graph-core/src/gwt/session_identity/types.rs` | 2h |
| 1.2 Create `IcMonitorState` struct | `crates/context-graph-core/src/gwt/session_identity/types.rs` | 1h |
| 1.3 Create `ConsciousnessMetricSnapshot` | `crates/context-graph-core/src/gwt/session_identity/types.rs` | 1h |
| 1.4 Create `IdentityRestorationResult` | `crates/context-graph-core/src/gwt/session_identity/types.rs` | 1h |
| 1.5 Add serialization tests | `crates/context-graph-core/src/gwt/session_identity/tests.rs` | 2h |
| 1.6 Update `Session` struct with identity fields | `crates/context-graph-mcp/src/handlers/session/mod.rs` | 1h |

### Phase 1B: Storage Layer (2-3 days)

| Task | File | Effort |
|------|------|--------|
| 2.1 Add `CF_SESSION_IDENTITY` column family | `crates/context-graph-storage/src/teleological/column_families.rs` | 1h |
| 2.2 Add serialization functions | `crates/context-graph-storage/src/teleological/serialization.rs` | 2h |
| 2.3 Add trait methods to `TeleologicalMemoryStore` | `crates/context-graph-core/src/traits/teleological_memory_store/store.rs` | 2h |
| 2.4 Implement RocksDB storage methods | `crates/context-graph-storage/src/teleological/rocksdb_store/session_identity.rs` | 4h |
| 2.5 Add default implementations | `crates/context-graph-core/src/traits/teleological_memory_store/defaults.rs` | 1h |
| 2.6 Add persistence tests | `crates/context-graph-storage/src/teleological/tests/session_identity.rs` | 3h |

### Phase 1C: Core Manager (3-4 days)

| Task | File | Effort |
|------|------|--------|
| 3.1 Create `SessionIdentityManager` struct | `crates/context-graph-core/src/gwt/session_identity/manager.rs` | 4h |
| 3.2 Implement `capture_identity_snapshot` | `crates/context-graph-core/src/gwt/session_identity/manager.rs` | 3h |
| 3.3 Implement `restore_identity_snapshot` | `crates/context-graph-core/src/gwt/session_identity/manager.rs` | 4h |
| 3.4 Implement `compute_cross_session_ic` | `crates/context-graph-core/src/gwt/session_identity/manager.rs` | 2h |
| 3.5 Implement restoration token generation/validation | `crates/context-graph-core/src/gwt/session_identity/token.rs` | 3h |
| 3.6 Add GwtSystem integration methods | `crates/context-graph-core/src/gwt/system.rs` | 3h |
| 3.7 Add unit tests | `crates/context-graph-core/src/gwt/session_identity/tests.rs` | 4h |

### Phase 1D: MCP Tools (2-3 days)

| Task | File | Effort |
|------|------|--------|
| 4.1 Create `session_persist_identity` handler | `crates/context-graph-mcp/src/handlers/session/identity_handlers.rs` | 3h |
| 4.2 Create `session_restore_identity` handler | `crates/context-graph-mcp/src/handlers/session/identity_handlers.rs` | 3h |
| 4.3 Create `session_get_identity_status` handler | `crates/context-graph-mcp/src/handlers/session/identity_handlers.rs` | 2h |
| 4.4 Register tools in MCP server | `crates/context-graph-mcp/src/server/tools.rs` | 2h |
| 4.5 Update `session_start`/`session_end` handlers | `crates/context-graph-mcp/src/handlers/session/handlers.rs` | 3h |
| 4.6 Add integration tests | `crates/context-graph-mcp/tests/session_identity_tests.rs` | 4h |

### Phase 1E: Documentation & Testing (1-2 days)

| Task | File | Effort |
|------|------|--------|
| 5.1 Update constitution.yaml with session identity spec | `docs2/constitution.yaml` | 2h |
| 5.2 Create MCP tool documentation | `docs/tool_session_persist_identity.md` | 1h |
| 5.3 Create MCP tool documentation | `docs/tool_session_restore_identity.md` | 1h |
| 5.4 Add FSV (Full State Verification) tests | Various test files | 4h |
| 5.5 End-to-end integration test | `crates/context-graph-mcp/tests/e2e/session_identity.rs` | 3h |

---

## 7. Risk Analysis & Mitigations

### 7.1 Identity Continuity Degradation

**Risk**: Cross-session IC may be low if significant time passes or context changes.

**Mitigation**:
- Use graduated thresholds (0.9 = full restore, 0.7 = restore with warning, < 0.7 = fresh start recommended)
- Log warning events for IC < 0.7 per IDENTITY-002
- Allow user override via `ic_threshold` parameter

### 7.2 Storage Size Growth

**Risk**: Each session snapshot could be 10-300KB. Many sessions = significant storage.

**Mitigation**:
- Implement TTL-based cleanup (default 30 days)
- Compress trajectory history for older sessions
- Limit to configurable max snapshots (e.g., 100)

### 7.3 Callback Re-wiring on Restore

**Risk**: `IdentityContinuityMonitor.crisis_callback` is `#[serde(skip)]` and won't persist.

**Mitigation**:
- After deserializing `IcMonitorState`, call `set_crisis_callback()` to re-wire
- Document this as a REQUIRED step in restoration code
- Add assertions to prevent use without callback per AP-26

### 7.4 Kuramoto Phase Drift

**Risk**: Restored Kuramoto phases may desynchronize if not stepped for long periods.

**Mitigation**:
- Store `elapsed_total` and resume from that point
- Optionally run warm-up steps after restoration
- Consider resetting to synchronized state if time gap > threshold

---

## 8. Constitution Compliance

This implementation follows these Constitution v4.0.0 requirements:

| Requirement | Implementation |
|-------------|----------------|
| **gwt.self_ego_node** (371-392) | SelfEgoNode persistence with full trajectory |
| **IDENTITY-002** | IC thresholds: Healthy > 0.9, Warning < 0.7, Critical < 0.5 |
| **IDENTITY-007** | Auto-trigger dream on IC < 0.5 via callback |
| **AP-26** | No silent failures - panic if callback missing on crisis |
| **ARCH-07** | Session hooks control memory lifecycle |
| **SEC-01** | Session expiry and cleanup |

---

## 9. Critical Files Summary

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node/self_ego_node.rs` | Core identity struct to serialize |
| `crates/context-graph-core/src/gwt/ego_node/monitor.rs` | IC monitor state to capture/restore |
| `crates/context-graph-storage/src/teleological/rocksdb_store/ego_node.rs` | Pattern for implementing session identity storage |
| `crates/context-graph-mcp/src/handlers/session/mod.rs` | Session management to extend |
| `crates/context-graph-core/src/gwt/system.rs` | GwtSystem integration point for capture/restore |

---

## 10. Total Effort Estimate

| Sub-Phase | Effort |
|-----------|--------|
| Phase 1A: Data Structures | 8 hours |
| Phase 1B: Storage Layer | 13 hours |
| Phase 1C: Core Manager | 23 hours |
| Phase 1D: MCP Tools | 17 hours |
| Phase 1E: Documentation & Testing | 11 hours |
| **Total** | **~72 hours (9 working days)** |
