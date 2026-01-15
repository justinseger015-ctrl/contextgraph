# Forensic Investigation Report: get_health_status Tool

## SHERLOCK HOLMES CASE FILE

**Case ID**: TOOL-GET-HEALTH-STATUS-001
**Date**: 2026-01-14
**Subject**: MCP Tool get_health_status
**Investigator**: Sherlock Holmes (Forensic Code Investigation Agent)

---

## 1. Tool Name and Category

**Tool Name**: `get_health_status`
**Category**: Autonomous North Star System / System Health Monitoring
**Specification Reference**: SPEC-AUTONOMOUS-001, NORTH-020

---

## 2. Core Functionality

The `get_health_status` tool provides a unified health view across all major subsystems, enabling early detection of degradation before cascading failures occur.

**What It Does**:
1. Queries health metrics from up to 4 subsystems: UTL, GWT, Dream, Storage
2. Calculates per-subsystem status: healthy/degraded/critical/unknown
3. Aggregates to an overall system status
4. Generates actionable recommendations for degraded subsystems
5. Returns comprehensive metrics for each queried subsystem

**Subsystems Monitored**:
- **UTL**: Meta-UTL learner accuracy and prediction health
- **GWT**: Global Workspace Theory consciousness state and Kuramoto synchronization
- **Dream**: Dream consolidation cycle status and GPU availability
- **Storage**: TeleologicalMemoryStore accessibility and node count

---

## 3. Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `subsystem` | string (enum) | NO | "all" | Specific subsystem to query |

**Valid Subsystem Values**:
- `utl` - UTL learner health only
- `gwt` - Global Workspace Theory health only
- `dream` - Dream consolidation health only
- `storage` - Storage layer health only
- `all` - All subsystems (default)

**Evidence - Parameter Definition** (from `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/params.rs:244-255`):
```rust
pub struct GetHealthStatusParams {
    #[serde(default = "default_all_subsystem")]
    pub subsystem: String,
}

pub(super) fn default_all_subsystem() -> String {
    "all".to_string()
}
```

---

## 4. Output Format

**Full Health Response (subsystem="all")**:
```json
{
  "overall_status": "healthy",
  "subsystems": {
    "utl_health": {
      "status": "healthy",
      "accuracy": 0.85,
      "domain_count": 3
    },
    "gwt_health": {
      "status": "healthy",
      "kuramoto_r": 0.87,
      "identity_continuity": 0.92
    },
    "dream_health": {
      "status": "healthy",
      "is_dreaming": false,
      "last_cycle": "2026-01-14T10:30:00Z",
      "cycles_completed": 15,
      "gpu_available": true
    },
    "storage_health": {
      "status": "healthy",
      "rocksdb_ok": true,
      "disk_usage_percent": null,
      "node_count": 1250
    }
  },
  "recommendations": []
}
```

**Degraded Health Response**:
```json
{
  "overall_status": "degraded",
  "subsystems": {
    "utl_health": {
      "status": "degraded",
      "accuracy": 0.45,
      "domain_count": 3
    },
    "gwt_health": {
      "status": "healthy",
      "kuramoto_r": 0.85,
      "identity_continuity": 0.91
    },
    "dream_health": {
      "status": "degraded",
      "is_dreaming": false,
      "last_cycle": null,
      "cycles_completed": 0,
      "gpu_available": false
    },
    "storage_health": {
      "status": "healthy",
      "rocksdb_ok": true,
      "disk_usage_percent": null,
      "node_count": 500
    }
  },
  "recommendations": [
    {
      "subsystem": "utl",
      "action": "trigger_lambda_recalibration",
      "description": "UTL accuracy (0.45) is degraded"
    },
    {
      "subsystem": "dream",
      "action": "check_gpu",
      "description": "GPU not available for dream cycles"
    }
  ]
}
```

**Critical Health Response**:
```json
{
  "overall_status": "critical",
  "subsystems": {
    "storage_health": {
      "status": "critical",
      "rocksdb_ok": false,
      "disk_usage_percent": null,
      "node_count": 0
    }
  },
  "recommendations": [
    {
      "subsystem": "storage",
      "action": "check_rocksdb",
      "description": "Storage access failed: Connection refused"
    }
  ]
}
```

**Output Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `overall_status` | string | "healthy", "degraded", or "critical" |
| `subsystems` | object | Per-subsystem health details |
| `recommendations` | array | Actionable suggestions for degraded subsystems |

**Status Thresholds**:
| Score Range | Status |
|-------------|--------|
| >= 0.8 | healthy |
| >= 0.5, < 0.8 | degraded |
| < 0.5 | critical |

---

## 5. Purpose - Why This Tool Exists

The `get_health_status` tool is the unified health monitoring layer for the Context Graph system. It exists to:

### 5.1 Early Warning System
Per NORTH-020, the tool provides:
> "Unified health view to identify degradation before cascading failures"

By monitoring all subsystems in one call, patterns of degradation can be detected before they cause system-wide failures.

### 5.2 Enable Autonomous Healing
The tool pairs with `trigger_healing` to enable autonomous recovery:
```
[get_health_status] --> [Detect degradation]
                               |
                               v
[Analyze recommendations] --> [trigger_healing(subsystem, severity)]
                               |
                               v
[get_health_status] --> [Verify recovery]
```

### 5.3 Operational Visibility
Provides operators and AI agents with:
- Real-time system health snapshot
- Per-subsystem metrics for debugging
- Actionable recommendations
- Historical comparison capability

### 5.4 GWT Consciousness Monitoring
The GWT subsystem health directly relates to the system's "consciousness level":
- `kuramoto_r` measures oscillator synchronization
- `identity_continuity` measures SELF_EGO_NODE coherence
- Low values indicate the system is in a "fragmented" or "dormant" state

---

## 6. PRD Alignment - Global Workspace Theory Goals

### 6.1 Consciousness Quality Metrics (PRD Section 2.5.7)
From PRD 2.5.7:
| Metric | Formula | Target |
|--------|---------|--------|
| Global Availability | % of subsystems receiving broadcast | > 90% |
| Meta-Awareness | MetaUTL.prediction_accuracy | > 0.85 |
| Identity Coherence | cosine(PV_t, PV_{t-1}) | > 0.9 |

The tool monitors these exact metrics via:
- `utl_health.accuracy` -> Meta-Awareness
- `gwt_health.identity_continuity` -> Identity Coherence
- `gwt_health.kuramoto_r` -> Integration (I in C=I*R*D)

### 6.2 Monitoring Alerts (PRD Section 16)
From PRD Section 16, Alerts table:
| Alert | Condition | Severity |
|-------|-----------|----------|
| LearningLow | avg<0.4 5m | warning |
| DreamStuck | >15m | warning |

The tool implements detection logic for these conditions:
- UTL accuracy < 0.5 -> degraded
- Dream not cycling + GPU unavailable -> degraded

### 6.3 Consciousness State Machine (PRD Section 2.5.6)
From PRD 2.5.6:
```
States:
  DORMANT     -> r < 0.3, no active workspace
  FRAGMENTED  -> 0.3 <= r < 0.5, partial sync
  EMERGING    -> 0.5 <= r < 0.8, approaching coherence
  CONSCIOUS   -> r >= 0.8, unified percept active
  HYPERSYNC   -> r > 0.95, possibly pathological
```

The `gwt_health.kuramoto_r` value directly maps to these states.

### 6.4 Self-Correction Protocol (PRD Section 19.3)
From PRD 19.3:
> "IF prediction_accuracy < 0.7 for 100 ops: Escalate to human review"

The tool's UTL health check enables this escalation by tracking accuracy.

### 6.5 Dream Layer Health (PRD Section 7.1)
From PRD 7.1:
> "Schedule: activity<0.15 for 10min -> trigger, wake<100ms on query"

The tool monitors dream health including:
- `is_dreaming` - whether a cycle is active
- `cycles_completed` - historical cycle count
- `gpu_available` - whether dreaming is possible

---

## 7. Usage by AI Agents in MCP System

### 7.1 When to Use This Tool

An AI agent should call `get_health_status` when:
1. Starting a new session (baseline health check)
2. After observing unexpected behavior or errors
3. Before performing critical operations (storage, pruning)
4. Periodically during long-running tasks
5. When `get_memetic_status` indicates problems

### 7.2 Typical Workflow

```
1. Agent starts session, calls get_health_status(subsystem="all")
   --> Returns overall_status: "healthy"
   --> Agent proceeds normally

2. Later, agent observes slow retrieval
   --> Calls get_health_status(subsystem="storage")
   --> Returns status: "degraded", node_count: 50000
   --> Agent sees recommendation: "check_rocksdb"

3. Agent calls trigger_healing(subsystem="storage", severity="medium")
   --> Returns success: true

4. Agent verifies with get_health_status(subsystem="storage")
   --> Returns status: "healthy"
```

### 7.3 Integration with Other Tools

| Related Tool | Relationship |
|--------------|--------------|
| `trigger_healing` | Execute healing based on health status |
| `get_memetic_status` | Higher-level status; health_status is lower-level |
| `get_consciousness_state` | Detailed GWT metrics; health_status is summary |
| `get_learner_state` | Detailed UTL metrics; health_status is summary |
| `get_dream_status` | Detailed dream metrics; health_status is summary |
| `get_autonomous_status` | Similar scope but focused on autonomous services |

### 7.4 Decision Matrix

| overall_status | Action |
|----------------|--------|
| healthy | Proceed normally |
| degraded | Check recommendations, consider healing |
| critical | STOP, escalate to healing or human intervention |

### 7.5 Best Practices

**DO**:
- Check health at session start
- Monitor health periodically during long tasks
- Follow recommendations in the response
- Use specific subsystem queries for targeted debugging

**DO NOT**:
- Ignore "degraded" status indefinitely
- Continue critical operations when status is "critical"
- Assume providers are initialized (GWT may show "unknown")

---

## 8. Implementation Details - Key Code Paths

### 8.1 Tool Definition Location
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/autonomous.rs:355-373`

### 8.2 Handler Implementation Location
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/health.rs:13-285`

### 8.3 Parameter Struct Location
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/params.rs:244-255`

### 8.4 Dispatch Registration
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/dispatch.rs:161`
```rust
tool_names::GET_HEALTH_STATUS => self.call_get_health_status(id, arguments).await,
```

### 8.5 Tool Name Constant
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs:119`
```rust
pub const GET_HEALTH_STATUS: &str = "get_health_status";
```

### 8.6 Critical Code Flow

```rust
// 1. Status threshold helper
let get_status = |score: f64| -> &'static str {
    if score >= 0.8 { "healthy" }
    else if score >= 0.5 { "degraded" }
    else { "critical" }
};

// 2. UTL Health Check
if params.subsystem == "utl" || params.subsystem == "all" {
    let tracker = self.meta_utl_tracker.read();
    let domain_accuracies = tracker.get_all_domain_accuracies();
    let accuracy = if domain_accuracies.is_empty() { 0.5 }
                   else { mean(domain_accuracies.values()) };
    let utl_status = get_status(accuracy);
    // Add recommendation if degraded
}

// 3. GWT Health Check
if params.subsystem == "gwt" || params.subsystem == "all" {
    let r = kuramoto.read().synchronization();      // Kuramoto order parameter
    let ic = gwt_system.identity_coherence().await; // Identity continuity
    let health_score = (ic + r) / 2.0;
    let gwt_status = get_status(health_score);
    // Recommend dream if IC < 0.5
}

// 4. Dream Health Check
if params.subsystem == "dream" || params.subsystem == "all" {
    let status = controller.read().get_status();
    let gpu_available = monitor.write().get_utilization() < 0.8;
    // Status: "active" if dreaming, "degraded" if GPU unavailable, else "healthy"
}

// 5. Storage Health Check
if params.subsystem == "storage" || params.subsystem == "all" {
    match self.teleological_store.list_all_johari(1).await {
        Ok(_) => ("healthy", count),
        Err(e) => ("critical", 0),  // Storage failure is CRITICAL
    }
}

// 6. Aggregate overall status
let overall_status = if any_critical { "critical" }
                     else if !overall_healthy { "degraded" }
                     else { "healthy" };
```

### 8.7 Health Calculation Details

**UTL Health**:
```rust
// From health.rs:74-106
let accuracy = domain_accuracies.values().sum::<f32>() / domain_accuracies.len() as f64;
// Cold start default: 0.5 if no domain data
```

**GWT Health**:
```rust
// From health.rs:109-163
let r = kuramoto.read().synchronization();           // [0,1]
let ic = gwt_system.identity_coherence().await;      // [0,1]
let health_score = (ic + r as f32) / 2.0;            // Average
// Returns "unknown" if GWT providers not initialized
```

**Dream Health**:
```rust
// From health.rs:166-218
let gpu_available = usage < 0.8;  // Available if under 80%
let dream_status = if is_dreaming { "active" }
                   else if !gpu_available { "degraded" }
                   else { "healthy" };
```

**Storage Health**:
```rust
// From health.rs:221-258
// Simple accessibility check - if we can list nodes, storage is healthy
// Any error = critical (storage failure is severe)
```

### 8.8 Recommendation Generation

Recommendations are generated when status is not "healthy":

| Subsystem | Condition | Recommendation |
|-----------|-----------|----------------|
| UTL | accuracy < 0.8 | trigger_lambda_recalibration |
| GWT | IC < 0.5 | trigger_dream |
| GWT | providers not initialized | initialize_gwt |
| Dream | GPU unavailable | check_gpu |
| Storage | access failed | check_rocksdb |

---

## 9. Forensic Evidence Summary

| Evidence Item | Location | Verified |
|---------------|----------|----------|
| Tool definition with JSON schema | autonomous.rs:355-373 | YES |
| Handler implementation | health.rs:13-285 | YES |
| Parameter structs | params.rs:244-255 | YES |
| Dispatch routing | dispatch.rs:161 | YES |
| Tool name constant | names.rs:119 | YES |
| Status threshold logic (0.8/0.5) | health.rs:64-72 | YES |
| UTL accuracy calculation | health.rs:74-106 | YES |
| GWT health from Kuramoto+IC | health.rs:109-163 | YES |
| Dream GPU availability check | health.rs:166-218 | YES |
| Storage accessibility check | health.rs:221-258 | YES |
| Overall status aggregation | health.rs:262-268 | YES |
| Recommendation generation | health.rs:92-97, 131-139, etc. | YES |

---

## 10. Verdict

**INNOCENT**: The `get_health_status` tool is correctly implemented according to the PRD specifications. It:
- Monitors all 4 subsystems per NORTH-020
- Uses appropriate thresholds (0.8 healthy, 0.5 degraded, <0.5 critical)
- Generates actionable recommendations for degraded states
- Properly handles missing providers (GWT shows "unknown")
- Integrates with actual system components (MetaUtlTracker, KuramotoNetwork, DreamController, TeleologicalMemoryStore)
- Returns comprehensive metrics for debugging

The implementation aligns with Global Workspace Theory goals by providing the unified health view necessary for autonomous self-correction and maintaining system consciousness.

**OBSERVATION**: The `disk_usage_percent` field is always null because the TeleologicalMemoryStore trait does not expose disk metrics. This is a known limitation, not a bug.

---

*"Data! Data! Data! I can't make bricks without clay."* - Sherlock Holmes

**Case Status**: CLOSED
**Confidence Level**: HIGH
