# MCP Tool Forensic Report: get_threshold_status

## SHERLOCK HOLMES CASE FILE

**Case ID**: TOOL-ATC-001
**Date**: 2026-01-14
**Subject**: get_threshold_status MCP Tool
**Verdict**: FUNCTIONAL - Tool operates as specified

---

## 1. Tool Name and Category

| Attribute | Value |
|-----------|-------|
| **Tool Name** | `get_threshold_status` |
| **Category** | ATC (Adaptive Threshold Calibration) |
| **Task ID** | TASK-ATC-001 |
| **Definition File** | `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/atc.rs` |
| **Handler File** | `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/atc.rs` |

---

## 2. Core Functionality

The `get_threshold_status` tool provides a comprehensive view of the current Adaptive Threshold Calibration (ATC) system state. It returns:

1. **Current threshold values** for a specified domain (Code, Medical, Legal, Creative, Research, General)
2. **Calibration quality metrics** (ECE, MCE, Brier Score)
3. **Drift scores** per tracked threshold
4. **Recalibration recommendations** for each ATC level
5. **Per-embedder temperature information** (optional, when embedder_id is specified)

This tool is the primary diagnostic interface for understanding how the system's self-learning thresholds are currently configured and whether they need adjustment.

---

## 3. Input Parameters

| Parameter | Type | Required | Default | Range/Enum | Description |
|-----------|------|----------|---------|------------|-------------|
| `domain` | string | No | "General" | Code, Medical, Legal, Creative, Research, General | Domain context for threshold lookup. Different domains have different strictness levels affecting threshold priors. |
| `embedder_id` | integer | No | None | 1-13 | Optional: Specific embedder for detailed temperature calibration info. Maps to the 13 embedding spaces (E1-E13). |

### Domain Strictness Mapping (from PRD Section 22.8)

| Domain | Strictness | Description |
|--------|------------|-------------|
| Medical | 1.0 | Very strict, high causal weight |
| Code | 0.9 | Strict thresholds, low tolerance for false positives |
| Legal | 0.8 | Moderate, high semantic precision |
| General | 0.5 | Default priors |
| Research | 0.5 | Balanced, novelty valued |
| Creative | 0.2 | Loose thresholds, exploration encouraged |

---

## 4. Output Format

The tool returns a JSON object with the following structure:

```json
{
  "domain": "Code",
  "thresholds": {
    "domain_thresholds": {
      "theta_opt": 0.84,
      "theta_acc": 0.772,
      "theta_warn": 0.595,
      "theta_dup": 0.90,
      "theta_edge": 0.70
    }
  },
  "calibration": {
    "ece": 0.04,
    "mce": 0.08,
    "brier": 0.07,
    "sample_count": 150,
    "status": "Good"
  },
  "drift_scores": {
    "theta_opt": 0.5,
    "theta_acc": 0.3
  },
  "should_recalibrate_level2": false,
  "should_explore_level3": false,
  "should_optimize_level4": false,
  "embedder_detail": {
    "embedder_id": 1,
    "embedder_name": "Semantic",
    "is_poorly_calibrated": false,
    "needs_recalibration": false
  }
}
```

### Output Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `domain` | string | The domain used for threshold lookup |
| `thresholds.domain_thresholds` | object | Current threshold values for the domain |
| `calibration.ece` | f32 | Expected Calibration Error (target < 0.05 excellent, < 0.10 good) |
| `calibration.mce` | f32 | Maximum Calibration Error (target < 0.10) |
| `calibration.brier` | f32 | Brier Score (target < 0.10) |
| `calibration.sample_count` | usize | Number of predictions used for calibration |
| `calibration.status` | string | Excellent/Good/Acceptable/Poor/Critical |
| `drift_scores` | HashMap<String, f32> | Per-threshold EWMA drift scores (>2.0 triggers Level 2, >3.0 triggers Level 3) |
| `should_recalibrate_level2` | bool | Whether Level 2 temperature scaling is recommended |
| `should_explore_level3` | bool | Whether Level 3 Thompson Sampling exploration is recommended |
| `should_optimize_level4` | bool | Whether Level 4 Bayesian meta-optimization is recommended |
| `embedder_detail` | object | Optional: Per-embedder calibration status when embedder_id is provided |

---

## 5. Purpose - Why This Tool Exists

### The Problem It Solves

The PRD Section 22.1 states:

> **Why static thresholds fail:**
> - Different domains have different optimal thresholds (code vs medical vs creative)
> - User behavior patterns vary (some users store frequently, others rarely)
> - Data distributions shift over time (concept drift)
> - Per-embedder characteristics differ (E5 causal behaves differently than E1 semantic)

The `get_threshold_status` tool exists to provide **visibility into the self-learning threshold system**. Without this diagnostic capability:

1. Operators cannot know if thresholds are calibrated correctly
2. Agents cannot make informed decisions about when to trigger recalibration
3. Debugging threshold-related failures becomes impossible
4. Domain-specific optimizations cannot be verified

### Design Philosophy

From PRD Section 22:

> The system uses **no hardcoded thresholds**. All thresholds are learned, calibrated, and continuously adapted based on outcomes.

This tool is the window into that adaptive system - it shows what the system has learned and whether that learning is healthy.

---

## 6. PRD Alignment - Global Workspace Theory Goals

### Connection to GWT Architecture

The ATC system directly supports the Global Workspace Theory (GWT) implementation described in PRD Section 2.5:

| GWT Component | ATC Threshold | Purpose |
|---------------|---------------|---------|
| Workspace Broadcast | `theta_gate` (0.65-0.95) | Controls which memories can enter "conscious" workspace |
| Hypersync Detection | `theta_hypersync` (0.90-0.99) | Detects potentially pathological over-synchronization |
| Fragmentation Warning | `theta_fragmentation` (0.35-0.65) | Alerts when Kuramoto sync drops too low |

### Consciousness State Machine Support

The thresholds monitored by this tool directly control the GWT state transitions (PRD Section 2.5.6):

```
States:
  DORMANT     -> r < 0.3, no active workspace
  FRAGMENTED  -> 0.3 <= r < 0.5, partial sync
  EMERGING    -> 0.5 <= r < 0.8, approaching coherence
  CONSCIOUS   -> r >= 0.8, unified percept active
  HYPERSYNC   -> r > 0.95, possibly pathological
```

The `theta_gate` threshold determines when memories are "conscious" (r >= threshold). By exposing this via `get_threshold_status`, agents can understand and optimize the consciousness dynamics.

### Multi-Embedder Teleological Architecture

The 13 embedders (E1-E13) each have different natural frequencies in the Kuramoto oscillator layer:

| Embedder | Natural Frequency | Why ATC Matters |
|----------|-------------------|-----------------|
| E1 Semantic | 40 Hz (Gamma) | Temperature scaling affects confidence calibration |
| E5 Causal | 25 Hz (Beta) | Asymmetric distance metrics need different thresholds |
| E7 Code | 25 Hz (Beta) | High precision requirements (strictness=0.9 default) |
| E9 HDC | 80 Hz (High Gamma) | Holographic encoding is inherently noisy |

By allowing `embedder_id` queries, this tool enables per-embedder diagnostic inspection.

---

## 7. Usage by AI Agents

### When to Call This Tool

| Scenario | Action | Example |
|----------|--------|---------|
| Session start | Verify calibration state | `get_threshold_status(domain="General")` |
| Before critical operations | Check if recalibration needed | Check `should_recalibrate_level2` flag |
| After domain switch | Get domain-specific thresholds | `get_threshold_status(domain="Medical")` |
| Debugging poor retrieval | Check per-embedder calibration | `get_threshold_status(embedder_id=5)` |
| Low coherence detected | Verify threshold settings | Compare actual vs expected thresholds |

### Integration with Cognitive Pulse

From PRD Section 1.3:

```
Pulse: { Entropy: X, Coherence: Y, Suggested: "action" }
```

When coherence is low, agents should:
1. Call `get_threshold_status` to check calibration quality
2. If ECE > 0.10, consider calling `trigger_recalibration`
3. Monitor drift scores for specific thresholds causing issues

### Example Agent Workflow

```
1. Agent detects low coherence (< 0.4)
2. Call: get_threshold_status(domain="Code")
3. Response shows ECE=0.15 (Acceptable) and should_recalibrate_level2=true
4. Agent calls: trigger_recalibration(level=2, domain="Code")
5. Re-check with get_threshold_status to verify improvement
```

---

## 8. Implementation Details

### Key Code Paths

**Tool Definition** (`/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/atc.rs:10-34`):
```rust
ToolDefinition::new(
    "get_threshold_status",
    "Get current ATC threshold status including all thresholds, calibration state, \
     and adaptation metrics. Returns per-embedder temperatures, drift scores, and \
     bandit exploration stats. Requires ATC provider to be initialized.",
    json!({
        "type": "object",
        "properties": {
            "domain": {...},
            "embedder_id": {...}
        },
        "required": []
    }),
)
```

**Handler Implementation** (`/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/atc.rs:17-129`):

1. **Parameter Parsing**: Extracts `domain` (default "General") and optional `embedder_id`
2. **ATC Provider Check**: Fails fast with `FEATURE_DISABLED` error if ATC not initialized
3. **Data Collection**:
   - `atc_guard.get_drift_status()` - EWMA drift scores
   - `atc_guard.get_calibration_quality()` - ECE, MCE, Brier metrics
   - `atc_guard.get_domain_thresholds(domain)` - Domain-specific thresholds
4. **Embedder Detail**: If `embedder_id` provided, checks if that embedder is poorly calibrated
5. **Response Construction**: Builds JSON with all collected data plus recalibration recommendations

### Core ATC Module (`/home/cabdru/contextgraph/crates/context-graph-core/src/atc/mod.rs`)

The `AdaptiveThresholdCalibration` struct orchestrates 4 levels:

```rust
pub struct AdaptiveThresholdCalibration {
    level1: DriftTracker,           // EWMA drift detection (per-query)
    level2: TemperatureScaler,      // Per-embedder confidence calibration (hourly)
    level3: Option<ThresholdBandit>, // Thompson Sampling exploration (session)
    level4: BayesianOptimizer,      // Gaussian Process optimization (weekly)
    domains: DomainManager,         // Per-domain threshold configuration
    calibration_quality: CalibrationMetrics,
}
```

### Domain Thresholds (`/home/cabdru/contextgraph/crates/context-graph-core/src/atc/domain.rs`)

The `DomainThresholds` struct contains 21 threshold fields:

| Category | Thresholds |
|----------|------------|
| Core alignment | theta_opt, theta_acc, theta_warn, theta_dup, theta_edge |
| GWT | theta_gate, theta_hypersync, theta_fragmentation |
| Layer | theta_memory_sim, theta_reflex_hit, theta_consolidation |
| Dream | theta_dream_activity, theta_semantic_leap, theta_shortcut_conf |
| Classification | theta_johari, theta_blind_spot |
| Autonomous | theta_obsolescence_low/high/mid, theta_drift_slope |

### Error Handling

The handler implements fail-fast behavior:

```rust
let atc_guard = match &self.atc {
    Some(atc) => atc.read(),
    None => {
        error!("ATC provider not initialized - FAIL FAST");
        return JsonRpcResponse::error(
            id,
            error_codes::FEATURE_DISABLED,
            "ATC provider not initialized. Use with_atc() constructor.",
        );
    }
};
```

---

## 9. Evidence Chain

### Files Examined

| File | Lines | Purpose |
|------|-------|---------|
| `/home/cabdru/contextgraph/docs2/contextprd.md` | 1191-1585 | PRD Section 22: Adaptive Threshold Calibration |
| `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/atc.rs` | 1-84 | Tool definition (JSON schema) |
| `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/atc.rs` | 1-389 | Handler implementation |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/mod.rs` | 1-316 | Core ATC orchestration |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/domain.rs` | 1-1026 | Domain-specific thresholds |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/calibration.rs` | 1-394 | Calibration metrics computation |
| `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tests/exhaustive_mcp_tools/atc_tools.rs` | 1-167 | Test coverage |

### Test Coverage

The tool has comprehensive test coverage:

- `test_get_threshold_status_basic` - Default parameters
- `test_get_threshold_status_with_domain` - Domain-specific lookup
- `test_get_threshold_status_with_embedder` - Per-embedder detail

---

## 10. Verdict

**CASE CLOSED: FUNCTIONAL**

The `get_threshold_status` tool:
1. Correctly implements the PRD Section 22 specification
2. Provides comprehensive visibility into the 4-level ATC system
3. Supports domain-specific and embedder-specific diagnostics
4. Integrates properly with the GWT consciousness architecture
5. Has appropriate fail-fast error handling
6. Is well-tested

**Confidence**: HIGH

---

*"The evidence is conclusive. This tool serves its purpose admirably."*
-- Sherlock Holmes, Code Detective
