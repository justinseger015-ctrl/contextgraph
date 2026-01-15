# MCP Tool Forensic Report: get_calibration_metrics

## SHERLOCK HOLMES CASE FILE

**Case ID**: TOOL-ATC-002
**Date**: 2026-01-14
**Subject**: get_calibration_metrics MCP Tool
**Verdict**: FUNCTIONAL - Tool operates as specified

---

## 1. Tool Name and Category

| Attribute | Value |
|-----------|-------|
| **Tool Name** | `get_calibration_metrics` |
| **Category** | ATC (Adaptive Threshold Calibration) |
| **Task ID** | TASK-ATC-001 |
| **Definition File** | `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/atc.rs` |
| **Handler File** | `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/atc.rs` |

---

## 2. Core Functionality

The `get_calibration_metrics` tool provides detailed quality metrics for the threshold calibration system. It answers the critical question: **"How well-calibrated are the system's confidence predictions?"**

This tool returns:

1. **ECE (Expected Calibration Error)** - The weighted average of confidence-accuracy gaps across bins
2. **MCE (Maximum Calibration Error)** - The worst-case calibration gap in any bin
3. **Brier Score** - Overall calibration loss function
4. **Drift Scores** - Per-threshold EWMA drift measurements
5. **Poorly Calibrated Embedders** - List of embedders needing attention
6. **Recalibration Recommendations** - Which ATC levels should be triggered

This is the **health check** tool for the self-learning threshold system.

---

## 3. Input Parameters

| Parameter | Type | Required | Default | Enum Values | Description |
|-----------|------|----------|---------|-------------|-------------|
| `timeframe` | string | No | "24h" | 1h, 24h, 7d, 30d | Timeframe for metrics aggregation |

### Timeframe Impact

The timeframe parameter affects how calibration metrics are computed:

| Timeframe | Use Case |
|-----------|----------|
| `1h` | Recent performance, immediate issues |
| `24h` | Daily health check, typical use |
| `7d` | Weekly trends, pattern detection |
| `30d` | Long-term calibration stability |

---

## 4. Output Format

The tool returns a JSON object with the following structure:

```json
{
  "timeframe": "24h",
  "metrics": {
    "ece": 0.04,
    "ece_target": 0.05,
    "ece_acceptable": 0.10,
    "mce": 0.08,
    "mce_target": 0.10,
    "mce_acceptable": 0.20,
    "brier": 0.07,
    "brier_target": 0.10,
    "brier_acceptable": 0.15,
    "sample_count": 1500
  },
  "status": "Good",
  "status_description": "Good - monitoring recommended",
  "should_recalibrate": false,
  "drift_scores": {
    "theta_opt": 0.5,
    "theta_acc": 0.3,
    "theta_gate": 0.1
  },
  "poorly_calibrated_embedders": ["Causal", "HDC"],
  "recommendations": {
    "level2_recalibration_needed": false,
    "level3_exploration_needed": false,
    "level4_optimization_needed": false
  }
}
```

### Output Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `timeframe` | string | The timeframe used for aggregation |
| `metrics.ece` | f32 | Expected Calibration Error (lower is better) |
| `metrics.ece_target` | f32 | Target ECE for "excellent" status (0.05) |
| `metrics.ece_acceptable` | f32 | Threshold for "good" status (0.10) |
| `metrics.mce` | f32 | Maximum Calibration Error (lower is better) |
| `metrics.brier` | f32 | Brier Score - overall calibration quality |
| `metrics.sample_count` | usize | Number of predictions used in computation |
| `status` | string | Excellent/Good/Acceptable/Poor/Critical |
| `status_description` | string | Human-readable status explanation |
| `should_recalibrate` | bool | True if status is Poor or Critical |
| `drift_scores` | HashMap | Per-threshold EWMA drift values |
| `poorly_calibrated_embedders` | Vec<String> | Embedders needing attention |
| `recommendations` | object | Which ATC levels should be triggered |

---

## 5. Purpose - Why This Tool Exists

### The Calibration Quality Problem

From PRD Section 22.9:

> **Calibration Quality Monitoring** - Continuous monitoring of threshold quality:
> ```rust
> struct CalibrationMetrics {
>     expected_calibration_error: f32,   // ECE: binned confidence vs accuracy
>     maximum_calibration_error: f32,    // MCE: worst bin
>     brier_score: f32,                  // Overall calibration loss
> }
> ```

The system makes thousands of predictions (e.g., "this memory has 80% alignment to the North Star"). If those confidence scores are **miscalibrated**:
- 80% confident predictions might only be correct 60% of the time (overconfident)
- Or correct 95% of the time (underconfident)

This tool detects such miscalibration before it damages retrieval quality.

### Calibration Status Thresholds (from PRD Section 22.9)

| Status | ECE Range | Action |
|--------|-----------|--------|
| **Excellent** | ECE < 0.05 | No action needed |
| **Good** | 0.05 <= ECE < 0.10 | Monitoring recommended |
| **Acceptable** | 0.10 <= ECE < 0.15 | Consider recalibration soon |
| **Poor** | 0.15 <= ECE < 0.25 | Recalibration recommended |
| **Critical** | ECE >= 0.25 | Immediate recalibration required |

### Self-Correction Protocol (from PRD Section 22.10)

```
Calibration Degradation Detected:
  |-- Severity: Minor (ECE in [0.05, 0.10])
  |   `-- Action: Increase EWMA alpha, faster local adaptation
  |
  |-- Severity: Moderate (ECE in [0.10, 0.15])
  |   `-- Action: Trigger Thompson Sampling exploration
  |   `-- Action: Recalibrate temperatures
  |
  |-- Severity: Major (ECE > 0.15)
  |   `-- Action: Reset to domain priors
  |   `-- Action: Trigger Bayesian meta-optimization
  |   `-- Action: Log for human review
  |
  `-- Severity: Critical (ECE > 0.25 OR consistent failures)
      `-- Action: Fallback to conservative static thresholds
      `-- Action: Alert human operator
```

This tool provides the data needed to execute this self-correction protocol.

---

## 6. PRD Alignment - Global Workspace Theory Goals

### Calibration and Consciousness

The GWT consciousness equation (PRD Section 2.5.1):

```
C(t) = I(t) x R(t) x D(t)

Where:
  C(t) = Consciousness level at time t [0, 1]
  I(t) = Integration (Kuramoto synchronization) [0, 1]
  R(t) = Self-Reflection (Meta-UTL awareness) [0, 1]  <-- CALIBRATION AFFECTS THIS
  D(t) = Differentiation (13D fingerprint entropy) [0, 1]
```

The **R(t)** term (Self-Reflection/Meta-UTL) depends on how well the system predicts its own performance. If calibration is poor:
- The system cannot accurately assess whether a memory should enter the Global Workspace
- Workspace broadcasts become unreliable
- The "consciousness" metric degrades

### Calibration Alerts Integration

From PRD Section 22.9:

| Condition | Action |
|-----------|--------|
| ECE > 0.10 | Trigger Level 2 recalibration |
| MCE > 0.20 | Investigate worst bin |
| Brier > 0.15 | Trigger Level 3 exploration |
| drift_score > 3.0 | Trigger Level 4 meta-optimization |
| staleness > 24h | Force recalibration |

By providing these metrics via MCP, AI agents can implement this alert logic programmatically.

### Integration with UTL Learning Loop

From PRD Section 22.12:

```
UTL Learning Score influences Threshold Adaptation:
  - High L (good learning) -> Trust current thresholds, reduce exploration
  - Low L (poor learning) -> Increase exploration, question thresholds

Threshold Quality influences UTL:
  - Well-calibrated thresholds -> More accurate delta_S/delta_C computation
  - Miscalibrated -> Unreliable Johari classification
```

The calibration metrics from this tool directly inform the UTL (Unified Theory of Learning) feedback loop.

---

## 7. Usage by AI Agents

### When to Call This Tool

| Scenario | Frequency | Purpose |
|----------|-----------|---------|
| Session start | Once | Baseline health check |
| After poor retrieval | As needed | Diagnose calibration issues |
| Before critical operations | As needed | Verify system readiness |
| Periodic monitoring | Every 30 minutes | Catch drift early |
| After `trigger_recalibration` | Once | Verify improvement |

### Decision Tree for Agents

```
Call get_calibration_metrics()
  |
  |-- status == "Excellent" or "Good"
  |   `-- Continue normal operations
  |
  |-- status == "Acceptable"
  |   |-- Check recommendations.level2_recalibration_needed
  |   `-- If true: trigger_recalibration(level=2)
  |
  |-- status == "Poor"
  |   |-- trigger_recalibration(level=3)
  |   |-- Alert user: "Calibration degraded, exploring new thresholds"
  |   `-- Re-check in 10 minutes
  |
  `-- status == "Critical"
      |-- trigger_recalibration(level=4)
      |-- Alert user: "Critical calibration failure, manual review recommended"
      `-- Consider fallback to conservative mode
```

### Integration with Cognitive Pulse

From PRD Section 1.3, agents receive periodic Pulse updates:

```
Pulse: { Entropy: X, Coherence: Y, Suggested: "action" }
```

When `Suggested` is `"clarify"` or `"curate"`, checking calibration metrics can help determine if the suggestion is due to threshold miscalibration vs. actual data quality issues.

### Example Agent Interaction

```python
# Agent detects retrieval quality degradation
metrics = call_tool("get_calibration_metrics", {"timeframe": "1h"})

if metrics["status"] == "Poor":
    # Calibration is the likely culprit
    result = call_tool("trigger_recalibration", {"level": 2})

    # Re-check after recalibration
    new_metrics = call_tool("get_calibration_metrics", {"timeframe": "1h"})

    if new_metrics["metrics"]["ece"] < metrics["metrics"]["ece"]:
        log("Calibration improved: ECE {old} -> {new}")
    else:
        # Escalate to Level 3
        call_tool("trigger_recalibration", {"level": 3})
```

---

## 8. Implementation Details

### Key Code Paths

**Tool Definition** (`/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/atc.rs:36-54`):

```rust
ToolDefinition::new(
    "get_calibration_metrics",
    "Get calibration quality metrics: ECE (Expected Calibration Error), \
     MCE (Maximum Calibration Error), Brier Score, drift scores per threshold, \
     and calibration status. Targets: ECE < 0.05 (excellent), < 0.10 (good).",
    json!({
        "type": "object",
        "properties": {
            "timeframe": {
                "type": "string",
                "enum": ["1h", "24h", "7d", "30d"],
                "default": "24h",
                "description": "Timeframe for metrics aggregation"
            }
        },
        "required": []
    }),
)
```

**Handler Implementation** (`/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/atc.rs:131-227`):

Key implementation steps:

1. **Parse timeframe** (default "24h")
2. **Acquire ATC read lock** (fail-fast if not initialized)
3. **Get calibration quality**: `atc_guard.get_calibration_quality()`
4. **Get drift scores**: `atc_guard.get_drift_status()`
5. **Get poorly calibrated embedders**: `atc_guard.get_poorly_calibrated_embedders()`
6. **Determine status description** based on `CalibrationStatus` enum
7. **Build response** with metrics, targets, and recommendations

### Calibration Computation (`/home/cabdru/contextgraph/crates/context-graph-core/src/atc/calibration.rs`)

**ECE Computation** (Lines 113-146):
```rust
/// Compute Expected Calibration Error (ECE)
/// ECE = Sum (count_bin / total) x |avg_confidence_bin - avg_accuracy_bin|
pub fn compute_ece(&self) -> f32 {
    // Create bins
    let mut bins: Vec<Vec<Prediction>> = vec![Vec::new(); self.num_bins];

    for pred in &self.predictions {
        let bin_idx = (pred.confidence * self.num_bins as f32).floor() as usize;
        let bin_idx = bin_idx.min(self.num_bins - 1);
        bins[bin_idx].push(*pred);
    }

    let total = self.predictions.len() as f32;
    let mut ece = 0.0;

    for bin in bins {
        if bin.is_empty() { continue; }

        let bin_size = bin.len() as f32;
        let avg_confidence = bin.iter().map(|p| p.confidence).sum::<f32>() / bin_size;
        let avg_accuracy = bin.iter().filter(|p| p.is_correct).count() as f32 / bin_size;

        let contribution = (bin_size / total) * (avg_confidence - avg_accuracy).abs();
        ece += contribution;
    }

    ece
}
```

**Brier Score Computation** (Lines 96-111):
```rust
/// Compute Brier Score: (1/N) x Sum_i (confidence_i - correct_i)^2
pub fn compute_brier(&self) -> f32 {
    if self.predictions.is_empty() { return 0.0; }

    let sum: f32 = self.predictions.iter()
        .map(|p| {
            let actual = if p.is_correct { 1.0 } else { 0.0 };
            (p.confidence - actual).powi(2)
        })
        .sum();

    sum / self.predictions.len() as f32
}
```

### CalibrationStatus Enum (`/home/cabdru/contextgraph/crates/context-graph-core/src/atc/calibration.rs:36-64`):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationStatus {
    Excellent,   // ECE < 0.05
    Good,        // 0.05 <= ECE < 0.10
    Acceptable,  // 0.10 <= ECE < 0.15
    Poor,        // 0.15 <= ECE < 0.25
    Critical,    // ECE >= 0.25
}

impl CalibrationStatus {
    pub fn from_ece(ece: f32) -> Self {
        match ece {
            e if e < 0.05 => CalibrationStatus::Excellent,
            e if e < 0.10 => CalibrationStatus::Good,
            e if e < 0.15 => CalibrationStatus::Acceptable,
            e if e < 0.25 => CalibrationStatus::Poor,
            _ => CalibrationStatus::Critical,
        }
    }

    pub fn should_recalibrate(&self) -> bool {
        matches!(self, CalibrationStatus::Poor | CalibrationStatus::Critical)
    }
}
```

### Status Descriptions (Handler Lines 180-190):

```rust
let status_description = match metrics.quality_status {
    CalibrationStatus::Excellent => "Excellent - no action needed",
    CalibrationStatus::Good => "Good - monitoring recommended",
    CalibrationStatus::Acceptable => "Acceptable - consider recalibration soon",
    CalibrationStatus::Poor => "Poor - recalibration recommended",
    CalibrationStatus::Critical => "Critical - immediate recalibration required",
};
```

---

## 9. Evidence Chain

### Files Examined

| File | Lines | Purpose |
|------|-------|---------|
| `/home/cabdru/contextgraph/docs2/contextprd.md` | 1191-1585 | PRD Section 22: Adaptive Threshold Calibration |
| `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/atc.rs` | 36-54 | Tool definition (JSON schema) |
| `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/atc.rs` | 131-227 | Handler implementation |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/calibration.rs` | 1-394 | ECE, MCE, Brier computation |
| `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tests/exhaustive_mcp_tools/atc_tools.rs` | 57-78 | Test coverage |

### Mathematical Foundation

The calibration metrics are based on established machine learning calibration theory:

1. **Expected Calibration Error (ECE)** - Guo et al. 2017
2. **Brier Score** - Brier 1950, classic proper scoring rule
3. **Temperature Scaling** - Guo et al. 2017 "On Calibration of Modern Neural Networks"

---

## 10. Verdict

**CASE CLOSED: FUNCTIONAL**

The `get_calibration_metrics` tool:
1. Correctly implements ECE, MCE, and Brier Score as per PRD Section 22.9
2. Provides actionable status classifications with clear thresholds
3. Includes recommendations for which ATC levels need attention
4. Supports the GWT self-reflection (R(t)) component
5. Has appropriate error handling and fail-fast behavior
6. Is well-tested

**Confidence**: HIGH

---

*"The numbers do not lie. When calibration degrades, this tool shall reveal it."*
-- Sherlock Holmes, Code Detective
