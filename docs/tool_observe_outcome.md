# Forensic Investigation Report: observe_outcome Tool

## SHERLOCK HOLMES CASE FILE

**Case ID**: TOOL-OBSERVE-OUTCOME-001
**Date**: 2026-01-14
**Subject**: MCP Tool observe_outcome
**Investigator**: Sherlock Holmes (Forensic Code Investigation Agent)

---

## 1. Tool Name and Category

**Tool Name**: `observe_outcome`
**Category**: Autonomous North Star System / Meta-UTL Learning
**Specification Reference**: SPEC-AUTONOMOUS-001, NORTH-009, METAUTL-001

---

## 2. Core Functionality

The `observe_outcome` tool records the actual outcome for a prior Meta-UTL prediction, enabling the system to learn from the difference between predicted and actual results.

**What It Does**:
1. Accepts a prediction ID and the actual outcome value
2. Looks up the original prediction from the PredictionHistory store
3. Calculates the prediction error (|actual - predicted|)
4. Triggers lambda weight adjustment if error exceeds 0.2 threshold
5. Updates accuracy tracking for domain-specific learning
6. Returns comprehensive feedback on the outcome recording

**FAIL FAST Behavior**:
- If the prediction_id is not found in history, the tool returns an error (no fallback to hardcoded default of 0.5)
- This enforces proper prediction tracking discipline throughout the system

---

## 3. Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prediction_id` | string (UUID) | YES | UUID of the prediction to update |
| `actual_outcome` | number (0.0-1.0) | YES | The actual outcome value observed |
| `context` | object | NO | Optional context containing domain and query_type |

**Context Object Schema**:
```json
{
  "domain": "string (Code, Medical, General)",
  "query_type": "string (retrieval, classification, etc.)"
}
```

**Evidence - Parameter Definition** (from `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/params.rs:205-227`):
```rust
pub struct ObserveOutcomeParams {
    pub prediction_id: String,
    pub actual_outcome: f32,
    pub context: Option<ObserveOutcomeContext>,
}

pub struct ObserveOutcomeContext {
    pub domain: Option<String>,
    pub query_type: Option<String>,
}
```

---

## 4. Output Format

**Successful Response**:
```json
{
  "accepted": true,
  "prediction_id": "uuid-string",
  "predicted_value": 0.75,
  "actual_outcome": 0.82,
  "prediction_error": 0.07,
  "lambda_adjusted": false,
  "new_accuracy": 0.93,
  "overall_accuracy": 0.85,
  "domain": "Code"
}
```

**Error Response (Prediction Not Found)**:
```json
{
  "error": {
    "code": -32610,
    "message": "Prediction {uuid} not found in history. Predictions expire after 24 hours (EC-AUTO-03). Ensure predictions are stored via PredictionHistory before calling observe_outcome."
  }
}
```

**Output Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `accepted` | boolean | Whether the outcome was recorded |
| `prediction_id` | string | The UUID of the prediction |
| `predicted_value` | number | Original predicted value from history |
| `actual_outcome` | number | The actual outcome provided |
| `prediction_error` | number | Absolute difference |actual - predicted| |
| `lambda_adjusted` | boolean | Whether lambda weights were adjusted |
| `new_accuracy` | number | 1.0 - prediction_error |
| `overall_accuracy` | number | Current tracker accuracy |
| `domain` | string | Domain extracted from prediction or context |

---

## 5. Purpose - Why This Tool Exists

The `observe_outcome` tool is a critical component of the Meta-UTL (Meta Unified Theory of Learning) self-correction system. It exists to:

### 5.1 Enable Self-Aware Learning
Per the PRD Section 19 (Meta-UTL), the system must "learn about its own learning." This tool provides the feedback mechanism where the system compares its predictions against reality.

### 5.2 Trigger Lambda Adjustment
Per METAUTL-001: "prediction_error > 0.2 triggers lambda adjustment." The tool enforces this protocol by checking if the error exceeds the threshold and flagging when adjustment is needed.

### 5.3 Maintain Domain-Specific Accuracy
The tool tracks accuracy per domain (Code, Medical, General), enabling the system to have calibrated confidence for different types of queries.

### 5.4 Complete the Prediction-Observation Loop
```
[Prediction Made] --> [store in PredictionHistory]
                              |
                              v
[Time Passes] --> [observe_outcome called with actual]
                              |
                              v
[Error Calculated] --> [Lambda Adjusted if > 0.2]
                              |
                              v
[Accuracy Updated] --> [System Improves]
```

---

## 6. PRD Alignment - Global Workspace Theory Goals

### 6.1 Steering Subsystem Integration (PRD Section 7.8)
The tool aligns with the Steering Subsystem described in PRD 7.8:
> "SteeringReward: reward:f32[-1,1], gardener_score, curator_score, assessor_score, explanation, suggestions"

The `observe_outcome` provides the ground truth that the steering system uses to adjust rewards and improve future storage decisions.

### 6.2 Self-Correction Protocol (PRD Section 19.3)
From PRD 19.3:
> "IF prediction_error > 0.2: Log to meta_learning_events, Adjust UTL parameters, Retrain predictor if persistent"

The tool directly implements the first two steps of this protocol.

### 6.3 Meta-Cognitive Loop (PRD Section 2.5.5)
The tool supports the Meta-UTL MetaScore calculation:
```
MetaScore = sigmoid(2 x (L_predicted - L_actual))
```

By recording actual outcomes, the tool provides `L_actual` for this calculation.

### 6.4 Per-Embedder Meta-Analysis (PRD Section 19.4)
> "Track which embedding spaces are most predictive"

The domain-specific accuracy tracking enables understanding which domains the system predicts well vs. poorly.

---

## 7. Usage by AI Agents in MCP System

### 7.1 When to Use This Tool

An AI agent should call `observe_outcome` when:
1. A prior prediction was made (e.g., retrieval quality prediction, storage impact prediction)
2. The actual result is now known
3. The prediction_id was tracked and is within the 24-hour TTL window

### 7.2 Typical Workflow

```
1. Agent calls store_memory with content
   --> System makes prediction about storage quality
   --> prediction_id returned (e.g., via Meta-UTL hooks)

2. Agent observes actual usage/utility of stored memory

3. Agent calls observe_outcome:
   {
     "prediction_id": "abc-123-def",
     "actual_outcome": 0.85,
     "context": {
       "domain": "Code"
     }
   }

4. System updates its learning weights based on error
```

### 7.3 Integration with Other Tools

| Related Tool | Relationship |
|--------------|--------------|
| `get_learner_state` | Check accuracy before/after observing outcomes |
| `trigger_lambda_recalibration` | Manual trigger if observe_outcome shows persistent errors |
| `get_meta_learning_status` | Monitor overall self-correction health |
| `store_memory` | Predictions are made when storing; outcomes observed later |

### 7.4 Error Handling Best Practices

```
IF observe_outcome returns "Prediction not found":
  - The prediction may have expired (24h TTL)
  - The prediction_id may be incorrect
  - The prediction may never have been stored

  DO NOT: Use a fallback value
  DO: Log the issue and skip this observation
```

---

## 8. Implementation Details - Key Code Paths

### 8.1 Tool Definition Location
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/autonomous.rs:287-323`

### 8.2 Handler Implementation Location
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/learner.rs:109-273`

### 8.3 Parameter Struct Location
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/params.rs:205-227`

### 8.4 Prediction History Dependency
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous/prediction_history.rs`

### 8.5 Dispatch Registration
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/dispatch.rs:159`
```rust
tool_names::OBSERVE_OUTCOME => self.call_observe_outcome(id, arguments).await,
```

### 8.6 Tool Name Constant
**File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/names.rs:115`
```rust
pub const OBSERVE_OUTCOME: &str = "observe_outcome";
```

### 8.7 Critical Code Flow

```rust
// 1. Parse prediction_id as UUID (FAIL FAST on invalid format)
let prediction_uuid = match Uuid::parse_str(&params.prediction_id) {...}

// 2. Validate actual_outcome range [0.0, 1.0]
if params.actual_outcome < 0.0 || params.actual_outcome > 1.0 {...}

// 3. Look up prediction from history (FAIL FAST if not found)
let prediction_entry = {
    let mut history = self.prediction_history.write();
    history.take(&prediction_uuid)
};

// 4. Calculate prediction error
let prediction_error = (params.actual_outcome - predicted_value).abs();

// 5. Per METAUTL-001: trigger lambda adjustment if error > 0.2
let lambda_adjusted = prediction_error > 0.2;

// 6. Calculate accuracy as 1.0 - error
let accuracy = 1.0 - prediction_error.min(1.0);
```

### 8.8 PredictionHistory TTL Enforcement
Per EC-AUTO-03, predictions expire after 24 hours:
```rust
const PREDICTION_TTL: Duration = Duration::from_secs(24 * 60 * 60);

pub fn is_expired(&self) -> bool {
    self.created_at.elapsed() > PREDICTION_TTL
}
```

---

## 9. Forensic Evidence Summary

| Evidence Item | Location | Verified |
|---------------|----------|----------|
| Tool definition with JSON schema | autonomous.rs:287-323 | YES |
| Handler implementation | learner.rs:109-273 | YES |
| Parameter structs | params.rs:205-227 | YES |
| Dispatch routing | dispatch.rs:159 | YES |
| Tool name constant | names.rs:115 | YES |
| PredictionHistory dependency | prediction_history.rs | YES |
| FAIL FAST behavior | learner.rs:203-219 | YES |
| Lambda adjustment threshold (0.2) | learner.rs:226 | YES |
| 24-hour TTL enforcement | prediction_history.rs:16 | YES |

---

## 10. Verdict

**INNOCENT**: The `observe_outcome` tool is correctly implemented according to the PRD specifications. It:
- Enforces FAIL FAST when predictions are not found (no fallbacks)
- Correctly calculates prediction error
- Triggers lambda adjustment per METAUTL-001 threshold of 0.2
- Integrates with the broader Meta-UTL self-correction system
- Respects the 24-hour TTL per EC-AUTO-03

The implementation aligns with Global Workspace Theory goals by enabling the system to learn from its prediction errors and improve over time.

---

*"The game is never lost till it is won."* - Sherlock Holmes

**Case Status**: CLOSED
**Confidence Level**: HIGH
