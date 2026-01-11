# Task Specification: MCP Tool Wiring for Meta-UTL

**Task ID:** TASK-METAUTL-P0-005
**Version:** 1.0.0
**Status:** Ready
**Layer:** Surface (Layer 3)
**Sequence:** 5
**Priority:** P0 (Critical)
**Estimated Complexity:** Medium

---

## 1. Metadata

### 1.1 Implements

| Requirement ID | Description |
|----------------|-------------|
| REQ-METAUTL-013 | System SHALL provide MCP tool `get_meta_learning_status` |
| REQ-METAUTL-014 | System SHALL provide MCP tool `trigger_lambda_recalibration` |

### 1.2 Dependencies

| Task ID | Description | Status |
|---------|-------------|--------|
| TASK-METAUTL-P0-001 | Core types | Must complete first |
| TASK-METAUTL-P0-002 | Lambda adjustment | Must complete first |
| TASK-METAUTL-P0-003 | Escalation logic | Must complete first |
| TASK-METAUTL-P0-004 | Event logging | Must complete first |

### 1.3 Blocked By

- TASK-METAUTL-P0-004 (all meta components must exist)

---

## 2. Context

This task wires the Meta-UTL self-correction capabilities to the MCP (Model Context Protocol) layer, exposing three tools for external introspection and control:

1. `get_meta_learning_status` - Read current self-correction state
2. `trigger_lambda_recalibration` - Manually trigger lambda optimization
3. `get_meta_learning_log` - Query historical meta-learning events

These tools enable operators to:
- Monitor the system's self-correction behavior
- Manually intervene when needed
- Debug and analyze meta-learning patterns
- Export data for external analysis

---

## 3. Input Context Files

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/handlers/mod.rs` | Existing handler patterns |
| `crates/context-graph-mcp/src/handlers/utl.rs` | UTL-related handlers |
| `crates/context-graph-mcp/src/schema/tools.rs` | Tool schema definitions |
| `crates/context-graph-utl/src/meta/` | All meta types from previous tasks |
| `specs/functional/SPEC-METAUTL-001.md` | MCP tool contracts |

---

## 4. Scope

### 4.1 In Scope

- Implement `handle_get_meta_learning_status` handler
- Implement `handle_trigger_lambda_recalibration` handler
- Implement `handle_get_meta_learning_log` handler
- Add tool schemas to MCP tool registry
- Input validation for all handlers
- Response serialization to JSON
- Integration with `MetaLearningService`
- Unit tests for handlers
- Integration tests with mock service

### 4.2 Out of Scope

- Full service implementation (handlers call existing service)
- MetaCognitiveLoop integration (TASK-METAUTL-P0-006)
- Performance optimization
- Rate limiting (future enhancement)

---

## 5. Prerequisites

| Check | Description |
|-------|-------------|
| [ ] | TASK-METAUTL-P0-004 completed |
| [ ] | All meta types compile |
| [ ] | MCP handler pattern understood |
| [ ] | `context-graph-mcp` crate exists |

---

## 6. Definition of Done

### 6.1 Required Signatures

#### File: `crates/context-graph-mcp/src/handlers/meta_learning.rs`

```rust
use crate::error::{McpError, McpResult};
use crate::schema::{ToolInput, ToolOutput};
use context_graph_utl::meta::{
    MetaLearningEventLog, AdaptiveLambdaWeights, EscalationManager,
    SelfCorrectionState, MetaLearningEvent, EventLogQuery,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Input for get_meta_learning_status tool
#[derive(Debug, Clone, Deserialize)]
pub struct GetMetaLearningStatusInput {
    /// Include detailed accuracy history (optional)
    #[serde(default)]
    pub include_accuracy_history: bool,
    /// Include per-embedder breakdown (optional)
    #[serde(default)]
    pub include_embedder_breakdown: bool,
}

/// Output for get_meta_learning_status tool
#[derive(Debug, Clone, Serialize)]
pub struct MetaLearningStatusOutput {
    /// Whether self-correction is enabled
    pub enabled: bool,
    /// Current global accuracy (rolling average)
    pub current_accuracy: f32,
    /// Consecutive low accuracy cycle count
    pub consecutive_low_count: u32,
    /// Current lambda weights
    pub current_lambdas: LambdaValues,
    /// Base lambda weights (from lifecycle)
    pub base_lambdas: LambdaValues,
    /// Deviation from base weights
    pub lambda_deviation: LambdaValues,
    /// Current escalation status
    pub escalation_status: String,
    /// Total adjustments made
    pub adjustment_count: u64,
    /// Recent events count (last 24h)
    pub recent_events_count: usize,
    /// Last adjustment timestamp (if any)
    pub last_adjustment_at: Option<DateTime<Utc>>,
    /// Accuracy history (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy_history: Option<Vec<f32>>,
    /// Per-embedder accuracy (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedder_accuracy: Option<Vec<f32>>,
}

/// Lambda weight values
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LambdaValues {
    pub lambda_s: f32,
    pub lambda_c: f32,
}

/// Input for trigger_lambda_recalibration tool
#[derive(Debug, Clone, Deserialize)]
pub struct TriggerRecalibrationInput {
    /// Force Bayesian optimization (skip gradient check)
    #[serde(default)]
    pub force_bayesian: bool,
    /// Target domain for calibration (optional)
    pub domain: Option<String>,
    /// Dry run - compute but don't apply
    #[serde(default)]
    pub dry_run: bool,
}

/// Output for trigger_lambda_recalibration tool
#[derive(Debug, Clone, Serialize)]
pub struct RecalibrationOutput {
    /// Whether recalibration succeeded
    pub success: bool,
    /// Adjustment applied (or that would be applied)
    pub adjustment: Option<AdjustmentDetails>,
    /// New lambda values
    pub new_lambdas: LambdaValues,
    /// Previous lambda values
    pub previous_lambdas: LambdaValues,
    /// Method used (gradient or bayesian)
    pub method: String,
    /// Bayesian optimization iterations (if applicable)
    pub bo_iterations: Option<usize>,
    /// Expected improvement (if BO)
    pub expected_improvement: Option<f32>,
    /// Whether this was a dry run
    pub dry_run: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Adjustment details
#[derive(Debug, Clone, Serialize)]
pub struct AdjustmentDetails {
    pub delta_s: f32,
    pub delta_c: f32,
    pub alpha: f32,
    pub trigger_error: f32,
}

/// Input for get_meta_learning_log tool
#[derive(Debug, Clone, Deserialize)]
pub struct GetMetaLearningLogInput {
    /// Start time (ISO 8601)
    pub start_time: Option<String>,
    /// End time (ISO 8601)
    pub end_time: Option<String>,
    /// Filter by event type
    pub event_type: Option<String>,
    /// Filter by domain
    pub domain: Option<String>,
    /// Maximum events to return
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Offset for pagination
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize { 100 }

/// Output for get_meta_learning_log tool
#[derive(Debug, Clone, Serialize)]
pub struct MetaLearningLogOutput {
    /// Events matching query
    pub events: Vec<MetaLearningEventOutput>,
    /// Total count (before limit/offset)
    pub total_count: usize,
    /// Whether there are more events
    pub has_more: bool,
    /// Query execution time (ms)
    pub query_time_ms: u32,
}

/// Serializable event representation
#[derive(Debug, Clone, Serialize)]
pub struct MetaLearningEventOutput {
    pub timestamp: String,
    pub event_type: String,
    pub prediction_error: f32,
    pub lambda_before: LambdaValues,
    pub lambda_after: LambdaValues,
    pub accuracy_avg: f32,
    pub escalated: bool,
    pub domain: Option<String>,
}

/// Handle get_meta_learning_status tool
///
/// Returns current state of the Meta-UTL self-correction system.
pub async fn handle_get_meta_learning_status(
    input: GetMetaLearningStatusInput,
    service: &MetaLearningService,
) -> McpResult<MetaLearningStatusOutput>;

/// Handle trigger_lambda_recalibration tool
///
/// Manually triggers lambda weight recalibration.
pub async fn handle_trigger_lambda_recalibration(
    input: TriggerRecalibrationInput,
    service: &mut MetaLearningService,
) -> McpResult<RecalibrationOutput>;

/// Handle get_meta_learning_log tool
///
/// Queries the meta-learning event log.
pub async fn handle_get_meta_learning_log(
    input: GetMetaLearningLogInput,
    service: &MetaLearningService,
) -> McpResult<MetaLearningLogOutput>;

/// Parse ISO 8601 timestamp
fn parse_timestamp(s: &str) -> McpResult<DateTime<Utc>>;

/// Parse event type string to enum
fn parse_event_type(s: &str) -> McpResult<MetaLearningEventType>;

/// Parse domain string to enum
fn parse_domain(s: &str) -> McpResult<Domain>;
```

#### File: `crates/context-graph-mcp/src/schema/meta_learning_tools.rs`

```rust
use serde_json::json;

/// Tool schema for get_meta_learning_status
pub fn get_meta_learning_status_schema() -> serde_json::Value {
    json!({
        "name": "get_meta_learning_status",
        "description": "Get current Meta-UTL self-correction status including accuracy, lambda weights, and escalation state",
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_accuracy_history": {
                    "type": "boolean",
                    "description": "Include rolling accuracy history values",
                    "default": false
                },
                "include_embedder_breakdown": {
                    "type": "boolean",
                    "description": "Include per-embedder (E1-E13) accuracy breakdown",
                    "default": false
                }
            },
            "required": []
        }
    })
}

/// Tool schema for trigger_lambda_recalibration
pub fn trigger_lambda_recalibration_schema() -> serde_json::Value {
    json!({
        "name": "trigger_lambda_recalibration",
        "description": "Manually trigger lambda weight recalibration using gradient adjustment or Bayesian optimization",
        "inputSchema": {
            "type": "object",
            "properties": {
                "force_bayesian": {
                    "type": "boolean",
                    "description": "Force Bayesian optimization instead of gradient adjustment",
                    "default": false
                },
                "domain": {
                    "type": "string",
                    "enum": ["code", "medical", "legal", "creative", "research", "general"],
                    "description": "Target domain for calibration (optional)"
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Compute adjustment but don't apply it",
                    "default": false
                }
            },
            "required": []
        }
    })
}

/// Tool schema for get_meta_learning_log
pub fn get_meta_learning_log_schema() -> serde_json::Value {
    json!({
        "name": "get_meta_learning_log",
        "description": "Query meta-learning event log with optional filters",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "string",
                    "description": "Start time (ISO 8601 format, e.g., 2024-01-01T00:00:00Z)"
                },
                "end_time": {
                    "type": "string",
                    "description": "End time (ISO 8601 format)"
                },
                "event_type": {
                    "type": "string",
                    "enum": ["lambda_adjustment", "bayesian_escalation", "accuracy_alert", "self_healing", "human_escalation"],
                    "description": "Filter by event type"
                },
                "domain": {
                    "type": "string",
                    "enum": ["code", "medical", "legal", "creative", "research", "general"],
                    "description": "Filter by domain"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 100,
                    "description": "Maximum events to return"
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": "Number of events to skip (pagination)"
                }
            },
            "required": []
        }
    })
}

/// Register all meta-learning tools
pub fn register_meta_learning_tools(registry: &mut ToolRegistry) {
    registry.register(get_meta_learning_status_schema());
    registry.register(trigger_lambda_recalibration_schema());
    registry.register(get_meta_learning_log_schema());
}
```

### 6.2 Service Interface

#### File: `crates/context-graph-utl/src/meta/service.rs`

```rust
use super::{
    AdaptiveLambdaWeights, EmbedderAccuracyTracker, EscalationManager,
    MetaLearningEventLog, SelfCorrectionConfig, SelfCorrectionState,
    MetaLearningEvent, LambdaAdjustment, Domain,
};
use crate::error::UtlResult;
use crate::lifecycle::LifecycleLambdaWeights;

/// Meta-learning service facade
///
/// Provides unified access to all meta-UTL self-correction components.
/// This is the primary interface for MCP handlers.
#[derive(Debug)]
pub struct MetaLearningService {
    /// Adaptive lambda weights
    adaptive_weights: AdaptiveLambdaWeights,
    /// Accuracy tracker
    accuracy_tracker: EmbedderAccuracyTracker,
    /// Escalation manager
    escalation_manager: EscalationManager,
    /// Event log
    event_log: MetaLearningEventLog,
    /// Configuration
    config: SelfCorrectionConfig,
    /// Enabled flag
    enabled: bool,
}

impl MetaLearningService {
    /// Create new service with configuration
    pub fn new(config: SelfCorrectionConfig) -> Self;

    /// Create with default configuration
    pub fn with_defaults() -> Self;

    /// Initialize from lifecycle weights
    pub fn from_lifecycle(base_weights: LifecycleLambdaWeights, config: SelfCorrectionConfig) -> Self;

    /// Check if service is enabled
    pub fn is_enabled(&self) -> bool;

    /// Enable/disable service
    pub fn set_enabled(&mut self, enabled: bool);

    /// Get current state snapshot
    pub fn get_state(&self) -> SelfCorrectionState;

    /// Get current corrected lambda weights
    pub fn current_lambdas(&self) -> LifecycleLambdaWeights;

    /// Get base lifecycle lambda weights
    pub fn base_lambdas(&self) -> LifecycleLambdaWeights;

    /// Get current global accuracy
    pub fn current_accuracy(&self) -> f32;

    /// Get per-embedder accuracy array
    pub fn embedder_accuracies(&self) -> [f32; 13];

    /// Get accuracy history
    pub fn accuracy_history(&self) -> Vec<f32>;

    /// Get consecutive low count
    pub fn consecutive_low_count(&self) -> u32;

    /// Get escalation status
    pub fn escalation_status(&self) -> EscalationStatus;

    /// Get adjustment count
    pub fn adjustment_count(&self) -> u64;

    /// Get last adjustment
    pub fn last_adjustment(&self) -> Option<&LambdaAdjustment>;

    /// Get recent events
    pub fn recent_events(&self, hours: u32) -> Vec<&MetaLearningEvent>;

    /// Record a prediction result
    ///
    /// Called by UtlProcessor after computing learning signal.
    pub fn record_prediction(
        &mut self,
        embedder_idx: usize,
        predicted: f32,
        actual: f32,
        domain: Option<Domain>,
        ach_level: f32,
    ) -> UtlResult<Option<LambdaAdjustment>>;

    /// Manually trigger recalibration
    ///
    /// # Arguments
    /// - `force_bayesian`: Skip gradient check and use Bayesian optimization
    /// - `dry_run`: Compute adjustment but don't apply
    ///
    /// # Returns
    /// Recalibration result including adjustment and new weights
    pub fn trigger_recalibration(
        &mut self,
        force_bayesian: bool,
        dry_run: bool,
    ) -> UtlResult<RecalibrationResult>;

    /// Query event log
    pub fn query_events(&self, query: &EventLogQuery) -> Vec<&MetaLearningEvent>;

    /// Get event log statistics
    pub fn event_stats(&self) -> EventLogStats;

    /// Reset to base weights
    pub fn reset_to_base(&mut self);

    /// Clear event log
    pub fn clear_events(&mut self);

    /// Export state for persistence
    pub fn export_state(&self) -> UtlResult<String>;

    /// Import state from persistence
    pub fn import_state(&mut self, json: &str) -> UtlResult<()>;
}

/// Result of a recalibration attempt
#[derive(Debug, Clone)]
pub struct RecalibrationResult {
    pub success: bool,
    pub adjustment: Option<LambdaAdjustment>,
    pub new_weights: LifecycleLambdaWeights,
    pub previous_weights: LifecycleLambdaWeights,
    pub method: RecalibrationMethod,
    pub bo_iterations: Option<usize>,
    pub expected_improvement: Option<f32>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecalibrationMethod {
    Gradient,
    Bayesian,
    None,
}

impl Default for MetaLearningService {
    fn default() -> Self;
}
```

### 6.3 Constraints

- Handlers MUST validate all input parameters
- Timestamps MUST be parsed with lenient ISO 8601 handling
- Event queries MUST support pagination correctly
- Dry run MUST NOT modify state
- All outputs MUST be JSON-serializable
- Handler errors MUST map to McpError with appropriate codes
- Service MUST be thread-safe (Arc<RwLock<MetaLearningService>>)

### 6.4 Verification Commands

```bash
# Type check
cargo check -p context-graph-mcp

# Unit tests
cargo test -p context-graph-mcp handlers::meta_learning

# Schema validation tests
cargo test -p context-graph-mcp schema::meta_learning_tools

# Integration tests
cargo test -p context-graph-mcp --test meta_learning_integration

# Clippy
cargo clippy -p context-graph-mcp -- -D warnings
```

---

## 7. Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-mcp/src/handlers/meta_learning.rs` | Handler implementations |
| `crates/context-graph-mcp/src/schema/meta_learning_tools.rs` | Tool schema definitions |
| `crates/context-graph-utl/src/meta/service.rs` | Service facade |
| `crates/context-graph-mcp/tests/meta_learning_integration.rs` | Integration tests |

---

## 8. Files to Modify

| Path | Modification |
|------|--------------|
| `crates/context-graph-mcp/src/handlers/mod.rs` | Add `pub mod meta_learning;` |
| `crates/context-graph-mcp/src/schema/mod.rs` | Add `pub mod meta_learning_tools;` |
| `crates/context-graph-mcp/src/router.rs` | Register meta-learning tools |
| `crates/context-graph-utl/src/meta/mod.rs` | Add `pub mod service;` |
| `crates/context-graph-utl/src/lib.rs` | Re-export `MetaLearningService` |

---

## 9. Pseudo-Code

### 9.1 handle_get_meta_learning_status

```
FUNCTION handle_get_meta_learning_status(input, service) -> McpResult<Output>:
    // Get state snapshot
    LET state = service.get_state()

    // Build response
    LET output = MetaLearningStatusOutput {
        enabled: state.enabled,
        current_accuracy: state.accuracy_tracker.global_average(),
        consecutive_low_count: state.accuracy_tracker.global_history.consecutive_low_count(),
        current_lambdas: LambdaValues::from(state.current_lambdas),
        base_lambdas: LambdaValues::from(state.base_lambdas),
        lambda_deviation: compute_deviation(state.current_lambdas, state.base_lambdas),
        escalation_status: state.escalation_status.to_string(),
        adjustment_count: state.adjustment_count,
        recent_events_count: service.recent_events(24).len(),
        last_adjustment_at: state.last_adjustment.map(|a| a.timestamp),
        accuracy_history: None,
        embedder_accuracy: None,
    }

    // Add optional fields
    IF input.include_accuracy_history:
        output.accuracy_history = Some(service.accuracy_history())

    IF input.include_embedder_breakdown:
        output.embedder_accuracy = Some(service.embedder_accuracies().to_vec())

    RETURN Ok(output)
```

### 9.2 handle_trigger_lambda_recalibration

```
FUNCTION handle_trigger_lambda_recalibration(input, service) -> McpResult<Output>:
    // Validate domain if provided
    LET domain = IF input.domain.is_some():
        Some(parse_domain(&input.domain.unwrap())?)
    ELSE:
        None

    // Get previous lambdas for comparison
    LET previous = service.current_lambdas()

    // Trigger recalibration
    LET result = service.trigger_recalibration(input.force_bayesian, input.dry_run)?

    // Build response
    LET output = RecalibrationOutput {
        success: result.success,
        adjustment: result.adjustment.map(|a| AdjustmentDetails {
            delta_s: a.delta_lambda_s,
            delta_c: a.delta_lambda_c,
            alpha: a.alpha,
            trigger_error: a.trigger_error,
        }),
        new_lambdas: LambdaValues::from(result.new_weights),
        previous_lambdas: LambdaValues::from(previous),
        method: result.method.to_string(),
        bo_iterations: result.bo_iterations,
        expected_improvement: result.expected_improvement,
        dry_run: input.dry_run,
        error: result.error,
    }

    RETURN Ok(output)
```

### 9.3 handle_get_meta_learning_log

```
FUNCTION handle_get_meta_learning_log(input, service) -> McpResult<Output>:
    LET start_time = chrono::Instant::now()

    // Build query
    LET mut query = EventLogQuery::new()

    IF let Some(start) = input.start_time:
        query = query.time_range(parse_timestamp(&start)?, ...)
    IF let Some(end) = input.end_time:
        query = query.time_range(..., parse_timestamp(&end)?)
    IF let Some(event_type) = input.event_type:
        query = query.event_type(parse_event_type(&event_type)?)
    IF let Some(domain) = input.domain:
        query = query.domain(parse_domain(&domain)?)

    query = query.limit(input.limit).offset(input.offset)

    // Execute query
    LET events = service.query_events(&query)

    // Get total count (without limit/offset)
    LET total_query = EventLogQuery { ...query, limit: None, offset: None }
    LET total_count = service.query_events(&total_query).len()

    // Convert to output format
    LET event_outputs = events.iter().map(|e| MetaLearningEventOutput {
        timestamp: e.timestamp.to_rfc3339(),
        event_type: e.event_type.to_string(),
        prediction_error: e.prediction_error,
        lambda_before: LambdaValues { lambda_s: e.lambda_before.0, lambda_c: e.lambda_before.1 },
        lambda_after: LambdaValues { lambda_s: e.lambda_after.0, lambda_c: e.lambda_after.1 },
        accuracy_avg: e.accuracy_avg,
        escalated: e.escalated,
        domain: e.domain.map(|d| d.to_string()),
    }).collect()

    LET query_time_ms = start_time.elapsed().as_millis() as u32

    RETURN Ok(MetaLearningLogOutput {
        events: event_outputs,
        total_count,
        has_more: input.offset + events.len() < total_count,
        query_time_ms,
    })
```

---

## 10. Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| Status tool returns valid JSON | Call and parse response |
| Recalibration respects dry_run | Call with dry_run=true, verify no state change |
| Log query pagination works | Query with offset/limit, verify results |
| Timestamp parsing handles formats | Test ISO 8601 variations |
| Invalid input returns error | Test with bad enum values |
| Thread safety | Concurrent handler calls |
| Performance < 100ms | Benchmark tool calls |

---

## 11. Test Cases

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_status_basic() {
        let service = MetaLearningService::with_defaults();
        let input = GetMetaLearningStatusInput::default();

        let result = handle_get_meta_learning_status(input, &service).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.enabled);
        assert!(output.current_accuracy >= 0.0 && output.current_accuracy <= 1.0);
    }

    #[tokio::test]
    async fn test_get_status_with_history() {
        let mut service = MetaLearningService::with_defaults();
        // Record some predictions
        for i in 0..10 {
            service.record_prediction(0, 0.8, 0.75, None, 0.001).unwrap();
        }

        let input = GetMetaLearningStatusInput {
            include_accuracy_history: true,
            include_embedder_breakdown: true,
        };

        let result = handle_get_meta_learning_status(input, &service).await.unwrap();

        assert!(result.accuracy_history.is_some());
        assert!(result.embedder_accuracy.is_some());
        assert_eq!(result.embedder_accuracy.unwrap().len(), 13);
    }

    #[tokio::test]
    async fn test_recalibration_dry_run() {
        let mut service = MetaLearningService::with_defaults();
        let original = service.current_lambdas();

        let input = TriggerRecalibrationInput {
            force_bayesian: false,
            domain: None,
            dry_run: true,
        };

        let result = handle_trigger_lambda_recalibration(input, &mut service).await.unwrap();

        assert!(result.dry_run);
        // Verify no state change
        assert_eq!(service.current_lambdas(), original);
    }

    #[tokio::test]
    async fn test_recalibration_gradient() {
        let mut service = MetaLearningService::with_defaults();

        // Record predictions to create error > 0.2
        for _ in 0..10 {
            service.record_prediction(0, 0.8, 0.5, None, 0.001).unwrap();
        }

        let input = TriggerRecalibrationInput {
            force_bayesian: false,
            domain: None,
            dry_run: false,
        };

        let result = handle_trigger_lambda_recalibration(input, &mut service).await.unwrap();

        assert!(result.success || result.method == "none");
    }

    #[tokio::test]
    async fn test_log_query_basic() {
        let mut service = MetaLearningService::with_defaults();

        // Record some events
        for _ in 0..5 {
            service.record_prediction(0, 0.8, 0.5, Some(Domain::Code), 0.001).unwrap();
        }

        let input = GetMetaLearningLogInput {
            start_time: None,
            end_time: None,
            event_type: None,
            domain: None,
            limit: 10,
            offset: 0,
        };

        let result = handle_get_meta_learning_log(input, &service).await.unwrap();

        assert!(result.events.len() <= 10);
        assert_eq!(result.total_count, result.events.len());
    }

    #[tokio::test]
    async fn test_log_query_with_filters() {
        let mut service = MetaLearningService::with_defaults();

        // Record events with different domains
        service.record_prediction(0, 0.8, 0.5, Some(Domain::Code), 0.001).unwrap();
        service.record_prediction(1, 0.8, 0.5, Some(Domain::Medical), 0.001).unwrap();

        let input = GetMetaLearningLogInput {
            start_time: None,
            end_time: None,
            event_type: Some("lambda_adjustment".to_string()),
            domain: Some("code".to_string()),
            limit: 100,
            offset: 0,
        };

        let result = handle_get_meta_learning_log(input, &service).await.unwrap();

        for event in &result.events {
            assert_eq!(event.domain, Some("code".to_string()));
        }
    }

    #[tokio::test]
    async fn test_log_query_pagination() {
        let mut service = MetaLearningService::with_defaults();

        // Record many events
        for i in 0..50 {
            service.record_prediction(i % 13, 0.8, 0.5, None, 0.001).unwrap();
        }

        let input = GetMetaLearningLogInput {
            start_time: None,
            end_time: None,
            event_type: None,
            domain: None,
            limit: 10,
            offset: 0,
        };

        let result = handle_get_meta_learning_log(input, &service).await.unwrap();

        assert_eq!(result.events.len(), 10);
        assert!(result.has_more);
        assert!(result.total_count > 10);
    }

    #[tokio::test]
    async fn test_parse_timestamp_valid() {
        let result = parse_timestamp("2024-01-15T10:30:00Z");
        assert!(result.is_ok());

        let result = parse_timestamp("2024-01-15T10:30:00+00:00");
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_parse_timestamp_invalid() {
        let result = parse_timestamp("not-a-timestamp");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_parse_event_type_valid() {
        assert!(parse_event_type("lambda_adjustment").is_ok());
        assert!(parse_event_type("bayesian_escalation").is_ok());
        assert!(parse_event_type("accuracy_alert").is_ok());
    }

    #[tokio::test]
    async fn test_parse_event_type_invalid() {
        let result = parse_event_type("invalid_type");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_schema_validation() {
        let status_schema = get_meta_learning_status_schema();
        assert!(status_schema.get("name").is_some());
        assert!(status_schema.get("inputSchema").is_some());

        let recal_schema = trigger_lambda_recalibration_schema();
        assert!(recal_schema.get("name").is_some());

        let log_schema = get_meta_learning_log_schema();
        assert!(log_schema.get("name").is_some());
    }
}
```

---

## 12. Rollback Plan

If this task fails validation:

1. Revert handler files
2. Revert schema files
3. Remove mod declarations
4. Previous meta components remain unaffected
5. Document failure in task notes

---

## 13. Notes

- Handlers follow existing MCP pattern in codebase
- Service facade simplifies handler implementation
- Thread safety via Arc<RwLock<>> in actual usage
- Pagination uses offset/limit (not cursor) for simplicity
- Timestamp parsing uses chrono's flexible parser
- Dry run enables testing without side effects

---

**Task History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | ContextGraph Team | Initial task specification |
