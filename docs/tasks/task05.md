# Task 05: Create MCP Request/Response DTOs for Topic and Curation Tools

## Metadata
- **Task ID**: TASK-GAP-005
- **Phase**: 2 (MCP Infrastructure)
- **Priority**: High
- **Complexity**: Medium
- **Dependencies**: task04 (tool names defined in `names.rs` - VERIFIED COMPLETE)
- **Spec Reference**: TECH_SPEC_PRD_GAPS.md Section 8

---

## CRITICAL: Read Before Starting

### Current Project State (Verified 2026-01-18)

**EXISTING FILES - DO NOT CREATE DUPLICATES:**
```
crates/context-graph-mcp/src/handlers/tools/
├── mod.rs              # Exports 5 submodules - NEEDS MODIFICATION
├── dispatch.rs         # Routes 6 tools - NEEDS MODIFICATION
├── memory_tools.rs     # inject_context, store_memory, search_graph
├── status_tools.rs     # get_memetic_status
├── consolidation.rs    # trigger_consolidation
└── helpers.rs          # Tool result builders
```

**FILES TO CREATE (DO NOT EXIST YET):**
```
crates/context-graph-mcp/src/handlers/tools/
├── topic_dtos.rs       # NEW - DTOs for 4 topic tools
├── curation_dtos.rs    # NEW - DTOs for 2 curation tools
├── topic_tools.rs      # NEW - Handler implementations (Task 06)
└── curation_tools.rs   # NEW - Handler implementations (Task 06)
```

**TOOL NAMES ALREADY DEFINED** (in `crates/context-graph-mcp/src/tools/names.rs`):
```rust
// These exist with #[allow(dead_code)] - will be used after this task
pub const GET_TOPIC_PORTFOLIO: &str = "get_topic_portfolio";
pub const GET_TOPIC_STABILITY: &str = "get_topic_stability";
pub const DETECT_TOPICS: &str = "detect_topics";
pub const GET_DIVERGENCE_ALERTS: &str = "get_divergence_alerts";
pub const FORGET_CONCEPT: &str = "forget_concept";
pub const BOOST_IMPORTANCE: &str = "boost_importance";
```

**EXISTING CORE TYPES** (in `context-graph-core/src/clustering/`):
- `Topic` - Full topic struct with id, name, profile, contributing_spaces, member_memories, confidence, stability
- `TopicProfile` - 13-element strength array with weighted_agreement() method
- `TopicPhase` - Enum: Emerging, Stable, Declining, Merging
- `TopicStability` - Per-topic metrics: phase, age_hours, membership_churn, centroid_drift
- `TopicStabilityTracker` - Portfolio-level tracking with snapshots, churn history, dream triggers

---

## Objective

Create **type-safe DTOs** for the 6 new MCP tools. These DTOs:
1. Serialize/deserialize tool parameters and results
2. Use serde derive macros following existing patterns
3. Match the JSON schemas defined in TECH_SPEC_PRD_GAPS.md Section 9
4. Will be used by handler implementations in Task 06

---

## Input Context Files (READ BEFORE CODING)

| File | Why Read It | Key Information |
|------|-------------|-----------------|
| `/home/cabdru/contextgraph/docs/TECH_SPEC_PRD_GAPS.md` | DTO definitions | Section 8: Complete DTO specs |
| `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/consolidation.rs` | Pattern reference | How TriggerConsolidationParams is structured |
| `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/merge.rs` | Pattern reference | MergeConceptsInput/Output structure |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/clustering/topic.rs` | Core types | Topic, TopicProfile, TopicPhase, TopicStability |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/clustering/stability.rs` | Core types | TopicStabilityTracker |

---

## Implementation Steps

### Step 1: Create topic_dtos.rs

**Path**: `crates/context-graph-mcp/src/handlers/tools/topic_dtos.rs`

```rust
//! DTOs for topic-related MCP tools.
//!
//! Per PRD v6 Section 10.2, these DTOs support:
//! - get_topic_portfolio: Retrieve all discovered topics with profiles
//! - get_topic_stability: Get portfolio-level stability metrics
//! - detect_topics: Force topic detection recalculation
//! - get_divergence_alerts: Check for divergence from recent activity
//!
//! Constitution References:
//! - ARCH-09: Topic threshold is weighted_agreement >= 2.5
//! - AP-60: Temporal embedders (E2-E4) NEVER count toward topic detection
//! - AP-62: Divergence alerts use SEMANTIC embedders only (E1, E5, E6, E7, E10, E12, E13)
//! - AP-70: Dream triggers when entropy > 0.7 AND churn > 0.5

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// REQUEST DTOs
// ============================================================================

/// Request parameters for get_topic_portfolio tool.
///
/// # Example JSON
/// ```json
/// {"format": "standard"}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct GetTopicPortfolioRequest {
    /// Output format: "brief", "standard", or "verbose"
    /// - brief: Topic names and confidence only
    /// - standard: Includes contributing spaces and member counts
    /// - verbose: Full topic profiles with all 13 strengths
    #[serde(default = "default_format")]
    pub format: String,
}

fn default_format() -> String {
    "standard".to_string()
}

/// Request parameters for get_topic_stability tool.
///
/// # Example JSON
/// ```json
/// {"hours": 6}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct GetTopicStabilityRequest {
    /// Lookback period in hours for computing averages (default 6, max 168)
    #[serde(default = "default_hours")]
    pub hours: u32,
}

fn default_hours() -> u32 {
    6
}

/// Request parameters for detect_topics tool.
///
/// # Example JSON
/// ```json
/// {"force": false}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct DetectTopicsRequest {
    /// Force detection even if recently computed
    #[serde(default)]
    pub force: bool,
}

/// Request parameters for get_divergence_alerts tool.
///
/// # Example JSON
/// ```json
/// {"lookback_hours": 2}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct GetDivergenceAlertsRequest {
    /// Hours to look back for recent activity comparison (default 2, max 48)
    #[serde(default = "default_lookback")]
    pub lookback_hours: u32,
}

fn default_lookback() -> u32 {
    2
}

// ============================================================================
// RESPONSE DTOs
// ============================================================================

/// Response for get_topic_portfolio tool.
#[derive(Debug, Clone, Serialize)]
pub struct TopicPortfolioResponse {
    /// List of discovered topics
    pub topics: Vec<TopicSummary>,

    /// Portfolio-level stability metrics
    pub stability: StabilityMetricsSummary,

    /// Total number of topics
    pub total_topics: usize,

    /// Current progressive tier (0-6) based on memory count
    /// - Tier 0: 0 memories
    /// - Tier 1: 1-2 memories
    /// - Tier 2: 3-9 memories (basic clustering)
    /// - Tier 3: 10-29 memories (divergence detection)
    /// - Tier 4: 30-99 memories (reliable statistics)
    /// - Tier 5: 100-499 memories (sub-clustering)
    /// - Tier 6: 500+ memories (full personalization)
    pub tier: u8,
}

/// Summary of a single topic for API response.
///
/// This is a serializable view of the core Topic type,
/// suitable for JSON transmission.
#[derive(Debug, Clone, Serialize)]
pub struct TopicSummary {
    /// Topic UUID
    pub id: Uuid,

    /// Optional human-readable name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Confidence score = weighted_agreement / 8.5 (range 0.0-1.0)
    pub confidence: f32,

    /// Weighted agreement score per ARCH-09
    /// - Threshold for topic: >= 2.5
    /// - Max possible: 8.5 (7 semantic + 2 relational*0.5 + 1 structural*0.5)
    pub weighted_agreement: f32,

    /// Number of memories belonging to this topic
    pub member_count: usize,

    /// Contributing embedding spaces (those with strength > 0.5)
    /// Example: ["Semantic", "Causal", "Code"]
    pub contributing_spaces: Vec<String>,

    /// Current lifecycle phase
    pub phase: String,
}

/// Portfolio-level stability metrics summary.
#[derive(Debug, Clone, Serialize)]
pub struct StabilityMetricsSummary {
    /// Churn rate [0.0-1.0] where 0.0=stable, 1.0=complete turnover
    /// Computed as |symmetric_difference| / |union| of topic IDs over time
    pub churn_rate: f32,

    /// Topic distribution entropy [0.0-1.0]
    pub entropy: f32,

    /// Whether portfolio is stable (churn < 0.3 per constitution)
    pub is_stable: bool,
}

/// Response for get_topic_stability tool.
#[derive(Debug, Clone, Serialize)]
pub struct TopicStabilityResponse {
    /// Current churn rate
    pub churn_rate: f32,

    /// Current entropy
    pub entropy: f32,

    /// Breakdown by lifecycle phase
    pub phases: PhaseBreakdown,

    /// Whether dream consolidation is recommended
    /// Per AP-70: entropy > 0.7 AND churn > 0.5
    pub dream_recommended: bool,

    /// Warning flag for high churn (churn >= 0.5)
    pub high_churn_warning: bool,

    /// Average churn over the requested lookback period
    pub average_churn: f32,
}

/// Count of topics in each lifecycle phase.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseBreakdown {
    /// Topics < 1hr old with high churn
    pub emerging: u32,

    /// Topics >= 24hr old with churn < 0.1
    pub stable: u32,

    /// Topics with churn >= 0.5
    pub declining: u32,

    /// Topics being absorbed into others
    pub merging: u32,
}

/// Response for detect_topics tool.
#[derive(Debug, Clone, Serialize)]
pub struct DetectTopicsResponse {
    /// Newly discovered topics from this detection run
    pub new_topics: Vec<TopicSummary>,

    /// Topics that were merged during detection
    pub merged_topics: Vec<MergedTopicInfo>,

    /// Total topic count after detection
    pub total_after: usize,

    /// Human-readable message about what happened
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Information about a topic merge operation.
#[derive(Debug, Clone, Serialize)]
pub struct MergedTopicInfo {
    /// ID of the topic that was absorbed
    pub absorbed_id: Uuid,

    /// ID of the topic it was merged into
    pub into_id: Uuid,
}

/// Response for get_divergence_alerts tool.
///
/// CRITICAL: Per AP-62, ONLY SEMANTIC embedders trigger alerts:
/// E1 (Semantic), E5 (Causal), E6 (Sparse), E7 (Code),
/// E10 (Multimodal), E12 (LateInteraction), E13 (SPLADE)
///
/// Temporal embedders (E2-E4) NEVER trigger divergence per AP-63.
#[derive(Debug, Clone, Serialize)]
pub struct DivergenceAlertsResponse {
    /// List of divergence alerts from SEMANTIC spaces only
    pub alerts: Vec<DivergenceAlert>,

    /// Overall severity: "none", "low", "medium", "high"
    pub severity: String,
}

/// A single divergence alert from a semantic embedding space.
#[derive(Debug, Clone, Serialize)]
pub struct DivergenceAlert {
    /// The semantic space that detected divergence
    /// One of: "E1_Semantic", "E5_Causal", "E6_Sparse", "E7_Code",
    /// "E10_Multimodal", "E12_LateInteraction", "E13_SPLADE"
    pub semantic_space: String,

    /// Similarity score between current activity and recent memory
    pub similarity_score: f32,

    /// Brief summary of the recent memory for context
    pub recent_memory_summary: String,

    /// The threshold that was crossed to trigger this alert
    /// Per constitution divergence_detection thresholds
    pub threshold: f32,
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_topic_portfolio_request_defaults() {
        let json = "{}";
        let req: GetTopicPortfolioRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.format, "standard");
    }

    #[test]
    fn test_get_topic_portfolio_request_custom_format() {
        let json = r#"{"format": "verbose"}"#;
        let req: GetTopicPortfolioRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.format, "verbose");
    }

    #[test]
    fn test_get_topic_stability_request_defaults() {
        let json = "{}";
        let req: GetTopicStabilityRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.hours, 6);
    }

    #[test]
    fn test_detect_topics_request_defaults() {
        let json = "{}";
        let req: DetectTopicsRequest = serde_json::from_str(json).unwrap();
        assert!(!req.force);
    }

    #[test]
    fn test_get_divergence_alerts_request_defaults() {
        let json = "{}";
        let req: GetDivergenceAlertsRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.lookback_hours, 2);
    }

    #[test]
    fn test_topic_summary_serialization() {
        let summary = TopicSummary {
            id: Uuid::nil(),
            name: Some("Test Topic".to_string()),
            confidence: 0.35,
            weighted_agreement: 3.0,
            member_count: 15,
            contributing_spaces: vec!["Semantic".to_string(), "Causal".to_string()],
            phase: "Stable".to_string(),
        };

        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("\"confidence\":0.35"));
        assert!(json.contains("\"weighted_agreement\":3.0"));
        assert!(json.contains("\"phase\":\"Stable\""));
    }

    #[test]
    fn test_topic_summary_no_name_skipped() {
        let summary = TopicSummary {
            id: Uuid::nil(),
            name: None,
            confidence: 0.35,
            weighted_agreement: 3.0,
            member_count: 15,
            contributing_spaces: vec![],
            phase: "Emerging".to_string(),
        };

        let json = serde_json::to_string(&summary).unwrap();
        assert!(!json.contains("\"name\""));
    }

    #[test]
    fn test_divergence_alert_serialization() {
        let alert = DivergenceAlert {
            semantic_space: "E1_Semantic".to_string(),
            similarity_score: 0.22,
            recent_memory_summary: "Working on auth...".to_string(),
            threshold: 0.30,
        };

        let json = serde_json::to_string(&alert).unwrap();
        assert!(json.contains("\"semantic_space\":\"E1_Semantic\""));
        assert!(json.contains("\"threshold\":0.3"));
    }

    #[test]
    fn test_phase_breakdown_serialization() {
        let phases = PhaseBreakdown {
            emerging: 2,
            stable: 8,
            declining: 1,
            merging: 0,
        };

        let json = serde_json::to_string(&phases).unwrap();
        assert!(json.contains("\"emerging\":2"));
        assert!(json.contains("\"stable\":8"));
    }
}
```

### Step 2: Create curation_dtos.rs

**Path**: `crates/context-graph-mcp/src/handlers/tools/curation_dtos.rs`

```rust
//! DTOs for curation-related MCP tools.
//!
//! Per PRD v6 Section 10.3, these DTOs support:
//! - forget_concept: Soft-delete a memory with 30-day recovery
//! - boost_importance: Adjust memory importance score
//!
//! Constitution References:
//! - SEC-06: Soft delete 30-day recovery
//! - BR-MCP-001: forget_concept uses soft delete by default
//! - BR-MCP-002: boost_importance clamps final value to [0.0, 1.0]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// REQUEST DTOs
// ============================================================================

/// Request parameters for forget_concept tool.
///
/// # Example JSON
/// ```json
/// {"node_id": "550e8400-e29b-41d4-a716-446655440000", "soft_delete": true}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ForgetConceptRequest {
    /// UUID of the memory to forget (required)
    /// Must be a valid UUID string
    pub node_id: String,

    /// Use soft delete with 30-day recovery window (default true per SEC-06)
    /// If false, memory is permanently deleted with no recovery option
    #[serde(default = "default_soft_delete")]
    pub soft_delete: bool,
}

fn default_soft_delete() -> bool {
    true
}

/// Request parameters for boost_importance tool.
///
/// # Example JSON
/// ```json
/// {"node_id": "550e8400-e29b-41d4-a716-446655440000", "delta": 0.2}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct BoostImportanceRequest {
    /// UUID of the memory to modify (required)
    pub node_id: String,

    /// Importance adjustment (-1.0 to 1.0)
    /// - Positive values increase importance
    /// - Negative values decrease importance
    /// - Final value is clamped to [0.0, 1.0] per BR-MCP-002
    pub delta: f32,
}

// ============================================================================
// RESPONSE DTOs
// ============================================================================

/// Response for forget_concept tool.
#[derive(Debug, Clone, Serialize)]
pub struct ForgetConceptResponse {
    /// UUID of the forgotten memory
    pub forgotten_id: Uuid,

    /// Whether soft delete was used
    pub soft_deleted: bool,

    /// When the memory can be recovered until (if soft deleted)
    /// Per SEC-06: 30 days from deletion
    /// Only present if soft_deleted is true
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recoverable_until: Option<DateTime<Utc>>,
}

/// Response for boost_importance tool.
#[derive(Debug, Clone, Serialize)]
pub struct BoostImportanceResponse {
    /// UUID of the modified memory
    pub node_id: Uuid,

    /// Importance value before modification
    pub old_importance: f32,

    /// Importance value after modification (clamped to [0.0, 1.0])
    pub new_importance: f32,

    /// Whether the final value was clamped
    /// True if (old + delta) was outside [0.0, 1.0]
    pub clamped: bool,
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_forget_concept_request_defaults() {
        let json = r#"{"node_id": "550e8400-e29b-41d4-a716-446655440000"}"#;
        let req: ForgetConceptRequest = serde_json::from_str(json).unwrap();

        assert_eq!(req.node_id, "550e8400-e29b-41d4-a716-446655440000");
        assert!(req.soft_delete, "soft_delete should default to true per SEC-06");
    }

    #[test]
    fn test_forget_concept_request_hard_delete() {
        let json = r#"{"node_id": "550e8400-e29b-41d4-a716-446655440000", "soft_delete": false}"#;
        let req: ForgetConceptRequest = serde_json::from_str(json).unwrap();

        assert!(!req.soft_delete);
    }

    #[test]
    fn test_boost_importance_request() {
        let json = r#"{"node_id": "550e8400-e29b-41d4-a716-446655440000", "delta": 0.3}"#;
        let req: BoostImportanceRequest = serde_json::from_str(json).unwrap();

        assert_eq!(req.node_id, "550e8400-e29b-41d4-a716-446655440000");
        assert!((req.delta - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_boost_importance_negative_delta() {
        let json = r#"{"node_id": "test-id", "delta": -0.5}"#;
        let req: BoostImportanceRequest = serde_json::from_str(json).unwrap();

        assert!((req.delta - (-0.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_forget_concept_response_serialization_soft() {
        let recovery_time = Utc::now() + Duration::days(30);
        let response = ForgetConceptResponse {
            forgotten_id: Uuid::nil(),
            soft_deleted: true,
            recoverable_until: Some(recovery_time),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"soft_deleted\":true"));
        assert!(json.contains("\"recoverable_until\""));
    }

    #[test]
    fn test_forget_concept_response_serialization_hard() {
        let response = ForgetConceptResponse {
            forgotten_id: Uuid::nil(),
            soft_deleted: false,
            recoverable_until: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"soft_deleted\":false"));
        assert!(!json.contains("\"recoverable_until\""), "recoverable_until should be skipped when None");
    }

    #[test]
    fn test_boost_importance_response_serialization() {
        let response = BoostImportanceResponse {
            node_id: Uuid::nil(),
            old_importance: 0.5,
            new_importance: 0.7,
            clamped: false,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"old_importance\":0.5"));
        assert!(json.contains("\"new_importance\":0.7"));
        assert!(json.contains("\"clamped\":false"));
    }

    #[test]
    fn test_boost_importance_response_clamped() {
        let response = BoostImportanceResponse {
            node_id: Uuid::nil(),
            old_importance: 0.9,
            new_importance: 1.0, // Was clamped from 1.1
            clamped: true,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"clamped\":true"));
    }
}
```

### Step 3: Update mod.rs

**Path**: `crates/context-graph-mcp/src/handlers/tools/mod.rs`

**Current content:**
```rust
//! MCP tool call handlers.
//!
//! PRD v6 Section 10 MCP Tools:
//! - inject_context, store_memory, search_graph (memory_tools.rs)
//! - get_memetic_status (status_tools.rs)
//! - trigger_consolidation (consolidation.rs)
//! - merge_concepts (../merge.rs)

mod consolidation;
mod dispatch;
mod helpers;
mod memory_tools;
mod status_tools;
```

**Replace with:**
```rust
//! MCP tool call handlers.
//!
//! PRD v6 Section 10 MCP Tools:
//! - inject_context, store_memory, search_graph (memory_tools.rs)
//! - get_memetic_status (status_tools.rs)
//! - trigger_consolidation (consolidation.rs)
//! - merge_concepts (../merge.rs)
//! - get_topic_portfolio, get_topic_stability, detect_topics, get_divergence_alerts (topic_tools.rs)
//! - forget_concept, boost_importance (curation_tools.rs)

mod consolidation;
mod dispatch;
mod helpers;
mod memory_tools;
mod status_tools;

// DTOs for PRD v6 gap tools (TASK-GAP-005)
pub mod curation_dtos;
pub mod topic_dtos;
```

---

## Definition of Done

### File Existence
- [ ] `crates/context-graph-mcp/src/handlers/tools/topic_dtos.rs` exists
- [ ] `crates/context-graph-mcp/src/handlers/tools/curation_dtos.rs` exists

### DTO Counts
- [ ] topic_dtos.rs contains 4 request DTOs: `GetTopicPortfolioRequest`, `GetTopicStabilityRequest`, `DetectTopicsRequest`, `GetDivergenceAlertsRequest`
- [ ] topic_dtos.rs contains 8+ response/helper structs
- [ ] curation_dtos.rs contains 2 request DTOs: `ForgetConceptRequest`, `BoostImportanceRequest`
- [ ] curation_dtos.rs contains 2 response DTOs: `ForgetConceptResponse`, `BoostImportanceResponse`

### Serde Compliance
- [ ] All request DTOs derive `Deserialize`
- [ ] All response DTOs derive `Serialize`
- [ ] Default values use `#[serde(default)]` or `#[serde(default = "...")]`
- [ ] Optional fields use `#[serde(skip_serializing_if = "Option::is_none")]`

### Module Exports
- [ ] mod.rs exports `pub mod topic_dtos;`
- [ ] mod.rs exports `pub mod curation_dtos;`

### Compilation
- [ ] `cargo check -p context-graph-mcp` passes
- [ ] `cargo clippy -p context-graph-mcp -- -D warnings` passes
- [ ] `cargo test -p context-graph-mcp` passes
- [ ] `cargo doc -p context-graph-mcp --no-deps` generates without errors

---

## Full State Verification

After completing the implementation, perform these verification steps:

### Source of Truth Verification

The source of truth for this task is:
1. **Filesystem**: The DTO files must exist at the specified paths
2. **Rust compiler**: The code must compile without errors
3. **Test runner**: All unit tests must pass

### Verification Commands

```bash
cd /home/cabdru/contextgraph

# 1. Verify files exist (SOURCE OF TRUTH: Filesystem)
echo "=== FILE EXISTENCE CHECK ==="
ls -la crates/context-graph-mcp/src/handlers/tools/topic_dtos.rs
ls -la crates/context-graph-mcp/src/handlers/tools/curation_dtos.rs

# 2. Verify struct counts
echo "=== STRUCT COUNT CHECK ==="
grep -c "^pub struct" crates/context-graph-mcp/src/handlers/tools/topic_dtos.rs
# Expected: 12 (4 request + 8 response/helper)
grep -c "^pub struct" crates/context-graph-mcp/src/handlers/tools/curation_dtos.rs
# Expected: 4 (2 request + 2 response)

# 3. Verify module exports in mod.rs
echo "=== MODULE EXPORT CHECK ==="
grep "pub mod topic_dtos" crates/context-graph-mcp/src/handlers/tools/mod.rs
grep "pub mod curation_dtos" crates/context-graph-mcp/src/handlers/tools/mod.rs

# 4. Verify compilation (SOURCE OF TRUTH: Rust compiler)
echo "=== COMPILATION CHECK ==="
cargo check -p context-graph-mcp 2>&1
echo "Exit code: $?"

# 5. Verify clippy (no warnings)
echo "=== CLIPPY CHECK ==="
cargo clippy -p context-graph-mcp -- -D warnings 2>&1
echo "Exit code: $?"

# 6. Run unit tests (SOURCE OF TRUTH: Test runner)
echo "=== UNIT TEST CHECK ==="
cargo test -p context-graph-mcp topic_dtos 2>&1
cargo test -p context-graph-mcp curation_dtos 2>&1
echo "Exit code: $?"

# 7. Verify documentation builds
echo "=== DOC CHECK ==="
cargo doc -p context-graph-mcp --no-deps 2>&1
echo "Exit code: $?"
```

### Expected Outputs

| Check | Expected Result |
|-------|-----------------|
| topic_dtos.rs exists | File found |
| curation_dtos.rs exists | File found |
| topic_dtos struct count | 12 |
| curation_dtos struct count | 4 |
| Module exports | Both lines found |
| cargo check | Exit code 0 |
| cargo clippy | Exit code 0, no warnings |
| cargo test | All tests pass |
| cargo doc | Exit code 0 |

---

## Manual Testing with Synthetic Data

### Test 1: Request DTO Deserialization

**Input**: JSON strings
**Expected Output**: Parsed Rust structs with correct values

```rust
// Test in cargo test or rustc --test

// Test 1a: GetTopicPortfolioRequest with defaults
let json = "{}";
let req: GetTopicPortfolioRequest = serde_json::from_str(json).unwrap();
assert_eq!(req.format, "standard"); // Default

// Test 1b: GetTopicPortfolioRequest with custom value
let json = r#"{"format": "verbose"}"#;
let req: GetTopicPortfolioRequest = serde_json::from_str(json).unwrap();
assert_eq!(req.format, "verbose");

// Test 1c: ForgetConceptRequest defaults
let json = r#"{"node_id": "test-uuid"}"#;
let req: ForgetConceptRequest = serde_json::from_str(json).unwrap();
assert!(req.soft_delete); // Default true per SEC-06

// Test 1d: BoostImportanceRequest negative delta
let json = r#"{"node_id": "test-uuid", "delta": -0.5}"#;
let req: BoostImportanceRequest = serde_json::from_str(json).unwrap();
assert!((req.delta - (-0.5)).abs() < 0.001);
```

### Test 2: Response DTO Serialization

**Input**: Rust structs
**Expected Output**: JSON strings with correct fields

```rust
// Test 2a: TopicSummary serializes correctly
let summary = TopicSummary {
    id: Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
    name: Some("Auth Topic".to_string()),
    confidence: 0.35,
    weighted_agreement: 3.0,
    member_count: 15,
    contributing_spaces: vec!["Semantic".to_string()],
    phase: "Stable".to_string(),
};
let json = serde_json::to_string(&summary).unwrap();
// Verify JSON contains expected fields
assert!(json.contains("\"confidence\":0.35"));
assert!(json.contains("\"weighted_agreement\":3.0"));

// Test 2b: Optional field skipped when None
let summary_no_name = TopicSummary {
    id: Uuid::nil(),
    name: None,
    confidence: 0.5,
    weighted_agreement: 4.25,
    member_count: 10,
    contributing_spaces: vec![],
    phase: "Emerging".to_string(),
};
let json = serde_json::to_string(&summary_no_name).unwrap();
assert!(!json.contains("\"name\"")); // Skipped

// Test 2c: ForgetConceptResponse with recovery date
let response = ForgetConceptResponse {
    forgotten_id: Uuid::nil(),
    soft_deleted: true,
    recoverable_until: Some(Utc::now()),
};
let json = serde_json::to_string(&response).unwrap();
assert!(json.contains("\"recoverable_until\""));

// Test 2d: ForgetConceptResponse without recovery date
let response = ForgetConceptResponse {
    forgotten_id: Uuid::nil(),
    soft_deleted: false,
    recoverable_until: None,
};
let json = serde_json::to_string(&response).unwrap();
assert!(!json.contains("\"recoverable_until\"")); // Skipped
```

### Test 3: Edge Cases

**Boundary conditions to verify:**

```rust
// Edge 1: Empty format string (should use default)
let json = r#"{"format": ""}"#;
let req: GetTopicPortfolioRequest = serde_json::from_str(json).unwrap();
// Note: Empty string is valid - handler should validate

// Edge 2: Zero hours lookback
let json = r#"{"hours": 0}"#;
let req: GetTopicStabilityRequest = serde_json::from_str(json).unwrap();
assert_eq!(req.hours, 0); // Handler should validate minimum

// Edge 3: Delta at boundaries
let json = r#"{"node_id": "test", "delta": 1.0}"#;
let req: BoostImportanceRequest = serde_json::from_str(json).unwrap();
assert!((req.delta - 1.0).abs() < f32::EPSILON);

let json = r#"{"node_id": "test", "delta": -1.0}"#;
let req: BoostImportanceRequest = serde_json::from_str(json).unwrap();
assert!((req.delta - (-1.0)).abs() < f32::EPSILON);

// Edge 4: Invalid UUID string (handler validates, DTO accepts string)
let json = r#"{"node_id": "not-a-uuid"}"#;
let req: ForgetConceptRequest = serde_json::from_str(json).unwrap();
assert_eq!(req.node_id, "not-a-uuid"); // Handler validates UUID format
```

---

## Evidence of Success Log Template

After completing all verification, record results:

```
=== TASK 05 COMPLETION LOG ===
Date: [YYYY-MM-DD HH:MM]
Agent: [Agent ID]

FILE EXISTENCE:
- topic_dtos.rs: [EXISTS/MISSING]
- curation_dtos.rs: [EXISTS/MISSING]

STRUCT COUNTS:
- topic_dtos.rs structs: [N] (expected: 12)
- curation_dtos.rs structs: [N] (expected: 4)

MODULE EXPORTS:
- topic_dtos exported: [YES/NO]
- curation_dtos exported: [YES/NO]

COMPILATION:
- cargo check exit code: [0/N]
- cargo clippy exit code: [0/N]
- cargo clippy warnings: [0/N]

TESTS:
- topic_dtos tests: [PASS/FAIL] ([N] passed)
- curation_dtos tests: [PASS/FAIL] ([N] passed)

DOCUMENTATION:
- cargo doc exit code: [0/N]

MANUAL TESTS:
- Request deserialization: [PASS/FAIL]
- Response serialization: [PASS/FAIL]
- Edge cases: [PASS/FAIL]

OVERALL STATUS: [COMPLETE/INCOMPLETE]
NOTES: [Any issues or observations]
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `cannot find type DateTime` | Missing import | Add `use chrono::{DateTime, Utc};` |
| `cannot find type Uuid` | Missing import | Add `use uuid::Uuid;` |
| Serde derive error | Missing feature | Ensure `serde = { features = ["derive"] }` in Cargo.toml |
| Module not found | mod.rs not updated | Add `pub mod topic_dtos;` to mod.rs |
| Clippy warning on unused | DTOs not used yet | This is expected - handlers come in Task 06 |

### Dependencies Required

Verify these are in `crates/context-graph-mcp/Cargo.toml`:
```toml
[dependencies]
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.6", features = ["v4", "serde"] }
```

---

## Next Task

After completing this task, proceed to **Task 06: Implement Topic Tool Handlers** which will:
1. Create `topic_tools.rs` with handler implementations
2. Create `curation_tools.rs` with handler implementations
3. Update `dispatch.rs` to route the 6 new tools
4. Wire handlers to core clustering and storage modules
