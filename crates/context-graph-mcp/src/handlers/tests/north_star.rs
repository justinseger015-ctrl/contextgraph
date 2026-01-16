//! North Star Integration Tests
//!
//! TASK-CORE-001: Updated tests after removing manual North Star methods per ARCH-03.
//!
//! ## Removed Methods (return METHOD_NOT_FOUND -32601):
//! - `purpose/north_star_alignment` - Use auto_bootstrap_north_star instead
//! - `purpose/north_star_update` - Use auto_bootstrap_north_star instead
//!
//! ## Still Available Methods:
//! - `goal/hierarchy_query` - Navigate goal hierarchy (North Star can be set programmatically)
//! - `purpose/drift_check` - Check alignment drift (requires North Star in hierarchy)
//! - `memory/store` - Store memories (works with or without North Star)
//!
//! ## Full State Verification (FSV)
//!
//! Each test directly inspects the GoalHierarchy to verify actual state changes.
//! Handler responses are cross-verified against the Source of Truth.

use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::json;

use context_graph_core::alignment::{DefaultAlignmentCalculator, GoalAlignmentCalculator};
use context_graph_core::purpose::{GoalDiscoveryMetadata, GoalHierarchy, GoalLevel, GoalNode};
use context_graph_core::stubs::{
    InMemoryTeleologicalStore, StubMultiArrayProvider, StubUtlProcessor,
};
use context_graph_core::traits::{
    MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor,
};
use context_graph_core::types::fingerprint::SemanticFingerprint;

use crate::handlers::Handlers;
use crate::protocol::JsonRpcId;

use super::make_request;

// =============================================================================
// Test Helper Functions
// =============================================================================

/// Create handlers with an EMPTY goal hierarchy (no North Star).
/// Returns handlers and shared hierarchy for FSV assertions.
fn create_handlers_no_north_star() -> (
    Handlers,
    Arc<InMemoryTeleologicalStore>,
    Arc<RwLock<GoalHierarchy>>,
) {
    let store = Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
        Arc::new(StubMultiArrayProvider::new());
    let alignment_calculator: Arc<dyn GoalAlignmentCalculator> =
        Arc::new(DefaultAlignmentCalculator::new());

    // Empty hierarchy - no North Star
    let hierarchy = GoalHierarchy::new();
    let shared_hierarchy = Arc::new(RwLock::new(hierarchy));

    let store_for_handlers: Arc<dyn TeleologicalMemoryStore> = store.clone();
    let handlers = Handlers::with_shared_hierarchy(
        store_for_handlers,
        utl_processor,
        multi_array_provider,
        alignment_calculator,
        shared_hierarchy.clone(),
    );

    (handlers, store, shared_hierarchy)
}

/// Create handlers WITH an existing North Star.
/// Returns handlers and shared hierarchy for FSV assertions.
fn create_handlers_with_north_star() -> (
    Handlers,
    Arc<InMemoryTeleologicalStore>,
    Arc<RwLock<GoalHierarchy>>,
) {
    let store = Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
        Arc::new(StubMultiArrayProvider::new());
    let alignment_calculator: Arc<dyn GoalAlignmentCalculator> =
        Arc::new(DefaultAlignmentCalculator::new());

    // Create hierarchy with North Star
    let mut hierarchy = GoalHierarchy::new();
    let discovery = GoalDiscoveryMetadata::bootstrap();

    let ns_goal = GoalNode::autonomous_goal(
        "Build the best AI assistant system".into(),
        GoalLevel::Strategic,
        SemanticFingerprint::zeroed(),
        discovery,
    )
    .expect("Failed to create North Star");
    hierarchy
        .add_goal(ns_goal)
        .expect("Failed to add initial North Star");

    let shared_hierarchy = Arc::new(RwLock::new(hierarchy));

    let store_for_handlers: Arc<dyn TeleologicalMemoryStore> = store.clone();
    let handlers = Handlers::with_shared_hierarchy(
        store_for_handlers,
        utl_processor,
        multi_array_provider,
        alignment_calculator,
        shared_hierarchy.clone(),
    );

    (handlers, store, shared_hierarchy)
}

/// Create a full test hierarchy with multiple levels.
/// TASK-P0-001: Updated for 3-level hierarchy (Strategic → Tactical → Immediate)
fn create_handlers_with_full_hierarchy() -> (
    Handlers,
    Arc<InMemoryTeleologicalStore>,
    Arc<RwLock<GoalHierarchy>>,
) {
    let store = Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
        Arc::new(StubMultiArrayProvider::new());
    let alignment_calculator: Arc<dyn GoalAlignmentCalculator> =
        Arc::new(DefaultAlignmentCalculator::new());

    let mut hierarchy = GoalHierarchy::new();
    let discovery = GoalDiscoveryMetadata::bootstrap();

    // Strategic goal 1 (top-level, no parent)
    let s1_goal = GoalNode::autonomous_goal(
        "Build comprehensive knowledge system".into(),
        GoalLevel::Strategic,
        SemanticFingerprint::zeroed(),
        discovery.clone(),
    )
    .expect("Failed to create Strategic goal 1");
    let s1_id = s1_goal.id;
    hierarchy
        .add_goal(s1_goal)
        .expect("Failed to add Strategic goal 1");

    // Strategic goal 2 (top-level, no parent)
    let s2_goal = GoalNode::autonomous_goal(
        "Improve knowledge retrieval".into(),
        GoalLevel::Strategic,
        SemanticFingerprint::zeroed(),
        discovery.clone(),
    )
    .expect("Failed to create Strategic goal 2");
    hierarchy
        .add_goal(s2_goal)
        .expect("Failed to add Strategic goal 2");

    // Tactical goal - child of Strategic 1
    let t1_goal = GoalNode::child_goal(
        "Implement semantic search".into(),
        GoalLevel::Tactical,
        s1_id,
        SemanticFingerprint::zeroed(),
        discovery.clone(),
    )
    .expect("Failed to create tactical goal");
    let t1_id = t1_goal.id;
    hierarchy
        .add_goal(t1_goal)
        .expect("Failed to add tactical goal");

    // Immediate goal - child of Tactical
    let i1_goal = GoalNode::child_goal(
        "Add vector embeddings".into(),
        GoalLevel::Immediate,
        t1_id,
        SemanticFingerprint::zeroed(),
        discovery,
    )
    .expect("Failed to create immediate goal");
    hierarchy
        .add_goal(i1_goal)
        .expect("Failed to add immediate goal");

    let shared_hierarchy = Arc::new(RwLock::new(hierarchy));

    let store_for_handlers: Arc<dyn TeleologicalMemoryStore> = store.clone();
    let handlers = Handlers::with_shared_hierarchy(
        store_for_handlers,
        utl_processor,
        multi_array_provider,
        alignment_calculator,
        shared_hierarchy.clone(),
    );

    (handlers, store, shared_hierarchy)
}

// =============================================================================
// TASK-CORE-001: Deprecated Method Tests
// =============================================================================

/// TASK-CORE-001: Verify purpose/north_star_update returns METHOD_NOT_FOUND.
///
/// This method was removed per ARCH-03 (autonomous-first architecture).
/// Manual North Star creates single 1024D embeddings incompatible with 13-embedder arrays.
#[tokio::test]
async fn test_north_star_update_returns_method_not_found() {
    let (handlers, _store, _hierarchy) = create_handlers_no_north_star();

    let params = json!({
        "description": "Test North Star",
        "keywords": ["test"],
        "replace": false
    });
    let request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Returns METHOD_NOT_FOUND (-32601) per TASK-CORE-001
    assert!(
        response.error.is_some(),
        "purpose/north_star_update must return error (deprecated)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601), got {}",
        error.code
    );
}

/// TASK-CORE-001: Verify purpose/north_star_alignment returns METHOD_NOT_FOUND.
///
/// This method was removed per ARCH-03 (autonomous-first architecture).
/// Manual alignment uses single 1024D embeddings incompatible with 13-embedder arrays.
#[tokio::test]
async fn test_north_star_alignment_returns_method_not_found() {
    let (handlers, _store, _hierarchy) = create_handlers_with_north_star();

    let params = json!({
        "fingerprint_id": "00000000-0000-0000-0000-000000000001"
    });
    let request = make_request(
        "purpose/north_star_alignment",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Returns METHOD_NOT_FOUND (-32601) per TASK-CORE-001
    assert!(
        response.error.is_some(),
        "purpose/north_star_alignment must return error (deprecated)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601), got {}",
        error.code
    );
}

// =============================================================================
// Valid Tests: goal/hierarchy_query (Still Available)
// =============================================================================

/// Test that goal/hierarchy_query returns correct Strategic (top-level) goal data.
///
/// FSV: Cross-verify returned data matches Source of Truth.
/// TASK-P0-001: Updated for 3-level hierarchy (is_north_star → is_top_level)
#[tokio::test]
async fn test_get_north_star_returns_data() {
    // SETUP: Handlers with existing Strategic goal
    let (handlers, _store, hierarchy) = create_handlers_with_north_star();

    // Get reference data from Source of Truth
    let expected_description: String;
    {
        let h = hierarchy.read();
        // TASK-P0-001: Bind to avoid temporary lifetime issue
        let top_level = h.top_level_goals();
        let ns = top_level.first().expect("Must have Strategic goal");
        expected_description = ns.description.clone();
    }

    // ACTION: Query goal hierarchy for get_all
    let params = json!({
        "operation": "get_all"
    });
    let request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Response contains Strategic goal data
    assert!(response.error.is_none(), "Get must succeed");
    let result = response.result.expect("Must have result");

    let goals = result
        .get("goals")
        .and_then(|v| v.as_array())
        .expect("Must have goals array");
    assert_eq!(goals.len(), 1, "Must have exactly 1 goal");

    let ns_goal = &goals[0];

    // Cross-verify with Source of Truth
    let returned_description = ns_goal
        .get("description")
        .and_then(|v| v.as_str())
        .expect("Must have description");
    assert_eq!(
        returned_description, expected_description,
        "FSV: Returned description must match Source of Truth"
    );

    // TASK-P0-001: Verify is_top_level flag (was is_north_star)
    let is_top = ns_goal
        .get("is_top_level")
        .and_then(|v| v.as_bool())
        .expect("Must have is_top_level");
    assert!(is_top, "FSV: is_top_level must be true for Strategic goals");
}

/// Test that drift_check works autonomously without North Star.
/// AUTONOMOUS OPERATION: drift_check succeeds with neutral response when no North Star.
///
/// When no North Star is configured, there is no goal to measure drift against.
/// The system returns a neutral "no drift" response instead of failing.
#[tokio::test]
async fn test_drift_check_works_autonomously_without_north_star() {
    // SETUP: Empty hierarchy
    let (handlers, _store, hierarchy) = create_handlers_no_north_star();

    // FSV BEFORE: Verify no North Star
    {
        let h = hierarchy.read();
        assert!(!h.has_top_level_goals(), "FSV BEFORE: Must NOT have North Star");
    }

    // ACTION: Try drift_check (returns neutral response when no North Star)
    let params = json!({
        "fingerprint_ids": ["00000000-0000-0000-0000-000000000001"]
    });
    let request = make_request(
        "purpose/drift_check",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Succeeds with autonomous mode response (no North Star = no drift to measure)
    assert!(
        response.error.is_none(),
        "Must succeed autonomously without North Star - response: {:?}",
        response.error
    );

    // Verify the response indicates autonomous operation
    let result = response.result.expect("Must have result");
    let overall_drift = result.get("overall_drift").expect("Must have overall_drift");
    assert_eq!(
        overall_drift.get("level").and_then(|v| v.as_str()),
        Some("None"),
        "Autonomous mode returns no drift"
    );
    assert_eq!(
        overall_drift.get("has_drifted").and_then(|v| v.as_bool()),
        Some(false),
        "Autonomous mode has no drift"
    );
    assert_eq!(
        result.get("autonomous_mode").and_then(|v| v.as_bool()),
        Some(true),
        "Must indicate autonomous_mode=true"
    );
}

// =============================================================================
// Valid Tests: memory/store (Works with or without North Star)
// =============================================================================

/// Test that memory/store succeeds autonomously without North Star configured.
///
/// AUTONOMOUS OPERATION: Per contextprd.md, the 13-embedding array IS the
/// teleological vector. Purpose alignment is SECONDARY metadata that defaults
/// to neutral [0.0; 13] when no North Star is configured.
#[tokio::test]
async fn test_store_memory_succeeds_without_north_star() {
    // SETUP: Empty hierarchy (no North Star configured)
    let (handlers, store, hierarchy) = create_handlers_no_north_star();

    // FSV BEFORE: Verify state
    {
        let h = hierarchy.read();
        assert!(!h.has_top_level_goals(), "FSV BEFORE: Must NOT have North Star");
    }
    let before_count = store.count().await.expect("count works");
    assert_eq!(before_count, 0, "FSV BEFORE: Store must be empty");

    // ACTION: Store memory without North Star
    let params = json!({
        "content": "Machine learning enables autonomous improvement",
        "importance": 0.8
    });
    let request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(params));
    let response = handlers.dispatch(request).await;

    // VERIFY: Succeeds with default purpose vector (AUTONOMOUS OPERATION)
    assert!(
        response.error.is_none(),
        "Store MUST succeed without North Star (AUTONOMOUS OPERATION). \
         Error: {:?}",
        response.error
    );
    let result = response.result.expect("Must have result");

    // Verify response contains fingerprint ID
    assert!(
        result.get("fingerprintId").is_some(),
        "Result must contain fingerprintId"
    );

    // FSV AFTER: Memory stored successfully
    let after_count = store.count().await.expect("count works");
    assert_eq!(after_count, 1, "FSV AFTER: Store must have 1 entry");
}

/// Test that memory/store succeeds when North Star is configured.
///
/// FSV: Verify fingerprint stored with correct purpose alignment.
#[tokio::test]
async fn test_store_memory_succeeds_with_north_star() {
    // SETUP: Handlers with North Star
    let (handlers, store, hierarchy) = create_handlers_with_north_star();

    // FSV BEFORE: Verify North Star exists
    {
        let h = hierarchy.read();
        assert!(h.has_top_level_goals(), "FSV BEFORE: Must have North Star");
    }
    let before_count = store.count().await.expect("count works");
    assert_eq!(before_count, 0, "FSV BEFORE: Store must be empty");

    // ACTION: Store memory
    let params = json!({
        "content": "Deep learning neural networks process information hierarchically",
        "importance": 0.9
    });
    let request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(params));
    let response = handlers.dispatch(request).await;

    // VERIFY: Succeeds
    assert!(
        response.error.is_none(),
        "Store must succeed with North Star: {:?}",
        response.error
    );
    let result = response.result.expect("Must have result");

    // Extract fingerprint ID
    let fingerprint_id = result
        .get("fingerprintId")
        .and_then(|v| v.as_str())
        .expect("Must return fingerprintId");
    let uuid = uuid::Uuid::parse_str(fingerprint_id).expect("Must be valid UUID");

    // FSV AFTER: Verify stored in Source of Truth
    let after_count = store.count().await.expect("count works");
    assert_eq!(after_count, 1, "FSV AFTER: Store must have 1 fingerprint");

    // Retrieve and verify fingerprint
    let fp = store
        .retrieve(uuid)
        .await
        .expect("retrieve works")
        .expect("Fingerprint must exist");

    assert_eq!(fp.id, uuid, "FSV: Retrieved ID must match");
    assert_eq!(
        fp.purpose_vector.alignments.len(),
        13,
        "FSV: Purpose vector must have 13 elements"
    );
}

// =============================================================================
// Valid Tests: goal/hierarchy_query tree operations
// =============================================================================

/// Test that goal/hierarchy_query returns structured tree with all levels.
///
/// FSV: Verify all 3 levels present and parent-child relationships correct.
/// TASK-P0-001: Updated for 3-level hierarchy (Strategic → Tactical → Immediate)
#[tokio::test]
async fn test_get_goal_hierarchy_returns_tree() {
    // SETUP: Full hierarchy with all levels
    let (handlers, _store, hierarchy) = create_handlers_with_full_hierarchy();

    // FSV BEFORE: Verify hierarchy structure
    // TASK-P0-001: Now have 3 levels with 4 goals (2 Strategic, 1 Tactical, 1 Immediate)
    {
        let h = hierarchy.read();
        assert!(h.has_top_level_goals(), "FSV BEFORE: Must have top-level goals");
        assert_eq!(h.len(), 4, "FSV BEFORE: Must have 4 goals");
        assert_eq!(
            h.at_level(GoalLevel::Strategic).len(),
            2,
            "Must have 2 Strategic goals"
        );
        assert_eq!(
            h.at_level(GoalLevel::Tactical).len(),
            1,
            "Must have 1 Tactical"
        );
        assert_eq!(
            h.at_level(GoalLevel::Immediate).len(),
            1,
            "Must have 1 Immediate"
        );
    }

    // ACTION: Query get_all
    let params = json!({
        "operation": "get_all"
    });
    let request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Response contains all goals
    assert!(response.error.is_none(), "get_all must succeed");
    let result = response.result.expect("Must have result");

    let goals = result
        .get("goals")
        .and_then(|v| v.as_array())
        .expect("Must have goals array");
    assert_eq!(goals.len(), 4, "Must return all 4 goals");

    // Verify hierarchy_stats
    let stats = result
        .get("hierarchy_stats")
        .expect("Must have hierarchy_stats");
    assert_eq!(
        stats.get("total_goals").and_then(|v| v.as_u64()),
        Some(4),
        "total_goals must be 4"
    );
    assert_eq!(
        stats.get("has_top_level_goals").and_then(|v| v.as_bool()),
        Some(true),
        "has_top_level_goals must be true"
    );

    // Verify level_counts - TASK-P0-001: Now only 3 levels
    let level_counts = stats.get("level_counts").expect("Must have level_counts");
    assert_eq!(
        level_counts.get("strategic").and_then(|v| v.as_u64()),
        Some(2),
        "strategic count must be 2"
    );
    assert_eq!(
        level_counts.get("tactical").and_then(|v| v.as_u64()),
        Some(1),
        "tactical count must be 1"
    );
    assert_eq!(
        level_counts.get("immediate").and_then(|v| v.as_u64()),
        Some(1),
        "immediate count must be 1"
    );

    // ACTION: Query get_children for a Strategic goal (top-level)
    // Extract a Strategic goal ID from the get_all response
    let strategic_goal = goals
        .iter()
        .find(|g| g.get("level").and_then(|v| v.as_str()) == Some("Strategic"))
        .expect("Must have Strategic goal");
    let strategic_id = strategic_goal
        .get("id")
        .and_then(|v| v.as_str())
        .expect("Strategic goal must have id");

    let children_params = json!({
        "operation": "get_children",
        "goal_id": strategic_id
    });
    let children_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(2)),
        Some(children_params),
    );
    let children_response = handlers.dispatch(children_request).await;

    assert!(
        children_response.error.is_none(),
        "get_children must succeed"
    );
    let children_result = children_response.result.expect("Must have result");

    let children = children_result
        .get("children")
        .and_then(|v| v.as_array())
        .expect("Must have children array");
    // Strategic goal may have 0 or 1 children depending on which one we picked
    // The first Strategic has a Tactical child, the second doesn't
    assert!(
        children.len() <= 1,
        "Strategic must have at most 1 child (Tactical)"
    );

    // ACTION: Query get_ancestors for Immediate goal
    // Extract the Immediate goal ID from the get_all response
    let immediate_goal = goals
        .iter()
        .find(|g| g.get("level").and_then(|v| v.as_str()) == Some("Immediate"))
        .expect("Must have Immediate goal");
    let immediate_id = immediate_goal
        .get("id")
        .and_then(|v| v.as_str())
        .expect("Immediate must have id");

    let ancestors_params = json!({
        "operation": "get_ancestors",
        "goal_id": immediate_id
    });
    let ancestors_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(3)),
        Some(ancestors_params),
    );
    let ancestors_response = handlers.dispatch(ancestors_request).await;

    assert!(
        ancestors_response.error.is_none(),
        "get_ancestors must succeed"
    );
    let ancestors_result = ancestors_response.result.expect("Must have result");

    let ancestors = ancestors_result
        .get("ancestors")
        .and_then(|v| v.as_array())
        .expect("Must have ancestors array");

    // Path: Immediate -> Tactical -> Strategic
    // Should have 2 ancestors (Tactical and Strategic)
    assert!(
        ancestors.len() >= 2,
        "Must have at least 2 ancestors in path to Strategic"
    );
}

/// Test goal/hierarchy_query with non-existent goal_id.
#[tokio::test]
async fn test_hierarchy_query_nonexistent_goal() {
    // SETUP: Full hierarchy
    let (handlers, _store, _hierarchy) = create_handlers_with_full_hierarchy();

    // ACTION: Query non-existent goal with a valid UUID format that doesn't exist
    // Using a well-formed UUID that won't be in the hierarchy
    let params = json!({
        "operation": "get_goal",
        "goal_id": "00000000-0000-0000-0000-000000000000"
    });
    let request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Fails with GOAL_NOT_FOUND
    assert!(response.error.is_some(), "Must fail with non-existent goal");
    let error = response.error.unwrap();
    assert_eq!(error.code, -32020, "Must return GOAL_NOT_FOUND (-32020)");
}

/// Test purpose/drift_check operates autonomously without North Star.
/// AUTONOMOUS OPERATION: Returns neutral "no drift" response when no goal to measure against.
#[tokio::test]
async fn test_drift_check_autonomous_operation() {
    // SETUP: Empty hierarchy
    let (handlers, _store, _hierarchy) = create_handlers_no_north_star();

    // ACTION: Try drift check (returns neutral response in autonomous mode)
    let params = json!({
        "fingerprint_ids": ["00000000-0000-0000-0000-000000000001"]
    });
    let request = make_request(
        "purpose/drift_check",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Succeeds with autonomous mode response
    assert!(
        response.error.is_none(),
        "Drift check succeeds autonomously - no North Star means no drift to measure"
    );

    // Verify autonomous mode is indicated in response
    let result = response.result.expect("Must have result");
    assert_eq!(
        result.get("autonomous_mode").and_then(|v| v.as_bool()),
        Some(true),
        "Must indicate autonomous_mode=true"
    );
}
