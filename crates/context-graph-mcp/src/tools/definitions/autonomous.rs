//! Autonomous North Star system tool definitions.
//! TASK-AUTONOMOUS-MCP: Bootstrap, drift, correction, pruning, consolidation, sub-goals, status.

use serde_json::json;
use crate::tools::types::ToolDefinition;

/// Returns Autonomous tool definitions (7 tools).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // auto_bootstrap_north_star - Initialize autonomous services from teleological embeddings
        ToolDefinition::new(
            "auto_bootstrap_north_star",
            "Bootstrap the autonomous North Star system from existing teleological embeddings. \
             Analyzes stored memories' 13-embedder teleological fingerprints to discover emergent \
             purpose patterns and initialize drift detection, pruning, consolidation, and sub-goal \
             discovery services. Works directly with teleological arrays for apples-to-apples comparisons.",
            json!({
                "type": "object",
                "properties": {
                    "confidence_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7,
                        "description": "Minimum confidence threshold for bootstrapping"
                    },
                    "max_candidates": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "Maximum number of candidates to evaluate during bootstrap"
                    }
                },
                "required": []
            }),
        ),

        // get_alignment_drift - Get drift state and history
        ToolDefinition::new(
            "get_alignment_drift",
            "Get the current alignment drift state including severity, trend, and recommendations. \
             Drift measures how far the system has deviated from the North Star goal alignment. \
             High drift indicates memories are becoming misaligned with the primary purpose.",
            json!({
                "type": "object",
                "properties": {
                    "timeframe": {
                        "type": "string",
                        "enum": ["1h", "24h", "7d", "30d"],
                        "default": "24h",
                        "description": "Timeframe to analyze for drift"
                    },
                    "include_history": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include full drift history in response"
                    }
                },
                "required": []
            }),
        ),

        // trigger_drift_correction - Manually trigger drift correction
        ToolDefinition::new(
            "trigger_drift_correction",
            "Manually trigger a drift correction cycle. Applies correction strategies based on \
             current drift severity: threshold adjustment, weight rebalancing, goal reinforcement, \
             or emergency intervention for severe drift.",
            json!({
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "default": false,
                        "description": "Force correction even if drift severity is low"
                    },
                    "target_alignment": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Target alignment to achieve (optional, uses adaptive if not set)"
                    }
                },
                "required": []
            }),
        ),

        // get_pruning_candidates - Get memories eligible for pruning
        ToolDefinition::new(
            "get_pruning_candidates",
            "Identify memories that are candidates for pruning based on staleness, low alignment, \
             redundancy, or orphaned status. Returns a ranked list with reasons and recommendations. \
             Use this for routine memory hygiene and to identify unused/outdated content.",
            json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 20,
                        "description": "Maximum number of candidates to return"
                    },
                    "min_staleness_days": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 30,
                        "description": "Minimum age in days for staleness consideration"
                    },
                    "min_alignment": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.4,
                        "description": "Memories below this alignment are candidates for low-alignment pruning"
                    }
                },
                "required": []
            }),
        ),

        // trigger_consolidation - Trigger memory consolidation
        ToolDefinition::new(
            "trigger_consolidation",
            "Trigger memory consolidation to merge similar memories and reduce redundancy. \
             Uses similarity-based, temporal, or semantic strategies to identify merge candidates. \
             Helps optimize memory storage and improve retrieval efficiency.",
            json!({
                "type": "object",
                "properties": {
                    "max_memories": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10000,
                        "default": 100,
                        "description": "Maximum memories to process in one batch"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["similarity", "temporal", "semantic"],
                        "default": "similarity",
                        "description": "Consolidation strategy to use"
                    },
                    "min_similarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.85,
                        "description": "Minimum similarity threshold for consolidation candidates"
                    }
                },
                "required": []
            }),
        ),

        // discover_sub_goals - Discover potential sub-goals
        ToolDefinition::new(
            "discover_sub_goals",
            "Discover potential sub-goals from memory clusters. Analyzes stored memories to find \
             emergent themes and patterns that could become strategic or tactical goals. \
             Helps evolve the goal hierarchy based on actual content.",
            json!({
                "type": "object",
                "properties": {
                    "min_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.6,
                        "description": "Minimum confidence for a discovered sub-goal"
                    },
                    "max_goals": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "Maximum number of sub-goals to discover"
                    },
                    "parent_goal_id": {
                        "type": "string",
                        "description": "Parent goal ID to discover sub-goals for (defaults to North Star)"
                    }
                },
                "required": []
            }),
        ),

        // get_autonomous_status - Get comprehensive autonomous system status
        ToolDefinition::new(
            "get_autonomous_status",
            "Get comprehensive status of the autonomous North Star system including all services: \
             drift detection, correction, pruning, consolidation, and sub-goal discovery. \
             Returns health scores, recommendations, and optional detailed metrics.",
            json!({
                "type": "object",
                "properties": {
                    "include_metrics": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include detailed per-service metrics"
                    },
                    "include_history": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include recent operation history"
                    },
                    "history_count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "Number of history entries to include"
                    }
                },
                "required": []
            }),
        ),
    ]
}
