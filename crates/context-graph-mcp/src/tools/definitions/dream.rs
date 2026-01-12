//! Dream consolidation tool definitions.
//! TASK-DREAM-MCP: Dream triggers, status, abort, amortized shortcuts.

use serde_json::json;
use crate::tools::types::ToolDefinition;

/// Returns Dream tool definitions (4 tools).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // trigger_dream - Manually trigger a dream consolidation cycle
        ToolDefinition::new(
            "trigger_dream",
            "Manually trigger a dream consolidation cycle. System must be idle (activity < 0.15). \
             Executes NREM (3 min) + REM (2 min) phases. Returns DreamReport with metrics. \
             Aborts automatically on external query (wake latency < 100ms).",
            json!({
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "default": false,
                        "description": "Force dream even if activity is above threshold (not recommended)"
                    }
                },
                "required": []
            }),
        ),

        // get_dream_status - Get current dream system status
        ToolDefinition::new(
            "get_dream_status",
            "Get current dream system status including state (Awake/NREM/REM/Waking), \
             GPU usage, activity level, and time since last dream cycle.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // abort_dream - Abort current dream cycle
        ToolDefinition::new(
            "abort_dream",
            "Abort the current dream cycle. Must complete wake within 100ms (constitution mandate). \
             Returns wake latency and partial dream report.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // get_amortized_shortcuts - Get shortcut candidates from amortized learning
        ToolDefinition::new(
            "get_amortized_shortcuts",
            "Get shortcut candidates from amortized learning. Returns paths traversed 5+ times \
             with 3+ hops that qualify for direct edge creation.",
            json!({
                "type": "object",
                "properties": {
                    "min_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7,
                        "description": "Minimum confidence threshold for shortcuts"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                        "description": "Maximum shortcuts to return"
                    }
                },
                "required": []
            }),
        ),
    ]
}
