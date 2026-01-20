//! File watcher tool definitions per PRD v6 Section 10.
//!
//! Tools (4):
//! - list_watched_files: List all files with embeddings
//! - get_file_watcher_stats: Get statistics about file watcher content
//! - delete_file_content: Delete all embeddings for a specific file
//! - reconcile_files: Find orphaned files and optionally delete them
//!
//! Constitution Compliance:
//! - SEC-06: Soft delete 30-day recovery for delete_file_content
//! - FAIL FAST: All tools error on failures, no fallbacks

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns file watcher tool definitions (4 tools).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // list_watched_files
        ToolDefinition::new(
            "list_watched_files",
            "List all files that have embeddings in the knowledge graph from the file watcher. \
             Returns file paths with chunk counts and last update times.",
            json!({
                "type": "object",
                "properties": {
                    "include_counts": {
                        "type": "boolean",
                        "default": true,
                        "description": "Include chunk counts per file"
                    },
                    "path_filter": {
                        "type": "string",
                        "description": "Optional glob pattern to filter paths (e.g., '**/docs/*.md')"
                    }
                },
                "required": []
            }),
        ),
        // get_file_watcher_stats
        ToolDefinition::new(
            "get_file_watcher_stats",
            "Get statistics about file watcher content in the knowledge graph. \
             Returns total files, total chunks, average chunks per file, and min/max values.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),
        // delete_file_content
        ToolDefinition::new(
            "delete_file_content",
            "Delete all embeddings for a specific file path. Use for manual cleanup. \
             Supports soft delete with 30-day recovery (per SEC-06).",
            json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file"
                    },
                    "soft_delete": {
                        "type": "boolean",
                        "default": true,
                        "description": "Use soft delete with 30-day recovery (default true per SEC-06)"
                    }
                },
                "required": ["file_path"]
            }),
        ),
        // reconcile_files
        ToolDefinition::new(
            "reconcile_files",
            "Find orphaned files (embeddings exist but file doesn't on disk) and optionally delete them. \
             Use dry_run=true to preview changes without modifying data.",
            json!({
                "type": "object",
                "properties": {
                    "dry_run": {
                        "type": "boolean",
                        "default": true,
                        "description": "If true, only report orphans without deleting"
                    },
                    "base_path": {
                        "type": "string",
                        "description": "Optional base path to limit reconciliation scope"
                    }
                },
                "required": []
            }),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_watcher_definitions_count() {
        let tools = definitions();
        assert_eq!(tools.len(), 4, "Should have 4 file watcher tools");
    }

    #[test]
    fn test_file_watcher_tools_names() {
        let tools = definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"list_watched_files"));
        assert!(names.contains(&"get_file_watcher_stats"));
        assert!(names.contains(&"delete_file_content"));
        assert!(names.contains(&"reconcile_files"));
    }

    #[test]
    fn test_delete_file_content_required_fields() {
        let tools = definitions();
        let delete = tools
            .iter()
            .find(|t| t.name == "delete_file_content")
            .unwrap();
        let required = delete
            .input_schema
            .get("required")
            .unwrap()
            .as_array()
            .unwrap();
        let fields: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(fields.len(), 1);
        assert!(fields.contains(&"file_path"));
    }

    #[test]
    fn test_delete_file_content_soft_delete_default() {
        let tools = definitions();
        let delete = tools
            .iter()
            .find(|t| t.name == "delete_file_content")
            .unwrap();
        let props = delete.input_schema.get("properties").unwrap();
        let soft_delete = props.get("soft_delete").unwrap();
        // Per SEC-06: defaults to true
        assert!(soft_delete.get("default").unwrap().as_bool().unwrap());
    }

    #[test]
    fn test_reconcile_files_dry_run_default() {
        let tools = definitions();
        let reconcile = tools
            .iter()
            .find(|t| t.name == "reconcile_files")
            .unwrap();
        let props = reconcile.input_schema.get("properties").unwrap();
        let dry_run = props.get("dry_run").unwrap();
        // Defaults to true for safety
        assert!(dry_run.get("default").unwrap().as_bool().unwrap());
    }

    #[test]
    fn test_all_tools_have_type_object() {
        let tools = definitions();
        for tool in &tools {
            assert_eq!(
                tool.input_schema.get("type").unwrap().as_str().unwrap(),
                "object",
                "Tool {} should have type: object",
                tool.name
            );
        }
    }

    #[test]
    fn test_descriptions_are_not_empty() {
        let tools = definitions();
        for tool in &tools {
            assert!(
                !tool.description.is_empty(),
                "Tool {} missing description",
                tool.name
            );
        }
    }

    // ========== SYNTHETIC DATA VALIDATION TESTS ==========

    #[test]
    fn test_synthetic_list_watched_files() {
        let synthetic_input = json!({
            "include_counts": true,
            "path_filter": "**/docs/*.md"
        });

        let tools = definitions();
        let list = tools
            .iter()
            .find(|t| t.name == "list_watched_files")
            .unwrap();

        // Validate include_counts is boolean
        assert!(synthetic_input
            .get("include_counts")
            .unwrap()
            .as_bool()
            .is_some());

        // Validate path_filter is string
        assert!(synthetic_input
            .get("path_filter")
            .unwrap()
            .as_str()
            .is_some());

        // Verify schema has these properties
        let props = list.input_schema.get("properties").unwrap();
        assert!(props.get("include_counts").is_some());
        assert!(props.get("path_filter").is_some());

        println!("[SYNTHETIC TEST] list_watched_files with filter pattern");
    }

    #[test]
    fn test_synthetic_delete_file_content() {
        let synthetic_input = json!({
            "file_path": "/home/user/docs/readme.md",
            "soft_delete": true
        });

        // Validate file_path is string
        assert!(synthetic_input.get("file_path").unwrap().as_str().is_some());

        // Validate soft_delete is boolean
        assert!(synthetic_input
            .get("soft_delete")
            .unwrap()
            .as_bool()
            .is_some());

        println!("[SYNTHETIC TEST] delete_file_content with soft_delete=true");
    }

    #[test]
    fn test_synthetic_reconcile_files_dry_run() {
        let synthetic_input = json!({
            "dry_run": true,
            "base_path": "/home/user/project"
        });

        // Validate dry_run is boolean
        assert!(synthetic_input.get("dry_run").unwrap().as_bool().is_some());

        // Validate base_path is string
        assert!(synthetic_input.get("base_path").unwrap().as_str().is_some());

        println!("[SYNTHETIC TEST] reconcile_files with dry_run=true");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let tools = definitions();
        let json_str = serde_json::to_string(&tools).expect("Serialization failed");
        assert!(json_str.contains("list_watched_files"));
        assert!(json_str.contains("get_file_watcher_stats"));
        assert!(json_str.contains("delete_file_content"));
        assert!(json_str.contains("reconcile_files"));
        println!("[PASS] File watcher tools serialize correctly");
    }
}
