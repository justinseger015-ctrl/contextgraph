//! Tool definitions per PRD v6 Section 10 (18 tools total).

pub(crate) mod core;
pub(crate) mod curation;
pub(crate) mod dream;
pub(crate) mod file_watcher;
pub(crate) mod merge;
pub(crate) mod topic;

use crate::tools::types::ToolDefinition;

/// Get all tool definitions for the `tools/list` response.
pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    let mut tools = Vec::with_capacity(18);

    // Core tools (5)
    tools.extend(core::definitions());

    // Merge tool (1 - part of curation)
    tools.extend(merge::definitions());

    // Curation tools (2)
    tools.extend(curation::definitions());

    // Topic tools (4)
    tools.extend(topic::definitions());

    // Dream tools (2)
    tools.extend(dream::definitions());

    // File watcher tools (4)
    tools.extend(file_watcher::definitions());

    tools
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_tool_count() {
        assert_eq!(get_tool_definitions().len(), 18);
    }

    #[test]
    fn test_all_tool_names_present() {
        let tools = get_tool_definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

        let expected = [
            // Core tools (5)
            "inject_context",
            "store_memory",
            "get_memetic_status",
            "search_graph",
            "trigger_consolidation",
            // Merge tool (1)
            "merge_concepts",
            // Curation tools (2)
            "forget_concept",
            "boost_importance",
            // Topic tools (4)
            "get_topic_portfolio",
            "get_topic_stability",
            "detect_topics",
            "get_divergence_alerts",
            // Dream tools (2)
            "trigger_dream",
            "get_dream_status",
            // File watcher tools (4)
            "list_watched_files",
            "get_file_watcher_stats",
            "delete_file_content",
            "reconcile_files",
        ];

        for name in expected {
            assert!(names.contains(&name), "Missing tool: {}", name);
        }
    }

    #[test]
    fn test_no_duplicate_tools() {
        let tools = get_tool_definitions();
        let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        let len_before = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), len_before);
    }

    #[test]
    fn test_all_tools_have_descriptions_and_schemas() {
        for tool in &get_tool_definitions() {
            assert!(
                !tool.description.is_empty(),
                "Tool {} missing description",
                tool.name
            );
            assert!(
                tool.input_schema.get("type").is_some(),
                "Tool {} missing schema type",
                tool.name
            );
        }
    }

    #[test]
    fn test_submodule_counts() {
        assert_eq!(core::definitions().len(), 5);
        assert_eq!(merge::definitions().len(), 1);
        assert_eq!(curation::definitions().len(), 2);
        assert_eq!(topic::definitions().len(), 4);
        assert_eq!(dream::definitions().len(), 2);
        assert_eq!(file_watcher::definitions().len(), 4);
    }
}
