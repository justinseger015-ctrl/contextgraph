//! Tool definitions per PRD v6 Section 10.
//!
//! Only PRD-required tools are exposed:
//! - Core: inject_context, store_memory, get_memetic_status, search_graph, trigger_consolidation
//! - Curation: merge_concepts

pub(crate) mod core;
pub mod merge;

use crate::tools::types::ToolDefinition;

/// Get all tool definitions for the `tools/list` response.
///
/// Per PRD v6, only 6 tools are exposed:
/// - inject_context
/// - store_memory
/// - get_memetic_status
/// - search_graph
/// - trigger_consolidation
/// - merge_concepts
pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    let mut tools = Vec::with_capacity(6);

    // Core tools (5)
    tools.extend(core::definitions());

    // Merge tools (1)
    tools.extend(merge::definitions());

    tools
}
