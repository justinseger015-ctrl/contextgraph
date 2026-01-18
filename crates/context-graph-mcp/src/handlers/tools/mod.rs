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
