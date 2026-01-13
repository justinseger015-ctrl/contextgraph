# TASK-MCP-015: Add tool registration to MCP server

```xml
<task_spec id="TASK-MCP-015" version="1.0">
<metadata>
  <title>Add tool registration to MCP server</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>41</sequence>
  <implements><requirement_ref>REQ-MCP-015</requirement_ref></implements>
  <depends_on>TASK-MCP-001, TASK-MCP-002, TASK-MCP-003, TASK-MCP-004, TASK-MCP-005, TASK-MCP-006, TASK-MCP-007, TASK-MCP-008, TASK-MCP-009, TASK-MCP-010, TASK-MCP-011, TASK-MCP-012, TASK-MCP-013, TASK-MCP-014</depends_on>
  <estimated_hours>3</estimated_hours>
</metadata>

<context>
All MCP tools must be registered in the server's tool registry.
This enables tool discovery and routing.
</context>

<scope>
<in_scope>
- Create tool registry struct
- Register all implemented tools
- Implement tools/list handler
- Implement tool dispatch by name
</in_scope>
<out_of_scope>
- Individual tool implementations (other tasks)
</out_of_scope>
</scope>

<definition_of_done>
<signatures>
```rust
// crates/context-graph-mcp/src/tools/registry.rs
use std::collections::HashMap;

pub struct ToolRegistry {
    tools: HashMap<String, ToolDefinition>,
}

pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

impl ToolRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, tool: ToolDefinition);
    pub fn get(&self, name: &str) -> Option<&ToolDefinition>;
    pub fn list(&self) -> Vec<&ToolDefinition>;
}

/// Register all context-graph MCP tools.
pub fn register_all_tools(registry: &mut ToolRegistry) {
    registry.register(epistemic_action_tool());
    registry.register(merge_concepts_tool());
    registry.register(get_johari_classification_tool());
    registry.register(get_coherence_state_tool());
    registry.register(trigger_dream_tool());
    registry.register(get_gpu_status_tool());
    registry.register(get_identity_continuity_tool());
    registry.register(get_kuramoto_state_tool());
    registry.register(set_coupling_strength_tool());
}
```
</signatures>
<constraints>
- All tools MUST be registered
- list MUST return all registered tools
- Dispatch MUST route by exact name match
</constraints>
<verification>
```bash
cargo test -p context-graph-mcp registry
cargo test -p context-graph-mcp test_all_tools_registered
```
</verification>
</definition_of_done>

<files_to_create>
- crates/context-graph-mcp/src/tools/registry.rs
</files_to_create>

<files_to_modify>
- crates/context-graph-mcp/src/tools/mod.rs (add registry)
- crates/context-graph-mcp/src/server.rs (use registry)
</files_to_modify>

<test_commands>
```bash
cargo test -p context-graph-mcp registry
```
</test_commands>
</task_spec>
```
