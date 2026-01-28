//! Tool registry for centralized tool management.
//!
//! TASK-41: Provides O(1) tool lookup and registration verification.
//! This module provides a formal `ToolRegistry` struct that:
//! - Stores tools in a HashMap for O(1) lookup by name
//! - Validates all tools are registered at startup
//! - Ensures no duplicate tool registrations

#![allow(dead_code)]

use std::collections::HashMap;

use super::types::ToolDefinition;

/// Registry holding all MCP tool definitions.
///
/// Provides:
/// - O(1) lookup by tool name
/// - List of all registered tools (sorted by name)
/// - Validation that all tools are registered
/// - Fail-fast on duplicate registrations
pub struct ToolRegistry {
    tools: HashMap<String, ToolDefinition>,
}

impl ToolRegistry {
    /// Create empty registry with pre-allocated capacity.
    pub fn new() -> Self {
        Self {
            tools: HashMap::with_capacity(50),
        }
    }

    /// Register a tool definition.
    ///
    /// # Panics
    ///
    /// Panics if a tool with the same name is already registered.
    pub fn register(&mut self, tool: ToolDefinition) {
        let name = tool.name.clone();
        if self.tools.contains_key(&name) {
            panic!(
                "TASK-41: Duplicate tool registration: '{}'. \
                 Each tool name must be unique.",
                name
            );
        }
        self.tools.insert(name, tool);
    }

    /// Get a tool definition by name.
    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    /// List all registered tools (sorted by name for deterministic output).
    pub fn list(&self) -> Vec<&ToolDefinition> {
        let mut tools: Vec<_> = self.tools.values().collect();
        tools.sort_by(|a, b| a.name.cmp(&b.name));
        tools
    }

    /// Get count of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Check if a tool exists by name.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get all tool names as a sorted vector.
    pub fn tool_names(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.tools.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// NOTE: register_all_tools() was removed as dead code.
// Production uses get_tool_definitions() from definitions/mod.rs which returns 49 tools.
// The old register_all_tools() only registered 19 tools and was never used in production.

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_registry_new_empty() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    #[should_panic(expected = "Duplicate tool registration")]
    fn test_duplicate_registration_panics() {
        let mut registry = ToolRegistry::new();
        let tool = ToolDefinition::new("test_tool", "Test", json!({"type": "object"}));
        registry.register(tool.clone());
        registry.register(tool);
    }

    #[test]
    fn test_registry_operations() {
        let mut registry = ToolRegistry::new();

        let tool = ToolDefinition::new("test_tool", "Test tool", json!({"type": "object"}));
        registry.register(tool);

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert!(registry.contains("test_tool"));
        assert!(!registry.contains("nonexistent"));

        let retrieved = registry.get("test_tool").expect("Tool must exist");
        assert_eq!(retrieved.name, "test_tool");

        let names = registry.tool_names();
        assert_eq!(names, vec!["test_tool"]);
    }
}
