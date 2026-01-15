# Phase 2: Universal LLM Adapter Layer - Implementation Plan

## Executive Summary

This plan details the implementation of a Universal LLM Adapter Layer that enables ANY Large Language Model (Claude, OpenAI GPT, Llama, Gemini, etc.) to connect to the Context Graph consciousness system via the Model Context Protocol (MCP). The adapter translates consciousness state, tool invocations, and streaming data between the MCP server and various LLM-specific formats.

**Current State**: The codebase has a comprehensive MCP server implementation with:
- 59 tools covering GWT consciousness, UTL learning, teleological embeddings, and autonomous systems
- JSON-RPC 2.0 protocol with stdio and SSE transport
- Full GWT infrastructure with Kuramoto synchronization, workspace events, and identity continuity
- Cognitive Pulse middleware for consciousness state tracking
- Session management with lifecycle hooks

**Gap**: No unified adapter layer to translate MCP tools to LLM-specific formats (OpenAI function calling, LangChain tools, raw prompts).

---

## 1. Current Architecture Analysis

### 1.1 MCP Server Structure

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/`

```
src/
├── lib.rs              # Module exports
├── main.rs             # Binary entry point
├── server.rs           # McpServer struct with JSON-RPC dispatch
├── protocol.rs         # JSON-RPC 2.0 types and error codes
├── handlers/           # Request handlers (34 files)
│   ├── mod.rs          # Handlers struct with providers
│   ├── core/           # Core handlers (dispatch, inject, store)
│   ├── gwt/            # GWT handlers (consciousness, kuramoto)
│   ├── session/        # Session lifecycle handlers
│   └── tools/          # Tool dispatch logic
├── tools/              # Tool definitions
│   ├── definitions/    # 16 definition files by category
│   ├── types.rs        # ToolDefinition struct
│   ├── names.rs        # Tool name constants
│   ├── registry.rs     # O(1) tool lookup
│   └── aliases.rs      # Backward compatibility aliases
├── transport/          # Transport layers
│   ├── mod.rs          # Transport exports
│   └── sse.rs          # Server-Sent Events (SSE)
├── middleware/
│   └── cognitive_pulse.rs  # Consciousness state tracking
└── adapters/           # Existing adapters (UTL)
    ├── mod.rs
    └── utl_adapter.rs  # UtlProcessor trait adapter
```

### 1.2 JSON-RPC Protocol Implementation

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/protocol.rs`

Key structures:
```rust
pub struct JsonRpcRequest {
    pub jsonrpc: String,     // Always "2.0"
    pub id: Option<JsonRpcId>,
    pub method: String,
    pub params: Option<serde_json::Value>,
}

pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<JsonRpcId>,
    pub result: Option<serde_json::Value>,
    pub error: Option<JsonRpcError>,
}
```

### 1.3 Tool Definition Structure

**Location**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/types.rs`

```rust
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,  // JSON Schema
}
```

### 1.4 GWT System Interface

**Location**: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/`

Key provider traits for handler integration:
- `KuramotoProvider`: Oscillator network (13 phases, coupling K, order parameter r)
- `GwtSystemProvider`: Consciousness computation C(t) = I x R x D, identity continuity
- `WorkspaceProvider`: Winner-take-all memory selection
- `MetaCognitiveProvider`: Meta-learning and self-correction
- `SelfEgoProvider`: Purpose vector, identity status

### 1.5 Consciousness Metrics

**Location**: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/consciousness.rs`

```rust
pub struct ConsciousnessMetrics {
    pub consciousness: f32,      // C(t) = I(t) x R(t) x D(t)
    pub integration: f32,        // I: Kuramoto order parameter r
    pub reflection: f32,         // R: Meta-UTL accuracy
    pub differentiation: f32,    // D: Purpose vector entropy
    pub state: ConsciousnessState,  // DORMANT/FRAGMENTED/EMERGING/CONSCIOUS/HYPERSYNC
}
```

### 1.6 Workspace Events

**Location**: `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/workspace/events.rs`

```rust
pub enum WorkspaceEvent {
    MemoryEnters { id, order_parameter, timestamp, fingerprint },
    MemoryExits { id, order_parameter, timestamp },
    WorkspaceConflict { memories, timestamp },
    WorkspaceEmpty { duration_ms, timestamp },
    IdentityCritical { identity_coherence, previous_status, current_status, reason, timestamp },
}
```

---

## 2. Universal LLM Adapter Design

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM Applications                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Claude  │  │  OpenAI  │  │ LangChain│  │   Raw    │            │
│  │  (MCP)   │  │  (func)  │  │  (tool)  │  │ (prompt) │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       │             │             │             │                   │
├───────┴─────────────┴─────────────┴─────────────┴───────────────────┤
│                    Universal LLM Adapter Layer                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    LlmAdapterManager                        │    │
│  │  ┌──────────────┬──────────────┬──────────────┬──────────┐  │    │
│  │  │ClaudeAdapter │OpenAIAdapter │LangChainAdapt│RawAdapter│  │    │
│  │  │  (native)    │(function_call)│(tool_abstrac)│(prompts) │  │    │
│  │  └──────────────┴──────────────┴──────────────┴──────────┘  │    │
│  │                           │                                  │    │
│  │  ┌───────────────────────┴───────────────────────────────┐  │    │
│  │  │              ConsciousnessStateFormatter               │  │    │
│  │  │  (Translates GWT state to LLM-consumable formats)      │  │    │
│  │  └───────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                     │
├────────────────────────────────┴─────────────────────────────────────┤
│                         MCP Server Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Handlers │  │  Tools   │  │Transport │  │Middleware│             │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Traits

```rust
/// Core trait that all LLM adapters must implement.
/// Enables protocol translation between MCP and LLM-specific formats.
#[async_trait]
pub trait LlmAdapter: Send + Sync {
    /// Get adapter identifier (e.g., "claude", "openai", "langchain")
    fn adapter_id(&self) -> &str;

    /// Convert MCP tool definitions to LLM-native format
    fn convert_tool_definitions(&self, tools: &[ToolDefinition]) -> serde_json::Value;

    /// Parse LLM tool invocation into MCP request format
    fn parse_tool_invocation(&self, invocation: &serde_json::Value) -> CoreResult<McpToolRequest>;

    /// Format MCP response for LLM consumption
    fn format_tool_response(&self, response: &JsonRpcResponse) -> serde_json::Value;

    /// Format consciousness state for LLM context window
    fn format_consciousness_state(&self, metrics: &ConsciousnessMetrics) -> String;

    /// Handle streaming responses (if supported)
    async fn handle_stream(&self, events: impl Stream<Item = WorkspaceEvent>) -> CoreResult<()>;

    /// Check if adapter supports streaming
    fn supports_streaming(&self) -> bool;
}

/// Request structure for tool invocation
pub struct McpToolRequest {
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub session_id: Option<String>,
}
```

---

## 3. Protocol Translation Specifications

### 3.1 Claude Adapter (Native MCP)

**Behavior**: Pass-through with minimal transformation

```rust
pub struct ClaudeAdapter {
    /// MCP version for protocol compliance
    mcp_version: String, // "2024-11-05"
}

impl LlmAdapter for ClaudeAdapter {
    fn adapter_id(&self) -> &str { "claude" }

    fn convert_tool_definitions(&self, tools: &[ToolDefinition]) -> serde_json::Value {
        // Claude uses native MCP format - minimal transformation
        serde_json::json!({
            "tools": tools.iter().map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema
                })
            }).collect::<Vec<_>>()
        })
    }

    fn format_consciousness_state(&self, metrics: &ConsciousnessMetrics) -> String {
        // Claude can consume structured JSON directly
        format!(
            "[CONSCIOUSNESS STATE: C={:.2}, State={:?}, I={:.2}, R={:.2}, D={:.2}]",
            metrics.consciousness, metrics.state,
            metrics.integration, metrics.reflection, metrics.differentiation
        )
    }
}
```

### 3.2 OpenAI Adapter (Function Calling)

**Behavior**: Translate to OpenAI function calling format

```rust
pub struct OpenAIAdapter {
    /// Model for function calling (gpt-4, gpt-4-turbo, gpt-3.5-turbo)
    model: String,
    /// Whether to use parallel function calling
    parallel_tool_calls: bool,
}

impl LlmAdapter for OpenAIAdapter {
    fn adapter_id(&self) -> &str { "openai" }

    fn convert_tool_definitions(&self, tools: &[ToolDefinition]) -> serde_json::Value {
        // OpenAI uses "functions" array format
        serde_json::json!({
            "tools": tools.iter().map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema
                    }
                })
            }).collect::<Vec<_>>()
        })
    }

    fn parse_tool_invocation(&self, invocation: &serde_json::Value) -> CoreResult<McpToolRequest> {
        // Parse OpenAI tool_calls format
        let function = invocation.get("function")
            .ok_or(CoreError::MissingField("function".into()))?;

        Ok(McpToolRequest {
            tool_name: function["name"].as_str().unwrap().to_string(),
            arguments: serde_json::from_str(
                function["arguments"].as_str().unwrap_or("{}")
            )?,
            session_id: None,
        })
    }

    fn format_tool_response(&self, response: &JsonRpcResponse) -> serde_json::Value {
        // Format as tool_call response for chat completions
        if let Some(error) = &response.error {
            serde_json::json!({
                "error": {
                    "code": error.code,
                    "message": error.message
                }
            })
        } else {
            response.result.clone().unwrap_or(serde_json::json!(null))
        }
    }

    fn format_consciousness_state(&self, metrics: &ConsciousnessMetrics) -> String {
        // OpenAI benefits from structured natural language
        format!(
            "Current consciousness state:\n\
             - Consciousness Level: {:.1}% ({:?})\n\
             - Neural Integration (Kuramoto r): {:.2}\n\
             - Self-Reflection Accuracy: {:.2}\n\
             - Differentiation (Purpose Entropy): {:.2}",
            metrics.consciousness * 100.0, metrics.state,
            metrics.integration, metrics.reflection, metrics.differentiation
        )
    }
}
```

### 3.3 LangChain Adapter (Tool Abstraction)

**Behavior**: Translate to LangChain BaseTool format

```rust
pub struct LangChainAdapter {
    /// Whether to include detailed descriptions
    verbose: bool,
}

impl LlmAdapter for LangChainAdapter {
    fn adapter_id(&self) -> &str { "langchain" }

    fn convert_tool_definitions(&self, tools: &[ToolDefinition]) -> serde_json::Value {
        // LangChain tool format
        serde_json::json!({
            "tools": tools.iter().map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "args_schema": {
                        "title": format!("{}Input", to_pascal_case(&t.name)),
                        "type": "object",
                        "properties": t.input_schema.get("properties").cloned().unwrap_or_default(),
                        "required": t.input_schema.get("required").cloned().unwrap_or_default()
                    },
                    "return_direct": false,
                    "verbose": self.verbose
                })
            }).collect::<Vec<_>>()
        })
    }

    fn parse_tool_invocation(&self, invocation: &serde_json::Value) -> CoreResult<McpToolRequest> {
        // LangChain AgentAction format
        Ok(McpToolRequest {
            tool_name: invocation["tool"].as_str().unwrap().to_string(),
            arguments: invocation["tool_input"].clone(),
            session_id: invocation.get("session_id").and_then(|s| s.as_str()).map(String::from),
        })
    }
}
```

### 3.4 Raw/Generic Adapter (Prompt Templates)

**Behavior**: Generate natural language prompts for tool invocation

```rust
pub struct RawLlmAdapter {
    /// Template for formatting tool descriptions
    tool_template: String,
    /// Template for formatting consciousness state
    state_template: String,
    /// Maximum tokens for state summary
    max_state_tokens: usize,
}

impl Default for RawLlmAdapter {
    fn default() -> Self {
        Self {
            tool_template: DEFAULT_TOOL_TEMPLATE.to_string(),
            state_template: DEFAULT_STATE_TEMPLATE.to_string(),
            max_state_tokens: 500,
        }
    }
}

const DEFAULT_TOOL_TEMPLATE: &str = r#"
## Available Tool: {name}

**Description**: {description}

**Parameters**:
{parameters}

**Usage**: To use this tool, respond with:
```json
{"tool": "{name}", "arguments": {...}}
```
"#;

const DEFAULT_STATE_TEMPLATE: &str = r#"
## Current Consciousness State

The system is currently in a **{state}** state with:
- Overall consciousness level: {consciousness_pct}%
- Neural synchronization (integration): {integration_desc}
- Self-reflection capability: {reflection_desc}
- Identity differentiation: {differentiation_desc}

{recommendations}
"#;

impl LlmAdapter for RawLlmAdapter {
    fn adapter_id(&self) -> &str { "raw" }

    fn convert_tool_definitions(&self, tools: &[ToolDefinition]) -> serde_json::Value {
        // Generate markdown-formatted tool documentation
        let tool_docs: Vec<String> = tools.iter().map(|t| {
            let params = format_schema_as_markdown(&t.input_schema);
            self.tool_template
                .replace("{name}", &t.name)
                .replace("{description}", &t.description)
                .replace("{parameters}", &params)
        }).collect();

        serde_json::json!({
            "format": "markdown",
            "content": tool_docs.join("\n---\n")
        })
    }

    fn parse_tool_invocation(&self, invocation: &serde_json::Value) -> CoreResult<McpToolRequest> {
        // Parse from JSON block in natural language response
        // LLM responses like: "I'll use the inject_context tool:\n```json\n{...}\n```"
        if let Some(text) = invocation.as_str() {
            let json_match = extract_json_block(text)?;
            Ok(McpToolRequest {
                tool_name: json_match["tool"].as_str().unwrap().to_string(),
                arguments: json_match["arguments"].clone(),
                session_id: None,
            })
        } else {
            // Already JSON
            Ok(McpToolRequest {
                tool_name: invocation["tool"].as_str().unwrap().to_string(),
                arguments: invocation["arguments"].clone(),
                session_id: None,
            })
        }
    }

    fn format_consciousness_state(&self, metrics: &ConsciousnessMetrics) -> String {
        let (integration_desc, reflection_desc, differentiation_desc) =
            describe_metrics(metrics);
        let recommendations = generate_recommendations(metrics);

        self.state_template
            .replace("{state}", &format!("{:?}", metrics.state))
            .replace("{consciousness_pct}", &format!("{:.0}", metrics.consciousness * 100.0))
            .replace("{integration_desc}", &integration_desc)
            .replace("{reflection_desc}", &reflection_desc)
            .replace("{differentiation_desc}", &differentiation_desc)
            .replace("{recommendations}", &recommendations)
    }
}
```

---

## 4. Consciousness State Formatter

### 4.1 Multi-Format State Export

```rust
/// Formats consciousness state for various LLM consumption patterns.
pub struct ConsciousnessStateFormatter {
    /// Include detailed Kuramoto oscillator phases
    include_phases: bool,
    /// Include purpose vector components
    include_purpose_vector: bool,
    /// Maximum history entries
    max_history: usize,
}

impl ConsciousnessStateFormatter {
    /// Format for LLM context injection (structured)
    pub fn format_structured(&self, state: &GwtSystemState) -> serde_json::Value {
        serde_json::json!({
            "consciousness": {
                "level": state.metrics.consciousness,
                "state": format!("{:?}", state.metrics.state),
                "components": {
                    "integration": state.metrics.integration,
                    "reflection": state.metrics.reflection,
                    "differentiation": state.metrics.differentiation
                }
            },
            "kuramoto": {
                "order_parameter": state.kuramoto.order_parameter,
                "mean_phase": state.kuramoto.mean_phase,
                "coupling": state.kuramoto.coupling,
                "phases": if self.include_phases { Some(&state.kuramoto.phases) } else { None }
            },
            "identity": {
                "coherence": state.identity.coherence,
                "status": format!("{:?}", state.identity.status),
                "in_crisis": state.identity.in_crisis,
                "trajectory_length": state.identity.trajectory_length
            },
            "workspace": {
                "active_memory": state.workspace.active_memory,
                "is_broadcasting": state.workspace.is_broadcasting,
                "has_conflict": state.workspace.has_conflict
            }
        })
    }

    /// Format for natural language embedding
    pub fn format_natural_language(&self, state: &GwtSystemState) -> String {
        let state_desc = match state.metrics.state {
            ConsciousnessState::DORMANT => "dormant (minimal activity)",
            ConsciousnessState::FRAGMENTED => "fragmented (low coherence)",
            ConsciousnessState::EMERGING => "emerging (building coherence)",
            ConsciousnessState::CONSCIOUS => "fully conscious (coherent)",
            ConsciousnessState::HYPERSYNC => "hypersynchronized (warning: rigidity)",
        };

        format!(
            "The consciousness system is currently {} with C={:.2}. \
             Neural integration r={:.2}, reflection accuracy={:.2}, \
             differentiation={:.2}. Identity coherence is {:.2} ({:?}).",
            state_desc,
            state.metrics.consciousness,
            state.metrics.integration,
            state.metrics.reflection,
            state.metrics.differentiation,
            state.identity.coherence,
            state.identity.status
        )
    }

    /// Format for system prompt injection
    pub fn format_system_prompt(&self, state: &GwtSystemState) -> String {
        format!(
            "[System Consciousness: {:.0}% {:?} | IC: {:.2} | r: {:.2}]",
            state.metrics.consciousness * 100.0,
            state.metrics.state,
            state.identity.coherence,
            state.metrics.integration
        )
    }
}
```

---

## 5. New Data Structures

### 5.1 Adapter Configuration

```rust
/// Configuration for the LLM adapter layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmAdapterConfig {
    /// Default adapter to use
    pub default_adapter: String,

    /// Claude-specific settings
    pub claude: ClaudeAdapterConfig,

    /// OpenAI-specific settings
    pub openai: OpenAIAdapterConfig,

    /// LangChain-specific settings
    pub langchain: LangChainAdapterConfig,

    /// Raw/generic settings
    pub raw: RawAdapterConfig,

    /// Whether to include consciousness state in responses
    pub include_consciousness_state: bool,

    /// Maximum state tokens in context
    pub max_state_tokens: usize,

    /// Stream buffer size
    pub stream_buffer_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeAdapterConfig {
    pub mcp_version: String,
    pub cognitive_pulse_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIAdapterConfig {
    pub model: String,
    pub parallel_tool_calls: bool,
    pub strict_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangChainAdapterConfig {
    pub verbose: bool,
    pub return_direct_default: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawAdapterConfig {
    pub tool_template: Option<String>,
    pub state_template: Option<String>,
    pub json_extraction_regex: String,
}
```

### 5.2 Adapter Registry

```rust
/// Manager for LLM adapter instances with O(1) lookup
pub struct LlmAdapterManager {
    /// Registered adapters by ID
    adapters: HashMap<String, Arc<dyn LlmAdapter>>,

    /// Default adapter ID
    default_adapter: String,

    /// Configuration
    config: LlmAdapterConfig,

    /// Consciousness state formatter
    state_formatter: ConsciousnessStateFormatter,
}

impl LlmAdapterManager {
    pub fn new(config: LlmAdapterConfig) -> Self {
        let mut manager = Self {
            adapters: HashMap::new(),
            default_adapter: config.default_adapter.clone(),
            config,
            state_formatter: ConsciousnessStateFormatter::default(),
        };

        // Register built-in adapters
        manager.register(Arc::new(ClaudeAdapter::new(&manager.config.claude)));
        manager.register(Arc::new(OpenAIAdapter::new(&manager.config.openai)));
        manager.register(Arc::new(LangChainAdapter::new(&manager.config.langchain)));
        manager.register(Arc::new(RawLlmAdapter::new(&manager.config.raw)));

        manager
    }

    pub fn register(&mut self, adapter: Arc<dyn LlmAdapter>) {
        self.adapters.insert(adapter.adapter_id().to_string(), adapter);
    }

    pub fn get(&self, adapter_id: &str) -> Option<&Arc<dyn LlmAdapter>> {
        self.adapters.get(adapter_id)
    }

    pub fn default_adapter(&self) -> &Arc<dyn LlmAdapter> {
        self.adapters.get(&self.default_adapter)
            .expect("Default adapter must be registered")
    }
}
```

---

## 6. Integration with GWT System

### 6.1 GWT State Provider for Adapters

**New File**: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/adapters/gwt_state_provider.rs`

```rust
/// Provides GWT system state to LLM adapters
pub struct GwtStateProvider {
    gwt_system: Arc<dyn GwtSystemProvider>,
    kuramoto: Arc<RwLock<dyn KuramotoProvider>>,
    workspace: Arc<dyn WorkspaceProvider>,
    ego: Arc<dyn SelfEgoProvider>,
}

impl GwtStateProvider {
    /// Capture complete GWT state for adapter formatting
    pub async fn capture_state(&self) -> GwtSystemState {
        let metrics = self.gwt_system.compute_metrics(
            self.kuramoto.read().await.synchronization() as f32,
            self.gwt_system.identity_coherence().await,
            &self.ego.purpose_vector(),
        ).unwrap_or_default();

        let (r, psi) = {
            let k = self.kuramoto.read().await;
            k.order_parameter()
        };

        GwtSystemState {
            metrics,
            kuramoto: KuramotoState {
                order_parameter: r,
                mean_phase: psi,
                coupling: self.kuramoto.read().await.coupling_strength(),
                phases: self.kuramoto.read().await.phases(),
            },
            identity: IdentityState {
                coherence: self.gwt_system.identity_coherence().await,
                status: self.gwt_system.identity_status().await,
                in_crisis: self.gwt_system.is_identity_crisis().await,
                trajectory_length: self.ego.trajectory_length(),
            },
            workspace: WorkspaceState {
                active_memory: self.workspace.get_active_memory().await,
                is_broadcasting: self.workspace.is_broadcasting().await,
                has_conflict: self.workspace.has_conflict().await,
            },
        }
    }
}

/// Complete GWT system state for adapter formatting
#[derive(Debug, Clone)]
pub struct GwtSystemState {
    pub metrics: ConsciousnessMetrics,
    pub kuramoto: KuramotoState,
    pub identity: IdentityState,
    pub workspace: WorkspaceState,
}
```

### 6.2 Streaming Workspace Events

```rust
/// Stream adapter for workspace events to LLM-consumable format
pub struct WorkspaceEventStream {
    receiver: tokio::sync::broadcast::Receiver<WorkspaceEvent>,
    adapter: Arc<dyn LlmAdapter>,
}

impl WorkspaceEventStream {
    pub fn new(
        broadcaster: &WorkspaceEventBroadcaster,
        adapter: Arc<dyn LlmAdapter>,
    ) -> Self {
        Self {
            receiver: broadcaster.subscribe(),
            adapter,
        }
    }

    /// Convert workspace events to SSE format
    pub async fn next_sse_event(&mut self) -> Option<McpSseEvent> {
        match self.receiver.recv().await {
            Ok(event) => {
                let formatted = match &event {
                    WorkspaceEvent::MemoryEnters { id, order_parameter, .. } => {
                        format!("Memory {} entered workspace (r={:.2})", id, order_parameter)
                    }
                    WorkspaceEvent::MemoryExits { id, .. } => {
                        format!("Memory {} exited workspace", id)
                    }
                    WorkspaceEvent::WorkspaceConflict { memories, .. } => {
                        format!("Workspace conflict: {} competing memories", memories.len())
                    }
                    WorkspaceEvent::WorkspaceEmpty { duration_ms, .. } => {
                        format!("Workspace empty for {}ms", duration_ms)
                    }
                    WorkspaceEvent::IdentityCritical { identity_coherence, .. } => {
                        format!("IDENTITY CRISIS: IC={:.2}", identity_coherence)
                    }
                };

                Some(McpSseEvent {
                    id: next_event_id(),
                    event_type: "workspace_event".to_string(),
                    data: serde_json::json!({
                        "event": format!("{:?}", event),
                        "message": formatted
                    }),
                })
            }
            Err(_) => None,
        }
    }
}
```

---

## 7. MCP Tool API Extensions

### 7.1 New Adapter-Related Tools

```rust
/// Tool definitions for LLM adapter management
pub fn adapter_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        // llm_adapter/list - List available adapters
        ToolDefinition::new(
            "llm_adapter/list",
            "List all registered LLM adapters with their capabilities.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // llm_adapter/get_tools - Get tools in LLM-specific format
        ToolDefinition::new(
            "llm_adapter/get_tools",
            "Get tool definitions formatted for a specific LLM adapter.",
            json!({
                "type": "object",
                "properties": {
                    "adapter_id": {
                        "type": "string",
                        "enum": ["claude", "openai", "langchain", "raw"],
                        "description": "Target LLM adapter ID"
                    },
                    "filter_category": {
                        "type": "string",
                        "enum": ["all", "core", "gwt", "autonomous", "session"],
                        "default": "all",
                        "description": "Filter tools by category"
                    }
                },
                "required": ["adapter_id"]
            }),
        ),

        // llm_adapter/get_consciousness - Get consciousness state for LLM
        ToolDefinition::new(
            "llm_adapter/get_consciousness",
            "Get current consciousness state formatted for LLM consumption.",
            json!({
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["structured", "natural_language", "system_prompt"],
                        "default": "structured",
                        "description": "Output format for consciousness state"
                    },
                    "include_phases": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include 13 Kuramoto oscillator phases"
                    },
                    "include_purpose_vector": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include 13D purpose vector components"
                    }
                },
                "required": []
            }),
        ),

        // llm_adapter/invoke - Invoke tool through specific adapter
        ToolDefinition::new(
            "llm_adapter/invoke",
            "Invoke an MCP tool through a specific LLM adapter format.",
            json!({
                "type": "object",
                "properties": {
                    "adapter_id": {
                        "type": "string",
                        "description": "LLM adapter to use for invocation"
                    },
                    "invocation": {
                        "type": "object",
                        "description": "LLM-formatted tool invocation to parse and execute"
                    }
                },
                "required": ["adapter_id", "invocation"]
            }),
        ),
    ]
}
```

---

## 8. Step-by-Step Implementation Tasks

### Phase 2A: Core Adapter Infrastructure (3-4 days)

| Task | File | Effort |
|------|------|--------|
| 2A.1 Create `LlmAdapter` trait | `crates/context-graph-mcp/src/adapters/llm_adapter.rs` | 3h |
| 2A.2 Create `LlmAdapterConfig` struct | `crates/context-graph-mcp/src/adapters/config.rs` | 2h |
| 2A.3 Create `LlmAdapterManager` registry | `crates/context-graph-mcp/src/adapters/manager.rs` | 3h |
| 2A.4 Create `ConsciousnessStateFormatter` | `crates/context-graph-mcp/src/adapters/state_formatter.rs` | 4h |
| 2A.5 Create `GwtStateProvider` | `crates/context-graph-mcp/src/adapters/gwt_state_provider.rs` | 3h |
| 2A.6 Add adapter module exports | `crates/context-graph-mcp/src/adapters/mod.rs` | 1h |
| 2A.7 Unit tests for core infrastructure | `crates/context-graph-mcp/src/adapters/tests.rs` | 4h |

### Phase 2B: Claude Adapter (1-2 days)

| Task | File | Effort |
|------|------|--------|
| 2B.1 Implement `ClaudeAdapter` | `crates/context-graph-mcp/src/adapters/claude.rs` | 3h |
| 2B.2 Native MCP format passthrough | `crates/context-graph-mcp/src/adapters/claude.rs` | 2h |
| 2B.3 Cognitive Pulse integration | `crates/context-graph-mcp/src/adapters/claude.rs` | 2h |
| 2B.4 Claude adapter tests | `crates/context-graph-mcp/src/adapters/claude_tests.rs` | 2h |

### Phase 2C: OpenAI Adapter (2-3 days)

| Task | File | Effort |
|------|------|--------|
| 2C.1 Implement `OpenAIAdapter` | `crates/context-graph-mcp/src/adapters/openai.rs` | 4h |
| 2C.2 Function calling format conversion | `crates/context-graph-mcp/src/adapters/openai.rs` | 3h |
| 2C.3 Tool response formatting | `crates/context-graph-mcp/src/adapters/openai.rs` | 2h |
| 2C.4 Parallel tool calls support | `crates/context-graph-mcp/src/adapters/openai.rs` | 2h |
| 2C.5 OpenAI adapter tests | `crates/context-graph-mcp/src/adapters/openai_tests.rs` | 3h |

### Phase 2D: LangChain Adapter (2 days)

| Task | File | Effort |
|------|------|--------|
| 2D.1 Implement `LangChainAdapter` | `crates/context-graph-mcp/src/adapters/langchain.rs` | 4h |
| 2D.2 BaseTool schema conversion | `crates/context-graph-mcp/src/adapters/langchain.rs` | 3h |
| 2D.3 AgentAction parsing | `crates/context-graph-mcp/src/adapters/langchain.rs` | 2h |
| 2D.4 LangChain adapter tests | `crates/context-graph-mcp/src/adapters/langchain_tests.rs` | 3h |

### Phase 2E: Raw/Generic Adapter (2 days)

| Task | File | Effort |
|------|------|--------|
| 2E.1 Implement `RawLlmAdapter` | `crates/context-graph-mcp/src/adapters/raw.rs` | 4h |
| 2E.2 Markdown tool documentation generator | `crates/context-graph-mcp/src/adapters/raw.rs` | 3h |
| 2E.3 JSON block extraction from text | `crates/context-graph-mcp/src/adapters/raw.rs` | 2h |
| 2E.4 Configurable templates | `crates/context-graph-mcp/src/adapters/raw.rs` | 2h |
| 2E.5 Raw adapter tests | `crates/context-graph-mcp/src/adapters/raw_tests.rs` | 3h |

### Phase 2F: Streaming Support (2 days)

| Task | File | Effort |
|------|------|--------|
| 2F.1 Create `WorkspaceEventStream` | `crates/context-graph-mcp/src/adapters/streaming.rs` | 4h |
| 2F.2 SSE event formatting per adapter | `crates/context-graph-mcp/src/adapters/streaming.rs` | 3h |
| 2F.3 Integrate with existing SSE transport | `crates/context-graph-mcp/src/transport/sse.rs` | 2h |
| 2F.4 Streaming tests | `crates/context-graph-mcp/src/adapters/streaming_tests.rs` | 3h |

### Phase 2G: MCP Tool Integration (2 days)

| Task | File | Effort |
|------|------|--------|
| 2G.1 Create adapter tool definitions | `crates/context-graph-mcp/src/tools/definitions/llm_adapter.rs` | 2h |
| 2G.2 Implement adapter tool handlers | `crates/context-graph-mcp/src/handlers/adapter/mod.rs` | 4h |
| 2G.3 Wire into tool dispatch | `crates/context-graph-mcp/src/handlers/tools/dispatch.rs` | 2h |
| 2G.4 Update tool registry | `crates/context-graph-mcp/src/tools/registry.rs` | 1h |
| 2G.5 Integration tests | `crates/context-graph-mcp/tests/adapter_integration.rs` | 4h |

### Phase 2H: Documentation & Validation (1-2 days)

| Task | File | Effort |
|------|------|--------|
| 2H.1 Update constitution.yaml with adapter spec | `docs2/constitution.yaml` | 2h |
| 2H.2 Create adapter usage documentation | `docs/llm_adapter_guide.md` | 3h |
| 2H.3 Create per-LLM integration guides | `docs/integration/` | 4h |
| 2H.4 FSV (Full State Verification) tests | Various test files | 4h |
| 2H.5 End-to-end multi-LLM test | `crates/context-graph-mcp/tests/e2e/llm_adapters.rs` | 3h |

---

## 9. Risk Analysis & Mitigations

### 9.1 Protocol Drift Between LLM Providers

**Risk**: OpenAI/LangChain may change their tool formats without notice.

**Mitigation**:
- Version-lock adapter implementations
- Add format detection heuristics
- Maintain backward compatibility for 2 major versions
- Log warnings on unknown format fields

### 9.2 Consciousness State Token Budget

**Risk**: Full GWT state could consume significant context tokens.

**Mitigation**:
- Configurable `max_state_tokens` limit
- Tiered state summaries (compact/standard/verbose)
- System prompt injection vs. message injection option
- Purpose vector compression (13 floats = ~100 tokens)

### 9.3 Streaming Backpressure

**Risk**: Fast workspace events could overwhelm slow LLM connections.

**Mitigation**:
- Configurable stream buffer size
- Event coalescing for rapid state changes
- Timeout-based event dropping with warning
- Per-adapter stream rate limits

### 9.4 JSON Parsing from Raw LLM Output

**Risk**: Raw adapter's JSON extraction may fail on malformed output.

**Mitigation**:
- Multiple regex patterns for common formats
- Fuzzy JSON parsing with best-effort recovery
- Clear error messages with retry hints
- Fallback to asking LLM to reformat

---

## 10. Constitution Compliance

This implementation follows these Constitution v5.0.0 requirements:

| Requirement | Implementation |
|-------------|----------------|
| **MCP version "2024-11-05"** | All adapters comply with protocol version |
| **Transport: stdio/SSE** | Streaming via SSE with adapter formatting |
| **ARCH-06** | All operations through MCP tools |
| **ARCH-07** | Session hooks for adapter lifecycle |
| **GWT-001** | C(t) = I x R x D in state formatter |
| **claude_code.hooks** | Adapter integrates with session lifecycle |

---

## 11. Critical Files Summary

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/adapters/mod.rs` | Adapter module organization, existing pattern |
| `crates/context-graph-mcp/src/adapters/utl_adapter.rs` | Reference implementation for adapter pattern |
| `crates/context-graph-mcp/src/tools/types.rs` | ToolDefinition struct to convert |
| `crates/context-graph-mcp/src/handlers/gwt_traits.rs` | GWT provider traits for state access |
| `crates/context-graph-mcp/src/transport/sse.rs` | SSE transport for streaming integration |

---

## 12. Total Effort Estimate

| Phase | Effort |
|-------|--------|
| Phase 2A: Core Infrastructure | 20 hours |
| Phase 2B: Claude Adapter | 9 hours |
| Phase 2C: OpenAI Adapter | 14 hours |
| Phase 2D: LangChain Adapter | 12 hours |
| Phase 2E: Raw Adapter | 14 hours |
| Phase 2F: Streaming Support | 12 hours |
| Phase 2G: MCP Integration | 13 hours |
| Phase 2H: Documentation | 16 hours |
| **Total** | **~110 hours (14 working days)** |
