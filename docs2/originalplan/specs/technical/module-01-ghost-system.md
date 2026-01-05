# Module 1: Ghost System - Technical Specification

```yaml
metadata:
  id: TECH-GHOST-001
  version: 1.0.0
  module: Ghost System
  phase: 0
  status: draft
  created: 2025-12-31
  functional_spec_ref: SPEC-GHOST
  author: Architecture Agent
```

---

## 1. Architecture Overview

### 1.1 Crate Dependency Graph

```
                    +------------------------+
                    |   context-graph-mcp    |
                    |      (binary crate)    |
                    +------------------------+
                               |
              +----------------+----------------+
              |                |                |
              v                v                v
+------------------+  +------------------+  +----------------------+
| context-graph-   |  | context-graph-   |  | context-graph-       |
|     core         |  |   embeddings     |  |     cuda             |
|  (library)       |  |   (library)      |  |   (library)          |
+------------------+  +------------------+  +----------------------+
        ^                     |                       |
        |                     |                       |
        +---------------------+-----------------------+
                    (depends on core)
```

### 1.2 Module Structure

```
context-graph/
├── Cargo.toml                      # Workspace manifest
├── rust-toolchain.toml             # Rust version pinning
├── .cargo/
│   └── config.toml                 # Cargo configuration
├── config/
│   ├── default.toml                # Base configuration
│   ├── development.toml            # Development overrides
│   ├── test.toml                   # Test configuration
│   └── production.toml             # Production settings
├── crates/
│   ├── context-graph-mcp/          # MCP protocol layer
│   ├── context-graph-core/         # Core types and traits
│   ├── context-graph-cuda/         # CUDA kernels (optional)
│   └── context-graph-embeddings/   # Embedding pipeline
├── tests/
│   └── integration/                # Integration tests
└── .github/
    └── workflows/
        └── ci.yml                  # CI/CD pipeline
```

---

## 2. Data Structures

### 2.1 Core Domain Types (context-graph-core)

```rust
// crates/context-graph-core/src/types/mod.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for memory nodes
pub type NodeId = Uuid;

/// Unique identifier for graph edges
pub type EdgeId = Uuid;

/// Embedding vector type (1536 dimensions for OpenAI-compatible)
pub type EmbeddingVector = Vec<f32>;

/// Memory node representing a stored memory unit
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryNode {
    /// Unique identifier
    pub id: NodeId,

    /// Raw content stored in this node
    pub content: String,

    /// Embedding vector (1536D)
    pub embedding: EmbeddingVector,

    /// Semantic importance score [0.0, 1.0]
    pub importance: f32,

    /// Access count for decay calculations
    pub access_count: u64,

    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Soft deletion marker
    pub deleted: bool,

    /// Johari quadrant classification
    pub johari_quadrant: JohariQuadrant,

    /// Source modality
    pub modality: Modality,

    /// Additional metadata
    pub metadata: NodeMetadata,
}

/// Johari Window quadrant classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum JohariQuadrant {
    /// Known to self and others
    Open,
    /// Known to others, unknown to self
    Blind,
    /// Known to self, hidden from others
    Hidden,
    /// Unknown to both self and others
    Unknown,
}

impl Default for JohariQuadrant {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Input modality classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Code,
    Image,
    Audio,
    Structured,
    Mixed,
}

impl Default for Modality {
    fn default() -> Self {
        Self::Text
    }
}

/// Additional node metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct NodeMetadata {
    /// Source identifier (conversation, file, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Content language
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Custom tags
    #[serde(default)]
    pub tags: Vec<String>,

    /// UTL learning score at creation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utl_score: Option<f32>,

    /// Consolidation status
    #[serde(default)]
    pub consolidated: bool,
}

/// Graph edge connecting two memory nodes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphEdge {
    /// Unique edge identifier
    pub id: EdgeId,

    /// Source node ID
    pub source: NodeId,

    /// Target node ID
    pub target: NodeId,

    /// Edge relationship type
    pub edge_type: EdgeType,

    /// Edge weight/strength [0.0, 1.0]
    pub weight: f32,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Edge metadata
    pub metadata: EdgeMetadata,
}

/// Types of relationships between nodes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    /// Semantic similarity
    Semantic,
    /// Temporal sequence
    Temporal,
    /// Causal relationship
    Causal,
    /// Hierarchical (parent-child)
    Hierarchical,
    /// Associative link
    Associative,
    /// Contradiction
    Contradicts,
    /// Supporting evidence
    Supports,
}

/// Edge metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EdgeMetadata {
    /// Confidence in relationship
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,

    /// Explanation of relationship
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
}
```

### 2.2 UTL Types

```rust
// crates/context-graph-core/src/types/utl.rs

use serde::{Deserialize, Serialize};

/// UTL (Unified Theory of Learning) metrics
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UtlMetrics {
    /// Entropy measure [0.0, 1.0]
    pub entropy: f32,

    /// Coherence measure [0.0, 1.0]
    pub coherence: f32,

    /// Computed learning score
    pub learning_score: f32,

    /// Surprise component (delta_S)
    pub surprise: f32,

    /// Coherence change component (delta_C)
    pub coherence_change: f32,

    /// Emotional weight (w_e)
    pub emotional_weight: f32,

    /// Alignment angle cosine (cos phi)
    pub alignment: f32,
}

impl Default for UtlMetrics {
    fn default() -> Self {
        Self {
            entropy: 0.5,
            coherence: 0.5,
            learning_score: 0.0,
            surprise: 0.0,
            coherence_change: 0.0,
            emotional_weight: 1.0,
            alignment: 1.0,
        }
    }
}

/// UTL computation context
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UtlContext {
    /// Prior beliefs/expectations
    pub prior_entropy: f32,

    /// Current system coherence
    pub current_coherence: f32,

    /// Emotional state modifier
    pub emotional_state: EmotionalState,

    /// Goal alignment vector
    pub goal_vector: Option<Vec<f32>>,
}

/// Emotional state for UTL weight computation
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum EmotionalState {
    #[default]
    Neutral,
    Curious,
    Focused,
    Stressed,
    Fatigued,
}
```

### 2.3 Cognitive Pulse Types

```rust
// crates/context-graph-core/src/types/pulse.rs

use serde::{Deserialize, Serialize};

/// Cognitive Pulse header included in all tool responses
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CognitivePulse {
    /// Current entropy level [0.0, 1.0]
    pub entropy: f32,

    /// Current coherence level [0.0, 1.0]
    pub coherence: f32,

    /// Suggested action based on state
    pub suggested_action: SuggestedAction,
}

impl Default for CognitivePulse {
    fn default() -> Self {
        Self {
            entropy: 0.5,
            coherence: 0.5,
            suggested_action: SuggestedAction::Continue,
        }
    }
}

/// Action suggestions based on cognitive state
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SuggestedAction {
    /// System ready, no action needed
    Ready,
    /// Continue current operation
    Continue,
    /// Consider exploring new areas
    Explore,
    /// Focus on consolidating knowledge
    Consolidate,
    /// Reduce complexity, prune low-value nodes
    Prune,
    /// High entropy - needs stabilization
    Stabilize,
    /// Review and verify recent additions
    Review,
}
```

### 2.4 Nervous System Layer Types

```rust
// crates/context-graph-core/src/types/nervous.rs

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Bio-nervous system layer identifier
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum LayerId {
    Sensing,
    Reflex,
    Memory,
    Learning,
    Coherence,
}

impl LayerId {
    /// Get the latency budget for this layer
    pub fn latency_budget(&self) -> Duration {
        match self {
            LayerId::Sensing => Duration::from_millis(5),
            LayerId::Reflex => Duration::from_micros(100),
            LayerId::Memory => Duration::from_millis(1),
            LayerId::Learning => Duration::from_millis(10),
            LayerId::Coherence => Duration::from_millis(10),
        }
    }

    /// Get human-readable layer name
    pub fn display_name(&self) -> &'static str {
        match self {
            LayerId::Sensing => "Sensing Layer",
            LayerId::Reflex => "Reflex Layer",
            LayerId::Memory => "Memory Layer",
            LayerId::Learning => "Learning Layer",
            LayerId::Coherence => "Coherence Layer",
        }
    }
}

/// Input to a nervous system layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInput {
    /// Request identifier for tracing
    pub request_id: String,

    /// Input content
    pub content: String,

    /// Optional embedding (if pre-computed)
    pub embedding: Option<Vec<f32>>,

    /// Context from previous layers
    pub context: LayerContext,
}

/// Context passed between layers
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerContext {
    /// Accumulated latency in microseconds
    pub accumulated_latency_us: u64,

    /// Results from previous layers
    pub layer_results: Vec<LayerResult>,

    /// Current pulse state
    pub pulse: super::CognitivePulse,
}

/// Output from a nervous system layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerOutput {
    /// Layer that produced this output
    pub layer: LayerId,

    /// Processing result
    pub result: LayerResult,

    /// Updated pulse
    pub pulse: super::CognitivePulse,

    /// Processing duration in microseconds
    pub duration_us: u64,
}

/// Result data from layer processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerResult {
    /// Layer identifier
    pub layer: LayerId,

    /// Whether processing was successful
    pub success: bool,

    /// Result data (layer-specific)
    pub data: serde_json::Value,

    /// Optional error message
    pub error: Option<String>,
}
```

### 2.5 MCP Protocol Types

```rust
// crates/context-graph-mcp/src/protocol/types.rs

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC 2.0 request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    pub id: RequestId,
}

/// Request identifier (number or string)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum RequestId {
    Number(i64),
    String(String),
}

/// JSON-RPC 2.0 response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    pub id: RequestId,
}

/// JSON-RPC error object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// MCP initialize result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    pub server_info: ServerInfo,
}

/// Server capabilities advertisement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCapability {
    #[serde(rename = "listChanged")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcesCapability {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subscribe: Option<bool>,
    #[serde(rename = "listChanged")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptsCapability {
    #[serde(rename = "listChanged")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Server information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

/// MCP tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// MCP tool call result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Vec<ToolContent>,
    #[serde(rename = "isError")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// Tool content item
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
    #[serde(rename = "resource")]
    Resource { uri: String, text: Option<String> },
}
```

---

## 3. Core Traits

### 3.1 UTL Processor Trait

```rust
// crates/context-graph-core/src/traits/utl_processor.rs

use crate::types::{MemoryNode, UtlContext, UtlMetrics};
use crate::error::CoreResult;
use async_trait::async_trait;

/// Universal Theory of Learning processor
///
/// Computes learning signals based on the UTL equation:
/// L = f((delta_S x delta_C) . w_e . cos phi)
///
/// Where:
/// - delta_S: Surprise (information gain)
/// - delta_C: Coherence change
/// - w_e: Emotional weight
/// - cos phi: Goal alignment
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_core::traits::UtlProcessor;
///
/// let processor = StubUtlProcessor::new();
/// let score = processor.compute_learning_score("new input", &context).await?;
/// println!("Learning score: {}", score);
/// ```
#[async_trait]
pub trait UtlProcessor: Send + Sync {
    /// Compute the full UTL learning score
    async fn compute_learning_score(
        &self,
        input: &str,
        context: &UtlContext,
    ) -> CoreResult<f32>;

    /// Compute surprise component (delta_S)
    /// Returns value in [0.0, 1.0] representing information gain
    async fn compute_surprise(
        &self,
        input: &str,
        context: &UtlContext,
    ) -> CoreResult<f32>;

    /// Compute coherence change (delta_C)
    /// Positive values indicate increased coherence
    async fn compute_coherence_change(
        &self,
        input: &str,
        context: &UtlContext,
    ) -> CoreResult<f32>;

    /// Compute emotional weight (w_e)
    /// Higher values for emotionally salient content
    async fn compute_emotional_weight(
        &self,
        input: &str,
        context: &UtlContext,
    ) -> CoreResult<f32>;

    /// Compute goal alignment (cos phi)
    /// Range [-1.0, 1.0] where 1.0 is perfect alignment
    async fn compute_alignment(
        &self,
        input: &str,
        context: &UtlContext,
    ) -> CoreResult<f32>;

    /// Determine if a node should be consolidated to long-term memory
    async fn should_consolidate(&self, node: &MemoryNode) -> CoreResult<bool>;

    /// Get full UTL metrics for input
    async fn compute_metrics(
        &self,
        input: &str,
        context: &UtlContext,
    ) -> CoreResult<UtlMetrics>;
}
```

### 3.2 Embedding Provider Trait

```rust
// crates/context-graph-embeddings/src/provider.rs

use async_trait::async_trait;
use thiserror::Error;

/// Embedding generation errors
#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("Input too long: {length} chars exceeds {max} limit")]
    InputTooLong { length: usize, max: usize },

    #[error("Empty input not allowed")]
    EmptyInput,

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

pub type EmbeddingResult<T> = Result<T, EmbeddingError>;

/// Provider for generating embedding vectors
///
/// Abstracts over different embedding models (OpenAI, local models, etc.)
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_embeddings::EmbeddingProvider;
///
/// let provider = StubEmbeddingProvider::new();
/// let embedding = provider.embed("Hello, world!").await?;
/// assert_eq!(embedding.len(), 1536);
/// ```
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding for a single text input
    async fn embed(&self, content: &str) -> EmbeddingResult<Vec<f32>>;

    /// Generate embeddings for multiple texts (batch processing)
    async fn batch_embed(&self, contents: &[String]) -> EmbeddingResult<Vec<Vec<f32>>>;

    /// Return the embedding dimension (e.g., 1536 for text-embedding-ada-002)
    fn embedding_dimension(&self) -> usize;

    /// Return the model identifier
    fn model_id(&self) -> &str;

    /// Maximum input length in characters
    fn max_input_length(&self) -> usize;
}
```

### 3.3 Memory Store Trait

```rust
// crates/context-graph-core/src/traits/memory_store.rs

use crate::types::{MemoryNode, NodeId};
use crate::error::CoreResult;
use async_trait::async_trait;

/// Query options for memory search
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Maximum results to return
    pub top_k: usize,
    /// Minimum similarity threshold [0.0, 1.0]
    pub min_similarity: Option<f32>,
    /// Filter by Johari quadrant
    pub johari_filter: Option<crate::types::JohariQuadrant>,
    /// Filter by modality
    pub modality_filter: Option<crate::types::Modality>,
    /// Include soft-deleted nodes
    pub include_deleted: bool,
}

/// Persistent memory storage abstraction
///
/// Provides CRUD operations for memory nodes with vector search capability.
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_core::traits::MemoryStore;
///
/// let store = InMemoryStore::new();
/// let id = store.store(node).await?;
/// let retrieved = store.retrieve(id).await?;
/// ```
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Store a memory node, returning its ID
    async fn store(&self, node: MemoryNode) -> CoreResult<NodeId>;

    /// Retrieve a node by ID, returns None if not found
    async fn retrieve(&self, id: NodeId) -> CoreResult<Option<MemoryNode>>;

    /// Search for nodes by semantic similarity
    async fn search(
        &self,
        query_embedding: &[f32],
        options: SearchOptions,
    ) -> CoreResult<Vec<(MemoryNode, f32)>>;

    /// Search by text query (embedding computed internally)
    async fn search_text(
        &self,
        query: &str,
        options: SearchOptions,
    ) -> CoreResult<Vec<(MemoryNode, f32)>>;

    /// Delete a node (soft or hard delete)
    async fn delete(&self, id: NodeId, soft: bool) -> CoreResult<bool>;

    /// Update an existing node
    async fn update(&self, node: MemoryNode) -> CoreResult<bool>;

    /// Get total node count
    async fn count(&self) -> CoreResult<usize>;

    /// Compact storage (remove tombstones, optimize indices)
    async fn compact(&self) -> CoreResult<()>;
}
```

### 3.4 Nervous Layer Trait

```rust
// crates/context-graph-core/src/traits/nervous_layer.rs

use crate::types::{LayerId, LayerInput, LayerOutput};
use crate::error::CoreResult;
use async_trait::async_trait;
use std::time::Duration;

/// Bio-nervous system layer abstraction
///
/// Each layer in the 5-layer architecture implements this trait.
/// Layers process input within their latency budget and pass results downstream.
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_core::traits::NervousLayer;
///
/// let layer = ReflexLayer::new();
/// let output = layer.process(input).await?;
/// assert!(output.duration_us < layer.latency_budget().as_micros() as u64);
/// ```
#[async_trait]
pub trait NervousLayer: Send + Sync {
    /// Process input through this layer
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput>;

    /// Get the latency budget for this layer
    fn latency_budget(&self) -> Duration;

    /// Get the layer identifier
    fn layer_id(&self) -> LayerId;

    /// Get human-readable layer name
    fn layer_name(&self) -> &'static str;

    /// Check if layer is healthy and ready
    async fn health_check(&self) -> CoreResult<bool>;
}
```

### 3.5 Graph Index Trait

```rust
// crates/context-graph-core/src/traits/graph_index.rs

use crate::types::NodeId;
use crate::error::CoreResult;
use async_trait::async_trait;

/// Vector index for similarity search
///
/// Abstracts over FAISS, HNSW, or other ANN implementations.
#[async_trait]
pub trait GraphIndex: Send + Sync {
    /// Add a vector to the index
    async fn add(&self, id: NodeId, vector: &[f32]) -> CoreResult<()>;

    /// Search for nearest neighbors
    async fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> CoreResult<Vec<(NodeId, f32)>>;

    /// Remove a vector from the index
    async fn remove(&self, id: NodeId) -> CoreResult<bool>;

    /// Get the dimension of vectors in this index
    fn dimension(&self) -> usize;

    /// Get the number of vectors in the index
    async fn size(&self) -> CoreResult<usize>;

    /// Rebuild the index for optimal search
    async fn rebuild(&self) -> CoreResult<()>;
}
```

---

## 4. Error Handling

### 4.1 Error Type Hierarchy

```rust
// crates/context-graph-core/src/error.rs

use thiserror::Error;

/// Top-level error type for context-graph-core
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Node not found: {id}")]
    NodeNotFound { id: uuid::Uuid },

    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Validation error: {field} - {message}")]
    ValidationError { field: String, message: String },

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Index error: {0}")]
    IndexError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("UTL computation error: {0}")]
    UtlError(String),

    #[error("Layer processing error in {layer}: {message}")]
    LayerError { layer: String, message: String },

    #[error("Feature disabled: {feature}")]
    FeatureDisabled { feature: String },

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type CoreResult<T> = Result<T, CoreError>;

// crates/context-graph-mcp/src/error.rs

use thiserror::Error;

/// MCP layer errors
#[derive(Debug, Error)]
pub enum McpError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Method not found: {method}")]
    MethodNotFound { method: String },

    #[error("Invalid parameters: {param} - {message}")]
    InvalidParams { param: String, message: String },

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Feature disabled: {feature}")]
    FeatureDisabled { feature: String },

    #[error("Payload too large: {size} bytes exceeds {limit} limit")]
    PayloadTooLarge { size: usize, limit: usize },

    #[error("Core error: {0}")]
    Core(#[from] context_graph_core::error::CoreError),
}

impl McpError {
    /// Convert to JSON-RPC error code
    pub fn to_json_rpc_code(&self) -> i32 {
        match self {
            McpError::ParseError(_) => -32700,
            McpError::InvalidRequest(_) => -32600,
            McpError::MethodNotFound { .. } => -32601,
            McpError::InvalidParams { .. } => -32602,
            McpError::Internal(_) => -32603,
            McpError::FeatureDisabled { .. } => -32001,
            McpError::PayloadTooLarge { .. } => -32003,
            McpError::Core(_) => -32603,
        }
    }
}

pub type McpResult<T> = Result<T, McpError>;
```

### 4.2 JSON-RPC Error Codes

| Code | Name | Description |
|------|------|-------------|
| -32700 | Parse error | Invalid JSON received |
| -32600 | Invalid Request | Not a valid JSON-RPC request |
| -32601 | Method not found | Unknown method/tool |
| -32602 | Invalid params | Invalid method parameters |
| -32603 | Internal error | Server internal error |
| -32001 | Feature disabled | Requested feature is disabled |
| -32002 | Config error | Configuration validation failed |
| -32003 | Payload too large | Request exceeds size limit |

---

## 5. Configuration Schema

### 5.1 TOML Configuration Structure

```toml
# config/default.toml

[server]
# Server name for MCP identification
name = "context-graph"
# Server version
version = "0.1.0-ghost"

[mcp]
# Transport type: "stdio" or "sse"
transport = "stdio"
# Maximum payload size in bytes
max_payload_size = 10485760  # 10MB
# Request timeout in seconds
request_timeout = 30

[logging]
# Log level: trace, debug, info, warn, error
level = "info"
# Log format: "pretty" or "json"
format = "pretty"
# Include source location in logs
include_location = false

[storage]
# Storage backend: "memory" or "rocksdb"
backend = "memory"
# Path for persistent storage
path = "./data/storage"
# Enable compression
compression = true

[embedding]
# Embedding model: "stub" or model identifier
model = "stub"
# Embedding dimension
dimension = 1536
# Maximum input length
max_input_length = 8191

[index]
# Index backend: "memory" or "faiss"
backend = "memory"
# Number of neighbors for HNSW
hnsw_m = 16
# Ef construction parameter
hnsw_ef_construction = 200

[utl]
# UTL computation mode: "stub" or "full"
mode = "stub"
# Default emotional weight
default_emotional_weight = 1.0
# Consolidation threshold
consolidation_threshold = 0.7

[features]
# Enable UTL processing
utl_enabled = true
# Enable dream layer (offline consolidation)
dream_enabled = false
# Enable neuromodulation
neuromodulation_enabled = false
# Enable active inference
active_inference_enabled = false
# Enable immune system (anomaly detection)
immune_enabled = false

[cuda]
# Enable CUDA acceleration
enabled = false
# Device ID to use
device_id = 0
# Memory limit in GB
memory_limit_gb = 4.0
```

### 5.2 Configuration Validation

```rust
// crates/context-graph-core/src/config.rs

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub server: ServerConfig,
    pub mcp: McpConfig,
    pub logging: LoggingConfig,
    pub storage: StorageConfig,
    pub embedding: EmbeddingConfig,
    pub index: IndexConfig,
    pub utl: UtlConfig,
    pub features: FeatureFlags,
    pub cuda: CudaConfig,
}

impl Config {
    /// Load configuration from files and environment
    pub fn load() -> Result<Self, ConfigError> {
        let env = std::env::var("CONTEXT_GRAPH_ENV")
            .unwrap_or_else(|_| "development".to_string());

        let builder = config::Config::builder()
            .add_source(config::File::with_name("config/default"))
            .add_source(config::File::with_name(&format!("config/{}", env)).required(false))
            .add_source(
                config::Environment::with_prefix("CONTEXT_GRAPH")
                    .separator("__")
            );

        let config: Config = builder.build()?.try_deserialize()?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate MCP settings
        if self.mcp.max_payload_size == 0 {
            return Err(ConfigError::Validation {
                field: "mcp.max_payload_size".into(),
                message: "must be greater than 0".into(),
            });
        }

        // Validate embedding dimension
        if self.embedding.dimension == 0 {
            return Err(ConfigError::Validation {
                field: "embedding.dimension".into(),
                message: "must be greater than 0".into(),
            });
        }

        // Validate storage path exists for non-memory backends
        if self.storage.backend != "memory" {
            let path = PathBuf::from(&self.storage.path);
            if let Some(parent) = path.parent() {
                if !parent.exists() {
                    return Err(ConfigError::Validation {
                        field: "storage.path".into(),
                        message: format!("parent directory does not exist: {}", parent.display()),
                    });
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpConfig {
    pub transport: String,
    pub max_payload_size: usize,
    pub request_timeout: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub include_location: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StorageConfig {
    pub backend: String,
    pub path: String,
    pub compression: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingConfig {
    pub model: String,
    pub dimension: usize,
    pub max_input_length: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IndexConfig {
    pub backend: String,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UtlConfig {
    pub mode: String,
    pub default_emotional_weight: f32,
    pub consolidation_threshold: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FeatureFlags {
    pub utl_enabled: bool,
    pub dream_enabled: bool,
    pub neuromodulation_enabled: bool,
    pub active_inference_enabled: bool,
    pub immune_enabled: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CudaConfig {
    pub enabled: bool,
    pub device_id: u32,
    pub memory_limit_gb: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Configuration file error: {0}")]
    File(#[from] config::ConfigError),

    #[error("Validation error: {field} - {message}")]
    Validation { field: String, message: String },
}
```

---

## 6. API Contracts

### 6.1 Public API Surface (context-graph-core)

```rust
// crates/context-graph-core/src/lib.rs

//! Context Graph Core Library
//!
//! Provides core domain types, traits, and stub implementations for the
//! Ultimate Context Graph system.

pub mod config;
pub mod error;
pub mod types;
pub mod traits;
pub mod stubs;

// Re-exports for convenience
pub use error::{CoreError, CoreResult};
pub use types::*;
pub use traits::*;
pub use config::Config;
```

### 6.2 Public API Surface (context-graph-embeddings)

```rust
// crates/context-graph-embeddings/src/lib.rs

//! Context Graph Embeddings Library
//!
//! Provides embedding generation abstractions and implementations.

mod provider;
mod stub;

pub use provider::{EmbeddingProvider, EmbeddingError, EmbeddingResult};
pub use stub::StubEmbeddingProvider;

/// Default embedding dimension (OpenAI text-embedding-ada-002 compatible)
pub const DEFAULT_DIMENSION: usize = 1536;

/// Maximum input length in characters
pub const DEFAULT_MAX_INPUT: usize = 8191;
```

### 6.3 Public API Surface (context-graph-cuda)

```rust
// crates/context-graph-cuda/src/lib.rs

//! Context Graph CUDA Library
//!
//! Provides GPU-accelerated operations with CPU fallback stubs.

mod stubs;

#[cfg(feature = "cuda")]
mod kernels;

pub use stubs::*;

/// Check if CUDA is available at runtime
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Check CUDA runtime
        false // Stub: always return false in ghost phase
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}
```

### 6.4 MCP Tool Definitions

```rust
// crates/context-graph-mcp/src/tools/mod.rs

use crate::protocol::ToolDefinition;
use serde_json::json;

/// Get all tool definitions
pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        // Primary Tools
        ToolDefinition {
            name: "inject_context".into(),
            description: "Retrieve and inject relevant context from the knowledge graph".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query for context retrieval"
                    },
                    "token_budget": {
                        "type": "integer",
                        "description": "Maximum tokens to return (default: 2000)"
                    },
                    "distillation": {
                        "type": "string",
                        "enum": ["none", "narrative", "bullets", "summary"],
                        "description": "Distillation mode for context"
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "store_memory".into(),
            description: "Store new information in the knowledge graph".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to store"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Importance score (0-1)"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Tags for categorization"
                    }
                },
                "required": ["content"]
            }),
        },
        ToolDefinition {
            name: "get_memetic_status".into(),
            description: "Get current cognitive state metrics".into(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
        ToolDefinition {
            name: "get_graph_manifest".into(),
            description: "Get system architecture and capabilities manifest".into(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
        ToolDefinition {
            name: "search_graph".into(),
            description: "Search the knowledge graph with advanced filters".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 10)"
                    },
                    "johari_filter": {
                        "type": "string",
                        "enum": ["open", "blind", "hidden", "unknown"],
                        "description": "Filter by Johari quadrant"
                    }
                },
                "required": ["query"]
            }),
        },
        // Add remaining tools...
    ]
}
```

---

## 7. Algorithms

### 7.1 Server Initialization Sequence

```rust
// Initialization sequence pseudocode

async fn initialize_server() -> Result<Server> {
    // 1. Load configuration
    let config = Config::load()?;
    info!("Loaded configuration for environment: {}", config.environment);

    // 2. Initialize tracing/logging
    init_tracing(&config.logging)?;
    info!("Logging initialized with level: {}", config.logging.level);

    // 3. Log feature flags
    log_feature_flags(&config.features);

    // 4. Initialize stub implementations
    let embedding_provider = StubEmbeddingProvider::new(config.embedding.dimension);
    let memory_store = InMemoryStore::new();
    let utl_processor = StubUtlProcessor::new();
    let layers = init_stub_layers();

    // 5. Create service registry
    let services = ServiceRegistry::new(
        embedding_provider,
        memory_store,
        utl_processor,
        layers,
    );

    // 6. Initialize MCP server
    let server = McpServer::new(config.mcp, services);

    // 7. Register tool handlers
    register_tool_handlers(&server)?;

    // 8. Start accepting connections
    info!("Server ready on {} transport", config.mcp.transport);
    Ok(server)
}
```

### 7.2 Request Processing Pipeline

```rust
// Request processing pseudocode

async fn process_request(request: JsonRpcRequest) -> JsonRpcResponse {
    let request_id = generate_request_id();
    let span = info_span!("request", id = %request_id);

    async move {
        // 1. Validate request structure
        if request.jsonrpc != "2.0" {
            return error_response(request.id, McpError::InvalidRequest(
                "jsonrpc must be '2.0'".into()
            ));
        }

        // 2. Route to handler
        let result = match request.method.as_str() {
            "initialize" => handle_initialize(request.params).await,
            "tools/list" => handle_tools_list().await,
            "tools/call" => handle_tool_call(request.params).await,
            method => Err(McpError::MethodNotFound { method: method.into() }),
        };

        // 3. Build response
        match result {
            Ok(value) => success_response(request.id, value),
            Err(e) => error_response(request.id, e),
        }
    }
    .instrument(span)
    .await
}
```

### 7.3 Stub Determinism Algorithm

```rust
// Deterministic mock data generation

fn generate_mock_embedding(content: &str, dimension: usize) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    let seed = hasher.finish();

    // Use seed to generate deterministic vector
    let mut rng = StdRng::seed_from_u64(seed);
    let mut embedding: Vec<f32> = (0..dimension)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // Normalize to unit vector
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut embedding {
        *x /= norm;
    }

    embedding
}

fn generate_mock_uuid(input: &str) -> Uuid {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    let hash = hasher.finish();

    // Create deterministic UUID v4 from hash
    Uuid::from_u64_pair(hash, hash.rotate_left(32))
}
```

---

## 8. Build System

### 8.1 Workspace Cargo.toml

```toml
# Cargo.toml (workspace root)

[workspace]
resolver = "2"
members = [
    "crates/context-graph-mcp",
    "crates/context-graph-core",
    "crates/context-graph-cuda",
    "crates/context-graph-embeddings",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
license = "MIT OR Apache-2.0"
repository = "https://github.com/org/context-graph"

[workspace.dependencies]
# Async runtime
tokio = { version = "1.35", features = ["full"] }
async-trait = "0.1"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"
toml = "0.8"

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }

# Configuration
config = "0.14"

# IDs and time
uuid = { version = "1.6", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }

# Utilities
rand = "0.8"

# Internal crates
context-graph-core = { path = "crates/context-graph-core" }
context-graph-embeddings = { path = "crates/context-graph-embeddings" }
context-graph-cuda = { path = "crates/context-graph-cuda" }

[profile.release]
lto = "thin"
codegen-units = 1
strip = true
```

### 8.2 Crate-specific Cargo.toml

```toml
# crates/context-graph-mcp/Cargo.toml

[package]
name = "context-graph-mcp"
version.workspace = true
edition.workspace = true
rust-version.workspace = true

[[bin]]
name = "context-graph-mcp"
path = "src/main.rs"

[dependencies]
context-graph-core.workspace = true
context-graph-embeddings.workspace = true
context-graph-cuda.workspace = true

tokio.workspace = true
async-trait.workspace = true
serde.workspace = true
serde_json.workspace = true
thiserror.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
config.workspace = true
uuid.workspace = true
chrono.workspace = true

[dev-dependencies]
tokio-test = "0.4"
```

```toml
# crates/context-graph-core/Cargo.toml

[package]
name = "context-graph-core"
version.workspace = true
edition.workspace = true
rust-version.workspace = true

[dependencies]
tokio.workspace = true
async-trait.workspace = true
serde.workspace = true
serde_json.workspace = true
thiserror.workspace = true
tracing.workspace = true
config.workspace = true
uuid.workspace = true
chrono.workspace = true
rand.workspace = true

[dev-dependencies]
tokio = { workspace = true, features = ["test-util"] }
```

```toml
# crates/context-graph-cuda/Cargo.toml

[package]
name = "context-graph-cuda"
version.workspace = true
edition.workspace = true
rust-version.workspace = true

[features]
default = []
cuda = ["cudarc"]

[dependencies]
context-graph-core.workspace = true
tokio.workspace = true
async-trait.workspace = true
thiserror.workspace = true
tracing.workspace = true

cudarc = { version = "0.10", optional = true }

[dev-dependencies]
tokio = { workspace = true, features = ["test-util"] }
```

```toml
# crates/context-graph-embeddings/Cargo.toml

[package]
name = "context-graph-embeddings"
version.workspace = true
edition.workspace = true
rust-version.workspace = true

[dependencies]
context-graph-core.workspace = true
tokio.workspace = true
async-trait.workspace = true
thiserror.workspace = true
tracing.workspace = true
rand.workspace = true

[dev-dependencies]
tokio = { workspace = true, features = ["test-util"] }
```

### 8.3 Rust Toolchain Configuration

```toml
# rust-toolchain.toml

[toolchain]
channel = "1.75"
components = ["rustfmt", "clippy", "rust-src"]
targets = ["x86_64-unknown-linux-gnu"]
```

### 8.4 Cargo Configuration

```toml
# .cargo/config.toml

[build]
rustflags = ["-C", "target-cpu=native"]

[alias]
check-all = "check --workspace --all-targets"
clippy-all = "clippy --workspace --all-targets -- -D warnings"
test-all = "test --workspace"

[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=lld"]
```

---

## 9. Dependencies

### 9.1 Core Dependencies

| Crate | Version | Purpose | License |
|-------|---------|---------|---------|
| tokio | 1.35+ | Async runtime | MIT |
| async-trait | 0.1+ | Async trait support | MIT OR Apache-2.0 |
| serde | 1.0+ | Serialization framework | MIT OR Apache-2.0 |
| serde_json | 1.0+ | JSON serialization | MIT OR Apache-2.0 |
| bincode | 1.3+ | Binary serialization | MIT |
| thiserror | 1.0+ | Error derive macros | MIT OR Apache-2.0 |
| anyhow | 1.0+ | Error context | MIT OR Apache-2.0 |
| tracing | 0.1+ | Structured logging | MIT |
| tracing-subscriber | 0.3+ | Log output | MIT |
| config | 0.14+ | Configuration loading | MIT OR Apache-2.0 |
| uuid | 1.6+ | UUID generation | MIT OR Apache-2.0 |
| chrono | 0.4+ | Date/time handling | MIT OR Apache-2.0 |
| rand | 0.8+ | Random number generation | MIT OR Apache-2.0 |
| toml | 0.8+ | TOML parsing | MIT OR Apache-2.0 |

### 9.2 Optional Dependencies

| Crate | Version | Purpose | Feature Flag |
|-------|---------|---------|--------------|
| cudarc | 0.10+ | CUDA bindings | cuda |

### 9.3 Dev Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| tokio-test | 0.4+ | Async test utilities |
| criterion | 0.5+ | Benchmarking |
| proptest | 1.4+ | Property-based testing |

---

## 10. Testing Strategy

### 10.1 Unit Test Structure

```rust
// Example unit test for StubEmbeddingProvider

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embed_returns_correct_dimension() {
        let provider = StubEmbeddingProvider::new(1536);
        let embedding = provider.embed("test content").await.unwrap();
        assert_eq!(embedding.len(), 1536);
    }

    #[tokio::test]
    async fn test_embed_is_deterministic() {
        let provider = StubEmbeddingProvider::new(1536);
        let embedding1 = provider.embed("test").await.unwrap();
        let embedding2 = provider.embed("test").await.unwrap();
        assert_eq!(embedding1, embedding2);
    }

    #[tokio::test]
    async fn test_embed_different_inputs_different_outputs() {
        let provider = StubEmbeddingProvider::new(1536);
        let embedding1 = provider.embed("hello").await.unwrap();
        let embedding2 = provider.embed("world").await.unwrap();
        assert_ne!(embedding1, embedding2);
    }

    #[tokio::test]
    async fn test_batch_embed() {
        let provider = StubEmbeddingProvider::new(1536);
        let inputs = vec!["a".into(), "b".into(), "c".into()];
        let embeddings = provider.batch_embed(&inputs).await.unwrap();
        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 1536);
        }
    }
}
```

### 10.2 Integration Test Structure

```rust
// tests/integration/mcp_protocol_test.rs

use context_graph_mcp::server::McpServer;
use serde_json::{json, Value};

#[tokio::test]
async fn test_initialize_handshake() {
    let server = create_test_server().await;

    let request = json!({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1,
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "test-client", "version": "1.0" }
        }
    });

    let response = server.process_request(request).await;

    assert!(response["result"]["protocolVersion"].is_string());
    assert_eq!(response["result"]["protocolVersion"], "2024-11-05");
    assert!(response["result"]["capabilities"]["tools"].is_object());
}

#[tokio::test]
async fn test_tools_list() {
    let server = create_test_server().await;

    let request = json!({
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 2
    });

    let response = server.process_request(request).await;
    let tools = response["result"]["tools"].as_array().unwrap();

    assert!(tools.len() >= 5);

    // Verify inject_context tool exists
    let inject = tools.iter()
        .find(|t| t["name"] == "inject_context")
        .expect("inject_context tool should exist");

    assert!(inject["inputSchema"]["properties"]["query"].is_object());
}

#[tokio::test]
async fn test_tool_call_inject_context() {
    let server = create_test_server().await;

    let request = json!({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 3,
        "params": {
            "name": "inject_context",
            "arguments": {
                "query": "test query"
            }
        }
    });

    let response = server.process_request(request).await;
    let content = &response["result"]["content"][0]["text"];
    let result: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();

    // Verify response structure
    assert!(result["context"].is_string());
    assert!(result["pulse"]["entropy"].is_number());
    assert!(result["pulse"]["coherence"].is_number());
    assert!(result["pulse"]["suggested_action"].is_string());
}
```

---

## 11. CI/CD Pipeline

```yaml
# .github/workflows/ci.yml

name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  check:
    name: Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable
        with:
          components: rustfmt, clippy

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Check formatting
        run: cargo fmt --all --check

      - name: Clippy
        run: cargo clippy --workspace --all-targets -- -D warnings

      - name: Build
        run: cargo build --workspace --release

  test:
    name: Test
    runs-on: ubuntu-latest
    needs: check
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Run tests
        run: cargo test --workspace --all-targets

      - name: Run doc tests
        run: cargo test --workspace --doc

  coverage:
    name: Coverage
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable
        with:
          components: llvm-tools-preview

      - name: Install cargo-llvm-cov
        uses: taiki-e/install-action@cargo-llvm-cov

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Generate coverage
        run: cargo llvm-cov --workspace --lcov --output-path lcov.info

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: lcov.info
          fail_ci_if_error: false

  docs:
    name: Documentation
    runs-on: ubuntu-latest
    needs: check
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Build docs
        run: cargo doc --workspace --no-deps
        env:
          RUSTDOCFLAGS: -D warnings
```

---

## 12. Traceability Matrix

| Technical Req | Functional Req | Implementation Location |
|---------------|----------------|-------------------------|
| Workspace structure | REQ-GHOST-001 | Cargo.toml (root) |
| MCP binary entry | REQ-GHOST-002 | context-graph-mcp/src/main.rs |
| Core types/traits | REQ-GHOST-003 | context-graph-core/src/lib.rs |
| CUDA stubs | REQ-GHOST-004 | context-graph-cuda/src/stubs.rs |
| Embedding trait | REQ-GHOST-005 | context-graph-embeddings/src/provider.rs |
| Zero warnings | REQ-GHOST-006 | .github/workflows/ci.yml |
| UTLProcessor trait | REQ-GHOST-007 | context-graph-core/src/traits/utl_processor.rs |
| EmbeddingProvider | REQ-GHOST-008 | context-graph-embeddings/src/provider.rs |
| MemoryStore trait | REQ-GHOST-009 | context-graph-core/src/traits/memory_store.rs |
| NervousLayer trait | REQ-GHOST-010 | context-graph-core/src/traits/nervous_layer.rs |
| JSON-RPC 2.0 | REQ-GHOST-011 | context-graph-mcp/src/protocol/ |
| MCP initialize | REQ-GHOST-012 | context-graph-mcp/src/handlers/initialize.rs |
| Tool handlers | REQ-GHOST-013 | context-graph-mcp/src/handlers/ |
| Cognitive Pulse | REQ-GHOST-014 | context-graph-core/src/types/pulse.rs |
| Error handling | REQ-GHOST-015 | context-graph-mcp/src/error.rs |
| TOML config | REQ-GHOST-016 | config/*.toml |
| Env overrides | REQ-GHOST-017 | context-graph-core/src/config.rs |
| Feature flags | REQ-GHOST-018 | config/default.toml [features] |
| Config validation | REQ-GHOST-019 | context-graph-core/src/config.rs |
| Tracing | REQ-GHOST-020 | context-graph-mcp/src/main.rs |
| Log structure | REQ-GHOST-021 | Tracing subscriber setup |
| Log format | REQ-GHOST-022 | context-graph-core/src/config.rs |
| Error context | REQ-GHOST-023 | thiserror derive macros |
| CI pipeline | REQ-GHOST-024 | .github/workflows/ci.yml |
| Coverage | REQ-GHOST-025 | .github/workflows/ci.yml |
| Integration tests | REQ-GHOST-026 | tests/integration/ |
| Determinism | REQ-GHOST-027 | Stub implementations |
| Schema compliance | REQ-GHOST-028 | Stub implementations |
| Graph manifest | REQ-GHOST-029 | context-graph-mcp/src/handlers/manifest.rs |
| inject_context stub | REQ-GHOST-030 | context-graph-mcp/src/handlers/inject_context.rs |
| Rustdoc | REQ-GHOST-031 | All public APIs |

---

*Document generated: 2025-12-31*
*Technical Specification Version: 1.0.0*
*Module: Ghost System (Phase 0)*
