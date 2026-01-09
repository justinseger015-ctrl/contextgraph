# TASK-CORE-014: Unified Error Type Hierarchy

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-CORE-014 |
| **Title** | Unified Error Type Hierarchy |
| **Status** | :white_circle: todo |
| **Layer** | Foundation |
| **Sequence** | 14 |
| **Estimated Days** | 1.5 |
| **Complexity** | Medium |

## Implements

- **REQ-ERROR-HANDLING-01**: Consistent error handling across crates
- **Coding Standard**: thiserror for library, anyhow for application

## Dependencies

| Task | Reason |
|------|--------|
| None | Foundation task establishing error patterns |

## Objective

Create a unified error type hierarchy using `thiserror` for library errors and `anyhow` for application errors, with proper context chaining.

## Context

The constitution (lines 338-345) specifies:
- Use `thiserror` for library error types
- Use `anyhow` for application-level errors
- Never panic in library code; return Result
- Propagate errors with `?` operator
- Add context with `.context()` or `.with_context()`

Currently, tasks define scattered error types:
- `TASK-LOGIC-005` defines `SearchError`
- `TASK-LOGIC-009` defines `DiscoveryError`
- `TASK-INTEG-001` references `HandlerError`

This task consolidates into a unified hierarchy.

## Scope

### In Scope

- `ContextGraphError` enum as top-level error
- `EmbeddingError` for embedding generation failures
- `StorageError` refinement with comprehensive variants
- `IndexError` for HNSW/inverted index operations
- `ConfigError` for configuration issues
- `GpuError` for CUDA operations
- `McpError` for MCP protocol errors
- Error context macros/helpers
- JSON-RPC error code mapping

### Out of Scope

- Application-level error handling (uses anyhow)
- Panic handling (never panic in library)
- Error metrics/logging (see observability tasks)

## Definition of Done

### Signatures

```rust
// crates/context-graph-core/src/error.rs

use thiserror::Error;
use context_graph_core::teleology::embedder::Embedder;

/// Top-level error type for context-graph library
#[derive(Debug, Error)]
pub enum ContextGraphError {
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("Index error: {0}")]
    Index(#[from] IndexError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),

    #[error("MCP error: {0}")]
    Mcp(#[from] McpError),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl ContextGraphError {
    /// Get JSON-RPC error code for MCP responses
    pub fn error_code(&self) -> i32;

    /// Is this error recoverable?
    pub fn is_recoverable(&self) -> bool;

    /// Should this error be logged at error level?
    pub fn is_critical(&self) -> bool;
}

/// Embedding-related errors
#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("Model not loaded for embedder {0:?}")]
    ModelNotLoaded(Embedder),

    #[error("Embedding generation failed for {embedder:?}: {reason}")]
    GenerationFailed { embedder: Embedder, reason: String },

    #[error("Quantization error: {0}")]
    Quantization(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Batch too large: {size} exceeds max {max}")]
    BatchTooLarge { size: usize, max: usize },

    #[error("Empty input text")]
    EmptyInput,

    #[error("Model warm-up failed: {0}")]
    WarmupFailed(String),
}

/// Storage-related errors
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Array not found: {0}")]
    NotFound(Uuid),

    #[error("Array already exists: {0}")]
    AlreadyExists(Uuid),

    #[error("Incomplete array: missing embedder {0:?}")]
    IncompleteArray(Embedder),

    #[error("Schema migration failed: {0}")]
    Migration(String),

    #[error("Corruption detected: {0}")]
    Corruption(String),

    #[error("Transaction failed: {0}")]
    Transaction(String),
}

/// Index-related errors
#[derive(Debug, Error)]
pub enum IndexError {
    #[error("HNSW index error: {0}")]
    Hnsw(String),

    #[error("Inverted index error: {0}")]
    Inverted(String),

    #[error("Index not found for embedder {0:?}")]
    NotFound(Embedder),

    #[error("Index rebuild required for embedder {0:?}")]
    RebuildRequired(Embedder),

    #[error("Index corruption in embedder {0:?}: {1}")]
    Corruption(Embedder, String),

    #[error("Search timeout after {0}ms")]
    Timeout(u64),
}

/// Configuration errors
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Missing configuration: {0}")]
    Missing(String),

    #[error("Invalid configuration: {field}: {reason}")]
    Invalid { field: String, reason: String },

    #[error("Environment variable not set: {0}")]
    EnvNotSet(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Parse error in {file}: {reason}")]
    ParseError { file: String, reason: String },
}

/// GPU/CUDA errors
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("No GPU available")]
    NotAvailable,

    #[error("GPU out of memory: requested {requested}, available {available}")]
    OutOfMemory { requested: u64, available: u64 },

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Device initialization failed: {0}")]
    InitFailed(String),

    #[error("Kernel launch failed: {0}")]
    KernelFailed(String),
}

/// MCP protocol errors
#[derive(Debug, Error)]
pub enum McpError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Method not found: {0}")]
    MethodNotFound(String),

    #[error("Invalid params: {0}")]
    InvalidParams(String),

    #[error("Rate limited: {0}")]
    RateLimited(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Session expired")]
    SessionExpired,

    #[error("PII detected")]
    PiiDetected,
}

// crates/context-graph-core/src/error.rs (continued)

/// Result type alias for context-graph operations
pub type Result<T> = std::result::Result<T, ContextGraphError>;

/// Extension trait for adding context to errors
pub trait ErrorContext<T> {
    fn context(self, msg: &str) -> Result<T>;
    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T>;
}

impl<T, E: Into<ContextGraphError>> ErrorContext<T> for std::result::Result<T, E> {
    fn context(self, msg: &str) -> Result<T> {
        self.map_err(|e| {
            let err: ContextGraphError = e.into();
            ContextGraphError::Internal(format!("{}: {}", msg, err))
        })
    }

    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T> {
        self.map_err(|e| {
            let err: ContextGraphError = e.into();
            ContextGraphError::Internal(format!("{}: {}", f(), err))
        })
    }
}

// JSON-RPC error code constants
pub mod error_codes {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;

    // Custom error codes (application-specific)
    pub const EMBEDDING_ERROR: i32 = -32001;
    pub const STORAGE_ERROR: i32 = -32002;
    pub const INDEX_ERROR: i32 = -32003;
    pub const GPU_ERROR: i32 = -32004;
    pub const RATE_LIMITED: i32 = -32005;
    pub const UNAUTHORIZED: i32 = -32006;
    pub const PII_DETECTED: i32 = -32007;
    pub const VALIDATION_ERROR: i32 = -32008;
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| No panic in library code | 100% compliance |
| All errors have codes | All variants mapped |
| Context preserved | Via From implementations |

## Verification

- [ ] All error types derive `Error` via `thiserror`
- [ ] All variants have descriptive messages
- [ ] `From` implementations connect error hierarchy
- [ ] JSON-RPC codes mapped for all MCP-facing errors
- [ ] `is_recoverable()` correctly classifies errors
- [ ] No `panic!` or `unwrap()` in library code

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/error.rs` | Main error module |
| Update `crates/context-graph-core/src/lib.rs` | Re-export errors |

## Files to Update

| File | Change |
|------|--------|
| `crates/context-graph-storage/src/error.rs` | Use new StorageError |
| `crates/context-graph-mcp/src/error.rs` | Use new McpError |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing code | Medium | Medium | Deprecation period |
| Missing error cases | Low | Low | Add variants as discovered |

## Traceability

- Source: Constitution coding_standards/error_handling (lines 338-345)
- Anti-Pattern: AP-13 - No .unwrap() in library code
