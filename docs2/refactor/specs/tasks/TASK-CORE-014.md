# TASK-CORE-014: Unified Error Type Hierarchy

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-CORE-014 |
| **Title** | Unified Error Type Hierarchy |
| **Status** | :green_circle: COMPLETED |
| **Layer** | Foundation |
| **Sequence** | 14 |
| **Completed Date** | 2026-01-09 |
| **Verified** | YES - 46 tests pass, cargo build success |

## Implements

- **Constitution rust_standards/error_handling (lines 136-141)**
- **AP-14**: No `.unwrap()` in library code

---

## COMPLETED IMPLEMENTATION SUMMARY

### Files Modified

| File | Changes |
|------|---------|
| `crates/context-graph-core/src/error.rs` | Added `ContextGraphError`, `EmbeddingError`, `StorageError`, `IndexError`, `ConfigError`, `GpuError`, `McpError`, `Result<T>` alias, `From` implementations |
| `crates/context-graph-core/src/lib.rs` | Re-exports: `ContextGraphError`, `EmbeddingError`, `StorageError`, `IndexError`, `ConfigError`, `GpuError`, `McpError`, `Result` |
| `crates/context-graph-mcp/src/protocol.rs` | Added `INDEX_ERROR = -32008`, `GPU_ERROR = -32009` error codes |

### Implementation Details

**1. Unified Error Hierarchy** (`crates/context-graph-core/src/error.rs`):

```rust
pub enum ContextGraphError {
    Embedding(EmbeddingError),   // -32005
    Storage(StorageError),       // -32004
    Index(IndexError),           // -32008
    Config(ConfigError),         // -32603
    Gpu(GpuError),              // -32009
    Mcp(McpError),              // varies
    Validation(String),          // -32602
    Internal(String),            // -32603
}
```

**2. Helper Methods**:
- `error_code()` -> JSON-RPC error code
- `is_recoverable()` -> Can retry (ModelNotLoaded, Transaction, Timeout, RateLimited, OOM)
- `is_critical()` -> System health issue (Corruption, NotAvailable, Internal)
- `internal(msg)`, `validation(msg)` -> Convenience constructors

**3. Sub-Error Types**:
- `EmbeddingError`: ModelNotLoaded, GenerationFailed, Quantization, DimensionMismatch, BatchTooLarge, EmptyInput, WarmupFailed, ModelNotFound, TensorError
- `StorageError`: Database, Serialization, NotFound, AlreadyExists, IncompleteArray, Migration, Corruption, Transaction, WriteFailed, ReadFailed
- `IndexError`: Hnsw, Inverted, NotFound, RebuildRequired, Corruption, Timeout, ConstructionFailed, InsertionFailed
- `ConfigError`: Missing, Invalid, EnvNotSet, FileNotFound, ParseError
- `GpuError`: NotAvailable, OutOfMemory, CudaError, InitFailed, KernelFailed
- `McpError`: InvalidRequest, MethodNotFound, InvalidParams, RateLimited, Unauthorized, SessionExpired, PiiDetected

**4. From Implementations**:
- `From<EmbeddingError> for ContextGraphError`
- `From<StorageError> for ContextGraphError`
- `From<IndexError> for ContextGraphError`
- `From<ConfigError> for ContextGraphError`
- `From<GpuError> for ContextGraphError`
- `From<McpError> for ContextGraphError`
- `From<crate::index::error::IndexError> for IndexError`
- `From<crate::index::error::IndexError> for ContextGraphError`
- `From<CoreError> for ContextGraphError` (legacy interop)

**5. Result Alias**:
```rust
pub type Result<T> = std::result::Result<T, ContextGraphError>;
```

**6. JSON-RPC Codes** (`protocol.rs` lines 103-104):
```rust
pub const INDEX_ERROR: i32 = -32008;
pub const GPU_ERROR: i32 = -32009;
```

---

## Verification Evidence

### Test Results (2026-01-09)

```
cargo test --package context-graph-core error:: -- --nocapture

running 46 tests
test error::tests::edge_case_all_embedders_in_errors ... ok
test error::tests::edge_case_empty_validation_message ... ok
test error::tests::edge_case_unicode_in_error ... ok
test error::tests::edge_case_nested_from_conversion ... ok
test error::tests::edge_case_zero_values ... ok
test error::tests::test_context_graph_error_codes ... ok
test error::tests::test_convenience_constructors ... ok
test error::tests::test_core_error_dimension_mismatch ... ok
test error::tests::test_core_error_display ... ok
test error::tests::test_core_to_context_graph_conversion ... ok
test error::tests::test_is_critical ... ok
test error::tests::test_is_recoverable ... ok
test error::tests::test_mcp_error_codes ... ok
... (46 total tests pass)

test result: ok. 46 passed; 0 failed
```

### Build Verification

```
cargo build -p context-graph-core
   Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.06s
```

### Exports Verification

Re-exports in `lib.rs`:
```rust
pub use error::{
    ConfigError, ContextGraphError, EmbeddingError, GpuError,
    IndexError, McpError, Result, StorageError,
};
```

---

## Source of Truth

| Verification | Location | Status |
|--------------|----------|--------|
| Error types exported | `crates/context-graph-core/src/lib.rs` | VERIFIED |
| All From impls work | Unit tests `test_core_to_context_graph_conversion` | VERIFIED |
| JSON-RPC codes correct | Unit tests `test_context_graph_error_codes`, `test_mcp_error_codes` | VERIFIED |
| is_recoverable() correct | Unit tests `test_is_recoverable` | VERIFIED |
| is_critical() correct | Unit tests `test_is_critical` | VERIFIED |
| Protocol codes added | `crates/context-graph-mcp/src/protocol.rs:103-104` | VERIFIED |

---

## Edge Cases Tested

| Edge Case | Test | Result |
|-----------|------|--------|
| Empty error message | `edge_case_empty_validation_message` | PASS - displays "Validation error: " |
| Unicode in messages | `edge_case_unicode_in_error` | PASS - Unicode preserved |
| Nested From conversion | `edge_case_nested_from_conversion` | PASS - message preserved |
| All 13 embedders | `edge_case_all_embedders_in_errors` | PASS - all embedders work |
| Zero/nil values | `edge_case_zero_values` | PASS - handles Uuid::nil(), 0ms, 0 bytes |

---

## Usage Examples

```rust
use context_graph_core::{ContextGraphError, EmbeddingError, Result, Embedder};

fn generate_embedding(text: &str) -> Result<Vec<f32>> {
    if text.is_empty() {
        return Err(ContextGraphError::Embedding(EmbeddingError::EmptyInput));
    }
    Ok(vec![0.0; 1024])
}

fn handle_error(err: ContextGraphError) {
    let code = err.error_code();      // -32005 for EmbeddingError
    let retry = err.is_recoverable(); // true for ModelNotLoaded
    let alert = err.is_critical();    // true for Corruption

    if alert {
        eprintln!("CRITICAL: {}", err);
    }
}
```

---

## Traceability

| Requirement | Implementation |
|-------------|----------------|
| Constitution error_handling (136-141) | `thiserror` derives, Result return, `?` propagation |
| AP-14 no unwrap | No `.unwrap()` in new code |
| JSON-RPC codes | `error_code()` method, protocol.rs constants |
| Fail-fast | No fallbacks, all errors propagate |

---

## Checklist

- [x] All error types derive `Error` via `thiserror`
- [x] All variants have descriptive messages with context
- [x] `From` implementations connect error hierarchy
- [x] JSON-RPC codes mapped (INDEX_ERROR=-32008, GPU_ERROR=-32009 added)
- [x] `is_recoverable()` correctly classifies errors
- [x] `is_critical()` correctly identifies system health issues
- [x] No `panic!` or `unwrap()` in new library code
- [x] Builds without errors
- [x] All 46 tests pass
- [x] Legacy `CoreError` retained for backwards compatibility
- [x] `Result<T>` type alias exported
