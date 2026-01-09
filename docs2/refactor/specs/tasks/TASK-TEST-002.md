# TASK-TEST-002: Fuzzing Infrastructure

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-TEST-002 |
| **Title** | Fuzzing Infrastructure |
| **Status** | :white_circle: todo |
| **Layer** | Testing |
| **Sequence** | 42 |
| **Estimated Days** | 2 |
| **Complexity** | Medium |

## Implements

- Security testing requirements (SEC-01)
- Input validation verification
- Robustness testing for all parsers

## Dependencies

| Task | Reason |
|------|--------|
| TASK-INTEG-011 | Security input validation to fuzz |
| TASK-INTEG-001 | MCP handlers to fuzz |

## Objective

Implement fuzzing infrastructure using `cargo-fuzz` (libFuzzer) and `afl.rs` to discover:
1. Panic conditions in parsers
2. Buffer overflows in unsafe code
3. Integer overflows
4. Malformed input handling
5. DoS vectors (stack overflow, OOM)

## Context

Fuzzing complements property-based testing by exploring input space more aggressively. It's essential for:
- Finding crashes that unit tests miss
- Discovering security vulnerabilities
- Testing parser robustness
- Verifying input validation completeness

## Scope

### In Scope

- `cargo-fuzz` setup with libFuzzer
- Fuzz targets for MCP tool inputs
- Fuzz targets for TeleologicalArray deserialization
- Fuzz targets for embedding operations
- Corpus management and minimization
- CI integration for continuous fuzzing

### Out of Scope

- AFL.rs setup (optional alternative)
- Distributed fuzzing infrastructure
- Formal verification

## Definition of Done

### Project Structure

```
fuzz/
├── Cargo.toml
├── fuzz_targets/
│   ├── fuzz_mcp_inject_context.rs
│   ├── fuzz_mcp_store_memory.rs
│   ├── fuzz_mcp_search_graph.rs
│   ├── fuzz_array_deserialize.rs
│   ├── fuzz_embedding_ops.rs
│   ├── fuzz_pii_detection.rs
│   └── fuzz_input_validation.rs
├── corpus/
│   ├── inject_context/
│   ├── store_memory/
│   ├── search_graph/
│   └── ...
└── artifacts/
    └── crashes/
```

### Fuzz Cargo.toml

```toml
# fuzz/Cargo.toml

[package]
name = "context-graph-fuzz"
version = "0.0.0"
authors = ["Automated"]
publish = false
edition = "2021"

[package.metadata]
cargo-fuzz = true

[dependencies]
libfuzzer-sys = "0.4"
arbitrary = { version = "1", features = ["derive"] }
context-graph-core = { path = "../crates/context-graph-core" }
context-graph-mcp = { path = "../crates/context-graph-mcp" }
context-graph-storage = { path = "../crates/context-graph-storage" }

[[bin]]
name = "fuzz_mcp_inject_context"
path = "fuzz_targets/fuzz_mcp_inject_context.rs"
test = false
doc = false

[[bin]]
name = "fuzz_mcp_store_memory"
path = "fuzz_targets/fuzz_mcp_store_memory.rs"
test = false
doc = false

[[bin]]
name = "fuzz_mcp_search_graph"
path = "fuzz_targets/fuzz_mcp_search_graph.rs"
test = false
doc = false

[[bin]]
name = "fuzz_array_deserialize"
path = "fuzz_targets/fuzz_array_deserialize.rs"
test = false
doc = false

[[bin]]
name = "fuzz_embedding_ops"
path = "fuzz_targets/fuzz_embedding_ops.rs"
test = false
doc = false

[[bin]]
name = "fuzz_pii_detection"
path = "fuzz_targets/fuzz_pii_detection.rs"
test = false
doc = false

[[bin]]
name = "fuzz_input_validation"
path = "fuzz_targets/fuzz_input_validation.rs"
test = false
doc = false
```

### Fuzz Target: MCP inject_context

```rust
// fuzz/fuzz_targets/fuzz_mcp_inject_context.rs

#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::{Arbitrary, Unstructured};
use context_graph_mcp::handlers::inject_context::InjectContextParams;

/// Arbitrary parameters for inject_context
#[derive(Debug, Arbitrary)]
struct FuzzInjectParams {
    content: String,
    metadata: Vec<(String, String)>,
    importance: Option<f32>,
}

fuzz_target!(|data: &[u8]| {
    // Parse arbitrary input
    let mut u = Unstructured::new(data);
    if let Ok(params) = FuzzInjectParams::arbitrary(&mut u) {
        // Convert to handler params
        let inject_params = InjectContextParams {
            content: params.content,
            metadata: params.metadata.into_iter().collect(),
            importance: params.importance,
        };

        // Validate input (should not panic)
        let _ = inject_params.validate();

        // If validation passes, input should be safe to process
        // (actual processing requires async runtime, skip for now)
    }
});
```

### Fuzz Target: Array Deserialization

```rust
// fuzz/fuzz_targets/fuzz_array_deserialize.rs

#![no_main]

use libfuzzer_sys::fuzz_target;
use context_graph_core::teleology::array::TeleologicalArray;

fuzz_target!(|data: &[u8]| {
    // Try to deserialize arbitrary bytes as TeleologicalArray
    // This should NEVER panic, only return errors for invalid data

    // Bincode deserialization
    let _ = bincode::deserialize::<TeleologicalArray>(data);

    // JSON deserialization (if supported)
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = serde_json::from_str::<TeleologicalArray>(s);
    }

    // MessagePack deserialization (if supported)
    let _ = rmp_serde::decode::from_slice::<TeleologicalArray>(data);
});
```

### Fuzz Target: Embedding Operations

```rust
// fuzz/fuzz_targets/fuzz_embedding_ops.rs

#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::{Arbitrary, Unstructured};
use context_graph_core::teleology::embedder::{EmbedderOutput, SparseVec};

#[derive(Debug, Arbitrary)]
struct FuzzEmbeddingOp {
    op: EmbeddingOp,
    vec1: Vec<f32>,
    vec2: Vec<f32>,
}

#[derive(Debug, Arbitrary)]
enum EmbeddingOp {
    CosineSimilarity,
    DotProduct,
    L2Distance,
    Normalize,
    Add,
    Scale(f32),
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    if let Ok(fuzz) = FuzzEmbeddingOp::arbitrary(&mut u) {
        // Skip empty vectors
        if fuzz.vec1.is_empty() || fuzz.vec2.is_empty() {
            return;
        }

        // Operations should handle any f32 values without panic
        // (including NaN, Inf, subnormals)
        match fuzz.op {
            EmbeddingOp::CosineSimilarity => {
                let _ = cosine_similarity(&fuzz.vec1, &fuzz.vec2);
            }
            EmbeddingOp::DotProduct => {
                let _ = dot_product(&fuzz.vec1, &fuzz.vec2);
            }
            EmbeddingOp::L2Distance => {
                let _ = l2_distance(&fuzz.vec1, &fuzz.vec2);
            }
            EmbeddingOp::Normalize => {
                let _ = normalize_l2(&fuzz.vec1);
            }
            EmbeddingOp::Add => {
                let _ = vec_add(&fuzz.vec1, &fuzz.vec2);
            }
            EmbeddingOp::Scale(s) => {
                let _ = vec_scale(&fuzz.vec1, s);
            }
        }
    }
});

// Stub implementations (actual from context-graph-core)
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 { 0.0 }
fn dot_product(a: &[f32], b: &[f32]) -> f32 { 0.0 }
fn l2_distance(a: &[f32], b: &[f32]) -> f32 { 0.0 }
fn normalize_l2(v: &[f32]) -> Vec<f32> { vec![] }
fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> { vec![] }
fn vec_scale(v: &[f32], s: f32) -> Vec<f32> { vec![] }
```

### Fuzz Target: PII Detection

```rust
// fuzz/fuzz_targets/fuzz_pii_detection.rs

#![no_main]

use libfuzzer_sys::fuzz_target;
use context_graph_mcp::security::pii_detection::{PiiDetector, PiiAction};

fuzz_target!(|data: &[u8]| {
    // PII detector should handle any UTF-8 input
    if let Ok(text) = std::str::from_utf8(data) {
        let detector = PiiDetector::new(PiiAction::Mask);

        // Detection should not panic
        let _ = detector.detect(text);

        // Processing should not panic
        let _ = detector.process(text);

        // contains_pii should not panic
        let _ = detector.contains_pii(text);
    }
});
```

### Fuzz Target: Input Validation

```rust
// fuzz/fuzz_targets/fuzz_input_validation.rs

#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::{Arbitrary, Unstructured};
use context_graph_mcp::security::input_validation::{InputValidator, ValidationConfig};

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    text: String,
    number: i64,
    uuid_str: String,
    json: String,
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    if let Ok(input) = FuzzInput::arbitrary(&mut u) {
        let validator = InputValidator::new(ValidationConfig::default());

        // Text validation should not panic
        let _ = validator.validate_text(&input.text);

        // Number validation should not panic
        let _ = validator.validate_number(input.number, i64::MIN, i64::MAX);

        // UUID validation should not panic
        let _ = validator.validate_uuid(&input.uuid_str);

        // JSON validation should not panic
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&input.json) {
            let _ = validator.validate_json(&json);
        }

        // Sanitize should not panic
        let _ = validator.sanitize(&input.text);
    }
});
```

### CI Integration Script

```bash
#!/bin/bash
# scripts/fuzz.sh

set -e

FUZZ_TIME="${FUZZ_TIME:-60}"  # Default 60 seconds per target
TARGETS=(
    "fuzz_mcp_inject_context"
    "fuzz_mcp_store_memory"
    "fuzz_mcp_search_graph"
    "fuzz_array_deserialize"
    "fuzz_embedding_ops"
    "fuzz_pii_detection"
    "fuzz_input_validation"
)

echo "Starting fuzzing campaign..."

for target in "${TARGETS[@]}"; do
    echo "Fuzzing $target for ${FUZZ_TIME}s..."
    cargo +nightly fuzz run "$target" -- \
        -max_total_time="$FUZZ_TIME" \
        -max_len=65536 \
        -timeout=10 \
        || true  # Continue on crash (report later)
done

# Check for crashes
if ls fuzz/artifacts/*/crash-* 1> /dev/null 2>&1; then
    echo "CRASHES FOUND!"
    ls -la fuzz/artifacts/*/crash-*
    exit 1
fi

echo "No crashes found."
```

### Constraints

| Constraint | Target |
|------------|--------|
| Fuzz time per target (CI) | 60s minimum |
| Fuzz time per target (nightly) | 3600s (1 hour) |
| Max input length | 64KB |
| Timeout per input | 10s |

## Verification

- [ ] All fuzz targets compile with nightly
- [ ] `cargo +nightly fuzz run` works for all targets
- [ ] Corpus directory populated with interesting inputs
- [ ] No crashes found in initial 60s run
- [ ] CI script integrated with GitHub Actions
- [ ] Crash artifacts are captured and reportable

## Files to Create

| File | Purpose |
|------|---------|
| `fuzz/Cargo.toml` | Fuzz crate configuration |
| `fuzz/fuzz_targets/fuzz_mcp_inject_context.rs` | MCP handler fuzzing |
| `fuzz/fuzz_targets/fuzz_mcp_store_memory.rs` | MCP handler fuzzing |
| `fuzz/fuzz_targets/fuzz_mcp_search_graph.rs` | MCP handler fuzzing |
| `fuzz/fuzz_targets/fuzz_array_deserialize.rs` | Deserialization fuzzing |
| `fuzz/fuzz_targets/fuzz_embedding_ops.rs` | Math operation fuzzing |
| `fuzz/fuzz_targets/fuzz_pii_detection.rs` | Security fuzzing |
| `fuzz/fuzz_targets/fuzz_input_validation.rs` | Validation fuzzing |
| `scripts/fuzz.sh` | CI fuzz runner |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Crashes discovered | High | Good! | Fix them |
| Slow fuzzing | Medium | Low | Nightly runs |
| Coverage gaps | Medium | Medium | Monitor coverage |

## Traceability

- Source: Constitution SEC-01 (input validation)
- Related: Security testing requirements
