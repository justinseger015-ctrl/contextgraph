---
id: "M04-T08"
title: "Define GraphError Enum"
description: |
  TASK STATUS: ✅ VERIFIED COMPLETE (2026-01-03)
  GraphError enum has 32 variants including StorageOpen.
  static_assertions::assert_impl_all! verifies Send+Sync+Error bounds at compile time.
  All 22 tests pass. Sherlock-Holmes verified.
layer: "foundation"
status: "completed"
priority: "high"
estimated_hours: 0.5
actual_hours: 0.5
completed_date: "2026-01-03"
sequence: 11
depends_on:
  - "M04-T00"
spec_refs:
  - "TECH-GRAPH-004 Section 9"
files_to_create: []
files_to_modify:
  - path: "crates/context-graph-graph/src/error.rs"
    description: "✅ StorageOpen variant added at line 67, static_assertions at line 190"
  - path: "crates/context-graph-graph/Cargo.toml"
    description: "✅ static_assertions = 1.1 added at line 28"
test_file: "crates/context-graph-graph/src/error.rs (inline #[cfg(test)])"
---

## ✅ IMPLEMENTATION COMPLETE

### Verified State (2026-01-03)

**File**: `crates/context-graph-graph/src/error.rs` (463 lines)

**GraphError enum has 32 variants:**
1. `FaissIndexCreation(String)` - line 29
2. `FaissTrainingFailed(String)` - line 33
3. `FaissSearchFailed(String)` - line 37
4. `FaissAddFailed(String)` - line 41
5. `IndexNotTrained` - line 45
6. `InsufficientTrainingData { required, provided }` - line 49
7. `GpuResourceAllocation(String)` - line 54
8. `GpuTransferFailed(String)` - line 58
9. `GpuDeviceUnavailable(String)` - line 62
10. **`StorageOpen { path, cause }`** - line 67 ✅ NEW
11. `Storage(String)` - line 71
12. `ColumnFamilyNotFound(String)` - line 75
13. `CorruptedData { location, details }` - line 79
14. `MigrationFailed(String)` - line 83
15. `InvalidConfig(String)` - line 88
16. `DimensionMismatch { expected, actual }` - line 92
17. `NodeNotFound(String)` - line 97
18. `EdgeNotFound(String, String)` - line 101
19. `DuplicateNode(String)` - line 105
20. `InvalidHyperbolicPoint { norm }` - line 110
21. `InvalidCurvature(f32)` - line 114
22. `MobiusOperationFailed(String)` - line 118
23. `InvalidAperture(f32)` - line 123
24. `ZeroConeAxis` - line 127
25. `PathNotFound(String, String)` - line 132
26. `DepthLimitExceeded(usize)` - line 136
27. `CycleDetected(String)` - line 140
28. `VectorIdMismatch(String)` - line 145
29. `InvalidNtWeights { field, value }` - line 149
30. `Serialization(String)` - line 154
31. `Deserialization(String)` - line 158
32. `Io(#[from] std::io::Error)` - line 163

**static_assertions**: Line 190 - `static_assertions::assert_impl_all!(GraphError: Send, Sync, std::error::Error);`

**Tests**: 22 tests pass (including 5 new tests for M04-T08)

### Verification Evidence

```bash
# Build: SUCCESS
$ cargo build -p context-graph-graph
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.08s

# Tests: 22 PASSED
$ cargo test -p context-graph-graph error -- --nocapture
test result: ok. 22 passed; 0 failed; 0 ignored

# StorageOpen exists:
$ grep -n "StorageOpen" crates/context-graph-graph/src/error.rs
67:    StorageOpen { path: String, cause: String },
290:        let err = GraphError::StorageOpen {
...

# static_assertions exists:
$ grep -n "assert_impl_all" crates/context-graph-graph/src/error.rs
190:static_assertions::assert_impl_all!(GraphError: Send, Sync, std::error::Error);
```

### Edge Cases Verified

1. **Empty path**: `StorageOpen { path: "", cause: "invalid" }` → "Failed to open storage at : invalid" ✅
2. **Unicode**: Russian/Chinese characters in path/cause display correctly ✅
3. **Long path**: 10000 character path preserved in message ✅

## Acceptance Criteria (All Met)

- [x] StorageOpen variant added with path and cause fields
- [x] static_assertions crate added to dependencies (Cargo.toml line 28)
- [x] assert_impl_all! macro verifies Send + Sync + Error (line 190)
- [x] All 32 variants compile
- [x] `cargo build -p context-graph-graph` succeeds
- [x] `cargo test -p context-graph-graph error` - 22 tests pass
- [x] `cargo clippy -p context-graph-graph --no-deps -- -D warnings` - no warnings in graph crate

## Source of Truth

- **File**: `crates/context-graph-graph/src/error.rs` (463 lines)
- **Cargo.toml**: `crates/context-graph-graph/Cargo.toml` (line 28: `static_assertions = "1.1"`)

## No Further Action Required

This task is complete. Proceed to M04-T09 (FAISS FFI Bindings).
