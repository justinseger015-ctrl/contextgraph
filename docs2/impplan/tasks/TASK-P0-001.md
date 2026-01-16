# TASK-P0-001: Remove North Star Alignment System

## Critical Context for AI Agent

**IMPORTANT**: The original task document was WRONG. This corrected version reflects the actual codebase structure.

### Scope Reality Check (VERIFIED 2026-01-16)
- **817** references to `north_star`/`NorthStar` across the codebase
- **120** references to `theta_to_north_star` field
- This is a **MAJOR REFACTOR**, not a simple deletion

The "North Star" is NOT a separate directory/module. It is:
1. An **enum variant** `GoalLevel::NorthStar` in the goal hierarchy
2. A **field** `theta_to_north_star` in `TeleologicalFingerprint`
3. Various **methods** throughout the codebase (`is_north_star()`, `has_north_star()`, etc.)
4. **Error types** in `GoalHierarchyError` (NoNorthStar, MultipleNorthStars)
5. **Constraints** in `GoalHierarchy::validate()` that require North Star

### DO NOT DELETE (Identity/Consciousness System)
- `ego_node` module - Contains identity continuity, NOT North Star
- `drift` module - Monitors alignment drift, will be refactored to TopicDrift
- `SelfEgoNode` - System identity tracking
- `IdentityContinuityMonitor` - IC monitoring (AP-26, AP-37, AP-38)

## Task Objective

Remove the manual North Star goal-setting system (ARCH-03 violation) and its alignment tracking (`theta_to_north_star`). The system should rely ONLY on autonomous emergent topic discovery (Phase 4) for goal patterns.

## Codebase Structure (VERIFIED 2026-01-16)

```
crates/
├── context-graph-core/
│   └── src/
│       ├── autonomous/
│       │   ├── mod.rs              # Re-exports drift, services, etc.
│       │   ├── bootstrap.rs        # North Star bootstrap types
│       │   ├── drift/              # Drift detection (KEEP - refactor to TopicDrift)
│       │   │   ├── mod.rs
│       │   │   ├── detector.rs     # DriftDetector impl
│       │   │   ├── history.rs
│       │   │   ├── types.rs
│       │   │   └── tests.rs
│       │   └── services/
│       │       ├── drift_detector/ # DriftDetector service (KEEP - refactor)
│       │       └── drift_corrector/ # DriftCorrector service (KEEP - refactor)
│       ├── gwt/
│       │   └── ego_node/           # KEEP - this is identity continuity, NOT North Star
│       │       ├── mod.rs
│       │       ├── self_ego_node.rs
│       │       ├── identity_continuity.rs
│       │       ├── crisis_protocol.rs
│       │       └── tests/
│       └── purpose/
│           └── goals/
│               ├── level.rs        # Contains GoalLevel::NorthStar enum
│               ├── node.rs         # GoalNode with is_north_star() method
│               └── hierarchy.rs    # GoalHierarchy with north_star() method
├── context-graph-embeddings/
│   └── src/storage/types/
│       └── fingerprint.rs          # Contains theta_to_north_star field
├── context-graph-mcp/
│   └── src/handlers/
│       ├── autonomous/
│       │   ├── bootstrap.rs        # auto_bootstrap_north_star handler
│       │   ├── drift.rs            # Drift-related handlers
│       │   └── mod.rs
│       └── purpose/
│           ├── drift.rs            # Purpose drift handlers
│           └── hierarchy.rs        # Goal hierarchy handlers
└── context-graph-storage/
    └── src/teleological/           # Uses theta_to_north_star in storage
```

## What to Remove vs What to Keep

### REMOVE (North Star Alignment System)
| Component | Location | Reason |
|-----------|----------|--------|
| `theta_to_north_star` field | `fingerprint.rs:58` | Manual alignment metric |
| `GoalLevel::NorthStar` enum variant | `level.rs:20` | Manual goal level |
| `is_north_star()` method | `node.rs` | Manual goal check |
| `has_north_star()` method | `hierarchy.rs` | Manual goal check |
| `north_star()` method | `hierarchy.rs` | Manual goal accessor |
| `path_to_north_star()` method | `hierarchy.rs` | Manual goal path |
| `auto_bootstrap_north_star` tool | `handlers/autonomous/bootstrap.rs` | Bootstraps manual goals |
| North Star constitution rules | `constitution.yaml` | AP-01, etc. |

### KEEP (Identity/Consciousness System - NOT North Star)
| Component | Location | Reason |
|-----------|----------|--------|
| `ego_node` module | `gwt/ego_node/` | Identity continuity, NOT North Star |
| `SelfEgoNode` struct | `ego_node/self_ego_node.rs` | System identity tracking |
| `IdentityContinuityMonitor` | `ego_node/identity_continuity.rs` | IC monitoring |
| `CrisisProtocol` | `ego_node/crisis_protocol.rs` | Crisis handling |
| Drift detection (refactored) | `autonomous/drift/` | Rename to TopicDrift later |

## Execution Steps

### Phase 1: Remove theta_to_north_star Field

**File**: `crates/context-graph-embeddings/src/storage/types/fingerprint.rs`

```rust
// BEFORE (line 58)
pub theta_to_north_star: f32,

// AFTER
// REMOVED: theta_to_north_star per TASK-P0-001 (ARCH-03)
```

**Impact Analysis**:
- Update `StoredQuantizedFingerprint::new()` to not compute this field
- Update serialization format (version bump required)
- Update all tests that reference this field
- Update storage layer serialization

### Phase 2: Remove GoalLevel::NorthStar

**File**: `crates/context-graph-core/src/purpose/goals/level.rs`

```rust
// REMOVE the NorthStar variant, keep Strategic as top level
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GoalLevel {
    /// Top-level strategic objectives (was NorthStar).
    Strategic = 0,
    /// Short-term tactical goals.
    Tactical = 1,
    /// Immediate context goals.
    Immediate = 2,
}
```

**Impact Analysis**:
- All code referencing `GoalLevel::NorthStar` must be updated
- Serialization format changes (version bump)
- Test fixtures must be updated

### Phase 3: Remove North Star Methods from GoalHierarchy

**File**: `crates/context-graph-core/src/purpose/goals/hierarchy.rs`

Remove methods:
- `has_north_star(&self) -> bool`
- `north_star(&self) -> Option<&GoalNode>`
- `path_to_north_star(&self, goal_id: &Uuid) -> Vec<Uuid>`

### Phase 4: Remove auto_bootstrap_north_star MCP Tool

**File**: `crates/context-graph-mcp/src/handlers/autonomous/bootstrap.rs`

Remove entire `call_auto_bootstrap_north_star` function and update router.

### Phase 5: Update Constitution

**File**: `docs2/constitution.yaml`

Remove/update:
- `ARCH-03` comment about North Star
- Any references to `north_star` in MCP tools
- Goal hierarchy references to NorthStar level

## Full State Verification Protocol

### Source of Truth
- **Primary**: RocksDB column families (`teleological`, `memories`)
- **Secondary**: In-memory goal hierarchy

### Pre-Execution Verification
```bash
# Count current North Star references
grep -rn "north_star\|NorthStar" crates/ --include="*.rs" | wc -l
# Record result: Expected > 100

# Count theta_to_north_star usages
grep -rn "theta_to_north_star" crates/ --include="*.rs" | wc -l
# Record result: Expected ~25

# Verify current serialization format version
grep -rn "SERIALIZATION_VERSION" crates/context-graph-storage/src/teleological/ --include="*.rs"
```

### Post-Execution Verification
```bash
# Verify NO North Star references (except comments)
grep -rn "north_star\|NorthStar" crates/ --include="*.rs" | grep -v "// " | grep -v "REMOVED" | wc -l
# Expected: 0

# Verify NO theta_to_north_star field
grep -rn "theta_to_north_star" crates/ --include="*.rs" | grep -v "// " | wc -l
# Expected: 0

# Compilation must succeed
cargo check --workspace
# Expected: Success

# All tests must pass
cargo test --workspace
# Expected: All pass

# Verify ego_node still exists (NOT deleted)
ls -la crates/context-graph-core/src/gwt/ego_node/
# Expected: Directory exists with all files
```

### Physical Database Verification
```rust
// After running migrations, verify no theta_to_north_star in stored data
// 1. Open RocksDB
// 2. Read a fingerprint from teleological CF
// 3. Deserialize and verify no theta_to_north_star field
// 4. Verify serialization version updated
```

## Edge Case Testing

### Edge Case 1: Empty Goal Hierarchy
**Input**: System with no goals stored
**Expected**: System boots successfully, no North Star errors
**Verification**: Check startup logs for absence of "north_star" errors

### Edge Case 2: Existing Data with theta_to_north_star
**Input**: Database with old fingerprints containing theta_to_north_star
**Expected**: Migration handles gracefully, old field ignored
**Verification**: Read old record, verify no crash, field absent in new reads

### Edge Case 3: Goal Level Serialization Migration
**Input**: Stored goal with old `GoalLevel::NorthStar` (value 0)
**Expected**: Migration converts to `GoalLevel::Strategic`
**Verification**: Read old goal, verify level is Strategic

## Synthetic Test Data

### Test Memory with North Star Reference (Before)
```json
{
  "id": "test-memory-001",
  "content": "Test content for North Star removal",
  "fingerprint": {
    "purpose_vector": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.85, 0.75, 0.65, 0.55],
    "theta_to_north_star": 0.523
  }
}
```

### Expected After Removal
```json
{
  "id": "test-memory-001",
  "content": "Test content for North Star removal",
  "fingerprint": {
    "purpose_vector": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.85, 0.75, 0.65, 0.55]
  }
}
```

## Success Criteria

1. **Compilation**: `cargo check --workspace` passes
2. **Tests**: `cargo test --workspace` all pass
3. **No References**: `grep -r "theta_to_north_star"` returns only comments
4. **ego_node Preserved**: `gwt/ego_node/` directory intact
5. **Identity Continuity Works**: IC monitoring still functional
6. **Database Migrated**: Old records readable without crash

## Dependencies

- **Depends On**: None (first task)
- **Blocks**: TASK-P0-002 (MCP handler cleanup), TASK-P0-003 (Constitution update)

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking serialization | HIGH | Version bump, migration script |
| Deleting ego_node by mistake | CRITICAL | Explicit KEEP list in task |
| Missing North Star references | MEDIUM | Comprehensive grep verification |
| Test failures | MEDIUM | Update all affected tests |

## Execution Checklist

- [ ] Backup current state (`git stash` or branch)
- [ ] **PRE-VERIFICATION**: Run all verification commands above
- [ ] Remove `theta_to_north_star` from `fingerprint.rs`
- [ ] Update serialization version
- [ ] Remove `GoalLevel::NorthStar` variant
- [ ] Update `propagation_weight()` for new enum values
- [ ] Remove North Star methods from `GoalHierarchy`
- [ ] Remove `auto_bootstrap_north_star` MCP tool
- [ ] Update all affected tests
- [ ] **POST-VERIFICATION**: Run all verification commands
- [ ] **EDGE CASE TESTS**: Run all 3 edge cases
- [ ] **DATABASE VERIFICATION**: Check physical storage
- [ ] Create commit with descriptive message

---

## Appendix A: Complete List of Affected Files (137 files)

### context-graph-core (57 files)

**Purpose Module (Critical - Start Here)**
```
crates/context-graph-core/src/purpose/goals/level.rs          # GoalLevel::NorthStar enum
crates/context-graph-core/src/purpose/goals/node.rs           # is_north_star(), child_goal assert
crates/context-graph-core/src/purpose/goals/hierarchy.rs      # north_star(), has_north_star(), path_to_north_star()
crates/context-graph-core/src/purpose/goals/error.rs          # NoNorthStar, MultipleNorthStars errors
crates/context-graph-core/src/purpose/goals/mod.rs            # Re-exports
crates/context-graph-core/src/purpose/goals/tests.rs          # Test fixtures
crates/context-graph-core/src/purpose/mod.rs                  # Module exports
crates/context-graph-core/src/purpose/tests.rs                # Integration tests
crates/context-graph-core/src/purpose/computer.rs             # Purpose computation
crates/context-graph-core/src/purpose/default_computer.rs     # Default impl
```

**Alignment Module (High Impact)**
```
crates/context-graph-core/src/alignment/calculator/default_calculator/mod.rs
crates/context-graph-core/src/alignment/calculator/default_calculator/patterns.rs
crates/context-graph-core/src/alignment/calculator/mod.rs
crates/context-graph-core/src/alignment/calculator/tests/*.rs
crates/context-graph-core/src/alignment/config.rs
crates/context-graph-core/src/alignment/error.rs
crates/context-graph-core/src/alignment/pattern.rs
crates/context-graph-core/src/alignment/score.rs
crates/context-graph-core/src/alignment/tests/*.rs
```

**Autonomous Module (Medium Impact)**
```
crates/context-graph-core/src/autonomous/bootstrap.rs
crates/context-graph-core/src/autonomous/evolution.rs
crates/context-graph-core/src/autonomous/discovery/pipeline.rs
crates/context-graph-core/src/autonomous/discovery/tests.rs
crates/context-graph-core/src/autonomous/services/gap_detection/service.rs
crates/context-graph-core/src/autonomous/services/gap_detection/tests/*.rs
crates/context-graph-core/src/autonomous/services/subgoal_discovery/service.rs
crates/context-graph-core/src/autonomous/services/subgoal_discovery/tests/*.rs
crates/context-graph-core/src/autonomous/workflow/events.rs
crates/context-graph-core/src/autonomous/workflow/status.rs
crates/context-graph-core/src/autonomous/workflow/tests.rs
```

**GWT Module (Low Impact - mostly comments/tests)**
```
crates/context-graph-core/src/gwt/ego_node/tests/tests_basic.rs
crates/context-graph-core/src/gwt/listeners/tests/identity_tests.rs
crates/context-graph-core/src/gwt/tests/common.rs
crates/context-graph-core/src/gwt/workspace/global.rs
crates/context-graph-core/src/gwt/workspace/mod.rs
```

**Types/Traits/Retrieval**
```
crates/context-graph-core/src/types/fingerprint/teleological/alignment.rs
crates/context-graph-core/src/types/fingerprint/teleological/core.rs
crates/context-graph-core/src/types/fingerprint/teleological/tests.rs
crates/context-graph-core/src/types/fingerprint/teleological/types.rs
crates/context-graph-core/src/traits/teleological_memory_store/ext.rs
crates/context-graph-core/src/traits/teleological_memory_store_tests.rs
crates/context-graph-core/src/retrieval/pipeline/default.rs
crates/context-graph-core/src/retrieval/pipeline/filtering.rs
crates/context-graph-core/src/retrieval/pipeline/tests.rs
crates/context-graph-core/src/stubs/teleological_store_stub/search.rs
```

### context-graph-embeddings (6 files)

```
crates/context-graph-embeddings/src/storage/types/fingerprint.rs      # theta_to_north_star field
crates/context-graph-embeddings/src/storage/types/query_results.rs    # Purpose alignment score
crates/context-graph-embeddings/src/storage/types/tests.rs
crates/context-graph-embeddings/tests/storage_roundtrip_test/edge_cases.rs
crates/context-graph-embeddings/tests/storage_roundtrip_test/fingerprint_creation.rs
crates/context-graph-embeddings/tests/storage_roundtrip_test/serialization_roundtrip.rs
```

### context-graph-mcp (69 files)

**Handlers - Autonomous (Critical)**
```
crates/context-graph-mcp/src/handlers/autonomous/bootstrap.rs         # auto_bootstrap_north_star
crates/context-graph-mcp/src/handlers/autonomous/discovery.rs
crates/context-graph-mcp/src/handlers/autonomous/drift.rs
crates/context-graph-mcp/src/handlers/autonomous/maintenance.rs
crates/context-graph-mcp/src/handlers/autonomous/mod.rs
crates/context-graph-mcp/src/handlers/autonomous/params.rs
crates/context-graph-mcp/src/handlers/autonomous/status.rs
```

**Handlers - Purpose**
```
crates/context-graph-mcp/src/handlers/purpose/aligned.rs
crates/context-graph-mcp/src/handlers/purpose/drift.rs
crates/context-graph-mcp/src/handlers/purpose/helpers.rs
crates/context-graph-mcp/src/handlers/purpose/hierarchy.rs
crates/context-graph-mcp/src/handlers/purpose/mod.rs
crates/context-graph-mcp/src/handlers/purpose/query.rs
crates/context-graph-mcp/src/handlers/purpose/tests.rs
```

**Handlers - Memory/Search/Core**
```
crates/context-graph-mcp/src/handlers/memory/inject.rs
crates/context-graph-mcp/src/handlers/memory/inject_batch.rs
crates/context-graph-mcp/src/handlers/memory/retrieve.rs
crates/context-graph-mcp/src/handlers/memory/search.rs
crates/context-graph-mcp/src/handlers/memory/store.rs
crates/context-graph-mcp/src/handlers/search.rs
crates/context-graph-mcp/src/handlers/merge.rs
crates/context-graph-mcp/src/handlers/steering.rs
crates/context-graph-mcp/src/handlers/core/dispatch.rs
crates/context-graph-mcp/src/handlers/core/handlers.rs
crates/context-graph-mcp/src/handlers/mod.rs
```

**Handlers - Tests (Many files)**
```
crates/context-graph-mcp/src/handlers/tests/north_star.rs             # Dedicated North Star tests
crates/context-graph-mcp/src/handlers/tests/purpose/north_star_alignment.rs
crates/context-graph-mcp/src/handlers/tests/purpose/north_star_update.rs
crates/context-graph-mcp/src/handlers/tests/purpose/*.rs
crates/context-graph-mcp/src/handlers/tests/memory/*.rs
crates/context-graph-mcp/src/handlers/tests/search/*.rs
crates/context-graph-mcp/src/handlers/tests/exhaustive_mcp_tools/*.rs
crates/context-graph-mcp/src/handlers/tests/full_state_verification*.rs
crates/context-graph-mcp/src/handlers/tests/integration_e2e/*.rs
crates/context-graph-mcp/src/handlers/tests/manual_fix_verification/*.rs
crates/context-graph-mcp/src/handlers/tests/phase*.rs
```

**Tools/Protocol**
```
crates/context-graph-mcp/src/handlers/tools/dispatch.rs
crates/context-graph-mcp/src/handlers/tools/gwt_workspace.rs
crates/context-graph-mcp/src/handlers/tools/memory_tools.rs
crates/context-graph-mcp/src/protocol.rs
crates/context-graph-mcp/src/tools/definitions/autonomous.rs
crates/context-graph-mcp/src/tools/mod.rs
crates/context-graph-mcp/src/tools/names.rs
crates/context-graph-mcp/src/tools/registry.rs
```

### context-graph-storage (5 files)

```
crates/context-graph-storage/src/teleological/tests/serialization.rs
crates/context-graph-storage/tests/full_integration_real_data/operations_tests.rs
crates/context-graph-storage/tests/full_state_verification/helpers.rs
crates/context-graph-storage/tests/purpose_vector_integration.rs
crates/context-graph-storage/tests/teleological_integration.rs
```

---

## Appendix B: Key Code Locations

### GoalLevel::NorthStar Definition
**File**: `crates/context-graph-core/src/purpose/goals/level.rs:17-21`
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum GoalLevel {
    NorthStar = 0,  // REMOVE THIS
    Strategic = 1,
    Tactical = 2,
    Immediate = 3,
}
```

### theta_to_north_star Field
**File**: `crates/context-graph-embeddings/src/storage/types/fingerprint.rs:58`
```rust
pub theta_to_north_star: f32,  // REMOVE THIS
```

### GoalHierarchy North Star Methods
**File**: `crates/context-graph-core/src/purpose/goals/hierarchy.rs:84-95,139-149`
```rust
// REMOVE THESE METHODS:
pub fn north_star(&self) -> Option<&GoalNode>
pub fn has_north_star(&self) -> bool
pub fn path_to_north_star(&self, goal_id: &Uuid) -> Vec<Uuid>
```

### auto_bootstrap_north_star MCP Tool
**File**: `crates/context-graph-mcp/src/handlers/autonomous/bootstrap.rs:36`
```rust
// REMOVE THIS FUNCTION:
pub(crate) async fn call_auto_bootstrap_north_star(...)
```

### Error Types
**File**: `crates/context-graph-core/src/purpose/goals/error.rs:29-36`
```rust
// REMOVE THESE ERROR VARIANTS:
#[error("No North Star goal defined")]
NoNorthStar,

#[error("Multiple North Star goals not allowed")]
MultipleNorthStars,
```

---

## Appendix C: Manual Testing Scripts

### Test 1: Pre-Removal State Capture
```bash
#!/bin/bash
# Run BEFORE making changes
echo "=== PRE-REMOVAL STATE CAPTURE ===" > /tmp/north_star_pre.log
date >> /tmp/north_star_pre.log

echo "Reference counts:" >> /tmp/north_star_pre.log
echo "north_star|NorthStar: $(grep -rn 'north_star\|NorthStar' crates/ --include='*.rs' | wc -l)" >> /tmp/north_star_pre.log
echo "theta_to_north_star: $(grep -rn 'theta_to_north_star' crates/ --include='*.rs' | wc -l)" >> /tmp/north_star_pre.log

echo "Compilation check:" >> /tmp/north_star_pre.log
cargo check --workspace 2>&1 | tail -5 >> /tmp/north_star_pre.log

echo "Test status:" >> /tmp/north_star_pre.log
cargo test --workspace --no-run 2>&1 | tail -5 >> /tmp/north_star_pre.log

cat /tmp/north_star_pre.log
```

### Test 2: Post-Removal Verification
```bash
#!/bin/bash
# Run AFTER making changes
echo "=== POST-REMOVAL VERIFICATION ===" > /tmp/north_star_post.log
date >> /tmp/north_star_post.log

echo "Reference counts (should be ~0 except comments):" >> /tmp/north_star_post.log
echo "north_star|NorthStar: $(grep -rn 'north_star\|NorthStar' crates/ --include='*.rs' | grep -v '//' | wc -l)" >> /tmp/north_star_post.log
echo "theta_to_north_star: $(grep -rn 'theta_to_north_star' crates/ --include='*.rs' | grep -v '//' | wc -l)" >> /tmp/north_star_post.log

echo "ego_node preserved check:" >> /tmp/north_star_post.log
if [ -d "crates/context-graph-core/src/gwt/ego_node" ]; then
    echo "  ego_node directory: EXISTS (GOOD)" >> /tmp/north_star_post.log
else
    echo "  ego_node directory: MISSING (ERROR!)" >> /tmp/north_star_post.log
fi

echo "Compilation check:" >> /tmp/north_star_post.log
cargo check --workspace 2>&1 | tail -10 >> /tmp/north_star_post.log

echo "Test execution:" >> /tmp/north_star_post.log
cargo test --workspace 2>&1 | tail -20 >> /tmp/north_star_post.log

cat /tmp/north_star_post.log
```

### Test 3: Edge Case - Empty Hierarchy
```bash
#!/bin/bash
# Test empty goal hierarchy handling
cd /home/cabdru/contextgraph

# Create test that verifies empty hierarchy doesn't crash
cat > /tmp/test_empty_hierarchy.rs << 'EOF'
// Test: Empty goal hierarchy should not require North Star
use context_graph_core::purpose::goals::GoalHierarchy;

fn main() {
    let hierarchy = GoalHierarchy::new();
    assert!(hierarchy.is_empty());
    // After removal, validate() should NOT require North Star
    // hierarchy.validate().expect("Empty hierarchy should be valid");
    println!("Empty hierarchy test: PASS");
}
EOF

echo "Edge case test created. Add to integration tests."
```

---

## Appendix D: Rollback Procedure

If the removal causes unexpected issues:

```bash
# 1. Revert all changes
git checkout -- .

# 2. If committed, revert commit
git revert HEAD

# 3. Verify restoration
grep -rn "theta_to_north_star" crates/ --include="*.rs" | wc -l
# Expected: 120 (original count)

cargo check --workspace
cargo test --workspace
```
