# TASK-NEURO-P2-002: MCP Steering Handler Integration for Direct Dopamine Feedback

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-NEURO-P2-002 |
| **Title** | Integrate Steering MCP Handler with NeuromodulationManager |
| **Status** | Ready |
| **Priority** | P2 (Minor Refinement) |
| **Layer** | surface |
| **Sequence** | 2 (after TASK-NEURO-P2-001) |
| **Estimated Complexity** | Low |
| **Estimated Duration** | 1-2 hours |
| **Implements** | SPEC-NEURO-001 Section 9 |
| **Depends On** | TASK-NEURO-P2-001 (on_goal_progress must exist) |

---

## 1. Context

This task completes the direct dopamine feedback loop by wiring the MCP steering handler to invoke `NeuromodulationManager.on_goal_progress()` after computing steering feedback.

**Current State**:
- `call_get_steering_feedback()` computes reward but does not modulate dopamine
- `NeuromodulationManager` is not accessible from `Handlers` struct
- Dopamine modulation only occurs via workspace entry events

**Target State**:
- `Handlers` struct includes `Arc<RwLock<NeuromodulationManager>>`
- `call_get_steering_feedback()` invokes `on_goal_progress(reward.value)`
- Response includes neuromodulation status
- Errors in neuromodulation do not fail the MCP response

---

## 2. Input Context Files

The agent MUST read these files before implementation:

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/handlers/mod.rs` | Handlers struct definition |
| `crates/context-graph-mcp/src/handlers/steering.rs` | Steering handler implementation |
| `crates/context-graph-mcp/src/lib.rs` | Server construction |
| `crates/context-graph-core/src/neuromod/mod.rs` | Neuromodulation exports |
| `specs/functional/SPEC-NEURO-001.md` | Specification (Section 9) |

---

## 3. Scope

### 3.1 In Scope

1. Add `neuromod_manager: Arc<RwLock<NeuromodulationManager>>` to `Handlers` struct
2. Update `Handlers::new()` to accept and store the neuromod manager
3. Update server construction to pass neuromod manager to handlers
4. Modify `call_get_steering_feedback()` to invoke `on_goal_progress()`
5. Add neuromodulation status to the steering feedback response
6. Add appropriate logging for the DA modulation
7. Handle lock acquisition failures gracefully (non-fatal)

### 3.2 Out of Scope

- Core neuromodulation logic (handled by TASK-NEURO-P2-001)
- Cascade effects to other neuromodulators (TASK-NEURO-P2-003)
- Changes to MCP tool schema
- Unit tests (integration test only)

---

## 4. Definition of Done

### 4.1 Required Changes

**File: `crates/context-graph-mcp/src/handlers/mod.rs`**

```rust
use context_graph_core::neuromod::NeuromodulationManager;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct Handlers {
    // ... existing fields ...

    /// Neuromodulation manager for DA/5HT/NE control
    pub neuromod_manager: Arc<RwLock<NeuromodulationManager>>,
}

impl Handlers {
    pub fn new(
        // ... existing params ...
        neuromod_manager: Arc<RwLock<NeuromodulationManager>>,
    ) -> Self {
        Self {
            // ... existing fields ...
            neuromod_manager,
        }
    }
}
```

**File: `crates/context-graph-mcp/src/handlers/steering.rs`**

```rust
use context_graph_core::neuromod::dopamine::DA_GOAL_SENSITIVITY;

impl Handlers {
    pub(super) async fn call_get_steering_feedback(
        &self,
        id: Option<JsonRpcId>,
    ) -> JsonRpcResponse {
        // ... existing code up to feedback computation ...

        let feedback = steering.compute_feedback(/* ... */);

        // Direct DA modulation from steering feedback
        let neuromod_updated = match self.neuromod_manager.try_write() {
            Ok(mut neuromod) => {
                let reward_value = feedback.reward.value;
                if !reward_value.is_nan() {
                    neuromod.on_goal_progress(reward_value);
                    debug!(
                        delta = reward_value,
                        da_adjustment = reward_value * DA_GOAL_SENSITIVITY,
                        "Steering feedback -> DA modulation"
                    );
                    true
                } else {
                    warn!("Steering reward is NaN, skipping DA modulation");
                    false
                }
            }
            Err(_) => {
                warn!("Could not acquire neuromod lock for goal progress");
                false
            }
        };

        self.tool_result_with_pulse(
            id,
            json!({
                // ... existing response fields ...
                "neuromod": {
                    "updated": neuromod_updated,
                    "da_delta": if neuromod_updated {
                        feedback.reward.value * DA_GOAL_SENSITIVITY
                    } else {
                        0.0
                    }
                }
            }),
        )
    }
}
```

### 4.2 Constraints

- [ ] Neuromod lock failure MUST NOT fail the MCP response
- [ ] NaN reward values MUST be skipped with warning log
- [ ] Logging MUST include delta and resulting DA adjustment
- [ ] Response MUST include `neuromod.updated` boolean
- [ ] Response MUST include `neuromod.da_delta` float
- [ ] All existing tests MUST continue to pass

### 4.3 Test Requirements

**Integration Test (in `tests/` directory)**:

```rust
#[tokio::test]
async fn test_steering_feedback_modulates_dopamine() {
    // Setup: Create handlers with neuromod manager
    let neuromod_manager = Arc::new(RwLock::new(NeuromodulationManager::new()));
    let handlers = create_test_handlers_with_neuromod(neuromod_manager.clone()).await;

    // Get initial DA
    let initial_da = neuromod_manager.read().await.get_hopfield_beta();

    // Call steering feedback
    let response = handlers.call_get_steering_feedback(Some(JsonRpcId::Number(1))).await;

    // Verify response includes neuromod status
    let result = response.result.unwrap();
    assert!(result.get("neuromod").is_some());
    assert!(result["neuromod"]["updated"].as_bool().unwrap_or(false));

    // Verify DA changed (direction depends on reward sign)
    let final_da = neuromod_manager.read().await.get_hopfield_beta();
    // Note: DA may or may not change depending on computed reward
    // The key assertion is that the integration doesn't crash
}
```

---

## 5. Files to Modify

| File Path | Action | Description |
|-----------|--------|-------------|
| `crates/context-graph-mcp/src/handlers/mod.rs` | MODIFY | Add neuromod_manager field |
| `crates/context-graph-mcp/src/handlers/steering.rs` | MODIFY | Invoke on_goal_progress |
| `crates/context-graph-mcp/src/lib.rs` | MODIFY | Pass neuromod manager to handlers |

---

## 6. Validation Criteria

### 6.1 Automated Validation

| Command | Expected Result |
|---------|-----------------|
| `cargo build -p context-graph-mcp` | Success, no warnings |
| `cargo test -p context-graph-mcp` | All tests pass |
| `cargo clippy -p context-graph-mcp` | No warnings |

### 6.2 Manual Validation

1. **Start MCP Server**: Launch the server with neuromod manager
2. **Call get_steering_feedback**: Verify response includes `neuromod` field
3. **Verify Logs**: Check DEBUG logs show DA modulation messages
4. **Lock Contention Test**: Acquire neuromod lock, call steering, verify graceful handling

---

## 7. Pseudo-Code

### 7.1 handlers/mod.rs Changes

```rust
// Add import
use context_graph_core::neuromod::NeuromodulationManager;
use std::sync::Arc;
use tokio::sync::RwLock;

// Add field to Handlers struct
pub struct Handlers {
    // ... existing fields ...

    /// Neuromodulation manager for dopamine feedback
    pub neuromod_manager: Arc<RwLock<NeuromodulationManager>>,
}

// Update constructor
impl Handlers {
    pub fn new(
        // ... existing params ...
        neuromod_manager: Arc<RwLock<NeuromodulationManager>>,
    ) -> Self {
        Self {
            // ... existing field assignments ...
            neuromod_manager,
        }
    }
}
```

### 7.2 handlers/steering.rs Changes

```rust
// Add import
use context_graph_core::neuromod::dopamine::DA_GOAL_SENSITIVITY;

impl Handlers {
    pub(super) async fn call_get_steering_feedback(
        &self,
        id: Option<JsonRpcId>,
    ) -> JsonRpcResponse {
        // ... existing code ...

        let feedback = steering.compute_feedback(/* ... */);

        // NEW: Direct DA modulation
        let neuromod_updated = self.try_modulate_da(feedback.reward.value).await;

        self.tool_result_with_pulse(
            id,
            json!({
                // ... existing fields ...
                "neuromod": {
                    "updated": neuromod_updated,
                    "da_delta": if neuromod_updated {
                        feedback.reward.value * DA_GOAL_SENSITIVITY
                    } else {
                        0.0
                    }
                }
            }),
        )
    }

    /// Attempt to modulate dopamine with steering reward.
    /// Returns true if modulation was applied.
    async fn try_modulate_da(&self, reward: f32) -> bool {
        if reward.is_nan() {
            warn!("Steering reward is NaN, skipping DA modulation");
            return false;
        }

        match self.neuromod_manager.try_write() {
            Ok(mut neuromod) => {
                neuromod.on_goal_progress(reward);
                debug!(
                    reward,
                    adjustment = reward * DA_GOAL_SENSITIVITY,
                    "Steering -> DA modulation applied"
                );
                true
            }
            Err(_) => {
                warn!("Could not acquire neuromod lock for DA modulation");
                false
            }
        }
    }
}
```

---

## 8. Implementation Checklist

- [ ] Read all input context files
- [ ] Add `neuromod_manager` field to `Handlers` struct
- [ ] Update `Handlers::new()` signature and implementation
- [ ] Update server construction to pass neuromod manager
- [ ] Add `try_modulate_da()` helper method
- [ ] Call `try_modulate_da()` in `call_get_steering_feedback()`
- [ ] Add `neuromod` field to JSON response
- [ ] Add DEBUG logging for DA modulation
- [ ] Run `cargo build -p context-graph-mcp`
- [ ] Run `cargo test -p context-graph-mcp`
- [ ] Run `cargo clippy -p context-graph-mcp`
- [ ] Update task status to COMPLETED

---

## 9. Notes for Implementation Agent

### 9.1 Error Handling Philosophy

The MCP response MUST NEVER fail due to neuromodulation errors. The steering feedback is the primary deliverable; DA modulation is a beneficial side effect. If the neuromod lock cannot be acquired, log a warning and continue.

### 9.2 Async Considerations

The `try_write()` method is non-blocking. If the lock is held, it returns immediately with an error. This is the correct behavior for MCP handlers - we don't want to block waiting for the lock.

### 9.3 Thread Safety

`NeuromodulationManager` is wrapped in `Arc<RwLock<>>` for thread-safe access across async handlers. The `RwLock` allows multiple readers but exclusive writers.

---

## 10. Traceability

| Requirement | Implemented By | Test Coverage |
|-------------|----------------|---------------|
| FR-NEURO-001-03 | `call_get_steering_feedback()` modification | Integration test |
| SPEC-NEURO-001 Section 9.1 | `try_modulate_da()` method | Integration test |
| SPEC-NEURO-001 Section 9.2 | Handlers struct changes | Compilation |
| SPEC-NEURO-001 Section 9.3 | Error handling in try_modulate_da | Manual verification |

---

## Appendix A: Response Schema Update

The steering feedback response will include a new `neuromod` field:

```json
{
  "reward": {
    "value": 0.7,
    "gardener_score": 0.8,
    "curator_score": 0.6,
    "assessor_score": 0.7
  },
  "neuromod": {
    "updated": true,
    "da_delta": 0.07
  }
}
```

This is a BACKWARDS COMPATIBLE addition - existing clients can ignore the new field.
