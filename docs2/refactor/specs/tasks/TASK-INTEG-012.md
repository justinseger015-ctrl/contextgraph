# TASK-INTEG-012: PreCompact and SubagentStop Hooks

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-INTEG-012 |
| **Title** | PreCompact and SubagentStop Hooks |
| **Status** | :white_circle: todo |
| **Layer** | Integration |
| **Sequence** | 32 |
| **Estimated Days** | 2 |
| **Complexity** | Medium |

## Implements

- **ARCH-07**: Hook-driven lifecycle (PreCompact, SubagentStop)
- Constitution lines 713-719

## Dependencies

| Task | Reason |
|------|--------|
| TASK-INTEG-004 | Hook protocol infrastructure |
| TASK-LOGIC-009 | Goal discovery for salience scoring |

## Objective

Implement two missing hooks required by the constitution:
1. **PreCompact** - Extract salient memories before context compaction (10000ms timeout)
2. **SubagentStop** - Merge subagent learnings into main memory (5000ms timeout)

## Context

**Constitution Requirements (lines 713-719):**

```xml
<hook event="PreCompact" required="false">
  Extract salient memories before context compaction.
  Timeout: 10000ms.
</hook>

<hook event="SubagentStop" required="false">
  Merge subagent learnings into main memory.
  Timeout: 5000ms.
</hook>
```

These hooks are currently **MISSING** from the task specifications despite being required by the constitution.

## Scope

### In Scope

- `PreCompactHandler` implementation
- `SubagentStopHandler` implementation
- Salient memory extraction logic
- Subagent learning merge logic
- Timeout enforcement per constitution
- Integration with existing hook protocol

### Out of Scope

- Context compaction logic (handled by Claude Code)
- Subagent spawning logic (see TASK-INTEG-014)

## Definition of Done

### PreCompact Hook

```rust
// crates/context-graph-mcp/src/hooks/pre_compact.rs

use std::time::Duration;

/// Handler for PreCompact hook
///
/// Triggered before Claude Code compacts context window.
/// Extracts salient memories to preserve important context.
pub struct PreCompactHandler {
    timeout: Duration,
    salience_threshold: f32,
    max_memories_to_extract: usize,
}

impl PreCompactHandler {
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_millis(10_000), // Constitution: 10000ms
            salience_threshold: 0.7,
            max_memories_to_extract: 50,
        }
    }

    /// Handle PreCompact event
    ///
    /// # Arguments
    /// * `context` - Current context about to be compacted
    ///
    /// # Returns
    /// Salient memories to preserve
    pub async fn handle(&self, context: PreCompactContext) -> HookResult<PreCompactOutput> {
        tokio::time::timeout(self.timeout, async {
            // 1. Analyze current context for important patterns
            let context_analysis = self.analyze_context(&context).await?;

            // 2. Score memories by salience relative to current work
            let scored_memories = self.score_by_salience(
                &context_analysis,
                &context.active_memories,
            ).await?;

            // 3. Extract memories above threshold
            let salient = scored_memories.into_iter()
                .filter(|(_, score)| *score >= self.salience_threshold)
                .take(self.max_memories_to_extract)
                .collect();

            // 4. Create extraction summary
            let output = PreCompactOutput {
                extracted_memories: salient,
                context_summary: context_analysis.summary,
                recommended_retention: context_analysis.key_entities,
            };

            Ok(output)
        }).await
        .map_err(|_| HookError::Timeout("PreCompact".into(), 10_000))?
    }

    async fn analyze_context(&self, context: &PreCompactContext) -> HookResult<ContextAnalysis>;

    async fn score_by_salience(
        &self,
        analysis: &ContextAnalysis,
        memories: &[TeleologicalArray],
    ) -> HookResult<Vec<(Uuid, f32)>>;
}

/// Input to PreCompact handler
#[derive(Debug)]
pub struct PreCompactContext {
    /// Current conversation context being compacted
    pub conversation_summary: String,
    /// Active memories currently in context
    pub active_memories: Vec<TeleologicalArray>,
    /// How much context is being removed
    pub compaction_ratio: f32,
}

/// Output from PreCompact handler
#[derive(Debug)]
pub struct PreCompactOutput {
    /// Memories to preserve (id -> salience score)
    pub extracted_memories: Vec<(Uuid, f32)>,
    /// Summary of what was important in compacted context
    pub context_summary: String,
    /// Key entities to remember
    pub recommended_retention: Vec<String>,
}
```

### SubagentStop Hook

```rust
// crates/context-graph-mcp/src/hooks/subagent_stop.rs

use std::time::Duration;

/// Handler for SubagentStop hook
///
/// Triggered when a subagent completes its work.
/// Merges subagent learnings into main session memory.
pub struct SubagentStopHandler {
    timeout: Duration,
    merge_strategy: MergeStrategy,
}

impl SubagentStopHandler {
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_millis(5_000), // Constitution: 5000ms
            merge_strategy: MergeStrategy::Selective,
        }
    }

    /// Handle SubagentStop event
    ///
    /// # Arguments
    /// * `context` - Subagent completion context
    ///
    /// # Returns
    /// Merge result
    pub async fn handle(&self, context: SubagentStopContext) -> HookResult<SubagentStopOutput> {
        tokio::time::timeout(self.timeout, async {
            // 1. Extract learnings from subagent session
            let learnings = self.extract_learnings(&context).await?;

            // 2. Score learnings by relevance to main session
            let scored = self.score_learnings(&learnings, &context.parent_session).await?;

            // 3. Filter and prepare for merge
            let to_merge = self.select_for_merge(&scored);

            // 4. Perform merge into main memory
            let merged = self.merge_into_main(&to_merge, &context.parent_session).await?;

            // 5. Update parent session context
            let output = SubagentStopOutput {
                merged_count: merged.len(),
                discarded_count: learnings.len() - merged.len(),
                summary: self.summarize_learnings(&merged),
                updated_goals: self.extract_goal_updates(&merged),
            };

            Ok(output)
        }).await
        .map_err(|_| HookError::Timeout("SubagentStop".into(), 5_000))?
    }

    async fn extract_learnings(&self, context: &SubagentStopContext) -> HookResult<Vec<Learning>>;

    async fn score_learnings(
        &self,
        learnings: &[Learning],
        parent: &SessionId,
    ) -> HookResult<Vec<(Learning, f32)>>;

    fn select_for_merge(&self, scored: &[(Learning, f32)]) -> Vec<Learning>;

    async fn merge_into_main(
        &self,
        learnings: &[Learning],
        parent: &SessionId,
    ) -> HookResult<Vec<Uuid>>;

    fn summarize_learnings(&self, merged: &[Uuid]) -> String;

    fn extract_goal_updates(&self, merged: &[Uuid]) -> Vec<GoalUpdate>;
}

/// Merge strategy options
#[derive(Debug, Clone, Copy)]
pub enum MergeStrategy {
    /// Merge all learnings
    All,
    /// Merge only high-relevance learnings (score > 0.7)
    Selective,
    /// Merge only if subagent was successful
    OnSuccess,
    /// No automatic merge (manual consolidation required)
    Manual,
}

/// Input to SubagentStop handler
#[derive(Debug)]
pub struct SubagentStopContext {
    /// ID of the stopping subagent
    pub subagent_id: SubagentId,
    /// Parent session to merge into
    pub parent_session: SessionId,
    /// What the subagent was working on
    pub task_description: String,
    /// Whether subagent completed successfully
    pub success: bool,
    /// Memories created during subagent execution
    pub created_memories: Vec<Uuid>,
    /// Duration of subagent execution
    pub duration: Duration,
}

/// Output from SubagentStop handler
#[derive(Debug)]
pub struct SubagentStopOutput {
    /// How many memories were merged
    pub merged_count: usize,
    /// How many were discarded
    pub discarded_count: usize,
    /// Human-readable summary of what was learned
    pub summary: String,
    /// Goal updates derived from learnings
    pub updated_goals: Vec<GoalUpdate>,
}

/// A learning from subagent to potentially merge
#[derive(Debug)]
pub struct Learning {
    pub memory_id: Uuid,
    pub learning_type: LearningType,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum LearningType {
    NewPattern,
    ConfirmedKnowledge,
    CorrectedMistake,
    NewEntity,
    CausalLink,
}

#[derive(Debug)]
pub struct GoalUpdate {
    pub goal_id: Uuid,
    pub update_type: GoalUpdateType,
    pub delta: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum GoalUpdateType {
    Progress,
    Completion,
    Refinement,
    Abandonment,
}
```

### Hook Registration

```rust
// Update crates/context-graph-mcp/src/hooks/mod.rs

pub mod pre_compact;
pub mod subagent_stop;

pub use pre_compact::{PreCompactHandler, PreCompactContext, PreCompactOutput};
pub use subagent_stop::{SubagentStopHandler, SubagentStopContext, SubagentStopOutput};

/// Extended hook dispatcher with new hooks
impl HookDispatcher {
    pub async fn dispatch_pre_compact(
        &self,
        context: PreCompactContext,
    ) -> HookResult<PreCompactOutput> {
        self.pre_compact_handler.handle(context).await
    }

    pub async fn dispatch_subagent_stop(
        &self,
        context: SubagentStopContext,
    ) -> HookResult<SubagentStopOutput> {
        self.subagent_stop_handler.handle(context).await
    }
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| PreCompact timeout | 10000ms |
| SubagentStop timeout | 5000ms |
| PreCompact max memories | 50 |
| Merge relevance threshold | 0.7 |

## Verification

- [ ] PreCompact hook triggers before context compaction
- [ ] PreCompact completes within 10s timeout
- [ ] Salient memories extracted above threshold
- [ ] SubagentStop hook triggers on subagent completion
- [ ] SubagentStop completes within 5s timeout
- [ ] Learnings merged into parent session correctly
- [ ] Failed merge doesn't block subagent cleanup

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/hooks/pre_compact.rs` | PreCompact handler |
| `crates/context-graph-mcp/src/hooks/subagent_stop.rs` | SubagentStop handler |
| Update `crates/context-graph-mcp/src/hooks/mod.rs` | Export new handlers |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Timeout too short | Medium | Medium | Log slow operations, tune threshold |
| Merge conflicts | Low | Low | Idempotent merge operations |
| Memory bloat from merge | Low | Medium | Selective merge strategy |

## Traceability

- Source: Constitution claude_code_integration/hooks (lines 713-719)
