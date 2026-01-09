# TASK-INTEG-014: Core Subagents (embedding-agent, search-agent, goal-agent, dream-agent)

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-INTEG-014 |
| **Title** | Core Subagents (embedding-agent, search-agent, goal-agent, dream-agent) |
| **Status** | :white_circle: todo |
| **Layer** | Integration |
| **Sequence** | 34 |
| **Estimated Days** | 3 |
| **Complexity** | High |

## Implements

- Constitution subagent specifications (lines 750-800+)
- ARCH-07: Hook-driven lifecycle (subagent coordination)
- All 4 required subagents for autonomous operation

## Dependencies

| Task | Reason |
|------|--------|
| TASK-INTEG-001 | MCP handler infrastructure |
| TASK-INTEG-004 | Hook protocol for subagent events |
| TASK-INTEG-012 | SubagentStop hook for learning merge |
| TASK-LOGIC-009 | Goal discovery for goal-agent |
| TASK-LOGIC-010 | Drift detection for dream-agent |

## Objective

Implement the 4 core subagents required by the constitution for autonomous context graph operation:
1. **embedding-agent** - Handles embedding generation with GPU coordination
2. **search-agent** - Executes searches with entry-point selection
3. **goal-agent** - Manages goal discovery and tracking
4. **dream-agent** - Performs consolidation and pattern discovery

## Context

**Constitution Requirements:**

The constitution specifies 4 subagents that run as background workers, each with specific responsibilities and coordination requirements. These are **MISSING** from the current task specifications.

```
Subagents:
├── embedding-agent  # GPU-aware embedding generation
├── search-agent     # Entry-point search execution
├── goal-agent       # Autonomous goal tracking
└── dream-agent      # Memory consolidation (REM)
```

## Scope

### In Scope

- All 4 subagent implementations
- Subagent spawning and lifecycle management
- Inter-subagent communication protocol
- Hook integration for subagent events
- Resource isolation per subagent
- Learning merge on SubagentStop

### Out of Scope

- Distributed subagent deployment
- Subagent auto-scaling
- Custom user-defined subagents

## Definition of Done

### Subagent Base Infrastructure

```rust
// crates/context-graph-mcp/src/subagents/mod.rs

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

/// Base trait for all subagents
#[async_trait]
pub trait Subagent: Send + Sync {
    /// Unique identifier for this subagent type
    fn agent_type(&self) -> SubagentType;

    /// Start the subagent
    async fn start(&self) -> SubagentResult<()>;

    /// Stop the subagent gracefully
    async fn stop(&self) -> SubagentResult<SubagentOutput>;

    /// Check if subagent is healthy
    async fn health_check(&self) -> SubagentHealth;

    /// Get current status
    fn status(&self) -> SubagentStatus;
}

/// Subagent types as defined by constitution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubagentType {
    Embedding,
    Search,
    Goal,
    Dream,
}

/// Subagent status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentStatus {
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed,
}

/// Subagent health
#[derive(Debug, Clone)]
pub struct SubagentHealth {
    pub status: SubagentStatus,
    pub last_heartbeat: std::time::Instant,
    pub tasks_completed: u64,
    pub errors: u64,
}

/// Output from a subagent session (for learning merge)
#[derive(Debug)]
pub struct SubagentOutput {
    pub agent_type: SubagentType,
    pub memories_created: Vec<Uuid>,
    pub patterns_discovered: Vec<PatternId>,
    pub duration: std::time::Duration,
    pub success: bool,
}
```

### Embedding Agent

```rust
// crates/context-graph-mcp/src/subagents/embedding_agent.rs

use context_graph_core::teleology::embedder::Embedder;

/// GPU-aware embedding generation subagent
pub struct EmbeddingAgent {
    gpu_pool: Arc<GpuMemoryPool>,
    model_registry: Arc<EmbedderModelRegistry>,
    task_queue: mpsc::Receiver<EmbeddingTask>,
    result_sender: mpsc::Sender<EmbeddingResult>,
    status: RwLock<SubagentStatus>,
}

/// Task for embedding generation
#[derive(Debug)]
pub struct EmbeddingTask {
    pub id: Uuid,
    pub content: String,
    pub embedders: Vec<Embedder>,
    pub priority: TaskPriority,
}

/// Result from embedding task
#[derive(Debug)]
pub struct EmbeddingResult {
    pub task_id: Uuid,
    pub embeddings: HashMap<Embedder, EmbedderOutput>,
    pub duration: std::time::Duration,
    pub gpu_memory_used: usize,
}

impl EmbeddingAgent {
    pub fn new(
        gpu_pool: Arc<GpuMemoryPool>,
        model_registry: Arc<EmbedderModelRegistry>,
    ) -> (Self, mpsc::Sender<EmbeddingTask>, mpsc::Receiver<EmbeddingResult>);

    /// Process embedding tasks from queue
    async fn process_tasks(&self) -> SubagentResult<()> {
        while let Some(task) = self.task_queue.recv().await {
            // 1. Acquire GPU memory
            let allocation = self.gpu_pool.allocate(
                self.estimate_memory(&task),
                task.priority.into(),
            ).await?;

            // 2. Generate embeddings for each requested embedder
            let mut results = HashMap::new();
            for embedder in &task.embedders {
                let model = self.model_registry.get(embedder).await?;
                let embedding = model.embed(&task.content).await?;
                results.insert(*embedder, embedding);
            }

            // 3. Release GPU memory
            drop(allocation);

            // 4. Send result
            self.result_sender.send(EmbeddingResult {
                task_id: task.id,
                embeddings: results,
                duration: start.elapsed(),
                gpu_memory_used: allocation.size(),
            }).await?;
        }
        Ok(())
    }

    fn estimate_memory(&self, task: &EmbeddingTask) -> usize {
        // Estimate based on content length and embedder count
        let base = task.content.len() * 4; // ~4 bytes per char
        let per_embedder = 4 * 1024 * 1024; // 4MB per model
        base + (task.embedders.len() * per_embedder)
    }
}

#[async_trait]
impl Subagent for EmbeddingAgent {
    fn agent_type(&self) -> SubagentType { SubagentType::Embedding }

    async fn start(&self) -> SubagentResult<()> {
        *self.status.write().await = SubagentStatus::Running;
        self.process_tasks().await
    }

    async fn stop(&self) -> SubagentResult<SubagentOutput> {
        *self.status.write().await = SubagentStatus::Stopping;
        // Drain remaining tasks
        Ok(SubagentOutput {
            agent_type: self.agent_type(),
            memories_created: vec![],
            patterns_discovered: vec![],
            duration: std::time::Duration::ZERO,
            success: true,
        })
    }

    async fn health_check(&self) -> SubagentHealth {
        SubagentHealth {
            status: *self.status.read().await,
            last_heartbeat: std::time::Instant::now(),
            tasks_completed: 0,
            errors: 0,
        }
    }

    fn status(&self) -> SubagentStatus {
        // Sync version
        SubagentStatus::Running
    }
}
```

### Search Agent

```rust
// crates/context-graph-mcp/src/subagents/search_agent.rs

use crate::search::{EntryPointSelector, SearchPipeline};

/// Entry-point search execution subagent
pub struct SearchAgent {
    pipeline: Arc<SearchPipeline>,
    entry_point_selector: Arc<dyn EntryPointSelector>,
    cache: Arc<SearchCache>,
    task_queue: mpsc::Receiver<SearchTask>,
    result_sender: mpsc::Sender<SearchTaskResult>,
    status: RwLock<SubagentStatus>,
}

/// Search task
#[derive(Debug)]
pub struct SearchTask {
    pub id: Uuid,
    pub query: TeleologicalArray,
    pub query_text: String,
    pub limit: usize,
    pub intent_hint: Option<QueryIntent>,
}

/// Search task result
#[derive(Debug)]
pub struct SearchTaskResult {
    pub task_id: Uuid,
    pub results: Vec<SearchResult>,
    pub entry_point_used: Embedder,
    pub cache_hit: bool,
    pub duration: std::time::Duration,
}

impl SearchAgent {
    pub fn new(
        pipeline: Arc<SearchPipeline>,
        entry_point_selector: Arc<dyn EntryPointSelector>,
        cache: Arc<SearchCache>,
    ) -> (Self, mpsc::Sender<SearchTask>, mpsc::Receiver<SearchTaskResult>);

    async fn process_tasks(&self) -> SubagentResult<()> {
        while let Some(task) = self.task_queue.recv().await {
            let start = std::time::Instant::now();

            // 1. Check cache first
            let query_hash = hash_query(&task.query);
            if let Some(cached) = self.cache.get(query_hash) {
                self.result_sender.send(SearchTaskResult {
                    task_id: task.id,
                    results: cached,
                    entry_point_used: Embedder::Semantic, // Unknown for cache
                    cache_hit: true,
                    duration: start.elapsed(),
                }).await?;
                continue;
            }

            // 2. Select optimal entry point
            let intent = task.intent_hint
                .unwrap_or_else(|| self.entry_point_selector.analyze_intent(&task.query_text));
            let entry_point = self.entry_point_selector
                .select_space(&task.query, Some(intent))
                .await?;

            // 3. Execute pipeline with selected entry point
            let results = self.pipeline
                .search_with_entry_point(&task.query, entry_point, task.limit)
                .await?;

            // 4. Cache results
            let ids: Vec<Uuid> = results.iter().map(|r| r.id).collect();
            self.cache.put_with_refs(query_hash, results.clone(), ids);

            // 5. Send result
            self.result_sender.send(SearchTaskResult {
                task_id: task.id,
                results,
                entry_point_used: entry_point,
                cache_hit: false,
                duration: start.elapsed(),
            }).await?;
        }
        Ok(())
    }
}

#[async_trait]
impl Subagent for SearchAgent {
    fn agent_type(&self) -> SubagentType { SubagentType::Search }
    // ... standard trait implementations
}
```

### Goal Agent

```rust
// crates/context-graph-mcp/src/subagents/goal_agent.rs

use crate::logic::goal_discovery::GoalDiscoverer;

/// Autonomous goal tracking subagent
pub struct GoalAgent {
    discoverer: Arc<GoalDiscoverer>,
    store: Arc<TeleologicalArrayStore>,
    active_goals: RwLock<Vec<TrackedGoal>>,
    status: RwLock<SubagentStatus>,
}

/// A goal being tracked
#[derive(Debug, Clone)]
pub struct TrackedGoal {
    pub id: Uuid,
    pub description: String,
    pub progress: f32,       // 0.0 - 1.0
    pub priority: f32,       // Higher = more important
    pub created_at: std::time::Instant,
    pub last_updated: std::time::Instant,
    pub related_memories: Vec<Uuid>,
}

impl GoalAgent {
    pub fn new(
        discoverer: Arc<GoalDiscoverer>,
        store: Arc<TeleologicalArrayStore>,
    ) -> Self;

    /// Discover new goals from recent memories
    pub async fn discover_goals(&self, limit: usize) -> SubagentResult<Vec<TrackedGoal>> {
        // 1. Get recent memories
        let recent = self.store.list_recent(100).await?;

        // 2. Discover emergent goals
        let discovered = self.discoverer
            .discover_emergent_goals(&recent, limit)
            .await?;

        // 3. Convert to tracked goals
        let goals: Vec<TrackedGoal> = discovered.into_iter()
            .map(|g| TrackedGoal {
                id: g.id,
                description: g.description,
                progress: 0.0,
                priority: g.confidence,
                created_at: std::time::Instant::now(),
                last_updated: std::time::Instant::now(),
                related_memories: g.member_ids,
            })
            .collect();

        // 4. Add to active goals
        self.active_goals.write().await.extend(goals.clone());

        Ok(goals)
    }

    /// Update goal progress based on new memories
    pub async fn update_progress(&self, memory_id: Uuid) -> SubagentResult<()> {
        let mut goals = self.active_goals.write().await;

        for goal in goals.iter_mut() {
            if goal.related_memories.contains(&memory_id) {
                // Simple progress increment (real impl would be smarter)
                goal.progress = (goal.progress + 0.1).min(1.0);
                goal.last_updated = std::time::Instant::now();
            }
        }

        // Remove completed goals
        goals.retain(|g| g.progress < 1.0);

        Ok(())
    }

    /// Get current active goals
    pub async fn get_active_goals(&self) -> Vec<TrackedGoal> {
        self.active_goals.read().await.clone()
    }

    /// Prune stale goals (not updated recently)
    pub async fn prune_stale(&self, max_age: std::time::Duration) -> usize {
        let mut goals = self.active_goals.write().await;
        let before = goals.len();
        goals.retain(|g| g.last_updated.elapsed() < max_age);
        before - goals.len()
    }
}

#[async_trait]
impl Subagent for GoalAgent {
    fn agent_type(&self) -> SubagentType { SubagentType::Goal }
    // ... standard trait implementations
}
```

### Dream Agent

```rust
// crates/context-graph-mcp/src/subagents/dream_agent.rs

use crate::handlers::consolidate::{ConsolidateHandler, ConsolidationMode};

/// Memory consolidation (REM/dreaming) subagent
pub struct DreamAgent {
    consolidator: Arc<ConsolidateHandler>,
    drift_detector: Arc<DriftDetector>,
    status: RwLock<SubagentStatus>,
    last_consolidation: RwLock<Option<std::time::Instant>>,
    patterns_discovered: RwLock<Vec<PatternSummary>>,
}

impl DreamAgent {
    pub fn new(
        consolidator: Arc<ConsolidateHandler>,
        drift_detector: Arc<DriftDetector>,
    ) -> Self;

    /// Run light consolidation (quick cleanup)
    pub async fn consolidate_light(&self) -> SubagentResult<ConsolidateResult> {
        let params = ConsolidateParams {
            mode: ConsolidationMode::Light,
            salience_threshold: 0.3,
            max_memories: 0,
            dry_run: false,
        };
        self.consolidator.handle(params).await
    }

    /// Run deep consolidation (thorough cleanup)
    pub async fn consolidate_deep(&self) -> SubagentResult<ConsolidateResult> {
        let params = ConsolidateParams {
            mode: ConsolidationMode::Deep,
            salience_threshold: 0.3,
            max_memories: 0,
            dry_run: false,
        };
        self.consolidator.handle(params).await
    }

    /// Run REM/dreaming consolidation (pattern discovery)
    pub async fn dream(&self) -> SubagentResult<ConsolidateResult> {
        let params = ConsolidateParams {
            mode: ConsolidationMode::Rem,
            salience_threshold: 0.3,
            max_memories: 0,
            dry_run: false,
        };

        let result = self.consolidator.handle(params).await?;

        // Store discovered patterns
        let mut patterns = self.patterns_discovered.write().await;
        patterns.extend(result.new_patterns.clone());

        // Update last consolidation time
        *self.last_consolidation.write().await = Some(std::time::Instant::now());

        Ok(result)
    }

    /// Check if consolidation is due
    pub async fn should_consolidate(&self) -> bool {
        let last = self.last_consolidation.read().await;
        match *last {
            None => true,
            Some(t) => t.elapsed() > std::time::Duration::from_secs(3600), // 1 hour
        }
    }

    /// Get all patterns discovered across dream sessions
    pub async fn get_patterns(&self) -> Vec<PatternSummary> {
        self.patterns_discovered.read().await.clone()
    }
}

#[async_trait]
impl Subagent for DreamAgent {
    fn agent_type(&self) -> SubagentType { SubagentType::Dream }
    // ... standard trait implementations
}
```

### Subagent Manager

```rust
// crates/context-graph-mcp/src/subagents/manager.rs

/// Manages all subagents lifecycle
pub struct SubagentManager {
    embedding_agent: Arc<EmbeddingAgent>,
    search_agent: Arc<SearchAgent>,
    goal_agent: Arc<GoalAgent>,
    dream_agent: Arc<DreamAgent>,
    hook_dispatcher: Arc<HookDispatcher>,
}

impl SubagentManager {
    pub fn new(
        embedding_agent: Arc<EmbeddingAgent>,
        search_agent: Arc<SearchAgent>,
        goal_agent: Arc<GoalAgent>,
        dream_agent: Arc<DreamAgent>,
        hook_dispatcher: Arc<HookDispatcher>,
    ) -> Self;

    /// Start all subagents
    pub async fn start_all(&self) -> SubagentResult<()> {
        tokio::try_join!(
            self.embedding_agent.start(),
            self.search_agent.start(),
            self.goal_agent.start(),
            self.dream_agent.start(),
        )?;
        Ok(())
    }

    /// Stop all subagents and merge learnings
    pub async fn stop_all(&self, parent_session: SessionId) -> SubagentResult<()> {
        // Stop each and collect outputs
        let outputs = vec![
            self.embedding_agent.stop().await?,
            self.search_agent.stop().await?,
            self.goal_agent.stop().await?,
            self.dream_agent.stop().await?,
        ];

        // Dispatch SubagentStop hook for each
        for output in outputs {
            let context = SubagentStopContext {
                subagent_id: SubagentId::new(),
                parent_session: parent_session.clone(),
                task_description: format!("{:?} agent", output.agent_type),
                success: output.success,
                created_memories: output.memories_created,
                duration: output.duration,
            };
            self.hook_dispatcher.dispatch_subagent_stop(context).await?;
        }

        Ok(())
    }

    /// Get health status of all subagents
    pub async fn health_check_all(&self) -> HashMap<SubagentType, SubagentHealth> {
        let mut health = HashMap::new();
        health.insert(SubagentType::Embedding, self.embedding_agent.health_check().await);
        health.insert(SubagentType::Search, self.search_agent.health_check().await);
        health.insert(SubagentType::Goal, self.goal_agent.health_check().await);
        health.insert(SubagentType::Dream, self.dream_agent.health_check().await);
        health
    }

    /// Get specific subagent
    pub fn embedding(&self) -> Arc<EmbeddingAgent> { self.embedding_agent.clone() }
    pub fn search(&self) -> Arc<SearchAgent> { self.search_agent.clone() }
    pub fn goal(&self) -> Arc<GoalAgent> { self.goal_agent.clone() }
    pub fn dream(&self) -> Arc<DreamAgent> { self.dream_agent.clone() }
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Subagent startup | < 500ms |
| Health check | < 10ms |
| Stop with merge | < 5s (SubagentStop timeout) |
| Max concurrent tasks | 100 per agent |

## Verification

- [ ] All 4 subagents implement Subagent trait
- [ ] EmbeddingAgent handles GPU memory coordination
- [ ] SearchAgent uses entry-point selection
- [ ] GoalAgent discovers and tracks goals
- [ ] DreamAgent performs consolidation
- [ ] SubagentStop hook triggers on stop
- [ ] Learnings merged into parent session
- [ ] Health checks return accurate status

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/subagents/mod.rs` | Subagent traits and types |
| `crates/context-graph-mcp/src/subagents/embedding_agent.rs` | Embedding subagent |
| `crates/context-graph-mcp/src/subagents/search_agent.rs` | Search subagent |
| `crates/context-graph-mcp/src/subagents/goal_agent.rs` | Goal subagent |
| `crates/context-graph-mcp/src/subagents/dream_agent.rs` | Dream subagent |
| `crates/context-graph-mcp/src/subagents/manager.rs` | Subagent lifecycle manager |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Subagent deadlock | Medium | High | Timeouts, health checks |
| Memory leak from stopped agents | Low | Medium | RAII, explicit cleanup |
| Learning loss on crash | Medium | Medium | Periodic checkpoints |

## Traceability

- Source: Constitution subagent specifications
- Depends on: TASK-INTEG-012 (SubagentStop hook)
