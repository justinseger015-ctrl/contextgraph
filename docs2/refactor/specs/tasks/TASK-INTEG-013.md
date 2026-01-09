# TASK-INTEG-013: consolidate_memories MCP Tool and Skill

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-INTEG-013 |
| **Title** | consolidate_memories MCP Tool and Skill |
| **Status** | :white_circle: todo |
| **Layer** | Integration |
| **Sequence** | 33 |
| **Estimated Days** | 2 |
| **Complexity** | Medium |

## Implements

- **ARCH-06**: MCP tool boundary - consolidate_memories
- Constitution MCP tools list (line 273)
- Constitution skills list (lines 743-749)

## Dependencies

| Task | Reason |
|------|--------|
| TASK-INTEG-001 | MCP handler infrastructure |
| TASK-LOGIC-009 | Goal discovery for salience |
| TASK-LOGIC-010 | Drift detection for pruning decisions |

## Objective

Implement the `consolidate_memories` MCP tool and corresponding `consolidate` skill with Light/Deep/REM modes for memory dreaming and pruning as specified in the constitution.

## Context

**Constitution Requirements:**

The constitution specifies 5 MCP tools (line 268-280):
- `inject_context` ✓ (covered)
- `store_memory` ✓ (covered)
- `search_graph` ✓ (covered)
- `discover_goals` ✓ (covered)
- **`consolidate_memories`** ❌ **NOT COVERED**

And 4 skills (lines 725-749):
- `memory-inject` ✓
- `semantic-search` ✓
- `goal-discovery` ✓
- **`consolidate`** ❌ **NOT COVERED**

Rate limit: 1 req/min per session (SEC-03)

## Scope

### In Scope

- `consolidate_memories` MCP tool handler
- `consolidate` skill YAML definition
- Light consolidation mode (quick, low impact)
- Deep consolidation mode (thorough cleanup)
- REM/dreaming mode (pattern discovery, sleep-like)
- Salience threshold pruning
- Memory clustering and deduplication

### Out of Scope

- Memory backup/restore
- Cross-session consolidation (handled by SubagentStop hook)
- Distributed consolidation

## Definition of Done

### MCP Tool Handler

```rust
// crates/context-graph-mcp/src/handlers/consolidate.rs

use serde::{Deserialize, Serialize};

/// consolidate_memories MCP tool handler
pub struct ConsolidateHandler {
    store: Arc<TeleologicalArrayStore>,
    comparator: Arc<TeleologicalComparator>,
    goal_discoverer: Arc<GoalDiscoverer>,
    drift_detector: Arc<DriftDetector>,
}

/// Consolidation modes
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConsolidationMode {
    /// Quick consolidation - prune obvious duplicates, low impact
    Light,
    /// Thorough consolidation - full dedup, clustering, salience pruning
    Deep,
    /// Dreaming mode - discover patterns, simulate hippocampal replay
    Rem,
}

impl Default for ConsolidationMode {
    fn default() -> Self {
        Self::Light
    }
}

/// Input parameters for consolidate_memories
#[derive(Debug, Deserialize)]
pub struct ConsolidateParams {
    /// Consolidation mode (light, deep, rem)
    #[serde(default)]
    pub mode: ConsolidationMode,
    /// Salience threshold for pruning (0.0 - 1.0)
    #[serde(default = "default_salience_threshold")]
    pub salience_threshold: f32,
    /// Maximum memories to process (0 = unlimited)
    #[serde(default)]
    pub max_memories: usize,
    /// Whether to perform dry run (report but don't modify)
    #[serde(default)]
    pub dry_run: bool,
}

fn default_salience_threshold() -> f32 { 0.3 }

/// Output from consolidate_memories
#[derive(Debug, Serialize)]
pub struct ConsolidateResult {
    /// Mode that was executed
    pub mode: String,
    /// Memories pruned (below salience)
    pub pruned_count: usize,
    /// Memories merged (duplicates)
    pub merged_count: usize,
    /// Clusters discovered (REM mode)
    pub clusters_discovered: usize,
    /// New patterns identified
    pub new_patterns: Vec<PatternSummary>,
    /// Total memories after consolidation
    pub final_memory_count: usize,
    /// Duration of consolidation
    pub duration_ms: u64,
    /// Was this a dry run?
    pub dry_run: bool,
}

#[derive(Debug, Serialize)]
pub struct PatternSummary {
    pub pattern_id: String,
    pub description: String,
    pub memory_count: usize,
    pub confidence: f32,
}

impl ConsolidateHandler {
    pub fn new(
        store: Arc<TeleologicalArrayStore>,
        comparator: Arc<TeleologicalComparator>,
        goal_discoverer: Arc<GoalDiscoverer>,
        drift_detector: Arc<DriftDetector>,
    ) -> Self;

    /// Handle consolidate_memories MCP call
    pub async fn handle(&self, params: ConsolidateParams) -> HandlerResult<ConsolidateResult> {
        match params.mode {
            ConsolidationMode::Light => self.consolidate_light(params).await,
            ConsolidationMode::Deep => self.consolidate_deep(params).await,
            ConsolidationMode::Rem => self.consolidate_rem(params).await,
        }
    }

    /// Light mode: Quick dedup and obvious pruning
    async fn consolidate_light(&self, params: ConsolidateParams) -> HandlerResult<ConsolidateResult> {
        let start = Instant::now();

        // 1. Find near-duplicates (similarity > 0.95)
        let duplicates = self.find_duplicates(0.95).await?;

        // 2. Merge duplicates
        let merged = if !params.dry_run {
            self.merge_memories(&duplicates).await?
        } else {
            duplicates.len()
        };

        // 3. Prune very low salience (< threshold)
        let low_salience = self.find_low_salience(params.salience_threshold).await?;
        let pruned = if !params.dry_run {
            self.prune_memories(&low_salience).await?
        } else {
            low_salience.len()
        };

        Ok(ConsolidateResult {
            mode: "light".into(),
            pruned_count: pruned,
            merged_count: merged,
            clusters_discovered: 0,
            new_patterns: vec![],
            final_memory_count: self.store.count().await?,
            duration_ms: start.elapsed().as_millis() as u64,
            dry_run: params.dry_run,
        })
    }

    /// Deep mode: Full consolidation with clustering
    async fn consolidate_deep(&self, params: ConsolidateParams) -> HandlerResult<ConsolidateResult> {
        let start = Instant::now();

        // 1. Light mode operations first
        let duplicates = self.find_duplicates(0.9).await?; // Lower threshold
        let merged = if !params.dry_run {
            self.merge_memories(&duplicates).await?
        } else {
            duplicates.len()
        };

        // 2. Comprehensive salience scoring
        let salience_scores = self.score_all_salience().await?;
        let to_prune: Vec<_> = salience_scores.into_iter()
            .filter(|(_, score)| *score < params.salience_threshold)
            .map(|(id, _)| id)
            .collect();
        let pruned = if !params.dry_run {
            self.prune_memories(&to_prune).await?
        } else {
            to_prune.len()
        };

        // 3. Cluster remaining memories
        let clusters = self.cluster_memories().await?;
        let clusters_count = clusters.len();

        // 4. Update memory relationships based on clusters
        if !params.dry_run {
            self.update_cluster_links(&clusters).await?;
        }

        Ok(ConsolidateResult {
            mode: "deep".into(),
            pruned_count: pruned,
            merged_count: merged,
            clusters_discovered: clusters_count,
            new_patterns: vec![],
            final_memory_count: self.store.count().await?,
            duration_ms: start.elapsed().as_millis() as u64,
            dry_run: params.dry_run,
        })
    }

    /// REM mode: Dreaming/pattern discovery (hippocampal replay simulation)
    async fn consolidate_rem(&self, params: ConsolidateParams) -> HandlerResult<ConsolidateResult> {
        let start = Instant::now();

        // 1. All deep mode operations
        let deep_result = self.consolidate_deep(ConsolidateParams {
            mode: ConsolidationMode::Deep,
            dry_run: params.dry_run,
            ..params
        }).await?;

        // 2. Pattern discovery via goal emergence
        let patterns = self.goal_discoverer.discover_emergent_goals(
            &self.store.list_all().await?,
            10, // max patterns to discover
        ).await?;

        // 3. Drift analysis
        let drift = self.drift_detector.detect_drift(
            &self.store.list_recent(100).await?,
        ).await?;

        // 4. Create pattern summaries
        let new_patterns: Vec<PatternSummary> = patterns.into_iter()
            .map(|p| PatternSummary {
                pattern_id: p.id.to_string(),
                description: p.description,
                memory_count: p.member_count,
                confidence: p.confidence,
            })
            .collect();

        // 5. Optionally adjust goals based on drift
        if !params.dry_run && drift.significant {
            self.adjust_goals_for_drift(&drift).await?;
        }

        Ok(ConsolidateResult {
            mode: "rem".into(),
            pruned_count: deep_result.pruned_count,
            merged_count: deep_result.merged_count,
            clusters_discovered: deep_result.clusters_discovered,
            new_patterns,
            final_memory_count: self.store.count().await?,
            duration_ms: start.elapsed().as_millis() as u64,
            dry_run: params.dry_run,
        })
    }

    // Helper methods
    async fn find_duplicates(&self, threshold: f32) -> HandlerResult<Vec<(Uuid, Uuid)>>;
    async fn merge_memories(&self, duplicates: &[(Uuid, Uuid)]) -> HandlerResult<usize>;
    async fn find_low_salience(&self, threshold: f32) -> HandlerResult<Vec<Uuid>>;
    async fn prune_memories(&self, ids: &[Uuid]) -> HandlerResult<usize>;
    async fn score_all_salience(&self) -> HandlerResult<Vec<(Uuid, f32)>>;
    async fn cluster_memories(&self) -> HandlerResult<Vec<MemoryCluster>>;
    async fn update_cluster_links(&self, clusters: &[MemoryCluster]) -> HandlerResult<()>;
    async fn adjust_goals_for_drift(&self, drift: &DriftReport) -> HandlerResult<()>;
}
```

### Skill Definition

```yaml
# .claude/skills/consolidate/SKILL.md
---
name: consolidate
description: Memory consolidation with Light/Deep/REM modes for dreaming and pruning
allowed-tools:
  - mcp__context-graph__consolidate_memories
model: sonnet
---

# Consolidate Skill

## Purpose

Perform memory consolidation to:
- Remove duplicate memories
- Prune low-salience memories
- Discover emergent patterns
- Simulate "dreaming" for pattern reinforcement

## When to Use

- After intensive work sessions
- When memory count is high (>10000)
- Before long breaks or session end
- When searching feels slow

## Modes

### Light Mode (Default)
Quick cleanup:
- Dedup obvious duplicates (>95% similar)
- Prune very low salience (<threshold)
- Fast, minimal disruption

### Deep Mode
Thorough cleanup:
- Aggressive dedup (>90% similar)
- Full salience scoring
- Memory clustering
- Updates memory relationships

### REM Mode
Pattern discovery (dreaming):
- All Deep operations
- Goal/pattern emergence
- Drift detection
- Goal adjustment
- Most compute-intensive

## Process

1. Assess memory state (count, staleness)
2. Select appropriate mode
3. Run consolidation with dry_run=true first
4. Review impact
5. Execute if acceptable

## Usage

```json
{
  "mode": "light",
  "salience_threshold": 0.3,
  "dry_run": true
}
```

## Rate Limit

**1 request per minute per session** (SEC-03)

Use sparingly. Schedule at session end via SessionEnd hook.
```

### Constraints

| Constraint | Target |
|------------|--------|
| Light mode duration | < 5s |
| Deep mode duration | < 30s |
| REM mode duration | < 60s (SessionEnd timeout) |
| Rate limit | 1 req/min/session |

## Verification

- [ ] Light mode completes in < 5s
- [ ] Deep mode completes in < 30s
- [ ] REM mode discovers patterns
- [ ] dry_run returns accurate counts without modifying
- [ ] Rate limit enforced (1/min)
- [ ] Pruned memories actually removed
- [ ] Merged memories properly consolidated

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/handlers/consolidate.rs` | MCP tool handler |
| `.claude/skills/consolidate/SKILL.md` | Skill definition |
| Update `crates/context-graph-mcp/src/handlers/mod.rs` | Register handler |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data loss from pruning | Medium | Critical | dry_run first, backups |
| Long consolidation blocks | Low | Medium | Timeout enforcement |
| Pattern discovery wrong | Medium | Low | Confidence thresholds |

## Traceability

- Source: Constitution lines 273 (MCP tool), 743-749 (skill)
- Rate limit: SEC-03 (1 req/min)
