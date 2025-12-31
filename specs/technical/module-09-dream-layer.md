# Module 9: Dream Layer - Technical Specification

```yaml
metadata:
  id: TECH-DREAM-009
  version: 1.0.0
  module: Dream Layer
  phase: 8
  status: draft
  created: 2025-12-31
  dependencies:
    - TECH-GDS-008 (Module 8: GPU Direct Storage)
    - TECH-BIONV-006 (Module 6: Bio-Nervous System)
    - TECH-GRAPH-004 (Module 4: Knowledge Graph)
  functional_spec_ref: SPEC-DREAM-009
  author: Architecture Agent
```

---

## 1. Architecture Overview

### 1.1 Dream Cycle Architecture

The Dream Layer implements biologically-inspired memory consolidation through alternating NREM and REM phases using the SRC (Sparse, Random, Clustered) algorithm.

| Phase | Duration | Purpose | GPU Budget |
|-------|----------|---------|------------|
| NREM | 3 min | Memory replay, Hebbian updates, consolidation | <30% |
| REM | 2 min | Creative exploration, blind spot discovery | <30% |
| Transition | 500ms | State transfer, neuromodulation adjustment | <10% |
| Wake | <100ms | Abort and restore on user query | N/A |

### 1.2 Module Structure

```
crates/context-graph-dream/src/
├── lib.rs                    # Public API
├── layer.rs                  # DreamLayer orchestrator
├── state.rs                  # DreamState machine
├── trigger.rs                # DreamTrigger conditions
├── abort.rs                  # AbortController
├── nrem/
│   ├── mod.rs                # NREMPhase orchestration
│   ├── replay.rs             # MemoryReplayEngine
│   ├── hebbian.rs            # HebbianUpdater
│   ├── consolidation.rs      # CouplingConsolidator
│   ├── redundancy.rs         # RedundancyDetector
│   └── schema.rs             # SchemaExtractor
├── rem/
│   ├── mod.rs                # REMPhase orchestration
│   ├── query_gen.rs          # SyntheticQueryGenerator
│   ├── blind_spot.rs         # BlindSpotDetector
│   └── edge_creator.rs       # EdgeCreator
├── src/
│   ├── algorithm.rs          # SRCAlgorithm core
│   ├── state.rs              # SRCState
│   ├── tuner.rs              # SRCTuner
│   └── shortcut.rs           # AmortizedShortcutCreation (Marblestone)
├── resource.rs               # DreamResourceManager
├── metrics.rs                # DreamMetrics collection
├── quality.rs                # DreamQualityAssessor
├── checkpoint.rs             # DreamCheckpoint persistence
└── mcp/
    └── tools.rs              # MCP tool implementations
```

---

## 2. Core Data Structures

### 2.1 DreamLayer Struct

```rust
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{watch, RwLock};
use uuid::Uuid;

/// REQ-DREAM-001: DreamLayer orchestrates the complete dream cycle
pub struct DreamLayer {
    /// NREM phase handler for memory replay and consolidation
    pub nrem_phase: Arc<RwLock<NREMPhase>>,

    /// REM phase handler for creative exploration
    pub rem_phase: Arc<RwLock<REMPhase>>,

    /// SRC algorithm for dream processing
    pub src_algorithm: Arc<RwLock<SRCAlgorithm>>,

    /// Simulated cycle duration (biological: 90 minutes -> 5 minutes)
    pub cycle_duration: Duration,

    /// Target compression ratio for NREM consolidation
    pub compression_ratio: f32,

    /// Activity monitor for trigger detection
    activity_monitor: ActivityMonitor,

    /// Dream trigger configuration
    trigger: DreamTrigger,

    /// Current dream state
    state: Arc<RwLock<DreamState>>,

    /// Abort controller
    abort_controller: AbortController,

    /// Resource manager
    resource_manager: DreamResourceManager,

    /// Metrics collector
    metrics: Arc<RwLock<DreamMetrics>>,

    /// Quality assessor
    quality_assessor: DreamQualityAssessor,

    /// Checkpoint manager
    checkpoint_manager: CheckpointManager,

    /// Knowledge graph reference
    graph: Arc<RwLock<KnowledgeGraph>>,

    /// Neuromodulation controller reference
    neuromod: Arc<RwLock<NeuromodulationController>>,
}

impl Default for DreamLayer {
    fn default() -> Self {
        Self {
            nrem_phase: Arc::new(RwLock::new(NREMPhase::default())),
            rem_phase: Arc::new(RwLock::new(REMPhase::default())),
            src_algorithm: Arc::new(RwLock::new(SRCAlgorithm::default())),
            cycle_duration: Duration::from_secs(300), // 5 minutes
            compression_ratio: 10.0,
            activity_monitor: ActivityMonitor::default(),
            trigger: DreamTrigger::default(),
            state: Arc::new(RwLock::new(DreamState::Awake)),
            abort_controller: AbortController::default(),
            resource_manager: DreamResourceManager::default(),
            metrics: Arc::new(RwLock::new(DreamMetrics::default())),
            quality_assessor: DreamQualityAssessor::default(),
            checkpoint_manager: CheckpointManager::default(),
            graph: Arc::new(RwLock::new(KnowledgeGraph::default())),
            neuromod: Arc::new(RwLock::new(NeuromodulationController::default())),
        }
    }
}
```

### 2.2 Dream State Machine

```rust
/// REQ-DREAM-001: Dream state enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum DreamState {
    /// System is awake and processing queries
    Awake,

    /// Transitioning into NREM phase
    EnteringNREM,

    /// NREM phase active
    NREM {
        started_at: Instant,
        phase: NREMSubPhase,
    },

    /// Transitioning between NREM and REM
    Transitioning,

    /// REM phase active
    REM {
        started_at: Instant,
        queries_generated: u32,
    },

    /// Waking up from dream
    Waking {
        reason: WakeReason,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum NREMSubPhase {
    /// Initial memory selection based on recency
    MemorySelection { progress: f32 },

    /// Replay with Hebbian updates
    HebbianReplay { iterations: u32 },

    /// Tight coupling consolidation
    Consolidation { clusters_processed: u32 },

    /// Redundancy elimination
    RedundancyElimination { candidates: u32 },

    /// Schema extraction
    SchemaExtraction { schemas_found: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum WakeReason {
    UserQuery,
    ResourcePressure,
    CycleComplete,
    ManualAbort,
    Error(String),
}

impl DreamState {
    /// Check if dream is active
    pub fn is_dreaming(&self) -> bool {
        matches!(
            self,
            DreamState::EnteringNREM
                | DreamState::NREM { .. }
                | DreamState::Transitioning
                | DreamState::REM { .. }
        )
    }

    /// Get phase name for logging
    pub fn phase_name(&self) -> &'static str {
        match self {
            DreamState::Awake => "awake",
            DreamState::EnteringNREM => "entering_nrem",
            DreamState::NREM { .. } => "nrem",
            DreamState::Transitioning => "transitioning",
            DreamState::REM { .. } => "rem",
            DreamState::Waking { .. } => "waking",
        }
    }
}
```

### 2.3 Dream Trigger

```rust
/// REQ-DREAM-002: Automatic dream trigger conditions
#[derive(Debug, Clone)]
pub struct DreamTrigger {
    /// Activity level below which dream may begin
    pub activity_threshold: f32,

    /// Duration of low activity before triggering
    pub idle_duration: Duration,

    /// Minimum time between dream cycles
    pub cooldown_period: Duration,

    /// Memory accumulation threshold for forced trigger
    pub memory_pressure_threshold: f32,

    /// Last dream completion time
    last_dream_completed: Option<Instant>,
}

impl Default for DreamTrigger {
    fn default() -> Self {
        Self {
            activity_threshold: 0.15,
            idle_duration: Duration::from_secs(600), // 10 minutes
            cooldown_period: Duration::from_secs(1800), // 30 minutes
            memory_pressure_threshold: 0.8,
            last_dream_completed: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TriggerDecision {
    Trigger(TriggerReason),
    Wait,
    Blocked(BlockReason),
}

#[derive(Debug, Clone)]
pub enum TriggerReason {
    IdleTimeout,
    MemoryPressure,
    Manual,
    Scheduled,
}

#[derive(Debug, Clone)]
pub enum BlockReason {
    CooldownActive { remaining: Duration },
    HighActivity { current: f32 },
    DreamInProgress,
    ResourcesUnavailable,
}

impl DreamTrigger {
    /// REQ-DREAM-002: Check if dream should be triggered
    pub fn should_trigger(&self, context: &SystemContext) -> TriggerDecision {
        // Check cooldown
        if let Some(last) = self.last_dream_completed {
            let elapsed = Instant::now().duration_since(last);
            if elapsed < self.cooldown_period {
                return TriggerDecision::Blocked(BlockReason::CooldownActive {
                    remaining: self.cooldown_period - elapsed,
                });
            }
        }

        let activity = context.get_activity_level();
        let idle_time = context.get_idle_duration();
        let memory_pressure = context.get_memory_pressure();

        // Memory pressure override
        if memory_pressure > self.memory_pressure_threshold {
            return TriggerDecision::Trigger(TriggerReason::MemoryPressure);
        }

        // Idle timeout trigger
        if activity < self.activity_threshold && idle_time >= self.idle_duration {
            return TriggerDecision::Trigger(TriggerReason::IdleTimeout);
        }

        // High activity blocks trigger
        if activity > self.activity_threshold {
            return TriggerDecision::Blocked(BlockReason::HighActivity { current: activity });
        }

        TriggerDecision::Wait
    }

    pub fn record_completion(&mut self) {
        self.last_dream_completed = Some(Instant::now());
    }
}
```

---

## 3. NREM Phase Implementation

### 3.1 NREMPhase Structure

```rust
/// REQ-DREAM-004: NREM phase for hippocampus-guided memory replay
#[derive(Debug, Clone)]
pub struct NREMPhase {
    /// Phase duration (default: 3 minutes)
    pub duration: Duration,

    /// Recency bias for memory selection (default: 0.8)
    pub replay_recency_bias: f32,

    /// Hebbian learning rate (default: 0.01)
    pub learning_rate: f32,

    /// Consolidation batch size (default: 64)
    pub batch_size: usize,

    /// Schema extractor
    schema_extractor: SchemaExtractor,

    /// Redundancy detector
    redundancy_detector: RedundancyDetector,

    /// Coupling consolidator
    consolidator: CouplingConsolidator,

    /// Memory replay engine
    replay_engine: MemoryReplayEngine,

    /// Hebbian updater
    hebbian_updater: HebbianUpdater,
}

impl Default for NREMPhase {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(180), // 3 minutes
            replay_recency_bias: 0.8,
            learning_rate: 0.01,
            batch_size: 64,
            schema_extractor: SchemaExtractor::default(),
            redundancy_detector: RedundancyDetector::default(),
            consolidator: CouplingConsolidator::default(),
            replay_engine: MemoryReplayEngine::default(),
            hebbian_updater: HebbianUpdater::default(),
        }
    }
}

impl NREMPhase {
    /// Execute complete NREM phase
    pub async fn execute(
        &mut self,
        graph: &mut KnowledgeGraph,
        src: &mut SRCAlgorithm,
        abort_rx: &watch::Receiver<bool>,
        metrics: &mut NREMMetrics,
    ) -> Result<NREMResult, DreamError> {
        let start = Instant::now();
        let deadline = start + self.duration;

        // Phase 1: Memory Selection
        let memories = self.replay_engine.select_memories(graph);
        metrics.memories_replayed = memories.len() as u32;

        if *abort_rx.borrow() {
            return Ok(NREMResult::aborted(start.elapsed()));
        }

        // Phase 2: Hebbian Replay with SRC
        let mut replay_paths = Vec::new();
        for batch in memories.chunks(self.batch_size) {
            if *abort_rx.borrow() || Instant::now() > deadline {
                break;
            }

            let batch_result = self.hebbian_updater
                .batch_update(batch, graph, src)
                .await?;

            metrics.edges_strengthened += batch_result.edges_strengthened;
            metrics.edges_weakened += batch_result.edges_weakened;
            metrics.edges_pruned += batch_result.edges_pruned;
            replay_paths.extend(batch_result.replay_paths);
        }

        // Phase 3: Amortized Shortcut Creation (Marblestone REQ-DREAM-036/037)
        let shortcuts = src.amortized_shortcut_creation(&replay_paths, 3, 0.7).await;
        for shortcut in shortcuts {
            graph.add_edge(shortcut)?;
        }

        if *abort_rx.borrow() {
            return Ok(NREMResult::aborted(start.elapsed()));
        }

        // Phase 4: Tight Coupling Consolidation
        let clusters = self.consolidator.find_coupled_clusters(graph);
        for cluster in clusters {
            if *abort_rx.borrow() || Instant::now() > deadline {
                break;
            }

            if let Ok(result) = self.consolidator.consolidate_cluster(&cluster, graph).await {
                metrics.clusters_consolidated += 1;
                metrics.nodes_merged += result.nodes_consolidated as u32;
            }
        }

        // Phase 5: Redundancy Elimination
        let redundant_pairs = self.redundancy_detector.find_redundant_pairs(graph).await;
        let redundancy_stats = self.redundancy_detector
            .eliminate_redundancy(redundant_pairs, graph)
            .await?;
        metrics.redundancies_eliminated = redundancy_stats.nodes_eliminated;

        // Phase 6: Schema Extraction
        let schemas = self.schema_extractor.extract_schemas(
            self.consolidator.find_coupled_clusters(graph)
        ).await;
        metrics.schemas_extracted = schemas.len() as u32;

        // Calculate compression ratio
        metrics.compression_ratio = self.calculate_compression(graph);
        metrics.duration_ms = start.elapsed().as_millis() as u64;

        Ok(NREMResult {
            completed: true,
            metrics: metrics.clone(),
            replay_paths,
        })
    }

    fn calculate_compression(&self, graph: &KnowledgeGraph) -> f32 {
        // Compression = initial_nodes / current_nodes
        let initial = graph.initial_node_count_this_cycle();
        let current = graph.node_count();
        if current == 0 { 1.0 } else { initial as f32 / current as f32 }
    }
}
```

### 3.2 Memory Replay Engine

```rust
/// REQ-DREAM-005: Memory replay with recency bias
#[derive(Debug, Clone)]
pub struct MemoryReplayEngine {
    /// Recency weight (higher = more recent bias)
    pub recency_bias: f32,

    /// Maximum memories to replay per cycle
    pub max_replay_count: usize,

    /// Importance floor for replay eligibility
    pub importance_floor: f32,
}

impl Default for MemoryReplayEngine {
    fn default() -> Self {
        Self {
            recency_bias: 0.8,
            max_replay_count: 1000,
            importance_floor: 0.3,
        }
    }
}

impl MemoryReplayEngine {
    /// REQ-DREAM-005: Select memories for replay using recency-weighted sampling
    pub fn select_memories(&self, graph: &KnowledgeGraph) -> Vec<MemoryNode> {
        let candidates = graph.get_nodes_since(Duration::from_secs(86400)); // 24 hours

        // Weight by recency: w(t) = recency_bias^(age_hours / 24)
        let weighted: Vec<(MemoryNode, f32)> = candidates
            .into_iter()
            .filter(|n| n.importance >= self.importance_floor)
            .map(|n| {
                let age_hours = n.age().as_secs_f32() / 3600.0;
                let weight = self.recency_bias.powf(age_hours / 24.0);
                (n, weight * n.importance)
            })
            .collect();

        // Sample without replacement, weighted by score
        weighted_sample(&weighted, self.max_replay_count)
    }

    /// Replay a single memory, strengthening connections
    pub async fn replay_memory(
        &self,
        node: &MemoryNode,
        context: &mut ReplayContext,
    ) -> Result<ReplayResult, DreamError> {
        // Activate node and propagate to neighbors
        context.activate(node.id, 1.0);

        // Get connected nodes
        let neighbors = context.graph.get_neighbors(node.id);

        // Co-activate based on edge weights
        for (neighbor_id, weight) in neighbors {
            let activation = weight * context.get_activation(node.id);
            context.activate(neighbor_id, activation);
        }

        Ok(ReplayResult {
            node_id: node.id,
            activations: context.get_active_nodes(),
        })
    }
}

/// Weighted sampling without replacement
fn weighted_sample<T: Clone>(items: &[(T, f32)], count: usize) -> Vec<T> {
    use rand::prelude::*;

    let mut rng = thread_rng();
    let total_weight: f32 = items.iter().map(|(_, w)| w).sum();

    if total_weight <= 0.0 {
        return Vec::new();
    }

    let mut selected = Vec::with_capacity(count.min(items.len()));
    let mut remaining: Vec<_> = items.iter().cloned().collect();

    while selected.len() < count && !remaining.is_empty() {
        let total: f32 = remaining.iter().map(|(_, w)| w).sum();
        let threshold = rng.gen::<f32>() * total;

        let mut cumulative = 0.0;
        let mut idx = 0;
        for (i, (_, w)) in remaining.iter().enumerate() {
            cumulative += w;
            if cumulative >= threshold {
                idx = i;
                break;
            }
        }

        selected.push(remaining.remove(idx).0);
    }

    selected
}
```

### 3.3 Hebbian Weight Update

```rust
/// REQ-DREAM-006: Hebbian learning for connection strengthening
#[derive(Debug, Clone)]
pub struct HebbianUpdater {
    /// Learning rate for weight updates (eta)
    pub learning_rate: f32,

    /// Weight decay factor
    pub decay_rate: f32,

    /// Maximum weight value
    pub weight_cap: f32,

    /// Minimum weight before pruning
    pub weight_floor: f32,
}

impl Default for HebbianUpdater {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,  // eta
            decay_rate: 0.001,
            weight_cap: 1.0,
            weight_floor: 0.05,
        }
    }
}

impl HebbianUpdater {
    /// REQ-DREAM-006: Apply Hebbian update: delta_w = eta * pre * post
    #[inline]
    pub fn update_weight(
        &self,
        edge: &mut GraphEdge,
        pre_activation: f32,
        post_activation: f32,
    ) {
        // Hebbian update: "neurons that fire together wire together"
        let delta_w = self.learning_rate * pre_activation * post_activation;

        // Apply decay
        let decayed = edge.weight * (1.0 - self.decay_rate);

        // Update with cap
        edge.weight = (decayed + delta_w).clamp(self.weight_floor, self.weight_cap);

        // Mark for potential pruning if below floor
        if edge.weight <= self.weight_floor {
            edge.mark_for_pruning();
        }
    }

    /// Batch update for all edges in replay sequence
    pub async fn batch_update(
        &self,
        replay_sequence: &[MemoryNode],
        graph: &mut KnowledgeGraph,
        src: &mut SRCAlgorithm,
    ) -> Result<HebbianUpdateStats, DreamError> {
        let mut stats = HebbianUpdateStats::default();
        let mut replay_paths = Vec::new();

        // Build activation map from replay
        let mut activations: HashMap<Uuid, f32> = HashMap::new();

        for node in replay_sequence {
            // Run SRC step to propagate activation
            let src_result = src.step(&mut SRCState::new(&activations), graph);

            // Update activations
            for (node_id, activation) in src_result.new_activations {
                activations.insert(node_id, activation);
            }
        }

        // Apply Hebbian updates to edges between co-activated nodes
        let active_nodes: Vec<_> = activations.keys().cloned().collect();

        for i in 0..active_nodes.len() {
            for j in (i + 1)..active_nodes.len() {
                let node_a = active_nodes[i];
                let node_b = active_nodes[j];

                if let Some(edge) = graph.get_edge_mut(node_a, node_b) {
                    let pre = activations.get(&node_a).copied().unwrap_or(0.0);
                    let post = activations.get(&node_b).copied().unwrap_or(0.0);

                    let old_weight = edge.weight;
                    self.update_weight(edge, pre, post);

                    if edge.weight > old_weight {
                        stats.edges_strengthened += 1;
                    } else if edge.weight < old_weight {
                        stats.edges_weakened += 1;
                    }

                    if edge.is_marked_for_pruning() {
                        stats.edges_pruned += 1;
                    }
                }
            }
        }

        // Track replay paths for shortcut creation
        replay_paths.push(ReplayPath {
            nodes: active_nodes.clone(),
            edges: self.extract_path_edges(&active_nodes, graph),
            total_confidence: self.compute_path_confidence(&active_nodes, graph),
        });

        stats.replay_paths = replay_paths;
        stats.average_delta = self.compute_average_delta(graph);

        Ok(stats)
    }

    fn extract_path_edges(&self, nodes: &[Uuid], graph: &KnowledgeGraph) -> Vec<GraphEdge> {
        let mut edges = Vec::new();
        for i in 0..(nodes.len().saturating_sub(1)) {
            if let Some(edge) = graph.get_edge(nodes[i], nodes[i + 1]) {
                edges.push(edge.clone());
            }
        }
        edges
    }

    fn compute_path_confidence(&self, nodes: &[Uuid], graph: &KnowledgeGraph) -> f32 {
        let mut confidence = 1.0;
        for i in 0..(nodes.len().saturating_sub(1)) {
            if let Some(edge) = graph.get_edge(nodes[i], nodes[i + 1]) {
                confidence *= edge.confidence;
            }
        }
        confidence
    }

    fn compute_average_delta(&self, _graph: &KnowledgeGraph) -> f32 {
        // Implementation: track deltas during updates
        0.0
    }
}

#[derive(Debug, Clone, Default)]
pub struct HebbianUpdateStats {
    pub edges_strengthened: u32,
    pub edges_weakened: u32,
    pub edges_pruned: u32,
    pub average_delta: f32,
    pub replay_paths: Vec<ReplayPath>,
}
```

---

## 4. REM Phase Implementation

### 4.1 REMPhase Structure

```rust
/// REQ-DREAM-010: REM phase for creative exploration
#[derive(Debug, Clone)]
pub struct REMPhase {
    /// Phase duration (default: 2 minutes)
    pub duration: Duration,

    /// Number of synthetic queries to generate
    pub synthetic_query_count: u32,

    /// Minimum semantic leap for connection
    pub min_semantic_leap: f32,

    /// Exploration temperature
    pub exploration_temperature: f32,

    /// New edge initial weight
    pub new_edge_weight: f32,

    /// New edge initial confidence
    pub new_edge_confidence: f32,

    /// Query generator
    query_generator: SyntheticQueryGenerator,

    /// Blind spot detector
    blind_spot_detector: BlindSpotDetector,

    /// Edge creator
    edge_creator: EdgeCreator,
}

impl Default for REMPhase {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(120), // 2 minutes
            synthetic_query_count: 100,
            min_semantic_leap: 0.7,
            exploration_temperature: 2.0,
            new_edge_weight: 0.3,
            new_edge_confidence: 0.5,
            query_generator: SyntheticQueryGenerator::default(),
            blind_spot_detector: BlindSpotDetector::default(),
            edge_creator: EdgeCreator::default(),
        }
    }
}

impl REMPhase {
    /// REQ-DREAM-010: Execute REM phase exploration
    pub async fn execute(
        &mut self,
        graph: &mut KnowledgeGraph,
        abort_rx: &watch::Receiver<bool>,
        metrics: &mut REMMetrics,
    ) -> Result<REMResult, DreamError> {
        let start = Instant::now();
        let deadline = start + self.duration;

        let mut queries_generated = 0u32;
        let mut connections_discovered = 0u32;
        let mut blind_spots_found = 0u32;
        let mut semantic_leaps = Vec::new();

        while Instant::now() < deadline && queries_generated < self.synthetic_query_count {
            // Check for abort
            if *abort_rx.borrow() {
                return Ok(REMResult::aborted(
                    queries_generated,
                    connections_discovered,
                    blind_spots_found,
                ));
            }

            // Generate synthetic query via random walk
            let query = self.query_generator.generate(graph);
            queries_generated += 1;

            // Explore for blind spots
            if let Some(discovery) = self.blind_spot_detector.detect(&query, graph).await {
                // Create edge for discovered blind spot
                if discovery.semantic_distance >= self.min_semantic_leap {
                    let edge = self.edge_creator.create_from_blind_spot(&discovery, graph)?;
                    connections_discovered += 1;
                    semantic_leaps.push(discovery.semantic_distance);

                    if discovery.is_significant_blind_spot() {
                        blind_spots_found += 1;
                    }
                }
            }
        }

        metrics.synthetic_queries = queries_generated;
        metrics.blind_spots_found = blind_spots_found;
        metrics.new_edges_created = connections_discovered;
        metrics.average_semantic_leap = if semantic_leaps.is_empty() {
            0.0
        } else {
            semantic_leaps.iter().sum::<f32>() / semantic_leaps.len() as f32
        };
        metrics.exploration_coverage = self.compute_coverage(graph);
        metrics.duration_ms = start.elapsed().as_millis() as u64;

        Ok(REMResult {
            queries_generated,
            connections_discovered,
            blind_spots_found,
            phase_completed: Instant::now() >= deadline,
        })
    }

    fn compute_coverage(&self, graph: &KnowledgeGraph) -> f32 {
        // Coverage = unique nodes visited / total nodes
        let visited = graph.nodes_visited_this_cycle();
        let total = graph.node_count();
        if total == 0 { 0.0 } else { visited as f32 / total as f32 }
    }
}
```

### 4.2 Synthetic Query Generator

```rust
/// REQ-DREAM-011: Synthetic query generation via random walk in hyperbolic space
#[derive(Debug, Clone)]
pub struct SyntheticQueryGenerator {
    /// Random walk step size in hyperbolic space
    pub step_size: f32,

    /// Walk length (number of steps)
    pub walk_length: u32,

    /// Exploration bias (higher = more diverse)
    pub exploration_bias: f32,

    /// Hyperbolic curvature (Poincare ball: -1.0)
    pub curvature: f32,
}

impl Default for SyntheticQueryGenerator {
    fn default() -> Self {
        Self {
            step_size: 0.3,
            walk_length: 5,
            exploration_bias: 0.7,
            curvature: -1.0,
        }
    }
}

impl SyntheticQueryGenerator {
    /// REQ-DREAM-011: Generate synthetic query via random walk
    pub fn generate(&self, graph: &KnowledgeGraph) -> SyntheticQuery {
        use rand::prelude::*;
        let mut rng = thread_rng();

        // Select random starting node weighted by importance
        let start = graph.sample_node_weighted_by_importance(&mut rng);

        // Perform random walk in hyperbolic space
        let mut position = start.hyperbolic_position.clone();
        let mut path = Vec::with_capacity(self.walk_length as usize);

        for _ in 0..self.walk_length {
            // Generate random direction in tangent space
            let direction = self.sample_tangent_vector(&position, &mut rng, graph);

            // Apply hyperbolic exponential map
            position = self.hyperbolic_exp_map(&position, &direction, self.step_size);

            // Ensure we stay in Poincare ball (||x|| < 1)
            position = self.project_to_ball(&position);

            path.push(position.clone());
        }

        // Convert final position to query embedding
        SyntheticQuery {
            embedding: position.to_embedding(),
            path,
            origin_node: start.id,
        }
    }

    /// Sample tangent vector with exploration bias toward sparse regions
    fn sample_tangent_vector(
        &self,
        position: &HyperbolicPoint,
        rng: &mut impl Rng,
        graph: &KnowledgeGraph,
    ) -> TangentVector {
        // Random unit vector
        let random_dir = TangentVector::random_unit(rng);

        // Gradient toward sparse regions
        let sparse_gradient = self.compute_sparsity_gradient(position, graph);

        // Blend: higher exploration_bias = more random
        random_dir.blend(&sparse_gradient, self.exploration_bias)
    }

    /// Compute gradient toward sparse regions of the graph
    fn compute_sparsity_gradient(
        &self,
        position: &HyperbolicPoint,
        graph: &KnowledgeGraph,
    ) -> TangentVector {
        // Find nearest nodes
        let neighbors = graph.knn(&position.to_embedding(), 10);

        if neighbors.is_empty() {
            return TangentVector::zero();
        }

        // Compute average direction away from neighbors (toward sparse regions)
        let mut gradient = TangentVector::zero();
        for neighbor in &neighbors {
            let diff = position.subtract(&neighbor.hyperbolic_position);
            gradient = gradient.add(&diff.to_tangent());
        }

        gradient.normalize()
    }

    /// Hyperbolic exponential map (Poincare ball model)
    fn hyperbolic_exp_map(
        &self,
        position: &HyperbolicPoint,
        direction: &TangentVector,
        step: f32,
    ) -> HyperbolicPoint {
        let norm = direction.norm();
        if norm < 1e-6 {
            return position.clone();
        }

        let c = -self.curvature;
        let lambda = 2.0 / (1.0 - position.norm_squared());
        let scaled_step = step * lambda;

        // Mobius addition for Poincare ball
        let scaled_dir = direction.scale(scaled_step.tanh() / norm);
        position.mobius_add(&scaled_dir.to_point(), c)
    }

    /// Project point back into Poincare ball (||x|| < 1)
    fn project_to_ball(&self, point: &HyperbolicPoint) -> HyperbolicPoint {
        let norm = point.norm();
        if norm >= 0.99 {
            point.scale(0.99 / norm)
        } else {
            point.clone()
        }
    }
}

#[derive(Debug, Clone)]
pub struct SyntheticQuery {
    pub embedding: Vector1536,
    pub path: Vec<HyperbolicPoint>,
    pub origin_node: Uuid,
}
```

### 4.3 Blind Spot Detection

```rust
/// REQ-DREAM-012: Blind spot discovery (high semantic distance + shared causal paths)
#[derive(Debug, Clone)]
pub struct BlindSpotDetector {
    /// Minimum semantic distance for blind spot
    pub min_semantic_distance: f32,

    /// Minimum causal path overlap
    pub min_causal_overlap: f32,

    /// Maximum search depth for causal paths
    pub max_causal_depth: u32,
}

impl Default for BlindSpotDetector {
    fn default() -> Self {
        Self {
            min_semantic_distance: 0.7,
            min_causal_overlap: 0.3,
            max_causal_depth: 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlindSpot {
    pub node_a: Uuid,
    pub node_b: Uuid,
    pub semantic_distance: f32,
    pub shared_causal_paths: Vec<CausalPath>,
    pub discovery_confidence: f32,
}

impl BlindSpot {
    pub fn is_significant_blind_spot(&self) -> bool {
        self.semantic_distance >= 0.7 && self.discovery_confidence >= 0.5
    }
}

impl BlindSpotDetector {
    /// REQ-DREAM-012: Detect blind spot between query location and existing nodes
    pub async fn detect(
        &self,
        query: &SyntheticQuery,
        graph: &KnowledgeGraph,
    ) -> Option<BlindSpot> {
        // Find nearest neighbors at different semantic distances
        let near_nodes = graph.knn(&query.embedding, 10);
        let far_nodes = graph.furthest_nodes(&query.embedding, 10);

        // Check for blind spot pattern: far semantically but causally connected
        for far_node in &far_nodes {
            let semantic_dist = cosine_distance(&query.embedding, &far_node.embedding);

            if semantic_dist < self.min_semantic_distance {
                continue;
            }

            // Check for shared causal paths with near nodes
            for near_node in &near_nodes {
                let shared_paths = graph.find_shared_causal_paths(
                    near_node.id,
                    far_node.id,
                    self.max_causal_depth,
                );

                let overlap = self.compute_path_overlap(&shared_paths);

                if overlap >= self.min_causal_overlap {
                    return Some(BlindSpot {
                        node_a: near_node.id,
                        node_b: far_node.id,
                        semantic_distance: semantic_dist,
                        shared_causal_paths: shared_paths,
                        discovery_confidence: overlap * semantic_dist,
                    });
                }
            }
        }

        None
    }

    fn compute_path_overlap(&self, paths: &[CausalPath]) -> f32 {
        if paths.is_empty() {
            return 0.0;
        }

        // Overlap = sum of path confidences / max possible
        let total_confidence: f32 = paths.iter()
            .map(|p| p.confidence)
            .sum();

        (total_confidence / paths.len() as f32).min(1.0)
    }
}

/// Cosine distance (1 - cosine_similarity)
fn cosine_distance(a: &Vector1536, b: &Vector1536) -> f32 {
    1.0 - cosine_similarity(a, b)
}
```

---

## 5. SRC Algorithm Implementation

### 5.1 SRC Core Algorithm

```rust
/// REQ-DREAM-014: SRC (Sparse, Random, Clustered) algorithm
#[derive(Debug, Clone)]
pub struct SRCAlgorithm {
    /// Fraction of nodes activated per step (sparsity)
    pub sparsity: f32,

    /// Exploration vs exploitation balance
    pub randomness: f32,

    /// Locality preference for activation
    pub clustering: f32,

    /// Activation decay rate
    pub decay_rate: f32,

    /// SRC tuner for adaptive parameters
    tuner: SRCTuner,
}

impl Default for SRCAlgorithm {
    fn default() -> Self {
        Self {
            sparsity: 0.1,    // 10% of nodes
            randomness: 0.5,   // Balance exploration/exploitation
            clustering: 0.7,   // Prefer local activation
            decay_rate: 0.9,   // Gradual decay
            tuner: SRCTuner::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SRCState {
    pub activations: HashMap<Uuid, f32>,
    pub step_count: u32,
    pub history: VecDeque<SRCStepResult>,
}

impl SRCState {
    pub fn new(initial: &HashMap<Uuid, f32>) -> Self {
        Self {
            activations: initial.clone(),
            step_count: 0,
            history: VecDeque::with_capacity(100),
        }
    }

    pub fn average_activation(&self) -> f32 {
        if self.activations.is_empty() {
            0.0
        } else {
            self.activations.values().sum::<f32>() / self.activations.len() as f32
        }
    }

    pub fn sample_active_node(&self, rng: &mut impl rand::Rng) -> Option<Uuid> {
        use rand::prelude::*;
        self.activations.keys().choose(rng).copied()
    }
}

#[derive(Debug, Clone)]
pub struct SRCStepResult {
    pub nodes_activated: usize,
    pub average_activation: f32,
    pub clustering_coefficient: f32,
    pub new_activations: HashMap<Uuid, f32>,
}

impl SRCAlgorithm {
    /// REQ-DREAM-014: Perform one SRC step
    pub fn step(&self, state: &mut SRCState, graph: &KnowledgeGraph) -> SRCStepResult {
        use rand::prelude::*;
        let mut rng = thread_rng();

        // Decay existing activations
        state.activations.iter_mut().for_each(|(_, v)| *v *= self.decay_rate);

        // Calculate number of nodes to activate
        let k = (graph.node_count() as f32 * self.sparsity) as usize;

        // Select activation candidates
        let candidates = self.select_candidates(state, graph, k, &mut rng);

        // Activate selected nodes
        let mut new_activations = HashMap::new();
        for node_id in candidates {
            let activation = self.compute_activation(node_id, state, graph);
            state.activations.insert(node_id, activation);
            new_activations.insert(node_id, activation);
        }

        // Prune low activations
        state.activations.retain(|_, v| *v > 0.01);

        state.step_count += 1;

        let result = SRCStepResult {
            nodes_activated: state.activations.len(),
            average_activation: state.average_activation(),
            clustering_coefficient: self.compute_clustering_coefficient(state, graph),
            new_activations,
        };

        // Store in history
        if state.history.len() >= 100 {
            state.history.pop_front();
        }
        state.history.push_back(result.clone());

        result
    }

    /// Select candidate nodes for activation
    fn select_candidates(
        &self,
        state: &SRCState,
        graph: &KnowledgeGraph,
        k: usize,
        rng: &mut impl rand::Rng,
    ) -> Vec<Uuid> {
        let mut candidates = Vec::with_capacity(k);

        for _ in 0..k {
            if rng.gen::<f32>() < self.randomness {
                // Random selection (exploration)
                if let Some(node) = graph.random_node(rng) {
                    candidates.push(node.id);
                }
            } else {
                // Clustered selection (exploitation)
                if let Some(active_node) = state.sample_active_node(rng) {
                    // Select neighbor with probability proportional to clustering
                    if rng.gen::<f32>() < self.clustering {
                        if let Some(neighbor) = graph.random_neighbor(active_node, rng) {
                            candidates.push(neighbor);
                            continue;
                        }
                    }
                }
                // Fallback to random
                if let Some(node) = graph.random_node(rng) {
                    candidates.push(node.id);
                }
            }
        }

        candidates
    }

    fn compute_activation(
        &self,
        node_id: Uuid,
        state: &SRCState,
        graph: &KnowledgeGraph,
    ) -> f32 {
        // Base activation from neighbors
        let neighbors = graph.get_neighbors(node_id);
        let neighbor_activation: f32 = neighbors
            .iter()
            .filter_map(|(n, w)| state.activations.get(n).map(|a| a * w))
            .sum();

        // Node importance boost
        let importance = graph.get_node(node_id).map(|n| n.importance).unwrap_or(0.5);

        // Combined activation
        (neighbor_activation * 0.7 + importance * 0.3).min(1.0)
    }

    fn compute_clustering_coefficient(&self, state: &SRCState, graph: &KnowledgeGraph) -> f32 {
        // Fraction of active nodes that are neighbors of other active nodes
        let active_nodes: Vec<_> = state.activations.keys().cloned().collect();

        if active_nodes.len() < 2 {
            return 0.0;
        }

        let mut connected_pairs = 0;
        let total_pairs = active_nodes.len() * (active_nodes.len() - 1) / 2;

        for i in 0..active_nodes.len() {
            for j in (i + 1)..active_nodes.len() {
                if graph.has_edge(active_nodes[i], active_nodes[j]) {
                    connected_pairs += 1;
                }
            }
        }

        connected_pairs as f32 / total_pairs as f32
    }

    /// REQ-DREAM-036/037: Amortized shortcut creation during sleep replay (Marblestone)
    ///
    /// During sleep replay, identifies causal chains with 3+ hops that are
    /// traversed frequently and creates direct shortcut edges to amortize
    /// future inference costs.
    ///
    /// # Algorithm
    /// 1. Identify paths with traversal_count > threshold
    /// 2. For 3+ hop paths, create direct source->target edge
    /// 3. Mark edge as is_amortized_shortcut = true
    /// 4. Set edge weight as product of path weights
    ///
    /// # Quality Gates
    /// - Path confidence >= 0.7 (all edges)
    /// - Traversal count >= 5 in current replay session
    /// - No existing shortcut for this path
    pub async fn amortized_shortcut_creation(
        &self,
        replay_paths: &[ReplayPath],
        min_hops: usize,
        confidence_threshold: f32,
    ) -> Vec<GraphEdge> {
        let mut shortcuts = Vec::new();
        let mut created_shortcuts: HashSet<(Uuid, Uuid)> = HashSet::new();

        for path in replay_paths {
            // REQ-DREAM-036: Skip paths that are too short (< 3 hops)
            if path.edges.len() < min_hops {
                continue;
            }

            // Skip if we already created a shortcut for this path
            let path_key = (
                path.nodes.first().copied().unwrap_or_default(),
                path.nodes.last().copied().unwrap_or_default(),
            );
            if created_shortcuts.contains(&path_key) {
                continue;
            }

            // REQ-DREAM-037: Quality gate - path confidence >= 0.7
            let path_confidence: f32 = path.edges.iter()
                .map(|e| e.confidence)
                .product();

            if path_confidence < confidence_threshold {
                continue;
            }

            // Calculate path weight as product of edge weights
            let path_weight: f32 = path.edges.iter()
                .map(|e| e.weight)
                .product();

            // Create shortcut edge
            let shortcut = GraphEdge {
                id: Uuid::new_v4(),
                source: path.nodes.first().copied().unwrap_or_default(),
                target: path.nodes.last().copied().unwrap_or_default(),
                edge_type: EdgeType::Causal,
                weight: path_weight,
                confidence: path_confidence,
                is_amortized_shortcut: true,
                shortcut_path_length: path.edges.len() as u32,
                steering_reward: 0.0,
                neurotransmitter_weights: NeurotransmitterWeights::default(),
                created_at: chrono::Utc::now(),
                metadata: EdgeMetadata {
                    origin: EdgeOrigin::AmortizedShortcut,
                    original_path_hops: path.edges.len() as u32,
                    creation_cycle: 0, // Set by caller
                },
            };

            created_shortcuts.insert(path_key);
            shortcuts.push(shortcut);
        }

        shortcuts
    }
}

/// REQ-DREAM-036: Path identified during sleep replay for potential shortcut creation
#[derive(Debug, Clone)]
pub struct ReplayPath {
    pub nodes: Vec<Uuid>,
    pub edges: Vec<GraphEdge>,
    pub total_confidence: f32,
}

impl ReplayPath {
    pub fn source(&self) -> Option<Uuid> {
        self.nodes.first().copied()
    }

    pub fn target(&self) -> Option<Uuid> {
        self.nodes.last().copied()
    }

    pub fn hop_count(&self) -> usize {
        self.edges.len()
    }
}
```

### 5.2 SRC Parameter Tuning

```rust
/// REQ-DREAM-015: Dynamic SRC parameter tuning
#[derive(Debug, Clone)]
pub struct SRCTuner {
    /// Target activation level
    pub target_activation: f32,

    /// Adjustment rate
    pub adjustment_rate: f32,

    /// Parameter bounds
    pub sparsity_bounds: (f32, f32),
    pub randomness_bounds: (f32, f32),
    pub clustering_bounds: (f32, f32),
}

impl Default for SRCTuner {
    fn default() -> Self {
        Self {
            target_activation: 0.05,
            adjustment_rate: 0.1,
            sparsity_bounds: (0.01, 0.3),
            randomness_bounds: (0.1, 0.9),
            clustering_bounds: (0.3, 0.95),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SRCStats {
    pub average_activation: f32,
    pub unique_nodes_visited: usize,
    pub expected_coverage: usize,
    pub consolidation_rate: f32,
}

impl SRCTuner {
    /// REQ-DREAM-015: Adjust parameters based on observed dynamics
    pub fn tune(&self, src: &mut SRCAlgorithm, stats: &SRCStats) {
        let activation_error = stats.average_activation - self.target_activation;

        // Adjust sparsity to hit target activation
        if activation_error.abs() > 0.01 {
            let adjustment = -activation_error * self.adjustment_rate;
            src.sparsity = (src.sparsity + adjustment)
                .clamp(self.sparsity_bounds.0, self.sparsity_bounds.1);
        }

        // Adjust randomness based on exploration needs
        if stats.unique_nodes_visited < stats.expected_coverage {
            src.randomness = (src.randomness + 0.05)
                .min(self.randomness_bounds.1);
        } else if stats.unique_nodes_visited > stats.expected_coverage * 2 {
            src.randomness = (src.randomness - 0.05)
                .max(self.randomness_bounds.0);
        }

        // Adjust clustering based on consolidation success
        if stats.consolidation_rate < 0.5 {
            src.clustering = (src.clustering + 0.05)
                .min(self.clustering_bounds.1);
        } else if stats.consolidation_rate > 0.9 {
            src.clustering = (src.clustering - 0.05)
                .max(self.clustering_bounds.0);
        }
    }
}
```

---

## 6. Sleep Cycle State Machine

### 6.1 Sleep Cycle Manager

```rust
/// REQ-DREAM-017: Dream cycle duration management
#[derive(Debug, Clone)]
pub struct DreamCycleManager {
    /// Full cycle duration (NREM + REM + transitions)
    pub cycle_duration: Duration,

    /// Simulated biological cycle ratio
    pub biological_ratio: Duration,

    /// Maximum continuous dream time
    pub max_dream_time: Duration,

    /// Minimum inter-cycle gap
    pub min_cycle_gap: Duration,

    /// Last cycle completion time
    last_completed: Option<Instant>,

    /// Cycle counter
    cycle_count: u64,
}

impl Default for DreamCycleManager {
    fn default() -> Self {
        Self {
            cycle_duration: Duration::from_secs(300),      // 5 minutes
            biological_ratio: Duration::from_secs(5400),   // 90 minutes
            max_dream_time: Duration::from_secs(1800),     // 30 minutes
            min_cycle_gap: Duration::from_secs(600),       // 10 minutes
            last_completed: None,
            cycle_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum DreamCyclePlan {
    Ready {
        nrem_duration: Duration,
        rem_duration: Duration,
        transition_time: Duration,
        estimated_completion: Instant,
    },
    Delayed {
        wait_until: Instant,
    },
    Blocked {
        reason: BlockReason,
    },
}

impl DreamCycleManager {
    /// Plan next dream cycle
    pub fn plan_cycle(&self, context: &SystemContext) -> DreamCyclePlan {
        // Check minimum gap
        if let Some(last) = self.last_completed {
            let elapsed = Instant::now().duration_since(last);
            if elapsed < self.min_cycle_gap {
                return DreamCyclePlan::Delayed {
                    wait_until: last + self.min_cycle_gap,
                };
            }
        }

        // Check max continuous time
        if let Some(dream_start) = context.current_dream_start() {
            let elapsed = Instant::now().duration_since(dream_start);
            if elapsed > self.max_dream_time {
                return DreamCyclePlan::Blocked {
                    reason: BlockReason::MaxDreamTimeExceeded,
                };
            }
        }

        // Calculate phase durations
        let nrem_duration = Duration::from_secs(180);  // 3 minutes
        let rem_duration = Duration::from_secs(120);   // 2 minutes
        let transition = Duration::from_millis(500);

        DreamCyclePlan::Ready {
            nrem_duration,
            rem_duration,
            transition_time: transition,
            estimated_completion: Instant::now() + self.cycle_duration,
        }
    }

    pub fn record_completion(&mut self) {
        self.last_completed = Some(Instant::now());
        self.cycle_count += 1;
    }

    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }
}

/// Full dream cycle orchestration
impl DreamLayer {
    /// Execute complete dream cycle
    pub async fn execute_cycle(&mut self) -> Result<DreamCycleResult, DreamError> {
        let start = Instant::now();
        let (abort_tx, abort_rx) = watch::channel(false);

        // Set up abort controller
        self.abort_controller.set_sender(abort_tx);

        let mut metrics = DreamMetrics::default();

        // Transition to NREM
        {
            let mut state = self.state.write().await;
            *state = DreamState::EnteringNREM;
        }

        // Apply NREM neuromodulation
        {
            let mut neuromod = self.neuromod.write().await;
            neuromod.dopamine = 0.3;
            neuromod.serotonin = 0.4;
            neuromod.acetylcholine = 0.2;
        }

        // Execute NREM phase
        let nrem_result = {
            let mut nrem = self.nrem_phase.write().await;
            let mut graph = self.graph.write().await;
            let mut src = self.src_algorithm.write().await;
            nrem.execute(&mut graph, &mut src, &abort_rx, &mut metrics.nrem).await?
        };

        if !nrem_result.completed {
            return Ok(DreamCycleResult::aborted(start.elapsed(), WakeReason::ManualAbort));
        }

        // Transition to REM
        {
            let mut state = self.state.write().await;
            *state = DreamState::Transitioning;
        }

        tokio::time::sleep(Duration::from_millis(500)).await;

        // Apply REM neuromodulation
        {
            let mut neuromod = self.neuromod.write().await;
            neuromod.dopamine = 0.7;
            neuromod.serotonin = 0.8;
            neuromod.acetylcholine = 0.6;
        }

        // Execute REM phase
        let rem_result = {
            let mut rem = self.rem_phase.write().await;
            let mut graph = self.graph.write().await;
            rem.execute(&mut graph, &abort_rx, &mut metrics.rem).await?
        };

        // Complete cycle
        metrics.cycle.total_duration_ms = start.elapsed().as_millis() as u64;

        // Assess quality
        let quality = self.quality_assessor.assess(&metrics);

        // Store metrics
        {
            let mut stored_metrics = self.metrics.write().await;
            *stored_metrics = metrics.clone();
        }

        Ok(DreamCycleResult {
            completed: rem_result.phase_completed,
            metrics,
            quality,
            duration: start.elapsed(),
        })
    }
}
```

---

## 7. Memory Consolidation Scoring

### 7.1 Consolidation Score Calculator

```rust
/// Memory consolidation scoring based on multiple factors
#[derive(Debug, Clone)]
pub struct ConsolidationScorer {
    /// Weight for recency factor
    pub recency_weight: f32,

    /// Weight for importance factor
    pub importance_weight: f32,

    /// Weight for connectivity factor
    pub connectivity_weight: f32,

    /// Weight for coherence factor
    pub coherence_weight: f32,
}

impl Default for ConsolidationScorer {
    fn default() -> Self {
        Self {
            recency_weight: 0.3,
            importance_weight: 0.3,
            connectivity_weight: 0.2,
            coherence_weight: 0.2,
        }
    }
}

impl ConsolidationScorer {
    /// Compute consolidation priority for a memory node
    pub fn score(&self, node: &MemoryNode, graph: &KnowledgeGraph) -> f32 {
        let recency = self.recency_score(node);
        let importance = node.importance;
        let connectivity = self.connectivity_score(node, graph);
        let coherence = self.coherence_score(node, graph);

        self.recency_weight * recency
            + self.importance_weight * importance
            + self.connectivity_weight * connectivity
            + self.coherence_weight * coherence
    }

    fn recency_score(&self, node: &MemoryNode) -> f32 {
        let age_hours = node.age().as_secs_f32() / 3600.0;
        // Exponential decay: more recent = higher score
        (-age_hours / 24.0).exp()
    }

    fn connectivity_score(&self, node: &MemoryNode, graph: &KnowledgeGraph) -> f32 {
        let edges = graph.get_edge_count(node.id);
        // Log scale to prevent highly connected nodes from dominating
        (edges as f32 + 1.0).ln() / 10.0
    }

    fn coherence_score(&self, node: &MemoryNode, graph: &KnowledgeGraph) -> f32 {
        // Average edge confidence for this node
        let edges = graph.get_edges_for_node(node.id);
        if edges.is_empty() {
            return 0.5;
        }

        edges.iter().map(|e| e.confidence).sum::<f32>() / edges.len() as f32
    }
}
```

---

## 8. Abort Controller

```rust
/// REQ-DREAM-003: Abort on query mechanism
#[derive(Debug)]
pub struct AbortController {
    /// Abort signal sender
    abort_tx: Option<watch::Sender<bool>>,

    /// Maximum time allowed to complete abort
    pub wake_latency_budget: Duration,

    /// State checkpoint for quick restore
    checkpoint: Option<DreamCheckpoint>,
}

impl Default for AbortController {
    fn default() -> Self {
        Self {
            abort_tx: None,
            wake_latency_budget: Duration::from_millis(100),
            checkpoint: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AbortResult {
    pub latency: Duration,
    pub checkpoint_saved: bool,
    pub reason: WakeReason,
}

impl AbortController {
    pub fn set_sender(&mut self, tx: watch::Sender<bool>) {
        self.abort_tx = Some(tx);
    }

    /// REQ-DREAM-003: Signal dream abort - MUST complete within wake_latency_budget
    pub async fn abort(&mut self, reason: WakeReason) -> Result<AbortResult, DreamError> {
        let start = Instant::now();

        // Send abort signal
        if let Some(ref tx) = self.abort_tx {
            tx.send(true).map_err(|_| DreamError::AbortChannelClosed)?;
        }

        // Wait for abort to propagate (with timeout)
        tokio::time::sleep(Duration::from_millis(10)).await;

        let latency = start.elapsed();
        if latency > self.wake_latency_budget {
            log::warn!("Wake latency exceeded budget: {:?}", latency);
        }

        Ok(AbortResult {
            latency,
            checkpoint_saved: self.checkpoint.is_some(),
            reason,
        })
    }

    pub fn save_checkpoint(&mut self, checkpoint: DreamCheckpoint) {
        self.checkpoint = Some(checkpoint);
    }

    pub fn clear_checkpoint(&mut self) {
        self.checkpoint = None;
    }
}
```

---

## 9. Resource Management

```rust
/// REQ-DREAM-016: GPU usage constraint enforcement
#[derive(Debug, Clone)]
pub struct DreamResourceManager {
    /// Maximum GPU utilization during dream
    pub max_gpu_usage: f32,

    /// Check interval
    pub check_interval: Duration,

    /// Throttle factor when exceeding limit
    pub throttle_factor: f32,

    /// Current throttle state
    throttle_active: bool,
}

impl Default for DreamResourceManager {
    fn default() -> Self {
        Self {
            max_gpu_usage: 0.30,  // 30%
            check_interval: Duration::from_millis(100),
            throttle_factor: 0.5,
            throttle_active: false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ResourceAction {
    Continue,
    Warning(f32),
    Throttle(f32),
    Pause,
    Abort,
}

impl DreamResourceManager {
    /// REQ-DREAM-016: Check and enforce GPU limits
    pub async fn enforce_limits(&mut self, gpu_monitor: &GpuMonitor) -> ResourceAction {
        let gpu_usage = gpu_monitor.current_utilization();

        if gpu_usage > self.max_gpu_usage {
            self.throttle_active = true;
            log::warn!(
                "Dream GPU usage {} exceeds limit {}, throttling",
                gpu_usage,
                self.max_gpu_usage
            );
            ResourceAction::Throttle(self.throttle_factor)
        } else if gpu_usage > self.max_gpu_usage * 0.8 {
            ResourceAction::Warning(gpu_usage)
        } else {
            if self.throttle_active {
                self.throttle_active = false;
                log::info!("GPU usage returned to normal, disabling throttle");
            }
            ResourceAction::Continue
        }
    }

    /// Check if resources are critical (external workload)
    pub async fn check_critical(&self, gpu_monitor: &GpuMonitor) -> bool {
        let gpu_usage = gpu_monitor.current_utilization();
        let gpu_memory = gpu_monitor.memory_used_percentage();

        // Pause if external workload needs GPU
        gpu_usage > 0.7 || gpu_memory > 0.85
    }
}
```

---

## 10. Metrics and Quality Assessment

### 10.1 Dream Metrics

```rust
/// REQ-DREAM-021: Comprehensive dream metrics
#[derive(Debug, Clone, Default)]
pub struct DreamMetrics {
    pub nrem: NREMMetrics,
    pub rem: REMMetrics,
    pub cycle: CycleMetrics,
    pub resources: ResourceMetrics,
}

#[derive(Debug, Clone, Default)]
pub struct NREMMetrics {
    pub memories_replayed: u32,
    pub edges_strengthened: u32,
    pub edges_weakened: u32,
    pub edges_pruned: u32,
    pub clusters_consolidated: u32,
    pub nodes_merged: u32,
    pub redundancies_eliminated: u32,
    pub schemas_extracted: u32,
    pub shortcuts_created: u32,
    pub compression_ratio: f32,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct REMMetrics {
    pub synthetic_queries: u32,
    pub blind_spots_found: u32,
    pub new_edges_created: u32,
    pub average_semantic_leap: f32,
    pub exploration_coverage: f32,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct CycleMetrics {
    pub total_duration_ms: u64,
    pub wake_events: u32,
    pub abort_reason: Option<String>,
    pub information_loss_estimate: f32,
    pub quality_improvement: f32,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceMetrics {
    pub peak_gpu_usage: f32,
    pub average_gpu_usage: f32,
    pub peak_memory_mb: u32,
    pub throttle_events: u32,
}
```

### 10.2 Quality Assessment

```rust
/// REQ-DREAM-022: Dream quality assessment
#[derive(Debug, Clone)]
pub struct DreamQualityAssessor {
    /// Minimum acceptable compression ratio
    pub min_compression: f32,

    /// Minimum blind spots per cycle
    pub min_blind_spots: u32,

    /// Maximum acceptable information loss
    pub max_information_loss: f32,

    /// Quality score thresholds
    pub thresholds: QualityThresholds,
}

impl Default for DreamQualityAssessor {
    fn default() -> Self {
        Self {
            min_compression: 5.0,
            min_blind_spots: 3,
            max_information_loss: 0.15,
            thresholds: QualityThresholds::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub excellent: f32,
    pub good: f32,
    pub fair: f32,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            excellent: 0.9,
            good: 0.7,
            fair: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DreamQualityReport {
    pub overall_score: f32,
    pub grade: QualityGrade,
    pub compression_achieved: bool,
    pub exploration_adequate: bool,
    pub information_preserved: bool,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityGrade {
    Excellent,
    Good,
    Fair,
    Poor,
}

impl QualityGrade {
    pub fn from_score(score: f32) -> Self {
        if score >= 0.9 { QualityGrade::Excellent }
        else if score >= 0.7 { QualityGrade::Good }
        else if score >= 0.5 { QualityGrade::Fair }
        else { QualityGrade::Poor }
    }
}

impl DreamQualityAssessor {
    /// REQ-DREAM-022: Assess dream cycle quality
    pub fn assess(&self, metrics: &DreamMetrics) -> DreamQualityReport {
        let mut score = 0.0;
        let mut recommendations = Vec::new();

        // Compression score (30%)
        let compression_score = (metrics.nrem.compression_ratio / 10.0).min(1.0);
        score += compression_score * 0.3;
        if compression_score < 0.5 {
            recommendations.push("Increase consolidation threshold".to_string());
        }

        // Exploration score (30%)
        let exploration_score = (metrics.rem.blind_spots_found as f32 / 5.0).min(1.0);
        score += exploration_score * 0.3;
        if exploration_score < 0.5 {
            recommendations.push("Increase exploration temperature".to_string());
        }

        // Preservation score (40%)
        let preservation_score = 1.0 - metrics.cycle.information_loss_estimate;
        score += preservation_score * 0.4;
        if preservation_score < 0.85 {
            recommendations.push("Reduce consolidation aggressiveness".to_string());
        }

        DreamQualityReport {
            overall_score: score,
            grade: QualityGrade::from_score(score),
            compression_achieved: metrics.nrem.compression_ratio >= self.min_compression,
            exploration_adequate: metrics.rem.blind_spots_found >= self.min_blind_spots,
            information_preserved: metrics.cycle.information_loss_estimate <= self.max_information_loss,
            recommendations,
        }
    }
}
```

---

## 11. Performance Targets

| Metric | Target | Enforcement |
|--------|--------|-------------|
| NREM Duration | 3 min | tokio::time::timeout |
| REM Duration | 2 min | tokio::time::timeout |
| Wake Latency | <100ms P95 | Abort controller |
| GPU Usage | <30% | Resource manager |
| Compression Ratio | 10:1 | Quality assessor |
| Blind Spots | >5/cycle | Quality assessor |
| Information Loss | <15% | Quality assessor |
| Shortcut Confidence | >=0.7 | REQ-DREAM-037 gate |

---

## 12. Configuration

```toml
[dream]
enabled = true

[dream.trigger]
activity_threshold = 0.15
idle_duration_minutes = 10
cooldown_minutes = 30
memory_pressure_threshold = 0.8

[dream.nrem]
duration_minutes = 3
replay_recency_bias = 0.8
learning_rate = 0.01
batch_size = 64
consolidation_threshold = 0.7
redundancy_threshold = 0.95

[dream.rem]
duration_minutes = 2
synthetic_query_count = 100
min_semantic_leap = 0.7
exploration_temperature = 2.0
new_edge_weight = 0.3
new_edge_confidence = 0.5

[dream.src]
sparsity = 0.1
randomness = 0.5
clustering = 0.7
decay_rate = 0.9

[dream.shortcut]
min_hops = 3
confidence_threshold = 0.7
traversal_threshold = 5

[dream.resources]
max_gpu_usage = 0.30
check_interval_ms = 100
wake_latency_ms = 100

[dream.quality]
target_compression_ratio = 10.0
min_blind_spots = 5
max_information_loss = 0.15
```

---

## 13. Dependencies

```toml
[dependencies]
context-graph-core = { path = "../context-graph-core" }
context-graph-nervous = { path = "../context-graph-nervous" }
context-graph-graph = { path = "../context-graph-graph" }
tokio = { version = "1.35", features = ["full", "time"] }
async-trait = "0.1"
uuid = { version = "1.6", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1.0", features = ["derive"] }
thiserror = "1.0"
log = "0.4"
rand = "0.8"
dashmap = "5.5"
parking_lot = "0.12"
```

---

## 14. Acceptance Criteria

### 14.1 Core Requirements
- [ ] DreamLayer orchestrates NREM (3min) + REM (2min) phases
- [ ] SRC algorithm implements sparse, random, clustered activation
- [ ] Hebbian updates: delta_w = eta * pre * post
- [ ] Wake latency <100ms on user query
- [ ] GPU usage <30% enforced

### 14.2 Marblestone Additions (REQ-DREAM-036/037)
- [ ] Amortized shortcuts created for 3+ hop paths
- [ ] Path confidence >= 0.7 quality gate
- [ ] Shortcuts marked with `is_amortized_shortcut = true`
- [ ] Edge weight = product of path weights

### 14.3 Performance Targets
- [ ] 10:1 compression ratio achieved
- [ ] >5 blind spots discovered per REM cycle
- [ ] <15% information loss
- [ ] P95 wake latency <100ms
- [ ] P99 wake latency <150ms

---

*Document Version: 1.0.0 | Generated: 2025-12-31 | Agent: Architecture Agent*
