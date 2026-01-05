# Module 9: Dream Layer - Functional Specification

**Version**: 1.0.0
**Status**: Draft
**Phase**: 8
**Duration**: 3 weeks
**Dependencies**: Module 8 (GPU Direct Storage)
**Last Updated**: 2025-12-31

---

## 1. Executive Summary

The Dream Layer implements offline memory consolidation inspired by sleep neuroscience, alternating between NREM (replay/compression) and REM (creative exploration) phases. It uses the SRC (Sparse, Random, Clustered) algorithm to achieve efficient memory consolidation while discovering novel semantic connections. This module transforms the knowledge graph from a passive storage system into an actively self-organizing memory architecture that improves retrieval quality over time.

### 1.1 Core Objectives

- Implement biologically-inspired memory consolidation during low-activity periods
- Achieve 10:1 compression ratio without significant information loss
- Discover novel semantic connections through random hyperbolic space exploration
- Maintain system responsiveness with <100ms wake latency on user query
- Operate within GPU resource budget (<30% utilization during dream cycles)

### 1.2 Key Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Compression Ratio | 10:1 | Nodes before/after NREM cycle |
| Information Loss | <15% | Key fact retention benchmark |
| Novel Connections | +15% recall improvement | Before/after retrieval accuracy |
| Wake Latency | <100ms | Time from query arrival to dream abort |
| GPU Usage | <30% | nvidia-smi during dream phase |
| Blind Spots Discovered | >5 per REM cycle | High semantic distance connections found |

---

## 2. Functional Requirements

### 2.1 Dream Layer Core Structure

#### REQ-DREAM-001: DreamLayer Struct Definition

**Priority**: Critical
**Description**: The system SHALL implement a DreamLayer struct that orchestrates the complete dream cycle.

```rust
pub struct DreamLayer {
    /// NREM phase handler for memory replay and consolidation
    pub nrem_phase: NREMPhase,

    /// REM phase handler for creative exploration
    pub rem_phase: REMPhase,

    /// Simulated cycle duration (biological: 90 minutes)
    pub cycle_duration: Duration,  // Default: 90 minutes simulated

    /// Target compression ratio for NREM consolidation
    pub compression_ratio: f32,    // Default: 10.0 (10:1)

    /// Activity monitor for trigger detection
    activity_monitor: ActivityMonitor,

    /// Current dream state
    state: DreamState,

    /// Abort signal receiver
    abort_rx: watch::Receiver<bool>,

    /// Metrics collector
    metrics: DreamMetrics,
}

pub enum DreamState {
    Awake,
    EnteringNREM,
    NREM { started_at: Instant, phase: NREMSubPhase },
    Transitioning,
    REM { started_at: Instant, queries_generated: u32 },
    Waking { reason: WakeReason },
}

pub enum WakeReason {
    UserQuery,
    ResourcePressure,
    CycleComplete,
    ManualAbort,
    Error(String),
}
```

**Acceptance Criteria**:
- [ ] DreamLayer struct compiles with all fields
- [ ] State machine transitions are valid and complete
- [ ] Default values match constitution.yaml specifications
- [ ] Thread-safe abort mechanism implemented

---

#### REQ-DREAM-002: Dream Trigger Conditions

**Priority**: Critical
**Description**: The system SHALL automatically trigger dream cycles when activity falls below threshold.

```rust
pub struct DreamTrigger {
    /// Activity level below which dream may begin
    pub activity_threshold: f32,       // Default: 0.15

    /// Duration of low activity before triggering
    pub idle_duration: Duration,       // Default: 10 minutes

    /// Minimum time between dream cycles
    pub cooldown_period: Duration,     // Default: 30 minutes

    /// Memory accumulation threshold
    pub memory_pressure_threshold: f32, // Default: 0.8
}

impl DreamTrigger {
    /// Check if dream should be triggered
    pub fn should_trigger(&self, context: &SystemContext) -> TriggerDecision {
        let activity = context.get_activity_level();
        let idle_time = context.get_idle_duration();
        let memory_pressure = context.get_memory_pressure();

        if activity < self.activity_threshold && idle_time >= self.idle_duration {
            TriggerDecision::Trigger(TriggerReason::IdleTimeout)
        } else if memory_pressure > self.memory_pressure_threshold {
            TriggerDecision::Trigger(TriggerReason::MemoryPressure)
        } else {
            TriggerDecision::Wait
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Activity threshold correctly detects low activity (0.15 default)
- [ ] Idle duration timer accurately tracks 10-minute window
- [ ] Cooldown prevents excessive dream cycles
- [ ] Memory pressure can force early dream trigger
- [ ] All trigger decisions are logged for debugging

---

#### REQ-DREAM-003: Abort on Query Mechanism

**Priority**: Critical
**Description**: The system SHALL immediately abort dream cycles when user queries arrive.

```rust
pub struct AbortController {
    /// Abort signal sender
    abort_tx: watch::Sender<bool>,

    /// Maximum time allowed to complete abort
    pub wake_latency_budget: Duration,  // Default: 100ms

    /// State snapshot for quick restore
    checkpoint: Option<DreamCheckpoint>,
}

impl AbortController {
    /// Signal dream abort - MUST complete within wake_latency_budget
    pub async fn abort(&self, reason: WakeReason) -> Result<AbortResult, DreamError> {
        let start = Instant::now();

        // Send abort signal
        self.abort_tx.send(true)?;

        // Wait for confirmation with timeout
        let result = timeout(self.wake_latency_budget, self.wait_for_abort()).await?;

        let latency = start.elapsed();
        if latency > self.wake_latency_budget {
            warn!("Wake latency exceeded budget: {:?}", latency);
        }

        Ok(AbortResult {
            latency,
            checkpoint_saved: self.checkpoint.is_some(),
            reason,
        })
    }
}
```

**Acceptance Criteria**:
- [ ] Abort signal propagates within 10ms
- [ ] Complete wake occurs within 100ms
- [ ] In-progress operations gracefully terminate
- [ ] State checkpoint saved for resumption
- [ ] No data corruption on abort
- [ ] Latency metrics recorded for monitoring

---

### 2.2 NREM Phase Requirements

#### REQ-DREAM-004: NREM Phase Structure

**Priority**: Critical
**Description**: The system SHALL implement NREM phase for hippocampus-guided memory replay.

```rust
pub struct NREMPhase {
    /// Phase duration
    pub duration: Duration,           // Default: 3 minutes

    /// Recency bias for memory selection
    pub replay_recency_bias: f32,     // Default: 0.8

    /// Hebbian learning rate
    pub learning_rate: f32,           // Default: 0.01

    /// Consolidation batch size
    pub batch_size: usize,            // Default: 64

    /// Schema extraction settings
    schema_extractor: SchemaExtractor,

    /// Redundancy eliminator
    redundancy_detector: RedundancyDetector,
}

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
```

**Acceptance Criteria**:
- [ ] NREM phase runs for 3 minutes (configurable)
- [ ] Recency bias correctly weights recent memories at 0.8
- [ ] All sub-phases execute in sequence
- [ ] Progress tracking for each sub-phase
- [ ] Graceful handling of phase interruption

---

#### REQ-DREAM-005: Memory Replay with Recency Bias

**Priority**: High
**Description**: The system SHALL replay recent memories with configurable recency bias.

```rust
pub struct MemoryReplayEngine {
    /// Recency weight (higher = more recent bias)
    recency_bias: f32,  // Default: 0.8

    /// Maximum memories to replay per cycle
    max_replay_count: usize,  // Default: 1000

    /// Importance floor for replay eligibility
    importance_floor: f32,  // Default: 0.3
}

impl MemoryReplayEngine {
    /// Select memories for replay using recency-weighted sampling
    pub fn select_memories(&self, graph: &KnowledgeGraph) -> Vec<MemoryNode> {
        let candidates = graph.get_nodes_since(Duration::from_hours(24));

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
    pub async fn replay_memory(&self, node: &MemoryNode, context: &mut ReplayContext)
        -> Result<ReplayResult, DreamError>;
}
```

**Acceptance Criteria**:
- [ ] Recent memories (last 24h) receive 0.8x weight advantage
- [ ] Importance filter excludes low-value memories
- [ ] Weighted sampling implemented correctly
- [ ] Replay results in connection strengthening
- [ ] Memory access patterns recorded

---

#### REQ-DREAM-006: Hebbian Weight Update

**Priority**: Critical
**Description**: The system SHALL apply Hebbian learning to strengthen co-activated connections.

```rust
pub struct HebbianUpdater {
    /// Learning rate for weight updates
    pub learning_rate: f32,  // Default: 0.01 (eta)

    /// Weight decay factor
    pub decay_rate: f32,     // Default: 0.001

    /// Maximum weight value
    pub weight_cap: f32,     // Default: 1.0

    /// Minimum weight before pruning
    pub weight_floor: f32,   // Default: 0.05
}

impl HebbianUpdater {
    /// Apply Hebbian update: delta_w = eta * pre * post
    pub fn update_weight(&self, edge: &mut GraphEdge, pre_activation: f32, post_activation: f32) {
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
    pub async fn batch_update(&self, replay_sequence: &[MemoryNode], graph: &mut KnowledgeGraph)
        -> Result<HebbianUpdateStats, DreamError>;
}

pub struct HebbianUpdateStats {
    pub edges_strengthened: u32,
    pub edges_weakened: u32,
    pub edges_pruned: u32,
    pub average_delta: f32,
}
```

**Acceptance Criteria**:
- [ ] Weight update formula: delta_w = eta x pre x post
- [ ] Learning rate default of 0.01 applied
- [ ] Weight decay prevents runaway strengthening
- [ ] Weight bounds enforced (0.05 to 1.0)
- [ ] Weak edges marked for pruning
- [ ] Update statistics recorded

---

#### REQ-DREAM-007: Tight Coupling Consolidation

**Priority**: High
**Description**: The system SHALL consolidate tightly coupled graph regions.

```rust
pub struct CouplingConsolidator {
    /// Minimum coupling strength for consolidation
    pub coupling_threshold: f32,  // Default: 0.7

    /// Maximum cluster size for merge
    pub max_cluster_size: usize,  // Default: 10

    /// Similarity threshold for merging
    pub merge_similarity: f32,    // Default: 0.9
}

impl CouplingConsolidator {
    /// Identify tightly coupled clusters
    pub fn find_coupled_clusters(&self, graph: &KnowledgeGraph) -> Vec<Cluster> {
        // Use community detection (Louvain or similar)
        let communities = graph.detect_communities(self.coupling_threshold);

        // Filter by size
        communities
            .into_iter()
            .filter(|c| c.size() <= self.max_cluster_size)
            .filter(|c| c.internal_coupling() >= self.coupling_threshold)
            .collect()
    }

    /// Consolidate cluster into representative node
    pub async fn consolidate_cluster(&self, cluster: &Cluster, graph: &mut KnowledgeGraph)
        -> Result<ConsolidationResult, DreamError> {

        // Check priors compatibility before merge
        let priors_check = cluster.check_priors_compatibility()?;
        if !priors_check.compatible && !self.force_merge {
            return Err(DreamError::IncompatiblePriors(priors_check));
        }

        // Create merged node with combined embedding
        let merged_embedding = self.fuse_embeddings(&cluster.nodes);
        let merged_content = self.summarize_contents(&cluster.nodes);

        let merged_node = MemoryNode::new(merged_content, merged_embedding);

        // Preserve edge connections
        self.transfer_edges(cluster, &merged_node, graph)?;

        // Soft-delete originals
        for node in &cluster.nodes {
            graph.soft_delete(node.id)?;
        }

        Ok(ConsolidationResult {
            merged_node_id: merged_node.id,
            nodes_consolidated: cluster.nodes.len(),
            compression_achieved: cluster.nodes.len() as f32 / 1.0,
        })
    }
}
```

**Acceptance Criteria**:
- [ ] Community detection identifies coupled clusters
- [ ] Coupling threshold of 0.7 enforced
- [ ] Cluster size limit of 10 nodes
- [ ] Priors compatibility checked before merge
- [ ] Edge connections preserved after merge
- [ ] Original nodes soft-deleted with recovery option

---

#### REQ-DREAM-008: Redundancy Elimination

**Priority**: High
**Description**: The system SHALL eliminate redundant memories during NREM.

```rust
pub struct RedundancyDetector {
    /// Minimum similarity for redundancy detection
    pub similarity_threshold: f32,  // Default: 0.95

    /// Maximum redundancy check candidates
    pub max_candidates: usize,      // Default: 10000

    /// Strategy for selecting survivor
    pub survivor_strategy: SurvivorStrategy,
}

pub enum SurvivorStrategy {
    /// Keep most recently accessed
    MostRecent,
    /// Keep highest importance
    HighestImportance,
    /// Keep most connected
    MostConnected,
    /// Keep earliest created (original)
    Oldest,
    /// Merge into single node
    Merge,
}

impl RedundancyDetector {
    /// Find redundant node pairs
    pub async fn find_redundant_pairs(&self, graph: &KnowledgeGraph)
        -> Vec<RedundantPair> {

        // Use LSH for efficient similarity detection
        let candidates = graph.lsh_candidates(self.max_candidates);

        candidates
            .par_iter()  // Parallel processing
            .filter_map(|(a, b)| {
                let sim = cosine_similarity(&a.embedding, &b.embedding);
                if sim >= self.similarity_threshold {
                    Some(RedundantPair { node_a: a.id, node_b: b.id, similarity: sim })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Eliminate redundancy by keeping survivor
    pub async fn eliminate_redundancy(&self, pairs: Vec<RedundantPair>, graph: &mut KnowledgeGraph)
        -> Result<RedundancyStats, DreamError>;
}

pub struct RedundancyStats {
    pub pairs_found: u32,
    pub nodes_eliminated: u32,
    pub space_recovered_bytes: u64,
}
```

**Acceptance Criteria**:
- [ ] 0.95 similarity threshold detects near-duplicates
- [ ] LSH enables efficient candidate generation
- [ ] Multiple survivor strategies supported
- [ ] Edge connections transferred to survivor
- [ ] Eliminated nodes soft-deleted
- [ ] Recovery possible within 30-day window

---

#### REQ-DREAM-009: Schema Extraction

**Priority**: Medium
**Description**: The system SHALL extract abstract schemas from concrete memory clusters.

```rust
pub struct SchemaExtractor {
    /// Minimum cluster size for schema extraction
    pub min_cluster_size: usize,  // Default: 5

    /// Schema abstraction level
    pub abstraction_level: f32,   // Default: 0.7 (0=concrete, 1=abstract)

    /// Maximum schemas per cycle
    pub max_schemas_per_cycle: usize,  // Default: 20
}

pub struct ExtractedSchema {
    pub id: Uuid,
    pub name: String,
    pub template: SchemaTemplate,
    pub source_nodes: Vec<Uuid>,
    pub confidence: f32,
    pub abstraction_level: f32,
}

impl SchemaExtractor {
    /// Extract schemas from memory clusters
    pub async fn extract_schemas(&self, clusters: Vec<Cluster>) -> Vec<ExtractedSchema> {
        clusters
            .into_iter()
            .filter(|c| c.size() >= self.min_cluster_size)
            .take(self.max_schemas_per_cycle)
            .filter_map(|c| self.extract_single_schema(&c))
            .collect()
    }

    /// Extract template from single cluster
    fn extract_single_schema(&self, cluster: &Cluster) -> Option<ExtractedSchema> {
        // Find common structural patterns
        let common_edges = cluster.find_common_edge_patterns();

        // Extract variable slots
        let variable_slots = cluster.find_variable_positions();

        // Generate schema if pattern is strong enough
        if common_edges.confidence >= self.abstraction_level {
            Some(ExtractedSchema {
                id: Uuid::new_v4(),
                name: self.generate_schema_name(&common_edges),
                template: SchemaTemplate::from_patterns(common_edges, variable_slots),
                source_nodes: cluster.node_ids(),
                confidence: common_edges.confidence,
                abstraction_level: self.abstraction_level,
            })
        } else {
            None
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Minimum cluster size of 5 nodes enforced
- [ ] Abstraction level configurable (default 0.7)
- [ ] Common patterns extracted from clusters
- [ ] Variable slots identified in templates
- [ ] Schema confidence scores computed
- [ ] Maximum 20 schemas per cycle limit

---

### 2.3 REM Phase Requirements

#### REQ-DREAM-010: REM Phase Structure

**Priority**: Critical
**Description**: The system SHALL implement REM phase for creative exploration.

```rust
pub struct REMPhase {
    /// Phase duration
    pub duration: Duration,                // Default: 2 minutes

    /// Number of synthetic queries to generate
    pub synthetic_query_count: u32,        // Default: 100

    /// Minimum semantic leap for connection
    pub min_semantic_leap: f32,            // Default: 0.7

    /// Exploration temperature
    pub exploration_temperature: f32,      // Default: 2.0

    /// New edge initial weight
    pub new_edge_weight: f32,              // Default: 0.3

    /// New edge initial confidence
    pub new_edge_confidence: f32,          // Default: 0.5
}

impl REMPhase {
    /// Execute REM phase exploration
    pub async fn execute(&mut self, graph: &mut KnowledgeGraph, abort_rx: &watch::Receiver<bool>)
        -> Result<REMResult, DreamError> {

        let mut queries_generated = 0;
        let mut connections_discovered = 0;
        let mut blind_spots_found = 0;

        let deadline = Instant::now() + self.duration;

        while Instant::now() < deadline && queries_generated < self.synthetic_query_count {
            // Check for abort
            if *abort_rx.borrow() {
                return Ok(REMResult::aborted(queries_generated, connections_discovered));
            }

            // Generate synthetic query via random walk
            let query = self.generate_synthetic_query(graph)?;
            queries_generated += 1;

            // Explore for blind spots
            if let Some(discovery) = self.explore_blind_spot(&query, graph)? {
                connections_discovered += 1;
                if discovery.is_blind_spot {
                    blind_spots_found += 1;
                }
            }
        }

        Ok(REMResult {
            queries_generated,
            connections_discovered,
            blind_spots_found,
            phase_completed: Instant::now() >= deadline,
        })
    }
}
```

**Acceptance Criteria**:
- [ ] REM phase runs for 2 minutes (configurable)
- [ ] 100 synthetic queries generated by default
- [ ] Abort signal checked between queries
- [ ] Progress tracked and logged
- [ ] Statistics collected for monitoring

---

#### REQ-DREAM-011: Synthetic Query Generation

**Priority**: High
**Description**: The system SHALL generate synthetic queries via random walk in hyperbolic space.

```rust
pub struct SyntheticQueryGenerator {
    /// Random walk step size in hyperbolic space
    pub step_size: f32,           // Default: 0.3

    /// Walk length (number of steps)
    pub walk_length: u32,         // Default: 5

    /// Exploration bias (higher = more diverse)
    pub exploration_bias: f32,    // Default: 0.7

    /// Hyperbolic curvature
    pub curvature: f32,           // Default: -1.0 (Poincare ball)
}

impl SyntheticQueryGenerator {
    /// Generate synthetic query via random walk
    pub fn generate(&self, graph: &KnowledgeGraph, rng: &mut impl Rng) -> SyntheticQuery {
        // Select random starting node weighted by importance
        let start = graph.sample_node_weighted_by_importance(rng);

        // Perform random walk in hyperbolic space
        let mut position = start.hyperbolic_position;
        let path = Vec::with_capacity(self.walk_length as usize);

        for _ in 0..self.walk_length {
            // Generate random direction in tangent space
            let direction = self.sample_tangent_vector(&position, rng);

            // Apply hyperbolic exponential map
            position = hyperbolic_exp_map(position, direction, self.step_size);

            // Ensure we stay in Poincare ball (||x|| < 1)
            position = project_to_ball(position);

            path.push(position);
        }

        // Convert final position to query embedding
        SyntheticQuery {
            embedding: position.to_embedding(),
            path,
            origin_node: start.id,
        }
    }

    /// Sample tangent vector with exploration bias
    fn sample_tangent_vector(&self, position: &HyperbolicPoint, rng: &mut impl Rng) -> TangentVector {
        // Mix random direction with gradient toward sparse regions
        let random_dir = TangentVector::random_unit(rng);
        let sparse_gradient = self.compute_sparsity_gradient(position);

        // Blend: higher exploration_bias = more random
        random_dir.blend(sparse_gradient, self.exploration_bias)
    }
}
```

**Acceptance Criteria**:
- [ ] Random walk starts from importance-weighted node
- [ ] Walk respects hyperbolic geometry (Poincare ball)
- [ ] Step size of 0.3 in hyperbolic space
- [ ] Walk length of 5 steps by default
- [ ] Exploration bias directs toward sparse regions
- [ ] All positions valid (||x|| < 1)

---

#### REQ-DREAM-012: Blind Spot Discovery

**Priority**: Critical
**Description**: The system SHALL discover blind spots defined as high semantic distance with shared causal paths.

```rust
pub struct BlindSpotDetector {
    /// Minimum semantic distance for blind spot
    pub min_semantic_distance: f32,     // Default: 0.7

    /// Minimum causal path overlap
    pub min_causal_overlap: f32,        // Default: 0.3

    /// Maximum search depth for causal paths
    pub max_causal_depth: u32,          // Default: 4
}

pub struct BlindSpot {
    pub node_a: Uuid,
    pub node_b: Uuid,
    pub semantic_distance: f32,
    pub shared_causal_paths: Vec<CausalPath>,
    pub discovery_confidence: f32,
}

impl BlindSpotDetector {
    /// Detect blind spot between query location and existing nodes
    pub async fn detect(&self, query: &SyntheticQuery, graph: &KnowledgeGraph)
        -> Option<BlindSpot> {

        // Find nearest neighbors at different semantic distances
        let near_nodes = graph.knn(&query.embedding, 10);
        let far_nodes = graph.furthest_nodes(&query.embedding, 10);

        // Check for blind spot pattern: far semantically but causally connected
        for far_node in far_nodes {
            let semantic_dist = cosine_distance(&query.embedding, &far_node.embedding);

            if semantic_dist < self.min_semantic_distance {
                continue;
            }

            // Check for shared causal paths
            for near_node in &near_nodes {
                let shared_paths = graph.find_shared_causal_paths(
                    near_node.id,
                    far_node.id,
                    self.max_causal_depth
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
}
```

**Acceptance Criteria**:
- [ ] Semantic distance threshold of 0.7
- [ ] Causal path overlap threshold of 0.3
- [ ] Shared causal paths identified up to depth 4
- [ ] Blind spots recorded with confidence scores
- [ ] Discovery rate tracked (target: >5 per REM cycle)

---

#### REQ-DREAM-013: Novel Edge Creation

**Priority**: High
**Description**: The system SHALL create new edges for discovered blind spots.

```rust
pub struct EdgeCreator {
    /// Initial weight for new edges
    pub initial_weight: f32,       // Default: 0.3

    /// Initial confidence for new edges
    pub initial_confidence: f32,   // Default: 0.5

    /// Edge type for dream-discovered connections
    pub edge_type: EdgeType,       // Default: EdgeType::DreamDiscovered

    /// Maximum new edges per cycle
    pub max_new_edges: u32,        // Default: 50
}

impl EdgeCreator {
    /// Create edge from blind spot discovery
    pub fn create_from_blind_spot(&self, blind_spot: &BlindSpot, graph: &mut KnowledgeGraph)
        -> Result<GraphEdge, DreamError> {

        let edge = GraphEdge {
            source: blind_spot.node_a,
            target: blind_spot.node_b,
            edge_type: self.edge_type,
            weight: self.initial_weight,
            confidence: self.initial_confidence,
            created_at: Utc::now(),
            metadata: EdgeMetadata {
                origin: EdgeOrigin::DreamDiscovery,
                semantic_distance: blind_spot.semantic_distance,
                causal_evidence: blind_spot.shared_causal_paths.len() as u32,
                discovery_cycle: graph.current_dream_cycle(),
            },
        };

        // Add edge to graph
        graph.add_edge(edge.clone())?;

        // Log for monitoring
        info!("Dream-discovered edge: {} -> {} (weight: {}, confidence: {})",
              blind_spot.node_a, blind_spot.node_b, self.initial_weight, self.initial_confidence);

        Ok(edge)
    }
}

pub enum EdgeOrigin {
    UserCreated,
    SystemInferred,
    DreamDiscovery,
    ImportedData,
}
```

**Acceptance Criteria**:
- [ ] New edges created with weight 0.3
- [ ] New edges created with confidence 0.5
- [ ] Edge type marked as DreamDiscovered
- [ ] Metadata includes discovery context
- [ ] Maximum 50 new edges per cycle
- [ ] Edge creation logged for audit

---

### 2.4 SRC Algorithm Requirements

#### REQ-DREAM-014: SRC Algorithm Implementation

**Priority**: Critical
**Description**: The system SHALL implement the SRC (Sparse, Random, Clustered) algorithm for dream processing.

```rust
pub struct SRCAlgorithm {
    /// Fraction of nodes activated per step (sparsity)
    pub sparsity: f32,           // Default: 0.1 (10% of nodes)

    /// Exploration vs exploitation balance
    pub randomness: f32,         // Default: 0.5

    /// Locality preference for activation
    pub clustering: f32,         // Default: 0.7

    /// Activation decay rate
    pub decay_rate: f32,         // Default: 0.9
}

impl SRCAlgorithm {
    /// Perform one SRC step
    pub fn step(&self, state: &mut SRCState, graph: &KnowledgeGraph) -> SRCStepResult {
        // Decay existing activations
        state.activations.iter_mut().for_each(|(_, v)| *v *= self.decay_rate);

        // Calculate number of nodes to activate
        let k = (graph.node_count() as f32 * self.sparsity) as usize;

        // Select activation candidates
        let candidates = self.select_candidates(state, graph, k);

        // Activate selected nodes
        for node_id in candidates {
            let activation = self.compute_activation(node_id, state, graph);
            state.activations.insert(node_id, activation);
        }

        // Prune low activations
        state.activations.retain(|_, v| *v > 0.01);

        SRCStepResult {
            nodes_activated: state.activations.len(),
            average_activation: state.average_activation(),
            clustering_coefficient: state.clustering_coefficient(graph),
        }
    }

    /// Select candidate nodes for activation
    fn select_candidates(&self, state: &SRCState, graph: &KnowledgeGraph, k: usize)
        -> Vec<Uuid> {

        let mut candidates = Vec::with_capacity(k);
        let mut rng = thread_rng();

        for _ in 0..k {
            if rng.gen::<f32>() < self.randomness {
                // Random selection (exploration)
                candidates.push(graph.random_node(&mut rng).id);
            } else {
                // Clustered selection (exploitation)
                if let Some(active_node) = state.sample_active_node(&mut rng) {
                    // Select neighbor with probability proportional to clustering
                    if rng.gen::<f32>() < self.clustering {
                        if let Some(neighbor) = graph.random_neighbor(active_node, &mut rng) {
                            candidates.push(neighbor);
                            continue;
                        }
                    }
                }
                // Fallback to random
                candidates.push(graph.random_node(&mut rng).id);
            }
        }

        candidates
    }

    /// Create amortized inference shortcuts from frequently traversed paths (Marblestone)
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
    ///
    /// # Arguments
    /// * `replay_paths` - Paths identified during NREM replay
    /// * `min_hops` - Minimum path length for shortcut creation (default: 3)
    /// * `confidence_threshold` - Minimum path confidence (default: 0.7)
    ///
    /// # Returns
    /// List of created shortcut edges
    pub async fn amortized_shortcut_creation(
        &self,
        replay_paths: &[ReplayPath],
        min_hops: usize,
        confidence_threshold: f32,
    ) -> Vec<GraphEdge> {
        let mut shortcuts = Vec::new();

        for path in replay_paths {
            // Skip paths that are too short
            if path.edges.len() < min_hops {
                continue;
            }

            // Check traversal frequency
            if path.traversal_count < 5 {
                continue;
            }

            // Calculate path confidence (product of edge confidences)
            let path_confidence: f32 = path.edges.iter()
                .map(|e| e.confidence)
                .product();

            if path_confidence < confidence_threshold {
                continue;
            }

            // Create shortcut edge
            let shortcut = GraphEdge {
                source: path.source,
                target: path.target,
                edge_type: EdgeType::Causal,
                weight: path_confidence,
                confidence: path_confidence,
                is_amortized_shortcut: true,
                steering_reward: 0.0,
                neurotransmitter_weights: NeurotransmitterWeights::default(),
                created_at: Utc::now(),
            };

            shortcuts.push(shortcut);
        }

        shortcuts
    }
}

/// Path identified during sleep replay for potential shortcut creation
pub struct ReplayPath {
    pub source: Uuid,
    pub target: Uuid,
    pub edges: Vec<GraphEdge>,
    pub traversal_count: u32,
}

pub struct SRCState {
    pub activations: HashMap<Uuid, f32>,
    pub step_count: u32,
    pub history: VecDeque<SRCStepResult>,
}
```

**Acceptance Criteria**:
- [ ] Sparsity controls activation fraction (default 10%)
- [ ] Randomness balances exploration/exploitation (default 0.5)
- [ ] Clustering promotes local activation (default 0.7)
- [ ] Decay prevents runaway activation (default 0.9)
- [ ] Step results tracked for analysis
- [ ] State can be serialized for checkpointing

---

#### REQ-DREAM-015: SRC Parameter Tuning

**Priority**: Medium
**Description**: The system SHALL allow dynamic tuning of SRC parameters.

```rust
pub struct SRCTuner {
    /// Target activation level
    pub target_activation: f32,   // Default: 0.05

    /// Adjustment rate
    pub adjustment_rate: f32,     // Default: 0.1

    /// Bounds for parameters
    pub sparsity_bounds: (f32, f32),    // (0.01, 0.3)
    pub randomness_bounds: (f32, f32),  // (0.1, 0.9)
    pub clustering_bounds: (f32, f32),  // (0.3, 0.95)
}

impl SRCTuner {
    /// Adjust parameters based on observed dynamics
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
        }

        // Adjust clustering based on consolidation success
        if stats.consolidation_rate < 0.5 {
            src.clustering = (src.clustering + 0.05)
                .min(self.clustering_bounds.1);
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Target activation level of 5% maintained
- [ ] Parameters stay within defined bounds
- [ ] Tuning adjusts based on observed metrics
- [ ] Changes are gradual (no sudden jumps)
- [ ] Tuning decisions logged

---

### 2.5 Resource Management Requirements

#### REQ-DREAM-016: GPU Usage Constraint

**Priority**: Critical
**Description**: The system SHALL limit GPU usage to less than 30% during dream cycles.

```rust
pub struct DreamResourceManager {
    /// Maximum GPU utilization during dream
    pub max_gpu_usage: f32,       // Default: 0.30 (30%)

    /// Check interval
    pub check_interval: Duration, // Default: 100ms

    /// Throttle factor when exceeding limit
    pub throttle_factor: f32,     // Default: 0.5

    /// GPU monitoring handle
    gpu_monitor: GpuMonitor,
}

impl DreamResourceManager {
    /// Check and enforce GPU limits
    pub async fn enforce_limits(&self, state: &mut DreamState) -> ResourceAction {
        let gpu_usage = self.gpu_monitor.current_utilization();

        if gpu_usage > self.max_gpu_usage {
            // Throttle dream operations
            state.set_throttle(self.throttle_factor);

            warn!("Dream GPU usage {} exceeds limit {}, throttling",
                  gpu_usage, self.max_gpu_usage);

            ResourceAction::Throttle(self.throttle_factor)
        } else if gpu_usage > self.max_gpu_usage * 0.8 {
            // Approaching limit
            ResourceAction::Warning(gpu_usage)
        } else {
            ResourceAction::Continue
        }
    }

    /// Pause dream if resources critical
    pub async fn check_critical(&self) -> bool {
        let gpu_usage = self.gpu_monitor.current_utilization();
        let gpu_memory = self.gpu_monitor.memory_used_percentage();

        // Pause if external workload needs GPU
        gpu_usage > 0.7 || gpu_memory > 0.85
    }
}

pub enum ResourceAction {
    Continue,
    Warning(f32),
    Throttle(f32),
    Pause,
    Abort,
}
```

**Acceptance Criteria**:
- [ ] GPU usage monitored at 100ms intervals
- [ ] Operations throttled when exceeding 30%
- [ ] Dream pauses if external load needs GPU
- [ ] Resource usage logged continuously
- [ ] No GPU memory leaks during dream

---

#### REQ-DREAM-017: Dream Cycle Duration Management

**Priority**: High
**Description**: The system SHALL manage dream cycle timing with configurable durations.

```rust
pub struct DreamCycleManager {
    /// Full cycle duration (NREM + REM + transitions)
    pub cycle_duration: Duration,      // Default: 5 minutes (3+2)

    /// Simulated biological cycle ratio
    pub biological_ratio: Duration,    // Default: 90 minutes

    /// Maximum continuous dream time
    pub max_dream_time: Duration,      // Default: 30 minutes

    /// Minimum inter-cycle gap
    pub min_cycle_gap: Duration,       // Default: 10 minutes
}

impl DreamCycleManager {
    /// Plan next dream cycle
    pub fn plan_cycle(&self, context: &SystemContext) -> DreamCyclePlan {
        let last_cycle = context.last_dream_completed_at();
        let time_since_last = Instant::now().duration_since(last_cycle);

        // Enforce minimum gap
        if time_since_last < self.min_cycle_gap {
            return DreamCyclePlan::Delayed {
                wait_until: last_cycle + self.min_cycle_gap,
            };
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
}

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
```

**Acceptance Criteria**:
- [ ] NREM phase runs for 3 minutes
- [ ] REM phase runs for 2 minutes
- [ ] Total cycle time approximately 5 minutes
- [ ] Minimum 10-minute gap between cycles
- [ ] Maximum 30-minute continuous dream time
- [ ] Cycle timing logged for analysis

---

### 2.6 Integration Requirements

#### REQ-DREAM-018: MCP Tool Integration

**Priority**: Critical
**Description**: The system SHALL expose dream functionality through MCP tools.

```rust
/// MCP tool: trigger_dream
pub struct TriggerDreamTool {
    pub name: &'static str,  // "trigger_dream"
    pub description: &'static str,
}

#[derive(Deserialize)]
pub struct TriggerDreamParams {
    /// Phase to execute
    #[serde(default = "default_phase")]
    pub phase: DreamPhaseSelector,  // Default: FullCycle

    /// Duration in minutes
    #[serde(default = "default_duration")]
    pub duration_minutes: u32,       // Default: 5

    /// Number of synthetic queries (REM only)
    #[serde(default = "default_query_count")]
    pub synthetic_query_count: u32,  // Default: 100

    /// Whether to block until complete
    #[serde(default)]
    pub blocking: bool,              // Default: false

    /// Whether to abort on user query
    #[serde(default = "default_true")]
    pub abort_on_query: bool,        // Default: true
}

#[derive(Serialize)]
pub struct TriggerDreamResult {
    pub dream_id: Uuid,
    pub status: DreamStatus,
    pub phase: DreamPhaseSelector,
    pub started_at: DateTime<Utc>,
    pub estimated_completion: Option<DateTime<Utc>>,
    pub metrics: Option<DreamMetrics>,
}

impl TriggerDreamTool {
    pub async fn execute(&self, params: TriggerDreamParams, ctx: &mut ToolContext)
        -> Result<TriggerDreamResult, ToolError> {

        let dream_layer = ctx.get_dream_layer()?;

        // Check if dream already in progress
        if dream_layer.is_dreaming() {
            return Err(ToolError::DreamInProgress);
        }

        // Start dream cycle
        let dream_id = dream_layer.start(params.phase, params.duration_minutes)?;

        if params.blocking {
            // Wait for completion
            let result = dream_layer.wait_for_completion(dream_id).await?;
            Ok(TriggerDreamResult {
                dream_id,
                status: DreamStatus::Completed,
                phase: params.phase,
                started_at: result.started_at,
                estimated_completion: None,
                metrics: Some(result.metrics),
            })
        } else {
            Ok(TriggerDreamResult {
                dream_id,
                status: DreamStatus::InProgress,
                phase: params.phase,
                started_at: Utc::now(),
                estimated_completion: Some(Utc::now() + Duration::from_secs(params.duration_minutes as u64 * 60)),
                metrics: None,
            })
        }
    }
}

pub enum DreamPhaseSelector {
    NremOnly,
    RemOnly,
    FullCycle,
}
```

**Acceptance Criteria**:
- [ ] trigger_dream tool registered with MCP server
- [ ] Phase selection (NREM/REM/full) supported
- [ ] Duration configurable (1-10 minutes)
- [ ] Blocking and non-blocking modes
- [ ] Abort on query configurable
- [ ] Returns dream ID for status tracking

---

#### REQ-DREAM-019: Dream Status Monitoring

**Priority**: High
**Description**: The system SHALL provide dream status through get_memetic_status.

```rust
/// Extension to get_memetic_status response
#[derive(Serialize)]
pub struct DreamStatusExtension {
    /// Whether dream is available to trigger
    pub dream_available: bool,

    /// Current dream state if active
    pub active_dream: Option<ActiveDreamInfo>,

    /// Last dream cycle results
    pub last_dream_results: Option<LastDreamResults>,

    /// Time until dream can be triggered
    pub cooldown_remaining: Option<Duration>,
}

#[derive(Serialize)]
pub struct ActiveDreamInfo {
    pub dream_id: Uuid,
    pub phase: String,
    pub progress: f32,
    pub started_at: DateTime<Utc>,
    pub estimated_remaining: Duration,
}

#[derive(Serialize)]
pub struct LastDreamResults {
    pub completed_at: DateTime<Utc>,
    pub nrem_stats: NREMStats,
    pub rem_stats: REMStats,
    pub blind_spots_discovered: u32,
    pub nodes_consolidated: u32,
    pub compression_achieved: f32,
}
```

**Acceptance Criteria**:
- [ ] dream_available indicates if triggering is possible
- [ ] Active dream info includes phase and progress
- [ ] Last dream results include all key metrics
- [ ] Cooldown time shown when applicable
- [ ] Integration with existing get_memetic_status

---

#### REQ-DREAM-020: Integration with Neuromodulation

**Priority**: Medium
**Description**: The system SHALL coordinate dream phases with neuromodulation system.

```rust
pub struct DreamNeuromodulation {
    /// Dopamine levels during different phases
    pub dopamine_nrem: f32,   // Default: 0.3 (low during NREM)
    pub dopamine_rem: f32,    // Default: 0.7 (higher during REM)

    /// Serotonin levels
    pub serotonin_nrem: f32,  // Default: 0.4
    pub serotonin_rem: f32,   // Default: 0.8 (high exploration)

    /// Acetylcholine levels
    pub acetylcholine_nrem: f32,  // Default: 0.2 (low)
    pub acetylcholine_rem: f32,   // Default: 0.6 (moderate)
}

impl DreamNeuromodulation {
    /// Set neuromodulators for dream phase
    pub fn apply_phase_modulation(&self, phase: &DreamPhase, controller: &mut NeuromodulationController) {
        match phase {
            DreamPhase::NREM => {
                controller.dopamine = self.dopamine_nrem;
                controller.serotonin = self.serotonin_nrem;
                controller.acetylcholine = self.acetylcholine_nrem;
            }
            DreamPhase::REM => {
                controller.dopamine = self.dopamine_rem;
                controller.serotonin = self.serotonin_rem;
                controller.acetylcholine = self.acetylcholine_rem;
            }
            DreamPhase::Transition => {
                // Interpolate between phases
                controller.smooth_transition(Duration::from_millis(500));
            }
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Neuromodulator levels change by phase
- [ ] NREM has low dopamine/acetylcholine
- [ ] REM has high serotonin for exploration
- [ ] Smooth transitions between phases
- [ ] Levels reset to baseline on wake

---

### 2.7 Metrics and Monitoring Requirements

#### REQ-DREAM-021: Dream Metrics Collection

**Priority**: High
**Description**: The system SHALL collect comprehensive metrics during dream cycles.

```rust
pub struct DreamMetrics {
    /// NREM phase metrics
    pub nrem: NREMMetrics,

    /// REM phase metrics
    pub rem: REMMetrics,

    /// Overall cycle metrics
    pub cycle: CycleMetrics,

    /// Resource usage
    pub resources: ResourceMetrics,
}

#[derive(Default, Serialize)]
pub struct NREMMetrics {
    pub memories_replayed: u32,
    pub edges_strengthened: u32,
    pub edges_weakened: u32,
    pub edges_pruned: u32,
    pub clusters_consolidated: u32,
    pub nodes_merged: u32,
    pub redundancies_eliminated: u32,
    pub schemas_extracted: u32,
    pub compression_ratio: f32,
    pub duration_ms: u64,
}

#[derive(Default, Serialize)]
pub struct REMMetrics {
    pub synthetic_queries: u32,
    pub blind_spots_found: u32,
    pub new_edges_created: u32,
    pub average_semantic_leap: f32,
    pub exploration_coverage: f32,
    pub duration_ms: u64,
}

#[derive(Default, Serialize)]
pub struct CycleMetrics {
    pub total_duration_ms: u64,
    pub wake_events: u32,
    pub abort_reason: Option<String>,
    pub information_loss_estimate: f32,
    pub quality_improvement: f32,
}

#[derive(Default, Serialize)]
pub struct ResourceMetrics {
    pub peak_gpu_usage: f32,
    pub average_gpu_usage: f32,
    pub peak_memory_mb: u32,
    pub throttle_events: u32,
}
```

**Acceptance Criteria**:
- [ ] All listed metrics collected
- [ ] Metrics available via API
- [ ] Metrics persisted for trend analysis
- [ ] Prometheus-compatible export
- [ ] Alerts on anomalous values

---

#### REQ-DREAM-022: Dream Quality Assessment

**Priority**: Medium
**Description**: The system SHALL assess dream cycle quality and effectiveness.

```rust
pub struct DreamQualityAssessor {
    /// Minimum acceptable compression ratio
    pub min_compression: f32,           // Default: 5.0

    /// Minimum blind spots per cycle
    pub min_blind_spots: u32,           // Default: 3

    /// Maximum acceptable information loss
    pub max_information_loss: f32,      // Default: 0.15

    /// Quality score thresholds
    pub quality_thresholds: QualityThresholds,
}

#[derive(Serialize)]
pub struct DreamQualityReport {
    pub overall_score: f32,              // 0.0-1.0
    pub grade: QualityGrade,
    pub compression_achieved: bool,
    pub exploration_adequate: bool,
    pub information_preserved: bool,
    pub recommendations: Vec<String>,
}

pub enum QualityGrade {
    Excellent,  // score >= 0.9
    Good,       // score >= 0.7
    Fair,       // score >= 0.5
    Poor,       // score < 0.5
}

impl DreamQualityAssessor {
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

**Acceptance Criteria**:
- [ ] Quality score computed from metrics
- [ ] Grade assigned (Excellent/Good/Fair/Poor)
- [ ] Individual criteria assessed
- [ ] Recommendations generated
- [ ] Report available via API

---

## 3. Non-Functional Requirements

### 3.1 Performance Requirements

#### REQ-DREAM-023: Wake Latency

**Priority**: Critical
**Description**: The system SHALL achieve wake latency under 100ms.

| Metric | Target | Measurement |
|--------|--------|-------------|
| Abort signal propagation | <10ms | Time to set abort flag |
| Operation cancellation | <50ms | Current operation stops |
| State checkpoint save | <30ms | Minimal state saved |
| Total wake latency | <100ms | Query received to ready |

**Acceptance Criteria**:
- [ ] P95 wake latency < 100ms
- [ ] P99 wake latency < 150ms
- [ ] No query blocked by dream operation
- [ ] State saved for later resumption

---

#### REQ-DREAM-024: GPU Resource Budget

**Priority**: Critical
**Description**: The system SHALL limit GPU usage during dream cycles.

| Resource | Budget | Measurement |
|----------|--------|-------------|
| GPU Compute | <30% | nvidia-smi utilization |
| GPU Memory | <4GB additional | Dream-specific allocations |
| Memory Bandwidth | <30% | Transfer monitoring |

**Acceptance Criteria**:
- [ ] GPU usage never exceeds 30%
- [ ] Dream pauses if external load increases
- [ ] Memory released immediately on wake
- [ ] No memory leaks over multiple cycles

---

### 3.2 Reliability Requirements

#### REQ-DREAM-025: Fault Tolerance

**Priority**: High
**Description**: The system SHALL handle failures gracefully during dream cycles.

```rust
pub struct DreamFaultHandler {
    /// Maximum retries for recoverable errors
    pub max_retries: u32,            // Default: 3

    /// Checkpoint interval
    pub checkpoint_interval: Duration, // Default: 30s

    /// Recovery strategy
    pub recovery_strategy: RecoveryStrategy,
}

pub enum RecoveryStrategy {
    /// Abort and log
    AbortOnError,
    /// Retry current operation
    RetryOperation,
    /// Skip failed operation, continue
    SkipAndContinue,
    /// Rollback to last checkpoint
    RollbackToCheckpoint,
}
```

**Acceptance Criteria**:
- [ ] Checkpoints saved every 30 seconds
- [ ] Recoverable errors retried up to 3 times
- [ ] Unrecoverable errors abort cleanly
- [ ] No data corruption on failure
- [ ] Errors logged with context

---

#### REQ-DREAM-026: Data Integrity

**Priority**: Critical
**Description**: The system SHALL maintain data integrity during consolidation.

**Acceptance Criteria**:
- [ ] Consolidated nodes preserve all essential information
- [ ] Edge relationships correctly transferred
- [ ] Soft-deleted nodes recoverable for 30 days
- [ ] Reversal hash generated for all merges
- [ ] Audit trail for all modifications

---

### 3.3 Scalability Requirements

#### REQ-DREAM-027: Graph Size Scaling

**Priority**: High
**Description**: The system SHALL scale dream operations with graph size.

| Graph Size | NREM Time | REM Time | Total Time |
|------------|-----------|----------|------------|
| 100K nodes | 3 min | 2 min | 5 min |
| 1M nodes | 5 min | 3 min | 8 min |
| 10M nodes | 10 min | 5 min | 15 min |

**Acceptance Criteria**:
- [ ] Sub-linear scaling with graph size
- [ ] Sampling strategies for large graphs
- [ ] Memory-efficient batch processing
- [ ] Progress tracking for long cycles

---

## 4. Data Requirements

### 4.1 Dream State Persistence

#### REQ-DREAM-028: State Checkpoint Format

**Priority**: High
**Description**: The system SHALL persist dream state for recovery.

```rust
#[derive(Serialize, Deserialize)]
pub struct DreamCheckpoint {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub phase: DreamPhase,
    pub progress: DreamProgress,
    pub src_state: SRCState,
    pub pending_operations: Vec<PendingOperation>,
    pub metrics_so_far: PartialDreamMetrics,
}

impl DreamCheckpoint {
    /// Save checkpoint to disk
    pub async fn save(&self, path: &Path) -> Result<(), DreamError>;

    /// Load checkpoint from disk
    pub async fn load(path: &Path) -> Result<Self, DreamError>;

    /// Resume dream from checkpoint
    pub async fn resume(&self, dream_layer: &mut DreamLayer) -> Result<(), DreamError>;
}
```

**Acceptance Criteria**:
- [ ] Checkpoints saved atomically
- [ ] Checkpoint size < 1MB typically
- [ ] Resume from checkpoint works correctly
- [ ] Old checkpoints cleaned up automatically

---

### 4.2 Dream History

#### REQ-DREAM-029: Dream History Recording

**Priority**: Medium
**Description**: The system SHALL maintain history of dream cycles.

```rust
pub struct DreamHistory {
    /// Maximum history entries
    pub max_entries: usize,  // Default: 100

    /// History storage
    entries: VecDeque<DreamHistoryEntry>,
}

#[derive(Serialize, Deserialize)]
pub struct DreamHistoryEntry {
    pub dream_id: Uuid,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub status: DreamCompletionStatus,
    pub metrics_summary: DreamMetricsSummary,
    pub quality_grade: QualityGrade,
}
```

**Acceptance Criteria**:
- [ ] Last 100 dream cycles recorded
- [ ] History queryable via API
- [ ] Metrics trends analyzable
- [ ] Old entries pruned automatically

---

## 5. Interface Requirements

### 5.1 MCP Tool Interfaces

#### REQ-DREAM-030: Tool Schema Compliance

**Priority**: Critical
**Description**: All dream tools SHALL comply with MCP JSON-RPC 2.0.

**trigger_dream Tool Schema**:
```json
{
  "name": "trigger_dream",
  "description": "Trigger memory consolidation dream cycle",
  "inputSchema": {
    "type": "object",
    "properties": {
      "phase": {
        "type": "string",
        "enum": ["nrem", "rem", "full_cycle"],
        "default": "full_cycle"
      },
      "duration_minutes": {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "default": 5
      },
      "synthetic_query_count": {
        "type": "integer",
        "minimum": 10,
        "maximum": 500,
        "default": 100
      },
      "blocking": {
        "type": "boolean",
        "default": false
      },
      "abort_on_query": {
        "type": "boolean",
        "default": true
      }
    }
  }
}
```

**Acceptance Criteria**:
- [ ] Schema matches MCP specification
- [ ] All parameters validated
- [ ] Default values applied correctly
- [ ] Error responses follow MCP format

---

## 6. Configuration Requirements

### 6.1 Dream Configuration Schema

#### REQ-DREAM-031: Configuration File Format

**Priority**: High
**Description**: The system SHALL support TOML-based dream configuration.

```toml
[dream]
enabled = true

[dream.trigger]
activity_threshold = 0.15
idle_duration_minutes = 10
cooldown_minutes = 30

[dream.nrem]
duration_minutes = 3
replay_recency_bias = 0.8
learning_rate = 0.01
batch_size = 64

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

[dream.resources]
max_gpu_usage = 0.30
wake_latency_ms = 100

[dream.quality]
target_compression_ratio = 10.0
min_blind_spots = 5
max_information_loss = 0.15
```

**Acceptance Criteria**:
- [ ] All parameters configurable via TOML
- [ ] Configuration validated on load
- [ ] Hot-reload supported
- [ ] Invalid config returns clear error

---

## 7. Error Handling Requirements

### 7.1 Dream Error Codes

#### REQ-DREAM-032: Error Code Catalog

**Priority**: High
**Description**: The system SHALL use consistent error codes for dream operations.

| Code | Name | Description |
|------|------|-------------|
| -32100 | DreamInProgress | Dream cycle already running |
| -32101 | DreamCooldown | Must wait before next cycle |
| -32102 | DreamAborted | Dream was aborted |
| -32103 | DreamTimeout | Dream exceeded time limit |
| -32104 | DreamResourceError | GPU resources unavailable |
| -32105 | DreamCheckpointError | Failed to save/load checkpoint |
| -32106 | DreamConsolidationError | Consolidation operation failed |
| -32107 | DreamExplorationError | REM exploration failed |

**Acceptance Criteria**:
- [ ] All errors mapped to codes
- [ ] Error messages descriptive
- [ ] Errors include recovery hints
- [ ] Errors logged with context

---

## 8. Testing Requirements

### 8.1 Unit Tests

#### REQ-DREAM-033: Unit Test Coverage

**Priority**: High
**Description**: The system SHALL have comprehensive unit test coverage.

| Component | Target Coverage |
|-----------|-----------------|
| SRC Algorithm | 95% |
| NREM Phase | 90% |
| REM Phase | 90% |
| Blind Spot Detection | 95% |
| Edge Creation | 90% |
| Resource Management | 85% |

**Acceptance Criteria**:
- [ ] Coverage targets met
- [ ] All edge cases tested
- [ ] Mocks for external dependencies
- [ ] Tests run in <30 seconds

---

### 8.2 Integration Tests

#### REQ-DREAM-034: Integration Test Scenarios

**Priority**: High
**Description**: The system SHALL pass integration test scenarios.

**Scenarios**:
1. Full cycle completion with graph modifications
2. Abort on query during NREM
3. Abort on query during REM
4. Resource throttling behavior
5. Recovery from checkpoint
6. Multiple sequential cycles
7. Integration with neuromodulation
8. MCP tool round-trip

**Acceptance Criteria**:
- [ ] All scenarios pass
- [ ] Tests use realistic graph data
- [ ] Timing requirements verified
- [ ] Resource limits enforced

---

### 8.3 Benchmark Tests

#### REQ-DREAM-035: Performance Benchmarks

**Priority**: Medium
**Description**: The system SHALL meet performance benchmarks.

| Benchmark | Target | Measurement |
|-----------|--------|-------------|
| NREM 100K nodes | <3 min | Wall clock time |
| REM 100 queries | <2 min | Wall clock time |
| Wake latency | <100ms | P95 |
| Compression ratio | 10:1 | Before/after nodes |
| Blind spot discovery | >5/cycle | Count |

**Acceptance Criteria**:
- [ ] Benchmarks automated in CI
- [ ] Regression detection enabled
- [ ] Historical trends tracked
- [ ] Performance reports generated

---

### 8.4 Amortized Shortcut Creation

#### REQ-DREAM-036: Amortized Shortcut Creation for Frequent Paths

**Priority**: Should
**Description**: SRC SHALL create amortized shortcuts for 3+ hop frequent paths during sleep replay to optimize future inference costs (Marblestone inference amortization).

| Requirement | Description | Priority | Reference |
|-------------|-------------|----------|-----------|
| REQ-DREAM-036 | SRC SHALL create amortized shortcuts for 3+ hop frequent paths | Should | Marblestone inference amortization |

**Acceptance Criteria**:
- [ ] Shortcuts created for paths with 3+ hops
- [ ] Traversal count threshold of 5 enforced
- [ ] Shortcut edges marked with `is_amortized_shortcut = true`
- [ ] Edge weight computed as product of path weights

---

#### REQ-DREAM-037: Shortcut Quality Gate

**Priority**: Must
**Description**: Shortcuts SHALL require path confidence >= 0.7 as a quality gate to ensure only reliable inference paths are amortized.

| Requirement | Description | Priority | Reference |
|-------------|-------------|----------|-----------|
| REQ-DREAM-037 | Shortcuts SHALL require path confidence >= 0.7 | Must | Quality gate |

**Acceptance Criteria**:
- [ ] Path confidence calculated as product of all edge confidences
- [ ] Shortcuts only created when path_confidence >= 0.7
- [ ] Quality gate prevents low-confidence shortcut creation
- [ ] Metrics track rejected shortcuts due to low confidence

---

## 9. Acceptance Criteria Summary

### 9.1 Critical Acceptance Criteria

1. [ ] Dream triggers automatically on 10-minute idle at <0.15 activity
2. [ ] NREM phase achieves 10:1 compression ratio
3. [ ] REM phase discovers >5 blind spots per cycle
4. [ ] Wake latency <100ms on user query
5. [ ] GPU usage <30% during dream
6. [ ] No data loss or corruption during consolidation
7. [ ] MCP tool integration complete
8. [ ] All quality gates pass

### 9.2 Quality Gates

| Gate | Criteria |
|------|----------|
| Code Review | All code reviewed and approved |
| Unit Tests | 90% coverage, all passing |
| Integration Tests | All scenarios passing |
| Performance | All benchmarks met |
| Documentation | API docs complete |

---

## 10. References

### 10.1 Internal References

- constitution.yaml: dream_layer configuration (lines 772-795)
- contextprd.md: Section 7.1 Dream Layer (SRC Algorithm)
- implementationplan.md: Module 9 specification

### 10.2 External Research

- NeuroDream: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5377250
- Sleep Replay Consolidation: https://www.nature.com/articles/s41467-022-34938-7

---

## 11. Appendix

### 11.1 Requirement Traceability Matrix

| Requirement ID | PRD Section | Constitution Reference | Test Case |
|---------------|-------------|----------------------|-----------|
| REQ-DREAM-001 | 7.1 | dream_layer | T-DREAM-001 |
| REQ-DREAM-002 | 7.1 | dream_layer.trigger | T-DREAM-002 |
| REQ-DREAM-003 | 7.1 | dream_layer.constraints | T-DREAM-003 |
| REQ-DREAM-004 | 7.1 | dream_layer.phases.nrem | T-DREAM-004 |
| REQ-DREAM-005 | 7.1 | dream_layer.phases.nrem.replay_recency_bias | T-DREAM-005 |
| REQ-DREAM-006 | 7.1 | Hebbian update | T-DREAM-006 |
| REQ-DREAM-007 | 7.1 | Consolidation | T-DREAM-007 |
| REQ-DREAM-008 | 7.1 | Redundancy elimination | T-DREAM-008 |
| REQ-DREAM-009 | 7.1 | Schema extraction | T-DREAM-009 |
| REQ-DREAM-010 | 7.1 | dream_layer.phases.rem | T-DREAM-010 |
| REQ-DREAM-011 | 7.1 | Synthetic queries | T-DREAM-011 |
| REQ-DREAM-012 | 7.1 | Blind spots | T-DREAM-012 |
| REQ-DREAM-013 | 7.1 | New edges | T-DREAM-013 |
| REQ-DREAM-014 | 7.1 | SRC algorithm | T-DREAM-014 |
| REQ-DREAM-015 | 7.1 | SRC tuning | T-DREAM-015 |
| REQ-DREAM-016 | 7.1 | dream_layer.constraints.gpu_usage | T-DREAM-016 |
| REQ-DREAM-017 | 7.1 | Cycle duration | T-DREAM-017 |
| REQ-DREAM-018 | 7.1 | trigger_dream tool | T-DREAM-018 |
| REQ-DREAM-019 | 7.1 | Dream status | T-DREAM-019 |
| REQ-DREAM-020 | 7.1/7.2 | Neuromodulation | T-DREAM-020 |
| REQ-DREAM-021 | 7.1 | Metrics | T-DREAM-021 |
| REQ-DREAM-022 | 7.1 | Quality assessment | T-DREAM-022 |
| REQ-DREAM-023 | 7.1 | dream_layer.constraints.wake_latency | T-DREAM-023 |
| REQ-DREAM-024 | 7.1 | dream_layer.constraints.gpu_usage | T-DREAM-024 |
| REQ-DREAM-025 | - | Fault tolerance | T-DREAM-025 |
| REQ-DREAM-026 | - | Data integrity | T-DREAM-026 |
| REQ-DREAM-027 | - | Scalability | T-DREAM-027 |
| REQ-DREAM-028 | - | Checkpoints | T-DREAM-028 |
| REQ-DREAM-029 | - | History | T-DREAM-029 |
| REQ-DREAM-030 | - | MCP compliance | T-DREAM-030 |
| REQ-DREAM-031 | - | Configuration | T-DREAM-031 |
| REQ-DREAM-032 | - | Error codes | T-DREAM-032 |
| REQ-DREAM-033 | - | Unit tests | T-DREAM-033 |
| REQ-DREAM-034 | - | Integration tests | T-DREAM-034 |
| REQ-DREAM-035 | - | Benchmarks | T-DREAM-035 |
| REQ-DREAM-036 | 7.1 | Marblestone inference amortization | T-DREAM-036 |
| REQ-DREAM-037 | 7.1 | Shortcut quality gate | T-DREAM-037 |

---

*Document Version: 1.0.0*
*Generated: 2025-12-31*
*Agent: #9/28 - Dream Layer Specification*
