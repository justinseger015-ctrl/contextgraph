# Module 4: Knowledge Graph - Functional Specification

**Version**: 1.0.0
**Status**: Draft
**Phase**: 3
**Duration**: 4 weeks
**Dependencies**: Module 3 (Embedding Pipeline)
**Last Updated**: 2025-12-31

---

## 1. Overview

### 1.1 Purpose

The Knowledge Graph module implements the graph storage layer combining FAISS GPU-accelerated vector similarity search with a property graph for structured relationships. This module enables both semantic retrieval via embedding similarity and graph traversal operations for relationship exploration.

### 1.2 Problem Statement

AI memory systems require:
- Sub-10ms semantic search across millions of vectors
- Graph-based relationship traversal for causal and hierarchical reasoning
- O(1) entailment queries for IS-A hierarchies using hyperbolic geometry
- Contradiction detection for maintaining knowledge consistency
- Incremental updates without expensive full index rebuilds

### 1.3 Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Vector Search Latency | <10ms for k=100 | Benchmark 1M vectors |
| FAISS Index Query | <2ms for k=100, 1M vectors | GPU profiling |
| Graph Traversal | 6+ depth efficiently | Stress test with 10M nodes |
| Vector Capacity | 10M+ vectors | Load test |
| Entailment Query | <1ms (O(1)) | Cone containment benchmark |
| Incremental Update | No full rebuild | Verify index state |

---

## 2. User Stories

### 2.1 US-KG-01: Semantic Memory Search

**Priority**: Must-Have

**Narrative**:
As an AI agent,
I want to search for semantically similar knowledge nodes,
So that I can retrieve relevant context for my current task.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-KG-01-01 | A graph with 1M embedded nodes | I query with a 1536D vector and k=100 | I receive top-100 similar nodes in <10ms |
| AC-KG-01-02 | A query embedding | Search completes | Results are ranked by cosine similarity |
| AC-KG-01-03 | The FAISS index is trained | A new node is added | Node is searchable without full rebuild |

### 2.2 US-KG-02: Graph Relationship Traversal

**Priority**: Must-Have

**Narrative**:
As an AI agent,
I want to traverse relationships between knowledge nodes,
So that I can discover causal chains and related concepts.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-KG-02-01 | A node with edges | I traverse with depth=3 | I receive all reachable nodes within 3 hops |
| AC-KG-02-02 | Edge type filters specified | I traverse with edge_types=[Causal] | Only Causal edges are followed |
| AC-KG-02-03 | A deeply connected graph | I traverse to depth=6 | Operation completes efficiently |

### 2.3 US-KG-03: Hierarchical Entailment Queries

**Priority**: Must-Have

**Narrative**:
As an AI agent,
I want to query IS-A hierarchies in constant time,
So that I can efficiently reason about concept relationships.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-KG-03-01 | Nodes positioned in Poincare ball | I query entailment for a concept | I receive ancestors/descendants in O(1) |
| AC-KG-03-02 | A node with entailment cone | I check if another node is entailed | Cone containment returns boolean in <1ms |
| AC-KG-03-03 | A concept hierarchy exists | I query for all subconcepts | All entailed concepts are returned |

### 2.4 US-KG-04: Contradiction Detection

**Priority**: Must-Have

**Narrative**:
As an AI agent,
I want the system to detect contradictory knowledge,
So that I can maintain consistency in my reasoning.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-KG-04-01 | A node with potential conflicts | I request contradiction detection | Conflicting nodes are identified |
| AC-KG-04-02 | High similarity but opposing content | System analyzes node pair | Contradiction is flagged |
| AC-KG-04-03 | Contradictions detected | Results returned | Each includes confidence score |

---

## 3. Requirements

### 3.1 Functional Requirements

#### 3.1.1 FAISS GPU Index

| ID | Requirement | Priority | Rationale |
|----|-------------|----------|-----------|
| REQ-KG-001 | System SHALL use FAISS GpuIndexIVFPQ for vector storage | Must | GPU acceleration for <2ms search |
| REQ-KG-002 | Index SHALL configure nlist=16384 clusters | Must | Optimal for 10M+ vectors |
| REQ-KG-003 | Search SHALL use nprobe=128 clusters | Must | Balance recall vs latency |
| REQ-KG-004 | Product quantization SHALL use pq_m=64 sub-quantizers | Must | Memory efficiency |
| REQ-KG-005 | Product quantization SHALL use pq_bits=8 bits per code | Must | 1536D vector compression |
| REQ-KG-006 | Index SHALL support incremental vector addition | Must | Avoid full rebuild |
| REQ-KG-007 | Index SHALL support vector removal by ID | Should | Memory cleanup |
| REQ-KG-008 | Index SHALL persist to disk for recovery | Must | Durability |

#### 3.1.2 Graph Storage

| ID | Requirement | Priority | Rationale |
|----|-------------|----------|-----------|
| REQ-KG-010 | System SHALL store nodes in RocksDB | Must | High-performance key-value store |
| REQ-KG-011 | Nodes SHALL reference embeddings by vector ID | Must | Decouple storage from FAISS |
| REQ-KG-012 | System SHALL support edge types: RELATES_TO, CAUSES, PRECEDES, CONTAINS, CONTRADICTS | Must | Relationship semantics |
| REQ-KG-013 | Edges SHALL be bidirectionally indexed | Must | Efficient traversal both directions |
| REQ-KG-014 | System SHALL implement graph partitioning for large-scale traversal | Should | Performance at scale |
| REQ-KG-015 | All graph mutations SHALL be atomic | Must | Data consistency |

#### 3.1.3 KnowledgeNode Structure

| ID | Requirement | Priority | Rationale |
|----|-------------|----------|-----------|
| REQ-KG-020 | Node id SHALL be UUID v4 | Must | Global uniqueness |
| REQ-KG-021 | Node content SHALL support max 65536 characters | Must | Large context storage |
| REQ-KG-022 | Node embedding SHALL be Vector1536 | Must | FuseMoE output dimension |
| REQ-KG-023 | Node SHALL track created_at and last_accessed timestamps | Must | Temporal queries |
| REQ-KG-024 | Node importance SHALL be f32 in range [0,1] | Must | Salience scoring |
| REQ-KG-025 | Node access_count SHALL be u32 | Must | Usage tracking |
| REQ-KG-026 | Node SHALL have johari_quadrant: Open, Blind, Hidden, Unknown | Must | UTL classification |
| REQ-KG-027 | Node SHALL track utl_state: {delta_s, delta_c, w_e, phi} | Must | Learning dynamics |
| REQ-KG-028 | Node SHALL have optional agent_id for multi-agent | Should | Agent isolation |
| REQ-KG-029 | Node SHALL have observer_perspective: {domain, confidence_priors} | Should | Perspective tracking |
| REQ-KG-030 | Node SHALL have optional semantic_cluster UUID | Should | Clustering support |
| REQ-KG-031 | Node SHALL have priors_vibe_check: {assumption_embedding[128], domain_priors, prior_confidence} | Must | Merge safety |

#### 3.1.4 GraphEdge Structure

| ID | Requirement | Priority | Rationale |
|----|-------------|----------|-----------|
| REQ-KG-040 | Edge SHALL have source and target UUIDs | Must | Node references |
| REQ-KG-041 | Edge edge_type SHALL be: Semantic, Temporal, Causal, Hierarchical, Relational | Must | Relationship classification |
| REQ-KG-042 | Edge weight SHALL be f32 in range [0,1] | Must | Relationship strength |
| REQ-KG-043 | Edge confidence SHALL be f32 in range [0,1] | Must | Certainty measure |
| REQ-KG-044 | Edge SHALL track created_at timestamp | Must | Temporal ordering |

#### 3.1.5 Hyperbolic Coordinates (Poincare Ball)

| ID | Requirement | Priority | Rationale |
|----|-------------|----------|-----------|
| REQ-KG-050 | All nodes SHALL have position in Poincare ball where norm < 1 | Must | Hyperbolic geometry constraint |
| REQ-KG-051 | System SHALL compute hyperbolic distance using formula: d(x,y) = arcosh(1 + 2*norm(x-y)^2 / ((1-norm(x)^2)*(1-norm(y)^2))) | Must | Correct distance metric |
| REQ-KG-052 | System SHALL implement EntailmentCone with apex, aperture, axis fields | Must | O(1) hierarchy queries |
| REQ-KG-053 | EntailmentCone.contains() SHALL check angle(tangent, axis) <= aperture | Must | Cone containment test |
| REQ-KG-054 | Hyperbolic coordinates SHALL be 64-dimensional | Should | Balance expressiveness and compute |

#### 3.1.6 Query Operations

| ID | Requirement | Priority | Rationale |
|----|-------------|----------|-----------|
| REQ-KG-060 | semantic_search(query, k) SHALL return top-k similar nodes | Must | Core retrieval |
| REQ-KG-061 | graph_traverse(start, depth, edge_types) SHALL perform BFS/DFS traversal | Must | Relationship exploration |
| REQ-KG-062 | entailment_query(concept) SHALL return IS-A hierarchy via cones | Must | Hierarchical reasoning |
| REQ-KG-063 | contradiction_detect(node) SHALL identify conflicting nodes | Must | Consistency maintenance |
| REQ-KG-064 | All queries SHALL respect perspective_lock for multi-agent isolation | Should | Agent safety |
| REQ-KG-065 | System SHALL support domain-aware edge weight modulation | Must | Marblestone neurotransmitter profiles |

### 3.2 Non-Functional Requirements

| ID | Category | Requirement | Metric |
|----|----------|-------------|--------|
| NFR-KG-001 | Performance | FAISS search <2ms for 1M vectors, k=100 | P95 latency |
| NFR-KG-002 | Performance | Full semantic search pipeline <10ms | P95 latency |
| NFR-KG-003 | Performance | Graph traversal depth=6 <100ms | P95 latency |
| NFR-KG-004 | Performance | Entailment cone containment <1ms | P95 latency |
| NFR-KG-005 | Scalability | Support 10M+ vectors | Capacity test |
| NFR-KG-006 | Scalability | Support 100M+ edges | Capacity test |
| NFR-KG-007 | Memory | GPU memory <24GB under load | Resource monitoring |
| NFR-KG-008 | Reliability | No data loss on process restart | Durability test |
| NFR-KG-009 | Concurrency | Support concurrent read/write | Thread safety test |

---

## 4. Data Models

### 4.1 KnowledgeNode

```rust
/// A node in the knowledge graph representing a unit of knowledge.
///
/// Constraint: content.len() <= 65536
/// Constraint: importance in [0.0, 1.0]
/// Constraint: All UTL state values in valid ranges
pub struct KnowledgeNode {
    /// Unique identifier (UUID v4)
    pub id: Uuid,

    /// Text content (max 65536 chars)
    pub content: String,

    /// Fused embedding from 12-model pipeline
    pub embedding: Vector1536,

    /// Position in Poincare ball for hyperbolic operations
    /// Constraint: ||hyperbolic_position|| < 1.0
    pub hyperbolic_position: HyperbolicPoint,

    /// Entailment cone for O(1) IS-A queries
    pub entailment_cone: Option<EntailmentCone>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,

    /// Importance/salience score [0,1]
    pub importance: f32,

    /// Number of times accessed
    pub access_count: u32,

    /// Johari Window classification
    pub johari_quadrant: JohariQuadrant,

    /// UTL learning state
    pub utl_state: UTLState,

    /// Optional agent identifier for multi-agent scenarios
    pub agent_id: Option<String>,

    /// Observer perspective metadata
    pub observer_perspective: Option<ObserverPerspective>,

    /// Optional semantic cluster assignment
    pub semantic_cluster: Option<Uuid>,

    /// Priors compatibility check for safe merging
    pub priors_vibe_check: PriorsVibeCheck,
}

/// Johari Window quadrant for knowledge classification
pub enum JohariQuadrant {
    /// Known to self and others - direct recall
    Open,
    /// Unknown to self - discovery zone
    Blind,
    /// Known to self only - private knowledge
    Hidden,
    /// Unknown to all - exploration frontier
    Unknown,
}

/// UTL learning state parameters
pub struct UTLState {
    /// Entropy change (novelty/surprise) [0,1]
    pub delta_s: f32,
    /// Coherence change (understanding) [0,1]
    pub delta_c: f32,
    /// Emotional modulation weight [0.5,1.5]
    pub w_e: f32,
    /// Phase synchronization angle [0,pi]
    pub phi: f32,
}

/// Observer perspective for multi-perspective knowledge
pub struct ObserverPerspective {
    /// Knowledge domain
    pub domain: String,
    /// Prior confidence levels per domain
    pub confidence_priors: HashMap<String, f32>,
}

/// Priors compatibility metadata for merge safety
pub struct PriorsVibeCheck {
    /// 128-dimensional assumption embedding
    pub assumption_embedding: [f32; 128],
    /// Domain prior weights
    pub domain_priors: HashMap<String, f32>,
    /// Overall prior confidence [0,1]
    pub prior_confidence: f32,
}
```

### 4.2 GraphEdge

```rust
/// An edge connecting two knowledge nodes in the graph.
///
/// Constraint: weight in [0.0, 1.0]
/// Constraint: confidence in [0.0, 1.0]
/// Constraint: source != target (no self-loops)
pub struct GraphEdge {
    /// Source node UUID
    pub source: Uuid,

    /// Target node UUID
    pub target: Uuid,

    /// Type of relationship
    pub edge_type: EdgeType,

    /// Relationship strength [0,1]
    pub weight: f32,

    /// Certainty of this relationship [0,1]
    pub confidence: f32,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Neurotransmitter-inspired weight modulation (Marblestone)
    /// Controls domain-specific excitatory/inhibitory balance
    pub neurotransmitter_weights: NeurotransmitterWeights,
}

/// Neurotransmitter-inspired edge weight modulation (Marblestone).
///
/// Models biological synaptic modulation for domain-aware edge weighting.
/// High excitatory + low inhibitory = strong activation.
/// High inhibitory + low excitatory = suppression.
/// Modulatory adjusts sensitivity.
///
/// Constraint: All values in [0.0, 1.0]
pub struct NeurotransmitterWeights {
    /// Excitatory contribution (like glutamate)
    /// Higher values strengthen edge activation
    pub excitatory: f32,

    /// Inhibitory contribution (like GABA)
    /// Higher values suppress edge activation
    pub inhibitory: f32,

    /// Modulatory contribution (like dopamine/serotonin)
    /// Adjusts overall sensitivity of modulation
    pub modulatory: f32,
}

impl Default for NeurotransmitterWeights {
    fn default() -> Self {
        Self {
            excitatory: 0.5,
            inhibitory: 0.5,
            modulatory: 0.0,
        }
    }
}

/// Types of edges in the knowledge graph
pub enum EdgeType {
    /// General semantic similarity
    Semantic,
    /// Temporal ordering (before/after)
    Temporal,
    /// Causal relationship (causes/effects)
    Causal,
    /// Hierarchical IS-A relationship
    Hierarchical,
    /// Other named relationship
    Relational,
}
```

### 4.3 Hyperbolic Geometry Types

```rust
/// A point in the Poincare ball model of hyperbolic space.
///
/// Constraint: ||coordinates|| < 1.0 (strict inequality)
/// The closer to the boundary, the "further" from origin in hyperbolic terms.
pub struct HyperbolicPoint {
    /// Coordinates in the Poincare ball (64-dimensional)
    /// Constraint: sum(coord^2) < 1.0
    pub coordinates: [f32; 64],
}

impl HyperbolicPoint {
    /// Compute hyperbolic distance between two points.
    ///
    /// Formula: d(x,y) = arcosh(1 + 2*||x-y||^2 / ((1-||x||^2)*(1-||y||^2)))
    ///
    /// # Arguments
    /// * `other` - The other point to compute distance to
    ///
    /// # Returns
    /// The hyperbolic distance (non-negative)
    ///
    /// # Panics
    /// Panics if either point has norm >= 1.0
    ///
    /// Constraint: Operation_Latency < 100μs
    pub fn distance(&self, other: &HyperbolicPoint) -> f32 {
        let x_norm_sq = self.norm_squared();
        let y_norm_sq = other.norm_squared();

        assert!(x_norm_sq < 1.0, "Point x outside Poincare ball");
        assert!(y_norm_sq < 1.0, "Point y outside Poincare ball");

        let diff_norm_sq = self.diff_norm_squared(other);

        let numerator = 2.0 * diff_norm_sq;
        let denominator = (1.0 - x_norm_sq) * (1.0 - y_norm_sq);

        // arcosh(z) = ln(z + sqrt(z^2 - 1))
        let z = 1.0 + numerator / denominator;
        (z + (z * z - 1.0).sqrt()).ln()
    }

    /// Compute squared Euclidean norm.
    fn norm_squared(&self) -> f32 {
        self.coordinates.iter().map(|x| x * x).sum()
    }

    /// Compute squared Euclidean distance to another point.
    fn diff_norm_squared(&self, other: &HyperbolicPoint) -> f32 {
        self.coordinates.iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }
}

/// An entailment cone for O(1) IS-A hierarchy queries.
///
/// A concept A entails concept B if B lies within A's entailment cone.
/// This enables constant-time hierarchical reasoning.
pub struct EntailmentCone {
    /// Apex point of the cone in Poincare ball
    pub apex: HyperbolicPoint,

    /// Half-aperture angle in radians [0, pi/2]
    /// Smaller aperture = more specific concept
    pub aperture: f32,

    /// Cone axis direction (unit vector in 1536D embedding space)
    pub axis: Vector1536,
}

impl EntailmentCone {
    /// Check if a point lies within this entailment cone.
    ///
    /// Uses tangent space projection to compute angle between
    /// the direction to the point and the cone axis.
    ///
    /// # Arguments
    /// * `point` - The hyperbolic point to test
    ///
    /// # Returns
    /// `true` if point is entailed (within cone), `false` otherwise
    ///
    /// Constraint: Operation_Latency < 1ms
    pub fn contains(&self, point: &HyperbolicPoint) -> bool {
        // Compute tangent vector from apex to point
        let tangent = self.compute_tangent_to(point);

        // Compute angle between tangent and cone axis
        let angle = self.angle_to_axis(&tangent);

        // Point is contained if angle <= aperture
        angle <= self.aperture
    }

    /// Compute tangent vector from apex to point in tangent space.
    fn compute_tangent_to(&self, point: &HyperbolicPoint) -> [f32; 64] {
        // Log map from apex to point gives tangent direction
        // Simplified: direction in ambient space scaled by hyperbolic factors
        let mut tangent = [0.0f32; 64];
        let apex_norm_sq = self.apex.norm_squared();
        let scale = 2.0 / (1.0 - apex_norm_sq);

        for i in 0..64 {
            tangent[i] = (point.coordinates[i] - self.apex.coordinates[i]) * scale;
        }
        tangent
    }

    /// Compute angle between tangent vector and cone axis.
    fn angle_to_axis(&self, tangent: &[f32; 64]) -> f32 {
        // Project axis to 64D for comparison (take first 64 dims)
        let axis_64: Vec<f32> = self.axis.values[..64].to_vec();

        // Cosine similarity
        let dot: f32 = tangent.iter()
            .zip(axis_64.iter())
            .map(|(a, b)| a * b)
            .sum();

        let tangent_norm: f32 = tangent.iter().map(|x| x * x).sum::<f32>().sqrt();
        let axis_norm: f32 = axis_64.iter().map(|x| x * x).sum::<f32>().sqrt();

        if tangent_norm < 1e-10 || axis_norm < 1e-10 {
            return std::f32::consts::PI; // Degenerate case
        }

        let cos_angle = (dot / (tangent_norm * axis_norm)).clamp(-1.0, 1.0);
        cos_angle.acos()
    }
}
```

### 4.4 FAISS Index Configuration

```rust
/// Configuration for the FAISS GPU index.
///
/// Uses IVF (Inverted File) with PQ (Product Quantization) for
/// memory-efficient approximate nearest neighbor search.
pub struct FAISSIndexConfig {
    /// Number of clusters for IVF
    /// Constraint: nlist = 16384 for 10M+ vectors
    pub nlist: u32,

    /// Number of clusters to search
    /// Constraint: nprobe = 128 for recall/latency balance
    pub nprobe: u32,

    /// Number of sub-quantizers for PQ
    /// Constraint: pq_m = 64 for 1536D vectors
    pub pq_m: u32,

    /// Bits per quantization code
    /// Constraint: pq_bits = 8
    pub pq_bits: u32,

    /// Vector dimension
    /// Constraint: dimension = 1536
    pub dimension: u32,

    /// GPU device ID
    pub gpu_device: i32,

    /// Use float16 for memory efficiency
    pub use_float16: bool,
}

impl Default for FAISSIndexConfig {
    fn default() -> Self {
        Self {
            nlist: 16384,
            nprobe: 128,
            pq_m: 64,
            pq_bits: 8,
            dimension: 1536,
            gpu_device: 0,
            use_float16: true,
        }
    }
}
```

---

## 5. API Contracts

### 5.1 KnowledgeGraph Interface

```rust
/// Core knowledge graph operations trait.
///
/// Implementations must ensure thread-safety via interior mutability
/// (Arc<RwLock<T>>) for concurrent access.
#[async_trait]
pub trait KnowledgeGraphOps {
    // =========== Node Operations ===========

    /// Create a new knowledge node.
    ///
    /// # Arguments
    /// * `content` - Text content (max 65536 chars)
    /// * `embedding` - Pre-computed 1536D embedding
    /// * `importance` - Initial importance score [0,1]
    ///
    /// # Returns
    /// The created node's UUID
    ///
    /// # Errors
    /// * `ContentTooLong` - Content exceeds 65536 chars
    /// * `InvalidEmbedding` - Embedding dimension mismatch
    /// * `StorageError` - Database write failure
    ///
    /// Constraint: Operation_Latency < 50ms
    async fn create_node(
        &self,
        content: String,
        embedding: Vector1536,
        importance: f32,
    ) -> Result<Uuid, GraphError>;

    /// Retrieve a node by ID.
    ///
    /// # Arguments
    /// * `id` - Node UUID
    ///
    /// # Returns
    /// The node if found
    ///
    /// # Errors
    /// * `NodeNotFound` - No node with given ID
    ///
    /// Constraint: Operation_Latency < 5ms
    async fn get_node(&self, id: Uuid) -> Result<KnowledgeNode, GraphError>;

    /// Update a node's content and/or embedding.
    ///
    /// # Arguments
    /// * `id` - Node UUID
    /// * `content` - Optional new content
    /// * `embedding` - Optional new embedding
    ///
    /// # Errors
    /// * `NodeNotFound` - No node with given ID
    /// * `ContentTooLong` - Content exceeds limit
    ///
    /// Constraint: Operation_Latency < 50ms
    async fn update_node(
        &self,
        id: Uuid,
        content: Option<String>,
        embedding: Option<Vector1536>,
    ) -> Result<(), GraphError>;

    /// Delete a node and its edges (soft delete by default).
    ///
    /// # Arguments
    /// * `id` - Node UUID
    /// * `soft_delete` - If true, mark deleted but retain for recovery
    ///
    /// # Errors
    /// * `NodeNotFound` - No node with given ID
    ///
    /// Constraint: Operation_Latency < 20ms
    async fn delete_node(&self, id: Uuid, soft_delete: bool) -> Result<(), GraphError>;

    // =========== Edge Operations ===========

    /// Create an edge between two nodes.
    ///
    /// # Arguments
    /// * `source` - Source node UUID
    /// * `target` - Target node UUID
    /// * `edge_type` - Type of relationship
    /// * `weight` - Relationship strength [0,1]
    /// * `confidence` - Certainty [0,1]
    ///
    /// # Errors
    /// * `NodeNotFound` - Source or target not found
    /// * `SelfLoopNotAllowed` - Source equals target
    /// * `DuplicateEdge` - Edge already exists
    ///
    /// Constraint: Operation_Latency < 10ms
    async fn create_edge(
        &self,
        source: Uuid,
        target: Uuid,
        edge_type: EdgeType,
        weight: f32,
        confidence: f32,
    ) -> Result<(), GraphError>;

    /// Get all edges for a node.
    ///
    /// # Arguments
    /// * `node_id` - Node UUID
    /// * `direction` - Incoming, Outgoing, or Both
    /// * `edge_types` - Optional filter by edge types
    ///
    /// # Returns
    /// List of edges
    ///
    /// Constraint: Operation_Latency < 10ms
    async fn get_edges(
        &self,
        node_id: Uuid,
        direction: EdgeDirection,
        edge_types: Option<Vec<EdgeType>>,
    ) -> Result<Vec<GraphEdge>, GraphError>;

    /// Delete an edge.
    ///
    /// # Arguments
    /// * `source` - Source node UUID
    /// * `target` - Target node UUID
    /// * `edge_type` - Type of relationship
    ///
    /// # Errors
    /// * `EdgeNotFound` - No such edge
    ///
    /// Constraint: Operation_Latency < 5ms
    async fn delete_edge(
        &self,
        source: Uuid,
        target: Uuid,
        edge_type: EdgeType,
    ) -> Result<(), GraphError>;

    // =========== Search Operations ===========

    /// Perform semantic similarity search.
    ///
    /// # Arguments
    /// * `query` - Query embedding vector
    /// * `k` - Number of results to return
    /// * `filters` - Optional search filters
    ///
    /// # Returns
    /// Top-k similar nodes with distances
    ///
    /// Constraint: Operation_Latency < 10ms for k=100
    async fn semantic_search(
        &self,
        query: Vector1536,
        k: usize,
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>, GraphError>;

    /// Traverse graph from starting node.
    ///
    /// # Arguments
    /// * `start` - Starting node UUID
    /// * `depth` - Maximum traversal depth
    /// * `edge_types` - Optional edge type filter
    /// * `max_nodes` - Maximum nodes to return
    ///
    /// # Returns
    /// All reachable nodes within depth
    ///
    /// Constraint: Operation_Latency < 100ms for depth=6
    async fn graph_traverse(
        &self,
        start: Uuid,
        depth: u32,
        edge_types: Option<Vec<EdgeType>>,
        max_nodes: usize,
    ) -> Result<TraversalResult, GraphError>;

    /// Query entailment hierarchy.
    ///
    /// # Arguments
    /// * `node_id` - Node to query
    /// * `direction` - Ancestors or Descendants
    /// * `max_depth` - Maximum hierarchy depth
    ///
    /// # Returns
    /// Entailed concepts via cone containment
    ///
    /// Constraint: Operation_Latency < 1ms per containment check
    async fn entailment_query(
        &self,
        node_id: Uuid,
        direction: EntailmentDirection,
        max_depth: u32,
    ) -> Result<Vec<KnowledgeNode>, GraphError>;

    /// Detect contradictions for a node.
    ///
    /// # Arguments
    /// * `node_id` - Node to check
    /// * `threshold` - Contradiction confidence threshold
    ///
    /// # Returns
    /// List of contradicting nodes with confidence scores
    ///
    /// Constraint: Operation_Latency < 50ms
    async fn contradiction_detect(
        &self,
        node_id: Uuid,
        threshold: f32,
    ) -> Result<Vec<ContradictionResult>, GraphError>;

    // =========== Neurotransmitter Modulation (Marblestone) ===========

    /// Get effective edge weight with neurotransmitter modulation (Marblestone).
    ///
    /// Applies domain-specific excitatory/inhibitory balance.
    /// High excitatory + low inhibitory = strong activation.
    /// High inhibitory + low excitatory = suppression.
    ///
    /// # Formula
    /// modulation = excitatory - inhibitory + (modulatory * 0.5)
    /// effective_weight = base_weight * (1 + modulation), clamped to [0, 1]
    ///
    /// # Arguments
    /// * `edge` - The graph edge to modulate
    /// * `domain` - Current query domain context
    ///
    /// # Returns
    /// Modulated weight in [0, 1]
    ///
    /// Constraint: Pure function, no side effects
    fn get_modulated_weight(&self, edge: &GraphEdge, domain: &str) -> f32 {
        let nt = &edge.neurotransmitter_weights;
        let base = edge.weight;
        let modulation = nt.excitatory - nt.inhibitory + (nt.modulatory * 0.5);
        (base * (1.0 + modulation)).clamp(0.0, 1.0)
    }

    /// Domain-aware semantic search with neurotransmitter modulation (Marblestone).
    ///
    /// Performs semantic similarity search with edge weights modulated by
    /// neurotransmitter profiles based on the query domain. This enables
    /// context-sensitive knowledge retrieval where certain connections are
    /// strengthened or suppressed based on domain relevance.
    ///
    /// # Arguments
    /// * `query` - Query embedding vector
    /// * `k` - Number of results to return
    /// * `domain` - Domain context for weight modulation
    ///
    /// # Returns
    /// Top-k similar nodes with domain-modulated relevance scores
    ///
    /// # Errors
    /// * `InvalidDomain` - Unknown domain identifier
    /// * `StorageError` - Index query failure
    ///
    /// Constraint: Operation_Latency < 15ms for k=100
    async fn domain_aware_search(
        &self,
        query: Vector1536,
        k: usize,
        domain: &str,
    ) -> Result<Vec<SearchResult>, GraphError>;

    // =========== Index Management ===========

    /// Train the FAISS index on current vectors.
    ///
    /// Should be called after bulk insertion before searches.
    ///
    /// Constraint: Operation_Latency < 1 hour for 10M vectors
    async fn train_index(&self) -> Result<(), GraphError>;

    /// Persist index and graph to disk.
    async fn persist(&self) -> Result<(), GraphError>;

    /// Load index and graph from disk.
    async fn load(&mut self) -> Result<(), GraphError>;
}
```

### 5.2 Query Types

```rust
/// Search result with similarity score.
pub struct SearchResult {
    /// Found node
    pub node: KnowledgeNode,
    /// Cosine similarity score [0,1] (higher = more similar)
    pub similarity: f32,
    /// FAISS distance (L2 squared, lower = more similar)
    pub distance: f32,
}

/// Search filters for semantic search.
pub struct SearchFilters {
    /// Minimum importance threshold
    pub min_importance: Option<f32>,
    /// Filter by Johari quadrants
    pub johari_quadrants: Option<Vec<JohariQuadrant>>,
    /// Filter by creation date
    pub created_after: Option<DateTime<Utc>>,
    /// Filter by agent ID
    pub agent_id: Option<String>,
    /// Perspective lock for multi-agent isolation
    pub perspective_lock: Option<PerspectiveLock>,
}

/// Perspective lock for multi-agent safety.
pub struct PerspectiveLock {
    /// Allowed domain
    pub domain: Option<String>,
    /// Allowed agent IDs
    pub agent_ids: Option<Vec<String>>,
    /// Excluded agent IDs
    pub exclude_agent_ids: Option<Vec<String>>,
}

/// Result of graph traversal.
pub struct TraversalResult {
    /// Nodes found during traversal
    pub nodes: Vec<KnowledgeNode>,
    /// Edges traversed
    pub edges: Vec<GraphEdge>,
    /// Nodes per depth level
    pub depth_counts: Vec<usize>,
}

/// Direction for entailment queries.
pub enum EntailmentDirection {
    /// Find concepts that entail this one (ancestors/parents)
    Ancestors,
    /// Find concepts entailed by this one (descendants/children)
    Descendants,
}

/// Direction for edge queries.
pub enum EdgeDirection {
    /// Edges where node is source
    Outgoing,
    /// Edges where node is target
    Incoming,
    /// Both directions
    Both,
}

/// Result of contradiction detection.
pub struct ContradictionResult {
    /// Contradicting node
    pub node: KnowledgeNode,
    /// Type of contradiction
    pub contradiction_type: ContradictionType,
    /// Confidence score [0,1]
    pub confidence: f32,
}

/// Types of contradictions.
pub enum ContradictionType {
    /// Directly opposing content
    DirectOpposition,
    /// Logically inconsistent
    LogicalInconsistency,
    /// Temporal conflict (same event, different facts)
    TemporalConflict,
    /// Causal chain conflict
    CausalConflict,
}
```

### 5.3 Error Types

```rust
/// Errors from knowledge graph operations.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Node not found: {0}")]
    NodeNotFound(Uuid),

    #[error("Edge not found: {source} -> {target}")]
    EdgeNotFound { source: Uuid, target: Uuid },

    #[error("Content too long: {length} > {max}")]
    ContentTooLong { length: usize, max: usize },

    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidEmbedding { expected: usize, actual: usize },

    #[error("Self-loop not allowed")]
    SelfLoopNotAllowed,

    #[error("Duplicate edge")]
    DuplicateEdge,

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("FAISS error: {0}")]
    Faiss(String),

    #[error("Index not trained")]
    IndexNotTrained,

    #[error("Invalid hyperbolic point: norm >= 1.0")]
    InvalidHyperbolicPoint,

    #[error("Concurrent modification conflict")]
    ConcurrencyConflict,
}
```

---

## 6. Edge Cases

### 6.1 Index State

| ID | Scenario | Expected Behavior |
|----|----------|-------------------|
| EC-KG-001 | Search before index trained | Return GraphError::IndexNotTrained |
| EC-KG-002 | Add vector after training | Add to index incrementally (no rebuild) |
| EC-KG-003 | Remove last vector from cluster | Handle gracefully, don't corrupt index |
| EC-KG-004 | Search with k > total vectors | Return all available vectors |

### 6.2 Graph Operations

| ID | Scenario | Expected Behavior |
|----|----------|-------------------|
| EC-KG-010 | Create edge to non-existent node | Return GraphError::NodeNotFound |
| EC-KG-011 | Create self-loop edge | Return GraphError::SelfLoopNotAllowed |
| EC-KG-012 | Delete node with edges | Delete all connected edges atomically |
| EC-KG-013 | Traverse with no outgoing edges | Return empty result, not error |
| EC-KG-014 | Circular graph traversal | Track visited nodes, don't infinite loop |

### 6.3 Hyperbolic Operations

| ID | Scenario | Expected Behavior |
|----|----------|-------------------|
| EC-KG-020 | Point at Poincare ball boundary | Clamp to norm < 0.999 |
| EC-KG-021 | Zero-length tangent vector | Return pi for angle (outside cone) |
| EC-KG-022 | Degenerate cone (aperture=0) | Only apex itself contained |
| EC-KG-023 | Distance between identical points | Return 0.0 |

### 6.4 Concurrent Access

| ID | Scenario | Expected Behavior |
|----|----------|-------------------|
| EC-KG-030 | Concurrent reads | Allow multiple readers |
| EC-KG-031 | Read during write | Read waits for write to complete |
| EC-KG-032 | Concurrent writes to same node | Serialize writes, no data corruption |
| EC-KG-033 | Index training during search | Queue searches until training complete |

---

## 7. Error States

| ID | HTTP Equivalent | Condition | Message | Recovery |
|----|-----------------|-----------|---------|----------|
| ERR-KG-001 | 404 | Node ID not in graph | "Node not found: {id}" | Verify ID, retry |
| ERR-KG-002 | 404 | Edge not in graph | "Edge not found: {source} -> {target}" | Verify IDs |
| ERR-KG-003 | 400 | Content exceeds 65536 | "Content too long: {len} > 65536" | Truncate content |
| ERR-KG-004 | 400 | Embedding not 1536D | "Invalid embedding dimension" | Fix embedding pipeline |
| ERR-KG-005 | 400 | Self-loop attempt | "Self-loop not allowed" | Use different target |
| ERR-KG-006 | 409 | Duplicate edge | "Duplicate edge" | Update existing edge |
| ERR-KG-007 | 500 | Storage failure | "Storage error: {details}" | Retry with backoff |
| ERR-KG-008 | 500 | FAISS error | "FAISS error: {details}" | Check GPU, restart |
| ERR-KG-009 | 412 | Index not trained | "Index not trained" | Call train_index() |
| ERR-KG-010 | 400 | Point outside ball | "Invalid hyperbolic point" | Normalize coordinates |
| ERR-KG-011 | 409 | Concurrent conflict | "Concurrent modification" | Retry operation |

---

## 8. Test Plan

### 8.1 Unit Tests

| ID | Test Case | Inputs | Expected Output | Implements |
|----|-----------|--------|-----------------|------------|
| TC-KG-001 | Create node with valid data | content, embedding, importance=0.5 | Node created with UUID | REQ-KG-020-031 |
| TC-KG-002 | Create node with oversized content | 70000 char content | ContentTooLong error | REQ-KG-021 |
| TC-KG-003 | Hyperbolic distance calculation | Two points in ball | Correct distance | REQ-KG-051 |
| TC-KG-004 | Hyperbolic distance at boundary | Points near ||x||=1 | No NaN/Inf | REQ-KG-050 |
| TC-KG-005 | EntailmentCone.contains() inside | Point within cone | true | REQ-KG-052-053 |
| TC-KG-006 | EntailmentCone.contains() outside | Point outside cone | false | REQ-KG-052-053 |
| TC-KG-007 | Create edge valid | source, target, type, weight, confidence | Edge created | REQ-KG-040-044 |
| TC-KG-008 | Create self-loop | source=target | SelfLoopNotAllowed error | EC-KG-011 |
| TC-KG-009 | Edge bidirectional index | Create edge | Indexed both directions | REQ-KG-013 |

### 8.2 Integration Tests

| ID | Test Case | Setup | Action | Verification | Implements |
|----|-----------|-------|--------|--------------|------------|
| TC-KG-101 | FAISS GPU search | 1M vectors in index | Search k=100 | <2ms latency, correct recall | REQ-KG-001-005, NFR-KG-001 |
| TC-KG-102 | Incremental add | Trained index | Add 1000 vectors | Searchable without rebuild | REQ-KG-006 |
| TC-KG-103 | Graph traverse depth 6 | Dense graph 10K nodes | Traverse depth=6 | <100ms, correct nodes | NFR-KG-003 |
| TC-KG-104 | Entailment query | Hierarchy with cones | Query ancestors | O(1) per check | NFR-KG-004 |
| TC-KG-105 | Contradiction detect | Nodes with conflicts | Detect contradictions | All conflicts found | REQ-KG-063 |
| TC-KG-106 | Persistence round-trip | Full graph | Persist and reload | Data integrity | REQ-KG-008 |
| TC-KG-107 | Concurrent read/write | Active readers/writers | Mixed operations | No deadlocks, data correct | NFR-KG-009 |

### 8.3 Performance Benchmarks

| ID | Benchmark | Parameters | Target | Method |
|----|-----------|------------|--------|--------|
| BM-KG-001 | FAISS search latency | 1M vectors, k=100 | P95 <2ms | GPU profiling |
| BM-KG-002 | Full semantic search | 1M vectors, k=100 | P95 <10ms | End-to-end timing |
| BM-KG-003 | Graph traversal | 10K nodes, depth=6 | P95 <100ms | Stress test |
| BM-KG-004 | Cone containment | 100K checks | P95 <1ms each | Batch timing |
| BM-KG-005 | Node creation | Batch 1000 | <50ms per node | Throughput test |
| BM-KG-006 | Index memory | 10M vectors | <24GB GPU | Memory profiling |

### 8.4 Critical Test Requirements

**NO MOCK FAISS**: All tests MUST use real FAISS GPU index. Mock implementations are forbidden for:
- Vector similarity search
- Index training
- Incremental updates

**Real GPU Required**: Tests marked `#[requires_gpu]` must run on actual CUDA hardware.

---

## 9. Implementation Notes

### 9.1 FAISS Index Setup

```rust
// Example configuration for production
let config = FAISSIndexConfig {
    nlist: 16384,           // sqrt(10M) * 16 for optimal clustering
    nprobe: 128,            // ~1% of clusters for recall/latency balance
    pq_m: 64,               // 1536 / 64 = 24 dimensions per sub-quantizer
    pq_bits: 8,             // 256 centroids per sub-quantizer
    dimension: 1536,
    gpu_device: 0,
    use_float16: true,
};

// Memory estimate: 1536D * 8 bits/64 = 192 bits = 24 bytes per vector
// 10M vectors = 240MB + cluster overhead ~300MB total
```

### 9.2 RocksDB Schema

```
Key Prefix | Value | Purpose
-----------+-------+---------
node:{uuid} | serialized KnowledgeNode | Node storage
edge:{source}:{target}:{type} | serialized GraphEdge | Edge storage (forward)
redge:{target}:{source}:{type} | serialized GraphEdge | Edge storage (reverse)
cone:{uuid} | serialized EntailmentCone | Cone storage
hyperbolic:{uuid} | serialized HyperbolicPoint | Position storage
meta:node_count | u64 | Statistics
meta:edge_count | u64 | Statistics
```

### 9.3 Concurrency Model

```rust
pub struct ConcurrentKnowledgeGraph {
    /// Node and edge storage
    inner: Arc<RwLock<RocksDBStore>>,

    /// FAISS GPU index
    faiss_index: Arc<RwLock<FaissGpuIndex>>,

    /// Lock ordering: inner -> faiss_index (prevents deadlocks)
}
```

### 9.4 Performance Optimizations

1. **Batch operations**: Accumulate writes for bulk commit
2. **Index caching**: Keep hot clusters in GPU memory
3. **Lazy cone computation**: Compute entailment cones on demand
4. **Edge prefetching**: Prefetch edges during traversal
5. **Bloom filters**: Quick existence checks before disk reads

---

## 10. Security Considerations

| Concern | Mitigation |
|---------|------------|
| Injection via content | Sanitize before storage |
| Embedding attacks | Validate dimension and range |
| Denial of service | Rate limit large traversals |
| Data exfiltration | Perspective lock enforcement |
| Memory exhaustion | GPU memory pooling with limits |

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Module | Dependency | Interface |
|--------|------------|-----------|
| Module 3 | Embedding Pipeline | Vector1536 type |
| Module 3 | FuseMoE | Fused embeddings |
| Module 2 | Core Types | UUID, DateTime |
| Module 2 | Storage | RocksDB abstraction |

### 11.2 External Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| faiss | 0.12+ | GPU vector search |
| rocksdb | 0.21+ | Graph storage |
| cudarc | 0.10+ | CUDA bindings |
| uuid | 1.6+ | Node identification |
| chrono | 0.4+ | Timestamps |
| serde | 1.0+ | Serialization |
| tokio | 1.35+ | Async runtime |

---

## 12. Glossary

| Term | Definition |
|------|------------|
| FAISS | Facebook AI Similarity Search - GPU-accelerated vector search library |
| IVF | Inverted File Index - clustering-based approximate search |
| PQ | Product Quantization - vector compression technique |
| Poincare Ball | Model of hyperbolic geometry where all points have norm < 1 |
| Entailment Cone | Geometric representation of IS-A hierarchies in hyperbolic space |
| UTL | Unified Theory of Learning - the core learning equation |
| Johari Quadrant | Classification of knowledge by self/other awareness |

---

## 13. Appendix: Mathematical Formulas

### 13.1 Hyperbolic Distance (Poincare Ball)

```
d(x, y) = arcosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2) * (1 - ||y||^2)))

Where:
- x, y are points in the Poincare ball (||x|| < 1, ||y|| < 1)
- ||.|| denotes Euclidean norm
- arcosh(z) = ln(z + sqrt(z^2 - 1))
```

### 13.2 Entailment Cone Containment

```
A concept x is entailed by concept y (x IS-A y) if:
  angle(log_apex(x), axis) <= aperture

Where:
- apex is the cone apex in Poincare ball
- axis is the cone direction (unit vector)
- aperture is the half-angle of the cone
- log_apex(x) is the logarithmic map from apex to x
```

### 13.3 Cosine Similarity

```
sim(x, y) = (x . y) / (||x|| * ||y||)

Where:
- x, y are embedding vectors
- . denotes dot product
- Result in [-1, 1], typically [0, 1] for embeddings
```

---

## TASK COMPLETION SUMMARY

**What I Did**: Created comprehensive functional specification for Module 4: Knowledge Graph covering FAISS GPU index configuration, graph storage with RocksDB, KnowledgeNode and GraphEdge data models, Poincare ball hyperbolic coordinates with exact distance formula, EntailmentCone struct with apex/aperture/axis fields for O(1) hierarchy queries, query operations (semantic_search, graph_traverse, entailment_query, contradiction_detect), performance requirements, and test plan that mandates real FAISS GPU tests (no mocks).

**Files Created/Modified**:
- Created: `/home/cabdru/contextgraph/specs/functional/module-04-knowledge-graph.md`

**Memory Locations**: `specs/functional/module-04-knowledge-graph`

**Next Agent Guidance (Module 5 UTL Integration needs)**:
- Module 5 should reference the `KnowledgeNode.utl_state` struct (delta_s, delta_c, w_e, phi) defined here
- Module 5 implements the UTL formula: `L = f((delta_S * delta_C) * w_e * cos(phi))`
- Module 5 will update `KnowledgeNode.importance` based on learning signals
- Module 5 will update `KnowledgeNode.johari_quadrant` based on entropy/coherence state
- Module 5 should use the `semantic_search` and `graph_traverse` operations for context retrieval
- Module 5 needs to integrate with the hyperbolic coordinates for coherence measurements
- The `Vector1536` type from Module 3 is used consistently here