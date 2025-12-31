# Module 4: Knowledge Graph - Atomic Tasks

```yaml
metadata:
  module_id: "module-04"
  module_name: "Knowledge Graph"
  version: "1.0.0"
  phase: 2
  total_tasks: 25
  approach: "inside-out-bottom-up"
  created: "2025-12-31"
  dependencies:
    - module-02-core-infrastructure
    - module-03-embedding-pipeline
  estimated_duration: "4 weeks"
  spec_refs:
    - SPEC-GRAPH-004 (Functional)
    - TECH-GRAPH-004 (Technical)
```

---

## Task Overview

This module implements the Knowledge Graph layer combining FAISS GPU-accelerated vector similarity search with hyperbolic geometry for hierarchical reasoning. Tasks are organized in inside-out, bottom-up order:

1. **Foundation Layer** (Tasks 1-8): Core types - HyperbolicPoint, EntailmentCone, IndexConfig
2. **Logic Layer** (Tasks 9-17): FAISS FFI, RocksDB graph storage, Poincare operations
3. **Surface Layer** (Tasks 18-25): Query operations, Marblestone integration, traversal

---

## Foundation Layer: Core Types

```yaml
tasks:
  # ============================================================
  # FOUNDATION: Configuration Types
  # ============================================================

  - id: "M04-T01"
    title: "Define IndexConfig for FAISS IVF-PQ"
    description: |
      Implement IndexConfig struct for FAISS GPU index configuration.
      Fields: dimension (1536), nlist (16384), nprobe (128), pq_segments (64),
      pq_bits (8), gpu_id (0), use_float16 (true), min_train_vectors (4_194_304).
      Include factory_string() method returning "IVF{nlist},PQ{pq_segments}x{pq_bits}".
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-graph/src/config.rs"
    dependencies: []
    acceptance_criteria:
      - "IndexConfig struct compiles with all 8 fields"
      - "Default returns nlist=16384, nprobe=128, pq_segments=64, pq_bits=8"
      - "factory_string() returns 'IVF16384,PQ64x8' for defaults"
      - "min_train_vectors = 256 * nlist = 4,194,304"
      - "Serde Serialize/Deserialize implemented"
    test_file: "crates/context-graph-graph/tests/config_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 2"
      - "REQ-KG-001 through REQ-KG-005"

  - id: "M04-T02"
    title: "Define HyperbolicConfig for Poincare Ball"
    description: |
      Implement HyperbolicConfig struct for 64D Poincare ball model.
      Fields: dim (64), curvature (-1.0), eps (1e-7), max_norm (1.0 - 1e-5).
      Include validation method ensuring curvature < 0.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1
    file_path: "crates/context-graph-graph/src/config.rs"
    dependencies: []
    acceptance_criteria:
      - "HyperbolicConfig struct with 4 fields"
      - "Default returns dim=64, curvature=-1.0"
      - "max_norm ensures points stay within ball boundary"
      - "eps prevents numerical instability"
    test_file: "crates/context-graph-graph/tests/config_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 5"
      - "REQ-KG-050, REQ-KG-054"

  - id: "M04-T03"
    title: "Define ConeConfig for Entailment Cones"
    description: |
      Implement ConeConfig struct for EntailmentCone parameters.
      Fields: min_aperture (0.1 rad), max_aperture (1.5 rad), base_aperture (1.0 rad),
      aperture_decay (0.85 per level), membership_threshold (0.7).
      Include compute_aperture(depth) method.
    layer: "foundation"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-graph/src/config.rs"
    dependencies: []
    acceptance_criteria:
      - "ConeConfig struct with 5 fields"
      - "compute_aperture(0) returns base_aperture"
      - "compute_aperture(n) = base * decay^n, clamped to [min, max]"
      - "Default values match spec"
    test_file: "crates/context-graph-graph/tests/config_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 6"
      - "REQ-KG-052"

  # ============================================================
  # FOUNDATION: Hyperbolic Geometry Types
  # ============================================================

  - id: "M04-T04"
    title: "Define PoincarePoint for 64D Hyperbolic Space"
    description: |
      Implement PoincarePoint struct with coords: [f32; 64].
      Constraint: ||coords|| < 1.0 (strict inequality for Poincare ball).
      Include methods: origin(), norm_squared(), norm(), project(&HyperbolicConfig).
      Use #[repr(C, align(64))] for SIMD optimization.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-graph/src/hyperbolic/poincare.rs"
    dependencies:
      - "M04-T02"
    acceptance_criteria:
      - "PoincarePoint struct with [f32; 64] array"
      - "origin() returns all zeros"
      - "norm() computes Euclidean norm"
      - "project() rescales if norm >= max_norm"
      - "Memory alignment 64 bytes for cache efficiency"
      - "Clone, Debug traits implemented"
    test_file: "crates/context-graph-graph/tests/poincare_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 5.1"
      - "REQ-KG-050"

  - id: "M04-T05"
    title: "Implement PoincareBall Mobius Operations"
    description: |
      Implement PoincareBall struct with Mobius algebra operations.
      Methods: mobius_add(x, y), distance(x, y), exp_map(x, v), log_map(x, y).
      Distance formula: d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c * ||x-y||^2 / ((1-c||x||^2)(1-c||y||^2))))
      Performance target: <10us per distance computation.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-graph/src/hyperbolic/mobius.rs"
    dependencies:
      - "M04-T04"
    acceptance_criteria:
      - "mobius_add() implements Mobius addition formula correctly"
      - "distance() returns Poincare ball distance in <10us"
      - "exp_map() maps tangent vector to point on manifold"
      - "log_map() returns tangent vector from x to y"
      - "All operations handle boundary cases (norm near 1.0)"
      - "Unit tests verify mathematical properties (symmetry, identity)"
    test_file: "crates/context-graph-graph/tests/mobius_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 5.2"
      - "REQ-KG-051"

  - id: "M04-T06"
    title: "Define EntailmentCone Struct"
    description: |
      Implement EntailmentCone struct for O(1) IS-A hierarchy queries.
      Fields: apex (PoincarePoint), aperture (f32), aperture_factor (f32), depth (u32).
      Include methods: new(), effective_aperture(), contains(), membership_score().
      Constraint: aperture in [0, pi/2], aperture_factor in [0.5, 2.0].
    layer: "foundation"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-graph/src/entailment/cones.rs"
    dependencies:
      - "M04-T03"
      - "M04-T05"
    acceptance_criteria:
      - "EntailmentCone struct with apex, aperture, aperture_factor, depth"
      - "new() computes aperture from depth using ConeConfig"
      - "effective_aperture() = aperture * aperture_factor"
      - "contains() returns bool in <50us"
      - "membership_score() returns soft [0,1] score"
      - "Serde serialization produces 268 bytes (256 coords + 4 aperture + 4 factor + 4 depth)"
    test_file: "crates/context-graph-graph/tests/entailment_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 6"
      - "REQ-KG-052, REQ-KG-053"

  - id: "M04-T07"
    title: "Implement EntailmentCone Containment Logic"
    description: |
      Implement EntailmentCone containment check algorithm.
      Algorithm:
      1. Compute tangent = log_map(apex, point)
      2. Compute to_origin = log_map(apex, origin)
      3. angle = arccos(dot(tangent, to_origin) / (||tangent|| * ||to_origin||))
      4. Return angle <= effective_aperture()
      Include update_aperture() for training.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-graph/src/entailment/cones.rs"
    dependencies:
      - "M04-T06"
    acceptance_criteria:
      - "contains() returns true for points within cone"
      - "contains() returns false for points outside cone"
      - "membership_score() returns 1.0 for contained, exp(-2*(angle-aperture)) otherwise"
      - "update_aperture() adjusts aperture_factor based on training signal"
      - "Edge cases handled: apex at origin, point at apex, degenerate cones"
      - "Performance: <50us per containment check"
    test_file: "crates/context-graph-graph/tests/entailment_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 6"
      - "REQ-KG-053"

  - id: "M04-T08"
    title: "Define GraphError Enum"
    description: |
      Implement comprehensive GraphError enum for knowledge graph operations.
      Variants: FaissIndexCreation, FaissTrainingFailed, FaissSearchFailed, FaissAddFailed,
      IndexNotTrained, InsufficientTrainingData, GpuResourceAllocation, GpuTransferFailed,
      StorageOpen, Storage, ColumnFamilyNotFound, CorruptedData, VectorIdMismatch, InvalidConfig,
      NodeNotFound, EdgeNotFound, InvalidHyperbolicPoint.
      Use thiserror for derivation.
    layer: "foundation"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-graph/src/error.rs"
    dependencies: []
    acceptance_criteria:
      - "GraphError enum with 17+ variants"
      - "All variants have descriptive #[error()] messages"
      - "From<rocksdb::Error> implemented"
      - "Error is Send + Sync"
      - "InsufficientTrainingData includes provided/required counts"
    test_file: "crates/context-graph-graph/tests/error_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 9"

  # ============================================================
  # LOGIC LAYER: FAISS GPU Index
  # ============================================================

  - id: "M04-T09"
    title: "Define FAISS FFI Bindings"
    description: |
      Implement faiss_ffi module with C bindings to FAISS library.
      Bindings: faiss_index_factory, faiss_StandardGpuResources_new/free,
      faiss_index_cpu_to_gpu, faiss_Index_train, faiss_Index_is_trained,
      faiss_Index_add_with_ids, faiss_Index_search, faiss_IndexIVF_nprobe_set,
      faiss_Index_ntotal, faiss_write_index, faiss_read_index, faiss_Index_free.
      Include GpuResources RAII wrapper with Send + Sync.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-graph/src/index/faiss_ffi.rs"
    dependencies:
      - "M04-T08"
    acceptance_criteria:
      - "All extern 'C' declarations compile"
      - "GpuResources wrapper handles allocation/deallocation"
      - "GpuResources is Send + Sync"
      - "MetricType enum with InnerProduct=0, L2=1"
      - "Link directive: #[link(name = 'faiss_c')]"
    test_file: "crates/context-graph-graph/tests/faiss_ffi_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 3.1"

  - id: "M04-T10"
    title: "Implement FaissGpuIndex Wrapper"
    description: |
      Implement FaissGpuIndex struct wrapping FAISS GPU index.
      Methods: new(config), train(vectors), search(queries, k), add_with_ids(vectors, ids),
      ntotal(), save(path), load(path).
      Use NonNull for GPU pointer, Arc<GpuResources> for resource sharing.
      Performance: <5ms for k=10 search on 10M vectors.
    layer: "logic"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-graph/src/index/gpu_index.rs"
    dependencies:
      - "M04-T01"
      - "M04-T09"
    acceptance_criteria:
      - "new() creates IVF-PQ index and transfers to GPU"
      - "train() requires min_train_vectors, sets nprobe after training"
      - "search() returns SearchResult with ids and distances"
      - "add_with_ids() adds vectors incrementally (no rebuild)"
      - "Drop impl frees GPU resources correctly"
      - "Send + Sync implemented (unsafe)"
    test_file: "crates/context-graph-graph/tests/gpu_index_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 3.2"
      - "REQ-KG-001 through REQ-KG-008"

  - id: "M04-T11"
    title: "Implement SearchResult Struct"
    description: |
      Implement SearchResult struct for FAISS query results.
      Fields: ids (Vec<i64>), distances (Vec<f32>), k (usize), num_queries (usize).
      Include query_results(idx) iterator method for extracting per-query results.
      Handle -1 sentinel IDs (no match found).
    layer: "logic"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-graph/src/index/gpu_index.rs"
    dependencies:
      - "M04-T10"
    acceptance_criteria:
      - "SearchResult struct with 4 fields"
      - "query_results(idx) returns iterator of (id, distance) pairs"
      - "Filters out -1 sentinel values"
      - "Clone, Debug implemented"
    test_file: "crates/context-graph-graph/tests/search_result_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 3.2"

  # ============================================================
  # LOGIC LAYER: RocksDB Graph Storage
  # ============================================================

  - id: "M04-T12"
    title: "Define Graph Storage Column Families"
    description: |
      Define RocksDB column families for knowledge graph storage.
      CFs: adjacency (edge lists), hyperbolic (64D coordinates), entailment_cones (cone data).
      Include get_column_family_descriptors() returning optimized CF options.
      Hyperbolic CF: 256 bytes per point (64 * 4), LZ4 compression.
      Cones CF: 268 bytes per cone, bloom filter enabled.
    layer: "logic"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-graph/src/storage/mod.rs"
    dependencies:
      - "M04-T08"
    acceptance_criteria:
      - "CF_ADJACENCY, CF_HYPERBOLIC, CF_CONES constants defined"
      - "get_column_family_descriptors() returns 3 CFs with options"
      - "Hyperbolic CF optimized for point lookups"
      - "Adjacency CF optimized for prefix scans"
    test_file: "crates/context-graph-graph/tests/storage_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 4"

  - id: "M04-T13"
    title: "Implement GraphStorage Backend"
    description: |
      Implement GraphStorage struct wrapping RocksDB for graph data.
      Methods: open(path, config), get_hyperbolic(node_id), put_hyperbolic(node_id, point),
      get_cone(node_id), put_cone(node_id, cone), get_adjacency(node_id), put_adjacency(node_id, edges).
      Use Arc<DB> for thread-safe sharing.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-graph/src/storage/rocksdb.rs"
    dependencies:
      - "M04-T04"
      - "M04-T06"
      - "M04-T12"
    acceptance_criteria:
      - "open() creates DB with all 3 CFs"
      - "get_hyperbolic() deserializes 256 bytes to PoincarePoint"
      - "put_hyperbolic() serializes point to 256 bytes"
      - "get_cone() deserializes 268 bytes to EntailmentCone"
      - "Proper error handling with GraphError variants"
    test_file: "crates/context-graph-graph/tests/storage_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 4.2"

  - id: "M04-T14"
    title: "Implement NeurotransmitterWeights for Edges (Marblestone)"
    description: |
      Implement NeurotransmitterWeights struct from Module 2 in graph context.
      Fields: excitatory (f32), inhibitory (f32), modulatory (f32), all in [0,1].
      Include for_domain(Domain) factory with domain-specific profiles.
      Include net_activation() computing excitatory - inhibitory + (modulatory * 0.5).
    layer: "logic"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-graph/src/storage/edges.rs"
    dependencies: []
    acceptance_criteria:
      - "NeurotransmitterWeights struct with 3 f32 fields"
      - "Default: excitatory=0.5, inhibitory=0.5, modulatory=0.0"
      - "for_domain(Code) = {0.7, 0.3, 0.2}"
      - "for_domain(Creative) = {0.8, 0.2, 0.5}"
      - "net_activation() formula correct"
      - "Serde serialization works"
    test_file: "crates/context-graph-graph/tests/marblestone_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 4.1"
      - "REQ-KG-065"

  - id: "M04-T15"
    title: "Implement GraphEdge with Marblestone Fields"
    description: |
      Implement GraphEdge struct with full Marblestone support.
      Fields: id, source, target, edge_type, weight, confidence, domain,
      neurotransmitter_weights, is_amortized_shortcut, steering_reward,
      traversal_count, created_at, last_traversed_at.
      Include get_modulated_weight(query_domain) method.
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-graph/src/storage/edges.rs"
    dependencies:
      - "M04-T14"
    acceptance_criteria:
      - "GraphEdge struct with all 13 fields"
      - "new() initializes with domain-appropriate NT weights"
      - "get_modulated_weight() applies formula: weight * (1 + net_activation + domain_bonus) * steering_factor"
      - "record_traversal() increments count and updates steering_reward with EMA"
      - "EdgeType enum: Semantic, Temporal, Causal, Hierarchical"
      - "Domain enum: Code, Legal, Medical, Creative, Research, General"
    test_file: "crates/context-graph-graph/tests/edge_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 4.1"
      - "REQ-KG-040 through REQ-KG-044, REQ-KG-065"

  # ============================================================
  # LOGIC LAYER: Graph Traversal
  # ============================================================

  - id: "M04-T16"
    title: "Implement BFS Graph Traversal"
    description: |
      Implement bfs_traverse(storage, start, params) function.
      BfsParams: max_depth (6), max_nodes (10000), edge_types (Option<Vec>), domain_filter.
      Returns BfsResult with nodes, edges, depth_counts.
      Use VecDeque for frontier, HashSet for visited tracking.
      Performance: <100ms for depth=6 on 10M node graph.
    layer: "logic"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-graph/src/traversal/bfs.rs"
    dependencies:
      - "M04-T13"
      - "M04-T15"
    acceptance_criteria:
      - "bfs_traverse() visits nodes level by level"
      - "Respects max_depth and max_nodes limits"
      - "edge_types filter restricts which edges to follow"
      - "domain_filter applies Marblestone domain matching"
      - "No infinite loops on cyclic graphs (visited set)"
      - "depth_counts tracks nodes found at each depth"
    test_file: "crates/context-graph-graph/tests/traversal_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 7.1"
      - "REQ-KG-061"

  - id: "M04-T17"
    title: "Implement DFS Graph Traversal"
    description: |
      Implement dfs_traverse(storage, start, max_depth, max_nodes) function.
      Use iterative stack-based approach (not recursive).
      Returns Vec<NodeId> of visited nodes in DFS order.
      Handle cycles via visited set.
    layer: "logic"
    priority: "medium"
    estimated_hours: 2
    file_path: "crates/context-graph-graph/src/traversal/dfs.rs"
    dependencies:
      - "M04-T13"
    acceptance_criteria:
      - "dfs_traverse() visits nodes in depth-first order"
      - "Uses iterative stack, not recursion"
      - "Respects max_depth and max_nodes"
      - "No stack overflow on deep graphs"
    test_file: "crates/context-graph-graph/tests/traversal_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 7.2"

  # ============================================================
  # SURFACE LAYER: Query Operations
  # ============================================================

  - id: "M04-T18"
    title: "Implement Semantic Search Operation"
    description: |
      Implement semantic_search(query, k, filters) on KnowledgeGraph.
      Uses FAISS GPU index for initial k-NN retrieval.
      Applies SearchFilters: min_importance, johari_quadrants, created_after, agent_id.
      Returns Vec<SearchResult> with node, similarity, distance.
      Performance: <10ms for k=100 on 10M vectors.
    layer: "surface"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-graph/src/lib.rs"
    dependencies:
      - "M04-T10"
      - "M04-T11"
    acceptance_criteria:
      - "semantic_search() calls FAISS search internally"
      - "Converts L2 distance to cosine similarity"
      - "Applies post-filters to results"
      - "Returns empty vec if index not trained"
      - "Performance meets <10ms target"
    test_file: "crates/context-graph-graph/tests/search_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 8"
      - "REQ-KG-060"

  - id: "M04-T19"
    title: "Implement Domain-Aware Search (Marblestone)"
    description: |
      Implement domain_aware_search(query, domain, k) with neurotransmitter modulation.
      Algorithm:
      1. FAISS k-NN search fetching 3x candidates
      2. Apply NeurotransmitterWeights modulation per domain
      3. Re-rank by modulated score
      4. Return top-k results
      Performance: <10ms for k=10 on 10M vectors.
    layer: "surface"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-graph/src/marblestone/domain_search.rs"
    dependencies:
      - "M04-T14"
      - "M04-T15"
      - "M04-T18"
    acceptance_criteria:
      - "domain_aware_search() over-fetches candidates (3x)"
      - "Applies NT modulation: sim * (1 + net_activation)"
      - "Re-ranks results by modulated score"
      - "Truncates to requested k"
      - "DomainSearchResult includes base_distance and modulated_score"
    test_file: "crates/context-graph-graph/tests/domain_search_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 8"
      - "REQ-KG-065"

  - id: "M04-T20"
    title: "Implement Entailment Query Operation"
    description: |
      Implement entailment_query(node_id, direction, max_depth) function.
      Uses EntailmentCone containment for O(1) IS-A hierarchy checks.
      Direction: Ancestors (concepts that entail this) or Descendants (concepts entailed by this).
      Returns Vec<KnowledgeNode> in hierarchy.
      Performance: <1ms per containment check.
    layer: "surface"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-graph/src/lib.rs"
    dependencies:
      - "M04-T07"
      - "M04-T13"
    acceptance_criteria:
      - "entailment_query() retrieves node's cone from storage"
      - "Checks containment against candidate nodes"
      - "Ancestors: finds cones that contain this node"
      - "Descendants: finds nodes contained by this cone"
      - "Respects max_depth limit"
      - "Performance: <1ms per cone check"
    test_file: "crates/context-graph-graph/tests/entailment_query_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 8"
      - "REQ-KG-062"

  - id: "M04-T21"
    title: "Implement Contradiction Detection"
    description: |
      Implement contradiction_detect(node_id, threshold) function.
      Algorithm:
      1. Semantic search for similar nodes (k=50)
      2. Check for CONTRADICTS edges
      3. Compute contradiction confidence based on similarity + edge weight
      4. Return ContradictionResult with node, contradiction_type, confidence.
      ContradictionType: DirectOpposition, LogicalInconsistency, TemporalConflict, CausalConflict.
    layer: "surface"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-graph/src/lib.rs"
    dependencies:
      - "M04-T18"
      - "M04-T16"
    acceptance_criteria:
      - "contradiction_detect() finds semantically similar nodes"
      - "Checks for explicit CONTRADICTS edge type"
      - "Computes confidence score in [0,1]"
      - "Filters by threshold"
      - "Classifies contradiction type"
    test_file: "crates/context-graph-graph/tests/contradiction_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 8"
      - "REQ-KG-063"

  - id: "M04-T22"
    title: "Implement get_modulated_weight Function (Marblestone)"
    description: |
      Implement get_modulated_weight(edge, domain) standalone function.
      Formula: modulation = excitatory - inhibitory + (modulatory * 0.5)
               effective_weight = base_weight * (1 + modulation), clamped to [0, 1]
      Pure function with no side effects.
      Used by traversal and search operations for domain-aware edge weighting.
    layer: "surface"
    priority: "high"
    estimated_hours: 1
    file_path: "crates/context-graph-graph/src/marblestone/mod.rs"
    dependencies:
      - "M04-T15"
    acceptance_criteria:
      - "get_modulated_weight() is pure function"
      - "Applies NT modulation formula correctly"
      - "Result clamped to [0.0, 1.0]"
      - "Domain match adds bonus"
      - "Unit tests verify edge cases"
    test_file: "crates/context-graph-graph/tests/marblestone_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 8"
      - "REQ-KG-065"

  # ============================================================
  # SURFACE LAYER: CUDA Kernels & Integration
  # ============================================================

  - id: "M04-T23"
    title: "Implement Poincare Distance CUDA Kernel"
    description: |
      Implement poincare_distance_batch CUDA kernel for GPU-accelerated hyperbolic distance.
      Input: queries[n_q][64], database[n_db][64], curvature c
      Output: distances[n_q][n_db]
      Use shared memory for query caching.
      Performance: <1ms for 1K x 1K distance matrix.
    layer: "surface"
    priority: "high"
    estimated_hours: 4
    file_path: "crates/context-graph-graph/kernels/poincare_distance.cu"
    dependencies:
      - "M04-T05"
    acceptance_criteria:
      - "CUDA kernel compiles with nvcc"
      - "Shared memory used for query vectors"
      - "Matches CPU implementation within 1e-5 tolerance"
      - "Performance: <1ms for 1K x 1K"
      - "Handles boundary cases (points near norm=1)"
    test_file: "crates/context-graph-graph/tests/cuda_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 10.1"

  - id: "M04-T24"
    title: "Implement Cone Membership CUDA Kernel"
    description: |
      Implement cone_check_batch CUDA kernel for batch entailment cone membership.
      Input: cones[n_cones][65] (64 apex coords + 1 aperture), points[n_pts][64]
      Output: scores[n_cones][n_pts]
      Performance: <2ms for 1K x 1K membership matrix.
    layer: "surface"
    priority: "high"
    estimated_hours: 4
    file_path: "crates/context-graph-graph/kernels/cone_check.cu"
    dependencies:
      - "M04-T07"
    acceptance_criteria:
      - "CUDA kernel compiles with nvcc"
      - "Shared memory used for cone data"
      - "Matches CPU implementation within 1e-5 tolerance"
      - "Performance: <2ms for 1K x 1K"
      - "Returns soft membership score [0,1]"
    test_file: "crates/context-graph-graph/tests/cuda_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 10.2"

  - id: "M04-T25"
    title: "Create Module Integration Tests"
    description: |
      Implement comprehensive integration tests for Module 4:
      - End-to-end FAISS index lifecycle (train, add, search)
      - Hyperbolic distance computation (CPU vs GPU comparison)
      - Entailment cone containment queries
      - Graph traversal with Marblestone edge modulation
      - Domain-aware search ranking
      - Contradiction detection pipeline
      Performance benchmarks against NFR targets.
    layer: "surface"
    priority: "critical"
    estimated_hours: 6
    file_path: "crates/context-graph-graph/tests/integration_tests.rs"
    dependencies:
      - "M04-T18"
      - "M04-T19"
      - "M04-T20"
      - "M04-T21"
      - "M04-T23"
      - "M04-T24"
    acceptance_criteria:
      - "FAISS search returns correct top-k in <10ms"
      - "Hyperbolic distance CPU/GPU match within tolerance"
      - "Entailment query finds correct hierarchy"
      - "BFS traversal respects depth limits"
      - "Domain-aware search re-ranks correctly"
      - "All tests use real FAISS index (no mocks per spec)"
      - "Tests marked #[requires_gpu] for CI skip on non-GPU"
    test_file: "crates/context-graph-graph/tests/integration_tests.rs"
    spec_refs:
      - "TECH-GRAPH-004 Section 11"
      - "All NFR-KG requirements"
```

---

## Dependency Graph

```
M04-T01 (IndexConfig) ────────────────────────────────────────────────┐
M04-T02 (HyperbolicConfig) ──► M04-T04 (PoincarePoint) ──► M04-T05 (PoincareBall) ──┐
M04-T03 (ConeConfig) ───────────────────────────────────────────────────────────────┼─► M04-T06 (EntailmentCone) ──► M04-T07 (Containment)
                                                                                    │
M04-T08 (GraphError) ──► M04-T09 (FAISS FFI) ──► M04-T10 (FaissGpuIndex) ──► M04-T11 (SearchResult)
                    │                                                                │
                    └─► M04-T12 (Column Families) ──► M04-T13 (GraphStorage) ◄───────┘
                                                            │
M04-T14 (NeurotransmitterWeights) ──► M04-T15 (GraphEdge) ──┘
                                            │
                                            ├─► M04-T16 (BFS) ──────────────┐
                                            └─► M04-T17 (DFS)                │
                                                                             │
M04-T10 + M04-T11 ──► M04-T18 (Semantic Search) ──► M04-T19 (Domain-Aware Search)
                                │                           │
M04-T07 + M04-T13 ──► M04-T20 (Entailment Query)           │
                                                            │
M04-T18 + M04-T16 ──► M04-T21 (Contradiction Detection)    │
                                                            │
M04-T15 ──► M04-T22 (get_modulated_weight)                 │
                                                            │
M04-T05 ──► M04-T23 (Poincare CUDA Kernel)                 │
                                                            │
M04-T07 ──► M04-T24 (Cone CUDA Kernel)                     │
                                                            │
M04-T18 + M04-T19 + M04-T20 + M04-T21 + M04-T23 + M04-T24 ─┴─► M04-T25 (Integration Tests)
```

---

## Implementation Order (Recommended)

### Week 1: Foundation Types
1. M04-T01: IndexConfig for FAISS
2. M04-T02: HyperbolicConfig for Poincare ball
3. M04-T03: ConeConfig for entailment
4. M04-T08: GraphError enum
5. M04-T04: PoincarePoint struct
6. M04-T05: PoincareBall Mobius operations
7. M04-T06: EntailmentCone struct
8. M04-T07: Cone containment logic

### Week 2: FAISS and Storage
9. M04-T09: FAISS FFI bindings
10. M04-T10: FaissGpuIndex wrapper
11. M04-T11: SearchResult struct
12. M04-T12: Column family definitions
13. M04-T13: GraphStorage backend
14. M04-T14: NeurotransmitterWeights (Marblestone)
15. M04-T15: GraphEdge with Marblestone

### Week 3: Traversal and Query
16. M04-T16: BFS traversal
17. M04-T17: DFS traversal
18. M04-T18: Semantic search
19. M04-T19: Domain-aware search (Marblestone)
20. M04-T20: Entailment query
21. M04-T21: Contradiction detection
22. M04-T22: get_modulated_weight function

### Week 4: CUDA Kernels and Integration
23. M04-T23: Poincare distance CUDA kernel
24. M04-T24: Cone membership CUDA kernel
25. M04-T25: Integration tests

---

## Quality Gates

| Gate | Criteria | Required For |
|------|----------|--------------|
| Foundation Complete | M04-T01 through M04-T08 pass all tests | Week 2 start |
| Index Functional | M04-T09 through M04-T15 pass all tests | Week 3 start |
| Queries Operational | M04-T16 through M04-T22 pass all tests | Week 4 start |
| Module Complete | All 25 tasks complete, GPU tests pass | Module 5 start |

---

## Performance Targets Summary

| Operation | Target | Conditions |
|-----------|--------|------------|
| FAISS k=10 search | <5ms | nprobe=128, 10M vectors |
| FAISS k=100 search | <10ms | nprobe=128, 10M vectors |
| Poincare distance (CPU) | <10us | Single pair |
| Poincare distance (GPU) | <1ms | 1K x 1K batch |
| Cone containment (CPU) | <50us | Single check |
| Cone containment (GPU) | <2ms | 1K x 1K batch |
| BFS depth=6 | <100ms | 10M nodes |
| Domain-aware search | <10ms | k=10, 10M vectors |
| Entailment query | <1ms | Per cone check |

---

## Memory Budget

| Component | Budget |
|-----------|--------|
| FAISS GPU index (10M vectors) | 8GB |
| Hyperbolic coordinates (10M nodes) | 2.5GB |
| Entailment cones (10M nodes) | 2.7GB |
| RocksDB cache | 8GB |
| **Total VRAM** | **24GB (RTX 5090)** |

---

## Marblestone Integration Summary

Tasks with Marblestone features:
- **M04-T14**: NeurotransmitterWeights struct (excitatory/inhibitory/modulatory)
- **M04-T15**: GraphEdge with domain, steering_reward, is_amortized_shortcut
- **M04-T16**: BFS with domain-aware edge filtering
- **M04-T19**: domain_aware_search() with NT modulation
- **M04-T22**: get_modulated_weight() pure function

---

## Critical Constraints

**NO MOCK FAISS**: Per spec REQ-KG-TEST, all tests MUST use real FAISS GPU index.
- Mock implementations are forbidden for vector similarity search
- Tests requiring GPU should be marked `#[requires_gpu]` for CI handling

**Hyperbolic Constraint**: All PoincarePoint instances MUST maintain ||coords|| < 1.0.
- project() must be called after any operation that could push points to boundary

---

*Generated: 2025-12-31*
*Module: 04 - Knowledge Graph*
*Version: 1.0.0*
*Total Tasks: 25*
