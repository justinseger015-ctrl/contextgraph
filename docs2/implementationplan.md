# Ultimate Context Graph - Implementation Plan

## Executive Summary

This document provides a comprehensive module-by-module breakdown for implementing the Ultimate Context Graph System - a bio-nervous memory system for AI agents based on the Unified Theory of Learning (UTL). The system implements a 5-layer nervous architecture with Modern Hopfield Networks, neuromodulation, and active inference capabilities.

**Total Estimated Duration**: 49 weeks (14 phases)
**Target Hardware**: RTX 5090 / CUDA 13.1
**Architecture**: 4-crate Rust system with MCP JSON-RPC 2.0 interface

---

## Module Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MCP Interface Layer                          │
│                    (JSON-RPC 2.0, 25+ Tools)                        │
├─────────────────────────────────────────────────────────────────────┤
│  Module 1: Ghost System │ Module 2: Core Infrastructure              │
├─────────────────────────────────────────────────────────────────────┤
│              Module 3: 12-Model Embedding Pipeline                   │
│        (Semantic, Temporal, Causal, Sparse, Code, Graph, etc.)      │
├─────────────────────────────────────────────────────────────────────┤
│  Module 4: Knowledge Graph │ Module 5: UTL Integration               │
│  (+ Neurotransmitter Edges) │ (+ Lifecycle Lambda Weights)           │
├─────────────────────────────────────────────────────────────────────┤
│              Module 6: Bio-Nervous System (5 Layers)                 │
│      (Sensing → Reflex → Memory → Learning → Coherence)             │
│                    (+ Formal Verification in L5)                     │
├─────────────────────────────────────────────────────────────────────┤
│  Module 7: CUDA Optimization │ Module 8: GPU Direct Storage          │
├─────────────────────────────────────────────────────────────────────┤
│  Module 9: Dream Layer         │ Module 10: Neuromodulation          │
│  (+ Amortized Inference)       │ (+ Steering Dopamine Feedback)      │
├─────────────────────────────────────────────────────────────────────┤
│  Module 11: Immune System │ Module 12: Active Inference              │
│                           │ (+ Omnidirectional Inference Engine)     │
├─────────────────────────────────────────────────────────────────────┤
│              Module 12.5: Steering Subsystem (NEW)                   │
│     (Gardener + Curator + Thought Assessor → Dopamine Rewards)      │
├─────────────────────────────────────────────────────────────────────┤
│  Module 13: MCP Hardening │ Module 14: Testing & Production          │
└─────────────────────────────────────────────────────────────────────┘
```

### Marblestone-Inspired Additions (v2.0)

The following neuroscience-inspired features (from Adam Marblestone's research) have been integrated:

| Feature | Module(s) | Description |
|---------|-----------|-------------|
| **Steering Subsystem** | 10, 12.5 | Gardener+Curator provide Dopamine reward signals to agents |
| **Lifecycle Lambda Weights** | 5 | Dynamic λ_ΔS/λ_ΔC based on Infancy/Growth/Maturity stage |
| **Neurotransmitter Edge Weights** | 2, 4 | Excitatory/inhibitory modulation per domain (Code, Legal, etc.) |
| **Amortized Inference** | 9 | Dream Layer creates shortcut edges from multi-hop paths |
| **Omnidirectional Inference** | 12 | Clamped variables with belief propagation (forward/backward/bridge) |
| **Formal Verification** | 6 | Lean-inspired SMT verification in Coherence layer (L5) |

---

## Module 1: Ghost System (Foundation)

**Phase**: 0
**Duration**: 2-4 weeks
**Dependencies**: None (Starting point)

### Description

The Ghost System provides the minimal viable infrastructure to bootstrap development. It implements stub interfaces for all major components, allowing parallel development of different modules. This "ghost" layer ensures that integration points are defined before actual implementation begins.

### Components

1. **Trait Definitions**
   - `UTLProcessor` trait with placeholder methods
   - `EmbeddingProvider` trait for embedding pipeline
   - `MemoryStore` trait for storage backends
   - `NervousLayer` trait for bio-nervous system

2. **Stub Implementations**
   - Mock embedding generator returning random 4096D vectors
   - In-memory key-value store for memory operations
   - Pass-through nervous system layers
   - Placeholder MCP tool handlers

3. **Configuration Framework**
   - YAML-based configuration loading
   - Environment variable override support
   - Feature flag system for gradual rollout
   - Logging infrastructure (tracing crate)

4. **Project Structure**
   ```
   context-graph/
   ├── context-graph-mcp/      # MCP server binary
   ├── context-graph-core/     # Core types and traits
   ├── context-graph-cuda/     # GPU acceleration
   └── context-graph-embeddings/ # Embedding pipeline
   ```

### Expected Capabilities After Module 1

- [ ] Project compiles and runs with `cargo run`
- [ ] MCP server starts and responds to JSON-RPC requests
- [ ] All tool stubs return placeholder responses
- [ ] Configuration can be modified via YAML files
- [ ] Basic logging and tracing operational
- [ ] Unit test framework in place with example tests
- [ ] CI/CD pipeline skeleton (build + lint + test)

### Quality Gates

- All trait definitions documented with rustdoc
- Zero compiler warnings
- Code coverage baseline established (>0%)
- Integration test demonstrating MCP round-trip

---

## Module 2: Core Infrastructure

**Phase**: 1
**Duration**: 4 weeks
**Dependencies**: Module 1 (Ghost System)

### Description

Core Infrastructure implements the foundational data types, memory management, and basic MCP tools that all other modules depend on. This includes the `Memex` storage layer, node/edge types for the knowledge graph, and the primary MCP tool implementations.

### Components

1. **Core Data Types**
   ```rust
   pub struct MemoryNode {
       pub id: Uuid,
       pub content: String,
       pub embedding: Vec<f32>,      // 4096D fused
       pub source: SourceType,
       pub timestamp: DateTime<Utc>,
       pub salience: f32,            // [0, 1]
       pub confidence: f32,          // [0, 1]
       pub johari_quadrant: JohariQuadrant,
       pub metadata: HashMap<String, Value>,
   }

   pub enum JohariQuadrant {
       Open,    // Known to self and others
       Blind,   // Unknown to self, known to others
       Hidden,  // Known to self, hidden from others
       Unknown, // Unknown to both
   }
   ```

2. **Memex Storage Layer**
   - RocksDB-based persistent storage
   - WAL (Write-Ahead Log) for durability
   - Bloom filters for existence checks
   - LRU cache for hot data

3. **Edge Types with Neurotransmitter Weights (Marblestone)**
   ```rust
   pub struct GraphEdge {
       pub source: Uuid,
       pub target: Uuid,
       pub edge_type: EdgeType,
       pub weight: f32,
       pub confidence: f32,
       pub created_at: DateTime<Utc>,
       // NEW: Molecularly-annotated neurotransmitter weights
       pub neurotransmitter: Option<NeurotransmitterWeights>,
       // NEW: Whether created by Dream Layer amortization
       pub is_amortized_shortcut: bool,
   }

   pub enum EdgeType {
       Semantic,     // Meaning similarity
       Temporal,     // Time-based connection
       Causal,       // Cause-effect relationship
       Hierarchical, // Parent-child structure
       Relational,   // Custom relationship
   }

   /// Domain-specific edge weight modulation
   pub struct NeurotransmitterWeights {
       pub excitatory: f32,  // Strengthens connection [0,1]
       pub inhibitory: f32,  // Weakens connection [0,1]
       pub domain: Domain,
   }

   pub enum Domain {
       General,   // No modulation
       Code,      // +excitatory for technical edges
       Legal,     // +inhibitory for speculative edges
       Medical,   // +inhibitory for unverified claims
       Creative,  // +excitatory for metaphorical edges
       Research,  // Balanced, prefers cited sources
   }
   ```

4. **Primary MCP Tools**
   - `inject_context` - Store new memory with embedding
   - `store_memory` - Persist memory node
   - `recall_memory` - Retrieve by semantic similarity
   - `get_memetic_status` - System health metrics
   - `set_verbosity` - Control response detail level

4. **Verbosity System**
   ```rust
   pub enum VerbosityLevel {
       RawOnly = 0,      // Just IDs/vectors
       TextAndIds = 1,   // Text + IDs (default)
       FullInsights = 2, // Complete analysis
   }
   ```

5. **Cognitive Pulse Headers**
   ```rust
   pub struct CognitivePulse {
       pub entropy: f32,        // System uncertainty
       pub coherence: f32,      // Internal consistency
       pub suggested_action: SuggestedAction,
   }
   ```

### Expected Capabilities After Module 2

- [ ] Memory nodes can be created, stored, and retrieved
- [ ] Persistence survives server restarts
- [ ] `inject_context` tool fully operational
- [ ] `store_memory` and `recall_memory` working (without embeddings)
- [ ] **NEW**: `store_memory` response includes `steering_reward` field
- [ ] Verbosity levels affect response format
- [ ] Cognitive Pulse included in all responses
- [ ] Basic metrics exposed via `get_memetic_status`
- [ ] RocksDB storage benchmarked at >10K writes/sec
- [ ] **NEW**: Edge types support optional NeurotransmitterWeights
- [ ] **NEW**: `is_amortized_shortcut` flag available on edges

### Quality Gates

- Unit tests for all core types
- Storage durability test (kill/restart)
- Memory leak detection via Valgrind
- API response schema validation
- **NEW**: NeurotransmitterWeights modulation tested per domain

---

## Module 3: 12-Model Embedding Pipeline

**Phase**: 2
**Duration**: 4 weeks
**Dependencies**: Module 2 (Core Infrastructure)

### Description

The Embedding Pipeline implements the 12-model ensemble that generates rich semantic representations for all memory content. Each model captures different aspects of meaning, which are then fused using FuseMoE (Mixture of Experts) with CAME-AB (Cross-Attention Modality Encoding with Adaptive Blending).

### Components

1. **Embedding Models**

   | Model | Dimension | Purpose | Latency Target |
   |-------|-----------|---------|----------------|
   | Semantic (E5-Large) | 1024 | General meaning | <50ms |
   | Temporal | 256 | Time relationships | <20ms |
   | Causal | 512 | Cause-effect chains | <30ms |
   | Sparse (SPLADE) | 30K→256 | Keyword matching | <40ms |
   | Code (CodeBERT) | 768 | Programming constructs | <60ms |
   | Graph (GAT) | 256 | Structural relationships | <100ms |
   | HDC | 4096 | Holographic binding | <10ms |
   | Multimodal (CLIP) | 512 | Cross-modal alignment | <80ms |
   | Entity (NER+Link) | 256 | Named entities | <50ms |
   | Late-Interaction | 128×N | Token-level matching | <150ms |
   | Hyperbolic | 64 | Hierarchical relations | <30ms |
   | Contextual | 768 | Discourse coherence | <50ms |

2. **FuseMoE Architecture**
   ```rust
   pub struct FuseMoE {
       pub experts: Vec<Expert>,      // 8 experts
       pub router: GatingNetwork,     // top-k=2 routing
       pub load_balancer: LoadBalancer,
   }
   ```

3. **CAME-AB Fusion**
   - Cross-attention between modality pairs
   - Adaptive blending weights learned per context
   - Final output: 4096D unified embedding

4. **Batching & Caching**
   - Request batching for GPU efficiency
   - LRU embedding cache (configurable size)
   - Async embedding computation

### Expected Capabilities After Module 3

- [ ] All 12 embedding models loaded and operational
- [ ] Content → 4096D fused embedding in <200ms (batch=1)
- [ ] Batch processing at >100 items/sec
- [ ] Embedding cache hit rate >80% for repeated queries
- [ ] Individual model outputs available for debugging
- [ ] FuseMoE routing weights inspectable
- [ ] Model hot-swapping without restart

### Quality Gates

- Embedding quality benchmarks (STS-B, MTEB subset)
- Latency P99 < 500ms for single items
- Memory usage < 16GB for all models
- Reproducibility test (same input → same output)

---

## Module 4: Knowledge Graph

**Phase**: 3
**Duration**: 4 weeks
**Dependencies**: Module 3 (Embedding Pipeline)

### Description

The Knowledge Graph module implements the graph storage layer using FAISS for GPU-accelerated vector similarity search, combined with a property graph for structured relationships. This enables both semantic retrieval and graph traversal operations.

### Components

1. **FAISS GPU Index**
   ```rust
   pub struct FAISSIndex {
       pub index: faiss::GpuIndexIVFPQ,
       pub nlist: u32,           // 16384 clusters
       pub nprobe: u32,          // 128 search clusters
       pub pq_m: u32,            // 64 sub-quantizers
       pub pq_bits: u32,         // 8 bits per code
   }
   ```

2. **Graph Storage**
   - Node storage: RocksDB with embedding references
   - Edge types: `RELATES_TO`, `CAUSES`, `PRECEDES`, `CONTAINS`, `CONTRADICTS`
   - Bidirectional edge indexing
   - Graph partitioning for large-scale traversal

3. **Neurotransmitter Edge Modulation (Marblestone)**
   ```rust
   impl GraphStorage {
       /// Apply domain-specific modulation when retrieving edges
       pub fn get_modulated_weight(&self, edge: &GraphEdge, domain: Domain) -> f32 {
           match &edge.neurotransmitter {
               Some(nt) if nt.domain == domain => {
                   let modulation = 1.0 + nt.excitatory - nt.inhibitory;
                   (edge.weight * modulation).clamp(0.0, 1.0)
               }
               _ => edge.weight
           }
       }

       /// Query with domain context for automatic modulation
       pub fn domain_aware_search(&self, query: &Query, domain: Domain) -> Vec<SearchResult> {
           let results = self.raw_search(query);
           results.into_iter()
               .map(|r| {
                   let modulated = self.get_modulated_weight(&r.edge, domain);
                   SearchResult { weight: modulated, ..r }
               })
               .collect()
       }
   }
   ```

4. **Hyperbolic Entailment Cones**
   ```rust
   pub struct HyperbolicCone {
       pub apex: PoincareBallPoint,  // 64D hyperbolic point
       pub aperture: f32,            // Cone opening angle
   }

   impl HyperbolicCone {
       /// O(1) entailment check
       pub fn contains(&self, point: &PoincareBallPoint) -> bool;
   }
   ```

4. **Query Operations**
   - `semantic_search(query, k)` - Top-k similar nodes
   - `graph_traverse(start, depth, edge_types)` - BFS/DFS traversal
   - `entailment_query(concept)` - Find all entailed concepts
   - `contradiction_detect(node)` - Find conflicting memories

### Expected Capabilities After Module 4

- [ ] FAISS index operational with GPU acceleration
- [ ] Semantic search returns top-k results in <10ms for k=100
- [ ] Graph traversal supports 6+ depth efficiently
- [ ] Hyperbolic entailment cones enable O(1) hierarchy queries
- [ ] Contradiction detection identifies conflicting memories
- [ ] Index supports 10M+ vectors
- [ ] Incremental index updates without full rebuild

### Quality Gates

- Recall@10 > 95% on synthetic benchmark
- Search latency P99 < 50ms
- Index build time < 1 hour for 10M vectors
- Memory efficiency: <1KB per vector stored

---

## Module 5: UTL Integration

**Phase**: 4
**Duration**: 4 weeks
**Dependencies**: Module 4 (Knowledge Graph)

### Description

UTL (Unified Theory of Learning) Integration implements the core learning equation that governs how the system acquires and strengthens memories. This module bridges the embedding pipeline and knowledge graph with learning dynamics.

### Components

1. **UTL Core Equation**
   ```
   L = f((ΔS × ΔC) ⋅ wₑ ⋅ cos φ)

   Where:
   - L: Learning magnitude
   - ΔS: Surprise (prediction error)
   - ΔC: Coherence change
   - wₑ: Emotional weight
   - φ: Phase angle (memory consolidation state)
   ```

2. **Lifecycle-Based Lambda Weights (Marblestone)**

   Dynamic weighting of ΔS vs ΔC based on system maturity:

   ```rust
   pub enum LifecycleStage {
       Infancy,   // 0-50 interactions
       Growth,    // 50-500 interactions
       Maturity,  // 500+ interactions
   }

   pub struct LifecycleLambdaWeights {
       pub lambda_ds: f32,  // Weight for entropy/surprise
       pub lambda_dc: f32,  // Weight for coherence/integration
   }

   impl LifecycleStage {
       pub fn lambda_weights(&self) -> LifecycleLambdaWeights {
           match self {
               // Infancy: Reward exploration/novelty
               Self::Infancy => LifecycleLambdaWeights {
                   lambda_ds: 0.7,
                   lambda_dc: 0.3,
               },
               // Growth: Balanced exploration + integration
               Self::Growth => LifecycleLambdaWeights {
                   lambda_ds: 0.5,
                   lambda_dc: 0.5,
               },
               // Maturity: Reward coherence/quality
               Self::Maturity => LifecycleLambdaWeights {
                   lambda_ds: 0.3,
                   lambda_dc: 0.7,
               },
           }
       }
   }

   /// Modified UTL computation with lifecycle awareness
   impl UTLProcessor {
       pub fn compute_learning_with_lifecycle(
           &self,
           ds: f32,
           dc: f32,
           we: f32,
           phi: f32,
           lifecycle: &LifecycleStage,
       ) -> f32 {
           let lambdas = lifecycle.lambda_weights();
           let weighted_ds = ds * lambdas.lambda_ds;
           let weighted_dc = dc * lambdas.lambda_dc;
           (weighted_ds * weighted_dc) * we * phi.cos()
       }
   }
   ```

   **Biological Rationale**: Infants babble freely (exploration), adults communicate precisely (integration).

3. **UTL Processor Implementation**
   ```rust
   pub struct UTLProcessor {
       pub surprise_threshold: f32,
       pub coherence_window: Duration,
       pub emotional_decay: f32,
       pub phase_oscillator: PhaseOscillator,
   }

   impl UTLProcessor {
       pub fn compute_learning(&self, input: &MemoryNode, context: &Context) -> LearningSignal;
       pub fn should_consolidate(&self, node: &MemoryNode) -> bool;
       pub fn update_salience(&mut self, node: &mut MemoryNode, signal: &LearningSignal);
   }
   ```

3. **Surprise Computation**
   - Prediction based on current context
   - KL divergence from expected distribution
   - Novelty detection via embedding distance

4. **Coherence Tracking**
   - Rolling window of recent memories
   - Graph-based coherence metric
   - Contradiction penalty

5. **Emotional Weighting**
   - Valence extraction from content
   - Arousal estimation
   - Decay over time

### Expected Capabilities After Module 5

- [ ] Learning magnitude computed for all new memories
- [ ] High-surprise content automatically prioritized
- [ ] Coherent information clusters form naturally
- [ ] Emotional content receives appropriate weighting
- [ ] Salience scores updated based on learning signals
- [ ] Phase oscillator tracks consolidation state
- [ ] UTL metrics exposed via `get_memetic_status`
- [ ] **NEW**: Lifecycle stage (Infancy/Growth/Maturity) tracked automatically
- [ ] **NEW**: Lambda weights (λ_ΔS, λ_ΔC) shift based on lifecycle stage
- [ ] **NEW**: Infancy prioritizes exploration, Maturity prioritizes coherence

### Quality Gates

- Learning signal correlates with human importance ratings (r > 0.7)
- Surprise detection catches novel information >90% of time
- Coherence metric stable over time (low variance)
- Emotional weighting benchmarked against sentiment datasets
- **NEW**: Lambda weight transitions verified at lifecycle boundaries

---

## Module 6: Bio-Nervous System

**Phase**: 5
**Duration**: 4 weeks
**Dependencies**: Module 5 (UTL Integration)

### Description

The Bio-Nervous System implements the 5-layer architecture inspired by biological neural systems. Each layer has specific responsibilities and latency budgets, creating a hierarchical processing pipeline from raw input to coherent understanding.

### Components

1. **Layer Architecture**

   | Layer | Latency Budget | Responsibility |
   |-------|---------------|----------------|
   | Sensing | <50ms | Raw input processing, tokenization |
   | Reflex | <100ms | Pattern matching, quick responses |
   | Memory | <500ms | Storage, retrieval, association |
   | Learning | <2000ms | UTL processing, consolidation |
   | Coherence | <5000ms | Global consistency, narrative |

2. **Sensing Layer**
   ```rust
   pub struct SensingLayer {
       pub tokenizer: Tokenizer,
       pub preprocessor: Preprocessor,
       pub input_buffer: RingBuffer<RawInput>,
   }
   ```

3. **Reflex Layer**
   - Fast pattern matching
   - Cached response templates
   - Bypass for urgent queries

4. **Memory Layer**
   - FAISS search integration
   - Working memory (recent context)
   - Long-term memory access

5. **Learning Layer**
   - UTL processor integration
   - Consolidation scheduling
   - Salience updates

6. **Coherence Layer (L5)**
   - Global narrative construction
   - Contradiction resolution
   - Perspective management
   - **NEW**: Formal Verification Integration (Marblestone/Lean-Inspired)

7. **Formal Verification Layer (Coherence L5)**
   ```rust
   /// Lean-inspired formal verification for code nodes
   pub struct FormalVerificationLayer {
       pub enable_smt: bool,              // Z3-style SMT solving
       pub verification_timeout_ms: u64, // Default: 5000ms
       pub proof_cache: ProofCache,      // Avoid re-verification
   }

   pub struct VerificationCondition {
       pub description: String,
       pub precondition: Option<String>,   // e.g., "x > 0 ∧ y > 0"
       pub postcondition: Option<String>,  // e.g., "result = x * y"
       pub invariants: Vec<String>,        // Loop invariants
       pub status: VerificationStatus,
   }

   pub enum VerificationStatus {
       Pending,
       Verified { proof_hash: String },
       Failed { counterexample: Option<String> },
       Timeout,
       NotApplicable,
   }

   impl FormalVerificationLayer {
       /// Called during store_memory for code nodes with specs
       pub fn coherence_verified_store(
           &mut self,
           node: &KnowledgeNode,
           graph: &mut KnowledgeGraph,
       ) -> CoherenceVerifiedResult {
           if let Some(spec) = node.metadata.get("verification_spec") {
               let result = self.verify_node(node, spec);
               // Verified: +0.2 coherence boost
               // Failed: -0.3 coherence penalty
               self.apply_coherence_adjustment(node, &result)
           } else {
               CoherenceVerifiedResult::NoSpec
           }
       }
   }
   ```

8. **Inter-Layer Communication**
   ```rust
   pub struct LayerMessage {
       pub source: NervousLayer,
       pub target: NervousLayer,
       pub payload: MessagePayload,
       pub priority: Priority,
       pub deadline: Instant,
   }
   ```

### Expected Capabilities After Module 6

- [ ] All 5 layers operational with proper latency budgets
- [ ] Sensing layer processes input in <50ms
- [ ] Reflex layer provides cached responses in <100ms
- [ ] Memory layer retrieves relevant context in <500ms
- [ ] Learning layer updates salience in <2000ms
- [ ] Coherence layer maintains narrative consistency
- [ ] Layer metrics exposed for monitoring
- [ ] Graceful degradation when latency budgets exceeded
- [ ] **NEW**: Coherence layer (L5) supports formal verification for code nodes
- [ ] **NEW**: `verify_code_node` MCP tool available
- [ ] **NEW**: Verified code nodes receive +0.2 coherence boost
- [ ] **NEW**: Failed verification nodes receive -0.3 coherence penalty

### Quality Gates

- Layer latency budgets met 95% of time
- End-to-end processing in <3s for standard queries
- Memory layer recall accuracy >90%
- Coherence layer detects contradictions >85%
- **NEW**: Formal verification completes within 5s timeout 95% of time
- **NEW**: SMT solver correctly validates test specifications

---

## Module 7: CUDA Optimization

**Phase**: 6
**Duration**: 3 weeks
**Dependencies**: Module 6 (Bio-Nervous System)

### Description

CUDA Optimization accelerates critical paths using GPU computation. This module implements custom CUDA kernels for embedding operations, FAISS searches, and neural network inference, targeting the RTX 5090 architecture.

### Components

1. **CUDA Configuration**
   ```rust
   pub struct CudaConfig {
       pub device_id: u32,
       pub compute_capability: (u32, u32),  // (9, 0) for RTX 5090
       pub memory_pool_size: usize,         // 32GB
       pub stream_count: u32,               // 8 streams
       pub green_contexts: bool,            // CUDA 13.1 feature
   }
   ```

2. **Custom Kernels**
   - `fused_embedding_kernel` - Multi-model embedding in single pass
   - `hopfield_attention_kernel` - Modern Hopfield energy computation
   - `neuromodulation_kernel` - Parallel neuromodulator updates
   - `cone_containment_kernel` - Batch hyperbolic cone checks

3. **Memory Management**
   - Unified Virtual Memory (UVM) for large graphs
   - Pinned memory for CPU-GPU transfers
   - Memory pool to reduce allocation overhead

4. **Stream Orchestration**
   - Multiple CUDA streams for parallelism
   - Dependency graph for kernel scheduling
   - Async memory transfers

5. **Green Contexts (CUDA 13.1)**
   - Power-efficient execution
   - Dynamic frequency scaling
   - Thermal management

### Expected Capabilities After Module 7

- [ ] All embedding models run on GPU
- [ ] FAISS searches fully GPU-accelerated
- [ ] Custom kernels achieve >80% GPU utilization
- [ ] Memory pool eliminates allocation latency
- [ ] Multi-stream execution reduces wall time by 3x
- [ ] Green contexts reduce power consumption by 20%
- [ ] Graceful fallback to CPU when GPU unavailable

### Quality Gates

- Kernel benchmarks exceed baseline by >5x
- GPU memory usage < 24GB under load
- No CUDA errors under stress test
- Power efficiency measured and documented

---

## Module 8: GPU Direct Storage (GDS)

**Phase**: 7
**Duration**: 3 weeks
**Dependencies**: Module 7 (CUDA Optimization)

### Description

GPU Direct Storage enables direct data paths between NVMe storage and GPU memory, bypassing CPU bottlenecks. This accelerates loading of large embedding models and memory indices.

### Components

1. **GDS Configuration**
   ```rust
   pub struct GDSConfig {
       pub enabled: bool,
       pub nvme_devices: Vec<PathBuf>,
       pub buffer_size: usize,      // 1GB default
       pub max_concurrent_ops: u32, // 64
   }
   ```

2. **Direct Load Operations**
   - Model weight loading via GDS
   - FAISS index loading via GDS
   - Memory shard streaming

3. **Async I/O Pipeline**
   - cuFile API integration
   - Double buffering for continuous streaming
   - Prefetch scheduling

4. **Fallback Mechanisms**
   - Standard filesystem fallback
   - Automatic detection of GDS support
   - Graceful degradation

### Expected Capabilities After Module 8

- [ ] Embedding models load 4x faster via GDS
- [ ] FAISS index loads directly to GPU memory
- [ ] Memory shards stream without CPU copies
- [ ] System detects GDS support automatically
- [ ] Fallback to standard I/O when GDS unavailable
- [ ] I/O bandwidth metrics exposed

### Quality Gates

- Model load time reduced >60%
- Index load time reduced >70%
- No data corruption under stress
- Works correctly without GDS hardware

---

## Module 9: Dream Layer

**Phase**: 8
**Duration**: 3 weeks
**Dependencies**: Module 8 (GPU Direct Storage)

### Description

The Dream Layer implements offline memory consolidation inspired by sleep neuroscience. It alternates between NREM (replay/compression) and REM (creative exploration) phases using the SRC (Sparse, Random, Clustered) algorithm.

### Components

1. **Dream Orchestrator**
   ```rust
   pub struct DreamLayer {
       pub nrem_phase: NREMPhase,
       pub rem_phase: REMPhase,
       pub cycle_duration: Duration,    // 90 minutes simulated
       pub compression_ratio: f32,      // Target 10:1
   }
   ```

2. **NREM Phase (Replay)**
   - High-salience memory replay
   - Redundancy elimination
   - Schema extraction
   - Index compaction

3. **REM Phase (Exploration)**
   - Random association generation
   - Novel connection discovery
   - Creative hypothesis formation
   - Edge weight adjustment

4. **Amortized Inference Phase (Marblestone)**

   When Dream Layer finds multi-hop causal paths (3+ hops), create direct "shortcut" edges so fast retrieval can find them instantly. This is neural "amortization" — precomputing inference results.

   ```rust
   impl SRCAlgorithm {
       /// NEW: Amortized shortcut creation after REM phase
       pub fn amortized_shortcut_creation(
           &self,
           graph: &mut KnowledgeGraph,
           hop_threshold: usize,  // Default: 3 hops
       ) -> Vec<AmortizedShortcut> {
           let mut shortcuts = Vec::new();

           // Find frequently-traversed multi-hop causal paths
           let causal_paths = graph.find_frequent_causal_paths(hop_threshold);

           for path in causal_paths {
               let (start, end) = (path.first(), path.last());
               if graph.has_edge(*start, *end) { continue; }

               // Compute shortcut weight as product of path weights
               let path_weight: f32 = path.edges.iter().map(|e| e.weight).product();
               let shortcut_weight = (path_weight * 1.5).min(0.9);

               graph.add_edge(GraphEdge {
                   source: *start,
                   target: *end,
                   edge_type: EdgeType::Causal,
                   weight: shortcut_weight,
                   is_amortized_shortcut: true,  // Mark as amortized
                   ..Default::default()
               });

               shortcuts.push(AmortizedShortcut {
                   start: *start,
                   end: *end,
                   original_path_length: path.nodes.len(),
               });
           }
           shortcuts
       }
   }
   ```

   **Benefits**:
   - 4-hop path A→B→C→D becomes direct A→D edge
   - Fast retrieval finds previously "hidden" connections
   - Reduces token cost of multi-hop reasoning

5. **SRC Algorithm**
   ```rust
   pub struct SRCAlgorithm {
       pub sparsity: f32,      // Fraction of nodes activated
       pub randomness: f32,    // Exploration vs exploitation
       pub clustering: f32,    // Locality preference
   }
   ```

5. **Consolidation Scheduling**
   - Background thread execution
   - Priority queue for memories
   - Resource-aware scheduling

### Expected Capabilities After Module 9

- [ ] Dream cycles run automatically in background
- [ ] NREM phase compresses redundant memories
- [ ] REM phase discovers novel associations
- [ ] SRC algorithm balances exploration/exploitation
- [ ] Memory count reduced by 30% without information loss
- [ ] Novel connections improve recall by 15%
- [ ] Dream metrics exposed for monitoring
- [ ] **NEW**: Amortized shortcuts created for 3+ hop causal paths
- [ ] **NEW**: `is_amortized_shortcut` edges bypass multi-hop traversal
- [ ] **NEW**: Shortcut creation logged for inspection

### Quality Gates

- Compression achieves 10:1 ratio on benchmark
- Recall quality maintained post-compression
- Novel associations rated useful by human eval
- Dream cycle completes within resource budget
- **NEW**: Amortized shortcuts reduce average path length by >40%
- **NEW**: Shortcut edges marked correctly with `is_amortized_shortcut=true`

---

## Module 10: Neuromodulation

**Phase**: 9
**Duration**: 3 weeks
**Dependencies**: Module 9 (Dream Layer)

### Description

The Neuromodulation module implements dynamic system parameter adjustment inspired by neurotransmitter systems. This enables adaptive behavior based on context, surprise, and task demands.

### Components

1. **Neuromodulator Channels**

   | Modulator | Target Parameter | Effect |
   |-----------|-----------------|--------|
   | Dopamine | `hopfield.beta` | Increases pattern sharpness |
   | Serotonin | `fuse_moe.top_k` | Broadens/narrows expert selection |
   | Noradrenaline | `attention.temperature` | Adjusts focus breadth |
   | Acetylcholine | `memory.retrieval_k` | Controls retrieval depth |

2. **Neuromodulation Controller**
   ```rust
   pub struct NeuromodulationController {
       pub dopamine: f32,       // [0, 1]
       pub serotonin: f32,      // [0, 1]
       pub noradrenaline: f32,  // [0, 1]
       pub acetylcholine: f32,  // [0, 1]
       pub update_rate: f32,    // Decay/growth rate
   }
   ```

3. **Trigger Conditions**
   - High surprise → Dopamine surge
   - Uncertainty → Noradrenaline increase
   - Exploration mode → Serotonin boost
   - Deep recall needed → Acetylcholine rise

4. **Parameter Mapping**
   - Smooth interpolation between states
   - Hysteresis to prevent oscillation
   - Bounds checking for stability

5. **Steering Dopamine Feedback Loop (Marblestone)**

   The Steering Subsystem (Gardener + Curator + Thought Assessor) provides Dopamine reward signals that modulate the learning system:

   ```rust
   /// Dopamine reward from Steering Subsystem
   pub struct SteeringDopamineFeedback {
       /// Gardener coherence verdict (graph pruning quality)
       pub gardener_reward: f32,      // [-1.0, 1.0]
       /// Curator relevance verdict (context appropriateness)
       pub curator_reward: f32,       // [-1.0, 1.0]
       /// Thought Assessor quality verdict (reasoning soundness)
       pub assessor_reward: f32,      // [-1.0, 1.0]
   }

   impl NeuromodulationController {
       /// Integrate Steering feedback into Dopamine channel
       pub fn apply_steering_feedback(&mut self, feedback: &SteeringDopamineFeedback) {
           let combined_reward = (
               feedback.gardener_reward * 0.3 +
               feedback.curator_reward * 0.4 +
               feedback.assessor_reward * 0.3
           );

           // Positive feedback → Dopamine surge → sharper patterns
           // Negative feedback → Dopamine dip → broader exploration
           self.dopamine = (self.dopamine + combined_reward * 0.2).clamp(0.0, 1.0);

           // Log for feedback loop analysis
           tracing::info!(
               dopamine = self.dopamine,
               gardener = feedback.gardener_reward,
               curator = feedback.curator_reward,
               assessor = feedback.assessor_reward,
               "steering_dopamine_update"
           );
       }
   }
   ```

   **Feedback Integration Point**: `store_memory` response includes `steering_reward` field that feeds into this Dopamine loop.

### Expected Capabilities After Module 10

- [ ] All 4 neuromodulator channels operational
- [ ] Parameters adjust dynamically based on context
- [ ] High surprise triggers dopamine response
- [ ] Uncertainty increases noradrenaline
- [ ] Neuromodulator levels visible in status
- [ ] System behavior adapts to task demands
- [ ] Manual override available for testing
- [ ] **NEW**: Steering Subsystem Dopamine feedback integrated
- [ ] **NEW**: Gardener/Curator/Assessor rewards modulate Dopamine channel
- [ ] **NEW**: `get_steering_feedback` MCP tool returns current Steering state

### Quality Gates

- Modulator responses match expected triggers
- Parameter changes are smooth (no jumps)
- System remains stable under extreme modulation
- Performance improves on adaptive tasks
- **NEW**: Steering feedback correctly influences Dopamine levels
- **NEW**: Positive Steering rewards increase pattern sharpness (higher beta)

---

## Module 11: Immune System (Adversarial Defense)

**Phase**: 10
**Duration**: 3 weeks
**Dependencies**: Module 10 (Neuromodulation)

### Description

The Immune System protects memory integrity through adversarial detection, semantic cancer prevention, and homeostatic plasticity. It ensures the knowledge base remains healthy and consistent.

### Components

1. **Adversarial Detection**
   ```rust
   pub struct AdversarialDetector {
       pub embedding_validator: EmbeddingValidator,
       pub anomaly_threshold: f32,
       pub quarantine_queue: Vec<MemoryNode>,
   }
   ```

2. **Semantic Cancer Detection**
   - Identifies nodes that grow connections without increasing usefulness
   - Tracks "malignant" growth patterns
   - Automatic pruning of cancerous nodes

3. **Homeostatic Plasticity**
   ```rust
   pub struct HomeostaticOptimizer {
       pub target_activity: f32,     // Desired mean salience
       pub scaling_window: Duration, // Time window for averaging
       pub importance_floor: f32,    // Minimum salience
   }
   ```

4. **Memory Quarantine**
   - Suspicious content isolated
   - Human review queue
   - Automatic release/deletion based on validation

5. **Health Metrics**
   - Graph density monitoring
   - Contradiction rate tracking
   - Growth rate analysis

### Expected Capabilities After Module 11

- [ ] Adversarial inputs detected and quarantined
- [ ] Semantic cancer identified and pruned
- [ ] Homeostatic plasticity maintains balanced activity
- [ ] Quarantine queue accessible for review
- [ ] Health metrics indicate system wellness
- [ ] Automatic recovery from detected issues
- [ ] Alert system for critical problems

### Quality Gates

- Adversarial detection >95% on benchmark attacks
- False positive rate <5%
- Semantic cancer pruning improves retrieval quality
- Homeostasis maintains stable salience distribution

---

## Module 12: Active Inference

**Phase**: 11
**Duration**: 2 weeks
**Dependencies**: Module 11 (Immune System)

### Description

Active Inference implements epistemic action generation, enabling the system to proactively seek information to reduce uncertainty. This transforms the system from passive storage to active learning.

### Components

1. **Epistemic Action Generator**
   ```rust
   pub struct EpistemicActionGenerator {
       pub uncertainty_threshold: f32,
       pub exploration_budget: u32,
       pub action_templates: Vec<ActionTemplate>,
   }

   pub enum EpistemicAction {
       SeekClarification(String),
       RequestExample(Concept),
       ProposeHypothesis(Hypothesis),
       SuggestExperiment(Experiment),
   }
   ```

2. **Uncertainty Quantification**
   - Entropy-based uncertainty
   - Ensemble disagreement
   - Out-of-distribution detection

3. **Action Templates**
   - Question generation for gaps
   - Example requests for abstract concepts
   - Hypothesis formation from partial data

4. **Suggested Actions in Response**
   - Part of CognitivePulse header
   - Prioritized by expected information gain
   - Contextually appropriate

5. **Omnidirectional Inference Engine (Marblestone)**

   Unlike traditional forward-only inference, this engine supports bidirectional belief propagation with clamped variables:

   ```rust
   /// Inference direction modes
   pub enum InferenceDirection {
       Forward,     // Given cause, predict effect
       Backward,    // Given effect, infer cause
       Bridge,      // Connect two clamped nodes
       Abduction,   // Best explanation for observation
   }

   /// A clamped variable is a known fact that constrains inference
   pub enum ClampedValue {
       Observation(String),   // "The server crashed at 3pm"
       Constraint(String),    // "Response time must be < 100ms"
       Goal(String),          // "Minimize memory usage"
   }

   pub struct OmniInferenceEngine {
       pub max_iterations: u32,           // Belief propagation iterations
       pub convergence_threshold: f32,    // Stop when change < threshold
       pub enable_abduction: bool,        // Allow best-explanation inference
   }

   impl OmniInferenceEngine {
       /// Run inference with clamped variables
       pub fn infer(
           &self,
           graph: &KnowledgeGraph,
           clamped: &[ClampedValue],
           direction: InferenceDirection,
       ) -> InferenceResult {
           match direction {
               // Forward: A→B→C, clamp A, propagate to C
               InferenceDirection::Forward => self.forward_propagate(graph, clamped),
               // Backward: A→B→C, clamp C, infer A
               InferenceDirection::Backward => self.backward_propagate(graph, clamped),
               // Bridge: clamp A and C, find path B
               InferenceDirection::Bridge => self.bridge_inference(graph, clamped),
               // Abduction: find best explanation for observations
               InferenceDirection::Abduction => self.abductive_inference(graph, clamped),
           }
       }

       fn backward_propagate(
           &self,
           graph: &KnowledgeGraph,
           clamped: &[ClampedValue],
       ) -> InferenceResult {
           // Use inverse edge weights for backward propagation
           // Iterate until beliefs converge
           let mut beliefs = self.initialize_beliefs(graph, clamped);
           for _ in 0..self.max_iterations {
               let delta = self.propagate_step_backward(&mut beliefs, graph);
               if delta < self.convergence_threshold {
                   break;
               }
           }
           InferenceResult { beliefs, converged: true }
       }
   }
   ```

   **Use Cases**:
   - **Forward**: "Given this code change, what might break?"
   - **Backward**: "Given this error, what caused it?"
   - **Bridge**: "How does concept A relate to concept Z?"
   - **Abduction**: "What's the best explanation for these symptoms?"

### Expected Capabilities After Module 12

- [ ] System identifies knowledge gaps
- [ ] Epistemic actions generated proactively
- [ ] Suggested actions appear in responses
- [ ] Actions are contextually relevant
- [ ] Uncertainty reduced through user interaction
- [ ] Information gain tracked over time
- [ ] **NEW**: Omnidirectional inference supports Forward/Backward/Bridge/Abduction
- [ ] **NEW**: `omni_infer` MCP tool available with direction parameter
- [ ] **NEW**: Clamped variables constrain belief propagation
- [ ] **NEW**: Backward inference enables causal reasoning from effects

### Quality Gates

- Actions rated helpful by users >70%
- Uncertainty reduction measurable
- Actions contextually appropriate >90%
- No spam/excessive action suggestions
- **NEW**: Omnidirectional inference converges within 100 iterations
- **NEW**: Backward inference correctly identifies causes >80% on benchmark
- **NEW**: Bridge inference finds valid paths between clamped nodes

---

## Module 12.5: Steering Subsystem (NEW - Marblestone)

**Phase**: 11.5 (parallel with Module 12)
**Duration**: 3 weeks
**Dependencies**: Module 10 (Neuromodulation), Module 12 (Active Inference)

### Description

The Steering Subsystem implements Adam Marblestone's distinction between the **Learning Subsystem** (acquiring knowledge) and the **Steering Subsystem** (guiding behavior). While the Learning Subsystem handles memory consolidation and pattern formation, the Steering Subsystem evaluates, curates, and rewards good cognitive behavior through Dopamine feedback signals.

This module implements the **Gardener**, **Curator**, and **Thought Assessor** components that together provide continuous quality assessment of the agent's memory operations.

### Components

1. **Steering Architecture**
   ```
   ┌──────────────────────────────────────────────────────────┐
   │                   STEERING SUBSYSTEM                      │
   │  ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐   │
   │  │  Gardener   │ │   Curator   │ │ Thought Assessor │   │
   │  │  (Prune)    │ │  (Curate)   │ │    (Judge)       │   │
   │  └──────┬──────┘ └──────┬──────┘ └────────┬─────────┘   │
   │         │               │                  │             │
   │         └───────────────┴──────────────────┘             │
   │                         │                                │
   │                 SteeringReward                           │
   │           (Dopamine feedback signal)                     │
   └─────────────────────────┬────────────────────────────────┘
                             │
                             ▼
                   Neuromodulation Controller
                     (Module 10 Dopamine)
   ```

2. **Gardener Component**
   ```rust
   /// Gardener: Prunes the knowledge graph for coherence
   pub struct Gardener {
       pub coherence_threshold: f32,     // Min coherence for edges
       pub pruning_interval: Duration,   // How often to prune
       pub max_orphan_age: Duration,     // When to remove orphan nodes
   }

   impl Gardener {
       /// Evaluate graph health and generate reward signal
       pub fn evaluate(&self, graph: &KnowledgeGraph) -> GardenerVerdict {
           let orphan_count = graph.count_orphans();
           let low_coherence_edges = graph.edges_below_threshold(self.coherence_threshold);
           let contradiction_count = graph.count_contradictions();

           let health_score = 1.0 - (
               orphan_count as f32 * 0.1 +
               low_coherence_edges as f32 * 0.2 +
               contradiction_count as f32 * 0.3
           ).min(1.0);

           GardenerVerdict {
               health_score,
               suggested_prunes: low_coherence_edges,
               reward: (health_score - 0.5) * 2.0, // [-1, 1]
           }
       }
   }
   ```

3. **Curator Component**
   ```rust
   /// Curator: Ensures retrieved context is relevant and appropriate
   pub struct Curator {
       pub relevance_threshold: f32,     // Min relevance for inclusion
       pub diversity_weight: f32,        // Balance relevance vs diversity
       pub recency_bias: f32,            // Preference for recent memories
   }

   impl Curator {
       /// Evaluate retrieval quality and generate reward signal
       pub fn evaluate(
           &self,
           query: &str,
           retrieved: &[MemoryNode],
           context: &Context,
       ) -> CuratorVerdict {
           let relevance = self.compute_relevance(query, retrieved);
           let diversity = self.compute_diversity(retrieved);
           let recency = self.compute_recency(retrieved);

           let quality_score =
               relevance * 0.6 +
               diversity * 0.2 +
               recency * self.recency_bias;

           CuratorVerdict {
               quality_score,
               filtered_results: retrieved.iter()
                   .filter(|n| n.relevance >= self.relevance_threshold)
                   .collect(),
               reward: (quality_score - 0.5) * 2.0, // [-1, 1]
           }
       }
   }
   ```

4. **Thought Assessor Component**
   ```rust
   /// Thought Assessor: Judges reasoning quality and soundness
   pub struct ThoughtAssessor {
       pub enable_formal_verification: bool,  // Use Module 6 verifier
       pub confidence_threshold: f32,         // Min confidence for claims
       pub speculation_penalty: f32,          // Penalty for ungrounded claims
   }

   impl ThoughtAssessor {
       /// Evaluate reasoning quality and generate reward signal
       pub fn evaluate(
           &self,
           thought: &ThoughtChain,
           supporting_evidence: &[MemoryNode],
       ) -> AssessorVerdict {
           let grounding = self.compute_grounding(thought, supporting_evidence);
           let logical_soundness = self.check_logical_consistency(thought);
           let speculation_level = self.detect_speculation(thought);

           let quality_score =
               grounding * 0.4 +
               logical_soundness * 0.4 -
               speculation_level * self.speculation_penalty;

           AssessorVerdict {
               quality_score: quality_score.max(0.0),
               issues: self.identify_issues(thought),
               reward: (quality_score - 0.5) * 2.0, // [-1, 1]
           }
       }
   }
   ```

5. **Combined Steering Reward**
   ```rust
   /// Combined steering reward signal
   pub struct SteeringReward {
       pub gardener_reward: f32,    // Graph health [-1, 1]
       pub curator_reward: f32,     // Retrieval quality [-1, 1]
       pub assessor_reward: f32,    // Reasoning quality [-1, 1]
       pub combined: f32,           // Weighted combination
       pub timestamp: DateTime<Utc>,
   }

   impl SteeringReward {
       pub fn compute(
           gardener: &GardenerVerdict,
           curator: &CuratorVerdict,
           assessor: &AssessorVerdict,
       ) -> Self {
           let combined =
               gardener.reward * 0.3 +
               curator.reward * 0.4 +
               assessor.reward * 0.3;

           Self {
               gardener_reward: gardener.reward,
               curator_reward: curator.reward,
               assessor_reward: assessor.reward,
               combined,
               timestamp: Utc::now(),
           }
       }
   }
   ```

6. **MCP Integration**
   - `store_memory` response includes `steering_reward` field
   - `get_steering_feedback` tool returns current Steering state
   - Dopamine feedback loop connects to Module 10 Neuromodulation

### Expected Capabilities After Module 12.5

- [ ] Gardener continuously monitors graph health
- [ ] Curator filters and ranks retrieved memories
- [ ] Thought Assessor evaluates reasoning quality
- [ ] Combined SteeringReward computed for every operation
- [ ] `store_memory` responses include steering_reward field
- [ ] `get_steering_feedback` MCP tool operational
- [ ] Dopamine feedback integrated with Module 10
- [ ] Steering metrics visible in `get_memetic_status`

### Quality Gates

- Gardener correctly identifies low-coherence edges >90%
- Curator relevance filtering improves retrieval precision by >20%
- Thought Assessor detects logical inconsistencies >85%
- Dopamine feedback correlates with external quality ratings (r > 0.6)
- Steering overhead < 50ms per operation

---

## Module 13: MCP Hardening

**Phase**: 12
**Duration**: 4 weeks
**Dependencies**: Module 12 (Active Inference), Module 12.5 (Steering Subsystem)

### Description

MCP Hardening fortifies the MCP server for production deployment. This includes security hardening, rate limiting, authentication, and comprehensive error handling.

### Components

1. **Security Measures**
   - Input validation and sanitization
   - Output encoding
   - SQL injection prevention (parameterized queries)
   - Path traversal prevention

2. **Authentication & Authorization**
   - API key authentication
   - Role-based access control
   - Session management
   - Audit logging

3. **Rate Limiting**
   ```rust
   pub struct RateLimiter {
       pub requests_per_minute: u32,
       pub burst_size: u32,
       pub by_api_key: bool,
   }
   ```

4. **Error Handling**
   - Structured error responses
   - Error codes catalog
   - Stack traces in debug mode only
   - Graceful degradation

5. **Monitoring & Observability**
   - Prometheus metrics endpoint
   - Distributed tracing (OpenTelemetry)
   - Health check endpoints
   - Log aggregation

6. **Resource Management**
   - Connection pooling
   - Request timeouts
   - Memory limits
   - Graceful shutdown

### Expected Capabilities After Module 13

- [ ] All inputs validated and sanitized
- [ ] Authentication required for sensitive operations
- [ ] Rate limiting prevents abuse
- [ ] Errors are informative but safe
- [ ] Full observability via metrics/traces/logs
- [ ] System survives malicious inputs
- [ ] Graceful degradation under load

### Quality Gates

- Security audit passed
- Penetration test completed
- Rate limiting tested under load
- Error handling covers all paths
- Observability dashboards functional

---

## Module 14: Testing & Production

**Phase**: 13-14
**Duration**: 8 weeks (4 testing + 4 deployment)
**Dependencies**: All previous modules

### Description

The final module encompasses comprehensive testing across all system components and production deployment with full operational procedures.

### Testing Components (Phase 13)

1. **Unit Testing**
   - All public functions tested
   - Edge cases covered
   - Mock-based isolation

2. **Integration Testing**
   - Module interaction tests
   - End-to-end workflows
   - Database integration

3. **Performance Testing**
   - Latency benchmarks
   - Throughput testing
   - Memory profiling
   - GPU utilization

4. **Stress Testing**
   - Load testing (10x expected)
   - Chaos engineering
   - Failover scenarios

5. **Security Testing**
   - OWASP top 10 validation
   - Fuzzing
   - Dependency scanning

### Production Components (Phase 14)

1. **Deployment Infrastructure**
   - Docker containerization
   - Kubernetes manifests
   - Helm charts
   - CI/CD pipelines

2. **Operational Procedures**
   - Runbooks for common issues
   - Backup/restore procedures
   - Scaling playbooks
   - Incident response

3. **Documentation**
   - API documentation
   - Operations manual
   - Troubleshooting guide
   - Architecture decision records

4. **Monitoring & Alerting**
   - Dashboard setup
   - Alert rules
   - On-call procedures
   - SLA tracking

### Expected Capabilities After Module 14

- [ ] >90% code coverage
- [ ] All performance targets met
- [ ] Security audit passed
- [ ] Production deployment operational
- [ ] Monitoring fully configured
- [ ] Runbooks complete
- [ ] Team trained on operations

### Quality Gates

- Code coverage >90%
- Zero critical/high security issues
- Performance meets all latency targets
- Deployment can be done in <1 hour
- Rollback tested and documented
- On-call procedures documented

---

## Module Dependency Graph

```
Module 1: Ghost System
    │
    ▼
Module 2: Core Infrastructure
    │
    ▼
Module 3: Embedding Pipeline
    │
    ▼
Module 4: Knowledge Graph
    │
    ▼
Module 5: UTL Integration
    │
    ▼
Module 6: Bio-Nervous System
    │
    ▼
Module 7: CUDA Optimization
    │
    ▼
Module 8: GPU Direct Storage
    │
    ▼
Module 9: Dream Layer
    │
    ▼
Module 10: Neuromodulation
    │
    ▼
Module 11: Immune System
    │
    ▼
Module 12: Active Inference
    │
    ▼
Module 13: MCP Hardening
    │
    ▼
Module 14: Testing & Production
```

---

## Cumulative Capability Matrix

| After Module | Key Capabilities |
|-------------|------------------|
| 1 | Project compiles, stubs respond, CI/CD skeleton |
| 2 | Memory CRUD, persistence, verbosity control |
| 3 | 12-model embeddings, FuseMoE fusion, <200ms latency |
| 4 | Vector search, graph traversal, entailment queries |
| 5 | Learning signals, surprise detection, salience updates |
| 6 | 5-layer processing, latency budgets, layer metrics |
| 7 | GPU acceleration, 5x speedup, <24GB VRAM |
| 8 | GDS loading, 4x faster model loads |
| 9 | Dream cycles, compression, novel connections |
| 10 | Adaptive parameters, context-aware behavior |
| 11 | Adversarial defense, semantic health, homeostasis |
| 12 | Proactive learning, uncertainty reduction |
| 13 | Security hardened, observable, rate limited |
| 14 | Production ready, documented, monitored |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GPU memory constraints | Quantization, model pruning, offloading |
| Embedding latency | Batching, caching, async processing |
| Graph scalability | Partitioning, approximate algorithms |
| Model quality degradation | Continuous evaluation, fallbacks |
| Security vulnerabilities | Regular audits, dependency updates |
| Integration complexity | Clear interfaces, contract testing |

---

## Success Metrics

1. **Performance**
   - End-to-end latency <3s for 95% of queries
   - Throughput >100 queries/second
   - GPU utilization >80%

2. **Quality**
   - Recall@10 >95%
   - Contradiction detection >85%
   - Adversarial detection >95%

3. **Operational**
   - Uptime >99.9%
   - Mean time to recovery <15 minutes
   - Deployment time <1 hour

4. **Development**
   - Code coverage >90%
   - Zero critical security issues
   - Documentation complete

---

## Appendix: MCP Tool Reference

### Core Tools
- `inject_context` - Store memory with embedding
- `store_memory` - Persist memory node (includes `steering_reward` in response)
- `recall_memory` - Retrieve by similarity
- `get_memetic_status` - System health
- `set_verbosity` - Response detail level

### Advanced Tools
- `reflect_on_memory` - Meta-analysis
- `get_node_lineage` - Provenance tracking
- `hydrate_citation` - Expand references
- `critique_context` - Quality assessment
- `priors_vibe_check` - Perspective alignment

### Steering Subsystem Tools (NEW - Marblestone)
- `get_steering_feedback` - Get current Steering state (Gardener/Curator/Assessor rewards)
- `omni_infer` - Run omnidirectional inference (forward/backward/bridge/abduction)
- `verify_code_node` - Formal verification for code nodes (Lean-inspired SMT)

### System Tools
- `reload_manifest` - Hot reload config
- `get_system_logs` - Debug information
- `trigger_dream_cycle` - Manual consolidation
- `get_neuromodulator_state` - Modulator levels

---

## Appendix: Marblestone Features Cross-Reference

| Feature | PRD Source | Implementation Module(s) |
|---------|------------|--------------------------|
| Steering Subsystem | vision_and_layers.md | Module 10, 12.5 |
| Lifecycle Lambda Weights | vision_and_layers.md | Module 5 |
| Neurotransmitter Edge Weights | vision_and_layers.md | Module 2, 4 |
| Amortized Inference | technical_engine.md | Module 9 |
| Omnidirectional Inference | technical_engine.md | Module 12 |
| Formal Verification | technical_engine.md | Module 6 |
| Dopamine Feedback Loop | execution_and_mcp.md | Module 10, 12.5 |
| Steering MCP Tools | execution_and_mcp2.md | Module 12.5, 13 |

---

*Document Version: 2.0*
*Generated: 2025-01-01*
*Based on PRD files in /docs2/*
*Updated with Marblestone neuroscience-inspired features*
