# Context Graph System Architecture - Comprehensive Report

**Version:** 6.5.0
**Generated:** 2026-01-26
**Repository:** /home/cabdru/contextgraph

---

## Executive Summary

The Context Graph is a sophisticated semantic memory system built in Rust using a **13-perspective embedder architecture**. It provides semantic search, memory storage, causal reasoning, and entity knowledge graph capabilities through an MCP (Model Context Protocol) JSON-RPC interface. The system is GPU-first (CUDA required for RTX 5090 Blackwell), uses RocksDB for persistent storage, and implements advanced multi-space clustering and graph linking.

**Core Philosophy:** 13 embedders = 13 unique perspectives on every memory. Each finds what OTHERS MISS. Combined = superior answers.

---

## Table of Contents

1. [Project Structure & Crates](#1-project-structure--crates)
2. [The 13-Embedder System (E1-E13)](#2-the-13-embedder-system-e1-e13)
3. [Memory Storage & Retrieval System](#3-memory-storage--retrieval-system)
4. [MCP Tools (43+ Total)](#4-mcp-tools-43-total)
5. [Topic Detection & Clustering](#5-topic-detection--clustering)
6. [Causal Reasoning](#6-causal-reasoning)
7. [Entity Knowledge Graph (E11 KEPLER)](#7-entity-knowledge-graph-e11-kepler)
8. [Code Embedding Pipeline](#8-code-embedding-pipeline)
9. [File Watcher Functionality](#9-file-watcher-functionality)
10. [Hooks System](#10-hooks-system)
11. [Graph Linking (Typed Edges & K-NN)](#11-graph-linking-typed-edges--k-nn)
12. [Storage Backends](#12-storage-backends)
13. [Key Data Structures](#13-key-data-structures)
14. [Benchmark Capabilities](#14-benchmark-capabilities)
15. [System Architecture & Integration](#15-system-architecture--integration)
16. [Constitution Compliance & Architecture Rules](#16-constitution-compliance--architecture-rules)
17. [Performance & Optimization](#17-performance--optimization)

---

## 1. Project Structure & Crates

### Workspace Organization

**Root:** `/home/cabdru/contextgraph/`

### Workspace Members (9 crates)

| Crate | Purpose | Key Dependencies |
|-------|---------|------------------|
| **context-graph-mcp** | MCP JSON-RPC server and tool handlers | All other crates |
| **context-graph-core** | Domain types, traits, business logic | RocksDB, tree-sitter, hnsw_rs |
| **context-graph-embeddings** | 13-embedder pipeline and model inference | Candle-core, Tokenizers, SafeTensors |
| **context-graph-storage** | Persistent storage with RocksDB backend | RocksDB, bincode, USearch |
| **context-graph-cuda** | GPU acceleration layer (RTX 5090 optimized) | CUDA 13.x support |
| **context-graph-graph** | Knowledge graph with FAISS GPU search | RocksDB, CUDA |
| **context-graph-cli** | Command-line interface for management | Clap, All core crates |
| **context-graph-benchmark** | Performance benchmarking suite | Criterion, various datasets |
| **context-graph-causal-agent** | LLM-based causal discovery agent | Core embeddings, MCP |

### Configuration Files

**MCP Server Config** (`.mcp.json`):
- Startup command: `/home/cabdru/contextgraph/target/release/context-graph-mcp`
- Arguments: `--daemon`
- Environment: `RUST_LOG`, `CONTEXT_GRAPH_STORAGE_PATH`, `CONTEXT_GRAPH_MODELS_PATH`, `MCP_TIMEOUT`

**Constitution** (`CLAUDE.md`): v6.5.0 - Defines 13-embedder collaboration rules, retrieval strategies, topic detection thresholds

---

## 2. The 13-Embedder System (E1-E13)

### Architecture Overview

Each memory is stored with **atomic TeleologicalArray**: ALL 13 embeddings or nothing (ARCH-01). Total storage: ~46KB per memory vs ~6KB fused = 67% information preservation.

**Key Principle:** NO FUSION - Each embedding space is preserved independently for per-space HNSW search, per-space teleological alignment, and full semantic preservation.

### Embedder Specifications

| ID | Name | Model | Dimension | Purpose | Topic Weight |
|----|------|-------|-----------|---------|--------------|
| **E1** | V_meaning | e5-large-v2 | 1024D | **Foundation semantic** | 1.0 |
| **E2** | V_freshness | Exponential Decay | 512D | Recency/temporal | 0.0 |
| **E3** | V_periodicity | Fourier Periodic | 512D | Time-of-day patterns | 0.0 |
| **E4** | V_ordering | Sinusoidal PE | 512D | Sequence before/after | 0.0 |
| **E5** | V_causality | Longformer SCM | 768D | **Causal chains (asymmetric)** | 1.0 |
| **E6** | V_selectivity | SPLADE (Sparse) | 30,522 vocab | Exact keyword matches | 1.0 |
| **E7** | V_correctness | Qodo-Embed-1.5B | 1536D | **Code patterns** | 1.0 |
| **E8** | V_connectivity | MiniLM | 384D | Graph structure (imports) | 0.5 |
| **E9** | V_robustness | HDC (projected) | 1024D | Noise-tolerant typos | 1.0 |
| **E10** | V_multimodality | CLIP | 768D | **Intent alignment (multiplicative boost)** | 1.0 |
| **E11** | V_factuality | KEPLER | 768D | **Entity knowledge (TransE)** | 0.5 |
| **E12** | V_precision | ColBERT | 128/token | Exact phrase (reranking only) | 1.0 |
| **E13** | V_keyword_precision | SPLADE v3 | 30,522 vocab | Term expansion (recall stage) | 1.0 |

### Embedder Categories

| Category | Embedders | Topic Weight | Purpose |
|----------|-----------|--------------|---------|
| **Semantic** | E1, E5, E6, E7, E10, E12, E13 | 1.0 | Foundation + 6 enhancers |
| **Relational** | E8, E11 | 0.5 | Structure & entities |
| **Temporal** | E2, E3, E4 | 0.0 | NEVER count toward topics (AP-60) |
| **Structural** | E9 | 0.5 | Robustness |

### Total Embedded Dimensions

- **Dense:** E1(1024) + E2(512) + E3(512) + E4(512) + E5(768) + E7(1536) + E8(384) + E9(1024) + E10(768) + E11(768) = **7,808 dimensions**
- **Sparse:** E6 & E13 (30,522 vocab each)
- **Variable:** E12 (128 per token, variable length)

---

## 3. Memory Storage & Retrieval System

### Memory Data Structure

```rust
Memory {
    id: Uuid,                              // Unique identifier (UUID v4)
    content: String,                       // Max 10,000 chars
    source: MemorySource,                  // HookDescription | ClaudeResponse | MDFileChunk
    created_at: DateTime<Utc>,             // Timestamp
    session_id: String,                    // Session identifier
    teleological_array: TeleologicalArray, // ALL 13 embeddings (ARCH-01)
    chunk_metadata: Option<ChunkMetadata>, // File/chunk info for MDFileChunk
    word_count: u32,                       // For statistics
}
```

### Memory Sources

| Source | Description | Trigger |
|--------|-------------|---------|
| **HookDescription** | Claude's description of tool use | PostToolUse hook |
| **ClaudeResponse** | Session summaries, significant responses | Stop hook |
| **MDFileChunk** | Markdown file chunks (200 words, 50 overlap) | File watcher |

### Retrieval Pipeline (5 Stages)

**ARCH-25:** Temporal boosts POST-retrieval only, NOT in similarity fusion

| Stage | Process | Output |
|-------|---------|--------|
| **S1** | E13 SPLADE Sparse Recall | ~10K candidates |
| **S2** | E1 Matryoshka ANN (HNSW) | ~1K candidates |
| **S3** | Weighted RRF (E5, E6, E7, E10) | ~100 candidates |
| **S4** | Topic Alignment (weighted_agreement >= 2.5) | ~50 candidates |
| **S5** | E12 ColBERT MaxSim Reranking | 10 final results |

### Search Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **E1Only** | Fast semantic queries | Simple searches |
| **MultiSpace** | E1 + enhancers via RRF | Default (blind spot coverage) |
| **Pipeline** | E13 recall → E1 dense → E12 rerank | Maximum precision |
| **CodeSearch** | E7-first with AST-aware results | Code queries |

### Asymmetric Similarity

| Embedder | Forward | Reverse | Application |
|----------|---------|---------|-------------|
| **E5 (Causal)** | 1.2x boost | 0.8x dampening | cause→effect direction |
| **E8 (Graph)** | 1.2x boost | 0.8x dampening | source→target direction |

### E10 Multiplicative Boost (ARCH-28-30)

| E1 Strength | Boost Amount | Purpose |
|-------------|--------------|---------|
| Strong (>0.8) | 5% | Refine good matches |
| Medium (0.4-0.8) | 10% | Moderate enhancement |
| Weak (<0.4) | 15% | Broaden weak results |

- Multiplier clamped to [0.8, 1.2]
- When E1=0, result=0 (no override per AP-84)

---

## 4. MCP Tools (43+ Total)

### Tool Categories

#### Core Tools (4)
| Tool | Purpose |
|------|---------|
| `store_memory` | Direct knowledge graph storage |
| `get_memetic_status` | System status, fingerprint count |
| `search_graph` | Multi-space semantic search |
| `trigger_consolidation` | Merge redundant memories |

#### Merge & Curation (3)
| Tool | Purpose |
|------|---------|
| `merge_concepts` | Unify related concept nodes |
| `boost_importance` | Adjust importance score |
| `forget_concept` | Soft-delete (30-day recovery) |

#### Topic Management (4)
| Tool | Purpose |
|------|---------|
| `get_topic_portfolio` | Discover emergent topics |
| `get_topic_stability` | Portfolio stability metrics |
| `detect_topics` | Force HDBSCAN recalculation |
| `get_divergence_alerts` | Detect semantic drift |

#### File Watcher (4)
| Tool | Purpose |
|------|---------|
| `list_watched_files` | Files with embeddings |
| `get_file_watcher_stats` | Aggregate statistics |
| `delete_file_content` | Clean file embeddings |
| `reconcile_files` | Find orphaned embeddings |

#### Sequence/Session Tools (4)
| Tool | Purpose |
|------|---------|
| `get_conversation_context` | Memories around turn |
| `get_session_timeline` | Ordered session memories |
| `traverse_memory_chain` | Multi-hop traversal |
| `compare_session_states` | State comparison |

#### Causal Tools (4)
| Tool | Purpose |
|------|---------|
| `get_causal_chain` | Build transitive chains |
| `search_causes` | Abductive reasoning |
| `discover_causal_relationships` | LLM inference |
| `validate_causal_hypothesis` | Score propositions |

#### Keyword Tools (1)
| Tool | Purpose |
|------|---------|
| `search_by_keywords` | Exact keyword matches with SPLADE |

#### Code Tools (1)
| Tool | Purpose |
|------|---------|
| `search_code` | E7-first code pattern search |

#### Graph Tools (2)
| Tool | Purpose |
|------|---------|
| `get_graph_path` | Multi-hop K-NN traversal |
| `search_connections` | Find connected memories |

#### Robustness Tools (1)
| Tool | Purpose |
|------|---------|
| `search_robust` | Noise-tolerant typo search |

#### Intent Tools (2)
| Tool | Purpose |
|------|---------|
| `search_by_intent` | Find same-goal work |
| `find_contextual_matches` | Situation-relevant memories |

#### Entity Tools (6)
| Tool | Purpose |
|------|---------|
| `extract_entities` | Canonicalize entities |
| `search_by_entities` | Find by entity |
| `infer_relationship` | TransE inference |
| `find_related_entities` | Relationship search |
| `validate_knowledge` | Triple validation |
| `get_entity_graph` | Visualize relationships |

#### Embedder-First Search (4)
| Tool | Purpose |
|------|---------|
| `search_by_embedder` | Query through specific embedder |
| `get_embedder_clusters` | Per-embedder clusters |
| `compare_embedder_views` | Side-by-side rankings |
| `list_embedder_indexes` | Embedder statistics |

#### Temporal Tools (2)
| Tool | Purpose |
|------|---------|
| `search_recent` | Recency-boosted search |
| `search_periodic` | Time pattern search |

#### Graph Linking Tools (4)
| Tool | Purpose |
|------|---------|
| `get_memory_neighbors` | K-NN in embedder space |
| `get_typed_edges` | Multi-relation edges |
| `traverse_graph` | Multi-hop typed edge traversal |
| `get_unified_neighbors` | RRF neighbors across all embedders |

---

## 5. Topic Detection & Clustering

### Weighted Agreement Formula

```
Topic.weighted_agreement = Σ(topic_weight_i × is_clustered_i)

Threshold: >= 2.5
Max: 8.5 (7×1.0 semantic + 2×0.5 relational + 1×0.5 structural)
Temporal: 0.0 (NEVER counted, ARCH-04)
```

### Clustering Algorithms

| Algorithm | Type | Use Case |
|-----------|------|----------|
| **HDBSCAN** | Batch | Full reclustering, topic detection |
| **BIRCH** | Incremental | Online updates, CF-tree O(log n) |

### Topic Lifecycle Phases

| Phase | Description |
|-------|-------------|
| **Emerging** | New topic < 5 memories |
| **Stable** | Consistent > 5 memories, low churn |
| **Declining** | Decreasing size, high churn |
| **Merging** | Converging with other topics |

### Stability Metrics

| Metric | Description |
|--------|-------------|
| **Churn Rate** | Memories enter/exit per hour |
| **Entropy** | Portfolio randomness (0.0-1.0) |
| **Dream Trigger** | entropy > 0.7 AND churn > 0.5 → NREM consolidation |

---

## 6. Causal Reasoning

### Structural Causal Models (SCM)

```rust
CausalNode {
    id: Uuid,
    name: String,
    domain: String,  // "physics", "software", etc.
}

CausalEdge {
    source: Uuid,
    target: Uuid,
    strength: f32,           // 0.0-1.0
    intervention_context: String,
}
```

### Causal Inference Modes

| Mode | Description |
|------|-------------|
| **Forward** | A → B (what are the effects of A?) |
| **Backward** | B ← A (what caused B?) |
| **Bidirectional** | A ↔ B (mutual influence) |
| **Bridge** | Cross-domain causal links |
| **Abduction** | Best hypothesis given observation |

### Asymmetric Similarity (E5)

| Direction | Modifier | Application |
|-----------|----------|-------------|
| cause→effect | × 1.2 | Boost forward direction |
| effect→cause | × 0.8 | Dampen reverse direction |

**Transitive Chain Scoring:**
- HOP_ATTENUATION: 0.9^hop
- MIN_CHAIN_SCORE: 0.1
- MAX_CHAIN_LENGTH: 10

---

## 7. Entity Knowledge Graph (E11 KEPLER)

### Entity Philosophy

KEPLER was trained on Wikidata5M (4.8M entities, 20M triples) with TransE objective. KEPLER embeddings ARE the entity detection - no keyword lookup needed.

### Entity Types

- ProgrammingLanguage
- Framework
- Database
- Cloud
- Company
- TechnicalTerm
- Unknown

### Entity Operations

| Operation | Example |
|-----------|---------|
| **Extract** | "Diesel ORM for Rust" → Diesel (Database), Rust (ProgrammingLanguage) |
| **Canonicalize** | postgres → postgresql, k8s → kubernetes |
| **TransE Inference** | (Tokio, uses, Rust) → score |
| **Validation** | Triple: (subject, predicate, object) → valid/uncertain/unlikely |

---

## 8. Code Embedding Pipeline

### Separate Code Storage

Per ARCH-CODE-01 to ARCH-CODE-04:
- Code entities stored SEPARATELY from teleological memories
- E7 (Qodo-Embed-1.5B) is PRIMARY embedder
- AST chunking preserves syntactic boundaries
- Tree-sitter for Rust parsing

### AST Code Chunking

| Parameter | Value |
|-----------|-------|
| **Parser** | tree-sitter (AST-aware) |
| **Target Size** | 500 chars (Qodo recommendation) |
| **Min Size** | 100 chars |
| **Max Size** | 1000 chars |

### CodeEntity Structure

```rust
CodeEntity {
    id: Uuid,
    file_path: String,
    language: CodeLanguage,      // Rust, Python, TypeScript, etc.
    entity_type: CodeEntityType, // Function, Struct, Trait, Impl, Module
    name: String,
    signature: String,
    body: String,
    line_range: (u32, u32),
}
```

### Code Search

| Parameter | Default |
|-----------|---------|
| **Primary** | E7 (60% weight) |
| **E1 Blend** | 40% |
| **Modes** | signature, pattern, semantic |

---

## 9. File Watcher Functionality

### GitFileWatcher

| Feature | Description |
|---------|-------------|
| **Watches** | Git-tracked files |
| **Trigger** | File modifications |
| **Text Chunking** | 200 words, 50-word overlap |
| **Code Chunking** | AST-based via tree-sitter |

### File Operations

| Tool | Purpose |
|------|---------|
| `list_watched_files` | All files with embeddings |
| `get_file_watcher_stats` | Aggregate statistics |
| `delete_file_content` | Cleanup embeddings |
| `reconcile_files` | Orphan detection (dry-run capable) |

---

## 10. Hooks System

### Hook Types

| Hook | Trigger | Action |
|------|---------|--------|
| **SessionStart** | Session begins | Load topic portfolio, warm indexes |
| **UserPromptSubmit** | User sends prompt | Embed, search, detect divergence, inject context |
| **PreToolUse** | Before tool execution | Inject brief relevant context |
| **PostToolUse** | After tool execution | Capture + embed as HookDescription |
| **Stop** | Response complete | Capture response summary |
| **SessionEnd** | Session ends | Persist state, run HDBSCAN |

### Configuration

File: `.claude/settings.json`
```json
{
  "hooks": {
    "SessionStart": true,
    "UserPromptSubmit": true,
    "PreToolUse": true,
    "PostToolUse": true,
    "Stop": true,
    "SessionEnd": true
  }
}
```

---

## 11. Graph Linking (Typed Edges & K-NN)

### K-NN Graph Construction

| Parameter | Value |
|-----------|-------|
| **Algorithm** | NN-Descent |
| **k** | 20 (default) |
| **Sampling Rate** | ρ=0.5 |
| **Iterations** | 8 |
| **Min Similarity** | 0.3 |

### Typed Edge Types

| Edge Type | Source | Description |
|-----------|--------|-------------|
| **semantic_similar** | E1,E5,E6,E7,E10,E12,E13 | Semantic agreement |
| **code_related** | E7 | Code structure links |
| **entity_shared** | E11 | Entity relationships |
| **causal_chain** | E5 | Causal direction links |
| **graph_connected** | E8 | Structure/import links |
| **intent_aligned** | E10 | Goal/purpose alignment |
| **keyword_overlap** | E6/E13 | Sparse matching |
| **multi_agreement** | 3+ embedders | High confidence |

### Typed Edge Structure

```rust
TypedEdge {
    source: NodeId,
    target: NodeId,
    edge_types: HashSet<GraphLinkEdgeType>,
    embedder_agreement: HashMap<u8, f32>,
    direction: DirectedRelation,
    weight: f32,
}
```

---

## 12. Storage Backends

### RocksDB Architecture

**Primary Storage:** `/home/cabdru/contextgraph/contextgraph_data`

### Column Families (39 total)

| Category | Count | Examples |
|----------|-------|----------|
| **Base** | 11 | nodes, edges, embeddings, metadata |
| **Teleological** | 15 | fingerprints, topic_profiles, e13_splade_inverted |
| **Quantized Embedder** | 13 | emb_0 through emb_12 |
| **Code** | 5 | code_entities, code_e7_embeddings |

### Serialization

| Format | Use Case |
|--------|----------|
| **Bincode** | Embeddings and fingerprints |
| **MessagePack** | Metadata |
| **SafeTensors** | Model weights |

### Performance Targets

| Operation | p95 Target | p99 Target |
|-----------|------------|------------|
| inject_context | < 25ms | < 50ms |
| store_node | < 1ms | - |
| get_node | < 500μs | - |

---

## 13. Key Data Structures

### TeleologicalArray (SemanticFingerprint)

```rust
SemanticFingerprint {
    // Dense: 7,808 total dimensions
    e1_embedding: [f32; 1024],
    e2_embedding: [f32; 512],
    e3_embedding: [f32; 512],
    e4_embedding: [f32; 512],
    e5_embedding: [f32; 768],
    e7_embedding: [f32; 1536],
    e8_embedding: [f32; 384],
    e9_embedding: [f32; 1024],
    e10_embedding: [f32; 768],
    e11_embedding: [f32; 768],

    // Sparse: E6, E13
    e6_sparse: SparseVector,  // 30,522 vocab
    e13_sparse: SparseVector, // 30,522 vocab

    // Variable: E12
    e12_tokens: Vec<[f32; 128]>,  // Per-token
}
```

### MemoryNode

```rust
MemoryNode {
    id: Uuid,
    content: String,
    source: MemorySource,
    created_at: DateTime<Utc>,
    session_id: String,
    fingerprint: TeleologicalFingerprint,
    metadata: NodeMetadata,
    tags: Vec<String>,
    chunk_metadata: Option<ChunkMetadata>,
}
```

### Topic

```rust
Topic {
    id: Uuid,
    name: String,
    profile: TopicProfile,
    members: Vec<Uuid>,
    stability: TopicStability,
    weighted_agreement: f32,  // >= 2.5 threshold
    silhouette_score: f32,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}
```

---

## 14. Benchmark Capabilities

### Benchmark Suite

Located: `/crates/context-graph-benchmark/`

### Benchmark Categories

| Category | Benchmarks |
|----------|------------|
| **Core** | comparative_suite, scaling_analysis, realdata-bench |
| **GPU** | gpu-bench, embedder-stress |
| **Embedders** | e1-semantic, e7-realdata, e11-entity, e2-manual-verification |
| **Temporal** | temporal-bench, temporal-realdata, e4-hybrid-session |
| **Causal** | causal-realdata-bench |
| **Graph** | graph-bench, graph-realdata, graph-linking-bench |
| **MCP** | mcp-bench, mcp-intent-bench |
| **Validation** | validation-bench, validation-realdata |
| **Weight Profiles** | weight-profile-bench, embedder-impact-bench |

### Output

HTML reports in `/benchmark_results/`

---

## 15. System Architecture & Integration

### MCP JSON-RPC Protocol

| Transport | Description |
|-----------|-------------|
| **stdio** | Default |
| **TCP** | Network access |
| **SSE** | Server-Sent Events |

### Server Startup

```bash
RUST_LOG=warn \
CONTEXT_GRAPH_STORAGE_PATH=/home/cabdru/contextgraph/contextgraph_data \
CONTEXT_GRAPH_MODELS_PATH=/home/cabdru/contextgraph/models \
context-graph-mcp --daemon
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--config PATH` | Config file |
| `--transport tcp\|stdio` | Transport mode |
| `--port PORT` | TCP port |
| `--warm-first` | Block until models loaded (default) |
| `--daemon` | Daemon mode (shared MCP) |
| `--daemon-port PORT` | Daemon port (default 3199) |

### GPU-First Architecture

| Requirement | Description |
|-------------|-------------|
| **GPU** | RTX 5090 Blackwell (CUDA 13.x) |
| **Framework** | Candle (HuggingFace) |
| **Fallback** | NONE - fail-fast |

---

## 16. Constitution Compliance & Architecture Rules

### Key Architectural Rules

| Rule | Requirement |
|------|-------------|
| ARCH-01 | TeleologicalArray is atomic - all 13 embeddings or nothing |
| ARCH-02 | Apples-to-apples only - compare E1↔E1, never E1↔E5 |
| ARCH-04 | Temporal (E2-E4) NEVER count toward topics |
| ARCH-05 | All 13 embedders required - missing = fatal |
| ARCH-06 | All memory ops through MCP tools only |
| ARCH-09 | Topic threshold: weighted_agreement >= 2.5 |
| ARCH-10 | Divergence detection: SEMANTIC embedders only |
| ARCH-12 | E1 is foundation - all retrieval starts with E1 |
| ARCH-17 | Strong E1 (>0.8): enhancers refine. Weak E1 (<0.4): enhancers broaden |
| ARCH-18 | E5 Causal: asymmetric similarity |
| ARCH-21 | Multi-space fusion: Weighted RRF, not weighted sum |
| ARCH-25 | Temporal boosts POST-retrieval only |
| ARCH-28-30 | E10 multiplicative boost: NOT linear blending |

### Anti-Patterns (Forbidden)

| Anti-Pattern | Prohibition |
|--------------|-------------|
| AP-02 | No cross-embedder comparison (E1↔E5) |
| AP-04 | No partial TeleologicalArray |
| AP-05 | No embedding fusion into single vector |
| AP-60 | Temporal MUST NOT count toward topics |
| AP-73 | Temporal MUST NOT be used in similarity fusion |
| AP-74 | E12 ColBERT: reranking ONLY |
| AP-75 | E13 SPLADE: Stage 1 recall ONLY |
| AP-77 | E5 MUST NOT use symmetric cosine |
| AP-79 | MUST NOT use simple weighted sum |
| AP-80 | E10 MUST NOT use linear blending |
| AP-84 | E10 MUST NOT override E1 |

---

## 17. Performance & Optimization

### Storage Optimization

| Optimization | Description |
|--------------|-------------|
| **HNSW K-NN** | 13 independent indexes for O(log n) search |
| **Matryoshka E1** | 128D compressed index for fast filtering |
| **SPLADE Inverted Index** | E6/E13 inverted index for sparse recall |
| **Column Family Sharding** | 39 CFs for data locality |

### GPU Acceleration

| Feature | Description |
|---------|-------------|
| **Candle Framework** | HuggingFace ML inference |
| **Batch Similarity** | Batch compute embedder agreement |
| **Quantization** | FP16 for E12 token embeddings |
| **Memory Pooling** | Persistent GPU memory for models |

### Caching

| Cache | Description |
|-------|-------------|
| **LRU Cache** | O(1) embedding lookups (moka) |
| **CFT** | BIRCH incremental clustering |
| **File Watcher** | Incremental markdown chunking |

---

## Conclusion

The Context Graph system is a production-grade semantic memory platform implementing a sophisticated 13-embedder architecture for multi-perspective understanding. Key innovations include:

1. **No-fusion philosophy:** Preserves 67% more information vs traditional fused embeddings
2. **Asymmetric similarity:** E5 causal and E8 graph relationships with directional semantics
3. **Adaptive enhancement:** E10 multiplicative boost adapts to E1 quality
4. **Multi-space clustering:** Topics require 2.5+ weighted agreement across embedders
5. **GPU-first design:** RTX 5090 required, no CPU fallback
6. **Graph linking:** K-NN construction with typed edge detection from embedder agreement
7. **Code separation:** E7 embeddings stored separately with AST-aware chunking
8. **Entity knowledge:** KEPLER TransE relationships without keyword lookup

The system provides 43+ MCP tools for semantic search, causal reasoning, entity extraction, temporal navigation, and multi-embedder analysis, backed by RocksDB persistence and comprehensive benchmarking infrastructure.

---

*Generated by Claude Code for Context Graph v6.5.0*
