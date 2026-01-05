# Module 3: 12-Model Embedding Pipeline - Functional Specification

```xml
<functional_spec id="SPEC-EMBED" version="1.0">
<metadata>
  <title>12-Model Embedding Pipeline - Multi-Modal Semantic Representation System</title>
  <module>Module 3</module>
  <phase>2</phase>
  <status>draft</status>
  <owner>Lead Architect / CUDA Engineer</owner>
  <created>2025-12-31</created>
  <last_updated>2025-12-31</last_updated>
  <duration>4 weeks</duration>
  <related_specs>
    <spec_ref>SPEC-CORE (Module 2) - Provides MemoryNode.embedding field (1536D -> 4096D upgrade)</spec_ref>
    <spec_ref>SPEC-GRAPH (Module 4) - Consumes fused embeddings for FAISS index</spec_ref>
    <spec_ref>SPEC-UTL (Module 5) - Uses embedding similarity for surprise/coherence computation</spec_ref>
    <spec_ref>SPEC-BIO (Module 6) - Layer 1 (Sensing) integrates embedding pipeline</spec_ref>
  </related_specs>
  <dependencies>
    <dependency>Module 2 (Core Infrastructure) - Storage layer, MemoryNode struct, MCP server</dependency>
  </dependencies>
</metadata>

<overview>
The 12-Model Embedding Pipeline implements a sophisticated multi-model ensemble that captures diverse semantic, temporal, causal, and structural aspects of knowledge. Each model specializes in a specific embedding domain, and their outputs are fused via FuseMoE (Mixture of Experts with Laplace-smoothed gating) and CAME-AB (Cross-Attention Modality Encoding with Adaptive Bridging) to produce a unified 1536D representation.

Key deliverables:
1. **12 Specialized Embedding Models**: Semantic, Temporal (3 variants), Causal, Sparse, Code, Graph/GNN, HDC, Multimodal, Entity/TransE, Late-Interaction
2. **FuseMoE Fusion Layer**: Top-k=4 expert routing with laplace_alpha=0.01 load balancing
3. **CAME-AB Cross-Modality Bridge**: 8-head cross-attention with bridge_learning_rate=0.001
4. **Batching and Caching Infrastructure**: Request batching for GPU efficiency, LRU embedding cache
5. **Model Management**: Hot-swapping capability, per-model latency budgets, GPU memory management

This module upgrades MemoryNode.embedding from 1536D to 4096D internally, with FuseMoE projecting back to 1536D for storage and FAISS compatibility.

Performance targets:
- Single embedding: less than 200ms end-to-end
- Batch processing: greater than 100 items/sec
- Cache hit rate: greater than 80%
- GPU memory: less than 16GB for all models loaded
</overview>

<!-- ============================================================================ -->
<!-- USER STORIES -->
<!-- ============================================================================ -->

<user_stories>

<story id="US-EMBED-01" priority="must-have">
  <narrative>
    As an AI agent
    I want content to be embedded using multiple specialized models
    So that different aspects of meaning (semantic, temporal, causal) are captured
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-EMBED-01-01">
      <given>I provide text content for embedding</given>
      <when>The embedding pipeline processes it</when>
      <then>All 12 models generate their specialized embeddings</then>
    </criterion>
    <criterion id="AC-EMBED-01-02">
      <given>Content contains code snippets</given>
      <when>The embedding pipeline processes it</when>
      <then>The Code model (E7) captures AST-aware semantic information</then>
    </criterion>
    <criterion id="AC-EMBED-01-03">
      <given>Content references temporal relationships</given>
      <when>The embedding pipeline processes it</when>
      <then>Temporal models (E2, E3, E4) capture recency, periodicity, and position</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-EMBED-02" priority="must-have">
  <narrative>
    As an AI agent
    I want the 12 model embeddings fused into a unified representation
    So that I can perform efficient similarity search on a single vector
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-EMBED-02-01">
      <given>12 specialized embeddings have been generated</given>
      <when>FuseMoE fusion is applied</when>
      <then>A unified 1536D embedding is produced</then>
    </criterion>
    <criterion id="AC-EMBED-02-02">
      <given>FuseMoE routes to top-k experts</given>
      <when>I inspect the routing weights</when>
      <then>The routing decision is explainable (weights sum to 1.0)</then>
    </criterion>
    <criterion id="AC-EMBED-02-03">
      <given>Different content types (code vs prose)</given>
      <when>FuseMoE processes them</when>
      <then>Routing weights differ appropriately (code content routes more to E7)</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-EMBED-03" priority="must-have">
  <narrative>
    As a developer
    I want embedding computation to meet latency targets
    So that the system remains responsive for real-time use
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-EMBED-03-01">
      <given>A single content item for embedding</given>
      <when>The full pipeline executes</when>
      <then>End-to-end latency is less than 200ms</then>
    </criterion>
    <criterion id="AC-EMBED-03-02">
      <given>A batch of 64 content items</given>
      <when>The pipeline processes them in parallel</when>
      <then>Throughput exceeds 100 items per second</then>
    </criterion>
    <criterion id="AC-EMBED-03-03">
      <given>Each individual model</given>
      <when>Processing a single input</when>
      <then>Latency stays within its specified budget (E1: less than 5ms, E9: less than 1ms, etc.)</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-EMBED-04" priority="must-have">
  <narrative>
    As a developer
    I want embedding results to be cached
    So that repeated queries do not incur recomputation cost
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-EMBED-04-01">
      <given>Content has been embedded before</given>
      <when>The same content is submitted again</when>
      <then>The cached embedding is returned without model inference</then>
    </criterion>
    <criterion id="AC-EMBED-04-02">
      <given>The cache is under normal load</given>
      <when>Measuring cache performance</when>
      <then>Cache hit rate exceeds 80%</then>
    </criterion>
    <criterion id="AC-EMBED-04-03">
      <given>Cache capacity is reached</given>
      <when>New embeddings are computed</when>
      <then>LRU eviction removes least recently used entries</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-EMBED-05" priority="must-have">
  <narrative>
    As an operator
    I want to hot-swap embedding models without restart
    So that I can upgrade models without service interruption
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-EMBED-05-01">
      <given>A new version of an embedding model is available</given>
      <when>I trigger model hot-swap via configuration</when>
      <then>The new model is loaded and activated without dropping requests</then>
    </criterion>
    <criterion id="AC-EMBED-05-02">
      <given>A model swap is in progress</given>
      <when>Embedding requests arrive</when>
      <then>Requests queue and complete after swap (no errors)</then>
    </criterion>
    <criterion id="AC-EMBED-05-03">
      <given>Hot-swap fails (model load error)</given>
      <when>The system handles the failure</when>
      <then>Previous model remains active, error is logged</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-EMBED-06" priority="must-have">
  <narrative>
    As a developer
    I want GPU memory usage to stay under budget
    So that the system runs on target hardware (RTX 5090 / less than 16GB for embeddings)
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-EMBED-06-01">
      <given>All 12 embedding models are loaded</given>
      <when>Measuring GPU memory usage</when>
      <then>Total VRAM usage is less than 16GB</then>
    </criterion>
    <criterion id="AC-EMBED-06-02">
      <given>The system is under peak load</given>
      <when>Processing concurrent batches</when>
      <then>GPU memory does not exceed 20GB including working memory</then>
    </criterion>
    <criterion id="AC-EMBED-06-03">
      <given>GPU memory is constrained</given>
      <when>Loading models</when>
      <then>System uses quantization (FP8) to fit within budget</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-EMBED-07" priority="must-have">
  <narrative>
    As an AI agent
    I want CAME-AB cross-modality bridging
    So that information flows between different embedding modalities
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-EMBED-07-01">
      <given>Embeddings from different modalities</given>
      <when>CAME-AB processes them</when>
      <then>Cross-attention aligns modality-specific representations</then>
    </criterion>
    <criterion id="AC-EMBED-07-02">
      <given>CAME-AB is configured with 8 attention heads</given>
      <when>Processing multi-modal content</when>
      <then>Each head captures different cross-modal relationships</then>
    </criterion>
    <criterion id="AC-EMBED-07-03">
      <given>Bridge weights need adaptation</given>
      <when>Training signal is available</when>
      <then>Weights update with bridge_learning_rate=0.001</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-EMBED-08" priority="should-have">
  <narrative>
    As a developer
    I want to inspect individual model outputs before fusion
    So that I can debug and analyze embedding quality
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-EMBED-08-01">
      <given>Debug mode is enabled</given>
      <when>Embedding pipeline processes content</when>
      <then>Individual model outputs are available in response</then>
    </criterion>
    <criterion id="AC-EMBED-08-02">
      <given>FuseMoE routing is complete</given>
      <when>Debug mode is enabled</when>
      <then>Expert routing weights are visible and sum to 1.0</then>
    </criterion>
    <criterion id="AC-EMBED-08-03">
      <given>An embedding seems incorrect</given>
      <when>I query the debug endpoint</when>
      <then>Per-model latencies and contribution scores are provided</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-EMBED-09" priority="should-have">
  <narrative>
    As an AI agent
    I want sparse embeddings for keyword matching
    So that exact term overlap is captured alongside semantic similarity
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-EMBED-09-01">
      <given>Content with specific keywords</given>
      <when>E6 (Sparse/SPLADE) model processes it</when>
      <then>High-value dimensions correspond to keyword tokens</then>
    </criterion>
    <criterion id="AC-EMBED-09-02">
      <given>Sparse embedding is generated</given>
      <when>Measuring sparsity</when>
      <then>Approximately 5% of dimensions are active (non-zero)</then>
    </criterion>
    <criterion id="AC-EMBED-09-03">
      <given>Two documents with shared keywords</given>
      <when>Computing sparse similarity</when>
      <then>Keyword overlap increases similarity score</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-EMBED-10" priority="should-have">
  <narrative>
    As an AI agent
    I want late-interaction embeddings for fine-grained matching
    So that token-level similarity can be computed (ColBERT-style)
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-EMBED-10-01">
      <given>A query and document</given>
      <when>E12 (Late-Interaction) processes them</when>
      <then>Per-token embeddings (128D per token) are generated</then>
    </criterion>
    <criterion id="AC-EMBED-10-02">
      <given>Query and document token embeddings</given>
      <when>MaxSim operation is applied</when>
      <then>Fine-grained similarity score is computed</then>
    </criterion>
    <criterion id="AC-EMBED-10-03">
      <given>Long documents with partial matches</given>
      <when>Late-interaction scoring is used</when>
      <then>Partial matches are detected that dense embeddings might miss</then>
    </criterion>
  </acceptance_criteria>
</story>

</user_stories>

<!-- ============================================================================ -->
<!-- FUNCTIONAL REQUIREMENTS -->
<!-- ============================================================================ -->

<requirements>

<!-- ==================== E1: Semantic Embedding Model ==================== -->

<requirement id="REQ-EMBED-001" story_ref="US-EMBED-01" priority="must">
  <description>The system SHALL implement E1 (Semantic) embedding model producing 1024D dense vectors using a Transformer architecture with FP8 quantization.</description>
  <rationale>Semantic embeddings capture general meaning and are the foundation for similarity search per PRD section 3.</rationale>
  <acceptance_criteria>
    <criterion>Output dimension: 1024D dense vector</criterion>
    <criterion>Architecture: Dense Transformer (e.g., E5-Large or equivalent)</criterion>
    <criterion>Hardware acceleration: Tensor Core FP8</criterion>
    <criterion>Latency budget: less than 5ms per input</criterion>
    <criterion>Input: Text string (max 8192 tokens)</criterion>
    <criterion>Output normalized to unit length</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== E2: Temporal-Recent Embedding Model ==================== -->

<requirement id="REQ-EMBED-002" story_ref="US-EMBED-01" priority="must">
  <description>The system SHALL implement E2 (Temporal-Recent) embedding model producing 512D vectors using exponential decay weighting.</description>
  <rationale>Recent information should be emphasized for context-sensitive retrieval per PRD section 3.</rationale>
  <acceptance_criteria>
    <criterion>Output dimension: 512D vector</criterion>
    <criterion>Algorithm: Exponential decay based on timestamp</criterion>
    <criterion>Decay formula: weight = exp(-lambda * (now - timestamp))</criterion>
    <criterion>Lambda configurable (default: 0.1 per day)</criterion>
    <criterion>Hardware: Vector Unit (CUDA cores)</criterion>
    <criterion>Latency budget: less than 2ms per input</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== E3: Temporal-Periodic Embedding Model ==================== -->

<requirement id="REQ-EMBED-003" story_ref="US-EMBED-01" priority="must">
  <description>The system SHALL implement E3 (Temporal-Periodic) embedding model producing 512D vectors using Fourier basis functions.</description>
  <rationale>Periodic patterns (daily, weekly, yearly) should be captured for recurring events per PRD section 3.</rationale>
  <acceptance_criteria>
    <criterion>Output dimension: 512D vector</criterion>
    <criterion>Algorithm: Fourier basis decomposition</criterion>
    <criterion>Periods captured: hourly, daily, weekly, monthly, yearly</criterion>
    <criterion>Hardware: FFT Unit (CUDA FFT)</criterion>
    <criterion>Latency budget: less than 2ms per input</criterion>
    <criterion>Handles timezone-aware timestamps</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== E4: Temporal-Positional Embedding Model ==================== -->

<requirement id="REQ-EMBED-004" story_ref="US-EMBED-01" priority="must">
  <description>The system SHALL implement E4 (Temporal-Positional) embedding model producing 512D vectors using sinusoidal positional encoding.</description>
  <rationale>Absolute position in sequence is important for ordering relationships per PRD section 3.</rationale>
  <acceptance_criteria>
    <criterion>Output dimension: 512D vector</criterion>
    <criterion>Algorithm: Sinusoidal positional encoding (Transformer-style)</criterion>
    <criterion>Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...)</criterion>
    <criterion>Hardware: CUDA Core</criterion>
    <criterion>Latency budget: less than 2ms per input</criterion>
    <criterion>Position derived from creation order or explicit sequence</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== E5: Causal Embedding Model ==================== -->

<requirement id="REQ-EMBED-005" story_ref="US-EMBED-01" priority="must">
  <description>The system SHALL implement E5 (Causal) embedding model producing 768D vectors using Structural Causal Model (SCM) intervention encoding.</description>
  <rationale>Causal relationships are essential for reasoning about cause-effect chains per PRD section 3.</rationale>
  <acceptance_criteria>
    <criterion>Output dimension: 768D vector</criterion>
    <criterion>Algorithm: SCM intervention encoding</criterion>
    <criterion>Captures: do(X) intervention semantics</criterion>
    <criterion>Hardware: Tensor Core</criterion>
    <criterion>Latency budget: less than 8ms per input</criterion>
    <criterion>Encodes direction of causation (A causes B vs B causes A)</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== E6: Sparse Embedding Model ==================== -->

<requirement id="REQ-EMBED-006" story_ref="US-EMBED-09" priority="must">
  <description>The system SHALL implement E6 (Sparse) embedding model producing approximately 30K dimensional vectors with 5% sparsity using Top-K activation.</description>
  <rationale>Sparse embeddings enable keyword matching alongside dense similarity per PRD section 3.</rationale>
  <acceptance_criteria>
    <criterion>Output dimension: approximately 30,000 (vocabulary size)</criterion>
    <criterion>Sparsity: 5% active dimensions (approximately 1500 non-zero values)</criterion>
    <criterion>Algorithm: SPLADE-style Top-K activation</criterion>
    <criterion>Hardware: Sparse Tensor operations</criterion>
    <criterion>Latency budget: less than 3ms per input</criterion>
    <criterion>Non-zero values correspond to important tokens</criterion>
    <criterion>Stored in sparse format (indices + values)</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== E7: Code Embedding Model ==================== -->

<requirement id="REQ-EMBED-007" story_ref="US-EMBED-01" priority="must">
  <description>The system SHALL implement E7 (Code) embedding model producing 1536D vectors using an AST-aware Transformer architecture.</description>
  <rationale>Code requires specialized embedding that understands syntax and structure per PRD section 3.</rationale>
  <acceptance_criteria>
    <criterion>Output dimension: 1536D vector</criterion>
    <criterion>Architecture: AST-aware Transformer (e.g., CodeBERT, GraphCodeBERT)</criterion>
    <criterion>Hardware: Tensor Core FP16</criterion>
    <criterion>Latency budget: less than 10ms per input</criterion>
    <criterion>Languages supported: Python, JavaScript, TypeScript, Rust, Go, Java (minimum)</criterion>
    <criterion>Captures: function signatures, variable scopes, control flow</criterion>
    <criterion>Falls back to text embedding for unsupported languages</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== E8: Graph/GNN Embedding Model ==================== -->

<requirement id="REQ-EMBED-008" story_ref="US-EMBED-01" priority="must">
  <description>The system SHALL implement E8 (Graph/GNN) embedding model producing 1536D vectors using message passing neural networks.</description>
  <rationale>Graph structure embeddings capture relational context between entities per PRD section 3.</rationale>
  <acceptance_criteria>
    <criterion>Output dimension: 1536D vector</criterion>
    <criterion>Algorithm: Message Passing Neural Network (MPNN)</criterion>
    <criterion>Message passing iterations: 3 (configurable)</criterion>
    <criterion>Hardware: CUDA Graph optimizations</criterion>
    <criterion>Latency budget: less than 5ms per input</criterion>
    <criterion>Input: Node features + adjacency information from knowledge graph</criterion>
    <criterion>Aggregation: Mean/Max pooling over neighborhood</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== E9: HDC Embedding Model ==================== -->

<requirement id="REQ-EMBED-009" story_ref="US-EMBED-01" priority="must">
  <description>The system SHALL implement E9 (HDC - Hyperdimensional Computing) embedding model producing 10K-bit binary vectors using XOR binding and Hamming distance.</description>
  <rationale>HDC provides ultra-fast similarity computation with robust noise tolerance per PRD section 3.</rationale>
  <acceptance_criteria>
    <criterion>Output dimension: 10,000 bits (binary vector)</criterion>
    <criterion>Operations: XOR binding, majority bundling</criterion>
    <criterion>Distance metric: Hamming distance</criterion>
    <criterion>Hardware: Vector Unit (bitwise operations)</criterion>
    <criterion>Latency budget: less than 1ms per input</criterion>
    <criterion>Noise tolerance: greater than 20% bit flip recovery</criterion>
    <criterion>Storage: Packed binary format (1250 bytes)</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== E10: Multimodal Embedding Model ==================== -->

<requirement id="REQ-EMBED-010" story_ref="US-EMBED-01" priority="must">
  <description>The system SHALL implement E10 (Multimodal) embedding model producing 1024D vectors using cross-attention between modalities.</description>
  <rationale>Content may reference images, audio, or other modalities that need unified representation per PRD section 3.</rationale>
  <acceptance_criteria>
    <criterion>Output dimension: 1024D vector</criterion>
    <criterion>Algorithm: Cross-attention fusion (CLIP-style)</criterion>
    <criterion>Supported modalities: text, image references, audio references</criterion>
    <criterion>Hardware: Tensor Core</criterion>
    <criterion>Latency budget: less than 15ms per input</criterion>
    <criterion>Falls back to text-only when non-text modalities absent</criterion>
    <criterion>Aligns embeddings across modality spaces</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== E11: Entity/TransE Embedding Model ==================== -->

<requirement id="REQ-EMBED-011" story_ref="US-EMBED-01" priority="must">
  <description>The system SHALL implement E11 (Entity/TransE) embedding model producing 256D vectors using translation-based knowledge graph embedding (h + r approximately equals t).</description>
  <rationale>Entity embeddings enable structured knowledge graph reasoning per PRD section 3.</rationale>
  <acceptance_criteria>
    <criterion>Output dimension: 256D vector</criterion>
    <criterion>Algorithm: TransE (h + r approximately equals t)</criterion>
    <criterion>Entity extraction: Named entity recognition</criterion>
    <criterion>Relation encoding: Learned relation vectors</criterion>
    <criterion>Hardware: CUDA Core</criterion>
    <criterion>Latency budget: less than 2ms per input</criterion>
    <criterion>Supports: entity-to-entity similarity, relation inference</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== E12: Late-Interaction Embedding Model ==================== -->

<requirement id="REQ-EMBED-012" story_ref="US-EMBED-10" priority="must">
  <description>The system SHALL implement E12 (Late-Interaction) embedding model producing 128D per-token vectors using ColBERT-style MaxSim scoring.</description>
  <rationale>Late interaction enables fine-grained token-level matching per PRD section 3.</rationale>
  <acceptance_criteria>
    <criterion>Output dimension: 128D per token</criterion>
    <criterion>Algorithm: ColBERT MaxSim</criterion>
    <criterion>Scoring: max(sim(q_i, d_j)) over all document tokens</criterion>
    <criterion>Hardware: CUDA Tile operations</criterion>
    <criterion>Latency budget: less than 8ms per input</criterion>
    <criterion>Max tokens: 512 (query), 8192 (document)</criterion>
    <criterion>Supports: passage retrieval, answer extraction</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Multi-Array Teleological Storage ==================== -->
<!-- NOTE: FuseMoE fusion is DEPRECATED. The system uses Multi-Array Storage where -->
<!-- all 13 embeddings (E1-E13) are stored separately. NO FUSION/CONCATENATION. -->

<requirement id="REQ-EMBED-013" story_ref="US-EMBED-02" priority="must">
  <description>The system SHALL implement Multi-Array Teleological Storage preserving all 13 embeddings (E1-E12 + E13 SPLADE) as separate arrays.</description>
  <rationale>Multi-array storage preserves 100% semantic information vs ~33% with top-k fusion. The 13-embedding array IS the teleological vector per PRD Royse 2026.</rationale>
  <acceptance_criteria>
    <criterion>Embedding count: 13 (E1-E12 dense + E13 SPLADE sparse)</criterion>
    <criterion>Storage: All 13 embeddings stored as separate arrays (SemanticFingerprint)</criterion>
    <criterion>NO FUSION: Embeddings are NOT combined into single vector</criterion>
    <criterion>Quantization: PQ-8, Float8, Binary, Sparse per embedder (~17KB total)</criterion>
    <criterion>Information preserved: 100% (vs ~33% with deprecated FuseMoE)</criterion>
    <criterion>Per-space indexes: 13 HNSW indexes (one per embedding space)</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-EMBED-014" story_ref="US-EMBED-02" priority="must">
  <description>The system SHALL implement 5-Stage Retrieval Pipeline for multi-space search.</description>
  <rationale>5-stage pipeline enables efficient search across 13 embedding spaces with <60ms latency at 1M memories.</rationale>
  <acceptance_criteria>
    <criterion>Stage 1: SPLADE sparse pre-filter (E13) → 10K candidates in <5ms</criterion>
    <criterion>Stage 2: Matryoshka 128D ANN (E1 truncated) → 1K candidates in <10ms</criterion>
    <criterion>Stage 3: RRF multi-space rerank (E1-E13) → 100 candidates in <20ms</criterion>
    <criterion>Stage 4: Teleological alignment filter (purpose vector) → 50 candidates in <10ms</criterion>
    <criterion>Stage 5: MaxSim late interaction (E12) → final top-10 in <15ms</criterion>
    <criterion>Total latency: <60ms at 1M memories</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-EMBED-015" story_ref="US-EMBED-02, US-EMBED-08" priority="must">
  <description>The system SHALL implement RRF (Reciprocal Rank Fusion) for multi-space similarity scoring.</description>
  <rationale>RRF fuses SCORES from per-space similarity, not vectors. This enables query-specific weighting per US-EMBED-08.</rationale>
  <acceptance_criteria>
    <criterion>RRF formula: RRF(d) = Σᵢ wᵢ/(k + rankᵢ(d)) where k=60</criterion>
    <criterion>Per-space similarity computed independently (cosine, asymmetric, MaxSim, etc.)</criterion>
    <criterion>Query-type weighting: semantic_search, causal_reasoning, code_search, etc.</criterion>
    <criterion>E5 Causal: asymmetric similarity (cause→effect=1.2×, effect→cause=0.8×)</criterion>
    <criterion>Explainable: per-space scores available for debugging</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== CAME-AB Cross-Modality ==================== -->

<requirement id="REQ-EMBED-016" story_ref="US-EMBED-07" priority="must">
  <description>The system SHALL implement CAME-AB (Cross-Attention Modality Encoding with Adaptive Bridging) with 8 attention heads and bridge_learning_rate=0.001.</description>
  <rationale>CAME-AB enables information flow between different embedding modalities per PRD section 3.2.</rationale>
  <acceptance_criteria>
    <criterion>Attention heads: 8</criterion>
    <criterion>Bridge learning rate: 0.001</criterion>
    <criterion>Cross-attention between all modality pairs</criterion>
    <criterion>Residual connections for stability</criterion>
    <criterion>Layer normalization after cross-attention</criterion>
    <criterion>Output: Modality-aligned representations</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-EMBED-017" story_ref="US-EMBED-07" priority="must">
  <description>The system SHALL implement adaptive bridge weights that learn optimal cross-modality mappings.</description>
  <rationale>Adaptive bridges enable the system to learn which modality combinations are most informative.</rationale>
  <acceptance_criteria>
    <criterion>Bridge weight matrix: 12x12 (all modality pairs)</criterion>
    <criterion>Weights initialized to uniform (1/12)</criterion>
    <criterion>Gradient updates with bridge_learning_rate=0.001</criterion>
    <criterion>Regularization: L2 penalty on bridge weights</criterion>
    <criterion>Sparsification: Prune weak bridges (less than 0.01)</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Batching Infrastructure ==================== -->

<requirement id="REQ-EMBED-018" story_ref="US-EMBED-03" priority="must">
  <description>The system SHALL implement request batching for GPU efficiency with configurable batch size and timeout.</description>
  <rationale>Batching amortizes GPU overhead for high throughput per PRD section 13.</rationale>
  <acceptance_criteria>
    <criterion>Default batch size: 64</criterion>
    <criterion>Maximum batch size: 256</criterion>
    <criterion>Batch timeout: 10ms (flush partial batch)</criterion>
    <criterion>Dynamic batching: Collect requests until batch full or timeout</criterion>
    <criterion>Per-model batching: Each model processes its batch in parallel</criterion>
    <criterion>Throughput target: greater than 100 items/sec at batch size 64</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-EMBED-019" story_ref="US-EMBED-03" priority="must">
  <description>The system SHALL implement async embedding computation with non-blocking request handling.</description>
  <rationale>Async processing prevents blocking the MCP server during embedding computation.</rationale>
  <acceptance_criteria>
    <criterion>Embedding requests return immediately with future/promise</criterion>
    <criterion>Background worker threads process batches</criterion>
    <criterion>CUDA streams for parallel model execution</criterion>
    <criterion>Request queue with bounded capacity (default 10000)</criterion>
    <criterion>Backpressure when queue full (return rate limit error)</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Caching Infrastructure ==================== -->

<requirement id="REQ-EMBED-020" story_ref="US-EMBED-04" priority="must">
  <description>The system SHALL implement LRU embedding cache with configurable capacity and TTL.</description>
  <rationale>Caching avoids recomputation for frequently accessed content per PRD section 13.</rationale>
  <acceptance_criteria>
    <criterion>Cache key: Content hash (SHA-256)</criterion>
    <criterion>Cache value: Full embedding result (1536D fused + routing metadata)</criterion>
    <criterion>Default capacity: 100,000 entries</criterion>
    <criterion>TTL: 24 hours (configurable)</criterion>
    <criterion>Eviction: LRU when capacity reached</criterion>
    <criterion>Target hit rate: greater than 80%</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-EMBED-021" story_ref="US-EMBED-04" priority="must">
  <description>The system SHALL expose cache metrics for monitoring and optimization.</description>
  <rationale>Cache metrics enable capacity planning and performance tuning.</rationale>
  <acceptance_criteria>
    <criterion>Metrics exposed: hit_count, miss_count, hit_rate, eviction_count</criterion>
    <criterion>Metrics exposed: cache_size_bytes, entry_count, avg_entry_size</criterion>
    <criterion>Prometheus-compatible metrics endpoint</criterion>
    <criterion>Cache warmup API for preloading common embeddings</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Model Hot-Swapping ==================== -->

<requirement id="REQ-EMBED-022" story_ref="US-EMBED-05" priority="must">
  <description>The system SHALL support hot-swapping embedding models without service restart.</description>
  <rationale>Hot-swap enables model upgrades without downtime per US-EMBED-05.</rationale>
  <acceptance_criteria>
    <criterion>Swap triggered via configuration change or API call</criterion>
    <criterion>New model loaded in background</criterion>
    <criterion>Traffic redirected atomically after load complete</criterion>
    <criterion>Requests queue during swap (no drops)</criterion>
    <criterion>Rollback if new model fails to load</criterion>
    <criterion>Maximum swap duration: 60 seconds</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-EMBED-023" story_ref="US-EMBED-05" priority="must">
  <description>The system SHALL validate model compatibility before completing hot-swap.</description>
  <rationale>Incompatible models would break downstream consumers.</rationale>
  <acceptance_criteria>
    <criterion>Validate output dimension matches expected</criterion>
    <criterion>Validate input format compatibility</criterion>
    <criterion>Run test embedding on validation inputs</criterion>
    <criterion>Compare output distribution to reference</criterion>
    <criterion>Reject swap if validation fails</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== GPU Memory Management ==================== -->

<requirement id="REQ-EMBED-024" story_ref="US-EMBED-06" priority="must">
  <description>The system SHALL manage GPU memory to stay within 16GB budget for all embedding models.</description>
  <rationale>Memory budget ensures compatibility with target hardware per PRD section 13.</rationale>
  <acceptance_criteria>
    <criterion>Total model VRAM: less than 16GB</criterion>
    <criterion>Working memory buffer: 4GB reserved</criterion>
    <criterion>Memory pool for allocation efficiency</criterion>
    <criterion>Automatic quantization if memory exceeded</criterion>
    <criterion>Model offloading to CPU if critical</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-EMBED-025" story_ref="US-EMBED-06" priority="must">
  <description>The system SHALL use FP8 quantization for eligible models to reduce memory footprint.</description>
  <rationale>FP8 reduces memory by 4x with minimal accuracy loss per PRD section 13.</rationale>
  <acceptance_criteria>
    <criterion>E1 (Semantic): FP8 on Tensor Cores</criterion>
    <criterion>E5 (Causal): FP8 optional</criterion>
    <criterion>E7 (Code): FP16 (higher precision for code)</criterion>
    <criterion>Quantization configurable per model</criterion>
    <criterion>Accuracy validation after quantization</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Performance Budgets ==================== -->

<requirement id="REQ-EMBED-026" story_ref="US-EMBED-03" priority="must">
  <description>Each embedding model SHALL meet its individual latency budget as specified in the constitution.</description>
  <rationale>Individual budgets ensure end-to-end latency target is achievable.</rationale>
  <acceptance_criteria>
    <criterion>E1 (Semantic): less than 5ms</criterion>
    <criterion>E2 (Temporal-Recent): less than 2ms</criterion>
    <criterion>E3 (Temporal-Periodic): less than 2ms</criterion>
    <criterion>E4 (Temporal-Positional): less than 2ms</criterion>
    <criterion>E5 (Causal): less than 8ms</criterion>
    <criterion>E6 (Sparse): less than 3ms</criterion>
    <criterion>E7 (Code): less than 10ms</criterion>
    <criterion>E8 (Graph/GNN): less than 5ms</criterion>
    <criterion>E9 (HDC): less than 1ms</criterion>
    <criterion>E10 (Multimodal): less than 15ms</criterion>
    <criterion>E11 (Entity/TransE): less than 2ms</criterion>
    <criterion>E12 (Late-Interaction): less than 8ms</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-EMBED-027" story_ref="US-EMBED-03" priority="must">
  <description>The system SHALL meet end-to-end latency of less than 200ms for single input and throughput of greater than 100 items/sec for batch.</description>
  <rationale>End-to-end performance targets per PRD section 13.</rationale>
  <acceptance_criteria>
    <criterion>Single input P95 latency: less than 200ms</criterion>
    <criterion>Batch throughput (size 64): greater than 100 items/sec</criterion>
    <criterion>Cache hit latency: less than 1ms</criterion>
    <criterion>FuseMoE fusion latency: less than 3ms</criterion>
    <criterion>CAME-AB bridging latency: less than 5ms</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Integration with Core Infrastructure ==================== -->

<requirement id="REQ-EMBED-028" story_ref="US-EMBED-02" priority="must">
  <description>The system SHALL upgrade MemoryNode.embedding from 1536D (Module 2) to 4096D internal representation with 1536D output.</description>
  <rationale>Higher internal dimension improves fusion quality while maintaining FAISS compatibility.</rationale>
  <acceptance_criteria>
    <criterion>Internal fusion dimension: 4096D</criterion>
    <criterion>Output projection: 4096D -> 1536D linear layer</criterion>
    <criterion>MemoryNode.embedding remains 1536D for storage</criterion>
    <criterion>Backward compatible with Module 2 stored embeddings</criterion>
    <criterion>Migration path for existing embeddings (re-embed on access)</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-EMBED-029" story_ref="US-EMBED-01" priority="must">
  <description>The system SHALL implement EmbeddingProvider trait from Module 1 Ghost System.</description>
  <rationale>Trait implementation enables drop-in replacement of stub embeddings.</rationale>
  <acceptance_criteria>
    <criterion>Implements: EmbeddingProvider::embed(content) -> Vec of f32</criterion>
    <criterion>Implements: EmbeddingProvider::embed_batch(contents) -> Vec of Vec of f32</criterion>
    <criterion>Implements: EmbeddingProvider::dimension() -> 1536</criterion>
    <criterion>Replaces stub provider from Module 1</criterion>
    <criterion>Thread-safe implementation</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Error Handling ==================== -->

<requirement id="REQ-EMBED-030" story_ref="US-EMBED-01, US-EMBED-03" priority="must">
  <description>The system SHALL handle embedding failures gracefully with fallback strategies.</description>
  <rationale>Embedding failures should not cascade to system-wide failures.</rationale>
  <acceptance_criteria>
    <criterion>Individual model failure: Use remaining models for fusion</criterion>
    <criterion>All models fail: Return cached result if available, else error</criterion>
    <criterion>Timeout exceeded: Return partial result with warning</criterion>
    <criterion>GPU OOM: Retry with smaller batch, then CPU fallback</criterion>
    <criterion>All failures logged with model ID and error details</criterion>
  </acceptance_criteria>
</requirement>

</requirements>

<!-- ============================================================================ -->
<!-- EDGE CASES -->
<!-- ============================================================================ -->

<edge_cases>

<edge_case id="EC-EMBED-001" req_ref="REQ-EMBED-001">
  <scenario>Input text exceeds maximum token limit (8192 tokens)</scenario>
  <expected_behavior>Truncate to max tokens with warning. Log original length. Include truncation flag in metadata.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-002" req_ref="REQ-EMBED-007">
  <scenario>Code content in unsupported programming language</scenario>
  <expected_behavior>Fall back to E1 (Semantic) embedding. Log unsupported language. Return language_fallback=true in metadata.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-003" req_ref="REQ-EMBED-006">
  <scenario>Sparse embedding produces zero active dimensions</scenario>
  <expected_behavior>Use uniform distribution over vocabulary. Log anomaly. Return sparsity_fallback=true.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-004" req_ref="REQ-EMBED-013">
  <scenario>FuseMoE router assigns zero weight to all experts</scenario>
  <expected_behavior>Use uniform weighting (1/12 each). Log routing failure. Return routing_fallback=true.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-005" req_ref="REQ-EMBED-018">
  <scenario>Batch timeout fires with single item in queue</scenario>
  <expected_behavior>Process single-item batch. This is expected behavior - no padding needed.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-006" req_ref="REQ-EMBED-019">
  <scenario>Request queue reaches capacity (10000)</scenario>
  <expected_behavior>Return error -32004 (RateLimitExceeded). Log queue depth. Caller should retry with backoff.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-007" req_ref="REQ-EMBED-020">
  <scenario>Cache key collision (two different contents produce same SHA-256)</scenario>
  <expected_behavior>Statistically impossible (1 in 2^256). If detected, log critical error and recompute.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-008" req_ref="REQ-EMBED-022">
  <scenario>Hot-swap initiated while previous swap in progress</scenario>
  <expected_behavior>Queue new swap request. Complete current swap first. Return swap_queued status.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-009" req_ref="REQ-EMBED-024">
  <scenario>GPU memory exhausted during batch processing</scenario>
  <expected_behavior>Retry with half batch size. If still fails, offload to CPU. Log memory pressure event.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-010" req_ref="REQ-EMBED-009">
  <scenario>HDC embedding receives empty input</scenario>
  <expected_behavior>Return zero vector (all bits 0). Log empty input warning.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-011" req_ref="REQ-EMBED-008">
  <scenario>Graph embedding requested but node has no neighbors</scenario>
  <expected_behavior>Return self-embedding only (no message passing). Log isolated node.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-012" req_ref="REQ-EMBED-012">
  <scenario>Late-interaction query longer than document</scenario>
  <expected_behavior>Swap query and document for MaxSim computation. Return with swap_flag=true.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-013" req_ref="REQ-EMBED-026">
  <scenario>Individual model exceeds latency budget</scenario>
  <expected_behavior>Continue with available results. Mark slow model in response. Emit latency_exceeded metric.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-014" req_ref="REQ-EMBED-002, REQ-EMBED-003, REQ-EMBED-004">
  <scenario>Temporal embedding requested for content without timestamp</scenario>
  <expected_behavior>Use current timestamp as default. Log missing_timestamp. Return timestamp_defaulted=true.</expected_behavior>
</edge_case>

<edge_case id="EC-EMBED-015" req_ref="REQ-EMBED-011">
  <scenario>Entity embedding finds no named entities</scenario>
  <expected_behavior>Return zero vector for E11. Other models still contribute to fusion.</expected_behavior>
</edge_case>

</edge_cases>

<!-- ============================================================================ -->
<!-- ERROR STATES -->
<!-- ============================================================================ -->

<error_states>

<error id="ERR-EMBED-001" json_rpc_code="-32003">
  <condition>All embedding models fail to produce output</condition>
  <message>Embedding error: All models failed - {details}</message>
  <recovery>Check GPU status, model files, and memory. Retry with smaller input.</recovery>
  <logging>Log each model's failure reason, GPU state, input size</logging>
</error>

<error id="ERR-EMBED-002" json_rpc_code="-32004">
  <condition>Embedding request queue at capacity</condition>
  <message>Rate limit exceeded: Embedding queue full (10000 pending)</message>
  <recovery>Retry with exponential backoff. Consider request batching on client side.</recovery>
  <logging>Log queue depth, request rate, oldest pending request age</logging>
</error>

<error id="ERR-EMBED-003" json_rpc_code="-32003">
  <condition>GPU out of memory during embedding computation</condition>
  <message>Embedding error: GPU memory exhausted</message>
  <recovery>Reduce batch size. Enable model quantization. Consider CPU fallback.</recovery>
  <logging>Log GPU memory state, batch size, model memory breakdown</logging>
</error>

<error id="ERR-EMBED-004" json_rpc_code="-32003">
  <condition>Model hot-swap fails</condition>
  <message>Model swap failed: {model_id} - {reason}</message>
  <recovery>Previous model remains active. Fix new model issue and retry swap.</recovery>
  <logging>Log model ID, file path, validation failure details</logging>
</error>

<error id="ERR-EMBED-005" json_rpc_code="-32602">
  <condition>Invalid input format for embedding</condition>
  <message>Invalid params: Embedding input must be non-empty string</message>
  <recovery>Provide valid text content for embedding</recovery>
  <logging>Log input type received, expected type</logging>
</error>

<error id="ERR-EMBED-006" json_rpc_code="-32003">
  <condition>Embedding timeout exceeded</condition>
  <message>Embedding error: Operation timed out after {timeout_ms}ms</message>
  <recovery>Retry with smaller input. Check GPU utilization.</recovery>
  <logging>Log timeout value, models that completed, models that timed out</logging>
</error>

<error id="ERR-EMBED-007" json_rpc_code="-32003">
  <condition>Model file not found or corrupted</condition>
  <message>Embedding error: Model {model_id} not found at {path}</message>
  <recovery>Verify model files exist. Re-download if corrupted.</recovery>
  <logging>Log model ID, expected path, file hash if available</logging>
</error>

<error id="ERR-EMBED-008" json_rpc_code="-32003">
  <condition>CUDA runtime error during embedding</condition>
  <message>Embedding error: CUDA error {code} - {description}</message>
  <recovery>Restart embedding service. Check GPU driver and CUDA installation.</recovery>
  <logging>Log CUDA error code, kernel name, GPU state</logging>
</error>

</error_states>

<!-- ============================================================================ -->
<!-- TEST PLAN -->
<!-- ============================================================================ -->

<test_plan>

<!-- Unit Tests: Individual Embedding Models -->

<test_case id="TC-EMBED-001" type="unit" req_ref="REQ-EMBED-001">
  <description>E1 (Semantic) produces 1024D normalized vector</description>
  <preconditions>E1 model loaded</preconditions>
  <inputs>{"content": "The quick brown fox jumps over the lazy dog"}</inputs>
  <expected>1024D vector with L2 norm approximately 1.0</expected>
  <data_requirements>Use REAL model inference, NO mock embeddings</data_requirements>
</test_case>

<test_case id="TC-EMBED-002" type="unit" req_ref="REQ-EMBED-002">
  <description>E2 (Temporal-Recent) applies exponential decay correctly</description>
  <preconditions>E2 model loaded</preconditions>
  <inputs>{"content": "test", "timestamp": "1 day ago"}, {"content": "test", "timestamp": "7 days ago"}</inputs>
  <expected>Recent embedding has higher magnitude in temporal dimensions</expected>
  <data_requirements>Use REAL model with actual timestamps</data_requirements>
</test_case>

<test_case id="TC-EMBED-003" type="unit" req_ref="REQ-EMBED-003">
  <description>E3 (Temporal-Periodic) captures daily/weekly patterns</description>
  <preconditions>E3 model loaded</preconditions>
  <inputs>{"content": "meeting", "timestamps": ["Monday 9am", "Monday 9am next week", "Tuesday 9am"]}</inputs>
  <expected>Same day/time embeddings more similar than different day</expected>
  <data_requirements>Use REAL Fourier basis computation</data_requirements>
</test_case>

<test_case id="TC-EMBED-004" type="unit" req_ref="REQ-EMBED-004">
  <description>E4 (Temporal-Positional) encodes sequence position</description>
  <preconditions>E4 model loaded</preconditions>
  <inputs>{"content": "item", "positions": [1, 2, 100, 101]}</inputs>
  <expected>Adjacent positions (1,2) and (100,101) more similar than distant (1,100)</expected>
  <data_requirements>Use REAL sinusoidal encoding</data_requirements>
</test_case>

<test_case id="TC-EMBED-005" type="unit" req_ref="REQ-EMBED-005">
  <description>E5 (Causal) encodes intervention semantics</description>
  <preconditions>E5 model loaded</preconditions>
  <inputs>{"content": "Rain causes wet ground"}, {"content": "Wet ground causes rain"}</inputs>
  <expected>Different embeddings for reversed causation direction</expected>
  <data_requirements>Use REAL SCM encoding</data_requirements>
</test_case>

<test_case id="TC-EMBED-006" type="unit" req_ref="REQ-EMBED-006">
  <description>E6 (Sparse) produces 5% sparsity</description>
  <preconditions>E6 model loaded</preconditions>
  <inputs>{"content": "machine learning neural network deep learning"}</inputs>
  <expected>Approximately 1500 non-zero values in 30K vector. Non-zero indices correspond to input tokens.</expected>
  <data_requirements>Use REAL SPLADE model</data_requirements>
</test_case>

<test_case id="TC-EMBED-007" type="unit" req_ref="REQ-EMBED-007">
  <description>E7 (Code) captures AST structure</description>
  <preconditions>E7 model loaded</preconditions>
  <inputs>{"content": "def foo(x): return x * 2", "language": "python"}</inputs>
  <expected>Embedding captures function definition, parameter, return statement</expected>
  <data_requirements>Use REAL AST-aware model</data_requirements>
</test_case>

<test_case id="TC-EMBED-008" type="unit" req_ref="REQ-EMBED-008">
  <description>E8 (Graph/GNN) incorporates neighbor information</description>
  <preconditions>E8 model loaded, node with 3 neighbors in graph</preconditions>
  <inputs>{"node_id": "...", "neighbors": ["n1", "n2", "n3"]}</inputs>
  <expected>Embedding differs from isolated node embedding</expected>
  <data_requirements>Use REAL message passing computation</data_requirements>
</test_case>

<test_case id="TC-EMBED-009" type="unit" req_ref="REQ-EMBED-009">
  <description>E9 (HDC) produces 10K-bit binary vector</description>
  <preconditions>E9 model loaded</preconditions>
  <inputs>{"content": "hyperdimensional computing test"}</inputs>
  <expected>10000 bits, Hamming distance meaningful for similarity</expected>
  <data_requirements>Use REAL XOR binding operations</data_requirements>
</test_case>

<test_case id="TC-EMBED-010" type="unit" req_ref="REQ-EMBED-010">
  <description>E10 (Multimodal) aligns text and image references</description>
  <preconditions>E10 model loaded</preconditions>
  <inputs>{"content": "A photo of a cat", "modalities": ["text", "image_ref"]}</inputs>
  <expected>Cross-modal alignment produces coherent 1024D embedding</expected>
  <data_requirements>Use REAL cross-attention model</data_requirements>
</test_case>

<test_case id="TC-EMBED-011" type="unit" req_ref="REQ-EMBED-011">
  <description>E11 (Entity/TransE) encodes entity relationships</description>
  <preconditions>E11 model loaded</preconditions>
  <inputs>{"content": "Paris is the capital of France"}</inputs>
  <expected>h(Paris) + r(capital_of) approximately equals t(France)</expected>
  <data_requirements>Use REAL TransE computation</data_requirements>
</test_case>

<test_case id="TC-EMBED-012" type="unit" req_ref="REQ-EMBED-012">
  <description>E12 (Late-Interaction) produces per-token embeddings</description>
  <preconditions>E12 model loaded</preconditions>
  <inputs>{"query": "what is machine learning", "document": "Machine learning is a subset of AI..."}</inputs>
  <expected>128D per token, MaxSim score computed correctly</expected>
  <data_requirements>Use REAL ColBERT model</data_requirements>
</test_case>

<!-- Integration Tests: FuseMoE -->

<test_case id="TC-EMBED-013" type="integration" req_ref="REQ-EMBED-013">
  <description>FuseMoE produces 1536D fused embedding</description>
  <preconditions>All 12 models loaded, FuseMoE initialized</preconditions>
  <inputs>{"content": "Test content for fusion"}</inputs>
  <expected>1536D fused vector, routing weights sum to 1.0</expected>
  <data_requirements>Use REAL model outputs for fusion</data_requirements>
</test_case>

<test_case id="TC-EMBED-014" type="integration" req_ref="REQ-EMBED-014">
  <description>FuseMoE routes to top-4 experts</description>
  <preconditions>FuseMoE with gating network</preconditions>
  <inputs>{"content": "def foo(): pass"}</inputs>
  <expected>Exactly 4 experts selected, code expert (E7) likely in top-4</expected>
  <data_requirements>Use REAL gating network inference</data_requirements>
</test_case>

<test_case id="TC-EMBED-015" type="integration" req_ref="REQ-EMBED-015">
  <description>FuseMoE routing weights are inspectable</description>
  <preconditions>FuseMoE with debug mode</preconditions>
  <inputs>{"content": "test", "debug": true}</inputs>
  <expected>Response includes routing_weights array of 12 floats summing to 1.0</expected>
  <data_requirements>Use REAL routing computation</data_requirements>
</test_case>

<!-- Integration Tests: CAME-AB -->

<test_case id="TC-EMBED-016" type="integration" req_ref="REQ-EMBED-016">
  <description>CAME-AB produces modality-aligned representations</description>
  <preconditions>CAME-AB initialized with 8 heads</preconditions>
  <inputs>{"modality_embeddings": [E1_out, E7_out, E10_out]}</inputs>
  <expected>Cross-attention aligns different modality outputs</expected>
  <data_requirements>Use REAL cross-attention computation</data_requirements>
</test_case>

<test_case id="TC-EMBED-017" type="integration" req_ref="REQ-EMBED-017">
  <description>CAME-AB bridge weights are learnable</description>
  <preconditions>CAME-AB in training mode</preconditions>
  <inputs>Training signal from downstream task</inputs>
  <expected>Bridge weights update with learning_rate=0.001</expected>
  <data_requirements>Use REAL gradient computation</data_requirements>
</test_case>

<!-- Integration Tests: Batching -->

<test_case id="TC-EMBED-018" type="integration" req_ref="REQ-EMBED-018">
  <description>Request batching collects items until batch full</description>
  <preconditions>Batch size 64, timeout 10ms</preconditions>
  <inputs>Submit 100 requests in quick succession</inputs>
  <expected>Two batches processed: 64 + 36</expected>
  <data_requirements>Use REAL batching infrastructure</data_requirements>
</test_case>

<test_case id="TC-EMBED-019" type="integration" req_ref="REQ-EMBED-019">
  <description>Async embedding returns future immediately</description>
  <preconditions>Embedding service running</preconditions>
  <inputs>{"content": "test"}</inputs>
  <expected>Request returns in less than 1ms with future. Future resolves in less than 200ms.</expected>
  <data_requirements>Use REAL async infrastructure</data_requirements>
</test_case>

<!-- Integration Tests: Caching -->

<test_case id="TC-EMBED-020" type="integration" req_ref="REQ-EMBED-020">
  <description>Cache hit returns result without model inference</description>
  <preconditions>Content previously embedded</preconditions>
  <inputs>Same content as previous request</inputs>
  <expected>Cache hit, latency less than 1ms, no GPU utilization</expected>
  <data_requirements>Use REAL cache with instrumentation</data_requirements>
</test_case>

<test_case id="TC-EMBED-021" type="integration" req_ref="REQ-EMBED-021">
  <description>Cache metrics are accurate</description>
  <preconditions>Cache with known state</preconditions>
  <inputs>10 cache hits, 5 cache misses</inputs>
  <expected>Metrics show hit_rate=66.7%, hit_count=10, miss_count=5</expected>
  <data_requirements>Use REAL metrics collection</data_requirements>
</test_case>

<!-- Integration Tests: Hot-Swap -->

<test_case id="TC-EMBED-022" type="integration" req_ref="REQ-EMBED-022">
  <description>Model hot-swap completes without request drops</description>
  <preconditions>Model A loaded, Model B available</preconditions>
  <inputs>Trigger swap while sending requests</inputs>
  <expected>All requests complete. Some may have higher latency during swap.</expected>
  <data_requirements>Use REAL model loading and swap</data_requirements>
</test_case>

<test_case id="TC-EMBED-023" type="integration" req_ref="REQ-EMBED-023">
  <description>Hot-swap validates model compatibility</description>
  <preconditions>Incompatible model (wrong dimension)</preconditions>
  <inputs>Trigger swap to incompatible model</inputs>
  <expected>Swap rejected, original model remains, error logged</expected>
  <data_requirements>Use REAL validation with intentionally wrong model</data_requirements>
</test_case>

<!-- Benchmark Tests -->

<test_case id="TC-EMBED-024" type="benchmark" req_ref="REQ-EMBED-026">
  <description>All models meet individual latency budgets</description>
  <preconditions>All models loaded on GPU</preconditions>
  <inputs>1000 requests per model</inputs>
  <expected>P95 latency within budget for each model</expected>
  <data_requirements>Use REAL GPU inference with timing</data_requirements>
</test_case>

<test_case id="TC-EMBED-025" type="benchmark" req_ref="REQ-EMBED-027">
  <description>End-to-end latency less than 200ms</description>
  <preconditions>Full pipeline operational</preconditions>
  <inputs>1000 single-item requests</inputs>
  <expected>P95 latency less than 200ms</expected>
  <data_requirements>Use REAL full pipeline with timing</data_requirements>
</test_case>

<test_case id="TC-EMBED-026" type="benchmark" req_ref="REQ-EMBED-027">
  <description>Batch throughput exceeds 100 items/sec</description>
  <preconditions>Batch size 64</preconditions>
  <inputs>Continuous batch submissions for 60 seconds</inputs>
  <expected>Average throughput greater than 100 items/sec</expected>
  <data_requirements>Use REAL batching with throughput measurement</data_requirements>
</test_case>

<test_case id="TC-EMBED-027" type="benchmark" req_ref="REQ-EMBED-024">
  <description>GPU memory stays under 16GB</description>
  <preconditions>All 12 models loaded</preconditions>
  <inputs>nvidia-smi measurement</inputs>
  <expected>Total VRAM less than 16GB</expected>
  <data_requirements>Use REAL GPU memory measurement</data_requirements>
</test_case>

<!-- Stress Tests -->

<test_case id="TC-EMBED-028" type="stress" req_ref="REQ-EMBED-019">
  <description>System handles request queue overflow gracefully</description>
  <preconditions>Queue capacity 10000</preconditions>
  <inputs>Submit 15000 requests rapidly</inputs>
  <expected>First 10000 queued, remaining 5000 return rate limit error</expected>
  <data_requirements>Use REAL queue with bounded capacity</data_requirements>
</test_case>

<test_case id="TC-EMBED-029" type="stress" req_ref="REQ-EMBED-030">
  <description>System recovers from GPU OOM</description>
  <preconditions>Artificially constrain GPU memory</preconditions>
  <inputs>Large batch that would exceed memory</inputs>
  <expected>Batch splits, retries with smaller size, eventually succeeds</expected>
  <data_requirements>Use REAL GPU with memory limit</data_requirements>
</test_case>

<test_case id="TC-EMBED-030" type="stress" req_ref="REQ-EMBED-020">
  <description>Cache performs under memory pressure</description>
  <preconditions>Cache at capacity</preconditions>
  <inputs>10000 new unique embeddings</inputs>
  <expected>LRU eviction works correctly, no memory growth beyond capacity</expected>
  <data_requirements>Use REAL cache with memory monitoring</data_requirements>
</test_case>

</test_plan>

<!-- ============================================================================ -->
<!-- CONSTRAINTS -->
<!-- ============================================================================ -->

<constraints>
  <constraint id="CON-EMBED-001">NO mock embeddings in tests - all tests must use REAL model inference</constraint>
  <constraint id="CON-EMBED-002">Each embedding model must have explicit latency budget defined and enforced</constraint>
  <constraint id="CON-EMBED-003">GPU memory budget: less than 16GB for all models combined</constraint>
  <constraint id="CON-EMBED-004">Tests MUST verify FuseMoE routing weights are inspectable and sum to 1.0</constraint>
  <constraint id="CON-EMBED-005">All code must be Rust 2021 edition with stable toolchain</constraint>
  <constraint id="CON-EMBED-006">Maximum 500 lines per source file (excluding tests)</constraint>
  <constraint id="CON-EMBED-007">All public APIs must have rustdoc comments with latency constraints</constraint>
  <constraint id="CON-EMBED-008">CUDA code confined to context-graph-cuda crate</constraint>
  <constraint id="CON-EMBED-009">Embedding dimensions must be compile-time constants where possible</constraint>
  <constraint id="CON-EMBED-010">Cache key computation must be deterministic (same content = same key)</constraint>
  <constraint id="CON-EMBED-011">Model files must be loaded from configurable path (no hardcoded paths)</constraint>
  <constraint id="CON-EMBED-012">All embedding models must support graceful degradation on failure</constraint>
</constraints>

<!-- ============================================================================ -->
<!-- DEPENDENCIES -->
<!-- ============================================================================ -->

<dependencies>
  <dependency type="module" name="Module 2 (Core Infrastructure)" version="1.0">
    <provides>
      <item>MemoryNode struct (REQ-CORE-001) - embedding field</item>
      <item>RocksDB storage layer (REQ-CORE-007) - embedding persistence</item>
      <item>MCP server infrastructure - embedding tool handlers</item>
      <item>Configuration framework - model paths, batch sizes</item>
    </provides>
  </dependency>
  <dependency type="module" name="Module 1 (Ghost System)" version="1.0">
    <provides>
      <item>EmbeddingProvider trait (REQ-GHOST-008) - interface contract</item>
      <item>CUDA configuration framework (REQ-GHOST-001)</item>
    </provides>
  </dependency>
  <dependency type="rust_crate" name="cudarc" version="0.10+" purpose="CUDA kernel bindings"/>
  <dependency type="rust_crate" name="half" version="2.3+" purpose="FP16/FP8 types"/>
  <dependency type="rust_crate" name="ndarray" version="0.15+" purpose="Multi-dimensional arrays"/>
  <dependency type="rust_crate" name="tokenizers" version="0.15+" purpose="Text tokenization"/>
  <dependency type="rust_crate" name="safetensors" version="0.4+" purpose="Model weight loading"/>
  <dependency type="rust_crate" name="lru" version="0.12+" purpose="Embedding cache"/>
  <dependency type="rust_crate" name="sha2" version="0.10+" purpose="Cache key hashing"/>
  <dependency type="rust_crate" name="tokio" version="1.35+" purpose="Async runtime"/>
  <dependency type="rust_crate" name="crossbeam" version="0.8+" purpose="Concurrent queues"/>
  <dependency type="rust_crate" name="metrics" version="0.21+" purpose="Performance metrics"/>
  <dependency type="external" name="CUDA 13.1" purpose="GPU acceleration"/>
  <dependency type="external" name="cuDNN 8.9+" purpose="Deep learning primitives"/>
  <dependency type="model" name="E5-Large or equivalent" purpose="E1 Semantic embeddings"/>
  <dependency type="model" name="CodeBERT or equivalent" purpose="E7 Code embeddings"/>
  <dependency type="model" name="CLIP or equivalent" purpose="E10 Multimodal embeddings"/>
</dependencies>

</functional_spec>
```

---

## Appendix A: Embedding Model Architecture Details

### E1: Semantic Embedding (1024D)

```rust
/// Semantic embedding using dense transformer architecture.
///
/// Constraint: Latency < 5ms
/// Constraint: FP8 quantization on Tensor Core
pub struct SemanticEmbedder {
    model: TransformerModel,
    tokenizer: Tokenizer,
    max_tokens: usize,  // 8192
}

impl SemanticEmbedder {
    pub fn embed(&self, content: &str) -> Result<Vec<f32>, EmbedError>;
}
```

### E2-E4: Temporal Embeddings (512D each)

```rust
/// Temporal embedding family for time-aware representations.
pub enum TemporalModel {
    /// Exponential decay for recency (E2)
    Recent { lambda: f32 },
    /// Fourier basis for periodicity (E3)
    Periodic { periods: Vec<Duration> },
    /// Sinusoidal positional encoding (E4)
    Positional { max_position: u32 },
}
```

### E5: Causal Embedding (768D)

```rust
/// Causal embedding using SCM intervention encoding.
///
/// Encodes: do(X) intervention semantics
/// Captures: Direction of causation
pub struct CausalEmbedder {
    scm_encoder: SCMEncoder,
    intervention_dim: usize,  // 768
}
```

### E6: Sparse Embedding (~30K, 5% active)

```rust
/// Sparse embedding using SPLADE-style top-k activation.
///
/// Constraint: ~5% sparsity (1500 active dimensions)
pub struct SparseEmbedder {
    vocab_size: usize,      // ~30000
    sparsity_target: f32,   // 0.05
    top_k: usize,           // 1500
}

/// Sparse vector representation for storage efficiency.
pub struct SparseVector {
    indices: Vec<u32>,
    values: Vec<f32>,
}
```

### E7: Code Embedding (1536D)

```rust
/// AST-aware code embedding.
///
/// Supported languages: Python, JavaScript, TypeScript, Rust, Go, Java
/// Constraint: Latency < 10ms
pub struct CodeEmbedder {
    model: ASTTransformer,
    language_detector: LanguageDetector,
    fallback: SemanticEmbedder,
}
```

### E8: Graph/GNN Embedding (1536D)

```rust
/// Graph neural network embedding using message passing.
///
/// Message passing iterations: 3
/// Aggregation: Mean pooling
pub struct GraphEmbedder {
    mpnn: MessagePassingNetwork,
    iterations: usize,  // 3
    aggregation: Aggregation,
}
```

### E9: HDC Embedding (10K-bit)

```rust
/// Hyperdimensional computing embedding.
///
/// Operations: XOR binding, majority bundling
/// Distance: Hamming
/// Constraint: Latency < 1ms
pub struct HDCEmbedder {
    dimension: usize,  // 10000
    item_memory: HashMap<String, BitVec>,
}

/// Binary vector for HDC operations.
pub struct BitVec {
    bits: Vec<u64>,  // Packed binary (10000 bits = 157 u64s)
}
```

### E10: Multimodal Embedding (1024D)

```rust
/// Cross-modal embedding using CLIP-style architecture.
///
/// Modalities: text, image, audio
pub struct MultimodalEmbedder {
    text_encoder: TextEncoder,
    image_encoder: ImageEncoder,
    cross_attention: CrossAttention,
}
```

### E11: Entity/TransE Embedding (256D)

```rust
/// Knowledge graph embedding using TransE.
///
/// Formula: h + r ≈ t
pub struct EntityEmbedder {
    entity_embeddings: HashMap<String, Vec<f32>>,
    relation_embeddings: HashMap<String, Vec<f32>>,
    ner_model: NERModel,
}
```

### E12: Late-Interaction Embedding (128D/token)

```rust
/// ColBERT-style late interaction for fine-grained matching.
///
/// Scoring: MaxSim over all token pairs
pub struct LateInteractionEmbedder {
    encoder: ColBERTEncoder,
    token_dim: usize,  // 128
}

impl LateInteractionEmbedder {
    pub fn embed_query(&self, query: &str) -> Vec<Vec<f32>>;
    pub fn embed_document(&self, doc: &str) -> Vec<Vec<f32>>;
    pub fn max_sim(query_vecs: &[Vec<f32>], doc_vecs: &[Vec<f32>]) -> f32;
}
```

---

## Appendix B: FuseMoE Architecture

```rust
/// Mixture of Experts fusion layer.
///
/// Experts: 12 (one per embedding model)
/// Routing: top-k=4 with Laplace smoothing
/// Output: 1536D
pub struct FuseMoE {
    experts: Vec<Expert>,
    router: GatingNetwork,
    top_k: usize,           // 4
    laplace_alpha: f32,     // 0.01
    output_dim: usize,      // 1536
}

/// Gating network for expert routing.
pub struct GatingNetwork {
    mlp: MLP,
    softmax_temp: f32,
}

impl FuseMoE {
    /// Fuse 12 embeddings into unified representation.
    ///
    /// Returns: (fused_embedding, routing_weights)
    pub fn fuse(
        &self,
        embeddings: &EmbeddingSet,
    ) -> Result<(Vec<f32>, RoutingWeights), FuseError>;
}

/// Routing decision with explainability.
pub struct RoutingWeights {
    pub weights: [f32; 12],        // All expert weights
    pub selected_experts: [u8; 4], // Indices of top-4
    pub selected_weights: [f32; 4], // Normalized weights for top-4
}
```

---

## Appendix C: CAME-AB Cross-Modality Bridge

```rust
/// Cross-Attention Modality Encoding with Adaptive Bridging.
///
/// Attention heads: 8
/// Bridge learning rate: 0.001
pub struct CAMEAB {
    attention_heads: usize,     // 8
    bridge_weights: [[f32; 12]; 12],  // Modality pair weights
    learning_rate: f32,         // 0.001
    layer_norm: LayerNorm,
}

impl CAMEAB {
    /// Apply cross-modality attention.
    pub fn forward(
        &self,
        modality_embeddings: &[Vec<f32>; 12],
    ) -> [Vec<f32>; 12];

    /// Update bridge weights with gradient.
    pub fn update_bridges(&mut self, gradients: &[[f32; 12]; 12]);
}
```

---

## Appendix D: Performance Budget Summary

| Model | Dimension | Latency Budget | Hardware |
|-------|-----------|----------------|----------|
| E1 Semantic | 1024D | < 5ms | Tensor Core FP8 |
| E2 Temporal-Recent | 512D | < 2ms | Vector Unit |
| E3 Temporal-Periodic | 512D | < 2ms | FFT Unit |
| E4 Temporal-Positional | 512D | < 2ms | CUDA Core |
| E5 Causal | 768D | < 8ms | Tensor Core |
| E6 Sparse | ~30K (5%) | < 3ms | Sparse Tensor |
| E7 Code | 1536D | < 10ms | Tensor Core FP16 |
| E8 Graph/GNN | 1536D | < 5ms | CUDA Graph |
| E9 HDC | 10K-bit | < 1ms | Vector Unit |
| E10 Multimodal | 1024D | < 15ms | Tensor Core |
| E11 Entity/TransE | 256D | < 2ms | CUDA Core |
| E12 Late-Interaction | 128D/tok | < 8ms | CUDA Tile |
| **FuseMoE Fusion** | 4096D -> 1536D | < 3ms | Tensor Core |
| **CAME-AB Bridge** | - | < 5ms | Tensor Core |
| **Total Single Item** | 1536D | < 200ms | Combined |
| **Batch (64 items)** | - | > 100/sec | Combined |

---

## Appendix E: Requirement Traceability Matrix

| Requirement ID | User Story | Test Cases | Priority | Status |
|---------------|------------|------------|----------|--------|
| REQ-EMBED-001 | US-EMBED-01 | TC-EMBED-001 | must | pending |
| REQ-EMBED-002 | US-EMBED-01 | TC-EMBED-002 | must | pending |
| REQ-EMBED-003 | US-EMBED-01 | TC-EMBED-003 | must | pending |
| REQ-EMBED-004 | US-EMBED-01 | TC-EMBED-004 | must | pending |
| REQ-EMBED-005 | US-EMBED-01 | TC-EMBED-005 | must | pending |
| REQ-EMBED-006 | US-EMBED-09 | TC-EMBED-006 | must | pending |
| REQ-EMBED-007 | US-EMBED-01 | TC-EMBED-007 | must | pending |
| REQ-EMBED-008 | US-EMBED-01 | TC-EMBED-008 | must | pending |
| REQ-EMBED-009 | US-EMBED-01 | TC-EMBED-009 | must | pending |
| REQ-EMBED-010 | US-EMBED-01 | TC-EMBED-010 | must | pending |
| REQ-EMBED-011 | US-EMBED-01 | TC-EMBED-011 | must | pending |
| REQ-EMBED-012 | US-EMBED-10 | TC-EMBED-012 | must | pending |
| REQ-EMBED-013 | US-EMBED-02 | TC-EMBED-013, TC-EMBED-014 | must | pending |
| REQ-EMBED-014 | US-EMBED-02 | TC-EMBED-014 | must | pending |
| REQ-EMBED-015 | US-EMBED-02, US-EMBED-08 | TC-EMBED-015 | must | pending |
| REQ-EMBED-016 | US-EMBED-07 | TC-EMBED-016 | must | pending |
| REQ-EMBED-017 | US-EMBED-07 | TC-EMBED-017 | must | pending |
| REQ-EMBED-018 | US-EMBED-03 | TC-EMBED-018 | must | pending |
| REQ-EMBED-019 | US-EMBED-03 | TC-EMBED-019 | must | pending |
| REQ-EMBED-020 | US-EMBED-04 | TC-EMBED-020 | must | pending |
| REQ-EMBED-021 | US-EMBED-04 | TC-EMBED-021 | must | pending |
| REQ-EMBED-022 | US-EMBED-05 | TC-EMBED-022 | must | pending |
| REQ-EMBED-023 | US-EMBED-05 | TC-EMBED-023 | must | pending |
| REQ-EMBED-024 | US-EMBED-06 | TC-EMBED-027 | must | pending |
| REQ-EMBED-025 | US-EMBED-06 | TC-EMBED-027 | must | pending |
| REQ-EMBED-026 | US-EMBED-03 | TC-EMBED-024 | must | pending |
| REQ-EMBED-027 | US-EMBED-03 | TC-EMBED-025, TC-EMBED-026 | must | pending |
| REQ-EMBED-028 | US-EMBED-02 | TC-EMBED-013 | must | pending |
| REQ-EMBED-029 | US-EMBED-01 | All model tests | must | pending |
| REQ-EMBED-030 | US-EMBED-01, US-EMBED-03 | TC-EMBED-029 | must | pending |

---

## Appendix F: Module 2 References

This module builds upon Module 2 (Core Infrastructure):

| Module 2 Item | Usage in Module 3 |
|--------------|-------------------|
| MemoryNode.embedding | Upgraded from 1536D stub to real 1536D fused output |
| EmbeddingProvider trait (Ghost) | Implemented by embedding pipeline |
| RocksDB storage | Stores computed embeddings |
| Configuration framework | Model paths, batch sizes, cache capacity |
| MCP server | Hosts embedding-related tool handlers |

---

*Document generated: 2025-12-31*
*Specification Version: 1.0*
*Module: 12-Model Embedding Pipeline (Phase 2)*
