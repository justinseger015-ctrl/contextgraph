# E7 Code Embedding Fine-Tuning Plan v1.1

## Executive Summary

This document outlines a comprehensive plan to:
1. Embed the Context Graph codebase using E7 (V_correctness)
2. Establish baseline metrics for E7 code retrieval quality
3. Fine-tune E7 parameters using the codebase as a ground-truth dataset
4. Integrate improvements back into MCP tools and the overall system

**Target Embedder**: E7 (V_correctness) - Qodo-Embed-1-1.5B, 1536D dense vectors

---

## Research-Based Best Practices (NEW)

This section incorporates industry best practices from:
- [Qodo's State-of-the-Art Code Retrieval](https://www.qodo.ai/blog/qodo-embed-1-code-embedding-code-retrieval/)
- [cAST: AST-Based Code Chunking (EMNLP 2025)](https://arxiv.org/html/2506.15655v1)
- [Qodo RAG for Large-Scale Code Repos](https://www.qodo.ai/blog/rag-for-large-scale-code-repos/)
- [Supermemory AST-Aware Chunking](https://supermemory.ai/blog/building-code-chunk-ast-aware-code-chunking/)
- [Qodo-Embed-1-1.5B Model Card](https://huggingface.co/Qodo/Qodo-Embed-1-1.5B)

### Key Insight: Why Naive Chunking Fails for Code

> "Naive chunking methods struggle with accurately delineating meaningful segments of code, leading to issues with boundary definition and the inclusion of irrelevant or incomplete information. Providing invalid or incomplete code segments to an LLM can actually hurt performance and increase hallucinations." — Qodo

Traditional text chunking (fixed-size, paragraph-based) **breaks syntactic structures** in code:
- Splits functions mid-declaration
- Separates return types from function bodies
- Loses import context
- Creates unusable fragments

### Best Practice #1: AST-Based Chunking (cAST Methodology)

**Use Abstract Syntax Trees** to identify natural split points:

```
Traditional:        │  AST-Based:
─────────────────── │  ───────────────────
Line 1-50 (chunk 1) │  Function A (complete)
Line 51-100 (chunk 2│) Function B (complete)
  ↑ Breaks mid-func │  Struct C (complete)
```

**Algorithm** (from cAST paper):
1. Parse source code into AST using **tree-sitter**
2. Traverse top-down, attempting to fit large nodes into single chunks
3. For nodes exceeding size limits, recursively split into children
4. Greedily merge adjacent sibling nodes to maximize information density
5. Measure chunks by **non-whitespace characters** (not lines)

**Performance Gains** (from cAST benchmarks):
| Metric | Improvement |
|--------|-------------|
| Precision/Recall@5 | +1.2–4.3 points |
| Pass@1 (SWE-bench) | +2.67 points |
| RepoEval average | +5.5 points |
| CrossCodeEval | +4.3 points |

### Best Practice #2: Optimal Chunk Size (~500 characters)

> "Embedding smaller chunks generally leads to better performance. Ideally, you want to have the smallest possible chunk that contains the relevant context — anything irrelevant that's included dilutes the semantic meaning." — Qodo

**Recommended Target**: ~500 non-whitespace characters per chunk

| Size | Pros | Cons |
|------|------|------|
| < 200 chars | Very focused | Loses context |
| 400-600 chars | Optimal balance | - |
| > 1000 chars | More context | Semantic dilution |

### Best Practice #3: Context Prepending Strategy

> "Embedding models are trained on natural language. When you embed `async getUser(id: string)`, the model doesn't inherently know this is inside a UserService class or that it uses a Database." — Supermemory

**Prepend semantic metadata** to each chunk:

```rust
// Before embedding, construct contextualized text:
let contextualized = format!(
    "File: {file_path}\n\
     Scope: {scope_chain}\n\
     Imports: {relevant_imports}\n\
     ---\n\
     {code_chunk}"
);
```

**Context Fields**:
| Field | Example | Purpose |
|-------|---------|---------|
| File path | `crates/embeddings/src/models/code/model.rs` | Location context |
| Scope chain | `CodeModel > embed > gpu_forward` | Hierarchical context |
| Entity signatures | `pub async fn embed(&self, input: &ModelInput) -> Result<ModelEmbedding>` | Type information |
| Imports | `use candle_core::Tensor;` | Dependency context |
| Parent definition | `impl EmbeddingModel for CodeModel { ... }` | Inheritance/trait context |

### Best Practice #4: Natural Language Descriptions (Dual Embedding)

> "Code embeddings often don't capture the semantic meaning of code, especially for natural language queries. We use LLMs to generate natural language descriptions for each code chunk." — Qodo

**Strategy**: Generate NL descriptions and embed BOTH code and description:

```python
# For each code chunk:
description = llm.generate(f"Describe what this code does in 1-2 sentences:\n{code_chunk}")

# Store both embeddings:
code_embedding = e7_model.encode(code_chunk)
description_embedding = e7_model.encode(description)

# During retrieval, search both indexes
```

**Description Variations** (from Qodo):
- Formal documentation style
- Concise natural language summary
- 10-30 word query-style description

### Best Practice #5: Qodo-Embed-1-1.5B Specific Settings

From the [model card](https://huggingface.co/Qodo/Qodo-Embed-1-1.5B):

| Setting | Recommended Value | Notes |
|---------|-------------------|-------|
| **max_length** | 8192 tokens | Optimal for transformers usage |
| **Pooling** | Last-token | NOT mean pooling (Qwen2 architecture) |
| **Normalization** | L2 normalize | `F.normalize(embeddings, p=2, dim=1)` |
| **Supported languages** | Python, C++, C#, Go, Java, JavaScript, PHP, Ruby, TypeScript | Rust works but isn't officially listed |
| **Context window** | 32K tokens | Maximum supported |
| **Embedding dim** | 1536 | Fixed |

**Code Usage Pattern**:
```python
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained('Qodo/Qodo-Embed-1-1.5B', trust_remote_code=True)
model = AutoModel.from_pretrained('Qodo/Qodo-Embed-1-1.5B', trust_remote_code=True)

# Tokenize with recommended max_length
batch_dict = tokenizer(
    input_texts,
    max_length=8192,
    padding=True,
    truncation=True,
    return_tensors='pt'
)

outputs = model(**batch_dict)
# Last-token pooling (critical for Qwen2 architecture)
embeddings = outputs.last_hidden_state[:, -1, :]
# L2 normalization
embeddings = F.normalize(embeddings, p=2, dim=1)
```

### Best Practice #6: Two-Stage Retrieval

> "Perform initial vector similarity search, then use LLM filtering and ranking to identify truly relevant results based on task context." — Qodo

**Pipeline**:
```
Query → E13 sparse recall (10K) → E7 dense search (100) → LLM rerank (10)
             ↓                           ↓                      ↓
      Broad candidate set        Code-aware filtering    Context-aware ranking
```

### Best Practice #7: Include Class/Struct Definitions with Methods

> "For a large class, we might create an embedding and index individual methods separately but include the class definition and relevant imports with each method chunk." — Qodo

**Example**: When chunking a method, include its parent:
```rust
// Chunk for `CodeModel::embed` includes:

// 1. Class/struct definition (abbreviated)
pub struct CodeModel {
    model_state: RwLock<ModelState>,
    model_path: PathBuf,
    // ... fields
}

// 2. The actual method
impl EmbeddingModel for CodeModel {
    async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
        // Full method body
    }
}
```

---

## Phase 1: Codebase Indexing Infrastructure

### 1.1 Extend File Watcher for Source Code

**Current State**: File watcher only monitors `./docs` for `.md` files.

**Required Changes**:

```toml
# config/default.toml additions
[watcher]
enabled = true
watch_paths = ["./docs", "./crates"]
session_id = "codebase-indexer"

[watcher.extensions]
markdown = [".md"]
rust = [".rs"]
toml = [".toml"]
yaml = [".yaml", ".yml"]

[watcher.exclude]
patterns = ["target/", ".git/", "*.lock", "node_modules/"]
```

**Files to Modify**:
- `config/default.toml` - Add source file extensions
- `crates/context-graph-mcp/src/tools/definitions/file_watcher.rs` - Extend to handle code files
- `crates/context-graph-core/src/memory/chunker.rs` - Add code-aware chunking

### 1.2 Code-Aware Chunking Strategy (AST-Based)

**Current TextChunker**: 200 words, 50-word overlap, sentence boundary detection — **NOT suitable for code**

**Proposed AstCodeChunker** (based on cAST methodology + Qodo best practices):

**Key Design Decisions**:
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Parsing | `tree-sitter` | Battle-tested, multi-language, used in Neovim/Helix/Zed |
| Size metric | Non-whitespace characters | Consistent across coding styles |
| Target size | ~500 chars (400-600 range) | Qodo recommendation for optimal retrieval |
| Context | Prepend file/scope/imports | Enhance semantic understanding |
| Descriptions | LLM-generated (optional) | Bridge NL queries to code |

**New Files**:
- `crates/context-graph-core/src/memory/ast_chunker.rs` - Main AST chunker
- `crates/context-graph-core/src/memory/ast_chunker/rust.rs` - Rust-specific rules
- `crates/context-graph-core/src/memory/ast_chunker/context.rs` - Context builder

**Core Design**:

```rust
use tree_sitter::{Language, Parser, Node};

/// AST-based code chunker following cAST methodology
///
/// References:
/// - cAST paper: https://arxiv.org/html/2506.15655v1
/// - Qodo best practices: ~500 char chunks with context
pub struct AstCodeChunker {
    parser: Parser,
    config: ChunkConfig,
}

#[derive(Clone)]
pub struct ChunkConfig {
    /// Target chunk size in non-whitespace characters (default: 500)
    pub target_size: usize,
    /// Minimum chunk size before merging with siblings (default: 100)
    pub min_size: usize,
    /// Maximum chunk size before recursive splitting (default: 1000)
    pub max_size: usize,
    /// Include parent struct/class definition with methods
    pub include_parent_context: bool,
    /// Include relevant imports in each chunk
    pub include_imports: bool,
    /// Generate natural language descriptions (requires LLM)
    pub generate_descriptions: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_size: 500,      // Qodo recommendation
            min_size: 100,         // Avoid tiny fragments
            max_size: 1000,        // Prevent semantic dilution
            include_parent_context: true,
            include_imports: true,
            generate_descriptions: false, // Off by default (expensive)
        }
    }
}

/// A code chunk with full context for embedding
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// Raw code content
    pub code: String,
    /// Contextualized text for embedding (includes metadata)
    pub contextualized_text: String,
    /// Optional NL description (if generate_descriptions=true)
    pub description: Option<String>,
    /// Metadata for storage
    pub metadata: CodeChunkMetadata,
}

#[derive(Debug, Clone)]
pub struct CodeChunkMetadata {
    pub file_path: String,
    pub language: String,
    pub scope_chain: Vec<String>,        // e.g., ["CodeModel", "embed"]
    pub entity_type: EntityType,          // Function, Struct, Impl, etc.
    pub entity_signature: Option<String>, // Full signature if available
    pub start_line: u32,
    pub end_line: u32,
    pub start_byte: usize,
    pub end_byte: usize,
    pub non_whitespace_chars: usize,      // Size metric per cAST
    pub imports: Vec<String>,             // Relevant imports
    pub parent_definition: Option<String>, // Parent struct/class if any
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntityType {
    Function,
    Method,
    Struct,
    Enum,
    Trait,
    Impl,
    Module,
    Const,
    Static,
    TypeAlias,
    Macro,
    Comment,
    Mixed,  // Merged siblings of different types
}

impl AstCodeChunker {
    pub fn new(language: Language, config: ChunkConfig) -> Self {
        let mut parser = Parser::new();
        parser.set_language(language).expect("Language should be valid");
        Self { parser, config }
    }

    /// Main chunking method following cAST divide-and-combine algorithm
    pub fn chunk(&mut self, source: &str, file_path: &str) -> Result<Vec<CodeChunk>, ChunkError> {
        let tree = self.parser.parse(source, None)
            .ok_or(ChunkError::ParseFailed)?;

        let root = tree.root_node();
        let imports = self.extract_imports(&root, source);

        let mut chunks = Vec::new();
        self.process_node(
            &root,
            source,
            file_path,
            &imports,
            &mut vec![],  // scope chain
            None,         // parent definition
            &mut chunks,
        )?;

        // Merge small adjacent chunks
        let merged = self.merge_small_chunks(chunks);

        // Build contextualized text for each chunk
        Ok(merged.into_iter().map(|c| self.contextualize(c)).collect())
    }

    /// Recursive divide-and-combine algorithm (per cAST paper)
    fn process_node(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        imports: &[String],
        scope_chain: &mut Vec<String>,
        parent_def: Option<&str>,
        chunks: &mut Vec<CodeChunk>,
    ) -> Result<(), ChunkError> {
        let size = self.non_whitespace_size(node, source);

        // Case 1: Node fits in a single chunk
        if size <= self.config.max_size {
            if let Some(chunk) = self.node_to_chunk(node, source, file_path, imports, scope_chain, parent_def) {
                chunks.push(chunk);
            }
            return Ok(());
        }

        // Case 2: Node too large - recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if self.is_chunkable_node(&child) {
                // Update scope chain for named nodes
                if let Some(name) = self.get_node_name(&child, source) {
                    scope_chain.push(name.clone());
                }

                // Get parent definition for methods
                let new_parent = if self.is_container_node(node) {
                    Some(self.get_abbreviated_definition(node, source))
                } else {
                    parent_def.map(|s| s.to_string())
                };

                self.process_node(
                    &child,
                    source,
                    file_path,
                    imports,
                    scope_chain,
                    new_parent.as_deref(),
                    chunks,
                )?;

                if self.get_node_name(&child, source).is_some() {
                    scope_chain.pop();
                }
            }
        }

        Ok(())
    }

    /// Build contextualized text for embedding
    fn contextualize(&self, mut chunk: CodeChunk) -> CodeChunk {
        let mut parts = Vec::new();

        // File path (truncated to last 3 segments)
        let path_parts: Vec<&str> = chunk.metadata.file_path.split('/').collect();
        let short_path = if path_parts.len() > 3 {
            path_parts[path_parts.len()-3..].join("/")
        } else {
            chunk.metadata.file_path.clone()
        };
        parts.push(format!("File: {}", short_path));

        // Scope chain
        if !chunk.metadata.scope_chain.is_empty() {
            parts.push(format!("Scope: {}", chunk.metadata.scope_chain.join(" > ")));
        }

        // Entity signature
        if let Some(sig) = &chunk.metadata.entity_signature {
            parts.push(format!("Signature: {}", sig));
        }

        // Imports (abbreviated)
        if self.config.include_imports && !chunk.metadata.imports.is_empty() {
            let import_str = chunk.metadata.imports.iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            parts.push(format!("Uses: {}", import_str));
        }

        parts.push("---".to_string());

        // Parent definition (abbreviated)
        if self.config.include_parent_context {
            if let Some(parent) = &chunk.metadata.parent_definition {
                parts.push(parent.clone());
                parts.push(String::new());
            }
        }

        // The actual code
        parts.push(chunk.code.clone());

        chunk.contextualized_text = parts.join("\n");
        chunk
    }

    /// Count non-whitespace characters (per cAST paper)
    fn non_whitespace_size(&self, node: &Node, source: &str) -> usize {
        let text = &source[node.start_byte()..node.end_byte()];
        text.chars().filter(|c| !c.is_whitespace()).count()
    }

    // ... additional helper methods
}
```

**Rust-Specific Node Types** (in `ast_chunker/rust.rs`):

```rust
/// Chunkable Rust AST node types
pub const RUST_CHUNKABLE_NODES: &[&str] = &[
    "function_item",
    "impl_item",
    "struct_item",
    "enum_item",
    "trait_item",
    "mod_item",
    "const_item",
    "static_item",
    "type_item",
    "macro_definition",
    "attribute_item",  // For doc comments
];

/// Container nodes that provide context for children
pub const RUST_CONTAINER_NODES: &[&str] = &[
    "impl_item",
    "struct_item",
    "enum_item",
    "trait_item",
    "mod_item",
];
```

**Dependencies** (add to `Cargo.toml`):
```toml
[dependencies]
tree-sitter = "0.20"
tree-sitter-rust = "0.20"

# Optional: for multi-language support
tree-sitter-python = "0.20"
tree-sitter-javascript = "0.20"
tree-sitter-typescript = "0.20"
```

### 1.3 Natural Language Description Generation

Per Qodo best practices, generate NL descriptions to bridge the gap between natural language queries and code:

**Implementation Options**:

| Option | Pros | Cons | Recommended For |
|--------|------|------|-----------------|
| Local LLM (Ollama) | Privacy, no API costs | Slower, less quality | Development/testing |
| Claude API | High quality | API costs | Production |
| Doc comments only | Zero cost | Limited coverage | MVP |

**Description Generation Prompt**:
```python
DESCRIPTION_PROMPT = """Describe what this code does in 1-2 concise sentences.
Focus on the PURPOSE and BEHAVIOR, not implementation details.

Code:
```{language}
{code_chunk}
```

Description:"""

# Generate multiple variations per chunk (per Qodo):
VARIATION_PROMPTS = [
    "Write a formal documentation comment for this code:",
    "In 10-20 words, what does this code accomplish?",
    "If a developer searched for this code, what query would they use?"
]
```

**Storage Strategy**:
```rust
pub struct EnrichedCodeChunk {
    /// Primary chunk data
    pub chunk: CodeChunk,
    /// E7 embedding of contextualized_text
    pub code_embedding: Vec<f32>,
    /// E7 embedding of NL description (if generated)
    pub description_embedding: Option<Vec<f32>>,
    /// E1 semantic embedding of description (for hybrid search)
    pub semantic_embedding: Option<Vec<f32>>,
}
```

**Dual-Index Search Strategy**:
```rust
// During retrieval, search both code and description indexes
pub async fn search_code_dual(
    &self,
    query: &str,
    top_k: usize,
) -> Vec<CodeSearchResult> {
    let query_embedding = self.embed_query(query);

    // Search code embeddings
    let code_results = self.code_index.search(&query_embedding, top_k * 2);

    // Search description embeddings (better for NL queries)
    let desc_results = self.description_index.search(&query_embedding, top_k * 2);

    // RRF fusion of both result sets
    self.rrf_merge(code_results, desc_results, top_k)
}
```

### 1.4 Codebase Statistics (Estimated)

| Metric | Estimate |
|--------|----------|
| Total `.rs` files | ~300 |
| Lines of code | ~50,000 |
| Functions | ~2,000 |
| Structs/Enums | ~500 |
| Estimated chunks (AST-based) | ~2,500-4,000 |
| Avg chunk size | ~500 non-WS chars |
| Storage (embeddings) | ~200MB (all 13 embedders) |
| Description generation cost | ~$0.50-1.00 (one-time, Claude API) |

---

## Phase 2: Ground-Truth Dataset Construction

### 2.1 Query-Document Pairs

Create manually curated ground-truth pairs for E7 evaluation:

**Categories**:

| Category | Example Query | Expected Results |
|----------|---------------|------------------|
| Function Search | "async function that handles memory storage" | `TeleologicalStore::store_async()` |
| Pattern Search | "error handling with Result type" | Error handling functions |
| Import Search | "code that uses tokio runtime" | Files with `use tokio::*` |
| Trait Search | "implementation of EmbeddingModel trait" | `CodeModel`, `SemanticModel`, etc. |
| Struct Search | "struct for storing fingerprints" | `SemanticFingerprint`, `TeleologicalArray` |

**Dataset File**: `data/e7_ground_truth/queries.jsonl`

```json
{"query": "function to embed code content", "relevant": ["crates/context-graph-embeddings/src/models/pretrained/code/model.rs:L255"], "category": "function"}
{"query": "async trait for storage operations", "relevant": ["crates/context-graph-storage/src/traits.rs:L42"], "category": "trait"}
```

### 2.2 Automatic Pair Generation

Use AST parsing to generate synthetic training pairs:

```rust
// Pseudo-code for pair generation
for file in crate_files {
    let ast = parse_rust_file(file);
    for function in ast.functions() {
        // Pair 1: Function name -> Function body
        pairs.push((function.name, function.body_text));

        // Pair 2: Doc comment -> Function body
        if let Some(doc) = function.doc_comment {
            pairs.push((doc, function.body_text));
        }

        // Pair 3: Signature -> Function body
        pairs.push((function.signature, function.body_text));
    }
}
```

**Tools**:
- `syn` crate for Rust AST parsing
- `tree-sitter` for multi-language support

---

## Phase 3: E7 Baseline Metrics

### 3.1 Metrics to Track

| Metric | Description | Target | Notes |
|--------|-------------|--------|-------|
| **P@1** | Precision at rank 1 | > 0.60 | Critical for single-result UX |
| **P@5** | Precision at rank 5 | > 0.45 | Standard code search metric |
| **P@10** | Precision at rank 10 | > 0.35 | Broader recall |
| **MRR** | Mean Reciprocal Rank | > 0.55 | Rank of first relevant result |
| **NDCG@10** | Normalized DCG at 10 | > 0.50 | Rank-weighted relevance |
| **IoU** | Intersection over Union | > 0.40 | **NEW** - Per Supermemory research |
| **Latency P95** | 95th percentile latency | < 50ms | User experience |

**IoU (Intersection over Union)** - Key metric from [Supermemory research](https://supermemory.ai/blog/building-code-chunk-ast-aware-code-chunking/):
> "Higher IoU means the retrieved chunks actually contain the relevant code, not just code from the same file."

```python
def compute_iou(retrieved_chunk: str, ground_truth_span: str) -> float:
    """
    Compute token-level IoU between retrieved chunk and ground truth.

    This measures how much of the retrieved chunk overlaps with
    the actual code the user was looking for.
    """
    retrieved_tokens = set(tokenize(retrieved_chunk))
    truth_tokens = set(tokenize(ground_truth_span))

    intersection = len(retrieved_tokens & truth_tokens)
    union = len(retrieved_tokens | truth_tokens)

    return intersection / union if union > 0 else 0.0
```

**Why IoU Matters for AST-Based Chunking**:
- Traditional metrics (P@K) only measure if the *file* was retrieved
- IoU measures if the *exact relevant code* is in the chunk
- AST-based chunks should have significantly higher IoU than naive chunking

### 3.2 Benchmark Suite

**File**: `crates/context-graph-benchmark/src/e7_evaluation.rs`

```rust
pub struct E7BenchmarkConfig {
    /// Ground truth dataset path
    pub ground_truth_path: PathBuf,
    /// K values for P@K metrics
    pub k_values: Vec<usize>,
    /// Number of warm-up queries
    pub warmup_queries: usize,
    /// Run comparison with E1 (semantic only)
    pub compare_with_e1: bool,
}

pub struct E7BenchmarkResult {
    pub precision_at_k: HashMap<usize, f64>,
    pub mrr: f64,
    pub ndcg_at_10: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub e7_unique_finds: usize,  // Results E7 found that E1 missed
    pub e1_unique_finds: usize,  // Results E1 found that E7 missed
}
```

### 3.3 E7 vs E1 Blind Spot Analysis

Key insight from constitution: E7 finds "code patterns, function signatures" that E1 misses by "treating code as natural language".

**Blind Spot Detection**:
```rust
// For each query, compare E7 and E1 rankings
for (query, ground_truth) in ground_truth_pairs {
    let e1_results = search_e1(query, top_k=20);
    let e7_results = search_e7(query, top_k=20);

    // E7 unique: found by E7, missed by E1 (E1 score < 0.3)
    let e7_unique = e7_results
        .filter(|r| r.e7_score > 0.5 && r.e1_score < 0.3);

    // E1 unique: found by E1, missed by E7
    let e1_unique = e1_results
        .filter(|r| r.e1_score > 0.5 && r.e7_score < 0.3);

    report.add_blind_spot_analysis(query, e7_unique, e1_unique);
}
```

---

## Phase 4: Parameter Tuning

### 4.1 Current E7 Parameters

From `crates/context-graph-mcp/src/handlers/tools/code_tools.rs`:

```rust
// Current defaults
let e7_blend = 0.4;  // E7 weight in E1+E7 blend
let min_score = 0.2;  // Minimum blended score threshold
let fetch_multiplier = 3;  // Over-fetch ratio for reranking
```

### 4.2 Parameters to Tune

| Parameter | Current | Range | Impact |
|-----------|---------|-------|--------|
| `e7_blend` | 0.4 | 0.2-0.8 | Higher = more E7 influence |
| `min_score` | 0.2 | 0.1-0.4 | Higher = stricter filtering |
| `fetch_multiplier` | 3 | 2-5 | Higher = more candidates |
| `language_boost` | N/A | 1.0-1.5 | Boost for detected language match |
| `ast_depth_weight` | N/A | 0.0-1.0 | Weight for AST structure similarity |

### 4.3 Tuning Methodology

**Grid Search**:
```python
# Pseudo-code for parameter tuning
best_params = None
best_mrr = 0.0

for e7_blend in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    for min_score in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for fetch_mult in [2, 3, 4, 5]:
            result = run_benchmark(e7_blend, min_score, fetch_mult)
            if result.mrr > best_mrr:
                best_mrr = result.mrr
                best_params = (e7_blend, min_score, fetch_mult)
```

**Bayesian Optimization** (preferred for larger search space):
```rust
// Using argmin crate for Bayesian optimization
use argmin::solver::particleswarm::ParticleSwarm;

let problem = E7TuningProblem::new(ground_truth_dataset);
let solver = ParticleSwarm::new((lower_bounds, upper_bounds), 40);
let result = Executor::new(problem, solver)
    .configure(|state| state.max_iters(100))
    .run()?;
```

---

## Phase 5: MCP Tools Integration

### 5.1 Enhanced `search_code` Tool

**Current Implementation**: `crates/context-graph-mcp/src/handlers/tools/code_tools.rs`

**Proposed Enhancements**:

```rust
pub struct SearchCodeRequest {
    pub query: String,
    pub top_k: usize,
    pub min_score: f32,
    pub blend_with_semantic: f32,
    pub include_content: bool,
    // NEW parameters
    pub language_hint: Option<String>,
    pub search_mode: CodeSearchMode,
    pub include_ast_context: bool,
}

pub enum CodeSearchMode {
    /// Blend E1 and E7 (current behavior)
    Hybrid,
    /// E7 only (for code-specific queries)
    E7Only,
    /// E1 with E7 reranking
    E1WithE7Rerank,
    /// Full pipeline: E13 recall -> E7 dense -> E12 rerank
    Pipeline,
}
```

### 5.2 New `search_by_embedder` Tool (Embedder-First)

From `docs2/constitution.yaml`, embedder-first search is already planned:

```rust
// Implementation in crates/context-graph-mcp/src/handlers/tools/embedder_tools.rs
pub async fn call_search_by_embedder(
    &self,
    id: Option<JsonRpcId>,
    args: serde_json::Value,
) -> JsonRpcResponse {
    let request: SearchByEmbedderRequest = parse_request(args)?;

    // Allow searching with any embedder as primary
    let embedder = match request.embedder.as_str() {
        "E7" | "E7_CODE" => EmbedderName::E7Code,
        // ... other embedders
    };

    let options = TeleologicalSearchOptions::quick(request.top_k)
        .with_primary_embedder(embedder)
        .with_include_all_scores(request.include_all_scores);

    // Search using specified embedder
    self.teleological_store
        .search_by_embedder(&query_embedding, embedder, options)
        .await
}
```

### 5.3 Enrichment Pipeline Integration

From `docs2/constitution.yaml` autonomous enrichment section:

```rust
// Query type detection for automatic E7 selection
pub fn detect_code_query(query: &str) -> bool {
    let patterns = [
        "::", "->", "fn ", "function", "impl", ".await",
        "struct ", "enum ", "trait ", "pub ", "async ",
        "mod ", "use ", "crate::", "self.", "super::",
    ];
    patterns.iter().any(|p| query.contains(p))
}

// In enrichment pipeline
if detect_code_query(&query) {
    selected_enhancers.push(EmbedderName::E7Code);
    // Also add E6 for exact keyword matching
    selected_enhancers.push(EmbedderName::E6Sparse);
}
```

---

## Phase 6: Fine-Tuning Infrastructure

### 6.1 Training Data Format

**Triplet Format** (anchor, positive, negative):
```json
{
  "anchor": "function to store memory with importance score",
  "positive": "pub async fn store_memory(&self, content: &str, importance: f32) -> Result<Uuid>",
  "negative": "pub fn format_date(timestamp: i64) -> String"
}
```

**Contrastive Format** (query, document):
```json
{
  "query": "error handling in teleological store",
  "document": "impl TeleologicalStore { ... async fn store(&self, ...) -> Result<...> { ... } }"
}
```

### 6.2 Training Script

Adapt existing training infrastructure (`models/entity/train_script.py`):

**New File**: `models/code/train_e7.py`

```python
"""
E7 Fine-Tuning Script for Context Graph Codebase

Uses contrastive learning to improve code retrieval:
1. Load pre-trained Qodo-Embed-1-1.5B
2. Fine-tune on codebase triplets
3. Validate on held-out ground truth
4. Export to safetensors format
"""

import torch
from transformers import AutoModel, AutoTokenizer, AdamW
from torch.utils.data import DataLoader

class CodeEmbeddingModel(nn.Module):
    def __init__(self, model_name="Qodo/Qodo-Embed-1-1.5B"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        # Last-token pooling (per Qwen2 architecture)
        last_token = outputs.last_hidden_state[:, -1, :]
        return F.normalize(last_token, p=2, dim=1)

def train_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    for batch in dataloader:
        anchor_emb = model(batch['anchor_ids'], batch['anchor_mask'])
        positive_emb = model(batch['positive_ids'], batch['positive_mask'])
        negative_emb = model(batch['negative_ids'], batch['negative_mask'])

        # InfoNCE loss with in-batch negatives
        loss = loss_fn(anchor_emb, positive_emb, negative_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 6.3 Validation Pipeline

```python
def validate_e7(model, ground_truth_path):
    """Validate E7 on ground truth dataset."""
    results = {
        'P@1': [], 'P@5': [], 'P@10': [],
        'MRR': [], 'NDCG@10': []
    }

    with open(ground_truth_path) as f:
        for line in f:
            item = json.loads(line)
            query_emb = model.encode(item['query'])

            # Search indexed codebase
            hits = search_e7(query_emb, top_k=10)

            # Calculate metrics
            relevant = set(item['relevant'])
            results['P@1'].append(precision_at_k(hits, relevant, 1))
            results['P@5'].append(precision_at_k(hits, relevant, 5))
            # ... etc

    return {k: np.mean(v) for k, v in results.items()}
```

---

## Phase 7: Implementation Roadmap

### Week 1: Infrastructure

| Task | Files | Priority |
|------|-------|----------|
| Extend file watcher for `.rs` files | `config/default.toml`, `file_watcher_tools.rs` | P0 |
| Implement CodeChunker | `memory/code_chunker.rs` | P0 |
| Add language detection | `code_tools.rs` | P1 |
| Create index job CLI | `bin/index_codebase.rs` | P1 |

### Week 2: Ground Truth & Baseline

| Task | Files | Priority |
|------|-------|----------|
| Parse codebase AST | `bin/generate_ground_truth.rs` | P0 |
| Generate training triplets | `data/e7_training/` | P0 |
| Curate manual test queries | `data/e7_ground_truth/queries.jsonl` | P0 |
| Baseline benchmark run | `benchmark/e7_evaluation.rs` | P0 |

### Week 3: Parameter Tuning

| Task | Files | Priority |
|------|-------|----------|
| Grid search implementation | `benchmark/e7_tuning.rs` | P0 |
| Bayesian optimization | `benchmark/e7_tuning.rs` | P1 |
| Optimal parameter validation | N/A | P0 |
| Update defaults in code_tools.rs | `code_tools.rs` | P0 |

### Week 4: Integration & Fine-Tuning

| Task | Files | Priority |
|------|-------|----------|
| Fine-tuning script | `models/code/train_e7.py` | P1 |
| Export fine-tuned model | `models/code/finetuned/` | P1 |
| Integrate fine-tuned model | `models/pretrained/code/` | P1 |
| Final benchmark comparison | N/A | P0 |

---

## Phase 8: Success Criteria

### 8.1 Quantitative Targets

| Metric | Baseline (Est.) | Target | Stretch |
|--------|-----------------|--------|---------|
| P@1 | 0.45 | 0.60 | 0.70 |
| P@5 | 0.35 | 0.50 | 0.60 |
| MRR | 0.40 | 0.55 | 0.65 |
| Latency P95 | 80ms | 50ms | 30ms |
| E7 Unique Finds | N/A | +15% over E1 | +25% |

### 8.2 Qualitative Criteria

- [ ] E7 correctly identifies function signatures from natural language queries
- [ ] E7 finds `impl` blocks when searching for "implementation of X"
- [ ] E7 retrieves related code even with different naming conventions
- [ ] E7 handles multi-language queries (Rust + Python references)

### 8.3 Integration Criteria

- [ ] `search_code` tool uses tuned parameters
- [ ] Enrichment pipeline auto-selects E7 for code queries
- [ ] `search_by_embedder` with E7 works correctly
- [ ] File watcher indexes `.rs` files with code-aware chunking

---

## Appendix A: File Inventory

### Files to Create

```
# AST-Based Chunking (Phase 1)
crates/context-graph-core/src/memory/ast_chunker.rs
crates/context-graph-core/src/memory/ast_chunker/mod.rs
crates/context-graph-core/src/memory/ast_chunker/rust.rs
crates/context-graph-core/src/memory/ast_chunker/context.rs
crates/context-graph-core/src/memory/ast_chunker/description.rs

# Benchmarking (Phase 3)
crates/context-graph-benchmark/src/e7_evaluation.rs
crates/context-graph-benchmark/src/e7_iou.rs
crates/context-graph-benchmark/src/e7_tuning.rs

# Tooling
crates/context-graph-mcp/src/bin/index_codebase.rs
crates/context-graph-mcp/src/bin/generate_descriptions.rs

# Data
data/e7_ground_truth/queries.jsonl
data/e7_ground_truth/iou_spans.jsonl
data/e7_training/triplets.jsonl
data/e7_training/descriptions.jsonl

# Training
models/code/train_e7.py
models/code/generate_descriptions.py
models/code/finetuned/
```

### Files to Modify

```
# Configuration
config/default.toml                                    # Add .rs file watching

# Dependencies
crates/context-graph-core/Cargo.toml                   # Add tree-sitter deps

# MCP Handlers
crates/context-graph-mcp/src/handlers/tools/code_tools.rs
crates/context-graph-mcp/src/handlers/tools/embedder_tools.rs
crates/context-graph-mcp/src/handlers/tools/file_watcher_tools.rs
crates/context-graph-mcp/src/handlers/tools/enrichment_pipeline.rs
crates/context-graph-mcp/src/handlers/tools/query_type_detector.rs

# Core Memory
crates/context-graph-core/src/memory/mod.rs            # Export ast_chunker
```

### New Dependencies

```toml
# crates/context-graph-core/Cargo.toml
[dependencies]
tree-sitter = "0.20"
tree-sitter-rust = "0.20"

# Optional multi-language support
tree-sitter-python = { version = "0.20", optional = true }
tree-sitter-javascript = { version = "0.20", optional = true }
tree-sitter-typescript = { version = "0.20", optional = true }

[features]
multi-language = ["tree-sitter-python", "tree-sitter-javascript", "tree-sitter-typescript"]
```

---

## Appendix B: Constitution Compliance

This plan adheres to the following constitution rules:

| Rule | Compliance |
|------|------------|
| ARCH-01 | TeleologicalArray remains atomic (all 13 embeddings) |
| ARCH-12 | E1 remains the foundation; E7 enhances |
| ARCH-13 | Supports all strategies: E1Only, MultiSpace, Pipeline |
| ARCH-17 | Adapts E7 boost based on E1 strength |
| ARCH-21 | Uses Weighted RRF for multi-space fusion |
| AP-02 | No cross-embedder comparison (E1↔E7) |
| AP-04 | No partial TeleologicalArray |

---

## Appendix C: GPU Resource Requirements

From `docs2/constitution.yaml` GPU section:

| Resource | Available | This Plan Uses |
|----------|-----------|----------------|
| VRAM Total | 32GB | ~12GB (all 13 embedders + indexes) |
| E7 Model Size | ~3GB | ~3GB |
| Batch Buffer | ~4GB | ~2GB for indexing |
| Training (if local) | N/A | ~8GB (can use gradient checkpointing) |

---

## Appendix D: Risk Mitigation

| Risk | Mitigation |
|------|------------|
| E7 over-fits to this codebase | Validate on external code datasets (CodeSearchNet) |
| Ground truth bias | Use automated + manual curation |
| Parameter tuning overfitting | Hold out 20% of ground truth for final validation |
| Training data quality | Use multiple negative sampling strategies |
| Integration breaks existing tools | Feature flag for new behavior, gradual rollout |

---

## Appendix E: Research Sources

This plan incorporates best practices from the following sources:

### Academic Papers

| Paper | Key Contribution | Citation |
|-------|------------------|----------|
| **cAST** (EMNLP 2025) | AST-based divide-and-combine chunking | [arXiv:2506.15655](https://arxiv.org/html/2506.15655v1) |

### Industry Best Practices

| Source | Key Insights |
|--------|--------------|
| [Qodo Blog: State-of-the-Art Code Retrieval](https://www.qodo.ai/blog/qodo-embed-1-code-embedding-code-retrieval/) | Dual embedding (code + NL descriptions), synthetic docstrings |
| [Qodo Blog: RAG for Large-Scale Code Repos](https://www.qodo.ai/blog/rag-for-large-scale-code-repos/) | 500-char chunks, class definitions with methods, two-stage retrieval |
| [Supermemory: AST-Aware Chunking](https://supermemory.ai/blog/building-code-chunk-ast-aware-code-chunking/) | Context prepending, scope chains, IoU evaluation |
| [Qodo-Embed-1-1.5B Model Card](https://huggingface.co/Qodo/Qodo-Embed-1-1.5B) | Last-token pooling, L2 normalization, 8192 max_length |

### Tools & Libraries

| Tool | Purpose | Link |
|------|---------|------|
| tree-sitter | Multi-language AST parsing | [GitHub](https://github.com/tree-sitter/tree-sitter) |
| tree-sitter-rust | Rust grammar for tree-sitter | [GitHub](https://github.com/tree-sitter/tree-sitter-rust) |
| ASTChunk | Python AST chunking toolkit | [GitHub](https://github.com/yilinjz/astchunk) |
| LlamaIndex CodeSplitter | Python implementation reference | [LlamaIndex Docs](https://docs.llamaindex.ai/) |

### Key Findings Summary

1. **AST-based chunking improves retrieval by 1.2-5.5 points** across benchmarks (cAST paper)
2. **500 characters is optimal chunk size** for code embeddings (Qodo)
3. **Context prepending (file path, scope, imports) significantly improves embedding quality** (Supermemory)
4. **Dual embedding (code + NL description) bridges NL queries to code** (Qodo)
5. **Last-token pooling is critical for Qwen2-based models** like Qodo-Embed (Model Card)
6. **IoU measures true code overlap**, not just file-level retrieval (Supermemory)

---

*Document Version: 1.1*
*Created: 2026-01-24*
*Updated: 2026-01-24 (Added research-based best practices)*
*Author: Claude Code (context-graph development)*
*Next Review: After Phase 2 completion*
