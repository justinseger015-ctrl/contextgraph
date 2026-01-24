# E11 KEPLER Integration Plan

**Version**: 1.0.0
**Date**: 2026-01-24
**Status**: Ready for Implementation

---

## Executive Summary

E11 (KEPLER) is now fully operational with GPU-accelerated TransE inference. This document outlines how to integrate E11's entity knowledge capabilities into the Context Graph MCP system to enhance semantic search with entity-aware reasoning.

**Key Insight**: E1 (semantic) finds content by meaning similarity, but misses entity relationships. E11 finds "Diesel" when searching for "database" because it *knows* Diesel is a database ORM. Combined, they produce superior answers.

### Good News: Core Infrastructure Already Complete!

The retrieval infrastructure is already E11-aware:
- **RRF Weights**: E11 weight = 0.5 (RELATIONAL_ENHANCER) in `SPACE_WEIGHTS`
- **Query Routing**: `QueryType::Entity` routes to `vec![0, 10]` (E1 + E11)
- **Index Config**: E11 at index 10, dimension 768 (KEPLER)
- **Insight Annotations**: E11 contributions are tracked and annotated

**What remains**: Active triggering of E11 paths through query detection and hook integration.

---

## 1. Current State Analysis

### 1.1 What's Working

| Component | Status | Location |
|-----------|--------|----------|
| KEPLER Model (768D) | **COMPLETE** | `models/kepler/` |
| GPU Forward Pass | **COMPLETE** | `kepler/pooling.rs`, `kepler/forward.rs` |
| TransE Operations | **COMPLETE** | `kepler/transe.rs` |
| E11 Tool Definitions (6) | **COMPLETE** | `tools/definitions/entity.rs` |
| E11 Tool Handlers | **COMPLETE** | `handlers/tools/entity_tools.rs` |
| Multi-Embedder Scoring | **COMPLETE** | `combine_multi_embedder_scores()` |

### 1.2 What's Missing

| Component | Status | Priority |
|-----------|--------|----------|
| E11 in RRF Pipeline | **COMPLETE** | - |
| Entity Query Detection | **COMPLETE** | - |
| QueryTypeAnalyzer Entity Detection | **COMPLETE** | - |
| E11 in UserPromptSubmit Hook | **NOT INTEGRATED** | P1 |
| E11 Topic Contribution | **NEEDS VERIFICATION** | P1 |
| E11 Divergence Detection | **EXCLUDED BY DESIGN** | - |

---

## 2. Integration Architecture

### 2.1 E11's Role in the 13-Embedder System

```
Query: "What databases work with Rust?"

E1 (Semantic):     Finds "database", "Rust" semantically → misses "Diesel"
E11 (Entity/KEPLER): Knows Diesel IS a database ORM → surfaces "Diesel"
E7 (Code):         Finds sqlx, diesel crate usage → code patterns
Combined:          All of the above = superior answer
```

**Constitution Alignment**:
- **ARCH-12**: E1 is foundation - E11 ENHANCES, doesn't replace
- **ARCH-20**: E11 uses entity linking for disambiguation
- **RELATIONAL_ENHANCER**: topic_weight = 0.5

### 2.2 Multi-Embedder Discovery Philosophy

E11 doesn't just boost E1's scores - it **discovers candidates E1 missed**.

```
┌─────────────────────────────────────────────────────────────┐
│                    Query Processing                          │
├─────────────────────────────────────────────────────────────┤
│  1. Detect query type (entity-focused vs general)           │
│  2. E1 semantic search → Candidate Set A                    │
│  3. E11 entity search → Candidate Set B (different!)        │
│  4. UNION(A, B) → Combined candidates                       │
│  5. Score with combined insights (E1 + E11 + Jaccard)       │
│  6. RRF fusion if multi-space strategy                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Plan

### Phase 1: Already Complete - Infrastructure ✓

The following components are **already implemented**:

#### 3.1.1 E11 in RRF Pipeline ✓

**File**: `crates/context-graph-core/src/retrieval/config.rs:54`

```rust
// SPACE_WEIGHTS already includes E11
0.5, // E11: Entity (Relational)
```

#### 3.1.2 Query Type Analyzer ✓

**File**: `crates/context-graph-core/src/retrieval/query_analyzer.rs`

```rust
// QueryType::Entity already routes to E1 + E11
QueryType::Entity => vec![0, 10], // E1 + E11

// Entity detection via score_entity_query()
fn score_entity_query(&self, entities: &[String], query_lower: &str) -> f32 {
    // Detects capitalized terms + patterns like "who is", "what is"
}
```

#### 3.1.3 E11 in Multi-Space Search ✓

**File**: `crates/context-graph-core/src/retrieval/in_memory_executor.rs:431`

```rust
SpaceInfo::dense_hnsw(10, 768, 0, true), // E11 Entity (KEPLER)
```

### Phase 2: E11 in UserPromptSubmit Hook (P1)

**Goal**: When user submits a prompt, detect entity queries and inject E11-discovered context.

**File**: `.claude/hooks/user_prompt_submit.py`

```python
async def process_user_prompt(prompt: str, session_id: str):
    # 1. Detect query type
    entities = await call_mcp_tool("extract_entities", {"text": prompt})

    if entities["total_count"] > 0:
        # Entity-focused query detected

        # 2. Search using E11-aware multi-embedder discovery
        results = await call_mcp_tool("search_by_entities", {
            "entities": [e["canonical_id"] for e in entities["entities"]],
            "matchMode": "any",
            "topK": 5,
            "includeContent": True
        })

        # 3. Inject E11-discovered context
        if results["results"]:
            inject_context(
                results["results"],
                priority=2,  # Topic matches priority
                source="E11_entity_discovery"
            )
```

### Phase 3: E11 Topic Contribution (P1)

**Goal**: Verify E11 contributes to topic detection with weight 0.5.

**File**: `crates/context-graph-core/src/topics/detection.rs`

```rust
/// Compute weighted agreement for topic detection.
pub fn compute_weighted_agreement(clustered: &[bool; 13]) -> f32 {
    const WEIGHTS: [f32; 13] = [
        1.0, // E1 - semantic
        0.0, // E2 - temporal (excluded)
        0.0, // E3 - temporal (excluded)
        0.0, // E4 - temporal (excluded)
        1.0, // E5 - semantic
        1.0, // E6 - semantic
        1.0, // E7 - semantic
        0.5, // E8 - relational
        0.5, // E9 - structural
        1.0, // E10 - semantic
        0.5, // E11 - relational (VERIFY THIS)
        1.0, // E12 - semantic
        1.0, // E13 - semantic
    ];

    clustered.iter()
        .zip(WEIGHTS.iter())
        .filter(|(c, _)| **c)
        .map(|(_, w)| *w)
        .sum()
}
```

### Phase 4: E11 in Divergence Detection (P2)

**Current**: ARCH-10 says divergence detection uses SEMANTIC embedders only (E1,E5,E6,E7,E10,E12,E13).

**Question**: Should E11 contribute to divergence detection?

**Recommendation**: NO - keep E11 out of divergence detection because:
1. E11 is RELATIONAL, not SEMANTIC
2. Entity similarity doesn't indicate topic drift
3. Constitution explicitly excludes non-SEMANTIC embedders

---

## 4. MCP Tool Enhancement

### 4.1 Enhanced `search_graph` Parameters

Add E11-aware options to `search_graph`:

```json
{
  "name": "search_graph",
  "input_schema": {
    "properties": {
      "enableEntityEnhancement": {
        "type": "boolean",
        "default": true,
        "description": "Include E11 entity embeddings in multi-space search. Auto-enabled when entities detected in query."
      },
      "entityBoost": {
        "type": "number",
        "default": 1.3,
        "minimum": 1.0,
        "maximum": 2.0,
        "description": "Boost multiplier for results containing query entities."
      }
    }
  }
}
```

### 4.2 New Tool: `search_with_entities`

Convenience tool combining semantic and entity search:

```json
{
  "name": "search_with_entities",
  "description": "Hybrid search combining E1 semantic similarity with E11 entity knowledge. Automatically extracts entities from query, searches both spaces, and returns unified results. Use when searching for specific technologies, frameworks, or named concepts.",
  "input_schema": {
    "type": "object",
    "required": ["query"],
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language query. Entities will be auto-extracted."
      },
      "topK": {
        "type": "integer",
        "default": 10
      },
      "minScore": {
        "type": "number",
        "default": 0.3
      }
    }
  }
}
```

---

## 5. Performance Budget

Per PRD Section 9:

| Operation | Target | Notes |
|-----------|--------|-------|
| E11 embedding | <50ms | Part of 1000ms all-13 budget |
| E11 HNSW search | <5ms | Per-space lookup |
| Entity extraction | <20ms | Pattern matching + KB lookup |
| TransE scoring | <1ms | Vector arithmetic |
| Combined E1+E11 search | <100ms | Parallel search + RRF |

---

## 6. Testing Plan

### 6.1 Unit Tests

```rust
#[test]
fn test_e11_finds_what_e1_misses() {
    // Query: "database frameworks"
    // E1 finds: "PostgreSQL adapter", "MySQL connector"
    // E11 should find: "Diesel ORM" (knows Diesel IS a database framework)

    let e1_results = search_e1_only("database frameworks");
    let e11_results = search_e11_only("database frameworks");

    // E11 should surface Diesel even though "Diesel" doesn't contain "database"
    assert!(e11_results.iter().any(|r| r.contains("Diesel")));
    assert!(!e1_results.iter().any(|r| r.contains("Diesel"))); // E1 missed it
}

#[test]
fn test_transe_validates_known_relationships() {
    // Valid triple: (Diesel, depends_on, Rust)
    let score = validate_knowledge("Diesel", "depends_on", "Rust");
    assert!(score.validation == "valid");

    // Invalid triple: (Diesel, depends_on, Python)
    let score = validate_knowledge("Diesel", "depends_on", "Python");
    assert!(score.validation == "unlikely");
}
```

### 6.2 Integration Tests

```rust
#[tokio::test]
async fn test_e11_in_rrf_pipeline() {
    // Search with multi-space strategy
    let results = search_graph(
        "What ORM should I use with Rust?",
        strategy: "MultiSpace",
        enableEntityEnhancement: true,
    ).await;

    // Should find Diesel via E11 even though query doesn't mention it
    assert!(results.iter().any(|r| r.entities.contains("Diesel")));
}
```

### 6.3 Benchmark: E11 Discovery Rate

Measure how many relevant results E11 finds that E1 misses:

```
Metric: E11 Unique Discovery Rate
Formula: |E11_results - E1_results| / |E1_results|
Target: > 15% for entity-focused queries
```

---

## 7. Implementation Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Phase 1 | E11 in RRF pipeline, entity query detection |
| 2 | Phase 2 | UserPromptSubmit hook integration |
| 3 | Phase 3 | Topic contribution verification, benchmarks |
| 4 | Phase 4 | Documentation, edge cases, polish |

---

## 8. Success Criteria

1. **E11 Discovery Rate**: >15% unique discoveries on entity queries
2. **No Latency Regression**: Combined E1+E11 search <100ms
3. **TransE Accuracy**: Valid triples score > -5.0, invalid < -10.0
4. **Topic Detection**: E11 contributes 0.5 weight to agreement score
5. **All Tests Pass**: 25 kepler unit tests + 6 doc tests

---

## 9. Appendix: KEPLER Model Details

### 9.1 Model Architecture

- **Base**: RoBERTa-base (125M parameters)
- **Dimension**: 768D (aligned with E5, E10)
- **Training**: TransE objective on Wikidata5M
- **Location**: `models/kepler/`

### 9.2 TransE Operations

```rust
// TransE: h + r ≈ t for valid triples
transe_score(h, r, t) = -||h + r - t||₂

// Relation inference: r̂ = t - h
predict_relation(h, t) -> Vec<f32>

// Tail prediction: t̂ = h + r
predict_tail(h, r) -> Vec<f32>
```

### 9.3 Score Interpretation

| Score Range | Interpretation | Confidence |
|-------------|----------------|------------|
| > -5.0 | Valid triple | High (>0.7) |
| -10.0 to -5.0 | Uncertain | Medium (0.3-0.7) |
| < -10.0 | Invalid triple | Low (<0.3) |

---

## 10. Summary: What's Done vs What Remains

### Already Complete ✓

| Component | Status | Files |
|-----------|--------|-------|
| KEPLER Model (768D GPU) | ✓ | `models/kepler/`, `kepler/pooling.rs` |
| TransE Operations | ✓ | `kepler/transe.rs` |
| 6 MCP Entity Tools | ✓ | `entity.rs`, `entity_tools.rs` |
| RRF Weight Config | ✓ | `config.rs` (weight=0.5) |
| Query Type Analyzer | ✓ | `query_analyzer.rs` |
| Multi-Space E11 Search | ✓ | `in_memory_executor.rs` |
| Insight Annotations | ✓ | `insight_annotation.rs` |

### Remaining Work

| Task | Priority | Effort |
|------|----------|--------|
| UserPromptSubmit Hook Integration | P1 | Medium |
| Topic Contribution Verification | P1 | Low |
| End-to-End Testing | P1 | Medium |
| Benchmarking E11 Discovery Rate | P2 | Low |

### Quick Start

To use E11 entity capabilities immediately:

```python
# Extract entities from text
entities = await mcp.call("extract_entities", {"text": "Using Diesel ORM with PostgreSQL in Rust"})
# Returns: [{"canonical_id": "diesel", "entity_type": "Framework"}, ...]

# Search by entities (multi-embedder discovery)
results = await mcp.call("search_by_entities", {
    "entities": ["Diesel", "PostgreSQL"],
    "matchMode": "any",
    "topK": 10
})
# Returns memories found by E1 OR E11 (E11 finds what E1 misses!)

# Validate a knowledge triple
validation = await mcp.call("validate_knowledge", {
    "subject": "Diesel",
    "predicate": "depends_on",
    "object": "Rust"
})
# Returns: {"validation": "valid", "confidence": 0.85}
```

---

## 11. References

- PRD: `/home/cabdru/contextgraph/docs2/contextprd.md`
- Constitution: `/home/cabdru/contextgraph/docs2/constitution.yaml`
- CLAUDE.md: `/home/cabdru/contextgraph/CLAUDE.md`
- E11 Tool Definitions: `crates/context-graph-mcp/src/tools/definitions/entity.rs`
- E11 Tool Handlers: `crates/context-graph-mcp/src/handlers/tools/entity_tools.rs`
- KEPLER Model: `crates/context-graph-embeddings/src/models/pretrained/kepler/`
- Query Analyzer: `crates/context-graph-core/src/retrieval/query_analyzer.rs`
- RRF Config: `crates/context-graph-core/src/retrieval/config.rs`
