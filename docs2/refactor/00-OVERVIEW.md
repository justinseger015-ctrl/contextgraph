# Teleological Array System Refactor

## Executive Summary

This refactor fundamentally changes how the context-graph system stores, searches, and compares memory embeddings. The core principle is **apples-to-apples comparisons** - embeddings from the same embedder type are compared to each other, and full teleological arrays are compared to other full arrays.

## The Problem

The current system has a broken North Star implementation that:
1. Creates single 1024D embeddings for "goals"
2. Attempts to compare these to 13-embedder teleological arrays
3. Uses meaningless "projection" (dimension reduction) to fake comparisons
4. Compares semantic embeddings to temporal, entity, syntactic embeddings - nonsensical

This is **apples to oranges** - you cannot meaningfully compare:
- A semantic embedding to a temporal embedding
- A single 1024D vector to a multi-array structure
- Different embedding spaces by just reducing dimensions

## The Solution

### Teleological Arrays as the Fundamental Unit

Every stored item is represented as a **teleological array** - a structured collection of 13 embedding vectors:

```
TeleologicalArray = [
  E1:  Semantic (1024D)      - Core meaning
  E2:  Temporal Recent (512D) - Recency patterns
  E3:  Temporal Periodic (512D) - Cyclical patterns
  E4:  Entity Relationship (768D) - Entity links
  E5:  Causal (512D)         - Cause-effect chains
  E6:  SPLADE Sparse         - Keyword precision
  E7:  Contextual (1024D)    - Discourse context
  E8:  Emotional (256D)      - Affective valence
  E9:  Syntactic (512D)      - Structural patterns
  E10: Pragmatic (512D)      - Intent/function
  E11: Cross-Modal (768D)    - Multi-modal links
  E12: Late Interaction (token-level) - Fine retrieval
  E13: Keyword SPLADE Sparse - Term matching
]
```

### Apples-to-Apples Comparison Modes

**Mode 1: Single Embedder Comparison**
- Compare E1 to E1, E4 to E4, etc.
- Each embedder captures different semantic dimensions
- Useful for targeted queries (e.g., "find entities like X")

**Mode 2: Full Array Comparison**
- Compare entire 13-element array to another 13-element array
- Weighted aggregation across all spaces
- Comprehensive similarity assessment

**Mode 3: Embedder Group Comparison**
- Compare subsets of embedders (e.g., temporal group: E2+E3)
- Domain-specific similarity (e.g., causal reasoning: E5)
- Functional groupings for specific use cases

**Mode 4: Matrix Search Strategies**
- Cross-embedder correlation analysis
- Identify patterns across embedding spaces
- Optimize search based on query characteristics

## Document Index

| Document | Description |
|----------|-------------|
| [01-ARCHITECTURE.md](./01-ARCHITECTURE.md) | Core system architecture and data models |
| [02-STORAGE.md](./02-STORAGE.md) | Storage layer refactor specification |
| [03-SEARCH.md](./03-SEARCH.md) | Search system with matrix strategies |
| [04-COMPARISON.md](./04-COMPARISON.md) | Comparison operations specification |
| [05-NORTH-STAR-REMOVAL.md](./05-NORTH-STAR-REMOVAL.md) | Removal of broken manual North Star system |
| [06-AUTONOMOUS-INTEGRATION.md](./06-AUTONOMOUS-INTEGRATION.md) | Integration with autonomous goal system |
| [07-TASK-BREAKDOWN.md](./07-TASK-BREAKDOWN.md) | Implementation tasks and phases |
| [08-MCP-TOOLS.md](./08-MCP-TOOLS.md) | Updated MCP tool specifications |

## Key Principles

1. **Type Safety**: Teleological arrays are strongly typed, ensuring correct comparisons
2. **Apples-to-Apples**: Only compare compatible embedding types
3. **Autonomous First**: Goals emerge from data patterns, not manual configuration
4. **Matrix Flexibility**: Support various search strategies via configurable matrices
5. **No Projection Hacks**: Never fake compatibility through dimension reduction

## Success Criteria

- [ ] All storage uses teleological arrays as fundamental unit
- [ ] All searches compare like-to-like embeddings
- [ ] Manual North Star creation completely removed
- [ ] Matrix search strategies fully operational
- [ ] Autonomous goal discovery working with real teleological data
- [ ] MCP tools updated to expose new capabilities
