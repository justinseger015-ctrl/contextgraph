# TASK-LOGIC-012: Entry-Point Selection Heuristics

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-LOGIC-012 |
| **Title** | Entry-Point Selection Heuristics |
| **Status** | :white_circle: todo |
| **Layer** | Logic |
| **Sequence** | 22 |
| **Estimated Days** | 2 |
| **Complexity** | Medium |

## Implements

- **ARCH-04**: Entry-Point Discovery for Retrieval (CRITICAL)
- **REQ-SEARCH-01**: Query enters through ONE embedding space
- **REQ-SEARCH-02**: Optimal embedder selection

## Dependencies

| Task | Reason |
|------|--------|
| TASK-CORE-002 | Uses Embedder enum definitions |

## Objective

Implement `EntryPointSelector` trait and `DefaultEntryPointSelector` that:
1. Analyzes query intent (semantic, temporal, causal, code, entity)
2. Selects optimal embedding space as entry point
3. Supports learned thresholds from ReasoningBank
4. Recommends candidates_per_space based on complexity

## Context

**ARCH-04 Requirement (Constitution lines 225-241):**
> All searches MUST follow the entry-point discovery pattern:
> 1. Query enters through ONE embedding space (entry point)
> 2. Fast ANN search in that single HNSW index
> 3. Retrieve full TeleologicalArrays for candidates
> 4. Multi-space reranking with RRF fusion

**Current Problem:** TASK-LOGIC-008 hardcodes E6 (SPLADE) as entry point, violating ARCH-04's dynamic selection requirement.

**Impact:** O(n) search vs O(13n) if searching all indices. MUVERA research shows 90% latency reduction.

## Scope

### In Scope

- `EntryPointSelector` trait for pluggable selection
- `DefaultEntryPointSelector` with pattern-based intent detection
- Intent patterns for: semantic, temporal, causal, code, entity, exploratory
- Candidate count recommendations per intent
- Integration point for learned thresholds

### Out of Scope

- ML-based intent classification (future enhancement)
- Query expansion before selection
- A/B testing infrastructure for selection strategies

## Definition of Done

### Signatures

```rust
// crates/context-graph-storage/src/teleological/search/entry_point.rs

use context_graph_core::teleology::{
    embedder::Embedder,
    array::TeleologicalArray,
};
use regex::Regex;
use std::collections::HashMap;

/// Detected query intent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryIntent {
    /// General semantic similarity (default)
    Semantic,
    /// Time-based queries ("recent", "yesterday", "last week")
    Temporal,
    /// Cause-effect queries ("why", "because", "caused by")
    Causal,
    /// Code/technical queries (function names, error messages)
    Code,
    /// Entity-focused queries ("who", "what is", named entities)
    Entity,
    /// Open-ended exploration (no clear pattern)
    Exploratory,
    /// Keyword/exact match queries
    Keyword,
}

impl QueryIntent {
    /// Get recommended entry-point embedder for this intent
    pub fn default_embedder(&self) -> Embedder {
        match self {
            Self::Semantic => Embedder::Semantic,           // E1
            Self::Temporal => Embedder::TemporalRecent,     // E2
            Self::Causal => Embedder::Causal,               // E5
            Self::Code => Embedder::Contextual,             // E7
            Self::Entity => Embedder::EntityRelationship,   // E4
            Self::Exploratory => Embedder::Semantic,        // E1 (safe default)
            Self::Keyword => Embedder::Splade,              // E6
        }
    }

    /// Get recommended candidate count for this intent
    pub fn recommended_candidates(&self) -> usize {
        match self {
            Self::Semantic => 100,
            Self::Temporal => 50,
            Self::Causal => 75,
            Self::Code => 100,
            Self::Entity => 50,
            Self::Exploratory => 150,
            Self::Keyword => 200,
        }
    }
}

/// Trait for entry-point selection strategies
#[async_trait]
pub trait EntryPointSelector: Send + Sync {
    /// Select optimal entry-point embedder for query
    ///
    /// # Arguments
    /// * `query` - The query as a TeleologicalArray
    /// * `intent` - Pre-detected intent (optional override)
    ///
    /// # Returns
    /// Selected embedder for entry-point search
    async fn select_space(
        &self,
        query: &TeleologicalArray,
        intent: Option<QueryIntent>,
    ) -> Result<Embedder, SearchError>;

    /// Analyze query text to detect intent
    fn analyze_intent(&self, query_text: &str) -> QueryIntent;

    /// Get recommended candidate count for intent
    fn recommend_candidates(&self, intent: QueryIntent) -> usize;

    /// Get all possible entry points with confidence scores
    fn rank_entry_points(
        &self,
        query_text: &str,
    ) -> Vec<(Embedder, f32)>;
}

/// Default entry-point selector using pattern matching
pub struct DefaultEntryPointSelector {
    intent_patterns: HashMap<QueryIntent, Vec<Regex>>,
    learned_weights: Option<HashMap<QueryIntent, f32>>,
}

impl DefaultEntryPointSelector {
    /// Create with default patterns
    pub fn new() -> Self;

    /// Create with custom patterns
    pub fn with_patterns(patterns: HashMap<QueryIntent, Vec<Regex>>) -> Self;

    /// Update learned weights from ReasoningBank
    pub fn update_weights(&mut self, weights: HashMap<QueryIntent, f32>);
}

impl Default for DefaultEntryPointSelector {
    fn default() -> Self {
        let mut patterns = HashMap::new();

        // Temporal patterns
        patterns.insert(QueryIntent::Temporal, vec![
            Regex::new(r"\b(recent|latest|yesterday|today|last\s+\w+|ago)\b").unwrap(),
            Regex::new(r"\b(when|time|date|morning|evening|week|month)\b").unwrap(),
        ]);

        // Causal patterns
        patterns.insert(QueryIntent::Causal, vec![
            Regex::new(r"\b(why|because|cause|reason|therefore|hence)\b").unwrap(),
            Regex::new(r"\b(led\s+to|resulted\s+in|due\s+to|effect)\b").unwrap(),
        ]);

        // Entity patterns
        patterns.insert(QueryIntent::Entity, vec![
            Regex::new(r"\b(who|what\s+is|define|named|called)\b").unwrap(),
            Regex::new(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+").unwrap(), // Proper nouns
        ]);

        // Code patterns
        patterns.insert(QueryIntent::Code, vec![
            Regex::new(r"\b(function|class|method|error|exception|bug)\b").unwrap(),
            Regex::new(r"[a-z]+_[a-z]+|[a-z]+[A-Z][a-z]+").unwrap(), // snake_case or camelCase
            Regex::new(r"\b\w+\(\)").unwrap(), // function calls
        ]);

        // Keyword patterns
        patterns.insert(QueryIntent::Keyword, vec![
            Regex::new(r#""[^"]+""#).unwrap(), // Quoted strings
            Regex::new(r"\bexact(ly)?\b").unwrap(),
        ]);

        Self {
            intent_patterns: patterns,
            learned_weights: None,
        }
    }
}

#[async_trait]
impl EntryPointSelector for DefaultEntryPointSelector {
    async fn select_space(
        &self,
        _query: &TeleologicalArray,
        intent: Option<QueryIntent>,
    ) -> Result<Embedder, SearchError> {
        let detected = intent.unwrap_or_else(|| QueryIntent::Semantic);
        Ok(detected.default_embedder())
    }

    fn analyze_intent(&self, query_text: &str) -> QueryIntent {
        let query_lower = query_text.to_lowercase();

        // Score each intent based on pattern matches
        let mut scores: HashMap<QueryIntent, f32> = HashMap::new();

        for (intent, patterns) in &self.intent_patterns {
            let match_count = patterns.iter()
                .filter(|p| p.is_match(&query_lower))
                .count();

            if match_count > 0 {
                let base_score = match_count as f32;
                let weight = self.learned_weights
                    .as_ref()
                    .and_then(|w| w.get(intent))
                    .copied()
                    .unwrap_or(1.0);
                scores.insert(*intent, base_score * weight);
            }
        }

        // Return highest scoring intent, default to Semantic
        scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(intent, _)| intent)
            .unwrap_or(QueryIntent::Semantic)
    }

    fn recommend_candidates(&self, intent: QueryIntent) -> usize {
        intent.recommended_candidates()
    }

    fn rank_entry_points(&self, query_text: &str) -> Vec<(Embedder, f32)> {
        let intent = self.analyze_intent(query_text);
        let primary = intent.default_embedder();

        // Return ranked list with primary first
        let mut ranked = vec![
            (primary, 1.0),
            (Embedder::Semantic, 0.8),  // Always include semantic as fallback
        ];

        // Add secondary options based on intent
        match intent {
            QueryIntent::Temporal => {
                ranked.push((Embedder::TemporalPeriodic, 0.7));
            }
            QueryIntent::Entity => {
                ranked.push((Embedder::Contextual, 0.6));
            }
            QueryIntent::Code => {
                ranked.push((Embedder::LateInteraction, 0.7));
            }
            _ => {}
        }

        ranked
    }
}

/// Integration with TASK-LOGIC-008 pipeline
pub struct PipelineEntryPointIntegration;

impl PipelineEntryPointIntegration {
    /// Update LOGIC-008 to use dynamic entry point
    ///
    /// Before: Stage 1 hardcoded to E6 (SPLADE)
    /// After: Stage 1 uses EntryPointSelector
    pub fn integrate_with_pipeline(
        selector: Box<dyn EntryPointSelector>,
    ) -> SearchPipelineConfig;
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Intent detection latency | < 1ms |
| Pattern matching | Regex compiled once |
| Default to safe option | Semantic if uncertain |

## Verification

- [ ] All 7 intent types correctly detected
- [ ] Temporal queries ("yesterday", "recent") -> E2
- [ ] Causal queries ("why", "because") -> E5
- [ ] Code queries (function names, errors) -> E7
- [ ] Entity queries ("who is", proper nouns) -> E4
- [ ] Keyword queries (quoted strings) -> E6
- [ ] Unknown patterns -> E1 (semantic, safe default)
- [ ] Performance: < 1ms for intent detection

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-storage/src/teleological/search/entry_point.rs` | Entry-point selection |
| Update `crates/context-graph-storage/src/teleological/search/mod.rs` | Export entry_point |

## Files to Update

| File | Change |
|------|--------|
| `TASK-LOGIC-008` | Add dependency on LOGIC-012, replace hardcoded E6 |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Wrong intent detected | Medium | Medium | Semantic fallback is safe |
| Patterns too narrow | Low | Low | Configurable patterns |
| Performance regression | Low | High | Benchmark integration |

## Traceability

- Source: Constitution ARCH-04 (lines 225-241)
- Fixes: TASK-LOGIC-008 hardcoded entry point violation
