//! Core tool definitions per PRD v6 Section 10.
//!
//! Tools: inject_context, store_memory, get_memetic_status, search_graph, trigger_consolidation

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns core tool definitions (5 tools per PRD).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // inject_context - primary context injection tool
        ToolDefinition::new(
            "inject_context",
            "Inject context into the knowledge graph with UTL processing. \
             Analyzes content for learning potential and stores with computed metrics.",
            json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to inject into the knowledge graph"
                    },
                    "rationale": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1024,
                        "description": "Why this context is relevant and should be stored (REQUIRED, 1-1024 chars)"
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["text", "code", "image", "audio", "structured", "mixed"],
                        "default": "text",
                        "description": "The type/modality of the content"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "description": "Importance score for the memory [0.0, 1.0]"
                    }
                },
                "required": ["content", "rationale"]
            }),
        ),
        // store_memory - store a memory node directly
        ToolDefinition::new(
            "store_memory",
            "Store a memory node directly in the knowledge graph without UTL processing.",
            json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to store"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "description": "Importance score for the memory [0.0, 1.0]"
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["text", "code", "image", "audio", "structured", "mixed"],
                        "default": "text",
                        "description": "The type/modality of the content"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional tags for categorization"
                    }
                },
                "required": ["content"]
            }),
        ),
        // get_memetic_status - get UTL metrics and system state
        ToolDefinition::new(
            "get_memetic_status",
            "Get current system status with LIVE UTL metrics from the UtlProcessor: \
             entropy (novelty), coherence (understanding), learning score (magnitude), \
             consolidation phase, and suggested action. \
             Also returns node count and 4-layer bio-nervous system status.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),
        // search_graph - semantic search with E5 causal asymmetric similarity (ARCH-15, AP-77)
        ToolDefinition::new(
            "search_graph",
            "Search the knowledge graph using multi-space semantic similarity. \
             For causal queries ('why', 'what happens'), automatically applies \
             asymmetric E5 similarity with direction modifiers (cause→effect 1.2x, \
             effect→cause 0.8x). Returns nodes matching the query with relevance scores.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query text"
                    },
                    "topK": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "Maximum number of results to return"
                    },
                    "minSimilarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.0,
                        "description": "Minimum similarity threshold [0.0, 1.0]"
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["text", "code", "image", "audio", "structured", "mixed"],
                        "description": "Filter results by modality"
                    },
                    "includeContent": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include content text in results"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["e1_only", "multi_space", "pipeline"],
                        "default": "e1_only",
                        "description": "Search strategy: e1_only (fast), multi_space (balanced), pipeline (accurate)"
                    },
                    "weightProfile": {
                        "type": "string",
                        "enum": ["semantic_search", "causal_reasoning", "code_search", "fact_checking", "temporal_navigation", "category_weighted"],
                        "description": "Weight profile for multi-space search (auto-selected for causal queries)"
                    },
                    "enableRerank": {
                        "type": "boolean",
                        "default": false,
                        "description": "Enable ColBERT E12 re-ranking (Stage 3)"
                    },
                    "enableAsymmetricE5": {
                        "type": "boolean",
                        "default": true,
                        "description": "Enable asymmetric E5 causal reranking for detected causal queries"
                    },
                    "causalDirection": {
                        "type": "string",
                        "enum": ["auto", "cause", "effect", "none"],
                        "default": "auto",
                        "description": "Causal direction: auto (detect from query), cause (seeking causes), effect (seeking effects), none (disable)"
                    },
                    "enableQueryExpansion": {
                        "type": "boolean",
                        "default": false,
                        "description": "Expand causal queries with related terms for better recall"
                    },
                    "temporalWeight": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.0,
                        "description": "Weight for temporal post-retrieval boost [0.0, 1.0]"
                    }
                },
                "required": ["query"]
            }),
        ),
        // trigger_consolidation - trigger memory consolidation (PRD Section 10.1)
        ToolDefinition::new(
            "trigger_consolidation",
            "Trigger memory consolidation to merge similar memories and reduce redundancy. \
             Uses similarity-based, temporal, or semantic strategies to identify merge candidates. \
             Helps optimize memory storage and improve retrieval efficiency.",
            json!({
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["similarity", "temporal", "semantic"],
                        "default": "similarity",
                        "description": "Consolidation strategy to use"
                    },
                    "min_similarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.85,
                        "description": "Minimum similarity threshold for consolidation candidates"
                    },
                    "max_memories": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10000,
                        "default": 100,
                        "description": "Maximum memories to process in one batch"
                    }
                },
                "required": []
            }),
        ),
    ]
}
