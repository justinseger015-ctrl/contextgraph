//! Training tool definitions for causal embedder fine-tuning.
//!
//! Tools:
//! - train_causal_embedder: Train, evaluate, or incrementally update E5 projections

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns training tool definitions (1 tool).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![ToolDefinition::new(
        "train_causal_embedder",
        "Train, evaluate, or incrementally update the E5 causal embedder's projection matrices. \
         Fine-tunes W_cause and W_effect using InfoNCE contrastive loss (τ=0.05), directional margin, \
         separation loss, and soft-label distillation from LLM confidence scores. \
         Modes: 'full' trains from scratch using seed pairs + LLM expansion, \
         'evaluate' runs metrics on held-out data, \
         'incremental' does online distillation from recently discovered causal relationships. \
         Returns training metrics including directional accuracy, topical MRR, and loss curves.",
        json!({
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["full", "evaluate", "incremental"],
                    "description": "Training mode: 'full' (train from scratch, ~50 epochs), 'evaluate' (run metrics only), 'incremental' (online distillation from recent discoveries)"
                },
                "epochs": {
                    "type": "integer",
                    "description": "Number of training epochs (default: 50 for full, 10 for incremental)",
                    "minimum": 1,
                    "maximum": 200
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Batch size for training (default: 32)",
                    "minimum": 4,
                    "maximum": 256
                },
                "data_source": {
                    "type": "string",
                    "enum": ["seed", "llm_discovered", "both"],
                    "description": "Data source: 'seed' (35 built-in pairs), 'llm_discovered' (pairs from causal discovery), 'both' (combined)"
                }
            },
            "additionalProperties": false
        }),
    )]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_definitions_count() {
        let tools = definitions();
        assert_eq!(tools.len(), 1, "Should have 1 training tool");
    }

    #[test]
    fn test_train_causal_embedder_definition() {
        let tools = definitions();
        let train = tools
            .iter()
            .find(|t| t.name == "train_causal_embedder")
            .unwrap();

        assert!(train.description.contains("InfoNCE"));
        assert!(train.description.contains("W_cause"));

        // Should have mode, epochs, batch_size, data_source properties
        let props = train.input_schema.get("properties").unwrap();
        let prop_obj = props.as_object().unwrap();
        assert!(prop_obj.contains_key("mode"));
        assert!(prop_obj.contains_key("epochs"));
        assert!(prop_obj.contains_key("batch_size"));
        assert!(prop_obj.contains_key("data_source"));
    }
}
