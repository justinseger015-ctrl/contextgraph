//! E5 Embedder Activator for confirmed causal relationships.
//!
//! Takes confirmed causal pairs from LLM analysis and:
//! 1. Generates asymmetric E5 embeddings using CausalModel
//! 2. Updates the teleological store with new embeddings
//! 3. Adds edges to the CausalGraph
//!
//! # Architecture
//!
//! The activator bridges the causal discovery pipeline to the storage layer:
//!
//! ```text
//! LLM Analysis Result
//!         │
//!         ▼
//! ┌─────────────────┐
//! │ E5 Activator    │
//! │  - Load memories│
//! │  - embed_dual() │
//! │  - Update FPs   │
//! │  - Add edges    │
//! └─────────────────┘
//!         │
//!         ▼
//! Storage Layer (TeleologicalStore + CausalGraph)
//! ```

use std::sync::Arc;

use parking_lot::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

use context_graph_core::causal::{CausalEdge, CausalGraph, CausalNode};

use crate::error::{CausalAgentError, CausalAgentResult};
use crate::types::{CausalAnalysisResult, CausalLinkDirection};

/// Configuration for the E5 activator.
#[derive(Debug, Clone)]
pub struct ActivatorConfig {
    /// Minimum confidence to embed a relationship.
    pub min_confidence: f32,

    /// Whether to update existing E5 embeddings (vs skip if present).
    pub update_existing: bool,

    /// Whether to add nodes to CausalGraph if not present.
    pub auto_create_nodes: bool,
}

impl Default for ActivatorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            update_existing: true,
            auto_create_nodes: true,
        }
    }
}

/// Statistics from activation operations.
#[derive(Debug, Clone, Default)]
pub struct ActivationStats {
    /// Number of relationships processed.
    pub processed: usize,

    /// Number of E5 embeddings generated.
    pub embeddings_generated: usize,

    /// Number of graph edges created.
    pub edges_created: usize,

    /// Number of relationships skipped (low confidence).
    pub skipped_low_confidence: usize,

    /// Number of relationships skipped (already exists).
    pub skipped_existing: usize,

    /// Number of errors encountered.
    pub errors: usize,
}

/// Activator for embedding confirmed causal relationships.
///
/// Coordinates between:
/// - CausalModel (for E5 asymmetric embeddings)
/// - TeleologicalStore (for fingerprint updates)
/// - CausalGraph (for relationship storage)
pub struct E5EmbedderActivator {
    /// Configuration.
    config: ActivatorConfig,

    /// Causal graph for storing relationships.
    causal_graph: Arc<RwLock<CausalGraph>>,

    /// Statistics.
    stats: RwLock<ActivationStats>,
}

impl E5EmbedderActivator {
    /// Create a new activator with default configuration.
    pub fn new(causal_graph: Arc<RwLock<CausalGraph>>) -> Self {
        Self::with_config(causal_graph, ActivatorConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(causal_graph: Arc<RwLock<CausalGraph>>, config: ActivatorConfig) -> Self {
        Self {
            config,
            causal_graph,
            stats: RwLock::new(ActivationStats::default()),
        }
    }

    /// Activate E5 embedding for a confirmed causal relationship.
    ///
    /// # Arguments
    ///
    /// * `cause_id` - UUID of the cause memory
    /// * `effect_id` - UUID of the effect memory
    /// * `cause_content` - Text content of the cause memory
    /// * `effect_content` - Text content of the effect memory
    /// * `analysis` - LLM analysis result
    ///
    /// # Returns
    ///
    /// Tuple of (cause_embedding, effect_embedding) if successful
    pub async fn activate_relationship(
        &self,
        cause_id: Uuid,
        effect_id: Uuid,
        cause_content: &str,
        effect_content: &str,
        analysis: &CausalAnalysisResult,
    ) -> CausalAgentResult<(Vec<f32>, Vec<f32>)> {
        let mut stats = self.stats.write();
        stats.processed += 1;

        // Check confidence threshold
        if analysis.confidence < self.config.min_confidence {
            debug!(
                cause = %cause_id,
                effect = %effect_id,
                confidence = analysis.confidence,
                threshold = self.config.min_confidence,
                "Skipping low confidence relationship"
            );
            stats.skipped_low_confidence += 1;
            return Err(CausalAgentError::ConfigError {
                message: format!(
                    "Confidence {} below threshold {}",
                    analysis.confidence, self.config.min_confidence
                ),
            });
        }

        // Check if relationship already exists in graph
        {
            let graph = self.causal_graph.read();
            if graph.has_direct_cause(cause_id, effect_id) {
                if !self.config.update_existing {
                    debug!(
                        cause = %cause_id,
                        effect = %effect_id,
                        "Skipping existing relationship"
                    );
                    stats.skipped_existing += 1;
                    return Err(CausalAgentError::ConfigError {
                        message: "Relationship already exists".to_string(),
                    });
                }
            }
        }

        // Generate E5 embeddings
        // In full implementation, this would call CausalModel.embed_dual()
        let (cause_embedding, effect_embedding) =
            self.generate_e5_embeddings(cause_content, effect_content).await?;

        stats.embeddings_generated += 2;

        // Add edge to causal graph
        self.add_graph_edge(cause_id, effect_id, cause_content, effect_content, analysis)?;
        stats.edges_created += 1;

        info!(
            cause = %cause_id,
            effect = %effect_id,
            confidence = analysis.confidence,
            mechanism = %analysis.mechanism,
            "Activated E5 causal relationship"
        );

        Ok((cause_embedding, effect_embedding))
    }

    /// Generate E5 asymmetric embeddings for cause and effect.
    ///
    /// In production, this calls CausalModel.embed_dual().
    /// For now, generates placeholder embeddings.
    async fn generate_e5_embeddings(
        &self,
        cause_content: &str,
        effect_content: &str,
    ) -> CausalAgentResult<(Vec<f32>, Vec<f32>)> {
        // In full implementation:
        // ```rust
        // let causal_model = self.causal_model.as_ref()
        //     .ok_or(CausalAgentError::EmbeddingError {
        //         message: "CausalModel not initialized".to_string()
        //     })?;
        //
        // let (cause_as_cause, _cause_as_effect) =
        //     causal_model.embed_dual(cause_content).await
        //         .map_err(CausalAgentError::embedding)?;
        //
        // let (_effect_as_cause, effect_as_effect) =
        //     causal_model.embed_dual(effect_content).await
        //         .map_err(CausalAgentError::embedding)?;
        //
        // Ok((cause_as_cause, effect_as_effect))
        // ```

        // For now, generate deterministic placeholder embeddings
        // based on content hash to enable testing
        let cause_emb = self.hash_to_embedding(cause_content, 768, true);
        let effect_emb = self.hash_to_embedding(effect_content, 768, false);

        Ok((cause_emb, effect_emb))
    }

    /// Generate a deterministic embedding from content hash.
    ///
    /// This is a placeholder for testing until CausalModel integration.
    fn hash_to_embedding(&self, content: &str, dim: usize, is_cause: bool) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        if is_cause {
            "cause".hash(&mut hasher);
        } else {
            "effect".hash(&mut hasher);
        }
        let hash = hasher.finish();

        // Generate pseudo-random normalized vector
        let mut embedding = Vec::with_capacity(dim);
        let mut current = hash;
        let mut sum_sq = 0.0f32;

        for _ in 0..dim {
            current = current.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let val = ((current >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            embedding.push(val);
            sum_sq += val * val;
        }

        // Normalize
        let norm = sum_sq.sqrt();
        if norm > f32::EPSILON {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }

    /// Add an edge to the causal graph.
    fn add_graph_edge(
        &self,
        cause_id: Uuid,
        effect_id: Uuid,
        cause_content: &str,
        effect_content: &str,
        analysis: &CausalAnalysisResult,
    ) -> CausalAgentResult<()> {
        let mut graph = self.causal_graph.write();

        // Auto-create nodes if configured
        if self.config.auto_create_nodes {
            if !graph.has_node(cause_id) {
                let node = CausalNode::with_id(
                    cause_id,
                    truncate_name(cause_content, 50),
                    "memory",
                );
                graph.add_node(node);
            }

            if !graph.has_node(effect_id) {
                let node = CausalNode::with_id(
                    effect_id,
                    truncate_name(effect_content, 50),
                    "memory",
                );
                graph.add_node(node);
            }
        }

        // Add edge based on direction
        match analysis.direction {
            CausalLinkDirection::ACausesB => {
                graph.add_edge(CausalEdge::new(
                    cause_id,
                    effect_id,
                    analysis.confidence,
                    &analysis.mechanism,
                ));
            }
            CausalLinkDirection::BCausesA => {
                graph.add_edge(CausalEdge::new(
                    effect_id,
                    cause_id,
                    analysis.confidence,
                    &analysis.mechanism,
                ));
            }
            CausalLinkDirection::Bidirectional => {
                // Add edges in both directions
                graph.add_edge(CausalEdge::new(
                    cause_id,
                    effect_id,
                    analysis.confidence * 0.8, // Slightly lower for bidirectional
                    &format!("{} (forward)", analysis.mechanism),
                ));
                graph.add_edge(CausalEdge::new(
                    effect_id,
                    cause_id,
                    analysis.confidence * 0.8,
                    &format!("{} (backward)", analysis.mechanism),
                ));
            }
            CausalLinkDirection::NoCausalLink => {
                // Should not happen, but handle gracefully
                warn!(
                    cause = %cause_id,
                    effect = %effect_id,
                    "Attempted to add edge for non-causal relationship"
                );
            }
        }

        Ok(())
    }

    /// Get activation statistics.
    pub fn stats(&self) -> ActivationStats {
        self.stats.read().clone()
    }

    /// Reset activation statistics.
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = ActivationStats::default();
    }

    /// Get the configuration.
    pub fn config(&self) -> &ActivatorConfig {
        &self.config
    }

    /// Get a reference to the causal graph.
    pub fn causal_graph(&self) -> &Arc<RwLock<CausalGraph>> {
        &self.causal_graph
    }
}

/// Truncate content to create a node name.
fn truncate_name(content: &str, max_len: usize) -> String {
    let trimmed = content.trim();
    if trimmed.len() <= max_len {
        trimmed.to_string()
    } else {
        format!("{}...", &trimmed[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_activate_relationship() {
        let graph = Arc::new(RwLock::new(CausalGraph::new()));
        let activator = E5EmbedderActivator::new(graph.clone());

        let cause_id = Uuid::new_v4();
        let effect_id = Uuid::new_v4();

        let analysis = CausalAnalysisResult {
            has_causal_link: true,
            direction: CausalLinkDirection::ACausesB,
            confidence: 0.85,
            mechanism: "Direct causation".to_string(),
            raw_response: None,
        };

        let result = activator
            .activate_relationship(
                cause_id,
                effect_id,
                "The bug caused the crash",
                "The crash affected users",
                &analysis,
            )
            .await;

        assert!(result.is_ok());

        // Check graph has the edge
        let graph = graph.read();
        assert!(graph.has_node(cause_id));
        assert!(graph.has_node(effect_id));
        assert!(graph.has_direct_cause(cause_id, effect_id));
    }

    #[tokio::test]
    async fn test_skip_low_confidence() {
        let graph = Arc::new(RwLock::new(CausalGraph::new()));
        let activator = E5EmbedderActivator::new(graph);

        let analysis = CausalAnalysisResult {
            has_causal_link: true,
            direction: CausalLinkDirection::ACausesB,
            confidence: 0.3, // Below threshold
            mechanism: "Weak evidence".to_string(),
            raw_response: None,
        };

        let result = activator
            .activate_relationship(
                Uuid::new_v4(),
                Uuid::new_v4(),
                "Content A",
                "Content B",
                &analysis,
            )
            .await;

        assert!(result.is_err());

        let stats = activator.stats();
        assert_eq!(stats.skipped_low_confidence, 1);
    }

    #[test]
    fn test_truncate_name() {
        assert_eq!(truncate_name("Short", 50), "Short");
        assert_eq!(
            truncate_name("This is a very long name that exceeds the maximum length", 20),
            "This is a very lo..."
        );
    }

    #[test]
    fn test_hash_to_embedding() {
        let graph = Arc::new(RwLock::new(CausalGraph::new()));
        let activator = E5EmbedderActivator::new(graph);

        let emb1 = activator.hash_to_embedding("test content", 768, true);
        let emb2 = activator.hash_to_embedding("test content", 768, true);
        let emb3 = activator.hash_to_embedding("test content", 768, false);

        // Same input = same output
        assert_eq!(emb1, emb2);

        // Different role = different output
        assert_ne!(emb1, emb3);

        // Check normalization
        let norm: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }
}
