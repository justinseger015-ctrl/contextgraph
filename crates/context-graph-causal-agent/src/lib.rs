//! Causal Discovery Agent for Context Graph
//!
//! This crate provides automated causal relationship discovery using a local LLM
//! (Qwen2.5-Instruct via Candle) to analyze existing memories and identify cause-effect
//! relationships, then triggers the E5 embedder to store genuine causal embeddings.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    CAUSAL DISCOVERY AGENT                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Memory Scanner → LLM Analysis → E5 Embedder Activation         │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Components
//!
//! - **LLM Module**: Candle-based Qwen2.5 inference with native CUDA (RTX 5090 optimized)
//! - **Scanner Module**: Finds candidate memory pairs for causal analysis
//! - **Activator Module**: Triggers E5 embedding for confirmed relationships
//! - **Service Module**: Background service for scheduled discovery
//!
//! # VRAM Budget (RTX 5090 32GB)
//!
//! | Model | Precision | VRAM | Performance |
//! |-------|-----------|------|-------------|
//! | Qwen2.5-3B | FP16 | ~6GB | ~60 tok/s |
//! | Qwen2.5-7B | FP16 | ~14GB | ~45 tok/s |
//! | Qwen2.5-7B | INT8 | ~8GB | ~80 tok/s |
//!
//! FP16/BF16 uses 5th-gen Tensor Cores for optimal performance.
//! INT8 leverages RTX 5090's 3,352 INT8 TOPS (2.5x vs RTX 4090).
//!
//! # Usage
//!
//! ```rust,ignore
//! use context_graph_causal_agent::{CausalDiscoveryService, CausalDiscoveryConfig};
//! use context_graph_causal_agent::types::MemoryForAnalysis;
//!
//! async fn example() {
//!     let config = CausalDiscoveryConfig::default();
//!     let service = CausalDiscoveryService::new(config).await.unwrap();
//!
//!     // Run a single discovery cycle with memories
//!     let memories: Vec<MemoryForAnalysis> = vec![]; // Load from storage
//!     let result = service.run_discovery_cycle(&memories).await.unwrap();
//!     println!("Discovered {} causal relationships", result.relationships_confirmed);
//! }
//! ```

pub mod activator;
pub mod error;
pub mod llm;
pub mod scanner;
pub mod service;
pub mod types;

// Re-exports
pub use activator::E5EmbedderActivator;
pub use error::{CausalAgentError, CausalAgentResult};
pub use llm::CausalDiscoveryLLM;
pub use scanner::MemoryScanner;
pub use service::{CausalDiscoveryConfig, CausalDiscoveryService, DiscoveryCycleResult};
pub use types::{CausalAnalysisResult, CausalCandidate, CausalLinkDirection};
