//! MCP Server implementation.
//!
//! TASK-S001: Updated to use TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
//! TASK-S003: Added GoalAlignmentCalculator and GoalHierarchy for purpose operations.
//! TASK-S004: Replaced stubs with REAL implementations (RocksDB, UTL adapter).
//!
//! NO BACKWARDS COMPATIBILITY with stubs. FAIL FAST with clear errors.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use context_graph_core::alignment::{DefaultAlignmentCalculator, GoalAlignmentCalculator};
use context_graph_core::config::Config;
use context_graph_core::error::CoreError;
use context_graph_core::purpose::GoalHierarchy;
use context_graph_core::traits::{MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor};

// REAL implementations - NO STUBS
use context_graph_storage::teleological::RocksDbTeleologicalStore;
use crate::adapters::UtlProcessorAdapter;

use crate::handlers::Handlers;
use crate::protocol::{JsonRpcRequest, JsonRpcResponse};

// ============================================================================
// LazyFailMultiArrayProvider - FAIL FAST placeholder
// ============================================================================

/// Placeholder MultiArrayEmbeddingProvider that fails on first use with a clear error.
///
/// This is NOT a stub that returns fake data. It exists only to provide a clear,
/// actionable error message when embedding operations are attempted before the
/// real GPU implementation is ready.
///
/// # FAIL FAST Policy
///
/// - `embed_all()` -> Returns error with instructions
/// - `embed_single()` -> Returns error with instructions
/// - No silent failures, no fake embeddings
struct LazyFailMultiArrayProvider {
    error_message: String,
}

impl LazyFailMultiArrayProvider {
    fn new(message: &str) -> Self {
        Self {
            error_message: message.to_string(),
        }
    }
}

use context_graph_core::traits::MultiArrayEmbeddingOutput;
use context_graph_core::types::fingerprint::NUM_EMBEDDERS;

#[async_trait::async_trait]
impl MultiArrayEmbeddingProvider for LazyFailMultiArrayProvider {
    async fn embed_all(
        &self,
        _text: &str,
    ) -> Result<MultiArrayEmbeddingOutput, CoreError> {
        error!("FAIL FAST: {}", self.error_message);
        Err(CoreError::Embedding(self.error_message.clone()))
    }

    async fn embed_batch_all(
        &self,
        _contents: &[String],
    ) -> Result<Vec<MultiArrayEmbeddingOutput>, CoreError> {
        error!("FAIL FAST: {}", self.error_message);
        Err(CoreError::Embedding(self.error_message.clone()))
    }

    fn model_ids(&self) -> [&str; NUM_EMBEDDERS] {
        // Return placeholder IDs indicating not implemented
        [
            "NOT_IMPLEMENTED",
            "NOT_IMPLEMENTED",
            "NOT_IMPLEMENTED",
            "NOT_IMPLEMENTED",
            "NOT_IMPLEMENTED",
            "NOT_IMPLEMENTED",
            "NOT_IMPLEMENTED",
            "NOT_IMPLEMENTED",
            "NOT_IMPLEMENTED",
            "NOT_IMPLEMENTED",
            "NOT_IMPLEMENTED",
            "NOT_IMPLEMENTED",
            "NOT_IMPLEMENTED",
        ]
    }

    fn is_ready(&self) -> bool {
        // Always return false - we are NOT ready (FAIL FAST)
        false
    }

    fn health_status(&self) -> [bool; NUM_EMBEDDERS] {
        // All embedders unhealthy - not implemented
        [false; NUM_EMBEDDERS]
    }
}

// ============================================================================
// MCP Server
// ============================================================================

/// MCP Server state.
///
/// TASK-S001: Uses TeleologicalMemoryStore for 13-embedding fingerprint storage.
#[allow(dead_code)]
pub struct McpServer {
    config: Config,
    /// Teleological memory store - stores TeleologicalFingerprint with 13 embeddings.
    teleological_store: Arc<dyn TeleologicalMemoryStore>,
    utl_processor: Arc<dyn UtlProcessor>,
    /// Multi-array embedding provider - generates all 13 embeddings.
    multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
    handlers: Handlers,
    initialized: Arc<RwLock<bool>>,
}

impl McpServer {
    /// Create a new MCP server with the given configuration.
    ///
    /// TASK-S001: Creates TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
    /// TASK-S004: Uses REAL implementations - RocksDbTeleologicalStore, UtlProcessorAdapter.
    ///
    /// # Errors
    ///
    /// - Returns error if RocksDB fails to open (path issues, permissions, corruption)
    /// - Returns error if MultiArrayEmbeddingProvider is not yet implemented (FAIL FAST)
    pub async fn new(config: Config) -> Result<Self> {
        info!("Initializing MCP Server with REAL implementations (NO STUBS)...");

        // ==========================================================================
        // 1. Create RocksDB teleological store (REAL persistent storage)
        // ==========================================================================
        let db_path = Self::resolve_storage_path(&config);
        info!("Opening RocksDbTeleologicalStore at {:?}...", db_path);

        let teleological_store: Arc<dyn TeleologicalMemoryStore> = Arc::new(
            RocksDbTeleologicalStore::open(&db_path).map_err(|e| {
                error!("FATAL: Failed to open RocksDB at {:?}: {}", db_path, e);
                anyhow::anyhow!(
                    "Failed to open RocksDbTeleologicalStore at {:?}: {}. \
                     Check path exists, permissions, and RocksDB isn't locked by another process.",
                    db_path,
                    e
                )
            })?,
        );
        info!(
            "Created RocksDbTeleologicalStore at {:?} (17 column families, persistent storage)",
            db_path
        );

        // ==========================================================================
        // 2. Create REAL UTL processor (6-component computation)
        // ==========================================================================
        let utl_processor: Arc<dyn UtlProcessor> = Arc::new(UtlProcessorAdapter::with_defaults());
        info!("Created UtlProcessorAdapter (REAL 6-component UTL computation: deltaS, deltaC, wE, phi, lambda, magnitude)");

        // ==========================================================================
        // 3. MultiArrayEmbeddingProvider - FAIL FAST until real GPU implementation
        // ==========================================================================
        // The real MultiArrayEmbeddingProvider requires:
        // - GPU (CUDA) initialization
        // - 13 embedding models loaded
        // - ~8GB+ VRAM for all models
        //
        // Until TASK-F007 completes the GPU embedding infrastructure, we FAIL FAST
        // with a clear error message instead of silently using stubs.
        let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> = Arc::new(
            LazyFailMultiArrayProvider::new(
                "MultiArrayEmbeddingProvider not yet implemented. \
                 GPU embedding infrastructure required (TASK-F007). \
                 See crates/context-graph-embeddings for implementation status. \
                 Required: CUDA GPU with 8GB+ VRAM, 13 embedding models."
            )
        );
        warn!(
            "MultiArrayEmbeddingProvider using LAZY-FAIL placeholder. \
             First embedding operation will return clear error."
        );

        // TASK-S003: Create alignment calculator and empty goal hierarchy
        let alignment_calculator: Arc<dyn GoalAlignmentCalculator> =
            Arc::new(DefaultAlignmentCalculator::new());
        let goal_hierarchy = GoalHierarchy::new();
        info!("Created DefaultAlignmentCalculator and empty GoalHierarchy");

        let handlers = Handlers::new(
            Arc::clone(&teleological_store),
            Arc::clone(&utl_processor),
            Arc::clone(&multi_array_provider),
            alignment_calculator,
            goal_hierarchy,
        );

        info!("MCP Server initialization complete - TeleologicalFingerprint mode active");

        Ok(Self {
            config,
            teleological_store,
            utl_processor,
            multi_array_provider,
            handlers,
            initialized: Arc::new(RwLock::new(false)),
        })
    }

    /// Run the server, reading from stdin and writing to stdout.
    pub async fn run(&self) -> Result<()> {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut stdout = stdout.lock();

        info!("Server ready, waiting for requests (TeleologicalMemoryStore mode)...");

        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    error!("Error reading stdin: {}", e);
                    break;
                }
            };

            if line.trim().is_empty() {
                continue;
            }

            debug!("Received: {}", line);

            let response = self.handle_request(&line).await;

            // Handle notifications (no response needed)
            if response.id.is_none() && response.result.is_none() && response.error.is_none() {
                debug!("Notification handled, no response needed");
                continue;
            }

            let response_json = serde_json::to_string(&response)?;
            debug!("Sending: {}", response_json);

            // MCP requires newline-delimited JSON on stdout
            writeln!(stdout, "{}", response_json)?;
            stdout.flush()?;

            // Check for shutdown
            if !*self.initialized.read().await {
                // Not initialized yet, continue
            }
        }

        info!("Server shutting down...");
        Ok(())
    }

    /// Handle a single JSON-RPC request.
    async fn handle_request(&self, input: &str) -> JsonRpcResponse {
        // Parse request
        let request: JsonRpcRequest = match serde_json::from_str(input) {
            Ok(r) => r,
            Err(e) => {
                warn!("Failed to parse request: {}", e);
                return JsonRpcResponse::error(
                    None,
                    crate::protocol::error_codes::PARSE_ERROR,
                    format!("Parse error: {}", e),
                );
            }
        };

        // Validate JSON-RPC version
        if request.jsonrpc != "2.0" {
            return JsonRpcResponse::error(
                request.id,
                crate::protocol::error_codes::INVALID_REQUEST,
                "Invalid JSON-RPC version",
            );
        }

        // Dispatch to handler
        self.handlers.dispatch(request).await
    }

    /// Resolve the storage path from configuration or environment.
    ///
    /// Priority order:
    /// 1. `CONTEXT_GRAPH_STORAGE_PATH` environment variable
    /// 2. `config.storage.path` from configuration
    /// 3. Default: `./contextgraph_data` in current directory
    ///
    /// Creates the directory if it doesn't exist.
    fn resolve_storage_path(config: &Config) -> PathBuf {
        // Check environment variable first
        if let Ok(env_path) = std::env::var("CONTEXT_GRAPH_STORAGE_PATH") {
            let path = PathBuf::from(env_path);
            info!("Using storage path from CONTEXT_GRAPH_STORAGE_PATH: {:?}", path);
            Self::ensure_directory_exists(&path);
            return path;
        }

        // Use config path if it's not the default "memory" backend
        if config.storage.backend != "memory" && !config.storage.path.is_empty() {
            let path = PathBuf::from(&config.storage.path);
            info!("Using storage path from config: {:?}", path);
            Self::ensure_directory_exists(&path);
            return path;
        }

        // Default path
        let default_path = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("contextgraph_data");
        info!("Using default storage path: {:?}", default_path);
        Self::ensure_directory_exists(&default_path);
        default_path
    }

    /// Ensure a directory exists, creating it if necessary.
    fn ensure_directory_exists(path: &PathBuf) {
        if !path.exists() {
            info!("Creating storage directory: {:?}", path);
            if let Err(e) = std::fs::create_dir_all(path) {
                warn!(
                    "Failed to create storage directory {:?}: {}. \
                     RocksDB may fail to open.",
                    path, e
                );
            }
        }
    }
}
