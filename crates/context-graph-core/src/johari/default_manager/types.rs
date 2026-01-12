//! Core type definitions for DefaultJohariManager.

use std::sync::Arc;

use tokio::sync::RwLock;

use crate::traits::TeleologicalMemoryStore;
use crate::types::fingerprint::JohariThresholds;
use crate::types::JohariTransition;

/// Default implementation using TeleologicalMemoryStore.
///
/// This implementation:
/// - Stores transitions implicitly in JohariFingerprint state
/// - Uses existing classification methods from JohariFingerprint
/// - Validates transitions using JohariQuadrant::can_transition_to()
/// - Persists transition history for stats and history queries
pub struct DefaultJohariManager<S: TeleologicalMemoryStore> {
    /// The storage backend for teleological fingerprints
    pub(crate) store: Arc<S>,

    /// Threshold for blind spot detection.
    ///
    /// Constitution: `utl.johari.Blind` - default 0.5 via `JohariThresholds::default_general().blind_spot`
    pub(crate) blind_spot_threshold: f32,

    /// Maximum transition history per memory (default: 100)
    #[allow(dead_code)]
    pub(crate) max_history_per_memory: usize,

    /// In-memory transition history storage
    /// Stored in reverse chronological order (newest first)
    pub(crate) transitions: Arc<RwLock<Vec<JohariTransition>>>,
}

impl<S: TeleologicalMemoryStore> DefaultJohariManager<S> {
    /// Create a new DefaultJohariManager with the given store.
    pub fn new(store: Arc<S>) -> Self {
        Self {
            store,
            blind_spot_threshold: JohariThresholds::default_general().blind_spot,
            max_history_per_memory: 100,
            transitions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Set the blind spot detection threshold.
    ///
    /// Signal strengths above this threshold will be considered blind spot candidates.
    pub fn with_blind_spot_threshold(mut self, threshold: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be [0,1], got {}",
            threshold
        );
        self.blind_spot_threshold = threshold;
        self
    }

    /// Record a transition in the history.
    pub(crate) async fn record_transition(&self, transition: JohariTransition) {
        let mut transitions = self.transitions.write().await;
        // Insert at beginning (newest first)
        transitions.insert(0, transition);
        // TODO: In a production system, we might want to limit the size
        // or persist to disk. For now we keep all transitions in memory.
    }
}

/// Dynamic (type-erased) version of DefaultJohariManager.
///
/// Use this when working with `Arc<dyn TeleologicalMemoryStore>` trait objects,
/// such as in the MCP Handlers struct.
///
/// TASK-S004: Required for johari/* handlers in context-graph-mcp.
pub struct DynDefaultJohariManager {
    /// The storage backend for teleological fingerprints (trait object)
    pub(crate) store: Arc<dyn TeleologicalMemoryStore>,

    /// Threshold for blind spot detection.
    ///
    /// Constitution: `utl.johari.Blind` - default 0.5 via `JohariThresholds::default_general().blind_spot`
    pub(crate) blind_spot_threshold: f32,

    /// Maximum transition history per memory (default: 100)
    #[allow(dead_code)]
    pub(crate) max_history_per_memory: usize,

    /// In-memory transition history storage
    /// Stored in reverse chronological order (newest first)
    pub(crate) transitions: Arc<RwLock<Vec<JohariTransition>>>,
}

impl DynDefaultJohariManager {
    /// Create a new DynDefaultJohariManager with a trait object store.
    pub fn new(store: Arc<dyn TeleologicalMemoryStore>) -> Self {
        Self {
            store,
            blind_spot_threshold: JohariThresholds::default_general().blind_spot,
            max_history_per_memory: 100,
            transitions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Set the blind spot detection threshold.
    pub fn with_blind_spot_threshold(mut self, threshold: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be [0,1], got {}",
            threshold
        );
        self.blind_spot_threshold = threshold;
        self
    }

    /// Record a transition in the history.
    pub(crate) async fn record_transition(&self, transition: JohariTransition) {
        let mut transitions = self.transitions.write().await;
        // Insert at beginning (newest first)
        transitions.insert(0, transition);
    }
}
