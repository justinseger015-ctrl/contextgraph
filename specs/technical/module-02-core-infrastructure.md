# Module 2: Core Infrastructure - Technical Specification

```yaml
metadata:
  id: TECH-CORE-002
  version: 1.0.0
  module: Core Infrastructure
  phase: 1
  status: draft
  created: 2025-12-31
  functional_spec_ref: SPEC-CORE-INFRA
  author: Architecture Agent
  dependencies:
    - TECH-GHOST-001 (Module 1: Ghost System)
```

---

## 1. Architecture Overview

### 1.1 Module Dependency Graph

```
                    +------------------------+
                    |   context-graph-mcp    |
                    |      (binary crate)    |
                    +------------------------+
                               |
              +----------------+----------------+
              |                |                |
              v                v                v
+------------------+  +------------------+  +----------------------+
| context-graph-   |  | context-graph-   |  | context-graph-       |
|     core         |  |   storage        |  |     embeddings       |
|  (MemoryNode)    |  |   (RocksDB)      |  |   (FuseMoE 1536D)    |
+------------------+  +------------------+  +----------------------+
        |                     |                       |
        v                     v                       v
+------------------+  +------------------+  +----------------------+
| Johari Quadrant  |  | Memex Column     |  | Cognitive Pulse      |
| Classification   |  | Families         |  | Generator            |
+------------------+  +------------------+  +----------------------+
```

### 1.2 Module Structure

```
context-graph/
├── crates/
│   ├── context-graph-core/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── memory_node.rs          # MemoryNode struct
│   │   │   ├── johari.rs               # Johari quadrant logic
│   │   │   ├── metadata.rs             # NodeMetadata handling
│   │   │   ├── pulse.rs                # Cognitive Pulse
│   │   │   └── validation.rs           # Input validation
│   │   └── Cargo.toml
│   │
│   ├── context-graph-storage/          # NEW: Storage crate
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── memex.rs                # Memex storage abstraction
│   │   │   ├── rocksdb_backend.rs      # RocksDB implementation
│   │   │   ├── column_families.rs      # Column family definitions
│   │   │   ├── serialization.rs        # Bincode serialization
│   │   │   └── indexes.rs              # Secondary indexes
│   │   └── Cargo.toml
│   │
│   └── context-graph-embeddings/
│       ├── src/
│       │   ├── lib.rs
│       │   ├── provider.rs
│       │   └── fusemoe.rs              # FuseMoE 1536D embeddings
│       └── Cargo.toml
```

---

## 2. Data Structures

### 2.1 MemoryNode Structure

```rust
// crates/context-graph-core/src/memory_node.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for memory nodes
pub type NodeId = Uuid;

/// Embedding vector type (1536 dimensions from FuseMoE)
pub type EmbeddingVector = Vec<f32>;

/// Core memory node representing a stored memory unit in the knowledge graph.
///
/// Each MemoryNode encapsulates:
/// - Content: The raw information stored
/// - Embedding: 1536-dimensional vector from FuseMoE
/// - Johari Classification: Awareness quadrant placement
/// - Temporal Metadata: Access patterns and decay tracking
/// - Cognitive Metrics: Importance and emotional valence
///
/// # Performance Characteristics
/// - Serialized size: ~6.5KB average (with 1536D embedding)
/// - Insert latency target: <1ms
/// - Retrieval latency target: <500us
///
/// # Example
///
/// ```rust
/// use context_graph_core::{MemoryNode, JohariQuadrant, NodeMetadata};
/// use uuid::Uuid;
/// use chrono::Utc;
///
/// let node = MemoryNode {
///     id: Uuid::new_v4(),
///     content: "Important project decision".to_string(),
///     embedding: vec![0.0; 1536],
///     quadrant: JohariQuadrant::Open,
///     importance: 0.85,
///     emotional_valence: 0.3,
///     created_at: Utc::now(),
///     accessed_at: Utc::now(),
///     access_count: 0,
///     metadata: NodeMetadata::default(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryNode {
    /// Unique identifier (UUID v4)
    pub id: NodeId,

    /// Raw content stored in this node (text, code, structured data)
    /// Maximum length: 1MB
    pub content: String,

    /// Embedding vector from FuseMoE (1536 dimensions)
    /// Normalized to unit length for cosine similarity
    pub embedding: EmbeddingVector,

    /// Johari Window quadrant classification
    /// Determines visibility and awareness level
    pub quadrant: JohariQuadrant,

    /// Semantic importance score [0.0, 1.0]
    /// Higher values indicate more critical memories
    /// Used for consolidation and retrieval ranking
    pub importance: f32,

    /// Emotional valence [-1.0, 1.0]
    /// Negative: unpleasant associations
    /// Positive: pleasant associations
    /// Affects memory consolidation and retrieval priority
    pub emotional_valence: f32,

    /// Creation timestamp (UTC)
    pub created_at: DateTime<Utc>,

    /// Last access timestamp (UTC)
    /// Updated on every retrieval operation
    pub accessed_at: DateTime<Utc>,

    /// Number of times this node has been accessed
    /// Used for decay calculations and popularity ranking
    pub access_count: u32,

    /// Extended metadata for categorization and filtering
    pub metadata: NodeMetadata,
}

impl MemoryNode {
    /// Create a new MemoryNode with default values
    pub fn new(content: String, embedding: EmbeddingVector) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            content,
            embedding,
            quadrant: JohariQuadrant::default(),
            importance: 0.5,
            emotional_valence: 0.0,
            created_at: now,
            accessed_at: now,
            access_count: 0,
            metadata: NodeMetadata::default(),
        }
    }

    /// Create a MemoryNode with a specific ID (for deserialization)
    pub fn with_id(id: NodeId, content: String, embedding: EmbeddingVector) -> Self {
        let mut node = Self::new(content, embedding);
        node.id = id;
        node
    }

    /// Record an access event, updating timestamps and count
    pub fn record_access(&mut self) {
        self.accessed_at = Utc::now();
        self.access_count = self.access_count.saturating_add(1);
    }

    /// Calculate memory age in seconds
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.created_at).num_seconds()
    }

    /// Calculate time since last access in seconds
    pub fn time_since_access_seconds(&self) -> i64 {
        (Utc::now() - self.accessed_at).num_seconds()
    }

    /// Compute decay factor based on access patterns
    /// Uses modified Ebbinghaus forgetting curve
    ///
    /// decay = e^(-t / (S * ln(access_count + e)))
    /// where:
    ///   t = time since last access (hours)
    ///   S = stability factor (derived from importance)
    pub fn compute_decay(&self) -> f32 {
        let hours_since_access = self.time_since_access_seconds() as f32 / 3600.0;
        let stability = 24.0 * (1.0 + self.importance);
        let access_factor = (self.access_count as f32 + std::f32::consts::E).ln();

        (-hours_since_access / (stability * access_factor)).exp()
    }

    /// Check if this node should be consolidated to long-term memory
    /// Consolidation threshold based on importance and access patterns
    pub fn should_consolidate(&self) -> bool {
        let decay = self.compute_decay();
        let consolidation_score = self.importance * 0.6
            + decay * 0.2
            + (self.access_count.min(100) as f32 / 100.0) * 0.2;
        consolidation_score > 0.7
    }

    /// Validate the node's internal consistency
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Check embedding dimension
        if self.embedding.len() != 1536 {
            return Err(ValidationError::InvalidEmbeddingDimension {
                expected: 1536,
                actual: self.embedding.len(),
            });
        }

        // Check importance bounds
        if self.importance < 0.0 || self.importance > 1.0 {
            return Err(ValidationError::OutOfBounds {
                field: "importance".into(),
                value: self.importance,
                min: 0.0,
                max: 1.0,
            });
        }

        // Check emotional valence bounds
        if self.emotional_valence < -1.0 || self.emotional_valence > 1.0 {
            return Err(ValidationError::OutOfBounds {
                field: "emotional_valence".into(),
                value: self.emotional_valence,
                min: -1.0,
                max: 1.0,
            });
        }

        // Check content length
        if self.content.len() > 1_048_576 {
            return Err(ValidationError::ContentTooLarge {
                size: self.content.len(),
                max: 1_048_576,
            });
        }

        // Verify embedding is normalized (within tolerance)
        let norm: f32 = self.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if (norm - 1.0).abs() > 0.01 {
            return Err(ValidationError::EmbeddingNotNormalized { norm });
        }

        Ok(())
    }
}

/// Validation errors for MemoryNode
#[derive(Debug, Clone, thiserror::Error)]
pub enum ValidationError {
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidEmbeddingDimension { expected: usize, actual: usize },

    #[error("Field {field} value {value} out of bounds [{min}, {max}]")]
    OutOfBounds {
        field: String,
        value: f32,
        min: f32,
        max: f32,
    },

    #[error("Content too large: {size} bytes exceeds {max} byte limit")]
    ContentTooLarge { size: usize, max: usize },

    #[error("Embedding not normalized: norm = {norm}")]
    EmbeddingNotNormalized { norm: f32 },
}

impl Default for MemoryNode {
    fn default() -> Self {
        Self::new(String::new(), vec![0.0; 1536])
    }
}
```

### 2.2 Johari Quadrant Implementation

```rust
// crates/context-graph-core/src/johari.rs

use serde::{Deserialize, Serialize};

/// Johari Window quadrant classification for memory nodes.
///
/// The Johari Window is a psychological model that describes awareness
/// in terms of what is known/unknown to self and others. In the context
/// of the memory system:
///
/// - **Open**: Explicitly shared context, openly accessible
/// - **Hidden**: Private knowledge, known but not shared
/// - **Blind**: Patterns inferred by others but not self-aware
/// - **Unknown**: Latent patterns discovered through analysis
///
/// # Quadrant Characteristics
///
/// | Quadrant | Known to Self | Known to Others | Access Pattern |
/// |----------|---------------|-----------------|----------------|
/// | Open     | Yes           | Yes             | Public API     |
/// | Hidden   | Yes           | No              | Private store  |
/// | Blind    | No            | Yes             | External input |
/// | Unknown  | No            | No              | Discovery      |
///
/// # Example
///
/// ```rust
/// use context_graph_core::JohariQuadrant;
///
/// let quadrant = JohariQuadrant::Open;
/// assert!(quadrant.is_self_aware());
/// assert!(quadrant.is_other_aware());
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum JohariQuadrant {
    /// Known to self and others - explicitly shared context
    ///
    /// Characteristics:
    /// - Directly accessible via public APIs
    /// - Included in context injection by default
    /// - Highest retrieval priority for matching queries
    Open,

    /// Known to self only - private/hidden knowledge
    ///
    /// Characteristics:
    /// - Requires explicit permission to access
    /// - Not included in standard context injection
    /// - Used for sensitive or personal information
    Hidden,

    /// Known to others only - blind spots
    ///
    /// Characteristics:
    /// - Patterns observed by external systems
    /// - User may not be consciously aware
    /// - Includes inferred preferences and behaviors
    Blind,

    /// Unknown to both - unconscious patterns
    ///
    /// Characteristics:
    /// - Discovered through deep analysis
    /// - Latent associations and connections
    /// - Emerges from dream consolidation process
    Unknown,
}

impl JohariQuadrant {
    /// Check if this quadrant implies self-awareness
    pub fn is_self_aware(&self) -> bool {
        matches!(self, Self::Open | Self::Hidden)
    }

    /// Check if this quadrant implies awareness by others
    pub fn is_other_aware(&self) -> bool {
        matches!(self, Self::Open | Self::Blind)
    }

    /// Get the default retrieval weight for this quadrant
    /// Higher weights are prioritized in search results
    pub fn default_retrieval_weight(&self) -> f32 {
        match self {
            Self::Open => 1.0,
            Self::Hidden => 0.3,  // Lower priority unless specifically requested
            Self::Blind => 0.7,  // Useful for insights
            Self::Unknown => 0.5, // Moderate priority for discovery
        }
    }

    /// Check if this quadrant should be included in default context injection
    pub fn include_in_default_context(&self) -> bool {
        matches!(self, Self::Open | Self::Blind)
    }

    /// Get human-readable description of the quadrant
    pub fn description(&self) -> &'static str {
        match self {
            Self::Open => "Openly shared knowledge, accessible to all",
            Self::Hidden => "Private knowledge, known but not shared",
            Self::Blind => "External observations, blind to self",
            Self::Unknown => "Latent patterns, undiscovered knowledge",
        }
    }

    /// Get the column family name for RocksDB storage
    pub fn column_family(&self) -> &'static str {
        match self {
            Self::Open => "johari_open",
            Self::Hidden => "johari_hidden",
            Self::Blind => "johari_blind",
            Self::Unknown => "johari_unknown",
        }
    }

    /// Parse from string representation
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "open" => Some(Self::Open),
            "hidden" => Some(Self::Hidden),
            "blind" => Some(Self::Blind),
            "unknown" => Some(Self::Unknown),
            _ => None,
        }
    }

    /// Iterator over all quadrants
    pub fn all() -> impl Iterator<Item = Self> {
        [Self::Open, Self::Hidden, Self::Blind, Self::Unknown].into_iter()
    }
}

impl Default for JohariQuadrant {
    fn default() -> Self {
        Self::Unknown
    }
}

impl std::fmt::Display for JohariQuadrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open => write!(f, "open"),
            Self::Hidden => write!(f, "hidden"),
            Self::Blind => write!(f, "blind"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Johari transition rules for automatic quadrant updates
#[derive(Debug, Clone)]
pub struct JohariTransition {
    /// Source quadrant
    pub from: JohariQuadrant,
    /// Target quadrant
    pub to: JohariQuadrant,
    /// Transition trigger
    pub trigger: TransitionTrigger,
}

/// Triggers for Johari quadrant transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionTrigger {
    /// User explicitly shares hidden knowledge
    ExplicitShare,
    /// User acknowledges blind spot
    SelfRecognition,
    /// Pattern discovered through analysis
    PatternDiscovery,
    /// User requests privacy for open knowledge
    Privatize,
    /// External system identifies pattern
    ExternalObservation,
    /// Dream consolidation reveals pattern
    DreamConsolidation,
}

impl JohariTransition {
    /// Get valid transitions for a given quadrant
    pub fn valid_transitions(from: JohariQuadrant) -> Vec<JohariTransition> {
        match from {
            JohariQuadrant::Open => vec![
                JohariTransition {
                    from,
                    to: JohariQuadrant::Hidden,
                    trigger: TransitionTrigger::Privatize,
                },
            ],
            JohariQuadrant::Hidden => vec![
                JohariTransition {
                    from,
                    to: JohariQuadrant::Open,
                    trigger: TransitionTrigger::ExplicitShare,
                },
            ],
            JohariQuadrant::Blind => vec![
                JohariTransition {
                    from,
                    to: JohariQuadrant::Open,
                    trigger: TransitionTrigger::SelfRecognition,
                },
                JohariTransition {
                    from,
                    to: JohariQuadrant::Hidden,
                    trigger: TransitionTrigger::SelfRecognition,
                },
            ],
            JohariQuadrant::Unknown => vec![
                JohariTransition {
                    from,
                    to: JohariQuadrant::Open,
                    trigger: TransitionTrigger::PatternDiscovery,
                },
                JohariTransition {
                    from,
                    to: JohariQuadrant::Blind,
                    trigger: TransitionTrigger::ExternalObservation,
                },
                JohariTransition {
                    from,
                    to: JohariQuadrant::Hidden,
                    trigger: TransitionTrigger::DreamConsolidation,
                },
            ],
        }
    }
}
```

### 2.3 Node Metadata Structure

```rust
// crates/context-graph-core/src/metadata.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Extended metadata for memory nodes.
///
/// Provides additional context, categorization, and filtering capabilities
/// beyond the core MemoryNode fields.
///
/// # Example
///
/// ```rust
/// use context_graph_core::NodeMetadata;
///
/// let mut metadata = NodeMetadata::default();
/// metadata.source = Some("conversation_123".to_string());
/// metadata.tags = vec!["project".to_string(), "decision".to_string()];
/// metadata.set_custom("priority", "high");
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct NodeMetadata {
    /// Source identifier (conversation ID, file path, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Content language (ISO 639-1 code)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Content modality
    #[serde(default)]
    pub modality: Modality,

    /// Custom tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,

    /// UTL learning score at creation time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utl_score: Option<f32>,

    /// Whether this node has been consolidated to long-term memory
    #[serde(default)]
    pub consolidated: bool,

    /// Consolidation timestamp (if consolidated)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consolidated_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Version number for optimistic concurrency control
    #[serde(default)]
    pub version: u32,

    /// Soft deletion marker
    #[serde(default)]
    pub deleted: bool,

    /// Deletion timestamp (if soft-deleted)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deleted_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Parent node ID (for hierarchical relationships)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<uuid::Uuid>,

    /// Child node IDs (for hierarchical relationships)
    #[serde(default)]
    pub child_ids: Vec<uuid::Uuid>,

    /// Custom key-value attributes
    #[serde(default)]
    pub custom: HashMap<String, String>,
}

impl NodeMetadata {
    /// Create metadata with a specific source
    pub fn with_source(source: impl Into<String>) -> Self {
        Self {
            source: Some(source.into()),
            ..Default::default()
        }
    }

    /// Add a tag
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Remove a tag
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        if let Some(pos) = self.tags.iter().position(|t| t == tag) {
            self.tags.remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if a tag exists
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Set a custom attribute
    pub fn set_custom(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.custom.insert(key.into(), value.into());
    }

    /// Get a custom attribute
    pub fn get_custom(&self, key: &str) -> Option<&str> {
        self.custom.get(key).map(|s| s.as_str())
    }

    /// Remove a custom attribute
    pub fn remove_custom(&mut self, key: &str) -> Option<String> {
        self.custom.remove(key)
    }

    /// Mark as consolidated
    pub fn mark_consolidated(&mut self) {
        self.consolidated = true;
        self.consolidated_at = Some(chrono::Utc::now());
    }

    /// Mark as deleted (soft delete)
    pub fn mark_deleted(&mut self) {
        self.deleted = true;
        self.deleted_at = Some(chrono::Utc::now());
    }

    /// Restore from soft deletion
    pub fn restore(&mut self) {
        self.deleted = false;
        self.deleted_at = None;
    }

    /// Increment version for optimistic concurrency
    pub fn increment_version(&mut self) {
        self.version = self.version.saturating_add(1);
    }

    /// Calculate estimated serialized size in bytes
    pub fn estimated_size(&self) -> usize {
        let mut size = 0;
        size += self.source.as_ref().map(|s| s.len()).unwrap_or(0);
        size += self.language.as_ref().map(|s| s.len()).unwrap_or(0);
        size += self.tags.iter().map(|t| t.len()).sum::<usize>();
        size += self.custom.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>();
        size += self.child_ids.len() * 16; // UUID size
        size += 64; // Fixed overhead for other fields
        size
    }
}

/// Input modality classification for content type detection
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Plain text content
    #[default]
    Text,
    /// Source code
    Code,
    /// Image description or reference
    Image,
    /// Audio transcription or reference
    Audio,
    /// Structured data (JSON, YAML, etc.)
    Structured,
    /// Mixed content types
    Mixed,
}

impl Modality {
    /// Detect modality from content heuristics
    pub fn detect(content: &str) -> Self {
        // Check for code patterns
        if content.contains("fn ")
            || content.contains("def ")
            || content.contains("function ")
            || content.contains("class ")
            || content.contains("import ")
            || content.contains("pub ")
        {
            return Self::Code;
        }

        // Check for structured data
        if content.trim().starts_with('{') && content.trim().ends_with('}') {
            return Self::Structured;
        }
        if content.trim().starts_with('[') && content.trim().ends_with(']') {
            return Self::Structured;
        }
        if content.contains("---\n") && content.lines().any(|l| l.contains(": ")) {
            return Self::Structured;
        }

        // Check for image references
        if content.contains("[image:") || content.contains("![") {
            return Self::Image;
        }

        // Check for audio references
        if content.contains("[audio:") || content.contains("transcription:") {
            return Self::Audio;
        }

        Self::Text
    }

    /// Get file extension hints for this modality
    pub fn file_extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Text => &["txt", "md", "rst"],
            Self::Code => &["rs", "py", "js", "ts", "go", "java", "c", "cpp"],
            Self::Image => &["png", "jpg", "jpeg", "gif", "webp", "svg"],
            Self::Audio => &["mp3", "wav", "ogg", "m4a", "flac"],
            Self::Structured => &["json", "yaml", "yml", "toml", "xml"],
            Self::Mixed => &[],
        }
    }
}
```

### 2.4 Cognitive Pulse Structure

```rust
// crates/context-graph-core/src/pulse.rs

use serde::{Deserialize, Serialize};

/// Cognitive Pulse header included in all JSON-RPC responses.
///
/// Provides real-time cognitive state metrics to guide LLM behavior:
/// - **Entropy**: Information uncertainty [0.0, 1.0]
/// - **Coherence**: Knowledge consistency [0.0, 1.0]
/// - **Curiosity Score**: Exploration drive [0.0, 1.0]
/// - **Confidence**: System certainty [0.0, 1.0]
/// - **Emotional State**: Current emotional context
///
/// # JSON-RPC Integration
///
/// The pulse is included as a header in every tool response:
/// ```json
/// {
///   "result": {
///     "pulse": {
///       "entropy": 0.35,
///       "coherence": 0.82,
///       "curiosity_score": 0.65,
///       "confidence": 0.78,
///       "emotional_state": "curious",
///       "suggested_action": "explore"
///     },
///     "data": { ... }
///   }
/// }
/// ```
///
/// # Real-Time Streaming Support
///
/// For streaming responses, pulse updates can be sent as SSE events:
/// ```text
/// event: pulse
/// data: {"entropy": 0.35, "coherence": 0.82, ...}
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CognitivePulse {
    /// Entropy level [0.0, 1.0]
    /// Higher values indicate more uncertainty/disorder
    /// - 0.0-0.3: Low entropy, stable knowledge
    /// - 0.3-0.6: Moderate entropy, balanced state
    /// - 0.6-1.0: High entropy, consider consolidation
    pub entropy: f32,

    /// Coherence level [0.0, 1.0]
    /// Higher values indicate more consistent knowledge
    /// - 0.0-0.3: Low coherence, fragmented knowledge
    /// - 0.3-0.7: Moderate coherence, developing structure
    /// - 0.7-1.0: High coherence, well-organized
    pub coherence: f32,

    /// Curiosity score [0.0, 1.0]
    /// Measures drive to explore new information
    /// - 0.0-0.3: Low curiosity, consolidation focus
    /// - 0.3-0.7: Balanced exploration/exploitation
    /// - 0.7-1.0: High curiosity, exploration focus
    pub curiosity_score: f32,

    /// Confidence level [0.0, 1.0]
    /// System confidence in current state
    pub confidence: f32,

    /// Current emotional state
    pub emotional_state: EmotionalState,

    /// Suggested action based on cognitive state
    pub suggested_action: SuggestedAction,

    /// Timestamp of this pulse
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl CognitivePulse {
    /// Create a new cognitive pulse with current timestamp
    pub fn new(
        entropy: f32,
        coherence: f32,
        curiosity_score: f32,
        confidence: f32,
        emotional_state: EmotionalState,
    ) -> Self {
        let mut pulse = Self {
            entropy: entropy.clamp(0.0, 1.0),
            coherence: coherence.clamp(0.0, 1.0),
            curiosity_score: curiosity_score.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            emotional_state,
            suggested_action: SuggestedAction::Continue,
            timestamp: chrono::Utc::now(),
        };
        pulse.suggested_action = pulse.compute_suggested_action();
        pulse
    }

    /// Compute suggested action based on current metrics
    fn compute_suggested_action(&self) -> SuggestedAction {
        // High entropy, low coherence -> stabilize
        if self.entropy > 0.7 && self.coherence < 0.4 {
            return SuggestedAction::Stabilize;
        }

        // Low entropy, high coherence -> ready for new input
        if self.entropy < 0.3 && self.coherence > 0.7 {
            return SuggestedAction::Ready;
        }

        // High curiosity -> explore
        if self.curiosity_score > 0.7 {
            return SuggestedAction::Explore;
        }

        // Low entropy, moderate coherence -> consolidate
        if self.entropy < 0.4 && self.coherence > 0.5 && self.coherence < 0.8 {
            return SuggestedAction::Consolidate;
        }

        // High entropy, moderate coherence -> prune
        if self.entropy > 0.6 && self.coherence > 0.4 && self.coherence < 0.7 {
            return SuggestedAction::Prune;
        }

        // Low confidence -> review
        if self.confidence < 0.4 {
            return SuggestedAction::Review;
        }

        SuggestedAction::Continue
    }

    /// Update pulse with new observations
    pub fn update(&mut self, delta_entropy: f32, delta_coherence: f32) {
        self.entropy = (self.entropy + delta_entropy).clamp(0.0, 1.0);
        self.coherence = (self.coherence + delta_coherence).clamp(0.0, 1.0);
        self.timestamp = chrono::Utc::now();
        self.suggested_action = self.compute_suggested_action();
    }

    /// Blend with another pulse (for averaging)
    pub fn blend(&self, other: &Self, weight: f32) -> Self {
        let w = weight.clamp(0.0, 1.0);
        let iw = 1.0 - w;

        Self::new(
            self.entropy * iw + other.entropy * w,
            self.coherence * iw + other.coherence * w,
            self.curiosity_score * iw + other.curiosity_score * w,
            self.confidence * iw + other.confidence * w,
            if w > 0.5 { other.emotional_state } else { self.emotional_state },
        )
    }

    /// Serialize to JSON for response header
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "entropy": self.entropy,
            "coherence": self.coherence,
            "curiosity_score": self.curiosity_score,
            "confidence": self.confidence,
            "emotional_state": self.emotional_state,
            "suggested_action": self.suggested_action,
            "timestamp": self.timestamp.to_rfc3339()
        })
    }
}

impl Default for CognitivePulse {
    fn default() -> Self {
        Self::new(0.5, 0.5, 0.5, 0.5, EmotionalState::Neutral)
    }
}

/// Emotional state classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "lowercase")]
pub enum EmotionalState {
    /// Baseline neutral state
    #[default]
    Neutral,
    /// Active exploration mode
    Curious,
    /// Deep concentration
    Focused,
    /// Elevated stress/urgency
    Stressed,
    /// Reduced capacity
    Fatigued,
    /// Positive engagement
    Engaged,
    /// Uncertainty/confusion
    Confused,
}

impl EmotionalState {
    /// Get the emotional weight modifier for UTL calculations
    pub fn weight_modifier(&self) -> f32 {
        match self {
            Self::Neutral => 1.0,
            Self::Curious => 1.2,   // Boost learning
            Self::Focused => 1.3,   // Enhanced consolidation
            Self::Stressed => 0.8,  // Reduced capacity
            Self::Fatigued => 0.6,  // Significantly reduced
            Self::Engaged => 1.15,  // Moderate boost
            Self::Confused => 0.9,  // Slightly reduced
        }
    }
}

/// Suggested actions based on cognitive state
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum SuggestedAction {
    /// System ready for new input
    Ready,
    /// Continue current operation
    #[default]
    Continue,
    /// Explore new areas
    Explore,
    /// Consolidate existing knowledge
    Consolidate,
    /// Prune low-value nodes
    Prune,
    /// Stabilize high entropy
    Stabilize,
    /// Review recent additions
    Review,
}

impl SuggestedAction {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Ready => "System stable, ready for new input",
            Self::Continue => "Continue current operation normally",
            Self::Explore => "Consider exploring new knowledge areas",
            Self::Consolidate => "Focus on consolidating existing knowledge",
            Self::Prune => "Consider pruning low-value memories",
            Self::Stabilize => "High entropy detected, stabilization recommended",
            Self::Review => "Review and verify recent additions",
        }
    }
}
```

### 2.5 Marblestone Features (GraphEdge with Neuromodulation)

```rust
// crates/context-graph-core/src/marblestone.rs

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Domain classification for context-aware neurotransmitter weighting (Marblestone)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "lowercase")]
pub enum Domain {
    /// Source code and technical documentation
    Code,
    /// Legal documents and contracts
    Legal,
    /// Medical and healthcare content
    Medical,
    /// Creative writing and artistic content
    Creative,
    /// Academic and research materials
    Research,
    /// General purpose (default)
    #[default]
    General,
}

/// Neurotransmitter-inspired edge weights for domain-specific modulation (Marblestone)
/// Based on biological neurotransmitter systems for different cognitive domains
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct NeurotransmitterWeights {
    /// Excitatory weight (Glutamate-like): promotes activation/retrieval
    /// Range: [0.0, 1.0]
    pub excitatory: f32,
    /// Inhibitory weight (GABA-like): suppresses competing activations
    /// Range: [0.0, 1.0]
    pub inhibitory: f32,
    /// Modulatory weight (Dopamine-like): controls learning rate/plasticity
    /// Range: [0.0, 1.0]
    pub modulatory: f32,
}

impl Default for NeurotransmitterWeights {
    fn default() -> Self {
        Self {
            excitatory: 0.5,
            inhibitory: 0.2,
            modulatory: 0.3,
        }
    }
}

impl NeurotransmitterWeights {
    /// Create domain-specific neurotransmitter profiles
    pub fn for_domain(domain: Domain) -> Self {
        match domain {
            Domain::Code => Self {
                excitatory: 0.6,   // Moderate activation for precision
                inhibitory: 0.3,  // Higher inhibition for focus
                modulatory: 0.4,  // Moderate plasticity
            },
            Domain::Legal => Self {
                excitatory: 0.4,   // Lower activation (conservative)
                inhibitory: 0.4,   // High inhibition (careful reasoning)
                modulatory: 0.2,   // Low plasticity (stable interpretations)
            },
            Domain::Medical => Self {
                excitatory: 0.5,
                inhibitory: 0.35,
                modulatory: 0.3,
            },
            Domain::Creative => Self {
                excitatory: 0.8,   // High activation (exploration)
                inhibitory: 0.1,   // Low inhibition (free association)
                modulatory: 0.6,   // High plasticity (novelty seeking)
            },
            Domain::Research => Self {
                excitatory: 0.65,
                inhibitory: 0.25,
                modulatory: 0.5,
            },
            Domain::General => Self::default(),
        }
    }

    /// Compute effective edge weight with neurotransmitter modulation
    pub fn compute_effective_weight(&self, base_weight: f32) -> f32 {
        let excitation = base_weight * self.excitatory;
        let inhibition = base_weight * self.inhibitory;
        let modulation = 1.0 + (self.modulatory - 0.5) * 0.4; // +/- 20% adjustment
        ((excitation - inhibition) * modulation).clamp(0.0, 1.0)
    }

    /// Validate neurotransmitter weights are within bounds
    pub fn validate(&self) -> Result<(), ValidationError> {
        for (name, value) in [
            ("excitatory", self.excitatory),
            ("inhibitory", self.inhibitory),
            ("modulatory", self.modulatory),
        ] {
            if value < 0.0 || value > 1.0 {
                return Err(ValidationError::OutOfBounds {
                    field: name.into(),
                    value,
                    min: 0.0,
                    max: 1.0,
                });
            }
        }
        Ok(())
    }
}

/// Edge type classification for graph relationships
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum EdgeType {
    /// Semantic similarity relationship
    Semantic,
    /// Temporal sequence relationship
    Temporal,
    /// Causal relationship
    Causal,
    /// Hierarchical parent-child relationship
    Hierarchical,
}

/// Graph edge with Marblestone neurotransmitter modulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Unique edge identifier
    pub id: Uuid,
    /// Source node ID
    pub source_id: Uuid,
    /// Target node ID
    pub target_id: Uuid,
    /// Edge type classification
    pub edge_type: EdgeType,
    /// Base edge weight [0.0, 1.0]
    pub weight: f32,
    /// Confidence score [0.0, 1.0]
    pub confidence: f32,
    /// Domain classification for neurotransmitter profiling
    pub domain: Domain,
    /// Neurotransmitter weights for domain-aware modulation (Marblestone)
    pub neurotransmitter_weights: NeurotransmitterWeights,
    /// Whether this edge is an amortized shortcut from sleep replay (Marblestone)
    pub is_amortized_shortcut: bool,
    /// Steering reward for traversal feedback (Marblestone) [-1.0, 1.0]
    pub steering_reward: f32,
    /// Number of times this edge has been traversed
    pub traversal_count: u32,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last traversal timestamp
    pub last_traversed_at: Option<DateTime<Utc>>,
}

impl GraphEdge {
    /// Create a new graph edge with default Marblestone settings
    pub fn new(
        source_id: Uuid,
        target_id: Uuid,
        edge_type: EdgeType,
        weight: f32,
        domain: Domain,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            source_id,
            target_id,
            edge_type,
            weight,
            confidence: 0.5,
            domain,
            neurotransmitter_weights: NeurotransmitterWeights::for_domain(domain),
            is_amortized_shortcut: false,
            steering_reward: 0.0,
            traversal_count: 0,
            created_at: Utc::now(),
            last_traversed_at: None,
        }
    }

    /// Get modulated edge weight considering neurotransmitters
    pub fn get_modulated_weight(&self) -> f32 {
        let base = self.weight * self.confidence;
        let nt_factor = self.neurotransmitter_weights.compute_effective_weight(base);
        let reward_factor = 1.0 + (self.steering_reward * 0.2); // +/- 20% from steering
        (nt_factor * reward_factor).clamp(0.0, 1.0)
    }

    /// Apply steering feedback from traversal (Marblestone)
    pub fn apply_steering_reward(&mut self, reward: f32) {
        self.steering_reward = reward.clamp(-1.0, 1.0);
    }

    /// Decay steering reward over time
    pub fn decay_steering(&mut self, decay_factor: f32) {
        self.steering_reward *= decay_factor;
    }

    /// Record edge traversal
    pub fn record_traversal(&mut self) {
        self.traversal_count = self.traversal_count.saturating_add(1);
        self.last_traversed_at = Some(Utc::now());
    }

    /// Check if this is a reliable amortized shortcut
    pub fn is_reliable_shortcut(&self) -> bool {
        self.is_amortized_shortcut
            && self.traversal_count >= 3
            && self.steering_reward > 0.3
            && self.confidence >= 0.7
    }

    /// Mark as amortized shortcut (created during sleep replay)
    pub fn mark_as_shortcut(&mut self) {
        self.is_amortized_shortcut = true;
    }
}
```

---

## 3. Memex Storage (RocksDB)

### 3.1 Column Family Definitions

```rust
// crates/context-graph-storage/src/column_families.rs

use rocksdb::{Options, ColumnFamilyDescriptor};

/// Column family names for Memex storage
pub mod cf_names {
    /// Primary node storage (id -> serialized MemoryNode)
    pub const NODES: &str = "nodes";

    /// Graph edges (edge_id -> serialized GraphEdge)
    pub const EDGES: &str = "edges";

    /// Embedding vectors (node_id -> raw f32 bytes)
    pub const EMBEDDINGS: &str = "embeddings";

    /// Node metadata index (various secondary indexes)
    pub const METADATA: &str = "metadata";

    /// Johari Open quadrant index
    pub const JOHARI_OPEN: &str = "johari_open";

    /// Johari Hidden quadrant index
    pub const JOHARI_HIDDEN: &str = "johari_hidden";

    /// Johari Blind quadrant index
    pub const JOHARI_BLIND: &str = "johari_blind";

    /// Johari Unknown quadrant index
    pub const JOHARI_UNKNOWN: &str = "johari_unknown";

    /// Temporal index (timestamp -> node_ids)
    pub const TEMPORAL: &str = "temporal";

    /// Tag index (tag -> node_ids)
    pub const TAGS: &str = "tags";

    /// Source index (source -> node_ids)
    pub const SOURCES: &str = "sources";

    /// System metadata and configuration
    pub const SYSTEM: &str = "system";
}

/// Get all column family descriptors with optimized options
pub fn get_column_family_descriptors() -> Vec<ColumnFamilyDescriptor> {
    vec![
        // Nodes CF - optimized for random reads
        ColumnFamilyDescriptor::new(cf_names::NODES, nodes_options()),

        // Edges CF - optimized for range scans
        ColumnFamilyDescriptor::new(cf_names::EDGES, edges_options()),

        // Embeddings CF - optimized for large sequential reads
        ColumnFamilyDescriptor::new(cf_names::EMBEDDINGS, embeddings_options()),

        // Metadata CF - balanced read/write
        ColumnFamilyDescriptor::new(cf_names::METADATA, default_options()),

        // Johari CFs - optimized for range scans
        ColumnFamilyDescriptor::new(cf_names::JOHARI_OPEN, johari_options()),
        ColumnFamilyDescriptor::new(cf_names::JOHARI_HIDDEN, johari_options()),
        ColumnFamilyDescriptor::new(cf_names::JOHARI_BLIND, johari_options()),
        ColumnFamilyDescriptor::new(cf_names::JOHARI_UNKNOWN, johari_options()),

        // Index CFs
        ColumnFamilyDescriptor::new(cf_names::TEMPORAL, index_options()),
        ColumnFamilyDescriptor::new(cf_names::TAGS, index_options()),
        ColumnFamilyDescriptor::new(cf_names::SOURCES, index_options()),

        // System CF
        ColumnFamilyDescriptor::new(cf_names::SYSTEM, system_options()),
    ]
}

/// Options for nodes column family
fn nodes_options() -> Options {
    let mut opts = Options::default();
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB
    opts.set_max_write_buffer_number(3);
    opts.set_target_file_size_base(64 * 1024 * 1024); // 64MB
    opts.set_level_zero_file_num_compaction_trigger(4);
    opts.set_bloom_locality(1);
    opts.optimize_for_point_lookup(256 * 1024 * 1024); // 256MB block cache
    opts
}

/// Options for edges column family
fn edges_options() -> Options {
    let mut opts = Options::default();
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
    opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16)); // UUID prefix
    opts.optimize_level_style_compaction(256 * 1024 * 1024);
    opts
}

/// Options for embeddings column family
fn embeddings_options() -> Options {
    let mut opts = Options::default();
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_write_buffer_size(128 * 1024 * 1024); // 128MB - embeddings are large
    opts.set_max_write_buffer_number(2);
    // Embeddings are mostly append-only, optimize for sequential reads
    opts.set_compaction_style(rocksdb::DBCompactionStyle::Universal);
    opts
}

/// Options for Johari index column families
fn johari_options() -> Options {
    let mut opts = Options::default();
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
    opts.optimize_level_style_compaction(64 * 1024 * 1024);
    opts
}

/// Options for secondary index column families
fn index_options() -> Options {
    let mut opts = Options::default();
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
    opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(32));
    opts
}

/// Options for system column family
fn system_options() -> Options {
    let mut opts = Options::default();
    opts.set_compression_type(rocksdb::DBCompressionType::None); // Small data, no compression
    opts.set_write_buffer_size(4 * 1024 * 1024); // 4MB
    opts
}

/// Default options for general-purpose column families
fn default_options() -> Options {
    let mut opts = Options::default();
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
    opts
}
```

### 3.2 RocksDB Backend Implementation

```rust
// crates/context-graph-storage/src/rocksdb_backend.rs

use crate::column_families::{cf_names, get_column_family_descriptors};
use crate::serialization::{serialize_node, deserialize_node, serialize_embedding, deserialize_embedding};
use context_graph_core::{MemoryNode, NodeId, JohariQuadrant};
use rocksdb::{DB, Options, WriteBatch, IteratorMode, ReadOptions};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

/// RocksDB-backed Memex storage implementation
pub struct RocksDbMemex {
    db: Arc<RwLock<DB>>,
    path: String,
}

impl RocksDbMemex {
    /// Open or create a RocksDB database at the given path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        db_opts.set_max_open_files(1000);
        db_opts.set_keep_log_file_num(5);
        db_opts.set_max_total_wal_size(256 * 1024 * 1024); // 256MB
        db_opts.set_wal_dir(format!("{}/wal", path_str));

        // Block cache: 256MB shared across all CFs
        let cache = rocksdb::Cache::new_lru_cache(256 * 1024 * 1024);
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_block_cache(&cache);
        block_opts.set_bloom_filter(10.0, true);

        let cf_descriptors = get_column_family_descriptors();
        let db = DB::open_cf_descriptors(&db_opts, &path_str, cf_descriptors)
            .map_err(|e| StorageError::OpenFailed(e.to_string()))?;

        Ok(Self {
            db: Arc::new(RwLock::new(db)),
            path: path_str,
        })
    }

    /// Store a memory node
    pub async fn store_node(&self, node: &MemoryNode) -> Result<(), StorageError> {
        node.validate().map_err(|e| StorageError::ValidationFailed(e.to_string()))?;

        let db = self.db.write().await;
        let mut batch = WriteBatch::default();

        // Serialize node
        let node_bytes = serialize_node(node)?;
        let key = node.id.as_bytes();

        // Write to nodes CF
        let nodes_cf = db.cf_handle(cf_names::NODES)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(cf_names::NODES.into()))?;
        batch.put_cf(nodes_cf, key, &node_bytes);

        // Write embedding to embeddings CF
        let embeddings_cf = db.cf_handle(cf_names::EMBEDDINGS)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(cf_names::EMBEDDINGS.into()))?;
        let embedding_bytes = serialize_embedding(&node.embedding)?;
        batch.put_cf(embeddings_cf, key, &embedding_bytes);

        // Write to Johari index
        let johari_cf_name = node.quadrant.column_family();
        let johari_cf = db.cf_handle(johari_cf_name)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(johari_cf_name.into()))?;
        batch.put_cf(johari_cf, key, &[]); // Key-only index

        // Write to temporal index
        let temporal_cf = db.cf_handle(cf_names::TEMPORAL)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(cf_names::TEMPORAL.into()))?;
        let timestamp_key = format!(
            "{}:{}",
            node.created_at.timestamp_millis(),
            node.id
        );
        batch.put_cf(temporal_cf, timestamp_key.as_bytes(), key);

        // Write to tag indexes
        let tags_cf = db.cf_handle(cf_names::TAGS)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(cf_names::TAGS.into()))?;
        for tag in &node.metadata.tags {
            let tag_key = format!("{}:{}", tag, node.id);
            batch.put_cf(tags_cf, tag_key.as_bytes(), key);
        }

        // Write to source index if present
        if let Some(source) = &node.metadata.source {
            let sources_cf = db.cf_handle(cf_names::SOURCES)
                .ok_or_else(|| StorageError::ColumnFamilyNotFound(cf_names::SOURCES.into()))?;
            let source_key = format!("{}:{}", source, node.id);
            batch.put_cf(sources_cf, source_key.as_bytes(), key);
        }

        // Execute batch write
        db.write(batch)
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(())
    }

    /// Retrieve a node by ID
    pub async fn get_node(&self, id: NodeId) -> Result<Option<MemoryNode>, StorageError> {
        let db = self.db.read().await;

        let nodes_cf = db.cf_handle(cf_names::NODES)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(cf_names::NODES.into()))?;

        let key = id.as_bytes();

        match db.get_cf(nodes_cf, key) {
            Ok(Some(bytes)) => {
                let node = deserialize_node(&bytes)?;
                Ok(Some(node))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(StorageError::ReadFailed(e.to_string())),
        }
    }

    /// Get embedding vector by node ID
    pub async fn get_embedding(&self, id: NodeId) -> Result<Option<Vec<f32>>, StorageError> {
        let db = self.db.read().await;

        let embeddings_cf = db.cf_handle(cf_names::EMBEDDINGS)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(cf_names::EMBEDDINGS.into()))?;

        let key = id.as_bytes();

        match db.get_cf(embeddings_cf, key) {
            Ok(Some(bytes)) => {
                let embedding = deserialize_embedding(&bytes)?;
                Ok(Some(embedding))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(StorageError::ReadFailed(e.to_string())),
        }
    }

    /// Delete a node (soft or hard)
    pub async fn delete_node(&self, id: NodeId, soft: bool) -> Result<bool, StorageError> {
        if soft {
            // Soft delete: update metadata
            if let Some(mut node) = self.get_node(id).await? {
                node.metadata.mark_deleted();
                self.store_node(&node).await?;
                return Ok(true);
            }
            return Ok(false);
        }

        // Hard delete
        let db = self.db.write().await;
        let mut batch = WriteBatch::default();
        let key = id.as_bytes();

        // Get node first to clean up indexes
        let nodes_cf = db.cf_handle(cf_names::NODES)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(cf_names::NODES.into()))?;

        let node_bytes = match db.get_cf(nodes_cf, key) {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return Ok(false),
            Err(e) => return Err(StorageError::ReadFailed(e.to_string())),
        };

        let node = deserialize_node(&node_bytes)?;

        // Delete from nodes CF
        batch.delete_cf(nodes_cf, key);

        // Delete from embeddings CF
        let embeddings_cf = db.cf_handle(cf_names::EMBEDDINGS)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(cf_names::EMBEDDINGS.into()))?;
        batch.delete_cf(embeddings_cf, key);

        // Delete from Johari index
        let johari_cf_name = node.quadrant.column_family();
        if let Some(johari_cf) = db.cf_handle(johari_cf_name) {
            batch.delete_cf(johari_cf, key);
        }

        // Delete from tag indexes
        if let Some(tags_cf) = db.cf_handle(cf_names::TAGS) {
            for tag in &node.metadata.tags {
                let tag_key = format!("{}:{}", tag, node.id);
                batch.delete_cf(tags_cf, tag_key.as_bytes());
            }
        }

        // Execute batch delete
        db.write(batch)
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(true)
    }

    /// Query nodes by Johari quadrant
    pub async fn query_by_quadrant(
        &self,
        quadrant: JohariQuadrant,
        limit: usize,
    ) -> Result<Vec<NodeId>, StorageError> {
        let db = self.db.read().await;

        let cf_name = quadrant.column_family();
        let cf = db.cf_handle(cf_name)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(cf_name.into()))?;

        let mut ids = Vec::with_capacity(limit);
        let iter = db.iterator_cf(cf, IteratorMode::Start);

        for item in iter.take(limit) {
            match item {
                Ok((key, _)) => {
                    if key.len() == 16 {
                        let id = NodeId::from_slice(&key)
                            .map_err(|e| StorageError::DeserializationFailed(e.to_string()))?;
                        ids.push(id);
                    }
                }
                Err(e) => return Err(StorageError::ReadFailed(e.to_string())),
            }
        }

        Ok(ids)
    }

    /// Get total node count
    pub async fn count(&self) -> Result<usize, StorageError> {
        let db = self.db.read().await;

        let nodes_cf = db.cf_handle(cf_names::NODES)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(cf_names::NODES.into()))?;

        let iter = db.iterator_cf(nodes_cf, IteratorMode::Start);
        Ok(iter.count())
    }

    /// Compact all column families
    pub async fn compact(&self) -> Result<(), StorageError> {
        let db = self.db.write().await;

        for cf_name in [
            cf_names::NODES,
            cf_names::EDGES,
            cf_names::EMBEDDINGS,
            cf_names::METADATA,
            cf_names::JOHARI_OPEN,
            cf_names::JOHARI_HIDDEN,
            cf_names::JOHARI_BLIND,
            cf_names::JOHARI_UNKNOWN,
            cf_names::TEMPORAL,
            cf_names::TAGS,
            cf_names::SOURCES,
        ] {
            if let Some(cf) = db.cf_handle(cf_name) {
                db.compact_range_cf(cf, None::<&[u8]>, None::<&[u8]>);
            }
        }

        Ok(())
    }

    /// Flush WAL to disk
    pub async fn flush(&self) -> Result<(), StorageError> {
        let db = self.db.write().await;
        db.flush().map_err(|e| StorageError::FlushFailed(e.to_string()))?;
        Ok(())
    }
}

/// Storage errors
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Failed to open database: {0}")]
    OpenFailed(String),

    #[error("Column family not found: {0}")]
    ColumnFamilyNotFound(String),

    #[error("Write failed: {0}")]
    WriteFailed(String),

    #[error("Read failed: {0}")]
    ReadFailed(String),

    #[error("Serialization failed: {0}")]
    SerializationFailed(String),

    #[error("Deserialization failed: {0}")]
    DeserializationFailed(String),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Flush failed: {0}")]
    FlushFailed(String),
}
```

### 3.3 Serialization (Bincode)

```rust
// crates/context-graph-storage/src/serialization.rs

use context_graph_core::MemoryNode;
use crate::rocksdb_backend::StorageError;

/// Serialize a MemoryNode to bytes using bincode
pub fn serialize_node(node: &MemoryNode) -> Result<Vec<u8>, StorageError> {
    bincode::serialize(node)
        .map_err(|e| StorageError::SerializationFailed(e.to_string()))
}

/// Deserialize a MemoryNode from bytes
pub fn deserialize_node(bytes: &[u8]) -> Result<MemoryNode, StorageError> {
    bincode::deserialize(bytes)
        .map_err(|e| StorageError::DeserializationFailed(e.to_string()))
}

/// Serialize an embedding vector to raw f32 bytes
/// Uses little-endian byte order for efficiency
pub fn serialize_embedding(embedding: &[f32]) -> Result<Vec<u8>, StorageError> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

/// Deserialize an embedding vector from raw bytes
pub fn deserialize_embedding(bytes: &[u8]) -> Result<Vec<f32>, StorageError> {
    if bytes.len() % 4 != 0 {
        return Err(StorageError::DeserializationFailed(
            "Invalid embedding byte length".into()
        ));
    }

    let mut embedding = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        embedding.push(value);
    }

    Ok(embedding)
}

/// Serialize a UUID to bytes
pub fn serialize_uuid(id: &uuid::Uuid) -> [u8; 16] {
    *id.as_bytes()
}

/// Deserialize a UUID from bytes
pub fn deserialize_uuid(bytes: &[u8]) -> Result<uuid::Uuid, StorageError> {
    uuid::Uuid::from_slice(bytes)
        .map_err(|e| StorageError::DeserializationFailed(e.to_string()))
}

/// Serialize a timestamp to bytes (milliseconds since epoch)
pub fn serialize_timestamp(ts: &chrono::DateTime<chrono::Utc>) -> [u8; 8] {
    ts.timestamp_millis().to_be_bytes()
}

/// Deserialize a timestamp from bytes
pub fn deserialize_timestamp(bytes: &[u8]) -> Result<chrono::DateTime<chrono::Utc>, StorageError> {
    if bytes.len() != 8 {
        return Err(StorageError::DeserializationFailed(
            "Invalid timestamp byte length".into()
        ));
    }

    let millis = i64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ]);

    chrono::DateTime::from_timestamp_millis(millis)
        .ok_or_else(|| StorageError::DeserializationFailed("Invalid timestamp".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_node_roundtrip() {
        let node = MemoryNode::new("test content".into(), vec![0.1; 1536]);
        let bytes = serialize_node(&node).unwrap();
        let decoded = deserialize_node(&bytes).unwrap();
        assert_eq!(node.id, decoded.id);
        assert_eq!(node.content, decoded.content);
    }

    #[test]
    fn test_embedding_roundtrip() {
        let embedding: Vec<f32> = (0..1536).map(|i| i as f32 / 1536.0).collect();
        let bytes = serialize_embedding(&embedding).unwrap();
        let decoded = deserialize_embedding(&bytes).unwrap();
        assert_eq!(embedding, decoded);
    }

    #[test]
    fn test_timestamp_roundtrip() {
        let ts = Utc::now();
        let bytes = serialize_timestamp(&ts);
        let decoded = deserialize_timestamp(&bytes).unwrap();
        assert_eq!(ts.timestamp_millis(), decoded.timestamp_millis());
    }
}
```

---

## 4. CRUD Operation Algorithms

### 4.1 Insert Algorithm

```rust
// crates/context-graph-storage/src/operations.rs

use crate::rocksdb_backend::{RocksDbMemex, StorageError};
use context_graph_core::{MemoryNode, NodeId, JohariQuadrant};

impl RocksDbMemex {
    /// Insert a new memory node with full indexing
    ///
    /// # Algorithm
    /// 1. Validate node structure and constraints
    /// 2. Generate deterministic embedding if not provided
    /// 3. Create atomic batch write:
    ///    a. Write node to nodes CF
    ///    b. Write embedding to embeddings CF
    ///    c. Update Johari quadrant index
    ///    d. Update temporal index
    ///    e. Update tag indexes
    ///    f. Update source index
    /// 4. Execute batch atomically
    /// 5. Return node ID
    ///
    /// # Performance Target: <1ms
    ///
    /// # Example
    /// ```rust
    /// let node = MemoryNode::new("content".into(), embedding);
    /// let id = memex.insert(node).await?;
    /// ```
    pub async fn insert(&self, mut node: MemoryNode) -> Result<NodeId, StorageError> {
        // 1. Validate
        node.validate()
            .map_err(|e| StorageError::ValidationFailed(e.to_string()))?;

        // 2. Ensure timestamps are set
        let now = chrono::Utc::now();
        if node.created_at > now {
            node.created_at = now;
        }
        node.accessed_at = now;

        // 3. Store node (batch write handled internally)
        self.store_node(&node).await?;

        // 4. Return ID
        Ok(node.id)
    }

    /// Batch insert multiple nodes
    ///
    /// # Performance Target: <10ms for 100 nodes
    pub async fn batch_insert(&self, nodes: Vec<MemoryNode>) -> Result<Vec<NodeId>, StorageError> {
        let mut ids = Vec::with_capacity(nodes.len());

        // Validate all nodes first
        for node in &nodes {
            node.validate()
                .map_err(|e| StorageError::ValidationFailed(e.to_string()))?;
        }

        // Insert in batch
        for node in nodes {
            let id = self.insert(node).await?;
            ids.push(id);
        }

        Ok(ids)
    }
}
```

### 4.2 Retrieve Algorithm

```rust
impl RocksDbMemex {
    /// Retrieve a node by ID with access tracking
    ///
    /// # Algorithm
    /// 1. Query nodes CF by ID
    /// 2. If found:
    ///    a. Deserialize node
    ///    b. Update access count and timestamp
    ///    c. Write updated node back
    /// 3. Return node or None
    ///
    /// # Performance Target: <500us
    pub async fn retrieve(&self, id: NodeId) -> Result<Option<MemoryNode>, StorageError> {
        let mut node = match self.get_node(id).await? {
            Some(n) => n,
            None => return Ok(None),
        };

        // Update access tracking
        node.record_access();

        // Write back updated access info (async, fire-and-forget)
        let self_clone = self.clone();
        tokio::spawn(async move {
            let _ = self_clone.store_node(&node).await;
        });

        Ok(Some(node))
    }

    /// Retrieve multiple nodes by IDs
    ///
    /// # Performance: O(n) where n = number of IDs
    pub async fn retrieve_many(&self, ids: &[NodeId]) -> Result<Vec<MemoryNode>, StorageError> {
        let mut nodes = Vec::with_capacity(ids.len());

        for &id in ids {
            if let Some(node) = self.get_node(id).await? {
                nodes.push(node);
            }
        }

        Ok(nodes)
    }

    /// Retrieve node without updating access tracking
    /// Used for internal operations and indexing
    pub async fn peek(&self, id: NodeId) -> Result<Option<MemoryNode>, StorageError> {
        self.get_node(id).await
    }
}
```

### 4.3 Update Algorithm

```rust
impl RocksDbMemex {
    /// Update an existing node
    ///
    /// # Algorithm
    /// 1. Retrieve existing node
    /// 2. Validate update constraints
    /// 3. Check version for optimistic concurrency
    /// 4. Update indexes if quadrant/tags changed
    /// 5. Write updated node
    ///
    /// # Concurrency: Optimistic locking via version field
    pub async fn update(&self, node: MemoryNode) -> Result<bool, StorageError> {
        // 1. Get existing node
        let existing = match self.get_node(node.id).await? {
            Some(n) => n,
            None => return Ok(false),
        };

        // 2. Validate
        node.validate()
            .map_err(|e| StorageError::ValidationFailed(e.to_string()))?;

        // 3. Check version for optimistic concurrency
        if node.metadata.version != existing.metadata.version {
            return Err(StorageError::ValidationFailed(
                "Version mismatch - concurrent modification detected".into()
            ));
        }

        // 4. Handle Johari quadrant change
        if node.quadrant != existing.quadrant {
            self.update_johari_index(node.id, existing.quadrant, node.quadrant).await?;
        }

        // 5. Handle tag changes
        let old_tags: std::collections::HashSet<_> = existing.metadata.tags.iter().collect();
        let new_tags: std::collections::HashSet<_> = node.metadata.tags.iter().collect();

        // Remove old tags
        for tag in old_tags.difference(&new_tags) {
            self.remove_tag_index(node.id, tag).await?;
        }

        // Add new tags
        for tag in new_tags.difference(&old_tags) {
            self.add_tag_index(node.id, tag).await?;
        }

        // 6. Increment version and store
        let mut updated_node = node;
        updated_node.metadata.increment_version();
        self.store_node(&updated_node).await?;

        Ok(true)
    }

    /// Update Johari quadrant index
    async fn update_johari_index(
        &self,
        id: NodeId,
        old_quadrant: JohariQuadrant,
        new_quadrant: JohariQuadrant,
    ) -> Result<(), StorageError> {
        let db = self.db.write().await;
        let mut batch = rocksdb::WriteBatch::default();
        let key = id.as_bytes();

        // Remove from old quadrant
        if let Some(old_cf) = db.cf_handle(old_quadrant.column_family()) {
            batch.delete_cf(old_cf, key);
        }

        // Add to new quadrant
        if let Some(new_cf) = db.cf_handle(new_quadrant.column_family()) {
            batch.put_cf(new_cf, key, &[]);
        }

        db.write(batch)
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(())
    }

    /// Add tag index entry
    async fn add_tag_index(&self, id: NodeId, tag: &str) -> Result<(), StorageError> {
        let db = self.db.write().await;
        let tags_cf = db.cf_handle(crate::column_families::cf_names::TAGS)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound("tags".into()))?;

        let tag_key = format!("{}:{}", tag, id);
        db.put_cf(tags_cf, tag_key.as_bytes(), id.as_bytes())
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(())
    }

    /// Remove tag index entry
    async fn remove_tag_index(&self, id: NodeId, tag: &str) -> Result<(), StorageError> {
        let db = self.db.write().await;
        let tags_cf = db.cf_handle(crate::column_families::cf_names::TAGS)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound("tags".into()))?;

        let tag_key = format!("{}:{}", tag, id);
        db.delete_cf(tags_cf, tag_key.as_bytes())
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(())
    }
}
```

### 4.4 Delete Algorithm

```rust
impl RocksDbMemex {
    /// Delete a node (soft or hard)
    ///
    /// # Soft Delete Algorithm
    /// 1. Retrieve node
    /// 2. Mark as deleted with timestamp
    /// 3. Keep in all indexes (for audit)
    ///
    /// # Hard Delete Algorithm
    /// 1. Retrieve node for index cleanup
    /// 2. Create atomic batch:
    ///    a. Delete from nodes CF
    ///    b. Delete from embeddings CF
    ///    c. Delete from Johari index
    ///    d. Delete from tag indexes
    ///    e. Delete from source index
    ///    f. Delete from temporal index
    /// 3. Execute batch atomically
    pub async fn delete(&self, id: NodeId, soft: bool) -> Result<bool, StorageError> {
        self.delete_node(id, soft).await
    }

    /// Purge soft-deleted nodes older than retention period
    ///
    /// # Algorithm
    /// 1. Scan nodes CF for deleted nodes
    /// 2. Filter by deletion timestamp > retention period
    /// 3. Hard delete matching nodes
    pub async fn purge_deleted(&self, retention_days: u32) -> Result<usize, StorageError> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(retention_days as i64);
        let mut purged = 0;

        let db = self.db.read().await;
        let nodes_cf = db.cf_handle(crate::column_families::cf_names::NODES)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound("nodes".into()))?;

        // Collect IDs to delete
        let mut to_delete = Vec::new();
        let iter = db.iterator_cf(nodes_cf, rocksdb::IteratorMode::Start);

        for item in iter {
            match item {
                Ok((_, value)) => {
                    let node: MemoryNode = crate::serialization::deserialize_node(&value)?;
                    if node.metadata.deleted {
                        if let Some(deleted_at) = node.metadata.deleted_at {
                            if deleted_at < cutoff {
                                to_delete.push(node.id);
                            }
                        }
                    }
                }
                Err(e) => return Err(StorageError::ReadFailed(e.to_string())),
            }
        }

        drop(db); // Release read lock

        // Delete collected nodes
        for id in to_delete {
            if self.delete_node(id, false).await? {
                purged += 1;
            }
        }

        Ok(purged)
    }
}
```

---

## 5. Indexing Strategy

### 5.1 Primary Index (Nodes CF)

```yaml
primary_index:
  name: "nodes"
  key_format: "UUID (16 bytes)"
  value_format: "Bincode serialized MemoryNode"
  access_pattern: "Point lookups by ID"

  optimization:
    bloom_filter: true
    bloom_bits_per_key: 10
    block_cache_size: "256MB"
    compression: "LZ4"
```

### 5.2 Embedding Index (Embeddings CF)

```yaml
embedding_index:
  name: "embeddings"
  key_format: "UUID (16 bytes)"
  value_format: "Raw f32 bytes (1536 * 4 = 6144 bytes)"
  access_pattern: "Bulk sequential reads for ANN search"

  optimization:
    compaction: "Universal"
    write_buffer_size: "128MB"
    compression: "LZ4"

  ann_integration:
    type: "HNSW"
    m: 16
    ef_construction: 200
    ef_search: 100
```

### 5.3 Johari Quadrant Indexes

```yaml
johari_indexes:
  - name: "johari_open"
    key_format: "UUID (16 bytes)"
    value_format: "Empty (key-only)"
    purpose: "Filter by Open quadrant"

  - name: "johari_hidden"
    key_format: "UUID (16 bytes)"
    value_format: "Empty"
    purpose: "Filter by Hidden quadrant"

  - name: "johari_blind"
    key_format: "UUID (16 bytes)"
    value_format: "Empty"
    purpose: "Filter by Blind quadrant"

  - name: "johari_unknown"
    key_format: "UUID (16 bytes)"
    value_format: "Empty"
    purpose: "Filter by Unknown quadrant"

  query_pattern: |
    For quadrant filtering:
    1. Get CF handle for target quadrant
    2. Iterate keys (all matching UUIDs)
    3. Batch retrieve nodes from nodes CF
```

### 5.4 Temporal Index

```yaml
temporal_index:
  name: "temporal"
  key_format: "timestamp_ms:UUID (8 + 16 = 24 bytes)"
  value_format: "UUID (16 bytes)"

  query_patterns:
    range_query: |
      Prefix scan with timestamp range:
      - Start key: min_timestamp + ":" + zero_uuid
      - End key: max_timestamp + ":" + max_uuid

    recent_nodes: |
      Reverse iteration from end:
      db.iterator_cf(temporal_cf, IteratorMode::End)
        .take(limit)
```

### 5.5 Tag Index

```yaml
tag_index:
  name: "tags"
  key_format: "tag:UUID"
  value_format: "UUID (16 bytes)"

  query_pattern: |
    For tag filtering:
    1. Prefix scan with tag prefix
    2. Extract UUIDs from matching keys
    3. Batch retrieve nodes

  example:
    key: "project:550e8400-e29b-41d4-a716-446655440000"
    value: [16 bytes UUID]
```

---

## 6. Error Handling

### 6.1 Storage Error Hierarchy

```rust
// crates/context-graph-storage/src/error.rs

use thiserror::Error;

/// Top-level storage errors
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Failed to open database: {0}")]
    OpenFailed(String),

    #[error("Column family not found: {0}")]
    ColumnFamilyNotFound(String),

    #[error("Write failed: {0}")]
    WriteFailed(String),

    #[error("Read failed: {0}")]
    ReadFailed(String),

    #[error("Serialization failed: {0}")]
    SerializationFailed(String),

    #[error("Deserialization failed: {0}")]
    DeserializationFailed(String),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Flush failed: {0}")]
    FlushFailed(String),

    #[error("Concurrent modification: {0}")]
    ConcurrentModification(String),

    #[error("Capacity exceeded: {0}")]
    CapacityExceeded(String),

    #[error("Index corruption detected: {0}")]
    IndexCorruption(String),
}

impl StorageError {
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            StorageError::WriteFailed(_) | StorageError::ConcurrentModification(_)
        )
    }

    /// Get error code for metrics/logging
    pub fn error_code(&self) -> &'static str {
        match self {
            StorageError::OpenFailed(_) => "STORAGE_OPEN_FAILED",
            StorageError::ColumnFamilyNotFound(_) => "STORAGE_CF_NOT_FOUND",
            StorageError::WriteFailed(_) => "STORAGE_WRITE_FAILED",
            StorageError::ReadFailed(_) => "STORAGE_READ_FAILED",
            StorageError::SerializationFailed(_) => "STORAGE_SERIALIZE_FAILED",
            StorageError::DeserializationFailed(_) => "STORAGE_DESERIALIZE_FAILED",
            StorageError::ValidationFailed(_) => "STORAGE_VALIDATION_FAILED",
            StorageError::FlushFailed(_) => "STORAGE_FLUSH_FAILED",
            StorageError::ConcurrentModification(_) => "STORAGE_CONCURRENT_MOD",
            StorageError::CapacityExceeded(_) => "STORAGE_CAPACITY_EXCEEDED",
            StorageError::IndexCorruption(_) => "STORAGE_INDEX_CORRUPT",
        }
    }
}

pub type StorageResult<T> = Result<T, StorageError>;
```

### 6.2 Error Recovery Strategies

```yaml
error_recovery:
  write_failed:
    strategy: "retry_with_backoff"
    max_retries: 3
    initial_backoff_ms: 10
    backoff_multiplier: 2

  concurrent_modification:
    strategy: "refresh_and_retry"
    max_retries: 2
    action: |
      1. Refresh node from storage
      2. Reapply changes
      3. Increment version
      4. Retry write

  index_corruption:
    strategy: "rebuild_index"
    action: |
      1. Log corruption details
      2. Trigger async index rebuild
      3. Return partial results with warning

  capacity_exceeded:
    strategy: "trigger_compaction"
    action: |
      1. Trigger manual compaction
      2. Run purge_deleted
      3. Retry operation
```

---

## 7. Performance Targets and Benchmarks

### 7.1 Latency Targets

| Operation | Target | P50 | P99 |
|-----------|--------|-----|-----|
| Node Insert | <1ms | 500us | 900us |
| Node Retrieval | <500us | 200us | 450us |
| Batch Insert (100) | <10ms | 5ms | 9ms |
| Quadrant Query (100) | <5ms | 2ms | 4ms |
| Tag Query (100) | <5ms | 2ms | 4ms |

### 7.2 Throughput Targets

| Operation | Target | Sustained |
|-----------|--------|-----------|
| Single Inserts | 5,000/sec | 3,000/sec |
| Single Reads | 20,000/sec | 15,000/sec |
| Batch Inserts | 50,000/sec | 30,000/sec |

### 7.3 Storage Efficiency

```yaml
storage_efficiency:
  node_size:
    average: "6.5KB"
    max: "1MB"
    breakdown:
      - embedding: "6144 bytes (1536 * 4)"
      - content: "~300 bytes average"
      - metadata: "~100 bytes"

  compression_ratio:
    nodes: "2.5:1"
    embeddings: "1.3:1"
    indexes: "3:1"

  memory_usage:
    block_cache: "256MB"
    write_buffer: "64MB per CF"
    total_estimated: "512MB"
```

---

## 8. Configuration

### 8.1 Storage Configuration

```toml
# config/default.toml

[storage]
# Storage backend: "memory" or "rocksdb"
backend = "rocksdb"

# Path for persistent storage
path = "./data/memex"

# Enable compression
compression = true

# Write-ahead log directory
wal_dir = "./data/memex/wal"

# Maximum open files
max_open_files = 1000

# Block cache size in MB
block_cache_mb = 256

# Write buffer size in MB
write_buffer_mb = 64

# Maximum write buffer count
max_write_buffers = 3

# Enable bloom filters
bloom_filter = true

# Bloom filter bits per key
bloom_bits_per_key = 10

[storage.compaction]
# Compaction style: "level" or "universal"
style = "level"

# Level 0 compaction trigger
level0_trigger = 4

# Target file size in MB
target_file_size_mb = 64

[storage.retention]
# Soft delete retention in days
soft_delete_retention_days = 30

# Auto-purge interval in hours
purge_interval_hours = 24
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_node_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let memex = RocksDbMemex::open(temp_dir.path()).unwrap();

        let node = MemoryNode::new("test".into(), vec![0.1; 1536]);
        let id = memex.insert(node.clone()).await.unwrap();

        let retrieved = memex.retrieve(id).await.unwrap().unwrap();
        assert_eq!(node.content, retrieved.content);
    }

    #[tokio::test]
    async fn test_johari_index() {
        let temp_dir = TempDir::new().unwrap();
        let memex = RocksDbMemex::open(temp_dir.path()).unwrap();

        let mut node = MemoryNode::new("open".into(), vec![0.1; 1536]);
        node.quadrant = JohariQuadrant::Open;
        memex.insert(node).await.unwrap();

        let ids = memex.query_by_quadrant(JohariQuadrant::Open, 10).await.unwrap();
        assert_eq!(ids.len(), 1);
    }

    #[tokio::test]
    async fn test_soft_delete() {
        let temp_dir = TempDir::new().unwrap();
        let memex = RocksDbMemex::open(temp_dir.path()).unwrap();

        let node = MemoryNode::new("test".into(), vec![0.1; 1536]);
        let id = memex.insert(node).await.unwrap();

        assert!(memex.delete(id, true).await.unwrap());

        let retrieved = memex.retrieve(id).await.unwrap().unwrap();
        assert!(retrieved.metadata.deleted);
    }

    #[tokio::test]
    async fn test_batch_insert_performance() {
        let temp_dir = TempDir::new().unwrap();
        let memex = RocksDbMemex::open(temp_dir.path()).unwrap();

        let nodes: Vec<_> = (0..100)
            .map(|i| MemoryNode::new(format!("node {}", i), vec![0.1; 1536]))
            .collect();

        let start = std::time::Instant::now();
        memex.batch_insert(nodes).await.unwrap();
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() < 10, "Batch insert took {}ms", elapsed.as_millis());
    }
}
```

### 9.2 Integration Tests

```rust
// tests/integration/storage_test.rs

use context_graph_storage::RocksDbMemex;
use context_graph_core::{MemoryNode, JohariQuadrant};
use tempfile::TempDir;

#[tokio::test]
async fn test_full_lifecycle() {
    let temp_dir = TempDir::new().unwrap();
    let memex = RocksDbMemex::open(temp_dir.path()).unwrap();

    // Insert
    let mut node = MemoryNode::new("lifecycle test".into(), vec![0.1; 1536]);
    node.quadrant = JohariQuadrant::Open;
    node.metadata.tags = vec!["test".into(), "lifecycle".into()];
    let id = memex.insert(node.clone()).await.unwrap();

    // Retrieve
    let retrieved = memex.retrieve(id).await.unwrap().unwrap();
    assert_eq!(retrieved.content, "lifecycle test");
    assert_eq!(retrieved.access_count, 1);

    // Update
    let mut updated = retrieved.clone();
    updated.quadrant = JohariQuadrant::Hidden;
    memex.update(updated).await.unwrap();

    // Verify update
    let verified = memex.retrieve(id).await.unwrap().unwrap();
    assert_eq!(verified.quadrant, JohariQuadrant::Hidden);

    // Soft delete
    memex.delete(id, true).await.unwrap();
    let deleted = memex.retrieve(id).await.unwrap().unwrap();
    assert!(deleted.metadata.deleted);

    // Hard delete
    memex.delete(id, false).await.unwrap();
    assert!(memex.retrieve(id).await.unwrap().is_none());
}
```

---

## 10. Traceability Matrix

| Technical Req | Functional Req | Implementation Location |
|---------------|----------------|-------------------------|
| MemoryNode struct | REQ-CORE-001 | context-graph-core/src/memory_node.rs |
| Johari quadrants | REQ-CORE-002 | context-graph-core/src/johari.rs |
| NodeMetadata | REQ-CORE-003 | context-graph-core/src/metadata.rs |
| CognitivePulse | REQ-CORE-004 | context-graph-core/src/pulse.rs |
| RocksDB backend | REQ-CORE-005 | context-graph-storage/src/rocksdb_backend.rs |
| Column families | REQ-CORE-006 | context-graph-storage/src/column_families.rs |
| Bincode serialization | REQ-CORE-007 | context-graph-storage/src/serialization.rs |
| Insert <1ms | REQ-CORE-008 | CRUD operations |
| Retrieve <500us | REQ-CORE-009 | CRUD operations |
| Batch insert <10ms | REQ-CORE-010 | Batch operations |
| Johari indexes | REQ-CORE-011 | Column families |
| Temporal index | REQ-CORE-012 | Temporal CF |
| Tag index | REQ-CORE-013 | Tags CF |
| Error handling | REQ-CORE-014 | error.rs |
| Compression LZ4 | REQ-CORE-015 | Column family options |
| Block cache 256MB | REQ-CORE-016 | RocksDB options |

---

*Document generated: 2025-12-31*
*Technical Specification Version: 1.0.0*
*Module: Core Infrastructure (Phase 1)*
