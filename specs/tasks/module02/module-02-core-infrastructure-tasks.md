# Module 2: Core Infrastructure - Atomic Tasks

```yaml
metadata:
  module_id: "module-02"
  module_name: "Core Infrastructure"
  version: "1.0.0"
  phase: 1
  total_tasks: 28
  approach: "inside-out-bottom-up"
  created: "2025-12-31"
  dependencies:
    - module-01-ghost-system
  estimated_duration: "3 weeks"
```

---

## Task Overview

This module implements the foundational data structures and storage layer for the Context Graph system. Tasks are organized in inside-out, bottom-up order:

1. **Foundation Layer** (Tasks 1-8): Core data structures - MemoryNode, GraphEdge, Marblestone
2. **Logic Layer** (Tasks 9-18): CRUD operations, validation, serialization, RocksDB backend
3. **Surface Layer** (Tasks 19-28): MCP integration, Cognitive Pulse, indexes

---

## Foundation Layer: Data Structures

```yaml
tasks:
  # ============================================================
  # FOUNDATION: Core Struct Definitions
  # ============================================================

  - id: "M02-T01"
    title: "Define JohariQuadrant Enum"
    description: |
      Implement the JohariQuadrant enum with all four quadrants (Open, Hidden, Blind, Unknown).
      Include methods: is_self_aware(), is_other_aware(), default_retrieval_weight(),
      include_in_default_context(), description(), column_family(), from_str(), all().
      Implement Default, Display traits.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-core/src/johari.rs"
    dependencies: []
    acceptance_criteria:
      - "JohariQuadrant enum compiles with Open, Hidden, Blind, Unknown variants"
      - "is_self_aware() returns true for Open/Hidden"
      - "is_other_aware() returns true for Open/Blind"
      - "default_retrieval_weight() returns correct weights (Open=1.0, Hidden=0.3, Blind=0.7, Unknown=0.5)"
      - "column_family() returns correct CF names for RocksDB"
      - "Unit tests pass for all methods"
    test_file: "crates/context-graph-core/tests/johari_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.2"
      - "REQ-CORE-001"

  - id: "M02-T02"
    title: "Define Modality Enum"
    description: |
      Implement the Modality enum for content type classification (Text, Code, Image, Audio, Structured, Mixed).
      Include detect() method for automatic content-based modality detection.
      Include file_extensions() method for each modality.
    layer: "foundation"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-core/src/metadata.rs"
    dependencies: []
    acceptance_criteria:
      - "Modality enum with 6 variants compiles"
      - "detect() correctly identifies code patterns (fn, def, class, import)"
      - "detect() correctly identifies structured data (JSON, YAML)"
      - "file_extensions() returns appropriate extensions per modality"
      - "Default trait implemented (returns Text)"
    test_file: "crates/context-graph-core/tests/modality_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.3"

  - id: "M02-T03"
    title: "Define NodeMetadata Struct"
    description: |
      Implement NodeMetadata struct with fields: source, language, modality, tags, utl_score,
      consolidated, consolidated_at, version, deleted, deleted_at, parent_id, child_ids, custom.
      Include methods: with_source(), add_tag(), remove_tag(), has_tag(), set_custom(), get_custom(),
      remove_custom(), mark_consolidated(), mark_deleted(), restore(), increment_version(), estimated_size().
    layer: "foundation"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-core/src/metadata.rs"
    dependencies:
      - "M02-T02"
    acceptance_criteria:
      - "NodeMetadata struct compiles with all 13 fields"
      - "Tag operations (add/remove/has) work correctly with deduplication"
      - "Custom attribute CRUD operations work"
      - "mark_consolidated() sets timestamp correctly"
      - "Soft delete/restore cycle works"
      - "Version incrementing saturates at u32::MAX"
      - "estimated_size() returns reasonable approximation"
    test_file: "crates/context-graph-core/tests/metadata_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.3"
      - "REQ-CORE-003"

  - id: "M02-T04"
    title: "Define ValidationError Enum"
    description: |
      Implement ValidationError enum with variants: InvalidEmbeddingDimension, OutOfBounds,
      ContentTooLarge, EmbeddingNotNormalized. Use thiserror for error derivation.
      Include clear error messages with field names and bounds.
    layer: "foundation"
    priority: "high"
    estimated_hours: 1
    file_path: "crates/context-graph-core/src/memory_node.rs"
    dependencies: []
    acceptance_criteria:
      - "ValidationError enum compiles with 4 variants"
      - "Error messages are human-readable and include context"
      - "thiserror derive works correctly"
      - "Display trait shows helpful information"
    test_file: "crates/context-graph-core/tests/validation_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.1"

  - id: "M02-T05"
    title: "Define MemoryNode Struct"
    description: |
      Implement MemoryNode struct with fields: id (UUID), content (String), embedding (Vec<f32>),
      quadrant (JohariQuadrant), importance (f32), emotional_valence (f32), created_at, accessed_at,
      access_count, metadata (NodeMetadata).
      Include doc comments with performance characteristics (6.5KB avg, <1ms insert, <500us retrieval).
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-core/src/memory_node.rs"
    dependencies:
      - "M02-T01"
      - "M02-T03"
      - "M02-T04"
    acceptance_criteria:
      - "MemoryNode struct compiles with all 10 fields"
      - "UUID type alias NodeId defined"
      - "EmbeddingVector type alias defined as Vec<f32>"
      - "Default embedding dimension is 1536"
      - "Serde Serialize/Deserialize traits work"
      - "Clone, Debug, PartialEq traits implemented"
    test_file: "crates/context-graph-core/tests/memory_node_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.1"
      - "REQ-CORE-002"

  - id: "M02-T06"
    title: "Implement MemoryNode Methods"
    description: |
      Implement MemoryNode methods: new(), with_id(), record_access(), age_seconds(),
      time_since_access_seconds(), compute_decay() (Ebbinghaus curve), should_consolidate(),
      validate(). Implement Default trait.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-core/src/memory_node.rs"
    dependencies:
      - "M02-T05"
    acceptance_criteria:
      - "new() creates node with UUID v4 and current timestamps"
      - "record_access() updates accessed_at and increments count safely"
      - "compute_decay() implements modified Ebbinghaus formula correctly"
      - "should_consolidate() threshold is 0.7 based on weighted score"
      - "validate() checks: embedding dim 1536, importance [0,1], valence [-1,1], content <=1MB, normalized embedding"
      - "All validation errors return correct ValidationError variants"
    test_file: "crates/context-graph-core/tests/memory_node_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.1"
      - "REQ-CORE-004"

  - id: "M02-T07"
    title: "Define Domain Enum (Marblestone)"
    description: |
      Implement Domain enum for context-aware neurotransmitter weighting per Marblestone architecture.
      Variants: Code, Legal, Medical, Creative, Research, General.
      Include domain-specific descriptions and Default trait (General).
    layer: "foundation"
    priority: "high"
    estimated_hours: 1
    file_path: "crates/context-graph-core/src/marblestone.rs"
    dependencies: []
    acceptance_criteria:
      - "Domain enum compiles with 6 variants"
      - "Default returns General"
      - "Serde serialization uses lowercase names"
    test_file: "crates/context-graph-core/tests/marblestone_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.5"
      - "Marblestone Integration Spec"

  - id: "M02-T08"
    title: "Define NeurotransmitterWeights Struct (Marblestone)"
    description: |
      Implement NeurotransmitterWeights struct with fields: excitatory, inhibitory, modulatory (all f32 [0,1]).
      Include for_domain() factory method with domain-specific profiles:
      - Code: excitatory=0.6, inhibitory=0.3, modulatory=0.4
      - Legal: excitatory=0.4, inhibitory=0.4, modulatory=0.2
      - Creative: excitatory=0.8, inhibitory=0.1, modulatory=0.6
      Include compute_effective_weight() and validate() methods.
    layer: "foundation"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-core/src/marblestone.rs"
    dependencies:
      - "M02-T07"
    acceptance_criteria:
      - "NeurotransmitterWeights struct compiles with 3 f32 fields"
      - "Default returns excitatory=0.5, inhibitory=0.2, modulatory=0.3"
      - "for_domain() returns correct profiles for all 6 domains"
      - "compute_effective_weight() applies formula: ((base*excit - base*inhib) * (1 + (mod-0.5)*0.4)).clamp(0,1)"
      - "validate() ensures all weights in [0,1]"
    test_file: "crates/context-graph-core/tests/marblestone_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.5"
      - "Marblestone Integration Spec"

  # ============================================================
  # FOUNDATION: GraphEdge with Marblestone
  # ============================================================

  - id: "M02-T09"
    title: "Define EdgeType Enum"
    description: |
      Implement EdgeType enum with variants: Semantic, Temporal, Causal, Hierarchical.
      Include descriptions for each relationship type.
    layer: "foundation"
    priority: "high"
    estimated_hours: 0.5
    file_path: "crates/context-graph-core/src/marblestone.rs"
    dependencies: []
    acceptance_criteria:
      - "EdgeType enum compiles with 4 variants"
      - "Serde serialization uses lowercase names"
      - "Copy trait implemented for efficiency"
    test_file: "crates/context-graph-core/tests/marblestone_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.5"

  - id: "M02-T10"
    title: "Define GraphEdge Struct with Marblestone Fields"
    description: |
      Implement GraphEdge struct with all fields including Marblestone features:
      id, source_id, target_id, edge_type, weight, confidence, domain,
      neurotransmitter_weights, is_amortized_shortcut, steering_reward,
      traversal_count, created_at, last_traversed_at.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-core/src/marblestone.rs"
    dependencies:
      - "M02-T07"
      - "M02-T08"
      - "M02-T09"
    acceptance_criteria:
      - "GraphEdge struct compiles with all 13 fields"
      - "is_amortized_shortcut defaults to false"
      - "steering_reward range is [-1.0, 1.0]"
      - "Serde Serialize/Deserialize work correctly"
      - "Clone and Debug traits implemented"
    test_file: "crates/context-graph-core/tests/graph_edge_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.5"
      - "Marblestone Integration Spec"

  - id: "M02-T11"
    title: "Implement GraphEdge Methods"
    description: |
      Implement GraphEdge methods: new(), get_modulated_weight(), apply_steering_reward(),
      decay_steering(), record_traversal(), is_reliable_shortcut(), mark_as_shortcut().
      Modulated weight formula: (nt_factor * (1 + steering_reward * 0.2)).clamp(0,1)
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2.5
    file_path: "crates/context-graph-core/src/marblestone.rs"
    dependencies:
      - "M02-T10"
    acceptance_criteria:
      - "new() creates edge with domain-appropriate neurotransmitter weights"
      - "get_modulated_weight() applies NT weights and steering reward"
      - "apply_steering_reward() clamps to [-1,1]"
      - "decay_steering() multiplies by decay factor"
      - "record_traversal() increments count and updates timestamp"
      - "is_reliable_shortcut() checks: is_amortized && traversal>=3 && reward>0.3 && confidence>=0.7"
    test_file: "crates/context-graph-core/tests/graph_edge_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.5"
      - "Marblestone Integration Spec"

  # ============================================================
  # LOGIC LAYER: Validation & Serialization
  # ============================================================

  - id: "M02-T12"
    title: "Implement Johari Transition Logic"
    description: |
      Implement JohariTransition struct and TransitionTrigger enum.
      Triggers: ExplicitShare, SelfRecognition, PatternDiscovery, Privatize, ExternalObservation, DreamConsolidation.
      Implement valid_transitions() returning allowed transitions per quadrant.
    layer: "logic"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-core/src/johari.rs"
    dependencies:
      - "M02-T01"
    acceptance_criteria:
      - "JohariTransition struct with from, to, trigger fields"
      - "TransitionTrigger enum with 6 variants"
      - "valid_transitions(Open) returns [Hidden via Privatize]"
      - "valid_transitions(Hidden) returns [Open via ExplicitShare]"
      - "valid_transitions(Blind) returns [Open/Hidden via SelfRecognition]"
      - "valid_transitions(Unknown) returns [Open/Blind/Hidden via various triggers]"
    test_file: "crates/context-graph-core/tests/johari_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.2"

  - id: "M02-T13"
    title: "Create Storage Crate Structure"
    description: |
      Create context-graph-storage crate with proper Cargo.toml dependencies:
      rocksdb, bincode, thiserror, uuid, chrono.
      Create module structure: lib.rs, memex.rs, rocksdb_backend.rs, column_families.rs,
      serialization.rs, indexes.rs.
    layer: "logic"
    priority: "critical"
    estimated_hours: 1.5
    file_path: "crates/context-graph-storage/Cargo.toml"
    dependencies:
      - "M02-T06"
    acceptance_criteria:
      - "Cargo.toml with rocksdb, bincode dependencies"
      - "lib.rs exports public modules"
      - "Crate compiles without errors"
      - "Re-exports from context-graph-core work"
    test_file: "crates/context-graph-storage/tests/integration_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 1.2"

  - id: "M02-T14"
    title: "Implement Bincode Serialization"
    description: |
      Implement serialization module with functions: serialize_node(), deserialize_node(),
      serialize_embedding(), deserialize_embedding(), serialize_edge(), deserialize_edge().
      Use bincode with default config for compact binary format.
      Handle errors with StorageError.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2.5
    file_path: "crates/context-graph-storage/src/serialization.rs"
    dependencies:
      - "M02-T13"
    acceptance_criteria:
      - "serialize_node() produces compact bincode bytes"
      - "Round-trip serialization preserves all MemoryNode fields"
      - "Embedding serialization is efficient (6KB for 1536D)"
      - "GraphEdge serialization preserves Marblestone fields"
      - "Error handling returns appropriate StorageError variants"
    test_file: "crates/context-graph-storage/tests/serialization_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 3"

  - id: "M02-T15"
    title: "Define Column Family Descriptors"
    description: |
      Implement column_families module with 12 column families:
      nodes, edges, embeddings, metadata, johari_open, johari_hidden, johari_blind,
      johari_unknown, temporal, tags, sources, system.
      Include optimized Options per CF (compression, buffer sizes, bloom filters).
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-storage/src/column_families.rs"
    dependencies:
      - "M02-T13"
    acceptance_criteria:
      - "cf_names module defines all 12 CF name constants"
      - "get_column_family_descriptors() returns Vec<ColumnFamilyDescriptor>"
      - "nodes_options() optimized for point lookups (256MB cache)"
      - "edges_options() optimized for range scans with prefix extractor"
      - "embeddings_options() optimized for large sequential reads"
      - "All CFs use LZ4 compression except system CF"
    test_file: "crates/context-graph-storage/tests/column_family_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 3.1"
      - "REQ-CORE-005"

  - id: "M02-T16"
    title: "Implement RocksDB Backend Open/Close"
    description: |
      Implement RocksDbMemex struct with open() method.
      Configure: create_if_missing, max_open_files=1000, WAL directory,
      shared 256MB block cache with bloom filter.
      Implement proper resource cleanup on drop.
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-storage/src/rocksdb_backend.rs"
    dependencies:
      - "M02-T15"
    acceptance_criteria:
      - "open() creates DB with all 12 column families"
      - "Missing CFs are created automatically"
      - "WAL directory is configured"
      - "Block cache is shared across CFs"
      - "Bloom filter enabled for nodes CF"
      - "Database closes cleanly on drop"
    test_file: "crates/context-graph-storage/tests/rocksdb_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 3.2"

  - id: "M02-T17"
    title: "Implement Node CRUD Operations"
    description: |
      Implement RocksDbMemex methods: store_node(), get_node(), update_node(), delete_node().
      Use WriteBatch for atomic multi-CF writes.
      Maintain indexes: Johari CF, temporal index, tag indexes, source index.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-storage/src/rocksdb_backend.rs"
    dependencies:
      - "M02-T14"
      - "M02-T16"
    acceptance_criteria:
      - "store_node() writes to nodes, embeddings, johari, temporal, tags, sources CFs atomically"
      - "get_node() retrieves and deserializes MemoryNode"
      - "update_node() handles index updates when quadrant/tags change"
      - "delete_node() removes from all indexes"
      - "Latency target: <1ms for store, <500us for get"
      - "All operations are async-safe with RwLock"
    test_file: "crates/context-graph-storage/tests/crud_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 3.2"
      - "REQ-CORE-006"

  - id: "M02-T18"
    title: "Implement Edge CRUD Operations"
    description: |
      Implement RocksDbMemex methods: store_edge(), get_edge(), update_edge(), delete_edge(),
      get_edges_from(), get_edges_to().
      Use composite keys for efficient edge lookups: source_id:target_id:edge_type.
    layer: "logic"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-storage/src/rocksdb_backend.rs"
    dependencies:
      - "M02-T11"
      - "M02-T17"
    acceptance_criteria:
      - "store_edge() writes to edges CF with composite key"
      - "get_edges_from() efficiently retrieves all outgoing edges"
      - "get_edges_to() efficiently retrieves all incoming edges"
      - "Marblestone fields (NT weights, steering_reward, is_amortized_shortcut) preserved"
      - "Prefix scan works for edge queries"
    test_file: "crates/context-graph-storage/tests/edge_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 3.2"

  # ============================================================
  # SURFACE LAYER: Cognitive Pulse & MCP Integration
  # ============================================================

  - id: "M02-T19"
    title: "Define EmotionalState Enum"
    description: |
      Implement EmotionalState enum: Neutral, Curious, Focused, Stressed, Fatigued, Engaged, Confused.
      Include weight_modifier() method for UTL calculations.
      Modifiers: Curious=1.2, Focused=1.3, Stressed=0.8, Fatigued=0.6, Engaged=1.15, Confused=0.9
    layer: "surface"
    priority: "high"
    estimated_hours: 1
    file_path: "crates/context-graph-core/src/pulse.rs"
    dependencies: []
    acceptance_criteria:
      - "EmotionalState enum with 7 variants"
      - "Default returns Neutral"
      - "weight_modifier() returns correct values"
      - "Serde serialization uses lowercase"
    test_file: "crates/context-graph-core/tests/pulse_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.4"

  - id: "M02-T20"
    title: "Define SuggestedAction Enum"
    description: |
      Implement SuggestedAction enum: Ready, Continue, Explore, Consolidate, Prune, Stabilize, Review.
      Include description() method for human-readable explanations.
    layer: "surface"
    priority: "high"
    estimated_hours: 1
    file_path: "crates/context-graph-core/src/pulse.rs"
    dependencies: []
    acceptance_criteria:
      - "SuggestedAction enum with 7 variants"
      - "Default returns Continue"
      - "description() returns meaningful strings"
      - "Serde serialization uses snake_case"
    test_file: "crates/context-graph-core/tests/pulse_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.4"

  - id: "M02-T21"
    title: "Define CognitivePulse Struct"
    description: |
      Implement CognitivePulse struct with fields: entropy, coherence, curiosity_score,
      confidence, emotional_state, suggested_action, timestamp.
      All metrics in [0,1] range except timestamp.
    layer: "surface"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-core/src/pulse.rs"
    dependencies:
      - "M02-T19"
      - "M02-T20"
    acceptance_criteria:
      - "CognitivePulse struct compiles with all 7 fields"
      - "Serde Serialize/Deserialize work"
      - "Clone, Debug, PartialEq implemented"
      - "Doc comments explain metric ranges"
    test_file: "crates/context-graph-core/tests/pulse_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.4"
      - "REQ-CORE-007"

  - id: "M02-T22"
    title: "Implement CognitivePulse Methods"
    description: |
      Implement CognitivePulse methods: new(), compute_suggested_action(), update(), blend(), to_json().
      Action computation rules:
      - entropy>0.7 && coherence<0.4 -> Stabilize
      - entropy<0.3 && coherence>0.7 -> Ready
      - curiosity>0.7 -> Explore
      - entropy<0.4 && coherence 0.5-0.8 -> Consolidate
    layer: "surface"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-core/src/pulse.rs"
    dependencies:
      - "M02-T21"
    acceptance_criteria:
      - "new() clamps all metrics to valid ranges"
      - "compute_suggested_action() applies all decision rules"
      - "update() modifies entropy/coherence and recomputes action"
      - "blend() linearly interpolates between two pulses"
      - "to_json() produces valid JSON with all fields"
      - "Default returns entropy=0.5, coherence=0.5, action=Continue"
    test_file: "crates/context-graph-core/tests/pulse_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 2.4"
      - "REQ-CORE-008"

  - id: "M02-T23"
    title: "Implement Secondary Index Operations"
    description: |
      Implement index query methods: get_nodes_by_quadrant(), get_nodes_by_tag(),
      get_nodes_by_source(), get_nodes_in_time_range().
      Use RocksDB prefix iterators for efficient scanning.
    layer: "surface"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-storage/src/indexes.rs"
    dependencies:
      - "M02-T17"
    acceptance_criteria:
      - "get_nodes_by_quadrant() uses johari_* CF prefix scan"
      - "get_nodes_by_tag() uses tags CF prefix scan"
      - "get_nodes_in_time_range() uses temporal CF range scan"
      - "Pagination supported via limit/offset"
      - "Results returned as Vec<NodeId>"
    test_file: "crates/context-graph-storage/tests/index_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 3"

  - id: "M02-T24"
    title: "Implement Embedding Storage Operations"
    description: |
      Implement embedding-specific methods: store_embedding(), get_embedding(), batch_get_embeddings().
      Optimize for 1536-dimensional vectors (~6KB each).
      Support batch retrieval for vector search.
    layer: "surface"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-storage/src/rocksdb_backend.rs"
    dependencies:
      - "M02-T14"
      - "M02-T16"
    acceptance_criteria:
      - "store_embedding() writes raw f32 bytes efficiently"
      - "get_embedding() deserializes to Vec<f32>"
      - "batch_get_embeddings() retrieves multiple in single call"
      - "Memory-efficient: no unnecessary allocations"
    test_file: "crates/context-graph-storage/tests/embedding_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 3"

  - id: "M02-T25"
    title: "Implement StorageError Enum"
    description: |
      Define comprehensive StorageError enum: OpenFailed, ColumnFamilyNotFound, ValidationFailed,
      SerializationFailed, DeserializationFailed, NotFound, WriteFailed, ReadFailed, IndexCorrupted.
      Use thiserror for derivation.
    layer: "surface"
    priority: "high"
    estimated_hours: 1.5
    file_path: "crates/context-graph-storage/src/lib.rs"
    dependencies:
      - "M02-T13"
    acceptance_criteria:
      - "StorageError enum with 9 variants"
      - "All variants include contextual information"
      - "From implementations for rocksdb::Error, bincode::Error"
      - "Error messages are actionable"
    test_file: "crates/context-graph-storage/tests/error_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 3"

  - id: "M02-T26"
    title: "Implement Memex Trait Abstraction"
    description: |
      Define Memex trait as storage abstraction layer with async methods:
      store_node, get_node, update_node, delete_node, store_edge, get_edge,
      query_by_quadrant, query_by_tag, get_embedding, health_check.
      RocksDbMemex implements this trait.
    layer: "surface"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-storage/src/memex.rs"
    dependencies:
      - "M02-T17"
      - "M02-T18"
      - "M02-T23"
    acceptance_criteria:
      - "Memex trait defined with all required methods"
      - "All methods are async"
      - "RocksDbMemex implements Memex"
      - "Trait is object-safe for dyn dispatch"
      - "Documentation explains abstraction purpose"
    test_file: "crates/context-graph-storage/tests/memex_trait_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 3"

  - id: "M02-T27"
    title: "Create Module Integration Tests"
    description: |
      Implement comprehensive integration tests for Module 2:
      - End-to-end node lifecycle (create, read, update, delete)
      - Edge creation with Marblestone features
      - Johari quadrant transitions
      - Cognitive Pulse generation
      - Index consistency after mutations
      - Concurrent access patterns
    layer: "surface"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-storage/tests/integration_tests.rs"
    dependencies:
      - "M02-T17"
      - "M02-T18"
      - "M02-T22"
      - "M02-T23"
    acceptance_criteria:
      - "All CRUD operations tested end-to-end"
      - "Marblestone edge features verified"
      - "Index consistency maintained after updates"
      - "Concurrent read/write test passes"
      - "Performance meets targets (<1ms store, <500us get)"
      - "Test coverage >80% for storage crate"
    test_file: "crates/context-graph-storage/tests/integration_tests.rs"
    spec_refs:
      - "TECH-CORE-002 Section 4"

  - id: "M02-T28"
    title: "Document Public API with Examples"
    description: |
      Add comprehensive doc comments to all public types and methods.
      Include usage examples in doc tests.
      Create examples/ directory with runnable examples:
      - basic_storage.rs: Simple node creation and retrieval
      - marblestone_edges.rs: Edge creation with neurotransmitter weights
      - cognitive_pulse.rs: Pulse generation and interpretation
    layer: "surface"
    priority: "medium"
    estimated_hours: 3
    file_path: "crates/context-graph-core/examples/"
    dependencies:
      - "M02-T27"
    acceptance_criteria:
      - "All public items have /// doc comments"
      - "Doc tests compile and pass"
      - "3 runnable examples in examples/ directory"
      - "README.md in each crate describes purpose"
      - "cargo doc generates clean documentation"
    test_file: "N/A (doc tests)"
    spec_refs:
      - "TECH-CORE-002"
```

---

## Dependency Graph

```
M02-T01 (JohariQuadrant) ─────┐
M02-T02 (Modality) ───────────┼─► M02-T03 (NodeMetadata) ─┐
M02-T04 (ValidationError) ────┼────────────────────────────┼─► M02-T05 (MemoryNode) ─► M02-T06 (MemoryNode Methods)
                              │                            │
M02-T07 (Domain) ─────────────┼─► M02-T08 (NeurotransmitterWeights) ─┐
M02-T09 (EdgeType) ───────────┼──────────────────────────────────────┼─► M02-T10 (GraphEdge) ─► M02-T11 (GraphEdge Methods)
                              │
M02-T01 ──────────────────────┴─► M02-T12 (Johari Transitions)

M02-T06 ─► M02-T13 (Storage Crate) ─► M02-T14 (Serialization) ─┬─► M02-T16 (RocksDB Open) ─► M02-T17 (Node CRUD)
                                                                │                            │
                                     M02-T15 (Column Families) ─┘                            │
                                                                                             │
M02-T11, M02-T17 ─────────────────────────────────────────────────────────────────────────► M02-T18 (Edge CRUD)

M02-T19 (EmotionalState) ─┬─► M02-T21 (CognitivePulse) ─► M02-T22 (Pulse Methods)
M02-T20 (SuggestedAction) ┘

M02-T17 ─► M02-T23 (Index Operations)
M02-T14, M02-T16 ─► M02-T24 (Embedding Operations)
M02-T13 ─► M02-T25 (StorageError)
M02-T17, M02-T18, M02-T23 ─► M02-T26 (Memex Trait)
M02-T17, M02-T18, M02-T22, M02-T23 ─► M02-T27 (Integration Tests)
M02-T27 ─► M02-T28 (Documentation)
```

---

## Implementation Order (Recommended)

### Week 1: Foundation
1. M02-T01: JohariQuadrant enum
2. M02-T02: Modality enum
3. M02-T04: ValidationError enum
4. M02-T03: NodeMetadata struct
5. M02-T05: MemoryNode struct
6. M02-T06: MemoryNode methods
7. M02-T07: Domain enum (Marblestone)
8. M02-T08: NeurotransmitterWeights struct (Marblestone)

### Week 2: Logic Layer
9. M02-T09: EdgeType enum
10. M02-T10: GraphEdge struct (Marblestone)
11. M02-T11: GraphEdge methods
12. M02-T12: Johari transition logic
13. M02-T13: Storage crate structure
14. M02-T14: Bincode serialization
15. M02-T15: Column family definitions
16. M02-T16: RocksDB backend open/close

### Week 3: Surface Layer
17. M02-T17: Node CRUD operations
18. M02-T18: Edge CRUD operations
19. M02-T19: EmotionalState enum
20. M02-T20: SuggestedAction enum
21. M02-T21: CognitivePulse struct
22. M02-T22: CognitivePulse methods
23. M02-T23: Secondary index operations
24. M02-T24: Embedding storage operations
25. M02-T25: StorageError enum
26. M02-T26: Memex trait abstraction
27. M02-T27: Integration tests
28. M02-T28: Documentation and examples

---

## Quality Gates

| Gate | Criteria | Required For |
|------|----------|--------------|
| Foundation Complete | M02-T01 through M02-T12 pass all tests | Week 2 start |
| Storage Functional | M02-T13 through M02-T18 pass all tests | Week 3 start |
| Module Complete | All 28 tasks complete, >80% coverage | Module 3 start |

---

## Marblestone Integration Summary

Tasks with Marblestone features:
- **M02-T07**: Domain enum for context-aware weighting
- **M02-T08**: NeurotransmitterWeights (excitatory/inhibitory/modulatory)
- **M02-T10**: GraphEdge with `is_amortized_shortcut`, `steering_reward`
- **M02-T11**: Modulated weight calculation, steering reward application
- **M02-T18**: Edge storage preserving Marblestone fields

---

*Generated: 2025-12-31*
*Module: 02 - Core Infrastructure*
*Version: 1.0.0*
