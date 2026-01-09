# TASK-CORE-002: Define Embedder Enumeration

```xml
<task_spec id="TASK-CORE-002" version="1.0">
<metadata>
  <title>Define Embedder Enumeration with Dimension Metadata</title>
  <status>todo</status>
  <layer>foundation</layer>
  <sequence>2</sequence>
  <implements>
    <requirement_ref>REQ-TELEOLOGICAL-01</requirement_ref>
    <requirement_ref>REQ-EMBEDDER-TYPES-01</requirement_ref>
  </implements>
  <depends_on><!-- None - can start immediately --></depends_on>
  <estimated_complexity>low</estimated_complexity>
  <estimated_days>1</estimated_days>
</metadata>

<context>
Foundation type that enumerates all 13 embedding models used in teleological arrays.
This enum is referenced by virtually every other component in the system. Must be
defined first before TeleologicalArray (TASK-CORE-003) which uses it.
</context>

<objective>
Create the Embedder enum with all 13 variants, dimension metadata, index methods,
and grouping utilities for the teleological array system.
</objective>

<rationale>
The 13-embedder architecture captures different semantic dimensions of memory:
- E1 (Semantic): General meaning via Matryoshka embeddings (1024D)
- E2 (TemporalRecent): Recency via exponential decay (512D)
- E3 (TemporalPeriodic): Cyclical patterns via Fourier (512D)
- E4 (EntityRelationship): Named entity positional encoding (512D)
- E5 (Causal): Cause-effect reasoning, asymmetric (768D)
- E6 (Splade): SPLADE sparse lexical expansion (~30K sparse)
- E7 (Contextual): Discourse context understanding (1536D)
- E8 (Emotional): Affective valence, sentiment (384D)
- E9 (Syntactic): Structural patterns via XOR/Hamming (1024D binary)
- E10 (Pragmatic): Intent and function understanding (768D)
- E11 (CrossModal): Multi-modal bridges text/code/diagrams (384D)
- E12 (LateInteraction): Token-level ColBERT MaxSim (128D/token)
- E13 (KeywordSplade): Term matching, complementary to E6 (~30K sparse)

A type-safe enum prevents index errors and enables compile-time verification.
</rationale>

<input_context_files>
  <file purpose="architecture_reference">docs2/refactor/01-ARCHITECTURE.md</file>
  <file purpose="embedder_specs">docs2/refactor/08-MCP-TOOLS.md#embedder-reference</file>
  <file purpose="existing_types">crates/context-graph-core/src/lib.rs</file>
</input_context_files>

<prerequisites>
  <check>context-graph-core crate exists and compiles</check>
  <check>No existing Embedder enum to conflict with</check>
</prerequisites>

<scope>
  <in_scope>
    <item>Create Embedder enum with 13 variants</item>
    <item>Create EmbedderDims struct for dimension info</item>
    <item>Create EmbedderMask bitmask type</item>
    <item>Create EmbedderGroup enum for predefined groupings</item>
    <item>Implement iteration support (Embedder::all())</item>
    <item>Implement index() method returning 0-12</item>
    <item>Implement expected_dims() method</item>
  </in_scope>
  <out_of_scope>
    <item>TeleologicalArray struct (TASK-CORE-003)</item>
    <item>Comparison types (TASK-CORE-004)</item>
    <item>Actual embedding generation logic</item>
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/teleology/embedder.rs">
      /// Represents the 13 embedding models in the teleological array.
      #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
      #[repr(u8)]
      pub enum Embedder {
          /// E1: Semantic understanding via Matryoshka (1024D, truncatable to 512/256/128)
          Semantic = 0,
          /// E2: Recent temporal context via exponential decay (512D)
          TemporalRecent = 1,
          /// E3: Periodic/cyclical patterns via Fourier (512D)
          TemporalPeriodic = 2,
          /// E4: Named entity positional encoding (512D)
          EntityRelationship = 3,
          /// E5: Causal reasoning, asymmetric (768D)
          Causal = 4,
          /// E6: SPLADE sparse expansion (~30K sparse)
          Splade = 5,
          /// E7: Contextual/discourse understanding (1536D)
          Contextual = 6,
          /// E8: Emotional/sentiment (384D)
          Emotional = 7,
          /// E9: Syntactic patterns via XOR/Hamming (1024D binary)
          Syntactic = 8,
          /// E10: Pragmatic intent/function (768D)
          Pragmatic = 9,
          /// E11: Cross-modal bridges text/code/diagrams (384D)
          CrossModal = 10,
          /// E12: Late interaction/ColBERT MaxSim (128D per token)
          LateInteraction = 11,
          /// E13: Keyword SPLADE term matching (~30K sparse)
          KeywordSplade = 12,
      }

      impl Embedder {
          pub const COUNT: usize = 13;
          pub fn index(self) -> usize;
          pub fn from_index(idx: usize) -> Option<Self>;
          pub fn expected_dims(self) -> EmbedderDims;
          pub fn all() -> impl Iterator<Item = Embedder>;
          pub fn name(self) -> &'static str;
          pub fn short_name(self) -> &'static str;
      }

      #[derive(Debug, Clone, Copy)]
      pub enum EmbedderDims {
          Dense(usize),
          Sparse { max_active: usize },
          TokenLevel { per_token: usize },
          Binary { bits: usize },
      }

      #[derive(Debug, Clone, Copy, Default)]
      pub struct EmbedderMask(u16);

      impl EmbedderMask {
          pub fn new() -> Self;
          pub fn all() -> Self;
          pub fn set(&mut self, embedder: Embedder);
          pub fn unset(&mut self, embedder: Embedder);
          pub fn contains(self, embedder: Embedder) -> bool;
          pub fn iter(self) -> impl Iterator<Item = Embedder>;
          pub fn count(self) -> usize;
      }

      #[derive(Debug, Clone, Copy, PartialEq, Eq)]
      pub enum EmbedderGroup {
          Temporal,    // E2, E3, E4 (time-related)
          Relational,  // E4, E5, E11 (entity/causal/cross-modal)
          Lexical,     // E6, E12, E13 (sparse/token-level)
          Dense,       // E1, E2, E3, E4, E5, E7, E8, E10, E11 (standard dense vectors)
          Binary,      // E9 (Syntactic - binary/Hamming)
          All,
      }

      impl EmbedderGroup {
          pub fn embedders(self) -> EmbedderMask;
      }
    </signature>
  </signatures>

  <constraints>
    <constraint>Enum must have exactly 13 variants</constraint>
    <constraint>Index values 0-12 must be stable (repr(u8))</constraint>
    <constraint>No 'any' or untyped dimensions</constraint>
    <constraint>All dimension values match 08-MCP-TOOLS.md specification</constraint>
    <constraint>Implements Serialize, Deserialize via serde</constraint>
  </constraints>

  <verification>
    <command>cargo check -p context-graph-core</command>
    <command>cargo test -p context-graph-core embedder</command>
    <command>cargo doc -p context-graph-core --no-deps</command>
  </verification>
</definition_of_done>

<pseudo_code>
// crates/context-graph-core/src/teleology/embedder.rs

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Embedder {
    Semantic = 0,          // E1: 1024D dense
    TemporalRecent = 1,    // E2: 512D dense
    TemporalPeriodic = 2,  // E3: 512D dense
    EntityRelationship = 3, // E4: 512D dense
    Causal = 4,            // E5: 768D dense
    Splade = 5,            // E6: ~30K sparse
    Contextual = 6,        // E7: 1536D dense
    Emotional = 7,         // E8: 384D dense
    Syntactic = 8,         // E9: 1024D binary
    Pragmatic = 9,         // E10: 768D dense
    CrossModal = 10,       // E11: 384D dense
    LateInteraction = 11,  // E12: 128D per token
    KeywordSplade = 12,    // E13: ~30K sparse
}

impl Embedder {
    pub const COUNT: usize = 13;

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::Semantic),
            // ... all 13
            _ => None,
        }
    }

    pub fn expected_dims(self) -> EmbedderDims {
        match self {
            Self::Semantic => EmbedderDims::Dense(1024),       // E1
            Self::TemporalRecent => EmbedderDims::Dense(512),  // E2
            Self::TemporalPeriodic => EmbedderDims::Dense(512), // E3
            Self::EntityRelationship => EmbedderDims::Dense(512), // E4
            Self::Causal => EmbedderDims::Dense(768),          // E5
            Self::Splade => EmbedderDims::Sparse { max_active: 30000 }, // E6
            Self::Contextual => EmbedderDims::Dense(1536),     // E7
            Self::Emotional => EmbedderDims::Dense(384),       // E8
            Self::Syntactic => EmbedderDims::Binary { bits: 1024 }, // E9
            Self::Pragmatic => EmbedderDims::Dense(768),       // E10
            Self::CrossModal => EmbedderDims::Dense(384),      // E11
            Self::LateInteraction => EmbedderDims::TokenLevel { per_token: 128 }, // E12
            Self::KeywordSplade => EmbedderDims::Sparse { max_active: 30000 }, // E13
        }
    }

    pub fn all() -> impl Iterator<Item = Embedder> {
        (0..Self::COUNT).filter_map(Self::from_index)
    }
}

// EmbedderDims, EmbedderMask, EmbedderGroup implementations...

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_count() {
        assert_eq!(Embedder::all().count(), 13);
    }

    #[test]
    fn test_index_roundtrip() {
        for e in Embedder::all() {
            assert_eq!(Embedder::from_index(e.index()), Some(e));
        }
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/teleology/embedder.rs">
    Embedder enum with all 13 variants and associated types
  </file>
  <file path="crates/context-graph-core/src/teleology/mod.rs">
    Module definition (if not exists)
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/lib.rs">
    Add pub mod teleology;
  </file>
</files_to_modify>

<validation_criteria>
  <criterion>Embedder::COUNT equals 13</criterion>
  <criterion>All 13 variants have correct dimensions per spec</criterion>
  <criterion>Index roundtrip test passes</criterion>
  <criterion>Serde serialization works</criterion>
  <criterion>EmbedderGroup::Temporal includes E2, E3</criterion>
  <criterion>EmbedderMask can represent any subset of embedders</criterion>
</validation_criteria>

<test_commands>
  <command>cargo test -p context-graph-core embedder -- --nocapture</command>
  <command>cargo check -p context-graph-core</command>
</test_commands>
</task_spec>
```
